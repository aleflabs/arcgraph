//! W13δ M5-01 + W17α M4-08+ — `arcgraph-mcp-stdio` binary.
//!
//! Thin wrapper around [`arcgraph_mcp::serve_stdio`] for the
//! v1.0-alpha first-network-surface. Reads JSON-RPC envelopes from
//! stdin (Content-Length-framed per Anthropic MCP spec 2025-11-25)
//! and writes responses to stdout. Shuts down cleanly on SIGTERM /
//! Ctrl-C via [`arcgraph_mcp::shutdown_on_term`], firing the
//! [`arcgraph_query::cancel::CancellationRegistry::cancel_all`] drain
//! per ADR-038 amendment-03 §TIER-1 GAP C.
//!
//! # Adapter wiring (W17α M4-08+)
//!
//! Every dispatcher slot wires through to [`arcgraph_mcp::storage`]:
//!
//! - `SchemaProvider` → [`StorageSchemaProvider`].
//! - `NodeInspector` → [`StorageNodeInspector`].
//! - `NeighborhoodExplorer` → [`StorageNeighborhoodExplorer`].
//! - `HybridSearcher` → [`StorageHybridSearcher`] (substrate
//!   availability is real; the search body is W17α scope-bound to
//!   `IndexUnavailable` per the module rustdoc).
//! - `IngestProvider` → [`StorageIngestProvider`].
//! - `RawQueryExecutor` → [`StorageRawQueryExecutor`].
//!
//! All adapters share an [`arcgraph_mcp::storage::StorageBackend`]
//! constructed from a fresh per-process
//! [`arcgraph_storage::router::MultiTenantRouter`] +
//! [`arcgraph_storage::transaction::TxnManager`] +
//! [`arcgraph_storage::InternTable`]. The catalog is bootstrapped
//! in-process so a freshly-spawned binary starts with a valid storage
//! substrate. Storage durability is selected by flags (W28 / ADR-183):
//! `--data <dir>` wires the durable substrate (file-backed pages + WAL +
//! recover-on-startup, so committed records survive process restart);
//! with no flag (or explicit `--in-memory`) the binary runs the ephemeral
//! in-memory substrate (data lost on process exit) — the prior default,
//! preserved for the embeddable / piped stdio surface. (The `arcgraph
//! serve` umbrella binary refuses to start without an explicit choice per
//! ADR-183 §Policy; the stdio binary keeps its ephemeral default + adds
//! opt-in `--data` durability.)
//!
//! # Tenant scoping
//!
//! v1.0-alpha pins the session to [`TenantId::DEFAULT`]
//! ([`TenantId::DEFAULT`] is the implicit tenant per
//! `crates/arcgraph-core/src/ids.rs:202`). Multi-tenant routing is a
//! forward-method per M5-12 (rate-limit + per-tenant config).
//!
//! # Hard requirements satisfied
//!
//! - **PD-1 (Apache-2.0/MIT)** — only workspace deps (`arcgraph-mcp`,
//!   `arcgraph-query`, `arcgraph-storage`, `arcgraph-core`, `tokio`,
//!   `tracing`, `anyhow`).
//! - **PD-3 (no `unsafe`)** — none in this file.
//! - **PD-4 (no `unwrap` outside `#[cfg(test)]`)** — every fallible
//!   call surfaces through `anyhow::Result` / explicit error handling.
//! - **code-quality policy recursion limit** — `#![recursion_limit = "256"]`.
//!
//! # Forward-deferred (v1.1+)
//!
//! - WAL replay on startup is SHIPPED for `--data <dir>` durable mode
//!   (W28 / ADR-183). Forward-deferred: multi-tenant *registry* recovery
//!   (non-`DEFAULT` tenants surviving restart — ADR-183 §Forward-pin) +
//!   refuse-to-start consistency for this stdio binary (it keeps its
//!   ephemeral default to preserve embeddable / integration-test
//!   ergonomics; the `arcgraph serve` umbrella owns the GA refuse-to-start
//!   policy).
//! - Multi-tenant session routing (M5-12) replaces the per-process
//!   `TenantId::DEFAULT` binding.
//! - OAuth / mTLS auth on the session-init handshake (M5-03).
//! - Hybrid-search body routing (vector + BM25 + RRF) per
//!   `arcgraph_mcp::storage::adapters` rustdoc.

#![recursion_limit = "256"]

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use arcgraph_cli::bootstrap::{BootstrapMode, DurabilityGuard};
// #765 PART-1 — served HNSW vector-search provider.
use arcgraph_cli::vector_search::VectorSearchTier;
use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider, SubstrateSearchProvider,
};
use arcgraph_mcp::{
    Dispatcher, RateLimiter, ServeStats, SessionScope, serve_stdio, shutdown_on_term,
};
use arcgraph_query::cancel::CancellationRegistry;

/// Parsed CLI args. `--data <dir>` / `--in-memory` select the storage
/// substrate (W28 / ADR-183). Future flags (e.g. `--tenant`) hang off
/// the same struct.
#[derive(Debug, Default)]
struct CliArgs {
    /// `Some(dir)` when `--data <dir>` was passed → durable substrate
    /// rooted at `<dir>` (W28 / ADR-183).
    data_dir: Option<PathBuf>,
    /// `true` when `--in-memory` was passed (explicit ephemeral). The
    /// ephemeral substrate is ALSO the default when neither flag is set —
    /// this binary is the embeddable / piped stdio surface, so it keeps
    /// the prior ephemeral default rather than the `arcgraph serve`
    /// refuse-to-start policy (ADR-183 §Policy is scoped to `serve`).
    in_memory: bool,
    /// `true` when `--rate-limit` was passed (#1186 / MUST-LLM-04).
    ///
    /// Opt-IN to the W14γ M5-12 per-tenant token-bucket rate-limiter
    /// (100 read / 10 write per MINUTE per tenant; `-32007` on exceed).
    /// Defaults to **OFF** so the trusted-local single-agent stdio
    /// workload stays unthrottled (the #833 protection: an agent driving
    /// more than 100 reads/min must NOT have reads silently coerced to
    /// empty by `-32007`). A multi-tenant network deployment that wants
    /// a per-tenant DoS / noisy-neighbor cap sets this flag explicitly.
    rate_limit: bool,
}

impl CliArgs {
    /// Resolve the storage substrate (W28 / ADR-183). Unlike `arcgraph
    /// serve`, the stdio binary defaults to the ephemeral in-memory
    /// substrate when neither flag is set (embeddable-surface ergonomics);
    /// `--data <dir>` opts into durability. `--data` + `--in-memory`
    /// together is an error.
    fn bootstrap_mode(&self) -> Result<BootstrapMode> {
        match (&self.data_dir, self.in_memory) {
            (Some(_), true) => anyhow::bail!(
                "--data and --in-memory are mutually exclusive. Pass exactly one \
                 (or neither, for the ephemeral default)."
            ),
            (Some(dir), false) => Ok(BootstrapMode::Durable {
                data_dir: dir.clone(),
            }),
            (None, _) => Ok(BootstrapMode::InMemory),
        }
    }
}

/// Parse `--data <dir>` / `--in-memory` out of argv. Returns an error on:
///
/// - `--data` with no value
/// - any unrecognized flag
fn parse_cli_args<I, S>(argv: I) -> Result<CliArgs>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut out = CliArgs::default();
    let mut it = argv.into_iter();
    // Skip argv[0] (the binary path).
    let _ = it.next();
    while let Some(arg) = it.next() {
        match arg.as_ref() {
            "--data" => {
                let value = it
                    .next()
                    .map(|v| PathBuf::from(v.as_ref()))
                    .ok_or_else(|| anyhow::anyhow!("--data requires a <dir> value"))?;
                out.data_dir = Some(value);
            }
            "--in-memory" => {
                out.in_memory = true;
            }
            "--rate-limit" => {
                out.rate_limit = true;
            }
            "--help" | "-h" => {
                println!(
                    "arcgraph-mcp-stdio — ArcGraph MCP server over stdio.\n\n\
                     USAGE:\n  arcgraph-mcp-stdio [--data <dir> | --in-memory] \
                     [--rate-limit]\n\n\
                     FLAGS:\n  \
                     --data <dir>           Durable store rooted at <dir> (file-backed \
                     pages + WAL;\n                         \
                     committed records survive restart). W28 / ADR-183.\n  \
                     --in-memory            Ephemeral, NON-DURABLE store (the default \
                     for this binary;\n                         \
                     all data lost on process exit).\n  \
                     --rate-limit           Enforce the per-tenant token-bucket rate cap \
                     (100 read /\n                         \
                     10 write per min per tenant; -32007 on exceed). OFF by \
                     default —\n                         \
                     the trusted-local stdio surface stays unthrottled (#833 \
                     protection).\n                         \
                     Opt in for multi-tenant network use.\n  \
                     -h, --help             Print this help.\n",
                );
                std::process::exit(0);
            }
            other => {
                anyhow::bail!("unrecognized argument: '{other}'. Use --help for usage.");
            }
        }
    }
    Ok(out)
}

/// Bootstrap the per-process storage substrate the production adapters
/// read from.
///
/// Thin re-export of [`arcgraph_cli::bootstrap::bootstrap_storage_backend`]
/// so this binary + the `arcgraph` umbrella share one canonical wire-pattern
/// per ADR-087 D-2 + W26-β-1 GA-BOOTSTRAP-WIRING (issue #439) + W28
/// durable-by-default (ADR-183). See the shared module's rustdoc for the
/// full construction order + ADR provenance.
///
/// The returned [`DurabilityGuard`] owns the WAL writer thread in durable
/// mode; callers MUST hold it for the serve loop's lifetime.
fn bootstrap_storage_backend(mode: &BootstrapMode) -> Result<(StorageBackend, DurabilityGuard)> {
    arcgraph_cli::bootstrap::bootstrap_storage_backend(mode)
}

/// Bind the SIGTERM-aware `arcgraph-mcp-stdio` server.
///
/// Returns the [`ServeStats`] on clean exit. Errors as
/// [`anyhow::Error`] only on a non-recoverable I/O fault on stdout
/// (the parent process is gone — there's no recovery) or a
/// storage-bootstrap failure.
async fn run(args: CliArgs) -> Result<ServeStats> {
    // Initialize tracing — RUST_LOG-controlled, default WARN. We use
    // tracing::subscriber's stdlib-only init avoids pulling
    // tracing-subscriber as a dependency. For v1.0-alpha the binary ships
    // without a structured-log sink; OpenTelemetry wiring is deferred.

    let session_tenant = TenantId::DEFAULT;
    // W28 / ADR-183 — resolve the storage substrate (default ephemeral for
    // the stdio surface; `--data <dir>` opts into durability). `durability`
    // owns the WAL writer thread (durable mode) for the serve loop lifetime.
    let mode = args.bootstrap_mode()?;
    let (backend, durability) = bootstrap_storage_backend(&mode)?;

    // SVC-1 / #849 / ADR-229 — spawn the background interval checkpointer
    // (Tokio work-stealing pool, NOT the hot path). Fires a full-state
    // checkpoint on the `WalCheckpointConfig` wall-clock interval so a
    // long-running durable serve keeps restart-recovery bounded even
    // between graceful shutdowns. `None` (in-memory / disabled) → no task.
    // Held for the serve loop lifetime; aborted on shutdown below.
    let _checkpoint_task = arcgraph_cli::bootstrap::spawn_interval_checkpointer(
        durability.checkpointer(),
        arcgraph_storage::config::WalCheckpointConfig::default(),
    );

    let schema_provider = Arc::new(StorageSchemaProvider::new(backend.clone()));
    let node_inspector = Arc::new(StorageNodeInspector::new(backend.clone()));
    let neighborhood_explorer = Arc::new(StorageNeighborhoodExplorer::new(backend.clone()));
    // #765 PART-1 / #1292 PART-3 — one served vector provider, shared by
    // graph.search + ArcQL RANK BY. Tier selected by `VectorSearchTier::from_env`:
    // HNSW (default) OR the RAM-decoupled SSD DiskANN tier (ADR-195,
    // `ARCGRAPH_VECTOR_TIER=ssd`, RSS ceiling enforced).
    let vector_provider: Arc<dyn SubstrateSearchProvider> =
        VectorSearchTier::from_env(args.data_dir.as_deref()).build_provider(backend.clone());
    let hybrid_searcher = Arc::new(
        StorageHybridSearcher::new(backend.clone())
            .with_search_provider(Arc::clone(&vector_provider)),
    );
    let ingest_provider = Arc::new(StorageIngestProvider::new(backend.clone()));
    let raw_query_executor = Arc::new({
        let exec = StorageRawQueryExecutor::new(backend.clone())
            .with_search_provider(Arc::clone(&vector_provider));
        // #1291 — enable the per-tenant memory budget with the served
        // default (1 GiB; `ARCGRAPH_TENANT_MEMORY_CAP_BYTES` overrides,
        // `0` disables). Same wiring as the umbrella `arcgraph serve`
        // binary — see `bin/arcgraph.rs::build_default_dispatcher`.
        match arcgraph_cli::ops::resolve_per_tenant_memory_cap() {
            Some(cap) => exec.with_per_tenant_memory_cap(cap),
            None => exec,
        }
    });

    // #1186 / #833 / #818 — per-tenant rate-limit is OPT-IN
    // (`--rate-limit`), default-OFF on this stdio surface.
    //
    // The stdio surface is the TRUSTED LOCAL MCP host (the single
    // process that spawned this binary), NOT a multi-tenant network
    // surface. The W14γ M5-12 per-tenant token-bucket rate-limiter (100
    // read / 10 write per MINUTE per tenant, per ADR-004 amendment-02 /
    // design-v2 §9.4) is a MULTI-TENANT NETWORK control; wiring it ON BY
    // DEFAULT onto the local stdio dispatcher silently throttled the
    // PRIMARY agent-native workload: a single agent issuing >100
    // sequential reads/min (a recall sweep, multi-hop exploration, batch
    // inspection — `graph.search` + `graph.raw_query` SHARE one
    // `(tenant, Read)` bucket of capacity 100) had every read past the
    // ~100th rejected with `-32007`, which agent clients (langchain, the
    // #818 recall harness) coerce to an EMPTY result-set => silently
    // WRONG answers from an agent-native read surface. #818's served-
    // vector recall pinned at 0.50 (100 of 200 queries) for exactly this
    // reason. So the local stdio dispatcher runs UNTHROTTLED BY DEFAULT,
    // mirroring the same trusted-local-vs-untrusted-network split #818
    // already applies to the frame cap (512 MiB local stdio vs 16 MiB
    // untrusted network; see `arcgraph_mcp::transport::stdio`
    // STDIO_MAX_MESSAGE_BYTES).
    //
    // #1186 (MUST-LLM-04) — the limiter must be ENFORCEABLE on the
    // served surface (the AC: "a customer can observe a per-tenant rate
    // cap"). `--rate-limit` opts INTO it: a multi-tenant network
    // deployment piping JSON-RPC over a shared stdio bridge (or an
    // operator running an adversarial-burst acceptance test) sets the
    // flag and gets the DoS / noisy-neighbor cap; the trusted-local
    // single-agent default stays unthrottled. The limiter primitive
    // (ADR-004 amendment-02, proptest-pinned) is identical to what the
    // untrusted network HTTP/Bolt surfaces consume.
    //
    // W16ζ M5-11: bind the v1.0-alpha session scope to
    // `SessionScope::Power` so the stdio binary's local user (the
    // process that spawned the binary) retains access to
    // `graph.raw_query`. M5-03 OAuth swaps this for a Bearer-
    // token-derived scope; an OAuth-empty session falls back to
    // `SessionScope::Read` (fail-closed) per ADR-004 amendment-03 §D-2.
    // #1186 — base dispatcher: with the per-tenant limiter when
    // `--rate-limit` is set, otherwise the unthrottled trusted-local
    // shape. `RateLimiter::new()` seeds the design-v2 §9.4 defaults
    // (100 read / 10 write per minute per tenant) lazily on first
    // observation; no per-tenant override config is wired at the
    // stdio surface (the `arcgraph serve` umbrella owns config landing).
    let base = if args.rate_limit {
        tracing::info!(
            target: "arcgraph_cli::mcp_stdio",
            "per-tenant rate-limit ENABLED (--rate-limit): 100 read / 10 write per minute per tenant",
        );
        Dispatcher::with_session_scope_and_rate_limiter(
            session_tenant,
            SessionScope::Power,
            schema_provider,
            node_inspector,
            neighborhood_explorer,
            hybrid_searcher,
            ingest_provider,
            raw_query_executor,
            RateLimiter::new(),
        )
    } else {
        Dispatcher::with_session_scope(
            session_tenant,
            SessionScope::Power,
            schema_provider,
            node_inspector,
            neighborhood_explorer,
            hybrid_searcher,
            ingest_provider,
            raw_query_executor,
        )
    };
    let dispatcher = base;
    let cancel_registry = CancellationRegistry::new();

    tracing::info!(
        target: "arcgraph_cli::mcp_stdio",
        tenant_id = session_tenant.raw(),
        "arcgraph-mcp-stdio binary starting (W17α M4-08+ production storage adapters wired)",
    );

    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    // W17δ #348 R1-MED-1 — SIGTERM-during-query handler (sister-site to
    // `arcgraph-cli::run_serve_stdio`).
    //
    // Mirrors the same shape: `shutdown_on_term()` resolves when SIGTERM
    // / Ctrl-C is observed; before letting `serve_stdio` see the
    // shutdown signal we fire every in-flight query's CancellationToken
    // via [`CancellationRegistry::cancel_all`] — the ADR-038 amendment-
    // 03 §TIER-1 GAP C "graceful drain at shutdown" seam. At v1.0-α the
    // dispatcher's stub providers do not register queries (no real
    // `QueryEngine` runs under the empty-tenant adapters), so
    // `cancel_all()` typically returns 0. The wire is load-bearing for
    // M4-08+ when production storage adapters plumb real `QueryEngine`
    // instances against this registry — at that point this handler
    // converts an operator's SIGTERM into a clean
    // `ExplainError::Cancelled` for every in-flight query.
    //
    // Without this wrap the binary's module-doc claim at lines 6–9
    // (which states `cancel_all()` fires on SIGTERM) would be FALSE for
    // this binary — `cancel_all()` would never run.
    let cancel_registry_for_shutdown = cancel_registry.clone();
    let shutdown = async move {
        shutdown_on_term().await;
        let fired = cancel_registry_for_shutdown.cancel_all();
        tracing::info!(
            target: "arcgraph_cli::mcp_stdio",
            fired_count = fired,
            "SIGTERM observed — fired {fired} in-flight cancellation tokens before shutdown",
        );
    };

    // AHP-1 (ADR-225 §3) — `serve_stdio` now takes `Arc<Dispatcher>` (the
    // `spawn_blocking` bulkhead needs an owned `'static` dispatcher).
    let stats = serve_stdio(
        std::sync::Arc::new(dispatcher),
        &cancel_registry,
        stdin,
        stdout,
        shutdown,
        None,
    )
    .await
    .context("serve_stdio loop returned an error")?;

    tracing::info!(
        target: "arcgraph_cli::mcp_stdio",
        messages_in = stats.messages_in,
        messages_out = stats.messages_out,
        parse_errors = stats.parse_errors,
        in_flight_cancelled = stats.in_flight_cancelled,
        exit_reason = ?stats.exit_reason,
        "arcgraph-mcp-stdio binary exiting cleanly",
    );

    // Keep `backend` + `durability` alive for the lifetime of
    // `serve_stdio` (the adapters hold their own `backend` clones; these
    // bindings pin the bootstrap's local-stack lifetime to the serve
    // loop's exit). `durability` owns the WAL writer thread in durable
    // mode — dropping it here (after the serve loop returns) drains +
    // fsyncs + joins the writer (graceful teardown).
    drop(backend);
    drop(durability);

    Ok(stats)
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // The dispatcher + serve_stdio stack already returns structured
    // errors — main's only job is to wire them to a non-zero exit
    // code. `anyhow::Result` from `run()` propagates here.
    let args = parse_cli_args(std::env::args())?;
    let _stats = run(args).await?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests — bootstrap discipline + adapter binding pin.
//
// The full end-to-end subprocess test lives in
// `crates/arcgraph-mcp/tests/mcp_stdio_integ.rs`. These tests pin
// that the production wiring (bootstrap + adapter construction)
// completes without panic.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bootstrap_storage_backend_succeeds_for_fresh_process() {
        // Pin: the per-process bootstrap completes without panic on a
        // fresh binary start. Future regressions in catalog bootstrap
        // (e.g., a new mandatory dep) would surface here. Uses the
        // ephemeral mode (the stdio binary's default; W28 / ADR-183).
        let (backend, _durability) =
            bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap");
        // Routing the default tenant must work after bootstrap.
        let handle = backend
            .router()
            .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
            .expect("route DEFAULT");
        assert_eq!(handle.tenant(), TenantId::DEFAULT);
    }

    #[test]
    fn production_adapters_construct_from_shared_backend() {
        // Pin: all six adapter types accept the same `StorageBackend`
        // bundle. A future surface-shift that broke the `Clone`
        // bound on `StorageBackend` would surface here.
        let (backend, _durability) =
            bootstrap_storage_backend(&BootstrapMode::InMemory).expect("bootstrap");
        let _ = StorageSchemaProvider::new(backend.clone());
        let _ = StorageNodeInspector::new(backend.clone());
        let _ = StorageNeighborhoodExplorer::new(backend.clone());
        let _ = StorageHybridSearcher::new(backend.clone());
        let _ = StorageIngestProvider::new(backend.clone());
        let _ = StorageRawQueryExecutor::new(backend);
    }

    // ─────────────────────────────────────────────────────────────
    // W28 / ADR-183 — storage-mode flag parsing + resolution.
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn parse_cli_args_default_mode_is_in_memory() {
        // The stdio binary defaults to the ephemeral substrate (no
        // refuse-to-start; that policy is `arcgraph serve`-only).
        let args = parse_cli_args(["arcgraph-mcp-stdio"]).expect("default parses");
        assert_eq!(
            args.bootstrap_mode().expect("mode resolves"),
            BootstrapMode::InMemory
        );
    }

    #[test]
    fn parse_cli_args_data_dir_is_durable() {
        let args = parse_cli_args(["arcgraph-mcp-stdio", "--data", "/tmp/arcgraph-stdio-x"])
            .expect("--data parses");
        assert_eq!(
            args.bootstrap_mode().expect("mode resolves"),
            BootstrapMode::Durable {
                data_dir: PathBuf::from("/tmp/arcgraph-stdio-x"),
            }
        );
    }

    #[test]
    fn parse_cli_args_explicit_in_memory_is_ephemeral() {
        let args =
            parse_cli_args(["arcgraph-mcp-stdio", "--in-memory"]).expect("--in-memory parses");
        assert_eq!(
            args.bootstrap_mode().expect("mode resolves"),
            BootstrapMode::InMemory
        );
    }

    #[test]
    fn parse_cli_args_data_plus_in_memory_is_rejected() {
        let args = parse_cli_args(["arcgraph-mcp-stdio", "--data", "/tmp/x", "--in-memory"])
            .expect("flags parse");
        let err = args
            .bootstrap_mode()
            .expect_err("--data + --in-memory must be rejected");
        assert!(format!("{err}").contains("mutually exclusive"));
    }

    #[test]
    fn parse_cli_args_data_requires_value() {
        let err = parse_cli_args(["arcgraph-mcp-stdio", "--data"]).expect_err("--data needs value");
        assert!(format!("{err}").contains("--data requires"));
    }

    // ─────────────────────────────────────────────────────────────
    // #1186 (MUST-LLM-04) — --rate-limit flag parsing (default-OFF).
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn parse_cli_args_rate_limit_defaults_off() {
        // Default-OFF preserves the #833 trusted-local protection: an
        // agent driving >100 reads/min must NOT be throttled by default.
        let args = parse_cli_args(["arcgraph-mcp-stdio"]).expect("default parses");
        assert!(!args.rate_limit, "rate_limit must default OFF");
    }

    #[test]
    fn parse_cli_args_rate_limit_flag_opts_in() {
        let args =
            parse_cli_args(["arcgraph-mcp-stdio", "--rate-limit"]).expect("--rate-limit parses");
        assert!(args.rate_limit, "--rate-limit must opt in");
    }

    #[test]
    fn parse_cli_args_rejects_unknown_flag() {
        let err =
            parse_cli_args(["arcgraph-mcp-stdio", "--garbage"]).expect_err("unknown flag rejected");
        assert!(err.to_string().contains("unrecognized"));
    }
}
