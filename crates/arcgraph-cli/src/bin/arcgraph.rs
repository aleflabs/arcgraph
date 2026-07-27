//! `arcgraph` umbrella CLI for the `v0.1.0-beta` bare database engine.
//!
//! The seven subcommands are `serve`, `check`, `dump`, `health`, `migrate`,
//! `backup`, and `load`. `serve --data DIR` uses durable page storage and WAL
//! recovery; `serve --in-memory` is explicitly ephemeral. The primary
//! transports are MCP over stdio, MCP over HTTPS, and Bolt 5.0.
//!
//! `check --data DIR` cold-opens and samples a committed store.
//! `dump --data DIR` refuses because a faithful storage-rooted logical export
//! is not implemented; cold backup and restore are available through
//! `backup`.
//!
//! # Why a single binary
//!
//! Operators install one binary and get the lifecycle commands plus three
//! transport modes. The umbrella facade is the single
//! place that wires storage / query / MCP into a deployment-friendly
//! surface — every subcommand uses [`arcgraph::query`] /
//! [`arcgraph::mcp`] / [`arcgraph::core`] re-exports so the
//! crate-boundary surface is the same one third-party embedded
//! callers see.
//!
//! The `--config` flag is accepted but ignored in this beta; use CLI flags.
//! The built-in `health` client accepts only `http://` URLs, while the MCP
//! network transport is HTTPS-only.

#![recursion_limit = "256"]

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use arcgraph::core::TenantId;
use arcgraph::mcp::storage::{
    StorageBackend, StorageBoltHandler, StorageHybridSearcher, StorageIngestProvider,
    StorageNeighborhoodExplorer, StorageNodeInspector, StorageRawQueryExecutor,
    StorageSchemaProvider, SubstrateSearchProvider,
};
use arcgraph::mcp::{
    BoltServerConfig, DispatchBulkhead, Dispatcher, HttpServerConfig, IngestProvider,
    MetricsRegistry, RateLimiter, SessionScope, serve_bolt_listener, serve_http, serve_stdio,
    shutdown_on_term,
};
// #761 slice 1 — `serve --http` TLS cert/key → live HTTPS MCP transport.
// The W13ε hot-reload resolver (file-backed cert provider + SIGHUP
// rotation loop) is consumed here to wire `serve_http` end-to-end.
use arcgraph::mcp::tls::{FileSystemCertProvider, HotReloadResolver, run_sighup_reload_loop};
use arcgraph::query::cancel::CancellationRegistry;
use arcgraph_cli::bootstrap::{
    BootstrapMode, DurabilityGuard, SecretsProviderKind, WalEncryptionConfig,
};
// #765 PART-1 / #1292 PART-3 — the served vector-search tier. `VectorSearchTier`
// selects between the in-memory HNSW provider (default) and the RAM-decoupled
// SSD-resident DiskANN tier (ADR-195, `ARCGRAPH_VECTOR_TIER=ssd`) and builds the
// concrete `SubstrateSearchProvider` behind the single trait seam.
use arcgraph_cli::vector_search::VectorSearchTier;
// W28 Feature #582 (ADR-045) — the storage-resident observability sink
// trait. `MetricsRegistry` impls it; the binary coerces
// `Arc<MetricsRegistry>` → `Arc<dyn MetricsSink>` to thread the
// hot-vertex (CrudStore) + query-plan-choice (StorageRawQueryExecutor)
// producers when `--metrics-http` is set.
// #761 slice 2 — JWKS file loading for Bolt HELLO OAuth bearer auth (ADR-049).
// `jsonwebtoken::jwk::JwkSet` deserializes the operator-staged RFC 7517 JWKS file;
// `DecodingKey::from_jwk` builds per-key verifiers. `Algorithm::from_str` converts
// the JWKS `alg` field string to the enum. All three come from the same `jsonwebtoken`
// workspace pin already consumed by `arcgraph-mcp` for JWT signature verification.
use std::str::FromStr as _;

use arcgraph_cli::migrate::{parse_csv_export, parse_cypher_export};
use arcgraph_cli::ops::{
    AdminHttpServerConfig, MetricsHttpServerConfig, ReadinessGate, TracingConfig, TracingGuard,
    init_tracing, serve_admin_http, serve_metrics_http,
};
use arcgraph_mcp::auth::oauth_pkce::{JsonWebKey, JsonWebKeySet, OAuthConfig};
use arcgraph_storage::metrics::MetricsSink;
use clap::{Args, Parser, Subcommand, ValueEnum};

// ─────────────────────────────────────────────────────────────────────
// CLI surface — clap derive
// ─────────────────────────────────────────────────────────────────────

/// `arcgraph` umbrella CLI.
#[derive(Debug, Parser)]
#[command(
    name = "arcgraph",
    version,
    about = "ArcGraph v0.1.0-beta — graph, vector, full-text, and traversal database",
    long_about = "ArcGraph umbrella binary. Seven subcommands:\n  \
                  serve  — start an MCP server (stdio / HTTP / Bolt).\n  \
                  check  — verify store integrity.\n  \
                  dump   — refuse an unavailable durable logical export.\n  \
                  health — probe a plain-HTTP /healthz endpoint.\n  \
                  migrate — upgrade a data directory or parse a Neo4j export \
                  into an ephemeral store.\n  \
                  backup  — create or restore a cold backup.\n  \
                  load    — bootstrap a native corpus into a new data directory.\n\n\
                  v0.1.0-beta uses the production durable storage engine when \
                  serve is given --data."
)]
pub struct Cli {
    /// The subcommand to run.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommands.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start an MCP server on the chosen transport.
    // Box<ServeArgs> keeps the enum-variant size within clippy::large_enum_variant
    // limits. `ServeArgs` grew to 328+ bytes after #761 slice 2 added the four
    // Bolt-auth flags; boxing is the recommended fix (zero functional change).
    Serve(Box<ServeArgs>),
    /// Verify store integrity.
    Check(CheckArgs),
    /// Refuse an unavailable durable logical export.
    Dump(DumpArgs),
    /// Probe a plain-HTTP `/healthz` endpoint.
    Health(HealthArgs),
    /// Upgrade a data directory or parse a Neo4j export into an ephemeral store.
    Migrate(MigrateArgs),
    /// Create or restore a verified cold backup of a durable data directory.
    Backup(BackupArgs),
    /// Bootstrap a native corpus into a new data directory.
    ///
    /// Populated directories are refused without mutation; rerunning resumes
    /// an interrupted load.
    Load(LoadArgs),
}

/// `arcgraph load` — offline native bootstrap ingest.
#[derive(Debug, Args)]
pub struct LoadArgs {
    /// Newline-delimited native JSON input file.
    #[arg(long, value_name = "PATH")]
    pub input: PathBuf,
    /// Virgin ArcGraph data directory to bootstrap (created if absent).
    #[arg(long, value_name = "DIR")]
    pub data_dir: PathBuf,
    /// Input parser boundary. This build accepts only `native`.
    #[arg(long, value_enum)]
    pub format: LoadFormatArg,
    /// Tenant receiving every input row (non-default, non-system).
    #[arg(long, value_name = "TENANT_ID")]
    pub tenant: u64,
}

/// CLI surface for the loader input boundary.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum LoadFormatArg {
    /// Newline-delimited native JSON records.
    Native,
}

/// `arcgraph migrate` — ArcGraph upgrades and external-source imports.
#[derive(Debug, Args)]
pub struct MigrateArgs {
    /// Which migration operation to run.
    #[command(subcommand)]
    pub source: MigrateSource,
    /// Tenant ID to ingest under. Defaults to [`TenantId::DEFAULT`].
    #[arg(long, value_name = "ID", default_value_t = TenantId::DEFAULT.raw())]
    pub tenant: u64,
}

/// `arcgraph migrate` operation subcommands.
#[derive(Debug, Subcommand)]
pub enum MigrateSource {
    /// Offline data-directory generation upgrade (v4->v5, then v5->v6).
    UpgradeDataDir {
        /// ArcGraph data directory to upgrade. The server must be stopped.
        #[arg(long, value_name = "PATH")]
        data_dir: PathBuf,
    },
    /// Parse an `apoc.export.cypher.all()` script (semicolon-separated
    /// CREATE statements) and ingest into the per-process store.
    FromNeo4jCypher {
        /// Path to the cypher script.
        #[arg(value_name = "PATH")]
        path: PathBuf,
    },
    /// Parse a `neo4j-admin export csv` two-file pair (nodes + rels).
    FromNeo4jCsv {
        /// Path to the nodes CSV.
        #[arg(long, value_name = "PATH")]
        nodes: PathBuf,
        /// Path to the relationships CSV.
        #[arg(long, value_name = "PATH")]
        rels: PathBuf,
    },
}

/// `arcgraph serve` — start an MCP server.
#[derive(Debug, Args)]
pub struct ServeArgs {
    /// Use the stdio MCP transport (default if no transport flag set).
    ///
    /// Conflicts with `--http` and `--bolt`. Reads Content-Length-framed
    /// JSON-RPC envelopes from stdin and writes responses to stdout.
    #[arg(long, conflicts_with_all = ["http", "bolt"])]
    pub stdio_mcp: bool,

    /// Bind address for the HTTP/TLS MCP transport (e.g. `127.0.0.1:8443`).
    ///
    /// `--tls-cert` and `--tls-key` are required; plaintext HTTP is not
    /// offered. Loopback binds are allowed by default. A non-loopback bind
    /// also requires `--allow-remote-http-bind`.
    #[arg(long, value_name = "ADDR", conflicts_with_all = ["stdio_mcp", "bolt"], requires_all = ["tls_cert", "tls_key"])]
    pub http: Option<String>,

    /// TLS server certificate chain (PEM). REQUIRED with [`--http`](Self::http).
    ///
    /// The PEM must contain an end-entity `CERTIFICATE` block; intermediates
    /// may follow. The certificate is reloaded for new connections on
    /// `SIGHUP`.
    #[arg(long, value_name = "PATH", requires = "http", conflicts_with_all = ["stdio_mcp", "bolt"])]
    pub tls_cert: Option<PathBuf>,

    /// TLS server private key (PEM). REQUIRED with [`--http`](Self::http).
    ///
    /// PKCS#8, PKCS#1, and SEC1 PEM are accepted. The key must match the
    /// end-entity certificate or startup fails.
    #[arg(long, value_name = "PATH", requires = "http", conflicts_with_all = ["stdio_mcp", "bolt"])]
    pub tls_key: Option<PathBuf>,

    /// Optional expected hostname for RFC 6125 SAN/CN verification of
    /// `--tls-cert` at load + on every SIGHUP reload.
    ///
    /// Omit to skip this load-time hostname check.
    #[arg(long, value_name = "NAME", requires = "http", conflicts_with_all = ["stdio_mcp", "bolt"])]
    pub tls_hostname: Option<String>,

    /// JWKS JSON file for HTTPS MCP `POST /mcp` bearer-token auth.
    ///
    /// When set, `serve --http` enforces `Authorization: Bearer <JWT>` on every
    /// `POST /mcp`. The JWT is verified against this operator-staged
    /// [RFC 7517 JWK Set] file. The token's `iss`
    /// claim must match [`--http-auth-issuer`](Self::http_auth_issuer), and
    /// its `aud` claim must include at least one
    /// [`--http-auth-audience`](Self::http_auth_audience). Method-level scope
    /// checks are the existing `arcgraph-mcp` HTTP enforcement table.
    ///
    /// Requires `--http` (HTTPS MCP transport must be enabled) and
    /// `--http-auth-issuer`. At least one `--http-auth-audience` must also be
    /// present (validated at startup — not expressible as a clap `requires`).
    ///
    /// Absent (the default) → **dev-mode**: loopback HTTPS requests are accepted
    /// without bearer verification. Non-loopback binds still require the
    /// explicit [`--allow-remote-http-bind`](Self::allow_remote_http_bind)
    /// opt-in and TLS.
    ///
    /// HTTP/MCP client usage: send `Authorization: Bearer <token>` alongside
    /// `content-type: application/json` and `x-arcgraph-tenant`.
    ///
    /// [RFC 7517 JWK Set]: https://datatracker.ietf.org/doc/html/rfc7517#section-5
    #[arg(long, value_name = "PATH", requires_all = ["http", "http_auth_issuer"])]
    pub http_auth_jwks: Option<PathBuf>,

    /// Expected JWT `iss` (issuer) claim for HTTPS MCP OAuth auth.
    ///
    /// #761 slice 3: must match the JWT `iss` claim exactly (RFC 7519 §4.1.1).
    /// Example: `https://auth.example.com/`. Required alongside
    /// [`--http-auth-jwks`](Self::http_auth_jwks).
    #[arg(long, value_name = "ISS", requires = "http_auth_jwks")]
    pub http_auth_issuer: Option<String>,

    /// Accepted JWT `aud` (audience) for HTTPS MCP OAuth auth (repeatable).
    ///
    /// #761 slice 3: the token's `aud` claim must include at least one entry
    /// from this list (RFC 7519 §4.1.3). Specify once per accepted audience:
    ///
    /// ```text
    /// --http-auth-audience arcgraph --http-auth-audience api.example.com
    /// ```
    ///
    /// Requires [`--http-auth-jwks`](Self::http_auth_jwks). At least one value
    /// is required when JWKS is configured (validated at startup, not as a clap
    /// constraint — clap allows empty `Vec` on `requires`).
    #[arg(long, value_name = "AUD", requires = "http_auth_jwks")]
    pub http_auth_audience: Vec<String>,

    /// Permit the HTTPS MCP transport to bind a non-loopback address.
    ///
    /// A loopback bind does not need this flag. TLS remains mandatory.
    #[arg(long, default_value_t = false, requires = "http")]
    pub allow_remote_http_bind: bool,

    /// Bind address for the Bolt 5.0 transport (e.g. `127.0.0.1:7687`).
    ///
    /// The Neo4j-driver-compatible listener executes ArcQL against the same
    /// production storage substrate as MCP. This beta documents plaintext
    /// Bolt on loopback only.
    #[arg(long, value_name = "ADDR", conflicts_with_all = ["stdio_mcp", "http"])]
    pub bolt: Option<String>,

    /// JWKS JSON file for Bolt HELLO bearer-token auth.
    ///
    /// When set, `serve --bolt` enforces RFC 8705 bearer auth on every Bolt
    /// HELLO: the client MUST send `scheme="bearer"` with `credentials=<JWT>`.
    /// The JWT is verified against this operator-staged [RFC 7517 JWK Set]
    /// file. Tokens must carry at least one of `arcgraph.{read,write}` in
    /// their scope claim (per `crates/arcgraph-mcp/src/transport/bolt/auth.rs`).
    /// The first `@tenant_id` suffix in the scope claim identifies the
    /// session's tenant.
    ///
    /// Requires `--bolt` (Bolt transport must be enabled) and
    /// `--bolt-auth-issuer`. At least one `--bolt-auth-audience` must also be
    /// present (validated at startup — not expressible as a clap `requires`).
    ///
    /// Absent (the default) → **dev-mode**: `none` / `basic` / `bearer`
    /// schemes are accepted without signature verification. Loopback-only by
    /// default; see `--allow-remote-bolt-bind`.
    ///
    /// Neo4j-driver usage: connect with `bearer_auth(token)` — the driver emits
    /// a HELLO with `scheme="bearer"` and `credentials=<token>`. See also:
    /// [neo4j-python-driver bearer_auth](https://neo4j.com/docs/api/python-driver/current/api.html#bearer-auth).
    ///
    /// [RFC 7517 JWK Set]: https://datatracker.ietf.org/doc/html/rfc7517#section-5
    #[arg(long, value_name = "PATH", requires_all = ["bolt", "bolt_auth_issuer"])]
    pub bolt_auth_jwks: Option<PathBuf>,

    /// Expected JWT `iss` (issuer) claim for Bolt OAuth auth.
    ///
    /// #761 slice 2: must match the JWT `iss` claim exactly (RFC 7519 §4.1.1).
    /// Example: `https://auth.example.com/`. Required alongside
    /// [`--bolt-auth-jwks`](Self::bolt_auth_jwks).
    #[arg(long, value_name = "ISS", requires = "bolt_auth_jwks")]
    pub bolt_auth_issuer: Option<String>,

    /// Accepted JWT `aud` (audience) for Bolt OAuth auth (repeatable).
    ///
    /// #761 slice 2: the token's `aud` claim must include at least one entry
    /// from this list (RFC 7519 §4.1.3). Specify once per accepted audience:
    ///
    /// ```text
    /// --bolt-auth-audience arcgraph --bolt-auth-audience api.example.com
    /// ```
    ///
    /// Requires [`--bolt-auth-jwks`](Self::bolt_auth_jwks). At least one value
    /// is required when JWKS is configured (validated at startup, not as a clap
    /// constraint — clap allows empty `Vec` on `requires`).
    #[arg(long, value_name = "AUD", requires = "bolt_auth_jwks")]
    pub bolt_auth_audience: Vec<String>,

    /// Opt in to the non-loopback Bolt bind check.
    ///
    /// Non-loopback Bolt also requires server TLS, but this CLI exposes no
    /// Bolt certificate/key flags in `v0.1.0-beta`. Therefore this flag alone
    /// is insufficient and a non-loopback Bolt start is rejected. Use
    /// loopback for this distribution.
    #[arg(long, default_value_t = false, requires = "bolt")]
    pub allow_remote_bolt_bind: bool,

    /// Data directory for the durable embedded store.
    ///
    /// When set, `serve` bootstraps a durable substrate rooted here:
    /// `<dir>/pages.db` (file-backed page store, fsync via `sync_data`) +
    /// `<dir>/wal/` (write-ahead log) + WAL recovery on startup. The
    /// `DEFAULT` tenant runs at `DurabilityTier::Strict` (fsync-before-ack),
    /// so acknowledged commits survive process restart. Created if absent.
    ///
    /// Mutually exclusive with [`--in-memory`](Self::in_memory). `serve`
    /// with neither flag refuses to start.
    #[arg(long, value_name = "DIR")]
    pub data: Option<PathBuf>,

    /// Run with an ephemeral, non-durable in-memory store.
    ///
    /// Uses no WAL. **All committed data is lost on process exit.** Intended
    /// for tests and ephemeral demos only. Mutually exclusive with
    /// [`--data`](Self::data); pass exactly one.
    #[arg(long, conflicts_with = "data")]
    pub in_memory: bool,

    /// Adopt an existing unstamped `--data` directory as the current format.
    ///
    /// A durable data dir created before the on-disk version stamp existed
    /// (a `<data>/VERSION` file) is refused at boot by default — the guard
    /// can't tell a same-format beta dir from an incompatible one, so it
    /// fails CLOSED rather than misparse. Pass this flag to explicitly
    /// assert "this directory is the current data-dir format": the server
    /// stamps `<data>/VERSION` at the
    /// current version and proceeds. This is the ONLY way to stamp a dir that
    /// already holds data — accidental binary-swaps (no flag) stay refused.
    ///
    /// It does NOT rescue an *incompatible* stamped version (the on-disk
    /// format really differs) — that still fails loud and needs a matching
    /// binary or a restore. Durable (`--data`) mode only; `requires = "data"`
    /// with `conflicts_with = "in_memory"` reject the flag on a non-durable
    /// start loudly (rather than silently no-op'ing on `--in-memory`).
    #[arg(long, requires = "data", conflicts_with = "in_memory")]
    pub adopt_legacy_datadir: bool,

    /// Enable WAL-at-rest encryption.
    ///
    /// When set, every WAL record payload is AES-256-GCM-encrypted at rest
    /// under a data-encryption key (DEK) that is itself wrapped by a
    /// key-encryption key (KEK) resolved from the selected secrets provider
    /// (see [`--wal-secrets-provider`](Self::wal_secrets_provider)). The
    /// wrapped DEK is persisted in `<data>/wal/wal.dek`; the plaintext DEK
    /// never touches disk. Durable (`--data`) mode only — `--in-memory` has
    /// no WAL.
    ///
    /// This is off by default. If the KEK is unresolvable at startup, the
    /// server refuses to start rather than writing plaintext WAL. This flag
    /// does not encrypt `pages.db`.
    #[arg(long, requires = "data")]
    pub wal_encryption: bool,

    /// Secrets provider backing the WAL-encryption KEK.
    ///
    /// `os-keyring` (default) resolves the KEK from the OS keyring (requires
    /// the binary be built `--features os-keyring`); `env` reads it from
    /// `ARCGRAPH_SECRET_*` env vars (DEVELOPMENT ONLY — emits an
    /// `unsafe_for_prod` warning). Ignored unless
    /// [`--wal-encryption`](Self::wal_encryption) is set.
    #[arg(
        long,
        value_name = "KIND",
        default_value = "os-keyring",
        requires = "wal_encryption"
    )]
    pub wal_secrets_provider: WalSecretsProviderArg,

    /// Enforce the per-tenant MCP token-bucket rate cap.
    ///
    /// Limits each tenant to 100 reads and 10 writes per minute in the MCP
    /// dispatcher (`-32007` when exceeded).
    ///
    /// **Scope: the MCP dispatch path only (`--http` + `--stdio-mcp`).**
    /// Both go through [`build_default_dispatcher`], which threads this
    /// flag into the [`Dispatcher`]'s per-request gate. The Bolt transport
    /// does not apply this limiter.
    ///
    /// This is off by default.
    #[arg(long, default_value_t = false)]
    pub rate_limit: bool,

    /// Maximum concurrent blocking dispatches per network transport.
    ///
    /// `0` (the default) resolves to `2 × logical cores`. Applies to the
    /// concurrent network transports (`--http`, `--bolt`); the stdio
    /// transport is strictly sequential (≤ 1 in-flight dispatch) so the
    /// cap is moot there and it always uses the default. Raise it if a
    /// deployment has many cores + a fast durable device and the default
    /// under-utilises the blocking pool; lower it to bound blocking-pool
    /// growth on a memory-constrained host.
    #[arg(long, value_name = "N", default_value_t = 0)]
    pub dispatch_bulkhead_permits: usize,

    /// Reserved server config TOML path.
    ///
    /// Accepted but ignored in this beta. Passing it emits a warning; use
    /// explicit CLI flags.
    #[arg(long, value_name = "PATH")]
    pub config: Option<PathBuf>,

    /// Admin HTTP server bind address (livez / readyz endpoints).
    ///
    /// Exposes `GET /livez` (always 200) and
    /// `GET /readyz` (200 when storage + WAL + index are loaded,
    /// 503 with not-ready component JSON otherwise). Non-loopback binds
    /// (e.g., `0.0.0.0:8090` for Kubernetes httpGet probes from
    /// the host network namespace) require explicit operator
    /// opt-in via [`--allow-remote-admin-bind`](Self::allow_remote_admin_bind).
    ///
    /// Default: `127.0.0.1:8090`. Empty string disables the admin HTTP
    /// server.
    #[arg(long, value_name = "ADDR", default_value = "127.0.0.1:8090")]
    pub admin_http: String,

    /// Operator opt-in to bind the admin HTTP server to a non-loopback
    /// address.
    ///
    /// Kubernetes probes from outside the pod's network namespace need
    /// `--admin-http 0.0.0.0:8090 --allow-remote-admin-bind`. The explicit
    /// flag prevents an accidental routable bind.
    ///
    /// When set with a loopback `--admin-http` address the flag is
    /// a no-op (validate accepts loopback regardless).
    #[arg(long, default_value_t = false)]
    pub allow_remote_admin_bind: bool,

    /// Graceful-shutdown drain grace period in seconds.
    ///
    /// On SIGTERM the readiness gate is flipped to `Draining` and the admin
    /// HTTP `/readyz` endpoint
    /// returns 503 immediately; the process then sleeps for this
    /// many seconds before letting `shutdown_on_term` resolve. The
    /// sleep gives Kubernetes Service endpoints time to remove the
    /// draining pod from the LB target pool BEFORE in-flight
    /// connections terminate (kube-proxy reconciles via the
    /// readyz=503 → endpoints-remove → iptables-update chain).
    ///
    /// Default: `15`. Set to `0` for synchronous shutdown (no drain
    /// window) — useful for `--stdio-mcp` deployments where the
    /// transport owns its own backpressure.
    #[arg(long, value_name = "SECONDS", default_value_t = 15)]
    pub drain_grace_seconds: u64,

    /// Bind address for the Prometheus `/metrics` scrape listener.
    ///
    /// **Default `127.0.0.1:9090` (metrics ON, loopback).** With no flag,
    /// the binary instantiates a [`MetricsRegistry`], threads it into the
    /// stdio/bolt transports (per-dispatch tool-invocation + read/write
    /// latency + the connection gauge), and binds a SEPARATE axum
    /// `/metrics` listener on loopback:9090. The
    /// listener is a distinct axum server on its own port, NOT in the MCP
    /// request hot path.
    ///
    /// **To DISABLE:** pass an explicit empty value — `--metrics-http ""`
    /// (or `--metrics-http=''`). A whitespace-only value hits the
    /// `trim().is_empty()` gate in `run_serve` → `metrics_registry = None`:
    /// no registry and no listener.
    ///
    /// The default is loopback, so it passes
    /// `MetricsHttpServerConfig::validate`; a non-loopback
    /// address (default OR override) STILL requires
    /// [`--allow-remote-metrics-bind`](Self::allow_remote_metrics_bind).
    /// Metrics use a listener separate from admin HTTP and MCP.
    #[arg(long, value_name = "ADDR", default_value = "127.0.0.1:9090")]
    pub metrics_http: String,

    /// Operator opt-in to bind the `--metrics-http` scrape listener to a
    /// non-loopback address.
    ///
    /// A Kubernetes `ServiceMonitor` scraping `0.0.0.0:9090` needs this;
    /// localhost-only operators leave it off. No-op when `--metrics-http`
    /// is a loopback address.
    #[arg(long, default_value_t = false)]
    pub allow_remote_metrics_bind: bool,

    /// Community-detection (Leiden) refresh cadence, in seconds.
    ///
    /// When `--metrics-http` is set, the serve binary starts a
    /// background community refresh scheduler so each successful per-tenant
    /// Leiden refresh fires `arcgraph_leiden_last_run_seconds{tenant}` into
    /// the scraped registry. This flag
    /// sets that scheduler's tick interval.
    ///
    /// Default: `86400` (24 hours). The metric is absent on `/metrics` until
    /// the first successful refresh completes in this process,
    /// so at the default cadence a freshly-started server exposes no series
    /// for up to a day. Operators validating the scrape end-to-end (or
    /// running on small graphs where a daily refresh is over-conservative)
    /// can lower this; the refresh cost is `O(edges)` per tenant per tick, so
    /// very low values on large graphs are wasteful.
    ///
    /// No-op when `--metrics-http` is unset: with no registry there is no
    /// observer to wire, so no scheduler is started.
    #[arg(long, value_name = "SECONDS", default_value_t = 86_400)]
    pub community_refresh_secs: u64,
}

/// `arcgraph check` — verify store integrity.
#[derive(Debug, Args)]
pub struct CheckArgs {
    /// Data directory to check.
    ///
    /// Without `--data`, checks an in-memory empty-tenant catalog. With
    /// `--data`, validates the directory and, when it contains a committed
    /// generation, cold-opens the production store, routes every catalog
    /// tenant, and hydrates a bounded sample of node and relationship property
    /// bags. This is not an exhaustive record scan or a chaos test.
    #[arg(long, value_name = "DIR")]
    pub data: Option<PathBuf>,
}

/// `arcgraph dump` — unavailable storage-rooted logical export.
#[derive(Debug, Args)]
pub struct DumpArgs {
    /// Durable data directory. Supplying it makes this beta refuse safely.
    #[arg(long, value_name = "DIR")]
    pub data: Option<PathBuf>,

    /// Output format. Defaults to JSON.
    #[arg(long, value_name = "FMT", default_value_t = DumpFormat::Json)]
    pub format: DumpFormat,

    /// Tenant ID to dump. Defaults to [`TenantId::DEFAULT`] (1).
    #[arg(long, value_name = "ID", default_value_t = TenantId::DEFAULT.raw())]
    pub tenant: u64,
}

/// `arcgraph health` — plain-HTTP liveness probe.
///
/// Sends a single `GET /healthz` against the configured URL and exits
/// `0` on a 2xx response, `1` on anything else (non-2xx, connect refused,
/// connect/read timeout, malformed status line).
///
/// Only `http://` URLs are accepted. The MCP network transport is HTTPS-only,
/// so this command cannot probe it; use a TLS-aware client for that listener.
#[derive(Debug, Args)]
pub struct HealthArgs {
    /// Target URL. Default `http://127.0.0.1:8080/healthz`.
    ///
    /// The path must address a plain-HTTP `/healthz` endpoint.
    #[arg(long, value_name = "URL", default_value = DEFAULT_HEALTH_URL)]
    pub addr: String,

    /// Per-attempt connect+write+read timeout in milliseconds.
    ///
    /// Default `2000` (2 seconds).
    #[arg(long, value_name = "MS", default_value_t = 2000)]
    pub timeout_ms: u64,
}

/// Default `--addr` for `arcgraph health`.
///
/// The default remains loopback-only.
const DEFAULT_HEALTH_URL: &str = "http://127.0.0.1:8080/healthz";

/// Output format for `arcgraph dump`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum DumpFormat {
    /// JSON (one object per line; tenant header first).
    Json,
    /// TOON envelope.
    Toon,
    /// openCypher CREATE statements.
    Cypher,
}

impl std::fmt::Display for DumpFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            DumpFormat::Json => "json",
            DumpFormat::Toon => "toon",
            DumpFormat::Cypher => "cypher",
        };
        f.write_str(s)
    }
}

/// CLI value for the WAL-encryption KEK secrets backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum WalSecretsProviderArg {
    /// OS keyring (production default; requires `--features os-keyring`).
    OsKeyring,
    /// Environment-variable provider (DEVELOPMENT ONLY).
    Env,
}

impl ServeArgs {
    /// Build the [`WalEncryptionConfig`] from the serve flags.
    pub(crate) fn wal_encryption_config(&self) -> WalEncryptionConfig {
        WalEncryptionConfig {
            enabled: self.wal_encryption,
            // The only current KeySourceKind is the secrets provider.
            key_source: arcgraph_cli::bootstrap::KeySourceKind::default(),
            secrets_provider: match self.wal_secrets_provider {
                WalSecretsProviderArg::OsKeyring => SecretsProviderKind::OsKeyring,
                WalSecretsProviderArg::Env => SecretsProviderKind::Env,
            },
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Production storage-backed adapters live in `arcgraph::mcp::storage`.
// The umbrella and standalone stdio binaries use the same composition.
// ─────────────────────────────────────────────────────────────────────

/// Bootstrap the per-process storage substrate for the umbrella
/// `arcgraph serve` subcommand.
///
/// Thin re-export of
/// [`arcgraph_cli::bootstrap::bootstrap_storage_backend_with_metrics`]
/// so this binary and the standalone `arcgraph-mcp-stdio` binary share one
/// construction path.
///
/// The returned [`DurabilityGuard`] owns the WAL writer thread in durable
/// mode; callers MUST hold it for the server loop's lifetime.
///
/// `metrics` carries the process [`MetricsRegistry`] when the operator did
/// not disable `--metrics-http`. It is
/// coerced to the storage-resident `dyn MetricsSink` and threaded into
/// the [`arcgraph_storage::crud::CrudStore`] so TEL overflow fires the
/// `arcgraph_hot_vertex_warnings_total{tenant}` counter. `None` disables
/// metrics.
fn bootstrap_storage_backend(
    mode: &BootstrapMode,
    metrics: Option<Arc<MetricsRegistry>>,
) -> Result<(StorageBackend, DurabilityGuard)> {
    // WAL encryption is off unless `serve --wal-encryption` selects the
    // `_with_encryption` helper below. `adopt_legacy = false`: the
    // legacy-datadir adopt (#1302) is a serve-only opt-in.
    bootstrap_storage_backend_with_encryption(mode, metrics, &WalEncryptionConfig::default(), false)
}

/// [`bootstrap_storage_backend`] variant that threads the operator's
/// [`WalEncryptionConfig`] into durable bootstrap.
///
/// SVC-2 / #1302 — also threads `adopt_legacy` (the
/// `serve --adopt-legacy-datadir` opt-in) into the durable data-dir version
/// guard so a beta operator can explicitly adopt an unstamped same-format dir.
fn bootstrap_storage_backend_with_encryption(
    mode: &BootstrapMode,
    metrics: Option<Arc<MetricsRegistry>>,
    wal_encryption: &WalEncryptionConfig,
    adopt_legacy: bool,
) -> Result<(StorageBackend, DurabilityGuard)> {
    let sink: Option<Arc<dyn MetricsSink>> = metrics.map(|r| r as Arc<dyn MetricsSink>);
    arcgraph_cli::bootstrap::bootstrap_storage_backend_with_metrics_encryption_and_adopt(
        mode,
        sink,
        wal_encryption,
        adopt_legacy,
    )
}

/// Start the background community refresh scheduler when the operator wired
/// `--metrics-http` (i.e. a process [`MetricsRegistry`] exists to observe
/// into). The scheduler is built over the SAME `catalog` / `crud` /
/// `txn_manager` the served `backend` reads (via the new
/// [`arcgraph_storage::router::MultiTenantRouter::catalog`] / `crud`
/// accessors + [`StorageBackend::txn_manager`]), so each Leiden refresh runs
/// on the served graph and sets
/// `arcgraph_leiden_last_run_seconds{tenant}`.
///
/// Called by every transport (`stdio` / `http` / `bolt`) BEFORE `backend`
/// is moved into the dispatcher, so it reads the handles through a borrow.
/// The returned scheduler owns a dedicated OS thread; the caller holds it
/// for the serve loop and calls [`CommunityRefreshScheduler::shutdown`]
/// after the transport returns (symmetric with the `DurabilityGuard`
/// WAL-thread ownership).
///
/// Returns `None` when `metrics` is `None`.
fn maybe_start_community_scheduler(
    backend: &StorageBackend,
    metrics: Option<&Arc<MetricsRegistry>>,
    refresh_secs: u64,
) -> Option<Arc<arcgraph::community::CommunityRefreshScheduler>> {
    let registry = metrics?;
    // Coerce the SAME registry the `/metrics` listener scrapes into the
    // community observer seam (ADR-202 §D-4 — mirrors the `dyn MetricsSink`
    // coercion in `bootstrap_storage_backend`). The gauge therefore lands
    // on the exact endpoint the operator scrapes.
    let observer: Arc<dyn arcgraph::community::RefreshObserver> =
        Arc::clone(registry) as Arc<dyn arcgraph::community::RefreshObserver>;
    let scheduler_config = arcgraph::community::SchedulerConfig {
        interval: Duration::from_secs(refresh_secs),
        ..arcgraph::community::SchedulerConfig::default()
    };
    let scheduler = arcgraph_cli::bootstrap::start_community_scheduler(
        Arc::clone(backend.router().catalog()),
        Arc::clone(backend.router().crud()),
        Arc::clone(backend.txn_manager()),
        observer,
        scheduler_config,
    );
    Some(scheduler)
}

// ─────────────────────────────────────────────────────────────────────
// Subcommand bodies
// ─────────────────────────────────────────────────────────────────────

/// Build a production-wired dispatcher pinned to [`TenantId::DEFAULT`].
///
/// All six adapter slots route through storage-backed
/// `arcgraph_mcp::storage::Storage*` types.
///
/// AHP-1 (ADR-225 §3) — build the dispatch bulkhead from the
/// `--dispatch-bulkhead-permits` flag: `0` resolves to the default cap
/// (2 × logical cores); any other value is used verbatim.
fn dispatch_bulkhead_from_args(args: &ServeArgs) -> DispatchBulkhead {
    if args.dispatch_bulkhead_permits == 0 {
        DispatchBulkhead::with_default_cap()
    } else {
        DispatchBulkhead::new(args.dispatch_bulkhead_permits)
    }
}

fn build_default_dispatcher(
    backend: StorageBackend,
    metrics: Option<Arc<MetricsRegistry>>,
    rate_limit: bool,
    data_dir: Option<&Path>,
) -> Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
> {
    let session_tenant = TenantId::DEFAULT;
    let schema_provider = Arc::new(StorageSchemaProvider::new(backend.clone()));
    let node_inspector = Arc::new(StorageNodeInspector::new(backend.clone()));
    let neighborhood_explorer = Arc::new(StorageNeighborhoodExplorer::new(backend.clone()));
    // #765 PART-1 / #1292 PART-3 — one served vector provider instance, shared by
    // graph.search (StorageHybridSearcher) AND ArcQL RANK BY
    // (StorageRawQueryExecutor's substrate). `VectorSearchTier::from_env` selects
    // the tier: HNSW (default; lazily builds a per-tenant ephemeral HNSW) OR the
    // RAM-decoupled SSD DiskANN tier (ADR-195; `ARCGRAPH_VECTOR_TIER=ssd`, RSS
    // ceiling enforced so a large ingest aborts cleanly instead of OOMing). The
    // SSD index directory co-locates under the serve `--data` root when durable.
    let vector_provider: Arc<dyn SubstrateSearchProvider> =
        VectorSearchTier::from_env(data_dir).build_provider(backend.clone());
    let hybrid_searcher = Arc::new(
        StorageHybridSearcher::new(backend.clone())
            .with_search_provider(Arc::clone(&vector_provider)),
    );
    let ingest_provider = Arc::new(StorageIngestProvider::new(backend.clone()));
    // W28 Feature #582 (ADR-045) — thread the process MetricsRegistry
    // (coerced to `dyn MetricsSink`) into the raw-query executor so each
    // `graph.raw_query` execution fires
    // `arcgraph_query_plan_choice{plan_type}` (§10.2 line 723). `None`
    // (no `--metrics-http`) leaves the executor's sink unset (no-op).
    let raw_query_executor = {
        // #765 PART-1 — bind the same provider so `RANK BY vector(...)` runs KNN.
        let exec = StorageRawQueryExecutor::new(backend.clone())
            .with_search_provider(Arc::clone(&vector_provider));
        // #1291 — enable the per-tenant memory budget with the served
        // default (1 GiB; `ARCGRAPH_TENANT_MEMORY_CAP_BYTES` overrides,
        // `0` disables). Without this the budget ships DISABLED and the
        // only guard is the ≈4.29 B-row runaway fallback → OOM under a
        // heavy `graph.raw_query`.
        let exec = match arcgraph_cli::ops::resolve_per_tenant_memory_cap() {
            Some(cap) => exec.with_per_tenant_memory_cap(cap),
            None => exec,
        };
        let exec = match metrics {
            Some(registry) => exec.with_metrics_sink(registry as Arc<dyn MetricsSink>),
            None => exec,
        };
        Arc::new(exec)
    };
    // #1186 / #833 / #818 — per-tenant rate-limit is OPT-IN
    // (`serve --rate-limit`), default-OFF on EVERY transport this builder
    // feeds (`--http`, `--bolt`, `--stdio-mcp` all share this dispatcher).
    //
    // The W14γ M5-12 limiter's per-MINUTE read cap (100/min, design-v2
    // §9.4 / ADR-004 amendment-02) is a multi-tenant NETWORK control. ON
    // BY DEFAULT it silently throttled the primary agent-native workload
    // — `graph.search` + `graph.raw_query` share one capacity-100
    // `(tenant, Read)` bucket, so an agent's >100-read/min recall sweep
    // had reads past ~#100 rejected with `-32007`, which agent clients
    // coerce to an empty result (#818 served-vector recall pinned at
    // 0.50). This mirrors the trusted-local-vs-network split #818 applies
    // to the stdio frame cap; see the fuller rationale in
    // `bin/arcgraph_mcp_stdio.rs::run`.
    //
    // #1186 (MUST-LLM-04) — the limiter must be ENFORCEABLE on the served
    // surface (the AC: "a customer can observe a per-tenant rate cap").
    // `serve --rate-limit` opts INTO it across all transports: a
    // multi-tenant network deployment gets the DoS / noisy-neighbor cap;
    // the default (no flag) stays unthrottled so the trusted-local
    // single-agent workload is unaffected. When the flag is OFF,
    // non-loopback HTTP exposure relies on the loopback-default bind gate
    // + TLS, with `--http-auth-jwks` enabling the existing bearer
    // verifier on every `POST /mcp`. A NETWORK record-rate limiter (a
    // DIFFERENT control from the per-minute request cap) remains tracked
    // for the RBAC-on-wire slice.
    //
    // W16ζ M5-11 — `with_session_scope` binds `SessionScope::Power`
    // (matching `arcgraph-mcp-stdio.rs`) so the embedded operator
    // retains access to `graph.raw_query`; M5-03 OAuth swaps this for
    // a Bearer-token-derived derivation across all transports.
    //
    // #1186 — base dispatcher: with the per-tenant limiter when
    // `--rate-limit` is set, otherwise the default unthrottled shape.
    // `RateLimiter::new()` seeds the design-v2 §9.4 defaults (100 read /
    // 10 write per minute per tenant) lazily on first observation.
    if rate_limit {
        tracing::info!(
            target: "arcgraph_cli::serve",
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
    }
}

/// `arcgraph serve --stdio-mcp` — mirror the standalone
/// `arcgraph-mcp-stdio` binary's loop.
async fn run_serve_stdio(
    args: &ServeArgs,
    gate: &ReadinessGate,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    metrics: Option<Arc<MetricsRegistry>>,
) -> Result<()> {
    warn_unwired_opts(args);
    // W28 / ADR-183 — resolve the durable-by-default storage mode (refuse
    // to start without `--data` XOR `--in-memory`).
    let mode = BootstrapMode::from_flags(args.data.as_deref(), args.in_memory)?;
    // W28 Feature #582 (ADR-045) — pass the process metrics registry so the
    // CrudStore fires the hot-vertex counter (§10.2 line 721) under
    // `--metrics-http`.
    // ADR-216 §D-4 / #1180 — thread the operator's WAL-encryption config
    // (OFF by default; `--wal-encryption` opts in).
    let (backend, _durability) = bootstrap_storage_backend_with_encryption(
        &mode,
        metrics.clone(),
        &args.wal_encryption_config(),
        args.adopt_legacy_datadir,
    )?;
    // W28 / ADR-183 — `bootstrap_storage_backend` ran WAL recovery
    // synchronously before returning in durable mode (`--in-memory` has
    // nothing to replay), so the readiness components are Ready the moment
    // it returns. `_durability` owns the WAL writer thread and MUST be held
    // for the serve loop's lifetime — dropping it shuts the WAL down. Its
    // Drop drains + fsyncs + joins on clean shutdown AND fires the ADR-229
    // graceful-shutdown checkpoint (#849) before the drain.
    // SVC-1 / #849 / ADR-229 — background interval checkpointer (Tokio
    // work-stealing pool, NOT the hot path): keeps restart-recovery bounded
    // between graceful shutdowns on a long-running durable serve. `None`
    // (in-memory / disabled) → no task. Held for the serve loop lifetime.
    let _checkpoint_task = arcgraph_cli::bootstrap::spawn_interval_checkpointer(
        _durability.checkpointer(),
        arcgraph_storage::config::WalCheckpointConfig::default(),
    );
    gate.mark_ready("storage");
    gate.mark_ready("wal");
    gate.mark_ready("index");
    // ADR-202 §D-8 — start the community refresh scheduler (with the
    // metrics observer) BEFORE `backend` is consumed by the dispatcher, so
    // the eighth §10.2 metric fires once a refresh completes. `None` (no
    // `--metrics-http`). Held for the serve loop; shut down after the loop.
    let community_scheduler =
        maybe_start_community_scheduler(&backend, metrics.as_ref(), args.community_refresh_secs);
    let dispatcher = build_default_dispatcher(
        backend,
        metrics.clone(),
        args.rate_limit,
        args.data.as_deref(),
    );

    let cancel_registry = CancellationRegistry::new();
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    // W17δ #280 — SIGTERM-during-query handler.
    //
    // `shutdown` is the shared draining future built in `run_serve`
    // (W24-OPS-α R1 fix-up M2: SIGTERM → mark_draining → sleep
    // --drain-grace-seconds → resolve). Before letting
    // `serve_stdio` see the shutdown signal we fire every in-flight
    // query's CancellationToken via
    // `CancellationRegistry::cancel_all()` — this is the ADR-038
    // amendment-03 §TIER-1 GAP C "graceful drain at shutdown" seam.
    // This converts an operator's SIGTERM into cancellation for every
    // in-flight query registered on this transport.
    let cancel_registry_for_shutdown = cancel_registry.clone();
    let shutdown = async move {
        shutdown.await;
        let fired = cancel_registry_for_shutdown.cancel_all();
        tracing::info!(
            target: "arcgraph_cli::serve_stdio",
            fired_count = fired,
            "shutdown signal received — fired {fired} in-flight cancellation tokens",
        );
    };

    tracing::info!(
        target: "arcgraph_cli::serve_stdio",
        tenant_id = TenantId::DEFAULT.raw(),
        "arcgraph serve --stdio-mcp starting with production storage adapters",
    );

    // Thread the process `MetricsRegistry` into `serve_stdio`. When `Some`,
    // `--metrics-http`), the stdio loop records BOTH
    // `active_connections{transport="stdio"}` (the W16γ M6-07 gauge) AND
    // — per the #588 dispatch wire added to `transport/stdio.rs` —
    // `arcgraph_mcp_tool_invocations` + `arcgraph_{read,write}_latency_ms`
    // on every dispatched JSON-RPC request. That is the data the
    // `--metrics-http` listener (spawned in `run_serve`) scrapes. When
    // `None` (no `--metrics-http`), the legacy zero-overhead path is
    // preserved (no registry is even instantiated). design-v2 §10.2.
    // AHP-1 (ADR-225 §3) — `serve_stdio` now takes `Arc<Dispatcher>` (the
    // `spawn_blocking` bulkhead needs an owned `'static` dispatcher). The
    // The dispatcher is moved into the stdio service.
    let stats = serve_stdio(
        Arc::new(dispatcher),
        &cancel_registry,
        stdin,
        stdout,
        shutdown,
        metrics,
    )
    .await
    .context("serve_stdio loop returned an error")?;

    tracing::info!(
        target: "arcgraph_cli::serve_stdio",
        messages_in = stats.messages_in,
        messages_out = stats.messages_out,
        exit_reason = ?stats.exit_reason,
        "arcgraph serve --stdio-mcp exiting cleanly",
    );

    // ADR-202 §D-8 — join the community scheduler's thread before returning
    // (the transport has stopped; the registry the observer writes is
    // dropped after `run_serve` drains the metrics task, so stop the
    // producer first). No-op when no scheduler was started.
    if let Some(scheduler) = community_scheduler {
        scheduler.shutdown();
    }

    Ok(())
}

/// Parse + gate the `--http` bind address.
///
/// design-v2 §9.4 line 668 + W14 retro IR L1-HIGH-4 (loopback-default
/// discipline): loopback binds (`127.0.0.1` / `::1`) are always allowed;
/// non-loopback binds (e.g. `0.0.0.0:8443`) require the operator to
/// opt in via `--allow-remote-http-bind`. This mirrors the
/// [`ServeArgs::allow_remote_admin_bind`] / [`ServeArgs::allow_remote_metrics_bind`]
/// gates already applied to the admin + metrics surfaces.
///
/// Checked in [`run_serve_http`] BEFORE any cert I/O or storage
/// bootstrap so a misconfigured bind fails fast with a crisp error.
/// [`HttpServerConfig::validate`] re-asserts the same invariant inside
/// `serve_http` (defense in depth); keeping the CLI-side gate means
/// removing it RED-fails the `http_bind_gate_*` unit tests.
fn validate_http_bind(addr: &str, allow_remote: bool) -> Result<SocketAddr> {
    let bind: SocketAddr = addr.parse().with_context(|| {
        format!("--http {addr}: not a valid SocketAddr (e.g. `127.0.0.1:8443`)")
    })?;
    if !bind.ip().is_loopback() && !allow_remote {
        bail!(
            "--http {addr}: refusing to bind a non-loopback address without \
             --allow-remote-http-bind (loopback-default per design-v2 §9.4 line 668 \
             + W14 retro IR L1-HIGH-4). Pass --allow-remote-http-bind to expose the \
             HTTPS MCP transport beyond loopback (ensure a network policy is in place)."
        );
    }
    Ok(bind)
}

/// Build the W13ε [`HotReloadResolver`] from staged PEM cert/key files.
///
/// [`HotReloadResolver::new`] runs the full
/// [`FileSystemCertProvider::load_validated`] pipeline (PEM parse,
/// key/cert match, validity window, optional RFC 6125 hostname) on the
/// initial load, so a missing / malformed / expired / key-mismatched
/// cert surfaces here as a clean startup error (the
/// [`arcgraph_mcp::tls::TlsResolverError`] path translated to
/// [`anyhow::Error`]) rather than an unwrap panic.
fn build_http_tls_resolver(
    cert_path: &std::path::Path,
    key_path: &std::path::Path,
    hostname: Option<String>,
) -> Result<Arc<HotReloadResolver>> {
    let provider = FileSystemCertProvider::new(cert_path, key_path, hostname);
    HotReloadResolver::new(Arc::new(provider)).with_context(|| {
        format!(
            "serve --http: failed to load TLS cert/key (cert={}, key={}) — check the PEM \
             files exist, the key matches the cert, and the cert is within its validity window",
            cert_path.display(),
            key_path.display(),
        )
    })
}

/// `arcgraph serve --http <addr>` — live HTTPS MCP transport.
///
/// Wires the TLS file flags into the [`HotReloadResolver`] and hands it,
/// together with the same storage substrate used by stdio and Bolt, to
/// [`serve_http`]. ArcGraph validates externally issued bearer tokens; it
/// does not issue them.
async fn run_serve_http(
    addr: &str,
    args: &ServeArgs,
    gate: &ReadinessGate,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    metrics: Option<Arc<MetricsRegistry>>,
) -> Result<()> {
    warn_unwired_opts(args);

    // design-v2 §9.4 line 668 + W14 retro IR L1-HIGH-4 — loopback-default
    // bind gate. Cheap (no I/O); fail fast before reading certs / spinning
    // storage on a misconfigured bind.
    let bind = validate_http_bind(addr, args.allow_remote_http_bind)?;

    // Refuse to start without exactly one of `--data` and `--in-memory`.
    // The stdio and Bolt paths enforce the same policy.
    let mode = BootstrapMode::from_flags(args.data.as_deref(), args.in_memory)?;

    // `serve --http` REQUIRES `--tls-cert` + `--tls-key`. Clap's
    // `requires_all` enforces this at parse time; this match re-checks
    // defensively so a programmatic caller gets a structured error
    // instead of an unwrap panic on the `Option`s.
    let (cert_path, key_path) = match (args.tls_cert.as_deref(), args.tls_key.as_deref()) {
        (Some(cert), Some(key)) => (cert, key),
        _ => bail!(
            "serve --http requires --tls-cert <PEM> and --tls-key <PEM> (server-side TLS is \
             mandatory for the HTTP MCP transport; design-v2 §9.4 enforces HTTPS for non-stdio)"
        ),
    };

    // Build the W13ε hot-reload resolver from the staged cert/key. A
    // missing / malformed / expired cert surfaces a clean startup error.
    let resolver = build_http_tls_resolver(cert_path, key_path, args.tls_hostname.clone())?;
    // #761 slice 3 (ADR-044/049) — wire HTTPS MCP OAuth when the operator
    // stages a JWKS file. `build_http_oauth_config` parses the RFC 7517 JWKS
    // JSON, constructs `OAuthConfig`, and returns `Some(Arc<OAuthConfig>)`.
    // When `None` (no `--http-auth-jwks`) HTTP runs in dev-mode: no bearer
    // verification, with loopback-default + TLS still enforced by the gates.
    let oauth_config = build_http_oauth_config(args)?;
    if !bind.ip().is_loopback() && args.allow_remote_http_bind && oauth_config.is_none() {
        tracing::warn!(
            target: "arcgraph_cli::serve_http",
            bind = %bind,
            "serve --http: non-loopback bind WITHOUT OAuth auth (no --http-auth-jwks). \
             Any HTTPS MCP client can issue requests with a matching tenant header. \
             Configure --http-auth-jwks for production deployments (design-v2 §9.4)."
        );
    }

    // Build the SAME storage substrate the stdio/bolt paths build
    // (durable via `--data`, ephemeral via `--in-memory`). W28 Feature
    // #582 (ADR-045) — thread the metrics registry so the CrudStore fires
    // the hot-vertex counter (§10.2 line 721) under `--metrics-http`.
    // `_durability` owns the WAL writer thread in durable mode and MUST
    // be held for the serve loop's lifetime.
    // ADR-216 §D-4 / #1180 — thread the operator's WAL-encryption config.
    let (backend, _durability) = bootstrap_storage_backend_with_encryption(
        &mode,
        metrics.clone(),
        &args.wal_encryption_config(),
        args.adopt_legacy_datadir,
    )?;
    gate.mark_ready("storage");
    gate.mark_ready("wal");
    gate.mark_ready("index");

    // ADR-202 §D-8 — start the community refresh scheduler (with the
    // metrics observer) BEFORE `backend`/`metrics` are moved into the
    // dispatcher, so the eighth §10.2 metric fires on the same registry the
    // `--metrics-http` listener scrapes. `None` without `--metrics-http`.
    let community_scheduler =
        maybe_start_community_scheduler(&backend, metrics.as_ref(), args.community_refresh_secs);

    // W28 Feature #582 (ADR-045) — `build_default_dispatcher` threads the
    // metrics registry into the raw-query executor (graph.raw_query
    // plan-choice counter, §10.2 line 723), same as the stdio path. The
    // `/metrics` scrape stays on the dedicated `--metrics-http` loopback
    // listener (ADR-093 §"Why a separate admin port") — we deliberately
    // do NOT mount `/metrics` on the public HTTPS data port.
    let dispatcher = Arc::new(build_default_dispatcher(
        backend,
        metrics,
        args.rate_limit,
        args.data.as_deref(),
    ));
    let cancel_registry = Arc::new(CancellationRegistry::new());

    // HttpServerConfig — pinned to `TenantId::DEFAULT` (the per-process
    // session pin, matching stdio's `session_tenant`) so a forged
    // `X-ArcGraph-Tenant` header is rejected at the transport boundary.
    // `allow_remote_bind` mirrors the CLI gate so `serve_http`'s own
    // `validate()` agrees (defense in depth). When `oauth_config` is Some,
    // `serve_http` verifies `Authorization: Bearer <jwt>` before body
    // parse/dispatch.
    let mut config = HttpServerConfig::new(bind, Arc::clone(&resolver))
        .with_allow_remote_bind(args.allow_remote_http_bind)
        .with_bound_tenant(TenantId::DEFAULT)
        // AHP-1 (ADR-225 §3) — install the operator-configured dispatch
        // bulkhead so a blocking write no longer starves reads (#999).
        .with_dispatch_bulkhead(dispatch_bulkhead_from_args(args));
    if let Some(cfg) = oauth_config {
        config = config.with_oauth(cfg);
    }

    // Spawn the SIGHUP-driven cert rotation loop so operators can
    // `kill -HUP` (or use k8s lifecycle hooks)
    // to rotate certs without a restart. `reload_stop` flips after
    // `serve_http` returns so the loop drains cleanly (no task leak).
    let (reload_stop_tx, reload_stop_rx) = tokio::sync::watch::channel(false);
    let reload_handle = tokio::spawn(run_sighup_reload_loop(
        Arc::clone(&resolver),
        reload_stop_rx,
    ));

    tracing::info!(
        target: "arcgraph_cli::serve_http",
        bind = %bind,
        tls_source = %resolver.source_descriptor(),
        allow_remote_bind = args.allow_remote_http_bind,
        oauth_enforced = config.oauth.is_some(),
        "arcgraph serve --http starting (#761 slices 1/3 — TLS hot-reload + OAuth-capable live MCP dispatch)",
    );

    // `serve_http` owns the `cancel_registry` and fires `cancel_all()`
    // itself when `shutdown` resolves (transport/http.rs), so — unlike the
    // stdio/bolt paths — we hand it the raw shutdown future with no
    // cancel-wrapper. It validates the config (bind gate + tenant-strategy
    // coherence) at the top, then binds + accepts until shutdown.
    let serve_result: Result<()> =
        serve_http(config, dispatcher, Arc::clone(&cancel_registry), shutdown)
            .await
            .map(|_stats| ())
            .context("serve_http loop returned an error");

    // Stop + join the SIGHUP reload loop regardless of how serve_http
    // exited (best-effort; the loop resolves Ok on a clean shutdown).
    let _ = reload_stop_tx.send(true);
    let _ = reload_handle.await;

    // ADR-202 §D-8 — join the community scheduler thread before propagating
    // the serve result (stop the metric producer before the registry is
    // dropped in `run_serve`'s metrics-task drain). No-op when unset.
    if let Some(scheduler) = community_scheduler {
        scheduler.shutdown();
    }

    serve_result?;

    tracing::info!(
        target: "arcgraph_cli::serve_http",
        "arcgraph serve --http exiting cleanly",
    );
    Ok(())
}

fn build_oauth_config_from_parts(
    jwks_path: &Path,
    issuer: &str,
    audiences: &[String],
    jwks_flag: &str,
    audience_flag: &str,
) -> Result<OAuthConfig> {
    if audiences.is_empty() {
        bail!(
            "{jwks_flag} requires at least one {audience_flag}; none provided. \
             Example: {audience_flag} arcgraph"
        );
    }
    let contents = std::fs::read_to_string(jwks_path)
        .with_context(|| format!("{jwks_flag} {}: cannot read JWKS file", jwks_path.display()))?;
    // Deserialize the RFC 7517 JWK Set. `serde_json` uses the same
    // `jsonwebtoken::jwk::JwkSet` shape that the production JWT verifier
    // consumes, so any key type `DecodingKey::from_jwk` supports is accepted.
    let jwt_jwks: jsonwebtoken::jwk::JwkSet =
        serde_json::from_str(&contents).with_context(|| {
            format!(
                "{jwks_flag} {}: invalid JWKS JSON (expected RFC 7517 JWK Set)",
                jwks_path.display()
            )
        })?;
    if jwt_jwks.keys.is_empty() {
        bail!(
            "{jwks_flag} {}: JWKS file contains zero keys",
            jwks_path.display()
        );
    }
    let mut keys = Vec::with_capacity(jwt_jwks.keys.len());
    for (i, jwk) in jwt_jwks.keys.iter().enumerate() {
        // `kid` is optional in RFC 7517 §4.5; fall back to an index-based
        // synthetic kid so single-key deployments (common in dev/test) work
        // without an explicit `kid` in the JWKS file. The `JsonWebKeySet`
        // resolver accepts no-kid when there's exactly one key.
        let kid = jwk
            .common
            .key_id
            .clone()
            .unwrap_or_else(|| format!("key-{i}"));
        // `alg` (RFC 7517 §4.4) is required for explicit algorithm pinning.
        // An absent `alg` is ambiguous (RSA vs ECDSA cannot be inferred from
        // key material alone in the general case) and would silently bypass
        // the OAuthConfig algorithm whitelist — fail loudly.
        let key_alg = jwk
            .common
            .key_algorithm
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{jwks_flag} {}: JWK[{i}] (kid={kid:?}) is missing the `alg` field; \
                     an explicit algorithm is required for security (prevents alg-confusion attacks)",
                    jwks_path.display()
                )
            })?;
        // KeyAlgorithm Display → Debug form matches Algorithm::from_str input
        // (e.g. "ES256", "RS256"). Private `KeyAlgorithm::to_algorithm()` does
        // the same thing; we replicate it here because it is not pub.
        let alg_str = format!("{key_alg:?}");
        let algorithm = jsonwebtoken::Algorithm::from_str(&alg_str).map_err(|_| {
            anyhow::anyhow!(
                "{jwks_flag} {}: JWK[{i}] (kid={kid:?}) unsupported algorithm `{alg_str}`; \
                 supported: RS256/384/512, ES256/384, PS256/384/512",
                jwks_path.display()
            )
        })?;
        let decoding_key = jsonwebtoken::DecodingKey::from_jwk(jwk).with_context(|| {
            format!(
                "{jwks_flag} {}: JWK[{i}] (kid={kid:?}): cannot build decoding key",
                jwks_path.display()
            )
        })?;
        keys.push(JsonWebKey {
            kid,
            algorithm,
            decoding_key,
        });
    }
    let jwks = JsonWebKeySet::new(keys)
        .map_err(|e| anyhow::anyhow!("{jwks_flag} {}: {e}", jwks_path.display()))?;
    Ok(OAuthConfig::new(
        issuer.to_string(),
        audiences.to_vec(),
        jwks,
    ))
}

/// Build the HTTP OAuth config from `--http-auth-jwks` /
/// `--http-auth-issuer` / `--http-auth-audience` CLI flags (#761 slice 3,
/// ADR-044 / ADR-049).
///
/// Returns `Ok(None)` when `--http-auth-jwks` is absent (dev-mode). Errors
/// on malformed JWKS JSON, missing `alg` / `kid` fields, unsupported algorithms,
/// or empty audiences (clap only enforces the `requires` constraint between flags
/// but cannot require a non-empty `Vec<String>` argument — validated here).
///
/// The returned `Arc<OAuthConfig>` is ready to pass directly to
/// [`HttpServerConfig::with_oauth`].
fn build_http_oauth_config(args: &ServeArgs) -> Result<Option<Arc<OAuthConfig>>> {
    let jwks_path = match &args.http_auth_jwks {
        Some(p) => p,
        None => return Ok(None),
    };
    // clap `requires_all = ["http", "http_auth_issuer"]` guarantees
    // http_auth_issuer is Some when http_auth_jwks is Some.
    let issuer = args
        .http_auth_issuer
        .as_deref()
        .expect("clap requires: http_auth_issuer is Some when http_auth_jwks is Some");
    let config = build_oauth_config_from_parts(
        jwks_path,
        issuer,
        &args.http_auth_audience,
        "--http-auth-jwks",
        "--http-auth-audience",
    )?;
    tracing::info!(
        target: "arcgraph_cli::serve_http",
        issuer,
        audiences = ?args.http_auth_audience,
        "HTTPS MCP OAuth enforced (ADR-044/049, #761 slice 3): only bearer JWTs accepted",
    );
    Ok(Some(Arc::new(config)))
}

/// Build the Bolt OAuth config from `--bolt-auth-jwks` / `--bolt-auth-issuer`
/// / `--bolt-auth-audience` CLI flags (#761 slice 2, ADR-049).
///
/// Returns `Ok(None)` when `--bolt-auth-jwks` is absent (dev-mode). Errors
/// on malformed JWKS JSON, missing `alg` / `kid` fields, unsupported algorithms,
/// or empty audiences (clap only enforces the `requires` constraint between flags
/// but cannot require a non-empty `Vec<String>` argument — validated here).
///
/// The returned `Arc<OAuthConfig>` is ready to pass directly to
/// [`StorageBoltHandler::with_oauth`].
fn build_bolt_oauth_config(args: &ServeArgs) -> Result<Option<Arc<OAuthConfig>>> {
    let jwks_path = match &args.bolt_auth_jwks {
        Some(p) => p,
        None => return Ok(None),
    };
    // clap `requires_all = ["bolt", "bolt_auth_issuer"]` guarantees
    // bolt_auth_issuer is Some when bolt_auth_jwks is Some.
    let issuer = args
        .bolt_auth_issuer
        .as_deref()
        .expect("clap requires: bolt_auth_issuer is Some when bolt_auth_jwks is Some");
    let config = build_oauth_config_from_parts(
        jwks_path,
        issuer,
        &args.bolt_auth_audience,
        "--bolt-auth-jwks",
        "--bolt-auth-audience",
    )?;
    tracing::info!(
        target: "arcgraph_cli::serve_bolt",
        issuer,
        audiences = ?args.bolt_auth_audience,
        "Bolt HELLO OAuth enforced (ADR-049, #761 slice 2): only bearer JWTs accepted",
    );
    Ok(Some(Arc::new(config)))
}

/// `arcgraph serve --bolt <addr>` — Bolt 5.0 (W14δ M5-13, #761 slice 2).
///
/// Loopback-default discipline (W14 retro IR L1-HIGH-4 + design-v2 §9.4
/// line 668): a non-loopback `<addr>` requires `--allow-remote-bolt-bind`
/// (landed in #761 slice 2) AND TLS; without either,
/// [`BoltServerConfig::validate`] refuses at startup — loud failure beats
/// silently-public servers.
///
/// OAuth bearer auth (ADR-049, #761 slice 2): when `--bolt-auth-jwks` is
/// configured, every Bolt HELLO is gated against the operator-staged JWKS.
/// Only `scheme="bearer"` with a valid JWT (≥1 of
/// `arcgraph.{read,write}` in scope) is admitted. Absent JWKS → dev-mode
/// (any HELLO scheme accepted; loopback-only by default).
async fn run_serve_bolt(
    addr: &str,
    args: &ServeArgs,
    gate: &ReadinessGate,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
    metrics: Option<Arc<MetricsRegistry>>,
) -> Result<()> {
    warn_unwired_opts(args);
    // W28 / ADR-183 — resolve the durable-by-default storage mode (refuse
    // to start without `--data` XOR `--in-memory`).
    let mode = BootstrapMode::from_flags(args.data.as_deref(), args.in_memory)?;
    let bind: SocketAddr = addr.parse().with_context(|| {
        format!("--bolt {addr}: not a valid SocketAddr (e.g. `127.0.0.1:7687`)")
    })?;
    // W28 / ADR-183 — `_durability` owns the WAL writer thread in durable
    // mode and MUST be held for the Bolt accept loop's lifetime.
    // W28 Feature #582 (ADR-045) — thread the metrics registry into the
    // CrudStore so Bolt-side ingest that overflows a TEL chain also fires
    // the hot-vertex counter (§10.2 line 721). The query-plan counter on the
    // Bolt query path follows the same MCP-adapter pattern (the
    // `StorageBoltHandler` constructs its own QueryEngine); it is verified
    // via the stdio raw_query path in this slice and the Bolt-side emit is a
    // tracked follow-up (see PR body).
    // ADR-216 §D-4 / #1180 — thread the operator's WAL-encryption config.
    let (backend, _durability) = bootstrap_storage_backend_with_encryption(
        &mode,
        metrics.clone(),
        &args.wal_encryption_config(),
        args.adopt_legacy_datadir,
    )?;
    // W24-OPS-α — same readiness wire as the stdio path. Durable recovery
    // (if any) completed synchronously inside `bootstrap_storage_backend`.
    gate.mark_ready("storage");
    gate.mark_ready("wal");
    gate.mark_ready("index");
    // ADR-202 §D-8 — start the community refresh scheduler (with the
    // metrics observer) BEFORE `backend`/`metrics` are consumed by the Bolt
    // handler, so the eighth §10.2 metric fires on the same registry the
    // `--metrics-http` listener scrapes. `None` without `--metrics-http`.
    let community_scheduler =
        maybe_start_community_scheduler(&backend, metrics.as_ref(), args.community_refresh_secs);
    // #765 PART-1 / #1292 PART-3 — bind the served vector provider so a Bolt
    // `RANK BY vector(n.embedding, $qv)` Cypher query runs real KNN (symmetric
    // with the stdio/HTTP graph.search + graph.raw_query wiring). Tier selected
    // by `VectorSearchTier::from_env`: HNSW (default) OR the RAM-decoupled SSD
    // DiskANN tier (ADR-195, `ARCGRAPH_VECTOR_TIER=ssd`, RSS ceiling enforced).
    let vector_provider: Arc<dyn SubstrateSearchProvider> =
        VectorSearchTier::from_env(args.data.as_deref()).build_provider(backend.clone());
    // #761 slice 2 (ADR-049) — wire Bolt HELLO OAuth when the operator stages
    // a JWKS file. `build_bolt_oauth_config` parses the RFC 7517 JWKS JSON,
    // constructs `OAuthConfig`, and returns `Some(Arc<OAuthConfig>)`. When
    // `None` (no `--bolt-auth-jwks`) the handler runs in dev-mode: any HELLO
    // scheme is accepted without signature verification (loopback-only by default
    // via the `allow_remote_bolt_bind` gate below).
    let oauth_config = build_bolt_oauth_config(args)?;
    // Warn on the dangerous combination: non-loopback + remote bind opted-in +
    // no OAuth. The bind gate + TLS requirement already block most accidental
    // exposure (validate() enforces non-loopback + TLS), but an operator who
    // sets up TLS-only without auth is running an unauthenticated public Bolt
    // endpoint — warn loudly per design-v2 §9.4 + W14 retro IR L1-HIGH-4.
    if !bind.ip().is_loopback() && args.allow_remote_bolt_bind && oauth_config.is_none() {
        tracing::warn!(
            target: "arcgraph_cli::serve_bolt",
            bind = addr,
            "serve --bolt: non-loopback bind WITHOUT OAuth auth (no --bolt-auth-jwks). \
             Any Bolt client can authenticate with any principal. \
             Configure --bolt-auth-jwks for production deployments (design-v2 §9.4)."
        );
    }
    let handler = {
        let base = StorageBoltHandler::new(backend).with_search_provider(vector_provider);
        // #1291 — enable the per-tenant memory budget with the served
        // default (1 GiB; `ARCGRAPH_TENANT_MEMORY_CAP_BYTES` overrides,
        // `0` disables) so a heavy Bolt RUN surfaces
        // `Neo.TransientError.General.OutOfMemoryError` instead of
        // OOMing the served process.
        let base = match arcgraph_cli::ops::resolve_per_tenant_memory_cap() {
            Some(cap) => base.with_per_tenant_memory_cap(cap),
            None => base,
        };
        // Chain .with_oauth() only when the operator staged a JWKS. This is the
        // ADR-049 seam: the same `Arc<OAuthConfig>` can be shared with a future
        // HTTP transport on the same binary invocation (one JWKS, two transports).
        match oauth_config {
            Some(cfg) => Arc::new(base.with_oauth(cfg)),
            None => Arc::new(base),
        }
    };
    let config = BoltServerConfig {
        bind,
        max_connections: 256,
        // #761 slice 2 — expose the loopback-default gate as a CLI flag.
        // `allow_remote_bolt_bind = false` (default) → any non-loopback bind
        // surfaces `BoltError::BindAddrForbidden` via `validate()` — loud
        // failure beats silently-public servers (design-v2 §9.4 + W14 retro
        // IR L1-HIGH-4). `true` → non-loopback allowed but TLS still required
        // by `validate()` (defense-in-depth; design-v2 §9.4 line 668).
        allow_remote_bind: args.allow_remote_bolt_bind,
        tls: None,
        // W20β-1: mTLS is opt-in via `BoltServerConfig::with_client_ca_pem`
        // when the operator stages the trust bundle. The CLI default
        // here is "no mTLS" — the loopback-default bind matches.
        client_verifier: None,
        // AHP-1 (ADR-225 §3) — install the operator-configured dispatch
        // bulkhead so a blocking RUN no longer starves reads on other
        // connections (#999).
        dispatch_bulkhead: Some(dispatch_bulkhead_from_args(args)),
    };
    // W17δ #280 — SIGTERM-during-query handler.
    //
    // `shutdown` is the shared draining future built in `run_serve`
    // (W24-OPS-α R1 fix-up M2: SIGTERM → mark_draining → sleep
    // --drain-grace-seconds → resolve).
    //
    // Bolt does not currently register queries in this cancellation
    // registry, so `cancel_all()` is normally a no-op.
    let cancel_registry = CancellationRegistry::new();
    let cancel_registry_for_shutdown = cancel_registry.clone();
    let shutdown = async move {
        shutdown.await;
        let fired = cancel_registry_for_shutdown.cancel_all();
        tracing::info!(
            target: "arcgraph_cli::serve_bolt",
            fired_count = fired,
            "shutdown signal received — fired {fired} in-flight cancellation tokens",
        );
    };

    tracing::info!(
        target: "arcgraph_cli::serve_bolt",
        bind = addr,
        oauth_enforced = handler.oauth_enforced(),
        allow_remote_bind = args.allow_remote_bolt_bind,
        "arcgraph serve --bolt starting (#761 slice 2: oauth_enforced={})", handler.oauth_enforced(),
    );

    // Thread the process `MetricsRegistry` into `serve_bolt_listener`. When
    // `Some`, the
    // Bolt accept loop records `active_connections{transport="bolt"}`.
    // The per-RUN dispatch metrics (tool-invocation counter + latency)
    // are deliberately NOT wired on the Bolt path in this slice: Bolt
    // RUN messages are not JSON-RPC-method-shaped, so their op-class
    // classification is a distinct follow-on. The #588 dispatch closure
    // targets the stdio transport (orchestrator scope) where
    // `op_class_for_method` applies directly. design-v2 §10.2.
    let stats = serve_bolt_listener(handler, config, shutdown, metrics)
        .await
        .with_context(|| format!("serve_bolt_listener bind={addr} failed"))?;

    tracing::info!(
        target: "arcgraph_cli::serve_bolt",
        accepted = stats.accepted,
        runs_succeeded = stats.runs_succeeded,
        runs_failed = stats.runs_failed,
        "arcgraph serve --bolt exiting cleanly",
    );

    // ADR-202 §D-8 — join the community scheduler thread before returning
    // (stop the metric producer before the registry is dropped). No-op when
    // no scheduler was started.
    if let Some(scheduler) = community_scheduler {
        scheduler.shutdown();
    }

    Ok(())
}

/// `arcgraph serve` dispatch on the transport flag.
///
/// # Layered initialization
///
/// 1. Build the readiness gate and register the retained components
///    (`storage`, `wal`, `index`). These are marked `Ready` by the
///    storage bootstrap helper inside each transport's run loop.
/// 2. Spawn the admin HTTP server (if `--admin-http` non-empty) in a
///    `tokio::task::spawn` alongside the MCP transport so livez and
///    readyz are available the entire process lifetime. The bind-
///    success / bind-failure outcome is observed synchronously via a
///    `oneshot` channel so the parent task can propagate
///    [`AdminHttpError::BindAddrForbidden`] up the call stack.
/// 3. Build a shared shutdown future that on SIGTERM flips the
///    readiness gate to `Draining` and sleeps
///    `--drain-grace-seconds`. This lets
///    Kubernetes Service endpoints remove the pod from the LB target
///    pool BEFORE in-flight connections terminate.
/// 4. Run the MCP transport against the shared shutdown future.
/// 5. Drain the admin HTTP task.
async fn run_serve(args: ServeArgs) -> Result<()> {
    let gate = ReadinessGate::new();
    gate.register("storage");
    gate.register("wal");
    gate.register("index");

    // Spawn the admin HTTP server task (if requested). The
    // shutdown channel fires when the MCP transport returns;
    // `with_graceful_shutdown` drains in-flight handlers cleanly.
    //
    // W24-OPS-α R1 fix-up (BLOCKER H1): pipe `--allow-remote-admin-bind`
    // through to `AdminHttpServerConfig::allow_remote_bind`. The
    // validate() path rejects non-loopback binds without the opt-in;
    // we propagate that failure synchronously here instead of letting
    // the spawned task swallow it via `tracing::error!` (M4).
    //
    // W24-OPS-α R1 fix-up (MED M4): observe bind outcome via a
    // `bind_ready` oneshot. If `validate()` rejects or `TcpListener::bind`
    // fails, we surface a clean `Err` and the binary exits without
    // pretending the admin port is up.
    let admin_handle = if args.admin_http.trim().is_empty() {
        None
    } else {
        let bind: SocketAddr = args.admin_http.parse().with_context(|| {
            format!(
                "--admin-http {}: not a valid SocketAddr (e.g. `127.0.0.1:8090`)",
                args.admin_http,
            )
        })?;
        let cfg = AdminHttpServerConfig {
            bind,
            allow_remote_bind: args.allow_remote_admin_bind,
        };
        // Validate synchronously — `BindAddrForbidden` is a user-input
        // error and surfacing it here gives the operator an immediate
        // actionable failure (matching the loud-failure discipline at
        // BoltServerConfig::validate).
        cfg.validate()
            .with_context(|| format!("admin HTTP bind {} rejected", cfg.bind))?;
        let gate_clone = gate.clone();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let shutdown = async move {
            let _ = rx.await;
        };
        let handle = tokio::spawn(async move {
            if let Err(e) = serve_admin_http(cfg, gate_clone, shutdown).await {
                tracing::error!(
                    target: "arcgraph_cli::serve",
                    error = %e,
                    "admin HTTP server exited with error"
                );
            }
        });
        Some((handle, tx))
    };

    // W28 #588 / OBS-1 — instantiate the process `MetricsRegistry` + spawn
    // the Prometheus `/metrics` scrape listener UNLESS metrics are disabled.
    // OBS-1 flips the default ON (`--metrics-http` default is
    // `127.0.0.1:9090`), so this fires by default; an operator disables it
    // by passing an explicit empty value (`--metrics-http ""`), which trips
    // the `trim().is_empty()` gate → `None` (no registry, no listener, zero
    // overhead — the pre-OBS-1 opt-out path). The SAME registry Arc is
    // threaded into the stdio/bolt transports below, so their per-dispatch +
    // connection metrics record into the registry this listener scrapes.
    // SEPARATE axum server from the admin port per ADR-093 §"Why a separate admin port";
    // Tokio background per design-v2 §4.1 (full justification on
    // `ops::metrics_http`). Two failure modes: a non-loopback bind without
    // `--allow-remote-metrics-bind` is a config error → `validate()` fails
    // LOUD synchronously here (the default loopback:9090 passes `validate()`
    // by construction, so the on-by-default posture does NOT weaken the
    // W14 loopback-default invariant); a runtime bind failure (EADDRINUSE)
    // inside the spawned task is LOGGED and the MCP server keeps serving
    // (ADR-093 §Decision item 2 — observability must not cascade into
    // unavailability).
    let metrics_registry: Option<Arc<MetricsRegistry>> = if args.metrics_http.trim().is_empty() {
        None
    } else {
        Some(MetricsRegistry::shared().context("metrics registry init failed")?)
    };
    let metrics_handle = match metrics_registry.clone() {
        None => None,
        Some(registry) => {
            let bind: SocketAddr = args.metrics_http.parse().with_context(|| {
                format!(
                    "--metrics-http {}: not a valid SocketAddr (e.g. `127.0.0.1:9090`)",
                    args.metrics_http,
                )
            })?;
            let cfg = MetricsHttpServerConfig {
                bind,
                allow_remote_bind: args.allow_remote_metrics_bind,
            };
            // Validate synchronously — non-loopback-without-opt-in is a
            // user-input error; surfacing it here (not inside the task)
            // gives the operator an immediate actionable failure (mirror
            // of the admin-bind discipline above).
            cfg.validate()
                .with_context(|| format!("metrics HTTP bind {} rejected", cfg.bind))?;
            let (tx, rx) = tokio::sync::oneshot::channel::<()>();
            let shutdown = async move {
                let _ = rx.await;
            };
            let handle = tokio::spawn(async move {
                if let Err(e) = serve_metrics_http(cfg, registry, shutdown).await {
                    tracing::error!(
                        target: "arcgraph_cli::serve",
                        error = %e,
                        "metrics HTTP server exited with error \
                         (observability degraded; MCP server unaffected)"
                    );
                }
            });
            Some((handle, tx))
        }
    };

    // W24-OPS-α R1 fix-up (MED M2): build a shared draining-shutdown
    // future. On SIGTERM:
    //   1. Flip the readiness gate to `Draining` (so /readyz starts
    //      returning 503 immediately; K8s Service endpoints remove
    //      the pod from the LB target pool on the next reconcile).
    //   2. Sleep `--drain-grace-seconds` so kube-proxy's iptables
    //      reconcile picks up the endpoint removal BEFORE we resolve
    //      and the transport begins its own teardown.
    //   3. Resolve — the transport sees the shutdown signal.
    let drain_grace = Duration::from_secs(args.drain_grace_seconds);
    let shutdown_future = {
        let gate_for_drain = gate.clone();
        async move {
            shutdown_on_term().await;
            // Flip every registered component to Draining so /readyz
            // surfaces the drain to load-balancers. Per the SIGTERM
            // contract the components are still serving in-flight
            // requests until cancel_all fires below; "Draining" is
            // the K8s-visible signal that drives endpoint removal.
            for name in ["storage", "wal", "index"] {
                gate_for_drain.mark_draining(name);
            }
            tracing::info!(
                target: "arcgraph_cli::serve",
                drain_grace_ms = drain_grace.as_millis() as u64,
                "SIGTERM observed — readiness gate marked draining; sleeping drain grace window",
            );
            if !drain_grace.is_zero() {
                tokio::time::sleep(drain_grace).await;
            }
        }
    };

    let outcome = if let Some(addr) = args.http.clone() {
        // #761 slice 1 — live HTTPS MCP transport. Shares the same
        // draining-shutdown future + metrics registry as stdio/bolt.
        run_serve_http(
            &addr,
            &args,
            &gate,
            shutdown_future,
            metrics_registry.clone(),
        )
        .await
    } else if let Some(addr) = args.bolt.clone() {
        run_serve_bolt(
            &addr,
            &args,
            &gate,
            shutdown_future,
            metrics_registry.clone(),
        )
        .await
    } else {
        // stdio_mcp == true OR no transport flag set (default).
        run_serve_stdio(&args, &gate, shutdown_future, metrics_registry.clone()).await
    };

    // Drain the admin HTTP server task.
    if let Some((handle, tx)) = admin_handle {
        let _ = tx.send(());
        let _ = handle.await;
    }

    // W28 #588 — drain the metrics HTTP server task (same shutdown
    // discipline as the admin task above; the MCP transport has already
    // returned, so signal the listener to stop + await its teardown).
    if let Some((handle, tx)) = metrics_handle {
        let _ = tx.send(());
        let _ = handle.await;
    }

    outcome
}

/// Warn that the reserved `--config` flag is accepted but ignored.
fn warn_unwired_opts(args: &ServeArgs) {
    if let Some(cfg) = &args.config {
        eprintln!(
            "warning: --config {} is accepted but ignored in v0.1.0-beta; use explicit CLI flags",
            cfg.display(),
        );
    }
}

/// `arcgraph check` — bounded integrity check.
fn run_check(args: CheckArgs) -> Result<()> {
    println!("arcgraph check (v0.1.0-beta bounded integrity check)");

    let status = match args.data {
        Some(dir) => {
            if !dir.exists() {
                bail!(
                    "arcgraph check --data {}: directory does not exist",
                    dir.display()
                );
            }
            if !dir.is_dir() {
                bail!(
                    "arcgraph check --data {}: path exists but is not a directory",
                    dir.display()
                );
            }
            println!("  data-dir: {} (exists, readable)", dir.display());
            // Recognize both generation-backed loader stores (`CURRENT`) and
            // direct page/WAL stores produced by `serve` or cold restore.
            // Bootstrap over a truly empty directory would create state, so
            // empty directories retain the non-mutating path check.
            if has_committed_store_state(&dir) {
                check_served_store(&dir)?;
                "ok (committed store cold-opened; bounded record/property sample readable)"
            } else {
                "ok (directory exists; no committed store found)"
            }
        }
        None => {
            println!("  data-dir: <none> (in-memory empty-tenant catalog)");
            "ok (in-memory empty-tenant catalog)"
        }
    };

    println!(
        "  tenant:   {} (TenantId::DEFAULT)",
        TenantId::DEFAULT.raw()
    );
    println!("  status:   {status}");
    Ok(())
}

fn has_committed_store_state(dir: &Path) -> bool {
    dir.join("CURRENT").is_file()
        || (dir.join("VERSION").is_file()
            && (dir.join("pages.db").is_file() || dir.join("wal").is_dir()))
}

/// Cold-open a committed generation and hydrate a bounded sample of
/// record property bags per catalog tenant (production bootstrap +
/// production read path). Budget (PD#5): ≤ `CHECK_SAMPLE_IDS` node +
/// relationship reads per tenant, each O(1) — the check stays O(tenants)
/// regardless of store size. This is not an exhaustive record scan.
fn check_served_store(dir: &std::path::Path) -> Result<()> {
    const CHECK_SAMPLE_IDS: u64 = 64;
    let (backend, guard) =
        arcgraph_cli::bootstrap::bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: dir.to_path_buf(),
        })
        .with_context(|| format!("cold-open committed store at {}", dir.display()))?;
    // Tenant census = catalog registry ∪ the committed generation's
    // MANIFEST `tenant_census` (the census that travels WITH the
    // generation — `manifest.rs`). Post-#1513 (M5-D1b) cold open
    // registers the MANIFEST census into the served catalog through the
    // production registration path, so the two sides normally coincide;
    // the union is kept as a CROSS-CHECK — a census tenant missing from
    // the catalog now fails LOUD at the per-tenant `route` below (the
    // #1513 condition), instead of being silently sampled through a
    // shared handle.
    let mut tenants: std::collections::BTreeSet<u64> = backend
        .router()
        .catalog()
        .list_tenants()
        .iter()
        .map(|entry| entry.tenant_id.raw())
        .collect();
    if let Some(generation) = arcgraph_cli::data_dir_migration::current_generation(dir)
        .context("resolve CURRENT generation during check")?
        && let Some(manifest) = arcgraph_storage::read_data_dir_manifest(&generation)
            .context("read generation MANIFEST during check")?
        && let Some(census) = manifest.tenant_census
    {
        tenants.extend(census);
    }
    for tenant in tenants {
        let tenant = TenantId::new(tenant);
        // #1513 (M5-D1b): the PER-TENANT production dispatch — the exact
        // route(tenant, PartitionId::ZERO) shape the MCP/Bolt adapters
        // issue. A servable-but-unregistered tenant fails here with
        // `UnknownTenant` (the checker must surface that, never mask it
        // behind a shared route(DEFAULT) handle).
        let routed = backend
            .router()
            .route(tenant, arcgraph_core::PartitionId::ZERO)
            .with_context(|| {
                format!(
                    "route(tenant {}, ZERO) during check — servable tenants must be \
                     catalog-registered at cold open (#1513)",
                    tenant.raw()
                )
            })?;
        let reader = backend.txn_manager().begin(tenant);
        let mut nodes = 0_u64;
        let mut rels = 0_u64;
        let mut bags = 0_u64;
        let mut bag_bytes = 0_u64;
        let mut checksum = 0_u32;
        for id in 1..=CHECK_SAMPLE_IDS {
            if let Some(record) = arcgraph_storage::crud::read_node_with_store(
                routed.crud(),
                &reader,
                arcgraph_core::NodeId::new(id),
            )
            .with_context(|| format!("check read node {id} tenant {}", tenant.raw()))?
            {
                nodes += 1;
                if let Some(blob_ref) =
                    arcgraph_storage::property::BlobRef::decode(record.property_ref)
                {
                    let bag = routed
                        .crud()
                        .blob_store()
                        .get_bag(tenant, blob_ref)
                        .with_context(|| {
                            format!("check hydrate node {id} bag tenant {}", tenant.raw())
                        })?;
                    bags += 1;
                    bag_bytes += bag.len() as u64;
                    checksum = crc32c::crc32c_append(checksum, &bag);
                }
            }
            if let Some(record) = arcgraph_storage::crud::read_rel_with_store(
                routed.crud(),
                &reader,
                arcgraph_core::RelId::new(id),
            )
            .with_context(|| format!("check read relationship {id} tenant {}", tenant.raw()))?
            {
                rels += 1;
                if let Some(blob_ref) =
                    arcgraph_storage::property::BlobRef::decode(record.property_ref)
                {
                    let bag = routed
                        .crud()
                        .blob_store()
                        .get_bag(tenant, blob_ref)
                        .with_context(|| {
                            format!(
                                "check hydrate relationship {id} bag tenant {}",
                                tenant.raw()
                            )
                        })?;
                    bags += 1;
                    bag_bytes += bag.len() as u64;
                    checksum = crc32c::crc32c_append(checksum, &bag);
                }
            }
        }
        println!(
            "  served:   tenant={} sampled_nodes={nodes} sampled_rels={rels} \
             hydrated_bags={bags} bag_bytes={bag_bytes} bag_crc32c={checksum:#010x} \
             (first {CHECK_SAMPLE_IDS} ids per class)",
            tenant.raw(),
        );
        reader.abort();
        drop(routed);
    }
    drop(backend);
    drop(guard);
    Ok(())
}

/// `arcgraph dump` — per-tenant graph export.
///
/// The storage-rooted export body is not implemented. A backup/export tool
/// must not report success while emitting a known-incomplete empty artifact,
/// so `--data` returns a non-zero exit instead of producing a false backup.
///
/// Behavior:
/// - **`--data <dir>` set** — refuse with an actionable error.
/// - **`--data` unset** — open no store and emit a clearly labelled empty
///   envelope.
fn run_dump(args: DumpArgs) -> Result<()> {
    if let Some(dir) = &args.data {
        if !dir.exists() {
            bail!(
                "arcgraph dump --data {}: directory does not exist",
                dir.display()
            );
        }
        bail!(
            "arcgraph dump --data {}: storage-rooted logical export is not \
             implemented in v0.1.0-beta; refusing to emit an incomplete \
             artifact. Use `arcgraph backup create --data {} --dest DIR` for \
             a verified cold backup.",
            dir.display(),
            dir.display(),
        );
    }

    eprintln!(
        "arcgraph dump: WARNING — no --data store was opened; output is empty \
         by construction and is not a storage-rooted export or backup"
    );

    let tenant = TenantId::new(args.tenant);
    match args.format {
        DumpFormat::Json => emit_dump_json(tenant, &args)?,
        DumpFormat::Toon => emit_dump_toon(tenant, &args)?,
        DumpFormat::Cypher => emit_dump_cypher(tenant, &args)?,
    }
    Ok(())
}

fn emit_dump_json(tenant: TenantId, args: &DumpArgs) -> Result<()> {
    let env = serde_json::json!({
        "format": "json",
        "tenant_id": tenant.raw(),
        "data_dir": args.data.as_ref().map(|p| p.display().to_string()),
        "nodes": [],
        "relationships": [],
        "note": "no --data store was opened; empty envelope only, not a storage-rooted export or backup",
    });
    let body = serde_json::to_string_pretty(&env).context("serialize dump envelope")?;
    println!("{body}");
    Ok(())
}

fn emit_dump_toon(tenant: TenantId, args: &DumpArgs) -> Result<()> {
    // The full TOON serializer targets tabular result sets, so this command
    // writes its empty envelope directly.
    println!("# arcgraph dump format=toon");
    println!("tenant_id: {}", tenant.raw());
    if let Some(dir) = &args.data {
        println!("data_dir: {}", dir.display());
    }
    println!("nodes: []");
    println!("relationships: []");
    println!(
        "# note: no --data store was opened; empty envelope only, not a storage-rooted export or backup"
    );
    Ok(())
}

fn emit_dump_cypher(tenant: TenantId, args: &DumpArgs) -> Result<()> {
    // Emit a header and zero CREATE statements because no store was opened.
    println!("// arcgraph dump format=cypher");
    println!("// tenant_id: {}", tenant.raw());
    if let Some(dir) = &args.data {
        println!("// data_dir:  {}", dir.display());
    }
    println!("// nodes:      0");
    println!("// relationships: 0");
    println!(
        "// note: no --data store was opened; empty envelope only, not a storage-rooted export or backup"
    );
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// `arcgraph health` — distroless `HEALTHCHECK` probe
// ─────────────────────────────────────────────────────────────────────

/// Parsed `--addr` URL.
///
/// We intentionally avoid pulling in a `url`-crate dep just for this:
/// the probe accepts only `http://host[:port][/path]`, which a few lines
/// of `str::split` handle without taking on a parser surface that would
/// then need feature-gating + license-verification per Prime Directive
/// #1.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HealthProbe {
    host: String,
    port: u16,
    path: String,
}

impl HealthProbe {
    fn parse(addr: &str) -> Result<Self> {
        // Reject HTTPS explicitly instead of failing later as a malformed
        // plain-HTTP status exchange.
        if addr.starts_with("https://") {
            bail!(
                "--addr {addr}: `arcgraph health` accepts only http:// URLs in \
                 v0.1.0-beta and cannot probe the HTTPS-only MCP listener; use \
                 a TLS-aware client with the server certificate"
            );
        }
        let rest = addr
            .strip_prefix("http://")
            .with_context(|| format!("--addr {addr}: must start with http://"))?;
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        if authority.is_empty() {
            bail!("--addr {addr}: empty authority");
        }
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => {
                let port: u16 = p
                    .parse()
                    .with_context(|| format!("--addr {addr}: invalid port `{p}`"))?;
                (h.to_string(), port)
            }
            None => (authority.to_string(), 80u16),
        };
        if host.is_empty() {
            bail!("--addr {addr}: empty host");
        }
        let path = if path.is_empty() {
            "/".to_string()
        } else {
            path.to_string()
        };
        Ok(HealthProbe { host, port, path })
    }

    fn connect_addr(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }
}

/// Issue one HTTP/1.1 GET against `probe` and return the response's
/// numeric status code.
///
/// All of connect / write / read share the same `timeout` budget — a
/// hung peer (TCP accept but no bytes) trips the read leg; a peer
/// behind iptables-DROP trips the connect leg. Per the W15β `Dockerfile`
/// `HEALTHCHECK --timeout=5s`, 2s here leaves Docker headroom for
/// process spawn + the `arcgraph` binary's own startup cost inside
/// the probe poll window.
async fn http_get_status(probe: &HealthProbe, timeout: Duration) -> Result<u16> {
    let connect_addr = probe.connect_addr();

    // ── 1. connect ──────────────────────────────────────────────────
    let mut stream = tokio::time::timeout(timeout, tokio::net::TcpStream::connect(&connect_addr))
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "connect timeout after {}ms ({connect_addr})",
                timeout.as_millis()
            )
        })?
        .with_context(|| format!("connect {connect_addr}"))?;

    // ── 2. write request ───────────────────────────────────────────
    let request = format!(
        "GET {} HTTP/1.1\r\n\
         Host: {}\r\n\
         Connection: close\r\n\
         User-Agent: arcgraph-health/1.0\r\n\
         Accept: */*\r\n\
         \r\n",
        probe.path, probe.host,
    );
    tokio::time::timeout(timeout, async {
        use tokio::io::AsyncWriteExt;
        stream.write_all(request.as_bytes()).await?;
        stream.flush().await?;
        Ok::<_, std::io::Error>(())
    })
    .await
    .map_err(|_| anyhow::anyhow!("write timeout after {}ms", timeout.as_millis()))?
    .context("write request")?;

    // ── 3. read until end-of-headers or EOF or 4KiB cap ─────────────
    let mut buf: Vec<u8> = Vec::with_capacity(512);
    const MAX_HEADER_BYTES: usize = 4096;
    tokio::time::timeout(timeout, async {
        use tokio::io::AsyncReadExt;
        let mut tmp = [0u8; 256];
        loop {
            let n = stream.read(&mut tmp).await?;
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                break;
            }
            if buf.len() >= MAX_HEADER_BYTES {
                break;
            }
        }
        Ok::<_, std::io::Error>(())
    })
    .await
    .map_err(|_| anyhow::anyhow!("read timeout after {}ms", timeout.as_millis()))?
    .context("read response")?;

    if buf.is_empty() {
        bail!("server closed connection without writing any bytes");
    }

    // ── 4. parse status line ────────────────────────────────────────
    parse_status_line(&buf)
}

/// Extract the numeric status code from a buffered HTTP response.
///
/// Split into its own function for unit-testing without spinning a TCP
/// listener.
fn parse_status_line(buf: &[u8]) -> Result<u16> {
    let first_line_end = buf
        .iter()
        .position(|&b| b == b'\n')
        .unwrap_or(buf.len())
        .min(buf.len());
    let line_bytes = &buf[..first_line_end];
    let line = std::str::from_utf8(line_bytes).context("non-UTF-8 status line")?;
    let mut parts = line.split_ascii_whitespace();
    let version = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("empty status line"))?;
    if !version.starts_with("HTTP/") {
        bail!("unexpected status line: {line:?}");
    }
    let code_str = parts
        .next()
        .ok_or_else(|| anyhow::anyhow!("status line missing code: {line:?}"))?;
    let code: u16 = code_str
        .parse()
        .with_context(|| format!("invalid status code `{code_str}` in {line:?}"))?;
    Ok(code)
}

/// `arcgraph health` body — exit 0 on 2xx, 1 (with a single-line
/// `arcgraph health: <reason>` stderr message) otherwise.
async fn run_health(args: HealthArgs) -> Result<()> {
    let probe = match HealthProbe::parse(&args.addr) {
        Ok(p) => p,
        Err(e) => {
            // Single-line stderr per the spawn brief — no anyhow `Error:`
            // prefix because the operator expected output is literal
            // `arcgraph health: <reason>`.
            eprintln!("arcgraph health: {e}");
            std::process::exit(1);
        }
    };
    let timeout = Duration::from_millis(args.timeout_ms);
    match http_get_status(&probe, timeout).await {
        Ok(status) if (200..300).contains(&status) => {
            tracing::debug!(
                target: "arcgraph_cli::health",
                addr = %args.addr,
                status,
                "arcgraph health: probe ok"
            );
            Ok(())
        }
        Ok(status) => {
            eprintln!("arcgraph health: HTTP {status}");
            std::process::exit(1);
        }
        Err(e) => {
            // anyhow's Display already chains contexts with `: ` separators.
            eprintln!("arcgraph health: {e}");
            std::process::exit(1);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Entry point
// ─────────────────────────────────────────────────────────────────────

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    // W24-OPS-α — `tracing_subscriber::registry` + optional
    // OTLP-gRPC exporter init. The returned guard MUST live for the
    // process lifetime so the OTLP batch processor flushes pending
    // spans before exit. Per ADR-093 §Decision item 3, OTLP build
    // failures degrade gracefully to stderr-only (do NOT abort).
    let _tracing_guard: TracingGuard = init_tracing(TracingConfig::from_env());

    let cli = Cli::parse();
    match cli.command {
        Command::Serve(args) => run_serve(*args).await,
        Command::Check(args) => run_check(args),
        Command::Dump(args) => run_dump(args),
        Command::Health(args) => run_health(args).await,
        Command::Migrate(args) => run_migrate(args),
        Command::Backup(args) => run_backup(args),
        Command::Load(args) => run_load(args),
    }
}

/// M5 leg-(c) — `arcgraph load` (amendment §2.2). The loader owns the whole
/// protocol: DataDirLock-first, virgin/resumable precondition, bounded build
/// inside its own generation namespace, ONE `CURRENT` commit object, VERSION
/// stamp LAST, restart-matrix resume on rerun.
fn run_load(args: LoadArgs) -> Result<()> {
    let tenant = TenantId::new(args.tenant);
    let format = match args.format {
        LoadFormatArg::Native => arcgraph_cli::m5_load::LoadFormat::Native,
    };
    let outcome = arcgraph_cli::m5_load::load_data_dir(&args.input, format, &args.data_dir, tenant)
        .with_context(|| {
            format!(
                "load {:?} input {} for tenant {}",
                args.format,
                args.input.display(),
                tenant.raw()
            )
        })?;
    match outcome {
        arcgraph_cli::m5_load::LoadOutcome::Loaded(report) => {
            println!(
                "arcgraph load: tenant={} records={} nodes={} relationships={} prop_pages={} \
                 chained_bags={} out_tel_entries={} in_tel_entries={} resumed={} committed={}",
                tenant.raw(),
                report.records,
                report.nodes,
                report.relationships,
                report.prop_pages,
                report.chained_bags,
                report.out_tel_entries,
                report.in_tel_entries,
                report.resumed,
                args.data_dir.display(),
            );
        }
        arcgraph_cli::m5_load::LoadOutcome::AlreadyLoaded { tenant_census } => {
            println!(
                "arcgraph load: already loaded (no-op); data_dir={} tenant_census={tenant_census:?}",
                args.data_dir.display(),
            );
        }
    }
    Ok(())
}

/// `arcgraph migrate from-neo4j-cypher <path>` / `from-neo4j-csv ...`.
/// `arcgraph backup ...` — ADR-204 cold backup + verified restore.
#[derive(Debug, Args)]
pub struct BackupArgs {
    /// Backup subcommand.
    #[command(subcommand)]
    pub command: BackupCommand,
}

/// ADR-204 D-3 — the two operator verbs. COLD-only at v1 (the verbs
/// take the same exclusive data-dir LOCK the server takes); online
/// backup is #405 stage-2.
#[derive(Debug, Subcommand)]
pub enum BackupCommand {
    /// Create a cold backup (requires the server to be stopped).
    Create {
        /// The durable data directory to back up.
        #[arg(long)]
        data: PathBuf,
        /// Destination directory (created; must be empty if present).
        #[arg(long)]
        dest: PathBuf,
    },
    /// Verify a backup and restore it into a FRESH data directory.
    Restore {
        /// The backup directory (containing BACKUP_MANIFEST.json).
        #[arg(long)]
        from: PathBuf,
        /// The new data directory (must not contain store state).
        #[arg(long)]
        data: PathBuf,
    },
}

/// ADR-204 — dispatch `arcgraph backup create|restore`.
fn run_backup(args: BackupArgs) -> Result<()> {
    use arcgraph_cli::ops::backup::{backup_create, backup_restore};
    match args.command {
        BackupCommand::Create { data, dest } => {
            let manifest = backup_create(&data, &dest).with_context(|| {
                format!("backup create {} -> {}", data.display(), dest.display())
            })?;
            println!(
                "backup created: {} file(s) at {} (manifest format v{})",
                manifest.files.len(),
                dest.display(),
                manifest.format_version,
            );
            Ok(())
        }
        BackupCommand::Restore { from, data } => {
            let manifest = backup_restore(&from, &data).with_context(|| {
                format!("backup restore {} -> {}", from.display(), data.display())
            })?;
            println!(
                "backup verified + restored: {} file(s) into {} — start the server                  normally; boot recovery replays the WAL",
                manifest.files.len(),
                data.display(),
            );
            Ok(())
        }
    }
}

fn run_migrate(args: MigrateArgs) -> Result<()> {
    if let MigrateSource::UpgradeDataDir { data_dir } = &args.source {
        let outcome =
            arcgraph_cli::data_dir_migration::upgrade_data_dir(data_dir).with_context(|| {
                format!(
                    "upgrade data dir {} by one offline generation",
                    data_dir.display()
                )
            })?;
        println!(
            "arcgraph migrate upgrade-data-dir --data-dir {}: {outcome:?}",
            data_dir.display()
        );
        return Ok(());
    }
    let tenant = TenantId::new(args.tenant);
    let batches = match &args.source {
        MigrateSource::UpgradeDataDir { .. } => unreachable!("handled above"),
        MigrateSource::FromNeo4jCypher { path } => {
            println!("arcgraph migrate from-neo4j-cypher {}", path.display());
            parse_cypher_export(path)
                .with_context(|| format!("parse_cypher_export({})", path.display()))?
        }
        MigrateSource::FromNeo4jCsv { nodes, rels } => {
            println!(
                "arcgraph migrate from-neo4j-csv --nodes {} --rels {}",
                nodes.display(),
                rels.display()
            );
            parse_csv_export(nodes, rels).with_context(|| {
                format!(
                    "parse_csv_export(nodes={}, rels={})",
                    nodes.display(),
                    rels.display()
                )
            })?
        }
    };

    // Neo4j migration commands ingest into an ephemeral in-memory store,
    // report counts, and exit. They do not create a durable data directory.
    // W28 #582: `migrate` has no `--metrics-http` listener — no sink (`None`).
    let (backend, _durability) = bootstrap_storage_backend(&BootstrapMode::InMemory, None)?;
    let provider = StorageIngestProvider::new(backend);
    let mut total_inserted: u64 = 0;
    let mut total_failed: u64 = 0;
    for (idx, batch) in batches.into_iter().enumerate() {
        let summary = provider
            .ingest(tenant, batch)
            .with_context(|| format!("ingest batch {idx} failed"))?;
        total_inserted += summary.inserted_count;
        total_failed += summary.failed_count;
    }
    println!(
        "arcgraph migrate: tenant={} inserted={} failed={}",
        tenant.raw(),
        total_inserted,
        total_failed,
    );
    if total_failed > 0 {
        bail!("migrate completed with {total_failed} per-record failures");
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// Unit tests — clap parsers + each subcommand surface.
//
// The full subprocess test set (run binary, verify stdout) lives in
// `crates/arcgraph-cli/tests/arcgraph_cli_subprocess.rs`.
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrate_upgrade_verb_cli_pinned() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "migrate",
            "upgrade-data-dir",
            "--data-dir",
            "/var/lib/arcgraph",
        ])
        .expect("parse offline M3 migration command");
        let Command::Migrate(args) = cli.command else {
            panic!("expected migrate command");
        };
        assert!(matches!(
            args.source,
            MigrateSource::UpgradeDataDir { data_dir }
                if data_dir == Path::new("/var/lib/arcgraph")
        ));
    }
    use clap::CommandFactory;

    #[test]
    fn clap_definition_is_valid() {
        // `debug_assert` inside clap's derive expansion fires if the
        // structure is misconfigured (e.g., conflicting flags
        // referencing missing fields). Calling `debug_assert` against
        // the CommandFactory is the canonical clap smoke test.
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_serve_stdio_mcp_default() {
        let cli = Cli::try_parse_from(["arcgraph", "serve", "--stdio-mcp"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert!(args.stdio_mcp);
        assert!(args.http.is_none());
        assert!(args.bolt.is_none());
    }

    #[test]
    fn parses_serve_no_transport_defaults_to_stdio() {
        // Per the binary's transport-selection rule (run_serve), if
        // no transport flag is set we default to stdio. The clap
        // surface accepts this: `serve` alone parses.
        let cli = Cli::try_parse_from(["arcgraph", "serve"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert!(!args.stdio_mcp);
        assert!(args.http.is_none());
        assert!(args.bolt.is_none());
    }

    #[test]
    fn parses_serve_http_addr() {
        // #761 slice 1 — `--http` now `requires` `--tls-cert` + `--tls-key`.
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/c",
            "--tls-key",
            "/k",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.http.as_deref(), Some("127.0.0.1:8443"));
    }

    #[test]
    fn parses_serve_bolt_addr() {
        let cli =
            Cli::try_parse_from(["arcgraph", "serve", "--bolt", "127.0.0.1:7687"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.bolt.as_deref(), Some("127.0.0.1:7687"));
    }

    // ─────────────────────────────────────────────────────────────────
    // `build_bolt_oauth_config` — the flag→`OAuthConfig` wire (#761 slice 2,
    // ADR-049). R1 #887 NIT-1: the OAuth seam tests all live in
    // `arcgraph-mcp`; before these three, the CLI's `--bolt-auth-*` flags →
    // `Option<Arc<OAuthConfig>>` mapping (the actual deliverable of #887)
    // shipped with NO automated regression guard. `arcgraph.rs:1234`'s
    // `DecodingKey::from_jwk` is the SOLE `from_jwk`/JWKS-JSON consumer in the
    // tree — every sibling OAuth test builds its key via `from_ec_pem` (PEM),
    // so the JSON-JWKS path below is exercised here and only here.
    // ─────────────────────────────────────────────────────────────────

    /// No `--bolt-auth-jwks` ⇒ dev-mode: the wire returns `Ok(None)` (OAuth
    /// not enforced). Guards the early `return Ok(None)` at the top of
    /// `build_bolt_oauth_config`.
    #[test]
    fn build_bolt_oauth_config_none_without_jwks() {
        let cli =
            Cli::try_parse_from(["arcgraph", "serve", "--bolt", "127.0.0.1:7687"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        let cfg = build_bolt_oauth_config(&args).expect("dev-mode build must not error");
        assert!(
            cfg.is_none(),
            "no --bolt-auth-jwks ⇒ dev-mode (expected Ok(None)); got Some",
        );
    }

    /// End-to-end CLI wire for the OAuth-enforced path: a real RFC 7517 §A.1
    /// P-256 JWK Set staged on disk ⇒ `Ok(Some(OAuthConfig))` carrying the
    /// issuer, audience, and exactly one decoding key. This is the sole
    /// regression guard for `build_bolt_oauth_config`'s full pipeline:
    /// `read_to_string → serde_json::JwkSet → DecodingKey::from_jwk →
    /// JsonWebKeySet::new → OAuthConfig::new` (arcgraph.rs:1176-1248).
    ///
    /// RED-on-revert (load-bearing pin): if `build_bolt_oauth_config` were
    /// severed to `Ok(None)` unconditionally — the flag→config wire cut — the
    /// `.expect(...Some...)` below would panic and this test FAILS. It is thus
    /// a true guard on the wire, not just a smoke test of the helper.
    ///
    /// Key material is the canonical RFC 7517 Appendix A.1 EC P-256 public key
    /// — a real, on-curve point, not invented bytes. `alg`/`use`/`kid` are
    /// added because `build_bolt_oauth_config` *requires* an explicit `alg`
    /// (alg-confusion guard, arcgraph.rs:1213-1222) and echoes `kid` into the
    /// `JsonWebKey`.
    #[test]
    fn build_bolt_oauth_config_some_from_valid_jwks() {
        use std::io::Write as _;

        const JWKS_JSON: &str = r#"{
  "keys": [
    {
      "kty": "EC",
      "crv": "P-256",
      "x": "MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
      "y": "4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
      "use": "sig",
      "alg": "ES256",
      "kid": "test-key-1"
    }
  ]
}"#;
        // `NamedTempFile` deletes on Drop; it stays in scope through the
        // `build_bolt_oauth_config` call below so the file exists during the read.
        let mut jwks_file = tempfile::NamedTempFile::new().expect("create temp JWKS file");
        jwks_file
            .write_all(JWKS_JSON.as_bytes())
            .expect("write JWKS JSON");
        jwks_file.flush().expect("flush JWKS file");
        let jwks_path = jwks_file.path().to_str().expect("temp path is valid UTF-8");

        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--bolt",
            "127.0.0.1:7687",
            "--bolt-auth-jwks",
            jwks_path,
            "--bolt-auth-issuer",
            "https://issuer.test",
            "--bolt-auth-audience",
            "arcgraph",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };

        let cfg = build_bolt_oauth_config(&args)
            .expect("valid JWKS file ⇒ build must succeed")
            .expect("valid JWKS file ⇒ Some(OAuthConfig), not dev-mode None");

        // Stronger oracle than `is_some()`: prove issuer, audience, and the
        // single P-256 key all threaded through the JSON→from_jwk pipeline.
        assert_eq!(cfg.issuer, "https://issuer.test");
        assert_eq!(cfg.audiences, vec!["arcgraph".to_string()]);
        assert_eq!(
            cfg.jwks.keys().count(),
            1,
            "the RFC-7517 P-256 JWK must round-trip through from_jwk into exactly one decoding key",
        );
    }

    /// Runtime guard clap cannot express: `--bolt-auth-jwks` present but ZERO
    /// `--bolt-auth-audience`. clap's `requires` chain forces `--bolt` +
    /// `--bolt-auth-issuer` alongside `--bolt-auth-jwks`, but a `Vec<String>`
    /// arg parses empty when no `--bolt-auth-audience` is given — so the "at
    /// least one audience" rule lives in `build_bolt_oauth_config` as a
    /// `bail!` (arcgraph.rs:1170-1175). The pre-`build` assertion pins that
    /// clap really did permit the empty `Vec` (proving the runtime check is
    /// not dead code). The empty-audience check precedes the JWKS file read,
    /// so the staged path need not exist.
    #[test]
    fn build_bolt_oauth_config_err_on_empty_audience() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--bolt",
            "127.0.0.1:7687",
            "--bolt-auth-jwks",
            "/nonexistent/jwks.json",
            "--bolt-auth-issuer",
            "https://issuer.test",
            // deliberately NO --bolt-auth-audience: clap permits the empty Vec.
        ])
        .expect("clap permits jwks+issuer with no audience (Vec may parse empty)");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert!(
            args.bolt_auth_audience.is_empty(),
            "precondition: clap parsed an empty --bolt-auth-audience Vec",
        );

        let err = build_bolt_oauth_config(&args)
            .expect_err("JWKS set with zero --bolt-auth-audience must be a runtime error");
        let msg = err.to_string();
        assert!(
            msg.contains("at least one --bolt-auth-audience"),
            "expected the empty-audience bail, got: {msg}",
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // `build_http_oauth_config` — the flag→`HttpServerConfig::with_oauth`
    // wire (#761 slice 3, ADR-044/049). Mirrors the #887 Bolt helper tests;
    // the subprocess e2e in `serve_http_tls_e2e.rs` proves this config reaches
    // the live HTTPS `POST /mcp` enforcement gate.
    // ─────────────────────────────────────────────────────────────────

    /// No `--http-auth-jwks` ⇒ dev-mode: the wire returns `Ok(None)` (OAuth
    /// not enforced). Guards the early `return Ok(None)` at the top of
    /// `build_http_oauth_config`.
    #[test]
    fn build_http_oauth_config_none_without_jwks() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/tmp/server.crt",
            "--tls-key",
            "/tmp/server.key",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        let cfg = build_http_oauth_config(&args).expect("dev-mode build must not error");
        assert!(
            cfg.is_none(),
            "no --http-auth-jwks => dev-mode (expected Ok(None)); got Some",
        );
    }

    /// End-to-end CLI wire for the HTTP OAuth-enforced helper path: a real
    /// RFC 7517 §A.1 P-256 JWK Set staged on disk ⇒
    /// `Ok(Some(OAuthConfig))` carrying issuer, audience, and decoding key.
    ///
    /// RED-on-revert: if `build_http_oauth_config` is severed to `Ok(None)`,
    /// the `.expect(...Some...)` below panics and this test fails.
    #[test]
    fn build_http_oauth_config_some_from_valid_jwks() {
        use std::io::Write as _;

        const JWKS_JSON: &str = r#"{
  "keys": [
    {
      "kty": "EC",
      "crv": "P-256",
      "x": "MKBCTNIcKUSDii11ySs3526iDZ8AiTo7Tu6KPAqv7D4",
      "y": "4Etl6SRW2YiLUrN5vfvVHuhp7x8PxltmWWlbbM4IFyM",
      "use": "sig",
      "alg": "ES256",
      "kid": "test-key-1"
    }
  ]
}"#;
        let mut jwks_file = tempfile::NamedTempFile::new().expect("create temp JWKS file");
        jwks_file
            .write_all(JWKS_JSON.as_bytes())
            .expect("write JWKS JSON");
        jwks_file.flush().expect("flush JWKS file");
        let jwks_path = jwks_file.path().to_str().expect("temp path is valid UTF-8");

        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/tmp/server.crt",
            "--tls-key",
            "/tmp/server.key",
            "--http-auth-jwks",
            jwks_path,
            "--http-auth-issuer",
            "https://issuer.test",
            "--http-auth-audience",
            "arcgraph",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };

        let cfg = build_http_oauth_config(&args)
            .expect("valid JWKS file => build must succeed")
            .expect("valid JWKS file => Some(OAuthConfig), not dev-mode None");

        assert_eq!(cfg.issuer, "https://issuer.test");
        assert_eq!(cfg.audiences, vec!["arcgraph".to_string()]);
        assert_eq!(
            cfg.jwks.keys().count(),
            1,
            "the RFC-7517 P-256 JWK must round-trip through from_jwk into exactly one decoding key",
        );
    }

    /// Runtime guard clap cannot express: `--http-auth-jwks` present but ZERO
    /// `--http-auth-audience`. The empty-audience check precedes the JWKS file
    /// read, so the staged path need not exist.
    #[test]
    fn build_http_oauth_config_err_on_empty_audience() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/tmp/server.crt",
            "--tls-key",
            "/tmp/server.key",
            "--http-auth-jwks",
            "/nonexistent/jwks.json",
            "--http-auth-issuer",
            "https://issuer.test",
            // deliberately NO --http-auth-audience: clap permits the empty Vec.
        ])
        .expect("clap permits jwks+issuer with no audience (Vec may parse empty)");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert!(
            args.http_auth_audience.is_empty(),
            "precondition: clap parsed an empty --http-auth-audience Vec",
        );

        let err = build_http_oauth_config(&args)
            .expect_err("JWKS set with zero --http-auth-audience must be a runtime error");
        let msg = err.to_string();
        assert!(
            msg.contains("at least one --http-auth-audience"),
            "expected the empty-audience bail, got: {msg}",
        );
    }

    #[test]
    fn rejects_conflicting_transports() {
        // --stdio-mcp + --http both set must be rejected by clap. Cert/key
        // are supplied so the SOLE violation is the transport conflict
        // (not the #761 `--http` requires-cert/key rule).
        let err = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/c",
            "--tls-key",
            "/k",
        ])
        .expect_err("conflicting transports must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("cannot be used with") || msg.contains("conflict"),
            "expected clap conflict error, got: {msg}",
        );
    }

    #[test]
    fn parses_serve_with_data_and_config() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--data",
            "/tmp/data",
            "--config",
            "/etc/arcgraph.toml",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(
            args.data.as_deref(),
            Some(PathBuf::from("/tmp/data").as_path())
        );
        assert_eq!(
            args.config.as_deref(),
            Some(PathBuf::from("/etc/arcgraph.toml").as_path()),
        );
    }

    #[test]
    fn parses_check() {
        let cli = Cli::try_parse_from(["arcgraph", "check"]).expect("parse");
        let Command::Check(args) = cli.command else {
            panic!("expected Check command")
        };
        assert!(args.data.is_none());
    }

    #[test]
    fn parses_check_with_data() {
        let cli = Cli::try_parse_from(["arcgraph", "check", "--data", "/var/lib/arcgraph"])
            .expect("parse");
        let Command::Check(args) = cli.command else {
            panic!("expected Check command")
        };
        assert_eq!(
            args.data.as_deref(),
            Some(PathBuf::from("/var/lib/arcgraph").as_path()),
        );
    }

    #[test]
    fn parses_dump_defaults() {
        let cli = Cli::try_parse_from(["arcgraph", "dump"]).expect("parse");
        let Command::Dump(args) = cli.command else {
            panic!("expected Dump command")
        };
        assert_eq!(args.format, DumpFormat::Json);
        assert_eq!(args.tenant, TenantId::DEFAULT.raw());
    }

    #[test]
    fn parses_dump_with_format_and_tenant() {
        let cli = Cli::try_parse_from(["arcgraph", "dump", "--format", "toon", "--tenant", "42"])
            .expect("parse");
        let Command::Dump(args) = cli.command else {
            panic!("expected Dump command")
        };
        assert_eq!(args.format, DumpFormat::Toon);
        assert_eq!(args.tenant, 42);
    }

    #[test]
    fn parses_dump_format_cypher() {
        let cli = Cli::try_parse_from(["arcgraph", "dump", "--format", "cypher"]).expect("parse");
        let Command::Dump(args) = cli.command else {
            panic!("expected Dump command")
        };
        assert_eq!(args.format, DumpFormat::Cypher);
    }

    #[test]
    fn check_subcommand_empty_data_dir_runs_to_completion() {
        // Drive the check body without --data; should print + return Ok.
        let args = CheckArgs { data: None };
        run_check(args).expect("check (no --data) runs to completion");
    }

    #[test]
    fn check_subcommand_rejects_missing_data_dir() {
        let args = CheckArgs {
            data: Some(PathBuf::from(
                "/tmp/this/path/does/not/exist/arcgraph-check",
            )),
        };
        let err = run_check(args).expect_err("missing dir must reject");
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn check_recognizes_direct_durable_layout_without_current() {
        let dir = tempfile::tempdir().expect("tempdir");
        std::fs::write(dir.path().join("VERSION"), b"4\n").expect("write VERSION");
        std::fs::create_dir(dir.path().join("wal")).expect("create wal");
        assert!(has_committed_store_state(dir.path()));
    }

    #[test]
    fn dump_subcommand_json_runs_to_completion() {
        let args = DumpArgs {
            data: None,
            format: DumpFormat::Json,
            tenant: TenantId::DEFAULT.raw(),
        };
        run_dump(args).expect("dump --format json runs");
    }

    #[test]
    fn dump_subcommand_cypher_runs_to_completion() {
        let args = DumpArgs {
            data: None,
            format: DumpFormat::Cypher,
            tenant: TenantId::DEFAULT.raw(),
        };
        run_dump(args).expect("dump --format cypher runs");
    }

    #[test]
    fn dump_subcommand_toon_runs_to_completion() {
        let args = DumpArgs {
            data: None,
            format: DumpFormat::Toon,
            tenant: 7,
        };
        run_dump(args).expect("dump --format toon runs");
    }

    /// A durable logical dump must refuse until it can export faithfully.
    #[test]
    fn dump_with_data_set_refuses_false_empty_backup_866() {
        let dir = tempfile::tempdir().expect("tempdir");
        let args = DumpArgs {
            data: Some(dir.path().to_path_buf()),
            format: DumpFormat::Cypher,
            tenant: TenantId::DEFAULT.raw(),
        };
        let err =
            run_dump(args).expect_err("dump --data must refuse, not silently emit an empty graph");
        let msg = err.to_string();
        assert!(
            msg.contains("storage-rooted logical export is not implemented"),
            "refusal must explain the missing export: {msg}"
        );
        assert!(
            msg.contains("arcgraph backup create"),
            "refusal must name the cold-backup alternative: {msg}"
        );
    }

    /// Refusal is format-independent.
    #[test]
    fn dump_with_data_set_refuses_for_every_format_866() {
        let dir = tempfile::tempdir().expect("tempdir");
        for format in [DumpFormat::Json, DumpFormat::Toon, DumpFormat::Cypher] {
            let args = DumpArgs {
                data: Some(dir.path().to_path_buf()),
                format,
                tenant: TenantId::DEFAULT.raw(),
            };
            let err = run_dump(args)
                .err()
                .unwrap_or_else(|| panic!("dump --data --format {format} must refuse"));
            assert!(
                err.to_string()
                    .contains("storage-rooted logical export is not implemented"),
                "refusal must explain the limitation for format {format}: {err}"
            );
        }
    }

    // ─── #761 slice 1 — serve --http TLS cert/key wiring ────────────
    //
    // The hermetic end-to-end TLS roundtrip lives in the subprocess test at
    // `tests/serve_http_tls_e2e.rs`; these pin the CLI and helper fault modes.

    #[test]
    fn serve_http_requires_tls_cert_and_key() {
        // `--http` without `--tls-cert`/`--tls-key` is a clean clap parse
        // error (NOT a panic): server-side TLS is mandatory for the HTTP
        // MCP transport (design-v2 §9.4).
        let err = Cli::try_parse_from(["arcgraph", "serve", "--http", "127.0.0.1:8443"])
            .expect_err("--http without cert/key must reject at parse time");
        let msg = err.to_string();
        assert!(
            msg.contains("tls-cert") && msg.contains("tls-key"),
            "clap must name BOTH missing required flags: {msg}"
        );
    }

    #[test]
    fn serve_http_with_only_cert_still_requires_key() {
        let err = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/tmp/x.crt",
        ])
        .expect_err("--http + only --tls-cert must reject");
        assert!(
            err.to_string().contains("tls-key"),
            "clap must name the still-missing --tls-key: {err}"
        );
    }

    #[test]
    fn tls_flags_conflict_with_non_http_transport() {
        // `--tls-cert`/`--tls-key` with `--stdio-mcp` is a clean conflict
        // error (the TLS flags `conflicts_with` the non-HTTP transports).
        let err = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--tls-cert",
            "/tmp/x.crt",
            "--tls-key",
            "/tmp/x.key",
        ])
        .expect_err("tls flags with --stdio-mcp must reject");
        let msg = err.to_string();
        assert!(
            msg.contains("cannot be used with") || msg.contains("conflict"),
            "expected clap conflict error for tls flags + --stdio-mcp: {msg}"
        );
    }

    #[test]
    fn tls_flags_without_transport_require_http() {
        // `--tls-cert`/`--tls-key` with NO transport flag still rejects:
        // the TLS flags `requires = "http"`.
        let err = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--tls-cert",
            "/tmp/x.crt",
            "--tls-key",
            "/tmp/x.key",
        ])
        .expect_err("tls flags without --http must reject");
        assert!(
            err.to_string().contains("http"),
            "clap must cite the required --http flag: {err}"
        );
    }

    #[test]
    fn parses_serve_http_with_tls_flags() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "127.0.0.1:8443",
            "--tls-cert",
            "/etc/arcgraph/server.crt",
            "--tls-key",
            "/etc/arcgraph/server.key",
            "--tls-hostname",
            "mcp.example.com",
            "--in-memory",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.http.as_deref(), Some("127.0.0.1:8443"));
        assert_eq!(
            args.tls_cert.as_deref(),
            Some(PathBuf::from("/etc/arcgraph/server.crt").as_path())
        );
        assert_eq!(
            args.tls_key.as_deref(),
            Some(PathBuf::from("/etc/arcgraph/server.key").as_path())
        );
        assert_eq!(args.tls_hostname.as_deref(), Some("mcp.example.com"));
        assert!(!args.allow_remote_http_bind);
    }

    #[test]
    fn parses_allow_remote_http_bind() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--http",
            "0.0.0.0:8443",
            "--tls-cert",
            "/c",
            "--tls-key",
            "/k",
            "--allow-remote-http-bind",
            "--in-memory",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert!(args.allow_remote_http_bind);
    }

    // ── validate_http_bind — loopback-default bind gate (design-v2 §9.4
    //    line 668 + W14 retro IR L1-HIGH-4). RED-on-revert: deleting the
    //    gate fails `http_bind_gate_rejects_non_loopback_without_optin`.

    #[test]
    fn http_bind_gate_allows_loopback() {
        let bind = validate_http_bind("127.0.0.1:8443", false).expect("loopback always allowed");
        assert_eq!(bind, "127.0.0.1:8443".parse::<SocketAddr>().unwrap());
        // ::1 (IPv6 loopback) too — opt-in not required.
        assert!(
            validate_http_bind("[::1]:8443", false).is_ok(),
            "::1 is loopback and must be allowed without opt-in"
        );
    }

    #[test]
    fn http_bind_gate_rejects_non_loopback_without_optin() {
        let err = validate_http_bind("0.0.0.0:8443", false)
            .expect_err("non-loopback bind without opt-in must refuse");
        let msg = err.to_string();
        assert!(
            msg.contains("--allow-remote-http-bind") && msg.contains("non-loopback"),
            "loopback-default refusal must cite the opt-in flag: {msg}"
        );
    }

    #[test]
    fn http_bind_gate_allows_non_loopback_with_optin() {
        let bind = validate_http_bind("0.0.0.0:8443", true)
            .expect("non-loopback bind allowed with --allow-remote-http-bind");
        assert_eq!(bind, "0.0.0.0:8443".parse::<SocketAddr>().unwrap());
    }

    #[test]
    fn http_bind_gate_rejects_unparseable_addr() {
        let err = validate_http_bind("not-a-socket-addr", false).expect_err("must reject");
        assert!(
            err.to_string().contains("not a valid SocketAddr"),
            "unparseable addr surfaces a clean error: {err}"
        );
    }

    // ── build_http_tls_resolver — clean startup errors (TlsResolverError
    //    → anyhow), NOT panics, on bad cert material.

    #[test]
    fn build_http_tls_resolver_rejects_missing_cert() {
        let err = build_http_tls_resolver(
            std::path::Path::new("/nonexistent/arcgraph-761/server.crt"),
            std::path::Path::new("/nonexistent/arcgraph-761/server.key"),
            None,
        )
        .expect_err("missing cert file must surface a clean error, not a panic");
        assert!(
            err.to_string().contains("failed to load TLS cert/key"),
            "startup-error context present: {err}"
        );
    }

    #[test]
    fn build_http_tls_resolver_rejects_malformed_pem() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cert = dir.path().join("server.crt");
        let key = dir.path().join("server.key");
        std::fs::write(
            &cert,
            b"-----BEGIN CERTIFICATE-----\nnot valid base64 @@@\n-----END CERTIFICATE-----\n",
        )
        .expect("write cert");
        std::fs::write(&key, b"definitely not a private key").expect("write key");
        let err = build_http_tls_resolver(&cert, &key, None)
            .expect_err("malformed PEM must surface a clean error, not a panic");
        assert!(
            err.to_string().contains("failed to load TLS cert/key"),
            "startup-error context present: {err}"
        );
    }

    // ─── arcgraph health subcommand (M6-10 / #310) ──────────────────

    #[test]
    fn parses_health_defaults() {
        let cli = Cli::try_parse_from(["arcgraph", "health"]).expect("parse");
        let Command::Health(args) = cli.command else {
            panic!("expected Health command")
        };
        assert_eq!(args.addr, DEFAULT_HEALTH_URL);
        assert_eq!(args.timeout_ms, 2000);
    }

    #[test]
    fn parses_health_with_addr_and_timeout() {
        let cli = Cli::try_parse_from([
            "arcgraph",
            "health",
            "--addr",
            "http://127.0.0.1:9999/custom",
            "--timeout-ms",
            "500",
        ])
        .expect("parse");
        let Command::Health(args) = cli.command else {
            panic!("expected Health command")
        };
        assert_eq!(args.addr, "http://127.0.0.1:9999/custom");
        assert_eq!(args.timeout_ms, 500);
    }

    #[test]
    fn health_probe_parses_default_url() {
        let probe = HealthProbe::parse(DEFAULT_HEALTH_URL).expect("parse");
        assert_eq!(probe.host, "127.0.0.1");
        assert_eq!(probe.port, 8080);
        assert_eq!(probe.path, "/healthz");
        assert_eq!(probe.connect_addr(), "127.0.0.1:8080");
    }

    #[test]
    fn health_probe_defaults_port_80_when_omitted() {
        let probe = HealthProbe::parse("http://example.com/healthz").expect("parse");
        assert_eq!(probe.host, "example.com");
        assert_eq!(probe.port, 80);
        assert_eq!(probe.path, "/healthz");
    }

    #[test]
    fn health_probe_defaults_path_when_omitted() {
        let probe = HealthProbe::parse("http://127.0.0.1:9090").expect("parse");
        assert_eq!(probe.port, 9090);
        assert_eq!(probe.path, "/");
    }

    #[test]
    fn health_probe_rejects_https_with_current_limitation() {
        let err = HealthProbe::parse("https://127.0.0.1:8443/healthz").expect_err("https rejects");
        let msg = err.to_string();
        assert!(msg.contains("accepts only http://"), "scheme limit: {msg}");
        assert!(msg.contains("TLS-aware client"), "workaround: {msg}");
    }

    #[test]
    fn health_probe_rejects_non_http_scheme() {
        let err = HealthProbe::parse("file:///etc/passwd").expect_err("file:// rejects");
        assert!(err.to_string().contains("must start with http://"));
    }

    #[test]
    fn health_probe_rejects_empty_authority() {
        let err = HealthProbe::parse("http:///healthz").expect_err("empty authority rejects");
        assert!(err.to_string().contains("empty authority"));
    }

    #[test]
    fn health_probe_rejects_invalid_port() {
        let err = HealthProbe::parse("http://127.0.0.1:notaport/").expect_err("bad port rejects");
        assert!(err.to_string().contains("invalid port"));
    }

    #[test]
    fn parse_status_line_extracts_200() {
        let buf = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(parse_status_line(buf).expect("parse"), 200);
    }

    #[test]
    fn parse_status_line_extracts_503() {
        let buf = b"HTTP/1.1 503 Service Unavailable\r\n\r\n";
        assert_eq!(parse_status_line(buf).expect("parse"), 503);
    }

    #[test]
    fn parse_status_line_accepts_http_1_0() {
        let buf = b"HTTP/1.0 204 No Content\r\n\r\n";
        assert_eq!(parse_status_line(buf).expect("parse"), 204);
    }

    #[test]
    fn parse_status_line_rejects_non_http() {
        let err = parse_status_line(b"GARBAGE\r\n\r\n").expect_err("non-HTTP rejects");
        assert!(err.to_string().contains("unexpected status line"));
    }

    // ─── W24-OPS-α R1 fix-up — CLI parser + helper regression tests ───

    #[test]
    fn parses_serve_admin_http_defaults() {
        // BLOCKER H1 regression: the default --admin-http loopback bind
        // does NOT need --allow-remote-admin-bind.
        let cli = Cli::try_parse_from(["arcgraph", "serve", "--stdio-mcp"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.admin_http, "127.0.0.1:8090");
        assert!(
            !args.allow_remote_admin_bind,
            "default omits --allow-remote-admin-bind",
        );
        assert_eq!(
            args.drain_grace_seconds, 15,
            "default drain grace = 15s per K8s sidecar best practice",
        );
    }

    #[test]
    fn parses_serve_admin_http_non_loopback_with_opt_in() {
        // BLOCKER H1 regression: non-loopback bind requires the explicit
        // --allow-remote-admin-bind opt-in. This test confirms clap
        // accepts the combination; admin_http_bind_validate_rejects_*
        // confirms the validate() body enforces the policy.
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--admin-http",
            "0.0.0.0:8090",
            "--allow-remote-admin-bind",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.admin_http, "0.0.0.0:8090");
        assert!(args.allow_remote_admin_bind);
    }

    #[test]
    fn parses_serve_drain_grace_seconds_override() {
        // MED M2 regression: --drain-grace-seconds is operator-tunable.
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--drain-grace-seconds",
            "45",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.drain_grace_seconds, 45);
    }

    #[test]
    fn parses_serve_community_refresh_secs_default_is_daily() {
        // ADR-202 §D-8 — the community refresh cadence defaults to the
        // ADR-040 §D-7 once-per-UTC-day (86 400 s) cadence. The scheduler
        // is only started when `--metrics-http` is also set (see
        // `maybe_start_community_scheduler`); this pins the flag default.
        let cli = Cli::try_parse_from(["arcgraph", "serve", "--stdio-mcp"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(
            args.community_refresh_secs, 86_400,
            "default community refresh cadence = 24h per ADR-040 §D-7",
        );
    }

    #[test]
    fn parses_serve_community_refresh_secs_override() {
        // ADR-202 §D-8 — operators validating the `/metrics` scrape
        // end-to-end (or on small graphs) can lower the cadence so the
        // gauge appears within a bounded window rather than after a day.
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--metrics-http",
            "127.0.0.1:9090",
            "--community-refresh-secs",
            "5",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.community_refresh_secs, 5);
        assert_eq!(args.metrics_http, "127.0.0.1:9090");
    }

    #[test]
    fn parses_serve_metrics_http_default_is_loopback_9090() {
        // OBS-1: metrics are ON by default. With no `--metrics-http` flag,
        // the default is the design-v2 §10.2 loopback scrape endpoint
        // `127.0.0.1:9090`, which is NON-empty → `run_serve` instantiates
        // the `MetricsRegistry` and binds the `/metrics` listener. The
        // loopback default passes `MetricsHttpServerConfig::validate` by
        // construction, so on-by-default does NOT weaken the W14 retro IR
        // L1-HIGH-4 loopback-default invariant (a non-loopback override
        // still needs `--allow-remote-metrics-bind`, asserted below).
        let cli = Cli::try_parse_from(["arcgraph", "serve", "--stdio-mcp"]).expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(
            args.metrics_http, "127.0.0.1:9090",
            "OBS-1: default --metrics-http is loopback:9090 (metrics on by default)"
        );
        assert!(
            !args.metrics_http.trim().is_empty(),
            "non-empty default must trip the run_serve Some-gate (registry + listener)",
        );
        // The default is loopback → the config the binary builds validates
        // without the opt-in, so the on-by-default posture is invariant-safe.
        let bind: SocketAddr = args
            .metrics_http
            .parse()
            .expect("default is a valid SocketAddr");
        let cfg = MetricsHttpServerConfig {
            bind,
            allow_remote_bind: args.allow_remote_metrics_bind,
        };
        cfg.validate().expect(
            "OBS-1 default (loopback:9090) must validate without --allow-remote-metrics-bind",
        );
        assert!(
            !args.allow_remote_metrics_bind,
            "default omits --allow-remote-metrics-bind",
        );
    }

    #[test]
    fn parses_serve_metrics_http_explicit_empty_disables() {
        // OBS-1 disable path (design (a)): an operator turns metrics OFF by
        // passing an explicit empty value that overrides the non-empty
        // default. The empty string trips the `trim().is_empty()` gate in
        // `run_serve` → `metrics_registry = None` (no registry, no listener,
        // zero overhead — the pre-OBS-1 opt-out posture).
        let cli = Cli::try_parse_from(["arcgraph", "serve", "--stdio-mcp", "--metrics-http", ""])
            .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(
            args.metrics_http, "",
            "explicit --metrics-http \"\" overrides the on-by-default value"
        );
        assert!(
            args.metrics_http.trim().is_empty(),
            "explicit empty must satisfy the run_serve None-gate (metrics OFF)",
        );
    }

    #[test]
    fn metrics_http_non_loopback_override_still_rejects_without_opt_in() {
        // OBS-1 invariant guard (W14 retro IR L1-HIGH-4): flipping the
        // default ON must NOT weaken the loopback-default security posture.
        // A non-loopback OVERRIDE — the address `run_serve` parses and
        // builds a `MetricsHttpServerConfig` from — is STILL refused unless
        // `--allow-remote-metrics-bind` is set. This is the exact config
        // path `run_serve` walks (parse → build cfg → validate()?), so the
        // invariant is asserted at the CLI wiring level, not just in
        // `ops::metrics_http`'s unit tests.
        use arcgraph_cli::ops::MetricsHttpError;
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--metrics-http",
            "0.0.0.0:9090",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert!(
            !args.allow_remote_metrics_bind,
            "no opt-in flag set in this scenario",
        );
        let bind: SocketAddr = args.metrics_http.parse().expect("valid SocketAddr");
        let cfg = MetricsHttpServerConfig {
            bind,
            allow_remote_bind: args.allow_remote_metrics_bind,
        };
        assert!(
            matches!(
                cfg.validate(),
                Err(MetricsHttpError::BindAddrForbidden { .. })
            ),
            "non-loopback override without --allow-remote-metrics-bind MUST reject",
        );
    }

    #[test]
    fn parses_serve_metrics_http_loopback_addr() {
        // W28 #588: operator opts into the Prometheus scrape listener on
        // the design-v2 §10.2 default port (9090), loopback (no opt-in
        // needed).
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--metrics-http",
            "127.0.0.1:9090",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.metrics_http, "127.0.0.1:9090");
        assert!(!args.allow_remote_metrics_bind);
    }

    #[test]
    fn parses_serve_metrics_http_non_loopback_with_opt_in() {
        // W28 #588: the canonical Kubernetes ServiceMonitor posture —
        // `--metrics-http 0.0.0.0:9090 --allow-remote-metrics-bind`
        // (loopback-default per W14 retro IR L1-HIGH-4; mirror of the
        // admin-bind opt-in). clap accepts the combination;
        // MetricsHttpServerConfig::validate enforces the policy (tested
        // in ops::metrics_http).
        let cli = Cli::try_parse_from([
            "arcgraph",
            "serve",
            "--stdio-mcp",
            "--metrics-http",
            "0.0.0.0:9090",
            "--allow-remote-metrics-bind",
        ])
        .expect("parse");
        let Command::Serve(args) = cli.command else {
            panic!("expected Serve command")
        };
        assert_eq!(args.metrics_http, "0.0.0.0:9090");
        assert!(args.allow_remote_metrics_bind);
    }

    #[test]
    fn admin_http_bind_validate_rejects_non_loopback_without_opt_in() {
        // BLOCKER H1 regression: validate() is the policy gate.
        // run_serve calls cfg.validate()? synchronously so the error
        // surfaces as an anyhow chain (operator sees an actionable
        // failure rather than a tracing::error! buried in journalctl).
        let cfg = AdminHttpServerConfig {
            bind: "0.0.0.0:8090".parse().expect("parse"),
            allow_remote_bind: false,
        };
        cfg.validate().expect_err("non-loopback bind must reject");
    }

    #[test]
    fn admin_http_bind_validate_accepts_non_loopback_with_opt_in() {
        // BLOCKER H1 regression: the opt-in flag lets the validate
        // gate accept 0.0.0.0 binds for K8s httpGet probes.
        let cfg = AdminHttpServerConfig {
            bind: "0.0.0.0:8090".parse().expect("parse"),
            allow_remote_bind: true,
        };
        cfg.validate().expect("opt-in must pass");
    }

    #[test]
    fn admin_http_bind_validate_accepts_loopback_unconditionally() {
        // BLOCKER H1 regression: loopback binds don't need the opt-in.
        let cfg = AdminHttpServerConfig {
            bind: "127.0.0.1:8090".parse().expect("parse"),
            allow_remote_bind: false,
        };
        cfg.validate().expect("loopback always passes");
    }
}
