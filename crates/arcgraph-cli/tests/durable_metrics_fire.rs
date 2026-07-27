//! W28 Feature #582 (ADR-045 / issue #727) — ADR-133 active verification that
//! `build_durable` threads the operator [`MetricsSink`] into the WAL writer +
//! catalog [`BufferPool`], so the design-v2 §10.2 storage producers actually
//! FIRE on the production `arcgraph serve --data <dir> --metrics-http <addr>`
//! path (not merely register).
//!
//! This file drives the SAME public surface the `arcgraph serve` binary uses:
//! `bootstrap_storage_backend_with_metrics(Durable, Some(registry))` (the
//! `--metrics-http` wiring at `bin/arcgraph.rs:557-562`), a real Strict-tier
//! commit through the production CRUD path, then a Prometheus text scrape of
//! the SAME [`MetricsRegistry`] the `/metrics` listener serves. The oracle is
//! the rendered observation COUNT, not the `# HELP` / `# TYPE` registration
//! line — per the spawn brief's "the actual COUNT must have incremented".
//!
//! # What fires, honestly (v1.2.0-GA exit-criteria §5.2)
//!
//! - `arcgraph_wal_fsync_duration_ms` (§10.2 line 704) — fires **LIVE**: every
//!   DEFAULT-tier (Strict) commit drives a WAL `fire()` → fsync →
//!   `observe_wal_fsync_ms` (`wal/writer.rs:788-792`) synchronously before the
//!   commit ack returns. PROVEN here by asserting a non-zero `_count`.
//!   ([`durable_serve_with_metrics_fires_wal_fsync_count_nonzero`])
//! - `arcgraph_storage_pages_total{kind}` (§10.2 line 703) — fires on REAL
//!   page reads at M10 stage-1 (ADR-207): `build_durable` §4 calls
//!   `SystemCatalog::attach_page_store`, which pins the dedicated catalog root
//!   page (read-back + materialize + verify), so a fresh durable boot serves
//!   ≥1 cold page read (`kind="miss"`) and a RESTART over the same dir serves
//!   the prior page's read-back (1 miss + ≥2 hits — the ADR-207 D-3 truth
//!   table). PROVEN here by count>0 strong oracles on both legs
//!   ([`durable_serve_with_metrics_fires_storage_pages_count_nonzero`] +
//!   [`durable_restart_reads_prior_catalog_page_hits_and_misses`]). This
//!   replaces the pre-ADR-207 honesty cut that stood in this header ("the
//!   catalog does not pin it … we therefore do NOT assert it fires") — the
//!   producer these tests were always waiting for now exists. RED-on-revert:
//!   neutering the `attach_page_store` call in `build_durable` returns these
//!   counters to 0 samples and fails both tests.

use arcgraph_cli::bootstrap::{
    BootstrapMode, bootstrap_storage_backend, bootstrap_storage_backend_with_metrics,
};
use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_mcp::MetricsRegistry;
use arcgraph_mcp::storage::StorageBackend;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::metrics::MetricsSink;
use std::sync::Arc;
use tempfile::TempDir;

/// The shared per-tenant [`CrudStore`] for `tenant` via the production router
/// surface (v1.0 routes every tenant to one store).
fn crud_for(backend: &StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

/// Commit one node under `tenant` via the bootstrapped backend. On the durable
/// `DEFAULT` (Strict) path this is an fsync-before-ack commit → WAL `fire()`.
/// Mirrors the production CRUD write path (and `durable_bootstrap_restart.rs`).
fn commit_node(backend: &StorageBackend, crud: &Arc<CrudStore>, tenant: TenantId, label: u32) {
    let mut tx = backend.txn_manager().begin(tenant);
    create_node(
        crud,
        &mut tx,
        tenant,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(11, 22),
    )
    .expect("create_node");
    commit(tx, crud).expect("commit node");
}

/// Parse the value of a single label-free Prometheus sample line
/// `"<metric> <value>"` out of the text-exposition body. Returns `None` if the
/// line is absent. Skips `# HELP` / `# TYPE` comment lines, and requires
/// whitespace immediately after the metric name so `_count` does not match a
/// `_count`-prefixed name and the bare name does not match a `_bucket` line.
fn scrape_value(body: &str, metric: &str) -> Option<f64> {
    body.lines().filter(|l| !l.starts_with('#')).find_map(|l| {
        let rest = l.strip_prefix(metric)?.strip_prefix(' ')?;
        rest.split_whitespace().next()?.parse::<f64>().ok()
    })
}

/// Parse a single-label counter sample
/// `<metric>{kind="<kind>"} <value>` out of the text-exposition body.
/// Returns `None` when the labeled series is absent (a counter-vec child
/// that never incremented renders no sample line).
fn scrape_kind_value(body: &str, metric: &str, kind: &str) -> Option<f64> {
    let prefix = format!("{metric}{{kind=\"{kind}\"}}");
    body.lines().filter(|l| !l.starts_with('#')).find_map(|l| {
        let rest = l.strip_prefix(prefix.as_str())?.strip_prefix(' ')?;
        rest.split_whitespace().next()?.parse::<f64>().ok()
    })
}

/// Sum of the storage_pages counter across `hit` + `miss` (the
/// `buffer_pool_hit_rate` denominator per §10.2 line 703).
fn storage_pages_hit_plus_miss(body: &str) -> f64 {
    scrape_kind_value(body, "arcgraph_storage_pages_total", "hit").unwrap_or(0.0)
        + scrape_kind_value(body, "arcgraph_storage_pages_total", "miss").unwrap_or(0.0)
}

// ─────────────────────────────────────────────────────────────────────
// PRIMARY oracle + ADR-133 active verification: wal_fsync fires LIVE on the
// durable `--metrics-http` path (the gap #727 closes).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn durable_serve_with_metrics_fires_wal_fsync_count_nonzero() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // Mirror `bin/arcgraph.rs:557-562`: the process MetricsRegistry coerced to
    // `dyn MetricsSink` and threaded into bootstrap on the `--metrics-http`
    // path. `registry.clone()` (an O(1) Arc-share of the prometheus registry)
    // unsizes to `Arc<dyn MetricsSink>`; we keep the concrete `Arc` to scrape.
    let registry = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let sink: Arc<dyn MetricsSink> = registry.clone();

    let (backend, guard) = bootstrap_storage_backend_with_metrics(
        &BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        },
        Some(sink),
    )
    .expect("durable bootstrap with metrics");
    assert!(guard.is_durable(), "durable mode must own a WAL writer");

    // A real Strict-tier commit through the production CRUD path → WAL fire()
    // → fsync → observe_wal_fsync_ms (synchronous, before the commit ack).
    let crud = crud_for(&backend, TenantId::DEFAULT);
    commit_node(&backend, &crud, TenantId::DEFAULT, 7);

    // STRONG oracle: scrape the SAME registry the `/metrics` listener serves
    // and assert the histogram's observation COUNT incremented — proving the
    // sink was threaded into the WalWriter (the gap this PR closes), not merely
    // that the metric is registered.
    let body = String::from_utf8(registry.gather_text().expect("gather_text")).expect("utf8");
    let count = scrape_value(&body, "arcgraph_wal_fsync_duration_ms_count")
        .unwrap_or_else(|| panic!("wal_fsync count line absent from /metrics body:\n{body}"));
    assert!(
        count > 0.0,
        "arcgraph_wal_fsync_duration_ms_count must be > 0 after a Strict commit \
         (proves build_durable threaded the sink into the WalWriter); got {count}\n{body}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// THE §5.2 LAST-GAP oracle (M10 stage-1, ADR-207): storage_pages fires on
// REAL catalog-page reads through the production durable bootstrap path.
// RED-on-revert: neutering the `attach_page_store` call in `build_durable`
// makes both tests below observe 0 samples and fail.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn durable_serve_with_metrics_fires_storage_pages_count_nonzero() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    let registry = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let sink: Arc<dyn MetricsSink> = registry.clone();

    let (_backend, guard) = bootstrap_storage_backend_with_metrics(
        &BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        },
        Some(sink),
    )
    .expect("durable bootstrap with metrics");
    assert!(guard.is_durable());

    // STRONG oracle: a fresh durable boot materializes the catalog root page
    // and round-trip-verifies it through the pool — the verify `pin_read`
    // cold-loads page 0 from pages.db, a REAL disk-served page read
    // (`kind="miss"`). Pre-ADR-207 this counter read 0 samples forever on
    // this path (the pool was a bootstrap-scoped local nothing pinned).
    let body = String::from_utf8(registry.gather_text().expect("gather_text")).expect("utf8");
    let miss = scrape_kind_value(&body, "arcgraph_storage_pages_total", "miss")
        .unwrap_or_else(|| panic!("storage_pages miss line absent from /metrics body:\n{body}"));
    assert!(
        miss > 0.0,
        "arcgraph_storage_pages_total{{kind=\"miss\"}} must be > 0 after a fresh durable \
         bootstrap (the ADR-207 attach verify read cold-loads the catalog root page); \
         got {miss}\n{body}"
    );
    assert!(
        storage_pages_hit_plus_miss(&body) > 0.0,
        "buffer_pool_hit_rate denominator (hit+miss) must be > 0 — the §5.2 producer fires"
    );
}

#[test]
fn durable_restart_reads_prior_catalog_page_hits_and_misses() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // First "process": materializes the catalog root page, then shuts down
    // (drop order releases the WAL writer and the #886 data-dir lock).
    {
        let (_backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("first durable bootstrap");
    }

    // Second "process" over the same dir, with a fresh registry: the attach
    // read-back serves the PRIOR page from disk. ADR-207 D-3 truth table for
    // a restart: 1 Miss (cold read-back) + ≥2 Hits (materialize pin_write +
    // verify pin_read on the now-resident page).
    let registry = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let sink: Arc<dyn MetricsSink> = registry.clone();
    let (_backend, guard) =
        bootstrap_storage_backend_with_metrics(&BootstrapMode::Durable { data_dir }, Some(sink))
            .expect("second durable bootstrap (restart)");
    assert!(guard.is_durable());

    let body = String::from_utf8(registry.gather_text().expect("gather_text")).expect("utf8");
    let miss = scrape_kind_value(&body, "arcgraph_storage_pages_total", "miss").unwrap_or(0.0);
    let hit = scrape_kind_value(&body, "arcgraph_storage_pages_total", "hit").unwrap_or(0.0);
    assert!(
        miss >= 1.0,
        "restart must cold-read the prior catalog root page (≥1 miss); got {miss}\n{body}"
    );
    assert!(
        hit >= 2.0,
        "restart must hit the resident page on materialize + verify (≥2 hits); got {hit}\n{body}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// None zero-overhead path pin — build_durable with metrics_sink: None still
// bootstraps + commits without panic (the operator did not pass
// `--metrics-http`). Mirrors `buffer.rs::metrics_sink_none_path_pin_works`.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn durable_serve_none_metrics_sink_commits_without_panic() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // `bootstrap_storage_backend` is the `None` variant (it calls
    // `..._with_metrics(mode, None)`), so this pins the legacy zero-overhead
    // path through all three sink consumers (BufferPool / WalConfig / CrudStore
    // all stay `metrics_sink: None`).
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable { data_dir })
        .expect("durable bootstrap (no metrics)");
    assert!(guard.is_durable(), "durable mode must own a WAL writer");

    // The Strict commit must succeed with no sink attached (no panic, no
    // overhead beyond the producers' nullable-ptr `None` checks).
    let crud = crud_for(&backend, TenantId::DEFAULT);
    commit_node(&backend, &crud, TenantId::DEFAULT, 9);
}

// ─────────────────────────────────────────────────────────────────────
// In-memory symmetry pin — `--in-memory` + `--metrics-http` threads the sink
// into the (catalog-only) BufferPool without panic. There is no WAL in this
// mode (`wal: None`), so wal_fsync is intentionally not wired/observed here —
// this only pins that the in-memory BufferPool wire compiles + runs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn in_memory_serve_with_metrics_bootstraps_and_commits_without_panic() {
    let registry = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let sink: Arc<dyn MetricsSink> = registry.clone();

    let (backend, guard) =
        bootstrap_storage_backend_with_metrics(&BootstrapMode::InMemory, Some(sink))
            .expect("in-memory bootstrap with metrics");
    assert!(
        !guard.is_durable(),
        "in-memory mode owns no WAL writer (no wal_fsync producer)"
    );
    assert!(
        guard.last_durable_lsn().is_none(),
        "in-memory mode has no WAL watermark"
    );

    // The write path works with the sink wired into the in-memory catalog pool.
    let crud = crud_for(&backend, TenantId::DEFAULT);
    commit_node(&backend, &crud, TenantId::DEFAULT, 3);

    // Honest: in-memory has no WAL, so the wal_fsync histogram never observes.
    // (We assert the absence of a positive count to document the mode boundary
    // — the count line is present-but-zero since the histogram is registered.
    // `< 1.0` ⟺ zero observations: the count is an integer-valued sample, and
    // a `<` comparison sidesteps float-equality lints.)
    let body = String::from_utf8(registry.gather_text().expect("gather_text")).expect("utf8");
    let count = scrape_value(&body, "arcgraph_wal_fsync_duration_ms_count").unwrap_or(0.0);
    assert!(
        count < 1.0,
        "in-memory mode must NOT observe any WAL fsync (no WAL writer); got {count}\n{body}"
    );

    // M10 stage-1 (ADR-207) symmetry: the in-memory flow ALSO attaches the
    // catalog root page, so storage_pages fires here too (the old "reads 0
    // samples until the page-backed read path lands" caveat is closed on
    // both paths).
    assert!(
        storage_pages_hit_plus_miss(&body) > 0.0,
        "in-memory bootstrap must fire storage_pages via the ADR-207 catalog attach\n{body}"
    );
}
