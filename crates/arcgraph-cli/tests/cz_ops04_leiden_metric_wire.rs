//! Customer Zero MUST-OPS-04 (Prometheus metrics leg) — close the 7-of-8
//! gap: `arcgraph_leiden_last_run_seconds` (design-v2 §10.2, the eighth
//! metric) now FIRES on the served `arcgraph serve --data <dir>
//! --metrics-http <addr>` path.
//!
//! # The gap CZ found
//!
//! ADR-202 shipped the `RefreshObserver` seam + `MetricsRegistry` impl +
//! `EngineConfig.refresh_observer` field, and proved them end-to-end
//! through `bootstrap_engine` (`arcgraph-mcp/tests/leiden_metrics_integ.rs`).
//! But ADR-202 §D-8 deliberately deferred the **serve-binary scheduler**:
//! `arcgraph serve` does NOT call `bootstrap_engine` — its bootstrap
//! (`arcgraph_cli::bootstrap::build_durable` / `build_in_memory`) builds a
//! `MultiTenantRouter` directly and starts NO community scheduler. So in
//! production the gauge could never be recorded (no scheduler ⇒ no refresh
//! ⇒ no observer call ⇒ Prometheus omits the family) — exactly CZ's 7/8.
//!
//! # What this slice wires (ADR-202 §D-8 / §Open-questions)
//!
//! `arcgraph_cli::bootstrap::start_community_scheduler` starts a community
//! refresh scheduler over the SAME `catalog` / `crud` / `txn_manager` the
//! served `StorageBackend` reads, wired with the process `MetricsRegistry`
//! as the `RefreshObserver`. `bin/arcgraph.rs::maybe_start_community_scheduler`
//! calls it from every transport when `--metrics-http` is set.
//!
//! # What this file proves
//!
//! It drives the SAME public surface the binary uses:
//! `bootstrap_storage_backend_with_metrics(Durable, Some(registry))`
//! (the `--metrics-http` storage wire), ingests a real two-triangle
//! topology through the served CRUD path, starts the scheduler via
//! `start_community_scheduler` with the registry as observer, forces ONE
//! real refresh tick (materialise from CrudStore → GveLeiden →
//! install_into → observer notify), then scrapes the SAME `MetricsRegistry`
//! the `/metrics` listener serves and asserts
//! `arcgraph_leiden_last_run_seconds{tenant="1"}` is PRESENT with the
//! refresh-completion Unix time inside a `[before, after]` bounded oracle
//! (per `feedback_review_oracle_relaxations.md` — no relaxed oracle).
//!
//! The companion NO-FAKE / RED-on-revert guard runs the SAME real refresh
//! WITHOUT the observer wired and asserts the series is ABSENT — the gauge
//! value can only originate from the scheduler's success notification,
//! never from a registration side effect. Neutering the
//! `start_with_observer(.., Some(observer))` wire (passing `None`) flips the
//! PRESENT assertion RED.
//!
//! Why test-driven `tick()` and not a live serve: the scheduler's natural
//! cadence is 24 h (ADR-040 §D-7), so a live serve-and-wait is impractical
//! in a test window. `tick()` runs the IDENTICAL `refresh_one_tenant`
//! producer path the background thread runs — the observer notification is
//! the same code regardless of what drives the tick. The production wire
//! (CLI passes the observer) is pinned by `bin/arcgraph.rs`'s
//! `maybe_start_community_scheduler` calling THIS helper, plus the
//! `--community-refresh-secs` flag's clap parse test.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arcgraph::community::{RefreshObserver, SchedulerConfig};
use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend_with_metrics};
use arcgraph_core::{LabelId, Lsn, PartitionId, TenantId, TypeId};
use arcgraph_mcp::MetricsRegistry;
use arcgraph_mcp::storage::StorageBackend;
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::metrics::MetricsSink;
use tempfile::TempDir;

/// The shared per-tenant [`CrudStore`] for `tenant` via the production
/// router surface (the SAME store the scheduler's hook materialises from).
fn crud_for(backend: &StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

/// Ingest a 6-node two-triangle topology for `tenant` through the served
/// CRUD path, so the scheduler tick has a real graph to run Leiden on.
/// Mirrors `arcgraph-mcp/tests/leiden_metrics_integ.rs::install_two_triangles`.
fn install_two_triangles(backend: &StorageBackend, crud: &Arc<CrudStore>, tenant: TenantId) {
    let label = LabelId::new(1);
    let ty = TypeId::new(1);
    let mut tx = backend.txn_manager().begin(tenant);
    let mut n: Vec<arcgraph_core::NodeId> = Vec::with_capacity(6);
    for _ in 0..6 {
        n.push(
            crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty)
                .expect("create_node"),
        );
    }
    let edges = [(0usize, 1usize), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5)];
    for &(u, v) in &edges {
        crud::create_rel(crud, &mut tx, tenant, n[u], n[v], ty, &PropertyData::Empty)
            .expect("create_rel");
    }
    crud::commit(tx, crud).expect("commit");
}

fn unix_now_secs() -> i64 {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock after epoch")
            .as_secs(),
    )
    .expect("fits i64")
}

/// Extract `arcgraph_leiden_last_run_seconds{tenant=N}` from the text
/// exposition; `None` when the series is absent.
fn leiden_gauge_value(text: &str, tenant: u64) -> Option<i64> {
    let needle = format!(r#"arcgraph_leiden_last_run_seconds{{tenant="{tenant}"}} "#);
    text.lines()
        .find_map(|l| l.strip_prefix(&needle))
        .map(|v| v.trim().parse::<i64>().expect("gauge value parses as i64"))
}

/// A scheduler config whose install-LSN floor sits above the ingest
/// commits and whose interval is long (the test drives `tick()` directly,
/// so the background cadence never fires within the window).
fn test_scheduler_config() -> SchedulerConfig {
    SchedulerConfig {
        interval: Duration::from_secs(3600),
        max_tick_duration: Duration::from_secs(60),
        initial_install_lsn: Lsn::new(1_000),
    }
}

// ─────────────────────────────────────────────────────────────────────
// PRIMARY oracle: on the SAME served bootstrap surface the binary uses,
// the scheduler wired with the metrics observer fires
// arcgraph_leiden_last_run_seconds after a REAL community refresh.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn served_durable_with_observer_fires_leiden_gauge() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // Mirror `bin/arcgraph.rs` `--metrics-http`: a process MetricsRegistry
    // coerced to `dyn MetricsSink` and threaded into the served bootstrap.
    // Keep the concrete `Arc` to (a) coerce to the community observer and
    // (b) scrape `/metrics`.
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

    // Real ingest through the served CRUD path → the scheduler's hook
    // materialises THIS graph on tick.
    let crud = crud_for(&backend, TenantId::DEFAULT);
    install_two_triangles(&backend, &crud, TenantId::DEFAULT);

    // THE WIRE UNDER TEST: start the community scheduler exactly as
    // `bin/arcgraph.rs::maybe_start_community_scheduler` does — same helper,
    // same registry coerced to the observer seam.
    let observer: Arc<dyn RefreshObserver> = registry.clone() as Arc<dyn RefreshObserver>;
    let scheduler = arcgraph_cli::bootstrap::start_community_scheduler(
        Arc::clone(backend.router().catalog()),
        Arc::clone(backend.router().crud()),
        Arc::clone(backend.txn_manager()),
        observer,
        test_scheduler_config(),
    );

    // Pre-tick: ADR-202 §D-6 absent-until-first-success.
    let pre = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        leiden_gauge_value(&pre, TenantId::DEFAULT.raw()).is_none(),
        "no series may exist before the first refresh; text was:\n{pre}"
    );

    // Forced tick = a REAL community run (the identical producer path the
    // background thread runs): materialise DEFAULT from CrudStore, GveLeiden
    // runs, install_into installs, THEN the observer fires.
    let before = unix_now_secs();
    scheduler.tick();
    let after = unix_now_secs();

    // The tick refreshed DEFAULT for real (not a skip / failure).
    let h = scheduler.health_check();
    assert_eq!(h.total_ticks, 1, "one forced tick");
    assert_eq!(h.total_refresh_failures, 0, "refresh must succeed");
    assert_eq!(h.total_soft_skips, 0, "DEFAULT must not soft-skip");

    // STRONG oracle: scrape the SAME registry the `/metrics` listener serves
    // and assert the gauge holds the run's completion Unix time within
    // `[before, after]`. A faked / duration / staleness value fails loudly.
    let text = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    let v = leiden_gauge_value(&text, TenantId::DEFAULT.raw()).unwrap_or_else(|| {
        panic!("arcgraph_leiden_last_run_seconds{{tenant=\"1\"}} must be PRESENT on /metrics after a successful served refresh; body was:\n{text}")
    });
    assert!(
        (before..=after).contains(&v),
        "gauge must hold the refresh completion Unix time; got {v}, bounds [{before}, {after}]\n{text}"
    );

    // Second tick: the timestamp may only move forward.
    scheduler.tick();
    let text2 = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    let v2 =
        leiden_gauge_value(&text2, TenantId::DEFAULT.raw()).expect("series persists across ticks");
    assert!(
        v2 >= v,
        "second run must not move the timestamp backward; {v2} < {v}"
    );

    scheduler.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// NO-FAKE / RED-on-revert guard: the SAME served stack + SAME real refresh
// WITHOUT the observer wired exposes NO series. The gauge value can only
// originate from the scheduler's success notification. Neutering the
// `Some(observer)` arg in `start_community_scheduler` flips the PRIMARY
// test's PRESENT assertion RED.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn served_durable_without_observer_exposes_no_leiden_series() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    let registry = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let sink: Arc<dyn MetricsSink> = registry.clone();
    let (backend, guard) = bootstrap_storage_backend_with_metrics(
        &BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        },
        Some(sink),
    )
    .expect("durable bootstrap with metrics");
    assert!(guard.is_durable());

    let crud = crud_for(&backend, TenantId::DEFAULT);
    install_two_triangles(&backend, &crud, TenantId::DEFAULT);

    // A NoOp observer (NOT the registry) — the refresh runs, installs, and
    // notifies THIS sink, which records nothing into the scraped registry.
    #[derive(Debug)]
    struct NoopObserver;
    impl RefreshObserver for NoopObserver {
        fn record_refresh_success(&self, _tenant: TenantId) {}
    }
    let observer: Arc<dyn RefreshObserver> = Arc::new(NoopObserver);
    let scheduler = arcgraph_cli::bootstrap::start_community_scheduler(
        Arc::clone(backend.router().catalog()),
        Arc::clone(backend.router().crud()),
        Arc::clone(backend.txn_manager()),
        observer,
        test_scheduler_config(),
    );

    scheduler.tick();
    let h = scheduler.health_check();
    assert_eq!(h.total_refresh_failures, 0, "refresh itself succeeds");
    assert_eq!(h.total_soft_skips, 0, "DEFAULT must not soft-skip");

    let text = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        !text.contains(r#"arcgraph_leiden_last_run_seconds{"#),
        "an un-wired registry must expose NO series even though a real refresh ran; body was:\n{text}"
    );

    scheduler.shutdown();
}

// ─────────────────────────────────────────────────────────────────────
// In-memory symmetry: the same wire works on `--in-memory` + `--metrics-http`
// (the gauge fires identically; the mode only changes the page substrate,
// not the community producer path).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn served_in_memory_with_observer_fires_leiden_gauge() {
    let registry = Arc::new(MetricsRegistry::new().expect("metrics registry"));
    let sink: Arc<dyn MetricsSink> = registry.clone();
    let (backend, guard) =
        bootstrap_storage_backend_with_metrics(&BootstrapMode::InMemory, Some(sink))
            .expect("in-memory bootstrap with metrics");
    assert!(!guard.is_durable(), "in-memory owns no WAL writer");

    let crud = crud_for(&backend, TenantId::DEFAULT);
    install_two_triangles(&backend, &crud, TenantId::DEFAULT);

    let observer: Arc<dyn RefreshObserver> = registry.clone() as Arc<dyn RefreshObserver>;
    let scheduler = arcgraph_cli::bootstrap::start_community_scheduler(
        Arc::clone(backend.router().catalog()),
        Arc::clone(backend.router().crud()),
        Arc::clone(backend.txn_manager()),
        observer,
        test_scheduler_config(),
    );

    let before = unix_now_secs();
    scheduler.tick();
    let after = unix_now_secs();

    let h = scheduler.health_check();
    assert_eq!(h.total_refresh_failures, 0, "refresh must succeed");
    assert_eq!(h.total_soft_skips, 0, "DEFAULT must not soft-skip");

    let text = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    let v = leiden_gauge_value(&text, TenantId::DEFAULT.raw()).unwrap_or_else(|| {
        panic!("leiden gauge must be PRESENT after an in-memory served refresh; body:\n{text}")
    });
    assert!(
        (before..=after).contains(&v),
        "gauge must hold completion Unix time; got {v}, bounds [{before}, {after}]"
    );

    scheduler.shutdown();
}
