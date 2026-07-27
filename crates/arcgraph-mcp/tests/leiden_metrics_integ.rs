//! ADR-202 integration test — `arcgraph_leiden_last_run_seconds`
//! fires after a REAL community run.
//!
//! This is the §5.2 honesty proof for the eighth design-v2 §10.2
//! metric (line 724): a production-composed engine
//! ([`bootstrap_engine`] — the same entry point embedded deployments
//! call) with a live [`MetricsRegistry`] threaded through
//! `EngineConfig::refresh_observer`, a real ingested topology, and a
//! forced scheduler tick (materialise from `CrudStore` → `GveLeiden`
//! → `install_into`) must expose the gauge on the Prometheus text
//! exposition with the tenant's refresh completion time — asserted
//! with an exact bounded real-value oracle (`[before, after]` Unix
//! seconds around the tick), per the
//! `feedback_review_oracle_relaxations.md` discipline (no relaxed
//! oracle; a faked/duration/staleness value fails loudly).
//!
//! The companion no-fake guard pins the inverse: an engine
//! bootstrapped WITHOUT the observer exposes NO series — the metric
//! cannot claim a run it was never told about (and the ADR-202 D-6
//! restart semantics — absent-until-first-success — follow from the
//! same property).
//!
//! Producer-side unit tests (observer fires once per success, never
//! on skip/failure, panic containment) live in
//! `arcgraph-community::scheduler::tests`; registry-side unit tests
//! (timestamp bounds, per-tenant series) in
//! `arcgraph-mcp::transport::metrics::tests`. This file pins the
//! cross-crate composition end-to-end.

use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use arcgraph_community::{CommunityIndexId, RefreshObserver, SchedulerConfig};
use arcgraph_core::{LabelId, Lsn, TenantId, TypeId};
use arcgraph_mcp::transport::metrics::MetricsRegistry;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::engine::{EngineConfig, bootstrap_engine};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::transaction::TxnManager;

/// Bootstrapped catalog + crud + txn manager (the storage substrate
/// a caller assembles before `bootstrap_engine`; mirrors
/// `arcgraph-storage/tests/engine_bootstrap_integration.rs`).
fn fixture() -> (Arc<SystemCatalog>, Arc<CrudStore>, Arc<TxnManager>) {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    (catalog, Arc::new(CrudStore::new()), mgr)
}

/// Ingest a 6-node two-triangle topology for `tenant` so the
/// scheduler tick has a real graph to run Leiden on.
fn install_two_triangles(crud: &Arc<CrudStore>, mgr: &Arc<TxnManager>, tenant: TenantId) {
    let label = LabelId::new(1);
    let ty = TypeId::new(1);
    let mut tx = mgr.begin(tenant);
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

/// Engine config with a long natural interval (tests drive `tick()`
/// explicitly) and an install-LSN floor above the ingest commits.
fn engine_cfg(
    catalog: Arc<SystemCatalog>,
    crud: Arc<CrudStore>,
    mgr: Arc<TxnManager>,
) -> EngineConfig {
    let mut cfg = EngineConfig::new(catalog, crud, mgr, CommunityIndexId::new(1));
    cfg.scheduler_config = SchedulerConfig {
        interval: Duration::from_secs(3600),
        max_tick_duration: Duration::from_secs(60),
        initial_install_lsn: Lsn::new(1_000),
    };
    cfg
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

/// Extract `arcgraph_leiden_last_run_seconds{tenant=N}` from the
/// text exposition; `None` when the series is absent.
fn leiden_gauge_value(text: &str, tenant: u64) -> Option<i64> {
    let needle = format!(r#"arcgraph_leiden_last_run_seconds{{tenant="{tenant}"}} "#);
    text.lines()
        .find_map(|l| l.strip_prefix(&needle))
        .map(|v| v.trim().parse::<i64>().expect("gauge value parses as i64"))
}

/// ADR-202 end-to-end: bootstrap → ingest → forced tick → the gauge
/// is exposed with the refresh completion timestamp (exact bounded
/// oracle).
#[test]
fn leiden_gauge_fires_after_real_community_run() {
    let (catalog, crud, mgr) = fixture();
    install_two_triangles(&crud, &mgr, TenantId::DEFAULT);

    let registry = MetricsRegistry::shared().expect("metrics registry");
    let mut cfg = engine_cfg(catalog, crud, mgr);
    cfg.refresh_observer = Some(Arc::clone(&registry) as Arc<dyn RefreshObserver>);
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    // Pre-tick: no run has happened in this process → no series
    // (ADR-202 D-6 absent-until-first-success).
    let pre = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        leiden_gauge_value(&pre, TenantId::DEFAULT.raw()).is_none(),
        "no series may exist before the first refresh; text was:\n{pre}"
    );

    // Forced tick = a REAL community run: ProductionRefreshHook
    // materialises DEFAULT's graph from CrudStore, GveLeiden runs,
    // install_into installs — then (and only then) the observer
    // fires.
    let before = unix_now_secs();
    handles.scheduler.tick();
    let after = unix_now_secs();

    // The tick refreshed DEFAULT for real (not a skip / failure).
    let h = handles.scheduler.health_check();
    assert_eq!(h.total_ticks, 1, "one forced tick");
    assert_eq!(h.total_refresh_failures, 0, "refresh must succeed");
    assert_eq!(h.total_soft_skips, 0, "DEFAULT must not soft-skip");

    // Exact bounded real-value oracle on the exposed series.
    let text = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    let v = leiden_gauge_value(&text, TenantId::DEFAULT.raw())
        .expect("gauge series must exist after a successful refresh");
    assert!(
        (before..=after).contains(&v),
        "gauge must hold the run's completion Unix time; got {v}, bounds [{before}, {after}]"
    );

    // Second tick: the timestamp may only move forward.
    handles.scheduler.tick();
    let text2 = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    let v2 =
        leiden_gauge_value(&text2, TenantId::DEFAULT.raw()).expect("series persists across ticks");
    assert!(v2 >= v, "second run must not move the timestamp backward");

    handles.scheduler.shutdown();
}

/// ADR-202 no-fake guard: the SAME composition WITHOUT the observer
/// runs the same real refresh, and the registry exposes NO series —
/// the gauge value can only originate from the scheduler's success
/// notification, never from registration side effects.
#[test]
fn leiden_gauge_absent_when_observer_not_wired() {
    let (catalog, crud, mgr) = fixture();
    install_two_triangles(&crud, &mgr, TenantId::DEFAULT);

    let registry = MetricsRegistry::shared().expect("metrics registry");
    // Default config: refresh_observer = None (the pre-ADR-202 path).
    let cfg = engine_cfg(catalog, crud, mgr);
    let handles = bootstrap_engine(cfg).expect("bootstrap");

    handles.scheduler.tick();
    let h = handles.scheduler.health_check();
    assert_eq!(h.total_refresh_failures, 0, "refresh itself succeeds");
    assert_eq!(h.total_soft_skips, 0);

    let text = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        !text.contains(r#"arcgraph_leiden_last_run_seconds{"#),
        "an unwired registry must expose no series even though a real refresh ran; text was:\n{text}"
    );

    handles.scheduler.shutdown();
}
