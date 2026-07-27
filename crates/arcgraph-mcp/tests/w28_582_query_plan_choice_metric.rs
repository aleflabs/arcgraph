//! W28 Feature #582 (ADR-045) — `arcgraph_query_plan_choice{plan_type}`
//! producer wire: end-to-end no-op-trampoline regression + None-path
//! fault injection.
//!
//! design-v2 §10.2 **line 723**: `arcgraph_query_plan_choice{plan_type}`
//! `(binary/wcoj/free_join)`.
//!
//! # Why this test exists (the no-op-trampoline guard)
//!
//! The companion `arcgraph_hot_vertex_warnings_total` metric was the
//! founding no-op trampoline (registered, never fired). `query_plan_choice`
//! is wired in the SAME slice; this test guards it against the same bug
//! class (`feedback_noop_trampoline_anti_pattern.md`). It uses the REAL
//! production sink — `arcgraph_mcp::MetricsRegistry` — wired into the
//! production `StorageRawQueryExecutor` and driven through the same
//! `Dispatcher` tower the stdio / Bolt / HTTP transports wrap, then
//! asserts the metric VALUE moved in the Prometheus text exposition.
//! A regression that drops the `record_query_plan_choice` emit fails
//! here even though the query still executes correctly.
//!
//! # Why the producer lives in the MCP adapter (not `arcgraph-query`)
//!
//! PD-7 bounded contexts: `arcgraph-query`'s library depends only on
//! `arcgraph-core` (its `arcgraph-storage` edge is a *dev*-dependency),
//! so the `QueryEngine` cannot reference the storage-resident
//! `MetricsSink`. `arcgraph-mcp` depends on BOTH query + storage, so
//! the `StorageRawQueryExecutor` — the production `graph.raw_query`
//! execution boundary — is the lowest layer that can reach the trait
//! AND observe a query execution. See `StorageRawQueryExecutor`'s
//! rustdoc for the full rationale.

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{Dispatcher, MetricsRegistry, RateLimiter, SessionScope};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::metrics::MetricsSink;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use serde_json::{Value, json};

type TestDispatcher = Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
>;

/// Production-shaped backend (mirrors
/// `arcgraph-cli/src/bootstrap.rs::bootstrap_storage_backend` +
/// `m4_08_finalize_end_to_end.rs::fresh_backend`): a `PrimaryIndex` is
/// wired so the per-tenant `CatalogStats` hook fires (without it,
/// `MATCH (n:Label)` falls through to ID(0) → zero rows).
fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None)
            .expect("PrimaryIndex::new"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

fn dispatcher_with(
    backend: StorageBackend,
    executor: Arc<StorageRawQueryExecutor>,
) -> TestDispatcher {
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        executor,
        RateLimiter::new(),
    )
}

/// Ingest a small service-dependency fixture so the join query below
/// resolves real rows (5 `Service` nodes + a `DEPENDS_ON` chain).
fn ingest_fixture(d: &TestDispatcher) {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": "svc-1", "label": "Service", "properties": {} },
                { "external_id": "svc-2", "label": "Service", "properties": {} },
                { "external_id": "svc-3", "label": "Service", "properties": {} },
                { "external_id": "svc-4", "label": "Service", "properties": {} },
                { "external_id": "svc-5", "label": "Service", "properties": {} }
            ],
            "relationships": [
                { "from_external_id": "svc-1", "to_external_id": "svc-2",
                  "rel_type": "DEPENDS_ON", "properties": {} },
                { "from_external_id": "svc-2", "to_external_id": "svc-3",
                  "rel_type": "DEPENDS_ON", "properties": {} },
                { "from_external_id": "svc-3", "to_external_id": "svc-4",
                  "rel_type": "DEPENDS_ON", "properties": {} },
                { "from_external_id": "svc-3", "to_external_id": "svc-5",
                  "rel_type": "DEPENDS_ON", "properties": {} }
            ],
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("ingest dispatch");
    assert!(
        resp["error"].is_null(),
        "fixture ingest must succeed: {resp:?}"
    );
}

fn raw_query(d: &TestDispatcher, query: &str) -> Value {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(2)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 1,
            "query": query,
            "max_rows": 100,
            "format": "json"
        }),
    };
    d.dispatch(req).expect("raw_query dispatch")
}

/// No-op-trampoline regression guard: a real `graph.raw_query` through
/// the production dispatcher MUST increment
/// `arcgraph_query_plan_choice{plan_type="binary"}` in the wired
/// `MetricsRegistry`. Strong oracle: exact (`==`) counter value per
/// executed query (the v1.0 engine is binary-only, deterministic).
#[test]
fn raw_query_fires_query_plan_choice_through_metrics_registry() {
    let backend = fresh_backend();
    // The REAL production sink (not a test fake).
    let registry = MetricsRegistry::shared().expect("metrics registry");
    let executor = Arc::new(
        StorageRawQueryExecutor::new(backend.clone())
            .with_metrics_sink(registry.clone() as Arc<dyn MetricsSink>),
    );
    let d = dispatcher_with(backend, executor);
    ingest_fixture(&d);

    // Cold: before any query, the *Vec metric has no `binary` cell.
    let cold = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        !cold.contains(r#"arcgraph_query_plan_choice{plan_type="binary"}"#),
        "no plan-choice cell before any query; scrape:\n{cold}"
    );

    // A NON-TRIVIAL plan choice: a 2-hop join sharing binding `b`
    // (scan → expand → join → expand). The v1.0 engine resolves this
    // to a binary join plan → plan_type="binary".
    let resp = raw_query(
        &d,
        "MATCH (a:Service)-[:DEPENDS_ON]->(b:Service), (b)-[:DEPENDS_ON]->(c:Service) RETURN a, b, c",
    );
    assert!(resp["error"].is_null(), "join query must succeed: {resp:?}");

    let after = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        after.contains(r#"arcgraph_query_plan_choice{plan_type="binary"} 1"#),
        "exactly one plan-choice increment after one executed query; scrape:\n{after}"
    );

    // Monotonic: a second executed query advances the counter to 2.
    let resp2 = raw_query(&d, "MATCH (s:Service) RETURN s");
    assert!(
        resp2["error"].is_null(),
        "scan query must succeed: {resp2:?}"
    );
    let after2 = String::from_utf8(registry.gather_text().expect("gather")).expect("utf-8");
    assert!(
        after2.contains(r#"arcgraph_query_plan_choice{plan_type="binary"} 2"#),
        "counter must advance monotonically to 2; scrape:\n{after2}"
    );

    // v1.0 emits ONLY `binary` — wcoj / free_join cells must be absent.
    assert!(
        !after2.contains(r#"arcgraph_query_plan_choice{plan_type="wcoj"}"#)
            && !after2.contains(r#"arcgraph_query_plan_choice{plan_type="free_join"}"#),
        "v1.0 must emit only plan_type=binary; scrape:\n{after2}"
    );
}

/// Fault-injection per failure mode: the unwired (`metrics_sink: None`)
/// executor must execute queries identically — no panic, no behavior
/// change. Pins the PD-5 `Option::is_none()` early-out on the query
/// producer path.
#[test]
fn raw_query_without_metrics_sink_is_noop_and_does_not_panic() {
    let backend = fresh_backend();
    // No `.with_metrics_sink(...)` — the legacy zero-overhead path.
    let executor = Arc::new(StorageRawQueryExecutor::new(backend.clone()));
    let d = dispatcher_with(backend, executor);
    ingest_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (a:Service)-[:DEPENDS_ON]->(b:Service), (b)-[:DEPENDS_ON]->(c:Service) RETURN a, b, c",
    );
    assert!(
        resp["error"].is_null(),
        "query must execute identically without a metrics sink: {resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    assert_eq!(
        rows["row_count"], 3,
        "None-sink path must not change results; body={body}"
    );
}
