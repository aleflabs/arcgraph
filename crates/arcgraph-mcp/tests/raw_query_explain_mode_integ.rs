//! `graph.raw_query` `explain:true` verb-consolidation mode — production
//! end-to-end integration through the JSON-RPC dispatcher against a real
//! `StorageRawQueryExecutor` over an in-memory backend.
//!
//! Operator ruling: plan introspection is consolidated INTO the existing
//! `graph.raw_query` tool via an optional `explain: bool` field (the
//! roadmap §"Notes for engineering" #3 verb-discrimination precedent) so
//! the MCP catalog stays at exactly six tools — NO separate
//! `graph.explain` tool is wired, NO ADR-004 amendment, NO tool removed.
//!
//! Acceptance gates (RED-on-revert oracles):
//!   1. `explain:true` returns the QUERY PLAN (plan-tree rows: op /
//!      details / estimated_cost / estimated_card / depth) — NOT the
//!      executed-query rows, NOT a MethodNotFound fault.
//!   2. `explain:false` (and absent) executes UNCHANGED — returns the
//!      executed-query data rows, no plan columns.
//!   3. `explain:true` STILL rejects a non-Power session with -32008
//!      (the scope gate inherits — a plan can leak schema / cardinality).
//!   4. The plan path is side-effect-free: an `explain:true` CREATE never
//!      commits (the plan is built, never run) — proven by a follow-up
//!      MATCH count of 0.
//!
//! All tests drive the dispatcher through `handle_raw_envelope` — the
//! same entry point `serve_stdio` / `serve_http` use — over the real
//! `StorageRawQueryExecutor`, NOT a stub fixture (per
//! `feedback_review_oracle_relaxations.md`).

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{Dispatcher, SessionScope, handle_raw_envelope};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
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

fn dispatcher_with_scope(scope: SessionScope, backend: StorageBackend) -> TestDispatcher {
    Dispatcher::with_session_scope(
        TenantId::DEFAULT,
        scope,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend)),
    )
}

fn raw_query_envelope(query: &str, explain: Option<bool>) -> Value {
    let mut params = json!({
        "tenant_id": 1,
        "query": query,
        "max_rows": 100,
        "format": "json"
    });
    if let Some(e) = explain {
        params["explain"] = json!(e);
    }
    json!({
        "jsonrpc": "2.0",
        "id": "rq-explain",
        "method": "graph.raw_query",
        "params": params
    })
}

fn body_of(resp: &Value) -> String {
    resp["result"]["body"]
        .as_str()
        .unwrap_or_else(|| panic!("expected result.body string, got: {resp}"))
        .to_string()
}

// ─────────────────────────────────────────────────────────────────────
// Gate 1 — explain:true returns the PLAN (not executed rows).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn explain_true_returns_query_plan_not_executed_rows() {
    let d = dispatcher_with_scope(SessionScope::Power, fresh_backend());
    let resp = handle_raw_envelope(
        &d,
        raw_query_envelope("MATCH (n:Person) RETURN n", Some(true)),
    )
    .expect("dispatcher response");

    // No error envelope — a plan is a successful result.
    assert!(
        resp.get("error").is_none(),
        "explain:true must not error on a valid query: {resp}"
    );
    let body = body_of(&resp);

    // Plan-row columns present (the #952 plan-row adapter shape:
    // [operator, details, est_cost, est_rows, depth]).
    assert!(body.contains("operator"), "plan column: {body}");
    assert!(body.contains("est_cost"), "plan column: {body}");
    assert!(body.contains("est_rows"), "plan column: {body}");
    assert!(body.contains("depth"), "plan column: {body}");
    // At least one plan operator row (a MATCH ... RETURN lowers to a
    // Scan under a Project) — the plan describes the query structure.
    assert!(
        body.contains("Scan") || body.contains("Project"),
        "plan operators present: {body}"
    );
    // Did NOT execute: a fresh backend has zero Person nodes, so an
    // executed query would return row_count 0 data rows — but a plan
    // always has >= 1 operator row regardless of data.
    let v: Value = serde_json::from_str(&body).expect("body is JSON");
    let row_count = v["row_count"].as_u64().expect("row_count");
    assert!(row_count >= 1, "plan has >= 1 operator row: {body}");
}

// ─────────────────────────────────────────────────────────────────────
// Gate 2 — explain:false (and absent) executes UNCHANGED.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn explain_false_executes_unchanged() {
    let d = dispatcher_with_scope(SessionScope::Power, fresh_backend());
    let resp = handle_raw_envelope(
        &d,
        raw_query_envelope("MATCH (n:Person) RETURN n", Some(false)),
    )
    .expect("dispatcher response");
    assert!(resp.get("error").is_none(), "valid execute: {resp}");
    let body = body_of(&resp);
    // Executed against an empty backend: 0 data rows, and crucially NOT
    // the plan columns.
    assert!(
        !body.contains("est_cost"),
        "execute path must not emit plan columns: {body}"
    );
    let v: Value = serde_json::from_str(&body).expect("body is JSON");
    assert_eq!(v["row_count"].as_u64(), Some(0), "empty backend: {body}");
}

#[test]
fn explain_absent_executes_unchanged() {
    // A request envelope that OMITS `explain` must execute (the
    // #[serde(default)] field deserializes to false) — deny_unknown_
    // fields stays satisfied.
    let d = dispatcher_with_scope(SessionScope::Power, fresh_backend());
    let resp = handle_raw_envelope(&d, raw_query_envelope("MATCH (n:Person) RETURN n", None))
        .expect("dispatcher response");
    assert!(resp.get("error").is_none(), "valid execute: {resp}");
    let body = body_of(&resp);
    assert!(
        !body.contains("est_cost"),
        "execute path (explain absent): {body}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Gate 3 — explain:true STILL Power-gated (scope holds).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn explain_true_rejects_read_scope_with_forbidden() {
    let d = dispatcher_with_scope(SessionScope::Read, fresh_backend());
    let resp = handle_raw_envelope(
        &d,
        raw_query_envelope("MATCH (n:Person) RETURN n", Some(true)),
    )
    .expect("dispatcher response");
    assert_eq!(
        resp["error"]["code"], -32008,
        "explain:true inherits the Power-scope gate: {resp}"
    );
    assert_eq!(resp["error"]["data"]["required_scope"], "arcgraph.power");
}

// ─────────────────────────────────────────────────────────────────────
// Gate 4 — explain:true is side-effect-free (never commits a write).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn explain_true_create_does_not_commit() {
    // The free `arcgraph_query::explain` fn builds the plan but never
    // executes it — so an `explain:true` CREATE must NOT persist any
    // node. We prove it by EXPLAIN-ing a CREATE then EXECUTE-ing a
    // count: the count is 0 (the create never ran).
    let backend = fresh_backend();
    let d = dispatcher_with_scope(SessionScope::Power, backend);

    // EXPLAIN a CREATE — returns a plan, commits nothing.
    let explain_resp = handle_raw_envelope(
        &d,
        raw_query_envelope("CREATE (n:Widget {name: 'w1'})", Some(true)),
    )
    .expect("dispatcher response");
    assert!(
        explain_resp.get("error").is_none(),
        "explain:true CREATE returns a plan: {explain_resp}"
    );
    let plan_body = body_of(&explain_resp);
    assert!(
        plan_body.contains("est_cost") && plan_body.contains("CreateNode"),
        "CREATE plan body: {plan_body}"
    );

    // EXECUTE a count of Widget nodes — must be 0 (the explained CREATE
    // never committed).
    let count_resp = handle_raw_envelope(
        &d,
        raw_query_envelope("MATCH (n:Widget) RETURN count(n) AS c", None),
    )
    .expect("dispatcher response");
    assert!(
        count_resp.get("error").is_none(),
        "count query executes: {count_resp}"
    );
    let count_body = body_of(&count_resp);
    let v: Value = serde_json::from_str(&count_body).expect("count body JSON");
    // The single count row's value must be 0 — the explained CREATE was
    // never run. (Row shape: rows[0] is a JSON array [0].)
    let count_cell = &v["rows"][0][0];
    assert_eq!(
        count_cell.as_u64(),
        Some(0),
        "explained CREATE must not have committed any node: {count_body}"
    );
}
