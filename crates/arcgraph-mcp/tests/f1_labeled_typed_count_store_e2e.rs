//! F1 (#1356 §F1) — labelled node counts + typed rel counts lower to the
//! O(1) count-store and stay EXACT across INSERT / DELETE.
//!
//! End-to-end through the JSON-RPC dispatcher against a real
//! `StorageRawQueryExecutor` over an in-memory backend with the
//! `PrimaryIndex` wired so the `CatalogStats` commit hook fires (mirrors
//! `raw_query_write_common::fresh_backend`). Without that hook the
//! per-label / per-type cardinalities never surface and a labelled MATCH
//! silently returns 0 rows (W23-M4-08-FINALIZE founding incident).
//!
//! Acceptance gates (RED-on-revert oracles):
//!   1. PLAN-SHAPE: `MATCH (n:User) RETURN count(n)` (and the typed rel
//!      form) LOWERS to `CountStore` — the explain body contains the
//!      `CountStore` operator and NO `Scan`. If F1 is reverted the plan
//!      falls back to Scan+Aggregate and this fails.
//!   2. EXACT: the count equals a full-scan count of the same fixture.
//!   3. EXACT-AFTER-MUTATION: the count reflects INSERTs and DELETEs
//!      because the CatalogStats counters are commit-maintained — the
//!      counter (not a scan) serves it, and it stays exact.
//!   4. INDEPENDENCE: per-label / per-type counters do not alias — a
//!      mutation on one label/type leaves the others' counts unchanged.

#![allow(clippy::unwrap_used)]

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{Dispatcher, RateLimiter, SessionScope, handle_raw_envelope};
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

/// In-memory backend with the `PrimaryIndex` wired so the `CatalogStats`
/// commit hook fires (the per-label / per-type counters F1 reads).
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

fn fresh_dispatcher() -> TestDispatcher {
    let backend = fresh_backend();
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend)),
        RateLimiter::new(),
    )
}

fn envelope(query: &str, explain: bool) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": "f1",
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 1,
            "query": query,
            "max_rows": 100,
            "format": "json",
            "explain": explain,
        }
    })
}

/// Run a query (execute mode) and return the parsed JSON body.
fn exec(d: &TestDispatcher, query: &str) -> Value {
    let resp = handle_raw_envelope(d, envelope(query, false)).expect("dispatch");
    assert!(
        resp.get("error").map(Value::is_null).unwrap_or(true),
        "query errored: {resp}"
    );
    let body = resp["result"]["body"].as_str().expect("body string");
    serde_json::from_str(body).expect("body JSON")
}

/// Run a query in explain mode and return the raw plan body string.
fn explain(d: &TestDispatcher, query: &str) -> String {
    let resp = handle_raw_envelope(d, envelope(query, true)).expect("dispatch");
    assert!(
        resp.get("error").map(Value::is_null).unwrap_or(true),
        "explain errored: {resp}"
    );
    resp["result"]["body"]
        .as_str()
        .expect("body string")
        .to_string()
}

/// Extract the single scalar count cell `rows[0][0]`.
fn count_of(body: &Value) -> i64 {
    body["rows"][0][0]
        .as_i64()
        .unwrap_or_else(|| panic!("count cell rows[0][0] missing/non-int: {body}"))
}

/// Assert a count query LOWERS to the count-store: the plan contains the
/// `CountStore` operator and NO `Scan` (the F1 point — a labelled/typed
/// count must not fall onto a full scan).
fn assert_lowers_to_count_store(d: &TestDispatcher, query: &str) {
    let plan = explain(d, query);
    assert!(
        plan.contains("CountStore"),
        "expected CountStore operator in plan for `{query}`: {plan}"
    );
    assert!(
        !plan.contains("Scan"),
        "count-store plan for `{query}` must not scan: {plan}"
    );
}

#[test]
fn f1_labeled_node_count_lowers_to_count_store_and_is_exact() {
    let d = fresh_dispatcher();
    // 3 :User, 2 :Post.
    for _ in 0..3 {
        let _ = exec(&d, "CREATE (n:User) RETURN n");
    }
    for _ in 0..2 {
        let _ = exec(&d, "CREATE (n:Post) RETURN n");
    }

    // Gate 1 — PLAN-SHAPE: the labelled count lowers to CountStore.
    assert_lowers_to_count_store(&d, "MATCH (n:User) RETURN count(n)");
    assert_lowers_to_count_store(&d, "MATCH (n:User) RETURN count(*)");

    // Gate 2 — EXACT (matches a full-scan count of the same fixture).
    assert_eq!(count_of(&exec(&d, "MATCH (n:User) RETURN count(n)")), 3);
    assert_eq!(count_of(&exec(&d, "MATCH (n:Post) RETURN count(n)")), 2);
    // Cross-check against the scan-materialised row count (row_count of a
    // non-count MATCH is served WITHOUT the count-store).
    let scan = exec(&d, "MATCH (n:User) RETURN n");
    assert_eq!(
        scan["row_count"].as_i64(),
        Some(3),
        "count-store total agrees with the full scan"
    );
}

#[test]
fn f1_labeled_node_count_exact_after_insert_and_delete() {
    let d = fresh_dispatcher();
    // (a:User)-[:KNOWS]->(b:User) => 2 users + 1 KNOWS; then a lone c:User.
    let _ = exec(&d, "CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r");
    let _ = exec(&d, "CREATE (c:User) RETURN c");

    // INSERT reflected: 3 users via the count-store path.
    assert_lowers_to_count_store(&d, "MATCH (n:User) RETURN count(n)");
    assert_eq!(count_of(&exec(&d, "MATCH (n:User) RETURN count(n)")), 3);

    // Partial DELETE: DETACH DELETE the KNOWS source `a` (cascades the
    // rel). `b` + `c` remain — a proper subset delete (3 → 2), not 0/all.
    let _ = exec(&d, "MATCH (a:User)-[r:KNOWS]->(b:User) DETACH DELETE a");

    // DELETE reflected in the commit-maintained counter — still exact,
    // still served by the count-store (not a scan).
    assert_lowers_to_count_store(&d, "MATCH (n:User) RETURN count(n)");
    assert_eq!(count_of(&exec(&d, "MATCH (n:User) RETURN count(n)")), 2);
    // The scan agrees with the counter post-delete.
    assert_eq!(
        exec(&d, "MATCH (n:User) RETURN n")["row_count"].as_i64(),
        Some(2)
    );
}

#[test]
fn f1_typed_rel_count_lowers_to_count_store_and_exact_after_mutation() {
    let d = fresh_dispatcher();
    // 2 KNOWS + 1 FOLLOWS.
    let _ = exec(&d, "CREATE (a:User)-[r:KNOWS]->(b:User) RETURN r");
    let _ = exec(&d, "CREATE (c:User)-[r:KNOWS]->(d:User) RETURN r");
    let _ = exec(&d, "CREATE (e:User)-[r:FOLLOWS]->(f:User) RETURN r");

    // Gate 1 — PLAN-SHAPE: typed rel count lowers to CountStore.
    assert_lowers_to_count_store(&d, "MATCH ()-[:KNOWS]->() RETURN count(*)");
    assert_lowers_to_count_store(&d, "MATCH (a)-[r:KNOWS]->(b) RETURN count(r)");

    // Gate 2 — EXACT per-type counts (independent counters).
    assert_eq!(
        count_of(&exec(&d, "MATCH ()-[:KNOWS]->() RETURN count(*)")),
        2
    );
    assert_eq!(
        count_of(&exec(&d, "MATCH ()-[:FOLLOWS]->() RETURN count(*)")),
        1
    );

    // DELETE all KNOWS rels (keep endpoints). The per-type counter
    // decrements to 0; FOLLOWS is UNTOUCHED (per-type independence).
    let _ = exec(&d, "MATCH (a:User)-[r:KNOWS]->(b:User) DELETE r");
    assert_eq!(
        count_of(&exec(&d, "MATCH ()-[:KNOWS]->() RETURN count(*)")),
        0
    );
    assert_eq!(
        count_of(&exec(&d, "MATCH ()-[:FOLLOWS]->() RETURN count(*)")),
        1
    );
    // Still served by the count-store post-delete.
    assert_lowers_to_count_store(&d, "MATCH ()-[:KNOWS]->() RETURN count(*)");
}
