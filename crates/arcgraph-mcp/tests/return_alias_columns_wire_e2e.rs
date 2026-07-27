//! **#353 (HIGH critical-path, customer-zero)** — user RETURN aliases
//! surface on BOTH wire surfaces: the MCP `graph.raw_query`
//! (`RawQueryRows::columns`) AND the Bolt `RunOutcome::fields`.
//!
//! # Why BOTH paths in one file
//!
//! #353's load-bearing requirement is that the column names are
//! user-meaningful on EVERY wire surface, not just one. langchain's
//! `Neo4jGraph` (and any neo4j Bolt driver) keys each result record by
//! the column name; a `RETURN n.name AS name` query must produce a
//! record keyed `name`, never `col_0`. Both renderers go through the
//! SAME `MaterializedResult::columns` →
//! `adapters::column_names_for_result` resolver, so this file pins that
//! they emit IDENTICAL names for the same query (the founding
//! requirement: never fix one wire path and leave the other on
//! `col_0`).
//!
//! # The DISCRIMINATING oracle (RED-without-the-fix)
//!
//! `RETURN n.name AS name` → MCP `columns == ["name"]` AND Bolt
//! `fields == ["name"]`. On `origin/main` without the fix both
//! synthesized `["col_0"]` (verified before implementing).
//!
//! # ADR-133 §D-4 "MCP" + "Driver" active-verification gates
//!
//! Hermetic (in-memory backend with PrimaryIndex wired so label-anchored
//! MATCH returns rows; in-process Bolt handler — no socket). The exact-
//! `columns` / exact-`fields` oracles are the openCypher implicit-
//! column-name rule + the `~/cz-apps/langchain-neo4j-test/cz_806_test.py`
//! repro (which keys `RETURN a.name AS a, b.name AS b` records by
//! `a` / `b`).

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageBoltHandler, StorageHybridSearcher, StorageIngestProvider,
    StorageNeighborhoodExplorer, StorageNodeInspector, StorageRawQueryExecutor,
    StorageSchemaProvider,
};
use arcgraph_mcp::transport::bolt::{BoltQueryHandler, BoltSessionAuth};
use arcgraph_mcp::{Dispatcher, RateLimiter, SessionScope};
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

/// Build a fresh in-memory backend with PrimaryIndex wired so the
/// `CatalogStats` hook fires inside `crud::commit` — without it,
/// label / rel-type cardinalities never reach the catalog and
/// `MATCH (n:Label)` silently returns 0 rows (W23-M4-08-FINALIZE
/// founding incident). Mirrors
/// `tests/raw_query_write_common/mod.rs::fresh_backend`.
fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("PrimaryIndex"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

/// Power-scope dispatcher over the SHARED backend (so the Bolt handler
/// built over the same backend observes the same seeded data).
fn dispatcher_over(backend: &StorageBackend) -> TestDispatcher {
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
        RateLimiter::new(),
    )
}

/// Dispatch a `graph.raw_query` and return the parsed JSON body.
fn raw_query_body(d: &TestDispatcher, query: &str) -> Value {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.raw_query".into(),
        params: json!({ "tenant_id": 1, "query": query, "max_rows": 100, "format": "json" }),
    };
    let resp = d.dispatch(req).expect("raw_query dispatch");
    assert!(
        resp["error"].is_null(),
        "raw_query error envelope: {resp:?}"
    );
    let body = resp["result"]["body"]
        .as_str()
        .unwrap_or_else(|| panic!("raw_query body missing: {resp:?}"));
    serde_json::from_str(body).expect("raw_query body JSON")
}

/// The MCP `columns` for a query, as a `Vec<String>` (empty if `null`).
fn mcp_columns(d: &TestDispatcher, query: &str) -> Vec<String> {
    let body = raw_query_body(d, query);
    match &body["columns"] {
        Value::Array(a) => a
            .iter()
            .map(|v| v.as_str().expect("column name is a string").to_string())
            .collect(),
        Value::Null => Vec::new(),
        other => panic!("unexpected columns shape: {other:?}"),
    }
}

/// The Bolt `RunOutcome::fields` for a query.
fn bolt_fields(h: &StorageBoltHandler, tenant: TenantId, query: &str) -> Vec<String> {
    let params: BTreeMap<String, arcgraph_mcp::transport::bolt::PackValue> = BTreeMap::new();
    let session = BoltSessionAuth::new(tenant, None, SessionScope::Power);
    h.run(&session, query, &params).expect("bolt run").fields
}

// ---------------------------------------------------------------------
// The discriminating oracle (RED-without-the-fix) — BOTH wire paths
// ---------------------------------------------------------------------

#[test]
fn both_wire_paths_emit_explicit_alias_not_col0() {
    let backend = fresh_backend();
    let d = dispatcher_over(&backend);
    let bolt = StorageBoltHandler::new(backend.clone());
    let tenant = TenantId::DEFAULT;

    // Seed one Person via the MCP write path (committed → visible to
    // both surfaces over the shared backend).
    let seed = raw_query_body(&d, "CREATE (n:Person {name: 'Ada'})");
    assert_eq!(seed["writes"]["nodes_created"], 1, "seed CREATE: {seed:?}");

    // The linchpin: `RETURN n.name AS name` → ["name"] on BOTH paths,
    // NOT ["col_0"]. (RED-as-["col_0"] on origin/main.)
    let q = "MATCH (n:Person) RETURN n.name AS name";
    assert_eq!(mcp_columns(&d, q), vec!["name"], "MCP columns");
    assert_eq!(bolt_fields(&bolt, tenant, q), vec!["name"], "Bolt fields");
}

// ---------------------------------------------------------------------
// The four acceptance shapes — verified on BOTH wire paths
// ---------------------------------------------------------------------

#[test]
fn both_wire_paths_match_acceptance_shapes() {
    let backend = fresh_backend();
    let d = dispatcher_over(&backend);
    let bolt = StorageBoltHandler::new(backend.clone());
    let tenant = TenantId::DEFAULT;

    raw_query_body(&d, "CREATE (n:Person {name: 'Ada', age: 36})");

    // (query, expected columns) — the #353 acceptance set.
    let cases: &[(&str, Vec<&str>)] = &[
        // bare variables → their names
        (
            "MATCH (n:Person) WITH n AS a, n AS b RETURN a, b",
            vec!["a", "b"],
        ),
        // explicit AS alias
        (
            "MATCH (n:Person) RETURN n.name AS person_name",
            vec!["person_name"],
        ),
        // un-aliased expression → source text
        ("MATCH (n:Person) RETURN n.name", vec!["n.name"]),
        // un-aliased aggregate → source text (aggregate-passthrough path)
        ("MATCH (n:Person) RETURN count(*)", vec!["count(*)"]),
        // aliased aggregate
        ("MATCH (n:Person) RETURN count(*) AS c", vec!["c"]),
        // mixed: bare + source-text + alias
        (
            "MATCH (n:Person) WITH n AS a RETURN a, a.name, a.name AS person_name",
            vec!["a", "a.name", "person_name"],
        ),
    ];

    for (q, expected) in cases {
        let want: Vec<String> = expected.iter().map(|s| (*s).to_string()).collect();
        assert_eq!(mcp_columns(&d, q), want, "MCP columns for `{q}`");
        assert_eq!(bolt_fields(&bolt, tenant, q), want, "Bolt fields for `{q}`");
    }
}

// ---------------------------------------------------------------------
// Wildcard → col_0..N fallback on BOTH paths (width-matched)
// ---------------------------------------------------------------------

#[test]
fn both_wire_paths_fall_back_to_col_n_for_wildcard() {
    let backend = fresh_backend();
    let d = dispatcher_over(&backend);
    let bolt = StorageBoltHandler::new(backend.clone());
    let tenant = TenantId::DEFAULT;

    raw_query_body(&d, "CREATE (n:Person {name: 'Ada'})");

    // `RETURN *` → one column (the single `n` binding) rendered as the
    // synthesized `col_0` fallback on BOTH paths (data-dependent width).
    let q = "MATCH (n:Person) RETURN *";
    assert_eq!(mcp_columns(&d, q), vec!["col_0"], "MCP wildcard fallback");
    assert_eq!(
        bolt_fields(&bolt, tenant, q),
        vec!["col_0"],
        "Bolt wildcard fallback"
    );
}

// ---------------------------------------------------------------------
// Row DATA unchanged — only column NAMES change (the #353 risk)
// ---------------------------------------------------------------------

#[test]
fn row_data_unchanged_on_mcp_path() {
    let backend = fresh_backend();
    let d = dispatcher_over(&backend);
    raw_query_body(&d, "CREATE (n:Person {name: 'Ada'})");

    let body = raw_query_body(&d, "MATCH (n:Person) RETURN n.name AS name");
    // Column name is the alias…
    assert_eq!(body["columns"], json!(["name"]));
    // …and the row DATA is the unchanged value.
    assert_eq!(body["rows"], json!([["Ada"]]));
    assert_eq!(body["row_count"], json!(1));
}
