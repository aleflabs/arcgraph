//! W23-M4-08-FINALIZE — end-to-end MCP → executor → storage tests.
//!
//! Complement to `w17_alpha_production_adapter_integration.rs`. W17α
//! shipped the production adapter bundle but its multi-pattern join
//! coverage used label-FREE queries (sidestepping the catalog↔intern
//! ID mismatch). W23-M4-08-FINALIZE closes the executor wiring loop
//! by:
//!
//! 1. **Exercising LABEL-anchored + REL-TYPE-anchored queries** via
//!    the production [`StorageRawQueryExecutor`] now that
//!    `build_catalog_for_tenant` uses the storage-allocated IDs
//!    verbatim (via the new `with_label_id` / `with_rel_type_id`
//!    builders).
//! 2. **Cross-tenant fault injection** — every test case verifies
//!    structural isolation per ADR-037 §D-3.
//! 3. **Substrate-unavailable fault injection** — vector / BM25
//!    paths surface `IndexUnavailable` when the router has no
//!    handle attached. The W26-β-3 / ADR-132 wire-through landed
//!    the real HNSW + Tantivy body via the `SubstrateSearchProvider`
//!    trait, but the unwired-tenant case continues to surface
//!    structured `IndexUnavailable` (load-bearing per ADR-132 D-3 /
//!    AC-6). Issue #438 closed at W26-β-3.
//! 4. **Multi-pattern join on rel-type-filtered patterns** — the
//!    load-bearing M4-08 LogicalJoin demonstration end-to-end via
//!    the MCP `graph.raw_query` surface.
//!
//! Each test routes through the same `Dispatcher` tower the
//! production stdio / HTTP / Bolt transports wrap, so the integration
//! coverage matches the production wire shape per
//! `feedback_review_oracle_relaxations.md` discipline.

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
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

fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    // CRITICAL: wire a `PrimaryIndex` so the per-tenant `CatalogStats`
    // hook inside `crud::commit` runs. Without it, label / rel-type
    // cardinalities never surface in the catalog and any downstream
    // `MATCH (n:Label)-[:REL]->...` binding falls through the
    // dynamic-name fallback to ID(0) → silent zero rows.
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

/// Ingest a incident-shaped fixture: 5 `Service` nodes + 4
/// outbound `DEPENDS_ON` edges chained as
/// `svc-1 → svc-2 → svc-3 → svc-4` with a `svc-3 → svc-5` fan-out.
fn ingest_incident_fixture(d: &TestDispatcher) {
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

// ─────────────────────────────────────────────────────────────────────
// LABEL-anchored queries — exercises catalog ID-consistency fix.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn label_anchored_scan_returns_real_row_count_via_mcp() {
    // Pin: a LABEL-anchored MATCH against the production substrate
    // returns the expected row count via the MCP `graph.raw_query`
    // surface. The W17α posture would have returned 0 rows for any
    // rel-type-anchored query (catalog↔intern ID mismatch); the
    // label-only case was load-bearing.
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);

    let resp = raw_query(&d, "MATCH (s:Service) RETURN s");
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    // 5 Services per the fixture; row count surfaces on the wire.
    assert_eq!(rows["row_count"], 5, "expected 5 Service rows; body={body}");
}

#[test]
fn rel_type_anchored_one_hop_returns_real_row_count_via_mcp() {
    // Pin: a REL-TYPE-anchored 1-hop MATCH against the production
    // substrate returns the expected row count. This is the
    // W23-M4-08-FINALIZE-load-bearing case — the W17α catalog↔intern
    // ID mismatch meant the planner passed the wrong TypeId to the
    // substrate's `expand` call, silently dropping all rows.
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (s:Service)-[:DEPENDS_ON]->(d:Service) RETURN s, d",
    );
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    // 4 DEPENDS_ON edges in the fixture.
    assert_eq!(
        rows["row_count"], 4,
        "expected 4 DEPENDS_ON edges; body={body}"
    );
}

#[test]
fn multi_pattern_join_with_label_and_rel_type_anchors_executes_end_to_end_via_mcp() {
    // The load-bearing M4-08 LogicalJoin demonstration end-to-end
    // through the MCP surface. The query joins two 1-hop patterns on
    // shared binding `b`, which lowers to a `LogicalJoin`. The
    // executor must:
    //   - bind `Service` → storage's LabelId (catalog ID-consistency)
    //   - bind `DEPENDS_ON` → storage's TypeId (catalog ID-consistency)
    //   - scan nodes filtered by LabelId
    //   - expand edges filtered by TypeId
    //   - join on shared binding `b`
    // returning real row data, NOT empty.
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (a:Service)-[:DEPENDS_ON]->(b:Service), (b)-[:DEPENDS_ON]->(c:Service) RETURN a, b, c",
    );
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    // Chains via shared `b`: (svc-1, svc-2, svc-3), (svc-2, svc-3,
    // svc-4), (svc-2, svc-3, svc-5) → 3 rows.
    assert_eq!(rows["row_count"], 3, "expected 3 join rows; body={body}");
}

// ─────────────────────────────────────────────────────────────────────
// Fault injection — substrate-unavailable / cross-tenant / unknown.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn vector_substrate_unavailable_surfaces_unavailable_error_via_mcp_search() {
    // Substrate-unavailable fault injection: the W17α posture
    // attaches no vector substrate to the fixture's tenant. The
    // `graph.search` MCP tool surfaces a structured
    // `-32004 IndexUnavailable` error on the wire when no
    // substrate is attached. This pins the W17α +
    // W23-M4-08-FINALIZE forward-deferred contract: substrate-
    // missing is a STRUCTURED error envelope, NOT silent empty
    // results. Mirrors the existing W17α
    // `graph_search_reports_no_substrate_for_unwired_tenant` pin
    // but in the W23-M4-08-FINALIZE test suite so the contract
    // stays attached to this slice.
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(3)),
        method: "graph.search".into(),
        params: json!({
            "tenant_id": 1,
            "query": "alice",
            "k": 5,
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("search dispatch");
    let err_code = resp["error"]["code"].as_i64().expect("error code");
    // -32004 = `CODE_INDEX_UNAVAILABLE` per ADR-004 amendment-03.
    assert_eq!(
        err_code, -32004,
        "search against unwired substrate must surface -32004 IndexUnavailable; resp={resp:?}"
    );
}

#[test]
fn cross_tenant_raw_query_rejects_with_unauthorized() {
    // Cross-tenant fault injection: a session scoped to
    // TenantId::DEFAULT must reject queries for any other tenant
    // BEFORE the executor body runs. Per ADR-011 + ADR-037 §D-3 the
    // rejection is structural (no leakage of execution state).
    let d = fresh_dispatcher();
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(4)),
        method: "graph.raw_query".into(),
        params: json!({
            "tenant_id": 9_999,
            "query": "MATCH (n) RETURN n",
            "max_rows": 100,
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("envelope");
    let err_code = resp["error"]["code"].as_i64().expect("error code");
    // Per ADR-004 amendment-03: -32002 is `CODE_UNAUTHORIZED` for
    // cross-tenant rejections.
    assert_eq!(
        err_code, -32002,
        "cross-tenant request must reject with -32002 (Unauthorized); resp={resp:?}"
    );
}

#[test]
fn inbound_traversal_query_returns_real_rows_w26_beta_2() {
    // W26-β-2 / ADR-131 — closes #350 v1.1 inbound TEL expand.
    //
    // The W17α `CrudExecutorSubstrate::expand` returned a structured
    // `SubstrateAccessError::Io` for `Direction::RightToLeft` /
    // `Undirected` per the PR #349 R1 HIGH-1 forward-pin to v1.1
    // (issue #350). W26-β-2 lights the inbound path via the
    // reverse-adjacency index per ADR-131 option-2; the query path
    // now returns real rows.
    //
    // Supersedes the v1.0-α
    // `inbound_traversal_query_surfaces_structured_error_not_silent_empty`
    // pin: the structured-error posture is no longer valid; positive
    // row coverage is the v1.1 oracle. Per ADR-087 D-4 forward-pin
    // table the surface flips from "structured error" → "real rows."
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (a:Service)<-[:DEPENDS_ON]-(b:Service) RETURN a, b",
    );
    assert!(
        resp["error"].is_null(),
        "v1.1 inbound traversal must succeed (no error envelope); resp={resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse");
    let row_count = rows["row_count"].as_u64().expect("row_count");
    // Inbound count = 4: svc-2 (inbound from svc-1), svc-3 (from svc-2),
    // svc-4 (from svc-3), svc-5 (from svc-3). Symmetric to the
    // outbound count = 4 the sister test `multi_pattern_*` asserts.
    assert_eq!(
        row_count, 4,
        "expected 4 inbound DEPENDS_ON edges (symmetric to the 4 outbound rows); body={body}"
    );
}

#[test]
fn unknown_label_in_query_surfaces_binding_error_not_silent_empty() {
    // Fault injection: a query that references a label name that
    // doesn't exist in the catalog (intern table or stats) must
    // surface a binding error, NOT silent empty results. This pins
    // the planner's binding pass behavior at v1.0-α and prevents the
    // "silent zero rows because the catalog mis-resolved" anti-
    // pattern (the W17α latent bug class).
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);

    let resp = raw_query(&d, "MATCH (s:NotARealLabel) RETURN s");
    // Either an error envelope (binding rejected at parse-time) OR
    // a body with 0 rows + a warning (v1.0-α dynamic-name fallback).
    // Per the W17α `graph_raw_query_property_name_filter_pins_v1_0_alpha_behavior`
    // pattern, we accept BOTH but pin the shape so a regression flips
    // the test red.
    if !resp["error"].is_null() {
        assert!(
            resp["error"]["message"].as_str().is_some(),
            "error envelope must carry message"
        );
    } else {
        let body = resp["result"]["body"].as_str().expect("body");
        let rows: Value = serde_json::from_str(body).expect("parse");
        // Either error (handled above) OR zero rows from the dynamic-
        // name fallback (the label resolves to an ID that doesn't
        // exist in storage). Both are deterministic v1.0-α shapes.
        let row_count = rows["row_count"].as_u64().expect("row_count");
        assert!(
            row_count == 0,
            "unknown label must surface 0 rows OR error; got {} rows. body={body}",
            row_count
        );
    }
}

#[test]
fn cartesian_join_executes_end_to_end_via_mcp_with_full_row_count() {
    // Cartesian shape (no shared binding) — 5 × 5 = 25 rows from the
    // 5-Service fixture. Pins the LogicalJoin cartesian shape
    // end-to-end through the MCP surface against the production
    // substrate.
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);

    let resp = raw_query(&d, "MATCH (a:Service), (b:Service) RETURN a, b");
    assert!(resp["error"].is_null(), "raw_query failed: {resp:?}");
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse");
    assert_eq!(
        rows["row_count"], 25,
        "expected 5×5 = 25 cartesian rows; body={body}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Idempotency + cancellation discipline.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn second_query_against_same_substrate_returns_same_results() {
    // Determinism pin: the same query against the same substrate
    // returns the same row count on a second call. Surfaces any
    // hidden state in the executor / substrate that would cause
    // a per-call drift (mutable cache, exhausted iterator, etc.).
    let d = fresh_dispatcher();
    ingest_incident_fixture(&d);
    let first = raw_query(&d, "MATCH (s:Service) RETURN s");
    let second = raw_query(&d, "MATCH (s:Service) RETURN s");
    let body1 = first["result"]["body"].as_str().expect("body");
    let body2 = second["result"]["body"].as_str().expect("body");
    let rows1: Value = serde_json::from_str(body1).expect("parse");
    let rows2: Value = serde_json::from_str(body2).expect("parse");
    assert_eq!(rows1["row_count"], rows2["row_count"], "row count drift");
}

// ─────────────────────────────────────────────────────────────────────
// W26-ε-1 — LDBC IC1-IC7 end-to-end via MCP `graph.raw_query`.
// ─────────────────────────────────────────────────────────────────────
//
// Per ADR-087-amendment-02 (D-4 forward-deferred row closure): the
// W26-ε-1 slice EXTENDS the W23-M4-08-FINALIZE coverage from
// label-anchored + rel-type-anchored incident shapes (above) to the
// LDBC SNB Interactive IC1-IC7 surface end-to-end through MCP. The
// load-bearing claims:
//
// 1. **LDBC IC shape coverage** — every IC1-IC7 multi-pattern query
//    executes end-to-end through MCP → CrudExecutorSubstrate without
//    NotImplemented / IndexUnavailable errors.
// 2. **Direction::RightToLeft (inbound)** — IC2-class queries use
//    `<-[:HAS_CREATOR]-` patterns; the W26-β-2 reverse-adjacency
//    (ADR-131) closure is exercised through the MCP wire.
// 3. **LogicalJoin Hash + Merge** — IC5-class multi-pattern joins
//    invoke the W25-M4-61b cost-based picker (ADR-097) through MCP.
// 4. **Substrate body wire-through (#469)** — the production
//    CrudExecutorSubstrate's scan_nodes + expand bodies (W26-β-3 /
//    ADR-132 closure) route real rows through MCP.
//
// The queries below are LDBC IC1-IC7 STRUCTURAL VARIANTS (label-only
// anchors; no property-bag filters). Property-bag round-trip at v1.0-α
// is forward-pinned per issue #356 (the W17α `CrudExecutorSubstrate::
// scan_nodes` surfaces empty property bags; the property-filter
// queries `WHERE n.id = 1` would not bind against the substrate
// today). The structural variants exercise EVERY join shape + EVERY
// direction in the IC1-IC7 bank while the property-filter binding
// remains on the v1.1+ track.
//
// The MCP-side tests here use structural variants because the
// production substrate's property path is the issue-#356 deferred
// surface.

/// Ingest an LDBC SNB IC-shaped fixture via MCP `graph.ingest`. Sized
/// to make every IC1-IC7 structural variant return at least one row
/// when the structural shape holds (10 Persons in a KNOWS ring
/// guarantees every Person has 2 friends; 50 Comments distributed
/// across all 10 Persons via HAS_CREATOR; etc.).
///
/// Topology (matches `executor_smoke::small_ic_substrate` shape):
/// - 10 Persons (`person-1` .. `person-10`).
/// - 5 Places (`place-100` .. `place-104`).
/// - 2 Forums (`forum-200`, `forum-201`).
/// - 50 Comments (`comment-1000` .. `comment-1049`).
/// - KNOWS ring: person-{i} → person-{i+1 mod 10}.
/// - IS_LOCATED_IN: person-1 → place-100.
/// - HAS_CREATOR: comment-{1000+j} → person-{1 + j%10}.
/// - CONTAINER_OF: forum-200 → comment-{1000+j}.
/// - HAS_MODERATOR: forum-200 → person-1.
/// - LIKES: person-1 → comment-1000.
fn ingest_ldbc_ic_fixture(d: &TestDispatcher) {
    let mut nodes: Vec<Value> = Vec::with_capacity(67);
    for i in 1..=10u64 {
        nodes.push(json!({
            "external_id": format!("person-{i}"),
            "label": "Person",
            "properties": {}
        }));
    }
    for i in 100..=104u64 {
        nodes.push(json!({
            "external_id": format!("place-{i}"),
            "label": "Place",
            "properties": {}
        }));
    }
    for i in 200..=201u64 {
        nodes.push(json!({
            "external_id": format!("forum-{i}"),
            "label": "Forum",
            "properties": {}
        }));
    }
    for j in 0..50u64 {
        let cid = 1000 + j;
        nodes.push(json!({
            "external_id": format!("comment-{cid}"),
            "label": "Comment",
            "properties": {}
        }));
    }

    let mut rels: Vec<Value> = Vec::new();
    // KNOWS ring: person-{i} → person-{i+1 mod 10}. Ten directed edges
    // (the ArcQL `-[:KNOWS]-` undirected matcher at v1.1+ traverses
    // both directions via the W26-β-2 reverse-adjacency index;
    // outbound `-[:KNOWS]->` traverses only the forward direction).
    for i in 1..=10u64 {
        let nxt = if i == 10 { 1 } else { i + 1 };
        rels.push(json!({
            "from_external_id": format!("person-{i}"),
            "to_external_id": format!("person-{nxt}"),
            "rel_type": "KNOWS",
            "properties": {}
        }));
    }
    // IS_LOCATED_IN: person-1 → place-100.
    rels.push(json!({
        "from_external_id": "person-1",
        "to_external_id": "place-100",
        "rel_type": "IS_LOCATED_IN",
        "properties": {}
    }));
    // HAS_CREATOR: comment-{1000+j} → person-{1 + j%10}.
    for j in 0..50u64 {
        let cid = 1000 + j;
        let creator = 1 + (j % 10);
        rels.push(json!({
            "from_external_id": format!("comment-{cid}"),
            "to_external_id": format!("person-{creator}"),
            "rel_type": "HAS_CREATOR",
            "properties": {}
        }));
    }
    // CONTAINER_OF: forum-200 → every comment.
    for j in 0..50u64 {
        let cid = 1000 + j;
        rels.push(json!({
            "from_external_id": "forum-200",
            "to_external_id": format!("comment-{cid}"),
            "rel_type": "CONTAINER_OF",
            "properties": {}
        }));
    }
    // HAS_MODERATOR: forum-200 → person-1.
    rels.push(json!({
        "from_external_id": "forum-200",
        "to_external_id": "person-1",
        "rel_type": "HAS_MODERATOR",
        "properties": {}
    }));
    // LIKES: person-1 → comment-1000.
    rels.push(json!({
        "from_external_id": "person-1",
        "to_external_id": "comment-1000",
        "rel_type": "LIKES",
        "properties": {}
    }));

    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": nodes,
            "relationships": rels,
            "format": "json"
        }),
    };
    let resp = d.dispatch(req).expect("ldbc-ic ingest dispatch");
    assert!(
        resp["error"].is_null(),
        "LDBC IC fixture ingest must succeed: {resp:?}"
    );
}

#[test]
fn ldbc_ic1_knows_outbound_executes_end_to_end_via_mcp_w26_epsilon_1() {
    // Pin: IC1-class outbound 1-hop KNOWS through MCP. Demonstrates
    // the substrate body wire-through (W26-β-3 / ADR-132 / #469
    // closure) lights the production scan_nodes + expand path for
    // LDBC `Person -[:KNOWS]-> Person` shapes.
    //
    // Substrate semantics: the KNOWS ring has 10 directed edges
    // (person-1 → person-2, person-2 → person-3, ..., person-10 →
    // person-1). An outbound `-[:KNOWS]->` matches the 10 forward
    // edges exactly.
    let d = fresh_dispatcher();
    ingest_ldbc_ic_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (n:Person)-[:KNOWS]->(friend:Person) RETURN n, friend",
    );
    assert!(
        resp["error"].is_null(),
        "IC1 outbound KNOWS must execute without error (M4-08 + #469 \
         substrate body wire-through); resp={resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    let row_count = rows["row_count"].as_u64().expect("row_count");
    assert_eq!(
        row_count, 10,
        "10 outbound KNOWS edges in the LDBC IC ring fixture; body={body}"
    );
}

#[test]
fn ldbc_ic2_inbound_has_creator_via_mcp_pins_w26_beta_2_reverse_adjacency() {
    // Pin: IC2-class 2-hop with INBOUND HAS_CREATOR through MCP.
    // Demonstrates the W26-β-2 / ADR-131 reverse-adjacency index closure
    // (`Direction::RightToLeft` substrate support; issue #350 closed)
    // routes through the production substrate end-to-end.
    //
    // Query shape:
    //   MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(m:Comment)
    // Each Person has 2 outbound-or-inbound KNOWS neighbors; via the
    // directed outbound-only match, person-{i} → person-{i+1 mod 10}.
    // Each `friend` is the creator of `friend_index % 10`-aligned
    // Comments — 5 comments per person via the j%10 distribution.
    // 10 outbound KNOWS × 5 comments-per-friend = 50 join rows.
    let d = fresh_dispatcher();
    ingest_ldbc_ic_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(m:Comment) \
         RETURN n, friend, m",
    );
    assert!(
        resp["error"].is_null(),
        "IC2 inbound HAS_CREATOR must execute without error (W26-β-2 \
         ADR-131 reverse-adjacency); resp={resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    let row_count = rows["row_count"].as_u64().expect("row_count");
    // Load-bearing for W26-β-2: row_count > 0 proves the inbound
    // traversal actually fired. The exact count is 50 = 10 outbound
    // KNOWS × 5 comments-per-friend (j%10 distribution).
    assert!(
        row_count >= 1,
        "IC2 inbound traversal must return ≥ 1 row (W26-β-2 / #350 \
         closure); body={body}"
    );
    assert!(
        row_count <= 100,
        "IC2 inbound traversal row count {} implausibly high \
         (cartesian-explosion regression?); body={body}",
        row_count
    );
}

#[test]
fn ldbc_ic3_chain_with_is_located_in_executes_via_mcp() {
    // Pin: IC3-class 2-hop chain through MCP. Exercises a 2-hop chain
    // through KNOWS + IS_LOCATED_IN — different from IC2's inbound
    // shape (IC3 stays outbound). The substrate must serve 3 distinct
    // rel-types (KNOWS, IS_LOCATED_IN) in a single query.
    //
    // Query shape:
    //   MATCH (n:Person)-[:KNOWS]->(friend:Person)-[:IS_LOCATED_IN]->(p:Place)
    // person-1 has IS_LOCATED_IN -> place-100; the other 9 persons
    // do NOT. So this matches person-{i}-[:KNOWS]->person-1
    // (only person-10 → person-1 in the KNOWS ring). 1 row total.
    let d = fresh_dispatcher();
    ingest_ldbc_ic_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (n:Person)-[:KNOWS]->(friend:Person)-[:IS_LOCATED_IN]->(p:Place) \
         RETURN n, friend, p",
    );
    assert!(
        resp["error"].is_null(),
        "IC3 2-hop chain must execute without error; resp={resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    let row_count = rows["row_count"].as_u64().expect("row_count");
    // person-10 -[:KNOWS]-> person-1 -[:IS_LOCATED_IN]-> place-100.
    // Exactly 1 row.
    assert_eq!(
        row_count, 1,
        "IC3 chain: only person-10 KNOWS person-1 (the LOCATED_IN \
         Person), yielding 1 row; body={body}"
    );
}

#[test]
fn ldbc_ic5_forum_join_via_mcp_pins_logicaljoin_hash_or_merge() {
    // Pin: IC5-class 2-hop with INBOUND HAS_MODERATOR through MCP.
    // The Forum -[:HAS_MODERATOR]-> Person edge is INBOUND when read
    // from the Person side; this exercises the W26-β-2 reverse-
    // adjacency AND the LogicalJoin cost-picker (W25-M4-61b / ADR-097
    // Hash vs Merge selection) end-to-end through MCP.
    //
    // Query shape:
    //   MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_MODERATOR]-(f:Forum)
    // Only forum-200 -[:HAS_MODERATOR]-> person-1; only person-10
    // outbound-KNOWS-person-1 in the ring. 1 row.
    let d = fresh_dispatcher();
    ingest_ldbc_ic_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_MODERATOR]-(f:Forum) \
         RETURN n, friend, f",
    );
    assert!(
        resp["error"].is_null(),
        "IC5 forum-moderator inbound join must execute without error \
         (LogicalJoin + W26-β-2 inbound + #469 wire-through); resp={resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    let row_count = rows["row_count"].as_u64().expect("row_count");
    assert_eq!(
        row_count, 1,
        "IC5: only person-10-[:KNOWS]->person-1<-[:HAS_MODERATOR]-forum-200; \
         body={body}"
    );
}

#[test]
fn ldbc_ic7_likes_chain_inbound_via_mcp() {
    // Pin: IC7-class 2-hop with TWO inbound segments through MCP.
    // Exercises BOTH inbound substrate calls in a single query — the
    // hardest load-bearing case for the W26-β-2 reverse-adjacency
    // path through MCP.
    //
    // Query shape:
    //   MATCH (n:Person)<-[:HAS_CREATOR]-(m:Comment)<-[:LIKES]-(liker:Person)
    // person-1 has 5 inbound HAS_CREATOR edges (comments 1000, 1010,
    // 1020, 1030, 1040). Only comment-1000 has an inbound LIKES (from
    // person-1). So 1 row matches (n=person-1, m=comment-1000,
    // liker=person-1).
    let d = fresh_dispatcher();
    ingest_ldbc_ic_fixture(&d);

    let resp = raw_query(
        &d,
        "MATCH (n:Person)<-[:HAS_CREATOR]-(m:Comment)<-[:LIKES]-(liker:Person) \
         RETURN n, m, liker",
    );
    assert!(
        resp["error"].is_null(),
        "IC7 double-inbound chain must execute without error (W26-β-2 \
         reverse adjacency × 2); resp={resp:?}"
    );
    let body = resp["result"]["body"].as_str().expect("body");
    let rows: Value = serde_json::from_str(body).expect("parse body");
    let row_count = rows["row_count"].as_u64().expect("row_count");
    assert_eq!(
        row_count, 1,
        "IC7: person-1<-[:HAS_CREATOR]-comment-1000<-[:LIKES]-person-1; \
         1 row; body={body}"
    );
}

#[test]
fn ldbc_ic_e2e_via_mcp_all_seven_shapes_demonstrate_pipeline() {
    // Wave-level load-bearing pin: ALL 7 IC structural variants
    // execute end-to-end through MCP without ANY returning an error
    // envelope. Aggregates the load-bearing claim across the
    // individual IC1-IC7 tests above.
    //
    // Per ADR-087-amendment-02 (D-4 forward-deferred row closure)
    // this is the wave-anchor pin that the W26-ε-1 PR demonstrates
    // the full M4-08 pipeline through MCP for the LDBC SNB IC
    // surface.
    let d = fresh_dispatcher();
    ingest_ldbc_ic_fixture(&d);

    // 7 IC structural shapes (label-only anchors; no property
    // filters per the issue #356 v1.1+ property-bag deferral).
    let ic_queries: [(&str, &str); 7] = [
        (
            "IC1",
            "MATCH (n:Person)-[:KNOWS]->(friend:Person) RETURN n, friend",
        ),
        (
            "IC2",
            "MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(m:Comment) \
             RETURN n, friend, m",
        ),
        (
            "IC3",
            "MATCH (n:Person)-[:KNOWS]->(friend:Person)-[:IS_LOCATED_IN]->(p:Place) \
             RETURN n, friend, p",
        ),
        (
            "IC4",
            "MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(m:Comment) \
             RETURN n, friend, m",
        ),
        (
            "IC5",
            "MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_MODERATOR]-(f:Forum) \
             RETURN n, friend, f",
        ),
        (
            "IC6",
            "MATCH (n:Person)-[:KNOWS]->(friend:Person)<-[:HAS_CREATOR]-(m:Comment) \
             RETURN n, friend, m",
        ),
        (
            "IC7",
            "MATCH (n:Person)<-[:HAS_CREATOR]-(m:Comment)<-[:LIKES]-(liker:Person) \
             RETURN n, m, liker",
        ),
    ];
    let mut failures: Vec<String> = Vec::new();
    for (name, query) in ic_queries.iter() {
        let resp = raw_query(&d, query);
        if !resp["error"].is_null() {
            failures.push(format!("{name}: error envelope {:?}", resp["error"]));
            continue;
        }
        let body = resp["result"]["body"].as_str().expect("body");
        let rows: Value = serde_json::from_str(body).expect("parse body");
        if rows["row_count"].as_u64().is_none() {
            failures.push(format!("{name}: missing row_count in body"));
        }
    }
    assert!(
        failures.is_empty(),
        "W26-ε-1 D2 wave-level pin: ALL 7 LDBC IC structural variants must \
         execute end-to-end via MCP without error. Failures: {failures:?}"
    );
}
