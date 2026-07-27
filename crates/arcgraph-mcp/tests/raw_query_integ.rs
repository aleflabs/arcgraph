//! W16ζ M5-11 integration tests — end-to-end `graph.raw_query`
//! through the JSON-RPC dispatcher surface.
//!
//! Acceptance per the W16ζ spawn prompt + ADR-004 amendment-03 §D-1:
//!   1. Power-scope session accepts; result envelope carries rows.
//!   2. Read-scope session rejects with -32008 (Forbidden) and
//!      `data.required_scope == "arcgraph.power"`.
//!   3. Cross-tenant request rejects with -32002 (Unauthorized) BEFORE
//!      the scope check fires — cross-tenant probes don't leak scope
//!      information.
//!   4. Oversized query rejects with -32602 (InvalidParams) BEFORE the
//!      executor body runs — `MAX_RAW_QUERY_BYTES = 1 MiB` cap per
//!      ADR-004 amendment-03 §D-1 point 2.
//!   5. `max_rows` truncation: executor returns >max_rows rows; result
//!      envelope sets `truncated: true` and carries exactly `max_rows`.
//!   6. Executor `QueryError` surfaces as -32005 with the message
//!      propagated through the JSON-RPC `data` field per the W13δ
//!      codec-local error-translation discipline.
//!
//! All six tests drive the dispatcher through [`handle_raw_envelope`]
//! — the same entry point [`arcgraph_mcp::serve_stdio`] /
//! [`arcgraph_mcp::serve_http`] use — so the integration coverage
//! matches the production transport's wire shape per
//! `feedback_review_oracle_relaxations.md` (NOT a pre-seeded fixture).

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::tools::explore::Neighborhood;
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestSummary};
use arcgraph_mcp::tools::inspect::{NodeInspection, NodeInspector};
use arcgraph_mcp::tools::raw_query::{MAX_RAW_QUERY_BYTES, RawQueryExecutor, RawQueryRows};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo, SchemaProvider,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, SearchHit};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, SessionScope, handle_raw_envelope,
};
use arcgraph_query::CancellationToken;
use serde_json::{Value, json};

// ─────────────────────────────────────────────────────────────────────
// Fixture: RawQueryFixture — returns 5 fixture rows for any query;
// records the requested max_rows for the truncation assertion.
// ─────────────────────────────────────────────────────────────────────

struct RawQueryFixture {
    tenant: TenantId,
    /// When `Some`, every call returns this error (used to test the
    /// QueryError path).
    forced_error: Option<&'static str>,
}

impl RawQueryExecutor for RawQueryFixture {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        max_rows: u32,
        cancel: &CancellationToken,
    ) -> Result<RawQueryRows, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        if let Some(detail) = self.forced_error {
            return Err(MCPError::QueryError(detail.into()));
        }
        // Five fixture rows; the executor pre-truncates to max_rows so
        // the dispatcher boundary's defensive truncation is exercised
        // when the caller's max_rows < 5.
        let all = vec![
            json!([1u64, "Alice", 30]),
            json!([2u64, "Bob", 28]),
            json!([3u64, "Carol", 35]),
            json!([4u64, "Dave", 41]),
            json!([5u64, "Eve", 24]),
        ];
        let truncated = all.len() > max_rows as usize;
        let mut emitted = all;
        if truncated {
            emitted.truncate(max_rows as usize);
        }
        let row_count = emitted.len();
        Ok(RawQueryRows {
            columns: Some(vec!["id".into(), "name".into(), "age".into()]),
            rows: emitted,
            row_count,
            truncated,
            // Read-side fixture — ADR-153 §D-2 says read queries
            // return a zero WriteSummary.
            writes: arcgraph_mcp::tools::raw_query::WriteSummary::default(),
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Minimal stubs for the unrelated adapters the dispatcher requires.
// ─────────────────────────────────────────────────────────────────────

struct DummySchema(TenantId);
impl SchemaProvider for DummySchema {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels: vec![LabelInfo {
                name: "Person".into(),
                cardinality: None,
                properties: vec![],
            }],
            rel_types: vec![RelTypeInfo {
                name: "KNOWS".into(),
                cardinality: None,
            }],
            indexes: vec![IndexDescriptor {
                kind: IndexKind::Vector,
                available: true,
            }],
            total_node_count: None,
            total_rel_count: None,
        })
    }
}

struct DummyInspect(TenantId);
impl NodeInspector for DummyInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(NodeInspection {
            id: node_id,
            label: Some("Person".into()),
            properties: BTreeMap::new(),
            neighbors: vec![],
        })
    }
}

struct DummyExplore(TenantId);
impl NeighborhoodExplorer for DummyExplore {
    fn explore(
        &self,
        tenant: TenantId,
        _seed: u64,
        _max_depth: u32,
        _rel_filter: Option<&[String]>,
        _direction: arcgraph_mcp::tools::explore::ExploreDirection,
        _cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub explore not exercised by W16ζ raw_query integ".into(),
        ))
    }
}

struct DummySearch(TenantId);
impl HybridSearcher for DummySearch {
    fn available_substrates(
        &self,
        tenant: TenantId,
        _cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(AvailableSubstrates {
            vector: false,
            bm25: false,
        })
    }
    fn search(
        &self,
        tenant: TenantId,
        _q: &str,
        _v: Option<&[f32]>,
        _k: u32,
        _cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub search not exercised by W16ζ raw_query integ".into(),
        ))
    }
}

struct DummyIngest(TenantId);
impl IngestProvider for DummyIngest {
    fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub ingest not exercised by W16ζ raw_query integ".into(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────
// Dispatcher builder
// ─────────────────────────────────────────────────────────────────────

type W16Dispatcher =
    Dispatcher<DummySchema, DummyInspect, DummyExplore, DummySearch, DummyIngest, RawQueryFixture>;

fn build_dispatcher(
    tenant: u64,
    scope: SessionScope,
    forced_error: Option<&'static str>,
) -> W16Dispatcher {
    let t = TenantId::new(tenant);
    Dispatcher::with_session_scope(
        t,
        scope,
        Arc::new(DummySchema(t)),
        Arc::new(DummyInspect(t)),
        Arc::new(DummyExplore(t)),
        Arc::new(DummySearch(t)),
        Arc::new(DummyIngest(t)),
        Arc::new(RawQueryFixture {
            tenant: t,
            forced_error,
        }),
    )
}

// ─────────────────────────────────────────────────────────────────────
// Test 1 — power-scope session accepts; result envelope carries rows.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_raw_query_power_scope_returns_rows_on_fixture() {
    let d = build_dispatcher(7, SessionScope::Power, None);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 7,
            "query": "MATCH (p:Person) RETURN p.id, p.name, p.age",
            "max_rows": 5,
            "format": "json"
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(resp["id"], 1);
    let body = resp["result"]["body"]
        .as_str()
        .expect("raw_query result.body is a string");
    assert!(body.contains("\"row_count\":5"), "5 rows: body={body}");
    assert!(body.contains("Alice"), "row 1 visible");
    assert!(body.contains("Eve"), "row 5 visible");
    assert!(body.contains("\"truncated\":false"), "not truncated");
    // Columns echoed.
    assert!(body.contains("\"id\""), "column id");
    assert!(body.contains("\"age\""), "column age");
}

// ─────────────────────────────────────────────────────────────────────
// Test 2 — read-scope session rejects with -32008 + arcgraph.power slug.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_raw_query_read_scope_rejects_with_forbidden_minus_32008() {
    // W16ζ M5-11 hard requirement per ADR-004 amendment-03 §D-1 +
    // design-v2 §9.5 JD stress test line 682.
    let d = build_dispatcher(7, SessionScope::Read, None);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 7,
            "query": "MATCH (n) RETURN n"
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(resp["id"], 2);
    assert_eq!(resp["error"]["code"], -32008);
    assert_eq!(resp["error"]["message"], "forbidden");
    // The data slot carries the required-scope slug per design-v2
    // §9.4 nomenclature.
    assert_eq!(resp["error"]["data"]["required_scope"], "arcgraph.power");
}

// ─────────────────────────────────────────────────────────────────────
// Test 3 — cross-tenant guard fires BEFORE the scope check.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_raw_query_cross_tenant_rejects_before_scope_check() {
    // The dispatcher is bound to tenant 7 with read scope. A request
    // for tenant 8 MUST surface -32002 (Unauthorized), NOT -32008
    // (Forbidden). The cross-tenant guard runs FIRST so a probe from
    // a different tenant cannot determine the scope of THIS session
    // (which would leak a scope-vs-no-scope side-channel).
    let d = build_dispatcher(7, SessionScope::Read, None);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 8,
            "query": "MATCH (n) RETURN n"
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(
        resp["error"]["code"], -32002,
        "cross-tenant guard runs BEFORE scope check"
    );
    assert_eq!(resp["error"]["message"], "unauthorized");
}

// ─────────────────────────────────────────────────────────────────────
// Test 4 — oversized query rejects with -32602 BEFORE executor runs.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_raw_query_oversized_query_rejects_at_minus_32602() {
    // Per ADR-004 amendment-03 §D-1 point 2 + per
    // feedback_security_class_first_network_surface.md. A query
    // length of MAX_RAW_QUERY_BYTES + 1 MUST reject BEFORE the
    // executor body runs. We use the fixture's forced_error to
    // confirm the executor was NOT called (if the executor were
    // called, it would return Cancelled — but the cap fires earlier
    // at -32602).
    let d = build_dispatcher(
        7,
        SessionScope::Power,
        Some("executor should not be called"),
    );
    let oversized: String = "x".repeat(MAX_RAW_QUERY_BYTES + 1);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 4,
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 7,
            "query": oversized
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(resp["error"]["code"], -32602);
    let detail = resp["error"]["data"].as_str().unwrap_or("");
    assert!(
        detail.contains("exceeds cap"),
        "detail names the cap: {detail}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 5 — max_rows truncation across the dispatcher boundary.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_raw_query_max_rows_2_truncates_5_row_fixture() {
    // The executor returns 5 rows; the caller asks for max_rows=2.
    // The result envelope MUST carry exactly 2 rows and
    // truncated=true. This pins both:
    //   - the executor honors max_rows (pre-truncates to 2);
    //   - the MCP boundary's defensive truncation (in
    //     raw_query_tool) is consistent with the executor's pre-
    //     truncation (no double-truncate gymnastics; the wire shape
    //     is uniformly truncated=true).
    let d = build_dispatcher(7, SessionScope::Power, None);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 5,
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 7,
            "query": "MATCH (p:Person) RETURN p",
            "max_rows": 2,
            "format": "json"
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    let body = resp["result"]["body"].as_str().expect("body string");
    assert!(body.contains("\"row_count\":2"), "row_count=2: body={body}");
    assert!(body.contains("\"truncated\":true"));
    assert!(body.contains("Alice"));
    assert!(body.contains("Bob"));
    assert!(!body.contains("Carol"), "row 3 dropped");
}

// ─────────────────────────────────────────────────────────────────────
// Test 6 — executor QueryError surfaces as -32005 with data propagated.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_raw_query_executor_query_error_routes_to_minus_32005() {
    // The W13δ codec-local error-translation discipline pins
    // ExecutionError::Plan (ArcQL parse / bind / type-check / cross-
    // substrate / lowering / NotImplemented) → MCPError::QueryError
    // → JSON-RPC -32005. We stub the executor to return QueryError
    // and assert the dispatcher envelope is well-formed.
    let d = build_dispatcher(7, SessionScope::Power, Some("unknown label XYZ"));
    let env = json!({
        "jsonrpc": "2.0",
        "id": 6,
        "method": "graph.raw_query",
        "params": {
            "tenant_id": 7,
            "query": "MATCH (n:XYZ) RETURN n"
        }
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert_eq!(resp["error"]["code"], -32005);
    // Data slot carries the inner detail so MCP clients can render
    // the ArcQL diagnostic without parsing the message string.
    let data = resp["error"]["data"].as_str().unwrap_or("");
    assert!(data.contains("unknown label XYZ"), "data: {data}");
}

// ─────────────────────────────────────────────────────────────────────
// Test 7 — read-scope session ALLOWS Tier-1 read tools (regression
// pin: the W16ζ slice MUST NOT break Tier-1 access for read sessions).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn integ_read_scope_session_still_admits_tier_1_schema() {
    // Sanity pin: a read-scope dispatcher must still serve Tier-1
    // tools (graph.schema in this case). The W16ζ scope check is
    // load-bearing ONLY on graph.raw_query — Tier-1 tools route
    // through the per-tenant + rate-limit gates, NOT the scope gate
    // (until M5-03 lights scope on every tool).
    let d = build_dispatcher(7, SessionScope::Read, None);
    let env = json!({
        "jsonrpc": "2.0",
        "id": 7,
        "method": "graph.schema",
        "params": {"tenant_id": 7}
    });
    let resp = handle_raw_envelope(&d, env).expect("response present");
    assert!(
        resp.get("result").is_some(),
        "read-scope session must admit graph.schema (Tier-1); envelope: {resp}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Helper — ensure no warnings on `Value` import (proxy for the
// dependency surface used by some tests; here unused intentionally).
// ─────────────────────────────────────────────────────────────────────

#[allow(dead_code)]
fn _value_import_used(_v: Value) {}
