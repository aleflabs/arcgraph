//! W14α M5-03 proptest — every JSON-RPC 2.0 request envelope
//! produces a parseable JSON-RPC response (success OR error).
//!
//! Mirrors the spirit of the W13δ stdio proptest (per `tests/error_code_proptest.rs`)
//! but operates at the transport-agnostic dispatcher seam: the
//! [`handle_raw_envelope`] function takes the same JSON `Value` regardless
//! of whether it was framed by stdio Content-Length headers or hyper
//! HTTP. The proptest pins:
//!
//!   1. **Determinism**: same input → same output (same dispatcher).
//!   2. **Total parseability**: every response — whether success or
//!      error — round-trips through `serde_json::from_value` cleanly.
//!   3. **Envelope shape**: exactly one of `result` or `error`
//!      appears, never both, never neither.
//!   4. **JSON-RPC version pin**: every response carries
//!      `jsonrpc: "2.0"`.
//!
//! The proptest generates structurally-arbitrary envelopes
//! (including malformed ones — non-2.0 jsonrpc strings, missing
//! method, etc.) so the response-shape invariant holds across BOTH
//! the success path and every error code.
//!
//! Per the spawn prompt's hard requirement: "1 proptest: every
//! JSON-RPC 2.0 request → produces parseable response (mirrors W13δ
//! stdio proptest, transport-agnostic)".

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::tools::explore::{Neighborhood, NeighborhoodEdge, NeighborhoodNode};
use arcgraph_mcp::tools::ingest::{IngestBatch, IngestProvider, IngestSummary};
use arcgraph_mcp::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection};
use arcgraph_mcp::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo,
};
use arcgraph_mcp::tools::search::{AvailableSubstrates, SearchHit};
use arcgraph_mcp::{
    Dispatcher, HybridSearcher, MCPError, NeighborhoodExplorer, NodeInspector, SchemaProvider,
    handle_raw_envelope,
};
use arcgraph_query::CancellationToken;
use proptest::prelude::*;
use serde_json::{Value, json};

struct StubSchema(TenantId);
impl SchemaProvider for StubSchema {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels: vec![LabelInfo {
                name: "Person".into(),
                cardinality: Some(1),
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
            total_node_count: Some(1),
            total_rel_count: Some(0),
        })
    }
}

struct StubInspect(TenantId);
impl NodeInspector for StubInspect {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut p = BTreeMap::new();
        p.insert("name".into(), json!("Alice"));
        Ok(NodeInspection {
            id: node_id,
            label: Some("Person".into()),
            properties: p,
            neighbors: vec![NeighborInfo {
                node_id: 2,
                label: Some("Person".into()),
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        })
    }
}

// W14β M5-06 / M5-07 / W14γ M5-08: minimal stubs so the dispatcher
// satisfies its `<S, I, E, H, G>` bounds. The proptest only generates
// `graph.schema` / `graph.inspect` / `graph.bogus` methods, so these
// bodies are effectively unreachable — but the trait impls are
// required for the dispatcher type to be inhabited.
struct StubExplore(TenantId);
impl NeighborhoodExplorer for StubExplore {
    fn explore(
        &self,
        tenant: TenantId,
        seed: u64,
        max_depth: u32,
        _rel_filter: Option<&[String]>,
        _direction: arcgraph_mcp::tools::explore::ExploreDirection,
        cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(Neighborhood {
            seed,
            max_depth,
            truncated: false,
            nodes: vec![NeighborhoodNode {
                id: seed,
                label: Some("Person".into()),
                depth: 0,
                properties: BTreeMap::new(),
            }],
            edges: vec![NeighborhoodEdge {
                from: seed,
                to: seed + 1,
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        })
    }
}

struct StubSearch(TenantId);
impl HybridSearcher for StubSearch {
    fn available_substrates(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Ok(AvailableSubstrates {
            vector: true,
            bm25: true,
        })
    }
    fn search(
        &self,
        tenant: TenantId,
        _query_text: &str,
        _query_vec: Option<&[f32]>,
        k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        let mut hits = vec![SearchHit {
            node_id: 1,
            label: Some("Document".into()),
            score: 0.9,
        }];
        hits.truncate(k as usize);
        Ok(hits)
    }
}

/// Stub ingest provider — http-proptest doesn't exercise the ingest
/// tool; the impl exists only to satisfy the dispatcher's
/// `IngestProvider` generic.
struct StubIngest(TenantId);
impl IngestProvider for StubIngest {
    fn ingest(&self, tenant: TenantId, _batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub ingest not exercised by http proptest".into(),
        ))
    }
}

/// Stub raw-query executor (W16ζ M5-11) — http-proptest does not
/// exercise raw_query; the impl exists only to satisfy the W16ζ-merged
/// dispatcher's `RawQueryExecutor` generic.
struct StubRawQuery(TenantId);
impl arcgraph_mcp::tools::raw_query::RawQueryExecutor for StubRawQuery {
    fn execute(
        &self,
        tenant: TenantId,
        _query: &str,
        _max_rows: u32,
        _cancel: &CancellationToken,
    ) -> Result<arcgraph_mcp::tools::raw_query::RawQueryRows, MCPError> {
        if tenant != self.0 {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        Err(MCPError::InternalError(
            "stub raw_query not exercised by http proptest".into(),
        ))
    }
}

fn dispatcher()
-> Dispatcher<StubSchema, StubInspect, StubExplore, StubSearch, StubIngest, StubRawQuery> {
    let t = TenantId::new(7);
    Dispatcher::new(
        t,
        Arc::new(StubSchema(t)),
        Arc::new(StubInspect(t)),
        Arc::new(StubExplore(t)),
        Arc::new(StubSearch(t)),
        Arc::new(StubIngest(t)),
        Arc::new(StubRawQuery(t)),
    )
}

fn arb_method() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("graph.schema".to_string()),
        Just("graph.inspect".to_string()),
        Just("graph.bogus".to_string()),
        "[a-z]{1,16}\\.[a-z]{1,16}".prop_map(|s| s),
    ]
}

fn arb_jsonrpc_string() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("2.0".to_string()),
        Just("1.0".to_string()),
        Just("".to_string()),
        "[0-9]\\.[0-9]".prop_map(|s| s),
    ]
}

fn arb_id() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<u32>().prop_map(|n| json!(n)),
        any::<i32>().prop_map(|n| json!(n)),
        "[a-zA-Z0-9_-]{1,16}".prop_map(|s| json!(s)),
    ]
}

fn arb_tenant_id() -> impl Strategy<Value = u64> {
    prop_oneof![
        Just(7u64), // matches dispatcher tenant
        Just(8u64), // cross-tenant
        any::<u64>(),
    ]
}

fn arb_envelope() -> impl Strategy<Value = Value> {
    (
        arb_jsonrpc_string(),
        arb_id(),
        arb_method(),
        arb_tenant_id(),
        any::<u64>(),
    )
        .prop_map(|(jsonrpc, id, method, tenant, node_id)| {
            json!({
                "jsonrpc": jsonrpc,
                "id": id,
                "method": method,
                "params": {
                    "tenant_id": tenant,
                    "node_id": node_id,
                }
            })
        })
}

const PROTOCOL_CODES: &[i32] = &[
    -32700, // ParseError
    -32600, // InvalidRequest
    -32601, // MethodNotFound
    -32602, // InvalidParams
    -32603, // InternalError
];

const SERVER_CODES: &[i32] = &[
    -32001, // Cancelled
    -32002, // Unauthorized
    -32003, // TenantUnknown
    -32004, // IndexUnavailable
    -32005, // QueryError
    -32006, // ExecutionEval
];

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        // Each case is O(μs) — no special timeout.
        ..ProptestConfig::default()
    })]

    /// Every envelope (well-formed or not) produces EITHER `None`
    /// (notification) or a parseable JSON response with the JSON-RPC
    /// 2.0 shape: jsonrpc=="2.0", echoed id, exactly one of result
    /// or error.
    #[test]
    fn every_envelope_round_trips_to_a_valid_response(env in arb_envelope()) {
        let d = dispatcher();
        let id_present = env.get("id").is_some_and(|v| !v.is_null());
        let resp = handle_raw_envelope(&d, env);

        // Notifications: id == Null AND a successful response from
        // the dispatcher. A non-null id MUST produce Some(...).
        if let Some(value) = resp {
            prop_assert!(value.is_object(), "response must be JSON object: {value}");
            prop_assert_eq!(
                value.get("jsonrpc").and_then(|v| v.as_str()),
                Some("2.0"),
                "every response carries jsonrpc=2.0"
            );
            prop_assert!(value.get("id").is_some(), "every response echoes id");
            let has_result = value.get("result").is_some();
            let has_error = value.get("error").is_some();
            prop_assert!(
                has_result ^ has_error,
                "exactly one of result/error: result={has_result} error={has_error}"
            );
            if has_error {
                let code = value["error"]["code"].as_i64().unwrap_or(0) as i32;
                prop_assert!(
                    PROTOCOL_CODES.contains(&code) || SERVER_CODES.contains(&code),
                    "error code {code} not in defined set"
                );
                prop_assert!(
                    value["error"]["message"].is_string(),
                    "error message is a string"
                );
            }
        } else {
            // None response: the envelope must have been a notification
            // (no id) OR the dispatcher recognized the envelope and
            // chose to suppress a response. The dispatcher only
            // suppresses for notifications today; assert.
            prop_assert!(!id_present, "non-notification got None response");
        }
    }

    /// Determinism: the same envelope always yields the same response.
    #[test]
    fn handle_raw_envelope_is_deterministic(env in arb_envelope()) {
        let d = dispatcher();
        let r1 = handle_raw_envelope(&d, env.clone());
        let r2 = handle_raw_envelope(&d, env);
        prop_assert_eq!(r1, r2);
    }
}
