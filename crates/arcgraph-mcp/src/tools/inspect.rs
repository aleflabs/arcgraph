//! W13δ M5-05 — `graph.inspect` Tier-1 MCP tool.
//!
//! Returns the per-node neighborhood: full property bag + 1-hop
//! neighbors with rel-type + direction info.
//!
//! # Surface seam — `NodeInspector`
//!
//! [`NodeInspector`] is the consumer-defined trait the tool reads
//! from. v1.0-alpha tests stub it in-line; production wiring at
//! M4-08+ implements it on the storage tenant handle (which already
//! exposes the CRUD + adjacency surfaces this tool composes).
//!
//! # Snapshot-LSN discipline
//!
//! Per ADR-038 amendment-03 §TIER-1 GAP E rule 1 (snapshot LSN
//! acquired at execute-time, before the first operator pulls a
//! batch), the inspector's storage-side reads MUST acquire a snapshot
//! LSN before the first batch pull (matching the executor's single-
//! statement materialize tail). The trait does not expose the LSN to
//! MCP callers — it's an implementation detail of the
//! [`NodeInspector::inspect`] body. v1.0-alpha stub impls have no
//! MVCC layer; production wiring at M4-08+ acquires per amendment-03
//! rule 1.
//!
//! # Access-control boundary
//!
//! The tool checks `request.tenant_id == session_tenant` BEFORE any
//! inspector call. Cross-tenant requests reject as
//! [`MCPError::Unauthorized`] (-32002). Principal-scoped requests
//! authorize the inspected node before storage and filter every denied
//! neighbor from the response. A principal-less request is admitted
//! only for an explicit [`SessionScope::Power`] session; all other
//! sessions fail closed with [`MCPError::Forbidden`] (-32008).
//!
//! # ADR provenance
//! - **ADR-004 §"Tier 1 (agent-facing, default)"** — `graph.inspect()`
//!   is the second Tier-1 tool in the 10-tool catalog.
//! - **ADR-038 amendment-03 §TIER-1 GAP E rule 1** — snapshot LSN
//!   acquired at execute-time (the canonical cite for single-statement
//!   read-only-tool snapshot binding; rule 2 covers multi-statement
//!   queries within ONE ArcQL query and does NOT apply to MCP cross-
//!   call surfaces).
//! - **ADR-037 D-1** — per-tenant routing; the cross-tenant guard
//!   inherits this posture.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::{NodeId, TenantId};
use arcgraph_storage::permissions::PermissionIndex;
use serde::{Deserialize, Serialize};

use crate::error::MCPError;
use crate::read_acl::authorize_read;
use crate::scope::SessionScope;
use crate::tools::ResponseFormat;

/// Adapter trait read by the [`inspect_tool`] entry point.
///
/// Implementations live OUTSIDE this crate: tests stub it in-line;
/// production wiring at M4-08+ implements it on the storage tenant
/// handle.
///
/// # Per-tenant scoping
///
/// `tenant: TenantId` parameter matches the [`crate::tools::schema::SchemaProvider`]
/// pattern — a single `NodeInspector` instance can serve multiple
/// tenants under a shared MCP router (forward-method per M5-12).
///
/// # `Send + Sync`
///
/// MCP transport runs on a tokio runtime; the inspector must be
/// shareable across awaits.
///
/// # Snapshot-LSN contract — IMPLEMENTOR HARD REQUIREMENT
///
/// Per ADR-038 amendment-03 §TIER-1 GAP E rule 1 (snapshot LSN
/// acquired at execute-time, before the first operator pulls a
/// batch), an `inspect()` call MUST acquire a snapshot LSN before
/// the first batch pull and hold it for the life of the call
/// (matching the executor's single-statement materialize tail). The
/// trait shape DELIBERATELY DOES NOT carry the LSN as a parameter —
/// v1.0-alpha stubs have no MVCC layer, and the production storage
/// handle is the natural source of LSN acquisition. The contract is
/// enforced by convention and the M4-08+ wiring slice's end-to-end
/// tests, not by the type signature.
///
/// **M4-08+ wiring slice MUST add a concurrent-inspect-during-write
/// proptest** that asserts: an `inspect()` call observed N writes; a
/// concurrent writer commits N+1 mid-call; `inspect()` returns the
/// snapshot at N (not N+1). Without this test the trait shape allows
/// a non-snapshot-isolated impl to type-check silently, so snapshot
/// isolation is part of this trait's contract.
pub trait NodeInspector: Send + Sync {
    /// Inspect the node identified by `node_id` in `tenant`.
    ///
    /// Errors as [`MCPError::TenantUnknown`] for an unbound tenant,
    /// [`MCPError::QueryError`] for a missing node id (rendered as
    /// "node not found" inside the query-error bucket — distinct from
    /// "tenant unknown"), or [`MCPError::ExecutionEval`] /
    /// [`MCPError::IndexUnavailable`] for substrate-level faults.
    ///
    /// Implementors MUST honor the snapshot-LSN contract on the trait
    /// doc comment above (amendment-03 §TIER-1 GAP E rule 1).
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError>;

    /// ADR-212 / ADR-218 — the per-tenant permission index used to
    /// authorize the inspected node and every neighbor disclosed in the
    /// response. The default is unavailable so principal-scoped calls
    /// fail closed rather than silently running without enforcement.
    fn permission_index(&self, tenant: TenantId) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
        let _ = tenant;
        Ok(None)
    }
}

/// Per-node inspection result returned by [`NodeInspector::inspect`].
///
/// Serializes as a JSON / YAML / TOON tree per M5-05.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NodeInspection {
    /// The inspected node's id.
    pub id: u64,
    /// Optional label. Single-label per ADR-038 §2 D-1 v1.0 grammar.
    pub label: Option<String>,
    /// Property bag — keyed by property name, values rendered as JSON
    /// values. `BTreeMap` for stable ordering across runs.
    pub properties: BTreeMap<String, serde_json::Value>,
    /// 1-hop neighborhood. Empty Vec if the node is isolated.
    pub neighbors: Vec<NeighborInfo>,
}

/// One entry in [`NodeInspection::neighbors`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeighborInfo {
    /// The neighbor's node id.
    pub node_id: u64,
    /// Optional neighbor label.
    pub label: Option<String>,
    /// Optional rel-type connecting this node and the neighbor.
    pub rel_type: Option<String>,
    /// Direction relative to the inspected node: `"out"`, `"in"`, or
    /// `"undirected"`.
    pub direction: NeighborDirection,
}

/// Direction tag on a [`NeighborInfo`].
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum NeighborDirection {
    /// Outbound: inspected node is the rel's `from`.
    Out,
    /// Inbound: inspected node is the rel's `to`.
    In,
    /// Undirected adjacency.
    Undirected,
}

// ─────────────────────────────────────────────────────────────────────
// Request envelope
// ─────────────────────────────────────────────────────────────────────

/// Request params for the `graph.inspect` tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InspectRequest {
    /// The tenant to inspect within.
    pub tenant_id: u64,
    /// The node id to inspect.
    pub node_id: u64,
    /// Optional render-format hint. Defaults to JSON (heterogeneous
    /// shape: scalar properties + nested neighbor list).
    #[serde(default)]
    pub format: Option<ResponseFormat>,
    /// ADR-212 end-user principal. An absent principal is admitted only
    /// for an explicit [`SessionScope::Power`] SYSTEM-TRUSTED session;
    /// non-power sessions fail closed with -32008 (#1488 / #1293).
    #[serde(default)]
    pub principal: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────

/// `graph.inspect` — return per-node neighborhood as JSON-RPC `result`.
///
/// # Cross-tenant guard
///
/// Same shape as [`crate::tools::schema::schema_tool`]: cross-tenant
/// requests reject as [`MCPError::Unauthorized`] before any
/// inspector call.
///
/// # Errors
///
/// - [`MCPError::Unauthorized`] — cross-tenant request.
/// - [`MCPError::Forbidden`] — a non-power session omitted `principal`.
/// - [`MCPError::IndexUnavailable`] — principal-scoped inspection has no
///   permission index and therefore refuses rather than serving content.
/// - [`MCPError::TenantUnknown`] / [`MCPError::QueryError`] —
///   propagated from [`NodeInspector::inspect`].
/// - [`MCPError::InternalError`] — serializer encode failure.
pub fn inspect_tool<I: NodeInspector + ?Sized>(
    inspector: &I,
    session_tenant: TenantId,
    session_scope: SessionScope,
    req: InspectRequest,
) -> Result<serde_json::Value, MCPError> {
    let request_tenant = TenantId::new(req.tenant_id);
    if request_tenant != session_tenant {
        return Err(MCPError::Unauthorized);
    }
    let access = authorize_read(
        "graph.inspect",
        req.principal.as_deref(),
        session_scope,
        || inspector.permission_index(request_tenant),
    )?;

    // Denied and missing nodes deliberately share the QueryError class
    // and message shape, and the denied branch does not touch storage.
    if !access.allows(NodeId::new(req.node_id)) {
        return Err(MCPError::QueryError(format!(
            "node {} not found",
            req.node_id
        )));
    }

    let mut inspection = inspector.inspect(request_tenant, req.node_id)?;
    inspection
        .neighbors
        .retain(|neighbor| access.allows(NodeId::new(neighbor.node_id)));
    let format = req.format.unwrap_or(ResponseFormat::Json);
    let value = serde_json::to_value(&inspection)
        .map_err(|e| MCPError::InternalError(format!("inspection serialize: {e}")))?;
    crate::tools::render_response(format, &value)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// In-memory fixture: tenant → node_id → NodeInspection.
    #[derive(Debug, Clone, Default)]
    struct StubNodeInspector {
        bound_tenant: Option<TenantId>,
        nodes: std::collections::HashMap<u64, NodeInspection>,
        permissions: Option<Arc<PermissionIndex>>,
    }

    impl StubNodeInspector {
        fn new(tenant: TenantId) -> Self {
            Self {
                bound_tenant: Some(tenant),
                nodes: Default::default(),
                permissions: None,
            }
        }
        fn with_node(mut self, n: NodeInspection) -> Self {
            self.nodes.insert(n.id, n);
            self
        }

        fn with_permissions(mut self, permissions: Arc<PermissionIndex>) -> Self {
            self.permissions = Some(permissions);
            self
        }
    }

    impl NodeInspector for StubNodeInspector {
        fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
            match self.bound_tenant {
                Some(t) if t == tenant => match self.nodes.get(&node_id).cloned() {
                    Some(n) => Ok(n),
                    None => Err(MCPError::QueryError(format!("node {node_id} not found"))),
                },
                _ => Err(MCPError::TenantUnknown(format!("{tenant:?}"))),
            }
        }

        fn permission_index(
            &self,
            _tenant: TenantId,
        ) -> Result<Option<Arc<PermissionIndex>>, MCPError> {
            Ok(self.permissions.clone())
        }
    }

    fn alice() -> NodeInspection {
        let mut props: BTreeMap<String, serde_json::Value> = BTreeMap::new();
        props.insert("name".into(), serde_json::json!("Alice"));
        props.insert("age".into(), serde_json::json!(30));
        NodeInspection {
            id: 1,
            label: Some("Person".into()),
            properties: props,
            neighbors: vec![NeighborInfo {
                node_id: 2,
                label: Some("Person".into()),
                rel_type: Some("KNOWS".into()),
                direction: NeighborDirection::Out,
            }],
        }
    }

    #[test]
    fn inspect_tool_returns_node_with_neighbors() {
        let i = StubNodeInspector::new(TenantId::new(1)).with_node(alice());
        let req = InspectRequest {
            tenant_id: 1,
            node_id: 1,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let resp = inspect_tool(&i, TenantId::new(1), SessionScope::Power, req).expect("ok");
        assert_eq!(resp["format"], "json");
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("Alice"));
        assert!(body.contains("KNOWS"));
        assert!(
            body.contains("\"out\""),
            "direction emitted as lowercase tag"
        );
    }

    #[test]
    fn inspect_tool_rejects_cross_tenant_request_with_unauthorized() {
        // The session is bound to tenant 1; the request asks for
        // tenant 2. MUST reject with -32002 BEFORE any inspector
        // call (we verify "before any inspector call" by binding the
        // inspector to NEITHER tenant; the inspector's TenantUnknown
        // branch does not fire — the outer Unauthorized does).
        let i = StubNodeInspector::default();
        let req = InspectRequest {
            tenant_id: 2,
            node_id: 1,
            format: None,
            principal: None,
        };
        let err = inspect_tool(&i, TenantId::new(1), SessionScope::Power, req)
            .expect_err("cross-tenant must reject");
        assert_eq!(err.code(), -32002);
        assert!(matches!(err, MCPError::Unauthorized));
    }

    #[test]
    fn inspect_tool_propagates_node_not_found_as_query_error() {
        // Same-tenant inspect on a missing node — surfaces -32005
        // (query error) per the M5-05 contract: "node not found" is
        // a query-domain issue, NOT an authorization issue.
        let i = StubNodeInspector::new(TenantId::new(1));
        let req = InspectRequest {
            tenant_id: 1,
            node_id: 999,
            format: None,
            principal: None,
        };
        let err = inspect_tool(&i, TenantId::new(1), SessionScope::Power, req)
            .expect_err("missing node must reject");
        assert_eq!(err.code(), -32005);
        match err {
            MCPError::QueryError(msg) => assert!(msg.contains("999")),
            other => panic!("expected QueryError, got {other:?}"),
        }
    }

    #[test]
    fn inspect_tool_default_format_is_json() {
        // Per the M5-05 wire contract: graph.inspect default = JSON
        // (heterogeneous shape; YAML / TOON pivot through Value
        // anyway). Pin this default explicitly so a future M5-02
        // sub-slice can't silently change it.
        let i = StubNodeInspector::new(TenantId::new(1)).with_node(alice());
        let req = InspectRequest {
            tenant_id: 1,
            node_id: 1,
            format: None,
            principal: None,
        };
        let resp = inspect_tool(&i, TenantId::new(1), SessionScope::Power, req).expect("ok");
        assert_eq!(resp["format"], "json");
    }

    #[test]
    fn principal_scoped_inspect_filters_denied_neighbor_content_1488() {
        let permissions = Arc::new(PermissionIndex::new());
        permissions.apply_doc_acl(
            NodeId::new(1),
            std::collections::BTreeSet::from(["alice".to_owned()]),
        );
        permissions.apply_doc_acl(
            NodeId::new(2),
            std::collections::BTreeSet::from(["bob".to_owned()]),
        );
        let i = StubNodeInspector::new(TenantId::new(1))
            .with_node(alice())
            .with_permissions(permissions);
        let req = InspectRequest {
            tenant_id: 1,
            node_id: 1,
            format: Some(ResponseFormat::Json),
            principal: Some("alice".into()),
        };
        let resp = inspect_tool(&i, TenantId::new(1), SessionScope::Read, req)
            .expect("principal-scoped inspect");
        let inspection: NodeInspection =
            serde_json::from_str(resp["body"].as_str().expect("body")).expect("inspection");
        assert!(inspection.neighbors.is_empty(), "denied neighbor omitted");
    }

    #[test]
    fn absent_principal_non_power_inspect_fails_closed_minus_32008_1488() {
        let i = StubNodeInspector::new(TenantId::new(1)).with_node(alice());
        let req = InspectRequest {
            tenant_id: 1,
            node_id: 1,
            format: Some(ResponseFormat::Json),
            principal: None,
        };
        let err = inspect_tool(&i, TenantId::new(1), SessionScope::Read, req)
            .expect_err("missing principal must fail closed");
        assert_eq!(err.code(), -32008);
    }

    #[test]
    fn denied_inspect_is_indistinguishable_from_missing_node_1488() {
        let permissions = Arc::new(PermissionIndex::new());
        permissions.apply_doc_acl(
            NodeId::new(1),
            std::collections::BTreeSet::from(["bob".to_owned()]),
        );
        let i = StubNodeInspector::new(TenantId::new(1))
            .with_node(alice())
            .with_permissions(permissions);
        let req = InspectRequest {
            tenant_id: 1,
            node_id: 1,
            format: Some(ResponseFormat::Json),
            principal: Some("alice".into()),
        };
        let err = inspect_tool(&i, TenantId::new(1), SessionScope::Read, req)
            .expect_err("denied node must not be inspected");
        assert_eq!(err.code(), -32005);
        assert_eq!(format!("{err}"), "query error: node 1 not found");
    }

    #[test]
    fn inspect_request_rejects_unknown_field() {
        let v = serde_json::json!({
            "tenant_id": 1,
            "node_id": 1,
            "fmt": "json"  // typo of `format`
        });
        let res: Result<InspectRequest, _> = serde_json::from_value(v);
        assert!(res.is_err());
    }

    #[test]
    fn neighbor_direction_serde_round_trip() {
        // Pin the wire shape: lowercase tags so MCP clients can route
        // on the string without case-sensitivity gymnastics.
        let n = NeighborInfo {
            node_id: 5,
            label: None,
            rel_type: None,
            direction: NeighborDirection::Undirected,
        };
        let s = serde_json::to_string(&n).unwrap();
        assert!(s.contains("\"undirected\""));
        let n2: NeighborInfo = serde_json::from_str(&s).unwrap();
        assert_eq!(n2.direction, NeighborDirection::Undirected);
    }

    #[test]
    fn node_inspection_round_trips_property_bag_sorted() {
        // BTreeMap → stable wire order; pin against any future swap
        // to HashMap that would break deterministic test diffs.
        let n = alice();
        let s = serde_json::to_string(&n).unwrap();
        let n2: NodeInspection = serde_json::from_str(&s).unwrap();
        assert_eq!(n, n2);
        // age comes before name lexicographically.
        let pos_age = s.find("\"age\"").expect("age key");
        let pos_name = s.find("\"name\"").expect("name key");
        assert!(pos_age < pos_name, "BTreeMap keys must serialize sorted");
    }
}
