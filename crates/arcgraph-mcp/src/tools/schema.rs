//! W13δ M5-04 — `graph.schema` Tier-1 MCP tool.
//!
//! Returns the per-tenant schema: labels, rel-types, optional
//! property descriptors, optional cardinalities, and substrate-
//! availability flags (vector / bm25 / community).
//!
//! # Surface seam — `SchemaProvider`
//!
//! [`SchemaProvider`] is the consumer-defined trait the tool reads
//! from. It is INTENTIONALLY broader than [`arcgraph_query::semantic::CatalogProvider`]:
//! the catalog provider only supports `lookup_label(name) -> Option<LabelId>`,
//! which cannot enumerate the full label set. The schema tool needs
//! enumeration; [`SchemaProvider`] adds it. Production wiring at
//! M4-08+ implements both traits on the storage tenant handle (the
//! storage layer already has both lookup + enumeration capabilities;
//! the trait split is purely a bounded-context concern).
//!
//! # Why not extend `CatalogProvider`?
//!
//! Per `feedback_avoid_speculative_scaffolding.md`: ship the trait
//! when first consumed, not when first imagined. The schema tool is
//! the FIRST consumer of an enumeration surface; extending
//! `CatalogProvider` with `labels() -> Vec<...>` defaults would have
//! been speculative for the M4-21 binding pass (which only needs
//! `lookup_label(name)`). Defining a fit-for-purpose
//! [`SchemaProvider`] trait here keeps the binding-pass surface
//! minimal AND lets future M5 tools (`graph.explore`) reuse this
//! enumeration without revisiting `CatalogProvider`.
//!
//! # ADR provenance
//! - **ADR-004 §"Tier 1 (agent-facing, default)"** — `graph.schema()`
//!   is the first Tier-1 tool in the 10-tool catalog.
//! - **ADR-038 amendment-03 §M5↔M4** — the contract surface MCP tools
//!   bind to.

use arcgraph_core::TenantId;
use serde::{Deserialize, Serialize};

use crate::error::MCPError;
use crate::tools::ResponseFormat;

/// Adapter trait read by the [`schema_tool`] entry point.
///
/// Implementations live OUTSIDE this crate: tests stub it in-line;
/// production wiring at M4-08+ implements it on the storage tenant
/// handle (which already carries the per-tenant catalog +
/// substrate-availability map).
///
/// # Per-tenant scoping
///
/// Every method takes `tenant: TenantId` so a single
/// `SchemaProvider` instance can serve multiple tenants under a
/// shared MCP router (the M5-12 forward-method). v1.0-alpha tests
/// typically pin a single tenant per impl.
///
/// # `Send + Sync`
///
/// The MCP transport runs on a tokio runtime (per design-v2 §4.1
/// "Tokio for the agent surface"); the provider must be safe to
/// share across awaits.
pub trait SchemaProvider: Send + Sync {
    /// Return the schema for `tenant`. Errors as
    /// [`MCPError::TenantUnknown`] if the tenant has no registered
    /// catalog binding.
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError>;
}

/// Per-tenant graph schema returned by [`SchemaProvider::schema`].
///
/// Serializes as a YAML / TOON / JSON tree per the W13δ M5-04 wire
/// shape (per design-v2 §9.1 bullet 6: "return the full graph schema
/// as YAML"). The structured shape lets clients render it in any of
/// the three supported formats.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphSchema {
    /// The tenant this schema describes (echoed for client-side
    /// disambiguation in multi-tenant deployments).
    pub tenant_id: u64,
    /// Per-label descriptors. Sort order is implementation-defined;
    /// callers MAY rely on insertion order from the producer (the
    /// stub provider used in tests sorts by `LabelId.raw()`).
    pub labels: Vec<LabelInfo>,
    /// Per-rel-type descriptors. Same sort-order semantics as
    /// `labels`.
    pub rel_types: Vec<RelTypeInfo>,
    /// Per-tenant attached substrates (vector / bm25 / community).
    /// MAY be empty if the tenant has no substrates attached.
    pub indexes: Vec<IndexDescriptor>,
    /// Tenant-wide totals reported by the catalog (per ADR-038 §2 D-25
    /// catalog stats). `None` when stats have not been collected yet.
    pub total_node_count: Option<u64>,
    /// Tenant-wide rel count. Same semantics as `total_node_count`.
    pub total_rel_count: Option<u64>,
}

/// Per-label descriptor in [`GraphSchema`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LabelInfo {
    /// The label name (the user-visible identifier).
    pub name: String,
    /// Optional cardinality (`label_cardinality` from the catalog).
    /// `None` when stats have not been collected yet.
    pub cardinality: Option<u64>,
    /// Properties observed for this label. v1.0-alpha catalog impls
    /// MAY return an empty Vec if per-label property tracking is
    /// deferred (M4-41 stats schema does not pin per-label property
    /// types yet).
    pub properties: Vec<PropertyDescriptor>,
}

/// Per-rel-type descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelTypeInfo {
    /// The rel-type name (the user-visible identifier).
    pub name: String,
    /// Optional cardinality.
    pub cardinality: Option<u64>,
}

/// Property descriptor on a label.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PropertyDescriptor {
    /// The property key.
    pub name: String,
    /// Cypher-style type slug. v1.0-alpha admits only a small set:
    /// `"INTEGER"`, `"FLOAT"`, `"STRING"`, `"BOOLEAN"`, `"NULL"`,
    /// `"LIST"`. Strict-schema catalogs (v1.1+) MAY emit richer
    /// types; the MCP surface keeps the slug as a free-form string
    /// for forward-compat.
    pub kind: String,
}

/// Substrate-availability descriptor.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexDescriptor {
    /// The index kind.
    pub kind: IndexKind,
    /// Whether the index is currently attached + readable for this
    /// tenant. `false` indicates a planned-but-unbuilt substrate.
    pub available: bool,
}

/// Index-kind tag.
///
/// `#[serde(rename_all = "lowercase")]` so the wire shape is
/// `"vector"` / `"bm25"` / `"community"` (matching the substrate-
/// availability slugs used elsewhere in the codebase: ADR-035 / ADR-
/// 039 / ADR-040).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum IndexKind {
    /// HNSW vector index (per ADR-035).
    Vector,
    /// BM25 text index (per ADR-039).
    Bm25,
    /// Community-detection index (per ADR-040).
    Community,
}

// ─────────────────────────────────────────────────────────────────────
// Request envelope
// ─────────────────────────────────────────────────────────────────────

/// Request params for the `graph.schema` tool.
///
/// `#[serde(deny_unknown_fields)]` under the code-quality policy config-strict-mode
/// convention — typo-friendly clients should fail fast rather than
/// silently degrade.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SchemaRequest {
    /// The tenant to enumerate. Cross-tenant requests reject as
    /// [`MCPError::Unauthorized`] before any catalog access.
    pub tenant_id: u64,
    /// Optional render-format hint. Defaults to YAML per
    /// [`ResponseFormat::default`] (nested tree).
    #[serde(default)]
    pub format: Option<ResponseFormat>,
}

// ─────────────────────────────────────────────────────────────────────
// Tool entry point
// ─────────────────────────────────────────────────────────────────────

/// `graph.schema` — return per-tenant schema as JSON-RPC `result`.
///
/// # Cross-tenant guard
///
/// `session_tenant` is the tenant the MCP session is bound to (the
/// dispatcher resolves it from the session-init handshake). The
/// request's `tenant_id` MUST match; mismatch returns
/// [`MCPError::Unauthorized`] before any [`SchemaProvider`] call.
///
/// # Errors
///
/// - [`MCPError::Unauthorized`] — cross-tenant request.
/// - [`MCPError::TenantUnknown`] — provider has no binding for the
///   tenant.
/// - [`MCPError::InternalError`] — serializer encode failure.
pub fn schema_tool<P: SchemaProvider + ?Sized>(
    provider: &P,
    session_tenant: TenantId,
    req: SchemaRequest,
) -> Result<serde_json::Value, MCPError> {
    let request_tenant = TenantId::new(req.tenant_id);
    if request_tenant != session_tenant {
        return Err(MCPError::Unauthorized);
    }
    let schema = provider.schema(request_tenant)?;
    let format = req.format.unwrap_or_default();
    let value = serde_json::to_value(&schema)
        .map_err(|e| MCPError::InternalError(format!("schema serialize: {e}")))?;
    crate::tools::render_response(format, &value)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny in-memory `SchemaProvider` impl for unit tests.
    #[derive(Debug, Clone)]
    struct StubSchemaProvider {
        tenant: TenantId,
        labels: Vec<LabelInfo>,
        rel_types: Vec<RelTypeInfo>,
        indexes: Vec<IndexDescriptor>,
        total_nodes: Option<u64>,
        total_rels: Option<u64>,
    }

    impl SchemaProvider for StubSchemaProvider {
        fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
            if tenant != self.tenant {
                return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
            }
            Ok(GraphSchema {
                tenant_id: tenant.raw(),
                labels: self.labels.clone(),
                rel_types: self.rel_types.clone(),
                indexes: self.indexes.clone(),
                total_node_count: self.total_nodes,
                total_rel_count: self.total_rels,
            })
        }
    }

    fn fixture_provider() -> StubSchemaProvider {
        StubSchemaProvider {
            tenant: TenantId::new(7),
            labels: vec![
                LabelInfo {
                    name: "Person".into(),
                    cardinality: Some(1_000),
                    properties: vec![
                        PropertyDescriptor {
                            name: "name".into(),
                            kind: "STRING".into(),
                        },
                        PropertyDescriptor {
                            name: "age".into(),
                            kind: "INTEGER".into(),
                        },
                    ],
                },
                LabelInfo {
                    name: "Doc".into(),
                    cardinality: Some(500),
                    properties: vec![],
                },
            ],
            rel_types: vec![RelTypeInfo {
                name: "KNOWS".into(),
                cardinality: Some(2_500),
            }],
            indexes: vec![
                IndexDescriptor {
                    kind: IndexKind::Vector,
                    available: true,
                },
                IndexDescriptor {
                    kind: IndexKind::Bm25,
                    available: true,
                },
            ],
            total_nodes: Some(1_500),
            total_rels: Some(2_500),
        }
    }

    #[test]
    fn schema_tool_returns_expected_label_and_rel_type_set() {
        let p = fixture_provider();
        let req = SchemaRequest {
            tenant_id: 7,
            format: Some(ResponseFormat::Json),
        };
        let resp = schema_tool(&p, TenantId::new(7), req).expect("ok");
        assert_eq!(resp["format"], "json");
        let body = resp["body"].as_str().expect("body");
        // JSON contains canonical field names + values.
        assert!(body.contains("Person"), "body must include Person label");
        assert!(body.contains("KNOWS"), "body must include KNOWS rel-type");
    }

    #[test]
    fn schema_tool_rejects_cross_tenant_request() {
        // Session is bound to tenant 1; request asks for tenant 2.
        // MUST reject with Unauthorized — per the W13δ M5-05
        // cross-tenant guard hard requirement.
        let p = fixture_provider();
        let req = SchemaRequest {
            tenant_id: 2,
            format: None,
        };
        let err = schema_tool(&p, TenantId::new(1), req).expect_err("cross-tenant must reject");
        assert!(matches!(err, MCPError::Unauthorized), "got {err:?}");
        assert_eq!(err.code(), -32002);
    }

    #[test]
    fn schema_tool_default_format_is_yaml() {
        let p = fixture_provider();
        let req = SchemaRequest {
            tenant_id: 7,
            format: None,
        };
        let resp = schema_tool(&p, TenantId::new(7), req).expect("ok");
        assert_eq!(resp["format"], "yaml");
        // YAML body must contain the YAML hyphen-list shape.
        let body = resp["body"].as_str().unwrap();
        assert!(body.contains("Person"));
    }

    #[test]
    fn schema_tool_supports_toon_format() {
        // Tabular-friendly TOON requested explicitly.
        let p = fixture_provider();
        let req = SchemaRequest {
            tenant_id: 7,
            format: Some(ResponseFormat::Toon),
        };
        let resp = schema_tool(&p, TenantId::new(7), req).expect("ok");
        assert_eq!(resp["format"], "toon");
    }

    #[test]
    fn schema_tool_propagates_tenant_unknown() {
        // Session-tenant matches request-tenant, but the provider has
        // no binding for tenant 9 — MUST surface
        // MCPError::TenantUnknown (-32003), NOT a generic internal
        // error.
        let p = fixture_provider();
        let req = SchemaRequest {
            tenant_id: 9,
            format: None,
        };
        let err = schema_tool(&p, TenantId::new(9), req).expect_err("missing-binding must reject");
        assert_eq!(err.code(), -32003);
    }

    #[test]
    fn schema_request_rejects_unknown_field() {
        // code-quality policy strict-mode discipline: a typo in a request
        // must reject at deserialize time, NOT silently route to a
        // default. Pin the deny_unknown_fields contract.
        let v = serde_json::json!({"tenant_id": 7, "fromat": "yaml"});
        let res: Result<SchemaRequest, _> = serde_json::from_value(v);
        assert!(res.is_err(), "typo must reject");
    }

    #[test]
    fn graph_schema_round_trips_through_serde_json() {
        // Pin: the GraphSchema shape is stable for serde round-trip
        // (Value → bytes → Value → struct equality). M5-04's wire
        // contract assumes this.
        let p = fixture_provider();
        let s = p.schema(TenantId::new(7)).expect("ok");
        let v = serde_json::to_value(&s).unwrap();
        let s2: GraphSchema = serde_json::from_value(v).unwrap();
        assert_eq!(s, s2);
    }
}
