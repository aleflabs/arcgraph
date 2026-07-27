//! W13δ M5-04 / M5-05 + W14β M5-06 / M5-07 + W14γ M5-08 — Tier-1 MCP tool implementations.
//!
//! This module hosts the five Tier-1 tools and one Tier-2 tool:
//!
//! - **`graph.schema`** ([`schema`] submodule) — return the per-tenant
//!   schema (labels, rel-types, indexes, optional cardinalities). W13δ
//!   M5-04.
//! - **`graph.inspect`** ([`inspect`] submodule) — return a single
//!   node's properties + 1-hop neighborhood. W13δ M5-05.
//! - **`graph.explore`** ([`explore`] submodule) — return an N-hop
//!   neighborhood graph rooted at a seed node id (W14β M5-06).
//! - **`graph.search`** ([`search`] submodule) — RRF-fused hybrid
//!   retrieval over the vector + BM25 substrates (W14β M5-07).
//! - **`graph.ingest`** ([`ingest`] submodule) — batch ingest of
//!   nodes + relationships, per-record idempotency on `external_id`,
//!   ADR-031 §Decision group-commit durability, cross-MCP-call
//!   reads-after-write via amendment-03 §TIER-1 GAP E rule 1 + LSN
//!   monotonicity. W14γ M5-08; first WRITE-side surface in the
//!   catalog. v1.0-α wire shape per ADR-004 amendment-01.
//! - **`graph.raw_query`** ([`raw_query`] submodule) — Tier-2 power-
//!   user MCP tool; requires `arcgraph.power` scope. Direct ArcQL
//!   execution through the [`raw_query::RawQueryExecutor`] adapter.
//!   W16ζ M5-11; first Tier-2 surface in the catalog. v1.0-α wire
//!   shape per ADR-004 amendment-03.
//!
//! The ADR-004 hard cap remains load-bearing. A catalog change must update
//! the dispatcher, tool-list schema, ACL mapping, and catalog-pin tests
//! together.
//!
//! # Output format selection
//!
//! Each tool accepts an optional `format` field on its request params:
//! `"toon"` (canonical for tabular results), `"yaml"` (canonical for
//! nested results), or `"json"` (fallback). The default depends on
//! the tool's natural shape per design-v2 §9.3:
//!
//! - `graph.schema` defaults to `"yaml"` (nested tree).
//! - `graph.inspect` defaults to `"json"` (mixed shape; the property
//!   bag is heterogeneous).
//! - `graph.explore` defaults to `"toon"` (uniform-shape node/edge
//!   rows; the design-v2 §9.3 token-savings path — TOON delivers
//!   40-60% fewer tokens than JSON on this shape).
//! - `graph.search` defaults to `"toon"` (uniform-shape hit rows).
//! - `graph.ingest` defaults to `"json"` (per-record outcomes carry
//!   heterogeneous tags; YAML / TOON pivot through `Value`).
//!
//! TOON / YAML rendering pivots through `serde_json::Value` per the
//! [`crate::serializers`] module, so any tool output is renderable in
//! any of the three formats.
//!
//! # Tenant scoping (W13δ M5-05 hard requirement)
//!
//! Every tool request carries a `tenant_id` field. The dispatcher
//! checks it against the session's bound tenant; a mismatch surfaces
//! [`crate::MCPError::Unauthorized`] BEFORE any storage access — per
//! the spawn prompt's "cross-tenant access rejected with
//! `MCPError::Unauthorized`" hard requirement.
//!
//! # Adapter traits (consumer-defined here)
//!
//! [`SchemaProvider`] and [`NodeInspector`] are MCP-side adapter
//! traits — defined here, not in `arcgraph-storage` — so the
//! bounded-context discipline is preserved. v1.0-alpha tests stub
//! them in-line; production wiring at M4-08+ implements them on the
//! storage tenant handle.

pub mod explore;
pub mod ingest;
pub mod inspect;
pub mod raw_query;
pub mod schema;
pub mod search;

pub use explore::{
    DEFAULT_EXPLORE_DEPTH, DEFAULT_EXPLORE_LIMIT, ExploreRequest, MAX_EXPLORE_DEPTH,
    MAX_EXPLORE_LIMIT, Neighborhood, NeighborhoodEdge, NeighborhoodExplorer, NeighborhoodNode,
    explore_tool,
};
pub use ingest::{
    IngestBatch, IngestError, IngestProvider, IngestRecordOutcome, IngestRequest, IngestSummary,
    NodeIngest, RelIngest, ingest_tool,
};
pub use inspect::{InspectRequest, NeighborInfo, NodeInspection, NodeInspector, inspect_tool};
pub use raw_query::{
    DEFAULT_RAW_QUERY_MAX_ROWS, MAX_RAW_QUERY_BYTES, MAX_RAW_QUERY_MAX_ROWS, RawQueryExecutor,
    RawQueryRequest, RawQueryRows, raw_query_tool,
};
pub use schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, PropertyDescriptor, RelTypeInfo,
    SchemaProvider, SchemaRequest, schema_tool,
};
pub use search::{
    AvailableSubstrates, DEFAULT_SEARCH_K, HybridSearcher, MAX_SEARCH_K, SUBSTRATE_SLUG_BM25,
    SUBSTRATE_SLUG_VECTOR, SearchHit, SearchRequest, SearchResult, search_tool, substrate_kinds,
};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::MCPError;

/// Render-format hint carried on tool requests.
///
/// Default selection is per-tool (see module docs). Clients that pin
/// a specific format set this field on the request params.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ResponseFormat {
    /// TOON — canonical for tabular row sets per design-v2 §9.3.
    Toon,
    /// YAML — canonical for nested results.
    #[default]
    Yaml,
    /// JSON — fallback. Useful for clients that don't speak TOON /
    /// YAML.
    Json,
}

/// Render `value` in the caller-selected format and wrap it in the
/// canonical `{ "format": ..., "body": ... }` envelope returned in
/// the JSON-RPC `result` slot.
///
/// All three formats are renderable from a `serde_json::Value`. The
/// returned wrapper carries the format slug so MCP clients can route
/// on it without sniffing the body string.
pub fn render_response(format: ResponseFormat, value: &Value) -> Result<Value, MCPError> {
    let body = match format {
        ResponseFormat::Toon => crate::serializers::to_toon(value)
            .map_err(|e| MCPError::InternalError(format!("toon encode: {e}")))?,
        ResponseFormat::Yaml => crate::serializers::to_yaml(value)
            .map_err(|e| MCPError::InternalError(format!("yaml encode: {e}")))?,
        ResponseFormat::Json => serde_json::to_string(value)
            .map_err(|e| MCPError::InternalError(format!("json encode: {e}")))?,
    };
    let slug = match format {
        ResponseFormat::Toon => "toon",
        ResponseFormat::Yaml => "yaml",
        ResponseFormat::Json => "json",
    };
    Ok(serde_json::json!({
        "format": slug,
        "body": body,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn render_response_yaml_emits_text_body() {
        let v = serde_json::json!({"labels": ["Person"], "indexes": []});
        let out = render_response(ResponseFormat::Yaml, &v).expect("ok");
        assert_eq!(out["format"], "yaml");
        let body = out["body"].as_str().expect("string body");
        assert!(body.contains("labels"), "body: {body:?}");
    }

    #[test]
    fn render_response_json_emits_canonical_json_body() {
        let v = serde_json::json!({"a": 1});
        let out = render_response(ResponseFormat::Json, &v).expect("ok");
        assert_eq!(out["format"], "json");
        let body = out["body"].as_str().unwrap();
        // Canonical serde_json output (compact, no extra spaces).
        assert!(body.contains("\"a\":1"), "body: {body:?}");
    }

    #[test]
    fn render_response_default_is_yaml() {
        // Pin the default — tools that don't override default to YAML
        // (nested-shape friendly per design-v2 §9.3).
        assert_eq!(ResponseFormat::default(), ResponseFormat::Yaml);
    }
}
