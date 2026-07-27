//! W17α M4-08+ — production storage-backed adapter implementations
//! for the MCP-side trait surfaces.
//!
//! Replaces the stub `SchemaProvider` / `NodeInspector` /
//! `NeighborhoodExplorer` / `HybridSearcher` / `IngestProvider` /
//! `RawQueryExecutor` adapters in `arcgraph_mcp_stdio.rs` with concrete
//! implementations that read / write through
//! [`arcgraph_storage::router::TenantHandle`] +
//! [`arcgraph_storage::transaction::TxnManager`] +
//! [`arcgraph_storage::InternTable`].
//!
//! # Bounded-context discipline
//!
//! Per `docs/bounded-contexts.md` the MCP layer owns the adapter
//! TRAITS (consumer-defined); the concrete IMPLEMENTATIONS live HERE
//! because they bridge the trait to `arcgraph-storage`. The
//! `arcgraph-storage` crate does NOT depend on `arcgraph-mcp`; the
//! reverse depend already exists. The adapter module sits at the same
//! cross-crate-trait-façade seam as `arcgraph_query::executor::ExecutorSubstrate`.
//!
//! # v1.0-alpha posture
//!
//! The adapters cover the load-bearing read/write paths:
//!
//! - **Schema enumeration** — labels + rel-types projected from
//!   [`arcgraph_storage::catalog::stats::CatalogStats::snapshot`] +
//!   resolved via [`arcgraph_storage::InternTable::resolve`].
//! - **Node inspection** — single-node read via
//!   [`arcgraph_storage::crud::read_node_with_store`] + 1-hop
//!   neighbors via [`arcgraph_storage::crud::scan_out`].
//! - **Neighborhood exploration** — BFS over the same
//!   [`arcgraph_storage::crud::scan_out`] surface with a per-call
//!   visited set.
//! - **Hybrid search availability** — surfaces the per-tenant
//!   substrate-attached flags from
//!   [`arcgraph_storage::router::TenantHandle::vector`] /
//!   [`arcgraph_storage::router::TenantHandle::bm25`]. The actual
//!   search BODY is forward-deferred to a v1.1 slice when the
//!   integrated HNSW + Tantivy query path is wired through the
//!   substrate trait — at W17α the availability gate is what's
//!   load-bearing (the MCP `graph.search` tool surfaces
//!   `MCPError::IndexUnavailable` cleanly when the substrate is
//!   not yet wired).
//! - **Ingest** — `create_node` + `create_rel` + `commit` via
//!   [`arcgraph_storage::crud`]. Property bags serialize through
//!   the existing [`arcgraph_storage::crud::PropertyData::Blob`]
//!   opaque-bytes path. Per-request idempotency tracking lives in
//!   an in-memory `(tenant, external_id) → NodeId` table on the
//!   provider instance (production restart resets the table; v1.1
//!   persists via the WAL intern records).
//! - **Raw query** — wraps [`arcgraph_query::QueryEngine`] with the
//!   storage-backed [`CrudExecutorSubstrate`] below; honors a
//!   per-request deadline via
//!   [`arcgraph_query::QueryEngine::execute_with_deadline`].
//!
//! # Forward-deferred (W17α scope-bound)
//!
//! - **Hybrid search BODY** — vector + BM25 + RRF fusion routed
//!   through the substrate's `vector_search` / `bm25_search`. At
//!   W17α the substrate surfaces availability + returns
//!   `IndexUnavailable` for the search body. v1.1 wires the real
//!   HNSW + Tantivy clients.
//! - **Streaming cursor** — the executor's
//!   [`arcgraph_query::cursor::StreamingCursor`] surface is held
//!   open at the substrate boundary; the production raw-query
//!   adapter currently materializes the full row set per
//!   `arcgraph_query::materialize` (matching the M5-11
//!   `RawQueryRows` wire shape). M5-streaming forward.
//! - **Property bag round-trip** — node properties serialize through
//!   a JSON blob; non-Blob `PropertyData` shapes (`Empty`,
//!   `InlineU32Pair`) round-trip as empty bags. Strict
//!   property typing is outside this adapter's scope.
//!
//! # ADR provenance
//! - **ADR-038 amendment-03 §M5↔M4** — the contract surface this
//!   module binds to.
//! - **ADR-037 §D-1** — `TenantHandle` per-tenant substrate
//!   composition.
//! - **ADR-031 §Decision** — group-commit `CommitBundle` per ingest
//!   call.
//! - **bounded-context policy** — implementer-vs-orchestrator discipline.

pub mod acl_ingest;
pub mod adapters;
pub mod arrow_batch;
pub mod bolt;
pub mod counting;
pub mod property_index;
pub mod property_payload;
pub mod substrate;

pub use acl_ingest::{AclDocIngest, AclIngestSummary, ingest_docs_with_acls};
pub use adapters::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider, build_catalog_for_tenant,
    json_to_value, property_data_for_json_map, value_to_json,
};
pub use bolt::StorageBoltHandler;
pub use counting::{CountingSubstrate, WriteCounters};
pub use substrate::{BoltHeldTxn, CrudExecutorSubstrate, SubstrateSearchProvider};
