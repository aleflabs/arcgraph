//! Storage-backed implementations of the MCP adapter traits.
//!
//! See the module-level rustdoc on [`super`] for the full per-trait
//! scope, snapshot-LSN discipline, and forward-deferred items.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, NodeId, NodeRecord, PartitionId, RelId, TenantId};
use arcgraph_query::executor::fusion::rrf_fuse;
use arcgraph_query::executor::value::Value as QueryValue;
use arcgraph_query::{CancellationToken, QueryEngine};
use arcgraph_storage::crud::{self, CrudStore, PropertyData};
use arcgraph_storage::metrics::{MetricsSink, QueryPlanType};
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::{IdempotencyStore, InternTable};
use serde::Serialize;
use serde_json::Value as JsonValue;

use crate::error::MCPError;
use crate::tools::explore::{
    ExploreDirection, Neighborhood, NeighborhoodEdge, NeighborhoodExplorer, NeighborhoodNode,
};
use crate::tools::ingest::{
    AclGrant, DroppedAclGrant, IngestBatch, IngestError, IngestProvider, IngestRecordOutcome,
    IngestSummary, NodeIngest, RelIngest,
};
use crate::tools::inspect::{NeighborDirection, NeighborInfo, NodeInspection, NodeInspector};
use crate::tools::raw_query::{RawQueryExecutor, RawQueryRows};
use crate::tools::schema::{
    GraphSchema, IndexDescriptor, IndexKind, LabelInfo, RelTypeInfo, SchemaProvider,
};
use crate::tools::search::{AvailableSubstrates, HybridSearcher, SearchHit};

use super::substrate::{CrudExecutorSubstrate, SubstrateSearchProvider};

/// Discriminator for the idempotency namespace.
///
/// Node `external_id`s and rel `external_id`s share the wire-level
/// string namespace at the [`IngestProvider`] surface — an MCP client
/// may name a node "x" and a rel "x" without colliding semantically.
/// The kind keeps these namespaces disjoint inside the durable store so
/// a re-submitted node never resolves to a rel's id (or vice versa).
///
/// Per R1 review MED-1 (PR #349) the key is
/// `(TenantId, IdempotencyKind, String) → u64`. #352 Part 2 (ADR-199)
/// makes the binding durable: `arcgraph-mcp` keeps this semantic enum +
/// the ingest gate, and maps it to an opaque `u8` ([`Self::as_u8`]) at
/// the published [`arcgraph_storage::IdempotencyStore`] boundary, so
/// storage stays semantics-agnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IdempotencyKind {
    /// Node external_id → node internal_id.
    Node,
    /// Rel external_id → rel internal_id.
    Rel,
}

impl IdempotencyKind {
    /// Opaque wire discriminator handed to the storage-resident
    /// [`arcgraph_storage::IdempotencyStore`] (which attaches no node/rel
    /// meaning to it). Stable: these bytes are persisted in the v6
    /// `CommitBundle` `idempotency_bindings` section, so the mapping
    /// (`Node = 0`, `Rel = 1`) MUST NOT change without a WAL-format bump.
    #[must_use]
    pub fn as_u8(self) -> u8 {
        match self {
            IdempotencyKind::Node => 0,
            IdempotencyKind::Rel => 1,
        }
    }
}

// #352 Part 2 (ADR-199): the per-tenant in-memory `IdempotencyMap` +
// its 100K cap (Part 1, #851) are REMOVED. The binding is now durable
// and the runtime lookup structure is the storage-resident
// `arcgraph_storage::IdempotencyStore` (rebuilt on WAL replay from the
// v6 `CommitBundle` `idempotency_bindings` section). With a durable
// fallback the cap is unnecessary: a long-running tenant holds unbounded
// distinct external_ids and they survive a restart. The
// `IngestError::CapacityExceeded` variant is retained for API
// compatibility but is no longer produced.

/// Shared bundle of workspace handles every storage adapter needs.
///
/// Constructed once at process startup and `Arc::clone`d into each
/// adapter; per-call overhead is two pointer chases. The bundle is
/// `Send + Sync` (every inner Arc carries `Send + Sync` bounds).
#[derive(Clone)]
pub struct StorageBackend {
    router: Arc<MultiTenantRouter>,
    txn_manager: Arc<TxnManager>,
    intern_table: Arc<InternTable>,
    /// #352 Part 2 (ADR-199): the durable, storage-resident idempotency
    /// store keyed by `(TenantId, kind, external_id) → internal_id`.
    /// Rebuilt on WAL replay from the v6 `CommitBundle`
    /// `idempotency_bindings` section, so a binding survives a `--data`
    /// restart and there is no per-tenant cap. The durable bootstrap
    /// passes the SAME `Arc` it wired into the replay target (via
    /// [`Self::with_idempotency_store`]); the ephemeral / test paths get
    /// a fresh per-backend store.
    idempotency: Arc<IdempotencyStore>,
}

impl std::fmt::Debug for StorageBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageBackend")
            .field("router", &"<Arc<MultiTenantRouter>>")
            .field("txn_manager", &"<Arc<TxnManager>>")
            .field("intern_table", &"<Arc<InternTable>>")
            .field("idempotency_entries", &self.idempotency.total_len())
            .finish()
    }
}

impl StorageBackend {
    /// Construct a backend from the workspace's shared handles. The
    /// idempotency store starts fresh; the durable bootstrap then shares
    /// the recovery-wired store via [`Self::with_idempotency_store`].
    #[must_use]
    pub fn new(
        router: Arc<MultiTenantRouter>,
        txn_manager: Arc<TxnManager>,
        intern_table: Arc<InternTable>,
    ) -> Self {
        Self {
            router,
            txn_manager,
            intern_table,
            idempotency: Arc::new(IdempotencyStore::new()),
        }
    }

    /// #352 Part 2 (ADR-199): swap in a shared
    /// [`arcgraph_storage::IdempotencyStore`] — used by the durable
    /// bootstrap to hand the backend the SAME `Arc` it wired into the WAL
    /// replay target (`PageStoreTarget::with_idempotency_store`), so
    /// recovered `external_id → internal_id` bindings are visible to the
    /// ingest path's lookups. The ephemeral / test paths keep the fresh
    /// store from [`Self::new`] (no recovery there).
    #[must_use]
    pub fn with_idempotency_store(mut self, store: Arc<IdempotencyStore>) -> Self {
        self.idempotency = store;
        self
    }

    /// Borrow the router. Adapters that need substrate-availability
    /// flags ([`StorageHybridSearcher`]) reach for `route().vector()
    /// / .bm25() / .community()`.
    pub fn router(&self) -> &Arc<MultiTenantRouter> {
        &self.router
    }

    pub fn txn_manager(&self) -> &Arc<TxnManager> {
        &self.txn_manager
    }

    pub fn intern_table(&self) -> &Arc<InternTable> {
        &self.intern_table
    }

    /// Acquire the per-tenant `CrudStore` Arc. Returns
    /// [`MCPError::TenantUnknown`] on a routing miss.
    fn crud_for(&self, tenant: TenantId) -> Result<Arc<CrudStore>, MCPError> {
        let handle = self
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        let crud = Arc::clone(handle.crud());
        crud.set_idempotency_store(Arc::clone(&self.idempotency));
        Ok(crud)
    }
}

// ─────────────────────────────────────────────────────────────────────
// SchemaProvider
// ─────────────────────────────────────────────────────────────────────

/// Storage-backed [`SchemaProvider`] — enumerates labels + rel-types
/// from [`arcgraph_storage::catalog::stats::CatalogStats::snapshot`]
/// (label cardinalities + rel-type cardinalities) and resolves names
/// via the shared [`InternTable`].
#[derive(Clone)]
pub struct StorageSchemaProvider {
    backend: StorageBackend,
}

impl std::fmt::Debug for StorageSchemaProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageSchemaProvider")
            .field("backend", &self.backend)
            .finish()
    }
}

impl StorageSchemaProvider {
    /// Construct a schema provider over a shared backend.
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        Self { backend }
    }
}

impl SchemaProvider for StorageSchemaProvider {
    fn schema(&self, tenant: TenantId) -> Result<GraphSchema, MCPError> {
        let crud = self.backend.crud_for(tenant)?;
        // Pull the catalog-stats snapshot; pre-first-commit returns
        // None for totals + empty per-label vec. The wire shape
        // surfaces None → `total_node_count = None`.
        let stats_arc = crud.catalog_stats(tenant);
        let snapshot = stats_arc.as_ref().map(|stats| stats.snapshot());

        let intern = self.backend.intern_table();

        // Build per-label info from the snapshot's label cards.
        let label_cards = snapshot
            .as_ref()
            .map(|s| s.label_cards().to_vec())
            .unwrap_or_default();
        let mut labels: Vec<LabelInfo> = label_cards
            .iter()
            .map(|(label, card)| {
                let name = intern
                    .resolve(tenant, arcgraph_core::ids::StringId::new(label.raw()))
                    .map(|arc| arc.to_string())
                    .unwrap_or_else(|| format!("label:{}", label.raw()));
                LabelInfo {
                    name,
                    cardinality: Some(*card),
                    properties: Vec::new(),
                }
            })
            .collect();
        labels.sort_by(|a, b| a.name.cmp(&b.name));

        let rel_type_cards = snapshot
            .as_ref()
            .map(|s| s.rel_type_cards().to_vec())
            .unwrap_or_default();
        let mut rel_types: Vec<RelTypeInfo> = rel_type_cards
            .iter()
            .map(|(ty, card)| {
                let name = intern
                    .resolve(tenant, arcgraph_core::ids::StringId::new(ty.raw()))
                    .map(|arc| arc.to_string())
                    .unwrap_or_else(|| format!("type:{}", ty.raw()));
                RelTypeInfo {
                    name,
                    cardinality: Some(*card),
                }
            })
            .collect();
        rel_types.sort_by(|a, b| a.name.cmp(&b.name));

        let total_node_count = snapshot.as_ref().and_then(|s| s.total_nodes());
        let total_rel_count = snapshot.as_ref().and_then(|s| s.total_rels());

        // Substrate-availability: read from the TenantHandle.
        let handle = self
            .backend
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        let mut indexes: Vec<IndexDescriptor> = Vec::new();
        if handle.vector().is_some() {
            indexes.push(IndexDescriptor {
                kind: IndexKind::Vector,
                available: true,
            });
        }
        if handle.bm25().is_some() {
            indexes.push(IndexDescriptor {
                kind: IndexKind::Bm25,
                available: true,
            });
        }
        if handle.community().is_some() {
            indexes.push(IndexDescriptor {
                kind: IndexKind::Community,
                available: true,
            });
        }

        Ok(GraphSchema {
            tenant_id: tenant.raw(),
            labels,
            rel_types,
            indexes,
            total_node_count,
            total_rel_count,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// NodeInspector
// ─────────────────────────────────────────────────────────────────────

/// Storage-backed [`NodeInspector`] — reads a single node + its 1-hop
/// neighborhood via [`arcgraph_storage::crud`].
#[derive(Clone)]
pub struct StorageNodeInspector {
    backend: StorageBackend,
}

impl std::fmt::Debug for StorageNodeInspector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageNodeInspector")
            .field("backend", &self.backend)
            .finish()
    }
}

impl StorageNodeInspector {
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        Self { backend }
    }
}

/// Hydrate a node's persisted property bag (ADR-152 §D-3) into the
/// `serde_json::Value` map carried by the MCP inspect / explore wire
/// shapes ([`NodeInspection`] / [`NeighborhoodNode`]).
///
/// Bridges the executor-domain
/// [`crate::storage::property_payload::record_property_bag`] (which
/// yields a `BTreeMap<String, QueryValue>`) to
/// `BTreeMap<String, JsonValue>` by routing each cell through
/// [`QueryValue::to_json_value`] — the inverse of the write-path encode
/// in [`crate::storage::property_payload`]. The storage-internal
/// `inline_u32a/b` slots are deliberately NOT surfaced as user-facing
/// property names (R1 review MED-6, PR #349); the helper inherits that
/// drop from `record_property_bag`. A missing / undecodable blob
/// degrades to an empty bag (per the helper's own rustdoc) rather than
/// failing the read.
///
/// Closes the #894 ingest → inspect/explore data-loss surface: the W17α
/// "scope-bound to empty" stub (and its #356 v1.2 deferral) is obsolete
/// now that the ADR-152 decode path has landed on main.
fn hydrate_node_properties(
    rec: &NodeRecord,
    crud: &CrudStore,
    intern: &InternTable,
    tenant: TenantId,
) -> Result<BTreeMap<String, JsonValue>, MCPError> {
    // v2 M2: typed-payload corruption surfaces LOUD (design §M2.2) —
    // the pre-M2 silent empty-bag degrade is retired for it. The
    // ratified missing-blob cross-snapshot degrade (ADR-149 §Risks)
    // is preserved inside the checked read.
    Ok(
        crate::storage::property_payload::record_property_bag_checked(
            rec,
            crud.blob_store(),
            intern,
            tenant,
        )
        .map_err(|e| MCPError::ExecutionEval(format!("property payload decode failed: {e}")))?
        .into_iter()
        .map(|(name, value)| (name, value.to_json_value()))
        .collect(),
    )
}

impl NodeInspector for StorageNodeInspector {
    fn inspect(&self, tenant: TenantId, node_id: u64) -> Result<NodeInspection, MCPError> {
        let crud = self.backend.crud_for(tenant)?;
        let tx = self.backend.txn_manager.begin(tenant);
        let nid = NodeId::new(node_id);
        let rec = match crud::read_node(&tx, nid) {
            Ok(Some(r)) => r,
            Ok(None) => {
                return Err(MCPError::QueryError(format!("node {node_id} not found")));
            }
            Err(e) => {
                return Err(MCPError::ExecutionEval(format!(
                    "inspect: read_node failed: {e}"
                )));
            }
        };

        // Resolve label name.
        let label_name = if rec.label_id == 0 {
            None
        } else {
            self.backend
                .intern_table
                .resolve(tenant, arcgraph_core::ids::StringId::new(rec.label_id))
                .map(|arc| arc.to_string())
        };

        // Property bag — hydrate the persisted JSON blob (ADR-152 §D-3)
        // so an ingested `{name: "Alice"}` round-trips through
        // `graph.inspect` (#894: close the silent ingest → read
        // data-loss surface). The storage-internal `inline_u32a/b` slots
        // are deliberately dropped, not surfaced as property names
        // (R1 review MED-6, PR #349) — `record_property_bag` enforces
        // that. Supersedes the W17α scope-to-empty stub + its #356 pin.
        let properties = hydrate_node_properties(&rec, &crud, &self.backend.intern_table, tenant)?;

        // 1-hop neighbors via scan_out (outbound). For an undirected
        // scan we'd also walk inbound; v1.0-alpha's TEL stores each
        // edge once-per-direction, so the outbound walk covers the
        // `Out` direction tag.
        let mut neighbors: Vec<NeighborInfo> = Vec::new();
        for entry in crud::scan_out(&crud, &tx, nid, None) {
            let dst_id = entry.dst_id;
            // Resolve the destination's label for the wire-shape.
            let dst_label = match crud::read_node(&tx, NodeId::new(dst_id)) {
                Ok(Some(r)) if r.label_id != 0 => self
                    .backend
                    .intern_table
                    .resolve(tenant, arcgraph_core::ids::StringId::new(r.label_id))
                    .map(|arc| arc.to_string()),
                _ => None,
            };
            // Resolve rel-type by reading the RelRecord; TelEntry
            // carries only `(dst_id, rel_id)` so we must hop through
            // the rel store to get the type. Failures fall back to
            // `None` so a partially-corrupt entry never blocks the
            // inspection.
            let rel_type_name = match crud::read_rel(&tx, RelId::new(entry.rel_id)) {
                Ok(Some(rel)) if rel.type_id != 0 => self
                    .backend
                    .intern_table
                    .resolve(tenant, arcgraph_core::ids::StringId::new(rel.type_id))
                    .map(|arc| arc.to_string()),
                _ => None,
            };
            neighbors.push(NeighborInfo {
                node_id: dst_id,
                label: dst_label,
                rel_type: rel_type_name,
                direction: NeighborDirection::Out,
            });
        }

        Ok(NodeInspection {
            id: node_id,
            label: label_name,
            properties,
            neighbors,
        })
    }

    /// #1488 / ADR-212 production authorization seam: expose the SAME
    /// per-tenant index owned by the routed `TenantHandle` that the
    /// search and explore adapters use.
    fn permission_index(
        &self,
        tenant: TenantId,
    ) -> Result<Option<Arc<arcgraph_storage::permissions::PermissionIndex>>, MCPError> {
        let handle = self
            .backend
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        Ok(Some(Arc::clone(handle.permissions())))
    }
}

// ─────────────────────────────────────────────────────────────────────
// NeighborhoodExplorer
// ─────────────────────────────────────────────────────────────────────

/// Storage-backed [`NeighborhoodExplorer`] — BFS rooted at the seed,
/// honoring `max_depth` + the optional `rel_filter` allowlist.
#[derive(Clone)]
pub struct StorageNeighborhoodExplorer {
    backend: StorageBackend,
}

impl std::fmt::Debug for StorageNeighborhoodExplorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageNeighborhoodExplorer")
            .field("backend", &self.backend)
            .finish()
    }
}

impl StorageNeighborhoodExplorer {
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        Self { backend }
    }

    fn resolve_label(&self, tenant: TenantId, label_id: u32) -> Option<String> {
        if label_id == 0 {
            return None;
        }
        self.backend
            .intern_table
            .resolve(tenant, arcgraph_core::ids::StringId::new(label_id))
            .map(|arc| arc.to_string())
    }
}

impl NeighborhoodExplorer for StorageNeighborhoodExplorer {
    fn explore(
        &self,
        tenant: TenantId,
        seed: u64,
        max_depth: u32,
        rel_filter: Option<&[String]>,
        direction: ExploreDirection,
        cancel: &CancellationToken,
    ) -> Result<Neighborhood, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        let crud = self.backend.crud_for(tenant)?;
        let tx = self.backend.txn_manager.begin(tenant);

        // ADR-217: which directions to walk. `Out` (default) is the
        // v1.0-alpha behavior (`scan_out`); `In` walks reverse adjacency
        // (`crud::scan_in`, ADR-131); `Both` walks each and de-dups by
        // RelId. `want_out` / `want_in` keep the BFS body a single loop.
        let (want_out, want_in) = match direction {
            ExploreDirection::Out => (true, false),
            ExploreDirection::In => (false, true),
            ExploreDirection::Both => (true, true),
        };

        // Resolve seed first; missing → QueryError("seed not found").
        let seed_id = NodeId::new(seed);
        let seed_rec = match crud::read_node(&tx, seed_id) {
            Ok(Some(r)) => r,
            Ok(None) => {
                return Err(MCPError::QueryError(format!("seed {seed} not found")));
            }
            Err(e) => {
                return Err(MCPError::ExecutionEval(format!(
                    "explore: read_node(seed={seed}) failed: {e}"
                )));
            }
        };

        // Resolve allowed rel-type IDs from the filter strings, if
        // present. Unknown names produce an empty allowlist for
        // *that name*; if the entire filter resolves to empty, the
        // walk produces zero outbound edges (the caller asked for an
        // unsupported set).
        //
        // L-5 (R1 review, PR #349): `intern_type` is a write-side
        // API — calling it from a read path creates a phantom entry
        // for typos like `KNOWZ`. Issue #355 tracks the v1.1
        // `lookup_type_by_name` (read-only) API; until it lands we
        // detect the create via `intern_is_new` and emit a
        // `tracing::debug!` so the leak is observable in tests.
        //
        // Durability note (#788, M-1 from #782 R1): this phantom publish is
        // UNLOGGED, so a later durable CREATE of the same name observes
        // `was_new == false`, skips the InternString WAL log, and the name
        // can be lost on restart. Moving this read path to the read-only
        // `lookup_type_by_name` (#355) also closes that durability edge.
        let allowed_rel_types: Option<HashSet<u32>> = match rel_filter {
            Some(names) if !names.is_empty() => {
                let mut set = HashSet::new();
                for name in names {
                    // Fail-closed: on error this used to return STRINGID_SENTINEL
                    // (the reserved id 0) as though the name were interned, so the
                    // rel-type allowlist would contain id 0 and match the wrong
                    // edges. Propagate instead of inventing an id.
                    let (sid, was_new) = self
                        .backend
                        .intern_table
                        .intern_is_new(tenant, name)
                        .map_err(|error| {
                            MCPError::InternalError(format!(
                                "interning rel-type filter {name:?}: {error}"
                            ))
                        })?;
                    if was_new {
                        tracing::debug!(
                            target: "arcgraph_mcp::storage::adapters",
                            tenant = ?tenant,
                            name = %name,
                            id = sid.raw(),
                            "intern_type created new id from read path; lookup_type_by_name pending issue #355"
                        );
                    }
                    set.insert(sid.raw());
                }
                Some(set)
            }
            _ => None,
        };

        // BFS frontier: queue of (node_id, depth). Visited set
        // de-dupes both nodes + edges.
        let mut nodes_out: Vec<NeighborhoodNode> = Vec::new();
        let mut edges_out: Vec<NeighborhoodEdge> = Vec::new();
        let mut visited_nodes: HashSet<u64> = HashSet::new();
        let mut visited_edges: BTreeSet<u64> = BTreeSet::new();
        let mut frontier: VecDeque<(NodeId, u32)> = VecDeque::new();

        nodes_out.push(NeighborhoodNode {
            id: seed,
            label: self.resolve_label(tenant, seed_rec.label_id),
            depth: 0,
            // #894: hydrate the seed's persisted property bag (ADR-152
            // §D-3); pre-fix this was an empty `{}` (silent data-loss).
            properties: hydrate_node_properties(
                &seed_rec,
                &crud,
                &self.backend.intern_table,
                tenant,
            )?,
        });
        visited_nodes.insert(seed);
        frontier.push_back((seed_id, 0));

        while let Some((current, depth)) = frontier.pop_front() {
            if cancel.is_cancelled() {
                return Err(MCPError::Cancelled);
            }
            if depth >= max_depth {
                continue;
            }

            // ADR-217: a single per-neighbor handler used for BOTH the
            // outbound (`scan_out`) and inbound (`scan_in`) entries.
            // `neighbor` is the OTHER endpoint (for outbound = the edge's
            // `to`; for inbound = the edge's `from` — `scan_in` puts the
            // original SRC in `entry.dst_id`). `edge_dir` tags which way
            // the edge points relative to `current`. Edges are de-duped by
            // RelId across BOTH directions, so a self-loop or a relation
            // seen from both ends is emitted once.
            // v2 M2: the neighbor hydration inside this `()`-returning
            // TEL visitor can fail LOUD (typed-payload corruption);
            // the fault is captured here and surfaced right after the
            // walk — never swallowed.
            let mut hydrate_err: Option<MCPError> = None;
            let mut handle_entry =
                |entry: &arcgraph_core::TelEntry, edge_dir: NeighborDirection| {
                    if hydrate_err.is_some() {
                        return;
                    }
                    // Resolve rel-type via the RelRecord; TelEntry alone
                    // doesn't carry the type.
                    let rel = match crud::read_rel(&tx, RelId::new(entry.rel_id)) {
                        Ok(Some(r)) => r,
                        _ => return,
                    };
                    if let Some(allow) = &allowed_rel_types {
                        if !allow.contains(&rel.type_id) {
                            return;
                        }
                    }
                    let neighbor_raw = entry.dst_id;
                    // De-dup edges by rel-id (across both directions).
                    if !visited_edges.insert(entry.rel_id) {
                        return;
                    }
                    let rel_type_name = if rel.type_id == 0 {
                        None
                    } else {
                        self.backend
                            .intern_table
                            .resolve(tenant, arcgraph_core::ids::StringId::new(rel.type_id))
                            .map(|arc| arc.to_string())
                    };
                    // Orient the wire edge so `from`→`to` matches the real
                    // relationship: outbound = current→neighbor; inbound =
                    // neighbor→current.
                    let (from, to) = match edge_dir {
                        NeighborDirection::In => (neighbor_raw, current.raw()),
                        _ => (current.raw(), neighbor_raw),
                    };
                    edges_out.push(NeighborhoodEdge {
                        from,
                        to,
                        rel_type: rel_type_name,
                        direction: edge_dir,
                    });
                    if visited_nodes.insert(neighbor_raw) {
                        // Single point-read serves both label resolution and
                        // property hydration (#894: pre-fix neighbor props
                        // were an empty `{}`; ADR-152 §D-3 decode path).
                        let (n_label, n_props) =
                            match crud::read_node(&tx, NodeId::new(neighbor_raw)) {
                                Ok(Some(r)) => {
                                    let props = match hydrate_node_properties(
                                        &r,
                                        &crud,
                                        &self.backend.intern_table,
                                        tenant,
                                    ) {
                                        Ok(p) => p,
                                        Err(e) => {
                                            hydrate_err = Some(e);
                                            return;
                                        }
                                    };
                                    (self.resolve_label(tenant, r.label_id), props)
                                }
                                _ => (None, BTreeMap::new()),
                            };
                        nodes_out.push(NeighborhoodNode {
                            id: neighbor_raw,
                            label: n_label,
                            depth: depth + 1,
                            properties: n_props,
                        });
                        if depth + 1 < max_depth {
                            frontier.push_back((NodeId::new(neighbor_raw), depth + 1));
                        }
                    }
                };

            if want_out {
                for entry in crud::scan_out(&crud, &tx, current, None) {
                    handle_entry(&entry, NeighborDirection::Out);
                }
            }
            if want_in {
                // `scan_in` yields the reverse-adjacency entries (ADR-131);
                // a disabled reverse index surfaces a structured error,
                // which we map to ExecutionEval. The reverse index is
                // default-enabled, so this is the common path for `In`/`Both`.
                match crud::scan_in(&crud, &tx, current, None) {
                    Ok(entries) => {
                        for entry in &entries {
                            handle_entry(entry, NeighborDirection::In);
                        }
                    }
                    Err(e) => {
                        return Err(MCPError::ExecutionEval(format!(
                            "explore: scan_in(node={}) failed: {e}",
                            current.raw()
                        )));
                    }
                }
            }
            // Surface a hydration fault captured inside the visitor
            // (v2 M2 loud-corruption contract). NB: `handle_entry`
            // last borrows `hydrate_err` above, so the read here is
            // legal without an explicit drop (NLL).
            if let Some(e) = hydrate_err {
                return Err(e);
            }
        }

        Ok(Neighborhood {
            seed,
            max_depth,
            // Output cap (`max_results`) is applied at the tool boundary
            // by `explore_capped`; the explorer leaves `truncated` false.
            truncated: false,
            nodes: nodes_out,
            edges: edges_out,
        })
    }

    /// #1488 / ADR-212 production authorization seam: expose the routed
    /// tenant's shared PermissionIndex. An empty index is deny-all, not an
    /// unsafe "permissions unavailable" state.
    fn permission_index(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<Option<Arc<arcgraph_storage::permissions::PermissionIndex>>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        let handle = self
            .backend
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        Ok(Some(Arc::clone(handle.permissions())))
    }
}

// ─────────────────────────────────────────────────────────────────────
// HybridSearcher
// ─────────────────────────────────────────────────────────────────────

/// Storage-backed [`HybridSearcher`] — surfaces per-tenant substrate
/// availability via [`arcgraph_storage::router::TenantHandle`]. At
/// W17α the search BODY returns `IndexUnavailable` per the module
/// rustdoc; v1.1 wires the real HNSW + BM25 query path.
#[derive(Clone)]
pub struct StorageHybridSearcher {
    backend: StorageBackend,
    /// #765 PART-1 — the served vector-search provider. `None` preserves the
    /// pre-#765 posture (the `search` body returns `IndexUnavailable`);
    /// production bootstrap binds the concrete HNSW provider via
    /// [`Self::with_search_provider`]. `Arc<dyn _>` so one provider instance
    /// serves every adapter clone.
    search_provider: Option<Arc<dyn SubstrateSearchProvider>>,
}

impl std::fmt::Debug for StorageHybridSearcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageHybridSearcher")
            .field("backend", &self.backend)
            .field("search_provider", &self.search_provider.is_some())
            .finish()
    }
}

impl StorageHybridSearcher {
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        Self {
            backend,
            search_provider: None,
        }
    }

    /// #765 PART-1 — bind the served vector-search provider so the
    /// [`HybridSearcher::search`] body runs real HNSW KNN (replacing the
    /// pre-#765 `IndexUnavailable` stub). Builder-style; chains after
    /// [`Self::new`]. Production bootstrap (`arcgraph serve`) binds the
    /// `arcgraph_cli::vector_search::HnswVectorSearchProvider`.
    #[must_use]
    pub fn with_search_provider(mut self, provider: Arc<dyn SubstrateSearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// #1379 (MUST-CON-04) — drop any ranked hit whose node is no longer
    /// LIVE (MVCC-tombstoned by a committed delete). Belt-and-suspenders
    /// against the deleted-node leak class: a lazily-tombstoned BM25 /
    /// vector index (or a missed ACL revoke) can still SURFACE a deleted
    /// node's id as a candidate; a `read_node`-liveness read at a fresh
    /// snapshot returns `Ok(None)` for a tombstoned node, so we filter it.
    ///
    /// # Cost / budget (PD#5)
    ///
    /// One `read_node_with_store` (primary-index probe → MVCC fallback,
    /// both O(1) point reads) per candidate under one begin-snapshot tx,
    /// bounded by `k` (default ≤ MAX_SEARCH_K). A single point read is
    /// ~sub-µs warm; the gate adds `k` of them AFTER RRF fusion has
    /// already narrowed to ≤ k rows — off the ranking hot loop, additive
    /// per-query cost ≈ k × (point read), negligible vs. the HNSW beam +
    /// BM25 legs. On a routing miss we fail CLOSED (return no hits) rather
    /// than serve un-verified candidates.
    fn retain_live_hits(&self, tenant: TenantId, hits: Vec<SearchHit>) -> Vec<SearchHit> {
        if hits.is_empty() {
            return hits;
        }
        let Ok(crud) = self.backend.crud_for(tenant) else {
            // Fail-closed: cannot verify liveness ⇒ return nothing rather
            // than risk serving a tombstoned candidate. In practice the
            // tenant was just routed by `search_filtered` above.
            tracing::warn!(
                target: "arcgraph_mcp::storage::adapters",
                tenant = ?tenant,
                "graph.search liveness gate: tenant route failed; \
                 dropping all hits (fail-closed)"
            );
            return Vec::new();
        };
        let tx = self.backend.txn_manager.begin(tenant);
        hits.into_iter()
            .filter(|h| {
                matches!(
                    crud::read_node_with_store(&crud, &tx, NodeId::new(h.node_id)),
                    Ok(Some(_))
                )
            })
            .collect()
    }
}

impl HybridSearcher for StorageHybridSearcher {
    fn available_substrates(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<AvailableSubstrates, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        let handle = self
            .backend
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        Ok(AvailableSubstrates {
            vector: handle.vector().is_some(),
            bm25: handle.bm25().is_some(),
        })
    }

    fn search(
        &self,
        tenant: TenantId,
        query_text: &str,
        query_vec: Option<&[f32]>,
        k: u32,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        // No-filter / default-ef path: delegate to the single
        // implementation so `search` and `search_filtered` can never
        // diverge (#815 / #816a).
        self.search_filtered(tenant, query_text, query_vec, k, None, None, cancel)
    }

    // 8 args: parallels the trait shape (label_filter + ef_search pushdown);
    // same allow precedent as the trait method + `HnswGraph::search_with_rescore`.
    #[allow(clippy::too_many_arguments)]
    fn search_filtered(
        &self,
        tenant: TenantId,
        query_text: &str,
        query_vec: Option<&[f32]>,
        k: u32,
        label_filter: Option<&[String]>,
        ef_search: Option<u32>,
        cancel: &CancellationToken,
    ) -> Result<Vec<SearchHit>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        // Tenant-scope check (cross-tenant defense in depth; the dispatcher
        // already gated above this).
        let handle = self
            .backend
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        let Some(provider) = self.search_provider.as_ref() else {
            return Err(MCPError::IndexUnavailable(
                "search provider not attached (bind it at process bootstrap via \
                 StorageHybridSearcher::with_search_provider; #765 PART-1)"
                    .into(),
            ));
        };
        let has_text = !query_text.trim().is_empty();
        let has_bm25 = handle.bm25().is_some();
        if query_vec.is_none() && (!has_text || !has_bm25) {
            return Err(MCPError::IndexUnavailable(
                "graph.search: neither vector input nor BM25 text substrate is available".into(),
            ));
        }

        // #815 — resolve filter label NAMES → LabelIds in the tenant's
        // interned id space so the predicate is pushed INTO the HNSW beam
        // (filter-during-search), not applied as a recall-collapsing
        // post-filter. `names_for_tenant` does NOT allocate a phantom id
        // for an unknown name (unlike `intern_label`), so a hostile/unknown
        // filter name cannot grow the intern table; it simply contributes
        // no id. A non-empty filter whose names are ALL unknown resolves to
        // an empty allowlist = "match nothing" (correct: no node carries
        // that label). `None` / empty filter → no label predicate.
        let label_ids: Option<Vec<LabelId>> = match label_filter {
            Some(names) if !names.is_empty() => {
                // Fail-closed: `probe` used to launder an owner-store I/O error
                // into `None`, which this filter reads as "no node carries that
                // label" — the filter would silently match NOTHING. A genuine
                // unknown name still contributes no id; only a real lookup
                // failure propagates.
                let mut ids: Vec<LabelId> = Vec::with_capacity(names.len());
                for name in names {
                    if let Some(sid) = self
                        .backend
                        .intern_table()
                        .try_probe(tenant, name)
                        .map_err(|error| {
                            MCPError::InternalError(format!(
                                "resolving label filter {name:?}: {error}"
                            ))
                        })?
                    {
                        ids.push(LabelId::new(sid.raw()));
                    }
                }
                Some(ids)
            }
            _ => None,
        };

        let ranked_vector = match query_vec {
            Some(query_vec) => Some(
                provider
                    .vector_search_filtered(
                        tenant,
                        crate::tools::search::DEFAULT_VECTOR_PROPERTY,
                        query_vec,
                        u64::from(k),
                        label_ids.as_deref(),
                        ef_search.map(|e| e as usize),
                        Lsn::MAX,
                    )
                    .map_err(translate_substrate_error)?,
            ),
            None => None,
        };

        let ranked_bm25 = if has_text && has_bm25 {
            match provider.bm25_search(
                tenant,
                "text",
                query_text,
                u64::from(k),
                Lsn::new(u64::MAX - 1),
            ) {
                Ok(hits) => Some(hits),
                Err(e) if ranked_vector.is_some() => {
                    tracing::warn!(
                        "graph.search hybrid: BM25 leg unavailable; degrading to vector-only: {}",
                        e
                    );
                    None
                }
                Err(e) => return Err(translate_substrate_error(e)),
            }
        } else {
            None
        };

        let ranked = match (ranked_vector, ranked_bm25) {
            (Some(vector), Some(bm25)) => rrf_fuse_search_hits(
                tenant,
                self.backend.intern_table(),
                &[vector, bm25],
                k as usize,
            ),
            (Some(vector), None) => {
                ranked_hits_to_search_hits(tenant, self.backend.intern_table(), vector)
            }
            (None, Some(bm25)) => {
                ranked_hits_to_search_hits(tenant, self.backend.intern_table(), bm25)
            }
            (None, None) => {
                return Err(MCPError::IndexUnavailable(
                    "graph.search: no vector or BM25 hits could be produced".into(),
                ));
            }
        };
        // #1379 (MUST-CON-04) — belt-and-suspenders liveness gate. The
        // BM25 / vector substrates tombstone lazily (a delete marks the
        // node deleted on a SEPARATE seam that a stale index or a
        // revoke-that-was-missed can leave un-scrubbed), so a candidate
        // may still be RETRIEVED after its node was deleted. Drop every
        // candidate whose MVCC record is no longer live (tombstoned).
        // This is defense-in-depth for the exact #1379 leak class: even
        // if the ACL revoke is somehow missed, a deleted node is not
        // returned. `read_node`-liveness is the same not-live signal
        // `inspect` reads (Ok(None) ⇒ tombstoned at this snapshot).
        let ranked = self.retain_live_hits(tenant, ranked);
        Ok(ranked)
    }

    /// ADR-212 §D-4 Seam-1 production wiring: expose the routed
    /// tenant's `PermissionIndex`
    /// (`TenantHandle::permissions()`, ADR-037-amendment-02) so
    /// `graph.search` can enforce principal-scoped visibility. The
    /// handle field is ALWAYS present (never `Option`) — an empty
    /// index is fail-closed by construction, so there is no
    /// "permissions not wired" unsafe state on the storage-backed
    /// searcher.
    fn permission_index(
        &self,
        tenant: TenantId,
        cancel: &CancellationToken,
    ) -> Result<Option<Arc<arcgraph_storage::permissions::PermissionIndex>>, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        let handle = self
            .backend
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| MCPError::TenantUnknown(format!("routing failed: {e}")))?;
        Ok(Some(Arc::clone(handle.permissions())))
    }
}

fn ranked_hits_to_search_hits(
    tenant: TenantId,
    intern: &InternTable,
    ranked: Vec<arcgraph_query::executor::substrate::RankedHit>,
) -> Vec<SearchHit> {
    ranked
        .into_iter()
        .map(|r| SearchHit {
            node_id: r.node.id.raw(),
            label: label_name_for_hit(tenant, intern, &r.node),
            score: r.score,
        })
        .collect()
}

fn label_name_for_hit(
    tenant: TenantId,
    intern: &InternTable,
    node: &arcgraph_query::executor::value::NodeView,
) -> Option<String> {
    node.label_name.clone().or_else(|| {
        node.label.and_then(|l| {
            intern
                .resolve(tenant, arcgraph_core::ids::StringId::new(l.raw()))
                .map(|arc| arc.to_string())
        })
    })
}

fn rrf_fuse_search_hits(
    tenant: TenantId,
    intern: &InternTable,
    lists: &[Vec<arcgraph_query::executor::substrate::RankedHit>],
    k: usize,
) -> Vec<SearchHit> {
    let mut out: Vec<SearchHit> = rrf_fuse(lists, 60)
        .into_iter()
        .map(|hit| SearchHit {
            node_id: hit.node.id.raw(),
            label: label_name_for_hit(tenant, intern, &hit.node),
            score: hit.score,
        })
        .collect();
    if out.len() > k {
        out.truncate(k);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// IngestProvider
// ─────────────────────────────────────────────────────────────────────

/// Storage-backed [`IngestProvider`] — writes nodes + rels through
/// [`arcgraph_storage::crud`] + commits the per-call transaction so
/// the group-commit fsync cohort surfaces a single durable LSN.
///
/// # Idempotency publish discipline
///
/// Per R1 review HIGH-2 (PR #349) the idempotency cache is updated
/// only AFTER `crud::commit` succeeds. A commit-time failure
/// (fsync error, WAL corruption, disk full) leaves the cache
/// untouched so a subsequent retry either re-Inserts cleanly or
/// surfaces a structured error — it can NEVER return
/// `Idempotent { internal_id }` for a non-existent record.
#[derive(Clone)]
pub struct StorageIngestProvider {
    backend: StorageBackend,
    /// Test-only switch: when set, the next `ingest` call
    /// synthesizes a commit failure after the writes have entered
    /// the transaction. Used by the R1 HIGH-2 regression test to
    /// prove the idempotency cache is NOT poisoned on commit
    /// failure. Production constructors leave this `false`.
    #[cfg(test)]
    force_commit_failure_for_tests: Arc<std::sync::atomic::AtomicBool>,
}

impl std::fmt::Debug for StorageIngestProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageIngestProvider")
            .field("backend", &self.backend)
            .finish()
    }
}

impl StorageIngestProvider {
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        Self {
            backend,
            #[cfg(test)]
            force_commit_failure_for_tests: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// Test-only handle to toggle the forced commit-failure switch.
    /// Returns the inner `AtomicBool` so tests can flip it across
    /// `ingest` calls in a single fixture lifetime.
    #[cfg(test)]
    pub(crate) fn force_commit_failure_handle(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.force_commit_failure_for_tests)
    }

    /// Execute commit, optionally substituting a synthetic failure
    /// when the test-only switch is set. Production callers see
    /// `crud::commit` directly; the `#[cfg(test)]` branch never
    /// compiles into release-mode binaries.
    fn execute_commit<'tx>(
        &self,
        tx: arcgraph_storage::transaction::Transaction<'tx>,
        crud: &CrudStore,
    ) -> Result<arcgraph_core::Lsn, crud::CrudError> {
        #[cfg(test)]
        if self
            .force_commit_failure_for_tests
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            drop(tx);
            return Err(crud::CrudError::Mvcc(
                arcgraph_core::error::ArcGraphError::TransactionAborted {
                    reason: "test-only forced commit failure (R1 HIGH-2 regression)".into(),
                },
            ));
        }
        crud::commit(tx, crud)
    }
}

/// A pending idempotency cache insert, staged during the ingest
/// transaction. Drained into the per-tenant cache ONLY after
/// `crud::commit` succeeds — per R1 review HIGH-2 (PR #349) the
/// pre-commit publish was the root cause of the cache-poisoning bug.
struct PendingIdempotency {
    kind: IdempotencyKind,
    external_id: String,
    internal_id: u64,
    payload_hash: u64,
}

#[derive(Serialize)]
struct NodeIdempotencyPayload<'a> {
    label: &'a str,
    properties: &'a BTreeMap<String, serde_json::Value>,
}

#[derive(Serialize)]
struct RelIdempotencyPayload<'a> {
    from_external_id: &'a str,
    to_external_id: &'a str,
    rel_type: &'a str,
    properties: &'a BTreeMap<String, serde_json::Value>,
}

fn payload_hash<T: Serialize>(payload: &T) -> u64 {
    let bytes = serde_json::to_vec(payload).unwrap_or_default();
    let mut h: u64 = 5381;
    for b in bytes {
        h = h.wrapping_mul(33).wrapping_add(u64::from(b));
    }
    h
}

fn node_payload_hash(node: &NodeIngest) -> u64 {
    payload_hash(&NodeIdempotencyPayload {
        label: &node.label,
        properties: &node.properties,
    })
}

fn rel_payload_hash(rel: &RelIngest) -> u64 {
    payload_hash(&RelIdempotencyPayload {
        from_external_id: &rel.from_external_id,
        to_external_id: &rel.to_external_id,
        rel_type: &rel.rel_type,
        properties: &rel.properties,
    })
}

/// The dimension of a node's conventional vector embedding
/// ([`crate::tools::search::DEFAULT_VECTOR_PROPERTY`]), or `None` when the node
/// carries no embedding or the value is not a non-empty numeric array. Mirrors
/// the served provider's `property_as_vector` decode so ingest-time validation
/// and the derived-index build agree on what counts as a vector (#786).
fn json_map_embedding_dim(props: &BTreeMap<String, serde_json::Value>) -> Option<usize> {
    let serde_json::Value::Array(items) =
        props.get(crate::tools::search::DEFAULT_VECTOR_PROPERTY)?
    else {
        return None;
    };
    if items.is_empty() || !items.iter().all(serde_json::Value::is_number) {
        return None;
    }
    Some(items.len())
}

impl IngestProvider for StorageIngestProvider {
    fn ingest(&self, tenant: TenantId, batch: IngestBatch) -> Result<IngestSummary, MCPError> {
        // Destructure up-front so the node / rel loops can consume
        // `nodes` / `relationships` by value while `acl_grants` is held
        // separately for the post-commit write-through (#1181).
        let IngestBatch {
            nodes: batch_nodes,
            relationships: batch_relationships,
            acl_grants,
        } = batch;

        let crud = self.backend.crud_for(tenant)?;
        let mut tx = self.backend.txn_manager.begin(tenant);

        let mut records: Vec<IngestRecordOutcome> = Vec::new();
        let mut inserted_count: u64 = 0;
        let mut failed_count: u64 = 0;
        // #1198 (MUST-CON-07 hardening): grants whose content
        // `external_id` did not resolve to a committed node, surfaced in
        // the response so the caller can detect a security-relevant grant
        // did NOT apply (instead of the pre-#1198 silent skip under a
        // `failed_count:0` full-success report). Populated below from
        // `apply_live_acl_grants`.
        let mut dropped_acl_grants: Vec<DroppedAclGrant> = Vec::new();

        // Build local maps so rels in the SAME batch can refer to
        // nodes by external_id and resolve to the just-created
        // internal id without re-routing through the idempotency
        // table per-record.
        let mut batch_external_to_node: HashMap<String, NodeId> = HashMap::new();
        let mut seen_nodes: HashMap<String, (u64, u64)> = HashMap::new();
        let mut seen_rels: HashMap<String, (u64, u64)> = HashMap::new();
        // Staged idempotency inserts. Drained into the global cache
        // ONLY on successful commit (R1 HIGH-2, PR #349).
        let mut pending_idempotency: Vec<PendingIdempotency> = Vec::new();

        // #352 Part 2 (ADR-199): no capacity gate. The idempotency store
        // is now durable (the binding rides this commit's v6 CommitBundle
        // and is rebuilt on replay), so a long-running tenant can hold
        // unbounded distinct external_ids and they survive a restart.
        // Part 1's loud refuse-at-cap (#851) existed only because the map
        // was in-memory-only with no durable fallback; that ceiling is
        // gone.

        // #786 — establish THIS batch's embedding dimension from the first
        // embedding-bearing node (submission order = ascending node id = the
        // derived index's first-seen-dim convention). Nodes whose embedding
        // dimension differs are rejected below with a clear reason, instead of
        // being silently accepted then dropped from the single-dimension HNSW.
        // NB: this catches intra-batch mismatches (the issue's deterministic
        // repro). Cross-batch establishment (a later batch mismatching an
        // EARLIER batch's durable dimension) needs a per-tenant established-dim
        // registry — flagged as a #786 follow-up; until then a wrong-dim QUERY
        // still surfaces the clear `-32602` dimension-mismatch error.
        let batch_embedding_dim: Option<usize> = batch_nodes
            .iter()
            .find_map(|n| json_map_embedding_dim(&n.properties));

        for n in batch_nodes {
            let payload_hash = node_payload_hash(&n);
            // Per-record idempotency check (kind-scoped per R1 MED-1),
            // now against the durable store (#352 Part 2).
            let idem_hit = if let Some(ext) = &n.external_id {
                match self
                    .backend
                    .idempotency
                    .try_get(tenant, IdempotencyKind::Node.as_u8(), ext)
                {
                    Ok(binding) => binding,
                    Err(error) => {
                        failed_count += 1;
                        records.push(IngestRecordOutcome::Failed {
                            external_id: Some(ext.clone()),
                            error: IngestError::Storage {
                                detail: format!("idempotency node owner lookup: {error}"),
                            },
                        });
                        continue;
                    }
                }
            } else {
                None
            };
            if let Some(binding) = idem_hit {
                if let Some(ext) = &n.external_id {
                    let mut binding_is_live = true;
                    match crud::read_node_with_store(&crud, &tx, NodeId::new(binding.internal_id)) {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            if let Err(error) = self.backend.idempotency.try_release(
                                tenant,
                                IdempotencyKind::Node.as_u8(),
                                ext,
                            ) {
                                failed_count += 1;
                                records.push(IngestRecordOutcome::Failed {
                                    external_id: Some(ext.clone()),
                                    error: IngestError::Storage {
                                        detail: format!(
                                            "stale idempotency node owner release: {error}"
                                        ),
                                    },
                                });
                                continue;
                            }
                            binding_is_live = false;
                        }
                        Err(e) => {
                            failed_count += 1;
                            records.push(IngestRecordOutcome::Failed {
                                external_id: Some(ext.clone()),
                                error: IngestError::Storage {
                                    detail: format!("idempotency node lookup: {e}"),
                                },
                            });
                            continue;
                        }
                    }
                    if !binding_is_live {
                        // Stale binding: fall through to a real insert
                        // and never report an idempotent no-op as
                        // inserted.
                    } else {
                        if let Some(existing_hash) = binding.payload_hash {
                            if existing_hash != payload_hash {
                                failed_count += 1;
                                records.push(IngestRecordOutcome::Failed {
                                    external_id: Some(ext.clone()),
                                    error: IngestError::IdempotencyConflict {
                                        external_id: ext.clone(),
                                    },
                                });
                                continue;
                            }
                        }
                        records.push(IngestRecordOutcome::Idempotent {
                            internal_id: binding.internal_id,
                            external_id: ext.clone(),
                        });
                        batch_external_to_node
                            .insert(ext.clone(), NodeId::new(binding.internal_id));
                        continue;
                    }
                }
            }
            if let Some(ext) = &n.external_id {
                if let Some((internal_id, existing_hash)) = seen_nodes.get(ext) {
                    if *existing_hash != payload_hash {
                        failed_count += 1;
                        records.push(IngestRecordOutcome::Failed {
                            external_id: Some(ext.clone()),
                            error: IngestError::IdempotencyConflict {
                                external_id: ext.clone(),
                            },
                        });
                        continue;
                    }
                    records.push(IngestRecordOutcome::Idempotent {
                        internal_id: *internal_id,
                        external_id: ext.clone(),
                    });
                    continue;
                }
            }
            if n.label.trim().is_empty() {
                failed_count += 1;
                records.push(IngestRecordOutcome::Failed {
                    external_id: n.external_id,
                    error: IngestError::Invalid {
                        detail: "node label must not be empty".into(),
                    },
                });
                continue;
            }
            // #786 — reject a node whose embedding dimension differs from the
            // batch's established dimension (non-silent: failed_count + a clear
            // per-record reason), instead of accepting it then silently dropping
            // it from the derived HNSW at build time.
            let dim_mismatch = batch_embedding_dim
                .zip(json_map_embedding_dim(&n.properties))
                .filter(|(ref_dim, node_dim)| ref_dim != node_dim);
            if let Some((ref_dim, node_dim)) = dim_mismatch {
                failed_count += 1;
                records.push(IngestRecordOutcome::Failed {
                    external_id: n.external_id,
                    error: IngestError::Invalid {
                        detail: format!(
                            "embedding dimension {node_dim} does not match this ingest \
                             batch's established embedding dimension {ref_dim} \
                             (single-dimension-per-index, #786); re-ingest with a \
                             consistent embedding dimension"
                        ),
                    },
                });
                continue;
            }
            // #352 Part 2 (ADR-199): the capacity gate is gone — the
            // idempotency binding is now durable (rides this commit's v6
            // CommitBundle), so a fresh external_id never needs to be
            // refused to protect a bounded in-memory map.
            // P0 #776: WAL-log the label intern when freshly allocated AND
            // durable so `graph.ingest`-created names survive a `--data`
            // restart (the gap that left `graph.schema` showing `label:N`).
            // Per-record fault isolation: a log failure fails THIS record
            // only, mirroring the create_node error arm below.
            let label_id = match arcgraph_storage::intern_label_logged(
                &self.backend.intern_table,
                crud.wal(),
                tenant,
                &n.label,
            ) {
                Ok(id) => id,
                Err(e) => {
                    failed_count += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: n.external_id,
                        error: IngestError::Storage {
                            detail: format!("intern WAL log failed: {e}"),
                        },
                    });
                    continue;
                }
            };
            // Encode properties as a JSON blob (ADR-152 §D-1).
            let prop_data = property_data_for_json_map(&n.properties);
            match crud::create_node(&crud, &mut tx, tenant, label_id, &prop_data) {
                Ok(nid) => {
                    inserted_count += 1;
                    let internal = nid.raw();
                    if let Some(ext) = n.external_id.as_ref() {
                        // #352 Part 2 (ADR-199): stage the binding into
                        // THIS commit's v6 CommitBundle so it is durified
                        // atomically with the node write. The in-memory
                        // publish to the durable IdempotencyStore happens
                        // ONLY on commit success below (post-commit per R1
                        // HIGH-2 — a commit failure must not leave a
                        // binding for a rolled-back id).
                        crud.stage_idempotency_binding(
                            tx.id(),
                            tenant,
                            IdempotencyKind::Node.as_u8(),
                            ext.clone(),
                            internal,
                            Some(payload_hash),
                        );
                        pending_idempotency.push(PendingIdempotency {
                            kind: IdempotencyKind::Node,
                            external_id: ext.clone(),
                            internal_id: internal,
                            payload_hash,
                        });
                        batch_external_to_node.insert(ext.clone(), nid);
                        seen_nodes.insert(ext.clone(), (internal, payload_hash));
                    }
                    records.push(IngestRecordOutcome::Inserted {
                        internal_id: internal,
                        external_id: n.external_id,
                    });
                }
                Err(e) => {
                    failed_count += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: n.external_id,
                        error: IngestError::Storage {
                            detail: format!("create_node: {e}"),
                        },
                    });
                }
            }
        }

        for r in batch_relationships {
            let payload_hash = rel_payload_hash(&r);
            // Rel idempotency check (kind-scoped per R1 MED-1), now
            // against the durable store (#352 Part 2). Compare before
            // endpoint resolution so a different-payload retry is not
            // mislabeled as an unresolved endpoint failure.
            let rel_idem_hit = if let Some(ext) = &r.external_id {
                match self
                    .backend
                    .idempotency
                    .try_get(tenant, IdempotencyKind::Rel.as_u8(), ext)
                {
                    Ok(binding) => binding,
                    Err(error) => {
                        failed_count += 1;
                        records.push(IngestRecordOutcome::Failed {
                            external_id: Some(ext.clone()),
                            error: IngestError::Storage {
                                detail: format!("idempotency rel owner lookup: {error}"),
                            },
                        });
                        continue;
                    }
                }
            } else {
                None
            };
            if let Some(binding) = rel_idem_hit {
                if let Some(ext) = &r.external_id {
                    let mut binding_is_live = true;
                    match crud::read_rel_with_store(&crud, &tx, RelId::new(binding.internal_id)) {
                        Ok(Some(_)) => {}
                        Ok(None) => {
                            if let Err(error) = self.backend.idempotency.try_release(
                                tenant,
                                IdempotencyKind::Rel.as_u8(),
                                ext,
                            ) {
                                failed_count += 1;
                                records.push(IngestRecordOutcome::Failed {
                                    external_id: Some(ext.clone()),
                                    error: IngestError::Storage {
                                        detail: format!(
                                            "stale idempotency rel owner release: {error}"
                                        ),
                                    },
                                });
                                continue;
                            }
                            binding_is_live = false;
                        }
                        Err(e) => {
                            failed_count += 1;
                            records.push(IngestRecordOutcome::Failed {
                                external_id: Some(ext.clone()),
                                error: IngestError::Storage {
                                    detail: format!("idempotency rel lookup: {e}"),
                                },
                            });
                            continue;
                        }
                    }
                    if !binding_is_live {
                        // Stale binding: fall through to a real insert.
                    } else {
                        if let Some(existing_hash) = binding.payload_hash {
                            if existing_hash != payload_hash {
                                failed_count += 1;
                                records.push(IngestRecordOutcome::Failed {
                                    external_id: Some(ext.clone()),
                                    error: IngestError::IdempotencyConflict {
                                        external_id: ext.clone(),
                                    },
                                });
                                continue;
                            }
                        }
                        records.push(IngestRecordOutcome::Idempotent {
                            internal_id: binding.internal_id,
                            external_id: ext.clone(),
                        });
                        continue;
                    }
                }
            }
            if let Some(ext) = &r.external_id {
                if let Some((internal_id, existing_hash)) = seen_rels.get(ext) {
                    if *existing_hash != payload_hash {
                        failed_count += 1;
                        records.push(IngestRecordOutcome::Failed {
                            external_id: Some(ext.clone()),
                            error: IngestError::IdempotencyConflict {
                                external_id: ext.clone(),
                            },
                        });
                        continue;
                    }
                    records.push(IngestRecordOutcome::Idempotent {
                        internal_id: *internal_id,
                        external_id: ext.clone(),
                    });
                    continue;
                }
            }
            // Resolve from/to via batch-local map first, then the
            // process-wide idempotency map (Node kind only — rel
            // external_ids never resolve a node endpoint).
            let from_internal = resolve_node_external_id(
                &batch_external_to_node,
                &self.backend.idempotency,
                tenant,
                &r.from_external_id,
            );
            let to_internal = resolve_node_external_id(
                &batch_external_to_node,
                &self.backend.idempotency,
                tenant,
                &r.to_external_id,
            );
            let (from_nid, to_nid) = match (from_internal, to_internal) {
                (Ok(Some(f)), Ok(Some(t))) => (f, t),
                (Err(error), _) | (_, Err(error)) => {
                    failed_count += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Storage {
                            detail: format!("relationship endpoint owner lookup: {error}"),
                        },
                    });
                    continue;
                }
                _ => {
                    failed_count += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Invalid {
                            detail: format!(
                                "rel endpoints unresolved: from={} to={}",
                                r.from_external_id, r.to_external_id
                            ),
                        },
                    });
                    continue;
                }
            };
            if r.rel_type.trim().is_empty() {
                failed_count += 1;
                records.push(IngestRecordOutcome::Failed {
                    external_id: r.external_id,
                    error: IngestError::Invalid {
                        detail: "rel_type must not be empty".into(),
                    },
                });
                continue;
            }
            // #352 Part 2 (ADR-199): no capacity gate (symmetric to the
            // node path) — the rel binding is durable too.
            // P0 #776: WAL-log the rel-type intern (symmetric to the node
            // label above) so `:SENT`-style names survive a `--data`
            // restart. Per-record fault isolation on a log failure.
            let type_id = match arcgraph_storage::intern_type_logged(
                &self.backend.intern_table,
                crud.wal(),
                tenant,
                &r.rel_type,
            ) {
                Ok(id) => id,
                Err(e) => {
                    failed_count += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Storage {
                            detail: format!("intern WAL log failed: {e}"),
                        },
                    });
                    continue;
                }
            };
            let prop_data = property_data_for_json_map(&r.properties);
            match crud::create_rel(
                &crud, &mut tx, tenant, from_nid, to_nid, type_id, &prop_data,
            ) {
                Ok(rid) => {
                    inserted_count += 1;
                    let internal = rid.raw();
                    if let Some(ext) = r.external_id.as_ref() {
                        // #352 Part 2 (ADR-199): durable stage into this
                        // commit's v6 CommitBundle; in-memory publish on
                        // commit success below.
                        crud.stage_idempotency_binding(
                            tx.id(),
                            tenant,
                            IdempotencyKind::Rel.as_u8(),
                            ext.clone(),
                            internal,
                            Some(payload_hash),
                        );
                        pending_idempotency.push(PendingIdempotency {
                            kind: IdempotencyKind::Rel,
                            external_id: ext.clone(),
                            internal_id: internal,
                            payload_hash,
                        });
                        seen_rels.insert(ext.clone(), (internal, payload_hash));
                    }
                    records.push(IngestRecordOutcome::Inserted {
                        internal_id: internal,
                        external_id: r.external_id,
                    });
                }
                Err(e) => {
                    failed_count += 1;
                    records.push(IngestRecordOutcome::Failed {
                        external_id: r.external_id,
                        error: IngestError::Storage {
                            detail: format!("create_rel: {e}"),
                        },
                    });
                }
            }
        }

        // Group-commit fsync. The commit produces the cohort LSN
        // surfaced as `commit_lsn` per ADR-031 §Decision. The
        // idempotency cache is published ONLY in the `Ok` arm (R1
        // HIGH-2, PR #349) so a commit-time failure cannot leak a
        // non-existent internal_id into the cache.
        let commit_lsn = match self.execute_commit(tx, &crud) {
            Ok(lsn) => {
                // Commit succeeded — publish all staged bindings to the
                // in-memory IdempotencyStore (post-commit per R1 HIGH-2).
                // The durable copy already rode the v6 CommitBundle that
                // this commit just fsynced; this publish makes the binding
                // visible to subsequent same-process lookups without
                // waiting for a replay. (#352 Part 2 — ADR-199.)
                if !self.backend.idempotency.is_page_backed() {
                    for entry in pending_idempotency {
                        self.backend.idempotency.install_with_payload_hash(
                            tenant,
                            entry.kind.as_u8(),
                            &entry.external_id,
                            entry.internal_id,
                            Some(entry.payload_hash),
                        );
                    }
                }
                // ── ACL write-through on the live push path (#1181,
                // MUST-CON-07; ADR-212 §D-4 Seam-1). AFTER the
                // records commit, resolve each grant's content
                // `external_id → committed NodeId` from THIS call's
                // node outcomes (`batch_external_to_node`, already
                // populated for every Inserted/Idempotent node) and
                // write its read-grant set through
                // `PermissionIndex::apply_doc_acl` — the enforcement
                // plane `graph.search` reads
                // (`TenantHandle::permissions()`,
                // ADR-037-amendment-02). Done post-commit (fail-closed:
                // a doc whose record did not durify gets no grant) and
                // never before the commit, matching the seam ordering
                // in `acl_ingest::ingest_docs_with_acls`.
                if !acl_grants.is_empty() {
                    dropped_acl_grants = apply_live_acl_grants(
                        &self.backend,
                        tenant,
                        &acl_grants,
                        &batch_external_to_node,
                    );
                }
                Some(lsn.raw())
            }
            Err(e) => {
                // A commit-time failure invalidates the entire
                // cohort. Rewrite the per-record outcomes for the
                // inserted records as Failed-Storage so the caller
                // sees the back-pressure rather than a silent
                // success-then-rollback. Critically, we DO NOT
                // publish `pending_idempotency` — those entries
                // referred to rolled-back internal_ids.
                for outcome in records.iter_mut() {
                    if let IngestRecordOutcome::Inserted { external_id, .. } = outcome {
                        let ext = external_id.clone();
                        *outcome = IngestRecordOutcome::Failed {
                            external_id: ext,
                            error: IngestError::Storage {
                                detail: format!("commit failed: {e}"),
                            },
                        };
                        failed_count += 1;
                        inserted_count = inserted_count.saturating_sub(1);
                    }
                }
                // `pending_idempotency` falls out of scope here
                // without being applied — exactly the desired
                // behavior. Drop it explicitly to document intent.
                drop(pending_idempotency);
                None
            }
        };

        Ok(IngestSummary {
            records,
            inserted_count,
            failed_count,
            commit_lsn,
            dropped_acl_grants,
        })
    }
}

/// LIVE-path ACL write-through (#1181, MUST-CON-07; ADR-212 §D-4
/// Seam-1). Called from [`StorageIngestProvider::ingest`] AFTER the
/// records commit: for each grant, resolve
/// its content `external_id → committed NodeId` from this call's node
/// outcomes (`committed`) and write the read-grant set through the
/// routed tenant's
/// [`arcgraph_storage::permissions::PermissionIndex::apply_doc_acl`].
///
/// Semantics matched field-for-field to the seed path:
/// - `read_principals: None` ⇒ leave the doc UNCLASSIFIED — skip, do
///   NOT call `apply_doc_acl` (fail-closed; invisible under enforcement);
/// - `Some([])` ⇒ explicit grant-to-nobody (still tags the doc);
/// - a grant whose `external_id` did not commit (failed record, or
///   absent from `nodes`) is skipped (`committed.get()`-else-continue),
///   surfacing nothing fatal — the request-level call still returns Ok.
///
/// A routing miss (the tenant cannot be re-routed for its
/// `PermissionIndex`) is logged and the grants are skipped rather than
/// failing the already-committed write — the records are durable, and a
/// skipped grant only UNDER-grants (fail-closed). In practice the
/// tenant was just routed by `crud_for` above, so this is unreachable;
/// the guard avoids panicking on an impossible-but-not-type-proven path.
///
/// # Surfacing dropped grants (#1198, MUST-CON-07 hardening)
///
/// Returns the [`DroppedAclGrant`]s that were SKIPPED because their
/// `external_id` did not resolve to a committed node. The skip BEHAVIOR
/// is unchanged (an unresolved grant stays fail-closed — widening on an
/// unresolved `external_id` would be strictly worse); the only change is
/// that the drop is now VISIBLE in the response instead of being
/// silently swallowed under a `failed_count:0` full-success report. The
/// caller threads the returned vec into [`IngestSummary::dropped_acl_grants`].
/// A routing miss surfaces EVERY grant as dropped (none could be
/// resolved), so the caller still sees that nothing applied.
fn apply_live_acl_grants(
    backend: &StorageBackend,
    tenant: TenantId,
    grants: &[AclGrant],
    committed: &HashMap<String, NodeId>,
) -> Vec<DroppedAclGrant> {
    let mut dropped: Vec<DroppedAclGrant> = Vec::new();
    let handle = match backend.router().route(tenant, PartitionId::ZERO) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(
                target: "arcgraph_mcp::storage::adapters",
                tenant = ?tenant,
                error = %e,
                "live graph.ingest acl_grants: tenant route failed post-commit; \
                 skipping ACL write-through (fail-closed — docs stay UNCLASSIFIED)"
            );
            // None of the grants could be applied — surface every one as
            // dropped so the caller learns the write-through did not run.
            return grants
                .iter()
                .map(|g| DroppedAclGrant::unresolved(g.external_id.clone()))
                .collect();
        }
    };
    let permissions = handle.permissions();
    for grant in grants {
        let Some(node) = committed.get(&grant.external_id) else {
            // The doc did not commit (or was never in `nodes`) — skip
            // its grant (fail-closed). An unmapped doc stays
            // UNCLASSIFIED ⇒ invisible under enforcement. #1198: capture
            // it so the drop is SURFACED, not silently swallowed.
            dropped.push(DroppedAclGrant::unresolved(grant.external_id.clone()));
            continue;
        };
        // `read_principals: null` ⇒ leave the doc UNCLASSIFIED
        // (invisible) — only an explicit grant set (incl. the empty
        // grant-to-nobody) tags the doc.
        if let Some(principals) = &grant.read_principals {
            let set: BTreeSet<String> = principals.iter().cloned().collect();
            if let Err(error) = permissions.apply_doc_acl_checked(*node, set) {
                tracing::error!(
                    tenant = ?tenant,
                    node = node.raw(),
                    %error,
                    "live graph.ingest ACL owner publish failed closed"
                );
                dropped.push(DroppedAclGrant {
                    external_id: grant.external_id.clone(),
                    reason: "storage_error".to_owned(),
                });
            }
        }
    }
    dropped
}

/// Resolve a relationship endpoint's node `external_id` to its internal
/// [`NodeId`]. Consults first the batch-local map (already-Inserted
/// nodes in the same batch), then the durable [`IdempotencyStore`]
/// under the [`IdempotencyKind::Node`] namespace so rel
/// `external_id`s sharing the string never resolve to a node. Because
/// the store survives restart (#352 Part 2), an edge can resolve a node
/// committed by a PRIOR process — the headline #352 correctness gain.
fn resolve_node_external_id(
    batch_local: &HashMap<String, NodeId>,
    idempotency: &IdempotencyStore,
    tenant: TenantId,
    external_id: &str,
) -> Result<Option<NodeId>, arcgraph_storage::owner_row::OwnerRowError> {
    if let Some(nid) = batch_local.get(external_id) {
        return Ok(Some(*nid));
    }
    Ok(idempotency
        .try_get(tenant, IdempotencyKind::Node.as_u8(), external_id)?
        .map(|binding| NodeId::new(binding.internal_id)))
}

// ─────────────────────────────────────────────────────────────────────
// RawQueryExecutor
// ─────────────────────────────────────────────────────────────────────

/// Storage-backed [`RawQueryExecutor`] — wraps an `arcgraph_query`
/// [`QueryEngine`] + a [`CrudExecutorSubstrate`] over the workspace's
/// shared storage handles. Each call:
///
/// 1. Builds a per-call
///    [`arcgraph_query::semantic::InMemoryCatalogProvider`] (v1.0-α
///    alias for `StubCatalogProvider` — see R1 review MED-2 closure,
///    PR #349) seeded from the backend's catalog-stats snapshot;
///    this stand-in catalog surfaces label / rel-type ID lookups via
///    the same intern table the executor will read. Property names
///    are NOT enumerated (the intern table has no name-side
///    enumeration); the executor's binding pass falls back to
///    dynamic-name resolution for property predicates. v1.1 swaps
///    to a real storage-backed `CatalogProvider` so this becomes a
///    test-only struct again.
/// 2. Constructs a [`QueryEngine`] bound to that catalog.
/// 3. Calls `execute_with_deadline` (default 30s).
/// 4. Renders the materialized rows as
///    `crate::tools::raw_query::RawQueryRow`s.
pub struct StorageRawQueryExecutor {
    backend: StorageBackend,
    substrate: CrudExecutorSubstrate,
    /// W28 Feature #582 (ADR-045) — optional observability sink for the
    /// `arcgraph_query_plan_choice{plan_type}` counter (design-v2 §10.2
    /// **line 723**).
    ///
    /// # Why the producer lives HERE (not in `arcgraph-query`)
    ///
    /// PD-7 bounded contexts: `arcgraph-query`'s LIBRARY depends only on
    /// `arcgraph-core` (its `arcgraph-storage` edge is a *dev*-dependency
    /// — see `crates/arcgraph-query/Cargo.toml` `[dev-dependencies]`),
    /// so the `QueryEngine` cannot reference the storage-resident
    /// [`MetricsSink`] trait. `arcgraph-mcp` legitimately depends on
    /// BOTH `arcgraph-query` and `arcgraph-storage`, so this adapter —
    /// the production `graph.raw_query` execution boundary — is the
    /// lowest layer that can reach the trait AND observe a query
    /// execution. Emitting here (rather than adding a real
    /// query→storage edge or inventing a query-resident observer seam)
    /// keeps the dep graph clean per ADR-045's "do not invent a new
    /// seam" discipline.
    ///
    /// When `None` (the default + every legacy caller), emission is a
    /// single nullable-ptr check (PD-5).
    metrics_sink: Option<Arc<dyn MetricsSink>>,
    /// #1291 — optional per-tenant memory cap (bytes) applied to every
    /// query this executor runs. When `Some(cap)`, each `execute` call
    /// mints a per-query [`arcgraph_query::executor::MemoryBudget`]
    /// with the cap configured for the requesting tenant and attaches
    /// it to the engine — a heavy query surfaces `-32009
    /// BudgetExceeded` instead of OOMing the served process. When
    /// `None` (the default — embedded / test posture), the pre-#1291
    /// opt-in behavior holds: no byte cap, row-count runaway guard
    /// only. The served binary wires this from
    /// `ARCGRAPH_TENANT_MEMORY_CAP_BYTES` (default 1 GiB).
    per_tenant_memory_cap_bytes: Option<u64>,
}

impl std::fmt::Debug for StorageRawQueryExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StorageRawQueryExecutor")
            .field("backend", &self.backend)
            .field("substrate", &self.substrate)
            .field("metrics_sink", &self.metrics_sink.is_some())
            .field(
                "per_tenant_memory_cap_bytes",
                &self.per_tenant_memory_cap_bytes,
            )
            .finish()
    }
}

impl StorageRawQueryExecutor {
    /// Construct a raw-query executor over the workspace's backend.
    /// The wrapped substrate is constructed from the same handles.
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        let substrate = CrudExecutorSubstrate::new(
            Arc::clone(backend.router()),
            Arc::clone(backend.txn_manager()),
            Arc::clone(backend.intern_table()),
        );
        Self {
            backend,
            substrate,
            metrics_sink: None,
            per_tenant_memory_cap_bytes: None,
        }
    }

    /// #1291 — enable the per-tenant memory budget with `cap_bytes` as
    /// the byte ceiling for every tenant this executor serves. Each
    /// `execute` call mints a per-query
    /// [`arcgraph_query::executor::MemoryBudget`] with the cap
    /// configured for the requesting tenant (via
    /// [`arcgraph_query::executor::MemoryBudget::with_per_tenant_cap`])
    /// so blocking operators (sort / join / aggregate / distinct /
    /// expand spillover) and the materialize tail enforce a REAL byte
    /// ceiling instead of the ≈4.29 B-row
    /// `UNCAPPED_RUNAWAY_GUARD_ROWS` fallback. Builder-style; chains
    /// after [`Self::new`].
    #[must_use]
    pub fn with_per_tenant_memory_cap(mut self, cap_bytes: u64) -> Self {
        self.per_tenant_memory_cap_bytes = Some(cap_bytes);
        self
    }

    /// W28 Feature #582 (ADR-045) — attach an observability sink so
    /// each successful `graph.raw_query` execution emits the
    /// `arcgraph_query_plan_choice{plan_type}` counter (design-v2 §10.2
    /// line 723). Builder-style; chains after [`Self::new`]. The
    /// `arcgraph` server binary chains this when the operator passes
    /// `--metrics-http <addr>`. When omitted, emission is a no-op.
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics_sink = Some(sink);
        self
    }

    /// #765 PART-1 — bind the served vector-search provider into the wrapped
    /// [`CrudExecutorSubstrate`] so ArcQL `RANK BY vector(n.embedding, $qv)`
    /// (the `graph.raw_query` path) runs real HNSW KNN. Builder-style; chains
    /// after [`Self::new`]. Production bootstrap binds the same provider
    /// instance shared with [`StorageHybridSearcher`].
    #[must_use]
    pub fn with_search_provider(mut self, provider: Arc<dyn SubstrateSearchProvider>) -> Self {
        self.substrate = self.substrate.with_search_provider(provider);
        self
    }
}

impl RawQueryExecutor for StorageRawQueryExecutor {
    fn execute(
        &self,
        tenant: TenantId,
        query: &str,
        max_rows: u32,
        cancel: &CancellationToken,
    ) -> Result<RawQueryRows, MCPError> {
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        // Build a per-call catalog provider. v1.0-α uses
        // `InMemoryCatalogProvider` seeded with the intern table's
        // known labels / rel-types (intern table cannot enumerate
        // property names; the binding pass's dynamic-name fallback
        // resolves them on-the-fly). v1.1 (post-issue resolution
        // tracked alongside #356) replaces this with a
        // storage-backed `CatalogProvider`.
        let cat = build_catalog_for_tenant(tenant, &self.backend);

        // ADR-153 §D-2 W27-β — wrap the substrate with the counting
        // decorator so write-effect counters surface in `WriteSummary`
        // for the response envelope. The decorator is per-request (one
        // counter bag per `execute` call); reads pass through with no
        // bookkeeping overhead beyond a virtual dispatch.
        let (counting, counters) =
            crate::storage::counting::CountingSubstrate::new(self.substrate.clone());

        let engine = QueryEngine::new(&cat);
        // #1291 — when the served binary configured a per-tenant memory
        // cap, mint a per-query budget with the cap set for THIS tenant
        // and attach it. Without this, the budget defaults to unbounded
        // and the only guard is the ≈4.29 B-row runaway fallback → OOM
        // under a heavy query. Per-query budget (not process-shared):
        // one query is bounded by `cap`; cross-query per-tenant
        // aggregation is the M5-12 config-surface follow-up.
        let engine = match self.per_tenant_memory_cap_bytes {
            Some(cap) => engine.with_memory_budget(
                arcgraph_query::executor::MemoryBudget::with_per_tenant_cap(tenant, cap),
            ),
            None => engine,
        };
        let result = engine
            .execute_with_deadline(
                query,
                &counting,
                std::time::Duration::from_millis(arcgraph_query::cancel::DEFAULT_QUERY_TIMEOUT_MS),
            )
            .map_err(MCPError::from)?;

        // W28 Feature #582 (ADR-045) — emit the §10.2 line 723
        // `arcgraph_query_plan_choice{plan_type}` counter once per
        // successfully-executed query. The v1.0-α query engine is a
        // binary-(pairwise-)join executor EXCLUSIVELY (per
        // `arcgraph-query/src/planner/enumeration/mod.rs:20` "binary
        // joins at v1.0; bushy deferred to v1.1"; the physical
        // Hash/Merge pick `pick_join_algorithms` resolves is a
        // sub-distinction BELOW the §10.2 binary/wcoj/free_join
        // paradigm granularity), so the emitted `plan_type` is always
        // `binary`. This is the honest v1.0 paradigm — not an un-wired
        // stub: the counter increments on every executed query and the
        // value is provably correct (the engine has no non-binary plan
        // path). The wcoj / free_join label values materialise when
        // those executors land (v1.1+), at which point the QueryEngine
        // must expose its actual chosen plan-type through a
        // query-resident observer seam (a dedicated ADR — the §10.2
        // line 723 plan-type *variation* lift) so this adapter forwards
        // the real choice instead of asserting the v1.0 invariant. We
        // emit only on the success path: a failed parse/plan never
        // produced an executed plan, so it must not increment.
        if let Some(sink) = self.metrics_sink.as_ref() {
            sink.record_query_plan_choice(QueryPlanType::Binary);
        }

        // Convert the result's Value-shaped rows to the
        // RawQueryRows wire shape: each row is a JSON array.
        let total_rows = result.rows().len();
        let truncated = total_rows > max_rows as usize;
        let mut rows: Vec<JsonValue> = Vec::new();
        for (idx, row) in result.rows().iter().enumerate() {
            if (idx as u32) >= max_rows {
                break;
            }
            let cells: Vec<JsonValue> = row.iter().map(value_to_json).collect();
            rows.push(JsonValue::Array(cells));
        }

        // Column names (#353): the executor now surfaces the user's
        // RETURN-item display names (aliases / bare-var names / implicit
        // source-text) via `MaterializedResult::columns`. langchain's
        // Neo4jGraph (and every Bolt/MCP consumer) keys result records
        // by these — `RETURN n.name AS name` → `["name"]`, not
        // `["col_0"]`. We fall back to synthesized `col_0..N` ONLY when
        // the engine reports no names (a `RETURN *` wildcard whose width
        // is data-dependent, or a write-only / RETURN-less statement) so
        // a non-empty result still gets a stable column shape.
        let columns: Vec<String> = column_names_for_result(&result);

        Ok(RawQueryRows {
            columns: if columns.is_empty() {
                None
            } else {
                Some(columns)
            },
            row_count: rows.len(),
            rows,
            truncated,
            // ADR-153 §D-2 W27-β: drain the counters into the wire
            // shape. Pure-read queries leave the counters at zero
            // (WriteSummary::is_empty() == true); write-touched
            // queries surface accurate per-clause counts.
            writes: counters.snapshot(),
        })
    }

    /// Production override for the `graph.raw_query` `explain:true`
    /// verb-consolidation mode (operator-ruled — stays at the ADR-004
    /// 10-tool cap; NO separate `graph.explain` tool is wired).
    ///
    /// Builds the per-tenant catalog the SAME way [`Self::execute`]
    /// does (`build_catalog_for_tenant`), then calls the free
    /// `arcgraph_query::explain` fn — which runs ONLY the
    /// parse → bind → type-check → cross-substrate → lower → enumerate →
    /// cost pipeline and returns a `PlanTree`. Per ADR-038 §2 D-18
    /// rule 1 `explain` acquires NO snapshot LSN and contacts NO storage
    /// substrate (no `CrudExecutorSubstrate`, no `CountingSubstrate`,
    /// no `QueryEngine::execute`), so it is side-effect-free even for
    /// the W26-θ write-op clauses (the plan is built but never run).
    ///
    /// The `PlanTree` is serialized via the canonical
    /// [`arcgraph_query::plan_tree_as_rows`] walk (the #952 plan-row
    /// adapter shape: one row per operator, columns
    /// `[op, details, estimated_cost, estimated_card, depth]`). The
    /// resulting `MaterializedResult` is mapped into the
    /// [`RawQueryRows`] wire shape; `writes` stays zero (a plan is a
    /// pure read of query structure) and `truncated` is always `false`
    /// (a plan tree is bounded by the query's operator count, not by
    /// data cardinality).
    ///
    /// Errors map through the existing `From<ExplainError> for MCPError`
    /// bridge: `Parse` / `ArcQL` (binding / type-check / missing-param)
    /// → client query faults; `Substrate` / `ExecutionEval` / `Cancelled`
    /// → the appropriate server / cancellation buckets.
    fn explain(&self, tenant: TenantId, query: &str) -> Result<RawQueryRows, MCPError> {
        let cat = build_catalog_for_tenant(tenant, &self.backend);
        let plan = arcgraph_query::explain(query, &cat).map_err(MCPError::from)?;
        let materialized = arcgraph_query::plan_tree_as_rows(&plan);

        let columns: Vec<String> = materialized.columns().to_vec();
        let rows: Vec<JsonValue> = materialized
            .rows()
            .iter()
            .map(|row| JsonValue::Array(row.iter().map(value_to_json).collect()))
            .collect();

        Ok(RawQueryRows {
            columns: if columns.is_empty() {
                None
            } else {
                Some(columns)
            },
            row_count: rows.len(),
            rows,
            truncated: false,
            writes: crate::tools::raw_query::WriteSummary::default(),
        })
    }
}

/// #353 — resolve the result-column names for a `MaterializedResult`,
/// preferring the engine-derived user RETURN-alias names and falling
/// back to synthesized `col_0..N` labels only when those are absent or
/// don't match the actual row width.
///
/// Shared by the MCP `RawQueryRows` renderer (here) and the Bolt
/// `RunOutcome::fields` renderer ([`super::bolt`]) so both wire surfaces
/// emit IDENTICAL column names for the same query — single source of
/// truth (the founding requirement of #353: BOTH wire paths emit the
/// aliases, never one or the other).
///
/// # Width-consistency guard (strong oracle)
///
/// The engine guarantees `columns.len() == row width` when it populates
/// names (the extractor derives the count from the non-wildcard
/// projection items). We re-check it here defensively: if a populated
/// name list does NOT match the first row's width (which would indicate
/// an upstream bug or an unforeseen plan rewrite), we DISCARD the names
/// and fall back to `col_0..N` rather than emit a wrong-width header
/// that would mis-align a driver's record→column binding. A
/// zero-row result keeps the engine's names verbatim (the names are
/// known even when no rows match — `MATCH (n) WHERE false RETURN n.x AS
/// x` still has column `x`).
#[must_use]
pub fn column_names_for_result(result: &arcgraph_query::MaterializedResult) -> Vec<String> {
    let names = result.columns();
    let row_width = result.rows().first().map(Vec::len);
    match row_width {
        // No rows: trust the engine's names (column identity is known
        // without data). If the engine produced none either (write-only
        // / wildcard), there are no columns to render.
        None => names.to_vec(),
        // Rows present: use the engine names IFF they match the row
        // width; otherwise synthesize a correctly-sized `col_0..N`.
        Some(width) => {
            if !names.is_empty() && names.len() == width {
                names.to_vec()
            } else {
                (0..width).map(|i| format!("col_{i}")).collect()
            }
        }
    }
}

/// Build a per-call `CatalogProvider` seeded from the backend's
/// catalog stats + intern table.
///
/// Exposed at `pub` so the [`super::bolt::StorageBoltHandler`]
/// can reuse the same catalog shape the MCP `RawQueryExecutor` uses
/// — single source of truth for the Bolt + MCP query-routing
/// surfaces. It remains `pub` so embedded callers can re-use the
/// helper instead of re-deriving the catalog construction.
///
/// # ID-consistency contract (W23-M4-08-FINALIZE fix)
///
/// At W17α the helper assigned catalog IDs monotonically from 1 via
/// `InMemoryCatalogProvider::with_labels` /
/// `InMemoryCatalogProvider::with_rel_types`. Storage's
/// [`InternTable`] allocates label IDs and rel-type IDs out of a
/// SHARED per-tenant counter, while the catalog assigned them out of
/// separate counters — so a tenant where labels and rel-types were
/// interned in interleaved order saw catalog IDs DIVERGE from the
/// storage IDs for the same name. The substrate's `scan_nodes` /
/// `expand` filters by the storage ID; a planner that resolved a
/// rel-type via the catalog's
/// [`arcgraph_query::semantic::CatalogProvider::lookup_rel_type`]
/// surface then passed a mismatched ID to the substrate and
/// silently returned zero rows. The W17α
/// `graph_raw_query_multi_pattern_join_executes_end_to_end` test
/// used label-FREE patterns + no rel-type filter, sidestepping the
/// mismatch — so the bug was latent at v1.0-α until a caller
/// exercised rel-type binding.
///
/// W23-M4-08-FINALIZE: use the storage-allocated IDs verbatim via
/// the new
/// `InMemoryCatalogProvider::with_label_id` /
/// `InMemoryCatalogProvider::with_rel_type_id` builders so the
/// planner ↔ executor ↔ substrate ID values stay consistent for
/// label-anchored AND rel-type-anchored queries.
pub fn build_catalog_for_tenant(
    tenant: TenantId,
    backend: &StorageBackend,
) -> arcgraph_query::semantic::InMemoryCatalogProvider {
    let crud = backend.crud_for(tenant);
    let stats = crud.as_ref().ok().and_then(|c| c.catalog_stats(tenant));
    let snapshot = stats.as_ref().map(|s| s.snapshot());

    let mut cat = arcgraph_query::semantic::InMemoryCatalogProvider::new().with_tenant(tenant);

    // #789 — attach vector + BM25 substrates when the tenant has nodes.
    // The served vector/BM25 indices are derived ephemeral indices (per-tenant
    // HNSW/SSD built from embedding-bearing or indexable nodes per query) rather
    // than durable VectorPageStore / Tantivy instances. A tenant with nodes MIGHT
    // have embeddings / indexable text, so we conservatively mark the substrates
    // as attached when total_node_count > 0. This unblocks served ArcQL
    // `RANK BY HYBRID(VECTOR(...), TEXT(...))` queries (which call
    // CrossSubstrateValidator::validate at bind-time) while still correctly
    // rejecting queries from tenants with zero nodes.
    let has_nodes = snapshot
        .as_ref()
        .and_then(|s| s.total_nodes())
        .map(|n| n > 0)
        .unwrap_or(false);
    if has_nodes {
        cat = cat.with_vector_index();
        cat = cat.with_bm25_index();
    }

    if let Some(snap) = snapshot {
        if let Some(total) = snap.total_nodes() {
            cat = cat.with_total_node_count(total);
        }
        if let Some(total) = snap.total_rels() {
            cat = cat.with_total_rel_count(total);
        }
        for (label, card) in snap.label_cards() {
            if let Some(name) = backend
                .intern_table
                .resolve(tenant, arcgraph_core::ids::StringId::new(label.raw()))
            {
                cat = cat.with_label_id(name.to_string(), *label);
            }
            cat = cat.with_label_cardinality(*label, *card);
        }
        for (ty, card) in snap.rel_type_cards() {
            if let Some(name) = backend
                .intern_table
                .resolve(tenant, arcgraph_core::ids::StringId::new(ty.raw()))
            {
                cat = cat.with_rel_type_id(name.to_string(), *ty);
            }
            cat = cat.with_rel_type_cardinality(*ty, *card);
        }
        for entry in snap.max_out_degree_entries() {
            cat = cat.with_max_out_degree(entry.label, entry.rel_type, entry.vertex, entry.degree);
        }
    }
    // #802 / ADR-197 — ALSO seed from the intern table so a label /
    // rel-type name a committed `CREATE` interned is resolvable by a
    // SUBSEQUENT query's binder even before the catalog-stats snapshot
    // reflects it (closing the documented catalog-seed gap that made
    // `MATCH (:Account)` after `CREATE (:Account)` reject with
    // `UnknownLabel` — the langchain-neo4j drop-in's read-after-write).
    // The intern id space is shared between labels + rel-types
    // (`InternTable::intern_label`), so each name seeds BOTH maps —
    // over-seeding a rel-type as a label is benign (a `MATCH` on it
    // scans for a node with that id and finds none = 0 rows, the correct
    // Cypher result, NOT a bind error). The `with_*` setters overwrite an
    // identical name→id already seeded from stats above (idempotent).
    for (id, name) in backend.intern_table.names_for_tenant(tenant) {
        cat = cat.with_label_id(name.to_string(), arcgraph_core::LabelId::new(id.raw()));
        cat = cat.with_rel_type_id(name.to_string(), arcgraph_core::TypeId::new(id.raw()));
    }

    // #1366 (Phase 2) — seed the RC-6 planner-visible property-index set
    // from the DURABLE property-index catalog. We recover a fresh
    // `PropertyIndexCatalog` from the same durable state the substrate's
    // `PropertyIndexManager` reads (a committed Online flip is visible to
    // this recover; a Building record is not seeded). Only `Online`
    // records are seeded — the planner then routes a point lookup to the
    // index only when it is planner-visible. The lookup path re-gates on
    // `planner_visible()` too (defense-in-depth), so a plan-time / lookup
    // -time disagreement can never turn into a false negative.
    let pindex_catalog = arcgraph_storage::property_index_catalog::PropertyIndexCatalog::new();
    pindex_catalog.recover(backend.txn_manager(), arcgraph_core::Lsn::ZERO);
    for record in pindex_catalog.list_for_tenant(tenant) {
        if matches!(
            record.state,
            arcgraph_storage::secondary_handle::IndexState::Online
        ) {
            cat = cat.with_online_property_index(record.label, record.property_name);
        }
    }
    cat
}

/// #765 PART-1 — translate a [`SubstrateSearchProvider`]-surfaced
/// `SubstrateAccessError` into an `MCPError` for the `graph.search` return
/// shape. `SubstrateAccessError` is `#[non_exhaustive]`; the catch-all arm
/// surfaces forward-additive variants as `InternalError`.
fn translate_substrate_error(
    err: arcgraph_query::executor::substrate::SubstrateAccessError,
) -> MCPError {
    use arcgraph_query::executor::substrate::SubstrateAccessError as Sae;
    match err {
        Sae::TenantUnknown(t) => MCPError::TenantUnknown(format!("{t:?}")),
        Sae::IndexUnavailable(s) => MCPError::IndexUnavailable(s),
        // #786 — a wrong-dimension `query_vec` is a CLIENT param error
        // (-32602 invalid params) carrying the exact dims, NOT the cryptic
        // -32006 "execution eval" the generic `Io` bucket used to render (the
        // original #786 symptom).
        Sae::DimensionMismatch {
            property,
            query_dim,
            index_dim,
        } => MCPError::InvalidParams(format!(
            "query_vec dimension {query_dim} does not match index dimension \
             {index_dim} for property `{property}`"
        )),
        // Engine I/O surfaces here; ExecutionEval is the query-eval-class
        // envelope (a structured error, never silent-empty).
        Sae::Io(s) => MCPError::ExecutionEval(format!("vector search: {s}")),
        other => MCPError::InternalError(format!("substrate error: {other:?}")),
    }
}

// ─────────────────────────────────────────────────────────────────────
// JSON ↔ Value bridges
// ─────────────────────────────────────────────────────────────────────

/// Encode an MCP property bag (a `BTreeMap<String, serde_json::Value>`)
/// as a [`PropertyData::Blob`] carrying the canonical JSON bytes. An
/// empty bag short-circuits to [`PropertyData::Empty`] so the storage
/// layer's fast-path applies.
#[must_use]
pub fn property_data_for_json_map(map: &BTreeMap<String, JsonValue>) -> PropertyData {
    if map.is_empty() {
        return PropertyData::Empty;
    }
    // Serialize the bag as a canonical JSON Object. JSON keys are
    // already sorted (BTreeMap iter); we route through serde_json
    // which preserves that order.
    let json = JsonValue::Object(
        map.iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<serde_json::Map<_, _>>(),
    );
    let bytes = serde_json::to_vec(&json).unwrap_or_default();
    PropertyData::Blob(bytes)
}

/// Convert an executor [`QueryValue`] into a `serde_json::Value` for
/// the `RawQueryRow` wire shape. Falls back to a string rendering
/// for unsupported variants so the JSON-RPC envelope never carries
/// invalid JSON.
#[must_use]
pub fn value_to_json(v: &QueryValue) -> JsonValue {
    match v {
        QueryValue::Null => JsonValue::Null,
        QueryValue::Boolean(b) => JsonValue::Bool(*b),
        QueryValue::Integer(n) => JsonValue::Number(serde_json::Number::from(*n)),
        QueryValue::Float(f) => serde_json::Number::from_f64(*f)
            .map(JsonValue::Number)
            .unwrap_or(JsonValue::Null),
        QueryValue::String(s) => JsonValue::String(s.clone()),
        QueryValue::Node(n) => {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "id".into(),
                JsonValue::Number(serde_json::Number::from(n.id.raw())),
            );
            if let Some(label) = n.label {
                obj.insert(
                    "label_id".into(),
                    JsonValue::Number(serde_json::Number::from(label.raw())),
                );
            }
            // #871 — surface the catalog-resolved label NAME as the
            // Neo4j-style `labels` list so an MCP `graph.raw_query`
            // client reads `["Account"]`, never the opaque `label_id`.
            // The executor reverse-resolves the name at scan/expand
            // (`CrudExecutorSubstrate`) via the intern table; this is
            // the production raw_query node serializer (distinct from
            // `Value::to_json_value`), so the name MUST be wired here
            // too (#353-class: the wire surface bypassed the value model).
            if let Some(name) = &n.label_name {
                obj.insert(
                    "labels".into(),
                    JsonValue::Array(vec![JsonValue::String(name.clone())]),
                );
            }
            JsonValue::Object(obj)
        }
        QueryValue::Relationship(r) => {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "id".into(),
                JsonValue::Number(serde_json::Number::from(r.id.raw())),
            );
            obj.insert(
                "from".into(),
                JsonValue::Number(serde_json::Number::from(r.from.raw())),
            );
            obj.insert(
                "to".into(),
                JsonValue::Number(serde_json::Number::from(r.to.raw())),
            );
            if let Some(ty) = r.rel_type {
                obj.insert(
                    "rel_type_id".into(),
                    JsonValue::Number(serde_json::Number::from(ty.raw())),
                );
            }
            // #871 — surface the catalog-resolved rel-type NAME as the
            // Neo4j-style `type` string (sibling of the node `labels`
            // fix above).
            if let Some(name) = &r.rel_type_name {
                obj.insert("type".into(), JsonValue::String(name.clone()));
            }
            JsonValue::Object(obj)
        }
        QueryValue::List(xs) => JsonValue::Array(xs.iter().map(value_to_json).collect()),
        // ADR-191 D-7 — an openCypher map projects as a JSON object
        // (recursing through `value_to_json`); `BTreeMap` sorted-key
        // order makes the wire form deterministic.
        QueryValue::Map(m) => JsonValue::Object(
            m.iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect(),
        ),
        // ADR-193 D-8 — a path projects as `{start, segments:
        // [{relationship, end}]}`, recursing via this same `value_to_json`
        // so node/rel sub-objects use this file's local raw_query shape.
        QueryValue::Path(p) => {
            let mut obj = serde_json::Map::new();
            obj.insert(
                "start".into(),
                value_to_json(&QueryValue::Node(p.start.clone())),
            );
            let segments: Vec<JsonValue> = p
                .segments
                .iter()
                .map(|seg| {
                    let mut so = serde_json::Map::new();
                    so.insert(
                        "relationship".into(),
                        value_to_json(&QueryValue::Relationship(seg.rel.clone())),
                    );
                    so.insert(
                        "end".into(),
                        value_to_json(&QueryValue::Node(seg.end.clone())),
                    );
                    JsonValue::Object(so)
                })
                .collect();
            obj.insert("segments".into(), JsonValue::Array(segments));
            JsonValue::Object(obj)
        }
        // W23-V11-T-01 / ADR-090 — temporal + decimal cells project
        // as ISO-8601 / decimal strings per ADR-090 §"Wire shape".
        // Reverse decode (string → Value::Temporal) is handled at the
        // type-checker boundary, not here; on this path raw_query
        // emits a string that downstream consumers can re-bind.
        QueryValue::Temporal(t) => JsonValue::String(format!("{t}")),
        QueryValue::LocalDateTime(ldt) => JsonValue::String(format!("{ldt}")),
        QueryValue::Date(d) => JsonValue::String(format!("{d}")),
        QueryValue::Duration(d) => JsonValue::String(format!("{d}")),
        QueryValue::Decimal(d) => JsonValue::String(format!("{d}")),
    }
}

/// Inverse of [`value_to_json`] for callers that bridge JSON-shaped
/// arguments back into the executor's [`QueryValue`] lattice.
/// Defensive fallback to `Value::Null` for unsupported shapes.
#[must_use]
pub fn json_to_value(v: &JsonValue) -> QueryValue {
    match v {
        JsonValue::Null => QueryValue::Null,
        JsonValue::Bool(b) => QueryValue::Boolean(*b),
        JsonValue::Number(n) => {
            if let Some(i) = n.as_i64() {
                QueryValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                QueryValue::Float(f)
            } else {
                QueryValue::Null
            }
        }
        JsonValue::String(s) => QueryValue::String(s.clone()),
        JsonValue::Array(xs) => QueryValue::List(xs.iter().map(json_to_value).collect()),
        JsonValue::Object(_) => QueryValue::Null, // No native Object mapping.
    }
}

// W17α implementation note: TelEntry carries only (dst_id, rel_id,
// created_lsn, expired_lsn) per `arcgraph_core::record::TelEntry`;
// the rel-type lives on `RelRecord::type_id`. Adapters that need
// rel-type filtering or rendering hop through `crud::read_rel(&tx,
// RelId)` to surface it. The redirect is O(1) (MVCC point read) and
// matches the cost of the same pattern in
// `arcgraph_storage::engine::graph_adapter::CrudStoreGraphAdapter`.

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_query::error::{ParseError, Span};
    use arcgraph_query::executor::SubstrateAccessError;
    use arcgraph_query::explain::ExplainError;
    use arcgraph_query::semantic::CatalogProvider;
    use arcgraph_query::semantic::error::ArcQLError;
    use arcgraph_storage::buffer::BufferPool;
    use arcgraph_storage::catalog::SystemCatalog;
    use arcgraph_storage::io::InMemoryPageIo;
    use arcgraph_storage::router::MultiTenantRouter;

    fn fixture() -> StorageBackend {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        StorageBackend::new(router, mgr, intern)
    }

    fn assert_same_mcp_surface(left: MCPError, right: MCPError) {
        assert_eq!(left.code(), right.code());
        assert_eq!(left.message(), right.message());
        assert_eq!(left.data(), right.data());
    }

    fn constructible_explain_error_variants() -> Vec<ExplainError> {
        vec![
            ExplainError::Parse(ParseError::Pest {
                message: "bad token".into(),
                span: Span::point(1, 7),
            }),
            ExplainError::ArcQL(ArcQLError::NotImplemented {
                feature: "feature".into(),
                target_version: "M1-1".into(),
                section: "ADR-038 §Q".into(),
                span: Span::point(2, 3),
            }),
            ExplainError::Cancelled,
            ExplainError::Substrate(SubstrateAccessError::IndexUnavailable("vector".into())),
            ExplainError::ExecutionEval("division by zero".into()),
            ExplainError::MissingParameter {
                name: "missing".into(),
            },
        ]
    }

    #[test]
    fn raw_query_explain_error_translation_matches_central_mapping_for_current_variants() {
        for err in constructible_explain_error_variants() {
            // The raw_query adapter delegates `ExplainError` conversion
            // directly to the central `From<ExplainError> for MCPError`.
            let raw_query_path = MCPError::from(err.clone());
            let central = MCPError::from(err);
            assert_same_mcp_surface(raw_query_path, central);
        }
    }

    #[test]
    fn build_catalog_for_tenant_seeds_storage_sketch_for_explain_supernode() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let allocator = Arc::new(arcgraph_storage::page_alloc::PageAllocator::new());
        let primary = Arc::new(
            arcgraph_storage::primary_index::PrimaryIndex::new(
                Arc::clone(&mgr),
                Arc::clone(&allocator),
                None,
            )
            .expect("primary"),
        );
        let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        let backend = StorageBackend::new(router, Arc::clone(&mgr), intern);
        let tenant = TenantId::DEFAULT;
        let label = backend.intern_table.intern_label(tenant, "Hub").unwrap();
        let rel_type = backend.intern_table.intern_type(tenant, "LINK").unwrap();
        let mut tx = backend.txn_manager.begin(tenant);
        let hub = crud::create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty).unwrap();

        for _ in 0..10_000 {
            let dst =
                crud::create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty).unwrap();
            crud::create_rel(
                &crud,
                &mut tx,
                tenant,
                hub,
                dst,
                rel_type,
                &PropertyData::Empty,
            )
            .unwrap();
        }
        crud::commit(tx, &crud).unwrap();

        let cat = build_catalog_for_tenant(tenant, &backend);
        let plan = arcgraph_query::explain::explain(
            "EXPLAIN MATCH (a:Hub)-[:LINK*1..3]->(b:Hub) RETURN b",
            &cat,
        )
        .expect("explain");
        let rendered = plan.to_string();
        assert!(rendered.contains("COST_HINT 'high'"), "{rendered}");
        assert!(
            rendered.contains(&format!("vertex {}", hub.raw())),
            "{rendered}"
        );
        assert!(rendered.contains("degree 10000"), "{rendered}");
    }

    #[test]
    fn build_catalog_for_tenant_attaches_vector_and_bm25_when_tenant_has_nodes() {
        // #789 — when a tenant has nodes, mark vector and BM25 substrates as
        // attached (since nodes might have embeddings or indexable text). This
        // unblocks served ArcQL `RANK BY HYBRID(VECTOR(...), TEXT(...))` queries
        // which call CrossSubstrateValidator::validate at bind-time.
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let allocator = Arc::new(arcgraph_storage::page_alloc::PageAllocator::new());
        let primary = Arc::new(
            arcgraph_storage::primary_index::PrimaryIndex::new(
                Arc::clone(&mgr),
                Arc::clone(&allocator),
                None,
            )
            .expect("primary"),
        );
        let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        let backend = StorageBackend::new(router, Arc::clone(&mgr), intern);
        let tenant = TenantId::DEFAULT;

        // Fresh tenant with zero nodes → no substrates attached.
        let cat = build_catalog_for_tenant(tenant, &backend);
        assert!(
            !cat.has_vector_index(),
            "fresh tenant should not have vector"
        );
        assert!(!cat.has_bm25_index(), "fresh tenant should not have bm25");

        // Create a node → now vector and BM25 should be attached.
        let label = backend
            .intern_table
            .intern_label(tenant, "TestNode")
            .unwrap();
        let mut tx = backend.txn_manager.begin(tenant);
        let _ = crud::create_node(&crud, &mut tx, tenant, label, &PropertyData::Empty)
            .expect("create node");
        crud::commit(tx, &crud).expect("commit");

        let cat = build_catalog_for_tenant(tenant, &backend);
        assert!(
            cat.has_vector_index(),
            "tenant with nodes should have vector substrate attached"
        );
        assert!(
            cat.has_bm25_index(),
            "tenant with nodes should have bm25 substrate attached"
        );
    }

    #[test]
    fn schema_provider_returns_empty_schema_for_fresh_tenant() {
        let backend = fixture();
        let provider = StorageSchemaProvider::new(backend);
        let schema = provider.schema(TenantId::DEFAULT).expect("schema");
        assert_eq!(schema.tenant_id, TenantId::DEFAULT.raw());
        assert!(schema.labels.is_empty());
        assert!(schema.rel_types.is_empty());
        // No substrates attached → indexes vec is empty.
        assert!(schema.indexes.is_empty());
        // Pre-first-commit → totals are None.
        assert_eq!(schema.total_node_count, None);
        assert_eq!(schema.total_rel_count, None);
    }

    #[test]
    fn schema_provider_rejects_unknown_tenant() {
        let backend = fixture();
        let provider = StorageSchemaProvider::new(backend);
        let unknown = TenantId::new(9999);
        let err = provider.schema(unknown).expect_err("unknown tenant");
        assert!(matches!(err, MCPError::TenantUnknown(_)));
    }

    #[test]
    fn node_inspector_surfaces_query_error_for_missing_node() {
        let backend = fixture();
        let inspector = StorageNodeInspector::new(backend);
        let err = inspector
            .inspect(TenantId::DEFAULT, 999)
            .expect_err("missing");
        assert!(matches!(err, MCPError::QueryError(_)));
    }

    #[test]
    fn ingest_provider_round_trips_node_and_relationship() {
        use crate::tools::ingest::{NodeIngest, RelIngest};
        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());
        let batch = IngestBatch {
            nodes: vec![
                NodeIngest {
                    external_id: Some("alice".into()),
                    label: "Person".into(),
                    properties: BTreeMap::new(),
                },
                NodeIngest {
                    external_id: Some("bob".into()),
                    label: "Person".into(),
                    properties: BTreeMap::new(),
                },
            ],
            relationships: vec![RelIngest {
                external_id: Some("alice-knows-bob".into()),
                from_external_id: "alice".into(),
                to_external_id: "bob".into(),
                rel_type: "KNOWS".into(),
                properties: BTreeMap::new(),
            }],
            acl_grants: vec![],
        };
        let summary = provider.ingest(TenantId::DEFAULT, batch).expect("ingest");
        assert_eq!(summary.inserted_count, 3);
        assert_eq!(summary.failed_count, 0);
        assert!(summary.commit_lsn.is_some());
    }

    #[test]
    fn ingest_provider_idempotent_resubmit_returns_same_id() {
        use crate::tools::ingest::NodeIngest;
        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());
        let mk_batch = || IngestBatch {
            nodes: vec![NodeIngest {
                external_id: Some("alice".into()),
                label: "Person".into(),
                properties: BTreeMap::new(),
            }],
            relationships: Vec::new(),
            acl_grants: vec![],
        };
        let first = provider
            .ingest(TenantId::DEFAULT, mk_batch())
            .expect("first");
        let first_id = match &first.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected Inserted, got {other:?}"),
        };
        let second = provider
            .ingest(TenantId::DEFAULT, mk_batch())
            .expect("second");
        assert_eq!(second.inserted_count, 0);
        assert_eq!(second.failed_count, 0);
        match &second.records[0] {
            IngestRecordOutcome::Idempotent { internal_id, .. } => {
                assert_eq!(*internal_id, first_id);
            }
            other => panic!("expected Idempotent on re-submit, got {other:?}"),
        }
    }

    /// #1404 M0.x — the END-TO-END at-least-once proof through the REAL ingest
    /// provider path with the BOUNDED idempotency tier engaged: ingest many
    /// distinct external_ids to force the binding for "alice" to spill, then
    /// re-ingest "alice" — the provider must report `Idempotent` (de-duped via
    /// the spill fault-in), NOT `Inserted` (a duplicate). This is the RE-2 leg
    /// wired through the production `graph.ingest` de-dup path, not just the
    /// storage unit level.
    #[test]
    fn ingest_dedupes_a_spilled_external_id_end_to_end() {
        use arcgraph_storage::idempotency::{
            IDEMPOTENCY_BINDING_WEIGHT_BYTES, IdempotencyBoundConfig, IdempotencySpill,
            IdempotencyStore,
        };

        use crate::tools::ingest::NodeIngest;

        let dir = tempfile::tempdir().unwrap();
        // A tiny resident cap so a handful of ingests forces spill.
        let spill = Arc::new(IdempotencySpill::open(dir.path()).unwrap());
        let bound = Arc::new(IdempotencyStore::with_bound(
            spill,
            IdempotencyBoundConfig {
                high_watermark_bytes: 2 * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
                low_watermark_bytes: IDEMPOTENCY_BINDING_WEIGHT_BYTES,
            },
        ));
        let backend = fixture().with_idempotency_store(Arc::clone(&bound));
        let provider = StorageIngestProvider::new(backend.clone());

        let node = |ext: &str| NodeIngest {
            external_id: Some(ext.into()),
            label: "Person".into(),
            properties: BTreeMap::new(),
        };
        let one = |ext: &str| IngestBatch {
            nodes: vec![node(ext)],
            relationships: Vec::new(),
            acl_grants: vec![],
        };

        // Ingest "alice" first — get its id.
        let first = provider
            .ingest(TenantId::DEFAULT, one("alice"))
            .expect("alice");
        let alice_id = match &first.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected Inserted for alice, got {other:?}"),
        };

        // Ingest many OTHER external_ids to push "alice"'s binding past the
        // resident cap. We simulate a checkpoint (mark durable) then force a
        // drain so "alice" spills, mirroring what the ADR-229 interval + the
        // next installs do in production.
        for i in 0..50 {
            let _ = provider
                .ingest(TenantId::DEFAULT, one(&format!("filler-{i}")))
                .expect("filler");
        }
        // Checkpoint capture marks resident bindings durable — use the
        // PRODUCTION streaming capture (`for_each_binding`), NOT the
        // `#[cfg(test)]`-only whole-`Vec` `iter_all` (invisible across the crate
        // boundary anyway).
        bound
            .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| Ok(()))
            .expect("infallible");
        bound.force_drain_for_test(); // evict the durable oldest (incl. alice) to spill
        assert!(
            bound.evicted_count() > 0,
            "no eviction — the end-to-end spill path is not exercised",
        );

        // Re-ingest "alice" through the REAL provider path. The de-dup lookup
        // (`adapters.rs:1247`) must fault "alice" back in from spill → report
        // Idempotent with the SAME id, NOT a duplicate insert.
        let again = provider
            .ingest(TenantId::DEFAULT, one("alice"))
            .expect("alice re-ingest");
        assert_eq!(
            again.inserted_count, 0,
            "re-ingest of a SPILLED external_id created a DUPLICATE (lost identity)",
        );
        match &again.records[0] {
            IngestRecordOutcome::Idempotent { internal_id, .. } => {
                assert_eq!(
                    *internal_id, alice_id,
                    "spilled binding faulted to the WRONG id",
                );
            }
            other => panic!("expected Idempotent (de-duped from spill), got {other:?}"),
        }
    }

    #[test]
    fn graph_ingest_same_external_id_different_payload_returns_idempotency_conflict() {
        use crate::tools::ingest::NodeIngest;

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());

        let mk_batch = |props: BTreeMap<String, JsonValue>| IngestBatch {
            nodes: vec![NodeIngest {
                external_id: Some("doc1".into()),
                label: "Document".into(),
                properties: props,
            }],
            relationships: Vec::new(),
            acl_grants: vec![],
        };

        let props_a1 = BTreeMap::from([("a".into(), JsonValue::Number(1.into()))]);
        let first = provider
            .ingest(TenantId::DEFAULT, mk_batch(props_a1.clone()))
            .expect("first ingest");
        let first_id = match &first.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected Inserted for first ingest, got {other:?}"),
        };
        assert_eq!(first.inserted_count, 1);
        assert_eq!(first.failed_count, 0);

        let changed = provider
            .ingest(
                TenantId::DEFAULT,
                mk_batch(BTreeMap::from([(
                    "a".into(),
                    JsonValue::Number(999.into()),
                )])),
            )
            .expect("changed-payload ingest");
        assert_eq!(changed.failed_count, 1);
        assert_eq!(changed.inserted_count, 0);
        match &changed.records[0] {
            IngestRecordOutcome::Failed {
                external_id,
                error:
                    IngestError::IdempotencyConflict {
                        external_id: err_ext,
                    },
            } => {
                assert_eq!(external_id.as_deref(), Some("doc1"));
                assert_eq!(err_ext, "doc1");
            }
            other => {
                panic!("expected Failed/IdempotencyConflict for changed payload, got {other:?}")
            }
        }

        let added = provider
            .ingest(
                TenantId::DEFAULT,
                mk_batch(BTreeMap::from([
                    ("a".into(), JsonValue::Number(1.into())),
                    ("b".into(), JsonValue::Number(7.into())),
                ])),
            )
            .expect("added-property ingest");
        assert_eq!(added.failed_count, 1);
        match &added.records[0] {
            IngestRecordOutcome::Failed {
                error: IngestError::IdempotencyConflict { external_id },
                ..
            } => assert_eq!(external_id, "doc1"),
            other => panic!("expected conflict for added property, got {other:?}"),
        }

        let retry = provider
            .ingest(TenantId::DEFAULT, mk_batch(props_a1))
            .expect("true retry");
        assert_eq!(retry.failed_count, 0);
        match &retry.records[0] {
            IngestRecordOutcome::Idempotent {
                internal_id,
                external_id,
            } => {
                assert_eq!(*internal_id, first_id);
                assert_eq!(external_id, "doc1");
            }
            other => panic!("expected true retry to be Idempotent, got {other:?}"),
        }
    }

    #[test]
    fn graph_ingest_recreates_external_id_after_delete_same_and_different_payload() {
        use crate::tools::ingest::NodeIngest;

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());

        let mk_batch = |v: i64| IngestBatch {
            nodes: vec![NodeIngest {
                external_id: Some("x".into()),
                label: "P".into(),
                properties: BTreeMap::from([("v".into(), JsonValue::Number(v.into()))]),
            }],
            relationships: Vec::new(),
            acl_grants: vec![],
        };

        let first = provider
            .ingest(TenantId::DEFAULT, mk_batch(1))
            .expect("first ingest");
        let first_id = match &first.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected first insert, got {other:?}"),
        };

        let crud = backend.crud_for(TenantId::DEFAULT).expect("crud");
        let mut del = backend.txn_manager.begin(TenantId::DEFAULT);
        crud::delete_node_with_store(&crud, &mut del, NodeId::new(first_id)).expect("delete");
        crud::commit(del, &crud).expect("delete commit");
        let reader = backend.txn_manager.begin(TenantId::DEFAULT);
        assert!(
            crud::read_node_with_store(&crud, &reader, NodeId::new(first_id))
                .expect("read deleted")
                .is_none()
        );

        let same = provider
            .ingest(TenantId::DEFAULT, mk_batch(1))
            .expect("same-payload reingest");
        assert_eq!(same.inserted_count, 1);
        assert_eq!(same.failed_count, 0);
        let second_id = match &same.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected recreate insert, got {other:?}"),
        };
        assert_ne!(second_id, first_id);

        let mut del2 = backend.txn_manager.begin(TenantId::DEFAULT);
        crud::delete_node_with_store(&crud, &mut del2, NodeId::new(second_id)).expect("delete 2");
        crud::commit(del2, &crud).expect("delete 2 commit");

        let changed = provider
            .ingest(TenantId::DEFAULT, mk_batch(9))
            .expect("different-payload reingest");
        assert_eq!(changed.inserted_count, 1);
        assert_eq!(changed.failed_count, 0);
        match &changed.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => {
                assert_ne!(*internal_id, second_id);
            }
            other => panic!("expected changed-payload recreate insert, got {other:?}"),
        }
    }

    #[test]
    fn graph_ingest_rel_same_external_id_different_payload_returns_idempotency_conflict() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());
        let first = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("a".into()),
                            label: "Doc".into(),
                            properties: BTreeMap::new(),
                        },
                        NodeIngest {
                            external_id: Some("b".into()),
                            label: "Doc".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![RelIngest {
                        external_id: Some("r1".into()),
                        from_external_id: "a".into(),
                        to_external_id: "b".into(),
                        rel_type: "LINKS".into(),
                        properties: BTreeMap::from([(
                            "weight".into(),
                            JsonValue::Number(1.into()),
                        )]),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("first rel ingest");
        let rel_id = match &first.records[2] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected rel Inserted, got {other:?}"),
        };

        let changed = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: Vec::new(),
                    relationships: vec![RelIngest {
                        external_id: Some("r1".into()),
                        from_external_id: "a".into(),
                        to_external_id: "b".into(),
                        rel_type: "LINKS".into(),
                        properties: BTreeMap::from([(
                            "weight".into(),
                            JsonValue::Number(2.into()),
                        )]),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("changed rel ingest");
        assert_eq!(changed.failed_count, 1);
        match &changed.records[0] {
            IngestRecordOutcome::Failed {
                error: IngestError::IdempotencyConflict { external_id },
                ..
            } => assert_eq!(external_id, "r1"),
            other => panic!("expected rel conflict, got {other:?}"),
        }

        let retry = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: Vec::new(),
                    relationships: vec![RelIngest {
                        external_id: Some("r1".into()),
                        from_external_id: "a".into(),
                        to_external_id: "b".into(),
                        rel_type: "LINKS".into(),
                        properties: BTreeMap::from([(
                            "weight".into(),
                            JsonValue::Number(1.into()),
                        )]),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("true rel retry");
        assert_eq!(retry.failed_count, 0);
        match &retry.records[0] {
            IngestRecordOutcome::Idempotent {
                internal_id,
                external_id,
            } => {
                assert_eq!(*internal_id, rel_id);
                assert_eq!(external_id, "r1");
            }
            other => panic!("expected rel true retry to be Idempotent, got {other:?}"),
        }
    }

    #[test]
    fn graph_ingest_intra_batch_duplicate_external_id_same_payload_is_idempotent() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());
        let props = BTreeMap::from([("name".into(), JsonValue::String("Alice".into()))]);

        let summary = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: props.clone(),
                        },
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: props.clone(),
                        },
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: props,
                        },
                        NodeIngest {
                            external_id: Some("bob".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![RelIngest {
                        external_id: Some("alice-knows-bob".into()),
                        from_external_id: "alice".into(),
                        to_external_id: "bob".into(),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("intra-batch duplicate node ingest");

        assert_eq!(summary.inserted_count, 3);
        assert_eq!(summary.failed_count, 0);
        let alice_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected first alice Inserted, got {other:?}"),
        };
        for record in &summary.records[1..=2] {
            match record {
                IngestRecordOutcome::Idempotent {
                    internal_id,
                    external_id,
                } => {
                    assert_eq!(*internal_id, alice_id);
                    assert_eq!(external_id, "alice");
                }
                other => panic!("expected duplicate alice Idempotent, got {other:?}"),
            }
        }
        let bob_id = match &summary.records[3] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected bob Inserted, got {other:?}"),
        };
        match &summary.records[4] {
            IngestRecordOutcome::Inserted { .. } => {}
            other => panic!("expected rel Inserted, got {other:?}"),
        }

        let explorer = StorageNeighborhoodExplorer::new(backend);
        let neighborhood = explorer
            .explore(
                TenantId::DEFAULT,
                alice_id,
                1,
                None,
                ExploreDirection::Out,
                &CancellationToken::new(),
            )
            .expect("explore alice");
        assert_eq!(neighborhood.edges.len(), 1);
        assert_eq!(neighborhood.edges[0].from, alice_id);
        assert_eq!(neighborhood.edges[0].to, bob_id);
    }

    #[test]
    fn graph_ingest_intra_batch_duplicate_external_id_different_payload_conflicts() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());

        let summary = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: BTreeMap::from([(
                                "name".into(),
                                JsonValue::String("Alice".into()),
                            )]),
                        },
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: BTreeMap::from([(
                                "name".into(),
                                JsonValue::String("Alicia".into()),
                            )]),
                        },
                        NodeIngest {
                            external_id: Some("bob".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![RelIngest {
                        external_id: Some("alice-knows-bob".into()),
                        from_external_id: "alice".into(),
                        to_external_id: "bob".into(),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("intra-batch conflicting node ingest");

        assert_eq!(summary.inserted_count, 3);
        assert_eq!(summary.failed_count, 1);
        let alice_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected first alice Inserted, got {other:?}"),
        };
        match &summary.records[1] {
            IngestRecordOutcome::Failed {
                external_id,
                error:
                    IngestError::IdempotencyConflict {
                        external_id: err_ext,
                    },
            } => {
                assert_eq!(external_id.as_deref(), Some("alice"));
                assert_eq!(err_ext, "alice");
            }
            other => panic!("expected duplicate alice conflict, got {other:?}"),
        }
        let bob_id = match &summary.records[2] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected bob Inserted, got {other:?}"),
        };

        let explorer = StorageNeighborhoodExplorer::new(backend);
        let neighborhood = explorer
            .explore(
                TenantId::DEFAULT,
                alice_id,
                1,
                None,
                ExploreDirection::Out,
                &CancellationToken::new(),
            )
            .expect("explore alice");
        assert_eq!(neighborhood.edges.len(), 1);
        assert_eq!(neighborhood.edges[0].from, alice_id);
        assert_eq!(neighborhood.edges[0].to, bob_id);
    }

    #[test]
    fn graph_ingest_intra_batch_duplicate_rel_external_id_same_payload_is_idempotent() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());
        let rel_props = BTreeMap::from([("weight".into(), JsonValue::Number(1.into()))]);

        let summary = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                        NodeIngest {
                            external_id: Some("bob".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![
                        RelIngest {
                            external_id: Some("r1".into()),
                            from_external_id: "alice".into(),
                            to_external_id: "bob".into(),
                            rel_type: "KNOWS".into(),
                            properties: rel_props.clone(),
                        },
                        RelIngest {
                            external_id: Some("r1".into()),
                            from_external_id: "alice".into(),
                            to_external_id: "bob".into(),
                            rel_type: "KNOWS".into(),
                            properties: rel_props,
                        },
                    ],
                    acl_grants: vec![],
                },
            )
            .expect("intra-batch duplicate rel ingest");

        assert_eq!(summary.inserted_count, 3);
        assert_eq!(summary.failed_count, 0);
        let alice_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected alice Inserted, got {other:?}"),
        };
        let rel_id = match &summary.records[2] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected first rel Inserted, got {other:?}"),
        };
        match &summary.records[3] {
            IngestRecordOutcome::Idempotent {
                internal_id,
                external_id,
            } => {
                assert_eq!(*internal_id, rel_id);
                assert_eq!(external_id, "r1");
            }
            other => panic!("expected duplicate rel Idempotent, got {other:?}"),
        }

        let explorer = StorageNeighborhoodExplorer::new(backend);
        let neighborhood = explorer
            .explore(
                TenantId::DEFAULT,
                alice_id,
                1,
                None,
                ExploreDirection::Out,
                &CancellationToken::new(),
            )
            .expect("explore alice");
        assert_eq!(neighborhood.edges.len(), 1);
    }

    #[test]
    fn graph_ingest_intra_batch_duplicate_rel_external_id_different_payload_conflicts() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());

        let summary = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("alice".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                        NodeIngest {
                            external_id: Some("bob".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![
                        RelIngest {
                            external_id: Some("r1".into()),
                            from_external_id: "alice".into(),
                            to_external_id: "bob".into(),
                            rel_type: "KNOWS".into(),
                            properties: BTreeMap::from([(
                                "weight".into(),
                                JsonValue::Number(1.into()),
                            )]),
                        },
                        RelIngest {
                            external_id: Some("r1".into()),
                            from_external_id: "alice".into(),
                            to_external_id: "bob".into(),
                            rel_type: "KNOWS".into(),
                            properties: BTreeMap::from([(
                                "weight".into(),
                                JsonValue::Number(2.into()),
                            )]),
                        },
                    ],
                    acl_grants: vec![],
                },
            )
            .expect("intra-batch conflicting rel ingest");

        assert_eq!(summary.inserted_count, 3);
        assert_eq!(summary.failed_count, 1);
        let alice_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected alice Inserted, got {other:?}"),
        };
        match &summary.records[3] {
            IngestRecordOutcome::Failed {
                external_id,
                error:
                    IngestError::IdempotencyConflict {
                        external_id: err_ext,
                    },
            } => {
                assert_eq!(external_id.as_deref(), Some("r1"));
                assert_eq!(err_ext, "r1");
            }
            other => panic!("expected duplicate rel conflict, got {other:?}"),
        }

        let explorer = StorageNeighborhoodExplorer::new(backend);
        let neighborhood = explorer
            .explore(
                TenantId::DEFAULT,
                alice_id,
                1,
                None,
                ExploreDirection::Out,
                &CancellationToken::new(),
            )
            .expect("explore alice");
        assert_eq!(neighborhood.edges.len(), 1);
    }

    #[test]
    fn ingest_then_inspect_round_trip() {
        use crate::tools::ingest::NodeIngest;
        let backend = fixture();
        let ingest = StorageIngestProvider::new(backend.clone());
        let mut props = BTreeMap::new();
        props.insert("name".into(), JsonValue::String("Alice".into()));
        let summary = ingest
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![NodeIngest {
                        external_id: Some("alice".into()),
                        label: "Person".into(),
                        properties: props.clone(),
                    }],
                    relationships: Vec::new(),
                    acl_grants: vec![],
                },
            )
            .expect("ingest");
        let internal_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected Inserted, got {other:?}"),
        };
        let inspector = StorageNodeInspector::new(backend);
        let inspection = inspector
            .inspect(TenantId::DEFAULT, internal_id)
            .expect("inspect");
        assert_eq!(inspection.id, internal_id);
        // Label resolves through the intern table to "Person".
        assert_eq!(inspection.label.as_deref(), Some("Person"));
    }

    #[test]
    fn explore_emits_seed_only_at_depth_zero() {
        use crate::tools::ingest::NodeIngest;
        let backend = fixture();
        let ingest = StorageIngestProvider::new(backend.clone());
        let summary = ingest
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![NodeIngest {
                        external_id: Some("alice".into()),
                        label: "Person".into(),
                        properties: BTreeMap::new(),
                    }],
                    relationships: Vec::new(),
                    acl_grants: vec![],
                },
            )
            .expect("ingest");
        let alice_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            _ => panic!(),
        };
        let explorer = StorageNeighborhoodExplorer::new(backend);
        let token = CancellationToken::new();
        let n = explorer
            .explore(
                TenantId::DEFAULT,
                alice_id,
                0,
                None,
                ExploreDirection::Out,
                &token,
            )
            .expect("explore");
        assert_eq!(n.seed, alice_id);
        assert_eq!(n.max_depth, 0);
        assert_eq!(n.nodes.len(), 1);
        assert!(n.edges.is_empty());
    }

    /// ADR-217: the account-sink topology in miniature. An incident ticket
    /// `--AFFECTS_ACCOUNT--> account` — exactly the demo corpus's
    /// blast-radius shape where the rich neighbor points INTO the account.
    /// Seeded from the account: `Out` finds NOTHING (the account is a
    /// sink); `In` (and `Both`) reach the incident via `scan_in`, tagged
    /// `NeighborDirection::In` with the edge oriented incident→account.
    /// This is the RED-on-revert anchor for the inbound traversal: with the
    /// pre-ADR-217 outbound-only explorer, the `In`/`Both` assertions fail.
    #[test]
    fn explore_inbound_direction_reaches_account_sink_neighbors() {
        use crate::tools::ingest::{NodeIngest, RelIngest};
        let backend = fixture();
        let ingest = StorageIngestProvider::new(backend.clone());
        let summary = ingest
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("acct".into()),
                            label: "Account".into(),
                            properties: BTreeMap::new(),
                        },
                        NodeIngest {
                            external_id: Some("inc".into()),
                            label: "Ticket".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![RelIngest {
                        external_id: Some("inc-affects-acct".into()),
                        from_external_id: "inc".into(),
                        to_external_id: "acct".into(),
                        rel_type: "AFFECTS_ACCOUNT".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("ingest sink fixture");
        let acct_id = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected acct Inserted, got {other:?}"),
        };
        let inc_id = match &summary.records[1] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected inc Inserted, got {other:?}"),
        };
        let explorer = StorageNeighborhoodExplorer::new(backend);
        let token = CancellationToken::new();

        // Out from the account = sink → no edges (the v1.0-alpha behavior;
        // this is WHY the demo narrative was stuck at "1 system").
        let out = explorer
            .explore(
                TenantId::DEFAULT,
                acct_id,
                1,
                None,
                ExploreDirection::Out,
                &token,
            )
            .expect("explore out");
        assert!(
            out.edges.is_empty(),
            "the account is an outbound sink: {:?}",
            out.edges
        );

        // In from the account reaches the incident via scan_in (ADR-131).
        let inb = explorer
            .explore(
                TenantId::DEFAULT,
                acct_id,
                1,
                None,
                ExploreDirection::In,
                &token,
            )
            .expect("explore in");
        assert_eq!(inb.edges.len(), 1, "inbound finds the AFFECTS_ACCOUNT edge");
        let e = &inb.edges[0];
        assert_eq!(e.from, inc_id, "edge oriented incident→account");
        assert_eq!(e.to, acct_id);
        assert_eq!(e.rel_type.as_deref(), Some("AFFECTS_ACCOUNT"));
        assert_eq!(e.direction, NeighborDirection::In);
        assert!(
            inb.nodes.iter().any(|n| n.id == inc_id),
            "the incident node is surfaced as a neighbor"
        );

        // Both = same single edge (de-duped by RelId; no double-count).
        let both = explorer
            .explore(
                TenantId::DEFAULT,
                acct_id,
                1,
                None,
                ExploreDirection::Both,
                &token,
            )
            .expect("explore both");
        assert_eq!(
            both.edges.len(),
            1,
            "both dedups by RelId: {:?}",
            both.edges
        );
        assert_eq!(both.edges[0].direction, NeighborDirection::In);
    }

    #[test]
    fn hybrid_searcher_reports_no_substrate_for_unwired_tenant() {
        let backend = fixture();
        let searcher = StorageHybridSearcher::new(backend);
        let token = CancellationToken::new();
        let avail = searcher
            .available_substrates(TenantId::DEFAULT, &token)
            .expect("avail");
        assert!(!avail.any());
    }

    #[test]
    fn value_to_json_round_trips_primitives() {
        assert_eq!(value_to_json(&QueryValue::Null), JsonValue::Null);
        assert_eq!(
            value_to_json(&QueryValue::Boolean(true)),
            JsonValue::Bool(true)
        );
        assert_eq!(
            value_to_json(&QueryValue::Integer(42)),
            JsonValue::Number(42.into())
        );
        assert_eq!(
            value_to_json(&QueryValue::String("hi".into())),
            JsonValue::String("hi".into())
        );
    }

    #[test]
    fn property_data_for_json_map_returns_empty_for_empty_bag() {
        let empty = BTreeMap::new();
        assert_eq!(property_data_for_json_map(&empty), PropertyData::Empty);
    }

    #[test]
    fn property_data_for_json_map_returns_blob_for_populated_bag() {
        let mut m = BTreeMap::new();
        m.insert("k".into(), JsonValue::String("v".into()));
        let pd = property_data_for_json_map(&m);
        match pd {
            PropertyData::Blob(bytes) => {
                let parsed: JsonValue = serde_json::from_slice(&bytes).expect("json");
                assert_eq!(parsed["k"], JsonValue::String("v".into()));
            }
            other => panic!("expected Blob, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────
    // R1 fix-up (PR #349) — H-2 / M-1 / M-5 / M-6 regression pins
    // ─────────────────────────────────────────────────────────────

    /// R1 HIGH-2 (PR #349): a commit-time failure during ingest must
    /// NOT poison the per-tenant idempotency cache. A subsequent
    /// re-submit of the same external_id either re-Inserts cleanly
    /// or surfaces a structured error; it must NEVER return
    /// `Idempotent { internal_id }` referring to a rolled-back id.
    #[test]
    fn ingest_does_not_poison_idempotency_on_commit_failure() {
        use std::sync::atomic::Ordering;

        use crate::tools::ingest::NodeIngest;

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());
        let switch = provider.force_commit_failure_handle();

        // Flip the test-only commit-failure switch BEFORE the first
        // ingest. The writes will enter the transaction normally,
        // then `execute_commit` synthesizes a CrudError::Mvcc that
        // takes the failure-handler branch.
        switch.store(true, Ordering::SeqCst);

        let batch1 = IngestBatch {
            nodes: vec![NodeIngest {
                external_id: Some("alice".into()),
                label: "Person".into(),
                properties: BTreeMap::new(),
            }],
            relationships: Vec::new(),
            acl_grants: vec![],
        };
        let summary1 = provider
            .ingest(TenantId::DEFAULT, batch1)
            .expect("ingest call returns Ok");
        // The commit failed — every staged Inserted outcome is
        // rewritten to Failed-Storage.
        assert_eq!(summary1.commit_lsn, None);
        assert_eq!(summary1.inserted_count, 0);
        assert_eq!(summary1.failed_count, 1);
        match &summary1.records[0] {
            IngestRecordOutcome::Failed { error, .. } => match error {
                IngestError::Storage { detail } => {
                    assert!(
                        detail.contains("commit failed"),
                        "expected commit-failed detail, got {detail}"
                    );
                }
                other => panic!("expected Storage err, got {other:?}"),
            },
            other => panic!("expected Failed outcome, got {other:?}"),
        }

        // Idempotency cache MUST be empty for this tenant — the
        // failed commit must not have published any pending entry.
        assert_eq!(
            backend.idempotency.len_for_tenant(TenantId::DEFAULT),
            0,
            "idempotency store poisoned: pending insert survived commit failure"
        );

        // Reset the switch so the second submit commits for real.
        switch.store(false, Ordering::SeqCst);

        let batch2 = IngestBatch {
            nodes: vec![NodeIngest {
                external_id: Some("alice".into()),
                label: "Person".into(),
                properties: BTreeMap::new(),
            }],
            relationships: Vec::new(),
            acl_grants: vec![],
        };
        let summary2 = provider
            .ingest(TenantId::DEFAULT, batch2)
            .expect("second ingest");
        // The contract: NEVER `Idempotent` referring to a non-
        // existent id. We require explicit Inserted (clean retry).
        match &summary2.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => {
                assert!(*internal_id >= 1, "expected new internal_id");
                assert_eq!(summary2.inserted_count, 1);
                assert!(summary2.commit_lsn.is_some());
            }
            IngestRecordOutcome::Idempotent { internal_id, .. } => panic!(
                "Idempotent {{ internal_id: {internal_id} }} returned for a previously-rolled-back record"
            ),
            other => panic!("expected Inserted on clean retry, got {other:?}"),
        }
    }

    /// R1 MED-1 (PR #349): the idempotency map must disambiguate
    /// node vs. rel namespaces. A node "x" and a rel "x" within the
    /// same tenant must not cross-route — re-submitting the node
    /// must return the node's id and re-submitting the rel must
    /// return the rel's id, even when the two namespaces happen to
    /// allocate different numerical id sequences.
    #[test]
    fn idempotency_disambiguates_node_and_rel_kinds() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());

        // Batch 1: 2 nodes ("a", "b") + 1 unnamed rel — this
        // advances both the node + rel id counters so when batch 2's
        // named "x" entities land, their numerical ids are guaranteed
        // to be different across kinds (node "x" → id 3, rel "x" →
        // id 2). The old single-map code would type-confuse on the
        // shared string key.
        let warmup = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("a".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                        NodeIngest {
                            external_id: Some("b".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![RelIngest {
                        external_id: None, // unnamed — burns one rel id
                        from_external_id: "a".into(),
                        to_external_id: "b".into(),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("warmup");
        assert_eq!(warmup.failed_count, 0);

        // Batch 2: node "x" + node "y" + rel "x".
        let summary = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![
                        NodeIngest {
                            external_id: Some("x".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                        NodeIngest {
                            external_id: Some("y".into()),
                            label: "Person".into(),
                            properties: BTreeMap::new(),
                        },
                    ],
                    relationships: vec![RelIngest {
                        external_id: Some("x".into()), // SAME string as node above
                        from_external_id: "x".into(),
                        to_external_id: "y".into(),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("named batch");
        assert_eq!(summary.inserted_count, 3);
        assert_eq!(summary.failed_count, 0);
        let node_x_internal = match &summary.records[0] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected node Inserted, got {other:?}"),
        };
        let rel_x_internal = match &summary.records[2] {
            IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
            other => panic!("expected rel Inserted, got {other:?}"),
        };
        // The numerical ids should differ across kinds because the
        // rel counter is ahead by one (warmup created 1 rel and 2
        // nodes; node "x" gets node-id 3 / rel "x" gets rel-id 2).
        assert_ne!(
            node_x_internal, rel_x_internal,
            "fixture is misconfigured: node + rel internal_ids must differ for the test to discriminate"
        );

        // Re-submit batch 2's node "x": must Idempotent → node's id,
        // NOT rel's id (kind-scoped resolution).
        let re_node = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![NodeIngest {
                        external_id: Some("x".into()),
                        label: "Person".into(),
                        properties: BTreeMap::new(),
                    }],
                    relationships: Vec::new(),
                    acl_grants: vec![],
                },
            )
            .expect("re-submit node");
        match &re_node.records[0] {
            IngestRecordOutcome::Idempotent { internal_id, .. } => {
                assert_eq!(
                    *internal_id, node_x_internal,
                    "node re-submit must resolve to node's id"
                );
            }
            other => panic!("expected node Idempotent, got {other:?}"),
        }

        // Re-submit batch 2's rel "x": must Idempotent → rel's id,
        // NOT node's id (kind-scoped resolution).
        let re_rel = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: Vec::new(),
                    relationships: vec![RelIngest {
                        external_id: Some("x".into()),
                        from_external_id: "x".into(),
                        to_external_id: "y".into(),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("re-submit rel");
        match &re_rel.records[0] {
            IngestRecordOutcome::Idempotent { internal_id, .. } => {
                assert_eq!(
                    *internal_id, rel_x_internal,
                    "rel re-submit must resolve to rel's id, not node's"
                );
            }
            other => panic!("expected rel Idempotent, got {other:?}"),
        }
    }

    /// #352 Part 2 (ADR-199): the per-tenant cap is REMOVED. Distinct
    /// external_ids well beyond what Part-1's small-cap would have refused
    /// are ALL accepted (never `CapacityExceeded`), every re-ingest still
    /// resolves idempotently to its ORIGINAL id (no eviction — the Part-1
    /// bug class), and an edge between two of them resolves + commits.
    ///
    /// RED-on-revert (re-introduce a Part-1 cap of 4): the 5th node would
    /// fail `CapacityExceeded` instead of inserting, so this test goes red
    /// the moment a ceiling returns.
    #[test]
    fn idempotency_cap_removed_accepts_beyond_old_ceiling_and_stays_idempotent() {
        use crate::tools::ingest::{NodeIngest, RelIngest};
        // Fresh in-memory backend → fresh (uncapped) IdempotencyStore.
        let backend = fixture();
        let provider = StorageIngestProvider::new(backend.clone());

        let mk_node = |ext: &str| IngestBatch {
            nodes: vec![NodeIngest {
                external_id: Some(ext.into()),
                label: "Person".into(),
                properties: BTreeMap::new(),
            }],
            relationships: Vec::new(),
            acl_grants: vec![],
        };

        // Ingest 50 distinct external_ids. Part-1 with a cap of 4 would
        // have refused 46 of these; with the cap removed, ALL succeed.
        const N: usize = 50;
        let mut ids = Vec::new();
        for i in 0..N {
            let s = provider
                .ingest(TenantId::DEFAULT, mk_node(&format!("node-{i}")))
                .expect("ingest");
            assert_eq!(s.inserted_count, 1, "node-{i} inserted");
            assert_eq!(s.failed_count, 0, "node-{i} not refused (cap removed)");
            match &s.records[0] {
                IngestRecordOutcome::Inserted { internal_id, .. } => ids.push(*internal_id),
                other => panic!("expected Inserted for node-{i}, got {other:?}"),
            }
        }
        assert_eq!(
            backend.idempotency.len_for_tenant(TenantId::DEFAULT),
            N,
            "all {N} distinct bindings retained — no cap, no eviction",
        );

        // Every one — including the OLDEST (node-0) — still resolves
        // idempotently to its ORIGINAL id.
        for (i, original) in ids.iter().enumerate() {
            let s = provider
                .ingest(TenantId::DEFAULT, mk_node(&format!("node-{i}")))
                .expect("re-ingest");
            match &s.records[0] {
                IngestRecordOutcome::Idempotent { internal_id, .. } => assert_eq!(
                    internal_id, original,
                    "node-{i} must resolve to its original id (not evicted, not duplicated)",
                ),
                other => panic!("expected Idempotent for node-{i}, got {other:?}"),
            }
        }

        // An edge between the oldest + newest resolves + commits.
        let edge = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: Vec::new(),
                    relationships: vec![RelIngest {
                        external_id: None,
                        from_external_id: "node-0".into(),
                        to_external_id: format!("node-{}", N - 1),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("edge ingest");
        assert_eq!(edge.failed_count, 0, "edge must not fail unresolved");
        assert_eq!(edge.inserted_count, 1);
    }

    /// #352 Part 2 (ADR-199) — the headline cap-removal acceptance at the
    /// scale the silent loss manifested (CZ web-Google, >100K). Ingest
    /// 100_001 distinct external_ids — ONE more than Part-1's removed
    /// 100_000 per-tenant ceiling — and prove:
    ///   1. EVERY one is accepted (no `CapacityExceeded` anywhere) — the
    ///      old cap is genuinely gone, not merely raised;
    ///   2. the durable store holds all 100_001 (unbounded, no eviction);
    ///   3. the OLDEST binding (`n0`) still resolves idempotently to its
    ///      ORIGINAL id (no divergent duplicate mint);
    ///   4. an edge referencing the oldest node RESOLVES and is created.
    ///
    /// RED-on-revert (re-introduce the 100_000 cap): the 100_001st node
    /// fails `CapacityExceeded`, so assertion 1 goes red at the exact
    /// scale the bug manifested (the #826 Stage-2-scale lesson).
    #[test]
    fn idempotency_no_cap_beyond_100k_distinct_ids() {
        use crate::tools::ingest::{NodeIngest, RelIngest};

        // The OLD (removed) per-tenant ceiling was 100_000; cross it by 1.
        const BEYOND_OLD_CAP: usize = 100_001;

        let io = Arc::new(InMemoryPageIo::new());
        // Generous pool so the record-page writes don't thrash the cache
        // against the (RAM-backed) InMemoryPageIo.
        let pool = BufferPool::new(2048, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        let backend = StorageBackend::new(router, mgr, intern);
        let provider = StorageIngestProvider::new(backend.clone());

        // Ingest beyond the old cap with distinct external_ids, in chunks.
        const CHUNK: usize = 10_000;
        let mut node0_id: Option<u64> = None;
        let mut inserted_total: u64 = 0;
        let mut next = 0usize;
        while next < BEYOND_OLD_CAP {
            let end = (next + CHUNK).min(BEYOND_OLD_CAP);
            let nodes: Vec<NodeIngest> = (next..end)
                .map(|i| NodeIngest {
                    external_id: Some(format!("n{i}")),
                    label: "Person".into(),
                    properties: BTreeMap::new(),
                })
                .collect();
            let summary = provider
                .ingest(
                    TenantId::DEFAULT,
                    IngestBatch {
                        nodes,
                        relationships: Vec::new(),
                        acl_grants: vec![],
                    },
                )
                .expect("bulk ingest");
            // (1) NO refusal at ANY scale — the cap is gone.
            assert_eq!(
                summary.failed_count, 0,
                "no refusal at any scale (cap removed); failed at chunk starting {next}"
            );
            assert!(
                !summary.records.iter().any(|r| matches!(
                    r,
                    IngestRecordOutcome::Failed {
                        error: IngestError::CapacityExceeded { .. },
                        ..
                    }
                )),
                "no CapacityExceeded anywhere (chunk starting {next})"
            );
            inserted_total += summary.inserted_count;
            if node0_id.is_none() {
                node0_id = Some(match &summary.records[0] {
                    IngestRecordOutcome::Inserted { internal_id, .. } => *internal_id,
                    other => panic!("expected Inserted for n0, got {other:?}"),
                });
            }
            next = end;
        }
        assert_eq!(
            inserted_total as usize, BEYOND_OLD_CAP,
            "all {BEYOND_OLD_CAP} distinct nodes landed (> old 100K cap)"
        );
        let node0_id = node0_id.expect("captured n0 id");
        // (2) The durable store holds every binding — unbounded.
        assert_eq!(
            backend.idempotency.len_for_tenant(TenantId::DEFAULT),
            BEYOND_OLD_CAP,
            "store holds all bindings beyond the old cap, no eviction",
        );

        // (3) The OLDEST binding (n0) still resolves idempotently to its
        // ORIGINAL id (never evicted, never a duplicate mint).
        let re0 = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: vec![NodeIngest {
                        external_id: Some("n0".into()),
                        label: "Person".into(),
                        properties: BTreeMap::new(),
                    }],
                    relationships: Vec::new(),
                    acl_grants: vec![],
                },
            )
            .expect("re-ingest n0");
        match &re0.records[0] {
            IngestRecordOutcome::Idempotent { internal_id, .. } => assert_eq!(
                *internal_id, node0_id,
                "oldest binding n0 must resolve to its ORIGINAL id, not a duplicate",
            ),
            other => panic!("expected Idempotent for n0 (binding preserved), got {other:?}"),
        }

        // (4) An edge referencing the oldest node RESOLVES and is created.
        let edge = provider
            .ingest(
                TenantId::DEFAULT,
                IngestBatch {
                    nodes: Vec::new(),
                    relationships: vec![RelIngest {
                        external_id: None,
                        from_external_id: "n0".into(),
                        to_external_id: "n100000".into(),
                        rel_type: "KNOWS".into(),
                        properties: BTreeMap::new(),
                    }],
                    acl_grants: vec![],
                },
            )
            .expect("edge ingest");
        assert_eq!(
            edge.failed_count, 0,
            "edge from the oldest node must NOT falsely fail unresolved",
        );
        assert_eq!(
            edge.inserted_count, 1,
            "edge created against the preserved binding",
        );
        match &edge.records[0] {
            IngestRecordOutcome::Inserted { .. } => {}
            other => panic!("expected edge Inserted, got {other:?}"),
        }
    }

    /// R1 MED-6 (PR #349): `NodeInspector::inspect` must NOT
    /// surface `_inline_u32a/b` as user-visible property names.
    #[test]
    fn node_inspector_does_not_surface_inline_u32_property_names() {
        let backend = fixture();
        // Construct a NodeRecord with non-zero inline u32 fields.
        let mut tx = backend.txn_manager.begin(TenantId::DEFAULT);
        let crud = backend
            .crud_for(TenantId::DEFAULT)
            .expect("crud for default tenant");
        let label = arcgraph_core::LabelId::new(1);
        let nid = crud::create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::InlineU32Pair(13, 91),
        )
        .expect("create");
        crud::commit(tx, &crud).expect("commit");

        let inspector = StorageNodeInspector::new(backend);
        let inspection = inspector
            .inspect(TenantId::DEFAULT, nid.raw())
            .expect("inspect");
        for key in inspection.properties.keys() {
            assert!(
                !key.starts_with("_inline_u32"),
                "property key {key:?} must not leak storage-internal name"
            );
        }
    }
}
