//! [`CrudExecutorSubstrate`] — production-side
//! [`arcgraph_query::ExecutorSubstrate`] impl backed by
//! [`arcgraph_storage::router::TenantHandle`] +
//! [`arcgraph_storage::transaction::TxnManager`].
//!
//! Bridges the query layer's substrate-access surface (scan_nodes /
//! expand / vector_search / bm25_search / community_members) to the
//! storage layer's CRUD scan + TEL adjacency.
//!
//! - **W17α**: graph reads (scan_nodes / expand outbound) lit.
//! - **W26-β-2 / ADR-131**: inbound + undirected `expand` lit
//!   (reverse-adjacency index).
//! - **W26-β-3 / ADR-132 (this slice)**: vector_search + bm25_search
//!   BODY wire-through through the [`SubstrateSearchProvider`]
//!   trait. When the router has a vector / BM25 handle attached AND
//!   a provider is wired via
//!   [`CrudExecutorSubstrate::with_search_provider`], the search
//!   methods return real
//!   [`arcgraph_query::executor::substrate::RankedHit`] rows. When
//!   the router handle is unattached (the long-standing v1.0-α
//!   posture), the methods continue to surface
//!   `SubstrateAccessError::IndexUnavailable` per the W23-M4-08-FINALIZE
//!   ADR-087 §"What this locks in" contract. The community_members
//!   body remains forward-pinned to M4-62b per the
//!   [`ExecutorSubstrate::community_members`] trait rustdoc.
//!
//! # Snapshot-LSN discipline
//!
//! A finite `read_lsn` is exact. The public CRUD engine can open a
//! transaction only at its current visible snapshot; after opening
//! that transaction, [`CrudExecutorSubstrate`] compares the actual
//! snapshot with the requested LSN. A mismatch returns
//! [`SubstrateAccessError::SnapshotUnavailable`] carrying both values.
//! It never ratchets forward and returns success.
//!
//! [`Lsn::MAX`] is the explicit read-latest sentinel. Each call
//! resolves it once to the transaction's actual snapshot. Graph reads
//! use that transaction directly; vector and BM25 providers receive
//! the resolved finite LSN, and post-hit node hydration reuses the same
//! transaction. Thus every successful call has one effective snapshot.
//! A caller that needs one point across multiple calls must supply the
//! same finite LSN; if that point is no longer the current snapshot,
//! the non-temporal CRUD surfaces reject it rather than silently
//! diverging from MVCC-native indexes.
//!
//! # Memory + tenant scoping
//!
//! Per ADR-037 §D-1, every call is scoped to the `tenant` argument;
//! the substrate's TenantHandle is acquired from the multi-tenant
//! router so cross-tenant leakage is structurally impossible. The
//! `CrudExecutorSubstrate` itself is shareable (Arc-friendly) so a
//! single instance can serve many concurrent executor sessions.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use arcgraph_core::{
    LabelId, Lsn, NodeId, NodeRecord, PartitionId, RelId, RelRecord, TenantId, TypeId,
};
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::context::HeldTxnAccess;
use arcgraph_query::executor::substrate::{
    BoundEdge, BoundEdgeCursor, BoundNode, ExecutorSubstrate, HeldTxnHandle, MergeGuard,
    PropertyIndexRegistration, RankedHit, RemoveNodeMutation, RemoveRelMutation, SetNodeMutation,
    SetRelMutation, SubstrateAccessError, VectorIndexCatalogEntry, VectorIndexRegistration,
};
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::logical_plan::Direction;
use arcgraph_storage::crud::{self, CrudStore, ScanInCursor, ScanOutCursor};
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::{OwnedTxn, Transaction, TxnManager};
use arcgraph_storage::{InternTable, RoutingError};
use parking_lot::{ArcMutexGuard, Mutex, RawMutex};

/// W26-β-2 / ADR-131 — internal in-flight edge record used by
/// `CrudExecutorSubstrate::expand` to dedup outbound + inbound walks
/// before materializing `BoundEdge`s.
///
/// Carries the canonical (rel_src, rel_dst) pair so the resulting
/// `RelView` reflects the actual edge orientation regardless of
/// which direction the substrate walked it from, plus the
/// "traversal far-end" node id used to populate `BoundEdge.dst`
/// per the substrate contract (`BoundEdge.dst` = the FAR end of
/// traversal — opposite of `LogicalExpand::from`).
#[derive(Debug, Clone, Copy)]
struct EdgePending {
    rel_id: u64,
    src: NodeId,
    dst: NodeId,
    far_end: NodeId,
}

enum CrudExpandPhase {
    Out(ScanOutCursor),
    In(ScanInCursor),
    Undirected {
        out: ScanOutCursor,
        in_: ScanInCursor,
        draining_out: bool,
    },
}

/// Streaming production one-hop expand cursor.
///
/// The owned transaction registers in the MVCC active set for the
/// cursor lifetime. It holds no locks and does not block writers; the
/// system cost is only that `oldest_active_snapshot()` cannot advance
/// past this cursor's snapshot until `OwnedTxn::Drop` deregisters it.
struct CrudExpandCursor {
    crud: Arc<CrudStore>,
    intern: Arc<InternTable>,
    owned: OwnedTxn,
    tenant: TenantId,
    rel_type: Option<TypeId>,
    phase: CrudExpandPhase,
    from: NodeId,
}

impl Iterator for CrudExpandCursor {
    type Item = Result<BoundEdge, SubstrateAccessError>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let pending = match &mut self.phase {
                CrudExpandPhase::Out(cursor) => {
                    let entry = cursor.next_entry(&self.crud, self.owned.txn())?;
                    let dst_id = NodeId::new(entry.dst_id);
                    EdgePending {
                        rel_id: entry.rel_id,
                        src: self.from,
                        dst: dst_id,
                        far_end: dst_id,
                    }
                }
                CrudExpandPhase::In(cursor) => {
                    let entry = cursor.next_entry(&self.crud, self.owned.txn())?;
                    let src_id = NodeId::new(entry.dst_id);
                    EdgePending {
                        rel_id: entry.rel_id,
                        src: src_id,
                        dst: self.from,
                        far_end: src_id,
                    }
                }
                CrudExpandPhase::Undirected {
                    out,
                    in_,
                    draining_out,
                } => {
                    if *draining_out {
                        if let Some(entry) = out.next_entry(&self.crud, self.owned.txn()) {
                            let dst_id = NodeId::new(entry.dst_id);
                            EdgePending {
                                rel_id: entry.rel_id,
                                src: self.from,
                                dst: dst_id,
                                far_end: dst_id,
                            }
                        } else {
                            *draining_out = false;
                            continue;
                        }
                    } else {
                        loop {
                            let entry = in_.next_entry(&self.crud, self.owned.txn())?;
                            let src_id = NodeId::new(entry.dst_id);
                            // A relationship can occur in both the forward
                            // and reverse adjacency of `from` iff it is a
                            // self-loop. Outbound drains first, so skipping
                            // precisely that inbound shape deduplicates with
                            // O(1) state instead of retaining every rel id.
                            if src_id != self.from {
                                break EdgePending {
                                    rel_id: entry.rel_id,
                                    src: src_id,
                                    dst: self.from,
                                    far_end: src_id,
                                };
                            }
                        }
                    }
                }
            };
            return match materialize_bound_edge(
                &self.crud,
                &self.intern,
                self.tenant,
                self.owned.txn(),
                pending,
                self.rel_type,
            ) {
                Ok(Some(edge)) => Some(Ok(edge)),
                Ok(None) => continue,
                Err(err) => Some(Err(err)),
            };
        }
    }
}

/// Streaming expand over the raw relationship-id space visible to a Bolt
/// explicit transaction.
///
/// TEL adjacency is populated only at commit, so staged relationships cannot
/// be served from the forward/reverse chains. Scanning the transaction's MVCC
/// view by raw id is the existing visibility contract. Capturing the allocator
/// high-water once and advancing a single integer preserves the prior
/// ascending-rel order with O(1) cursor state instead of collecting all
/// matching edges.
struct HeldCrudExpandCursor {
    crud: Arc<CrudStore>,
    intern: Arc<InternTable>,
    held: HeldTxnAccess,
    tenant: TenantId,
    from: NodeId,
    rel_type: Option<TypeId>,
    direction: Direction,
    next_raw: u64,
    high_water: u64,
    finished: bool,
}

impl HeldCrudExpandCursor {
    fn take_next_raw(&mut self) -> Option<u64> {
        if self.finished || self.high_water == 0 {
            self.finished = true;
            return None;
        }
        let raw = self.next_raw;
        if raw >= self.high_water {
            self.finished = true;
        } else {
            self.next_raw += 1;
        }
        Some(raw)
    }
}

impl Iterator for HeldCrudExpandCursor {
    type Item = Result<BoundEdge, SubstrateAccessError>;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(raw) = self.take_next_raw() {
            let result = self.held.with_mut(|handle| {
                let owned = handle
                    .as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })?
                    .owned_mut()?;
                let tx = owned.txn();
                let rel_id = RelId::new(raw);
                let rec = crud::read_rel(tx, rel_id).map_err(|e| {
                    SubstrateAccessError::Io(format!("expand: read_rel({raw}) failed: {e}"))
                })?;
                let Some(rec) = rec else {
                    return Ok(None);
                };
                if !rel_matches_type(&rec, self.rel_type) {
                    return Ok(None);
                }
                let src = NodeId::new(rec.src_id);
                let dst = NodeId::new(rec.dst_id);
                let far_end = match self.direction {
                    Direction::LeftToRight if src == self.from => dst,
                    Direction::RightToLeft if dst == self.from => src,
                    Direction::Undirected if src == self.from => dst,
                    Direction::Undirected if dst == self.from => src,
                    _ => return Ok(None),
                };
                materialize_bound_edge(
                    &self.crud,
                    &self.intern,
                    self.tenant,
                    tx,
                    EdgePending {
                        rel_id: raw,
                        src,
                        dst,
                        far_end,
                    },
                    self.rel_type,
                )
            });

            match result {
                Some(Ok(Some(edge))) => return Some(Ok(edge)),
                Some(Ok(None)) => continue,
                Some(Err(err)) => {
                    self.finished = true;
                    return Some(Err(err));
                }
                None => {
                    self.finished = true;
                    return Some(Err(SubstrateAccessError::Io(
                        "held transaction was reclaimed while expand cursor was active".into(),
                    )));
                }
            }
        }
        None
    }
}

fn rel_matches_type(rec: &RelRecord, rel_type: Option<TypeId>) -> bool {
    rel_type.is_none_or(|ty| rec.type_id == ty.raw())
}

fn resolve_label_name(intern: &InternTable, tenant: TenantId, label_id: u32) -> Option<String> {
    if label_id == 0 {
        return None;
    }
    intern
        .resolve(tenant, arcgraph_core::ids::StringId::new(label_id))
        .map(|arc| arc.to_string())
}

fn resolve_rel_type_name(intern: &InternTable, tenant: TenantId, type_id: u32) -> Option<String> {
    if type_id == 0 {
        return None;
    }
    intern
        .resolve(tenant, arcgraph_core::ids::StringId::new(type_id))
        .map(|arc| arc.to_string())
}

/// **#1366 (Phase 2) — the property-index verify recheck equality.**
/// Whether a hydrated node's property value `stored` matches the
/// looked-up `looked_up` value under the engine's `=` coercion, so the
/// index path returns EXACTLY the rows the full-scan `Filter(n.prop =
/// v)` would. RC-supported lookup values are `String` / `Integer` /
/// `Boolean` (Float is dropped from the RC index at the planner). The
/// only coercion the engine `=` applies across those + their neighbours
/// is numeric (`Integer` ⇄ `Float`): `5 = 5.0` is `true`. Every other
/// pair is a same-variant equality. NULL never equality-matches (the
/// caller already excludes a NULL lookup value; a stored NULL is absent
/// from the bag).
fn index_value_eq_coerced(
    stored: &arcgraph_query::executor::value::Value,
    looked_up: &arcgraph_query::executor::value::Value,
) -> bool {
    use arcgraph_query::executor::value::Value as V;
    match (stored, looked_up) {
        // Numeric coercion, mirroring engine `=` (NN-4): an Integer
        // lookup matches a stored Float of equal magnitude and vice
        // versa. `as f64` is exact for the i64 range the engine admits
        // for equality against a float literal.
        (V::Integer(a), V::Float(b)) | (V::Float(b), V::Integer(a)) => (*a as f64) == *b,
        // Everything else is same-variant equality (String / Integer /
        // Boolean / Float-Float). `PartialEq` gives the exact engine `=`
        // for these.
        (a, b) => a == b,
    }
}

fn materialize_bound_edge(
    crud: &CrudStore,
    intern: &InternTable,
    tenant: TenantId,
    tx: &Transaction<'_>,
    ep: EdgePending,
    rel_type_filter: Option<TypeId>,
) -> Result<Option<BoundEdge>, SubstrateAccessError> {
    let blobs = crud.blob_store();
    let (far_label, far_label_name, far_props) =
        match crud::read_node_with_store(crud, tx, ep.far_end) {
            Ok(Some(r)) => {
                let label = if r.label_id == 0 {
                    None
                } else {
                    Some(LabelId::new(r.label_id))
                };
                let label_name = resolve_label_name(intern, tenant, r.label_id);
                let props = crate::storage::property_payload::record_property_bag_checked(
                    &r, blobs, intern, tenant,
                )?;
                (label, label_name, props)
            }
            Ok(None) => return Ok(None),
            Err(e) => {
                return Err(SubstrateAccessError::Io(format!(
                    "expand: read_node(far_end={}) failed: {e}",
                    ep.far_end.raw()
                )));
            }
        };
    let rel_id_typed = arcgraph_core::RelId::new(ep.rel_id);
    let (rel_type_name, rel_props) = match crud::read_rel_with_store(crud, tx, rel_id_typed) {
        Ok(Some(r)) => {
            let name = resolve_rel_type_name(intern, tenant, r.type_id);
            let props = crate::storage::property_payload::rel_record_property_bag_checked(
                &r, blobs, intern, tenant,
            )?;
            (name, props)
        }
        Ok(None) | Err(_) => (None, std::collections::BTreeMap::new()),
    };
    let rel = RelView {
        id: rel_id_typed,
        from: ep.src,
        to: ep.dst,
        rel_type: rel_type_filter,
        rel_type_name,
        properties: rel_props,
    };
    let dst = NodeView {
        id: ep.far_end,
        label: far_label,
        label_name: far_label_name,
        properties: far_props,
    };
    Ok(Some(BoundEdge { rel, dst }))
}

/// W26-β-3 / ADR-132 — substrate-body search provider.
///
/// Production seam between
/// [`CrudExecutorSubstrate::vector_search`] /
/// [`CrudExecutorSubstrate::bm25_search`] and the per-tenant HNSW +
/// Tantivy search machinery. The trait is consumer-defined HERE in
/// `arcgraph-mcp::storage` (parallel to the `HybridSearcher` MCP-tool
/// adapter trait pattern in `crates/arcgraph-mcp/src/tools/search.rs`)
/// to keep the search-side wire-through out of the
/// `arcgraph-query → arcgraph-storage` bounded-context edge documented in
/// `docs/bounded-contexts.md`.
///
/// # Why a trait instead of holding concrete handles
///
/// 1. **HNSW + BM25 are two structurally distinct search-side stores.**
///    BM25 has a canonical entry-point (`Bm25Service::handle(tenant,
///    IndexId::DEFAULT_BM25).search(query, k, read_lsn)` per ADR-039
///    §D-8); HNSW's per-tenant search-side state is the
///    `(TenantId, IndexId)` → `FilteredHnsw` / DiskANN
///    `BackendSet<'a>` borrow (per ADR-035-amendment-04 dispatcher
///    pattern). Conflating them into a single concrete-field would
///    fork the substrate by index kind.
/// 2. **Tests need deterministic injection** — production providers
///    wrap Tantivy directories + HNSW arenas that are heavyweight to
///    construct per-test; the trait lets a unit test inject a stub
///    that returns canned hits without spinning up the underlying
///    engines.
/// 3. **Bounded-context discipline**: substrate-side construction of
///    BackendSet + Bm25Service is the consumer's responsibility (the
///    binary's main-side wiring in `arcgraph-cli::bootstrap`), not
///    the substrate's. The trait pushes the concrete dependency
///    upward without burdening the substrate-layer with crate
///    dependencies on `arcgraph-vector` + `arcgraph-bm25`.
///
/// # Snapshot semantics under `read_lsn` (ADR-132 D-2)
///
/// Implementations MUST honor `read_lsn` per the ADR-035 (HNSW MVCC
/// visibility) + ADR-039 §D-3 (BM25 MVCC visibility) contracts.
/// [`CrudExecutorSubstrate`] resolves `Lsn::MAX` to the actual CRUD
/// transaction snapshot and passes that finite effective LSN; an exact
/// finite request is passed unchanged or rejected before dispatch.
///
/// # Tenant scoping
///
/// Implementations MUST scope every call to the `tenant` argument
/// per ADR-011 + ADR-037 §D-1. The substrate already verifies the
/// router can `route(tenant, PartitionId::ZERO)` BEFORE invoking the
/// provider, so providers can assume the tenant is routed-known.
pub trait SubstrateSearchProvider: Send + Sync + std::fmt::Debug {
    /// HNSW top-K vector search for `(tenant, property)` at
    /// `read_lsn`. Returns up to `k` ranked hits sorted in score
    /// descending order per the [`RankedHit`] contract; an empty
    /// `Vec` is a valid result for an empty index or a query with
    /// no matches.
    ///
    /// # Errors
    ///
    /// Implementations MUST translate engine-level errors into
    /// [`SubstrateAccessError`]:
    /// - Tenant-unknown / index-not-built → `IndexUnavailable`.
    /// - I/O / arena-corruption → `Io`.
    /// - Filter-not-supported (per ADR-035-amendment-04 D-3
    ///   escalation) → `IndexUnavailable` with detail.
    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError>;

    /// BM25 top-K text search for `(tenant, property)` at
    /// `read_lsn`. Same shape + error-translation contract as
    /// [`Self::vector_search`].
    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError>;

    /// Notify an in-process search provider that a node was tombstoned in the
    /// primary store. Providers with derived insert-only indexes can use this
    /// to suppress stale resident entries at query time.
    ///
    /// Default no-op preserves back-compat for providers that do not maintain a
    /// resident vector graph or already honor delete visibility natively.
    fn mark_vector_node_deleted(&self, _tenant: TenantId, _node: NodeId) {}

    /// Notify an in-process search provider that a node's served vector
    /// `property` changed in the primary store. Insert-only derived indexes can
    /// use this to suppress the old resident vector and reinsert the current
    /// value on the next search refresh.
    ///
    /// Default no-op preserves back-compat for providers that do not maintain a
    /// resident vector graph or already honor update visibility natively.
    fn mark_vector_node_updated(&self, _tenant: TenantId, _property: &str, _node: NodeId) {}

    /// Notify an in-process search provider that a node was tombstoned in the
    /// primary store and should be removed from any derived BM25 text index.
    ///
    /// Default no-op preserves back-compat for providers that do not maintain a
    /// resident BM25 index or already honor delete visibility natively.
    fn mark_bm25_node_deleted(&self, _tenant: TenantId, _node: NodeId) {}

    /// Notify an in-process search provider that a node's indexable text changed
    /// in the primary store. Derived BM25 indexes can use this to re-read the
    /// committed node text before the next search.
    ///
    /// Default no-op preserves back-compat for providers that do not maintain a
    /// resident BM25 index or already honor update visibility natively.
    fn mark_bm25_node_updated(&self, _tenant: TenantId, _node: NodeId) {}

    /// #815 / #816a — HNSW top-K vector search with the label filter
    /// pushed INTO the traversal (filter-during-search) plus an
    /// optional query-time `ef_search` recall knob.
    ///
    /// - `label_filter`: when `Some(&[..])` is non-empty, only nodes
    ///   whose label is in the set count toward the top-`k`. The
    ///   candidate frontier still traverses non-matching nodes for
    ///   connectivity, so a SELECTIVE filter returns `k` true matches
    ///   instead of the `k · selectivity` a retrieve-then-discard
    ///   post-filter yields (the #815 recall collapse). `None` /
    ///   empty = no label filter (identical to [`Self::vector_search`]).
    /// - `ef_search`: query-time beam width (HNSW `ef` / Qdrant
    ///   `hnsw_ef` / Milvus `ef`). `None` = the engine default; `Some(n)`
    ///   trades recall for latency — higher → higher recall (#816a).
    ///
    /// The default impl is back-compat: it IGNORES both knobs and
    /// delegates to [`Self::vector_search`] (top-`k` by distance). The
    /// MCP `graph.search` boundary still applies the label post-filter
    /// for such providers, so their behavior is unchanged. The served
    /// HNSW provider OVERRIDES this to do real filter-during-search and
    /// to honor `ef_search`.
    ///
    /// # Errors
    ///
    /// Same translation contract as [`Self::vector_search`].
    // 8 args: the search knobs (label_filter + ef_search) parallel the
    // existing `vector_search` shape; bundling into a struct adds ceremony
    // without clarity at the call sites — same allow precedent as
    // `HnswGraph::search_with_rescore` + `FilteredHnsw::filtered_search`.
    #[allow(clippy::too_many_arguments)]
    fn vector_search_filtered(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        label_filter: Option<&[LabelId]>,
        ef_search: Option<usize>,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        // Back-compat default: knobs unsupported → distance-only top-k.
        // The MCP boundary post-filters by label for these providers.
        let _ = (label_filter, ef_search);
        self.vector_search(tenant, property, query_vec, k, read_lsn)
    }
}

/// **NN-4 (#1384)** — the per-`(tenant, merge-key)` MERGE serialization
/// lock table. See [`CrudExecutorSubstrate::merge_locks`] for the full
/// design + lock-order argument. Aliased to keep the field type readable
/// (clippy `type_complexity`).
type MergeLockTable = Arc<Mutex<HashMap<(TenantId, String), Arc<Mutex<()>>>>>;

/// Storage-backed [`ExecutorSubstrate`] implementation.
///
/// Constructed once per process. Calls are stateless — each call
/// opens its own short-lived [`arcgraph_storage::transaction::Transaction`].
#[derive(Clone)]
pub struct CrudExecutorSubstrate {
    router: Arc<MultiTenantRouter>,
    txn_manager: Arc<TxnManager>,
    /// Intern table used to resolve `LabelId` names + property keys
    /// when the executor surfaces a node back through `NodeView`.
    /// `None` when name resolution is not needed (the trait shape
    /// passes `LabelId` directly; names are only relevant in
    /// downstream MCP serializers).
    intern_table: Arc<InternTable>,
    /// W26-β-3 / ADR-132 — optional per-process search provider.
    /// When `None`, [`Self::vector_search`] + [`Self::bm25_search`]
    /// continue to surface
    /// [`SubstrateAccessError::IndexUnavailable`] even when the
    /// router's [`MultiTenantRouter::route`] handle has a vector /
    /// BM25 attached. Production wiring (CLI / server bootstrap)
    /// constructs the concrete provider and binds it via
    /// [`Self::with_search_provider`].
    search_provider: Option<Arc<dyn SubstrateSearchProvider>>,
    /// **#830 / ADR-200** — the per-tenant vector-index catalog (the
    /// D2/D3 half of ADR-198 §OQ-7). `CREATE VECTOR INDEX` registers a
    /// metadata entry; `SHOW VECTOR INDEXES` reads it;
    /// `db.index.vector.queryNodes` resolves `name → property` against
    /// it. **IN-MEMORY, per-tenant, process-lifetime** — re-created on
    /// restart (acceptable for the langchain happy path, which always
    /// `CREATE … IF NOT EXISTS` first; a persistent catalog is a GA
    /// follow-on per ADR-200). `Arc<RwLock<…>>` so it is shared across
    /// `Clone`s of the substrate (every Bolt connection observes the
    /// same per-tenant catalog) and the `&self` trait methods can mutate
    /// it. The served HNSW BUILD is auto-on-ingest (#765 PART-1) — this
    /// catalog carries NO index data, only metadata.
    vector_index_catalog: Arc<RwLock<HashMap<TenantId, Vec<VectorIndexCatalogEntry>>>>,
    /// **NN-4 (#1384)** — the per-`(tenant, merge-key)` MERGE
    /// serialization lock table. Each entry is a `parking_lot::Mutex`
    /// whose `lock_arc()` returns an OWNING guard (the `arc_lock`
    /// feature); [`ExecutorSubstrate::merge_guard`] hands that guard to
    /// the executor's [`arcgraph_query::executor::ops::MergeOp`], which
    /// holds it across the whole match→create span so two concurrent
    /// `MERGE` on the SAME key cannot both create.
    ///
    /// The OUTER `Mutex<HashMap<…>>` is a SHORT-lived latch held ONLY to
    /// look up / insert the per-key inner `Arc<Mutex<()>>` — it is NEVER
    /// held across the match→create span (that would serialize ALL merges
    /// tenant-wide). The INNER per-key mutex is what the critical section
    /// holds. `Arc`-wrapped so all `Clone`s of the substrate (one per Bolt
    /// connection) share ONE lock table — the racers must contend on the
    /// same inner mutex.
    ///
    /// Lock order (no deadlock — NN-4 §Risks): `merge_locks` outer latch
    /// → inner per-key mutex; and the inner per-key mutex is STRICTLY
    /// OUTER of the MVCC `commit_gate` (no commit path ever acquires a
    /// merge-key mutex). The outer latch is never held while the inner
    /// mutex is acquired (see [`Self::merge_guard`]), so the two-level
    /// table cannot self-deadlock.
    ///
    /// GROWTH: entries are never evicted (a merge key seen once keeps its
    /// mutex for process lifetime). This is bounded in practice by the
    /// distinct-merge-key cardinality of the workload; a GA follow-on may
    /// add LRU eviction of idle entries. At v1.0-α the leak is acceptable
    /// (a mutex is ~1 machine word; get-or-create keys are typically a
    /// bounded set of dedup identities).
    merge_locks: MergeLockTable,
    /// **#1366 (task #248, Phase 1).** The user-visible property-index
    /// manager (durable catalog + per-index secondary B+trees). Shared
    /// (`Arc`) across `Clone`s so every Bolt connection observes one
    /// catalog. Lazily initialized on the first `create_property_index`
    /// (it needs a `PageAllocator` + WAL, which it derives from the
    /// tenant's `CrudStore` at CREATE time). `None` until first use.
    property_index: Arc<RwLock<Option<crate::storage::property_index::PropertyIndexManager>>>,
}

impl std::fmt::Debug for CrudExecutorSubstrate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrudExecutorSubstrate")
            .field("router", &"<Arc<MultiTenantRouter>>")
            .field("txn_manager", &"<Arc<TxnManager>>")
            .field("intern_table", &"<Arc<InternTable>>")
            .field(
                "search_provider",
                match self.search_provider {
                    Some(_) => &"Some(<Arc<dyn SubstrateSearchProvider>>)",
                    None => &"None",
                },
            )
            .field(
                "vector_index_catalog",
                &"<Arc<RwLock<HashMap<TenantId, Vec<VectorIndexCatalogEntry>>>>>",
            )
            .field(
                "merge_locks",
                &"<Arc<Mutex<HashMap<(TenantId, String), Arc<Mutex<()>>>>>>",
            )
            .field(
                "property_index",
                &"<Arc<RwLock<Option<PropertyIndexManager>>>>",
            )
            .finish()
    }
}

/// ADR-197 — newtype wrapping a storage [`OwnedTxn`] so it can be
/// carried opaquely through `arcgraph-query`'s
/// [`HeldTxnHandle`] seam (the orphan rule forbids
/// `impl HeldTxnHandle for OwnedTxn` directly — `HeldTxnHandle` lives
/// in `arcgraph-query`, `OwnedTxn` in `arcgraph-storage`, and this
/// crate owns neither; a local newtype is the canonical bridge).
///
/// The Bolt explicit-transaction handler boxes this as
/// `Box<dyn HeldTxnHandle>` onto the [`ExecutionContext`] before
/// EXECUTE; the substrate's write ops downcast it back here to stage
/// CRUD writes into the held transaction. The handler reclaims it
/// after EXECUTE to `commit()` / `abort()` at the Bolt COMMIT /
/// ROLLBACK message.
///
/// Holds `Option<OwnedTxn>` (not bare `OwnedTxn`) so the COMMIT /
/// ROLLBACK handler can MOVE the transaction out through the
/// `&mut dyn Any` downcast seam (`take()`) and consume it with
/// `commit(self)` / `abort(self)` — trait-object upcasting
/// (`Box<dyn HeldTxnHandle>` → `Box<dyn Any>`) is not available at the
/// 1.85 MSRV, so the `Option::take` path is the MSRV-safe move-out.
/// `None` after the tx has been committed/aborted (a subsequent
/// stage attempt surfaces a clear error rather than a panic).
pub struct BoltHeldTxn {
    owned: Option<OwnedTxn>,
    abort_crud: Option<Arc<CrudStore>>,
    pending_vector_hooks: Vec<PendingVectorHook>,
    pending_bm25_hooks: Vec<PendingBm25Hook>,
    /// ADR-197-amendment-01 D-5 — the transaction's pinned snapshot
    /// LSN, captured at construction so [`HeldTxnHandle::snapshot_lsn`]
    /// is infallible even after the `OwnedTxn` has been moved out
    /// (committed/aborted). `Lsn::ZERO` only for the
    /// finalized-handle placeholder (the repo's reserved
    /// "before any install" sentinel), which is a `mem::replace`
    /// dummy dropped immediately after the swap.
    snapshot_lsn: Lsn,
}

impl std::fmt::Debug for BoltHeldTxn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoltHeldTxn")
            .field("owned", &self.owned)
            .field(
                "abort_crud",
                &self.abort_crud.as_ref().map(|_| "<CrudStore>"),
            )
            .field("pending_vector_hooks", &self.pending_vector_hooks)
            .field("pending_bm25_hooks", &self.pending_bm25_hooks)
            .field("snapshot_lsn", &self.snapshot_lsn)
            .finish()
    }
}

#[derive(Debug, Clone)]
enum PendingVectorHook {
    Deleted {
        tenant: TenantId,
        node: NodeId,
    },
    Updated {
        tenant: TenantId,
        property: String,
        node: NodeId,
    },
}

#[derive(Debug, Clone, Copy)]
enum PendingBm25Hook {
    Deleted { tenant: TenantId, node: NodeId },
    Updated { tenant: TenantId, node: NodeId },
}

impl BoltHeldTxn {
    #[must_use]
    pub fn new(owned: OwnedTxn) -> Self {
        Self::new_with_abort_crud(owned, None)
    }

    #[must_use]
    pub fn new_with_abort_store(owned: OwnedTxn, abort_crud: Arc<CrudStore>) -> Self {
        Self::new_with_abort_crud(owned, Some(abort_crud))
    }

    fn new_with_abort_crud(owned: OwnedTxn, abort_crud: Option<Arc<CrudStore>>) -> Self {
        let snapshot_lsn = owned.snapshot();
        Self {
            owned: Some(owned),
            abort_crud,
            pending_vector_hooks: Vec::new(),
            pending_bm25_hooks: Vec::new(),
            snapshot_lsn,
        }
    }

    pub(crate) fn new_empty_for_finalized_handle() -> Self {
        Self {
            owned: None,
            abort_crud: None,
            pending_vector_hooks: Vec::new(),
            pending_bm25_hooks: Vec::new(),
            snapshot_lsn: Lsn::ZERO,
        }
    }

    fn owned_mut(&mut self) -> Result<&mut OwnedTxn, SubstrateAccessError> {
        self.owned.as_mut().ok_or_else(|| {
            SubstrateAccessError::Io("held tx already committed/aborted (ADR-197)".into())
        })
    }

    pub fn take_owned(&mut self) -> Option<OwnedTxn> {
        self.owned.take()
    }

    pub fn abort(mut self) {
        if let Some(owned) = self.owned.take() {
            if let Some(crud) = self.abort_crud.take() {
                crud.discard_pending_blob_emits(owned.id());
            }
            owned.abort();
        }
    }

    fn queue_vector_node_deleted(&mut self, tenant: TenantId, node: NodeId) {
        self.pending_vector_hooks
            .push(PendingVectorHook::Deleted { tenant, node });
    }

    fn queue_vector_node_updated(&mut self, tenant: TenantId, property: &str, node: NodeId) {
        self.pending_vector_hooks.push(PendingVectorHook::Updated {
            tenant,
            property: property.to_string(),
            node,
        });
    }

    fn queue_bm25_node_deleted(&mut self, tenant: TenantId, node: NodeId) {
        self.pending_bm25_hooks
            .push(PendingBm25Hook::Deleted { tenant, node });
    }

    fn queue_bm25_node_updated(&mut self, tenant: TenantId, node: NodeId) {
        self.pending_bm25_hooks
            .push(PendingBm25Hook::Updated { tenant, node });
    }
}

impl Drop for BoltHeldTxn {
    fn drop(&mut self) {
        if let (Some(owned), Some(crud)) = (&self.owned, &self.abort_crud) {
            crud.discard_pending_blob_emits(owned.id());
        }
    }
}

impl HeldTxnHandle for BoltHeldTxn {
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }
}

/// **NN-4 (#1384)** — the concrete [`MergeGuard`] returned by
/// [`CrudExecutorSubstrate::merge_guard`].
///
/// Wraps a `parking_lot::ArcMutexGuard` — an OWNING mutex guard (via the
/// `arc_lock` feature) that keeps the per-key `Arc<Mutex<()>>` alive AND
/// holds its lock. Dropping this releases the inner mutex, unblocking the
/// next racer on the same merge key. The struct is otherwise inert: the
/// `MergeGuard` trait is empty (release is `Drop`), and the executor only
/// needs to keep the value bound for the critical-section span.
struct CrudMergeGuard {
    // Held for its `Drop` (releases the per-key mutex). The lock is
    // acquired in `merge_guard`; this field simply keeps it held.
    _guard: ArcMutexGuard<RawMutex, ()>,
}

// Manual `Debug` — the inner `ArcMutexGuard` is not `Debug`, but the
// `MergeGuard` trait requires `Debug` (so a guard can be a field of the
// `#[derive(Debug)]` `ExecutionContext` that now stashes it across the
// commit — NN-4 re-spin Fix 1). The guard carries no diagnostic state
// worth printing (release is `Drop`), so an opaque marker suffices.
impl std::fmt::Debug for CrudMergeGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CrudMergeGuard(held)")
    }
}

impl MergeGuard for CrudMergeGuard {}

impl CrudExecutorSubstrate {
    /// Construct a new substrate over the workspace's shared
    /// multi-tenant router + txn manager + intern table. The
    /// [`SubstrateSearchProvider`] is left unbound — call
    /// [`Self::with_search_provider`] post-construction to light the
    /// HNSW + BM25 search bodies (W26-β-3 / ADR-132).
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
            search_provider: None,
            vector_index_catalog: Arc::new(RwLock::new(HashMap::new())),
            // NN-4 (#1384) — the per-(tenant, merge-key) serialization
            // table starts empty; entries are minted lazily on first
            // MERGE for a given key (see `merge_guard`).
            merge_locks: Arc::new(Mutex::new(HashMap::new())),
            // #1366 (task #248) — the property-index manager is lazily
            // built on the first CREATE INDEX (it derives its allocator +
            // WAL from the tenant's CrudStore).
            property_index: Arc::new(RwLock::new(None)),
        }
    }

    /// W26-β-3 / ADR-132 — bind the per-process search provider.
    ///
    /// Builder-style for ergonomic single-line construction at
    /// process bootstrap:
    ///
    /// ```ignore
    /// let substrate = CrudExecutorSubstrate::new(router, txn_mgr, intern)
    ///     .with_search_provider(Arc::clone(&search_provider) as _);
    /// ```
    ///
    /// The substrate holds the provider as `Arc<dyn
    /// SubstrateSearchProvider>` so a single provider can serve many
    /// concurrent executor sessions. Re-calling
    /// `with_search_provider` REPLACES the previously-bound
    /// provider; the substrate accepts at most one provider at any
    /// given time.
    #[must_use]
    pub fn with_search_provider(mut self, provider: Arc<dyn SubstrateSearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// Borrow the currently-bound search provider, if any. Returns
    /// `None` until [`Self::with_search_provider`] is called.
    /// Test / observability accessor; not on the hot path.
    #[inline]
    #[must_use]
    pub fn search_provider(&self) -> Option<&Arc<dyn SubstrateSearchProvider>> {
        self.search_provider.as_ref()
    }

    /// Acquire the per-tenant `CrudStore` handle. Surfaces
    /// `SubstrateAccessError::TenantUnknown` on a routing miss.
    pub(crate) fn crud_for(
        &self,
        tenant: TenantId,
    ) -> Result<Arc<CrudStore>, SubstrateAccessError> {
        let handle = self
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| translate_routing_error(e, tenant))?;
        Ok(Arc::clone(handle.crud()))
    }

    /// Resolve the public snapshot contract against a transaction's
    /// actual snapshot. A finite request is exact; [`Lsn::MAX`] is the
    /// read-latest sentinel and resolves to `available`.
    #[inline]
    fn resolve_read_snapshot(requested: Lsn, available: Lsn) -> Result<Lsn, SubstrateAccessError> {
        if requested == Lsn::MAX || requested == available {
            Ok(available)
        } else {
            Err(SubstrateAccessError::SnapshotUnavailable {
                requested,
                available,
            })
        }
    }

    /// Open and validate one borrowed read transaction.
    fn begin_read_txn(
        &self,
        tenant: TenantId,
        requested: Lsn,
    ) -> Result<Transaction<'_>, SubstrateAccessError> {
        let tx = self.txn_manager.begin(tenant);
        Self::resolve_read_snapshot(requested, tx.snapshot())?;
        Ok(tx)
    }

    /// Open and validate one owned read transaction for a streaming
    /// cursor.
    fn begin_owned_read_txn(
        &self,
        tenant: TenantId,
        requested: Lsn,
    ) -> Result<OwnedTxn, SubstrateAccessError> {
        let tx = self.txn_manager.begin_owned(tenant);
        Self::resolve_read_snapshot(requested, tx.snapshot())?;
        Ok(tx)
    }

    /// **#1366 (task #248, Phase 1).** Lazily build-or-return the shared
    /// property-index manager. It needs a `PageAllocator` + WAL for its
    /// secondary B+trees, which it derives from `crud` (the tenant's
    /// `CrudStore`) on first use. Subsequent calls return the same
    /// manager (shared across `Clone`s of the substrate). If the store
    /// has no allocator (a bare in-memory unit fixture) the manager
    /// falls back to a fresh allocator — the index is still functional,
    /// just not sharing page-id space with the store (acceptable: the
    /// secondary index owns a disjoint SYSTEM-tenant page domain).
    pub fn property_index_manager(
        &self,
        crud: &CrudStore,
    ) -> Result<crate::storage::property_index::PropertyIndexManager, SubstrateAccessError> {
        if let Some(m) = self.property_index.read().ok().and_then(|g| g.clone()) {
            return Ok(m);
        }
        let mut guard = self.property_index.write().map_err(|e| {
            SubstrateAccessError::Io(format!("property-index manager lock poisoned: {e}"))
        })?;
        if let Some(m) = guard.as_ref() {
            return Ok(m.clone());
        }
        let allocator = crud
            .allocator()
            .cloned()
            .unwrap_or_else(|| Arc::new(arcgraph_storage::page_alloc::PageAllocator::new()));
        let catalog =
            Arc::new(arcgraph_storage::property_index_catalog::PropertyIndexCatalog::new());
        // Recover the catalog from durable state (best-effort — a fresh
        // process comes up empty; a restart re-reads the header).
        catalog.recover(&self.txn_manager, arcgraph_core::Lsn::ZERO);
        let manager = crate::storage::property_index::PropertyIndexManager::new(
            catalog,
            Arc::clone(&self.txn_manager),
            allocator,
            crud.wal().cloned(),
        );
        *guard = Some(manager.clone());
        Ok(manager)
    }

    /// **#1366 (task #248).** Apply property-index maintenance for a
    /// node write, best-effort. No-op (zero cost) until at least one
    /// property index has been declared (the manager is `None`), so the
    /// common no-index path pays only one `RwLock::read`. Maintenance
    /// failures are logged, never propagated (the index is a read
    /// accelerator; a failed insert leaves a residual-filter fallback).
    fn maintain_property_index_best_effort(
        &self,
        tenant: TenantId,
        node: NodeId,
        label: LabelId,
        old_bag: Option<&std::collections::BTreeMap<String, Value>>,
        new_bag: &std::collections::BTreeMap<String, Value>,
    ) {
        // Fast path: no manager ⇒ no declared index ⇒ nothing to do.
        let Some(manager) = self.property_index.read().ok().and_then(|g| g.clone()) else {
            return;
        };
        if manager.indexes_on(tenant, label).is_empty() {
            return;
        }
        if let Err(e) = manager.maintain_node(tenant, node, label, old_bag, new_bag) {
            tracing::warn!(
                target: "arcgraph_mcp::property_index",
                node = node.raw(),
                error = %e,
                "property-index maintenance failed (write succeeded; index may lag until rebuild)"
            );
        }
    }

    /// Convenience: borrow the intern table so adapters can map
    /// [`LabelId`] / [`TypeId`] to user-visible names.
    pub fn intern_table(&self) -> &Arc<InternTable> {
        &self.intern_table
    }

    /// #871 — reverse-resolve a raw label id to its catalog NAME via the
    /// intern table, for surfacing on [`NodeView::label_name`] at scan /
    /// expand materialization. The interned `LabelId` is opaque outside
    /// the catalog; `labels(n)` + the Bolt / JSON serializers need the
    /// human name. Returns `None` for the no-label sentinel (id `0`) or
    /// an id absent from the table (so the serializers emit an empty
    /// labels list rather than leak the id). Mirrors the existing
    /// resolution in [`crate::storage::adapters`] (`inspect` / `explore`).
    fn resolve_label_name(&self, tenant: TenantId, label_id: u32) -> Option<String> {
        if label_id == 0 {
            return None;
        }
        self.intern_table
            .resolve(tenant, arcgraph_core::ids::StringId::new(label_id))
            .map(|arc| arc.to_string())
    }

    /// #871 — reverse-resolve a raw rel-type id to its catalog NAME (the
    /// rel-type sibling of [`Self::resolve_label_name`]), for
    /// [`RelView::rel_type_name`] / `type(r)`.
    fn resolve_rel_type_name(&self, tenant: TenantId, type_id: u32) -> Option<String> {
        if type_id == 0 {
            return None;
        }
        self.intern_table
            .resolve(tenant, arcgraph_core::ids::StringId::new(type_id))
            .map(|arc| arc.to_string())
    }

    /// Convenience: borrow the router for adapters that need other
    /// substrate handles (vector / BM25 / community).
    pub fn router(&self) -> &Arc<MultiTenantRouter> {
        &self.router
    }

    /// Convenience: borrow the txn manager for adapters that open
    /// their own transactions outside the substrate trait calls.
    pub fn txn_manager(&self) -> &Arc<TxnManager> {
        &self.txn_manager
    }

    /// ADR-197 — run a single CRUD write `op` either in EXPLICIT-tx
    /// mode (stage into the connection's held transaction; NO commit —
    /// the Bolt COMMIT message commits it later) or in AUTO-COMMIT mode
    /// (the v1.0-α one-call-one-tx path: `begin → op → commit`,
    /// byte-for-byte preserved).
    ///
    /// `op(&CrudStore, &mut Transaction) -> Result<R, CrudError>` stages
    /// the write into the supplied transaction WITHOUT committing —
    /// exactly what the `crud::*` free functions do. The branch on
    /// [`ExecutionContext::has_held_txn`] decides which transaction `op`
    /// stages into and whether a commit follows.
    ///
    /// On error in auto-commit mode the partial transaction is
    /// discarded (the `Transaction`'s Drop aborts; the
    /// `discard_pending*` calls are defense-in-depth, mirroring the
    /// pre-ADR-197 per-method bodies). In explicit mode a staging error
    /// leaves the held transaction intact for the handler to abort at
    /// ROLLBACK / RESET (the Bolt FSM moves to Failed).
    fn stage_or_commit<R>(
        &self,
        tenant: TenantId,
        ctx: &ExecutionContext,
        crud: &CrudStore,
        op: impl FnOnce(&CrudStore, &mut Transaction<'_>) -> Result<R, crud::CrudError>,
    ) -> Result<R, SubstrateAccessError> {
        if ctx.has_held_txn() {
            // EXPLICIT mode: stage into the held tx; do NOT commit.
            // The downcast recovers the concrete OwnedTxn the Bolt
            // handler installed via `BoltHeldTxn`.
            ctx.with_held_txn_mut(|h| {
                let owned = h
                    .as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })?
                    .owned_mut()?;
                op(crud, owned.txn_mut())
                    .map_err(|e| SubstrateAccessError::Io(format!("write staged-in-tx: {e}")))
            })
            .expect("has_held_txn() == true ⇒ with_held_txn_mut yields Some")
        } else {
            // AUTO-COMMIT mode: begin → op → commit (the v1.0-α path).
            let mut tx = self.txn_manager.begin(tenant);
            let out = match op(crud, &mut tx) {
                Ok(out) => out,
                Err(e) => {
                    crud.discard_pending(tx.id());
                    crud.discard_pending_installs(tx.id());
                    return Err(SubstrateAccessError::Io(format!(
                        "write storage-rejected: {e}"
                    )));
                }
            };
            match arcgraph_storage::crud::commit(tx, crud) {
                Ok(_lsn) => Ok(out),
                // #907 — preserve an MVCC write-write conflict as the
                // typed retriable variant; everything else stays `Io`.
                Err(e) => Err(commit_err_to_substrate("write commit failed", e)),
            }
        }
    }

    /// ADR-197 — general multi-op variant of [`Self::stage_or_commit`]
    /// for read-modify-write methods that stage several CRUD ops (and
    /// possibly validation reads) into ONE transaction (`delete_node`'s
    /// cascade, the `set_*` / `remove_*` read-merge-write). The `body`
    /// closure owns ALL staging + validation and returns a
    /// [`SubstrateAccessError`] directly (so non-`CrudError` validation
    /// failures like "relationships attached" surface unwrapped).
    ///
    /// EXPLICIT mode: run `body` against the held tx; do NOT commit (the
    /// Bolt COMMIT commits; a `body` error leaves the held tx for the
    /// handler to abort at ROLLBACK / RESET). AUTO-COMMIT mode: `begin →
    /// body → commit`; on `body` error discard the partial tx.
    ///
    /// NOTE: `body` MUST NOT commit/abort the transaction itself — it
    /// only stages. The held tx's multi-op atomicity (all-or-nothing at
    /// the Bolt COMMIT) is exactly the desired behavior for a
    /// multi-statement explicit transaction.
    fn run_txn<R>(
        &self,
        tenant: TenantId,
        ctx: &ExecutionContext,
        body: impl FnOnce(&CrudStore, &mut Transaction<'_>) -> Result<R, SubstrateAccessError>,
    ) -> Result<R, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        if ctx.has_held_txn() {
            ctx.with_held_txn_mut(|h| {
                let owned = h
                    .as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })?
                    .owned_mut()?;
                body(&crud, owned.txn_mut())
            })
            .expect("has_held_txn() == true ⇒ with_held_txn_mut yields Some")
        } else {
            let mut tx = self.txn_manager.begin(tenant);
            match body(&crud, &mut tx) {
                Ok(out) => match arcgraph_storage::crud::commit(tx, &crud) {
                    Ok(_lsn) => Ok(out),
                    // #907 — preserve an MVCC write-write conflict as the
                    // typed retriable variant; everything else stays `Io`.
                    Err(e) => Err(commit_err_to_substrate("write commit failed", e)),
                },
                Err(e) => {
                    crud.discard_pending(tx.id());
                    crud.discard_pending_installs(tx.id());
                    Err(e)
                }
            }
        }
    }

    /// ADR-197 #802 R1 finding #1 — commit a held [`OwnedTxn`] (the Bolt
    /// explicit-transaction handle) through the FULL
    /// [`arcgraph_storage::crud::commit`] machinery, converging the
    /// explicit-tx COMMIT with the auto-commit path.
    ///
    /// The held tx buffered its primary-index installs + TEL appends into
    /// THIS substrate's per-tenant `CrudStore` (keyed by
    /// the tx id) during `run_in_txn` staging (the EXPLICIT branch of
    /// `Self::stage_or_commit` / `Self::run_txn`). Routing back to
    /// the SAME store via `Self::crud_for` and calling `crud::commit`
    /// drains them by that id under one CommitBundle fsync — the
    /// IDENTICAL `take_installs` + `install_create` +
    /// `primary.upsert_deferred` + WAL work the auto-commit path runs
    /// (see `Self::stage_or_commit`'s AUTO-COMMIT
    /// branch). The difference between auto-commit and explicit is ONLY
    /// the tx lifetime (one-call vs held-across-RUNs), NOT the commit
    /// semantics: they converge here.
    ///
    /// Before this fix the Bolt COMMIT path called `OwnedTxn::commit`
    /// (MVCC-version-store ONLY), silently skipping the primary-index
    /// dual-write + WAL for managed-tx writes — a write "committed"
    /// to MVCC but invisible to a primary-index lookup / a WAL recovery
    /// (R1 finding #1).
    ///
    /// On commit failure the `CrudError` surfaces via
    /// `commit_err_to_substrate`: a write-write **MVCC conflict** as the
    /// typed retriable [`SubstrateAccessError::Conflict`] (#907 — so the
    /// Bolt COMMIT maps it to a retriable `Neo.TransientError.*`), every
    /// other fault as [`SubstrateAccessError::Io`]. `crud::commit`
    /// internally discards its own pending installs on the error
    /// path, so no partial side effect leaks from a failed commit.
    pub fn commit_held_txn(&self, owned: OwnedTxn) -> Result<Lsn, SubstrateAccessError> {
        let crud = self.crud_for(owned.tenant())?;
        // #907 — preserve an MVCC write-write conflict as the typed
        // retriable variant so the Bolt COMMIT path maps it to a
        // `Neo.TransientError.*` (driver auto-retry) instead of a fatal
        // `Neo.DatabaseError`; every other commit fault stays `Io`.
        arcgraph_storage::crud::commit(owned.into_inner(), &crud)
            .map_err(|e| commit_err_to_substrate("explicit-tx commit failed", e))
    }

    /// #963 — commit a Bolt held transaction and fire any served-HNSW
    /// maintenance hooks queued while statements staged into it. The
    /// hooks run only after `crud::commit` succeeds; on commit failure
    /// or rollback/drop the queue is discarded with the held handle.
    pub fn commit_bolt_held_txn(&self, mut held: BoltHeldTxn) -> Result<Lsn, SubstrateAccessError> {
        let owned = held.take_owned().ok_or_else(|| {
            SubstrateAccessError::Io("held tx already committed/aborted (ADR-197)".into())
        })?;
        let lsn = self.commit_held_txn(owned)?;
        if let Some(provider) = self.search_provider.as_ref() {
            for hook in held.pending_vector_hooks.drain(..) {
                match hook {
                    PendingVectorHook::Deleted { tenant, node } => {
                        provider.mark_vector_node_deleted(tenant, node);
                    }
                    PendingVectorHook::Updated {
                        tenant,
                        property,
                        node,
                    } => {
                        provider.mark_vector_node_updated(tenant, &property, node);
                    }
                }
            }
            for hook in held.pending_bm25_hooks.drain(..) {
                match hook {
                    PendingBm25Hook::Deleted { tenant, node } => {
                        provider.mark_bm25_node_deleted(tenant, node);
                    }
                    PendingBm25Hook::Updated { tenant, node } => {
                        provider.mark_bm25_node_updated(tenant, node);
                    }
                }
            }
        }
        Ok(lsn)
    }

    pub fn commit_bolt_held_handle(
        &self,
        mut held: Box<dyn HeldTxnHandle>,
    ) -> Result<Lsn, SubstrateAccessError> {
        let held = held
            .as_any_mut()
            .downcast_mut::<BoltHeldTxn>()
            .ok_or_else(|| {
                SubstrateAccessError::Io(
                    "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                )
            })
            .map(|b| std::mem::replace(b, BoltHeldTxn::new_empty_for_finalized_handle()))?;
        self.commit_bolt_held_txn(held)
    }

    /// Return every node property whose vector tier must be invalidated on an
    /// in-place property-bag mutation. The default served convention applies
    /// without catalog metadata; registered indexes extend it per tenant.
    fn vector_properties_for_node_mutation(
        &self,
        tenant: TenantId,
        operation: &str,
    ) -> Result<HashSet<String>, SubstrateAccessError> {
        let catalog = self.vector_index_catalog.read().map_err(|e| {
            SubstrateAccessError::Io(format!(
                "vector-index catalog read lock poisoned during {operation}: {e}"
            ))
        })?;
        let mut properties =
            HashSet::from([crate::tools::search::DEFAULT_VECTOR_PROPERTY.to_string()]);
        if let Some(entries) = catalog.get(&tenant) {
            properties.extend(entries.iter().map(|entry| entry.property.clone()));
        }
        Ok(properties)
    }

    fn mark_or_queue_vector_node_deleted(
        &self,
        tenant: TenantId,
        node: NodeId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        if ctx.has_held_txn() {
            ctx.with_held_txn_mut(|h| {
                h.as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })
                    .map(|held| held.queue_vector_node_deleted(tenant, node))
            })
            .expect("has_held_txn() == true ⇒ with_held_txn_mut yields Some")
        } else {
            if let Some(provider) = self.search_provider.as_ref() {
                provider.mark_vector_node_deleted(tenant, node);
            }
            Ok(())
        }
    }

    fn mark_or_queue_vector_node_updated(
        &self,
        tenant: TenantId,
        property: &str,
        node: NodeId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        if ctx.has_held_txn() {
            ctx.with_held_txn_mut(|h| {
                h.as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })
                    .map(|held| held.queue_vector_node_updated(tenant, property, node))
            })
            .expect("has_held_txn() == true ⇒ with_held_txn_mut yields Some")
        } else {
            if let Some(provider) = self.search_provider.as_ref() {
                provider.mark_vector_node_updated(tenant, property, node);
            }
            Ok(())
        }
    }

    fn mark_or_queue_bm25_node_deleted(
        &self,
        tenant: TenantId,
        node: NodeId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        if ctx.has_held_txn() {
            ctx.with_held_txn_mut(|h| {
                h.as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })
                    .map(|held| held.queue_bm25_node_deleted(tenant, node))
            })
            .expect("has_held_txn() == true ⇒ with_held_txn_mut yields Some")
        } else {
            if let Some(provider) = self.search_provider.as_ref() {
                provider.mark_bm25_node_deleted(tenant, node);
            }
            Ok(())
        }
    }

    fn mark_or_queue_bm25_node_updated(
        &self,
        tenant: TenantId,
        node: NodeId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        if ctx.has_held_txn() {
            ctx.with_held_txn_mut(|h| {
                h.as_any_mut()
                    .downcast_mut::<BoltHeldTxn>()
                    .ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                        )
                    })
                    .map(|held| held.queue_bm25_node_updated(tenant, node))
            })
            .expect("has_held_txn() == true ⇒ with_held_txn_mut yields Some")
        } else {
            if let Some(provider) = self.search_provider.as_ref() {
                provider.mark_bm25_node_updated(tenant, node);
            }
            Ok(())
        }
    }

    /// Materialize a stored [`NodeRecord`] into the wire-shaped
    /// [`NodeView`]: resolve its interned [`LabelId`], decode its
    /// persisted property bag, and reverse-resolve its label NAME.
    ///
    /// This is the SINGLE node-materialization idiom shared by the MATCH
    /// path ([`Self::scan_id_range`]) and the `CALL
    /// db.index.vector.queryNodes` path ([`ExecutorSubstrate::vector_search`]
    /// post-hydration, #830 D4). Centralizing it guarantees a
    /// `queryNodes`-returned node is shaped IDENTICALLY to a
    /// MATCH-returned node (same `label` / `label_name` / `properties`),
    /// so a copy-paste drift between the two paths — the #830 D4 defect,
    /// where the served HNSW provider emitted an empty-`NodeView::new`
    /// hit — cannot recur silently.
    ///
    /// - ADR-152 §D-3 — decode the persisted property bag from the
    ///   record's `property_ref` blob (empty when the bag is
    ///   zero-length). The storage-internal `inline_u32a/b` fields are
    ///   deliberately NOT surfaced on the wire (R1 review MED-6, PR #349)
    ///   — `record_property_bag` drops them.
    /// - #871 — reverse-resolve the label NAME so `labels(n)` + the Bolt
    ///   / JSON / MCP serializers surface `["Doc"]`, never the opaque
    ///   `LabelId` debug form (`"LabelId(1)"`).
    fn hydrate_node_view(
        &self,
        tenant: TenantId,
        crud: &CrudStore,
        rec: &NodeRecord,
    ) -> Result<NodeView, SubstrateAccessError> {
        crate::storage::property_payload::hydrate_node_view(
            tenant,
            crud,
            &self.intern_table,
            rec,
            |label_id| self.resolve_label_name(tenant, label_id),
        )
        .map_err(SubstrateAccessError::from)
    }

    /// **#1401 — snapshot-consistent property-index backfill scan.**
    ///
    /// Begins ONE read tx (snapshot S) and derives the id upper bound
    /// (`high_water`) from WITHIN that snapshot — so the two are
    /// consistent. This closes the second W1 race path: the pre-`#1401`
    /// `create_property_index` sampled `high_water` (substrate.rs:2895)
    /// BEFORE the scan's begin (substrate.rs:1238), so a node created in
    /// that sub-window was visible-to-snapshot yet had `id > high_water`
    /// and fell OUTSIDE the `1..=high_water` iteration range — dropped
    /// from the backfill Vec even though it was snapshot-visible.
    ///
    /// A node visible to S committed before S began; its id was allocated
    /// before it committed (alloc precedes the data commit in
    /// `create_node`), hence before S. `high_water` is the
    /// monotonic max allocated id, sampled after S begins, so every
    /// snapshot-visible node has `id ≤ high_water`. No visible node falls
    /// past the bound.
    fn scan_for_backfill(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        // Begin the snapshot FIRST, then sample high_water inside it —
        // consistent id bound (no high_water-vs-scan-begin skew).
        let tx = self.txn_manager.begin(tenant);
        let high_water = crud.node_high_water(tenant);
        if high_water == 0 {
            return Ok(Vec::new());
        }
        self.scan_id_range_in_tx(tenant, &crud, &tx, 1..=high_water, label)
    }

    /// Point-read every node id in `range`, decode its property bag,
    /// and return the live nodes ascending by id (tombstoned /
    /// not-yet-visible ids are silently absent).
    fn scan_id_range_in_tx(
        &self,
        tenant: TenantId,
        crud: &CrudStore,
        tx: &Transaction<'_>,
        range: std::ops::RangeInclusive<u64>,
        label: Option<LabelId>,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut out: Vec<BoundNode> = Vec::new();
        for raw in range {
            let nid = NodeId::new(raw);
            match crud::read_node(tx, nid) {
                Ok(Some(rec)) => {
                    if let Some(filter) = label {
                        if rec.label_id != filter.raw() {
                            continue;
                        }
                    }
                    // Materialize via the shared idiom so a MATCH-returned
                    // node is shaped identically to a
                    // `queryNodes`-returned one (#830 D4): decode the
                    // property bag (ADR-152 §D-3) + reverse-resolve the
                    // label NAME (#871). See [`Self::hydrate_node_view`].
                    let node = self.hydrate_node_view(tenant, crud, &rec)?;
                    out.push(BoundNode { node });
                }
                Ok(None) => {
                    // Tombstoned / not-yet-visible at this snapshot.
                }
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "scan_nodes: read_node({raw}) failed: {e}"
                    )));
                }
            }
        }
        // Deterministic order: ascending by NodeId (matches the
        // stub substrate's contract).
        out.sort_by_key(|b| b.node.id.raw());
        Ok(out)
    }

    fn node_by_id_in_tx(
        &self,
        tenant: TenantId,
        crud: &CrudStore,
        tx: &Transaction<'_>,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        match crud::read_node(tx, id) {
            Ok(Some(rec)) => {
                let node = self.hydrate_node_view(tenant, crud, &rec)?;
                Ok(Some(BoundNode { node }))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(SubstrateAccessError::Io(format!(
                "node_by_id_with_context: read_node({raw}) failed: {e}",
                raw = id.raw()
            ))),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn verify_property_candidates_in_tx(
        &self,
        tenant: TenantId,
        crud: &CrudStore,
        tx: &Transaction<'_>,
        candidates: Vec<NodeId>,
        label: LabelId,
        property: &str,
        value: &Value,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for candidate in candidates {
            let candidate = NodeId::new(candidate.raw());
            if !seen.insert(candidate) {
                continue;
            }
            let Some(bound) = self.node_by_id_in_tx(tenant, crud, tx, candidate)? else {
                continue;
            };
            if bound.node.label == Some(label)
                && bound
                    .node
                    .properties
                    .get(property)
                    .is_some_and(|candidate_value| index_value_eq_coerced(candidate_value, value))
            {
                out.push(bound);
            }
        }
        Ok(out)
    }

    /// #787 — O(delta) incremental delta-scan: the live nodes whose id is
    /// in `(low_exclusive, high_inclusive]`. The served HNSW provider
    /// (`arcgraph_cli::vector_search`) calls this with `low_exclusive` = the
    /// high-water mark its cached index was last built at and
    /// `high_inclusive` = the current high-water, so a read-after-write
    /// inserts only the newly-allocated nodes instead of paying a full
    /// O(N) `scan_nodes` + rebuild. An empty / inverted range is a no-op
    /// (returns no nodes), which is exactly the "no new nodes, reuse the
    /// cached index" fast path. Returns nodes ascending by id.
    pub fn scan_nodes_in_id_range(
        &self,
        tenant: TenantId,
        low_exclusive: u64,
        high_inclusive: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        let tx = self.begin_read_txn(tenant, read_lsn)?;
        if high_inclusive <= low_exclusive {
            return Ok(Vec::new());
        }
        self.scan_id_range_in_tx(
            tenant,
            &crud,
            &tx,
            (low_exclusive + 1)..=high_inclusive,
            None,
        )
    }

    /// v2 M2 (design §M2.3) — the PROJECTED scan body: like
    /// [`Self::scan_id_range_in_tx`] but each node's bag materializes
    /// ONLY the projection's key_ids through the typed zero-decode
    /// read ([`crate::storage::property_payload::record_property_bag_projected`]
    /// — `PropBlockView` touches nothing else). The name → key_id
    /// resolution happens ONCE here (never per row).
    ///
    /// Budget (PD#5): O(|projection|) intern probes once + per node
    /// O(|projection| × log 64) header probes + O(K) value
    /// materialization — vs the full-bag O(|bag|) decode.
    #[allow(clippy::too_many_arguments)] // the scan context set + the projection — mirrors scan_id_range_in_tx
    fn scan_id_range_in_tx_projected(
        &self,
        tenant: TenantId,
        crud: &CrudStore,
        tx: &Transaction<'_>,
        range: std::ops::RangeInclusive<u64>,
        label: Option<LabelId>,
        projected: &[String],
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        // Fail-closed: a probe failure here would otherwise drop the property
        // from every row of this scan (a silently wrong answer).
        let projection = crate::storage::property_payload::ResolvedProjection::resolve(
            projected,
            &self.intern_table,
            tenant,
        )
        .map_err(|error| {
            SubstrateAccessError::Io(format!("resolving projected property names: {error}"))
        })?;
        let blobs = crud.blob_store();
        let mut out: Vec<BoundNode> = Vec::new();
        for raw in range {
            let nid = NodeId::new(raw);
            match crud::read_node(tx, nid) {
                Ok(Some(rec)) => {
                    if let Some(filter) = label {
                        if rec.label_id != filter.raw() {
                            continue;
                        }
                    }
                    let properties =
                        crate::storage::property_payload::record_property_bag_projected(
                            &rec,
                            blobs,
                            &self.intern_table,
                            tenant,
                            &projection,
                        )?;
                    let node = NodeView {
                        id: nid,
                        label: (rec.label_id != 0).then(|| LabelId::new(rec.label_id)),
                        label_name: self.resolve_label_name(tenant, rec.label_id),
                        properties,
                    };
                    out.push(BoundNode { node });
                }
                Ok(None) => {}
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "scan_nodes_projected: read_node({raw}) failed: {e}"
                    )));
                }
            }
        }
        out.sort_by_key(|b| b.node.id.raw());
        Ok(out)
    }
}

impl ExecutorSubstrate for CrudExecutorSubstrate {
    fn count_store(
        &self,
        tenant: TenantId,
        source: arcgraph_query::logical_plan::CountStoreSource,
    ) -> Result<u64, SubstrateAccessError> {
        use arcgraph_query::logical_plan::CountStoreSource as Src;
        let crud = self.crud_for(tenant)?;
        if let Some(stats) = crud.catalog_stats(tenant) {
            let snapshot = stats.snapshot();
            let count = match source {
                Src::Nodes => snapshot.total_nodes(),
                Src::Relationships => snapshot.total_rels(),
                // F1 (#1356 §F1): read the EXISTING per-label / per-type
                // counter the commit pipeline already maintains — an O(1)
                // read that lowers `MATCH (n:Label) RETURN count(n)` off the
                // full label scan (8.76 → ~1.0 ms). Gate on the tenant-wide
                // total being observed (mirrors the unlabelled arms above):
                // a `None` total means the stats aren't populated yet → fall
                // through to the defensive scan below. A LIVE total with no
                // entry for this label/type means the label/type has zero
                // rows (`*_card` None → 0), NOT "unknown".
                Src::NodesWithLabel(label) => snapshot
                    .total_nodes()
                    .map(|_| snapshot.label_card(label).unwrap_or(0)),
                Src::RelsWithType(rel_type) => snapshot
                    .total_rels()
                    .map(|_| snapshot.rel_type_card(rel_type).unwrap_or(0)),
            };
            if let Some(count) = count {
                return Ok(count);
            }
        }

        match source {
            Src::Nodes => Ok(self.scan_nodes(tenant, None, Lsn::MAX)?.len() as u64),
            // F1 fallback (only reached when `CatalogStats` is unwired /
            // unpopulated): a filtered scan — still CORRECT, just not O(1).
            // Production always has the stats hook wired, so this is the
            // defensive path.
            Src::NodesWithLabel(label) => {
                Ok(self.scan_nodes(tenant, Some(label), Lsn::MAX)?.len() as u64)
            }
            Src::Relationships => {
                let mut count = 0u64;
                for node in self.scan_nodes(tenant, None, Lsn::MAX)? {
                    let edges =
                        self.expand(tenant, node.node.id, None, Direction::LeftToRight, Lsn::MAX)?;
                    count = count.checked_add(edges.len() as u64).ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "count_store: scanned relationship count overflow".into(),
                        )
                    })?;
                }
                Ok(count)
            }
            // F1 fallback: filtered expand — pass the rel-type to `expand`
            // so each node contributes only its matching-type out-edges
            // (each directed edge is outbound from exactly one node → counted
            // once).
            Src::RelsWithType(rel_type) => {
                let mut count = 0u64;
                for node in self.scan_nodes(tenant, None, Lsn::MAX)? {
                    let edges = self.expand(
                        tenant,
                        node.node.id,
                        Some(rel_type),
                        Direction::LeftToRight,
                        Lsn::MAX,
                    )?;
                    count = count.checked_add(edges.len() as u64).ok_or_else(|| {
                        SubstrateAccessError::Io(
                            "count_store: scanned typed-relationship count overflow".into(),
                        )
                    })?;
                }
                Ok(count)
            }
        }
    }

    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        // W17α perf budget: O(node_high_water) iteration with
        // per-node `read_node`. A 100K-node tenant performs 100K
        // point reads per query; v1.1 wires a label-index path
        // (issue #351) that walks the label's id-set directly. The
        // #787 served-HNSW read-after-write path instead uses the
        // O(delta) `scan_nodes_in_id_range` so it does not pay this
        // full scan on every query that follows an ingest.
        let crud = self.crud_for(tenant)?;
        let tx = self.begin_read_txn(tenant, read_lsn)?;
        let high_water = crud.node_high_water(tenant);
        self.scan_id_range_in_tx(tenant, &crud, &tx, 1..=high_water, label)
    }

    fn scan_nodes_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        if !ctx.has_held_txn() {
            return self.scan_nodes(ctx.tenant(), label, read_lsn);
        }
        let tenant = ctx.tenant();
        let crud = self.crud_for(tenant)?;
        let high_water = crud.node_high_water(tenant);
        ctx.with_held_txn_mut(|h| {
            let owned = h
                .as_any_mut()
                .downcast_mut::<BoltHeldTxn>()
                .ok_or_else(|| {
                    SubstrateAccessError::Io(
                        "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                    )
                })?
                .owned_mut()?;
            Self::resolve_read_snapshot(read_lsn, owned.snapshot())?;
            self.scan_id_range_in_tx(tenant, &crud, owned.txn_mut(), 1..=high_water, label)
        })
        .expect("has_held_txn() == true => with_held_txn_mut yields Some")
    }

    /// v2 M2 (design §M2.3) — the production projected scan. Mirrors
    /// [`Self::scan_nodes_with_context`]'s snapshot/held-txn handling;
    /// the bag materialization is restricted to the pushed projection
    /// (the zero-decode read — bytes are touched only for the
    /// projected key_ids).
    fn scan_nodes_projected_with_context(
        &self,
        ctx: &ExecutionContext,
        label: Option<LabelId>,
        read_lsn: Lsn,
        projected_properties: &[String],
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let tenant = ctx.tenant();
        let crud = self.crud_for(tenant)?;
        let high_water = crud.node_high_water(tenant);
        if !ctx.has_held_txn() {
            let tx = self.begin_read_txn(tenant, read_lsn)?;
            return self.scan_id_range_in_tx_projected(
                tenant,
                &crud,
                &tx,
                1..=high_water,
                label,
                projected_properties,
            );
        }
        ctx.with_held_txn_mut(|h| {
            let owned = h
                .as_any_mut()
                .downcast_mut::<BoltHeldTxn>()
                .ok_or_else(|| {
                    SubstrateAccessError::Io(
                        "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                    )
                })?
                .owned_mut()?;
            Self::resolve_read_snapshot(read_lsn, owned.snapshot())?;
            self.scan_id_range_in_tx_projected(
                tenant,
                &crud,
                owned.txn_mut(),
                1..=high_water,
                label,
                projected_properties,
            )
        })
        .expect("has_held_txn() == true => with_held_txn_mut yields Some")
    }

    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        let tenant = ctx.tenant();
        let crud = self.crud_for(tenant)?;
        if !ctx.has_held_txn() {
            let tx = self.txn_manager.begin(tenant);
            return self.node_by_id_in_tx(tenant, &crud, &tx, id);
        }

        ctx.with_held_txn_mut(|h| {
            let owned = h
                .as_any_mut()
                .downcast_mut::<BoltHeldTxn>()
                .ok_or_else(|| {
                    SubstrateAccessError::Io(
                        "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                    )
                })?
                .owned_mut()?;
            self.node_by_id_in_tx(tenant, &crud, owned.txn_mut(), id)
        })
        .expect("has_held_txn() == true => with_held_txn_mut yields Some")
    }

    fn property_index_lookup_with_context(
        &self,
        ctx: &ExecutionContext,
        label: LabelId,
        property: &str,
        value: &Value,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let tenant = ctx.tenant();
        let crud = self.crud_for(tenant)?;
        // Resolve the property-index manager. `lookup_candidates` enforces
        // the RC-6 planner-visible gate (Online-only) + returns
        // CANDIDATE-ONLY NodeIds (may be stale / dup / snapshot-invisible).
        let manager = self.property_index_manager(&crud)?;
        // Candidate-then-verify + dedup by NodeId. Each candidate is
        // hydrated through ONE validated MVCC transaction, then
        // re-checked: the LIVE node must
        // still carry `label` AND `property == value` under engine `=`
        // coercion. The index NEVER determines visibility — a candidate
        // that fails hydration (tombstoned / invisible) or the recheck
        // (stale ghost / hash collision) is DROPPED, never surfaced.
        if !ctx.has_held_txn() {
            let tx = self.begin_read_txn(tenant, read_lsn)?;
            let candidates = manager
                .lookup_candidates(tenant, label, property, value)
                .map_err(|error| {
                    SubstrateAccessError::Io(format!(
                        "property-index lookup_candidates failed: {error}"
                    ))
                })?;
            return self.verify_property_candidates_in_tx(
                tenant, &crud, &tx, candidates, label, property, value,
            );
        }

        ctx.with_held_txn_mut(|handle| {
            let owned = handle
                .as_any_mut()
                .downcast_mut::<BoltHeldTxn>()
                .ok_or_else(|| {
                    SubstrateAccessError::Io(
                        "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                    )
                })?
                .owned_mut()?;
            Self::resolve_read_snapshot(read_lsn, owned.snapshot())?;
            let candidates = manager
                .lookup_candidates(tenant, label, property, value)
                .map_err(|error| {
                    SubstrateAccessError::Io(format!(
                        "property-index lookup_candidates failed: {error}"
                    ))
                })?;
            self.verify_property_candidates_in_tx(
                tenant,
                &crud,
                owned.txn_mut(),
                candidates,
                label,
                property,
                value,
            )
        })
        .expect("has_held_txn() == true => with_held_txn_mut yields Some")
    }

    /// **#1366 (Phase 2) — the op's index-vs-scan-fallback gate (#1415).**
    /// A resolved runtime value is index-keyable iff production
    /// `canonical_key_for` yields a key for it. When it does NOT (a
    /// fractional / out-of-i64-range `Float`, a NEGATIVE `Integer`, a
    /// `List` / `Map`), `lookup_candidates` would return EMPTY and the op
    /// must fall back to a full Scan+Filter instead of dropping rows. This
    /// mirrors the EXACT `None` set the lookup path uses (both route
    /// through `canonical_key_for`), so a keyable value never scans (no
    /// perf regression) and an unkeyable value never silently returns
    /// fewer rows than a scan.
    fn value_is_indexable(&self, value: &Value) -> bool {
        crate::storage::property_index::canonical_key_for(value).is_some()
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        // W26-β-2 / ADR-131 — full directional support.
        //
        // - `Direction::LeftToRight`: walk the forward TEL chain at
        //   `(tenant, src=from, channel)` via `crud::scan_out`.
        // - `Direction::RightToLeft`: walk the REVERSE TEL chain at
        //   `(tenant, dst=from, channel)` via `crud::scan_in`. Each
        //   reverse entry's `dst_id` field holds the ORIGINAL SRC
        //   (the neighbor of `from` on the other end of the edge).
        // - `Direction::Undirected`: union outbound + inbound walks,
        //   deduplicated by `RelId` so a self-loop (src == dst ==
        //   from) appears exactly once. The `BoundEdge.dst` (= "far
        //   end of traversal") is whichever endpoint is NOT `from`;
        //   for self-loops the far end IS `from`.
        //
        // When [`CrudStore::reverse_index_enabled`] is `false`,
        // `crud::scan_in` returns
        // `Err(ScanInError::ReverseIndexDisabled)` which translates
        // to a structured
        // [`SubstrateAccessError::IndexUnavailable("reverse-adjacency")`]
        // per AC-4 fault-injection discipline
        // (`feedback_load_bearing_pr_requires_fault_injection_tests.md`):
        // operators MUST see a structured error, never silent-empty
        // results.
        let crud = self.crud_for(tenant)?;
        let tx = self.begin_read_txn(tenant, read_lsn)?;

        // Collect (rel_id, src, dst, far_end_id) tuples deduplicated
        // by rel_id; the `BoundEdge` materialization step reads node
        // labels once per unique far-end node.
        let mut edges: Vec<EdgePending> = Vec::new();
        match direction {
            Direction::LeftToRight => {
                for entry in crud::scan_out(&crud, &tx, from, rel_type) {
                    let dst_id = NodeId::new(entry.dst_id);
                    edges.push(EdgePending {
                        rel_id: entry.rel_id,
                        src: from,
                        dst: dst_id,
                        // Far end of traversal = the destination.
                        far_end: dst_id,
                    });
                }
            }
            Direction::RightToLeft => {
                let inbound = crud::scan_in(&crud, &tx, from, rel_type).map_err(|e| {
                    SubstrateAccessError::IndexUnavailable(format!("reverse-adjacency: {e}"))
                })?;
                for entry in inbound {
                    // Reverse entry's `dst_id` = original src.
                    let src_id = NodeId::new(entry.dst_id);
                    edges.push(EdgePending {
                        rel_id: entry.rel_id,
                        src: src_id,
                        dst: from,
                        // Far end of traversal = the source.
                        far_end: src_id,
                    });
                }
            }
            Direction::Undirected => {
                // Outbound walk — far end = dst.
                for entry in crud::scan_out(&crud, &tx, from, rel_type) {
                    let dst_id = NodeId::new(entry.dst_id);
                    edges.push(EdgePending {
                        rel_id: entry.rel_id,
                        src: from,
                        dst: dst_id,
                        far_end: dst_id,
                    });
                }
                // Inbound walk — far end = src. Errors from the
                // reverse index are surfaced (structured, not
                // silent-empty) per AC-4.
                let inbound = crud::scan_in(&crud, &tx, from, rel_type).map_err(|e| {
                    SubstrateAccessError::IndexUnavailable(format!("reverse-adjacency: {e}"))
                })?;
                for entry in inbound {
                    let src_id = NodeId::new(entry.dst_id);
                    edges.push(EdgePending {
                        rel_id: entry.rel_id,
                        src: src_id,
                        dst: from,
                        far_end: src_id,
                    });
                }
                // Dedup by rel_id (self-loop guard). Stable order:
                // sort by rel_id ascending, then keep first
                // occurrence. The substrate contract does not pin a
                // specific order, but ascending rel_id is the
                // canonical AC-1/AC-2 oracle.
                edges.sort_by_key(|e| e.rel_id);
                edges.dedup_by_key(|e| e.rel_id);
            }
        }

        let mut out: Vec<BoundEdge> = Vec::with_capacity(edges.len());
        for ep in edges {
            if let Some(edge) =
                materialize_bound_edge(&crud, &self.intern_table, tenant, &tx, ep, rel_type)?
            {
                out.push(edge);
            }
        }
        let _ = tx;
        Ok(out)
    }

    fn expand_cursor(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        let owned = self.begin_owned_read_txn(tenant, read_lsn)?;
        let phase = match direction {
            Direction::LeftToRight => {
                CrudExpandPhase::Out(ScanOutCursor::new(&crud, owned.txn(), from, rel_type))
            }
            Direction::RightToLeft => {
                let inbound =
                    ScanInCursor::new(&crud, owned.txn(), from, rel_type).map_err(|e| {
                        SubstrateAccessError::IndexUnavailable(format!("reverse-adjacency: {e}"))
                    })?;
                CrudExpandPhase::In(inbound)
            }
            Direction::Undirected => {
                let inbound =
                    ScanInCursor::new(&crud, owned.txn(), from, rel_type).map_err(|e| {
                        SubstrateAccessError::IndexUnavailable(format!("reverse-adjacency: {e}"))
                    })?;
                CrudExpandPhase::Undirected {
                    out: ScanOutCursor::new(&crud, owned.txn(), from, rel_type),
                    in_: inbound,
                    draining_out: true,
                }
            }
        };
        Ok(Box::new(CrudExpandCursor {
            crud,
            intern: Arc::clone(&self.intern_table),
            owned,
            tenant,
            rel_type,
            phase,
            from,
        }))
    }

    fn expand_cursor_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<BoundEdgeCursor, SubstrateAccessError> {
        if !ctx.has_held_txn() {
            return self.expand_cursor(ctx.tenant(), from, rel_type, direction, read_lsn);
        }

        let tenant = ctx.tenant();
        let crud = self.crud_for(tenant)?;
        let held = ctx.held_txn_access();
        match held.with_mut(|handle| {
            handle
                .as_any_mut()
                .downcast_mut::<BoltHeldTxn>()
                .ok_or_else(|| {
                    SubstrateAccessError::Io(
                        "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                    )
                })?
                .owned_mut()
                .and_then(|owned| Self::resolve_read_snapshot(read_lsn, owned.snapshot()))
                .map(|_| ())
        }) {
            Some(result) => result?,
            None => {
                return Err(SubstrateAccessError::Io(
                    "held transaction disappeared before expand cursor opened".into(),
                ));
            }
        }

        let high_water = crud.rel_high_water(tenant);
        Ok(Box::new(HeldCrudExpandCursor {
            crud,
            intern: Arc::clone(&self.intern_table),
            held,
            tenant,
            from,
            rel_type,
            direction,
            next_raw: 1,
            high_water,
            finished: false,
        }))
    }

    fn expand_with_context(
        &self,
        ctx: &ExecutionContext,
        from: NodeId,
        rel_type: Option<TypeId>,
        direction: Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
        if !ctx.has_held_txn() {
            return self.expand(ctx.tenant(), from, rel_type, direction, read_lsn);
        }
        let tenant = ctx.tenant();
        let crud = self.crud_for(tenant)?;
        ctx.with_held_txn_mut(|h| {
            let owned = h
                .as_any_mut()
                .downcast_mut::<BoltHeldTxn>()
                .ok_or_else(|| {
                    SubstrateAccessError::Io(
                        "held-txn handle is not a BoltHeldTxn (ADR-197 downcast)".into(),
                    )
                })?
                .owned_mut()?;
            Self::resolve_read_snapshot(read_lsn, owned.snapshot())?;
            let tx = owned.txn_mut();

            let mut edges: Vec<EdgePending> = Vec::new();
            for raw in 1..=crud.rel_high_water(tenant) {
                let rel_id = RelId::new(raw);
                let rec = match crud::read_rel(&*tx, rel_id) {
                    Ok(Some(rec)) => rec,
                    Ok(None) => continue,
                    Err(e) => {
                        return Err(SubstrateAccessError::Io(format!(
                            "expand: read_rel({raw}) failed: {e}"
                        )));
                    }
                };
                if !rel_matches_type(&rec, rel_type) {
                    continue;
                }
                let src = NodeId::new(rec.src_id);
                let dst = NodeId::new(rec.dst_id);
                match direction {
                    Direction::LeftToRight if src == from => edges.push(EdgePending {
                        rel_id: raw,
                        src,
                        dst,
                        far_end: dst,
                    }),
                    Direction::RightToLeft if dst == from => edges.push(EdgePending {
                        rel_id: raw,
                        src,
                        dst,
                        far_end: src,
                    }),
                    Direction::Undirected if src == from || dst == from => {
                        let far_end = if src == from { dst } else { src };
                        edges.push(EdgePending {
                            rel_id: raw,
                            src,
                            dst,
                            far_end,
                        });
                    }
                    _ => {}
                }
            }
            edges.sort_by_key(|e| e.rel_id);
            edges.dedup_by_key(|e| e.rel_id);

            let blobs = crud.blob_store();
            let mut out: Vec<BoundEdge> = Vec::with_capacity(edges.len());
            for ep in edges {
                let (far_label, far_label_name, far_props) =
                    match crud::read_node_with_store(&crud, &*tx, ep.far_end) {
                        Ok(Some(r)) => {
                            let label = if r.label_id == 0 {
                                None
                            } else {
                                Some(LabelId::new(r.label_id))
                            };
                            let label_name = self.resolve_label_name(tenant, r.label_id);
                            let props =
                                crate::storage::property_payload::record_property_bag_checked(
                                    &r,
                                    blobs,
                                    &self.intern_table,
                                    tenant,
                                )?;
                            (label, label_name, props)
                        }
                        Ok(None) => (None, None, std::collections::BTreeMap::new()),
                        Err(e) => {
                            return Err(SubstrateAccessError::Io(format!(
                                "expand: read_node(far_end={}) failed: {e}",
                                ep.far_end.raw()
                            )));
                        }
                    };
                let rel_id_typed = arcgraph_core::RelId::new(ep.rel_id);
                let (rel_type_name, rel_props) =
                    match crud::read_rel_with_store(&crud, &*tx, rel_id_typed) {
                        Ok(Some(r)) => {
                            let name = self.resolve_rel_type_name(tenant, r.type_id);
                            let props =
                                crate::storage::property_payload::rel_record_property_bag_checked(
                                    &r,
                                    blobs,
                                    &self.intern_table,
                                    tenant,
                                )?;
                            (name, props)
                        }
                        Ok(None) | Err(_) => (None, std::collections::BTreeMap::new()),
                    };
                let rel = RelView {
                    id: rel_id_typed,
                    from: ep.src,
                    to: ep.dst,
                    rel_type,
                    rel_type_name,
                    properties: rel_props,
                };
                let dst_node = NodeView {
                    id: ep.far_end,
                    label: far_label,
                    label_name: far_label_name,
                    properties: far_props,
                };
                out.push(BoundEdge { rel, dst: dst_node });
            }
            Ok(out)
        })
        .expect("has_held_txn() == true => with_held_txn_mut yields Some")
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        // Step 1: verify the router has a vector handle attached
        // for this tenant. Preserved from the W17α posture —
        // unwired tenants continue to see structured
        // `IndexUnavailable("vector")` per AC-6 (W23-M4-08-FINALIZE
        // ADR-087 §"What this locks in" contract is binding through
        // v1.0-GA for unwired substrates).
        let handle = self
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| translate_routing_error(e, tenant))?;
        if handle.vector().is_none() {
            return Err(SubstrateAccessError::IndexUnavailable("vector".into()));
        }
        // Step 2: dispatch through the SubstrateSearchProvider.
        // When a substrate is router-attached but no provider is
        // bound (= production bootstrap deferred / test fixture
        // builds without it), surface a structured error rather
        // than silent-empty results per
        // `feedback_review_oracle_relaxations.md`. The error message
        // names the binder so a misconfigured deployment surfaces
        // the resolution path immediately.
        let Some(provider) = self.search_provider.as_ref() else {
            return Err(SubstrateAccessError::IndexUnavailable(
                "vector search provider not attached \
                 (call CrudExecutorSubstrate::with_search_provider \
                 at process bootstrap; ADR-132 D-3)"
                    .into(),
            ));
        };
        // Resolve read-latest once, or reject an unavailable finite
        // snapshot. The provider and record hydration below share this
        // exact effective LSN.
        let tx = self.begin_read_txn(tenant, read_lsn)?;
        let effective_lsn = tx.snapshot();
        let mut hits = provider.vector_search(tenant, property, query_vec, k, effective_lsn)?;

        // #830 D4 — hydrate stored properties + label name onto each
        // ranked hit's node. The served HNSW provider
        // (`arcgraph_cli::vector_search`) builds every `RankedHit` from
        // only the resident `(node_id, label)` sidecar, so its
        // `NodeView` carries an EMPTY property bag + an unresolved label
        // name (`NodeView::new`). Without this overlay,
        // `CALL db.index.vector.queryNodes(...) YIELD node` returns nodes
        // whose `node.text` / `node{.*}` are `None` / `{}` and whose
        // `labels(n)` leaks `"LabelId(1)"` — so langchain `Neo4jVector`
        // RAG retrieves documents with no content/metadata (the #830 D4
        // residual). Re-read the record store by node-id and rebuild each
        // node's view through the SAME idiom the MATCH path uses
        // ([`Self::hydrate_node_view`] — also called by
        // [`Self::scan_id_range`]), so a `queryNodes`-returned node is
        // shaped IDENTICALLY to a MATCH-returned node.
        //
        // STRONG-ORACLE INVARIANT: this overlay changes ONLY each node's
        // `properties` + `label_name`. It NEVER changes the hit's `id`,
        // `score`, rank ORDER, or count — the `hits` vec is iterated
        // in-place (no re-sort, no filter, no push/pop). The
        // `debug_assert_eq!` below pins the id-preservation half.
        //
        // SNAPSHOT: hydration reuses the transaction that resolved the
        // provider's effective LSN. Provider visibility and properties
        // therefore cannot silently come from different snapshots.
        //
        // EDGE RACE: an `Ok(None)` (tombstoned / not visible at this
        // snapshot) is left as the un-hydrated `(id, label)` node rather
        // than DROPPED — dropping a ranked hit would silently change the
        // result count + rank order, which the strong-oracle invariant
        // forbids (per `feedback_review_oracle_relaxations.md`). The
        // provider already applied MVCC visibility, so an `Ok(None)`
        // here is a narrow read-skew window, not a routine miss.
        let crud = self.crud_for(tenant)?;
        for hit in &mut hits {
            match crud::read_node(&tx, hit.node.id) {
                Ok(Some(rec)) => {
                    debug_assert_eq!(
                        rec.id,
                        hit.node.id.raw(),
                        "read_node(id) must return the record AT that id"
                    );
                    hit.node = self.hydrate_node_view(tenant, &crud, &rec)?;
                }
                Ok(None) => {
                    // Edge race (see above) — keep the un-hydrated node.
                }
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "vector_search hydrate: read_node({}) failed: {e}",
                        hit.node.id.raw()
                    )));
                }
            }
        }
        let _ = tx; // reads complete; tx drops here.
        Ok(hits)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        // Same shape + escalation as `vector_search` above. The
        // router attachment gate + provider binding gate are
        // STRUCTURALLY identical so a regression on one is
        // load-bearing for the other.
        let handle = self
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| translate_routing_error(e, tenant))?;
        if handle.bm25().is_none() {
            return Err(SubstrateAccessError::IndexUnavailable("bm25".into()));
        }
        let Some(provider) = self.search_provider.as_ref() else {
            return Err(SubstrateAccessError::IndexUnavailable(
                "bm25 search provider not attached \
                 (call CrudExecutorSubstrate::with_search_provider \
                 at process bootstrap; ADR-132 D-3)"
                    .into(),
            ));
        };
        // Resolve read-latest once, or reject an unavailable finite
        // snapshot. The provider and record hydration below share this
        // exact effective LSN.
        let tx = self.begin_read_txn(tenant, read_lsn)?;
        let effective_lsn = tx.snapshot();
        let mut hits = provider.bm25_search(tenant, property, query_text, k, effective_lsn)?;

        // #903 mirrors #890's vector post-hydration: the BM25 provider
        // returns ranked `(id, label)` sidecars, then the substrate
        // re-reads each live node and materializes it through the shared
        // MATCH/queryNodes idiom so stored properties + label names are
        // present without changing id, score, rank order, or count. The
        // hydration point-reads reuse the same validated transaction as
        // the provider's effective LSN.
        let crud = self.crud_for(tenant)?;
        for hit in &mut hits {
            match crud::read_node(&tx, hit.node.id) {
                Ok(Some(rec)) => {
                    debug_assert_eq!(
                        rec.id,
                        hit.node.id.raw(),
                        "read_node(id) must return the record AT that id"
                    );
                    hit.node = self.hydrate_node_view(tenant, &crud, &rec)?;
                }
                Ok(None) => {
                    // Tombstone / edge race: keep the un-hydrated ranked row.
                }
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "bm25_search hydrate: read_node({}) failed: {e}",
                        hit.node.id.raw()
                    )));
                }
            }
        }
        let _ = tx; // reads complete; tx drops here.
        Ok(hits)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        _community_id: i64,
        _read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let handle = self
            .router
            .route(tenant, PartitionId::ZERO)
            .map_err(|e| translate_routing_error(e, tenant))?;
        if handle.community().is_none() {
            return Err(SubstrateAccessError::IndexUnavailable("community".into()));
        }
        Err(SubstrateAccessError::IndexUnavailable(
            "community membership lookup body not yet wired (W17α + v1.1 forward)".into(),
        ))
    }

    fn has_vector_substrate(&self) -> bool {
        // We can answer this only when given a tenant; the trait's
        // current shape is tenant-free for the availability query.
        // v1.0 deployments wire substrates at the router level, so
        // an attached vector on ANY tenant implies the substrate is
        // available process-wide. v1.1 makes this per-tenant.
        // We default to false; impls that need per-tenant accuracy
        // bypass this method and route through the
        // [`crate::tools::search::HybridSearcher::available_substrates`]
        // surface (which DOES carry tenant).
        false
    }

    fn has_bm25_substrate(&self) -> bool {
        false
    }

    fn has_community_substrate(&self) -> bool {
        false
    }

    // ── D-2 (ADR-147 §D-8 / W26-θ Phase 5) — statement-scoped ──
    //    autocommit transaction (begin-once → stage → commit-once)
    //
    // The executor (arcgraph-query `materialize`) calls `begin_statement`
    // at the start of an AUTO-COMMIT write statement, then `commit_statement`
    // (success) or `rollback_statement` (failure) at the end. Between them
    // every `create_node` / `create_rel` / `delete_*` / `set_*` call sees a
    // held txn on `ctx` and STAGES into it via the EXPLICIT branch of
    // `stage_or_commit` / `run_txn` — the SAME staging the Bolt BEGIN…COMMIT
    // path uses. This collapses a `CREATE (a)-[:R]->(b)` spine's 3 durable
    // commits (create_node ×2 + create_rel ×1) into ONE CommitBundle / fsync
    // AND makes the statement atomic (a mid-spine fault rolls back the whole
    // spine — no partial 2-of-3). The commit reuses `commit_bolt_held_txn`
    // so the #963 HNSW + BM25 maintenance hooks fire ONCE at statement
    // commit (queued while the spine staged), identical to the explicit-tx
    // COMMIT semantics.

    fn begin_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        // Defense-in-depth: the executor only calls this when
        // `!ctx.has_held_txn()` (D-2 must not nest inside a Bolt
        // BEGIN…COMMIT). Re-assert here so a future caller that mis-wires
        // the guard cannot silently clobber an explicit transaction.
        if ctx.has_held_txn() {
            return Err(SubstrateAccessError::Io(
                "begin_statement called with a held txn already installed \
                 (D-2 must not nest inside an explicit Bolt transaction)"
                    .into(),
            ));
        }
        // Open ONE owned MVCC transaction the statement holds across all
        // its write ops; install it as a BoltHeldTxn so the write ops'
        // EXPLICIT staging branch (and the held-txn read path for a
        // `MATCH … CREATE` spine) route through it. Its snapshot LSN is
        // captured at BEGIN and seeded onto `ctx` (read-your-writes).
        let crud = self.crud_for(ctx.tenant())?;
        let owned = self.txn_manager.begin_owned(ctx.tenant());
        ctx.install_held_txn(Box::new(BoltHeldTxn::new_with_abort_store(owned, crud)));
        Ok(())
    }

    fn commit_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
        // Reclaim the statement's held txn and commit it through the SAME
        // full machinery the Bolt COMMIT uses: `commit_bolt_held_txn` drains
        // the buffered primary-index installs + WAL CommitBundle under ONE
        // fsync, THEN fires the queued #963 HNSW +
        // BM25 hooks ONCE (post-commit). If no held txn is present the
        // statement staged nothing durable (defensive no-op).
        let Some(held) = ctx.take_held_txn() else {
            return Ok(());
        };
        // Downcast back to the concrete BoltHeldTxn (the only handle
        // `begin_statement` installs) to reach `commit_bolt_held_txn`,
        // which owns the hook-firing order.
        self.commit_bolt_held_handle(held).map(|_lsn| ())
    }

    fn rollback_statement(&self, ctx: &ExecutionContext) {
        // Reclaim + abort the statement's held txn — discards every staged
        // write of the spine (the real ROLLBACK). The queued HNSW / BM25
        // hooks are dropped WITH the handle (no index maintenance fires for
        // an aborted statement). Aborting the OwnedTxn cannot fail; if the
        // downcast/move-out somehow fails, dropping the box still aborts via
        // OwnedTxn's Drop (no leak). Mirrors the ADR-197 Bolt ROLLBACK path.
        if let Some(mut held) = ctx.take_held_txn() {
            if let Some(bolt) = held.as_any_mut().downcast_mut::<BoltHeldTxn>() {
                if let Some(owned) = bolt.take_owned() {
                    owned.abort();
                }
            }
            // `held` drops here — any not-yet-taken OwnedTxn aborts via Drop.
        }
    }

    // ADR-147 W26-θ Phase 1 — CREATE node production wire-through.
    //
    // Opens a per-tenant `Transaction` per ADR-031 + ADR-033, calls
    // `arcgraph_storage::crud::create_node` with the interned label
    // (resolved via `InternTable::intern_label`), then commits via
    // `arcgraph_storage::crud::commit`. The commit returns the
    // assigned `Lsn`; the new node-id is returned to the executor.
    //
    // # Property handling
    //
    // Per ADR-152 §D-1 the `properties` slice IS persisted: a non-empty
    // bag serializes to canonical JSON via `properties_to_property_data`
    // and routes into `PropertyData::Blob` (empty → `PropertyData::Empty`
    // fast-path). The v1.2 strict-schema property *typing*
    // (catalog-bound `PropertyId`) remains forward-pinned to issue #356.
    //
    // # Transaction discipline (Phase 1 → D-2)
    //
    // D-2 (ADR-147 §D-8 / W26-θ Phase 5) LANDED the statement-scoped tx:
    // in AUTO-COMMIT mode the executor's `materialize` calls
    // `begin_statement` before the write pipeline, so this `create_node`
    // sees a held txn on `ctx` and STAGES into it (via `stage_or_commit`'s
    // EXPLICIT branch) rather than begin→create_node→commit per call. The
    // whole statement's spine (every node + rel) commits ONCE at
    // `commit_statement`. The bare one-call-one-tx path below survives only
    // for the degenerate "no statement txn open" case (a direct substrate
    // caller outside the executor); the executor path always wraps.
    //
    // # Error semantics
    //
    // Any failure (routing miss / store-side error / commit
    // failure) returns `Err(_)` without partial side effect. The
    // `crud::commit` consumes the transaction; on error it surfaces
    // via the `crud::commit` -> `CrudError` return — the executor
    // propagates this through `ExecutionError::Substrate`.
    fn create_node(
        &self,
        tenant: TenantId,
        label: Option<&str>,
        properties: &[(String, Value)],
        ctx: &ExecutionContext,
    ) -> Result<NodeId, SubstrateAccessError> {
        // Step 1: resolve the per-tenant CrudStore handle.
        let crud = self.crud_for(tenant)?;

        // Step 2: intern the label name (if any) via the per-process
        // InternTable. Returns a stable LabelId across calls; first
        // call for a given (tenant, name) allocates a fresh id.
        //
        // P0 #776: WAL-log the binding when the name is freshly
        // allocated AND the store is durable (`crud.wal()` is `Some`), so
        // a label name first allocated on this logged create path survives
        // a `--data` restart. The InternString record is appended
        // (Strict-tier: fsynced) BEFORE the commit below, so its LSN ≤ the
        // commit LSN — any fsync that durifies the node also durifies that
        // label name, and a log failure here aborts the create.
        //
        // This does NOT cover a name already published in-memory by an
        // unlogged path (e.g. `graph.explore`-before-create, #355): the
        // freshly-allocated (`was_new`) latch then suppresses the log and
        // the name can be lost on restart — strictly better than pre-#776
        // (all names lost), residual edge tracked in #788.
        let label_id = match label {
            Some(name) => {
                arcgraph_storage::intern_label_logged(&self.intern_table, crud.wal(), tenant, name)
                    .map_err(|e| {
                        SubstrateAccessError::Io(format!(
                            "create_node: intern WAL log failed for label {name:?}: {e}"
                        ))
                    })?
            }
            None => LabelId::new(0),
        };

        // Step 3: serialize the property bag as a v2 M2 TYPED block
        // (ADR-230 row M2 — no JSON encode on the write path; property
        // names intern WAL-logged so their key_ids are durable BEFORE
        // the commit that references them, the label-intern ordering).
        // Empty bag routes to PropertyData::Empty (fast path).
        let property_data = crate::storage::property_payload::properties_to_property_data_typed(
            properties,
            &self.intern_table,
            crud.wal(),
            tenant,
        )?;
        // Step 4+5: stage the create into the held tx (EXPLICIT mode)
        // OR begin+commit (AUTO-COMMIT mode) per ADR-197. The closure
        // stages WITHOUT committing; `stage_or_commit` owns the
        // begin/commit/discard policy.
        let node = self.stage_or_commit(tenant, ctx, &crud, move |crud, tx| {
            arcgraph_storage::crud::create_node(crud, tx, tenant, label_id, &property_data)
        })?;
        // #1366 (task #248): maintain declared property indexes for the
        // new node (write-follows-declare; a node created while an index
        // is Building/Online is inserted). Best-effort. The new bag IS
        // the in-hand `properties`; there is no old bag on a create.
        let new_bag: std::collections::BTreeMap<String, Value> =
            properties.iter().cloned().collect();
        self.maintain_property_index_best_effort(tenant, node, label_id, None, &new_bag);
        Ok(node)
    }

    // NN-4 (#1384) — MERGE get-or-create serialization. Hands the
    // executor's `MergeOp` an OWNING per-(tenant, key) mutex guard that
    // it holds across the match→create span, so two concurrent `MERGE`
    // on the SAME key cannot both create (the SI + OCC double-create
    // hole). See the `merge_locks` field doc for the lock-order argument.
    //
    // # Two-level table access (no self-deadlock)
    //
    // The OUTER `merge_locks` latch is held ONLY to look up / lazily mint
    // the inner per-key `Arc<Mutex<()>>`, then RELEASED (the `entry` scope
    // ends) BEFORE the inner mutex is acquired. This is essential: holding
    // the outer latch while blocking on the inner mutex would serialize
    // EVERY merge tenant-wide (and could deadlock a second racer trying to
    // mint a DIFFERENT key). We clone the inner `Arc` out, drop the outer
    // latch, THEN block on the inner mutex.
    //
    // # Blocking
    //
    // `lock_arc()` BLOCKS the calling thread until the inner mutex is
    // free. The loser therefore parks here until the winner's `MergeOp`
    // drops its guard (after the create has committed); it then returns
    // and `MergeOp` RE-PROBES the match branch (seeing the winner's node)
    // — the pessimistic get-or-create contract. The served executor runs
    // per-connection on the Tokio/blocking pool, so a blocked MERGE
    // parks its own worker thread without starving unrelated work.
    fn merge_guard(
        &self,
        tenant: TenantId,
        key: &str,
    ) -> Result<Option<Box<dyn MergeGuard>>, SubstrateAccessError> {
        // Phase A — look up / mint the per-key inner mutex under the
        // SHORT-lived outer latch, then RELEASE the latch (scope ends).
        let inner: Arc<Mutex<()>> = {
            let mut table = self.merge_locks.lock();
            Arc::clone(
                table
                    .entry((tenant, key.to_owned()))
                    .or_insert_with(|| Arc::new(Mutex::new(()))),
            )
        };
        // Phase B — BLOCK on the inner per-key mutex (outer latch already
        // dropped). `lock_arc` returns an owning guard that keeps `inner`
        // alive for as long as it is held.
        let guard = inner.lock_arc();
        Ok(Some(Box::new(CrudMergeGuard { _guard: guard })))
    }

    // ADR-148 W26-θ Phase 2 — CREATE-rel production wire-through.
    //
    // Opens a per-tenant `Transaction` per ADR-031 + ADR-033, calls
    // `arcgraph_storage::crud::create_rel` with the interned rel-type
    // (resolved via `InternTable::intern_type`), then commits via
    // `arcgraph_storage::crud::commit`. The commit returns the
    // assigned `Lsn`; the new rel-id is returned to the executor.
    //
    // # Direction handling at Phase 2
    //
    // The executor's `CreateRelOp` canonicalizes (source → target)
    // wire order BEFORE calling this method (RightToLeft AST direction
    // swaps source/target at the executor boundary). This impl
    // ALWAYS sees source-to-target wire order; the storage record's
    // `from` = source, `to` = target.
    //
    // # Property handling at Phase 2
    //
    // Per ADR-152 §D-1 the `properties` slice IS persisted (symmetric to
    // `create_node`): non-empty → `PropertyData::Blob` (canonical JSON);
    // empty → `PropertyData::Empty`. The v1.2 strict-schema property
    // *typing* remains forward-pinned to issue #356.
    //
    // # Transaction discipline (Phase 2 → D-2)
    //
    // D-2 (ADR-147 §D-8 / W26-θ Phase 5) LANDED the statement-scoped tx
    // (see `create_node` above): in AUTO-COMMIT mode the executor opens ONE
    // statement txn via `begin_statement`, so this `create_rel` STAGES into
    // the held txn alongside its spine's `create_node`s and the whole
    // statement commits ONCE at `commit_statement`. The bare
    // one-call-one-tx path below survives only for a direct substrate
    // caller outside the executor.
    //
    // # Error semantics
    //
    // Any failure (routing miss / store-side error / commit failure)
    // returns `Err(_)` without partial side effect.
    fn create_rel(
        &self,
        tenant: TenantId,
        source: NodeId,
        target: NodeId,
        label: &str,
        properties: &[(String, Value)],
        ctx: &ExecutionContext,
    ) -> Result<RelId, SubstrateAccessError> {
        // Step 1: resolve the per-tenant CrudStore handle.
        let crud = self.crud_for(tenant)?;

        // Step 2: intern the rel-type name via the per-process
        // InternTable. Returns a stable TypeId across calls; first
        // call for a given (tenant, name) allocates a fresh id.
        //
        // P0 #776: WAL-log the binding when freshly allocated AND
        // durable, so a rel-type name first allocated on this logged path
        // survives a `--data` restart (symmetric to `create_node`; see that
        // method for the durable-before-commit ordering contract and the
        // unlogged-prior-publish residual edge tracked in #788).
        let type_id =
            arcgraph_storage::intern_type_logged(&self.intern_table, crud.wal(), tenant, label)
                .map_err(|e| {
                    SubstrateAccessError::Io(format!(
                        "create_rel: intern WAL log failed for rel-type {label:?}: {e}"
                    ))
                })?;

        // Step 3: serialize the property bag as a v2 M2 TYPED block
        // (ADR-230 row M2; see create_node's Step-3 note). Empty bag →
        // PropertyData::Empty.
        let property_data = crate::storage::property_payload::properties_to_property_data_typed(
            properties,
            &self.intern_table,
            crud.wal(),
            tenant,
        )?;
        // Step 4+5: stage (EXPLICIT) or begin+commit (AUTO-COMMIT) per
        // ADR-197 — symmetric with create_node.
        self.stage_or_commit(tenant, ctx, &crud, move |crud, tx| {
            arcgraph_storage::crud::create_rel(
                crud,
                tx,
                tenant,
                source,
                target,
                type_id,
                &property_data,
            )
        })
    }

    // ADR-149 W26-θ Phase 3 — DELETE-node production wire-through.
    //
    // Per ADR-149 §D-7:
    //
    // 1. Resolve per-tenant CrudStore.
    // 2. Open a per-tenant Transaction (ADR-031 + ADR-033).
    // 3. When DETACH=true: walk `scan_out` + `scan_in` to enumerate
    //    attached rels; call `delete_rel_with_store` per attached rel.
    // 4. When DETACH=false AND any attached rel exists: surface
    //    `SubstrateAccessError::Io("delete_node: node has
    //    relationships attached; use DETACH DELETE")` — the executor
    //    maps this to ExecutionError::Eval per ADR-149 §D-1. The
    //    message contains the canonical openCypher v9 §6 contract
    //    substring "relationships attached" (ADR-149 §D-7).
    // 5. Call `delete_node_with_store` (stages MVCC tombstone +
    //    primary-index dual-write).
    // 6. Commit.
    //
    // Transaction discipline (Phase 3): one-call-one-transaction per
    // ADR-149 §D-8. The rel-deletes + node-delete all stage into the
    // SAME transaction (atomicity within a single delete_node call).
    fn delete_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        detach: bool,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        // ADR-197: all reads + cascade-rel-deletes + node-delete stage
        // into ONE transaction (the held tx in EXPLICIT mode; a fresh
        // begin+commit in AUTO-COMMIT mode). Multi-op atomicity within
        // a single delete_node is preserved in both modes.
        self.run_txn(tenant, ctx, |crud, tx| {
            // Enumerate attached rels via scan_out (outbound) + scan_in
            // (inbound; reverse-adjacency index per ADR-131). scan_in
            // returns ScanInError::ReverseIndexDisabled when the
            // reverse-index is disabled at the store level — Phase 3
            // gracefully handles this by treating "scan_in unavailable"
            // as "no inbound rels visible".
            let outbound: Vec<RelId> = arcgraph_storage::crud::scan_out(crud, tx, node, None)
                .map(|entry| RelId::new(entry.rel_id))
                .collect();
            let inbound: Vec<RelId> = match arcgraph_storage::crud::scan_in(crud, tx, node, None) {
                Ok(entries) => entries
                    .into_iter()
                    .map(|entry| RelId::new(entry.rel_id))
                    .collect(),
                Err(_) => Vec::new(),
            };
            let mut attached: Vec<RelId> = outbound;
            for r in inbound {
                if !attached.contains(&r) {
                    attached.push(r);
                }
            }
            if !attached.is_empty() {
                if !detach {
                    // Validation error (non-CrudError) — surfaced
                    // unwrapped. run_txn discards the partial tx
                    // (auto-commit) or leaves it for the handler to
                    // abort (explicit).
                    return Err(SubstrateAccessError::Io(
                        "delete_node: node has relationships attached; use DETACH DELETE".into(),
                    ));
                }
                // DETACH=true: tombstone each attached rel first within
                // the same transaction.
                for rel_id in &attached {
                    arcgraph_storage::crud::delete_rel_with_store(crud, tx, *rel_id).map_err(
                        |e| {
                            SubstrateAccessError::Io(format!(
                                "delete_node: cascade rel-delete failed: {e}"
                            ))
                        },
                    )?;
                }
            }
            // Tombstone the node itself.
            arcgraph_storage::crud::delete_node_with_store(crud, tx, node).map_err(|e| {
                SubstrateAccessError::Io(format!("delete_node: storage rejected: {e}"))
            })
        })?;
        self.mark_or_queue_vector_node_deleted(tenant, node, ctx)?;
        self.mark_or_queue_bm25_node_deleted(tenant, node, ctx)?;
        // #1379 (MUST-CON-04) — revoke the deleted node's doc-ACL, the
        // symmetric op to the ingest write-through (`apply_live_acl_grants`
        // reaches `handle.permissions().apply_doc_acl`; DELETE reaches the
        // symmetric `revoke_doc`). WITHOUT this, the tombstoned node's
        // `doc_class` mapping survives: `is_visible(node, P)` stays true and
        // the node keeps leaking to its principals forever. `revoke_doc`
        // durifies a WAL `Revoke` via the wired `AclWalSink` (#1221 /
        // ADR-218) on the durable backend, so the revoke survives a bare
        // restart. Reached AFTER the tombstone commits (delete-then-revoke):
        // a revoke that races the tombstone only ever under-grants (the doc
        // goes invisible early), never a widen. A routing miss here can only
        // leave the doc UNCLASSIFIED-or-revoked — never widen — but the
        // tenant was just routed by `crud_for` above, so this is the same
        // "unreachable in practice" guard the ingest write-through carries.
        match self.router.route(tenant, PartitionId::ZERO) {
            Ok(handle) => handle
                .permissions()
                .revoke_doc_checked(node)
                .map_err(|error| {
                    SubstrateAccessError::Io(format!("delete_node ACL revoke: {error}"))
                })?,
            Err(e) => {
                tracing::warn!(
                    target: "arcgraph_mcp::storage::substrate",
                    tenant = ?tenant,
                    node = node.raw(),
                    error = %e,
                    "delete_node: tenant route failed post-commit; could not revoke \
                     doc-ACL (node is tombstoned so search-index marks + the \
                     search_filtered liveness gate still deny it; the ACL entry \
                     will not be re-applied)"
                );
            }
        }
        Ok(())
    }

    // ADR-149 W26-θ Phase 3 — DELETE-rel production wire-through.
    //
    // Per ADR-149 §D-7: symmetric to delete_node but without the
    // cascade walk (rels have no attached children at the storage
    // layer at v1.0-α).
    fn delete_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        self.stage_or_commit(tenant, ctx, &crud, move |crud, tx| {
            arcgraph_storage::crud::delete_rel_with_store(crud, tx, rel)
        })
    }

    // ADR-150 W26-θ Phase 4 — SET-node production wire-through.
    //
    // Per ADR-150 §D-7:
    // 1. Resolve per-tenant `CrudStore` via the `MultiTenantRouter`.
    // 2. Property mutations (Assign / Replace / Merge): open a
    //    per-tenant `Transaction` per ADR-031 + ADR-033, read the
    //    current bag, apply the mutation, and write the merged bag back
    //    via `crud::update_node` (ADR-152 §D-2 — the bag IS persisted as
    //    a `PropertyData::Blob`, NOT ignored). The v1.2 strict-schema
    //    property *typing* remains forward-pinned to issue #356.
    // 3. Label mutation (LabelAdd): the storage `update_node`
    //    primitive preserves `label_id` immutably per `crud.rs:3754`
    //    "PR #170 reviewer Finding 4". Surface
    //    `SubstrateAccessError::IndexUnavailable("...forward-pinned
    //    to v1.1...")` per ADR-150 §D-9.
    //
    // Transaction discipline (Phase 4): one-call-one-transaction per
    // ADR-150 §D-8. Each call opens + commits a per-tenant tx; multi-
    // item / multi-row SET statements open multiple transactions
    // (Phase 5 batches into one statement-scoped tx).
    fn set_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &SetNodeMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        // Short-circuit label mutation BEFORE opening a transaction
        // (no rollback needed for an unsupported surface).
        if matches!(mutation, SetNodeMutation::LabelAdd(_)) {
            return Err(SubstrateAccessError::IndexUnavailable(
                "write-op set_node label-add unavailable at v1.0-α; forward-pinned to v1.1 per \
                 ADR-150 §D-9 (requires schema-migration support per the storage layer's \
                 immutable-label invariant at crud.rs:3754 \"PR #170 reviewer Finding 4\")"
                    .into(),
            ));
        }
        // ADR-197: read-modify-write stages into the held tx (EXPLICIT)
        // or a fresh begin+commit (AUTO-COMMIT).
        // The default served convention remains update-aware even without a
        // catalog entry. Registered index properties extend that set precisely;
        // a SET on an unrelated property does not disturb a vector cache slot.
        let vector_properties = self.vector_properties_for_node_mutation(tenant, "SET")?;
        let (text_changed, node_label, old_index_bag, new_index_bag) =
            self.run_txn(tenant, ctx, |crud, tx| {
                // ADR-152 §D-2 — read current bag, apply mutation, write
                // back. Property mutations route through `update_node` with
                // the merged JSON-blob payload.
                // #1366 R1 NIT-1: capture the REAL label_id from THIS first
                // read (the found path), BEFORE `r` is consumed. The prior
                // code did a redundant SECOND `read_node(...).unwrap_or(0)`
                // which, if the 2nd read raced to None, silently maintained
                // the property index at label_id=0 (the WRONG index slot).
                // Capturing here means maintenance always runs on the found
                // node's real label — never at 0.
                let (current_bag, node_label_raw) =
                    match arcgraph_storage::crud::read_node(tx, node) {
                        Ok(Some(r)) => {
                            let label_id = r.label_id;
                            (
                                crate::storage::property_payload::record_property_bag_checked(
                                    &r,
                                    crud.blob_store(),
                                    &self.intern_table,
                                    tenant,
                                )?,
                                label_id,
                            )
                        }
                        Ok(None) => {
                            return Err(SubstrateAccessError::Io(format!(
                                "set_node: node {} not visible at snapshot",
                                node.raw()
                            )));
                        }
                        Err(e) => {
                            return Err(SubstrateAccessError::Io(format!(
                                "set_node: read_node failed: {e}"
                            )));
                        }
                    };
                let old_text = indexable_string_text(&current_bag);
                // #1366 (task #248): capture the OLD bag + label BEFORE the
                // mutation consumes `current_bag`, for property-index
                // maintenance after the write.
                let old_bag_for_index = current_bag.clone();
                // #1366 R1 NIT-1: use the label captured from the FIRST read
                // (found path) — no redundant second read, no unwrap_or(0).
                let node_label = LabelId::new(node_label_raw);
                let new_bag = apply_set_node_mutation(current_bag, mutation);
                let text_changed = old_text != indexable_string_text(&new_bag);
                let props = crate::storage::property_payload::property_map_to_property_data_typed(
                    &new_bag,
                    &self.intern_table,
                    crud.wal(),
                    tenant,
                )?;
                arcgraph_storage::crud::update_node(crud, tx, node, &props).map_err(|e| {
                    SubstrateAccessError::Io(format!("set_node: storage rejected: {e}"))
                })?;
                Ok((text_changed, node_label, old_bag_for_index, new_bag))
            })?;
        let changed_vector_properties =
            changed_vector_properties(&vector_properties, &old_index_bag, &new_index_bag);
        for property in changed_vector_properties {
            self.mark_or_queue_vector_node_updated(tenant, &property, node, ctx)?;
        }
        if text_changed {
            self.mark_or_queue_bm25_node_updated(tenant, node, ctx)?;
        }
        // #1366: maintain declared property indexes (write-follows-
        // declare; insert-only). Best-effort — a maintenance failure
        // must not fail the SET (the index is a read accelerator).
        self.maintain_property_index_best_effort(
            tenant,
            node,
            node_label,
            Some(&old_index_bag),
            &new_index_bag,
        );
        Ok(())
    }

    // ADR-150 W26-θ Phase 4 — SET-rel production wire-through.
    //
    // Per ADR-150 §D-7: symmetric to set_node property mutations
    // (rels carry no labels; no LabelAdd variant in `SetRelMutation`).
    fn set_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &SetRelMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        // ADR-197: read-modify-write into the held tx (EXPLICIT) or a
        // fresh begin+commit (AUTO-COMMIT) — symmetric with set_node.
        self.run_txn(tenant, ctx, |crud, tx| {
            // ADR-152 §D-2 — same read/apply/write shape as set_node.
            let current_bag = match arcgraph_storage::crud::read_rel(tx, rel) {
                Ok(Some(r)) => crate::storage::property_payload::rel_record_property_bag_checked(
                    &r,
                    crud.blob_store(),
                    &self.intern_table,
                    tenant,
                )?,
                Ok(None) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "set_rel: rel {} not visible at snapshot",
                        rel.raw()
                    )));
                }
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "set_rel: read_rel failed: {e}"
                    )));
                }
            };
            let new_bag = apply_set_rel_mutation(current_bag, mutation);
            let props = crate::storage::property_payload::property_map_to_property_data_typed(
                &new_bag,
                &self.intern_table,
                crud.wal(),
                tenant,
            )?;
            arcgraph_storage::crud::update_rel(crud, tx, rel, &props)
                .map_err(|e| SubstrateAccessError::Io(format!("set_rel: storage rejected: {e}")))
        })
    }

    // ADR-150 W26-θ Phase 4 — REMOVE-node production wire-through.
    //
    // Property removal reads the current bag, drops the named key(s),
    // and writes the reduced bag back via `crud::update_node` (ADR-152
    // §D-2 — the reduced bag IS persisted as a `PropertyData::Blob`, or
    // `PropertyData::Empty` if the bag is now empty). The v1.2
    // strict-schema property *typing* remains forward-pinned to issue #356.
    //
    // Label removal surfaces `IndexUnavailable` per ADR-150 §D-9.
    fn remove_node(
        &self,
        tenant: TenantId,
        node: NodeId,
        mutation: &RemoveNodeMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        if matches!(mutation, RemoveNodeMutation::LabelRemove(_)) {
            return Err(SubstrateAccessError::IndexUnavailable(
                "write-op remove_node label-remove unavailable at v1.0-α; forward-pinned to \
                 v1.1 per ADR-150 §D-9 (requires schema-migration support per the storage \
                 layer's immutable-label invariant at crud.rs:3754 \"PR #170 reviewer \
                 Finding 4\")"
                    .into(),
            ));
        }
        // ADR-197: read-modify-write into the held tx (EXPLICIT) or a
        // fresh begin+commit (AUTO-COMMIT) — symmetric with set_node.
        let vector_properties = self.vector_properties_for_node_mutation(tenant, "REMOVE")?;
        let result = self.run_txn(tenant, ctx, |crud, tx| {
            // ADR-152 §D-2 — read current bag, remove key, write back.
            // #1366 R1 NIT-1: capture the REAL label_id from THIS first read
            // (found path), BEFORE `r` is consumed — replacing the redundant
            // second `read_node(...).unwrap_or(0)` that silently maintained
            // the property index at label_id=0 (wrong slot) on a 2nd-read
            // miss. Maintenance now always uses the found node's real label.
            let (current_bag, node_label_raw) = match arcgraph_storage::crud::read_node(tx, node) {
                Ok(Some(r)) => {
                    let label_id = r.label_id;
                    (
                        crate::storage::property_payload::record_property_bag_checked(
                            &r,
                            crud.blob_store(),
                            &self.intern_table,
                            tenant,
                        )?,
                        label_id,
                    )
                }
                Ok(None) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "remove_node: node {} not visible at snapshot",
                        node.raw()
                    )));
                }
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "remove_node: read_node failed: {e}"
                    )));
                }
            };
            // #1366: capture OLD bag + label before the mutation.
            let old_bag_for_index = current_bag.clone();
            // #1366 R1 NIT-1: use the label captured from the FIRST read
            // (found path) — no redundant second read, no unwrap_or(0).
            let node_label = LabelId::new(node_label_raw);
            let new_bag = apply_remove_node_mutation(current_bag, mutation);
            let props = crate::storage::property_payload::property_map_to_property_data_typed(
                &new_bag,
                &self.intern_table,
                crud.wal(),
                tenant,
            )?;
            arcgraph_storage::crud::update_node(crud, tx, node, &props).map_err(|e| {
                SubstrateAccessError::Io(format!("remove_node: storage rejected: {e}"))
            })?;
            Ok((node_label, old_bag_for_index, new_bag))
        })?;
        // #1366: maintain declared property indexes. Under insert-only
        // maintenance a REMOVE leaves the old entry as a verify-filtered
        // ghost (nothing new to insert); the call is a correct no-op when
        // the property is gone, and covers the SET-then-different-value
        // shape uniformly.
        let (node_label, old_index_bag, new_index_bag) = result;
        let changed_vector_properties =
            changed_vector_properties(&vector_properties, &old_index_bag, &new_index_bag);
        for property in changed_vector_properties {
            self.mark_or_queue_vector_node_updated(tenant, &property, node, ctx)?;
        }
        self.maintain_property_index_best_effort(
            tenant,
            node,
            node_label,
            Some(&old_index_bag),
            &new_index_bag,
        );
        Ok(())
    }

    // ADR-150 W26-θ Phase 4 — REMOVE-rel production wire-through.
    fn remove_rel(
        &self,
        tenant: TenantId,
        rel: RelId,
        mutation: &RemoveRelMutation,
        ctx: &ExecutionContext,
    ) -> Result<(), SubstrateAccessError> {
        // ADR-197: read-modify-write into the held tx (EXPLICIT) or a
        // fresh begin+commit (AUTO-COMMIT) — symmetric with remove_node.
        self.run_txn(tenant, ctx, |crud, tx| {
            // ADR-152 §D-2 — read current bag, remove key, write back.
            let current_bag = match arcgraph_storage::crud::read_rel(tx, rel) {
                Ok(Some(r)) => crate::storage::property_payload::rel_record_property_bag_checked(
                    &r,
                    crud.blob_store(),
                    &self.intern_table,
                    tenant,
                )?,
                Ok(None) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "remove_rel: rel {} not visible at snapshot",
                        rel.raw()
                    )));
                }
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "remove_rel: read_rel failed: {e}"
                    )));
                }
            };
            let new_bag = apply_remove_rel_mutation(current_bag, mutation);
            let props = crate::storage::property_payload::property_map_to_property_data_typed(
                &new_bag,
                &self.intern_table,
                crud.wal(),
                tenant,
            )?;
            arcgraph_storage::crud::update_rel(crud, tx, rel, &props)
                .map_err(|e| SubstrateAccessError::Io(format!("remove_rel: storage rejected: {e}")))
        })
    }

    // ── #830 / ADR-200 — vector-index catalog (in-memory, per-tenant) ──
    //
    // METADATA only. The served HNSW BUILD is auto-on-ingest (#765
    // PART-1) — registering an entry does NOT touch storage / WAL / the
    // HNSW. The catalog is process-lifetime + re-created on restart
    // (acceptable for the langchain happy path; persistent catalog is a
    // GA follow-on). Defense-in-depth tenant routing check mirrors the
    // other trait methods so a mis-wired tenant catches here.
    fn register_vector_index(
        &self,
        tenant: TenantId,
        entry: VectorIndexCatalogEntry,
        if_not_exists: bool,
    ) -> Result<VectorIndexRegistration, SubstrateAccessError> {
        self.router
            .route(tenant, PartitionId::ZERO)
            .map_err(|_| SubstrateAccessError::TenantUnknown(tenant))?;
        let mut catalog = self.vector_index_catalog.write().map_err(|e| {
            SubstrateAccessError::Io(format!("vector-index catalog write lock poisoned: {e}"))
        })?;
        let bucket = catalog.entry(tenant).or_default();
        if bucket.iter().any(|e| e.name == entry.name) {
            if if_not_exists {
                return Ok(VectorIndexRegistration::AlreadyExists);
            }
            return Err(SubstrateAccessError::IndexAlreadyExists { name: entry.name });
        }
        bucket.push(entry);
        Ok(VectorIndexRegistration::Created)
    }

    fn list_vector_indexes(&self, tenant: TenantId) -> Vec<VectorIndexCatalogEntry> {
        self.vector_index_catalog
            .read()
            .ok()
            .and_then(|catalog| catalog.get(&tenant).cloned())
            .unwrap_or_default()
    }

    // `resolve_vector_index` uses the trait default (filter over
    // `list_vector_indexes`) — a tenant has O(1) named vector indexes at
    // v1.0-α (the langchain happy path registers exactly one).

    // ── #1366 (task #248, Phase 1) — property-index CREATE + backfill ──
    //
    // #1401 missed-node W1 race fix: register the durable catalog
    // `Building` record FIRST, THEN take the backfill snapshot, THEN
    // backfill, THEN flip `Online`. Ordering matters — see below. Owns
    // JSON decoding here (PD#7): the storage/index layers see only typed
    // key deltas.
    //
    // # Ordering (the invariant + why it holds)
    //
    // 1. `register_building` commits the catalog `Building` record (and
    //    publishes the runtime handle). From this instant `indexes_on` is
    //    non-empty, so every subsequent write's `maintain_property_index_
    //    best_effort` MAINTAINS the index (no more empty-catalog no-op).
    // 2. `scan_for_backfill` begins the backfill snapshot S *after* the
    //    register commit and derives `high_water` from within S (no
    //    high_water-vs-scan-begin skew — the second W1 path).
    // 3. `backfill_and_flip` inserts the frozen snapshot then flips
    //    `Online`.
    //
    // Invariant: a concurrent writer is EITHER maintained (it committed
    // after the register, catalog non-empty → step 1) OR captured in the
    // snapshot (it committed before the register, hence before S → step
    // 2). Never neither, so no node is permanently missing. Overlap
    // (a writer maintained during build that is ALSO in the snapshot)
    // collapses via idempotent insert (candidate-then-verify tolerates
    // the dup).
    fn create_property_index(
        &self,
        tenant: TenantId,
        name: &str,
        if_not_exists: bool,
        label: &str,
        property: &str,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        // Resolve the label + property-key ids (the property KEY stays
        // interned; the property VALUE is hashed — RC-4).
        //
        // v2 M2 A4 round-2 (#1452): BOTH ids are embedded verbatim in
        // the DURABLE catalog record that `register_building` commits
        // below, so BOTH legs go through the durable-proof logged
        // intern — the same ordering discipline as the property-write
        // path. Pre-fix these were the UNLOGGED `intern_label`/`intern`:
        // a crash after the catalog commit recovered the catalog record
        // but neither binding, so the unseeded allocator re-handed both
        // ids and the durable index silently rebound to whatever
        // unrelated names interned first ("indexed_prop" resolving as
        // "unrelated_prop" — the silent-rebind class). The fsync-
        // blocking `InternString` appends RETURN before the catalog tx
        // commits (strictly lower LSNs on the ONE shared WAL —
        // `property_index_manager` wires `self.txn_manager` +
        // `crud.wal()` from the same store), so recovery always replays
        // the bindings the recovered catalog references; an append
        // failure aborts the CREATE before any catalog record exists.
        // Gate: `m2_catalog_intern_durability_gate.rs` (arcgraph-mcp).
        let label_id =
            arcgraph_storage::intern_label_logged(&self.intern_table, crud.wal(), tenant, label)
                .map_err(|e| {
                    SubstrateAccessError::Io(format!(
                        "create_property_index: intern WAL log failed for label {label:?}: {e}"
                    ))
                })?;
        let property_key = arcgraph_storage::intern_string_logged(
            &self.intern_table,
            crud.wal(),
            tenant,
            property,
        )
        .map_err(|e| {
            SubstrateAccessError::Io(format!(
                "create_property_index: intern WAL log failed for property key {property:?}: {e}"
            ))
        })?;

        let manager = self.property_index_manager(&crud)?;

        let map_err = |e: crate::storage::property_index::PropertyIndexError| {
            match e {
            crate::storage::property_index::PropertyIndexError::Catalog(
                arcgraph_storage::property_index_catalog::PropertyIndexCatalogError::AlreadyExists {
                    name,
                },
            ) => SubstrateAccessError::IndexAlreadyExists { name },
            other => SubstrateAccessError::Io(format!("property-index CREATE failed: {other}")),
        }
        };

        // STEP 1 — register the catalog `Building` record + publish the
        // handle BEFORE any snapshot. `None` ⇒ `IF NOT EXISTS` idempotent
        // no-op (nothing to backfill).
        let handle = match manager
            .register_building(crate::storage::property_index::CreateIndexSpec {
                tenant,
                name,
                if_not_exists,
                label: label_id,
                property_key,
                property_name: property,
            })
            .map_err(map_err)?
        {
            None => return Ok(PropertyIndexRegistration::AlreadyExists),
            Some(h) => h,
        };

        // #1401 test seam — fire the barrier (if a test installed one on
        // THIS thread) at the register↔snapshot boundary. In the FIXED
        // ordering this point is AFTER the catalog `Building` commit and
        // BEFORE the backfill snapshot, so a writer released here is
        // MAINTAINED (catalog non-empty). In the pre-fix ordering (where
        // the snapshot precedes the register — the reverted code the
        // RED-on-revert run exercises) the same seam sits AFTER the
        // snapshot and BEFORE the register, so the released writer is
        // captured by NEITHER. No-op in production (the thread-local is
        // never set off the test module).
        #[cfg(test)]
        tests::fire_create_index_register_barrier();

        // STEP 2 — snapshot-scan the tenant's MVCC-visible nodes ONCE for
        // the backfill, under a snapshot taken AFTER the register (with a
        // consistent `high_water` bound). Any node a concurrent writer
        // committed after the register is instead maintained by that
        // writer (step 1), so it need not appear here. We own the JSON
        // decode via the existing scan; the manager owns key extraction.
        let nodes: Vec<(NodeId, LabelId, std::collections::BTreeMap<String, Value>)> = self
            .scan_for_backfill(tenant, Some(label_id))
            .map_err(|e| {
                SubstrateAccessError::Io(format!("property-index backfill scan failed: {e}"))
            })?
            .into_iter()
            .map(|bn| {
                let id = NodeId::new(bn.node.id.raw());
                let bag = bn.node.properties.clone();
                (id, label_id, bag)
            })
            .collect();

        // STEP 3 — backfill the frozen snapshot into the published
        // handle, then flip `Online`.
        let outcome = manager
            .backfill_and_flip(
                &handle,
                tenant,
                name,
                nodes.iter().map(|(n, l, b)| (*n, *l, b)),
            )
            .map_err(map_err)?;

        Ok(match outcome {
            arcgraph_storage::property_index_catalog::CreateOutcome::Created => {
                PropertyIndexRegistration::Created
            }
            arcgraph_storage::property_index_catalog::CreateOutcome::AlreadyExists => {
                PropertyIndexRegistration::AlreadyExists
            }
        })
    }
}

/// **#1401 RED-on-revert oracle — the PRE-FIX buggy ordering.**
///
/// Test-only inherent method (NOT part of the `ExecutorSubstrate` trait —
/// production has only the fixed `create_property_index`). Reproduces the
/// ordering `create_property_index` had BEFORE the #1401 fix so the
/// barrier-gated missed-node race test can drive the SAME interleaving
/// against both orderings and prove the assertion is RED here (pre-fix)
/// and GREEN on the fixed method (a non-discriminating test would be a
/// fail).
#[cfg(test)]
impl CrudExecutorSubstrate {
    /// The pre-fix ordering, **path 1 (register-after-snapshot)**: the
    /// backfill snapshot S is captured (tx begun) BEFORE the barrier,
    /// then the catalog `Building` record is registered ONLY AFTER. A
    /// concurrent writer released at the barrier commits at LSN>S, so it
    /// is invisible to the frozen scan (excluded from the Vec) AND its
    /// maintain reads an empty catalog (register hasn't happened) → no-op.
    /// NEITHER → the missed node. Used by the insert + update variants.
    fn create_property_index_prefix_ordering(
        &self,
        tenant: TenantId,
        name: &str,
        if_not_exists: bool,
        label: &str,
        property: &str,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        // A4 round-2 (#1452): the oracles keep the fixed method's
        // durable-proof intern legs — they reproduce the #1401
        // snapshot-vs-register ordering ONLY, so their discriminating
        // seam (the barrier placement) stays isolated from the intern
        // durability class.
        let label_id =
            arcgraph_storage::intern_label_logged(&self.intern_table, crud.wal(), tenant, label)
                .map_err(|e| {
                    SubstrateAccessError::Io(format!(
                        "prefix oracle: intern WAL log failed for label {label:?}: {e}"
                    ))
                })?;
        let property_key = arcgraph_storage::intern_string_logged(
            &self.intern_table,
            crud.wal(),
            tenant,
            property,
        )
        .map_err(|e| {
            SubstrateAccessError::Io(format!(
                "prefix oracle: intern WAL log failed for property key {property:?}: {e}"
            ))
        })?;
        let manager = self.property_index_manager(&crud)?;

        // PRE-FIX STEP 1 — BEGIN the backfill snapshot S FIRST (pin the
        // read tx), THEN fire the barrier so W commits at LSN>S (invisible
        // to S), THEN scan under S. This reproduces `[scan begins @ S] →
        // [writer commits @ LSN>S] → [catalog Building commits]`.
        let high_water = crud.node_high_water(tenant);
        let tx = self.txn_manager.begin(tenant);
        // Barrier: W commits its write AFTER S is pinned, BEFORE register.
        tests::fire_create_index_register_barrier();
        let nodes: Vec<(NodeId, LabelId, std::collections::BTreeMap<String, Value>)> = if high_water
            == 0
        {
            Vec::new()
        } else {
            self.scan_id_range_in_tx(tenant, &crud, &tx, 1..=high_water, Some(label_id))
                .map_err(|e| {
                    SubstrateAccessError::Io(format!("property-index backfill scan failed: {e}"))
                })?
                .into_iter()
                .map(|bn| {
                    let id = NodeId::new(bn.node.id.raw());
                    let bag = bn.node.properties.clone();
                    (id, label_id, bag)
                })
                .collect()
        };

        // PRE-FIX STEP 2 — register + backfill + flip in one shot (the old
        // `create_index` wrapper): register commits AFTER the snapshot, so
        // W's earlier maintain saw an empty catalog and was a no-op.
        let outcome = self.finish_prefix_create(
            &manager,
            tenant,
            name,
            if_not_exists,
            label_id,
            property_key,
            property,
            nodes,
        )?;
        Ok(outcome)
    }

    /// The pre-fix ordering, **path 2 (high_water-before-scan-begin)**:
    /// `high_water` is sampled BEFORE the barrier (so a writer that
    /// creates a NEW node at the barrier gets `id = high_water + 1`),
    /// then the scan begins AFTER the barrier over `1..=high_water` — so
    /// W is visible-to-snapshot yet `id > high_water` and is dropped from
    /// the range. Used by the high_water variant.
    fn create_property_index_prefix_high_water(
        &self,
        tenant: TenantId,
        name: &str,
        if_not_exists: bool,
        label: &str,
        property: &str,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        let crud = self.crud_for(tenant)?;
        // A4 round-2 (#1452): durable-proof intern legs, as in the
        // fixed method — see the `prefix_ordering` oracle's note.
        let label_id =
            arcgraph_storage::intern_label_logged(&self.intern_table, crud.wal(), tenant, label)
                .map_err(|e| {
                    SubstrateAccessError::Io(format!(
                        "prefix oracle: intern WAL log failed for label {label:?}: {e}"
                    ))
                })?;
        let property_key = arcgraph_storage::intern_string_logged(
            &self.intern_table,
            crud.wal(),
            tenant,
            property,
        )
        .map_err(|e| {
            SubstrateAccessError::Io(format!(
                "prefix oracle: intern WAL log failed for property key {property:?}: {e}"
            ))
        })?;
        let manager = self.property_index_manager(&crud)?;

        // Sample high_water FIRST (pre-barrier), then let W create a new
        // node (id = high_water + 1), then scan the STALE range.
        let high_water = crud.node_high_water(tenant);
        tests::fire_create_index_register_barrier();
        let nodes: Vec<(NodeId, LabelId, std::collections::BTreeMap<String, Value>)> = if high_water
            == 0
        {
            Vec::new()
        } else {
            let tx = self.txn_manager.begin(tenant);
            self.scan_id_range_in_tx(tenant, &crud, &tx, 1..=high_water, Some(label_id))
                .map_err(|e| {
                    SubstrateAccessError::Io(format!("property-index backfill scan failed: {e}"))
                })?
                .into_iter()
                .map(|bn| {
                    let id = NodeId::new(bn.node.id.raw());
                    let bag = bn.node.properties.clone();
                    (id, label_id, bag)
                })
                .collect()
        };
        self.finish_prefix_create(
            &manager,
            tenant,
            name,
            if_not_exists,
            label_id,
            property_key,
            property,
            nodes,
        )
    }

    /// Shared tail for the pre-fix oracles: register+backfill+flip in one
    /// shot (the old eager `create_index` wrapper).
    #[allow(clippy::too_many_arguments)]
    fn finish_prefix_create(
        &self,
        manager: &crate::storage::property_index::PropertyIndexManager,
        tenant: TenantId,
        name: &str,
        if_not_exists: bool,
        label_id: LabelId,
        property_key: arcgraph_core::ids::StringId,
        property: &str,
        nodes: Vec<(NodeId, LabelId, std::collections::BTreeMap<String, Value>)>,
    ) -> Result<PropertyIndexRegistration, SubstrateAccessError> {
        let outcome = manager
            .create_index(
                crate::storage::property_index::CreateIndexSpec {
                    tenant,
                    name,
                    if_not_exists,
                    label: label_id,
                    property_key,
                    property_name: property,
                },
                nodes.iter().map(|(n, l, b)| (*n, *l, b)),
            )
            .map_err(|e| match e {
                crate::storage::property_index::PropertyIndexError::Catalog(
                    arcgraph_storage::property_index_catalog::PropertyIndexCatalogError::AlreadyExists {
                        name,
                    },
                ) => SubstrateAccessError::IndexAlreadyExists { name },
                other => SubstrateAccessError::Io(format!("property-index CREATE failed: {other}")),
            })?;
        Ok(match outcome {
            arcgraph_storage::property_index_catalog::CreateOutcome::Created => {
                PropertyIndexRegistration::Created
            }
            arcgraph_storage::property_index_catalog::CreateOutcome::AlreadyExists => {
                PropertyIndexRegistration::AlreadyExists
            }
        })
    }
}

// ─────────────────────────────────────────────────────────────────────
// ADR-152 §D-2 — SET / REMOVE mutation appliers
// ─────────────────────────────────────────────────────────────────────
//
// Apply a SET / REMOVE mutation to the in-memory property bag pulled
// from the current MVCC snapshot, then route the merged bag back
// through `update_node` / `update_rel`. PropertyAssign overwrites the
// single key; PropertyReplace clears the bag + inserts the new entries
// (`SET n = {k: v}` semantic per ADR-150 §D-1); PropertyMerge inserts
// the new entries on top of the existing bag (`SET n += {k: v}`
// semantic per ADR-150 §D-1). Remove::Property drops the named key.
//
// LabelAdd / LabelRemove never reach these helpers — the surface-level
// substrate methods short-circuit to `IndexUnavailable` per ADR-150
// §D-9. The mutation enums' Label* variants are pattern-arms that
// surface a defensive panic-free no-op (preserving the bag).

fn apply_set_node_mutation(
    mut bag: std::collections::BTreeMap<String, Value>,
    mutation: &SetNodeMutation,
) -> std::collections::BTreeMap<String, Value> {
    match mutation {
        SetNodeMutation::PropertyAssign { name, value } => {
            bag.insert(name.clone(), value.clone());
        }
        SetNodeMutation::PropertyReplace(entries) => {
            bag.clear();
            for (k, v) in entries {
                bag.insert(k.clone(), v.clone());
            }
        }
        SetNodeMutation::PropertyMerge(entries) => {
            for (k, v) in entries {
                bag.insert(k.clone(), v.clone());
            }
        }
        SetNodeMutation::LabelAdd(_) => {
            // Short-circuit prevented this in set_node; defensive
            // no-op to keep the helper total.
        }
    }
    bag
}

fn indexable_string_text(bag: &std::collections::BTreeMap<String, Value>) -> String {
    let mut out = String::new();
    for value in bag.values() {
        if let Value::String(text) = value {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    out
}

fn changed_vector_properties(
    vector_properties: &HashSet<String>,
    old_bag: &std::collections::BTreeMap<String, Value>,
    new_bag: &std::collections::BTreeMap<String, Value>,
) -> Vec<String> {
    vector_properties
        .iter()
        .filter(|property| old_bag.get(property.as_str()) != new_bag.get(property.as_str()))
        .cloned()
        .collect()
}

fn apply_set_rel_mutation(
    mut bag: std::collections::BTreeMap<String, Value>,
    mutation: &SetRelMutation,
) -> std::collections::BTreeMap<String, Value> {
    match mutation {
        SetRelMutation::PropertyAssign { name, value } => {
            bag.insert(name.clone(), value.clone());
        }
        SetRelMutation::PropertyReplace(entries) => {
            bag.clear();
            for (k, v) in entries {
                bag.insert(k.clone(), v.clone());
            }
        }
        SetRelMutation::PropertyMerge(entries) => {
            for (k, v) in entries {
                bag.insert(k.clone(), v.clone());
            }
        }
    }
    bag
}

fn apply_remove_node_mutation(
    mut bag: std::collections::BTreeMap<String, Value>,
    mutation: &RemoveNodeMutation,
) -> std::collections::BTreeMap<String, Value> {
    match mutation {
        RemoveNodeMutation::Property(name) => {
            bag.remove(name);
        }
        RemoveNodeMutation::LabelRemove(_) => {
            // Short-circuit prevented this in remove_node; defensive
            // no-op.
        }
    }
    bag
}

fn apply_remove_rel_mutation(
    mut bag: std::collections::BTreeMap<String, Value>,
    mutation: &RemoveRelMutation,
) -> std::collections::BTreeMap<String, Value> {
    match mutation {
        RemoveRelMutation::Property(name) => {
            bag.remove(name);
        }
    }
    bag
}

/// Translate a [`RoutingError`] into the substrate trait's error
/// taxonomy.
fn translate_routing_error(err: RoutingError, tenant: TenantId) -> SubstrateAccessError {
    match err {
        RoutingError::UnknownTenant { .. } => SubstrateAccessError::TenantUnknown(tenant),
        RoutingError::PartitionNotSupported { partition_raw } => SubstrateAccessError::Io(format!(
            "routing: partition {partition_raw} not supported at v1.0 (substrate uses ZERO)"
        )),
    }
}

/// **#907.** Translate a [`crud::commit`](arcgraph_storage::crud::commit)
/// [`CrudError`](crud::CrudError) into the substrate trait's error
/// taxonomy, preserving a write-write **MVCC conflict** as the TYPED
/// [`SubstrateAccessError::Conflict`] variant instead of flattening it
/// into the generic [`SubstrateAccessError::Io`] bucket.
///
/// We match the TYPED `CrudError::Mvcc(ArcGraphError::MvccConflict)`
/// variant — NOT the error *string* (string-matching the rendered
/// "MVCC commit failed: …" is fragile and was explicitly rejected by
/// #907). Carrying the conflict as its own variant lets the public
/// boundary (the Bolt FAILURE mapper) classify it as a *retriable*
/// `Neo.TransientError.Transaction.*` code (which drivers auto-retry
/// under `session.execute_write`) rather than the *fatal*, non-retriable
/// `Neo.DatabaseError.General.UnknownError` the `Io` bucket mapped to —
/// the #907 defect that broke optimistic-concurrency retry AND leaked
/// the storage-layer wrapping to the client.
///
/// Every OTHER commit fault (genuine I/O, WAL, codec, primary-index,
/// id-space exhaustion) keeps the `Io` bucket with the `context`-prefixed
/// detail and STILL maps to `Neo.DatabaseError` downstream — only the
/// *logical* MVCC conflict is reclassified; nothing else is broadened.
fn commit_err_to_substrate(context: &str, e: crud::CrudError) -> SubstrateAccessError {
    match e {
        crud::CrudError::Mvcc(arcgraph_core::ArcGraphError::MvccConflict { target }) => {
            SubstrateAccessError::Conflict { target }
        }
        other => SubstrateAccessError::Io(format!("{context}: {other}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arcgraph_storage::buffer::BufferPool;
    use arcgraph_storage::catalog::SystemCatalog;
    use arcgraph_storage::crud::{PropertyData, commit, create_node, create_rel};
    use arcgraph_storage::io::InMemoryPageIo;
    use arcgraph_storage::router::MultiTenantRouter;

    /// Build a minimal substrate fixture: catalog bootstrapped,
    /// CrudStore + TxnManager + intern table all wired through a
    /// fresh MultiTenantRouter.
    fn fixture() -> (
        CrudExecutorSubstrate,
        Arc<CrudStore>,
        Arc<TxnManager>,
        Arc<MultiTenantRouter>,
    ) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
        let crud = Arc::new(CrudStore::new());
        let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
        let intern = Arc::new(InternTable::new());
        let sub =
            CrudExecutorSubstrate::new(Arc::clone(&router), Arc::clone(&mgr), Arc::clone(&intern));
        (sub, crud, mgr, router)
    }

    /// ADR-197 — a fresh AUTO-COMMIT `ExecutionContext` for the
    /// substrate write-method tests (no held tx ⇒ each call
    /// begins+commits its own tx, the pre-ADR-197 behavior these tests
    /// pin).
    fn tctx() -> ExecutionContext {
        ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
    }

    /// #907 — a REAL write-write MVCC conflict driven through the ACTUAL
    /// [`CrudExecutorSubstrate::commit_held_txn`] path (the Bolt
    /// explicit-tx COMMIT / driver `execute_write` path). Mirrors the
    /// canonical `arcgraph-storage` `mvcc_ww_conflict.rs` staging — a
    /// winner commits a version of `KEY` AFTER the loser's snapshot — but
    /// routes the losing commit through `crud::commit` + the substrate
    /// boundary. The OCC loser MUST surface as the TYPED
    /// [`SubstrateAccessError::Conflict`], NOT the generic `Io` bucket
    /// that (pre-#907) flattened it and made the Bolt boundary emit a
    /// fatal `Neo.DatabaseError` instead of a retriable
    /// `Neo.TransientError` — the defect that broke driver auto-retry.
    ///
    /// RED-on-revert: revert [`commit_err_to_substrate`] to
    /// `SubstrateAccessError::Io(format!(…))` and `err` becomes `Io(_)`,
    /// failing the `matches!` assert.
    #[test]
    fn mvcc_write_conflict_through_commit_held_txn_surfaces_typed_conflict() {
        use bytes::Bytes;
        let (sub, _crud, mgr, _router) = fixture();
        const KEY: u64 = 7;

        // Loser begins FIRST → its snapshot precedes the winner's commit.
        let mut loser = mgr.begin_owned(TenantId::DEFAULT);
        loser
            .txn_mut()
            .write(KEY, Bytes::from_static(b"from_loser"));
        let mut winner = mgr.begin_owned(TenantId::DEFAULT);
        winner
            .txn_mut()
            .write(KEY, Bytes::from_static(b"from_winner"));

        // Winner commits a version of KEY after the loser's snapshot.
        sub.commit_held_txn(winner).expect("winner commits cleanly");
        // Loser must now lose the OCC validation race.
        let err = sub
            .commit_held_txn(loser)
            .expect_err("OCC loser must conflict");

        assert!(
            matches!(err, SubstrateAccessError::Conflict { .. }),
            "MVCC loser must surface the TYPED Conflict variant (not Io); got {err:?}"
        );
    }

    /// #907 — [`commit_err_to_substrate`] reclassifies ONLY the MVCC
    /// conflict; every other `CrudError` keeps the `Io` bucket (so it
    /// still maps to a fatal `Neo.DatabaseError` downstream — the
    /// retriable class is NOT over-broadened).
    #[test]
    fn commit_err_to_substrate_only_reclassifies_mvcc_conflict() {
        // Conflict → typed retriable Conflict, carrying the target verbatim.
        let conflict = crud::CrudError::Mvcc(arcgraph_core::ArcGraphError::MvccConflict {
            target: "key:6404".into(),
        });
        match commit_err_to_substrate("write commit failed", conflict) {
            SubstrateAccessError::Conflict { target } => assert_eq!(target, "key:6404"),
            other => panic!("MVCC conflict must map to Conflict, got {other:?}"),
        }

        // A genuine non-conflict commit fault stays `Io` (fatal class).
        let exhausted = crud::CrudError::NodeIdExhausted {
            tenant: TenantId::DEFAULT,
        };
        match commit_err_to_substrate("write commit failed", exhausted) {
            SubstrateAccessError::Io(msg) => {
                assert!(
                    msg.starts_with("write commit failed: "),
                    "non-conflict fault keeps context-prefixed Io detail; got {msg}"
                );
            }
            other => panic!("non-conflict fault must stay Io, got {other:?}"),
        }
    }

    #[test]
    fn scan_nodes_returns_empty_for_empty_tenant() {
        let (sub, _crud, _mgr, _router) = fixture();
        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan_nodes");
        assert!(rows.is_empty());
    }

    #[test]
    fn scan_nodes_finds_inserted_nodes() {
        let (sub, crud, mgr, _router) = fixture();
        let label = LabelId::new(1);
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let n1 = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("create n1");
        let n2 = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("create n2");
        commit(tx, &crud).expect("commit");

        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan");
        assert_eq!(rows.len(), 2);
        assert!(rows.iter().any(|b| b.node.id == n1));
        assert!(rows.iter().any(|b| b.node.id == n2));
    }

    #[test]
    fn scan_nodes_label_filter_excludes_others() {
        let (sub, crud, mgr, _router) = fixture();
        let l1 = LabelId::new(1);
        let l2 = LabelId::new(2);
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let _ = create_node(&crud, &mut tx, TenantId::DEFAULT, l1, &PropertyData::Empty)
            .expect("create l1");
        let n2 = create_node(&crud, &mut tx, TenantId::DEFAULT, l2, &PropertyData::Empty)
            .expect("create l2");
        commit(tx, &crud).expect("commit");

        let l1_rows = sub
            .scan_nodes(TenantId::DEFAULT, Some(l1), Lsn::MAX)
            .expect("scan l1");
        assert_eq!(l1_rows.len(), 1);
        let l2_rows = sub
            .scan_nodes(TenantId::DEFAULT, Some(l2), Lsn::MAX)
            .expect("scan l2");
        assert_eq!(l2_rows.len(), 1);
        assert_eq!(l2_rows[0].node.id, n2);
    }

    /// Populates a small graph with a single outbound `KNOWS` edge
    /// from `n1` to `n2`. Returns `(sub, n1, n2, ty)`.
    fn small_graph_fixture() -> (CrudExecutorSubstrate, NodeId, NodeId, TypeId) {
        let (sub, crud, mgr, _router) = fixture();
        let label = LabelId::new(1);
        let ty = TypeId::new(1);
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let n1 = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("n1");
        let n2 = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("n2");
        let _ = create_rel(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            n1,
            n2,
            ty,
            &PropertyData::Empty,
        )
        .expect("rel");
        commit(tx, &crud).expect("commit");
        (sub, n1, n2, ty)
    }

    #[test]
    fn expand_left_to_right_remains_complete() {
        // No-regression pin for R1 fix (PR #349 HIGH-1): the outbound
        // happy path returns every matching edge.
        let (sub, n1, n2, ty) = small_graph_fixture();
        let edges = sub
            .expand(
                TenantId::DEFAULT,
                n1,
                Some(ty),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("expand");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].dst.id, n2);
    }

    #[test]
    fn expand_right_to_left_returns_reverse_view_w26_beta_2() {
        // W26-β-2 / ADR-131 — closes #350 v1.1 reverse-adjacency.
        // The forward edge `n1 -[KNOWS]-> n2` becomes visible to a
        // `Direction::RightToLeft` expand from `n2`: the inbound
        // walk yields a `BoundEdge` with the rel oriented n1→n2 and
        // `dst` (= far end of traversal) = n1.
        //
        // Supersedes the v1.0-α R1 pin
        // `expand_right_to_left_surfaces_unwired_error_not_silent_empty`
        // (PR #349 HIGH-1 — pinned the structured-error posture
        // forward-deferred to issue #350; this test pins the v1.1
        // positive-row posture per ADR-087 D-4).
        let (sub, n1, n2, ty) = small_graph_fixture();
        let edges = sub
            .expand(
                TenantId::DEFAULT,
                n2,
                Some(ty),
                Direction::RightToLeft,
                Lsn::MAX,
            )
            .expect("expand RightToLeft must succeed at v1.1");
        assert_eq!(edges.len(), 1, "exactly the inbound n1→n2 edge");
        assert_eq!(edges[0].rel.from, n1, "rel oriented n1→n2 canonically");
        assert_eq!(edges[0].rel.to, n2);
        assert_eq!(
            edges[0].dst.id, n1,
            "BoundEdge.dst = far end of traversal = n1"
        );
    }

    #[test]
    fn expand_undirected_returns_out_plus_in_no_double_counting_w26_beta_2() {
        // W26-β-2 / ADR-131 — closes #350 v1.1 undirected expand.
        // Undirected from `n1` yields exactly the outbound n1→n2 edge
        // (n1 has 1 outbound + 0 inbound = 1 total); undirected from
        // `n2` yields exactly the inbound n1→n2 edge (n2 has 0
        // outbound + 1 inbound = 1 total). No double-counting (the
        // single edge appears once per perspective, never twice).
        //
        // Supersedes the v1.0-α R1 pin
        // `expand_undirected_surfaces_unwired_error_not_partial`
        // (PR #349 HIGH-1).
        let (sub, n1, n2, ty) = small_graph_fixture();
        let from_n1 = sub
            .expand(
                TenantId::DEFAULT,
                n1,
                Some(ty),
                Direction::Undirected,
                Lsn::MAX,
            )
            .expect("expand Undirected from n1");
        assert_eq!(from_n1.len(), 1, "n1 sees the n1→n2 edge once");
        assert_eq!(from_n1[0].dst.id, n2);

        let from_n2 = sub
            .expand(
                TenantId::DEFAULT,
                n2,
                Some(ty),
                Direction::Undirected,
                Lsn::MAX,
            )
            .expect("expand Undirected from n2");
        assert_eq!(from_n2.len(), 1, "n2 sees the n1→n2 edge once (inbound)");
        assert_eq!(from_n2[0].dst.id, n1);
    }

    #[test]
    fn expand_undirected_self_loop_appears_exactly_once_w26_beta_2() {
        // W26-β-2 / ADR-131 §D-5 — self-loop dedup invariant: an
        // edge n1→n1 appears in BOTH the forward chain at (n1, ty)
        // AND the reverse chain at (n1, ty). `Direction::Undirected`
        // MUST dedup by `RelId` so the self-loop yields exactly one
        // BoundEdge.
        let (sub, crud, mgr, _router) = fixture();
        let label = LabelId::new(1);
        let ty = TypeId::new(1);
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let n1 = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("n1");
        let _ = create_rel(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            n1,
            n1,
            ty,
            &PropertyData::Empty,
        )
        .expect("self-loop");
        commit(tx, &crud).expect("commit");

        let edges = sub
            .expand(
                TenantId::DEFAULT,
                n1,
                Some(ty),
                Direction::Undirected,
                Lsn::MAX,
            )
            .expect("expand Undirected self-loop");
        assert_eq!(edges.len(), 1, "self-loop deduped to 1 BoundEdge");
        assert_eq!(edges[0].rel.from, n1);
        assert_eq!(edges[0].rel.to, n1);
        assert_eq!(edges[0].dst.id, n1);
    }

    #[test]
    fn expand_cursor_matches_expand_directional_and_undirected_multiset() {
        let (sub, n1, n2, ty) = small_graph_fixture();

        let eager_ltr = sub
            .expand(
                TenantId::DEFAULT,
                n1,
                Some(ty),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("eager ltr");
        let cursor_ltr = sub
            .expand_cursor(
                TenantId::DEFAULT,
                n1,
                Some(ty),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("cursor ltr")
            .collect::<Result<Vec<_>, _>>()
            .expect("cursor rows");
        assert_eq!(cursor_ltr, eager_ltr);

        let eager_rtl = sub
            .expand(
                TenantId::DEFAULT,
                n2,
                Some(ty),
                Direction::RightToLeft,
                Lsn::MAX,
            )
            .expect("eager rtl");
        let cursor_rtl = sub
            .expand_cursor(
                TenantId::DEFAULT,
                n2,
                Some(ty),
                Direction::RightToLeft,
                Lsn::MAX,
            )
            .expect("cursor rtl")
            .collect::<Result<Vec<_>, _>>()
            .expect("cursor rows");
        assert_eq!(cursor_rtl, eager_rtl);

        let eager_undir = sub
            .expand(
                TenantId::DEFAULT,
                n2,
                Some(ty),
                Direction::Undirected,
                Lsn::MAX,
            )
            .expect("eager undirected");
        let cursor_undir = sub
            .expand_cursor(
                TenantId::DEFAULT,
                n2,
                Some(ty),
                Direction::Undirected,
                Lsn::MAX,
            )
            .expect("cursor undirected")
            .collect::<Result<Vec<_>, _>>()
            .expect("cursor rows");
        let mut eager_ids: Vec<u64> = eager_undir.iter().map(|e| e.rel.id.raw()).collect();
        let mut cursor_ids: Vec<u64> = cursor_undir.iter().map(|e| e.rel.id.raw()).collect();
        eager_ids.sort_unstable();
        cursor_ids.sort_unstable();
        assert_eq!(cursor_ids, eager_ids);
    }

    #[test]
    fn held_expand_cursor_streams_staged_rels_in_fixed_high_water_order() {
        let (sub, crud, mgr, _router) = fixture();
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(1);
        let ty = TypeId::new(7);

        let mut seed = mgr.begin(tenant);
        let root = create_node(&crud, &mut seed, tenant, label, &PropertyData::Empty).unwrap();
        let a = create_node(&crud, &mut seed, tenant, label, &PropertyData::Empty).unwrap();
        let b = create_node(&crud, &mut seed, tenant, label, &PropertyData::Empty).unwrap();
        commit(seed, &crud).unwrap();

        let mut owned = mgr.begin_owned(tenant);
        let r_out = create_rel(
            &crud,
            owned.txn_mut(),
            tenant,
            root,
            a,
            ty,
            &PropertyData::Empty,
        )
        .unwrap();
        let r_in = create_rel(
            &crud,
            owned.txn_mut(),
            tenant,
            b,
            root,
            ty,
            &PropertyData::Empty,
        )
        .unwrap();
        let r_loop = create_rel(
            &crud,
            owned.txn_mut(),
            tenant,
            root,
            root,
            ty,
            &PropertyData::Empty,
        )
        .unwrap();
        let _unrelated = create_rel(
            &crud,
            owned.txn_mut(),
            tenant,
            a,
            b,
            ty,
            &PropertyData::Empty,
        )
        .unwrap();

        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO)
            .with_held_txn(Box::new(BoltHeldTxn::new(owned)));
        let cursor = sub
            .expand_cursor_with_context(&ctx, root, Some(ty), Direction::Undirected, Lsn::MAX)
            .unwrap();

        // Allocation after open is staged and visible to later reads, but is
        // outside this cursor's fixed raw-id high-water.
        let late = ctx
            .with_held_txn_mut(|handle| {
                let held = handle.as_any_mut().downcast_mut::<BoltHeldTxn>().unwrap();
                create_rel(
                    &crud,
                    held.owned_mut().unwrap().txn_mut(),
                    tenant,
                    root,
                    b,
                    ty,
                    &PropertyData::Empty,
                )
            })
            .unwrap()
            .unwrap();

        let edges = cursor.collect::<Result<Vec<_>, _>>().unwrap();
        let ids: Vec<u64> = edges.iter().map(|edge| edge.rel.id.raw()).collect();
        assert_eq!(ids, vec![r_out.raw(), r_in.raw(), r_loop.raw()]);
        assert!(!ids.contains(&late.raw()));
        assert_eq!(
            ids.iter().filter(|&&id| id == r_loop.raw()).count(),
            1,
            "held undirected self-loop appears exactly once"
        );
    }

    #[test]
    fn expand_right_to_left_surfaces_index_unavailable_when_reverse_disabled_ac4() {
        // W26-β-2 / ADR-131 AC-4 — fault injection: when the reverse
        // adjacency index is DISABLED (or unbuilt at recovery), the
        // substrate MUST surface a structured
        // `SubstrateAccessError::IndexUnavailable` for RightToLeft
        // and Undirected — NOT silent-empty results. Per
        // `feedback_load_bearing_pr_requires_fault_injection_tests.md`.
        let (sub, _n1, n2, ty) = small_graph_fixture();
        // Flip the reverse-index flag off post-population to simulate
        // "rels were created but the reverse index has not yet been
        // populated" (= a post-recovery / pre-rebuild posture).
        let crud = sub.crud_for(TenantId::DEFAULT).expect("crud handle");
        crud.set_reverse_index_enabled(false);

        let rtl = sub.expand(
            TenantId::DEFAULT,
            n2,
            Some(ty),
            Direction::RightToLeft,
            Lsn::MAX,
        );
        match rtl {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert!(
                    detail.contains("reverse-adjacency"),
                    "expected reverse-adjacency IndexUnavailable, got {detail}"
                );
            }
            other => panic!("expected IndexUnavailable error, got {other:?}"),
        }

        let undir = sub.expand(
            TenantId::DEFAULT,
            n2,
            Some(ty),
            Direction::Undirected,
            Lsn::MAX,
        );
        match undir {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert!(
                    detail.contains("reverse-adjacency"),
                    "expected reverse-adjacency IndexUnavailable, got {detail}"
                );
            }
            other => panic!("expected IndexUnavailable error, got {other:?}"),
        }

        // LeftToRight remains operative — the forward chain is
        // untouched by the reverse-index disable flag.
        let _ltr = sub
            .expand(
                TenantId::DEFAULT,
                _n1,
                Some(ty),
                Direction::LeftToRight,
                Lsn::MAX,
            )
            .expect("LeftToRight must remain operative");
    }

    #[test]
    fn expand_cursor_surfaces_index_unavailable_at_open_when_reverse_disabled_ac4() {
        let (sub, _n1, n2, ty) = small_graph_fixture();
        let crud = sub.crud_for(TenantId::DEFAULT).expect("crud handle");
        crud.set_reverse_index_enabled(false);

        let rtl = sub.expand_cursor(
            TenantId::DEFAULT,
            n2,
            Some(ty),
            Direction::RightToLeft,
            Lsn::MAX,
        );
        match rtl {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert!(detail.contains("reverse-adjacency"));
            }
            Ok(_) => panic!("expected cursor-open IndexUnavailable, got cursor"),
            Err(other) => panic!("expected cursor-open IndexUnavailable, got {other:?}"),
        }

        let undir = sub.expand_cursor(
            TenantId::DEFAULT,
            n2,
            Some(ty),
            Direction::Undirected,
            Lsn::MAX,
        );
        match undir {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert!(detail.contains("reverse-adjacency"));
            }
            Ok(_) => panic!("expected cursor-open IndexUnavailable, got cursor"),
            Err(other) => panic!("expected cursor-open IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn scan_nodes_does_not_surface_inline_u32_property_names() {
        // R1 fix (PR #349 MED-6): `_inline_u32a` / `_inline_u32b` are
        // storage-internal optimization fields and must not surface as
        // user-visible property names on the wire. v1.2 (issue #356)
        // lands the real JSON-blob property decode.
        use arcgraph_storage::crud::PropertyData;

        let (sub, crud, mgr, _router) = fixture();
        let label = LabelId::new(1);
        let mut tx = mgr.begin(TenantId::DEFAULT);
        // Inline-pair payload populates `inline_u32a/b` on the
        // record; the old code path surfaced these as
        // `_inline_u32a/b` properties.
        let _n1 = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::InlineU32Pair(7, 42),
        )
        .expect("create n1");
        commit(tx, &crud).expect("commit");

        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan");
        assert_eq!(rows.len(), 1);
        let node = &rows[0].node;
        // No `_inline_u32*` property must surface on the wire.
        for key in node.properties.keys() {
            assert!(
                !key.starts_with("_inline_u32"),
                "property key {key:?} must not leak storage-internal name"
            );
        }
    }

    #[test]
    fn vector_search_unavailable_when_substrate_not_attached() {
        // AC-3 negative case (vector): unwired-tenant continues to
        // surface structured `IndexUnavailable("vector")` per
        // W23-M4-08-FINALIZE ADR-087 §"What this locks in" contract
        // — preserved verbatim through W26-β-3 / ADR-132 AC-6 since
        // the v1.0-α posture remains binding for the UNWIRED case.
        let (sub, _crud, _mgr, _router) = fixture();
        let r = sub.vector_search(TenantId::DEFAULT, "embedding", &[0.0], 5, Lsn::MAX);
        match r {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                // Must be the "router has no vector attached" detail,
                // NOT the "provider not bound" detail — the unwired-
                // router error path is the load-bearing one for
                // unconfigured tenants.
                assert_eq!(
                    detail, "vector",
                    "unwired-tenant must surface the legacy `vector` \
                     detail (router-level), not the provider-bound one; \
                     got {detail:?}"
                );
            }
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn bm25_search_unavailable_when_substrate_not_attached() {
        // AC-3 negative case (BM25): symmetric to the vector test
        // above. Preserved through W26-β-3 / ADR-132 AC-6.
        let (sub, _crud, _mgr, _router) = fixture();
        let r = sub.bm25_search(TenantId::DEFAULT, "content", "alice", 5, Lsn::MAX);
        match r {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert_eq!(
                    detail, "bm25",
                    "unwired-tenant must surface the legacy `bm25` \
                     detail (router-level); got {detail:?}"
                );
            }
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn unknown_tenant_surfaces_tenant_unknown() {
        let (sub, _crud, _mgr, _router) = fixture();
        let unknown = TenantId::new(9999);
        let r = sub.scan_nodes(unknown, None, Lsn::MAX);
        match r {
            Err(SubstrateAccessError::TenantUnknown(t)) => assert_eq!(t, unknown),
            other => panic!("expected TenantUnknown, got {other:?}"),
        }
    }

    // =================================================================
    // W26-β-3 / ADR-132 — substrate-body search wire-through fixtures.
    // =================================================================

    /// Commit-side `VectorPageStoreHandle` stub. The router only
    /// inspects `is_some()` on the handle — the actual install /
    /// restore traffic is exercised by `arcgraph-storage`'s own
    /// recovery tests. We give the router a `Some(_)` so its
    /// `vector()` accessor surfaces a populated handle to the
    /// `vector_search` body's attachment gate.
    #[derive(Debug)]
    struct NoopVectorStore;

    impl arcgraph_storage::vector_store::VectorPageStoreHandle for NoopVectorStore {
        fn install_or_replace(
            &self,
            _tenant: TenantId,
            _page_id: arcgraph_core::PageId,
            _bytes: &[u8],
        ) -> Result<(), arcgraph_storage::vector_store::VectorStoreError> {
            Ok(())
        }
        fn restore_page_bytes(
            &self,
            _tenant: TenantId,
            _page_id: arcgraph_core::PageId,
            _bytes: &[u8],
        ) -> Result<(), arcgraph_storage::vector_store::VectorStoreError> {
            Ok(())
        }
    }

    /// Commit-side `Bm25IndexStoreHandle` stub — same shape as the
    /// `NoopVectorStore` above, mirrored from
    /// `crates/arcgraph-storage/tests/multi_tenant_routing.rs::LocalNoopBm25Store`.
    #[derive(Debug)]
    struct NoopBm25Store;

    impl arcgraph_storage::mutation_log::Bm25IndexStoreHandle for NoopBm25Store {
        fn commit_pending(
            &self,
            _tenant: TenantId,
        ) -> Result<(), arcgraph_storage::mutation_log::Bm25StoreError> {
            Ok(())
        }
        fn rollback_pending(
            &self,
            _tenant: TenantId,
        ) -> Result<(), arcgraph_storage::mutation_log::Bm25StoreError> {
            Ok(())
        }
    }

    /// Captured invocation of a `SubstrateSearchProvider` method.
    /// Used by AC-3 positive + snapshot tests to assert the
    /// substrate forwards `(tenant, property, k)` unchanged and supplies
    /// the resolved effective `read_lsn`.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ProviderCall {
        kind: &'static str,
        tenant: TenantId,
        property: String,
        k: u64,
        read_lsn: Lsn,
        // For vector calls: number of f32s in the query (we don't
        // capture the full vector because tests rarely care about
        // bit-exact equality; the substrate's responsibility is to
        // forward the slice unchanged, and that's what we assert).
        query_vec_len: Option<usize>,
        // For BM25 calls: the literal query text.
        query_text: Option<String>,
    }

    /// W26-β-3 / ADR-132 AC-3 — recording `SubstrateSearchProvider`
    /// stub. Captures the last call's args + returns a canned
    /// `Vec<RankedHit>` so tests assert (a) the substrate forwards query
    /// args unchanged with the effective snapshot, and (b) the body
    /// returns real hits, not `IndexUnavailable`.
    #[derive(Debug)]
    struct RecordingSearchProvider {
        canned_hits: parking_lot::Mutex<Vec<RankedHit>>,
        last_call: parking_lot::Mutex<Option<ProviderCall>>,
    }

    impl RecordingSearchProvider {
        fn new(canned_hits: Vec<RankedHit>) -> Self {
            Self {
                canned_hits: parking_lot::Mutex::new(canned_hits),
                last_call: parking_lot::Mutex::new(None),
            }
        }
        fn take_last_call(&self) -> Option<ProviderCall> {
            self.last_call.lock().take()
        }
    }

    impl SubstrateSearchProvider for RecordingSearchProvider {
        fn vector_search(
            &self,
            tenant: TenantId,
            property: &str,
            query_vec: &[f32],
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            *self.last_call.lock() = Some(ProviderCall {
                kind: "vector",
                tenant,
                property: property.into(),
                k,
                read_lsn,
                query_vec_len: Some(query_vec.len()),
                query_text: None,
            });
            Ok(self.canned_hits.lock().clone())
        }
        fn bm25_search(
            &self,
            tenant: TenantId,
            property: &str,
            query_text: &str,
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            *self.last_call.lock() = Some(ProviderCall {
                kind: "bm25",
                tenant,
                property: property.into(),
                k,
                read_lsn,
                query_vec_len: None,
                query_text: Some(query_text.into()),
            });
            Ok(self.canned_hits.lock().clone())
        }
    }

    /// Fixture variant that attaches Noop vector + BM25 to the
    /// router so the substrate's `route().vector()` /
    /// `route().bm25()` attachment gate passes. The substrate's
    /// `SubstrateSearchProvider` is left UNBOUND — callers that want
    /// the provider attached chain `.with_search_provider(...)`
    /// per their test scope.
    fn fixture_with_attached_substrates() -> (
        CrudExecutorSubstrate,
        Arc<CrudStore>,
        Arc<TxnManager>,
        Arc<MultiTenantRouter>,
    ) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = Arc::new(TxnManager::new());
        let catalog = Arc::new(SystemCatalog::new());
        catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
        let crud = Arc::new(CrudStore::new());
        let vector: Arc<dyn arcgraph_storage::vector_store::VectorPageStoreHandle> =
            Arc::new(NoopVectorStore);
        let bm25: Arc<dyn arcgraph_storage::mutation_log::Bm25IndexStoreHandle> =
            Arc::new(NoopBm25Store);
        let router = Arc::new(MultiTenantRouter::new_with_bm25(
            catalog,
            Arc::clone(&crud),
            Some(vector),
            Some(bm25),
        ));
        let intern = Arc::new(InternTable::new());
        let sub =
            CrudExecutorSubstrate::new(Arc::clone(&router), Arc::clone(&mgr), Arc::clone(&intern));
        (sub, crud, mgr, router)
    }

    // ─── AC-3 vector tests (3) ────────────────────────────────────

    #[test]
    fn vector_search_returns_hits_when_provider_attached_w26_beta_3() {
        // AC-1 + AC-3 positive case (vector): when the router has
        // a vector handle AND a `SubstrateSearchProvider` is
        // attached via `with_search_provider`, the substrate
        // forwards query args + its resolved snapshot and returns the
        // provider's real ranked hits — NOT `IndexUnavailable`.
        let (sub, _crud, mgr, _router) = fixture_with_attached_substrates();
        let expected_lsn = mgr.current_lsn();
        let canned = vec![
            RankedHit {
                node: NodeView::new(NodeId::new(7), Some(LabelId::new(1))),
                score: 0.95,
            },
            RankedHit {
                node: NodeView::new(NodeId::new(11), Some(LabelId::new(1))),
                score: 0.85,
            },
        ];
        let provider = Arc::new(RecordingSearchProvider::new(canned.clone()));
        let sub =
            sub.with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

        let query = [0.1_f32, 0.2, 0.3, 0.4];
        let hits = sub
            .vector_search(TenantId::DEFAULT, "embedding", &query, 10, Lsn::MAX)
            .expect("vector_search must succeed at W26-β-3 wire-through");
        assert_eq!(hits, canned, "substrate must return provider hits verbatim");

        // Assert the provider was called with the args the substrate
        // received. `Lsn::MAX` resolves once to the actual CRUD
        // transaction snapshot before provider dispatch.
        let call = provider.take_last_call().expect("provider was called");
        assert_eq!(call.kind, "vector");
        assert_eq!(call.tenant, TenantId::DEFAULT);
        assert_eq!(call.property, "embedding");
        assert_eq!(call.k, 10);
        assert_eq!(call.read_lsn, expected_lsn);
        assert_eq!(call.query_vec_len, Some(query.len()));
    }

    /// **#830 D4 — active verification (ADR-133 §D-4 Query class).**
    ///
    /// Drives `CALL db.index.vector.queryNodes(...) YIELD node, score
    /// RETURN node, score` through the FULL arcgraph-query executor
    /// (parse → bind → type-check → cross-substrate → lower →
    /// materialize) against the REAL [`CrudExecutorSubstrate`] — NOT the
    /// arcgraph-query `StubExecutorSubstrate` the in-crate proc-body unit
    /// tests use. This is the "run the real served `vector_search` path
    /// at least once" gate: it proves the new proc-body reaches the
    /// production substrate's `vector_search` (router route → vector
    /// handle check → `SubstrateSearchProvider` dispatch) and surfaces
    /// the provider's ranked hits as `(node, score)` rows, with the
    /// advisory index name resolved to the served `"embedding"` property
    /// and `k` forwarded verbatim.
    #[test]
    fn query_nodes_proc_drives_real_served_vector_search_cz830() {
        use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
        use arcgraph_query::semantic::{
            BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
        };
        use arcgraph_query::{materialize, parse};

        let (sub, _crud, _mgr, _router) = fixture_with_attached_substrates();
        let canned = vec![
            RankedHit {
                node: NodeView::new(NodeId::new(7), Some(LabelId::new(1))),
                score: 0.95,
            },
            RankedHit {
                node: NodeView::new(NodeId::new(11), Some(LabelId::new(1))),
                score: 0.85,
            },
        ];
        let provider = Arc::new(RecordingSearchProvider::new(canned.clone()));
        let sub =
            sub.with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

        // Build the plan via the real front-end. StubCatalogProvider's
        // tenant is TenantId::DEFAULT — matches the fixture's handle.
        let q = "CALL db.index.vector.queryNodes('cztest', 2, [0.1, 0.2, 0.3, 0.4]) \
                 YIELD node, score RETURN node, score";
        let cat = StubCatalogProvider::new();
        let stmt = parse(q).expect("parse");
        let mut bound = BindingVisitor::bind(&stmt, q, &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
        CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate");
        let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");

        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let result = materialize::materialize(&plan, &sub, &ctx).expect("materialize");

        // The proc-body emitted the served provider's hits as (node, score).
        assert_eq!(
            result.rows().to_vec(),
            vec![
                vec![
                    Value::Node(NodeView::new(NodeId::new(7), Some(LabelId::new(1)))),
                    Value::Float(0.95),
                ],
                vec![
                    Value::Node(NodeView::new(NodeId::new(11), Some(LabelId::new(1)))),
                    Value::Float(0.85),
                ],
            ],
            "queryNodes must emit the served provider's ranked hits as (node, score) rows"
        );

        // Prove the REAL served dispatch happened: the proc-body called
        // CrudExecutorSubstrate::vector_search → SubstrateSearchProvider,
        // resolving the advisory index name to the served "embedding"
        // property and forwarding k=2 + the 4-dim query vector verbatim.
        let call = provider
            .take_last_call()
            .expect("served provider was called");
        assert_eq!(call.kind, "vector");
        assert_eq!(call.tenant, TenantId::DEFAULT);
        assert_eq!(
            call.property, "embedding",
            "advisory index name → served vector property"
        );
        assert_eq!(call.k, 2, "the proc forwarded k=2 to the served substrate");
        assert_eq!(
            call.query_vec_len,
            Some(4),
            "the 4-dim query vector reached the served substrate"
        );
    }

    #[test]
    fn vector_search_returns_index_unavailable_when_provider_not_attached_w26_beta_3() {
        // AC-3 negative case (vector — wired-substrate path): when
        // the router has a vector handle BUT no
        // `SubstrateSearchProvider` is bound, the substrate
        // STRUCTURALLY surfaces `IndexUnavailable` (with a detail
        // that names the binder) — NOT silent-empty hits per
        // `feedback_review_oracle_relaxations.md`. This guards
        // against a misconfigured production bootstrap that wires
        // the router but forgets to call `with_search_provider`.
        let (sub, _crud, _mgr, _router) = fixture_with_attached_substrates();
        // Deliberately NOT calling with_search_provider.
        let r = sub.vector_search(TenantId::DEFAULT, "embedding", &[0.0], 5, Lsn::MAX);
        match r {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert!(
                    detail.contains("provider not attached"),
                    "router-attached + provider-unbound must surface \
                     `provider not attached`; got {detail:?}"
                );
                assert!(
                    detail.contains("ADR-132"),
                    "the structured detail MUST point at the ADR slot \
                     so a misconfigured deployment surfaces the \
                     resolution path; got {detail:?}"
                );
            }
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn vector_search_forwards_exact_available_lsn_w26_beta_3() {
        // AC-3 snapshot consistency (vector): an exact finite snapshot
        // that is currently available is forwarded unchanged.
        let (sub, _crud, mgr, _router) = fixture_with_attached_substrates();
        let requested = mgr.current_lsn();
        let provider = Arc::new(RecordingSearchProvider::new(vec![]));
        let sub =
            sub.with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

        let _ = sub
            .vector_search(TenantId::DEFAULT, "embedding", &[0.0], 3, requested)
            .expect("provider returns empty hits at the exact available snapshot");
        let call = provider.take_last_call().expect("provider was called");
        assert_eq!(
            call.read_lsn, requested,
            "substrate MUST forward the exact available read_lsn; got {:?}",
            call.read_lsn
        );
    }

    #[test]
    fn search_rejects_unavailable_snapshot_before_provider_dispatch() {
        let (sub, crud, mgr, _router) = fixture_with_attached_substrates();
        let requested = mgr.current_lsn();
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Empty,
        )
        .expect("advance snapshot with one node");
        commit(tx, &crud).expect("commit snapshot advance");
        let available = mgr.current_lsn();
        assert_ne!(requested, available);

        let provider = Arc::new(RecordingSearchProvider::new(vec![]));
        let sub =
            sub.with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);
        let expected = SubstrateAccessError::SnapshotUnavailable {
            requested,
            available,
        };

        let vector_error = sub
            .vector_search(TenantId::DEFAULT, "embedding", &[0.0], 3, requested)
            .expect_err("vector search must reject an unavailable snapshot");
        assert_eq!(vector_error, expected);
        let bm25_error = sub
            .bm25_search(TenantId::DEFAULT, "content", "alice", 3, requested)
            .expect_err("BM25 search must reject an unavailable snapshot");
        assert_eq!(bm25_error, expected);
        assert!(
            provider.take_last_call().is_none(),
            "an unavailable snapshot must fail before provider dispatch"
        );
    }

    // ─── AC-3 BM25 tests (3) ──────────────────────────────────────

    #[test]
    fn bm25_search_returns_hits_when_provider_attached_w26_beta_3() {
        // AC-2 + AC-3 positive case (BM25): symmetric to the
        // `vector_search_returns_hits_when_provider_attached_w26_beta_3`
        // test above. The substrate forwards `(tenant, property,
        // query_text, k)` unchanged with the resolved read LSN + returns the
        // provider's real hits.
        let (sub, _crud, mgr, _router) = fixture_with_attached_substrates();
        let expected_lsn = mgr.current_lsn();
        let canned = vec![RankedHit {
            node: NodeView::new(NodeId::new(42), Some(LabelId::new(2))),
            score: 12.5,
        }];
        let provider = Arc::new(RecordingSearchProvider::new(canned.clone()));
        let sub =
            sub.with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

        let hits = sub
            .bm25_search(
                TenantId::DEFAULT,
                "content",
                "rust storage engine",
                10,
                Lsn::MAX,
            )
            .expect("bm25_search must succeed at W26-β-3 wire-through");
        assert_eq!(hits, canned);

        let call = provider.take_last_call().expect("provider was called");
        assert_eq!(call.kind, "bm25");
        assert_eq!(call.tenant, TenantId::DEFAULT);
        assert_eq!(call.property, "content");
        assert_eq!(call.k, 10);
        assert_eq!(call.read_lsn, expected_lsn);
        assert_eq!(call.query_text.as_deref(), Some("rust storage engine"));
    }

    #[test]
    fn bm25_search_returns_index_unavailable_when_provider_not_attached_w26_beta_3() {
        // AC-3 negative case (BM25 — wired-substrate path):
        // symmetric to the vector test above.
        let (sub, _crud, _mgr, _router) = fixture_with_attached_substrates();
        let r = sub.bm25_search(TenantId::DEFAULT, "content", "alice", 5, Lsn::MAX);
        match r {
            Err(SubstrateAccessError::IndexUnavailable(detail)) => {
                assert!(
                    detail.contains("provider not attached"),
                    "router-attached + provider-unbound must surface \
                     `provider not attached`; got {detail:?}"
                );
                assert!(
                    detail.contains("ADR-132"),
                    "the structured detail MUST point at the ADR slot; \
                     got {detail:?}"
                );
            }
            other => panic!("expected IndexUnavailable, got {other:?}"),
        }
    }

    #[test]
    fn bm25_search_forwards_exact_available_lsn_w26_beta_3() {
        // AC-3 snapshot consistency (BM25): symmetric to the vector
        // test above. The substrate forwards the exact available LSN
        // to the provider; production providers (e.g.,
        // `Bm25IndexHandle::search(query, k, read_lsn)` per ADR-039
        // §D-3) honor it natively.
        let (sub, _crud, mgr, _router) = fixture_with_attached_substrates();
        let requested = mgr.current_lsn();
        let provider = Arc::new(RecordingSearchProvider::new(vec![]));
        let sub =
            sub.with_search_provider(Arc::clone(&provider) as Arc<dyn SubstrateSearchProvider>);

        let _ = sub
            .bm25_search(TenantId::DEFAULT, "content", "alice", 3, requested)
            .expect("provider returns empty hits at the exact available snapshot");
        let call = provider.take_last_call().expect("provider was called");
        assert_eq!(
            call.read_lsn, requested,
            "substrate MUST forward the exact available read_lsn to BM25; got {:?}",
            call.read_lsn
        );
    }

    // ─── ADR-147 W26-θ Phase 1 — CREATE node production wire-through ─

    #[test]
    fn create_node_persists_via_crud_executor_substrate() {
        // ADR-147 W26-θ Phase 1: CREATE node opens a per-tenant
        // Transaction, calls crud::create_node, commits — the
        // resulting node MUST be observable via scan_nodes.
        let (sub, _crud, _mgr, _router) = fixture();
        let node_id = sub
            .create_node(TenantId::DEFAULT, Some("User"), &[], &tctx())
            .expect("create_node succeeds");
        // The new node-id must be non-zero (the CrudStore allocator
        // starts at 1).
        assert!(node_id.raw() > 0, "allocated node id is non-zero");
        // scan_nodes must observe the new node.
        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan_nodes after CREATE");
        assert_eq!(rows.len(), 1, "exactly one node after CREATE");
        assert_eq!(rows[0].node.id, node_id);
    }

    #[test]
    fn create_node_interns_label_via_intern_table() {
        // ADR-147 §D-7: the substrate's `intern_table` MUST be
        // consulted to resolve the label name to a LabelId. Two
        // CREATEs with the SAME label name produce nodes sharing
        // the SAME label_id.
        let (sub, _crud, _mgr, _router) = fixture();
        let n1 = sub
            .create_node(TenantId::DEFAULT, Some("User"), &[], &tctx())
            .expect("create n1");
        let n2 = sub
            .create_node(TenantId::DEFAULT, Some("User"), &[], &tctx())
            .expect("create n2");
        assert_ne!(n1, n2, "fresh ids per call");
        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan after 2 CREATEs");
        assert_eq!(rows.len(), 2);
        // Both nodes share the same label_id (interned via the same
        // tenant + name).
        let l1 = rows[0].node.label;
        let l2 = rows[1].node.label;
        assert_eq!(l1, l2, "shared label_id from intern_table");
        assert!(l1.is_some(), "label_id was interned");
    }

    #[test]
    fn create_node_distinct_labels_get_distinct_label_ids() {
        let (sub, _crud, _mgr, _router) = fixture();
        let _ = sub
            .create_node(TenantId::DEFAULT, Some("User"), &[], &tctx())
            .expect("create User");
        let _ = sub
            .create_node(TenantId::DEFAULT, Some("Article"), &[], &tctx())
            .expect("create Article");
        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan");
        assert_eq!(rows.len(), 2);
        let labels: std::collections::HashSet<_> = rows.iter().map(|b| b.node.label).collect();
        assert_eq!(
            labels.len(),
            2,
            "two distinct labels → two distinct LabelIds"
        );
    }

    #[test]
    fn create_node_with_no_label_uses_zero_label_id() {
        // `CREATE (n {...})` carries no label — the substrate stores
        // `LabelId::new(0)` (the sentinel for label-less nodes per the
        // existing v1.0-α convention).
        let (sub, _crud, _mgr, _router) = fixture();
        let _ = sub
            .create_node(TenantId::DEFAULT, None, &[], &tctx())
            .expect("create label-less");
        let rows = sub
            .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
            .expect("scan");
        assert_eq!(rows.len(), 1);
        // The scan_nodes maps label_id 0 → None per the existing
        // CrudExecutorSubstrate::scan_nodes posture.
        assert_eq!(rows[0].node.label, None, "label-less node round-trips");
    }

    #[test]
    fn create_node_filterable_by_label_at_scan_time() {
        // ADR-147 §D-9: CREATE node + MATCH-by-label round-trip is
        // the load-bearing v1.0-α property (the property-bag
        // round-trip is forward-pinned to v1.2).
        let (sub, _crud, _mgr, _router) = fixture();
        let user_id = sub
            .create_node(TenantId::DEFAULT, Some("User"), &[], &tctx())
            .expect("create User");
        let _ = sub
            .create_node(TenantId::DEFAULT, Some("Article"), &[], &tctx())
            .expect("create Article");
        // Resolve the "User" label_id via the intern_table.
        let user_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "User")
            .unwrap();
        let user_rows = sub
            .scan_nodes(TenantId::DEFAULT, Some(user_label), Lsn::MAX)
            .expect("scan by User label");
        assert_eq!(user_rows.len(), 1, "exactly one User node");
        assert_eq!(user_rows[0].node.id, user_id);
    }

    #[test]
    fn create_node_unknown_tenant_returns_routing_error() {
        // Defense-in-depth: a routing miss (tenant not registered)
        // must surface a structured `TenantUnknown` error, not a
        // silent succeed.
        let (sub, _crud, _mgr, _router) = fixture();
        // The fixture only sets up TenantId::DEFAULT; an alternate
        // tenant is unknown.
        let alt = TenantId::new(99);
        let r = sub.create_node(alt, Some("User"), &[], &tctx());
        match r {
            Err(SubstrateAccessError::TenantUnknown(_)) => {} // expected
            other => panic!("expected TenantUnknown, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // NN-4 (#1384) — concurrent MERGE double-create serialization.
    //
    // These tests drive the FULL query front-end (parse → bind →
    // type-check → cross-substrate → lower → execute) against the SHARED,
    // `Arc`-cloned production `CrudExecutorSubstrate` — the same seam the
    // Bolt / MCP server executes on — so the REAL `MergeOp` critical
    // section + the REAL per-(tenant, key) `merge_guard` lock table are
    // exercised. `StubExecutorSubstrate` cannot reproduce the race (it is
    // an in-memory, per-instance fixture with no shared MVCC + no snapshot
    // isolation), so this bug is only observable at the production seam.
    // ─────────────────────────────────────────────────────────────────

    /// Build the `MERGE (u:User {email:$e}) RETURN u` plan against a
    /// catalog whose `User` label id MATCHES the production intern-table
    /// id (so the lowered match branch is a real `Scan{Some(User)}` +
    /// property-filter that queries the substrate — NOT the provably-empty
    /// `LogicalEmpty` the uninterned-label case lowers to).
    ///
    /// NOTE: the email is an inline STRING LITERAL, not a `$e` parameter.
    /// On this base (main) the type-checker enforces ADR-147 §D-4
    /// literal-only inline MERGE/CREATE property values
    /// (`CreatePropertyValueNotLiteral`); the `$e` parameter form from the
    /// issue title lands with the ADR-147-amendment-03 non-literal-value
    /// lift (#1374, in-flight, not yet on main). The `resolve_merge_key`
    /// path is nonetheless PARAMETER-READY (it evaluates the value
    /// expression against the bound bag via `eval::evaluate`), so the
    /// literal and parameter forms produce the SAME merge key for the same
    /// resolved value once that lift lands.
    fn build_merge_user_email_plan(
        user_label: LabelId,
        email: &str,
    ) -> arcgraph_query::logical_plan::LogicalPlan {
        use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
        use arcgraph_query::parse;
        use arcgraph_query::semantic::{
            BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
        };

        // Single-quote the literal; raced emails are test-controlled ASCII
        // with no quotes, so no escaping is needed.
        let q = format!("MERGE (u:User {{email: '{email}'}}) RETURN u");
        let cat = StubCatalogProvider::new().with_label_id("User", user_label);
        let stmt = parse(&q).expect("parse MERGE");
        let mut bound = BindingVisitor::bind(&stmt, &q, &cat).expect("bind MERGE");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check MERGE");
        CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate MERGE");
        LogicalPlanLoweringVisitor::lower(&bound).expect("lower MERGE")
    }

    /// **NN-4 (#1384) re-spin, Fix 4** — build a plan for
    /// `MERGE (n:N {v: <value_literal>}) RETURN n` where `value_literal` is
    /// interpolated VERBATIM (e.g. `9007199254740993` for an Integer, or
    /// `9007199254740992.0` for a Float). Used by the adversarial
    /// large-i64-vs-rounded-float race: two plans built with an Integer and
    /// its rounded-Float image must resolve to the SAME merge lock key so
    /// the concurrent get-or-create serializes to ONE node.
    fn build_merge_num_value_plan(
        n_label: LabelId,
        value_literal: &str,
    ) -> arcgraph_query::logical_plan::LogicalPlan {
        use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
        use arcgraph_query::parse;
        use arcgraph_query::semantic::{
            BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
        };

        let q = format!("MERGE (n:N {{v: {value_literal}}}) RETURN n");
        let cat = StubCatalogProvider::new().with_label_id("N", n_label);
        let stmt = parse(&q).expect("parse numeric MERGE");
        let mut bound = BindingVisitor::bind(&stmt, &q, &cat).expect("bind numeric MERGE");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check numeric MERGE");
        CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate numeric MERGE");
        LogicalPlanLoweringVisitor::lower(&bound).expect("lower numeric MERGE")
    }

    /// Count committed `N` nodes whose `v` property equals `Integer(i)`
    /// OR `Float(f)` — i.e. either the integer or its rounded-float image.
    /// Used to assert the adversarial Integer/Float MERGE race collapsed to
    /// ONE node regardless of which racer won the create.
    fn count_n_with_int_or_float(
        sub: &CrudExecutorSubstrate,
        n_label: LabelId,
        i: i64,
        f: f64,
    ) -> usize {
        sub.scan_nodes(TenantId::DEFAULT, Some(n_label), Lsn::MAX)
            .expect("scan N nodes")
            .iter()
            .filter(|bn| {
                matches!(bn.node.properties.get("v"),
                    Some(Value::Integer(x)) if *x == i)
                    || matches!(bn.node.properties.get("v"),
                        Some(Value::Float(y)) if *y == f)
            })
            .count()
    }

    /// Recursively strip the `merge_keys` from every `LogicalMerge` in the
    /// plan → empty. This NEUTERS the NN-4 fix (the executor then runs the
    /// match→create span WITHOUT the critical section — byte-identical to
    /// the pre-fix code path), which is how the RED-on-revert control below
    /// proves BOTH that the race is real AND that the merge-key
    /// serialization is what closes it.
    fn strip_merge_key(
        plan: arcgraph_query::logical_plan::LogicalPlan,
    ) -> arcgraph_query::logical_plan::LogicalPlan {
        use arcgraph_query::logical_plan::LogicalPlan as LP;
        match plan {
            LP::Merge(mut m) => {
                m.merge_keys = Vec::new();
                m.match_branch = Box::new(strip_merge_key(*m.match_branch));
                m.create_branch = Box::new(strip_merge_key(*m.create_branch));
                LP::Merge(m)
            }
            LP::Project(mut p) => {
                p.input = Box::new(strip_merge_key(*p.input));
                LP::Project(p)
            }
            other => other,
        }
    }

    /// Count committed `User` nodes whose `email` property equals `email`,
    /// read at the latest committed snapshot.
    fn count_users_with_email(
        sub: &CrudExecutorSubstrate,
        user_label: LabelId,
        email: &str,
    ) -> usize {
        sub.scan_nodes(TenantId::DEFAULT, Some(user_label), Lsn::MAX)
            .expect("scan User nodes")
            .iter()
            .filter(|bn| bn.node.properties.get("email") == Some(&Value::String(email.into())))
            .count()
    }

    /// **NN-4 test seam** — a decorator [`ExecutorSubstrate`] that
    /// delegates every call to an inner production substrate BUT rendezvous
    /// all racing threads at the MERGE match-scan.
    ///
    /// Its `scan_nodes_with_context` (the `MergeOp` match probe) runs the
    /// REAL inner scan, then WAITS on a `Barrier` before returning. So all
    /// N threads hold their (empty) match result at the SAME instant —
    /// every thread then takes the create branch. This DETERMINISTICALLY
    /// forces the concurrent-double-create window WITHOUT sleeps or any
    /// production test-hook: it is a pure test decorator around the real
    /// substrate. (This is why the RED-on-revert control is deterministic:
    /// the raw wall-clock race is too tight to observe reliably — the first
    /// thread's create commits before the others scan — so we pin the
    /// interleaving with an out-of-band rendezvous.)
    ///
    /// It is used ONLY on the UNSERIALIZED (merge-key-stripped) path: with
    /// the fix active, N-1 threads block on `merge_guard` BEFORE the scan,
    /// so only one reaches this barrier and an N-way rendezvous would
    /// deadlock. The fixed (headline) path therefore relies on the lock's
    /// unconditional exactly-one guarantee, not on a forced interleaving.
    struct ScanBarrierSubstrate {
        inner: Arc<CrudExecutorSubstrate>,
        scan_barrier: Arc<std::sync::Barrier>,
        /// When true, `merge_guard` returns `Ok(None)` — simulating the
        /// UN-FIXED production substrate (the fix's OTHER half is the real
        /// per-key lock table in `CrudExecutorSubstrate::merge_guard`). Lets
        /// the RED-on-revert control prove the PRODUCTION guard is
        /// load-bearing even when the plan's merge_key is present.
        neuter_merge_guard: bool,
    }

    impl ExecutorSubstrate for ScanBarrierSubstrate {
        fn scan_nodes(
            &self,
            tenant: TenantId,
            label: Option<LabelId>,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            self.inner.scan_nodes(tenant, label, read_lsn)
        }

        fn scan_nodes_with_context(
            &self,
            ctx: &ExecutionContext,
            label: Option<LabelId>,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            // Run the REAL match scan first (each thread observes the store
            // BEFORE any racer's create commits, since all are pre-barrier)…
            let out = self.inner.scan_nodes_with_context(ctx, label, read_lsn)?;
            // …then rendezvous: block until ALL racing threads have their
            // match result in hand. When the barrier releases, every thread
            // has seen 0 matching rows → every thread takes the create
            // branch → the unserialized path double-creates deterministically.
            self.scan_barrier.wait();
            Ok(out)
        }

        fn expand(
            &self,
            tenant: TenantId,
            from: NodeId,
            rel_type: Option<TypeId>,
            direction: Direction,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
            self.inner
                .expand(tenant, from, rel_type, direction, read_lsn)
        }

        fn create_node(
            &self,
            tenant: TenantId,
            label: Option<&str>,
            properties: &[(String, Value)],
            ctx: &ExecutionContext,
        ) -> Result<NodeId, SubstrateAccessError> {
            self.inner.create_node(tenant, label, properties, ctx)
        }

        fn merge_guard(
            &self,
            tenant: TenantId,
            key: &str,
        ) -> Result<Option<Box<dyn MergeGuard>>, SubstrateAccessError> {
            if self.neuter_merge_guard {
                // Simulate the UN-FIXED production substrate: no lock. The
                // plan's merge_key is present (MergeOp DOES call this), but
                // returning None means no serialization → double-create.
                return Ok(None);
            }
            self.inner.merge_guard(tenant, key)
        }

        fn vector_search(
            &self,
            tenant: TenantId,
            property: &str,
            query_vec: &[f32],
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            self.inner
                .vector_search(tenant, property, query_vec, k, read_lsn)
        }

        fn bm25_search(
            &self,
            tenant: TenantId,
            property: &str,
            query: &str,
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            self.inner.bm25_search(tenant, property, query, k, read_lsn)
        }

        fn community_members(
            &self,
            tenant: TenantId,
            community_id: i64,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            self.inner.community_members(tenant, community_id, read_lsn)
        }
    }

    /// Race `thread_count` concurrent `MERGE (u:User {email:'…'})` on the
    /// SAME key. `strip = true` neuters the NN-4 fix AND drives the threads
    /// through the `ScanBarrierSubstrate` so the double-create window is
    /// forced deterministically (all threads match-empty before any
    /// create). `strip = false` (the headline path) runs the REAL fix with
    /// a plain start-barrier and relies on the lock's exactly-one guarantee.
    /// Returns the number of committed `User` nodes carrying the raced email.
    #[derive(Clone, Copy)]
    enum RaceMode {
        /// The REAL fix: merge_key present, production `merge_guard`
        /// returns a real lock. No scan-barrier (the lock serializes; a
        /// scan-barrier would deadlock since N-1 threads block pre-scan).
        Fixed,
        /// Revert-half-1: the plan's merge_key is STRIPPED → `MergeOp`
        /// never calls `merge_guard`. Driven through the scan-barrier so
        /// the double-create window is forced deterministically.
        StripKey,
        /// Revert-half-2: merge_key PRESENT but `merge_guard` returns None
        /// (simulating the un-fixed production substrate). Proves the
        /// PRODUCTION guard's real lock is load-bearing. Scan-barrier-forced.
        NeuterGuard,
    }

    /// Which query-driver entry point each racing thread executes through.
    ///
    /// This distinction is LOAD-BEARING for NN-4 (#1384) re-spin, Fix 1.
    /// The two paths differ in WHEN a MERGE create becomes durable:
    /// - [`Driver::ExecuteWithContext`] — the test-only eager entry point
    ///   ([`arcgraph_query::executor::execute_with_context`]). NO D-2
    ///   statement-txn wrap: each write op auto-commits inside its own
    ///   `next_batch`, so the create is durable at `next_batch` return.
    /// - [`Driver::Materialize`] — the SHIPPED auto-commit path
    ///   ([`arcgraph_query::materialize::materialize`], reached by Bolt
    ///   RUN / MCP `graph.raw_query`). The D-2 wrap opens ONE statement
    ///   txn: a MERGE create only STAGES inside `next_batch` and COMMITS at
    ///   `commit_statement` — AFTER `next_batch` returns. The pre-respin
    ///   guard (dropped at `next_batch` return) therefore released BEFORE
    ///   the node was durable → the loser re-probed an uncommitted snapshot
    ///   → double-create. This is the exact ultracode-verify REJECT bug;
    ///   the production-path test below drives THIS path so the fix is
    ///   verified where it actually ships.
    #[derive(Clone, Copy)]
    enum Driver {
        ExecuteWithContext,
        Materialize,
    }

    /// Drive `plan` through the selected [`Driver`], panicking on any
    /// executor error. Both drivers execute the SAME `MergeOp`; they
    /// differ only in the statement-txn commit shape (see [`Driver`]).
    fn run_merge<S: ExecutorSubstrate>(
        driver: Driver,
        plan: &arcgraph_query::logical_plan::LogicalPlan,
        sub: &S,
        ctx: &ExecutionContext,
    ) {
        match driver {
            Driver::ExecuteWithContext => {
                arcgraph_query::executor::execute_with_context(plan, sub, ctx)
                    .expect("MERGE executes without error (execute_with_context)");
            }
            Driver::Materialize => {
                arcgraph_query::materialize::materialize(plan, sub, ctx)
                    .expect("MERGE executes without error (materialize)");
            }
        }
    }

    fn race_merge_same_key(thread_count: usize, mode: RaceMode) -> usize {
        race_merge_same_key_via(thread_count, mode, Driver::ExecuteWithContext)
    }

    fn race_merge_same_key_via(thread_count: usize, mode: RaceMode, driver: Driver) -> usize {
        use std::sync::Barrier;

        let (sub, _crud, _mgr, _router) = fixture();
        let email = "race@example.com";

        // Pre-intern the `User` label by creating a THROWAWAY User node
        // carrying a DIFFERENT email, so the raced MERGE's match branch
        // lowers to a real `Scan{Some(User)}`+filter (not `LogicalEmpty`).
        // This also proves the fix works when nodes of the label already
        // exist (the realistic get-or-create-into-a-populated-tenant case).
        sub.create_node(
            TenantId::DEFAULT,
            Some("User"),
            &[(
                "email".into(),
                Value::String("someone-else@example.com".into()),
            )],
            &tctx(),
        )
        .expect("pre-create throwaway User");
        let user_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "User")
            .unwrap();
        let sub = Arc::new(sub);

        // Build one plan; each thread clones it (plans are immutable +
        // Send; the per-thread physical pipeline is built inside execute).
        let mut plan = build_merge_user_email_plan(user_label, email);
        if matches!(mode, RaceMode::StripKey) {
            plan = strip_merge_key(plan);
        }
        let plan = Arc::new(plan);
        let barrier_wrapped = matches!(mode, RaceMode::StripKey | RaceMode::NeuterGuard);
        let neuter_guard = matches!(mode, RaceMode::NeuterGuard);
        // The shipped materialize path needs a controlled winner/loser
        // hand-off: a raw start race can let the winner commit before the
        // loser reaches the guard, making a pre-commit guard-drop revert
        // false-PASS. The rendezvous admits one winner first and pins the
        // loser's re-probe against the winner's guard release.
        let rendezvous = if matches!(mode, RaceMode::Fixed) && matches!(driver, Driver::Materialize)
        {
            Some(Arc::new(ConcurrentRaceState::new()))
        } else {
            None
        };

        // Start-barrier: release all threads together. The barrier-wrapped
        // (revert) modes ALSO use a scan-barrier (inside
        // ScanBarrierSubstrate) to force the interleaving; the Fixed path
        // uses only this start-barrier and relies on the lock.
        let start = Arc::new(Barrier::new(thread_count));
        let scan_barrier = Arc::new(Barrier::new(thread_count));

        let handles: Vec<_> = (0..thread_count)
            .map(|_| {
                let start = Arc::clone(&start);
                let scan_barrier = Arc::clone(&scan_barrier);
                let sub = Arc::clone(&sub);
                let plan = Arc::clone(&plan);
                let rendezvous = rendezvous.clone();
                std::thread::spawn(move || {
                    // AUTO-COMMIT mode: each thread's MergeOp probes +
                    // creates in its own per-call txn (either the eager
                    // execute_with_context path or the D-2-wrapped
                    // materialize path — see `Driver`).
                    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                    start.wait();
                    if barrier_wrapped {
                        let wrapped = ScanBarrierSubstrate {
                            inner: Arc::clone(&sub),
                            scan_barrier,
                            neuter_merge_guard: neuter_guard,
                        };
                        run_merge(driver, &plan, &wrapped, &ctx);
                    } else if let Some(state) = rendezvous {
                        let wrapped = ConcurrentRaceSubstrate {
                            inner: Arc::clone(&sub),
                            state,
                            acquired: AtomicBool::new(false),
                            is_loser: AtomicBool::new(false),
                        };
                        run_merge(driver, &plan, &wrapped, &ctx);
                    } else {
                        run_merge(driver, &plan, sub.as_ref(), &ctx);
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().expect("MERGE thread joined");
        }

        count_users_with_email(&sub, user_label, email)
    }

    /// **NN-4 (#1384) — the HEADLINE test.** N=8 threads race
    /// `MERGE (u:User {email:$e})` on the SAME key. With the fix, the
    /// `MergeOp` critical section serializes match→create on the merge key:
    /// exactly ONE thread creates the node; the other 7 BLOCK on
    /// `merge_guard`, re-probe after acquiring, see the winner's node, and
    /// take the match branch. Get-or-create uniqueness holds.
    #[test]
    fn concurrent_merge_same_key_creates_exactly_one_node_nn4() {
        let created = race_merge_same_key(8, RaceMode::Fixed);
        assert_eq!(
            created, 1,
            "NN-4: 8 concurrent MERGE on the SAME key must create EXACTLY \
             ONE node (get-or-create uniqueness) — got {created}"
        );
    }

    /// **NN-4 (#1384) re-spin, Fix 4 — the PRODUCTION-PATH headline test
    /// (MANDATORY).** A controlled two-thread same-key race where both
    /// threads drive
    /// [`arcgraph_query::materialize::materialize`] — the SHIPPED Bolt
    /// RUN / MCP `graph.raw_query` auto-commit path with the D-2
    /// statement-txn wrap — NOT the test-only `execute_with_context`.
    ///
    /// This is the test that the ultracode-verify REJECT proved was
    /// missing: under the D-2 wrap a MERGE create only STAGES inside
    /// `next_batch` and COMMITS at `commit_statement`, AFTER `next_batch`
    /// returns. The PRE-RESPIN guard (dropped at `next_batch` return)
    /// therefore released BEFORE the winner's node was durable → the loser
    /// re-probed an uncommitted snapshot → double-create (reproduced as
    /// created==N in the verdict). With Fix 1 the guard is STASHED on the
    /// ExecutionContext and dropped only AFTER `commit_statement`, so the
    /// loser blocks until the winner's node is COMMITTED, re-probes, sees
    /// it, and takes the match branch → EXACTLY ONE node on the exact path
    /// production ships.
    #[test]
    fn concurrent_merge_same_key_via_materialize_creates_exactly_one_node_nn4() {
        let created = race_merge_same_key_via(2, RaceMode::Fixed, Driver::Materialize);
        assert_eq!(
            created, 1,
            "NN-4 PRODUCTION PATH: 2 concurrent MERGE on the SAME key via \
             materialize::materialize (the shipped D-2 auto-commit path) must \
             create EXACTLY ONE node. created={created} means the guard did \
             NOT span the commit — the ultracode REJECT bug survives."
        );
    }

    /// **NN-4 (#1384) re-spin, Fix 4 — production-path RED-on-revert
    /// control (strip the key, materialize driver).** SAME production-path
    /// race but with the plan's merge key STRIPPED (so `MergeOp` never
    /// requests a guard), driven through the `ScanBarrierSubstrate` that
    /// rendezvous every thread at the match scan. Under the D-2 wrap all N
    /// threads stage + then commit their creates → N duplicate nodes. This
    /// proves the race is REAL on the production path (not just on the
    /// `execute_with_context` path the pre-respin control used) AND that
    /// the merge-key serialization is what closes it.
    #[test]
    fn concurrent_merge_stripped_key_via_materialize_double_creates_nn4() {
        let n = 8;
        let created = race_merge_same_key_via(n, RaceMode::StripKey, Driver::Materialize);
        assert_eq!(
            created, n,
            "NN-4 production-path RED-on-revert (strip-key): WITHOUT the \
             merge-key critical section the scan-barrier forces all {n} \
             threads to match-empty then create → {n} duplicate nodes on the \
             materialize path; got {created}."
        );
    }

    /// **NN-4 (#1384) re-spin, Fix 4 — production-path RED-on-revert
    /// control (neuter the guard, materialize driver).** SAME production
    /// path with the merge key PRESENT but the wrapper's `merge_guard`
    /// returning `Ok(None)` — the UN-FIXED production substrate. Proves the
    /// real per-key lock table is load-bearing on the shipped path too.
    #[test]
    fn concurrent_merge_neutered_guard_via_materialize_double_creates_nn4() {
        let n = 8;
        let created = race_merge_same_key_via(n, RaceMode::NeuterGuard, Driver::Materialize);
        assert_eq!(
            created, n,
            "NN-4 production-path RED-on-revert (neuter-guard): with the \
             production merge_guard returning None, {n} concurrent MERGE via \
             materialize double-create → {n} nodes; got {created}."
        );
    }

    /// **NN-4 (#1384) re-spin, Fix 4 — adversarial large-i64-vs-rounded-float
    /// race (the verdict's reproducing scenario).** Half the threads run
    /// `MERGE (n:N {v: 9007199254740993})` (Integer 2^53+1) and half run
    /// `MERGE (n:N {v: 9007199254740992.0})` (Float 2^53). These two values
    /// are EQUAL to the match filter (`(2^53+1) as f64 == 2^53.0`), so a
    /// correct get-or-create MUST create EXACTLY ONE node — every thread's
    /// match probe sees the winner's node regardless of which literal form
    /// won the create.
    ///
    /// The bug (F4 residual): `canonicalize_key_value` used to render the
    /// Integer verbatim (`I:...993`) and normalize the Float to
    /// `Integer(2^53)` (`I:...992`) → the two lock keys SPLIT → the Integer
    /// threads and the Float threads took DIFFERENT mutexes → both cohorts
    /// created → DUPLICATE. Fix 4 routes both through the FLOAT bucket
    /// (`F:<(v as f64).to_bits()>`), so all threads acquire the SAME mutex
    /// and serialize to ONE node.
    ///
    /// This is the direct concurrency proof of the boundary regression that
    /// [`resolve_merge_key_large_int_vs_rounded_float_collide_above_2p53_nn4`]
    /// asserts at the pure key-resolution layer. It drives the SHIPPED
    /// `materialize::materialize` auto-commit path.
    #[test]
    fn adversarial_large_i64_vs_rounded_float_false_split_nn4() {
        use std::sync::Barrier;

        const P53: i64 = 1_i64 << 53; // 9_007_199_254_740_992
        let int_lit = (P53 + 1).to_string(); // "9007199254740993"
        // f64 literal for 2^53 — MUST carry a decimal point so it parses as
        // a Float, not an Integer.
        let float_lit = format!("{:.1}", P53 as f64); // "9007199254740992.0"

        let (sub, _crud, _mgr, _router) = fixture();
        // Pre-intern `N` with a THROWAWAY node (distinct value) so the raced
        // MERGE match branch lowers to a real `Scan{Some(N)}`+filter.
        sub.create_node(
            TenantId::DEFAULT,
            Some("N"),
            &[("v".into(), Value::Integer(1))],
            &tctx(),
        )
        .expect("pre-create throwaway N");
        let n_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "N")
            .unwrap();
        let sub = Arc::new(sub);

        let int_plan = Arc::new(build_merge_num_value_plan(n_label, &int_lit));
        let float_plan = Arc::new(build_merge_num_value_plan(n_label, &float_lit));

        // One Integer racer and one Float racer. The rendezvous below makes
        // this smaller race stronger than the former raw 8-thread stress:
        // if canonicalization splits their keys, both match scans are held
        // empty before either create can commit.
        let total = 2;
        let start = Arc::new(Barrier::new(total));
        let state = Arc::new(ConcurrentRaceState::new());
        let mut handles = Vec::new();
        for t in 0..total {
            let start = Arc::clone(&start);
            let sub = Arc::clone(&sub);
            let state = Arc::clone(&state);
            // Even threads race the Integer literal, odd threads the Float —
            // so both cohorts contend the SAME merge key concurrently.
            let plan = if t % 2 == 0 {
                Arc::clone(&int_plan)
            } else {
                Arc::clone(&float_plan)
            };
            handles.push(std::thread::spawn(move || {
                let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                let wrapped = ConcurrentRaceSubstrate {
                    inner: Arc::clone(&sub),
                    state,
                    acquired: AtomicBool::new(false),
                    is_loser: AtomicBool::new(false),
                };
                start.wait();
                // SHIPPED D-2 auto-commit path.
                arcgraph_query::materialize::materialize(&plan, &wrapped, &ctx)
                    .expect("numeric MERGE executes");
            }));
        }
        for h in handles {
            h.join().expect("MERGE thread joined");
        }

        let created = count_n_with_int_or_float(&sub, n_label, P53 + 1, P53 as f64);
        // Attribution breadcrumbs for any failure: whether the harness saw
        // the keys split (the F4-residual regime) and what the loser's
        // pre-filter re-probe scan returned (1 = throwaway only ⇒ the
        // winner's commit was NOT visible; ≥2 ⇒ it was).
        let keys_split = state.keys_split.load(Ordering::Acquire);
        let reprobe = state.loser_reprobe_rows.load(Ordering::Acquire);
        assert_eq!(
            created, 1,
            "NN-4 Fix 4: {total} concurrent MERGE mixing Integer(2^53+1) and \
             Float(2^53.0) — which the match filter calls EQUAL — MUST create \
             EXACTLY ONE node. created={created} means the lock keys SPLIT \
             above 2^53 (the F4 residual): the Integer cohort and the Float \
             cohort took DIFFERENT mutexes → double-create. RED against the \
             small-int-only canonicalize (`I:...993` vs normalized `I:...992`). \
             [diagnostics: keys_split={keys_split}, loser pre-filter re-probe \
             rows={reprobe:?} (usize::MAX = never recorded)]"
        );
    }

    /// **NN-4 (#1384) — RED-on-revert control (half 1: strip the key).**
    /// SAME 8-thread race, but with the plan's merge key STRIPPED
    /// (`strip_merge_key`) so `MergeOp` NEVER calls `merge_guard` — i.e.
    /// the executor runs the match→create span WITHOUT the critical
    /// section, byte-identical to the PRE-FIX code path — AND driven
    /// through the `ScanBarrierSubstrate` that rendezvous every thread at
    /// the match scan (a pure test decorator, no sleeps, no production
    /// hook). All 8 threads therefore see 0 match rows at the SAME instant
    /// → all 8 take the create branch → 8 duplicate nodes. The OCC
    /// commit-check only iterates write keys (disjoint fresh node ids) so
    /// every create commits. Proves the race is REAL and the merge-key
    /// serialization is what closes it.
    #[test]
    fn concurrent_merge_stripped_key_double_creates_nn4() {
        let n = 8;
        let created = race_merge_same_key(n, RaceMode::StripKey);
        assert_eq!(
            created, n,
            "NN-4 RED-on-revert (strip-key): WITHOUT the merge-key critical \
             section, the scan-barrier forces all {n} threads to match-empty \
             then create → {n} duplicate nodes; got {created}. Anything < \
             {n} means the forced interleaving did not hold; = 1 means the \
             race did not reproduce and the headline test is not load-bearing."
        );
    }

    /// **NN-4 (#1384) — RED-on-revert control (half 2: neuter the
    /// production guard).** SAME 8-thread race with the merge key PRESENT
    /// (so `MergeOp` DOES call `merge_guard`), but the wrapper's
    /// `merge_guard` returns `Ok(None)` — simulating the UN-FIXED
    /// production `CrudExecutorSubstrate::merge_guard`. This directly proves
    /// the PRODUCTION lock table is the load-bearing half: with the real
    /// guard (headline test) → 1 node; with the guard neutered → 8 nodes.
    /// (The two halves — the plan's merge_key AND the substrate's real
    /// guard — are both required; neutering EITHER re-opens the race.)
    #[test]
    fn concurrent_merge_neutered_guard_double_creates_nn4() {
        let n = 8;
        let created = race_merge_same_key(n, RaceMode::NeuterGuard);
        assert_eq!(
            created, n,
            "NN-4 RED-on-revert (neuter-guard): with the production \
             merge_guard returning None, {n} concurrent MERGE double-create \
             → {n} nodes; got {created}. This is the direct proof the real \
             per-key lock table in CrudExecutorSubstrate::merge_guard is \
             load-bearing."
        );
    }

    /// **NN-4 no-regression — single-threaded get-or-create.** A lone
    /// `MERGE (u:User {email:'…'})` run TWICE must create on the first call
    /// (create branch) and MATCH on the second (no second node): the fix's
    /// critical section is a no-op for the uncontended path. Exercises both
    /// the create branch and the match branch through the serialized path.
    #[test]
    fn single_thread_merge_creates_then_matches_nn4() {
        let (sub, _crud, _mgr, _router) = fixture();
        // Pre-intern `User` so the match branch is a real Scan+filter.
        sub.create_node(
            TenantId::DEFAULT,
            Some("User"),
            &[("email".into(), Value::String("other@example.com".into()))],
            &tctx(),
        )
        .expect("pre-create");
        let user_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "User")
            .unwrap();
        let email = "solo@example.com";
        let plan = build_merge_user_email_plan(user_label, email);

        // First MERGE → create branch (no such email yet).
        let ctx1 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let rows1 = arcgraph_query::executor::execute_with_context(&plan, &sub, &ctx1)
            .expect("first MERGE");
        assert_eq!(rows1.len(), 1, "MERGE emits the created binding row");
        assert_eq!(
            count_users_with_email(&sub, user_label, email),
            1,
            "first MERGE creates exactly one node"
        );

        // Second MERGE on the SAME key → match branch (no new node).
        let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let rows2 = arcgraph_query::executor::execute_with_context(&plan, &sub, &ctx2)
            .expect("second MERGE");
        assert_eq!(rows2.len(), 1, "MERGE emits the matched binding row");
        assert_eq!(
            count_users_with_email(&sub, user_label, email),
            1,
            "second MERGE takes the match branch — still exactly one node"
        );
    }

    /// **NN-4 no-deadlock — merge under a distinct concurrent commit.**
    /// While N threads race a MERGE on key A, another N threads
    /// concurrently `create_node` under a DIFFERENT label (a plain commit
    /// path). The merge-key lock is strictly OUTER of the MVCC commit gate
    /// (no commit path acquires a merge-key lock), so this cannot deadlock;
    /// the test asserts the whole barrier-synchronized mix COMPLETES (join
    /// returns) and get-or-create still holds for key A. A deadlock would
    /// hang the test (CI timeout) rather than fail an assert — the join
    /// completing IS the no-deadlock signal.
    #[test]
    fn merge_and_concurrent_commit_do_not_deadlock_nn4() {
        use std::sync::Barrier;

        let (sub, _crud, _mgr, _router) = fixture();
        sub.create_node(
            TenantId::DEFAULT,
            Some("User"),
            &[("email".into(), Value::String("seed@example.com".into()))],
            &tctx(),
        )
        .expect("seed User");
        let user_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "User")
            .unwrap();
        let email = "nodeadlock@example.com";
        let plan = Arc::new(build_merge_user_email_plan(user_label, email));
        let sub = Arc::new(sub);

        let n = 6;
        // 2*n participants: n racing the MERGE, n hammering plain creates.
        let barrier = Arc::new(Barrier::new(2 * n));

        let mut handles = Vec::new();
        for _ in 0..n {
            let barrier = Arc::clone(&barrier);
            let sub = Arc::clone(&sub);
            let plan = Arc::clone(&plan);
            handles.push(std::thread::spawn(move || {
                let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                barrier.wait();
                arcgraph_query::executor::execute_with_context(&plan, sub.as_ref(), &ctx)
                    .expect("MERGE executes");
            }));
        }
        for i in 0..n {
            let barrier = Arc::clone(&barrier);
            let sub = Arc::clone(&sub);
            handles.push(std::thread::spawn(move || {
                // A plain commit path (create under a distinct label) that
                // takes the MVCC commit_gate but NEVER a merge-key lock.
                let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                barrier.wait();
                sub.create_node(
                    TenantId::DEFAULT,
                    Some("Account"),
                    &[("n".into(), Value::Integer(i as i64))],
                    &ctx,
                )
                .expect("plain create commits");
            }));
        }
        for h in handles {
            h.join().expect("no thread panicked / deadlocked");
        }

        assert_eq!(
            count_users_with_email(&sub, user_label, email),
            1,
            "NN-4: get-or-create still exactly one under concurrent commit load"
        );
    }

    /// **NN-4 no-regression — held-txn (explicit transaction) single-txn
    /// get-or-create.** A SINGLE Bolt explicit transaction that MERGEs the
    /// same key TWICE must create once then MATCH (read-your-writes): the
    /// second MERGE's match probe reads the first's STAGED create through
    /// the held txn → match branch → no second node. This proves the NN-4
    /// guard does not break the held-txn path (the guard is a no-op /
    /// harmless there; intra-txn correctness comes from read-your-writes).
    /// Concurrent-explicit-txn MERGE-same-key is a documented SI limitation
    /// (see `MergeOp::next_batch` — forward-deferred to the OCC read-set
    /// approach); this test pins the single-txn correctness the fix keeps.
    #[test]
    fn held_txn_single_transaction_merge_twice_matches_second_nn4() {
        let (sub, _crud, mgr, _router) = fixture();
        // Pre-intern `User` so the match branch is a real Scan+filter.
        sub.create_node(
            TenantId::DEFAULT,
            Some("User"),
            &[("email".into(), Value::String("pre@example.com".into()))],
            &tctx(),
        )
        .expect("pre-create");
        let user_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "User")
            .unwrap();
        let email = "held@example.com";
        let plan = build_merge_user_email_plan(user_label, email);

        // One held (explicit) transaction, installed on the context. Both
        // MERGEs stage into the SAME txn; the executor reads-its-own-writes.
        let owned = mgr.begin_owned(TenantId::DEFAULT);
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
            .with_held_txn(Box::new(BoltHeldTxn::new(owned)));

        // First MERGE → stages a create into the held txn.
        let rows1 = arcgraph_query::executor::execute_with_context(&plan, &sub, &ctx)
            .expect("first held-txn MERGE");
        assert_eq!(rows1.len(), 1, "first MERGE emits the created binding row");

        // Second MERGE on the SAME key → match branch (read-your-writes
        // sees the staged create) → NO second create staged.
        let rows2 = arcgraph_query::executor::execute_with_context(&plan, &sub, &ctx)
            .expect("second held-txn MERGE");
        assert_eq!(rows2.len(), 1, "second MERGE emits the matched binding row");

        // Commit the held txn, then verify exactly ONE new User with this
        // email is durable (the pre-created `pre@` user is a different key).
        let held = ctx
            .take_held_txn()
            .expect("held txn still installed")
            .as_any_mut()
            .downcast_mut::<BoltHeldTxn>()
            .expect("BoltHeldTxn")
            .take_owned()
            .expect("owned txn present");
        sub.commit_held_txn(held).expect("held txn commits");

        assert_eq!(
            count_users_with_email(&sub, user_label, email),
            1,
            "held-txn MERGE-twice on the same key stages exactly one create"
        );
    }

    // ─────────────────────────────────────────────────────────────────
    // NN-4 (#1384) re-spin, Fix 4 — deterministic probe-AFTER-lock
    // rendezvous.
    //
    // The barrier controls above (strip-key / neuter-guard) prove the
    // race is real by NEUTERING the guard entirely — they do NOT exercise
    // the after-lock re-probe invariant (the loser must, once it acquires
    // the guard, re-probe and SEE the winner's COMMITTED node). This test
    // pins that invariant DETERMINISTICALLY: it admits exactly one thread
    // (the winner) past the guard first, holds the loser at a rendezvous
    // AFTER it unblocks from the guard + BEFORE its re-probe, and asserts
    // the loser's re-probe sees the winner's node. It RED-flips if the
    // probe is moved BEFORE the lock (the loser would re-probe an
    // uncommitted / empty snapshot).
    // ─────────────────────────────────────────────────────────────────

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// Shared coordination for the deterministic concurrent race tests.
    ///
    /// # Per-test isolation (review fix-back, #1392)
    ///
    /// One instance is constructed INSIDE each test (`Arc::new(…)`), shared
    /// only by that test's two racer threads, and dropped when the last
    /// racer exits — there is NO static / global / registry state, so
    /// nothing can leak into another test. The exactly-2-racers protocol is
    /// ENFORCED (a 3rd `merge_guard` arrival panics; see
    /// [`ConcurrentRaceSubstrate::merge_guard`]) rather than assumed.
    ///
    /// # Determinism
    ///
    /// The winner's commit-visibility is serialized BEFORE the loser's
    /// snapshot pin by construction: the loser blocks INSIDE
    /// `merge_guard` — which the driver calls BEFORE `begin_statement`
    /// pins the statement snapshot — until the winner's post-commit guard
    /// release. The loser therefore always pins a fresh snapshot AFTER the
    /// winner's node is durable and re-probes it (Fix 1); no scheduler
    /// assumption or sleep anywhere.
    ///
    /// # Bounded waits (no suite-wide hang)
    ///
    /// Every wait is bounded by [`RACE_WAIT_BUDGET`]. An UNBOUNDED wait
    /// would convert one dead racer (panic / substrate error under load)
    /// into a forever-parked thread → a hung test binary (the whole run
    /// under `--test-threads=1`) → cross-test fallout when the runner kills
    /// the stalled suite. Demonstrated by fault injection: with a loser
    /// dying pre-re-probe, the unbounded version hung the binary on ~40% of
    /// runs; the bounded version fails THAT test cleanly every time.
    struct ConcurrentRaceState {
        /// Order-of-acquisition counter: the 0th acquirer is the WINNER,
        /// the 1st is the LOSER. Bumped inside the wrapper's `merge_guard`
        /// AFTER the real (inner) guard is acquired.
        guard_order: AtomicUsize,
        /// Signalled once the WINNER has acquired the real guard, so the
        /// loser only *calls* `merge_guard` after the winner already holds
        /// it — guaranteeing the winner wins the race deterministically
        /// (the loser then BLOCKS on the real inner lock until the winner
        /// commits + drops).
        winner_has_guard: Arc<Barrier2>,
        /// The winner's resolved lock key. The loser compares its key with
        /// this before attempting the real inner lock, then signals
        /// `loser_key_classified`. That lets the winner distinguish the
        /// fixed same-key path from the reverted split-key path without a
        /// timeout or scheduler assumption.
        winner_key: Mutex<Option<String>>,
        /// Signalled after the loser has compared both resolved keys. The
        /// winner waits for this while holding the real guard, guaranteeing
        /// the loser has reached the controlled guard point before the
        /// winner can execute its match/create branch.
        loser_key_classified: Arc<Barrier2>,
        /// True only when the two supposedly equal MERGE values resolved to
        /// different lock keys. In that reverted case both racers can pass
        /// different real guards, so their first match scans rendezvous.
        keys_split: AtomicBool,
        /// Bounded symmetric rendezvous holding BOTH split-key racers' empty
        /// match scans open until each has scanned (revert regime only).
        split_scan_barrier: Rendezvous2,
        /// The row-count the LOSER observed on its re-probe scan (the first
        /// scan it runs AFTER acquiring the guard). `usize::MAX` = not yet
        /// recorded.
        loser_reprobe_rows: AtomicUsize,
        /// Signalled after the loser has completed its first post-guard
        /// inner scan. The winner's test guard waits here after releasing
        /// the real mutex, pinning pre-commit-drop reverts so the loser must
        /// make its create decision before the winner can commit.
        loser_reprobed: Arc<Barrier2>,
    }

    impl ConcurrentRaceState {
        fn new() -> Self {
            Self {
                guard_order: AtomicUsize::new(0),
                winner_has_guard: Arc::new(Barrier2::new()),
                winner_key: Mutex::new(None),
                loser_key_classified: Arc::new(Barrier2::new()),
                keys_split: AtomicBool::new(false),
                split_scan_barrier: Rendezvous2::new(),
                loser_reprobe_rows: AtomicUsize::new(usize::MAX),
                loser_reprobed: Arc::new(Barrier2::new()),
            }
        }
    }

    /// A tiny 2-party rendezvous (a `Barrier` of size 2 would also work,
    /// but this makes the winner→loser HAND-OFF explicit + one-directional:
    /// the winner signals, the loser waits). Uses `parking_lot`'s
    /// `Mutex` + `Condvar` (the crate's std-mutex family) to match the
    /// surrounding production types.
    struct Barrier2 {
        m: Mutex<bool>,
        cv: parking_lot::Condvar,
    }
    impl Barrier2 {
        fn new() -> Self {
            Self {
                m: Mutex::new(false),
                cv: parking_lot::Condvar::new(),
            }
        }
        fn signal(&self) {
            let mut g = self.m.lock();
            *g = true;
            self.cv.notify_all();
        }
        /// Bounded wait — parks at most [`RACE_WAIT_BUDGET`]. Returns
        /// `true` iff the signal arrived. The healthy-path wait is
        /// micro/milliseconds (the peer's signal is MANDATORY on its very
        /// next step), so a timeout only fires when the peer thread DIED
        /// (panic / substrate error under load). Callers turn `false` into
        /// a loud per-test failure instead of parking forever — an
        /// unbounded wait here converts one dead racer into a suite-wide
        /// hang of the test binary (a leaked forever-parked libtest worker,
        /// or the WHOLE run under `--test-threads=1`), which is a
        /// cross-test side effect no test may have.
        fn wait_bounded(&self) -> bool {
            let mut g = self.m.lock();
            let deadline = std::time::Instant::now() + RACE_WAIT_BUDGET;
            while !*g {
                if self.cv.wait_until(&mut g, deadline).timed_out() {
                    return *g;
                }
            }
            true
        }
    }

    /// Upper bound for every wait in the [`ConcurrentRaceState`] rendezvous.
    /// Healthy-path waits complete in micro/milliseconds even under a
    /// background-QoS + 12-CPU-hog stress regime (measured ≥135× scheduler
    /// slowdown); 30 s is >4 orders of magnitude of margin, so a timeout
    /// can only mean the peer racer is DEAD — never a slow-but-healthy peer.
    const RACE_WAIT_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

    /// A bounded SYMMETRIC 2-party rendezvous (both parties block until
    /// both have arrived) used for the split-key revert regime's scan
    /// hold-open. `std::sync::Barrier` would park FOREVER if the peer died
    /// — see [`Barrier2::wait_bounded`] for why that is a cross-test
    /// side effect; this variant returns `false` on timeout instead.
    struct Rendezvous2 {
        m: Mutex<usize>,
        cv: parking_lot::Condvar,
    }
    impl Rendezvous2 {
        fn new() -> Self {
            Self {
                m: Mutex::new(0),
                cv: parking_lot::Condvar::new(),
            }
        }
        /// Arrive + wait (bounded) for the 2nd party. Returns `true` iff
        /// both parties arrived within [`RACE_WAIT_BUDGET`].
        fn arrive_and_wait_bounded(&self) -> bool {
            let mut g = self.m.lock();
            *g += 1;
            self.cv.notify_all();
            let deadline = std::time::Instant::now() + RACE_WAIT_BUDGET;
            while *g < 2 {
                if self.cv.wait_until(&mut g, deadline).timed_out() {
                    return *g >= 2;
                }
            }
            true
        }
    }

    /// Per-thread decorator that (a) sequences the guard acquisition so the
    /// winner acquires first, and (b) records the loser's post-guard
    /// re-probe scan result. Wraps the shared production substrate.
    ///
    /// The per-thread flags use [`AtomicBool`](std::sync::atomic::AtomicBool)
    /// (not `Cell`) so the wrapper is naturally `Send + Sync` — no `unsafe`
    /// needed to satisfy the `ExecutorSubstrate: Send + Sync` bound. Each
    /// wrapper is built + driven on ONE thread, so the atomics are
    /// uncontended; `Relaxed`/`Acquire`/`Release` ordering just gives a
    /// safe `Sync` type.
    struct ConcurrentRaceSubstrate {
        inner: Arc<CrudExecutorSubstrate>,
        state: Arc<ConcurrentRaceState>,
        /// Set once THIS thread has passed `merge_guard`. Its NEXT
        /// `scan_nodes_with_context` is the re-probe.
        acquired: AtomicBool,
        /// Whether THIS thread is the loser (2nd guard acquirer). Set in
        /// `merge_guard`.
        is_loser: AtomicBool,
    }

    /// Test-only wrapper around the winner's real merge guard. Dropping it
    /// releases the production mutex first, then waits until the loser has
    /// completed its re-probe. With the fix, this drop is post-commit and
    /// the loser observes the committed node. With a pre-commit-drop
    /// revert, the loser deterministically observes the empty snapshot and
    /// commits a second node before the winner is allowed to continue.
    struct ConcurrentRaceMergeGuard {
        inner: Option<Box<dyn MergeGuard>>,
        state: Arc<ConcurrentRaceState>,
    }

    impl std::fmt::Debug for ConcurrentRaceMergeGuard {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("ConcurrentRaceMergeGuard(held)")
        }
    }

    impl MergeGuard for ConcurrentRaceMergeGuard {}

    impl Drop for ConcurrentRaceMergeGuard {
        fn drop(&mut self) {
            // Release the REAL guard before waiting, otherwise the loser
            // could never acquire it and reach its re-probe.
            drop(self.inner.take());
            // If the WINNER is already unwinding (its materialize errored /
            // a wrapper seam panicked), do not pin at all: the loser may
            // never re-probe, and a panic during unwind would abort the
            // whole process. The test fails via the winner's join anyway.
            if std::thread::panicking() {
                return;
            }
            // BOUNDED pin: wait for the loser's re-probe, but never park
            // forever. Drop-glue MUST NOT panic (a Drop panic escalates to
            // a process abort inside libtest's unwind), so a timeout —
            // which can only mean the loser racer DIED — just releases the
            // pin with a diagnostic; the dead loser's join().expect then
            // fails the test cleanly. Pre-fix this wait was unbounded and a
            // dead loser hung the whole test binary (~40% of fault-injected
            // runs; a leaked forever-parked thread on the rest).
            if !self.state.loser_reprobed.wait_bounded() {
                eprintln!(
                    "ConcurrentRaceMergeGuard: loser never re-probed within \
                     {RACE_WAIT_BUDGET:?} — peer racer died; releasing the \
                     commit pin (its join failure will fail the test)"
                );
            }
        }
    }

    impl ExecutorSubstrate for ConcurrentRaceSubstrate {
        fn scan_nodes(
            &self,
            tenant: TenantId,
            label: Option<LabelId>,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            self.inner.scan_nodes(tenant, label, read_lsn)
        }

        fn scan_nodes_with_context(
            &self,
            ctx: &ExecutionContext,
            label: Option<LabelId>,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            let out = self.inner.scan_nodes_with_context(ctx, label, read_lsn)?;
            let first_reprobe = self.acquired.swap(false, Ordering::AcqRel);
            if first_reprobe
                && self.state.keys_split.load(Ordering::Acquire)
                && !self.state.split_scan_barrier.arrive_and_wait_bounded()
            {
                // Reverted numeric canonicalization gave the racers
                // different guards; hold both empty scans open until each
                // racer has scanned. A timeout = the peer racer died —
                // fail THIS test loudly instead of parking forever (which
                // would hang the whole test binary; see
                // `Barrier2::wait_bounded`). Panicking here is safe: we
                // are NOT in drop-glue, and the unwind releases this
                // thread's guards via the driver's RAII drain.
                panic!(
                    "ConcurrentRaceState: split-key peer never reached its \
                     match scan within {RACE_WAIT_BUDGET:?} — peer racer died"
                );
            }
            // If THIS thread is the loser and this is its FIRST scan after
            // acquiring the guard (the re-probe), record how many nodes it
            // saw. Under Fix 1 the winner's node is COMMITTED before the
            // loser's guard unblocked, so this scan sees it (≥1). If the
            // probe were BEFORE the lock, the loser would scan the
            // uncommitted snapshot and see 0 → the assert RED-flips.
            if self.is_loser.load(Ordering::Acquire)
                && first_reprobe
                && self.state.loser_reprobe_rows.load(Ordering::Acquire) == usize::MAX
            {
                self.state
                    .loser_reprobe_rows
                    .store(out.len(), Ordering::Release);
                self.state.loser_reprobed.signal();
            }
            Ok(out)
        }

        fn expand(
            &self,
            tenant: TenantId,
            from: NodeId,
            rel_type: Option<TypeId>,
            direction: Direction,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundEdge>, SubstrateAccessError> {
            self.inner
                .expand(tenant, from, rel_type, direction, read_lsn)
        }

        fn create_node(
            &self,
            tenant: TenantId,
            label: Option<&str>,
            properties: &[(String, Value)],
            ctx: &ExecutionContext,
        ) -> Result<NodeId, SubstrateAccessError> {
            self.inner.create_node(tenant, label, properties, ctx)
        }

        fn begin_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
            self.inner.begin_statement(ctx)
        }
        fn commit_statement(&self, ctx: &ExecutionContext) -> Result<(), SubstrateAccessError> {
            self.inner.commit_statement(ctx)
        }
        fn rollback_statement(&self, ctx: &ExecutionContext) {
            self.inner.rollback_statement(ctx)
        }

        fn merge_guard(
            &self,
            tenant: TenantId,
            key: &str,
        ) -> Result<Option<Box<dyn MergeGuard>>, SubstrateAccessError> {
            // Sequence the acquisition so the WINNER acquires first. The
            // first caller to reach here is the winner: it acquires the real
            // inner guard, then SIGNALS. The loser WAITS for that signal
            // BEFORE it calls the inner `merge_guard`, so it is guaranteed
            // to arrive second and BLOCK on the real lock until the winner
            // commits + drops (Fix 1: post-commit drop).
            let order = self.state.guard_order.fetch_add(1, Ordering::AcqRel);
            if order == 0 {
                // Winner — acquire the real guard, then release the loser to
                // race for the (now-held) lock.
                let g = self.inner.merge_guard(tenant, key)?;
                *self.state.winner_key.lock() = Some(key.to_owned());
                self.acquired.store(true, Ordering::Release);
                self.state.winner_has_guard.signal();
                // Keep holding the real guard until the loser has reached
                // this same seam and classified its key. On the fixed path
                // the loser then blocks on this guard; on the split-key
                // revert it passes its distinct guard and joins the scan
                // rendezvous. BOUNDED: if the loser died before reaching
                // `merge_guard`, fail loudly — the unwind releases the real
                // guard (local `g` drops) so nothing stays locked.
                if !self.state.loser_key_classified.wait_bounded() {
                    panic!(
                        "ConcurrentRaceState: loser never reached merge_guard \
                         within {RACE_WAIT_BUDGET:?} — peer racer died"
                    );
                }
                Ok(g.map(|inner| {
                    Box::new(ConcurrentRaceMergeGuard {
                        inner: Some(inner),
                        state: Arc::clone(&self.state),
                    }) as Box<dyn MergeGuard>
                }))
            } else if order == 1 {
                // Loser — wait until the winner holds the guard, THEN block
                // on the real inner lock (parks until the winner's
                // post-commit drop). On return the winner's node is durable.
                self.is_loser.store(true, Ordering::Release);
                if !self.state.winner_has_guard.wait_bounded() {
                    panic!(
                        "ConcurrentRaceState: winner never signalled guard \
                         acquisition within {RACE_WAIT_BUDGET:?} — peer racer died"
                    );
                }
                let keys_split = self.state.winner_key.lock().as_deref() != Some(key);
                self.state.keys_split.store(keys_split, Ordering::Release);
                self.state.loser_key_classified.signal();
                let g = self.inner.merge_guard(tenant, key)?;
                self.acquired.store(true, Ordering::Release);
                Ok(g)
            } else {
                // The protocol is EXACTLY two racers (winner + loser). A 3rd
                // arrival means the harness was mis-wired (e.g. a future
                // n>2 caller, or one thread re-entering merge_guard) — the
                // winner/loser hand-offs above would silently corrupt, so
                // reject it loudly at the seam instead.
                panic!(
                    "ConcurrentRaceState supports EXACTLY 2 racers; \
                     merge_guard arrival #{} is a test-harness wiring bug",
                    order + 1
                );
            }
        }

        fn vector_search(
            &self,
            tenant: TenantId,
            property: &str,
            query_vec: &[f32],
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            self.inner
                .vector_search(tenant, property, query_vec, k, read_lsn)
        }

        fn bm25_search(
            &self,
            tenant: TenantId,
            property: &str,
            query: &str,
            k: u64,
            read_lsn: Lsn,
        ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
            self.inner.bm25_search(tenant, property, query, k, read_lsn)
        }

        fn community_members(
            &self,
            tenant: TenantId,
            community_id: i64,
            read_lsn: Lsn,
        ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
            self.inner.community_members(tenant, community_id, read_lsn)
        }
    }

    /// **NN-4 (#1384) re-spin, Fix 4 — deterministic probe-after-lock
    /// rendezvous.** Two threads race `MERGE (u:User {email:'…'})` via the
    /// PRODUCTION materialize path. The winner acquires the guard first,
    /// stages + COMMITS its create, then drops the guard (post-commit under
    /// Fix 1). The loser blocks on the real guard until then, unblocks, and
    /// RE-PROBES — this test records the loser's re-probe row-count and
    /// asserts it saw the winner's COMMITTED node, AND that exactly one
    /// node exists at the end.
    ///
    /// # Round-2 fix-back (#1392): same wrap-and-pin machinery as the
    /// materialize / adversarial tests
    ///
    /// The first spin of this test used a WEAKER private wrapper
    /// (`RendezvousSubstrate`, now deleted) that returned the winner's RAW
    /// inner guard: no post-commit pin, and no wait for the loser to even
    /// ARRIVE at `merge_guard`. Two consequences, both measured:
    ///
    /// 1. **~2.5% RED (1/40) under a Fix-1 revert** — with the guard
    ///    dropped pre-commit, the loser's re-probe merely RACED the
    ///    winner's fast commit and usually still saw the committed node →
    ///    false GREEN. Not a detector.
    /// 2. **Vacuous-pass mode under load** — nothing stopped the winner
    ///    from completing its ENTIRE statement before the slow-to-spawn
    ///    loser reached `merge_guard`, so the "race" silently degenerated
    ///    to sequential execution.
    ///
    /// This spin drives [`ConcurrentRaceSubstrate`] — the SAME machinery
    /// the materialize + adversarial tests use — whose winner (a) HOLDS the
    /// real guard until the loser has arrived + classified its key (the
    /// contended interleaving is forced BY CONSTRUCTION, not by scheduler
    /// luck), and (b) is wrapped in [`ConcurrentRaceMergeGuard`], which on
    /// drop releases the real guard and then PINS the winner until the
    /// loser's re-probe has run. Under a Fix-1 revert (guard dropped
    /// pre-commit) the pin forces the loser to re-probe BEFORE the winner
    /// can commit → the loser deterministically scans the uncommitted
    /// snapshot → sees ONLY the throwaway row → creates a duplicate →
    /// BOTH asserts below RED-flip, 100% of runs (20/20 measured under 12
    /// CPU-hog load), matching the materialize test's detection strength.
    ///
    /// The row-count assert is `>= 2`, not `>= 1`: the label scan the
    /// re-probe runs is PRE-filter, so it always carries the pre-created
    /// throwaway `User` (1 row). `>= 1` was trivially true and detected
    /// nothing; `>= 2` (throwaway + the winner's committed node) is the
    /// real probe-after-lock invariant.
    #[test]
    fn probe_after_lock_loser_sees_committed_node_nn4() {
        let (sub, _crud, _mgr, _router) = fixture();
        // Pre-intern `User` so the match branch lowers to a real
        // Scan{Some(User)} + property filter (not LogicalEmpty).
        sub.create_node(
            TenantId::DEFAULT,
            Some("User"),
            &[("email".into(), Value::String("other@example.com".into()))],
            &tctx(),
        )
        .expect("pre-create throwaway User");
        let user_label = sub
            .intern_table()
            .intern_label(TenantId::DEFAULT, "User")
            .unwrap();
        let email = "rendezvous@example.com";
        let plan = Arc::new(build_merge_user_email_plan(user_label, email));
        let sub = Arc::new(sub);

        let state = Arc::new(ConcurrentRaceState::new());

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let sub = Arc::clone(&sub);
                let plan = Arc::clone(&plan);
                let state = Arc::clone(&state);
                std::thread::spawn(move || {
                    let wrapper = ConcurrentRaceSubstrate {
                        inner: Arc::clone(&sub),
                        state,
                        acquired: AtomicBool::new(false),
                        is_loser: AtomicBool::new(false),
                    };
                    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
                    // Drive the PRODUCTION materialize path (D-2 wrap).
                    arcgraph_query::materialize::materialize(&plan, &wrapper, &ctx)
                        .expect("MERGE via materialize executes");
                })
            })
            .collect();
        for h in handles {
            h.join().expect("MERGE thread joined (no deadlock)");
        }

        // The loser's re-probe MUST have seen the winner's COMMITTED node.
        let reprobe = state.loser_reprobe_rows.load(Ordering::Acquire);
        assert_ne!(
            reprobe,
            usize::MAX,
            "the loser must have re-probed after acquiring the guard \
             (the rendezvous did not fire — test wiring bug)"
        );
        assert!(
            reprobe >= 2,
            "PROBE-AFTER-LOCK: the loser's re-probe (after acquiring the \
             merge guard) MUST see the winner's COMMITTED node — the \
             pre-filter label scan carries the throwaway User (1 row) PLUS \
             the winner's node, so a spanning guard yields >= 2 rows; got \
             {reprobe}. Exactly 1 row (throwaway only) means the guard did \
             NOT span the commit (winner released before its node was \
             durable) → the double-create bug. This is the invariant that \
             RED-flips if the probe is moved before the lock."
        );
        // And end-to-end: exactly one node.
        assert_eq!(
            count_users_with_email(&sub, user_label, email),
            1,
            "probe-after-lock: get-or-create yields EXACTLY ONE node \
             (loser re-probe saw {reprobe} pre-filter rows)"
        );
    }

    // ─── #1366 (task #248, Phase 1) — property-index end-to-end ─────────

    /// Resolve a node's label id via the intern table (the label a
    /// CREATE INDEX declares on).
    fn label_of(sub: &CrudExecutorSubstrate, name: &str) -> LabelId {
        LabelId::new(
            sub.intern_table()
                .intern_label(TenantId::DEFAULT, name)
                .unwrap()
                .raw(),
        )
    }

    #[test]
    fn e2e_create_index_backfills_and_lookup_finds_nodes() {
        let (sub, crud, _mgr, _router) = fixture();
        // Seed 3 User nodes with an email property.
        for i in 1..=3u64 {
            sub.create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String(format!("user{i}@x.com")))],
                &tctx(),
            )
            .expect("create user");
        }
        // CREATE INDEX FOR (n:User) ON (n.email) — registers + backfills.
        let reg = sub
            .create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create_property_index");
        assert_eq!(reg, PropertyIndexRegistration::Created);
        // Every backfilled email is findable via the manager's lookup
        // (candidate-then-verify is the caller's job; here we assert the
        // candidate set is non-empty for a seeded value + empty for an
        // unseen one).
        let mgr = sub.property_index_manager(&crud).unwrap();
        let user_label = label_of(&sub, "User");
        for i in 1..=3u64 {
            let cands = mgr
                .lookup_candidates(
                    TenantId::DEFAULT,
                    user_label,
                    "email",
                    &Value::String(format!("user{i}@x.com")),
                )
                .unwrap();
            assert!(
                !cands.is_empty(),
                "backfilled user{i} must be a lookup candidate"
            );
        }
        // A never-seen value returns no candidates (and does NOT grow the
        // InternTable — the value is HASHED, RC-4).
        let intern_len_before = sub.intern_table().len(TenantId::DEFAULT);
        let none = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                user_label,
                "email",
                &Value::String("nobody@x.com".into()),
            )
            .unwrap();
        assert!(none.is_empty(), "unseen value has no candidates");
        assert_eq!(
            sub.intern_table().len(TenantId::DEFAULT),
            intern_len_before,
            "RC-4: a never-seen VALUE lookup must NOT intern (value is hashed)"
        );
    }

    #[test]
    fn e2e_write_follows_declare_and_red_on_revert_marker() {
        // Write-follows-declare (RC-2): a node SET after the index exists
        // is maintained + findable. The RED-on-revert form: if maintenance
        // were gated on Online-only (it is NOT — the manager applies for
        // Building|Online via maintenance_active), a Building-window write
        // would be absent. Here the index is Online post-backfill; the
        // load-bearing assertion is that the LATER write is found.
        let (sub, crud, _mgr, _router) = fixture();
        sub.create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create index (empty backfill)");
        // Now CREATE a node with the declared property — must be maintained.
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("late@x.com".into()))],
                &tctx(),
            )
            .expect("create late user");
        let mgr = sub.property_index_manager(&crud).unwrap();
        let user_label = label_of(&sub, "User");
        let cands = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                user_label,
                "email",
                &Value::String("late@x.com".into()),
            )
            .unwrap();
        assert!(
            cands.contains(&node),
            "write-follows-declare: a node created AFTER CREATE INDEX must be maintained + found"
        );
    }

    /// #1366 R1 NIT-1 — `set_node` / `remove_node` must maintain the
    /// property index at the node's REAL label, NEVER at label 0.
    ///
    /// The pre-fix code did a redundant SECOND `read_node(...).map(|r|
    /// r.label_id).unwrap_or(0)` for the maintenance label; on a 2nd-read
    /// miss the `unwrap_or(0)` silently keyed the index at label 0 (the
    /// WRONG slot). The fix captures the label from the FIRST read.
    ///
    /// Discriminating RED-on-revert: substitute the maintenance label with
    /// `LabelId::new(0)` (the exact silent-0 failure mode of the reverted
    /// `unwrap_or(0)`). The positive assert (found at the REAL label) then
    /// fails and the negative assert (NOT found at label 0) also fails,
    /// because the SET value lands under label 0's key. Both go RED, so the
    /// test genuinely enforces "maintain at the real label, not 0" rather
    /// than merely passing on the happy path.
    ///
    /// Note: restoring the *literal* 2nd `read_node(...).unwrap_or(0)` does
    /// NOT flip this test on its own — in a single-snapshot, single-thread
    /// unit the 2nd read succeeds and returns the same real label, so the
    /// `unwrap_or(0)` fallback never fires. The label-0 substitution above
    /// reproduces that fallback's runtime effect deterministically.
    #[test]
    fn e2e_set_and_remove_maintain_index_at_real_label_not_zero() {
        let (sub, crud, _mgr, _router) = fixture();
        // A User node with a real (non-zero) label and an email property.
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("orig@x.com".into()))],
                &tctx(),
            )
            .expect("create user");
        sub.create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create index on email");
        let mgr = sub.property_index_manager(&crud).unwrap();
        let user_label = label_of(&sub, "User");
        // Precondition: the real label is genuinely non-zero, else the
        // "not 0" discrimination would be vacuous.
        assert_ne!(user_label.raw(), 0, "test needs a real (non-zero) label");

        // SET a NEW email value → maintenance must key the new value under
        // the node's REAL label (insert-only leaves the old as a ghost).
        sub.set_node(
            TenantId::DEFAULT,
            node,
            &SetNodeMutation::PropertyAssign {
                name: "email".into(),
                value: Value::String("new@x.com".into()),
            },
            &tctx(),
        )
        .expect("set_node email");

        // Positive: the SET value is a candidate at the node's REAL label.
        let at_real = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                user_label,
                "email",
                &Value::String("new@x.com".into()),
            )
            .unwrap();
        assert!(
            at_real.contains(&node),
            "set_node must maintain the index at the node's REAL label \
             (RED if maintenance keyed at label 0)"
        );
        // Negative: nothing was keyed under the label-0 slot.
        let at_zero = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                LabelId::new(0),
                "email",
                &Value::String("new@x.com".into()),
            )
            .unwrap();
        assert!(
            at_zero.is_empty(),
            "set_node must NEVER maintain the index at label 0 \
             (RED if the silent unwrap_or(0) fallback fired)"
        );

        // REMOVE the property → maintenance must again route to the REAL
        // label (the call is a correct no-op for lookup here, but MUST NOT
        // touch label 0's slot).
        sub.remove_node(
            TenantId::DEFAULT,
            node,
            &RemoveNodeMutation::Property("email".into()),
            &tctx(),
        )
        .expect("remove_node email");
        // remove_node maintenance still must not have populated label 0.
        let at_zero_after_remove = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                LabelId::new(0),
                "email",
                &Value::String("new@x.com".into()),
            )
            .unwrap();
        assert!(
            at_zero_after_remove.is_empty(),
            "remove_node must NEVER maintain the index at label 0"
        );
    }

    #[test]
    fn e2e_int_float_boundary_red_on_revert() {
        // RC-5: n.age = 42.0 must find stored int 42. This is the
        // int-float normalize invariant — RED-on-revert: breaking the
        // integral-float→integer normalize makes the float lookup miss the
        // int-stored value.
        let (sub, crud, _mgr, _router) = fixture();
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("Person"),
                &[("age".into(), Value::Integer(42))],
                &tctx(),
            )
            .expect("create person age=42");
        sub.create_property_index(TenantId::DEFAULT, "person_age", false, "Person", "age")
            .expect("create index on age");
        let mgr = sub.property_index_manager(&crud).unwrap();
        let person_label = label_of(&sub, "Person");
        // Lookup with a FLOAT literal 42.0 must find the int-42 node.
        let cands_float = mgr
            .lookup_candidates(TenantId::DEFAULT, person_label, "age", &Value::Float(42.0))
            .unwrap();
        assert!(
            cands_float.contains(&node),
            "int/float coercion (RC-5): n.age=42.0 must find stored int 42"
        );
        // And an int lookup finds it too (positive control).
        let cands_int = mgr
            .lookup_candidates(TenantId::DEFAULT, person_label, "age", &Value::Integer(42))
            .unwrap();
        assert!(cands_int.contains(&node));
    }

    #[test]
    fn e2e_negative_int_absent_but_write_succeeds() {
        // RC-5: a negative int is unsupported → absent from the index, but
        // the write STILL succeeds (not rejected).
        let (sub, crud, _mgr, _router) = fixture();
        sub.create_property_index(TenantId::DEFAULT, "person_age", false, "Person", "age")
            .expect("create index");
        // A create with a NEGATIVE age — must succeed.
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("Person"),
                &[("age".into(), Value::Integer(-5))],
                &tctx(),
            )
            .expect("create with negative age must succeed (write not rejected)");
        assert!(node.raw() > 0);
        // But the negative value is absent from the index.
        let mgr = sub.property_index_manager(&crud).unwrap();
        let person_label = label_of(&sub, "Person");
        let cands = mgr
            .lookup_candidates(TenantId::DEFAULT, person_label, "age", &Value::Integer(-5))
            .unwrap();
        assert!(cands.is_empty(), "negative int is absent from the index");
    }

    #[test]
    fn e2e_if_not_exists_idempotent() {
        let (sub, _crud, _mgr, _router) = fixture();
        assert_eq!(
            sub.create_property_index(TenantId::DEFAULT, "e", true, "User", "email")
                .unwrap(),
            PropertyIndexRegistration::Created
        );
        assert_eq!(
            sub.create_property_index(TenantId::DEFAULT, "e", true, "User", "email")
                .unwrap(),
            PropertyIndexRegistration::AlreadyExists
        );
    }

    // ─── #1401 — missed-node W1 backfill race (barrier-gated threads) ───
    //
    // The discriminating regression. A concurrent write whose data commit
    // lands in the window between the backfill snapshot and the catalog
    // `Building` commit is, in the PRE-FIX ordering, neither in the
    // snapshot Vec (LSN > snapshot S) nor maintained (its maintain reads
    // an empty catalog → early-return) — a permanently missing index
    // entry. The FIX registers Building FIRST, so the writer is EITHER
    // maintained OR captured in the post-register snapshot.

    thread_local! {
        /// A one-shot callback fired inside `create_property_index`(_prefix)
        /// at the register↔snapshot boundary, on the thread that runs it
        /// (Thread B). The race test installs it to release Thread W into
        /// the window and join it back. `RefCell<Option<…>>` so it is
        /// per-thread + reset after firing.
        static CREATE_INDEX_REGISTER_BARRIER: std::cell::RefCell<
            Option<Box<dyn FnOnce()>>,
        > = const { std::cell::RefCell::new(None) };
    }

    /// Install the register↔snapshot barrier for the CURRENT thread
    /// (Thread B). Fired at most once (taken on fire).
    fn install_create_index_register_barrier(f: Box<dyn FnOnce()>) {
        CREATE_INDEX_REGISTER_BARRIER.with(|c| *c.borrow_mut() = Some(f));
    }

    /// Fire (and clear) the register↔snapshot barrier if one is installed
    /// on this thread. Called by `create_property_index` and the pre-fix
    /// oracle. No-op when unset (the common case).
    pub(super) fn fire_create_index_register_barrier() {
        let hook = CREATE_INDEX_REGISTER_BARRIER.with(|c| c.borrow_mut().take());
        if let Some(f) = hook {
            f();
        }
    }

    /// Seed `n` `(:User {email})` nodes so the backfill scan iterates a
    /// non-trivial id range, returning the highest seeded id (high-water).
    fn seed_users(sub: &CrudExecutorSubstrate, n: u64) {
        for i in 1..=n {
            sub.create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String(format!("seed{i}@x.com")))],
                &tctx(),
            )
            .expect("seed user");
        }
    }

    /// Run the barrier-gated missed-node interleaving and return whether
    /// the writer's node ended up in the index after the flip.
    ///
    /// `use_prefix_ordering = true` runs the PRE-FIX (buggy) ordering
    /// (RED expected); `false` runs the FIXED ordering (GREEN expected).
    /// The SAME interleaving drives both — the only difference is the
    /// method under test, so the test genuinely discriminates.
    fn run_missed_node_race(use_prefix_ordering: bool) -> bool {
        use std::sync::{Arc, Barrier};

        let (sub, crud, _mgr, _router) = fixture();
        // Force the label + manager to exist BEFORE the threads start
        // (so both threads share the same lazily-built manager instance
        // and the label id is stable). Seed pre-existing nodes.
        seed_users(&sub, 5);
        let _ = sub.property_index_manager(&crud).unwrap();
        let user_label = label_of(&sub, "User");
        let sub = Arc::new(sub);

        // `w_go` releases W into the window; `w_done` signals W finished
        // its create+maintain so B may proceed past the barrier.
        let w_go = Arc::new(Barrier::new(2));
        let w_done = Arc::new(Barrier::new(2));

        let w_handle = {
            let sub = Arc::clone(&sub);
            let w_go = Arc::clone(&w_go);
            let w_done = Arc::clone(&w_done);
            std::thread::spawn(move || {
                // Wait until B is at the register↔snapshot boundary.
                w_go.wait();
                // Create W's node (data commit + maintain) STRICTLY inside
                // the window. In pre-fix, maintain reads an empty catalog
                // → no-op; in fixed, the catalog is non-empty → maintained.
                let w = sub
                    .create_node(
                        TenantId::DEFAULT,
                        Some("User"),
                        &[("email".into(), Value::String("w@x.com".into()))],
                        &tctx(),
                    )
                    .expect("W create");
                // Release B to continue (take snapshot / register etc.).
                w_done.wait();
                w
            })
        };

        // Thread B: install the barrier, then run CREATE INDEX.
        let barrier_fn: Box<dyn FnOnce()> = {
            let w_go = Arc::clone(&w_go);
            let w_done = Arc::clone(&w_done);
            Box::new(move || {
                w_go.wait(); // let W run its create+maintain
                w_done.wait(); // wait for W to finish before B proceeds
            })
        };
        install_create_index_register_barrier(barrier_fn);
        let reg = if use_prefix_ordering {
            sub.create_property_index_prefix_ordering(
                TenantId::DEFAULT,
                "idx_email",
                false,
                "User",
                "email",
            )
        } else {
            sub.create_property_index(TenantId::DEFAULT, "idx_email", false, "User", "email")
        }
        .expect("create index");
        assert_eq!(reg, PropertyIndexRegistration::Created);

        let w_node = w_handle.join().expect("W thread");

        // After Online, is W's node a lookup candidate?
        let mgr = sub.property_index_manager(&crud).unwrap();
        let cands = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                user_label,
                "email",
                &Value::String("w@x.com".into()),
            )
            .expect("lookup");
        // Sanity: the seeded nodes are always present (backfill works).
        let seed_cands = mgr
            .lookup_candidates(
                TenantId::DEFAULT,
                user_label,
                "email",
                &Value::String("seed1@x.com".into()),
            )
            .expect("seed lookup");
        assert!(
            !seed_cands.is_empty(),
            "backfill must always index the pre-existing seeded nodes"
        );
        cands.contains(&w_node)
    }

    /// **RED-on-revert proof.** The SAME barrier interleaving on the
    /// PRE-FIX ordering LOSES W's node — it is in neither the snapshot Vec
    /// nor maintained. This asserts the bug reproduces (a non-
    /// discriminating test would be a fail).
    #[test]
    fn missed_node_w1_race_is_lost_on_prefix_ordering_red() {
        let found = run_missed_node_race(true);
        assert!(
            !found,
            "PRE-FIX ordering must LOSE the window writer (RED-on-revert oracle): the node \
             committed between snapshot and register is in neither the backfill Vec nor maintained"
        );
    }

    /// **GREEN after fix.** The FIXED ordering (register Building FIRST,
    /// then snapshot) maintains the window writer → W's node is indexed.
    #[test]
    fn missed_node_w1_race_is_captured_after_fix_green() {
        let found = run_missed_node_race(false);
        assert!(
            found,
            "FIXED ordering must capture the window writer: registered Building first, so W's \
             maintain lands (catalog non-empty) — W@x.com must be a lookup candidate after Online"
        );
    }

    /// **W1b UPDATE variant.** W does `set_node` email A→B inside the
    /// window (the A-ghost-only failure mode: pre-fix leaves only a stale
    /// A entry so a lookup on B misses forever). The fixed ordering
    /// maintains the B insert. RED-on-revert proven by the prefix leg.
    fn run_update_variant(use_prefix_ordering: bool) -> bool {
        use std::sync::{Arc, Barrier};

        let (sub, crud, _mgr, _router) = fixture();
        seed_users(&sub, 5);
        // W's node exists BEFORE the index with email = A; the in-window
        // UPDATE flips it to B.
        let w_node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("a@x.com".into()))],
                &tctx(),
            )
            .expect("pre-create W with email=A");
        let _ = sub.property_index_manager(&crud).unwrap();
        let user_label = label_of(&sub, "User");
        let sub = Arc::new(sub);

        let w_go = Arc::new(Barrier::new(2));
        let w_done = Arc::new(Barrier::new(2));

        let w_handle = {
            let sub = Arc::clone(&sub);
            let w_go = Arc::clone(&w_go);
            let w_done = Arc::clone(&w_done);
            std::thread::spawn(move || {
                w_go.wait();
                sub.set_node(
                    TenantId::DEFAULT,
                    w_node,
                    &SetNodeMutation::PropertyAssign {
                        name: "email".into(),
                        value: Value::String("b@x.com".into()),
                    },
                    &tctx(),
                )
                .expect("W set email A→B");
                w_done.wait();
            })
        };

        let barrier_fn: Box<dyn FnOnce()> = {
            let w_go = Arc::clone(&w_go);
            let w_done = Arc::clone(&w_done);
            Box::new(move || {
                w_go.wait();
                w_done.wait();
            })
        };
        install_create_index_register_barrier(barrier_fn);
        let reg = if use_prefix_ordering {
            sub.create_property_index_prefix_ordering(
                TenantId::DEFAULT,
                "idx_email",
                false,
                "User",
                "email",
            )
        } else {
            sub.create_property_index(TenantId::DEFAULT, "idx_email", false, "User", "email")
        }
        .expect("create index");
        assert_eq!(reg, PropertyIndexRegistration::Created);
        w_handle.join().expect("W thread");

        let mgr = sub.property_index_manager(&crud).unwrap();
        // Lookup on the NEW value B must contain W.
        mgr.lookup_candidates(
            TenantId::DEFAULT,
            user_label,
            "email",
            &Value::String("b@x.com".into()),
        )
        .expect("lookup B")
        .contains(&w_node)
    }

    #[test]
    fn missed_node_w1b_update_variant_prefix_red_fixed_green() {
        assert!(
            !run_update_variant(true),
            "PRE-FIX: an in-window A→B update leaves only the stale A ghost; lookup on B misses"
        );
        assert!(
            run_update_variant(false),
            "FIXED: the in-window A→B update is maintained; lookup on B finds W"
        );
    }

    /// Drive the interleaving through the path-2 (high_water) pre-fix
    /// oracle (RED) or the fixed method (GREEN). W creates a NEW node at
    /// the barrier; in the path-2 oracle high_water was sampled BEFORE the
    /// barrier so W's id (= high_water + 1) is outside the scan range even
    /// though the post-barrier scan snapshot CAN see it — the pure
    /// snapshot-choice miss (not maskable by the maintain path, because in
    /// path-2 the maintain also saw an empty catalog).
    fn run_high_water_variant(use_prefix_ordering: bool) -> bool {
        use std::sync::{Arc, Barrier};

        let (sub, crud, _mgr, _router) = fixture();
        seed_users(&sub, 5);
        let _ = sub.property_index_manager(&crud).unwrap();
        let user_label = label_of(&sub, "User");
        let sub = Arc::new(sub);

        let w_go = Arc::new(Barrier::new(2));
        let w_done = Arc::new(Barrier::new(2));

        let w_handle = {
            let sub = Arc::clone(&sub);
            let w_go = Arc::clone(&w_go);
            let w_done = Arc::clone(&w_done);
            std::thread::spawn(move || {
                w_go.wait();
                let w = sub
                    .create_node(
                        TenantId::DEFAULT,
                        Some("User"),
                        &[("email".into(), Value::String("w@x.com".into()))],
                        &tctx(),
                    )
                    .expect("W create (id = high_water + 1)");
                w_done.wait();
                w
            })
        };

        let barrier_fn: Box<dyn FnOnce()> = {
            let w_go = Arc::clone(&w_go);
            let w_done = Arc::clone(&w_done);
            Box::new(move || {
                w_go.wait();
                w_done.wait();
            })
        };
        install_create_index_register_barrier(barrier_fn);
        let reg = if use_prefix_ordering {
            sub.create_property_index_prefix_high_water(
                TenantId::DEFAULT,
                "idx_email",
                false,
                "User",
                "email",
            )
        } else {
            sub.create_property_index(TenantId::DEFAULT, "idx_email", false, "User", "email")
        }
        .expect("create index");
        assert_eq!(reg, PropertyIndexRegistration::Created);
        let w_node = w_handle.join().expect("W thread");

        let mgr = sub.property_index_manager(&crud).unwrap();
        mgr.lookup_candidates(
            TenantId::DEFAULT,
            user_label,
            "email",
            &Value::String("w@x.com".into()),
        )
        .expect("lookup")
        .contains(&w_node)
    }

    /// **high_water variant (the second W1 path).** W's create commits
    /// after `high_water` is sampled but before the scan begins — a node
    /// visible-to-snapshot with `id > high_water`. Pre-fix drops it from
    /// the `1..=high_water` range even though it is snapshot-visible; the
    /// fix samples `high_water` from WITHIN the post-register snapshot so
    /// the id bound is consistent.
    #[test]
    fn missed_node_high_water_variant_prefix_red_fixed_green() {
        assert!(
            !run_high_water_variant(true),
            "high_water/prefix: the id>high_water window writer is dropped from the scan range"
        );
        assert!(
            run_high_water_variant(false),
            "high_water/fixed: high_water sampled inside the post-register snapshot bounds W in"
        );
    }

    // ─── #1366 (Phase 2) — the production property-index-lookup SEAM ────
    // These drive `property_index_lookup_with_context` against the REAL
    // B+tree + MVCC hydrate path (not the stub), proving the load-bearing
    // candidate-then-verify + dedup + RC-6 gate over live storage.

    /// The verified NodeIds the seam returns for a lookup, sorted.
    fn seam_ids(
        sub: &CrudExecutorSubstrate,
        label: LabelId,
        property: &str,
        value: &Value,
    ) -> Vec<u64> {
        let mut ids: Vec<u64> = sub
            .property_index_lookup_with_context(&tctx(), label, property, value, Lsn::MAX)
            .expect("seam lookup OK")
            .into_iter()
            .map(|bn| bn.node.id.raw())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// The full-scan result for the same `(label, property = value)`, as
    /// the ground-truth the seam must reproduce EXACTLY.
    fn scan_ids(
        sub: &CrudExecutorSubstrate,
        label: LabelId,
        property: &str,
        value: &Value,
    ) -> Vec<u64> {
        let mut ids: Vec<u64> = sub
            .scan_nodes(TenantId::DEFAULT, Some(label), Lsn::MAX)
            .expect("scan OK")
            .into_iter()
            .filter(|bn| bn.node.properties.get(property) == Some(value))
            .map(|bn| bn.node.id.raw())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// IDENTICAL-RESULTS over the real backend: the seam's verified set
    /// equals the full-scan set for every seeded value.
    #[test]
    fn seam_verified_set_equals_full_scan() {
        let (sub, _crud, _mgr, _router) = fixture();
        for i in 1..=5u64 {
            sub.create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String(format!("u{i}@x.com")))],
                &tctx(),
            )
            .expect("create");
        }
        sub.create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create index");
        let label = label_of(&sub, "User");
        for i in 1..=5u64 {
            let v = Value::String(format!("u{i}@x.com"));
            assert_eq!(
                seam_ids(&sub, label, "email", &v),
                scan_ids(&sub, label, "email", &v),
                "seam must equal full scan for u{i}"
            );
        }
        // Absent value → empty on both paths.
        let ghost = Value::String("ghost@x.com".into());
        assert!(seam_ids(&sub, label, "email", &ghost).is_empty());
        assert!(scan_ids(&sub, label, "email", &ghost).is_empty());
    }

    /// MVCC-VERIFY — stale ghost dropped. A SET changes a node's email;
    /// the insert-only index leaves a STALE candidate for the OLD value.
    /// The seam's recheck must drop it (the OLD value now finds NOTHING,
    /// the NEW value finds the node) — matching the scan exactly.
    #[test]
    fn seam_drops_stale_ghost_candidate() {
        let (sub, _crud, _mgr, _router) = fixture();
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("old@x.com".into()))],
                &tctx(),
            )
            .expect("create");
        sub.create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create index");
        // SET a new email — the OLD value's index slot becomes a stale ghost.
        sub.set_node(
            TenantId::DEFAULT,
            node,
            &SetNodeMutation::PropertyAssign {
                name: "email".into(),
                value: Value::String("new@x.com".into()),
            },
            &tctx(),
        )
        .expect("set email");
        let label = label_of(&sub, "User");
        // OLD value: the seam drops the stale ghost → empty == scan.
        let old = Value::String("old@x.com".into());
        assert_eq!(seam_ids(&sub, label, "email", &old), Vec::<u64>::new());
        assert_eq!(scan_ids(&sub, label, "email", &old), Vec::<u64>::new());
        // NEW value: found by both.
        let new = Value::String("new@x.com".into());
        assert_eq!(
            seam_ids(&sub, label, "email", &new),
            scan_ids(&sub, label, "email", &new),
        );
        assert_eq!(seam_ids(&sub, label, "email", &new), vec![node.raw()]);
    }

    /// MVCC-VERIFY — tombstoned candidate dropped. A DELETE tombstones a
    /// node; the index still holds its candidate slot (insert-only). The
    /// seam hydration returns None for the tombstoned node → dropped,
    /// matching the scan (which also excludes it).
    #[test]
    fn seam_drops_tombstoned_candidate() {
        let (sub, _crud, _mgr, _router) = fixture();
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("doomed@x.com".into()))],
                &tctx(),
            )
            .expect("create");
        sub.create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create index");
        sub.delete_node(TenantId::DEFAULT, node, false, &tctx())
            .expect("delete");
        let label = label_of(&sub, "User");
        let v = Value::String("doomed@x.com".into());
        // Both the seam (candidate hydrate → None) and the scan exclude the
        // tombstoned node → empty.
        assert_eq!(seam_ids(&sub, label, "email", &v), Vec::<u64>::new());
        assert_eq!(scan_ids(&sub, label, "email", &v), Vec::<u64>::new());
    }

    /// DEDUP — two nodes sharing a value each verify; a SET re-inserting
    /// the SAME value leaves a duplicate slot for one node. The seam must
    /// emit each surviving NodeId exactly once (== the scan multiset).
    #[test]
    fn seam_dedups_duplicate_candidate_slots() {
        let (sub, _crud, _mgr, _router) = fixture();
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("dup@x.com".into()))],
                &tctx(),
            )
            .expect("create");
        sub.create_property_index(TenantId::DEFAULT, "user_email", false, "User", "email")
            .expect("create index");
        // SET the SAME value TWICE via two distinct writes so the
        // insert-only path may leave duplicate slots (idempotent-value
        // writes are skipped, so force distinct intermediate values).
        for v in ["other@x.com", "dup@x.com"] {
            sub.set_node(
                TenantId::DEFAULT,
                node,
                &SetNodeMutation::PropertyAssign {
                    name: "email".into(),
                    value: Value::String(v.into()),
                },
                &tctx(),
            )
            .expect("set");
        }
        let label = label_of(&sub, "User");
        let v = Value::String("dup@x.com".into());
        // One row for the one node, regardless of how many candidate slots.
        assert_eq!(seam_ids(&sub, label, "email", &v), vec![node.raw()]);
        assert_eq!(
            seam_ids(&sub, label, "email", &v),
            scan_ids(&sub, label, "email", &v),
        );
    }

    /// RC-6 gate — a BUILDING index serves NO candidates through the seam.
    /// We register a Building index (register-only, no flip) and assert the
    /// seam returns empty even though a matching live node exists (the
    /// full scan finds it). This is the planner-visible gate at the lookup
    /// entry (defense-in-depth).
    ///
    /// RED-on-revert: gating `lookup_candidates` on `maintenance_active()`
    /// (Building|Online) instead of `planner_visible()` (Online-only) would
    /// make the seam serve the Building index → the seam would find the
    /// node → this assertion (`seam empty` while `scan non-empty`) flips RED.
    #[test]
    fn seam_rc6_building_index_serves_no_candidates() {
        let (sub, crud, _mgr, _router) = fixture();
        let node = sub
            .create_node(
                TenantId::DEFAULT,
                Some("User"),
                &[("email".into(), Value::String("b@x.com".into()))],
                &tctx(),
            )
            .expect("create");
        // Register a BUILDING index (no backfill+flip → stays Building).
        let mgr = sub.property_index_manager(&crud).unwrap();
        let label = label_of(&sub, "User");
        let property_key = sub
            .intern_table()
            .intern(TenantId::DEFAULT, "email")
            .unwrap();
        mgr.register_building(crate::storage::property_index::CreateIndexSpec {
            tenant: TenantId::DEFAULT,
            name: "user_email_building",
            if_not_exists: false,
            label,
            property_key,
            property_name: "email",
        })
        .expect("register building");
        // Precondition: the index really is Building (not planner-visible).
        assert!(
            !mgr.has_online_index(TenantId::DEFAULT, label, "email"),
            "index must be Building (not Online) for this test"
        );
        // The seam serves NO candidates for a Building index...
        let v = Value::String("b@x.com".into());
        assert_eq!(
            seam_ids(&sub, label, "email", &v),
            Vec::<u64>::new(),
            "RC-6: a Building index must serve NO candidates via the seam"
        );
        // ...even though the node IS live (the full scan finds it).
        assert_eq!(scan_ids(&sub, label, "email", &v), vec![node.raw()]);
    }

    /// `has_online_index` (the planner gate) is TRUE only after the flip
    /// to Online, FALSE while Building.
    #[test]
    fn has_online_index_true_only_when_online() {
        let (sub, crud, _mgr, _router) = fixture();
        let mgr = sub.property_index_manager(&crud).unwrap();
        let label = label_of(&sub, "User");
        let property_key = sub
            .intern_table()
            .intern(TenantId::DEFAULT, "email")
            .unwrap();
        // Building → not planner-visible.
        mgr.register_building(crate::storage::property_index::CreateIndexSpec {
            tenant: TenantId::DEFAULT,
            name: "ue",
            if_not_exists: false,
            label,
            property_key,
            property_name: "email",
        })
        .expect("register");
        assert!(!mgr.has_online_index(TenantId::DEFAULT, label, "email"));
        // A different, fully-created (Online) index IS planner-visible.
        sub.create_property_index(TenantId::DEFAULT, "ue_online", false, "User", "email")
            .expect("create online");
        assert!(
            mgr.has_online_index(TenantId::DEFAULT, label, "email"),
            "an Online index on (User, email) must be planner-visible"
        );
    }

    // ─── #1415 — the production-wire indexed==full-scan EQUIVALENCE ─────
    // The single test class that catches the REJECT-class silent-wrong-
    // results bug: for ANY DB state + ANY equality `$param`, the
    // `PropertyIndexScan` OP driven against the REAL CrudExecutorSubstrate
    // (its `property_index_lookup_with_context` + `value_is_indexable` +
    // `scan_nodes_with_context` seams) must return EXACTLY the rows a full
    // scan's `Filter(prop = v)` would. An unkeyable `$param` (fractional /
    // out-of-i64-range Float, negative Integer, List, Map) is routed to
    // the op UNCONDITIONALLY at plan time; pre-fix the op takes the empty
    // index candidate set verbatim and silently drops rows. This must be
    // RED at HEAD af663835 and GREEN with the op's scan-fallback.
    //
    // These live in arcgraph-mcp because `cargo test -p arcgraph-query`
    // never exercises the production seam (the stub cannot catch this).

    use arcgraph_core::datetime::ZonedDateTime;
    use arcgraph_query::ast::BinOp;
    use arcgraph_query::error::Span;
    use arcgraph_query::executor::ThreeValued;
    use arcgraph_query::executor::eval::{Parameters, evaluate};
    use arcgraph_query::executor::ops::PropertyIndexScanOp;
    use arcgraph_query::semantic::bound_ast::BoundPropertyRef;
    use arcgraph_query::semantic::{BindingId, BoundExpression};
    use proptest::prelude::*;

    /// FULL-SCAN ORACLE: the node ids a full scan's `Filter(prop = v)`
    /// keeps.
    ///
    /// **#1415 re-ultracode (test-oracle soundness).** This oracle MUST be
    /// a real `Scan(label) + Filter(prop = v)` — i.e. it keeps each node
    /// under the TRUE engine `=` `values_equal_3vl` (dropping BOTH `False`
    /// AND `Unknown` per 3VL `passes_filter`), EXACTLY as the #1415
    /// scan-fallback does at
    /// `arcgraph-query/src/executor/ops/property_index_scan.rs`. It does
    /// this by synthesizing the SAME `PropertyAccess(binding, property) =
    /// value_expr` equality the fallback builds and evaluating it over each
    /// hydrated node's `[Node]` row via [`evaluate`] → `values_equal_3vl`.
    ///
    /// It MUST NOT filter via `index_value_eq_coerced` (the seam's own
    /// recheck function). Doing so was tautological on the index path — the
    /// op and the oracle would both route through the SAME fn, so the gate
    /// could never catch a future coercion-drift where
    /// `index_value_eq_coerced` and `values_equal_3vl` disagree (e.g. a
    /// Temporal/Decimal `=` added to one but not the other). It ALSO
    /// diverges from a real scan on live-storable classes today —
    /// `List([Null])`, a `Temporal`, a stored `Null` all give
    /// `index_value_eq_coerced = KEEP` (structural `PartialEq`) but a real
    /// scan (`values_equal_3vl` → `passes_filter`) DROPS them (Unknown for
    /// `List([Null])`/`Null`; the `_ => Some(false)` type-mismatch arm for
    /// same-variant Temporal, which is out of scope for engine `=`). By
    /// single-sourcing the oracle onto `values_equal_3vl`, this gate now
    /// genuinely asserts `indexed == real-scan`. Sorted.
    fn full_scan_ids(
        sub: &CrudExecutorSubstrate,
        label: LabelId,
        property: &str,
        v: &Value,
    ) -> Vec<u64> {
        // The SAME equality predicate the scan-fallback carries as its
        // Filter: `n.<property> = $p`, rooted at binding 0 (the op's single
        // binding). Evaluated over a `[Node]` row, this reuses the engine
        // `=` (`values_equal_3vl`) verbatim.
        let span = Span::point(1, 1);
        let equality = BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(BoundExpression::PropertyAccess {
                base: Box::new(BoundExpression::VariableRef {
                    name: String::new(),
                    binding_id: BindingId::new(0),
                    span: span.clone(),
                    type_info: None,
                }),
                path: vec![BoundPropertyRef {
                    name: property.to_string(),
                    property_id: None,
                    span: span.clone(),
                }],
                span: span.clone(),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::Parameter {
                name: "p".into(),
                span: span.clone(),
                type_info: None,
            }),
            span,
            type_info: None,
        };
        let mut params = Parameters::new();
        params.insert("p".into(), v.clone());
        // Binding 0 maps to the single-column `[Node]` row's index 0.
        let binding0 = BindingId::new(0);
        let schema = move |b: BindingId| (b == binding0).then_some(0usize);

        let mut ids: Vec<u64> = sub
            .scan_nodes(TenantId::DEFAULT, Some(label), Lsn::MAX)
            .expect("scan OK")
            .into_iter()
            .filter(|bn| {
                let row = [Value::Node(bn.node.clone())];
                let eq = evaluate(&equality, &row, &schema, &params)
                    .expect("oracle equality evaluate OK");
                // 3VL: `False` AND `Unknown` both drop the row (byte-
                // identical to the planner's `Filter` and the #1415
                // scan-fallback at property_index_scan.rs:201).
                ThreeValued::from_value(&eq).passes_filter()
            })
            .map(|bn| bn.node.id.raw())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// INDEXED RESULT: drive the REAL `PropertyIndexScanOp` (index path OR
    /// #1415 scan-fallback) against the production substrate for an
    /// equality `$param` bound to `v`, exhausting every batch. Sorted ids.
    fn op_indexed_ids(
        sub: &CrudExecutorSubstrate,
        label: LabelId,
        property: &str,
        v: &Value,
    ) -> Vec<u64> {
        let mut params = Parameters::new();
        params.insert("p".into(), v.clone());
        let value_expr = BoundExpression::Parameter {
            name: "p".into(),
            span: Span::point(1, 1),
            type_info: None,
        };
        let mut op = PropertyIndexScanOp::new(
            BindingId::new(0),
            label,
            property.to_string(),
            value_expr,
            None,
            Lsn::MAX,
        )
        .with_parameters(params);
        let ctx = tctx();
        let mut ids = Vec::new();
        loop {
            let batch = op.next_batch(&ctx, sub).expect("op next_batch OK");
            if batch.is_empty() {
                break;
            }
            for r in 0..batch.row_count() {
                match &batch.row(r)[0] {
                    Value::Node(n) => ids.push(n.id.raw()),
                    other => panic!("op row must be [Node], got {other:?}"),
                }
            }
        }
        ids.sort_unstable();
        ids
    }

    /// Also compare against the RAW production seam for KEYABLE values (a
    /// keyable value MUST use the index, not the fallback) — proves the
    /// fix does not regress the fast path into an always-scan.
    fn seam_lookup_ids(
        sub: &CrudExecutorSubstrate,
        label: LabelId,
        property: &str,
        v: &Value,
    ) -> Vec<u64> {
        let mut ids: Vec<u64> = sub
            .property_index_lookup_with_context(&tctx(), label, property, v, Lsn::MAX)
            .expect("seam lookup OK")
            .into_iter()
            .map(|bn| bn.node.id.raw())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Seed N nodes with `label:Kind` and a `prop` value drawn from the
    /// supplied list (round-robin), create the Online index, return the
    /// label id. Values may be any storable type — String / Integer (incl.
    /// negative) / Boolean / Float (incl. fractional + |v|>i64::MAX) /
    /// List. (Map is write-fenced as a PROPERTY, so it appears only as a
    /// LOOKUP value below, never a stored one.)
    fn seed_indexed(
        sub: &CrudExecutorSubstrate,
        kind: &str,
        prop: &str,
        values: &[Value],
    ) -> LabelId {
        for v in values {
            sub.create_node(
                TenantId::DEFAULT,
                Some(kind),
                &[(prop.into(), v.clone())],
                &tctx(),
            )
            .expect("create node");
        }
        sub.create_property_index(TenantId::DEFAULT, "idx", false, kind, prop)
            .expect("create index");
        label_of(sub, kind)
    }

    /// A proptest generator over the full storable-value type span,
    /// including the bug's unkeyable classes.
    ///
    /// **#1415 re-ultracode.** The last three arms
    /// (`List([Null])` / `Temporal`) are the classes where the OLD
    /// `index_value_eq_coerced` oracle diverged from a real scan: structural
    /// `PartialEq` KEEPS them, but the engine `=` `values_equal_3vl` DROPS
    /// them (Unknown for a nested `Null`; the `_ => Some(false)`
    /// type-mismatch arm for same-variant Temporal, out of scope for engine
    /// `=` at this slice). They are live-storable node props today
    /// (`literal_to_value` at eval.rs maps `Literal::Temporal` /
    /// `Literal::List([Null])` straight to stored `Value`s), so exercising
    /// them as BOTH stored and lookup values proves the re-pointed oracle
    /// (a real `Scan + Filter(prop = v)` via `values_equal_3vl`) agrees with
    /// the op — which, for these unkeyable classes, itself takes the #1415
    /// scan-fallback (`canonical_key_for` returns `None`). All three are
    /// unkeyable, so they NEVER regress the index fast path.
    fn any_lookup_value() -> impl Strategy<Value = Value> {
        prop_oneof![
            // Keyable classes (must USE the index).
            "[a-z]{1,6}".prop_map(Value::String),
            (0i64..1000).prop_map(Value::Integer),
            any::<bool>().prop_map(Value::Boolean),
            // Integral in-range float (keyable → int bucket).
            (0i64..1000).prop_map(|n| Value::Float(n as f64)),
            // ── Unkeyable classes (the bug: index empty, scan matches) ──
            (-1000i64..0).prop_map(Value::Integer), // negative
            (1u32..100).prop_map(|n| Value::Float(f64::from(n) + 0.5)), // fractional
            Just(Value::Float(1e30)),               // |v| > i64::MAX
            Just(Value::Float(-1e30)),
            prop::collection::vec((0i64..10).prop_map(Value::Integer), 0..3).prop_map(Value::List),
            // ── #1415 re-ultracode: the oracle-divergent unkeyable classes.
            // A List containing a Null element: `values_equal_3vl` folds the
            // nested Null to Unknown → DROP; structural PartialEq kept it.
            Just(Value::List(vec![Value::Null])),
            // A stored/lookup Temporal: engine `=` hits the `_ => Some(false)`
            // type-mismatch arm (same-variant temporal equality is out of
            // scope) → DROP; structural PartialEq kept it. A small set of
            // distinct instants so a `stored == lookup` pair can occur.
            prop_oneof![
                Just(Value::Temporal(ZonedDateTime::from_utc_nanos(0))),
                Just(Value::Temporal(ZonedDateTime::from_utc_nanos(
                    1_700_000_000_000_000_000
                ))),
            ],
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(96))]

        /// THE equivalence gate: for a randomized DB state + a randomized
        /// equality `$param`, the indexed op result equals the full scan.
        #[test]
        fn indexed_op_equals_full_scan_over_all_value_types(
            stored in prop::collection::vec(any_lookup_value(), 1..12),
            lookup in any_lookup_value(),
        ) {
            let (sub, _crud, _mgr, _router) = fixture();
            let label = seed_indexed(&sub, "Thing", "v", &stored);
            let indexed = op_indexed_ids(&sub, label, "v", &lookup);
            let scan = full_scan_ids(&sub, label, "v", &lookup);
            prop_assert_eq!(
                indexed, scan,
                "indexed op must equal full scan for lookup {:?} over stored {:?}",
                lookup, stored
            );
        }

        /// No perf regression: a KEYABLE lookup value uses the INDEX seam,
        /// not a scan — the op result equals the raw seam result (so the
        /// fast path is preserved for the values that can key).
        #[test]
        fn keyable_lookup_uses_index_seam(
            stored in prop::collection::vec(any_lookup_value(), 1..12),
            // Keyable-only generator: String / non-neg in-range Int / Bool
            // / integral in-range Float.
            keyable in prop_oneof![
                "[a-z]{1,6}".prop_map(Value::String),
                (0i64..1000).prop_map(Value::Integer),
                any::<bool>().prop_map(Value::Boolean),
                (0i64..1000).prop_map(|n| Value::Float(n as f64)),
            ],
        ) {
            let (sub, _crud, _mgr, _router) = fixture();
            prop_assume!(crate::storage::property_index::canonical_key_for(&keyable).is_some());
            let label = seed_indexed(&sub, "Thing", "v", &stored);
            let op_ids = op_indexed_ids(&sub, label, "v", &keyable);
            let seam_ids = seam_lookup_ids(&sub, label, "v", &keyable);
            // The op for a keyable value takes the INDEX path, so its
            // result IS the seam's verified set (no residual here).
            prop_assert_eq!(
                op_ids.clone(), seam_ids,
                "keyable lookup {:?} must use the index seam (no scan fallback)",
                keyable
            );
            // And that index result still equals the full scan (soundness).
            prop_assert_eq!(op_ids, full_scan_ids(&sub, label, "v", &keyable));
        }
    }

    /// #1415 re-ultracode OPTIONAL NIT — tie the two coercion functions so a
    /// future ONE-SIDED change fails LOUD.
    ///
    /// The seam recheck (`index_value_eq_coerced`) and the engine `=`
    /// (`values_equal_3vl` → `passes_filter`) are two SEPARATE functions with
    /// no compiler-enforced link. For every KEYABLE value class (the ones the
    /// seam actually rechecks — String / non-negative in-range Integer /
    /// Boolean / integral in-range Float) they MUST agree; otherwise the seam
    /// could keep a candidate the engine `=` drops (or vice-versa) = silent
    /// wrong results. This asserts that agreement directly so a future
    /// Temporal/Decimal `=` (or any coercion tweak) added to one fn but not
    /// the other is caught here even if the proptest generator never reaches
    /// the exact divergent pair.
    #[test]
    fn seam_recheck_agrees_with_engine_eq_for_keyable_values() {
        // The keyable value universe (a representative cross-product incl. the
        // Int⇄Float coercion boundary the recheck special-cases).
        let keyable: Vec<Value> = vec![
            Value::String("alpha".into()),
            Value::String("beta".into()),
            Value::Boolean(true),
            Value::Boolean(false),
            Value::Integer(0),
            Value::Integer(7),
            Value::Integer(42),
            Value::Float(0.0),
            Value::Float(7.0),
            Value::Float(42.0),
        ];
        for a in &keyable {
            for b in &keyable {
                let seam = index_value_eq_coerced(a, b);
                // The engine `=`: `values_equal_3vl` → `passes_filter` drops
                // BOTH False AND Unknown (identical to a real Scan+Filter).
                let equality = BoundExpression::BinaryOp {
                    op: BinOp::Eq,
                    lhs: Box::new(BoundExpression::Literal {
                        value: value_to_literal(a),
                        span: Span::point(1, 1),
                        type_info: None,
                    }),
                    rhs: Box::new(BoundExpression::Literal {
                        value: value_to_literal(b),
                        span: Span::point(1, 1),
                        type_info: None,
                    }),
                    span: Span::point(1, 1),
                    type_info: None,
                };
                let no_binding = |_b: BindingId| -> Option<usize> { None };
                let eq = evaluate(&equality, &[], &no_binding, &Parameters::new())
                    .expect("keyable equality evaluate OK");
                let engine = ThreeValued::from_value(&eq).passes_filter();
                assert_eq!(
                    seam, engine,
                    "index_value_eq_coerced({a:?}, {b:?}) = {seam} must equal \
                     engine `=` values_equal_3vl(...).passes_filter() = {engine}"
                );
            }
        }
    }

    /// #1415 re-ultracode coverage-caveat closer: a DETERMINISTIC unit test
    /// asserting `index_value_eq_coerced(a, b)` equals the TRUE engine `=`
    /// `values_equal_3vl(...).passes_filter()` over a FIXED set of
    /// boundary/edge keyable-adjacent values — so a future one-sided change to
    /// EITHER coercion function fails LOUD regardless of proptest sampling luck
    /// (the 96-case random proptest may not deterministically hit the divergent
    /// pair; this test always does). Covers the Int⇄Float coercion boundary
    /// exactly where a wrong `=` would silently drift: i64 extremes, the 2^53
    /// float-integer-precision boundary, integral-float cross-type equality,
    /// and −0.0/0.0.
    #[test]
    fn coercion_fns_agree_on_fixed_edge_values_deterministic() {
        // Fixed edge/boundary values that stress the Int⇄Float coercion arm.
        let edges: Vec<Value> = vec![
            Value::Integer(0),
            Value::Integer(1),
            Value::Integer(-1),
            Value::Integer(i64::MAX),
            Value::Integer(i64::MIN),
            // 2^53: the largest integer exactly representable as f64; beyond
            // this, `as f64` coercion loses precision — the exact boundary a
            // one-sided coercion change would get wrong.
            Value::Integer(1i64 << 53),
            Value::Integer((1i64 << 53) + 1),
            Value::Float(0.0),
            Value::Float(-0.0),
            Value::Float(1.0),
            Value::Float(-1.0),
            Value::Float((1i64 << 53) as f64),
        ];
        for a in &edges {
            for b in &edges {
                let seam = index_value_eq_coerced(a, b);
                // The engine `=`: build `a = b` as a literal equality and
                // evaluate it through the SAME path a real Scan+Filter uses.
                let equality = BoundExpression::BinaryOp {
                    op: BinOp::Eq,
                    lhs: Box::new(BoundExpression::Literal {
                        value: value_to_literal(a),
                        span: Span::point(1, 1),
                        type_info: None,
                    }),
                    rhs: Box::new(BoundExpression::Literal {
                        value: value_to_literal(b),
                        span: Span::point(1, 1),
                        type_info: None,
                    }),
                    span: Span::point(1, 1),
                    type_info: None,
                };
                let no_binding = |_b: BindingId| -> Option<usize> { None };
                let eq = evaluate(&equality, &[], &no_binding, &Parameters::new())
                    .expect("edge equality evaluate OK");
                let engine = ThreeValued::from_value(&eq).passes_filter();
                assert_eq!(
                    seam, engine,
                    "coercion drift on fixed edge values: \
                     index_value_eq_coerced({a:?}, {b:?}) = {seam} \
                     must equal engine `=` values_equal_3vl(...).passes_filter() = {engine}"
                );
            }
        }
    }

    /// Lift a KEYABLE test `Value` back into the [`Literal`] the parser would
    /// have produced, so the engine `=` in
    /// [`seam_recheck_agrees_with_engine_eq_for_keyable_values`] evaluates over
    /// literal operands (no binding needed). Only the keyable scalar classes
    /// are supported — anything else panics (a test-author guard).
    fn value_to_literal(v: &Value) -> arcgraph_query::ast::Literal {
        use arcgraph_query::ast::Literal;
        match v {
            Value::String(s) => Literal::String(s.clone()),
            Value::Boolean(b) => Literal::Bool(*b),
            Value::Integer(i) => Literal::Integer(*i),
            Value::Float(f) => Literal::Float(*f),
            other => panic!("value_to_literal: unsupported keyable value {other:?}"),
        }
    }

    /// Named drop-in RED regression #1 (verdict): a fractional-float
    /// `$param` is unkeyable → pre-fix the op returns EMPTY while the full
    /// scan finds the node. RED at HEAD; GREEN with the op scan-fallback.
    #[test]
    fn skeptic_fractional_float_param_indexed_vs_scan() {
        let (sub, _crud, _mgr, _router) = fixture();
        // A stored fractional-float property + a non-matching sibling.
        let label = seed_indexed(
            &sub,
            "Product",
            "price",
            &[Value::Float(19.99), Value::Float(5.0), Value::Float(19.99)],
        );
        let lookup = Value::Float(19.99);
        let indexed = op_indexed_ids(&sub, label, "price", &lookup);
        let scan = full_scan_ids(&sub, label, "price", &lookup);
        // Two nodes carry price=19.99; the full scan finds both.
        assert_eq!(scan.len(), 2, "full scan finds both price=19.99 nodes");
        assert_eq!(
            indexed, scan,
            "fractional-float param: the op MUST fall back to scan, not return empty"
        );
    }

    /// Named drop-in RED regression #2 (verdict): a HUGE integral float
    /// (|v| > i64::MAX) `$param` is unkeyable (out of the i64 range even
    /// though it is integral) → pre-fix the op returns EMPTY while the
    /// full scan finds the stored-float node. RED at HEAD; GREEN post-fix.
    #[test]
    fn skeptic_huge_integral_float_param_indexed_vs_scan() {
        let (sub, _crud, _mgr, _router) = fixture();
        let huge = 1e30_f64; // integral (fract()==0) but |v| >> i64::MAX
        assert!(
            crate::storage::property_index::canonical_key_for(&Value::Float(huge)).is_none(),
            "1e30 must be unkeyable (out of i64 range)"
        );
        let label = seed_indexed(
            &sub,
            "Measure",
            "reading",
            &[Value::Float(huge), Value::Float(1.0)],
        );
        let lookup = Value::Float(huge);
        let indexed = op_indexed_ids(&sub, label, "reading", &lookup);
        let scan = full_scan_ids(&sub, label, "reading", &lookup);
        assert_eq!(scan, vec![1u64], "full scan finds the huge-float node");
        assert_eq!(
            indexed, scan,
            "huge integral-float param: op MUST fall back to scan, not return empty"
        );
    }
}
