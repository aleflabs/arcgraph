//! #765 PART-1 — served HNSW vector-search provider.
//! #1292 PART-3 — served SSD-resident DiskANN tier (ADR-195).
//!
//! The concrete [`SubstrateSearchProvider`] impls ADR-132 + ADR-087 forward-pinned
//! to "the v1.0-GA bootstrap-wiring slice" (#765). They bind the real KNN body
//! behind MCP `graph.search` (via [`arcgraph_mcp::storage::StorageHybridSearcher`]),
//! ArcQL `RANK BY vector(...)`, and Bolt. Lives in `arcgraph-cli` (the wiring
//! layer) because `arcgraph-mcp` defines the trait but must NOT depend on
//! `arcgraph-vector` under the bounded-context rules.
//!
//! **#1292 (PART-3 wiring):** The SSD-resident DiskANN serving tier (ADR-195)
//! swaps the HNSW provider for a RAM-decoupled version (SsdVectorSearchProvider)
//! that bounds RSS at ~14 GB for 10M×768 ingest vs. the HNSW path's unbounded
//! full-f32 RAM (OOMs at ~30 GB). Tier selection is config-driven: HNSW by
//! default (for small ingests), SSD tier opt-in (for 10M+ scale). The RSS guard
//! detects-and-aborts gracefully instead of OOM-kill.
//!
//! **HNSW (PART-1):** The per-tenant HNSW is a DERIVED index (D4): ephemeral,
//! lazily built from the tenant's durable nodes' `embedding` property (the
//! WAL-durable nodes are the source of truth — like the M4-41 `CatalogStats`
//! cold-start rebuild), bare [`HnswGraph`] (`L2F32`, dim inferred+validated)
//! behind the trait so PART-3 (DiskANN-served, ADR-189/195) swaps the impl.
//!
//! ## #787 — incremental, vector-aware maintenance (read-after-write)
//!
//! The cached index is brought up to date INCREMENTALLY: a query that follows an
//! ingest delta-scans only the newly-allocated node ids
//! (`(built_high_water, current_high_water]` via
//! [`CrudExecutorSubstrate::scan_nodes_in_id_range`], `O(delta)`) and
//! [`HnswGraph::insert`]s only the new embedding-bearing nodes into the EXISTING
//! graph. The cost of a read-after-write is therefore `warm + O(delta)`, NOT the
//! `O(N)` full scan + rebuild the first cut paid on every query after any ingest
//! (the #787 cliff). A NON-vector ingest contributes zero inserts, so the index
//! is reused untouched (the vector-aware invalidation #787 also asked for).
//!
//! ### Invalidation signal + its bound
//!
//! The freshness key is the per-tenant node high-water (an append-only allocator
//! watermark). Inserts advance it; an in-place served-vector UPDATE or DELETE of
//! an already-indexed node does NOT (the watermark never decreases). Slice A handles
//! DELETE through the per-tenant tombstone set pushed by
//! `mark_vector_node_deleted` and applied as a query-time filter, so deleted ids
//! are not returned even if they remain in HNSW. Slice B handles embedding
//! UPDATE through `mark_vector_node_updated`: the next property-scoped search
//! marks the old resident `VectorId`s stale and inserts the node's current
//! property value with a fresh `VectorId`, with no storage watermark.
//!
//! Follow-ons: PART-2 durable persist via `VectorPageStore` (+ delete/update
//! invalidation); PART-3 DiskANN; BM25 served substrate.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_mcp::storage::{
    CrudExecutorSubstrate, StorageBackend, SubstrateSearchProvider, property_payload,
};
use arcgraph_query::executor::substrate::{ExecutorSubstrate, RankedHit, SubstrateAccessError};
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_storage::crud;
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use arcgraph_vector::VectorId;
use arcgraph_vector::diskann::rss_guard::{DEFAULT_RSS_CAP_MB, RssGuard};
use arcgraph_vector::diskann::ssd::{NavQuantizer, SsdBuildConfig, SsdDiskAnnIndex};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{HnswGraph, HnswParams};
use arcgraph_vector::quantizer::Sq8Trainer;
use arcgraph_vector::{Metric, VectorIndexError};
use parking_lot::RwLock;

/// Vector search provider tier selection and configuration.
///
/// Defaults to HNSW (in-memory, unbounded RAM); opt-in to SSD tier for large ingests.
#[derive(Debug, Clone, Default)]
pub enum VectorSearchTier {
    /// Ephemeral in-memory HNSW: no RSS limit, fast small-scale queries,
    /// OOMs at ~30 GB for 10M×768.
    #[default]
    Hnsw,
    /// RAM-decoupled SSD-resident DiskANN (ADR-195): bounded RSS (~14 GB @10M),
    /// slower queries but RSS-safe large ingest. Requires a directory path for
    /// the f32 page store.
    Ssd {
        /// Path where the SSD tier builds/opens its index directory.
        index_dir: PathBuf,
        /// RSS ceiling in MB (default `DEFAULT_RSS_CAP_MB` = 14000). The guard
        /// detects-and-aborts if exceeded instead of OOM-kill.
        rss_cap_mb: u64,
    },
}

impl VectorSearchTier {
    /// Environment variable selecting the served vector tier: `ssd` | `hnsw`
    /// (default `hnsw`, so nothing regresses without an explicit opt-in).
    pub const TIER_ENV: &'static str = "ARCGRAPH_VECTOR_TIER";
    /// Environment variable overriding the SSD tier's index directory
    /// (default: `<system temp>/arcgraph-vector-ssd`).
    pub const DIR_ENV: &'static str = "ARCGRAPH_VECTOR_SSD_DIR";
    /// Environment variable overriding the SSD RSS ceiling in MB
    /// (default `DEFAULT_RSS_CAP_MB` = 14000).
    pub const RSS_CAP_ENV: &'static str = "ARCGRAPH_VECTOR_RSS_CAP_MB";

    /// Opt-in to the SSD tier with the given index directory and default RSS cap.
    #[must_use]
    pub fn ssd_with_dir(index_dir: PathBuf) -> Self {
        Self::Ssd {
            index_dir,
            rss_cap_mb: DEFAULT_RSS_CAP_MB,
        }
    }

    /// Opt-in to the SSD tier with a custom RSS cap.
    #[must_use]
    pub fn ssd_with_cap(index_dir: PathBuf, rss_cap_mb: u64) -> Self {
        Self::Ssd {
            index_dir,
            rss_cap_mb,
        }
    }

    /// Resolve the served vector tier from the process environment.
    ///
    /// - `ARCGRAPH_VECTOR_TIER=ssd` (case-insensitive) → the RAM-decoupled SSD
    ///   tier (ADR-195). `ARCGRAPH_VECTOR_SSD_DIR` overrides the index directory
    ///   (default `<temp>/arcgraph-vector-ssd`); `ARCGRAPH_VECTOR_RSS_CAP_MB`
    ///   overrides the RSS ceiling (default `DEFAULT_RSS_CAP_MB` = 14000).
    /// - anything else (unset, `hnsw`, empty) → [`VectorSearchTier::Hnsw`]
    ///   (the pre-#1292 default; nothing regresses without an explicit opt-in).
    ///
    /// `data_dir` (the serve `--data <dir>` root, when durable) is preferred as
    /// the parent for the SSD index directory so the f32 page store co-locates
    /// with the durable dataset; `None` (in-memory serve) falls back to the
    /// system temp dir.
    #[must_use]
    pub fn from_env(data_dir: Option<&std::path::Path>) -> Self {
        let tier = std::env::var(Self::TIER_ENV).unwrap_or_default();
        if !tier.eq_ignore_ascii_case("ssd") {
            return Self::Hnsw;
        }
        let index_dir = std::env::var_os(Self::DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                let parent = data_dir
                    .map(std::path::Path::to_path_buf)
                    .unwrap_or_else(std::env::temp_dir);
                parent.join("arcgraph-vector-ssd")
            });
        let rss_cap_mb = std::env::var(Self::RSS_CAP_ENV)
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_RSS_CAP_MB);
        Self::Ssd {
            index_dir,
            rss_cap_mb,
        }
    }

    /// Construct the concrete [`SubstrateSearchProvider`] for this tier over the
    /// given storage backend. The single factory both wiring sites (stdio/HTTP
    /// dispatcher + Bolt handler) call so the tier decision is made exactly once
    /// per served-provider construction and both transports agree.
    ///
    /// HNSW stays the default so a serve with no `ARCGRAPH_VECTOR_TIER=ssd`
    /// behaves exactly as pre-#1292.
    #[must_use]
    pub fn build_provider(&self, backend: StorageBackend) -> Arc<dyn SubstrateSearchProvider> {
        match self {
            Self::Hnsw => Arc::new(HnswVectorSearchProvider::new(backend)),
            Self::Ssd {
                index_dir,
                rss_cap_mb,
            } => {
                tracing::info!(
                    target: "arcgraph_cli::vector_search",
                    index_dir = %index_dir.display(),
                    rss_cap_mb = *rss_cap_mb,
                    "served vector tier: SSD-resident DiskANN (ADR-195, RAM-decoupled, \
                     RSS ceiling enforced)"
                );
                Arc::new(SsdVectorSearchProvider::new(
                    backend,
                    index_dir.clone(),
                    *rss_cap_mb,
                ))
            }
        }
    }
}

/// The mutable per-`(tenant, property)` index state, guarded by the enclosing
/// [`TenantHnsw`]'s lock. Mutated in place by the #787 incremental delta-insert
/// so the graph persists across queries (never re-allocated from scratch on a
/// read-after-write).
struct HnswState {
    /// `None` until the tenant's first embedding-bearing node is seen
    /// (an empty index → search returns no hits, honestly, never an error).
    graph: Option<HnswGraph>,
    /// Inferred vector dimension (components). `0` when `graph` is `None`.
    dim: usize,
    /// `VectorId(i)` → the node it was built from. Indexed by `VectorId.0`.
    /// Append-only + dense: `VectorId`s are allocated as `map.len()` so they
    /// stay stable as the graph grows incrementally.
    map: Vec<(NodeId, Option<LabelId>)>,
    /// Resident vectors superseded by an embedding UPDATE. HNSW is insert-only,
    /// so old vectors stay in the graph but are filtered from results.
    stale_vectors: HashSet<VectorId>,
    /// Node high-water mark this index has consumed up to: every node id
    /// `<= built_high_water` has already been considered for insertion. The
    /// delta-scan resumes from here on the next read-after-write.
    built_high_water: u64,
}

impl HnswState {
    fn empty() -> Self {
        Self {
            graph: None,
            dim: 0,
            map: Vec::new(),
            stale_vectors: HashSet::new(),
            built_high_water: 0,
        }
    }
}

/// A per-`(tenant, property)` derived HNSW index plus the `VectorId → node`
/// sidecar map (the engine's `VectorId` is a dense `u32` allocated as we
/// insert; `NodeId` is a `u64`, so a sidecar is required — never a downcast).
///
/// The state is behind an `RwLock`: concurrent searches take a read lock; a
/// read-after-write that must extend the graph takes the write lock, brings the
/// index up to date incrementally, then searches under that guard.
struct TenantHnsw {
    state: RwLock<HnswState>,
}

struct Bm25State {
    built_high_water: u64,
}

impl Bm25State {
    fn empty() -> Self {
        Self {
            built_high_water: 0,
        }
    }
}

struct TenantBm25 {
    state: RwLock<Bm25State>,
}

struct SearchOptions<'a> {
    k: u64,
    ef: usize,
    label_filter: Option<&'a [LabelId]>,
    deleted_nodes: Option<&'a HashSet<NodeId>>,
}

/// Lifetime observability counters for the served provider. Exposed via
/// [`HnswVectorSearchProvider::metrics`] — used by the #787 regression test as a
/// STRONG oracle (a read-after-write must add exactly `delta` inserts + scan
/// exactly `delta` nodes, never `O(N)`), and useful for production telemetry.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProviderMetrics {
    /// Cumulative count of `HnswGraph::insert` calls across all tenants. A
    /// full O(N) rebuild would bump this by `N`; the incremental path bumps it
    /// by the number of NEW embedding nodes only.
    pub vectors_inserted: u64,
    /// Cumulative count of nodes returned by delta-scans (the per-query work to
    /// bring an index up to date). A read-after-write scans only the delta.
    pub nodes_scanned: u64,
}

/// Served HNSW [`SubstrateSearchProvider`] over the workspace storage backend.
///
/// One instance per process; shared (`Arc`) across the two search call paths.
/// The per-tenant HNSW cache is lazy + maintained incrementally (#787).
pub struct HnswVectorSearchProvider {
    backend: StorageBackend,
    cache: RwLock<HashMap<(TenantId, String), Arc<TenantHnsw>>>,
    bm25_cache: RwLock<HashMap<TenantId, Arc<TenantBm25>>>,
    deleted_nodes: RwLock<HashMap<TenantId, HashSet<NodeId>>>,
    bm25_reindex_nodes: RwLock<HashMap<TenantId, HashSet<NodeId>>>,
    reindex_nodes: RwLock<HashMap<(TenantId, String), HashSet<NodeId>>>,
    vectors_inserted: AtomicU64,
    nodes_scanned: AtomicU64,
}

impl std::fmt::Debug for HnswVectorSearchProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HnswVectorSearchProvider")
            .field("cached_indexes", &self.cache.read().len())
            .field("cached_bm25_indexes", &self.bm25_cache.read().len())
            .field(
                "vectors_inserted",
                &self.vectors_inserted.load(Ordering::Relaxed),
            )
            .finish()
    }
}

/// Encode an f32 slice as the native-endian byte layout the `simsimd`-backed
/// `L2F32` kernel reads (`Encoding::F32` = `dim * 4` bytes, raw f32). Matches
/// the `bytemuck::cast_slice` layout the HNSW insert/search paths expect.
fn encode_f32(v: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for &f in v {
        bytes.extend_from_slice(&f.to_ne_bytes());
    }
    bytes
}

/// Pull a node property as a `Vec<f32>`, returning `None` when the property is
/// absent or not a numeric list. `Value::List` of `Integer`/`Float` elements
/// (the `graph.ingest` JSON-array round-trip) maps via `Value::as_f64`.
fn property_as_vector(
    props: &std::collections::BTreeMap<String, Value>,
    key: &str,
) -> Option<Vec<f32>> {
    let Value::List(items) = props.get(key)? else {
        return None;
    };
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        out.push(it.as_f64()? as f32);
    }
    if out.is_empty() { None } else { Some(out) }
}

/// BM25 v1 text convention: concatenate every string-valued node property in
/// key order. This keeps `text` working as the common case while indexing
/// `title`, `body`, or other string fields ingested through the same property
/// bag without another schema knob.
fn indexable_string_text(props: &std::collections::BTreeMap<String, Value>) -> Option<String> {
    let mut out = String::new();
    for value in props.values() {
        if let Value::String(text) = value {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(text);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

impl HnswVectorSearchProvider {
    /// Construct a provider over the workspace storage backend.
    #[must_use]
    pub fn new(backend: StorageBackend) -> Self {
        Self {
            backend,
            cache: RwLock::new(HashMap::new()),
            bm25_cache: RwLock::new(HashMap::new()),
            deleted_nodes: RwLock::new(HashMap::new()),
            bm25_reindex_nodes: RwLock::new(HashMap::new()),
            reindex_nodes: RwLock::new(HashMap::new()),
            vectors_inserted: AtomicU64::new(0),
            nodes_scanned: AtomicU64::new(0),
        }
    }

    /// Snapshot the lifetime observability counters.
    #[must_use]
    pub fn metrics(&self) -> ProviderMetrics {
        ProviderMetrics {
            vectors_inserted: self.vectors_inserted.load(Ordering::Relaxed),
            nodes_scanned: self.nodes_scanned.load(Ordering::Relaxed),
        }
    }

    /// The tenant's current node high-water mark (the freshness key).
    fn current_high_water(&self, tenant: TenantId) -> Result<u64, SubstrateAccessError> {
        let handle = self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .map_err(|_| SubstrateAccessError::TenantUnknown(tenant))?;
        Ok(handle.crud().node_high_water(tenant))
    }

    fn resolve_label_name(&self, tenant: TenantId, label_id: u32) -> Option<String> {
        if label_id == 0 {
            return None;
        }
        self.backend
            .intern_table()
            .resolve(tenant, arcgraph_core::ids::StringId::new(label_id))
            .map(|arc| arc.to_string())
    }

    /// Get-or-create the persistent per-`(tenant, property)` index slot. The
    /// slot (and its graph) lives across queries so the #787 incremental path
    /// can extend it instead of rebuilding.
    fn slot_for(&self, tenant: TenantId, property: &str) -> Arc<TenantHnsw> {
        let key = (tenant, property.to_string());
        if let Some(slot) = self.cache.read().get(&key) {
            return Arc::clone(slot);
        }
        let mut w = self.cache.write();
        // Re-check under the write lock (another thread may have created it).
        Arc::clone(w.entry(key).or_insert_with(|| {
            Arc::new(TenantHnsw {
                state: RwLock::new(HnswState::empty()),
            })
        }))
    }

    fn bm25_slot_for(&self, tenant: TenantId) -> Arc<TenantBm25> {
        if let Some(slot) = self.bm25_cache.read().get(&tenant) {
            return Arc::clone(slot);
        }
        let mut w = self.bm25_cache.write();
        Arc::clone(w.entry(tenant).or_insert_with(|| {
            Arc::new(TenantBm25 {
                state: RwLock::new(Bm25State::empty()),
            })
        }))
    }

    fn bm25_service(
        &self,
        tenant: TenantId,
    ) -> Result<Arc<dyn Bm25IndexStoreHandle>, SubstrateAccessError> {
        let handle = self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .map_err(|_| SubstrateAccessError::TenantUnknown(tenant))?;
        handle
            .bm25()
            .cloned()
            .ok_or_else(|| SubstrateAccessError::IndexUnavailable("bm25".into()))
    }

    fn bm25_upsert_node(
        &self,
        bm25: &Arc<dyn Bm25IndexStoreHandle>,
        tenant: TenantId,
        node: &NodeView,
        commit_lsn: Lsn,
    ) -> Result<bool, SubstrateAccessError> {
        if self
            .deleted_nodes
            .read()
            .get(&tenant)
            .is_some_and(|deleted| deleted.contains(&node.id))
        {
            return Ok(false);
        }
        let Some(text) = indexable_string_text(&node.properties) else {
            return Ok(false);
        };
        bm25.upsert_document(tenant, node.id, &text, commit_lsn)
            .map_err(|e| SubstrateAccessError::Io(format!("bm25 upsert failed: {e}")))?;
        Ok(true)
    }

    fn commit_bm25_pending(
        bm25: &Arc<dyn Bm25IndexStoreHandle>,
        tenant: TenantId,
    ) -> Result<(), SubstrateAccessError> {
        bm25.commit_pending(tenant)
            .map_err(|e| SubstrateAccessError::Io(format!("bm25 commit failed: {e}")))
    }

    fn bring_bm25_up_to_date(
        &self,
        st: &mut Bm25State,
        tenant: TenantId,
        bm25: &Arc<dyn Bm25IndexStoreHandle>,
        read_lsn: Lsn,
    ) -> Result<(), SubstrateAccessError> {
        let target_high_water = self.current_high_water(tenant)?;
        if st.built_high_water >= target_high_water {
            return Ok(());
        }

        let scanner = CrudExecutorSubstrate::new(
            Arc::clone(self.backend.router()),
            Arc::clone(self.backend.txn_manager()),
            Arc::clone(self.backend.intern_table()),
        );
        let new_nodes = scanner.scan_nodes_in_id_range(
            tenant,
            st.built_high_water,
            target_high_water,
            read_lsn,
        )?;
        let mut wrote = false;
        for bound in new_nodes {
            wrote |= self.bm25_upsert_node(bm25, tenant, &bound.node, read_lsn)?;
        }
        if wrote {
            Self::commit_bm25_pending(bm25, tenant)?;
        }
        st.built_high_water = target_high_water;
        Ok(())
    }

    fn process_bm25_reindex_nodes(
        &self,
        tenant: TenantId,
        bm25: &Arc<dyn Bm25IndexStoreHandle>,
        read_lsn: Lsn,
    ) -> Result<(), SubstrateAccessError> {
        let pending: Vec<NodeId> = self
            .bm25_reindex_nodes
            .read()
            .get(&tenant)
            .map(|nodes| nodes.iter().copied().collect())
            .unwrap_or_default();
        if pending.is_empty() {
            return Ok(());
        }

        let scanner = CrudExecutorSubstrate::new(
            Arc::clone(self.backend.router()),
            Arc::clone(self.backend.txn_manager()),
            Arc::clone(self.backend.intern_table()),
        );
        let mut wrote = false;
        for node in pending {
            if self
                .deleted_nodes
                .read()
                .get(&tenant)
                .is_some_and(|deleted| deleted.contains(&node))
            {
                self.bm25_reindex_nodes
                    .write()
                    .entry(tenant)
                    .or_default()
                    .remove(&node);
                continue;
            }
            let mut rows = scanner.scan_nodes_in_id_range(
                tenant,
                node.raw().saturating_sub(1),
                node.raw(),
                read_lsn,
            )?;
            if let Some(bound) = rows.drain(..).find(|bound| bound.node.id == node) {
                if self.bm25_upsert_node(bm25, tenant, &bound.node, read_lsn)? {
                    wrote = true;
                } else {
                    bm25.delete_document(tenant, node, Lsn::MAX).map_err(|e| {
                        SubstrateAccessError::Io(format!("bm25 delete failed: {e}"))
                    })?;
                    wrote = true;
                }
            } else {
                bm25.delete_document(tenant, node, read_lsn)
                    .map_err(|e| SubstrateAccessError::Io(format!("bm25 delete failed: {e}")))?;
                wrote = true;
            }
            self.bm25_reindex_nodes
                .write()
                .entry(tenant)
                .or_default()
                .remove(&node);
        }
        if wrote {
            Self::commit_bm25_pending(bm25, tenant)?;
        }
        Ok(())
    }

    fn ensure_bm25_up_to_date(
        &self,
        tenant: TenantId,
        bm25: &Arc<dyn Bm25IndexStoreHandle>,
        read_lsn: Lsn,
    ) -> Result<(), SubstrateAccessError> {
        let slot = self.bm25_slot_for(tenant);
        let target_high_water = self.current_high_water(tenant)?;
        let has_reindex = || {
            self.bm25_reindex_nodes
                .read()
                .get(&tenant)
                .is_some_and(|nodes| !nodes.is_empty())
        };
        {
            let st = slot.state.read();
            if st.built_high_water == target_high_water && !has_reindex() {
                return Ok(());
            }
        }
        let mut wst = slot.state.write();
        self.process_bm25_reindex_nodes(tenant, bm25, read_lsn)?;
        self.bring_bm25_up_to_date(&mut wst, tenant, bm25, read_lsn)
    }

    /// Bring `st` up to the tenant's current high-water by delta-scanning only
    /// the newly-allocated node ids and incrementally inserting the new
    /// embedding-bearing nodes (#787). Idempotent under concurrent writers (a
    /// no-op once `built_high_water` has caught up).
    fn bring_up_to_date(
        &self,
        st: &mut HnswState,
        tenant: TenantId,
        property: &str,
    ) -> Result<(), SubstrateAccessError> {
        let target_high_water = self.current_high_water(tenant)?;
        if st.built_high_water >= target_high_water {
            // Already fresh — another writer caught us up, or only the
            // freshness race lost (the high-water did not actually advance).
            return Ok(());
        }

        // O(delta) scan: only the node ids allocated since the last build.
        // Reuses the production scan + property-decode path so the served
        // index sees exactly what a MATCH would.
        let scanner = CrudExecutorSubstrate::new(
            Arc::clone(self.backend.router()),
            Arc::clone(self.backend.txn_manager()),
            Arc::clone(self.backend.intern_table()),
        );
        let new_nodes = scanner.scan_nodes_in_id_range(
            tenant,
            st.built_high_water,
            target_high_water,
            Lsn::MAX,
        )?;
        self.nodes_scanned
            .fetch_add(new_nodes.len() as u64, Ordering::Relaxed);

        let kernel = L2F32;
        let mut skipped_dim = 0usize;
        for bn in new_nodes {
            let Some(vec) = property_as_vector(&bn.node.properties, property) else {
                // No (or non-numeric) embedding — not a vector node. Vector-
                // aware: a non-vector ingest contributes zero work here.
                continue;
            };
            if st.graph.is_none() {
                st.dim = vec.len();
                st.graph = Some(HnswGraph::new(HnswParams::default(), st.dim, &kernel));
            }
            if vec.len() != st.dim {
                // #786 — single-dimension-per-index: a wrong-dim embedding is
                // NOT silently dropped. Count it; an aggregated WARN fires
                // below (ingest-time validation rejects single-batch mismatches
                // up front — see `StorageIngestProvider::ingest`).
                skipped_dim += 1;
                continue;
            }
            let vid = VectorId::new(st.map.len() as u32);
            let bytes = encode_f32(&vec);
            st.graph
                .as_mut()
                .expect("graph is Some once dim is set")
                .insert(vid, &bytes, &kernel)
                .map_err(|e| {
                    SubstrateAccessError::Io(format!("served HNSW incremental insert: {e}"))
                })?;
            st.map.push((bn.node.id, bn.node.label));
            self.vectors_inserted.fetch_add(1, Ordering::Relaxed);
        }

        if skipped_dim > 0 {
            tracing::warn!(
                target: "arcgraph_cli::vector_search",
                tenant = tenant.raw(),
                property = %property,
                skipped = skipped_dim,
                index_dim = st.dim,
                "served HNSW: skipped {skipped_dim} node(s) whose embedding dimension does \
                 not match the index dimension {}; they are absent from vector search \
                 (single-dimension-per-index, #786)",
                st.dim,
            );
        }

        st.built_high_water = target_high_water;
        Ok(())
    }

    /// Drain pending vector-property UPDATEs for nodes already represented in this
    /// resident index. UPDATEs for ids above `built_high_water` are discarded
    /// here because the normal #787 delta insert will read their current
    /// embedding. DELETE wins: a deleted node is never reinserted.
    fn process_reindex_nodes(
        &self,
        st: &mut HnswState,
        tenant: TenantId,
        property: &str,
    ) -> Result<(), SubstrateAccessError> {
        let reindex_key = (tenant, property.to_string());
        let pending: Vec<NodeId> = self
            .reindex_nodes
            .read()
            .get(&reindex_key)
            .map(|nodes| nodes.iter().copied().collect())
            .unwrap_or_default();
        if pending.is_empty() {
            return Ok(());
        }

        let scanner = CrudExecutorSubstrate::new(
            Arc::clone(self.backend.router()),
            Arc::clone(self.backend.txn_manager()),
            Arc::clone(self.backend.intern_table()),
        );
        let kernel = L2F32;
        for node in pending {
            if self
                .deleted_nodes
                .read()
                .get(&tenant)
                .is_some_and(|deleted| deleted.contains(&node))
            {
                self.reindex_nodes
                    .write()
                    .entry(reindex_key.clone())
                    .or_default()
                    .remove(&node);
                continue;
            }
            if node.raw() > st.built_high_water {
                self.reindex_nodes
                    .write()
                    .entry(reindex_key.clone())
                    .or_default()
                    .remove(&node);
                continue;
            }

            let old_vids: Vec<VectorId> = st
                .map
                .iter()
                .enumerate()
                .filter(|(_, (mapped_node, _))| *mapped_node == node)
                .map(|(idx, _)| VectorId::new(idx as u32))
                .filter(|vid| !st.stale_vectors.contains(vid))
                .collect();

            for vid in &old_vids {
                st.stale_vectors.insert(*vid);
            }

            let mut rows = scanner.scan_nodes_in_id_range(
                tenant,
                node.raw().saturating_sub(1),
                node.raw(),
                Lsn::MAX,
            )?;
            let Some(bound) = rows.drain(..).find(|bound| bound.node.id == node) else {
                self.reindex_nodes
                    .write()
                    .entry(reindex_key.clone())
                    .or_default()
                    .remove(&node);
                continue;
            };
            let Some(vec) = property_as_vector(&bound.node.properties, property) else {
                self.reindex_nodes
                    .write()
                    .entry(reindex_key.clone())
                    .or_default()
                    .remove(&node);
                continue;
            };
            if st.graph.is_none() {
                st.dim = vec.len();
                st.graph = Some(HnswGraph::new(HnswParams::default(), st.dim, &kernel));
            }
            if vec.len() != st.dim {
                tracing::warn!(
                    target: "arcgraph_cli::vector_search",
                    tenant = tenant.raw(),
                    property = %property,
                    node = node.raw(),
                    index_dim = st.dim,
                    update_dim = vec.len(),
                    "served HNSW: skipped re-index for updated node whose embedding dimension \
                     does not match the resident index dimension",
                );
                self.reindex_nodes
                    .write()
                    .entry(reindex_key.clone())
                    .or_default()
                    .remove(&node);
                continue;
            }

            let vid = VectorId::new(st.map.len() as u32);
            let bytes = encode_f32(&vec);
            st.graph
                .as_mut()
                .expect("graph is initialized before re-index insert")
                .insert(vid, &bytes, &kernel)
                .map_err(|e| {
                    SubstrateAccessError::Io(format!("served HNSW re-index insert: {e}"))
                })?;
            st.map.push((bound.node.id, bound.node.label));
            self.vectors_inserted.fetch_add(1, Ordering::Relaxed);
            self.reindex_nodes
                .write()
                .entry(reindex_key.clone())
                .or_default()
                .remove(&node);
        }
        Ok(())
    }

    /// Run the KNN over an already-fresh state snapshot. Stateless on the
    /// provider; reads only `st`.
    ///
    /// `ef` is the query-time beam width (#816a): `0` → the engine's
    /// `HnswParams::ef_search` default (128, the back-compat behavior);
    /// `> 0` trades recall for latency.
    ///
    /// `label_filter` (#815): `None` → the standard distance-only beam.
    /// `Some(&[..])` non-empty → filter-during-search via
    /// [`arcgraph_vector::hnsw::predicate_filtered_search`], pushing the
    /// label predicate INTO the traversal so a SELECTIVE filter returns
    /// `k` true matches (not the `k · selectivity` a post-filter yields).
    /// `Some(&[])` → an all-unknown filter resolved to nothing → honest
    /// empty result (never a silent full scan, never "match everything").
    /// The predicate reads the resident `VectorId → label` map, so there
    /// is NO per-candidate store hit and the standard `O(log N)`
    /// [`HnswGraph::insert`] (no payload-edge augmentation) is preserved.
    fn search_state(
        &self,
        st: &HnswState,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        opts: SearchOptions<'_>,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        let Some(graph) = st.graph.as_ref() else {
            // No vectors ingested for this property yet → no hits (honest empty
            // result, never a silent error).
            return Ok(Vec::new());
        };
        // Dimension mismatch is a structured, CLIENT-facing error (#786): the
        // MCP boundary maps `DimensionMismatch` to `-32602 invalid params` with
        // the dims, never the cryptic `-32006 execution eval` the generic `Io`
        // bucket rendered (per the #765 honesty gate +
        // `feedback_review_oracle_relaxations`).
        if query_vec.len() != st.dim {
            return Err(SubstrateAccessError::DimensionMismatch {
                property: property.to_string(),
                query_dim: query_vec.len(),
                index_dim: st.dim,
            });
        }
        let query_bytes = encode_f32(query_vec);
        // #815 + #909 Slice A — when a label filter OR tombstone set is
        // present, push the predicate INTO the HNSW traversal. The predicate
        // is evaluated against the resident `VectorId → (NodeId, label)`
        // sidecar, so deleted-node exclusion is O(1) per candidate with no
        // store round-trip and no storage-format change.
        let raw = match opts.label_filter {
            Some([]) => return Ok(Vec::new()),
            Some(allow) => {
                let is_allowed = |vid: VectorId| {
                    st.map.get(vid.0 as usize).is_some_and(|(node_id, label)| {
                        !opts
                            .deleted_nodes
                            .is_some_and(|deleted| deleted.contains(node_id))
                            && !st.stale_vectors.contains(&vid)
                            && label.is_some_and(|l| allow.contains(&l))
                    })
                };
                arcgraph_vector::hnsw::predicate_filtered_search(
                    graph,
                    &query_bytes,
                    opts.k as usize,
                    opts.ef,
                    &L2F32,
                    &is_allowed,
                )
            }
            None if opts
                .deleted_nodes
                .is_some_and(|deleted| !deleted.is_empty()) =>
            {
                let is_allowed = |vid: VectorId| {
                    st.map.get(vid.0 as usize).is_some_and(|(node_id, _)| {
                        !opts
                            .deleted_nodes
                            .is_some_and(|deleted| deleted.contains(node_id))
                            && !st.stale_vectors.contains(&vid)
                    })
                };
                arcgraph_vector::hnsw::predicate_filtered_search(
                    graph,
                    &query_bytes,
                    opts.k as usize,
                    opts.ef,
                    &L2F32,
                    &is_allowed,
                )
            }
            None if !st.stale_vectors.is_empty() => {
                let is_allowed = |vid: VectorId| !st.stale_vectors.contains(&vid);
                arcgraph_vector::hnsw::predicate_filtered_search(
                    graph,
                    &query_bytes,
                    opts.k as usize,
                    opts.ef,
                    &L2F32,
                    &is_allowed,
                )
            }
            // `ef = 0` → the engine uses its `HnswParams::ef_search` default.
            None => graph.search(&query_bytes, opts.k as usize, opts.ef, &L2F32),
        }
        .map_err(|e| SubstrateAccessError::Io(format!("served HNSW search: {e}")))?;
        // `raw` is ascending by L2 (squared) distance (closest first; `L2F32`
        // is the sqeuclidean kernel — distance.rs sqeuclidean). Map to a
        // monotone-decreasing score in (0, 1] so "higher is better" + the
        // closest-first rank order is preserved exactly (RRF / graph.search
        // consume rank order; the absolute score is a documented monotone
        // transform of the L2 (squared) distance, not a calibrated similarity).
        let handle = self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .map_err(|_| SubstrateAccessError::TenantUnknown(tenant))?;
        let crud = handle.crud();
        let tx = self.backend.txn_manager().begin(tenant);
        let mut hits = Vec::with_capacity(raw.len());
        for (vid, dist) in raw {
            let Some((node_id, label)) = st.map.get(vid.0 as usize) else {
                continue;
            };
            let node = match crud::read_node(&tx, *node_id) {
                Ok(Some(rec)) => property_payload::hydrate_node_view(
                    tenant,
                    crud,
                    self.backend.intern_table(),
                    &rec,
                    |label_id| self.resolve_label_name(tenant, label_id),
                )
                .map_err(arcgraph_query::executor::substrate::SubstrateAccessError::from)?,
                Ok(None) => NodeView::new(*node_id, *label),
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "served HNSW hydrate: read_node({}) failed: {e}",
                        node_id.raw()
                    )));
                }
            };
            hits.push(RankedHit {
                node,
                score: 1.0 / (1.0 + f64::from(dist)),
            });
        }
        let _ = tx;
        Ok(hits)
    }
}

impl SubstrateSearchProvider for HnswVectorSearchProvider {
    fn mark_vector_node_deleted(&self, tenant: TenantId, node: NodeId) {
        self.deleted_nodes
            .write()
            .entry(tenant)
            .or_default()
            .insert(node);
        for ((pending_tenant, _), nodes) in self.reindex_nodes.write().iter_mut() {
            if *pending_tenant == tenant {
                nodes.remove(&node);
            }
        }
    }

    fn mark_vector_node_updated(&self, tenant: TenantId, property: &str, node: NodeId) {
        if self
            .deleted_nodes
            .read()
            .get(&tenant)
            .is_some_and(|deleted| deleted.contains(&node))
        {
            return;
        }
        self.reindex_nodes
            .write()
            .entry((tenant, property.to_string()))
            .or_default()
            .insert(node);
    }

    fn mark_bm25_node_deleted(&self, tenant: TenantId, node: NodeId) {
        self.deleted_nodes
            .write()
            .entry(tenant)
            .or_default()
            .insert(node);
        self.bm25_reindex_nodes
            .write()
            .entry(tenant)
            .or_default()
            .remove(&node);
        let Ok(bm25) = self.bm25_service(tenant) else {
            return;
        };
        if let Err(e) = bm25.delete_document(tenant, node, Lsn::MAX) {
            tracing::warn!(
                target: "arcgraph_cli::vector_search",
                tenant = tenant.raw(),
                node = node.raw(),
                "served BM25: failed to buffer delete for tombstoned node: {e}",
            );
            return;
        }
        if let Err(e) = Self::commit_bm25_pending(&bm25, tenant) {
            tracing::warn!(
                target: "arcgraph_cli::vector_search",
                tenant = tenant.raw(),
                node = node.raw(),
                "served BM25: failed to commit delete for tombstoned node: {e}",
            );
        }
    }

    fn mark_bm25_node_updated(&self, tenant: TenantId, node: NodeId) {
        if self
            .deleted_nodes
            .read()
            .get(&tenant)
            .is_some_and(|deleted| deleted.contains(&node))
        {
            return;
        }
        self.bm25_reindex_nodes
            .write()
            .entry(tenant)
            .or_default()
            .insert(node);
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        // No-filter / default-ef path: delegate to the single
        // implementation so the two entry points never diverge
        // (#815 / #816a).
        self.vector_search_filtered(tenant, property, query_vec, k, None, None, read_lsn)
    }

    // 8 args: parallels the trait shape (label_filter + ef_search pushdown);
    // same allow precedent as the trait method + `HnswGraph::search_with_rescore`.
    #[allow(clippy::too_many_arguments)]
    fn vector_search_filtered(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        label_filter: Option<&[LabelId]>,
        ef_search: Option<usize>,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        // _read_lsn is intentionally unused at PART-1: the derived HNSW has no
        // per-read MVCC visibility filter (it reflects the committed snapshot
        // at build time). PART-2's durable VectorPageStore wire adds read_lsn
        // honoring per ADR-041.
        //
        // #816a: `ef_search = None` → `ef = 0`, which tells the engine to use
        // its `HnswParams::ef_search` default (128) — identical to the
        // pre-#816a behavior. `Some(n)` threads the query-time beam width.
        let ef = ef_search.unwrap_or(0);
        let slot = self.slot_for(tenant, property);
        let target_high_water = self.current_high_water(tenant)?;
        let has_reindex = || {
            self.reindex_nodes
                .read()
                .get(&(tenant, property.to_string()))
                .is_some_and(|nodes| !nodes.is_empty())
        };

        // Fast path: the index is already fresh for the current high-water →
        // search under a shared read lock (no scan, no rebuild). This is the
        // warm-query path AND the post-WRITE path once the index is caught up.
        {
            let st = slot.state.read();
            if st.built_high_water == target_high_water && !has_reindex() {
                let deleted = self.deleted_nodes.read();
                return self.search_state(
                    &st,
                    tenant,
                    property,
                    query_vec,
                    SearchOptions {
                        k,
                        ef,
                        label_filter,
                        deleted_nodes: deleted.get(&tenant),
                    },
                );
            }
        }

        // Slow path: bring the index up to date INCREMENTALLY (delta-scan +
        // delta-insert, #787), then search. Held under the write lock so only
        // one thread extends the graph; `bring_up_to_date` re-reads the
        // high-water so a concurrent advance is handled idempotently.
        let mut wst = slot.state.write();
        self.process_reindex_nodes(&mut wst, tenant, property)?;
        self.bring_up_to_date(&mut wst, tenant, property)?;
        let deleted = self.deleted_nodes.read();
        self.search_state(
            &wst,
            tenant,
            property,
            query_vec,
            SearchOptions {
                k,
                ef,
                label_filter,
                deleted_nodes: deleted.get(&tenant),
            },
        )
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        _property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        // `Lsn::MAX` is the query executor's read-latest sentinel, but
        // Tantivy's MVCC visibility filter reserves the absolute u64
        // boundary because it must form `read_lsn + 1`. Normalize the
        // sentinel at the concrete BM25 provider boundary, before both
        // lazy indexing and search, to the same safe read-all value the
        // served `graph.search` path uses. Real snapshot LSNs are
        // forwarded unchanged.
        let read_lsn = if read_lsn == Lsn::MAX {
            Lsn::new(u64::MAX - 1)
        } else {
            read_lsn
        };
        let handle = self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .map_err(|_| SubstrateAccessError::TenantUnknown(tenant))?;
        let bm25 = handle
            .bm25()
            .ok_or_else(|| SubstrateAccessError::IndexUnavailable("bm25".into()))?;
        self.ensure_bm25_up_to_date(tenant, bm25, read_lsn)?;
        let limit = usize::try_from(k).unwrap_or(usize::MAX);
        let rows = bm25
            .search(tenant, query_text, limit, read_lsn)
            .map_err(|e| SubstrateAccessError::Io(format!("bm25 search failed: {e}")))?;

        let crud = handle.crud();
        let tx = self.backend.txn_manager().begin(tenant);
        let mut hits = Vec::with_capacity(rows.len());
        for (node_id, score) in rows {
            let node = match crud::read_node(&tx, node_id) {
                Ok(Some(rec)) => property_payload::hydrate_node_view(
                    tenant,
                    crud,
                    self.backend.intern_table(),
                    &rec,
                    |label_id| self.resolve_label_name(tenant, label_id),
                )
                .map_err(arcgraph_query::executor::substrate::SubstrateAccessError::from)?,
                Ok(None) => NodeView::new(node_id, None),
                Err(e) => {
                    return Err(SubstrateAccessError::Io(format!(
                        "bm25_search hydrate: read_node({}) failed: {e}",
                        node_id.raw()
                    )));
                }
            };
            hits.push(RankedHit {
                node,
                score: f64::from(score),
            });
        }
        let _ = tx;
        Ok(hits)
    }
}

/// Per-`(tenant, property)` SSD index holder (PART-3 ADR-195).
#[derive(Debug)]
struct TenantSsdIndex {
    /// Immutable built SSD index.
    index: SsdDiskAnnIndex,
    /// `VectorId -> NodeId` reverse map.
    map: Vec<NodeId>,
    /// Node high-water mark captured before this immutable index was built.
    built_high_water: u64,
}

/// SSD-resident DiskANN [`SubstrateSearchProvider`] impl (ADR-195 PART-3).
///
/// Serves KNN from SsdDiskAnnIndex with bounded RSS (~14 GB @ 10M×768).
/// Lazy-builds indexes on first search per tenant + property.
/// Deleted nodes are filtered at query time.
#[derive(Debug)]
pub struct SsdVectorSearchProvider {
    backend: StorageBackend,
    cache: RwLock<HashMap<(TenantId, String), Arc<TenantSsdIndex>>>,
    deleted_nodes: RwLock<HashMap<TenantId, HashSet<NodeId>>>,
    index_dir: PathBuf,
    rss_cap_mb: u64,
}

impl SsdVectorSearchProvider {
    /// New SSD provider with RSS cap.
    #[must_use]
    pub fn new(backend: StorageBackend, index_dir: PathBuf, rss_cap_mb: u64) -> Self {
        Self {
            backend,
            cache: RwLock::new(HashMap::new()),
            deleted_nodes: RwLock::new(HashMap::new()),
            index_dir,
            rss_cap_mb,
        }
    }

    /// The tenant's current node high-water mark (the freshness key).
    fn current_high_water(&self, tenant: TenantId) -> Result<u64, SubstrateAccessError> {
        let handle = self
            .backend
            .router()
            .route(tenant, PartitionId::ZERO)
            .map_err(|_| SubstrateAccessError::TenantUnknown(tenant))?;
        Ok(handle.crud().node_high_water(tenant))
    }

    /// Lazy-build or get a cached index fresh at the tenant's node high-water.
    fn slot_for(
        &self,
        tenant: TenantId,
        property: &str,
    ) -> Result<Arc<TenantSsdIndex>, SubstrateAccessError> {
        let key = (tenant, property.to_string());
        let target_high_water = self.current_high_water(tenant)?;
        {
            let cache = self.cache.read();
            if let Some(slot) = cache
                .get(&key)
                .filter(|slot| slot.built_high_water == target_high_water)
            {
                return Ok(Arc::clone(slot));
            }
        }

        let mut cache = self.cache.write();
        // Re-read after serializing with another possible builder. Capturing the
        // high-water before the full scan cannot falsely mark a concurrent ingest
        // as built: a later advance makes the next equality gate rebuild again.
        let target_high_water = self.current_high_water(tenant)?;
        if let Some(slot) = cache
            .get(&key)
            .filter(|slot| slot.built_high_water == target_high_water)
        {
            return Ok(Arc::clone(slot));
        }

        let index = self.build_ssd_index(tenant, property, target_high_water)?;
        let slot = Arc::new(index);
        cache.insert(key, Arc::clone(&slot));
        Ok(slot)
    }

    /// Build index by scanning tenant nodes.
    fn build_ssd_index(
        &self,
        tenant: TenantId,
        property: &str,
        built_high_water: u64,
    ) -> Result<TenantSsdIndex, SubstrateAccessError> {
        // Scan tenant nodes using CrudExecutorSubstrate (matches HNSW pattern).
        let scanner = CrudExecutorSubstrate::new(
            Arc::clone(self.backend.router()),
            Arc::clone(self.backend.txn_manager()),
            Arc::clone(self.backend.intern_table()),
        );

        // Full-tenant scan bounded by the node high-water (NOT `u64::MAX` — that
        // would iterate the entire u64 id space, an effectively infinite loop).
        // `scan_nodes` bounds the scan at `node_high_water` internally, matching
        // the served-HNSW cold-build path. This is a one-time O(N) build cost;
        // the derived index is cached for reuse thereafter.
        let all_nodes = scanner
            .scan_nodes(tenant, None, Lsn::MAX)
            .map_err(|e| SubstrateAccessError::Io(format!("scan_nodes: {e}")))?;

        let mut vectors: Vec<(VectorId, Vec<f32>)> = Vec::new();
        let mut map: Vec<NodeId> = Vec::new();
        let mut dim: Option<usize> = None;

        // Extract vectors from nodes.
        for (slot, bn) in all_nodes.into_iter().enumerate() {
            if let Some(vec) = property_as_vector(&bn.node.properties, property) {
                // Validate dimension.
                if let Some(d) = dim {
                    if vec.len() != d {
                        return Err(SubstrateAccessError::Io("vector dimension mismatch".into()));
                    }
                } else {
                    dim = Some(vec.len());
                }
                vectors.push((VectorId::new(slot as u32), vec));
                map.push(bn.node.id);
            }
        }

        if vectors.is_empty() {
            return Err(SubstrateAccessError::Io(
                "no vectors found in tenant for SSD index build".into(),
            ));
        }

        let dim = dim.expect("dim must be set if vectors non-empty");
        let index_path =
            self.index_dir
                .join(format!("tenant_{}_prop_{}.idx", tenant.raw(), property));

        // Create directory if needed.
        std::fs::create_dir_all(&self.index_dir)
            .map_err(|e| SubstrateAccessError::Io(format!("mkdir: {e}")))?;

        // Train SQ8 on sample.
        let sample_refs: Vec<&[f32]> = vectors
            .iter()
            .step_by((vectors.len() / 100).max(1))
            .map(|(_, v)| v.as_slice())
            .collect();

        let sq8_codebook = Sq8Trainer
            .train(&sample_refs)
            .map_err(|e| SubstrateAccessError::Io(format!("sq8 train: {e}")))?;

        // Arm the RSS guard: the ADR-195 §2.2 detect-and-abort backstop. The
        // build polls it every `BUILD_GUARD_CHECK_EVERY` vectors; a breach latches
        // and surfaces `VectorIndexError::RssCapExceeded` at the next checkpoint
        // (a CLEAN abort, not an OOM-kill). 100 ms sample cadence keeps the macOS
        // subprocess sampler off the build hot path.
        let guard = RssGuard::spawn(self.rss_cap_mb, Duration::from_millis(100));

        // DiskAnnParams: at dim > 256 use the ADR-195 §2.1 dim-scaled curve
        // (R=128 / L_construction=200 — the 128-d defaults are graph-starved at
        // 768d, the V-1 #740 finding); the small defaults suffice below that.
        use arcgraph_vector::diskann::graph::DiskAnnParams;
        let params = if dim > 256 {
            DiskAnnParams {
                r: 128,
                l_construction: 200,
                ..DiskAnnParams::default()
            }
        } else {
            DiskAnnParams::default()
        };

        // Build strategy: the rayon-parallel Vamana refinement
        // (`parallel_build_batch`) is required for the 10M build to be iterable,
        // but adds coordination overhead that is pure loss at small N. Use the
        // deterministic single-threaded build below a batch worth of vectors and
        // the parallel build only once the corpus is genuinely large.
        const PARALLEL_BUILD_THRESHOLD: usize = 8192;
        let parallel_build_batch = if vectors.len() >= PARALLEL_BUILD_THRESHOLD {
            Some(4096)
        } else {
            None
        };

        let index = SsdDiskAnnIndex::build(
            &index_path,
            &SsdBuildConfig {
                dim,
                metric: Metric::L2,
                params,
                pool_frames: 256,
                rerank_factor: 5,
                parallel_build_batch,
            },
            NavQuantizer::Sq8(sq8_codebook),
            vectors,
            &guard,
        )
        .map_err(|e| match e {
            // The RSS ceiling tripped — surface it as a distinct, operator-legible
            // error so the served path aborts cleanly (ADR-195 §2.2) rather than
            // reporting a generic I/O failure. Preserve the observed/cap numbers.
            VectorIndexError::RssCapExceeded {
                observed_mb,
                cap_mb,
            } => SubstrateAccessError::Io(format!(
                "SSD build RSS ceiling exceeded: observed {observed_mb} MB > cap {cap_mb} MB \
                 (ARCGRAPH_VECTOR_RSS_CAP_MB); aborted cleanly per ADR-195 §2.2"
            )),
            other => SubstrateAccessError::Io(format!("ssd build: {other}")),
        })?;

        // Finalize adjacency (post-build pass; improves on-disk durability but
        // optional for serving — the serving path reads adjacency from the in-RAM
        // graph). Non-fatal if it fails.
        let _ = index.finalize_adjacency();

        Ok(TenantSsdIndex {
            index,
            map,
            built_high_water,
        })
    }
}

impl SubstrateSearchProvider for SsdVectorSearchProvider {
    fn mark_vector_node_deleted(&self, tenant: TenantId, node: NodeId) {
        self.deleted_nodes
            .write()
            .entry(tenant)
            .or_default()
            .insert(node);
    }

    fn mark_vector_node_updated(&self, tenant: TenantId, property: &str, _node: NodeId) {
        // An in-place update does not advance the node allocator high-water.
        // Evict the changed property's immutable slot so it is rebuilt from
        // durable node state on the next search.
        self.cache.write().remove(&(tenant, property.to_string()));
    }

    fn mark_bm25_node_deleted(&self, _tenant: TenantId, _node: NodeId) {}
    fn mark_bm25_node_updated(&self, _tenant: TenantId, _node: NodeId) {}

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        self.vector_search_filtered(tenant, property, query_vec, k, None, None, read_lsn)
    }

    #[allow(clippy::too_many_arguments)]
    fn vector_search_filtered(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        _label_filter: Option<&[LabelId]>,
        _ef_search: Option<usize>,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        let slot = self.slot_for(tenant, property)?;

        // Dimension check.
        if query_vec.len() != slot.index.dim() {
            return Err(SubstrateAccessError::Io(format!(
                "query dim {} != index dim {}",
                query_vec.len(),
                slot.index.dim()
            )));
        }

        if k == 0 || slot.index.is_empty() {
            return Ok(Vec::new());
        }

        // 2-phase SSD search (SQ8 nav + f32 rerank).
        let candidates = slot
            .index
            .search(query_vec, k as usize)
            .map_err(|e| SubstrateAccessError::Io(format!("ssd search: {e}")))?;

        let deleted = self.deleted_nodes.read();
        let deleted_set = deleted.get(&tenant);

        let mut hits = Vec::new();
        for (vid, dist) in candidates {
            let slot_idx = vid.0 as usize;
            if slot_idx >= slot.map.len() {
                continue;
            }
            let node_id = slot.map[slot_idx];

            // Skip deleted nodes.
            if deleted_set.is_some_and(|d| d.contains(&node_id)) {
                continue;
            }

            // Build ranked hit (minimal node view).
            let node = NodeView::new(node_id, None);
            let score = 1.0 / (1.0 + f64::from(dist));
            hits.push(RankedHit { node, score });
        }

        Ok(hits)
    }

    fn bm25_search(
        &self,
        _tenant: TenantId,
        _property: &str,
        _query_text: &str,
        _k: u64,
        _read_lsn: Lsn,
    ) -> Result<Vec<RankedHit>, SubstrateAccessError> {
        Err(SubstrateAccessError::IndexUnavailable("bm25".into()))
    }
}
