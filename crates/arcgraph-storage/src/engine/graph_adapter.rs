//! `CrudStore` → [`arcgraph_community::Graph`] adapter (M3.d-3 sub-task A).
//!
//! Materialises a per-tenant CSR-shaped graph from a
//! [`crate::crud::CrudStore`] at a transaction-pinned snapshot,
//! suitable for consumption by `GveLeiden::run` (ADR-040 §D-7).
//!
//! ## Invariants
//!
//! Per the spawn-prompt sub-task A invariants and ADR-041 §D-3b
//! cross-substrate MVCC:
//!
//! - **Snapshot consistency.** Nodes and edges are enumerated under
//!   a SINGLE [`crate::transaction::Transaction`] whose `snapshot()`
//!   pins the visible LSN for both halves. The CRUD-layer
//!   `read_node` / `scan_out` paths honour `tx.snapshot()` so the
//!   resulting Graph is a coherent snapshot.
//! - **Per-tenant isolation.** Every CRUD-layer call is keyed by
//!   `tx.tenant()`; cross-tenant entries cannot leak (per the
//!   I-V2-equivalent invariant in ADR-011 + ADR-040 §D-3).
//! - **Memory budget.** The CSR `Graph` is `8n + 16m` bytes (per
//!   `arcgraph-community/src/graph.rs:40-57`). Concretely at n=1M:
//!   sparse (avg deg 2) ~24 MiB; moderate (avg deg 10) ~88 MiB;
//!   typical KG (avg deg 20) ~168 MiB; dense KG (avg deg 100)
//!   ~808 MiB. v1.0 holds the full Graph in RAM per
//!   [`bootstrap_engine`](super::bootstrap::bootstrap_engine);
//!   operators with multi-tenant deployments should size against
//!   the upper density end. v1.1 may stream-materialise per
//!   `docs/roadmap.md` M5+.
//!
//! ## Construction strategy
//!
//! The Graph type ([`arcgraph_community::Graph`]) requires
//! **dense `0..n` vertex indexing**. CrudStore-allocated `NodeId`s
//! are **1-indexed** (per `CrudStore::alloc_node`'s `prev + 1`
//! convention; `NodeId::ZERO` is reserved as a sentinel). The
//! adapter resolves this by sizing the Graph to
//! `n = node_high_water + 1` so vertex `i` corresponds to
//! `NodeId::new(i)`, with vertex `0` being a phantom orphan that
//! never appears in lookups (because no production CrudStore call
//! ever returns `NodeId::ZERO`). This identity mapping matches
//! [`arcgraph_community::Graph::vertex_to_node_id`]'s contract and
//! the `GveLeiden::install_into` post-mapping.
//!
//! ## Why one adapter, not a streaming iterator
//!
//! Per ADR-040 §D-7 the static GVE-Leiden algorithm requires the
//! full graph snapshot in memory (Sahu §III.A — "static frozen
//! snapshot"). A streaming materialiser would have to buffer the
//! full edge set anyway. Eager materialisation matches the
//! algorithm's contract and is correctness-equivalent.

use std::sync::Arc;

use arcgraph_community::Graph;
use arcgraph_core::{Lsn, NodeId, TenantId};
use thiserror::Error;
use tracing::debug;

use crate::crud::{self, CrudStore};
use crate::transaction::TxnManager;

/// Errors surfaced by [`CrudStoreGraphAdapter`].
///
/// Per ADR-011 §"Core error freeze", these are codec-local; the
/// engine bootstrap layer translates to
/// [`super::bootstrap::EngineError`] at the public boundary.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum GraphAdapterError {
    /// A node enumerated through `node_high_water` could not be
    /// decoded via [`crud::read_node`]. Returned only on a record-
    /// codec failure (corrupted bytes); orphan / tombstoned nodes
    /// are silently skipped per the snapshot-visibility contract.
    #[error("CRUD read_node failed for node_id {node_id} in tenant {tenant_raw}: {source}")]
    ReadNode {
        /// The raw node id whose decode failed.
        node_id: u64,
        /// Raw u64 of the affected tenant.
        tenant_raw: u64,
        /// Underlying CRUD error (`CrudError::Mvcc(InvalidRecordLength {..})` etc.).
        source: crud::CrudError,
    },

    /// A tenant's `node_high_water` exceeds [`u32::MAX`], which
    /// would overflow the CSR vertex index. The v1.0 envelope
    /// (1M nodes per tenant per spawn-prompt sub-task A) leaves
    /// 4 095× headroom; this surfaces only as a defensive guard
    /// against pathological inputs.
    #[error(
        "tenant {tenant_raw} node_high_water {high_water} exceeds u32::MAX (CSR vertex index limit)"
    )]
    NodeIdSpaceOverflow {
        /// Raw u64 of the affected tenant.
        tenant_raw: u64,
        /// Observed high-water mark.
        high_water: u64,
    },
}

/// Stateless materialiser for [`arcgraph_community::Graph`] from a
/// [`CrudStore`] tenant snapshot.
///
/// Holds no mutable state of its own — the adapter is a function
/// object that bundles a `CrudStore` Arc + a `TxnManager` Arc so
/// callers don't have to thread two handles through every call.
/// `materialize` may be called concurrently from multiple threads
/// (it begins a fresh `Transaction` per call).
#[derive(Clone)]
pub struct CrudStoreGraphAdapter {
    crud: Arc<CrudStore>,
    txn_manager: Arc<TxnManager>,
}

impl CrudStoreGraphAdapter {
    /// Construct an adapter over a shared [`CrudStore`] +
    /// [`TxnManager`] pair.
    #[must_use]
    pub fn new(crud: Arc<CrudStore>, txn_manager: Arc<TxnManager>) -> Self {
        Self { crud, txn_manager }
    }

    /// Materialise the per-tenant [`Graph`] at the transaction-
    /// manager's current visible snapshot.
    ///
    /// Returns `(Graph, snapshot_lsn)` where `snapshot_lsn` is the
    /// **actual** snapshot the graph was built from — the snapshot
    /// captured by the inner Transaction at the start of the
    /// materialisation. Callers can thread this LSN into downstream
    /// substrates (e.g., the install LSN on the membership index
    /// per ADR-041 §D-3b) with the guarantee that the LSN matches
    /// the read snapshot.
    ///
    /// ## Issue #239 — outer/inner Tx divergence fix (ADR-040 amendment-05 §D-6)
    ///
    /// Pre-amendment-05, `materialize` opened a redundant outer
    /// `Transaction` to capture the snapshot LSN, then called the
    /// inner materialiser which opened its own Transaction. If a
    /// concurrent commit landed between the two `begin` calls, the
    /// returned `(graph, outer_snapshot)` tuple was a "lying LSN":
    /// the graph reflected the inner txn's later snapshot but the
    /// LSN reported was the outer's earlier one. v1.0-alpha's
    /// FROZEN-GRAPH workaround (PR #235) never tripped this — boot
    /// is single-threaded — but amendment-05's per-tick re-mat does
    /// trip it under continuous ingest.
    ///
    /// This implementation drops the outer Transaction entirely.
    /// The function returns exactly the snapshot used for the
    /// reads, by construction. No second Tx, no anchor question,
    /// no warning path.
    pub fn materialize(&self, tenant: TenantId) -> Result<(Graph, Lsn), GraphAdapterError> {
        // The inner call opens its own Tx, captures the snapshot,
        // builds the graph, returns the (Graph, Lsn) pair. No
        // outer Tx; no lying-LSN class of bug per issue #239.
        self.materialize_with_snapshot(tenant)
    }

    fn materialize_with_snapshot(
        &self,
        tenant: TenantId,
    ) -> Result<(Graph, Lsn), GraphAdapterError> {
        let high_water = self.crud.node_high_water(tenant);
        if high_water > u64::from(u32::MAX) {
            return Err(GraphAdapterError::NodeIdSpaceOverflow {
                tenant_raw: tenant.raw(),
                high_water,
            });
        }

        // Open a single Transaction. Its snapshot is the LSN we
        // return; reads happen at this snapshot; no outer-tx
        // divergence is possible.
        let tx = self.txn_manager.begin(tenant);
        let snapshot_lsn = tx.snapshot();

        // ─── Phase 1: enumerate visible nodes ────────────────────
        //
        // Iterate `1..=high_water` and consult the MVCC store for
        // each id. `read_node` returns `Ok(None)` for tombstoned
        // ids (per its rustdoc). `NodeId::ZERO` is reserved and
        // never allocated by `CrudStore::alloc_node`, so we skip
        // it; vertex 0 in the resulting Graph is a phantom orphan
        // (degree 0, never queried).
        //
        // The CSR vertex space is `0..(high_water + 1)` so vertex
        // `i` corresponds to `NodeId::new(i)` (identity mapping per
        // `Graph::vertex_to_node_id`). This wastes O(1) bytes per
        // tenant (one phantom slot) but eliminates the
        // index-translation pass that an `n` mapping would require
        // and matches `GveLeiden::install_into`'s output convention.
        let n = (high_water + 1) as u32;
        let mut visible: Vec<bool> = vec![false; n as usize];
        // Vertex 0 stays `false` — sentinel. Visible vector seeds
        // the edge-filter step so we don't emit edges that
        // reference tombstoned endpoints.
        for raw in 1..=high_water {
            let nid = NodeId::new(raw);
            // We use `read_node` (the MVCC-only path) rather than
            // `read_node_with_store` (the dual-write path) so the
            // adapter works against any `CrudStore` posture, not
            // just deployments with the dual-write primary index
            // wired. The MVCC chain is the authoritative ground
            // truth (per ADR-023).
            match crud::read_node(&tx, nid) {
                Ok(Some(_)) => {
                    visible[raw as usize] = true;
                }
                Ok(None) => {
                    // Tombstoned at this snapshot. Vertex stays
                    // `false`; no edges referencing it will be
                    // emitted.
                }
                Err(e) => {
                    return Err(GraphAdapterError::ReadNode {
                        node_id: raw,
                        tenant_raw: tenant.raw(),
                        source: e,
                    });
                }
            }
        }

        // ─── Phase 2: enumerate visible undirected edges ─────────
        //
        // For each visible source, scan its outgoing TEL chains.
        // `scan_out` honours the snapshot via two filters:
        //   1. TelBlock::scan applies `created_lsn ≤ snapshot
        //      < expired_lsn` per-entry (ADR-018).
        //   2. The MVCC tombstone probe drops entries whose `rel`
        //      has been deleted at our snapshot (ADR-023).
        //
        // Edges are interpreted as **undirected** at v1.0 per
        // ADR-040 §D-7's GVE-Leiden contract (the algorithm
        // operates on undirected graphs). To avoid double-counting
        // in the half-edge representation, we deduplicate by
        // `(min(u,v), max(u,v))` — each TEL chain emits one
        // half-edge per direction, but the underlying RelRecord
        // exists ONCE per relationship; we keep only the
        // canonical-direction half-edge.
        //
        // Self-loops (u == v) are kept once per `Graph`'s
        // contract.
        let mut canonical_edges: std::collections::BTreeSet<(u32, u32)> =
            std::collections::BTreeSet::new();
        for raw in 1..=high_water {
            if !visible[raw as usize] {
                continue;
            }
            let src = NodeId::new(raw);
            for entry in crud::scan_out(&self.crud, &tx, src, None) {
                // dst lives in the TEL entry; we don't decode the
                // full RelRecord — the tombstone probe inside
                // `scan_out` already filtered visible rels.
                let dst_raw = entry.dst_id;
                if dst_raw > u64::from(u32::MAX) {
                    return Err(GraphAdapterError::NodeIdSpaceOverflow {
                        tenant_raw: tenant.raw(),
                        high_water: dst_raw,
                    });
                }
                let dst = dst_raw as u32;
                if (dst as usize) >= visible.len() || !visible[dst as usize] {
                    // Edge endpoint past the high-water mark or
                    // tombstoned at this snapshot: drop the edge.
                    // (`high_water` was sampled BEFORE phase 1, so
                    // a concurrent ingest could land a new node
                    // with a higher id; we conservatively skip
                    // such "edge from older src to newer dst"
                    // cases, matching `Transaction`'s
                    // snapshot-isolated read semantics.)
                    continue;
                }
                let u = raw as u32;
                let v = dst;
                let canonical = if u <= v { (u, v) } else { (v, u) };
                canonical_edges.insert(canonical);
            }
        }

        // ─── Phase 3: build CSR Graph ────────────────────────────
        //
        // Edge weights are uniformly 1.0 at v1.0 — relationship
        // properties carry no canonical "weight" projection (per
        // ADR-040 §D-7 the v1.0 community detection is unweighted).
        // v1.1 may surface a weight projection through the
        // adapter's construction args.
        let edges: Vec<(u32, u32, f32)> = canonical_edges
            .into_iter()
            .map(|(u, v)| (u, v, 1.0_f32))
            .collect();

        debug!(
            tenant = tenant.raw(),
            snapshot_lsn = snapshot_lsn.raw(),
            n,
            edges = edges.len(),
            "CrudStoreGraphAdapter materialised tenant graph"
        );

        let graph = Graph::from_edges_undirected(n, &edges);
        drop(tx);
        Ok((graph, snapshot_lsn))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arcgraph_core::{LabelId, TypeId};

    use crate::buffer::BufferPool;
    use crate::catalog::SystemCatalog;
    use crate::crud::PropertyData;
    use crate::io::InMemoryPageIo;

    /// Build a CrudStore + TxnManager pair with the catalog
    /// bootstrapped (so `TenantId::DEFAULT` is registered).
    fn make_store() -> (Arc<CrudStore>, Arc<TxnManager>) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(8, io);
        let mgr = TxnManager::new();
        let catalog = SystemCatalog::new();
        catalog.bootstrap(&pool, &mgr).expect("bootstrap");
        (Arc::new(CrudStore::new()), Arc::new(mgr))
    }

    /// Convenience: install `n` nodes + the given edges into the
    /// store under `tenant`, committing one transaction per pass.
    /// Edges are interpreted as `(src_idx, dst_idx)` over the
    /// 1-indexed `NodeId` space the store allocates (so `(0, 1)`
    /// means `NodeId::new(1) -> NodeId::new(2)` after the first
    /// allocation).
    fn install_topology(
        crud: &Arc<CrudStore>,
        mgr: &Arc<TxnManager>,
        tenant: TenantId,
        node_count: u64,
        edges: &[(u64, u64)],
    ) -> Vec<NodeId> {
        let label = LabelId::new(1);
        let ty = TypeId::new(1);
        let mut node_ids: Vec<NodeId> = Vec::with_capacity(node_count as usize);

        let mut tx = mgr.begin(tenant);
        for _ in 0..node_count {
            let nid = crud::create_node(crud, &mut tx, tenant, label, &PropertyData::Empty)
                .expect("create_node");
            node_ids.push(nid);
        }
        for &(u, v) in edges {
            let src = node_ids[u as usize];
            let dst = node_ids[v as usize];
            crud::create_rel(crud, &mut tx, tenant, src, dst, ty, &PropertyData::Empty)
                .expect("create_rel");
        }
        crud::commit(tx, crud).expect("commit");
        node_ids
    }

    #[test]
    fn materialize_empty_tenant_yields_phantom_zero_only() {
        let (crud, mgr) = make_store();
        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let (graph, _lsn) = adapter
            .materialize(TenantId::DEFAULT)
            .expect("materialize empty");
        // No allocations -> high_water = 0 -> n = 1 (just the
        // phantom vertex 0).
        assert_eq!(graph.n(), 1);
        assert_eq!(graph.neighbors(0).count(), 0);
    }

    #[test]
    fn materialize_two_disconnected_pairs() {
        let (crud, mgr) = make_store();
        // 4 nodes, 2 disconnected edges. CrudStore allocates
        // NodeId 1..=4; vertex indices are NodeId.raw() so vertex
        // 1↔2 and 3↔4.
        install_topology(&crud, &mgr, TenantId::DEFAULT, 4, &[(0, 1), (2, 3)]);

        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let (graph, _lsn) = adapter.materialize(TenantId::DEFAULT).expect("materialize");
        assert_eq!(graph.n(), 5, "n = high_water + 1 = 4 + 1");
        // Vertex 0 is the phantom orphan.
        assert_eq!(graph.neighbors(0).count(), 0);
        // Vertices 1↔2 connected.
        let n1: Vec<u32> = graph.neighbors(1).map(|(v, _)| v).collect();
        assert_eq!(n1, vec![2]);
        let n2: Vec<u32> = graph.neighbors(2).map(|(v, _)| v).collect();
        assert_eq!(n2, vec![1]);
        // Vertices 3↔4 connected.
        let n3: Vec<u32> = graph.neighbors(3).map(|(v, _)| v).collect();
        assert_eq!(n3, vec![4]);
        let n4: Vec<u32> = graph.neighbors(4).map(|(v, _)| v).collect();
        assert_eq!(n4, vec![3]);
        // Total weight = 2m = 2 * 2 edges * 1.0 = 4.
        assert!((graph.total_weight_2m() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn materialize_per_tenant_isolation() {
        let (crud, mgr) = make_store();
        // Tenant DEFAULT: 3 nodes in a triangle (0↔1, 1↔2, 0↔2).
        install_topology(&crud, &mgr, TenantId::DEFAULT, 3, &[(0, 1), (1, 2), (0, 2)]);
        // Tenant SYSTEM: 2 nodes, 1 edge.
        install_topology(&crud, &mgr, TenantId::SYSTEM, 2, &[(0, 1)]);

        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let (g_default, _) = adapter
            .materialize(TenantId::DEFAULT)
            .expect("materialize DEFAULT");
        let (g_system, _) = adapter
            .materialize(TenantId::SYSTEM)
            .expect("materialize SYSTEM");
        // DEFAULT: 3 nodes -> n = 4; 3 edges (triangle) -> 2m = 6.
        assert_eq!(g_default.n(), 4);
        assert!((g_default.total_weight_2m() - 6.0).abs() < 1e-6);
        // SYSTEM: 2 nodes -> n = 3; 1 edge -> 2m = 2.
        assert_eq!(g_system.n(), 3);
        assert!((g_system.total_weight_2m() - 2.0).abs() < 1e-6);
        // Each tenant's graph has only its own vertices populated;
        // vertex 0 in both is the phantom orphan.
        assert_eq!(g_default.neighbors(0).count(), 0);
        assert_eq!(g_system.neighbors(0).count(), 0);
    }

    #[test]
    fn materialize_dedupes_undirected_edges() {
        let (crud, mgr) = make_store();
        // 3 nodes; install (0->1), (1->0), (0->2). The TEL chain
        // for src=0 holds two outgoing entries (channels are by
        // type, both rel-typed at TypeId(1)). Each rel produces
        // ONE half-edge in the TEL. When materialised as
        // undirected, (0,1) and (1,0) both emit a single canonical
        // (1,2) edge — dedupe must drop the duplicate.
        install_topology(&crud, &mgr, TenantId::DEFAULT, 3, &[(0, 1), (1, 0), (0, 2)]);
        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let (graph, _) = adapter.materialize(TenantId::DEFAULT).expect("materialize");
        // 3 visible nodes -> n = 4; 2 unique undirected edges
        // (1↔2, 1↔3) -> 2m = 4 (each edge contributes weight 1
        // twice, once per direction).
        assert_eq!(graph.n(), 4);
        assert!(
            (graph.total_weight_2m() - 4.0).abs() < 1e-6,
            "duplicate (0->1)/(1->0) must be deduped to a single \
             undirected edge; expected total_weight_2m = 4.0, got {}",
            graph.total_weight_2m()
        );
    }

    #[test]
    fn materialize_emits_self_loop() {
        let (crud, mgr) = make_store();
        // 1 node, self-loop at NodeId(1).
        install_topology(&crud, &mgr, TenantId::DEFAULT, 1, &[(0, 0)]);
        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let (graph, _) = adapter.materialize(TenantId::DEFAULT).expect("materialize");
        assert_eq!(graph.n(), 2);
        // Vertex 1 has a self-loop; degree contains the self-edge
        // weight once per Graph::from_edges_undirected's
        // self-loop convention (weight 1.0 added once on the self-
        // loop pass).
        let n1: Vec<u32> = graph.neighbors(1).map(|(v, _)| v).collect();
        assert_eq!(n1, vec![1], "self-loop yields one neighbour entry");
    }

    #[test]
    fn materialize_skips_tombstoned_node() {
        let (crud, mgr) = make_store();
        let label = LabelId::new(1);
        let ty = TypeId::new(1);
        // Install 3 nodes + 2 edges in tx1; commit.
        let mut tx1 = mgr.begin(TenantId::DEFAULT);
        let n1 = crud::create_node(
            &crud,
            &mut tx1,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("n1");
        let n2 = crud::create_node(
            &crud,
            &mut tx1,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("n2");
        let n3 = crud::create_node(
            &crud,
            &mut tx1,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("n3");
        let _r12 = crud::create_rel(
            &crud,
            &mut tx1,
            TenantId::DEFAULT,
            n1,
            n2,
            ty,
            &PropertyData::Empty,
        )
        .expect("r12");
        let _r23 = crud::create_rel(
            &crud,
            &mut tx1,
            TenantId::DEFAULT,
            n2,
            n3,
            ty,
            &PropertyData::Empty,
        )
        .expect("r23");
        crud::commit(tx1, &crud).expect("commit tx1");

        // tx2: delete node 2; commit.
        let mut tx2 = mgr.begin(TenantId::DEFAULT);
        crud::delete_node(&mut tx2, n2).expect("delete n2");
        crud::commit(tx2, &crud).expect("commit tx2");

        let adapter = CrudStoreGraphAdapter::new(crud, mgr);
        let (graph, _) = adapter
            .materialize(TenantId::DEFAULT)
            .expect("materialize post-delete");
        // n1 and n3 are visible; n2 is tombstoned.
        // n = high_water + 1 = 3 + 1 = 4 (high_water didn't drop
        // even though n2 was deleted).
        assert_eq!(graph.n(), 4);
        // Edges that touch n2 are dropped (both r12 and r23
        // referenced n2 as one endpoint).
        let n1_neighbours: Vec<u32> = graph.neighbors(n1.raw() as u32).map(|(v, _)| v).collect();
        let n3_neighbours: Vec<u32> = graph.neighbors(n3.raw() as u32).map(|(v, _)| v).collect();
        assert!(
            n1_neighbours.is_empty(),
            "n1 should have no neighbours after n2 tombstone, got {n1_neighbours:?}"
        );
        assert!(
            n3_neighbours.is_empty(),
            "n3 should have no neighbours after n2 tombstone, got {n3_neighbours:?}"
        );
    }
}
