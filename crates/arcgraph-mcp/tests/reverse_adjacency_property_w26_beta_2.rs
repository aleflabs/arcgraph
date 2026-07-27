//! W26-β-2 / ADR-131 — reverse-adjacency property tests.
//!
//! Closes #350 v1.1 inbound TEL expand acceptance criteria:
//!
//! - **AC-1** [`expand_right_to_left_returns_reverse_view_of_outbound`]:
//!   For every random graph, the multiset of edges yielded by
//!   `Direction::RightToLeft` from `n` equals the multiset of edges
//!   yielded by `Direction::LeftToRight` from every `m` whose outbound
//!   walk reaches `n`. Concretely: the set of `(src, dst, rel_id)`
//!   triples observed on the reverse walk from `n` is exactly the set
//!   of `(src, dst, rel_id)` triples observed on the forward walk that
//!   produced edges with `dst == n`.
//!
//! - **AC-2** [`expand_undirected_is_out_plus_in_with_rel_id_dedup`]:
//!   For every random graph, `Direction::Undirected` from `n` yields
//!   the multiset union of `LeftToRight from n` + `RightToLeft from n`
//!   deduplicated by `RelId` (the self-loop guard — an edge n→n
//!   appears in BOTH the forward chain at `(n, ty)` and the reverse
//!   chain at `(n, ty)` and MUST be counted exactly once).
//!
//! Per W26-β-2 spawn-prompt + `feedback_load_bearing_pr_requires_fault_injection_tests.md`
//! discipline (load-bearing PRs require property + fault-injection
//! coverage for every new failure mode).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId, TypeId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_query::logical_plan::Direction;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, create_rel};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use proptest::collection::vec as pvec;
use proptest::prelude::*;

/// A single random edge declaration: src/dst are 0-based node
/// indices into the materialized node vector, type is in [0, NUM_TYPES).
#[derive(Debug, Clone, Copy)]
struct EdgeDecl {
    src_idx: usize,
    dst_idx: usize,
    type_idx: u32,
}

/// Number of relationship types used by the property test. Keeping
/// this small (3) ensures the type-filter path gets hit on a meaningful
/// fraction of the generated graphs.
const NUM_TYPES: u32 = 3;

/// Build a substrate fixture identical in shape to
/// `crud::tests::fixture` (the substrate's own unit-test harness)
/// but accessible from the integration-test crate.
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

/// Materialize a random graph of `n_nodes` nodes + `edges`
/// declarations into the substrate. Returns the `Vec<NodeId>` so
/// callers can map node indices back to NodeId. Self-loops + parallel
/// edges are honored (no dedup at the storage layer).
///
/// The substrate-side `scan_out` / `scan_in` walks are the canonical
/// rel-index lookup for the oracle assertions — there is no need to
/// return a parallel `BTreeMap<rel_id, (src, dst, ty)>` from this
/// helper (the property-test oracles re-derive that map from the
/// substrate's `expand` output).
fn build_graph(
    crud: &Arc<CrudStore>,
    mgr: &Arc<TxnManager>,
    n_nodes: usize,
    edges: &[EdgeDecl],
) -> Vec<NodeId> {
    let label = LabelId::new(1);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let mut nodes: Vec<NodeId> = Vec::with_capacity(n_nodes);
    for _ in 0..n_nodes {
        let nid = create_node(
            crud,
            &mut tx,
            TenantId::DEFAULT,
            label,
            &PropertyData::Empty,
        )
        .expect("create_node");
        nodes.push(nid);
    }
    for e in edges {
        let src = nodes[e.src_idx];
        let dst = nodes[e.dst_idx];
        let ty = TypeId::new(e.type_idx);
        create_rel(
            crud,
            &mut tx,
            TenantId::DEFAULT,
            src,
            dst,
            ty,
            &PropertyData::Empty,
        )
        .expect("create_rel");
    }
    commit(tx, crud).expect("commit");
    nodes
}

/// Reduce the substrate's `Vec<BoundEdge>` to a comparable
/// `(src, dst, rel_id, ty)` set for oracle assertions. The `ty`
/// field is unwrapped via `rel_type` (substrate populates this
/// from the `expand` argument, not from the underlying record).
fn edge_keyset(
    edges: &[arcgraph_query::executor::substrate::BoundEdge],
) -> BTreeSet<(u64, u64, u64, Option<u32>)> {
    edges
        .iter()
        .map(|e| {
            (
                e.rel.from.raw(),
                e.rel.to.raw(),
                e.rel.id.raw(),
                e.rel.rel_type.map(|t| t.raw()),
            )
        })
        .collect()
}

/// Strategy: random graph with up to 8 nodes + up to 16 edges,
/// drawn from 3 relationship types. Bounds chosen so the property
/// test can run ~256 cases per test in under a few seconds per the
/// canonical proptest default budget. The graph topology covers
/// self-loops (src==dst), multi-edges (same src/dst/ty appear
/// multiple times), and type-filtered queries.
fn random_graph_strategy() -> impl Strategy<Value = (usize, Vec<EdgeDecl>)> {
    (2usize..=8usize).prop_flat_map(|n_nodes| {
        pvec(
            (0..n_nodes, 0..n_nodes, (0..NUM_TYPES).prop_map(|t| t + 1)).prop_map(|(s, d, t)| {
                EdgeDecl {
                    src_idx: s,
                    dst_idx: d,
                    type_idx: t,
                }
            }),
            0..=16,
        )
        .prop_map(move |edges| (n_nodes, edges))
    })
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        max_shrink_iters: 256,
        .. ProptestConfig::default()
    })]

    /// AC-1: For every random graph, the reverse walk from `n` yields
    /// the same edge set as the forward walk's reverse view (the set
    /// of edges with `dst == n`).
    ///
    /// The oracle: for each node `n`, compute the forward-side
    /// inbound set by walking EVERY node's outbound chain and
    /// collecting edges with `dst == n`. Compare this to the
    /// substrate's `RightToLeft` walk from `n`.
    #[test]
    fn expand_right_to_left_returns_reverse_view_of_outbound(
        (n_nodes, edges) in random_graph_strategy()
    ) {
        let (sub, crud, mgr, _router) = fixture();
        let nodes = build_graph(&crud, &mgr, n_nodes, &edges);

        for (n_idx, &n) in nodes.iter().enumerate() {
            // Build the "forward-side inbound" oracle: for each
            // candidate source m, walk its outbound and collect
            // edges with dst == n.
            let mut oracle: BTreeSet<(u64, u64, u64, Option<u32>)> = BTreeSet::new();
            for &m in &nodes {
                let outs = sub
                    .expand(
                        TenantId::DEFAULT,
                        m,
                        None, // all types
                        Direction::LeftToRight,
                        Lsn::MAX,
                    )
                    .expect("LeftToRight expand from m");
                for e in &outs {
                    if e.rel.to == n {
                        oracle.insert((
                            e.rel.from.raw(),
                            e.rel.to.raw(),
                            e.rel.id.raw(),
                            e.rel.rel_type.map(|t| t.raw()),
                        ));
                    }
                }
            }

            // Substrate-side reverse walk.
            let inbound = sub
                .expand(
                    TenantId::DEFAULT,
                    n,
                    None,
                    Direction::RightToLeft,
                    Lsn::MAX,
                )
                .expect("RightToLeft expand from n");
            let observed = edge_keyset(&inbound);

            prop_assert_eq!(
                observed.clone(),
                oracle.clone(),
                "AC-1 failed at node index {}: reverse walk must equal forward-side inbound oracle. \
                 observed={:?} oracle={:?}",
                n_idx, observed, oracle
            );
        }
    }

    /// AC-2: For every random graph, `Direction::Undirected` from `n`
    /// yields the union of outbound + inbound walks deduplicated by
    /// `RelId`. The dedup is load-bearing for self-loops (which
    /// appear in both forward + reverse chains).
    #[test]
    fn expand_undirected_is_out_plus_in_with_rel_id_dedup(
        (n_nodes, edges) in random_graph_strategy()
    ) {
        let (sub, crud, mgr, _router) = fixture();
        let nodes = build_graph(&crud, &mgr, n_nodes, &edges);

        for (n_idx, &n) in nodes.iter().enumerate() {
            // Outbound + inbound walks (the substrate's own surfaces).
            let outs = sub
                .expand(
                    TenantId::DEFAULT,
                    n,
                    None,
                    Direction::LeftToRight,
                    Lsn::MAX,
                )
                .expect("LeftToRight expand");
            let ins = sub
                .expand(
                    TenantId::DEFAULT,
                    n,
                    None,
                    Direction::RightToLeft,
                    Lsn::MAX,
                )
                .expect("RightToLeft expand");

            // Oracle: dedup by RelId across out + in. The
            // substrate's Undirected walk has the same semantics.
            let mut by_rel_id: BTreeMap<u64, (u64, u64, Option<u32>)> = BTreeMap::new();
            for e in outs.iter().chain(ins.iter()) {
                // First occurrence wins (matches dedup_by_key behavior).
                by_rel_id
                    .entry(e.rel.id.raw())
                    .or_insert((
                        e.rel.from.raw(),
                        e.rel.to.raw(),
                        e.rel.rel_type.map(|t| t.raw()),
                    ));
            }
            let oracle: BTreeSet<(u64, u64, u64, Option<u32>)> = by_rel_id
                .into_iter()
                .map(|(rel_id, (src, dst, ty))| (src, dst, rel_id, ty))
                .collect();

            // Substrate-side Undirected walk.
            let und = sub
                .expand(
                    TenantId::DEFAULT,
                    n,
                    None,
                    Direction::Undirected,
                    Lsn::MAX,
                )
                .expect("Undirected expand");
            let observed = edge_keyset(&und);

            prop_assert_eq!(
                observed.clone(),
                oracle.clone(),
                "AC-2 failed at node index {}: Undirected walk must equal (out ∪ in) deduped by RelId. \
                 observed={:?} oracle={:?}",
                n_idx, observed, oracle
            );

            // Cardinality cross-check: |Undirected| ≤ |Out| + |In|
            // (equality iff no self-loops in either set).
            prop_assert!(
                und.len() <= outs.len() + ins.len(),
                "AC-2: |Undirected| MUST NOT exceed |Out| + |In|; got {} > {} + {} = {}",
                und.len(), outs.len(), ins.len(), outs.len() + ins.len()
            );
        }
    }

    /// AC-2 supplementary: type-filter symmetry. For every random
    /// graph + every type, the reverse walk filtered by `Some(ty)`
    /// equals the type-filtered forward-side inbound oracle.
    #[test]
    fn expand_right_to_left_type_filter_symmetric_with_outbound(
        (n_nodes, edges) in random_graph_strategy()
    ) {
        let (sub, crud, mgr, _router) = fixture();
        let nodes = build_graph(&crud, &mgr, n_nodes, &edges);

        for &n in &nodes {
            for ty_raw in 1..=NUM_TYPES {
                let ty = TypeId::new(ty_raw);
                let mut oracle: BTreeSet<(u64, u64, u64, Option<u32>)> = BTreeSet::new();
                for &m in &nodes {
                    let outs = sub
                        .expand(
                            TenantId::DEFAULT,
                            m,
                            Some(ty),
                            Direction::LeftToRight,
                            Lsn::MAX,
                        )
                        .expect("LeftToRight expand w/ type filter");
                    for e in &outs {
                        if e.rel.to == n {
                            oracle.insert((
                                e.rel.from.raw(),
                                e.rel.to.raw(),
                                e.rel.id.raw(),
                                e.rel.rel_type.map(|t| t.raw()),
                            ));
                        }
                    }
                }
                let inbound = sub
                    .expand(
                        TenantId::DEFAULT,
                        n,
                        Some(ty),
                        Direction::RightToLeft,
                        Lsn::MAX,
                    )
                    .expect("RightToLeft expand w/ type filter");
                let observed = edge_keyset(&inbound);
                prop_assert_eq!(
                    observed.clone(),
                    oracle.clone(),
                    "AC-2 (type-filter symmetry) failed: observed={:?} oracle={:?}",
                    observed, oracle
                );
            }
        }
    }
}
