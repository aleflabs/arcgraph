//! The [`EdgeSource`] adapter seam (ADR-205 §D-1).
//!
//! Consumers adapt their substrate to this trait; the crate never sees
//! `ExecutorSubstrate`, TEL, tenancy, or LSNs. The source trait lives in
//! this leaf crate and adapters live in consumers; [`crate::MemoryEdgeSource`]
//! is the in-memory reference.

use arcgraph_core::{NodeId, RelId, TypeId};

/// Direction of edge traversal relative to the visited node.
///
/// Maps onto the substrate's `Direction` at the adapter
/// (`Outbound`→`LeftToRight`, `Inbound`→`RightToLeft` — real rows since
/// ADR-131 closed issue #350 — `Undirected`→`Undirected`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraversalDirection {
    /// Follow edges from their source to their destination.
    Outbound,
    /// Follow edges from their destination back to their source.
    Inbound,
    /// Follow edges in both orientations.
    Undirected,
}

impl TraversalDirection {
    /// The opposite orientation (`Undirected` is its own inverse). Used by
    /// the backward frontier in [`crate::shortest::bidirectional_shortest`].
    #[must_use]
    pub fn inverse(self) -> Self {
        match self {
            Self::Outbound => Self::Inbound,
            Self::Inbound => Self::Outbound,
            Self::Undirected => Self::Undirected,
        }
    }
}

/// Edge predicate pushed down into the source.
///
/// v1.1 carries the rel-type axis only (mirroring
/// `ExecutorSubstrate::expand`'s `rel_type: Option<TypeId>`); future axes
/// (property predicates, label sets) are additive `Option` fields with
/// `Default` semantics so existing adapters keep compiling.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EdgeFilter {
    /// Match only relationships of this type; `None` matches every type.
    pub rel_type: Option<TypeId>,
}

impl EdgeFilter {
    /// Filter matching every edge.
    #[must_use]
    pub fn any() -> Self {
        Self::default()
    }

    /// Filter matching one relationship type.
    #[must_use]
    pub fn rel_type(rel_type: TypeId) -> Self {
        Self {
            rel_type: Some(rel_type),
        }
    }
}

/// One adjacent `(edge, destination)` pair emitted by an [`EdgeSource`].
///
/// `edge_cost` / `node_cost` are caller-defined cost units consumed by the
/// k-hop budget accounting (the PRIM-1 adapter supplies its existing
/// `estimate_edge_bytes` / `estimate_node_bytes`); an adapter that charges
/// `0` opts out of cost budgets and relies on node/limit caps alone
/// (documented honesty per ADR-205 §Consequences).
#[derive(Debug, Clone)]
pub struct Neighbor<N, E> {
    /// Identity of the traversed relationship (dedupe axis: an edge is
    /// charged + recorded at most once per traversal, the
    /// `collect_reachable` `edge_ids` discipline).
    pub rel_id: RelId,
    /// The node reached over `rel_id`.
    pub dst: NodeId,
    /// Caller payload for `dst` (e.g. `NodeView`), carried through
    /// traversal so consumers never need a second fetch pass.
    pub dst_data: N,
    /// Caller payload for the relationship (e.g. `RelView`).
    pub edge_data: E,
    /// Cost charged when this edge is first recorded.
    pub edge_cost: u64,
    /// Cost charged when `dst` is first retained.
    pub node_cost: u64,
}

/// One materialized neighbor batch (clippy type-complexity sugar shared
/// by the trait and its adapters).
pub type NeighborBatch<N, E, Err> = Result<Vec<Neighbor<N, E>>, Err>;

/// The substrate adapter trait (ADR-205 §D-1).
///
/// `neighbors` returns a **materialized batch** because that is the
/// substrate's real shape today (`ExecutorSubstrate::expand` returns
/// `Vec<BoundEdge>`; the streaming lift is V11-S-03 and lands inside
/// adapters without changing this trait). Implementations MUST be
/// deterministic in batch order for a fixed underlying snapshot —
/// traversal determinism (BFS order, seeded reservoir) is built on it.
pub trait EdgeSource {
    /// Per-node payload carried through traversal (use `()` for
    /// topology-only callers).
    type NodeData: Clone;
    /// Per-edge payload carried through traversal (use `()` for
    /// topology-only callers).
    type EdgeData: Clone;
    /// Adapter error, surfaced losslessly as
    /// [`crate::TraversalError::Source`].
    type Error: core::error::Error + Send + Sync + 'static;

    /// All `(edge, destination)` pairs adjacent to `node` in `direction`,
    /// post-`filter`.
    fn neighbors(
        &self,
        node: NodeId,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> NeighborBatch<Self::NodeData, Self::EdgeData, Self::Error>;

    /// Filtered degree of `node` in `direction`.
    ///
    /// Default is `O(deg)` (materializes one neighbor batch) — honest but
    /// not free; adapters override with an O(1) stat when one exists
    /// (ADR-205 OQ-2 forward-pins the V11-S-02 per-`(tenant, src)` index
    /// as the provider).
    fn degree(
        &self,
        node: NodeId,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> Result<u64, Self::Error> {
        Ok(self.neighbors(node, direction, filter)?.len() as u64)
    }
}

impl<S: EdgeSource + ?Sized> EdgeSource for &S {
    type NodeData = S::NodeData;
    type EdgeData = S::EdgeData;
    type Error = S::Error;

    fn neighbors(
        &self,
        node: NodeId,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> NeighborBatch<Self::NodeData, Self::EdgeData, Self::Error> {
        (**self).neighbors(node, direction, filter)
    }

    fn degree(
        &self,
        node: NodeId,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> Result<u64, Self::Error> {
        (**self).degree(node, direction, filter)
    }
}
