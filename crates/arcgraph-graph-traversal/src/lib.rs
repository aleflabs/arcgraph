//! Canonical bounded graph traversal for ArcGraph.
//!
//! Per ADR-205 (V11-S-04; ratifies `docs/roadmap.md` v1.1 Scale primitives
//! row + ADR-036 §D-4 + K2 B-6 and absorbs PRIM-1's hand-rolled BFS per
//! ADR-048-prim1-v1-alpha-narrowings), this crate owns the *pure-algorithmic*
//! traversal surface that the query layer (and, through it, the MCP catalog)
//! consumes.
//!
//! ## Public surface
//!
//! - [`GraphTraversalHandle::k_hop`] — bounded k-hop expansion with cost
//!   budgets (the PRIM-1 H-1 mid-loop trip semantics), LIMIT pushdown, the
//!   ADR-025 §5 supernode-firewall frontier-cap hook, and
//!   [`SamplingStrategy::ReservoirVitterL`] degree-aware per-hop reservoir
//!   sampling (J6 §5 line 234). See [`khop`] module docs.
//! - [`GraphTraversalHandle::bidirectional_shortest`] — meet-in-the-middle
//!   unweighted shortest path. See [`shortest`] module docs.
//! - [`GraphTraversalHandle::k_shortest_paths`] — Yen 1971 k shortest
//!   loopless paths (hop-count metric at v1.1). See [`yen`] module docs.
//! - [`merge_top_k`] — deterministic local multi-source result merge.
//! - [`EdgeSource`] — the adapter trait consumers implement
//!   (ADR-179 `EgonetSource` idiom: source trait in the leaf crate,
//!   adapters in consumers). [`MemoryEdgeSource`] is the in-memory
//!   reference adapter.
//!
//! ## What this crate is NOT
//!
//! - **Not a storage layer.** No TEL, no LSN, no tenancy — the adapter
//!   closes over `(tenant, snapshot_lsn)` (bounded contexts, bounded-context policy).
//! - **Not a streaming engine.** Neighbor batches are materialized `Vec`s
//!   because that is the substrate's real shape today; the streaming lift
//!   is V11-S-03 and lands inside adapters, not in this API.
//! - **Not a weighted-shortest-path library.** Hop-count metric only at
//!   v1.1 (ADR-205 OQ-1).
//! - **No `unsafe`.** Crate root carries `#![deny(unsafe_code)]`.
//! - **No I/O, no async runtime** — callable from either substrate.

#![deny(unsafe_code)]
#![warn(missing_docs)]
#![recursion_limit = "256"]

pub mod error;
pub mod khop;
pub mod memory;
pub mod merge;
pub mod rng;
pub mod shortest;
pub mod source;
pub mod yen;

pub use error::TraversalError;
pub use khop::{KHopEdge, KHopNode, KHopRequest, KHopResult, SamplingStrategy, Truncation};
pub use memory::MemoryEdgeSource;
pub use merge::merge_top_k;
pub use shortest::PathResult;
pub use source::{EdgeFilter, EdgeSource, Neighbor, TraversalDirection};

use arcgraph_core::NodeId;

/// The handle named by `docs/roadmap.md` V11-S-04 + ADR-036 §D-4.
///
/// A thin wrapper binding an [`EdgeSource`] adapter to the traversal
/// algorithms. ADR-036's 2025 sketch held storage handles directly; per
/// ADR-205 §D-1 the handle instead holds the adapter, which owns tenancy +
/// snapshot visibility (the seam that keeps this crate a leaf).
#[derive(Debug)]
pub struct GraphTraversalHandle<S> {
    source: S,
}

impl<S: EdgeSource> GraphTraversalHandle<S> {
    /// Bind a traversal handle to an [`EdgeSource`] adapter.
    pub fn new(source: S) -> Self {
        Self { source }
    }

    /// Borrow the underlying source adapter.
    pub fn source(&self) -> &S {
        &self.source
    }

    /// Consume the handle, returning the adapter.
    pub fn into_source(self) -> S {
        self.source
    }

    /// Bounded k-hop expansion from `anchor`. See [`khop::k_hop`].
    ///
    /// `anchor_data` / `anchor_cost` seed the result + cost accounting with
    /// the caller-fetched anchor payload (the PRIM-1 H-3 discipline: the
    /// anchor's real data is passed in, never synthesized here).
    pub fn k_hop(
        &self,
        anchor: NodeId,
        anchor_data: S::NodeData,
        anchor_cost: u64,
        req: &KHopRequest,
    ) -> khop::KHopOutcome<S> {
        khop::k_hop(&self.source, anchor, anchor_data, anchor_cost, req)
    }

    /// Unweighted shortest path via meet-in-the-middle bidirectional BFS.
    /// See [`shortest::bidirectional_shortest`].
    pub fn bidirectional_shortest(
        &self,
        src: NodeId,
        dst: NodeId,
        max_hops: u32,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> Result<Option<PathResult>, TraversalError<S::Error>> {
        shortest::bidirectional_shortest(&self.source, src, dst, max_hops, direction, filter)
    }

    /// Yen's k shortest loopless paths (hop-count metric). See
    /// [`yen::k_shortest_paths`].
    #[allow(clippy::too_many_arguments)] // &self + the 6-arg roadmap-bound vocabulary
    // (src, dst, k, max_hops, direction, filter); see yen.rs for the request-struct
    // trade-off note.
    pub fn k_shortest_paths(
        &self,
        src: NodeId,
        dst: NodeId,
        k: usize,
        max_hops: u32,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> Result<Vec<PathResult>, TraversalError<S::Error>> {
        yen::k_shortest_paths(&self.source, src, dst, k, max_hops, direction, filter)
    }

    /// Filtered degree of `node` (needed for reservoir weighting per the
    /// roadmap row). Default trait impl is `O(deg)` (one neighbor batch);
    /// adapters override with an O(1) stat when one exists (the V11-S-02
    /// per-`(tenant, src)` channel index is the natural provider —
    /// ADR-205 OQ-2).
    pub fn degree(
        &self,
        node: NodeId,
        direction: TraversalDirection,
        filter: &EdgeFilter,
    ) -> Result<u64, TraversalError<S::Error>> {
        self.source
            .degree(node, direction, filter)
            .map_err(TraversalError::Source)
    }
}
