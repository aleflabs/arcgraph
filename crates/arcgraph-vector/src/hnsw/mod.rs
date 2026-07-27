//! HNSW index — Slice C in-memory baseline.
//!
//! See ADR-035 §5.2 (build path) and ADR-003 (HNSW with layered
//! deletion strategies).
//!
//! ## What this slice ships
//!
//! `HnswGraph` — a single-tenant, single-partition, in-memory
//! Hierarchical Navigable Small World index implementing
//! Malkov & Yashunin TPAMI 2018 §3.1 / §4 (Algorithms 1–4):
//!
//! - [`HnswGraph::new(M, ef_construction)`](HnswGraph::new) —
//!   stochastic-layered graph with the standard parameter set.
//!   Per ADR-003 the workload-validated defaults are `M=32`,
//!   `ef_construction=200`, `ef_search=128` (ANN-Benchmarks +
//!   Codastra "Vector DB Showdown" 2025); the constructor leaves
//!   the choice to the caller so test-suite-friendly smaller
//!   parameters are expressible.
//! - [`HnswGraph::insert`](HnswGraph::insert) — incremental
//!   per-vector insert per Algorithm 1, with the
//!   `select_neighbors_heuristic` pruning of Algorithm 4
//!   (`extendCandidates=false`, `keepPrunedConnections=true` —
//!   the paper's robust default and what hnswlib + qdrant
//!   ship at v1.0).
//! - [`HnswGraph::search`](HnswGraph::search) — top-`k` beam
//!   search per Algorithm 5 (search-layer + greedy zoom).
//! - [`HnswGraph::mark_deleted`](HnswGraph::mark_deleted) —
//!   tombstone-bitmap deletion per ADR-003 Strategy 1.
//!   Tombstoned vectors remain navigation hubs so the graph
//!   does not lose connectivity until a rebuild fires.
//! - [`HnswGraph::detect_unreachable`] + [`HnswGraph::mn_ru_repair`]
//!   — MN-RU connectivity repair per ADR-003 Strategy 2 (Xiao
//!   et al. arxiv 2407.07871 Algorithm 1). Run after a
//!   delete-heavy phase to re-attach unreachable points before
//!   recall@10 degrades.
//!
//! ## What this slice does NOT ship
//!
//! - **Filter-aware traversal** — Slice F.2 owns
//!   `hnsw/filtered.rs`; the Phase-2 Slice C agent does not
//!   touch it.
//! - **Rescore (full-precision re-ranking)** — Slice E.2 owns
//!   `hnsw/search.rs` rescore wiring; Slice C ships only the
//!   primary-kernel beam.
//! - **Persistence + arena routing** — Slice F.1 (per-tenant
//!   `VectorArena`) and Slice G (`VectorPageStore`) own the
//!   bytes-on-disk path; Slice C stores vectors as `Vec<u8>` per
//!   node, owned by the graph itself.
//! - **MN-RU back-up index + dual-search** — Xiao 2024's full
//!   strategy uses a back-up index for the unreachable set;
//!   v1.0's MN-RU is the simpler "detect + reinsert" form which
//!   ADR-003 §Decision Strategy 2 calls out as preserved. The
//!   back-up-index variant is a v1.1 follow-up.
//!
//! `HnswGraph` is local and carries no partition field. The
//! [`crate::VectorIndexHandle`] identifies the owning tenant and index.

pub mod filtered;
mod graph;
mod insert;
#[cfg(test)]
mod property_tests;
mod repair;
mod search;

pub use filtered::{FilteredHnsw, Payload, predicate_filtered_search};
pub use graph::{HnswGraph, HnswParams};

// The canonical `Filter` enum + `PropertyKey` / `PropertyValue`
// types live in [`crate::query`] per ADR-035 amendment-03 (issue
// #127). Re-exported here so existing call sites that resolve
// against `arcgraph_vector::hnsw::Filter` keep compiling — the
// canonical resolution path is `arcgraph_vector::Filter`.
pub use crate::query::{Filter, PropertyKey, PropertyValue};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VectorId;
    use crate::distance::L2F32;

    /// Build a 4-vector toy graph and search for the exact match.
    /// This is a smoke test that the public API is wired top to
    /// bottom (insert → search → recall the planted vector).
    #[test]
    fn smoke_build_and_search_returns_planted_id() {
        let params = HnswParams::default();
        let mut g = HnswGraph::new(params, 3, &L2F32);
        let kernel = L2F32;

        let bytes = |xs: [f32; 3]| {
            let v: [f32; 3] = xs;
            bytemuck::cast_slice(&v).to_vec()
        };

        let v0 = bytes([1.0, 0.0, 0.0]);
        let v1 = bytes([0.0, 1.0, 0.0]);
        let v2 = bytes([0.0, 0.0, 1.0]);
        let v3 = bytes([0.5, 0.5, 0.0]);

        g.insert(VectorId::new(0), &v0, &kernel).unwrap();
        g.insert(VectorId::new(1), &v1, &kernel).unwrap();
        g.insert(VectorId::new(2), &v2, &kernel).unwrap();
        g.insert(VectorId::new(3), &v3, &kernel).unwrap();

        let q = bytes([1.0, 0.0, 0.0]);
        let results = g.search(&q, 1, 10, &kernel).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, VectorId::new(0));
    }

    #[test]
    fn empty_graph_search_returns_empty() {
        let g = HnswGraph::new(HnswParams::default(), 3, &L2F32);
        let q = vec![0u8; 12];
        let r = g.search(&q, 5, 10, &L2F32).unwrap();
        assert!(r.is_empty());
    }
}
