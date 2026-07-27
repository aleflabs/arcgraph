//! Join-graph shape classification for the M4-52 DP enumerator.
//!
//! At v1.0 the shape classification is **descriptive, not algorithmic**:
//! the left-deep DP in [`super::dp`] handles all three shapes
//! identically. Shape detection exists to:
//!
//! 1. Document the shape in EXPLAIN output / debug-prints (forward
//!    consumer at M4-91);
//! 2. Provide a knob for v1.1 sketch-aware pruning (e.g., a star
//!    detector might reorder the (center, leaf) pairs first to
//!    short-circuit branches that can't beat the current best);
//! 3. Make the test suite assert on shape classification (per M4-52
//!    roadmap row "DP enumeration on 2/3/4/5-way joins + star +
//!    linear + bushy" — the shape classifier is what those tests
//!    pin).
//!
//! Per ADR-038 amendment-02 §M4.e bushy enumeration is OUT OF SCOPE
//! for v1.0 (deferred to v1.1). The classifier is honest about this:
//! a "bushy" topology classifies as [`JoinShape::Mixed`] but the DP
//! still runs left-deep over it (cost-optimal-among-left-deep-orderings
//! is the v1.0 contract).
//!
//! # Shape detection algorithm
//!
//! Given N relations + their pairwise binding-overlap edges:
//!
//! - **Star**: exactly one relation (the center) has degree N-1; all
//!   others have degree 1 (each connected only to the center). Common
//!   in LDBC SNB IS3 / IS5 (a person anchor expanding to multiple
//!   substrates).
//! - **Linear**: every relation has degree 1 or 2; exactly two have
//!   degree 1 (the chain endpoints); the rest have degree 2. Common
//!   in long traversals.
//! - **Trivial**: ≤ 1 relation; nothing to enumerate.
//! - **Mixed**: anything else (T-shape, ring, bushy multi-anchor,
//!   etc.).
//!
//! Detection is `O(N²)` (enumerate pairs to count edges); at v1.0
//! plan sizes (N ≤ 8) this is `≤ 64` ops, well under any budget.

use std::collections::BTreeSet;

use super::bindings_in;
use crate::logical_plan::LogicalPlan;
use crate::semantic::bound_ast::BindingId;

/// Classification of a join graph topology.
///
/// Descriptive only at v1.0; the DP itself runs left-deep over every
/// shape per ADR-038 amendment-02 §M4.e. v1.1 sketch-aware pruning
/// may specialize per shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinShape {
    /// 0 or 1 relations — no enumeration needed.
    Trivial,
    /// One center connected to N-1 leaves; leaves connected only to
    /// the center.
    Star,
    /// Chain: each relation connected to its neighbors; endpoints
    /// have degree 1.
    Linear,
    /// Anything else (T-shape, ring, bushy multi-anchor, etc.).
    Mixed,
}

/// Classify the join-graph shape of a leaf list + their inferred
/// edges (via `bindings_in` overlap).
///
/// This is `O(N²)`-time over the leaves; at v1.0 N ≤ 8 the cost is
/// ≤ 64 operations.
#[must_use]
pub fn detect_shape(leaves: &[LogicalPlan]) -> JoinShape {
    let n = leaves.len();
    if n <= 1 {
        return JoinShape::Trivial;
    }

    // Pre-compute binding sets for each leaf.
    let bindings: Vec<BTreeSet<BindingId>> = leaves.iter().map(bindings_in).collect();

    // Count edges per node — an edge between i and j exists when
    // bindings[i] ∩ bindings[j] is non-empty.
    let mut degree = vec![0_usize; n];
    for i in 0..n {
        for j in (i + 1)..n {
            if bindings[i].intersection(&bindings[j]).next().is_some() {
                degree[i] += 1;
                degree[j] += 1;
            }
        }
    }

    classify_from_degrees(&degree)
}

/// Classify a degree sequence into [`JoinShape`].
///
/// Pulled out so unit tests can pin the classifier without
/// constructing full [`LogicalPlan`] sub-trees.
#[must_use]
pub(crate) fn classify_from_degrees(degree: &[usize]) -> JoinShape {
    let n = degree.len();
    if n <= 1 {
        return JoinShape::Trivial;
    }

    // Star: one center has degree N-1, the rest have degree 1.
    let centers = degree.iter().filter(|&&d| d == n - 1).count();
    let leaves_only = degree.iter().filter(|&&d| d == 1).count();
    if centers == 1 && leaves_only == n - 1 {
        return JoinShape::Star;
    }

    // Linear: exactly two nodes with degree 1 (endpoints); the rest
    // have degree 2.
    let endpoints = degree.iter().filter(|&&d| d == 1).count();
    let middles = degree.iter().filter(|&&d| d == 2).count();
    // For N=2 the "linear" case is two-degree-1-endpoints, no middles.
    if n == 2 && endpoints == 2 {
        return JoinShape::Linear;
    }
    if endpoints == 2 && middles == n - 2 {
        return JoinShape::Linear;
    }

    JoinShape::Mixed
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial: 0 or 1 relations.
    #[test]
    fn classify_trivial_zero_relations() {
        assert_eq!(classify_from_degrees(&[]), JoinShape::Trivial);
    }

    #[test]
    fn classify_trivial_one_relation() {
        assert_eq!(classify_from_degrees(&[0]), JoinShape::Trivial);
    }

    /// Two-relation join: classified as Linear (the degenerate chain).
    #[test]
    fn classify_linear_two_relations() {
        assert_eq!(classify_from_degrees(&[1, 1]), JoinShape::Linear);
    }

    /// 3-way star: center degree 2, leaves degree 1 each.
    #[test]
    fn classify_star_three_relations() {
        assert_eq!(classify_from_degrees(&[2, 1, 1]), JoinShape::Star);
    }

    /// 4-way star: center degree 3, leaves degree 1 each.
    #[test]
    fn classify_star_four_relations() {
        assert_eq!(classify_from_degrees(&[3, 1, 1, 1]), JoinShape::Star);
    }

    /// 5-way star: center degree 4, leaves degree 1 each.
    #[test]
    fn classify_star_five_relations() {
        assert_eq!(classify_from_degrees(&[4, 1, 1, 1, 1]), JoinShape::Star);
    }

    /// 4-way linear chain: 1-2-2-1.
    #[test]
    fn classify_linear_four_relations() {
        assert_eq!(classify_from_degrees(&[1, 2, 2, 1]), JoinShape::Linear);
    }

    /// 5-way linear chain.
    #[test]
    fn classify_linear_five_relations() {
        assert_eq!(classify_from_degrees(&[1, 2, 2, 2, 1]), JoinShape::Linear);
    }

    /// T-shape (one node has degree 3, two have degree 1, others
    /// have degree 2): Mixed.
    #[test]
    fn classify_mixed_t_shape() {
        // 4 nodes: center has 3 neighbors, three leaves have 1.
        // Degree pattern: [3, 1, 1, 1] — but n=4, so center = n-1 = 3
        // and leaves = n-1 = 3. That's the STAR case actually. Let me
        // re-design: T-shape with 5 nodes.
        // Center has 3 neighbors; one branch extends 1 deeper.
        // Degrees: [3, 2, 1, 1, 1] — center=3, branch-mid=2, three
        // leaves=1.
        assert_eq!(classify_from_degrees(&[3, 2, 1, 1, 1]), JoinShape::Mixed);
    }

    /// Triangle (ring) of 3 nodes — every node has degree 2.
    #[test]
    fn classify_mixed_triangle() {
        assert_eq!(classify_from_degrees(&[2, 2, 2]), JoinShape::Mixed);
    }

    /// Bushy 4: two pairs of degree 2 nodes — diamond pattern.
    #[test]
    fn classify_mixed_diamond() {
        // 4 nodes, every node has degree 2 (cycle/diamond).
        // Endpoints=0, middles=4 → not Linear (needs endpoints=2).
        // Centers=0 (no node has degree 3=n-1) → not Star.
        // → Mixed.
        assert_eq!(classify_from_degrees(&[2, 2, 2, 2]), JoinShape::Mixed);
    }

    /// Disconnected: degree 0 means the relation isn't connected to
    /// any other. Falls through to Mixed.
    #[test]
    fn classify_mixed_disconnected_pair() {
        // 3 nodes, only 2 connected to each other; 3rd isolated.
        // Degrees: [1, 1, 0]. endpoints=2, middles=0 (need n-2=1 for
        // linear), so not Linear.
        assert_eq!(classify_from_degrees(&[1, 1, 0]), JoinShape::Mixed);
    }

    /// `detect_shape` linear-chain: 4-relation chain. (n=3 is
    /// ambiguous because the path graph K_{1,2} = P_3 — degree
    /// sequence [1,2,1] is identical to a star K_{1,2}; the
    /// classifier picks Star first. n=4 disambiguates: K_{1,3} has
    /// degrees [3,1,1,1], P_4 has [1,2,2,1].)
    #[test]
    fn detect_shape_four_relation_linear_chain() {
        use crate::error::Span;
        use crate::logical_plan::{Direction, LogicalExpand};

        // Chain: a -[1]-> b -[2]-> c -[3]-> d -[4]-> e
        //   leaf 0: expand(0→1) → {0, 1}
        //   leaf 1: expand(1→2) → {1, 2}
        //   leaf 2: expand(2→3) → {2, 3}
        //   leaf 3: expand(3→4) → {3, 4}
        // Edges: 0—1 (1), 1—2 (2), 2—3 (3); none cross-skip.
        // Degrees: 1, 2, 2, 1 → Linear.
        let leaves = vec![
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(1),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: Span::point(1, 1),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(1),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: Span::point(1, 2),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(2),
                to: BindingId::new(3),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: Span::point(1, 3),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(3),
                to: BindingId::new(4),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: Span::point(1, 4),
            }),
        ];
        assert_eq!(detect_shape(&leaves), JoinShape::Linear);
    }

    /// `detect_shape` clique: all 3 leaves share a common binding.
    /// In binding-overlap topology this is a triangle (mathematically
    /// NOT a graph-theoretic star — true binding-overlap stars require
    /// pairwise-disjoint non-center bindings, an unusual shape for
    /// graph queries where a shared anchor variable is the norm).
    /// The DP correctly handles cliques as Mixed with no behavior
    /// change versus Star; this test pins the classifier behavior so
    /// future readers don't expect Star here.
    #[test]
    fn detect_shape_shared_anchor_is_mixed_not_star() {
        use crate::error::Span;
        use crate::logical_plan::{Direction, LogicalExpand, LogicalScan};
        use arcgraph_core::{LabelId, Lsn};

        // 3 leaves all share binding 0 (a multi-pattern MATCH with a
        // shared anchor — the typical LDBC SNB IS3 / IS5 shape).
        // Binding-overlap edges: (0,1), (0,2), (1,2) — triangle.
        let leaves = vec![
            LogicalPlan::Scan(LogicalScan {
                label: Some(LabelId::new(1)),
                var: BindingId::new(0),
                read_lsn: Lsn::MAX,
                span: Span::point(1, 1),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(1),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: Span::point(1, 2),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: Span::point(1, 3),
            }),
        ];
        // Each pair shares binding 0 → triangle / clique → Mixed.
        assert_eq!(detect_shape(&leaves), JoinShape::Mixed);
    }

    /// Trivial-shape sanity: no leaves.
    #[test]
    fn detect_shape_zero_leaves_is_trivial() {
        assert_eq!(detect_shape(&[]), JoinShape::Trivial);
    }
}
