//! System-R-style left-deep DP join-ordering enumerator.
//!
//! # Algorithm
//!
//! Given a list of N leaf relations + their inferred binding-overlap
//! edges, the DP builds the cost-optimal left-deep join tree.
//!
//! State: `dp[S] = (subtree_cost, output_card, plan)` for each subset
//! `S ⊆ {0, …, N-1}`. State count is `2^N - 1` (skipping the empty
//! set). Subsets are encoded as `u32` bitmasks indexed into a flat
//! `Vec<Option<DpEntry>>` of size `2^N` — the fastest deterministic
//! key form. Bitmask iteration order (`mask = 1..2^N`) is naturally
//! sorted, satisfying the "stable iteration order" key requirement
//! per ADR-038 amendment-02 §M4.e.
//!
//! Transitions (left-deep restriction; bushy bushy deferred to v1.1):
//!
//! ```text
//!     for size in 2..=N:
//!       for subset S of size:
//!         for r in S:
//!           let L = S \ {r}
//!           if dp[L] is None: continue        # unreachable
//!           let shared = bindings(L) ∩ bindings({r})
//!           if shared.is_empty(): continue    # reject Cartesian
//!           let local_cost = cost_join(...)
//!           let candidate_cost = local_cost + dp[L].cost + dp[{r}].cost
//!           if candidate_cost < dp[S].cost: dp[S] = candidate
//! ```
//!
//! The right side is always a singleton `{r}` — that's the left-deep
//! restriction. Bushy DP would also enumerate left-subset splits
//! `L ⊊ S` non-singleton, deferred to v1.1.
//!
//! # Per-candidate cost (incremental)
//!
//! Costing each candidate via [`crate::planner::cost::estimate_costs`]
//! would walk the entire candidate plan (`O(N)`), giving an overall
//! `O(N² × 2^N)` walk-cost product. At N=8 that's ~16K cost-walks
//! × ~5 µs ≈ 80 ms, far over the ADR-036 §D-25 5 ms budget.
//!
//! Instead the DP costs each candidate in **O(1)** via
//! [`crate::planner::cost::operator::cost_join`]:
//!
//! ```text
//!     local_cost = cost_join(synthetic_join, left_card, right_card)
//!     candidate_subtree_cost = local_cost + left_subtree_cost + right_subtree_cost
//!     candidate_output_card  = cost_join's returned output_card
//! ```
//!
//! This makes the total DP work `O(N × 2^N)` operations + N initial
//! `estimate_costs` walks. At N=8: ~256 × 8 = 2K ops + 8 × 5 µs ≈
//! 80 µs total. Well inside budget.
//!
//! # Snapshot-once contract
//!
//! All N initial `estimate_costs` calls go through a
//! `super::FrozenCatalog` reference — they read the same captured
//! snapshot. Every per-candidate `cost_join` call uses the same
//! captured snapshot's totals. Apples-to-apples cost comparison
//! across all candidates.
//!
//! # Determinism
//!
//! - Subsets enumerated in bitmask order (deterministic).
//! - Within a subset, splits enumerated in bit-position order
//!   (deterministic).
//! - Tie-break: when two candidates have equal cost, the one
//!   processed FIRST (smaller right-singleton bit) wins. This pins
//!   the input-order tie-break per the M4-52 spawn prompt.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 — `LogicalPlan` exhaustive-match contract (DP
//!   composes via `LogicalJoin` only — exhaustive match deferred to
//!   the rewriter in `super::rewrite`).
//! - ADR-038 amendment-02 §M4.e — left-deep DP scope; bushy v1.1.
//! - ADR-036 §D-25 — 5 ms plan-build budget.

use std::collections::BTreeSet;

use crate::error::Span;
use crate::logical_plan::{JoinAlgorithm, JoinCondition, LogicalJoin, LogicalPlan};
use crate::planner::cost::operator::cost_join;
use crate::planner::cost::{Cardinality, Cost, estimate_costs};
use crate::semantic::CatalogProvider;
use crate::semantic::bound_ast::BindingId;

use super::reroot::build_left_deep;
use super::{FrozenCatalog, MAX_DP_RELATIONS, bindings_in, join_condition_for};

/// One DP-table entry — the best candidate for a subset.
#[derive(Debug, Clone)]
struct DpEntry {
    /// Cumulative subtree cost — the value the DP minimizes.
    subtree_cost: Cost,
    /// Output cardinality of the subtree's root operator (used by
    /// downstream `cost_join` calls when this entry is consumed as
    /// the left input of a parent join).
    output_card: Cardinality,
    /// The cost-optimal plan for this subset.
    plan: LogicalPlan,
}

/// Light-weight stats produced by the DP for telemetry / EXPLAIN
/// integration. Not required at v1.0; we track these so M4-91
/// EXPLAIN can later show "the planner considered N candidates".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DpStats {
    /// Number of leaf relations enumerated over.
    pub leaves: usize,
    /// Number of candidate plans evaluated (`O(N × 2^N)` upper
    /// bound; actual count may be smaller because Cartesian
    /// candidates are rejected).
    pub candidates: usize,
    /// **Why** the DP fell back to input order (or `None` if the DP
    /// ran to a chosen-plan completion). Replaces the W9b `fallback:
    /// bool` per W9b cross-PR review F-6 (LOW). The four kinds carry
    /// different operational semantics:
    ///
    /// - [`DpFallbackReason::Empty`] — n == 0 (defensive; never
    ///   expected from the rewriter; logs at WARN-equivalent verbosity).
    /// - [`DpFallbackReason::OverCap`] — n > [`MAX_DP_RELATIONS`].
    ///   Expected; budget-driven; safe.
    /// - [`DpFallbackReason::Disconnected`] — connectivity check
    ///   rejected; preserves user's Cartesian intent. Expected; safe.
    /// - [`DpFallbackReason::UniverseMissing`] — connected DP completed
    ///   but produced no universe entry. Should-not-happen invariant
    ///   violation; logs loudly.
    ///
    /// M4-91 EXPLAIN forward-consumer (post W9d M4-52b CRIT-1 wiring
    /// landed) can render the reason directly so an MCP-agent
    /// inspecting EXPLAIN output can distinguish "DP didn't run for
    /// budget" from "DP ran but produced bad output".
    pub fallback_reason: Option<DpFallbackReason>,
}

/// Why the DP fell back to input order (W9b cross-PR review F-6 LOW
/// closure; co-packed in W9d M4-52b).
///
/// Each variant maps 1:1 to a fallback site in `enumerate_inner`.
/// Variants carry the leaf count `n` so a downstream consumer can
/// distinguish "5-leaf disconnected" from "12-leaf over-cap" without
/// re-deriving from the input plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DpFallbackReason {
    /// `n == 0`. Defensive — the rewriter never sends an empty list.
    /// If this fires, the rewriter's leaf-extraction has a bug.
    Empty,
    /// `n > MAX_DP_RELATIONS`. Per ADR-036 §D-25 the DP cost exceeds
    /// the 5 ms plan-build budget for `n > 8`. Operational; expected
    /// at v1.0; safe.
    OverCap { n: usize },
    /// The binding-overlap graph is disconnected. The DP can only
    /// enumerate within connected components; v1.0 falls back to
    /// input order which preserves the user's Cartesian intent.
    Disconnected { n: usize },
    /// Connectivity check passed but the DP produced no universe
    /// entry. Should-not-happen given the connectivity invariant
    /// (every connected subset has at least one non-Cartesian split);
    /// surfaces here loudly. Falls back to input order so the user
    /// still gets a valid plan.
    UniverseMissing { n: usize },
}

/// Run the DP over a list of leaf relations and return the
/// cost-optimal left-deep [`LogicalPlan`].
///
/// `frozen` is the snapshot-locked catalog wrapper (see
/// [`super::FrozenCatalog`]); the DP threads `frozen as
/// &dyn CatalogProvider` through every cost call so all candidate
/// plans see identical catalog state.
///
/// `original_span` is the span of the original join-tree root in the
/// input plan; the new joins inherit it so EXPLAIN line-tracks back
/// to the source.
///
/// # Fallback paths
///
/// - `leaves.len() == 0` → returns an `LogicalPlan::Empty` (defensive
///   — the rewriter never sends an empty list).
/// - `leaves.len() == 1` → returns the single leaf unchanged.
/// - `leaves.len() > MAX_DP_RELATIONS` → builds a left-deep tree in
///   input order and returns it (the input plan was already left-deep
///   per M4-31 lowering convention).
/// - Disconnected join graph → builds a left-deep tree in input order
///   (preserves the user's Cartesian intent).
///
/// # Determinism
///
/// Same `leaves` + same snapshot → same output plan. Tie-breaks
/// favor smaller right-singleton bit (input-order preference).
#[must_use]
pub(super) fn enumerate(
    leaves: Vec<LogicalPlan>,
    frozen: &FrozenCatalog<'_>,
    original_span: &Span,
) -> LogicalPlan {
    let (plan, _stats) = enumerate_inner(leaves, frozen, original_span, false);
    plan
}

/// Test-only DP entry that picks the **maximum** cost candidate
/// instead of the minimum. The Phase 4.2 controlled-mutation probe
/// per the M4-52 spawn prompt + PR #232 review §"controlled-mutation
/// probe" relies on this hook: an external proptest can compare the
/// "max-pick" output's cost against a brute-force minimum and
/// confirm the cost-optimality oracle is non-vacuous.
///
/// Production code paths NEVER reach this — it is `pub(crate)` and
/// is only called from the M4-52 proptest harness in
/// `tests/m4_52_join_enumeration_proptest.rs`. Keeping the hook as a
/// real code path (rather than a `cfg(test)` toggle) means the
/// reviewer's reproduction is mechanical: flip the parameter, re-run.
#[must_use]
pub(crate) fn enumerate_pick_max_for_test(
    leaves: Vec<LogicalPlan>,
    catalog: &dyn CatalogProvider,
    original_span: &Span,
) -> LogicalPlan {
    let snapshot = catalog.snapshot();
    let frozen = FrozenCatalog::new(catalog, snapshot);
    let (plan, _stats) = enumerate_inner(leaves, &frozen, original_span, true);
    plan
}

/// Same as [`enumerate`] but also returns [`DpStats`] for telemetry.
///
/// Pulled out as a separate entry so tests + the future M4-91
/// EXPLAIN consumer can pin the candidate count.
#[cfg(test)]
#[must_use]
pub(super) fn enumerate_with_stats(
    leaves: Vec<LogicalPlan>,
    frozen: &FrozenCatalog<'_>,
    original_span: &Span,
) -> (LogicalPlan, DpStats) {
    enumerate_inner(leaves, frozen, original_span, false)
}

/// Internal DP entry — `pick_max` toggles min vs max selection. See
/// [`enumerate_pick_max_for_test`].
#[must_use]
fn enumerate_inner(
    leaves: Vec<LogicalPlan>,
    frozen: &FrozenCatalog<'_>,
    original_span: &Span,
    pick_max: bool,
) -> (LogicalPlan, DpStats) {
    let n = leaves.len();
    if n == 0 {
        // Defensive — rewriter never sends an empty list. If this
        // fires, an upstream invariant is broken.
        tracing::debug!(
            target: "arcgraph_query::planner::dp",
            n = 0,
            reason = "empty",
            "DP fallback: empty leaf list (defensive — rewriter invariant violated)"
        );
        return (
            super::reroot::empty_plan(original_span.clone()),
            DpStats {
                leaves: 0,
                candidates: 0,
                fallback_reason: Some(DpFallbackReason::Empty),
            },
        );
    }
    if n == 1 {
        return (
            leaves.into_iter().next().expect("checked n==1"),
            DpStats {
                leaves: 1,
                candidates: 0,
                fallback_reason: None,
            },
        );
    }
    if n > MAX_DP_RELATIONS {
        // DP is intractable for v1.0 budget; fall back to input order.
        tracing::debug!(
            target: "arcgraph_query::planner::dp",
            n,
            cap = MAX_DP_RELATIONS,
            reason = "over_cap",
            "DP fallback: n > MAX_DP_RELATIONS — using input-order left-deep tree"
        );
        return (
            build_left_deep(leaves, original_span),
            DpStats {
                leaves: n,
                candidates: 0,
                fallback_reason: Some(DpFallbackReason::OverCap { n }),
            },
        );
    }

    // Pre-compute per-leaf binding sets.
    let leaf_bindings: Vec<BTreeSet<BindingId>> = leaves.iter().map(bindings_in).collect();

    // Connectivity check: BFS over leaves using binding-overlap as
    // the edge relation. If disconnected, the DP can only enumerate
    // within connected components — at v1.0 we just fall back to
    // input order (preserves user-intended Cartesian groups).
    if !is_connected(&leaf_bindings) {
        // The binding-overlap graph is disconnected — preserve the
        // user's Cartesian intent by falling back to input order.
        tracing::debug!(
            target: "arcgraph_query::planner::dp",
            n,
            reason = "disconnected",
            "DP fallback: disconnected join graph — preserving Cartesian intent"
        );
        return (
            build_left_deep(leaves, original_span),
            DpStats {
                leaves: n,
                candidates: 0,
                fallback_reason: Some(DpFallbackReason::Disconnected { n }),
            },
        );
    }

    // Initial cost evaluation — one estimate_costs walk per leaf.
    // estimate_costs takes the plan by value, so we clone the leaf
    // for the cost walk; the original moves into the DP table.
    let dp_size = 1_usize << n;
    let mut dp: Vec<Option<DpEntry>> = vec![None; dp_size];
    let frozen_dyn: &dyn CatalogProvider = frozen;
    for (i, leaf) in leaves.iter().enumerate() {
        let costed = estimate_costs(leaf.clone(), frozen_dyn);
        dp[1_usize << i] = Some(DpEntry {
            subtree_cost: costed.total_cost(),
            output_card: costed.output_card(),
            plan: leaf.clone(),
        });
    }

    // Bottom-up DP over left-deep splits. For each subset S of size
    // ≥ 2, try each singleton {r} ⊂ S as the right side; the
    // accumulated left side is L = S \ {r}.
    let mut candidates = 0_usize;
    for mask in 1_u32..(dp_size as u32) {
        let popcount = mask.count_ones();
        if popcount < 2 {
            continue;
        }
        // Iterate over each set bit r ∈ S.
        let mut bits = mask;
        while bits != 0 {
            let r = bits.trailing_zeros();
            let r_mask: u32 = 1 << r;
            let l_mask: u32 = mask & !r_mask;

            let l_entry_opt = dp[l_mask as usize].clone();
            let r_entry_opt = dp[r_mask as usize].clone();
            if let (Some(l_entry), Some(r_entry)) = (l_entry_opt, r_entry_opt) {
                // Derive SharedBindings from binding-set intersection
                // between the accumulated left bindings (union of all
                // leaves in L) and the right singleton's bindings.
                let l_bindings = bindings_for_mask(l_mask, &leaf_bindings);
                let r_bindings = &leaf_bindings[r as usize];
                let shared = join_condition_for(&l_bindings, r_bindings);
                if shared.is_empty() {
                    // Cartesian split — reject (a connected join
                    // graph always has a non-Cartesian alternative).
                    bits &= bits - 1; // clear lowest bit
                    continue;
                }

                let synthetic_join = LogicalJoin {
                    left: Box::new(l_entry.plan.clone()),
                    right: Box::new(r_entry.plan.clone()),
                    on: JoinCondition::SharedBindings(shared.clone()),
                    algorithm: JoinAlgorithm::Auto,
                    span: original_span.clone(),
                };
                let (local_cost, output_card) =
                    cost_join(&synthetic_join, l_entry.output_card, r_entry.output_card);
                let candidate_subtree_cost = local_cost
                    .plus(l_entry.subtree_cost)
                    .plus(r_entry.subtree_cost);

                candidates += 1;

                let take = match &dp[mask as usize] {
                    None => true,
                    // Strictly less wins (production path) — equal
                    // cost loses (input-order preference: an earlier
                    // candidate already recorded wins ties).
                    // Phase 4.2 controlled-mutation hook: when
                    // pick_max=true, picks the MAXIMUM-cost candidate
                    // instead. Used by the M4-52 proptest harness to
                    // demonstrate the cost-optimality oracle is
                    // non-vacuous.
                    Some(existing) => {
                        let candidate_total = candidate_subtree_cost.total();
                        let existing_total = existing.subtree_cost.total();
                        if pick_max {
                            candidate_total > existing_total
                        } else {
                            candidate_total < existing_total
                        }
                    }
                };
                if take {
                    let candidate_plan = LogicalPlan::Join(synthetic_join);
                    dp[mask as usize] = Some(DpEntry {
                        subtree_cost: candidate_subtree_cost,
                        output_card,
                        plan: candidate_plan,
                    });
                }
            }
            bits &= bits - 1; // clear lowest bit
        }
    }

    // Pull the universe entry — the best plan covering all N leaves.
    let universe_mask = dp_size as u32 - 1;
    let chosen = dp[universe_mask as usize].take();
    let fallback_reason = if chosen.is_none() {
        // Should-not-happen: connectivity check passed but no universe
        // entry. Surfaces loudly.
        tracing::debug!(
            target: "arcgraph_query::planner::dp",
            n,
            reason = "universe_missing",
            "DP fallback: connectivity check passed but no universe entry — falling back to input order"
        );
        Some(DpFallbackReason::UniverseMissing { n })
    } else {
        None
    };
    let stats = DpStats {
        leaves: n,
        candidates,
        fallback_reason,
    };
    let plan = chosen
        .map(|e| e.plan)
        // Defensive: if connectivity check passed but DP somehow
        // produced no universe entry (cannot happen given the
        // connectivity invariant — every connected subset has at
        // least one non-Cartesian split), fall back to input order.
        .unwrap_or_else(|| build_left_deep(leaves, original_span));

    (plan, stats)
}

/// Union of binding sets for the leaves selected by `mask`.
fn bindings_for_mask(mask: u32, leaf_bindings: &[BTreeSet<BindingId>]) -> BTreeSet<BindingId> {
    let mut out = BTreeSet::new();
    let mut bits = mask;
    while bits != 0 {
        let i = bits.trailing_zeros() as usize;
        out.extend(leaf_bindings[i].iter().copied());
        bits &= bits - 1;
    }
    out
}

/// Connectivity check via union-find over binding-overlap edges.
fn is_connected(leaf_bindings: &[BTreeSet<BindingId>]) -> bool {
    let n = leaf_bindings.len();
    if n <= 1 {
        return true;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] == x {
            x
        } else {
            let root = find(parent, parent[x]);
            parent[x] = root;
            root
        }
    }
    fn union(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            if leaf_bindings[i]
                .intersection(&leaf_bindings[j])
                .next()
                .is_some()
            {
                union(&mut parent, i, j);
            }
        }
    }
    let root = find(&mut parent, 0);
    (1..n).all(|i| find(&mut parent, i) == root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::types::*;
    use crate::semantic::StubCatalogProvider;
    use crate::semantic::bound_ast::BindingId;
    use arcgraph_core::{LabelId, Lsn};

    fn span() -> Span {
        Span::point(1, 1)
    }

    fn scan(label_raw: u32, var_raw: u64) -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(label_raw)),
            var: BindingId::new(var_raw),
            read_lsn: Lsn::MAX,
            span: span(),
        })
    }

    /// Empty leaf list → returns LogicalPlan::Empty.
    #[test]
    fn enumerate_empty_returns_empty_plan() {
        let cat = StubCatalogProvider::new();
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        let (plan, stats) = enumerate_with_stats(Vec::new(), &frozen, &span());
        assert!(matches!(plan, LogicalPlan::Empty(_)));
        assert_eq!(stats.candidates, 0);
        assert!(stats.fallback_reason.is_some());
    }

    /// Single leaf → returns the leaf unchanged.
    #[test]
    fn enumerate_single_leaf_returns_leaf() {
        let cat = StubCatalogProvider::new();
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        let leaf = scan(1, 0);
        let (plan, stats) = enumerate_with_stats(vec![leaf.clone()], &frozen, &span());
        assert_eq!(plan, leaf);
        assert_eq!(stats.candidates, 0);
    }

    /// 2-way connected join → DP picks (smaller, larger) ordering.
    #[test]
    fn enumerate_two_way_picks_smaller_first() {
        // Both scans share var=0 (equi-join key). Label 1 has 1000
        // rows, label 2 has 10. The cost-optimal left-deep plan
        // joins from the smaller side.
        // Note: cost_join(l, r) charges (l + r) * HASH_JOIN_COST;
        // that's symmetric in l and r, but the OUTPUT cardinality
        // differs because each leaf scan's cost varies. The DP picks
        // by subtree cost.
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 1_000)
            .with_label_cardinality(LabelId::new(2), 10);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        let leaves = vec![scan(1, 0), scan(2, 0)];
        let (plan, stats) = enumerate_with_stats(leaves, &frozen, &span());
        assert_eq!(stats.leaves, 2);
        // 2-way: 2 candidates evaluated (left=A, right=B; left=B, right=A).
        assert_eq!(stats.candidates, 2);
        // Result is a Join with the equi-join key.
        match plan {
            LogicalPlan::Join(j) => match j.on {
                JoinCondition::SharedBindings(ids) => {
                    assert_eq!(ids, vec![BindingId::new(0)]);
                }
            },
            _ => panic!("expected Join at root"),
        }
    }

    /// 3-way star: center binding 0, two leaves binding (0,1) and
    /// (0,2). The DP should put the center first.
    #[test]
    fn enumerate_three_way_star_picks_optimal_ordering() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_total_rel_count(50_000)
            .with_label_cardinality(LabelId::new(1), 100) // center scan
            .with_label_cardinality(LabelId::new(2), 1_000)
            .with_label_cardinality(LabelId::new(3), 1_000);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());

        // Leaves all share binding 0 (the center).
        let leaves = vec![
            scan(1, 0), // center {0}
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(1),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
        ];
        let (plan, stats) = enumerate_with_stats(leaves, &frozen, &span());
        assert_eq!(stats.leaves, 3);
        // 3-way: subset {0,1,2} has 3 splits (right=0, right=1,
        // right=2); plus subsets of size 2 (3 of them × 2 splits each
        // = 6); total candidates = 9. (Some may be Cartesian-rejected.)
        assert!(stats.candidates >= 3);
        assert!(stats.fallback_reason.is_none());
        // Root must be a Join.
        assert!(matches!(plan, LogicalPlan::Join(_)));
    }

    /// 4-way linear chain: a-b-c-d. DP picks left-deep traversal.
    #[test]
    fn enumerate_four_way_linear_chain() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_total_rel_count(50_000);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());

        // Chain: each expand connects to the next via shared
        // intermediate binding.
        let leaves = vec![
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(1),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(1),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(2),
                to: BindingId::new(3),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(3),
                to: BindingId::new(4),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
        ];
        let (plan, stats) = enumerate_with_stats(leaves, &frozen, &span());
        assert_eq!(stats.leaves, 4);
        assert!(stats.fallback_reason.is_none());
        assert!(matches!(plan, LogicalPlan::Join(_)));
    }

    /// 5-way mixed: DP runs full enumeration.
    #[test]
    fn enumerate_five_way_mixed() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_total_rel_count(50_000);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());

        // Mixed: leaf 0 connects to 1 and 2; leaf 3 connects to 4.
        // Bridge between the two clusters via leaf 2 connecting to 3.
        let leaves = vec![
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(1),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(2),
                to: BindingId::new(3),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(3),
                to: BindingId::new(4),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(4),
                to: BindingId::new(5),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
        ];
        let (plan, stats) = enumerate_with_stats(leaves, &frozen, &span());
        assert_eq!(stats.leaves, 5);
        assert!(stats.fallback_reason.is_none());
        assert!(matches!(plan, LogicalPlan::Join(_)));
    }

    /// Memoization correctness: running the DP twice on the same
    /// inputs produces byte-equal output (deterministic).
    #[test]
    fn enumerate_is_deterministic_across_invocations() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 50)
            .with_label_cardinality(LabelId::new(3), 25);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());

        let leaves_a = vec![
            scan(1, 0),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(1),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
            LogicalPlan::Expand(LogicalExpand {
                from: BindingId::new(0),
                to: BindingId::new(2),
                direction: Direction::LeftToRight,
                rel_type: None,
                length_range: None,
                rel_var: None,
                span: span(),
            }),
        ];
        let leaves_b = leaves_a.clone();

        let (plan_a, _) = enumerate_with_stats(leaves_a, &frozen, &span());
        let (plan_b, _) = enumerate_with_stats(leaves_b, &frozen, &span());
        assert_eq!(plan_a, plan_b);
    }

    /// Tie-break determinism: two leaves with identical cost; DP
    /// picks input order (left = leaves[0]).
    #[test]
    fn enumerate_tie_break_input_order() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(1_000)
            .with_label_cardinality(LabelId::new(1), 100)
            .with_label_cardinality(LabelId::new(2), 100); // SAME card → equal cost
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        let leaves = vec![scan(1, 0), scan(2, 0)];
        let (plan, _) = enumerate_with_stats(leaves, &frozen, &span());
        // First processed candidate (right=leaves[0]) wins ties via
        // strictly-less-than oracle. So the chosen plan has
        // right=leaves[0]=scan(1,0) and left=leaves[1]=scan(2,0).
        match plan {
            LogicalPlan::Join(j) => {
                match *j.left {
                    LogicalPlan::Scan(s) => assert_eq!(s.label, Some(LabelId::new(2))),
                    _ => panic!("expected Scan on left"),
                }
                match *j.right {
                    LogicalPlan::Scan(s) => assert_eq!(s.label, Some(LabelId::new(1))),
                    _ => panic!("expected Scan on right"),
                }
            }
            _ => panic!("expected Join"),
        }
    }

    /// Disconnected join graph → fallback to input order
    /// (preserves Cartesian intent).
    #[test]
    fn enumerate_disconnected_returns_input_order() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        let leaves = vec![scan(1, 0), scan(2, 1), scan(3, 2)]; // disjoint bindings
        let (plan, stats) = enumerate_with_stats(leaves.clone(), &frozen, &span());
        // W9b F-6 closure: structured fallback reason — Disconnected.
        assert!(matches!(
            stats.fallback_reason,
            Some(DpFallbackReason::Disconnected { n: 3 })
        ));
        // Fallback plan is left-deep in input order.
        match plan {
            LogicalPlan::Join(outer) => match *outer.right {
                LogicalPlan::Scan(s) => assert_eq!(s.label, Some(LabelId::new(3))),
                _ => panic!("rightmost leaf should be scan(3)"),
            },
            _ => panic!("expected Join at root for fallback"),
        }
    }

    /// N > MAX_DP_RELATIONS → fallback to input order.
    #[test]
    fn enumerate_oversized_returns_fallback_left_deep() {
        let cat = StubCatalogProvider::new().with_total_node_count(1_000);
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        // Build N+1 connected scans (all share var=0).
        let n = MAX_DP_RELATIONS + 1;
        let leaves: Vec<_> = (0..n).map(|i| scan(i as u32 + 1, 0)).collect();
        let (plan, stats) = enumerate_with_stats(leaves, &frozen, &span());
        // W9b F-6 closure: structured fallback reason — OverCap.
        match stats.fallback_reason {
            Some(DpFallbackReason::OverCap { n: actual_n }) => {
                assert_eq!(actual_n, n);
            }
            other => panic!("expected OverCap fallback, got {other:?}"),
        }
        assert_eq!(stats.candidates, 0);
        assert!(matches!(plan, LogicalPlan::Join(_)));
    }

    /// Empty leaf list → fallback reason is `Empty`. W9b F-6 pin.
    #[test]
    fn enumerate_empty_carries_empty_fallback_reason() {
        let cat = StubCatalogProvider::new();
        let frozen = FrozenCatalog::new(&cat, cat.snapshot());
        let (_, stats) = enumerate_with_stats(Vec::new(), &frozen, &span());
        assert_eq!(stats.fallback_reason, Some(DpFallbackReason::Empty));
    }

    /// is_connected: 3 leaves with star-shaped overlap → connected.
    #[test]
    fn is_connected_star_shape_true() {
        let bindings = vec![
            [BindingId::new(0)].iter().copied().collect::<BTreeSet<_>>(),
            [BindingId::new(0), BindingId::new(1)]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            [BindingId::new(0), BindingId::new(2)]
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
        ];
        assert!(is_connected(&bindings));
    }

    /// is_connected: 3 leaves with disjoint bindings → disconnected.
    #[test]
    fn is_connected_disjoint_false() {
        let bindings = vec![
            [BindingId::new(0)].iter().copied().collect::<BTreeSet<_>>(),
            [BindingId::new(1)].iter().copied().collect::<BTreeSet<_>>(),
            [BindingId::new(2)].iter().copied().collect::<BTreeSet<_>>(),
        ];
        assert!(!is_connected(&bindings));
    }
}
