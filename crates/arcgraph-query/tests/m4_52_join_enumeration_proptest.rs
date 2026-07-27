//! Property tests for M4-52 (M4-05b) DP-based binary-join ordering.
//!
//! # Cost-optimality invariant (load-bearing oracle)
//!
//! For every random N-leaf connected join graph, the DP-chosen
//! plan's total cost (re-derived via [`estimate_costs`]) is **less
//! than or equal to** every left-deep permutation's cost. The DP is
//! the System-R algorithm; this is its defining correctness
//! property. The oracle is the brute-force enumeration.
//!
//! Total cases: 256 (default) + 10K under `PROPTEST_CASES=10000`
//! per the M4-52 spawn-prompt empirical-gauntlet.
//!
//! # Phase 4.2 controlled-mutation probe (proves the oracle is
//! non-vacuous)
//!
//! Per the M4-52 spawn-prompt + PR #232 review §"controlled-mutation
//! probe": the proptest oracle must catch a faulty DP. We verify
//! this by:
//!
//! 1. Running the proptest with the production DP — MUST PASS.
//! 2. Running the proptest with the test-only "pick-max" DP variant
//!    — MUST FAIL at minimal input.
//! 3. (Reviewer side: revert mutation; verify PASS again.)
//!
//! The deterministic [`phase_4_2_controlled_mutation_probe_oracle_is_non_vacuous`]
//! test executes (1) and (2) on a fixed N=3 case; the broadened
//! [`arb_connected_join_case`] strategy below feeds the same probe
//! across the full input envelope (linear / T-shape / star / clique /
//! mixed at varying N + RankByHybrid leaves) — when production
//! `rewrite` is locally mutated to `pick_max=true` (PR #242 round-2
//! reviewer-side three-state cycle), the proptest fails at minimal
//! input across all sampled shapes, restored, passes at
//! `PROPTEST_CASES=10000`.
//!
//! # Generators
//!
//! [`arb_connected_join_case`] generates a random connected
//! N-leaf join graph by:
//!
//! 1. Sampling N ∈ {2..=6} (small; brute-force enumeration is
//!    `O(N!)` so we keep it tractable). N=7 / N=8 correctness is
//!    pinned by bench + unit tests in
//!    [`arcgraph_query::planner::enumeration::dp`] rather than
//!    proptest, because brute-force at N=8 (40,320 perms × ~50 µs
//!    walk) blows the 10K-case PROPTEST runtime budget.
//! 2. Sampling shape ∈ {linear chain, star, clique, T-shape, mixed
//!    sparse connected graph}. Each shape's edge set drives the
//!    binding-overlap graph the DP enumerates over.
//! 3. Sampling per-leaf operator kind ∈ {[`LogicalScan`],
//!    [`LogicalRankByHybrid`]}. Single-binding leaves are randomly
//!    one or the other; multi-binding leaves are forced to
//!    multi-operand RankByHybrid (Scan only carries one binding).
//! 4. Sampling per-leaf cardinality in `[1, 1_000_000]` so the
//!    cost-ordering is meaningfully driven by stats (Scan leaves
//!    use it as label cardinality; RankByHybrid leaves use it as
//!    `K`).
//!
//! N=6 keeps brute-force at 720 candidates per case × ~50 µs walk
//! ≈ 36 ms per case × 256 cases ≈ 9 s wall-time on CI under default
//! config. Within budget.
//!
//! # Round-2 review fix-up note (PR #242 review M-1 + M-3)
//!
//! Round-1 reviewer flagged the original strategy as degenerate:
//! 5 hand-crafted shared-anchor cases regardless of `PROPTEST_CASES`.
//! This file was rewritten in round-2 to sample real edge sets
//! across multiple shapes + per-leaf binding sets + per-leaf
//! operator kinds. The Phase 4.2 controlled-mutation probe was
//! re-verified on the broadened strategy (mutation FAILS at minimal
//! input across the full envelope, restored, passes at 10K cases).
//!
//! Round-2 also folds in M-3: the local [`collect_bindings`] helper
//! (used by the brute-force enumerator's [`build_left_deep`])
//! mirrors the production `enumeration::bindings_in` exhaustively
//! over all 20 [`LogicalPlan`] variants — preventing a silent
//! footgun where extending the strategy to non-Scan leaves would
//! compute incorrect `SharedBindings` for the brute-force baseline.

use std::collections::HashSet;

use arcgraph_core::{LabelId, Lsn};
use arcgraph_query::ast::Literal;
use arcgraph_query::error::Span;
use arcgraph_query::logical_plan::{
    HybridOperand, HybridOperandKind, JoinAlgorithm, JoinCondition, LogicalJoin, LogicalPlan,
    LogicalRankByHybrid, LogicalScan,
};
use arcgraph_query::planner::cost::estimate_costs;
use arcgraph_query::planner::enumerate_join_order;
use arcgraph_query::planner::enumeration::enumerate_join_order_pick_max_for_proptest;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};
use proptest::prelude::*;

fn span() -> Span {
    Span::point(1, 1)
}

/// Join-graph shape sampled by [`arb_connected_join_case`].
///
/// Each variant defines the EDGE set of the binding-overlap graph
/// the DP enumerates over. Leaves are fully connected to a fresh
/// per-edge `BindingId`; the leaf's binding set is the union of
/// edges incident to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JoinShape {
    /// Linear chain: edges `(0,1), (1,2), …, (n-2, n-1)`. The
    /// canonical LDBC SNB IS7 shape.
    Linear,
    /// Star: edges `(0,1), (0,2), …, (0, n-1)` — center binding 0.
    /// LDBC SNB IS6 forum-membership shape.
    Star,
    /// Clique: all pairs `(i, j)` for `i < j`. Worst-case dense
    /// shape; every leaf shares a binding with every other.
    Clique,
    /// T-shape: linear chain plus 1 branch (chain interior →
    /// chain end). Degenerates to Linear at `N ≤ 3` (no spare
    /// leaf for the branch).
    TShape,
    /// Mixed: spanning-tree (linear chain) plus 1–3 random extra
    /// edges. Produces non-canonical sparse connected graphs.
    Mixed,
}

/// Operator kind for a single-binding leaf. Multi-binding leaves
/// are forced to [`LogicalRankByHybrid`] (Scan only carries one
/// binding).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeafKind {
    Scan,
    RankByHybrid,
}

#[derive(Debug, Clone)]
struct JoinCase {
    /// The leaves; each is either [`LogicalScan`] or
    /// [`LogicalRankByHybrid`] depending on its binding-set
    /// cardinality.
    leaves: Vec<LogicalPlan>,
    /// Catalog snapshot to cost against.
    catalog: StubCatalogProvider,
    /// Sampled shape (recorded for diagnostic counter-example
    /// minimization output).
    #[allow(dead_code)]
    shape: JoinShape,
}

impl JoinCase {
    /// Build the input plan as a left-deep tree in input order
    /// (matches the M4-31 lowering convention for a multi-pattern
    /// MATCH).
    fn input_plan(&self) -> LogicalPlan {
        build_left_deep(&self.leaves)
    }

    /// Brute-force minimum cost over all N! left-deep permutations
    /// of leaves. Returns the minimum total cost as f64.
    fn brute_force_min_cost(&self) -> f64 {
        let mut perm: Vec<usize> = (0..self.leaves.len()).collect();
        let mut best = f64::INFINITY;
        permute(&mut perm, 0, &mut |p| {
            let leaves: Vec<LogicalPlan> = p.iter().map(|&i| self.leaves[i].clone()).collect();
            let plan = build_left_deep(&leaves);
            let cost = estimate_costs(plan, &self.catalog).total_cost();
            if cost.total() < best {
                best = cost.total();
            }
        });
        best
    }
}

/// Build a left-deep [`LogicalPlan`] tree from a slice of leaves,
/// computing each Join's [`JoinCondition`] from binding-set
/// intersection.
///
/// Direct port of `enumeration::reroot::build_left_deep`, kept
/// here so the brute-force enumerator doesn't import a `pub(super)`
/// symbol.
fn build_left_deep(leaves: &[LogicalPlan]) -> LogicalPlan {
    assert!(!leaves.is_empty());
    let mut acc = leaves[0].clone();
    for leaf in &leaves[1..] {
        let l_b = collect_bindings(&acc);
        let r_b = collect_bindings(leaf);
        let mut shared: Vec<BindingId> = l_b.intersection(&r_b).copied().collect();
        shared.sort_by_key(|b| b.raw());
        acc = LogicalPlan::Join(LogicalJoin {
            left: Box::new(acc),
            right: Box::new(leaf.clone()),
            on: JoinCondition::SharedBindings(shared),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        });
    }
    acc
}

/// Exhaustive over all 20 [`LogicalPlan`] variants — mirrors
/// production `enumeration::bindings_in`.
///
/// Round-2 reviewer (M-3) flagged the original `_ => {}`-fallback
/// implementation as a silent footgun on strategy extension: the
/// broadened strategy plants [`LogicalRankByHybrid`] leaves whose
/// operand bindings the original helper would have dropped,
/// producing incorrect `SharedBindings` in the brute-force
/// baseline. This version is exhaustive — adding a new
/// [`LogicalPlan`] variant requires updating this match.
fn collect_bindings(plan: &LogicalPlan) -> HashSet<BindingId> {
    let mut out = HashSet::new();
    visit_bindings(plan, &mut out);
    out
}

fn visit_bindings(plan: &LogicalPlan, out: &mut HashSet<BindingId>) {
    match plan {
        LogicalPlan::Scan(s) => {
            out.insert(s.var);
        }
        LogicalPlan::PropertyIndexScan(p) => {
            out.insert(p.var);
        }
        LogicalPlan::CountStore(c) => {
            out.insert(c.output_id);
        }
        LogicalPlan::Expand(e) => {
            out.insert(e.from);
            out.insert(e.to);
            if let Some(rv) = e.rel_var {
                out.insert(rv);
            }
        }
        LogicalPlan::Filter(f) => visit_bindings(&f.input, out),
        LogicalPlan::Project(p) => visit_bindings(&p.input, out),
        LogicalPlan::Limit(l) => visit_bindings(&l.input, out),
        LogicalPlan::Skip(s) => visit_bindings(&s.input, out),
        LogicalPlan::DynamicLimit(l) => visit_bindings(&l.input, out),
        LogicalPlan::Sort(s) => visit_bindings(&s.input, out),
        LogicalPlan::Distinct(d) => visit_bindings(&d.input, out),
        LogicalPlan::Unwind(u) => {
            visit_bindings(&u.input, out);
            out.insert(u.var);
        }
        LogicalPlan::ProcedureCall(p) => {
            visit_bindings(&p.input, out);
            for (_, bid) in &p.columns {
                out.insert(*bid);
            }
        }
        LogicalPlan::Aggregate(a) => visit_bindings(&a.input, out),
        LogicalPlan::CommunityLookup(c) => {
            visit_bindings(&c.input, out);
            out.insert(c.node_var);
        }
        LogicalPlan::NamedPath(n) => {
            visit_bindings(&n.input, out);
            out.insert(n.path_var);
        }
        LogicalPlan::Join(j) => {
            visit_bindings(&j.left, out);
            visit_bindings(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            visit_bindings(&j.left, out);
            visit_bindings(&j.right, out);
        }
        LogicalPlan::Fusion(f) => {
            for input in &f.inputs {
                visit_bindings(input, out);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                visit_bindings(arm, out);
            }
        }
        LogicalPlan::RankByHybrid(r) => {
            for op in &r.operands {
                out.insert(op.var);
            }
            if let Some(score) = r.score_binding {
                out.insert(score);
            }
        }
        LogicalPlan::VectorNear(v) => {
            out.insert(v.var);
        }
        LogicalPlan::TextMatch(t) => {
            out.insert(t.var);
        }
        LogicalPlan::CreateNode(c) => {
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        LogicalPlan::CreateVectorIndex(_) => {}
        LogicalPlan::CreatePropertyIndex(_) => {}
        LogicalPlan::CreateRel(c) => {
            visit_bindings(&c.source_plan, out);
            visit_bindings(&c.target_plan, out);
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        LogicalPlan::Delete(d) => visit_bindings(&d.input, out),
        LogicalPlan::Set(s) => visit_bindings(&s.input, out),
        LogicalPlan::Remove(r) => visit_bindings(&r.input, out),
        LogicalPlan::Merge(m) => {
            visit_bindings(&m.match_branch, out);
            visit_bindings(&m.create_branch, out);
        }
        // ADR-192 (#623): CALL{} output bindings = driving input ++
        // returned; the seed carries its imported set.
        LogicalPlan::Call(c) => {
            visit_bindings(&c.input, out);
            for b in &c.returned {
                out.insert(*b);
            }
        }
        LogicalPlan::CorrelationSeed(s) => {
            for b in &s.imported {
                out.insert(*b);
            }
        }
        LogicalPlan::Empty(_) => {}
    }
}

/// Heap's algorithm for permutations, in-place.
fn permute<F: FnMut(&[usize])>(p: &mut [usize], k: usize, callback: &mut F) {
    if k == p.len() {
        callback(p);
        return;
    }
    for i in k..p.len() {
        p.swap(k, i);
        permute(p, k + 1, callback);
        p.swap(k, i);
    }
}

/// Compute the canonical edge set for a sampled shape. All edges
/// are returned in canonical form `(lo, hi)` with `lo < hi`,
/// deduplicated.
fn shape_edges(n: usize, shape: JoinShape, mixed_extras: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut edges: Vec<(usize, usize)> = match shape {
        JoinShape::Linear => (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect(),
        JoinShape::Star => (1..n).map(|i| (0, i)).collect(),
        JoinShape::Clique => {
            let mut e = Vec::new();
            for i in 0..n {
                for j in (i + 1)..n {
                    e.push((i, j));
                }
            }
            e
        }
        JoinShape::TShape => {
            // Linear chain plus a single chain-interior → chain-end
            // branch. At N ≤ 3 the branch would coincide with an
            // existing chain edge, so the shape collapses to Linear
            // (handled by the dedup below).
            let mut e: Vec<_> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
            if n >= 4 {
                e.push((1, n - 1));
            }
            e
        }
        JoinShape::Mixed => {
            // Spanning tree (linear chain) plus sampled extras.
            // The chain guarantees connectivity; extras add density.
            let mut e: Vec<_> = (0..n.saturating_sub(1)).map(|i| (i, i + 1)).collect();
            for &(a, b) in mixed_extras {
                if a < n && b < n && a != b {
                    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                    e.push((lo, hi));
                }
            }
            e
        }
    };
    edges.sort_unstable();
    edges.dedup();
    edges
}

/// Connectivity check via union-find — defensive validation that
/// the sampled edge set produces a connected binding-overlap
/// graph. Every shape generator is connected by construction, but
/// the check guards against a future shape addition silently
/// breaking the contract.
fn edges_connect(n: usize, edges: &[(usize, usize)]) -> bool {
    if n <= 1 {
        return true;
    }
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        if parent[x] == x {
            x
        } else {
            let r = find(parent, parent[x]);
            parent[x] = r;
            r
        }
    }
    fn unify(parent: &mut [usize], a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent[ra] = rb;
        }
    }
    for &(a, b) in edges {
        unify(&mut parent, a, b);
    }
    let root = find(&mut parent, 0);
    (1..n).all(|i| find(&mut parent, i) == root)
}

/// Build a [`JoinCase`] from a sampled shape + per-leaf operator
/// kinds + per-leaf cardinalities.
///
/// Each edge `(i, j)` allocates a fresh [`BindingId`]; both leaves
/// `i` and `j` register that binding. A leaf's binding set is the
/// union of all edges it's incident to.
///
/// Leaf-type policy:
/// - Single-binding leaf: honor the sampled `kinds[i]` (Scan or
///   1-operand RankByHybrid).
/// - Multi-binding leaf: forced to multi-operand RankByHybrid
///   ([`LogicalScan`] only carries one binding).
fn build_join_case(
    n: usize,
    edges: &[(usize, usize)],
    kinds: &[LeafKind],
    cards: &[u64],
    shape: JoinShape,
) -> JoinCase {
    let mut leaf_bindings: Vec<Vec<BindingId>> = vec![Vec::new(); n];
    for (idx, &(i, j)) in edges.iter().enumerate() {
        // Edge bindings start at 1_000 to leave low IDs free for
        // any future per-leaf "private" bindings.
        let bid = BindingId::new(1_000 + idx as u64);
        leaf_bindings[i].push(bid);
        leaf_bindings[j].push(bid);
    }
    // Defensive: any connected graph with N ≥ 2 has every leaf
    // incident to ≥ 1 edge, so this branch is unreachable in
    // practice — but we shore it up so a future degenerate shape
    // does not produce a zero-binding leaf that breaks the
    // brute-force baseline.
    for (i, b) in leaf_bindings.iter_mut().enumerate() {
        if b.is_empty() {
            b.push(BindingId::new(2_000 + i as u64));
        }
    }
    for b in leaf_bindings.iter_mut() {
        b.sort_by_key(|x| x.raw());
        b.dedup();
    }

    let mut leaves = Vec::with_capacity(n);
    for i in 0..n {
        let leaf = if leaf_bindings[i].len() == 1 && kinds[i] == LeafKind::Scan {
            LogicalPlan::Scan(LogicalScan {
                label: Some(LabelId::new((i + 1) as u32)),
                var: leaf_bindings[i][0],
                read_lsn: Lsn::MAX,
                span: span(),
            })
        } else {
            // Multi-binding leaf, OR single-binding leaf where the
            // sample chose RankByHybrid. Build a hybrid leaf with
            // one operand per binding; the K parameter (sampled in
            // `cards[i]`) drives the cost variation.
            let k_sample = cards[i].max(1);
            let operands: Vec<HybridOperand> = leaf_bindings[i]
                .iter()
                .map(|b| HybridOperand {
                    kind: HybridOperandKind::Vector,
                    var: *b,
                    property: "embedding".to_string(),
                    query: BoundExpression::Literal {
                        value: Literal::Bool(true),
                        span: span(),
                        type_info: None,
                    },
                    k: k_sample,
                    read_lsn: Lsn::MAX,
                    span: span(),
                })
                .collect();
            LogicalPlan::RankByHybrid(LogicalRankByHybrid {
                operands,
                score_binding: None,
                fusion: None,
                span: span(),
            })
        };
        leaves.push(leaf);
    }

    // Catalog: per-leaf cardinality stamped at the leaf's label
    // (used by Scan leaves; RankByHybrid leaves derive their
    // output cardinality from `K` directly).
    let mut cat = StubCatalogProvider::new()
        .with_total_node_count(1_000_000)
        .with_total_rel_count(5_000_000);
    for (i, c) in cards.iter().enumerate() {
        cat = cat.with_label_cardinality(LabelId::new((i + 1) as u32), *c);
    }
    JoinCase {
        leaves,
        catalog: cat,
        shape,
    }
}

fn arb_join_shape() -> impl Strategy<Value = JoinShape> {
    prop_oneof![
        Just(JoinShape::Linear),
        Just(JoinShape::Star),
        Just(JoinShape::Clique),
        Just(JoinShape::TShape),
        Just(JoinShape::Mixed),
    ]
}

fn arb_leaf_kind() -> impl Strategy<Value = LeafKind> {
    prop_oneof![Just(LeafKind::Scan), Just(LeafKind::RankByHybrid)]
}

/// Strategy: pick `N ∈ {2..=6}` × shape × per-leaf kind × per-leaf
/// cardinality and build a connected-by-construction join graph.
///
/// Brute-force budget guard: N is bounded at 6 because the
/// brute-force oracle in [`JoinCase::brute_force_min_cost`]
/// enumerates `N!` permutations. At N=6 = 720 perms × ~50 µs walk
/// per perm = 36 ms per case; at N=8 it would be 40,320 perms ≈
/// 2 sec per case, blowing the 10K-case PROPTEST runtime budget.
///
/// N=7,8 correctness for the production envelope is pinned by:
/// - Bench: `benches/m4_52_dp_enumeration.rs` (performance budget).
/// - Unit tests: `dp::tests::enumerate_oversized_returns_fallback_left_deep`
///   pins `N > MAX_DP_RELATIONS` fallback.
/// - The `MAX_DP_RELATIONS = 8` cap deterministically falls back
///   to input-order at N > 8.
fn arb_connected_join_case() -> impl Strategy<Value = JoinCase> {
    (2usize..=6, arb_join_shape())
        .prop_flat_map(|(n, shape)| {
            // Mixed shape needs random extras; other shapes ignore.
            let extras_strat: BoxedStrategy<Vec<(usize, usize)>> = match shape {
                JoinShape::Mixed if n >= 3 => prop::collection::vec((0..n, 0..n), 1..=3).boxed(),
                _ => Just(Vec::<(usize, usize)>::new()).boxed(),
            };
            let kinds_strat = prop::collection::vec(arb_leaf_kind(), n);
            // Cardinalities span 4 orders of magnitude so the
            // ordering is meaningfully cost-driven (Selinger plan
            // shapes pivot at ~10× cardinality skews).
            let cards_strat = prop::collection::vec(1u64..=1_000_000, n);
            (Just(n), Just(shape), extras_strat, kinds_strat, cards_strat)
        })
        .prop_filter_map(
            "binding-overlap graph must be connected for DP to enumerate",
            |(n, shape, extras, kinds, cards)| {
                let edges = shape_edges(n, shape, &extras);
                if !edges_connect(n, &edges) {
                    return None;
                }
                Some(build_join_case(n, &edges, &kinds, &cards, shape))
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// **Cost-optimality invariant (load-bearing oracle).**
    ///
    /// For every random connected join case sampled from the
    /// broadened strategy (shapes ∈ {linear, star, clique, T,
    /// mixed} × leaf kinds ∈ {Scan, RankByHybrid} × cardinalities
    /// ∈ [1, 1_000_000]), the DP-chosen plan's total cost ≤ every
    /// left-deep permutation's cost.
    #[test]
    fn dp_chosen_plan_is_cost_optimal_among_left_deep_permutations(
        case in arb_connected_join_case(),
    ) {
        let input = case.input_plan();
        let dp_plan = enumerate_join_order(input, &case.catalog);
        let dp_cost = estimate_costs(dp_plan, &case.catalog).total_cost();
        let bf_min = case.brute_force_min_cost();
        // Allow ε for floating-point round-off across cost-tree
        // re-walks. The DP's cost is computed incrementally; the
        // brute-force is computed via estimate_costs walks. Both
        // should agree to within a few ULP; use 1e-3 as a generous
        // band that catches all realistic divergences.
        prop_assert!(
            dp_cost.total() <= bf_min + 1e-3,
            "DP cost {} > brute-force min {} (Δ={})",
            dp_cost.total(),
            bf_min,
            dp_cost.total() - bf_min,
        );
    }

    /// **Determinism invariant.** The DP is fully deterministic —
    /// running it twice on the same inputs produces identical plans.
    #[test]
    fn dp_is_deterministic(case in arb_connected_join_case()) {
        let plan_a = enumerate_join_order(case.input_plan(), &case.catalog);
        let plan_b = enumerate_join_order(case.input_plan(), &case.catalog);
        prop_assert_eq!(plan_a, plan_b);
    }
}

/// **Phase 4.2 controlled-mutation probe — proves the
/// cost-optimality oracle is non-vacuous.**
///
/// We run a hand-crafted N=3 case through the production DP
/// (`enumerate_join_order`) and through the test-only "pick-max" DP
/// (`enumerate_join_order_pick_max_for_proptest`).
///
/// - Production DP cost ≤ brute-force min — the oracle PASSES.
/// - Pick-max DP cost > brute-force min — the oracle would FAIL on
///   this case. Proves any future regression that flips min/max
///   selection (or otherwise picks a sub-optimal candidate) WILL be
///   caught by the proptest above.
///
/// This test does NOT use proptest itself; we want a deterministic
/// minimal case to demonstrate the mutation behavior. The probe
/// case:
///
/// - 3 leaves all sharing var=0 (anchor).
/// - Cardinalities: 1000 / 100 / 10 — large spread so min-cost
///   ordering is materially different from max-cost ordering.
///
/// The broadened [`arb_connected_join_case`] strategy + the
/// reviewer-side three-state cycle (described in the
/// `phase_4_2_controlled_mutation_probe` module-doc note) extends
/// this non-vacuity proof from the N=3 hand-crafted case to the
/// full input envelope.
#[test]
fn phase_4_2_controlled_mutation_probe_oracle_is_non_vacuous() {
    let cat = StubCatalogProvider::new()
        .with_total_node_count(10_000)
        .with_label_cardinality(LabelId::new(1), 1_000)
        .with_label_cardinality(LabelId::new(2), 100)
        .with_label_cardinality(LabelId::new(3), 10);

    let leaves = vec![
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        }),
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(2)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        }),
        LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(3)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        }),
    ];

    // Brute-force min over all 3! = 6 left-deep permutations.
    let case = JoinCase {
        leaves: leaves.clone(),
        catalog: cat.clone(),
        shape: JoinShape::Clique,
    };
    let bf_min = case.brute_force_min_cost();

    // Production DP cost.
    let dp_plan = enumerate_join_order(case.input_plan(), &cat);
    let dp_cost = estimate_costs(dp_plan, &cat).total_cost().total();

    // Pick-max DP cost.
    let mutated_plan = enumerate_join_order_pick_max_for_proptest(case.input_plan(), &cat);
    let mutated_cost = estimate_costs(mutated_plan, &cat).total_cost().total();

    // Oracle PASSES on production DP.
    assert!(
        dp_cost <= bf_min + 1e-3,
        "production DP must be cost-optimal (dp={}, bf_min={})",
        dp_cost,
        bf_min,
    );

    // Oracle FAILS (i.e., catches the mutation) on pick-max DP. The
    // mutated cost is the MAXIMUM-cost ordering; for a non-degenerate
    // input it strictly exceeds the brute-force minimum.
    assert!(
        mutated_cost > bf_min + 1e-3,
        "pick-max mutation must produce a strictly-worse plan to demonstrate oracle is non-vacuous (mutated={}, bf_min={})",
        mutated_cost,
        bf_min,
    );
}

/// Parity pin for round-2 fix-up M-3: the local exhaustive
/// [`collect_bindings`] helper produces the expected binding set
/// for the variants the broadened strategy generates
/// ([`LogicalScan`] and [`LogicalRankByHybrid`] are the
/// strategy's leaf types; [`LogicalJoin`] composes them in the
/// brute-force baseline).
///
/// This test does NOT call into the production
/// `enumeration::bindings_in` (which is `pub(crate)`); instead it
/// pins the local helper's behavior directly. The maintenance
/// contract is: if production `bindings_in` adds handling for a
/// new variant, this helper's `match` must follow.
#[test]
fn collect_bindings_handles_strategy_leaf_types_exhaustively() {
    let scan = LogicalPlan::Scan(LogicalScan {
        label: Some(LabelId::new(1)),
        var: BindingId::new(7),
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let bs = collect_bindings(&scan);
    assert!(bs.contains(&BindingId::new(7)));
    assert_eq!(bs.len(), 1);

    let rbh = LogicalPlan::RankByHybrid(LogicalRankByHybrid {
        operands: vec![
            HybridOperand {
                kind: HybridOperandKind::Vector,
                var: BindingId::new(10),
                property: "p".to_string(),
                query: BoundExpression::Literal {
                    value: Literal::Bool(true),
                    span: span(),
                    type_info: None,
                },
                k: 5,
                read_lsn: Lsn::MAX,
                span: span(),
            },
            HybridOperand {
                kind: HybridOperandKind::Vector,
                var: BindingId::new(11),
                property: "p".to_string(),
                query: BoundExpression::Literal {
                    value: Literal::Bool(true),
                    span: span(),
                    type_info: None,
                },
                k: 5,
                read_lsn: Lsn::MAX,
                span: span(),
            },
            HybridOperand {
                kind: HybridOperandKind::Vector,
                var: BindingId::new(12),
                property: "p".to_string(),
                query: BoundExpression::Literal {
                    value: Literal::Bool(true),
                    span: span(),
                    type_info: None,
                },
                k: 5,
                read_lsn: Lsn::MAX,
                span: span(),
            },
        ],
        score_binding: None,
        fusion: None,
        span: span(),
    });
    let bs = collect_bindings(&rbh);
    assert!(bs.contains(&BindingId::new(10)));
    assert!(bs.contains(&BindingId::new(11)));
    assert!(bs.contains(&BindingId::new(12)));
    assert_eq!(bs.len(), 3);

    // Composition: Join over Scan + RankByHybrid leaves yields the
    // union of their binding sets.
    let join = LogicalPlan::Join(LogicalJoin {
        left: Box::new(scan),
        right: Box::new(rbh),
        on: JoinCondition::SharedBindings(Vec::new()),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    });
    let bs = collect_bindings(&join);
    assert!(bs.contains(&BindingId::new(7)));
    assert!(bs.contains(&BindingId::new(10)));
    assert!(bs.contains(&BindingId::new(11)));
    assert!(bs.contains(&BindingId::new(12)));
    assert_eq!(bs.len(), 4);
}

/// Shape-edge generator pin: each shape produces the expected
/// edge set, ensuring the strategy's marketing in the file-level
/// rustdoc matches what's actually generated.
#[test]
fn shape_edges_generate_expected_canonical_form() {
    // Linear at N=4: chain edges only.
    let e = shape_edges(4, JoinShape::Linear, &[]);
    assert_eq!(e, vec![(0, 1), (1, 2), (2, 3)]);

    // Star at N=4: all edges incident to leaf 0.
    let e = shape_edges(4, JoinShape::Star, &[]);
    assert_eq!(e, vec![(0, 1), (0, 2), (0, 3)]);

    // Clique at N=4: all (i, j) for i < j.
    let e = shape_edges(4, JoinShape::Clique, &[]);
    assert_eq!(e, vec![(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]);

    // T at N=4: chain plus (1, 3) branch.
    let e = shape_edges(4, JoinShape::TShape, &[]);
    assert_eq!(e, vec![(0, 1), (1, 2), (1, 3), (2, 3)]);

    // T at N=3: degenerates to chain (no spare leaf for branch).
    let e = shape_edges(3, JoinShape::TShape, &[]);
    assert_eq!(e, vec![(0, 1), (1, 2)]);

    // Mixed at N=4 with no extras: spanning tree (linear chain).
    let e = shape_edges(4, JoinShape::Mixed, &[]);
    assert_eq!(e, vec![(0, 1), (1, 2), (2, 3)]);

    // Mixed at N=4 with extra (0, 2): chain + diagonal.
    let e = shape_edges(4, JoinShape::Mixed, &[(0, 2)]);
    assert_eq!(e, vec![(0, 1), (0, 2), (1, 2), (2, 3)]);
}

/// Connectivity pin: every shape generator produces a connected
/// binding-overlap graph at every N ∈ {2..=6}.
#[test]
fn every_shape_is_connected_by_construction() {
    for n in 2..=6 {
        for shape in [
            JoinShape::Linear,
            JoinShape::Star,
            JoinShape::Clique,
            JoinShape::TShape,
            JoinShape::Mixed,
        ] {
            let edges = shape_edges(n, shape, &[]);
            assert!(
                edges_connect(n, &edges),
                "shape {:?} at N={} produced disconnected graph: {:?}",
                shape,
                n,
                edges,
            );
        }
    }
}
