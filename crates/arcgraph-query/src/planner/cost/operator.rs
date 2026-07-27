//! Per-operator cost functions for the M4-51 cost model.
//!
//! Each function takes the relevant inputs (input cardinality +
//! catalog snapshot data + per-operator parameters) and returns a
//! `(local_cost, output_card)` pair. The
//! `crate::planner::cost::walker` threads the upstream
//! `output_card` into each downstream call's `input_card`.
//!
//! # Cost-constant calibration (Prime Directive 5)
//!
//! Constants are picked from Selinger 1979 / Postgres / DuckDB
//! mainstream defaults plus per-operator complexity-class arguments
//! (linear, log-linear, quadratic). Each constant's rustdoc carries
//! the back-of-envelope justification. **Tuning at v1.0-alpha exit**
//! is informed by:
//! - The M4-04d empirical bench (issue #209) — measures real-data
//!   per-tuple processing cost across Scan / Expand / Filter /
//!   Hybrid retrieval at LDBC SNB 1M-row scale;
//! - The M4-71 row-count observer (forward) — feeds OBSERVED
//!   cardinalities back into `crate::semantic::CatalogStats` so the
//!   selectivity inputs become empirically tuned over a tenant's
//!   lifetime.
//!
//! M4-04d is being implemented in parallel; M4-51 does NOT block on
//! it. Initial values come from Selinger / Postgres precedent.
//!
//! # Cost units
//!
//! Unit-less. See [`crate::planner::cost`] module docs §"Cost units".
//! The intent is **plan-relative ordering** — a Scan-then-Filter
//! plan has a strictly smaller estimated cost than a Scan-then-Sort
//! plan over the same input, because the Filter constant
//! ([`FILTER_COST_PER_ROW`]) is smaller than the Sort
//! n*log(n) factor.
//!
//! # Output-cardinality discipline
//!
//! Each function returns `Cardinality::new(...)` — the constructor
//! saturates NaN / Inf / negative inputs to zero / `f64::MAX`. This
//! is defense-in-depth against a future formula refinement that
//! could violate the invariant.

use crate::logical_plan::{
    FusionKind, HybridOperandKind, JoinAlgorithm, JoinCondition, LogicalAggregate,
    LogicalCommunityLookup, LogicalDistinct, LogicalDynamicLimit, LogicalExpand, LogicalFilter,
    LogicalFusion, LogicalJoin, LogicalLeftOuterJoin, LogicalLimit, LogicalNamedPath,
    LogicalProject, LogicalPropertyIndexScan, LogicalRankByHybrid, LogicalScan, LogicalSkip,
    LogicalSort, LogicalTextMatch, LogicalUnwind, LogicalVectorNear, PathAlgorithm,
};
use crate::semantic::{CatalogSnapshot, DEFAULT_LABEL_SELECTIVITY, DEFAULT_REL_TYPE_SELECTIVITY};

use super::{Cardinality, Cost};

// =====================================================================
// Cost constants (back-of-envelope calibration)
// =====================================================================

/// Per-tuple cost of a label-filtered or label-free node-pattern scan.
///
/// Selinger 1979 baseline: scanning a tuple from the heap is the cost
/// unit (1.0). Postgres `seq_page_cost = 1.0` + `cpu_tuple_cost = 0.01`;
/// at v1.0 we collapse both into a single per-tuple constant since
/// our buffer-pool tuple access is not page-amortised the same way.
pub const SCAN_COST_PER_ROW: f64 = 1.0;

/// Per-traversal cost of a relationship-pattern expand. Each expand
/// touches the source endpoint's adjacency-list head (~1 buffer-pool
/// hit) and walks the type-filtered run (~2 hits at v1.0 page sizes
/// for an LDBC SNB-class adjacency degree). Postgres charges
/// ~`random_page_cost = 4.0` per index probe; we use 3.0 to match
/// our adjacency-list contiguous layout (cheaper than a B-tree probe
/// per page).
pub const EXPAND_COST_PER_ROW: f64 = 3.0;

/// Per-row cost of evaluating a filter predicate. WHERE-clause
/// expression evaluation is sub-tuple cost (~0.01–0.1 per row in
/// Postgres). Conservative pick: 0.1, dominated by the upstream
/// scan cost in any non-degenerate plan.
pub const FILTER_COST_PER_ROW: f64 = 0.1;

/// Per-row cost of a projection. Pure rebinding (extracting field
/// references); ~0.01 per row in Postgres. Cheap enough that
/// project-vs-no-project rarely changes plan shape.
pub const PROJECT_COST_PER_ROW: f64 = 0.05;

/// Per-row cost of a hash join (the natural join shape from
/// [`LogicalJoin`] with non-empty `SharedBindings`). Postgres
/// charges hash-build at ~`cpu_operator_cost ≈ 0.0025` per row;
/// our v1.0 cost lumps build + probe into a single constant
/// dominated by hash-table-touch latency. 2.0 picks up the memory-
/// hierarchy traversal from a moderately-sized hash table.
pub const HASH_JOIN_COST_PER_ROW: f64 = 2.0;

/// Per-row merge constant for a sort-merge join. Once both sides
/// arrive sorted, the merge walk is a sequential dual-cursor probe
/// — sub-tuple per-row cost (cache-friendly linear pass). Postgres
/// `cpu_operator_cost = 0.0025` per comparison; the v1.0-α merge
/// cost lumps comparison + tuple-projection at 0.5 per processed
/// row. Cheaper than hash-join per row (no hash-table touch), but
/// the sort prefix (when not free) usually dominates.
pub const MERGE_JOIN_MERGE_COST_PER_ROW: f64 = 0.5;

/// Per-row cost of a sort (multiplied by `log2(input_card)`). Cypher
/// 9 §6.6 ORDER BY semantics; Postgres `cpu_operator_cost = 0.0025`
/// per comparison + `log2(N)` factor. We lump comparison + tuple-move
/// into 0.5 per (row × log2(row)).
pub const SORT_COST_PER_ROW_LOG: f64 = 0.5;

/// Per-row cost of LIMIT / SKIP / DynamicLimit. Trivial counter
/// increment + early-exit; minimum cost in the model.
pub const LIMIT_COST_PER_ROW: f64 = 0.01;

/// Per-row cost of DISTINCT. Hash-set-touch + tuple-equality;
/// dominated by the hash-set ops. Roughly hash-join cost minus
/// the probe phase.
pub const DISTINCT_COST_PER_ROW: f64 = 1.5;

/// Per-row cost of UNWIND. List-iteration is cheap; cost is
/// proportional to elements emitted, not to the input rows
/// themselves.
pub const UNWIND_COST_PER_ELEMENT: f64 = 0.05;

/// Per-row cost of an aggregation hash-build. Hash-key + merge
/// per group; same complexity class as a hash-join build phase.
pub const AGGREGATE_COST_PER_ROW: f64 = 1.5;

/// Per-row cost of an RRF fusion combine step over candidate
/// rankings. ADR-036 §D-9 RRF fusion is `O(K log K)` for top-K;
/// we charge a flat per-input cost since K is bounded by each
/// operand's K cap.
pub const FUSION_COST_PER_ROW: f64 = 0.2;

/// Per-row cost of a community-membership lookup. ADR-040 §D-3
/// community-index handle is a hash lookup keyed by `(TenantId,
/// Level, NodeId)` — same complexity class as a B-tree point
/// lookup.
pub const COMMUNITY_LOOKUP_COST_PER_ROW: f64 = 1.0;

/// **#1366 (Phase 2).** Per-lookup cost of an indexed property point
/// lookup. A `SecondaryIndex` B+tree point lookup is `O(log N)` +
/// hydrate + verify of a tiny candidate set — the SAME complexity class
/// as the community hash lookup (`COMMUNITY_LOOKUP_COST_PER_ROW = 1.0`),
/// and DRAMATICALLY below `SCAN_COST_PER_ROW × label_card` (the anchor
/// scan it replaces). Set equal to the community-lookup constant so the
/// planner always prefers the index over the scan whenever an Online
/// index exists (design §"Cost model hook"). The output cardinality is
/// carried separately (unique index ⇒ 1; see [`cost_property_index_scan`]).
pub const PROPERTY_INDEX_POINT_LOOKUP_COST: f64 = COMMUNITY_LOOKUP_COST_PER_ROW;

/// Per-K cost of a vector ANN retrieval. Per ADR-035 D-7 the
/// HNSW search is `O(K log N)`; we lump the log-N factor into the
/// constant since at v1.0 tenant sizes (≤ 1B vectors) the log
/// factor is ~30.
pub const VECTOR_NEAR_COST_PER_K: f64 = 30.0;

/// Per-K cost of a BM25 text retrieval. Per ADR-039 §D-3 Tantivy
/// BM25 search is `O(K log N)`; same lumped-constant rationale as
/// vector. Slightly cheaper because Tantivy's posting-list
/// iteration is more cache-friendly than HNSW graph traversal.
pub const TEXT_MATCH_COST_PER_K: f64 = 20.0;

/// Per-K cost of an SSSP-shaped shortest-path traversal. Cypher 9
/// §6.5 SHORTEST_PATH is `O(V + E)` worst case; for v1.0 LDBC SNB-
/// class graphs the average expanded-fringe size is the dominant
/// term. 5.0 captures a few-hop average traversal.
pub const SHORTEST_PATH_COST_PER_NODE: f64 = 5.0;

// =====================================================================
// Per-operator cost functions
// =====================================================================

/// Cost for [`LogicalScan`].
///
/// Estimated cost: `label_card * SCAN_COST_PER_ROW` (label-filtered)
/// or `total_nodes * SCAN_COST_PER_ROW` (label-free).
///
/// Output cardinality:
/// - `label_card` if the snapshot has the label observed;
/// - `total_nodes * DEFAULT_LABEL_SELECTIVITY` if the label was not
///   observed but totals are present (cold-start; bias toward
///   index-aware plans);
/// - `0.0` if `total_nodes` is `Some(0)` (observed-then-deleted);
/// - a fallback constant if no stats are present.
///
/// The fallback constant is `1000.0` — a "typical small tenant"
/// guess that lets the cost-model produce comparable plan shapes
/// for fresh tenants without forcing the planner to short-circuit.
pub const FALLBACK_TENANT_NODE_COUNT: f64 = 1000.0;

/// Per-rel-type fallback for [`cost_expand`] when total_rels is
/// missing. Symmetric with [`FALLBACK_TENANT_NODE_COUNT`].
#[allow(dead_code)]
pub const FALLBACK_TENANT_REL_COUNT: f64 = 5000.0;

/// Per-row average outgoing degree fallback for [`cost_expand`]
/// when the catalog has no rel-type stats. LDBC SNB Person-KNOWS
/// graph has avg degree ~50; v1.0 picks the median-graph estimate
/// of 5 (closer to common transactional graph fan-out).
const FALLBACK_AVG_DEGREE: f64 = 5.0;

#[must_use]
pub fn cost_scan(scan: &LogicalScan, snapshot: &CatalogSnapshot) -> (Cost, Cardinality) {
    let card = match scan.label {
        Some(label) => match (snapshot.label_card(label), snapshot.total_nodes()) {
            (Some(c), _) => c as f64,
            (None, Some(0)) => 0.0,
            (None, Some(t)) => t as f64 * DEFAULT_LABEL_SELECTIVITY,
            (None, None) => FALLBACK_TENANT_NODE_COUNT * DEFAULT_LABEL_SELECTIVITY,
        },
        None => match snapshot.total_nodes() {
            Some(t) => t as f64,
            None => FALLBACK_TENANT_NODE_COUNT,
        },
    };
    let local_cost = Cost::new(card * SCAN_COST_PER_ROW);
    (local_cost, Cardinality::new(card))
}

/// **#1366 (Phase 2).** Cost for [`LogicalPropertyIndexScan`].
///
/// A B+tree point lookup + hydrate + verify — a flat
/// [`PROPERTY_INDEX_POINT_LOOKUP_COST`] per lookup (independent of
/// `label_card`, unlike [`cost_scan`]). Output cardinality is `1` for an
/// exact equality on a property index: RC-MVP indexes are declared
/// single-property and the planner only routes an exact-equality here,
/// so the estimate is one node (unique-ish). If the index has known
/// duplicate-per-value stats in a future slice this can widen, but `1`
/// is the correct RC estimate for a point lookup and keeps the index
/// path decisively below the anchor scan. Duplicate-value corpora still
/// return `O(matches)` rows at runtime; the cost estimate is a planner
/// hint, not a runtime cap.
#[must_use]
pub fn cost_property_index_scan(_lookup: &LogicalPropertyIndexScan) -> (Cost, Cardinality) {
    (
        Cost::new(PROPERTY_INDEX_POINT_LOOKUP_COST),
        Cardinality::new(1.0),
    )
}

/// Cost for [`LogicalExpand`].
///
/// Estimated cost (two-term, matches the formula at the assignment
/// site below): `input * EXPAND_COST_PER_ROW + output * EXPAND_COST_PER_ROW * 0.1`.
/// The first term charges per-input-row adjacency-list head touch; the
/// second amortizes the per-output-row materialization at 10% of the
/// per-input cost (Postgres-style index probe + tuple-build split).
///
/// Where:
/// - `input` is `input_card.rows()`.
/// - `output = input * avg_degree * rel_type_selectivity * length_factor`
///   (the per-tuple expansion factor; `avg_degree` is derived from
///   `total_rels / total_nodes` or the fallback constant; the
///   selectivity / length factor account for typed expansion + variable
///   length matching).
///
/// Output cardinality: `output` per the formula above (× rel-type
/// selectivity if a type filter is present, × length-range midpoint if
/// variable-length).
#[must_use]
pub fn cost_expand(
    expand: &LogicalExpand,
    input_card: Cardinality,
    snapshot: &CatalogSnapshot,
) -> (Cost, Cardinality) {
    let input = input_card.rows();

    // Average outgoing degree — proxy = total_rels / total_nodes.
    // Falls back to FALLBACK_AVG_DEGREE if either total is missing.
    let avg_degree = match (snapshot.total_rels(), snapshot.total_nodes()) {
        (Some(r), Some(n)) if n > 0 => r as f64 / n as f64,
        _ => FALLBACK_AVG_DEGREE,
    };

    // Apply rel-type selectivity if the expand carries a type filter.
    let rel_type_selectivity = match expand.rel_type {
        Some(rt) => {
            let card = snapshot.rel_type_card(rt);
            let total = snapshot.total_rels();
            match (card, total) {
                (Some(_), Some(0)) => 0.0,
                (Some(c), Some(t)) if t > 0 => (c as f64 / t as f64).clamp(0.0, 1.0),
                (Some(_), _) | (None, _) => DEFAULT_REL_TYPE_SELECTIVITY,
            }
        }
        None => 1.0,
    };

    // Variable-length expansion (`*N..M`) inflates avg_degree by the
    // length range's expected hop count — k-1 expansion at LDBC SNB
    // IC1 is bounded by `~50` per ADR-036 §D-7 acceptance, but v1.0
    // estimator uses a conservative midpoint: max(1, midpoint(min, max)).
    //
    // Default-hop fallback when max is unbounded (`*N..` or `*`):
    // assume a 5-hop midpoint; v1.0 LDBC SNB IC1-class queries
    // typically cap at 3–5 hops in practice.
    const DEFAULT_UNBOUNDED_MAX_HOPS: u32 = 5;
    let length_factor = match &expand.length_range {
        Some(crate::ast::LengthRange::Unbounded) => DEFAULT_UNBOUNDED_MAX_HOPS as f64,
        Some(crate::ast::LengthRange::Cypher { min, max })
        | Some(crate::ast::LengthRange::Quantified { min, max }) => {
            let max_hops = max.unwrap_or(DEFAULT_UNBOUNDED_MAX_HOPS).max(*min);
            let min = (*min).max(1) as f64;
            let max = max_hops as f64;
            // Midpoint approximation. v1.1 (M4-04c sketches) can
            // refine using path-length distribution sketches.
            ((min + max) / 2.0).max(1.0)
        }
        None => 1.0,
    };

    let expanded_per_input = avg_degree * rel_type_selectivity * length_factor;
    let output = (input * expanded_per_input).max(0.0);
    let local_cost = Cost::new(input * EXPAND_COST_PER_ROW + output * EXPAND_COST_PER_ROW * 0.1);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalFilter`].
///
/// Estimated cost: `input_card * FILTER_COST_PER_ROW` (every input
/// row pays predicate-evaluation cost regardless of selectivity).
///
/// Output cardinality: `input_card * predicate_selectivity`. The
/// `predicate_selectivity` is computed by the
/// [`crate::planner::cost::predicate`] walker via
/// [`crate::planner::cost::composition`].
#[must_use]
pub fn cost_filter(
    _filter: &LogicalFilter,
    input_card: Cardinality,
    predicate_selectivity: f64,
) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let selectivity = predicate_selectivity.clamp(0.0, 1.0);
    let output = input * selectivity;
    let local_cost = Cost::new(input * FILTER_COST_PER_ROW);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalProject`].
///
/// Estimated cost: `input_card * PROJECT_COST_PER_ROW`. Projection
/// preserves cardinality (per-row rebinding); v1.0 ignores
/// width-aware cost.
#[must_use]
pub fn cost_project(_project: &LogicalProject, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let local_cost = Cost::new(input * PROJECT_COST_PER_ROW);
    (local_cost, input_card)
}

/// Cost for [`LogicalJoin`] (equi-join from multi-pattern MATCH).
///
/// Per ADR-097 (W25-M4-61b), the cost dispatches on the join's
/// `algorithm` field:
/// - [`JoinAlgorithm::HashJoin`] → [`cost_hash_join`] — `(L + R) ·
///   HASH_JOIN_COST_PER_ROW`.
/// - [`JoinAlgorithm::MergeJoin`] → [`cost_merge_join`] —
///   `sort(L) + sort(R) + (L + R) · MERGE_JOIN_MERGE_COST_PER_ROW`.
/// - [`JoinAlgorithm::Auto`] → returns `min(hash_cost, merge_cost)`.
///   The [`crate::planner::pick_join_algorithms`] pass rewrites
///   `Auto` → concrete variant after costing.
///
/// Output cardinality is INDEPENDENT of algorithm choice — it is a
/// logical property of the join shape:
/// - **`SharedBindings([])`** (Cartesian) — `left * right`.
/// - **`SharedBindings(non-empty)`** (equi-join) — at v1.0 we apply
///   a moderate join selectivity proxy: `(left * right) / max(left,
///   right)`. This gives `min(left, right)` for the perfect-equi-
///   join case (1:1 match) and tracks a v1.0 conservative estimate
///   for less selective joins. Without per-binding cardinality
///   estimates (deferred to v1.1), this is the best v1.0 proxy.
#[must_use]
pub fn cost_join(
    join: &LogicalJoin,
    left_card: Cardinality,
    right_card: Cardinality,
) -> (Cost, Cardinality) {
    let l = left_card.rows();
    let r = right_card.rows();

    let local_cost = match join.algorithm {
        JoinAlgorithm::HashJoin => Cost::new(cost_hash_join(l, r)),
        JoinAlgorithm::MergeJoin => Cost::new(cost_merge_join(l, r, &join.on)),
        JoinAlgorithm::Auto => {
            let h = cost_hash_join(l, r);
            let m = cost_merge_join(l, r, &join.on);
            // Cartesian (empty SharedBindings) cannot run as merge —
            // cost_merge_join returns +∞ for it; min() collapses to
            // hash regardless. The picker also enforces this rule.
            Cost::new(h.min(m))
        }
    };

    let output = match &join.on {
        // Cartesian: left × right.
        JoinCondition::SharedBindings(ids) if ids.is_empty() => l * r,
        // Equi-join: assume the joined-on bindings cap the output at
        // the larger side's cardinality (perfect 1:1 match yields
        // `min(l, r)`; we use `max(l, r)` as a conservative upper
        // bound to reflect non-uniform bucket distribution).
        JoinCondition::SharedBindings(_) => {
            if l == 0.0 || r == 0.0 {
                0.0
            } else {
                (l * r) / l.max(r)
            }
        }
    };
    (local_cost, Cardinality::new(output))
}

/// Hash-join cost: every left + right row pays the build/probe
/// constant. Cheap when one side is small (the BUILD bucket fits
/// the per-tenant byte budget); independent of input sort order.
#[must_use]
pub fn cost_hash_join(left: f64, right: f64) -> f64 {
    (left + right) * HASH_JOIN_COST_PER_ROW
}

/// Merge-join cost: `sort(L) + sort(R) + (L + R) · merge_const`.
///
/// At v1.0-α we conservatively assume neither input is pre-sorted on
/// the join keys (the ScanOp emits storage order, not key order); the
/// sort terms are `N · log2(N) · SORT_COST_PER_ROW_LOG` per side.
/// When M4-71 / M4-72 forward layers expose "sort-property"
/// annotations (e.g., "this Scan produces rows sorted by `n.id`
/// because the underlying storage is row-sorted on the primary key"),
/// the per-side sort term collapses to zero and merge-join becomes
/// strictly cheaper than hash-join.
///
/// Cartesian (`SharedBindings([])`) returns `f64::MAX` to suppress
/// merge-join from `Auto` selection — merge-join is undefined without
/// join keys (no walk-comparison key exists).
#[must_use]
pub fn cost_merge_join(left: f64, right: f64, on: &JoinCondition) -> f64 {
    let JoinCondition::SharedBindings(keys) = on;
    if keys.is_empty() {
        // No join keys → merge-join is structurally inapplicable.
        return f64::MAX;
    }
    let sort_left = if left > 1.0 {
        left * left.log2() * SORT_COST_PER_ROW_LOG
    } else {
        0.0
    };
    let sort_right = if right > 1.0 {
        right * right.log2() * SORT_COST_PER_ROW_LOG
    } else {
        0.0
    };
    let merge = (left + right) * MERGE_JOIN_MERGE_COST_PER_ROW;
    sort_left + sort_right + merge
}

/// Cost for [`LogicalLeftOuterJoin`] (OPTIONAL MATCH lowering).
///
/// Estimated cost: same shape as [`cost_join`]; the left-outer
/// semantics affect output cardinality, NOT the join cost.
///
/// Output cardinality: same as inner-equi-join, plus the un-matched
/// left rows that produce NULL fills. v1.0 lower bound: at least
/// `left_card` rows (every left row contributes at least once).
#[must_use]
pub fn cost_left_outer_join(
    join: &LogicalLeftOuterJoin,
    left_card: Cardinality,
    right_card: Cardinality,
) -> (Cost, Cardinality) {
    let l = left_card.rows();
    let r = right_card.rows();
    let local_cost = Cost::new((l + r) * HASH_JOIN_COST_PER_ROW);

    // Left-outer join produces ≥ left_card rows (every left row
    // matches at least once, with NULL fills for unmatched rows).
    let inner_estimate = match &join.on {
        JoinCondition::SharedBindings(ids) if ids.is_empty() => l * r,
        JoinCondition::SharedBindings(_) => {
            if l == 0.0 {
                0.0
            } else if r == 0.0 {
                l
            } else {
                ((l * r) / l.max(r)).max(l)
            }
        }
    };
    let output = inner_estimate.max(l);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalLimit`]. Output cardinality is `min(input,
/// count)`. Cost is per-row constant (counter increment).
#[must_use]
pub fn cost_limit(limit: &LogicalLimit, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let count = limit.count as f64;
    let output = input.min(count);
    let local_cost = Cost::new(output * LIMIT_COST_PER_ROW);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalSkip`]. Output cardinality is `max(0, input -
/// count)`. Cost is per-row constant (counter increment).
#[must_use]
pub fn cost_skip(skip: &LogicalSkip, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let count = skip.count as f64;
    let output = (input - count).max(0.0);
    let local_cost = Cost::new(input * LIMIT_COST_PER_ROW);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalDynamicLimit`]. Without runtime evaluation of
/// the count expression, we estimate via the input cardinality —
/// `min(input, input * 0.5)` for LIMIT (assume 50% trim) and
/// `max(0, input - input * 0.5)` for SKIP. Cost is per-row constant.
///
/// **v1.1 forward-link.** When parameter binding lands at execute
/// time, the planner's parameter-aware cost-model swap can read the
/// actual integer and produce a precise estimate. Today it's a
/// conservative midpoint.
#[must_use]
pub fn cost_dynamic_limit(
    dyn_limit: &LogicalDynamicLimit,
    input_card: Cardinality,
) -> (Cost, Cardinality) {
    use crate::logical_plan::DynamicLimitKind;
    let input = input_card.rows();
    let half = input * 0.5;
    let output = match dyn_limit.kind {
        DynamicLimitKind::Limit => input.min(half),
        DynamicLimitKind::Skip => (input - half).max(0.0),
    };
    let local_cost = Cost::new(input * LIMIT_COST_PER_ROW);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalSort`]. `n * log2(n) * SORT_COST_PER_ROW_LOG`,
/// preserves cardinality.
#[must_use]
pub fn cost_sort(_sort: &LogicalSort, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    // log2 saturates to 0 at input ≤ 1.0 (no comparisons needed).
    let log_factor = if input > 1.0 { input.log2() } else { 0.0 };
    let local_cost = Cost::new(input * log_factor * SORT_COST_PER_ROW_LOG);
    (local_cost, input_card)
}

/// Cost for [`LogicalDistinct`]. Hash-set-touch per row, output is
/// `input * 0.7` (assume 30% duplicates by default; a conservative
/// midpoint until M4-71 feedback refines).
///
/// **v1.1 forward-link.** Per-binding NDV (number-of-distinct-values)
/// estimates would refine this; deferred to M4-04c sketches.
#[must_use]
pub fn cost_distinct(_distinct: &LogicalDistinct, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let local_cost = Cost::new(input * DISTINCT_COST_PER_ROW);
    let output = input * 0.7;
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalUnwind`]. Output cardinality is `input *
/// avg_list_len`. Without per-list NDV, v1.0 uses a midpoint
/// estimate of 5 elements per list.
const FALLBACK_AVG_LIST_LEN: f64 = 5.0;

#[must_use]
pub fn cost_unwind(_unwind: &LogicalUnwind, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let avg_list_len = FALLBACK_AVG_LIST_LEN;
    let output = input * avg_list_len;
    let local_cost = Cost::new(output * UNWIND_COST_PER_ELEMENT);
    (local_cost, Cardinality::new(output))
}

/// **ADR-197 (#802)** — cost for [`crate::logical_plan::LogicalProcedureCall`].
/// Schema-introspection procedures + SHOW return a SMALL, catalog-sized
/// rowset (labels / rel-types / property-keys / a default-db row);
/// model a fixed small output independent of the (unit-row) input. The
/// op is never on a join's hot path, so a flat estimate suffices.
pub fn cost_procedure_call(_input_card: Cardinality) -> (Cost, Cardinality) {
    // A conservative small fixed output (catalog-sized). The exact
    // value is non-critical — the op is a leading generator, not a
    // join operand the DP enumerator reorders.
    const PROCEDURE_OUTPUT_ROWS: f64 = 16.0;
    let local_cost = Cost::new(PROCEDURE_OUTPUT_ROWS);
    (local_cost, Cardinality::new(PROCEDURE_OUTPUT_ROWS))
}

/// Cost for [`LogicalAggregate`]. Hash-build cost plus per-group
/// output. Output cardinality:
/// - empty `group_by` → 1 row (single-row aggregate per Cypher 9
///   §6.4);
/// - non-empty `group_by` → estimated NDV of the group-by key,
///   capped at input cardinality. v1.0 estimate: `min(input,
///   input * 0.5)` (assume ~50% group-by NDV ratio; midpoint).
#[must_use]
pub fn cost_aggregate(aggr: &LogicalAggregate, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let local_cost = Cost::new(input * AGGREGATE_COST_PER_ROW);
    let output = if aggr.group_by.is_empty() {
        1.0
    } else {
        (input * 0.5).min(input).max(1.0)
    };
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalRankByHybrid`].
///
/// Estimated cost: sum of per-operand retrieval costs (vector +
/// text + ...) plus the fusion combine step. Output cardinality:
/// the largest operand's K (RRF preserves the top-K of the unioned
/// hits per ADR-036 §D-9).
#[must_use]
pub fn cost_rank_by_hybrid(rank: &LogicalRankByHybrid) -> (Cost, Cardinality) {
    let mut total_cost = 0.0;
    let mut max_k = 0u64;

    for operand in &rank.operands {
        let per_operand = match operand.kind {
            HybridOperandKind::Vector => operand.k as f64 * VECTOR_NEAR_COST_PER_K,
            HybridOperandKind::Text => operand.k as f64 * TEXT_MATCH_COST_PER_K,
        };
        total_cost += per_operand;
        max_k = max_k.max(operand.k);
    }
    // Fusion step — ADR-036 §D-9 RRF over union of operand hits.
    let n_operands = rank.operands.len() as f64;
    total_cost += max_k as f64 * n_operands * FUSION_COST_PER_ROW;

    let local_cost = Cost::new(total_cost);
    (local_cost, Cardinality::new(max_k as f64))
}

/// Cost for [`LogicalFusion`] (top-level RRF over input candidate
/// rankings). Linear in input size + log K for the heap merge.
#[must_use]
pub fn cost_fusion(fusion: &LogicalFusion, input_cards: &[Cardinality]) -> (Cost, Cardinality) {
    let total_input: f64 = input_cards.iter().map(|c| c.rows()).sum();
    let local_cost = match fusion.spec.kind {
        FusionKind::Rrf => Cost::new(total_input * FUSION_COST_PER_ROW),
    };
    // Output is bounded by the union of inputs (RRF preserves top-K).
    let output = total_input.max(0.0);
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalCommunityLookup`]. Per-row hash lookup against
/// the community-index handle.
///
/// Output cardinality: input × community-membership-selectivity
/// (rough v1.0 estimate of 0.1 — most communities cover ~10% of a
/// tenant per LDBC SNB ground-truth distributions; refine via
/// M4-04c sketches). Until per-community NDV stats land, the v1.0
/// estimator uses a midpoint constant.
const FALLBACK_COMMUNITY_SELECTIVITY: f64 = 0.1;

#[must_use]
pub fn cost_community_lookup(
    _lookup: &LogicalCommunityLookup,
    input_card: Cardinality,
) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let local_cost = Cost::new(input * COMMUNITY_LOOKUP_COST_PER_ROW);
    let output = input * FALLBACK_COMMUNITY_SELECTIVITY;
    (local_cost, Cardinality::new(output))
}

/// Cost for [`LogicalVectorNear`]. K hits at `VECTOR_NEAR_COST_PER_K`.
/// Output cardinality is K (or 0 if K is unspecified — bare-expression
/// surface).
#[must_use]
pub fn cost_vector_near(near: &LogicalVectorNear) -> (Cost, Cardinality) {
    let k = near.k as f64;
    let local_cost = Cost::new(k * VECTOR_NEAR_COST_PER_K);
    (local_cost, Cardinality::new(k))
}

/// Cost for [`LogicalTextMatch`]. K hits at `TEXT_MATCH_COST_PER_K`.
/// Output cardinality is K (or a fallback constant if K is `None`).
const FALLBACK_TEXT_MATCH_K: f64 = 100.0;

#[must_use]
pub fn cost_text_match(text: &LogicalTextMatch) -> (Cost, Cardinality) {
    let k = text.k.map_or(FALLBACK_TEXT_MATCH_K, |k| k as f64);
    let local_cost = Cost::new(k * TEXT_MATCH_COST_PER_K);
    (local_cost, Cardinality::new(k))
}

/// Cost for [`LogicalNamedPath`]. Plain enumeration is bounded by
/// the underlying scan + expand chain; SHORTEST_PATH adds a
/// per-node BFS-frontier cost.
#[must_use]
pub fn cost_named_path(np: &LogicalNamedPath, input_card: Cardinality) -> (Cost, Cardinality) {
    let input = input_card.rows();
    let local_cost = match np.algorithm {
        // Plain path enumeration: cost dominated by the input subtree.
        // Per-row materialization adds a small constant (path packaging).
        PathAlgorithm::Plain => Cost::new(input * 0.1),
        // SHORTEST_PATH / shortestPath: BFS frontier; per-node cost.
        PathAlgorithm::ShortestPath => Cost::new(input * SHORTEST_PATH_COST_PER_NODE),
        // allShortestPaths: same per-node BFS frontier as ShortestPath
        // (the frontier expansion dominates); enumerating ALL equal-min
        // meeting paths adds bounded extra work. ADR-194 OQ-194-2 defers a
        // refined multi-path cardinality/cost estimate to post-LDBC — this
        // placeholder reuses the ShortestPath per-node cost.
        PathAlgorithm::AllShortestPaths => Cost::new(input * SHORTEST_PATH_COST_PER_NODE),
    };
    // Path enumeration preserves cardinality (one path emitted per
    // input row). SHORTEST_PATH may produce ≤ input rows (paths that
    // don't reach the target are dropped); allShortestPaths may produce ≥
    // input rows (multiple equal-min paths per (src,target) pair —
    // ADR-194 OQ-194-2 rough estimate). v1.0/GA uses input for all.
    (local_cost, input_card)
}

/// Cost for [`crate::logical_plan::LogicalPlan::Empty`] (degenerate
/// empty-clauses sentinel). Zero cost, zero rows.
#[must_use]
pub fn cost_empty() -> (Cost, Cardinality) {
    (Cost::zero(), Cardinality::zero())
}

// =====================================================================
// Hybrid retrieval helpers
// =====================================================================

/// Helper: an aggregation over an empty input still produces 1 row
/// for the single-row-aggregate case (Cypher 9 §6.4) — exposed so
/// the test suite can pin the boundary.
#[must_use]
pub fn aggregate_output_for_empty_input(aggr: &LogicalAggregate) -> Cardinality {
    if aggr.group_by.is_empty() {
        Cardinality::new(1.0)
    } else {
        Cardinality::zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{LengthRange, Literal};
    use crate::error::Span;
    use crate::logical_plan::types::*;
    use crate::semantic::CatalogSnapshot;
    use crate::semantic::bound_ast::{BindingId, BoundExpression};
    use arcgraph_core::{LabelId, Lsn};

    fn span() -> Span {
        Span::point(1, 1)
    }

    fn snapshot(
        total_nodes: u64,
        total_rels: u64,
        label_card: Option<(LabelId, u64)>,
    ) -> CatalogSnapshot {
        let label_cards = label_card.map(|p| vec![p]).unwrap_or_default();
        CatalogSnapshot::from_parts(
            Some(total_nodes),
            Some(total_rels),
            label_cards,
            Vec::new(),
            0,
        )
    }

    #[test]
    fn scan_with_known_label_uses_label_card() {
        let snap = snapshot(1_000, 5_000, Some((LabelId::new(1), 200)));
        let scan = LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let (cost, card) = cost_scan(&scan, &snap);
        assert_eq!(card.rows(), 200.0);
        assert_eq!(cost.total(), 200.0 * SCAN_COST_PER_ROW);
    }

    #[test]
    fn scan_without_label_uses_total_nodes() {
        let snap = snapshot(1_000, 5_000, None);
        let scan = LogicalScan {
            label: None,
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let (cost, card) = cost_scan(&scan, &snap);
        assert_eq!(card.rows(), 1_000.0);
        assert_eq!(cost.total(), 1_000.0 * SCAN_COST_PER_ROW);
    }

    #[test]
    fn scan_falls_back_to_constant_when_stats_empty() {
        let snap = CatalogSnapshot::empty();
        let scan = LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let (_, card) = cost_scan(&scan, &snap);
        // No stats: fallback × DEFAULT_LABEL_SELECTIVITY.
        assert!(
            (card.rows() - FALLBACK_TENANT_NODE_COUNT * DEFAULT_LABEL_SELECTIVITY).abs() < 1e-9
        );
    }

    #[test]
    fn expand_uses_avg_degree_and_rel_type_filter() {
        let snap = snapshot(1_000, 5_000, None);
        // avg_degree = 5_000 / 1_000 = 5.0
        let expand = LogicalExpand {
            from: BindingId::new(0),
            to: BindingId::new(1),
            direction: Direction::LeftToRight,
            rel_type: None,
            length_range: None,
            rel_var: None,
            span: span(),
        };
        let (_, card) = cost_expand(&expand, Cardinality::new(100.0), &snap);
        // 100 input × 5 avg_degree × 1.0 (no type filter) = 500.0
        assert_eq!(card.rows(), 500.0);
    }

    #[test]
    fn filter_applies_predicate_selectivity() {
        let filter = LogicalFilter {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            predicate: BoundExpression::Literal {
                value: Literal::Bool(true),
                span: span(),
                type_info: None,
            },
            span: span(),
        };
        let (cost, card) = cost_filter(&filter, Cardinality::new(1_000.0), 0.25);
        assert_eq!(card.rows(), 250.0);
        assert_eq!(cost.total(), 1_000.0 * FILTER_COST_PER_ROW);
    }

    #[test]
    fn project_preserves_cardinality_and_costs_linear() {
        let project = LogicalProject {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            items: Vec::new(),
            span: span(),
        };
        let (cost, card) = cost_project(&project, Cardinality::new(123.0));
        assert_eq!(card.rows(), 123.0);
        assert_eq!(cost.total(), 123.0 * PROJECT_COST_PER_ROW);
    }

    #[test]
    fn join_cartesian_yields_left_times_right() {
        let join = LogicalJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(Vec::new()),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        };
        let (_, card) = cost_join(&join, Cardinality::new(10.0), Cardinality::new(20.0));
        assert_eq!(card.rows(), 200.0);
    }

    #[test]
    fn join_equi_estimates_yield_max_side() {
        let join = LogicalJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        };
        let (_, card) = cost_join(&join, Cardinality::new(100.0), Cardinality::new(50.0));
        // (100 * 50) / 100 = 50
        assert_eq!(card.rows(), 50.0);
    }

    #[test]
    fn cost_hash_join_is_linear_in_total_input() {
        assert_eq!(cost_hash_join(100.0, 50.0), 150.0 * HASH_JOIN_COST_PER_ROW);
        assert_eq!(cost_hash_join(0.0, 0.0), 0.0);
    }

    #[test]
    fn cost_merge_join_includes_sort_and_merge_terms() {
        // 100 + 50 inputs, equi-join on one binding.
        let on = JoinCondition::SharedBindings(vec![BindingId::new(0)]);
        let cost = cost_merge_join(100.0, 50.0, &on);
        // sort_left = 100 * log2(100) * 0.5 ≈ 100 * 6.643 * 0.5 ≈ 332.2
        // sort_right = 50 * log2(50) * 0.5 ≈ 50 * 5.644 * 0.5 ≈ 141.1
        // merge = (100 + 50) * 0.5 = 75
        let expected = 100.0 * 100.0_f64.log2() * 0.5
            + 50.0 * 50.0_f64.log2() * 0.5
            + (100.0 + 50.0) * MERGE_JOIN_MERGE_COST_PER_ROW;
        assert!((cost - expected).abs() < 1e-6);
    }

    #[test]
    fn cost_merge_join_cartesian_is_max_f64() {
        let on = JoinCondition::SharedBindings(Vec::new());
        // Cartesian: merge-join undefined → +∞ sentinel so Auto picks hash.
        assert_eq!(cost_merge_join(100.0, 50.0, &on), f64::MAX);
    }

    #[test]
    fn cost_join_auto_picks_cheaper_of_hash_or_merge() {
        // Large equal-sized inputs: hash beats merge because the
        // n*log2(n) sort cost dominates.
        let join = LogicalJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        };
        let (cost_auto, _) = cost_join(&join, Cardinality::new(1_000.0), Cardinality::new(1_000.0));
        let hash_only = cost_hash_join(1_000.0, 1_000.0);
        let merge_only = cost_merge_join(1_000.0, 1_000.0, &join.on);
        let expected = hash_only.min(merge_only);
        assert!((cost_auto.total() - expected).abs() < 1e-6);
        // hash < merge under the v1.0-α constants for large equal sides.
        assert!(cost_auto.total() <= hash_only);
    }

    #[test]
    fn cost_join_pinned_to_hash_when_algorithm_is_hash() {
        let join = LogicalJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            algorithm: JoinAlgorithm::HashJoin,
            span: span(),
        };
        let (cost, _) = cost_join(&join, Cardinality::new(100.0), Cardinality::new(50.0));
        assert_eq!(cost.total(), cost_hash_join(100.0, 50.0));
    }

    #[test]
    fn cost_join_pinned_to_merge_when_algorithm_is_merge() {
        let join = LogicalJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            algorithm: JoinAlgorithm::MergeJoin,
            span: span(),
        };
        let (cost, _) = cost_join(&join, Cardinality::new(100.0), Cardinality::new(50.0));
        assert_eq!(cost.total(), cost_merge_join(100.0, 50.0, &join.on));
    }

    #[test]
    fn cost_join_cartesian_forces_hash_under_auto() {
        // Cartesian + Auto → merge is +∞, so hash wins by min().
        let join = LogicalJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(Vec::new()),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        };
        let (cost, _) = cost_join(&join, Cardinality::new(100.0), Cardinality::new(50.0));
        assert_eq!(cost.total(), cost_hash_join(100.0, 50.0));
    }

    #[test]
    fn left_outer_join_preserves_left_card_lower_bound() {
        let join = LogicalLeftOuterJoin {
            left: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            right: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            span: span(),
        };
        // Right is empty → all left rows produce NULL fills; output ≥ left.
        let (_, card) = cost_left_outer_join(&join, Cardinality::new(50.0), Cardinality::zero());
        assert_eq!(card.rows(), 50.0);
    }

    #[test]
    fn limit_caps_output_at_count() {
        let limit = LogicalLimit {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            count: 10,
            span: span(),
        };
        let (_, card) = cost_limit(&limit, Cardinality::new(1_000.0));
        assert_eq!(card.rows(), 10.0);
        // Input < count → output = input.
        let (_, card) = cost_limit(&limit, Cardinality::new(5.0));
        assert_eq!(card.rows(), 5.0);
    }

    #[test]
    fn sort_n_log_n_cost_with_log_zero_at_n_le_1() {
        let sort = LogicalSort {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            order_by: Vec::new(),
            span: span(),
        };
        let (cost, card) = cost_sort(&sort, Cardinality::new(1_000.0));
        // 1000 * log2(1000) ≈ 1000 * 9.97 ≈ 9966
        let expected = 1_000.0 * 1_000.0_f64.log2() * SORT_COST_PER_ROW_LOG;
        assert!((cost.total() - expected).abs() < 1e-6);
        assert_eq!(card.rows(), 1_000.0);

        // Trivial input case — no comparisons needed.
        let (cost, _) = cost_sort(&sort, Cardinality::new(1.0));
        assert_eq!(cost.total(), 0.0);
    }

    #[test]
    fn rank_by_hybrid_aggregates_per_operand_costs() {
        let rank = LogicalRankByHybrid {
            operands: vec![
                HybridOperand {
                    kind: HybridOperandKind::Vector,
                    var: BindingId::new(0),
                    property: "embedding".into(),
                    query: BoundExpression::Parameter {
                        name: "q".into(),
                        span: span(),
                        type_info: None,
                    },
                    k: 10,
                    read_lsn: Lsn::MAX,
                    span: span(),
                },
                HybridOperand {
                    kind: HybridOperandKind::Text,
                    var: BindingId::new(0),
                    property: "content".into(),
                    query: BoundExpression::Parameter {
                        name: "q".into(),
                        span: span(),
                        type_info: None,
                    },
                    k: 20,
                    read_lsn: Lsn::MAX,
                    span: span(),
                },
            ],
            score_binding: None,
            fusion: None,
            span: span(),
        };
        let (cost, card) = cost_rank_by_hybrid(&rank);
        // Vector: 10 * 30 = 300; Text: 20 * 20 = 400; Fusion: 20 * 2 * 0.2 = 8.
        let expected = 10.0 * VECTOR_NEAR_COST_PER_K
            + 20.0 * TEXT_MATCH_COST_PER_K
            + 20.0 * 2.0 * FUSION_COST_PER_ROW;
        assert!((cost.total() - expected).abs() < 1e-9);
        // Output is the largest operand's K.
        assert_eq!(card.rows(), 20.0);
    }

    #[test]
    fn aggregate_empty_group_by_yields_one_row() {
        let aggr = LogicalAggregate {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            group_by: Vec::new(),
            aggregations: Vec::new(),
            span: span(),
        };
        let (_, card) = cost_aggregate(&aggr, Cardinality::new(1_000.0));
        assert_eq!(card.rows(), 1.0);
        assert_eq!(aggregate_output_for_empty_input(&aggr).rows(), 1.0);
    }

    #[test]
    fn vector_near_cost_proportional_to_k() {
        let v = LogicalVectorNear {
            var: BindingId::new(0),
            property: "embedding".into(),
            query_vector: BoundExpression::Parameter {
                name: "q".into(),
                span: span(),
                type_info: None,
            },
            k: 50,
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let (cost, card) = cost_vector_near(&v);
        assert_eq!(card.rows(), 50.0);
        assert_eq!(cost.total(), 50.0 * VECTOR_NEAR_COST_PER_K);
    }

    #[test]
    fn text_match_cost_uses_fallback_when_k_none() {
        let t = LogicalTextMatch {
            var: BindingId::new(0),
            property: "content".into(),
            query_text: BoundExpression::Parameter {
                name: "q".into(),
                span: span(),
                type_info: None,
            },
            k: None,
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let (cost, card) = cost_text_match(&t);
        assert_eq!(card.rows(), FALLBACK_TEXT_MATCH_K);
        assert_eq!(cost.total(), FALLBACK_TEXT_MATCH_K * TEXT_MATCH_COST_PER_K);
    }

    #[test]
    fn community_lookup_applies_fallback_selectivity() {
        let lookup = LogicalCommunityLookup {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            node_var: BindingId::new(0),
            community_id: BoundExpression::Parameter {
                name: "cid".into(),
                span: span(),
                type_info: None,
            },
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let (cost, card) = cost_community_lookup(&lookup, Cardinality::new(1_000.0));
        assert_eq!(cost.total(), 1_000.0 * COMMUNITY_LOOKUP_COST_PER_ROW);
        assert_eq!(card.rows(), 1_000.0 * FALLBACK_COMMUNITY_SELECTIVITY);
    }

    #[test]
    fn shortest_path_costs_more_per_node_than_plain() {
        let plain = LogicalNamedPath {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            path_var: BindingId::new(0),
            algorithm: PathAlgorithm::Plain,
            plain_shape: None,
            source: None,
            target: None,
            span: span(),
        };
        let shortest = LogicalNamedPath {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            path_var: BindingId::new(0),
            algorithm: PathAlgorithm::ShortestPath,
            plain_shape: None,
            source: None,
            target: None,
            span: span(),
        };
        let (plain_cost, _) = cost_named_path(&plain, Cardinality::new(100.0));
        let (sp_cost, _) = cost_named_path(&shortest, Cardinality::new(100.0));
        assert!(sp_cost.total() > plain_cost.total());
    }

    #[test]
    fn variable_length_expand_grows_with_midpoint_hops() {
        let snap = snapshot(1_000, 5_000, None);
        let expand = LogicalExpand {
            from: BindingId::new(0),
            to: BindingId::new(1),
            direction: Direction::LeftToRight,
            rel_type: None,
            length_range: Some(LengthRange::Cypher {
                min: 2,
                max: Some(4),
            }),
            rel_var: None,
            span: span(),
        };
        let (_, card) = cost_expand(&expand, Cardinality::new(10.0), &snap);
        // 10 input × 5 avg_degree × midpoint(2..4)=3 = 150
        assert_eq!(card.rows(), 150.0);
    }

    #[test]
    fn empty_operator_zero_cost_zero_rows() {
        let (cost, card) = cost_empty();
        assert_eq!(cost.total(), 0.0);
        assert_eq!(card.rows(), 0.0);
    }
}
