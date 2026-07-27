//! M4-91 EXPLAIN end-to-end integration tests per ADR-038 §2 D-19 +
//! amendment-03 §TIER-1 GAP B.
//!
//! # Pin set (matches the M4-91 roadmap-row test artifacts)
//!
//! 1. `explain_simple_match_returns_project_scan_tree` — EXPLAIN
//!    on `MATCH (n:Person) RETURN n` lowers to `Project(Scan)` with
//!    cost + cardinality flowing.
//! 2. `explain_match_where_threads_filter_node` — `MATCH (n:Person)
//!    WHERE n.age > 30 RETURN n.name` produces
//!    `Project(Filter(Scan))`.
//! 3. `explain_rank_by_hybrid_lowers_to_hybrid_tree` — RANK BY HYBRID
//!    \+ WITH FUSION = RRF surfaces both Fusion and RankByHybrid in
//!    the resulting tree.
//! 4. `explain_optional_match_emits_left_outer_join` — OPTIONAL MATCH
//!    surfaces `LeftOuterJoin` (per ADR-006 amendment-01 §A-2).
//! 5. `explain_pipelined_with_chain_lowers_cleanly` — multi-clause
//!    WITH pipeline (closest analog to v1.0 "multi-statement"; real
//!    multi-statement awaits M4-83 per ADR-038 §5.4.1).
//! 6. `explain_does_not_acquire_snapshot_lsn` — the BoundQuery
//!    produced inside the EXPLAIN flow MUST have
//!    `snapshot_lsn == None` (per ADR-038 §2 D-18 rule 1; pinned
//!    against future regressions that might wire the executor LSN
//!    handoff into the planner-only path).
//! 7. `explain_propagates_cross_substrate_error` — running EXPLAIN on
//!    a query that requires a substrate the catalog does NOT carry
//!    surfaces an `ArcQLError::CrossSubstrate` (not a NotImplemented
//!    or Internal error).
//! 8. `explain_renders_dp_chosen_join_order_on_skewed_chain` —
//!    Wave 9d M4-52b transit pin (W9b CRIT-1 closure). On a 3-leaf
//!    skewed-cardinality chain whose **input** order has the LARGEST
//!    leaf leftmost, the EXPLAIN-rendered [`PlanTree`]'s leftmost
//!    leaf MUST be the SMALLEST-cardinality scan (not the largest).
//!    This pin distinguishes a wired EXPLAIN (DP runs and re-orders)
//!    from an unwired EXPLAIN (M4-31 input-order plan rendered as-is).
//!    Phase 4.3 reverse-test cycle: verified that disabling the
//!    `enumerate_join_order` call in `plan_tree_for` flips the
//!    leftmost-leaf cardinality from 10 (small) to 1_000_000 (large).
//!
//! # ADR provenance
//! - ADR-038 §2 D-19 — EXPLAIN return shape contract.
//! - ADR-038 §2 D-18 — snapshot LSN binding rule 1 (EXPLAIN does
//!   NOT acquire).
//! - ADR-038 amendment-03 §TIER-1 GAP B — M4-91 sub-slice scope.
//! - ADR-038 amendment-02 §M4.e — M4-52 (M4-05b) DP join ordering.
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH lowers to
//!   LeftOuterJoin.

use std::sync::atomic::{AtomicU32, Ordering};

use arcgraph_core::{LabelId, PartitionId, PropertyId, TenantId, TypeId};
use arcgraph_query::ast::Statement;
use arcgraph_query::explain::{ExecutionMetrics, ExplainError, PlanTree, PlanTreeOp};
use arcgraph_query::semantic::{
    ArcQLError, BindingVisitor, BoundStatement, CatalogProvider, CatalogSnapshot,
    StubCatalogProvider,
};
use arcgraph_query::{explain, parse, profile};

mod common;
use common::m4_04d_person_tenant::{PLACE_COUNT, PersonTenant};

// ---------------------------------------------------------------------
// Catalog fixtures
// ---------------------------------------------------------------------

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Comment"])
        .with_rel_types(["KNOWS", "REPLY_OF", "HAS_CREATOR"])
        .with_properties(["age", "name", "id"])
}

fn cat_hybrid() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Doc", "Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["embedding", "content", "name"])
        .with_vector_index()
        .with_bm25_index()
        .with_community_index()
}

// Walk the plan tree and produce a flat pre-order operator-name list
// — useful for shape pins that don't depend on cost numbers.
fn shape(pt: &PlanTree) -> Vec<&'static str> {
    let mut out = Vec::new();
    walk(pt, &mut out);
    out
}

fn walk(pt: &PlanTree, out: &mut Vec<&'static str>) {
    out.push(pt.op.name());
    for c in &pt.children {
        walk(c, out);
    }
}

// ---------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------

#[test]
fn explain_simple_match_returns_project_scan_tree() {
    let pt = explain("EXPLAIN MATCH (n:Person) RETURN n", &cat_basic()).expect("explain");
    assert_eq!(pt.op, PlanTreeOp::Project);
    let s = shape(&pt);
    assert_eq!(s, vec!["Project", "Scan"]);
    // Cost monotonicity: the Project's subtree cost MUST be >= the
    // Scan's local cost (the scan is the only contributor to the
    // bottom-up sum). We check via the cost field on the scan child.
    let scan = &pt.children[0];
    assert!(
        pt.estimated_cost.total() >= scan.estimated_cost.total(),
        "project cost >= scan cost",
    );
}

#[test]
fn explain_match_where_threads_filter_node() {
    let pt = explain(
        "EXPLAIN MATCH (n:Person) WHERE n.age > 30 RETURN n.name",
        &cat_basic(),
    )
    .expect("explain");
    let s = shape(&pt);
    assert_eq!(
        s,
        vec!["Project", "Filter", "Scan"],
        "MATCH (n:Person) WHERE … RETURN … must thread through Project→Filter→Scan",
    );
}

#[test]
fn explain_rank_by_hybrid_lowers_to_hybrid_tree() {
    // The RANK BY HYBRID + WITH FUSION surface produces a tree
    // containing both `Fusion` and `RankByHybrid` somewhere. The
    // exact M4-32 lowering shape is `Project + Fusion + Join + Scan
    // + RankByHybrid`; we assert presence (not exact ordering) to
    // keep the test resilient to M4-05 future re-ordering.
    let pt = explain(
        "EXPLAIN MATCH (a:Doc) RANK BY HYBRID(VECTOR(a.embedding, $qv, K = 20), \
         TEXT(a.content, $qt, K = 20)) WITH FUSION = RRF(k = 60) RETURN a LIMIT 10",
        &cat_hybrid(),
    )
    .expect("explain hybrid");
    let s = shape(&pt);
    assert!(
        s.contains(&"RankByHybrid"),
        "tree must contain RankByHybrid: shape={s:?}",
    );
    assert!(
        s.contains(&"Fusion"),
        "tree must contain Fusion: shape={s:?}",
    );
    // Fusion node carries the RRF kind + k=60 in its annotations
    // (regardless of where in the tree it sits).
    let fusion = find_first(&pt, PlanTreeOp::Fusion).expect("Fusion node present");
    assert_eq!(
        fusion.annotations.get("kind").map(String::as_str),
        Some("RRF"),
    );
    assert_eq!(fusion.annotations.get("k").map(String::as_str), Some("60"));
}

#[test]
fn explain_optional_match_emits_left_outer_join() {
    // OPTIONAL MATCH lowers to LeftOuterJoin per ADR-006 amendment-01
    // §A-2 (M4-32). EXPLAIN surfaces this directly.
    let pt = explain(
        "EXPLAIN MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b:Person) RETURN a, b",
        &cat_basic(),
    )
    .expect("explain optional match");
    let s = shape(&pt);
    assert!(
        s.contains(&"LeftOuterJoin"),
        "OPTIONAL MATCH must lower to a LeftOuterJoin: shape={s:?}",
    );
    let loj = find_first(&pt, PlanTreeOp::LeftOuterJoin).expect("LeftOuterJoin present");
    // The LeftOuterJoin's `condition` annotation is non-empty —
    // either Cartesian or shared-bindings, never absent.
    assert!(
        loj.annotations.contains_key("condition"),
        "LeftOuterJoin must carry a `condition` annotation: anns={:?}",
        loj.annotations,
    );
}

#[test]
fn explain_pipelined_with_chain_lowers_cleanly() {
    // True multi-statement parsing (`q1; q2;`) is deferred to M4-83
    // per ADR-038 §5.4.1 — the v1.0 grammar admits a single
    // `read_query` per `query` parse. The closest analog at M4-91 is
    // the WITH pipeline, which chains multiple MATCH-like sub-stages
    // through scope-rotating WITH clauses. The test pin: EXPLAIN
    // accepts a WITH-chained query and produces a non-degenerate
    // plan tree (>= 2 nodes).
    //
    // When M4-83 lights real multi-statement, the EXPLAIN entry point
    // will need to handle a sequence of statements; for now this pin
    // exercises the end-to-end path that's actually admissible at
    // v1.0.
    let pt = explain(
        "EXPLAIN MATCH (n:Person) WITH n MATCH (n)-[:KNOWS]->(m:Person) RETURN m",
        &cat_basic(),
    )
    .expect("explain pipelined");
    let s = shape(&pt);
    assert!(
        s.len() >= 2,
        "pipelined EXPLAIN must produce a non-degenerate tree: shape={s:?}",
    );
    // The last clause is a RETURN, so the root is always a Project.
    assert_eq!(
        pt.op,
        PlanTreeOp::Project,
        "root must be Project: shape={s:?}"
    );
}

#[test]
fn explain_does_not_acquire_snapshot_lsn() {
    // EXPLAIN MUST NOT acquire a snapshot LSN per ADR-038 §2 D-18
    // rule 1. The discipline is enforced by inspection: the EXPLAIN
    // entry point routes through `BindingVisitor::bind` →
    // `TypeCheckVisitor::check` → `CrossSubstrateValidator::validate`
    // → `LogicalPlanLoweringVisitor::lower` → `estimate_costs` →
    // `PlanTree::from_costed_plan`. NONE of those passes write to
    // `BoundQuery::snapshot_lsn` (M4-21 always sets it to `None`;
    // amendment-03 §TIER-1 GAP E reserves the slot for M4-61's
    // pre-first-batch acquisition).
    //
    // Here we replicate the binding step from the entry point's
    // pipeline + introspect the resulting `BoundQuery` to assert
    // `snapshot_lsn == None`. If a future M4-91 refactor hooks the
    // executor LSN handoff into the planner-only path, this test
    // reds.
    let cat = cat_basic();
    let input = "EXPLAIN MATCH (n:Person) WHERE n.age > 30 RETURN n.name";

    // 1) Confirm the public entry point succeeds (sanity).
    let _pt = explain(input, &cat).expect("explain success");

    // 2) Replicate the binding step + inspect.
    let stmt = parse(input).expect("parse");
    // EXPLAIN-wrap the input so the binding pass sees the same shape
    // the entry point processes (the entry point's
    // `strip_explain_or_profile_wrapper` runs BEFORE binding; the
    // pipeline is shape-identical to a bare-read-query bind).
    let bound = match &stmt {
        Statement::Explain(_) => {
            BindingVisitor::bind(&stmt, input, &cat).expect("bind explain wrapper")
        }
        _ => panic!("expected Statement::Explain"),
    };
    match bound {
        BoundStatement::Read(q) => {
            assert!(
                q.snapshot_lsn.is_none(),
                "EXPLAIN path must leave BoundQuery::snapshot_lsn unset \
                 (per ADR-038 §2 D-18 rule 1) — got {:?}",
                q.snapshot_lsn,
            );
        }
        other => panic!("expected BoundStatement::Read, got {other:?}"),
    }
}

#[test]
fn explain_propagates_cross_substrate_error() {
    // The cross-substrate validator rejects RANK BY HYBRID against
    // a catalog WITHOUT vector + BM25 indices. EXPLAIN surfaces the
    // CrossSubstrate error verbatim (no swallowing, no double-wrap).
    let bare_cat = StubCatalogProvider::new()
        .with_labels(["Doc"])
        .with_properties(["embedding", "content"]);
    // No `.with_vector_index()` / `.with_bm25_index()` — so the
    // hybrid query MUST be rejected at the cross-substrate layer.
    let err = explain(
        "EXPLAIN MATCH (a:Doc) RANK BY HYBRID(VECTOR(a.embedding, $qv, K = 5), \
         TEXT(a.content, $qt, K = 5)) WITH FUSION = RRF(k = 60) RETURN a",
        &bare_cat,
    )
    .expect_err("cross-substrate rejection");
    match err {
        ExplainError::ArcQL(ArcQLError::CrossSubstrate(_)) => {}
        other => panic!("expected ArcQL(CrossSubstrate), got {other:?}"),
    }
}

#[test]
fn profile_returns_plan_tree_and_metrics_at_w12gamma() {
    // W12γ M4-91 PROFILE wire-up pin (replaces the previous
    // NotImplemented stub per ADR-038 amendment-03 §TIER-1 GAP B).
    // PROFILE now drives the executor end-to-end and returns a
    // `(PlanTree, ExecutionMetrics)` tuple: the plan tree shape is
    // the same as EXPLAIN's; the metrics carry top-level
    // wall_time_ms + rows_emitted (per-operator decomposition is
    // forward-deferred to W12β's M4-71 RowCountObserver).
    let s = arcgraph_query::executor::StubExecutorSubstrate::new();
    let registry = arcgraph_query::CancellationRegistry::new();
    let (pt, metrics) = profile(
        "PROFILE MATCH (n:Person) RETURN n",
        &cat_basic(),
        &s,
        &registry,
        std::time::Duration::from_millis(arcgraph_query::DEFAULT_QUERY_TIMEOUT_MS),
    )
    .expect("profile succeeds at W12γ");
    assert_eq!(pt.op, PlanTreeOp::Project, "plan tree shape preserved");
    // The substrate carries no Person rows; rows_emitted is 0.
    assert_eq!(metrics.rows_emitted, 0);
    // Forward-bind: M4-64a populates this field; v1.0-alpha = 0.
    assert_eq!(metrics.memory_bytes_high_water, 0);
}

#[test]
fn execution_metrics_default_constructs_zero_struct() {
    // ExecutionMetrics default = zero. Pinned so M4-71 wiring knows
    // the safe placeholder when the row-count observer hasn't fired
    // yet.
    let m = ExecutionMetrics::default();
    assert_eq!(m.wall_time_ms, 0);
    assert_eq!(m.memory_bytes_high_water, 0);
    assert_eq!(m.rows_emitted, 0);
}

/// **Wave 9d M4-52b transit pin (W9b CRIT-1 closure).**
///
/// This is the load-bearing wave-level integration test for the W9b
/// CRIT-1 finding: "M4-52's 3,500-LOC DP enumerator has zero production
/// callers". After M4-52b-WIRING threads `enumerate_join_order` between
/// `LogicalPlanLoweringVisitor::lower` and `estimate_costs` inside
/// `crate::explain::plan_tree_for`, this test asserts that EXPLAIN's
/// rendered [`PlanTree`] reflects the cost-optimal join order rather
/// than the M4-31 input order.
///
/// # Construction
///
/// We build a 3-leaf inner-join chain whose INPUT is intentionally
/// worst-ordered (largest leaf leftmost). Cardinalities are skewed by
/// five orders of magnitude:
///
/// | Label | Catalog cardinality | Role |
/// |---|---:|---|
/// | `Large`  | 1_000_000 | leftmost in input |
/// | `Medium` |     1_000 | middle |
/// | `Small`  |        10 | rightmost in input |
///
/// Without DP, M4-31 lowering produces a left-deep input-order plan
/// whose **leftmost** Scan leaf is `Large` (1 M rows). With DP, the
/// enumerator picks the cost-optimal left-deep order: smallest-leaf
/// leftmost (`Small`, 10 rows). The reverse-test cycle (state 1 PASS →
/// remove the wiring → state 2 FAIL → restore → state 3 PASS) is the
/// non-vacuity oracle described in
/// `feedback_anchor_to_consumer_transit_pinning.md` §"reverse-test
/// discipline".
///
/// # Phase 4.3 reverse-test (verified empirically at slice-build time)
///
/// 1. **State 1** (wiring on, baseline) — `assert leftmost_card == 10`
///    PASSES; total cost ≈ 11.37 M.
/// 2. **State 2** (wiring removed: comment out `enumerate_join_order`
///    call in `plan_tree_for`) — `assert leftmost_card == 10` FAILS;
///    leftmost is `Large` (1 M rows); total cost ≈ 12.37 M.
/// 3. **State 3** (wiring restored) — back to State 1.
///
/// # ADR cites
/// - ADR-038 amendment-02 §M4.e — DP join ordering at v1.0 (left-deep
///   only; bushy deferred to v1.1).
/// - ADR-038 §2 D-19 — EXPLAIN return shape; the cost in
///   [`PlanTree::estimated_cost`] is the post-DP cost.
#[test]
fn explain_renders_dp_chosen_join_order_on_skewed_chain() {
    // Skewed-cardinality catalog: 5 orders of magnitude between
    // smallest and largest leaf.
    let cat = StubCatalogProvider::new()
        .with_labels(["Small", "Medium", "Large"])
        .with_rel_types(["R1", "R2"])
        .with_total_node_count(1_001_010)
        .with_total_rel_count(2_000_000)
        .with_label_cardinality(LabelId::new(1), 10) // Small
        .with_label_cardinality(LabelId::new(2), 1_000) // Medium
        .with_label_cardinality(LabelId::new(3), 1_000_000); // Large

    // Input order: Large → Medium → Small (worst-case left-deep).
    // M4-31 lowering produces a left-deep Join chain whose leftmost
    // Scan is Large; the DP must re-root with Small leftmost.
    let pt = explain(
        "EXPLAIN MATCH (c:Large)-[:R1]->(b:Medium)-[:R2]->(a:Small) RETURN a, b, c",
        &cat,
    )
    .expect("explain skewed chain");

    let leftmost = leftmost_scan(&pt).expect("plan has at least one Scan leaf");
    assert_eq!(
        leftmost.op,
        PlanTreeOp::Scan,
        "leftmost leaf must be a Scan: tree=\n{pt}",
    );
    assert!(
        (leftmost.estimated_card.rows() - 10.0).abs() < 1e-9,
        "DP must re-order so smallest-leaf (Small=10) is leftmost; got {} rows. \
         Likely cause: `enumerate_join_order` is not wired into `plan_tree_for`. \
         Tree:\n{pt}",
        leftmost.estimated_card.rows(),
    );
    // Strong negative form: the leftmost is NOT the input-order
    // largest leaf (Large=1M). Distinguishes a wired EXPLAIN from an
    // unwired one even if a future cost-model refinement changes the
    // numeric leftmost cardinality.
    assert!(
        (leftmost.estimated_card.rows() - 1_000_000.0).abs() > 1e-3,
        "leftmost MUST NOT be the M4-31 input-order Large scan (1_000_000 rows). \
         If this fires the DP enumerator likely is not running. Tree:\n{pt}",
    );
}

/// **Wave 10b issue #261 closure — single-FrozenCatalog threading pin.**
///
/// Per W9d retro Agent A §A-LOW-1, `crate::explain::plan_tree_for`
/// previously captured up to THREE independent
/// [`arcgraph_query::semantic::CatalogSnapshot`]s within a single
/// EXPLAIN call — one for the cache watermark, one for the M4-52 DP
/// enumerator, one for the M4-51 cost walker. Under v1.1+ multi-
/// tenant concurrent writers the snapshots could drift, producing
/// apples-to-oranges cost annotations within a single EXPLAIN.
///
/// This test wraps the catalog with a [`CountingCatalog`] adapter that
/// counts every `snapshot()` call. Post-fix, EXPLAIN MUST produce
/// `snapshot_count() == 1` (single capture, threaded through all
/// stages). Pre-fix the count would be ≥ 2 (one for the enumerator,
/// one for the cost walker) or ≥ 3 (with cache: also one for the
/// watermark check).
///
/// # Reverse-test cycle (Phase 4.3, verified at slice-build time)
///
/// 1. **State 1** (post-fix, baseline) — `assert snapshot_count == 1`
///    PASSES under both the bare-EXPLAIN path and the cached path.
/// 2. **State 2** (revert: re-introduce per-stage `catalog.snapshot()`
///    calls in `enumerate_join_order` / `estimate_costs` instead of
///    threading the [`crate::planner::enumeration::FrozenCatalog`]) —
///    `assert snapshot_count == 1` FAILS (count would be 2 or 3).
/// 3. **State 3** (restore) — back to State 1.
///
/// # ADR cites
///
/// - ADR-038 §2 D-25 — catalog stats schema.
/// - ADR-038 amendment-03 §M4-04e — cross-key snapshot consistency
///   contract (PR #220 producer; M4-51 + M4-52 + M4-53 + M4-91
///   consumers).
#[test]
fn explain_pipeline_uses_single_frozen_catalog() {
    // Bare-EXPLAIN path: no cache attached. Single capture expected.
    let inner = cat_basic().with_total_node_count(1_000);
    let counting = CountingCatalog::new(inner);
    let _pt = explain("MATCH (n:Person) RETURN n", &counting).expect("bare explain");
    assert_eq!(
        counting.snapshot_count(),
        1,
        "bare-EXPLAIN must capture exactly ONE catalog snapshot per call \
         (issue #261); pre-fix was 2 (enumerate + cost). Tree fan-out \
         is irrelevant — the snapshot is hoisted to plan_tree_for.",
    );

    // Cached-EXPLAIN path: with a plan cache attached, the watermark
    // capture is folded into the same single snapshot capture.
    use arcgraph_query::explain::{QueryEngine, explain_with_cache};
    use arcgraph_query::planner::cache::PlanCache;
    use std::sync::Arc;

    let inner = cat_basic().with_total_node_count(1_000);
    let counting = CountingCatalog::new(inner);
    let cache = PlanCache::new();
    let _pt = explain_with_cache("MATCH (n:Person) RETURN n", &counting, &cache)
        .expect("cached explain miss path");
    assert_eq!(
        counting.snapshot_count(),
        1,
        "cached-EXPLAIN miss path must capture exactly ONE catalog snapshot \
         per call (issue #261); pre-fix was 3 (watermark + enumerate + cost).",
    );

    // Cache HIT path: the second EXPLAIN of the same query returns
    // from cache; we still expect exactly one snapshot (the cache
    // watermark check). Pre-fix: also 1 — this state is unaffected
    // by the fix; included to pin the upper bound on the hit path.
    let _pt = explain_with_cache("MATCH (n:Person) RETURN n", &counting, &cache)
        .expect("cached explain hit path");
    assert_eq!(
        counting.snapshot_count(),
        2,
        "cached-EXPLAIN hit path must add exactly ONE catalog snapshot \
         per call to the running counter (cache watermark only).",
    );

    // QueryEngine entry-point parity: routing through the public
    // engine wrapper produces the same snapshot-count semantics as
    // the free function (the engine just forwards).
    let inner = cat_basic().with_total_node_count(1_000);
    let counting = CountingCatalog::new(inner);
    let engine = QueryEngine::new(&counting).with_cache(Arc::new(PlanCache::new()));
    let _pt = engine
        .explain("MATCH (n:Person) RETURN n")
        .expect("engine explain");
    assert_eq!(
        counting.snapshot_count(),
        1,
        "QueryEngine cached EXPLAIN must capture exactly ONE catalog \
         snapshot per call (issue #261).",
    );
}

/// **Wave 10b issue #262 closure — M4-04d EMPIRICAL fixture transit pin
/// (W9d retro CR-A-1 residual).**
///
/// PR #258's MED-1 wave-level transit pin
/// (`explain_renders_dp_chosen_join_order_on_skewed_chain` above) verifies
/// the lowering → DP → cost → EXPLAIN wiring chain is composed correctly,
/// but uses a synthetic [`StubCatalogProvider`] (`Small`/`Medium`/`Large`
/// labels) for cardinality inputs — NOT the actual M4-04d empirical
/// `PersonTenant` fixture published by PR #234.
///
/// Result: the producer (M4-04d empirical `DEFAULT_*_SELECTIVITY`
/// constants → fixture cardinalities) → consumer (M4-51 cost walker)
/// → output (EXPLAIN PlanTree) transit was composed STRUCTURALLY but
/// not pinned end-to-end against the real empirical fixture. This test
/// closes the gap by running EXPLAIN on a 3-leaf inner-join chain over
/// the M4-04d auxiliary tenant cardinalities (Person=10K, Comment=100K,
/// Place=1K at SF-0.01) and asserting the DP picks the smallest-leaf
/// (Place=1K) leftmost — the cost-optimal left-deep order under those
/// cardinalities.
///
/// # Why SF-0.01 instead of SF-1.0
///
/// SF-1.0 (1M Persons) is the canonical empirical anchor cited by
/// ADR-038 amendment-07, but for an EXPLAIN integration test the
/// absolute scale is irrelevant: the cost walker reads cardinalities,
/// not row data, and the plan-order pin is invariant under uniform
/// scaling. SF-0.01 keeps cargo-test build-time effectively zero
/// (a few HashMap inserts via [`PersonTenant::build_catalog`]) — the
/// W10b spawn-prompt §Item 5 explicitly authorizes this scale-down.
///
/// # ADR cites
///
/// - ADR-038 amendment-07 — M4-04d empirical selectivity tuning;
///   `PersonTenant`'s SF-1.0 anchor.
/// - ADR-038 amendment-02 §M4.e — DP join ordering at v1.0.
/// - W9d retro Agent A §8.4 / CR-A-1 — empirical-fixture transit gap.
#[test]
fn explain_renders_dp_chosen_join_order_on_m4_04d_person_tenant_fixture() {
    let tenant = PersonTenant::seed(); // SF-0.01.
    let cat = tenant.build_catalog();

    // 3-leaf inner-join chain whose INPUT order is worst-case
    // (largest leaf leftmost — Comment=100K). Cardinalities:
    // - Comment = 100_000 (largest)
    // - Person  =  10_000
    // - Place   =   1_000 (smallest)
    // The DP must re-root with Place leftmost.
    let pt = explain(
        "EXPLAIN MATCH (c:Comment)-[:KNOWS]->(p:Person)-[:IS_LOCATED_IN]->(pl:Place) \
         RETURN c, p, pl",
        &cat,
    )
    .expect("explain over M4-04d Person tenant");

    let leftmost = leftmost_scan(&pt).expect("plan has at least one Scan leaf");
    assert_eq!(
        leftmost.op,
        PlanTreeOp::Scan,
        "leftmost leaf must be a Scan; tree=\n{pt}",
    );
    assert!(
        (leftmost.estimated_card.rows() - PLACE_COUNT as f64).abs() < 1e-9,
        "DP must re-root with Place ({} rows — smallest leaf in M4-04d \
         fixture) leftmost; got {} rows. Likely cause: enumerate_join_order \
         is not consuming the M4-04d cardinalities. Tree:\n{pt}",
        PLACE_COUNT,
        leftmost.estimated_card.rows(),
    );
    // Strong negative form: the leftmost is NOT the input-order
    // largest leaf (Comment=100K). Distinguishes a wired EXPLAIN
    // from an unwired one even if a future cost-model refinement
    // changes the numeric leftmost cardinality.
    assert!(
        (leftmost.estimated_card.rows() - 100_000.0).abs() > 1e-3,
        "leftmost MUST NOT be the M4-31 input-order Comment scan \
         (100_000 rows). Tree:\n{pt}",
    );
}

/// **Wave 10b issue #262 closure — Phase 4.2 controlled-mutation oracle
/// non-vacuity cycle.**
///
/// Per `feedback_anchor_to_consumer_transit_pinning.md` reverse-test
/// discipline + the W10b spawn-prompt §Item 5 acceptance criterion,
/// this test demonstrates the M4-04d → M4-51 → EXPLAIN transit is
/// LOAD-BEARING by perturbing the producer-side cardinalities and
/// asserting the consumer-side EXPLAIN cost output shifts proportionally.
///
/// # Mutation choice
///
/// The W10b spawn-prompt §Item 5 phrases the cycle as "10×-scale
/// `DEFAULT_LABEL_SELECTIVITY`". That constant is consulted ONLY in
/// the fallback path where `snapshot.label_card(label) == None` and
/// `snapshot.total_nodes() == Some(N)` — the cost walker computes
/// `N * DEFAULT_LABEL_SELECTIVITY` for each leaf. The M4-04d fixture
/// EXPLICITLY populates per-label cardinalities, so the fallback path
/// is bypassed. The functionally-equivalent producer-side mutation is
/// to scale every label cardinality by 10× directly (each leaf's
/// estimated rows × 10 — observably identical effect on the cost
/// walker). [`PersonTenant::scale_all_label_cards`] returns the
/// 10×-mutated copy.
///
/// # Phase 4.2 oracle
///
/// 1. **State 1 (baseline)** — explain over `tenant.build_catalog()`,
///    capture `cost_a`.
/// 2. **State 2 (mutation)** — explain over
///    `tenant.scale_all_label_cards(10).build_catalog()`,
///    capture `cost_b`. Assert `cost_b > cost_a * 1.5` — a strict
///    lower bound demonstrating non-trivial cardinality flow-through.
///    The empirically-observed shift is ~2× (not the naive 10×) because
///    `cost_expand`'s formula uses `avg_degree = total_rels / total_nodes`;
///    a 10× label scaling propagates via [`PersonTenant::total_nodes`]
///    to scale `total_nodes` 10× while `total_edges` stays constant, so
///    `avg_degree` halves and partially cancels the input-side 10×
///    scaling on Expand operators. Scan operators see the full 10×
///    shift (no avg_degree term). The aggregate is dominantly Scan-cost
///    on a 3-leaf chain — `~2× shift` is the load-bearing signal.
/// 3. **State 3 (restoration)** — explain over the unmutated catalog
///    again; assert `cost_restored == cost_a`. Pins determinism.
///
/// # ADR cites
///
/// - W9d retro Agent A §8.4 / CR-A-1 — empirical-fixture transit gap.
/// - PR #234 — M4-04d empirical `DEFAULT_*_SELECTIVITY` constants.
/// - `feedback_anchor_to_consumer_transit_pinning.md` — reverse-test
///   discipline.
#[test]
fn empirical_fixture_phase_4_2_mutation_on_default_label_selectivity() {
    let baseline = PersonTenant::seed();
    let cat_a = baseline.build_catalog();
    let pt_a = explain(
        "EXPLAIN MATCH (c:Comment)-[:KNOWS]->(p:Person)-[:IS_LOCATED_IN]->(pl:Place) \
         RETURN c, p, pl",
        &cat_a,
    )
    .expect("baseline explain");
    let cost_a = pt_a.estimated_cost.total();
    let leftmost_a = leftmost_scan(&pt_a)
        .expect("baseline tree has Scan leaf")
        .estimated_card
        .rows();

    // State 2: 10× scale every label cardinality (analog of 10×-scaling
    // DEFAULT_LABEL_SELECTIVITY in the fallback path).
    let mutated = baseline.scale_all_label_cards(10);
    let cat_b = mutated.build_catalog();
    let pt_b = explain(
        "EXPLAIN MATCH (c:Comment)-[:KNOWS]->(p:Person)-[:IS_LOCATED_IN]->(pl:Place) \
         RETURN c, p, pl",
        &cat_b,
    )
    .expect("mutated explain");
    let cost_b = pt_b.estimated_cost.total();
    let leftmost_b = leftmost_scan(&pt_b)
        .expect("mutated tree has Scan leaf")
        .estimated_card
        .rows();

    // Phase 4.2 oracle non-vacuity: cost shifts non-trivially with
    // cardinality scaling. The 1.5× lower bound accommodates the
    // avg_degree absorption documented above — the empirical shift
    // on this 3-leaf Person tenant chain is ~2× (Scan terms scale
    // 10×, Expand terms cancel partially).
    assert!(
        cost_b > cost_a * 1.5,
        "cost did not shift after 10× cardinality scaling: \
         cost_a={cost_a:.3}, cost_b={cost_b:.3}, ratio={:.3}× — \
         empirical-fixture → cost-walker transit is NOT load-bearing. \
         Likely cause: cost walker reads stub cardinalities or hard-codes \
         a constant.",
        cost_b / cost_a.max(1e-9),
    );
    // Leftmost-leaf cardinality also shifts (smallest leaf = Place;
    // its scaled cardinality is 10K under the 10× mutation).
    assert!(
        (leftmost_b - leftmost_a * 10.0).abs() < 1.0,
        "leftmost-leaf cardinality did not shift 10×: a={leftmost_a}, b={leftmost_b}",
    );

    // State 3: restoration — re-explain over the original catalog
    // and confirm idempotence.
    let cat_restored = baseline.build_catalog();
    let pt_restored = explain(
        "EXPLAIN MATCH (c:Comment)-[:KNOWS]->(p:Person)-[:IS_LOCATED_IN]->(pl:Place) \
         RETURN c, p, pl",
        &cat_restored,
    )
    .expect("restored explain");
    assert!(
        (pt_restored.estimated_cost.total() - cost_a).abs() < 1e-9,
        "restoration not idempotent — explain cost differs across re-runs \
         on the same fixture: a={cost_a}, restored={}",
        pt_restored.estimated_cost.total(),
    );
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

/// `CatalogProvider` adapter that counts `snapshot()` calls — used by
/// the issue #261 single-snapshot regression pin. All other methods
/// delegate to the inner catalog unchanged.
struct CountingCatalog<C: CatalogProvider> {
    inner: C,
    snapshot_count: AtomicU32,
}

impl<C: CatalogProvider> CountingCatalog<C> {
    fn new(inner: C) -> Self {
        Self {
            inner,
            snapshot_count: AtomicU32::new(0),
        }
    }

    fn snapshot_count(&self) -> u32 {
        self.snapshot_count.load(Ordering::Relaxed)
    }
}

impl<C: CatalogProvider> CatalogProvider for CountingCatalog<C> {
    fn lookup_label(&self, name: &str) -> Option<LabelId> {
        self.inner.lookup_label(name)
    }
    fn lookup_rel_type(&self, name: &str) -> Option<TypeId> {
        self.inner.lookup_rel_type(name)
    }
    fn lookup_property(&self, name: &str) -> Option<PropertyId> {
        self.inner.lookup_property(name)
    }
    fn tenant(&self) -> TenantId {
        self.inner.tenant()
    }
    fn partition(&self) -> PartitionId {
        self.inner.partition()
    }
    fn has_vector_index(&self) -> bool {
        self.inner.has_vector_index()
    }
    fn has_bm25_index(&self) -> bool {
        self.inner.has_bm25_index()
    }
    fn has_community_index(&self) -> bool {
        self.inner.has_community_index()
    }
    fn label_cardinality(&self, label: LabelId) -> Option<u64> {
        self.inner.label_cardinality(label)
    }
    fn rel_type_cardinality(&self, rel_type: TypeId) -> Option<u64> {
        self.inner.rel_type_cardinality(rel_type)
    }
    fn total_node_count(&self) -> Option<u64> {
        self.inner.total_node_count()
    }
    fn total_rel_count(&self) -> Option<u64> {
        self.inner.total_rel_count()
    }
    fn snapshot(&self) -> CatalogSnapshot {
        self.snapshot_count.fetch_add(1, Ordering::Relaxed);
        self.inner.snapshot()
    }
}

fn find_first(pt: &PlanTree, op: PlanTreeOp) -> Option<&PlanTree> {
    if pt.op == op {
        return Some(pt);
    }
    for c in &pt.children {
        if let Some(found) = find_first(c, op) {
            return Some(found);
        }
    }
    None
}

/// Walk leftmost-children to the first [`PlanTreeOp::Scan`] reached.
///
/// The DP's left-deep output places the chosen-leftmost leaf at
/// `children[0].children[0]...` until a `Scan` is reached; the
/// [`PlanTreeOp::Expand`] adjacent leaves bubble up as right siblings
/// of internal Joins. This helper makes the MED-1 transit pin's
/// leftmost-leaf assertion concise.
fn leftmost_scan(pt: &PlanTree) -> Option<&PlanTree> {
    let mut node = pt;
    loop {
        if node.op == PlanTreeOp::Scan {
            return Some(node);
        }
        node = node.children.first()?;
    }
}
