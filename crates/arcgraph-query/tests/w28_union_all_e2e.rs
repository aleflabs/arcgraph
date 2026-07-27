//! W28 #649-A1 — openCypher v9 §8 UNION ALL + standalone DISTINCT
//! end-to-end + pinning tests (ADR-185).
//!
//! These are the PE FROZEN CONTRACT pinning tests. Each maps to a
//! contract item and (where applicable) to the vendored openCypher TCK
//! `tck/features/clauses/union/Union{1,2,3}.feature` golden semantics.
//! All oracles are STRONG `==` over the row multiset / golden order.
//!
//! # Contract → test map
//!
//! - Item 1 (RC-2: tail binds whole union): `tail_binds_whole_union_*`
//!   (parse-structure pin + executable golden-order pin).
//! - Item 2 (RC-1: standalone dedup op lights RETURN DISTINCT):
//!   `return_distinct_now_executes`.
//! - Item 3 (column-compat, by-name set-equality, FAILs on mismatch):
//!   `column_mismatch_fails_at_bind` + `same_alias_diff_expr_compatible`
//!   + `column_order_independent_realignment`.
//! - TCK Union3 (no mixing): `mixed_union_and_union_all_rejected`.
//! - TCK Union2 (UNION ALL keeps dupes): `union_all_keeps_duplicates`.
//! - TCK Union1 (#649-A2: bare UNION removes dupes):
//!   `bare_union_distinct_dedups_across_arms` +
//!   `bare_union_distinct_dedups_identical_rows_from_both_arms` +
//!   `bare_union_distinct_realigns_columns_before_dedup` +
//!   `bare_union_distinct_tail_limit_applies_post_dedup`; plan-shape pins
//!   `bare_union_distinct_now_lowers_to_distinct_over_union` +
//!   `union_all_still_lowers_to_bare_union_no_distinct`.
//!
//! # Why MATCH-based arms (not literal `RETURN 1`)
//!
//! The v1.0 executor's `EmptyOp` emits zero rows, so a MATCH-less
//! `RETURN <literal>` arm currently produces no output rows (a
//! pre-existing singleton-leaf gap, out of #649-A1 scope). The TCK
//! literal scenarios (`Union2[1..3]`) are therefore exercised here in
//! their MATCH-bearing form (the `Union2[4]` shape), which produces
//! real substrate rows; the literal forms are pinned at the
//! parse/bind layer instead.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::ast::{Clause, Statement};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::error::BindingError;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

// Labels are 1-based in declaration order: A=1, B=2, X=3.
fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["A", "B", "X"])
        .with_rel_types(["KNOWS"])
        .with_properties(["v", "w", "g"])
}

const LABEL_A: u32 = 1;
const LABEL_B: u32 = 2;
const LABEL_X: u32 = 3;

fn node(id: u64, label: u32) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
}

/// Bind + type-check + validate + lower + materialize a query against a
/// substrate, returning the result rows. Panics on any pipeline error
/// (the tests that expect an error call the lower-level entry points).
fn run(
    query: &str,
    substrate: &StubExecutorSubstrate,
    catalog: &StubCatalogProvider,
) -> Vec<Vec<Value>> {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, catalog).expect("bind");
    TypeCheckVisitor::check(&mut bound, catalog).expect("type-check");
    CrossSubstrateValidator::validate(&bound, catalog).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(catalog.tenant(), catalog.partition());
    let result = materialize::materialize(&plan, substrate, &ctx).expect("materialize");
    result.rows().to_vec()
}

/// Extract a single-column integer result as a sorted Vec (multiset
/// oracle — order-independent comparison for the "keeps dupes" pins).
fn sorted_ints(rows: &[Vec<Value>]) -> Vec<i64> {
    let mut out: Vec<i64> = rows
        .iter()
        .map(|r| match &r[0] {
            Value::Integer(n) => *n,
            other => panic!("expected Integer column, got {other:?}"),
        })
        .collect();
    out.sort_unstable();
    out
}

/// Extract a single-column integer result preserving ORDER (golden
/// row-order oracle — for the ORDER BY tail pins).
fn ordered_ints(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter()
        .map(|r| match &r[0] {
            Value::Integer(n) => *n,
            other => panic!("expected Integer column, got {other:?}"),
        })
        .collect()
}

// =====================================================================
// Contract item 2 (RC-1) — RETURN DISTINCT now EXECUTES
// (closes the prior pipeline.rs Distinct NotImplemented)
// =====================================================================

#[test]
fn return_distinct_now_executes() {
    // Three X-nodes with g ∈ {1, 1, 2}. RETURN DISTINCT n.g dedups to
    // {1, 2}. Before #649-A1 this path returned ExecutionError::
    // NotImplemented at pipeline.rs:251.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_X).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_X).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3, LABEL_X).with_property("g", Value::Integer(2)),
        );
    let rows = run("MATCH (n:X) RETURN DISTINCT n.g", &s, &cat());
    assert_eq!(sorted_ints(&rows), vec![1, 2], "RETURN DISTINCT dedups n.g");
}

// =====================================================================
// TCK Union2 — UNION ALL keeps duplicates (strong multiset oracle)
// =====================================================================

#[test]
fn union_all_keeps_duplicates() {
    // X-nodes g ∈ {1, 1, 2}. Each arm scans X → [1,1,2]; UNION ALL of
    // the two arms keeps every duplicate → multiset {1×4, 2×2}.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_X).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_X).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3, LABEL_X).with_property("g", Value::Integer(2)),
        );
    let rows = run(
        "MATCH (n:X) RETURN n.g AS g UNION ALL MATCH (m:X) RETURN m.g AS g",
        &s,
        &cat(),
    );
    assert_eq!(rows.len(), 6, "UNION ALL keeps all duplicate rows");
    assert_eq!(sorted_ints(&rows), vec![1, 1, 1, 1, 2, 2]);
}

#[test]
fn union_all_concatenates_arms_in_source_order() {
    // arm 0: A-nodes (v=10); arm 1: B-nodes (v=20). Concat = [10, 20].
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_A).with_property("v", Value::Integer(10)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_B).with_property("v", Value::Integer(20)),
        );
    let rows = run(
        "MATCH (a:A) RETURN a.v AS x UNION ALL MATCH (b:B) RETURN b.v AS x",
        &s,
        &cat(),
    );
    assert_eq!(
        ordered_ints(&rows),
        vec![10, 20],
        "arm-0 rows precede arm-1"
    );
}

// =====================================================================
// Contract item 1 (RC-2) — the post-union ORDER BY / LIMIT binds the
// WHOLE union, not the last arm
// =====================================================================

#[test]
fn tail_binds_whole_union_parse_structure() {
    // The tail must factor OUT of the arms onto the union: each arm's
    // clause list carries NO Tail{OrderBy,Skip,Limit} clause; the
    // UnionQuery::tail carries them.
    let stmt = parse(
        "MATCH (a:A) RETURN a.v AS x UNION ALL MATCH (b:B) RETURN b.v AS x ORDER BY x LIMIT 5",
    )
    .expect("parse");
    let Statement::Union(u) = stmt else {
        panic!("expected Statement::Union, got {stmt:?}");
    };
    assert_eq!(u.arms.len(), 2);
    assert_eq!(u.all, vec![true], "single UNION ALL boundary");
    // Tail bound to the whole union.
    assert_eq!(u.tail.order_by.len(), 1, "ORDER BY on the union");
    assert!(u.tail.limit.is_some(), "LIMIT on the union");
    // NEITHER arm carries a tail clause (the RC-2 defect would leave
    // ORDER BY / LIMIT inside the last arm).
    for arm in &u.arms {
        for c in &arm.clauses {
            assert!(
                !matches!(
                    c,
                    Clause::TailOrderBy(_) | Clause::TailSkip(_) | Clause::TailLimit(_)
                ),
                "no tail clause may live inside a union arm",
            );
        }
    }
}

#[test]
fn tail_binds_whole_union_executable_limit_spans_arm_boundary() {
    // arm 0: ONE A-node v = 10; arm 1: TWO B-nodes v ∈ {20, 30}
    // (NodeId-ascending scan order). UNION ALL (concat) = [10, 20, 30].
    // `LIMIT 2` over the COMBINED stream = [10, 20] — a window that
    // SPANS the arm-0/arm-1 boundary (arm-0's only row + arm-1's first
    // row) AND drops arm-1's `30`. This is ONLY possible if LIMIT binds
    // the WHOLE union (the RC-2 fix). Had the tail mis-bound to the
    // last arm (the RC-2 defect: `arm0 UNION ALL (arm1 LIMIT 2)`), the
    // result would be [10, 20, 30] (3 rows) — arm-1's two rows pass the
    // per-arm LIMIT and arm-0's row is un-limited.
    //
    // (SKIP + ORDER BY over a union are intentionally NOT executed
    // here: the v1.0 executor surfaces NotImplemented for SKIP /
    // dynamic-LIMIT, and resolves ORDER BY keys only against
    // PASS-THROUGH output columns — both pre-existing read-query-tail
    // limitations, documented in ADR-185. The ORDER-BY /
    // SKIP-binds-the-whole-union STRUCTURE is pinned by
    // `tail_binds_whole_union_parse_structure`.)
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_A).with_property("v", Value::Integer(10)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_B).with_property("v", Value::Integer(20)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3, LABEL_B).with_property("v", Value::Integer(30)),
        );
    let rows = run(
        "MATCH (a:A) RETURN a.v AS x UNION ALL MATCH (b:B) RETURN b.v AS x LIMIT 2",
        &s,
        &cat(),
    );
    assert_eq!(
        ordered_ints(&rows),
        vec![10, 20],
        "LIMIT applies to the COMBINED union (window spans the arm boundary), not the last arm",
    );
}

// =====================================================================
// Contract item 3 (Q2) — column-compatibility: by-name set-equality,
// order-independent; FAILs on mismatch (NOT a silent skip)
// =====================================================================

#[test]
fn column_mismatch_fails_at_bind() {
    // Different aliases ⇒ different column-name SETS ⇒ reject.
    let query = "MATCH (a:A) RETURN a.v AS p UNION ALL MATCH (b:B) RETURN b.v AS q";
    let stmt = parse(query).expect("parse");
    let err = BindingVisitor::bind(&stmt, query, &cat())
        .expect_err("column mismatch MUST fail at bind (not a silent skip)");
    assert!(
        err.iter()
            .any(|e| matches!(e, BindingError::UnionColumnMismatch { .. })),
        "expected UnionColumnMismatch, got {err:?}",
    );
}

#[test]
fn same_alias_diff_expr_compatible() {
    // openCypher v9 §8 (TCK): the column NAME set is what matters, not
    // the projected expression — `... AS p UNION ALL ... AS p` is
    // compatible even though the underlying expressions differ.
    let query = "MATCH (a:A) RETURN a.v AS p UNION ALL MATCH (b:B) RETURN b.w AS p";
    let stmt = parse(query).expect("parse");
    let bound = BindingVisitor::bind(&stmt, query, &cat());
    assert!(
        bound.is_ok(),
        "same alias `p` on both arms is compatible: {bound:?}"
    );
}

#[test]
fn column_order_independent_realignment() {
    // arm 0 projects [p, q]; arm 1 projects [q, p] (same NAME set,
    // different ORDER). §8 result columns follow arm-0's order, so the
    // executor must realign arm-1's columns. We assert column 0 is
    // always `p`'s value and column 1 is always `q`'s value across both
    // arms' rows.
    //   A-node: v=10 (→p), w=11 (→q)   B-node: v=20 (→p), w=21 (→q)
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_A)
                .with_property("v", Value::Integer(10))
                .with_property("w", Value::Integer(11)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_B)
                .with_property("v", Value::Integer(20))
                .with_property("w", Value::Integer(21)),
        );
    let rows = run(
        "MATCH (a:A) RETURN a.v AS p, a.w AS q \
         UNION ALL \
         MATCH (b:B) RETURN b.w AS q, b.v AS p",
        &s,
        &cat(),
    );
    // Expect row-for-A = [10, 11], row-for-B = [20, 21] — i.e. arm-1's
    // [q, p] source order was realigned to canonical [p, q].
    let as_pairs: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| match (&r[0], &r[1]) {
            (Value::Integer(p), Value::Integer(q)) => (*p, *q),
            other => panic!("expected two Integer columns, got {other:?}"),
        })
        .collect();
    assert!(
        as_pairs.contains(&(10, 11)),
        "A row realigned to (p=10, q=11): {as_pairs:?}"
    );
    assert!(
        as_pairs.contains(&(20, 21)),
        "B row realigned to (p=20, q=21): {as_pairs:?}"
    );
}

// =====================================================================
// TCK Union3 — mixing UNION and UNION ALL is rejected
// =====================================================================

#[test]
fn mixed_union_and_union_all_rejected() {
    let query = "MATCH (a:A) RETURN a.v AS x \
                 UNION \
                 MATCH (b:B) RETURN b.v AS x \
                 UNION ALL \
                 MATCH (c:X) RETURN c.v AS x";
    let stmt = parse(query).expect("parse");
    let err = BindingVisitor::bind(&stmt, query, &cat())
        .expect_err("mixing UNION and UNION ALL MUST be rejected (TCK InvalidClauseComposition)");
    assert!(
        err.iter()
            .any(|e| matches!(e, BindingError::UnionMixedSetOps { .. })),
        "expected UnionMixedSetOps, got {err:?}",
    );
}

// =====================================================================
// #649-A2 — bare UNION (distinct) now EXECUTES + dedups
// (openCypher v9 §8: `UNION` removes duplicate rows)
// =====================================================================

#[test]
fn bare_union_distinct_now_lowers_to_distinct_over_union() {
    // #649-A2 lifts the A1 lowering deferral: bare UNION binds +
    // type-checks + LOWERS cleanly to a Distinct-over-Union plan
    // (the PE FROZEN CONTRACT item 2 composition). Before A2 this
    // surfaced a structured LogicalPlanError deferral.
    use arcgraph_query::logical_plan::LogicalPlan;
    let query = "MATCH (a:A) RETURN a.v AS x UNION MATCH (b:B) RETURN b.v AS x";
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, &cat()).expect("bind (compat check passes)");
    TypeCheckVisitor::check(&mut bound, &cat()).expect("type-check");
    let plan = LogicalPlanLoweringVisitor::lower(&bound)
        .expect("bare UNION (distinct) now lowers (A2 lifts the deferral)");
    // The top of the plan is a Distinct WRAPPING a Union (dedup is the
    // standalone DistinctOp OVER UnionOp; NOT buried in UnionOp).
    let LogicalPlan::Distinct(d) = &plan else {
        panic!("bare UNION must lower to a Distinct at the top, got {plan:?}");
    };
    assert!(
        matches!(d.input.as_ref(), LogicalPlan::Union(_)),
        "Distinct must wrap a Union (Distinct-over-Union composition), got {:?}",
        d.input
    );
}

#[test]
fn union_all_still_lowers_to_bare_union_no_distinct() {
    // Honest contrast pin: UNION ALL must NOT gain a Distinct wrapper —
    // it stays a bare Union concat (keep duplicates). Guards against a
    // regression that over-dedups UNION ALL.
    use arcgraph_query::logical_plan::LogicalPlan;
    let query = "MATCH (a:A) RETURN a.v AS x UNION ALL MATCH (b:B) RETURN b.v AS x";
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, &cat()).expect("bind");
    TypeCheckVisitor::check(&mut bound, &cat()).expect("type-check");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    assert!(
        matches!(plan, LogicalPlan::Union(_)),
        "UNION ALL must stay a bare Union (no Distinct wrapper), got {plan:?}",
    );
}

#[test]
fn bare_union_distinct_dedups_across_arms() {
    // The conformance heart of #649-A2 (openCypher v9 §8 UNION removes
    // duplicates). X-nodes g ∈ {1, 1, 2}. Each arm scans X → multiset
    // [1,1,2]; the pre-dedup concat is [1,1,2,1,1,2] (6 rows, what
    // UNION ALL returns — see `union_all_keeps_duplicates`). Bare UNION
    // dedups the COMBINED stream to the SET {1, 2} → exactly 2 rows.
    //
    // STRONG oracle: this asserts BOTH the deduped multiset {1,2} AND
    // the exact row COUNT (==2). It FAILS loudly if bare UNION behaved
    // like UNION ALL (6 rows) — i.e. if the distinct path were not wired.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_X).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_X).with_property("g", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3, LABEL_X).with_property("g", Value::Integer(2)),
        );
    let rows = run(
        "MATCH (n:X) RETURN n.g AS g UNION MATCH (m:X) RETURN m.g AS g",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        2,
        "bare UNION dedups the combined stream to the SET {{1,2}} (NOT 6 rows like UNION ALL)",
    );
    assert_eq!(
        sorted_ints(&rows),
        vec![1, 2],
        "deduped values are {{1, 2}}"
    );
}

#[test]
fn bare_union_distinct_dedups_identical_rows_from_both_arms() {
    // arm 0: A-nodes v ∈ {10, 10} (duplicate within the arm); arm 1:
    // A-nodes again (same scan) → the concat is [10,10,10,10], all
    // identical. Bare UNION collapses to a SINGLE row [10]. This proves
    // dedup spans (a) within-arm dups AND (b) cross-arm dups.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_A).with_property("v", Value::Integer(10)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_A).with_property("v", Value::Integer(10)),
        );
    let rows = run(
        "MATCH (a:A) RETURN a.v AS x UNION MATCH (b:A) RETURN b.v AS x",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        1,
        "four identical [10] rows collapse to one under bare UNION",
    );
    assert_eq!(sorted_ints(&rows), vec![10]);
}

#[test]
fn bare_union_distinct_realigns_columns_before_dedup() {
    // §8 order-independence COMPOSES with dedup: arm 0 projects [p, q];
    // arm 1 projects [q, p] (same NAME set, different ORDER) over the
    // SAME node values. After realignment both arms yield the identical
    // canonical row (p=10, q=11), so bare UNION dedups them to ONE row.
    // Had realignment not happened before dedup, arm-1's row would be
    // (11, 10) ≠ (10, 11) and we'd wrongly keep 2 rows.
    let s = StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        node(1, LABEL_A)
            .with_property("v", Value::Integer(10))
            .with_property("w", Value::Integer(11)),
    );
    let rows = run(
        "MATCH (a:A) RETURN a.v AS p, a.w AS q \
         UNION \
         MATCH (b:A) RETURN b.w AS q, b.v AS p",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        1,
        "realigned-then-deduped: both arms' canonical row is (p=10, q=11) → 1 row",
    );
    match (&rows[0][0], &rows[0][1]) {
        (Value::Integer(p), Value::Integer(q)) => {
            assert_eq!(
                (*p, *q),
                (10, 11),
                "canonical column order is arm-0's [p, q]"
            );
        }
        other => panic!("expected two Integer columns, got {other:?}"),
    }
}

#[test]
fn bare_union_distinct_tail_limit_applies_post_dedup() {
    // The tail binds the WHOLE union AFTER dedup (§8): the plan nests
    // `Limit < Distinct < Union`, so `LIMIT 2` truncates the DEDUPED
    // row-set, not the pre-dedup concat.
    //
    // This fixture is chosen so the row COUNT alone distinguishes
    // tail-post-dedup from tail-pre-dedup (NIT-1 oracle strengthen): all
    // X-nodes carry the SAME value g=7, so the two X-scans concatenate to
    // 6 identical rows that dedup to a SINGLE distinct row {7}.
    //   - Correct (LIMIT over the deduped set): {7} has 1 row; `LIMIT 2`
    //     over a 1-row set yields exactly 1 row.
    //   - Broken (dedup not run, OR `LIMIT` applied to the pre-dedup
    //     concat): the 6-row concat truncated by `LIMIT 2` yields 2 rows.
    // The deduped distinct count (1) is strictly LESS than the LIMIT (2),
    // so the two regimes give DIFFERENT counts. Asserting `len() == 1`
    // therefore FAILS (got 2) if the tail mis-applies pre-dedup — unlike
    // the prior {1,1,2,3}/LIMIT 2 fixture, which yielded 2 rows either way
    // and so could not catch a dedup-disabled regression.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            node(1, LABEL_X).with_property("g", Value::Integer(7)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(2, LABEL_X).with_property("g", Value::Integer(7)),
        )
        .with_node(
            TenantId::DEFAULT,
            node(3, LABEL_X).with_property("g", Value::Integer(7)),
        );
    let rows = run(
        "MATCH (n:X) RETURN n.g AS g UNION MATCH (m:X) RETURN m.g AS g LIMIT 2",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        1,
        "LIMIT 2 over the DEDUPED union: dedup collapses 6 copies of g=7 to \
         the single distinct row {{7}}; LIMIT 2 over 1 row keeps 1. A pre-dedup \
         tail (or no dedup) would truncate the 6-row concat to 2 rows.",
    );
    match &rows[0][0] {
        Value::Integer(n) => assert_eq!(
            *n, 7,
            "the single surviving row is the deduped value g=7, got {n}"
        ),
        other => panic!("expected Integer, got {other:?}"),
    }
}
