//! #814 — multi-pattern Cartesian cross-product cardinality (HIGH,
//! §2.3 silent-wrong) end-to-end regression.
//!
//! # The bug (issue #814)
//!
//! `MATCH (a:D),(b:D) RETURN count(*)` over `N` distinct `:D` nodes
//! returned **`2000 + N`** rows instead of **`N²`** — and the full
//! materialization (`RETURN a.uid, b.uid`) agreed on the same wrong
//! value, so a customer got a silently, drastically undercounted result
//! with no error (the worst class — `count(*)` lied).
//!
//! # Root cause (NOT the planner; the executor)
//!
//! The logical plan was already correct: `MATCH (a),(b)` lowers to a
//! `LogicalJoin` with `SharedBindings([])` (Cartesian) which the picker
//! routes to a HashJoin whose `@CARTESIAN` sentinel bucket holds every
//! left row. The defect was in
//! [`arcgraph_query::executor`]'s `HashJoinOp::next_batch`: when the
//! output batch filled MID-`right_batch`, it did `return Ok(out)` and
//! DROPPED the still-unprocessed tail of the consumed right batch
//! (`Batch::into_rows()` is by-value). The next call pulled a FRESH
//! right batch, skipping those rows. For `N=50` the right scan returns
//! all 50 rows in one batch; the probe overflowed `BATCH_ROWS` (2048)
//! after ~41 right rows, so the trailing 9 vanished → `|left| × 41 =
//! 2050` (the `2000 + N` signature). The keyed `WHERE a.uid = b.uid`
//! path (Cartesian + Filter) lost rows the SAME way (returned 41/50).
//!
//! The fix (one change in `HashJoinOp::next_batch`) mirrors the sibling
//! `ExpandOp` discipline: process the ENTIRE right batch (emit-or-spill
//! every joined row — overflow is preserved in `spillover`, never
//! dropped), and break the outer loop only at its top when the output
//! is full.
//!
//! # Oracles (exact cardinality — the discriminating test)
//!
//! Small `N` is used so `N²` / `N³` fit well under the executor's
//! cap-free spillover bound (`SPILLOVER_MAX_ROWS = 131072`) and the
//! oracle is exact + deterministic. The `count(*) == N²` (RED as
//! `2000 + N` against the pre-fix executor) is the load-bearing
//! discriminator; the single-anchor + keyed-join cases pin the
//! no-regression contract.
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md` +
//! the doctrine §3 strong-oracle rule: this exercises the real
//! parse→bind→typecheck→lower→pick→execute pipeline via `QueryEngine`.

use std::collections::BTreeSet;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::semantic::{CatalogProvider, StubCatalogProvider};

/// `n` distinct `:D` nodes (`LabelId(1)`), `NodeId(1..=n)`, each with a
/// distinct `uid` property `0..n`.
fn substrate_d(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for k in 0..n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(k + 1), Some(LabelId::new(1)))
                .with_property("uid", Value::Integer(k as i64)),
        );
    }
    s
}

fn cat_d() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["D"])
        .with_properties(["uid"])
}

fn count_star<C: CatalogProvider>(
    engine: &QueryEngine<'_, C>,
    s: &StubExecutorSubstrate,
    q: &str,
) -> i64 {
    let r = engine
        .execute(q, s)
        .unwrap_or_else(|e| panic!("execute {q:?}: {e:?}"));
    assert_eq!(
        r.rows().len(),
        1,
        "count(*) returns exactly one row for {q:?}"
    );
    match &r.rows()[0][0] {
        Value::Integer(i) => *i,
        other => panic!("count(*) not Integer for {q:?}: {other:?}"),
    }
}

/// **Load-bearing discriminator.** `MATCH (a:D),(b:D) RETURN count(*)`
/// over N distinct nodes is exactly `N²` (RED as `2000 + N` against the
/// pre-fix executor that dropped the overflowing right-batch tail).
#[test]
fn two_way_cartesian_count_is_n_squared() {
    let cat = cat_d();
    let engine = QueryEngine::new(&cat);
    for n in [5u64, 10] {
        let s = substrate_d(n);
        let got = count_star(&engine, &s, "MATCH (a:D),(b:D) RETURN count(*)");
        assert_eq!(
            got,
            (n * n) as i64,
            "N={n}: cartesian count(*) must be N²={}, not 2000+N={} (issue #814)",
            n * n,
            2000 + n
        );
    }
}

/// Materialization matches count: `RETURN a.uid, b.uid` yields exactly
/// `N²` rows, every (a, b) pair present, no dupes, no missing.
#[test]
fn two_way_cartesian_materializes_full_product() {
    let n: u64 = 10;
    let s = substrate_d(n);
    let cat = cat_d();
    let engine = QueryEngine::new(&cat);
    let r = engine
        .execute("MATCH (a:D),(b:D) RETURN a.uid, b.uid", &s)
        .expect("materialize cartesian");
    assert_eq!(r.rows().len(), (n * n) as usize, "N² materialized rows");

    let mut pairs: BTreeSet<(i64, i64)> = BTreeSet::new();
    for row in r.rows() {
        let a = match &row[0] {
            Value::Integer(i) => *i,
            other => panic!("a.uid not Integer: {other:?}"),
        };
        let b = match &row[1] {
            Value::Integer(i) => *i,
            other => panic!("b.uid not Integer: {other:?}"),
        };
        assert!(pairs.insert((a, b)), "duplicate pair ({a}, {b})");
    }
    assert_eq!(pairs.len(), (n * n) as usize, "no dupes");
    for a in 0..n as i64 {
        for b in 0..n as i64 {
            assert!(pairs.contains(&(a, b)), "missing pair ({a}, {b})");
        }
    }
}

/// 3-way cross product generalizes: `MATCH (a:D),(b:D),(c:D) RETURN
/// count(*)` is exactly `N³` (the pre-fix bug returned `2000 + N` here
/// too — identical to the 2-way value, proving the product collapsed).
#[test]
fn three_way_cartesian_count_is_n_cubed() {
    let n: u64 = 5; // 5³ = 125 << spillover bound
    let s = substrate_d(n);
    let cat = cat_d();
    let engine = QueryEngine::new(&cat);
    let got = count_star(&engine, &s, "MATCH (a:D),(b:D),(c:D) RETURN count(*)");
    assert_eq!(
        got,
        (n * n * n) as i64,
        "N={n}: 3-way cartesian count(*) must be N³={}, not 2000+N={}",
        n * n * n,
        2000 + n
    );
}

/// No-regression: single-anchor `MATCH (a:D {uid:0}),(b:D)` enumerates
/// all N distinct b → count(*) = N. (The issue confirms pinning one
/// side was always correct; this must STAY correct.)
#[test]
fn single_anchor_count_stays_n() {
    let cat = cat_d();
    let engine = QueryEngine::new(&cat);
    for n in [5u64, 10] {
        let s = substrate_d(n);
        let got = count_star(&engine, &s, "MATCH (a:D {uid:0}),(b:D) RETURN count(*)");
        assert_eq!(got, n as i64, "N={n}: single-anchor count(*) must be N={n}");
    }
}

/// No-regression: a keyed equi-join `MATCH (a:D),(b:D) WHERE a.uid =
/// b.uid` matches each uid to itself → count(*) = N. (Pre-fix this
/// returned 41 of 50 via the SAME dropped-tail bug — Cartesian + Filter
/// overflowed and lost the tail. Must now be exactly N.)
#[test]
fn keyed_join_count_stays_n() {
    let cat = cat_d();
    let engine = QueryEngine::new(&cat);
    for n in [5u64, 10] {
        let s = substrate_d(n);
        let got = count_star(
            &engine,
            &s,
            "MATCH (a:D),(b:D) WHERE a.uid = b.uid RETURN count(*)",
        );
        assert_eq!(got, n as i64, "N={n}: keyed-join count(*) must be N={n}");
    }
}

/// Larger-N exact pin (still under the cap-free spillover bound): N=50
/// → 2-way = 2500, materialization = 2500, keyed = 50. This is the
/// exact N from the issue's repro table where the bug yielded 2050
/// (2-way) and 41 (keyed). Hermetic, deterministic.
#[test]
fn issue_repro_n50_exact() {
    let n: u64 = 50; // 50² = 2500 < SPILLOVER_MAX_ROWS (131072)
    let s = substrate_d(n);
    let cat = cat_d();
    let engine = QueryEngine::new(&cat);

    let two = count_star(&engine, &s, "MATCH (a:D),(b:D) RETURN count(*)");
    assert_eq!(two, 2500, "N=50 cartesian count(*) = 2500 (bug: 2050)");

    let mat = engine
        .execute("MATCH (a:D),(b:D) RETURN a.uid, b.uid", &s)
        .expect("materialize");
    assert_eq!(
        mat.rows().len(),
        2500,
        "N=50 materialized = 2500 (bug: 2050)"
    );
    let distinct_b: BTreeSet<i64> = mat
        .rows()
        .iter()
        .map(|r| match &r[1] {
            Value::Integer(i) => *i,
            other => panic!("b.uid not Integer: {other:?}"),
        })
        .collect();
    assert_eq!(
        distinct_b.len() as u64,
        n,
        "all 50 distinct b survive (bug: 41)"
    );

    let keyed = count_star(
        &engine,
        &s,
        "MATCH (a:D),(b:D) WHERE a.uid = b.uid RETURN count(*)",
    );
    assert_eq!(keyed, 50, "N=50 keyed-join count(*) = 50 (bug: 41)");
}
