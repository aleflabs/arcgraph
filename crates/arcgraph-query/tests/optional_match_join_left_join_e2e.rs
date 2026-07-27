//! #771 (CZ Wave-1 L-OPT) — OPTIONAL MATCH left-join correctness e2e.
//!
//! Regression oracle for the §2.3 silent-wrong bug: OPTIONAL MATCH over a
//! labeled-node-with-rel pattern `(c:Commit)-[:FIXES]->(i)` used to ALWAYS
//! null-coalesce — its right side lowers to a `Join(Scan(c), Expand(c→i))`,
//! which the pipeline's right-side builder did not handle, and the resulting
//! `NotImplemented` was silently swallowed into all-NULL by the per-row
//! factory's `unwrap_or_else(|_| EmptyOp)`. Every left-join query was
//! silently wrong (SWE "issues + fixing commit" reported no fixes; AML
//! "accounts + flag if any" reported no flags).
//!
//! These are EXACT-ROW oracles (the strong-oracle bar). The **B** test is the
//! discriminating left-join oracle: it FAILS on the old all-NULL behavior AND
//! on any fix that forgets to null-extend the non-matching row.
//!
//! Fixture (matches issue #771 + `cz_swe.py` S15):
//!   `(c2:Commit {sha:'bbb'})-[:FIXES]->(b1:Issue {id:'b1'})`, plus a
//!   `(b3:Issue {id:'b3'})` with no fixing commit.
//!
//! # ADR provenance
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH lowers to left-outer join.
//! - ADR-038 amendment-03 §TIER-1 GAP D — exec-time null-row emission.
//! - ADR-097 — Join algorithm dispatch (the right-side Join build mirrors it).

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

// Catalog interns in declared order: Issue→LabelId(1), Commit→LabelId(2),
// FIXES→TypeId(1), id→PropertyId(1), sha→PropertyId(2). (Confirmed by
// plan-dump during #771 diagnosis.)
fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Issue", "Commit"])
        .with_rel_types(["FIXES"])
        .with_properties(["id", "sha"])
}

// b1 = Issue#1 (fixed by c2), b3 = Issue#3 (no fixing commit),
// c2 = Commit#2 {sha:'bbb'}. FIXES edge: c2 -> b1.
fn substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("id", Value::String("b1".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(1)))
                .with_property("id", Value::String("b3".into())),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(2)))
                .with_property("sha", Value::String("bbb".into())),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(10),
                NodeId::new(2), // c2 (Commit) — the FIXES source
                NodeId::new(1), // b1 (Issue)  — the FIXES target
                Some(TypeId::new(1)),
            ),
        )
}

/// Render one `(i.id, c.sha)` row as `(String, Option<String>)`. A NULL `c`
/// (the OPTIONAL non-match) makes `c.sha` NULL per Cypher 3VL.
fn row_pair(row: &[Value]) -> (String, Option<String>) {
    let a = match &row[0] {
        Value::String(s) => s.clone(),
        other => panic!("col 0 (i.id) expected String, got {other:?}"),
    };
    let b = match &row[1] {
        Value::String(s) => Some(s.clone()),
        Value::Null => None,
        other => panic!("col 1 (c.sha) expected String|Null, got {other:?}"),
    };
    (a, b)
}

/// Execute + return rows sorted by `i.id` for order-independent oracles.
fn run_sorted(query: &str) -> Vec<(String, Option<String>)> {
    let cat = cat();
    let engine = QueryEngine::new(&cat);
    let result = engine
        .execute(query, &substrate())
        .unwrap_or_else(|e| panic!("execute failed for `{query}`: {e:?}"));
    let mut rows: Vec<(String, Option<String>)> =
        result.rows().iter().map(|r| row_pair(r)).collect();
    rows.sort();
    rows
}

// ---------------------------------------------------------------------
// CONTROL — plain (non-optional) multi-pattern MATCH proves the pattern
// `(c:Commit)-[:FIXES]->(i)` IS findable. Must stay passing.
// ---------------------------------------------------------------------
#[test]
fn control_plain_match_finds_the_fix() {
    let rows = run_sorted("MATCH (i:Issue) MATCH (c:Commit)-[:FIXES]->(i) RETURN i.id, c.sha");
    assert_eq!(
        rows,
        vec![("b1".to_string(), Some("bbb".to_string()))],
        "CONTROL: plain MATCH finds the FIXES edge (b3 has no fixing commit → dropped)"
    );
}

// ---------------------------------------------------------------------
// A — single-issue OPTIONAL MATCH returns the REAL match, not NULL.
// ---------------------------------------------------------------------
#[test]
fn a_optional_match_returns_real_match_not_null() {
    let rows = run_sorted(
        "MATCH (i:Issue {id:'b1'}) OPTIONAL MATCH (c:Commit)-[:FIXES]->(i) RETURN i.id, c.sha",
    );
    assert_eq!(
        rows,
        vec![("b1".to_string(), Some("bbb".to_string()))],
        "A: OPTIONAL MATCH must yield the real fix bbb (was silently null pre-#771)"
    );
}

// ---------------------------------------------------------------------
// B — THE discriminating left-join oracle: b1 matches, b3 null-extends.
// Fails on old all-NULL AND on a fix that fails to null-extend non-matches.
// ---------------------------------------------------------------------
#[test]
fn b_left_join_matches_and_null_extends() {
    let rows =
        run_sorted("MATCH (i:Issue) OPTIONAL MATCH (c:Commit)-[:FIXES]->(i) RETURN i.id, c.sha");
    assert_eq!(
        rows,
        vec![
            ("b1".to_string(), Some("bbb".to_string())),
            ("b3".to_string(), None),
        ],
        "B (left-join): b1 → bbb (real match), b3 → null (null-extension)"
    );
}

// ---------------------------------------------------------------------
// No-match null-extension — the OPTIONAL semantics that must still hold:
// a genuinely-unmatched left row gets NULL (not dropped, not errored).
// ---------------------------------------------------------------------
#[test]
fn no_match_null_extends() {
    let rows = run_sorted(
        "MATCH (i:Issue {id:'b3'}) OPTIONAL MATCH (c:Commit)-[:FIXES]->(i) RETURN i.id, c.sha",
    );
    assert_eq!(
        rows,
        vec![("b3".to_string(), None)],
        "no-match: b3 has no fixing commit → null-extended (correct OPTIONAL semantics)"
    );
}

// ---------------------------------------------------------------------
// Backward shape — `(i)<-[:FIXES]-(c:Commit)` where the seed `i` is the
// Expand's `from` (nested-Join lowering). Same left-join result as B.
// ---------------------------------------------------------------------
#[test]
fn backward_shape_left_join() {
    let rows =
        run_sorted("MATCH (i:Issue) OPTIONAL MATCH (i)<-[:FIXES]-(c:Commit) RETURN i.id, c.sha");
    assert_eq!(
        rows,
        vec![
            ("b1".to_string(), Some("bbb".to_string())),
            ("b3".to_string(), None),
        ],
        "backward `(i)<-[:FIXES]-(c:Commit)` is the same left-join as B"
    );
}

// ---------------------------------------------------------------------
// FAIL-LOUD pin (#771 part-1): a right-side shape the builder does NOT
// support must propagate NotImplemented at BUILD time — correct-or-loud,
// NEVER the prior swallowed all-NULL. The OPTIONAL MATCH right side
// supports Scan/Expand/Filter/Project/Join; here we hand it a `Limit`
// (which a real OPTIONAL pattern never lowers to) and pin that the build
// errors instead of silently constructing an all-null OptionalExpandOp.
// ---------------------------------------------------------------------
#[test]
fn unbuildable_right_side_fails_loud_not_silent() {
    use arcgraph_core::Lsn;
    use arcgraph_query::error::Span;
    use arcgraph_query::executor::{ExecutionError, Pipeline};
    use arcgraph_query::logical_plan::{
        JoinCondition, LogicalLeftOuterJoin, LogicalLimit, LogicalPlan, LogicalScan,
    };
    use arcgraph_query::semantic::bound_ast::BindingId;

    let scan = |var| {
        LogicalPlan::Scan(LogicalScan {
            label: None,
            var,
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        })
    };
    let plan = LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
        left: Box::new(scan(BindingId::new(0))),
        right: Box::new(LogicalPlan::Limit(LogicalLimit {
            input: Box::new(scan(BindingId::new(0))),
            count: 5,
            span: Span::point(1, 1),
        })),
        on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
        span: Span::point(1, 1),
    });
    let built = Pipeline::build(&plan);
    assert!(
        matches!(built, Err(ExecutionError::NotImplemented { .. })),
        "unbuildable OPTIONAL MATCH right side must fail LOUD at build time \
         (correct-or-loud per #771); silent all-NULL is forbidden. got: {built:?}"
    );
}
