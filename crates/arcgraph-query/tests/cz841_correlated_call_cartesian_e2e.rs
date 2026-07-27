//! #841 — correlated `CALL { WITH a … }` subquery silently executed as a
//! CARTESIAN (×|outer| row inflation + corrupted subquery-internal
//! aggregates) end-to-end regression. HIGH / silent-wrong-result, found
//! by CZ; treated with data-loss-class rigor.
//!
//! # The bug (issue #841)
//!
//! A correlated `CALL { WITH a MATCH (a)-[:SENT]->(b) … }` returned the
//! correct logical rows **multiplied by `|outer rows|`** — the
//! dependent-join was lowered as a CARTESIAN: the importing-`WITH a`
//! RE-DECLARES `a` with a FRESH binding id (#746 `output_id`), so the
//! body's `MATCH (a)` re-references that PROJECTED id while the
//! `CorrelationSeed` carries the OUTER `a`. Both row count AND a
//! subquery-internal `count(b)` were silently wrong (a1 with one out-edge
//! reported `count = |Account|`, not 1) — the dangerous class: no error.
//!
//! # Root cause (the LOWERING — NOT CallOp, NOT the planner)
//!
//! `lower_match` derives a correlated body's equi-join key from
//! `shared_bindings(prev, pattern)`. `collect_bindings(Project)` recursed
//! into the Project's INPUT instead of reporting its OUTPUT schema (the
//! projected `output_id`s, per `ProjectOp::derive_schema`). A `WITH a`
//! renames `a` (id 0 → 1), so the seed-rooted `Project` reported the
//! PRE-projection id {0} while the body pattern carried the projected id
//! {1}: `shared_bindings = ∅` → an EMPTY join key → a silent CARTESIAN of
//! the seed (one driving row) with the body's full re-scan. `Aggregate`
//! had the identical latent defect (its output schema is its group-by +
//! aggregation `output_id`s). The fix reports each renaming operator's
//! true OUTPUT schema in `collect_bindings` (one locus,
//! `logical_plan/lowering.rs`).
//!
//! Because the defect is in the GENERIC join-key derivation, it also
//! silently broke the NON-`CALL` re-reference form
//! `MATCH (a) WITH a MATCH (a)-[…]->(b)` (a plain cartesian, no subquery).
//! Both forms are pinned below.
//!
//! # Oracles (doctrine §3 — strong `==`, the make-or-break for silent-
//! correctness). The CRITICAL oracle is the per-driving-row `count(b)`
//! aggregate `[a1→1, a2→1, a3→0]`: it proves the correlation BINDS (each
//! driving row's real out-degree), not merely that a dedupe masked the
//! row-count inflation. RED-on-revert was captured by reverting both
//! fixes against these exact tests (pre-fix: 6 rows / count(*)=6 /
//! per-driving count == |Account| = 3, not 1).
//!
//! Full real pipeline (parse→bind→typecheck→lower→pick→execute) via
//! `QueryEngine::execute` (ADR-133 §D-4 Query-class active verification).

use std::collections::BTreeMap;

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

const ACCOUNT: u32 = 1; // first label ⇒ LabelId::new(1)
const SENT: u32 = 1; // first rel-type ⇒ TypeId::new(1)

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Account"])
        .with_rel_types(["SENT"])
        .with_properties(["id"])
}

/// `id`-tagged Account node (`id` property == NodeId, for stable oracles).
fn acct(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(ACCOUNT)))
        .with_property("id", Value::Integer(id as i64))
}

fn sent(rel_id: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(rel_id),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(SENT)),
    )
}

/// The bug-report fixture EXACTLY: 3 Accounts a1,a2,a3; SENT edges
/// a1→a2, a2→a3. Out-degrees: a1=1, a2=1, a3=0.
fn graph_3() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, acct(1))
        .with_node(TenantId::DEFAULT, acct(2))
        .with_node(TenantId::DEFAULT, acct(3))
        .with_edge(TenantId::DEFAULT, sent(100, 1, 2))
        .with_edge(TenantId::DEFAULT, sent(101, 2, 3))
}

/// 4 Accounts a1..a4; SENT edges a1→a2, a2→a3 (a3, a4 have out-degree 0).
/// Used to show the multiplier is NOT data-dependent (×|outer|=×4 pre-fix).
fn graph_4() -> StubExecutorSubstrate {
    graph_3().with_node(TenantId::DEFAULT, acct(4))
}

fn run(s: &StubExecutorSubstrate, q: &str) -> Vec<Vec<Value>> {
    let cat = cat();
    let engine = QueryEngine::new(&cat);
    engine
        .execute(q, s)
        .unwrap_or_else(|e| panic!("execute {q:?}: {e:?}"))
        .rows()
        .to_vec()
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(i) => *i,
        other => panic!("expected Integer, got {other:?}"),
    }
}

/// Sorted `(col0_int, col1_int)` pairs — for `RETURN a.id, b.id` shapes.
fn int_pairs(rows: &[Vec<Value>]) -> Vec<(i64, i64)> {
    let mut p: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "expected 2 columns, got {}", r.len());
            (int(&r[0]), int(&r[1]))
        })
        .collect();
    p.sort_unstable();
    p
}

// =====================================================================
// CONTROL — the plain join the correlated CALL{} must logically equal.
// =====================================================================

#[test]
fn control_plain_join_is_two_rows() {
    // The non-correlated reference result: (a1,a2),(a2,a3). The
    // correlated CALL{} forms below must produce the IDENTICAL set.
    let s = graph_3();
    let rows = run(&s, "MATCH (a:Account)-[:SENT]->(b) RETURN a.id, b.id");
    assert_eq!(int_pairs(&rows), vec![(1, 2), (2, 3)], "control = 2 rows");
}

// =====================================================================
// #841 CORE — correlated CALL{} with importing-WITH binds per-driving-row.
// =====================================================================

#[test]
fn correlated_call_with_import_is_not_cartesian() {
    // THE #841 repro. Pre-fix: 6 rows — each control pair ×|Account|=3
    // (the CARTESIAN). Post-fix: the 2 control rows, exactly.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN b } RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "correlated CALL{{}} == control (2 rows), NOT ×|outer|=6"
    );
}

#[test]
fn correlated_call_count_star_is_two_not_six() {
    // The silent-wrong `count(*)` (the worst class — the aggregate lied).
    // Pre-fix: 6 (= 2 control rows × |Account|=3). Post-fix: 2.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN b } RETURN count(*)",
    );
    assert_eq!(rows.len(), 1, "count(*) is a single row");
    assert_eq!(int(&rows[0][0]), 2, "count(*) == 2 (pre-fix bug: 6)");
}

#[test]
fn correlated_call_subquery_internal_aggregate_binds_per_driving_row() {
    // *** THE CRITICAL ORACLE (doctrine §3). ***
    //
    // A subquery-INTERNAL `count(b)` per driving row must equal that
    // driving row's REAL out-degree: a1→1, a2→1, a3→0. Pre-fix
    // (captured RED-on-revert) it reported a1→3, a2→3, a3→0 — i.e.
    // count == |Account|=3 for any driving row WITH an out-edge: the DP
    // derived a `Cartesian` (empty join key) between the correlation
    // seed's projected `a` and the body's independent re-`Scan` of all
    // accounts (the optimized plan shows `condition: Cartesian` over
    // `Project(WITH a)/CorrelationSeed` × `Scan(a')`), so the count
    // inflated to |Account|. This oracle proves the correlation BINDS
    // (real out-degree), not merely that the row count was de-duplicated:
    // a dedupe fix would still report the corrupted count. a3
    // (out-degree 0) is PRESERVED with count 0 (the openCypher
    // empty-aggregate identity row — ADR-192 D-8), NOT dropped.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN count(b) AS c } RETURN a.id, c",
    );
    let got: BTreeMap<i64, i64> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "schema = [a.id, c]");
            (int(&r[0]), int(&r[1]))
        })
        .collect();
    assert_eq!(
        got,
        BTreeMap::from([(1, 1), (2, 1), (3, 0)]),
        "per-driving-row out-degree: a1→1, a2→1, a3→0 (pre-fix: a1→3,a2→3 = |Account| — the corrupted aggregate)"
    );
}

#[test]
fn correlated_call_outer_where_reduces_driving_rows() {
    // An outer WHERE filtering to ONE driving row (a1) yields exactly its
    // ONE correlated result (a1,a2). Pre-fix (captured RED-on-revert): 3
    // rows — the single driving a1 was still inflated ×|Account|=3 (the
    // body re-scanned the whole graph), proving the filter does not rein
    // in the cartesian. Post-fix: exactly 1.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) WHERE a.id = 1 CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN b } RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2)],
        "outer WHERE → 1 driving row → 1 correlated result, NOT ×|outer|"
    );
}

#[test]
fn correlated_call_multiplier_is_outer_size_not_data() {
    // Multiplier is |outer rows|, not an edge count: growing the graph to
    // 4 Accounts (a4 isolated, NO new edges) left the CORRECT result
    // unchanged (still (a1,a2),(a2,a3)) but pre-fix inflated it ×4 (8
    // rows). Post-fix the extra isolated driving row contributes nothing.
    let s = graph_4();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN b } RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "adding an isolated 4th account does NOT inflate the result (pre-fix: 8 rows)"
    );
}

#[test]
fn correlated_call_aggregate_on_4_accounts_no_outer_inflation() {
    // The aggregate oracle on graph_4: a1→1, a2→1, a3→0, a4→0. Pre-fix a1
    // reported |Account|=4 (the bug-report's "[a1,4]" signature). Strong
    // proof the per-driving count tracks real out-degree, not |outer|.
    let s = graph_4();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN count(b) AS c } RETURN a.id, c",
    );
    let got: BTreeMap<i64, i64> = rows.iter().map(|r| (int(&r[0]), int(&r[1]))).collect();
    assert_eq!(
        got,
        BTreeMap::from([(1, 1), (2, 1), (3, 0), (4, 0)]),
        "a1→1 (NOT 4=|Account|), a2→1, a3→0, a4→0"
    );
}

// =====================================================================
// #841 — the implicit-import (no-WITH) form must STILL be correct
// (no-regression: the pre-existing path the bug did NOT touch).
// =====================================================================

#[test]
fn implicit_import_no_with_still_correct() {
    // The Cypher-25 implicit import (NO `WITH a`) — already correct on
    // main (`call_subquery_e2e::test2`). Pinned here so the fix is proven
    // to leave it identical to the WITH form (both == control).
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { MATCH (a)-[:SENT]->(b) RETURN b } RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "implicit form == control"
    );
}

// =====================================================================
// SISTER SHAPES (the same root cause — swept per the spawn brief).
// =====================================================================

#[test]
fn nested_call_with_inner_with_binds() {
    // Nested CALL{} where BOTH levels open with `WITH a` — the
    // correlation must compose across levels (each WITH rename surfaced
    // its output id). Pre-fix: 6 (cartesian). Post-fix: the 2 control rows.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) CALL { WITH a CALL { WITH a MATCH (a)-[:SENT]->(b) RETURN b } RETURN b } RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "nested correlated CALL{{}} with inner WITH == control"
    );
}

#[test]
fn non_call_with_rename_then_match_is_not_cartesian() {
    // THE BROADER FIX: the SAME root cause silently broke the plain
    // (no-subquery) re-reference `MATCH (a) WITH a MATCH (a)-[…]->(b)`.
    // Pre-fix: 6 (full cartesian — `WITH a` renamed `a`, the 2nd MATCH's
    // `a` joined on an EMPTY key). Post-fix: the 2 control rows.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) WITH a MATCH (a)-[:SENT]->(b) RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "WITH-a rename then re-MATCH (no CALL) == control, NOT cartesian"
    );
}

#[test]
fn non_call_with_chain_double_rename_then_match() {
    // A WITH-chain that renames twice (`a → a2 → a3`) then re-MATCHes the
    // final name. Each rename surfaces a fresh output id; the join key
    // must track the LAST one. Pre-fix: 6 (cartesian). Post-fix: 2.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) WITH a AS a2 WITH a2 AS a3 MATCH (a3)-[:SENT]->(b) RETURN a3.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "double WITH-rename == control"
    );
}

#[test]
fn non_call_with_aggregate_groupkey_then_match() {
    // The Aggregate-arm sister: `WITH a, count(a) AS n` groups by `a`
    // (renaming it via the group-by output id) then re-MATCHes `a`.
    // Pre-fix: 6 (the `collect_bindings(Aggregate)` input-recursion gave
    // an empty join key). Post-fix: 2.
    let s = graph_3();
    let rows = run(
        &s,
        "MATCH (a:Account) WITH a, count(a) AS n MATCH (a)-[:SENT]->(b) RETURN a.id, b.id",
    );
    assert_eq!(
        int_pairs(&rows),
        vec![(1, 2), (2, 3)],
        "WITH-aggregate group-key then re-MATCH == control"
    );
}
