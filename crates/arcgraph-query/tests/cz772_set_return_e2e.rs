//! **#772 (Customer-Zero AML; umbrella #649/#624) — `MATCH … SET … RETURN …`
//! silently dropped its projection rows (HIGH, §2.3 silent-wrong),
//! END-TO-END.**
//!
//! # The bug this pins
//!
//! `MATCH (a:Account {id:'a1'}) SET a.flagged = true RETURN a.id` persisted
//! the write correctly but returned **0 rows** instead of `[['a1']]`. The
//! universal "update a record and return it" idiom (AML: "flag this account
//! and return it for the SAR queue") yielded empty despite the flag landing.
//! Silent-wrong is the worst correctness class — a GA correctness gate.
//!
//! # The mechanism (verified on main @ 78cd01a5)
//!
//! A [`arcgraph_query::executor::ops::SetOp`] defaults to **terminal**: a
//! RETURN-less `SET …` drains its input rows and emits 0 result rows (the
//! openCypher v9 / ADR-149/150 §D / ADR-182 terminal-write contract). The
//! build flips a SET/REMOVE to **stacked** (pass-through) only when its
//! parent is ANOTHER write-op (the `SET … SET …` / `SET … REMOVE …` chain,
//! the #709 fix — `Pipeline::build`'s `Set`/`Remove` arms call
//! `mark_writeop_input_stacked` on their child).
//!
//! `SET … RETURN …` lowers to `Project(Set(…))`. Pre-fix, the `Project`
//! build arm did NOT flip its `Set`/`Remove` child to stacked, so the SET
//! stayed terminal, drained its rows, and the RETURN (`Project`) projected
//! over 0 rows → `[]`. (Contrast MERGE…RETURN / CREATE…RETURN, which emit
//! their row — confirming the drop was specific to SET/REMOVE terminal-drain
//! under a `Project`. The CREATE…RETURN control here proves that contrast.)
//!
//! # The fix
//!
//! `Pipeline::build`'s `Project` arm now flips a `Set`/`Remove` INPUT to
//! stacked — the SAME `mark_writeop_input_stacked` the `Set`/`Remove` arms
//! use — so the write-op passes its mutated rows through to the RETURN. A
//! companion in-row-view mirror on the stacked SET/REMOVE path (set.rs,
//! RC-2 per ADR-151-amendment-01 §D-2 — the SAME pattern MERGE's
//! RETURN-after-MERGE already uses) keeps the projected post-mutation
//! property values in lock-step with the substrate, so `SET a.x = 99
//! RETURN a.x` returns `99`, not stale `NULL`.
//!
//! The terminal RETURN-less contract is PRESERVED: a bare `SET …` (no
//! RETURN, no `Project`) stays terminal → 0 rows. Only the
//! `Project`-over-`Set/Remove` case flips the CHILD.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every positive assertion drives a REAL ArcQL query through the FULL
//! pipeline (`parse → bind → type-check → cross-substrate → lower →
//! Pipeline::build → execute`, via `execute_with_context` — the exact path
//! the production `StorageRawQueryExecutor` drives, exercising the
//! build-time terminal/stacked discriminator rather than bypassing it)
//! against a data-bearing substrate, asserting the EXACT result rows + the
//! EXACT persisted property bag. The oracle is the #772 `cz_micro.json`
//! repro + the openCypher SET/REMOVE row-emission semantics.
//!
//! # Fixtures via CREATE (Stub fidelity)
//!
//! The fixture node is seeded with a `CREATE` query (not `with_node`): the
//! Stub's `create_node` initializes the per-node `node_properties` sidecar
//! with the full bag, so a subsequent `SET`/`REMOVE` MERGES into it and the
//! production read path (`scan_nodes`, which replaces the scanned bag with
//! the sidecar) preserves the un-mutated keys. (A `with_node`-prebaked node
//! does NOT seed the sidecar, so a prebaked-then-SET node loses its original
//! keys on re-scan — a Stub test-fidelity quirk orthogonal to #772's
//! row-emission fix; the `set_then_match_by_property_smoke` convention seeds
//! via CREATE for the same reason.) Catalog + substrate agree on the
//! `Account` label id (1024 — the Stub's `create_node` interns at 1024+).

use std::collections::BTreeMap;

use arcgraph_core::{LabelId, Lsn, TenantId};
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, execute_with_context};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

/// The Stub interns `create_node` labels at 1024+; the catalog is pinned to
/// the SAME id so `MATCH (a:Account)` (lowered to `Scan{Some(1024)}`) finds
/// the CREATE-introduced fixture node.
const ACCOUNT: u32 = 1024;

/// Catalog with `Account` interned at the Stub's create-side label id.
fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new().with_label_id("Account", LabelId::new(ACCOUNT))
}

/// `parse → bind → typecheck → cross-substrate → lower`. Mirrors the
/// `writeop_terminal_vs_stacked_pin` / `set_then_match_by_property_smoke`
/// harness so SET/REMOVE/CREATE write-ops lower correctly.
fn lower(query: &str, c: &StubCatalogProvider) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, c).expect("typecheck OK");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// Full pipeline → result rows through the REAL driver (`Pipeline::build`
/// inside `execute_with_context`). A fresh context per call reads the
/// latest substrate state (read-your-writes across statements).
fn run(query: &str, c: &StubCatalogProvider, s: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    let plan = lower(query, c);
    let ctx = ExecutionContext::new(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO);
    execute_with_context(&plan, s, &ctx).expect("execute OK")
}

/// Seed `(a1:Account {id:'a1', balance:5000})` via CREATE (full sidecar).
fn fresh() -> (StubCatalogProvider, StubExecutorSubstrate) {
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let _ = run("CREATE (a:Account {id: 'a1', balance: 5000})", &c, &s);
    (c, s)
}

/// Seed `(a1:Account {id:'a1', balance:5000, flagged:true})` for REMOVE.
fn fresh_flagged() -> (StubCatalogProvider, StubExecutorSubstrate) {
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let _ = run(
        "CREATE (a:Account {id: 'a1', balance: 5000, flagged: true})",
        &c,
        &s,
    );
    (c, s)
}

/// Seed N Accounts (`a1`..`aN`) via CREATE (full sidecar each), every node
/// `balance: 5000`. The multi-node fixture for the aggregate-over-write
/// oracles: `MATCH (a:Account) SET … RETURN count(a)` must fold over ALL N
/// mutated rows, so N > 1 distinguishes "counts the real matched set"
/// from "accidentally returns 1".
fn fresh_n(n: usize) -> (StubCatalogProvider, StubExecutorSubstrate) {
    let c = cat();
    let s = StubExecutorSubstrate::new();
    for i in 1..=n {
        let q = format!("CREATE (a:Account {{id: 'a{i}', balance: 5000}})");
        let _ = run(&q, &c, &s);
    }
    (c, s)
}

/// Count of `Account` nodes in the substrate via the production read path —
/// the persist-check oracle for the multi-node aggregate tests.
fn account_count(s: &StubExecutorSubstrate) -> usize {
    s.scan_nodes(TenantId::DEFAULT, Some(LabelId::new(ACCOUNT)), Lsn::MAX)
        .expect("scan_nodes OK")
        .len()
}

/// Current persisted bag of the (single) Account node via the production
/// read path (`scan_nodes` merges the post-SET/REMOVE sidecar).
fn account_bag(s: &StubExecutorSubstrate) -> BTreeMap<String, Value> {
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, Some(LabelId::new(ACCOUNT)), Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        nodes.len(),
        1,
        "fixture has exactly one Account node: {nodes:?}"
    );
    nodes[0].node.properties.clone()
}

fn sval(s: &str) -> Value {
    Value::String(s.into())
}

// ====================================================================
// CONTROL A — the read works (sanity: the fixture + match + project path
// is sound, isolating the bug to the SET-under-Project drop).
// ====================================================================

#[test]
fn control_read_returns_id() {
    let (c, s) = fresh();
    let rows = run("MATCH (a:Account {id: 'a1'}) RETURN a.id", &c, &s);
    assert_eq!(
        rows,
        vec![vec![sval("a1")]],
        "CONTROL: plain MATCH … RETURN a.id returns the row"
    );
}

// ====================================================================
// CONTROL B — CREATE … RETURN works (the contrast: CreateOp emits its
// row, so RETURN-over-CREATE was never dropped — unlike SET/REMOVE).
// ====================================================================

#[test]
fn control_create_return_returns_row() {
    let c = cat();
    let s = StubExecutorSubstrate::new();
    let rows = run(
        "CREATE (a:Account {id: 'a2', balance: 10}) RETURN a.id",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![sval("a2")]],
        "CONTROL: CREATE … RETURN a.id emits the created row (contrast to SET)"
    );
}

// ====================================================================
// A — THE DISCRIMINATING ORACLE. SET … RETURN returns the projected row
// (was `[]` on the bug). This test goes RED (`[]`) on the pre-fix code.
// ====================================================================

#[test]
fn set_return_returns_projected_row() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.flagged = true RETURN a.id",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![sval("a1")]],
        "A (#772): SET … RETURN a.id must return [['a1']], not [] (silent-wrong)"
    );
}

// ====================================================================
// B — the write still persists (proof the SET fired despite emitting a
// row): production read-back + cross-statement read-your-writes query.
// ====================================================================

#[test]
fn set_return_still_persists_the_write() {
    let (c, s) = fresh();
    let _ = run(
        "MATCH (a:Account {id: 'a1'}) SET a.flagged = true RETURN a.id",
        &c,
        &s,
    );
    // Production read-back: the flag landed AND the original keys survive.
    let persisted = account_bag(&s);
    assert_eq!(
        persisted.get("flagged"),
        Some(&Value::Boolean(true)),
        "B: SET a.flagged = true persisted to the substrate"
    );
    assert_eq!(
        persisted.get("id"),
        Some(&sval("a1")),
        "B: the SET did not clobber the original id"
    );
    // Cross-statement read-your-writes (the task's exact B query form).
    let rows = run("MATCH (a:Account {id: 'a1'}) RETURN a.flagged", &c, &s);
    assert_eq!(
        rows,
        vec![vec![Value::Boolean(true)]],
        "B: a later MATCH … RETURN a.flagged observes the persisted flag"
    );
}

// ====================================================================
// C — multi-column RETURN reading the JUST-SET property. This is the
// in-row-view-staleness oracle: `a.risk` must read 99, not stale NULL.
// ====================================================================

#[test]
fn set_return_multicol_reads_set_property() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.risk = 99 RETURN a.id, a.risk",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![sval("a1"), Value::Integer(99)]],
        "C: SET a.risk = 99 RETURN a.id, a.risk must return [['a1', 99]] \
         (the projected a.risk reads the post-SET value, not stale NULL)"
    );
    assert_eq!(
        account_bag(&s).get("risk"),
        Some(&Value::Integer(99)),
        "C: a.risk persisted to the substrate"
    );
}

// ====================================================================
// TERMINAL RETURN-less — MUST stay 0 rows (the openCypher contract). The
// guard that the fix does NOT over-reach: a bare SET (no RETURN, no
// Project) still drains + emits nothing, while still persisting.
// ====================================================================

#[test]
fn terminal_set_returnless_stays_zero_rows_and_persists() {
    let (c, s) = fresh();
    let rows = run("MATCH (a:Account {id: 'a1'}) SET a.x = 1", &c, &s);
    assert_eq!(
        rows.len(),
        0,
        "TERMINAL: bare SET (no RETURN) must emit 0 rows, got {rows:?}"
    );
    assert_eq!(
        account_bag(&s).get("x"),
        Some(&Value::Integer(1)),
        "TERMINAL: the bare SET still persisted a.x = 1"
    );
}

// ====================================================================
// CHAINED — SET … SET … RETURN returns the row with BOTH mutations. The
// outer SET is the Project's child (flipped to stacked here); the inner
// SET is already stacked by the outer SET's arm — the flips compose.
// ====================================================================

#[test]
fn chained_set_set_return_carries_both_mutations() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.p = 1 SET a.q = 2 RETURN a.p, a.q",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(1), Value::Integer(2)]],
        "CHAINED: SET a.p=1 SET a.q=2 RETURN a.p, a.q must return [[1, 2]]"
    );
    let bag = account_bag(&s);
    assert_eq!(
        bag.get("p"),
        Some(&Value::Integer(1)),
        "chained p persisted"
    );
    assert_eq!(
        bag.get("q"),
        Some(&Value::Integer(2)),
        "chained q persisted"
    );
}

// ====================================================================
// WITH-over-SET — SET … WITH … RETURN. WITH also lowers to a `Project`,
// so the SAME Project-arm flip handles it. `a.w` reads the post-SET 7.
// ====================================================================

#[test]
fn with_over_set_return_reads_set_property() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.w = 7 WITH a RETURN a.w",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(7)]],
        "WITH-over-SET: SET a.w=7 WITH a RETURN a.w must return [[7]]"
    );
    assert_eq!(
        account_bag(&s).get("w"),
        Some(&Value::Integer(7)),
        "WITH-over-SET w persisted"
    );
}

// ====================================================================
// REMOVE-RETURN — the Remove arm parallel: REMOVE … RETURN returns rows.
// Fixture seeds `flagged` so the REMOVE clears a real property.
// ====================================================================

#[test]
fn remove_return_returns_projected_row() {
    let (c, s) = fresh_flagged();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) REMOVE a.flagged RETURN a.id",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![sval("a1")]],
        "REMOVE … RETURN a.id must return [['a1']] (Remove arm parallel to Set)"
    );
    assert_eq!(
        account_bag(&s).get("flagged"),
        None,
        "REMOVE cleared the property despite emitting a row"
    );
}

// ====================================================================
// SET OVERWRITE — RETURN reads the NEW value of a pre-existing property
// (the mirror is a per-key overwrite, not just an additive insert).
// ====================================================================

#[test]
fn set_return_overwrites_existing_property() {
    let (c, s) = fresh(); // balance starts at 5000
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.balance = 1 RETURN a.balance",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(1)]],
        "SET a.balance=1 RETURN a.balance must read the OVERWRITTEN value (1), not 5000"
    );
    assert_eq!(
        account_bag(&s).get("balance"),
        Some(&Value::Integer(1)),
        "overwrite persisted"
    );
}

// ====================================================================
// REMOVE then READ the removed property — RETURN must project NULL (the
// removal mirror, not the stale pre-REMOVE value). The Remove-side
// in-view-staleness oracle, parallel to the SET test C.
// ====================================================================

#[test]
fn remove_return_reads_removed_property_as_null() {
    let (c, s) = fresh_flagged(); // flagged = true seeded
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) REMOVE a.flagged RETURN a.flagged",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Null]],
        "REMOVE a.flagged RETURN a.flagged must project NULL (the in-view \
         mirror reflects the removal, not the stale pre-REMOVE `true`)"
    );
}

// ====================================================================
// TERMINAL REMOVE RETURN-less — MUST stay 0 rows (Remove-side guard).
// ====================================================================

#[test]
fn terminal_remove_returnless_stays_zero_rows() {
    let (c, s) = fresh_flagged();
    let rows = run("MATCH (a:Account {id: 'a1'}) REMOVE a.flagged", &c, &s);
    assert_eq!(
        rows.len(),
        0,
        "TERMINAL REMOVE (no RETURN) must emit 0 rows, got {rows:?}"
    );
    assert_eq!(
        account_bag(&s).get("flagged"),
        None,
        "terminal REMOVE still cleared a.flagged"
    );
}

// ====================================================================
// AGGREGATE over SET — `SET … RETURN count(*)` / `sum(…)`. These lower
// to `Project(Aggregate(Set(…)))`: the Aggregate is the SET's direct
// parent, so the Aggregate arm (not the Project) flips the SET to
// stacked. Without it the SET drains and the aggregate folds over 0 rows
// → count(*)=0 / sum=NULL (a silent-wrong).
// ====================================================================

#[test]
fn set_return_aggregate_count_counts_matched_rows() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.balance = 1 RETURN count(*)",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(1)]],
        "SET … RETURN count(*) must count the matched/mutated row (1), not 0"
    );
}

#[test]
fn set_return_aggregate_sum_reads_set_value() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.risk = 9 RETURN sum(a.risk)",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(9)]],
        "SET a.risk=9 RETURN sum(a.risk) must sum the post-SET value (9), not NULL \
         (the SET passes its mirrored row to the Aggregate)"
    );
}

// ====================================================================
// AGGREGATE over SET, MULTI-NODE — `RETURN count(a)` (the NODE-VARIABLE
// count, not count(*)). THE LOAD-BEARING DISCRIMINATOR for the #793 R1
// Aggregate-intermediate finding: `SET … RETURN count(a)` lowers to
// `Project(Aggregate(Set(…)))`; the Aggregate is the SET's direct parent,
// so the Aggregate build arm must flip the SET to stacked. WITHOUT that
// flip the SET drains all 3 rows → the Aggregate folds count(a) over 0
// rows → `[[0]]` (silent-wrong: the writes persist but the count is wrong).
// N = 3 distinguishes "counts the real matched-and-set set (3)" from "0"
// AND from an accidental "1". `count(a)` (not count(*)) additionally proves
// each mutated row still carries a NON-NULL node binding `a` post-SET.
// ====================================================================

#[test]
fn set_return_aggregate_count_node_var_counts_all_matched_rows() {
    let (c, s) = fresh_n(3);
    let rows = run(
        "MATCH (a:Account) SET a.flagged = true RETURN count(a)",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(3)]],
        "SET … RETURN count(a) over 3 Accounts must count the matched-and-set \
         rows (3), NOT 0 (the Aggregate-intermediate silent-wrong: a drained \
         terminal SET makes the Aggregate fold over 0 rows)"
    );
    // Persist check: all 3 writes landed despite the aggregate being returned.
    assert_eq!(account_count(&s), 3, "fixture still has all 3 Accounts");
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, Some(LabelId::new(ACCOUNT)), Lsn::MAX)
        .expect("scan_nodes OK");
    for n in &nodes {
        assert_eq!(
            n.node.properties.get("flagged"),
            Some(&Value::Boolean(true)),
            "each Account.flagged persisted to the substrate"
        );
    }
}

// ====================================================================
// AGGREGATE over SET, MULTI-NODE SUM — `RETURN sum(a.risk)` over 3 nodes
// each SET to risk=1 → 3. Stacks the Aggregate-flip with the in-row mirror:
// `sum` reads the POST-SET `a.risk` per row (the mirror's property
// freshness), and the Aggregate folds over all 3 mutated rows (the flip's
// row flow). Without the flip → folds over 0 rows → NULL.
// ====================================================================

#[test]
fn set_return_aggregate_sum_multinode_reads_set_values() {
    let (c, s) = fresh_n(3);
    let rows = run(
        "MATCH (a:Account) SET a.risk = 1 RETURN sum(a.risk)",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(3)]],
        "SET a.risk=1 RETURN sum(a.risk) over 3 Accounts must sum the post-SET \
         values (1+1+1=3), not NULL (drained) and not a stale pre-SET sum"
    );
}

// ====================================================================
// AGGREGATE under WITH (not RETURN) — `SET … WITH count(a) AS c RETURN c`.
// WITH-with-aggregate ALSO lowers to `Project(Aggregate(Set(…)))` (the WITH
// path's `lower_aggregation_clause`), so the SAME Aggregate-arm flip covers
// it. Proves the fix is not RETURN-specific: the mid-query aggregate
// horizon over a write-op folds over the real rows too.
// ====================================================================

#[test]
fn set_with_aggregate_count_node_var_counts_all_matched_rows() {
    let (c, s) = fresh_n(3);
    let rows = run(
        "MATCH (a:Account) SET a.x = 1 WITH count(a) AS c RETURN c",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![vec![Value::Integer(3)]],
        "SET … WITH count(a) AS c RETURN c over 3 Accounts must carry the \
         real count (3) through the WITH-aggregate horizon, not 0"
    );
}

// ====================================================================
// Sanity: the multi-node fixture seeds exactly 3 distinct Accounts (so the
// count(a)→3 oracle is grounded, not a fixture artifact).
// ====================================================================

#[test]
fn fresh_n_seeds_three_distinct_accounts() {
    let (_c, s) = fresh_n(3);
    assert_eq!(
        account_count(&s),
        3,
        "fresh_n(3) seeds exactly 3 Account nodes"
    );
}

// ====================================================================
// UNWIND over SET — `SET … UNWIND … RETURN …` lowers to `Unwind(Set(…))`:
// UNWIND is the SET's direct parent and must flip it to stacked (else the
// SET drains and UNWIND expands 0 rows → []).
// ====================================================================

#[test]
fn set_unwind_return_expands_over_mutated_rows() {
    let (c, s) = fresh();
    let rows = run(
        "MATCH (a:Account {id: 'a1'}) SET a.risk = 9 UNWIND [1, 2] AS y RETURN a.risk, y",
        &c,
        &s,
    );
    assert_eq!(
        rows,
        vec![
            vec![Value::Integer(9), Value::Integer(1)],
            vec![Value::Integer(9), Value::Integer(2)],
        ],
        "SET a.risk=9 UNWIND [1,2] AS y RETURN a.risk, y must expand to 2 rows \
         each carrying the post-SET a.risk=9"
    );
}

// ====================================================================
// ORDER BY / DISTINCT / LIMIT over SET-RETURN — these wrap a `Project`
// (`Sort/Distinct/Limit(Project(Set(…)))`), so the SET is flipped by the
// Project arm and the modifier sees the rows. Regression guards proving
// the intervening-Project cases need NO extra arm flip.
// ====================================================================

#[test]
fn set_return_with_orderby_distinct_limit_via_project() {
    let (c, s) = fresh();
    // ORDER BY (resolves to the RETURN output alias per #767).
    assert_eq!(
        run(
            "MATCH (a:Account {id: 'a1'}) SET a.risk = 9 RETURN a.risk AS r ORDER BY r",
            &c,
            &s,
        ),
        vec![vec![Value::Integer(9)]],
        "SET … RETURN a.risk AS r ORDER BY r works via the intervening Project"
    );
    // DISTINCT.
    let (c2, s2) = fresh();
    assert_eq!(
        run(
            "MATCH (a:Account {id: 'a1'}) SET a.risk = 9 RETURN DISTINCT a.risk",
            &c2,
            &s2,
        ),
        vec![vec![Value::Integer(9)]],
        "SET … RETURN DISTINCT a.risk works via the intervening Project"
    );
    // LIMIT.
    let (c3, s3) = fresh();
    assert_eq!(
        run(
            "MATCH (a:Account {id: 'a1'}) SET a.risk = 9 RETURN a.risk LIMIT 5",
            &c3,
            &s3,
        ),
        vec![vec![Value::Integer(9)]],
        "SET … RETURN a.risk LIMIT 5 works via the intervening Project"
    );
}

// ====================================================================
// Sanity: the fixture/read path is label-scoped + single-node.
// ====================================================================

#[test]
fn fixture_has_exactly_one_account() {
    let (_c, s) = fresh();
    let nodes = s
        .scan_nodes(TenantId::DEFAULT, Some(LabelId::new(ACCOUNT)), Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(nodes.len(), 1, "fixture has exactly one Account node");
}
