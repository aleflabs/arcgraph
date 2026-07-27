//! **#773 (Customer-Zero AML; umbrella #649) — openCypher post-WITH
//! `WHERE`: the pipeline filter (G1) + `HAVING`-over-aggregate-alias
//! (G2), END-TO-END.**
//!
//! # The two failures this pins
//!
//! Both were the WITH `WHERE` binding in the PRE-WITH scope instead of
//! the WITH PROJECTION-OUTPUT scope (`bind_with_clause`, `semantic/
//! binding.rs`). openCypher evaluates a `WITH … WHERE` AFTER the
//! projection (clauses/with-where) — the predicate references the
//! projected columns, so it must resolve to the projection OUTPUT ids,
//! the same rule the #767 ORDER-BY-to-projection-output fix established
//! for RETURN. Lowering already places the `Filter` ABOVE the WITH
//! `Project` (and above the `Aggregate` beneath it), so once the
//! predicate's binding-ids match the Project/Aggregate OUTPUT schema the
//! filter runs correctly.
//!
//! - **G1 — non-aggregate pipeline filter** (was `-32006` Eval):
//!   `MATCH (a:Account) WITH a WHERE a.balance > 50000 RETURN …`. The
//!   passthrough `a` was bound to the pre-WITH Scan id; the WITH
//!   `Project` re-emits `a` under a fresh `output_id` (#746), so the
//!   `Filter` above it keyed on the Scan id was "missing from row schema"
//!   at runtime. Now `a` resolves to the WITH-projection output id.
//! - **G2 — `HAVING` over an aggregate alias** (was `-32005` bind):
//!   `MATCH (d)<-[t:SENT]-() WITH d, sum(t.amount) AS s WHERE s > 20000
//!   RETURN …`. The aggregate alias `s` is declared ONLY in the new
//!   scope, so a pre-WITH bind was `UndeclaredVariable`. Now `s` resolves
//!   to the aggregate-projection OUTPUT id and the filter runs AFTER the
//!   aggregate: `Filter(s>…, Aggregate(sum(t.amount) AS s, group by d))`.
//!   This IS the Customer-Zero AML "mule-by-volume" query.
//!
//! A companion type-check fix (`ArgKind::Numeric` admits the
//! `Property{..}` dynamic-schema sentinel, `semantic/functions.rs`)
//! unblocks `sum(t.amount)` / `avg(prop)` end-to-end — the v1.0 catalog
//! under-types every property access as `Property::String`, so the prior
//! `Property{Integer|Float}`-only rule false-positived EVERY
//! `sum(prop)`. The G2 `sum`-of-amount oracle exercises BOTH fixes.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every positive assertion drives a REAL ArcQL query through the FULL
//! pipeline (`QueryEngine::execute`: parse → bind → type-check →
//! cross-substrate → lower → execute) against a data-bearing substrate
//! and asserts the EXACT result rows. The negative scoping-fence
//! assertion drives the binder directly for a STRONG `UndeclaredVariable`
//! oracle. The oracle is the openCypher `clauses/with-where` (post-WITH
//! WHERE / HAVING) semantics + the Customer-Zero `cz_finance.py` G1/G2
//! repro fixtures.
//!
//! # Faithful-oracle note on the literal G1 spawn-brief query
//!
//! The spawn brief's G1 query reads `RETURN a.id ORDER BY a.id`. The
//! trailing `ORDER BY a.id` (ordering by a NON-identifier property
//! expression) is a PRE-EXISTING, WITH-INDEPENDENT #767-DEFERRED
//! limitation — `MATCH (a:Account) RETURN a.id ORDER BY a.id` fails
//! identically with NO `WITH` at all (#767 "NOT covered: ORDER BY by a
//! rendered non-identifier expression name"). It is OUTSIDE the
//! WITH-WHERE footprint. So the G1 ordered oracle here uses the working
//! `RETURN a.id AS id ORDER BY id` form (ORDER BY resolves to the RETURN
//! output alias per #767) PLUS an unordered `RETURN a.id` form sorted in
//! the harness — both faithfully prove the WITH-WHERE FILTER selects the
//! over-threshold accounts; neither depends on the orthogonal
//! ORDER-BY-by-property gap.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::executor::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::parse;
use arcgraph_query::semantic::error::BindingError;
use arcgraph_query::semantic::{BindingVisitor, StubCatalogProvider};

const ACCOUNT: u32 = 1; // first label ⇒ LabelId::new(1)
const DST: u32 = 2; // second label ⇒ LabelId::new(2)
const A: u32 = 3; // third label ⇒ LabelId::new(3)
const B: u32 = 4; // fourth label ⇒ LabelId::new(4)
const SENT: u32 = 1; // first rel-type ⇒ TypeId::new(1)
const KNOWS: u32 = 2; // second rel-type ⇒ TypeId::new(2)

/// Catalog: the CZ AML property/label/rel-type names. Property value
/// types are NOT tracked at v1.0 (every access types as the
/// `Property::String` sentinel — see the module doc); the substrate
/// stores the actual `Integer` values.
fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Account", "Dst", "A", "B"])
        .with_rel_types(["SENT", "KNOWS"])
        .with_properties(["id", "balance", "amount", "name"])
}

fn account(id: u64, balance: i64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(ACCOUNT)))
        .with_property("id", Value::Integer(id as i64))
        .with_property("balance", Value::Integer(balance))
}

fn dst(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(DST)))
        .with_property("id", Value::Integer(id as i64))
}

fn sender(id: u64) -> NodeView {
    // The anonymous `()` source of a SENT edge. Labelled `Account` so it
    // is a valid node; the G2 pattern leaves it unbound.
    NodeView::new(NodeId::new(id), Some(LabelId::new(ACCOUNT)))
}

fn named(label: u32, id: u64, name: &str) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
        .with_property("name", Value::String(name.into()))
}

fn sent(rel: u64, from: u64, to: u64, amount: i64) -> RelView {
    RelView::new(
        RelId::new(rel),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(SENT)),
    )
    .with_property("amount", Value::Integer(amount))
}

fn knows(rel: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(rel),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(KNOWS)),
    )
}

/// G1 fixture: three accounts with balances [30000, 60000, 80000]
/// (`cz_finance.py` G1 repro). Only the 60k + 80k accounts clear the
/// `> 50000` threshold.
fn g1_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, account(1, 30000))
        .with_node(TenantId::DEFAULT, account(2, 60000))
        .with_node(TenantId::DEFAULT, account(3, 80000))
}

/// G2 fixture (`cz_finance.py` G2 repro): `(src)-[:SENT {amount}]->(dst)`.
/// d1 receives one 15000 transfer (total 15000, count 1); d2 receives
/// 10000 + 15000 (total 25000, count 2). Only d2's sum clears `> 20000`;
/// only d2's count clears `> 1`.
fn g2_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, dst(1))
        .with_node(TenantId::DEFAULT, dst(2))
        .with_node(TenantId::DEFAULT, sender(10))
        .with_node(TenantId::DEFAULT, sender(11))
        .with_node(TenantId::DEFAULT, sender(12))
        .with_edge(TenantId::DEFAULT, sent(100, 10, 1, 15000))
        .with_edge(TenantId::DEFAULT, sent(101, 11, 2, 10000))
        .with_edge(TenantId::DEFAULT, sent(102, 12, 2, 15000))
}

/// Dropped relationship-variable fixture: one `A` and two `B` nodes
/// with no relationships, so OPTIONAL MATCH emits `r = NULL` and the
/// WITH-WHERE predicate must run before `WITH c` drops `r`.
fn dropped_rel_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, named(A, 20, "a1"))
        .with_node(TenantId::DEFAULT, named(B, 21, "c1"))
        .with_node(TenantId::DEFAULT, named(B, 22, "c2"))
}

/// Dropped node-variable fixture: `a1` has a KNOWS neighbor, `a2` does
/// not. `WITH a WHERE other IS NULL` should keep only `a2`.
fn dropped_node_substrate() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, named(A, 30, "a1"))
        .with_node(TenantId::DEFAULT, named(A, 31, "a2"))
        .with_node(TenantId::DEFAULT, named(B, 32, "other"))
        .with_edge(TenantId::DEFAULT, knows(300, 30, 32))
}

/// Full pipeline → result rows (panics on any stage error).
fn run<S: ExecutorSubstrate>(query: &str, c: &StubCatalogProvider, s: &S) -> Vec<Vec<Value>> {
    let engine = QueryEngine::new(c);
    engine.execute(query, s).expect("execute").rows
}

fn int(v: &Value) -> i64 {
    match v {
        Value::Integer(n) => *n,
        other => panic!("expected Integer, got {other:?}"),
    }
}

/// Single-column rows → the column-0 integers in ROW ORDER (as returned;
/// used where an ORDER BY makes the order deterministic).
fn col0(rows: &[Vec<Value>]) -> Vec<i64> {
    rows.iter()
        .map(|r| {
            assert_eq!(r.len(), 1, "expected single-column rows, got {r:?}");
            int(&r[0])
        })
        .collect()
}

/// Single-column rows → SORTED column-0 integers (order-independent
/// oracle for the no-ORDER-BY forms).
fn col0_sorted(rows: &[Vec<Value>]) -> Vec<i64> {
    let mut v = col0(rows);
    v.sort_unstable();
    v
}

/// Two-column `(id, agg)` rows → SORTED pairs (order-independent).
fn pairs_sorted(rows: &[Vec<Value>]) -> Vec<(i64, i64)> {
    let mut v: Vec<(i64, i64)> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 2, "expected two-column rows, got {r:?}");
            (int(&r[0]), int(&r[1]))
        })
        .collect();
    v.sort_unstable();
    v
}

fn strings_sorted(rows: &[Vec<Value>]) -> Vec<String> {
    let mut v: Vec<String> = rows
        .iter()
        .map(|r| {
            assert_eq!(r.len(), 1, "expected single-column rows, got {r:?}");
            match &r[0] {
                Value::String(s) => s.clone(),
                other => panic!("expected String, got {other:?}"),
            }
        })
        .collect();
    v.sort();
    v
}

// =====================================================================
// G1 — non-aggregate post-WITH WHERE (pipeline filter). Was -32006.
// =====================================================================

#[test]
fn g1_with_where_filters_passthrough_unordered() {
    // `WITH a WHERE a.balance > 50000` selects the 60k + 80k accounts
    // (ids 2, 3) and EXCLUDES the 30k account (id 1). Unordered form,
    // sorted in-harness — the WITH-WHERE FILTER is the unit under test.
    let rows = run(
        "MATCH (a:Account) WITH a WHERE a.balance > 50000 RETURN a.id",
        &cat(),
        &g1_substrate(),
    );
    assert_eq!(col0_sorted(&rows), vec![2, 3]);
}

#[test]
fn g1_with_where_filters_passthrough_ordered() {
    // Faithful ordered G1 oracle (see module doc): the alias form whose
    // ORDER BY resolves to the RETURN output (#767). Deterministic.
    let rows = run(
        "MATCH (a:Account) WITH a WHERE a.balance > 50000 RETURN a.id AS id ORDER BY id",
        &cat(),
        &g1_substrate(),
    );
    assert_eq!(col0(&rows), vec![2, 3], "ascending, threshold-filtered");
}

#[test]
fn g1_with_where_returns_the_full_node() {
    // `RETURN a` (the node) — proves the passthrough `a` carries its
    // properties through the WITH projection + filter. Exactly the two
    // over-threshold nodes survive.
    let rows = run(
        "MATCH (a:Account) WITH a WHERE a.balance > 50000 RETURN a",
        &cat(),
        &g1_substrate(),
    );
    let mut ids: Vec<u64> = rows
        .iter()
        .map(|r| match &r[0] {
            Value::Node(n) => n.id.raw(),
            other => panic!("expected Node, got {other:?}"),
        })
        .collect();
    ids.sort_unstable();
    assert_eq!(ids, vec![2, 3]);
}

// =====================================================================
// G2 — HAVING over an aggregate alias. Was -32005. THE discriminating
// oracle (Customer-Zero AML "mule-by-volume").
// =====================================================================

#[test]
fn g2_having_over_sum_aggregate_alias() {
    // `WITH d, sum(t.amount) AS s WHERE s > 20000` — the WHERE references
    // the aggregate alias `s`, which MUST resolve to the aggregate
    // projection OUTPUT and filter AFTER the aggregate. d1's total is
    // 15000 (dropped); only d2's 25000 survives. The EXACT CZ G2 repro.
    let rows = run(
        "MATCH (d:Dst)<-[t:SENT]-() WITH d, sum(t.amount) AS s WHERE s > 20000 RETURN d.id, s",
        &cat(),
        &g2_substrate(),
    );
    assert_eq!(
        pairs_sorted(&rows),
        vec![(2, 25000)],
        "HAVING sum>20000: only d2 (s=25000); d1 (s=15000) filtered post-aggregate"
    );
}

#[test]
fn g2_having_over_count_aggregate_alias() {
    // Count-based HAVING — `count` is `ArgKind::Any`, so this ISOLATES
    // the binding-scope fix from the `ArgKind::Numeric` type-check fix.
    // d1 has 1 SENT edge, d2 has 2; `WHERE c > 1` keeps only d2.
    let rows = run(
        "MATCH (d:Dst)<-[t:SENT]-() WITH d, count(t) AS c WHERE c > 1 RETURN d.id, c",
        &cat(),
        &g2_substrate(),
    );
    assert_eq!(
        pairs_sorted(&rows),
        vec![(2, 2)],
        "HAVING count>1: only d2 (2 edges)"
    );
}

// =====================================================================
// Dropped pre-WITH inputs — openCypher WITH-WHERE scope. These are the
// TriadicSelection1 / WithWhere1 [3]/[4] cases that #773's output-only
// fence made too strict.
// =====================================================================

#[test]
fn with_where_sees_dropped_typed_relationship_var() {
    let rows = run(
        "MATCH (a:A),(c:B) OPTIONAL MATCH (a)-[r:KNOWS]->(c) WITH c WHERE r IS NULL RETURN c.name",
        &cat(),
        &dropped_rel_substrate(),
    );
    assert_eq!(
        strings_sorted(&rows),
        vec!["c1".to_string(), "c2".to_string()]
    );
}

#[test]
fn with_where_sees_dropped_untyped_relationship_var() {
    let rows = run(
        "MATCH (a:A),(c:B) OPTIONAL MATCH (a)-[r]->(c) WITH c WHERE r IS NULL RETURN c.name",
        &cat(),
        &dropped_rel_substrate(),
    );
    assert_eq!(
        strings_sorted(&rows),
        vec!["c1".to_string(), "c2".to_string()]
    );
}

#[test]
fn with_where_sees_dropped_node_var() {
    let rows = run(
        "MATCH (a:A) OPTIONAL MATCH (a)-[:KNOWS]->(other:B) WITH a WHERE other IS NULL RETURN a.name",
        &cat(),
        &dropped_node_substrate(),
    );
    assert_eq!(strings_sorted(&rows), vec!["a2".to_string()]);
}

// =====================================================================
// Scoping fence — genuinely undeclared variables still fail cleanly.
// =====================================================================

#[test]
fn with_where_rejects_genuinely_undeclared_var() {
    let query = "MATCH (c:B) WITH c WHERE zzz IS NULL RETURN c";
    let stmt = parse(query).expect("parse");
    match BindingVisitor::bind(&stmt, query, &cat()) {
        Ok(_) => panic!("expected an UndeclaredVariable bind error for `zzz`"),
        Err(errs) => assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "zzz"
            )),
            "expected UndeclaredVariable{{zzz}}; got {errs:?}"
        ),
    }
}

// =====================================================================
// Regressions — the load-bearing surfaces the WITH-stage binder change
// must NOT disturb (#749/#767 binding-id lessons).
// =====================================================================

#[test]
fn regression_with_aggregate_projection_without_where_still_works() {
    // The WITH-aggregation projection WITHOUT a WHERE — unchanged by this
    // slice. Both d1 (15000) and d2 (25000) project; no filtering.
    let rows = run(
        "MATCH (d:Dst)<-[t:SENT]-() WITH d, sum(t.amount) AS s RETURN d.id, s",
        &cat(),
        &g2_substrate(),
    );
    assert_eq!(pairs_sorted(&rows), vec![(1, 15000), (2, 25000)]);
}

#[test]
fn regression_pre_with_plain_match_where_still_works() {
    // A plain `MATCH … WHERE … RETURN` (pre-WITH WHERE, the
    // already-working path) is unaffected: ids 2 + 3 clear the threshold.
    let rows = run(
        "MATCH (a:Account) WHERE a.balance > 50000 RETURN a.id",
        &cat(),
        &g1_substrate(),
    );
    assert_eq!(col0_sorted(&rows), vec![2, 3]);
}
