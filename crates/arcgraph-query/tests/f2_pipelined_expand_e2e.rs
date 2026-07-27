//! F2 (PE-1 §F2) — pipelined anchor-seeded Expand for inner-join
//! traversals. Pins the plan-shape fold (RED-on-revert via the F2
//! toggle) AND the LOAD-BEARING multiset (openCypher bag) identity of
//! the pipelined Expand vs the hash-join reference path.
//!
//! ## Why the toggle-based A/B is the multiset oracle
//!
//! F2 rewrites `Join(…, Expand, …)` into a pipelined `Expand` in
//! `Pipeline::build`. `Pipeline::set_pipelined_expand_enabled(false)`
//! reverts THE SAME query to the pre-F2 hash-join path (per-thread, no
//! global race), so every `assert_ab_identical` compares the fast path
//! against the exact operator tree it replaced — the "full-scan + join"
//! result the memo calls the known-correct baseline. Each case ALSO
//! pins a hand-computed expected row count so the A/B is not vacuous,
//! and asserts F2 genuinely removed the hash joins (else the A/B would
//! trivially pass by NOT firing).
//!
//! ## Enumerated shapes covered (see `f2_folds_*` plan-shape tests)
//!
//! The M4-52 join enumerator reorders the naive lowering; F2 keys on the
//! ENUMERATED shape. Covered: bare-Expand on the left (anchored / plain /
//! incoming / count), the to-label semi-join fold (both-ends / only-b-
//! labeled), and the recursive two-hop fold.

#![allow(clippy::too_many_lines)]

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::executor::{Pipeline, StubExecutorSubstrate};
use arcgraph_query::logical_plan::{
    LogicalPlanLoweringVisitor, rewrite_unfiltered_count_to_count_store,
};
use arcgraph_query::planner::{enumerate_join_order, pick_join_algorithms};
use arcgraph_query::semantic::{BindingVisitor, StubCatalogProvider, TypeCheckVisitor};
use arcgraph_query::{MaterializedResult, QueryEngine};

const PERSON: u32 = 1;
const COMPANY: u32 = 2;
const KNOWS: u32 = 1;

fn cat() -> StubCatalogProvider {
    // Cardinalities feed the M4-52 enumerator's cost model so the
    // benchmark-representative swapped shape is produced.
    StubCatalogProvider::new()
        .with_labels(["Person", "Company"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
        .with_total_node_count(1000)
        .with_total_rel_count(2000)
        .with_label_cardinality(LabelId::new(PERSON), 900)
        .with_label_cardinality(LabelId::new(COMPANY), 100)
        .with_rel_type_cardinality(TypeId::new(KNOWS), 2000)
}

fn person(id: u64, name: &str) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(PERSON)))
        .with_label_name("Person")
        .with_property("name", Value::String(name.to_string()))
}

fn company(id: u64, name: &str) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(COMPANY)))
        .with_label_name("Company")
        .with_property("name", Value::String(name.to_string()))
}

fn knows(rel_id: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(rel_id),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(KNOWS)),
    )
    .with_rel_type_name("KNOWS")
}

/// Alice(1) KNOWS Bob(2), Carol(3) [both Person] AND Acme(4) [Company].
/// Bob/Carol/Acme have no outbound edges. Exercises multi-out fan-out +
/// the to-label filter (Acme must be dropped by `(b:Person)`).
fn substrate_fanout() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, person(1, "Alice"))
        .with_node(TenantId::DEFAULT, person(2, "Bob"))
        .with_node(TenantId::DEFAULT, person(3, "Carol"))
        .with_node(TenantId::DEFAULT, company(4, "Acme"))
        .with_edge(TenantId::DEFAULT, knows(10, 1, 2))
        .with_edge(TenantId::DEFAULT, knows(11, 1, 3))
        .with_edge(TenantId::DEFAULT, knows(12, 1, 4))
}

/// z1(10) KNOWS a(12); z2(11) KNOWS a(12); a(12) KNOWS b1(13), b2(14).
/// `a` is fed to the second expand TWICE (once per z) — the duplicate-
/// from multiset case.
fn substrate_two_hop() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, person(10, "z1"))
        .with_node(TenantId::DEFAULT, person(11, "z2"))
        .with_node(TenantId::DEFAULT, person(12, "a"))
        .with_node(TenantId::DEFAULT, person(13, "b1"))
        .with_node(TenantId::DEFAULT, person(14, "b2"))
        .with_edge(TenantId::DEFAULT, knows(20, 10, 12))
        .with_edge(TenantId::DEFAULT, knows(21, 11, 12))
        .with_edge(TenantId::DEFAULT, knows(22, 12, 13))
        .with_edge(TenantId::DEFAULT, knows(23, 12, 14))
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn run(
    q: &str,
    cat: &StubCatalogProvider,
    s: &StubExecutorSubstrate,
    f2: bool,
) -> MaterializedResult {
    let prev = Pipeline::set_pipelined_expand_enabled(f2);
    let out = QueryEngine::new(cat).execute(q, s).expect("execute");
    Pipeline::set_pipelined_expand_enabled(prev);
    out
}

/// Ordering-independent multiset key: sort per-row debug strings. Both
/// paths project through the terminal RETURN, so column order is
/// normalized and the raw values are byte-identical (Scan + Expand
/// materialize the same `NodeView` for a given node).
fn bag(r: &MaterializedResult) -> Vec<String> {
    let mut keys: Vec<String> = r.rows().iter().map(|row| format!("{row:?}")).collect();
    keys.sort();
    keys
}

/// (hash_join_count, expand_count) in the physical plan built through the
/// REAL pipeline (lower → count-store rewrite → enumerate → pick → build)
/// with F2 forced to `f2`.
fn physical_shape(q: &str, cat: &StubCatalogProvider, f2: bool) -> (usize, usize) {
    let stmt = arcgraph_query::parser::parse(q).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, q, cat).expect("bind");
    TypeCheckVisitor::check(&mut bound, cat).expect("typecheck");
    let lowered = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let lowered = rewrite_unfiltered_count_to_count_store(lowered);
    let optimized = pick_join_algorithms(enumerate_join_order(lowered, cat), cat);
    let prev = Pipeline::set_pipelined_expand_enabled(f2);
    let phys = Pipeline::build(&optimized).expect("build");
    Pipeline::set_pipelined_expand_enabled(prev);
    let dbg = format!("{phys:?}");
    (
        dbg.matches("HashJoin(").count(),
        dbg.matches("Expand(ExpandOp").count(),
    )
}

/// The load-bearing assertion: F2 (on) produces the SAME multiset as the
/// hash-join reference (F2 off), F2 actually removed the hash joins, and
/// the reference genuinely had them (so the A/B is non-vacuous). Returns
/// the row count for the caller's hand-computed check.
fn assert_ab_identical(q: &str, cat: &StubCatalogProvider, s: &StubExecutorSubstrate) -> usize {
    let f2 = run(q, cat, s, true);
    let reference = run(q, cat, s, false);
    assert_eq!(
        bag(&f2),
        bag(&reference),
        "F2 multiset MUST equal the hash-join reference for: {q}\n f2={:#?}\n ref={:#?}",
        f2.rows(),
        reference.rows()
    );
    let (hj_on, exp_on) = physical_shape(q, cat, true);
    let (hj_off, _) = physical_shape(q, cat, false);
    assert_eq!(hj_on, 0, "F2 must remove all hash joins for: {q}");
    assert!(exp_on >= 1, "F2 must build at least one Expand for: {q}");
    assert!(
        hj_off > 0,
        "reference (F2 off) must retain hash join(s) for: {q} — else the A/B is vacuous"
    );
    f2.len()
}

// ---------------------------------------------------------------------
// Plan-shape (RED-on-revert): F2 folds the enumerated traversal shapes.
// ---------------------------------------------------------------------

#[test]
fn f2_folds_plain_traversal_to_pipelined_expand() {
    let cat = cat();
    // b UNLABELED — the single-join from-seeded shape.
    let (hj_on, exp_on) =
        physical_shape("MATCH (a:Person)-[r:KNOWS]->(b) RETURN a, r, b", &cat, true);
    assert_eq!(
        (hj_on, exp_on),
        (0, 1),
        "F2 on: no hash join, one pipelined Expand"
    );
    let (hj_off, _) = physical_shape(
        "MATCH (a:Person)-[r:KNOWS]->(b) RETURN a, r, b",
        &cat,
        false,
    );
    assert!(hj_off > 0, "RED-on-revert: F2 off restores the hash join");
}

#[test]
fn f2_folds_both_ends_labeled_to_label_filtered_expand() {
    let cat = cat();
    // Both ends labeled — the nested to-label semi-join shape. F2 folds
    // BOTH joins into ONE to-label-filtered Expand.
    let (hj_on, exp_on) = physical_shape(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b",
        &cat,
        true,
    );
    assert_eq!(
        (hj_on, exp_on),
        (0, 1),
        "F2 on: both joins gone, one Expand"
    );
    let (hj_off, _) = physical_shape(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b",
        &cat,
        false,
    );
    assert_eq!(hj_off, 2, "RED-on-revert: F2 off restores BOTH hash joins");
}

#[test]
fn f2_folds_count_benchmark_queries() {
    let cat = cat();
    // The exact `expand_count` / `anchored_expand`-class count queries.
    for q in [
        "MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b)",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(b)",
    ] {
        let (hj_on, exp_on) = physical_shape(q, &cat, true);
        assert_eq!(hj_on, 0, "F2 on: no hash join for {q}");
        assert!(exp_on >= 1, "F2 on: pipelined Expand for {q}");
        let (hj_off, _) = physical_shape(q, &cat, false);
        assert!(hj_off > 0, "RED-on-revert for {q}");
    }
}

#[test]
fn f2_folds_two_hop_recursively() {
    let cat = cat();
    let (hj_on, exp_on) = physical_shape(
        "MATCH (z:Person)-[:KNOWS]->(a)-[:KNOWS]->(b) RETURN z, a, b",
        &cat,
        true,
    );
    assert_eq!(
        (hj_on, exp_on),
        (0, 2),
        "F2 fully pipelines the 2-hop: Expand(Expand(Scan)), no hash join"
    );
}

// ---------------------------------------------------------------------
// Bushy fallthrough: F2 does NOT touch shapes it can't prove identical.
// ---------------------------------------------------------------------

#[test]
fn cartesian_and_bushy_joins_keep_hash_join() {
    let cat = cat();
    // Cartesian (no shared binding) — F2 must not fire.
    let (hj_cart, _) = physical_shape("MATCH (a:Person), (b:Person) RETURN a, b", &cat, true);
    assert!(hj_cart > 0, "cartesian join stays a hash join under F2");

    // Two disconnected patterns sharing nothing on the join spine — a
    // genuinely bushy multi-way shape retains join machinery.
    let (hj_bushy, _) = physical_shape(
        "MATCH (a:Person)-[:KNOWS]->(b), (c:Person)-[:KNOWS]->(d) RETURN a, b, c, d",
        &cat,
        true,
    );
    assert!(
        hj_bushy > 0,
        "the cross-pattern (bushy) join is retained; only the per-pattern expands fold"
    );
}

// ---------------------------------------------------------------------
// Multiset identity (LOAD-BEARING): F2 == hash-join reference.
// ---------------------------------------------------------------------

#[test]
fn multiset_plain_traversal() {
    // (a) plain `(a)-[:T]->(b)` — Alice's 2 KNOWS edges (Bob, Carol) +
    // the Company edge (b unlabeled → kept) = 3 rows.
    let (cat, s) = (cat(), substrate_fanout());
    let n = assert_ab_identical("MATCH (a:Person)-[r:KNOWS]->(b) RETURN a, r, b", &cat, &s);
    assert_eq!(n, 3, "Alice → Bob, Carol, Acme (b unlabeled)");
}

#[test]
fn multiset_to_side_label_drops_nonmatching() {
    // (b) to-side label `(b:Person)` — the Company node (Acme) MUST be
    // dropped by the folded per-edge label filter → 2 rows.
    let (cat, s) = (cat(), substrate_fanout());
    let n = assert_ab_identical(
        "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, b",
        &cat,
        &s,
    );
    assert_eq!(
        n, 2,
        "Acme (Company) dropped by (b:Person); Bob + Carol remain"
    );

    // And confirm the surviving b's are exactly Bob + Carol (not Acme).
    let rows = run(
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.name",
        &cat,
        &s,
        true,
    );
    let mut names: Vec<String> = rows
        .rows()
        .iter()
        .map(|r| match &r[0] {
            Value::String(x) => x.clone(),
            other => panic!("expected string name, got {other:?}"),
        })
        .collect();
    names.sort();
    assert_eq!(names, vec!["Bob".to_string(), "Carol".to_string()]);
}

#[test]
fn multiset_duplicate_from_rows() {
    // (c) duplicate `from` feeding the expand: `a` is reached from z1 AND
    // z2, so the second hop expands `a` TWICE → 2 z × 2 b = 4 rows. F2's
    // pipelined expand must NOT dedup.
    let (cat, s) = (cat(), substrate_two_hop());
    let n = assert_ab_identical(
        "MATCH (z:Person)-[:KNOWS]->(a)-[:KNOWS]->(b) RETURN z, a, b",
        &cat,
        &s,
    );
    assert_eq!(n, 4, "2 z-sources × 2 b-targets — duplicate-from preserved");
}

#[test]
fn multiset_direction_incoming() {
    // (d) direction variant `<-`: incoming KNOWS. Nodes 2,3 each have one
    // incoming edge (from Alice=1); Alice has none → 2 rows.
    let (cat, s) = (cat(), substrate_fanout());
    let n = assert_ab_identical("MATCH (a:Person)<-[r:KNOWS]-(b) RETURN a, b", &cat, &s);
    assert_eq!(
        n, 2,
        "Bob and Carol each have one incoming KNOWS (from Alice)"
    );
}

#[test]
fn multiset_multiple_out_edges_per_source() {
    // (e) a node with multiple out-edges (M-per-source): Alice → 3.
    let (cat, s) = (cat(), substrate_fanout());
    let n = assert_ab_identical("MATCH (a:Person)-[:KNOWS]->(b) RETURN b", &cat, &s);
    assert_eq!(n, 3, "Alice's 3 out-edges, one row each");
}

#[test]
fn multiset_anchored_expand() {
    // The headline `anchored_expand` shape: a 1-row anchor seeds the
    // expand. `{name:'Alice'}` wraps Scan(a) in a Filter that becomes the
    // pipelined Expand's child.
    let (cat, s) = (cat(), substrate_fanout());
    let n = assert_ab_identical(
        "MATCH (a:Person {name:'Alice'})-[:KNOWS]->(b) RETURN b.name",
        &cat,
        &s,
    );
    assert_eq!(n, 3, "only Alice matches the anchor; her 3 edges expand");

    // A non-matching anchor yields zero rows on both paths.
    let empty = run(
        "MATCH (a:Person {name:'Nobody'})-[:KNOWS]->(b) RETURN b",
        &cat,
        &s,
        true,
    );
    assert_eq!(empty.len(), 0);
}

#[test]
fn multiset_count_benchmark_queries() {
    // The `expand_count` benchmark queries — F2 vs hash-join reference
    // must agree on the aggregate.
    let (cat, s) = (cat(), substrate_fanout());
    for (q, expected) in [
        ("MATCH (a:Person)-[:KNOWS]->(b) RETURN count(b)", 3),
        ("MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN count(b)", 2),
    ] {
        let f2 = run(q, &cat, &s, true);
        let reference = run(q, &cat, &s, false);
        assert_eq!(bag(&f2), bag(&reference), "count A/B mismatch for {q}");
        match &f2.rows()[0][0] {
            Value::Integer(c) => assert_eq!(*c, expected, "count for {q}"),
            other => panic!("expected integer count, got {other:?}"),
        }
    }
}

#[test]
fn multiset_empty_substrate_zero_rows() {
    // Degenerate: no data → both paths emit zero rows (no divergence,
    // no panic).
    let cat = cat();
    let s = StubExecutorSubstrate::new();
    let n = assert_ab_no_shape_check("MATCH (a:Person)-[:KNOWS]->(b) RETURN a, b", &cat, &s);
    assert_eq!(n, 0);
}

/// Like [`assert_ab_identical`] but skips the "reference retains a hash
/// join" non-vacuity check (used for the empty-substrate degenerate case
/// where we only care that both paths agree at zero rows).
fn assert_ab_no_shape_check(
    q: &str,
    cat: &StubCatalogProvider,
    s: &StubExecutorSubstrate,
) -> usize {
    let f2 = run(q, cat, s, true);
    let reference = run(q, cat, s, false);
    assert_eq!(
        bag(&f2),
        bag(&reference),
        "F2 vs reference mismatch for {q}"
    );
    f2.len()
}
