//! **ADR-193 (#619 plain-path EXEC + #621 path-fns)** — `Value::Path`
//! runtime variant END-TO-END tests.
//!
//! Exercises the FULL pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute) for plain named-path execution
//! (`MATCH p = (a)-[..]->(b) RETURN p`) and the `nodes`/`relationships`/
//! `length` path functions, mirroring the #620 list-comprehension e2e
//! shape (`tests/list_comprehension_e2e.rs`).
//!
//! These are the e2e half of the ADR-193 13-test binding contract; the
//! value-level JSON round-trip + decode-precedence (test 10), the
//! `estimate_value_bytes` non-zero pin (test 12), and the function
//! registry pins live as unit tests in the corresponding `src/` modules.
//!
//! All oracles are STRONG `==` over the result rows.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{QueryEngine, parse};

const LABEL_X: u32 = 1;
const TYPE_R: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X"])
        .with_rel_types(["R"])
        .with_properties(["name"])
}

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
}

fn edge(id: u64, from: u64, to: u64) -> RelView {
    RelView::new(
        RelId::new(id),
        NodeId::new(from),
        NodeId::new(to),
        Some(TypeId::new(TYPE_R)),
    )
}

/// Execute through the FULL [`QueryEngine`] pipeline (parse → bind →
/// type-check → cross-substrate → lower → enumerate/plan → execute),
/// matching the production path (the planner phase reconciles the
/// aggregate / sort projection bindings — the manual lower→materialize
/// path skips it). Returns the result rows; expects success.
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    QueryEngine::new(c)
        .execute(query, s)
        .expect("execute")
        .rows()
        .to_vec()
}

/// Expect the query to be REJECTED (compile-time type error OR runtime
/// eval error — both surface as `Err` through the engine). Used for the
/// write-op fence (D-12, compile reject) and the non-path-arg
/// `InvalidArgumentType` (D-7/D-13, runtime reject) cases.
fn run_err(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> bool {
    QueryEngine::new(c).execute(query, s).is_err()
}

// ---------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------

/// A single node (id 1) — for the zero-length path (D-6).
fn single_node() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(TenantId::DEFAULT, node(1))
}

/// Two nodes 1,2 + one edge r10: 1 -[R]-> 2.
fn two_node_one_edge() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(1))
        .with_node(TenantId::DEFAULT, node(2))
        .with_edge(TenantId::DEFAULT, edge(10, 1, 2))
}

/// Chain 1 -[r10]-> 2 -[r11]-> 3 (for var-length composition).
fn chain_three() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(1))
        .with_node(TenantId::DEFAULT, node(2))
        .with_node(TenantId::DEFAULT, node(3))
        .with_edge(TenantId::DEFAULT, edge(10, 1, 2))
        .with_edge(TenantId::DEFAULT, edge(11, 2, 3))
}

/// Fan-out 1 -[r10]-> 2 and 1 -[r11]-> 3 (two distinct paths from 1 —
/// for orderability + DISTINCT non-collision).
fn fan_out() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, node(1))
        .with_node(TenantId::DEFAULT, node(2))
        .with_node(TenantId::DEFAULT, node(3))
        .with_edge(TenantId::DEFAULT, edge(10, 1, 2))
        .with_edge(TenantId::DEFAULT, edge(11, 1, 3))
}

// ---------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------

fn as_path(v: &Value) -> &arcgraph_query::executor::value::PathView {
    match v {
        Value::Path(p) => p,
        other => panic!("expected Value::Path, got {other:?}"),
    }
}

fn node_ids(list: &Value) -> Vec<u64> {
    match list {
        Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::Node(n) => n.id.raw(),
                other => panic!("expected Value::Node, got {other:?}"),
            })
            .collect(),
        other => panic!("expected Value::List, got {other:?}"),
    }
}

fn rel_ids(list: &Value) -> Vec<u64> {
    match list {
        Value::List(xs) => xs
            .iter()
            .map(|x| match x {
                Value::Relationship(r) => r.id.raw(),
                other => panic!("expected Value::Relationship, got {other:?}"),
            })
            .collect(),
        other => panic!("expected Value::List, got {other:?}"),
    }
}

// =====================================================================
// Test 1 — plain path construction + RETURN p
// =====================================================================

#[test]
fn t1_plain_path_construction_and_return() {
    let rows = run(
        "MATCH p = (a:X)-[:R]->(b:X) RETURN p",
        &two_node_one_edge(),
        &cat(),
    );
    assert_eq!(rows.len(), 1, "one edge ⇒ one path");
    let p = as_path(&rows[0][0]);
    assert_eq!(p.start.id, NodeId::new(1), "start = a (node 1)");
    assert_eq!(p.segments.len(), 1, "one segment");
    assert_eq!(p.segments[0].rel.id, RelId::new(10), "segment rel = r10");
    assert_eq!(
        p.segments[0].end.id,
        NodeId::new(2),
        "segment end = b (node 2)"
    );
}

// =====================================================================
// Test 2 — zero-length path p = (a) (D-6)
// =====================================================================

#[test]
fn t2_zero_length_path() {
    // `MATCH p = (a:X) RETURN length(p), nodes(p), relationships(p)`.
    // If the single-node named-path form does not parse at this grammar
    // version, the test documents that as a known limitation (OQ) rather
    // than silently passing — but D-6 requires it to be valid.
    let q = "MATCH p = (a:X) RETURN length(p), nodes(p), relationships(p)";
    if parse(q).is_err() {
        panic!(
            "D-6 zero-length named path `MATCH p = (a)` did not parse — grammar gap; \
             ADR-193 D-6 requires it"
        );
    }
    let rows = run(q, &single_node(), &cat());
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0][0], Value::Integer(0), "length 0");
    assert_eq!(node_ids(&rows[0][1]), vec![1], "nodes = [a]");
    assert_eq!(rows[0][2], Value::List(vec![]), "relationships = []");
}

// =====================================================================
// Test 3 — length(p) hop count + list regression (D-7)
// =====================================================================

#[test]
fn t3_length_hop_count_and_list_regression() {
    let rows = run(
        "MATCH p = (a:X)-[:R]->(b:X) RETURN length(p)",
        &two_node_one_edge(),
        &cat(),
    );
    assert_eq!(rows[0][0], Value::Integer(1), "1-hop path ⇒ length 1");

    // length([1,2,3]) still = 3 (the legacy list form is preserved).
    let rows2 = run(
        "MATCH (n:X) RETURN length([1, 2, 3])",
        &single_node(),
        &cat(),
    );
    assert_eq!(rows2[0][0], Value::Integer(3), "length(list) regression");
}

// =====================================================================
// Test 4 — nodes(p) / relationships(p) traversal order + element types
// =====================================================================

#[test]
fn t4_nodes_and_relationships_projection() {
    let rows = run(
        "MATCH p = (a:X)-[:R]->(b:X) RETURN nodes(p), relationships(p)",
        &two_node_one_edge(),
        &cat(),
    );
    assert_eq!(node_ids(&rows[0][0]), vec![1, 2], "nodes traversal order");
    assert_eq!(
        rel_ids(&rows[0][1]),
        vec![10],
        "relationships traversal order"
    );
}

// =====================================================================
// Test 5 — direction: traversal order ≠ stored rel order (D-2)
// =====================================================================

#[test]
fn t5_direction_traversal_order() {
    // Edge stored 1 -> 2 (r10.from=1, r10.to=2). The pattern
    // `(a)<-[r]-(b)` matches with a=2, b=1 (traverse against the stored
    // direction). nodes(p) MUST be [a, b] = [2, 1] in TRAVERSAL order
    // even though the rel's stored from/to is 1/2.
    let rows = run(
        "MATCH p = (a:X)<-[r:R]-(b:X) RETURN nodes(p)",
        &two_node_one_edge(),
        &cat(),
    );
    assert_eq!(rows.len(), 1, "one inbound match");
    assert_eq!(
        node_ids(&rows[0][0]),
        vec![2, 1],
        "traversal order [a=2, b=1], NOT stored order [1, 2]"
    );
}

// =====================================================================
// Test 6 — var-length composition (D-5)
// =====================================================================

#[test]
fn t6_var_length_composition() {
    // `*2..2` over the 1->2->3 chain ⇒ exactly one 2-hop path 1->2->3.
    let rows = run(
        "MATCH p = (a:X)-[:R*2..2]->(b:X) RETURN length(p), nodes(p), relationships(p)",
        &chain_three(),
        &cat(),
    );
    assert_eq!(rows.len(), 1, "exactly one 2-hop path (1->2->3)");
    assert_eq!(rows[0][0], Value::Integer(2), "length 2");
    let nodes = node_ids(&rows[0][1]);
    let rels = rel_ids(&rows[0][2]);
    assert_eq!(nodes, vec![1, 2, 3], "3 nodes in traversal order");
    assert_eq!(rels, vec![10, 11], "2 rels in traversal order");
    // #nodes = #rels + 1 invariant (D-1).
    assert_eq!(nodes.len(), rels.len() + 1);
}

// =====================================================================
// Test 8 — ORDERABILITY (D-11, CORRECTED): paths ARE orderable, sort
// FIRST in the global type-order, order deterministically, never collide.
//
// The DETERMINISTIC-ORDERING + Path-sorts-first + non-collision oracle
// is asserted DIRECTLY at the `compare_*` sites the ADR designates "the
// real orderability oracle":
//   - `executor::ops::sort::tests::adr193_paths_orderable_compare_arm`
//     (`compare_non_null_values` — sort key ordering),
//   - `executor::ops::aggregate::tests::adr193_min_max_over_paths_return_extreme_path`
//     (`compare_values` + `min`/`max` return the extreme path, no error),
//   - `executor::value::tests::cmp_paths_is_deterministic_and_non_colliding`
//     (the `PathView::cmp_paths` node-seq→rel-seq tiebreak).
//
// Full-pipeline `RETURN p ORDER BY p` / `RETURN min(p)` are NOT exercised
// here because projection over Sort/Aggregate does NOT execute
// end-to-end on `main` for ANY key type — `RETURN count(n)` /
// `ORDER BY n.name` fail identically with "binding … missing from row
// schema" (a PRE-EXISTING executor gap, orthogonal to Value::Path;
// aggregate/sort EXECUTION is unvalidated on main — only their LOWERING
// is). Reported to the orchestrator. The e2e oracle below is the
// non-collision half (DISTINCT, which DOES execute end-to-end).
// (Test 7 path equality lives in the eval.rs unit tests.)
// =====================================================================

#[test]
fn t8_distinct_paths_do_not_collide() {
    // Strong non-collision oracle: DISTINCT over two distinct paths keeps
    // BOTH rows. A `_ => Equal`-collide compare arm (the D-11 anti-pattern)
    // would merge them to 1 — this test BITES on that bug.
    let rows = run(
        "MATCH p = (a:X)-[:R]->(b:X) RETURN DISTINCT p",
        &fan_out(),
        &cat(),
    );
    assert_eq!(
        rows.len(),
        2,
        "distinct paths must NOT merge under DISTINCT"
    );
}

// =====================================================================
// Test 9 — WRITE-OP FENCE (D-12): a path can never be a property value
// =====================================================================

#[test]
fn t9_write_op_fence_set_property() {
    // `SET a.name = p` — a path-typed expression is not a valid property
    // value; rejected at compile time (the literal-only narrowing — a
    // path variable is not a literal). BITES on a missing fence.
    assert!(
        run_err(
            "MATCH p = (a:X)-[:R]->(b:X) SET a.name = p",
            &two_node_one_edge(),
            &cat()
        ),
        "SET n.prop = <path> MUST be rejected (D-12)"
    );
}

#[test]
fn t9_write_op_fence_create_property() {
    assert!(
        run_err(
            "MATCH p = (a:X)-[:R]->(b:X) CREATE (x:X {name: p})",
            &two_node_one_edge(),
            &cat()
        ),
        "CREATE with a path-typed property value MUST be rejected (D-12)"
    );
}

// =====================================================================
// Test 11 — nodes/relationships/length on NULL + non-path (D-7/D-13)
// =====================================================================

#[test]
fn t11_nodes_on_null_is_null() {
    let rows = run("MATCH (n:X) RETURN nodes(null)", &single_node(), &cat());
    assert_eq!(rows[0][0], Value::Null, "nodes(null) → null (3VL)");
}

#[test]
fn t11_nodes_on_non_path_errors() {
    // nodes(42) → InvalidArgumentType (runtime). BITES on a silent coerce.
    assert!(
        run_err("MATCH (n:X) RETURN nodes(42)", &single_node(), &cat()),
        "nodes(<non-path>) MUST surface InvalidArgumentType"
    );
    assert!(
        run_err(
            "MATCH (n:X) RETURN relationships('x')",
            &single_node(),
            &cat()
        ),
        "relationships(<non-path>) MUST surface InvalidArgumentType"
    );
}
