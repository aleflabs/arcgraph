//! **#870 / TCK `Literals7`/`8`** — negative numeric literals in COLLECTION
//! (list/map literal elements) and PROPERTY-VALUE (CREATE / SET) positions,
//! END-TO-END through `QueryEngine::execute`.
//!
//! # Root cause (shared)
//!
//! A negative numeric literal parses as `UnaryOp(Neg, <numeric literal>)`,
//! NOT a bare `Literal` (`-5` ⇒ `UnaryOp(Neg, Integer(5))`). The parse is
//! correct everywhere; the bug was literal-ONLY handling that dropped the
//! `UnaryOp` to `Null` (collection read-path) or rejected it as "not a
//! literal" (CREATE/SET write-path). This suite is the strong oracle: it
//! asserts the EXACT negative `Value` (`==`), not "parses" — so a revert of
//! any of the three fix sites (eval, literal_lift, type_check) turns it RED.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

/// Execute against a bare catalog + empty substrate; return the single cell.
fn cell(cypher: &str) -> Value {
    let rows = QueryEngine::new(&StubCatalogProvider::new())
        .execute(cypher, &StubExecutorSubstrate::new())
        .unwrap_or_else(|e| panic!("execute `{cypher}`: {e:?}"))
        .rows;
    assert_eq!(rows.len(), 1, "one row for `{cypher}`");
    assert_eq!(rows[0].len(), 1, "one column for `{cypher}`");
    rows[0][0].clone()
}

fn list(vs: Vec<Value>) -> Value {
    Value::List(vs)
}
fn i(n: i64) -> Value {
    Value::Integer(n)
}

// =====================================================================
// PART A — COLLECTION read-path (TCK Literals7/8). EXACT values.
// =====================================================================

#[test]
fn negative_int_in_list() {
    assert_eq!(cell("RETURN [-5] AS x"), list(vec![i(-5)]));
    assert_eq!(
        cell("RETURN [1, 2, -3] AS x"),
        list(vec![i(1), i(2), i(-3)])
    );
}

#[test]
fn negative_hex_in_list() {
    // `[-0x1f]` ⇒ [-31] (TCK Literals7 [5]).
    assert_eq!(cell("RETURN [-0x1f] AS x"), list(vec![i(-31)]));
}

#[test]
fn negative_float_in_list() {
    // `[-.1e-5]` ⇒ [-0.000001] (TCK Literals7 [7]).
    match cell("RETURN [-.1e-5] AS x") {
        Value::List(v) => match v.as_slice() {
            [Value::Float(f)] => assert!((f - -1e-6).abs() < 1e-18, "got {f}"),
            other => panic!("expected [Float], got {other:?}"),
        },
        other => panic!("expected List, got {other:?}"),
    }
}

#[test]
fn negative_in_map() {
    // `{k: -5}` ⇒ {k: -5} (TCK Literals8 [9]/[11] shape).
    match cell("RETURN {k: -5} AS x") {
        Value::Map(m) => assert_eq!(m.get("k"), Some(&i(-5))),
        other => panic!("expected Map, got {other:?}"),
    }
}

#[test]
fn nested_negatives_recurse() {
    assert_eq!(
        cell("RETURN [-1, [-2, -3]] AS x"),
        list(vec![i(-1), list(vec![i(-2), i(-3)])])
    );
}

#[test]
fn unary_plus_in_list_is_identity() {
    assert_eq!(cell("RETURN [+5] AS x"), list(vec![i(5)]));
}

#[test]
fn positive_in_collection_still_works() {
    // Regression guard — the fix must not perturb the already-working
    // positive path.
    assert_eq!(cell("RETURN [5] AS x"), list(vec![i(5)]));
    match cell("RETURN {k: 5} AS x") {
        Value::Map(m) => assert_eq!(m.get("k"), Some(&i(5))),
        other => panic!("expected Map, got {other:?}"),
    }
}

// =====================================================================
// PART B — PROPERTY-VALUE write-path (#870 — CREATE / SET). EXACT values.
// =====================================================================

#[test]
fn create_node_negative_property() {
    assert_eq!(cell("CREATE (n {x: -5}) RETURN n.x"), i(-5));
}

#[test]
fn create_node_negative_float_property() {
    match cell("CREATE (n {x: -.1e-5}) RETURN n.x") {
        Value::Float(f) => assert!((f - -1e-6).abs() < 1e-18, "got {f}"),
        other => panic!("expected Float, got {other:?}"),
    }
}

#[test]
fn set_negative_property() {
    assert_eq!(cell("CREATE (n) SET n.x = -3 RETURN n.x"), i(-3));
}

#[test]
fn create_node_negative_list_property() {
    assert_eq!(
        cell("CREATE (n {xs: [-1, -2]}) RETURN n.xs"),
        list(vec![i(-1), i(-2)])
    );
}

#[test]
fn create_positive_property_still_works() {
    assert_eq!(cell("CREATE (n {x: 5}) RETURN n.x"), i(5));
}

// =====================================================================
// PART C — scope guard: post ADR-147-amendment-03 (D-1), a variable /
// property-access property value is ADMITTED at type-check (it may
// resolve to a persistable scalar). A variable bound to a NODE, however,
// is rejected at the RUNTIME value-type gate — an entity is never a
// persistable property value (openCypher / ADR-191 D-11). The rejection
// MOVED from type-check to execution; it is still a rejection.
// =====================================================================

#[test]
fn entity_valued_property_rejected_at_runtime_value_gate() {
    use arcgraph_core::{NodeId, PartitionId, TenantId};
    use arcgraph_query::executor::ExecutionContext;
    use arcgraph_query::executor::value::NodeView;

    // Seed one node so `MATCH (m)` yields a row that flows into the CREATE.
    let tenant = TenantId::DEFAULT;
    let substrate =
        StubExecutorSubstrate::new().with_node(tenant, NodeView::new(NodeId::new(1), None));
    let _ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

    // `{x: m}` — `m` is a bound Node. Admitted at type-check (VariableRef),
    // but the runtime value-type gate rejects the Node before the write.
    let r = QueryEngine::new(&StubCatalogProvider::new())
        .execute("MATCH (m) CREATE (n {x: m}) RETURN n", &substrate);
    assert!(
        r.is_err(),
        "a Node-valued CREATE property must be rejected at the runtime value gate"
    );
    let msg = format!("{:?}", r.err());
    assert!(
        msg.contains("Node") || msg.contains("property value"),
        "the rejection names the entity/property fence; got {msg}"
    );
}
