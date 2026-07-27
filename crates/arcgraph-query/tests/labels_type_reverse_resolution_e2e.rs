//! **#871** — Bolt/MCP clients must read a node's label NAME (and a
//! relationship's type NAME), not the opaque interned id. END-TO-END
//! through the FULL `QueryEngine::execute` pipeline against a populated
//! stub substrate.
//!
//! # Root cause (this suite pins the fix)
//!
//! The catalog has FORWARD resolution (name → `LabelId`, used by
//! `MATCH (a:Account)` filters) but no REVERSE (`LabelId` → name) at the
//! read/serialize boundary. So before the fix:
//!   - `labels(n)` errored (`-32005` over Bolt) — no eval arm.
//!   - a MATCH-returned node serialized its label as `"LabelId(1)"`.
//!   - a CREATE-returned node carried NO label at all (`labels == []`).
//!
//! The fix carries the catalog-resolved NAME on `NodeView::label_name` /
//! `RelView::rel_type_name` (resolved at materialization — the same point
//! the property bag is resolved to string keys), which `labels()` /
//! `type()` + the JSON / Bolt serializers surface.
//!
//! # ADR-133 §D-4 "Query" active-verification gate
//!
//! Every assertion drives a REAL ArcQL query through `QueryEngine::execute`
//! against a POPULATED substrate and asserts the returned rows == a
//! hand-computed oracle. The discriminating pin is the NAME string
//! (`"Account"` / `"Widget"` / `"KNOWS"`), never the id form.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::StubCatalogProvider;

const ACCOUNT: u32 = 1;
const KNOWS: u32 = 1;

/// A labeled `Account` node carrying the catalog-resolved NAME — exactly
/// what production materialization (`CrudExecutorSubstrate::scan`)
/// populates by reverse-resolving the `LabelId` via the intern table.
fn account(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(ACCOUNT)))
        .with_label_name("Account")
        .with_property("id", Value::Integer(id as i64))
}

/// An UNLABELED node (no label, no name) — `labels()` must return `[]`.
fn anon(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), None)
}

/// Two `Account`s with `a1-[:KNOWS]->a2`; the edge carries the resolved
/// rel-type NAME (production `expand` reverse-resolves it).
fn graph() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, account(1))
        .with_node(TenantId::DEFAULT, account(2))
        .with_node(TenantId::DEFAULT, anon(3))
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(1),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(KNOWS)),
            )
            .with_rel_type_name("KNOWS"),
        )
}

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Account"])
        .with_rel_types(["KNOWS"])
        .with_properties(["id"])
}

fn run(cypher: &str) -> Vec<Vec<Value>> {
    QueryEngine::new(&catalog())
        .execute(cypher, &graph())
        .unwrap_or_else(|e| panic!("execute must not error for `{cypher}`: {e:?}"))
        .rows
}

/// Single-cell helper: the sole cell of the sole row.
fn one_cell(cypher: &str) -> Value {
    let rows = run(cypher);
    assert_eq!(rows.len(), 1, "expected exactly 1 row for `{cypher}`");
    assert_eq!(rows[0].len(), 1, "expected exactly 1 cell for `{cypher}`");
    rows[0][0].clone()
}

// =====================================================================
// FACET 1 — `labels(n)` returns the label NAME(s) as a list of strings.
// (Pre-fix: `-32005` SyntaxError — no eval arm; `labels` fell through to
// the NotImplemented catch-all.)
// =====================================================================

#[test]
fn facet1_labels_returns_name_list() {
    // Oracle: labels(a) == ['Account'] (the NAME, exact ==).
    assert_eq!(
        one_cell("MATCH (a:Account) WHERE a.id = 1 RETURN labels(a)"),
        Value::List(vec![Value::String("Account".to_string())]),
    );
}

#[test]
fn facet1_labels_index_zero_is_the_name_string() {
    // Oracle: labels(a)[0] == 'Account' (the task's explicit pin).
    assert_eq!(
        one_cell("MATCH (a:Account) WHERE a.id = 1 RETURN labels(a)[0]"),
        Value::String("Account".to_string()),
    );
}

#[test]
fn facet1_labels_does_not_error() {
    // The pre-fix behaviour raised `-32005`; `labels()` must return Ok.
    let r = QueryEngine::new(&catalog()).execute("MATCH (a:Account) RETURN labels(a)", &graph());
    assert!(r.is_ok(), "labels() must NOT error (#871): {r:?}");
}

#[test]
fn facet1_unlabeled_node_labels_is_empty_list() {
    // openCypher: a node with no label ⇒ labels(n) == [] (NOT null,
    // NOT an error). Node 3 is anonymous.
    assert_eq!(
        one_cell("MATCH (a) WHERE a.id IS NULL RETURN labels(a)"),
        Value::List(vec![]),
    );
}

#[test]
fn facet1_labels_never_leaks_labelid_debug_form() {
    // The exact #871 regression guard: the cell is NEVER the
    // `"LabelId(1)"` debug string under any reachable path.
    let cell = one_cell("MATCH (a:Account) WHERE a.id = 1 RETURN labels(a)");
    let Value::List(items) = cell else {
        panic!("labels() must be a list, got {cell:?}");
    };
    for item in items {
        let Value::String(s) = item else {
            panic!("label element must be a String, got {item:?}");
        };
        assert!(
            !s.starts_with("LabelId("),
            "label name must be resolved, not the id debug form: {s:?}"
        );
    }
}

// =====================================================================
// FACET 2 — a MATCH-returned node serializes its label NAME. The MCP
// `graph.raw_query` path encodes each cell via `Value::to_json_value`
// (raw_query.rs §"Each row is ... JSON-encoded cells via to_json_value").
// (Pre-fix MCP JSON carried the numeric id; Bolt carried "LabelId(1)".)
// =====================================================================

#[test]
fn facet2_match_returned_node_json_carries_label_name() {
    let cell = one_cell("MATCH (a:Account) WHERE a.id = 1 RETURN a");
    let Value::Node(n) = &cell else {
        panic!("RETURN a must yield a Node, got {cell:?}");
    };
    // The in-memory view carries the resolved name.
    assert_eq!(n.label_name.as_deref(), Some("Account"));

    // The MCP/raw_query wire shape (JSON) carries the Neo4j-style
    // `labels` list == ["Account"] (the value that reaches the client).
    let json = cell.to_json_value();
    assert_eq!(
        json.get("labels"),
        Some(&serde_json::json!(["Account"])),
        "MCP node JSON must surface labels=['Account'], got {json}"
    );
}

// =====================================================================
// FACET 3 — a CREATE-returned node carries its label NAME. (Pre-fix:
// `node.labels == []` — `label_id_from_name` dropped the label entirely.)
// =====================================================================

#[test]
fn facet3_create_returned_node_carries_label_name() {
    let cell = one_cell("CREATE (d:Widget) RETURN d");
    let Value::Node(n) = &cell else {
        panic!("CREATE … RETURN d must yield a Node, got {cell:?}");
    };
    assert_eq!(
        n.label_name.as_deref(),
        Some("Widget"),
        "CREATE-returned node must carry its label name"
    );
    let json = cell.to_json_value();
    assert_eq!(json.get("labels"), Some(&serde_json::json!(["Widget"])));
}

#[test]
fn facet3_create_then_labels_is_widget() {
    // Oracle: CREATE (d:Widget) RETURN labels(d) == ['Widget'] (not []).
    assert_eq!(
        one_cell("CREATE (d:Widget) RETURN labels(d)"),
        Value::List(vec![Value::String("Widget".to_string())]),
    );
}

// =====================================================================
// SISTER BUG — `type(r)` returns the rel-type NAME (same root cause:
// TypeId → name reverse-resolution). (Pre-fix: no eval arm ⇒ same class
// as labels(); Bolt packed "TypeId(1)".)
// =====================================================================

#[test]
fn sister_type_returns_rel_type_name() {
    // Oracle: type(r) == 'KNOWS' (the NAME, not "TypeId(1)").
    assert_eq!(
        one_cell("MATCH (a:Account)-[r:KNOWS]->(b) RETURN type(r)"),
        Value::String("KNOWS".to_string()),
    );
}

#[test]
fn sister_match_returned_rel_json_carries_type_name() {
    let cell = one_cell("MATCH (a:Account)-[r:KNOWS]->(b) RETURN r");
    let Value::Relationship(r) = &cell else {
        panic!("RETURN r must yield a Relationship, got {cell:?}");
    };
    assert_eq!(r.rel_type_name.as_deref(), Some("KNOWS"));
    let json = cell.to_json_value();
    assert_eq!(
        json.get("type"),
        Some(&serde_json::json!("KNOWS")),
        "MCP rel JSON must surface type='KNOWS', got {json}"
    );
}
