//! #904 — O(1) counts-store fast path for exact unfiltered count queries.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::logical_plan::CountStoreSource;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{PlanTree, PlanTreeOp, QueryEngine, explain};

fn catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_label_id("A", LabelId::new(1))
        .with_label_id("B", LabelId::new(2))
        .with_rel_type_id("T", TypeId::new(1))
        .with_rel_type_id("U", TypeId::new(2))
        .with_properties(["x"])
}

fn run(query: &str, substrate: &StubExecutorSubstrate) -> Vec<Vec<Value>> {
    QueryEngine::new(&catalog())
        .execute(query, substrate)
        .expect("execute")
        .rows
}

fn assert_count(query: &str, substrate: &StubExecutorSubstrate, expected: i64) {
    assert_eq!(run(query, substrate), vec![vec![Value::Integer(expected)]]);
}

fn contains_op(tree: &PlanTree, op: PlanTreeOp) -> bool {
    tree.op == op || tree.children.iter().any(|child| contains_op(child, op))
}

fn assert_count_store_plan(query: &str) {
    let tree = explain(query, &catalog()).expect("explain");
    assert!(
        contains_op(&tree, PlanTreeOp::CountStore),
        "expected CountStore in plan: {tree:#?}"
    );
    assert!(
        !contains_op(&tree, PlanTreeOp::Scan),
        "fast-path plan must not scan: {tree:#?}"
    );
}

fn assert_no_count_store_plan(query: &str) {
    let tree = explain(query, &catalog()).expect("explain");
    assert!(
        !contains_op(&tree, PlanTreeOp::CountStore),
        "expected no CountStore in plan: {tree:#?}"
    );
}

#[test]
fn unfiltered_node_count_reads_total_node_count() {
    let s = StubExecutorSubstrate::new().with_count_store_total(
        TenantId::DEFAULT,
        CountStoreSource::Nodes,
        875_713,
    );

    assert_count("MATCH (n) RETURN count(n)", &s, 875_713);
    assert_count("MATCH (n) RETURN count(*)", &s, 875_713);
    assert_count_store_plan("MATCH (n) RETURN count(n)");
}

#[test]
fn unfiltered_relationship_count_reads_total_rel_count() {
    let s = StubExecutorSubstrate::new().with_count_store_total(
        TenantId::DEFAULT,
        CountStoreSource::Relationships,
        5_105_039,
    );

    assert_count("MATCH ()-[r]->() RETURN count(r)", &s, 5_105_039);
    assert_count("MATCH ()-->() RETURN count(*)", &s, 5_105_039);
    assert_count_store_plan("MATCH ()-[r]->() RETURN count(r)");
}

#[test]
fn label_filtered_count_does_not_use_total_node_count() {
    let s = StubExecutorSubstrate::new()
        .with_count_store_total(TenantId::DEFAULT, CountStoreSource::Nodes, 999)
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(2))),
        );

    assert_count("MATCH (n:A) RETURN count(n)", &s, 2);
    // F1 (#1356 §F1): the labelled count now LOWERS to the count-store
    // (the per-label counter path), not a Scan+Aggregate. The answer is
    // still exact (2) and it still does NOT read the tenant-wide total
    // (999) — a different label keys a different counter.
    assert_count_store_plan("MATCH (n:A) RETURN count(n)");
}

#[test]
fn where_filtered_count_does_not_use_total_node_count() {
    let s = StubExecutorSubstrate::new()
        .with_count_store_total(TenantId::DEFAULT, CountStoreSource::Nodes, 999)
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), None).with_property("x", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), None).with_property("x", Value::Integer(1)),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), None).with_property("x", Value::Integer(2)),
        );

    assert_count("MATCH (n) WHERE n.x = 1 RETURN count(n)", &s, 2);
}

#[test]
fn rel_type_filtered_count_does_not_use_total_rel_count() {
    let s = StubExecutorSubstrate::new()
        .with_count_store_total(TenantId::DEFAULT, CountStoreSource::Relationships, 999)
        .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(1), None))
        .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(2), None))
        .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(3), None))
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(10),
                NodeId::new(1),
                NodeId::new(2),
                Some(TypeId::new(1)),
            ),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(11),
                NodeId::new(2),
                NodeId::new(3),
                Some(TypeId::new(2)),
            ),
        );

    assert_count("MATCH (a)-[r:T]->(b) RETURN count(r)", &s, 1);
    // F1 (#1356 §F1): the typed rel count now lowers to the per-type
    // counter path (RelsWithType), not a scan + per-row expand. Exact (1),
    // and it does NOT read the tenant-wide rel total (999).
    assert_count_store_plan("MATCH (a)-[r:T]->(b) RETURN count(r)");
}

#[test]
fn undirected_relationship_count_does_not_use_total_rel_count() {
    let s = StubExecutorSubstrate::new()
        .with_count_store_total(TenantId::DEFAULT, CountStoreSource::Relationships, 999)
        .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(1), None))
        .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(2), None))
        .with_node(TenantId::DEFAULT, NodeView::new(NodeId::new(3), None))
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(RelId::new(10), NodeId::new(1), NodeId::new(2), None),
        )
        .with_edge(
            TenantId::DEFAULT,
            RelView::new(RelId::new(11), NodeId::new(2), NodeId::new(3), None),
        );

    let rows = run("MATCH ()-[r]-() RETURN r", &s);
    assert_eq!(rows.len(), 4);
    assert_count("MATCH ()-[r]-() RETURN count(r)", &s, rows.len() as i64);
    assert_count("MATCH ()--() RETURN count(*)", &s, rows.len() as i64);
    assert_no_count_store_plan("MATCH ()-[r]-() RETURN count(r)");
    assert_no_count_store_plan("MATCH ()--() RETURN count(*)");
}

// ─────────────────────────────────────────────────────────────────────
// F1 (#1356 §F1) — labelled node counts + typed rel counts lower to the
// count-store (the existing per-label / per-type CatalogStats counters),
// an O(1) read instead of a full scan.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn f1_labeled_node_count_reads_per_label_counter_o1() {
    // Seed the per-label counter with a large value and ZERO fixture
    // nodes: a count of 875_713 proves the O(1) count-store path served it
    // (a full scan would return 0). This is the labelled analogue of
    // `unfiltered_node_count_reads_total_node_count`.
    let s = StubExecutorSubstrate::new().with_count_store_total(
        TenantId::DEFAULT,
        CountStoreSource::NodesWithLabel(LabelId::new(1)),
        875_713,
    );

    assert_count("MATCH (n:A) RETURN count(n)", &s, 875_713);
    assert_count("MATCH (n:A) RETURN count(*)", &s, 875_713);
    // THE F1 point: the plan lowers to CountStore with no Scan.
    assert_count_store_plan("MATCH (n:A) RETURN count(n)");
    assert_count_store_plan("MATCH (n:A) RETURN count(*)");
    // The source is keyed on `LabelId`, so a DIFFERENT label reads its own
    // (unseeded → 0) counter — B does not alias A's 875_713.
    assert_count("MATCH (n:B) RETURN count(n)", &s, 0);
}

#[test]
fn f1_typed_rel_count_reads_per_type_counter_o1() {
    // Seed the per-rel-type counter high with zero fixture edges: a count
    // of 5_105_039 proves the O(1) count-store path (not a scan+expand).
    let s = StubExecutorSubstrate::new().with_count_store_total(
        TenantId::DEFAULT,
        CountStoreSource::RelsWithType(TypeId::new(1)),
        5_105_039,
    );

    assert_count("MATCH (a)-[r:T]->(b) RETURN count(r)", &s, 5_105_039);
    assert_count("MATCH (a)-[:T]->(b) RETURN count(*)", &s, 5_105_039);
    assert_count_store_plan("MATCH (a)-[r:T]->(b) RETURN count(r)");
    assert_count_store_plan("MATCH (a)-[:T]->(b) RETURN count(*)");
    // A different rel-type reads its own (unseeded → 0) counter.
    assert_count("MATCH (a)-[:U]->(b) RETURN count(*)", &s, 0);
}

#[test]
fn f1_labeled_count_over_fixture_is_exact_via_count_store() {
    // Without a seeded counter the stub falls back to a filtered scan —
    // the answer must still be EXACT (the O(1) win is the production
    // catalog path; the stub owes correctness). Two `:A`, one `:B`.
    let s = StubExecutorSubstrate::new()
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(1), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(2), Some(LabelId::new(1))),
        )
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(3), Some(LabelId::new(2))),
        );

    assert_count("MATCH (n:A) RETURN count(n)", &s, 2);
    assert_count("MATCH (n:B) RETURN count(n)", &s, 1);
    assert_count_store_plan("MATCH (n:A) RETURN count(n)");
}

#[test]
fn f1b_labeled_anchor_rel_count_stays_on_scan_path() {
    // Scope guard: F1 covers a SINGLE rel-type over an UNLABELLED anchor.
    // A labelled anchor (`(a:A)-[:T]->(b)`) is the F1b `(src,type,dst)`
    // triple form — explicitly OUT of F1 scope (the per-type counter can
    // NOT answer a source-label-filtered count). It must NOT lower to the
    // count store; it stays on the scan path.
    assert_no_count_store_plan("MATCH (a:A)-[:T]->(b) RETURN count(*)");
    assert_no_count_store_plan("MATCH (a:A)-[r:T]->(b) RETURN count(r)");
}
