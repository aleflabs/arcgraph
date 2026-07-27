//! ADR-149 W26-θ Phase 3 — DETACH DELETE end-to-end smoke test.
//!
//! Pins the DETACH-cascade semantic at the Stub-substrate layer:
//! deleting a node with attached rels via `delete_node(_, _, detach=true)`
//! tombstones every attached rel FIRST, then the node itself.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView};

#[test]
fn detach_delete_node_cascade_tombstones_rels_first() {
    // Build a fixture: n1 -[r:KNOWS]-> n2. DETACH DELETE n1 should
    // tombstone r THEN n1; n2 remains.
    let tenant = TenantId::DEFAULT;
    let lbl = LabelId::new(1024);
    let n1 = NodeView::new(NodeId::new(1), Some(lbl));
    let n2 = NodeView::new(NodeId::new(2), Some(lbl));
    let r = RelView::new(RelId::new(100), n1.id, n2.id, Some(TypeId::new(1024)));
    let s = StubExecutorSubstrate::new()
        .with_node(tenant, n1.clone())
        .with_node(tenant, n2.clone())
        .with_edge(tenant, r.clone());

    // Sanity: both nodes + the rel are initially visible.
    let pre_nodes = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(pre_nodes.len(), 2);
    let pre_edges = s
        .expand(
            tenant,
            n1.id,
            None,
            arcgraph_query::logical_plan::Direction::LeftToRight,
            arcgraph_core::Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(pre_edges.len(), 1);

    // DETACH DELETE n1 — cascade-tombstones r first, then n1.
    s.delete_node(
        tenant,
        n1.id,
        true,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("DETACH DELETE OK");

    // n1 is gone; n2 remains.
    let post_nodes = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(post_nodes.len(), 1, "only n2 remains: {post_nodes:?}");
    assert_eq!(post_nodes[0].node.id, n2.id);
    // r is tombstoned — `expand` from any direction returns 0.
    let post_edges = s
        .expand(
            tenant,
            n2.id,
            None,
            arcgraph_query::logical_plan::Direction::Undirected,
            arcgraph_core::Lsn::MAX,
        )
        .expect("expand OK");
    assert_eq!(post_edges.len(), 0, "rel cascade-deleted: {post_edges:?}");
}

#[test]
fn detach_delete_isolated_node_succeeds_without_cascade() {
    // A node with no attached rels: DETACH=true is a no-op for the
    // cascade walk; the node itself is tombstoned.
    let tenant = TenantId::DEFAULT;
    let lbl = LabelId::new(1024);
    let n = NodeView::new(NodeId::new(7), Some(lbl));
    let s = StubExecutorSubstrate::new().with_node(tenant, n.clone());
    s.delete_node(
        tenant,
        n.id,
        true,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("DETACH OK");
    let post = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(post.len(), 0, "isolated node tombstoned: {post:?}");
}

#[test]
fn detach_delete_with_multiple_rels_cascades_all() {
    // n1 has 2 outbound rels + 1 inbound rel. DETACH DELETE
    // tombstones all 3 + the node.
    let tenant = TenantId::DEFAULT;
    let lbl = LabelId::new(1024);
    let n1 = NodeView::new(NodeId::new(1), Some(lbl));
    let n2 = NodeView::new(NodeId::new(2), Some(lbl));
    let n3 = NodeView::new(NodeId::new(3), Some(lbl));
    let n4 = NodeView::new(NodeId::new(4), Some(lbl));
    let r12 = RelView::new(RelId::new(100), n1.id, n2.id, Some(TypeId::new(1024)));
    let r13 = RelView::new(RelId::new(101), n1.id, n3.id, Some(TypeId::new(1024)));
    let r41 = RelView::new(RelId::new(102), n4.id, n1.id, Some(TypeId::new(1024)));
    let s = StubExecutorSubstrate::new()
        .with_node(tenant, n1.clone())
        .with_node(tenant, n2.clone())
        .with_node(tenant, n3.clone())
        .with_node(tenant, n4.clone())
        .with_edge(tenant, r12)
        .with_edge(tenant, r13)
        .with_edge(tenant, r41);

    s.delete_node(
        tenant,
        n1.id,
        true,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    )
    .expect("DETACH OK");

    // n1 is gone; n2/n3/n4 remain.
    let post = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(post.len(), 3, "3 surviving nodes: {post:?}");
    // No expand from / to any survivor finds n1's rels.
    for node in &[n2.id, n3.id, n4.id] {
        let edges = s
            .expand(
                tenant,
                *node,
                None,
                arcgraph_query::logical_plan::Direction::Undirected,
                arcgraph_core::Lsn::MAX,
            )
            .expect("expand OK");
        assert_eq!(
            edges.len(),
            0,
            "no edges visible after DETACH DELETE n1 (from {node:?}): {edges:?}"
        );
    }
}
