//! ADR-149 W26-θ Phase 3 — bare DELETE (no DETACH) over a node with
//! attached rels MUST surface a substrate error mapped to
//! `ExecutionError::Substrate(SubstrateAccessError::Io("..."))` per
//! ADR-149 §D-1 / §D-7. This pins the openCypher v9 §6 contract that
//! bare DELETE protects referential integrity.

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::{StubExecutorSubstrate, SubstrateAccessError};
use arcgraph_query::executor::value::{NodeView, RelView};

#[test]
fn bare_delete_on_node_with_attached_rels_surfaces_error() {
    // Build a fixture: n1 -[r]-> n2. Calling delete_node with
    // detach=false MUST return SubstrateAccessError::Io with the
    // "relationships attached" message.
    let tenant = TenantId::DEFAULT;
    let lbl = LabelId::new(1024);
    let n1 = NodeView::new(NodeId::new(1), Some(lbl));
    let n2 = NodeView::new(NodeId::new(2), Some(lbl));
    let r = RelView::new(RelId::new(100), n1.id, n2.id, Some(TypeId::new(1024)));
    let s = StubExecutorSubstrate::new()
        .with_node(tenant, n1.clone())
        .with_node(tenant, n2.clone())
        .with_edge(tenant, r);

    let result = s.delete_node(
        tenant,
        n1.id,
        false,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    );
    assert!(
        matches!(result, Err(SubstrateAccessError::Io(_))),
        "expected Io error for bare DELETE over attached node, got {result:?}"
    );
    // Message contains the canonical openCypher v9 §6 cue.
    if let Err(SubstrateAccessError::Io(msg)) = &result {
        assert!(
            msg.contains("relationships attached") || msg.contains("DETACH DELETE"),
            "Io message references attached rels: {msg}"
        );
    }

    // The node is STILL there — the bare-DELETE failure did not
    // tombstone anything (no partial side effect).
    let post = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(
        post.len(),
        2,
        "n1 + n2 both still visible after rejected bare DELETE: {post:?}"
    );
}

#[test]
fn bare_delete_on_isolated_node_succeeds() {
    // Sanity: bare DELETE over a node with NO attached rels works.
    let tenant = TenantId::DEFAULT;
    let lbl = LabelId::new(1024);
    let n = NodeView::new(NodeId::new(7), Some(lbl));
    let s = StubExecutorSubstrate::new().with_node(tenant, n.clone());
    let result = s.delete_node(
        tenant,
        n.id,
        false,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    );
    assert!(result.is_ok(), "isolated node bare-DELETE OK: {result:?}");
    let post = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(post.len(), 0, "isolated node tombstoned: {post:?}");
}

#[test]
fn bare_delete_then_detach_delete_succeeds() {
    // Retry-after-fail pattern: bare DELETE fails on n1 (attached
    // rels exist); DETACH DELETE then succeeds.
    let tenant = TenantId::DEFAULT;
    let lbl = LabelId::new(1024);
    let n1 = NodeView::new(NodeId::new(1), Some(lbl));
    let n2 = NodeView::new(NodeId::new(2), Some(lbl));
    let r = RelView::new(RelId::new(100), n1.id, n2.id, Some(TypeId::new(1024)));
    let s = StubExecutorSubstrate::new()
        .with_node(tenant, n1.clone())
        .with_node(tenant, n2.clone())
        .with_edge(tenant, r);
    // First: bare DELETE fails.
    let bare = s.delete_node(
        tenant,
        n1.id,
        false,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    );
    assert!(bare.is_err());
    // Then: DETACH DELETE succeeds.
    let detach = s.delete_node(
        tenant,
        n1.id,
        true,
        &arcgraph_query::executor::ExecutionContext::new(tenant, arcgraph_core::PartitionId::ZERO),
    );
    assert!(detach.is_ok(), "DETACH DELETE OK: {detach:?}");
    // Post: n2 remains; n1 + rel tombstoned.
    let post = s
        .scan_nodes(tenant, None, arcgraph_core::Lsn::MAX)
        .expect("scan_nodes OK");
    assert_eq!(post.len(), 1);
    assert_eq!(post[0].node.id, n2.id);
}
