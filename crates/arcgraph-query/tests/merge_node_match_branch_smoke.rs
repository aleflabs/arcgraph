//! ADR-151 W26-θ Phase 5 — MERGE node-shape match-branch smoke test.
//!
//! Pre-bake a `:User` node; then `MERGE (n:User)` — the match-branch's
//! Scan returns the pre-baked node → match-branch fires (NO new node
//! created). The test asserts the post-MERGE node count is unchanged.
//!
//! ADR-152-amendment-01 update: the match-branch now ENFORCES the
//! pattern label (`Scan{label: Some(id)}` when the label is interned —
//! §D-2). The pre-baked node's `LabelId` is therefore aligned with the
//! catalog's `User` id (`USER_LABEL`) so the label-enforced scan finds
//! it. Pre-amendment these tests pre-baked an UNRELATED label id and
//! "passed" only because the match-branch matched ANY node — the audit
//! O-3 bug this amendment closes.

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

use arcgraph_query::executor::substrate::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

/// The interned `LabelId` for `User`. The bind catalog maps the name
/// `User` to this id (§D-1), and the pre-baked node carries it, so the
/// label-enforced match-branch `Scan{label: Some(USER_LABEL)}` resolves
/// the pre-baked node.
const USER_LABEL: u32 = 1024;

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    // ADR-152-amendment-01 §D-1 — `User` is interned in the bind
    // catalog so the MERGE match-branch lowers to
    // `Scan{label: Some(USER_LABEL)}` (NOT the pre-amendment
    // label-agnostic `Scan{None}`).
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(USER_LABEL));
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

fn drain(plan: &LogicalPlan, stub: &StubExecutorSubstrate, ctx: &ExecutionContext) {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(ctx, stub).expect("batch OK");
        if b.is_empty() {
            break;
        }
    }
}

#[test]
fn merge_node_match_branch_fires_on_prebaked_node() {
    // ADR-151 §D-7 + ADR-152-amendment-01 §D-2: pre-bake a `:User`
    // node (label == USER_LABEL); `MERGE (n:User)` lowers to
    // `Scan{label: Some(USER_LABEL)}` → returns the pre-baked node →
    // match-branch fires → NO new node created.
    let tenant = TenantId::DEFAULT;
    let pre = NodeView::new(NodeId::new(1), Some(LabelId::new(USER_LABEL)));
    let s = StubExecutorSubstrate::new().with_node(tenant, pre);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let pre_count = s.scan_nodes(tenant, None, Lsn::MAX).unwrap().len();
    assert_eq!(pre_count, 1, "fixture: 1 pre-baked :User node");
    let plan = lower("MERGE (n:User)");
    drain(&plan, &s, &ctx);
    let post_count = s.scan_nodes(tenant, None, Lsn::MAX).unwrap().len();
    assert_eq!(
        post_count, 1,
        "label-enforced match-branch fired on the pre-baked :User; no \
         new node should have been created"
    );
}

#[test]
fn merge_node_idempotent_double_merge_on_prebaked() {
    // Calling MERGE twice on a pre-baked `:User` store — the
    // label-enforced match-branch fires both times; no duplicate is
    // created.
    let tenant = TenantId::DEFAULT;
    let pre = NodeView::new(NodeId::new(1), Some(LabelId::new(USER_LABEL)));
    let s = StubExecutorSubstrate::new().with_node(tenant, pre);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    let plan = lower("MERGE (n:User)");
    drain(&plan, &s, &ctx);
    drain(&plan, &s, &ctx);
    let count = s.scan_nodes(tenant, None, Lsn::MAX).unwrap().len();
    assert_eq!(
        count, 1,
        "MERGE-then-MERGE on pre-baked :User store is idempotent"
    );
}

#[test]
fn merge_bare_label_creates_when_prebaked_node_has_different_label() {
    // ADR-152-amendment-01 §D-2/§D-3 regression pin (the audit O-3 bug
    // the prior version of this test masked): a pre-baked node under a
    // DIFFERENT label must NOT satisfy `MERGE (n:User)`. With `User`
    // un-interned (absent from the catalog) the match-branch lowers to
    // `LogicalEmpty` (§D-3) → the create-branch fires → a real `:User`
    // is minted. Pre-amendment this matched the foreign-label node and
    // created nothing.
    let tenant = TenantId::DEFAULT;
    // Pre-bake a node under an UNRELATED label (not User).
    let other_label = LabelId::new(2048);
    let pre = NodeView::new(NodeId::new(1), Some(other_label));
    let s = StubExecutorSubstrate::new().with_node(tenant, pre);
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    // Bind with an EMPTY catalog → `lookup_label("User") == None` →
    // §D-3 LogicalEmpty match-branch → create-branch fires.
    let stmt = parse("MERGE (n:User)").expect("parse OK");
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, "MERGE (n:User)", &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK");
    drain(&plan, &s, &ctx);
    let post_count = s.scan_nodes(tenant, None, Lsn::MAX).unwrap().len();
    assert_eq!(
        post_count, 2,
        "the foreign-label node must NOT satisfy MERGE (n:User); a \
         :User is created (closes audit O-3)"
    );
    // The created node carries the minted User label (first interned by
    // the create-branch → 1024), distinct from the pre-baked 2048.
    let users = s
        .scan_nodes(tenant, Some(LabelId::new(USER_LABEL)), Lsn::MAX)
        .unwrap();
    assert_eq!(users.len(), 1, "exactly one :User was minted");
}
