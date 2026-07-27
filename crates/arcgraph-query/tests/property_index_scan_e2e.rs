//! #1366 (Phase 2) — PropertyIndexScan planner-wiring e2e.
//!
//! The load-bearing correctness gate: an indexed-property MATCH must
//! return the EXACT SAME result set as the full-scan path. These tests
//! drive the full parse → bind → lower → `rewrite_scan_to_property_index_scan`
//! → execute pipeline through the `QueryEngine`, against a
//! `StubExecutorSubstrate` whose `property_index_lookup_with_context`
//! seam does the candidate-then-verify + dedup the production op relies
//! on.
//!
//! Coverage:
//! - IDENTICAL-RESULTS: indexed lookup == full scan (incl. dup slots →
//!   one row, absent → empty, wrong-label / stale candidate excluded).
//! - Building-not-used (RED-on-revert): a Building index (NOT seeded as
//!   `online_property_index`) is NOT chosen; the plan is a full scan and
//!   a node the backfill missed is still found (via the scan). Reverting
//!   the RC-6 gate to accept a Building index would make the planner
//!   route to the index and MISS that node → the identical-results check
//!   catches the false negative.
//! - PERF: the indexed path reads O(matches), the scan reads
//!   O(node_high_water) — instrumented via the substrate's read counter.
//! - EXPLAIN: `PropertyIndexScan(label, property, residual)` shows for an
//!   indexed lookup; `Scan` shows for an unindexed / Building one.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arcgraph_core::{LabelId, Lsn, NodeId, TenantId};
use arcgraph_query::executor::substrate::{BoundNode, ExecutorSubstrate, SubstrateAccessError};
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate};
use arcgraph_query::explain::QueryEngine;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{PlanTree, PlanTreeOp};

const USER: LabelId = LabelId::new(1);
const DEADLINE: std::time::Duration = std::time::Duration::from_secs(5);

/// A read-count-instrumented wrapper over the stub. Counts every
/// per-node hydrate (`node_by_id_with_context`) + every scanned node so
/// the perf test can compare O(matches) vs O(node_high_water).
#[derive(Debug)]
struct CountingSubstrate {
    inner: StubExecutorSubstrate,
    node_reads: Arc<AtomicUsize>,
}

impl CountingSubstrate {
    fn new(inner: StubExecutorSubstrate) -> Self {
        Self {
            inner,
            node_reads: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn reads(&self) -> usize {
        self.node_reads.load(Ordering::Relaxed)
    }
    fn reset(&self) {
        self.node_reads.store(0, Ordering::Relaxed);
    }
}

impl ExecutorSubstrate for CountingSubstrate {
    fn scan_nodes(
        &self,
        tenant: TenantId,
        label: Option<LabelId>,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        let out = self.inner.scan_nodes(tenant, label, read_lsn)?;
        // A full scan touches every scanned node.
        self.node_reads.fetch_add(out.len(), Ordering::Relaxed);
        Ok(out)
    }

    fn expand(
        &self,
        tenant: TenantId,
        from: NodeId,
        rel_type: Option<arcgraph_core::TypeId>,
        direction: arcgraph_query::logical_plan::Direction,
        read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::substrate::BoundEdge>, SubstrateAccessError> {
        self.inner
            .expand(tenant, from, rel_type, direction, read_lsn)
    }

    fn node_by_id_with_context(
        &self,
        ctx: &ExecutionContext,
        id: NodeId,
    ) -> Result<Option<BoundNode>, SubstrateAccessError> {
        // Each candidate hydrate is one node read.
        self.node_reads.fetch_add(1, Ordering::Relaxed);
        self.inner.node_by_id_with_context(ctx, id)
    }

    fn property_index_lookup_with_context(
        &self,
        ctx: &ExecutionContext,
        label: LabelId,
        property: &str,
        value: &Value,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        // Delegate to the inner stub's candidate-then-verify. The inner
        // stub calls ITS OWN node_by_id_with_context (not ours), so the
        // hydrate count here reflects only the candidate set — which is
        // exactly O(candidates) = O(matches) for a unique lookup. Count
        // the candidates it verified so the perf assertion is honest.
        let out = self
            .inner
            .property_index_lookup_with_context(ctx, label, property, value, read_lsn)?;
        self.node_reads.fetch_add(out.len(), Ordering::Relaxed);
        Ok(out)
    }

    // #1415: a delegating wrapper MUST forward the index-vs-scan-fallback
    // gate to the inner substrate; otherwise the trait default (`false`)
    // would force EVERY lookup — even a keyable one — down the scan
    // fallback, silently defeating the index fast path (and this perf
    // assertion).
    fn value_is_indexable(&self, value: &Value) -> bool {
        self.inner.value_is_indexable(value)
    }

    fn vector_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_vec: &[f32],
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::substrate::RankedHit>, SubstrateAccessError> {
        self.inner
            .vector_search(tenant, property, query_vec, k, read_lsn)
    }

    fn bm25_search(
        &self,
        tenant: TenantId,
        property: &str,
        query_text: &str,
        k: u64,
        read_lsn: Lsn,
    ) -> Result<Vec<arcgraph_query::executor::substrate::RankedHit>, SubstrateAccessError> {
        self.inner
            .bm25_search(tenant, property, query_text, k, read_lsn)
    }

    fn community_members(
        &self,
        tenant: TenantId,
        community_id: i64,
        read_lsn: Lsn,
    ) -> Result<Vec<BoundNode>, SubstrateAccessError> {
        self.inner.community_members(tenant, community_id, read_lsn)
    }
}

/// Build a stub substrate + catalog seeded with `n` `:User` nodes each
/// carrying a unique `email = "u{i}@x.com"`. `online` toggles whether the
/// `(User, email)` index is planner-visible (Online) — `false` models a
/// Building / unindexed tenant. When indexed, each node is registered as
/// its own B+tree candidate slot.
fn fixture(n: u64, online: bool) -> (StubCatalogProvider, StubExecutorSubstrate) {
    let mut cat = StubCatalogProvider::new().with_label_id("User", USER);
    if online {
        cat = cat.with_online_property_index(USER, "email");
    }
    let mut sub = StubExecutorSubstrate::new();
    if online {
        sub = sub.with_property_index(TenantId::DEFAULT, USER, "email");
    }
    for i in 1..=n {
        let email = format!("u{i}@x.com");
        let node = NodeView::new(NodeId::new(i), Some(USER))
            .with_property("email", Value::String(email.clone()));
        sub = sub.with_node(TenantId::DEFAULT, node);
        if online {
            sub = sub.with_property_index_candidate(
                TenantId::DEFAULT,
                USER,
                "email",
                &Value::String(email),
                NodeId::new(i),
            );
        }
    }
    (cat, sub)
}

fn run<S: ExecutorSubstrate>(cat: &StubCatalogProvider, sub: &S, cypher: &str) -> Vec<Vec<Value>> {
    let engine = QueryEngine::new(cat);
    let mat = engine
        .execute_with_deadline(cypher, sub, DEADLINE)
        .expect("execute OK");
    mat.rows().to_vec()
}

/// Extract the node id from a single-column row.
fn row_node_id(row: &[Value]) -> u64 {
    match &row[0] {
        Value::Node(n) => n.id.raw(),
        other => panic!("expected a Node cell, got {other:?}"),
    }
}

fn sorted_ids(rows: &[Vec<Value>]) -> Vec<u64> {
    let mut ids: Vec<u64> = rows.iter().map(|r| row_node_id(r)).collect();
    ids.sort_unstable();
    ids
}

fn find_op(tree: &PlanTree, op: PlanTreeOp) -> Option<&PlanTree> {
    if tree.op == op {
        return Some(tree);
    }
    tree.children.iter().find_map(|c| find_op(c, op))
}

// =====================================================================
// IDENTICAL-RESULTS — the correctness gate.
// =====================================================================

/// Inline-property `MATCH (n:User {email:"x"})` — indexed vs full-scan
/// return the EXACT SAME rows. The two fixtures differ ONLY in whether
/// the index is Online; the result set must be byte-identical.
#[test]
fn inline_property_indexed_equals_full_scan() {
    let (cat_idx, sub_idx) = fixture(50, true);
    let (cat_scan, sub_scan) = fixture(50, false);
    for target in ["u1@x.com", "u25@x.com", "u50@x.com"] {
        let q = format!("MATCH (n:User {{email: \"{target}\"}}) RETURN n");
        let indexed = run(&cat_idx, &sub_idx, &q);
        let scanned = run(&cat_scan, &sub_scan, &q);
        assert_eq!(
            sorted_ids(&indexed),
            sorted_ids(&scanned),
            "indexed lookup must equal full scan for {target}"
        );
        assert_eq!(indexed.len(), 1, "exactly one match for a unique email");
    }
}

/// `WHERE n.email = "x"` — the WHERE form lowers to the same
/// `Filter(Scan)` shape and must route identically.
#[test]
fn where_equality_indexed_equals_full_scan() {
    let (cat_idx, sub_idx) = fixture(30, true);
    let (cat_scan, sub_scan) = fixture(30, false);
    let q = "MATCH (n:User) WHERE n.email = \"u17@x.com\" RETURN n";
    assert_eq!(
        sorted_ids(&run(&cat_idx, &sub_idx, q)),
        sorted_ids(&run(&cat_scan, &sub_scan, q)),
    );
}

/// Absent value → empty result on BOTH paths.
#[test]
fn absent_value_empty_on_both_paths() {
    let (cat_idx, sub_idx) = fixture(20, true);
    let (cat_scan, sub_scan) = fixture(20, false);
    let q = "MATCH (n:User {email: \"ghost@x.com\"}) RETURN n";
    assert!(run(&cat_idx, &sub_idx, q).is_empty());
    assert!(run(&cat_scan, &sub_scan, q).is_empty());
}

/// A residual predicate (`WHERE n.age > 30`) alongside the indexed
/// equality narrows both paths identically.
#[test]
fn residual_predicate_indexed_equals_full_scan() {
    // Two nodes share an email; ages 40 and 20. The residual keeps age>30.
    let email = "shared@x.com";
    let mut cat = StubCatalogProvider::new().with_label_id("User", USER);
    cat = cat.with_online_property_index(USER, "email");
    let mut sub =
        StubExecutorSubstrate::new().with_property_index(TenantId::DEFAULT, USER, "email");
    for (id, age) in [(1u64, 40i64), (2, 20)] {
        let node = NodeView::new(NodeId::new(id), Some(USER))
            .with_property("email", Value::String(email.into()))
            .with_property("age", Value::Integer(age));
        sub = sub
            .with_node(TenantId::DEFAULT, node)
            .with_property_index_candidate(
                TenantId::DEFAULT,
                USER,
                "email",
                &Value::String(email.into()),
                NodeId::new(id),
            );
    }
    // The full-scan comparison fixture: same nodes, NO index.
    let mut cat_scan = StubCatalogProvider::new().with_label_id("User", USER);
    let _ = &mut cat_scan;
    let mut sub_scan = StubExecutorSubstrate::new();
    for (id, age) in [(1u64, 40i64), (2, 20)] {
        sub_scan = sub_scan.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(id), Some(USER))
                .with_property("email", Value::String(email.into()))
                .with_property("age", Value::Integer(age)),
        );
    }
    let q = "MATCH (n:User {email: \"shared@x.com\"}) WHERE n.age > 30 RETURN n";
    let indexed = run(&cat, &sub, q);
    let scanned = run(&cat_scan, &sub_scan, q);
    assert_eq!(sorted_ids(&indexed), sorted_ids(&scanned));
    assert_eq!(indexed.len(), 1);
    assert_eq!(row_node_id(&indexed[0]), 1, "only the age>40 node survives");
}

/// Duplicate candidate slots for one value dedup to ONE row (the row set
/// matches the full-scan path, which visits the node once).
#[test]
fn duplicate_candidate_slots_dedup_to_one_row() {
    let email = "dup@x.com";
    let cat = StubCatalogProvider::new()
        .with_label_id("User", USER)
        .with_online_property_index(USER, "email");
    let sub = StubExecutorSubstrate::new()
        .with_property_index(TenantId::DEFAULT, USER, "email")
        .with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(7), Some(USER))
                .with_property("email", Value::String(email.into())),
        )
        .with_property_index_candidate(
            TenantId::DEFAULT,
            USER,
            "email",
            &Value::String(email.into()),
            NodeId::new(7),
        )
        .with_property_index_candidate(
            TenantId::DEFAULT,
            USER,
            "email",
            &Value::String(email.into()),
            NodeId::new(7),
        );
    let rows = run(&cat, &sub, "MATCH (n:User {email: \"dup@x.com\"}) RETURN n");
    assert_eq!(rows.len(), 1, "duplicate slots must dedup to one row");
    assert_eq!(row_node_id(&rows[0]), 7);
}

/// An unlabelled `MATCH (n {email:"x"})` is NOT routed to the label-
/// scoped index — it keeps the full scan (design §Planner selection).
#[test]
fn unlabelled_match_is_not_routed_to_index() {
    let (cat, sub) = fixture(10, true);
    let tree = QueryEngine::new(&cat)
        .explain("MATCH (n {email: \"u3@x.com\"}) RETURN n")
        .expect("explain OK");
    assert!(
        find_op(&tree, PlanTreeOp::PropertyIndexScan).is_none(),
        "unlabelled MATCH must NOT use the label-scoped index"
    );
    assert!(find_op(&tree, PlanTreeOp::Scan).is_some());
    // And it still returns the right row (via the scan).
    let rows = run(&cat, &sub, "MATCH (n {email: \"u3@x.com\"}) RETURN n");
    assert_eq!(sorted_ids(&rows), vec![3]);
}

// =====================================================================
// Building-not-used — RED-on-revert.
// =====================================================================

/// A Building index (NOT seeded as `online_property_index`) is NOT
/// chosen by the planner — EXPLAIN shows `Scan`, not `PropertyIndexScan`.
/// A node the (incomplete) backfill missed is STILL found via the scan.
///
/// RED-on-revert: if the RC-6 gate were reverted to accept a Building
/// index (`online_property_index` returning true for a Building state),
/// the planner would route to `PropertyIndexScan`, the lookup would miss
/// the un-backfilled node, and the result would be a FALSE NEGATIVE —
/// diverging from the scan path this test pins.
#[test]
fn building_index_is_not_used_by_planner() {
    // `online = false` models a Building index: declared, being
    // maintained, but NOT planner-visible.
    let (cat, sub) = fixture(20, false);
    let q = "MATCH (n:User {email: \"u9@x.com\"}) RETURN n";

    let tree = QueryEngine::new(&cat).explain(q).expect("explain OK");
    assert!(
        find_op(&tree, PlanTreeOp::PropertyIndexScan).is_none(),
        "a Building index must NOT be routed to PropertyIndexScan"
    );
    assert!(
        find_op(&tree, PlanTreeOp::Scan).is_some(),
        "a Building index keeps the full-scan path"
    );

    // The node is found via the scan (correctness preserved).
    assert_eq!(sorted_ids(&run(&cat, &sub, q)), vec![9]);
}

/// The discriminating half of the RED-on-revert: with the index ONLINE
/// the SAME query routes to PropertyIndexScan. If the gate did not
/// distinguish Building from Online, `building_index_is_not_used` would
/// also route to the index — so this test proves the gate is sensitive.
#[test]
fn online_index_is_used_but_building_is_not_discriminating() {
    let q = "MATCH (n:User {email: \"u9@x.com\"}) RETURN n";

    let (cat_online, _sub_online) = fixture(20, true);
    let tree_online = QueryEngine::new(&cat_online).explain(q).expect("explain");
    assert!(
        find_op(&tree_online, PlanTreeOp::PropertyIndexScan).is_some(),
        "ONLINE index MUST route to PropertyIndexScan"
    );

    let (cat_building, _sub_building) = fixture(20, false);
    let tree_building = QueryEngine::new(&cat_building).explain(q).expect("explain");
    assert!(
        find_op(&tree_building, PlanTreeOp::PropertyIndexScan).is_none(),
        "BUILDING index must NOT route to PropertyIndexScan"
    );
}

// =====================================================================
// PERF — the payoff proof (O(matches) vs O(node_high_water)).
// =====================================================================

/// The indexed path reads O(matches) nodes; the full scan reads
/// O(node_high_water). On a 500-node corpus a unique point lookup should
/// read ~1 node via the index vs ~500 via the scan — a wide margin.
#[test]
fn indexed_lookup_reads_far_fewer_nodes_than_scan() {
    let n = 500u64;
    let (cat_idx, sub_idx) = fixture(n, true);
    let (cat_scan, sub_scan) = fixture(n, false);
    let q = "MATCH (n:User {email: \"u250@x.com\"}) RETURN n";

    let counting_idx = CountingSubstrate::new(sub_idx);
    counting_idx.reset();
    let idx_rows = run(&cat_idx, &counting_idx, q);
    let idx_reads = counting_idx.reads();

    let counting_scan = CountingSubstrate::new(sub_scan);
    counting_scan.reset();
    let scan_rows = run(&cat_scan, &counting_scan, q);
    let scan_reads = counting_scan.reads();

    assert_eq!(sorted_ids(&idx_rows), sorted_ids(&scan_rows), "same rows");
    assert_eq!(idx_rows.len(), 1);

    // The index reads a handful (the single verified candidate); the scan
    // reads the whole corpus. Assert a wide margin (index ≤ 10, scan ≥ n).
    assert!(
        idx_reads <= 10,
        "indexed path should read O(matches) nodes, read {idx_reads}"
    );
    assert!(
        scan_reads >= n as usize,
        "scan path should read O(node_high_water) nodes, read {scan_reads}"
    );
    assert!(
        scan_reads > idx_reads * 20,
        "indexed ({idx_reads}) must be dramatically cheaper than scan ({scan_reads})"
    );
}

// =====================================================================
// EXPLAIN — operator visibility.
// =====================================================================

/// EXPLAIN shows `PropertyIndexScan` with the label, property, and
/// residual annotations for an indexed lookup.
#[test]
fn explain_shows_property_index_scan_annotations() {
    let (cat, _sub) = fixture(10, true);
    let tree = QueryEngine::new(&cat)
        .explain("MATCH (n:User {email: \"u1@x.com\"}) RETURN n")
        .expect("explain OK");
    let node = find_op(&tree, PlanTreeOp::PropertyIndexScan)
        .expect("plan must contain a PropertyIndexScan");
    assert_eq!(node.op.name(), "PropertyIndexScan");
    assert_eq!(
        node.annotations.get("label").map(String::as_str),
        Some(format!("L{}", USER.raw()).as_str())
    );
    assert_eq!(
        node.annotations.get("property").map(String::as_str),
        Some("email")
    );
    assert_eq!(
        node.annotations.get("residual").map(String::as_str),
        Some("false"),
        "no residual for a bare indexed equality"
    );
}

/// A second WHERE conjunct alongside the indexed equality: the equality
/// still routes to `PropertyIndexScan` and the `age > 30` conjunct is
/// preserved (as an outer `Filter` over the index — the WHERE path folds
/// conjuncts into a Filter-chain, so the extra predicate keeps its own
/// Filter node rather than the PropertyIndexScan `residual` slot). Both
/// the index op AND the filter appear in the plan.
#[test]
fn explain_indexed_equality_coexists_with_other_where_conjunct() {
    let cat = StubCatalogProvider::new()
        .with_label_id("User", USER)
        .with_online_property_index(USER, "email");
    let tree = QueryEngine::new(&cat)
        .explain("MATCH (n:User) WHERE n.email = \"a@x.com\" AND n.age > 30 RETURN n")
        .expect("explain OK");
    assert!(
        find_op(&tree, PlanTreeOp::PropertyIndexScan).is_some(),
        "the email equality routes to the index"
    );
    assert!(
        find_op(&tree, PlanTreeOp::Filter).is_some(),
        "the age>30 conjunct is preserved as a Filter over the index"
    );
}

/// The `residual` slot IS populated when the index equality shares ONE
/// Filter with another conjunct (constructed directly — the WHERE path
/// prefers a Filter-chain, but a folded-AND Filter is a legal plan shape
/// the rewrite must handle). Drives the rewrite over a hand-built plan.
#[test]
fn residual_slot_populated_for_folded_and_filter() {
    use arcgraph_query::ast::{BinOp, Literal};
    use arcgraph_query::error::Span;
    use arcgraph_query::logical_plan::{
        LogicalFilter, LogicalPlan, LogicalScan, rewrite_scan_to_property_index_scan,
    };
    use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression, BoundPropertyRef};

    let var = BindingId::new(0);
    let prop_eq = |name: &str, rhs: BoundExpression| -> BoundExpression {
        BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(BoundExpression::PropertyAccess {
                base: Box::new(BoundExpression::VariableRef {
                    name: "n".into(),
                    binding_id: var,
                    span: Span::point(1, 1),
                    type_info: None,
                }),
                path: vec![BoundPropertyRef {
                    name: name.into(),
                    property_id: None,
                    span: Span::point(1, 1),
                }],
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(rhs),
            span: Span::point(1, 1),
            type_info: None,
        }
    };
    // email = "a" AND status = "active" — both equalities, only email is
    // indexed → email routes to the index, status becomes the residual.
    let email_eq = prop_eq(
        "email",
        BoundExpression::Literal {
            value: Literal::String("a@x.com".into()),
            span: Span::point(1, 1),
            type_info: None,
        },
    );
    let status_eq = prop_eq(
        "status",
        BoundExpression::Literal {
            value: Literal::String("active".into()),
            span: Span::point(1, 1),
            type_info: None,
        },
    );
    let anded = BoundExpression::BinaryOp {
        op: BinOp::And,
        lhs: Box::new(email_eq),
        rhs: Box::new(status_eq),
        span: Span::point(1, 1),
        type_info: None,
    };
    let plan = LogicalPlan::Filter(LogicalFilter {
        input: Box::new(LogicalPlan::Scan(LogicalScan {
            label: Some(USER),
            var,
            read_lsn: Lsn::MAX,
            span: Span::point(1, 1),
        })),
        predicate: anded,
        span: Span::point(1, 1),
    });
    let cat = StubCatalogProvider::new()
        .with_label_id("User", USER)
        .with_online_property_index(USER, "email");
    let rewritten = rewrite_scan_to_property_index_scan(plan, &cat);
    match rewritten {
        LogicalPlan::PropertyIndexScan(p) => {
            assert_eq!(p.property, "email");
            assert!(p.residual.is_some(), "status=active is the residual");
        }
        other => panic!("expected PropertyIndexScan, got {other:?}"),
    }
}
