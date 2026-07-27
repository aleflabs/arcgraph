//! Q4 adversarial sweep — SET / REMOVE / DELETE write-op SEMANTIC
//! invariants, exercised END-TO-END through the executor pipeline
//! (`parse → bind → typecheck → cross-substrate → lower → Pipeline →
//! Scan/Create + Set/Remove/Delete op → StubExecutorSubstrate`) with
//! STRONG read-back oracles (exact `Value` equality / exact edge
//! presence — NOT "doesn't panic").
//!
//! # Why this file (the gap it closes)
//!
//! The sibling `set_remove_proptest.rs` asserts only parse / Display
//! round-trip / lowering shape (`has_set` / `has_remove` is present in
//! the plan) — it NEVER drains the pipeline, so it cannot catch a wrong
//! value being persisted. `delete_proptest.rs` drains for the
//! tombstone-a-lone-node case + the detach-flag-threads-through-lower
//! case, but does NOT cover a node with an INCIDENT EDGE (the
//! dangling-edge corruption surface). This file pins the *execution
//! semantics* those leave uncovered:
//!
//! - **SET last-writer-wins / idempotent / preserves-siblings** — read
//!   back the persisted `Value` after the executor dispatches
//!   `set_node`, asserting EXACT equality (ADR-150 §D-7
//!   `PropertyAssign` → per-key insert).
//! - **REMOVE removes-exactly / idempotent / preserves-q** — read back
//!   the post-REMOVE bag, asserting the removed key is ABSENT and the
//!   sibling key is UNTOUCHED (ADR-150 §D-7 `Property` → per-key clear).
//! - **DELETE leaves no dangling edge (HIGHEST VALUE)** — per
//!   openCypher v9 §6 + ADR-149 §D-7, a plain `DELETE` of a node that
//!   still has incident relationships MUST be a runtime error (the
//!   executor surfaces `ExecutionError::Substrate(Io("relationships
//!   attached"))`); it MUST NOT silently orphan the edge. We assert (a)
//!   the error fires, (b) the failed op leaves NO partial side-effect
//!   (node + edge both intact), and (c) a subsequent `DETACH DELETE`
//!   removes BOTH the node AND its incident edges (no dangling edge
//!   pointing at a tombstoned node).
//!
//! # Oracle strength
//!
//! Every property asserts EXACT equality (`prop_assert_eq!` on a
//! concrete `Value` / a node count / an edge count), never a
//! panic-free or `>=` weakening — per the engineering doctrine "a green
//! test that can't fail on its bug is worse than no test."
//!
//! # ADR provenance
//! - **ADR-149** (W26-θ Phase 3) — DELETE / DETACH DELETE semantics +
//!   §D-7 "relationships attached" runtime-error contract.
//! - **ADR-150** (W26-θ Phase 4) — SET / REMOVE mutation semantics
//!   (§D-7 per-key insert / clear).
//! - **ADR-152 §D-3** (W27-α) — SET/REMOVE property-persistence wire
//!   (scan_nodes merges the post-mutation bag).
//! - openCypher v9 §6 — plain DELETE of a connected node is an error.
//! - Refs #521 (Epic — QUALITY track).

use std::collections::BTreeMap;

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use proptest::prelude::*;

use arcgraph_query::executor::error::ExecutionError;
use arcgraph_query::executor::substrate::{
    ExecutorSubstrate, StubExecutorSubstrate, SubstrateAccessError,
};
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{Direction, LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

/// The id the Stub's catalog interns the (single) test label name to.
/// Pre-baked fixture nodes carry `Some(LabelId::new(STUB_LABEL_ID))` so
/// the `MATCH (n:Label)` scan (lowered to `Scan{Some(STUB_LABEL_ID)}`
/// once the label is interned) finds them — mirrors the
/// `merge_proptest.rs` label-interning convention (ADR-152-amendment-01
/// §D-1).
const STUB_LABEL_ID: u32 = 1024;

// --------------------------------------------------------------------
// Harness — parse → bind → typecheck → cross-substrate → lower, with
// the test label interned so `MATCH (n:Label)` resolves to the
// pre-baked fixture node.
// --------------------------------------------------------------------

fn lower(query: &str, label: &str) -> Result<LogicalPlan, String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let cat = StubCatalogProvider::new().with_label_id(label, LabelId::new(STUB_LABEL_ID));
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("typecheck: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &cat)
        .map_err(|e| format!("cross-substrate: {e:?}"))?;
    LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))
}

/// Drain a pipeline to EOS, propagating any execution error (the
/// dangling-edge invariant relies on observing the error verbatim).
fn drain(
    plan: &LogicalPlan,
    s: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Result<(), ExecutionError> {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(ctx, s)?;
        if b.is_empty() {
            break;
        }
    }
    Ok(())
}

/// Read back the single MATCH-able node's persisted property bag via
/// `scan_nodes` (the production read path — `scan_nodes` merges the
/// post-SET/REMOVE sidecar per ADR-152 §D-3). The fixture always
/// pre-bakes exactly one labelled node, so the scan yields exactly one.
fn read_back_bag(s: &StubExecutorSubstrate, tenant: TenantId) -> BTreeMap<String, Value> {
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan_nodes OK");
    assert_eq!(nodes.len(), 1, "fixture has exactly one node: {nodes:?}");
    nodes[0].node.properties.clone()
}

/// Build a fresh single-node fixture (known id `1`, interned label) +
/// a fresh context. The known id lets a later `node_properties(id)`
/// read-back assert against an EXACT id.
fn one_node_fixture() -> (StubExecutorSubstrate, ExecutionContext, NodeId, TenantId) {
    let tenant = TenantId::DEFAULT;
    let nid = NodeId::new(1);
    let s = StubExecutorSubstrate::new().with_node(
        tenant,
        NodeView::new(nid, Some(LabelId::new(STUB_LABEL_ID))),
    );
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    (s, ctx, nid, tenant)
}

/// Build an `n -[r]-> m` fixture with known ids (n=1, m=2, r=10).
///
/// ONLY the source endpoint `n` carries the interned label, so
/// `MATCH (n:Label)` binds EXACTLY `n` (one row). The sink endpoint
/// `m` is deliberately UNLABELLED: it exists only as the edge's other
/// endpoint and is never a `MATCH (n:Label)` target. (A `MATCH`-able
/// `m` would bind a SECOND row → `DELETE n` would iterate per-row and
/// delete `m` too — exactly the over-deletion these tests must NOT
/// trip on; `m` is the survivor whose presence + dangling-edge state
/// the oracle inspects.) `scan_nodes(None)` still counts `m`
/// (no-label scan returns all nodes) and `expand` is adjacency-driven
/// (label-independent), so `m`'s read-back paths are unaffected.
fn edge_fixture() -> (
    StubExecutorSubstrate,
    ExecutionContext,
    NodeId,
    NodeId,
    RelId,
    TenantId,
) {
    let tenant = TenantId::DEFAULT;
    let n = NodeId::new(1);
    let m = NodeId::new(2);
    let r = RelId::new(10);
    let s = StubExecutorSubstrate::new()
        .with_node(tenant, NodeView::new(n, Some(LabelId::new(STUB_LABEL_ID))))
        .with_node(tenant, NodeView::new(m, None))
        .with_edge(tenant, RelView::new(r, n, m, Some(TypeId::new(1))));
    let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
    (s, ctx, n, m, r, tenant)
}

// --------------------------------------------------------------------
// Strategies — identifier-safe property names + non-keyword labels +
// proptest-generated integer values (distinct A ≠ B for last-writer).
// --------------------------------------------------------------------

fn label_strategy() -> impl Strategy<Value = String> {
    // The canonical reserved-word set lives in grammar.pest; keep this
    // self-maintaining by generating labels that cannot equal a bare
    // keyword.
    "[A-Z][a-zA-Z0-9_]{0,7}".prop_map(|s| format!("L_{s}"))
}

fn prop_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-zA-Z0-9_]{0,7}".prop_filter("non-keyword", |s| !is_reserved(s))
}

/// Two DISTINCT lowercase property names (for the preserves-siblings /
/// removes-exactly invariants where `p != q` matters).
fn two_distinct_props() -> impl Strategy<Value = (String, String)> {
    (prop_strategy(), prop_strategy()).prop_filter("p != q", |(a, b)| a != b)
}

fn is_reserved(s: &str) -> bool {
    matches!(
        s,
        "MATCH"
            | "WHERE"
            | "RETURN"
            | "WITH"
            | "UNWIND"
            | "AS"
            | "DISTINCT"
            | "ORDER"
            | "BY"
            | "ASC"
            | "DESC"
            | "LIMIT"
            | "SKIP"
            | "AND"
            | "OR"
            | "NOT"
            | "IN"
            | "IS"
            | "NULL"
            | "TRUE"
            | "FALSE"
            | "FOR"
            | "ALL"
            | "NEAR"
            | "RANK"
            | "DEFINE"
            | "OPTIONAL"
            | "EXPLAIN"
            | "PROFILE"
            | "CREATE"
            | "DELETE"
            | "DETACH"
            | "SET"
            | "REMOVE"
            | "MERGE"
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    // ================================================================
    // SET — last-writer-wins
    // ================================================================

    /// `SET n.p = A` then `SET n.p = B` (A ≠ B) → read-back yields B
    /// EXACTLY. Driven via `MATCH (n:L) SET n.p = A SET n.p = B` on a
    /// pre-baked node (one statement, two SET clauses, source order
    /// A-then-B). Oracle: exact `Value::Integer(B)` (ADR-150 §D-7 per-key
    /// insert; the second assign overwrites the first).
    #[test]
    fn set_last_writer_wins(
        label in label_strategy(),
        prop in prop_strategy(),
        a in 0i64..500_000,
        delta in 1i64..500_000,
    ) {
        let b = a + delta; // guarantees b != a
        let (s, ctx, nid, tenant) = one_node_fixture();
        let q = format!("MATCH (n:{label}) SET n.{prop} = {a} SET n.{prop} = {b}");
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("drain OK");

        // Strong oracle A — direct sidecar read-back by exact id.
        let bag = s.node_properties(tenant, nid).expect("node_properties present");
        prop_assert_eq!(
            bag.get(&prop), Some(&Value::Integer(b)),
            "last-writer: SET {}={} then ={} must persist {}, got {:?}",
            prop, a, b, b, bag.get(&prop)
        );
        // Strong oracle B — production read path (scan_nodes merge).
        let scanned = read_back_bag(&s, tenant);
        prop_assert_eq!(scanned.get(&prop), Some(&Value::Integer(b)),
            "scan_nodes read path must also observe {}", b);
    }

    /// `SET n.p = A` then `SET n.p = A` (same value twice) → idempotent:
    /// exactly one value, == A. Oracle: exact `Value::Integer(A)`.
    #[test]
    fn set_idempotent_same_value(
        label in label_strategy(),
        prop in prop_strategy(),
        a in 0i64..1_000_000,
    ) {
        let (s, ctx, nid, tenant) = one_node_fixture();
        let q = format!("MATCH (n:{label}) SET n.{prop} = {a} SET n.{prop} = {a}");
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("drain OK");
        let bag = s.node_properties(tenant, nid).expect("node_properties present");
        prop_assert_eq!(bag.get(&prop), Some(&Value::Integer(a)),
            "idempotent SET must persist {} exactly once", a);
        prop_assert_eq!(bag.len(), 1, "exactly one property after idempotent SET: {:?}", bag);
    }

    /// `SET n.a = X` must NOT alter `n.b`. We seed `n.b` via a first
    /// SET, then SET a DIFFERENT property, and assert `n.b` is unchanged
    /// AND `n.a` took the new value. Oracle: both exact.
    #[test]
    fn set_preserves_siblings(
        label in label_strategy(),
        (pa, pb) in two_distinct_props(),
        xa in 0i64..1_000_000,
        xb in 0i64..1_000_000,
    ) {
        let (s, ctx, nid, tenant) = one_node_fixture();
        // SET the sibling pb first, then SET pa — pb must survive.
        let q = format!("MATCH (n:{label}) SET n.{pb} = {xb} SET n.{pa} = {xa}");
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("drain OK");
        let bag = s.node_properties(tenant, nid).expect("node_properties present");
        prop_assert_eq!(bag.get(&pa), Some(&Value::Integer(xa)), "target prop set");
        prop_assert_eq!(bag.get(&pb), Some(&Value::Integer(xb)),
            "SET n.{} must NOT alter sibling n.{}", pa, pb);
        prop_assert_eq!(bag.len(), 2, "exactly the two props present: {:?}", bag);
    }

    // ================================================================
    // REMOVE — removes exactly
    // ================================================================

    /// `SET n.p = X; REMOVE n.p` → `n.p` ABSENT on read-back. Oracle:
    /// `bag.get(p) == None` (ADR-150 §D-7 per-key clear).
    #[test]
    fn remove_removes_exactly(
        label in label_strategy(),
        prop in prop_strategy(),
        x in 0i64..1_000_000,
    ) {
        let (s, ctx, nid, tenant) = one_node_fixture();
        let q = format!("MATCH (n:{label}) SET n.{prop} = {x} REMOVE n.{prop}");
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("drain OK");
        let bag = s.node_properties(tenant, nid).expect("node_properties present");
        prop_assert_eq!(bag.get(&prop), None,
            "REMOVE n.{} must clear the key, bag={:?}", prop, bag);
        // Production read path agrees.
        let scanned = read_back_bag(&s, tenant);
        prop_assert_eq!(scanned.get(&prop), None, "scan read path: key absent");
    }

    /// REMOVE-then-REMOVE is idempotent: `SET n.p=X; REMOVE n.p;
    /// REMOVE n.p` → still absent, no error. Oracle: `None` + no panic.
    #[test]
    fn remove_idempotent(
        label in label_strategy(),
        prop in prop_strategy(),
        x in 0i64..1_000_000,
    ) {
        let (s, ctx, nid, tenant) = one_node_fixture();
        let q = format!(
            "MATCH (n:{label}) SET n.{prop} = {x} REMOVE n.{prop} REMOVE n.{prop}"
        );
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("drain OK");
        let bag = s.node_properties(tenant, nid).expect("node_properties present");
        prop_assert_eq!(bag.get(&prop), None, "double-REMOVE idempotent: still absent");
    }

    /// `REMOVE n.p` does NOT remove `n.q`. Seed both p and q, REMOVE p,
    /// assert q survives EXACTLY and p is gone. Oracle: both exact.
    #[test]
    fn remove_preserves_other(
        label in label_strategy(),
        (pp, pq) in two_distinct_props(),
        xp in 0i64..1_000_000,
        xq in 0i64..1_000_000,
    ) {
        let (s, ctx, nid, tenant) = one_node_fixture();
        let q = format!(
            "MATCH (n:{label}) SET n.{pp} = {xp} SET n.{pq} = {xq} REMOVE n.{pp}"
        );
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("drain OK");
        let bag = s.node_properties(tenant, nid).expect("node_properties present");
        prop_assert_eq!(bag.get(&pp), None, "REMOVE n.{} clears it", pp);
        prop_assert_eq!(bag.get(&pq), Some(&Value::Integer(xq)),
            "REMOVE n.{} must NOT remove sibling n.{}", pp, pq);
        prop_assert_eq!(bag.len(), 1, "exactly q remains: {:?}", bag);
    }

    // ================================================================
    // DELETE — leaves no dangling edge (HIGHEST VALUE)
    // ================================================================

    /// Plain `DELETE` of a node that still has an incident edge MUST be
    /// a runtime error per openCypher v9 §6 + ADR-149 §D-7 — it MUST
    /// NOT silently orphan the edge.
    ///
    /// Oracle (exact, three-part):
    /// 1. `drain` returns `Err(ExecutionError::Substrate(Io(msg)))`
    ///    with `msg` reporting the attached-relationships condition (see
    ///    the message-contract note below).
    /// 2. No partial side-effect: the node `n` is STILL visible
    ///    (`scan_nodes` count unchanged at 2).
    /// 3. No partial side-effect: the incident edge is STILL present
    ///    (`expand` from `m` still yields the edge).
    ///
    /// If `drain` returns `Ok(())` here (edge silently orphaned), THAT
    /// IS A REAL EXECUTOR BUG and this assertion fails loudly.
    ///
    /// # Message-contract assertion — ADR-149 §D-7 (BUG #710 FIXED)
    /// ADR-149 §D-7 + the `delete_node` trait doc-comment
    /// (`substrate.rs:253-255`) document the contract phrase
    /// `"relationships attached"`. The substrate emits
    /// `"delete_node: node has relationships attached; use DETACH
    /// DELETE"` (`substrate.rs:1350`), which CONTAINS that canonical
    /// substring — so this test asserts the documented contract phrase
    /// directly, NOT a weaker fallback. BUG #710 (the earlier word-order
    /// drift, where the live message said `"attached relationships"` and
    /// the documented `"relationships attached"` substring did NOT appear)
    /// is now reconciled: the two emission sites
    /// (`arcgraph-query` Stub + `arcgraph-mcp` production `CrudExecutor`),
    /// the ADR's three internal references (§D-7:361/:377/:553), and this
    /// oracle all quote the same canonical phrase. The
    /// `prop_assert!(msg.contains("relationships attached"))` below is a
    /// genuine contract assertion: it FAILS if the emitted message ever
    /// regresses to the flipped word order.
    #[test]
    fn plain_delete_with_incident_edge_errors_and_no_side_effect(
        label in label_strategy(),
    ) {
        let (s, ctx, _n, m, _r, tenant) = edge_fixture();
        let q = format!("MATCH (n:{label}) DELETE n");
        let plan = lower(&q, &label).expect("lower OK");
        let res = drain(&plan, &s, &ctx);

        // Part 1 — the error MUST fire (not a silent orphan). We assert
        // the documented ADR-149 §D-7 contract substring
        // `"relationships attached"` (BUG #710 reconciled — the emitted
        // message now CONTAINS this canonical phrase). This rejects the
        // wrong-error case and the silent-orphan (`Ok`) case below, and
        // FAILS if the message regresses to the flipped word order.
        match res {
            Err(ExecutionError::Substrate(SubstrateAccessError::Io(msg))) => {
                prop_assert!(
                    // ADR-149 §D-7 canonical contract phrase. Genuine
                    // assertion: fails on word-order regression to
                    // "attached relationships" (the old BUG #710 drift).
                    msg.contains("relationships attached"),
                    "expected the ADR-149 §D-7 \"relationships attached\" runtime \
                     error, got Io({})",
                    msg
                );
            }
            Err(other) => {
                return Err(TestCaseError::fail(format!(
                    "plain DELETE of a connected node errored, but with the WRONG \
                     error (expected Io 'relationships attached'): {other:?}"
                )));
            }
            Ok(()) => {
                return Err(TestCaseError::fail(
                    "BUG: plain DELETE of a node with an incident edge returned Ok \
                     — the edge was SILENTLY ORPHANED (dangling edge / graph \
                     corruption). openCypher v9 §6 + ADR-149 §D-7 require a runtime \
                     error."
                        .to_string(),
                ));
            }
        }

        // Part 2 — failed plain DELETE left the node intact (atomicity).
        let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
        prop_assert_eq!(nodes.len(), 2,
            "failed plain DELETE must leave BOTH nodes; got {:?}", nodes);

        // Part 3 — failed plain DELETE left the incident edge intact.
        let edges = s
            .expand(tenant, m, None, Direction::Undirected, Lsn::MAX)
            .expect("expand OK");
        prop_assert_eq!(edges.len(), 1,
            "failed plain DELETE must leave the incident edge; got {:?}", edges);
    }

    /// After `DETACH DELETE n`, the incident edges are GONE and the node
    /// is gone — no dangling edge pointing at a tombstoned node.
    ///
    /// Oracle (exact):
    /// 1. `drain` succeeds (DETACH is the legal way to delete a
    ///    connected node).
    /// 2. `expand` from the SURVIVING endpoint `m` yields 0 edges (the
    ///    incident rel was tombstoned BEFORE the node).
    /// 3. The deleted node `n` is no longer visible.
    #[test]
    fn detach_delete_removes_incident_edges(
        label in label_strategy(),
    ) {
        let (s, ctx, n, m, _r, tenant) = edge_fixture();
        let q = format!("MATCH (n:{label}) DETACH DELETE n");
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("DETACH DELETE drains cleanly");

        // No dangling edge: expand from the surviving endpoint m is empty.
        let edges_from_m = s
            .expand(tenant, m, None, Direction::Undirected, Lsn::MAX)
            .expect("expand OK");
        prop_assert_eq!(edges_from_m.len(), 0,
            "DETACH DELETE must remove the incident edge (no dangling edge to \
             tombstoned node); got {:?}", edges_from_m);

        // expand from the deleted endpoint n is also empty.
        let edges_from_n = s
            .expand(tenant, n, None, Direction::Undirected, Lsn::MAX)
            .expect("expand OK");
        prop_assert_eq!(edges_from_n.len(), 0,
            "no edges remain incident to the deleted node n; got {:?}", edges_from_n);

        // The deleted node is no longer visible; only m survives.
        let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
        prop_assert_eq!(nodes.len(), 1, "only the surviving endpoint m remains: {:?}", nodes);
        prop_assert_eq!(nodes[0].node.id, m, "the survivor is m, not the deleted n");
    }

    /// A node with NO incident edge can be plain-`DELETE`d cleanly
    /// (control: the error in `plain_delete_with_incident_edge_errors`
    /// is SPECIFIC to the attached-edge case, not DELETE-in-general).
    /// Without this control the error test could pass for the wrong
    /// reason (DELETE always erroring). Oracle: Ok + node gone.
    #[test]
    fn plain_delete_lone_node_succeeds(label in label_strategy()) {
        let (s, ctx, _nid, tenant) = one_node_fixture();
        let q = format!("MATCH (n:{label}) DELETE n");
        let plan = lower(&q, &label).expect("lower OK");
        drain(&plan, &s, &ctx).expect("plain DELETE of a lone node drains cleanly");
        let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan OK");
        prop_assert_eq!(nodes.len(), 0, "lone node tombstoned: {:?}", nodes);
    }
}
