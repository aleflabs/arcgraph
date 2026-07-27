//! #709 R1-narrowing — **terminal-vs-stacked row-cardinality pin**
//! (load-bearing oracle), end-to-end through `parse → bind → typecheck
//! → cross-substrate → lower → Pipeline::build → execute`.
//!
//! # Why this file (the gap it closes)
//!
//! PR #716's first cut made `SetOp`/`RemoveOp` pass rows through
//! UNCONDITIONALLY. That fixed the #709 chained-clause composition bug
//! (`SET a=0 SET a=1` → last-writer-wins) BUT also made a RETURN-less
//! TERMINAL write (`MATCH (n) SET n.p = x`, no RETURN) emit a row to the
//! driver — breaking the openCypher TCK write-op RowSet conformance gate
//! (the TCK row-set harness), which asserts a terminal
//! SET/REMOVE produces **0 rows** (openCypher v9 / ADR-149/150 §D /
//! ADR-182 v1.0-α contract).
//!
//! The R1-narrowed fix makes emission **terminal-vs-stacked**: a stacked
//! write-op (the inner clause of `SET … SET …` / `SET … REMOVE …`) passes
//! its rows through so the outer op composes; a terminal write-op (the
//! pipeline root / no write-op consumer above it) drains the upstream and
//! emits 0 rows.
//!
//! # The load-bearing pin (per R1 verdict §"Required narrowing")
//!
//! The two requirements can **silently trade off** against each other: a
//! mechanism that makes the terminal case emit 0 rows by REVERTING the
//! pass-through would re-break composition; a mechanism that fixes
//! composition by passing through unconditionally re-breaks the terminal
//! cardinality. This module asserts BOTH, in the SAME test bodies, so
//! neither can regress silently:
//!
//! 1. **Terminal → 0 rows** — `MATCH (n:L) SET n.p = x` and
//!    `MATCH (n:L) REMOVE n.p` each return EXACTLY 0 result rows from the
//!    driver (`execute_with_context`), via the REAL lowering +
//!    `Pipeline::build` (so the build-time terminal discriminator
//!    `mark_writeop_input_stacked` is exercised, not bypassed).
//! 2. **Stacked → composes** — `SET a=0 SET a=1` persists `a==1`
//!    (last-writer-wins: the inner SET MUST have passed its row to the
//!    outer SET, else only `a=0` would persist) AND
//!    `SET a=1 REMOVE a` clears `a` (the inner SET passed its row to the
//!    outer REMOVE). The composition is proven by PERSISTED STATE
//!    read-back, independent of row output.
//! 3. **No silent trade-off** — the chained queries ALSO emit 0 rows
//!    (their OUTERMOST write-op is terminal) WHILE composing. A single
//!    test body asserts the chained query's row-count == 0 AND its
//!    persisted value, so you cannot satisfy the cardinality by breaking
//!    composition or vice-versa.
//!
//! Oracle strength: EXACT row-count (`== 0`) + EXACT persisted `Value`
//! equality — never `>=` / panic-free weakening (engineering doctrine:
//! "a green test that can't fail on its bug is worse than no test").
//!
//! # ADR provenance
//! - **ADR-150** (W26-θ Phase 4) — SET / REMOVE mutation semantics.
//! - **ADR-149** §D + **ADR-182** §"Forward-deferred" row 3 — RETURN-less
//!   terminal SET/REMOVE produces 0 rows at v1.0-α.
//! - **#709** — chained SET/REMOVE last-writer-wins composition.
//! - openCypher v9 §SET / §REMOVE.

use std::collections::BTreeMap;

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, Value};
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, execute_with_context};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

/// The id the Stub's catalog interns the (single) test label name to —
/// matches the `writeop_semantics_proptest.rs` convention so the
/// `MATCH (n:Label)` scan (lowered to `Scan{Some(STUB_LABEL_ID)}`) finds
/// the pre-baked fixture node.
const STUB_LABEL_ID: u32 = 1024;
const LABEL: &str = "Person";

/// Harness — `parse → bind → typecheck → cross-substrate → lower`, with
/// the test label interned so `MATCH (n:Label)` resolves to the
/// pre-baked fixture node. Mirrors `writeop_semantics_proptest.rs::lower`.
fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let cat = StubCatalogProvider::new().with_label_id(LABEL, LabelId::new(STUB_LABEL_ID));
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("typecheck OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

/// Fresh single-node fixture (known id `1`, interned label) + context.
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

/// Execute `query` through the REAL driver (`Pipeline::build` inside
/// `execute_with_context`) and return the materialized result rows.
/// This is the exact path the production `StorageRawQueryExecutor`
/// drives, and it exercises the build-time terminal discriminator.
fn run_rows(query: &str, s: &StubExecutorSubstrate, ctx: &ExecutionContext) -> Vec<Vec<Value>> {
    let plan = lower(query);
    execute_with_context(&plan, s, ctx).expect("execute OK")
}

/// Persisted property bag of the single fixture node via the production
/// read path (`scan_nodes` merges the post-mutation sidecar).
fn read_back_bag(s: &StubExecutorSubstrate, tenant: TenantId) -> BTreeMap<String, Value> {
    let nodes = s.scan_nodes(tenant, None, Lsn::MAX).expect("scan_nodes OK");
    assert_eq!(nodes.len(), 1, "fixture has exactly one node: {nodes:?}");
    nodes[0].node.properties.clone()
}

// ====================================================================
// DIMENSION 1 — TERMINAL write-op emits 0 rows.
// ====================================================================

/// `MATCH (n:L) SET n.p = "alice"` is a RETURN-less TERMINAL SET → it
/// MUST emit 0 result rows (openCypher / ADR-149/150 §D / ADR-182). The
/// mutation still happens (asserted) — the 0-row contract is about
/// driver-visible cardinality, not the write.
///
/// Pre-R1-narrowing (unconditional pass-through), this query returned 1
/// row. This assertion catches that regression at the query layer.
#[test]
fn terminal_set_emits_zero_rows_and_still_writes() {
    let (s, ctx, nid, tenant) = one_node_fixture();
    let rows = run_rows(
        &format!("MATCH (n:{LABEL}) SET n.name = \"alice\""),
        &s,
        &ctx,
    );
    assert_eq!(
        rows.len(),
        0,
        "terminal SET (no RETURN) must emit 0 rows, got {} row(s): {:?}",
        rows.len(),
        rows
    );
    // The write happened despite 0 rows emitted.
    let bag = s
        .node_properties(tenant, nid)
        .expect("node_properties present");
    assert_eq!(
        bag.get("name"),
        Some(&Value::String("alice".into())),
        "terminal SET applied the mutation, got {:?}",
        bag.get("name")
    );
}

/// `MATCH (n:L) REMOVE n.p` is a RETURN-less TERMINAL REMOVE → 0 rows.
#[test]
fn terminal_remove_emits_zero_rows_and_still_writes() {
    let (s, ctx, nid, tenant) = one_node_fixture();
    // Seed the property with a terminal SET first, then REMOVE it.
    let seed = run_rows(
        &format!("MATCH (n:{LABEL}) SET n.name = \"alice\""),
        &s,
        &ctx,
    );
    assert_eq!(seed.len(), 0, "terminal SET seed emits 0 rows");
    let rows = run_rows(&format!("MATCH (n:{LABEL}) REMOVE n.name"), &s, &ctx);
    assert_eq!(
        rows.len(),
        0,
        "terminal REMOVE (no RETURN) must emit 0 rows, got {} row(s): {:?}",
        rows.len(),
        rows
    );
    // The removal happened.
    let bag = s
        .node_properties(tenant, nid)
        .expect("node_properties present");
    assert_eq!(
        bag.get("name"),
        None,
        "terminal REMOVE cleared the property, got {:?}",
        bag.get("name")
    );
}

// ====================================================================
// DIMENSION 2 + 3 — STACKED write-ops compose (rows flow inner→outer)
// AND the chained query is STILL terminal (0 rows). Asserted together
// so the two requirements cannot silently trade off.
// ====================================================================

/// **The load-bearing pin (SET→SET).** `MATCH (n:L) SET n.a=0 SET n.a=1`
/// lowers to `Set(a=1, Set(a=0, Scan))`. This test asserts, in ONE body:
///
/// - **Composition (DIMENSION 2):** the persisted `a == 1`. This is only
///   possible if the inner SET (a=0) PASSED ITS ROW THROUGH to the outer
///   SET (a=1) — i.e., the inner op is stacked. Pre-#709-fix the inner's
///   empty batch was read as EOS → the outer never ran → `a == 0`. The
///   `== 1` oracle is the composition proof.
/// - **Terminal cardinality (DIMENSION 3):** the chained query emits 0
///   rows — its OUTERMOST SET is terminal. If a future change makes
///   composition work by passing through UNCONDITIONALLY (re-breaking the
///   terminal contract), THIS row-count assertion fails. If a change
///   makes the terminal case 0-rows by dropping the pass-through
///   (re-breaking composition), the `a == 1` assertion fails. Neither can
///   regress without tripping this test.
#[test]
fn stacked_set_set_composes_and_outer_is_terminal() {
    let (s, ctx, nid, tenant) = one_node_fixture();
    let rows = run_rows(
        &format!("MATCH (n:{LABEL}) SET n.a = 0 SET n.a = 1"),
        &s,
        &ctx,
    );

    // DIMENSION 3 — outer SET is terminal → 0 rows.
    assert_eq!(
        rows.len(),
        0,
        "chained SET…SET: the outer SET is terminal → 0 rows, got {} row(s): {:?}",
        rows.len(),
        rows
    );

    // DIMENSION 2 — composition: last-writer-wins requires the inner SET
    // to have passed its row to the outer SET (stacked). a == 1 proves it.
    let bag = s
        .node_properties(tenant, nid)
        .expect("node_properties present");
    assert_eq!(
        bag.get("a"),
        Some(&Value::Integer(1)),
        "stacked SET a=0 then a=1 must persist 1 (last-writer-wins via \
         inner→outer row flow); a={:?} would mean the outer SET never saw \
         the inner's row (composition broken)",
        bag.get("a")
    );
    // Production read path agrees.
    let scanned = read_back_bag(&s, tenant);
    assert_eq!(
        scanned.get("a"),
        Some(&Value::Integer(1)),
        "scan_nodes read path also observes last-writer-wins"
    );
}

/// **The load-bearing pin (SET→REMOVE).** `MATCH (n:L) SET n.a=1 REMOVE
/// n.a` lowers to `Remove(Set(Scan))`. Same dual assertion: the outer
/// REMOVE is terminal (0 rows) AND it composed with the inner SET (the
/// inner SET passed its row to the REMOVE, so `a` is cleared). Pre-fix,
/// the inner SET's empty batch was read as EOS → REMOVE never ran → `a`
/// stayed `1`.
#[test]
fn stacked_set_remove_composes_and_outer_is_terminal() {
    let (s, ctx, nid, tenant) = one_node_fixture();
    let rows = run_rows(
        &format!("MATCH (n:{LABEL}) SET n.a = 1 REMOVE n.a"),
        &s,
        &ctx,
    );

    // Outer REMOVE is terminal → 0 rows.
    assert_eq!(
        rows.len(),
        0,
        "chained SET…REMOVE: the outer REMOVE is terminal → 0 rows, got {} row(s): {:?}",
        rows.len(),
        rows
    );

    // Composition: the REMOVE cleared the SET-written value → it received
    // the inner SET's passed-through row.
    let bag = s
        .node_properties(tenant, nid)
        .expect("node_properties present");
    assert_eq!(
        bag.get("a"),
        None,
        "stacked SET a=1 then REMOVE a must clear `a` (inner→outer row \
         flow); a={:?} would mean the REMOVE never saw the SET's row",
        bag.get("a")
    );
}

/// Three-deep chain `SET a=1 SET a=2 SET a=3` → only the ROOT (outermost,
/// a=3) is terminal; the two inner SETs are stacked. Asserts 0 rows AND
/// `a == 3` — proving the stacked discriminator marks EVERY non-root
/// write-op pass-through (not just the first), so composition threads the
/// whole chain.
#[test]
fn three_deep_set_chain_composes_and_root_is_terminal() {
    let (s, ctx, nid, tenant) = one_node_fixture();
    let rows = run_rows(
        &format!("MATCH (n:{LABEL}) SET n.a = 1 SET n.a = 2 SET n.a = 3"),
        &s,
        &ctx,
    );
    assert_eq!(
        rows.len(),
        0,
        "three-deep SET chain: only the root SET is terminal → 0 rows, got {:?}",
        rows
    );
    let bag = s
        .node_properties(tenant, nid)
        .expect("node_properties present");
    assert_eq!(
        bag.get("a"),
        Some(&Value::Integer(3)),
        "three-deep SET a=1,2,3 must persist 3 (last-writer-wins across the \
         whole chain — every non-root SET passed its row through); got {:?}",
        bag.get("a")
    );
}
