//! #717 regression — `DELETE` / `DETACH DELETE` must drain ALL matched
//! rows across ALL pages, not just the first [`BATCH_ROWS`].
//!
//! ## The bug (#717, MED-HIGH data-correctness)
//!
//! Pre-fix, [`arcgraph_query::executor::ops::DeleteOp::next_batch`]
//! tombstoned ONE upstream batch then returned `Batch::empty(0)`. The
//! materialize drain loop ([`arcgraph_query::executor::execute_with_context`])
//! treats the first empty batch as EOS and STOPS — so a `DELETE`
//! matching more than one [`BATCH_ROWS`] page (BATCH_ROWS = 2048) only
//! deleted the FIRST 2048 rows; every row on a later page was silently
//! NOT deleted (silent partial-delete data-corruption). This is the
//! sibling of the #709/#716 SET/REMOVE drain bug.
//!
//! ## The fix
//!
//! `DeleteOp` is **terminal** (it is never stacked under another
//! write-op — cf. `Pipeline::build`'s `mark_writeop_input_stacked`,
//! which only flips SET/REMOVE), so it now INTERNALLY DRAINS its
//! upstream to real EOS — keeps pulling + tombstoning batches until the
//! Scan is genuinely exhausted — then emits 0 rows to the driver
//! (openCypher v9 §6 RETURN-less terminal-write contract, same as
//! terminal SET/REMOVE per ADR-149/150 §D + ADR-182).
//!
//! ## Why this is the load-bearing proof
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests`: the
//! `> BATCH_ROWS` case below FAILS on the unfixed operator (only the
//! first 2048 of 4103 nodes are deleted → 2055 survive) and PASSES
//! after the drain fix (0 survive). It exercises the full query-side
//! pipeline (parse → bind → type-check → cross-substrate → lower →
//! Pipeline → execute) so the regression covers the real EOS-drain
//! contract, not a hand-built operator shape.
//!
//! ScanOp reads `scan_nodes` ONCE at first-batch + paginates the cached
//! result vec (see `executor/ops/scan.rs`), so mid-drain tombstones do
//! NOT perturb the iteration set — the oracle (post-execute re-scan
//! observes 0 survivors) is exact, not racy.

use arcgraph_core::{LabelId, NodeId, TenantId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{BATCH_ROWS, ExecutionContext};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

// ---------------------------------------------------------------------
// Fixtures — mirror tests/m4_81_materialize_integration.rs
// ---------------------------------------------------------------------

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
}

/// A substrate pre-baked with `n` `:Person` nodes (label id 1 — the
/// first label registered by `cat_basic`). Each node carries trivial
/// properties so the scan emits a realistic row width.
fn substrate_with_n_persons(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
        );
    }
    s
}

/// Walk Parse → Bind → TypeCheck → CrossSubstrate → Lower for `query`.
fn lower_to_plan(
    query: &str,
    catalog: &StubCatalogProvider,
) -> arcgraph_query::logical_plan::LogicalPlan {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, catalog).expect("bind");
    TypeCheckVisitor::check(&mut bound, catalog).expect("type-check");
    CrossSubstrateValidator::validate(&bound, catalog).expect("cross-substrate");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower")
}

/// Count the currently-visible `:Person` nodes (label id 1) — the
/// post-execute oracle for "how many survived the DELETE".
fn surviving_persons(s: &StubExecutorSubstrate) -> usize {
    s.scan_nodes(
        TenantId::DEFAULT,
        Some(LabelId::new(1)),
        arcgraph_core::Lsn::MAX,
    )
    .expect("scan_nodes OK")
    .len()
}

// =====================================================================
// #717 — multi-page DELETE drains all matched rows
// =====================================================================

#[test]
fn delete_node_deletes_all_rows_past_batch_rows() {
    // BATCH_ROWS * 2 + 7 = 4103 nodes → 3 scan batches (2048 + 2048 + 7).
    // Mirrors the materialize/scan multi-batch pins
    // (m4_81_materialize_integration.rs:106, scan.rs:222).
    let n = (BATCH_ROWS as u64 * 2) + 7;
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();

    // Sanity: all n nodes are initially visible.
    assert_eq!(
        surviving_persons(&s),
        n as usize,
        "fixture: all {n} visible pre-DELETE"
    );

    let plan = lower_to_plan("MATCH (n:Person) DELETE n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let result = materialize::materialize(&plan, &s, &ctx).expect("materialize OK");

    // Terminal DELETE emits 0 result rows to the driver (openCypher
    // contract — unchanged by the drain fix).
    assert_eq!(
        result.len(),
        0,
        "terminal DELETE yields 0 result rows, got {}",
        result.len()
    );

    // **Load-bearing assertion (the #717 regression):** EVERY matched
    // node across ALL pages is deleted. Pre-fix this observes
    // `n - BATCH_ROWS` (= 2055) survivors because only the first batch
    // was tombstoned; the drain fix tombstones all of them → 0 survive.
    let remaining = surviving_persons(&s);
    assert_eq!(
        remaining,
        0,
        "ALL {n} matched nodes must be deleted across all {}-row pages; \
         {remaining} survived (pre-#717 leaves the > BATCH_ROWS tail of \
         {} undeleted)",
        BATCH_ROWS,
        n as usize - BATCH_ROWS
    );
}

#[test]
fn detach_delete_node_deletes_all_rows_past_batch_rows() {
    // Same fixture, DETACH DELETE (no rels attached → cascade is a
    // no-op, but the drain path is identical). Proves the drain fix
    // applies to the DETACH variant too.
    let n = (BATCH_ROWS as u64 * 2) + 7;
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    assert_eq!(
        surviving_persons(&s),
        n as usize,
        "fixture: all {n} visible pre-DELETE"
    );

    let plan = lower_to_plan("MATCH (n:Person) DETACH DELETE n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let result = materialize::materialize(&plan, &s, &ctx).expect("materialize OK");

    assert_eq!(
        result.len(),
        0,
        "terminal DETACH DELETE yields 0 result rows"
    );
    let remaining = surviving_persons(&s);
    assert_eq!(
        remaining, 0,
        "ALL {n} matched nodes must be DETACH-deleted across all pages; {remaining} survived"
    );
}

#[test]
fn delete_node_single_batch_still_deletes_all() {
    // Guard against an over-correction that breaks the small (≤ 1 page)
    // case: a sub-BATCH_ROWS match set must still delete every node.
    let n = 5u64;
    let s = substrate_with_n_persons(n);
    let cat = cat_basic();
    assert_eq!(surviving_persons(&s), n as usize);

    let plan = lower_to_plan("MATCH (n:Person) DELETE n", &cat);
    let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
    let result = materialize::materialize(&plan, &s, &ctx).expect("materialize OK");

    assert_eq!(result.len(), 0, "terminal DELETE yields 0 result rows");
    assert_eq!(
        surviving_persons(&s),
        0,
        "single-batch DELETE removes all {n} nodes"
    );
}
