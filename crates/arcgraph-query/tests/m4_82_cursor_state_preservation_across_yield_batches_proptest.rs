//! W13β M4-82 cursor-state preservation proptest per spawn prompt:
//! "1 proptest (`m4_82_cursor_state_preservation_across_yield_batches`):
//! cursor-state is preserved across yield-batch boundaries (no row
//! dropped or duplicated between batches)."
//!
//! # Invariant
//!
//! For a substrate carrying N nodes, the cursor's full sequence of
//! `next_batch()` calls (until EOS) emits EXACTLY N rows total, with
//! each substrate node appearing in exactly one yielded row. The
//! invariant covers:
//! - **No row drop** — no node from the substrate is missing from the
//!   cumulative cursor output.
//! - **No row duplication** — no node appears twice across batches.
//! - **Batch-boundary continuity** — the cursor's per-batch chunking
//!   doesn't skip a row at the boundary OR re-emit one.
//!
//! # Strategy
//!
//! Random N in `0..=20480` (10× BATCH_ROWS). The strategy includes
//! 0 (degenerate empty case), exact BATCH_ROWS (boundary case),
//! BATCH_ROWS + 1 (boundary-plus-one case), and random N up to 10x
//! to exercise multi-batch chunking under varying row counts.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.h** — primary M4-82 cite.
//! - **ADR-038 §2 D-26** — uniform per-batch row schema invariant
//!   (this proptest pins the per-batch shape too — every yielded row
//!   has the same column count).

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{StreamingCursor, parse};

use proptest::prelude::*;

fn cat_basic() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["name", "age"])
}

fn substrate_with_n_persons(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64)),
        );
    }
    s
}

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

/// Generate random N in `0..=20480` with bias toward boundary values.
fn n_strategy() -> impl Strategy<Value = u64> {
    use arcgraph_query::executor::BATCH_ROWS;
    let br = BATCH_ROWS as u64;
    prop_oneof![
        // Heavy weight on random N — exercises the hot path.
        20 => 0u64..=20480_u64,
        // Boundary cases.
        1 => Just(0u64),
        1 => Just(1u64),
        1 => Just(br - 1),
        1 => Just(br),
        1 => Just(br + 1),
        1 => Just(2 * br),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        // 256 cases on the integration-test path (vs the M4-81's
        // 10K) — each case opens a cursor + walks N rows; 256 covers
        // the boundary value bias above + a representative random
        // sample.
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// Cursor-state preservation across yield-batch boundaries:
    /// every node in the substrate appears in exactly one yielded
    /// row, and the cumulative row count equals N.
    #[test]
    fn cursor_state_preserved_across_yield_batches(n in n_strategy()) {
        let s = substrate_with_n_persons(n);
        let cat = cat_basic();
        let plan = lower_to_plan("MATCH (n:Person) RETURN n", &cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        let mut cursor = StreamingCursor::open(&plan, ctx, &s).expect("open");

        let mut total_rows: u64 = 0;
        let mut seen_node_ids: std::collections::BTreeSet<u64> =
            std::collections::BTreeSet::new();
        let mut col_counts: std::collections::BTreeSet<usize> =
            std::collections::BTreeSet::new();

        loop {
            let batch_opt = cursor.next_batch().expect("next_batch");
            let Some(rows) = batch_opt else {
                break;
            };
            for row in &rows {
                col_counts.insert(row.len());
                // Each row has one Value::Node cell at column 0
                // (the bare RETURN n projection).
                match row.first() {
                    Some(Value::Node(nv)) => {
                        let new = seen_node_ids.insert(nv.id.raw());
                        prop_assert!(
                            new,
                            "duplicate node id {} observed across batches",
                            nv.id.raw()
                        );
                    }
                    other => prop_assert!(
                        false,
                        "expected Value::Node at column 0, got {other:?}"
                    ),
                }
            }
            total_rows += rows.len() as u64;
        }

        // Total rows == N (no drops, no extras).
        prop_assert_eq!(total_rows, n, "cumulative row count must match substrate N");
        // Every node appeared exactly once.
        prop_assert_eq!(
            seen_node_ids.len() as u64,
            n,
            "every substrate node appeared once across batches"
        );
        // Per-row column count is uniform (per ADR-038 §2 D-26).
        prop_assert!(
            col_counts.len() <= 1,
            "per-batch row schema is uniform (col_counts: {col_counts:?})"
        );
        // Cursor auto-closed on EOS.
        prop_assert!(cursor.is_closed(), "post-EOS: cursor closed");
        // No-leak: cursor.rows_emitted() matches our total.
        prop_assert_eq!(cursor.rows_emitted(), total_rows);
    }
}
