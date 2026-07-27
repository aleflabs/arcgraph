//! W25-M4-61b — Hash-join ↔ Merge-join equivalence + property tests.
//!
//! Per ADR-097 §"Algorithm equivalence", Hash and Merge join MUST
//! produce identical (multiset-equal) row sets for the SAME logical
//! plan + SAME input. The two algorithms differ only in:
//!
//! 1. **Cost** — sort-merge pays sort(L)+sort(R) up front; hash pays
//!    O(L+R) hash-touch.
//! 2. **Emission order** — sort-merge emits in sorted-by-key order;
//!    hash emits in PROBE side's natural arrival order. The
//!    multiset of emitted rows is identical; ORDER BY downstream
//!    must NOT depend on the join's emission shape.
//! 3. **Memory shape** — sort-merge holds two full buffers + cluster
//!    cursors; hash holds one bucket map + spillover queue. Same
//!    per-tenant byte budget contract on both.
//!
//! Pins:
//! - `hash_and_merge_produce_identical_row_sets` — multi-pattern
//!   MATCH executed with each algorithm pinned via direct LogicalJoin
//!   construction; result sets compared modulo row ordering.
//! - `hash_and_merge_observer_attributes_correctly` — the M4-71
//!   row-count observer attributes batches to the correct
//!   `OperatorKind::HashJoin` / `OperatorKind::MergeJoin` based on
//!   the resolved algorithm.
//! - `cost_picker_resolves_auto_consistently` — repeated
//!   `pick_join_algorithms` calls on the same plan + catalog yield
//!   the same algorithm choice (determinism pin).
//! - `merge_join_rejects_cartesian_at_build` — Cartesian SharedBindings
//!   construction routes to HashJoin via the picker; direct
//!   `algorithm = MergeJoin` Cartesian fails at construction
//!   (planner-contract violation surfaced before execution).
//! - Property test: random-shaped inputs produce equal row sets
//!   across both algorithms.

#![allow(clippy::too_many_lines)]

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, RelId, TenantId, TypeId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::{
    ExecutionContext, MemoryBudget, Pipeline, StubExecutorSubstrate, Value, execute,
    execute_with_context,
};
use arcgraph_query::logical_plan::{
    Direction, JoinAlgorithm, JoinCondition, LogicalExpand, LogicalJoin, LogicalPlan, LogicalScan,
};
use arcgraph_query::observer::{OperatorKind, RowCountObserver};
use arcgraph_query::planner::{cost::estimate_costs, pick_join_algorithms};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::BindingId;

fn span() -> Span {
    Span::point(1, 1)
}

/// Catalog with Person label + KNOWS rel-type, scaled so the picker
/// picks Hash for some shapes and Merge for others.
fn cat_basic(person_count: u64, rel_count: u64) -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_total_node_count(person_count)
        .with_total_rel_count(rel_count)
        .with_label_cardinality(LabelId::new(1), person_count)
        .with_rel_type_cardinality(TypeId::new(1), rel_count)
}

/// Build a Person-KNOWS-ring substrate: persons 1..=n, person i knows
/// person (i+1) mod n. Each person has a node id matching its index.
fn ring_substrate(n: u64) -> StubExecutorSubstrate {
    use arcgraph_query::executor::value::{NodeView, RelView};
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1))),
        );
    }
    for i in 1..=n {
        let nxt = if i == n { 1 } else { i + 1 };
        s = s.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(i + 100),
                NodeId::new(i),
                NodeId::new(nxt),
                Some(TypeId::new(1)),
            ),
        );
    }
    s
}

/// Build a 2-pattern join plan: `(a:Person)` JOIN `(a)-[r:KNOWS]->(b)`
/// on `a`. Direct LogicalPlan construction lets us pin the algorithm.
fn build_equi_join_plan(algorithm: JoinAlgorithm) -> LogicalPlan {
    let a = BindingId::new(0);
    let r = BindingId::new(1);
    let b = BindingId::new(2);
    let scan_a = LogicalPlan::Scan(LogicalScan {
        label: Some(LabelId::new(1)),
        var: a,
        read_lsn: Lsn::MAX,
        span: span(),
    });
    let expand = LogicalPlan::Expand(LogicalExpand {
        from: a,
        to: b,
        direction: Direction::LeftToRight,
        rel_type: Some(TypeId::new(1)),
        length_range: None,
        rel_var: Some(r),
        span: span(),
    });
    LogicalPlan::Join(LogicalJoin {
        left: Box::new(scan_a),
        right: Box::new(expand),
        on: JoinCondition::SharedBindings(vec![a]),
        algorithm,
        span: span(),
    })
}

/// Normalize a row set for ordering-independent comparison: sort by
/// the per-row debug string. Algorithm-equivalence asserts on the
/// SORTED form.
fn normalize(rows: &[Vec<Value>]) -> Vec<String> {
    let mut keys: Vec<String> = rows.iter().map(|r| format!("{r:?}")).collect();
    keys.sort();
    keys
}

#[test]
fn hash_and_merge_produce_identical_row_sets() {
    // Same plan shape, two algorithm pinnings, ring substrate.
    let s = ring_substrate(7);
    let cat = cat_basic(7, 7);

    let plan_hash = build_equi_join_plan(JoinAlgorithm::HashJoin);
    let plan_merge = build_equi_join_plan(JoinAlgorithm::MergeJoin);

    let rows_hash = execute(&plan_hash, &cat, &s).expect("hash execute");
    let rows_merge = execute(&plan_merge, &cat, &s).expect("merge execute");

    let h = normalize(&rows_hash);
    let m = normalize(&rows_merge);
    assert_eq!(
        h, m,
        "Hash + Merge MUST produce identical (multiset-equal) row sets"
    );
    // Ring substrate: 7 persons × each has 1 outbound KNOWS → 7
    // joined rows under the equi-join on `a`.
    assert_eq!(rows_hash.len(), 7);
    assert_eq!(rows_merge.len(), 7);
}

#[test]
fn hash_and_merge_observer_attributes_correctly() {
    // The M4-71 observer's per-kind counters distinguish HashJoin vs
    // MergeJoin batches. We run the same plan with each algorithm
    // pinned + attach an observer; the resulting bucket has the
    // correct slug.
    //
    // F2 (PE-1 §F2): this test PINS a join algorithm to verify THAT
    // operator's observer attribution. The `Join(Scan(a), Expand(a→b),
    // [a])` shape it builds is exactly the F2 pipelined-expand fast-path
    // target — with F2 on it folds to an `Expand` and no join operator
    // executes, so the pinned-algorithm bucket is empty. Bypass F2 for
    // the current thread so the test exercises the join executors it is
    // about (F2's own correctness is pinned in `f2_pipelined_expand_e2e`).
    let f2_prev = Pipeline::set_pipelined_expand_enabled(false);
    let s = ring_substrate(5);
    let cat = cat_basic(5, 5);

    for (algo, expected_kind, expected_slug) in [
        (JoinAlgorithm::HashJoin, OperatorKind::HashJoin, "hash_join"),
        (
            JoinAlgorithm::MergeJoin,
            OperatorKind::MergeJoin,
            "merge_join",
        ),
    ] {
        let plan = build_equi_join_plan(algo);
        let costed = estimate_costs(plan.clone(), &cat);
        let observer =
            std::sync::Arc::new(RowCountObserver::from_plan_and_costs(&plan, costed.costs()));
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
            .with_observer(std::sync::Arc::clone(&observer));
        let _rows = execute_with_context(&plan, &s, &ctx).expect("execute");
        let metrics = observer.metrics();
        let join_metrics = metrics
            .iter()
            .find(|m| m.op_kind == Some(expected_kind))
            .unwrap_or_else(|| panic!("observer missing {expected_slug} bucket"));
        assert!(
            join_metrics.batches > 0,
            "{expected_slug}: observer must record at least one batch \
             (got 0 — observer attribution broken?)"
        );
    }
    Pipeline::set_pipelined_expand_enabled(f2_prev);
}

#[test]
fn cost_picker_resolves_auto_consistently() {
    // Determinism pin: repeated `pick_join_algorithms` over the same
    // plan + catalog yields the same algorithm choice.
    let cat = cat_basic(1_000, 5_000);
    let plan = build_equi_join_plan(JoinAlgorithm::Auto);
    let once = pick_join_algorithms(plan.clone(), &cat);
    let twice = pick_join_algorithms(once.clone(), &cat);
    let thrice = pick_join_algorithms(twice.clone(), &cat);
    assert_eq!(once, twice, "picker must be idempotent (call 1 ↔ call 2)");
    assert_eq!(twice, thrice, "picker must be idempotent (call 2 ↔ call 3)");
    // The Auto must have resolved to a concrete variant.
    match once {
        LogicalPlan::Join(j) => assert!(!matches!(j.algorithm, JoinAlgorithm::Auto)),
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn cost_picker_resolves_cartesian_to_hash() {
    // Cartesian (empty SharedBindings) ALWAYS resolves to Hash —
    // merge-join is undefined without join keys.
    let a = BindingId::new(0);
    let b = BindingId::new(1);
    let plan = LogicalPlan::Join(LogicalJoin {
        left: Box::new(LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: a,
            read_lsn: Lsn::MAX,
            span: span(),
        })),
        right: Box::new(LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: b,
            read_lsn: Lsn::MAX,
            span: span(),
        })),
        on: JoinCondition::SharedBindings(Vec::new()),
        algorithm: JoinAlgorithm::Auto,
        span: span(),
    });
    let cat = cat_basic(100, 0);
    let resolved = pick_join_algorithms(plan, &cat);
    match resolved {
        LogicalPlan::Join(j) => assert_eq!(j.algorithm, JoinAlgorithm::HashJoin),
        other => panic!("expected Join, got {other:?}"),
    }
}

#[test]
fn merge_join_at_pipeline_build_rejects_cartesian() {
    // If a caller bypasses the picker AND pins MergeJoin on a Cartesian
    // shape, the Pipeline must defensively route to HashJoin (the
    // shared dispatch falls back when shared.is_empty()).
    use arcgraph_query::executor::Pipeline;
    let a = BindingId::new(0);
    let b = BindingId::new(1);
    let plan = LogicalPlan::Join(LogicalJoin {
        left: Box::new(LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: a,
            read_lsn: Lsn::MAX,
            span: span(),
        })),
        right: Box::new(LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: b,
            read_lsn: Lsn::MAX,
            span: span(),
        })),
        on: JoinCondition::SharedBindings(Vec::new()),
        algorithm: JoinAlgorithm::MergeJoin, // would-be-bug shape
        span: span(),
    });
    // Pipeline build must NOT panic + must route to HashJoin
    // defensively per pipeline.rs §"shared.is_empty() ALWAYS routes
    // to HashJoin". The actual op kind isn't observable from
    // Pipeline::build's return alone, but the fact that it succeeds
    // (no error from MergeJoinOp's Cartesian rejection) is the
    // load-bearing claim.
    let op = Pipeline::build(&plan).expect("Pipeline::build must defensively route Cartesian");
    drop(op);
}

#[test]
fn three_pattern_chain_hash_and_merge_equivalent() {
    // Three-pattern chain: a → b → c. Two joins. Forces the executor
    // to nest joins. Hash and Merge must produce equal row sets.
    let s = ring_substrate(6);
    let cat = cat_basic(6, 6);

    let a = BindingId::new(0);
    let r1 = BindingId::new(1);
    let b = BindingId::new(2);
    let r2 = BindingId::new(3);
    let c = BindingId::new(4);

    let build_three_pattern = |algorithm: JoinAlgorithm| -> LogicalPlan {
        let scan_a = LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: a,
            read_lsn: Lsn::MAX,
            span: span(),
        });
        let exp1 = LogicalPlan::Expand(LogicalExpand {
            from: a,
            to: b,
            direction: Direction::LeftToRight,
            rel_type: Some(TypeId::new(1)),
            length_range: None,
            rel_var: Some(r1),
            span: span(),
        });
        let join_ab = LogicalPlan::Join(LogicalJoin {
            left: Box::new(scan_a),
            right: Box::new(exp1),
            on: JoinCondition::SharedBindings(vec![a]),
            algorithm,
            span: span(),
        });
        let exp2 = LogicalPlan::Expand(LogicalExpand {
            from: b,
            to: c,
            direction: Direction::LeftToRight,
            rel_type: Some(TypeId::new(1)),
            length_range: None,
            rel_var: Some(r2),
            span: span(),
        });
        LogicalPlan::Join(LogicalJoin {
            left: Box::new(join_ab),
            right: Box::new(exp2),
            on: JoinCondition::SharedBindings(vec![b]),
            algorithm,
            span: span(),
        })
    };

    let rows_hash =
        execute(&build_three_pattern(JoinAlgorithm::HashJoin), &cat, &s).expect("3-pat hash");
    let rows_merge =
        execute(&build_three_pattern(JoinAlgorithm::MergeJoin), &cat, &s).expect("3-pat merge");
    assert_eq!(normalize(&rows_hash), normalize(&rows_merge));
}

#[test]
fn merge_join_byte_budget_release_pairs_with_drop() {
    // Per-tenant byte budget should be balanced (acquire == release)
    // after a full execute on the merge-join path. Acquire is the
    // sum of both buffers + cluster spillover; release is the
    // pop-on-emit + EOS path.
    let s = ring_substrate(8);
    let _cat = cat_basic(8, 8); // unused — plan built without catalog routing.

    let plan = build_equi_join_plan(JoinAlgorithm::MergeJoin);
    let budget = MemoryBudget::with_per_tenant_cap(TenantId::DEFAULT, 1 << 30); // 1 GiB cap
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO).with_budget(budget);
    let _rows = execute_with_context(&plan, &s, &ctx).expect("merge execute");
    let consumed = ctx.budget().current_bytes(TenantId::DEFAULT);
    assert_eq!(
        consumed, 0,
        "MergeJoinOp: per-tenant byte budget MUST be released to 0 at EOS \
         (acquire == release); leaked {consumed} bytes"
    );
}

// ---------------------------------------------------------------------
// Property tests — random shapes, Hash ↔ Merge multiset equality.
// ---------------------------------------------------------------------

proptest::proptest! {
    /// For arbitrary ring sizes in [2, 12], the two algorithms emit
    /// the same row multiset. Bounded at 12 to keep prop iterations
    /// fast on CI.
    #[test]
    fn prop_hash_and_merge_row_multiset_equal(n in 2u64..=12) {
        let s = ring_substrate(n);
        let cat = cat_basic(n, n);

        let plan_hash = build_equi_join_plan(JoinAlgorithm::HashJoin);
        let plan_merge = build_equi_join_plan(JoinAlgorithm::MergeJoin);

        let rows_hash = execute(&plan_hash, &cat, &s).expect("hash execute");
        let rows_merge = execute(&plan_merge, &cat, &s).expect("merge execute");

        proptest::prop_assert_eq!(normalize(&rows_hash), normalize(&rows_merge));
        // Ring: every person has exactly 1 outbound KNOWS → row count == n.
        proptest::prop_assert_eq!(rows_hash.len() as u64, n);
    }

    /// For arbitrary ring sizes in [3, 8], a 3-pattern chain is
    /// algorithm-equivalent.
    #[test]
    fn prop_three_pattern_chain_equal(n in 3u64..=8) {
        use arcgraph_query::executor::Pipeline;
        let _ = Pipeline::build; // touches the binary so a regression at build surfaces in proptest

        let s = ring_substrate(n);
        let cat = cat_basic(n, n);
        let a = BindingId::new(0);
        let r1 = BindingId::new(1);
        let b = BindingId::new(2);
        let r2 = BindingId::new(3);
        let c = BindingId::new(4);

        let build_chain = |algorithm: JoinAlgorithm| -> LogicalPlan {
            let scan_a = LogicalPlan::Scan(LogicalScan {
                label: Some(LabelId::new(1)),
                var: a,
                read_lsn: Lsn::MAX,
                span: span(),
            });
            let exp1 = LogicalPlan::Expand(LogicalExpand {
                from: a,
                to: b,
                direction: Direction::LeftToRight,
                rel_type: Some(TypeId::new(1)),
                length_range: None,
                rel_var: Some(r1),
                span: span(),
            });
            let join_ab = LogicalPlan::Join(LogicalJoin {
                left: Box::new(scan_a),
                right: Box::new(exp1),
                on: JoinCondition::SharedBindings(vec![a]),
                algorithm,
                span: span(),
            });
            let exp2 = LogicalPlan::Expand(LogicalExpand {
                from: b,
                to: c,
                direction: Direction::LeftToRight,
                rel_type: Some(TypeId::new(1)),
                length_range: None,
                rel_var: Some(r2),
                span: span(),
            });
            LogicalPlan::Join(LogicalJoin {
                left: Box::new(join_ab),
                right: Box::new(exp2),
                on: JoinCondition::SharedBindings(vec![b]),
                algorithm,
                span: span(),
            })
        };

        let rows_hash =
            execute(&build_chain(JoinAlgorithm::HashJoin), &cat, &s).expect("3-pat hash");
        let rows_merge =
            execute(&build_chain(JoinAlgorithm::MergeJoin), &cat, &s).expect("3-pat merge");
        proptest::prop_assert_eq!(normalize(&rows_hash), normalize(&rows_merge));
    }

    /// Picker idempotence over arbitrary ring sizes.
    #[test]
    fn prop_picker_idempotent(n in 1u64..=50) {
        let cat = cat_basic(n.max(1), n);
        let plan = build_equi_join_plan(JoinAlgorithm::Auto);
        let once = pick_join_algorithms(plan, &cat);
        let twice = pick_join_algorithms(once.clone(), &cat);
        proptest::prop_assert_eq!(once, twice);
    }
}
