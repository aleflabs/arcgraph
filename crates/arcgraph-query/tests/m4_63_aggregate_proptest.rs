//! M4-63 aggregate associativity proptest per ADR-038 amendment-02
//! §M4.f + amendment-03 §TIER-2-b.
//!
//! # Invariant
//!
//! For any partition of an input row stream into two halves A and B,
//! the aggregate-over-(A ⊕ B) MUST equal the partial-aggregate-over-A
//! MERGED with the partial-aggregate-over-B for an associative
//! aggregate function (COUNT/SUM/MIN/MAX). This pins the canonical
//! "blocking aggregate" invariant; a future M4-72 streaming /
//! pre-sorted aggregate optimization MUST preserve it.
//!
//! # Hardening
//!
//! `PROPTEST_CASES=10000` per the M4-63 spawn brief. The default
//! shipped with the crate is `proptest`'s default (256 cases per
//! invocation); the gauntlet ratchet runs at 10000 via env override
//! per W12α spawn brief test artifacts §"M4-63 tests".
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-63 cite.
//! - **ADR-038 amendment-03 §TIER-2-b** — 3VL aggregate semantics
//!   (NULL exclusion across COUNT/SUM/MIN/MAX).
//! - `feedback_determinism_oracle_concurrency_tests.md` — proptest
//!   reference-model discipline for deterministic algorithms.

use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};
use arcgraph_query::error::Span;
use arcgraph_query::executor::ops::{AggregateCall, AggregateOp, PhysicalOperator, ScanOp};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::AggregationKind;
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression, BoundPropertyRef};
use proptest::prelude::*;

fn make_substrate(values: &[Option<i64>]) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for (i, v) in values.iter().enumerate() {
        let cell = match v {
            Some(n) => Value::Integer(*n),
            None => Value::Null,
        };
        s = s.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new((i + 1) as u64), Some(LabelId::new(1)))
                .with_property("age", cell),
        );
    }
    s
}

fn person_scan() -> ScanOp {
    ScanOp::new(BindingId::new(0), Some(LabelId::new(1)), Lsn::MAX)
}

fn ctx() -> ExecutionContext {
    ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
}

fn prop_age() -> BoundExpression {
    BoundExpression::PropertyAccess {
        base: Box::new(BoundExpression::VariableRef {
            name: "n".into(),
            binding_id: BindingId::new(0),
            span: Span::point(1, 1),
            type_info: None,
        }),
        path: vec![BoundPropertyRef {
            name: "age".into(),
            property_id: None,
            span: Span::point(1, 1),
        }],
        span: Span::point(1, 1),
        type_info: None,
    }
}

fn run_count(values: &[Option<i64>]) -> i64 {
    let s = make_substrate(values);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Count,
            arg: prop_age(),
            output_id: BindingId::new(2),
        }],
    );
    let ctx = ctx();
    let b = op.next_batch(&ctx, &s).unwrap();
    match b.row(0)[0] {
        Value::Integer(n) => n,
        _ => panic!("count must be Integer"),
    }
}

fn run_sum(values: &[Option<i64>]) -> Option<i64> {
    let s = make_substrate(values);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Sum,
            arg: prop_age(),
            output_id: BindingId::new(2),
        }],
    );
    let ctx = ctx();
    let b = op.next_batch(&ctx, &s).unwrap();
    match &b.row(0)[0] {
        Value::Integer(n) => Some(*n),
        Value::Float(f) => Some(*f as i64),
        Value::Null => None,
        other => panic!("expected numeric or null; got {other:?}"),
    }
}

fn run_min(values: &[Option<i64>]) -> Option<i64> {
    let s = make_substrate(values);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Min,
            arg: prop_age(),
            output_id: BindingId::new(2),
        }],
    );
    let ctx = ctx();
    let b = op.next_batch(&ctx, &s).unwrap();
    match &b.row(0)[0] {
        Value::Integer(n) => Some(*n),
        Value::Null => None,
        other => panic!("expected integer or null; got {other:?}"),
    }
}

fn run_max(values: &[Option<i64>]) -> Option<i64> {
    let s = make_substrate(values);
    let mut op = AggregateOp::new(
        PhysicalOperator::Scan(person_scan()),
        Vec::new(),
        vec![AggregateCall {
            distinct: false,
            star: false,
            kind: AggregationKind::Max,
            arg: prop_age(),
            output_id: BindingId::new(2),
        }],
    );
    let ctx = ctx();
    let b = op.next_batch(&ctx, &s).unwrap();
    match &b.row(0)[0] {
        Value::Integer(n) => Some(*n),
        Value::Null => None,
        other => panic!("expected integer or null; got {other:?}"),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Default cases; PROPTEST_CASES=10000 env override per the
        // M4-63 spawn brief gauntlet step 6.
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// COUNT is associative under partition + merge: COUNT(A ⊕ B)
    /// = COUNT(A) + COUNT(B). Pins the amendment-03 §TIER-2-b NULL
    /// exclusion: NULLs in either half are excluded from the merged
    /// total just as they are from the whole-stream COUNT.
    #[test]
    fn count_is_associative_under_partition_merge(
        a in proptest::collection::vec(proptest::option::of(any::<i64>()), 0..30),
        b in proptest::collection::vec(proptest::option::of(any::<i64>()), 0..30),
    ) {
        let combined: Vec<Option<i64>> = a.iter().chain(b.iter()).copied().collect();
        let count_combined = run_count(&combined);
        let count_a = run_count(&a);
        let count_b = run_count(&b);
        prop_assert_eq!(count_combined, count_a + count_b);
    }

    /// SUM is associative under partition + merge with NULL exclusion.
    /// Use bounded integers to avoid overflow (the operator surfaces
    /// `ExecutionError::Eval("integer overflow")` on overflow; the
    /// proptest works in a safe range so the merge invariant holds).
    #[test]
    fn sum_is_associative_under_partition_merge(
        a in proptest::collection::vec(proptest::option::of(-1_000_000i64..=1_000_000i64), 0..30),
        b in proptest::collection::vec(proptest::option::of(-1_000_000i64..=1_000_000i64), 0..30),
    ) {
        let combined: Vec<Option<i64>> = a.iter().chain(b.iter()).copied().collect();
        let sum_combined = run_sum(&combined);
        let sum_a = run_sum(&a);
        let sum_b = run_sum(&b);
        let merged = match (sum_a, sum_b) {
            (None, None) => {
                if combined.iter().all(|v| v.is_none()) {
                    None
                } else {
                    // Both halves report None (all-NULL) but combined
                    // had a non-NULL — impossible.
                    panic!("merge invariant violated");
                }
            }
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (Some(a), Some(b)) => Some(a + b),
        };
        prop_assert_eq!(sum_combined, merged);
    }

    /// MIN is idempotent under merge: MIN(A ⊕ B) = min(MIN(A), MIN(B))
    /// where None (= "no non-NULL operand") is the identity element.
    #[test]
    fn min_is_idempotent_under_merge(
        a in proptest::collection::vec(proptest::option::of(any::<i64>()), 0..30),
        b in proptest::collection::vec(proptest::option::of(any::<i64>()), 0..30),
    ) {
        let combined: Vec<Option<i64>> = a.iter().chain(b.iter()).copied().collect();
        let min_combined = run_min(&combined);
        let min_a = run_min(&a);
        let min_b = run_min(&b);
        let merged = match (min_a, min_b) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(a.min(b)),
        };
        prop_assert_eq!(min_combined, merged);
    }

    /// MAX is idempotent under merge.
    #[test]
    fn max_is_idempotent_under_merge(
        a in proptest::collection::vec(proptest::option::of(any::<i64>()), 0..30),
        b in proptest::collection::vec(proptest::option::of(any::<i64>()), 0..30),
    ) {
        let combined: Vec<Option<i64>> = a.iter().chain(b.iter()).copied().collect();
        let max_combined = run_max(&combined);
        let max_a = run_max(&a);
        let max_b = run_max(&b);
        let merged = match (max_a, max_b) {
            (None, x) | (x, None) => x,
            (Some(a), Some(b)) => Some(a.max(b)),
        };
        prop_assert_eq!(max_combined, merged);
    }
}
