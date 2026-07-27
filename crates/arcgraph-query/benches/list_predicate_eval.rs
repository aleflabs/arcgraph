//! Criterion benchmark for the **ADR-188 list-predicate / reduce
//! evaluator** over a ≥1k-element list.
//!
//! Per ADR-188 Decision 1 §Back-of-envelope + §Open-questions "Bench"
//! item + testing strategy ("every hot path has a benchmark; >10% regression
//! blocks merge"): the per-element extended-row synthesis mechanism must
//! add NO asymptotic overhead over the inner-predicate cost, and — the
//! load-bearing BoE claim — must do **no per-element heap allocation in
//! the scalar case**. The implementation realizes this with a **reused
//! scratch buffer** (ADR-188 Decision 1): a single `Vec<Value>` =
//! `input_row + scoped-var-slot(s)` is allocated **once per evaluation**
//! (one fresh `Vec` per nesting-level *invocation*, NOT one per element),
//! and the scoped-var slot is **overwritten in place** per element. The
//! per-element cost is therefore the scalar clone (a `memcpy` of a
//! ≤32-byte scalar `Value`) plus the inner `evaluate(pred)` — no
//! per-element heap allocation.
//!
//! ## Why two input-row widths
//!
//! The `*_empty_row` benches evaluate against an **empty** input row
//! (`&[]`): the scratch buffer is just the scoped-var slot(s), so they
//! isolate the per-element overwrite + inner-eval cost. The `*_wide_row`
//! benches evaluate against a **16-column** input row: the one-time
//! buffer allocation copies `input_width` cells, so these prove the
//! buffer is allocated **once** and reused — a regression to
//! fresh-`Vec`-per-element would scale the `extend_from_slice(input_row)`
//! copy by N (1000×), which the wide-row shape makes visible (a 16-cell
//! copy × 1000 elements vs once). With the reused buffer the wide-row
//! cost is a single 16-cell copy regardless of N; with per-element
//! allocation it would be 16 000 cell-copies. The empty-row baseline
//! masks this (a 0-cell copy is free either way) — hence the wide-row
//! variant is the load-bearing evidence for the "no per-element heap
//! allocation; buffer reused per evaluation" claim.
//!
//! Run: `cargo bench -p arcgraph-query --bench list_predicate_eval`.

use arcgraph_query::Span;
use arcgraph_query::ast::{BinOp, Expression, Literal, Quantifier};
use arcgraph_query::executor::Value;
use arcgraph_query::executor::eval::{Parameters, evaluate};
use arcgraph_query::semantic::bound_ast::{BindingId, BoundExpression};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

const N: usize = 1000;

// The scoped iteration variable's binding id. With an empty input row,
// the evaluator appends the element at slot 0 and the scoped closure maps
// this id → slot 0.
const X_BID: BindingId = BindingId::new(0);
// reduce: acc at slot 0, x at slot 1.
const ACC_BID: BindingId = BindingId::new(0);
const RX_BID: BindingId = BindingId::new(1);

fn sp() -> Span {
    Span::point(1, 1)
}

fn lit_int(n: i64) -> BoundExpression {
    BoundExpression::Literal {
        value: Literal::Integer(n),
        span: sp(),
        type_info: None,
    }
}

/// A 1000-element integer list literal `BoundExpression`.
fn big_list() -> BoundExpression {
    let inner: Vec<Expression> = (0..N as i64)
        .map(|n| Expression::Literal(Literal::Integer(n)))
        .collect();
    BoundExpression::Literal {
        value: Literal::List(inner),
        span: sp(),
        type_info: None,
    }
}

fn var_x() -> BoundExpression {
    BoundExpression::VariableRef {
        name: "x".into(),
        binding_id: X_BID,
        span: sp(),
        type_info: None,
    }
}

/// `x > threshold` predicate.
fn pred_x_gt(threshold: i64) -> BoundExpression {
    BoundExpression::BinaryOp {
        op: BinOp::Gt,
        lhs: Box::new(var_x()),
        rhs: Box::new(lit_int(threshold)),
        span: sp(),
        type_info: None,
    }
}

fn list_pred(q: Quantifier, pred: BoundExpression) -> BoundExpression {
    BoundExpression::ListPredicate {
        quantifier: q,
        var_bid: X_BID,
        list: Box::new(big_list()),
        predicate: Box::new(pred),
        span: sp(),
        type_info: None,
    }
}

fn no_schema() -> impl Fn(BindingId) -> Option<usize> {
    |_| None
}

/// Width of the "wide" input row — large enough that a regression to
/// fresh-`Vec`-per-element (which would `extend_from_slice` this many
/// cells N=1000 times instead of once) is visible above noise.
const WIDE: usize = 16;

/// A 16-column input row of scalar `Value`s. These cells are never
/// addressed by any binding id in the predicate / fold body (the scoped
/// closure inside the evaluator maps the scoped var to slot `row.len()`,
/// and `no_schema` returns `None` for every outer binding) — they are
/// pure ballast so the one-time scratch-buffer `extend_from_slice(row)`
/// copies `WIDE` cells. With the reused buffer that copy happens ONCE
/// per evaluation; a per-element-allocation regression would do it
/// N=1000 times, scaling this `WIDE`-cell copy by 1000×.
fn wide_row() -> Vec<Value> {
    (0..WIDE as i64).map(Value::Integer).collect()
}

/// `reduce(s = 0, x IN [0..1000] | s + x)` as a `BoundExpression`.
fn reduce_sum_expr() -> BoundExpression {
    let body = BoundExpression::BinaryOp {
        op: BinOp::Add,
        lhs: Box::new(BoundExpression::VariableRef {
            name: "s".into(),
            binding_id: ACC_BID,
            span: sp(),
            type_info: None,
        }),
        rhs: Box::new(BoundExpression::VariableRef {
            name: "x".into(),
            binding_id: RX_BID,
            span: sp(),
            type_info: None,
        }),
        span: sp(),
        type_info: None,
    };
    BoundExpression::Reduce {
        acc_bid: ACC_BID,
        init: Box::new(lit_int(0)),
        var_bid: RX_BID,
        list: Box::new(big_list()),
        expr: Box::new(body),
        span: sp(),
        type_info: None,
    }
}

fn bench_all_no_short_circuit(c: &mut Criterion) {
    // all(x IN [0..1000] WHERE x >= 0) — every element passes ⇒ NO
    // short-circuit ⇒ all 1000 inner evals run. This is the worst-case
    // (full-scan) shape that the BoE's "1000 slot-overwrites + 1000 inner
    // evals, single reused scratch buffer, no per-element heap alloc"
    // number describes.
    let e = list_pred(Quantifier::All, pred_x_gt(-1));
    let s = no_schema();
    let params = Parameters::new();
    // Empty input row: scratch buffer is just the scoped slot; isolates
    // the per-element overwrite + inner-eval cost.
    c.bench_function("list_predicate::all_1k_empty_row", |b| {
        b.iter(|| {
            black_box(evaluate(black_box(&e), &[], &s, &params).unwrap());
        });
    });
    // 16-column input row: the one-time scratch-buffer allocation copies
    // WIDE cells. Reused-buffer ⇒ ONE 16-cell copy regardless of N;
    // a fresh-Vec-per-element regression ⇒ 16 000 cell-copies. This is
    // the load-bearing "no per-element heap alloc" evidence.
    let wide = wide_row();
    c.bench_function("list_predicate::all_1k_wide_row", |b| {
        b.iter(|| {
            black_box(evaluate(black_box(&e), black_box(&wide), &s, &params).unwrap());
        });
    });
}

fn bench_any_no_short_circuit(c: &mut Criterion) {
    // any(x IN [0..1000] WHERE x > 10000) — no element matches ⇒ NO
    // short-circuit ⇒ full 1000-element scan returning false.
    let e = list_pred(Quantifier::Any, pred_x_gt(10_000));
    let s = no_schema();
    let params = Parameters::new();
    c.bench_function("list_predicate::any_1k_empty_row", |b| {
        b.iter(|| {
            black_box(evaluate(black_box(&e), &[], &s, &params).unwrap());
        });
    });
    let wide = wide_row();
    c.bench_function("list_predicate::any_1k_wide_row", |b| {
        b.iter(|| {
            black_box(evaluate(black_box(&e), black_box(&wide), &s, &params).unwrap());
        });
    });
}

fn bench_reduce_sum_1k(c: &mut Criterion) {
    // reduce(s = 0, x IN [0..1000] | s + x) — full 1000-element fold.
    // Each iteration OVERWRITES the two scoped slots [acc, x] on the
    // reused scratch buffer (allocated once, capacity row.len()+2) — the
    // BoE's per-element cost. No HashMap, no N-row materialization, no
    // per-iteration heap allocation.
    let e = reduce_sum_expr();
    let s = no_schema();
    let params = Parameters::new();
    c.bench_function("list_predicate::reduce_sum_1k_empty_row", |b| {
        b.iter(|| {
            black_box(evaluate(black_box(&e), &[], &s, &params).unwrap());
        });
    });
    // 16-column input row: the one-time buffer allocation copies WIDE
    // cells once; a per-iteration-allocation regression copies them
    // 1000× — the reused-buffer evidence for `reduce`.
    let wide = wide_row();
    c.bench_function("list_predicate::reduce_sum_1k_wide_row", |b| {
        b.iter(|| {
            black_box(evaluate(black_box(&e), black_box(&wide), &s, &params).unwrap());
        });
    });
}

criterion_group!(
    benches,
    bench_all_no_short_circuit,
    bench_any_no_short_circuit,
    bench_reduce_sum_1k,
);
criterion_main!(benches);
