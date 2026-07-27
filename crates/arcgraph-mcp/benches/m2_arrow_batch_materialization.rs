//! v2 M2 (§M2.4) — Criterion bench for
//! [`arcgraph_mcp::storage::arrow_batch::projected_rows_to_record_batch`],
//! under the testing strategy ("Every public performance-sensitive API has a
//! benchmark") + the design's batch-materialization motivation.
//!
//! Compares, over the same 2,048 projected rows × 4 columns
//! (the executor's BATCH_ROWS grain):
//! - `arrow_columnar` — the §M2.4 column-wise typed-builder
//!   materialization into a `RecordBatch`.
//! - `row_major_baseline` — the v1.0-α executor's row shape
//!   (`Vec<Vec<Value>>` clone-per-cell), the IR the batch path
//!   bypasses.
//!
//! The numbers establish the baseline the M4-64b vectorized-executor
//! swap will be measured against; regressions > 10% block merges per
//! testing strategy.

use std::collections::BTreeMap;

use arcgraph_mcp::storage::arrow_batch::projected_rows_to_record_batch;
use arcgraph_query::executor::value::Value;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// One projected row: `(node_id, projected bag)` — the shape
/// `projected_rows_to_record_batch` consumes.
type ProjectedRow = (u64, BTreeMap<String, Value>);

fn fixture(rows: usize) -> (Vec<String>, Vec<ProjectedRow>) {
    let projected: Vec<String> = ["sev", "attempt", "score", "open"]
        .into_iter()
        .map(str::to_string)
        .collect();
    let data = (0..rows as u64)
        .map(|i| {
            let mut bag = BTreeMap::new();
            bag.insert("sev".to_string(), Value::String(format!("P{}", i % 4)));
            bag.insert("attempt".to_string(), Value::Integer(i as i64 * 37));
            bag.insert("score".to_string(), Value::Float(i as f64 * 0.31));
            bag.insert("open".to_string(), Value::Boolean(i % 3 == 0));
            (i + 1, bag)
        })
        .collect();
    (projected, data)
}

fn bench_batch_materialization(c: &mut Criterion) {
    let (projected, rows) = fixture(2048);

    c.bench_function("m2_arrow_columnar_2048x4", |b| {
        b.iter(|| {
            let batch = projected_rows_to_record_batch(black_box(&projected), black_box(&rows))
                .expect("batch");
            black_box(batch.num_rows());
        });
    });

    c.bench_function("m2_row_major_baseline_2048x4", |b| {
        b.iter(|| {
            // The executor's row-major IR: one Vec<Value> per row,
            // cells cloned out of the bag (the shape ScanOp emits).
            let out: Vec<Vec<Value>> = rows
                .iter()
                .map(|(_, bag)| {
                    projected
                        .iter()
                        .map(|k| bag.get(k).cloned().unwrap_or(Value::Null))
                        .collect()
                })
                .collect();
            black_box(out.len());
        });
    });
}

criterion_group!(benches, bench_batch_materialization);
criterion_main!(benches);
