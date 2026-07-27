//! TEL micro-benchmarks (M2-a handoff).
//!
//! Measures:
//! - `scan_full_block`: sequential scan of a `MAX_BLOCK_BYTES` block
//!   (2047 live entries, snapshot LSN admits all).
//! - `append_into_full_block`: single-threaded append hot path at a
//!   mid-sized (1024-byte) block.
//!
//! Run: `cargo bench -p arcgraph-storage --bench tel`.
//! Reports: `target/criterion/tel/`.

use std::hint::black_box;

use arcgraph_core::{LabelId, Lsn, NodeId, RelId, TelEntry, TenantId};
use arcgraph_storage::tel::{MAX_BLOCK_BYTES, TelBlock};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn filled_block(block_size: u32) -> TelBlock {
    let b = TelBlock::new(
        NodeId::new(1),
        LabelId::new(1),
        block_size,
        TenantId::DEFAULT,
    )
    .unwrap();
    for i in 0..b.capacity_entries() {
        let e = TelEntry::new(
            NodeId::new(1000 + u64::from(i)),
            RelId::new(u64::from(i)),
            Lsn::new(u64::from(i) + 1),
        );
        b.append(e).unwrap();
    }
    b
}

fn bench_scan_full_block(c: &mut Criterion) {
    let block = filled_block(MAX_BLOCK_BYTES);
    let snap = Lsn::new(u64::MAX - 1);
    let mut group = c.benchmark_group("tel");
    group.throughput(Throughput::Elements(u64::from(block.capacity_entries())));
    group.bench_function("scan_full_block_2047_entries", |b| {
        b.iter(|| {
            let mut n = 0u64;
            for e in block.scan(black_box(snap)) {
                n = n.wrapping_add(e.rel_id);
            }
            black_box(n);
        });
    });
    group.finish();
}

fn bench_append_hot_path(c: &mut Criterion) {
    let mut group = c.benchmark_group("tel");
    group.throughput(Throughput::Elements(1));
    group.bench_function("append_single_entry_1024b_block", |b| {
        b.iter_batched(
            || TelBlock::new(NodeId::new(1), LabelId::new(1), 1024, TenantId::DEFAULT).unwrap(),
            |block| {
                let e = TelEntry::new(NodeId::new(42), RelId::new(1), Lsn::new(1));
                block.append(black_box(e)).unwrap();
                black_box(block);
            },
            criterion::BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_scan_full_block, bench_append_hot_path);
criterion_main!(benches);
