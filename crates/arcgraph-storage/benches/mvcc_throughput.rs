//! MVCC microbenchmarks (M2.b).
//!
//! These benches exercise the MVCC kernel in isolation from WAL
//! (M2.c wires WAL appends into `commit`). Numbers here establish
//! the in-memory ceiling; the end-to-end M2.e exit criterion
//! (5 K TPS write throughput) is measured separately against a
//! version of `commit` that enqueues a WAL record.
//!
//! Run with:
//!
//!     cargo bench -p arcgraph-storage --bench mvcc_throughput
//!
//! Results are captured in `docs/benchmarks/M2-mvcc.md`.

use arcgraph_core::TenantId;
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

fn bench_lsn_allocate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc/lsn");
    group.throughput(Throughput::Elements(1));
    group.bench_function("allocate_single_thread", |b| {
        let m = TxnManager::new();
        b.iter(|| {
            let _ = m.current_lsn();
        });
    });
    group.finish();
}

fn bench_single_writer_commit(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc/commit");
    group.throughput(Throughput::Elements(1));
    group.bench_function("single_writer_one_key", |b| {
        let m = TxnManager::new();
        let mut i = 0u64;
        b.iter(|| {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(i & 0xFFFF, Bytes::from_static(b"payload"));
            t.commit().unwrap();
            i += 1;
        });
    });

    group.bench_function("single_writer_eight_keys", |b| {
        let m = TxnManager::new();
        let mut i = 0u64;
        b.iter(|| {
            let mut t = m.begin(TenantId::DEFAULT);
            for j in 0..8u64 {
                t.write((i + j) & 0xFFFF, Bytes::from_static(b"payload"));
            }
            t.commit().unwrap();
            i += 8;
        });
    });
    group.finish();
}

fn bench_read_snapshot(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc/read");
    // Seed 4096 keys with one version each.
    let m = TxnManager::new();
    for k in 0..4096u64 {
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(k, Bytes::from_static(b"v"));
        t.commit().unwrap();
    }
    group.throughput(Throughput::Elements(1));
    group.bench_function("read_live_key_warm", |b| {
        let reader = m.begin(TenantId::DEFAULT);
        let mut i = 0u64;
        b.iter(|| {
            let _ = reader.read(i & 0xFFF);
            i += 1;
        });
    });
    group.finish();
}

fn bench_ww_conflict_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc/ww_conflict");
    group.throughput(Throughput::Elements(2));
    group.bench_function("two_writers_same_key", |b| {
        b.iter_batched(
            TxnManager::new,
            |m| {
                let mut a = m.begin(TenantId::DEFAULT);
                let mut b = m.begin(TenantId::DEFAULT);
                a.write(1, Bytes::from_static(b"a"));
                b.write(1, Bytes::from_static(b"b"));
                let _ = a.commit();
                let _ = b.commit();
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn bench_gc(c: &mut Criterion) {
    let mut group = c.benchmark_group("mvcc/gc");
    group.bench_function("gc_over_1k_keys_5_versions_each", |b| {
        b.iter_batched(
            || {
                let m = TxnManager::new();
                for k in 0..1024u64 {
                    for v in 0..5u8 {
                        let mut t = m.begin(TenantId::DEFAULT);
                        t.write(k, Bytes::copy_from_slice(&[v]));
                        t.commit().unwrap();
                    }
                }
                m
            },
            |m| {
                let _ = m.gc();
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_lsn_allocate,
    bench_single_writer_commit,
    bench_read_snapshot,
    bench_ww_conflict_rate,
    bench_gc,
);
criterion_main!(benches);
