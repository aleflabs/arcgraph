//! Ingest hot-path bench for Property-Index Phase 0 (#1366) — the
//! RC-6 bench gate.
//!
//! Measures the **statement-txn ingest hot path** (the D-1/D-2 shape:
//! one statement = one `create_node` + `commit`) with **one declared
//! secondary index active**, so the per-commit cost of the Phase-0
//! changes is captured:
//!
//! - RC-1 insert-only maintenance: the create path is insert-only
//!   (no removals), so RC-1's queue is untouched here except for two
//!   early-return checks per commit (empty batch, empty queue).
//! - Z-1 F-1 rollback-drain: the secondary write path now threads a
//!   `TxnMutationLog` and captures pre-W bytes of every touched page +
//!   records fresh split/overflow pages. That extra per-insert
//!   byte-copy is the load this bench measures.
//!
//! Gate (RC-6): ingest p50 regression < 10% vs the pre-#1366 write
//! path. Run:
//!
//!     cargo bench -p arcgraph-index --bench ingest_declared_index
//!
//! Criterion reports p50 as the median of the sample; compare the
//! `median` line before/after (Criterion also prints a `change:` line
//! against the stored baseline in `target/criterion/`).

use std::sync::Arc;

use arcgraph_core::{LabelId, TenantId};
use arcgraph_index::SecondaryIndex;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::secondary_handle::SecondaryIndexHandle;
use arcgraph_storage::transaction::TxnManager;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

/// Build a dual-write store with ONE declared secondary index active
/// (mirrors production ingest with a single `CREATE INDEX`).
fn build_indexed_store() -> (Arc<TxnManager>, CrudStore) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let secondary =
        Arc::new(SecondaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None).unwrap());
    let handle: Arc<dyn SecondaryIndexHandle> = secondary as _;
    let store = CrudStore::new_with_indices(None, primary, Some(handle), alloc);
    (txn_mgr, store)
}

/// One statement-txn: create a node with two indexed inline properties
/// and commit. This is the D-1/D-2 UNWIND hot-path unit (one row → one
/// create → one commit).
#[inline]
fn ingest_one(store: &CrudStore, mgr: &TxnManager, seq: u32) {
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let _id = create_node(
        store,
        &mut tx,
        TenantId::DEFAULT,
        LabelId::new(1),
        // Distinct values per row so each insert touches the B-tree for
        // real (unique-ish email/name/app-id shape).
        &PropertyData::InlineU32Pair(seq, seq.wrapping_mul(2654435761)),
    )
    .unwrap();
    commit(tx, store).unwrap();
}

fn bench_ingest_one_statement_txn(c: &mut Criterion) {
    let mut group = c.benchmark_group("ingest_declared_index");
    group.throughput(Throughput::Elements(1));

    // Per-iteration: fresh store + one ingest. Using a per-iter fresh
    // store keeps each measured commit near the same tree size (a small
    // tree) so the number is the per-statement floor, not a
    // grows-with-N artifact. `SmallInput` amortizes setup.
    group.bench_function("create_commit_p50", |b| {
        let mut seq: u32 = 0;
        b.iter_batched(
            || {
                seq = seq.wrapping_add(1);
                (build_indexed_store(), seq)
            },
            |((mgr, store), s)| {
                ingest_one(&store, &mgr, s);
            },
            BatchSize::SmallInput,
        );
    });

    // Sustained: ingest into a growing tree (100 rows / measured batch)
    // to capture the realistic hot-path shape where the B-tree already
    // has entries and inserts may split.
    group.bench_function("sustained_100_rows", |b| {
        b.iter_batched(
            build_indexed_store,
            |(mgr, store)| {
                for s in 0..100u32 {
                    ingest_one(&store, &mgr, s);
                }
            },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_ingest_one_statement_txn);
criterion_main!(benches);
