//! Bench: `CrudExecutorSubstrate::scan_nodes` perf budget at
//! v1.0-α small-N.
//!
//! Per R1 review MED-3 (PR #349) the v1.0-α `scan_nodes`
//! implementation is O(node_high_water) — every call iterates
//! `1..=high_water` and reads each id individually. This is
//! acceptable at small-N (the v1.0-α deployment target) but a
//! perf cliff at scale.
//!
//! This bench:
//!
//! 1. Pins the small-N perf budget (10K nodes) so a regression below
//!    the current cost is visible.
//! 2. Documents the perf curve so the v1.1 label-index swap (tracked
//!    as issue #351) has an objective acceptance bar (≥10× speedup
//!    on the same fixture).
//!
//! # Why this is a Criterion bench, not a unit test
//!
//! The metric we care about — wall-clock latency at small-N — is
//! exactly what Criterion measures with confidence intervals. A
//! `#[test]` with `Instant::now()` is noisy; Criterion's variance
//! tracking is the right shape for "v1.1 ≥ 10× the v1.0-α perf"
//! acceptance.
//!
//! # The v1.1 swap (issue #351)
//!
//! When the label-index path lands, this bench's `--baseline` should
//! show a step-function improvement at the same fixture. The bench
//! file stays; only the underlying substrate changes.

use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

/// Build a substrate populated with `n_nodes` nodes under the default
/// tenant. All nodes carry `LabelId::new(1)` so a label-filtered scan
/// returns all of them too.
fn build_populated_substrate(n_nodes: u64) -> CrudExecutorSubstrate {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());

    let label = arcgraph_core::LabelId::new(1);
    // Batch the inserts into chunks so the WAL doesn't blow up; chunk
    // size of 256 keeps each commit small while finishing 10K nodes
    // in <2s on Apple M-series.
    let chunk = 256u64;
    let mut inserted = 0u64;
    while inserted < n_nodes {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let end = (inserted + chunk).min(n_nodes);
        for _ in inserted..end {
            let _ = create_node(
                &crud,
                &mut tx,
                TenantId::DEFAULT,
                label,
                &PropertyData::Empty,
            )
            .expect("create_node");
        }
        commit(tx, &crud).expect("commit");
        inserted = end;
    }

    CrudExecutorSubstrate::new(router, mgr, intern)
}

fn bench_scan_nodes_small_n(c: &mut Criterion) {
    let n_nodes: u64 = 10_000;
    let sub = build_populated_substrate(n_nodes);
    c.bench_function("scan_nodes_10k_v1_0_alpha", |b| {
        b.iter(|| {
            let rows = sub
                .scan_nodes(black_box(TenantId::DEFAULT), None, Lsn::MAX)
                .expect("scan");
            black_box(rows);
        });
    });
}

criterion_group!(benches, bench_scan_nodes_small_n);
criterion_main!(benches);
