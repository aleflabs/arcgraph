//! `CrudStoreGraphAdapter::materialize` cost bench (issue #248).
//!
//! Closes PR #244 round-2 reviewer NIT-2 + the ADR-040 amendment-05
//! §4.3 risk-register's "concrete benchmark pending" placeholder.
//!
//! # Why this bench
//!
//! The amendment-05 §4.3 cost line cited Sahu et al. 2024 §VI's 25
//! ms-per-tenant-per-tick number — which covers the GVE-Leiden RUN
//! only, NOT the `materialize(tenant)` pre-pass that builds the
//! `arcgraph_community::Graph` from the per-tenant `CrudStore`
//! snapshot. The materialise pre-pass adds ~2-3× to the per-tick
//! cost (per spot checks in PR #244 review), so at 1k-tenant
//! v1.1 deployments operators sizing against the published number
//! undercount actual capacity demand.
//!
//! This bench supplies the measured combined number so the
//! amendment can be amended-again with a concrete value (and so
//! future regressions surface against the Criterion baseline).
//!
//! # Workload shape
//!
//! Three sample points matching the issue acceptance:
//! - `n = 10K` nodes (small tenant)
//! - `n = 100K` nodes (mid tenant)
//! - `n = 1M` nodes (large tenant; Sahu §VI scale anchor)
//!
//! Each tenant is seeded with `n` nodes + avg fan-out 20 edges
//! per src (matching the KG-density assumption in §4.3). The
//! bench measures one full `materialize(tenant)` call against
//! the seeded fixture.
//!
//! # 1M sample is GA-gated
//!
//! Building a 1M-node × 20-fanout fixture costs ~2-3 minutes per
//! warm-up. Criterion's default 100-iter loop blows the bench
//! budget. The 1M sample is gated behind
//! `ARCGRAPH_BENCH_MATERIALIZE_FULL_GA=1` — the default sweep covers
//! 10K and 100K.
//!
//! Run: `cargo bench -p arcgraph-storage --bench materialize_bench`.
//! Reports: `target/criterion/materialize/`.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

use arcgraph_core::{LabelId, NodeId, TenantId, TypeId};
use arcgraph_storage::CrudStoreGraphAdapter;
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, create_rel};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

const FAN: u64 = 20;
const NODES_PER_TX: u64 = 1_000;
const EDGES_PER_TX: u64 = 1_000;
const SCALE_POINTS: &[u64] = &[10_000, 100_000];

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

fn build_tenant_fixture(n_nodes: u64) -> (Arc<TxnManager>, Arc<CrudStore>) {
    let txn_mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&txn_mgr), Arc::clone(&alloc), None)
            .expect("PrimaryIndex::new"),
    );
    let store = Arc::new(CrudStore::new_with_index(
        None,
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));

    // ───── Node phase ──────────────────────────────────────────────
    let phase = Instant::now();
    let mut node_ids: Vec<NodeId> = Vec::with_capacity(n_nodes as usize);
    let mut created = 0u64;
    while created < n_nodes {
        let this_batch = (n_nodes - created).min(NODES_PER_TX);
        let mut tx = txn_mgr.begin(TenantId::DEFAULT);
        for local in 0..this_batch {
            let global_i = created + local;
            let label = LabelId::new((global_i % 64) as u32);
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                label,
                &PropertyData::Empty,
            )
            .expect("create_node");
            node_ids.push(id);
        }
        commit(tx, &store).expect("seed commit nodes");
        created += this_batch;
    }
    eprintln!(
        "[materialize_bench] N={n_nodes}: node phase {:?}",
        phase.elapsed()
    );

    // ───── Edge phase ──────────────────────────────────────────────
    // Scope-isolated so `tx` is dropped (releasing its borrow of
    // `txn_mgr`) before we return the Arc tuple.
    {
        let phase = Instant::now();
        let mut rng: u64 = 0x1357_9BDF_0246_8ACE;
        let mut edges_in_tx = 0u64;
        let mut tx = Some(txn_mgr.begin(TenantId::DEFAULT));
        let n_us = n_nodes as usize;
        for src_idx in 0..n_us {
            let src = node_ids[src_idx];
            for h in 0..FAN {
                let stride = (xorshift64(&mut rng) as usize) % (n_us - 1);
                let dst_idx = (src_idx + stride + 1) % n_us;
                let dst = node_ids[dst_idx];
                let ty = TypeId::new((h as u32) % 16);
                let tref = tx.as_mut().expect("tx live");
                create_rel(
                    &store,
                    tref,
                    TenantId::DEFAULT,
                    src,
                    dst,
                    ty,
                    &PropertyData::Empty,
                )
                .expect("create_rel");
                edges_in_tx += 1;
                if edges_in_tx >= EDGES_PER_TX {
                    commit(tx.take().expect("tx live"), &store).expect("commit edges");
                    tx = Some(txn_mgr.begin(TenantId::DEFAULT));
                    edges_in_tx = 0;
                }
            }
        }
        if let Some(t) = tx.take() {
            if edges_in_tx > 0 {
                commit(t, &store).expect("commit edges final");
            } else {
                t.abort();
            }
        }
        eprintln!(
            "[materialize_bench] N={n_nodes}: edge phase {:?}",
            phase.elapsed()
        );
    }

    (txn_mgr, store)
}

fn bench_materialize(c: &mut Criterion) {
    let mut scale_points: Vec<u64> = SCALE_POINTS.to_vec();
    if std::env::var("ARCGRAPH_BENCH_MATERIALIZE_FULL_GA").is_ok() {
        scale_points.push(1_000_000);
    }

    let mut group = c.benchmark_group("materialize");
    for &n in &scale_points {
        let (txn_mgr, store) = build_tenant_fixture(n);
        let adapter = CrudStoreGraphAdapter::new(store, txn_mgr);

        let id = BenchmarkId::from_parameter(format!("N={n}"));
        group.bench_with_input(id, &n, |b, _| {
            b.iter(|| {
                let (graph, lsn) = adapter
                    .materialize(black_box(TenantId::DEFAULT))
                    .expect("materialize");
                black_box((graph, lsn));
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_materialize);
criterion_main!(benches);
