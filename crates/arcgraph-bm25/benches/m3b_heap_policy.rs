//! M3.b-heap-policy bench (ADR-039 amendment-01 §D-11(a)+(b)+(c) /
//! amendment-02 §D-12 + §D-14 + §D-15).
//!
//! Closes OPEN-Q-2 (per-tenant `parking_lot::Mutex` contention
//! measurement) per ADR-039 amendment-01 §D-10 item 3 + amendment-02
//! §D-12 / §D-15.
//!
//! ## Shape
//!
//! Three Criterion groups, all running against `Bm25Service` with
//! the v1.0 default heap (`16 MiB` per amendment-01 §D-11(a)) and
//! various pool sizes:
//!
//! - `m3b_heap_policy/cold_start` — first-touch
//!   `Bm25Service::handle` + `upsert_document` allocation latency.
//!   The handle materialises the per-tenant directory and the pool
//!   admits a fresh permit.
//! - `m3b_heap_policy/warm_cache` — repeated `upsert_document` on
//!   an already-allocated writer. Measures the steady-state hot-
//!   path; this is the floor for per-write cost.
//! - `m3b_heap_policy/evicted_rewrite` — write, drive eviction,
//!   write again. Measures the eviction-recreate cycle's overhead
//!   over `cold_start` (re-opens an existing Tantivy directory
//!   rather than creating one).
//! - `m3b_heap_policy/pool_exhaustion` — pool size = 4 with 8
//!   concurrent commit threads. Measures the tail latency of
//!   permit acquisition under saturation; the on-full sweeper +
//!   commit-tail evict-idle sweep are the only mechanisms that
//!   keep this from deadlocking.
//!
//! ## Hardware context
//!
//! Records hardware via `eprintln!` at first dereference of the
//! corpus (Apple Silicon laptop / release build / 2026-05-03
//! spawn — record per-run actual). The numbers feed
//! ADR-039 amendment-02 §D-12 verbatim.
//!
//! ## Running
//!
//! ```bash
//! cargo bench -p arcgraph-bm25 --bench m3b_heap_policy
//! ```
//!
//! Wall-clock: ~30 s per group at default sample size; pool-
//! exhaustion bench is the longest (~60 s) due to the cross-thread
//! coordination.

use std::hint::black_box;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_bm25::{Bm25Service, IndexId};
use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use criterion::{Criterion, criterion_group, criterion_main};
use tempfile::TempDir;

// ---------------------------------------------------------------------------
// Per-bench-process tempdir
// ---------------------------------------------------------------------------

/// Build a fresh service rooted at a unique tempdir, with the given
/// pool size. The TempDir is leaked into the Criterion-process
/// scope via `Box::leak` so the bench doesn't have to thread the
/// guard through every closure (Criterion measures only the closure
/// body; ambient cleanup at process exit is sufficient for the
/// bench setting).
fn fresh_service(pool_size: usize) -> (PathBuf, Arc<Bm25Service>) {
    let tmp = Box::leak(Box::new(TempDir::new().expect("tempdir")));
    let path = tmp.path().to_path_buf();
    let svc = Bm25Service::with_pool_size(path.clone(), pool_size);
    (path, svc)
}

const SHORT_DOC: &str = "alpha quick brown fox jumps over the lazy dog. \
Lorem ipsum dolor sit amet, consectetur adipiscing elit. The quick brown \
fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. \
Sphinx of black quartz, judge my vow. The five boxing wizards jump quickly.";

/// Group 1 — cold-start: fresh tenant, first upsert. Measures the
/// `handle()` + `Index::open_or_create` + `Index::writer` +
/// `add_document` cost end-to-end.
///
/// Each iteration uses a brand-new tenant id so we don't measure
/// the warm-cache shape.
fn bench_cold_start(c: &mut Criterion) {
    eprintln!(
        "[m3b_heap_policy bench] starting (Apple Silicon laptop / release / \
         2026-05-03 spawn — record actual host on review packet)"
    );
    let (_path, svc) = fresh_service(arcgraph_bm25::WRITER_POOL_SIZE);
    let counter = AtomicU64::new(0);
    let mut group = c.benchmark_group("m3b_heap_policy/cold_start");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("first_upsert_per_tenant", |b| {
        b.iter(|| {
            let tid = counter.fetch_add(1, Ordering::SeqCst);
            let tenant = TenantId::new(100_000 + tid);
            let h = svc.handle(tenant, IndexId::DEFAULT_BM25).expect("handle");
            h.upsert_document(NodeId::new(1), black_box(SHORT_DOC), Lsn::new(1))
                .expect("upsert");
            black_box(h);
        });
    });
    group.finish();
}

/// Group 2 — warm-cache hot path: repeated upsert against an
/// already-allocated writer. Measures the steady-state floor.
fn bench_warm_cache(c: &mut Criterion) {
    let (_path, svc) = fresh_service(arcgraph_bm25::WRITER_POOL_SIZE);
    let h = svc
        .handle(TenantId::new(99_999), IndexId::DEFAULT_BM25)
        .expect("handle");
    // Prime: one upsert to allocate the writer.
    h.upsert_document(NodeId::new(0), SHORT_DOC, Lsn::new(0))
        .expect("prime upsert");
    let counter = AtomicU64::new(1);
    let mut group = c.benchmark_group("m3b_heap_policy/warm_cache");
    group.sample_size(50);
    group.measurement_time(Duration::from_secs(10));
    group.bench_function("repeat_upsert_warm", |b| {
        b.iter(|| {
            let n = counter.fetch_add(1, Ordering::SeqCst);
            h.upsert_document(NodeId::new(n + 1), black_box(SHORT_DOC), Lsn::new(n + 1))
                .expect("upsert");
        });
    });
    group.finish();
}

/// Group 3 — evicted-rewrite: per request-scoped semantics
/// (amendment-02 §D-14), commit drops the writer; the next write
/// re-allocates against the existing on-disk Tantivy directory.
/// Measures the cost of that `Index::writer` re-allocation.
fn bench_evicted_rewrite(c: &mut Criterion) {
    let (_path, svc) = fresh_service(arcgraph_bm25::WRITER_POOL_SIZE);
    let counter = AtomicU64::new(0);
    let mut group = c.benchmark_group("m3b_heap_policy/evicted_rewrite");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    group.bench_function("post_commit_first_upsert", |b| {
        b.iter_custom(|iters| {
            // Per-iteration setup: fresh tenant, write + commit
            // (request-scoped commit drops writer). Measure ONLY
            // the next upsert (which reallocates the writer).
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let tid = counter.fetch_add(1, Ordering::SeqCst);
                let tenant = TenantId::new(200_000 + tid);
                let h = svc.handle(tenant, IndexId::DEFAULT_BM25).expect("handle");
                h.upsert_document(NodeId::new(1), SHORT_DOC, Lsn::new(1))
                    .expect("seed upsert");
                h.commit().expect("first commit drops writer");
                debug_assert!(!h.has_active_writer());

                // Measured region: next upsert after the commit
                // dropped the writer.
                let start = Instant::now();
                h.upsert_document(NodeId::new(2), black_box(SHORT_DOC), Lsn::new(2))
                    .expect("post-commit reallocate upsert");
                total += start.elapsed();
            }
            total
        });
    });
    group.finish();
}

/// Group 4 — pool exhaustion: 4-permit pool with 8 concurrent
/// commit threads. Measures the tail latency on permit acquisition
/// under sustained saturation.
///
/// Each thread loops: upsert + commit_pending. The post-commit
/// evict_idle sweep + the on-full sweeper closure together keep
/// the system live; without them this would deadlock.
fn bench_pool_exhaustion(c: &mut Criterion) {
    let mut group = c.benchmark_group("m3b_heap_policy/pool_exhaustion");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));
    group.bench_function("pool4_with_8_threads", |b| {
        b.iter_custom(|iters| {
            let mut total = Duration::ZERO;
            for _ in 0..iters {
                let (_path, svc) = fresh_service(4);
                let start = Instant::now();
                let handles: Vec<_> = (0..8u64)
                    .map(|i| {
                        let svc = Arc::clone(&svc);
                        thread::spawn(move || {
                            let tenant = TenantId::new(900_000 + i);
                            let h = svc.handle(tenant, IndexId::DEFAULT_BM25).expect("handle");
                            let trait_obj: Arc<dyn Bm25IndexStoreHandle> = Arc::clone(&svc) as _;
                            // 4 docs per thread; the second-onwards
                            // commit will (transitively) cross the
                            // commit-axis idle threshold for some
                            // tenants and the sweeper will keep the
                            // pool live.
                            for j in 0..4u64 {
                                h.upsert_document(
                                    NodeId::new(i * 10 + j + 1),
                                    SHORT_DOC,
                                    Lsn::new(j + 1),
                                )
                                .expect("upsert");
                                // commit_pending under request-
                                // scoped semantics drops the
                                // writer + returns the permit at
                                // the natural cadence; no extra
                                // sweeper-driver work needed.
                                trait_obj.commit_pending(tenant).expect("commit_pending");
                            }
                        })
                    })
                    .collect();
                for h in handles {
                    h.join().expect("thread joined cleanly");
                }
                total += start.elapsed();
            }
            total
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_cold_start,
    bench_warm_cache,
    bench_evicted_rewrite,
    bench_pool_exhaustion,
);
criterion_main!(benches);
