#![cfg(unix)]

//! M6.4 / #1532 — caller-visible eviction-flush convoy characterization.
//!
//! `WriteBehindCheckpointer::flush_priority_keys` is the priority-flush
//! handshake used by M6.1 eviction. The #1528 durability fix deliberately
//! serializes that complete handshake — admission, copy, real home fsync, and
//! generation-matched DPT completion — through one per-checkpointer mutex.
//! This release-only characterization measures the resulting caller-visible
//! latency with a one-worker control and under synchronized eight-way pressure.
//!
//! Run with:
//! `cargo test -p arcgraph-storage --release --features fault-injection \
//!    --test m6_convoy_latency_characterization -- --ignored --nocapture`

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, Result, TenantId};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::io::{PageBuf, PageIo, PosixPageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::STORE_RECORD;

const CONTROL_WORKERS: usize = 1;
const PRESSURE_WORKERS: usize = 8;
const WARMUP_ROUNDS: usize = 8;
const MEASURED_ROUNDS: usize = 64;

/// Measurement-derived #1532 bound (2026-07-18, 12-core Apple M3 Pro,
/// APFS `/tmp`, release profile). The first calibration run measured an
/// eight-way caller-visible p99 of 37,035 us; `ceil(37,035 * 1.30)` is
/// 48,146 us. A second run measured 32,987 us and remains below this bar.
const MEASURED_P99_BOUND_US: u64 = 48_146;

#[derive(Debug, Clone, Copy)]
struct Sample {
    start_ns: u128,
    end_ns: u128,
    latency_ns: u64,
}

#[derive(Debug)]
struct PressureReport {
    workers: usize,
    samples: usize,
    p50_us: u64,
    p99_us: u64,
    rps: f64,
    home_calls: u64,
    home_pages: u64,
}

/// Delegates to the real disk-backed record store while making vacuous runs
/// impossible: every successful priority-flush sample must cross this home
/// write surface with exactly one page.
struct CountingTarget {
    inner: Arc<BufferedRecordPageStore>,
    home_calls: AtomicU64,
    home_pages: AtomicU64,
}

impl CountingTarget {
    fn new(inner: Arc<BufferedRecordPageStore>) -> Self {
        Self {
            inner,
            home_calls: AtomicU64::new(0),
            home_pages: AtomicU64::new(0),
        }
    }
}

impl PageFlushTarget for CountingTarget {
    fn copy_page_pinned(&self, tenant: TenantId, page_id: PageId) -> Result<Option<Box<PageBuf>>> {
        self.inner
            .copy_page_pinned_for_tenant(tenant, page_id)
            .map_err(|error| {
                arcgraph_core::ArcGraphError::Io(std::io::Error::other(error.to_string()))
            })
    }

    fn write_pages_home(&self, images: &[(TenantId, PageId, Box<PageBuf>)]) -> Result<()> {
        let result = self.inner.write_pages_home_qualified(images);
        if result.is_ok() {
            self.home_calls.fetch_add(1, Ordering::Relaxed);
            self.home_pages.fetch_add(
                u64::try_from(images.len()).expect("flush page count fits u64"),
                Ordering::Relaxed,
            );
        }
        result
    }
}

fn percentile_ns(samples: &mut [u64], percentile: f64) -> u64 {
    assert!(
        !samples.is_empty(),
        "latency distribution must not be empty"
    );
    samples.sort_unstable();
    let index = ((samples.len() as f64 - 1.0) * percentile).round() as usize;
    samples[index]
}

fn ns_to_us_ceil(nanoseconds: u64) -> u64 {
    nanoseconds.saturating_add(999) / 1_000
}

fn run_pressure(workers: usize) -> PressureReport {
    assert!(workers > 0);

    let dir = tempfile::tempdir().expect("create convoy scratch directory");
    let io: Arc<dyn PageIo> = Arc::new(
        PosixPageIo::create(dir.path().join("record.store")).expect("create POSIX record home"),
    );
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: workers * 2 + 1,
            write_fraction: 0.0,
        },
    ));
    let store = Arc::new(BufferedRecordPageStore::with_cache_cap(
        pools,
        workers * 2 + 1,
    ));

    let dpt = Arc::new(DirtyPageTable::new());
    let target = Arc::new(CountingTarget::new(Arc::clone(&store)));
    let props_target: Arc<dyn PageFlushTarget> = store.clone();
    let records_target: Arc<dyn PageFlushTarget> = target.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::with_batch_pages(
        Arc::clone(&dpt),
        props_target,
        records_target,
        1,
    ));

    let mut keys = Vec::with_capacity(workers);
    for worker in 0..workers {
        let page_id = PageId::new(u64::try_from(worker + 1).expect("worker page id fits u64"));
        store
            .install_fresh(page_id, PageType::Node, TenantId::DEFAULT)
            .expect("install worker page");
        keys.push(DirtyPageKey {
            tenant_id: TenantId::DEFAULT,
            store_id: STORE_RECORD,
            page_no: page_id.raw(),
        });
    }

    let round_barrier = Arc::new(Barrier::new(workers));
    let epoch = Instant::now();
    let mut handles = Vec::with_capacity(workers);
    for (worker, key) in keys.into_iter().enumerate() {
        let worker_store = Arc::clone(&store);
        let worker_dpt = Arc::clone(&dpt);
        let worker_checkpointer = Arc::clone(&checkpointer);
        let worker_barrier = Arc::clone(&round_barrier);
        handles.push(thread::spawn(move || {
            let mut samples = Vec::with_capacity(MEASURED_ROUNDS);
            for round in 0..(WARMUP_ROUNDS + MEASURED_ROUNDS) {
                // Each worker owns a distinct page, so the only intentional
                // shared bottleneck is the checkpointer's admission mutex.
                {
                    let pinned = worker_store
                        .latch_pinned_for_tenant(TenantId::DEFAULT, PageId::new(key.page_no))
                        .expect("pin worker page");
                    let mut guard = pinned.latch().write();
                    guard.as_mut()[PAGE_SIZE - 1] =
                        u8::try_from((worker + round) % (usize::from(u8::MAX) + 1))
                            .expect("marker is bounded to u8");
                }
                worker_dpt.mark_dirty(
                    key,
                    Lsn::new(u64::try_from(round + 1).expect("round LSN fits u64")),
                );

                // Release every caller into the priority-flush method as one
                // pressure wave, then wait for the complete wave before any
                // worker re-dirties its page for the next round.
                worker_barrier.wait();
                let start_ns = epoch.elapsed().as_nanos();
                let started = Instant::now();
                let completed = worker_checkpointer
                    .flush_priority_keys(&[key])
                    .expect("priority eviction flush");
                let latency_ns = u64::try_from(started.elapsed().as_nanos())
                    .expect("single flush latency fits u64 nanoseconds");
                let end_ns = epoch.elapsed().as_nanos();
                assert!(
                    completed.contains(&key),
                    "every marked key must complete a real generation-matched flush"
                );
                worker_barrier.wait();

                if round >= WARMUP_ROUNDS {
                    samples.push(Sample {
                        start_ns,
                        end_ns,
                        latency_ns,
                    });
                }
            }
            samples
        }));
    }

    let mut samples = Vec::with_capacity(workers * MEASURED_ROUNDS);
    for handle in handles {
        samples.extend(handle.join().expect("convoy worker panicked"));
    }

    let expected_home_ops = u64::try_from(workers * (WARMUP_ROUNDS + MEASURED_ROUNDS))
        .expect("expected home operation count fits u64");
    let home_calls = target.home_calls.load(Ordering::Relaxed);
    let home_pages = target.home_pages.load(Ordering::Relaxed);
    assert_eq!(
        home_calls, expected_home_ops,
        "vacuity guard: every warmup and measured call must reach real home I/O"
    );
    assert_eq!(
        home_pages, expected_home_ops,
        "vacuity guard: every priority flush must write exactly one page"
    );
    assert!(
        dpt.is_empty(),
        "all generation-matched priority flushes must drain the DPT"
    );
    assert_eq!(
        samples.len(),
        workers * MEASURED_ROUNDS,
        "every worker must contribute every measured round"
    );

    let first_start = samples
        .iter()
        .map(|sample| sample.start_ns)
        .min()
        .expect("at least one sample");
    let last_end = samples
        .iter()
        .map(|sample| sample.end_ns)
        .max()
        .expect("at least one sample");
    let measured_window_ns = last_end
        .checked_sub(first_start)
        .expect("measurement timestamps are monotonic")
        .max(1);
    let rps = samples.len() as f64 * 1_000_000_000.0 / measured_window_ns as f64;

    let mut latencies: Vec<u64> = samples.iter().map(|sample| sample.latency_ns).collect();
    let p50_ns = percentile_ns(&mut latencies.clone(), 0.50);
    let p99_ns = percentile_ns(&mut latencies, 0.99);

    PressureReport {
        workers,
        samples: samples.len(),
        p50_us: ns_to_us_ceil(p50_ns),
        p99_us: ns_to_us_ceil(p99_ns),
        rps,
        home_calls,
        home_pages,
    }
}

#[test]
#[ignore = "release-only real-fsync performance characterization; run explicitly with --ignored"]
fn m6_convoy_latency_characterization() {
    let control = run_pressure(CONTROL_WORKERS);
    let pressure = run_pressure(PRESSURE_WORKERS);

    println!(
        "m6_convoy_latency_characterization \
         workers={} samples={} p50_us={} p99_us={} rps={:.2} \
         home_calls={} home_pages={}",
        control.workers,
        control.samples,
        control.p50_us,
        control.p99_us,
        control.rps,
        control.home_calls,
        control.home_pages,
    );
    println!(
        "m6_convoy_latency_characterization \
         workers={} samples={} p50_us={} p99_us={} rps={:.2} \
         home_calls={} home_pages={}",
        pressure.workers,
        pressure.samples,
        pressure.p50_us,
        pressure.p99_us,
        pressure.rps,
        pressure.home_calls,
        pressure.home_pages,
    );

    if let Some(measured_bound_us) =
        std::num::NonZeroU64::new(MEASURED_P99_BOUND_US).map(std::num::NonZeroU64::get)
    {
        assert!(
            pressure.p99_us <= measured_bound_us,
            "#1532 eight-way priority eviction-flush p99 {} us exceeds the \
             measured-plus-margin bound {} us (1-way control p99 {} us)",
            pressure.p99_us,
            measured_bound_us,
            control.p99_us,
        );
    } else {
        println!(
            "m6_convoy_latency_characterization calibration=report-only \
             measured_8way_p99_us={} bound_us=unset",
            pressure.p99_us
        );
    }
}
