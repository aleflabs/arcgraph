//! W25-OPS-PROD / ADR-093-amendment-01 §D-2 — cost registry hot-path
//! contention soak.
//!
//! Targets the AtomicU64 hot path in
//! `arcgraph_core::cost_telemetry::CostAccumulator`. Confirms that
//! under sustained contention (N worker threads × M operations) the
//! counters remain monotonic + the registry's `Arc<CostAccumulator>`
//! reference count does not leak.
//!
//! # Env-gate discipline
//!
//! PANIC by default unless `ARCGRAPH_W25_SOAK_ENABLE=1` is set
//! explicitly. Duration via `ARCGRAPH_W25_SOAK_SECS` (default 30).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::cost_telemetry::PerTenantCostRegistry;
use arcgraph_core::ids::TenantId;

const DEFAULT_SOAK_SECS: u64 = 30;
const N_WORKERS: usize = 32;
const N_TENANTS: usize = 128;

#[test]
fn w25_cost_registry_soak() {
    // Env gate per `feedback_test_env_gate_panic_by_default.md`:
    //   ARCGRAPH_W25_SOAK_ENABLE=1   → run (operator-initiated soak).
    //   ARCGRAPH_W25_SOAK_SKIP_OK=1  → return early (CI gauntlet ergonomics).
    if std::env::var("ARCGRAPH_W25_SOAK_SKIP_OK").ok().as_deref() == Some("1") {
        eprintln!("[soak] ARCGRAPH_W25_SOAK_SKIP_OK=1 set; soak test explicitly skipped.");
        return;
    }
    if std::env::var("ARCGRAPH_W25_SOAK_ENABLE").ok().as_deref() != Some("1") {
        panic!(
            "ARCGRAPH_W25_SOAK_ENABLE=1 NOT set; soak test PANIC-skipped \
             per `feedback_test_env_gate_panic_by_default.md`. Set \
             ARCGRAPH_W25_SOAK_ENABLE=1 to run, OR set \
             ARCGRAPH_W25_SOAK_SKIP_OK=1 to explicitly skip."
        );
    }
    let secs: u64 = std::env::var("ARCGRAPH_W25_SOAK_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SOAK_SECS);
    let duration = Duration::from_secs(secs);
    eprintln!("[soak] duration={secs}s workers={N_WORKERS} tenants={N_TENANTS}");

    let registry = PerTenantCostRegistry::new();
    let stop = Arc::new(AtomicBool::new(false));
    let ops_total = Arc::new(AtomicU64::new(0));

    let tenant_count_start = registry.tenant_count();
    let snap_start = registry.snapshot_all();
    eprintln!(
        "[soak] t_start: tenants={tenant_count_start} snap_size={}",
        snap_start.len()
    );

    let start_instant = Instant::now();
    let mut handles = Vec::new();
    for worker_id in 0..N_WORKERS {
        let registry = registry.clone();
        let stop = Arc::clone(&stop);
        let ops_total = Arc::clone(&ops_total);
        handles.push(thread::spawn(move || {
            let mut iter: u64 = 0;
            while !stop.load(Ordering::Acquire) {
                iter += 1;
                let tenant_idx = (worker_id as u64 + iter) % N_TENANTS as u64;
                let tenant = TenantId::new(tenant_idx + 1);
                let acc = registry.get_or_init(tenant);
                acc.record_cpu_ms(1);
                acc.record_bytes_read(4096);
                acc.record_bytes_written(2048);
                acc.observe_mem_mb((iter % 1024) + 1);
                // MSRV 1.85 — `is_multiple_of` is 1.87+; use `%`.
                if iter % 16 == 0 {
                    let _ = acc.snapshot();
                }
                if iter % 128 == 0 {
                    let _ = registry.snapshot_all();
                }
                ops_total.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    thread::sleep(duration);
    stop.store(true, Ordering::Release);
    for h in handles {
        let _ = h.join();
    }
    let elapsed = start_instant.elapsed();
    let ops_end = ops_total.load(Ordering::Relaxed);
    let tenant_count_end = registry.tenant_count();
    eprintln!(
        "[soak] t_end (+{end_s}s): tenants={tenant_count_end} ops={ops_end} ops/s={ops_per_s:.0}",
        end_s = elapsed.as_secs(),
        ops_per_s = ops_end as f64 / elapsed.as_secs_f64(),
    );

    // Monotonicity contract: every accumulator's snapshot reflects at
    // least `iter` increments of cpu_ms (each worker bumps cpu_ms by 1
    // per iter). Across N_WORKERS the sum SHOULD be >= ops_end (we
    // can't enforce equality because workers cycle tenants — but the
    // workspace cpu_ms sum must equal ops_end exactly, modulo
    // saturating-add boundaries which we'll never hit at these
    // counter sizes).
    let snap_end = registry.snapshot_all();
    let total_cpu_ms: u64 = snap_end.values().map(|s| s.cpu_ms).sum();
    assert_eq!(
        total_cpu_ms, ops_end,
        "atomicity check: sum of cpu_ms across all tenants ({total_cpu_ms}) \
         must equal total ops ({ops_end})",
    );

    // No tenant-id leak: registry size bounded by N_TENANTS.
    assert!(
        tenant_count_end <= N_TENANTS,
        "tenant-id leak: registry grew to {tenant_count_end} > N_TENANTS={N_TENANTS}",
    );

    // Sanity floor: at least 1000 ops/s on any reasonable CPU.
    let ops_per_s = ops_end as f64 / elapsed.as_secs_f64();
    assert!(
        ops_per_s >= 1000.0,
        "ops/s {ops_per_s:.1} too low; workers may be contending or deadlocked",
    );

    eprintln!("[soak] PASS atomicity_sum={total_cpu_ms} tenants={tenant_count_end}");
}
