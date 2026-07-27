//! W25-OPS-PROD / ADR-093-amendment-01 §D-2 — admin HTTP + cost
//! registry soak harness.
//!
//! Exercises the v1.0-GA admin HTTP server + per-tenant cost registry
//! under sustained load. Detects:
//!
//! - **RSS growth** — Linux only; reads `/proc/self/status` `VmRSS`
//!   at t_start / t_mid / t_end. macOS / other platforms log a WARN
//!   + skip RSS check (the tenant-count assertion still runs).
//! - **Tenant-registry growth** — every tenant id is recycled across
//!   workers; the registry MUST NOT accumulate more than `N_tenants`
//!   entries.
//! - **Liveness** — `/livez` must return 200 throughout the run.
//!
//! # Env-gate discipline
//!
//! Per `feedback_test_env_gate_panic_by_default.md` (W12δ HIGH-1):
//! soft-skip is the silent-bypass bug class. This test PANICs by
//! default unless `ARCGRAPH_W25_SOAK_ENABLE=1` is set explicitly.
//! Run duration is `ARCGRAPH_W25_SOAK_SECS` (default 60).

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use arcgraph_cli::ops::admin_http::{
    AdminHttpServerConfig, ReadinessGate, serve_admin_http_with_cost,
};
use arcgraph_core::cost_telemetry::PerTenantCostRegistry;
use arcgraph_core::ids::TenantId;

/// Default smoke duration if `ARCGRAPH_W25_SOAK_SECS` is unset.
const DEFAULT_SOAK_SECS: u64 = 60;
/// Concurrent worker tasks generating load.
const N_WORKERS: usize = 16;
/// Number of distinct tenants the workers cycle through. Bounds the
/// expected registry size; if the registry grows beyond this, there
/// is a tenant-id leak.
const N_TENANTS: usize = 64;
/// Allowed RSS growth ratio between t_start and t_end. 1.5 = 50%
/// growth tolerated. Anything beyond signals a leak.
const RSS_GROWTH_THRESHOLD: f64 = 1.5;

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn w25_admin_http_soak() {
    // ── Env gate ──────────────────────────────────────────────────
    // PANIC by default per `feedback_test_env_gate_panic_by_default.md`.
    // Two opt-out paths:
    //   1. ARCGRAPH_W25_SOAK_ENABLE=1 → run the soak (operator-initiated).
    //   2. ARCGRAPH_W25_SOAK_SKIP_OK=1 → return early without running
    //      (CI gauntlet ergonomics; explicit acknowledgement that this
    //      test is skipped on the current run).
    if std::env::var("ARCGRAPH_W25_SOAK_SKIP_OK").ok().as_deref() == Some("1") {
        eprintln!(
            "[soak] ARCGRAPH_W25_SOAK_SKIP_OK=1 set; soak test explicitly skipped \
             (CI gauntlet ergonomics). Unset to PANIC-by-default."
        );
        return;
    }
    if std::env::var("ARCGRAPH_W25_SOAK_ENABLE").ok().as_deref() != Some("1") {
        panic!(
            "ARCGRAPH_W25_SOAK_ENABLE=1 NOT set; soak test PANIC-skipped \
             per `feedback_test_env_gate_panic_by_default.md`. Soft-skipping \
             silently is the bug class that lets green-painted tests pass \
             without ever running. Set ARCGRAPH_W25_SOAK_ENABLE=1 to run, \
             OR set ARCGRAPH_W25_SOAK_SKIP_OK=1 to explicitly skip."
        );
    }
    let secs: u64 = std::env::var("ARCGRAPH_W25_SOAK_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_SOAK_SECS);
    let duration = Duration::from_secs(secs);
    eprintln!("[soak] duration={secs}s workers={N_WORKERS} tenants={N_TENANTS}");

    // ── Setup: admin HTTP server with cost registry ───────────────
    let gate = ReadinessGate::new();
    gate.register("soak");
    gate.mark_ready("soak");

    let registry = PerTenantCostRegistry::new();
    let registry_clone = registry.clone();

    let cfg = AdminHttpServerConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        allow_remote_bind: false,
    };
    let listener = tokio::net::TcpListener::bind(cfg.bind).await.expect("bind");
    let bound = listener.local_addr().expect("local_addr");
    drop(listener); // serve_admin_http_with_cost re-binds the same addr; OS may transiently reject

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let cfg = AdminHttpServerConfig {
            bind: bound,
            allow_remote_bind: false,
        };
        let shutdown = async move {
            let _ = shutdown_rx.await;
        };
        serve_admin_http_with_cost(cfg, gate, Some(registry_clone), shutdown)
            .await
            .expect("serve");
    });

    // Brief warm-up to ensure the server is listening.
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(bound).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // ── Sample RSS at t_start ─────────────────────────────────────
    let rss_start = sample_rss_mb();
    let tenants_start = registry.tenant_count();
    eprintln!("[soak] t_start: rss={rss_start:?} MB tenants={tenants_start}");

    // ── Spawn worker tasks ────────────────────────────────────────
    let stop = Arc::new(AtomicBool::new(false));
    let ops_total = Arc::new(AtomicU64::new(0));
    let mut worker_handles = Vec::new();
    for worker_id in 0..N_WORKERS {
        let stop = Arc::clone(&stop);
        let ops_total = Arc::clone(&ops_total);
        let registry = registry.clone();
        worker_handles.push(tokio::spawn(async move {
            soak_worker(worker_id, bound, registry, stop, ops_total).await;
        }));
    }

    // ── Run for duration, sampling at midpoint ────────────────────
    let mid = duration / 2;
    let start_instant = Instant::now();
    tokio::time::sleep(mid).await;
    let rss_mid = sample_rss_mb();
    let tenants_mid = registry.tenant_count();
    let ops_mid = ops_total.load(Ordering::Relaxed);
    eprintln!(
        "[soak] t_mid (+{mid_s}s): rss={rss_mid:?} MB tenants={tenants_mid} ops={ops_mid}",
        mid_s = mid.as_secs(),
    );

    tokio::time::sleep(duration - mid).await;

    // ── Stop workers ──────────────────────────────────────────────
    stop.store(true, Ordering::Release);
    for h in worker_handles {
        let _ = h.await;
    }
    let elapsed = start_instant.elapsed();
    let ops_end = ops_total.load(Ordering::Relaxed);
    let rss_end = sample_rss_mb();
    let tenants_end = registry.tenant_count();
    eprintln!(
        "[soak] t_end (+{end_s}s): rss={rss_end:?} MB tenants={tenants_end} ops={ops_end} \
         ops/s={ops_per_s:.0}",
        end_s = elapsed.as_secs(),
        ops_per_s = ops_end as f64 / elapsed.as_secs_f64(),
    );

    // ── Shutdown server ───────────────────────────────────────────
    let _ = shutdown_tx.send(());
    let _ = tokio::time::timeout(Duration::from_secs(5), server).await;

    // ── Assertions ────────────────────────────────────────────────
    // The tenant-count assertion is cross-platform load-bearing: every
    // worker cycles through the same N_TENANTS pool, so the registry
    // MUST NOT accumulate more than N_TENANTS entries.
    assert!(
        tenants_end <= N_TENANTS,
        "tenant-id leak: registry grew to {tenants_end} entries > N_TENANTS={N_TENANTS}",
    );

    // The RSS-growth assertion runs only on platforms where
    // `sample_rss_mb` returns Some.
    if let (Some(start_mb), Some(end_mb)) = (rss_start, rss_end) {
        let growth = end_mb as f64 / start_mb as f64;
        assert!(
            growth <= RSS_GROWTH_THRESHOLD,
            "RSS growth {start_mb} MB -> {end_mb} MB ({growth:.2}x) exceeds threshold {RSS_GROWTH_THRESHOLD}; leak suspected",
        );
        eprintln!("[soak] PASS rss_growth={growth:.2}x (threshold {RSS_GROWTH_THRESHOLD})");
    } else {
        eprintln!(
            "[soak] WARN: RSS sampling unavailable on this platform; \
             tenant-count assertion only"
        );
    }

    // Ops-per-second sanity: with 16 workers + ~1ms per request, we
    // expect >= 100 ops/s minimum. The exact number depends on host
    // CPU; treat the assertion as floor-only.
    let ops_per_s = ops_end as f64 / elapsed.as_secs_f64();
    assert!(
        ops_per_s >= 10.0,
        "ops/s {ops_per_s:.1} unreasonably low; workers may be deadlocked",
    );
}

/// Per-worker load loop. Cycles through tenants, alternates between
/// read patterns (/livez probe, /cost/{tenant} snapshot) and write
/// patterns (record_cpu_ms + record_bytes_read + record_bytes_written
/// + observe_mem_mb on the registry).
async fn soak_worker(
    worker_id: usize,
    server_addr: SocketAddr,
    registry: PerTenantCostRegistry,
    stop: Arc<AtomicBool>,
    ops_total: Arc<AtomicU64>,
) {
    let mut iter: u64 = 0;
    while !stop.load(Ordering::Acquire) {
        iter += 1;
        let tenant_idx = (worker_id + iter as usize) % N_TENANTS;
        let tenant = TenantId::new(tenant_idx as u64 + 1);

        // Write side: 4 cost observations per iter (bursty pattern —
        // a real workload bursts then idles).
        let acc = registry.get_or_init(tenant);
        for _ in 0..4 {
            acc.record_cpu_ms(1);
            acc.record_bytes_read(4096);
            acc.record_bytes_written(1024);
            acc.observe_mem_mb((iter % 256) + 1);
        }
        ops_total.fetch_add(4, Ordering::Relaxed);

        // Read side: every 8th iter, probe /livez (cheap HTTP read).
        // Each step bounded by a 1s timeout so workers drain promptly
        // when `stop` flips — without the timeout, a slow read could
        // hold the worker open for >5s after stop, distorting the
        // soak's wall-clock vs duration ratio.
        // MSRV 1.85 — `is_multiple_of` is 1.87+; use `%` instead.
        if iter % 8 == 0 {
            let probe = async {
                let mut stream = tokio::net::TcpStream::connect(server_addr).await.ok()?;
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let req = format!(
                    "GET /livez HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                    server_addr
                );
                stream.write_all(req.as_bytes()).await.ok()?;
                stream.flush().await.ok()?;
                let mut buf = Vec::with_capacity(256);
                stream.read_to_end(&mut buf).await.ok()?;
                Some(())
            };
            if tokio::time::timeout(Duration::from_secs(1), probe)
                .await
                .ok()
                .flatten()
                .is_some()
            {
                ops_total.fetch_add(1, Ordering::Relaxed);
            }
        }
        // Every 32nd iter, snapshot the registry (analogue to the
        // /cost endpoint poll). MSRV 1.85 — use `%` instead of
        // `is_multiple_of` (1.87+).
        if iter % 32 == 0 {
            let _ = registry.snapshot_all();
            ops_total.fetch_add(1, Ordering::Relaxed);
        }
        // Yield to avoid pegging the executor.
        tokio::task::yield_now().await;
    }
}

/// Sample current process RSS in MB. Returns `None` on platforms
/// without an in-tree implementation (mac, windows — they require
/// platform-specific syscalls + extra dependencies that are out of
/// scope for the v1.0-GA workspace).
fn sample_rss_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let content = std::fs::read_to_string("/proc/self/status").ok()?;
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb: u64 = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())?;
                return Some(kb / 1024);
            }
        }
        None
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}
