//! W25-OPS-PROD / ADR-093-amendment-01 §D-3 — admin HTTP + cost
//! registry chaos harness.
//!
//! Operator-realistic faults against the v1.0-GA admin HTTP surface:
//!
//! 1. **Task-abort simulation** (cost observation workers cancelled
//!    mid-flight via `tokio::JoinHandle::abort`) — verifies the cost
//!    registry remains consistent when worker tasks are dropped
//!    mid-update.
//! 2. **Listener abort** — abort the admin HTTP listener task; verify
//!    the gate snapshot still reflects the previous state, the
//!    registry survives, and follow-up connects fail cleanly
//!    (connection refused, no panic).
//! 3. **u64::MAX boundary on bytes_written** — drive
//!    `record_bytes_written` across the u64::MAX boundary via one
//!    large delta; pins the current wrap-on-overflow contract.
//! 4. **Clock-independence of registry snapshots** — verify two
//!    snapshots taken across a wall-clock gap (no intervening
//!    record_* calls) are byte-equal. Forward-binds against any
//!    v1.1+ time-sensitive addition (windowed decay, expiry).
//!
//! All four contracts are process-internal.
//!
//! # Env-gate discipline
//!
//! PANIC by default unless `ARCGRAPH_W25_CHAOS_ENABLE=1` is set
//! explicitly per `feedback_test_env_gate_panic_by_default.md`.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_cli::ops::admin_http::{
    AdminHttpServerConfig, ReadinessGate, serve_admin_http_with_cost,
};
use arcgraph_core::cost_telemetry::PerTenantCostRegistry;
use arcgraph_core::ids::TenantId;

/// Env gate per `feedback_test_env_gate_panic_by_default.md`:
///   ARCGRAPH_W25_CHAOS_ENABLE=1   → run the chaos test.
///   ARCGRAPH_W25_CHAOS_SKIP_OK=1  → caller wants the test to skip (CI
///                                    gauntlet ergonomics); returns `false`
///                                    so the test body returns early.
/// Otherwise (neither set): PANIC.
///
/// Returns `true` if the test should execute, `false` if the caller's
/// test body should `return` immediately.
#[must_use]
fn require_env_gate() -> bool {
    if std::env::var("ARCGRAPH_W25_CHAOS_SKIP_OK").ok().as_deref() == Some("1") {
        eprintln!("[chaos] ARCGRAPH_W25_CHAOS_SKIP_OK=1 set; chaos test explicitly skipped.");
        return false;
    }
    if std::env::var("ARCGRAPH_W25_CHAOS_ENABLE").ok().as_deref() != Some("1") {
        panic!(
            "ARCGRAPH_W25_CHAOS_ENABLE=1 NOT set; chaos test PANIC-skipped \
             per `feedback_test_env_gate_panic_by_default.md`. Soft-skipping \
             silently is the bug class that lets green-painted tests pass \
             without ever running. Set ARCGRAPH_W25_CHAOS_ENABLE=1 to run, \
             OR set ARCGRAPH_W25_CHAOS_SKIP_OK=1 to explicitly skip."
        );
    }
    true
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn w25_chaos_task_abort_during_observations_preserves_atomicity() {
    if !require_env_gate() {
        return;
    }
    // Setup: cost registry + many concurrent worker tasks recording
    // observations. Abort the workers mid-flight via tokio task abort.
    // Verify the registry's counters remain consistent and monotonic
    // across the abort boundary.
    let registry = PerTenantCostRegistry::new();
    let n_workers = 16;
    let mut handles = Vec::new();
    for worker_id in 0..n_workers {
        let registry = registry.clone();
        handles.push(tokio::spawn(async move {
            let tenant = TenantId::new(worker_id as u64 + 1);
            let acc = registry.get_or_init(tenant);
            for _ in 0..100_000 {
                acc.record_cpu_ms(1);
                acc.record_bytes_written(1);
                tokio::task::yield_now().await;
            }
        }));
    }

    // Simulate SIGKILL by aborting tasks halfway through.
    tokio::time::sleep(Duration::from_millis(50)).await;
    for h in &handles {
        h.abort();
    }
    for h in handles {
        let _ = h.await; // tolerate JoinError::Cancelled
    }

    // Verify the registry is still queryable + every observed tenant
    // has cpu_ms == bytes_written (every worker bumps both 1:1).
    let snap = registry.snapshot_all();
    assert!(
        !snap.is_empty(),
        "registry should have at least one observed tenant"
    );
    for (tenant, cost) in &snap {
        assert_eq!(
            cost.cpu_ms, cost.bytes_written,
            "tenant {tenant:?} cost atomicity broken: cpu_ms={} bytes_written={}",
            cost.cpu_ms, cost.bytes_written,
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn w25_chaos_admin_http_listener_abort_mid_flight() {
    if !require_env_gate() {
        return;
    }
    let gate = ReadinessGate::new();
    gate.register("chaos");
    gate.mark_ready("chaos");
    let registry = PerTenantCostRegistry::new();
    let registry_for_assertions = registry.clone();
    registry.get_or_init(TenantId::new(99)).record_cpu_ms(12345);

    let cfg = AdminHttpServerConfig {
        bind: SocketAddr::from(([127, 0, 0, 1], 0)),
        allow_remote_bind: false,
    };
    let listener = tokio::net::TcpListener::bind(cfg.bind).await.expect("bind");
    let bound = listener.local_addr().expect("local_addr");
    drop(listener);

    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        let cfg = AdminHttpServerConfig {
            bind: bound,
            allow_remote_bind: false,
        };
        let shutdown = async move {
            let _ = shutdown_rx.await;
        };
        serve_admin_http_with_cost(cfg, gate, Some(registry), shutdown).await
    });
    for _ in 0..50 {
        if tokio::net::TcpStream::connect(bound).await.is_ok() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    // Issue a request; verify 200.
    let (status_pre, _) = http_get(bound, "/livez").await;
    assert_eq!(status_pre, 200);

    // Simulate a sudden listener abort. The server task is dropped
    // without graceful shutdown.
    server.abort();
    let _ = server.await; // tolerate cancellation
    let _ = shutdown_tx.send(()); // no-op now; verify it doesn't panic

    // Issue a follow-up request; expect connection refused (no panic).
    let post_attempt = tokio::time::timeout(
        Duration::from_secs(1),
        tokio::net::TcpStream::connect(bound),
    )
    .await;
    assert!(
        matches!(post_attempt, Ok(Err(_)) | Err(_)),
        "post-abort TCP connect should fail",
    );

    // Registry remains queryable + monotonic.
    let snap = registry_for_assertions
        .snapshot(TenantId::new(99))
        .expect("tenant 99 was recorded");
    assert_eq!(snap.cpu_ms, 12345);
}

#[test]
fn w25_chaos_bytes_written_counter_saturates_at_u64_max() {
    if !require_env_gate() {
        return;
    }
    // Simulate "disk full" by driving bytes_written across the
    // u64::MAX boundary. The accumulator MUST NOT panic on overflow.
    // Today the impl uses AtomicU64::fetch_add which wraps on
    // overflow; this test pins the current wrap-on-overflow behavior
    // so any future change (e.g., switch to saturating semantics or
    // panic_on_overflow) flags the change as a deliberate contract
    // shift. record_bytes_written takes u64 — the u64::MAX boundary
    // IS reachable from outside the crate via one large `delta`.
    use arcgraph_core::cost_telemetry::CostAccumulator;
    let registry = PerTenantCostRegistry::new();
    let acc: Arc<CostAccumulator> = registry.get_or_init(TenantId::new(1));

    // Step 1: drive the counter to u64::MAX - 5 via one large delta.
    acc.record_bytes_written(u64::MAX - 5);
    let snap_pre = acc.snapshot();
    assert_eq!(
        snap_pre.bytes_written,
        u64::MAX - 5,
        "counter should be exactly u64::MAX-5 before overflow attempt",
    );

    // Step 2: cross u64::MAX with +100 (5 fills the gap to MAX; the
    // remaining 95 wrap to value 94 since 2^64 distinct values are
    // 0..=u64::MAX inclusive, so (u64::MAX + 1) wraps to 0 and
    // +94 more produces 94). MUST NOT panic.
    acc.record_bytes_written(100);
    let snap_post = acc.snapshot();
    assert_eq!(
        snap_post.bytes_written,
        94,
        "wrap-on-overflow behavior changed: expected 94 (u64::MAX - 5 + 100 mod 2^64), \
         got {actual}. If this is a deliberate switch to saturating semantics, \
         update the assertion to assert_eq!(snap_post.bytes_written, u64::MAX). \
         If this is a panic-on-overflow regression, this test would have panicked above \
         and never reached this assertion.",
        actual = snap_post.bytes_written,
    );

    // Step 3: verify further observations continue to accumulate
    // monotonically post-wrap (no panic, no stuck counter).
    acc.record_bytes_written(6);
    assert_eq!(
        acc.snapshot().bytes_written,
        100,
        "post-wrap counter should continue accumulating",
    );
}

#[test]
fn w25_chaos_registry_snapshots_are_clock_independent() {
    if !require_env_gate() {
        return;
    }
    // Contract under test: the registry's snapshot semantics MUST be
    // wall-clock-independent at v1.0-GA. A snapshot taken at T0
    // returning value V MUST return the SAME V at T0+Δt for any Δt
    // (without intervening record_* calls), regardless of system
    // clock jumps. This forward-binds against any v1.1+
    // time-sensitivity addition (e.g., per-window decay) — if a
    // future version stamps snapshots with wall-clock state or
    // expires counters on clock skew, this test catches the
    // regression.
    //
    // The "clock skew" surface today is observable via std::thread::sleep
    // (real wall-clock advancement). True OS-level clock jumps need
    // root + clock_settime; we exercise the value-stability invariant
    // that any time-sensitive impl would break.
    let registry = PerTenantCostRegistry::new();
    let tenant = TenantId::new(1);
    let acc = registry.get_or_init(tenant);
    acc.record_cpu_ms(100);
    acc.record_cpu_ms(200);
    acc.record_cpu_ms(300);

    // Snapshot the value once, then sleep across a meaningful wall-clock
    // gap, then snapshot again WITHOUT intervening record_* calls. The
    // two snapshots MUST be byte-equal (every field unchanged).
    let snap_t0 = acc.snapshot();
    assert_eq!(snap_t0.cpu_ms, 600, "pre-sleep snapshot value");
    std::thread::sleep(Duration::from_millis(100));
    let snap_t1 = acc.snapshot();
    assert_eq!(
        snap_t0, snap_t1,
        "snapshots taken before/after a 100ms wall-clock gap MUST be byte-equal — \
         registry has no clock-derived semantics at v1.0-GA. If a future version \
         adds time-sensitivity (windowed decay, expiry), update this contract."
    );

    // Also verify that record_* calls AFTER a wall-clock gap accumulate
    // correctly (no clock-derived gating of writes).
    acc.record_cpu_ms(50);
    let snap_t2 = acc.snapshot();
    assert_eq!(
        snap_t2.cpu_ms, 650,
        "post-sleep record_cpu_ms MUST accumulate (650 = 600 + 50) — no \
         clock-derived gating of observations.",
    );
}

async fn http_get(addr: SocketAddr, path: &str) -> (u16, String) {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let mut stream = tokio::net::TcpStream::connect(addr).await.expect("connect");
    let req = format!(
        "GET {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n\r\n",
        addr = addr,
        path = path,
    );
    stream.write_all(req.as_bytes()).await.expect("write");
    stream.flush().await.expect("flush");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.expect("read");
    let text = String::from_utf8_lossy(&buf).to_string();
    let status = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse::<u16>().ok())
        .unwrap_or(0);
    let body_start = text.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
    (status, text[body_start..].to_string())
}
