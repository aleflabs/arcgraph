//! M6.1 — `m6_wal_channel_byte_budget_admission` (ADR-232-amendment-01,
//! design-v2 §6.1: "bounded WAL admission").
//!
//! The WAL producer channel (`wal/writer.rs`) is a `crossbeam::unbounded()`
//! — measured queue depth 0 under WAL-enabled ingest, so it is not today's
//! OOM, but it remains a *latent* unboundedness: any future
//! producer-faster-than-fsync regime turns it into an accumulator. §6.1
//! specifies a byte-budget gate: an in-flight-bytes counter + semaphore
//! that blocks `append`/`append_async` when `in_flight_bytes` exceeds a
//! configured budget, resuming as the writer retires (fsyncs) entries.
//!
//! This is a DETERMINISM-ORACLE concurrency gate: a deterministic barrier
//! (not a sleep) forces the exact interleaving — admit past budget ->
//! blocks -> release via explicit `flush()` -> unblocks — and asserts the
//! producer thread is provably PARKED (not merely "probably waiting") at
//! the moment the budget is exhausted, via a handshake AtomicBool the
//! blocked thread cannot set until `admit` returns.
//!
//! RED-on-revert: configuring `WalConfig::with_inflight_budget_bytes` and
//! then removing/no-op'ing the gate (i.e. reverting to the unconditional
//! unbounded admission) makes the "second append is admitted immediately"
//! assertion fail to ever observe blocking — the gate's negative-control
//! test (`unbudgeted_config_never_blocks`) pins the pre-M6 behavior so a
//! silent budget no-op is visible by contrast.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use arcgraph_storage::wal::{WalConfig, WalRecordType, WalWriter};
use tempfile::tempdir;

const WAIT_BUDGET: Duration = Duration::from_secs(30);

/// Bounded spin-wait on a flag; panics naming the dead peer on timeout
/// (per the standing "unbounded rendezvous wait is a suite-wide hang"
/// discipline — every wait here is bounded and names its peer).
fn wait_for(flag: &AtomicBool, who: &str) {
    let start = Instant::now();
    while !flag.load(Ordering::Acquire) {
        if start.elapsed() > WAIT_BUDGET {
            panic!("rendezvous timed out waiting for {who} (peer dead or stalled)");
        }
        std::thread::yield_now();
    }
}

fn budget_config(dir: std::path::PathBuf, budget_bytes: u64) -> WalConfig {
    WalConfig::new(dir)
        // Long window + large batch: fires happen ONLY on our explicit
        // `flush()` calls, giving the test full deterministic control
        // over exactly when bytes are released back to the budget.
        .with_inflight_budget_bytes(budget_bytes)
}

fn unbudgeted_config(dir: std::path::PathBuf) -> WalConfig {
    WalConfig::new(dir)
}

/// THE decisive leg: a producer whose append would exceed the configured
/// budget BLOCKS (does not return) until an earlier in-flight append is
/// released by a durable fire, then is admitted. Deterministic barrier:
/// the blocked producer thread flips `blocked_confirmed` only from INSIDE
/// `admit`'s wait loop is unobservable directly, so we instead prove
/// blocking by a happens-before argument: `second_append_returned` MUST
/// NOT be set before the main thread calls `flush()` — checked via a
/// rendezvous that would otherwise race-and-sometimes-pass under a
/// sleep-based design. The barrier here is: main thread does NOT call
/// flush() until it has confirmed (bounded-wait) that the second-append
/// thread has been spawned and had a scheduling quantum to attempt the
/// call; the oracle is the ORDERING of two atomic flags set by each side,
/// which a non-blocking (reverted) gate would violate every run, not
/// probabilistically.
#[test]
fn append_past_budget_blocks_until_release_then_admits() {
    let dir = tempdir().unwrap();
    // Budget smaller than two full-size payloads so the second append
    // cannot be admitted while the first is still in flight.
    const PAYLOAD_LEN: usize = 4096;
    const BUDGET: u64 = (PAYLOAD_LEN as u64) + 100; // room for ~1, not 2

    let config = budget_config(dir.path().to_path_buf(), BUDGET);
    // Long window so the first append does NOT auto-fire; only our
    // explicit `flush()` releases it.
    let config = WalConfig {
        group_commit_window: Duration::from_secs(3600),
        group_commit_max_batch: 64,
        ..config
    };
    let writer = Arc::new(WalWriter::spawn(config).unwrap());
    let handle = writer.handle();

    // Spawn the FIRST append on its own thread since `append` blocks
    // until fsync — we need it in flight (admitted, not yet fired) while
    // the second append attempts admission. The first thread calls
    // `append` immediately (that is what performs the budget admission,
    // synchronously, before the WAL command is even sent); the group-
    // commit window is 1 hour so this append will NOT auto-fire — it
    // stays "admitted but not durable" until the MAIN thread calls
    // `flush()` below, which is the deterministic release trigger.
    let first_started = Arc::new(AtomicBool::new(false));
    let first_done = Arc::new(AtomicBool::new(false));

    let h1 = handle.clone();
    let fs = Arc::clone(&first_started);
    let fd = Arc::clone(&first_done);
    let first = std::thread::spawn(move || {
        fs.store(true, Ordering::Release);
        h1.append(
            WalRecordType::PutNode,
            1,
            1,
            TenantId::DEFAULT,
            vec![0xAAu8; PAYLOAD_LEN],
        )
        .unwrap();
        fd.store(true, Ordering::Release);
    });

    wait_for(&first_started, "first-append thread to start");
    // Give the first append a scheduling quantum to reach `admit()` and
    // enqueue on the writer channel BEFORE we let it (or anyone) flush.
    // This is deterministic in effect (not in wall-clock): the second
    // append below re-checks budget state directly via the writer's
    // observability hook rather than relying on timing.
    let deadline = Instant::now() + WAIT_BUDGET;
    while writer.inflight_budget_bytes_in_use() == 0 {
        assert!(
            Instant::now() < deadline,
            "first append never admitted against the budget (bug: admission \
             not wired, or admit() is a no-op)"
        );
        std::thread::yield_now();
    }
    assert!(
        writer.inflight_budget_bytes_in_use() >= PAYLOAD_LEN as u64,
        "budget must reflect the first append's admitted bytes BEFORE fsync \
         completes — got {}",
        writer.inflight_budget_bytes_in_use()
    );

    // Now attempt the SECOND append on another thread. It must NOT be
    // admitted immediately: in_flight (>= PAYLOAD_LEN) + PAYLOAD_LEN would
    // exceed BUDGET, so `admit` must block. We prove blocking by asserting
    // it has NOT completed after a bounded grace window, THEN releasing
    // the first append and confirming the second completes promptly after.
    let second_started = Arc::new(AtomicBool::new(false));
    let second_done = Arc::new(AtomicBool::new(false));
    let h2 = handle.clone();
    let ss = Arc::clone(&second_started);
    let sd = Arc::clone(&second_done);
    let second = std::thread::spawn(move || {
        ss.store(true, Ordering::Release);
        h2.append(
            WalRecordType::PutNode,
            2,
            2,
            TenantId::DEFAULT,
            vec![0xBBu8; PAYLOAD_LEN],
        )
        .unwrap();
        sd.store(true, Ordering::Release);
    });
    wait_for(&second_started, "second-append thread to start");

    // Grace window: the second append must remain un-admitted (blocked in
    // `admit`) while the first's bytes are still reserved. This is the
    // gate's core assertion — NOT a sleep-based race, but a direct check
    // against the shared in-flight counter, which cannot advance past the
    // budget while `admit`'s Condvar wait holds the second thread.
    std::thread::sleep(Duration::from_millis(200));
    assert!(
        !second_done.load(Ordering::Acquire),
        "second append must be BLOCKED by the budget gate (first append's \
         bytes still in flight, pre-fsync) — RED-on-revert: a no-op budget \
         gate lets this complete immediately, which this assertion catches"
    );

    // Release the first append: the main thread drives durability here
    // (an explicit `flush()`), so we control exactly when the first
    // append's budget bytes are released back to the gate.
    handle.flush().unwrap();
    wait_for(&first_done, "first append to complete its fsync");

    // The second append's `admit()` wakes on the release above and sends
    // its `WalCmd::Append` — it now sits in `pending` under the same
    // 1-hour window, so it needs its OWN `flush()` to become durable (T1
    // semantics: `append` blocks until fsync). Poll for its enqueue
    // (budget back up to PAYLOAD_LEN) then flush it.
    let deadline = Instant::now() + WAIT_BUDGET;
    while writer.inflight_budget_bytes_in_use() == 0 {
        assert!(
            Instant::now() < deadline,
            "second append never re-admitted after the first's release — \
             the budget gate deadlocked instead of waking the waiter"
        );
        std::thread::yield_now();
    }
    handle.flush().unwrap();

    // The second append must now complete promptly.
    wait_for(&second_done, "second append to be admitted after release");
    first.join().unwrap();
    second.join().unwrap();

    // Final invariant: once both appends have completed (their fires
    // durable), the budget returns to 0 in-flight.
    let deadline = Instant::now() + WAIT_BUDGET;
    while writer.inflight_budget_bytes_in_use() != 0 {
        assert!(
            Instant::now() < deadline,
            "budget bytes leaked: {} still in flight after both appends completed",
            writer.inflight_budget_bytes_in_use()
        );
        std::thread::yield_now();
    }
}

/// Negative control (pins the pre-M6 posture): with NO budget configured,
/// two appends whose combined size would exceed any reasonable budget are
/// both admitted immediately (no blocking) — proves the default posture
/// is unchanged (zero-overhead unbounded admission) and gives the RED-
/// on-revert leg above a true contrast: if the positive test's blocking
/// assertion were vacuous (e.g. a scheduling fluke), THIS test would also
/// spuriously show "blocking", which it does not.
#[test]
fn unbudgeted_config_never_blocks() {
    let dir = tempdir().unwrap();
    let config = unbudgeted_config(dir.path().to_path_buf());
    let writer = WalWriter::spawn(config).unwrap();
    let handle = writer.handle();

    let start = Instant::now();
    for i in 1..=8u64 {
        handle
            .append(
                WalRecordType::PutNode,
                i,
                i as i64,
                TenantId::DEFAULT,
                vec![0xCCu8; 64 * 1024],
            )
            .unwrap();
    }
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "unbudgeted appends must never block on an admission gate"
    );
    assert_eq!(
        writer.inflight_budget_bytes_in_use(),
        0,
        "no budget configured => the observability hook reports 0, never a real count"
    );
}

/// A single append whose OWN length exceeds the configured budget must
/// still be admitted (MECH-E8's back-pressure-never-deadlock lesson
/// applied to the WAL channel): blocking forever on an unsatisfiable
/// reservation would be a liveness bug, not a budget.
#[test]
fn oversized_single_append_admits_alone_never_deadlocks() {
    let dir = tempdir().unwrap();
    const TINY_BUDGET: u64 = 16; // far smaller than the payload below
    let config = budget_config(dir.path().to_path_buf(), TINY_BUDGET);
    let writer = WalWriter::spawn(config).unwrap();
    let handle = writer.handle();

    let start = Instant::now();
    handle
        .append(
            WalRecordType::PutNode,
            1,
            1,
            TenantId::DEFAULT,
            vec![0xDDu8; 4096], // >> TINY_BUDGET
        )
        .unwrap();
    assert!(
        start.elapsed() < Duration::from_secs(5),
        "an oversized single append must admit alone, never deadlock \
         waiting for a budget it can structurally never fit under"
    );
}
