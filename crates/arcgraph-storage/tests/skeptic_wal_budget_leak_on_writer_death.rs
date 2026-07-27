//! M6.1 (#1521 P1-4) — `skeptic_wal_budget_leak_on_writer_death`.
//!
//! `WalByteBudget::admit` blocks the CALLER's thread until room is
//! available; the ONLY release point is the writer thread's `fire()` call
//! (which releases a whole batch's reservation up front, before attempting
//! the fsync — see `fire`'s doc comment). If the writer thread dies (panics)
//! AFTER a command left a caller's `admit()`-reserved bytes in its
//! `pending` batch but BEFORE the next `fire()` runs, NOBODY ever calls
//! `release` for those bytes: they are stranded in the budget forever, and
//! any OTHER caller concurrently blocked in `admit`'s `Condvar::wait` for
//! room would wait FOREVER — an unbounded hang, not a bounded error.
//!
//! The fix: `WalByteBudget` gained a `dead` flag + `poison()` (drains the
//! stranded reservation to 0 and wakes every waiter); `WalWriter::shutdown`
//! and `Drop` call `poison()` unconditionally once the writer thread's
//! `JoinHandle` has been joined (covering BOTH a clean exit and a panic —
//! the thread is definitely gone either way); `admit`'s wait loop checks
//! `dead` on every wake and returns `Err(())`, surfaced to callers as
//! [`ArcGraphError::WalUnavailable`].
//!
//! DETERMINISM: this gate uses
//! `WalWriter::spawn_from_with_panic_after_first_pending_for_gate` (a
//! `#[doc(hidden)]` test-only seam, never reachable from production
//! `spawn`/`spawn_from`) to make the writer thread panic at a PRECISE,
//! deterministic point — immediately after the first `Append` command is
//! pushed onto `pending` (post-`admit`, pre-`fire`) — reproducing the
//! stranded-reservation race without any sleep, OS-scheduling luck, or
//! unrelated I/O fault.
//!
//! RED-on-revert: reverting the `WalByteBudget::poison`/`dead` mechanism
//! (or the `shutdown`/`Drop` call sites that invoke it) makes
//! `blocked_admit_unblocks_with_wal_unavailable_after_writer_death` hang
//! forever waiting on the second caller's `admit()` to return — this
//! gate's own bounded wait-with-timeout turns that hang into a deterministic
//! panic (not a silent false-green), which is the decisive negative signal.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::wal::{WalConfig, WalRecordType, WalWriter};
use tempfile::tempdir;

const WAIT: Duration = Duration::from_secs(30);

fn wait_for(flag: &AtomicBool, who: &str) {
    let start = Instant::now();
    while !flag.load(Ordering::Acquire) {
        if start.elapsed() > WAIT {
            panic!("rendezvous timed out waiting for {who} (peer dead or stalled)");
        }
        std::thread::yield_now();
    }
}

/// THE decisive leg: a first append's bytes are admitted and then
/// STRANDED (writer thread panics with the command in `pending`, pre-fire)
/// — a second, concurrent append that would exceed the remaining budget
/// must NOT hang forever waiting for room that can never be released; it
/// must observe the poison and return `WalUnavailable` within a BOUNDED
/// window.
#[test]
fn blocked_admit_unblocks_with_wal_unavailable_after_writer_death() {
    let dir = tempdir().unwrap();
    const PAYLOAD_LEN: usize = 4096;
    const BUDGET: u64 = (PAYLOAD_LEN as u64) + 100; // room for ~1, not 2.

    let config = WalConfig {
        // Long window: the first append's `pending` entry does NOT
        // auto-fire — it sits there until this gate's injected panic
        // fires (pre-fire, deterministically), never reaching a `fire()`
        // call that would have released its budget normally.
        group_commit_window: Duration::from_secs(3600),
        group_commit_max_batch: 64,
        ..WalConfig::new(dir.path().to_path_buf())
    }
    .with_inflight_budget_bytes(BUDGET);

    let writer =
        WalWriter::spawn_from_with_panic_after_first_pending_for_gate(config, Lsn::ZERO).unwrap();
    let handle = writer.handle();

    // First append: admits successfully (budget starts at 0), gets pushed
    // to `pending`, then the injected seam panics the writer thread
    // BEFORE any `fire()` call — its budget reservation is now stranded
    // (no release ever comes) and the writer thread is dead. The
    // `append` call itself returns `Err` (its `ack_rx.recv()` fails once
    // the writer thread's `ack` sender is dropped by the panic unwind).
    let first_started = Arc::new(AtomicBool::new(false));
    let h1 = handle.clone();
    let fs = Arc::clone(&first_started);
    let first = std::thread::spawn(move || {
        fs.store(true, Ordering::Release);
        let _ = h1.append(
            WalRecordType::PutNode,
            1,
            1,
            TenantId::DEFAULT,
            vec![0xAAu8; PAYLOAD_LEN],
        );
    });
    wait_for(&first_started, "first-append thread to start");

    // Wait for the first append's bytes to genuinely be admitted against
    // the budget (confirms the seam is exercising the REAL admission
    // path, not racing ahead of it).
    let deadline = Instant::now() + WAIT;
    while writer.inflight_budget_bytes_in_use() == 0 {
        assert!(
            Instant::now() < deadline,
            "first append never admitted against the budget"
        );
        std::thread::yield_now();
    }
    first.join().unwrap();

    // Second append: would need PAYLOAD_LEN more room, but the first
    // append's PAYLOAD_LEN is (per this schedule) STRANDED — under the
    // pre-fix behavior this would block in `admit`'s Condvar wait
    // FOREVER (no release, no poison, no notify_all ever comes). Drive
    // it on its own thread with a BOUNDED wait so the gate itself never
    // hangs even if the fix regresses.
    let second_started = Arc::new(AtomicBool::new(false));
    let second_done = Arc::new(AtomicBool::new(false));
    let second_result: Arc<parking_lot::Mutex<Option<bool>>> =
        Arc::new(parking_lot::Mutex::new(None));
    let h2 = handle.clone();
    let ss = Arc::clone(&second_started);
    let sd = Arc::clone(&second_done);
    let sr = Arc::clone(&second_result);
    let second = std::thread::spawn(move || {
        ss.store(true, Ordering::Release);
        let result = h2.append(
            WalRecordType::PutNode,
            2,
            2,
            TenantId::DEFAULT,
            vec![0xBBu8; PAYLOAD_LEN],
        );
        *sr.lock() = Some(result.is_err());
        sd.store(true, Ordering::Release);
    });
    wait_for(&second_started, "second-append thread to start");

    // THE decisive wait: bounded, not unbounded — this is what turns a
    // reverted fix's infinite hang into a deterministic RED (a panic
    // naming the dead peer) instead of the test process silently
    // hanging forever.
    wait_for(
        &second_done,
        "second append to unblock via poison (RED-on-revert: this hangs \
         forever without the WalByteBudget::poison fix)",
    );
    second.join().unwrap();

    assert_eq!(
        *second_result.lock(),
        Some(true),
        "the second (previously blocked) append must return Err \
         (WalUnavailable) once the budget is poisoned by the dead writer \
         thread — not silently succeed and not hang"
    );

    // Final invariant: the budget itself reflects the poison (drained to
    // 0), not a permanently-stranded nonzero count.
    assert_eq!(
        writer.inflight_budget_bytes_in_use(),
        0,
        "a poisoned budget must drain its stranded reservation to 0, not \
         leave it permanently nonzero"
    );
}

/// Sensitivity/negative-control leg: WITHOUT a budget configured, the
/// exact same writer-death schedule (injected panic after the first
/// pending command) still surfaces `WalUnavailable` to any subsequent
/// caller via the normal channel-disconnect path — proving the harness's
/// death-injection seam genuinely kills the writer thread (this would
/// fail if the seam were a no-op), independent of the budget mechanism
/// entirely.
#[test]
fn writer_death_surfaces_wal_unavailable_even_without_budget() {
    let dir = tempdir().unwrap();
    let config = WalConfig {
        group_commit_window: Duration::from_secs(3600),
        group_commit_max_batch: 64,
        ..WalConfig::new(dir.path().to_path_buf())
    };
    let writer =
        WalWriter::spawn_from_with_panic_after_first_pending_for_gate(config, Lsn::ZERO).unwrap();
    let handle = writer.handle();

    // First append triggers the injected panic (writer thread dies).
    let _ = handle.append(
        WalRecordType::PutNode,
        1,
        1,
        TenantId::DEFAULT,
        vec![0xAAu8; 64],
    );

    // A subsequent append must observe the dead channel and return
    // `WalUnavailable` promptly (never hang), confirming the writer
    // thread is genuinely gone.
    let deadline = Instant::now() + WAIT;
    loop {
        match handle.append(
            WalRecordType::PutNode,
            2,
            2,
            TenantId::DEFAULT,
            vec![0xBBu8; 64],
        ) {
            Err(arcgraph_core::ArcGraphError::WalUnavailable) => break,
            Err(other) => panic!("unexpected error after writer death: {other:?}"),
            Ok(_) => {
                assert!(
                    Instant::now() < deadline,
                    "writer thread never actually died from the injected panic"
                );
                std::thread::yield_now();
            }
        }
    }
}
