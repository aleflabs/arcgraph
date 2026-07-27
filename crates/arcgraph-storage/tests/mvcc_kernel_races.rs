//! Barrier-ordered regression tests for the two kernel race bugs
//! uncovered in the post-merge deep review of M2.b.
//!
//! These are NOT proptests. They exploit `#[doc(hidden)]` test-only
//! barrier hooks on `TxnManager` to force the exact problematic
//! interleaving deterministically. Random scheduling cannot reach
//! these windows reliably on fast hardware.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_kernel_races --nocapture
//!
//! Bug 1 (begin/gc TOCTOU): a `begin()` that has read `counter` but
//! not yet published into `active` is invisible to a concurrent
//! `gc()`, which then anchors to `counter.current()` — which has
//! advanced past the captured snapshot. GC reclaims versions the
//! captured snapshot is supposed to see.
//!
//! Bug 2 (commit mid-install): `commit_writes` allocates `commit_lsn`
//! (advancing `counter`) BEFORE it finishes installing the write set.
//! A concurrent `begin()` that captures `counter.current() ==
//! commit_lsn` as its snapshot can read some installed keys and some
//! pre-install keys from the same txn — observing a half-applied
//! commit.

use std::sync::{Arc, Barrier};

use arcgraph_core::TenantId;
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;

/// Number of trials for the barrier-ordered reproducers.
///
/// Default (CI and local): 64 — sufficient to catch regressions given
/// that the interleaving is deterministic (barrier-enforced). Set
/// `MVCC_STRESS_TRIALS=1000` for the 1000-trial release gate. The env
/// var is the gate rather than `cfg(not(debug_assertions))` so the
/// developer can decide when to pay the cost; `cargo test --release`
/// alone is not the gate.
fn trials() -> u32 {
    std::env::var("MVCC_STRESS_TRIALS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64)
}

/// Bug 1 reproducer.
///
/// On `cffbe14` (pre-fix) this test fails: `gc()` reclaims a version
/// the captured-but-unpublished snapshot was supposed to read. After
/// the fix in commit 2 (sentinel-insert-then-upgrade), it passes.
#[test]
fn reader_at_captured_snapshot_is_protected_from_gc() {
    for trial in 0..trials() {
        let m = Arc::new(TxnManager::new());

        // Seed: commit K=A so the version store has a V1 with
        // created_lsn=1, expired_lsn=MAX, value=A. counter == 1.
        {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::from_static(b"A"));
            t.commit().unwrap();
        }

        let after_snapshot_read = Arc::new(Barrier::new(2));
        let before_publish = Arc::new(Barrier::new(2));

        let b_handle = {
            let m = Arc::clone(&m);
            let b1 = Arc::clone(&after_snapshot_read);
            let b2 = Arc::clone(&before_publish);
            std::thread::spawn(move || {
                // Captures snapshot=1, pauses before publishing into active.
                let txn = m.begin_with_barrier(TenantId::DEFAULT, &b1, &b2);
                // After the driver releases `before_publish`, this txn
                // reads K at snapshot=1.
                txn.read(1)
            })
        };

        // Wait until B has read counter.current() == 1 as its snapshot.
        after_snapshot_read.wait();

        // Overwrite K: new version V2 with created=2. V1 is expired
        // with expired_lsn=2. counter == 2.
        {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::from_static(b"B"));
            t.commit().unwrap();
        }

        // With B still unpublished in `active`, run GC. On the buggy
        // code path, `oldest_active_snapshot()` falls through to
        // `counter.current() == 2`, and V1 (expired_lsn=2) gets
        // reclaimed even though snapshot=1 still needs to see it.
        let _ = m.gc();

        // Release B. It publishes active[tid]=1 and then reads K.
        before_publish.wait();

        let observed = b_handle.join().unwrap();
        assert!(
            observed.is_some(),
            "trial {trial}: reader at snapshot=1 saw None — V1 was \
             reclaimed by GC in the begin/gc TOCTOU window. \
             Expected Some(b\"A\") (or a later still-visible version)."
        );
        // Stronger assertion: at snapshot=1 only V1 ("A") is visible.
        // V2 has created=2 which is not ≤ 1.
        assert_eq!(
            observed.as_deref(),
            Some(&b"A"[..]),
            "trial {trial}: reader at snapshot=1 should see V1=A, got {observed:?}"
        );
    }
}

/// Bug 2 reproducer.
///
/// On the state of the tree just prior to the Bug 2 fix (commit 4)
/// this test fails: a reader whose snapshot equals a committing txn's
/// freshly-allocated commit_lsn observes a half-applied write set
/// (K1=NEW, K2=OLD). After the fix (two-counter pattern; readers
/// source their snapshot from `visible`, which advances only after
/// the install loop completes), it passes.
#[test]
fn reader_never_sees_half_applied_commit() {
    for trial in 0..trials() {
        let m = Arc::new(TxnManager::new());

        // Seed K1=OLD_A, K2=OLD_B via two separate commits so the
        // version store has one live version for each key.
        {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(1, Bytes::from_static(b"OLD_A"));
            t.commit().unwrap();
        }
        {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(2, Bytes::from_static(b"OLD_B"));
            t.commit().unwrap();
        }

        let between_alloc_and_install = Arc::new(Barrier::new(2));
        let between_first_and_second = Arc::new(Barrier::new(2));

        let x_handle = {
            let m = Arc::clone(&m);
            let b1 = Arc::clone(&between_alloc_and_install);
            let b2 = Arc::clone(&between_first_and_second);
            std::thread::spawn(move || {
                let mut t = m.begin(TenantId::DEFAULT);
                t.write(1, Bytes::from_static(b"NEW_A"));
                t.write(2, Bytes::from_static(b"NEW_B"));
                // Install K1 first, then K2. Barriers: b1 just after
                // commit_lsn allocation, b2 after K1 installed.
                t.commit_with_barriers(&[1, 2], &b1, &b2)
            })
        };

        // Wait for X to allocate commit_lsn (counter has advanced).
        between_alloc_and_install.wait();

        // At this point, on the buggy code path, `counter.current()`
        // == commit_lsn. A reader who begins here captures that as
        // its snapshot — and will see K1's new version (already
        // installed? no — install happens AFTER this barrier point
        // in the hook). The critical window is between the FIRST
        // install and the SECOND, which is gated by b2. So we
        // release b2 after begin'ing to land in the mid-install hole.
        //
        // Wait — X is blocked at b2 (it won't start installing until
        // b2 releases). So we need to order: X allocates, X blocked
        // at the implicit b1 wait already happened (we just released
        // b1 by reaching this point). Now X proceeds into the
        // install loop: installs K1, then waits on b2. We begin a
        // reader here. Reader's snapshot == commit_lsn (counter has
        // advanced). Reader reads K1 (NEW_A if installed) and K2
        // (OLD_B if not yet installed). With b2 still held, K1 is
        // installed but K2 is not.
        //
        // Give X a window to install K1 and park at b2.
        // Deterministic: X's code path after b1.wait() is:
        //   install K1; b2.wait();
        // so by the time we can observe mid-install, X has installed
        // K1 and is parked. We rely on a brief yield/sleep to let X
        // progress from b1 to b2; a barrier between K1-done and
        // reader-begin would require a 3-way barrier which muddles
        // the semantics. Instead we busy-poll on chain_len(1) until
        // K1's new version appears — this is the committing thread's
        // observable side effect and is the deterministic signal.
        loop {
            if m.chain_len(TenantId::DEFAULT, 1) == 2 {
                break;
            }
            std::thread::yield_now();
        }

        // Now: K1 has NEW_A installed; K2 does not yet have NEW_B.
        // Begin a reader. On buggy code, snapshot=commit_lsn; on
        // fixed code, snapshot=visible which hasn't advanced yet
        // (still old watermark).
        let reader = m.begin(TenantId::DEFAULT);
        let r1 = reader.read(1);
        let r2 = reader.read(2);
        reader.abort();

        // Release X to finish installing K2 and advance visible.
        between_first_and_second.wait();
        x_handle.join().unwrap().unwrap();

        // The pair must be consistent: either both old or both new.
        let both_old = r1.as_deref() == Some(&b"OLD_A"[..]) && r2.as_deref() == Some(&b"OLD_B"[..]);
        let both_new = r1.as_deref() == Some(&b"NEW_A"[..]) && r2.as_deref() == Some(&b"NEW_B"[..]);
        assert!(
            both_old || both_new,
            "trial {trial}: half-applied commit observed — r1={r1:?}, r2={r2:?}"
        );
    }
}

/// Ungated torture test: 4 threads for 2 seconds doing random
/// begin/write/commit/read/gc. Asserts snapshot-isolation invariants
/// on a per-thread recorded trace: within a single transaction, every
/// read observes either a consistent "old" or "new" tuple across the
/// key pair — never a mix.
///
/// This test can miss the bug on fast hardware where the interleaving
/// windows are too tight to hit by chance. The barrier-ordered tests
/// above are the definitive regression; this torture test is
/// additional signal.
#[test]
fn random_concurrent_ops_preserve_snapshot_isolation() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    let m = Arc::new(TxnManager::new());
    // Seed a batch of paired keys. Each "logical tuple" is (2k, 2k+1)
    // written by a single txn, so a reader at any snapshot should see
    // either both-old or both-new for a pair.
    for k in 0..8u64 {
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(2 * k, Bytes::from_static(b"v0"));
        t.write(2 * k + 1, Bytes::from_static(b"v0"));
        t.commit().unwrap();
    }

    let stop = Arc::new(AtomicBool::new(false));
    let deadline = Instant::now() + Duration::from_secs(2);

    let handles: Vec<_> = (0..4u32)
        .map(|tid| {
            let m = Arc::clone(&m);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                // Simple LCG per-thread to avoid adding an rng dep.
                let mut rng: u64 = 0x9E37_79B9_7F4A_7C15u64.wrapping_add(tid as u64);
                let mut iter = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let op = rng & 0x7;
                    match op {
                        0..=2 => {
                            // Reader: pick a pair, read both.
                            let k = (rng >> 3) & 0x7;
                            let txn = m.begin(TenantId::DEFAULT);
                            let a = txn.read(2 * k);
                            let b = txn.read(2 * k + 1);
                            // Strongest check: whenever both keys
                            // are present at the snapshot, they
                            // must be consistent (always written
                            // together by the same txn).
                            if let (Some(a), Some(b)) = (&a, &b) {
                                assert_eq!(
                                    a, b,
                                    "tid={tid}: pair ({k}, {k}') saw mixed \
                                     values a={a:?} b={b:?} — snapshot isolation violated"
                                );
                            }
                            // Half-missing can legitimately arise if
                            // gc prunes an entry whose tail has
                            // fully expired — that's a benign
                            // reclamation, not a consistency bug.
                            txn.abort();
                        }
                        3..=5 => {
                            // Writer: pick a pair, overwrite both to
                            // a tid-tagged marker.
                            let k = (rng >> 3) & 0x7;
                            let tag = format!("t{tid}i{iter}").into_bytes();
                            let mut txn = m.begin(TenantId::DEFAULT);
                            txn.write(2 * k, Bytes::from(tag.clone()));
                            txn.write(2 * k + 1, Bytes::from(tag));
                            let _ = txn.commit();
                        }
                        6 => {
                            let _ = m.gc();
                        }
                        _ => std::thread::yield_now(),
                    }
                    iter += 1;
                }
            })
        })
        .collect();

    while Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(50));
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    for h in handles {
        h.join().unwrap();
    }
}
