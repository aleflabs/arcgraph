//! M2-E4 — Jepsen-style snapshot-isolation torture (single-node, v1.0).
//!
//! Random 8-worker × 10-min begin/read/write/commit/abort trace,
//! followed by an in-process verifier that asserts the four SI
//! invariants from the M2.e prompt §M2-E4 contract against the
//! linearized commit-LSN order:
//!
//!   1. **Snapshot isolation.** For every successful `read(key)` by
//!      txn T at snapshot LSN S, the value returned corresponds to
//!      the version V with the largest `V.commit_lsn ≤ S` (or None
//!      if the key was unwritten up to S). Values newer than S MUST
//!      NOT be returned.
//!   2. **No lost updates.** For every pair of committed txns T1,
//!      T2 writing to the same key with `T1.commit_lsn <
//!      T2.commit_lsn`, we must have `T2.snapshot_lsn ≥
//!      T1.commit_lsn` — otherwise T2 missed T1's write and yet
//!      committed, which is a lost-update violation of first-
//!      committer-wins.
//!   3. **No dirty reads.** Every observed non-None read returns
//!      bytes authored by a committed transaction with
//!      `author.commit_lsn ≤ S`. (Subsumed by (1), but worth
//!      asserting directly as a cheaper sanity check.)
//!   4. **Read-your-writes.** Within a single txn, `read(key)` after
//!      an earlier `write(key, v)` returns `v`.
//!
//! ## Scope
//!
//! Tests **single-node SI** (the v1.0 shape per ADR-024 §5).
//! HLC-based cluster-wide SI is a v1.1 concern and the verifier will
//! need extension for that trace shape when distribution lands.
//!
//! ## Runtime gating
//!
//! `#[ignore]` by default — a 10-minute torture does not belong in
//! the default `cargo test --workspace` loop. Run explicitly:
//!
//!   cargo test -p arcgraph-storage --release \
//!     --test m2e_jepsen_snapshot_isolation -- --ignored --nocapture
//!
//! ### Environment overrides
//!
//!   M2E_JEPSEN_WORKERS        (default 8)
//!   M2E_JEPSEN_DURATION_SECS  (default 600 = 10 min per M2.e prompt)
//!   M2E_JEPSEN_KEYSPACE       (default 256 — small enough to force
//!                              conflicts, large enough for meaningful
//!                              interleavings)
//!   M2E_JEPSEN_WAL_WINDOW_MS  (default 1)
//!   M2E_JEPSEN_WAL_BATCH      (default 16)
//!
//! ## Value encoding (24 bytes per write)
//!
//! ```text
//! [0..8]   author_txn_id   (u64 LE)
//! [8..16]  seq_in_txn      (u64 LE — 0,1,2,... per txn in write order)
//! [16..24] key             (u64 LE — self-integrity sanity)
//! ```
//!
//! The observed bytes on a read identify the authoring txn, which the
//! verifier cross-references with the commit-LSN log.
//!
//! ## Interleaving density under pre-fix commit-gate (M2-E2)
//!
//! Per the round-2 prompt: this harness runs on pre-fix main where
//! `commit_gate` serializes across WAL `fsync`, so 8 writers achieve
//! ~165 TPS aggregate. Over 600 s that produces ~100 K commits /
//! ~400 K events — enough interleavings to falsify SI if it were
//! broken (Jepsen's CockroachDB SI defects surfaced at ≤10 K ops).
//! A post-fix re-run at 5 K TPS would give ~30× denser coverage but
//! the invariant-truth of the verifier verdict is orthogonal to
//! throughput. If the post-fix re-run on `m2-e-verdict-round3` turns
//! up anything that this run missed, that is a verifier limitation
//! to investigate, not a property of the commit-gate fix.

use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use arcgraph_storage::transaction::{MvccKey, TxnManager};
use arcgraph_storage::{WalConfig, WalWriter};
use bytes::Bytes;

// ─── env helpers ──────────────────────────────────────────────────

fn env_u64(name: &str, default: u64) -> u64 {
    env::var(name)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(default)
}

// ─── deterministic xorshift (no external RNG dep) ─────────────────

#[inline]
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

// ─── value codec ──────────────────────────────────────────────────

fn encode_value(author_txn_id: u64, seq: u64, key: u64) -> [u8; 24] {
    let mut out = [0u8; 24];
    out[0..8].copy_from_slice(&author_txn_id.to_le_bytes());
    out[8..16].copy_from_slice(&seq.to_le_bytes());
    out[16..24].copy_from_slice(&key.to_le_bytes());
    out
}

fn decode_value(v: &[u8]) -> Option<(u64, u64, u64)> {
    if v.len() != 24 {
        return None;
    }
    let mut b8 = [0u8; 8];
    b8.copy_from_slice(&v[0..8]);
    let author = u64::from_le_bytes(b8);
    b8.copy_from_slice(&v[8..16]);
    let seq = u64::from_le_bytes(b8);
    b8.copy_from_slice(&v[16..24]);
    let key = u64::from_le_bytes(b8);
    Some((author, seq, key))
}

// ─── events ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum EventKind {
    Begin,
    Read { key: u64, observed: Option<Vec<u8>> },
    Write { key: u64, value: Vec<u8> },
    Commit { commit_lsn: u64 },
    CommitConflict,
    Abort,
}

#[derive(Clone, Debug)]
struct Event {
    /// Preserved per M2.e §M2-E4 contract (event tuple includes
    /// `worker_id` for trace reproducibility). The verifier does not
    /// consume it — txn_id alone uniquely identifies a txn across
    /// workers — but the field lets a post-mortem correlate a
    /// violating event back to its producing thread.
    #[allow(dead_code)]
    worker_id: u8,
    txn_id: u64,
    snapshot_lsn: u64,
    /// Preserved per M2.e §M2-E4 contract. Not consumed by the
    /// verifier; retained for post-mortem time-correlation.
    #[allow(dead_code)]
    wall_ns: u64,
    kind: EventKind,
}

// ─── worker loop ──────────────────────────────────────────────────

/// Txn-shape selection weights (out of 100):
///   40 — read-only
///   40 — write-only
///   15 — mixed read+write
///    5 — begin-then-abort
#[inline]
fn pick_shape(r: u64) -> TxnShape {
    let m = (r % 100) as u8;
    if m < 40 {
        TxnShape::ReadOnly
    } else if m < 80 {
        TxnShape::WriteOnly
    } else if m < 95 {
        TxnShape::Mixed
    } else {
        TxnShape::AbortEarly
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TxnShape {
    ReadOnly,
    WriteOnly,
    Mixed,
    AbortEarly,
}

fn worker_loop(
    worker_id: u8,
    txn_mgr: Arc<TxnManager>,
    stop: Arc<AtomicBool>,
    tenant: TenantId,
    keyspace: u64,
    t0: Instant,
) -> Vec<Event> {
    let mut events: Vec<Event> = Vec::with_capacity(200_000);
    // Each worker seeds its RNG with a different salt so traces are
    // reproducible but not coincidentally aligned.
    let mut rng: u64 =
        0x9E37_79B9_7F4A_7C15u64 ^ ((worker_id as u64).wrapping_mul(0x5851_F42D_4C95_7F2Du64));
    while !stop.load(Ordering::Relaxed) {
        let shape = pick_shape(xorshift64(&mut rng));
        let n_ops = 1 + (xorshift64(&mut rng) % 6); // 1..=6 ops per txn
        let mut tx = txn_mgr.begin(tenant);
        let txn_id = tx.id();
        let snap = tx.snapshot().raw();
        let wall = t0.elapsed().as_nanos() as u64;
        events.push(Event {
            worker_id,
            txn_id,
            snapshot_lsn: snap,
            wall_ns: wall,
            kind: EventKind::Begin,
        });

        let mut seq: u64 = 0;
        // Track this tx's own writes so RYW reads can reuse prior
        // write keys in Mixed mode. Without this, a Mixed-mode read's
        // chance of hitting a prior write on the same key is ~1/
        // keyspace (~0.4 % at default 256), and the verifier's RYW
        // coverage goes to zero — which was the pathology the 30 s
        // smoke test surfaced on the first harness revision.
        let mut own_writes: Vec<u64> = Vec::new();
        for op_idx in 0..n_ops {
            let choose_read = match shape {
                TxnShape::ReadOnly | TxnShape::AbortEarly => true,
                TxnShape::WriteOnly => false,
                // Mixed: first op is a write (seeds own_writes), last
                // op is a read (exercises RYW against whichever prior
                // write happened to land on the read's key), middle
                // ops are coin-flip alternating read/write.
                TxnShape::Mixed => {
                    if op_idx == 0 {
                        false // seed a write
                    } else if op_idx == n_ops - 1 {
                        true // exercise a trailing read
                    } else {
                        (xorshift64(&mut rng) & 1) == 0
                    }
                }
            };
            // Key selection:
            //   - WriteOnly / ReadOnly / AbortEarly: uniform over keyspace.
            //   - Mixed: 50/50 between "reuse a prior own-write key"
            //     (to drive RYW coverage) and fresh keyspace (to
            //     still exercise cross-tx SI reads). When own_writes
            //     is empty, fall back to fresh.
            let k = if matches!(shape, TxnShape::Mixed)
                && !own_writes.is_empty()
                && (xorshift64(&mut rng) & 1) == 1
            {
                let idx = (xorshift64(&mut rng) as usize) % own_writes.len();
                own_writes[idx]
            } else {
                xorshift64(&mut rng) % keyspace
            };
            let wall = t0.elapsed().as_nanos() as u64;
            if choose_read {
                let observed = tx.read(k as MvccKey);
                events.push(Event {
                    worker_id,
                    txn_id,
                    snapshot_lsn: snap,
                    wall_ns: wall,
                    kind: EventKind::Read {
                        key: k,
                        observed: observed.map(|b| b.to_vec()),
                    },
                });
            } else {
                let v = encode_value(txn_id, seq, k);
                tx.write(k as MvccKey, Bytes::copy_from_slice(&v));
                events.push(Event {
                    worker_id,
                    txn_id,
                    snapshot_lsn: snap,
                    wall_ns: wall,
                    kind: EventKind::Write {
                        key: k,
                        value: v.to_vec(),
                    },
                });
                own_writes.push(k);
                seq += 1;
            }
        }

        // Finalize.
        let wall = t0.elapsed().as_nanos() as u64;
        if matches!(shape, TxnShape::AbortEarly) {
            tx.abort();
            events.push(Event {
                worker_id,
                txn_id,
                snapshot_lsn: snap,
                wall_ns: wall,
                kind: EventKind::Abort,
            });
        } else {
            match tx.commit() {
                Ok(lsn) => events.push(Event {
                    worker_id,
                    txn_id,
                    snapshot_lsn: snap,
                    wall_ns: wall,
                    kind: EventKind::Commit {
                        commit_lsn: lsn.raw(),
                    },
                }),
                Err(_) => events.push(Event {
                    worker_id,
                    txn_id,
                    snapshot_lsn: snap,
                    wall_ns: wall,
                    kind: EventKind::CommitConflict,
                }),
            }
        }
    }
    events
}

// ─── verifier ─────────────────────────────────────────────────────

#[derive(Debug)]
struct CommittedTxn {
    txn_id: u64,
    snapshot_lsn: u64,
    commit_lsn: u64,
    /// Final per-key write installed at commit (writes overwrite
    /// within a tx; tx only installs the last value per key).
    writes: HashMap<u64, Vec<u8>>,
}

#[derive(Debug)]
struct VerifierResult {
    n_events: usize,
    n_txns: usize,
    n_committed: usize,
    n_conflicts: usize,
    n_aborted: usize,
    n_reads: usize,
    n_writes: usize,
    n_si_reads_checked: usize,
    n_ryw_reads_checked: usize,
    si_violations: Vec<String>,
    lost_update_violations: Vec<String>,
    dirty_read_violations: Vec<String>,
    ryw_violations: Vec<String>,
}

fn verify(events: &[Event]) -> VerifierResult {
    let mut result = VerifierResult {
        n_events: events.len(),
        n_txns: 0,
        n_committed: 0,
        n_conflicts: 0,
        n_aborted: 0,
        n_reads: 0,
        n_writes: 0,
        n_si_reads_checked: 0,
        n_ryw_reads_checked: 0,
        si_violations: Vec::new(),
        lost_update_violations: Vec::new(),
        dirty_read_violations: Vec::new(),
        ryw_violations: Vec::new(),
    };

    // Group events by txn_id, in order.
    let mut by_txn: HashMap<u64, Vec<&Event>> = HashMap::new();
    for e in events {
        by_txn.entry(e.txn_id).or_default().push(e);
    }
    result.n_txns = by_txn.len();

    // Extract committed txns with their installed writes.
    //
    // IMPORTANT: If a tx writes key k multiple times, only the last
    // write survives (HashMap-per-tx semantics inside the commit
    // path — see arcgraph-storage/src/transaction.rs `writes:
    // HashMap<MvccKey, Option<Bytes>>`). We mirror that here so the
    // verifier's model matches the production model.
    let mut committed: HashMap<u64, CommittedTxn> = HashMap::new();
    for (txn_id, evs) in &by_txn {
        let mut snapshot_lsn = 0u64;
        let mut commit_lsn_opt: Option<u64> = None;
        let mut finalized_as = None; // "commit" | "conflict" | "abort"
        let mut writes: HashMap<u64, Vec<u8>> = HashMap::new();
        let mut reads = 0usize;
        let mut wrs = 0usize;
        for e in evs {
            match &e.kind {
                EventKind::Begin => {
                    snapshot_lsn = e.snapshot_lsn;
                }
                EventKind::Read { .. } => reads += 1,
                EventKind::Write { key, value } => {
                    wrs += 1;
                    writes.insert(*key, value.clone());
                }
                EventKind::Commit { commit_lsn } => {
                    commit_lsn_opt = Some(*commit_lsn);
                    finalized_as = Some("commit");
                }
                EventKind::CommitConflict => {
                    finalized_as = Some("conflict");
                }
                EventKind::Abort => {
                    finalized_as = Some("abort");
                }
            }
        }
        result.n_reads += reads;
        result.n_writes += wrs;
        match finalized_as {
            Some("commit") => {
                if let Some(c) = commit_lsn_opt {
                    committed.insert(
                        *txn_id,
                        CommittedTxn {
                            txn_id: *txn_id,
                            snapshot_lsn,
                            commit_lsn: c,
                            writes,
                        },
                    );
                    result.n_committed += 1;
                } else {
                    result
                        .si_violations
                        .push(format!("txn {txn_id}: Commit event without commit_lsn"));
                }
            }
            Some("conflict") => result.n_conflicts += 1,
            Some("abort") => result.n_aborted += 1,
            None => { /* truncated trace — last txn never got a finalizer event */ }
            _ => unreachable!(),
        }
    }

    // Sorted per-key version chain: (commit_lsn, author_txn_id, value)
    let mut per_key_chain: HashMap<u64, Vec<(u64, u64, Vec<u8>)>> = HashMap::new();
    for t in committed.values() {
        for (k, v) in &t.writes {
            per_key_chain
                .entry(*k)
                .or_default()
                .push((t.commit_lsn, t.txn_id, v.clone()));
        }
    }
    for chain in per_key_chain.values_mut() {
        chain.sort_by_key(|(lsn, _, _)| *lsn);
    }

    // Invariant 2 — no lost updates. For each (key, chain), iterate
    // pairs in commit order; assert later snapshot ≥ earlier commit.
    for (k, chain) in &per_key_chain {
        for (i, (c2, t2, _)) in chain.iter().enumerate() {
            for (c1, t1, _) in &chain[..i] {
                // c1 < c2 by sort order.
                if let Some(t2_info) = committed.get(t2) {
                    if t2_info.snapshot_lsn < *c1 {
                        // t2 committed without seeing t1's commit of same key — violation.
                        result.lost_update_violations.push(format!(
                            "key {}: txn {} (snap {}) committed at {} but missed txn {} committed at {}",
                            k, t2, t2_info.snapshot_lsn, c2, t1, c1
                        ));
                    }
                }
            }
        }
    }

    // Invariants 1, 3, 4 — walk every Read event.
    for (txn_id, evs) in &by_txn {
        // Pre-compute per-tx own-write map (last write wins per key,
        // matching the commit-path semantics). Populate incrementally
        // so we can tell "at the time of the read, what was my pending
        // write on this key".
        let mut pending: HashMap<u64, Option<Vec<u8>>> = HashMap::new();
        for e in evs {
            match &e.kind {
                EventKind::Write { key, value } => {
                    pending.insert(*key, Some(value.clone()));
                }
                EventKind::Read { key, observed } => {
                    if let Some(p) = pending.get(key) {
                        // Invariant 4 — RYW
                        result.n_ryw_reads_checked += 1;
                        let expected = p.clone();
                        if observed.as_ref() != expected.as_ref() {
                            result.ryw_violations.push(format!(
                                "txn {txn_id} key {key}: RYW: expected {expected:?} observed {observed:?}"
                            ));
                        }
                    } else {
                        // Invariant 1 — SI
                        result.n_si_reads_checked += 1;
                        let s = e.snapshot_lsn;
                        let chain = per_key_chain.get(key);
                        let expected: Option<&[u8]> = chain.and_then(|c| {
                            // Largest commit_lsn ≤ s.
                            let idx = c.binary_search_by_key(&s, |(lsn, _, _)| *lsn);
                            let take_idx: Option<usize> = match idx {
                                Ok(i) => Some(i),
                                Err(0) => None,
                                Err(i) => Some(i - 1),
                            };
                            take_idx.map(|i| c[i].2.as_slice())
                        });
                        let obs_slice: Option<&[u8]> = observed.as_deref();
                        if obs_slice != expected {
                            result.si_violations.push(format!(
                                "txn {txn_id} key {key} snap {s}: SI: expected {:?} observed {:?}",
                                expected.map(decode_value),
                                obs_slice.map(decode_value),
                            ));
                        }
                        // Invariant 3 — dirty read cross-check.
                        if let Some(bytes) = obs_slice
                            && let Some((author, _, _)) = decode_value(bytes)
                        {
                            let author_lsn = committed.get(&author).map(|c| c.commit_lsn);
                            match author_lsn {
                                Some(al) if al <= s => { /* ok */ }
                                Some(al) => {
                                    result.dirty_read_violations.push(format!(
                                        "txn {txn_id} key {key} snap {s}: observed author {author} commit_lsn {al} > snapshot"
                                    ));
                                }
                                None => {
                                    result.dirty_read_violations.push(format!(
                                        "txn {txn_id} key {key} snap {s}: observed author {author} is not in committed set"
                                    ));
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    result
}

// ─── test ─────────────────────────────────────────────────────────

#[test]
#[ignore = "10-minute Jepsen-style SI torture; run with --ignored per M2-E4 contract"]
fn jepsen_snapshot_isolation_torture() {
    let workers = env_u64("M2E_JEPSEN_WORKERS", 8).max(1) as u8;
    let duration_secs = env_u64("M2E_JEPSEN_DURATION_SECS", 600).max(1);
    let keyspace = env_u64("M2E_JEPSEN_KEYSPACE", 256).max(1);
    let wal_window_ms = env_u64("M2E_JEPSEN_WAL_WINDOW_MS", 1).max(1);
    let wal_batch = env_u64("M2E_JEPSEN_WAL_BATCH", 16).max(1) as usize;

    eprintln!(
        "M2-E4 Jepsen SI torture: {workers} workers × {duration_secs}s  \
         keyspace={keyspace}  wal_window={wal_window_ms}ms  wal_batch={wal_batch}"
    );

    // ─── bring up WAL-backed stack ─────────────────────────────────
    let wal_dir = tempfile::tempdir().expect("tempdir");
    let wal_config = WalConfig {
        dir: wal_dir.path().to_path_buf(),
        segment_size_bytes: 256 * 1024 * 1024,
        group_commit_window: Duration::from_millis(wal_window_ms),
        group_commit_max_batch: wal_batch,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    };
    let wal_writer = WalWriter::spawn(wal_config).expect("wal writer spawn");
    let wal_handle = wal_writer.handle();
    let txn_mgr = Arc::new(TxnManager::with_wal(wal_handle.clone()));

    // ─── spawn workers ─────────────────────────────────────────────
    let stop = Arc::new(AtomicBool::new(false));
    let t0 = Instant::now();
    let mut handles = Vec::with_capacity(workers as usize);
    let worker_log_slots: Vec<Arc<parking_lot::Mutex<Option<Vec<Event>>>>> = (0..workers)
        .map(|_| Arc::new(parking_lot::Mutex::new(None)))
        .collect();
    for w_id in 0..workers {
        let txn_mgr = Arc::clone(&txn_mgr);
        let stop = Arc::clone(&stop);
        let slot = Arc::clone(&worker_log_slots[w_id as usize]);
        handles.push(
            thread::Builder::new()
                .name(format!("m2e-jepsen-{w_id}"))
                .spawn(move || {
                    let log = worker_loop(w_id, txn_mgr, stop, TenantId::DEFAULT, keyspace, t0);
                    *slot.lock() = Some(log);
                })
                .expect("spawn jepsen worker"),
        );
    }

    // Periodic progress ticks every 60 s so a 10-min run doesn't
    // look hung. `--nocapture` surfaces these.
    let tick_stop = Arc::clone(&stop);
    let tick_tx = Arc::clone(&txn_mgr);
    let tick_commits = Arc::new(AtomicU64::new(0));
    let tick_commits_clone = Arc::clone(&tick_commits);
    let tick_handle = thread::Builder::new()
        .name("m2e-jepsen-tick".to_owned())
        .spawn(move || {
            let last_lsn = tick_tx.current_lsn().raw();
            let start = Instant::now();
            while !tick_stop.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(60));
                if tick_stop.load(Ordering::Relaxed) {
                    break;
                }
                let cur = tick_tx.current_lsn().raw();
                let dt = start.elapsed().as_secs_f64();
                eprintln!(
                    "  [tick] elapsed={dt:.0}s  current_lsn={cur}  (δ from start {})",
                    cur.saturating_sub(last_lsn)
                );
                tick_commits_clone.store(cur.saturating_sub(last_lsn), Ordering::Relaxed);
            }
        })
        .expect("spawn tick thread");

    // ─── run for duration ──────────────────────────────────────────
    thread::sleep(Duration::from_secs(duration_secs));
    stop.store(true, Ordering::Relaxed);
    let run_wall = t0.elapsed();

    for h in handles {
        h.join().expect("worker join");
    }
    tick_handle.join().expect("tick join");

    // ─── aggregate event logs ──────────────────────────────────────
    let mut all_events: Vec<Event> = Vec::new();
    for slot in &worker_log_slots {
        if let Some(log) = slot.lock().take() {
            all_events.extend(log);
        }
    }
    eprintln!(
        "M2-E4 trace captured: {} events across {} workers in {:.1} s",
        all_events.len(),
        workers,
        run_wall.as_secs_f64(),
    );

    // ─── teardown WAL ──────────────────────────────────────────────
    drop(txn_mgr);
    drop(wal_handle);
    wal_writer.shutdown().expect("wal shutdown");

    // ─── verify ────────────────────────────────────────────────────
    let result = verify(&all_events);

    eprintln!();
    eprintln!("═══════════════════════ M2-E4 VERIFIER ═══════════════════════");
    eprintln!("  events:              {}", result.n_events);
    eprintln!("  txns:                {}", result.n_txns);
    eprintln!("    committed:         {}", result.n_committed);
    eprintln!("    conflicts:         {}", result.n_conflicts);
    eprintln!("    aborts:            {}", result.n_aborted);
    eprintln!("  reads:               {}", result.n_reads);
    eprintln!("    SI-checked:        {}", result.n_si_reads_checked);
    eprintln!("    RYW-checked:       {}", result.n_ryw_reads_checked);
    eprintln!("  writes:              {}", result.n_writes);
    eprintln!("  SI violations:       {}", result.si_violations.len());
    eprintln!(
        "  lost updates:        {}",
        result.lost_update_violations.len()
    );
    eprintln!(
        "  dirty reads:         {}",
        result.dirty_read_violations.len()
    );
    eprintln!("  RYW violations:      {}", result.ryw_violations.len());
    if !result.si_violations.is_empty() {
        eprintln!("\n  first 10 SI violations:");
        for v in result.si_violations.iter().take(10) {
            eprintln!("    {v}");
        }
    }
    if !result.lost_update_violations.is_empty() {
        eprintln!("\n  first 10 lost-update violations:");
        for v in result.lost_update_violations.iter().take(10) {
            eprintln!("    {v}");
        }
    }
    if !result.dirty_read_violations.is_empty() {
        eprintln!("\n  first 10 dirty-read violations:");
        for v in result.dirty_read_violations.iter().take(10) {
            eprintln!("    {v}");
        }
    }
    if !result.ryw_violations.is_empty() {
        eprintln!("\n  first 10 RYW violations:");
        for v in result.ryw_violations.iter().take(10) {
            eprintln!("    {v}");
        }
    }
    eprintln!("══════════════════════════════════════════════════════════════");

    let total_violations = result.si_violations.len()
        + result.lost_update_violations.len()
        + result.dirty_read_violations.len()
        + result.ryw_violations.len();

    // Sanity: if the run produced zero committed txns the trace is
    // useless and the verifier verdict is vacuous. Fail loudly.
    assert!(
        result.n_committed > 0,
        "M2-E4: no commits observed — trace is empty, verifier verdict would be vacuous"
    );
    assert!(
        result.n_si_reads_checked + result.n_ryw_reads_checked > 0,
        "M2-E4: no reads checked — workload mix pathology"
    );

    assert_eq!(
        total_violations,
        0,
        "M2-E4: {} total invariant violations (si={}, lost={}, dirty={}, ryw={})",
        total_violations,
        result.si_violations.len(),
        result.lost_update_violations.len(),
        result.dirty_read_violations.len(),
        result.ryw_violations.len(),
    );

    println!(
        "M2-E4 verifier GREEN: {} events, {} commits, {} SI reads + {} RYW reads checked, 0 violations",
        result.n_events, result.n_committed, result.n_si_reads_checked, result.n_ryw_reads_checked
    );
}

// ─── unit smoke: runs on every cargo test, 200 ms, deterministic ─
//
// Asserts the verifier catches a known SI violation when we feed it
// a fabricated trace. This is the "is the verifier's test-harness
// itself correct" check — if the big 10-min run passes green, this
// tiny test prevents a silent false-GREEN from a broken verifier.

#[test]
fn verifier_detects_stale_read() {
    // Trace: tx=1 writes key=7, commits at lsn=10.
    //        tx=2 begins at snapshot=10 (after tx=1), reads key=7 and
    //        SEES NOTHING. That is a SI violation: tx=2's snapshot
    //        covers tx=1's commit, so the read MUST return tx=1's value.
    let v_by_t1 = encode_value(1, 0, 7).to_vec();
    let events = vec![
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 1,
            kind: EventKind::Begin,
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 2,
            kind: EventKind::Write {
                key: 7,
                value: v_by_t1.clone(),
            },
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 3,
            kind: EventKind::Commit { commit_lsn: 10 },
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 10,
            wall_ns: 4,
            kind: EventKind::Begin,
        },
        // SI violation: snapshot 10 covers commit 10, so the read
        // MUST return v_by_t1, not None.
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 10,
            wall_ns: 5,
            kind: EventKind::Read {
                key: 7,
                observed: None,
            },
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 10,
            wall_ns: 6,
            kind: EventKind::Commit { commit_lsn: 11 },
        },
    ];
    let r = verify(&events);
    assert_eq!(r.si_violations.len(), 1, "verifier missed the stale-read");
}

#[test]
fn verifier_detects_lost_update() {
    // tx=1 begins at snap=5, writes key=3, commits at lsn=10.
    // tx=2 begins at snap=5 (concurrently), writes key=3, commits at
    // lsn=11 — this is a LOST UPDATE (tx=2 didn't see tx=1 yet its
    // write overwrote tx=1's). The production OCC path should have
    // aborted tx=2 with Conflict; a green commit here is a bug.
    let v1 = encode_value(1, 0, 3).to_vec();
    let v2 = encode_value(2, 0, 3).to_vec();
    let events = vec![
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 1,
            kind: EventKind::Begin,
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 2,
            kind: EventKind::Write {
                key: 3,
                value: v1.clone(),
            },
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 3,
            kind: EventKind::Commit { commit_lsn: 10 },
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 5,
            wall_ns: 4,
            kind: EventKind::Begin,
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 5,
            wall_ns: 5,
            kind: EventKind::Write {
                key: 3,
                value: v2.clone(),
            },
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 5,
            wall_ns: 6,
            kind: EventKind::Commit { commit_lsn: 11 },
        },
    ];
    let r = verify(&events);
    assert_eq!(
        r.lost_update_violations.len(),
        1,
        "verifier missed lost update"
    );
}

#[test]
fn verifier_accepts_ryw_and_conflict() {
    // Sanity: a well-formed trace with RYW and a conflict produces no
    // violations.
    let v1 = encode_value(1, 0, 3).to_vec();
    let events = vec![
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 1,
            kind: EventKind::Begin,
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 2,
            kind: EventKind::Write {
                key: 3,
                value: v1.clone(),
            },
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 3,
            kind: EventKind::Read {
                key: 3,
                observed: Some(v1.clone()),
            },
        },
        Event {
            worker_id: 0,
            txn_id: 1,
            snapshot_lsn: 5,
            wall_ns: 4,
            kind: EventKind::Commit { commit_lsn: 10 },
        },
        // tx=2 started concurrent, wrote same key, aborted as conflict.
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 5,
            wall_ns: 5,
            kind: EventKind::Begin,
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 5,
            wall_ns: 6,
            kind: EventKind::Write {
                key: 3,
                value: encode_value(2, 0, 3).to_vec(),
            },
        },
        Event {
            worker_id: 1,
            txn_id: 2,
            snapshot_lsn: 5,
            wall_ns: 7,
            kind: EventKind::CommitConflict,
        },
    ];
    let r = verify(&events);
    assert_eq!(r.si_violations.len(), 0);
    assert_eq!(r.ryw_violations.len(), 0);
    assert_eq!(r.lost_update_violations.len(), 0);
    assert_eq!(r.dirty_read_violations.len(), 0);
    assert_eq!(r.n_committed, 1);
    assert_eq!(r.n_conflicts, 1);
}
