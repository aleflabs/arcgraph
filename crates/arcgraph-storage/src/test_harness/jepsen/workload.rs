//! Workload generators for Jepsen-style isolation testing.
//!
//! ## BankTransferWorkload (Bailis et al. 2014 §3.1)
//!
//! The canonical SI sanity workload: `N` accounts, each seeded with
//! an initial balance, totalling `expected_sum`. Each operation
//! picks two random accounts (src, dst) and transfers a random
//! amount from src to dst inside one transaction. The transaction:
//!
//! 1. Reads src's balance.
//! 2. Reads dst's balance.
//! 3. If `src_balance >= amount`, writes `src_balance - amount` to
//!    src and `dst_balance + amount` to dst. Otherwise the
//!    transaction is a read-only sample (still recorded so the
//!    checker can verify the snapshot was consistent).
//! 4. Commits.
//!
//! Under correct SI, the sum of balances at *every committed LSN*
//! is invariant — any non-atomic transfer (e.g., a read that saw
//! src's old balance and a write that landed on dst's new balance)
//! breaks the sum, and the checker fires.
//!
//! ### Anomaly coverage at v0.1.0-alpha.0+1
//!
//! - **G1a / G1b (dirty / intermediate read)** — caught by the
//!   per-op visibility predicate in the checker.
//! - **Subset of G1c (cyclic information flow)** — caught when a
//!   two-txn cycle breaks the sum invariant. Longer cycles need
//!   full Elle (v1.1).
//! - **Commit-atomicity torn writes** between `transaction.rs`
//!   Phase 2 (WAL append) and Phase 3 (`visible.store`) — any
//!   partial application breaks the sum.
//!
//! **G-single (write skew) is NOT exercised by this workload.** Every
//! committing transaction writes to BOTH `src_key` AND `dst_key`, so
//! two concurrent transfers on overlapping (src, dst) pairs WW-conflict
//! and OCC aborts one of them — write skew structurally requires
//! non-overlapping write sets. The canonical Bailis 2014 §3.1 write-
//! skew witness (read both accounts → check overdraft → write only
//! src) is a v1.1 follow-up workload variant.
//!
//! ## RNG choice — XorShift64, not `rand`
//!
//! Per [`super::super::k1::injection`]'s rationale: `XorShift` is
//! already in use across the crate's tests, deterministic, seedable,
//! and zero new dep. The Jepsen workload reuses the same choice for
//! consistency. `arcgraph-storage` does not depend on `rand`; adding
//! it for one test-only module would inflate the dep surface
//! unnecessarily.
//!
//! ## Deferred to v1.1
//!
//! `ListAppendWorkload` (Kingsbury 2018 blog series + Elle paper
//! §3.2) is the canonical Elle workload for full cycle detection.
//! It is intentionally NOT in this module — adding it requires the
//! Elle checker, which is the v1.1 follow-up per ADR-047.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use arcgraph_core::{ArcGraphError, TenantId};
use bytes::Bytes;

use crate::transaction::{MvccKey, TxnManager};

use super::history::{OpBuilder, OperationHistory, RecordedOp};

/// Configuration for [`run_bank_transfer`]. All fields public so
/// the test can override individual knobs.
///
/// Defaults track ADR-047 §"Decision" §"First test landing":
/// 4 clients × 100 ops × N=10 accounts × initial balance 100.
#[derive(Debug, Clone, Copy)]
pub struct BankTransferConfig {
    /// Number of concurrent worker threads.
    pub clients: u32,
    /// Per-client transfer ops.
    pub ops_per_client: u64,
    /// Number of accounts. Each account is keyed by a u64 in
    /// `[ACCOUNT_KEY_BASE .. ACCOUNT_KEY_BASE + accounts)`.
    pub accounts: u32,
    /// Initial balance per account.
    pub initial_balance: u64,
    /// Maximum transfer amount per op (uniform in
    /// `[MIN_TRANSFER .. max_transfer]`).
    pub max_transfer: u64,
    /// OCC retries per op before recording the op as
    /// `OpOutcome::Aborted`.
    pub max_retries: u32,
    /// XorShift64 seed driving (src, dst, amount) selection across
    /// the workload. The same seed produces the same
    /// (per-client-id) deterministic generator sequence; the
    /// *interleaving* of those ops is non-deterministic.
    pub seed: u64,
    /// Tenant under which all transactions run. Defaults to
    /// `TenantId::DEFAULT`; multi-tenant variants are a v1.1
    /// follow-up.
    pub tenant: TenantId,
}

/// Default seed for [`BankTransferConfig::default`]. Picked to be
/// distinguishable in logs from other test seeds.
pub const DEFAULT_SEED: u64 = 0x4A45_5053_454E_5F31; // ASCII "JEPS\x45N\x5F1"

impl Default for BankTransferConfig {
    fn default() -> Self {
        Self {
            clients: 4,
            ops_per_client: 100,
            accounts: 10,
            initial_balance: 100,
            max_transfer: 50,
            max_retries: 8,
            seed: DEFAULT_SEED,
            tenant: TenantId::DEFAULT,
        }
    }
}

/// First `MvccKey` reserved for the bank-transfer account namespace.
/// Picked to avoid clashing with any test-internal keying conventions.
pub const ACCOUNT_KEY_BASE: MvccKey = 0xBA1F_0000_0000_0000;

/// Minimum transfer amount per op. Anything below this would make
/// the read-only "src_balance < amount" branch dominant, defeating
/// the workload's purpose of stressing the commit path.
pub const MIN_TRANSFER: u64 = 1;

/// Expected total balance after seeding `accounts × initial_balance`.
#[must_use]
pub fn expected_sum(cfg: &BankTransferConfig) -> u64 {
    u64::from(cfg.accounts) * cfg.initial_balance
}

/// `client_id` reserved for seed-phase ops in the recorded history.
/// Picked as `u32::MAX` so it sorts after every real worker thread
/// and is distinguishable in printed histories.
pub const SEED_CLIENT_ID: u32 = u32::MAX;

/// Seed every account at `(ACCOUNT_KEY_BASE + i, initial_balance)`
/// via a sequence of single-key transactions.
///
/// Runs serially (one txn per account) before the workload starts;
/// this is the equivalent of Bailis 2014's "initial state setup."
/// Returns the LSN of the last seeding commit.
///
/// When `history` is `Some`, each seed commit is recorded with
/// `client_id = SEED_CLIENT_ID` and `op_id = i` so the checker can
/// walk the seeded initial state as part of its committed prefix.
/// The unit-test path (no history) is preserved for pure
/// state-setup callers that don't run the checker.
pub fn seed_accounts(
    mgr: &TxnManager,
    cfg: &BankTransferConfig,
    history: Option<&OperationHistory>,
) -> arcgraph_core::Lsn {
    let mut last = mgr.current_lsn();
    let initial_bytes = encode_balance(cfg.initial_balance);
    for i in 0..cfg.accounts {
        let key = ACCOUNT_KEY_BASE.wrapping_add(u64::from(i));
        let mut t = mgr.begin(cfg.tenant);
        let start = t.snapshot();
        let mut builder = OpBuilder::new(SEED_CLIENT_ID, u64::from(i), cfg.tenant, start);
        builder.intend_write(key, Some(initial_bytes.clone()));
        t.write(key, initial_bytes.clone());
        match t.commit() {
            Ok(lsn) => {
                last = lsn;
                if let Some(h) = history {
                    h.push(builder.into_committed(lsn));
                }
            }
            Err(e) => panic!("seed_accounts: account {i} commit failed unexpectedly: {e:?}"),
        }
    }
    last
}

/// Encode a `u64` balance as bytes for MVCC storage. Little-endian
/// fixed-size so the workload + checker agree on layout without a
/// codec dep.
#[must_use]
pub fn encode_balance(balance: u64) -> Bytes {
    Bytes::copy_from_slice(&balance.to_le_bytes())
}

/// Decode bytes back into a `u64` balance.
///
/// Returns `None` if the bytes are not exactly 8 long — the workload
/// and checker treat that as "this key is not a bank-transfer
/// account" and skip it (defensive: protects against accidental key
/// collisions if the workload is mixed with other tests).
#[must_use]
pub fn decode_balance(bytes: &[u8]) -> Option<u64> {
    if bytes.len() != 8 {
        return None;
    }
    let mut arr = [0u8; 8];
    arr.copy_from_slice(bytes);
    Some(u64::from_le_bytes(arr))
}

/// Run the bank-transfer workload to completion across
/// `cfg.clients` worker threads. Returns when every worker
/// completes its `ops_per_client` ops; the [`OperationHistory`] is
/// populated for the checker.
///
/// **Synchronous join** — this function blocks until every worker
/// thread finishes. The SIGKILL variant is invoked separately via
/// the [`super::fault_injection`] module; this entry point is the
/// steady-state path.
pub fn run_bank_transfer(
    mgr: Arc<TxnManager>,
    cfg: BankTransferConfig,
    history: Arc<OperationHistory>,
) {
    seed_accounts(&mgr, &cfg, Some(&history));

    let global_op_counter = Arc::new(AtomicU64::new(0));
    let handles: Vec<_> = (0..cfg.clients)
        .map(|client_id| {
            let mgr = Arc::clone(&mgr);
            let history = Arc::clone(&history);
            let global_op_counter = Arc::clone(&global_op_counter);
            // Each client gets a derived RNG seed (avalanche multiplier
            // from xxHash64) so they're independent but the workload as
            // a whole is reproducible from `cfg.seed`.
            let client_seed = cfg
                .seed
                .wrapping_add(u64::from(client_id).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            thread::Builder::new()
                .name(format!("jepsen-bank-client-{client_id}"))
                .spawn(move || {
                    run_client(
                        client_id,
                        client_seed,
                        &mgr,
                        &cfg,
                        &history,
                        &global_op_counter,
                    );
                })
                .expect("spawn jepsen worker thread")
        })
        .collect();

    for h in handles {
        h.join().expect("jepsen worker thread panicked");
    }
}

/// One client's main loop: pick (src, dst, amount), open a txn,
/// transfer, commit. Retries on `MvccConflict` up to `cfg.max_retries`.
fn run_client(
    client_id: u32,
    seed: u64,
    mgr: &TxnManager,
    cfg: &BankTransferConfig,
    history: &OperationHistory,
    global_op_counter: &AtomicU64,
) {
    let mut rng = XorShift64::new(seed);
    for _ in 0..cfg.ops_per_client {
        // op_id is a global monotonic counter so checker output is
        // unambiguous when ops from different clients interleave.
        let op_id = global_op_counter.fetch_add(1, Ordering::Relaxed);
        let recorded = run_one_transfer(client_id, op_id, &mut rng, mgr, cfg);
        history.push(recorded);
    }
}

/// One transfer attempt with retries.
fn run_one_transfer(
    client_id: u32,
    op_id: u64,
    rng: &mut XorShift64,
    mgr: &TxnManager,
    cfg: &BankTransferConfig,
) -> RecordedOp {
    // Pick (src, dst, amount) ONCE per op; retries reuse the same
    // (src, dst, amount) so a deterministic seed gives a
    // deterministic per-op decision sequence regardless of retry
    // count. The interleaving of WHICH retry attempt wins is still
    // non-deterministic — that's the Jepsen contract.
    let (src_idx, dst_idx) = pick_distinct_accounts(rng, cfg.accounts);
    let amount = uniform_range(rng, MIN_TRANSFER, cfg.max_transfer);
    let src_key = account_key(src_idx);
    let dst_key = account_key(dst_idx);

    for attempt in 0..=cfg.max_retries {
        let tx = mgr.begin(cfg.tenant);
        let start_lsn = tx.snapshot();
        let mut builder = OpBuilder::new(client_id, op_id, cfg.tenant, start_lsn);
        for _ in 0..attempt {
            builder.bump_retry();
        }
        // Read both accounts at the snapshot.
        let src_bytes = tx.read(src_key);
        let dst_bytes = tx.read(dst_key);
        builder.observe_read(src_key, src_bytes.clone());
        builder.observe_read(dst_key, dst_bytes.clone());

        let src_balance = src_bytes.as_deref().and_then(decode_balance).unwrap_or(0);
        let dst_balance = dst_bytes.as_deref().and_then(decode_balance).unwrap_or(0);

        let mut tx = tx; // re-bind as mutable for buffered writes
        if src_balance >= amount {
            let new_src = src_balance - amount;
            let new_dst = dst_balance.saturating_add(amount);
            let src_v = encode_balance(new_src);
            let dst_v = encode_balance(new_dst);
            tx.write(src_key, src_v.clone());
            tx.write(dst_key, dst_v.clone());
            builder.intend_write(src_key, Some(src_v));
            builder.intend_write(dst_key, Some(dst_v));
        }
        // (If src_balance < amount, this is a read-only sample —
        // commit anyway so the history records the consistent
        // snapshot.)

        match tx.commit() {
            Ok(commit_lsn) => return builder.into_committed(commit_lsn),
            Err(ArcGraphError::MvccConflict { .. }) => {
                if attempt == cfg.max_retries {
                    return builder.into_aborted();
                }
                // else: loop and retry with a fresh transaction.
            }
            Err(e) => {
                // Non-conflict failures are recorded as aborted. The
                // SIGKILL variant intentionally produces I/O errors
                // mid-commit; the checker's job is to reconcile against
                // post-recovery state, not to fail loudly here.
                tracing::warn!(
                    client_id,
                    op_id,
                    attempt,
                    error = %e,
                    "bank-transfer op failed with non-conflict error"
                );
                return builder.into_aborted();
            }
        }
    }
    unreachable!("retry loop should return on every attempt")
}

/// Pick two *distinct* account indices uniformly from `0..accounts`.
fn pick_distinct_accounts(rng: &mut XorShift64, accounts: u32) -> (u32, u32) {
    assert!(
        accounts >= 2,
        "BankTransferWorkload requires at least 2 accounts"
    );
    let a = uniform_range_u32(rng, 0, accounts - 1);
    let mut b = uniform_range_u32(rng, 0, accounts.saturating_sub(2));
    if b >= a {
        b += 1;
    }
    (a, b)
}

/// Uniform `u64` in `[lo ..= hi]`. Inclusive on both ends.
fn uniform_range(rng: &mut XorShift64, lo: u64, hi: u64) -> u64 {
    debug_assert!(lo <= hi);
    let span = hi - lo + 1;
    if span == 0 {
        return lo;
    }
    lo + rng.next_u64() % span
}

/// Uniform `u32` in `[lo ..= hi]`. Inclusive on both ends.
fn uniform_range_u32(rng: &mut XorShift64, lo: u32, hi: u32) -> u32 {
    debug_assert!(lo <= hi);
    let span = u64::from(hi - lo) + 1;
    if span == 0 {
        return lo;
    }
    lo + ((rng.next_u64() % span) as u32)
}

/// `MvccKey` for the `idx`-th account.
#[inline]
#[must_use]
pub fn account_key(idx: u32) -> MvccKey {
    ACCOUNT_KEY_BASE.wrapping_add(u64::from(idx))
}

// ──────────────────────────────────────────────────────────────────
// XorShift64 — same shape as `super::super::k1::injection::XorShift64`
// but kept module-local so the two harnesses evolve independently.
// ──────────────────────────────────────────────────────────────────

/// Deterministic 64-bit XorShift PRNG. Marsaglia 2003.
///
/// Not cryptographically secure; not intended to be. Good enough for
/// workload op selection where the only requirement is "different
/// seeds give different sequences, same seed gives the same sequence,
/// distribution is reasonably flat across 64-bit space."
#[derive(Debug)]
pub struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Fallback when caller passes seed=0 (XorShift's degenerate
    /// fixed point). Matches the K-1 convention.
    const SEED_FALLBACK: u64 = 0xDEAD_BEEF_CAFE_F00D;

    /// Construct from a seed. Seed 0 falls back to a non-zero
    /// constant; XorShift64 is otherwise undefined at 0.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { Self::SEED_FALLBACK } else { seed },
        }
    }

    /// Next 64-bit pseudo-random value.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode_balance_roundtrip() {
        for v in [0u64, 1, 100, 1_000_000, u64::MAX] {
            let b = encode_balance(v);
            assert_eq!(decode_balance(&b), Some(v));
        }
    }

    #[test]
    fn decode_balance_rejects_wrong_length() {
        assert_eq!(decode_balance(&[]), None);
        assert_eq!(decode_balance(&[0u8; 4]), None);
        assert_eq!(decode_balance(&[0u8; 16]), None);
    }

    #[test]
    fn pick_distinct_accounts_never_returns_duplicates() {
        let mut rng = XorShift64::new(0xDEAD_BEEF);
        for _ in 0..10_000 {
            let (a, b) = pick_distinct_accounts(&mut rng, 10);
            assert_ne!(a, b);
            assert!(a < 10);
            assert!(b < 10);
        }
    }

    #[test]
    fn pick_distinct_accounts_handles_two_accounts() {
        let mut rng = XorShift64::new(1);
        for _ in 0..1_000 {
            let (a, b) = pick_distinct_accounts(&mut rng, 2);
            assert_ne!(a, b);
            assert!(a < 2);
            assert!(b < 2);
        }
    }

    #[test]
    fn uniform_range_inclusive_bounds() {
        let mut rng = XorShift64::new(7);
        let mut seen_lo = false;
        let mut seen_hi = false;
        for _ in 0..1_000 {
            let v = uniform_range(&mut rng, 1, 5);
            assert!((1..=5).contains(&v));
            if v == 1 {
                seen_lo = true;
            }
            if v == 5 {
                seen_hi = true;
            }
        }
        assert!(seen_lo, "uniform_range never produced lower bound");
        assert!(seen_hi, "uniform_range never produced upper bound");
    }

    #[test]
    fn seed_accounts_sets_expected_initial_state() {
        let mgr = TxnManager::new();
        let cfg = BankTransferConfig {
            accounts: 5,
            initial_balance: 100,
            ..BankTransferConfig::default()
        };
        seed_accounts(&mgr, &cfg, None);
        for i in 0..cfg.accounts {
            let tx = mgr.begin(cfg.tenant);
            let bytes = tx.read(account_key(i)).expect("seeded account present");
            assert_eq!(decode_balance(&bytes), Some(100));
        }
    }

    #[test]
    fn seed_accounts_records_into_history_when_passed() {
        let mgr = TxnManager::new();
        let cfg = BankTransferConfig {
            accounts: 3,
            initial_balance: 50,
            ..BankTransferConfig::default()
        };
        let h = OperationHistory::new();
        seed_accounts(&mgr, &cfg, Some(&h));
        let drained = h.drain_sorted();
        assert_eq!(drained.len(), 3, "one op per seeded account");
        for op in &drained {
            assert_eq!(op.client_id, SEED_CLIENT_ID);
            assert_eq!(op.writes.len(), 1);
        }
    }

    #[test]
    fn expected_sum_is_accounts_times_initial() {
        let cfg = BankTransferConfig {
            accounts: 10,
            initial_balance: 100,
            ..BankTransferConfig::default()
        };
        assert_eq!(expected_sum(&cfg), 1000);
    }

    #[test]
    fn xorshift_same_seed_same_sequence() {
        let mut a = XorShift64::new(0x42);
        let mut b = XorShift64::new(0x42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn xorshift_zero_seed_falls_back() {
        let mut z = XorShift64::new(0);
        let mut f = XorShift64::new(XorShift64::SEED_FALLBACK);
        // Same fallback path → same sequence.
        for _ in 0..10 {
            assert_eq!(z.next_u64(), f.next_u64());
        }
    }

    #[test]
    fn end_to_end_tiny_workload_preserves_sum() {
        // Smoke test inside unit-test scope: a tiny workload should
        // preserve the sum invariant under steady-state operation.
        // The full proptest variant lives in
        // `tests/jepsen_bank_transfer_snapshot.rs`.
        let mgr = Arc::new(TxnManager::new());
        let cfg = BankTransferConfig {
            clients: 2,
            ops_per_client: 20,
            accounts: 4,
            initial_balance: 50,
            max_transfer: 10,
            max_retries: 4,
            seed: 0xCAFE_F00D_DEAD_BEEF,
            tenant: TenantId::DEFAULT,
        };
        let history = Arc::new(OperationHistory::new());
        run_bank_transfer(Arc::clone(&mgr), cfg, Arc::clone(&history));

        // Read every account at the latest snapshot and verify the sum.
        let tx = mgr.begin(cfg.tenant);
        let mut sum: u64 = 0;
        for i in 0..cfg.accounts {
            let b = tx
                .read(account_key(i))
                .and_then(|bs| decode_balance(&bs))
                .unwrap_or(0);
            sum += b;
        }
        assert_eq!(sum, expected_sum(&cfg), "sum invariant violated");
        assert!(
            !history.is_empty(),
            "history should be populated by workers"
        );
    }
}
