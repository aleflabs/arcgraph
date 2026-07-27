//! Snapshot-isolation checker for recorded operation histories.
//!
//! ## Scope (per ADR-047 §"Decision")
//!
//! This is the v0.1.0-alpha.0+1 foundation, NOT full Elle. It
//! verifies two predicates over a recorded
//! [`super::history::OperationHistory`]:
//!
//! - **Sum-invariant** (Bailis 2014 §3.1): for every committed
//!   prefix in the history, the sum of every account's most-recent
//!   committed balance equals `expected_sum`. Catches any
//!   non-atomic transfer — a commit-atomicity torn write between
//!   `transaction.rs` Phase 2 (WAL append) and Phase 3
//!   (`visible.store`). (G-single write skew on this workload is
//!   not caught; see ADR-047 §"Consequences" for the full list.)
//! - **Per-op visibility consistency**: every observed read in a
//!   committed transaction is consistent with the snapshot prefix
//!   at the reader's `start_lsn`. Specifically, a read of key `k`
//!   returning value `v` must equal the latest-committed value of
//!   `k` whose `commit_lsn ≤ reader.start_lsn` (per
//!   `Version::visible_to` in `transaction.rs`). Catches dirty
//!   reads (G1a), aborted-read (G1b), and a subset of G1c.
//!
//! Full Elle (per-key dependency graphs + cycle detection across
//! WR/WW/RW edges) is the v1.1 deliverable. The current checker is
//! the *strongest* per-op predicate available without that
//! machinery and is sufficient to catch the SI anomalies the
//! bank-transfer workload's overlapping write sets exercise:
//! G1a/G1b, a subset of G1c, and commit-atomicity torn writes. See
//! ADR-047 §"Consequences" for the explicit not-caught list; G-single
//! (write skew) is a v1.1 follow-up workload
//! (`bank_transfer_with_overdraft_invariant`).
//!
//! ## Returning structured verdicts (not panics)
//!
//! The checker returns a [`CheckerVerdict`] enum so the caller can
//! decide how to surface failures. Integration tests typically
//! `assert!(verdict.is_ok())` and `panic!("{verdict}")` on
//! violation; the SIGKILL variant uses the verdict as input to a
//! post-recovery reconciliation step that distinguishes "the
//! recovery dropped a pending op (legal)" from "the recovery
//! corrupted the sum (illegal)."
//!
//! ## Aborted/pending op handling
//!
//! - **Aborted**: their writes never landed; their reads were
//!   speculative; the checker MAY still verify per-op visibility
//!   consistency (a snapshot read should be SI-consistent even if
//!   the txn later aborts) but it does NOT count the writes
//!   toward any committed prefix.
//! - **Pending**: per ADR-047, treated as "may or may not have
//!   committed." The checker skips them by default; the SIGKILL
//!   variant invokes
//!   [`SnapshotIsolationChecker::reconcile_pending_with_recovery`]
//!   to verify the post-recovery state is consistent with one of
//!   the two possible outcomes.

use std::collections::BTreeMap;

use bytes::Bytes;

use super::history::{OpOutcome, RecordedOp};
use super::workload::{ACCOUNT_KEY_BASE, decode_balance};
use crate::transaction::MvccKey;

/// Result of running the checker against a recorded history.
///
/// `Ok` carries summary stats; `Violation` carries enough context to
/// reproduce the failure offline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckerVerdict {
    /// History is SI-legal. Carries summary counts for telemetry.
    Ok(CheckerSummary),
    /// At least one violation was detected. The vec is non-empty.
    Violations(Vec<Violation>),
}

impl CheckerVerdict {
    /// True iff no violations were detected.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, CheckerVerdict::Ok(_))
    }

    /// Convenience: extract the summary on the OK path, or `None`
    /// on violations.
    #[must_use]
    pub fn summary(&self) -> Option<&CheckerSummary> {
        if let CheckerVerdict::Ok(s) = self {
            Some(s)
        } else {
            None
        }
    }

    /// Convenience: borrow the violations on the violation path,
    /// or `None` on OK.
    #[must_use]
    pub fn violations(&self) -> Option<&[Violation]> {
        if let CheckerVerdict::Violations(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

impl std::fmt::Display for CheckerVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckerVerdict::Ok(s) => write!(
                f,
                "SI checker: OK — {} committed, {} aborted, {} pending, {} sum checkpoints",
                s.committed_count, s.aborted_count, s.pending_count, s.sum_checkpoints
            ),
            CheckerVerdict::Violations(vs) => {
                writeln!(f, "SI checker: {} violation(s)", vs.len())?;
                for (i, v) in vs.iter().enumerate() {
                    writeln!(f, "  [{i}] {v}")?;
                }
                Ok(())
            }
        }
    }
}

/// Per-run summary stats for the checker's OK path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CheckerSummary {
    /// Number of committed ops.
    pub committed_count: u64,
    /// Number of aborted ops.
    pub aborted_count: u64,
    /// Number of pending ops (SIGKILL variant only).
    pub pending_count: u64,
    /// Number of distinct commit LSNs at which the sum invariant
    /// was checked.
    pub sum_checkpoints: u64,
}

/// One detected SI violation. Carries enough context to reproduce
/// the failing scenario offline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// The sum of committed balances diverged from `expected_sum`
    /// at some commit LSN.
    SumInvariant {
        /// LSN at which the divergence was detected.
        at_commit_lsn: u64,
        /// Sum observed.
        observed: u64,
        /// Sum expected (from `expected_sum(cfg)`).
        expected: u64,
        /// Per-account balances at the divergence point.
        balances: BTreeMap<MvccKey, u64>,
    },
    /// A committed transaction's read observed a value inconsistent
    /// with the committed prefix at its `start_lsn`.
    VisibilityMismatch {
        /// Client that observed the inconsistency.
        client_id: u32,
        /// Op id.
        op_id: u64,
        /// Snapshot LSN the reader started at.
        reader_snapshot: u64,
        /// Key read.
        key: MvccKey,
        /// Value observed.
        observed: Option<Bytes>,
        /// Value expected per the committed prefix at
        /// `reader_snapshot`.
        expected: Option<Bytes>,
    },
}

impl std::fmt::Display for Violation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Violation::SumInvariant {
                at_commit_lsn,
                observed,
                expected,
                balances,
            } => {
                write!(
                    f,
                    "sum-invariant violated at commit_lsn={at_commit_lsn}: observed={observed}, expected={expected}, balances={balances:?}"
                )
            }
            Violation::VisibilityMismatch {
                client_id,
                op_id,
                reader_snapshot,
                key,
                observed,
                expected,
            } => {
                write!(
                    f,
                    "visibility-mismatch client={client_id} op={op_id} snapshot={reader_snapshot} key={key:#x}: observed={observed:?}, expected={expected:?}"
                )
            }
        }
    }
}

/// Visibility-walk equality oracle for an account key.
///
/// Decodes both sides as `u64` balances per
/// [`super::workload::encode_balance`] and compares those when both
/// decode cleanly. Falls back to the raw `Option<Bytes>` equality
/// otherwise (tombstones and any future non-decodable value), keeping
/// the tombstone (None) case strictly byte-equal. Per R1 F-L7
/// (PR #344): the byte-only oracle is correct today (encoding is
/// stable 8-byte LE u64) but would silently spurious-fail under any
/// future encoding-format drift (e.g., switching to varint); decoding
/// belt-and-suspenders against that.
fn value_matches(expected: &Option<Bytes>, observed: &Option<Bytes>) -> bool {
    match (expected, observed) {
        (Some(e), Some(o)) => {
            // If both decode as balances, compare the u64 values; if
            // either side fails to decode (length mismatch), fall back
            // to byte-equality so future formats still produce a
            // detectable failure.
            match (
                super::workload::decode_balance(e),
                super::workload::decode_balance(o),
            ) {
                (Some(ev), Some(ov)) => ev == ov,
                _ => e == o,
            }
        }
        (None, None) => true,
        // Tombstone-vs-present is always a mismatch.
        _ => false,
    }
}

/// Checker entry point for the bank-transfer workload.
pub struct SnapshotIsolationChecker {
    /// `accounts × initial_balance` from the workload config.
    expected_sum: u64,
    /// Account-key range the checker scopes to. Reads/writes
    /// outside this range are ignored (so the checker composes
    /// cleanly with other workloads sharing the same `TxnManager`).
    account_key_lo: MvccKey,
    account_key_hi: MvccKey,
}

impl SnapshotIsolationChecker {
    /// Build a checker for a bank-transfer workload with `accounts`
    /// accounts each seeded with `initial_balance`.
    ///
    /// Panics if `accounts < 2` — mirrors
    /// [`super::workload::BankTransferConfig`]'s guard in
    /// `pick_distinct_accounts`. With one account the workload has no
    /// meaningful (src, dst) pair, and the checker's key-range
    /// arithmetic (`account_key_hi = base + accounts.saturating_sub(1)`)
    /// would degenerate to `base` only (R1 F-L5).
    #[must_use]
    pub fn for_bank_transfer(accounts: u32, initial_balance: u64) -> Self {
        assert!(
            accounts >= 2,
            "SnapshotIsolationChecker::for_bank_transfer requires at least 2 accounts"
        );
        let expected_sum = u64::from(accounts) * initial_balance;
        let account_key_lo = ACCOUNT_KEY_BASE;
        let account_key_hi = ACCOUNT_KEY_BASE.wrapping_add(u64::from(accounts).saturating_sub(1));
        Self {
            expected_sum,
            account_key_lo,
            account_key_hi,
        }
    }

    /// True iff `key` is in the checker's account-key range.
    #[inline]
    fn is_account_key(&self, key: MvccKey) -> bool {
        key >= self.account_key_lo && key <= self.account_key_hi
    }

    /// Run both predicates over `history` (assumed pre-sorted by
    /// commit LSN per [`super::history::OperationHistory::drain_sorted`]).
    pub fn check(&self, history: &[RecordedOp]) -> CheckerVerdict {
        let mut violations = Vec::new();
        let mut summary = CheckerSummary::default();

        // Phase 1: per-op accounting.
        for op in history {
            match op.outcome {
                OpOutcome::Committed => summary.committed_count += 1,
                OpOutcome::Aborted => summary.aborted_count += 1,
                OpOutcome::Pending => summary.pending_count += 1,
            }
        }

        // Phase 2: sum-invariant walk.
        //
        // Walk committed ops in commit-LSN order. After each commit,
        // recompute the balance per account by replaying every
        // committed write so far. Verify sum.
        let mut balances: BTreeMap<MvccKey, u64> = BTreeMap::new();
        for op in history {
            if op.outcome != OpOutcome::Committed {
                continue;
            }
            for w in &op.writes {
                if !self.is_account_key(w.key) {
                    continue;
                }
                let bal = w.value.as_deref().and_then(decode_balance).unwrap_or(0);
                balances.insert(w.key, bal);
            }
            let observed_sum: u64 = balances.values().sum();
            // Don't check the sum until every seeded account has at
            // least one balance entry (otherwise the partial-state
            // sum is trivially wrong for the seed phase itself).
            // The seed-phase commits push every account, so after the
            // last seed commit `balances.len() == accounts`.
            let accounts_seen = balances.len() as u32;
            let total_accounts = (self.account_key_hi - self.account_key_lo + 1) as u32;
            if accounts_seen < total_accounts {
                continue;
            }
            summary.sum_checkpoints += 1;
            if observed_sum != self.expected_sum {
                violations.push(Violation::SumInvariant {
                    at_commit_lsn: op.commit_lsn.expect("committed op has commit_lsn").raw(),
                    observed: observed_sum,
                    expected: self.expected_sum,
                    balances: balances.clone(),
                });
            }
        }

        // Phase 3: visibility consistency walk.
        //
        // For each committed (or aborted — speculative reads still
        // should be SI-consistent) op, verify every recorded read
        // matches the committed value at the reader's start_lsn.
        //
        // "Committed value at start_lsn" = the value with the
        // largest `commit_lsn ≤ start_lsn` among committed writes
        // to that key.
        //
        // Comparison is **balance-decoded then byte-decoded**: we
        // first attempt to decode both sides as `u64` balances
        // (per [`super::workload::encode_balance`]) and compare those;
        // if either side is `None` (tombstone) the byte-Option
        // equality is the right oracle. Decoding twice belt-and-
        // suspenders against a future encoding format change that
        // might add padding or vary length (R1 F-L7).
        let committed_writes_by_key = self.index_committed_writes(history);
        for op in history {
            if op.outcome == OpOutcome::Pending {
                continue; // SIGKILL ops handled in reconcile_pending_with_recovery
            }
            let snapshot = op.start_lsn.raw();
            for r in &op.reads {
                if !self.is_account_key(r.key) {
                    continue;
                }
                let expected = self.expected_value_at(&committed_writes_by_key, r.key, snapshot);
                if !value_matches(&expected, &r.value) {
                    violations.push(Violation::VisibilityMismatch {
                        client_id: op.client_id,
                        op_id: op.op_id,
                        reader_snapshot: snapshot,
                        key: r.key,
                        observed: r.value.clone(),
                        expected,
                    });
                }
            }
        }

        if violations.is_empty() {
            CheckerVerdict::Ok(summary)
        } else {
            CheckerVerdict::Violations(violations)
        }
    }

    /// Index committed writes by key for `expected_value_at` lookups.
    /// Inner vec is `(commit_lsn, value)` pairs sorted ascending
    /// by `commit_lsn`.
    fn index_committed_writes(
        &self,
        history: &[RecordedOp],
    ) -> BTreeMap<MvccKey, Vec<(u64, Option<Bytes>)>> {
        let mut idx: BTreeMap<MvccKey, Vec<(u64, Option<Bytes>)>> = BTreeMap::new();
        for op in history {
            if op.outcome != OpOutcome::Committed {
                continue;
            }
            let Some(commit_lsn) = op.commit_lsn else {
                continue;
            };
            for w in &op.writes {
                if !self.is_account_key(w.key) {
                    continue;
                }
                idx.entry(w.key)
                    .or_default()
                    .push((commit_lsn.raw(), w.value.clone()));
            }
        }
        // History is already sorted by commit_lsn (per drain_sorted);
        // the per-key vec is therefore already ascending. But the
        // checker is a public API and `history` may come from any
        // source — re-sort defensively.
        for v in idx.values_mut() {
            v.sort_by_key(|(lsn, _)| *lsn);
        }
        idx
    }

    /// Compute the expected value of `key` at `snapshot` per the
    /// indexed committed writes. Returns the value of the
    /// most-recent committed write with `commit_lsn ≤ snapshot`,
    /// or `None` if no such write exists.
    fn expected_value_at(
        &self,
        idx: &BTreeMap<MvccKey, Vec<(u64, Option<Bytes>)>>,
        key: MvccKey,
        snapshot: u64,
    ) -> Option<Bytes> {
        let writes = idx.get(&key)?;
        // Binary search for the largest commit_lsn ≤ snapshot.
        let pos = writes.partition_point(|(lsn, _)| *lsn <= snapshot);
        if pos == 0 {
            None
        } else {
            writes[pos - 1].1.clone()
        }
    }

    /// SIGKILL-variant reconciliation. Given the pre-crash history
    /// and the post-recovery account balances, verify that for every
    /// pending op the recovered state is consistent with EITHER:
    /// - the pending op did NOT commit (so its writes are absent),
    ///   OR
    /// - the pending op DID commit (so its writes are present).
    ///
    /// If the recovered state is consistent with neither, surface a
    /// `SumInvariant` violation (the pending op was partially
    /// applied — torn write, ADR-031 §R3 violation).
    ///
    /// Returns `CheckerVerdict::Ok` if the recovered state is
    /// consistent with one of the two outcomes, otherwise a
    /// violation that flags which pending op produced the conflict.
    ///
    /// Note: at v0.1.0-alpha.0+1 this method is a simple
    /// sum-invariant check on the post-recovery state. v1.1 will
    /// extend with per-key reconciliation once list-append + Elle
    /// land.
    pub fn reconcile_pending_with_recovery(
        &self,
        post_recovery_balances: &BTreeMap<MvccKey, u64>,
    ) -> CheckerVerdict {
        let sum: u64 = post_recovery_balances
            .iter()
            .filter(|(k, _)| self.is_account_key(**k))
            .map(|(_, v)| *v)
            .sum();
        if sum == self.expected_sum {
            CheckerVerdict::Ok(CheckerSummary {
                sum_checkpoints: 1,
                ..CheckerSummary::default()
            })
        } else {
            CheckerVerdict::Violations(vec![Violation::SumInvariant {
                at_commit_lsn: 0, // unknown post-recovery
                observed: sum,
                expected: self.expected_sum,
                balances: post_recovery_balances.clone(),
            }])
        }
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{Lsn, TenantId};

    use super::super::history::{OpBuilder, OperationHistory};
    use super::super::workload::{ACCOUNT_KEY_BASE, encode_balance};
    use super::*;

    fn key(idx: u64) -> MvccKey {
        ACCOUNT_KEY_BASE.wrapping_add(idx)
    }

    /// Build a tiny synthetic history: 2 accounts seeded to 100
    /// each, then one transfer of 30 from acct 0 → acct 1.
    fn happy_path_history() -> Vec<RecordedOp> {
        let h = OperationHistory::new();
        // Seed: account 0 = 100.
        let mut b = OpBuilder::new(0, 0, TenantId::DEFAULT, Lsn::new(0));
        b.intend_write(key(0), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(1)));
        // Seed: account 1 = 100.
        let mut b = OpBuilder::new(0, 1, TenantId::DEFAULT, Lsn::new(1));
        b.intend_write(key(1), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(2)));
        // Transfer: read 100 from each, write 70 + 130.
        let mut b = OpBuilder::new(1, 0, TenantId::DEFAULT, Lsn::new(2));
        b.observe_read(key(0), Some(encode_balance(100)));
        b.observe_read(key(1), Some(encode_balance(100)));
        b.intend_write(key(0), Some(encode_balance(70)));
        b.intend_write(key(1), Some(encode_balance(130)));
        h.push(b.into_committed(Lsn::new(3)));
        h.drain_sorted()
    }

    #[test]
    fn happy_path_history_passes() {
        let checker = SnapshotIsolationChecker::for_bank_transfer(2, 100);
        let v = checker.check(&happy_path_history());
        assert!(v.is_ok(), "happy path should pass: {v}");
        let s = v.summary().expect("ok carries summary");
        assert_eq!(s.committed_count, 3);
        assert_eq!(s.sum_checkpoints, 2);
    }

    #[test]
    fn sum_invariant_violation_detected() {
        // Synthesize a history where a transfer adds 30 to dst but
        // FORGETS to subtract from src — net +30, sum 230 ≠ 200.
        let h = OperationHistory::new();
        let mut b = OpBuilder::new(0, 0, TenantId::DEFAULT, Lsn::new(0));
        b.intend_write(key(0), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(1)));
        let mut b = OpBuilder::new(0, 1, TenantId::DEFAULT, Lsn::new(1));
        b.intend_write(key(1), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(2)));
        let mut b = OpBuilder::new(1, 0, TenantId::DEFAULT, Lsn::new(2));
        b.intend_write(key(1), Some(encode_balance(130))); // only dst, no src
        h.push(b.into_committed(Lsn::new(3)));

        let checker = SnapshotIsolationChecker::for_bank_transfer(2, 100);
        let v = checker.check(&h.drain_sorted());
        let violations = v.violations().expect("must violate");
        assert!(
            matches!(
                violations[0],
                Violation::SumInvariant {
                    observed: 230,
                    expected: 200,
                    ..
                }
            ),
            "got {violations:?}"
        );
    }

    #[test]
    fn visibility_mismatch_detected() {
        // Synthesize: account 0 is committed at LSN 1 with value 100.
        // A reader at snapshot LSN 2 observes value 50. That's not
        // what the committed prefix at LSN 2 says.
        let h = OperationHistory::new();
        let mut b = OpBuilder::new(0, 0, TenantId::DEFAULT, Lsn::new(0));
        b.intend_write(key(0), Some(encode_balance(100)));
        b.intend_write(key(1), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(1)));
        // Reader at snapshot 2 mis-observes acct 0.
        let mut b = OpBuilder::new(1, 0, TenantId::DEFAULT, Lsn::new(2));
        b.observe_read(key(0), Some(encode_balance(50))); // wrong!
        b.observe_read(key(1), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(3)));

        let checker = SnapshotIsolationChecker::for_bank_transfer(2, 100);
        let v = checker.check(&h.drain_sorted());
        let violations = v.violations().expect("must violate");
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, Violation::VisibilityMismatch { .. })),
            "got {violations:?}"
        );
    }

    #[test]
    fn aborted_op_writes_do_not_count() {
        // Aborted op writes 9999 to acct 0; committed reader reads
        // the seeded 100 — both should pass.
        let h = OperationHistory::new();
        let mut b = OpBuilder::new(0, 0, TenantId::DEFAULT, Lsn::new(0));
        b.intend_write(key(0), Some(encode_balance(100)));
        b.intend_write(key(1), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(1)));
        // Aborted op — intends to write 9999, but never committed.
        let mut b = OpBuilder::new(1, 0, TenantId::DEFAULT, Lsn::new(1));
        b.intend_write(key(0), Some(encode_balance(9999)));
        h.push(b.into_aborted());
        // Committed reader sees the seeded 100.
        let mut b = OpBuilder::new(2, 0, TenantId::DEFAULT, Lsn::new(1));
        b.observe_read(key(0), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(2)));

        let checker = SnapshotIsolationChecker::for_bank_transfer(2, 100);
        let v = checker.check(&h.drain_sorted());
        assert!(v.is_ok(), "aborted writes must not affect sum: {v}");
    }

    #[test]
    fn reconcile_pending_ok_when_sum_holds() {
        let checker = SnapshotIsolationChecker::for_bank_transfer(3, 100);
        let mut bal = BTreeMap::new();
        bal.insert(key(0), 70);
        bal.insert(key(1), 130);
        bal.insert(key(2), 100);
        let v = checker.reconcile_pending_with_recovery(&bal);
        assert!(v.is_ok());
    }

    #[test]
    fn reconcile_pending_violation_when_sum_breaks() {
        let checker = SnapshotIsolationChecker::for_bank_transfer(2, 100);
        let mut bal = BTreeMap::new();
        bal.insert(key(0), 50);
        bal.insert(key(1), 100); // torn: should be 150, got 100 — sum 150 ≠ 200
        let v = checker.reconcile_pending_with_recovery(&bal);
        let violations = v.violations().expect("must violate");
        assert!(matches!(
            violations[0],
            Violation::SumInvariant {
                observed: 150,
                expected: 200,
                ..
            }
        ));
    }

    #[test]
    fn pending_ops_ignored_by_visibility_walk() {
        // A pending op's reads are NOT checked (per ADR-047 — the
        // SIGKILL variant uses reconcile_pending_with_recovery
        // instead). Confirms the walk skips them.
        let h = OperationHistory::new();
        let mut b = OpBuilder::new(0, 0, TenantId::DEFAULT, Lsn::new(0));
        b.intend_write(key(0), Some(encode_balance(100)));
        b.intend_write(key(1), Some(encode_balance(100)));
        h.push(b.into_committed(Lsn::new(1)));
        let mut b = OpBuilder::new(1, 0, TenantId::DEFAULT, Lsn::new(2));
        b.observe_read(key(0), Some(encode_balance(999))); // would fail if checked
        let pending = b.into_pending();
        h.push(pending);
        let checker = SnapshotIsolationChecker::for_bank_transfer(2, 100);
        let v = checker.check(&h.drain_sorted());
        // No visibility violations should fire because the pending
        // op is skipped.
        let vio = v.violations();
        assert!(
            vio.is_none()
                || !vio
                    .unwrap()
                    .iter()
                    .any(|x| matches!(x, Violation::VisibilityMismatch { .. })),
            "pending op should not produce visibility violations: {v}"
        );
    }
}
