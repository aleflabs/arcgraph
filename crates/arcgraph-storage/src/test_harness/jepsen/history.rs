//! Operation history recorder for Jepsen-style isolation testing.
//!
//! ## Shape
//!
//! Each worker thread records a [`RecordedOp`] per transaction
//! it attempts. The op carries enough context for the checker to
//! verify the SI invariant over the *full* history:
//!
//! - `client_id` / `op_id` — origin disambiguation (Elle's "process
//!   ID" / "logical clock" pair).
//! - `start_lsn` — the snapshot LSN the transaction `begin`-ed at
//!   (from [`crate::transaction::Transaction::snapshot`]).
//! - `outcome` — committed (with `commit_lsn`), aborted (with reason),
//!   or pending (the SIGKILL variant may leave a tx with neither commit
//!   nor abort recorded; the checker treats pending ops as "may or may
//!   not have committed" per Jepsen's standard practice).
//! - `reads` / `writes` — every key the transaction read or wrote,
//!   with the observed/intended value. The checker uses these to
//!   verify (a) reads were consistent with the snapshot prefix at
//!   `start_lsn` and (b) writes are observable from any later snapshot
//!   ≥ `commit_lsn`.
//!
//! ## Why a flat `Vec<RecordedOp>` (vs. per-client streams)?
//!
//! Two reasons:
//!
//! 1. The checker needs **global ordering** by commit LSN to
//!    reconstruct the committed prefix at any reader's snapshot.
//!    Per-client streams would require an extra merge pass that
//!    adds nothing.
//! 2. The history is small: 400 ops × ~200 bytes = ~80 KB per run.
//!    A flat vec under a [`Mutex`] is fast enough that the
//!    serialization cost is negligible compared to the transaction
//!    body.
//!
//! ## What this module is NOT
//!
//! - Not an Elle DSL parser. Elle's Clojure DSL ingests EDN; we
//!   record native Rust structs.
//! - Not a persistent history store. The history lives in memory
//!   for the duration of the test and is dropped on completion. The
//!   SIGKILL variant captures a pre-crash history snapshot via the
//!   K-1 ledger pattern (`super::k1::subprocess`), but that lives
//!   under K-1's subprocess module, not here.

use std::sync::Mutex;

use arcgraph_core::{Lsn, TenantId};
use bytes::Bytes;

use crate::transaction::MvccKey;

/// One operation a client attempted. Drives the checker.
///
/// `commit_lsn` is `Some(_)` iff `outcome == OpOutcome::Committed`;
/// the field is a separate `Option<Lsn>` (rather than carried inside
/// `OpOutcome`) so the checker can sort by commit LSN without
/// pattern-matching every op.
#[derive(Debug, Clone)]
pub struct RecordedOp {
    /// Worker-thread origin (0-indexed).
    pub client_id: u32,
    /// Per-client monotonic op sequence number.
    pub op_id: u64,
    /// Tenant the transaction ran under.
    pub tenant: TenantId,
    /// Snapshot LSN captured at [`crate::transaction::TxnManager::begin`].
    pub start_lsn: Lsn,
    /// Commit LSN if the transaction committed; `None` otherwise.
    pub commit_lsn: Option<Lsn>,
    /// Outcome (committed / aborted / pending).
    pub outcome: OpOutcome,
    /// Reads observed during the transaction body, in the order they
    /// were issued. `value` is the bytes the reader saw (`None` =
    /// tombstone / unseen key).
    pub reads: Vec<ObservedRead>,
    /// Writes the transaction buffered. On committed outcomes these
    /// values are durable at `commit_lsn`; on aborted outcomes they
    /// were rolled back.
    pub writes: Vec<IntendedWrite>,
    /// Number of OCC retry attempts before this op concluded. Useful
    /// for "is the workload contention-bound" sanity checks but not
    /// load-bearing on the checker.
    pub retry_count: u32,
}

/// One read issued inside a transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservedRead {
    /// Key read.
    pub key: MvccKey,
    /// Value observed (`None` = key absent / tombstone at the snapshot).
    pub value: Option<Bytes>,
}

/// One write buffered inside a transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntendedWrite {
    /// Key written.
    pub key: MvccKey,
    /// Value buffered (`None` = tombstone / delete).
    pub value: Option<Bytes>,
}

/// Transaction outcome categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpOutcome {
    /// Transaction committed; `commit_lsn` is set.
    Committed,
    /// Transaction aborted (OCC conflict after `retry_count` retries,
    /// or explicit caller abort).
    Aborted,
    /// Transaction was interrupted before commit/abort could be
    /// observed (SIGKILL variant). The checker treats pending ops as
    /// "may or may not be durable"; per ADR-047 the SIGKILL variant
    /// re-reads the committed state post-recovery and reconciles
    /// pending ops against it.
    Pending,
}

/// Append-only history. Thread-safe; workers push from worker threads
/// and the checker reads after all workers join.
///
/// `Mutex<Vec<RecordedOp>>` rather than a lock-free MPMC channel
/// because (a) the ops are pushed at transaction-commit cadence
/// (~µs gap), so the Mutex lock cost is negligible, and (b) the
/// checker wants the full vec at once, not a stream.
#[derive(Debug, Default)]
pub struct OperationHistory {
    ops: Mutex<Vec<RecordedOp>>,
}

impl OperationHistory {
    /// Construct an empty history.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one op. Called from worker threads after each
    /// transaction concludes (commit / abort / SIGKILL).
    pub fn push(&self, op: RecordedOp) {
        // Lock-poison panics here would surface a worker-thread panic
        // immediately; we'd rather a hard panic than silently drop ops.
        let mut guard = self.ops.lock().expect("OperationHistory mutex poisoned");
        guard.push(op);
    }

    /// Number of recorded ops.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops
            .lock()
            .expect("OperationHistory mutex poisoned")
            .len()
    }

    /// Empty?
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drain the history into an owned `Vec<RecordedOp>` for checking.
    /// Leaves the recorder empty so it can be reused. Sorts by
    /// `(commit_lsn.unwrap_or(start_lsn), client_id, op_id)` so the
    /// checker can walk the committed prefix in order.
    pub fn drain_sorted(&self) -> Vec<RecordedOp> {
        let mut ops: Vec<RecordedOp> = {
            let mut guard = self.ops.lock().expect("OperationHistory mutex poisoned");
            std::mem::take(&mut *guard)
        };
        ops.sort_by_key(|op| {
            (
                op.commit_lsn.unwrap_or(op.start_lsn).raw(),
                op.client_id,
                op.op_id,
            )
        });
        ops
    }

    /// Read a snapshot of the history without draining. Useful for
    /// debugging mid-run state without disturbing the recorder.
    /// O(N) clone; only call sparingly.
    #[must_use]
    pub fn snapshot(&self) -> Vec<RecordedOp> {
        self.ops
            .lock()
            .expect("OperationHistory mutex poisoned")
            .clone()
    }
}

/// Convenience builder. Workers construct via the builder, populate
/// reads/writes during the txn body, then push to the history.
///
/// The builder owns its own `Vec<ObservedRead>` and `Vec<IntendedWrite>`
/// so the worker thread doesn't need to allocate per-op outside it.
#[derive(Debug)]
pub struct OpBuilder {
    op: RecordedOp,
}

impl OpBuilder {
    /// Start a new op record.
    #[must_use]
    pub fn new(client_id: u32, op_id: u64, tenant: TenantId, start_lsn: Lsn) -> Self {
        Self {
            op: RecordedOp {
                client_id,
                op_id,
                tenant,
                start_lsn,
                commit_lsn: None,
                outcome: OpOutcome::Pending,
                reads: Vec::new(),
                writes: Vec::new(),
                retry_count: 0,
            },
        }
    }

    /// Record a read observation.
    pub fn observe_read(&mut self, key: MvccKey, value: Option<Bytes>) {
        self.op.reads.push(ObservedRead { key, value });
    }

    /// Record an intended write (buffered, not yet committed).
    pub fn intend_write(&mut self, key: MvccKey, value: Option<Bytes>) {
        self.op.writes.push(IntendedWrite { key, value });
    }

    /// Mark this op as a retry attempt. The checker uses
    /// `retry_count` as a sanity diagnostic, not as part of the
    /// SI predicate.
    pub fn bump_retry(&mut self) {
        self.op.retry_count += 1;
    }

    /// Finalize as committed. Sets `commit_lsn` and `outcome`.
    #[must_use]
    pub fn into_committed(mut self, commit_lsn: Lsn) -> RecordedOp {
        self.op.commit_lsn = Some(commit_lsn);
        self.op.outcome = OpOutcome::Committed;
        self.op
    }

    /// Finalize as aborted. `reason` is stored separately by the
    /// workload (we don't expand `RecordedOp` to carry abort reasons
    /// inline because the common path is committed; the abort-reason
    /// field would be `None` for most ops).
    #[must_use]
    pub fn into_aborted(mut self) -> RecordedOp {
        self.op.outcome = OpOutcome::Aborted;
        self.op
    }

    /// Finalize as pending (SIGKILL variant — used by
    /// [`crate::test_harness::jepsen::fault_injection`]).
    #[must_use]
    pub fn into_pending(self) -> RecordedOp {
        // `outcome` is already `Pending` from `new()`; this is the
        // identity finalization.
        self.op
    }
}

#[cfg(test)]
mod tests {
    use arcgraph_core::TenantId;

    use super::*;

    fn lsn(n: u64) -> Lsn {
        Lsn::new(n)
    }

    #[test]
    fn op_builder_round_trip_committed() {
        let mut b = OpBuilder::new(0, 1, TenantId::DEFAULT, lsn(10));
        b.observe_read(7, Some(Bytes::from_static(b"v")));
        b.intend_write(7, Some(Bytes::from_static(b"v2")));
        let op = b.into_committed(lsn(11));
        assert_eq!(op.client_id, 0);
        assert_eq!(op.op_id, 1);
        assert_eq!(op.start_lsn, lsn(10));
        assert_eq!(op.commit_lsn, Some(lsn(11)));
        assert_eq!(op.outcome, OpOutcome::Committed);
        assert_eq!(op.reads.len(), 1);
        assert_eq!(op.writes.len(), 1);
    }

    #[test]
    fn op_builder_round_trip_aborted() {
        let mut b = OpBuilder::new(3, 2, TenantId::DEFAULT, lsn(5));
        b.bump_retry();
        b.bump_retry();
        let op = b.into_aborted();
        assert_eq!(op.outcome, OpOutcome::Aborted);
        assert!(op.commit_lsn.is_none());
        assert_eq!(op.retry_count, 2);
    }

    #[test]
    fn op_builder_round_trip_pending() {
        let b = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(0));
        let op = b.into_pending();
        assert_eq!(op.outcome, OpOutcome::Pending);
        assert!(op.commit_lsn.is_none());
    }

    #[test]
    fn history_drain_sorted_by_commit_lsn() {
        let h = OperationHistory::new();
        // Push out-of-order: lsn 5, 1, 3.
        h.push(OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(4)).into_committed(lsn(5)));
        h.push(OpBuilder::new(0, 1, TenantId::DEFAULT, lsn(0)).into_committed(lsn(1)));
        h.push(OpBuilder::new(0, 2, TenantId::DEFAULT, lsn(2)).into_committed(lsn(3)));
        let sorted = h.drain_sorted();
        let lsns: Vec<u64> = sorted.iter().map(|o| o.commit_lsn.unwrap().raw()).collect();
        assert_eq!(lsns, vec![1, 3, 5]);
        // Drain leaves the recorder empty.
        assert!(h.is_empty());
    }

    #[test]
    fn history_drain_aborted_sorts_by_start_lsn() {
        let h = OperationHistory::new();
        // Aborted ops have no commit_lsn; sort key falls back to start_lsn.
        h.push(OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(7)).into_aborted());
        h.push(OpBuilder::new(0, 1, TenantId::DEFAULT, lsn(3)).into_aborted());
        let sorted = h.drain_sorted();
        assert_eq!(sorted[0].start_lsn, lsn(3));
        assert_eq!(sorted[1].start_lsn, lsn(7));
    }

    #[test]
    fn history_snapshot_does_not_drain() {
        let h = OperationHistory::new();
        h.push(OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(0)).into_committed(lsn(1)));
        let snap = h.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(h.len(), 1, "snapshot should not drain");
    }
}
