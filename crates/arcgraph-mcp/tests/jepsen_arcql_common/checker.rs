//! Snapshot-isolation anomaly checker for recorded ArcQL operation
//! histories (W27-ν / ADR-163).
//!
//! The checker is the **oracle** (per the determinism contract in the
//! parent module): it runs predicates over the *whole* recorded
//! history rather than comparing against a reference snapshot. It
//! implements the Adya 2000 §4 / Bailis 2014 §3 snapshot-isolation
//! anomaly taxonomy at node-identity granularity:
//!
//! - **G0 (dirty write)** — a torn / unreadable node record at a
//!   committed snapshot ([`ArcqlViolation::TornRead`]).
//! - **Snapshot-read consistency** — the load-bearing predicate. Every
//!   committed MATCH's observed node-set must equal the committed
//!   prefix visible at its snapshot LSN. Catches G1a (a read of an
//!   aborted/uncommitted CREATE), G1b (a partial read of a multi-write
//!   transaction), and any snapshot violation
//!   ([`ArcqlViolation::SnapshotRead`]). Directly analogous to the
//!   ADR-047 bank-transfer checker's per-op visibility predicate.
//! - **G1a (aborted read)** — an explicit, separately-counted witness:
//!   no MATCH observes a node burned by an aborted CREATE
//!   ([`ArcqlViolation::AbortedReadObserved`]).
//! - **G1c (circular information flow)** — the ww∪wr direct
//!   serialization graph (Adya 2000 §4.2) must be acyclic
//!   ([`ArcqlViolation::G1cCycle`]). This is an independent second
//!   witness of the same SI property the snapshot-read predicate
//!   verifies, via the dependency-graph formalism.
//! - **G2-item (write skew)** — counted as an informational witness,
//!   **NOT** a violation. Snapshot isolation does not forbid write
//!   skew (Adya 2000 §4.3 / Berenson 1995 A5B); the boundary between
//!   SI and serializability *is* write skew. The G2 workload therefore
//!   demonstrates the surface is SI (skew is observable) while
//!   exhibiting no G1c.
//! - **Lost update (RMW on a single counter)** — W27-ν-2 write-side
//!   activation (ADR-163 §FD-1). When the harness records a counter
//!   key whose committed RMW writes carry the post-increment value as
//!   8-byte big-endian bytes (see [`encode_counter`]), the checker
//!   verifies the **maximum surviving committed value == seed + count
//!   of committed increments on that key**
//!   ([`ArcqlViolation::LostUpdate`]). A lost update (two RMWs both
//!   commit but the final value reflects only one — Bailis 2014 §3.1)
//!   is exactly the case a correct SI/OCC kernel forbids (the second
//!   writer loses the WW race), so this predicate is the
//!   property-level analog of the read-side G0 dirty-write predicate.
//!   It is **opt-in**: a history with no counter-tagged writes is
//!   unaffected — the read-side workloads record no counter key and so
//!   skip the predicate entirely (it runs only when
//!   [`encode_counter`]-tagged writes are present).
//!
//! What this checker is NOT: full Elle (Kingsbury & Alvaro VLDB 2020).
//! List-append + per-key dependency graphs with rw-anti-dependency
//! cycle classification are forward-deferred to v1.1 alongside the
//! storage-layer Elle work (ADR-047 §"Open questions" / ADR-163
//! §"Forward-deferred").

use std::collections::HashMap;

use arcgraph_core::Lsn;
use arcgraph_storage::test_harness::jepsen::history::{OpOutcome, RecordedOp};
use bytes::Bytes;

use super::{POISONED_READ_MARKER, SCAN_SENTINEL_KEY, is_match_op};

/// Encode a counter value as the canonical 8-byte big-endian payload
/// the lost-update predicate decodes (W27-ν-2 / ADR-163 §FD-1). The
/// write recorders for the lost-update workload pass this as the
/// `IntendedWrite::value` so the checker can reconstruct the surviving
/// value per counter key WITHOUT reaching into storage. Big-endian so
/// the byte order sorts identically to the numeric order (handy when
/// eyeballing a printed history).
#[must_use]
pub fn encode_counter(value: u64) -> Bytes {
    Bytes::copy_from_slice(&value.to_be_bytes())
}

/// Decode a counter value previously written via [`encode_counter`].
/// Returns `None` for any payload that is not exactly 8 bytes (e.g. a
/// `present_marker()` node-existence write, which the lost-update
/// predicate must skip — only counter-tagged writes participate).
#[must_use]
pub fn decode_counter(value: &Bytes) -> Option<u64> {
    let bytes: [u8; 8] = value.as_ref().try_into().ok()?;
    Some(u64::from_be_bytes(bytes))
}

/// Canonical **edge-identity** write key for `(src, type, dst)` (S7d-3 /
/// ADR-182 §2.2). The recovery-reconciliation predicate
/// ([`ArcqlSiChecker::reconcile_arcql_pending_with_recovery`]) reasons
/// over an opaque `u64` write-key domain (the same `MvccKey = u64` the
/// rest of the harness uses); to give the *edge* half of a `MERGE
/// (a)-[:R]->(b)` its own per-key identity — distinct from either
/// endpoint's `NodeId::raw()` node key — the synthetic crash histories
/// (and the future S7d-2 SIGKILL-during-MERGE workload) tag an edge write
/// with this key instead of a raw node id.
///
/// # Namespace (collision-free with node keys + the reserved sentinels)
///
/// The returned key always has **bit 63 set** (`>= 1 << 63`). Node ids
/// are allocated from 1 upward and stay in the low half of the u64 space
/// (`< 1 << 63`) for any realistic workload, so an edge key can never
/// alias a node key. The two reserved harness sentinels
/// ([`super::SCAN_SENTINEL_KEY`] `= u64::MAX` and
/// [`super::POISONED_READ_MARKER`] `= u64::MAX - 1`) occupy the very top
/// of the space; this encoder masks the result down so it can never
/// land on either (see the `min` clamp below), keeping the edge-key
/// domain disjoint from both node keys and the read-side markers.
///
/// The mix is a deterministic FxHash-style fold (no `rand`, no new dep —
/// matching the harness's zero-dep convention); it is *not* a
/// cryptographic hash and is used only to derive distinct test keys, so a
/// fast integer mix is appropriate (the predicate never inverts it).
#[must_use]
pub fn edge_key(src: u64, ty: u64, dst: u64) -> u64 {
    // FxHash-style multiplicative fold (the constant is the standard
    // rustc-hash 64-bit seed). Deterministic, well-distributed enough to
    // separate distinct (src,ty,dst) triples for hand-built histories.
    const K: u64 = 0x51_7c_c1_b7_27_22_0a_95;
    let mut h = 0u64;
    for v in [src, ty, dst] {
        h = (h.rotate_left(5) ^ v).wrapping_mul(K);
    }
    // Force into the high half (bit 63 set) so the key is disjoint from
    // node ids, and clamp strictly below the two reserved top sentinels
    // (`u64::MAX` and `u64::MAX - 1`) so it can never alias them.
    let high = h | (1u64 << 63);
    high.min(u64::MAX - 2)
}

/// Outcome of running the checker against a drained history.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArcqlVerdict {
    /// History is SI-legal. Carries summary counts for telemetry.
    Ok(ArcqlSummary),
    /// At least one violation was detected (the vec is non-empty).
    Violations(Vec<ArcqlViolation>),
}

impl ArcqlVerdict {
    /// True iff no violations were detected.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        matches!(self, ArcqlVerdict::Ok(_))
    }

    /// Borrow the summary on the OK path.
    #[must_use]
    pub fn summary(&self) -> Option<&ArcqlSummary> {
        match self {
            ArcqlVerdict::Ok(s) => Some(s),
            ArcqlVerdict::Violations(_) => None,
        }
    }

    /// Borrow the violations on the violation path.
    #[must_use]
    pub fn violations(&self) -> Option<&[ArcqlViolation]> {
        match self {
            ArcqlVerdict::Violations(v) => Some(v),
            ArcqlVerdict::Ok(_) => None,
        }
    }
}

impl std::fmt::Display for ArcqlVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArcqlVerdict::Ok(s) => write!(
                f,
                "ArcQL SI checker: OK — {} committed / {} aborted; {} MATCH ops ({} snapshot-checks); \
                 {} aborted-write ids tracked; {} write-skew witnesses (SI-permitted); \
                 {} counter(s) lost-update-checked",
                s.committed,
                s.aborted,
                s.match_ops,
                s.snapshot_checks,
                s.aborted_writes,
                s.writeskew_witnesses,
                s.counters_checked,
            ),
            ArcqlVerdict::Violations(vs) => {
                writeln!(f, "ArcQL SI checker: {} violation(s)", vs.len())?;
                for (i, v) in vs.iter().enumerate() {
                    writeln!(f, "  [{i}] {v}")?;
                }
                Ok(())
            }
        }
    }
}

/// Per-run summary stats (OK path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ArcqlSummary {
    pub committed: usize,
    pub aborted: usize,
    pub match_ops: usize,
    pub write_ops: usize,
    /// Number of committed MATCH ops whose read-set was checked against
    /// the computed committed prefix.
    pub snapshot_checks: usize,
    /// Number of node ids burned by aborted CREATEs (G1a tracking set
    /// size). Non-zero confirms the abort-injection workload actually
    /// injected faults.
    pub aborted_writes: usize,
    /// Number of write-skew witness pairs (SI-permitted; informational).
    pub writeskew_witnesses: usize,
    /// Number of distinct counter keys whose lost-update invariant was
    /// checked (W27-ν-2). Zero on every read-side history (no counter
    /// writes recorded); non-zero confirms the write-side lost-update
    /// workload actually tagged increments.
    pub counters_checked: usize,
}

/// A detected SI violation, carrying enough context to reproduce.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArcqlViolation {
    /// A MATCH observed a torn / unreadable node record (G0).
    TornRead { client_id: u32, op_id: u64 },
    /// A MATCH's observed node-set diverged from the committed prefix
    /// visible at its snapshot LSN.
    SnapshotRead {
        client_id: u32,
        op_id: u64,
        start_lsn: u64,
        /// Nodes that should have been visible but were not observed.
        missing: Vec<u64>,
        /// Nodes that were observed but should not have been visible.
        extra: Vec<u64>,
    },
    /// A MATCH observed a node burned by an aborted CREATE (explicit
    /// G1a witness).
    AbortedReadObserved {
        reader_client: u32,
        reader_op: u64,
        node: u64,
    },
    /// The ww∪wr dependency graph contains a cycle (G1c — circular
    /// information flow, forbidden under SI). Carries the op identities
    /// `(client_id, op_id)` on the cycle.
    G1cCycle { ops: Vec<(u32, u64)> },
    /// A lost update on a counter key (W27-ν-2 / ADR-163 §FD-1): the
    /// maximum committed counter value disagrees with `seed +
    /// committed_increments`, i.e. at least one committed increment was
    /// overwritten without being applied (Bailis 2014 §3.1). A correct
    /// SI/OCC kernel never produces this (the losing writer aborts).
    LostUpdate {
        /// The counter node key.
        key: u64,
        /// Number of committed increment ops observed on this key.
        committed_increments: u64,
        /// The maximum committed value actually recorded for this key.
        final_value: u64,
        /// The value `final_value` SHOULD have held if no increment was
        /// lost (`seed + committed_increments`).
        expected_value: u64,
    },
    /// A **torn multi-statement commit observed post-recovery** (S7d-3 /
    /// ADR-182 §2.2). An op (typically a `MERGE` that should atomically
    /// create a node AND its edge) was reconciled against the recovered
    /// committed state and found *half-applied*: SOME of its writes
    /// survived recovery while OTHERS did not. ADR-031 §Decision
    /// ("Record-level commit atomicity — no partial-commit state is ever
    /// observable post-restart") forbids this: a `CommitBundle` is either
    /// fully durable or fully torn-and-dropped, so a correct
    /// ArcQL→MVCC→WAL→recovery path makes this state unreachable. The
    /// predicate fails loudly if it ever manifests (e.g. a regression that
    /// splits a MERGE across two bundles, or a recovery bug that replays a
    /// bundle partially).
    ///
    /// This is the ArcQL/per-key analog of the storage-level
    /// `Violation::SumInvariant` "torn write" (the bank-transfer
    /// reconciliation predicate
    /// `arcgraph_storage::test_harness::jepsen::checker::reconcile_pending_with_recovery`,
    /// whose doc-comment names this lift: *"v1.1 will extend with per-key
    /// reconciliation"*). Granularity is per write-key, where a node key
    /// is its `NodeId::raw()` and an edge key is [`edge_key`]`(src, ty,
    /// dst)` — so `present`/`absent` enumerate exactly which graph
    /// elements of the op survived vs vanished.
    PartialMergeCommit {
        /// The torn op's identity `(client_id, op_id)`.
        op: (u32, u64),
        /// The op's write keys that ARE present in the recovered state
        /// (sorted ascending for a deterministic witness).
        present: Vec<u64>,
        /// The op's write keys that are ABSENT from the recovered state
        /// (sorted ascending). A non-empty `present` AND a non-empty
        /// `absent` together witness the torn (half-committed) state.
        absent: Vec<u64>,
    },
    /// **Durability loss of an acked commit** (S7d-3 / ADR-182 §2.2
    /// bullet 1) — the single most severe crash-atomicity violation. An op
    /// that was `Committed` with `commit_lsn ≤ watermark` (i.e. its WAL
    /// fsync completed and the commit was *acked* to the client, ADR-034
    /// §Slice-B "commit is durable before ack") had its **ENTIRE** write-set
    /// vanish post-recovery (`present = []`). ADR-031 §Decision +
    /// ADR-183 §R2 require an acked commit to survive recovery in full; a
    /// total loss is a durability violation (the client was told "durable",
    /// recovery says "never happened"). This is the *all-absent* leg of
    /// bullet 1 — distinct from [`Self::PartialMergeCommit`] (the
    /// *some-absent* straddle leg): both are bullet-1 violations, but the
    /// total-loss class names the worst case explicitly so a future
    /// `verdict.is_ok()` over an acked-loss history fails loudly instead of
    /// passing silently.
    AckedCommitLoss {
        /// The lost op's identity `(client_id, op_id)`.
        op: (u32, u64),
        /// The acked commit LSN that was at or below the recovery
        /// watermark (so the op was durable-before-ack and MUST survive).
        commit_lsn: u64,
        /// The recovery watermark the op's `commit_lsn` was compared
        /// against (`commit_lsn ≤ watermark` ⟹ acked-durable).
        watermark: u64,
        /// The op's created write keys, ALL of which are absent from the
        /// recovered state (sorted ascending for a deterministic witness).
        lost: Vec<u64>,
    },
    /// **Phantom commit of a non-acked op** (S7d-3 / ADR-182 §2.2 bullet 2)
    /// — an op that was `Aborted`, or `Pending` (SIGKILL'd before
    /// commit/abort), or `Committed` but PAST the watermark (`commit_lsn >
    /// watermark` — it lost the ack race), had SOME of its writes *appear*
    /// in the recovered state (`present` is non-empty). Bullet 2 says none
    /// of a non-acked op's writes may be present post-recovery: an aborted
    /// transaction's writes must have been rolled back, and a
    /// past-watermark/never-acked commit must be as-if-absent. A surviving
    /// write is a phantom — recovery resurrected state that was never
    /// durably committed (the dual of [`Self::AckedCommitLoss`]).
    PhantomCommit {
        /// The phantom op's identity `(client_id, op_id)`.
        op: (u32, u64),
        /// Why the op was non-acked: `"aborted"`, `"pending"`, or
        /// `"committed-past-watermark"` — names the bullet-2 sub-case.
        reason: &'static str,
        /// The op's write keys that ARE present in the recovered state
        /// despite the op never being acked (sorted ascending). Non-empty
        /// witnesses the phantom.
        present: Vec<u64>,
    },
}

impl std::fmt::Display for ArcqlViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArcqlViolation::TornRead { client_id, op_id } => write!(
                f,
                "G0 torn-read: client {client_id} op {op_id} observed an unreadable node record"
            ),
            ArcqlViolation::SnapshotRead {
                client_id,
                op_id,
                start_lsn,
                missing,
                extra,
            } => write!(
                f,
                "snapshot-read violation: client {client_id} op {op_id} @ start_lsn={start_lsn} \
                 missing={missing:?} extra={extra:?}"
            ),
            ArcqlViolation::AbortedReadObserved {
                reader_client,
                reader_op,
                node,
            } => write!(
                f,
                "G1a aborted-read: client {reader_client} op {reader_op} observed aborted node {node}"
            ),
            ArcqlViolation::G1cCycle { ops } => {
                write!(f, "G1c cycle in ww∪wr dependency graph: ")?;
                for (i, (c, o)) in ops.iter().enumerate() {
                    if i > 0 {
                        write!(f, " → ")?;
                    }
                    write!(f, "(c{c},op{o})")?;
                }
                Ok(())
            }
            ArcqlViolation::LostUpdate {
                key,
                committed_increments,
                final_value,
                expected_value,
            } => write!(
                f,
                "lost update on counter key {key}: {committed_increments} increment(s) committed \
                 but final value is {final_value} (expected {expected_value}); \
                 {} increment(s) were lost",
                expected_value.saturating_sub(*final_value)
            ),
            ArcqlViolation::PartialMergeCommit {
                op: (client_id, op_id),
                present,
                absent,
            } => write!(
                f,
                "partial-MERGE-commit (torn multi-statement commit, ADR-031 §Decision violation): \
                 client {client_id} op {op_id} survived recovery half-applied — \
                 present keys {present:?} but absent keys {absent:?}"
            ),
            ArcqlViolation::AckedCommitLoss {
                op: (client_id, op_id),
                commit_lsn,
                watermark,
                lost,
            } => write!(
                f,
                "acked-commit-loss (durability loss of an acked commit, ADR-182 §2.2 bullet 1 / \
                 ADR-031 §Decision / ADR-034 §Slice-B violation): client {client_id} op {op_id} \
                 committed at lsn {commit_lsn} ≤ watermark {watermark} (durable-before-ack) but \
                 its ENTIRE write-set vanished post-recovery — lost keys {lost:?}"
            ),
            ArcqlViolation::PhantomCommit {
                op: (client_id, op_id),
                reason,
                present,
            } => write!(
                f,
                "phantom-commit (non-acked op's writes appeared post-recovery, ADR-182 §2.2 \
                 bullet 2 violation): client {client_id} op {op_id} was {reason} yet \
                 present keys {present:?} survived recovery"
            ),
        }
    }
}

/// The ArcQL snapshot-isolation checker.
#[derive(Debug, Default, Clone, Copy)]
pub struct ArcqlSiChecker;

impl ArcqlSiChecker {
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Run every predicate over `ops` (expected to be drained +
    /// sorted by commit LSN via `OperationHistory::drain_sorted`).
    #[must_use]
    pub fn check(&self, ops: &[RecordedOp]) -> ArcqlVerdict {
        let mut violations = Vec::new();
        let mut summary = ArcqlSummary::default();

        // ── Index committed writes: creator (value Some) + deleter
        //    (value None) per node id. Each id is created at most once.
        let mut creator_lsn: HashMap<u64, u64> = HashMap::new();
        let mut creator_idx: HashMap<u64, usize> = HashMap::new();
        let mut deleter_lsn: HashMap<u64, u64> = HashMap::new();
        // Aborted-write tracking set (G1a): node ids burned by an
        // aborted op's CREATE.
        let mut aborted_nodes: HashMap<u64, ()> = HashMap::new();

        for (idx, op) in ops.iter().enumerate() {
            match op.outcome {
                OpOutcome::Committed => summary.committed += 1,
                OpOutcome::Aborted => summary.aborted += 1,
                OpOutcome::Pending => {}
            }
            let is_match = is_match_op(op);
            if is_match {
                summary.match_ops += 1;
            } else if !op.writes.is_empty() {
                summary.write_ops += 1;
            }

            match op.outcome {
                OpOutcome::Committed => {
                    if let Some(commit) = op.commit_lsn {
                        for w in &op.writes {
                            if w.value.is_some() {
                                creator_lsn.insert(w.key, commit.raw());
                                creator_idx.insert(w.key, idx);
                            } else {
                                deleter_lsn.insert(w.key, commit.raw());
                            }
                        }
                    }
                }
                OpOutcome::Aborted | OpOutcome::Pending => {
                    for w in &op.writes {
                        if w.value.is_some() {
                            aborted_nodes.insert(w.key, ());
                        }
                    }
                }
            }
        }
        summary.aborted_writes = aborted_nodes.len();

        // ── Predicate 1 (G0): poisoned / torn reads.
        for op in ops {
            if op.reads.iter().any(|r| r.key == POISONED_READ_MARKER) {
                violations.push(ArcqlViolation::TornRead {
                    client_id: op.client_id,
                    op_id: op.op_id,
                });
            }
        }

        // ── Predicate 2 (snapshot-read consistency) + Predicate 3 (G1a).
        for op in ops {
            if !is_match_op(op) || op.outcome != OpOutcome::Committed {
                continue;
            }
            let start = op.start_lsn.raw();
            summary.snapshot_checks += 1;

            // Observed: every read key except the sentinel + the
            // torn-read marker (handled by predicate 1).
            let mut observed: Vec<u64> = op
                .reads
                .iter()
                .map(|r| r.key)
                .filter(|&k| k != SCAN_SENTINEL_KEY && k != POISONED_READ_MARKER)
                .collect();
            observed.sort_unstable();
            observed.dedup();

            // Expected committed prefix at `start`: created at or before
            // `start` AND not deleted at or before `start`.
            let mut expected: Vec<u64> = creator_lsn
                .iter()
                .filter(|&(_, &clsn)| clsn <= start)
                .map(|(&k, _)| k)
                .filter(|k| match deleter_lsn.get(k) {
                    Some(&dlsn) => dlsn > start,
                    None => true,
                })
                .collect();
            expected.sort_unstable();

            if observed != expected {
                let missing: Vec<u64> = expected
                    .iter()
                    .filter(|k| observed.binary_search(k).is_err())
                    .copied()
                    .collect();
                let extra: Vec<u64> = observed
                    .iter()
                    .filter(|k| expected.binary_search(k).is_err())
                    .copied()
                    .collect();
                violations.push(ArcqlViolation::SnapshotRead {
                    client_id: op.client_id,
                    op_id: op.op_id,
                    start_lsn: start,
                    missing,
                    extra,
                });
            }

            // Explicit G1a: no observed node may be an aborted-burned id.
            for k in &observed {
                if aborted_nodes.contains_key(k) {
                    violations.push(ArcqlViolation::AbortedReadObserved {
                        reader_client: op.client_id,
                        reader_op: op.op_id,
                        node: *k,
                    });
                }
            }
        }

        // ── Counter keys (repeatedly-SET multi-version keys) are owned
        //    by the lost-update predicate, NOT the node-identity G1c /
        //    write-skew predicates. Identify them once (any key with an
        //    8-byte counter payload on a committed write).
        let counter_keys = counter_keys_of(ops);

        // ── Predicate 4 (G1c): ww∪wr dependency graph acyclicity over
        //    the node-identity domain (counter keys excluded — see
        //    `detect_g1c_cycle` domain-boundary note).
        if let Some(cycle) = detect_g1c_cycle(ops, &creator_idx, &counter_keys) {
            violations.push(ArcqlViolation::G1cCycle {
                ops: cycle
                    .iter()
                    .map(|&i| (ops[i].client_id, ops[i].op_id))
                    .collect(),
            });
        }

        // ── G2-item write-skew witnesses (informational; SI-permitted).
        summary.writeskew_witnesses = count_write_skew_witnesses(ops, &counter_keys);

        // ── Predicate 5 (lost update): for each counter key, the max
        //    committed value must equal seed + #committed increments
        //    (W27-ν-2 / ADR-163 §FD-1). Opt-in: only runs on keys whose
        //    committed writes carry an 8-byte counter payload.
        let (counters_checked, lost_updates) = check_lost_updates(ops);
        summary.counters_checked = counters_checked;
        violations.extend(lost_updates);

        if violations.is_empty() {
            ArcqlVerdict::Ok(summary)
        } else {
            ArcqlVerdict::Violations(violations)
        }
    }

    /// **Recovery-reconciliation predicate** (S7d-3 / ADR-182 §2.2). Given
    /// the pre-crash recorded history `history` (`H_pre`), the recovery
    /// `watermark` (the `committed_fsync_watermark` boundary, ADR-034
    /// §Slice-B), and the set of write keys present in the post-recovery
    /// committed state `recovered` (`S_rec`), verify that **every op
    /// committed all-or-nothing *per its outcome and its position relative
    /// to the watermark*** — the **outcome- and watermark-sensitive**
    /// ArcQL/MVCC-level statement of ADR-031's bundle-atomicity guarantee
    /// ("no partial-commit state is ever observable post-restart").
    ///
    /// # The predicate (per op `o ∈ H_pre`, over its created write-key set)
    ///
    /// Let `W(o)` be the set of keys `o` *created* (its `IntendedWrite`s
    /// with `value.is_some()` — a `MERGE (a)-[:R]->(b)` contributes
    /// `{ a, b, edge_key(a,R,b) }`). `o` is **acked-durable** iff
    /// `o.outcome == Committed` AND `o.commit_lsn ≤ watermark` (its WAL
    /// fsync completed and the commit was acked to the client, ADR-034
    /// §Slice-B "commit is durable before ack"). Partition `W(o)` by
    /// membership in `recovered` and branch on whether `o` is acked-durable:
    ///
    /// **ADR-182 §2.2 bullet 1 — `o` acked-durable (`Committed{lsn ≤
    /// watermark}`):** EVERY write of `o` MUST be present in `S_rec`.
    ///   - all present ⟹ legal (the whole acked bundle survived).
    ///   - some present AND some absent ⟹ [`ArcqlViolation::PartialMergeCommit`]
    ///     (the torn / half-applied straddle ADR-031 forbids).
    ///   - **ALL absent ⟹ [`ArcqlViolation::AckedCommitLoss`]** — the most
    ///     severe class: durability loss of an *acked* commit (the client
    ///     was told "durable", recovery says "never happened"). This is the
    ///     arm the outcome-blind predecessor was missing.
    ///
    /// **ADR-182 §2.2 bullet 2 — `o` non-acked (`Aborted`, OR `Pending`
    /// SIGKILL'd before commit/abort, OR `Committed{lsn > watermark}` —
    /// lost the ack race):** NONE of `o`'s writes may be present in `S_rec`.
    ///   - none present ⟹ legal (an aborted bundle was rolled back; a
    ///     past-watermark commit is legitimately as-if-absent).
    ///   - any present ⟹ [`ArcqlViolation::PhantomCommit`] (recovery
    ///     resurrected state that was never durably acked — the dual of
    ///     `AckedCommitLoss`).
    ///
    /// **ADR-182 §2.2 bullet 3 ("anything else"):** subsumed by the two
    /// bullets above — the straddle (`PartialMergeCommit`) is bullet 1's
    /// some-absent leg.
    ///
    /// The predicate is **key-domain-agnostic** over the write keys
    /// (per-`(node_id)` ∪ per-`(src,type,dst)` granularity: nodes carry
    /// `NodeId::raw()`, edges carry [`edge_key`], so `present` / `absent`
    /// enumerate exactly which graph elements survived) but **outcome- and
    /// watermark-sensitive** over each op (the legality of an all-absent or
    /// some-present recovered state depends on `o.outcome` + `o.commit_lsn`
    /// vs `watermark`, per ADR-182 §2.2).
    ///
    /// ## Watermark source
    ///
    /// The in-process predicate uses the **history-declared commit
    /// watermark** passed by the caller: the hand-built adversarial
    /// histories in the non-vacuity self-tests control each op's
    /// `commit_lsn` and the watermark directly, so the bullet-1/bullet-2
    /// distinction is fully expressible in-test today. The forward path for
    /// the live SIGKILL-during-MERGE workload (S7d-2) is executor-tier
    /// commit-LSN observability (ADR-182 §Open-questions FD-2 / ADR-163) so
    /// the recovered store's actual `committed_fsync_watermark` can be read
    /// back; the durable fixture's crud-tier handle already records the
    /// per-op `commit_lsn`, so S7d-2 supplies the same `watermark` argument
    /// from the recovered WAL boundary. The predicate does NOT block on
    /// FD-2: the watermark is a plain `Lsn` argument the caller chooses.
    ///
    /// Ops with no created writes (a read-only `MATCH`, or a pure
    /// tombstone/delete op) contribute nothing and are skipped — the
    /// predicate is the multi-statement-*creation*-atomicity checker
    /// (`MERGE`), not a delete checker.
    ///
    /// Lift of the storage-level sum-invariant reconciliation
    /// (`arcgraph_storage::test_harness::jepsen::checker::reconcile_pending_with_recovery`)
    /// to the ArcQL per-key level, extended with the outcome+watermark
    /// fidelity ADR-182 §2.2 specifies (the storage predicate is
    /// outcome-blind; this one is not).
    #[must_use]
    pub fn reconcile_arcql_pending_with_recovery(
        &self,
        history: &[RecordedOp],
        watermark: Lsn,
        recovered: &std::collections::HashSet<u64>,
    ) -> ArcqlVerdict {
        let mut summary = ArcqlSummary::default();
        let mut violations = Vec::new();

        for op in history {
            match op.outcome {
                OpOutcome::Committed => summary.committed += 1,
                OpOutcome::Aborted => summary.aborted += 1,
                OpOutcome::Pending => {}
            }

            // The op's CREATED keys: present-marker writes (value Some).
            // A tombstone (value None) is a delete, not a MERGE creation,
            // and is excluded from the all-or-nothing creation check.
            let mut created: Vec<u64> = op
                .writes
                .iter()
                .filter(|w| w.value.is_some())
                .map(|w| w.key)
                .collect();
            created.sort_unstable();
            created.dedup();
            if created.is_empty() {
                continue; // read-only / delete-only op — nothing to reconcile.
            }

            let (present, absent): (Vec<u64>, Vec<u64>) =
                created.into_iter().partition(|k| recovered.contains(k));

            // An op is acked-durable iff it committed at or below the
            // recovery watermark (its fsync completed → commit was acked,
            // ADR-034 §Slice-B). `commit_lsn` is `Some` iff `Committed`.
            let acked_durable = op.outcome == OpOutcome::Committed
                && op.commit_lsn.is_some_and(|lsn| lsn <= watermark);

            if acked_durable {
                // ── ADR-182 §2.2 bullet 1: EVERY write must be present.
                if absent.is_empty() {
                    // all present ⟹ legal (whole acked bundle survived).
                } else if present.is_empty() {
                    // ALL absent ⟹ durability loss of an acked commit —
                    // the single most severe crash-atomicity violation.
                    violations.push(ArcqlViolation::AckedCommitLoss {
                        op: (op.client_id, op.op_id),
                        commit_lsn: op
                            .commit_lsn
                            .expect("acked_durable ⇒ commit_lsn is Some")
                            .raw(),
                        watermark: watermark.raw(),
                        lost: absent,
                    });
                } else {
                    // some present AND some absent ⟹ torn straddle.
                    violations.push(ArcqlViolation::PartialMergeCommit {
                        op: (op.client_id, op.op_id),
                        present,
                        absent,
                    });
                }
            } else {
                // ── ADR-182 §2.2 bullet 2: NONE of `o`'s writes may be
                //    present (aborted rollback / past-watermark loss).
                if !present.is_empty() {
                    let reason = match op.outcome {
                        OpOutcome::Aborted => "aborted",
                        OpOutcome::Pending => "pending",
                        // Committed here ⇒ commit_lsn > watermark (the
                        // acked_durable branch above already consumed
                        // lsn ≤ watermark); it lost the ack race.
                        OpOutcome::Committed => "committed-past-watermark",
                    };
                    violations.push(ArcqlViolation::PhantomCommit {
                        op: (op.client_id, op.op_id),
                        reason,
                        present,
                    });
                }
                // none present ⟹ legal (correctly atomic-absent).
            }
        }

        if violations.is_empty() {
            ArcqlVerdict::Ok(summary)
        } else {
            ArcqlVerdict::Violations(violations)
        }
    }
}

/// The set of *counter keys* in a history: any node key with an 8-byte
/// big-endian counter payload ([`super::checker::encode_counter`]) on a
/// committed write. Read-side histories (no counter writes) return an
/// empty set, so the node-identity predicates run exactly as before.
fn counter_keys_of(ops: &[RecordedOp]) -> std::collections::HashSet<u64> {
    use crate::common::checker::decode_counter;
    let mut keys = std::collections::HashSet::new();
    for op in ops {
        if op.outcome != OpOutcome::Committed {
            continue;
        }
        for w in &op.writes {
            if w.value.as_ref().and_then(decode_counter).is_some() {
                keys.insert(w.key);
            }
        }
    }
    keys
}

/// Lost-update predicate (W27-ν-2 / ADR-163 §FD-1). A *counter key* is
/// any node key whose committed writes carry an 8-byte big-endian
/// counter payload ([`super::checker::encode_counter`]). For each such
/// key the workload models an initial CREATE that writes `seed` (the
/// minimum committed value observed) followed by `n` increment SETs;
/// under correct SI/OCC each increment that COMMITS advances the value
/// by exactly one (a concurrent loser aborts), so the **maximum
/// committed value == seed + (committed_value_writes − 1)** (the `−1`
/// drops the seeding CREATE itself from the increment count). The `−1`
/// is correct precisely because the seed CREATE is itself counter-tagged
/// (`seed_counter_node` writes `encode_counter(seed)`), so it occupies
/// the first `values` slot that the `−1` drops. A lost update leaves the
/// max value short of that expectation.
///
/// Returns `(counters_checked, violations)`. A history with no
/// counter-tagged writes returns `(0, [])` — read-side histories are
/// untouched.
fn check_lost_updates(ops: &[RecordedOp]) -> (usize, Vec<ArcqlViolation>) {
    use crate::common::checker::decode_counter;

    // Per counter key: collect every committed counter value written.
    let mut values_by_key: HashMap<u64, Vec<u64>> = HashMap::new();
    for op in ops {
        if op.outcome != OpOutcome::Committed {
            continue;
        }
        for w in &op.writes {
            if let Some(v) = w.value.as_ref().and_then(decode_counter) {
                values_by_key.entry(w.key).or_default().push(v);
            }
        }
    }

    let mut violations = Vec::new();
    let mut checked = 0usize;
    for (key, values) in &values_by_key {
        // The seed is the minimum committed value (the CREATE that
        // initializes the counter); the count of committed value-writes
        // beyond the seed is the number of applied increments.
        let Some(&seed) = values.iter().min() else {
            continue;
        };
        let Some(&final_value) = values.iter().max() else {
            continue;
        };
        let committed_increments = (values.len() as u64).saturating_sub(1);
        let expected_value = seed.saturating_add(committed_increments);
        checked += 1;
        if final_value != expected_value {
            violations.push(ArcqlViolation::LostUpdate {
                key: *key,
                committed_increments,
                final_value,
                expected_value,
            });
        }
    }
    (checked, violations)
}

/// Build the ww∪wr direct-serialization-graph among committed ops and
/// detect a cycle. Returns the op indices on a witness cycle, if any.
///
/// - **wr(A→B)**: B observed node x; A is x's committed creator. B read
///   the version A wrote.
/// - **ww(A→B)**: A and B both committed a write to node x and
///   `commit_lsn(A) < commit_lsn(B)` (version order).
///
/// Under correct SI this graph is acyclic by construction (every edge
/// advances LSN); a cycle witnesses G1c.
///
/// **Domain boundary (W27-ν-2 / ADR-163 §FD-1):** this detector models
/// the **write-once node-identity** domain — each node key is created by
/// exactly one committed writer (its single `creator_idx` entry) and
/// optionally tombstoned once. `counter_keys` are EXCLUDED: a counter is
/// a *repeatedly-SET multi-version key*, for which the single-creator
/// wr-edge approximation is unsound (a reader observes the version at
/// its own snapshot, not the final writer's), and whose serializability
/// is instead verified by the dedicated [`check_lost_updates`] predicate
/// (Predicate 5). Excluding them keeps each predicate sound on its own
/// key domain with zero overlap — the read-side CREATE/DELETE workloads
/// (which use no counter keys) are entirely unaffected.
fn detect_g1c_cycle(
    ops: &[RecordedOp],
    creator_idx: &HashMap<u64, usize>,
    counter_keys: &std::collections::HashSet<u64>,
) -> Option<Vec<usize>> {
    let n = ops.len();
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];

    // wr edges.
    for (b_idx, op) in ops.iter().enumerate() {
        if op.outcome != OpOutcome::Committed {
            continue;
        }
        for r in &op.reads {
            if r.key == SCAN_SENTINEL_KEY
                || r.key == POISONED_READ_MARKER
                || counter_keys.contains(&r.key)
            {
                continue;
            }
            if let Some(&a_idx) = creator_idx.get(&r.key) {
                if a_idx != b_idx {
                    adj[a_idx].push(b_idx);
                }
            }
        }
    }

    // ww edges: per item, chain committed writers in commit-LSN order.
    let mut writers_of: HashMap<u64, Vec<(u64, usize)>> = HashMap::new();
    for (idx, op) in ops.iter().enumerate() {
        if op.outcome != OpOutcome::Committed {
            continue;
        }
        if let Some(commit) = op.commit_lsn {
            for w in &op.writes {
                if counter_keys.contains(&w.key) {
                    continue;
                }
                writers_of
                    .entry(w.key)
                    .or_default()
                    .push((commit.raw(), idx));
            }
        }
    }
    for writers in writers_of.values_mut() {
        if writers.len() < 2 {
            continue;
        }
        writers.sort_unstable();
        for pair in writers.windows(2) {
            let (_, a) = pair[0];
            let (_, b) = pair[1];
            if a != b {
                adj[a].push(b);
            }
        }
    }

    find_cycle(&adj)
}

/// Iterative DFS cycle finder (explicit stack — never recurses, so it
/// is safe against adversarial history depth). Returns one witness
/// cycle's node indices if the graph is cyclic.
fn find_cycle(adj: &[Vec<usize>]) -> Option<Vec<usize>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Color {
        White,
        Gray,
        Black,
    }
    let n = adj.len();
    let mut color = vec![Color::White; n];
    // Parent pointers to reconstruct the cycle witness.
    let mut parent = vec![usize::MAX; n];

    for start in 0..n {
        if color[start] != Color::White {
            continue;
        }
        // Stack of (node, child-cursor) frames.
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        color[start] = Color::Gray;
        while let Some(&(node, cursor)) = stack.last() {
            if cursor < adj[node].len() {
                stack.last_mut().unwrap().1 += 1;
                let next = adj[node][cursor];
                match color[next] {
                    Color::White => {
                        color[next] = Color::Gray;
                        parent[next] = node;
                        stack.push((next, 0));
                    }
                    Color::Gray => {
                        // Back-edge node→next: reconstruct next..node.
                        let mut cycle = vec![next];
                        let mut cur = node;
                        while cur != next && cur != usize::MAX {
                            cycle.push(cur);
                            cur = parent[cur];
                        }
                        cycle.reverse();
                        return Some(cycle);
                    }
                    Color::Black => {}
                }
            } else {
                color[node] = Color::Black;
                stack.pop();
            }
        }
    }
    None
}

/// Count write-skew witness pairs: two committed read-modify-write ops
/// that are concurrent (LSN intervals overlap) with disjoint write
/// sets. This is the canonical write-skew shape (two transactions each
/// read a shared predicate, then write different items). Snapshot
/// isolation **permits** this (Adya 2000 §4.3), so it is reported as an
/// informational witness, never a violation.
///
/// `counter_keys` (W27-ν-2) are excluded: counter RMW ops read+write the
/// SAME single key (not disjoint) and belong to the lost-update domain,
/// so they are never write-skew witnesses — excluding them keeps the
/// informational count clean and the domain partition strict.
fn count_write_skew_witnesses(
    ops: &[RecordedOp],
    counter_keys: &std::collections::HashSet<u64>,
) -> usize {
    // Committed read-modify-write ops: have at least one real read AND
    // at least one write, on a NON-counter key (counter RMW is the
    // lost-update predicate's domain).
    let touches_only_counter =
        |o: &RecordedOp| -> bool { o.writes.iter().all(|w| counter_keys.contains(&w.key)) };
    let rmw: Vec<&RecordedOp> = ops
        .iter()
        .filter(|o| o.outcome == OpOutcome::Committed)
        .filter(|o| !touches_only_counter(o))
        .filter(|o| {
            let has_read = o.reads.iter().any(|r| {
                r.key != SCAN_SENTINEL_KEY
                    && r.key != POISONED_READ_MARKER
                    && !counter_keys.contains(&r.key)
            });
            let has_write = o.writes.iter().any(|w| !counter_keys.contains(&w.key));
            has_read && has_write
        })
        .collect();

    let mut witnesses = 0usize;
    for i in 0..rmw.len() {
        for j in (i + 1)..rmw.len() {
            let a = rmw[i];
            let b = rmw[j];
            let (Some(a_commit), Some(b_commit)) = (a.commit_lsn, b.commit_lsn) else {
                continue;
            };
            // Concurrent: intervals [start, commit] overlap.
            let concurrent =
                a.start_lsn.raw() < b_commit.raw() && b.start_lsn.raw() < a_commit.raw();
            if !concurrent {
                continue;
            }
            // Disjoint write sets.
            let a_writes: std::collections::HashSet<u64> = a.writes.iter().map(|w| w.key).collect();
            let disjoint = b.writes.iter().all(|w| !a_writes.contains(&w.key));
            if disjoint {
                witnesses += 1;
            }
        }
    }
    witnesses
}

#[cfg(test)]
mod adversarial_oracle_tests {
    //! Synthetic-bad-history self-tests that prove the SI checker
    //! *detects* planted anomalies — i.e. the positive engine tests in
    //! `jepsen_arcql_si_read.rs` (every g0/g1a/g1b/g1c case asserts
    //! `verdict.is_ok()` against the REAL engine) are NOT vacuous. If
    //! `find_cycle` / `detect_g1c_cycle` (or the SI predicates) were a
    //! no-op that always returned "legal", the tests below FAIL.
    //! (W27-ν R2 fix-up; closes the ADR-165 M1 clause-e oracle-strength
    //! gap flagged by Tier-B R1.)

    use std::collections::HashMap;

    use arcgraph_core::{Lsn, TenantId};
    use arcgraph_storage::test_harness::jepsen::history::OpBuilder;
    use bytes::Bytes;

    use super::{ArcqlSiChecker, ArcqlViolation, detect_g1c_cycle, find_cycle};

    fn lsn(n: u64) -> Lsn {
        Lsn::new(n)
    }

    fn node_value() -> Option<Bytes> {
        Some(Bytes::from_static(b"v"))
    }

    /// `find_cycle` MUST report a planted 2-cycle (0 → 1 → 0). A
    /// constant-`None` stub would make the G1c predicate a no-op and
    /// every g1c engine test would green-pass vacuously.
    #[test]
    fn find_cycle_detects_two_cycle() {
        let adj = vec![vec![1usize], vec![0usize]];
        let cycle = find_cycle(&adj).expect("find_cycle must report the planted 2-cycle");
        assert!(
            cycle.contains(&0) && cycle.contains(&1),
            "cycle witness must contain both nodes; got {cycle:?}"
        );
    }

    /// Control: `find_cycle` MUST return `None` on a DAG (0 → 1 → 2).
    /// Proves it is not a constant-`Some` stub either.
    #[test]
    fn find_cycle_accepts_acyclic_graph() {
        let adj = vec![vec![1usize], vec![2usize], vec![]];
        assert!(
            find_cycle(&adj).is_none(),
            "find_cycle must accept an acyclic graph"
        );
    }

    /// `detect_g1c_cycle` MUST report a cycle in a hand-built ww∪wr
    /// graph: op A creates node 1 and reads node 2; op B creates node 2
    /// and reads node 1. wr(A→B) [B read 1, A wrote 1] ∪ wr(B→A) [A read
    /// 2, B wrote 2] = a 2-cycle that SI forbids (Adya 2000 §4.2).
    #[test]
    fn detect_g1c_cycle_detects_ww_wr_cycle() {
        let mut a = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(1));
        a.intend_write(1, node_value());
        a.observe_read(2, node_value());
        let op_a = a.into_committed(lsn(10));

        let mut b = OpBuilder::new(1, 1, TenantId::DEFAULT, lsn(2));
        b.intend_write(2, node_value());
        b.observe_read(1, node_value());
        let op_b = b.into_committed(lsn(20));

        let ops = vec![op_a, op_b];
        // creator_idx exactly as `check()` builds it from these ops.
        let creator_idx: HashMap<u64, usize> = HashMap::from([(1u64, 0usize), (2u64, 1usize)]);
        // No counter keys in this node-identity history → empty exclusion
        // set (these writes carry 1-byte markers, not 8-byte counters).
        let counter_keys = std::collections::HashSet::new();

        let cycle = detect_g1c_cycle(&ops, &creator_idx, &counter_keys)
            .expect("detect_g1c_cycle must report the planted ww∪wr cycle");
        assert!(!cycle.is_empty(), "cycle witness must be non-empty");
    }

    /// End-to-end through the public `check()` API: the same ww∪wr cycle
    /// surfaces as `ArcqlViolation::G1cCycle` rather than a no-op `Ok`.
    #[test]
    fn check_reports_g1c_cycle_violation() {
        let mut a = OpBuilder::new(0, 0, TenantId::DEFAULT, lsn(1));
        a.intend_write(1, node_value());
        a.observe_read(2, node_value());
        let op_a = a.into_committed(lsn(10));

        let mut b = OpBuilder::new(1, 1, TenantId::DEFAULT, lsn(2));
        b.intend_write(2, node_value());
        b.observe_read(1, node_value());
        let op_b = b.into_committed(lsn(20));

        let verdict = ArcqlSiChecker::new().check(&[op_a, op_b]);
        assert!(!verdict.is_ok(), "checker must FAIL on a G1c-cycle history");
        let violations = verdict.violations().expect("violations on the bad path");
        assert!(
            violations
                .iter()
                .any(|v| matches!(v, ArcqlViolation::G1cCycle { .. })),
            "expected ArcqlViolation::G1cCycle; got {violations:?}"
        );
    }
}
