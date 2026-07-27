//! WAL recovery reader (roadmap M1-34).
//!
//! Iterates every record in the WAL directory in segment order, from
//! the lowest segment number to the highest. The caller applies each
//! record to the in-memory state; after exhausting the iterator, the
//! reader exposes:
//!
//! - [`WalRecoveryReader::last_lsn`] — the maximum LSN seen. Feed this
//!   into [`crate::wal::WalWriter::spawn_from`] so new appends continue
//!   the monotonic sequence.
//! - [`WalRecoveryReader::torn_tail`] — `Some(TornTail)` when the last
//!   segment ended with a truncated record (typical after a crash
//!   between `write_all_at` and `fdatasync`). This is expected; the
//!   pre-tear prefix was still replayed successfully. A **zero-filled**
//!   record boundary (the `length` header field reads `0`) in the
//!   terminal segment is treated identically — #1521/#1457 M6.1 MF6:
//!   a SIGKILL can leave a fully zero-filled sector/page past the last
//!   valid record rather than a short/partial one, and that shape is
//!   just as legitimate a torn tail as a short read.
//!
//! Hard-corruption behaviour: if decoding fails mid-stream (CRC
//! mismatch or bad framing somewhere that is **not** the last segment's
//! tail), the iterator returns `Some(Err(...))` with the underlying
//! [`ArcGraphError`]. Applying recovered state past this point is on
//! the caller. Typical policy for M1.e exit: stop replay and surface
//! the error to the operator. A zero-filled record boundary in a
//! **non-terminal** segment still hard-errors this way — only the
//! terminal segment's tail gets the clean-truncation treatment.
//!
//! Reserved record-type behaviour: bytes 13–17 are valid format
//! reservations, not corruption. The bare engine has no payload type or
//! producer for those prediction-event records, so the reader advances past
//! each CRC-valid reserved record and emits a structured `warn!` containing
//! its byte, LSN, segment, and offset. The skipped record's LSN still advances
//! [`WalRecoveryReader::last_lsn`] so a subsequent writer cannot reuse it.
//!
//! The reader never reads a whole WAL into memory at once — it holds
//! one segment's bytes at a time. Segments are 64 MiB by default.

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arcgraph_core::{ArcGraphError, Lsn, Result};
use tracing::warn;

use crate::encryption::{PayloadEncryption, WalEncryption};
use crate::transaction::TxnManager;
use crate::vector_store::recovery::{ArenaRecoveryJob, RecoveredArena, recover_all_arenas};
use crate::vector_store::{VectorArenaPageStore, VectorPageStoreHandle};
use crate::wal::record::WalRecord;
use crate::wal::replay::{PageStoreTarget, ReplayConfig, ReplayExecutor, ReplayMetricsSnapshot};
use crate::wal::segment::{SegmentHeader, fsync_dir, list_segments, segment_filename};

/// Position where a partially-written record was detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TornTail {
    /// Segment in which the torn record was observed.
    pub segment: u64,
    /// Byte offset within the segment at which the torn record starts.
    pub offset: usize,
}

/// Truncate a terminal torn WAL segment to the last valid recovery boundary.
///
/// Recovery computes `tail.offset` while decoding the WAL prefix. Before a
/// writer reuses the directory, callers must remove the torn suffix; otherwise
/// fresh records can be appended after garbage and the next recovery may either
/// hard-fail or silently skip the fresh suffix. This function makes the
/// truncation durable by syncing both the segment file and its parent directory.
pub fn truncate_torn_tail(dir: &Path, tail: TornTail) -> Result<()> {
    let path = dir.join(segment_filename(tail.segment));
    let file = OpenOptions::new().read(true).write(true).open(&path)?;
    file.set_len(tail.offset as u64)?;
    file.sync_data()?;
    fsync_dir(dir)?;
    Ok(())
}

#[derive(Debug)]
struct SegmentBuffer {
    seg_no: u64,
    bytes: Vec<u8>,
    cursor: usize,
}

impl SegmentBuffer {
    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }
}

/// Iterator over records in a WAL directory.
///
/// Created via [`WalRecoveryReader::open`]. Implements `Iterator<
/// Item = Result<WalRecord>>`. After iteration, call [`Self::last_lsn`]
/// and [`Self::torn_tail`].
///
/// **W20β-3 / ADR-052**: when an encrypted WAL is being recovered,
/// pass a [`WalEncryption`] via [`Self::with_encryption`]. The reader
/// transparently decrypts encrypted payloads on yield; plaintext
/// records pass through unchanged (peek by AEAD magic). A reader
/// without `with_encryption` set yields encrypted payloads as-is —
/// downstream consumers (e.g., the bundle decoder) will fail with
/// `WalCorruption` because they cannot interpret the AEAD wrapper.
#[derive(Debug)]
pub struct WalRecoveryReader {
    dir: PathBuf,
    remaining_segments: Vec<u64>, // popped from the back
    current: Option<SegmentBuffer>,
    last_lsn: Lsn,
    torn_tail: Option<TornTail>,
    done: bool,
    encryption: Option<WalEncryption>,
}

impl WalRecoveryReader {
    /// Open the WAL directory for replay. Lists segments upfront; the
    /// iterator reads segment bytes lazily, one segment at a time.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let mut segments = list_segments(&dir)?;
        // We pop from the back but want lowest first → reverse.
        segments.reverse();
        let mut me = Self {
            dir,
            remaining_segments: segments,
            current: None,
            last_lsn: Lsn::ZERO,
            torn_tail: None,
            done: false,
            encryption: None,
        };
        me.advance_segment()?;
        Ok(me)
    }

    /// W20β-3 / ADR-052: attach a [`WalEncryption`] config so the
    /// reader transparently decrypts encrypted record payloads. A
    /// reader opened without this method continues to operate
    /// pre-encryption (and will fail downstream if it encounters an
    /// encrypted payload).
    #[must_use]
    pub fn with_encryption(mut self, encryption: WalEncryption) -> Self {
        self.encryption = Some(encryption);
        self
    }

    /// Highest LSN decoded so far.
    #[must_use]
    pub fn last_lsn(&self) -> Lsn {
        self.last_lsn
    }

    /// WAL directory this reader was opened against. Exposed for the
    /// ADR-032 replay executor (Slice 3) which needs to read each
    /// segment's header for format_version dispatch.
    #[inline]
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Segment number currently being read, or `None` if the reader
    /// is positioned past the last segment (post-EOF). The ADR-032
    /// replay executor queries this after each `next()` to route
    /// per-segment `format_version` into the bundle codec dispatcher.
    #[inline]
    #[must_use]
    pub fn current_seg_no(&self) -> Option<u64> {
        self.current.as_ref().map(|c| c.seg_no)
    }

    /// Position of the torn tail, if recovery reached one.
    #[must_use]
    pub fn torn_tail(&self) -> Option<TornTail> {
        self.torn_tail
    }

    /// Convenience: collect all records into a Vec, short-circuiting
    /// on error. Use the iterator form directly if streaming matters.
    pub fn collect_all(self) -> Result<Vec<WalRecord>> {
        let mut out = Vec::new();
        for item in self {
            out.push(item?);
        }
        Ok(out)
    }

    fn advance_segment(&mut self) -> Result<()> {
        if let Some(seg_no) = self.remaining_segments.pop() {
            let path = self.dir.join(segment_filename(seg_no));
            let bytes = std::fs::read(&path)?;
            // Issue #39: validate the segment header BEFORE any record
            // decoding. Unknown version / wrong magic is propagated as
            // a structured error so operators see "upgrade required"
            // instead of downstream "WAL corrupt".
            //
            // Exception: a terminal segment whose file is shorter than
            // the 8-byte header is a creation-crash torn tail (the
            // writer create()'d the file but crashed before the header
            // reached disk). No records can have been durable there,
            // so we fold it into the existing torn-tail semantics.
            // Non-terminal segments with a short or unrecognized header
            // always hard-fail.
            let terminal = self.remaining_segments.is_empty();
            if bytes.len() < SegmentHeader::SIZE && terminal {
                self.torn_tail = Some(TornTail {
                    segment: seg_no,
                    offset: 0,
                });
                self.done = true;
                self.current = None;
                return Ok(());
            }
            let _header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE.min(bytes.len())])?;
            self.current = Some(SegmentBuffer {
                seg_no,
                bytes,
                cursor: SegmentHeader::SIZE,
            });
        } else {
            self.current = None;
        }
        Ok(())
    }

    fn is_terminal_segment(&self) -> bool {
        self.remaining_segments.is_empty()
    }
}

enum DecodeOutcome {
    Record(WalRecord, usize),
    Reserved { byte: u8, consumed: usize, lsn: Lsn },
    Truncated,
    Failure(ArcGraphError),
}

impl Iterator for WalRecoveryReader {
    type Item = Result<WalRecord>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        loop {
            // Make sure we have a segment with bytes to decode.
            let needs_advance = match &self.current {
                Some(b) => b.remaining() == 0,
                None => true,
            };
            if needs_advance {
                match self.advance_segment() {
                    Ok(()) => {}
                    Err(e) => {
                        self.done = true;
                        return Some(Err(e));
                    }
                }
                if self.current.is_none() {
                    self.done = true;
                    return None;
                }
                continue;
            }

            // Decode inside a tight scope so the `&mut buf` borrow is
            // released before we mutate `self.last_lsn` / `self.done`.
            let (outcome, seg_no, cursor) = {
                let buf = self
                    .current
                    .as_ref()
                    .expect("current non-None after advance");
                let seg_no = buf.seg_no;
                let cursor = buf.cursor;
                let remaining = &buf.bytes[cursor..];
                let outcome = match WalRecord::decode(remaining) {
                    Ok((record, consumed)) => DecodeOutcome::Record(record, consumed),
                    Err(ArcGraphError::WalRecordTypeReserved { byte }) => {
                        // `WalRecord::decode` reaches the record-type parser
                        // only after length, CRC, and reserved-header-byte
                        // validation. It is therefore safe to use the
                        // validated framing length to advance over this
                        // unsupported-but-intact record.
                        let consumed =
                            u32::from_le_bytes(remaining[4..8].try_into().expect("4-byte length"))
                                as usize;
                        let lsn = Lsn::new(u64::from_le_bytes(
                            remaining[20..28].try_into().expect("8-byte LSN"),
                        ));
                        DecodeOutcome::Reserved {
                            byte,
                            consumed,
                            lsn,
                        }
                    }
                    Err(ArcGraphError::InvalidRecordLength { .. }) => DecodeOutcome::Truncated,
                    // #1521/#1457 M6.1 MF6 — a zero-filled record boundary
                    // (`length` field reads 0) is `WalCorruption` out of
                    // `WalRecord::decode` (the `length < HEADER_SIZE` guard
                    // fires before the CRC/type checks even run), NOT
                    // `InvalidRecordLength` — so it never took the
                    // short-tail `Truncated` branch above even though a
                    // zero-fill immediately after the last valid record is
                    // exactly the same "torn tail" shape a SIGKILL landing
                    // on a record boundary produces (the OS/filesystem can
                    // leave a fully-zeroed sector/page rather than a
                    // short/partial one). Recognize this EXACT shape —
                    // `length == 0` at the current cursor, with at least
                    // `HEADER_SIZE` bytes available to inspect it — and
                    // route it through the same `Truncated` classification
                    // as a short tail; a genuinely corrupt record (nonzero
                    // garbage length, bad CRC, bad reserved bytes, unknown
                    // type) is untouched and still hard-errors via
                    // `DecodeOutcome::Failure` below.
                    Err(ArcGraphError::WalCorruption { .. })
                        if remaining.len() >= WalRecord::HEADER_SIZE
                            && remaining[4..8] == [0u8, 0, 0, 0] =>
                    {
                        DecodeOutcome::Truncated
                    }
                    Err(e) => DecodeOutcome::Failure(e),
                };
                (outcome, seg_no, cursor)
            };

            match outcome {
                DecodeOutcome::Record(mut record, consumed) => {
                    let buf = self.current.as_mut().expect("checked above");
                    buf.cursor += consumed;
                    if record.lsn > self.last_lsn {
                        self.last_lsn = record.lsn;
                    }
                    // W20β-3 / ADR-052: transparent decryption on
                    // yield. Plaintext records pass through unchanged.
                    if matches!(
                        PayloadEncryption::peek(&record.payload),
                        PayloadEncryption::Encrypted { .. }
                    ) {
                        match self.encryption.as_ref() {
                            Some(enc) => match enc.decrypt(seg_no, record.lsn, &record.payload) {
                                Ok(pt) => record.payload = pt,
                                Err(e) => {
                                    self.done = true;
                                    return Some(Err(e));
                                }
                            },
                            None => {
                                self.done = true;
                                return Some(Err(ArcGraphError::WalDecryptionFailed {
                                    lsn: record.lsn,
                                    key_version: PayloadEncryption::peek(&record.payload)
                                        .as_key_version()
                                        .unwrap_or(0),
                                    reason: "WalRecoveryReader has no encryption config; \
                                             encountered encrypted payload during recovery — \
                                             use WalRecoveryReader::with_encryption(...) at \
                                             open time"
                                        .to_owned(),
                                }));
                            }
                        }
                    }
                    return Some(Ok(record));
                }
                DecodeOutcome::Reserved {
                    byte,
                    consumed,
                    lsn,
                } => {
                    let buf = self.current.as_mut().expect("checked above");
                    buf.cursor += consumed;
                    if lsn > self.last_lsn {
                        self.last_lsn = lsn;
                    }
                    warn!(
                        record_type_byte = byte,
                        lsn = lsn.raw(),
                        segment = seg_no,
                        offset = cursor,
                        "reserved WAL record type is not produced or applied by this build; \
                         skipping record"
                    );
                    continue;
                }
                DecodeOutcome::Truncated if self.is_terminal_segment() => {
                    // Torn tail in the final segment — the expected
                    // shape of a crash between write_all_at and fsync.
                    self.torn_tail = Some(TornTail {
                        segment: seg_no,
                        offset: cursor,
                    });
                    self.done = true;
                    return None;
                }
                DecodeOutcome::Truncated => {
                    self.done = true;
                    return Some(Err(ArcGraphError::WalCorruption {
                        lsn: Lsn::ZERO,
                        reason: format!(
                            "truncated record in non-terminal segment {seg_no} at offset {cursor}"
                        ),
                    }));
                }
                DecodeOutcome::Failure(e) => {
                    self.done = true;
                    return Some(Err(e));
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// ADR-032 Slice 3d: public recovery entry point
// ─────────────────────────────────────────────────────────────────

/// Stateful summary returned by [`recover_from_wal`].
///
/// Exposes post-replay state (max applied commit_lsn, reader's
/// last WAL LSN, metrics snapshot) so the caller can feed the
/// value into the WAL writer's `spawn_from(last_lsn + 1)`
/// entry point and surface operator-facing numbers.
///
/// **M3.a Slice G.3 fold-in.** `vector_arenas` carries the per-
/// `(tenant, index)` arenas reloaded by
/// [`crate::vector_store::recover_arena`] when a non-empty
/// `vector_recovery_jobs` was passed to [`recover_from_wal`].
/// The list is empty for deployments without vector indexes (the
/// pre-G.3 default) so existing M2.e callers see no behavioural
/// change.
#[derive(Debug)]
pub struct RecoveryReport {
    /// Max `commit_lsn` applied. Equals the post-replay
    /// `TxnManager::current_lsn()`.
    pub applied_commit_lsn: Lsn,
    /// Last WAL record LSN observed. Feed this into
    /// `WalWriter::spawn_from(last_wal_lsn + 1)`.
    pub last_wal_lsn: Lsn,
    /// Whether the WAL ended with a torn tail.
    pub torn_tail: Option<TornTail>,
    /// Observability snapshot (13 counters + 4 gauges).
    pub metrics: ReplayMetricsSnapshot,
    /// M3.a Slice G.3: vector arenas reloaded after the WAL replay
    /// completed. One entry per `(tenant, index)` arena passed in
    /// `vector_recovery_jobs`. Empty when the caller did not opt
    /// into vector recovery.
    pub vector_arenas: Vec<RecoveredArena>,
}

/// Run a full ADR-032 §3 replay pass against the WAL in `dir`.
///
/// This is the caller-facing entry point invoked at process start
/// (or by another offline recovery caller). It:
///
/// 1. Opens a [`WalRecoveryReader`] on `dir`.
/// 2. Constructs a [`ReplayExecutor`] against `txn_mgr` + `target`.
/// 3. Executes the replay (§R1 → §R7).
/// 4. Returns a [`RecoveryReport`] with the post-replay high
///    water + metrics.
///
/// The `target` routes `IndexPage` entries to the owning page
/// stores (primary + optional secondary). See
/// [`PageStoreTarget`] for
/// construction; typical deployments use
/// `PageStoreTarget::primary_only(primary)` or
/// `PageStoreTarget::new(primary, secondary)`.
///
/// The `config` argument tunes the buffer bounds and spill dir.
/// `None` ⇒ `ReplayConfig::with_wal_dir(dir)` (env-var overrides
/// applied + spill under `{dir}/replay-spill`).
///
/// Errors:
///
/// - [`ArcGraphError::WalCorruption`] — halt (§R5 / §6).
/// - [`ArcGraphError::WalFormatMismatch`] — segment header
///   version unsupported.
/// - [`ArcGraphError::UnrecoverableOrphans`] — orphan pages
///   detected and `bootstrap_from_mvcc` failed (§Slice 3c).
pub fn recover_from_wal(
    dir: &Path,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    config: Option<ReplayConfig>,
) -> Result<RecoveryReport> {
    recover_from_wal_with_vector_arenas(dir, txn_mgr, target, config, None, &[])
}

/// W20β-3 / ADR-052: encryption-aware recovery entry point. When the
/// WAL was written with encryption, callers MUST pass the same
/// [`WalEncryption`] config here so the reader can decrypt records on
/// yield. Without this, the recovery path will fail with
/// [`ArcGraphError::WalDecryptionFailed`] on the first encrypted
/// record.
///
/// This is a thin wrapper that opens the [`WalRecoveryReader`] with
/// the encryption knob, then runs the full ADR-032 replay against
/// `txn_mgr` + `target`. Equivalent to [`recover_from_wal`] when
/// `encryption` is `None`.
pub fn recover_from_wal_encrypted(
    dir: &Path,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    config: Option<ReplayConfig>,
    encryption: Option<WalEncryption>,
) -> Result<RecoveryReport> {
    recover_from_wal_encrypted_anchored(dir, txn_mgr, target, config, encryption, Lsn::ZERO)
}

/// SVC-1 / #849 / ADR-229 — checkpoint-anchored encryption-aware
/// recovery. Identical to [`recover_from_wal_encrypted`] except the
/// replay is anchored at `checkpoint_floor`: WAL records with
/// `commit_lsn <= checkpoint_floor` are SKIPPED (their effects are
/// already durable in the restored checkpoint snapshot — the caller must
/// have restored it BEFORE calling this) and only records with
/// `commit_lsn > checkpoint_floor` are replayed. This bounds
/// restart-recovery to `O(WAL-since-checkpoint)` (the #849 rc-blocker).
///
/// `checkpoint_floor == Lsn::ZERO` = no checkpoint / replay from the
/// beginning (back-compat, exactly the pre-ADR-229 behaviour).
///
/// The `applied_commit_lsn` in the returned report is
/// `max(checkpoint_floor, highest-applied-post-frontier-commit)`, so the
/// downstream `seed_after_replay` / CDC-watermark seed observe the true
/// post-restart high-water even when the WAL-since-checkpoint is empty.
pub fn recover_from_wal_encrypted_anchored(
    dir: &Path,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    config: Option<ReplayConfig>,
    encryption: Option<WalEncryption>,
    checkpoint_floor: Lsn,
) -> Result<RecoveryReport> {
    recover_from_wal_encrypted_frontiers(
        dir,
        txn_mgr,
        target,
        config,
        encryption,
        checkpoint_floor,
        None,
    )
}

/// M3 incremental-checkpoint recovery with distinct logical and physical
/// frontiers. Owners captured in metadata skip through `checkpoint_floor`;
/// store-0/1 page redo starts at `redo_floor`.
#[allow(clippy::too_many_arguments)]
pub fn recover_from_wal_encrypted_incremental(
    dir: &Path,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    config: Option<ReplayConfig>,
    encryption: Option<WalEncryption>,
    checkpoint_floor: Lsn,
    redo_floor: Lsn,
) -> Result<RecoveryReport> {
    recover_from_wal_encrypted_frontiers(
        dir,
        txn_mgr,
        target,
        config,
        encryption,
        checkpoint_floor,
        Some(redo_floor),
    )
}

#[allow(clippy::too_many_arguments)]
fn recover_from_wal_encrypted_frontiers(
    dir: &Path,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    config: Option<ReplayConfig>,
    encryption: Option<WalEncryption>,
    checkpoint_floor: Lsn,
    redo_floor: Option<Lsn>,
) -> Result<RecoveryReport> {
    let cfg = config.unwrap_or_else(|| ReplayConfig::with_wal_dir(dir));
    let mut reader = WalRecoveryReader::open(dir)?;
    if let Some(enc) = encryption.clone() {
        reader = reader.with_encryption(enc);
    }
    let exec = ReplayExecutor::new(cfg, Arc::clone(&txn_mgr), target);
    let mut exec = match redo_floor {
        Some(redo_floor) => exec.with_incremental_checkpoint(checkpoint_floor, redo_floor),
        None => exec.with_checkpoint_floor(checkpoint_floor),
    };
    let applied = exec.run(reader)?;
    // #1115-M3 double-pass NOTE (deferred, ADR-229 §OQ/P1 scope note):
    // recovery still opens a SECOND reader to harvest last_lsn / torn_tail.
    // A checkpoint-anchored single pass could harvest both, but `run`
    // consumes the reader by value and is called from ~15 test sites, so
    // folding the two passes is a wider refactor deferred to a follow-on
    // (it does NOT block the bound — the second pass is a cheap linear
    // decode with no apply, and P2 segment-reclamation shrinks it to the
    // post-frontier tail anyway).
    let mut final_reader = WalRecoveryReader::open(dir)?;
    if let Some(enc) = encryption {
        final_reader = final_reader.with_encryption(enc);
    }
    #[allow(clippy::while_let_on_iterator)]
    while let Some(_) = final_reader.next() {
        // drain
    }
    Ok(RecoveryReport {
        applied_commit_lsn: applied,
        last_wal_lsn: final_reader.last_lsn(),
        torn_tail: final_reader.torn_tail(),
        metrics: exec.metrics().snapshot(),
        vector_arenas: Vec::new(),
    })
}

/// M3.a Slice G.3 fold-in: extended recovery entry point that ALSO
/// reloads per-`(tenant, index)` vector arenas after the WAL replay
/// completes.
///
/// The caller supplies:
///
/// - `snapshot_dir`: path to the directory holding
///   `arena-{tenant}-{index}-{lsn}.snap` files. Typically a
///   sibling of the WAL directory; production wires this from
///   `Config::vector_snapshot_dir`. `None` skips vector recovery.
/// - `vector_recovery_jobs`: one entry per
///   `(tenant, index)` arena in the index catalog. The driver calls
///   [`crate::vector_store::recover_arena`] for each in sequence;
///   per ADR-035 §4.6 the operations are independent, so a per-
///   arena failure surfaces immediately.
///
/// Each vector arena recovery is **independent** of the global WAL
/// replay (the WAL replay applies any
/// [`crate::wal::bundle::BundlePageKind::Vector`] entries to the
/// `target.vector_store` if registered; the per-arena recovery
/// then decodes the snapshot, applies its own filtered post-snapshot
/// WAL deltas, and runs the §4.6 step 4 sanity check). Sharing
/// `target.vector_store` across both phases is intentional —
/// idempotence (Lemma I2) makes it safe.
///
/// # When to use which entry point
///
/// - [`recover_from_wal`] — pre-G.3 deployments without vector
///   indexes; equivalent to passing `None` for `snapshot_dir` and
///   `&[]` for jobs. The compatibility shim.
/// - This function — M3.a+ deployments with at least one vector
///   index in the catalog.
///
/// # Errors
///
/// Inherits every error from [`recover_from_wal`] plus:
///
/// - [`ArcGraphError::VectorIndexInconsistency`] — a per-arena
///   sanity check failed (ADR-035 §4.6 step 4).
/// - [`ArcGraphError::Io`] — `snapshot_dir` is unreadable.
pub fn recover_from_wal_with_vector_arenas(
    dir: &Path,
    txn_mgr: Arc<TxnManager>,
    target: PageStoreTarget,
    config: Option<ReplayConfig>,
    snapshot_dir: Option<&Path>,
    vector_recovery_jobs: &[ArenaRecoveryJob<'_>],
) -> Result<RecoveryReport> {
    let cfg = config.unwrap_or_else(|| ReplayConfig::with_wal_dir(dir));
    let reader = WalRecoveryReader::open(dir)?;
    let mut exec = ReplayExecutor::new(cfg, Arc::clone(&txn_mgr), target);
    // Borrow `reader` via temporary to capture torn_tail + last_lsn
    // after `run` consumes it.
    let last_lsn_before = reader.last_lsn();
    let torn_tail = reader.torn_tail();
    let applied = exec.run(reader)?;
    // `run` has consumed the reader; re-open briefly to pick up
    // the final last_lsn and torn_tail after iteration completed.
    // (The run path updates the reader's state; we already
    // captured `last_lsn_before` which is ZERO pre-iteration, so
    // we need the final value from a fresh borrow.)
    let final_reader = WalRecoveryReader::open(dir)?;
    let mut r = final_reader;
    #[allow(clippy::while_let_on_iterator)]
    while let Some(_) = r.next() {
        // drain
    }
    let final_last_lsn = r.last_lsn();
    let final_torn = r.torn_tail();
    // Prefer the second-pass numbers; fall back to pre-run if
    // re-open failed in pathological cases.
    let _ = last_lsn_before;
    let _ = torn_tail;

    // M3.a Slice G.3: per-arena vector recovery. Driven by the
    // catalog-supplied job list; independent of the global WAL
    // replay above. Idempotence (Lemma I2) makes the shared
    // `vector_store` handle safe across both phases.
    let vector_arenas = match (snapshot_dir, vector_recovery_jobs.is_empty()) {
        (Some(snap_dir), false) => {
            // The caller did not register a `VectorPageStoreHandle`
            // on the `PageStoreTarget` (vector_store: None) when no
            // implementor was wired. For the per-arena recovery side
            // we instantiate a default in-memory
            // `VectorArenaPageStore` so the recovery hook can run
            // end-to-end without the upstream implementor; production
            // callers that want to share the runtime arena's state
            // call `recover_all_arenas` directly with their own handle.
            //
            // Pre-existing PageStoreTarget vector_store wiring path
            // is preserved; the in-process default here is purely a
            // fallback for the test surface.
            let handle: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
            recover_all_arenas(handle, snap_dir, vector_recovery_jobs)?
        }
        _ => Vec::new(),
    };

    Ok(RecoveryReport {
        applied_commit_lsn: applied,
        last_wal_lsn: final_last_lsn,
        torn_tail: final_torn,
        metrics: exec.metrics().snapshot(),
        vector_arenas,
    })
}

#[cfg(test)]
mod tests {
    use std::fs::OpenOptions;
    use std::os::unix::fs::FileExt;

    use arcgraph_core::{Lsn, TenantId};
    use proptest::prelude::*;
    use tempfile::tempdir;

    use super::*;
    use crate::wal::record::{WalRecord, WalRecordType};
    use crate::wal::segment::{list_segments, segment_filename};
    use crate::wal::writer::{WalConfig, WalWriter};

    fn test_config(dir: impl Into<PathBuf>) -> WalConfig {
        WalConfig {
            dir: dir.into(),
            segment_size_bytes: 64 * 1024 * 1024,
            group_commit_window: std::time::Duration::from_millis(2),
            group_commit_max_batch: 4,
            metrics_sink: None,
            encryption: None,
            inflight_budget_bytes: None,
        }
    }

    fn write_records(dir: &Path, ops: &[(WalRecordType, u64, Vec<u8>)]) -> Vec<Lsn> {
        let writer = WalWriter::spawn(test_config(dir.to_path_buf())).unwrap();
        let handle = writer.handle();
        let mut lsns = Vec::new();
        // Parallel enqueues so appends don't starve on their own ack.
        let mut producers = Vec::new();
        for (ty, txn, payload) in ops {
            let h = handle.clone();
            let ty = *ty;
            let txn = *txn;
            let payload = payload.clone();
            producers.push(std::thread::spawn(move || {
                h.append(ty, txn, 0, TenantId::DEFAULT, payload)
            }));
        }
        for p in producers {
            lsns.push(p.join().unwrap().unwrap());
        }
        writer.shutdown().unwrap();
        lsns.sort();
        lsns
    }

    #[test]
    fn open_empty_directory_returns_empty_reader() {
        let dir = tempdir().unwrap();
        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let records = reader.collect_all().unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn clean_recovery_preserves_all_records() {
        let dir = tempdir().unwrap();
        let ops: Vec<_> = (1u64..=10)
            .map(|i| (WalRecordType::PutNode, i, vec![i as u8]))
            .collect();
        let written = write_records(dir.path(), &ops);

        let mut reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut recovered = Vec::new();
        for r in reader.by_ref() {
            recovered.push(r.unwrap());
        }
        assert_eq!(recovered.len(), 10);
        for (w, r) in written.iter().zip(&recovered) {
            assert_eq!(*w, r.lsn);
        }
        assert_eq!(reader.last_lsn(), *written.last().unwrap());
        assert!(reader.torn_tail().is_none());
    }

    #[tracing_test::traced_test]
    #[test]
    fn recovery_skips_reserved_record_with_warning_without_corruption() {
        let dir = tempdir().unwrap();
        let written = write_records(dir.path(), &[(WalRecordType::PutNode, 1, Vec::new())]);

        // Build a byte-valid WAL record carrying reserved type 13 without
        // making that type constructible through the bare engine's encoder.
        let segment = list_segments(dir.path()).unwrap()[0];
        let path = dir.path().join(segment_filename(segment));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .unwrap();
        let record_offset = crate::wal::segment::SegmentHeader::SIZE as u64;
        let mut bytes = [0u8; WalRecord::HEADER_SIZE];
        file.read_exact_at(&mut bytes, record_offset).unwrap();
        bytes[8] = 13;
        let crc = crc32c::crc32c(&bytes[4..]);
        bytes[0..4].copy_from_slice(&crc.to_le_bytes());
        file.write_all_at(&bytes, record_offset).unwrap();
        file.sync_data().unwrap();

        let mut reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut recovered = Vec::new();
        for item in reader.by_ref() {
            recovered.push(item.expect("reserved record must not surface as corruption"));
        }

        assert!(
            recovered.is_empty(),
            "the bare engine must not materialize a removed prediction-event record"
        );
        assert_eq!(
            reader.last_lsn(),
            written[0],
            "skipping a reserved record must still advance the recovery LSN"
        );
        assert!(reader.torn_tail().is_none());
        logs_assert(|lines: &[&str]| {
            if lines.iter().any(|line| {
                line.contains("reserved WAL record type")
                    && line.contains("skipping record")
                    && line.contains("record_type_byte=13")
            }) {
                Ok(())
            } else {
                Err(format!(
                    "reserved-record recovery warning was not emitted; logs: {lines:?}"
                ))
            }
        });
    }

    #[test]
    fn recovery_spans_multiple_segments() {
        let dir = tempdir().unwrap();
        let config = WalConfig {
            segment_size_bytes: 128, // forces rotation
            group_commit_max_batch: 1,
            metrics_sink: None,
            encryption: None,
            group_commit_window: std::time::Duration::from_millis(1),
            ..test_config(dir.path().to_path_buf())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();
        for i in 1u64..=12 {
            handle
                .append(
                    WalRecordType::PutRel,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 32],
                )
                .unwrap();
        }
        writer.shutdown().unwrap();
        assert!(list_segments(dir.path()).unwrap().len() >= 2);

        let recovered = WalRecoveryReader::open(dir.path())
            .unwrap()
            .collect_all()
            .unwrap();
        assert_eq!(recovered.len(), 12);
        for w in recovered.windows(2) {
            assert!(w[0].lsn < w[1].lsn);
        }
    }

    #[test]
    fn torn_tail_is_reported_not_errored() {
        let dir = tempdir().unwrap();
        let ops: Vec<_> = (1u64..=5)
            .map(|i| (WalRecordType::PutNode, i, vec![i as u8]))
            .collect();
        let written = write_records(dir.path(), &ops);

        // Truncate the last segment by 5 bytes to mimic a crash mid-write.
        let segs = list_segments(dir.path()).unwrap();
        let last_seg = *segs.last().unwrap();
        let path = dir.path().join(segment_filename(last_seg));
        let len = std::fs::metadata(&path).unwrap().len();
        let new_len = len.saturating_sub(5);
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(new_len)
            .unwrap();

        let mut reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut recovered = Vec::new();
        for r in reader.by_ref() {
            recovered.push(r.unwrap());
        }
        // All but the last record survive (5 bytes knocked out only the
        // trailing record's CRC or payload tail → decode as truncated).
        assert_eq!(recovered.len(), written.len() - 1);
        assert!(reader.torn_tail().is_some());
    }

    /// #1521/#1457 M6.1 MF6 — a zero-filled tail (every byte 0, so the
    /// `length` header field reads exactly 0) in the TERMINAL segment
    /// must recover cleanly as a torn tail, mirroring
    /// `torn_tail_is_reported_not_errored`'s short-tail case rather than
    /// hard-erroring. Before the fix, `WalRecord::decode`'s
    /// `length < HEADER_SIZE` guard returned `WalCorruption` (not
    /// `InvalidRecordLength`) for this exact shape, so
    /// `recover_from_wal` refused to boot on a perfectly legitimate
    /// post-SIGKILL zero-filled tail.
    #[test]
    fn terminal_segment_zero_fill_tail_recovers_cleanly() {
        let dir = tempdir().unwrap();
        let ops: Vec<_> = (1u64..=5)
            .map(|i| (WalRecordType::PutNode, i, vec![i as u8]))
            .collect();
        let written = write_records(dir.path(), &ops);

        // Append a zero-filled "record" after the last valid one — a
        // legitimate shape for a SIGKILL landing exactly on a record
        // boundary (the OS/filesystem can leave a fully-zeroed
        // sector/page rather than a short/partial write).
        let segs = list_segments(dir.path()).unwrap();
        let last_seg = *segs.last().unwrap();
        let path = dir.path().join(segment_filename(last_seg));
        let mut file = OpenOptions::new().append(true).open(&path).unwrap();
        use std::io::Write;
        file.write_all(&[0u8; WalRecord::HEADER_SIZE]).unwrap();

        let mut reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut recovered = Vec::new();
        for r in reader.by_ref() {
            recovered.push(r.unwrap());
        }
        assert_eq!(
            recovered.len(),
            written.len(),
            "every genuinely-written record must survive; the zero-fill \
             tail must not be mistaken for corruption of the real records"
        );
        assert!(
            reader.torn_tail().is_some(),
            "a terminal zero-filled record boundary must be reported as \
             a clean torn tail, not silently ignored nor hard-errored"
        );
    }

    /// Sibling negative control: the SAME zero-filled shape in a
    /// NON-terminal segment must still hard-error — the terminal-only
    /// carve-out in `terminal_segment_zero_fill_tail_recovers_cleanly`
    /// must not accidentally swallow real corruption earlier in the log
    /// (e.g. a genuinely zeroed-out record from disk/bitrot damage that
    /// happens to sit before the true end of the WAL).
    #[test]
    fn non_terminal_segment_zero_fill_still_errors() {
        let dir = tempdir().unwrap();
        // Force rotation: tiny segments, many records, so there is a
        // real non-terminal segment to corrupt.
        let config = WalConfig {
            segment_size_bytes: 64,
            group_commit_max_batch: 1,
            metrics_sink: None,
            encryption: None,
            group_commit_window: std::time::Duration::from_millis(1),
            ..test_config(dir.path().to_path_buf())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();
        for i in 1u64..=6 {
            handle
                .append(
                    WalRecordType::PutNode,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 32],
                )
                .unwrap();
        }
        writer.shutdown().unwrap();

        let segs = list_segments(dir.path()).unwrap();
        assert!(segs.len() >= 2, "expected rotation: {segs:?}");
        // Zero out an entire record's header in the FIRST (non-terminal)
        // segment — the exact same "length reads 0" shape the terminal
        // case above recovers cleanly from, but here it must still
        // hard-error since it is not at the true end of the log.
        let first_seg = segs[0];
        let path = dir.path().join(segment_filename(first_seg));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let record_offset = crate::wal::segment::SegmentHeader::SIZE as u64;
        file.write_all_at(&[0u8; WalRecord::HEADER_SIZE], record_offset)
            .unwrap();

        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut saw_err = false;
        for r in reader {
            if r.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(
            saw_err,
            "a zero-filled record boundary in a NON-terminal segment must \
             still hard-error — only the terminal segment's tail gets the \
             clean-truncation treatment"
        );
    }

    #[test]
    fn structural_corruption_in_non_terminal_segment_errors() {
        let dir = tempdir().unwrap();
        // Force rotation: tiny segments, many records.
        let config = WalConfig {
            segment_size_bytes: 64,
            group_commit_max_batch: 1,
            metrics_sink: None,
            encryption: None,
            group_commit_window: std::time::Duration::from_millis(1),
            ..test_config(dir.path().to_path_buf())
        };
        let writer = WalWriter::spawn(config).unwrap();
        let handle = writer.handle();
        for i in 1u64..=6 {
            handle
                .append(
                    WalRecordType::PutNode,
                    i,
                    0,
                    TenantId::DEFAULT,
                    vec![0u8; 32],
                )
                .unwrap();
        }
        writer.shutdown().unwrap();

        let segs = list_segments(dir.path()).unwrap();
        assert!(segs.len() >= 2, "expected rotation: {segs:?}");
        // Flip a byte inside the first (non-terminal) segment's payload
        // area to force a CRC mismatch there.
        let first_seg = segs[0];
        let path = dir.path().join(segment_filename(first_seg));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut b = [0u8; 1];
        // Offset past the segment header + record header so we hit
        // the first record's payload.
        let payload_offset =
            crate::wal::segment::SegmentHeader::SIZE as u64 + WalRecord::HEADER_SIZE as u64 + 2;
        file.read_exact_at(&mut b, payload_offset).unwrap();
        b[0] ^= 0xFF;
        file.write_all_at(&b, payload_offset).unwrap();

        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let mut saw_err = false;
        for r in reader {
            if r.is_err() {
                saw_err = true;
                break;
            }
        }
        assert!(saw_err, "expected corruption error on non-terminal segment");
    }

    #[test]
    fn last_lsn_tracks_highest() {
        let dir = tempdir().unwrap();
        let ops: Vec<_> = (1u64..=7)
            .map(|i| (WalRecordType::Commit, i, vec![]))
            .collect();
        let written = write_records(dir.path(), &ops);
        let expected_last = *written.iter().max().unwrap();

        let reader = WalRecoveryReader::open(dir.path()).unwrap();
        let records = reader.collect_all().unwrap();
        assert_eq!(records.len(), 7);

        // last_lsn exposed via a fresh iteration (collect_all consumed the reader).
        let mut r2 = WalRecoveryReader::open(dir.path()).unwrap();
        for _ in r2.by_ref() {}
        assert_eq!(r2.last_lsn(), expected_last);
    }

    // ---- M1.5-06: tenant_id demultiplexing ---------------------------------
    //
    // Each replayed WalRecord carries its tenant_id.  The caller routes it
    // to the right MVCC tenant via `TxnManager::begin(record.tenant_id)`.
    // With one global TxnManager in v1.0 this is trivial, but the field must
    // survive the WAL round-trip end-to-end.

    #[test]
    fn tenant_id_round_trips_through_recovery() {
        let dir = tempdir().unwrap();
        let writer = WalWriter::spawn(test_config(dir.path().to_path_buf())).unwrap();
        let handle = writer.handle();

        // Three distinct tenants: SYSTEM (0), DEFAULT (1), and a custom one.
        let custom = TenantId::new(999);
        let tenants = [TenantId::SYSTEM, TenantId::DEFAULT, custom];
        for (i, &tid) in tenants.iter().enumerate() {
            handle
                .append(WalRecordType::PutNode, i as u64 + 1, 0, tid, vec![i as u8])
                .unwrap();
        }
        writer.shutdown().unwrap();

        let recovered = WalRecoveryReader::open(dir.path())
            .unwrap()
            .collect_all()
            .unwrap();
        assert_eq!(recovered.len(), tenants.len());

        // Sort by txn_id so order matches our insertion order.
        let mut recovered = recovered;
        recovered.sort_by_key(|r| r.txn_id);

        for (r, &expected_tid) in recovered.iter().zip(tenants.iter()) {
            assert_eq!(
                r.tenant_id, expected_tid,
                "tenant_id not preserved through WAL recovery: got {:?}, want {:?}",
                r.tenant_id, expected_tid,
            );
        }
    }

    // ---- M1.e torture proptest ------------------------------------------
    //
    // The roadmap's exit gate is "crash recovery proptest passes 10K
    // iterations". 10K cases at ~100 ms/case is ~17 minutes — too slow
    // for every PR. We ship 256 cases by default (a few seconds) and
    // let the M1.e gate run 10K via ARCGRAPH_TORTURE_CASES=10000.

    fn torture_cases() -> u32 {
        std::env::var("ARCGRAPH_TORTURE_CASES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256)
    }

    fn arb_op() -> impl Strategy<Value = (WalRecordType, u64, Vec<u8>)> {
        (
            1u8..=8,
            any::<u64>(),
            prop::collection::vec(any::<u8>(), 0..=64),
        )
            .prop_map(|(t, txn, payload)| (WalRecordType::from_byte(t).unwrap(), txn, payload))
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: torture_cases(),
            .. ProptestConfig::default()
        })]

        #[test]
        fn torture_recovered_is_prefix_of_written(
            ops in prop::collection::vec(arb_op(), 1..=40),
            truncate_bytes in 0usize..512,
        ) {
            let dir = tempdir().unwrap();
            let written = write_records(dir.path(), &ops);

            // Optionally truncate the tail to simulate a crash.
            if truncate_bytes > 0 {
                let segs = list_segments(dir.path()).unwrap();
                if let Some(&last) = segs.last() {
                    let path = dir.path().join(segment_filename(last));
                    let len = std::fs::metadata(&path).unwrap().len();
                    let new_len = len.saturating_sub(truncate_bytes as u64);
                    OpenOptions::new().write(true).open(&path).unwrap()
                        .set_len(new_len).unwrap();
                }
            }

            let reader = WalRecoveryReader::open(dir.path()).unwrap();
            let mut recovered: Vec<Lsn> = Vec::new();
            for item in reader {
                match item {
                    Ok(r) => recovered.push(r.lsn),
                    Err(_) => break,
                }
            }

            prop_assert!(recovered.len() <= written.len(),
                "recovered={}, written={}", recovered.len(), written.len());
            for (w, r) in written.iter().zip(&recovered) {
                prop_assert_eq!(*w, *r);
            }
        }
    }
}
