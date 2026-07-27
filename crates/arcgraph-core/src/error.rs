//! ArcGraph error taxonomy.
//!
//! One error enum spans the workspace; each variant carries enough
//! context that a caller can decide whether to retry, abort, or fail
//! the transaction. See `docs/arcgraph-design-v2.md` §3.2 and §4 for
//! the durability / MVCC invariants these variants defend.

use thiserror::Error;

use crate::ids::{Lsn, PageId};

/// Canonical `Result` alias for the workspace.
pub type Result<T> = std::result::Result<T, ArcGraphError>;

/// Every error produced by the ArcGraph workspace.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ArcGraphError {
    /// Underlying I/O failure (disk, io_uring, network).
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// A page failed checksum validation or had an inconsistent header.
    #[error("page corruption at {page_id:?}: {reason}")]
    PageCorruption {
        /// Which page failed.
        page_id: PageId,
        /// Human-readable cause for operator diagnosis.
        reason: String,
    },

    /// A WAL record failed CRC32C or framing validation.
    #[error("wal corruption at lsn {lsn:?}: {reason}")]
    WalCorruption {
        /// LSN of the first byte of the corrupt record, as best we know.
        lsn: Lsn,
        /// Human-readable cause.
        reason: String,
    },

    /// A CRC-valid WAL record carries a record-type discriminant reserved by
    /// the on-disk format, but this build has no producer or typed payload
    /// representation for it. Distinct from [`Self::WalCorruption`] because
    /// the record framing and discriminant are valid; recovery callers can
    /// deliberately skip or halt without misreporting intact data as corrupt.
    #[error("wal record type byte {byte} is reserved and not produced by this build")]
    WalRecordTypeReserved {
        /// Reserved record-type discriminant observed on disk.
        byte: u8,
    },

    /// A commit lost an OCC validation race; the caller may retry.
    #[error("mvcc conflict on {target}")]
    MvccConflict {
        /// Short description of the contention point (vertex id, edge id).
        target: String,
    },

    /// Decoder received a slice that is not the expected fixed size.
    #[error("invalid record length: got {got} bytes, expected {expected}")]
    InvalidRecordLength {
        /// Bytes received.
        got: usize,
        /// Bytes expected.
        expected: usize,
    },

    /// Unknown byte in the `page_type` field.
    #[error("unknown page type byte: {0}")]
    UnknownPageType(u8),

    /// Record-format version byte does not match any known version.
    #[error("unsupported record format version: {0}")]
    UnsupportedRecordVersion(u8),

    /// Page magic does not match the expected sentinel.
    #[error("bad page magic: got 0x{got:08x}, expected 0x{expected:08x}")]
    BadPageMagic {
        /// Bytes we saw.
        got: u32,
        /// Bytes we expected.
        expected: u32,
    },

    /// Buffer pool could not allocate a frame (all frames pinned).
    #[error("buffer pool exhausted")]
    BufferPoolExhausted,

    /// Transaction was aborted explicitly.
    #[error("transaction aborted: {reason}")]
    TransactionAborted {
        /// Cause.
        reason: String,
    },

    /// The WAL writer thread is not running (crashed, shut down, or not yet spawned).
    #[error("wal writer unavailable")]
    WalUnavailable,

    /// A WAL segment header carries an on-disk format version this
    /// binary does not support. Raised by `WalRecoveryReader` and
    /// `SegmentWriter::open` before any record is decoded, so the
    /// operator sees "upgrade required" instead of "WAL corrupt".
    #[error(
        "wal format mismatch: found version {found_version}, this binary supports {supported_versions:?} — upgrade required"
    )]
    WalFormatMismatch {
        /// Version stamped into the offending segment header.
        found_version: u16,
        /// Versions this binary knows how to read.
        supported_versions: &'static [u16],
    },

    /// A WAL segment's magic bytes do not match the expected
    /// sentinel. Distinct from `WalFormatMismatch` so a caller can
    /// tell "wrong file type entirely" from "right file, unknown
    /// version".
    #[error("wal bad magic: got {got:02x?}, expected {expected:02x?}")]
    WalBadMagic {
        /// Bytes read from the segment header.
        got: [u8; 4],
        /// Bytes we expect (`b"AGWL"`).
        expected: [u8; 4],
    },

    /// The storage mount has options that compromise fsync durability.
    /// Startup refuses to proceed when this is detected (see
    /// design-v2 §3.4 and ADR-001).
    #[error("unsafe mount options at {mountpoint}: {reason}")]
    UnsafeMountOptions {
        /// Mount point that failed the audit.
        mountpoint: String,
        /// Operator-facing description of the unsafe option.
        reason: String,
    },

    /// ADR-033 Z-1 (b): the transaction's WAL fsync failed, and the
    /// in-memory rollback machinery successfully unwound MVCC
    /// versions + page-state mutations. The caller may retry with a
    /// fresh transaction; the failed commit's effects are fully
    /// reversed.
    ///
    /// `source` is the underlying WAL error (typically `Io`,
    /// `WalUnavailable`, or `WalCorruption`) — useful for operator
    /// diagnostics but NOT load-bearing for retry decisions; any
    /// `WalErrorRolledBack` is retryable by construction.
    ///
    /// Emitted by `commit_with_bundle_and_rollback` when the WAL
    /// error policy is `Rollback` (the default; see ADR-033 §8).
    /// The `abort` policy short-circuits the process before this
    /// error is ever constructed.
    ///
    /// `commit_with_bundle_and_rollback`: crate  // link placeholder;
    /// the actual method lives in arcgraph_storage::Transaction.
    #[error("wal fsync failed; transaction rolled back (retryable): {source}")]
    WalErrorRolledBack {
        /// Underlying WAL error. Boxed so the variant stays compact
        /// in the enum's layout.
        #[source]
        source: Box<ArcGraphError>,
    },

    /// ADR-032 §Slice 3c orphan escalation: a pre-ADR-031 legacy WAL
    /// was observed to contain `IndexPage = 11` records without a
    /// matching subsequent `CommitBundle = 12`, AND the post-replay
    /// `bootstrap_from_mvcc` recovery attempt failed. Operator
    /// intervention required — the replay protocol cannot salvage the
    /// index state from this WAL.
    ///
    /// Distinct from `WalCorruption` because the WAL bytes themselves
    /// are intact; only the logical invariant that every orphan page
    /// can be reindexed from MVCC has been violated. Operator message:
    /// "replay halted: orphan pages detected and bootstrap recovery
    /// failed; manual recovery required."
    #[error(
        "wal replay halted: orphan pages detected and bootstrap recovery failed; manual recovery required ({reason})"
    )]
    UnrecoverableOrphans {
        /// Number of orphan `IndexPage` records observed in the WAL.
        orphan_count: u64,
        /// Human-readable cause (typically the inner bootstrap error).
        reason: String,
    },

    /// ADR-035 §4.6 step 4 ship-blocking sanity check: the recovered
    /// vector arena's `vectors_count` does not equal the graph
    /// section's `node_count`. Replay halts; operator-triggered
    /// rebuild via `bootstrap_from_mvcc` is required (§9.1) — the
    /// inconsistency is a correctness violation that surfaces as
    /// missing or extra search results, so silent continuation is
    /// forbidden. Diagnostics are operator-actionable: the message
    /// names `(tenant, index)` and the count delta directly.
    ///
    /// Spec name in ADR-035 §4.6 is
    /// `WalReplayFailure::VectorIndexInconsistency`; the workspace
    /// uses one error enum (`ArcGraphError`) so the variant lives
    /// here rather than in a sub-enum, matching `UnrecoverableOrphans`.
    /// W20β-3 / ADR-052: a WAL record's encrypted payload failed to
    /// decrypt. Distinct from `WalCorruption` so the operator can
    /// distinguish "bit-flip on disk" (CRC catches; emit
    /// `WalCorruption`) from "wrong key / tag mismatch / IV
    /// mismatch" (GCM authentication tag catches; emit
    /// `WalDecryptionFailed`). Both halt replay; the structured
    /// error MUST propagate — silent fallback to plaintext is
    /// FORBIDDEN per `feedback_noop_trampoline_anti_pattern.md`.
    #[error(
        "wal decryption failed at lsn {lsn:?} key_version {key_version}: {reason}; \
         check that the SecretsProvider holds the historical key version"
    )]
    WalDecryptionFailed {
        /// LSN of the record whose payload failed to decrypt.
        lsn: Lsn,
        /// Key version stamped in the encrypted payload's header.
        key_version: u16,
        /// Operator-facing cause (NOT the secret bytes — values are
        /// not safe to log).
        reason: String,
    },

    /// W20β-3 / ADR-052: a page failed to decrypt. Distinct from
    /// `PageCorruption` so the operator can distinguish on-disk
    /// bit-flip (CRC catches; emit `PageCorruption`) from
    /// "wrong key / tag mismatch" (GCM authentication tag catches;
    /// emit `PageDecryptionFailed`). Silent fallback to plaintext is
    /// FORBIDDEN.
    #[error(
        "page decryption failed at {page_id:?} key_version {key_version}: {reason}; \
         check that the SecretsProvider holds the page-store key for this version"
    )]
    PageDecryptionFailed {
        /// The page whose ciphertext failed to authenticate.
        page_id: PageId,
        /// Key version stamped in the encrypted page slot's header.
        key_version: u16,
        /// Operator-facing cause.
        reason: String,
    },

    #[error(
        "wal replay halted: vector index inconsistency for tenant={tenant_id} index={index_id} \
         snapshot_lsn={snapshot_lsn}: vectors_count={observed_vectors_count} \
         graph_node_count={observed_graph_node_count} (delta={delta}, \
         wal_replay_high_lsn={wal_replay_high_lsn}); operator rebuild via bootstrap_from_mvcc required"
    )]
    VectorIndexInconsistency {
        /// Tenant that owns the inconsistent arena (raw `TenantId` u64).
        tenant_id: u64,
        /// Index that owns the inconsistent arena (per-tenant id).
        index_id: u64,
        /// Snapshot LSN of the loaded snapshot. `0` when the inconsistency
        /// surfaced after a `bootstrap_from_mvcc` reconstruction (no
        /// snapshot was the source).
        snapshot_lsn: u64,
        /// Vector count read from the arena's `vectors_count` field
        /// (snapshot header `vector_count` + post-snapshot WAL deltas).
        observed_vectors_count: u64,
        /// Node count read from the graph section header (HNSW
        /// `node_count` at offset 16 or VAMA `node_count` at offset 17).
        observed_graph_node_count: u64,
        /// Highest `commit_lsn` applied post-snapshot. Equals
        /// `snapshot_lsn` when no WAL deltas existed; equals the last
        /// post-snapshot bundle's `commit_lsn` otherwise.
        wal_replay_high_lsn: u64,
        /// Signed `observed_vectors_count - observed_graph_node_count`
        /// for at-a-glance operator diagnostics. Pre-computed at
        /// construction so the error message stays a single
        /// `format!`-able string.
        delta: i64,
    },
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;
    use crate::ids::{Lsn, PageId};

    #[test]
    fn display_io_includes_underlying() {
        let e = ArcGraphError::Io(io::Error::new(io::ErrorKind::PermissionDenied, "denied"));
        let s = format!("{e}");
        assert!(s.starts_with("i/o error: "), "got: {s}");
        assert!(s.contains("denied"), "got: {s}");
    }

    #[test]
    fn display_page_corruption_includes_fields() {
        let e = ArcGraphError::PageCorruption {
            page_id: PageId::new(42),
            reason: "bad checksum".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("page corruption"), "got: {s}");
        assert!(s.contains("42"), "got: {s}");
        assert!(s.contains("bad checksum"), "got: {s}");
    }

    #[test]
    fn display_wal_corruption_includes_fields() {
        let e = ArcGraphError::WalCorruption {
            lsn: Lsn::new(7),
            reason: "crc mismatch".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("wal corruption"), "got: {s}");
        assert!(s.contains("7"), "got: {s}");
        assert!(s.contains("crc mismatch"), "got: {s}");
    }

    #[test]
    fn display_reserved_wal_record_type_is_not_corruption() {
        let e = ArcGraphError::WalRecordTypeReserved { byte: 13 };
        let s = format!("{e}");
        assert_eq!(
            s,
            "wal record type byte 13 is reserved and not produced by this build"
        );
        assert!(!s.contains("corruption"), "got: {s}");
    }

    #[test]
    fn display_mvcc_conflict_names_target() {
        let e = ArcGraphError::MvccConflict {
            target: "vertex 13".to_owned(),
        };
        assert_eq!(format!("{e}"), "mvcc conflict on vertex 13");
    }

    #[test]
    fn display_invalid_record_length_names_sizes() {
        let e = ArcGraphError::InvalidRecordLength {
            got: 17,
            expected: 36,
        };
        let s = format!("{e}");
        assert!(s.contains("17"), "got: {s}");
        assert!(s.contains("36"), "got: {s}");
    }

    #[test]
    fn display_unknown_page_type_shows_byte() {
        let e = ArcGraphError::UnknownPageType(99);
        assert_eq!(format!("{e}"), "unknown page type byte: 99");
    }

    #[test]
    fn display_unsupported_record_version_shows_byte() {
        let e = ArcGraphError::UnsupportedRecordVersion(7);
        assert_eq!(format!("{e}"), "unsupported record format version: 7");
    }

    #[test]
    fn display_bad_page_magic_shows_both_hex() {
        let e = ArcGraphError::BadPageMagic {
            got: 0xDEAD_BEEF,
            expected: 0x4743_5241,
        };
        let s = format!("{e}");
        assert!(s.contains("0xdeadbeef"), "got: {s}");
        assert!(s.contains("0x47435241"), "got: {s}");
    }

    #[test]
    fn display_buffer_pool_exhausted_is_stable() {
        let e = ArcGraphError::BufferPoolExhausted;
        assert_eq!(format!("{e}"), "buffer pool exhausted");
    }

    #[test]
    fn display_transaction_aborted_includes_reason() {
        let e = ArcGraphError::TransactionAborted {
            reason: "deadlock".to_owned(),
        };
        assert_eq!(format!("{e}"), "transaction aborted: deadlock");
    }

    #[test]
    fn display_wal_unavailable_is_stable() {
        let e = ArcGraphError::WalUnavailable;
        assert_eq!(format!("{e}"), "wal writer unavailable");
    }

    #[test]
    fn display_wal_format_mismatch_names_versions() {
        let e = ArcGraphError::WalFormatMismatch {
            found_version: 999,
            supported_versions: &[1],
        };
        let s = format!("{e}");
        assert!(s.contains("999"), "got: {s}");
        assert!(s.contains("upgrade required"), "got: {s}");
    }

    #[test]
    fn display_wal_bad_magic_shows_both_hex() {
        let e = ArcGraphError::WalBadMagic {
            got: *b"XXXX",
            expected: *b"AGWL",
        };
        let s = format!("{e}");
        assert!(s.contains("58"), "expected X=0x58 in display, got: {s}");
        assert!(s.contains("41"), "expected A=0x41 in display, got: {s}");
    }

    #[test]
    fn display_unsafe_mount_options_includes_mountpoint() {
        let e = ArcGraphError::UnsafeMountOptions {
            mountpoint: "/data".to_owned(),
            reason: "ext4 with nobarrier".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("/data"), "got: {s}");
        assert!(s.contains("ext4"), "got: {s}");
    }

    #[test]
    fn from_std_io_error_via_question_mark() {
        fn touch() -> Result<()> {
            let _ = std::fs::metadata("/definitely/does/not/exist/arcgraph-test")?;
            Ok(())
        }
        let err = touch().unwrap_err();
        assert!(matches!(err, ArcGraphError::Io(_)));
    }

    #[test]
    fn source_of_io_variant_is_the_underlying() {
        use std::error::Error as StdError;
        let io_err = io::Error::other("underlying");
        let e = ArcGraphError::Io(io_err);
        let src = e.source().expect("Io variant must expose its source");
        assert!(src.to_string().contains("underlying"));
    }

    // ─── ADR-033 Z-1 (b) WalErrorRolledBack variant ───

    #[test]
    fn display_wal_error_rolled_back_carries_source() {
        let inner = ArcGraphError::Io(io::Error::other("fsync eio"));
        let e = ArcGraphError::WalErrorRolledBack {
            source: Box::new(inner),
        };
        let s = format!("{e}");
        assert!(s.contains("rolled back"), "got: {s}");
        assert!(s.contains("fsync eio"), "got: {s}");
    }

    #[test]
    fn source_of_wal_error_rolled_back_exposes_underlying() {
        use std::error::Error as StdError;
        let inner = ArcGraphError::WalUnavailable;
        let e = ArcGraphError::WalErrorRolledBack {
            source: Box::new(inner),
        };
        let src = e
            .source()
            .expect("WalErrorRolledBack must expose its source");
        assert_eq!(src.to_string(), "wal writer unavailable");
    }

    #[test]
    fn wal_error_rolled_back_is_pattern_matchable() {
        // Retry logic matches on this variant; ensure the pattern
        // compiles against the public enum surface.
        let e = ArcGraphError::WalErrorRolledBack {
            source: Box::new(ArcGraphError::WalUnavailable),
        };
        match e {
            ArcGraphError::WalErrorRolledBack { .. } => {}
            other => panic!("expected WalErrorRolledBack, got {other:?}"),
        }
    }

    // ─── ADR-032 §Slice 3c UnrecoverableOrphans variant ───

    #[test]
    fn display_unrecoverable_orphans_carries_count_and_reason() {
        let e = ArcGraphError::UnrecoverableOrphans {
            orphan_count: 7,
            reason: "bootstrap_from_mvcc: missing MVCC state".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("orphan"), "got: {s}");
        assert!(s.contains("manual recovery required"), "got: {s}");
        assert!(s.contains("bootstrap_from_mvcc"), "got: {s}");
    }

    #[test]
    fn unrecoverable_orphans_is_pattern_matchable() {
        let e = ArcGraphError::UnrecoverableOrphans {
            orphan_count: 1,
            reason: "test".to_owned(),
        };
        match e {
            ArcGraphError::UnrecoverableOrphans { orphan_count, .. } => {
                assert_eq!(orphan_count, 1);
            }
            other => panic!("expected UnrecoverableOrphans, got {other:?}"),
        }
    }

    // ─── ADR-035 §4.6 step 4 VectorIndexInconsistency variant ───

    #[test]
    fn display_vector_inconsistency_names_tenant_index_and_delta() {
        let e = ArcGraphError::VectorIndexInconsistency {
            tenant_id: 7,
            index_id: 42,
            snapshot_lsn: 1000,
            observed_vectors_count: 1024,
            observed_graph_node_count: 1023,
            wal_replay_high_lsn: 1100,
            delta: 1,
        };
        let s = format!("{e}");
        assert!(s.contains("vector index inconsistency"), "got: {s}");
        assert!(s.contains("tenant=7"), "got: {s}");
        assert!(s.contains("index=42"), "got: {s}");
        assert!(s.contains("snapshot_lsn=1000"), "got: {s}");
        assert!(s.contains("vectors_count=1024"), "got: {s}");
        assert!(s.contains("graph_node_count=1023"), "got: {s}");
        assert!(s.contains("delta=1"), "got: {s}");
        assert!(s.contains("wal_replay_high_lsn=1100"), "got: {s}");
        assert!(s.contains("bootstrap_from_mvcc"), "got: {s}");
    }

    // ─── W20β-3 ADR-052 WalDecryptionFailed variant ───────────────

    #[test]
    fn display_wal_decryption_failed_carries_lsn_and_version() {
        let e = ArcGraphError::WalDecryptionFailed {
            lsn: Lsn::new(42),
            key_version: 3,
            reason: "AES-GCM tag mismatch".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("wal decryption failed"), "got: {s}");
        assert!(s.contains("42"), "got: {s}");
        assert!(s.contains("key_version 3"), "got: {s}");
        assert!(s.contains("AES-GCM tag mismatch"), "got: {s}");
        assert!(
            s.contains("historical key version"),
            "operator hint must point at the key-version mismatch path"
        );
    }

    #[test]
    fn wal_decryption_failed_is_pattern_matchable() {
        let e = ArcGraphError::WalDecryptionFailed {
            lsn: Lsn::new(1),
            key_version: 1,
            reason: "x".to_owned(),
        };
        match e {
            ArcGraphError::WalDecryptionFailed {
                lsn, key_version, ..
            } => {
                assert_eq!(lsn.raw(), 1);
                assert_eq!(key_version, 1);
            }
            other => panic!("expected WalDecryptionFailed, got {other:?}"),
        }
    }

    // ─── W20β-3 ADR-052 PageDecryptionFailed variant ──────────────

    #[test]
    fn display_page_decryption_failed_carries_page_and_version() {
        let e = ArcGraphError::PageDecryptionFailed {
            page_id: PageId::new(7),
            key_version: 2,
            reason: "wrong key".to_owned(),
        };
        let s = format!("{e}");
        assert!(s.contains("page decryption failed"), "got: {s}");
        assert!(s.contains("7"), "got: {s}");
        assert!(s.contains("key_version 2"), "got: {s}");
        assert!(s.contains("wrong key"), "got: {s}");
    }

    #[test]
    fn page_decryption_failed_is_pattern_matchable() {
        let e = ArcGraphError::PageDecryptionFailed {
            page_id: PageId::new(99),
            key_version: 5,
            reason: "x".to_owned(),
        };
        match e {
            ArcGraphError::PageDecryptionFailed {
                page_id,
                key_version,
                ..
            } => {
                assert_eq!(page_id.raw(), 99);
                assert_eq!(key_version, 5);
            }
            other => panic!("expected PageDecryptionFailed, got {other:?}"),
        }
    }

    #[test]
    fn vector_inconsistency_is_pattern_matchable() {
        let e = ArcGraphError::VectorIndexInconsistency {
            tenant_id: 1,
            index_id: 2,
            snapshot_lsn: 3,
            observed_vectors_count: 4,
            observed_graph_node_count: 5,
            wal_replay_high_lsn: 6,
            delta: -1,
        };
        match e {
            ArcGraphError::VectorIndexInconsistency {
                tenant_id,
                index_id,
                delta,
                ..
            } => {
                assert_eq!(tenant_id, 1);
                assert_eq!(index_id, 2);
                assert_eq!(delta, -1);
            }
            other => panic!("expected VectorIndexInconsistency, got {other:?}"),
        }
    }
}
