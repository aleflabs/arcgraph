//! Write-ahead log (WAL) — on-disk format, writer, segments, recovery, audit.
//!
//! v1.0 shape (design-v2 §4.2):
//!
//! - [`WalRecord`] — one log entry on the wire (M1-30).
//! - [`WalWriter`] — dedicated OS thread with group commit (M1-31, M1-32).
//! - [`SegmentWriter`] — 64 MiB segment rotation (M1-33).
//! - [`WalRecoveryReader`] — replay on startup (M1-34).
//! - [`audit_fsync_barriers`] — reject nobarrier mounts (M1-35).

pub mod audit;
pub mod background_fsync;
pub mod bundle;
pub mod delta;
pub mod reclaim;
pub mod record;
pub mod recovery;
pub mod replay;
pub mod segment;
pub mod spill;
pub mod writer;

pub use audit::{MountInfo, audit_fsync_barriers, audit_mount, find_mount_for_path, parse_mounts};
pub use background_fsync::{
    BackgroundFsyncFailAction, BackgroundFsyncMetrics, BackgroundFsyncScheduler,
};
pub use bundle::{
    AclGrantEntry, AclGrantOp, AllocatorAdvance, AllocatorKind, BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2,
    BUNDLE_FORMAT_V3, BUNDLE_FORMAT_V4, BUNDLE_FORMAT_V5, BUNDLE_FORMAT_V6, BUNDLE_FORMAT_V7,
    BUNDLE_FORMAT_V8, BUNDLE_FORMAT_V9, BUNDLE_FORMAT_V10, BundlePageKind, DecodedCommitBundle,
    DecodedIndexPage, DecodedStagedPage, IdempotencyBindingEntry, IdempotencyBindingOp,
    SideChannelWrite, StagedEmit, VectorPageEntry, decode_commit_bundle,
    decode_commit_bundle_for_version, decode_commit_bundle_v1, decode_commit_bundle_v2,
    decode_commit_bundle_v3, decode_commit_bundle_v4, decode_commit_bundle_v5,
    decode_commit_bundle_v6, decode_commit_bundle_v7, decode_commit_bundle_v8,
    decode_commit_bundle_v9, decode_commit_bundle_v10, encode_commit_bundle,
    encode_commit_bundle_current, encode_commit_bundle_v2, encode_commit_bundle_v3,
    encode_commit_bundle_v4, encode_commit_bundle_v5, encode_commit_bundle_v6,
    encode_commit_bundle_v7, encode_commit_bundle_v8, encode_commit_bundle_v9,
    encode_commit_bundle_v10, is_delta_bundle_format,
};
pub use delta::{
    DeltaIntent, DeltaOp, DeltaOpKind, MAX_PROP_BLOCK_PAYLOAD, STORE_BLOB_OVERFLOW, STORE_GRANTS,
    STORE_INTERN, STORE_NODE_BINDINGS, STORE_PRIMARY_INDEX, STORE_PROPS, STORE_RECORD,
    STORE_REL_BINDINGS, STORE_RELS, STORE_SECONDARY_INDEX, STORE_TEL,
};
pub use reclaim::{
    ReclaimReport, StopReason, active_segment_number, reclaim_segments_below, segment_count,
};
pub use record::{WalRecord, WalRecordType};
pub use recovery::{
    RecoveryReport, TornTail, WalRecoveryReader, recover_from_wal, recover_from_wal_encrypted,
    recover_from_wal_encrypted_anchored, recover_from_wal_encrypted_incremental,
    truncate_torn_tail,
};
pub use replay::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, ReplayConfig, ReplayExecutor, ReplayMetrics, ReplayMetricsSnapshot,
    ReplayPhase, SecondaryPageStoreHandle,
};
pub use segment::{
    CURRENT_WAL_FORMAT_VERSION, SEGMENT_FILENAME_PREFIX, SEGMENT_FILENAME_SUFFIX,
    SUPPORTED_WAL_FORMAT_VERSIONS, SegmentHeader, SegmentWriter, WAL_SEGMENT_MAGIC, fsync_dir,
    list_segments, parse_segment_filename, segment_filename,
};
pub use writer::{WalConfig, WalFireMetrics, WalHandle, WalWriter};
