//! Storage engine for ArcGraph.
//!
//! Scope: user-space buffer pool (no mmap on the hot path — see
//! ADR-001 and design-v2 §3.4), WAL with group commit, LiveGraph-style
//! TEL adjacency, MVCC transactions, and the CRUD surface that higher
//! crates call into.
//!
//! All I/O goes through here. Nothing above this layer touches the disk.

#![recursion_limit = "256"]

pub mod address;
pub mod addressed_store;
pub mod blob;
pub mod buffer;
pub mod catalog;
pub mod checkpoint;
pub mod config;
pub mod crud;
pub mod data_dir_version;
pub mod encryption;
pub mod engine;
pub mod extent;
pub mod idempotency;
pub mod intern;
pub mod io;
pub mod m3_migration;
pub mod m4_migration;
pub mod manifest;
pub mod metrics;
pub mod migrate;
pub mod mutation_log;
pub mod owner_budget;
pub mod owner_index;
pub mod owner_payload;
pub mod owner_rewrite;
pub mod owner_row;
pub mod page_alloc;
pub mod page_store;
pub mod permissions;
pub mod pin;
pub mod primary_index;
pub mod prop_block;
pub mod property;
pub mod property_index_catalog;
pub mod record_store;
pub mod records;
pub mod recovery;
pub mod redo;
pub mod router;
pub mod secondary_handle;
pub mod spill;
pub mod tel;
pub mod test_harness;
pub mod transaction;
pub mod vector_store;
pub mod wal;

pub use addressed_store::{AddressedRecordStore, AddressedStoreError, address_read_disposition};
pub use blob::{
    BLOB_CHUNK_BYTES, BLOB_MAX_BYTES, BLOB_PAGE_HEADER, BlobBoundConfig, BlobError,
    BlobPageSnapshot, BlobSpill, BlobStore, BlobStoreHandle, StagedBlob, decode_put_blob_payload,
    encode_put_blob_payload,
};
pub use buffer::{
    BufferPool, DEFAULT_WRITE_FRACTION, Frame, FrameId, FrameReadGuard, FrameWriteGuard, PageTable,
};
pub use catalog::{
    CATALOG_PAGE_ID, CatalogPageAttachReport, CatalogStats, SystemCatalog, TenantRecord,
    decode_durability_tier,
};
// SVC-1 / #849 / ADR-229 — WAL checkpoint producer + checkpoint-anchored
// recovery. Bounds restart-recovery to O(WAL-since-checkpoint). See the
// `checkpoint` module docs for the v1.0 durability reality (OQ-2) that
// makes the full-state snapshot mandatory.
pub use checkpoint::{
    BlobPageFlushTarget, CHECKPOINT_INCREMENTAL_PREFIX, CHECKPOINT_SIDECAR_FILE,
    CHECKPOINT_SNAPSHOT_FILE, CheckpointSidecar, CheckpointSnapshot,
    DEFAULT_WRITE_BEHIND_BATCH_PAGES, DOUBLEWRITE_FILE, DoublewriteArea, DoublewriteKey,
    DoublewriteRestoreReport, DoublewriteRestoreTarget, ExtentDirectoryDoublewriteHome,
    INCREMENTAL_METADATA_FORMAT_VERSION, IncrementalCheckpointMetadata,
    IncrementalCheckpointReport, IncrementalCheckpointRestore, IncrementalMetadataReport,
    M3DoublewriteHome, PageFlushTarget, SnapshotOwnerCounts, WriteBehindCheckpointer,
    WriteBehindReport, incremental_checkpoint, incremental_metadata_path,
    read_incremental_metadata, read_latest_sidecar, read_snapshot,
    restore_latest_incremental_checkpoint, write_sidecar_atomic, write_snapshot_atomic,
};
pub use config::{WalCheckpointConfig, WalErrorPolicy};
pub use crud::{
    CrudAclWalSink, CrudAllocatorSeedHandle, crud_allocator_seed_handle,
    crud_allocator_seed_handle_with_owners,
};
// SVC-2 / #1302 — on-disk data-dir version stamp + boot-time compatibility
// guard (upgrade-safety). Mirrors the WAL / catalog-page version guards.
pub use data_dir_version::{
    DATA_DIR_FORMAT_VERSION, DATA_DIR_VERSION_CHAINED_V1, DATA_DIR_VERSION_DELTA_M3,
    DATA_DIR_VERSION_DIRECT_M4, DATA_DIR_VERSION_MAGIC, DATA_DIR_VERSION_SLOTTED_M1,
    DATA_DIR_VERSION_TYPED_M2, DataDirVersionError, SUPPORTED_DATA_DIR_VERSIONS, VERSION_FILE,
    check_or_stamp_data_dir, check_tel_ref_format, stamp_data_dir,
    stamp_data_dir_with_parent_sync_error_for_test, version_file_path,
};
// v2 M1 (ADR-230) — data-dir MANIFEST + the migrate-on-open sweep.
pub use manifest::{
    DataDirManifest, DataDirManifestError, TEL_REF_FORMAT_BARE_PAGE_ID,
    TEL_REF_FORMAT_PAGE_SLOT_V1, read_data_dir_manifest, write_data_dir_manifest,
};
pub use migrate::{
    M1MigrateOptions, M1MigrateReport, M2MigrateError, M2MigrateOptions, M2MigrateReport,
    M2ReencodeFn, run_m1_migrate_on_open, run_m2_migrate_on_open,
};
// W20β-3 / ADR-052: secrets-at-rest encryption primitives. v1.0-α
// ships WAL encryption + the credential seam; page-store encryption is
// deferred to v1.1 by the PR #373 R1 scope narrowing.
// The `install_random_key` helper is re-exported at the top level
// (alongside `WalEncryption`) for API consistency per PR #373 R1 N-3.
pub use encryption::{
    AEAD_KEY_LEN, AES_GCM_IV_LEN, AES_GCM_TAG_LEN, Aes256GcmCipher, CipherError,
    ENCRYPTION_KEY_NAMESPACE_WAL, KeyRing, KeyRingError, PayloadEncryption,
    SECRETS_PROVIDER_KEY_SOURCE_PREFIX, SecretsProviderKeySource, SidecarCodecError,
    SidecarIoError, WAL_DEK_SIDECAR_FILE, WAL_ENCRYPTION_MAGIC, WAL_PAYLOAD_HEADER_LEN,
    WalDekSidecar, WalEncryption, WalEncryptionBootstrap, WalEncryptionBootstrapError,
    bootstrap_wal_encryption, decrypt_wal_payload, encrypt_wal_payload, install_random_key,
    is_encrypted_wal_payload, sidecar_path,
};
pub use engine::{
    CrudStoreGraphAdapter, EngineConfig, EngineError, EngineHandles, GraphAdapterError,
    ProductionRefreshHook, ProductionRefreshHookError, bootstrap_engine,
};
pub use idempotency::{
    IdempotencyBinding, IdempotencyBoundConfig, IdempotencySpill, IdempotencyStore,
};
pub use intern::{
    InternTable, STRINGID_SENTINEL, decode_intern_payload, encode_intern_payload,
    intern_label_logged, intern_logged, intern_string_logged, intern_type_logged,
};
pub use io::{InMemoryPageIo, PageIo, PosixPageIo};
pub use metrics::{MetricsSink, QueryPlanType, StoragePageKind, WalWriteOutcome};
// `CountingMetricsSink` is a test fixture gated `#[cfg(test)]` in
// `metrics.rs`; not part of the public API surface per
// `feedback_avoid_speculative_scaffolding.md` (ship test-utility types
// where consumed, not in the production export). Downstream tests
// wanting a counting sink can either implement `MetricsSink` directly
// (3 methods) or re-introduce a `test-utils` cargo feature when the
// first consumer lands.
pub use mutation_log::{IndexHandle, PageStoreKind, TxnMutationLog};
pub use owner_budget::{
    BulkClassCensus, OWNER_BUDGET_FLOOR_BYTES, OwnerBulkBudgets, OwnerSubstrateBudget,
};
pub use owner_index::{
    OWNER_INDEX_DISK_CAP_BYTES, OwnerForwardIndex, OwnerIndexError, str_hash_56,
};
pub use owner_payload::{OWNER_PAYLOAD_DISK_CAP_BYTES, OwnerPayloadError, OwnerPayloadStore};
pub use owner_rewrite::{
    BoundedOwnerSorter, OWNER_REWRITE_MAX_RECORD_BYTES, OWNER_REWRITE_RUN_BUFFER_BYTES,
    OWNER_REWRITE_SCRATCH_CAP_BYTES, OwnerRewriteError, OwnerRewriteScratchBudget,
};
pub use owner_row::{
    OWNER_IDS_PER_CLASS, OWNER_ROW_BYTES, OWNER_ROW_MAX_PAYLOAD, OWNER_ROWS_PER_PAGE, OwnerRow,
    OwnerRowAddress, OwnerRowClass, OwnerRowError, OwnerRowRegistry, OwnerRowStore,
    is_owner_store_id, owner_direct_row_disposition,
};
pub use page_alloc::PageAllocator;
// W26-ε-2 / ADR-140: on-disk page-store wire-through. The
// `BufferedRecordPageStore` is the substrate that closes the
// W22-DB-α-1-cap RSS-linear ingest blocker per the independent
// auditor's critical-path item B (Scale 3→4 lift). Pairs with
// ADR-138 / W25-β-1 (Scale 1→3 lift) — together advancing the
// W23-COMPETE Scale dimension from 1/5 → 4/5.
pub use page_store::{
    BufferedRecordPageStore, DEFAULT_CACHE_CAP_PAGES, GenerationId, PageStoreIdentity,
    PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend, RecordPageKey,
};
pub use property::{
    BlobRef, InlineShape, OVERFLOW_BIT, OVERFLOW_PAGE_BITS, OVERFLOW_PAGE_MASK, OVERFLOW_SLOT_BITS,
    OVERFLOW_SLOT_MASK, PropertyReadout, decode_node as decode_property_node,
    decode_rel as decode_property_rel, encode_inline_node, encode_inline_rel, encode_overflow_node,
    encode_overflow_rel,
};
pub use record_store::{RecordPageStore, RecordStoreError};
pub use redo::{
    DeltaPageStore, DirtyPageKey, DirtyPageSnapshot, DirtyPageTable, RecoveryDeltaOutcome,
    RedoLsnRange, RedoOrderError, RedoOrderStats, apply_physical_delta, apply_recovery_delta,
    apply_redo_if_newer, sort_by_redo_range,
};
pub use router::{MultiTenantRouter, RoutingError, TenantHandle};
pub use secondary_handle::{SecondaryIndexHandle, SecondaryIndexHandleError, SecondaryIndexValue};
pub use spill::{
    DEFAULT_ORPHAN_SWEEP_QUERY_INTERVAL, DEFAULT_SPILL_QUOTA_MULTIPLIER,
    DEFAULT_SPILL_STAGING_MEMORY_BYTES, DEFAULT_VOLUME_HEADROOM_PERCENT, QueryEpoch, SpillBatch,
    SpillEncryptionPolicy, SpillError, SpillManager, SpillManagerConfig, SpillQuery,
    SpillQueryConfig, SpillRejectReason, SpillRun, SpillRunIdentity, SpillRunReader,
    SpillRunWriter, SpillSweepReport, VolumeHeadroom, VolumeSpace,
};
pub use vector_store::{
    VectorPageStore, VectorPageStoreArc, VectorPageStoreHandle, VectorStoreError,
};
pub use wal::{
    AclGrantEntry, AclGrantOp, AllocatorAdvance, AllocatorKind, AllocatorSeedHandle,
    BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2, BUNDLE_FORMAT_V3, BUNDLE_FORMAT_V4, BUNDLE_FORMAT_V8,
    BackgroundFsyncFailAction, BackgroundFsyncMetrics, BackgroundFsyncScheduler,
    DecodedCommitBundle, DecodedIndexPage, SideChannelWrite, StagedEmit, TornTail, WalConfig,
    WalFireMetrics, WalHandle, WalRecord, WalRecordType, WalRecoveryReader, WalWriter,
    audit_fsync_barriers, decode_commit_bundle, decode_commit_bundle_for_version,
    decode_commit_bundle_v1, decode_commit_bundle_v2, decode_commit_bundle_v3,
    decode_commit_bundle_v4, decode_commit_bundle_v8, encode_commit_bundle,
    encode_commit_bundle_v2, encode_commit_bundle_v3, encode_commit_bundle_v4,
    encode_commit_bundle_v8,
};
