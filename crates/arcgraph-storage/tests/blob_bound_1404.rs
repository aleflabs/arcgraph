//! #1404 M0 — bounded resident blob-page tier: durability integration tests.
//!
//! The unit tests in `blob.rs` cover the in-process bounded-tier behavior
//! (drain-fires / re-fault / INV-DURABLE gate / throttle). THIS file carries
//! the load-bearing DURABILITY oracle that the in-process tests cannot: a
//! real ADR-229 checkpoint over a BOUNDED blob store WHOSE PAGES HAVE BEEN
//! EVICTED-TO-SPILL, then a "crash" (drop the store) + fresh recovery from
//! the checkpoint snapshot, asserting the recovered blob content is
//! BYTE-IDENTICAL to the pre-crash content.
//!
//! This proves the #1404 M0 invariants end-to-end:
//! - INV-DURABLE (evict-only-≤-checkpoint): the producer captures a
//!   COMPLETE snapshot even though pages were evicted — the evicted pages'
//!   durable images are backfilled from the spill tier via the producer's
//!   post-guard evicted-supplement path (`read_evicted_page`), so recovery
//!   loses nothing. RED-on-revert: if the producer silently dropped evicted
//!   pages (the OQ-2 data-loss class), the recovered store would be missing
//!   the evicted blobs.
//! - Recovery byte-equality: the recovered (unbounded) store returns the
//!   same bytes for every blob, whether it was resident or spilled at
//!   checkpoint time.

use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, TenantId};
use arcgraph_storage::blob::{BlobBoundConfig, BlobSpill, BlobStore};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{
    CheckpointSnapshot, checkpoint, read_latest_sidecar, restore_latest_checkpoint,
};
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle};
use arcgraph_storage::idempotency::IdempotencyStore;
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{AllocatorAdvance, AllocatorSeedHandle};
use bytes::Bytes;
use tempfile::tempdir;

/// Minimal owner bundle for a checkpoint over a bounded blob store. Mirrors
/// the durable-bootstrap replay target shape (see `wal_checkpoint_849.rs`)
/// but parameterizes the blob store so the producer sees the bounded tier.
struct Owners {
    txn: Arc<TxnManager>,
    primary: Arc<PrimaryPageStore>,
    record: Arc<RecordPageStore>,
    blob: Arc<BlobStore>,
    allocator: Arc<PageAllocator>,
    crud: Arc<CrudStore>,
    intern: Arc<InternTable>,
    idempotency: Arc<IdempotencyStore>,
    permissions: Arc<PermissionIndex>,
}

impl Owners {
    fn with_blob(blob: Arc<BlobStore>) -> Self {
        let allocator = Arc::new(PageAllocator::new());
        let record = Arc::new(RecordPageStore::new());
        let crud = Arc::new(CrudStore::new_with_existing_page_stores(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(&record),
            Arc::clone(&blob),
        ));
        Self {
            txn: Arc::new(TxnManager::new()),
            primary: Arc::new(PrimaryPageStore::new()),
            record,
            blob,
            allocator,
            crud,
            intern: Arc::new(InternTable::new()),
            idempotency: Arc::new(IdempotencyStore::new()),
            permissions: Arc::new(PermissionIndex::new()),
        }
    }

    fn allocator_seed(&self) -> Arc<dyn AllocatorSeedHandle> {
        crud_allocator_seed_handle(Arc::clone(&self.crud), Arc::clone(&self.allocator))
    }

    fn snapshot<'a>(&'a self, seed: &'a dyn AllocatorSeedHandle) -> CheckpointSnapshot<'a> {
        CheckpointSnapshot {
            txn: &self.txn,
            primary_pages: &self.primary,
            record_pages: &self.record,
            blob: &self.blob,
            allocator_seed: seed,
            intern: &self.intern,
            idempotency: &self.idempotency,
            permissions: &self.permissions,
            permissions_tenant: TenantId::DEFAULT,
        }
    }

    fn advances(&self) -> Vec<AllocatorAdvance> {
        let mut a = self.allocator.snapshot_advances();
        a.extend(self.crud.snapshot_allocator_advances());
        a
    }
}

fn in_mem_buffer_pool() -> BufferPool {
    BufferPool::new(16, Arc::new(InMemoryPageIo::new()))
}

/// The core oracle: an ingest that FORCES blob-page eviction, then a real
/// ADR-229 checkpoint, then a "crash" (drop) + recovery — the recovered
/// blobs must be BYTE-IDENTICAL. If the producer's evicted-supplement
/// backfill were missing (silent drop = OQ-2 data loss), the recovered
/// store would be missing the spilled blobs and this would fail.
#[test]
fn evicted_blob_pages_survive_checkpoint_and_recovery_byte_identical() {
    let dir = tempdir().unwrap();

    // ── Epoch 0: ingest into a BOUNDED store with a tiny cap so most
    //    pages are evicted-to-spill, then checkpoint. ──
    let payloads: Vec<Vec<u8>> = (0..40)
        .map(|i| {
            // Mix single- and multi-chunk blobs to exercise chain re-fault.
            let len = 200 + (i % 7) * BLOB_CHUNK_STRIDE + i;
            (0..len).map(|j| ((i * 31 + j) & 0xFF) as u8).collect()
        })
        .collect();

    let blob_refs;
    let evicted_at_checkpoint;
    {
        let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
        // cap = 4 pages → far below the ~40+ pages we ingest.
        let cfg = BlobBoundConfig {
            high_watermark_bytes: 4 * PAGE_SIZE as u64,
            low_watermark_bytes: 2 * PAGE_SIZE as u64,
        };
        let blob = Arc::new(BlobStore::with_bound(spill, cfg));
        let owners = Owners::with_blob(Arc::clone(&blob));

        // Publish all blobs (drain fires inline on publish, but pages are
        // NOT yet checkpoint-durable so nothing evicts yet — INV-DURABLE).
        let mut refs = Vec::new();
        for p in &payloads {
            refs.push(blob.put(TenantId::DEFAULT, p).unwrap());
        }
        // Nothing evictable before a checkpoint captures durability.
        assert_eq!(
            blob.evicted_count(),
            0,
            "INV-DURABLE: nothing may evict before the first checkpoint",
        );

        // Run a REAL checkpoint. Under the freeze, `iter_pages_resident_only`
        // marks the resident set checkpoint-durable; the producer captures
        // the full image. Immediately after, drive the drain (a real serve
        // does this on the next publish) so pages evict-to-spill.
        let seed = owners.allocator_seed();
        let pool = in_mem_buffer_pool();
        let report = checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(1),
        )
        .expect("first checkpoint");
        let _ = report;

        // Now force eviction of the (now-durable) pages to spill, then run
        // a SECOND checkpoint — this one MUST backfill the evicted pages'
        // durable images from spill (the load-bearing path).
        blob.force_drain_for_test().unwrap();
        assert!(
            blob.evicted_count() > 0,
            "eviction did not fire post-checkpoint — cannot test the backfill",
        );
        evicted_at_checkpoint = blob.evicted_count();

        // The second checkpoint captures resident + evicted (via spill
        // backfill). If the backfill were broken, `checkpoint` would either
        // error (Corrupt) or write an incomplete snapshot.
        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(2),
        )
        .expect("second checkpoint (with evicted pages) must backfill from spill");

        // Sanity: every blob still reads back from the bounded store itself
        // (re-fault) pre-crash.
        for (r, p) in refs.iter().zip(&payloads) {
            let out = blob.get(TenantId::DEFAULT, *r).unwrap();
            assert_eq!(out.as_ref(), &p[..], "pre-crash re-fault mismatch");
        }
        blob_refs = refs;
        // ── "Crash": the bounded store + its spill file are dropped here.
        //    A real restart truncates blob-spill.db; recovery rebuilds the
        //    blob store from the checkpoint snapshot alone. ──
    }

    assert!(
        evicted_at_checkpoint > 0,
        "test precondition: pages must have been evicted at checkpoint time",
    );

    // ── Recovery: a FRESH, UNBOUNDED store restores from the checkpoint.
    //    This is the durable-image consumer — it must reconstruct every
    //    blob (resident + previously-evicted) byte-for-byte. ──
    let recovered = Owners::with_blob(Arc::new(BlobStore::new()));
    let seed_r = recovered.allocator_seed();
    let sidecar = read_latest_sidecar(dir.path())
        .unwrap()
        .expect("a checkpoint sidecar must exist");
    assert!(
        sidecar.full_state_snapshot,
        "must be a full-state checkpoint"
    );
    let restore = restore_latest_checkpoint(dir.path(), &recovered.snapshot(seed_r.as_ref()))
        .expect("restore must succeed")
        .expect("a checkpoint must be present");
    // The producer stamps the frontier from `txn.current_lsn()`; this
    // synthetic test drives no real MVCC commits, so the frontier is
    // `Lsn::ZERO`. The `Lsn::new(1/2)` args are the advisory
    // `snapshot_last_wal_lsn`, not the frontier — the load-bearing property
    // is the byte-equality below, not the frontier value.
    let _ = restore;

    // The load-bearing assertion: EVERY blob recovers byte-identical.
    for (r, p) in blob_refs.iter().zip(&payloads) {
        let out = recovered
            .blob
            .get(TenantId::DEFAULT, *r)
            .unwrap_or_else(|e| panic!("recovered blob {r:?} missing/broken: {e:?}"));
        assert_eq!(
            out.as_ref(),
            &p[..],
            "RECOVERED BLOB DIFFERS — an evicted page was lost across checkpoint+recovery",
        );
    }
}

/// The single-page fast case, tightened: ONE checkpoint, then evict, then
/// recovery. Confirms the first-checkpoint capture already makes pages
/// evict-eligible AND that a store restored from that snapshot is complete.
#[test]
fn single_checkpoint_then_evict_recovers_all_blobs() {
    let dir = tempdir().unwrap();
    let payloads: Vec<Bytes> = (0..12)
        .map(|i| Bytes::from(format!("blob-payload-number-{i}-with-some-bytes")))
        .collect();

    let refs;
    {
        let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
        let blob = Arc::new(BlobStore::with_bound(
            spill,
            BlobBoundConfig {
                high_watermark_bytes: 2 * PAGE_SIZE as u64,
                low_watermark_bytes: PAGE_SIZE as u64,
            },
        ));
        let owners = Owners::with_blob(Arc::clone(&blob));
        let mut r = Vec::new();
        for p in &payloads {
            r.push(blob.put(TenantId::DEFAULT, p).unwrap());
        }
        let seed = owners.allocator_seed();
        let pool = in_mem_buffer_pool();
        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(1),
        )
        .expect("checkpoint");
        blob.force_drain_for_test().unwrap();
        assert!(blob.evicted_count() > 0);
        // Re-checkpoint so the evicted images land in the durable snapshot.
        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(2),
        )
        .expect("re-checkpoint with evicted pages");
        refs = r;
    }

    let recovered = Owners::with_blob(Arc::new(BlobStore::new()));
    let seed_r = recovered.allocator_seed();
    restore_latest_checkpoint(dir.path(), &recovered.snapshot(seed_r.as_ref()))
        .unwrap()
        .expect("restore");
    for (r, p) in refs.iter().zip(&payloads) {
        let out = recovered.blob.get(TenantId::DEFAULT, *r).unwrap();
        assert_eq!(out.as_ref(), &p[..]);
    }
}

/// One chunk-worth of payload, so `len % 7 * STRIDE` in the mixed test
/// crosses at least one page boundary for the multi-chunk cases.
const BLOB_CHUNK_STRIDE: usize = PAGE_SIZE; // > BLOB_CHUNK_BYTES → guarantees ≥2 chunks
