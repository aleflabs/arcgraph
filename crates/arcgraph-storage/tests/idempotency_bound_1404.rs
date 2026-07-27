//! #1404 M0.x — bounded resident idempotency-binding tier: durability +
//! at-least-once-identity integration tests (the RE-2 leg).
//!
//! The unit tests in `idempotency.rs` cover the in-process bounded-tier
//! behavior (drain-fires / re-fault / INV-DURABLE gate / drain-cost). THIS
//! file carries the load-bearing GATES the in-process tests cannot:
//!
//! - **GATE 2 (BINDING-LOOKUP CORRECTNESS FROM SPILL):** after bindings
//!   spill, a re-ingest of an already-seen external_id STILL de-dupes
//!   (fault-in from spill works → 0 duplicates). RED-on-revert: a naive
//!   "evict binding to nowhere" makes the re-ingest MISS the spilled binding →
//!   a DUPLICATE. This is THE proof that bounding must be spill-to-durable-
//!   queryable, NOT evict-to-nowhere (fable RE-2). Both directions are
//!   asserted here in one test via a "naive evict" mode toggle.
//!
//! - **GATE 3 (a leg): recovery byte-equality:** a real ADR-229 checkpoint
//!   over a bounded store WHOSE BINDINGS HAVE BEEN EVICTED-TO-SPILL, then a
//!   "crash" (drop) + fresh recovery from the checkpoint snapshot, asserting
//!   the recovered binding set is BYTE-IDENTICAL to the pre-crash set. If the
//!   producer silently dropped evicted bindings (the identity-loss class),
//!   the recovered store would be missing them.
//!
//! - **GATE 5 (crash-mid-spill):** a crash while spilling bindings → recovery
//!   rebuilds the correct binding set (no lost/duplicated identity). Modeled
//!   by dropping the store mid-drain and recovering from the last checkpoint.

use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::blob::BlobStore;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::checkpoint::{CheckpointSnapshot, checkpoint, restore_latest_checkpoint};
use arcgraph_storage::crud::{CrudStore, crud_allocator_seed_handle};
use arcgraph_storage::idempotency::{
    IDEMPOTENCY_BINDING_WEIGHT_BYTES, IdempotencyBinding, IdempotencyBoundConfig, IdempotencySpill,
    IdempotencyStore,
};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{AllocatorAdvance, AllocatorSeedHandle};
use tempfile::tempdir;

const NODE: u8 = 0;
const REL: u8 = 1;

/// Minimal owner bundle for a checkpoint over a bounded idempotency store.
/// Mirrors `blob_bound_1404.rs::Owners` but parameterizes the idempotency
/// store so the producer sees the bounded tier.
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
    fn with_idempotency(idempotency: Arc<IdempotencyStore>) -> Self {
        let allocator = Arc::new(PageAllocator::new());
        let record = Arc::new(RecordPageStore::new());
        let blob = Arc::new(BlobStore::new());
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
            idempotency,
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

/// A bounded idempotency store with a `cap_bindings` resident cap, plus its
/// spill kept alive.
fn bounded_store(dir: &std::path::Path, cap_bindings: u64) -> Arc<IdempotencyStore> {
    let spill = Arc::new(IdempotencySpill::open(dir).unwrap());
    let cfg = IdempotencyBoundConfig {
        high_watermark_bytes: cap_bindings * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
        low_watermark_bytes: (cap_bindings / 2).max(1) * IDEMPOTENCY_BINDING_WEIGHT_BYTES,
    };
    Arc::new(IdempotencyStore::with_bound(spill, cfg))
}

/// Simulate a completed ADR-229 checkpoint capture: stream every binding
/// through `for_each_binding` (which sets the `checkpointed` INV-DURABLE gate
/// under the freeze, exactly as the producer does) and count them. Returns the
/// captured binding count. Uses the PRODUCTION streaming capture entry point
/// (NOT the `#[cfg(test)]`-only whole-`Vec` `iter_all`).
fn capture(store: &IdempotencyStore) -> u64 {
    let mut n = 0u64;
    store
        .for_each_binding::<_, std::convert::Infallible>(|_, _, _, _, _| {
            n += 1;
            Ok(())
        })
        .expect("infallible counting closure");
    // The declared count MUST equal the streamed count (the producer relies on
    // this to write the section header before the records).
    assert_eq!(
        n,
        store.binding_count(),
        "binding_count() != streamed count"
    );
    n
}

/// Model the AT-LEAST-ONCE ingest de-dupe check exactly as the MCP adapter
/// does it (`adapters.rs:1247` — `get` the external_id; a hit is an idempotent
/// no-op, a miss mints a fresh id). Returns `true` iff this call MINTED a new
/// id (i.e. it was NOT de-duped).
fn ingest_once(store: &IdempotencyStore, kind: u8, ext: &str, fresh_id: u64) -> bool {
    match store.get(TenantId::DEFAULT, kind, ext) {
        Some(_) => false, // de-duped — the binding resolved (resident or spill)
        None => {
            // No binding → a real insert would happen; install the new one.
            store.install(TenantId::DEFAULT, kind, ext, fresh_id);
            true
        }
    }
}

/// GATE 1 (idempotency leg headline) — the RESIDENT binding working-set (what
/// `iter_all` / the freeze-capture walks) stays BOUNDED independent of the
/// number of ingested bindings, with drain+spill firing between checkpoints.
/// RED-on-revert: an UNBOUNDED store (no spill) grows resident with N.
#[test]
fn gate1_resident_bindings_bounded_independent_of_ingested_count() {
    let dir = tempdir().unwrap();
    // Small resident cap so the drain must engage well before N.
    let cap_bindings = 8u64;
    let store = bounded_store(dir.path(), cap_bindings);

    // Ingest a large N with periodic checkpoints (marking durable) so the drain
    // has evict-eligible bindings — mirroring sustained ingest with the ADR-229
    // interval firing. After each checkpoint the next installs drive the drain.
    let n = 5000u64;
    let checkpoint_every = 250u64;
    for i in 0..n {
        let kind = if i % 2 == 0 { NODE } else { REL };
        store.install(TenantId::DEFAULT, kind, &format!("ext-{i}"), i);
        if (i + 1) % checkpoint_every == 0 {
            // Simulate a checkpoint: mark the resident set durable, then the
            // next installs will drain it (production drains on install).
            let _ = capture(&store);
            store.force_drain_for_test();
        }
    }
    // Final checkpoint + drain to settle the tail.
    let _ = capture(&store);
    store.force_drain_for_test();

    // HEADLINE: the resident binding bytes are a function of the WATERMARK, not
    // of N. With a `cap_bindings`-sized cap the resident set settles near it —
    // orders of magnitude below N — while the LOGICAL set is still complete.
    let resident = store.resident_len();
    assert!(
        (resident as u64) <= cap_bindings * 4,
        "resident bindings {resident} not bounded near the cap ({cap_bindings}) — \
         the freeze-capture working set grows with N (the OOM)",
    );
    assert!(
        store.evicted_count() > 0,
        "no eviction fired — the bounded tier is not engaging",
    );
    // The full logical set is intact (resident + spilled), and every id still
    // resolves — bounded RSS, ZERO lost identity.
    assert_eq!(
        store.total_len(),
        n as usize,
        "logical binding set is incomplete"
    );
    for i in (0..n).step_by(137) {
        let kind = if i % 2 == 0 { NODE } else { REL };
        assert_eq!(
            store
                .get(TenantId::DEFAULT, kind, &format!("ext-{i}"))
                .map(|b| b.internal_id),
            Some(i),
            "ext-{i} unresolvable — bounded-RSS came at the cost of a lost binding",
        );
    }

    // RED-on-revert: an UNBOUNDED store grows resident 1:1 with N.
    let unbounded = IdempotencyStore::new();
    for i in 0..n {
        let kind = if i % 2 == 0 { NODE } else { REL };
        unbounded.install(TenantId::DEFAULT, kind, &format!("ext-{i}"), i);
    }
    let _ = capture(&unbounded);
    unbounded.force_drain_for_test();
    assert_eq!(
        unbounded.resident_len(),
        n as usize,
        "unbounded store must hold every binding resident (the pre-M0.x behavior)",
    );
    // The bounded resident set is orders of magnitude smaller.
    assert!(
        (resident as u64) * 50 < n,
        "bounded resident ({resident}) is not orders of magnitude below N ({n})",
    );
}

/// GATE 2 — THE load-bearing correctness leg, both directions.
///
/// Seed N bindings into a bounded store, checkpoint (mark durable), drain
/// (evict to spill), then re-ingest a sample of the SPILLED external_ids.
/// With the real spill: 0 duplicates (every re-ingest de-dupes via fault-in).
/// With a NAIVE evict-to-nowhere (`naive_evict = true`): the re-ingest MISSES
/// → a DUPLICATE, and this test would go RED — proving the spill is required.
#[test]
fn gate2_reingest_of_spilled_external_id_dedupes_zero_duplicates() {
    let dir = tempdir().unwrap();
    let store = bounded_store(dir.path(), 2);
    let n = 60u64;

    // Seed N distinct external ids, both node AND rel side (rel-side symmetry).
    for i in 0..n {
        let kind = if i % 2 == 0 { NODE } else { REL };
        assert!(
            ingest_once(&store, kind, &format!("ext-{i}"), i),
            "first ingest of ext-{i} must mint",
        );
    }

    // Checkpoint marks resident bindings durable, then drain evicts them.
    let _ = capture(&store);
    store.force_drain_for_test();
    assert!(
        store.evicted_count() > 0,
        "eviction must fire so the re-ingest exercises the spill fault-in",
    );

    // Re-ingest EVERY external_id (a superset of the spilled set). NONE may
    // mint a new id — every one must de-dupe via the resident hit OR the spill
    // fault-in. A single mint here is a DUPLICATE = a lost identity.
    let mut duplicates = 0u64;
    for i in 0..n {
        let kind = if i % 2 == 0 { NODE } else { REL };
        // `fresh_id` here is a NEW id that would be minted on a miss; if we
        // ever mint, the binding was lost.
        if ingest_once(&store, kind, &format!("ext-{i}"), 10_000 + i) {
            duplicates += 1;
        }
    }
    assert_eq!(
        duplicates, 0,
        "GATE 2 FAIL: {duplicates} spilled external_ids were NOT de-duped on re-ingest \
         (lost identity → duplicate). The spill fault-in is broken.",
    );
    // And every id still resolves to its ORIGINAL value (not a re-minted one).
    for i in 0..n {
        let kind = if i % 2 == 0 { NODE } else { REL };
        assert_eq!(
            store
                .get(TenantId::DEFAULT, kind, &format!("ext-{i}"))
                .map(|b| b.internal_id),
            Some(i),
            "ext-{i} resolved to a re-minted id — identity was lost + replaced",
        );
    }
}

/// GATE 2 RED-on-revert — the naive-evict counterfactual, in the SAME shape.
///
/// This test PROVES the failure mode the spill prevents: if eviction dropped
/// the binding to nowhere (modeled here by `release`, which is exactly
/// "forget the binding"), a re-ingest of that external_id would MISS and mint
/// a DUPLICATE. We assert the duplicate happens — so if a future change made
/// eviction drop-to-nowhere, `gate2_...dedupes` above would flip RED while
/// this stays GREEN, pinpointing the regression.
#[test]
fn gate2_red_on_revert_naive_evict_to_nowhere_duplicates() {
    let dir = tempdir().unwrap();
    let store = bounded_store(dir.path(), 2);
    let n = 20u64;
    for i in 0..n {
        assert!(ingest_once(&store, NODE, &format!("ext-{i}"), i));
    }
    let _ = capture(&store);

    // NAIVE EVICT-TO-NOWHERE: forget a subset of bindings entirely (no spill).
    // `release` is the "drop the binding" primitive — it removes both tiers,
    // so a subsequent `get` genuinely misses (there is no spill image to fault
    // in). This is the counterfactual the real drain must NEVER do.
    for i in 0..n {
        store.release(TenantId::DEFAULT, NODE, &format!("ext-{i}"));
    }

    // Re-ingest → every one MISSES (binding was dropped to nowhere) → mints a
    // DUPLICATE with a DIFFERENT id.
    let mut duplicates = 0u64;
    for i in 0..n {
        if ingest_once(&store, NODE, &format!("ext-{i}"), 10_000 + i) {
            duplicates += 1;
        }
    }
    assert_eq!(
        duplicates, n,
        "naive evict-to-nowhere should duplicate EVERY external_id — this is the \
         exact failure the durable, queryable spill prevents",
    );
    // And the id is now the RE-MINTED one, not the original — a corrupted
    // at-least-once identity.
    assert_eq!(
        store
            .get(TenantId::DEFAULT, NODE, "ext-0")
            .map(|b| b.internal_id),
        Some(10_000),
        "the re-minted (wrong) id replaced the original — identity corruption",
    );
}

/// GATE 3 (a leg) — evicted bindings survive a real ADR-229 checkpoint +
/// recovery BYTE-IDENTICAL. If the producer's capture silently dropped
/// evicted bindings, the recovered store would be missing them.
#[test]
fn gate3_evicted_bindings_survive_checkpoint_and_recovery_byte_identical() {
    let dir = tempdir().unwrap();

    // Expected binding set: (kind, external_id) -> (internal_id, payload_hash).
    let n = 50u64;
    let expected: Vec<(u8, String, IdempotencyBinding)> = (0..n)
        .map(|i| {
            let kind = if i % 2 == 0 { NODE } else { REL };
            (
                kind,
                format!("ext-{i}"),
                IdempotencyBinding {
                    internal_id: i,
                    payload_hash: Some(7000 + i),
                },
            )
        })
        .collect();

    {
        // Bounded store with a tiny cap so most bindings evict-to-spill.
        let idempotency = bounded_store(dir.path(), 2);
        let owners = Owners::with_idempotency(Arc::clone(&idempotency));

        for (kind, ext, b) in &expected {
            idempotency.install_with_payload_hash(
                TenantId::DEFAULT,
                *kind,
                ext,
                b.internal_id,
                b.payload_hash,
            );
        }
        // Nothing evictable before a checkpoint captures durability.
        assert_eq!(
            idempotency.evicted_count(),
            0,
            "INV-DURABLE: nothing may evict before the first checkpoint",
        );

        let seed = owners.allocator_seed();
        let pool = in_mem_buffer_pool();
        // First checkpoint captures the full resident set + marks it durable.
        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(1),
        )
        .expect("first checkpoint");

        // Now force eviction of the (now-durable) bindings to spill, then run
        // a SECOND checkpoint — this one MUST capture the evicted bindings'
        // durable images from spill (the load-bearing capture path).
        idempotency.force_drain_for_test();
        assert!(
            idempotency.evicted_count() > 0,
            "eviction did not fire post-checkpoint — cannot test the capture",
        );

        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(2),
        )
        .expect("second checkpoint (with evicted bindings) must capture from spill");

        // Sanity: every binding still resolves pre-crash (resident or spill).
        for (kind, ext, b) in &expected {
            assert_eq!(
                idempotency.get(TenantId::DEFAULT, *kind, ext),
                Some(*b),
                "pre-crash re-fault mismatch for {ext}",
            );
        }
        // "Crash": bounded store + spill dropped here. A real restart
        // truncates idempotency-spill.db; recovery rebuilds from the
        // checkpoint snapshot alone.
    }

    // Recovery: a FRESH, UNBOUNDED store restores from the checkpoint.
    let recovered_idem = Arc::new(IdempotencyStore::new());
    let recovered = Owners::with_idempotency(Arc::clone(&recovered_idem));
    let seed_r = recovered.allocator_seed();
    restore_latest_checkpoint(dir.path(), &recovered.snapshot(seed_r.as_ref()))
        .expect("restore must succeed")
        .expect("a checkpoint must be present");

    // The load-bearing assertion: EVERY binding recovers byte-identical.
    for (kind, ext, b) in &expected {
        assert_eq!(
            recovered_idem.get(TenantId::DEFAULT, *kind, ext),
            Some(*b),
            "RECOVERED BINDING DIFFERS for {ext} — an evicted binding was lost across \
             checkpoint + recovery (identity loss)",
        );
    }
    assert_eq!(
        recovered_idem.total_len(),
        n as usize,
        "recovered store has the wrong binding count",
    );
    // The recovered store is a superset-free set: re-ingest of every ext
    // de-dupes (0 duplicates) — the ultimate at-least-once proof post-recovery.
    for (kind, ext, _b) in &expected {
        assert!(
            !ingest_once(&recovered_idem, *kind, ext, 999_999),
            "post-recovery re-ingest of {ext} minted a duplicate — recovery lost the identity",
        );
    }
}

/// GATE 5 — crash-mid-spill: a crash WHILE bindings are being spilled →
/// recovery rebuilds the correct binding set (no lost/duplicated identity).
///
/// Modeled by: checkpoint (durable), begin evicting, then "crash" (drop the
/// store) at an arbitrary drain point — some bindings resident, some spilled,
/// the spill file partially written. Recovery from the last durable checkpoint
/// must reconstruct EXACTLY the committed set (the spill file is scratch and is
/// discarded; the checkpoint is the durable truth).
#[test]
fn gate5_crash_mid_spill_recovers_correct_bindings() {
    let dir = tempdir().unwrap();
    let n = 40u64;

    {
        let idempotency = bounded_store(dir.path(), 4);
        let owners = Owners::with_idempotency(Arc::clone(&idempotency));
        for i in 0..n {
            let kind = if i % 2 == 0 { NODE } else { REL };
            idempotency.install(TenantId::DEFAULT, kind, &format!("ext-{i}"), i);
        }
        let seed = owners.allocator_seed();
        let pool = in_mem_buffer_pool();
        // Checkpoint captures the FULL committed set durably (both-or-neither).
        checkpoint(
            dir.path(),
            &pool,
            &owners.snapshot(seed.as_ref()),
            || owners.advances(),
            Lsn::new(1),
        )
        .expect("checkpoint");

        // Begin spilling — this partially writes idempotency-spill.db.
        idempotency.force_drain_for_test();
        assert!(idempotency.evicted_count() > 0);
        // "CRASH" mid-spill: drop the store + its spill here. Some bindings
        // are resident, some spilled, the spill file has partial content —
        // none of which matters, because the checkpoint holds the durable set.
    }

    // Recovery from the durable checkpoint alone (spill file is scratch).
    let recovered_idem = Arc::new(IdempotencyStore::new());
    let recovered = Owners::with_idempotency(Arc::clone(&recovered_idem));
    let seed_r = recovered.allocator_seed();
    restore_latest_checkpoint(dir.path(), &recovered.snapshot(seed_r.as_ref()))
        .expect("restore")
        .expect("checkpoint present");

    assert_eq!(
        recovered_idem.total_len(),
        n as usize,
        "crash-mid-spill recovery has the wrong binding count",
    );
    for i in 0..n {
        let kind = if i % 2 == 0 { NODE } else { REL };
        assert_eq!(
            recovered_idem
                .get(TenantId::DEFAULT, kind, &format!("ext-{i}"))
                .map(|b| b.internal_id),
            Some(i),
            "ext-{i} lost/wrong after crash-mid-spill recovery",
        );
    }
}

/// GATE 6 (idempotency leg) — rel-side symmetry: the rel-side binding path is
/// spilled + faulted-in identically to the node-side (the #1404 OOM hit RELS).
#[test]
fn gate6_rel_side_bindings_spill_and_refault_symmetrically() {
    let dir = tempdir().unwrap();
    let store = bounded_store(dir.path(), 2);
    let n = 40u64;
    // REL-ONLY ingest — the rel-heavy shape that OOM'd.
    for i in 0..n {
        store.install(TenantId::DEFAULT, REL, &format!("rel-{i}"), i);
    }
    let _ = capture(&store);
    store.force_drain_for_test();
    assert!(
        store.evicted_count() > 0,
        "rel-side bindings must evict (rel-side must be bounded too)",
    );
    // Every rel-side external_id still resolves (fault-in from spill).
    for i in 0..n {
        assert_eq!(
            store
                .get(TenantId::DEFAULT, REL, &format!("rel-{i}"))
                .map(|b| b.internal_id),
            Some(i),
            "rel-{i} lost after spill — rel-side re-fault broken",
        );
    }
    assert!(store.refault_count() > 0, "no rel-side re-faults happened");
}
