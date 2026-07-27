//! M6.1/M6.2 — `m6_release_enforcement_is_not_debug_only`.
//!
//! Proves the MECH-E1..E8 eviction enforcement and OOC-1 spill enforcement
//! are NOT gated behind `cfg(debug_assertions)` (or any other debug-only
//! mechanism) — i.e. the mechanisms run identically, and are equally
//! load-bearing, in a `--release` build. Per the standing lesson: a
//! `cfg(not(debug_assertions))` (or equivalent) enforcement gate that
//! silently no-ops in the debug builds CI normally runs is vacuous —
//! the reverse failure mode (enforcement that ONLY works in debug and
//! silently degrades in release, the actual production configuration)
//! is exactly what this gate targets, since release is what ships.
//!
//! Four legs:
//!
//! 1. **Static audit**: `evict_for_capacity` and every MECH-E helper it
//!    calls (`is_clean`, `evict_dirty_via_checkpointer`,
//!    `remove_cached_page_if_unpinned`) contain NO `cfg(debug_assertions)`
//!    or `debug_assert!`-gated enforcement — the mechanism's control
//!    flow is unconditional. This test greps the source directly so a
//!    future edit that adds a debug-only enforcement branch fails loudly
//!    here rather than silently shipping a release-mode no-op.
//! 2. **Spill static audit**: the production portion of `spill.rs` contains
//!    no `cfg(debug_assertions)` or `debug_assert!` enforcement. Quota,
//!    volume-headroom, stale-epoch, framing-integrity, and mandatory-
//!    encryption decisions therefore remain present in optimized builds.
//! 3. **Eviction behavioral leg, compiled in `--release`**: re-runs the
//!    decisive MECH-E3 correctness assertion (dirty page never evicted
//!    before its checkpointer home write durably completes) under a
//!    RELEASE profile build of THIS test binary — proving the mechanism
//!    is not merely correct in the `debug_assertions`-enabled dev
//!    profile the rest of this PR's gates run under by default.
//! 4. **Spill behavioral leg, compiled in `--release` with
//!    `fault-injection`**: enables tenant encryption and proves a run is
//!    obligatorily encrypted, then crosses a deliberately tiny quota and
//!    proves the reject remains a typed `ResourceExhausted` carrying the
//!    already-spilled byte count.
//!
//! Run the release leg explicitly:
//! ```ignore
//! cargo test -p arcgraph-storage --release --features fault-injection --test m6_release_enforcement_is_not_debug_only
//! ```

use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::io::{PageIo, PosixPageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::STORE_RECORD;

/// Static audit: no `cfg(debug_assertions)` / debug-only gating inside
/// the MECH-E enforcement path. Greps the actual shipped source (not a
/// copy), so it audits exactly what `cargo build --release` compiles.
#[test]
fn evict_for_capacity_control_flow_has_no_debug_only_gate() {
    let source = include_str!("../src/page_store.rs");
    let mech_e_start = source
        .find("fn dirty_page_key(")
        .expect("dirty_page_key must exist — MECH-E5 identity helper");
    let mech_e_end = source
        .find("// ─────────────────────────────────────────────────────────────────\n    // ADR-140-amendment-01 — the pin-coupled concurrent flush surface")
        .expect("the MECH-E block must end before the pre-existing pin-coupled surface section");
    assert!(
        mech_e_end > mech_e_start,
        "MECH-E block boundaries not found in the expected order — page_store.rs \
         was restructured; update this audit's anchors"
    );
    let mech_e_block = &source[mech_e_start..mech_e_end];

    assert!(
        !mech_e_block.contains("cfg(debug_assertions)")
            && !mech_e_block.contains("cfg(not(debug_assertions))"),
        "MECH-E1..E8 enforcement block contains a `cfg(debug_assertions)` gate \
         — release-mode enforcement would silently differ from debug-mode. \
         The eviction mechanism's control flow must be unconditional."
    );
    // `debug_assert!`/`debug_assert_eq!` ARE permitted elsewhere in the
    // crate (pure sanity checks over an already-impossible-in-correct-
    // code condition), but NOT inside the MECH-E block itself: an
    // enforcement decision (evict vs retain) must never be a
    // debug_assert (which compiles to nothing in release — the ONE
    // thing that MUST NOT compile to nothing is the mechanism's actual
    // reclaim/retain decision).
    assert!(
        !mech_e_block.contains("debug_assert!") && !mech_e_block.contains("debug_assert_eq!"),
        "MECH-E1..E8 block contains a debug_assert — if this assert is part \
         of the enforcement DECISION (not a pure diagnostic), it silently \
         no-ops in release. Move any load-bearing check to an unconditional \
         check (or justify the debug_assert as diagnostic-only in this test's \
         audit)."
    );
}

/// Static audit for M6.2 OOC-1: none of the production spill decisions may
/// compile away with debug assertions. Test-only retention and corruption
/// controls are permitted behind `feature = "fault-injection"`; the quota,
/// headroom, identity, integrity, and encryption policy itself is not.
#[test]
fn spill_enforcement_control_flow_has_no_debug_only_gate() {
    let source = include_str!("../src/spill.rs");
    let production = source
        .split_once("#[cfg(test)]\nmod tests")
        .map_or(source, |(production, _tests)| production);

    assert!(
        !production.contains("cfg(debug_assertions)")
            && !production.contains("cfg(not(debug_assertions))"),
        "M6.2 spill enforcement contains a `cfg(debug_assertions)` gate \
         — quota, headroom, stale-epoch, integrity, and mandatory-encryption \
         decisions must compile into the shipped release binary"
    );
    assert!(
        !production.contains("debug_assert!") && !production.contains("debug_assert_eq!"),
        "M6.2 spill production code contains a debug assertion; a \
         load-bearing spill enforcement decision must be unconditional"
    );
}

/// Release-fault-injection behavioral pin for M6.2. The feature gate makes
/// this leg part of the same optimized CI lane as the bounded retention and
/// corruption seams, while the exercised policy and accounting calls are the
/// unconditional production implementations.
#[cfg(feature = "fault-injection")]
#[test]
fn spill_mandatory_encryption_and_typed_quota_hold_under_this_build_profile() {
    use arcgraph_storage::spill::{
        SpillEncryptionPolicy, SpillError, SpillManager, SpillManagerConfig, SpillQueryConfig,
        SpillRejectReason,
    };

    let dir = tempfile::tempdir().expect("create spill release-gate data dir");
    let manager = SpillManager::new(SpillManagerConfig::new(dir.path()))
        .expect("construct production spill manager");

    // Charge real allocation-block deltas: four units admit the run metadata
    // plus header, and one more admits the first block-sized encrypted frame.
    // The second frame needs another physical unit and must be rejected.
    let unit = manager
        .volume_space()
        .expect("measure release-gate volume")
        .allocation_unit_bytes;
    let quota = unit * 5;
    let batch = vec![0x6D; unit as usize];
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, 0x6201, 0, quota);
    config.spill_quota_bytes = Some(quota);
    config.encryption = SpillEncryptionPolicy {
        tenant_encryption_enabled: true,
        force_encryption: false,
    };
    let query = manager
        .begin_query(config)
        .expect("begin encrypted spill query");
    let mut writer = query.create_run().expect("create encrypted spill run");

    assert!(
        writer.is_encrypted(),
        "tenant encryption must mandate spill encryption in this build profile; \
         it may not be debug-only or independently disabled"
    );
    writer
        .append_batch(&batch)
        .expect("first encrypted frame fits the tiny quota");
    let spilled_before_reject = query.spilled_bytes();
    assert!(
        spilled_before_reject > 0,
        "successful spill must be reflected in quota accounting"
    );

    let error = writer
        .append_batch(&batch)
        .expect_err("second encrypted frame must cross the tiny quota");
    match error {
        SpillError::ResourceExhausted {
            reason: SpillRejectReason::TenantQuota,
            requested_bytes,
            spilled_bytes,
            limit_bytes,
            available_bytes,
            ..
        } => {
            assert!(requested_bytes > 0, "reject must carry the write delta");
            assert_eq!(spilled_bytes, spilled_before_reject);
            assert_eq!(limit_bytes, quota);
            assert_eq!(available_bytes, None);
        }
        other => panic!(
            "quota enforcement must remain a typed spilled_bytes-carrying reject \
             in release; got {other:?}"
        ),
    }

    // A quota refusal occurs before touching the file, so the already-written
    // frame remains sealable and readable. This also proves the mandatory
    // encryption path is operational rather than merely setting a flag.
    let run = writer
        .finish()
        .expect("seal encrypted run after quota reject");
    assert!(run.is_encrypted());
    let mut reader = run
        .into_reader(query.epoch())
        .expect("open encrypted run under its active epoch");
    let restored = reader
        .next_batch()
        .expect("decrypt frame")
        .expect("one encrypted frame");
    assert_eq!(restored.as_ref(), batch.as_slice());
    drop(restored);
    assert!(reader.next_batch().expect("reach run EOF").is_none());
}

fn new_store(dir: &std::path::Path, cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> =
        Arc::new(PosixPageIo::open_or_create(dir.join("record.store")).expect("open page io"));
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 32,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

/// Behavioral leg: the decisive MECH-E3 correctness property (a dirty
/// page's durable home write must complete before eviction reclaims its
/// RAM copy) holds under WHATEVER profile this test binary was compiled
/// with — including `--release`, where `debug_assert!`/`cfg(debug_assertions)`
/// gates compile to nothing. If MECH-E3's enforcement were accidentally
/// riding on a debug-only mechanism, this exact test would silently pass
/// in dev and silently misbehave in release; running it explicitly under
/// `--release` (see the module doc) is what closes that gap.
#[test]
fn mech_e3_holds_under_this_build_profile() {
    let dir = tempfile::tempdir().unwrap();
    let store = new_store(dir.path(), 4);
    let pid = PageId::new(1);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();

    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = store.clone();
    let records_target: Arc<dyn PageFlushTarget> = store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    store.attach_m6_dirty_page_table(dpt.clone());
    store.attach_m6_checkpointer(checkpointer.clone());

    {
        let latch =
            RecordPageBackend::latch_for_tenant(store.as_ref(), TenantId::DEFAULT, pid).unwrap();
        latch.write().as_mut()[PAGE_SIZE - 1] = 0xEE;
    }
    dpt.mark_dirty(
        DirtyPageKey {
            tenant_id: TenantId::DEFAULT,
            store_id: STORE_RECORD,
            page_no: pid.raw(),
        },
        Lsn::new(1),
    );

    // Force capacity pressure.
    store
        .install_fresh(PageId::new(2), PageType::Node, TenantId::DEFAULT)
        .unwrap();
    let evicted = store.evict_for_capacity(1).unwrap();
    assert_eq!(
        evicted, 1,
        "the dirty page must be reclaimed via the checkpointer handshake \
         under this build profile (release or debug)"
    );

    // The checkpointer must have durably written the home BEFORE
    // reclaim — verified directly against the raw disk bytes.
    let io: Arc<dyn PageIo> = Arc::new(
        PosixPageIo::open(dir.path().join("record.store")).expect("reopen disk file directly"),
    );
    let mut buf = [0u8; PAGE_SIZE];
    io.read_page(pid, &mut buf).expect("read raw disk page");
    assert_eq!(
        buf[PAGE_SIZE - 1],
        0xEE,
        "MECH-E3 violated under this build profile: the disk home does not \
         carry the committed mutation despite the page having been reclaimed \
         — durable-home-write-before-reclaim did not hold"
    );

    // Round-trip: fault back in and confirm the byte.
    store.fault_in(pid).unwrap();
    let latch = store.latch(pid).unwrap();
    assert_eq!(latch.read().as_ref()[PAGE_SIZE - 1], 0xEE);
}
