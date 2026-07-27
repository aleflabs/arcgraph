//! #37 [A-1] — Multi-page crash atomicity in the **standalone**
//! (non-bundle-aware) index-write path.
//!
//! ## The hazard (issue #37)
//!
//! Before this fix, `PrimaryIndex::write` / `SecondaryIndex::write`
//! (the public `insert`/`upsert`/`remove` wrappers +
//! `bootstrap_from_mvcc`) drained each staged index page into its OWN
//! `WalRecordType::IndexPage` record via a per-page loop
//! (`drain_staged_emits`). A single logical index operation that
//! mutates 2-3 pages — a B-tree leaf **split** (`apply_leaf_op`) or an
//! **overflow-successor** allocation (`append_at_or_past_tail`) —
//! therefore emitted 2-3 independent WAL records. A crash *between*
//! two of those records left one sibling durable and the other not:
//! on replay the survivor is an **orphan page**. ADR-032 Invariant 13
//! ("orphan-page tolerance") + §R1 (record-type classification)
//! classify every legacy, non-bundle `IndexPage` record as an orphan
//! precisely because it is not atomically grouped with its siblings.
//! (§R2 Step 3c is a *different* rule — applying `IndexPage` entries
//! carried *inside a `CommitBundle`* — and is NOT the legacy-orphan
//! basis; #769 R1 NIT #4 corrected this cite.)
//!
//! The fix folds every page of one logical standalone op — plus the
//! grow_root SYSTEM root-pointer write — into ONE
//! `WalRecordType::CommitBundle` record
//! (`TxnManager::commit_index_pages_atomic`, realizing ADR-031 for the
//! standalone path). One CRC-framed record is crash-atomic: replay
//! applies ALL pages or NONE (a torn bundle fails CRC and is dropped,
//! ADR-031 §R5).
//!
//! ## Oracles
//!
//! The crash tests ([`crash_mid_split_op_leaves_no_orphan`],
//! [`crash_mid_overflow_successor_op_leaves_no_orphan`]) drive a
//! WAL-truncation sweep and assert TWO independent properties at every
//! crash offset (see [`assert_crash_atomic_over_sweep`]):
//!
//! 1. **No legacy orphan** — `orphan_pages_detected` stays at the
//!    construction baseline. `ReplayMetricsSnapshot::orphan_pages_detected`
//!    counts legacy `IndexPage` records (each is `+1` in
//!    `apply_legacy_index_page`); these are exactly the
//!    "torn-from-its-siblings" survivors #37 is about (ADR-032
//!    Invariant 13 + §R1). Pre-fix the workload emits these per page,
//!    so the count grows → RED.
//!
//! 2. **Bundle-quantized application** (the structural strengthening,
//!    #769 R1 finding #2). A truncation only ever drops a WAL *suffix*,
//!    so the surviving records are always a *prefix* of bundles
//!    `{B₁..Bⱼ}`. A genuinely all-or-nothing replay therefore applies
//!    *exactly* the pages of the `j` complete bundles, so the number of
//!    complete bundles pins the page count: `bundles_applied = j`
//!    ⇒ `index_pages_applied = Σᵢ pᵢ`. We assert that across the sweep
//!    `index_pages_applied` is a **well-defined, non-decreasing function
//!    of `bundles_applied`**. If a torn bundle were ever applied
//!    *partially* (k < pⱼ₊₁ of its pages, without the bundle counting
//!    as applied), the same `bundles_applied` value would appear with
//!    two different `index_pages_applied` values → the function is
//!    violated. This observes partial-bundle application, which the
//!    record-count oracle (1) structurally **cannot** (post-fix the
//!    workload emits zero legacy `IndexPage` records, so no truncation
//!    offset can manufacture an orphan — oracle 1 alone is near-vacuous
//!    post-fix; oracle 2 supplies the teeth).
//!
//! ### Why a metric oracle, not a reconstruct-and-walk oracle
//!
//! R1 finding #2 listed two acceptable strengthenings: the
//! bundle-quantized assertion (used here) OR reconstructing the index
//! over the replayed store and walking it for dangling/orphan pointers.
//! The walk is *confounded* on this surface and would risk a false
//! oracle (doctrine §4): `SecondaryPageStore` exposes no page
//! enumeration (a `DashMap`; you can only walk from the root), and the
//! root pointer is read from MVCC while `new()`'s fresh-root leaf is
//! emitted as a *legacy `IndexPage`* that replay routes to the
//! **primary** store as an orphan — so at low truncation offsets (no
//! insert bundle yet durable) the MVCC root references a page absent
//! from the secondary store, which a naive reachability walk would
//! mis-flag at the baseline. The bundle-quantization oracle observes
//! the identical all-or-nothing invariant directly from replay metrics
//! with none of that confound.
//!
//! ## RED-then-GREEN contract (doctrine §3)
//!
//! - **Pre-fix:** each standalone insert emits N non-atomic `IndexPage`
//!   records, NOT a bundle → `orphan_pages_detected` grows (oracle 1
//!   RED) AND `bundles_applied` stays flat while `index_pages_applied`
//!   climbs per surviving record, so one `bundles_applied` value maps
//!   to many `index_pages_applied` values (oracle 2 RED). Both asserts
//!   FAIL (exit 101).
//! - **Post-fix:** each standalone op emits ONE `CommitBundle` → every
//!   truncation lands on a clean bundle prefix → both oracles GREEN.
//!
//! The construction-only baseline is the `t == off_a` sweep point
//! (index built, zero inserts applied), so the workload's contribution
//! is isolated exactly. A non-vacuity guard asserts the full prefix
//! applies strictly more pages than the baseline (the staged pages
//! really did flow through the bundle path, not silently vanish).
//!
//! ## Hermetic, runs by default
//!
//! This is the in-process fault-injection variant the issue requires
//! to run under the default `cargo test` line (no subprocess). The
//! truncation sweeps (`crash_mid_split_op_leaves_no_orphan` for the
//! split path, `crash_mid_overflow_successor_op_leaves_no_orphan` for
//! the overflow-successor path — one fault-injection test PER failure
//! mode, doctrine §3) inject a crash at many byte offsets across the
//! workload's WAL region — the deterministic stand-in for a SIGKILL
//! landing between two page emits. The heavy 100-round
//! subprocess-SIGKILL harness over the same hazard sites lives at
//! `arcgraph-index/tests/m2e_wal_recovery_replay.rs` (`#[ignore]`).

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, NodeId, StringId, TenantId};
use arcgraph_index::{PropertyValue, SecondaryIndex, SecondaryKey, SecondaryPageStore};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    PageStoreTarget, PrimaryPageStoreHandle, ReplayMetricsSnapshot, SecondaryPageStoreHandle,
    WalConfig, WalWriter, recover_from_wal,
};
use tempfile::tempdir;

// ─── fixture helpers ──────────────────────────────────────────────

fn wal_config(dir: &Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 16,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// Bring up a WAL-backed secondary index on `dir`. Returns the writer
/// (caller flushes + shuts it down) plus the live index. The
/// `TxnManager` is kept alive inside the returned tuple via the index's
/// `Arc` clone of it.
fn build_secondary(dir: &Path) -> (WalWriter, Arc<SecondaryIndex>) {
    let writer = WalWriter::spawn(wal_config(dir)).expect("wal spawn");
    let handle = writer.handle();
    let txn_mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let secondary = Arc::new(
        SecondaryIndex::new(Arc::clone(&txn_mgr), alloc, Some(handle.clone()))
            .expect("SecondaryIndex::new"),
    );
    (writer, secondary)
}

/// Bulk unique-key inserts → forces leaf splits (and grow_root once
/// the root leaf fills). `LEAF_CAPACITY == 127`, so `n > 127`
/// guarantees ≥1 split. Mirrors the split hazard driver in
/// `m2e_wal_recovery_replay.rs`.
fn insert_unique(secondary: &SecondaryIndex, n: u64) {
    let tenant = TenantId::new(777);
    let label = LabelId::new(42);
    let property_key = StringId::new(1);
    for i in 0..n {
        let key = SecondaryKey::new(tenant, label, property_key, PropertyValue::U64(i + 1));
        secondary
            .insert(key, NodeId::new(i + 1))
            .expect("unique insert");
    }
}

/// Bulk duplicate-key inserts on ONE key → fills the 4 inline slots,
/// then the first overflow page (1017 slots), then forces the
/// **overflow-successor** allocation in `append_at_or_past_tail`
/// (the second hazard #37 names). Mirrors the overflow hazard driver
/// in `m2e_wal_recovery_replay.rs`.
fn insert_dups(secondary: &SecondaryIndex, n: u64) {
    let tenant = TenantId::new(888);
    let label = LabelId::new(99);
    let property_key = StringId::new(2);
    let key = SecondaryKey::new(tenant, label, property_key, PropertyValue::U64(123_456));
    for i in 0..n {
        secondary
            .insert(key, NodeId::new(1_000_000 + i))
            .expect("dup insert");
    }
}

/// Locate the single WAL segment file in `dir` (top-level `wal-*.log`;
/// the recovery spill subdir is skipped because it is a directory, not
/// a `.log` file).
fn segment_path(dir: &Path) -> PathBuf {
    let mut segs: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("read wal dir")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension().is_some_and(|x| x == "log"))
        .collect();
    segs.sort();
    assert_eq!(segs.len(), 1, "expected exactly one WAL segment in {dir:?}");
    segs.pop().expect("one segment")
}

fn segment_len(dir: &Path) -> u64 {
    std::fs::metadata(segment_path(dir))
        .expect("segment metadata")
        .len()
}

/// Shrink the WAL segment to `len` bytes — simulates a crash that left
/// only the first `len` bytes durable.
fn truncate_segment(seg: &Path, len: u64) {
    let f = std::fs::OpenOptions::new()
        .write(true)
        .open(seg)
        .expect("open segment for truncate");
    f.set_len(len).expect("set_len");
}

/// Replay the WAL in `dir` into FRESH primary+secondary page stores and
/// return the replay metrics snapshot. The #37 oracles read
/// `orphan_pages_detected` (no legacy orphan) plus the
/// `(bundles_applied, index_pages_applied)` pair (bundle-quantized
/// application) — see [`assert_crash_atomic_over_sweep`].
fn replay_metrics(dir: &Path) -> ReplayMetricsSnapshot {
    let txn_mgr = Arc::new(TxnManager::new());
    let primary: Arc<dyn PrimaryPageStoreHandle> = Arc::new(PrimaryPageStore::new());
    let secondary: Arc<dyn SecondaryPageStoreHandle> = Arc::new(SecondaryPageStore::new());
    let target = PageStoreTarget::new(primary, secondary);
    recover_from_wal(dir, txn_mgr, target, None)
        .expect("recover_from_wal")
        .metrics
}

// ─── Test 1: split path — clean-replay routes-through-bundle ───────
//
// NOTE (#769 R1 NIT #3): this test injects NO crash. It is a
// clean-replay assertion that the split workload routes through the
// atomic `CommitBundle` path (zero legacy orphans, pages applied via
// bundles). The actual fault-injection lives in
// `crash_mid_split_op_leaves_no_orphan`.

#[test]
fn standalone_leaf_split_routes_through_atomic_bundle() {
    // Baseline: index construction only (the new() fresh-root emit),
    // measured in its own WAL dir so we can subtract its contribution.
    let base_dir = tempdir().expect("base tempdir");
    {
        let (writer, _secondary) = build_secondary(base_dir.path());
        writer.handle().flush().expect("flush base");
        writer.shutdown().expect("shutdown base");
    }
    let base = replay_metrics(base_dir.path());

    // Full: 300 unique inserts → many leaf splits + ≥1 grow_root.
    let full_dir = tempdir().expect("full tempdir");
    {
        let (writer, secondary) = build_secondary(full_dir.path());
        insert_unique(&secondary, 300);
        writer.handle().flush().expect("flush full");
        writer.shutdown().expect("shutdown full");
    }
    let full = replay_metrics(full_dir.path());

    // #37 ROUTES-THROUGH-BUNDLE: the splitting workload adds ZERO orphan
    // pages on a clean replay — every multi-page split rides ONE atomic
    // CommitBundle. Pre-fix each split emits N non-atomic IndexPage
    // records → orphans grow → RED. (The torn-mid-op case is covered by
    // crash_mid_split_op_leaves_no_orphan.)
    assert_eq!(
        full.orphan_pages_detected, base.orphan_pages_detected,
        "leaf-split inserts must add ZERO orphan pages on replay \
         (base={}, full={}); pre-fix each split emits non-atomic \
         IndexPage records that replay classifies as orphans",
        base.orphan_pages_detected, full.orphan_pages_detected,
    );

    // Non-vacuity: the split pages really flowed through the bundle
    // path (applied as staged_pages), not silently dropped.
    assert!(
        full.index_pages_applied > base.index_pages_applied,
        "split workload must apply staged pages via CommitBundle \
         (base={}, full={})",
        base.index_pages_applied,
        full.index_pages_applied,
    );
}

// ─── Test 2: overflow path — clean-replay routes-through-bundle ────
//
// NOTE (#769 R1 NIT #3): this test injects NO crash. It is a
// clean-replay assertion that the overflow-successor workload routes
// through the atomic `CommitBundle` path. The actual fault-injection
// lives in `crash_mid_overflow_successor_op_leaves_no_orphan`.

#[test]
fn standalone_overflow_successor_routes_through_atomic_bundle() {
    let base_dir = tempdir().expect("base tempdir");
    {
        let (writer, _secondary) = build_secondary(base_dir.path());
        writer.handle().flush().expect("flush base");
        writer.shutdown().expect("shutdown base");
    }
    let base = replay_metrics(base_dir.path());

    // Full: 1030 duplicate inserts on one key → inline(4) + first
    // overflow page(1017) + overflow-successor allocations. Each
    // successor allocation is a 2-page emit (new tail + old tail's
    // updated `next` pointer) — the hazard #37 names.
    let full_dir = tempdir().expect("full tempdir");
    {
        let (writer, secondary) = build_secondary(full_dir.path());
        insert_dups(&secondary, 1030);
        writer.handle().flush().expect("flush full");
        writer.shutdown().expect("shutdown full");
    }
    let full = replay_metrics(full_dir.path());

    assert_eq!(
        full.orphan_pages_detected, base.orphan_pages_detected,
        "overflow-successor inserts must add ZERO orphan pages on \
         replay (base={}, full={}); pre-fix the (new-tail, old-tail) \
         pair emits two non-atomic IndexPage records → orphan on crash",
        base.orphan_pages_detected, full.orphan_pages_detected,
    );
    assert!(
        full.index_pages_applied > base.index_pages_applied,
        "overflow workload must apply staged pages via CommitBundle \
         (base={}, full={})",
        base.index_pages_applied,
        full.index_pages_applied,
    );
}

// ─── Shared crash-sweep oracle ─────────────────────────────────────

/// Sweep a WAL-truncation crash across the insert region
/// `[off_a, full_len]` of segment `seg` (under `dir`) and assert #37
/// crash-atomicity at EVERY crash offset, with the two oracles
/// documented at the module header:
///
/// * **Oracle 1 — no legacy orphan:** `orphan_pages_detected` equals
///   the construction baseline (`t == off_a` sweep point) at every
///   offset. Pre-fix each surviving non-atomic `IndexPage` record is
///   classified as an orphan (ADR-032 Invariant 13 + §R1) so the count
///   grows → RED.
/// * **Oracle 2 — bundle-quantized application:** truncation drops only
///   a WAL *suffix*, so survivors are always a *prefix* of bundles;
///   `index_pages_applied` must therefore be a well-defined,
///   non-decreasing FUNCTION of `bundles_applied` across the sweep. A
///   partially-applied torn bundle would make one `bundles_applied`
///   value map to two `index_pages_applied` values → violated. Pre-fix
///   the per-page `IndexPage` drain grows `index_pages_applied` while
///   `bundles_applied` stays flat between (rare) root-pointer bundles →
///   violated → RED. This is the partial-bundle teeth oracle 1 lacks.
///
/// The descending sweep matters: `set_len` only ever shrinks the file,
/// so we truncate from `full_len` down to `off_a`.
fn assert_crash_atomic_over_sweep(seg: &Path, dir: &Path, off_a: u64, full_len: u64) {
    const STEPS: u64 = 24;
    // (offset, orphan_pages_detected, bundles_applied, index_pages_applied)
    let mut points: Vec<(u64, u64, u64, u64)> = Vec::with_capacity(STEPS as usize + 1);
    for i in (0..=STEPS).rev() {
        let t = off_a + (full_len - off_a) * i / STEPS;
        truncate_segment(seg, t);
        let m = replay_metrics(dir);
        points.push((
            t,
            m.orphan_pages_detected,
            m.bundles_applied,
            m.index_pages_applied,
        ));
    }
    // i==STEPS (t==full_len) was pushed first; i==0 (t==off_a) last.
    let (_, base_orphan, _, base_pages) = *points.last().expect("≥1 sweep point");
    let (_, _, _, full_pages) = *points.first().expect("≥1 sweep point");

    // Oracle 1 — no legacy orphan at any crash offset.
    for &(t, orphan, _, _) in &points {
        assert_eq!(
            orphan, base_orphan,
            "crash at WAL offset {t} (off_a={off_a}, full_len={full_len}) left {orphan} orphan \
             pages; expected {base_orphan} — a torn standalone multi-page op must leave NO orphan \
             page (issue #37 acceptance: killed between emits ⇒ replay has no orphan)"
        );
    }

    // Oracle 2 — bundle-quantized: index_pages_applied is a FUNCTION of
    // bundles_applied (a torn bundle is never applied as a partial page
    // set). Same bundle-count ⇒ same page-count, at every offset.
    let mut by_bundles: HashMap<u64, u64> = HashMap::new();
    for &(t, _, bundles, pages) in &points {
        match by_bundles.entry(bundles) {
            Entry::Occupied(e) => assert_eq!(
                *e.get(),
                pages,
                "crash at WAL offset {t}: bundles_applied={bundles} observed with TWO different \
                 index_pages_applied values ({} vs {pages}) — a torn CommitBundle was applied \
                 PARTIALLY, violating the all-or-nothing invariant. Pre-fix each op emits N \
                 non-atomic IndexPage records, so bundles_applied stays flat while \
                 index_pages_applied climbs per surviving record → this assert fails (RED).",
                *e.get(),
            ),
            Entry::Vacant(v) => {
                v.insert(pages);
            }
        }
    }
    // Monotonicity: more complete bundles ⇒ at least as many applied
    // pages (prefix containment).
    let mut quanta: Vec<(u64, u64)> = by_bundles.into_iter().collect();
    quanta.sort_unstable();
    for w in quanta.windows(2) {
        assert!(
            w[1].1 >= w[0].1,
            "index_pages_applied must be non-decreasing in bundles_applied (prefix containment): \
             bundles {}→{} but pages {}→{}",
            w[0].0,
            w[1].0,
            w[0].1,
            w[1].1,
        );
    }

    // Non-vacuity: the full prefix applies strictly more pages than the
    // construction baseline, so the sweep really exercised multi-page
    // bundle boundaries (the pages flowed through bundles, not vanished).
    assert!(
        full_pages > base_pages,
        "workload must apply staged pages via CommitBundle across the sweep \
         (base={base_pages}, full={full_pages})"
    );
}

// ─── Test 3: crash BETWEEN emits, SPLIT path (truncation sweep) ────

#[test]
fn crash_mid_split_op_leaves_no_orphan() {
    // One dir: build the index, capture the post-construction WAL
    // offset, run the splitting workload, capture the full offset.
    let dir = tempdir().expect("tempdir");
    let off_a;
    let full_len;
    {
        let (writer, secondary) = build_secondary(dir.path());
        writer.handle().flush().expect("flush after new()");
        off_a = segment_len(dir.path()); // record boundary after new()
        insert_unique(&secondary, 300);
        writer.handle().flush().expect("flush after inserts");
        full_len = segment_len(dir.path());
        writer.shutdown().expect("shutdown");
    }
    assert!(
        full_len > off_a,
        "splitting workload must have grown the WAL (off_a={off_a}, full_len={full_len})"
    );
    let seg = segment_path(dir.path());
    assert_crash_atomic_over_sweep(&seg, dir.path(), off_a, full_len);
}

// ─── Test 4: crash BETWEEN emits, OVERFLOW-SUCCESSOR path ──────────
//
// #769 R1 MUST-FIX #1: the overflow-successor allocation in
// `append_at_or_past_tail` stages a structurally DIFFERENT 2-page set
// from a leaf split (a new overflow tail + the prior tail's updated
// `next` pointer, vs. a split's leaf + sibling + parent). Doctrine §3
// requires a fault-injection test PER failure mode; the split sweep
// (Test 3) does not exercise the torn overflow-tail window. This is its
// dedicated sweep.

#[test]
fn crash_mid_overflow_successor_op_leaves_no_orphan() {
    // 2100 duplicate inserts on ONE key: inline(4) + first overflow
    // page(1017) + ≥2 overflow-successor allocations (2100 − 1021 =
    // 1079 ⇒ a full second overflow page + a third successor). Each
    // successor allocation is a 2-page emit (new tail + old tail's
    // updated `next`) — the second hazard #37 names. One key never
    // splits the B-tree, so this path stays in the overflow chain (no
    // grow_root), isolating the overflow-successor failure mode.
    let dir = tempdir().expect("tempdir");
    let off_a;
    let full_len;
    {
        let (writer, secondary) = build_secondary(dir.path());
        writer.handle().flush().expect("flush after new()");
        off_a = segment_len(dir.path()); // record boundary after new()
        insert_dups(&secondary, 2100);
        writer.handle().flush().expect("flush after inserts");
        full_len = segment_len(dir.path());
        writer.shutdown().expect("shutdown");
    }
    assert!(
        full_len > off_a,
        "overflow-successor workload must have grown the WAL (off_a={off_a}, full_len={full_len})"
    );
    let seg = segment_path(dir.path());
    assert_crash_atomic_over_sweep(&seg, dir.path(), off_a, full_len);
}
