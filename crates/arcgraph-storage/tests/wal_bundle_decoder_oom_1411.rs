//! #1411 rc-BLOCKER regression: the `CommitBundle` decoder must NOT
//! pre-allocate a container sized to an untrusted on-wire `u32` element
//! count before validating a single element byte.
//!
//! The bug: `decode_commit_bundle_v1..v8` read each section's element
//! count as an untrusted `u32` straight off the on-disk bytes (the
//! crash-recovery and spill-reload paths are attacker-influenceable) and
//! then called `Vec::with_capacity(count)` or
//! `HashMap::with_capacity(count)`. A crafted count near `u32::MAX`
//! (~4.29e9) forced a multi-gigabyte pre-alloc, an OOM/DoS on the
//! recovery path (surfaced by the #1287 CommitBundle fuzz target). The
//! canonical repro is a 33-byte v8 bundle with `n_acl_grants =
//! 0xffff_ff00` and 1 payload byte remaining.
//!
//! The fix caps the capacity hint at `remaining_bytes / MIN_ELEM`, so a
//! nonsense count is refused up front and the decode then hits the SAME
//! in-loop overrun/truncation guard, returning `WalCorruption`.
//!
//! # Why this test lives in its own integration binary
//!
//! It installs a **peak-single-allocation-tracking** `#[global_allocator]`
//! so the RED-on-revert proof is DETERMINISTIC and PLATFORM-INDEPENDENT.
//! On macOS the naive "does it OOM?" check does NOT discriminate: the
//! system allocator lazily overcommits a ~172 GB `Vec<AclGrantEntry>`
//! virtual mapping without committing physical pages (and macOS ignores
//! `RLIMIT_AS`), so a reverted (unbounded) decoder still returns `Err`
//! from the later read guard and the test would falsely PASS. This
//! allocator records the *requested* size of the largest single
//! allocation during the decode call — which is ~172 GB in the RED
//! (reverted) state and a few bytes in the GREEN (fixed) state —
//! regardless of whether the OS chooses to back it. A dedicated binary
//! keeps the custom global allocator from perturbing the lib test suite.
//!
//! RED-on-revert (verified): reverting the v8 `n_acl` bound to raw
//! `Vec::with_capacity(n_acl)` makes `max_single_alloc` for the repro
//! jump to ~1.7e11 bytes and this test FAILS its `< THRESHOLD` assert.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use arcgraph_core::{ArcGraphError, TenantId};
use arcgraph_storage::wal::bundle::{decode_commit_bundle_v1, decode_commit_bundle_v8};
use arcgraph_storage::wal::spill::load_one_spill_file;

/// A pass-through allocator that, while ARMED, records the size of the
/// largest single `alloc` request. Only the *request size* is recorded
/// (before delegating to the system allocator), so it discriminates the
/// unbounded pre-alloc even on overcommitting OSes that would satisfy it.
struct PeakAlloc;

static ARMED: AtomicBool = AtomicBool::new(false);
static MAX_SINGLE_ALLOC: AtomicUsize = AtomicUsize::new(0);

// SAFETY: `PeakAlloc` delegates every `alloc`/`dealloc`/`realloc` to the
// standard `System` allocator with the exact same `Layout`, so all
// GlobalAlloc safety invariants (valid layout in, pointer from the same
// allocator out, matching layout on free) are upheld by `System`. The
// only added behavior is a relaxed atomic `fetch_max` of the requested
// size, which allocates nothing and touches no returned memory.
unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            MAX_SINGLE_ALLOC.fetch_max(layout.size(), Ordering::Relaxed);
        }
        // SAFETY: forwarding the caller's valid `layout` unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `System.alloc` with this same `layout`.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if ARMED.load(Ordering::Relaxed) {
            MAX_SINGLE_ALLOC.fetch_max(new_size, Ordering::Relaxed);
        }
        // SAFETY: `ptr`/`layout` came from a prior `System` alloc; `new_size`
        // is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

/// Serializes the armed measurement region. The `ARMED`/`MAX_SINGLE_ALLOC`
/// counters are process-global, so two concurrently-running tests would
/// cross-contaminate each other's peak (cargo runs `#[test]`s on parallel
/// threads by default). This lock makes each measurement mutually
/// exclusive so the reported peak is exactly the allocation surface of the
/// single `f` under measurement — deterministic regardless of scheduling.
static MEASURE_LOCK: Mutex<()> = Mutex::new(());

/// Run `f` with allocation-peak tracking armed; return the largest single
/// allocation-request size observed during `f`. The `MEASURE_LOCK` guard is
/// acquired BEFORE arming (so the lock's own bookkeeping is never counted)
/// and the process-global counters are exclusive for the armed window.
fn measure_peak_alloc<T>(f: impl FnOnce() -> T) -> (T, usize) {
    // Acquire (and hold) the serialization lock before arming so no other
    // measurement's allocations land in this window, and so acquiring the
    // lock itself is not recorded. `.unwrap_or_else` recovers a poisoned
    // lock (a prior test panicked mid-measurement) — the `()` payload is
    // trivially valid, we only need mutual exclusion.
    let _guard = MEASURE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    MAX_SINGLE_ALLOC.store(0, Ordering::SeqCst);
    ARMED.store(true, Ordering::SeqCst);
    let out = f();
    ARMED.store(false, Ordering::SeqCst);
    let peak = MAX_SINGLE_ALLOC.load(Ordering::SeqCst);
    (out, peak)
}

/// The canonical 33-byte `.v8_acl_oom_repro` fixture (#1411): a well-
/// formed v8 header with every section count = 0 up to the trailing
/// `n_acl_grants = 0xffff_ff00` (~4.29e9), with 1 payload byte remaining.
const V8_ACL_OOM_REPRO: [u8; 33] = [
    0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // commit_lsn = 3
    0x00, 0x00, 0x00, 0x00, // n_mvcc = 0
    0x00, 0x00, 0x00, 0x00, // n_staged_pages = 0
    0x00, 0x00, 0x00, 0x00, // n_allocator_advances = 0
    0x00, 0x00, 0x00, 0x00, // n_vector_pages = 0
    0x00, 0x00, 0x00, 0x00, // n_idempotency_bindings = 0
    0x00, 0xff, 0xff, 0xff, // n_acl_grants = 0xffff_ff00 (bytes 28..32)
    0xff, // 1 trailing byte
];

/// Any single allocation during the decode of a 33-byte input must stay
/// far below this. The bounded decoder allocates only small housekeeping;
/// the reverted (unbounded) decoder requests ~172 GB in one call.
const PEAK_ALLOC_THRESHOLD: usize = 1 << 20; // 1 MiB — generous headroom.

#[test]
fn v8_acl_oom_repro_decodes_without_giant_alloc() {
    let (res, peak) =
        measure_peak_alloc(|| decode_commit_bundle_v8(&V8_ACL_OOM_REPRO, TenantId::DEFAULT));

    // The decode must reject the crafted bundle...
    assert!(
        res.is_err(),
        "crafted v8 acl repro must be rejected, got Ok"
    );

    // ...and it must NOT have tried to pre-allocate a container sized to
    // the ~4.29e9 element count. RED-on-revert: with the `n_acl` bound
    // removed this peak is ~1.7e11 bytes (n_acl * sizeof(AclGrantEntry))
    // and this assertion fails.
    assert!(
        peak < PEAK_ALLOC_THRESHOLD,
        "decode of a 33-byte bundle requested a {peak}-byte single allocation \
         (threshold {PEAK_ALLOC_THRESHOLD}) — the untrusted n_acl count was not \
         bounded before Vec::with_capacity (#1411 regression)"
    );
}

/// Synthetic max-count on the v1 mvcc section (the first with_capacity a
/// v1 bundle hits): a `u32::MAX` count with no element bytes must reject
/// without a giant pre-alloc.
#[test]
fn v1_max_mvcc_count_decodes_without_giant_alloc() {
    // commit_lsn u64 = 1, n_mvcc u32 = u32::MAX, no element bytes.
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&1u64.to_le_bytes());
    bytes.extend_from_slice(&u32::MAX.to_le_bytes());

    let (res, peak) = measure_peak_alloc(|| decode_commit_bundle_v1(&bytes, TenantId::DEFAULT));

    assert!(res.is_err(), "u32::MAX n_mvcc must be rejected");
    assert!(
        peak < PEAK_ALLOC_THRESHOLD,
        "v1 decode requested a {peak}-byte single allocation for a truncated \
         u32::MAX mvcc count (threshold {PEAK_ALLOC_THRESHOLD}) — #1411 regression"
    );
}

// ─── #1411 spill-reload path (the REAL crash-recovery twin) ──────────
//
// The decoder tests above cover `decode_commit_bundle_*` in isolation.
// This test covers the SEPARATE, load-bearing spill-reload path reached
// at recovery via `WalReplayer::final_drain → load_all_spill_bundles →
// load_one_spill_file`. That function reads `n_bundles` as an untrusted
// u32 from the spill-file header (bytes 12..16) and, pre-fix, called
// `Vec::with_capacity(n_bundles)` before validating a single entry byte
// — the same OOM/DoS class as the decoder, on the path the #1411 issue
// title literally names ("spill-reload").
//
// CRC ordering (verified in `load_one_spill_file`): the crc32c trailer is
// validated (spill.rs) BEFORE the `n_bundles` read and the
// `with_capacity` call. So the crafted file MUST carry a VALID crc32c
// over bytes `0..(size-4)` — computed with the SAME `crc32c::crc32c` the
// writer uses (spill.rs write path) — or the decode bails at the CRC
// check and never reaches the allocation site (proving nothing). We
// craft a header-only file (zero entries on the wire) with
// `n_bundles = u32::MAX`, so after the header parse `cursor == crc_offset
// == 36`, the bounded hint is `min(u32::MAX, 0/13) == 0`, and the first
// loop iteration hits the existing `cursor + 13 > crc_offset` truncation
// guard → `WalCorruption`. Reverting the bound to
// `Vec::with_capacity(n_bundles)` requests
// `u32::MAX * sizeof(DecodedCommitBundle)` bytes in one call ≫ 1 MiB.

/// Spill-file fixed header size (offsets 0..36 per spill.rs module doc).
const SPILL_HEADER_SIZE: usize = 36;

/// Build a well-formed spill-file *header* (36 bytes) whose untrusted
/// `n_bundles` field (bytes 12..16) is `u32::MAX`, followed by NO entry
/// bytes and a VALID trailing crc32c over the 36 header bytes. Total = 40
/// bytes. Reaches the `with_capacity(n_bundles)` site and then the
/// in-loop truncation guard.
fn craft_malicious_spill_file(n_bundles: u32) -> Vec<u8> {
    let mut header = Vec::with_capacity(SPILL_HEADER_SIZE);
    header.extend_from_slice(b"ARCGSPIL"); // 0..8  magic
    header.extend_from_slice(&1u16.to_le_bytes()); // 8..10 spill_format_version = 1
    header.extend_from_slice(&0u16.to_le_bytes()); // 10..12 reserved
    header.extend_from_slice(&n_bundles.to_le_bytes()); // 12..16 n_bundles (untrusted)
    header.extend_from_slice(&0u64.to_le_bytes()); // 16..24 min_commit_lsn
    header.extend_from_slice(&0u64.to_le_bytes()); // 24..32 max_commit_lsn
    header.extend_from_slice(&0u32.to_le_bytes()); // 32..36 reserved
    assert_eq!(header.len(), SPILL_HEADER_SIZE, "header must be 36 bytes");

    // Valid crc32c over bytes 0..(size-4) — same fn the writer uses, so
    // the reader's CRC check (which precedes the alloc site) passes and
    // execution reaches `with_capacity(n_bundles)`.
    let crc = crc32c::crc32c(&header);
    header.extend_from_slice(&crc.to_le_bytes());
    header
}

#[test]
fn spill_max_n_bundles_reloads_without_giant_alloc() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("evil.spill");
    let bytes = craft_malicious_spill_file(u32::MAX);
    std::fs::write(&path, &bytes).expect("write spill fixture");

    let (res, peak) = measure_peak_alloc(|| load_one_spill_file(&path));

    // The crafted file must be rejected as corrupt (the crc is valid so
    // it reaches the entry loop, which finds no entry bytes for the
    // claimed count → truncated-entry-header WalCorruption).
    assert!(
        matches!(res, Err(ArcGraphError::WalCorruption { .. })),
        "crafted spill file with n_bundles=u32::MAX must return WalCorruption, got {res:?}"
    );

    // ...and it must NOT have pre-allocated a container sized to the
    // ~4.29e9 element count. RED-on-revert: with the `n_bundles` bound
    // removed this peak is `u32::MAX * sizeof(DecodedCommitBundle)` bytes
    // (≫ 1 MiB) and this assertion fails.
    assert!(
        peak < PEAK_ALLOC_THRESHOLD,
        "reload of a 40-byte spill file requested a {peak}-byte single allocation \
         (threshold {PEAK_ALLOC_THRESHOLD}) — the untrusted n_bundles header field \
         was not bounded before Vec::with_capacity (#1411 spill-reload regression)"
    );
}

/// Positive control: a spill file with a *truthful* small `n_bundles`
/// where the entries are genuinely absent still rejects (truncated) but
/// NEVER over-allocates — the bound is a no-op for honest small counts.
#[test]
fn spill_small_truthful_n_bundles_reloads_without_giant_alloc() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("small.spill");
    // Claim 1 bundle but supply no entry bytes: honest-but-truncated.
    let bytes = craft_malicious_spill_file(1);
    std::fs::write(&path, &bytes).expect("write spill fixture");

    let (res, peak) = measure_peak_alloc(|| load_one_spill_file(&path));

    assert!(
        matches!(res, Err(ArcGraphError::WalCorruption { .. })),
        "truncated 1-bundle spill file must return WalCorruption, got {res:?}"
    );
    assert!(
        peak < PEAK_ALLOC_THRESHOLD,
        "small-count spill reload requested a {peak}-byte single allocation \
         (threshold {PEAK_ALLOC_THRESHOLD}) — #1411 regression"
    );
}
