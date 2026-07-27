//! User-space buffer pool over 8 KiB pages.
//!
//! Covers roadmap tasks M1-10 (`Frame`), M1-11 (`PageTable`),
//! M1-12 (`ClockSweepEvictor`), M1-13 (`BufferPool` pin / unpin /
//! mark_dirty / flush_all) and M1-14 (split read/write pool; default
//! 20% write / 80% read per design-v2 §3.4). No mmap on the hot
//! path — see ADR-001.
//!
//! Latency budget (from `docs/arcgraph-design-v2.md` §A.3):
//! pin/unpin cache hit ≤ 200 ns single-threaded. Measured 23 ns
//! in `target/criterion/buffer_pool/pin_read_cache_hit` on
//! aarch64-apple-darwin (see `docs/benchmarks/M1.md`). The slow path
//! (miss → evict → read) is bounded by `PageIo::read_page`, which
//! is out of scope for this module.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use arcgraph_core::{ArcGraphError, PAGE_SIZE, PageId, Result, TenantId};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::io::{PageBuf, PageIo};
use crate::metrics::{MetricsSink, StoragePageKind};

/// Opaque frame identifier — an index into the buffer pool's frame array.
pub type FrameId = usize;

// ---------- Frame (M1-10) ----------------------------------------------------

/// One physical slot in the buffer pool. Holds at most one page at a
/// time, along with the metadata required for pin / eviction / flush.
pub struct Frame {
    id: FrameId,
    pin_count: AtomicU32,
    ref_bit: AtomicBool,
    dirty: AtomicBool,
    tenant: AtomicU64,
    page_id: Mutex<Option<PageId>>,
    data: RwLock<Box<PageBuf>>,
}

impl Frame {
    fn new(id: FrameId) -> Self {
        Self {
            id,
            pin_count: AtomicU32::new(0),
            ref_bit: AtomicBool::new(false),
            dirty: AtomicBool::new(false),
            tenant: AtomicU64::new(TenantId::DEFAULT.raw()),
            page_id: Mutex::new(None),
            data: RwLock::new(Box::new([0u8; PAGE_SIZE])),
        }
    }

    /// Frame index within the pool.
    #[must_use]
    pub fn id(&self) -> FrameId {
        self.id
    }

    /// Current pin count (for diagnostics).
    #[must_use]
    pub fn pin_count(&self) -> u32 {
        self.pin_count.load(Ordering::Acquire)
    }

    /// Is this frame currently pinned?
    #[must_use]
    pub fn is_pinned(&self) -> bool {
        self.pin_count() > 0
    }

    /// Is this frame dirty (has in-memory writes not yet flushed)?
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    /// Tenant that last faulted a page into this frame.
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        TenantId::new(self.tenant.load(Ordering::Acquire))
    }

    /// Page currently held, if any.
    #[must_use]
    pub fn current_page(&self) -> Option<PageId> {
        *self.page_id.lock()
    }

    fn pin(&self) {
        self.pin_count.fetch_add(1, Ordering::AcqRel);
        self.ref_bit.store(true, Ordering::Release);
    }

    fn unpin(&self) {
        // Saturating subtract guards against an API misuse bug from
        // poisoning the whole pool.
        let prev = self
            .pin_count
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                Some(count.saturating_sub(1))
            })
            .expect("fetch_update closure always returns Some");
        debug_assert!(prev > 0, "unpin of unpinned frame {}", self.id);
    }

    fn set_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    fn clear_dirty(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn set_tenant(&self, t: TenantId) {
        self.tenant.store(t.raw(), Ordering::Release);
    }

    fn take_ref_bit(&self) -> bool {
        self.ref_bit.swap(false, Ordering::AcqRel)
    }

    /// CAS-y pin: increments pin count, then verifies the frame still
    /// holds `expected`. If not, unwinds the pin and returns false.
    fn try_pin_with_page(&self, expected: PageId) -> bool {
        self.pin();
        let matches = {
            let g = self.page_id.lock();
            *g == Some(expected)
        };
        if matches {
            true
        } else {
            self.unpin();
            false
        }
    }
}

// ---------- guards -----------------------------------------------------------

/// Read guard over a frame's page buffer. Holds both the RwLock read
/// lock and the pin; releases both on drop.
pub struct FrameReadGuard<'a> {
    frame: &'a Frame,
    // `Option` so we can drop the read lock before unpinning in `Drop`.
    data: Option<RwLockReadGuard<'a, Box<PageBuf>>>,
}

impl<'a> FrameReadGuard<'a> {
    fn new(frame: &'a Frame, data: RwLockReadGuard<'a, Box<PageBuf>>) -> Self {
        Self {
            frame,
            data: Some(data),
        }
    }

    /// Frame this guard belongs to.
    #[must_use]
    pub fn frame(&self) -> &Frame {
        self.frame
    }

    /// Tenant recorded when this page was faulted in.
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.frame.tenant()
    }

    /// Raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &PageBuf {
        self.data.as_ref().expect("guard is live").as_ref()
    }
}

impl Drop for FrameReadGuard<'_> {
    fn drop(&mut self) {
        self.data.take();
        self.frame.unpin();
    }
}

/// Write guard over a frame's page buffer. Holds the RwLock write lock
/// and the pin. Any modification to the bytes sets the dirty bit on
/// drop; callers that *read* via a write guard without mutating can
/// explicitly call [`FrameWriteGuard::forget_dirty`].
pub struct FrameWriteGuard<'a> {
    frame: &'a Frame,
    data: Option<RwLockWriteGuard<'a, Box<PageBuf>>>,
    mutated: bool,
}

impl<'a> FrameWriteGuard<'a> {
    fn new(frame: &'a Frame, data: RwLockWriteGuard<'a, Box<PageBuf>>) -> Self {
        Self {
            frame,
            data: Some(data),
            mutated: false,
        }
    }

    /// Frame this guard belongs to.
    #[must_use]
    pub fn frame(&self) -> &Frame {
        self.frame
    }

    /// Raw bytes (immutable view).
    #[must_use]
    pub fn as_bytes(&self) -> &PageBuf {
        self.data.as_ref().expect("guard is live").as_ref()
    }

    /// Mutable raw bytes. Marks the guard as mutated so the frame's
    /// dirty bit is set on drop.
    #[must_use]
    pub fn as_bytes_mut(&mut self) -> &mut PageBuf {
        self.mutated = true;
        self.data.as_mut().expect("guard is live").as_mut()
    }

    /// Opt out of auto dirty-marking for a read-only write guard.
    pub fn forget_dirty(&mut self) {
        self.mutated = false;
    }
}

impl Drop for FrameWriteGuard<'_> {
    fn drop(&mut self) {
        if self.mutated {
            self.frame.set_dirty();
        }
        self.data.take();
        self.frame.unpin();
    }
}

// ---------- PageTable (M1-11) -----------------------------------------------

/// Concurrent mapping from `PageId` to `FrameId`.
#[derive(Default)]
pub struct PageTable {
    map: DashMap<PageId, FrameId>,
}

impl PageTable {
    /// Empty table.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up the frame holding `page_id`, if any.
    #[must_use]
    pub fn lookup(&self, page_id: PageId) -> Option<FrameId> {
        self.map.get(&page_id).map(|v| *v)
    }

    /// Install a mapping, returning the previous mapping if any.
    pub fn insert(&self, page_id: PageId, frame_id: FrameId) -> Option<FrameId> {
        self.map.insert(page_id, frame_id)
    }

    /// Remove a mapping.
    pub fn remove(&self, page_id: PageId) -> Option<FrameId> {
        self.map.remove(&page_id).map(|(_, v)| v)
    }

    /// Remove a mapping only if it still points at `expected_frame`.
    pub fn remove_if(&self, page_id: PageId, expected_frame: FrameId) -> bool {
        self.map
            .remove_if(&page_id, |_, &v| v == expected_frame)
            .is_some()
    }

    /// Number of live mappings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if no mappings exist.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

// ---------- ClockSweepEvictor (M1-12 + M1-14) -------------------------------

/// Clock-sweep (CLOCK-second-chance) eviction policy over a contiguous
/// range of frames `[start, end)`. Sweeps at most `2 × span` frames per
/// call to find an unpinned frame whose reference bit is clear.
///
/// M1-14 split pool constructs two independent evictors — one over the
/// read range, one over the write range — so a write burst never
/// evicts the read working set.
///
/// PostgreSQL, Oracle, and MySQL InnoDB all use variants of this
/// policy. LRU is intentionally not used — its linked-list maintenance
/// hurts throughput on NUMA boxes.
pub struct ClockSweepEvictor {
    hand: Mutex<usize>,
    start: FrameId,
    end: FrameId,
}

impl ClockSweepEvictor {
    /// New evictor over frames in `[start, end)`.
    #[must_use]
    pub fn new(start: FrameId, end: FrameId) -> Self {
        assert!(start <= end, "invalid frame range");
        Self {
            hand: Mutex::new(start),
            start,
            end,
        }
    }

    /// Number of frames this evictor manages.
    #[must_use]
    pub fn span(&self) -> usize {
        self.end - self.start
    }

    /// Sweep until we find an unpinned, unreferenced frame within the
    /// evictor's range, PIN it, and return its id. Clears reference bits
    /// as it goes. Returns `None` if every frame in-range is pinned, or
    /// the range is empty.
    ///
    /// ADR-226 §4 S6: the selected victim is pinned WHILE the `hand`
    /// mutex is still held. This is the correctness linchpin of fault-in
    /// striping: with the old single global `load_lock` removed, two
    /// concurrent slow-path loads (different pages, different stripes)
    /// could otherwise both `find_victim` and reserve the SAME frame,
    /// then each read a different page into it — a corrupting double
    /// bind. Pinning under `hand` makes reservation atomic: a
    /// concurrently-sweeping thread observes the frame `is_pinned()` and
    /// skips it. The caller owns the returned pin (it is the single pin
    /// that `load_into_fresh_frame` leaves on the fresh frame).
    pub fn find_and_pin_victim(&self, frames: &[Frame]) -> Option<FrameId> {
        let span = self.span();
        if span == 0 {
            return None;
        }
        let mut hand = self.hand.lock();
        if *hand < self.start || *hand >= self.end {
            *hand = self.start;
        }
        // Two full sweeps is enough: first sweep clears ref bits, second
        // sweep evicts the frames whose ref bits are now clear.
        for _ in 0..(2 * span) {
            let idx = *hand;
            *hand = if idx + 1 >= self.end {
                self.start
            } else {
                idx + 1
            };
            let frame = &frames[idx];
            if frame.is_pinned() {
                continue;
            }
            if frame.take_ref_bit() {
                // Gave this frame a second chance; sweep will revisit.
                continue;
            }
            // Reserve the victim ATOMICALLY under `hand`: pin before
            // releasing the mutex so no concurrent sweep can pick it.
            frame.pin();
            return Some(idx);
        }
        None
    }
}

// ---------- BufferPool (M1-13 + M1-14) --------------------------------------

/// User-space buffer pool. Owns the frame array; delegates I/O to a
/// `PageIo` implementation. See ADR-001 for why we don't use mmap.
///
/// Split read/write pool (M1-14, design-v2 §3.4): frames are
/// partitioned into a *write pool* of `write_fraction × N` frames and
/// a *read pool* of the remainder. `pin_read` allocates only from the
/// read pool on miss; `pin_write` allocates only from the write pool
/// on miss. A write burst can therefore only churn the write frames —
/// the read working set is isolated.
///
/// Cache hits are pool-agnostic: once a page is mapped, `pin_read`
/// and `pin_write` both reach it via the page table regardless of
/// which pool holds it. There is no migration on first write.
pub struct BufferPool {
    frames: Vec<Frame>,
    page_table: PageTable,
    read_evictor: ClockSweepEvictor,
    write_evictor: ClockSweepEvictor,
    read_range: (FrameId, FrameId),
    write_range: (FrameId, FrameId),
    io: Arc<dyn PageIo>,
    // ADR-226 §4 S6 — striped fault-in locks. A fault-in for page P
    // takes `load_stripes[stripe_for(P)]` (page-id hashed) instead of
    // one global mutex, so fault-ins of DIFFERENT pages that hash to
    // DIFFERENT stripes read from disk in PARALLEL; only same-stripe
    // collisions serialize. The duplicate-load-prevention invariant is
    // preserved because same page_id → same stripe is guaranteed by the
    // deterministic hash: two threads faulting the SAME cold page take
    // the SAME stripe, serialize, and exactly one loads (the other's
    // double-check against `page_table` hits the entry the first
    // installed). Victim selection is made collision-safe independently
    // by pinning the victim under the evictor hand-lock
    // (`find_and_pin_victim`) so two different-stripe slow paths never
    // reserve the same frame. Fast-path (cache-hit) pin/unpin is
    // lock-free against these stripes.
    //
    // LOCK ORDER: a fault-in holds exactly ONE stripe at a time (never
    // two), so there is no stripe-vs-stripe lock-order hazard. While a
    // stripe is held, the slow path further acquires the evictor
    // `hand` mutex (inside `find_and_pin_victim`) and per-frame
    // `page_id` / `data` locks — always in the order
    // stripe → hand → frame-locks, never the reverse. `invalidate_page`
    // takes the same stripe (keyed on the page being invalidated) so it
    // serializes against a concurrent fault-in of that same page.
    //
    // Budget: replaces the ~8.3K faults/s global convoy ceiling with
    // ~LOAD_STRIPES× headroom under a cold miss storm (ADR-226 §4 S6,
    // CONC-A tail / CONC-E). Sized at 32 stripes (ADR range 16–32); one
    // `parking_lot::Mutex<()>` is 1 byte, so 32 stripes cost 32 bytes.
    load_stripes: Box<[Mutex<()>; LOAD_STRIPES]>,
    /// W16γ M6-07 — optional metrics sink (ADR-045). On `Some`, the
    /// pool reports per-pin events (`StoragePageKind::Hit` on a fast-
    /// path success; `Miss` on a slow-path load; `Eviction` when the
    /// slow-path displaces a prior mapping). The `None` path is the
    /// legacy zero-overhead branch — operators not wiring metrics
    /// pay only one nullable-ptr check per pin.
    ///
    /// Budget: cache-hit pin is 23 ns single-threaded
    /// (buffer.rs:10–13). Adding `Option<&Arc<dyn MetricsSink>>::is_none()`
    /// is ~1 ns; the `Some` branch's vtable + atomic increment is
    /// ~5–10 ns (22–43% overhead on cache-hit when metrics wired).
    /// Within v1.0-α tolerance per ADR-045 §"Consequences".
    metrics_sink: Option<Arc<dyn MetricsSink>>,
    #[cfg(test)]
    hooks: TestHooks,
}

#[cfg(test)]
#[derive(Default)]
struct TestHooks {
    before_evict_take: Mutex<Option<TestHook>>,
    before_load_data_write: Mutex<Option<TestHook>>,
    flush_after_snapshot: Mutex<Option<TestHook>>,
}

#[cfg(test)]
type TestHook = Arc<dyn Fn(PageId, FrameId) + Send + Sync>;

/// Default write-pool fraction per design-v2 §3.4.
pub const DEFAULT_WRITE_FRACTION: f64 = 0.20;

/// ADR-226 §4 S6 — number of page-id-hashed fault-in stripes. Within
/// the ADR's 16–32 range; a power of two so `stripe_for` can mask
/// instead of taking a remainder. Bigger = less same-stripe contention
/// under a miss storm at 32 bytes total (`Mutex<()>` is 1 byte each).
const LOAD_STRIPES: usize = 32;

impl BufferPool {
    /// Construct a pool with `frame_count` frames backed by `io`, split
    /// 20% write / 80% read per design-v2 §3.4. For `frame_count == 1`
    /// the pool is unified (the single frame serves both pin paths).
    #[must_use]
    pub fn new(frame_count: usize, io: Arc<dyn PageIo>) -> Self {
        let fraction = if frame_count < 2 {
            0.0
        } else {
            DEFAULT_WRITE_FRACTION
        };
        Self::with_split(frame_count, io, fraction)
    }

    /// Explicit control of the write-pool fraction. `write_fraction`
    /// must be in `[0.0, 1.0]`. `0.0` yields a unified pool (one
    /// clock hand over every frame); non-zero reserves that fraction
    /// of frames for writes. For `frame_count == 1` the split is
    /// always collapsed to unified regardless of fraction.
    #[must_use]
    pub fn with_split(frame_count: usize, io: Arc<dyn PageIo>, write_fraction: f64) -> Self {
        assert!(frame_count > 0, "buffer pool frame_count must be > 0");
        assert!(
            (0.0..=1.0).contains(&write_fraction),
            "write_fraction must be in [0.0, 1.0]"
        );
        let mut frames = Vec::with_capacity(frame_count);
        for i in 0..frame_count {
            frames.push(Frame::new(i));
        }
        let write_frames = if frame_count >= 2 && write_fraction > 0.0 {
            let target = (frame_count as f64 * write_fraction).round() as usize;
            target.clamp(1, frame_count - 1)
        } else {
            0
        };
        let write_range = (0, write_frames);
        let read_range = (write_frames, frame_count);
        Self {
            frames,
            page_table: PageTable::new(),
            read_evictor: ClockSweepEvictor::new(read_range.0, read_range.1),
            write_evictor: ClockSweepEvictor::new(write_range.0, write_range.1),
            read_range,
            write_range,
            io,
            // ADR-226 §4 S6 — 32 page-id-hashed stripes. `[(); N].map`
            // yields the fixed-size array without requiring `Mutex: Copy`
            // (an array literal `[Mutex::new(()); N]` would); boxed to
            // avoid inflating `BufferPool` by an inline 32-lock array.
            load_stripes: Box::new([(); LOAD_STRIPES].map(|()| Mutex::new(()))),
            metrics_sink: None,
            #[cfg(test)]
            hooks: TestHooks::default(),
        }
    }

    #[cfg(test)]
    fn set_before_evict_take_hook(&self, hook: Option<TestHook>) {
        *self.hooks.before_evict_take.lock() = hook;
    }

    #[cfg(test)]
    fn set_before_load_data_write_hook(&self, hook: Option<TestHook>) {
        *self.hooks.before_load_data_write.lock() = hook;
    }

    #[cfg(test)]
    fn set_flush_after_snapshot_hook(&self, hook: Option<TestHook>) {
        *self.hooks.flush_after_snapshot.lock() = hook;
    }

    #[cfg(test)]
    fn before_evict_take_hook(&self, page_id: PageId, frame_id: FrameId) {
        let hook = self.hooks.before_evict_take.lock().clone();
        if let Some(hook) = hook {
            hook(page_id, frame_id);
        }
    }

    #[cfg(test)]
    fn flush_after_snapshot_hook(&self, page_id: PageId, frame_id: FrameId) {
        let hook = self.hooks.flush_after_snapshot.lock().clone();
        if let Some(hook) = hook {
            hook(page_id, frame_id);
        }
    }

    #[cfg(test)]
    fn before_load_data_write_hook(&self, page_id: PageId, frame_id: FrameId) {
        let hook = self.hooks.before_load_data_write.lock().clone();
        if let Some(hook) = hook {
            hook(page_id, frame_id);
        }
    }

    /// W16γ M6-07 — attach an observability sink (ADR-045).
    ///
    /// Builder-style; chains after [`Self::new`] / [`Self::with_split`]
    /// at construction. Legacy callers leave the field `None` (legacy
    /// zero-overhead path).
    #[must_use]
    pub fn with_metrics_sink(mut self, sink: Arc<dyn MetricsSink>) -> Self {
        self.metrics_sink = Some(sink);
        self
    }

    /// Internal hot-path emitter. Inlined so the `None` branch is a
    /// single nullable-ptr check; the `Some` branch is one vtable
    /// call + one atomic increment inside the prometheus crate.
    #[inline(always)]
    fn record_page(&self, kind: StoragePageKind) {
        if let Some(sink) = self.metrics_sink.as_ref() {
            sink.record_storage_page(kind);
        }
    }

    /// Number of frames in the pool.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.frames.len()
    }

    /// Number of frames in the read pool.
    #[must_use]
    pub fn read_pool_size(&self) -> usize {
        self.read_range.1 - self.read_range.0
    }

    /// Number of frames in the write pool.
    #[must_use]
    pub fn write_pool_size(&self) -> usize {
        self.write_range.1 - self.write_range.0
    }

    /// Is `frame_id` part of the write pool?
    #[must_use]
    pub fn is_in_write_pool(&self, frame_id: FrameId) -> bool {
        frame_id >= self.write_range.0 && frame_id < self.write_range.1
    }

    /// Number of frames currently holding a mapped page.
    #[must_use]
    pub fn mapped(&self) -> usize {
        self.page_table.len()
    }

    /// Bind `page_id` into an empty live frame WITHOUT any backing I/O,
    /// exactly as a leaked invisible-build page would appear to the
    /// [`Self::mapped`] census. Gate-only negative control for the INV-M5.10
    /// build-isolation gate: it proves the buffer-pool census assertion goes
    /// RED when the live pool acquires a build page. Only empty, unpinned
    /// frames are used so no live mapping is ever displaced; callers run it
    /// against a quiesced pool.
    #[cfg(feature = "fault-injection")]
    pub fn map_phantom_page_for_build_isolation_gate(&self, page_id: PageId) -> Result<()> {
        if self.page_table.lookup(page_id).is_some() {
            return Ok(());
        }
        for (frame_id, frame) in self.frames.iter().enumerate() {
            let mut binding = frame.page_id.lock();
            if binding.is_some() || frame.is_pinned() {
                continue;
            }
            *binding = Some(page_id);
            drop(binding);
            self.page_table.insert(page_id, frame_id);
            return Ok(());
        }
        Err(ArcGraphError::BufferPoolExhausted)
    }

    fn pin_write_evictor(&self) -> &ClockSweepEvictor {
        if self.write_evictor.span() == 0 {
            // Unified pool — writes fall through to the read range.
            &self.read_evictor
        } else {
            &self.write_evictor
        }
    }

    /// ADR-226 §4 S6 — the fault-in stripe for `page_id`. DETERMINISTIC
    /// on `page_id`: the SAME page always maps to the SAME stripe, which
    /// is what preserves duplicate-load prevention (two threads faulting
    /// the same cold page serialize on this stripe; exactly one loads).
    ///
    /// The id is a `u64` allocated roughly sequentially, so its low bits
    /// alone would cluster consecutive pages into a handful of stripes
    /// under a sequential cold scan — the exact miss-storm S6 targets.
    /// We therefore mix the whole id with the fastrange-style
    /// fibonacci-hash multiply (Knuth's 2^64/φ constant) and take the
    /// high bits, spreading consecutive ids across all stripes. `& mask`
    /// (LOAD_STRIPES is a power of two) is the modulo.
    #[inline]
    fn stripe_for(page_id: PageId) -> usize {
        const FIB: u64 = 0x9E37_79B9_7F4A_7C15;
        let mixed = page_id.raw().wrapping_mul(FIB);
        // Fold the top bits down; `& (N-1)` is `% N` for power-of-two N.
        (mixed >> (64 - LOAD_STRIPES.trailing_zeros())) as usize & (LOAD_STRIPES - 1)
    }

    /// Pin `page_id` for reading. Cache hit is lock-free over the
    /// `DashMap`; a miss takes the page's fault-in stripe
    /// (`load_stripes[stripe_for(page_id)]`, ADR-226 §4 S6) to serialize
    /// concurrent fault-in of THIS page, then allocates a fresh frame
    /// from the read pool's evictor. Fault-ins of other pages on other
    /// stripes proceed in parallel.
    pub fn pin_read(&self, page_id: PageId) -> Result<FrameReadGuard<'_>> {
        if let Some(guard) = self.try_fast_pin_read(page_id) {
            // ADR-045: cache-hit metric. The hot path emits Hit on
            // the lock-free fast path; the re-check inside the
            // stripe-locked branch also emits Hit because that path
            // succeeded via the same fast-pin mechanism (a concurrent
            // thread populated the page table while we were waiting
            // on the fault-in stripe).
            //
            // OPERATIONAL NOTE — slow-path "Hit" semantics:
            //
            // Under concurrent fault-in of the same cold page from N
            // threads, the observed split is 1 Miss + (N-1) Hits, NOT
            // N Misses. The N threads all hash to the SAME stripe
            // (same page_id → same stripe, ADR-226 §4 S6). The (N-1)
            // waiters take the slow path, acquire that stripe after the
            // loader has populated the table, and their re-check
            // succeeds — they truly served from the cache (no disk
            // I/O). The Hit-rate signal `Hit / (Hit + Miss)` is
            // therefore biased upward under concurrent cold-start.
            //
            // Why this is defensible: the cache table DID serve the
            // re-check lookup; the request was not disk-bound. A
            // future v1.1 may introduce a `kind="slow_hit"` label to
            // let operators differentiate true fast-path hits from
            // post-stripe-lock cache hits without taking disk I/O. v1.0
            // ships the operationally-correct semantics with this
            // documented bias.
            self.record_page(StoragePageKind::Hit);
            return Ok(guard);
        }
        // ADR-226 §4 S6: take only THIS page's stripe. Same page_id →
        // same stripe, so concurrent fault-ins of this page serialize
        // and exactly one loads; fault-ins of pages on other stripes
        // run in parallel.
        let _load = self.load_stripes[Self::stripe_for(page_id)].lock();
        // Re-check under the stripe: another thread on this same stripe
        // (necessarily the same page) may have faulted us in. See the
        // OPERATIONAL NOTE above on slow-path Hit semantics. This
        // double-check is the duplicate-load-prevention invariant.
        if let Some(guard) = self.try_fast_pin_read(page_id) {
            self.record_page(StoragePageKind::Hit);
            return Ok(guard);
        }
        let frame_id =
            self.load_into_fresh_frame(page_id, TenantId::DEFAULT, &self.read_evictor)?;
        let frame = &self.frames[frame_id];
        // `load_into_fresh_frame` leaves the frame pinned once; downgrade
        // the write lock to a read lock while holding that pin.
        let write = frame.data.write();
        let read = RwLockWriteGuard::downgrade(write);
        Ok(FrameReadGuard::new(frame, read))
    }

    /// Pin `page_id` for writing. On miss allocates from the write
    /// pool (or the read pool if the pool is unified). Records `tenant`
    /// on the frame when a fresh frame is allocated; cache hits retain
    /// the tenant recorded at fault time.
    pub fn pin_write(&self, page_id: PageId, tenant: TenantId) -> Result<FrameWriteGuard<'_>> {
        if let Some(guard) = self.try_fast_pin_write(page_id) {
            self.record_page(StoragePageKind::Hit);
            return Ok(guard);
        }
        // ADR-226 §4 S6: same page-id-hashed stripe as `pin_read` uses,
        // so a concurrent `pin_read` and `pin_write` of the SAME page
        // still serialize (they share the stripe) and only one loads.
        let _load = self.load_stripes[Self::stripe_for(page_id)].lock();
        // Re-check: see OPERATIONAL NOTE in `pin_read` on slow-path
        // Hit semantics — same applies to `pin_write`.
        if let Some(guard) = self.try_fast_pin_write(page_id) {
            self.record_page(StoragePageKind::Hit);
            return Ok(guard);
        }
        let evictor = self.pin_write_evictor();
        let frame_id = self.load_into_fresh_frame(page_id, tenant, evictor)?;
        let frame = &self.frames[frame_id];
        let write = frame.data.write();
        Ok(FrameWriteGuard::new(frame, write))
    }

    fn try_fast_pin_read(&self, page_id: PageId) -> Option<FrameReadGuard<'_>> {
        let frame_id = self.page_table.lookup(page_id)?;
        let frame = &self.frames[frame_id];
        if !frame.try_pin_with_page(page_id) {
            return None;
        }
        let data = frame.data.read();
        Some(FrameReadGuard::new(frame, data))
    }

    fn try_fast_pin_write(&self, page_id: PageId) -> Option<FrameWriteGuard<'_>> {
        let frame_id = self.page_table.lookup(page_id)?;
        let frame = &self.frames[frame_id];
        if !frame.try_pin_with_page(page_id) {
            return None;
        }
        let data = frame.data.write();
        Some(FrameWriteGuard::new(frame, data))
    }

    /// Mark a page dirty without holding a write guard. Useful when a
    /// caller has just released a write guard and realized the mutation
    /// flag got lost.
    pub fn mark_dirty(&self, page_id: PageId) -> Result<()> {
        match self.page_table.lookup(page_id) {
            Some(frame_id) => {
                self.frames[frame_id].set_dirty();
                Ok(())
            }
            None => Err(ArcGraphError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("page {} not currently pinned", page_id.raw()),
            ))),
        }
    }

    /// Invalidate a clean cached mapping after the caller has written
    /// canonical bytes through another I/O path.
    ///
    /// This is intentionally narrower than eviction: it removes only the
    /// page-table entry for `page_id` if it still points at the observed
    /// frame, then clears that frame's page binding. Holding
    /// `page_id`'s fault-in stripe (ADR-226 §4 S6) serializes against a
    /// concurrent fault-in of THIS SAME page (which takes the same
    /// stripe); `remove_if` avoids deleting a newer mapping if a future
    /// path changes the binding between lookup and removal.
    ///
    /// LOCK ORDER: takes exactly one stripe (this page's), consistent
    /// with the fault-in path — no two stripes are ever held at once.
    pub fn invalidate_page(&self, page_id: PageId) -> bool {
        let _load = self.load_stripes[Self::stripe_for(page_id)].lock();
        let Some(frame_id) = self.page_table.lookup(page_id) else {
            return false;
        };
        let frame = &self.frames[frame_id];
        if !self.page_table.remove_if(page_id, frame_id) {
            return false;
        }
        let mut current = frame.page_id.lock();
        if *current == Some(page_id) {
            *current = None;
            frame.clear_dirty();
            true
        } else {
            false
        }
    }

    /// Write every dirty frame back to storage and clear its dirty bit.
    /// Does not evict; pages stay mapped.
    /// Re-verifies the frame-to-page binding under each frame's data guard;
    /// frames rebound mid-flush are skipped.
    /// This does not provide point-in-time snapshot semantics and is not a
    /// consistent checkpoint primitive by itself; checkpointing must add its
    /// own quiescence or epoch mechanism.
    pub fn flush_all(&self) -> Result<()> {
        for frame in &self.frames {
            if !frame.is_dirty() {
                continue;
            }
            let page_id = match frame.current_page() {
                Some(p) => p,
                None => continue,
            };
            #[cfg(test)]
            self.flush_after_snapshot_hook(page_id, frame.id());
            // Acquire the data read lock so we serialize against writers
            // but allow concurrent readers. Re-check the binding under
            // this guard: eviction may have rebound the frame after the
            // dirty/page snapshot above.
            let guard = frame.data.read();
            if frame.current_page() != Some(page_id) {
                continue;
            }
            self.io.write_page(page_id, guard.as_ref())?;
            frame.clear_dirty();
        }
        self.io.flush()
    }

    // --- slow path ---

    /// Caller must hold `page_id`'s fault-in stripe
    /// (`load_stripes[stripe_for(page_id)]`, ADR-226 §4 S6). Leaves the
    /// victim frame pinned once and installs the mapping in the page
    /// table. Does *not* install the RwLock guard — caller does that
    /// immediately afterwards. `evictor` must point to the pool that
    /// should absorb this fault (read vs write).
    ///
    /// The victim is selected AND pinned atomically by
    /// `find_and_pin_victim` (the pin closes the victim-collision race
    /// that the old global `load_lock` used to close); the single pin it
    /// returns is exactly the "pinned once" this function must leave. On
    /// any error path the reservation pin is released so the frame is not
    /// leaked as permanently pinned.
    fn load_into_fresh_frame(
        &self,
        page_id: PageId,
        tenant: TenantId,
        evictor: &ClockSweepEvictor,
    ) -> Result<FrameId> {
        // Victim is returned ALREADY pinned (reserved under the evictor
        // hand-lock) so no other stripe's slow path can also pick it.
        let victim_id = evictor
            .find_and_pin_victim(&self.frames)
            .ok_or(ArcGraphError::BufferPoolExhausted)?;
        let victim = &self.frames[victim_id];

        // Evict old page.
        #[cfg(test)]
        self.before_evict_take_hook(page_id, victim_id);
        let old = {
            let mut g = victim.page_id.lock();
            g.take()
        };
        // ADR-045: an eviction is the case where the victim frame
        // held a different page before this load. Cold-slot loads
        // (victim was empty) report Miss WITHOUT Eviction. Hit-rate
        // computed as `rate(Hit) / rate(Hit+Miss)` is the design-v2
        // §10.2 line 703 buffer_pool_hit_rate signal.
        let evicted_old = old.is_some();

        // Load new page.
        #[cfg(test)]
        self.before_load_data_write_hook(page_id, victim_id);
        {
            let mut data = victim.data.write();
            if let Some(old_id) = old {
                if victim.is_dirty() {
                    // Latency budget: eviction slow path may hold this
                    // exclusive guard across one write_page so dirty
                    // sampling, write-back, and rebind are ordered.
                    if let Err(err) = self.io.write_page(old_id, data.as_ref()) {
                        *victim.page_id.lock() = Some(old_id);
                        // Release the reservation pin: the frame is left
                        // holding its original page, not leaked pinned.
                        drop(data);
                        victim.unpin();
                        return Err(err);
                    }
                    victim.clear_dirty();
                }
                self.page_table.remove_if(old_id, victim_id);
            }
            if let Err(err) = self.io.read_page(page_id, data.as_mut()) {
                // The old mapping (if any) is already removed and the
                // frame's page binding is None; release the reservation
                // pin so the empty frame returns to the eviction pool.
                drop(data);
                victim.unpin();
                return Err(err);
            }
            *victim.page_id.lock() = Some(page_id);
        }
        victim.set_tenant(tenant);
        self.page_table.insert(page_id, victim_id);
        // No `victim.pin()` here: the single pin was taken atomically at
        // selection by `find_and_pin_victim` (ADR-226 §4 S6).
        // ADR-045: emit Miss (always on slow path) and Eviction
        // (only when a prior mapping was displaced). The order
        // matches the read-from-disk sequence: cache lookup
        // failed (Miss); a prior frame had to be reclaimed
        // (Eviction).
        self.record_page(StoragePageKind::Miss);
        if evicted_old {
            self.record_page(StoragePageKind::Eviction);
        }
        Ok(victim_id)
    }
}

// ---------- tests ------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
    use std::sync::{Condvar, Mutex as StdMutex};
    use std::thread;

    use arcgraph_core::{PageId, TenantId};
    use parking_lot::RwLock as ParkingRwLock;
    use proptest::prelude::*;

    use super::*;
    use crate::io::InMemoryPageIo;

    const NO_FAIL_PAGE: u64 = u64::MAX;

    struct FailingPageIo {
        pages: ParkingRwLock<HashMap<PageId, Box<PageBuf>>>,
        fail_next_write_page: AtomicU64,
    }

    impl FailingPageIo {
        fn new() -> Self {
            Self {
                pages: ParkingRwLock::new(HashMap::new()),
                fail_next_write_page: AtomicU64::new(NO_FAIL_PAGE),
            }
        }

        fn fail_next_write(&self, page_id: PageId) {
            self.fail_next_write_page
                .store(page_id.raw(), AtomicOrdering::Release);
        }

        fn disk_byte(&self, page_id: PageId) -> u8 {
            self.pages
                .read()
                .get(&page_id)
                .expect("page must exist")
                .as_ref()[0]
        }
    }

    impl PageIo for FailingPageIo {
        fn read_page(&self, page_id: PageId, buf: &mut PageBuf) -> Result<()> {
            let pages = self.pages.read();
            match pages.get(&page_id) {
                Some(page) => {
                    buf.copy_from_slice(page.as_ref());
                    Ok(())
                }
                None => Err(ArcGraphError::Io(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("page {} not found", page_id.raw()),
                ))),
            }
        }

        fn write_page(&self, page_id: PageId, buf: &PageBuf) -> Result<()> {
            if self
                .fail_next_write_page
                .compare_exchange(
                    page_id.raw(),
                    NO_FAIL_PAGE,
                    AtomicOrdering::AcqRel,
                    AtomicOrdering::Acquire,
                )
                .is_ok()
            {
                return Err(ArcGraphError::Io(std::io::Error::other(
                    "injected write failure",
                )));
            }
            self.pages.write().insert(page_id, Box::new(*buf));
            Ok(())
        }

        fn flush(&self) -> Result<()> {
            Ok(())
        }
    }

    fn make_pool(frames: usize) -> (Arc<InMemoryPageIo>, BufferPool) {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(frames, io.clone());
        (io, pool)
    }

    fn seed(io: &InMemoryPageIo, page: u64, byte: u8) {
        let mut buf = [0u8; PAGE_SIZE];
        buf[0] = byte;
        io.write_page(PageId::new(page), &buf).unwrap();
    }

    fn seed_page(io: &dyn PageIo, page: u64, byte: u8) {
        let mut buf = [0u8; PAGE_SIZE];
        buf[0] = byte;
        io.write_page(PageId::new(page), &buf).unwrap();
    }

    // ---- Frame ----

    #[test]
    fn frame_pin_unpin_accounting() {
        let f = Frame::new(0);
        assert_eq!(f.pin_count(), 0);
        f.pin();
        f.pin();
        assert_eq!(f.pin_count(), 2);
        f.unpin();
        assert_eq!(f.pin_count(), 1);
        assert!(f.is_pinned());
        f.unpin();
        assert!(!f.is_pinned());
    }

    #[test]
    fn frame_unpin_saturates_instead_of_wrapping_on_double_unpin() {
        let f = Frame::new(0);
        f.pin();
        f.unpin();
        assert_eq!(f.pin_count(), 0);

        let _ = catch_unwind(AssertUnwindSafe(|| f.unpin()));
        assert_eq!(
            f.pin_count(),
            0,
            "release-mode double-unpin must not wrap pin_count"
        );
    }

    #[test]
    fn frame_dirty_flag_sticks() {
        let f = Frame::new(0);
        assert!(!f.is_dirty());
        f.set_dirty();
        assert!(f.is_dirty());
        f.clear_dirty();
        assert!(!f.is_dirty());
    }

    // ---- PageTable ----

    #[test]
    fn page_table_insert_lookup_remove() {
        let t = PageTable::new();
        assert!(t.lookup(PageId::new(1)).is_none());
        assert!(t.insert(PageId::new(1), 7).is_none());
        assert_eq!(t.lookup(PageId::new(1)), Some(7));
        assert_eq!(t.len(), 1);
        assert_eq!(t.remove(PageId::new(1)), Some(7));
        assert!(t.is_empty());
    }

    #[test]
    fn page_table_remove_if_only_matches() {
        let t = PageTable::new();
        t.insert(PageId::new(1), 5);
        assert!(!t.remove_if(PageId::new(1), 9));
        assert_eq!(t.len(), 1);
        assert!(t.remove_if(PageId::new(1), 5));
        assert!(t.is_empty());
    }

    // ---- ClockSweepEvictor ----

    #[test]
    fn evictor_returns_none_when_all_pinned() {
        let frames: Vec<Frame> = (0..4).map(Frame::new).collect();
        for f in &frames {
            f.pin();
        }
        let e = ClockSweepEvictor::new(0, frames.len());
        assert!(e.find_and_pin_victim(&frames).is_none());
    }

    #[test]
    fn evictor_gives_second_chance_on_ref_bit() {
        let frames: Vec<Frame> = (0..2).map(Frame::new).collect();
        // Frame 0 was recently referenced; frame 1 was not.
        frames[0].ref_bit.store(true, Ordering::Release);
        let e = ClockSweepEvictor::new(0, frames.len());
        assert_eq!(e.find_and_pin_victim(&frames), Some(1));
        // ADR-226 §4 S6: the selected victim is returned pinned.
        assert!(frames[1].is_pinned(), "selected victim must be reserved");
    }

    #[test]
    fn evictor_respects_range_boundary() {
        let frames: Vec<Frame> = (0..6).map(Frame::new).collect();
        // Pin frames outside the evictor's [2, 5) range to prove the
        // evictor ignores them even if they'd otherwise be victims.
        frames[0].pin();
        frames[1].pin();
        frames[5].pin();
        let e = ClockSweepEvictor::new(2, 5);
        let v = e.find_and_pin_victim(&frames).unwrap();
        assert!((2..5).contains(&v), "victim {v} must be in [2, 5)");
    }

    #[test]
    fn evictor_empty_range_returns_none() {
        let frames: Vec<Frame> = (0..2).map(Frame::new).collect();
        let e = ClockSweepEvictor::new(1, 1);
        assert!(e.find_and_pin_victim(&frames).is_none());
    }

    #[test]
    fn evictor_pins_selected_victim_so_concurrent_sweep_skips_it() {
        // ADR-226 §4 S6 victim-collision guard: once a frame is
        // selected, it is pinned, so a second sweep of the SAME evictor
        // must NOT return it again — it picks a different frame.
        let frames: Vec<Frame> = (0..2).map(Frame::new).collect();
        let e = ClockSweepEvictor::new(0, frames.len());
        let first = e.find_and_pin_victim(&frames).expect("first victim");
        let second = e.find_and_pin_victim(&frames).expect("second victim");
        assert_ne!(
            first, second,
            "a pinned victim must not be re-selected — this is the \
             different-page-different-stripe no-collision invariant"
        );
    }

    // ---- BufferPool happy paths ----

    #[test]
    fn pin_read_hits_after_miss() {
        let (io, pool) = make_pool(2);
        seed(&io, 1, 0xAB);

        {
            let g = pool.pin_read(PageId::new(1)).unwrap();
            assert_eq!(g.as_bytes()[0], 0xAB);
        }
        assert_eq!(io.reads(), 1);

        // Second pin hits the cache.
        {
            let g = pool.pin_read(PageId::new(1)).unwrap();
            assert_eq!(g.as_bytes()[0], 0xAB);
        }
        assert_eq!(io.reads(), 1, "second read must be a cache hit");
    }

    #[test]
    fn pin_write_marks_dirty_and_flush_writes_back() {
        let (io, pool) = make_pool(2);
        seed(&io, 1, 0x01);

        {
            let mut g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0x99;
        }
        assert_eq!(io.writes(), 1, "seed counts as 1 write");

        pool.flush_all().unwrap();
        assert_eq!(io.writes(), 2, "flush must write the dirty page once");

        // Read through the pool still reflects the mutation.
        let g = pool.pin_read(PageId::new(1)).unwrap();
        assert_eq!(g.as_bytes()[0], 0x99);
    }

    #[test]
    fn eviction_writes_dirty_page_before_reuse() {
        let (io, pool) = make_pool(1);
        seed(&io, 1, 0x11);
        seed(&io, 2, 0x22);

        {
            let mut g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0xAA;
        }
        // Pin a second page → forces eviction of page 1, which must
        // write back because of the dirty mutation above.
        {
            let g = pool.pin_read(PageId::new(2)).unwrap();
            assert_eq!(g.as_bytes()[0], 0x22);
        }
        assert_eq!(io.writes(), 3, "seed×2 + one write-back on eviction");
    }

    #[test]
    fn exhaustion_returns_error() {
        let (io, pool) = make_pool(1);
        seed(&io, 1, 0x11);
        seed(&io, 2, 0x22);
        let _held = pool.pin_read(PageId::new(1)).unwrap();
        match pool.pin_read(PageId::new(2)) {
            Err(ArcGraphError::BufferPoolExhausted) => {}
            Err(other) => panic!("wrong error variant: {other:?}"),
            Ok(_) => panic!("expected BufferPoolExhausted"),
        }
    }

    #[test]
    fn dropping_guard_unpins() {
        let (io, pool) = make_pool(1);
        seed(&io, 1, 0x11);
        seed(&io, 2, 0x22);
        {
            let _g = pool.pin_read(PageId::new(1)).unwrap();
        }
        // Frame is unpinned now; pinning a different page should succeed.
        let _g = pool.pin_read(PageId::new(2)).unwrap();
    }

    #[test]
    fn write_guard_without_mutation_is_not_dirty() {
        let (io, pool) = make_pool(1);
        seed(&io, 1, 0x11);
        {
            // Acquire write but don't mutate.
            let _g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
        }
        pool.flush_all().unwrap();
        // Seed was 1 write; no further writes from an unmarked write guard.
        assert_eq!(io.writes(), 1);
    }

    #[test]
    fn forget_dirty_suppresses_dirty_mark() {
        let (io, pool) = make_pool(1);
        seed(&io, 1, 0x11);
        {
            let mut g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
            // Pretend we mutated, then change our mind.
            let _ = g.as_bytes_mut();
            g.forget_dirty();
        }
        pool.flush_all().unwrap();
        assert_eq!(io.writes(), 1);
    }

    // ---- concurrency smoke tests ----

    #[test]
    fn concurrent_readers_all_see_the_same_page() {
        let (io, pool) = make_pool(8);
        seed(&io, 1, 0x55);
        let pool = Arc::new(pool);
        thread::scope(|s| {
            for _ in 0..8 {
                let p = pool.clone();
                s.spawn(move || {
                    for _ in 0..100 {
                        let g = p.pin_read(PageId::new(1)).unwrap();
                        assert_eq!(g.as_bytes()[0], 0x55);
                    }
                });
            }
        });
    }

    #[test]
    fn concurrent_miss_storm_loads_each_page_at_most_once_per_fault() {
        let (io, pool) = make_pool(64);
        for i in 1..=32u64 {
            seed(&io, i, i as u8);
        }
        let reads_before = io.reads();
        let pool = Arc::new(pool);
        thread::scope(|s| {
            for t in 0..16 {
                let p = pool.clone();
                s.spawn(move || {
                    for i in 1..=32u64 {
                        let page_id = PageId::new(((i + t as u64) % 32) + 1);
                        let g = p.pin_read(page_id).unwrap();
                        assert_eq!(g.as_bytes()[0], page_id.raw() as u8);
                    }
                });
            }
        });
        let loads = io.reads() - reads_before;
        // At worst, every page faults in once — no duplicate loads under
        // the per-page fault-in stripes (ADR-226 §4 S6): every access of
        // a given page hashes to the same stripe, so concurrent
        // fault-ins of it serialize and exactly one reads from disk. We
        // allow ≤32 here (one per distinct page).
        assert!(loads <= 32, "too many disk reads: {}", loads);
    }

    /// ADR-226 §4 S6 HEADLINE TEST — the duplicate-load-prevention
    /// invariant under the new striped locks. N ≥ 8 threads race to
    /// fault-in the SAME cold page; the deterministic page-id hash sends
    /// them all to the SAME stripe, so exactly ONE performs the disk read
    /// and the rest observe the frame it installed. We assert a single
    /// disk read (`io.reads()` delta == 1) and that every thread saw the
    /// same bytes. RED on revert: if the stripe re-check (double-checked
    /// locking) is removed, the (N-1) waiters each fault the page again
    /// and this delta blows past 1.
    #[test]
    fn s6_same_cold_page_loaded_from_disk_exactly_once_under_contention() {
        const THREADS: usize = 16;
        let (io, pool) = make_pool(64);
        seed(&io, 7, 0xAB);
        let reads_before = io.reads();
        let pool = Arc::new(pool);

        // A start barrier maximizes the odds all threads hit the miss
        // window simultaneously (worst case for duplicate loads).
        let barrier = Arc::new(std::sync::Barrier::new(THREADS));
        thread::scope(|s| {
            for _ in 0..THREADS {
                let p = pool.clone();
                let b = barrier.clone();
                s.spawn(move || {
                    b.wait();
                    let g = p.pin_read(PageId::new(7)).expect("pin cold page");
                    assert_eq!(
                        g.as_bytes()[0],
                        0xAB,
                        "every thread must observe the one loaded frame"
                    );
                });
            }
        });

        let loads = io.reads() - reads_before;
        assert_eq!(
            loads, 1,
            "the same cold page must be read from disk EXACTLY once \
             under {THREADS}-way contention (got {loads}) — the per-stripe \
             double-checked-locking invariant is broken"
        );
    }

    /// ADR-226 §4 S6 — determinism of the page-id hash. The SAME page_id
    /// ALWAYS maps to the SAME stripe (this is what makes same-page
    /// fault-ins serialize), and the hash lands in range for every id.
    #[test]
    fn s6_stripe_for_is_deterministic_and_in_range() {
        for raw in [0u64, 1, 2, 7, 63, 64, 1_000, u64::MAX] {
            let a = BufferPool::stripe_for(PageId::new(raw));
            let b = BufferPool::stripe_for(PageId::new(raw));
            assert_eq!(a, b, "stripe_for({raw}) must be deterministic");
            assert!(a < LOAD_STRIPES, "stripe {a} out of range for id {raw}");
        }
    }

    /// ADR-226 §4 S6 — the hash spreads consecutive page ids across
    /// stripes (the sequential-cold-scan miss-storm case). If low bits
    /// alone were used, ids 0..N would cluster; the fibonacci mix must
    /// hit a healthy fraction of the 32 stripes over a 256-id sweep.
    #[test]
    fn s6_consecutive_ids_spread_across_stripes() {
        let mut seen = std::collections::HashSet::new();
        for raw in 0..256u64 {
            seen.insert(BufferPool::stripe_for(PageId::new(raw)));
        }
        assert!(
            seen.len() >= LOAD_STRIPES / 2,
            "hash clusters consecutive ids: only {}/{} stripes hit",
            seen.len(),
            LOAD_STRIPES
        );
    }

    /// ADR-226 §4 S6 — parallel fault-in of DIFFERENT pages: N threads
    /// each fault a distinct cold page into a pool with room for all of
    /// them. Every page is read from disk exactly once and each thread
    /// observes its own correct frame. This proves striping did not
    /// break the different-page path (victims are reserved without
    /// collision by `find_and_pin_victim`).
    #[test]
    fn s6_parallel_fault_in_of_distinct_pages_each_loaded_once() {
        const PAGES: u64 = 24;
        let (io, pool) = make_pool(64);
        for p in 1..=PAGES {
            seed(&io, p, p as u8);
        }
        let reads_before = io.reads();
        let pool = Arc::new(pool);
        let barrier = Arc::new(std::sync::Barrier::new(PAGES as usize));
        thread::scope(|s| {
            for p in 1..=PAGES {
                let pool = pool.clone();
                let b = barrier.clone();
                s.spawn(move || {
                    b.wait();
                    let g = pool.pin_read(PageId::new(p)).expect("pin distinct page");
                    assert_eq!(
                        g.as_bytes()[0],
                        p as u8,
                        "thread must observe its own page's bytes"
                    );
                });
            }
        });
        let loads = io.reads() - reads_before;
        assert_eq!(
            loads, PAGES,
            "each distinct page must be read from disk exactly once \
             (got {loads} for {PAGES} pages) — a victim collision would \
             cause a reload or a corrupt frame"
        );
    }

    #[test]
    fn concurrent_pin_write_eviction_writes_in_flight_first_mutation() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = Arc::new(BufferPool::with_split(1, io.clone(), 0.0));
        seed(&io, 1, 0x11);
        seed(&io, 2, 0x22);

        {
            let g = pool.pin_read(PageId::new(1)).unwrap();
            assert_eq!(g.as_bytes()[0], 0x11);
        }

        let start = Arc::new((StdMutex::new(false), Condvar::new()));
        let ready = Arc::new((StdMutex::new(false), Condvar::new()));
        let evict_waiting = Arc::new((StdMutex::new(false), Condvar::new()));
        let release = Arc::new((StdMutex::new(false), Condvar::new()));

        let writer = {
            let pool = pool.clone();
            let start = start.clone();
            let ready = ready.clone();
            let release = release.clone();
            thread::spawn(move || {
                let (lock, cv) = &*start;
                let mut started = lock.lock().expect("start mutex");
                while !*started {
                    started = cv.wait(started).expect("start condvar");
                }
                drop(started);

                let mut guard = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
                guard.as_bytes_mut()[0] = 0x5A;

                let (lock, cv) = &*ready;
                *lock.lock().expect("ready mutex") = true;
                cv.notify_all();

                let (lock, cv) = &*release;
                let mut released = lock.lock().expect("release mutex");
                while !*released {
                    released = cv.wait(released).expect("release condvar");
                }
            })
        };

        {
            let start = start.clone();
            let ready = ready.clone();
            pool.set_before_evict_take_hook(Some(Arc::new(move |page_id, _frame_id| {
                if page_id != PageId::new(2) {
                    return;
                }
                let (lock, cv) = &*start;
                *lock.lock().expect("start mutex") = true;
                cv.notify_all();

                let (lock, cv) = &*ready;
                let mut is_ready = lock.lock().expect("ready mutex");
                while !*is_ready {
                    is_ready = cv.wait(is_ready).expect("ready condvar");
                }
            })));
        }
        {
            let evict_waiting = evict_waiting.clone();
            pool.set_before_load_data_write_hook(Some(Arc::new(move |page_id, _frame_id| {
                if page_id != PageId::new(2) {
                    return;
                }
                let (lock, cv) = &*evict_waiting;
                *lock.lock().expect("evict-waiting mutex") = true;
                cv.notify_all();
            })));
        }

        let evictor = {
            let pool = pool.clone();
            thread::spawn(move || {
                let g = pool.pin_read(PageId::new(2)).unwrap();
                assert_eq!(g.as_bytes()[0], 0x22);
            })
        };

        {
            let (lock, cv) = &*evict_waiting;
            let mut is_waiting = lock.lock().expect("evict-waiting mutex");
            while !*is_waiting {
                is_waiting = cv.wait(is_waiting).expect("evict-waiting condvar");
            }
        }
        {
            let (lock, cv) = &*release;
            *lock.lock().expect("release mutex") = true;
            cv.notify_all();
        }

        writer.join().expect("writer thread");
        evictor.join().expect("evictor thread");
        pool.set_before_evict_take_hook(None);
        pool.set_before_load_data_write_hook(None);

        let mut disk = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(1), &mut disk).unwrap();
        assert_eq!(disk[0], 0x5A, "eviction must persist the in-flight write");
    }

    #[test]
    fn flush_all_rechecks_binding_before_writing_snapshot_page() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = Arc::new(BufferPool::with_split(1, io.clone(), 0.0));
        seed(&io, 1, 0xA1);
        seed(&io, 2, 0xB2);

        {
            let mut g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0xA9;
        }

        let fired = Arc::new(AtomicBool::new(false));
        {
            let hook_pool = pool.clone();
            let fired = fired.clone();
            pool.set_flush_after_snapshot_hook(Some(Arc::new(move |page_id, _frame_id| {
                if page_id != PageId::new(1) || fired.swap(true, Ordering::AcqRel) {
                    return;
                }
                let g = hook_pool.pin_read(PageId::new(2)).unwrap();
                assert_eq!(g.as_bytes()[0], 0xB2);
            })));
        }

        pool.flush_all().unwrap();
        pool.set_flush_after_snapshot_hook(None);

        let mut disk_a = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(1), &mut disk_a).unwrap();
        assert_eq!(
            disk_a[0], 0xA9,
            "flush_all must not write page 2 bytes over page 1"
        );
    }

    #[test]
    fn eviction_write_back_error_restores_old_dirty_binding() {
        let io = Arc::new(FailingPageIo::new());
        seed_page(io.as_ref(), 1, 0x11);
        seed_page(io.as_ref(), 2, 0x22);
        let pool = BufferPool::with_split(1, io.clone(), 0.0);

        {
            let mut g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0xCC;
        }

        io.fail_next_write(PageId::new(1));
        let err = match pool.pin_read(PageId::new(2)) {
            Ok(_) => panic!("expected injected write failure"),
            Err(err) => err,
        };
        assert!(matches!(err, ArcGraphError::Io(_)));

        let frame_id = pool
            .page_table
            .lookup(PageId::new(1))
            .expect("old table entry must remain");
        let frame = &pool.frames[frame_id];
        assert_eq!(frame.current_page(), Some(PageId::new(1)));
        assert!(frame.is_dirty(), "old dirty bytes must remain flushable");

        {
            let g = pool.pin_read(PageId::new(1)).unwrap();
            assert_eq!(
                g.as_bytes()[0],
                0xCC,
                "fast pin must still serve the dirty in-memory bytes"
            );
        }
        assert_eq!(io.disk_byte(PageId::new(1)), 0x11);

        pool.flush_all().unwrap();
        assert_eq!(io.disk_byte(PageId::new(1)), 0xCC);
    }

    // ---- M1-14 split read/write pool ----

    #[test]
    fn split_pool_default_sizes_20_80() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(10, io);
        assert_eq!(pool.write_pool_size(), 2);
        assert_eq!(pool.read_pool_size(), 8);
    }

    #[test]
    fn split_pool_collapses_on_size_one() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::new(1, io);
        assert_eq!(pool.write_pool_size(), 0);
        assert_eq!(pool.read_pool_size(), 1);
    }

    #[test]
    fn split_pool_explicit_zero_fraction_is_unified() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::with_split(10, io, 0.0);
        assert_eq!(pool.write_pool_size(), 0);
        assert_eq!(pool.read_pool_size(), 10);
    }

    #[test]
    fn split_pool_rounds_and_clamps() {
        let io = Arc::new(InMemoryPageIo::new());
        // 5 * 0.20 = 1.0 → 1
        let p1 = BufferPool::with_split(5, io.clone(), 0.20);
        assert_eq!(p1.write_pool_size(), 1);
        assert_eq!(p1.read_pool_size(), 4);
        // 5 * 0.95 would be 4.75 → round → 5, clamped to N-1 = 4
        let p2 = BufferPool::with_split(5, io.clone(), 0.95);
        assert_eq!(p2.write_pool_size(), 4);
        assert_eq!(p2.read_pool_size(), 1);
        // 5 * 0.02 = 0.1 → round → 0, clamp up to 1 (never zero when fraction > 0)
        let p3 = BufferPool::with_split(5, io, 0.02);
        assert_eq!(p3.write_pool_size(), 1);
    }

    #[test]
    fn pin_write_miss_allocates_from_write_pool() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::with_split(10, io.clone(), 0.20);
        seed(&io, 1, 0xAA);
        let guard = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
        let fid = guard.frame().id();
        assert!(
            pool.is_in_write_pool(fid),
            "write fault landed in frame {fid}, not the write pool"
        );
    }

    #[test]
    fn pin_read_miss_allocates_from_read_pool() {
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::with_split(10, io.clone(), 0.20);
        seed(&io, 1, 0xAA);
        let guard = pool.pin_read(PageId::new(1)).unwrap();
        let fid = guard.frame().id();
        assert!(
            !pool.is_in_write_pool(fid),
            "read fault landed in frame {fid}, but that's in the write pool"
        );
    }

    #[test]
    fn write_burst_does_not_evict_read_working_set() {
        // Roadmap M1-14 acceptance: write-pool isolation under burst load.
        //
        // Pool of 10 frames: 2 write + 8 read. Fault in 8 hot pages
        // (fills the read pool). Then run a 40-page write burst.
        // All 8 hot pages must still be cached; no re-reads from IO.
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::with_split(10, io.clone(), 0.20);

        // Seed 50 pages. Hot = 0..8, cold = 100..140.
        for i in 0..8u64 {
            seed(&io, i, 0xAA);
        }
        for i in 100u64..140 {
            seed(&io, i, 0x55);
        }

        // Prime the read pool with hot pages.
        for i in 0..8u64 {
            let _g = pool.pin_read(PageId::new(i)).unwrap();
        }
        let reads_after_prime = io.reads();
        assert_eq!(reads_after_prime, 8, "expected 8 initial reads for priming");

        // 40-page write burst on cold pages.
        for i in 100u64..140 {
            let mut g = pool.pin_write(PageId::new(i), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0x99;
        }

        // Hot pages are re-pinned for read: all should be cache hits.
        let reads_before_hot = io.reads();
        for i in 0..8u64 {
            let g = pool.pin_read(PageId::new(i)).unwrap();
            assert_eq!(g.as_bytes()[0], 0xAA);
        }
        let hot_reads = io.reads() - reads_before_hot;
        assert_eq!(
            hot_reads, 0,
            "read working set was evicted by the write burst"
        );
    }

    #[test]
    fn unified_pool_write_burst_does_evict_read_working_set() {
        // The counter-evidence: a write burst on a unified pool DOES
        // evict read pages. This is the motivation for M1-14.
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::with_split(10, io.clone(), 0.0);

        for i in 0..8u64 {
            seed(&io, i, 0xAA);
        }
        for i in 100u64..140 {
            seed(&io, i, 0x55);
        }

        for i in 0..8u64 {
            let _g = pool.pin_read(PageId::new(i)).unwrap();
        }
        for i in 100u64..140 {
            let mut g = pool.pin_write(PageId::new(i), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0x99;
        }

        let reads_before_hot = io.reads();
        for i in 0..8u64 {
            let _g = pool.pin_read(PageId::new(i)).unwrap();
        }
        let hot_reads = io.reads() - reads_before_hot;
        assert!(
            hot_reads > 0,
            "unified pool was expected to re-read hot pages after write burst; got {hot_reads}"
        );
    }

    #[test]
    fn page_faulted_for_read_stays_in_read_pool_on_later_write() {
        // Option (a) from the design: no migration on first write.
        // A page faulted via pin_read lives in the read pool; if a
        // subsequent pin_write touches it, it stays there.
        let io = Arc::new(InMemoryPageIo::new());
        let pool = BufferPool::with_split(10, io.clone(), 0.20);
        seed(&io, 1, 0xAA);

        let read_fid = {
            let g = pool.pin_read(PageId::new(1)).unwrap();
            g.frame().id()
        };
        assert!(!pool.is_in_write_pool(read_fid));

        let write_fid = {
            let mut g = pool.pin_write(PageId::new(1), TenantId::DEFAULT).unwrap();
            g.as_bytes_mut()[0] = 0xBB;
            g.frame().id()
        };
        assert_eq!(
            read_fid, write_fid,
            "page should stay in its original frame on write"
        );
    }

    // ---- properties ----

    proptest! {
        #[test]
        fn property_pin_unpin_symmetry(n in 1usize..200) {
            let f = Frame::new(0);
            for _ in 0..n {
                f.pin();
            }
            prop_assert_eq!(f.pin_count() as usize, n);
            for _ in 0..n {
                f.unpin();
            }
            prop_assert_eq!(f.pin_count(), 0);
        }

        #[test]
        fn property_roundtrip_via_pool(values in prop::collection::vec(any::<u8>(), 1..=16)) {
            let io = Arc::new(InMemoryPageIo::new());
            let pool = BufferPool::new(4, io.clone());
            for (i, &v) in values.iter().enumerate() {
                let page = PageId::new(i as u64 + 1);
                let mut seed = [0u8; PAGE_SIZE];
                seed[0] = v;
                io.write_page(page, &seed).unwrap();
            }
            for (i, &v) in values.iter().enumerate() {
                let page = PageId::new(i as u64 + 1);
                let g = pool.pin_read(page).unwrap();
                prop_assert_eq!(g.as_bytes()[0], v);
            }
        }

        #[test]
        fn property_mutations_persist_through_eviction(muts in prop::collection::vec(any::<u8>(), 2..=8)) {
            let io = Arc::new(InMemoryPageIo::new());
            // Pool of size 1 guarantees every new pin evicts.
            let pool = BufferPool::new(1, io.clone());
            for (i, _) in muts.iter().enumerate() {
                let mut seed = [0u8; PAGE_SIZE];
                io.write_page(PageId::new(i as u64 + 1), &seed).unwrap();
                // Silence the unused warning in the loop.
                let _ = &mut seed;
            }
            for (i, &v) in muts.iter().enumerate() {
                let page = PageId::new(i as u64 + 1);
                let mut g = pool.pin_write(page, TenantId::DEFAULT).unwrap();
                g.as_bytes_mut()[0] = v;
            }
            pool.flush_all().unwrap();
            for (i, &v) in muts.iter().enumerate() {
                let page = PageId::new(i as u64 + 1);
                let g = pool.pin_read(page).unwrap();
                prop_assert_eq!(g.as_bytes()[0], v);
            }
        }
    }

    // -----------------------------------------------------------------
    // W16γ M6-07 — MetricsSink wire pins (ADR-045)
    // -----------------------------------------------------------------

    use crate::metrics::{CountingMetricsSink, StoragePageKind};

    /// Pin: when no metrics sink is attached, the pool behaves
    /// identically to the legacy path — no panics, no observable
    /// state change, no perf overhead beyond a nullable-ptr check
    /// (which is the unobservable hot path's `Option::None` branch).
    #[test]
    fn metrics_sink_none_path_pin_works() {
        let (io, pool) = make_pool(2);
        let p1 = PageId::new(1);
        seed(&io, 1, 0xab);
        // pin_read / pin_write should succeed without panicking even
        // when metrics_sink is None.
        let _g = pool.pin_read(p1).unwrap();
    }

    /// Pin: cache HIT on `pin_read` increments
    /// `storage_pages_total{kind="hit"}` and does NOT increment
    /// Miss or Eviction.
    #[test]
    fn metrics_sink_records_hit_on_pin_read_cache_hit() {
        let io = Arc::new(InMemoryPageIo::new());
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_dyn: Arc<dyn MetricsSink> = sink.clone();
        let pool = BufferPool::new(4, io.clone()).with_metrics_sink(sink_dyn);
        let p1 = PageId::new(1);
        seed(&io, 1, 0xab);
        // Initial pin populates the page table (Miss).
        let g1 = pool.pin_read(p1).unwrap();
        drop(g1);
        // Reset baseline counts.
        let miss_after_first = sink.storage_pages_count(StoragePageKind::Miss);
        let evict_after_first = sink.storage_pages_count(StoragePageKind::Eviction);
        let hit_after_first = sink.storage_pages_count(StoragePageKind::Hit);

        // Second pin must hit the cache.
        let _g2 = pool.pin_read(p1).unwrap();
        assert_eq!(
            sink.storage_pages_count(StoragePageKind::Hit),
            hit_after_first + 1,
            "second pin_read must report exactly one Hit"
        );
        // Miss + Eviction unchanged.
        assert_eq!(
            sink.storage_pages_count(StoragePageKind::Miss),
            miss_after_first
        );
        assert_eq!(
            sink.storage_pages_count(StoragePageKind::Eviction),
            evict_after_first
        );
    }

    /// Pin: cold pool MISS on `pin_read` increments
    /// `storage_pages_total{kind="miss"}` and does NOT increment
    /// Eviction (cold slot, no displacement).
    #[test]
    fn metrics_sink_records_miss_without_eviction_on_cold_pool() {
        let io = Arc::new(InMemoryPageIo::new());
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_dyn: Arc<dyn MetricsSink> = sink.clone();
        let pool = BufferPool::new(4, io.clone()).with_metrics_sink(sink_dyn);
        seed(&io, 1, 0xab);
        let _g = pool.pin_read(PageId::new(1)).unwrap();
        assert_eq!(sink.storage_pages_count(StoragePageKind::Miss), 1);
        assert_eq!(sink.storage_pages_count(StoragePageKind::Eviction), 0);
        // First-time pin emits Miss but not Hit (the fast-path
        // try_fast_pin_read returned None — that's how we know we
        // took the slow path).
        assert_eq!(sink.storage_pages_count(StoragePageKind::Hit), 0);
    }

    /// Pin: a full pool's miss displaces a prior page; the metric
    /// sink observes BOTH Miss and Eviction.
    #[test]
    fn metrics_sink_records_eviction_on_full_pool_miss() {
        let io = Arc::new(InMemoryPageIo::new());
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_dyn: Arc<dyn MetricsSink> = sink.clone();
        // Pool of size 1: every new pin evicts the prior page.
        let pool = BufferPool::new(1, io.clone()).with_metrics_sink(sink_dyn);
        seed(&io, 1, 0xab);
        seed(&io, 2, 0xcd);
        let g1 = pool.pin_read(PageId::new(1)).unwrap();
        drop(g1);
        let miss_before = sink.storage_pages_count(StoragePageKind::Miss);
        let evict_before = sink.storage_pages_count(StoragePageKind::Eviction);
        // Pinning a different page displaces the prior mapping.
        let _g2 = pool.pin_read(PageId::new(2)).unwrap();
        assert_eq!(
            sink.storage_pages_count(StoragePageKind::Miss),
            miss_before + 1,
            "second pin (different page) must report Miss"
        );
        assert_eq!(
            sink.storage_pages_count(StoragePageKind::Eviction),
            evict_before + 1,
            "second pin (different page) must report Eviction (displaced prior)"
        );
    }

    /// Pin: hit-rate is computable from
    /// `Hit / (Hit + Miss)` per ADR-045 §"Open questions" — pin a
    /// page twice and verify the ratio is 0.5 (one Miss + one Hit).
    #[test]
    fn metrics_sink_supports_buffer_pool_hit_rate_computation() {
        let io = Arc::new(InMemoryPageIo::new());
        let sink = Arc::new(CountingMetricsSink::new());
        let sink_dyn: Arc<dyn MetricsSink> = sink.clone();
        let pool = BufferPool::new(4, io.clone()).with_metrics_sink(sink_dyn);
        let p = PageId::new(1);
        seed(&io, 1, 0xab);
        let g1 = pool.pin_read(p).unwrap();
        drop(g1);
        let _g2 = pool.pin_read(p).unwrap();

        let hit = sink.storage_pages_count(StoragePageKind::Hit) as f64;
        let miss = sink.storage_pages_count(StoragePageKind::Miss) as f64;
        let hit_rate = hit / (hit + miss);
        assert!(
            (hit_rate - 0.5).abs() < f64::EPSILON,
            "hit_rate must be 0.5 (1 hit / 1 hit + 1 miss); got {}",
            hit_rate
        );
    }
}
