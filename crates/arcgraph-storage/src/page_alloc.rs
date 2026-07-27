//! Monotonic page-id allocator (DEC-13).
//!
//! Hands out fresh [`PageId`]s per `(TenantId, PageType)` pair. Each
//! pair gets an independent `AtomicU64` counter starting at `1`, so the
//! first allocation on a fresh counter is `PageId(1)` (matching the
//! "primary root lives at `PageId(1)`" expectation in DEC-10). Counters
//! are strictly ascending and never recycle — M2.d deliberately skips
//! the free-list (Philosophy §H).
//!
//! The allocator is owned by `CrudStore` and threaded into
//! [`crate::primary_index::PrimaryIndex`] via `Arc` so record-page and
//! index-page allocations share the same monotonic sequence per
//! `(tenant, page_type)` — different page types do not collide because
//! the key tuple discriminates them.
//!
//! Locking: one `DashMap::entry` per allocation. Hot path is a single
//! `AtomicU64::fetch_add` once the counter exists; the entry creation
//! is a one-shot cold path per `(tenant, page_type)` pair.
//!
//! # Exhaustion
//!
//! `PageId` is a `u64`. At 1 M allocations/s this saturates in ≈ 585
//! millennia — treated as "never" for alpha. Overflow wraps and
//! produces `PageId(0)`, which is reserved for the system catalog; a
//! downstream install would reject duplicate-id registration. M2.d
//! does not add a guard.
//!
//! # WAL persistence (issue #129 P0 fix)
//!
//! Pre-fix the allocator state was in-memory only; on WAL recovery the
//! DashMap was reconstructed empty and post-recovery `alloc` returned
//! `PageId(1)` regardless of how many pages pre-fault commits had
//! consumed. The first such recycled `PageId` collided with an existing
//! installed page; the primary index (or its delegated record-store
//! pointer) routed reads through the replay-installed bytes, but a
//! subsequent commit's `install_or_replace(PageId(1), new_bytes)`
//! superseded those bytes and orphaned the prior commit's record
//! through the primary index. ADR-034 D-1 (Strict tier durability)
//! was violated.
//!
//! Post-fix the allocator high-water for every `(tenant, page_type)`
//! pair is durified atomically with each commit via the v4
//! `CommitBundle`'s `allocator_advances` section. Replay seeds the
//! live counter to `max(current, observed + 1)` per (tenant, kind) —
//! Lemma I3 makes the seed monotonic and idempotent under double
//! replay. See [`crate::wal::bundle::AllocatorAdvance`] and the
//! recovery hooks in `Self::seed_from_advance` /
//! `Self::snapshot_advances`.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{PageId, PageType, TenantId};
use dashmap::DashMap;

use crate::wal::bundle::{AllocatorAdvance, AllocatorKind};

/// Monotonic `PageId` allocator, partitioned by `(TenantId, PageType)`.
///
/// Clone-friendly — wrap in [`Arc`] to share across subsystems.
#[derive(Debug, Default)]
pub struct PageAllocator {
    counters: DashMap<(TenantId, u8), Arc<AtomicU64>>,
}

impl PageAllocator {
    /// Canonical [`PageType`] under which the allocator tracks the
    /// **record store**'s single flat page-id domain (#811).
    ///
    /// `RecordPageStore` hosts BOTH [`PageType::Node`] and
    /// [`PageType::Rel`] slotted pages in ONE flat `PageId` keyspace
    /// (`record_store.rs`: `pages: DashMap<PageId, _>`), and its
    /// eventual `BufferPool` home per DEC-17 is likewise a flat
    /// `PageId → frame` map. The per-`(tenant, page_type)`
    /// partitioning this allocator uses elsewhere is safe ONLY because
    /// every *other* page type lives in its own disjoint store; the
    /// record store is the one keyspace fed by two page types. Drawing
    /// Node and Rel record pages from independent counters — each
    /// starting at `PageId(1)` — let the first Rel page collide with
    /// the first Node page in that shared store: the silent dual-write
    /// divergence #811 (and the collateral-corruption half of #812).
    ///
    /// Both record page types therefore MUST draw from ONE monotonic
    /// sequence per tenant. We canonically key it as
    /// [`PageType::Node`] (the record store's first/primary page
    /// type). This is an *allocation-domain* decision only: the
    /// slotted page is still installed with its REAL type stamped in
    /// its [`PageHeader`], so on-disk page identity, the commit
    /// bundle's `Record` staged-page entries, and replay are all
    /// unchanged. Index / blob / vector page types keep their own
    /// per-type sequences (disjoint stores), so DEC-10's "primary root
    /// at `PageId(1)`" is preserved.
    ///
    /// [`PageHeader`]: arcgraph_core::PageHeader
    pub const RECORD_PAGE_DOMAIN: PageType = PageType::Node;

    /// Fresh allocator with no counters provisioned yet.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counters: DashMap::new(),
        }
    }

    /// Allocate the next page id for `(tenant, page_type)`. The first
    /// call on a fresh `(tenant, page_type)` pair returns `PageId(1)`.
    #[must_use]
    pub fn alloc(&self, tenant: TenantId, page_type: PageType) -> PageId {
        let key = (tenant, page_type.as_byte());
        let counter = match self.counters.get(&key) {
            Some(r) => Arc::clone(r.value()),
            None => {
                // Cold path: insert or race with a concurrent insert; either
                // way we end up with exactly one `AtomicU64` per key.
                let fresh = Arc::new(AtomicU64::new(1));
                self.counters
                    .entry(key)
                    .or_insert_with(|| fresh)
                    .value()
                    .clone()
            }
        };
        PageId::new(counter.fetch_add(1, Ordering::Relaxed))
    }

    /// #811: allocate a fresh `PageId` for the **record store**'s
    /// single flat page-id domain — `PageType::Node` and
    /// `PageType::Rel` slotted pages share one `RecordPageStore`
    /// keyspace, so they MUST draw from one monotonic per-tenant
    /// sequence rather than independent `(tenant, page_type)` counters
    /// that both start at `PageId(1)`. The colliding-first-rel bug is
    /// described in detail on [`Self::RECORD_PAGE_DOMAIN`].
    ///
    /// The caller installs the returned page with its REAL page type
    /// (`Node` or `Rel`) in the page header; only the *allocation
    /// domain* is unified, not the page identity.
    #[must_use]
    pub fn alloc_record_page(&self, tenant: TenantId) -> PageId {
        self.alloc(tenant, Self::RECORD_PAGE_DOMAIN)
    }

    /// Peek the next id `alloc(tenant, page_type)` would return without
    /// consuming it. Useful for tests and diagnostics; lock-order safe
    /// because it doesn't mutate the counter.
    #[must_use]
    pub fn peek_next(&self, tenant: TenantId, page_type: PageType) -> PageId {
        let key = (tenant, page_type.as_byte());
        let raw = self
            .counters
            .get(&key)
            .map_or(1, |r| r.value().load(Ordering::Relaxed));
        PageId::new(raw)
    }

    /// Last allocated `PageId` for `(tenant, page_type)`, or `0` if no
    /// allocations have been made yet for this pair. Issue #129 P0 fix
    /// — drained at commit time into the v4 `CommitBundle`'s
    /// `allocator_advances` section so post-recovery `alloc` cannot
    /// reuse an id a pre-fault commit consumed.
    ///
    /// Counter convention: the counter stores the **next** id to
    /// allocate (starting at `1`), so `last_allocated = counter - 1`
    /// when the counter has advanced past pristine, else `0`.
    #[must_use]
    pub fn current_high_water(&self, tenant: TenantId, page_type: PageType) -> u64 {
        let key = (tenant, page_type.as_byte());
        self.counters
            .get(&key)
            .map_or(0, |r| r.value().load(Ordering::Acquire).saturating_sub(1))
    }

    /// Idempotent monotonic seed: ensures the next allocation for
    /// `(tenant, page_type)` returns at least `high_water + 1`.
    /// Replays in commit_lsn order from the v4 `CommitBundle`'s
    /// `allocator_advances` section. Lemma I3 — applying the same
    /// advance twice (or applying an older advance after a newer one)
    /// is a no-op.
    pub fn seed_from_advance(&self, tenant: TenantId, page_type: PageType, high_water: u64) {
        let key = (tenant, page_type.as_byte());
        let target = high_water.saturating_add(1);
        let counter = match self.counters.get(&key) {
            Some(r) => Arc::clone(r.value()),
            None => {
                // Cold path: insert-or-race symmetric with `alloc`.
                // Construct with `target` directly so the cmpxchg
                // loop below has nothing to do in the unraced case.
                let fresh = Arc::new(AtomicU64::new(target));
                self.counters
                    .entry(key)
                    .or_insert_with(|| fresh)
                    .value()
                    .clone()
            }
        };
        // Monotonic-max raise. Race-tolerant: a concurrent alloc on
        // the same counter may have already advanced past `target`,
        // in which case the cmpxchg returns the higher value and we
        // are done. A concurrent alloc that hasn't yet observed our
        // raise will observe it on its next fetch_add.
        let mut cur = counter.load(Ordering::Acquire);
        while cur < target {
            match counter.compare_exchange_weak(cur, target, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// Issue #129 P0 fix: dispatch a single
    /// [`AllocatorAdvance`] into this allocator if the kind names
    /// a `Page*` variant. CRUD-layer `Node` / `Rel` variants are
    /// silently ignored (they route to
    /// [`crate::crud::CrudStore::apply_allocator_advance`]).
    /// Idempotent monotonic-max (Lemma I3).
    pub fn apply_allocator_advance(&self, advance: AllocatorAdvance) {
        if let Some(pt) = advance.kind.page_type() {
            self.seed_from_advance(advance.tenant, pt, advance.new_high_water);
        }
    }

    /// Snapshot the current high-water for every provisioned
    /// `(tenant, page_type)` pair as a vec of [`AllocatorAdvance`]
    /// entries (one per pair where any allocation has happened —
    /// pristine pairs are omitted to keep the wire payload tight).
    ///
    /// Drained at commit time by [`crate::crud::commit`] into the v4
    /// `CommitBundle`'s `allocator_advances` section. The drain is
    /// over the GLOBAL allocator state — all tenants, all page types
    /// — because the encode point is per-commit and the cost is
    /// negligible (≤ N_tenants × N_page_types entries × 17 B). On
    /// replay only the matching (tenant, kind) counters are seeded;
    /// commits that touch only one tenant still durify the global
    /// state, which is the correct conservative behaviour (replay
    /// over-counts harmlessly under monotonic-max).
    #[must_use]
    pub fn snapshot_advances(&self) -> Vec<AllocatorAdvance> {
        let mut out: Vec<AllocatorAdvance> = Vec::new();
        for entry in self.counters.iter() {
            let (tenant, page_type_byte) = *entry.key();
            let counter = entry.value().load(Ordering::Acquire);
            // Skip pristine counters (counter == 1, no allocations yet).
            if counter <= 1 {
                continue;
            }
            // Convert the page_type byte back to PageType. The byte
            // came from `PageType::as_byte()` so `from_byte` will
            // not fail in production; defensively skip on error
            // rather than panic.
            let Ok(page_type) = PageType::from_byte(page_type_byte) else {
                continue;
            };
            out.push(AllocatorAdvance {
                tenant,
                kind: AllocatorKind::for_page_type(page_type),
                new_high_water: counter.saturating_sub(1),
            });
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::thread;

    use super::*;

    #[test]
    fn allocator_monotonic() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        let t = PageType::IndexLeaf;
        // First allocation is PageId(1).
        let p1 = alloc.alloc(tenant, t);
        assert_eq!(p1, PageId::new(1));
        let p2 = alloc.alloc(tenant, t);
        assert_eq!(p2, PageId::new(2));
        let p3 = alloc.alloc(tenant, t);
        assert_eq!(p3, PageId::new(3));
    }

    #[test]
    fn allocator_tenant_isolated() {
        let alloc = PageAllocator::new();
        let t = PageType::IndexLeaf;
        let a = alloc.alloc(TenantId::new(1), t);
        let b = alloc.alloc(TenantId::new(2), t);
        // Different tenants share no sequence; both start from 1.
        assert_eq!(a, PageId::new(1));
        assert_eq!(b, PageId::new(1));
        let a2 = alloc.alloc(TenantId::new(1), t);
        assert_eq!(a2, PageId::new(2));
    }

    #[test]
    fn allocator_page_type_disjoint() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        let leaf = alloc.alloc(tenant, PageType::IndexLeaf);
        let internal = alloc.alloc(tenant, PageType::IndexInternal);
        let node = alloc.alloc(tenant, PageType::Node);
        // Each page type has its own sequence.
        assert_eq!(leaf, PageId::new(1));
        assert_eq!(internal, PageId::new(1));
        assert_eq!(node, PageId::new(1));
        let leaf2 = alloc.alloc(tenant, PageType::IndexLeaf);
        assert_eq!(leaf2, PageId::new(2));
    }

    /// #811: `alloc_record_page` hands Node-bound and Rel-bound record
    /// pages a SINGLE monotonic per-tenant sequence, so a fresh Rel
    /// record page can never collide with a fresh Node record page in
    /// the shared `RecordPageStore` keyspace. RED pre-fix: the old code
    /// allocated Node and Rel from independent `(tenant, PageType)`
    /// counters that BOTH started at `PageId(1)` (see
    /// `allocator_page_type_disjoint`), so the first of each was
    /// `PageId(1)` — a collision in the flat record keyspace.
    #[test]
    fn alloc_record_page_unifies_node_and_rel_domain() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        // Every record-page allocation — regardless of whether the
        // caller will stamp the header Node or Rel — draws from one
        // dense sequence.
        let p1 = alloc.alloc_record_page(tenant);
        let p2 = alloc.alloc_record_page(tenant);
        let p3 = alloc.alloc_record_page(tenant);
        assert_eq!(p1, PageId::new(1));
        assert_eq!(p2, PageId::new(2));
        assert_eq!(p3, PageId::new(3));
        // No two record pages ever share an id.
        let mut seen = HashSet::new();
        for p in [p1, p2, p3] {
            assert!(seen.insert(p), "record-page id {p:?} handed out twice");
        }
        // The unified domain is canonically keyed as `Node`, but it is
        // NOT the same as an independent Rel counter: a direct
        // `alloc(tenant, Rel)` (which no production path uses post-#811)
        // would still start at 1 and thus COLLIDE — proving the bug the
        // unified domain avoids.
        let rogue_rel = alloc.alloc(tenant, PageType::Rel);
        assert_eq!(
            rogue_rel,
            PageId::new(1),
            "an independent Rel counter restarts at 1 — exactly the #811 collision \
             that alloc_record_page avoids by sharing the Node-keyed domain"
        );
        // Index pages keep their OWN disjoint sequence (DEC-10: primary
        // root at PageId(1) is preserved, untouched by record-page
        // allocation).
        assert_eq!(
            alloc.alloc(tenant, PageType::IndexLeaf),
            PageId::new(1),
            "index pages live in a disjoint store and keep their own sequence"
        );
        // And the record domain continued past the 3 we took.
        assert_eq!(alloc.alloc_record_page(tenant), PageId::new(4));
    }

    /// #811: the unified record-page high-water survives the
    /// snapshot→seed bundle round-trip under the canonical
    /// `RECORD_PAGE_DOMAIN` key, so post-recovery record-page
    /// allocation cannot reuse an id a pre-crash commit consumed.
    #[test]
    fn alloc_record_page_high_water_round_trips_via_snapshot() {
        let original = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        // 7 record-page allocations (mix of "node" and "rel" intent —
        // all one domain).
        for _ in 0..7 {
            let _ = original.alloc_record_page(tenant);
        }
        let snap = original.snapshot_advances();
        // Exactly one advance, under the canonical record-page domain.
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].tenant, tenant);
        assert_eq!(
            snap[0].kind,
            AllocatorKind::for_page_type(PageAllocator::RECORD_PAGE_DOMAIN)
        );
        assert_eq!(snap[0].new_high_water, 7);

        // Seed a fresh allocator and prove the next record page is 8
        // (no reuse of 1..=7).
        let restored = PageAllocator::new();
        for adv in &snap {
            let pt = adv.kind.page_type().expect("page-kinded advance");
            restored.seed_from_advance(adv.tenant, pt, adv.new_high_water);
        }
        assert_eq!(restored.alloc_record_page(tenant), PageId::new(8));
    }

    #[test]
    fn peek_next_reports_without_advancing() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        let t = PageType::IndexLeaf;
        assert_eq!(alloc.peek_next(tenant, t), PageId::new(1));
        let _ = alloc.alloc(tenant, t);
        assert_eq!(alloc.peek_next(tenant, t), PageId::new(2));
        assert_eq!(alloc.peek_next(tenant, t), PageId::new(2));
    }

    #[test]
    fn current_high_water_pristine_is_zero() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        // No allocations yet — `0` (no last-allocated id exists).
        assert_eq!(alloc.current_high_water(tenant, PageType::Node), 0);
        let _ = alloc.alloc(tenant, PageType::Node);
        assert_eq!(alloc.current_high_water(tenant, PageType::Node), 1);
        let _ = alloc.alloc(tenant, PageType::Node);
        assert_eq!(alloc.current_high_water(tenant, PageType::Node), 2);
    }

    #[test]
    fn seed_from_advance_advances_pristine_counter() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        // No counter for (tenant, Node) yet. Seeding to high_water=10
        // sets the counter so the next alloc returns PageId(11).
        alloc.seed_from_advance(tenant, PageType::Node, 10);
        assert_eq!(alloc.current_high_water(tenant, PageType::Node), 10);
        assert_eq!(alloc.peek_next(tenant, PageType::Node), PageId::new(11));
        let id = alloc.alloc(tenant, PageType::Node);
        assert_eq!(id, PageId::new(11));
    }

    #[test]
    fn seed_from_advance_is_monotonic_idempotent() {
        let alloc = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        let pt = PageType::IndexLeaf;

        // Seed to 100; current high-water = 100.
        alloc.seed_from_advance(tenant, pt, 100);
        assert_eq!(alloc.current_high_water(tenant, pt), 100);

        // Re-seed with a LOWER value: no regression — high-water stays 100.
        alloc.seed_from_advance(tenant, pt, 50);
        assert_eq!(alloc.current_high_water(tenant, pt), 100);

        // Re-seed with an EQUAL value: idempotent no-op.
        alloc.seed_from_advance(tenant, pt, 100);
        assert_eq!(alloc.current_high_water(tenant, pt), 100);

        // Re-seed with a HIGHER value: advances cleanly.
        alloc.seed_from_advance(tenant, pt, 200);
        assert_eq!(alloc.current_high_water(tenant, pt), 200);
        // Next alloc returns 201.
        assert_eq!(alloc.alloc(tenant, pt), PageId::new(201));
    }

    #[test]
    fn snapshot_advances_skips_pristine_counters() {
        let alloc = PageAllocator::new();
        // Even after a `peek_next` (which reads but does NOT allocate)
        // the snapshot is empty.
        let _ = alloc.peek_next(TenantId::DEFAULT, PageType::Node);
        assert!(alloc.snapshot_advances().is_empty());
        // After one allocation, the snapshot has one entry with
        // high_water = 1.
        let _ = alloc.alloc(TenantId::DEFAULT, PageType::Node);
        let snap = alloc.snapshot_advances();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].tenant, TenantId::DEFAULT);
        assert_eq!(snap[0].kind, AllocatorKind::PageNode);
        assert_eq!(snap[0].new_high_water, 1);
    }

    #[test]
    fn snapshot_advances_covers_all_provisioned_pairs() {
        let alloc = PageAllocator::new();
        let t_a = TenantId::new(42);
        let t_b = TenantId::new(99);
        // 3 allocs on (t_a, Node)
        for _ in 0..3 {
            let _ = alloc.alloc(t_a, PageType::Node);
        }
        // 7 allocs on (t_a, IndexLeaf)
        for _ in 0..7 {
            let _ = alloc.alloc(t_a, PageType::IndexLeaf);
        }
        // 2 allocs on (t_b, Node)
        for _ in 0..2 {
            let _ = alloc.alloc(t_b, PageType::Node);
        }
        let mut snap = alloc.snapshot_advances();
        snap.sort_by_key(|a| (a.tenant.raw(), a.kind.as_byte()));
        assert_eq!(snap.len(), 3);

        // (t_a, Node) → high_water=3
        let a_node = snap
            .iter()
            .find(|a| a.tenant == t_a && a.kind == AllocatorKind::PageNode)
            .unwrap();
        assert_eq!(a_node.new_high_water, 3);
        // (t_a, IndexLeaf) → high_water=7
        let a_leaf = snap
            .iter()
            .find(|a| a.tenant == t_a && a.kind == AllocatorKind::PageIndexLeaf)
            .unwrap();
        assert_eq!(a_leaf.new_high_water, 7);
        // (t_b, Node) → high_water=2
        let b_node = snap
            .iter()
            .find(|a| a.tenant == t_b && a.kind == AllocatorKind::PageNode)
            .unwrap();
        assert_eq!(b_node.new_high_water, 2);
    }

    #[test]
    fn snapshot_then_seed_round_trips_high_water() {
        // The bundle round-trip path: alloc N, snapshot, fresh
        // allocator, seed each advance, verify current_high_water +
        // next alloc match.
        let original = PageAllocator::new();
        let tenant = TenantId::DEFAULT;
        for _ in 0..10 {
            let _ = original.alloc(tenant, PageType::Node);
        }
        for _ in 0..5 {
            let _ = original.alloc(tenant, PageType::IndexLeaf);
        }
        let snap = original.snapshot_advances();

        let restored = PageAllocator::new();
        for adv in &snap {
            let pt = adv
                .kind
                .page_type()
                .expect("snapshot only emits page-kinded advances");
            restored.seed_from_advance(adv.tenant, pt, adv.new_high_water);
        }
        assert_eq!(restored.current_high_water(tenant, PageType::Node), 10);
        assert_eq!(restored.current_high_water(tenant, PageType::IndexLeaf), 5);
        // Next alloc on (tenant, Node) returns PageId(11) — proves
        // recovery cannot reuse PageIds 1..=10.
        assert_eq!(restored.alloc(tenant, PageType::Node), PageId::new(11));
    }

    #[test]
    fn concurrent_alloc_produces_unique_ids() {
        let alloc = Arc::new(PageAllocator::new());
        let tenant = TenantId::DEFAULT;
        let t = PageType::IndexLeaf;
        const THREADS: usize = 4;
        const PER_THREAD: usize = 500;
        let mut handles = Vec::with_capacity(THREADS);
        for _ in 0..THREADS {
            let alloc = Arc::clone(&alloc);
            handles.push(thread::spawn(move || {
                let mut out = Vec::with_capacity(PER_THREAD);
                for _ in 0..PER_THREAD {
                    out.push(alloc.alloc(tenant, t));
                }
                out
            }));
        }
        let mut collected = HashSet::new();
        for h in handles {
            for id in h.join().unwrap() {
                // Every id unique.
                assert!(collected.insert(id), "duplicate page id: {id:?}");
            }
        }
        assert_eq!(collected.len(), THREADS * PER_THREAD);
        // Strictly dense from 1..=THREADS*PER_THREAD.
        for i in 1..=(THREADS * PER_THREAD) as u64 {
            assert!(collected.contains(&PageId::new(i)));
        }
    }
}
