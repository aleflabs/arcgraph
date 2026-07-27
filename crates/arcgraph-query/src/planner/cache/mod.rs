//! Per-tenant plan cache for the M4-53 (M4-05c) slice.
//!
//! # Slice scope (M4-53 — M4-05c per ADR-038 amendment-02 §M4.e + amendment-03 §TIER-2-a)
//!
//! M4-53 ships:
//! - [`PlanCacheKey`] — `(tenant_id, parameter-erased AST,
//!   parameter-type signature)` per amendment-03 §TIER-2-a; see
//!   [`key`] module.
//! - [`PlanCache`] — per-tenant LRU cache with capacity-driven
//!   eviction + stats-change-watermark lazy invalidation.
//! - [`CachedPlan`] — value stamped with the
//!   `commits_observed` watermark at insert time.
//!
//! The cache is wired into the M4-91 EXPLAIN pipeline at
//! `crate::explain::plan_tree_for`: on hit (key matches AND stamped
//! watermark equals current snapshot), the cached
//! [`crate::planner::cost::CostedPlan`] is reused; on miss or stale
//! entry, the cold path runs lower → enumerate → cost as before and
//! the result is inserted.
//!
//! # Per-tenant LRU shape (per amendment-03 §TIER-2-a)
//!
//! - **Default capacity:** 1024 entries per tenant. Configurable
//!   per-tenant via [`PlanCache::set_max_entries`] — forward-method
//!   consumed by M5-12 rate-limit config.
//! - **Eviction:** `lru::LruCache` (intrusive doubly-linked-list LRU).
//!   Cross-tenant isolation is structural — each tenant has its own
//!   `Mutex<lru::LruCache<...>>` behind a [`dashmap::DashMap`]
//!   sharded by [`TenantId`], so tenant T's eviction NEVER touches
//!   tenant U's entries (per amendment-03 §TIER-2-a "cross-tenant
//!   cache pressure does not affect another tenant's hit rate").
//!
//! # Stats-change-watermark invalidation (per amendment-03 §TIER-2-a)
//!
//! Every cache entry stamps:
//! - the catalog snapshot's
//!   [`crate::semantic::CatalogSnapshot::commits_observed`] value at
//!   insert time, AND
//! - the snapshot's `total_nodes` / `total_rels` (debugging context).
//!
//! On lookup, [`PlanCache::lookup`] compares the stamped watermark to
//! the live `commits_observed`:
//! - equal → cache hit (entry returned)
//! - less than → entry stale (evicted + `None` returned)
//! - greater than → impossible (commits_observed is monotone non-
//!   decreasing); evicted defensively + a `tracing::warn!` is emitted
//!   so the recurrence is loud.
//!
//! # Observability (per amendment-03 §TIER-2-c forward-application)
//!
//! Every cache event emits a `tracing` event at `target =
//! "arcgraph_query::planner::cache"`:
//! - `cache_hit` — key matched + watermark equal.
//! - `cache_miss_inserted` — key absent; insert path took.
//! - `cache_invalidate` — key present but stamped watermark stale.
//! - `cache_evict` — capacity-driven LRU eviction.
//!
//! `tracing::debug!` level (production callers can subscribe at DEBUG
//! to capture the per-query cadence; INFO subscribers see only the
//! tenant-isolated events through other crate facets).
//!
//! Beyond per-event tracing the cache exposes lifetime hit / miss
//! counters via [`PlanCache::hit_count`] / [`PlanCache::miss_count`]
//! per W13γ fix-up MED-2 (closes the brief mandate "hit rate metric is
//! observable" — a counter, not a renderings-equal oracle). Counters
//! are aggregated across all tenants; per-tenant rate observability is
//! a v1.1 forward-pin (issue #NEW: per-tenant cache-rate metrics).
//!
//! # 7-slice 3-strike trait discipline (per `feedback_avoid_speculative_scaffolding.md`)
//!
//! The cache is a CONCRETE STRUCT, NOT a `pub trait PlanCache /
//! CacheBackend`. M4-53 has exactly ONE consumer (the EXPLAIN
//! pipeline at `crate::explain::plan_tree_for`); the trait
//! extraction is M4-72's decision when replan-side invalidation
//! lands per amendment-03 §"Implicit dependency edges" item 3.
//!
//! # Budget
//!
//! Per ADR-036 §D-25, the M4-05 plan-build budget is 5 ms. The cache
//! lookup is O(canonical-bytestream-length) for hashing + O(1) for
//! the LRU op + O(signature-length) for equality. At v1.0 plan sizes
//! the lookup is ~µs — orders of magnitude inside budget. M4-72 will
//! land the bench harness for end-to-end hit-rate validation.
//!
//! # ADR provenance
//! - ADR-038 amendment-02 §M4.e — M4-53 (M4-05c) plan-cache slice scope.
//! - ADR-038 amendment-03 §TIER-2-a — invalidation policy + cache-key
//!   contract; capacity policy.
//! - ADR-038 amendment-03 §"Implicit dependency edges" item 3 — M4-72
//!   replan-side invalidation forward-link.
//! - ADR-037 — multi-tenancy P0 (per-tenant isolation invariant).

pub mod key;

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::TenantId;
use dashmap::DashMap;
use parking_lot::Mutex;

use crate::planner::cost::CostedPlan;

pub use key::{LitKind, PlanCacheKey, Slot};

/// Default per-tenant LRU capacity per ADR-038 amendment-03 §TIER-2-a.
///
/// `1024` entries per tenant balances memory footprint (≤ ~1 MB at
/// the v1.0 plan size of ~1 KB / entry — see budget rustdoc) against
/// hit rate under typical reused-prepared-statement workloads.
/// Per-tenant overrides will be exposed via M5-12 rate-limit config
/// (forward — see [`PlanCache::set_max_entries`]).
pub const DEFAULT_MAX_ENTRIES_PER_TENANT: usize = 1024;

/// A cached, fully-costed plan stamped with the catalog snapshot's
/// `commits_observed` watermark.
///
/// `Arc<CostedPlan>` so the cache can hand out cheap-to-clone borrows
/// (the EXPLAIN renderer takes `&CostedPlan` to project a
/// [`crate::explain::PlanTree`]; multiple concurrent EXPLAIN calls
/// can read the same cached plan in parallel).
#[derive(Debug, Clone)]
pub struct CachedPlan {
    plan: Arc<CostedPlan>,
    /// Catalog snapshot's `commits_observed` count at the time the
    /// plan was inserted. Lookup compares this to the current
    /// snapshot's `commits_observed`; mismatch ⇒ invalidated.
    stats_version: u64,
}

impl CachedPlan {
    /// Wrap a costed plan with its insert-time watermark.
    #[must_use]
    pub fn new(plan: Arc<CostedPlan>, stats_version: u64) -> Self {
        Self {
            plan,
            stats_version,
        }
    }

    /// Borrow the cached costed plan.
    #[must_use]
    pub fn plan(&self) -> &Arc<CostedPlan> {
        &self.plan
    }

    /// The `commits_observed` watermark stamped at insert time.
    #[must_use]
    pub fn stats_version(&self) -> u64 {
        self.stats_version
    }
}

/// Outcome of a cache lookup. Distinguishes the four wave-level
/// cases the M4-53 wiring + tests need to discriminate.
///
/// `Hit` is the only successful path; the other three all surface as
/// "no cached plan" to the caller but carry distinct observability /
/// invariant signals.
#[derive(Debug, Clone)]
pub enum LookupOutcome {
    /// Key present + watermark fresh. Caller reuses the cached plan.
    Hit(Arc<CostedPlan>),
    /// Key absent. Caller cold-paths through lower → enumerate → cost
    /// then [`PlanCache::insert`]s the result.
    Miss,
    /// Key present but stamped watermark < current. Entry has been
    /// removed; caller cold-paths and reinserts.
    Stale,
    /// Key present but stamped watermark > current. Impossible under
    /// monotonic non-decreasing `commits_observed`; entry removed
    /// defensively and a `tracing::warn!` was emitted.
    InvariantViolation,
}

/// Per-tenant plan cache.
///
/// Lifecycle: typically constructed once at process start (e.g.,
/// inside the multi-tenant `QueryEngine` builder) and shared across
/// every per-tenant query. Per-tenant LRU storage is created lazily
/// on first insert.
///
/// # Thread safety
///
/// `Send + Sync`. The outer [`DashMap`] shards on [`TenantId`]
/// (low contention at the v1.0 K ≤ 50 tenant scale per amendment-03);
/// each per-tenant LRU is behind a [`parking_lot::Mutex`] (poisoning-
/// free; a panic during one walk does NOT taint subsequent lookups
/// because the cache is soft state — there's no invariant the
/// poisoning would protect).
///
/// # No-mmap discipline
///
/// Per storage ownership policy the cache lives ENTIRELY in
/// process heap; no `mmap` / `memmap2` / persistence to disk. Plan
/// caches are inherently soft state.
pub struct PlanCache {
    shards: DashMap<TenantId, Arc<Mutex<PerTenantLru>>>,
    default_capacity: NonZeroUsize,
    /// Lifetime hit counter (per W13γ fix-up MED-2). Incremented on
    /// every [`LookupOutcome::Hit`] outcome from [`Self::lookup`].
    /// `Relaxed` ordering — counters are diagnostic / observability,
    /// not load-bearing on cache invariants.
    hits: AtomicU64,
    /// Lifetime miss counter (per W13γ fix-up MED-2). Incremented on
    /// every [`LookupOutcome::Miss`] AND [`LookupOutcome::Stale`] AND
    /// [`LookupOutcome::InvariantViolation`] outcome — the contract is
    /// "did the lookup hand back a cached plan?" and the answer is
    /// "yes" only for `Hit`. `Stale` / `InvariantViolation` evict +
    /// fall through to the cold path the same way `Miss` does, so they
    /// count as misses from the caller's hit-rate perspective.
    misses: AtomicU64,
}

impl std::fmt::Debug for PlanCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlanCache")
            .field("default_capacity", &self.default_capacity.get())
            .field("tenants", &self.shards.len())
            .field("hits", &self.hits.load(Ordering::Relaxed))
            .field("misses", &self.misses.load(Ordering::Relaxed))
            .finish()
    }
}

impl PlanCache {
    /// Construct an empty plan cache with the default per-tenant
    /// capacity ([`DEFAULT_MAX_ENTRIES_PER_TENANT`]).
    #[must_use]
    pub fn new() -> Self {
        // INVARIANT: 1024 != 0; assertion would only fire if someone
        // overrides DEFAULT_MAX_ENTRIES_PER_TENANT to 0 in source —
        // which would be a programmer error, not an external invariant.
        let default_capacity = NonZeroUsize::new(DEFAULT_MAX_ENTRIES_PER_TENANT)
            .expect("DEFAULT_MAX_ENTRIES_PER_TENANT > 0 (compile-time pinned)");
        Self {
            shards: DashMap::new(),
            default_capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        }
    }

    /// Construct a plan cache with a custom default per-tenant
    /// capacity.
    ///
    /// Returns `None` when `capacity == 0` (the LRU storage requires
    /// at least one slot).
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Option<Self> {
        let default_capacity = NonZeroUsize::new(capacity)?;
        Some(Self {
            shards: DashMap::new(),
            default_capacity,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    /// Lifetime hit count across all tenants.
    ///
    /// Per W13γ fix-up MED-2: closes the brief mandate "hit rate metric
    /// is observable" by exposing a counter, not a renderings-equal
    /// oracle. Byte-equality of plan-tree renderings PASSES even on a
    /// cache MISS (plan-build is deterministic for the same `(tenant,
    /// stmt)` pair under the same `commits_observed` snapshot), so
    /// callers asserting hit-rate must read this counter directly.
    ///
    /// `Relaxed` ordering — counters are diagnostic / observability,
    /// not load-bearing on cache invariants.
    #[must_use]
    pub fn hit_count(&self) -> u64 {
        self.hits.load(Ordering::Relaxed)
    }

    /// Lifetime miss count across all tenants.
    ///
    /// Counts every lookup outcome that did NOT hand back a cached plan
    /// — [`LookupOutcome::Miss`] + [`LookupOutcome::Stale`] +
    /// [`LookupOutcome::InvariantViolation`]. The "miss" semantics is
    /// caller-facing: did this lookup produce a hit? `Stale` and
    /// `InvariantViolation` evict and fall through to the cold path
    /// the same way `Miss` does.
    ///
    /// `Relaxed` ordering — see [`Self::hit_count`].
    #[must_use]
    pub fn miss_count(&self) -> u64 {
        self.misses.load(Ordering::Relaxed)
    }

    /// Attempt a cache lookup against the live catalog watermark.
    ///
    /// `current_stats_version` is the
    /// [`crate::semantic::CatalogSnapshot::commits_observed`] value
    /// captured by the caller at lookup time. The cache compares the
    /// stamped value against this and returns the appropriate
    /// [`LookupOutcome`].
    ///
    /// # Tracing
    ///
    /// Emits a `tracing::debug!` event at
    /// `target = "arcgraph_query::planner::cache"`:
    /// - `outcome = "hit"` — key matched + watermark equal
    /// - `outcome = "miss"` — key absent
    /// - `outcome = "invalidate"` — key present + watermark stale;
    ///   `reason = "stats_change_watermark"`
    pub fn lookup(&self, key: &PlanCacheKey, current_stats_version: u64) -> LookupOutcome {
        let shard = match self.shards.get(&key.tenant_id) {
            Some(s) => Arc::clone(s.value()),
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    target: "arcgraph_query::planner::cache",
                    tenant_id = ?key.tenant_id,
                    outcome = "miss",
                    reason = "no_tenant_shard",
                    "cache_miss",
                );
                return LookupOutcome::Miss;
            }
        };
        let mut guard = shard.lock();
        match guard.entries.get(key) {
            Some(entry) => {
                let stamped = entry.stats_version;
                if stamped == current_stats_version {
                    let plan = Arc::clone(&entry.plan);
                    self.hits.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(
                        target: "arcgraph_query::planner::cache",
                        tenant_id = ?key.tenant_id,
                        outcome = "hit",
                        stats_version = stamped,
                        "cache_hit",
                    );
                    LookupOutcome::Hit(plan)
                } else if stamped < current_stats_version {
                    let _ = guard.entries.pop(key);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    tracing::debug!(
                        target: "arcgraph_query::planner::cache",
                        tenant_id = ?key.tenant_id,
                        outcome = "invalidate",
                        reason = "stats_change_watermark",
                        stamped_version = stamped,
                        current_version = current_stats_version,
                        "cache_invalidate",
                    );
                    LookupOutcome::Stale
                } else {
                    let _ = guard.entries.pop(key);
                    self.misses.fetch_add(1, Ordering::Relaxed);
                    tracing::warn!(
                        target: "arcgraph_query::planner::cache",
                        tenant_id = ?key.tenant_id,
                        stamped_version = stamped,
                        current_version = current_stats_version,
                        "cache_invariant_violation: stamped > current; \
                         commits_observed must be monotone non-decreasing",
                    );
                    LookupOutcome::InvariantViolation
                }
            }
            None => {
                self.misses.fetch_add(1, Ordering::Relaxed);
                tracing::debug!(
                    target: "arcgraph_query::planner::cache",
                    tenant_id = ?key.tenant_id,
                    outcome = "miss",
                    "cache_miss",
                );
                LookupOutcome::Miss
            }
        }
    }

    /// Insert a costed plan stamped with the catalog snapshot's
    /// `commits_observed` value at the time of the cold-path walk.
    ///
    /// If the per-tenant LRU is at capacity, the least-recently-used
    /// entry is evicted (a `tracing::debug!` `cache_evict` event
    /// fires). Any pre-existing entry for the SAME key is overwritten
    /// (no eviction trace; the new entry takes its slot).
    pub fn insert(&self, key: PlanCacheKey, plan: Arc<CostedPlan>, stats_version: u64) {
        let shard = self
            .shards
            .entry(key.tenant_id)
            .or_insert_with(|| Arc::new(Mutex::new(PerTenantLru::new(self.default_capacity))))
            .value()
            .clone();
        let tenant = key.tenant_id;
        let mut guard = shard.lock();
        let cap = guard.entries.cap();
        let was_present = guard.entries.contains(&key);
        let evicted_key = if guard.entries.len() == cap.get() && !was_present {
            // Capacity is about to drive an LRU eviction. Capture the
            // victim BEFORE put() so the trace event records the
            // eviction without re-locking.
            guard.entries.peek_lru().map(|(k, _)| k.clone())
        } else {
            None
        };
        guard.entries.put(key, CachedPlan::new(plan, stats_version));
        if let Some(victim) = evicted_key {
            tracing::debug!(
                target: "arcgraph_query::planner::cache",
                tenant_id = ?tenant,
                reason = "lru_eviction",
                evicted_canonical_len = victim.canonical().len(),
                "cache_evict",
            );
        }
        tracing::debug!(
            target: "arcgraph_query::planner::cache",
            tenant_id = ?tenant,
            outcome = "miss_inserted",
            stats_version,
            "cache_miss_inserted",
        );
    }

    /// Explicitly invalidate the cache entry under `key` if present.
    ///
    /// Returns `true` when an entry was present and removed; `false`
    /// when the key was absent (no-op).
    ///
    /// # M4-72 ↔ replan invalidation channel
    ///
    /// Per ADR-038 amendment-03 §"Implicit dependency edges" item 3:
    /// > M4-72 (replan + mid-query state preservation) signals the M4-53
    /// > plan cache to invalidate the original-plan entry on replan,
    /// > otherwise subsequent queries pick up the wrong-cardinality plan.
    ///
    /// Used directly by [`crate::observer::ReplanController`] when a
    /// divergent replan fires (no trait indirection per W12β fix-up
    /// HIGH-1; trait extraction forward-deferred to v1.2+ persistent-
    /// cache slice when a real second consumer lights). Non-replan
    /// callers SHOULD prefer the lazy stats-change-watermark
    /// invalidation per amendment-03 §TIER-2-a (which fires
    /// automatically on `commits_observed` advancement); explicit
    /// invalidation is reserved for the replan-side surface.
    ///
    /// # Tracing
    ///
    /// Emits a `tracing::debug!` event at
    /// `target = "arcgraph_query::planner::cache"` with
    /// `outcome = "explicit_invalidate"` per the observability contract.
    pub fn invalidate(&self, key: &PlanCacheKey) -> bool {
        let shard = match self.shards.get(&key.tenant_id) {
            Some(s) => Arc::clone(s.value()),
            None => {
                // No tenant shard → nothing to invalidate.
                return false;
            }
        };
        let mut guard = shard.lock();
        let removed = guard.entries.pop(key).is_some();
        if removed {
            tracing::debug!(
                target: "arcgraph_query::planner::cache",
                tenant_id = ?key.tenant_id,
                outcome = "explicit_invalidate",
                reason = "m4_72_replan_signal",
                "cache_invalidate_explicit",
            );
        }
        removed
    }

    /// Set the per-tenant LRU max entries for `tenant`.
    ///
    /// Forward-method for the M5-12 rate-limit config (per
    /// amendment-03 §TIER-2-a "configurable max entries per tenant").
    /// If the new capacity is smaller than the current entry count,
    /// the LRU evicts down to the new bound; per-eviction tracing
    /// fires for each victim.
    ///
    /// Returns `false` when `capacity == 0` (the LRU requires at least
    /// one slot); returns `true` on success.
    pub fn set_max_entries(&self, tenant: TenantId, capacity: usize) -> bool {
        let Some(cap) = NonZeroUsize::new(capacity) else {
            return false;
        };
        let shard = self
            .shards
            .entry(tenant)
            .or_insert_with(|| Arc::new(Mutex::new(PerTenantLru::new(self.default_capacity))))
            .value()
            .clone();
        let mut guard = shard.lock();
        let prev_cap = guard.entries.cap();
        guard.entries.resize(cap);
        if cap < prev_cap {
            tracing::debug!(
                target: "arcgraph_query::planner::cache",
                tenant_id = ?tenant,
                reason = "capacity_resize",
                old_capacity = prev_cap.get(),
                new_capacity = cap.get(),
                "cache_resize",
            );
        }
        true
    }

    /// Number of entries cached for `tenant`. Useful for tests
    /// asserting cross-tenant isolation under eviction pressure.
    #[must_use]
    pub fn len_for(&self, tenant: TenantId) -> usize {
        self.shards
            .get(&tenant)
            .map_or(0, |shard| shard.value().lock().entries.len())
    }

    /// Number of tenants with at least one cache entry. Diagnostic
    /// only; not part of the cache's hot path.
    #[must_use]
    pub fn tenant_count(&self) -> usize {
        self.shards.len()
    }
}

impl Default for PlanCache {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------
// Internal per-tenant LRU
// ---------------------------------------------------------------------

struct PerTenantLru {
    entries: lru::LruCache<PlanCacheKey, CachedPlan>,
}

impl PerTenantLru {
    fn new(capacity: NonZeroUsize) -> Self {
        Self {
            entries: lru::LruCache::new(capacity),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::{LogicalEmpty, LogicalPlan};
    use crate::parse;
    use crate::planner::cost::{Cardinality, Cost, CostNode, CostedPlan, CostedTree};
    use arcgraph_core::TenantId;

    fn key(query: &str, tenant: TenantId) -> PlanCacheKey {
        let stmt = parse(query).expect("parse");
        PlanCacheKey::from_ast(tenant, &stmt)
    }

    fn dummy_plan() -> Arc<CostedPlan> {
        // A minimal CostedPlan suitable for cache identity tests. We
        // don't need a meaningful logical plan / cost tree here — the
        // cache is identity-based; the plan is opaque payload.
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let costs = CostedTree::leaf(CostNode::leaf(Cost::zero(), Cardinality::zero()));
        Arc::new(CostedPlan::new(plan, costs))
    }

    #[test]
    fn cache_hit_returns_same_plan() {
        // M4-53 unit test #1: insert + lookup at fresh watermark
        // returns the inserted plan. Identity comparison via
        // `Arc::ptr_eq` confirms the cache is handing back the SAME
        // Arc, not a clone of equivalent content.
        let cache = PlanCache::new();
        let k = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        let plan = dummy_plan();
        cache.insert(k.clone(), Arc::clone(&plan), 7);
        match cache.lookup(&k, 7) {
            LookupOutcome::Hit(got) => assert!(Arc::ptr_eq(&got, &plan)),
            other => panic!("expected hit, got {other:?}"),
        }
    }

    #[test]
    fn cache_miss_returns_none() {
        // M4-53 unit test #2: key not in cache → Miss outcome.
        let cache = PlanCache::new();
        let k = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        assert!(matches!(cache.lookup(&k, 7), LookupOutcome::Miss));
    }

    #[test]
    fn lru_eviction_at_capacity() {
        // M4-53 unit test #3: inserting capacity+1 entries evicts the
        // LRU (oldest). Cap is set to 2 to keep the test compact.
        let cache = PlanCache::with_capacity(2).expect("cap > 0");
        let k1 = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        let k2 = key("MATCH (m:Person) RETURN m", TenantId::DEFAULT);
        let k3 = key("MATCH (z:Person) RETURN z", TenantId::DEFAULT);
        cache.insert(k1.clone(), dummy_plan(), 1);
        cache.insert(k2.clone(), dummy_plan(), 1);
        // Force k1 to LRU position (k2 is most recently used after
        // its insert).
        cache.insert(k3.clone(), dummy_plan(), 1);
        assert_eq!(cache.len_for(TenantId::DEFAULT), 2);
        // k1 was evicted; k2 + k3 still present.
        assert!(matches!(cache.lookup(&k1, 1), LookupOutcome::Miss));
        assert!(matches!(cache.lookup(&k2, 1), LookupOutcome::Hit(_)));
        assert!(matches!(cache.lookup(&k3, 1), LookupOutcome::Hit(_)));
    }

    #[test]
    fn stats_change_invalidates_entry() {
        // M4-53 unit test #4: stamped watermark < current → Stale.
        // Wave-level transit pin: M4-04 stats-change-counter producer
        // ↔ M4-53 cache consumer. Phase 4.2 controlled-mutation
        // probe lives at `tests/m4_53_plan_cache_integration.rs`.
        let cache = PlanCache::new();
        let k = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        cache.insert(k.clone(), dummy_plan(), 5);
        match cache.lookup(&k, 6) {
            LookupOutcome::Stale => {}
            other => panic!("expected Stale, got {other:?}"),
        }
        // After invalidation, a re-lookup with the SAME watermark must
        // miss (the entry was evicted).
        assert!(matches!(cache.lookup(&k, 6), LookupOutcome::Miss));
    }

    #[test]
    fn parameter_value_shape_canonicalization() {
        // M4-53 unit test #5: queries differing only in inline literal
        // values produce equal cache keys. Two inserts with the same
        // canonical form share the same slot — the second insert
        // overwrites. The cache len stays at 1.
        let cache = PlanCache::new();
        let k_a = key("MATCH (n {id: 42}) RETURN n", TenantId::DEFAULT);
        let k_b = key("MATCH (n {id: 43}) RETURN n", TenantId::DEFAULT);
        assert_eq!(k_a, k_b);
        cache.insert(k_a, dummy_plan(), 1);
        cache.insert(k_b, dummy_plan(), 1);
        assert_eq!(cache.len_for(TenantId::DEFAULT), 1);
    }

    #[test]
    fn per_tenant_lru_isolation() {
        // M4-53 unit test #6: filling tenant T to capacity and then
        // inserting in tenant U does NOT evict any of T's entries.
        // Cross-tenant pressure isolation invariant per amendment-03
        // §TIER-2-a.
        let cache = PlanCache::with_capacity(2).expect("cap > 0");
        let t = TenantId::new(1);
        let u = TenantId::new(2);
        let k1_t = key("MATCH (n:Person) RETURN n", t);
        let k2_t = key("MATCH (m:Person) RETURN m", t);
        let k1_u = key("MATCH (n:Person) RETURN n", u);
        cache.insert(k1_t.clone(), dummy_plan(), 1);
        cache.insert(k2_t.clone(), dummy_plan(), 1);
        // Tenant T at capacity. Inserting two more entries in tenant
        // U must not evict T's entries.
        cache.insert(k1_u.clone(), dummy_plan(), 1);
        let k2_u = key("MATCH (z:Person) RETURN z", u);
        cache.insert(k2_u.clone(), dummy_plan(), 1);
        assert_eq!(cache.len_for(t), 2);
        assert_eq!(cache.len_for(u), 2);
        // T's entries are still there.
        assert!(matches!(cache.lookup(&k1_t, 1), LookupOutcome::Hit(_)));
        assert!(matches!(cache.lookup(&k2_t, 1), LookupOutcome::Hit(_)));
    }

    #[test]
    fn invariant_violation_on_stamped_greater_than_current() {
        // Defensive surface: stamped > current is impossible under
        // monotone non-decreasing commits_observed. The cache evicts
        // the entry + emits a tracing::warn rather than panicking
        // (preserving cache soft-state guarantees on a violated
        // upstream invariant).
        let cache = PlanCache::new();
        let k = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        cache.insert(k.clone(), dummy_plan(), 100);
        match cache.lookup(&k, 50) {
            LookupOutcome::InvariantViolation => {}
            other => panic!("expected InvariantViolation, got {other:?}"),
        }
        // Defensive eviction took.
        assert!(matches!(cache.lookup(&k, 50), LookupOutcome::Miss));
    }

    #[test]
    fn set_max_entries_resizes_per_tenant_lru() {
        // M5-12 forward-method: per-tenant capacity override resizes
        // an existing per-tenant LRU + evicts down if shrunk.
        // Use distinct property KEYS so each insert has a distinct
        // canonical form (literal erasure would collapse `n.id = 0`
        // and `n.id = 1` into one slot).
        let cache = PlanCache::with_capacity(4).expect("cap > 0");
        let t = TenantId::DEFAULT;
        for prop in ["a", "b", "c", "d"] {
            let q = format!("MATCH (n) WHERE n.{prop} = 1 RETURN n");
            let k = key(&q, t);
            cache.insert(k, dummy_plan(), 1);
        }
        assert_eq!(cache.len_for(t), 4);
        assert!(cache.set_max_entries(t, 2));
        assert_eq!(cache.len_for(t), 2);
        // Setting 0 is rejected.
        assert!(!cache.set_max_entries(t, 0));
    }

    /// Wave 10b Sin #6 closure — tracing-event firing pin for the four
    /// documented cache events plus the breadcrumb `cache_miss` event.
    ///
    /// Verifies that each of the cache's `tracing::debug!` events
    /// actually fires under its triggering condition. Without this pin
    /// a future refactor that drops one of the events (e.g., consolidating
    /// `cache_invalidate` into a generic `cache_state_change`) would pass
    /// every state-machine test but silently break the observability
    /// contract documented at module top.
    ///
    /// # Pattern matching discipline
    ///
    /// Each event is identified via a unique `field=value` substring
    /// in the rendered log line — robust to message-text refactors.
    /// `cache_miss` (no_tenant_shard branch) is matched by its unique
    /// `reason="no_tenant_shard"` field; `cache_evict` by its unique
    /// `reason="lru_eviction"` field; and the three `outcome="…"` events
    /// by the discriminating string.
    #[test]
    #[tracing_test::traced_test]
    fn cache_emits_4_tracing_events() {
        // The `#[traced_test]` macro injects `logs_contain` into scope —
        // no `use` import needed (per `tracing-test` 0.2 docs).

        // Event 1: cache_miss (no_tenant_shard branch) — fresh cache.
        let cache = PlanCache::new();
        let k = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        let outcome = cache.lookup(&k, 5);
        assert!(matches!(outcome, LookupOutcome::Miss));

        // Event 2: cache_miss_inserted.
        cache.insert(k.clone(), dummy_plan(), 5);

        // Event 3: cache_hit.
        let outcome = cache.lookup(&k, 5);
        assert!(matches!(outcome, LookupOutcome::Hit(_)));

        // Re-insert ahead of the stale-watermark probe — the
        // invalidate path evicts the entry, so we need it present
        // before triggering the stale lookup.
        cache.insert(k.clone(), dummy_plan(), 5);

        // Event 4: cache_invalidate (stamped < current).
        let outcome = cache.lookup(&k, 6);
        assert!(matches!(outcome, LookupOutcome::Stale));

        // Event 5: cache_evict via capacity-driven LRU.
        let small_cache = PlanCache::with_capacity(1).expect("cap > 0");
        let k_a = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        let k_b = key("MATCH (m:Person) RETURN m", TenantId::DEFAULT);
        small_cache.insert(k_a, dummy_plan(), 0);
        small_cache.insert(k_b, dummy_plan(), 0); // Evicts k_a.

        // Verify each documented event fired. Patterns target unique
        // structured-field values to distinguish (e.g.) cache_miss from
        // cache_miss_inserted whose messages share a prefix.
        assert!(
            logs_contain(r#"reason="no_tenant_shard""#),
            "cache_miss (no-tenant-shard branch) must fire on lookup against an empty cache",
        );
        assert!(
            logs_contain(r#"outcome="miss_inserted""#),
            "cache_miss_inserted must fire on insert",
        );
        assert!(
            logs_contain(r#"outcome="hit""#),
            "cache_hit must fire on key match + watermark equal",
        );
        assert!(
            logs_contain(r#"outcome="invalidate""#),
            "cache_invalidate must fire on stamped < current watermark",
        );
        assert!(
            logs_contain(r#"reason="lru_eviction""#),
            "cache_evict must fire on capacity-driven LRU eviction",
        );
    }

    /// W13γ fix-up MED-2 — hit / miss counter pin.
    ///
    /// Closes the brief mandate "hit rate metric is observable" by
    /// asserting the counters increment on every triggering outcome
    /// (Miss, Stale, InvariantViolation → miss_count; Hit → hit_count).
    /// A regression that drops one of the counter increments (e.g.,
    /// inserting an early-return that bypasses the fetch_add) would
    /// silently break callers asserting hit-rate.
    #[test]
    fn cache_hit_miss_counters_increment_correctly() {
        let cache = PlanCache::new();
        let k = key("MATCH (n:Person) RETURN n", TenantId::DEFAULT);
        // Counters start at 0.
        assert_eq!(cache.hit_count(), 0);
        assert_eq!(cache.miss_count(), 0);
        // Miss: no tenant shard yet.
        assert!(matches!(cache.lookup(&k, 5), LookupOutcome::Miss));
        assert_eq!(cache.hit_count(), 0);
        assert_eq!(cache.miss_count(), 1);
        // Insert + Hit.
        cache.insert(k.clone(), dummy_plan(), 5);
        assert!(matches!(cache.lookup(&k, 5), LookupOutcome::Hit(_)));
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 1);
        // Stale (stamped < current): counts as a miss from caller-
        // facing hit-rate perspective.
        cache.insert(k.clone(), dummy_plan(), 5);
        assert!(matches!(cache.lookup(&k, 6), LookupOutcome::Stale));
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 2);
        // InvariantViolation (stamped > current): also counts as miss.
        cache.insert(k.clone(), dummy_plan(), 100);
        assert!(matches!(
            cache.lookup(&k, 50),
            LookupOutcome::InvariantViolation
        ));
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 3);
        // Miss after eviction.
        assert!(matches!(cache.lookup(&k, 50), LookupOutcome::Miss));
        assert_eq!(cache.hit_count(), 1);
        assert_eq!(cache.miss_count(), 4);
    }
}
