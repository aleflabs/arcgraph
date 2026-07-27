//! W25-OPS-PROD — per-tenant cost-telemetry accumulator.
//!
//! # Scope
//!
//! Runtime-observed cost telemetry, distinct from the planner's
//! pre-execution cost estimates at
//! `crates/arcgraph-query/src/planner/cost/`. The planner cost model
//! produces unitless ordering-only scores for "pick the cheaper of two
//! plans"; this module accumulates physical-unit observations of what
//! a tenant has actually consumed (so operators can bill, throttle,
//! diagnose hot tenants).
//!
//! Four cost dimensions per ADR-093-amendment-01 §D-6:
//!
//! | Field | Unit | Semantics |
//! |---|---|---|
//! | `cpu_ms` | milliseconds | Wall-clock CPU time consumed (sum across all worker tasks). |
//! | `mem_mb_peak` | mebibytes | Peak RSS attributable to the tenant during the snapshot window. |
//! | `bytes_read` | bytes | Total bytes read from storage (page cache + cold reads). |
//! | `bytes_written` | bytes | Total bytes written to WAL + page store. |
//!
//! # Concurrency
//!
//! Counters are `AtomicU64` so any worker thread can update them
//! without a lock. `mem_mb_peak` uses a CAS loop because peak-tracking
//! needs `fetch_max` semantics (stable across both glibc + musl
//! since `AtomicU64::fetch_max` is in the Rust std since 1.45).
//!
//! # Cost-attribution boundary
//!
//! Attribution is at the tenant granularity, NOT per-query — per-query
//! cost surfaces through the planner's `CostedPlan` (estimates) +
//! the M4-71 observer's actual row-counts (runtime). This module
//! aggregates the per-query observations into per-tenant totals.
//!
//! # Snapshot semantics
//!
//! [`CostAccumulator::snapshot`] reads each counter independently —
//! the four reads are NOT atomic together. A snapshot is therefore a
//! per-field consistent read but not a globally-consistent one. The
//! operator-facing semantics: numbers MAY be off by the in-flight
//! query's contribution at the read instant; an idle tenant's
//! snapshot is exact. This trade is intentional — the alternative
//! (`RwLock<CostSnapshot>`) would serialize every counter update
//! through the same lock and starve high-RPS tenants.
//!
//! # Reset
//!
//! Counters are monotonic across the registry's lifetime. Operators
//! who need windowed totals (e.g., per-hour billing) snapshot at the
//! window boundaries and subtract; the registry does NOT offer a
//! reset/zero verb because that would race with mid-query updates
//! (the in-flight query would record its accumulated cost against
//! the reset zero, double-counting). v1.1+ may add a `snapshot_and_reset`
//! verb if a use case emerges that justifies the contention.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::ids::TenantId;

/// Per-tenant cost accumulator. Cheaply cloneable via `Arc`; any
/// worker thread can hold a clone + record observations without a
/// lock.
///
/// Use [`PerTenantCostRegistry::get_or_init`] to obtain one; do not
/// construct directly outside the registry (per-tenant uniqueness is
/// the registry's invariant).
#[derive(Debug)]
pub struct CostAccumulator {
    cpu_ms: AtomicU64,
    mem_mb_peak: AtomicU64,
    bytes_read: AtomicU64,
    bytes_written: AtomicU64,
}

impl CostAccumulator {
    /// Internal: registry constructs these. External callers obtain
    /// via the registry's `get_or_init`.
    fn new() -> Self {
        Self {
            cpu_ms: AtomicU64::new(0),
            mem_mb_peak: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
        }
    }

    /// Add `delta` milliseconds of CPU time. Saturating: caller's
    /// individual contribution capped at `u64::MAX` (a single tenant
    /// would need ~584 million years of CPU time to overflow, so this
    /// is a theoretical cap — but `wrapping_add` would silently zero
    /// on overflow which would be very confusing).
    pub fn record_cpu_ms(&self, delta: u64) {
        // saturating_add semantics via fetch_add + max-clamp would
        // require CAS; instead use saturating arithmetic on the
        // pre-load, then store with Release ordering. The pre-load
        // can race with another thread's update — that's OK because
        // the worst case is one tenant's reading vs. another's
        // simultaneous record_cpu_ms saturation by 1 unit.
        let prev = self.cpu_ms.load(Ordering::Acquire);
        if prev.saturating_add(delta) == u64::MAX {
            self.cpu_ms.store(u64::MAX, Ordering::Release);
        } else {
            self.cpu_ms.fetch_add(delta, Ordering::Relaxed);
        }
    }

    /// Observe a tenant's current RSS in mebibytes. Maintains the
    /// peak via `fetch_max`. Idempotent if the same value is observed
    /// repeatedly (which is the common case for a polling sampler).
    pub fn observe_mem_mb(&self, current_mb: u64) {
        self.mem_mb_peak.fetch_max(current_mb, Ordering::Relaxed);
    }

    /// Add `delta` bytes read from storage.
    pub fn record_bytes_read(&self, delta: u64) {
        self.bytes_read.fetch_add(delta, Ordering::Relaxed);
    }

    /// Add `delta` bytes written (WAL + page store combined).
    pub fn record_bytes_written(&self, delta: u64) {
        self.bytes_written.fetch_add(delta, Ordering::Relaxed);
    }

    /// Take a snapshot of the four cost dimensions. Per-field
    /// consistent but not globally consistent — see module doc.
    #[must_use]
    pub fn snapshot(&self) -> CostSnapshot {
        CostSnapshot {
            cpu_ms: self.cpu_ms.load(Ordering::Acquire),
            mem_mb_peak: self.mem_mb_peak.load(Ordering::Acquire),
            bytes_read: self.bytes_read.load(Ordering::Acquire),
            bytes_written: self.bytes_written.load(Ordering::Acquire),
        }
    }
}

/// Snapshot of a tenant's accumulated cost across four dimensions.
///
/// Serializable for the admin HTTP `/cost/{tenant}` endpoint + the
/// CLI `arcgraph cost summary` subcommand. The serialized wire shape
/// is part of the v1.0-GA operator surface — do NOT change field
/// names without an amendment to ADR-093.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostSnapshot {
    /// Total CPU milliseconds consumed across all worker tasks.
    pub cpu_ms: u64,
    /// Peak RSS in mebibytes observed since accumulator construction.
    pub mem_mb_peak: u64,
    /// Total bytes read from storage.
    pub bytes_read: u64,
    /// Total bytes written (WAL + page store).
    pub bytes_written: u64,
}

impl CostSnapshot {
    /// Element-wise difference (self - earlier). Useful for windowed
    /// billing: take two snapshots an hour apart, subtract. Underflow
    /// saturates at zero (defensive — a later snapshot SHOULD have
    /// monotonically larger counters; underflow means clock-skew or
    /// process restart between the two reads).
    #[must_use]
    pub fn diff(&self, earlier: CostSnapshot) -> CostSnapshot {
        CostSnapshot {
            cpu_ms: self.cpu_ms.saturating_sub(earlier.cpu_ms),
            mem_mb_peak: self.mem_mb_peak.saturating_sub(earlier.mem_mb_peak),
            bytes_read: self.bytes_read.saturating_sub(earlier.bytes_read),
            bytes_written: self.bytes_written.saturating_sub(earlier.bytes_written),
        }
    }
}

/// Process-global per-tenant cost registry.
///
/// One accumulator per tenant; lazily-created on first
/// `get_or_init`. The registry itself is `Clone` (cheap; backed by
/// `Arc`) so different bounded contexts can hold their own handle to
/// the same shared state.
///
/// # Memory shape
///
/// One `Arc<CostAccumulator>` per tenant (32 bytes) + the HashMap
/// overhead. At 10 000 tenants the registry is ~400 KB, well below
/// any deployment-relevant threshold.
#[derive(Debug, Clone, Default)]
pub struct PerTenantCostRegistry {
    inner: Arc<RwLock<HashMap<TenantId, Arc<CostAccumulator>>>>,
}

impl PerTenantCostRegistry {
    /// Construct an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Return the accumulator for `tenant`, creating it on first
    /// access. Thread-safe; concurrent first-access from multiple
    /// threads creates only one accumulator (the loser of the race
    /// drops its candidate).
    pub fn get_or_init(&self, tenant: TenantId) -> Arc<CostAccumulator> {
        if let Some(existing) = self.inner.read().get(&tenant) {
            return Arc::clone(existing);
        }
        let mut guard = self.inner.write();
        // Double-check after acquiring the write lock — another thread
        // may have inserted between our read-release and write-acquire.
        if let Some(existing) = guard.get(&tenant) {
            return Arc::clone(existing);
        }
        let acc = Arc::new(CostAccumulator::new());
        guard.insert(tenant, Arc::clone(&acc));
        acc
    }

    /// Snapshot a single tenant. Returns `None` if the tenant has no
    /// recorded activity (no accumulator was ever initialized).
    #[must_use]
    pub fn snapshot(&self, tenant: TenantId) -> Option<CostSnapshot> {
        self.inner.read().get(&tenant).map(|acc| acc.snapshot())
    }

    /// Snapshot all known tenants. The returned map is a point-in-time
    /// copy; the registry's internal state may have advanced by the
    /// time the caller iterates it.
    #[must_use]
    pub fn snapshot_all(&self) -> HashMap<TenantId, CostSnapshot> {
        self.inner
            .read()
            .iter()
            .map(|(tenant, acc)| (*tenant, acc.snapshot()))
            .collect()
    }

    /// Number of tenants with at least one observation. Useful for
    /// the admin HTTP `/cost` summary index + Prometheus metric label
    /// cardinality monitoring.
    #[must_use]
    pub fn tenant_count(&self) -> usize {
        self.inner.read().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn empty_accumulator_snapshot_is_zero() {
        let acc = CostAccumulator::new();
        let snap = acc.snapshot();
        assert_eq!(
            snap,
            CostSnapshot {
                cpu_ms: 0,
                mem_mb_peak: 0,
                bytes_read: 0,
                bytes_written: 0,
            }
        );
    }

    #[test]
    fn record_cpu_ms_accumulates() {
        let acc = CostAccumulator::new();
        acc.record_cpu_ms(100);
        acc.record_cpu_ms(250);
        assert_eq!(acc.snapshot().cpu_ms, 350);
    }

    #[test]
    fn observe_mem_mb_tracks_peak_not_last() {
        let acc = CostAccumulator::new();
        acc.observe_mem_mb(64);
        acc.observe_mem_mb(128);
        acc.observe_mem_mb(96);
        // Peak should be 128, not 96 (last value).
        assert_eq!(acc.snapshot().mem_mb_peak, 128);
    }

    #[test]
    fn record_bytes_read_and_written_track_separately() {
        let acc = CostAccumulator::new();
        acc.record_bytes_read(1024);
        acc.record_bytes_written(2048);
        let snap = acc.snapshot();
        assert_eq!(snap.bytes_read, 1024);
        assert_eq!(snap.bytes_written, 2048);
    }

    #[test]
    fn record_cpu_ms_saturates_at_max_not_wraps() {
        let acc = CostAccumulator::new();
        acc.cpu_ms.store(u64::MAX - 10, Ordering::Release);
        acc.record_cpu_ms(100);
        assert_eq!(acc.snapshot().cpu_ms, u64::MAX);
    }

    #[test]
    fn registry_get_or_init_is_idempotent_per_tenant() {
        let registry = PerTenantCostRegistry::new();
        let t1 = TenantId::new(1);
        let acc_a = registry.get_or_init(t1);
        let acc_b = registry.get_or_init(t1);
        // Same Arc target — both pointers refer to the same accumulator.
        assert!(Arc::ptr_eq(&acc_a, &acc_b));
        // Updating via one is visible via the other.
        acc_a.record_cpu_ms(42);
        assert_eq!(acc_b.snapshot().cpu_ms, 42);
    }

    #[test]
    fn registry_snapshot_missing_tenant_returns_none() {
        let registry = PerTenantCostRegistry::new();
        assert!(registry.snapshot(TenantId::new(999)).is_none());
    }

    #[test]
    fn registry_snapshot_all_returns_every_tenant() {
        let registry = PerTenantCostRegistry::new();
        let t1 = TenantId::new(1);
        let t2 = TenantId::new(2);
        registry.get_or_init(t1).record_cpu_ms(100);
        registry.get_or_init(t2).record_cpu_ms(200);
        let snaps = registry.snapshot_all();
        assert_eq!(snaps.len(), 2);
        assert_eq!(snaps[&t1].cpu_ms, 100);
        assert_eq!(snaps[&t2].cpu_ms, 200);
    }

    #[test]
    fn cost_snapshot_diff_is_element_wise_saturating() {
        let later = CostSnapshot {
            cpu_ms: 1000,
            mem_mb_peak: 512,
            bytes_read: 4096,
            bytes_written: 8192,
        };
        let earlier = CostSnapshot {
            cpu_ms: 250,
            mem_mb_peak: 256,
            bytes_read: 1024,
            bytes_written: 2048,
        };
        let delta = later.diff(earlier);
        assert_eq!(
            delta,
            CostSnapshot {
                cpu_ms: 750,
                mem_mb_peak: 256,
                bytes_read: 3072,
                bytes_written: 6144,
            }
        );
    }

    #[test]
    fn cost_snapshot_diff_saturates_at_zero_on_underflow() {
        let later = CostSnapshot {
            cpu_ms: 100,
            mem_mb_peak: 0,
            bytes_read: 0,
            bytes_written: 0,
        };
        let earlier = CostSnapshot {
            cpu_ms: 500,
            mem_mb_peak: 0,
            bytes_read: 0,
            bytes_written: 0,
        };
        let delta = later.diff(earlier);
        assert_eq!(delta.cpu_ms, 0);
    }

    #[test]
    fn concurrent_updates_do_not_lose_observations() {
        let registry = PerTenantCostRegistry::new();
        let tenant = TenantId::new(7);
        let acc = registry.get_or_init(tenant);

        let threads: Vec<_> = (0..8)
            .map(|_| {
                let acc = Arc::clone(&acc);
                thread::spawn(move || {
                    for _ in 0..1000 {
                        acc.record_cpu_ms(1);
                        acc.record_bytes_read(2);
                        acc.record_bytes_written(3);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }

        let snap = acc.snapshot();
        assert_eq!(snap.cpu_ms, 8 * 1000);
        assert_eq!(snap.bytes_read, 8 * 1000 * 2);
        assert_eq!(snap.bytes_written, 8 * 1000 * 3);
    }

    #[test]
    fn concurrent_get_or_init_returns_same_accumulator() {
        let registry = PerTenantCostRegistry::new();
        let tenant = TenantId::new(42);

        let threads: Vec<_> = (0..16)
            .map(|_| {
                let registry = registry.clone();
                thread::spawn(move || registry.get_or_init(tenant))
            })
            .collect();
        let accs: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();

        let first = &accs[0];
        for other in &accs[1..] {
            assert!(Arc::ptr_eq(first, other));
        }
        assert_eq!(registry.tenant_count(), 1);
    }

    #[test]
    fn registry_clone_shares_same_state() {
        let registry_a = PerTenantCostRegistry::new();
        let registry_b = registry_a.clone();
        let tenant = TenantId::new(1);
        registry_a.get_or_init(tenant).record_cpu_ms(123);
        // The clone sees the same accumulator.
        let snap = registry_b.snapshot(tenant).expect("tenant exists");
        assert_eq!(snap.cpu_ms, 123);
    }

    #[test]
    fn observe_mem_mb_zero_does_not_clobber_existing_peak() {
        let acc = CostAccumulator::new();
        acc.observe_mem_mb(100);
        acc.observe_mem_mb(0);
        assert_eq!(acc.snapshot().mem_mb_peak, 100);
    }
}
