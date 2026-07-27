//! M4-64a per-tenant memory budget enforcement (correctness primitive).
//!
//! Tracks per-tenant byte allocation across all operators participating
//! in a single query. Replaces the W11Z #272 `SPILLOVER_MAX_ROWS` row-
//! count heuristic with a proper memory-byte budget per amendment-03
//! §Structural-1.
//!
//! # Why a correctness primitive (not a perf optimization)
//!
//! Per amendment-03 §Structural-1, M4-64a is the **correctness floor**
//! — if a multi-tenant deployment leaks memory, it OOMs silently under
//! production load. This module's job is to make that fault impossible
//! to silently induce: if a budget is configured, ANY operator-level
//! reservation that would cross the cap surfaces
//! [`ArcQLError::ResourceExhausted`] BEFORE the allocation lands.
//!
//! # Surface
//!
//! - [`MemoryBudget`] — `Arc`-backed per-tenant byte tracker. Cheap to
//!   clone (cloning shares the inner state); `Send + Sync` so the same
//!   budget can be threaded through every operator in a parallel
//!   pipeline (forward-pin for M4-64b SIMD work-stealing).
//! - [`MemoryReservation`] — RAII guard releasing on drop. Used by
//!   tests + single-shot allocations.
//! - [`MemoryBudget::try_reserve_unscoped`] /
//!   [`MemoryBudget::release`] — manual API for operators with
//!   spillover queues that store the byte cost alongside the row data.
//!
//! # Forward-binding
//!
//! M5-12 rate-limit config consumes
//! [`MemoryBudget::set_per_tenant_cap`] at server-startup time per
//! amendment-03 §TIER-2-a (parallels the M4-53 plan-cache
//! `set_max_entries` forward-method shape). At v1.0-alpha the
//! per-tenant cap is `None` (unbounded) by default and the row-count
//! fallback ([`BUDGET_FALLBACK_ROWS`]) applies; M5-12 will flip the
//! default to a configured byte cap once the rate-limit config surface
//! lights.
//!
//! # Factorized intermediate forward-pin (amendment-03 §Structural-1)
//!
//! v1.0-alpha tracks bytes via the row-tuple [`crate::executor::Batch`]
//! shape. The factorized intermediate (vector / column-major) re-shape
//! is M4-64b / v1.1+ scope; this module's per-tenant byte counter is
//! the surface that the future column-major code path will reuse — the
//! API is shape-agnostic, only the [`estimate_value_bytes`] /
//! [`estimate_row_bytes`] estimators change.
//!
//! # Concurrency
//!
//! Per-tenant counters live behind a single `Mutex<HashMap<...>>`
//! guard. The lock is held briefly: a `try_reserve` is `lookup +
//! integer add + integer compare + write-back`. Contention is bounded
//! to within a single query's operator pipeline (currently sequential
//! at v1.0-alpha; a parallel executor at M4-64b+ will see ~N-operator
//! contention on the same per-tenant slot). A SeqLock-style consistent
//! read pattern is NOT warranted at v1.0-alpha: there's no
//! panic-recovery story for the budget counter (a panic mid-reservation
//! drops the [`MemoryReservation`] guard, which releases its bytes via
//! Drop — the bookkeeping naturally heals). Per
//! `feedback_seqlock_panic_safety_primitive.md`, the SeqLock primitive
//! is for `commits_started == commits_observed` markers under
//! per-tenant fault isolation; that pattern doesn't apply here.
//!
//! # Forward-pin: contention characterization (M4-64b)
//!
//! Contention behavior is covered by the concurrent reservation tests
//! in this module.
//!
//! # ADR provenance
//!
//! - **ADR-038 amendment-02 §M4.f** — primary M4-64a slice cite.
//! - **ADR-038 amendment-03 §Structural-1** — split out from M4-64
//!   bundled SIMD; correctness primitive justification.
//! - **ADR-038 amendment-03 §TIER-2-a** — M5-12 `set_per_tenant_cap`
//!   forward-binding pattern (mirrors M4-53 plan-cache).
//! - **W11Z #272 retro MED-3** — the conservative `SPILLOVER_MAX_ROWS`
//!   row-count cap this module supersedes (and gracefully retains as
//!   `BUDGET_FALLBACK_ROWS` fallback for unconfigured tenants).

use std::collections::HashMap;
use std::sync::Arc;

use arcgraph_core::TenantId;
use parking_lot::Mutex;

use crate::error::Span;
use crate::executor::error::ExecutionError;
use crate::executor::value::Value;
use crate::semantic::error::ArcQLError;

/// Default fallback row cap when no per-tenant byte budget is configured.
///
/// True alias for [`crate::executor::ops::expand::SPILLOVER_MAX_ROWS`]
/// (the W11Z #272 row cap). Aliasing — rather than redefining `64 *
/// BATCH_ROWS` — guarantees a single source of truth: a future tune of
/// `SPILLOVER_MAX_BATCHES` flows here automatically (W12α fix-up LOW-2
/// per PR #277 retro). The unit test
/// `tests::budget_fallback_rows_equals_spillover_max_rows` pins the
/// equality.
///
/// # Semantics
///
/// - When [`MemoryBudget::has_cap`] returns `true` for the tenant, the
///   byte cap takes precedence and this row count is informational
///   only.
/// - When [`MemoryBudget::has_cap`] returns `false`, operators with
///   spillover queues SHOULD enforce this row cap directly (the
///   v1.0-alpha test surface relies on the row cap when no byte cap is
///   set).
pub const BUDGET_FALLBACK_ROWS: usize = crate::executor::ops::expand::SPILLOVER_MAX_ROWS;

/// Per-tenant memory budget tracker.
///
/// Cloning shares the inner state via `Arc` — every clone observes the
/// same per-tenant counters. Construct one budget per query (typically
/// owned by the [`crate::executor::ExecutionContext`]) and clone into
/// every operator that needs to track allocations.
#[derive(Debug, Clone, Default)]
pub struct MemoryBudget {
    inner: Arc<MemoryBudgetInner>,
}

#[derive(Debug, Default)]
struct MemoryBudgetInner {
    /// Per-tenant state. The `Mutex` guards both the reads (current /
    /// peak) and writes (cap change, reserve, release). Lock-free
    /// fast-path is M4-64b+ scope.
    state: Mutex<HashMap<TenantId, BudgetState>>,
}

/// Per-tenant accounting state.
#[derive(Debug, Default, Clone)]
struct BudgetState {
    /// Configured byte cap. `None` = unbounded; the row-count fallback
    /// applies in the operator layer.
    cap_bytes: Option<u64>,
    /// Currently-reserved bytes (sum of all live reservations).
    current_bytes: u64,
    /// Peak observed bytes (high-water mark for forward M4-71 /
    /// M4-91 PROFILE consumption per amendment-03 §Structural-3 edge 6).
    peak_bytes: u64,
}

impl MemoryBudget {
    /// Construct an unbounded budget. v1.0-alpha default. M5-12 will
    /// override per-tenant caps via [`Self::set_per_tenant_cap`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a budget with `cap_bytes` configured for `tenant`.
    /// Convenience wrapper around `new() + set_per_tenant_cap()`.
    #[must_use]
    pub fn with_per_tenant_cap(tenant: TenantId, cap_bytes: u64) -> Self {
        let b = Self::new();
        b.set_per_tenant_cap(tenant, cap_bytes);
        b
    }

    /// Configure / re-configure the per-tenant byte cap.
    ///
    /// # M5-12 forward-binding
    ///
    /// M5-12 rate-limit config consumes this method at server-startup
    /// time. Mirrors the [`crate::planner::PlanCache::set_max_entries`]
    /// forward-method shape per amendment-03 §TIER-2-a.
    ///
    /// # Concurrency
    ///
    /// Idempotent and safe to call concurrently with reservations.
    /// Tightening the cap below the current reserved total does NOT
    /// retroactively reject in-flight reservations (they keep their
    /// bytes); only NEW reservations see the tightened cap.
    pub fn set_per_tenant_cap(&self, tenant: TenantId, cap_bytes: u64) {
        let mut state = self.inner.state.lock();
        let entry = state.entry(tenant).or_default();
        entry.cap_bytes = Some(cap_bytes);
    }

    /// Read the configured cap for `tenant`. Returns `None` if no cap
    /// is configured (unbounded — fallback row-count applies in
    /// operators).
    #[must_use]
    pub fn cap_bytes(&self, tenant: TenantId) -> Option<u64> {
        self.inner
            .state
            .lock()
            .get(&tenant)
            .and_then(|e| e.cap_bytes)
    }

    /// `true` iff a per-tenant cap is configured for `tenant`.
    /// Operators check this before falling back to the
    /// [`BUDGET_FALLBACK_ROWS`] row-count cap.
    #[inline]
    #[must_use]
    pub fn has_cap(&self, tenant: TenantId) -> bool {
        self.cap_bytes(tenant).is_some()
    }

    /// Read the currently-reserved byte total for `tenant`.
    ///
    /// Used by tests + future M4-91 PROFILE rendering. Returns 0 if
    /// the tenant has not yet appeared in any reservation.
    #[must_use]
    pub fn current_bytes(&self, tenant: TenantId) -> u64 {
        self.inner
            .state
            .lock()
            .get(&tenant)
            .map(|e| e.current_bytes)
            .unwrap_or(0)
    }

    /// Read the peak (high-water-mark) byte total for `tenant`.
    ///
    /// Used by future M4-71 row-count observer + M4-91 PROFILE
    /// per-operator memory annotation per amendment-03 §Structural-3
    /// edge 6.
    #[must_use]
    pub fn peak_bytes(&self, tenant: TenantId) -> u64 {
        self.inner
            .state
            .lock()
            .get(&tenant)
            .map(|e| e.peak_bytes)
            .unwrap_or(0)
    }

    /// Try to reserve `bytes` for `tenant`, returning a RAII guard.
    ///
    /// On drop the guard calls [`Self::release`], decrementing the
    /// per-tenant counter. Useful for tests + single-shot allocations
    /// where the lifetime of the reservation matches a Rust scope.
    ///
    /// Operators with spillover queues (where the lifetime crosses
    /// `next_batch` calls) prefer [`Self::try_reserve_unscoped`].
    pub fn try_reserve(
        &self,
        tenant: TenantId,
        bytes: u64,
        feature: &'static str,
    ) -> Result<MemoryReservation, ExecutionError> {
        self.try_reserve_unscoped(tenant, bytes, feature)?;
        Ok(MemoryReservation {
            budget: self.clone(),
            tenant,
            bytes,
        })
    }

    /// Try to reserve `bytes` for `tenant` WITHOUT a RAII guard.
    ///
    /// On success, the byte counter is bumped; the caller MUST call
    /// [`Self::release`] when the bytes are no longer in use (typically
    /// when an operator pops a row from its spillover queue and emits
    /// it).
    ///
    /// On failure (would push past `cap_bytes`), returns
    /// [`ExecutionError::Plan`] wrapping
    /// [`ArcQLError::ResourceExhausted`] with the byte numbers and the
    /// `feature` label naming the surface that triggered exhaustion.
    pub fn try_reserve_unscoped(
        &self,
        tenant: TenantId,
        bytes: u64,
        feature: &'static str,
    ) -> Result<(), ExecutionError> {
        let mut state = self.inner.state.lock();
        let entry = state.entry(tenant).or_default();
        let projected = entry.current_bytes.saturating_add(bytes);
        if let Some(cap) = entry.cap_bytes {
            if projected > cap {
                Err(ExecutionError::Plan(ArcQLError::ResourceExhausted {
                    feature: feature.to_owned(),
                    requested_bytes: bytes,
                    cap_bytes: cap,
                    projected_bytes: projected,
                    span: Span::point(0, 0),
                }))
            } else {
                entry.current_bytes = projected;
                entry.peak_bytes = entry.peak_bytes.max(projected);
                Ok(())
            }
        } else {
            entry.current_bytes = projected;
            entry.peak_bytes = entry.peak_bytes.max(projected);
            Ok(())
        }
    }

    /// Release `bytes` previously reserved via
    /// [`Self::try_reserve_unscoped`].
    ///
    /// Decrement is saturating: a release of more bytes than currently
    /// tracked clamps to zero rather than underflowing (defensive — a
    /// double-release bug at the operator layer surfaces as a stale
    /// counter, not a panic).
    pub fn release(&self, tenant: TenantId, bytes: u64) {
        let mut state = self.inner.state.lock();
        if let Some(entry) = state.get_mut(&tenant) {
            entry.current_bytes = entry.current_bytes.saturating_sub(bytes);
        }
    }
}

/// RAII guard releasing the reserved bytes on drop.
#[derive(Debug)]
pub struct MemoryReservation {
    budget: MemoryBudget,
    tenant: TenantId,
    bytes: u64,
}

impl MemoryReservation {
    /// Number of bytes this reservation holds.
    #[inline]
    #[must_use]
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Tenant this reservation is scoped to.
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.tenant
    }
}

impl Drop for MemoryReservation {
    fn drop(&mut self) {
        self.budget.release(self.tenant, self.bytes);
    }
}

/// Estimate the byte cost of a single [`Value`] cell.
///
/// Conservative upper-bound — counts heap allocations (string bytes,
/// list / map element costs) plus the stack-side variant size. Used by
/// [`estimate_row_bytes`] to size operator spillover allocations.
///
/// # Why "conservative upper-bound"?
///
/// The budget primitive's job is to prevent OOMs. A small overestimate
/// rejects a few queries that would have fit; a small underestimate
/// admits queries that OOM under load. We pick the safer side.
///
/// # Forward-pin for the factorized intermediate
///
/// The factorized (column-major) re-shape at M4-64b will replace this
/// with per-column estimators reading from the column type's storage
/// layout. The function's call-site signature is shape-agnostic.
#[must_use]
pub fn estimate_value_bytes(v: &Value) -> usize {
    use std::mem::size_of;
    let stack = size_of::<Value>();
    let heap = match v {
        Value::Null | Value::Boolean(_) | Value::Integer(_) | Value::Float(_) => 0,
        Value::String(s) => s.capacity(),
        Value::Node(n) => {
            // BTreeMap node overhead is ~48 bytes per entry on 64-bit
            // platforms (per `std::collections::BTreeMap` source); the
            // estimate is intentionally generous.
            n.properties
                .iter()
                .map(|(k, v)| k.capacity() + 48 + estimate_value_bytes(v))
                .sum()
        }
        Value::Relationship(r) => r
            .properties
            .iter()
            .map(|(k, v)| k.capacity() + 48 + estimate_value_bytes(v))
            .sum(),
        Value::List(elems) => elems.iter().map(estimate_value_bytes).sum::<usize>(),
        // ADR-191 D-13 — a map's heap cost RECURSES into each key + value
        // (mirroring the Node/Rel property-bag arms). NOT `=> 0`: a
        // non-recursive stub would under-count nested maps and defeat the
        // COLLECT-fold + output-row memory reservation the per-tenant
        // byte cap protects.
        Value::Map(m) => m
            .iter()
            .map(|(k, v)| k.capacity() + 48 + estimate_value_bytes(v))
            .sum(),
        // ADR-193 D-9 — a path's heap cost is its start node's property
        // bag plus, per segment, the relationship's + end node's property
        // bags (the `NodeView`/`RelView` accounting reused via wrapping
        // in the matching `Value` variant). The `segments` Vec's own
        // backing allocation is `segments.len() * size_of::<PathSegment>`.
        Value::Path(p) => {
            let segs_backing =
                p.segments.len() * std::mem::size_of::<crate::executor::value::PathSegment>();
            let nodes_rels: usize = estimate_value_bytes(&Value::Node(p.start.clone()))
                + p.segments
                    .iter()
                    .map(|s| {
                        estimate_value_bytes(&Value::Relationship(s.rel.clone()))
                            + estimate_value_bytes(&Value::Node(s.end.clone()))
                    })
                    .sum::<usize>();
            segs_backing + nodes_rels
        }
        // W23-V11-T-01 / ADR-090 — temporal + decimal cells are all
        // POD-sized (i64 + i32 / structured tuples / i128); no heap.
        Value::Temporal(_)
        | Value::LocalDateTime(_)
        | Value::Date(_)
        | Value::Duration(_)
        | Value::Decimal(_) => 0,
    };
    stack + heap
}

/// Estimate the byte cost of a row (= sum of cell costs + Vec overhead).
///
/// Vec overhead = 24 bytes (ptr + len + cap on 64-bit) — the same
/// stack-side cost a `Vec<Value>` carries regardless of element count.
#[must_use]
pub fn estimate_row_bytes(row: &[Value]) -> usize {
    24 + row.iter().map(estimate_value_bytes).sum::<usize>()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t1() -> TenantId {
        TenantId::DEFAULT
    }
    fn t2() -> TenantId {
        TenantId::new(42)
    }

    // -------------------------------------------------------------
    // Allocator accounting (8 unit tests per amendment-03 row)
    // -------------------------------------------------------------

    #[test]
    fn unit_1_default_budget_is_unbounded_and_zero() {
        // Default budget has no cap and zero observed bytes.
        let b = MemoryBudget::new();
        assert!(!b.has_cap(t1()));
        assert_eq!(b.cap_bytes(t1()), None);
        assert_eq!(b.current_bytes(t1()), 0);
        assert_eq!(b.peak_bytes(t1()), 0);
    }

    #[test]
    fn unit_2_set_per_tenant_cap_lights_has_cap() {
        // M5-12 forward-method: set_per_tenant_cap configures the cap;
        // has_cap reflects the configuration immediately.
        let b = MemoryBudget::new();
        b.set_per_tenant_cap(t1(), 1024);
        assert!(b.has_cap(t1()));
        assert_eq!(b.cap_bytes(t1()), Some(1024));
    }

    #[test]
    fn unit_3_reserve_within_cap_succeeds_and_tracks() {
        // try_reserve_unscoped within cap succeeds; current + peak
        // both reflect the reserved bytes.
        let b = MemoryBudget::with_per_tenant_cap(t1(), 1000);
        b.try_reserve_unscoped(t1(), 500, "test").unwrap();
        assert_eq!(b.current_bytes(t1()), 500);
        assert_eq!(b.peak_bytes(t1()), 500);
    }

    #[test]
    fn unit_4_reserve_exceeding_cap_returns_resource_exhausted() {
        // try_reserve_unscoped past cap returns
        // ArcQLError::ResourceExhausted; bytes are NOT reserved.
        let b = MemoryBudget::with_per_tenant_cap(t1(), 1000);
        let err = b.try_reserve_unscoped(t1(), 1500, "test").unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })
        ));
        // Counter NOT bumped on rejection.
        assert_eq!(b.current_bytes(t1()), 0);
    }

    #[test]
    fn unit_5_release_decrements_current_not_peak() {
        // release reduces current bytes; peak high-water-mark is
        // preserved (M4-71 / M4-91 PROFILE consumer pin).
        let b = MemoryBudget::with_per_tenant_cap(t1(), 1000);
        b.try_reserve_unscoped(t1(), 800, "test").unwrap();
        assert_eq!(b.peak_bytes(t1()), 800);
        b.release(t1(), 800);
        assert_eq!(b.current_bytes(t1()), 0);
        // Peak preserved.
        assert_eq!(b.peak_bytes(t1()), 800);
    }

    #[test]
    fn unit_6_raii_guard_releases_on_drop() {
        // try_reserve returns a guard that releases on drop.
        let b = MemoryBudget::with_per_tenant_cap(t1(), 1000);
        {
            let _g = b.try_reserve(t1(), 500, "test").unwrap();
            assert_eq!(b.current_bytes(t1()), 500);
        }
        // After the scope, the guard dropped and bytes were released.
        assert_eq!(b.current_bytes(t1()), 0);
        // Peak preserved.
        assert_eq!(b.peak_bytes(t1()), 500);
    }

    #[test]
    fn unit_7_factorized_intermediate_shape_forward_pin() {
        // amendment-03 Structural-1 forward-pin: estimate_row_bytes
        // returns a non-zero, monotone byte estimate for the row-tuple
        // shape. The factorized re-shape at M4-64b will swap the
        // estimator without changing the API surface.
        let row1: Vec<Value> = vec![Value::Integer(42)];
        let row2: Vec<Value> = vec![Value::Integer(42), Value::String("hello".into())];
        let row3: Vec<Value> = vec![
            Value::Integer(42),
            Value::String("hello".into()),
            Value::Float(3.5),
        ];
        // Monotone: more cells → more bytes.
        assert!(estimate_row_bytes(&row1) < estimate_row_bytes(&row2));
        assert!(estimate_row_bytes(&row2) < estimate_row_bytes(&row3));
        // Non-zero baseline (Vec overhead + variant stack size).
        assert!(estimate_row_bytes(&[]) >= 24);
    }

    #[test]
    fn unit_8_budget_enforcement_boundary_at_exact_cap() {
        // Reservation that fills the cap EXACTLY succeeds; the next
        // byte beyond rejects. Boundary behavior pinned per
        // amendment-03 Structural-1 "per-batch enforcement boundary".
        let b = MemoryBudget::with_per_tenant_cap(t1(), 1000);
        b.try_reserve_unscoped(t1(), 1000, "test").unwrap();
        assert_eq!(b.current_bytes(t1()), 1000);
        // One more byte trips the cap.
        let err = b.try_reserve_unscoped(t1(), 1, "test").unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })
        ));
    }

    // -------------------------------------------------------------
    // Multi-tenant isolation
    // -------------------------------------------------------------

    #[test]
    fn budget_isolates_tenants() {
        // Two tenants. Configuring tenant 1 does NOT affect tenant 2's
        // accounting; reserving bytes for tenant 1 does NOT debit
        // tenant 2's counter.
        let b = MemoryBudget::new();
        b.set_per_tenant_cap(t1(), 1000);
        b.set_per_tenant_cap(t2(), 500);
        b.try_reserve_unscoped(t1(), 800, "test").unwrap();
        assert_eq!(b.current_bytes(t1()), 800);
        assert_eq!(b.current_bytes(t2()), 0);
        // Tenant 2's cap is unrelated to tenant 1's bytes; reserving
        // 400 for tenant 2 succeeds.
        b.try_reserve_unscoped(t2(), 400, "test").unwrap();
        assert_eq!(b.current_bytes(t2()), 400);
        // Tenant 2 hits its cap independently.
        let err = b.try_reserve_unscoped(t2(), 200, "test").unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })
        ));
    }

    #[test]
    fn unbounded_tenant_admits_any_reservation_size() {
        // No per-tenant cap configured = unbounded; reservation at
        // arbitrary size succeeds. Operators rely on the row-count
        // fallback (BUDGET_FALLBACK_ROWS) when no byte cap is set.
        let b = MemoryBudget::new();
        b.try_reserve_unscoped(t1(), u64::MAX / 2, "test").unwrap();
        assert!(!b.has_cap(t1()));
    }

    #[test]
    fn cap_tightening_does_not_retroactively_reject_inflight() {
        // set_per_tenant_cap can be called after reservations exist;
        // tightening below current does NOT retroactively kick out
        // existing bytes (they hold). NEW reservations see the
        // tightened cap.
        let b = MemoryBudget::with_per_tenant_cap(t1(), 2000);
        b.try_reserve_unscoped(t1(), 1500, "test").unwrap();
        assert_eq!(b.current_bytes(t1()), 1500);
        // Tighten cap to 1000 (below current 1500).
        b.set_per_tenant_cap(t1(), 1000);
        // Existing 1500 still tracked.
        assert_eq!(b.current_bytes(t1()), 1500);
        // New reservation rejects (already over).
        let err = b.try_reserve_unscoped(t1(), 1, "test").unwrap_err();
        assert!(matches!(
            err,
            ExecutionError::Plan(ArcQLError::ResourceExhausted { .. })
        ));
    }

    // -------------------------------------------------------------
    // Concurrency (Send + Sync)
    // -------------------------------------------------------------

    #[test]
    fn budget_is_send_sync() {
        // Compile-time pin: MemoryBudget MUST be Send + Sync so a
        // future M4-64b parallel executor can thread it across worker
        // threads.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MemoryBudget>();
        assert_send_sync::<MemoryReservation>();
    }

    #[test]
    fn budget_clones_share_state() {
        // Cloning shares the inner Arc — both clones observe the same
        // counters.
        let a = MemoryBudget::with_per_tenant_cap(t1(), 1000);
        let b = a.clone();
        a.try_reserve_unscoped(t1(), 500, "test").unwrap();
        assert_eq!(a.current_bytes(t1()), 500);
        assert_eq!(b.current_bytes(t1()), 500);
    }

    // -------------------------------------------------------------
    // BUDGET_FALLBACK_ROWS pinning
    // -------------------------------------------------------------

    #[test]
    fn budget_fallback_rows_matches_w11z_constant() {
        // Pin: the v1.0-alpha row-count fallback equals the W11Z
        // SPILLOVER_MAX_ROWS = 64 × BATCH_ROWS = 131072. A future tune
        // of BATCH_ROWS would change this — that's load-bearing for
        // operator-level fallback paths.
        use crate::executor::batch::BATCH_ROWS;
        assert_eq!(BUDGET_FALLBACK_ROWS, 64 * BATCH_ROWS);
        assert_eq!(BUDGET_FALLBACK_ROWS, 131072);
    }

    #[test]
    fn budget_fallback_rows_equals_spillover_max_rows() {
        // W12α fix-up LOW-2 (PR #277 retro): pin that
        // BUDGET_FALLBACK_ROWS is a TRUE alias for SPILLOVER_MAX_ROWS
        // (a single source of truth). A future tune of
        // SPILLOVER_MAX_BATCHES at expand.rs flows here automatically.
        use crate::executor::ops::expand::SPILLOVER_MAX_ROWS;
        assert_eq!(BUDGET_FALLBACK_ROWS, SPILLOVER_MAX_ROWS);
    }

    // -------------------------------------------------------------
    // estimate_value_bytes / estimate_row_bytes shape
    // -------------------------------------------------------------

    #[test]
    fn estimate_value_bytes_is_non_zero_for_every_variant() {
        // Every Value variant carries non-zero stack size.
        assert!(estimate_value_bytes(&Value::Null) > 0);
        assert!(estimate_value_bytes(&Value::Boolean(true)) > 0);
        assert!(estimate_value_bytes(&Value::Integer(0)) > 0);
        assert!(estimate_value_bytes(&Value::Float(0.0)) > 0);
        assert!(estimate_value_bytes(&Value::String("x".into())) > 0);
        assert!(estimate_value_bytes(&Value::List(vec![Value::Null])) > 0);
        // ADR-191 D-13 — a map carries non-zero bytes too.
        let m = Value::Map([("a".to_string(), Value::Integer(1))].into_iter().collect());
        assert!(estimate_value_bytes(&m) > 0);
        // ADR-193 D-9 — a path cell is non-zero and recurses into its
        // node/rel accounting (a zero-length path is still non-zero —
        // the start node's stack size).
        use crate::executor::value::{NodeView, PathView, RelView};
        use arcgraph_core::{LabelId, NodeId, RelId, TypeId};
        let path = Value::Path(
            PathView::new(NodeView::new(NodeId::new(1), Some(LabelId::new(1)))).with_segment(
                RelView::new(
                    RelId::new(10),
                    NodeId::new(1),
                    NodeId::new(2),
                    Some(TypeId::new(1)),
                ),
                NodeView::new(NodeId::new(2), None),
            ),
        );
        assert!(estimate_value_bytes(&path) > 0);
        assert!(
            estimate_value_bytes(&Value::Path(PathView::new(NodeView::new(
                NodeId::new(9),
                None
            )))) > 0
        );
    }

    #[test]
    fn estimate_value_bytes_grows_with_string_length() {
        let short = Value::String("a".into());
        let long = Value::String("a".repeat(1000));
        assert!(estimate_value_bytes(&long) > estimate_value_bytes(&short));
    }

    #[test]
    fn estimate_value_bytes_recurses_into_maps() {
        // ADR-191 D-13 — the map arm RECURSES (not `=> 0`): a bigger
        // map estimates strictly larger, and a nested map contributes
        // its inner bytes (so the per-tenant byte cap can't be defeated
        // by hiding payload inside a nested map).
        use std::collections::BTreeMap;
        let small = Value::Map([("a".to_string(), Value::Integer(1))].into_iter().collect());
        let mut big_map = BTreeMap::new();
        big_map.insert("a".to_string(), Value::Integer(1));
        big_map.insert("b".to_string(), Value::String("x".repeat(500)));
        let big = Value::Map(big_map);
        assert!(estimate_value_bytes(&big) > estimate_value_bytes(&small));
        let nested = Value::Map([("inner".to_string(), big.clone())].into_iter().collect());
        assert!(estimate_value_bytes(&nested) > estimate_value_bytes(&big));
    }
}
