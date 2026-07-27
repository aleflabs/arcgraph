//! W14γ M5-12 — per-tenant token-bucket rate-limit.
//!
//! Per ADR-037 D-1 multi-tenant routing + ADR-038 amendment-03
//! §Structural-1 + §TIER-2-a + §TIER-2-c: every MCP request flows
//! through a per-tenant rate-limit gate before it reaches a tool
//! body. This module owns the token-bucket primitive +
//! per-`(TenantId, OpClass)` state map; the dispatcher in
//! [`crate::transport`] consults it on the request path.
//!
//! # Default policy (per design-v2 §9.4)
//!
//! Per ADR-004 amendment-02 (`docs/adr/amendments/ADR-004-amendment-02-rate-limit-defaults.md`),
//! the v1.0-alpha defaults align with design-v2 §9.4's
//! 100 read / 10 write **per minute**:
//!
//! - **Read class** (`graph.schema`, `graph.inspect`, future
//!   `graph.search` / `graph.explore`): 100 req/**min** per tenant
//!   (refill ≈ 1.667 tokens/sec); burst capacity = 100 tokens.
//! - **Write class** (`graph.ingest`): 10 req/**min** per tenant
//!   (refill ≈ 0.167 tokens/sec); burst capacity = 10 tokens.
//! - **Admin class**: design-v2 §9.4 specifies 1/min, but no admin-
//!   tier MCP tool ships at M5; the `OpClass::Admin` variant is
//!   deferred to M6+ when `graph.admin.vacuum` / `graph.admin.health`
//!   land (per ADR-004 amendment-02 §D-3).
//!
//! Per-tenant overrides land via [`RateLimiter::set_per_tenant`]; the
//! W12α `MemoryBudget::set_per_tenant_cap` shape is mirrored here so
//! a server-startup config can wire both surfaces in one pass.
//!
//! # Token-bucket math (continuous refill)
//!
//! - Capacity: `c` tokens.
//! - Refill: `r` tokens per second.
//! - On every request: refill = `(now - last_refill).as_secs_f64() * r`,
//!   clamp `tokens = min(c, tokens + refill)`, then attempt to
//!   subtract 1.
//! - Invariant: under any time window of `Δt`, the number of accepted
//!   requests ≤ `c + r·Δt`. Proven by the proptest in
//!   `tests/m5_12_rate_limit_proptest.rs`.
//!
//! # Lock contention back-of-envelope
//!
//! Per-tenant state is guarded by a `parking_lot::Mutex`. At the
//! v1.0-alpha 100 req/min steady-state refill, a saturating tenant
//! takes the lock ~1.67 times per second; parking_lot's lock+unlock
//! fast path is ~30ns, so per-tenant CPU overhead is well under 1ppm.
//! At burst-saturated 100 req/sec (the 100-token initial-fill drawn
//! in one second), the lock cost is ~3µs/sec/tenant; at 10K active
//! bursting tenants this is ~3%, still well inside the v1.0 budget.
//! v1.1+ may switch to a lock-free atomic-CAS path if profiling shows
//! the Mutex as the dominant cost.
//!
//! # Surface seam — `RateLimiter`
//!
//! [`RateLimiter`] is the public type the dispatcher composes. It is
//! `Arc`-cloneable; clones share the same per-tenant state via
//! `Arc<RateLimiterInner>`. Construct one per server process; share
//! across the dispatcher + every tool body that wants per-record
//! token checks (forward-method per the W14δ Bolt slice).
//!
//! # Forward-deferred M5-12 surfaces (per ADR-038 amendment-03)
//!
//! amendment-03 positions M5-12 as the **per-tenant config surface**
//! for FOUR sibling subsystems; this slice lits only the first.
//! The remaining surfaces land in their consuming slices:
//!
//! - **Rate-limit** (this module) — `RateLimitConfig.tenants[].read |
//!   write` per `(TenantId, OpClass)` token bucket. ✓ this slice.
//! - **Per-tenant memory budget** (amendment-03 §Structural-1) —
//!   `MemoryBudget::set_per_tenant_cap` exists in `arcgraph-query`
//!   today as a forward-binding; M4-08+ wiring routes
//!   `RateLimitConfig`-side memory caps through it.
//! - **Plan-cache eviction LRU max-entries-per-tenant** (amendment-03
//!   §TIER-2-a) — wires at M5-02 streamable-HTTP or M5-13 Bolt when
//!   the `QueryEngine` constructor accepts a per-tenant policy bag.
//! - **Slow-query log threshold** (amendment-03 §TIER-2-c) — wires at
//!   M6-08 ops surface when `graph.stats` ships the slow-query log.
//! - **Per-tenant query timeout** (amendment-03 §TIER-1 GAP C) —
//!   W12γ owns the consumer at `arcgraph-query::executor::cancel`
//!   (see `DEFAULT_QUERY_TIMEOUT_MS = 30_000` there). The
//!   `RateLimiter`-side per-tenant timeout helper is deferred until
//!   M5-02 / M5-13 wires it through the `QueryEngine` constructor.
//!
//! # ADR provenance
//! - **design-v2 §9.4** — canonical rate-limit defaults (100/min
//!   read, 10/min write, 1/min admin) restored in this slice's
//!   `DEFAULT_*_REFILL_PER_SEC` constants per ADR-004 amendment-02.
//! - **ADR-004 amendment-02 §D-1 + §D-2 + §D-3** — per-tenant keying
//!   ratified; per-minute defaults restored; admin op class deferred
//!   to M6+.
//! - **ADR-038 amendment-03 §Structural-1 + §TIER-2-a + §TIER-2-c** —
//!   M5-12 is the per-tenant config surface for rate-limit + memory
//!   budget + plan-cache eviction + slow-query log threshold.
//! - **ADR-037 D-1** — per-tenant routing inherited via the bucket's
//!   `(TenantId, OpClass)` keying.
//! - **Config strictness** — [`RateLimitConfig`] rejects unknown fields.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────
// Defaults (design-v2 §9.4 — restored by ADR-004 amendment-02 §D-2)
// ─────────────────────────────────────────────────────────────────────

/// Default read-class burst capacity (tokens) per design-v2 §9.4.
/// 100 tokens is the maximum back-to-back read consumption a tenant
/// can drive before the bucket throttles.
pub const DEFAULT_READ_CAPACITY: u32 = 100;

/// Default read-class refill rate: 100 tokens / minute (per design-v2
/// §9.4). Continuous-refill math expresses this as tokens / second,
/// so 100/60 ≈ 1.667. Restored from the W14γ M5-12 60×-over-permissive
/// initial ship per ADR-004 amendment-02 §D-2.
pub const DEFAULT_READ_REFILL_PER_SEC: f64 = 100.0 / 60.0;

/// Default write-class burst capacity (tokens) per design-v2 §9.4.
/// Lower than read by design — write workloads land far heavier on
/// group-commit fsync cohorts per ADR-031.
pub const DEFAULT_WRITE_CAPACITY: u32 = 10;

/// Default write-class refill rate: 10 tokens / minute (per design-v2
/// §9.4). 10/60 ≈ 0.167 tokens/sec.
pub const DEFAULT_WRITE_REFILL_PER_SEC: f64 = 10.0 / 60.0;

// ─────────────────────────────────────────────────────────────────────
// OpClass — read vs write bucketing
// ─────────────────────────────────────────────────────────────────────

/// Operation class used to key the per-tenant token bucket.
///
/// Read-side and write-side workloads are kept in separate buckets so
/// a tenant rate-limited on writes can still serve reads (and vice
/// versa). The default policy reflects this asymmetry: 100 read req/s
/// vs 10 write req/s.
///
/// `#[non_exhaustive]` under the strict public-contract policy: future v1.1+ may add
/// `Admin` / `Bulk` classes; downstream pattern-matchers MUST keep a
/// wildcard arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum OpClass {
    /// Read-side ops: `graph.schema`, `graph.inspect`, `graph.search`,
    /// and `graph.explore`.
    Read,
    /// Write-side ops: `graph.ingest`, future `graph.delete`.
    Write,
}

impl OpClass {
    /// Default capacity for this op class.
    #[must_use]
    pub fn default_capacity(self) -> u32 {
        match self {
            OpClass::Read => DEFAULT_READ_CAPACITY,
            OpClass::Write => DEFAULT_WRITE_CAPACITY,
        }
    }

    /// Default refill rate (tokens/sec) for this op class.
    #[must_use]
    pub fn default_refill_per_sec(self) -> f64 {
        match self {
            OpClass::Read => DEFAULT_READ_REFILL_PER_SEC,
            OpClass::Write => DEFAULT_WRITE_REFILL_PER_SEC,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// RateLimitError
// ─────────────────────────────────────────────────────────────────────

/// Codec-local error type for the rate-limit surface.
///
/// `#[non_exhaustive]` under the strict public-contract policy: production wiring may
/// add per-class soft-limit or quota-exceeded variants; downstream
/// pattern-matchers MUST keep a wildcard arm.
#[derive(Debug, Clone, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum RateLimitError {
    /// Bucket has zero tokens; the caller MUST retry after
    /// `retry_after`.
    #[error("rate limit exceeded; retry after {}ms", retry_after.as_millis())]
    Exceeded {
        /// Suggested back-off duration (the time remaining until the
        /// next token refills under the configured rate).
        retry_after: Duration,
    },
}

// ─────────────────────────────────────────────────────────────────────
// TokenBucket — per-(tenant, class) state
// ─────────────────────────────────────────────────────────────────────

/// Token-bucket state for a single (tenant, op_class) pair.
///
/// The bucket carries `f64` tokens so the continuous-refill math is
/// exact under sub-second windows (e.g., 5ms gap between requests at
/// 100 req/s refills 0.5 tokens). `try_consume` rounds the integer
/// take down to the available floor.
#[derive(Debug, Clone, Copy)]
struct TokenBucket {
    /// Configured maximum (burst) capacity.
    capacity: u32,
    /// Refill rate in tokens per second.
    refill_per_sec: f64,
    /// Current token count (refilled lazily on consume).
    tokens: f64,
    /// Last refill timestamp (monotonic).
    last_refill: Instant,
}

impl TokenBucket {
    fn new(capacity: u32, refill_per_sec: f64) -> Self {
        Self {
            capacity,
            refill_per_sec,
            tokens: f64::from(capacity),
            last_refill: Instant::now(),
        }
    }

    /// Refill based on elapsed time, clamping at capacity.
    fn refill(&mut self, now: Instant) {
        let elapsed = now.saturating_duration_since(self.last_refill);
        let added = elapsed.as_secs_f64() * self.refill_per_sec;
        self.tokens = (self.tokens + added).min(f64::from(self.capacity));
        self.last_refill = now;
    }

    /// Try to take 1 token. Returns:
    /// - `Ok(())` if the token was consumed.
    /// - `Err(retry_after)` with the time until the next token
    ///   refills (i.e., the time the caller should sleep before
    ///   retrying).
    fn try_consume(&mut self, now: Instant) -> Result<(), Duration> {
        self.refill(now);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            Ok(())
        } else {
            // Compute sub-token deficit, convert to time-until-refill.
            let deficit = 1.0 - self.tokens;
            let secs = if self.refill_per_sec > 0.0 {
                deficit / self.refill_per_sec
            } else {
                // Zero-rate buckets never refill; suggest 1s as a
                // sentinel back-off.
                1.0
            };
            // Defensive clamp — never suggest a back-off shorter
            // than 1ms (avoids a hot-spin loop on the client).
            let secs = secs.max(0.001);
            Err(Duration::from_secs_f64(secs))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// RateLimiter
// ─────────────────────────────────────────────────────────────────────

/// Per-tenant per-op-class token-bucket rate-limiter.
///
/// `Arc`-cloneable; clones share state via `Arc<RateLimiterInner>`.
/// Construct one per server process. The dispatcher consults
/// [`RateLimiter::try_consume`] on every request before invoking
/// the tool body.
///
/// # Tenant-isolation invariant
///
/// One tenant's bucket draining to zero MUST NOT affect another
/// tenant's bucket. Pinned by the
/// `rate_limiter_buckets_are_per_tenant_isolated` test.
///
/// # Op-class isolation invariant
///
/// One op class's bucket draining to zero MUST NOT affect the other
/// op class's bucket on the same tenant. Pinned by the
/// `rate_limiter_op_class_buckets_are_independent` test.
#[derive(Debug, Clone, Default)]
pub struct RateLimiter {
    inner: Arc<RateLimiterInner>,
}

#[derive(Debug, Default)]
struct RateLimiterInner {
    /// Per-(tenant, class) bucket state.
    buckets: Mutex<HashMap<(TenantId, OpClass), TokenBucket>>,
    /// Per-tenant configured override for per-class capacity + refill.
    /// `None`-keyed tenants fall back to [`OpClass::default_capacity`]
    /// + [`OpClass::default_refill_per_sec`] on first observation.
    overrides: Mutex<HashMap<(TenantId, OpClass), (u32, f64)>>,
}

impl RateLimiter {
    /// Construct a new rate-limiter with default per-class policy.
    /// Tenants are configured lazily — the first observation of a
    /// `(tenant, class)` pair seeds the bucket from defaults (or any
    /// configured override).
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set per-tenant `(capacity, refill_per_sec)` for `op_class`.
    ///
    /// Mirrors the [`arcgraph_query::executor::MemoryBudget::set_per_tenant_cap`]
    /// shape (per amendment-03 §TIER-1 GAP A: "per-tenant override
    /// via M5-12 rate-limit config"). Idempotent and safe to call
    /// concurrently with `try_consume`. Tightening the capacity does
    /// NOT retroactively eject tokens already issued; only NEW
    /// requests see the tightened cap.
    pub fn set_per_tenant(
        &self,
        tenant: TenantId,
        op_class: OpClass,
        capacity: u32,
        refill_per_sec: f64,
    ) {
        let mut overrides = self.inner.overrides.lock();
        overrides.insert((tenant, op_class), (capacity, refill_per_sec));
        // Refresh the live bucket if one already exists, so the new
        // policy applies on the very next request (don't wait for a
        // bucket-eviction). We DO preserve the current `tokens` count
        // (clamping at the new capacity) so an in-flight burst isn't
        // erased by a config touch-up.
        let mut buckets = self.inner.buckets.lock();
        if let Some(b) = buckets.get_mut(&(tenant, op_class)) {
            b.capacity = capacity;
            b.refill_per_sec = refill_per_sec;
            b.tokens = b.tokens.min(f64::from(capacity));
        }
    }

    /// Read the configured `(capacity, refill_per_sec)` for
    /// `(tenant, op_class)`, falling back to defaults if no override
    /// is configured.
    #[must_use]
    pub fn per_tenant_config(&self, tenant: TenantId, op_class: OpClass) -> (u32, f64) {
        self.inner
            .overrides
            .lock()
            .get(&(tenant, op_class))
            .copied()
            .unwrap_or_else(|| {
                (
                    op_class.default_capacity(),
                    op_class.default_refill_per_sec(),
                )
            })
    }

    /// Try to consume one token from the `(tenant, op_class)` bucket.
    ///
    /// On exhaustion returns [`RateLimitError::Exceeded`] with a
    /// suggested back-off duration. The caller (typically the
    /// dispatcher) maps this onto the MCP wire surface as
    /// [`crate::error::MCPError::RateLimited`] with `retry_after_ms`.
    pub fn try_consume(&self, tenant: TenantId, op_class: OpClass) -> Result<(), RateLimitError> {
        self.try_consume_at(tenant, op_class, Instant::now())
    }

    /// `try_consume` variant accepting a synthetic `now` timestamp —
    /// the unit-test harness uses this to drive the refill clock
    /// deterministically without sleeping.
    pub fn try_consume_at(
        &self,
        tenant: TenantId,
        op_class: OpClass,
        now: Instant,
    ) -> Result<(), RateLimitError> {
        let (capacity, refill_per_sec) = self.per_tenant_config(tenant, op_class);
        let mut buckets = self.inner.buckets.lock();
        let bucket = buckets.entry((tenant, op_class)).or_insert_with(|| {
            let mut b = TokenBucket::new(capacity, refill_per_sec);
            b.last_refill = now;
            b
        });
        // If the bucket existed with a stale capacity, sync to current.
        bucket.capacity = capacity;
        bucket.refill_per_sec = refill_per_sec;
        bucket.tokens = bucket.tokens.min(f64::from(capacity));
        match bucket.try_consume(now) {
            Ok(()) => Ok(()),
            Err(retry_after) => Err(RateLimitError::Exceeded { retry_after }),
        }
    }

    /// Read the current token count for diagnostics. Refills lazily
    /// (so a long-idle bucket reads at capacity, not at the
    /// last-stored stale value).
    #[must_use]
    pub fn current_tokens(&self, tenant: TenantId, op_class: OpClass) -> f64 {
        let (capacity, refill_per_sec) = self.per_tenant_config(tenant, op_class);
        let mut buckets = self.inner.buckets.lock();
        let bucket = buckets
            .entry((tenant, op_class))
            .or_insert_with(|| TokenBucket::new(capacity, refill_per_sec));
        bucket.refill(Instant::now());
        bucket.tokens
    }
}

// ─────────────────────────────────────────────────────────────────────
// RateLimitConfig — server-startup config struct
// ─────────────────────────────────────────────────────────────────────

/// Server-startup rate-limit configuration.
///
/// `#[serde(deny_unknown_fields)]` under the code-quality policy strict-mode.
/// Loaded once from server config; passed to [`RateLimiter::from_config`]
/// to seed the per-tenant overrides.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct RateLimitConfig {
    /// Per-tenant overrides. Each entry seeds both read + write
    /// buckets. v1.0-alpha admits one override per tenant; v1.1+
    /// MAY add per-class overrides if a tenant needs an asymmetric
    /// policy.
    #[serde(default)]
    pub tenants: Vec<TenantPolicy>,
}

/// Per-tenant rate-limit policy entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TenantPolicy {
    /// Tenant id.
    pub tenant_id: u64,
    /// Optional read-class override `(capacity, refill_per_sec)`.
    /// `None` → inherit defaults.
    #[serde(default)]
    pub read: Option<ClassPolicy>,
    /// Optional write-class override.
    #[serde(default)]
    pub write: Option<ClassPolicy>,
}

/// Per-class capacity + refill policy entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClassPolicy {
    /// Burst capacity (max tokens).
    pub capacity: u32,
    /// Refill rate (tokens/sec).
    pub refill_per_sec: f64,
}

impl RateLimiter {
    /// Construct a rate-limiter from a server-startup config.
    #[must_use]
    pub fn from_config(cfg: &RateLimitConfig) -> Self {
        let limiter = Self::new();
        for tp in &cfg.tenants {
            let tenant = TenantId::new(tp.tenant_id);
            if let Some(read) = &tp.read {
                limiter.set_per_tenant(tenant, OpClass::Read, read.capacity, read.refill_per_sec);
            }
            if let Some(write) = &tp.write {
                limiter.set_per_tenant(
                    tenant,
                    OpClass::Write,
                    write.capacity,
                    write.refill_per_sec,
                );
            }
        }
        limiter
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn t1() -> TenantId {
        TenantId::new(1)
    }
    fn t2() -> TenantId {
        TenantId::new(2)
    }

    // ── Token-bucket math ─────────────────────────────────────────

    #[test]
    fn rate_limiter_consumes_tokens_within_capacity() {
        // Empty default-policy bucket starts at capacity = 100 (read).
        // 100 consumes back-to-back must succeed; the 101st must fail.
        let r = RateLimiter::new();
        for i in 0..100 {
            r.try_consume(t1(), OpClass::Read)
                .unwrap_or_else(|e| panic!("token {i} failed: {e}"));
        }
        let err = r
            .try_consume(t1(), OpClass::Read)
            .expect_err("101st must reject");
        match err {
            RateLimitError::Exceeded { retry_after } => {
                assert!(
                    retry_after >= Duration::from_millis(1),
                    "retry_after must be >= 1ms"
                );
                assert!(
                    retry_after <= Duration::from_secs(2),
                    "retry_after must be sane"
                );
            }
        }
    }

    #[test]
    fn rate_limiter_refills_over_time() {
        // Drive the synthetic clock forward under the design-v2 §9.4
        // default refill of 100/min (≈ 1.667 tokens/sec). After 60s
        // the bucket refills to its 100-token capacity.
        let r = RateLimiter::new();
        let t0 = Instant::now();
        for _ in 0..100 {
            r.try_consume_at(t1(), OpClass::Read, t0).unwrap();
        }
        // Bucket empty.
        assert!(r.try_consume_at(t1(), OpClass::Read, t0).is_err());
        // Advance 60 seconds; bucket refills to 100 (capped at capacity).
        let t_full = t0 + Duration::from_secs(60);
        for _ in 0..100 {
            r.try_consume_at(t1(), OpClass::Read, t_full)
                .expect("should refill to capacity");
        }
        // Now empty again.
        assert!(r.try_consume_at(t1(), OpClass::Read, t_full).is_err());
    }

    // ── Per-tenant isolation ──────────────────────────────────────

    #[test]
    fn rate_limiter_buckets_are_per_tenant_isolated() {
        // Drain tenant 1's read bucket; tenant 2 must still serve.
        // ADR-037 D-1 multi-tenant routing inheritance.
        let r = RateLimiter::new();
        for _ in 0..100 {
            r.try_consume(t1(), OpClass::Read).unwrap();
        }
        assert!(r.try_consume(t1(), OpClass::Read).is_err());
        // Tenant 2 unaffected.
        for _ in 0..100 {
            r.try_consume(t2(), OpClass::Read)
                .expect("tenant 2 must not be rate-limited");
        }
    }

    // ── Per-class isolation ───────────────────────────────────────

    #[test]
    fn rate_limiter_op_class_buckets_are_independent() {
        // Drain tenant 1's WRITE bucket (10 tokens at default); the
        // READ bucket on the same tenant must still serve. Default
        // asymmetry (100 read / 10 write) is the canonical M5-12
        // policy.
        let r = RateLimiter::new();
        for _ in 0..10 {
            r.try_consume(t1(), OpClass::Write).unwrap();
        }
        assert!(r.try_consume(t1(), OpClass::Write).is_err());
        // Read bucket on the same tenant is independent.
        for _ in 0..100 {
            r.try_consume(t1(), OpClass::Read)
                .expect("read bucket must not see write drain");
        }
    }

    // ── Per-tenant override (set_per_tenant) ──────────────────────

    #[test]
    fn rate_limiter_per_tenant_override_applies_immediately() {
        // Override tenant 1's write to (capacity=2, refill=2/s); only
        // 2 consumes succeed before the bucket empties.
        let r = RateLimiter::new();
        r.set_per_tenant(t1(), OpClass::Write, 2, 2.0);
        let cfg = r.per_tenant_config(t1(), OpClass::Write);
        assert_eq!(cfg, (2, 2.0));
        for _ in 0..2 {
            r.try_consume(t1(), OpClass::Write).unwrap();
        }
        assert!(r.try_consume(t1(), OpClass::Write).is_err());
    }

    #[test]
    fn rate_limiter_override_does_not_leak_across_tenants() {
        // Override tenant 1; tenant 2's policy stays at defaults.
        let r = RateLimiter::new();
        r.set_per_tenant(t1(), OpClass::Read, 5, 5.0);
        let cfg1 = r.per_tenant_config(t1(), OpClass::Read);
        assert_eq!(cfg1, (5, 5.0));
        let cfg2 = r.per_tenant_config(t2(), OpClass::Read);
        assert_eq!(cfg2, (DEFAULT_READ_CAPACITY, DEFAULT_READ_REFILL_PER_SEC));
    }

    // ── retry_after sanity ────────────────────────────────────────

    #[test]
    fn rate_limit_error_carries_sane_retry_after() {
        // Drain 1-token capacity; the next consume must return a
        // RateLimitError::Exceeded with retry_after ≈ 1/refill_rate.
        let r = RateLimiter::new();
        r.set_per_tenant(t1(), OpClass::Write, 1, 5.0);
        let t0 = Instant::now();
        r.try_consume_at(t1(), OpClass::Write, t0).unwrap();
        let err = r
            .try_consume_at(t1(), OpClass::Write, t0)
            .expect_err("must reject");
        let retry = match err {
            RateLimitError::Exceeded { retry_after } => retry_after,
        };
        // Refill rate 5 tokens/sec → 1 token in 200ms.
        assert!(retry >= Duration::from_millis(150), "retry={retry:?}");
        assert!(retry <= Duration::from_millis(250), "retry={retry:?}");
    }

    // ── Send/Sync ─────────────────────────────────────────────────

    #[test]
    fn rate_limiter_is_send_sync_clone_shares_state() {
        // RateLimiter must be Send + Sync so the dispatcher can clone
        // it across awaits + share with tools.
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RateLimiter>();

        let r1 = RateLimiter::new();
        let r2 = r1.clone();
        // Drain via r1; r2 sees the same state.
        for _ in 0..100 {
            r1.try_consume(t1(), OpClass::Read).unwrap();
        }
        assert!(r2.try_consume(t1(), OpClass::Read).is_err());
    }

    // ── from_config ──────────────────────────────────────────────

    #[test]
    fn rate_limiter_from_config_seeds_overrides() {
        let cfg = RateLimitConfig {
            tenants: vec![TenantPolicy {
                tenant_id: 1,
                read: Some(ClassPolicy {
                    capacity: 5,
                    refill_per_sec: 5.0,
                }),
                write: Some(ClassPolicy {
                    capacity: 1,
                    refill_per_sec: 1.0,
                }),
            }],
        };
        let r = RateLimiter::from_config(&cfg);
        assert_eq!(r.per_tenant_config(t1(), OpClass::Read), (5, 5.0));
        assert_eq!(r.per_tenant_config(t1(), OpClass::Write), (1, 1.0));
    }

    #[test]
    fn rate_limit_config_rejects_unknown_field() {
        // Strict-mode discipline.
        let v = serde_json::json!({"tenants": [{"tenant_id": 1, "reed": null}]});
        let res: Result<RateLimitConfig, _> = serde_json::from_value(v);
        assert!(res.is_err(), "typo must reject");
    }

    // ── current_tokens diagnostics ────────────────────────────────

    #[test]
    fn rate_limiter_current_tokens_reflects_consumption() {
        let r = RateLimiter::new();
        let before = r.current_tokens(t1(), OpClass::Read);
        // Default capacity = 100, the lazy seed reads at capacity.
        assert!(before >= 99.0);
        r.try_consume(t1(), OpClass::Read).unwrap();
        let after = r.current_tokens(t1(), OpClass::Read);
        // After 1 consume + a tiny refill, we expect ~99.
        assert!(after < before, "consumption must decrement");
    }

    // ── tightening capacity preserves liveness ────────────────────

    #[test]
    fn rate_limiter_tightening_capacity_clamps_existing_bucket() {
        // Per the doc-comment commitment on set_per_tenant: tightening
        // the cap below the current `tokens` clamps tokens to the new
        // capacity (fairness vs. an in-flight burst is a fairness call;
        // the contract is that the new policy applies on the next
        // request).
        let r = RateLimiter::new();
        // Seed the bucket with a default 100-token read pool.
        let _ = r.try_consume(t1(), OpClass::Read);
        // Tighten capacity to 5.
        r.set_per_tenant(t1(), OpClass::Read, 5, 5.0);
        // The very next 5 consumes succeed; the 6th rejects.
        for _ in 0..5 {
            r.try_consume(t1(), OpClass::Read)
                .expect("post-tightening burst");
        }
        assert!(r.try_consume(t1(), OpClass::Read).is_err());
    }
}
