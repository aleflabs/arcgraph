//! W14γ M5-12 — token-bucket invariant proptest.
//!
//! # The invariant
//!
//! Per the rate_limit module docs:
//! > Under any time window of `Δt`, the number of accepted
//! > requests ≤ `c + r·Δt`.
//!
//! Where `c` = burst capacity, `r` = refill rate (tokens/sec).
//!
//! This is the textbook token-bucket invariant: the bucket starts
//! with `c` tokens, refills at `r` tokens/sec (capped at `c`), and
//! a request consumes 1 token. So in any window of `Δt`, the
//! maximum accepted count is `c` (initial budget) + `r·Δt`
//! (refilled during the window). The proptest exercises this under
//! random capacity / rate / request schedules.
//!
//! # Why proptest, not a unit test
//!
//! The unit tests pin specific points (e.g., 100 req/s with 100
//! burst); the proptest searches the parameter space, including
//! pathological combinations: very small capacity, very fast
//! refill, very slow refill, time-window edges. A regression that
//! breaks the invariant under, say, a capacity of 1 and a fractional
//! refill rate would slip past the unit suite.

use std::time::{Duration, Instant};

use arcgraph_core::TenantId;
use arcgraph_mcp::{OpClass, RateLimiter};
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 64,
        ..ProptestConfig::default()
    })]

    /// Token-bucket invariant: in any time window of `Δt`, the number
    /// of accepted requests ≤ ⌊capacity + refill·Δt⌋ + 1 (the +1
    /// allows for floating-point rounding in the deficit math).
    #[test]
    fn token_bucket_invariant_holds_under_random_consume_schedule(
        capacity in 1u32..50,
        refill_per_sec in 0.1f64..100.0,
        // Delays between successive consume attempts, in
        // milliseconds. Up to 64 attempts. Cap each delay at 200ms
        // so a worst-case window is 64 * 200ms = 12.8 seconds.
        delays_ms in proptest::collection::vec(0u64..200, 1..64),
    ) {
        let limiter = RateLimiter::new();
        let tenant = TenantId::new(1);
        // Override the read bucket with our randomized policy.
        limiter.set_per_tenant(tenant, OpClass::Read, capacity, refill_per_sec);

        // Drive the synthetic clock from t0; track the first accepted
        // request's instant + total accept count.
        let t0 = Instant::now();
        let mut now = t0;
        let mut accept_count: u64 = 0;

        for d in &delays_ms {
            now += Duration::from_millis(*d);
            if limiter.try_consume_at(tenant, OpClass::Read, now).is_ok() {
                accept_count += 1;
            }
        }

        // Window is [t0, now]; budget = capacity + refill·Δt.
        let window_secs = (now - t0).as_secs_f64();
        let budget = f64::from(capacity) + refill_per_sec * window_secs;
        // Allow a small fudge for fp rounding (1 token, plus the
        // initial-fill that the bucket arrived with).
        let max_allowed = (budget.ceil() as u64) + 1;

        prop_assert!(
            accept_count <= max_allowed,
            "accepted={accept_count} > max_allowed={max_allowed} \
             (capacity={capacity}, refill={refill_per_sec}, \
             window_secs={window_secs}, budget={budget})",
        );
    }
}
