//! W22-DB-ε Pillar 3 — Traversal-depth DoS protection.
//!
//! # What this validates
//!
//! Per ADR-051 §"Resource limits" + ADR-079 Pillar 3, the v1.0-α
//! traversal-depth cap protects the executor from pathological
//! substrates with low diameter + dense cycles. The cap is `64`
//! hops (per LDBC SNB Interactive's worst-case diameter of ~6, the
//! cap is intentionally generous — but BOUNDED).
//!
//! This test pins the constant + the contract — a doc-binding
//! assertion that future M5+ changes can't silently lift the cap
//! without surfacing a test failure.
//!
//! # Why pin the constant?
//!
//! The cap is the v1.0-α DoS-protection floor. A future "let's
//! make it configurable" refactor that defaults to `u32::MAX`
//! removes the protection without surfacing in the existing
//! executor test suite. A pin here forces the refactorer to either
//! keep `64` as the floor or document the change.

#![allow(clippy::expect_used)]

use arcgraph_query::executor::ops::path::DEFAULT_MAX_DEPTH;

/// W22-DB-ε binding: `DEFAULT_MAX_DEPTH` is the v1.0-α DoS-protection
/// floor at 64 hops. Lifting this requires an ADR-079 amendment +
/// fault-injection coverage of the new bound.
#[test]
fn w22_dbe_default_max_depth_pinned_at_64() {
    assert_eq!(
        DEFAULT_MAX_DEPTH, 64,
        "DEFAULT_MAX_DEPTH MUST stay at 64 hops (LDBC SNB Interactive diameter ~6 + headroom; ADR-051 §\"Resource limits\")"
    );
}

/// Bound-class invariant: `DEFAULT_MAX_DEPTH` must be in the closed
/// range `[32, 256]`. Below 32 the cap blocks legitimate queries
/// against LDBC-class diameter-12 substrates; above 256 the cap
/// stops protecting against pathological cycles + low-diameter
/// substrates per the design-v2 §10.5 budget.
///
/// Implemented as a `const _:` assertion so a future refactor that
/// makes `DEFAULT_MAX_DEPTH` configurable (i.e., non-`const`) would
/// require explicit re-pinning of the safety-band invariant at the
/// config-load entry point.
#[test]
#[allow(clippy::assertions_on_constants)]
fn w22_dbe_default_max_depth_within_safety_band() {
    // Const-evaluated guard — compile-time check.
    const _: () = assert!(
        DEFAULT_MAX_DEPTH >= 32,
        "DEFAULT_MAX_DEPTH is too low: LDBC diameter-12 substrates need ≥ 32 hops"
    );
    const _: () = assert!(
        DEFAULT_MAX_DEPTH <= 256,
        "DEFAULT_MAX_DEPTH is too high: pathological-cycle DoS bound is 256 per ADR-051"
    );
    // Runtime guard — also asserts, so the safety-band shows in test
    // output. The const_ above is the load-bearing one; this is the
    // explicit-discoverability surface for `cargo test --list`.
    assert!(
        DEFAULT_MAX_DEPTH >= 32,
        "DEFAULT_MAX_DEPTH = {DEFAULT_MAX_DEPTH} is too low"
    );
    assert!(
        DEFAULT_MAX_DEPTH <= 256,
        "DEFAULT_MAX_DEPTH = {DEFAULT_MAX_DEPTH} is too high"
    );
}
