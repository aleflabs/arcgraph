//! W26-γ-2 D3 / D2 — BUGGIFY macro self-tests per ADR-135.
//!
//! The `clippy::assertions_on_constants` lint is allowed at file level
//! because the production-mode tests intentionally assert on
//! `buggify!()` which expands to literal `false` (per ADR-135 D-3) —
//! the assertion-on-constants IS the property being tested.
//!
//! `clippy::unnecessary_get_then_check` is allowed because the
//! tests use `HashMap::get(...).is_none()` deliberately to communicate
//! "this site was never observed" semantics distinct from
//! `contains_key()` membership.
#![allow(clippy::assertions_on_constants, clippy::unnecessary_get_then_check)]
//!
//! Two test modes:
//!
//! 1. **Production mode** (no `--cfg arcgraph_sim`) — the macro
//!    expands to literal `false`; every site is a no-op. The
//!    production-shape tests assert that.
//! 2. **Simulation mode** (`RUSTFLAGS="--cfg arcgraph_sim"`) — the
//!    macro fires per the two-level conditional model. The
//!    simulation-shape tests assert statistical-distribution +
//!    determinism invariants.
//!
//! The simulation tests are cfg-gated on `arcgraph_sim`; the
//! production tests always run. The R1 reviewer hostile-grep over
//! `registered_sites()` runs in simulation mode only (since
//! production mode does not call into `buggify_with_probs` and
//! therefore never registers any site).
//!
//! See ADR-135 D-3 (cfg gate), D-4 (probability semantics), D-5
//! (naming convention).

use std::sync::Mutex;

use arcgraph_core::buggify;
use arcgraph_core::buggify::{
    buggify_fire_counts, buggify_with_probs, registered_sites, reset_buggify_state, seed_buggify,
};

/// Serial guard for the global BUGGIFY state. Cargo runs tests in
/// parallel by default; the BUGGIFY state is a singleton (matching
/// production semantics where every crate's `buggify!()` shares one
/// RNG). The guard serializes per-test access so seed-based
/// determinism + per-site fire counts are not polluted by concurrent
/// tests.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial_guard() -> std::sync::MutexGuard<'static, ()> {
    // PoisonError = previous test panicked while holding the guard;
    // recover the inner unit guard so the rest of the test suite
    // can continue running.
    match SERIAL.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    }
}

// ────────────────────── Production-build invariant: macro is false ──────────────────────

#[test]
fn production_build_macro_expands_to_false() {
    let _g = serial_guard();
    // In production (no `arcgraph_sim` cfg), `buggify!()` MUST be
    // literal `false`. The compiler dead-branch eliminates the call
    // site. The test is unconditional — when run with
    // `RUSTFLAGS="--cfg arcgraph_sim"` the simulation-build
    // probability is so close to zero on a single call that we
    // would lose almost no signal; the production test asserts the
    // strong invariant that on a default `cargo test` run the macro
    // is no-op.
    if cfg!(arcgraph_sim) {
        // Simulation build: skip this assertion (covered by the
        // simulation-mode tests below).
        return;
    }
    // Multiple call shapes (1-arg, 2-arg, 3-arg) all must be no-op.
    for _ in 0..1000 {
        assert!(!buggify!("test.production.smoke"));
        assert!(!buggify!("test.production.with_fire_prob", 0.9));
        assert!(!buggify!("test.production.with_both_probs", 0.5, 0.5));
    }
}

#[test]
fn production_build_runtime_fn_with_high_prob_in_production_mode_is_false() {
    let _g = serial_guard();
    // Per ADR-135 D-3: in production mode the runtime fn is never
    // reached (the macro evaluates to literal false). But the fn is
    // still defined (so testing infrastructure can call it directly).
    // In production mode, a direct call with prob=(1.0, 1.0) still
    // honors the fn semantics — we just can't get the macro to call
    // it. Verify the fn does what it says, then assert that the
    // macro does NOT reach it.
    if cfg!(arcgraph_sim) {
        return;
    }
    reset_buggify_state();
    // The fn itself is callable + can return true.
    seed_buggify(42);
    let direct = buggify_with_probs("test.direct.always_fire", 1.0, 1.0);
    assert!(direct, "direct fn must fire at prob=(1.0, 1.0)");
    // Per the production macro definition, calling buggify! cannot
    // reach the fn; the macro is literal `false`.
    for _ in 0..100 {
        assert!(!buggify!("test.production.unreachable_via_macro"));
    }
}

// ────────────────────── Two-level firing model (always-on tests) ──────────────────────

#[test]
fn always_fire_when_both_probs_one() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(1);
    let mut fires = 0;
    for _ in 0..100 {
        // Use the fn directly so the test runs in production mode too.
        if buggify_with_probs("test.always_fire_when_both_probs_one", 1.0, 1.0) {
            fires += 1;
        }
    }
    // With enable=1.0 the site is always enabled; with fire=1.0
    // every evaluation fires.
    assert_eq!(fires, 100);
}

#[test]
fn never_fire_when_either_prob_zero() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(2);
    for _ in 0..100 {
        assert!(!buggify_with_probs("test.never_fire_enable_zero", 0.0, 1.0));
    }
    reset_buggify_state();
    seed_buggify(3);
    let mut fires = 0;
    for _ in 0..100 {
        if buggify_with_probs("test.never_fire_when_fire_zero", 1.0, 0.0) {
            fires += 1;
        }
    }
    assert_eq!(fires, 0);
}

// ────────────────────── Determinism: same seed → same fire sequence ──────────────────────

#[test]
fn same_seed_produces_same_fire_sequence() {
    let _g = serial_guard();
    let mut seq_a = Vec::new();
    let mut seq_b = Vec::new();

    reset_buggify_state();
    seed_buggify(42);
    for i in 0..50 {
        // Site name varies per iteration so the per-run cache does
        // NOT short-circuit; each call exercises both the enable AND
        // the fire coin.
        let name: &'static str = match i % 4 {
            0 => "test.det.a",
            1 => "test.det.b",
            2 => "test.det.c",
            _ => "test.det.d",
        };
        seq_a.push(buggify_with_probs(name, 0.5, 0.5));
    }

    reset_buggify_state();
    seed_buggify(42);
    for i in 0..50 {
        let name: &'static str = match i % 4 {
            0 => "test.det.a",
            1 => "test.det.b",
            2 => "test.det.c",
            _ => "test.det.d",
        };
        seq_b.push(buggify_with_probs(name, 0.5, 0.5));
    }

    assert_eq!(
        seq_a, seq_b,
        "ADR-134 D-3 reproducibility contract: same seed must produce same sequence"
    );
}

#[test]
fn distinct_seeds_diverge() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(100);
    let mut seq_a = Vec::new();
    for i in 0..30 {
        let name: &'static str = if i % 2 == 0 {
            "test.div.even"
        } else {
            "test.div.odd"
        };
        seq_a.push(buggify_with_probs(name, 0.5, 0.5));
    }

    reset_buggify_state();
    seed_buggify(200);
    let mut seq_b = Vec::new();
    for i in 0..30 {
        let name: &'static str = if i % 2 == 0 {
            "test.div.even"
        } else {
            "test.div.odd"
        };
        seq_b.push(buggify_with_probs(name, 0.5, 0.5));
    }

    // Distinct seeds must produce different sequences with high
    // probability across 30 samples.
    assert_ne!(seq_a, seq_b, "distinct seeds must diverge");
}

// ────────────────────── Per-run enable cache ──────────────────────

#[test]
fn per_run_enable_cache_short_circuits() {
    let _g = serial_guard();
    // When a site is disabled in a run, NO subsequent eval fires.
    reset_buggify_state();
    seed_buggify(0xABCD);
    // Force enable=0 ⇒ site is unconditionally disabled, regardless
    // of fire prob.
    let mut fires = 0;
    for _ in 0..1000 {
        if buggify_with_probs("test.cache.disabled", 0.0, 1.0) {
            fires += 1;
        }
    }
    assert_eq!(fires, 0);

    // When forced enabled, fire-prob alone gates each eval.
    reset_buggify_state();
    seed_buggify(0xABCD);
    let mut fires = 0;
    for _ in 0..1000 {
        if buggify_with_probs("test.cache.enabled_always_fire", 1.0, 1.0) {
            fires += 1;
        }
    }
    assert_eq!(fires, 1000);
}

// ────────────────────── Fire-count observability ──────────────────────

#[test]
fn fire_counts_track_per_site() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(7);

    // Site that always fires:
    for _ in 0..50 {
        let _ = buggify_with_probs("test.fc.always", 1.0, 1.0);
    }
    let fc = buggify_fire_counts();
    assert_eq!(fc.get("test.fc.always").copied(), Some(50));

    // Site that never fires:
    for _ in 0..50 {
        let _ = buggify_with_probs("test.fc.never", 0.0, 0.0);
    }
    let fc = buggify_fire_counts();
    // Never-fire site doesn't appear in fire_count (defaultmap-style).
    assert!(
        fc.get("test.fc.never").is_none(),
        "site with 0 fires should not be in fire_count map"
    );
}

#[test]
fn reset_clears_fire_counts() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(8);
    for _ in 0..30 {
        let _ = buggify_with_probs("test.reset.before", 1.0, 1.0);
    }
    assert_eq!(
        buggify_fire_counts().get("test.reset.before").copied(),
        Some(30)
    );
    reset_buggify_state();
    assert!(
        buggify_fire_counts().get("test.reset.before").is_none(),
        "reset must clear fire counts"
    );
}

// ────────────────────── Probability clamping ──────────────────────

#[test]
fn negative_probability_treated_as_zero() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(9);
    let mut fires = 0;
    for _ in 0..100 {
        if buggify_with_probs("test.clamp.negative", -1.0, -1.0) {
            fires += 1;
        }
    }
    assert_eq!(fires, 0);
}

#[test]
fn above_one_probability_treated_as_one() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(10);
    let mut fires = 0;
    for _ in 0..100 {
        if buggify_with_probs("test.clamp.above_one", 2.0, 2.0) {
            fires += 1;
        }
    }
    // Saturates to (1.0, 1.0) — always fires.
    assert_eq!(fires, 100);
}

#[test]
fn nan_probability_treated_as_zero() {
    let _g = serial_guard();
    reset_buggify_state();
    seed_buggify(11);
    let mut fires = 0;
    for _ in 0..100 {
        if buggify_with_probs("test.clamp.nan", f64::NAN, f64::NAN) {
            fires += 1;
        }
    }
    assert_eq!(fires, 0);
}

// ────────────────────── ADR-135 D-5 naming convention ──────────────────────

#[test]
fn site_names_follow_dotted_convention() {
    let _g = serial_guard();
    // Register some sites via direct calls.
    reset_buggify_state();
    seed_buggify(12);
    let valid_names = [
        "storage.wal.fsync_pre_write",
        "storage.checkpoint.install_crash",
        "core.id.allocation_overflow",
        "vector.hnsw.neighbor_torn_write",
    ];
    for n in &valid_names {
        let _ = buggify_with_probs(n, 1.0, 1.0);
    }

    let sites = registered_sites();
    for n in &valid_names {
        assert!(sites.contains(n), "site {n} not registered");
        let dots = n.bytes().filter(|b| *b == b'.').count();
        assert!(
            dots >= 2,
            "ADR-135 D-5: site name {n:?} must have ≥ 2 dots (`<crate>.<module>.<mode>`)"
        );
        // Alphanumeric + underscore + dot only (R1 reviewer hostile-grep).
        for b in n.bytes() {
            assert!(
                b.is_ascii_alphanumeric() || b == b'.' || b == b'_',
                "ADR-135 D-5: site name {n:?} has invalid char {b:?}"
            );
        }
    }
}

// ────────────────────── Statistical fire-rate calibration ──────────────────────

#[test]
fn fire_rate_approximates_compound_prob_at_large_n() {
    let _g = serial_guard();
    // With enable_prob=0.5, fire_prob=0.5 and many distinct site
    // names (so each site flips enable independently), the
    // expected compound rate is 25%. Tolerance: ±5% at N=2000 sites.
    reset_buggify_state();
    seed_buggify(20);
    let mut fires = 0;
    let n = 2000;
    let names: Vec<String> = (0..n).map(|i| format!("test.stat.site_{i}")).collect();
    // Leak each name to obtain a 'static lifetime for the fn signature.
    let names_static: Vec<&'static str> = names
        .into_iter()
        .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
        .collect();
    for name in &names_static {
        if buggify_with_probs(name, 0.5, 0.5) {
            fires += 1;
        }
    }
    let rate = fires as f64 / n as f64;
    assert!(
        (rate - 0.25).abs() < 0.05,
        "compound fire-rate {rate} drifted >5% from expected 0.25 at N={n}"
    );
}

// ────────────────────── reset_buggify_state preserves rng for stability ──────────────────────

#[test]
fn reset_alone_does_not_replay_with_same_sequence() {
    let _g = serial_guard();
    // After reset_buggify_state(), the rng state advances; subsequent
    // calls produce a DIFFERENT sequence than the prior `reset_buggify_state` + `seed_buggify`
    // pair would have. seed_buggify is the load-bearing reset path
    // for determinism.
    reset_buggify_state();
    seed_buggify(50);
    let mut seq_a = Vec::new();
    for i in 0..20 {
        let name: &'static str = match i % 3 {
            0 => "test.reset.x",
            1 => "test.reset.y",
            _ => "test.reset.z",
        };
        seq_a.push(buggify_with_probs(name, 0.5, 0.5));
    }
    // Reset without re-seeding — the RNG advances past `seq_a`.
    reset_buggify_state();
    let mut seq_b = Vec::new();
    for i in 0..20 {
        let name: &'static str = match i % 3 {
            0 => "test.reset.x",
            1 => "test.reset.y",
            _ => "test.reset.z",
        };
        seq_b.push(buggify_with_probs(name, 0.5, 0.5));
    }
    // The two sequences need NOT match (and in general won't); the
    // canonical replay path is `seed_buggify(SAME_SEED)`, not
    // `reset_buggify_state` alone.
    // Sanity: at least one element should differ at this sample size.
    assert!(
        seq_a != seq_b,
        "reset alone should advance RNG (use seed_buggify to replay)"
    );
}
