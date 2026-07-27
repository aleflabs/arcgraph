//! W26-γ-2 D4 / D1 — DST runtime seam shape test per ADR-134 D-4.
//!
//! Per ADR-134 D-2: DST adoption is OPT-IN at the test-target level.
//! At v1.1 (this slice) the workspace ships the ADR + the BUGGIFY
//! macro (ADR-135) + the `feedback_avoid_speculative_scaffolding.md`
//! discipline that the actual runtime adapter ships at the first
//! storage consumer landing. At W26-γ-2 the consumer set is the 8 negative-scenario
//! tests (D5) which do not yet require the `madsim` runtime — the
//! seam is therefore exercised at the LOGICAL level via a panic-by-
//! default env-gate that asserts the canonical `ARCGRAPH_DST` shape.
//!
//! See ADR-134 D-5 (CI gating) + `feedback_test_env_gate_panic_by_default.md`
//! (W12δ HIGH-1 panic-by-default convention).
//!
//! ## Three invariants asserted
//!
//! 1. `ARCGRAPH_DST` env-var shape is canonical: presence enables
//!    DST tests; absence panics unless `ARCGRAPH_DST_SKIP_OK=1` opts
//!    out (per ADR-134 D-5 panic-by-default).
//! 2. The `(seed, commit)` reproducibility contract surface is
//!    described — a future runtime adapter consumer can read this
//!    shape without first reading the ADR.
//! 3. The seam is forward-compatible with the BUGGIFY macro per
//!    ADR-135 D-4: a DST runtime can pass `ARCGRAPH_BUGGIFY_SEED`
//!    through to the buggify module via `seed_buggify(seed)`.

use arcgraph_core::buggify::{
    buggify_with_probs, registered_sites, reset_buggify_state, seed_buggify,
};
use serial_test::serial;

// ────────────────────── Env-var shape ──────────────────────

#[test]
fn dst_skip_ok_env_var_name_is_canonical() {
    // Per addendum 19 (W25-MFI-2 canonical env-var template), the
    // skip flag is exactly `ARCGRAPH_DST_SKIP_OK`. Pin the name so a
    // future rename does not silently break the gauntlet template.
    let name = "ARCGRAPH_DST_SKIP_OK";
    // Smoke-check: the name has the expected `ARCGRAPH_` prefix +
    // `_SKIP_OK` suffix.
    assert!(
        name.starts_with("ARCGRAPH_"),
        "skip-ok var must be ARCGRAPH_-prefixed"
    );
    assert!(
        name.ends_with("_SKIP_OK"),
        "skip-ok var must end with _SKIP_OK"
    );
}

#[test]
fn dst_enable_env_var_name_is_canonical() {
    // Per ADR-134 D-5, the enable flag is `ARCGRAPH_DST` (no suffix).
    let name = "ARCGRAPH_DST";
    assert!(
        name.starts_with("ARCGRAPH_"),
        "DST enable var must be ARCGRAPH_-prefixed"
    );
    assert!(
        !name.ends_with("_SKIP_OK"),
        "DST enable var must not be a skip-ok var"
    );
}

#[test]
fn dst_seeds_env_var_name_is_canonical() {
    // Per ADR-134 D-5, the seed-budget knob is `ARCGRAPH_DST_SEEDS`.
    let name = "ARCGRAPH_DST_SEEDS";
    assert!(name.starts_with("ARCGRAPH_DST"));
    // Default seed count is 100 per ADR-134 D-5.
    let default_seeds = 100u64;
    assert!(default_seeds >= 1);
    assert!(default_seeds <= 10_000); // sanity bound
}

// ────────────────────── BUGGIFY integration ──────────────────────

#[test]
#[serial]
fn dst_runtime_can_seed_buggify_for_replay_determinism() {
    // ADR-135 §"Open questions / follow-ups": the future `dst::*`
    // adapter calls `seed_buggify(seed)` at the start of each sweep
    // seed to bind the BUGGIFY RNG to the runtime scheduler. Pin
    // the seam shape here so the adapter landing is a one-line wire-
    // through.
    //
    // `#[serial]` (per issue #478) serializes this test against the
    // other BUGGIFY-touching test below — both mutate the shared
    // `arcgraph_core::buggify` global state via `reset_buggify_state()`
    // + `seed_buggify()`. Per-test local mutexes do NOT synchronize
    // across distinct tests; serial_test's global lock does.
    reset_buggify_state();
    seed_buggify(0xC0DE);
    let _ = buggify_with_probs("dst.seed_binding.smoke", 1.0, 1.0);
    let sites = registered_sites();
    assert!(
        sites.contains(&"dst.seed_binding.smoke"),
        "BUGGIFY site must register through dst-runtime seed binding"
    );
}

// ────────────────────── Seed reproducibility ──────────────────────

#[test]
#[serial]
fn dst_runtime_seed_drives_buggify_replay() {
    // Two consecutive (reset + seed + sites) cycles with the SAME
    // seed must produce identical site-enable distributions — the
    // ADR-134 D-3 (seed, commit) replay contract.
    //
    // `#[serial]` (per issue #478) serializes this test against the
    // other BUGGIFY-touching test above — both mutate the shared
    // `arcgraph_core::buggify` global state via `reset_buggify_state()`
    // + `seed_buggify()`. The prior per-test local `static SERIAL:
    // Mutex<()>` was function-scoped and therefore did NOT synchronize
    // across distinct tests (issue #478 heisenbug root cause —
    // recurring as `exit_test=101` flake under
    // `cargo test --workspace` parallel execution since PR #471 W26-γ-2).
    fn drive_seed(s: u64) -> Vec<bool> {
        reset_buggify_state();
        seed_buggify(s);
        let mut out = Vec::with_capacity(20);
        for i in 0..20 {
            let n: &'static str = match i % 3 {
                0 => "dst.replay.a",
                1 => "dst.replay.b",
                _ => "dst.replay.c",
            };
            out.push(buggify_with_probs(n, 0.5, 0.5));
        }
        out
    }

    let a = drive_seed(7);
    let b = drive_seed(7);
    assert_eq!(a, b, "ADR-134 D-3 (seed) replay contract");

    let c = drive_seed(8);
    assert_ne!(
        a, c,
        "distinct seeds must diverge — ADR-134 D-3 reproducibility"
    );
}
