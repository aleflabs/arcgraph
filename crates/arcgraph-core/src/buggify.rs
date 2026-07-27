//! FoundationDB-style BUGGIFY macro for in-source fault injection.
//!
//! Per ADR-135 W26-γ-2 BUGGIFY MACRO. The `buggify!` macro lets
//! production code declare fault-injection sites inline; the macro is
//! a guaranteed no-op (literal `false`) in production builds, and a
//! two-level-conditional firing site in simulation builds.
//!
//! ## Two-level firing model (ADR-135 D-4)
//!
//! 1. **Per-run enabling** — the first call to `buggify!("<name>")` in
//!    a run flips a coin with `enable_prob` (default 25 %). The result
//!    caches in a `HashMap<&'static str, bool>`; subsequent calls to
//!    the SAME `name` use the cached enable.
//! 2. **Per-evaluation firing** — when an enabled site is evaluated,
//!    flip a second coin with `fire_prob` (default 25 %). Disabled sites
//!    return `false` unconditionally.
//!
//! Compound fire rate: `enable_prob × fire_prob = 6.25 %` per evaluation
//! across many seeds; some sites fire heavily within a single run
//! (sustained adversarial) while others never fire (per-run-disabled).
//! The two-level structure is FoundationDB-canonical (per Wilson et al.,
//! Strange Loop 2014).
//!
//! ## Production-build guarantee (ADR-135 D-3)
//!
//! The `buggify!` macro is cfg-gated on `arcgraph_sim`. Production
//! builds (no `--cfg arcgraph_sim` rustflag) expand the macro to a
//! literal `false`:
//!
//! ```text
//! #[cfg(not(arcgraph_sim))]
//! macro_rules! buggify { ($($tt:tt)*) => { false } }
//! ```
//!
//! The compiler dead-branch eliminates `if false { … }` at -O1+ so
//! production binary overhead is zero. Verified by the
//! `crates/arcgraph-core/tests/buggify_macro.rs` self-test that
//! compares `size_of` of equivalent functions with and without BUGGIFY
//! sites (production build).
//!
//! ## Determinism + DST integration (ADR-134 D-4 / ADR-135 §"Open questions")
//!
//! The per-run RNG is seeded from `ARCGRAPH_BUGGIFY_SEED` env var (if
//! present) OR via `seed_buggify(seed)` for direct DST runtime
//! integration. A given seed produces a deterministic enable + fire
//! sequence.

#![allow(dead_code)] // many fns are exposed only when `arcgraph_sim` is on

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Per-run BUGGIFY state. One global state; reset between sweep seeds via
/// [`reset_buggify_state`].
#[derive(Debug)]
struct BuggifyState {
    /// Per-site enable cache. `true` = site fires per-eval; `false` = site
    /// is unconditionally no-op for this run.
    site_enabled: HashMap<&'static str, bool>,
    /// Per-site fire count for observability + audit (per ADR-135 D-5).
    fire_count: HashMap<&'static str, u64>,
    /// Deterministic XorShift RNG state. Seeded by [`seed_buggify`] or
    /// (failing that) by the `ARCGRAPH_BUGGIFY_SEED` env var or (failing
    /// THAT) by a fixed default `0xDEADBEEF`. The fixed default keeps
    /// test runs reproducible even when callers forget to seed.
    rng: XorShift,
}

/// Tiny deterministic RNG. XorShift is the same primitive used by
/// `arcgraph_storage::test_harness::k1::injection::InjectionDecisionRng`;
/// matching the K-1 harness's RNG choice keeps the two harnesses
/// statistically comparable.
#[derive(Debug)]
struct XorShift {
    state: u64,
}

impl XorShift {
    fn new(seed: u64) -> Self {
        // XorShift can't take 0 as a seed; substitute a non-zero default.
        Self {
            state: if seed == 0 { 0xDEAD_BEEF } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform `f64` in `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        // Use the top 53 bits to build a uniform f64 in [0, 1).
        let bits = self.next_u64() >> 11;
        (bits as f64) / ((1u64 << 53) as f64)
    }
}

fn state() -> &'static Mutex<BuggifyState> {
    static STATE: OnceLock<Mutex<BuggifyState>> = OnceLock::new();
    STATE.get_or_init(|| {
        let seed = std::env::var("ARCGRAPH_BUGGIFY_SEED")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0xDEAD_BEEF);
        Mutex::new(BuggifyState {
            site_enabled: HashMap::new(),
            fire_count: HashMap::new(),
            rng: XorShift::new(seed),
        })
    })
}

/// Two-level BUGGIFY firing decision. Returns `true` with
/// `enable_prob × fire_prob` compound probability per call, modulo the
/// per-run-enable cache.
///
/// In production builds (no `arcgraph_sim` cfg) this function is never
/// reached — the macro expands to literal `false` and the compiler dead-
/// branch-eliminates the call site. In simulation builds, the function
/// is called inline per BUGGIFY site.
///
/// `enable_prob` and `fire_prob` are clamped to `[0.0, 1.0]`. Negative
/// values clamp to 0 (never fires); values above 1.0 clamp to 1.0
/// (always fires when reached). NaN clamps to 0 (defensive).
#[must_use]
pub fn buggify_with_probs(name: &'static str, enable_prob: f64, fire_prob: f64) -> bool {
    let enable_prob = clamp_prob(enable_prob);
    let fire_prob = clamp_prob(fire_prob);

    let mut state = state().lock().expect("buggify state mutex poisoned");

    // Per-run enable cache. If the site is not yet in the cache, flip
    // the enable coin and record the result. Subsequent calls to the
    // same site name reuse the cached enable.
    let enabled = match state.site_enabled.get(name) {
        Some(&b) => b,
        None => {
            let b = state.rng.next_f64() < enable_prob;
            state.site_enabled.insert(name, b);
            b
        }
    };

    if !enabled {
        return false;
    }

    // Per-evaluation fire.
    let fire = state.rng.next_f64() < fire_prob;
    if fire {
        *state.fire_count.entry(name).or_insert(0) += 1;
    }
    fire
}

/// Set the BUGGIFY RNG seed for deterministic-replay (ADR-134 D-3).
/// Also clears the per-run enable cache (since a new seed produces a
/// new enable distribution) and the per-site fire counts.
pub fn seed_buggify(seed: u64) {
    let mut state = state().lock().expect("buggify state mutex poisoned");
    state.rng = XorShift::new(seed);
    state.site_enabled.clear();
    state.fire_count.clear();
}

/// Reset BUGGIFY state between test runs. Clears the per-run enable
/// cache AND the per-site fire counts; preserves the current RNG state
/// (callers can re-seed via [`seed_buggify`] if they want a specific
/// seed).
pub fn reset_buggify_state() {
    let mut state = state().lock().expect("buggify state mutex poisoned");
    state.site_enabled.clear();
    state.fire_count.clear();
}

/// Returns a snapshot of per-site fire counts. Used by tests +
/// observability (per ADR-135 D-5).
#[must_use]
pub fn buggify_fire_counts() -> HashMap<&'static str, u64> {
    state()
        .lock()
        .expect("buggify state mutex poisoned")
        .fire_count
        .clone()
}

/// Returns the set of site names that have been EVALUATED (enabled or
/// not) since the last reset. Used by D-5 site-naming-convention tests.
#[must_use]
pub fn registered_sites() -> Vec<&'static str> {
    state()
        .lock()
        .expect("buggify state mutex poisoned")
        .site_enabled
        .keys()
        .copied()
        .collect()
}

#[inline]
fn clamp_prob(p: f64) -> f64 {
    if p.is_nan() || p < 0.0 {
        0.0
    } else if p > 1.0 {
        1.0
    } else {
        p
    }
}

/// FoundationDB-style BUGGIFY macro.
///
/// In `#[cfg(arcgraph_sim)]` builds, returns `true` with
/// `enable_prob × fire_prob` compound probability (per the two-level
/// firing model — see module docs).
///
/// In production builds (no `arcgraph_sim` cfg), expands to literal
/// `false`; the compiler dead-branch eliminates the call site.
///
/// # Examples
///
/// Single-arg (default 25% enable, 25% fire):
///
/// ```ignore
/// pub fn flush(&self) -> Result<()> {
///     if buggify!("storage.wal.fsync_pre_write") {
///         return Err(ArcGraphError::WalUnavailable);
///     }
///     // … real fsync code …
/// }
/// ```
///
/// Two-arg (custom fire probability, default enable 25%):
///
/// ```ignore
/// if buggify!("storage.snapshot.atomic_rename", 0.5) { panic!() }
/// ```
///
/// Three-arg (custom enable + fire probabilities):
///
/// ```ignore
/// if buggify!("storage.wal.append_delay", 0.1, 0.9) { sleep(rand_dur()) }
/// ```
///
/// # Naming convention (ADR-135 D-5)
///
/// Site names MUST follow `<crate>.<module>.<failure_mode>`:
///
/// ```text
/// storage.wal.fsync_pre_write
/// storage.checkpoint.install_crash
/// core.id.allocation_overflow
/// ```
///
/// Self-tests at `crates/arcgraph-core/tests/buggify_macro.rs` enforce
/// the convention via hostile-grep over registered site names.
#[cfg(arcgraph_sim)]
#[macro_export]
macro_rules! buggify {
    ($name:literal) => {
        $crate::buggify::buggify_with_probs($name, 0.25, 0.25)
    };
    ($name:literal, $fire_prob:expr) => {
        $crate::buggify::buggify_with_probs($name, 0.25, $fire_prob)
    };
    ($name:literal, $enable_prob:expr, $fire_prob:expr) => {
        $crate::buggify::buggify_with_probs($name, $enable_prob, $fire_prob)
    };
}

/// Production-build BUGGIFY macro — expands to literal `false`. The
/// compiler dead-branch eliminates the call site at -O1+.
#[cfg(not(arcgraph_sim))]
#[macro_export]
macro_rules! buggify {
    ($($tt:tt)*) => {
        false
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_prob_handles_nan_and_negatives() {
        assert_eq!(clamp_prob(f64::NAN), 0.0);
        assert_eq!(clamp_prob(-1.0), 0.0);
        assert_eq!(clamp_prob(0.5), 0.5);
        assert_eq!(clamp_prob(1.5), 1.0);
        assert_eq!(clamp_prob(0.0), 0.0);
        assert_eq!(clamp_prob(1.0), 1.0);
    }

    #[test]
    fn xorshift_is_deterministic() {
        let mut a = XorShift::new(42);
        let mut b = XorShift::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn xorshift_distinct_seeds_diverge() {
        let mut a = XorShift::new(1);
        let mut b = XorShift::new(2);
        // Across 1000 samples, divergent seeds must produce divergent
        // sequences (high probability; XorShift can't accidentally
        // collide for 1000 steps in a row at distinct seeds).
        let mut diverged = false;
        for _ in 0..1000 {
            if a.next_u64() != b.next_u64() {
                diverged = true;
                break;
            }
        }
        assert!(diverged, "distinct seeds must diverge");
    }

    #[test]
    fn xorshift_f64_in_unit_interval() {
        let mut rng = XorShift::new(7);
        for _ in 0..100_000 {
            let f = rng.next_f64();
            assert!((0.0..1.0).contains(&f), "f={f} outside [0, 1)");
        }
    }

    #[test]
    fn xorshift_seed_zero_substitutes() {
        // Seed 0 would brick XorShift; the constructor must substitute.
        let mut rng = XorShift::new(0);
        // Across many samples we'd see at least one non-zero output.
        let mut saw_nonzero = false;
        for _ in 0..100 {
            if rng.next_u64() != 0 {
                saw_nonzero = true;
                break;
            }
        }
        assert!(saw_nonzero, "seed 0 must not brick the RNG");
    }
}
