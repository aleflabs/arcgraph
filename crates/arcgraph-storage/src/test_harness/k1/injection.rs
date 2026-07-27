//! Per-op rate-based fault-injection API for K-1.
//!
//! ## Design (per spec D1, D2)
//!
//! Phase 5.5's torture harness (`tests/phase_5_5_torture.rs`) injects
//! faults on a **wall-clock cadence** (every N seconds). That model is
//! correct for a 30 s smoke run because per-op probability is flaky
//! under CPU contention at short durations. K-1 needs the
//! **per-op rate-based** model so multi-hour campaigns reach the
//! 1 % / 0.5 % / 0.1 % numbers asymptotically — exactly what the
//! ADR-038 amendment-03 §"Slice K" + amendment-02 PR #128 review
//! fold-in #5 framing pins.
//!
//! The decision is made by [`InjectionDecisionRng`], a deterministic
//! `XorShift` so a given `(config, op_count, seed)` triple produces a
//! reproducible fault sequence. Reproducibility is load-bearing for
//! debugging a K-3 multi-hour campaign that surfaces an ordering bug:
//! re-running with the same seed must reproduce the same fault
//! sequence (modulo the WAL fsync timing, which is non-deterministic
//! by design).
//!
//! ## InjectionKind taxonomy
//!
//! Five kinds; the harness picks one weighted by config rate:
//!
//! - `WalFsyncFail` — `WalHandle::flush()` returns `Err`; downstream
//!   commit ack returns failure (the caller observes a transaction
//!   abort). Recovery from any prior commit must still be intact.
//! - `WalPartialWrite` — the WAL bundle is truncated mid-record
//!   before fsync; `recover_from_wal` must detect the torn tail
//!   (per ADR-031 §R3 commit-bundle atomicity) and discard it.
//! - `SnapshotInstallFail` — `flush_snapshot_with_crash_point`
//!   crashes at a randomly-chosen `CrashPoint` (per G.2 §10.3
//!   atomic-rename graceful-artifact contract).
//! - `ProcessCrash` — the harness sends SIGKILL to the workload
//!   subprocess (see [`super::subprocess`]). Recovery happens from
//!   pure WAL replay (no Drop runs; no async cleanup; no panic
//!   handlers fire).
//! - `BackgroundFsyncFail` — drives `BackgroundFsyncScheduler`'s
//!   fail-action path (the periodic-tier T3 fsync surface). T3
//!   commits can be RPO-lost per ADR-034 D-2; the oracle accepts
//!   the loss within the rpo_ms window.
//!
//! ## Rate semantics
//!
//! Each rate is a probability in `[0.0, 1.0]`. The harness invokes
//! `maybe_inject_*` once per op; with `rate = 0.01` and a uniform-
//! random decision, ~1 % of calls return `Some(InjectionKind::*)`.
//! Calls returning `None` proceed normally.
//!
//! Default rates per spec D2:
//!
//! - `wal_failure_rate` = 0.01 (1 % of WAL fsyncs fail)
//! - `snapshot_failure_rate` = 0.005 (0.5 % of snapshot installs crash)
//! - `process_crash_rate` = 0.001 (0.1 % of ops trigger SIGKILL)
//! - `background_fsync_failure_rate` = 0.005 (0.5 % of T3 background
//!   fsyncs fail; rate matches snapshot because both are background-
//!   tier failures)
//! - `wal_partial_write_rate` = 0.0 (off by default; opt-in for
//!   torn-tail-specific campaigns since recovery semantics vary by
//!   FS)
//!
//! Rates above 1.0 saturate to 1.0 (always inject); below 0.0 saturate
//! to 0.0 (never inject). Negative rates pass through `validate()` as
//! a saturated-to-zero op rather than panicking — a "0 % fault rate"
//! config is the canonical no-op, and a misconfigured negative rate
//! is best surfaced as "no faults fired" + a warning rather than a
//! hard crash mid-campaign.
//!
//! ## Determinism + concurrency
//!
//! `InjectionDecisionRng` is a `Mutex<XorShift>` — concurrent ops
//! across worker threads serialize on the Mutex but the underlying
//! XorShift is fast (~1 ns per `next_u64`), so the lock cost is
//! negligible at K-1 op rates. The Mutex is necessary because
//! XorShift is not `Sync` and we need a SHARED rng so the global
//! fault count matches the configured rate (a per-thread rng would
//! produce N × the configured rate at N threads).
//!
//! ## Why not `rand::random()`?
//!
//! Adding `rand` as a dev-dep is fine, but `XorShift` is already in
//! use across the crate's tests (`phase_5_5_torture.rs`,
//! `mvcc_kernel_races.rs`, etc.) and is sufficient for fault-
//! injection-decision quality. It's deterministic + seedable + free
//! of FFI + zero added dep.

use std::sync::Mutex;

/// Per-seam fault rates. See module doc for semantics.
///
/// Construct via [`Self::default`] for the spec D2 baseline, or
/// [`Self::no_op`] for a no-faults sanity run, then override any
/// individual rate via the field accessors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InjectionConfig {
    /// Probability that a WAL fsync at the bundle-write seam fails
    /// (returns `Err`). Default 0.01 per spec D2.
    pub wal_failure_rate: f64,
    /// Probability that a snapshot install at the atomic-rename seam
    /// crashes at a random `CrashPoint`. Default 0.005.
    pub snapshot_failure_rate: f64,
    /// Probability that an op triggers SIGKILL via the subprocess
    /// harness. Default 0.001.
    pub process_crash_rate: f64,
    /// Probability that the background T3 fsync scheduler's
    /// scheduled-fsync fails. Default 0.005.
    pub background_fsync_failure_rate: f64,
    /// Probability that a WAL bundle write is truncated mid-record
    /// before fsync (torn-tail injection). Default 0.0 (off; opt-in).
    pub wal_partial_write_rate: f64,
}

impl Default for InjectionConfig {
    fn default() -> Self {
        Self {
            wal_failure_rate: 0.01,
            snapshot_failure_rate: 0.005,
            process_crash_rate: 0.001,
            background_fsync_failure_rate: 0.005,
            wal_partial_write_rate: 0.0,
        }
    }
}

impl InjectionConfig {
    /// All rates zero — useful for harness-shape verification (smoke
    /// run with no faults) and for the oracle's "no-injection-baseline"
    /// reference run.
    pub fn no_op() -> Self {
        Self {
            wal_failure_rate: 0.0,
            snapshot_failure_rate: 0.0,
            process_crash_rate: 0.0,
            background_fsync_failure_rate: 0.0,
            wal_partial_write_rate: 0.0,
        }
    }

    /// Saturate every rate to `[0.0, 1.0]` and return the validated
    /// copy. Negative rates clamp to 0.0; rates above 1.0 clamp to
    /// 1.0. Pure (does not mutate `self`).
    pub fn validated(&self) -> Self {
        let clamp = |r: f64| r.clamp(0.0, 1.0);
        Self {
            wal_failure_rate: clamp(self.wal_failure_rate),
            snapshot_failure_rate: clamp(self.snapshot_failure_rate),
            process_crash_rate: clamp(self.process_crash_rate),
            background_fsync_failure_rate: clamp(self.background_fsync_failure_rate),
            wal_partial_write_rate: clamp(self.wal_partial_write_rate),
        }
    }

    /// True iff every rate is ≤ 0.0 (no fault will ever fire).
    pub fn is_no_op(&self) -> bool {
        self.wal_failure_rate <= 0.0
            && self.snapshot_failure_rate <= 0.0
            && self.process_crash_rate <= 0.0
            && self.background_fsync_failure_rate <= 0.0
            && self.wal_partial_write_rate <= 0.0
    }
}

/// Five fault kinds. See module doc for semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InjectionKind {
    /// WAL fsync at bundle-write returns Err. Caller-side commit
    /// observes a transaction abort.
    WalFsyncFail,
    /// WAL bundle truncated mid-record before fsync. Recovery must
    /// detect the torn tail (ADR-031 §R3) and discard it.
    WalPartialWrite,
    /// Snapshot install crashes at a `CrashPoint`. Per G.2 §10.3 the
    /// next clean flush succeeds; recovery sees a graceful artifact.
    SnapshotInstallFail,
    /// Process receives SIGKILL. Recovery from pure WAL replay.
    ProcessCrash,
    /// Background T3 fsync fails. T3 commits within `rpo_ms` may be
    /// lost per ADR-034 D-2; the oracle accepts the loss inside the
    /// window.
    BackgroundFsyncFail,
}

/// Deterministic decision rng for injection. Wraps XorShift in a
/// Mutex so multiple worker threads can share it without producing
/// `N × rate` fault firings.
///
/// Reproducibility: `InjectionDecisionRng::new(seed)` produces the
/// same fault sequence for the same seed across runs. The
/// `(config, op_count, seed)` triple is the campaign manifest the
/// harness records on every K-1 run; replaying the manifest with the
/// same seed reproduces the same `Some(_) / None` decision sequence
/// at every `maybe_inject_*` call site.
pub struct InjectionDecisionRng {
    inner: Mutex<XorShift64>,
}

impl InjectionDecisionRng {
    /// Construct from a u64 seed. Seed `0` is replaced with a
    /// well-known constant so the rng never collapses into the
    /// identity stream — this matches the convention in
    /// `phase_5_5_torture.rs::XorShift::new`.
    pub fn new(seed: u64) -> Self {
        Self {
            inner: Mutex::new(XorShift64::new(seed)),
        }
    }

    /// Return `Some(kind)` with probability `rate`, else `None`.
    /// `kind` is the InjectionKind to fire if the roll succeeds.
    pub fn roll(&self, rate: f64, kind: InjectionKind) -> Option<InjectionKind> {
        if rate <= 0.0 {
            return None;
        }
        let r = rate.min(1.0);
        let v = self.next_unit_f64();
        if v < r { Some(kind) } else { None }
    }

    /// Pull a single uniform `[0.0, 1.0)` draw from the underlying
    /// rng. Used by [`maybe_inject_wal_failure`]'s codex L-6 single-
    /// roll threshold switch (replaces the prior 2-roll
    /// double-rng-draw pattern).
    pub fn next_unit_f64(&self) -> f64 {
        let mut guard = self
            .inner
            .lock()
            .expect("InjectionDecisionRng mutex poisoned");
        guard.next_unit_f64()
    }
}

/// `op_count` is informational — it identifies the call site that
/// fired but the decision is rate-based, not Nth-call-based. Per-op
/// schedules ("inject at op 137") are reserved for K-2 deterministic
/// reproducer campaigns; K-1 sticks with rate-based.
///
/// Reproducibility: callers who need a deterministic op-count → fault
/// mapping should use `InjectionDecisionRng::new(seed)` with the
/// same seed across runs; the rng's internal state is what produces
/// determinism, not the op_count argument.
///
/// ## Codex L-6 fix — single-roll threshold switch
///
/// Pre-fix this rolled the rng twice per call (once for partial-write,
/// once for fsync-fail). At call rates of N, the effective fsync-fail
/// rate was approximately `2 * wal_failure_rate * (1 - wal_partial_
/// write_rate)` — a 2× rate skew that K-2 rate tuning would trip over
/// silently. Post-fix: ONE rng roll per call, partitioned across
/// `[0, wal_partial_write_rate) ⇒ WalPartialWrite`,
/// `[wal_partial_write_rate, wal_partial_write_rate + wal_failure_rate)
/// ⇒ WalFsyncFail`, else `None`. Mathematically:
///
/// - `P(WalPartialWrite) = wal_partial_write_rate`
/// - `P(WalFsyncFail) = wal_failure_rate` (whenever
///   `wal_partial_write_rate + wal_failure_rate ≤ 1.0`)
/// - `P(None) = 1 - wal_partial_write_rate - wal_failure_rate`
///
/// The combined-rate clamp at 1.0 prevents the partial-write+fsync-fail
/// sum from exceeding 100%; if a campaign sets both rates above their
/// joint sum to `> 1.0` (eg 0.8 + 0.5 = 1.3), the WalFsyncFail rate
/// shrinks to fill `[wal_partial_write_rate, 1.0)` and the
/// "more severe" semantic (partial-write strictly subsumes fsync-fail)
/// is preserved.
pub fn maybe_inject_wal_failure(
    config: &InjectionConfig,
    rng: &InjectionDecisionRng,
    _op_count: u64,
) -> Option<InjectionKind> {
    let partial_rate = config.wal_partial_write_rate.clamp(0.0, 1.0);
    let fsync_rate = config.wal_failure_rate.clamp(0.0, 1.0);
    if partial_rate <= 0.0 && fsync_rate <= 0.0 {
        return None;
    }
    let v = rng.next_unit_f64();
    if v < partial_rate {
        Some(InjectionKind::WalPartialWrite)
    } else if v < (partial_rate + fsync_rate).min(1.0) {
        Some(InjectionKind::WalFsyncFail)
    } else {
        None
    }
}

/// `op_count` is informational — see [`maybe_inject_wal_failure`].
pub fn maybe_inject_snapshot_failure(
    config: &InjectionConfig,
    rng: &InjectionDecisionRng,
    _op_count: u64,
) -> Option<InjectionKind> {
    rng.roll(
        config.snapshot_failure_rate,
        InjectionKind::SnapshotInstallFail,
    )
}

/// Roll for a SIGKILL process crash on this op.
pub fn maybe_inject_process_crash(
    config: &InjectionConfig,
    rng: &InjectionDecisionRng,
) -> Option<InjectionKind> {
    rng.roll(config.process_crash_rate, InjectionKind::ProcessCrash)
}

/// Roll for a background T3 fsync failure.
pub fn maybe_inject_background_fsync_failure(
    config: &InjectionConfig,
    rng: &InjectionDecisionRng,
) -> Option<InjectionKind> {
    rng.roll(
        config.background_fsync_failure_rate,
        InjectionKind::BackgroundFsyncFail,
    )
}

/// Tally of fault firings across a campaign run. Used by the
/// recovery oracle to validate the harness fired SOME faults (a 0-
/// fault run is a harness regression, not a clean run).
#[derive(Debug, Default)]
pub struct InjectionTally {
    counts: Mutex<std::collections::HashMap<InjectionKind, u64>>,
}

impl InjectionTally {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&self, kind: InjectionKind) {
        let mut g = self.counts.lock().expect("InjectionTally poisoned");
        *g.entry(kind).or_insert(0) += 1;
    }

    pub fn count(&self, kind: InjectionKind) -> u64 {
        self.counts
            .lock()
            .expect("InjectionTally poisoned")
            .get(&kind)
            .copied()
            .unwrap_or(0)
    }

    pub fn total(&self) -> u64 {
        self.counts
            .lock()
            .expect("InjectionTally poisoned")
            .values()
            .sum()
    }

    pub fn snapshot(&self) -> std::collections::HashMap<InjectionKind, u64> {
        self.counts.lock().expect("InjectionTally poisoned").clone()
    }
}

// ──────────────────────────────────────────────────────────────────
// XorShift64 — internal, deterministic
// ──────────────────────────────────────────────────────────────────

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    const SEED_FALLBACK: u64 = 0xDEAD_BEEF_CAFE_F00D;

    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { Self::SEED_FALLBACK } else { seed },
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

    /// Uniform `[0.0, 1.0)` — extracts the top 53 bits (f64 mantissa).
    fn next_unit_f64(&mut self) -> f64 {
        // f64 has 52 mantissa bits; using 53 bits (incl. implicit 1)
        // gives a uniform in `[0, 1)` with no rounding bias.
        let bits = self.next_u64() >> 11; // 53-bit value
        (bits as f64) * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec_d2() {
        let c = InjectionConfig::default();
        assert!((c.wal_failure_rate - 0.01).abs() < 1e-9);
        assert!((c.snapshot_failure_rate - 0.005).abs() < 1e-9);
        assert!((c.process_crash_rate - 0.001).abs() < 1e-9);
    }

    #[test]
    fn no_op_config_is_no_op() {
        let c = InjectionConfig::no_op();
        assert!(c.is_no_op());
        let rng = InjectionDecisionRng::new(42);
        for op in 0..10_000 {
            assert!(maybe_inject_wal_failure(&c, &rng, op).is_none());
            assert!(maybe_inject_snapshot_failure(&c, &rng, op).is_none());
            assert!(maybe_inject_process_crash(&c, &rng).is_none());
            assert!(maybe_inject_background_fsync_failure(&c, &rng).is_none());
        }
    }

    #[test]
    fn validated_clamps_out_of_range() {
        let c = InjectionConfig {
            wal_failure_rate: -0.5,
            snapshot_failure_rate: 2.0,
            process_crash_rate: 0.0,
            background_fsync_failure_rate: 0.5,
            wal_partial_write_rate: 1.5,
        };
        let v = c.validated();
        assert_eq!(v.wal_failure_rate, 0.0);
        assert_eq!(v.snapshot_failure_rate, 1.0);
        assert_eq!(v.wal_partial_write_rate, 1.0);
        assert_eq!(v.background_fsync_failure_rate, 0.5);
    }

    #[test]
    fn rate_converges_to_configured() {
        // 10 % rate, 100 K rolls → expect ~10 K hits, allow ±2 σ
        // (σ ≈ √(N·p·(1-p)) ≈ 95 for N=100K, p=0.1).
        let c = InjectionConfig {
            wal_failure_rate: 0.1,
            snapshot_failure_rate: 0.0,
            process_crash_rate: 0.0,
            background_fsync_failure_rate: 0.0,
            wal_partial_write_rate: 0.0,
        };
        let rng = InjectionDecisionRng::new(0xCAFE_BABE);
        let mut hits = 0;
        for op in 0..100_000 {
            if maybe_inject_wal_failure(&c, &rng, op).is_some() {
                hits += 1;
            }
        }
        // expect 10K ± ~190 (2σ ≈ 190); use a generous ±500
        // tolerance so flakiness doesn't gate CI on RNG quirks.
        let diff = (hits as i64 - 10_000_i64).unsigned_abs();
        assert!(
            diff < 500,
            "rate convergence drift: got {hits} hits, expected ~10000 (diff {diff})"
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let c = InjectionConfig::default();
        let rng_a = InjectionDecisionRng::new(0xABCD);
        let rng_b = InjectionDecisionRng::new(0xABCD);
        for op in 0..1_000 {
            let a = maybe_inject_wal_failure(&c, &rng_a, op);
            let b = maybe_inject_wal_failure(&c, &rng_b, op);
            assert_eq!(
                a, b,
                "same seed must produce same fault sequence at op {op}"
            );
        }
    }

    #[test]
    fn interleaved_per_op_decision_sequence_deterministic() {
        // Codex L-5 pin: pre-fix `deterministic_with_same_seed`
        // exercised only one helper (`maybe_inject_wal_failure`) per
        // op. K-2 / K-3 campaigns interleave ALL 4 helpers (wal /
        // snapshot / process_crash / background_fsync) per op against
        // the SAME shared rng — the interleaved sequence's
        // determinism is what re-running a (seed, config) campaign
        // manifest with byte-equal output relies on.
        //
        // This test runs N=1000 ops where each op rolls every helper,
        // records the full Vec<(u8 helper_id, Option<InjectionKind>,
        // op)> sequence, then repeats with the same seed and asserts
        // byte-equal sequences.
        let c = InjectionConfig {
            // All non-zero so every helper has a chance to fire.
            wal_failure_rate: 0.05,
            wal_partial_write_rate: 0.05,
            snapshot_failure_rate: 0.05,
            process_crash_rate: 0.05,
            background_fsync_failure_rate: 0.05,
        };
        let collect = |seed: u64| -> Vec<(u8, Option<InjectionKind>, u64)> {
            let rng = InjectionDecisionRng::new(seed);
            let mut out = Vec::with_capacity(1_000 * 4);
            for op in 0..1_000u64 {
                out.push((0, maybe_inject_wal_failure(&c, &rng, op), op));
                out.push((1, maybe_inject_snapshot_failure(&c, &rng, op), op));
                out.push((2, maybe_inject_process_crash(&c, &rng), op));
                out.push((3, maybe_inject_background_fsync_failure(&c, &rng), op));
            }
            out
        };
        let a = collect(0xDECA_FBAD_u64);
        let b = collect(0xDECA_FBAD_u64);
        assert_eq!(
            a, b,
            "interleaved per-op decision sequence must be byte-equal across runs \
             with the same seed (campaign manifest replay precondition)"
        );
        // Sanity: at least one helper fired (else the test is vacuous —
        // a buggy implementation that returns None for everything would
        // also be "deterministic").
        let any_fired = a.iter().any(|(_, k, _)| k.is_some());
        assert!(
            any_fired,
            "no helper fired in 4000 rolls — rate too low or rng broken"
        );
    }

    #[test]
    fn tally_records_and_snapshots() {
        let t = InjectionTally::new();
        t.record(InjectionKind::WalFsyncFail);
        t.record(InjectionKind::WalFsyncFail);
        t.record(InjectionKind::SnapshotInstallFail);
        assert_eq!(t.count(InjectionKind::WalFsyncFail), 2);
        assert_eq!(t.count(InjectionKind::SnapshotInstallFail), 1);
        assert_eq!(t.total(), 3);
    }
}
