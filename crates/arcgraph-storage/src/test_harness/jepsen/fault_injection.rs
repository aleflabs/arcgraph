//! Fault-injection hooks for Jepsen-style isolation testing.
//!
//! ## Status at v0.1.0-alpha.0+1: API-shape only, no production consumer
//!
//! Per R1 review F-M4 (PR #344): the types in this module
//! ([`FaultInjectionContext`], [`jepsen_default_rates`],
//! [`no_op_rates`]) are scaffolding ahead of the v1.1 SIGKILL pipeline.
//! At v0.1.0-alpha.0+1 the production workload entry point
//! [`super::workload::run_bank_transfer`] takes no fault-injector
//! parameter; no committed code path invokes [`FaultInjectionContext::should_crash`]
//! or [`FaultInjectionContext::should_wal_fail`]. The unit tests below
//! exercise the adapter against itself, not against a consumer.
//!
//! The module ships now (rather than landing with v1.1) for two
//! reasons:
//!
//! 1. **Surface freeze** — the shape of the workload→K-1 binding is
//!    visible in the ADR-047 design discussion. Future implementers
//!    extending the harness can read the adapter and understand the
//!    intended K-1 producer / Jepsen consumer split without
//!    re-deriving it from prose.
//! 2. **Bounded-context contract** — locks in "the Jepsen module is
//!    the consumer; the K slices are the producers" so v1.1 work
//!    doesn't grow a parallel injection system inside `jepsen/`.
//!
//! Per `feedback_avoid_speculative_scaffolding.md`: this is
//! **acknowledged** scaffolding (the module rustdoc + ADR-047
//! §"Open questions" both say "consumed at v1.1"). Unacknowledged
//! scaffolding is the failure mode the discipline warns against;
//! the acknowledged variant is acceptable when (a) the shape is
//! load-bearing on a near-term consumer and (b) the lack of consumer
//! is documented in-line.
//!
//! ## Why the K-1 adapter is thin
//!
//! The K-1 module ([`super::super::k1`]) already provides the heavy
//! fault-injection machinery:
//!
//! - [`super::super::k1::injection::InjectionConfig`] — per-seam
//!   fault rates with deterministic XorShift RNG.
//! - [`super::super::k1::injection::InjectionDecisionRng`] — the
//!   per-op decision primitive.
//! - `super::super::k1::subprocess::SubprocessHandle` — the
//!   SIGKILL fork + restart harness.
//! - [`super::super::k1::oracle`] — the post-recovery oracle.
//!
//! This module is a **thin adapter layer** that:
//!
//! 1. Configures K-1's primitives with Jepsen-workload-appropriate
//!    defaults (lower rates than K-1's CI-smoke defaults because the
//!    workload is denser; a 1 % WAL fail rate at 400 ops/iteration
//!    fires 4× per iteration on average, which is the right ratio
//!    for catching recovery races without overwhelming the run with
//!    spurious aborts).
//! 2. Composes K-1's hooks with the [`super::history::OperationHistory`]
//!    recorder so faults that interrupt a commit produce
//!    `OpOutcome::Pending` records the checker can reconcile against
//!    post-recovery state.
//!
//! ## What lives here vs. K-1
//!
//! - **K-1**: the seam (WAL fsync, snapshot install, SIGKILL); the
//!   rate-based RNG; the recovery oracle.
//! - **Jepsen** (this module): the **workload-level integration** —
//!   how to bind a workload run to those seams, how the history
//!   recorder participates, when to invoke recovery.
//!
use std::sync::Arc;

use super::super::k1::injection::{InjectionConfig, InjectionDecisionRng};
use super::history::OperationHistory;

/// Per-seam fault rates tuned for the Jepsen bank-transfer workload.
///
/// Lower than the K-1 default ([`InjectionConfig::default`]) because
/// the workload is denser (400 ops vs. K-1's per-second cadence).
///
/// - `wal_failure_rate` = 0.0025 (0.25 % of WAL fsyncs fail — ~1 per
///   400-op iteration on average).
/// - `process_crash_rate` = 0.0005 (0.05 % per op — ~1 per 2000 ops,
///   so the SIGKILL variant fires roughly every 5th iteration).
/// - Other rates default to K-1's `no_op` (snapshot / background-fsync /
///   partial-write off — these seams aren't exercised by the
///   bank-transfer workload).
#[must_use]
pub fn jepsen_default_rates() -> InjectionConfig {
    InjectionConfig {
        wal_failure_rate: 0.0025,
        snapshot_failure_rate: 0.0,
        process_crash_rate: 0.0005,
        background_fsync_failure_rate: 0.0,
        wal_partial_write_rate: 0.0,
    }
    .validated()
}

/// All-zero rates — useful for the steady-state baseline run that
/// verifies the harness shape without injecting any faults. Identical
/// to [`InjectionConfig::no_op`] but exposed here so callers can
/// build a [`FaultInjectionContext`] without importing the K-1 type.
#[must_use]
pub fn no_op_rates() -> InjectionConfig {
    InjectionConfig::no_op()
}

/// Hook bound to a workload run. Decides whether to fire a fault on
/// each op and records the decision into the [`OperationHistory`].
///
/// Construct via [`FaultInjectionContext::new`]; pass to the workload
/// runner via [`FaultInjectionContext::should_crash`].
///
/// Not `Clone` — the inner [`InjectionDecisionRng`] is a `Mutex`-
/// guarded `XorShift` and the workload uses a single shared instance
/// across worker threads (per K-1's reproducibility argument).
pub struct FaultInjectionContext {
    config: InjectionConfig,
    rng: Arc<InjectionDecisionRng>,
    history: Arc<OperationHistory>,
}

impl FaultInjectionContext {
    /// Construct a fault-injection context bound to `history`.
    ///
    /// `seed` drives the deterministic decision RNG so a given
    /// `(config, seed, workload-seed)` triple produces the same
    /// fault sequence on replay. The workload's interleaving is
    /// still non-deterministic — the fault decisions are the only
    /// reproducible axis.
    #[must_use]
    pub fn new(config: InjectionConfig, seed: u64, history: Arc<OperationHistory>) -> Self {
        Self {
            config: config.validated(),
            rng: Arc::new(InjectionDecisionRng::new(seed)),
            history,
        }
    }

    /// Per-op decision: should this op trigger a process crash?
    /// Consumes one RNG decision from the shared `rng`.
    ///
    /// Note: this is the *workload-level* hook. The K-1 module fires
    /// the actual SIGKILL through its subprocess harness; this
    /// method tells the workload "the next op should run in a
    /// subprocess that will be killed mid-commit." The integration
    /// test glues the two together.
    pub fn should_crash(&self) -> bool {
        super::super::k1::injection::maybe_inject_process_crash(&self.config, &self.rng).is_some()
    }

    /// Per-op decision: should this op trigger a WAL fsync failure?
    ///
    /// `op_count` is informational only (the K-1 helper threads it
    /// through for telemetry but doesn't gate on it). We pass `0`
    /// because the Jepsen harness doesn't yet wire op-count
    /// reporting into the decision RNG.
    pub fn should_wal_fail(&self) -> bool {
        super::super::k1::injection::maybe_inject_wal_failure(&self.config, &self.rng, 0).is_some()
    }

    /// Borrow the bound history recorder. Useful when the caller
    /// wants to push an `OpOutcome::Pending` op right before a
    /// SIGKILL fires.
    #[must_use]
    pub fn history(&self) -> &Arc<OperationHistory> {
        &self.history
    }

    /// Borrow the injection config (read-only).
    #[must_use]
    pub fn config(&self) -> &InjectionConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_context_never_crashes() {
        let history = Arc::new(OperationHistory::new());
        let ctx = FaultInjectionContext::new(no_op_rates(), 42, history);
        for _ in 0..10_000 {
            assert!(!ctx.should_crash(), "no_op rates must never fire");
            assert!(!ctx.should_wal_fail(), "no_op rates must never fire");
        }
    }

    #[test]
    fn default_jepsen_rates_fire_within_expected_range() {
        // 400 trials × 0.0025 wal_failure_rate ≈ 1 expected firing.
        // Allow [0, 5] inclusive — wide tolerance because XorShift +
        // tail of binomial distribution.
        let history = Arc::new(OperationHistory::new());
        let ctx = FaultInjectionContext::new(jepsen_default_rates(), 0xCAFE, history);
        let mut wal_fires = 0;
        let mut crash_fires = 0;
        for _ in 0..400 {
            if ctx.should_wal_fail() {
                wal_fires += 1;
            }
            if ctx.should_crash() {
                crash_fires += 1;
            }
        }
        // 400 × 0.0025 = 1 expected; allow up to ~5 (5-sigma wide).
        assert!(
            wal_fires <= 5,
            "wal_fires={wal_fires} far above expected ~1"
        );
        // 400 × 0.0005 = 0.2 expected; allow up to 3.
        assert!(
            crash_fires <= 3,
            "crash_fires={crash_fires} far above expected ~0"
        );
    }

    #[test]
    fn same_seed_produces_same_fault_sequence() {
        let h1 = Arc::new(OperationHistory::new());
        let h2 = Arc::new(OperationHistory::new());
        let c1 = FaultInjectionContext::new(jepsen_default_rates(), 0xDEAD, h1);
        let c2 = FaultInjectionContext::new(jepsen_default_rates(), 0xDEAD, h2);
        let seq1: Vec<bool> = (0..500).map(|_| c1.should_crash()).collect();
        let seq2: Vec<bool> = (0..500).map(|_| c2.should_crash()).collect();
        assert_eq!(seq1, seq2, "same seed must produce same fault sequence");
    }

    #[test]
    fn rates_are_validated_on_construction() {
        // Caller passes out-of-range rates; constructor clamps to [0, 1].
        let cfg = InjectionConfig {
            wal_failure_rate: 2.5,       // → 1.0
            snapshot_failure_rate: -0.5, // → 0.0
            process_crash_rate: 0.5,
            background_fsync_failure_rate: 0.0,
            wal_partial_write_rate: 0.0,
        };
        let history = Arc::new(OperationHistory::new());
        let ctx = FaultInjectionContext::new(cfg, 1, history);
        let c = ctx.config();
        assert!((c.wal_failure_rate - 1.0).abs() < 1e-9);
        assert!((c.snapshot_failure_rate - 0.0).abs() < 1e-9);
        assert!((c.process_crash_rate - 0.5).abs() < 1e-9);
    }
}
