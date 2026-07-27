//! K-1b — Multi-tenant workload generator (issue #214).
//!
//! ## Why a separate module
//!
//! K-1a's [`super::subprocess`] focuses on the SIGKILL fork harness +
//! single-tenant pre-crash ledger. K-1b extends to:
//!
//! - **N-tenant interleaved commits.** A single-process workload that
//!   issues commit ops across multiple tenants interleaved per a
//!   chosen [`Interleave`] strategy.
//! - **Per-tenant fault injection.** Each tenant carries its own
//!   [`InjectionConfig`]; the rates apply ONLY to that tenant's
//!   operations. A tenant absent from `per_tenant_injection`
//!   inherits [`InjectionConfig::no_op`].
//! - **Cross-tenant isolation.** A fault that fires for tenant A's op
//!   tears down + restarts the WAL stack (mirroring the K-1a 30 s
//!   smoke pattern). The workload's contract — verified by the
//!   K-1b cross-tenant oracle in [`super::oracle`] — is that
//!   tenants B + C's recovered state never contains a row that
//!   originated under tenant A.
//!
//! ## Determinism contract (extends K-1a)
//!
//! Per K-1a injection.rs, `(config, op_count, seed)` is the campaign
//! manifest the harness records on every K-1 run; replaying with the
//! same seed reproduces the same fault sequence. K-1b extends this:
//!
//! - **Per-tenant rng partition.** Every tenant gets its OWN
//!   `InjectionDecisionRng` keyed by `(injection_seed_base, tenant)`.
//!   Tenants with the SAME injection rate but DIFFERENT
//!   `(seed_base, tenant)` pairs produce independent fault
//!   sequences — preserving "1 % rate ⇒ ~1 % fault ratio" per
//!   tenant.
//! - **Tenant-selection rng.** A separate `XorShift64` (seeded with
//!   `workload_seed`) chooses the next tenant per
//!   [`Interleave::WeightedByRemaining`] /
//!   [`Interleave::RandomFromUniform`] / [`Interleave::RoundRobin`]
//!   (RoundRobin is deterministic without an rng).
//!
//! ## Hooks vs. production
//!
//! The fault path mirrors K-1a's [`super::injection`] — the harness
//! drives a WAL teardown + restart cycle from the test side via the
//! caller-supplied `restart_wal` closure. NO production source edit
//! is required to support K-1b (per K-1 mod.rs §"Hooks vs.
//! production").
//!
//! ## What this module is NOT
//!
//! - It is NOT a multi-process / SIGKILL workload (K-1a's
//!   `k1_subprocess_smoke.rs` covers SIGKILL recovery from a single
//!   tenant; K-1b extends multi-tenant in-thread WAL teardown +
//!   restart, which exercises the same recovery contract from a
//!   shared WAL).
//! - It is NOT a multi-FS variation (that's K-2 per ADR-038
//!   amendment-03 §"Slice K").
//! - It does NOT alter the [`super::oracle`]'s per-tenant
//!   invariants — those are unchanged. K-1b adds a NEW cross-tenant
//!   invariant in `super::oracle` checked via
//!   [`super::oracle::verify_cross_tenant_invariants`].

use std::collections::HashMap;

use arcgraph_core::TenantId;

use super::injection::{
    InjectionConfig, InjectionDecisionRng, InjectionKind, InjectionTally,
    maybe_inject_background_fsync_failure, maybe_inject_process_crash,
    maybe_inject_snapshot_failure, maybe_inject_wal_failure,
};
use super::subprocess::PreCrashLedger;

// ─────────────────────────────────────────────────────────────────
// Public surface
// ─────────────────────────────────────────────────────────────────

/// Strategy for interleaving N tenants' commits in a multi-tenant
/// workload.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interleave {
    /// Tenants are visited in declaration order, one commit per
    /// tenant per round, repeating until every tenant hits its
    /// per-tenant target. Fully deterministic — no rng draws.
    RoundRobin,
    /// Each round, the tenant with the HIGHEST remaining commits is
    /// preferred (ties broken by the tenant-selection rng). The
    /// effect is "all tenants finish at roughly the same time"
    /// regardless of per-tenant target sizes.
    WeightedByRemaining,
    /// Each round, a tenant is sampled uniformly at random from those
    /// still having commits to issue. Deterministic given
    /// `workload_seed`.
    RandomFromUniform,
}

/// Configuration for the multi-tenant workload.
///
/// Each `(tenant -> InjectionConfig)` entry applies its rates ONLY to
/// that tenant's operations. A tenant absent from
/// `per_tenant_injection` inherits [`InjectionConfig::no_op`] —
/// "this tenant is the clean control".
#[derive(Debug, Clone)]
pub struct MultiTenantWorkloadConfig {
    /// Tenants to interleave. Declaration order is the
    /// [`Interleave::RoundRobin`] cycle order.
    pub tenants: Vec<TenantId>,
    /// Per-tenant fault rates. Tenants absent from this map inherit
    /// [`InjectionConfig::no_op`].
    pub per_tenant_injection: HashMap<TenantId, InjectionConfig>,
    /// Per-tenant commit targets. Tenants absent from this map default
    /// to 0 commits (no-op for that tenant).
    pub commits_per_tenant: HashMap<TenantId, u64>,
    /// Tenant-selection strategy.
    pub interleave: Interleave,
    /// Seed for the tenant-selection rng (used by
    /// [`Interleave::WeightedByRemaining`] /
    /// [`Interleave::RandomFromUniform`]).
    pub workload_seed: u64,
    /// Base seed for per-tenant [`InjectionDecisionRng`]s. The
    /// per-tenant rng seed is `injection_seed_base ^ tenant.raw()`,
    /// so tenants with identical config but distinct `(seed_base,
    /// tenant)` produce independent fault sequences.
    pub injection_seed_base: u64,
    /// Safety net: cap on total commit attempts across all tenants.
    /// The workload terminates early if hit. Default 100 000.
    pub max_total_attempts: u64,
}

impl MultiTenantWorkloadConfig {
    /// Construct a baseline config with `tenants` interleaved
    /// round-robin and every tenant carrying [`InjectionConfig::no_op`]
    /// + the given `per_tenant_target` commits.
    ///
    /// Test code typically starts here, then overrides one tenant's
    /// `per_tenant_injection` to elevate its fault rate (the canonical
    /// K-1b cross-tenant pattern: tenant A faulty, tenants B + C
    /// clean).
    pub fn baseline(
        tenants: Vec<TenantId>,
        per_tenant_target: u64,
        workload_seed: u64,
        injection_seed_base: u64,
    ) -> Self {
        let mut per_injection = HashMap::with_capacity(tenants.len());
        let mut per_target = HashMap::with_capacity(tenants.len());
        for t in &tenants {
            per_injection.insert(*t, InjectionConfig::no_op());
            per_target.insert(*t, per_tenant_target);
        }
        Self {
            tenants,
            per_tenant_injection: per_injection,
            commits_per_tenant: per_target,
            interleave: Interleave::RoundRobin,
            workload_seed,
            injection_seed_base,
            max_total_attempts: 100_000,
        }
    }

    /// Override one tenant's [`InjectionConfig`]. Returns `Self` for
    /// builder chaining.
    pub fn with_injection(mut self, tenant: TenantId, config: InjectionConfig) -> Self {
        self.per_tenant_injection.insert(tenant, config);
        self
    }

    /// Override one tenant's per-tenant target. Returns `Self` for
    /// builder chaining.
    pub fn with_target(mut self, tenant: TenantId, commits: u64) -> Self {
        self.commits_per_tenant.insert(tenant, commits);
        self
    }

    /// Override the [`Interleave`] strategy.
    pub fn with_interleave(mut self, interleave: Interleave) -> Self {
        self.interleave = interleave;
        self
    }
}

/// One commit attempt's contract: the test-side `commit_op` closure
/// returns this on success, or `None` on commit failure (e.g.,
/// because the WAL is being torn down).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommitOutcome {
    pub node_id_raw: u64,
    pub label: u32,
    pub a: u32,
    pub b: u32,
    /// Durability tier as encoded by [`super::oracle::tier_to_u8`]
    /// (1 = Strict; 3 = Periodic).
    pub tier: u8,
}

/// Per-tenant counter snapshot from one workload run.
#[derive(Debug, Default, Clone)]
pub struct PerTenantReport {
    pub commits_attempted: u64,
    pub commits_acked: u64,
    pub commits_failed: u64,
    /// Number of injection rolls fired against this tenant's ops
    /// (some may have rolled `None` and produced no fault).
    pub injection_rolls: u64,
    /// Per-kind fault count for THIS tenant.
    pub fault_counts: HashMap<InjectionKind, u64>,
}

impl PerTenantReport {
    /// Sum of every fault kind for this tenant.
    pub fn total_faults(&self) -> u64 {
        self.fault_counts.values().sum()
    }
}

/// Aggregate report from one [`run_multi_tenant_workload`] run.
#[derive(Debug, Clone)]
pub struct MultiTenantWorkloadReport {
    pub per_tenant: HashMap<TenantId, PerTenantReport>,
    /// Total acked commits across every tenant.
    pub total_commits_acked: u64,
    /// Total faults across every tenant + every kind.
    pub total_faults: u64,
    /// Number of `restart_wal` invocations (one per fault that
    /// required a WAL teardown — `WalFsyncFail` /
    /// `WalPartialWrite` / `ProcessCrash`).
    pub wal_restarts: u64,
    /// Tenant-selection draws actually issued (≤ sum of
    /// `commits_per_tenant`; may be less if `max_total_attempts` was
    /// hit).
    pub total_attempts: u64,
}

/// Run the multi-tenant workload synchronously in the calling thread.
///
/// `commit_op` is called once per tenant op to issue a commit; it
/// returns `Some(outcome)` on success (the row is then recorded to
/// `ledger`) or `None` on failure (counted but not recorded). The
/// caller's closure is responsible for choosing the bytes (label /
/// a / b) — typically a per-tenant counter so tenant tagging is
/// deterministic.
///
/// `restart_wal` is invoked once per [`InjectionKind`] that requires
/// a WAL teardown (`WalFsyncFail` / `WalPartialWrite` /
/// `ProcessCrash`). The closure is expected to swap the underlying
/// stack (drain + recover) — the in-memory K1Stack pattern from the
/// K-1a 30 s smoke. `SnapshotInstallFail` / `BackgroundFsyncFail`
/// are tallied but do NOT trigger a restart at K-1b (those seams
/// land at K-1c+d).
pub fn run_multi_tenant_workload(
    config: &MultiTenantWorkloadConfig,
    ledger: &PreCrashLedger,
    mut commit_op: impl FnMut(TenantId) -> Option<CommitOutcome>,
    mut restart_wal: impl FnMut(),
) -> MultiTenantWorkloadReport {
    // Per-tenant injection rngs. Seeding rule:
    // `injection_seed_base ^ tenant.raw()` — distinct seeds per
    // tenant so identical configs don't produce identical fault
    // sequences.
    //
    // ## #219 CONCERN-2 carry-forward (PR #219 codex review)
    //
    // If `injection_seed_base == tenant.raw()` the per-tenant seed
    // collapses to 0 and `XorShift64::new` substitutes
    // `SEED_FALLBACK = 0xDEAD_BEEF_CAFE_F00D` (per
    // [`super::injection::XorShift64::new`]). Independence still holds
    // unless a SECOND tenant ALSO XORs to 0 (statistically vanishing
    // in a 64-bit space) — but a campaign that uses e.g. the build
    // commit hash as `injection_seed_base` could silently lose
    // per-tenant independence on accidental collision. The
    // `debug_assert!` below surfaces this loudly in dev/test builds;
    // K-2/K-3 multi-hour campaigns running release builds get the
    // doc-warning above.
    let mut per_tenant_rng: HashMap<TenantId, InjectionDecisionRng> =
        HashMap::with_capacity(config.tenants.len());
    for t in &config.tenants {
        let seed = config.injection_seed_base ^ t.raw();
        debug_assert!(
            seed != 0,
            "K-2 #219 CONCERN-2 carry-forward: per-tenant rng seed collision — \
             injection_seed_base {} XOR tenant {} = 0; determinism would be \
             defeated (XorShift64 substitutes SEED_FALLBACK and per-tenant \
             independence collapses to a shared stream). Pick a different \
             injection_seed_base.",
            config.injection_seed_base,
            t.raw()
        );
        per_tenant_rng.insert(*t, InjectionDecisionRng::new(seed));
    }

    // Per-tenant remaining-commits + per-tenant report scaffold.
    let mut remaining: HashMap<TenantId, u64> = HashMap::with_capacity(config.tenants.len());
    let mut report: HashMap<TenantId, PerTenantReport> =
        HashMap::with_capacity(config.tenants.len());
    for t in &config.tenants {
        let target = config.commits_per_tenant.get(t).copied().unwrap_or(0);
        remaining.insert(*t, target);
        report.insert(*t, PerTenantReport::default());
    }

    // Tenant-selection rng (separate from injection rngs).
    let mut select_rng = XorShift64::new(config.workload_seed);

    // Aggregate counters.
    let mut total_commits_acked: u64 = 0;
    let mut total_faults: u64 = 0;
    let mut wal_restarts: u64 = 0;
    let mut total_attempts: u64 = 0;
    // Round-robin cursor — incremented modulo tenants.len(). Skips
    // tenants whose remaining is 0.
    let mut rr_cursor: usize = 0;

    while total_attempts < config.max_total_attempts {
        // Find the next tenant per the strategy. If no tenant has
        // remaining > 0, terminate.
        let Some(tenant) = pick_next_tenant(
            &config.tenants,
            &remaining,
            config.interleave,
            &mut rr_cursor,
            &mut select_rng,
        ) else {
            break;
        };

        total_attempts += 1;
        let tenant_report = report.entry(tenant).or_default();
        tenant_report.commits_attempted += 1;
        // Decrement remaining BEFORE the op so a fault during this op
        // doesn't infinite-loop on the same tenant. The op MAY fail;
        // we still consumed the slot.
        if let Some(slot) = remaining.get_mut(&tenant) {
            *slot = slot.saturating_sub(1);
        }

        // Per-tenant injection roll. Tenants without an explicit
        // InjectionConfig inherit no_op() (no faults).
        let injection = config
            .per_tenant_injection
            .get(&tenant)
            .copied()
            .unwrap_or_else(InjectionConfig::no_op);
        let rng = per_tenant_rng
            .get(&tenant)
            .expect("per-tenant rng was seeded above for every config.tenants entry");

        // Roll across all four helpers (mirrors K-1a's
        // `interleaved_per_op_decision_sequence_deterministic` test —
        // every helper rolls per op for full determinism).
        tenant_report.injection_rolls += 1;
        let mut fired_kinds: Vec<InjectionKind> = Vec::new();
        if let Some(k) = maybe_inject_wal_failure(&injection, rng, total_attempts) {
            fired_kinds.push(k);
        }
        if let Some(k) = maybe_inject_snapshot_failure(&injection, rng, total_attempts) {
            fired_kinds.push(k);
        }
        if let Some(k) = maybe_inject_process_crash(&injection, rng) {
            fired_kinds.push(k);
        }
        if let Some(k) = maybe_inject_background_fsync_failure(&injection, rng) {
            fired_kinds.push(k);
        }

        // Tally faults BEFORE the commit attempt — a fault tells the
        // harness "treat this op as failing" + "tear down WAL" (for
        // the WAL/process-crash kinds). The commit itself is then
        // skipped.
        let mut commit_skipped_due_to_fault = false;
        for kind in &fired_kinds {
            *tenant_report.fault_counts.entry(*kind).or_insert(0) += 1;
            total_faults += 1;
            match kind {
                InjectionKind::WalFsyncFail
                | InjectionKind::WalPartialWrite
                | InjectionKind::ProcessCrash => {
                    // These kinds disrupt the WAL — restart it. The
                    // commit_op is skipped because the stack is being
                    // torn down.
                    commit_skipped_due_to_fault = true;
                    restart_wal();
                    wal_restarts += 1;
                }
                InjectionKind::SnapshotInstallFail | InjectionKind::BackgroundFsyncFail => {
                    // Tallied; no WAL teardown at K-1b. K-1c will
                    // wire snapshot / background-fsync failure paths
                    // through to a real graceful-artifact teardown.
                }
            }
        }

        if commit_skipped_due_to_fault {
            tenant_report.commits_failed += 1;
            continue;
        }

        // Issue the commit. The closure returns None on commit
        // failure (e.g., transient stack contention); we count
        // failure but don't escalate.
        match commit_op(tenant) {
            Some(outcome) => {
                tenant_report.commits_acked += 1;
                total_commits_acked += 1;
                // K-2 #219 NIT-1 carry-forward: escalate
                // `ledger.record` failures to a hard `panic!`.
                //
                // K-1b silently logged a tracing::error and
                // continued, leaving the row in the WAL but ABSENT
                // from `pre_crash.any_history` (since the ledger is
                // the oracle's pre-crash ground truth). On recovery
                // the row reappears under the recovered tenant — and
                // the oracle's ghost-direction iteration over
                // `recovered.bytes_by_key` fires a `UnknownKey`
                // violation that points at the symptom (recovered
                // bytes have no pre-crash entry) instead of the cause
                // (the ledger write silently failed).
                //
                // Ledger write failures at K-1+ are harness-level
                // signals (ENOSPC, permission, fsync error) — the
                // entire campaign is suspect when one fires. K-2's
                // contract: surface the harness bug LOUDLY at the
                // commit site, not as a downstream oracle false
                // alarm. A multi-tenant workload with hundreds of
                // commits per tenant cannot proceed correctly with a
                // half-recorded ledger.
                ledger
                    .record(
                        tenant.raw(),
                        outcome.node_id_raw,
                        outcome.label,
                        outcome.a,
                        outcome.b,
                        outcome.tier,
                    )
                    .unwrap_or_else(|e| {
                        panic!(
                            "K-2 hard escalation per #219 NIT-1 carry-forward: \
                             ledger.record failed for tenant {tenant_raw}, node \
                             {node_id_raw}, label {label}: {e}. Harness invariant \
                             violated; tests CANNOT proceed because the ledger is \
                             the oracle's pre-crash ground truth.",
                            tenant_raw = tenant.raw(),
                            node_id_raw = outcome.node_id_raw,
                            label = outcome.label,
                        )
                    });
            }
            None => {
                tenant_report.commits_failed += 1;
            }
        }
    }

    let tally_total = report.values().map(|r| r.total_faults()).sum::<u64>();
    debug_assert_eq!(
        tally_total, total_faults,
        "per-tenant fault counts must sum to the global counter"
    );

    MultiTenantWorkloadReport {
        per_tenant: report,
        total_commits_acked,
        total_faults,
        wal_restarts,
        total_attempts,
    }
}

/// Build a per-tenant [`InjectionTally`] view from a
/// [`MultiTenantWorkloadReport`]. Useful when callers want to
/// compare against a per-tenant K-1a oracle without manually
/// rebuilding the tally — the K-1b report carries richer per-tenant
/// state, so this is purely a view-side convenience.
pub fn per_tenant_tally(report: &MultiTenantWorkloadReport) -> HashMap<TenantId, InjectionTally> {
    let mut out: HashMap<TenantId, InjectionTally> =
        HashMap::with_capacity(report.per_tenant.len());
    for (tenant, per) in &report.per_tenant {
        let tally = InjectionTally::new();
        for (kind, count) in &per.fault_counts {
            for _ in 0..*count {
                tally.record(*kind);
            }
        }
        out.insert(*tenant, tally);
    }
    out
}

// ─────────────────────────────────────────────────────────────────
// Tenant-selection helpers
// ─────────────────────────────────────────────────────────────────

fn pick_next_tenant(
    tenants: &[TenantId],
    remaining: &HashMap<TenantId, u64>,
    strategy: Interleave,
    rr_cursor: &mut usize,
    rng: &mut XorShift64,
) -> Option<TenantId> {
    let any_remaining = tenants
        .iter()
        .any(|t| remaining.get(t).copied().unwrap_or(0) > 0);
    if !any_remaining {
        return None;
    }
    match strategy {
        Interleave::RoundRobin => {
            // Advance the cursor until we find a tenant with
            // remaining > 0. Bounded by tenants.len() iterations
            // because `any_remaining == true`.
            for _ in 0..tenants.len() {
                let idx = *rr_cursor % tenants.len();
                *rr_cursor = rr_cursor.wrapping_add(1);
                let t = tenants[idx];
                if remaining.get(&t).copied().unwrap_or(0) > 0 {
                    return Some(t);
                }
            }
            unreachable!("any_remaining was true but RR couldn't find a tenant")
        }
        Interleave::RandomFromUniform => {
            // Filter tenants with remaining > 0; pick one uniformly.
            let candidates: Vec<TenantId> = tenants
                .iter()
                .copied()
                .filter(|t| remaining.get(t).copied().unwrap_or(0) > 0)
                .collect();
            let idx = (rng.next_u64() as usize) % candidates.len();
            Some(candidates[idx])
        }
        Interleave::WeightedByRemaining => {
            // Pick the tenant with the highest remaining; ties broken
            // by an rng draw to avoid bias toward declaration order.
            let max_remaining = tenants
                .iter()
                .map(|t| remaining.get(t).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            let candidates: Vec<TenantId> = tenants
                .iter()
                .copied()
                .filter(|t| remaining.get(t).copied().unwrap_or(0) == max_remaining)
                .collect();
            let idx = (rng.next_u64() as usize) % candidates.len();
            Some(candidates[idx])
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// Internal XorShift64 (separate from the injection rng — this rng
// is for tenant SELECTION; the injection rngs decide FAULT firing.
// Mirroring the K-1a injection.rs internal XorShift64 keeps the
// determinism style consistent.)
// ─────────────────────────────────────────────────────────────────

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn tenant(raw: u64) -> TenantId {
        TenantId::new(raw)
    }

    fn baseline_config(tenants: &[TenantId], per_target: u64) -> MultiTenantWorkloadConfig {
        MultiTenantWorkloadConfig::baseline(
            tenants.to_vec(),
            per_target,
            0xC0FF_EE12_3456_7890,
            0xDECA_FBAD_CAFE_BABE,
        )
    }

    #[test]
    fn baseline_seeds_no_op_for_every_tenant() {
        let ts = vec![tenant(1), tenant(2), tenant(3)];
        let cfg = MultiTenantWorkloadConfig::baseline(ts.clone(), 10, 1, 2);
        for t in &ts {
            let inj = cfg.per_tenant_injection.get(t).copied().unwrap();
            assert!(inj.is_no_op(), "baseline tenant {t:?} must inherit no_op");
            assert_eq!(cfg.commits_per_tenant.get(t).copied(), Some(10));
        }
    }

    #[test]
    fn round_robin_visits_each_tenant_in_declaration_order() {
        let ts = vec![tenant(1), tenant(2), tenant(3)];
        let cfg = baseline_config(&ts, 4).with_interleave(Interleave::RoundRobin);
        let workdir = TempDir::new().unwrap();
        let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();

        let visit_order = std::cell::RefCell::new(Vec::<u64>::new());
        let mut node_counter = 0u64;
        let report = run_multi_tenant_workload(
            &cfg,
            &ledger,
            |t| {
                visit_order.borrow_mut().push(t.raw());
                node_counter += 1;
                Some(CommitOutcome {
                    node_id_raw: node_counter,
                    label: 100,
                    a: 11,
                    b: 22,
                    tier: 1,
                })
            },
            || panic!("baseline_config has no faults; restart_wal must NEVER be called"),
        );
        let order = visit_order.borrow().clone();
        // 4 commits per tenant × 3 tenants = 12 ops. RR cycles
        // (1, 2, 3, 1, 2, 3, ...).
        assert_eq!(order.len(), 12);
        for (i, raw) in order.iter().enumerate() {
            assert_eq!(*raw, ts[i % 3].raw(), "RR drift at op {i}: {order:?}");
        }
        assert_eq!(report.total_commits_acked, 12);
        assert_eq!(report.total_faults, 0);
        assert_eq!(report.wal_restarts, 0);
    }

    #[test]
    fn random_uniform_is_deterministic_per_seed() {
        // K-1b determinism contract pin: same workload_seed produces
        // the same tenant-selection sequence across runs. K-1a's
        // determinism extends to multi-tenant.
        let ts = vec![tenant(1), tenant(2), tenant(3)];
        let cfg = baseline_config(&ts, 10).with_interleave(Interleave::RandomFromUniform);

        let collect = || -> Vec<u64> {
            let workdir = TempDir::new().unwrap();
            let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();
            let visit_order = std::cell::RefCell::new(Vec::<u64>::new());
            let mut counter = 0u64;
            run_multi_tenant_workload(
                &cfg,
                &ledger,
                |t| {
                    visit_order.borrow_mut().push(t.raw());
                    counter += 1;
                    Some(CommitOutcome {
                        node_id_raw: counter,
                        label: 100,
                        a: 11,
                        b: 22,
                        tier: 1,
                    })
                },
                || {},
            );
            visit_order.borrow().clone()
        };
        let a = collect();
        let b = collect();
        assert_eq!(
            a, b,
            "same workload_seed must produce same tenant-selection sequence"
        );
        // Each tenant got exactly its target.
        for tid in &ts {
            let n = a.iter().filter(|r| **r == tid.raw()).count();
            assert_eq!(n, 10, "tenant {tid:?} target was 10; got {n}");
        }
    }

    #[test]
    fn per_tenant_injection_is_isolated_to_that_tenant() {
        // K-1b cross-tenant pin: tenant A has elevated WAL fault
        // rate; tenants B + C have NO injection rate. Faults must
        // ONLY land on tenant A's ops; B + C must have 0 faults.
        let ts = vec![tenant(1), tenant(2), tenant(3)];
        // 1.0 = saturating WAL-fail rate; every roll for tenant 1
        // fires a fault; tenants 2 + 3 stay at no_op (0 rate).
        let cfg = baseline_config(&ts, 5)
            .with_injection(
                tenant(1),
                InjectionConfig {
                    wal_failure_rate: 1.0,
                    ..InjectionConfig::no_op()
                },
            )
            .with_interleave(Interleave::RoundRobin);

        let workdir = TempDir::new().unwrap();
        let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();
        let mut counter = 0u64;
        let restart_count = std::cell::Cell::new(0u64);
        let report = run_multi_tenant_workload(
            &cfg,
            &ledger,
            |_t| {
                counter += 1;
                Some(CommitOutcome {
                    node_id_raw: counter,
                    label: 100,
                    a: 11,
                    b: 22,
                    tier: 1,
                })
            },
            || {
                restart_count.set(restart_count.get() + 1);
            },
        );
        let t1 = report.per_tenant.get(&tenant(1)).unwrap();
        let t2 = report.per_tenant.get(&tenant(2)).unwrap();
        let t3 = report.per_tenant.get(&tenant(3)).unwrap();
        assert!(
            t1.total_faults() > 0,
            "tenant 1 with rate=1.0 must have fired SOMETHING"
        );
        assert_eq!(t2.total_faults(), 0, "tenant 2 with rate=0 must NEVER fire");
        assert_eq!(t3.total_faults(), 0, "tenant 3 with rate=0 must NEVER fire");
        assert_eq!(
            report.wal_restarts,
            t1.total_faults(),
            "every WalFsyncFail on tenant 1 should trigger a restart"
        );
        assert_eq!(restart_count.get(), report.wal_restarts);
    }

    #[test]
    fn faults_with_saturating_rate_skip_commit_for_that_op() {
        // Pin: an op that fires a WAL fault has its commit_op SKIPPED
        // (the WAL is being torn down). The tenant's commits_failed
        // counter increments instead of commits_acked.
        let ts = vec![tenant(1)];
        let cfg = baseline_config(&ts, 5).with_injection(
            tenant(1),
            InjectionConfig {
                wal_failure_rate: 1.0,
                ..InjectionConfig::no_op()
            },
        );
        let workdir = TempDir::new().unwrap();
        let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();
        let commit_op_calls = std::cell::Cell::new(0u64);
        let report = run_multi_tenant_workload(
            &cfg,
            &ledger,
            |_t| {
                commit_op_calls.set(commit_op_calls.get() + 1);
                Some(CommitOutcome {
                    node_id_raw: 1,
                    label: 100,
                    a: 11,
                    b: 22,
                    tier: 1,
                })
            },
            || {},
        );
        let t1 = report.per_tenant.get(&tenant(1)).unwrap();
        assert_eq!(
            commit_op_calls.get(),
            0,
            "saturating WAL fault rate must skip every commit_op call"
        );
        assert_eq!(t1.commits_acked, 0);
        assert_eq!(t1.commits_failed, 5);
        assert_eq!(t1.commits_attempted, 5);
        assert_eq!(report.total_commits_acked, 0);
    }

    #[test]
    fn weighted_by_remaining_finishes_largest_target_last() {
        // WeightedByRemaining picks the tenant with the LARGEST
        // remaining. Setup: tenant A target 5, B target 10, C
        // target 1. A starts at 5, B at 10, C at 1; B is picked first
        // (max remaining=10), and over the run B's slots get drained
        // until parity with A. In the steady state, the strategy
        // visits the tenant with the largest remaining first.
        let ts = vec![tenant(10), tenant(20), tenant(30)];
        let cfg = baseline_config(&ts, 0)
            .with_target(tenant(10), 5)
            .with_target(tenant(20), 10)
            .with_target(tenant(30), 1)
            .with_interleave(Interleave::WeightedByRemaining);

        let workdir = TempDir::new().unwrap();
        let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();
        let mut counter = 0u64;
        let visit_order = std::cell::RefCell::new(Vec::<u64>::new());
        let report = run_multi_tenant_workload(
            &cfg,
            &ledger,
            |t| {
                visit_order.borrow_mut().push(t.raw());
                counter += 1;
                Some(CommitOutcome {
                    node_id_raw: counter,
                    label: 100,
                    a: 11,
                    b: 22,
                    tier: 1,
                })
            },
            || {},
        );

        // Sanity: every tenant hit its target.
        let n_a = visit_order.borrow().iter().filter(|r| **r == 10).count();
        let n_b = visit_order.borrow().iter().filter(|r| **r == 20).count();
        let n_c = visit_order.borrow().iter().filter(|r| **r == 30).count();
        assert_eq!(n_a, 5);
        assert_eq!(n_b, 10);
        assert_eq!(n_c, 1);
        assert_eq!(report.total_commits_acked, 16);
        // Strategy property: tenant B (target 10) is the FIRST visit
        // because it has the largest remaining at the start.
        assert_eq!(
            visit_order.borrow().first().copied(),
            Some(20),
            "WeightedByRemaining must visit the largest-remaining tenant first"
        );
    }

    #[test]
    fn max_total_attempts_terminates_runaway() {
        // Safety net: even if a configuration somehow doesn't
        // converge, max_total_attempts caps the loop. Build a
        // pathological case: every op is a fault that does NOT
        // decrement commits_acked, but we DO decrement remaining per
        // op so the loop terminates naturally; max_total_attempts
        // serves as a backstop.
        let ts = vec![tenant(1)];
        let mut cfg = baseline_config(&ts, 1_000_000); // huge target
        cfg.max_total_attempts = 100;
        let workdir = TempDir::new().unwrap();
        let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();
        let report = run_multi_tenant_workload(
            &cfg,
            &ledger,
            |_t| {
                Some(CommitOutcome {
                    node_id_raw: 1,
                    label: 100,
                    a: 11,
                    b: 22,
                    tier: 1,
                })
            },
            || {},
        );
        assert!(
            report.total_attempts <= 100,
            "max_total_attempts cap exceeded; got {}",
            report.total_attempts
        );
    }

    #[test]
    fn ledger_routes_per_tenant_under_real_workload() {
        // End-to-end pin: PreCrashLedger::PerTenantDir mode + the
        // multi-tenant workload generator → per-tenant CSVs each
        // contain ONLY that tenant's rows (no cross-pollution at the
        // ledger layer).
        let ts = vec![tenant(1), tenant(1001), tenant(1002)];
        let cfg = baseline_config(&ts, 7).with_interleave(Interleave::RoundRobin);
        let workdir = TempDir::new().unwrap();
        let ledger_dir = workdir.path().join("ledger");
        let ledger = PreCrashLedger::create_in_dir(&ledger_dir).unwrap();
        let mut counter = 0u64;
        let _ = run_multi_tenant_workload(
            &cfg,
            &ledger,
            |t| {
                counter += 1;
                let label_offset = match t.raw() {
                    1 => 100_000,
                    1001 => 200_000,
                    1002 => 300_000,
                    _ => 0,
                };
                Some(CommitOutcome {
                    node_id_raw: counter,
                    label: label_offset + counter as u32,
                    a: 11,
                    b: 22,
                    tier: 1,
                })
            },
            || {},
        );
        drop(ledger);

        // Each tenant's CSV has exactly 7 rows + every row's
        // tenant_raw matches the file name.
        for t in &ts {
            let rows = PreCrashLedger::read_for(&ledger_dir, t.raw()).unwrap();
            assert_eq!(
                rows.len(),
                7,
                "tenant {} expected 7 rows; got {}",
                t.raw(),
                rows.len()
            );
            assert!(
                rows.iter().all(|r| r.tenant_raw == t.raw()),
                "tenant {}'s CSV contains foreign tenant rows: {rows:?}",
                t.raw()
            );
        }
    }

    #[test]
    fn per_tenant_workload_is_deterministic_across_runs() {
        // K-1b determinism contract pin: same MultiTenantWorkloadConfig
        // (same seeds + same per-tenant injection + same interleave)
        // produces byte-identical per-tenant fault counts across
        // independent runs. K-1a injection.rs's
        // `interleaved_per_op_decision_sequence_deterministic` is the
        // single-tenant analog; this is its multi-tenant lift.
        //
        // The check is binary-equal across two runs (per
        // feedback_determinism_oracle_concurrency_tests.md — when the
        // underlying algorithm is deterministic, use binary-equal
        // reference snapshot as the assertion oracle).
        let ts = vec![tenant(1), tenant(2), tenant(3)];
        let cfg = baseline_config(&ts, 100)
            .with_injection(
                tenant(1),
                InjectionConfig {
                    wal_failure_rate: 0.10,
                    ..InjectionConfig::no_op()
                },
            )
            .with_injection(
                tenant(2),
                InjectionConfig {
                    wal_failure_rate: 0.05,
                    ..InjectionConfig::no_op()
                },
            )
            .with_interleave(Interleave::RoundRobin);

        let collect = || -> HashMap<u64, (u64, u64, u64)> {
            // Returns (commits_acked, commits_failed, total_faults) per
            // tenant raw id.
            let workdir = TempDir::new().unwrap();
            let ledger = PreCrashLedger::create_in_dir(workdir.path().join("ledger")).unwrap();
            let mut counter = 0u64;
            let report = run_multi_tenant_workload(
                &cfg,
                &ledger,
                |_t| {
                    counter += 1;
                    Some(CommitOutcome {
                        node_id_raw: counter,
                        label: 100,
                        a: 11,
                        b: 22,
                        tier: 1,
                    })
                },
                || {},
            );
            report
                .per_tenant
                .iter()
                .map(|(t, r)| {
                    (
                        t.raw(),
                        (r.commits_acked, r.commits_failed, r.total_faults()),
                    )
                })
                .collect()
        };
        let a = collect();
        let b = collect();
        assert_eq!(
            a, b,
            "same config (seeds + per-tenant injection + interleave) must produce \
             byte-equal per-tenant counters across runs"
        );
        // Sanity: tenant 1 (10 % rate) fired > tenant 2 (5 % rate) on
        // an expectation basis, and tenant 3 (no_op) fired 0.
        let t1 = a.get(&1).copied().unwrap();
        let t3 = a.get(&3).copied().unwrap();
        assert!(t1.2 > 0, "tenant 1 with rate=0.10 must have fired");
        assert_eq!(
            t3.2, 0,
            "tenant 3 inherits no_op (no entry in per_tenant_injection); must have 0 faults"
        );
    }
}
