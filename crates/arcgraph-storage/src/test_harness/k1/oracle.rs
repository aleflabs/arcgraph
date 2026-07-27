//! Recovery validation oracle for K-1.
//!
//! ## Contract (per spec D4)
//!
//! Post-recovery invariants the oracle enforces:
//!
//! 1. **All committed transactions are replayed.** Every entry in
//!    the pre-crash ledger that the workload tagged "T1 strict" is
//!    observable post-recovery via `read_node_with_store`. T3
//!    periodic entries within `rpo_ms` of the crash MAY be missing
//!    per ADR-034 D-2.
//! 2. **No partial transactions are visible.** The recovered store
//!    never returns bytes that no committed entry ever wrote
//!    (no I-V1 ghosts).
//! 3. **1:1 unique:total CRUD invariant preserved.** Each unique
//!    `(tenant, NodeId)` key appears exactly once in the recovered
//!    store, mirroring the Phase 5.5 baseline. (No duplicate-key
//!    visibility from a partially-applied recovery.)
//! 4. **870/870-style T1-strict-satisfied count preserved.** Of the
//!    keys with T1 history, the recovered store reproduces the
//!    LATEST T1 commit's bytes byte-identically. This is the
//!    PR #130 / issue #129 P0 contract from `phase_5_5_torture.rs`.
//! 5. **Catalog stats post-recovery match committed state.** Per
//!    M4-41, the per-tenant catalog stats are the cardinality
//!    backing for the M4-05 cost planner (M4-05 wires the trait;
//!    M4-31+ executor reads it). Per PR #170 reviewer Finding 1,
//!    persistence/recovery of stats becomes load-bearing at K-1
//!    oracle time. ADR-038 amendment-06 §D-25.1 ratified the
//!    cold-start rebuild architecture (option (a)) and the M4-41
//!    implementation slice landed it (this PR). The post-rebuild
//!    contract:
//!
//!    - Stats are NOT persisted at v1.0 alpha (no on-disk format
//!      ships per amendment-06 §D-25.1 properties). The MVCC
//!      primary store is the source of truth; the cold-start
//!      rebuild path
//!      ([`crate::recovery::stats_rebuild::rebuild_all_tenant_stats`])
//!      walks recovered MVCC at the recovered LSN and repopulates
//!      per-tenant `CatalogStats`. Synchronous at recovery time,
//!      BEFORE serving the first query (per amendment-06 §D-25.1
//!      step 4).
//!    - The commit-hook (`crud::commit` →
//!      `tenant_catalog_stats(tenant).increment_label /
//!      increment_total_nodes / observe_commit`) is INSIDE
//!      `crud::commit` — NOT inside the WAL `ReplayExecutor` (per
//!      amendment-06 locked invariant I-Q17). The cold-start
//!      rebuild path is what reconciles post-recovery stats with
//!      the recovered MVCC state; the rebuild module lives
//!      SEPARATELY from `wal::replay`.
//!    - The oracle compares: post-rebuild stats EQUAL what an
//!      in-process replay of the pre-crash ledger would compute,
//!      adjusted for rebuild semantics
//!      (`commits_observed = 1` per tenant from the single
//!      coalesced begin/observe bracket per amendment-06 §D-25.1
//!      step 2). The K-1 smokes' `build_committed_state` helpers
//!      encode this normalisation explicitly.
//!    - Both sides of the comparison unify on the rebuild-shape
//!      counter semantics: the rebuilt-from-ledger side emits
//!      `commits_observed = 1` per tenant (rebuild semantics) and
//!      the snapshot side reads
//!      [`CatalogStats::commits_observed_count`](crate::catalog::CatalogStats::commits_observed_count)
//!      `= 1` (the rebuild's coalesced bracket). Cardinality
//!      counts (`label_counts`, `rel_type_counts`, `total_nodes`,
//!      `total_rels`) match exactly: the rebuild walks every
//!      recovered MVCC node + rel record; under T1-only K-1
//!      workloads (the case for these smokes) every committed
//!      record survives recovery, so cardinality is preserved.
//!
//!    Pre-M4-41-implementation contract (still pinned by the unit
//!    test `tests::stats_consistency_check_with_multi_commit_tenant`)
//!    — the live commit-pipeline path. Each `crud::commit` call
//!    bumps `observe_commit()` once per touched tenant per
//!    transaction, so a 5-commit tenant's `commits_observed = 5`
//!    in steady state. The K-1 oracle's rebuild-semantics path
//!    only kicks in post-recovery (where the rebuild path's
//!    coalesced bracket is the source of truth).
//!
//! ## SnapshotState abstraction
//!
//! The oracle compares two `SnapshotState`s — a pre-crash one
//! (assembled from the pre-crash ledger) and a post-recovery one
//! (assembled by reading back from the recovered store). The oracle
//! is independent of how each `SnapshotState` is built; the smoke
//! tests in `tests/k1_*.rs` build them via `read_node_with_store` +
//! `catalog_stats` accessors.
//!
//! ## OracleViolation taxonomy
//!
//! Six violation kinds covering the five invariants above. Invariant 3
//! (1:1 unique:total) has TWO failure directions per the Phase 5.5
//! baseline: **drift** (key in ledger missing from recovered — caught
//! by `T1Missing`) + **ghost** (key in recovered with no row in the
//! pre-crash ledger — caught by `UnknownKey`, codex H-2 fix). Codex
//! L-4 noted the pre-fix module doc said "five invariants ↔ five
//! violation kinds 1:1" while the enum already had six variants —
//! pre-fix `UnknownKey` was defensively defined but never constructed
//! (regression on the Phase 5.5 contract).
//!
//! The oracle returns `Result<OracleReport, OracleViolation>` where
//! `OracleReport` carries the satisfied counts (for telemetry) and
//! `OracleViolation` carries the FIRST violation observed (with
//! enough context to debug). Future K-2 / K-3 may extend the taxonomy
//! with multi-FS-specific kinds.

use std::collections::{HashMap, HashSet};

use arcgraph_core::{DurabilityTier, LabelId, Lsn, NodeId, TenantId, TypeId};

use crate::catalog::CatalogStats;

/// Per-`(tenant, NodeId)` committed-state shadow for one workload.
///
/// `t1_history[(t, n)]` holds the LATEST T1 commit's bytes at that
/// key (or `None` if the key only ever saw T3 commits). `any_history
/// [(t, n)]` holds every historical commit at that key (for
/// I-V1 ghost-byte detection).
#[derive(Debug, Default, Clone)]
pub struct CommittedState {
    /// (tenant, node_id) → latest T1 commit bytes at this key.
    /// Absent ⇒ no T1 commit at this key.
    pub latest_t1: HashMap<(TenantId, NodeId), CommittedBytes>,
    /// (tenant, node_id) → set of every historical commit's bytes.
    /// Used to detect I-V1 ghost bytes (post-recovery store returns
    /// bytes that no committed entry ever wrote).
    pub any_history: HashMap<(TenantId, NodeId), HashSet<CommittedBytes>>,
    /// Total successful commits across all keys (incl. overwrites).
    pub total_commits: u64,
    /// Per-tenant stats: (LabelId → count, TypeId → count, total_nodes,
    /// total_rels, commits_observed).
    pub stats_by_tenant: HashMap<TenantId, CommittedStatsRebuild>,
}

/// Stats expected post-recovery, computed by replaying the pre-crash
/// ledger in-process. Matches the surface of [`CatalogStats`] so the
/// oracle can compare directly.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct CommittedStatsRebuild {
    pub label_counts: HashMap<LabelId, u64>,
    pub rel_type_counts: HashMap<TypeId, u64>,
    pub total_nodes: u64,
    pub total_rels: u64,
    pub commits_observed: u64,
}

/// Bytes triple (label, a, b). Mirrors the Phase 5.5 oracle's
/// `(u32, u32, u32)` history entry shape.
pub type CommittedBytes = (u32, u32, u32);

/// Post-recovery state read back from the recovered store. The
/// smoke tests build this by iterating every `(tenant, NodeId)` key
/// in the pre-crash committed state and calling
/// `read_node_with_store`.
#[derive(Debug, Default, Clone)]
pub struct RecoveredState {
    /// (tenant, NodeId) → bytes the recovered store returns.
    /// Absent ⇒ key not visible post-recovery.
    pub bytes_by_key: HashMap<(TenantId, NodeId), CommittedBytes>,
    /// Per-tenant `CatalogStats` snapshot. Built by reading
    /// the post-recovery `CrudStore::catalog_stats(tenant)` for
    /// every tenant the workload touched.
    pub stats_by_tenant: HashMap<TenantId, CommittedStatsRebuild>,
}

/// Successful oracle report. Carries the satisfied counts so the
/// caller can log a campaign summary.
#[derive(Debug, Clone)]
pub struct OracleReport {
    /// Number of unique `(tenant, NodeId)` keys in the pre-crash
    /// ledger.
    pub unique_keys: u64,
    /// Total commits recorded by the workload.
    pub total_commits: u64,
    /// Number of keys with at least one T1 commit.
    pub t1_keys: u64,
    /// Number of T1 keys whose latest T1 bytes match the recovered
    /// store's bytes.
    pub t1_satisfied: u64,
    /// Number of keys where the recovered store's bytes match SOME
    /// historical commit at that key.
    pub historical_match: u64,
    /// Number of T3 keys missing post-recovery (within ADR-034 D-2
    /// rpo tolerance).
    pub t3_rpo_lost: u64,
    /// Per-tenant stats consistency: `true` if recovered stats match
    /// rebuilt-from-ledger stats; populated for every tenant the
    /// workload touched.
    pub stats_consistent_by_tenant: HashMap<TenantId, bool>,
    /// Codex L-2 fix: every violation collected when running with
    /// `OracleConfig::fail_fast=false`. With the default
    /// (`fail_fast=true`) the oracle returns `Err(first violation)`
    /// and never reaches the `Ok(OracleReport { ... })` arm, so
    /// this field stays empty — it's reserved for K-2 collect-all
    /// callers that want every violation in one report instead of
    /// the first.
    pub violations: Vec<OracleViolation>,
}

/// Boxed payload for [`OracleViolation::StatsInconsistent`]. Holding
/// the two stats snapshots inline blew the enum's stack size past
/// clippy's `result-large-err` floor; boxing is the canonical fix.
#[derive(Debug, Clone)]
pub struct StatsInconsistentPayload {
    pub recovered: CommittedStatsRebuild,
    pub rebuilt: CommittedStatsRebuild,
}

/// Oracle violation taxonomy.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OracleViolation {
    /// I-V1 ghost: post-recovery store returns bytes that no
    /// historical commit ever wrote at that key.
    #[error(
        "I-V1 ghost: post-recovery store returned bytes {observed:?} for \
         (tenant {tenant_raw}, node {node_id_raw}) — no historical commit \
         ever wrote those bytes"
    )]
    GhostBytes {
        tenant_raw: u64,
        node_id_raw: u64,
        observed: CommittedBytes,
    },

    /// ADR-034 I-D1 / I-V6: T1 strict commit's bytes drifted
    /// post-recovery (allocator-gap regression or commit-bundle
    /// replay regression). Mirrors the Phase 5.5 assertion in
    /// `phase_5_5_torture.rs`.
    #[error(
        "ADR-034 I-D1 violation: T1 strict bytes drifted at \
         (tenant {tenant_raw}, node {node_id_raw}); observed {observed:?}, \
         expected latest T1 {expected:?}"
    )]
    T1StrictDrift {
        tenant_raw: u64,
        node_id_raw: u64,
        observed: CommittedBytes,
        expected: CommittedBytes,
    },

    /// ADR-034 I-D1: T1 strict commit missing post-recovery. T1 ack
    /// returns only after WAL fsync; recovery MUST reproduce the
    /// commit. Missing T1 bytes is a hard violation.
    #[error(
        "ADR-034 I-D1 violation: T1 strict key missing post-recovery: \
         (tenant {tenant_raw}, node {node_id_raw}); expected latest T1 {expected:?}"
    )]
    T1Missing {
        tenant_raw: u64,
        node_id_raw: u64,
        expected: CommittedBytes,
    },

    /// 1:1 unique:total invariant violated — recovered store
    /// returned bytes for a key the pre-crash ledger never wrote.
    /// This is a stricter version of GhostBytes (the key itself
    /// is unknown).
    #[error(
        "Phase-5.5 1:1 invariant violation: recovered store reports \
         {observed:?} bytes at (tenant {tenant_raw}, node {node_id_raw}) \
         but the pre-crash ledger has no commit at this key"
    )]
    UnknownKey {
        tenant_raw: u64,
        node_id_raw: u64,
        observed: CommittedBytes,
    },

    /// M4-41 / PR #170 Finding 1: post-recovery catalog stats do
    /// NOT match the rebuilt-from-ledger stats. Either the commit-
    /// hook is broken, OR stats persistence is missing and recovery
    /// doesn't rebuild them. K-1 surfaces; doesn't fix.
    ///
    /// The payload is boxed because [`CommittedStatsRebuild`] carries
    /// two `HashMap`s and the unboxed variant dominates the enum's
    /// stack size by an order of magnitude (clippy
    /// `large-enum-variant` / `result-large-err`). Every other
    /// `OracleViolation` variant fits in ≤ 64 bytes.
    #[error("M4-41 stats inconsistency at tenant {tenant_raw}: see boxed payload")]
    StatsInconsistent {
        tenant_raw: u64,
        payload: Box<StatsInconsistentPayload>,
    },

    /// T3 RPO loss exceeded the ADR-034 D-2 tolerance. v1.0 default
    /// tolerance is 20 % per the Phase 5.5 oracle's `recovery_rate
    /// >= 0.80` floor; configurable via [`OracleConfig::t3_rpo_floor`].
    #[error(
        "ADR-034 D-2 violation: T3 RPO loss {observed_rate:.4} exceeds floor \
         {floor:.4}; {found}/{total} keys recovered"
    )]
    T3RpoLossExceeded {
        observed_rate: f64,
        floor: f64,
        found: u64,
        total: u64,
    },

    /// K-1b cross-tenant contamination (issue #214): a row whose bytes
    /// were committed under tenant `source` appears in tenant
    /// `target`'s recovered state at NodeId `key`. The canonical
    /// detection: tenant `target`'s recovered store returns bytes at
    /// `(target, key)` that:
    ///
    /// - exist in tenant `source`'s pre-crash any_history at NodeId
    ///   `key`, AND
    /// - are NOT in tenant `target`'s OWN any_history at NodeId
    ///   `key` (so target never wrote them).
    ///
    /// This rules out coincidental NodeId aliasing (two tenants both
    /// committing different rows at NodeId 5 is fine — the
    /// contamination signal requires the BYTES to come from `source`'s
    /// commits).
    ///
    /// The `lsn` field is reserved for K-2 / K-3 LSN-aware ledger
    /// formats; at K-1b the variant carries [`Lsn::ZERO`] because the
    /// K-1 ledger format does not record per-row LSN. K-2's binary
    /// format will populate this with the actual commit LSN.
    #[error(
        "K-1b cross-tenant contamination: bytes committed under tenant \
         {source_raw} appear in tenant {target_raw}'s recovered state \
         at node {key_raw} (lsn={lsn_raw}). Storage layer mis-tagged a \
         tenant's row, OR recovery replayed a row under the wrong tenant."
    )]
    CrossTenantContamination {
        source_raw: u64,
        target_raw: u64,
        key_raw: u64,
        lsn_raw: u64,
    },

    /// K-1b defensive: the oracle's per-tenant ledger map is missing
    /// an entry for a tenant the harness expected to find. Distinct
    /// from "tenant has 0 commits" — this fires when the ledger
    /// loading code returned no entry at all (e.g., the per-tenant
    /// CSV file is absent). Test-side ledger reconstruction uses this
    /// to surface a configuration mismatch loudly instead of silently
    /// treating the tenant as "had no commits".
    #[error(
        "K-1b ledger gap: pre-crash ledger has no entry for tenant {tenant_raw} \
         — expected at least an empty CSV. Possible test misconfiguration: \
         the workload may have skipped this tenant or the workdir is wrong."
    )]
    LedgerNotFoundForTenant { tenant_raw: u64 },
}

/// Oracle configuration knobs.
#[derive(Debug, Clone)]
pub struct OracleConfig {
    /// Minimum fraction of pre-crash ledger entries that MUST be
    /// observable post-recovery. Default 0.80 — matches the Phase
    /// 5.5 floor; rationale: ADR-034 D-2 RPO loss is bounded by
    /// `rpo_ms` and the test fault rate.
    pub t3_rpo_floor: f64,
    /// Whether to fail-fast on the first violation (default `true`)
    /// or report every violation in `OracleReport`. Fail-fast keeps
    /// the error message short; collect-all is useful when the
    /// harness regresses and the parent wants the full picture.
    ///
    /// Codex L-2 fix: pre-fix this field was documented but unused.
    /// Now: per-violation `return Err(...)` paths short-circuit only
    /// when `fail_fast=true`. With `fail_fast=false`, the oracle
    /// continues past the first violation (current K-1a non-stats
    /// violation paths still exit early on first hit; this knob
    /// reserves the name for K-2 collect-all callers and is wired
    /// for the stats-consistency block today).
    pub fail_fast: bool,
    /// Whether stats inconsistency is fatal. The M4-41
    /// implementation slice (per ADR-038 amendment-06 §3 R1) flipped
    /// every K-1 smoke to `true` because the cold-start MVCC rebuild
    /// path
    /// ([`crate::recovery::stats_rebuild::rebuild_all_tenant_stats`])
    /// repopulates per-tenant `CatalogStats` from the recovered MVCC
    /// state, making the strict-mode oracle non-vacuous. The `false`
    /// arm remains for K-2 / K-3 collect-all long-running campaign
    /// callers that want to accumulate every violation rather than
    /// fail-fast (mirrors the `fail_fast` knob's semantics for the
    /// stats-consistency block).
    pub stats_inconsistency_fatal: bool,
    /// `rpo_ms` parameter to apply when `u8_to_tier` decodes a
    /// `Periodic` row from the pre-crash ledger. Codex L-1 fix:
    /// pre-fix this was hardcoded to 1_000 ms — campaigns testing
    /// tighter (or looser) RPO budgets had no path to override the
    /// decoded tier shape. Default 1_000 ms (matches Phase 5.5 +
    /// the prior hardcoded value); K-1c will sweep tighter values
    /// under the rate-injection campaign.
    pub t3_rpo_ms: u64,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            t3_rpo_floor: 0.80,
            fail_fast: true,
            stats_inconsistency_fatal: true,
            t3_rpo_ms: 1_000,
        }
    }
}

/// The oracle entry point. Compares pre-crash committed state vs.
/// recovered state under `config` knobs.
///
/// Returns `Ok(OracleReport)` on success or `Err(OracleViolation)`
/// on the first violation (per `fail_fast`).
pub fn verify_post_recovery_invariants(
    pre_crash: &CommittedState,
    recovered: &RecoveredState,
    config: &OracleConfig,
) -> Result<OracleReport, OracleViolation> {
    let mut t1_keys: u64 = 0;
    let mut t1_satisfied: u64 = 0;
    let mut historical_match: u64 = 0;
    let mut t3_rpo_lost: u64 = 0;
    // Codex L-2 fix: the `fail_fast` knob is now load-bearing. With
    // `fail_fast=true` (default) the macro early-returns `Err(first)`
    // — preserves the prior behavior 1:1. With `fail_fast=false`
    // every violation is pushed into `violations` and surfaced on
    // the `Ok(OracleReport { ..., violations })` path so K-2
    // collect-all callers see the full picture in one report.
    let mut violations: Vec<OracleViolation> = Vec::new();

    // Local helper: push + maybe-short-circuit. Inline-style closure
    // over `&mut violations` + `config.fail_fast` would borrow-check
    // poorly across the iteration loops; we use a macro instead.
    macro_rules! record_violation {
        ($v:expr) => {{
            let v = $v;
            if config.fail_fast {
                return Err(v);
            }
            violations.push(v);
        }};
    }

    let total_keys = pre_crash.any_history.len() as u64;

    for ((tenant, node_id), history_set) in &pre_crash.any_history {
        let key = (*tenant, *node_id);
        let observed_opt = recovered.bytes_by_key.get(&key).copied();
        let latest_t1 = pre_crash.latest_t1.get(&key).copied();

        match (observed_opt, latest_t1) {
            (Some(observed), Some(t1_bytes)) => {
                t1_keys += 1;
                // I-V1 ghost: observed must match SOME historical
                // commit at this key.
                if !history_set.contains(&observed) {
                    record_violation!(OracleViolation::GhostBytes {
                        tenant_raw: tenant.raw(),
                        node_id_raw: node_id.raw(),
                        observed,
                    });
                    continue;
                }
                historical_match += 1;
                // T1 strict: must equal latest T1 bytes.
                if observed == t1_bytes {
                    t1_satisfied += 1;
                } else {
                    record_violation!(OracleViolation::T1StrictDrift {
                        tenant_raw: tenant.raw(),
                        node_id_raw: node_id.raw(),
                        observed,
                        expected: t1_bytes,
                    });
                }
            }
            (Some(observed), None) => {
                // No T1 history; bytes must still match SOME
                // historical commit (T3 last-write-wins or earlier
                // T1 superseded by a later T3).
                if !history_set.contains(&observed) {
                    record_violation!(OracleViolation::GhostBytes {
                        tenant_raw: tenant.raw(),
                        node_id_raw: node_id.raw(),
                        observed,
                    });
                    continue;
                }
                historical_match += 1;
            }
            (None, Some(t1_bytes)) => {
                // T1 commit missing post-recovery — hard violation.
                record_violation!(OracleViolation::T1Missing {
                    tenant_raw: tenant.raw(),
                    node_id_raw: node_id.raw(),
                    expected: t1_bytes,
                });
            }
            (None, None) => {
                // T3-only key, RPO-lost. Counted toward the rpo
                // floor below.
                t3_rpo_lost += 1;
            }
        }
    }

    // 1:1 unique:total ghost direction (codex H-2). The 1:1
    // invariant has TWO failure modes by symmetry:
    // - drift: a key in pre_crash.any_history is missing from
    //   recovered (caught above by the (None, _) match arms).
    // - ghost: a key in recovered has NO row in pre_crash.any_history
    //   (the recovered store has materialised a key that NO commit
    //   ever wrote — partial-recovery / replay-divergence regression).
    // Phase 5.5 (tests/phase_5_5_torture.rs / PR #130) catches both.
    // Pre-codex H-2 the K-1 oracle defensively defined UnknownKey
    // but iterated only pre_crash.any_history — the ghost direction
    // was unenforced, a silent regression on the Phase 5.5 contract.
    for (recovered_key, observed_bytes) in &recovered.bytes_by_key {
        if !pre_crash.any_history.contains_key(recovered_key) {
            record_violation!(OracleViolation::UnknownKey {
                tenant_raw: recovered_key.0.raw(),
                node_id_raw: recovered_key.1.raw(),
                observed: *observed_bytes,
            });
        }
    }

    // T3 RPO floor.
    if total_keys > 0 {
        let recovery_rate = historical_match as f64 / total_keys as f64;
        if recovery_rate < config.t3_rpo_floor {
            record_violation!(OracleViolation::T3RpoLossExceeded {
                observed_rate: recovery_rate,
                floor: config.t3_rpo_floor,
                found: historical_match,
                total: total_keys,
            });
        }
    }

    // M4-41 stats consistency.
    let mut stats_consistent_by_tenant: HashMap<TenantId, bool> = HashMap::new();
    for (tenant, rebuilt) in &pre_crash.stats_by_tenant {
        let recovered_stats = recovered
            .stats_by_tenant
            .get(tenant)
            .cloned()
            .unwrap_or_default();
        let consistent = &recovered_stats == rebuilt;
        stats_consistent_by_tenant.insert(*tenant, consistent);
        if !consistent && config.stats_inconsistency_fatal {
            record_violation!(OracleViolation::StatsInconsistent {
                tenant_raw: tenant.raw(),
                payload: Box::new(StatsInconsistentPayload {
                    recovered: recovered_stats,
                    rebuilt: rebuilt.clone(),
                }),
            });
        }
    }

    Ok(OracleReport {
        unique_keys: total_keys,
        total_commits: pre_crash.total_commits,
        t1_keys,
        t1_satisfied,
        historical_match,
        t3_rpo_lost,
        stats_consistent_by_tenant,
        violations,
    })
}

// ─────────────────────────────────────────────────────────────────
// K-1b cross-tenant invariants (issue #214)
// ─────────────────────────────────────────────────────────────────

/// Per-tenant pre-crash + recovered states bundled for the K-1b
/// cross-tenant oracle. Each tenant is keyed by its [`TenantId`] and
/// carries the same single-tenant shape as
/// [`verify_post_recovery_invariants`]'s inputs.
#[derive(Debug, Default, Clone)]
pub struct CrossTenantOracleInput {
    pub pre_crash_per_tenant: HashMap<TenantId, CommittedState>,
    pub recovered_per_tenant: HashMap<TenantId, RecoveredState>,
}

/// Aggregate report from the K-1b cross-tenant oracle. The
/// per-tenant single-tenant report is keyed by the tenant; the
/// cross-tenant violation list (empty on success) carries every
/// `CrossTenantContamination` / `LedgerNotFoundForTenant` firing.
#[derive(Debug, Clone)]
pub struct CrossTenantOracleReport {
    /// Per-tenant single-tenant report. Built by running
    /// [`verify_post_recovery_invariants`] for each tenant.
    pub per_tenant: HashMap<TenantId, OracleReport>,
    /// Cross-tenant invariant violations (empty on success).
    pub cross_tenant_violations: Vec<OracleViolation>,
    /// Tenants the oracle iterated over (the union of pre/rec keys).
    pub tenants_checked: Vec<TenantId>,
    /// Number of `(source, target, key)` triples checked across all
    /// tenant pairs.
    pub cross_tenant_checks: u64,
}

/// K-1b cross-tenant oracle (issue #214).
///
/// Runs the per-tenant [`verify_post_recovery_invariants`] for each
/// tenant + an additional cross-tenant invariant: NO row whose bytes
/// were committed under tenant A appears in tenant B's recovered
/// state at the same NodeId UNLESS B's own pre-crash any_history
/// also contains those bytes.
///
/// ## Why "bytes also exist in source's ledger" is the canonical signal
///
/// NodeIds are allocated per-tenant (the per-tenant counter pattern in
/// `CrudStore`). Two tenants both committing rows at NodeId 5 is
/// COMPLETELY EXPECTED — they share the integer NodeId space but live
/// in disjoint `(TenantId, NodeId)` keyspaces in storage. The
/// cross-tenant invariant cannot fire on "same NodeId in both" alone
/// (that's a coincidence, not a bug). The contamination signal is
/// "tenant B's recovered state at NodeId X has BYTES that appear ONLY
/// in tenant A's pre-crash ledger at the same NodeId". This is
/// false-positive-free as long as test workloads ensure tenants commit
/// distinct bytes (e.g., via per-tenant disjoint label spaces).
///
/// ## fail_fast semantics
///
/// Mirrors [`verify_post_recovery_invariants`]: with `fail_fast=true`
/// (default) the oracle returns `Err(first violation)` on the first
/// per-tenant OR cross-tenant violation; with `fail_fast=false` every
/// violation is collected into `cross_tenant_violations` (the
/// per-tenant violations remain inside each `OracleReport`).
///
/// ### K-2 caller discipline (#219 CONCERN-3 carry-forward)
///
/// **To distinguish [`OracleViolation::CrossTenantContamination`] from
/// per-tenant [`OracleViolation::UnknownKey`] violations, callers MUST
/// use `fail_fast=false` and inspect `cross_tenant_violations`
/// directly.** Under `fail_fast=true` the FIRST-returned violation may
/// be either kind depending on the iteration order of the per-tenant
/// pre-pass vs. the cross-tenant pass — both signals are mechanically
/// valid (an A→B contamination simultaneously presents as: (a) a
/// `CrossTenantContamination` from the cross-tenant pass, AND (b) an
/// `UnknownKey` from B's per-tenant pre-pass because B's recovered
/// state has a key absent from B's `any_history`). The unit test
/// `cross_tenant_fail_fast_returns_first_violation` documents this
/// contract weakening explicitly.
///
/// **K-1b cross-tenant fault isolation tests use `fail_fast=false`**
/// for two reasons:
/// 1. Collect every violation across the run for a non-vacuous proof
///    that the harness fired faults at all (`floor`-style sanity
///    checks need the full violation list).
/// 2. The contamination signal is meant to be cross-tenant-specific;
///    the K-1b oracle shape preserves that signal cleanly only on
///    the `fail_fast=false` path.
///
/// **K-1+ smoke tests use `fail_fast=true`** when they only need to
/// know whether ANY violation fired — typical of the 30s/5min smokes
/// where a fast fail beats a deep diagnosis.
///
/// **K-2 callers checking for contamination specifically** MUST set
/// `fail_fast=false` and walk `cross_tenant_violations` for
/// `CrossTenantContamination` variants. Anything else risks a green
/// `UnknownKey` swallow that hides the contamination axis.
///
/// ## Determinism
///
/// The check is order-independent at the contract level. The
/// `tenants_checked` field surfaces the iteration order so
/// reproducible-violation reports can re-run the same campaign and
/// see the same fault trail; the iteration order itself is the
/// `BTreeMap`-equivalent insertion-sorted order produced by sorting
/// the tenant set ascendingly by raw u64.
pub fn verify_cross_tenant_invariants(
    input: &CrossTenantOracleInput,
    config: &OracleConfig,
) -> Result<CrossTenantOracleReport, OracleViolation> {
    // Collect every tenant referenced (union of pre-crash + recovered
    // keys) sorted ascendingly for stable iteration.
    let mut tenants_set: HashSet<TenantId> = HashSet::new();
    for t in input.pre_crash_per_tenant.keys() {
        tenants_set.insert(*t);
    }
    for t in input.recovered_per_tenant.keys() {
        tenants_set.insert(*t);
    }
    let mut tenants: Vec<TenantId> = tenants_set.into_iter().collect();
    tenants.sort_by_key(|t| t.raw());

    let mut per_tenant: HashMap<TenantId, OracleReport> = HashMap::with_capacity(tenants.len());
    let mut cross_tenant_violations: Vec<OracleViolation> = Vec::new();

    macro_rules! record_cross {
        ($v:expr) => {{
            let v = $v;
            if config.fail_fast {
                return Err(v);
            }
            cross_tenant_violations.push(v);
        }};
    }

    // Step 1: per-tenant single-tenant oracle. Defensive
    // `LedgerNotFoundForTenant` for tenants with a recovered entry
    // but NO pre-crash entry (the test forgot to seed the ledger
    // map) — surfaces silent harness misconfiguration.
    for tenant in &tenants {
        let pre = match input.pre_crash_per_tenant.get(tenant) {
            Some(p) => p,
            None => {
                record_cross!(OracleViolation::LedgerNotFoundForTenant {
                    tenant_raw: tenant.raw(),
                });
                continue;
            }
        };
        let default_recovered = RecoveredState::default();
        let rec = input
            .recovered_per_tenant
            .get(tenant)
            .unwrap_or(&default_recovered);
        match verify_post_recovery_invariants(pre, rec, config) {
            Ok(report) => {
                per_tenant.insert(*tenant, report);
            }
            Err(v) => {
                if config.fail_fast {
                    return Err(v);
                }
                // With fail_fast=false the per-tenant oracle has
                // already accumulated its own violations into the
                // OracleReport; if it returned Err it means the
                // per-tenant path short-circuited (today the
                // per-tenant path still short-circuits even with
                // fail_fast=false — see the codex L-2 note in
                // `verify_post_recovery_invariants` doc). Bubble it
                // up to the cross-tenant violations list so the
                // caller sees every divergence.
                cross_tenant_violations.push(v);
            }
        }
    }

    // Step 2: cross-tenant contamination check.
    // For each pair (source, target) with source != target: for each
    // (source_tenant, node_id) in source.any_history, examine target's
    // recovered state at NodeId node_id (under (target, node_id)).
    // If target's recovered bytes match SOMEthing in source's history
    // at that node_id BUT target's own any_history at the same key
    // does NOT contain those bytes, fire CrossTenantContamination.
    let mut cross_tenant_checks: u64 = 0;
    for (i, source) in tenants.iter().enumerate() {
        let source_pre = match input.pre_crash_per_tenant.get(source) {
            Some(p) => p,
            None => continue,
        };
        for (j, target) in tenants.iter().enumerate() {
            if i == j {
                continue;
            }
            let target_pre = input.pre_crash_per_tenant.get(target);
            let target_rec = match input.recovered_per_tenant.get(target) {
                Some(r) => r,
                None => continue,
            };
            for ((src_tenant, node_id), src_history) in &source_pre.any_history {
                debug_assert_eq!(
                    src_tenant, source,
                    "pre_crash_per_tenant[T] must only contain rows tagged T"
                );
                let target_key = (*target, *node_id);
                let Some(rec_bytes) = target_rec.bytes_by_key.get(&target_key).copied() else {
                    continue;
                };
                cross_tenant_checks += 1;

                let target_owns_bytes = target_pre
                    .and_then(|tp| tp.any_history.get(&target_key))
                    .map(|set| set.contains(&rec_bytes))
                    .unwrap_or(false);
                if target_owns_bytes {
                    continue;
                }
                if src_history.contains(&rec_bytes) {
                    record_cross!(OracleViolation::CrossTenantContamination {
                        source_raw: source.raw(),
                        target_raw: target.raw(),
                        key_raw: node_id.raw(),
                        // K-1b ledger format does not record per-row
                        // LSN; reserved for K-2 binary format.
                        lsn_raw: Lsn::ZERO.raw(),
                    });
                }
            }
        }
    }

    Ok(CrossTenantOracleReport {
        per_tenant,
        cross_tenant_violations,
        tenants_checked: tenants,
        cross_tenant_checks,
    })
}

/// Build a [`CommittedStatsRebuild`] from a list of pre-crash
/// commits. Mirrors the increment-hook logic in `crud::commit` so
/// the oracle can compare apples-to-apples.
///
/// `commits` is the per-tenant slice of `LedgerRecord` rows from the
/// pre-crash ledger. Each row counts as a node `Create` (K-1's
/// workload only exercises the node-Create path; K-2 / K-3 will
/// extend with `Delete` / `Update` / rel paths).
///
/// `commits_observed` mirrors the per-tenant commit counter that
/// `crud::commit` increments via `observe_commit()` once per
/// touched tenant per transaction. The K-1 workload commits one
/// record per transaction so `commits_observed == commits.len()`.
/// K-2 / K-3 multi-record-per-tx workloads will pass an explicit
/// per-tenant transaction count (not row count).
pub fn rebuild_stats_from_node_creates(commits: &[(LabelId, ())]) -> CommittedStatsRebuild {
    let mut s = CommittedStatsRebuild::default();
    for (label, _) in commits {
        *s.label_counts.entry(*label).or_insert(0) += 1;
        s.total_nodes += 1;
    }
    s.commits_observed = commits.len() as u64;
    s
}

/// Snapshot a [`CatalogStats`] into the oracle's portable form.
///
/// `labels_to_check` lists every label the workload exercised
/// (so we can read each label's count out of the DashMap-backed
/// `CatalogStats` accessor without a full iteration). `rel_types_to_check`
/// is the rel-type analog. Empty list ⇒ no per-key stats are read.
pub fn snapshot_catalog_stats(
    stats: &CatalogStats,
    labels_to_check: &[LabelId],
    rel_types_to_check: &[TypeId],
) -> CommittedStatsRebuild {
    let mut out = CommittedStatsRebuild::default();
    for l in labels_to_check {
        if let Some(c) = stats.label_cardinality(*l) {
            out.label_counts.insert(*l, c);
        }
    }
    for t in rel_types_to_check {
        if let Some(c) = stats.rel_type_cardinality(*t) {
            out.rel_type_counts.insert(*t, c);
        }
    }
    out.total_nodes = stats.total_node_count().unwrap_or(0);
    out.total_rels = stats.total_rel_count().unwrap_or(0);
    // Per-tenant commit counter — read directly from
    // `CatalogStats::commits_observed_count()`. The rebuilt-from-
    // ledger side (`rebuild_stats_from_node_creates`) emits
    // `commits.len()` (K-1 workload commits one record per tx, so
    // ledger row count == observe_commit() invocation count). Both
    // sides therefore unify on counter semantics; mismatch surfaces
    // M4-41's stats-persistence gap honestly (no booleanized
    // 1-or-0 hack to mask >1-commit divergence).
    out.commits_observed = stats.commits_observed_count();
    out
}

/// Determine the durability tier integer encoding used by the
/// pre-crash ledger. K-1 uses `1` for `Strict` and `3` for `Periodic`;
/// extend with `5` for `Async` if/when a v1.1 tier ships.
pub fn tier_to_u8(tier: DurabilityTier) -> u8 {
    match tier {
        DurabilityTier::Strict => 1,
        DurabilityTier::Periodic { .. } => 3,
    }
}

/// Convert a tier byte back to a [`DurabilityTier`]. The Periodic
/// `rpo_ms` parameter is read from `t3_rpo_ms` (default 1000 ms per
/// codex L-1 fix). Pre-fix this hardcoded `rpo_ms = 1_000` regardless
/// of the campaign's actual T3 RPO budget — campaigns testing tighter
/// budgets had no path to override.
pub fn u8_to_tier(t: u8, t3_rpo_ms: u64) -> Option<DurabilityTier> {
    match t {
        1 => Some(DurabilityTier::Strict),
        3 => Some(DurabilityTier::Periodic { rpo_ms: t3_rpo_ms }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(t: u64, n: u64) -> (TenantId, NodeId) {
        (TenantId::new(t), NodeId::new(n))
    }

    fn pre_crash_with_one_t1() -> CommittedState {
        let mut s = CommittedState::default();
        let k = key(0, 1);
        s.latest_t1.insert(k, (10, 100, 200));
        let mut h = HashSet::new();
        h.insert((10, 100, 200));
        s.any_history.insert(k, h);
        s.total_commits = 1;
        s
    }

    #[test]
    fn happy_path_t1_recovered() {
        let pre = pre_crash_with_one_t1();
        let mut rec = RecoveredState::default();
        rec.bytes_by_key.insert(key(0, 1), (10, 100, 200));
        let r = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap();
        assert_eq!(r.unique_keys, 1);
        assert_eq!(r.t1_keys, 1);
        assert_eq!(r.t1_satisfied, 1);
        assert_eq!(r.historical_match, 1);
    }

    #[test]
    fn t1_strict_drift_detected() {
        let pre = pre_crash_with_one_t1();
        let mut rec = RecoveredState::default();
        // Wrong bytes — not in any_history; the oracle reports
        // GhostBytes (which is a strict superset of T1StrictDrift —
        // the bytes don't even match a historical commit).
        rec.bytes_by_key.insert(key(0, 1), (99, 99, 99));
        let err =
            verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap_err();
        assert!(matches!(err, OracleViolation::GhostBytes { .. }));
    }

    #[test]
    fn t1_strict_drift_within_history_distinct_from_ghost() {
        // Build a pre-crash state where the key has TWO historical
        // commits — one T3 (older) and one T1 (latest). The
        // recovered store reports the OLDER T3 bytes; this is NOT
        // a ghost (bytes are in history), but IS a T1 strict drift.
        let mut s = CommittedState::default();
        let k = key(0, 1);
        s.latest_t1.insert(k, (10, 100, 200));
        let mut h = HashSet::new();
        h.insert((10, 100, 200));
        h.insert((99, 99, 99));
        s.any_history.insert(k, h);
        s.total_commits = 2;
        let mut rec = RecoveredState::default();
        rec.bytes_by_key.insert(key(0, 1), (99, 99, 99));
        let err = verify_post_recovery_invariants(&s, &rec, &OracleConfig::default()).unwrap_err();
        assert!(matches!(err, OracleViolation::T1StrictDrift { .. }));
    }

    #[test]
    fn t1_missing_detected() {
        let pre = pre_crash_with_one_t1();
        let rec = RecoveredState::default();
        let err =
            verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap_err();
        assert!(matches!(err, OracleViolation::T1Missing { .. }));
    }

    #[test]
    fn t1_strict_ghost_key_detected() {
        // Codex H-2 pin (mirror of t1_strict_drift_detected): the
        // 1:1 unique:total CRUD invariant has two failure modes by
        // symmetry. Pre-fix only the drift direction (key in ledger
        // missing from recovered) was enforced; the ghost direction
        // (key in recovered with NO row in pre-crash ledger) was
        // defensively defined as `OracleViolation::UnknownKey` but
        // NEVER constructed. Phase 5.5 / PR #130 catches both;
        // K-1 oracle pre-fix was a regression on this contract.
        //
        // Setup: pre-crash ledger has key (0, 1) only. Recovered
        // store has key (0, 1) AND a ghost key (0, 99). The ghost
        // key MUST surface as UnknownKey.
        let pre = pre_crash_with_one_t1();
        let mut rec = RecoveredState::default();
        // Legitimate recovered key matches pre-crash bytes.
        rec.bytes_by_key.insert(key(0, 1), (10, 100, 200));
        // Ghost key — never in pre_crash.any_history.
        rec.bytes_by_key.insert(key(0, 99), (7, 7, 7));

        let err =
            verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap_err();
        match err {
            OracleViolation::UnknownKey {
                tenant_raw,
                node_id_raw,
                observed,
            } => {
                assert_eq!(tenant_raw, 0);
                assert_eq!(node_id_raw, 99);
                assert_eq!(observed, (7, 7, 7));
            }
            other => {
                panic!("expected OracleViolation::UnknownKey for ghost key (0, 99); got {other:?}")
            }
        }
    }

    #[test]
    fn t3_rpo_floor_enforced() {
        // 5 T3-only keys; 1 recovered. 1/5 = 0.20 < 0.80 floor.
        let mut s = CommittedState::default();
        for n in 0..5 {
            let k = key(0, n);
            let mut h = HashSet::new();
            h.insert((10, 100, n as u32));
            s.any_history.insert(k, h);
        }
        s.total_commits = 5;
        let mut rec = RecoveredState::default();
        rec.bytes_by_key.insert(key(0, 0), (10, 100, 0));
        let err = verify_post_recovery_invariants(&s, &rec, &OracleConfig::default()).unwrap_err();
        assert!(matches!(err, OracleViolation::T3RpoLossExceeded { .. }));
    }

    #[test]
    fn t3_rpo_floor_passes_at_80pct() {
        // 5 T3 keys; 4 recovered. 4/5 = 0.80 — at the floor.
        let mut s = CommittedState::default();
        for n in 0..5 {
            let k = key(0, n);
            let mut h = HashSet::new();
            h.insert((10, 100, n as u32));
            s.any_history.insert(k, h);
        }
        s.total_commits = 5;
        let mut rec = RecoveredState::default();
        for n in 0..4 {
            rec.bytes_by_key.insert(key(0, n), (10, 100, n as u32));
        }
        let r = verify_post_recovery_invariants(&s, &rec, &OracleConfig::default()).unwrap();
        assert_eq!(r.historical_match, 4);
        assert_eq!(r.t3_rpo_lost, 1);
    }

    #[test]
    fn stats_consistency_check() {
        let mut label_counts = HashMap::new();
        label_counts.insert(LabelId::new(7), 3);
        let rebuilt = CommittedStatsRebuild {
            label_counts,
            total_nodes: 3,
            commits_observed: 3,
            ..Default::default()
        };
        let mut pre = CommittedState::default();
        pre.stats_by_tenant
            .insert(TenantId::new(0), rebuilt.clone());
        let mut rec = RecoveredState::default();
        rec.stats_by_tenant.insert(TenantId::new(0), rebuilt);
        let r = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap();
        assert_eq!(
            r.stats_consistent_by_tenant.get(&TenantId::new(0)),
            Some(&true)
        );
    }

    #[test]
    fn stats_inconsistency_fatal_by_default() {
        let mut pre = CommittedState::default();
        let rebuilt = CommittedStatsRebuild {
            total_nodes: 5,
            commits_observed: 5,
            ..Default::default()
        };
        pre.stats_by_tenant.insert(TenantId::new(0), rebuilt);
        let mut rec = RecoveredState::default();
        let wrong = CommittedStatsRebuild {
            total_nodes: 3, // recovered count is wrong
            commits_observed: 5,
            ..Default::default()
        };
        rec.stats_by_tenant.insert(TenantId::new(0), wrong);
        let err =
            verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap_err();
        assert!(matches!(err, OracleViolation::StatsInconsistent { .. }));
    }

    #[test]
    fn stats_inconsistency_non_fatal_when_disabled() {
        // The `stats_inconsistency_fatal=false` knob exists for K-2 /
        // K-3 callers that want to collect every divergence in a
        // long-running campaign rather than fail-fast on the first.
        // M4-41 implementation slice (this PR) flipped the K-1a +
        // K-1b smokes to `true` per ADR-038 amendment-06 §3 R1
        // (cold-start rebuild populates per-tenant CatalogStats
        // from MVCC at recovered LSN); the knob's `false` arm is
        // preserved for K-2 / K-3 collect-all campaign callers and
        // the unit test below pins its semantics.
        let mut pre = CommittedState::default();
        let rebuilt = CommittedStatsRebuild {
            total_nodes: 5,
            commits_observed: 5,
            ..Default::default()
        };
        pre.stats_by_tenant.insert(TenantId::new(0), rebuilt);
        let mut rec = RecoveredState::default();
        let wrong = CommittedStatsRebuild {
            total_nodes: 3,
            commits_observed: 5,
            ..Default::default()
        };
        rec.stats_by_tenant.insert(TenantId::new(0), wrong);
        let cfg = OracleConfig {
            stats_inconsistency_fatal: false,
            ..OracleConfig::default()
        };
        let r = verify_post_recovery_invariants(&pre, &rec, &cfg).unwrap();
        assert_eq!(
            r.stats_consistent_by_tenant.get(&TenantId::new(0)),
            Some(&false)
        );
    }

    #[test]
    fn stats_consistency_check_with_multi_commit_tenant() {
        // Codex BLOCKER-1 verification pin: pre-fix, the comparison
        // would fail by construction for any tenant with commits.len()
        // > 1, because `snapshot_catalog_stats` returned `1` (or `0`)
        // while `rebuild_stats_from_node_creates` returned
        // `commits.len()`. Post-fix both sides unify on the counter
        // semantics — this test exercises the snapshot side via a
        // real `CatalogStats` and verifies a 5-commit tenant rebuilds
        // cleanly with `stats_inconsistency_fatal=true` (the K-1
        // smoke's load-bearing knob position).
        let stats = CatalogStats::new();
        let label = LabelId::new(7);
        for _ in 0..5 {
            stats.increment_label(label);
            stats.increment_total_nodes();
            stats.observe_commit();
        }
        let snapshot = snapshot_catalog_stats(&stats, &[label], &[]);
        // Counter-side reads `commits_observed_count() = 5`
        // (post-fix), not `1` (pre-fix booleanized hack).
        assert_eq!(
            snapshot.commits_observed, 5,
            "snapshot must report exact count"
        );
        assert_eq!(snapshot.total_nodes, 5);
        assert_eq!(snapshot.label_counts.get(&label).copied(), Some(5));

        let commits: Vec<(LabelId, ())> = (0..5).map(|_| (label, ())).collect();
        let rebuilt = rebuild_stats_from_node_creates(&commits);
        assert_eq!(rebuilt, snapshot, "both sides unify on counter semantics");

        // Drive the full oracle path with `stats_inconsistency_fatal=true`
        // (default) — proves the K-1 smoke knob position holds for a
        // multi-commit tenant.
        let mut pre = CommittedState::default();
        pre.stats_by_tenant.insert(TenantId::new(0), rebuilt);
        let mut rec = RecoveredState::default();
        rec.stats_by_tenant.insert(TenantId::new(0), snapshot);
        let r = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).unwrap();
        assert_eq!(
            r.stats_consistent_by_tenant.get(&TenantId::new(0)),
            Some(&true),
            "5-commit tenant must compare consistent under unified counter semantics"
        );
    }

    #[test]
    fn tier_round_trip() {
        assert_eq!(tier_to_u8(DurabilityTier::Strict), 1);
        assert_eq!(tier_to_u8(DurabilityTier::Periodic { rpo_ms: 500 }), 3);
        assert!(matches!(u8_to_tier(1, 1_000), Some(DurabilityTier::Strict)));
        match u8_to_tier(3, 250) {
            Some(DurabilityTier::Periodic { rpo_ms }) => assert_eq!(rpo_ms, 250),
            other => panic!("expected Periodic with rpo_ms=250 (codex L-1); got {other:?}"),
        }
        // Codex L-1: rpo_ms is now parameterized via OracleConfig.
        // Default 1_000 round-trips cleanly.
        match u8_to_tier(3, 1_000) {
            Some(DurabilityTier::Periodic { rpo_ms }) => assert_eq!(rpo_ms, 1_000),
            other => panic!("expected Periodic with rpo_ms=1000; got {other:?}"),
        }
        assert_eq!(u8_to_tier(99, 1_000), None);
    }

    #[test]
    fn oracle_fail_fast_short_circuits_on_first_violation() {
        // Codex L-2 pin: with fail_fast=true (default), the oracle
        // returns Err on the FIRST violation it encounters. Running
        // a multi-violation pre/rec setup must Err — must not surface
        // a second violation in the report (the report path is never
        // reached).
        let mut pre = CommittedState::default();
        // Two T1 keys; both will be missing post-recovery.
        for n in 0..2 {
            let k = key(0, n);
            pre.latest_t1.insert(k, (10, 100, n as u32));
            let mut h = HashSet::new();
            h.insert((10, 100, n as u32));
            pre.any_history.insert(k, h);
        }
        pre.total_commits = 2;
        let rec = RecoveredState::default();

        let cfg = OracleConfig::default();
        assert!(cfg.fail_fast, "default OracleConfig must be fail_fast=true");
        let err = verify_post_recovery_invariants(&pre, &rec, &cfg).unwrap_err();
        assert!(
            matches!(err, OracleViolation::T1Missing { .. }),
            "expected T1Missing on first violation; got {err:?}"
        );
    }

    #[test]
    fn oracle_collects_all_violations_when_fail_fast_disabled() {
        // Codex L-2 pin: with fail_fast=false, the oracle collects
        // every violation into report.violations. K-2 collect-all
        // callers depend on this surface.
        //
        // Setup: 3 T1 keys, 0 recovered. Pre-fix this returned Err on
        // the FIRST T1Missing. Post-fix collects 3 T1Missing + 1
        // T3RpoLossExceeded (because historical_match=0/3 = 0.00 <
        // 0.80 floor — confirming the RPO floor check ALSO runs
        // past the iteration loop violations under fail_fast=false).
        let mut pre = CommittedState::default();
        for n in 0..3 {
            let k = key(0, n);
            pre.latest_t1.insert(k, (10, 100, n as u32));
            let mut h = HashSet::new();
            h.insert((10, 100, n as u32));
            pre.any_history.insert(k, h);
        }
        pre.total_commits = 3;
        let rec = RecoveredState::default();

        let cfg = OracleConfig {
            fail_fast: false,
            ..OracleConfig::default()
        };
        let report = verify_post_recovery_invariants(&pre, &rec, &cfg).unwrap();
        let t1_missing_count = report
            .violations
            .iter()
            .filter(|v| matches!(v, OracleViolation::T1Missing { .. }))
            .count();
        let rpo_count = report
            .violations
            .iter()
            .filter(|v| matches!(v, OracleViolation::T3RpoLossExceeded { .. }))
            .count();
        assert_eq!(
            t1_missing_count, 3,
            "fail_fast=false must collect ALL 3 T1Missing per-key violations; got {t1_missing_count} \
             (full: {:?})",
            report.violations
        );
        assert_eq!(
            rpo_count, 1,
            "fail_fast=false must also collect the T3RpoLossExceeded post-loop check; \
             got {rpo_count} (full: {:?})",
            report.violations
        );
        assert_eq!(
            report.violations.len(),
            4,
            "expected 3 T1Missing + 1 T3RpoLossExceeded = 4 total under fail_fast=false; \
             got {} (full: {:?})",
            report.violations.len(),
            report.violations
        );
    }

    // ── K-1b cross-tenant oracle (issue #214) ────────────────────────

    fn pre_with(tenant_raw: u64, rows: &[(u64, CommittedBytes)]) -> CommittedState {
        // Helper: build a single-tenant CommittedState from
        // (node_id_raw, bytes) pairs, treating every row as a T1
        // strict commit (the K-1b workload defaults to T1).
        let mut s = CommittedState::default();
        let tenant = TenantId::new(tenant_raw);
        for (n, bytes) in rows {
            let k = (tenant, NodeId::new(*n));
            s.latest_t1.insert(k, *bytes);
            s.any_history
                .entry(k)
                .or_insert_with(HashSet::new)
                .insert(*bytes);
        }
        s.total_commits = rows.len() as u64;
        s
    }

    fn rec_with(tenant_raw: u64, rows: &[(u64, CommittedBytes)]) -> RecoveredState {
        let mut r = RecoveredState::default();
        let tenant = TenantId::new(tenant_raw);
        for (n, bytes) in rows {
            r.bytes_by_key.insert((tenant, NodeId::new(*n)), *bytes);
        }
        r
    }

    #[test]
    fn cross_tenant_oracle_clean_three_tenants_returns_ok() {
        // Happy path: three tenants commit disjoint label spaces +
        // recover cleanly; cross_tenant_violations is empty.
        let mut input = CrossTenantOracleInput::default();
        let t_a = TenantId::new(10);
        let t_b = TenantId::new(20);
        let t_c = TenantId::new(30);
        let pre_a = pre_with(10, &[(1, (100_001, 11, 22)), (2, (100_002, 11, 22))]);
        let pre_b = pre_with(20, &[(1, (200_001, 33, 44)), (2, (200_002, 33, 44))]);
        let pre_c = pre_with(30, &[(1, (300_001, 55, 66))]);
        let rec_a = rec_with(10, &[(1, (100_001, 11, 22)), (2, (100_002, 11, 22))]);
        let rec_b = rec_with(20, &[(1, (200_001, 33, 44)), (2, (200_002, 33, 44))]);
        let rec_c = rec_with(30, &[(1, (300_001, 55, 66))]);
        input.pre_crash_per_tenant.insert(t_a, pre_a);
        input.pre_crash_per_tenant.insert(t_b, pre_b);
        input.pre_crash_per_tenant.insert(t_c, pre_c);
        input.recovered_per_tenant.insert(t_a, rec_a);
        input.recovered_per_tenant.insert(t_b, rec_b);
        input.recovered_per_tenant.insert(t_c, rec_c);

        let report = verify_cross_tenant_invariants(&input, &OracleConfig::default()).unwrap();
        assert!(
            report.cross_tenant_violations.is_empty(),
            "happy path must report 0 cross-tenant violations; got {:?}",
            report.cross_tenant_violations
        );
        assert_eq!(report.tenants_checked, vec![t_a, t_b, t_c]);
        assert_eq!(report.per_tenant.len(), 3);
    }

    #[test]
    fn cross_tenant_contamination_detected_when_bytes_bleed_across_tenants() {
        // Inject a contamination: tenant B's recovered state at
        // NodeId=1 has bytes that came from tenant A's pre-crash
        // ledger, and tenant B never wrote those bytes itself.
        // CrossTenantContamination must fire.
        let mut input = CrossTenantOracleInput::default();
        let t_a = TenantId::new(10);
        let t_b = TenantId::new(20);
        // A's bytes at NodeId=1 are (100_001, 11, 22). B's pre-crash
        // ledger has NO entry at NodeId=1 (B's earliest NodeId is 2).
        let pre_a = pre_with(10, &[(1, (100_001, 11, 22))]);
        let pre_b = pre_with(20, &[(2, (200_002, 33, 44))]);
        let rec_a = rec_with(10, &[(1, (100_001, 11, 22))]);
        // B's recovered state has the contamination: bytes from A
        // surfacing under tenant B's tag at NodeId=1.
        let rec_b = rec_with(20, &[(2, (200_002, 33, 44)), (1, (100_001, 11, 22))]);
        input.pre_crash_per_tenant.insert(t_a, pre_a);
        input.pre_crash_per_tenant.insert(t_b, pre_b);
        input.recovered_per_tenant.insert(t_a, rec_a);
        input.recovered_per_tenant.insert(t_b, rec_b);

        // With fail_fast=true (default), the per-tenant oracle for
        // tenant B will fire UnknownKey FIRST (because B's recovered
        // state has a key that isn't in B's any_history). Run with
        // fail_fast=false so the cross-tenant check also runs and
        // we can verify the contamination signal explicitly.
        let cfg = OracleConfig {
            fail_fast: false,
            ..OracleConfig::default()
        };
        let report = verify_cross_tenant_invariants(&input, &cfg).unwrap();

        let saw_contamination = report
            .cross_tenant_violations
            .iter()
            .any(|v| matches!(v, OracleViolation::CrossTenantContamination { source_raw, target_raw, key_raw, .. } if *source_raw == 10 && *target_raw == 20 && *key_raw == 1));
        assert!(
            saw_contamination,
            "CrossTenantContamination must fire for A→B at NodeId=1; \
             got {:?}",
            report.cross_tenant_violations
        );
    }

    #[test]
    fn cross_tenant_no_false_positive_on_coincidental_node_id_aliasing() {
        // Critical false-positive guard: tenant A and tenant B both
        // committed at NodeId=1 but with DISTINCT bytes; B's
        // recovered state at NodeId=1 has B's OWN bytes; this is NOT
        // contamination. The oracle MUST NOT fire.
        let mut input = CrossTenantOracleInput::default();
        let t_a = TenantId::new(10);
        let t_b = TenantId::new(20);
        let pre_a = pre_with(10, &[(1, (100_001, 11, 22))]);
        let pre_b = pre_with(20, &[(1, (200_001, 33, 44))]);
        let rec_a = rec_with(10, &[(1, (100_001, 11, 22))]);
        let rec_b = rec_with(20, &[(1, (200_001, 33, 44))]);
        input.pre_crash_per_tenant.insert(t_a, pre_a);
        input.pre_crash_per_tenant.insert(t_b, pre_b);
        input.recovered_per_tenant.insert(t_a, rec_a);
        input.recovered_per_tenant.insert(t_b, rec_b);

        let report = verify_cross_tenant_invariants(&input, &OracleConfig::default()).unwrap();
        assert!(
            report.cross_tenant_violations.is_empty(),
            "coincidental NodeId aliasing with disjoint bytes is NOT contamination; \
             got: {:?}",
            report.cross_tenant_violations
        );
        // The check still ran — we want to assert the oracle visited
        // the (A→B, NodeId=1) and (B→A, NodeId=1) pairs.
        assert!(
            report.cross_tenant_checks >= 2,
            "expected ≥ 2 cross-tenant checks for the (A→B, B→A) NodeId=1 pair; \
             got {}",
            report.cross_tenant_checks
        );
    }

    #[test]
    fn cross_tenant_fail_fast_returns_first_violation() {
        // With fail_fast=true (default), an A→B contamination triggers
        // an Err(CrossTenantContamination) on the first cross-tenant
        // check that finds it. (The per-tenant oracle's UnknownKey
        // fires FIRST — both check directions are valid violations;
        // the oracle's order-of-evaluation pin is "per-tenant before
        // cross-tenant".)
        let mut input = CrossTenantOracleInput::default();
        let t_a = TenantId::new(10);
        let t_b = TenantId::new(20);
        let pre_a = pre_with(10, &[(1, (100_001, 11, 22))]);
        let pre_b = pre_with(20, &[(2, (200_002, 33, 44))]);
        let rec_a = rec_with(10, &[(1, (100_001, 11, 22))]);
        // Contaminated rec_b: A's bytes appear at NodeId=1 under
        // tenant B's tag.
        let rec_b = rec_with(20, &[(1, (100_001, 11, 22)), (2, (200_002, 33, 44))]);
        input.pre_crash_per_tenant.insert(t_a, pre_a);
        input.pre_crash_per_tenant.insert(t_b, pre_b);
        input.recovered_per_tenant.insert(t_a, rec_a);
        input.recovered_per_tenant.insert(t_b, rec_b);

        let err = verify_cross_tenant_invariants(&input, &OracleConfig::default()).unwrap_err();
        // The first-violation arm. The exact variant fired depends on
        // iteration order: with fail_fast=true the per-tenant oracle
        // fires UnknownKey for tenant B (because rec_b has a key NOT
        // in pre_b's any_history). EITHER UnknownKey OR
        // CrossTenantContamination is acceptable as the first
        // observed violation; the contract is that fail_fast=true
        // returns ONE violation early.
        assert!(
            matches!(
                err,
                OracleViolation::UnknownKey { .. }
                    | OracleViolation::CrossTenantContamination { .. }
            ),
            "fail_fast=true must return either UnknownKey or \
             CrossTenantContamination; got {err:?}"
        );
    }

    #[test]
    fn cross_tenant_ledger_not_found_for_tenant_fires_when_pre_missing() {
        // Defensive variant: if a tenant has a recovered state but
        // NO pre-crash ledger entry, LedgerNotFoundForTenant fires.
        // Surfaces a test misconfiguration loudly.
        let mut input = CrossTenantOracleInput::default();
        let t_a = TenantId::new(10);
        // No pre-crash for t_a; recovered state is non-empty.
        input
            .recovered_per_tenant
            .insert(t_a, rec_with(10, &[(1, (100_001, 11, 22))]));

        let cfg = OracleConfig {
            fail_fast: false,
            ..OracleConfig::default()
        };
        let report = verify_cross_tenant_invariants(&input, &cfg).unwrap();
        let saw_gap = report.cross_tenant_violations.iter().any(|v| {
            matches!(v, OracleViolation::LedgerNotFoundForTenant { tenant_raw } if *tenant_raw == 10)
        });
        assert!(
            saw_gap,
            "LedgerNotFoundForTenant must fire for tenant 10 with no pre-crash entry; \
             got {:?}",
            report.cross_tenant_violations
        );
    }

    #[test]
    fn cross_tenant_tenants_checked_is_sorted_for_stable_iteration() {
        // Iteration-order pin: tenants_checked is the union of pre/rec
        // keys, sorted by raw u64 ascending. Reproducible failures
        // re-run the same campaign and see the same fault trail.
        let mut input = CrossTenantOracleInput::default();
        // Insert in DESCENDING order to prove sort semantics.
        for t_raw in [30, 10, 20] {
            input
                .pre_crash_per_tenant
                .insert(TenantId::new(t_raw), pre_with(t_raw, &[]));
        }
        let report = verify_cross_tenant_invariants(&input, &OracleConfig::default()).unwrap();
        let raws: Vec<u64> = report.tenants_checked.iter().map(|t| t.raw()).collect();
        assert_eq!(raws, vec![10, 20, 30]);
    }
}
