//! K-1c+d — Encoding-mismatch I-V coverage primitives (issue #215).
//!
//! Per ADR-038 amendment-03 §"Slice K" K-1c row + K-1d row (per
//! `docs/roadmap.md` M4.i): "encoding-mismatch I-V replay paths
//! exercised end-to-end against the K-1 harness; per-class violation
//! reporting wired through `OracleViolation`."
//!
//! ## Mapping the 5 invariants → `OracleViolation`
//!
//! Each invariant is an *encoding surface* the WAL codec / recovery
//! pipeline could mis-encode. The K-1 oracle already taxonomises the
//! resulting *post-recovery observable states*; this module provides
//! deterministic primitives that produce each observable state from a
//! clean `RecoveredState`, so a strict-mode oracle pass surfaces a
//! SPECIFIC violation per surface.
//!
//! | Invariant | Encoding surface | Inject primitive | Oracle violation |
//! |-----------|------------------|------------------|------------------|
//! | **I-1** | Record format (NodeRecord field bytes; ADR-022 codec) | [`inject_record_format_mismatch`] | [`super::oracle::OracleViolation::GhostBytes`] |
//! | **I-2** | Key format (`(TenantId, NodeId)` keying; primary-index ADR-023) | [`inject_key_format_mismatch`] | [`super::oracle::OracleViolation::UnknownKey`] |
//! | **I-3** | MVCC chain layout (latest-T1 vs. earlier-historical; ADR-018) | [`inject_mvcc_chain_layout_mismatch`] | [`super::oracle::OracleViolation::T1StrictDrift`] |
//! | **I-4** | WAL commit-bundle atomicity (ADR-031 §R5 torn-tail drop) | [`inject_wal_commit_bundle_atomicity_mismatch`] | [`super::oracle::OracleViolation::T1Missing`] |
//! | **I-5** | Cross-tenant serialization (ADR-011 tenancy boundary) | [`inject_cross_tenant_serialization_mismatch`] | [`super::oracle::OracleViolation::CrossTenantContamination`] |
//!
//! Each primitive is deterministic, side-effect-only (mutates the
//! `RecoveredState` in place), and reversible (callers comment out the
//! call to run the Phase 4.3 reverse-test cycle: `inject → PASS;
//! comment out → FAIL; restore → PASS`).
//!
//! ## Why post-recovery state instead of byte-flipping the WAL
//!
//! The WAL's framing (`WalRecord::encode` + length + CRC32C) detects
//! ANY single byte flip via CRC failure — that drops the entire
//! record at `recover_from_wal` time. For K-1c+d we want each of the
//! 5 invariants to surface a DISTINCT [`super::oracle::OracleViolation`]
//! variant. A naive WAL byte-flip collapses all 5 to `T1Missing`
//! (every record dropped → every T1 commit absent), which fails the
//! "per-class violation reporting" exit criterion in `docs/roadmap.md`
//! M4.i K-1d row.
//!
//! Instead, each primitive reproduces the post-recovery observable
//! state that the named encoding mismatch would produce IF a hostile
//! corruption bypassed CRC (say, a same-byte XOR that preserves CRC,
//! or a higher-layer mis-encoding inside an otherwise-valid bundle).
//! The recovery oracle's job is to surface the post-recovery
//! discrepancy regardless of cause; the oracle is what's being tested.
//!
//! ## Hooks-vs-production discipline (per K-1 mod.rs §"Hooks vs production")
//!
//! These primitives operate on the K-1 harness data structures
//! (`CommittedState`, `RecoveredState`) — never on production WAL
//! bytes, never on production page stores. Production source paths
//! are unmodified. This is a TEST-HARNESS-ONLY module per the K-1
//! discipline.
//!
//! ## Reverse-test discipline (Phase 4.3)
//!
//! Each test that uses these primitives MUST follow the reverse-test
//! discipline: with the inject call active, the oracle returns
//! `Err(OracleViolation::*)` and the test passes; with the inject
//! call commented out, the oracle returns `Ok(_)` and the test's
//! `expect_err(...)` panics, FAILING the test. Restoring the inject
//! call restores the test pass. This 3-step cycle proves the test is
//! non-vacuous — i.e., the assertion DEPENDS on the corruption.

use arcgraph_core::{NodeId, TenantId};

use super::oracle::{CommittedBytes, CommittedState, RecoveredState};

// ─────────────────────────────────────────────────────────────────
// I-1: Record format mismatch
// ─────────────────────────────────────────────────────────────────

/// **I-1 — record format mismatch.**
///
/// Simulates the post-recovery state that arises when a record's
/// field bytes are mis-encoded such that the recovery pipeline
/// returns *different* bytes for the SAME `(tenant, node)` key than
/// any historical commit ever wrote. The off-history bytes are
/// constructed by XORing each component with `0xDEAD_BEEF` so they
/// are deterministic AND probabilistically non-equal to any 32-bit
/// value the workload commits (the workload uses an `XorShift32`
/// whose period is 2^32 - 1, but 0xDEAD_BEEF is outside its
/// first-million-element orbit at the seeds the K-1 smokes use; in
/// practice the XOR produces values that the workload demonstrably
/// never wrote).
///
/// In addition to the probabilistic argument above, this primitive
/// **asserts** that the XOR'd triple is absent from
/// `pre_crash.any_history` at `target` — defense in depth, mirroring
/// the I-2 / I-3 / I-5 signature shape so the same load-bearing
/// non-coincidence check protects every encoding-mismatch surface. If
/// the workload ever evolves to commit a triple whose XOR with
/// 0xDEAD_BEEF lies inside its own history, the assertion fires and
/// the test author is forced to pick a fresh corruption pattern (per
/// docstring contract: "GhostBytes requires the recovered triple to
/// be outside `any_history`").
///
/// Asserted oracle violation: [`super::oracle::OracleViolation::GhostBytes`].
///
/// # Panics
/// - Panics if `target` is not present in `rec.bytes_by_key` — i.e.,
///   the caller must pre-condition that the recovered store
///   materialised SOMETHING at the target key (otherwise there's
///   nothing to corrupt).
/// - Panics if the XOR'd-with-`0xDEAD_BEEF` triple coincides with
///   `pre_crash.any_history[target]` — the corruption would surface
///   as `T1StrictDrift` (or pass entirely if it equals latest T1),
///   not `GhostBytes`. See the defense-in-depth rationale above.
pub fn inject_record_format_mismatch(
    rec: &mut RecoveredState,
    pre_crash: &CommittedState,
    target: (TenantId, NodeId),
) {
    let original = rec
        .bytes_by_key
        .get(&target)
        .copied()
        .expect("inject_record_format_mismatch: target key absent from RecoveredState");
    let drifted: CommittedBytes = (
        original.0 ^ 0xDEAD_BEEF,
        original.1 ^ 0xDEAD_BEEF,
        original.2 ^ 0xDEAD_BEEF,
    );
    if let Some(history) = pre_crash.any_history.get(&target) {
        assert!(
            !history.contains(&drifted),
            "inject_record_format_mismatch: corrupted triple {drifted:?} \
             (= original {original:?} XOR 0xDEAD_BEEF componentwise) coincides \
             with pre_crash.any_history at {target:?}; corruption would not \
             surface as GhostBytes (it's a valid alternate / earlier T1). \
             Pick a fresh corruption pattern; see docstring."
        );
    }
    rec.bytes_by_key.insert(target, drifted);
}

// ─────────────────────────────────────────────────────────────────
// I-2: Key format mismatch
// ─────────────────────────────────────────────────────────────────

/// **I-2 — key format mismatch.**
///
/// Simulates the post-recovery state that arises when the keying
/// (`(TenantId, NodeId)`) of a record is mis-encoded such that the
/// recovery pipeline materialises a key the pre-crash ledger never
/// wrote. This is the GHOST-DIRECTION of the 1:1 unique:total
/// invariant (codex H-2 fix; per [`super::oracle::OracleViolation::UnknownKey`]
/// rustdoc).
///
/// The phantom key + bytes are deterministic via the caller-supplied
/// values; tests pick `(TenantId::SYSTEM, NodeId::new(u64::MAX))` as
/// a "no-workload-ever-touches-this" sentinel.
///
/// Asserted oracle violation: [`super::oracle::OracleViolation::UnknownKey`].
///
/// # Panics
/// Panics if `(phantom_tenant, phantom_node)` is already present in
/// `pre_crash.any_history` — if the pre-crash ledger DID write that
/// key, the corruption is not a "phantom" key and the oracle
/// surfaces a different violation.
pub fn inject_key_format_mismatch(
    rec: &mut RecoveredState,
    pre_crash: &CommittedState,
    phantom_tenant: TenantId,
    phantom_node: NodeId,
    phantom_bytes: CommittedBytes,
) {
    let key = (phantom_tenant, phantom_node);
    assert!(
        !pre_crash.any_history.contains_key(&key),
        "inject_key_format_mismatch: phantom key {key:?} unexpectedly present in pre-crash ledger; \
         test setup is wrong (the corruption is supposed to materialise a key the ledger NEVER wrote)"
    );
    rec.bytes_by_key.insert(key, phantom_bytes);
}

// ─────────────────────────────────────────────────────────────────
// I-3: MVCC chain layout mismatch
// ─────────────────────────────────────────────────────────────────

/// **I-3 — MVCC chain layout mismatch.**
///
/// Simulates the post-recovery state that arises when the MVCC
/// version-chain layout is mis-encoded such that recovery surfaces
/// an EARLIER historical T1 commit's bytes at a key whose latest T1
/// is different. Specifically, the recovered store returns bytes
/// that ARE in `any_history` (so this is NOT a ghost) but are NOT
/// the latest T1's bytes (so T1 strict drift fires).
///
/// Pre-condition: the target key has at least 2 distinct historical
/// T1 commits — one earlier, one latest. The earlier T1's bytes are
/// what we install at the recovered position; the latest T1's bytes
/// are what the oracle EXPECTS.
///
/// Asserted oracle violation: [`super::oracle::OracleViolation::T1StrictDrift`].
///
/// # Panics
/// Panics if `older_t1_bytes` is identical to the latest T1 at
/// `target` (no drift would be observed) OR if `older_t1_bytes` is
/// not in `target`'s `any_history` (that would be a ghost, not a
/// chain-layout drift).
pub fn inject_mvcc_chain_layout_mismatch(
    rec: &mut RecoveredState,
    pre_crash: &CommittedState,
    target: (TenantId, NodeId),
    older_t1_bytes: CommittedBytes,
) {
    let latest_t1 = pre_crash
        .latest_t1
        .get(&target)
        .copied()
        .expect("inject_mvcc_chain_layout_mismatch: target has no latest T1 in pre-crash");
    assert_ne!(
        latest_t1, older_t1_bytes,
        "inject_mvcc_chain_layout_mismatch: older_t1_bytes equals latest T1 — no drift to inject"
    );
    let history = pre_crash
        .any_history
        .get(&target)
        .expect("inject_mvcc_chain_layout_mismatch: target has no any_history");
    assert!(
        history.contains(&older_t1_bytes),
        "inject_mvcc_chain_layout_mismatch: older_t1_bytes {older_t1_bytes:?} not in any_history for {target:?} \
         — would surface as GhostBytes, not T1StrictDrift"
    );
    rec.bytes_by_key.insert(target, older_t1_bytes);
}

// ─────────────────────────────────────────────────────────────────
// I-4: WAL commit-bundle atomicity mismatch
// ─────────────────────────────────────────────────────────────────

/// **I-4 — WAL commit-bundle atomicity mismatch.**
///
/// Simulates the post-recovery state that arises when a WAL
/// commit-bundle's atomicity is violated such that a T1 commit's
/// MVCC writes apply but the bundle is otherwise lost — e.g., a
/// torn-tail bundle the codec wrongly accepted. The post-recovery
/// state is "T1 commit completely absent at this key" even though
/// the pre-crash ledger has a T1 entry there.
///
/// Asserted oracle violation: [`super::oracle::OracleViolation::T1Missing`].
///
/// # Panics
/// Panics if `target` has no T1 entry in `pre_crash.latest_t1` — a
/// non-T1 key dropping post-recovery is not an atomicity violation,
/// it's an RPO loss (T3 tier per ADR-034 D-2).
pub fn inject_wal_commit_bundle_atomicity_mismatch(
    rec: &mut RecoveredState,
    pre_crash: &CommittedState,
    target: (TenantId, NodeId),
) {
    assert!(
        pre_crash.latest_t1.contains_key(&target),
        "inject_wal_commit_bundle_atomicity_mismatch: target {target:?} has no T1 commit in pre-crash; \
         dropping it would surface as T3 RPO loss, not T1Missing"
    );
    rec.bytes_by_key.remove(&target);
}

// ─────────────────────────────────────────────────────────────────
// I-5: Cross-tenant serialization mismatch
// ─────────────────────────────────────────────────────────────────

/// **I-5 — cross-tenant serialization mismatch.**
///
/// Simulates the post-recovery state that arises when a record's
/// `tenant_id` field is mis-encoded such that bytes committed under
/// tenant `source` appear in tenant `destination`'s recovered state
/// at the SAME `NodeId`. Per [`super::oracle::OracleViolation::CrossTenantContamination`]
/// the canonical detection requires the foreign bytes to BE in
/// `source`'s `any_history` AND to NOT be in `destination`'s own
/// `any_history` — false-positive-free as long as workloads commit
/// disjoint byte spaces per tenant.
///
/// This primitive mutates `rec_destination_per_tenant`'s entry for
/// `destination` so that at NodeId `node`, the recovered bytes
/// match `foreign_bytes_from_source`. The caller pre-validates that
/// `foreign_bytes_from_source` is in source-tenant's pre-crash
/// `any_history` and NOT in destination-tenant's.
///
/// Asserted oracle violation: [`super::oracle::OracleViolation::CrossTenantContamination`]
/// (surfaced via [`super::oracle::verify_cross_tenant_invariants`]).
///
/// # Panics
/// Panics if the `rec_destination_per_tenant` map has no entry for
/// `destination_tenant` — the test must seed an empty
/// `RecoveredState` for the destination tenant before injection.
#[allow(clippy::too_many_arguments)]
pub fn inject_cross_tenant_serialization_mismatch(
    rec_per_tenant: &mut std::collections::HashMap<TenantId, RecoveredState>,
    pre_crash_source: &CommittedState,
    pre_crash_destination: &CommittedState,
    source_tenant: TenantId,
    destination_tenant: TenantId,
    target_node: NodeId,
    foreign_bytes_from_source: CommittedBytes,
) {
    let source_history = pre_crash_source
        .any_history
        .get(&(source_tenant, target_node))
        .expect("inject_cross_tenant_serialization_mismatch: source has no any_history at target");
    assert!(
        source_history.contains(&foreign_bytes_from_source),
        "inject_cross_tenant_serialization_mismatch: foreign_bytes {foreign_bytes_from_source:?} \
         not in source tenant {source_tenant:?}'s any_history at NodeId {target_node:?} — \
         would not surface as CrossTenantContamination"
    );
    let dest_key = (destination_tenant, target_node);
    if let Some(dest_history) = pre_crash_destination.any_history.get(&dest_key) {
        assert!(
            !dest_history.contains(&foreign_bytes_from_source),
            "inject_cross_tenant_serialization_mismatch: destination tenant {destination_tenant:?} \
             already has bytes {foreign_bytes_from_source:?} at NodeId {target_node:?} — coincidence; \
             not a contamination signal"
        );
    }
    let dest_rec = rec_per_tenant.get_mut(&destination_tenant).expect(
        "inject_cross_tenant_serialization_mismatch: rec_per_tenant missing destination_tenant",
    );
    dest_rec
        .bytes_by_key
        .insert(dest_key, foreign_bytes_from_source);
}

// ─────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_harness::k1::oracle::{
        CommittedBytes, CommittedState, OracleConfig, OracleViolation, RecoveredState,
        verify_post_recovery_invariants,
    };
    use std::collections::HashSet;

    fn fixture_pre_crash() -> CommittedState {
        let mut s = CommittedState::default();
        let key = (TenantId::DEFAULT, NodeId::new(1));
        let latest: CommittedBytes = (10, 100, 200);
        let earlier: CommittedBytes = (10, 50, 100);
        let mut hist = HashSet::new();
        hist.insert(latest);
        hist.insert(earlier);
        s.any_history.insert(key, hist);
        s.latest_t1.insert(key, latest);
        s.total_commits = 2;
        s
    }

    fn fixture_recovered_clean(pre: &CommittedState) -> RecoveredState {
        let mut r = RecoveredState::default();
        for key in pre.any_history.keys() {
            let latest = pre.latest_t1.get(key).copied().unwrap_or((0, 0, 0));
            r.bytes_by_key.insert(*key, latest);
        }
        r
    }

    #[test]
    fn i1_record_format_mismatch_surfaces_ghost_bytes() {
        let pre = fixture_pre_crash();
        let mut rec = fixture_recovered_clean(&pre);
        let target = (TenantId::DEFAULT, NodeId::new(1));
        inject_record_format_mismatch(&mut rec, &pre, target);
        let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
            .expect_err("oracle must detect ghost bytes");
        assert!(
            matches!(err, OracleViolation::GhostBytes { .. }),
            "I-1 expected GhostBytes, got {err:?}"
        );
    }

    /// MEDIUM-1 fix-up reverse-test: verify the defensive
    /// non-coincidence assertion fires when the XOR'd corrupted triple
    /// is seeded into `pre_crash.any_history` BEFORE injection. This
    /// proves the symmetry-with-I-2/I-3/I-5 assertion is load-bearing,
    /// not decorative.
    #[test]
    #[should_panic(expected = "coincides with pre_crash.any_history")]
    fn i1_record_format_mismatch_asserts_non_coincidence() {
        let mut pre = fixture_pre_crash();
        let mut rec = fixture_recovered_clean(&pre);
        let target = (TenantId::DEFAULT, NodeId::new(1));
        // The clean recovered triple at `target` is `latest = (10, 100,
        // 200)`; XORing each component with 0xDEAD_BEEF yields the
        // adversarial coincident triple. Seed it into pre_crash's
        // history so the inject's assertion MUST fire.
        let coincident: CommittedBytes = (10 ^ 0xDEAD_BEEF, 100 ^ 0xDEAD_BEEF, 200 ^ 0xDEAD_BEEF);
        pre.any_history
            .get_mut(&target)
            .expect("fixture has history at target")
            .insert(coincident);
        inject_record_format_mismatch(&mut rec, &pre, target);
    }

    #[test]
    fn i2_key_format_mismatch_surfaces_unknown_key() {
        let pre = fixture_pre_crash();
        let mut rec = fixture_recovered_clean(&pre);
        let phantom_tenant = TenantId::new(9_999);
        let phantom_node = NodeId::new(u64::MAX);
        inject_key_format_mismatch(&mut rec, &pre, phantom_tenant, phantom_node, (42, 42, 42));
        let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
            .expect_err("oracle must detect unknown phantom key");
        assert!(
            matches!(err, OracleViolation::UnknownKey { .. }),
            "I-2 expected UnknownKey, got {err:?}"
        );
    }

    #[test]
    fn i3_mvcc_chain_layout_mismatch_surfaces_t1_strict_drift() {
        let pre = fixture_pre_crash();
        let mut rec = fixture_recovered_clean(&pre);
        let target = (TenantId::DEFAULT, NodeId::new(1));
        let earlier_t1: CommittedBytes = (10, 50, 100);
        inject_mvcc_chain_layout_mismatch(&mut rec, &pre, target, earlier_t1);
        let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
            .expect_err("oracle must detect T1 strict drift");
        assert!(
            matches!(err, OracleViolation::T1StrictDrift { .. }),
            "I-3 expected T1StrictDrift, got {err:?}"
        );
    }

    #[test]
    fn i4_wal_commit_bundle_atomicity_mismatch_surfaces_t1_missing() {
        let pre = fixture_pre_crash();
        let mut rec = fixture_recovered_clean(&pre);
        let target = (TenantId::DEFAULT, NodeId::new(1));
        inject_wal_commit_bundle_atomicity_mismatch(&mut rec, &pre, target);
        let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
            .expect_err("oracle must detect T1 missing post-recovery");
        assert!(
            matches!(err, OracleViolation::T1Missing { .. }),
            "I-4 expected T1Missing, got {err:?}"
        );
    }

    #[test]
    fn baseline_clean_state_returns_ok() {
        // Reverse-test sanity: clean state with no injection passes
        // the oracle. This proves the injection — not the workload —
        // is what surfaces each violation in the i1/i2/i3/i4 unit
        // tests above.
        let pre = fixture_pre_crash();
        let rec = fixture_recovered_clean(&pre);
        let report = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
            .expect("clean state must pass oracle");
        assert_eq!(report.t1_keys, 1);
        assert_eq!(report.t1_satisfied, 1);
    }
}
