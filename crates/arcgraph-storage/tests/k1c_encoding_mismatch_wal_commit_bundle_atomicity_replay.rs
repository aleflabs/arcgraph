//! K-1c+d encoding-mismatch I-4 (WAL commit-bundle atomicity) replay
//! test.
//!
//! Per ADR-038 amendment-03 §"Slice K" K-1d row + issue #215. Drives
//! the K-1 oracle through a workload + recovery pass with an injected
//! **WAL commit-bundle atomicity mismatch** at one `(TenantId, NodeId)`
//! key — the bundle's MVCC writes apply but the bundle is otherwise
//! lost so a T1 commit is COMPLETELY ABSENT post-recovery.
//!
//! ## Hard contract
//!
//! Asserts the strict-mode oracle returns
//! `Err(OracleViolation::T1Missing { .. })` for the dropped T1 commit.
//! Per
//! [`arcgraph_storage::test_harness::k1::encoding_mismatch::inject_wal_commit_bundle_atomicity_mismatch`]
//! the corruption removes the target `(tenant, NodeId)` key from
//! `RecoveredState::bytes_by_key`; the pre-crash ledger has a T1
//! latest at that key, so the oracle's `(None, Some(t1_bytes))` arm
//! fires `T1Missing` per the ADR-034 I-D1 strict contract.
//!
//! ## Why "T1 missing", not "torn-tail dropped"?
//!
//! ADR-031 §R5 specifies the codec drops torn-tail bundles cleanly —
//! that's a CORRECT behavior, not an encoding-mismatch failure mode.
//! The encoding-mismatch failure mode this test pins is a
//! HOSTILE corruption that bypasses CRC + length validation (e.g.,
//! same-byte XOR preserving CRC, or a higher-layer codec mis-encoding
//! that fails to surface the missing record). The oracle MUST detect
//! the resulting post-recovery absence regardless of cause.
//!
//! ## Phase 4.3 reverse-test discipline (mandatory)
//!
//! 1. Run as-is → test PASSES (oracle DETECTS T1Missing).
//! 2. Comment out the line marked `// REVERSE-TEST PIN: comment out
//!    to prove non-vacuity` → test FAILS.
//! 3. Restore → test PASSES.

use arcgraph_core::TenantId;
use arcgraph_storage::test_harness::k1::encoding_mismatch::inject_wal_commit_bundle_atomicity_mismatch;
use arcgraph_storage::test_harness::k1::oracle::{
    OracleConfig, OracleViolation, verify_post_recovery_invariants,
};

mod k1c_common;

#[test]
fn k1c_wal_commit_bundle_atomicity_mismatch_surfaces_t1_missing() {
    let (_workspace, wal_dir) = k1c_common::fresh_workdir();

    // ── Phase 1: clean workload ────────────────────────────────────
    let plan = k1c_common::plan_workload(TenantId::DEFAULT, TenantId::new(1_001));
    let stack = k1c_common::K1cStack::build(&wal_dir);
    let allocated = k1c_common::run_workload(&stack, &plan);
    stack.shutdown();

    // ── Phase 2: clean recovery ────────────────────────────────────
    let recovered = k1c_common::K1cStack::recover(&wal_dir);
    let pre = k1c_common::build_pre_crash_state(&plan, &allocated);
    let labels = k1c_common::workload_labels(&plan);
    let mut rec = k1c_common::build_recovered_state(&recovered, &pre, &labels);
    k1c_common::assert_clean_recovery(&pre, &rec);

    // ── Phase 3: inject I-4 commit-bundle atomicity mismatch ───────
    let target = k1c_common::pick_target_key(&pre);
    let expected_t1 = *pre
        .latest_t1
        .get(&target)
        .expect("k1c I-4: target must have a T1 latest pre-injection");
    inject_wal_commit_bundle_atomicity_mismatch(&mut rec, &pre, target); // REVERSE-TEST PIN: comment out to prove non-vacuity

    // ── Phase 4: oracle MUST detect T1Missing ──────────────────────
    let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default()).expect_err(
        "k1c I-4: strict oracle MUST detect commit-bundle atomicity mismatch as T1Missing",
    );
    match err {
        OracleViolation::T1Missing {
            tenant_raw,
            node_id_raw,
            expected,
        } => {
            assert_eq!(tenant_raw, target.0.raw());
            assert_eq!(node_id_raw, target.1.raw());
            assert_eq!(
                expected, expected_t1,
                "T1Missing expected bytes must match the dropped T1 commit's latest bytes"
            );
        }
        other => panic!(
            "k1c I-4: expected OracleViolation::T1Missing for commit-bundle atomicity mismatch; got {other:?}"
        ),
    }

    recovered.shutdown();
}
