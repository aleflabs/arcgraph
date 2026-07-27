//! K-1c+d encoding-mismatch I-3 (MVCC chain layout) replay test.
//!
//! Per ADR-038 amendment-03 §"Slice K" K-1d row + issue #215. Drives
//! the K-1 oracle through a workload + recovery pass with an injected
//! **MVCC chain-layout mismatch** at one `(TenantId, NodeId)` key —
//! the recovered store returns an EARLIER historical T1 commit's bytes
//! instead of the latest T1's bytes (chain-layout drift; per
//! ADR-018 MVCC-only alpha contract violated).
//!
//! ## Hard contract
//!
//! Asserts the strict-mode oracle returns
//! `Err(OracleViolation::T1StrictDrift { .. })` for the chain-layout
//! drift. Per
//! [`arcgraph_storage::test_harness::k1::encoding_mismatch::inject_mvcc_chain_layout_mismatch`]
//! the corruption installs an *earlier-but-historical* T1 byte triple
//! at the recovered position; the bytes ARE in `any_history` (so this
//! is NOT a ghost) but are NOT the latest T1 (so T1 strict drift
//! fires per the ADR-034 I-D1 contract enforced by the oracle).
//!
//! Workload pre-condition: the planned workload commits 2 distinct T1
//! triples per `(tenant, NodeId)` — one create, one overwrite. The
//! create's bytes are the earlier-historical T1; the overwrite's
//! bytes are the latest.
//!
//! ## Phase 4.3 reverse-test discipline (mandatory)
//!
//! 1. Run as-is → test PASSES (oracle DETECTS T1StrictDrift).
//! 2. Comment out the line marked `// REVERSE-TEST PIN: comment out
//!    to prove non-vacuity` → test FAILS.
//! 3. Restore → test PASSES.

use arcgraph_core::TenantId;
use arcgraph_storage::test_harness::k1::encoding_mismatch::inject_mvcc_chain_layout_mismatch;
use arcgraph_storage::test_harness::k1::oracle::{
    OracleConfig, OracleViolation, verify_post_recovery_invariants,
};

mod k1c_common;

#[test]
fn k1c_mvcc_chain_layout_mismatch_surfaces_t1_strict_drift() {
    let (_workspace, wal_dir) = k1c_common::fresh_workdir();

    // ── Phase 1: clean workload (≥2 T1 commits per key) ────────────
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

    // ── Phase 3: inject I-3 chain-layout drift ─────────────────────
    let target = k1c_common::pick_target_key(&pre);
    let older = k1c_common::older_t1_bytes_for(&pre, target);
    inject_mvcc_chain_layout_mismatch(&mut rec, &pre, target, older); // REVERSE-TEST PIN: comment out to prove non-vacuity

    // ── Phase 4: oracle MUST detect T1StrictDrift ──────────────────
    let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
        .expect_err("k1c I-3: strict oracle MUST detect chain-layout drift as T1StrictDrift");
    match err {
        OracleViolation::T1StrictDrift {
            tenant_raw,
            node_id_raw,
            observed,
            expected,
        } => {
            assert_eq!(tenant_raw, target.0.raw());
            assert_eq!(node_id_raw, target.1.raw());
            assert_eq!(
                observed, older,
                "T1StrictDrift observed bytes must match the injected older-T1"
            );
            assert_ne!(
                expected, older,
                "T1StrictDrift expected bytes are the latest T1 — must differ from the injected older-T1"
            );
        }
        other => panic!(
            "k1c I-3: expected OracleViolation::T1StrictDrift for chain-layout mismatch; got {other:?}"
        ),
    }

    recovered.shutdown();
}
