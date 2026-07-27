//! K-1c+d encoding-mismatch I-1 (record format) replay test.
//!
//! Per ADR-038 amendment-03 §"Slice K" K-1d row + issue #215. Drives
//! the K-1 oracle through a workload + recovery pass with an injected
//! **record-format mismatch** at one `(TenantId, NodeId)` key — the
//! recovered store returns bytes that no historical commit ever wrote
//! at that key.
//!
//! ## Hard contract
//!
//! Asserts the strict-mode oracle (`OracleConfig::default()` —
//! `stats_inconsistency_fatal=true`, `fail_fast=true`) returns
//! `Err(OracleViolation::GhostBytes { .. })` for the injected
//! record-format corruption. Per
//! [`arcgraph_storage::test_harness::k1::encoding_mismatch::inject_record_format_mismatch`]
//! the corruption is `(label, a, b) ⊕ 0xDEAD_BEEF` componentwise; the
//! oracle's I-V1 ghost-byte check fires because the XOR'd triple is
//! not in `pre_crash.any_history` for that key.
//!
//! ## Phase 4.3 reverse-test discipline (mandatory)
//!
//! 1. Run as-is → test PASSES (oracle DETECTS GhostBytes).
//! 2. Comment out the line marked `// REVERSE-TEST PIN: comment out
//!    to prove non-vacuity` → test FAILS (oracle returns Ok(_); the
//!    `expect_err` below panics).
//! 3. Restore → test PASSES.
//!
//! The K-1c+d review packet captures all three runs as the
//! non-vacuity proof for this invariant.
//!
//! Run:
//!
//! ```ignore
//! cargo test -p arcgraph-storage --release \
//!   --test k1c_encoding_mismatch_record_format_replay
//! ```

use arcgraph_core::TenantId;
use arcgraph_storage::test_harness::k1::encoding_mismatch::inject_record_format_mismatch;
use arcgraph_storage::test_harness::k1::oracle::{
    OracleConfig, OracleViolation, verify_post_recovery_invariants,
};

mod k1c_common;

#[test]
fn k1c_record_format_mismatch_surfaces_ghost_bytes() {
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

    // Sanity: clean recovery passes oracle. Any subsequent violation
    // is attributable to the injection below, not a pre-existing bug.
    k1c_common::assert_clean_recovery(&pre, &rec);

    // ── Phase 3: inject I-1 record-format mismatch ─────────────────
    let target = k1c_common::pick_target_key(&pre);
    inject_record_format_mismatch(&mut rec, &pre, target); // REVERSE-TEST PIN: comment out to prove non-vacuity

    // ── Phase 4: oracle MUST detect GhostBytes ─────────────────────
    let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
        .expect_err("k1c I-1: strict oracle MUST detect record-format mismatch as GhostBytes");
    match err {
        OracleViolation::GhostBytes {
            tenant_raw,
            node_id_raw,
            ..
        } => {
            assert_eq!(
                tenant_raw,
                target.0.raw(),
                "GhostBytes must fire on the corrupted target tenant"
            );
            assert_eq!(
                node_id_raw,
                target.1.raw(),
                "GhostBytes must fire on the corrupted target NodeId"
            );
        }
        other => panic!(
            "k1c I-1: expected OracleViolation::GhostBytes for record-format mismatch; got {other:?}"
        ),
    }

    recovered.shutdown();
}
