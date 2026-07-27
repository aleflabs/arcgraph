//! K-1c+d encoding-mismatch I-2 (key format) replay test.
//!
//! Per ADR-038 amendment-03 §"Slice K" K-1d row + issue #215. Drives
//! the K-1 oracle through a workload + recovery pass with an injected
//! **key-format mismatch** — the recovered store materialises a
//! `(TenantId, NodeId)` key the pre-crash ledger NEVER wrote (the
//! ghost-direction of the 1:1 unique:total invariant; codex H-2 fix
//! per `oracle.rs` lines 497–516).
//!
//! ## Hard contract
//!
//! Asserts the strict-mode oracle returns
//! `Err(OracleViolation::UnknownKey { .. })` for the phantom key.
//! Per [`arcgraph_storage::test_harness::k1::encoding_mismatch::inject_key_format_mismatch`]
//! the corruption inserts `(TenantId::new(9_999), NodeId::new(u64::MAX))`
//! into `RecoveredState::bytes_by_key` — a pair the workload never
//! touches — so the `UnknownKey` arm of the oracle's ghost-direction
//! pass fires.
//!
//! ## Phase 4.3 reverse-test discipline (mandatory)
//!
//! 1. Run as-is → test PASSES (oracle DETECTS UnknownKey).
//! 2. Comment out the line marked `// REVERSE-TEST PIN: comment out
//!    to prove non-vacuity` → test FAILS.
//! 3. Restore → test PASSES.

use arcgraph_core::{NodeId, TenantId};
use arcgraph_storage::test_harness::k1::encoding_mismatch::inject_key_format_mismatch;
use arcgraph_storage::test_harness::k1::oracle::{
    OracleConfig, OracleViolation, verify_post_recovery_invariants,
};

mod k1c_common;

#[test]
fn k1c_key_format_mismatch_surfaces_unknown_key() {
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

    // ── Phase 3: inject I-2 key-format mismatch ────────────────────
    //
    // The phantom (tenant=9_999, node=u64::MAX) pair is intentionally
    // outside the workload's tenant set ({DEFAULT=1, 1_001}) and
    // outside any NodeId allocator's first-million-element range.
    // The key is verifiably absent from the pre-crash ledger.
    let phantom_tenant = TenantId::new(9_999);
    let phantom_node = NodeId::new(u64::MAX);
    let phantom_bytes = (0xCAFE_BABE_u32, 0xFEED_FACE_u32, 0xDEAD_F00D_u32);
    inject_key_format_mismatch(&mut rec, &pre, phantom_tenant, phantom_node, phantom_bytes); // REVERSE-TEST PIN: comment out to prove non-vacuity

    // ── Phase 4: oracle MUST detect UnknownKey ─────────────────────
    let err = verify_post_recovery_invariants(&pre, &rec, &OracleConfig::default())
        .expect_err("k1c I-2: strict oracle MUST detect key-format mismatch as UnknownKey");
    match err {
        OracleViolation::UnknownKey {
            tenant_raw,
            node_id_raw,
            observed,
        } => {
            assert_eq!(
                tenant_raw,
                phantom_tenant.raw(),
                "UnknownKey must fire on the phantom tenant"
            );
            assert_eq!(
                node_id_raw,
                phantom_node.raw(),
                "UnknownKey must fire on the phantom NodeId"
            );
            assert_eq!(
                observed, phantom_bytes,
                "UnknownKey observed bytes must match the injected phantom"
            );
        }
        other => panic!(
            "k1c I-2: expected OracleViolation::UnknownKey for key-format mismatch; got {other:?}"
        ),
    }

    recovered.shutdown();
}
