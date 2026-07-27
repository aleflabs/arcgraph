//! K-1c+d encoding-mismatch I-5 (cross-tenant serialization) replay
//! test.
//!
//! Per ADR-038 amendment-03 §"Slice K" K-1d row + issue #215. Drives
//! the K-1b cross-tenant oracle through a workload + recovery pass
//! with an injected **cross-tenant serialization mismatch** — bytes
//! committed under tenant A appear in tenant B's recovered state at
//! the same `NodeId`, even though tenant B's pre-crash ledger has no
//! such commit.
//!
//! ## Hard contract
//!
//! Asserts the strict-mode K-1b cross-tenant oracle
//! ([`verify_cross_tenant_invariants`]) surfaces
//! `OracleViolation::CrossTenantContamination { .. }` for the foreign
//! bytes. Per
//! [`arcgraph_storage::test_harness::k1::encoding_mismatch::inject_cross_tenant_serialization_mismatch`]
//! the corruption installs source-tenant bytes under destination-tenant
//! keying; the canonical detection signal — "foreign bytes EXIST in
//! source's `any_history` AND do NOT exist in destination's
//! `any_history`" — fires per the K-1b oracle's `pre_crash_per_tenant`
//! invariants.
//!
//! ## Why fail_fast=false
//!
//! Per the K-1b oracle's #219 CONCERN-3 carry-forward documented in
//! `oracle.rs::verify_cross_tenant_invariants` rustdoc: a true A→B
//! contamination simultaneously presents as (a) a
//! `CrossTenantContamination` violation in the cross-tenant pass AND
//! (b) an `UnknownKey` violation in B's per-tenant pre-pass. Under
//! `fail_fast=true` the FIRST returned violation may be either kind
//! depending on iteration order — so to assert SPECIFICALLY on
//! `CrossTenantContamination` we MUST set `fail_fast=false` and walk
//! the `cross_tenant_violations` vector. This is the same discipline
//! the K-1b cross-tenant fault isolation pin uses.
//!
//! ## Phase 4.3 reverse-test discipline (mandatory)
//!
//! 1. Run as-is → test PASSES (oracle DETECTS CrossTenantContamination).
//! 2. Comment out the line marked `// REVERSE-TEST PIN: comment out
//!    to prove non-vacuity` → test FAILS.
//! 3. Restore → test PASSES.

use std::collections::HashMap;

use arcgraph_core::{NodeId, TenantId};
use arcgraph_storage::test_harness::k1::encoding_mismatch::inject_cross_tenant_serialization_mismatch;
use arcgraph_storage::test_harness::k1::oracle::{
    CommittedBytes, CommittedState, CrossTenantOracleInput, OracleConfig, OracleViolation,
    RecoveredState, verify_cross_tenant_invariants,
};

mod k1c_common;

#[test]
fn k1c_cross_tenant_serialization_mismatch_surfaces_contamination() {
    let (_workspace, wal_dir) = k1c_common::fresh_workdir();

    // ── Phase 1: clean 2-tenant workload (disjoint label spaces) ───
    let tenant_source = TenantId::DEFAULT;
    let tenant_destination = TenantId::new(1_001);
    let plan = k1c_common::plan_workload(tenant_source, tenant_destination);
    let stack = k1c_common::K1cStack::build(&wal_dir);
    let allocated = k1c_common::run_workload(&stack, &plan);
    stack.shutdown();

    // ── Phase 2: clean recovery ────────────────────────────────────
    let recovered = k1c_common::K1cStack::recover(&wal_dir);
    let pre = k1c_common::build_pre_crash_state(&plan, &allocated);
    let labels = k1c_common::workload_labels(&plan);
    let rec = k1c_common::build_recovered_state(&recovered, &pre, &labels);
    k1c_common::assert_clean_recovery(&pre, &rec);

    // ── Phase 3: split per-tenant + inject cross-tenant byte ───────
    let (pre_by_tenant, mut rec_by_tenant) = k1c_common::split_by_tenant(pre, rec);
    let pre_source = pre_by_tenant
        .get(&tenant_source)
        .cloned()
        .expect("source tenant present in pre-crash split");
    let pre_destination = pre_by_tenant
        .get(&tenant_destination)
        .cloned()
        .expect("destination tenant present in pre-crash split");

    // Pick a SOURCE NodeId whose bytes do NOT collide with anything
    // in the destination's any_history at the same NodeId. The
    // workload uses disjoint label spaces (source labels start at
    // 100_000, destination at 200_000), so the bytes triples
    // (label, a, b) are guaranteed disjoint by label alone.
    let (source_key, foreign_bytes) = pick_source_byte_for_injection(&pre_source);
    let target_node = source_key.1;
    let foreign_bytes: CommittedBytes = foreign_bytes;

    inject_cross_tenant_serialization_mismatch(
        &mut rec_by_tenant,
        &pre_source,
        &pre_destination,
        tenant_source,
        tenant_destination,
        target_node,
        foreign_bytes,
    ); // REVERSE-TEST PIN: comment out to prove non-vacuity

    // ── Phase 4: oracle MUST detect CrossTenantContamination ───────
    let cfg = OracleConfig {
        fail_fast: false, // collect-all per #219 CONCERN-3 carry-forward
        ..OracleConfig::default()
    };
    let input = CrossTenantOracleInput {
        pre_crash_per_tenant: pre_by_tenant,
        recovered_per_tenant: rec_by_tenant,
    };
    let report = verify_cross_tenant_invariants(&input, &cfg)
        .expect("k1c I-5 collect-all run must return Ok(report) with violations populated");
    let mut saw_contamination = false;
    for v in &report.cross_tenant_violations {
        if let OracleViolation::CrossTenantContamination {
            source_raw,
            target_raw,
            key_raw,
            ..
        } = v
        {
            assert_eq!(*source_raw, tenant_source.raw());
            assert_eq!(*target_raw, tenant_destination.raw());
            assert_eq!(*key_raw, target_node.raw());
            saw_contamination = true;
        }
    }
    assert!(
        saw_contamination,
        "k1c I-5: expected at least one OracleViolation::CrossTenantContamination in cross_tenant_violations; \
         got {:?}",
        report.cross_tenant_violations
    );

    recovered.shutdown();
}

/// Pick a `(TenantId, NodeId)` key from the source tenant's pre-crash
/// any_history along with one of its committed bytes triples. Stable:
/// sorts the (key, bytes) pairs ascendingly and returns the first.
fn pick_source_byte_for_injection(
    pre_source: &CommittedState,
) -> ((TenantId, NodeId), CommittedBytes) {
    let mut rows: Vec<((TenantId, NodeId), CommittedBytes)> = Vec::new();
    for (key, history) in &pre_source.any_history {
        for bytes in history {
            rows.push((*key, *bytes));
        }
    }
    rows.sort_by_key(|((t, n), b)| (t.raw(), n.raw(), b.0, b.1, b.2));
    *rows
        .first()
        .expect("pick_source_byte_for_injection: source tenant has no committed history")
}

// Suppress unused-import warning for HashMap when the test compiler
// pulls only the helper-module re-exports.
#[allow(dead_code)]
fn _hashmap_marker() -> HashMap<TenantId, RecoveredState> {
    HashMap::new()
}
