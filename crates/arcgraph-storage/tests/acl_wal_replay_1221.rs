//! #1221 (ADR-218) — storage-level pins that WAL replay reconstructs the
//! [`PermissionIndex`] document-ACL enforcement state from the v8+
//! `CommitBundle` `acl_grants` section, AND that the op is **atomic with
//! its commit** (the crash-before-commit "both-or-neither" property).
//!
//! `PermissionIndex` is in-memory only at stage-1 — a bare `serve --data`
//! restart without an auxiliary seed path comes up deny-all (every doc
//! UNCLASSIFIED). The ADR-218 fold folds each `apply_doc_acl` /
//! `revoke_doc` write-through into the WAL's `acl_grants` section and
//! replays it on open. This file isolates the replay arm: a v8 bundle's
//! `acl_grants` re-drives `apply_doc_acl_replayed` / `revoke_doc_replayed`
//! into the served index when wired via
//! [`PageStoreTarget::with_permission_index`].
//!
//! THE MANDATORY ORACLE (Director-required, RED-on-revert):
//! `granted_acl_survives_restart_and_non_granted_denies_all` sets an ACL
//! via the durable write-through ([`PermissionIndex::apply_doc_acl`] with a
//! real [`CrudAclWalSink`]) → commits/fsyncs → re-opens the index off the
//! SAME WAL dir (NO re-seed) → asserts the granted principal sees EXACTLY
//! the granted docs and a non-granted principal denies-all. Reverting the
//! fold (skip `with_permission_index`, i.e. the no-index path) makes the
//! reopened index empty ⇒ deny-all (RED) — proven by
//! `acl_replay_without_index_is_noop`.

use std::collections::BTreeSet;
use std::sync::Arc;

use arcgraph_core::{Lsn, NodeId, TenantId};
use arcgraph_storage::crud::{CrudAclWalSink, CrudStore};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::permissions::{PUBLIC_PRINCIPAL, PermissionIndex};
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AclGrantEntry, AclGrantOp, PageStoreTarget, PrimaryPageStoreHandle, WalConfig, WalRecordType,
    WalWriter, encode_commit_bundle_v8, list_segments, recover_from_wal, segment_filename,
};
use tempfile::TempDir;

fn test_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn grants(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// Append a single hand-crafted v8 `CommitBundle` record carrying only an
/// `acl_grants` section (empty MVCC write-set is legal — the ACL ops ride
/// the bundle's trailing section).
fn write_v8_acl_bundle(
    wal_dir: &std::path::Path,
    tenant: TenantId,
    commit_lsn: Lsn,
    acl_grants: &[AclGrantEntry],
) {
    let payload = encode_commit_bundle_v8(
        commit_lsn,
        tenant,
        &std::collections::HashMap::new(),
        &[],
        &[],
        &[],
        &[],
        &[],
        acl_grants,
    );
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    handle
        .append(WalRecordType::CommitBundle, 0, 0, tenant, payload)
        .expect("append CommitBundle");
    writer.shutdown().unwrap();
}

/// Recover the WAL at `wal_dir` into a throwaway primary store, wiring
/// `index` as the permission index iff `Some`. Returns the recovery report
/// so callers can read `applied_commit_lsn` + `acl_grants_recovered`.
fn recover_into(
    wal_dir: &std::path::Path,
    index: Option<Arc<PermissionIndex>>,
) -> arcgraph_storage::wal::RecoveryReport {
    let mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary"));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let mut target = PageStoreTarget::primary_only(primary_handle);
    if let Some(idx) = index {
        target = target.with_permission_index(idx);
    }
    recover_from_wal(wal_dir, mgr, target, None).expect("recover_from_wal")
}

// ─────────────────────────────────────────────────────────────────────
// THE ORACLE — granted ACL survives a bare restart; non-granted denies.
// Drives the FULL write-through: a real CrudAclWalSink fires the durable
// commit from inside `apply_doc_acl`, exactly as the seed/live path does.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn granted_acl_survives_restart_and_non_granted_denies_all() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    // ── Process 1: set ACLs via the durable write-through, then "crash"
    //    (drop the writer, which fsyncs on shutdown). ──
    {
        let mgr = Arc::new(TxnManager::new());
        let alloc = Arc::new(PageAllocator::new());
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        // The WAL must be attached to the TxnManager (Phase 2 reads the
        // handle off the manager) AND the CrudStore — exactly the durable
        // bootstrap wiring (bootstrap.rs §6: txn_manager.attach_wal +
        // crud_store.attach_wal).
        mgr.attach_wal(writer.handle());
        // TEST-HARNESS SYMMETRY (not a production invariant): the
        // `recover_into` helper below builds a `PrimaryIndex` on the SAME
        // recovery `TxnManager` BEFORE calling `recover_from_wal`, and that
        // root-page alloc advances the recovery manager's `current_lsn` to
        // 1 — so the replay baseline (`applied_high_water`) seeds at 1, not
        // 0, in THIS test setup. To keep the write side symmetric (so an
        // ACL commit is never `<= 1` and thus never skip-if-applied), the
        // write side ALSO builds a PrimaryIndex here (consuming write-side
        // commit_lsn 1 for its root) so the first ACL write-through commit
        // lands at lsn ≥ 2.
        //
        // NOTE: in the PRODUCTION durable bootstrap the recovery baseline
        // is `Lsn::ZERO` (the bootstrap recovers a fresh-at-ZERO
        // `TxnManager`, BEFORE any PrimaryIndex attaches — bootstrap.rs §5
        // runs before §7), so even an `acl_grants` commit at the lowest
        // real LSN (1) is `1 <= 0 == false` ⇒ always applied. This
        // PrimaryIndex pairing is purely a test-manager-sharing artifact,
        // NOT a fold bug and NOT a production ordering requirement.
        let primary =
            Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).unwrap());
        let mut s = CrudStore::new();
        s.attach_wal(writer.handle());
        s.attach_primary_index(Arc::clone(&primary));
        let store = Arc::new(s);

        let index = PermissionIndex::new();
        index.set_wal_sink(Arc::new(CrudAclWalSink::new(
            Arc::clone(&mgr),
            Arc::clone(&store),
            tenant,
        )));

        // doc 1 → alice; doc 2 → bob; doc 3 → __public__.
        index.apply_doc_acl(NodeId::new(1), grants(&["alice"]));
        index.apply_doc_acl(NodeId::new(2), grants(&["bob"]));
        index.apply_doc_acl(NodeId::new(3), grants(&[PUBLIC_PRINCIPAL]));
        // doc 5 → alice, then revoke-by-removal (Apply then Revoke).
        index.apply_doc_acl(NodeId::new(5), grants(&["alice"]));
        index.revoke_doc(NodeId::new(5));

        // Sanity in-process before restart.
        assert!(index.effective("alice").is_visible(NodeId::new(1)));
        assert!(!index.effective("alice").is_visible(NodeId::new(5)));

        writer.shutdown().unwrap();
    }

    // ── Process 2 (bare restart): re-open a FRESH index off the SAME WAL
    //    dir, NO re-seed. Replay must rebuild enforcement. ──
    let recovered = Arc::new(PermissionIndex::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&recovered)));

    // ≥1 ACL op replayed (the observable proof ACLs survived the bounce).
    assert!(
        report.metrics.acl_grants_recovered >= 4,
        "expected ≥4 acl ops recovered (apply×4 + revoke×1), got {}",
        report.metrics.acl_grants_recovered,
    );

    // GRANTED principal sees EXACTLY the granted docs.
    let alice = recovered.effective("alice");
    assert!(alice.is_visible(NodeId::new(1)), "alice keeps doc 1");
    assert!(!alice.is_visible(NodeId::new(2)), "alice never had doc 2");
    assert!(alice.is_visible(NodeId::new(3)), "public doc 3 visible");
    assert!(
        !alice.is_visible(NodeId::new(5)),
        "doc 5 was Apply-then-Revoke ⇒ UNCLASSIFIED ⇒ invisible (revoke durability)"
    );

    let bob = recovered.effective("bob");
    assert!(bob.is_visible(NodeId::new(2)), "bob keeps doc 2");
    assert!(!bob.is_visible(NodeId::new(1)), "bob never had doc 1");
    assert!(
        bob.is_visible(NodeId::new(3)),
        "public doc 3 visible to bob"
    );

    // NON-GRANTED principal denies-all (except public docs).
    let mallory = recovered.effective("mallory");
    assert!(!mallory.is_visible(NodeId::new(1)));
    assert!(!mallory.is_visible(NodeId::new(2)));
    assert!(
        mallory.is_visible(NodeId::new(3)),
        "public doc is visible to everyone (the only exception)"
    );
    assert!(!mallory.is_visible(NodeId::new(5)));
    // A doc that never got any grant is UNCLASSIFIED for everyone.
    assert!(!mallory.is_visible(NodeId::new(99)));
    assert!(!alice.is_visible(NodeId::new(99)));
}

// ─────────────────────────────────────────────────────────────────────
// RED-on-revert proof: with NO index wired (= the fold reverted), replay
// is a no-op ⇒ the reopened index is empty ⇒ DENY-ALL. This is the exact
// state the bug (#1221) produced. The oracle above goes RED if the
// `with_permission_index` wiring (or the apply arm) is removed.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn acl_replay_without_index_is_noop_deny_all() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    write_v8_acl_bundle(
        &wal_dir,
        tenant,
        Lsn::new(5),
        &[AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant,
            doc: NodeId::new(1),
            grants: grants(&["alice"]),
        }],
    );

    // Recover WITHOUT wiring a permission index: the bundle still applies
    // (its commit recovers) but the acl_grants apply arm is skipped.
    let report = recover_into(&wal_dir, None);
    assert_eq!(
        report.metrics.acl_grants_recovered, 0,
        "no index wired ⇒ acl apply arm is a no-op (the #1221 deny-all bug state)"
    );

    // A freshly-wired index that did NOT receive replay is empty ⇒
    // deny-all (UNCLASSIFIED for every principal).
    let fresh = PermissionIndex::new();
    assert!(
        !fresh.effective("alice").is_visible(NodeId::new(1)),
        "without replay, even the granted principal denies-all"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Clean recovery via a hand-crafted v8 bundle (apply + revoke), in order.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn acl_grants_recovered_from_v8_bundle_replay() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    write_v8_acl_bundle(
        &wal_dir,
        tenant,
        Lsn::new(5),
        &[
            AclGrantEntry {
                op: AclGrantOp::Apply,
                tenant,
                doc: NodeId::new(7),
                grants: grants(&["alice", "bob"]),
            },
            // Same doc, narrowed — last (this) wins.
            AclGrantEntry {
                op: AclGrantOp::Apply,
                tenant,
                doc: NodeId::new(7),
                grants: grants(&["alice"]),
            },
        ],
    );

    let index = Arc::new(PermissionIndex::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&index)));
    assert_eq!(report.metrics.acl_grants_recovered, 2);
    assert_eq!(report.applied_commit_lsn, Lsn::new(5));

    // Last-writer-wins per doc (append order): bob was narrowed out.
    assert!(index.effective("alice").is_visible(NodeId::new(7)));
    assert!(
        !index.effective("bob").is_visible(NodeId::new(7)),
        "the narrower (later, in-order) Apply wins — bob denied"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Crash-before-commit — a TORN bundle drops the ACL op (no torn state).
// fail-closed: the doc stays UNCLASSIFIED ⇒ invisible (no partial widen).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn torn_acl_bundle_drops_grant_fail_closed() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    write_v8_acl_bundle(
        &wal_dir,
        tenant,
        Lsn::new(5),
        &[AclGrantEntry {
            op: AclGrantOp::Apply,
            tenant,
            doc: NodeId::new(1),
            grants: grants(&["alice"]),
        }],
    );

    // Truncate the tail so the single bundle record is torn.
    let segs = list_segments(&wal_dir).unwrap();
    let last = *segs.last().unwrap();
    let path = wal_dir.join(segment_filename(last));
    let len = std::fs::metadata(&path).unwrap().len();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .set_len(len.saturating_sub(8))
        .unwrap();

    let index = Arc::new(PermissionIndex::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&index)));

    assert_eq!(
        report.metrics.acl_grants_recovered, 0,
        "torn bundle MUST NOT replay the ACL op"
    );
    assert!(
        !index.effective("alice").is_visible(NodeId::new(1)),
        "fail-closed: a torn ACL write leaves the doc UNCLASSIFIED (no partial widen)"
    );
    assert!(
        report.torn_tail.is_some(),
        "the truncated bundle is a torn tail, not a corruption halt"
    );
}

// ─────────────────────────────────────────────────────────────────────
// Cross-tenant isolation — an entry whose tenant != the bundle's tenant
// is skipped (defense-in-depth, ADR-212 §5 Q3).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn acl_entry_for_mismatched_tenant_is_skipped() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let bundle_tenant = TenantId::DEFAULT;
    let other_tenant = TenantId::new(bundle_tenant.raw() + 1);

    write_v8_acl_bundle(
        &wal_dir,
        bundle_tenant,
        Lsn::new(5),
        &[
            AclGrantEntry {
                op: AclGrantOp::Apply,
                tenant: bundle_tenant,
                doc: NodeId::new(1),
                grants: grants(&["alice"]),
            },
            // Wrong tenant — must be skipped, not applied to this index.
            AclGrantEntry {
                op: AclGrantOp::Apply,
                tenant: other_tenant,
                doc: NodeId::new(2),
                grants: grants(&["alice"]),
            },
        ],
    );

    let index = Arc::new(PermissionIndex::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&index)));

    // Only the matching-tenant entry replayed.
    assert_eq!(report.metrics.acl_grants_recovered, 1);
    assert!(index.effective("alice").is_visible(NodeId::new(1)));
    assert!(
        !index.effective("alice").is_visible(NodeId::new(2)),
        "a cross-tenant entry must NOT land in this tenant's index"
    );
}
