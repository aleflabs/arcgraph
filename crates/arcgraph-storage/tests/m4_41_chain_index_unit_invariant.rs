//! Issue #238 — per-tenant chain index unit invariants.
//!
//! Companion file `m4_41_chain_index_stress.rs` carries the
//! cold-start rebuild acceptance-gate stress test (K=50 N=200K —
//! 200K per tenant, 10M aggregate; sub-watermark per amendment-06
//! §D-25.2's per-tenant 10M trigger — plus the Tier-1
//! relative gate at K=50 N=50K); this file pins the correctness
//! invariants the stress test relies on:
//!
//! 1. **Index population on commit** — production commit pushes a
//!    `(tenant, key)` pair into `tenant_chain_keys[tenant]` so the
//!    per-tenant rebuild driver finds it.
//! 2. **Index population on replay** — replay path
//!    (`apply_replay_mvcc_write`) populates the index alongside the
//!    chain it rebuilds; post-replay `for_each_visible_record`
//!    returns the replayed records.
//! 3. **Index population on sidechannel apply** — sidechannel writes
//!    (typically SYSTEM-tenant root-pointer mutations) populate the
//!    index so the rebuild driver finds the SYSTEM tenant alongside
//!    user tenants.
//! 4. **Cross-tenant isolation** — keys committed under tenant A
//!    never appear in tenant B's `for_each_visible_record` callback
//!    invocations, regardless of MvccKey collisions.
//! 5. **Tombstone visibility under index** — a tombstone-superseded
//!    key returns no callback (the visibility filter handles the
//!    "latest visible is None" case correctly even when the key is
//!    in the index).
//! 6. **Replay idempotent path** — Lemma I1 idempotent skip does
//!    not double-insert into the index.
//! 7. **Empty-tenant query** — `for_each_visible_record` for a
//!    tenant with no chains is a zero-allocation no-op.
//! 8. **Concurrency** — concurrent commits across tenants populate
//!    the index correctly under DashMap's lock-free shard discipline.
//!
//! The PR #243 round-2 MED-1 regression test (GC + concurrent
//! commit deterministic interleave) lives in
//! `crate::transaction::tests::gc_does_not_drop_chain_index_entry_under_racing_commit`.
//! It uses private-field access (`m.versions.insert(...)` to
//! synthesise an already-expired chain) which is unavailable across
//! the integration-test boundary; the in-module placement mirrors
//! the existing
//! `gc_pruned_keys_excludes_racing_repopulations` pattern.

use arcgraph_core::{Lsn, TenantId};
use arcgraph_storage::transaction::{ReplayApplyOutcome, TxnManager};
use bytes::Bytes;

const KEY_A: u64 = 100;
const KEY_B: u64 = 200;
const KEY_C: u64 = 300;

fn tenant(raw: u64) -> TenantId {
    TenantId::new(raw)
}

// ─────────────────────────────────────────────────────────────────
// Test 1 — index populated on commit; visible records walk.
// ─────────────────────────────────────────────────────────────────

#[test]
fn commit_path_populates_index_and_visible_walk_returns_committed_records() {
    let mgr = TxnManager::new();
    let t = tenant(1);

    let mut tx = mgr.begin(t);
    tx.write(KEY_A, Bytes::from_static(b"alpha"));
    tx.write(KEY_B, Bytes::from_static(b"beta"));
    tx.commit().unwrap();

    // Index population invariant.
    assert!(
        mgr.tenant_chain_index_contains(t, KEY_A),
        "commit must register KEY_A in the per-tenant chain index"
    );
    assert!(
        mgr.tenant_chain_index_contains(t, KEY_B),
        "commit must register KEY_B in the per-tenant chain index"
    );
    assert_eq!(
        mgr.tenant_chain_key_count(t),
        2,
        "exactly the committed keys should be indexed for tenant {t:?}"
    );

    // Visible-walk semantics — what the rebuild driver consumes.
    let snap = mgr.current_lsn();
    let mut seen: Vec<(u64, Vec<u8>)> = Vec::new();
    mgr.for_each_visible_record(t, snap, |k, v| seen.push((k, v.to_vec())));
    seen.sort_by_key(|(k, _)| *k);
    assert_eq!(
        seen,
        vec![(KEY_A, b"alpha".to_vec()), (KEY_B, b"beta".to_vec())],
        "for_each_visible_record must yield exactly the committed live records"
    );
}

// ─────────────────────────────────────────────────────────────────
// Test 2 — replay path populates the index.
// ─────────────────────────────────────────────────────────────────

#[test]
fn replay_path_populates_index_and_visible_walk_after_seed() {
    let mgr = TxnManager::new();
    let t = tenant(7);

    // Replay three records at strictly-monotone commit LSNs (per
    // ADR-032 §R2 the executor sorts bundles before apply).
    let outcomes = [
        mgr.apply_replay_mvcc_write(
            Lsn::new(10),
            t,
            KEY_A,
            Some(Bytes::from_static(b"a-replay")),
        ),
        mgr.apply_replay_mvcc_write(
            Lsn::new(11),
            t,
            KEY_B,
            Some(Bytes::from_static(b"b-replay")),
        ),
        mgr.apply_replay_mvcc_write(
            Lsn::new(12),
            t,
            KEY_C,
            Some(Bytes::from_static(b"c-replay")),
        ),
    ];
    for o in outcomes {
        assert_eq!(o, ReplayApplyOutcome::Applied);
    }
    mgr.seed_after_replay(Lsn::new(12));

    // Index populated for each replayed key.
    for k in [KEY_A, KEY_B, KEY_C] {
        assert!(
            mgr.tenant_chain_index_contains(t, k),
            "replay must register key {k} in the per-tenant chain index"
        );
    }
    assert_eq!(mgr.tenant_chain_key_count(t), 3);

    // Visible-walk after seed_after_replay returns all three records.
    let mut seen: Vec<(u64, Vec<u8>)> = Vec::new();
    mgr.for_each_visible_record(t, mgr.current_lsn(), |k, v| seen.push((k, v.to_vec())));
    seen.sort_by_key(|(k, _)| *k);
    assert_eq!(
        seen,
        vec![
            (KEY_A, b"a-replay".to_vec()),
            (KEY_B, b"b-replay".to_vec()),
            (KEY_C, b"c-replay".to_vec()),
        ],
    );
}

// ─────────────────────────────────────────────────────────────────
// Test 3 — sidechannel apply populates the index.
// ─────────────────────────────────────────────────────────────────

#[test]
fn sidechannel_apply_populates_index() {
    let mgr = TxnManager::new();

    // Sidechannel apply primitive — bypasses Phase 3 `visible.store`
    // because the production caller (`commit_with_bundle_writes`
    // Phase 3) advances visible after applying. We assert (a) index
    // population (the property under test) and (b) that `read_at`
    // sees the version at its commit_lsn (sanity check that the
    // chain push happened).
    let sys = TenantId::SYSTEM;
    mgr.apply_sidechannel_mvcc_write(Lsn::new(5), sys, KEY_A, Some(Bytes::from_static(b"root")));

    assert!(
        mgr.tenant_chain_index_contains(sys, KEY_A),
        "sidechannel apply must register the (tenant, key) pair in the index"
    );
    assert_eq!(
        mgr.read_at(sys, KEY_A, Lsn::new(5)),
        Some(Bytes::from_static(b"root")),
        "sidechannel-installed version must be visible at its commit_lsn"
    );
    // Cross-tenant isolation — a different tenant's index is empty.
    assert_eq!(mgr.tenant_chain_key_count(tenant(99)), 0);
}

// ─────────────────────────────────────────────────────────────────
// Test 4 — cross-tenant isolation: same MvccKey under different
// tenants does not pollute either tenant's visible-walk.
// ─────────────────────────────────────────────────────────────────

#[test]
fn cross_tenant_isolation_preserved_under_index() {
    let mgr = TxnManager::new();
    let t_a = tenant(1);
    let t_b = tenant(2);

    // Same MvccKey under both tenants.
    let mut tx_a = mgr.begin(t_a);
    tx_a.write(KEY_A, Bytes::from_static(b"a-tenant-a"));
    tx_a.commit().unwrap();

    let mut tx_b = mgr.begin(t_b);
    tx_b.write(KEY_A, Bytes::from_static(b"a-tenant-b"));
    tx_b.commit().unwrap();

    // Each tenant's index has exactly one key.
    assert_eq!(mgr.tenant_chain_key_count(t_a), 1);
    assert_eq!(mgr.tenant_chain_key_count(t_b), 1);
    assert!(mgr.tenant_chain_index_contains(t_a, KEY_A));
    assert!(mgr.tenant_chain_index_contains(t_b, KEY_A));

    // Visible walk for tenant A only sees tenant A's value.
    let snap = mgr.current_lsn();
    let mut seen_a: Vec<Vec<u8>> = Vec::new();
    mgr.for_each_visible_record(t_a, snap, |k, v| {
        assert_eq!(k, KEY_A);
        seen_a.push(v.to_vec());
    });
    assert_eq!(seen_a, vec![b"a-tenant-a".to_vec()]);

    // Visible walk for tenant B only sees tenant B's value.
    let mut seen_b: Vec<Vec<u8>> = Vec::new();
    mgr.for_each_visible_record(t_b, snap, |k, v| {
        assert_eq!(k, KEY_A);
        seen_b.push(v.to_vec());
    });
    assert_eq!(seen_b, vec![b"a-tenant-b".to_vec()]);

    // tenants_with_chains enumerates both — sorted by raw TenantId.
    assert_eq!(mgr.tenants_with_chains(), vec![t_a, t_b]);
}

// ─────────────────────────────────────────────────────────────────
// Test 5 — tombstone-superseded key produces no callback even
// though the key is in the index. Pins the staleness contract: the
// visibility filter handles "latest visible is None" correctly.
// ─────────────────────────────────────────────────────────────────

#[test]
fn tombstone_supersession_under_index_emits_no_callback() {
    let mgr = TxnManager::new();
    let t = tenant(3);

    let mut tx = mgr.begin(t);
    tx.write(KEY_A, Bytes::from_static(b"v1"));
    tx.commit().unwrap();
    let mut tx = mgr.begin(t);
    tx.delete(KEY_A); // tombstone
    tx.commit().unwrap();

    // Index entry remains (chain is non-empty: PUT + tombstone).
    assert!(mgr.tenant_chain_index_contains(t, KEY_A));

    // Visibility — the latest version at current_lsn is the tombstone
    // (`value == None`), so for_each_visible_record skips this key.
    let snap = mgr.current_lsn();
    let mut callbacks = 0u32;
    mgr.for_each_visible_record(t, snap, |_, _| callbacks += 1);
    assert_eq!(
        callbacks, 0,
        "tombstoned key must not produce a callback under the index"
    );
}

// ─────────────────────────────────────────────────────────────────
// Test 6 — historical visibility: an older snapshot still sees the
// pre-tombstone value, even though the index contains the same key
// post-tombstone.
// ─────────────────────────────────────────────────────────────────

#[test]
fn historical_snapshot_sees_pre_tombstone_value_under_index() {
    let mgr = TxnManager::new();
    let t = tenant(4);

    let mut tx = mgr.begin(t);
    tx.write(KEY_A, Bytes::from_static(b"v1"));
    let lsn1 = tx.commit().unwrap();

    let mut tx = mgr.begin(t);
    tx.delete(KEY_A);
    tx.commit().unwrap();

    // At lsn1 (pre-tombstone), the live PUT is visible.
    let mut seen: Vec<Vec<u8>> = Vec::new();
    mgr.for_each_visible_record(t, lsn1, |_, v| seen.push(v.to_vec()));
    assert_eq!(seen, vec![b"v1".to_vec()]);
}

// ─────────────────────────────────────────────────────────────────
// Test 7 — empty-tenant query: no allocations, no callbacks.
// ─────────────────────────────────────────────────────────────────

#[test]
fn for_each_visible_record_on_tenant_with_no_chains_returns_nothing() {
    let mgr = TxnManager::new();
    let t = tenant(99);

    let mut callbacks = 0u32;
    mgr.for_each_visible_record(t, mgr.current_lsn(), |_, _| callbacks += 1);
    assert_eq!(callbacks, 0);
    assert_eq!(mgr.tenant_chain_key_count(t), 0);
    assert!(mgr.tenants_with_chains().is_empty());
}

// ─────────────────────────────────────────────────────────────────
// Test 8 — multi-tenant tenants_with_chains determinism.
// ─────────────────────────────────────────────────────────────────

#[test]
fn tenants_with_chains_returns_sorted_unique_tenants_post_index_lookup() {
    let mgr = TxnManager::new();
    // Insert in non-sorted order across tenants 5, 1, 3, 5, 1.
    for (raw, key, val) in [
        (5u64, KEY_A, b"5a".as_slice()),
        (1, KEY_A, b"1a".as_slice()),
        (3, KEY_A, b"3a".as_slice()),
        (5, KEY_B, b"5b".as_slice()),
        (1, KEY_C, b"1c".as_slice()),
    ] {
        let mut tx = mgr.begin(tenant(raw));
        tx.write(key, Bytes::copy_from_slice(val));
        tx.commit().unwrap();
    }
    let tenants = mgr.tenants_with_chains();
    assert_eq!(tenants, vec![tenant(1), tenant(3), tenant(5)]);
}

// ─────────────────────────────────────────────────────────────────
// Test 9 — replay idempotent path does NOT double-insert.
// ─────────────────────────────────────────────────────────────────

#[test]
fn replay_idempotent_path_keeps_index_unchanged() {
    let mgr = TxnManager::new();
    let t = tenant(11);

    // First replay — Applied.
    let o1 = mgr.apply_replay_mvcc_write(Lsn::new(7), t, KEY_A, Some(Bytes::from_static(b"r1")));
    assert_eq!(o1, ReplayApplyOutcome::Applied);
    assert_eq!(mgr.tenant_chain_key_count(t), 1);

    // Second replay at the same commit_lsn — Idempotent (Lemma I1).
    let o2 = mgr.apply_replay_mvcc_write(Lsn::new(7), t, KEY_A, Some(Bytes::from_static(b"r1")));
    assert_eq!(o2, ReplayApplyOutcome::Idempotent);
    // DashSet semantics guarantee no duplicate. Index unchanged.
    assert_eq!(mgr.tenant_chain_key_count(t), 1);
}

// ─────────────────────────────────────────────────────────────────
// Test 10 — concurrent commits across tenants populate the index
// without contention or deadlock.
// ─────────────────────────────────────────────────────────────────

#[test]
fn concurrent_commits_across_tenants_populate_index_correctly() {
    use std::sync::Arc;
    use std::thread;

    let mgr = Arc::new(TxnManager::new());
    let n_tenants = 8;
    let writes_per_tenant = 200u64;

    let handles: Vec<_> = (0..n_tenants)
        .map(|raw| {
            let mgr = Arc::clone(&mgr);
            thread::spawn(move || {
                let t = tenant(raw);
                for k in 0..writes_per_tenant {
                    let mut tx = mgr.begin(t);
                    tx.write(k, Bytes::from(format!("t{raw}-k{k}")));
                    tx.commit().unwrap();
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }

    // Every tenant has exactly `writes_per_tenant` keys indexed.
    for raw in 0..n_tenants {
        assert_eq!(
            mgr.tenant_chain_key_count(tenant(raw)),
            writes_per_tenant as usize,
            "tenant {raw} index miscount under concurrency"
        );
    }
    // Visibility check on a sampled tenant.
    let snap = mgr.current_lsn();
    let mut count = 0u64;
    mgr.for_each_visible_record(tenant(3), snap, |_, _| count += 1);
    assert_eq!(count, writes_per_tenant);
}

// ─────────────────────────────────────────────────────────────────
// Test 11 — write-then-overwrite under the same tenant: the index
// stays at one entry (key was already registered from the first
// commit; the second commit's `register_chain_key` is idempotent).
// ─────────────────────────────────────────────────────────────────

#[test]
fn overwrite_keeps_index_count_at_one() {
    let mgr = TxnManager::new();
    let t = tenant(13);

    let mut tx = mgr.begin(t);
    tx.write(KEY_A, Bytes::from_static(b"v1"));
    tx.commit().unwrap();
    assert_eq!(mgr.tenant_chain_key_count(t), 1);

    let mut tx = mgr.begin(t);
    tx.write(KEY_A, Bytes::from_static(b"v2"));
    tx.commit().unwrap();
    // Same (tenant, key); DashSet::insert is idempotent.
    assert_eq!(mgr.tenant_chain_key_count(t), 1);

    // Visibility — only the latest committed version is callback-fed.
    let mut seen: Vec<Vec<u8>> = Vec::new();
    mgr.for_each_visible_record(t, mgr.current_lsn(), |_, v| seen.push(v.to_vec()));
    assert_eq!(seen, vec![b"v2".to_vec()]);
}
