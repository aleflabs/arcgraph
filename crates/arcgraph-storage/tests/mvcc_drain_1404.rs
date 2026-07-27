//! #1404 M0.x — MVCC version-row drain leg: the frontier-advance-triggered
//! `gc()` driver keeps the resident superseded-version set bounded during
//! sustained ingest (Leg 1), with INV-DRAIN preserved.
//!
//! The reclaimer `gc()` was DRIVEN only at the ADR-229 checkpoint trigger
//! (`bootstrap.rs:801`, rare — ~1 GiB-WAL / ~300 s). Between checkpoints,
//! superseded MVCC versions (update/delete churn + the REL-side adjacency
//! updates the #1404 acceptance OOM'd on) accumulated resident with NOTHING
//! driving their reclamation, contributing to the freeze-capture working set.
//! `TxnManager::with_gc_drive_interval(N)` drives `gc()` every N commits so the
//! resident version-chain set stays bounded. INV-DRAIN is UNCHANGED: `gc()`
//! still reclaims a version iff `expired_lsn ≤ oldest_active_snapshot`.
//!
//! Gates:
//! - **GATE 1 (MVCC leg headline):** a churn workload (repeated overwrites of a
//!   FIXED key set) with the drain driver ON keeps the resident version bytes
//!   BOUNDED independent of the number of commits. RED-on-revert: interval = 0
//!   (driver disabled — the legacy behavior) → the resident version chains grow
//!   with the commit count (unbounded between checkpoints).
//! - **GATE 4 (INV-DRAIN):** a version visible to a HELD-open snapshot is NOT
//!   drained — the held reader still reads it after many drain passes.
//! - **GATE 6 (rel-side symmetry):** rel-side keys (the REL_TAG_BIT-tagged
//!   MvccKey space the adjacency writes use) are drained symmetrically with
//!   node-side keys.

use arcgraph_core::TenantId;
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;

/// Overwrite each of `keys` `rounds` times (each overwrite supersedes the
/// prior version → a reclaimable chain entry once no snapshot pins it).
fn churn(m: &TxnManager, keys: &[u64], rounds: u64) {
    for r in 0..rounds {
        for &k in keys {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(k, Bytes::copy_from_slice(&(r as u32).to_le_bytes()));
            t.commit().unwrap();
        }
    }
}

/// GATE 1 (MVCC leg) — the resident superseded-version set is BOUNDED with the
/// drain driver ON, and GROWS without it (RED-on-revert).
#[test]
fn gate1_mvcc_drain_bounds_resident_versions_and_grows_without() {
    // A FIXED small key set churned many times. With the driver ON, `gc()`
    // fires periodically and reclaims the superseded versions, so the total
    // resident version count (chains × depth) stays bounded near the live set
    // (one live version per key + at most one drive-interval of churn).
    let keys: Vec<u64> = (0..8).collect();
    let rounds = 2000u64; // 8 keys × 2000 = 16_000 commits

    // ── Driver ON: interval small enough to fire many times ──
    let bounded = TxnManager::new().with_gc_drive_interval(64);
    churn(&bounded, &keys, rounds);
    // Drive a final pass to reclaim the tail (no reader pins anything).
    let final_stats = bounded.gc();
    let _ = final_stats;
    let bounded_resident: usize = keys
        .iter()
        .map(|&k| bounded.chain_len(TenantId::DEFAULT, k))
        .sum();

    // The driver fired many passes and reclaimed many versions.
    assert!(
        bounded.driven_gc_passes() > 0,
        "the drain driver never fired — gate is not exercising the driver",
    );
    assert!(
        bounded.driven_gc_reclaimed() > 0,
        "the drain driver reclaimed nothing — no superseded versions were drained",
    );
    // Bounded: with no active reader, gc reclaims every superseded version, so
    // the resident chains collapse to ~one live version per key.
    assert!(
        bounded_resident <= keys.len() * 2,
        "resident version count {bounded_resident} is not bounded near the live set \
         ({} keys) — the drain did not bound the working set",
        keys.len(),
    );

    // ── RED-on-revert: driver OFF (interval = 0, legacy) → no drain drive → the
    //    superseded versions accumulate resident with nothing reclaiming them. ──
    let unbounded = TxnManager::new().with_gc_drive_interval(0);
    churn(&unbounded, &keys, rounds);
    // NO gc() is driven (mirrors the pre-M0.x behavior between checkpoints).
    let unbounded_resident: usize = keys
        .iter()
        .map(|&k| unbounded.chain_len(TenantId::DEFAULT, k))
        .sum();
    assert_eq!(
        unbounded.driven_gc_passes(),
        0,
        "driver must be disabled at interval 0",
    );
    // Each key's chain holds ALL `rounds` versions (nothing reclaimed).
    assert_eq!(
        unbounded_resident,
        keys.len() * rounds as usize,
        "without the drain, every superseded version stays resident",
    );

    // The load-bearing comparison: the drain shrinks the resident set by orders
    // of magnitude, and its size is a function of the drive cadence, NOT of the
    // commit count.
    assert!(
        bounded_resident * 100 < unbounded_resident,
        "drain did not meaningfully bound the resident set: bounded={bounded_resident} \
         vs unbounded={unbounded_resident}",
    );
}

/// GATE 4 (INV-DRAIN) — a version visible to a HELD snapshot is never drained.
/// The drain driver runs many passes while a reader holds an open snapshot; the
/// reader must still read exactly what was committed at its snapshot.
#[test]
fn gate4_inv_drain_held_snapshot_version_not_reclaimed() {
    let m = TxnManager::new().with_gc_drive_interval(4); // fire aggressively

    let key = 7u64;
    // Commit v=10 at the reader's snapshot.
    {
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(key, Bytes::copy_from_slice(&[10u8]));
        t.commit().unwrap();
    }
    // The reader pins the snapshot where key == 10. While it is OPEN, its
    // version MUST survive every drain pass (INV-DRAIN anchor).
    let reader = m.begin(TenantId::DEFAULT);
    assert_eq!(reader.read(key).map(|b| b[0]), Some(10));

    // Heavy churn on the SAME key: each overwrite supersedes the prior version,
    // and the drain driver fires many passes trying to reclaim.
    for v in 11u8..=200 {
        let mut t = m.begin(TenantId::DEFAULT);
        t.write(key, Bytes::copy_from_slice(&[v]));
        t.commit().unwrap();
    }
    // Force extra passes for good measure.
    for _ in 0..10 {
        let _ = m.gc();
    }
    assert!(
        m.driven_gc_passes() > 0,
        "the drain driver never fired under the held reader",
    );

    // The load-bearing INV-DRAIN assertion: the reader's held snapshot STILL
    // sees v == 10. If the drain had reclaimed the snapshot-visible version,
    // this would return None or a newer value — a SILENT WRONG-READ.
    assert_eq!(
        reader.read(key).map(|b| b[0]),
        Some(10),
        "INV-DRAIN VIOLATED: the drain reclaimed a version visible to a held snapshot \
         (silent wrong-read)",
    );
}

/// GATE 6 (MVCC rel-side symmetry) — rel-side keys drain like node-side keys.
/// The MvccKey space is shared (rels are tagged with REL_TAG_BIT by the crud
/// layer); the drain is key-agnostic, so a rel-tagged key set churns + drains
/// identically. This proves the rel-side MVCC path is drained too (the OOM hit
/// RELS).
#[test]
fn gate6_rel_side_keys_drain_symmetrically() {
    // Use high-bit-set keys to model the rel-tagged MvccKey space (the drain is
    // agnostic to the tag; this just exercises a distinct key range).
    const REL_TAG_BIT: u64 = 1 << 63;
    let rel_keys: Vec<u64> = (0..8).map(|k| REL_TAG_BIT | k).collect();
    let rounds = 1000u64;

    let m = TxnManager::new().with_gc_drive_interval(64);
    churn(&m, &rel_keys, rounds);
    let _ = m.gc();

    let resident: usize = rel_keys
        .iter()
        .map(|&k| m.chain_len(TenantId::DEFAULT, k))
        .sum();
    assert!(
        m.driven_gc_reclaimed() > 0,
        "rel-side drain reclaimed nothing — rel-side MVCC not drained",
    );
    assert!(
        resident <= rel_keys.len() * 2,
        "rel-side resident version count {resident} not bounded — rel-side leak",
    );
}

/// The drain driver is a NO-OP for correctness: the committed/visible state is
/// identical with the driver ON vs OFF (the drain is a memory-governor, not a
/// behavior change).
#[test]
fn drain_driver_does_not_change_visible_state() {
    let keys: Vec<u64> = (0..16).collect();
    let rounds = 50u64;

    let with_drive = TxnManager::new().with_gc_drive_interval(8);
    churn(&with_drive, &keys, rounds);
    let _ = with_drive.gc();

    let without = TxnManager::new().with_gc_drive_interval(0);
    churn(&without, &keys, rounds);
    let _ = without.gc();

    // The final committed value per key is `rounds - 1` in both; visible state
    // is identical regardless of drain cadence.
    let lsn_a = with_drive.current_lsn();
    let lsn_b = without.current_lsn();
    for &k in &keys {
        let a = with_drive
            .read_at(TenantId::DEFAULT, k, lsn_a)
            .map(|b| u32::from_le_bytes(b[..4].try_into().unwrap()));
        let b = without
            .read_at(TenantId::DEFAULT, k, lsn_b)
            .map(|b| u32::from_le_bytes(b[..4].try_into().unwrap()));
        assert_eq!(
            a,
            Some(rounds as u32 - 1),
            "with-drive lost the latest value"
        );
        assert_eq!(
            b,
            Some(rounds as u32 - 1),
            "without-drive lost the latest value"
        );
        assert_eq!(a, b, "drain cadence changed the visible value at key {k}");
    }
}
