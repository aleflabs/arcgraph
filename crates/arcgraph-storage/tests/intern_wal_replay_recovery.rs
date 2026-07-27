//! P0 #776 — storage-level pin that WAL replay reconstructs the
//! [`InternTable`] label / rel-type name↔id mapping from
//! [`WalRecordType::InternString`] records.
//!
//! This isolates the replay arm fixed by #776: the production write
//! path logs each new intern via `intern_label_logged` /
//! `intern_type_logged`, and `recover_from_wal` — when wired with
//! [`PageStoreTarget::with_intern_table`] — decodes + installs them into
//! the served table via [`InternTable::intern_install`]. The end-to-end
//! `graph.schema` + typed-query oracle lives in
//! `crates/arcgraph-cli/tests/durable_intern_restart_776.rs`; this file
//! pins the storage-crate seam in isolation (no MCP / query stack).
//!
//! RED→GREEN: before the fix the `InternString` replay arm was a no-op,
//! so `recovered.try_resolve(..).unwrap()` returned `None` and `interns_recovered`
//! was 0; after the fix the names resolve by their ORIGINAL ids.

use std::sync::{Arc, Barrier};

use arcgraph_core::{StringId, TenantId};
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::intern::{
    InternTable, intern_label_logged, intern_logged, intern_type_logged,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    PageStoreTarget, PrimaryPageStoreHandle, WalConfig, WalRecordType, WalRecoveryReader,
    WalWriter, recover_from_wal,
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

/// A fresh recovery-side `PageStoreTarget` over a throwaway primary
/// store. `intern_table` is wired iff `intern` is `Some`.
fn recover_into(
    wal_dir: &std::path::Path,
    intern: Option<Arc<InternTable>>,
) -> arcgraph_storage::wal::RecoveryReport {
    let mgr = Arc::new(TxnManager::new());
    let alloc = Arc::new(PageAllocator::new());
    let primary =
        Arc::new(PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), None).expect("primary"));
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let mut target = PageStoreTarget::primary_only(primary_handle);
    if let Some(table) = intern {
        target = target.with_intern_table(table);
    }
    recover_from_wal(wal_dir, mgr, target, None).expect("recover_from_wal")
}

#[test]
fn intern_names_recovered_from_wal_replay_776() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    // ── Session 1: WAL-log a label + a rel-type via the durable
    //    write helpers (Some(handle)). Re-interning the same name must
    //    NOT re-log (idempotent).
    let (account_raw, sent_raw) = {
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        let handle = writer.handle();
        let table = InternTable::new();
        let account = intern_label_logged(&table, Some(&handle), tenant, "Account").unwrap();
        let sent = intern_type_logged(&table, Some(&handle), tenant, "SENT").unwrap();
        let account_again = intern_label_logged(&table, Some(&handle), tenant, "Account").unwrap();
        assert_eq!(account, account_again, "re-intern returns the same id");
        writer.shutdown().unwrap();
        (account.raw(), sent.raw())
    };

    // ── Session 2: recover into a FRESH empty table.
    let recovered = Arc::new(InternTable::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&recovered)));

    // Reverse direction (graph.schema name rendering): the original ids
    // resolve to the original names.
    assert_eq!(
        &*recovered
            .try_resolve(tenant, StringId::new(account_raw))
            .unwrap()
            .expect("Account id resolves post-replay"),
        "Account",
    );
    assert_eq!(
        &*recovered
            .try_resolve(tenant, StringId::new(sent_raw))
            .unwrap()
            .expect("SENT id resolves post-replay"),
        "SENT",
    );
    // Forward direction (binder lookup): the name re-interns to the
    // SAME id, never a fresh one (allocator was bumped past it).
    assert_eq!(
        recovered.intern(tenant, "Account").unwrap().raw(),
        account_raw,
        "forward lookup returns the recovered id, not a fresh allocation",
    );
    // Two DISTINCT interns recovered (the idempotent re-intern did not
    // emit a third record).
    assert_eq!(
        report.metrics.interns_recovered, 2,
        "exactly two InternString records replayed",
    );
}

#[test]
fn intern_replay_without_table_is_noop_776() {
    // Pre-fix-behaviour preservation: a replay-shape caller that does
    // NOT wire an intern table still recovers cleanly; the InternString
    // arm is a no-op and the counter stays 0.
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    {
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        let handle = writer.handle();
        let table = InternTable::new();
        let _ = intern_label_logged(&table, Some(&handle), tenant, "Account").unwrap();
        writer.shutdown().unwrap();
    }

    let report = recover_into(&wal_dir, None);
    assert_eq!(
        report.metrics.interns_recovered, 0,
        "no intern table wired ⇒ InternString arm is a no-op",
    );
}

// ─── v2 M2 A4 — the durable-logged-set protocol (storage grain).
//     The full crash-shaped repro (racing loser commits a TYPED BLOCK
//     referencing the id) is the mcp gate
//     `m2_intern_durability_gate.rs`; these pin the intern seam. ─────

/// A4 loser path: a binding published by an UNLOGGED path (the racing
/// loser's view — or #355's phantom-intern) carries no durable proof,
/// so the FIRST logged-path reference must append it. RED-on-revert:
/// gating on `was_new` skips the append (`was_new == false`) and the
/// fresh-table recovery below loses the binding.
#[test]
fn a4_unproven_binding_is_relogged_by_next_logged_caller() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let raw = {
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        let handle = writer.handle();
        let table = InternTable::new();
        // The unlogged publish — exactly what a racing loser observes
        // as `was_new == false` before any InternString record exists.
        let published = table.intern(tenant, "RaceKey").unwrap();
        // The logged-path reference: must re-log despite the publish.
        let logged = intern_label_logged(&table, Some(&handle), tenant, "RaceKey").unwrap();
        assert_eq!(logged.raw(), published.raw(), "same id, no re-allocation");
        writer.shutdown().unwrap();
        published.raw()
    };

    let recovered = Arc::new(InternTable::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&recovered)));
    assert_eq!(
        report.metrics.interns_recovered, 1,
        "the logged-path reference appended the unproven binding",
    );
    assert_eq!(
        &*recovered
            .try_resolve(tenant, StringId::new(raw))
            .unwrap()
            .expect("unproven-then-logged binding survives recovery (A4)"),
        "RaceKey",
    );
}

/// A4 failure path: an append FAILURE must not leave an apparently-
/// established binding that later writers reference without durable
/// proof — the failing caller gets `Err` (aborts its commit) and the
/// next logged caller re-attempts the append.
#[test]
fn a4_append_failure_does_not_poison_durable_proof() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let table = InternTable::new();
    // A dead WAL: every append fails.
    let dead_handle = {
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        let handle = writer.handle();
        writer.shutdown().unwrap();
        handle
    };
    let err = intern_label_logged(&table, Some(&dead_handle), tenant, "Fragile");
    assert!(err.is_err(), "append against a dead WAL must propagate Err");

    // The binding is published in-memory (idempotent id) but UNPROVEN:
    // a live-WAL retry through the SAME table must append it.
    let raw = {
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        let handle = writer.handle();
        let id = intern_label_logged(&table, Some(&handle), tenant, "Fragile").unwrap();
        writer.shutdown().unwrap();
        id.raw()
    };

    let recovered = Arc::new(InternTable::new());
    recover_into(&wal_dir, Some(Arc::clone(&recovered)));
    assert_eq!(
        &*recovered
            .try_resolve(tenant, StringId::new(raw))
            .unwrap()
            .expect("retried binding is durable (A4 failure path)"),
        "Fragile",
    );
}

/// A4 race regime: 8 threads race `intern_logged` over the SAME name
/// pool through one WAL. EVERY name must be durable — a fresh-table
/// recovery resolves each live id. Duplicate InternString appends are
/// permitted (bounded waste; replay is idempotent) — what is NOT
/// permitted is a name with fewer than one record.
#[test]
fn a4_concurrent_intern_logged_every_name_durable() {
    const THREADS: usize = 8;
    const POOL: usize = 64;
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let live = {
        let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
        let handle = writer.handle();
        let table = Arc::new(InternTable::new());
        let names: Arc<Vec<String>> =
            Arc::new((0..POOL).map(|i| format!("race_label_{i:03}")).collect());
        let mut hs = Vec::new();
        for _ in 0..THREADS {
            let table = Arc::clone(&table);
            let handle = handle.clone();
            let names = Arc::clone(&names);
            hs.push(std::thread::spawn(move || {
                for n in names.iter() {
                    intern_label_logged(&table, Some(&handle), tenant, n).unwrap();
                }
            }));
        }
        for h in hs {
            h.join().expect("racer panicked");
        }
        writer.shutdown().unwrap();
        table
    };

    let recovered = Arc::new(InternTable::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&recovered)));
    assert!(
        report.metrics.interns_recovered >= POOL as u64,
        "at least one InternString record per name (got {})",
        report.metrics.interns_recovered,
    );
    for i in 0..POOL {
        let name = format!("race_label_{i:03}");
        let live_id = live.intern(tenant, &name).unwrap();
        assert_eq!(
            recovered
                .try_resolve(tenant, live_id)
                .unwrap()
                .as_deref()
                .map(String::as_str),
            Some(name.as_str()),
            "every raced name durable under its live id (A4)",
        );
    }
}

#[test]
fn intern_logged_none_wal_is_in_memory_only_776() {
    // The ephemeral (`--in-memory`) path passes `None` for the WAL and
    // must behave exactly like the pre-fix pure in-memory intern: the id
    // is allocated, nothing is logged (no WAL to log to), and a later
    // recovery of an empty dir finds nothing.
    let table = InternTable::new();
    let tenant = TenantId::DEFAULT;
    let id = intern_label_logged(&table, None, tenant, "Account").unwrap();
    assert_eq!(
        &*table
            .try_resolve(tenant, StringId::new(id.raw()))
            .unwrap()
            .unwrap(),
        "Account",
        "no-WAL intern still populates the live table",
    );
    // Sanity: the helper agrees with the bare in-memory CrudStore-free
    // intern path (no panic, stable id).
    assert_eq!(table.intern(tenant, "Account").unwrap().raw(), id.raw());
    // Touch CrudStore to keep the import meaningful (a no-WAL store has
    // `wal() == None`, the gate the MCP write path consults).
    assert!(CrudStore::new().wal().is_none());
}

/// **A4 round-2 (#1452, codex re-check) — the LSN-ordering leg of the
/// durable-proof contract, under a 16-way race seeded by a FAILED
/// append.** `intern_logged`'s `Ok` promises the binding's
/// `InternString` record is durable BEFORE the caller's referencing
/// commit can append (strictly lower LSN). The forced-failure prologue
/// pins the retry arm: a binding whose append FAILED stays published
/// in-memory but UNPROVEN, so the first live-WAL caller must re-append
/// it rather than trust the publish.
///
/// RED-on-revert: gate the append on `was_new` (the pre-A4 latch) and
/// the dead-writer publish suppresses every live append — no
/// `InternString` record exists and the `!intern_lsns.is_empty()`
/// assert fails; alternatively insert durable-proof BEFORE the append
/// returns and a racing marker can land below the record, failing the
/// ordering assert.
#[test]
fn a4_logged_proof_never_precedes_durable_append_and_failure_retries() {
    const THREADS: usize = 16;
    let tenant = TenantId::DEFAULT;
    let table = Arc::new(InternTable::new());

    // Force the append-failure path first: a WAL whose writer is
    // already shut down fails every append. The binding is now
    // published in memory, but a subsequent live-WAL caller must still
    // emit InternString (no durable proof was recorded).
    let dead_tmp = TempDir::new().unwrap();
    let dead = {
        let writer = WalWriter::spawn(test_wal_config(dead_tmp.path())).unwrap();
        let handle = writer.handle();
        writer.shutdown().unwrap();
        handle
    };
    assert!(
        intern_logged(&table, &dead, tenant, "race_key").is_err(),
        "append on a dead WAL must surface Err (and must NOT mark proof)",
    );

    let live_tmp = TempDir::new().unwrap();
    let writer = WalWriter::spawn(test_wal_config(live_tmp.path())).unwrap();
    let wal = writer.handle();
    let start = Arc::new(Barrier::new(THREADS));
    let mut threads = Vec::new();
    for thread_no in 0..THREADS {
        let table = Arc::clone(&table);
        let wal = wal.clone();
        let start = Arc::clone(&start);
        threads.push(std::thread::spawn(move || {
            start.wait();
            let id = intern_logged(&table, &wal, tenant, "race_key").unwrap();
            // Caller-side stand-in for the referencing commit. If proof
            // could become visible before InternString was durable, a
            // racing marker could receive an earlier LSN.
            let marker_lsn = wal
                .append(
                    WalRecordType::Checkpoint,
                    thread_no as u64 + 1,
                    0,
                    tenant,
                    id.raw().to_le_bytes().to_vec(),
                )
                .unwrap();
            (id, marker_lsn)
        }));
    }
    let markers: Vec<_> = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .collect();
    writer.shutdown().unwrap();

    let records = WalRecoveryReader::open(live_tmp.path())
        .unwrap()
        .collect_all()
        .unwrap();
    let intern_lsns: Vec<_> = records
        .iter()
        .filter(|record| record.record_type == WalRecordType::InternString)
        .map(|record| record.lsn)
        .collect();
    assert!(
        !intern_lsns.is_empty(),
        "the retry after the forced failure must emit InternString",
    );
    let first_intern = *intern_lsns.iter().min().unwrap();
    let expected_id = markers[0].0;
    for (id, marker_lsn) in markers {
        assert_eq!(id, expected_id, "one binding, one id, all racers");
        assert!(
            marker_lsn > first_intern,
            "referencing marker {marker_lsn:?} preceded durable intern {first_intern:?}",
        );
    }
}

/// **A4 round-2 (#1452, codex re-check) — recovery SEEDS the durable-
/// proof set.** A binding installed by WAL replay carries durable proof
/// by provenance, so the first post-recovery logged intern of the same
/// name must NOT re-append it (exactly one `InternString` record ever
/// exists for the binding).
///
/// RED-on-revert: drop the `logged.insert` from
/// [`InternTable::intern_install`] and the post-recovery re-intern
/// appends a duplicate — the final count reads 2, not 1.
#[test]
fn a4_recovery_seeds_durable_proof_and_suppresses_reappend() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let tenant = TenantId::DEFAULT;

    let live = InternTable::new();
    let writer = WalWriter::spawn(test_wal_config(&wal_dir)).unwrap();
    let id = intern_logged(&live, &writer.handle(), tenant, "recovered_key").unwrap();
    writer.shutdown().unwrap();

    let recovered = Arc::new(InternTable::new());
    let report = recover_into(&wal_dir, Some(Arc::clone(&recovered)));
    assert_eq!(
        recovered
            .try_resolve(tenant, id)
            .unwrap()
            .as_deref()
            .map(String::as_str),
        Some("recovered_key"),
    );

    // Post-recovery re-intern through the logged path: same id, and —
    // because replay seeded the durable-proof set — NO re-append.
    let writer = WalWriter::spawn_from(test_wal_config(&wal_dir), report.last_wal_lsn).unwrap();
    assert_eq!(
        intern_logged(&recovered, &writer.handle(), tenant, "recovered_key").unwrap(),
        id,
    );
    writer.shutdown().unwrap();

    let intern_count = WalRecoveryReader::open(&wal_dir)
        .unwrap()
        .collect_all()
        .unwrap()
        .into_iter()
        .filter(|record| record.record_type == WalRecordType::InternString)
        .count();
    assert_eq!(
        intern_count, 1,
        "recovery must seed the durable-proof set (a second record means \
         the replay-installed binding was treated as unproven)",
    );
}
