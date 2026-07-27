//! W28 / ADR-183 — GA durable-by-default server bootstrap: restart-survival,
//! fault-injection (fsync-watermark boundary), ephemeral negative pin, and a
//! multi-tenant registry forward-pin.
//!
//! This file IS the ADR-133 active verification for the durable bootstrap
//! slice (PR `feat(cli+storage): GA durable-by-default server bootstrap`).
//! It drives the production `arcgraph_cli::bootstrap::bootstrap_storage_backend`
//! surface end-to-end against a real `--data <dir>`:
//!
//! - [`restart_survival_node_and_rel_round_trip`] (ADR-183 R1 + R3) —
//!   commit real NODE + RELATIONSHIP records under `TenantId::DEFAULT`
//!   through the bootstrapped backend, "restart" (drop the bundle → WAL
//!   drain + fsync + join), re-bootstrap the SAME dir, and read the
//!   records back BYTE-IDENTICAL. NOT just CatalogStats (ADR-183 R1
//!   explicitly rejects the `primary_only` stats-rebuild template as
//!   sufficient evidence). Also empirically verifies the DEFAULT tenant is
//!   present in the recovered registry (ADR-183 R3 — bootstrap-on-existing-
//!   dir re-registers DEFAULT without conflict).
//! - [`fault_injection_fsync_watermark_is_the_crash_boundary`]
//!   (ADR-183 R2 + ADR-034 §Slice B / §I-D1) — every acked Strict commit is
//!   at/below the WAL committed-fsync watermark BEFORE `commit()` returns
//!   (the §I-D1 durable-before-ack invariant); a commit attempted
//!   after the WAL becomes unavailable FAILS (rolled back, never durable)
//!   and does NOT survive restart. The crash-consistency boundary IS the
//!   fsync watermark (ADR-034 §Slice B): survivors == {commits ≤ watermark}.
//! - [`in_memory_is_ephemeral_negative_pin`] — `--in-memory` mode: a
//!   committed node does NOT survive a fresh in-memory bootstrap (documents
//!   the opt-in non-durable behavior).
//! - [`multi_tenant_registry_recovery_is_forward_pinned`] (ADR-183
//!   §Forward-pin) — a NON-`DEFAULT` tenant with committed durable data is
//!   NOT recovered into the catalog registry after restart (not in
//!   `router().tenants()`; `route()` rejects it). Default-tenant durability
//!   is the GA scope; multi-tenant registry recovery needs the M10
//!   catalog-recover-from-pages path.
//!
//! These exercise ONLY the public bootstrap API + `StorageBackend`
//! accessors + the `arcgraph_storage::crud` free functions — the same
//! surfaces a production operator / the MCP ingest path reach.

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId, TypeId};
use arcgraph_mcp::storage::CrudExecutorSubstrate;
use arcgraph_query::executor::substrate::ExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, read_node_with_store,
    read_rel_with_store,
};
use arcgraph_storage::wal::{
    SegmentHeader, WalRecord, WalRecordType, list_segments, segment_filename,
};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::sync::Arc;
use tempfile::TempDir;

/// The shared per-tenant `CrudStore` for `tenant` (v1.0 routes every tenant
/// to one store) via the production router surface.
fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

/// Commit one node under `tenant` via the bootstrapped backend; return its
/// id + the commit LSN. Mirrors the production CRUD write path.
fn commit_node(
    backend: &arcgraph_mcp::storage::StorageBackend,
    crud: &Arc<CrudStore>,
    tenant: TenantId,
    label: u32,
    a: u32,
    b: u32,
) -> (NodeId, arcgraph_core::Lsn) {
    let mut tx = backend.txn_manager().begin(tenant);
    let id = create_node(
        crud,
        &mut tx,
        tenant,
        LabelId::new(label),
        &PropertyData::InlineU32Pair(a, b),
    )
    .expect("create_node");
    let lsn = commit(tx, crud).expect("commit node");
    (id, lsn)
}

// ─────────────────────────────────────────────────────────────────────
// ADR-183 R1 + R3 — restart-survival round-trip (the active verification).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn restart_survival_node_and_rel_round_trip() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: bootstrap durable, commit a NODE + a RELATIONSHIP.
    let (src_id, dst_id, rel_id) = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");

        let crud = crud_for(&backend, TenantId::DEFAULT);

        // NODE (label=7, inline bytes 111/222) + a dst NODE (label=8).
        let (src_id, n_lsn) = commit_node(&backend, &crud, TenantId::DEFAULT, 7, 111, 222);
        // R2 sanity: the acked Strict commit is at/below the fsync watermark
        // BEFORE commit() returned (no acknowledged-commit loss).
        assert!(
            guard.last_durable_lsn().expect("durable watermark") >= n_lsn,
            "ADR-183 R2: acked Strict commit {n_lsn:?} must be ≤ fsync watermark {:?}",
            guard.last_durable_lsn(),
        );
        let (dst_id, _) = commit_node(&backend, &crud, TenantId::DEFAULT, 8, 333, 444);

        // RELATIONSHIP src -> dst (type=5).
        let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
        let rel_id = create_rel(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            src_id,
            dst_id,
            TypeId::new(5),
            &PropertyData::Empty,
        )
        .expect("create_rel");
        let r_lsn = commit(tx, &crud).expect("commit rel");
        assert!(
            guard.last_durable_lsn().expect("durable watermark") >= r_lsn,
            "ADR-183 R2: acked Strict rel commit {r_lsn:?} must be ≤ fsync watermark",
        );

        // Drop the bundle → DurabilityGuard's WalWriter drains + fsyncs +
        // joins (graceful "process restart").
        (src_id, dst_id, rel_id)
    };

    // ── Session 2: re-bootstrap the SAME dir → WAL recovery on startup.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    // ADR-183 R3 — empirical: the DEFAULT tenant is present post-recover
    // (re-bootstrapped, a redundant-but-harmless SYSTEM MVCC version; no
    // page conflict / corruption).
    assert!(
        backend2.router().tenants().contains(&TenantId::DEFAULT),
        "ADR-183 R3: DEFAULT tenant must be present after restart over an existing dir",
    );

    let crud2 = crud_for(&backend2, TenantId::DEFAULT);
    let tx = backend2.txn_manager().begin(TenantId::DEFAULT);

    // ADR-183 R1 — the NODE survives BYTE-IDENTICAL (not just CatalogStats).
    let node = read_node_with_store(&crud2, &tx, src_id)
        .expect("read node")
        .expect("committed node MUST survive restart (ADR-183 R1)");
    assert_eq!(node.label_id, 7, "node label survives restart");
    assert_eq!(node.inline_u32a, 111, "node inline_u32a survives restart");
    assert_eq!(node.inline_u32b, 222, "node inline_u32b survives restart");

    // ADR-183 R1 — the RELATIONSHIP survives byte-identical.
    let rel = read_rel_with_store(&crud2, &tx, rel_id)
        .expect("read rel")
        .expect("committed rel MUST survive restart (ADR-183 R1)");
    assert_eq!(rel.src_id, src_id.raw(), "rel src survives restart");
    assert_eq!(rel.dst_id, dst_id.raw(), "rel dst survives restart");
    assert_eq!(rel.type_id, 5, "rel type survives restart");
}

// ─────────────────────────────────────────────────────────────────────
// P0 #820 — acked data must survive the SECOND (and Nth) durable restart.
// CARDINAL durability: a graceful close+reopen, repeated, must never lose
// or duplicate an acknowledged (fsync'd) commit. Strong oracle = every
// committed (NodeId, seq) reads back its EXACT seq after each restart
// (count + distinct-seq + missing + wrong are reported on failure so the
// "count grows / distinct stuck at last-batch" signature is visible).
// ─────────────────────────────────────────────────────────────────────

/// Read every accumulated `(NodeId, expected_seq)` through the recovered
/// backend and assert all are present with their exact seq. The panic
/// message surfaces count(present) / distinct_seq / missing / wrong so the
/// #820 signature (count 20, DISTINCT 10 = last batch, earlier seqs
/// missing, dups) is legible in the RED capture.
fn assert_all_acked_present(
    backend: &arcgraph_mcp::storage::StorageBackend,
    crud: &Arc<CrudStore>,
    expected: &[(NodeId, u32)],
    when: &str,
) {
    let tx = backend.txn_manager().begin(TenantId::DEFAULT);
    let mut present_seqs: Vec<u32> = Vec::new();
    let mut missing: Vec<(NodeId, u32)> = Vec::new();
    let mut wrong: Vec<(NodeId, u32, u32)> = Vec::new(); // (id, expected, got)
    for (id, seq) in expected {
        match read_node_with_store(crud, &tx, *id).expect("read node") {
            Some(rec) => {
                present_seqs.push(rec.inline_u32a);
                if rec.inline_u32a != *seq {
                    wrong.push((*id, *seq, rec.inline_u32a));
                }
            }
            None => missing.push((*id, *seq)),
        }
    }
    let count = present_seqs.len();
    let mut distinct = present_seqs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        missing.is_empty() && wrong.is_empty(),
        "{when}: every acked commit MUST survive (no loss, no dups). \
         expected={} present(count)={} distinct_seq={} missing={:?} wrong(id,exp,got)={:?}",
        expected.len(),
        count,
        distinct.len(),
        missing,
        wrong,
    );
}

/// Commit nodes `seq` in `[lo, hi)` under DEFAULT (label `LABEL_820`) and
/// push each `(NodeId, seq)` onto `expected`.
///
/// `commit()` returning `Ok` on the `Strict`-tier `DEFAULT` tenant IS the
/// durability ack — the commit is fsync-durable before the call returns
/// (ADR-034 §I-D1; the companion `fault_injection_*` test proves a commit
/// FAILS when the WAL is unavailable). We deliberately do NOT compare
/// `commit_lsn` to `DurabilityGuard::last_durable_lsn()` here: post-restart
/// those are different clocks — `commit_lsn` is the global
/// `TxnManager` counter (seeded across restarts) while the framing
/// watermark is this process's per-spawn WAL-LSN — and conflating them is
/// the very `last_wal_lsn < applied_commit_lsn` confusion at the heart of
/// #820 (bootstrap.rs §"Post-recovery divergence").
fn write_acked_batch(
    backend: &arcgraph_mcp::storage::StorageBackend,
    crud: &Arc<CrudStore>,
    lo: u32,
    hi: u32,
    expected: &mut Vec<(NodeId, u32)>,
) {
    const LABEL_820: u32 = 1;
    for seq in lo..hi {
        let (id, _lsn) = commit_node(backend, crud, TenantId::DEFAULT, LABEL_820, seq, seq);
        expected.push((id, seq));
    }
}

fn append_unacked_torn_wal_record(data_dir: &std::path::Path) {
    let wal_dir = data_dir.join("wal");
    let segments = list_segments(&wal_dir).expect("list WAL segments");
    let last = segments.last().copied().expect("at least one WAL segment");
    let path = wal_dir.join(segment_filename(last));
    let mut file = OpenOptions::new()
        .append(true)
        .read(true)
        .open(&path)
        .expect("open terminal WAL segment");
    let original_len = file.metadata().expect("terminal segment metadata").len();
    assert!(
        original_len >= SegmentHeader::SIZE as u64,
        "terminal WAL segment must at least contain its header"
    );

    let fake = WalRecord {
        record_type: WalRecordType::Checkpoint,
        txn_id: 0,
        lsn: Lsn::new(1_000_000),
        timestamp_ms: 0,
        tenant_id: TenantId::DEFAULT,
        payload: vec![0x5a; 64],
    }
    .encode_to_vec()
    .expect("encode fake in-flight WAL record");
    file.write_all(&fake)
        .expect("append fake in-flight WAL record");
    file.set_len(original_len + 20)
        .expect("tear fake WAL record with set_len");
}

#[test]
fn torn_tail_recovered_before_writer_attach_preserves_pre_and_post_restart_acked_commits_1109() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let mut expected: Vec<(NodeId, u32)> = Vec::new();

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap before tear");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        write_acked_batch(&backend, &crud, 100, 103, &mut expected);
    }

    append_unacked_torn_wal_record(&data_dir);

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap must recover and truncate torn tail before writer attach");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        assert_all_acked_present(&backend, &crud, &expected, "after torn-tail recovery");
        write_acked_batch(&backend, &crud, 200, 203, &mut expected);
    }

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap after post-tear commits");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        assert_all_acked_present(
            &backend,
            &crud,
            &expected,
            "after post-tear commits and second recovery",
        );
    }
}

#[test]
fn acked_data_survives_repeated_durable_restarts_820() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let mut expected: Vec<(NodeId, u32)> = Vec::new();

    // ── Epoch 0: fresh durable store; write seq 0..9 (acked/fsync'd).
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (epoch 0)");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        write_acked_batch(&backend, &crud, 0, 10, &mut expected);
        // drop(guard) → WAL drain + fsync + join (graceful SIGTERM-equivalent).
    }

    // ── 1st restart: recover, verify {0..9}, then write seq 10..19.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (1st restart)");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        assert_all_acked_present(&backend, &crud, &expected, "after 1st restart");
        write_acked_batch(&backend, &crud, 10, 20, &mut expected);
    }

    // ── 2nd restart (the #820 failing one): recover, verify ALL {0..19}.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (2nd restart)");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        assert_all_acked_present(&backend, &crud, &expected, "after 2nd restart");
        write_acked_batch(&backend, &crud, 20, 30, &mut expected);
    }

    // ── 3rd restart: recover, verify ALL {0..29} — holds across N cycles.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (3rd restart)");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        assert_all_acked_present(&backend, &crud, &expected, "after 3rd restart");
    }
}

/// Build a production `CrudExecutorSubstrate` over a bootstrapped backend —
/// the exact surface the served `MATCH` / `CREATE` query path uses.
fn substrate_for(backend: &arcgraph_mcp::storage::StorageBackend) -> CrudExecutorSubstrate {
    CrudExecutorSubstrate::new(
        Arc::clone(backend.router()),
        Arc::clone(backend.txn_manager()),
        Arc::clone(backend.intern_table()),
    )
}

/// `CREATE (n:K {seq: s})` for `s` in `[lo, hi)` through the served substrate;
/// each named `seq` property persists as a `PropertyData::Blob` JSON chain
/// (the path #820 corrupts). Accumulate `(NodeId, seq)`.
fn create_named_seq_nodes(
    sub: &CrudExecutorSubstrate,
    lo: i64,
    hi: i64,
    expected: &mut Vec<(NodeId, i64)>,
) {
    for seq in lo..hi {
        let id = sub
            .create_node(
                TenantId::DEFAULT,
                Some("K"),
                &[("seq".to_string(), Value::Integer(seq))],
                &arcgraph_query::executor::ExecutionContext::new(
                    TenantId::DEFAULT,
                    PartitionId::ZERO,
                ),
            )
            .expect("CREATE (n:K {seq})");
        expected.push((id, seq));
    }
}

/// `MATCH (n) RETURN n.seq` (all DEFAULT nodes) through the served substrate;
/// assert every accumulated `(NodeId, seq)` reads back its EXACT seq, the row
/// count equals the acked count (no dups/extras), and report
/// count/distinct/missing/wrong so the #820 signature is legible on RED.
fn assert_named_seq_survives(sub: &CrudExecutorSubstrate, expected: &[(NodeId, i64)], when: &str) {
    let rows = sub
        .scan_nodes(TenantId::DEFAULT, None, Lsn::MAX)
        .expect("scan_nodes (MATCH n)");
    let mut got: HashMap<u64, i64> = HashMap::new();
    let mut got_seqs: Vec<i64> = Vec::new();
    for bn in &rows {
        if let Some(s) = bn.node.properties.get("seq").and_then(Value::as_i64) {
            got.insert(bn.node.id.raw(), s);
            got_seqs.push(s);
        }
    }
    let mut missing: Vec<(u64, i64)> = Vec::new();
    let mut wrong: Vec<(u64, i64, i64)> = Vec::new(); // (id, expected, got)
    for (id, seq) in expected {
        match got.get(&id.raw()) {
            Some(g) if g == seq => {}
            Some(g) => wrong.push((id.raw(), *seq, *g)),
            None => missing.push((id.raw(), *seq)),
        }
    }
    let mut distinct = got_seqs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        missing.is_empty() && wrong.is_empty(),
        "{when}: every acked CREATE (n:K {{seq}}) MUST MATCH back its EXACT seq \
         (no loss, no overwrite). expected={} scan_rows={} present(seq)={} distinct_seq={} \
         missing(id,seq)={:?} wrong(id,exp,got)={:?}",
        expected.len(),
        rows.len(),
        got_seqs.len(),
        distinct.len(),
        missing,
        wrong,
    );
    assert_eq!(
        rows.len(),
        expected.len(),
        "{when}: scan row-count must equal #acked nodes (no duplicate/extra rows); got {} want {}",
        rows.len(),
        expected.len(),
    );
}

/// P0 #820 — the CARDINAL repro through the SERVED query path. A named
/// integer property (`{seq: i}`) persists as a `PropertyData::Blob` chain;
/// `MATCH (n) RETURN n.seq` reads it back via `scan_nodes` → `read_node` →
/// `record_property_bag` → `BlobStore::get`. Pre-fix, the blob page-id
/// allocator (`BlobStore::next_page`) resets to 0 on every process start and
/// is NOT advanced by replay's `install_or_replace`, so epoch N+1's property
/// blobs reuse epoch N's blob page-ids; the next restart's replay overwrites
/// earlier nodes' property blobs with later nodes' values — "count grows,
/// DISTINCT stuck at the last batch, earlier seqs gone, last batch dup'd".
#[test]
fn acked_named_property_survives_repeated_durable_restarts_820() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let mut expected: Vec<(NodeId, i64)> = Vec::new();

    // ── Epoch 0: fresh durable; CREATE (n:K {seq}) for 0..9 (acked/fsync'd).
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (epoch 0)");
        let sub = substrate_for(&backend);
        create_named_seq_nodes(&sub, 0, 10, &mut expected);
        assert_named_seq_survives(&sub, &expected, "in-session epoch 0");
        // drop(guard) → WAL drain + fsync + join (graceful SIGTERM-equivalent).
    }

    // ── 1st restart: recover, MATCH {0..9}, then CREATE {seq} for 10..19.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (1st restart)");
        let sub = substrate_for(&backend);
        assert_named_seq_survives(&sub, &expected, "after 1st restart");
        create_named_seq_nodes(&sub, 10, 20, &mut expected);
    }

    // ── 2nd restart (the #820 failing one): recover, MATCH ALL {0..19}.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (2nd restart)");
        let sub = substrate_for(&backend);
        assert_named_seq_survives(&sub, &expected, "after 2nd restart");
        create_named_seq_nodes(&sub, 20, 30, &mut expected);
    }

    // ── 3rd restart: recover, MATCH ALL {0..29} — holds across N cycles.
    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (3rd restart)");
        let sub = substrate_for(&backend);
        assert_named_seq_survives(&sub, &expected, "after 3rd restart");
    }
}

// ─────────────────────────────────────────────────────────────────────
// ADR-183 R2 + ADR-034 §Slice B / §I-D1 — the fsync watermark IS the crash
// boundary (§Slice B adds the watermark; §I-D1 is durable-before-ack).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn fault_injection_fsync_watermark_is_the_crash_boundary() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    const N_ACKED: u32 = 8;
    let mut acked: Vec<(NodeId, u32, u32)> = Vec::with_capacity(N_ACKED as usize);

    // Commit N acked Strict nodes; each MUST be ≤ the fsync watermark before
    // commit() returns. Then inject a WAL-loss fault and attempt one more
    // commit — it must FAIL (never durable, never advances the watermark).
    let doomed_id = {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap");
        let crud = crud_for(&backend, TenantId::DEFAULT);

        for i in 1..=N_ACKED {
            let (id, lsn) =
                commit_node(&backend, &crud, TenantId::DEFAULT, i, i, i.wrapping_mul(7));
            // ADR-034 §I-D1 (witnessed by §Slice B watermark): durable BEFORE ack.
            assert!(
                guard.last_durable_lsn().expect("watermark") >= lsn,
                "acked Strict commit {lsn:?} must be ≤ fsync watermark {:?}",
                guard.last_durable_lsn(),
            );
            acked.push((id, i, i.wrapping_mul(7)));
        }

        // FAULT INJECTION: drop the DurabilityGuard → the WalWriter thread
        // shuts down (models a crash / the WAL becoming unavailable). The
        // CrudStore + TxnManager still hold now-disconnected WAL handles.
        drop(guard);

        // A commit attempted past the fault FAILS — the WAL append errors,
        // the tx rolls back (ADR-033), nothing becomes durable and the
        // watermark never advances for it.
        let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
        let doomed_id = create_node(
            &crud,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(99),
            &PropertyData::InlineU32Pair(999, 999),
        )
        .expect("create_node buffers in-tx (no WAL append yet)");
        let doomed = commit(tx, &crud);
        assert!(
            doomed.is_err(),
            "commit after WAL loss MUST fail (un-acked → not durable); got {doomed:?}",
        );
        doomed_id
    };

    // ── Recover over the same dir.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (recover)");
    let crud2 = crud_for(&backend2, TenantId::DEFAULT);
    let tx = backend2.txn_manager().begin(TenantId::DEFAULT);

    // Every ACKED (≤ watermark) commit survives byte-identical.
    for (id, a, b) in &acked {
        let rec = read_node_with_store(&crud2, &tx, *id)
            .expect("read acked node")
            .unwrap_or_else(|| {
                panic!("acked commit (NodeId={id:?}) MUST survive — it was ≤ watermark")
            });
        assert_eq!(
            rec.inline_u32a, *a,
            "acked node {id:?} inline_u32a survives"
        );
        assert_eq!(
            rec.inline_u32b, *b,
            "acked node {id:?} inline_u32b survives"
        );
    }

    // The DOOMED (post-fault, un-acked, > watermark) commit does NOT survive.
    // This is the crash-consistency boundary: survivors == {commits ≤ fsync
    // watermark} (ADR-034 §Slice B).
    assert!(
        read_node_with_store(&crud2, &tx, doomed_id)
            .expect("read doomed node")
            .is_none(),
        "un-acked commit (NodeId={doomed_id:?}) MUST NOT survive — the fsync watermark is the crash boundary (ADR-034 §Slice B)",
    );
}

// ─────────────────────────────────────────────────────────────────────
// --in-memory negative pin — ephemeral / non-durable (ADR-183 §In-memory).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn in_memory_is_ephemeral_negative_pin() {
    // Session 1: in-memory, commit a node, confirm it reads back in-session.
    let committed_id = {
        let (backend, guard) =
            bootstrap_storage_backend(&BootstrapMode::InMemory).expect("in-memory bootstrap");
        assert!(!guard.is_durable(), "in-memory mode owns no WAL writer");
        assert!(guard.last_durable_lsn().is_none());
        let crud = crud_for(&backend, TenantId::DEFAULT);
        let (id, _) = commit_node(&backend, &crud, TenantId::DEFAULT, 1, 42, 43);
        // In-session readback works (the store is live).
        let tx = backend.txn_manager().begin(TenantId::DEFAULT);
        assert!(
            read_node_with_store(&crud, &tx, id)
                .expect("read")
                .is_some(),
            "in-memory node is visible in-session",
        );
        id
    };

    // Session 2: a FRESH in-memory bootstrap has no shared backing store, so
    // the previously-committed node is gone — documents the opt-in
    // NON-DURABLE behavior of --in-memory.
    let (backend2, _guard2) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("in-memory bootstrap 2");
    let crud2 = crud_for(&backend2, TenantId::DEFAULT);
    let tx = backend2.txn_manager().begin(TenantId::DEFAULT);
    assert!(
        read_node_with_store(&crud2, &tx, committed_id)
            .expect("read")
            .is_none(),
        "--in-memory is NON-DURABLE: a committed node MUST NOT survive a fresh in-memory bootstrap",
    );
}

// ─────────────────────────────────────────────────────────────────────
// ADR-183 §Forward-pin — multi-tenant registry recovery NOT in S1.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn multi_tenant_registry_recovery_is_forward_pinned() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");
    // A non-DEFAULT tenant. v1.0 has no `register_tenant` API, so this
    // tenant is never in the catalog's in-memory registry; its committed
    // records nonetheless write durable WAL bundles.
    let non_default = TenantId::new(0xBEEF);

    {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap");
        let crud = crud_for(&backend, TenantId::DEFAULT);

        // Commit a DEFAULT-tenant node (the GA scope — must recover) ...
        commit_node(&backend, &crud, TenantId::DEFAULT, 1, 10, 20);
        // ... and a NON-DEFAULT-tenant node (durable WAL write under a tenant
        // that is NOT in the registry; routed directly through the shared
        // store + txn manager, since route() would reject it).
        let mut tx = backend.txn_manager().begin(non_default);
        create_node(
            &crud,
            &mut tx,
            non_default,
            LabelId::new(1),
            &PropertyData::InlineU32Pair(1, 2),
        )
        .expect("create_node (non-default tenant)");
        commit(tx, &crud).expect("commit (non-default tenant) — durable WAL write");
    }

    // Restart.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (recover)");

    // GA scope: DEFAULT recovered.
    assert!(
        backend2.router().tenants().contains(&TenantId::DEFAULT),
        "DEFAULT tenant must be present after restart",
    );
    // FORWARD-PIN: the non-DEFAULT tenant is NOT in the recovered registry
    // (the catalog tenant list is bootstrap-derived, DEFAULT-only — it is
    // NOT recovered from pages). Its records replayed into the shared MVCC
    // store, but the tenant is not addressable through the multi-tenant
    // routing surface. Multi-tenant registry recovery needs the M10
    // catalog-recover-from-pages path (ADR-183 §Forward-pin). When that
    // lands, THIS assertion flips.
    assert!(
        !backend2.router().tenants().contains(&non_default),
        "FORWARD-PIN: non-DEFAULT tenant must NOT be in the recovered registry at S1",
    );
    assert!(
        backend2
            .router()
            .route(non_default, PartitionId::ZERO)
            .is_err(),
        "FORWARD-PIN: a non-DEFAULT tenant must NOT be routable after restart (registry recovery is M10)",
    );
}

// ─────────────────────────────────────────────────────────────────────
