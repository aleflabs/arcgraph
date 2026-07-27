//! P0 #776 / #780 — durable `--data` restart must recover the InternTable
//! label / rel-type **name↔id** mapping so `graph.schema` shows real names and
//! typed queries (`MATCH (a:Account)`) resolve after a process restart.
//!
//! # The bug (two independent reports)
//!
//! - **#776 (Customer-Zero):** after restarting a `--data` durable store, node
//!   data + properties + intern **IDs** survive, but label / rel-type **NAMES**
//!   are not recovered — `graph.schema` shows `label:1` / `type:2` instead of
//!   `Account` / `SENT`, and every typed query fails -32005 (`unknown label`).
//! - **#780 / perf-lane G1:** same restart, ALSO `count(r) = 0` (relationships
//!   not traversable post-restart).
//!
//! # Root cause (pinned — three-way gap for names)
//!
//! 1. **Write-side:** the production CREATE / ingest path interned names via
//!    `InternTable::intern_label` / `intern_type` (in-memory only); the
//!    WAL-logging `intern_logged` was test-only. So `WalRecordType::InternString`
//!    records were NEVER written in production.
//! 2. **Replay-side:** `wal/replay.rs` treated `InternString` as a no-op
//!    ("legacy pre-M2.c record … apply path not in ADR-032 scope").
//! 3. **Construction-side:** the durable bootstrap built a fresh empty
//!    `InternTable::new()` *after* recovery — nothing repopulated it.
//!
//! Because `graph.schema` + the query binder resolve names through the intern
//! table (an empty table → `format!("label:{id}")` synthetic names + a
//! `lookup_label` miss → `UnknownLabel` / -32005), all three gaps had to close
//! for names to round-trip.
//!
//! # This is the ADR-133 active verification for the #776 names fix.
//!
//! The first test drives the production `bootstrap_storage_backend(--data)`
//! surface end-to-end: CREATE typed nodes plus a typed relationship through the
//! name-interning `StorageRawQueryExecutor`, restart (drop the backend so the
//! WAL writer drains, fsyncs, and joins), re-bootstrap the SAME dir, and assert
//! `graph.schema` shows the real names plus a typed
//! `MATCH (a:Account) RETURN count(a)` returns the right count. The strong
//! oracle is the real typed-query result post-restart, NOT a proxy.
//!
//! The second test is the **#780 acceptance test** (originally a forward-pin
//! that asserted `count == 0`; FLIPPED here when the fix landed). After a
//! durable restart the relationship RECORDS survive (the #776 names-fix makes
//! `:SENT` resolve), AND — with the #780 fix — the in-memory TEL adjacency is
//! rebuilt from those recovered records by
//! `arcgraph_storage::recovery::rebuild_all_tenant_adjacency` (wired into the
//! durable bootstrap at §8b, mirroring the §8 `CatalogStats` rebuild). So
//! `MATCH ()-[r]->()` now traverses the recovered edge post-restart. The
//! companion `durable_relationship_restart_780.rs` carries the stronger
//! multi-edge chain round-trip (count + typed-count + real traversal rows).
//!
//! The TEL did not participate in the CommitBundle (the MVCC↔TEL atomicity
//! gap, `crud.rs` "Drain TEL appends AFTER commit … issue #20") and
//! `tel_append` had no replay caller — that was the root cause; the cold-start
//! rebuild closes it without changing the on-disk format.

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{StorageRawQueryExecutor, StorageSchemaProvider};
use arcgraph_mcp::tools::raw_query::{RawQueryExecutor, RawQueryRows};
use arcgraph_mcp::tools::schema::SchemaProvider;
use arcgraph_query::CancellationToken;
use tempfile::TempDir;

/// Run a `graph.raw_query` through the production storage executor over
/// `backend`, returning the materialized rows. Panics on executor error so a
/// post-restart -32005 (`unknown label`) surfaces as a test failure with the
/// diagnostic attached.
fn run_query(backend: &arcgraph_mcp::storage::StorageBackend, query: &str) -> RawQueryRows {
    let exec = StorageRawQueryExecutor::new(backend.clone());
    let cancel = CancellationToken::new();
    exec.execute(TenantId::DEFAULT, query, 1000, &cancel)
        .unwrap_or_else(|e| panic!("query {query:?} failed: {e:?}"))
}

/// `count(*)`-style scalar: run `query` (expected to RETURN a single count
/// column) and read row 0 / col 0 as a u64.
fn run_count(backend: &arcgraph_mcp::storage::StorageBackend, query: &str) -> u64 {
    let rows = run_query(backend, query);
    let first = rows
        .rows
        .first()
        .unwrap_or_else(|| panic!("query {query:?} returned no rows"));
    // The count scalar is the single column of the single row.
    let val = first
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or_else(|| panic!("query {query:?} row 0 has no col 0: {first:?}"));
    val.as_u64()
        .unwrap_or_else(|| panic!("query {query:?} col 0 is not a u64: {val:?}"))
}

// ─────────────────────────────────────────────────────────────────────
// #776 — InternTable name↔id mapping survives a durable restart.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn intern_names_survive_durable_restart_776() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: bootstrap durable, CREATE typed nodes + a typed rel.
    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");

        // Two Account nodes + one SENT relationship between them, in one
        // statement, through the name-interning CREATE path. This interns
        // "Account" + "SENT" and (with the fix) WAL-logs the new interns.
        let rows = run_query(
            &backend,
            "CREATE (a:Account)-[r:SENT]->(b:Account) RETURN r",
        );
        assert_eq!(rows.row_count, 1, "CREATE-rel emits one row binding r");

        // In-session sanity: the names are visible BEFORE restart (proves the
        // write path + intern table are live in-process).
        let schema = StorageSchemaProvider::new(backend.clone())
            .schema(TenantId::DEFAULT)
            .expect("in-session schema");
        assert!(
            schema.labels.iter().any(|l| l.name == "Account"),
            "in-session schema must show the real label name; got {:?}",
            schema.labels,
        );
        assert_eq!(
            run_count(&backend, "MATCH (a:Account) RETURN count(a)"),
            2,
            "in-session typed node count",
        );

        // Drop the guard → the WalWriter drains + fsyncs + joins (graceful
        // "process restart"). Both the InternString records and the
        // CommitBundle are durable past this point.
        drop(guard);
    }

    // ── Session 2: re-bootstrap the SAME dir → WAL recovery on startup.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    // #776 ORACLE 1 — `graph.schema` shows the REAL names, NOT `label:N`.
    let schema = StorageSchemaProvider::new(backend2.clone())
        .schema(TenantId::DEFAULT)
        .expect("post-restart schema");
    assert!(
        schema.labels.iter().any(|l| l.name == "Account"),
        "#776: label name `Account` MUST survive restart (got synthetic/empty: {:?})",
        schema.labels,
    );
    assert!(
        schema.labels.iter().all(|l| !l.name.starts_with("label:")),
        "#776: NO label may surface as a synthetic `label:N` after restart: {:?}",
        schema.labels,
    );
    assert!(
        schema.rel_types.iter().any(|t| t.name == "SENT"),
        "#776: rel-type name `SENT` MUST survive restart (got synthetic/empty: {:?})",
        schema.rel_types,
    );
    assert!(
        schema
            .rel_types
            .iter()
            .all(|t| !t.name.starts_with("type:")),
        "#776: NO rel-type may surface as a synthetic `type:N` after restart: {:?}",
        schema.rel_types,
    );

    // #776 ORACLE 2 — the typed node query RESOLVES the label name and returns
    // the right count post-restart (pre-fix this is -32005 `unknown label`).
    assert_eq!(
        run_count(&backend2, "MATCH (a:Account) RETURN count(a)"),
        2,
        "#776: typed `MATCH (a:Account)` MUST resolve + count 2 nodes after restart",
    );
}

// ─────────────────────────────────────────────────────────────────────
// #780 — relationship TEL adjacency RECOVERS after a durable restart.
// (Was a forward-pin asserting `count == 0`; FLIPPED when the fix landed.)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn relationship_traversal_recovers_after_durable_restart_780() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: CREATE a typed relationship; confirm it traverses IN-SESSION.
    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");

        run_query(
            &backend,
            "CREATE (a:Account)-[r:SENT]->(b:Account) RETURN r",
        );

        // In-session the live TEL adjacency chain is populated → traversal
        // sees the edge. This proves the edge data exists; the post-restart
        // assertions below prove the recovery rebuilds the adjacency.
        let in_session = run_query(&backend, "MATCH (a)-[r]->(b) RETURN r");
        assert_eq!(
            in_session.row_count, 1,
            "in-session edge traversal sees the live SENT edge",
        );

        drop(guard);
    }

    // ── Session 2: recover.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    // The #776 names fix makes `:SENT` RESOLVE (Ok, not -32005), and the #780
    // TEL cold-start rebuild repopulates the adjacency from the recovered rel
    // record — so the typed rel traversal now counts the edge post-restart.
    let typed_rel_count = run_count(&backend2, "MATCH ()-[t:SENT]->() RETURN count(t)");
    assert_eq!(
        typed_rel_count, 1,
        "#780: typed rel traversal MUST count the recovered SENT edge post-restart \
         (TEL adjacency rebuilt by rebuild_all_tenant_adjacency). Pre-fix this was 0.",
    );
    let untyped_rel = run_query(&backend2, "MATCH (a)-[r]->(b) RETURN r");
    assert_eq!(
        untyped_rel.row_count, 1,
        "#780: untyped edge traversal MUST see the recovered edge post-restart. \
         Pre-fix this was 0 (TEL adjacency not rebuilt).",
    );
}
