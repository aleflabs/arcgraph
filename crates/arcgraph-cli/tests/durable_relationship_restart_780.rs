//! P0 #780 — durable `--data` restart must recover RELATIONSHIPS so they are
//! traversable again. The #776/#782 fix recovered node data + intern names; a
//! durable restart still lost ALL relationships from traversal (`count(r) = 0`
//! of N durably-committed rels — perf-lane G1 + customer-zero).
//!
//! # Root cause (pinned)
//!
//! The relationship RECORDS survive a durable restart — WAL replay reinstates
//! them into the MVCC + record stores via the CommitBundle. But the in-memory
//! **TEL adjacency** index (`CrudStore::tel_chains` / `reverse_tel_chains`),
//! which `scan_out` / `scan_in` walk to serve `MATCH ()-[r]->()`, does NOT
//! participate in the CommitBundle (the MVCC↔TEL atomicity gap, issue #20) and
//! `tel_append` had no replay caller. So after a restart the adjacency chains
//! are empty and every relationship traversal returns 0 rows.
//!
//! # The fix
//!
//! `arcgraph_storage::recovery::rebuild_all_tenant_adjacency` — a cold-start
//! TEL rebuild that walks the recovered MVCC-visible rels and re-appends the
//! forward + reverse adjacency, mirroring the live commit drain and the §8
//! `CatalogStats` rebuild. Wired into the durable bootstrap at §8b.
//!
//! # This is the ADR-133 active verification for #780 (Storage/recovery class).
//!
//! Strong oracle = REAL relationship traversal RESULTS post-restart, not a
//! proxy: `count(r)` and `count(t:SENT)` are computed by scanning the
//! adjacency (NOT a CatalogStats fast-path — proven by the pre-fix value being
//! 0, never the catalog-rebuilt cardinality), and a `MATCH (a)-[:SENT]->(b)`
//! traversal returns the actual edge endpoint rows.
//!
//! RED→GREEN: on pre-fix code these counts/rows are 0 (the §8b rebuild absent);
//! post-fix they equal the committed edge count. The companion flipped test in
//! `durable_intern_restart_776.rs::relationship_traversal_recovers_after_durable_restart_780`
//! covers the single-edge case.

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::TenantId;
use arcgraph_mcp::tools::raw_query::{RawQueryExecutor, RawQueryRows};
use arcgraph_query::CancellationToken;
use tempfile::TempDir;

/// Run a `graph.raw_query` through the production storage executor, panicking
/// on executor error so a post-restart name/binding error surfaces as a test
/// failure with the diagnostic attached.
fn run_query(backend: &arcgraph_mcp::storage::StorageBackend, query: &str) -> RawQueryRows {
    use arcgraph_mcp::storage::StorageRawQueryExecutor;
    let exec = StorageRawQueryExecutor::new(backend.clone());
    let cancel = CancellationToken::new();
    exec.execute(TenantId::DEFAULT, query, 1000, &cancel)
        .unwrap_or_else(|e| panic!("query {query:?} failed: {e:?}"))
}

/// `count(*)`-style scalar: run `query` (RETURNing a single count column) and
/// read row 0 / col 0 as a u64.
fn run_count(backend: &arcgraph_mcp::storage::StorageBackend, query: &str) -> u64 {
    let rows = run_query(backend, query);
    let first = rows
        .rows
        .first()
        .unwrap_or_else(|| panic!("query {query:?} returned no rows"));
    let val = first
        .as_array()
        .and_then(|a| a.first())
        .unwrap_or_else(|| panic!("query {query:?} row 0 has no col 0: {first:?}"));
    val.as_u64()
        .unwrap_or_else(|| panic!("query {query:?} col 0 is not a u64: {val:?}"))
}

/// TWO typed relationships survive a durable `--data` restart: both are
/// traversable, the typed and untyped counts are correct, and a real
/// `MATCH (a)-[:SENT]->(b)` traversal returns BOTH edge rows.
///
/// This is the primary #780 acceptance test (the multi-edge superset of the
/// single-edge flipped test in `durable_intern_restart_776.rs`).
#[test]
fn multiple_relationships_survive_durable_restart_780() {
    let tmp = TempDir::new().expect("tempdir");
    let data_dir = tmp.path().join("db");

    // ── Session 1: durable bootstrap; CREATE a 2-edge SENT chain.
    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: data_dir.clone(),
        })
        .expect("durable bootstrap (session 1)");
        assert!(guard.is_durable(), "Durable mode must own a WAL writer");

        // Two typed SENT relationships (4 nodes, 2 edges), each through the
        // name-interning single-hop CREATE path. Multiple edges prove the
        // rebuild reinstates EVERY recovered rel (not just the first), and the
        // post-restart traversal must materialize BOTH as rows. (The parser
        // only takes single-hop CREATE patterns and CREATE cannot reference a
        // MATCH-bound node, so two independent edges is the robust shape; the
        // shared-vertex / reverse-adjacency path is covered by the storage
        // unit tests `rebuild_*` in `recovery::tel_rebuild`.)
        run_query(&backend, "CREATE (a:Account)-[:SENT]->(b:Account)");
        run_query(&backend, "CREATE (c:Account)-[:SENT]->(d:Account)");

        // In-session sanity: the live TEL adjacency sees BOTH edges.
        assert_eq!(
            run_count(&backend, "MATCH ()-[r]->() RETURN count(r)"),
            2,
            "in-session untyped edge count is 2",
        );
        assert_eq!(
            run_count(&backend, "MATCH (n) RETURN count(n)"),
            4,
            "in-session node count is 4",
        );
        assert_eq!(
            run_count(&backend, "MATCH ()-[t:SENT]->() RETURN count(t)"),
            2,
            "in-session typed SENT count is 2",
        );

        // Drop the guard → the WalWriter drains + fsyncs + joins (graceful
        // "process restart"). Both edges' CommitBundles are durable past this.
        drop(guard);
    }

    // ── Session 2: re-bootstrap the SAME dir → WAL recovery + §8b TEL rebuild.
    let (backend2, _guard2) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    })
    .expect("durable bootstrap (session 2 — recover)");

    // ORACLE 1 — untyped traversal count. Pre-fix: 0 (TEL empty). Post-fix: 2.
    assert_eq!(
        run_count(&backend2, "MATCH ()-[r]->() RETURN count(r)"),
        2,
        "#780: count(r) MUST be 2 after restart — both committed rels traversable",
    );

    // ORACLE 2 — count-store fast-path fallback: a durable-restart context may
    // not have per-tenant CatalogStats attached, so unfiltered count(n) must
    // degrade to the same scan path instead of surfacing counts-store
    // unavailable.
    assert_eq!(
        run_count(&backend2, "MATCH (n) RETURN count(n)"),
        4,
        "#904/#968 re-land: count(n) MUST scan-fallback to 4 after restart",
    );

    // ORACLE 3 — typed traversal count (channel projection rebuilt correctly).
    assert_eq!(
        run_count(&backend2, "MATCH ()-[t:SENT]->() RETURN count(t)"),
        2,
        "#780: typed count(t:SENT) MUST be 2 after restart",
    );

    // ORACLE 4 — a REAL traversal returns the actual edge endpoint rows (not a
    // count proxy): both `(a)-[:SENT]->(b)` hops materialize as rows.
    let traversal = run_query(&backend2, "MATCH (a)-[:SENT]->(b) RETURN a, b");
    assert_eq!(
        traversal.row_count, 2,
        "#780: MATCH (a)-[:SENT]->(b) MUST return both edge rows after restart; \
         got {traversal:?}",
    );
}
