//! #835 storage-EXONERATION regression guard — a reverse supernode
//! ingested through a SERVER-PARITY storage stack must keep ALL its
//! inbound edges scannable via `scan_in` past the overflow boundary.
//!
//! # Why this test exists
//!
//! #835 reports that a SINK with >2048 inbound edges of one type loses
//! all but ~2048 of them, observed through the MCP server
//! (`graph.ingest` → `graph.raw_query`), in BOTH `--in-memory` and
//! `--data`. The report attributes this to the storage reverse-TEL
//! overflow path (`tel_append_reverse`) — the reverse analogue of the
//! forward #812/#826 silent-edge-loss cap.
//!
//! The investigation REFUTES that attribution at the storage layer. The
//! storage reverse path (`tel_append_reverse` + `scan_in`) returns ALL N
//! inbound edges at every storage configuration tested:
//!   - in-memory (`CrudStore::new`) — `crud::tests::supernode_inbound_fanin_beyond_cap_all_queryable`
//!   - durable cold-start rebuild — `recovery::tel_rebuild::tests::rebuild_reinstates_reverse_supernode_past_overflow_boundary`
//!   - THIS test: full SERVER-PARITY config (WAL + dual-write record
//!     store, the exact `bootstrap.rs::build_*` shape),
//!     with the MCP-ingest record ORDER (all nodes first, then all rels,
//!     in ONE transaction).
//!
//! The customer-observed loss was bisected to the reverse-direction
//! READ path through the MCP/query stack (the SAME rels read via FORWARD
//! expand `MATCH (l:Leaf)-[:FOLLOWS]->()` return in full = 10000; via
//! REVERSE expand `MATCH ()-[:FOLLOWS]->(s:Sink)` they collapse to
//! 7953), NOT the storage write/index. This test PINS the storage
//! invariant so a future storage regression on the reverse overflow
//! chain is caught — and documents, in code, that the storage layer is
//! exonerated for the #835 symptom.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use arcgraph_core::{LabelId, TenantId, TypeId};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, scan_in, scan_out,
};
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BackgroundFsyncFailAction, BackgroundFsyncScheduler, WalConfig, WalWriter,
};
use tempfile::TempDir;

fn wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        dir,
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: Duration::from_millis(2),
        group_commit_max_batch: 32,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// `FANOUT` > 2 × MAX_ENTRIES (4094) so the reverse chain spans three
/// blocks and exercises two distinct overflow events plus every
/// intervening grow-after-overflow — the scale the #835 report shows
/// losing one block (~2047) through the MCP read path.
const FANOUT: usize = 5000;

#[test]
fn reverse_supernode_survives_server_parity_ingest() {
    let workspace = TempDir::new().unwrap();
    let wal_dir = workspace.path().join("wal");
    std::fs::create_dir_all(&wal_dir).unwrap();

    // Server-parity storage stack (bootstrap.rs §5): WAL `Some(handle)` +
    // PrimaryIndex + dual-write record store.
    let writer = WalWriter::spawn(wal_config(wal_dir)).unwrap();
    let scheduler = BackgroundFsyncScheduler::start(
        writer.handle(),
        BackgroundFsyncFailAction::RollbackAndContinue,
    );
    let handle = writer.handle();
    let mut mgr_inner = TxnManager::with_wal(handle.clone());
    let catalog = Arc::new(SystemCatalog::new());
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    catalog.bootstrap(&pool, &mgr_inner).unwrap();
    mgr_inner.set_durability_lookup(catalog.clone());
    let mgr = Arc::new(mgr_inner);
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    );

    let tenant = TenantId::new(1);
    let ty = TypeId::new(1);

    // MCP `graph.ingest` parity: ONE transaction, ALL nodes first, THEN
    // all rels (StorageIngestProvider::ingest order). Every rel points at
    // the single SINK → a reverse fan-in supernode.
    let mut tx = mgr.begin(tenant);
    let sink = create_node(
        &store,
        &mut tx,
        tenant,
        LabelId::new(1),
        &PropertyData::Empty,
    )
    .unwrap();
    let mut srcs = Vec::with_capacity(FANOUT);
    for _ in 0..FANOUT {
        srcs.push(
            create_node(
                &store,
                &mut tx,
                tenant,
                LabelId::new(1),
                &PropertyData::Empty,
            )
            .unwrap(),
        );
    }
    for &src in &srcs {
        create_rel(&store, &mut tx, tenant, src, sink, ty, &PropertyData::Empty).unwrap();
    }
    commit(tx, &store).unwrap();

    let reader = mgr.begin(tenant);

    // Reverse read: the SINK's full inbound fan-in is scannable.
    let in_edges = scan_in(&store, &reader, sink, Some(ty)).expect("reverse index enabled");
    assert_eq!(
        in_edges.len(),
        FANOUT,
        "server-parity reverse supernode: every inbound edge must remain scannable \
         via scan_in (no reverse overflow drop at the storage layer)"
    );

    // Forward read of the SAME edges (each src has exactly one out-edge):
    // the MCP bisect showed forward returns the full count while reverse
    // collapses — pin that the storage layer agrees in BOTH directions.
    let fwd_total: usize = srcs
        .iter()
        .map(|s| scan_out(&store, &reader, *s, Some(ty)).count())
        .sum();
    assert_eq!(
        fwd_total, FANOUT,
        "server-parity: every source's single out-edge must remain scannable"
    );

    let _ = scheduler.shutdown();
    let _ = writer.shutdown();
}
