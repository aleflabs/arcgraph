//! v2 M2 A4 round-2 (#1452) — the property-index CATALOG intern-
//! durability RED-on-revert gate (codex re-check residual-1; promoted
//! from the QA scratch `l1_m2_recheck_scratch.rs`, flipped from
//! assert-the-bug to assert-the-contract).
//!
//! # The defect this pins closed
//!
//! Round-1's A4 fix covered the property-WRITE path (`intern_logged`
//! callers), but `CrudExecutorSubstrate::create_property_index` still
//! resolved its label + property-key ids through the UNLOGGED
//! `intern_label` / `intern` — then embedded both ids verbatim in the
//! durable `PropertyIndexCatalog` transaction. Crash after the CREATE:
//! recovery replays the catalog record but has NO `InternString` for
//! either id, so the unseeded allocator re-hands both ids to whatever
//! unrelated names intern first — the durable index silently rebinds
//! ("indexed_prop" resolving as "unrelated_prop"), and every
//! subsequent maintain / lookup runs against the wrong property. Same
//! silent-wrong-name class as round-1, reached through the CATALOG
//! embedder instead of the block encoder.
//!
//! # The fix
//!
//! All three `create_property_index*` variants route BOTH legs through
//! the durable-proof logged intern (`intern_label_logged` +
//! `intern_string_logged`): the fsync-blocking `InternString` appends
//! RETURN before `register_building` commits the catalog record on the
//! SAME WAL, so the catalog tx's LSN is strictly greater and recovery
//! always installs the bindings the recovered record references. An
//! append failure aborts the CREATE before any catalog record exists.
//!
//! # RED-on-revert
//!
//! Revert the substrate's two logged legs to the unlogged
//! `intern_label` / `intern` and this gate fails 100%
//! deterministically: zero `InternString` records exist, recovery
//! resolves neither id (assert 2), the fresh allocator re-hands both
//! raw ids to the unrelated names (assert 3), and the WAL-order sweep
//! finds no intern record below the catalog record (assert 4). No
//! timing dependence — the "crash" is a WAL-writer shutdown and the
//! recovery side starts from a FRESH `InternTable`.

use std::sync::Arc;

use arcgraph_core::{Lsn, StringId, TenantId};
use arcgraph_index::SecondaryPageStore;
use arcgraph_mcp::storage::CrudExecutorSubstrate;
use arcgraph_query::executor::substrate::{ExecutorSubstrate, PropertyIndexRegistration};
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::property_index_catalog::PropertyIndexCatalog;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::secondary_handle::IndexState;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    PageStoreTarget, PrimaryPageStoreHandle, SecondaryPageStoreHandle, WalConfig, WalRecordType,
    WalRecoveryReader, WalWriter, recover_from_wal,
};
use tempfile::tempdir;

fn wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(1),
        group_commit_max_batch: 1,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

/// The A4 round-2 catalog gate: CREATE INDEX through the REAL
/// substrate on a durable stack, crash, recover into a FRESH intern
/// table, and require the recovered catalog record to resolve to its
/// ORIGINAL names — never a rebind.
#[test]
fn crash_after_create_property_index_resolves_original_names_never_a_rebind() {
    let dir = tempdir().expect("tempdir");
    let tenant = TenantId::DEFAULT;

    // ── Live side: a WAL-backed substrate (the production durable
    //    wiring: ONE WalHandle shared by the TxnManager — which the
    //    property-index manager commits the catalog tx through — and
    //    the CrudStore whose `wal()` the logged interns append to).
    let writer = WalWriter::spawn(wal_config(dir.path())).expect("spawn WAL");
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone()))
            .expect("primary"),
    );
    let crud = Arc::new(CrudStore::new_with_index(
        Some(handle),
        primary,
        Arc::clone(&alloc),
    ));
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(8, io);
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("bootstrap catalog");
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let live_intern = Arc::new(InternTable::new());
    let sub = CrudExecutorSubstrate::new(router, Arc::clone(&mgr), Arc::clone(&live_intern));

    // The CREATE under test: both ids are freshly allocated here and
    // embedded in the durable catalog record.
    let outcome = sub
        .create_property_index(
            tenant,
            "idx_pre_data",
            false,
            "IndexedBeforeData",
            "indexed_prop",
        )
        .expect("create_property_index");
    assert_eq!(outcome, PropertyIndexRegistration::Created);
    let live_label = live_intern
        .try_probe(tenant, "IndexedBeforeData")
        .expect("intern forward lookup")
        .expect("label interned by the create");
    let live_prop = live_intern
        .try_probe(tenant, "indexed_prop")
        .expect("intern forward lookup")
        .expect("property key interned by the create");

    // Crash-shaped shutdown: everything durable is in the WAL; the
    // live InternTable dies with the process.
    writer.shutdown().expect("shutdown");
    drop(sub);

    // ── Recovery side: FRESH TxnManager + FRESH InternTable. The
    //    secondary handle is registered because `register_building`
    //    bootstraps the index B+tree's root page through the WAL.
    let recovered_mgr = Arc::new(TxnManager::new());
    let rec_alloc = Arc::new(PageAllocator::new());
    let rec_primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&recovered_mgr), Arc::clone(&rec_alloc), None)
            .expect("primary"),
    );
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(rec_primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let secondary_handle: Arc<dyn SecondaryPageStoreHandle> = Arc::new(SecondaryPageStore::new());
    let recovered_intern = Arc::new(InternTable::new());
    let target = PageStoreTarget::new(primary_handle, secondary_handle)
        .with_intern_table(Arc::clone(&recovered_intern));
    let report =
        recover_from_wal(dir.path(), Arc::clone(&recovered_mgr), target, None).expect("recover");

    // (1) Exactly the create's two bindings were replayed (label +
    // property key — the durable-proof appends the fix added).
    assert_eq!(
        report.metrics.interns_recovered, 2,
        "the CREATE must have logged exactly its label + property-key interns",
    );

    // (2) The recovered catalog record resolves BOTH ids to their
    // ORIGINAL names in the fresh table.
    let recovered_catalog = PropertyIndexCatalog::new();
    recovered_catalog.recover(&recovered_mgr, Lsn::ZERO);
    let record = recovered_catalog
        .resolve(tenant, "idx_pre_data")
        .expect("durable catalog record recovered");
    assert_eq!(record.label.raw(), live_label.raw(), "label id round-trips");
    assert_eq!(
        record.property_key, live_prop,
        "property-key id round-trips"
    );
    assert_eq!(
        record.state,
        IndexState::Online,
        "both catalog txs recovered"
    );
    assert_eq!(
        recovered_intern
            .resolve(tenant, record.property_key)
            .as_deref()
            .map(String::as_str),
        Some("indexed_prop"),
        "the durable index's property id must resolve to its ORIGINAL name",
    );
    assert_eq!(
        recovered_intern
            .resolve(tenant, StringId::new(record.label.raw()))
            .as_deref()
            .map(String::as_str),
        Some("IndexedBeforeData"),
        "the durable index's label id must resolve to its ORIGINAL name",
    );

    // (3) No rebind: the recovered allocator is seeded PAST both ids,
    // so unrelated post-recovery interns get FRESH ids (pre-fix these
    // collided — the silent-rebind detonator).
    let unrelated_label = recovered_intern
        .intern(tenant, "UnrelatedLabel")
        .expect("intern");
    let unrelated_prop = recovered_intern
        .intern(tenant, "unrelated_prop")
        .expect("intern");
    assert_ne!(
        unrelated_label.raw(),
        record.label.raw(),
        "an unrelated post-crash intern must never reuse the index's label id",
    );
    assert_ne!(
        unrelated_prop, record.property_key,
        "an unrelated post-crash intern must never reuse the index's property id",
    );
    assert_eq!(
        recovered_intern
            .resolve(tenant, record.property_key)
            .as_deref()
            .map(String::as_str),
        Some("indexed_prop"),
        "the property id keeps its original binding after unrelated interns",
    );

    // (4) Ordering: every InternString record precedes the FIRST WAL
    // record carrying the catalog record's name bytes (the Building
    // commit). The durable-proof contract is append-RETURNS-then-
    // commit, so the intern LSNs are strictly lower.
    let records = WalRecoveryReader::open(dir.path())
        .expect("open WAL")
        .collect_all()
        .expect("collect");
    let max_intern_lsn = records
        .iter()
        .filter(|r| r.record_type == WalRecordType::InternString)
        .map(|r| r.lsn)
        .max()
        .expect("intern records exist (assert 1)");
    let first_catalog_lsn = records
        .iter()
        .filter(|r| {
            r.record_type != WalRecordType::InternString
                && r.payload
                    .windows(b"idx_pre_data".len())
                    .any(|w| w == b"idx_pre_data")
        })
        .map(|r| r.lsn)
        .min()
        .expect("a WAL record carries the catalog record's name bytes");
    assert!(
        max_intern_lsn < first_catalog_lsn,
        "every InternString ({max_intern_lsn:?}) must precede the catalog \
         record that references the ids ({first_catalog_lsn:?})",
    );
}
