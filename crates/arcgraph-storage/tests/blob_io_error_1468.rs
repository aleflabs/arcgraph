#![cfg(debug_assertions)]

//! #1468 RC gate: eviction spill ENOSPC is a typed error, never a panic.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_core::record::PAGE_SIZE;
use arcgraph_storage::blob::{BlobBoundConfig, BlobError, BlobSpill, BlobStore};
use tempfile::tempdir;

#[test]
fn eviction_enospc_propagates_typed_error_without_panicking() {
    let dir = tempdir().unwrap();
    let spill = Arc::new(BlobSpill::open(dir.path()).unwrap());
    let store = BlobStore::with_bound(
        Arc::clone(&spill),
        BlobBoundConfig::from_cap_bytes((PAGE_SIZE * 2) as u64),
    );
    let tenant = TenantId::new(1_468);
    let first = store.put(tenant, b"first non-default tenant page").unwrap();
    store
        .put(tenant, b"second non-default tenant page")
        .unwrap();
    store
        .for_each_resident_overflow_page(|_, _, _| Ok::<(), ()>(()))
        .unwrap();

    spill.__test_fail_next_write_enospc();
    let unwind = catch_unwind(AssertUnwindSafe(|| store.force_drain_for_test()));
    let drain = unwind.expect("an injected spill I/O failure must not unwind");
    let error = drain.expect_err("the injected spill I/O failure must propagate");
    match &error {
        BlobError::SpillIo {
            operation,
            tenant: error_tenant,
            page_id,
            kind,
            raw_os_error,
            message,
            ..
        } => {
            assert_eq!(*operation, "write");
            assert_eq!(*error_tenant, tenant);
            assert_eq!(*page_id, first.page_id);
            assert_eq!(*kind, std::io::ErrorKind::StorageFull);
            #[cfg(unix)]
            assert_eq!(*raw_os_error, Some(28));
            #[cfg(not(unix))]
            assert_eq!(*raw_os_error, None);
            assert!(
                message.contains("No space left on device")
                    || message.contains("injected blob spill storage-full failure")
            );
        }
        other => panic!("expected typed BlobError::SpillIo, got {other:?}"),
    }

    // Evict-after-durable still holds on failure: the page remains resident,
    // readable, and at the front of the queue for an explicit retry.
    assert_eq!(store.evicted_count(), 0);
    assert_eq!(
        store.get(tenant, first).unwrap().as_ref(),
        b"first non-default tenant page"
    );
    store.force_drain_for_test().unwrap();
    assert!(store.evicted_count() > 0);
}
