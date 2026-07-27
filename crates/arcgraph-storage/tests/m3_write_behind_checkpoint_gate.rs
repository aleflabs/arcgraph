//! M3 write-behind/DPT gate: pin-coupled copy, durable home write,
//! generation-checked removal, and min-recLSN frontier.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::Duration;

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType};
use arcgraph_core::{LabelId, Lsn, NodeId, PageId, Result, TenantId};
use arcgraph_storage::io::{InMemoryPageIo, PageBuf, PageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::records::{SlotId, SlottedPage, SlottedPageRef};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::{STORE_BLOB_OVERFLOW, STORE_PROPS, STORE_RECORD};
use arcgraph_storage::{PageFlushTarget, WriteBehindCheckpointer};

const WAIT: Duration = Duration::from_secs(30);

fn new_store() -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 16,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, 32))
}

fn install_node_page(store: &BufferedRecordPageStore, page_no: u64, node_id: u64) {
    let pid = PageId::new(page_no);
    store
        .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
        .unwrap();
    let pinned = store.latch_pinned(pid).unwrap();
    let mut guard = pinned.latch().write();
    let mut page = SlottedPage::open(guard.as_mut()).unwrap();
    page.put_node_at(
        SlotId(0),
        &NodeRecord::new(NodeId::new(node_id), LabelId::new(1), Lsn::new(1)),
    )
    .unwrap();
}

fn install_prop_page(store: &BufferedRecordPageStore, page_no: u64, payload: &[u8]) {
    let pid = PageId::new(page_no);
    let mut bytes = [0u8; PAGE_SIZE];
    let mut page = SlottedPage::init(
        &mut bytes,
        PageHeader::new(pid, PageType::PropSlotted, TenantId::DEFAULT),
    )
    .unwrap();
    page.put_bag_at(SlotId(0), payload).unwrap();
    RecordPageBackend::install_or_replace(store, pid, Box::new(bytes)).unwrap();
}

fn key(store_id: u16, page_no: u64) -> DirtyPageKey {
    DirtyPageKey {
        tenant_id: TenantId::DEFAULT,
        store_id,
        page_no,
    }
}

struct HookTarget {
    inner: Arc<BufferedRecordPageStore>,
    before_home: Mutex<Option<Box<dyn FnOnce() + Send>>>,
    max_batch: AtomicUsize,
}

impl HookTarget {
    fn new(inner: Arc<BufferedRecordPageStore>) -> Self {
        Self {
            inner,
            before_home: Mutex::new(None),
            max_batch: AtomicUsize::new(0),
        }
    }

    fn set_before_home(&self, hook: impl FnOnce() + Send + 'static) {
        *self.before_home.lock().unwrap() = Some(Box::new(hook));
    }
}

impl PageFlushTarget for HookTarget {
    fn copy_page_pinned(&self, _tenant: TenantId, page_id: PageId) -> Result<Option<Box<PageBuf>>> {
        self.inner.copy_page_pinned(page_id).map_err(|error| {
            arcgraph_core::ArcGraphError::Io(std::io::Error::other(error.to_string()))
        })
    }

    fn write_pages_home(&self, images: &[(TenantId, PageId, Box<PageBuf>)]) -> Result<()> {
        self.max_batch.fetch_max(images.len(), Ordering::AcqRel);
        // Drop the hook-slot mutex before invoking a rendezvous that may
        // intentionally block. This also lets a test arm the next home-write
        // callback while the current call is parked.
        let hook = self.before_home.lock().unwrap().take();
        if let Some(hook) = hook {
            hook();
        }
        let pages: Vec<_> = images
            .iter()
            .map(|(_, page_id, page)| (*page_id, page.clone()))
            .collect();
        self.inner.write_pages_home(&pages)
    }
}

#[test]
fn clean_pages_flush_home_leave_cache_and_advance_redo_to_checkpoint() {
    let props = new_store();
    let records = new_store();
    install_node_page(&records, 1, 10);
    install_prop_page(&props, 1, b"typed-props");
    let dpt = Arc::new(DirtyPageTable::new());
    dpt.mark_dirty(key(STORE_RECORD, 1), Lsn::new(10));
    dpt.mark_dirty(key(STORE_PROPS, 1), Lsn::new(11));

    let checkpointer = WriteBehindCheckpointer::new(dpt.clone(), props.clone(), records.clone());
    let report = checkpointer.flush_pass(Lsn::new(50)).unwrap();
    assert_eq!(report.snapshot_pages, 2);
    assert_eq!(report.flushed_pages, 2);
    assert_eq!(report.retained_redirties, 0);
    assert_eq!(report.redo_lsn, Lsn::new(50));
    assert!(dpt.is_empty());
    assert!(records.is_cached(PageId::new(1)), "flush is not eviction");

    assert!(records.try_evict_page_pinned(PageId::new(1), || true));
    records.fault_in(PageId::new(1)).unwrap();
    let latch = records.latch(PageId::new(1)).unwrap();
    let guard = latch.read();
    let page = SlottedPageRef::open(guard.as_ref()).unwrap();
    assert_eq!(page.read_node(SlotId(0)).unwrap().unwrap().id, 10);

    assert!(props.try_evict_page_pinned(PageId::new(1), || true));
    props.fault_in(PageId::new(1)).unwrap();
    let latch = props.latch(PageId::new(1)).unwrap();
    let guard = latch.read();
    assert_eq!(
        SlottedPageRef::open(guard.as_ref())
            .unwrap()
            .read_bag(SlotId(0))
            .unwrap()
            .unwrap(),
        b"typed-props"
    );
}

#[test]
fn concurrent_redirty_between_copy_and_home_survives_compare_and_remove() {
    let records = new_store();
    install_node_page(&records, 2, 20);
    let props = new_store();
    let dpt = Arc::new(DirtyPageTable::new());
    let page_key = key(STORE_RECORD, 2);
    dpt.mark_dirty(page_key, Lsn::new(10));

    let hooked = Arc::new(HookTarget::new(records.clone()));
    let (writer_go_tx, writer_go_rx) = mpsc::channel();
    let (writer_done_tx, writer_done_rx) = mpsc::channel();
    hooked.set_before_home(move || {
        writer_go_tx.send(()).unwrap();
        writer_done_rx.recv_timeout(WAIT).expect("writer stalled");
    });

    let writer_store = records.clone();
    let writer_dpt = dpt.clone();
    let writer = std::thread::spawn(move || {
        writer_go_rx.recv_timeout(WAIT).expect("flusher stalled");
        let pinned = writer_store.latch_pinned(PageId::new(2)).unwrap();
        let mut guard = pinned.latch().write();
        let mut page = SlottedPage::open(guard.as_mut()).unwrap();
        page.put_node_at(
            SlotId(0),
            &NodeRecord::new(NodeId::new(21), LabelId::new(1), Lsn::new(20)),
        )
        .unwrap();
        writer_dpt.mark_dirty(page_key, Lsn::new(20));
        writer_done_tx.send(()).unwrap();
    });

    let checkpointer = WriteBehindCheckpointer::with_batch_pages(dpt.clone(), props, hooked, 1);
    let report = checkpointer.flush_pass(Lsn::new(30)).unwrap();
    writer.join().unwrap();
    assert_eq!(report.retained_redirties, 1);
    assert_eq!(report.redo_lsn, Lsn::new(10));
    assert_eq!(dpt.len(), 1, "stale flush must retain the re-dirtied entry");

    let retry = WriteBehindCheckpointer::new(dpt.clone(), new_store(), records.clone());
    let retry_report = retry.flush_pass(Lsn::new(30)).unwrap();
    assert_eq!(retry_report.redo_lsn, Lsn::new(30));
    assert!(dpt.is_empty());
    assert!(records.try_evict_page_pinned(PageId::new(2), || true));
    records.fault_in(PageId::new(2)).unwrap();
    let latch = records.latch(PageId::new(2)).unwrap();
    let guard = latch.read();
    assert_eq!(
        SlottedPageRef::open(guard.as_ref())
            .unwrap()
            .read_node(SlotId(0))
            .unwrap()
            .unwrap()
            .id,
        21
    );
}

/// #1528 reproducer. Two priority flush calls in the shape used by eviction
/// must not publish home images out of admission order. This schedule parks an
/// older copied image before home I/O, publishes a newer dirty generation, and
/// probes whether the newer pass can overtake it.
#[test]
fn overlapping_priority_flushes_never_publish_stale_home_after_newer_generation() {
    const SERIALIZATION_PROBE: Duration = Duration::from_secs(5);

    let records = new_store();
    install_node_page(&records, 2, 20);
    let dpt = Arc::new(DirtyPageTable::new());
    let page_key = key(STORE_RECORD, 2);
    dpt.mark_dirty(page_key, Lsn::new(10));

    let hooked = Arc::new(HookTarget::new(records.clone()));
    let checkpointer = Arc::new(WriteBehindCheckpointer::with_batch_pages(
        dpt.clone(),
        new_store(),
        hooked.clone(),
        1,
    ));

    let (older_at_home_tx, older_at_home_rx) = mpsc::channel();
    let (release_older_tx, release_older_rx) = mpsc::channel();
    hooked.set_before_home(move || {
        older_at_home_tx.send(()).unwrap();
        release_older_rx
            .recv_timeout(WAIT)
            .expect("controller failed to release the older flush");
    });

    let older_checkpointer = checkpointer.clone();
    let older = std::thread::spawn(move || {
        older_checkpointer
            .flush_priority_keys(&[page_key])
            .expect("older priority flush")
    });
    older_at_home_rx
        .recv_timeout(WAIT)
        .expect("older flush never reached the pre-home rendezvous");
    #[cfg(feature = "fault-injection")]
    assert!(
        checkpointer.pass_admission_is_held_for_gate(),
        "the pass-admission guard must span the older pass's home write"
    );

    // Publish generation 2 while generation 1's copied image is parked.
    // Keep the production pin across mutation + DPT mark.
    {
        let pinned = records.latch_pinned(PageId::new(2)).unwrap();
        {
            let mut guard = pinned.latch().write();
            let mut page = SlottedPage::open(guard.as_mut()).unwrap();
            page.put_node_at(
                SlotId(0),
                &NodeRecord::new(NodeId::new(21), LabelId::new(1), Lsn::new(20)),
            )
            .unwrap();
        }
        dpt.mark_dirty(page_key, Lsn::new(20));
    }

    let (newer_at_home_tx, newer_at_home_rx) = mpsc::channel();
    hooked.set_before_home(move || {
        // The fixed schedule may release the probe receiver before this
        // serialized second callback runs.
        let _ = newer_at_home_tx.send(());
    });
    let (newer_started_tx, newer_started_rx) = mpsc::channel();
    let (newer_done_tx, newer_done_rx) = mpsc::channel();
    let newer_checkpointer = checkpointer.clone();
    let newer = std::thread::spawn(move || {
        newer_started_tx.send(()).unwrap();
        let completed = newer_checkpointer
            .flush_priority_keys(&[page_key])
            .expect("newer priority flush");
        // The serialized path times out its overtake probe and drops the
        // receiver before this admitted pass can finish.
        let _ = newer_done_tx.send(());
        completed
    });
    newer_started_rx
        .recv_timeout(WAIT)
        .expect("newer flush thread never started");

    let controller = std::thread::spawn(move || {
        let newer_overtook = match newer_at_home_rx.recv_timeout(SERIALIZATION_PROBE) {
            Ok(()) => {
                // Force the pre-fix stale-last ordering: newer must finish
                // its durable home + DPT removal before older is released.
                newer_done_rx
                    .recv_timeout(WAIT)
                    .expect("newer flush reached home but did not finish");
                true
            }
            Err(mpsc::RecvTimeoutError::Timeout) => false,
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                panic!("newer-home rendezvous disconnected")
            }
        };
        release_older_tx
            .send(())
            .expect("older flush dropped its release rendezvous");
        newer_overtook
    });

    let newer_overtook = controller.join().unwrap();
    let newer_completed = newer.join().unwrap();
    let older_completed = older.join().unwrap();

    assert!(dpt.is_empty(), "both flushes must leave the DPT drained");
    assert!(records.try_evict_page_pinned(PageId::new(2), || true));
    records.fault_in(PageId::new(2)).unwrap();
    let recovered_id = {
        let latch = records.latch(PageId::new(2)).unwrap();
        let guard = latch.read();
        SlottedPageRef::open(guard.as_ref())
            .unwrap()
            .read_node(SlotId(0))
            .unwrap()
            .unwrap()
            .id
    };

    assert_eq!(
        recovered_id, 21,
        "#1528 INV-M6.2 H1: stale generation-1 home image survived after \
         publishing dirty generation 2; newer_overtook={newer_overtook}, \
         older_completed={older_completed:?}, newer_completed={newer_completed:?}"
    );
    assert!(
        !newer_overtook,
        "newer priority flush reached home while the older pass was parked"
    );
    assert!(
        older_completed.is_empty(),
        "older generation must be retained when generation 2 is dirty"
    );
    assert_eq!(newer_completed.len(), 1);
    assert!(newer_completed.contains(&page_key));
}

#[test]
fn stale_completion_cannot_remove_a_reinserted_dirty_page_aba() {
    let dpt = DirtyPageTable::new();
    let page_key = key(STORE_RECORD, 7);

    let old_epoch = dpt.mark_dirty(page_key, Lsn::new(10));
    assert!(dpt.complete_flush(old_epoch));

    let new_epoch = dpt.mark_dirty(page_key, Lsn::new(20));
    assert_ne!(
        old_epoch.dirty_gen, new_epoch.dirty_gen,
        "remove/reinsert must allocate a fresh monotonic completion stamp"
    );
    assert!(
        !dpt.complete_flush(old_epoch),
        "a stale completion token must not remove the new dirty epoch"
    );
    assert_eq!(dpt.snapshot_key(page_key), Some(new_epoch));
    assert!(dpt.complete_flush(new_epoch));
}

#[test]
fn restored_dirty_generation_advances_the_runtime_stamp_clock() {
    let page_key = key(STORE_RECORD, 8);
    let source = DirtyPageTable::new();
    let mut restored = source.mark_dirty(page_key, Lsn::new(1));
    for lsn in 2..=7 {
        restored = source.mark_dirty(page_key, Lsn::new(lsn));
    }

    let dpt = DirtyPageTable::new();
    dpt.restore(&[restored]);
    assert!(dpt.complete_flush(restored));

    let redirtied = dpt.mark_dirty(page_key, Lsn::new(8));
    assert!(redirtied.dirty_gen > restored.dirty_gen);
    assert!(!dpt.complete_flush(restored));
    assert_eq!(dpt.snapshot_key(page_key), Some(redirtied));
}

#[test]
fn batches_are_resident_bounded_and_flush_every_snapshot_page() {
    let records = new_store();
    let dpt = Arc::new(DirtyPageTable::new());
    for page_no in 10..15u64 {
        install_node_page(&records, page_no, page_no);
        dpt.mark_dirty(key(STORE_RECORD, page_no), Lsn::new(page_no));
    }
    let target = Arc::new(HookTarget::new(records));
    let checkpointer =
        WriteBehindCheckpointer::with_batch_pages(dpt.clone(), new_store(), target.clone(), 2);
    let report = checkpointer.flush_pass(Lsn::new(100)).unwrap();
    assert_eq!(report.flushed_pages, 5);
    assert_eq!(target.max_batch.load(Ordering::Acquire), 2);
    assert!(dpt.is_empty());
}

#[test]
fn redo_lsn_is_minimum_retained_rec_lsn() {
    let records = new_store();
    let dpt = Arc::new(DirtyPageTable::new());
    // Missing page 99 forces the pass to stop before clearing either entry.
    install_node_page(&records, 3, 3);
    dpt.mark_dirty(key(STORE_RECORD, 3), Lsn::new(40));
    dpt.mark_dirty(key(STORE_RECORD, 99), Lsn::new(25));
    let checkpointer = WriteBehindCheckpointer::new(dpt.clone(), new_store(), records);
    assert!(checkpointer.flush_pass(Lsn::new(100)).is_err());
    assert_eq!(dpt.redo_lsn(Lsn::new(100)), Lsn::new(25));
}

#[test]
fn store_five_never_enters_the_m3_write_behind_set() {
    let dpt = Arc::new(DirtyPageTable::new());
    dpt.mark_dirty(key(STORE_BLOB_OVERFLOW, 5), Lsn::new(5));
    let checkpointer = WriteBehindCheckpointer::new(dpt.clone(), new_store(), new_store());
    let error = checkpointer.flush_pass(Lsn::new(10)).unwrap_err();
    assert!(error.to_string().contains("outside the M3 delta set"));
    assert_eq!(dpt.len(), 1);
}
