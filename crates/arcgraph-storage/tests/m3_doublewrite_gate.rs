//! M3 IMPL-DEC-7 doublewrite ordering and torn-page restore gate.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use arcgraph_core::record::{PAGE_SIZE, PageHeader, PageType};
use arcgraph_core::{ArcGraphError, Lsn, PageId, Result, TenantId};
use arcgraph_storage::io::PageBuf;
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::wal::{STORE_PROPS, STORE_RECORD};
use arcgraph_storage::{
    DoublewriteArea, DoublewriteKey, DoublewriteRestoreTarget, PageFlushTarget,
    WriteBehindCheckpointer,
};

fn page(key: DoublewriteKey, page_type: PageType, lsn: u64) -> Box<PageBuf> {
    let mut page = Box::new([0u8; PAGE_SIZE]);
    arcgraph_storage::records::SlottedPage::init(
        page.as_mut(),
        PageHeader::new(PageId::new(key.page_no), page_type, key.tenant_id),
    )
    .unwrap();
    page[16..24].copy_from_slice(&lsn.to_le_bytes());
    page
}

#[derive(Default)]
struct MemoryHome {
    pages: HashMap<DoublewriteKey, Box<PageBuf>>,
    writes: usize,
    syncs: usize,
    fail_write_number: Option<usize>,
}

impl DoublewriteRestoreTarget for MemoryHome {
    fn read_home(&mut self, key: DoublewriteKey) -> Result<Option<Box<PageBuf>>> {
        Ok(self.pages.get(&key).cloned())
    }

    fn write_home(&mut self, key: DoublewriteKey, page: &PageBuf) -> Result<()> {
        self.writes += 1;
        if self.fail_write_number == Some(self.writes) {
            return Err(ArcGraphError::Io(std::io::Error::other(
                "injected restore crash",
            )));
        }
        self.pages.insert(key, Box::new(*page));
        Ok(())
    }

    fn sync_home(&mut self) -> Result<()> {
        self.syncs += 1;
        Ok(())
    }
}

#[test]
fn torn_home_is_restored_and_second_restore_is_a_noop() {
    let dir = tempfile::tempdir().unwrap();
    let area = DoublewriteArea::new(dir.path());
    let key = DoublewriteKey {
        tenant_id: TenantId::DEFAULT,
        store_id: STORE_RECORD,
        page_no: 7,
    };
    let good = page(key, PageType::Node, 20);
    area.stage_batch(&[(key, good.as_ref())]).unwrap();

    let mut torn = good.clone();
    torn[PAGE_SIZE - 1] ^= 0xFF;
    let mut home = MemoryHome::default();
    home.pages.insert(key, torn);
    let first = area.restore(&mut home).unwrap();
    assert_eq!(first.restored_pages, 1);
    assert_eq!(home.pages[&key], good);
    assert_eq!(home.syncs, 1);

    let second = area.restore(&mut home).unwrap();
    assert_eq!(second.restored_pages, 0);
    assert_eq!(second.skipped_newer_homes, 1);
    assert_eq!(home.syncs, 1);
}

#[test]
fn torn_dwb_batch_is_ignored_before_any_home_restore() {
    let dir = tempfile::tempdir().unwrap();
    let area = DoublewriteArea::new(dir.path());
    let key = DoublewriteKey {
        tenant_id: TenantId::DEFAULT,
        store_id: STORE_PROPS,
        page_no: 8,
    };
    let good = page(key, PageType::PropSlotted, 30);
    area.stage_batch(&[(key, good.as_ref())]).unwrap();
    let file = OpenOptions::new().write(true).open(area.path()).unwrap();
    let len = file.metadata().unwrap().len();
    file.set_len(len - 17).unwrap();
    file.sync_data().unwrap();

    let mut home = MemoryHome::default();
    let report = area.restore(&mut home).unwrap();
    assert!(report.ignored_torn_batch);
    assert_eq!(report.restored_pages, 0);
    assert_eq!(home.writes, 0);
}

#[test]
fn interrupted_restore_converges_on_retry() {
    let dir = tempfile::tempdir().unwrap();
    let area = DoublewriteArea::new(dir.path());
    let a = DoublewriteKey {
        tenant_id: TenantId::DEFAULT,
        store_id: STORE_RECORD,
        page_no: 1,
    };
    let b = DoublewriteKey { page_no: 2, ..a };
    let pa = page(a, PageType::Node, 10);
    let pb = page(b, PageType::Node, 11);
    area.stage_batch(&[(a, pa.as_ref()), (b, pb.as_ref())])
        .unwrap();

    let mut home = MemoryHome {
        fail_write_number: Some(2),
        ..MemoryHome::default()
    };
    assert!(area.restore(&mut home).is_err());
    assert_eq!(home.pages.get(&a), Some(&pa));
    assert!(!home.pages.contains_key(&b));
    home.fail_write_number = None;
    let report = area.restore(&mut home).unwrap();
    assert_eq!(report.skipped_newer_homes, 1);
    assert_eq!(report.restored_pages, 1);
    assert_eq!(home.pages.get(&b), Some(&pb));
}

struct OrderingTarget {
    image: Box<PageBuf>,
    dwb_path: std::path::PathBuf,
    home_called_after_dwb: AtomicBool,
}

impl PageFlushTarget for OrderingTarget {
    fn copy_page_pinned(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
    ) -> Result<Option<Box<PageBuf>>> {
        Ok(Some(self.image.clone()))
    }

    fn write_pages_home(&self, _images: &[(TenantId, PageId, Box<PageBuf>)]) -> Result<()> {
        let durable_shape_present = std::fs::metadata(&self.dwb_path)
            .map(|metadata| metadata.len() > PAGE_SIZE as u64)
            .unwrap_or(false);
        self.home_called_after_dwb
            .store(durable_shape_present, Ordering::Release);
        Ok(())
    }
}

#[test]
fn write_behind_stages_and_fsyncs_dwb_before_home_callback() {
    let dir = tempfile::tempdir().unwrap();
    let area = Arc::new(DoublewriteArea::new(dir.path()));
    let key = DoublewriteKey {
        tenant_id: TenantId::DEFAULT,
        store_id: STORE_RECORD,
        page_no: 42,
    };
    let target = Arc::new(OrderingTarget {
        image: page(key, PageType::Node, 42),
        dwb_path: area.path().to_path_buf(),
        home_called_after_dwb: AtomicBool::new(false),
    });
    let dpt = Arc::new(DirtyPageTable::new());
    dpt.mark_dirty(
        DirtyPageKey {
            tenant_id: key.tenant_id,
            store_id: key.store_id,
            page_no: key.page_no,
        },
        Lsn::new(42),
    );
    let props: Arc<dyn PageFlushTarget> = target.clone();
    let records: Arc<dyn PageFlushTarget> = target.clone();
    let report = WriteBehindCheckpointer::new(dpt, props, records)
        .with_doublewrite_area(area)
        .flush_pass(Lsn::new(50))
        .unwrap();
    assert_eq!(report.flushed_pages, 1);
    assert!(target.home_called_after_dwb.load(Ordering::Acquire));
}
