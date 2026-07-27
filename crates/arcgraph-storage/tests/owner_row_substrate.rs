//! M4 Slice-3b-1 owner-row durability/correctness gates.

use std::collections::BTreeMap;
use std::sync::Arc;

use arcgraph_core::{Lsn, PAGE_SIZE, PageHeader, PageId, PageType, TenantId};
use arcgraph_storage::extent::{
    DIRECTORY_HEAD_BYTES, EXTENT_BYTES, EXTENT_PAGES, ExtentAllocation, ExtentDataPageStore,
    ExtentDirectory,
};
use arcgraph_storage::io::{PageIo, PosixPageIo};
use arcgraph_storage::owner_row::{
    OWNER_ROWS_PER_PAGE, OwnerRow, OwnerRowClass, OwnerRowError, OwnerRowStore,
};
use arcgraph_storage::records::{PageError, SLOT_AREA_START, SLOT_SIZE, SlottedPage};
use arcgraph_storage::redo::{DeltaPageStore, DirtyPageTable};
use arcgraph_storage::wal::DeltaIntent;
use tempfile::TempDir;

struct ProductionOwnerFixture {
    _root: TempDir,
    stores: BTreeMap<(TenantId, u16), Arc<ExtentDataPageStore>>,
    dpt: Arc<DirtyPageTable>,
    next_lsn: u64,
}

fn format_owner_page(
    tenant: TenantId,
    store_id: u16,
    page_no: u64,
    lsn: Lsn,
) -> Box<[u8; PAGE_SIZE]> {
    let mut bytes = Box::new([0_u8; PAGE_SIZE]);
    let mut header = PageHeader::new(PageId::new(page_no), PageType::PropSlotted, tenant);
    header.flags = store_id;
    header.lsn = lsn.raw();
    SlottedPage::init(bytes.as_mut(), header).unwrap();
    bytes
}

impl ProductionOwnerFixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let dpt = Arc::new(DirtyPageTable::new());
        let mut stores = BTreeMap::new();
        for tenant in [TenantId::new(41), TenantId::new(73)] {
            for store_id in [7_u16, 8, 9, 10] {
                let path = root
                    .path()
                    .join(format!("tenant-{}-{store_id}.store", tenant.raw()));
                let physical: Arc<dyn PageIo> = Arc::new(PosixPageIo::create(path).unwrap());
                let directory = Arc::new(ExtentDirectory::new(tenant, store_id, physical, 16));
                stores.insert(
                    (tenant, store_id),
                    Arc::new(ExtentDataPageStore::new(directory, 64)),
                );
            }
        }
        Self {
            _root: root,
            stores,
            dpt,
            next_lsn: 1,
        }
    }

    fn store(&self, tenant: TenantId, class: OwnerRowClass) -> Arc<ExtentDataPageStore> {
        Arc::clone(self.stores.get(&(tenant, class.store_id())).unwrap())
    }

    fn next_lsn(&mut self) -> Lsn {
        let lsn = Lsn::new(self.next_lsn);
        self.next_lsn += 1;
        lsn
    }

    fn ensure_extent(&mut self, store: &Arc<ExtentDataPageStore>, page_no: u64) {
        let logical_extent = page_no / EXTENT_PAGES;
        if store.directory().mapping(logical_extent).unwrap().is_some() {
            return;
        }
        let physical_offset = store.directory().recover_next_physical_offset().unwrap();
        let allocation = ExtentAllocation {
            logical_extent,
            physical_offset,
            pairing: u32::try_from(logical_extent).unwrap(),
        };
        let lsn = self.next_lsn();
        let op = DeltaIntent::extent_alloc(
            store.directory().store_id(),
            store.directory().tenant(),
            allocation,
        )
        .assign(lsn, lsn)
        .unwrap();
        store
            .directory()
            .apply_extent_alloc(&op, self.dpt.as_ref())
            .unwrap();
        let ordinal = physical_offset.checked_sub(DIRECTORY_HEAD_BYTES).unwrap() / EXTENT_BYTES;
        assert_eq!(
            physical_offset,
            DIRECTORY_HEAD_BYTES + ordinal * EXTENT_BYTES,
            "fixture must preserve the production dense-offset law"
        );
    }

    fn install_owner_row(&mut self, tenant: TenantId, class: OwnerRowClass, id: u64) {
        let store = self.store(tenant, class);
        let address = class.address(id).unwrap();
        self.ensure_extent(&store, address.page_no);
        let lsn = self.next_lsn();
        let mut bytes = format_owner_page(tenant, class.store_id(), address.page_no, lsn);
        let row =
            OwnerRow::new(class, id, format!("{tenant:?}:{class:?}:{id}").into_bytes()).unwrap();
        SlottedPage::open(bytes.as_mut())
            .unwrap()
            .put_bag_at(address.slot, &row.encode())
            .unwrap();
        store
            .install_page_from_redo(tenant, PageId::new(address.page_no), bytes)
            .unwrap();
    }

    fn install_wrong_type_page(&mut self, tenant: TenantId, class: OwnerRowClass, id: u64) {
        let store = self.store(tenant, class);
        let address = class.address(id).unwrap();
        self.ensure_extent(&store, address.page_no);
        let mut bytes = Box::new([0_u8; PAGE_SIZE]);
        let mut header = PageHeader::new(PageId::new(address.page_no), PageType::Node, tenant);
        header.flags = class.store_id();
        SlottedPage::init(bytes.as_mut(), header).unwrap();
        store
            .install_page_from_redo(tenant, PageId::new(address.page_no), bytes)
            .unwrap();
    }

    fn install_checksum_corruption(&mut self, tenant: TenantId, class: OwnerRowClass, id: u64) {
        let store = self.store(tenant, class);
        let address = class.address(id).unwrap();
        self.ensure_extent(&store, address.page_no);
        let lsn = self.next_lsn();
        let mut bytes = format_owner_page(tenant, class.store_id(), address.page_no, lsn);
        let row = OwnerRow::new(class, id, b"checksum-probe".to_vec()).unwrap();
        SlottedPage::open(bytes.as_mut())
            .unwrap()
            .put_bag_at(address.slot, &row.encode())
            .unwrap();
        bytes[PAGE_SIZE - 1] ^= 0x5a;
        store
            .install_page_from_redo(tenant, PageId::new(address.page_no), bytes)
            .unwrap();
    }

    fn install_read_bag_seam_corruption(
        &mut self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
    ) {
        let store = self.store(tenant, class);
        let address = class.address(id).unwrap();
        let high_address = class.address(id + 1).unwrap();
        assert_eq!(address.page_no, high_address.page_no);
        assert_eq!(address.slot.raw() + 1, high_address.slot.raw());
        self.ensure_extent(&store, address.page_no);
        let lsn = self.next_lsn();
        let mut bytes = format_owner_page(tenant, class.store_id(), address.page_no, lsn);
        let high_row = OwnerRow::new(class, id + 1, b"seam-high-water".to_vec()).unwrap();
        SlottedPage::open(bytes.as_mut())
            .unwrap()
            .put_bag_at(high_address.slot, &high_row.encode())
            .unwrap();

        // Keep the page checksum-valid and the target below high-water, but
        // make read_bag itself reject the target's slot entry as structural
        // corruption. Open and owner-page validation must both succeed first.
        let entry = SLOT_AREA_START + usize::from(address.slot.raw()) * SLOT_SIZE;
        bytes[entry..entry + 2].copy_from_slice(&1_u16.to_le_bytes());
        bytes[entry + 2..entry + SLOT_SIZE].copy_from_slice(&1_u16.to_le_bytes());
        let mut header = PageHeader::from_bytes(
            (&bytes[..PageHeader::SIZE])
                .try_into()
                .expect("fixed page header"),
        )
        .unwrap();
        header.checksum = crc32c::crc32c(&bytes[PageHeader::SIZE..]);
        bytes[..PageHeader::SIZE].copy_from_slice(&header.to_bytes());
        store
            .install_page_from_redo(tenant, PageId::new(address.page_no), bytes)
            .unwrap();
    }
}

#[test]
fn owner_unused_slot_read_taxonomy_defined() {
    let mut fixture = ProductionOwnerFixture::new();
    for tenant in [TenantId::new(41), TenantId::new(73)] {
        for class in OwnerRowClass::ALL {
            // Writing slot 2 creates two deterministic interior gaps.
            fixture.install_owner_row(tenant, class, 2);
            let reader = OwnerRowStore::new(fixture.store(tenant, class));
            assert!(reader.read(class, 0).unwrap().is_none());
            assert!(reader.read(class, 1).unwrap().is_none());
            assert_eq!(reader.read(class, 2).unwrap().unwrap().id(), 2);

            // Exact high-water boundary (slot 3, count 3) and a later slot are
            // both NotFound, never corruption and never another row's bytes.
            assert!(reader.read(class, 3).unwrap().is_none());
            assert!(reader.read(class, 4).unwrap().is_none());

            let wrong_type_id = OWNER_ROWS_PER_PAGE;
            fixture.install_wrong_type_page(tenant, class, wrong_type_id);
            assert!(matches!(
                reader.read(class, wrong_type_id),
                Err(OwnerRowError::PageIdentity(_))
            ));

            let corrupt_id = OWNER_ROWS_PER_PAGE * 2;
            fixture.install_checksum_corruption(tenant, class, corrupt_id);
            assert!(matches!(
                reader.read(class, corrupt_id),
                Err(OwnerRowError::Page(PageError::ChecksumMismatch { .. }))
            ));

            let seam_corrupt_id = OWNER_ROWS_PER_PAGE * 3 + 1;
            fixture.install_read_bag_seam_corruption(tenant, class, seam_corrupt_id);
            match reader.read(class, seam_corrupt_id) {
                Err(OwnerRowError::Page(PageError::Format(reason))) => assert!(
                    reason.contains("points outside record area"),
                    "unexpected read_bag seam error: {reason}"
                ),
                result => panic!(
                    "below-high-water read_bag corruption must remain a hard error, got {result:?}"
                ),
            }
        }
    }
}

#[test]
fn owner_fixture_offset_constants_are_consistent() {
    assert_eq!(DIRECTORY_HEAD_BYTES % PAGE_SIZE as u64, 0);
    assert_eq!(EXTENT_BYTES, EXTENT_PAGES * PAGE_SIZE as u64);
}
