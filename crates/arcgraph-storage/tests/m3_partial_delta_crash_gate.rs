//! M3 partial-delta crash gate: every torn-tail offset yields the last
//! complete durable prefix, never a partially applied bundle.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use arcgraph_core::record::{NodeRecord, PageType};
use arcgraph_core::{LabelId, Lsn, NodeId, PageId, TenantId};
use arcgraph_storage::io::PageBuf;
use arcgraph_storage::primary_index::PrimaryPageStore;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    BUNDLE_FORMAT_V9, DeltaOp, DeltaOpKind, PageStoreTarget, ReplayConfig, ReplayExecutor,
    STORE_RECORD, SegmentHeader, WalRecord, WalRecordType, WalRecoveryReader,
    encode_commit_bundle_v9, segment_filename,
};
use arcgraph_storage::{DeltaPageStore, DirtyPageTable};
use bytes::Bytes;
use tempfile::tempdir;

#[derive(Default)]
struct TenantPages {
    pages: Mutex<HashMap<(TenantId, PageId), Box<PageBuf>>>,
}

impl DeltaPageStore for TenantPages {
    fn read_page_for_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
    ) -> arcgraph_core::Result<Option<Box<PageBuf>>> {
        Ok(self.pages.lock().unwrap().get(&(tenant, page_id)).cloned())
    }

    fn install_page_from_redo(
        &self,
        tenant: TenantId,
        page_id: PageId,
        page: Box<PageBuf>,
    ) -> arcgraph_core::Result<()> {
        self.pages.lock().unwrap().insert((tenant, page_id), page);
        Ok(())
    }
}

fn bundle(page_no: u64, base_lsn: u64) -> Vec<u8> {
    let mut alloc_payload = Vec::with_capacity(9);
    alloc_payload.push(PageType::Node.as_byte());
    alloc_payload.extend_from_slice(&page_no.to_le_bytes());
    let record = NodeRecord::new(
        NodeId::new(page_no),
        LabelId::new(1),
        Lsn::new(base_lsn + 1),
    );
    let deltas = [
        DeltaOp::new(
            DeltaOpKind::PageAlloc,
            STORE_RECORD,
            TenantId::DEFAULT,
            page_no,
            0,
            Lsn::new(base_lsn),
            Bytes::from(alloc_payload),
        )
        .unwrap(),
        DeltaOp::new(
            DeltaOpKind::PutRecord,
            STORE_RECORD,
            TenantId::DEFAULT,
            page_no,
            0,
            Lsn::new(base_lsn + 1),
            Bytes::copy_from_slice(&record.to_bytes()),
        )
        .unwrap(),
    ];
    encode_commit_bundle_v9(
        Lsn::new(base_lsn + 1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &deltas,
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .unwrap()
}

fn frame(txn_id: u64, lsn: u64, payload: Vec<u8>) -> Vec<u8> {
    let mut bytes = Vec::new();
    WalRecord {
        record_type: WalRecordType::CommitBundle,
        txn_id,
        lsn: Lsn::new(lsn),
        timestamp_ms: 0,
        tenant_id: TenantId::DEFAULT,
        payload,
    }
    .encode(&mut bytes)
    .unwrap();
    bytes
}

#[test]
fn every_partial_delta_frame_recovers_the_last_complete_prefix() {
    let first = frame(1, 2, bundle(1, 1));
    let second = frame(2, 11, bundle(2, 10));
    for cut in 0..second.len() {
        let dir = tempdir().unwrap();
        let mut segment = SegmentHeader {
            format_version: BUNDLE_FORMAT_V9,
        }
        .encode()
        .to_vec();
        segment.extend_from_slice(&first);
        segment.extend_from_slice(&second[..cut]);
        std::fs::write(dir.path().join(segment_filename(0)), segment).unwrap();

        let props = Arc::new(TenantPages::default());
        let records = Arc::new(TenantPages::default());
        let dpt = Arc::new(DirtyPageTable::new());
        let target = PageStoreTarget::primary_only(Arc::new(PrimaryPageStore::new()))
            .with_delta_stores(props, Arc::clone(&records) as Arc<dyn DeltaPageStore>, dpt);
        let mut replay = ReplayExecutor::new(
            ReplayConfig::with_wal_dir(dir.path()),
            Arc::new(TxnManager::new()),
            target,
        );
        assert_eq!(
            replay
                .run(WalRecoveryReader::open(dir.path()).unwrap())
                .unwrap(),
            Lsn::new(2),
            "tail cut {cut} advanced beyond the durable prefix"
        );
        let pages = records.pages.lock().unwrap();
        assert!(pages.contains_key(&(TenantId::DEFAULT, PageId::new(1))));
        assert!(
            !pages.contains_key(&(TenantId::DEFAULT, PageId::new(2))),
            "tail cut {cut} partially installed the torn bundle"
        );
    }
}
