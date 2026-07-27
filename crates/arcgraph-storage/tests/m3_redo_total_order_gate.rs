//! M3 IMPL-DEC-2/4 gate: full sub-LSN ordering is a first-class
//! contract, independent of physical WAL append order.

use std::sync::{Arc, Barrier, Mutex};

use arcgraph_core::record::{NodeRecord, PAGE_SIZE, PageHeader, PageType};
use arcgraph_core::{LabelId, Lsn, NodeId, PageId, TenantId};
use arcgraph_storage::records::SlottedPage;
use arcgraph_storage::redo::{
    DirtyPageKey, DirtyPageTable, RedoLsnRange, apply_redo_if_newer, sort_by_redo_range,
};
use arcgraph_storage::transaction::LsnCounter;

#[test]
fn same_page_multi_op_bundle_applies_every_unique_sub_lsn() {
    const OPS: usize = 128;
    let counter = LsnCounter::new();
    let range = counter.allocate_range(OPS);
    assert_eq!(range.len(), OPS as u64);
    assert_eq!(range.commit_lsn(), Lsn::new(OPS as u64));

    let mut page_lsn = Lsn::ZERO;
    let mut applied = Vec::new();
    for index in 0..OPS {
        let op_lsn = range.op_lsn(index).unwrap();
        assert!(
            apply_redo_if_newer(&mut page_lsn, op_lsn, || {
                applied.push(index);
                Ok::<_, ()>(())
            })
            .unwrap()
        );
    }
    assert_eq!(applied, (0..OPS).collect::<Vec<_>>());
    assert_eq!(page_lsn, range.commit_lsn());

    // Always-on sensitivity control for cx/fb-M3-1: collapsing every
    // op back to commit_lsn makes the strict page-LSN rule drop ops
    // 2..N. This proves the equality oracle detects the named revert.
    let mut collapsed_page_lsn = Lsn::ZERO;
    let mut collapsed_applies = 0usize;
    for _ in 0..OPS {
        let _ = apply_redo_if_newer(&mut collapsed_page_lsn, range.commit_lsn(), || {
            collapsed_applies += 1;
            Ok::<_, ()>(())
        })
        .unwrap();
    }
    assert_eq!(collapsed_applies, 1, "sensitivity control became vacuous");
}

#[test]
fn slotted_page_stamps_each_full_sub_lsn() {
    let range = RedoLsnRange::new(Lsn::new(50), Lsn::new(52)).unwrap();
    let mut bytes = [0u8; PAGE_SIZE];
    let mut page = SlottedPage::init(
        &mut bytes,
        PageHeader::new(PageId::new(7), PageType::Node, TenantId::DEFAULT),
    )
    .unwrap();

    for index in 0..3usize {
        let op_lsn = range.op_lsn(index).unwrap();
        let record = NodeRecord::new(
            NodeId::new(index as u64 + 1),
            LabelId::new(9),
            range.commit_lsn(),
        );
        assert!(
            page.apply_redo_if_newer(op_lsn, |page| { page.insert_node(&record).map(|_| ()) })
                .unwrap()
        );
        assert_eq!(page.page_lsn(), op_lsn);
    }
    assert_eq!(page.slot_count(), 3);

    let first = range.op_lsn(0).unwrap();
    assert!(
        !page
            .apply_redo_if_newer(first, |_page| -> Result<(), ()> {
                panic!("already-covered sub-LSN must not reapply")
            })
            .unwrap()
    );
    assert_eq!(page.slot_count(), 3);
    assert_eq!(page.page_lsn(), range.commit_lsn());
}

#[derive(Debug, Clone)]
struct Batch {
    range: RedoLsnRange,
    values: Vec<u64>,
}

#[test]
fn recovery_sorts_out_of_append_order_with_gaps_and_duplicates() {
    let a = Batch {
        range: RedoLsnRange::new(Lsn::new(1), Lsn::new(3)).unwrap(),
        values: vec![1, 2, 3],
    };
    let b = Batch {
        range: RedoLsnRange::new(Lsn::new(6), Lsn::new(7)).unwrap(),
        values: vec![6, 7],
    };
    let c = Batch {
        range: RedoLsnRange::new(Lsn::new(8), Lsn::new(8)).unwrap(),
        values: vec![8],
    };

    // Physical order is deliberately later-first; B is duplicated.
    let mut physical = vec![c.clone(), b.clone(), a, b];
    let stats = sort_by_redo_range(&mut physical, |batch| batch.range).unwrap();
    assert_eq!(stats.duplicate_ranges, 1);
    assert_eq!(stats.gaps, 1);

    let mut page_lsn = Lsn::ZERO;
    let mut recovered = Vec::new();
    for batch in physical {
        for (index, value) in batch.values.into_iter().enumerate() {
            let op_lsn = batch.range.op_lsn(index).unwrap();
            let _ = apply_redo_if_newer(&mut page_lsn, op_lsn, || {
                recovered.push(value);
                Ok::<_, ()>(())
            })
            .unwrap();
        }
    }
    assert_eq!(recovered, vec![1, 2, 3, 6, 7, 8]);
    assert_eq!(page_lsn, Lsn::new(8));
}

#[test]
fn overlapping_nonduplicate_ranges_are_corruption() {
    let mut batches = [
        RedoLsnRange::new(Lsn::new(10), Lsn::new(12)).unwrap(),
        RedoLsnRange::new(Lsn::new(12), Lsn::new(14)).unwrap(),
    ];
    let err = sort_by_redo_range(&mut batches, |range| *range).unwrap_err();
    assert_eq!(err.previous, batches[0]);
    assert_eq!(err.overlapping, batches[1]);
}

#[test]
fn concurrent_range_allocation_is_unique_and_gapless() {
    const THREADS: usize = 8;
    const RANGES_PER_THREAD: usize = 100;
    let counter = Arc::new(LsnCounter::new());
    let start = Arc::new(Barrier::new(THREADS));
    let ranges = Arc::new(Mutex::new(Vec::new()));

    let mut workers = Vec::new();
    for worker in 0..THREADS {
        let counter = Arc::clone(&counter);
        let start = Arc::clone(&start);
        let ranges = Arc::clone(&ranges);
        workers.push(std::thread::spawn(move || {
            start.wait();
            let mut local = Vec::with_capacity(RANGES_PER_THREAD);
            for i in 0..RANGES_PER_THREAD {
                local.push(counter.allocate_range((worker + i) % 5 + 1));
            }
            ranges.lock().unwrap().extend(local);
        }));
    }
    for worker in workers {
        worker.join().unwrap();
    }

    let mut ranges = Arc::try_unwrap(ranges).unwrap().into_inner().unwrap();
    ranges.sort_by_key(|range| range.base());
    assert_eq!(ranges.len(), THREADS * RANGES_PER_THREAD);
    assert_eq!(ranges[0].base(), Lsn::new(1));
    for pair in ranges.windows(2) {
        assert_eq!(
            pair[1].base().raw(),
            pair[0].end().raw() + 1,
            "concurrent range allocation overlapped or left an internal gap"
        );
    }
    assert_eq!(counter.current(), ranges.last().unwrap().end());
}

#[test]
fn dpt_redirty_survives_generation_checked_flush_and_holds_redo_lsn() {
    let dpt = Arc::new(DirtyPageTable::new());
    let key = DirtyPageKey {
        tenant_id: TenantId::DEFAULT,
        store_id: 1,
        page_no: 42,
    };
    let first = dpt.mark_dirty(key, Lsn::new(10));
    let flush_snapshot = dpt.snapshot()[0];
    assert_eq!(first, flush_snapshot);

    let redirty_go = Arc::new(Barrier::new(2));
    let worker_dpt = Arc::clone(&dpt);
    let worker_go = Arc::clone(&redirty_go);
    let worker = std::thread::spawn(move || {
        worker_go.wait();
        worker_dpt.mark_dirty(key, Lsn::new(20))
    });
    redirty_go.wait();
    let redirtied = worker.join().unwrap();

    assert_eq!(redirtied.rec_lsn, Lsn::new(10));
    assert_eq!(redirtied.dirty_gen, flush_snapshot.dirty_gen + 1);
    assert!(
        !dpt.complete_flush(flush_snapshot),
        "stale flush snapshot must not clear a re-dirtied page"
    );
    assert_eq!(dpt.redo_lsn(Lsn::new(100)), Lsn::new(10));

    assert!(dpt.complete_flush(redirtied));
    assert!(dpt.is_empty());
    assert_eq!(dpt.redo_lsn(Lsn::new(100)), Lsn::new(100));
}

#[test]
fn redo_lsn_is_minimum_rec_lsn_not_checkpoint_lsn() {
    let dpt = DirtyPageTable::new();
    for (page_no, rec_lsn) in [(1, 80), (2, 55), (3, 90)] {
        dpt.mark_dirty(
            DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: 0,
                page_no,
            },
            Lsn::new(rec_lsn),
        );
    }
    assert_eq!(dpt.redo_lsn(Lsn::new(100)), Lsn::new(55));
}
