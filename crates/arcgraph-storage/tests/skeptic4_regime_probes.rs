//! SKEPTIC-4 scratch — regime-honesty probes for the M1 headline
//! (332 B/node batch) + honesty control (25,373 B/commit single-record).
//! NOT for commit. Every probe prints `SKEPTIC4 <name>: ...` and makes
//! no tight assertion — the skeptic reads the numbers.
//!
//! Run twice:
//!   cargo test -p arcgraph-storage --test skeptic4_regime_probes -- --nocapture --test-threads=1
//!   ARCGRAPH_M1_FORCE_CHAINED_BAGS=1 cargo test ... (pre-M1 proxy, same binary)

use std::sync::Arc;

use arcgraph_core::{LabelId, TenantId, TypeId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, update_node,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{WalConfig, WalWriter};
use tempfile::TempDir;

fn test_wal_config(dir: &std::path::Path) -> WalConfig {
    WalConfig {
        dir: dir.to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 8,
        metrics_sink: None,
        encryption: None,
        inflight_budget_bytes: None,
    }
}

fn build_stack(
    wal_dir: &std::path::Path,
) -> (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
) {
    let writer = WalWriter::spawn(test_wal_config(wal_dir)).unwrap();
    let handle = writer.handle();
    let mgr = Arc::new(TxnManager::with_wal(handle.clone()));
    let alloc = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&alloc), Some(handle.clone())).unwrap(),
    );
    let store = Arc::new(CrudStore::new_with_index(
        Some(handle.clone()),
        Arc::clone(&primary),
        Arc::clone(&alloc),
    ));
    (writer, mgr, primary, store)
}

fn wal_dir_bytes(wal_dir: &std::path::Path) -> u64 {
    std::fs::read_dir(wal_dir)
        .expect("read wal dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_file())
        .map(|e| e.metadata().map(|m| m.len()).unwrap_or(0))
        .sum()
}

/// The gate's exact ~60 B incident bag.
fn small_bag(i: u32) -> Vec<u8> {
    format!(
        r#"{{"svc":"api-{i:04}","sev":"P{}","region":"us-east-1"}}"#,
        i % 4
    )
    .into_bytes()
}

/// A realistic single-large-string bag of ~`size` bytes (1-4 KiB
/// description/stack-trace-shaped value).
fn large_value_bag(i: u32, size: usize) -> Vec<u8> {
    let overhead = r#"{"svc":"api-0000","desc":""}"#.len();
    let pad = size.saturating_sub(overhead);
    format!(r#"{{"svc":"api-{i:04}","desc":"{}"}}"#, "x".repeat(pad)).into_bytes()
}

/// A realistic K-property JSON bag (~40 B per property).
fn many_props_bag(i: u32, k: usize) -> Vec<u8> {
    let mut s = String::from("{");
    for p in 0..k {
        s.push_str(&format!(r#""prop_{p:02}":"value-{i:06}-{p:02}-abcdefgh","#));
    }
    s.pop();
    s.push('}');
    s.into_bytes()
}

fn regime() -> &'static str {
    if std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref() == Ok("1") {
        "FORCED-CHAINED(pre-M1 proxy)"
    } else {
        "M1-slotted"
    }
}

/// Batch-ingest N nodes with `bag(i)` in ONE txn; B/node.
fn measure_batch<F: Fn(u32) -> Vec<u8>>(n: u32, bag: F) -> u64 {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 0..n {
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Blob(bag(i)),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    wal_dir_bytes(&wal_dir) / u64::from(n)
}

#[test]
fn probe_batch_60b_n200_control() {
    let per = measure_batch(200, small_bag);
    eprintln!(
        "SKEPTIC4 [{}] batch_60b_n200_control: {} B/node",
        regime(),
        per
    );
}

#[test]
fn probe_batch_60b_n2000_scale() {
    let per = measure_batch(2000, small_bag);
    eprintln!(
        "SKEPTIC4 [{}] batch_60b_n2000_scale: {} B/node",
        regime(),
        per
    );
}

#[test]
fn probe_batch_1kib_n200() {
    let per = measure_batch(200, |i| large_value_bag(i, 1024));
    eprintln!("SKEPTIC4 [{}] batch_1kib_n200: {} B/node", regime(), per);
}

#[test]
fn probe_batch_2kib_n200() {
    let per = measure_batch(200, |i| large_value_bag(i, 2048));
    eprintln!("SKEPTIC4 [{}] batch_2kib_n200: {} B/node", regime(), per);
}

#[test]
fn probe_batch_4kib_n200() {
    let per = measure_batch(200, |i| large_value_bag(i, 4096));
    eprintln!("SKEPTIC4 [{}] batch_4kib_n200: {} B/node", regime(), per);
}

#[test]
fn probe_batch_manyprops20_n200() {
    let sz = many_props_bag(0, 20).len();
    let per = measure_batch(200, |i| many_props_bag(i, 20));
    eprintln!(
        "SKEPTIC4 [{}] batch_manyprops20_n200 (bag={} B): {} B/node",
        regime(),
        sz,
        per
    );
}

#[test]
fn probe_batch_manyprops50_n200() {
    let sz = many_props_bag(0, 50).len();
    let per = measure_batch(200, |i| many_props_bag(i, 50));
    eprintln!(
        "SKEPTIC4 [{}] batch_manyprops50_n200 (bag={} B): {} B/node",
        regime(),
        sz,
        per
    );
}

/// Realistic mixed shape: 70% small, 20% 1 KiB, 10% > PROP_BAG_MAX (chained).
#[test]
fn probe_batch_mixed_realistic_n200() {
    let per = measure_batch(200, |i| match i % 10 {
        0 => vec![0xBB; 8300], // stays chained (> 8148)
        1 | 2 => large_value_bag(i, 1024),
        _ => small_bag(i),
    });
    eprintln!(
        "SKEPTIC4 [{}] batch_mixed_realistic_n200 (70%60B/20%1K/10%chained): {} B/node",
        regime(),
        per
    );
}

/// Middle regime: 200 nodes across 25 txns of 8 (small-batch OLTP).
#[test]
fn probe_small_batches_8_per_txn() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let mut n = 0u32;
    for _ in 0..25 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for _ in 0..8 {
            create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(1),
                &PropertyData::Blob(small_bag(n)),
            )
            .unwrap();
            n += 1;
        }
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();
    let per = wal_dir_bytes(&wal_dir) / u64::from(n);
    eprintln!(
        "SKEPTIC4 [{}] small_batches_8_per_txn (25 txns x 8): {} B/node",
        regime(),
        per
    );
}

/// The honesty-control shape: 16 lone single-record commits.
#[test]
fn probe_single_record_16() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    const N: u32 = 16;
    for i in 0..N {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(1),
            &PropertyData::Blob(small_bag(i)),
        )
        .unwrap();
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();
    let per = wal_dir_bytes(&wal_dir) / u64::from(N);
    eprintln!(
        "SKEPTIC4 [{}] single_record_16 (honesty shape): {} B/commit",
        regime(),
        per
    );
}

/// Rel-heavy: 100 nodes + 500 rels, all with ~60 B bags, ONE txn.
#[test]
fn probe_rel_heavy_100n_500r() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let mut tx = mgr.begin(TenantId::DEFAULT);
    let mut ids = Vec::new();
    for i in 0..100u32 {
        ids.push(
            create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(1),
                &PropertyData::Blob(small_bag(i)),
            )
            .unwrap(),
        );
    }
    for r in 0..500u32 {
        let src = ids[(r as usize) % ids.len()];
        let dst = ids[(r as usize * 7 + 1) % ids.len()];
        create_rel(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            src,
            dst,
            TypeId::new(5),
            &PropertyData::Blob(small_bag(1000 + r)),
        )
        .unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    let total = wal_dir_bytes(&wal_dir);
    eprintln!(
        "SKEPTIC4 [{}] rel_heavy_100n_500r: total={} B, {} B/element (600 elements)",
        regime(),
        total,
        total / 600
    );
}

/// Update-churn A: same node updated 100x, ONE txn.
#[test]
fn probe_churn_100_updates_one_txn() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let mut tx0 = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx0,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Blob(small_bag(0)),
    )
    .unwrap();
    commit(tx0, &store).unwrap();
    let base = wal_dir_bytes(&wal_dir);

    let mut tx = mgr.begin(TenantId::DEFAULT);
    for i in 1..=100u32 {
        update_node(&store, &mut tx, id, &PropertyData::Blob(small_bag(i))).unwrap();
    }
    commit(tx, &store).unwrap();
    writer.shutdown().unwrap();
    let total = wal_dir_bytes(&wal_dir) - base;
    eprintln!(
        "SKEPTIC4 [{}] churn_100_updates_one_txn: {} B total for the churn bundle",
        regime(),
        total
    );
}

/// Update-churn B: same node updated 100x, one txn PER update
/// (per-commit trajectory: does the filling pooled page change cost?).
#[test]
fn probe_churn_100_updates_txn_each() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let mut tx0 = mgr.begin(TenantId::DEFAULT);
    let id = create_node(
        &store,
        &mut tx0,
        TenantId::DEFAULT,
        LabelId::new(1),
        &PropertyData::Blob(small_bag(0)),
    )
    .unwrap();
    commit(tx0, &store).unwrap();
    let base = wal_dir_bytes(&wal_dir);

    let mut per_commit = Vec::new();
    let mut prev = base;
    for i in 1..=100u32 {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        update_node(&store, &mut tx, id, &PropertyData::Blob(small_bag(i))).unwrap();
        commit(tx, &store).unwrap();
        let now = wal_dir_bytes(&wal_dir);
        per_commit.push(now - prev);
        prev = now;
    }
    writer.shutdown().unwrap();
    let total: u64 = per_commit.iter().sum();
    eprintln!(
        "SKEPTIC4 [{}] churn_100_updates_txn_each: avg={} B/commit first={} mid={} last={}",
        regime(),
        total / 100,
        per_commit[0],
        per_commit[49],
        per_commit[99]
    );
}
