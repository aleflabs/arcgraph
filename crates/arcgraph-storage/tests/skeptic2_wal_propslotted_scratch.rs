//! SKEPTIC-2 SCRATCH — adversarial probes for PR #1437 (v2 M1).
//! Target property: "WAL PropSlotted encoding: replay byte-equality,
//! MIXED-version compat, untrusted-count bounds". NOT FOR COMMIT.
//!
//! s2a — upgrade-shape mixed WAL: OLD-BINARY-SHAPE chained SMALL bags
//!       (written by a subprocess under ARCGRAPH_M1_FORCE_CHAINED_BAGS=1,
//!       which is byte-identical to the pre-M1 write path) interleaved in
//!       ONE log with new PropSlotted bundles, replayed TWICE.
//! s2b — truncation sweep: cut the segment file at (nearly) every byte;
//!       recovery must never panic; Ok => readable prefix is byte-equal
//!       and never torn (record readable => bag readable + byte-equal).
//! s2c — CRC-smuggle bit-flips: corrupt bytes INSIDE a staged PropSlotted
//!       page image, then RECOMPUTE the WAL record CRC so the framing
//!       passes — the install-time full validation must reject LOUDLY.
//!       Also: staged-entry kind-byte swaps (5->3, 5->2, 5->invalid).
//! s2d — cross-txn pooled-page coalescing: two txns' bundles carry
//!       superseding images of the SAME page; replay order must yield
//!       the union, byte-equal.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::process::Command;
use std::sync::Arc;

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle, read_node_with_store,
};
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal,
};
use tempfile::TempDir;

// ── harness (mirrors tests/m1_slotted_packing.rs) ────────────────────

fn test_wal_config(dir: &Path) -> WalConfig {
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

type Stack = (
    WalWriter,
    Arc<TxnManager>,
    Arc<PrimaryIndex>,
    Arc<CrudStore>,
);

fn build_stack_with_alloc(wal_dir: &Path) -> (Stack, Arc<PageAllocator>) {
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
    ((writer, mgr, primary, store), alloc)
}

fn build_stack(wal_dir: &Path) -> Stack {
    build_stack_with_alloc(wal_dir).0
}

/// Like the m1 harness's `recover_stack`, but recovery errors are
/// RETURNED (the truncation/corruption sweeps need Err, not expect).
fn try_recover_stack(wal_dir: &Path) -> Result<Stack, String> {
    let ((writer, mgr, primary, store), alloc) = build_stack_with_alloc(wal_dir);
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
        store
            .records()
            .expect("dual-write stack has a record store"),
    ) as Arc<dyn RecordPageStoreHandle>;
    let blob_handle: Arc<dyn BlobStoreHandle> =
        Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
    let allocator_seed: Arc<dyn AllocatorSeedHandle> =
        crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
    let target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed);
    match recover_from_wal(wal_dir, Arc::clone(&mgr), target, None) {
        Ok(_) => Ok((writer, mgr, primary, store)),
        Err(e) => {
            writer.shutdown().ok();
            Err(format!("{e}"))
        }
    }
}

fn recover_stack(wal_dir: &Path) -> Stack {
    try_recover_stack(wal_dir).expect("recovery")
}

/// Deterministic small bag, distinct shape from the m1 suite's.
fn old_bag(i: u32) -> Vec<u8> {
    format!(
        r#"{{"legacy":{i},"pad":"{:08x}"}}"#,
        i.wrapping_mul(0x9E37_79B9)
    )
    .into_bytes()
}

fn new_bag(i: u32) -> Vec<u8> {
    format!(
        r#"{{"fresh":{i},"pad":"{:08x}"}}"#,
        i.wrapping_mul(0x85EB_CA6B)
    )
    .into_bytes()
}

fn read_bag_of(
    store: &Arc<CrudStore>,
    mgr: &Arc<TxnManager>,
    id: NodeId,
) -> Result<(u16, Vec<u8>), String> {
    let tx = mgr.begin(TenantId::DEFAULT);
    let rec = read_node_with_store(store, &tx, id)
        .map_err(|e| format!("read_node: {e}"))?
        .ok_or_else(|| "node missing".to_owned())?;
    let bref = BlobRef::decode(rec.property_ref).ok_or_else(|| "no overflow ref".to_owned())?;
    let got = store
        .blob_store()
        .get(TenantId::DEFAULT, bref)
        .map_err(|e| format!("blob get: {e}"))?;
    Ok((bref.slot_id, got.as_ref().to_vec()))
}

// ── s2a — upgrade-shape mixed WAL ────────────────────────────────────

const S2A_N_OLD: u32 = 12;

/// Subprocess-only helper: writes S2A_N_OLD SMALL bags with the forced
/// pre-M1 chain path into the wal dir given by S2A_DIR. Prints ids.
#[test]
#[ignore = "subprocess helper for s2a_upgrade_shape_mixed_wal_replays_both"]
fn helper_s2a_forced_chained_writer() {
    assert_eq!(
        std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref(),
        Ok("1"),
        "helper must run forced-chained"
    );
    let wal_dir = std::path::PathBuf::from(std::env::var("S2A_DIR").expect("S2A_DIR set"));
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    for i in 0..S2A_N_OLD {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        let id = create_node(
            &store,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::Blob(old_bag(i)),
        )
        .unwrap();
        commit(tx, &store).unwrap();
        println!("S2A_ID {} {}", i, id.raw());
    }
    writer.shutdown().unwrap();
}

/// (a) the REAL upgrade path: an old binary's log shape (small bags as
/// chained blobs, `BundlePageKind::Blob`, slot 0) interleaved in ONE
/// log with new PropSlotted bundles; replayed by the new code TWICE.
#[test]
fn s2a_upgrade_shape_mixed_wal_replays_both() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();

    // Phase 1 — "old binary": subprocess with the forced-chain lever
    // (byte-identical to the pre-M1 small-bag write path).
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(exe)
        .args([
            "--exact",
            "helper_s2a_forced_chained_writer",
            "--ignored",
            "--nocapture",
        ])
        .env("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")
        .env("S2A_DIR", &wal_dir)
        .output()
        .expect("spawn old-binary writer");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "old-binary writer failed.\nstdout:\n{stdout}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut old_ids: Vec<(u32, NodeId)> = Vec::new();
    for line in stdout.lines() {
        if let Some(rest) = line.strip_prefix("S2A_ID ") {
            let mut it = rest.split_whitespace();
            let i: u32 = it.next().unwrap().parse().unwrap();
            let raw: u64 = it.next().unwrap().parse().unwrap();
            old_ids.push((i, NodeId::new(raw)));
        }
    }
    assert_eq!(
        old_ids.len() as u32,
        S2A_N_OLD,
        "helper must report every id"
    );

    // Phase 2 — new binary opens the old log: replay chained bundles,
    // verify old-shape (slot 0) byte-equality, then append NEW slotted
    // bundles into the SAME log.
    let (writer2, mgr2, primary2, store2) = recover_stack(&wal_dir);
    for (i, id) in &old_ids {
        let (slot, got) = read_bag_of(&store2, &mgr2, *id).unwrap();
        assert_eq!(slot, 0, "pre-M1 bag must still be CHAINED (node {id:?})");
        assert_eq!(
            got,
            old_bag(*i),
            "chained bag byte-equal post-replay (node {id:?})"
        );
    }
    let mut new_ids: Vec<(u32, NodeId)> = Vec::new();
    let mut tx = mgr2.begin(TenantId::DEFAULT);
    for i in 0..10u32 {
        let id = create_node(
            &store2,
            &mut tx,
            TenantId::DEFAULT,
            LabelId::new(7),
            &PropertyData::Blob(new_bag(i)),
        )
        .unwrap();
        new_ids.push((i, id));
    }
    commit(tx, &store2).unwrap();
    for (i, id) in &new_ids {
        let (slot, got) = read_bag_of(&store2, &mgr2, *id).unwrap();
        assert!(slot >= 1, "new small bag must be SLOTTED (node {id:?})");
        assert_eq!(got, new_bag(*i));
    }
    writer2.shutdown().unwrap();
    drop((store2, primary2, mgr2));

    // Phase 3 — replay the MIXED log (chained + PropSlotted bundles in
    // one WAL) in one recovery pass.
    let (writer3, mgr3, _primary3, store3) = recover_stack(&wal_dir);
    for (i, id) in &old_ids {
        let (slot, got) = read_bag_of(&store3, &mgr3, *id).unwrap();
        assert_eq!(
            slot, 0,
            "mixed-log replay: old bag stays chained (node {id:?})"
        );
        assert_eq!(
            got,
            old_bag(*i),
            "mixed-log replay: old bag byte-equal (node {id:?})"
        );
    }
    for (i, id) in &new_ids {
        let (slot, got) = read_bag_of(&store3, &mgr3, *id).unwrap();
        assert!(
            slot >= 1,
            "mixed-log replay: new bag stays slotted (node {id:?})"
        );
        assert_eq!(
            got,
            new_bag(*i),
            "mixed-log replay: new bag byte-equal (node {id:?})"
        );
    }
    writer3.shutdown().unwrap();
}

// ── shared fixture for s2b / s2c ─────────────────────────────────────

/// Build a small store whose log ends with PropSlotted bundles; return
/// (tempdir, wal_dir, expected [(id, payload)]).
fn build_slotted_fixture() -> (TempDir, std::path::PathBuf, Vec<(NodeId, Vec<u8>)>) {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    let mut expected = Vec::new();
    // txn 1: two bags; txn 2: one bag (reuses the pooled page).
    for (t, n) in [(0u32, 2u32), (1, 1)] {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for i in 0..n {
            let payload = new_bag(t * 100 + i);
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(9),
                &PropertyData::Blob(payload.clone()),
            )
            .unwrap();
            expected.push((id, payload));
        }
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();
    (tmp, wal_dir, expected)
}

fn segment_file(wal_dir: &Path) -> std::path::PathBuf {
    let mut files: Vec<_> = std::fs::read_dir(wal_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    files.sort();
    assert_eq!(files.len(), 1, "fixture must fit one segment: {files:?}");
    files.remove(0)
}

/// One recovery attempt against a doctored image; classifies outcome.
enum Outcome {
    Ok { readable: usize, torn: Vec<String> },
    LoudErr(String),
    Panic(String),
}

fn probe_recovery(seg_name: &str, image: &[u8], expected: &[(NodeId, Vec<u8>)]) -> Outcome {
    let dir = TempDir::new().unwrap();
    let wal_dir = dir.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    std::fs::write(wal_dir.join(seg_name), image).unwrap();
    let res = catch_unwind(AssertUnwindSafe(|| {
        match try_recover_stack(&wal_dir) {
            Ok((writer, mgr, _primary, store)) => {
                let mut readable = 0usize;
                let mut torn = Vec::new();
                for (id, payload) in expected {
                    let tx = mgr.begin(TenantId::DEFAULT);
                    match read_node_with_store(&store, &tx, *id) {
                        Ok(Some(rec)) => {
                            // record present => bag MUST be present + byte-equal
                            match BlobRef::decode(rec.property_ref) {
                                Some(bref) => {
                                    match store.blob_store().get(TenantId::DEFAULT, bref) {
                                        Ok(got) if got.as_ref() == payload.as_slice() => {
                                            readable += 1;
                                        }
                                        Ok(_) => torn.push(format!(
                                            "node {id:?}: bag bytes DIFFER (silent corruption)"
                                        )),
                                        Err(e) => torn.push(format!(
                                            "node {id:?}: record readable but bag errored: {e}"
                                        )),
                                    }
                                }
                                None => torn.push(format!("node {id:?}: ref undecodable")),
                            }
                        }
                        Ok(None) => {} // not in replayed prefix — fine
                        Err(e) => torn.push(format!("node {id:?}: read errored: {e}")),
                    }
                }
                writer.shutdown().ok();
                Outcome::Ok { readable, torn }
            }
            Err(e) => Outcome::LoudErr(e),
        }
    }));
    match res {
        Ok(o) => o,
        Err(p) => {
            let msg = p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                .unwrap_or_else(|| "non-string panic".to_owned());
            Outcome::Panic(msg)
        }
    }
}

// ── s2b — truncation sweep ───────────────────────────────────────────

/// (b) truncate the log at every byte of the tail region (and a stride
/// over the head): NEVER panic; Ok => byte-equal never-torn prefix.
#[test]
fn s2b_truncation_sweep_never_panics_never_torn() {
    let (_tmp, wal_dir, expected) = build_slotted_fixture();
    let seg = segment_file(&wal_dir);
    let seg_name = seg.file_name().unwrap().to_str().unwrap().to_owned();
    let pristine = std::fs::read(&seg).unwrap();
    let len = pristine.len();
    eprintln!("s2b: segment = {len} bytes");

    // Dense over the final 14 KiB (the last PropSlotted bundle + tail),
    // stride 11 over the head. Cut 0 and 1..8 (segment header) included.
    let dense_from = len.saturating_sub(14 * 1024);
    let mut cuts: Vec<usize> = (0..dense_from).step_by(11).collect();
    cuts.extend(dense_from..=len);

    let mut n_ok = 0usize;
    let mut n_err = 0usize;
    for &cut in &cuts {
        match probe_recovery(&seg_name, &pristine[..cut], &expected) {
            Outcome::Ok { torn, .. } => {
                assert!(
                    torn.is_empty(),
                    "cut={cut}: TORN INSTALL on Ok recovery: {torn:?}"
                );
                n_ok += 1;
            }
            Outcome::LoudErr(_) => n_err += 1,
            Outcome::Panic(msg) => panic!("cut={cut}: recovery PANICKED: {msg}"),
        }
    }
    // Sanity: the sweep exercised both prefix-Ok and (possibly) err.
    eprintln!(
        "s2b: {} cuts -> ok={} loud-err={} (full-len must be Ok+all-readable)",
        cuts.len(),
        n_ok,
        n_err
    );
    match probe_recovery(&seg_name, &pristine, &expected) {
        Outcome::Ok { readable, torn } => {
            assert!(torn.is_empty(), "pristine torn: {torn:?}");
            assert_eq!(readable, expected.len(), "pristine must replay everything");
        }
        other => panic!(
            "pristine image must recover cleanly, got {}",
            match other {
                Outcome::LoudErr(e) => format!("err: {e}"),
                Outcome::Panic(m) => format!("panic: {m}"),
                Outcome::Ok { .. } => unreachable!(),
            }
        ),
    }
}

// ── s2c — CRC-smuggle corruption + kind-byte swaps ───────────────────

const REC_HDR: usize = 44; // WalRecord::HEADER_SIZE
const SEG_HDR: usize = 8; // segment file header (recovery.rs torn-tail note)
const SLOTTED_PREFIX: [u8; 6] = [0x41, 0x52, 0x43, 0x47, 0x02, 0x09]; // "ARCG"|ver 2|type 9

/// Walk record frames; return (record_start, record_len) list.
fn walk_records(image: &[u8]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut pos = SEG_HDR;
    while pos + REC_HDR <= image.len() {
        let len = u32::from_le_bytes(image[pos + 4..pos + 8].try_into().unwrap()) as usize;
        if len < REC_HDR || pos + len > image.len() {
            break;
        }
        out.push((pos, len));
        pos += len;
    }
    out
}

/// Find every staged PropSlotted page image start (absolute offset).
fn find_slotted_pages(image: &[u8]) -> Vec<usize> {
    (0..image.len().saturating_sub(6))
        .filter(|&i| image[i..i + 6] == SLOTTED_PREFIX)
        .collect()
}

fn refresh_record_crc(image: &mut [u8], rec_start: usize, rec_len: usize) {
    let crc = crc32c::crc32c(&image[rec_start + 4..rec_start + rec_len]);
    image[rec_start..rec_start + 4].copy_from_slice(&crc.to_le_bytes());
}

fn enclosing_record(records: &[(usize, usize)], off: usize) -> (usize, usize) {
    *records
        .iter()
        .find(|(s, l)| off >= *s && off < s + l)
        .expect("offset inside some record")
}

#[test]
fn s2c_crc_smuggled_page_corruption_is_loud_never_silent() {
    let (_tmp, wal_dir, expected) = build_slotted_fixture();
    let seg = segment_file(&wal_dir);
    let seg_name = seg.file_name().unwrap().to_str().unwrap().to_owned();
    let pristine = std::fs::read(&seg).unwrap();
    let records = walk_records(&pristine);
    assert!(!records.is_empty(), "must parse record frames");
    let pages = find_slotted_pages(&pristine);
    assert!(!pages.is_empty(), "fixture must stage PropSlotted images");
    eprintln!(
        "s2c: {} records, {} slotted page images",
        records.len(),
        pages.len()
    );
    let page = *pages.last().unwrap(); // last staged slotted image

    // Case 1..3: flip bytes INSIDE the page image (body middle, last
    // body byte, slot directory area) + refresh the record CRC — the
    // framing passes, so ONLY install-time validation can save us.
    for (label, off) in [
        ("body middle", page + 4096),
        ("last body byte", page + 8191),
        ("slot dir first entry", page + 40),
    ] {
        let mut img = pristine.clone();
        img[off] ^= 0xFF;
        let (rs, rl) = enclosing_record(&records, off);
        refresh_record_crc(&mut img, rs, rl);
        match probe_recovery(&seg_name, &img, &expected) {
            Outcome::LoudErr(e) => {
                eprintln!("s2c[{label}]: LOUD err (good): {e}");
            }
            Outcome::Ok { torn, .. } => panic!(
                "s2c[{label}]: smuggled page corruption ACCEPTED (torn={torn:?}) — install \
                 validation failed to catch a body flip"
            ),
            Outcome::Panic(m) => panic!("s2c[{label}]: PANICKED: {m}"),
        }
    }

    // Case 4: header slot_count/free_space bytes (36..40) are OUTSIDE
    // the body CRC — flip each + refresh record CRC. Any outcome except
    // panic/silent-wrong-bytes is acceptable (bounded reads).
    for hoff in [page + 36, page + 37, page + 38, page + 39] {
        let mut img = pristine.clone();
        img[hoff] ^= 0x01;
        let (rs, rl) = enclosing_record(&records, hoff);
        refresh_record_crc(&mut img, rs, rl);
        match probe_recovery(&seg_name, &img, &expected) {
            Outcome::Panic(m) => panic!("s2c[hdr {hoff}]: PANICKED: {m}"),
            Outcome::Ok { torn, .. } => {
                for t in &torn {
                    assert!(
                        !t.contains("silent corruption"),
                        "s2c[hdr {hoff}]: SILENT WRONG BYTES: {t}"
                    );
                }
                eprintln!("s2c[hdr {hoff}]: Ok, degraded-loud={torn:?}");
            }
            Outcome::LoudErr(e) => eprintln!("s2c[hdr {hoff}]: LOUD err: {e}"),
        }
    }

    // Case 5: staged-entry KIND byte swaps. kind sits 21 bytes before
    // the page image (kind|page_id 8|tenant 8|n_bytes 4|page).
    for (label, kind, must_err) in [
        (
            "kind 5->3 (Blob tag, classifier must still install slotted)",
            3u8,
            false,
        ),
        ("kind 5->2 (Record store route)", 2u8, false),
        ("kind 5->6 (invalid)", 6u8, true),
    ] {
        let koff = page - 21;
        let mut img = pristine.clone();
        assert_eq!(img[koff], 5, "expected PropSlotted kind byte at {koff}");
        img[koff] = kind;
        let (rs, rl) = enclosing_record(&records, koff);
        refresh_record_crc(&mut img, rs, rl);
        match probe_recovery(&seg_name, &img, &expected) {
            Outcome::Panic(m) => panic!("s2c[{label}]: PANICKED: {m}"),
            Outcome::LoudErr(e) => {
                eprintln!("s2c[{label}]: LOUD err: {e}");
            }
            Outcome::Ok { readable, torn } => {
                assert!(
                    !must_err,
                    "s2c[{label}]: invalid kind byte must be WalCorruption, got Ok"
                );
                for t in &torn {
                    assert!(
                        !t.contains("silent corruption"),
                        "s2c[{label}]: SILENT WRONG BYTES: {t}"
                    );
                }
                eprintln!("s2c[{label}]: Ok readable={readable} degraded-loud={torn:?}");
            }
        }
    }
}

// ── s2d — cross-txn pooled-page supersession through replay ─────────

#[test]
fn s2d_pooled_page_supersession_replays_union() {
    let tmp = TempDir::new().unwrap();
    let wal_dir = tmp.path().join("wal");
    std::fs::create_dir(&wal_dir).unwrap();
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);

    // txn A: 2 bags. txn B (separate bundle): 1 bag on the SAME page.
    let mut refs = Vec::new();
    let mut expected = Vec::new();
    for (t, n) in [(0u32, 2u32), (1, 1)] {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for i in 0..n {
            let payload = new_bag(t * 1000 + i);
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(4),
                &PropertyData::Blob(payload.clone()),
            )
            .unwrap();
            expected.push((id, payload));
        }
        commit(tx, &store).unwrap();
    }
    let tx = mgr.begin(TenantId::DEFAULT);
    for (id, _) in &expected {
        let rec = read_node_with_store(&store, &tx, *id).unwrap().unwrap();
        let bref = BlobRef::decode(rec.property_ref).unwrap();
        refs.push(bref);
    }
    drop(tx);
    // The gate is only meaningful if txn B actually REUSED txn A's page.
    assert_eq!(
        refs[0].page_id, refs[2].page_id,
        "txn B must coalesce onto txn A's pooled page (else this probe is vacuous)"
    );
    assert!(refs.iter().all(|r| r.slot_id >= 1), "all slotted");
    let slots: std::collections::HashSet<u16> = refs.iter().map(|r| r.slot_id).collect();
    assert_eq!(slots.len(), 3, "3 distinct slots on the shared page");
    writer.shutdown().unwrap();
    drop((store, mgr));

    // Replay: bundle A's image (2 bags) then bundle B's (3 bags) — the
    // union must survive; wrong install order would tombstone B's slot.
    let (writer2, mgr2, _primary2, store2) = recover_stack(&wal_dir);
    for (id, payload) in &expected {
        let (slot, got) = read_bag_of(&store2, &mgr2, *id).unwrap();
        assert!(slot >= 1);
        assert_eq!(&got, payload, "post-replay byte-equality (node {id:?})");
    }
    writer2.shutdown().unwrap();
}

// ── s2e — DIAGNOSTIC: what are the truncation errors, and does the
//    PRE-M1 (forced-chained) log shape error the same way? ───────────

#[test]
#[ignore = "subprocess helper for s2e"]
fn helper_s2e_chained_fixture_writer() {
    assert_eq!(
        std::env::var("ARCGRAPH_M1_FORCE_CHAINED_BAGS").as_deref(),
        Ok("1"),
    );
    let wal_dir = std::path::PathBuf::from(std::env::var("S2E_DIR").unwrap());
    let (writer, mgr, _primary, store) = build_stack(&wal_dir);
    for (t, n) in [(0u32, 2u32), (1, 1)] {
        let mut tx = mgr.begin(TenantId::DEFAULT);
        for i in 0..n {
            let id = create_node(
                &store,
                &mut tx,
                TenantId::DEFAULT,
                LabelId::new(9),
                &PropertyData::Blob(new_bag(t * 100 + i)),
            )
            .unwrap();
            println!("S2E_ID {}", id.raw());
        }
        commit(tx, &store).unwrap();
    }
    writer.shutdown().unwrap();
}

fn sparse_sweep(seg_name: &str, pristine: &[u8], expected: &[(NodeId, Vec<u8>)], label: &str) {
    let len = pristine.len();
    let cuts: Vec<usize> = (0..len).step_by(len / 300).collect();
    let mut ok = 0;
    let mut errs: Vec<(usize, String)> = Vec::new();
    for &cut in &cuts {
        match probe_recovery(seg_name, &pristine[..cut], expected) {
            Outcome::Ok { torn, .. } => {
                assert!(torn.is_empty(), "{label} cut={cut}: torn {torn:?}");
                ok += 1;
            }
            Outcome::LoudErr(e) => errs.push((cut, e)),
            Outcome::Panic(m) => panic!("{label} cut={cut}: PANIC {m}"),
        }
    }
    eprintln!(
        "{label}: len={len} cuts={} ok={ok} err={}",
        cuts.len(),
        errs.len()
    );
    let mut seen = std::collections::HashSet::new();
    for (cut, e) in &errs {
        let key: String = e.chars().take(60).collect();
        if seen.insert(key) {
            eprintln!("{label}: cut={cut}: {e}");
        }
    }
}

#[test]
fn s2e_truncation_error_shape_slotted_vs_pre_m1_chained() {
    // Slotted fixture (this PR's shape).
    let (_tmp1, wal_dir1, expected1) = build_slotted_fixture();
    let seg1 = segment_file(&wal_dir1);
    let name1 = seg1.file_name().unwrap().to_str().unwrap().to_owned();
    let bytes1 = std::fs::read(&seg1).unwrap();
    sparse_sweep(&name1, &bytes1, &expected1, "SLOTTED");

    // Pre-M1 forced-chained fixture (old-binary log shape), same data.
    let tmp2 = TempDir::new().unwrap();
    let wal_dir2 = tmp2.path().join("wal");
    std::fs::create_dir(&wal_dir2).unwrap();
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(exe)
        .args([
            "--exact",
            "helper_s2e_chained_fixture_writer",
            "--ignored",
            "--nocapture",
        ])
        .env("ARCGRAPH_M1_FORCE_CHAINED_BAGS", "1")
        .env("S2E_DIR", &wal_dir2)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let ids: Vec<NodeId> = stdout
        .lines()
        .filter_map(|l| l.strip_prefix("S2E_ID "))
        .map(|r| NodeId::new(r.trim().parse().unwrap()))
        .collect();
    assert_eq!(ids.len(), 3);
    let expected2: Vec<(NodeId, Vec<u8>)> = vec![
        (ids[0], new_bag(0)),
        (ids[1], new_bag(1)),
        (ids[2], new_bag(100)),
    ];
    let seg2 = segment_file(&wal_dir2);
    let name2 = seg2.file_name().unwrap().to_str().unwrap().to_owned();
    let bytes2 = std::fs::read(&seg2).unwrap();
    sparse_sweep(&name2, &bytes2, &expected2, "PRE-M1-CHAINED");
}
