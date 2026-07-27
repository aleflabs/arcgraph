//! v2 M2 — G5, the RULE-MT differential gate (build-plan §2 M2 EXIT
//! 5): **"M1 JSON store vs M2 typed store, identical materialized
//! values for EVERY projection"** under ≥ 8-writer concurrency, on a
//! BOUNDED store with spill attached + small watermarks + GET-refault
//! readers — the M0.x/M1-REJECT-2 lesson baked in: the bounded-tier
//! actors MUST be present (an unbounded-store MT gate is structurally
//! blind to the evict/refault interleavings production runs by
//! default). Live AND post-recovery legs.
//!
//! # Shape
//!
//! **Part A (live, bounded + refault):** two `BlobStore::with_bound`
//! stores — J carries the M1 JSON encoding, T the M2 typed encoding of
//! the SAME deterministic logical bags — with ZERO watermarks (drain
//! on every publish = the maximum-duty eviction regime, skeptic5's
//! posture). 8 writer threads stage+publish into BOTH stores in
//! lockstep; 4 reader threads per store hammer the acked prefix
//! through the PRODUCTION mcp decode with ROTATING projections
//! (full bag / K=1 / K=2 / absent key), asserting J-vs-T value
//! equality on every read while pages evict + refault underneath.
//!
//! **Part B (post-recovery):** two REAL WAL stacks (the M1 headline
//! gate's harness) take the same logical bags through the production
//! `create_node` + `commit` path — J as `PropertyData::Blob(json)`,
//! T as `PropertyData::TypedBlock` — under 8 concurrent writer
//! threads, then BOTH recover via `recover_from_wal` and EVERY node's
//! materialized bag is compared J-vs-T under every projection shape.
//!
//! # A4 regime (L1 review — "the gate ran a different regime than
//! production")
//!
//! Both parts run **intern-logging ON** (`build_typed_bag` receives a
//! real `WalHandle`, exactly like the production write path), and Part
//! B's typed stack recovers into a **FRESH `InternTable`** wired via
//! `PageStoreTarget::with_intern_table` — the pre-fix gate passed
//! `None` (logging off) and reused the original populated table on
//! recovery, which made it structurally blind to the A4 intern
//! durability race (a committed typed block referencing an id with no
//! durable `InternString` record). Post-fix, every typed read below
//! resolves key names through the RECOVERED table only; a missing
//! binding fails loudly (`unknown interned key_id`).
//!
//! RED-on-revert: any typed-vs-JSON materialization divergence — a
//! mis-decoded tag, an ULP float drift (the `m1_float_fidelity` pin),
//! a lost overflow value, a refault mis-classification — fails the
//! equality on the exact (record, projection) that diverged. Reverting
//! the A4 durable-logged-set fix fails Part B's typed leg at the first
//! fresh-table key resolution.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_mcp::storage::property_payload::{
    ResolvedProjection, build_typed_bag, record_property_bag_checked, record_property_bag_projected,
};
use arcgraph_query::executor::value::Value;
use arcgraph_storage::blob::{BlobBoundConfig, BlobSpill, BlobStore};
use arcgraph_storage::intern::InternTable;
use arcgraph_storage::prop_block::patch_overflow_tail;
use arcgraph_storage::property::{BlobRef, encode_overflow_node};
use arcgraph_storage::wal::{WalConfig, WalWriter};
use tempfile::tempdir;

/// Shared WAL config (Part A spawns a writer for the A4 intern-logging
/// regime; Part B's stacks reuse it via `super::`).
fn wal_config(dir: &std::path::Path) -> WalConfig {
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

const WRITERS: usize = 8;
const READERS_PER_STORE: usize = 2; // × 2 stores = 4 reader threads
const BAGS_PER_WRITER: u64 = 160;

/// Deterministic logical bag for (writer, seq) — mixed scalar shapes
/// incl. a float (the ULP-fidelity surface), a bool, an int, strings,
/// and every 16th bag a LARGE value (the overflow/chain surface on
/// both representations).
fn logical_bag(writer: u64, seq: u64) -> Vec<(String, Value)> {
    let mut bag = vec![
        (
            "incident".to_string(),
            Value::String(format!("inc-{writer:02}-{seq:05}")),
        ),
        ("open".to_string(), Value::Boolean(seq % 3 == 0)),
        ("attempt".to_string(), Value::Integer(seq as i64 * 37 - 11)),
        (
            "score".to_string(),
            Value::Float((seq as f64) * 0.31 + writer as f64 * 1e-7),
        ),
        ("sev".to_string(), Value::String(format!("P{}", seq % 4))),
    ];
    if seq % 16 == 9 {
        bag.push((
            "dump".to_string(),
            Value::String("d".repeat(300 + (seq as usize % 5) * 111)),
        ));
    }
    bag
}

/// The M1 JSON encoding of the logical bag (the production pre-M2
/// wire: `Value::to_json_value` per cell, canonical object).
fn json_encoding(bag: &[(String, Value)]) -> Vec<u8> {
    let mut m = serde_json::Map::new();
    for (k, v) in bag {
        m.insert(k.clone(), v.to_json_value());
    }
    serde_json::to_vec(&serde_json::Value::Object(m)).expect("json encode")
}

fn bounded_store(dir: &std::path::Path) -> Arc<BlobStore> {
    let spill = Arc::new(BlobSpill::open(dir).unwrap());
    Arc::new(BlobStore::with_bound(
        spill,
        BlobBoundConfig {
            high_watermark_bytes: 0, // drain on every publish
            low_watermark_bytes: 0,  // evict every durable page
        },
    ))
}

/// One acked logical record: the (writer, seq) identity + each store's
/// staged ref.
#[derive(Clone, Copy)]
struct AckedPair {
    writer: u64,
    seq: u64,
    json_ref: BlobRef,
    typed_ref: BlobRef,
}

/// Decode via the production mcp path from raw refs (fixture records
/// pointing at each payload).
fn decode_via_mcp(
    store: &BlobStore,
    intern: &InternTable,
    bref: BlobRef,
    projection: Option<&ResolvedProjection>,
) -> std::collections::BTreeMap<String, Value> {
    let mut rec =
        arcgraph_core::NodeRecord::new(NodeId::new(1), LabelId::new(1), arcgraph_core::Lsn::new(1));
    encode_overflow_node(bref, &mut rec);
    match projection {
        None => record_property_bag_checked(&rec, store, intern, TenantId::DEFAULT)
            .expect("checked read"),
        Some(p) => record_property_bag_projected(&rec, store, intern, TenantId::DEFAULT, p)
            .expect("projected read"),
    }
}

#[test]
fn g5_live_bounded_refault_differential_every_projection() {
    let jdir = tempdir().unwrap();
    let tdir = tempdir().unwrap();
    let jstore = bounded_store(jdir.path());
    let tstore = bounded_store(tdir.path());
    let intern = Arc::new(InternTable::new());
    let tenant = TenantId::DEFAULT;

    // A4 regime: intern-logging ON — the typed encoders below run the
    // PRODUCTION logged-intern path (8 writers racing `intern_logged`
    // on the shared key names through a real WAL), not the `None`
    // in-memory shortcut the pre-fix gate ran.
    let wal_dir = tempdir().unwrap();
    let wal_writer = WalWriter::spawn(wal_config(wal_dir.path())).expect("spawn intern WAL");
    let wal_handle = wal_writer.handle();

    let stop = Arc::new(AtomicBool::new(false));
    let acked: Arc<Mutex<Vec<AckedPair>>> = Arc::new(Mutex::new(Vec::new()));
    let failures: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // ── Checkpoint-capture markers (one per store): the ADR-229
    //    freeze-side reader that flips INV-DURABLE bits — WITHOUT this
    //    actor nothing is ever evict-eligible and the whole run is the
    //    unbounded regime RULE-MT forbids (the skeptic5 harness's
    //    marker, reproduced).
    let markers: Vec<_> = [Arc::clone(&jstore), Arc::clone(&tstore)]
        .into_iter()
        .map(|store| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut n = 0u64;
                while !stop.load(Ordering::Acquire) {
                    let (resident, _evicted) = store.iter_pages_resident_only();
                    drop(resident);
                    n += 1;
                    std::thread::yield_now();
                }
                n
            })
        })
        .collect();

    // ── 8 writers: stage + publish the SAME logical bag into BOTH
    //    stores (disjoint txn ids per (writer, store)), ack the pair.
    let mut writers = Vec::new();
    for w in 0..WRITERS as u64 {
        let jstore = Arc::clone(&jstore);
        let tstore = Arc::clone(&tstore);
        let intern = Arc::clone(&intern);
        let acked = Arc::clone(&acked);
        let wal_handle = wal_handle.clone();
        writers.push(std::thread::spawn(move || {
            for seq in 0..BAGS_PER_WRITER {
                let bag = logical_bag(w, seq);
                let txn = w * 1_000_000 + seq + 1;

                // J: the M1 JSON write shape.
                let jbytes = json_encoding(&bag);
                let (jref, _) = jstore.stage_bag(tenant, txn, &jbytes).expect("stage json");
                jstore.publish_txn_slotted(txn).unwrap();

                // T: the M2 typed write shape (overflow staged first,
                // tail patched — the production stager's flow), with
                // intern-logging ON (the A4 production regime).
                let parts = build_typed_bag(
                    bag.iter().map(|(k, v)| (k.as_str(), v)),
                    &intern,
                    Some(&wal_handle),
                    tenant,
                )
                .expect("typed encode")
                .expect("non-empty");
                let mut block = parts.block;
                if let Some(of) = &parts.overflow {
                    let (oref, _) = tstore.stage_bag(tenant, txn, of).expect("stage overflow");
                    patch_overflow_tail(&mut block, oref).expect("patch tail");
                }
                let (tref, _) = tstore.stage_bag(tenant, txn, &block).expect("stage block");
                tstore.publish_txn_slotted(txn).unwrap();

                acked.lock().unwrap().push(AckedPair {
                    writer: w,
                    seq,
                    json_ref: jref,
                    typed_ref: tref,
                });
            }
        }));
    }

    // ── Refault readers: hammer the acked prefix through the mcp
    //    decode with rotating projections; assert J == T every read.
    let mut readers = Vec::new();
    for r in 0..(READERS_PER_STORE * 2) {
        let jstore = Arc::clone(&jstore);
        let tstore = Arc::clone(&tstore);
        let intern = Arc::clone(&intern);
        let stop = Arc::clone(&stop);
        let acked = Arc::clone(&acked);
        let failures = Arc::clone(&failures);
        readers.push(std::thread::spawn(move || {
            let projections: [Option<Vec<String>>; 4] = [
                None,                                              // full bag
                Some(vec!["score".to_string()]),                   // K=1 (the float)
                Some(vec!["sev".to_string(), "dump".to_string()]), // K=2 incl. overflow
                Some(vec!["nope_never_written".to_string()]),      // absent key
            ];
            let mut reads = 0u64;
            let mut i = r; // stagger start
            while !stop.load(Ordering::Acquire) {
                let snap: Vec<AckedPair> = {
                    let g = acked.lock().unwrap();
                    let start = g.len().saturating_sub(64);
                    g[start..].to_vec()
                };
                if snap.is_empty() {
                    std::thread::yield_now();
                    continue;
                }
                for pair in &snap {
                    let proj_names = &projections[i % projections.len()];
                    i += 1;
                    let proj = proj_names
                        .as_ref()
                        .map(|names| ResolvedProjection::resolve(names, &intern, tenant).unwrap());
                    let j = decode_via_mcp(&jstore, &intern, pair.json_ref, proj.as_ref());
                    let t = decode_via_mcp(&tstore, &intern, pair.typed_ref, proj.as_ref());
                    if j != t {
                        failures.lock().unwrap().push(format!(
                            "DIVERGENCE (live) writer={} seq={} projection={proj_names:?}: \
                             J={j:?} T={t:?}",
                            pair.writer, pair.seq
                        ));
                        stop.store(true, Ordering::Release);
                        return reads;
                    }
                    reads += 1;
                }
            }
            reads
        }));
    }

    for w in writers {
        w.join().expect("writer panicked");
    }
    wal_writer.shutdown().expect("intern WAL shutdown");
    // Let the readers sweep the FULL acked set once more, post-write.
    std::thread::sleep(std::time::Duration::from_millis(300));
    stop.store(true, Ordering::Release);
    let total_reads: u64 = readers
        .into_iter()
        .map(|r| r.join().expect("reader panicked"))
        .sum();
    let marker_runs: u64 = markers
        .into_iter()
        .map(|m| m.join().expect("marker panicked"))
        .sum();
    assert!(marker_runs > 0, "the capture markers must have run");

    let fails = failures.lock().unwrap();
    assert!(
        fails.is_empty(),
        "differential failures:\n{}",
        fails.join("\n")
    );
    assert!(
        total_reads > 10_000,
        "the refault readers must have exercised the regime (reads = {total_reads})"
    );
    assert!(
        jstore.evicted_count() > 0 && tstore.evicted_count() > 0,
        "the bounded tier must have EVICTED on both stores (J = {}, T = {}) — \
         an unbounded-regime run is structurally blind (RULE-MT)",
        jstore.evicted_count(),
        tstore.evicted_count()
    );
    assert!(
        jstore.refault_count() > 0 && tstore.refault_count() > 0,
        "the readers must have REFAULTED evicted pages on both stores (J = {}, T = {})",
        jstore.refault_count(),
        tstore.refault_count()
    );

    // ── Post-drain full sweep: force a final evict cycle, then compare
    //    EVERY acked pair under EVERY projection through the refault
    //    path (the tier's durable round trip).
    jstore.force_drain_for_test().unwrap();
    tstore.force_drain_for_test().unwrap();
    let all = acked.lock().unwrap().clone();
    assert_eq!(all.len() as u64, WRITERS as u64 * BAGS_PER_WRITER);
    let projections: [Option<Vec<String>>; 4] = [
        None,
        Some(vec!["score".to_string()]),
        Some(vec!["sev".to_string(), "dump".to_string()]),
        Some(vec!["incident".to_string(), "attempt".to_string()]),
    ];
    for pair in &all {
        for proj_names in &projections {
            let proj = proj_names
                .as_ref()
                .map(|names| ResolvedProjection::resolve(names, &intern, tenant).unwrap());
            let j = decode_via_mcp(&jstore, &intern, pair.json_ref, proj.as_ref());
            let t = decode_via_mcp(&tstore, &intern, pair.typed_ref, proj.as_ref());
            assert_eq!(
                j, t,
                "post-drain divergence: writer={} seq={} projection={proj_names:?}",
                pair.writer, pair.seq
            );
        }
    }
}

// ─── Part B — post-recovery differential over REAL WAL stacks ────────

mod recovery {
    use arcgraph_storage::crud::{
        CrudStore, PropertyData, commit, create_node, crud_allocator_seed_handle,
        read_node_with_store,
    };
    use arcgraph_storage::page_alloc::PageAllocator;
    use arcgraph_storage::primary_index::PrimaryIndex;
    use arcgraph_storage::transaction::TxnManager;
    use arcgraph_storage::wal::{
        AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
        RecordPageStoreHandle, WalWriter, recover_from_wal,
    };

    use super::*;

    fn build_stack(dir: &std::path::Path) -> (WalWriter, Arc<TxnManager>, Arc<CrudStore>) {
        let writer = WalWriter::spawn(wal_config(dir)).unwrap();
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
        (writer, mgr, store)
    }

    /// Recovery-side stack. `intern` — when `Some` — is the FRESH
    /// [`InternTable`] the replay must reconstruct name bindings into
    /// (the A4 regime; the pre-fix gate reused the original populated
    /// table, hiding missing `InternString` records).
    fn recover_stack(
        dir: &std::path::Path,
        intern: Option<Arc<InternTable>>,
    ) -> (WalWriter, Arc<TxnManager>, Arc<CrudStore>) {
        let writer = WalWriter::spawn(wal_config(dir)).unwrap();
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
        let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
            Arc::clone(primary.page_store()) as Arc<dyn PrimaryPageStoreHandle>;
        let records_handle: Arc<dyn RecordPageStoreHandle> = Arc::clone(
            store
                .records()
                .expect("dual-write stack has a record store"),
        )
            as Arc<dyn RecordPageStoreHandle>;
        let blob_handle: Arc<dyn BlobStoreHandle> =
            Arc::clone(store.blob_store()) as Arc<dyn BlobStoreHandle>;
        let allocator_seed: Arc<dyn AllocatorSeedHandle> =
            crud_allocator_seed_handle(Arc::clone(&store), Arc::clone(&alloc));
        let mut target = PageStoreTarget::primary_only(primary_handle)
            .with_record_store(records_handle)
            .with_blob_store(blob_handle)
            .with_allocator_seed(allocator_seed);
        if let Some(table) = intern {
            target = target.with_intern_table(table);
        }
        recover_from_wal(dir, Arc::clone(&mgr), target, None).expect("recovery");
        (writer, mgr, store)
    }

    /// 8 concurrent writers commit the same logical bags into a JSON
    /// stack and a TYPED stack through the PRODUCTION create+commit
    /// path; both recover from their WALs; every node's materialized
    /// bag must be value-identical J-vs-T under every projection.
    #[test]
    fn g5_post_recovery_differential_every_projection() {
        const PER_WRITER: u64 = 40;
        let tenant = TenantId::DEFAULT;
        let jdir = tempdir().unwrap();
        let tdir = tempdir().unwrap();
        let intern = Arc::new(InternTable::new());

        // Write phase — 8 writers per stack, concurrent.
        for (dir, typed) in [(&jdir, false), (&tdir, true)] {
            let (writer, mgr, store) = build_stack(dir.path());
            let mut hs = Vec::new();
            for w in 0..WRITERS as u64 {
                let mgr = Arc::clone(&mgr);
                let store = Arc::clone(&store);
                let intern = Arc::clone(&intern);
                hs.push(std::thread::spawn(move || {
                    for seq in 0..PER_WRITER {
                        let bag = logical_bag(w, seq);
                        let props = if typed {
                            // A4 regime: intern-logging ON through the
                            // stack's own WAL — the production write
                            // path's exact shape.
                            let parts = build_typed_bag(
                                bag.iter().map(|(k, v)| (k.as_str(), v)),
                                &intern,
                                store.wal(),
                                tenant,
                            )
                            .expect("typed encode")
                            .expect("non-empty");
                            PropertyData::TypedBlock(parts)
                        } else {
                            PropertyData::Blob(json_encoding(&bag))
                        };
                        let mut tx = mgr.begin(tenant);
                        create_node(&store, &mut tx, tenant, LabelId::new(1), &props)
                            .expect("create");
                        commit(tx, &store).expect("commit");
                    }
                }));
            }
            for h in hs {
                h.join().expect("writer panicked");
            }
            writer.shutdown().unwrap();
        }

        // Recover BOTH stacks; compare every node, every projection.
        // The typed stack recovers into a FRESH InternTable (the A4
        // regime): every key name below resolves through what the WAL
        // REPLAY reconstructed — never through the live table the
        // writers populated. A committed typed block whose intern
        // binding was not durably logged fails here loudly.
        let fresh_intern = Arc::new(InternTable::new());
        let (_jw, jmgr, jstore) = recover_stack(jdir.path(), None);
        let (_tw, tmgr, tstore) = recover_stack(tdir.path(), Some(Arc::clone(&fresh_intern)));
        let jtx = jmgr.begin(tenant);
        let ttx = tmgr.begin(tenant);
        let total = WRITERS as u64 * PER_WRITER;
        let projections: [Option<Vec<String>>; 4] = [
            None,
            Some(vec!["score".to_string()]),
            Some(vec!["sev".to_string(), "dump".to_string()]),
            Some(vec!["open".to_string(), "attempt".to_string()]),
        ];
        let mut compared = 0u64;
        for id in 1..=total {
            let jrec = read_node_with_store(&jstore, &jtx, NodeId::new(id))
                .expect("read")
                .expect("node exists (json stack)");
            let trec = read_node_with_store(&tstore, &ttx, NodeId::new(id))
                .expect("read")
                .expect("node exists (typed stack)");
            for proj_names in &projections {
                // Per-stack projection resolution: the typed stack's
                // key_ids come from the RECOVERED table (A4 regime).
                let jproj = proj_names
                    .as_ref()
                    .map(|names| ResolvedProjection::resolve(names, &intern, tenant).unwrap());
                let tproj = proj_names.as_ref().map(|names| {
                    ResolvedProjection::resolve(names, &fresh_intern, tenant).unwrap()
                });
                let j = match &jproj {
                    None => {
                        record_property_bag_checked(&jrec, jstore.blob_store(), &intern, tenant)
                            .expect("json read")
                    }
                    Some(p) => record_property_bag_projected(
                        &jrec,
                        jstore.blob_store(),
                        &intern,
                        tenant,
                        p,
                    )
                    .expect("json projected read"),
                };
                let t = match &tproj {
                    None => record_property_bag_checked(
                        &trec,
                        tstore.blob_store(),
                        &fresh_intern,
                        tenant,
                    )
                    .expect("typed read (fresh recovered intern table)"),
                    Some(p) => record_property_bag_projected(
                        &trec,
                        tstore.blob_store(),
                        &fresh_intern,
                        tenant,
                        p,
                    )
                    .expect("typed projected read (fresh recovered intern table)"),
                };
                // NB: node ids across the two stacks may map to
                // DIFFERENT (writer, seq) bags (concurrent id
                // allocation order differs) — the differential
                // therefore compares the DECODED bag against the
                // derived oracle for ITS OWN identity, then the two
                // oracles' equality closes the loop.
                let j_full =
                    record_property_bag_checked(&jrec, jstore.blob_store(), &intern, tenant)
                        .expect("json id read");
                let t_full =
                    record_property_bag_checked(&trec, tstore.blob_store(), &fresh_intern, tenant)
                        .expect("typed id read (fresh recovered intern table)");
                let j_id = j_full
                    .get("incident")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .expect("incident key");
                let t_id = t_full
                    .get("incident")
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .expect("incident key");
                // Derive (writer, seq) from the identity, recompute the
                // oracle bag, and assert each stack's read equals ITS
                // oracle under the projection.
                let parse = |s: &str| -> (u64, u64) {
                    let mut it = s.trim_start_matches("inc-").split('-');
                    (
                        it.next().unwrap().parse().unwrap(),
                        it.next().unwrap().parse().unwrap(),
                    )
                };
                let (jw_, js_) = parse(&j_id);
                let (tw_, ts_) = parse(&t_id);
                let oracle = |w: u64, s: u64, proj: &Option<ResolvedProjection>| {
                    let bag = logical_bag(w, s);
                    let jb = json_encoding(&bag);
                    let m: serde_json::Value = serde_json::from_slice(&jb).unwrap();
                    let obj = m.as_object().unwrap();
                    let mut out = std::collections::BTreeMap::new();
                    for (k, v) in obj {
                        if let Some(p) = proj {
                            if !p.entries.iter().any(|(n, _)| n == k) {
                                continue;
                            }
                        }
                        // #1444: the map-only stored-bag bridge.
                        out.insert(k.clone(), Value::try_from_json_property_value(v).unwrap());
                    }
                    out
                };
                assert_eq!(
                    j,
                    oracle(jw_, js_, &jproj),
                    "json stack node {id} vs oracle under {proj_names:?}"
                );
                assert_eq!(
                    t,
                    oracle(tw_, ts_, &tproj),
                    "typed stack node {id} vs oracle under {proj_names:?}"
                );
                compared += 1;
            }
        }
        assert_eq!(
            compared,
            total * 4,
            "every node × every projection compared"
        );
    }
}
