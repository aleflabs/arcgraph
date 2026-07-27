//! M3.a Slice G.4 — v5 CommitBundle codec boundary tests.
//!
//! Pinned tests for the production v5 wire format introduced by Slice
//! G.4 (commit-bundle vector page staging). Each test guards a
//! correctness invariant the prompt enumerates:
//!
//!   1. proptest_v5_round_trip — random encode → decode → byte-identity
//!      across `(commit_lsn, mvcc, staged, allocator, vector_pages)`,
//!      1024+ cases.
//!   2. v5_reader_decodes_v4_bundle_with_empty_vector_pages — backward
//!      compat: v4 bytes through the v5-aware dispatcher synthesize
//!      `vector_pages: Vec::new()`.
//!   3. v4_reader_rejects_v5_bundle — forward compat: a v4-only reader
//!      fed v5 bytes returns `WalCorruption` (the trailing
//!      `n_vector_pages` u32 + entries form unparsed trailing bytes
//!      from v4's perspective).
//!   4. v5_multi_tenant_4_tenants_no_cross_tenant_leak — encode +
//!      replay-apply 4 tenants' worth of vector pages; assert each
//!      tenant's pages land in the correct logical bucket via the
//!      `VectorPageStoreHandle` abstraction.
//!   5. v5_mixed_sections_apply_in_lemma_i3_order — record per-store
//!      apply timeline; assert staged → vector → allocator order
//!      (Lemma I3).
//!   6. v5_empty_vector_pages_round_trips — `vector_pages: &[]` shape
//!      identical to v4 prefix + `0u32` trailer.
//!   7. v5_truncated_vector_section_returns_truncated_bundle_error —
//!      truncate mid-vector section; decode surfaces `WalCorruption`.
//!
//! Plus regression guards:
//!   - vector_page_entry_partition_id_always_zero_at_v1
//!   - vector_page_entry_index_id_always_zero_at_v1
//!   - vector_page_entry_encoded_len_is_pinned
//!
//! Per ADR-031 amendment-02 + ADR-035 §4.5/§4.6 + issue #131 item 3
//! + PR #130 (v4 codec / AllocatorAdvance precedent).

use std::collections::HashMap;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use arcgraph_core::{
    ArcGraphError, Lsn, PAGE_SIZE, PageId, PartitionId, Result as ArcResult, TenantId,
};
use arcgraph_storage::vector_store::{VectorPageStoreHandle, VectorStoreError};
use arcgraph_storage::wal::bundle::{
    AllocatorAdvance, AllocatorKind, BUNDLE_FORMAT_V4, BUNDLE_FORMAT_V5, BundlePageKind,
    DecodedCommitBundle, SideChannelWrite, VectorPageEntry, decode_commit_bundle_for_version,
    decode_commit_bundle_v4, decode_commit_bundle_v5, encode_commit_bundle_v4,
    encode_commit_bundle_v5, encode_commit_bundle_v8,
};
use bytes::Bytes;
use proptest::prelude::*;

// ─── Helpers ────────────────────────────────────────────────────

fn mk_page_bytes(fill: u8) -> Box<[u8; PAGE_SIZE]> {
    Box::new([fill; PAGE_SIZE])
}

fn mk_staged(
    kind: BundlePageKind,
    page_id: u64,
    tenant: TenantId,
    fill: u8,
) -> (BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>) {
    (kind, PageId::new(page_id), tenant, mk_page_bytes(fill))
}

fn mk_vec_entry(tenant_raw: u64, page_id: u64, commit_lsn: u64, fill: u8) -> VectorPageEntry {
    VectorPageEntry {
        tenant: TenantId::new(tenant_raw),
        partition: PartitionId::ZERO,
        index_id: 0,
        page_id: PageId::new(page_id),
        commit_lsn: Lsn::new(commit_lsn),
        bytes: mk_page_bytes(fill),
    }
}

// ─── Test 1: proptest round-trip (≥ 1024 cases) ──────────────────

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1024,
        ..ProptestConfig::default()
    })]

    #[test]
    fn proptest_v5_round_trip(
        commit_lsn_raw in 0u64..u64::MAX,
        mvcc_kvs in proptest::collection::vec(
            (any::<u64>(), proptest::option::of(proptest::collection::vec(any::<u8>(), 0..32))),
            0..6,
        ),
        sc_count in 0usize..3,
        n_staged in 0usize..3,
        n_advances in 0usize..3,
        n_vector in 0usize..3,
        vec_fill in any::<u8>(),
        staged_fill in any::<u8>(),
    ) {
        // Build the input materials.
        let primary: HashMap<u64, Option<Bytes>> = mvcc_kvs
            .into_iter()
            .map(|(k, v)| (k, v.map(Bytes::from)))
            .collect();

        let sidechannel: Vec<SideChannelWrite> = (0..sc_count)
            .map(|i| SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: 1_000 + i as u64,
                value: Some(Bytes::from(format!("sc-{i}").into_bytes())),
            })
            .collect();

        let staged: Vec<(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)> = (0..n_staged)
            .map(|i| {
                mk_staged(
                    BundlePageKind::Record,
                    100 + i as u64,
                    TenantId::DEFAULT,
                    staged_fill,
                )
            })
            .collect();

        let advances: Vec<AllocatorAdvance> = (0..n_advances)
            .map(|i| AllocatorAdvance {
                tenant: TenantId::DEFAULT,
                kind: AllocatorKind::Node,
                new_high_water: 7 + i as u64,
            })
            .collect();

        let vector_pages: Vec<VectorPageEntry> = (0..n_vector)
            .map(|i| mk_vec_entry(
                TenantId::DEFAULT.raw(),
                200 + i as u64,
                commit_lsn_raw.wrapping_add(i as u64),
                vec_fill,
            ))
            .collect();

        // Encode + decode + structural-equality on every field.
        let encoded = encode_commit_bundle_v5(
            Lsn::new(commit_lsn_raw),
            TenantId::DEFAULT,
            &primary,
            &sidechannel,
            &staged,
            &advances,
            &vector_pages,
        );
        let decoded = decode_commit_bundle_v5(&encoded, TenantId::DEFAULT)
            .expect("v5 round-trip must decode cleanly");

        // commit_lsn round-trips.
        prop_assert_eq!(decoded.commit_lsn.raw(), commit_lsn_raw);
        // primary mvcc set is preserved (HashMap equality).
        prop_assert_eq!(decoded.mvcc_writes.len(), primary.len());
        for (k, v) in &primary {
            prop_assert_eq!(decoded.mvcc_writes.get(k), Some(v));
        }
        // sidechannel preserved (count + per-entry tenant/key/value
        // equality). O-G (W28-S3): was count-only, so blind to a codec
        // that preserved arity but corrupted any entry's tenant/key/
        // value — mirrors the `mvcc_writes` membership check above
        // (`SideChannelWrite: PartialEq`; decode returns `(tenant,key)`
        // sorted order, but membership is robust regardless).
        prop_assert_eq!(decoded.sidechannel_writes.len(), sidechannel.len());
        for sc in &sidechannel {
            prop_assert!(
                decoded.sidechannel_writes.iter().any(|d| d == sc),
                "decoded sidechannel_writes missing entry (tenant {:?}, key {})",
                sc.tenant_id,
                sc.key
            );
        }
        // staged_pages preserved (count + each entry's page_id + FULL
        // byte-equality across the whole PAGE_SIZE body). O-A (W28-S3):
        // was `bytes[0]`-only, so blind to a codec that preserved the
        // first byte but corrupted/truncated the rest of the page —
        // mirrors the `vector_pages` full-`bytes` check below.
        prop_assert_eq!(decoded.staged_pages.len(), staged.len());
        for (i, p) in decoded.staged_pages.iter().enumerate() {
            prop_assert_eq!(p.page_id, staged[i].1);
            prop_assert!(
                p.bytes[..] == staged[i].3[..],
                "staged page {} (page_id {:?}) byte mismatch (O-A full byte-equality)",
                i,
                p.page_id
            );
        }
        // allocator_advances preserved (multiset equality).
        prop_assert_eq!(decoded.allocator_advances.len(), advances.len());
        // vector_pages preserved (multiset equality).
        prop_assert_eq!(decoded.vector_pages.len(), vector_pages.len());
        for entry in &vector_pages {
            // Entry must be present (encoder may sort, so check by
            // membership, not index).
            prop_assert!(
                decoded.vector_pages.iter().any(|e| {
                    e.tenant == entry.tenant
                        && e.page_id == entry.page_id
                        && e.commit_lsn == entry.commit_lsn
                        && e.bytes == entry.bytes
                }),
                "decoded vector_pages missing entry at page {:?}",
                entry.page_id
            );
        }

        // Re-encode the decoded bundle as v5; must be byte-identical
        // (the encoder's deterministic sort is the canonicalization).
        let staged_for_reencode: Vec<(BundlePageKind, PageId, TenantId, Box<[u8; PAGE_SIZE]>)> =
            decoded
                .staged_pages
                .iter()
                .map(|p| (p.kind, p.page_id, p.tenant_id, p.bytes.clone()))
                .collect();
        let re_encoded = encode_commit_bundle_v5(
            decoded.commit_lsn,
            decoded.primary_tenant,
            &decoded.mvcc_writes,
            &decoded.sidechannel_writes,
            &staged_for_reencode,
            &decoded.allocator_advances,
            &decoded.vector_pages,
        );
        prop_assert_eq!(
            re_encoded, encoded,
            "v5 encoder must be deterministic for fixed inputs"
        );
    }
}

// ─── Test 2: v5 reader on v4 bytes — empty vector_pages ──────────

#[test]
fn v5_reader_decodes_v4_bundle_with_empty_vector_pages() {
    // A v4 bundle with all sections populated.
    let primary: HashMap<u64, Option<Bytes>> = [(1u64, Some(Bytes::from_static(b"v4-data")))]
        .into_iter()
        .collect();
    let sidechannel = vec![SideChannelWrite {
        tenant_id: TenantId::SYSTEM,
        key: 99,
        value: Some(Bytes::from_static(b"root-ptr")),
    }];
    let staged = vec![mk_staged(
        BundlePageKind::Record,
        200,
        TenantId::DEFAULT,
        0xCC,
    )];
    let advances = vec![AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::PageNode,
        new_high_water: 17,
    }];

    let v4_bytes = encode_commit_bundle_v4(
        Lsn::new(42),
        TenantId::DEFAULT,
        &primary,
        &sidechannel,
        &staged,
        &advances,
    );

    // Route v4 bytes through the version-aware dispatcher with
    // BUNDLE_FORMAT_V4. Decoder synthesizes vector_pages: Vec::new()
    // for back-compat — pins the v5 reader's "empty vector_pages on
    // legacy bundle" contract.
    let decoded =
        decode_commit_bundle_for_version(&v4_bytes, BUNDLE_FORMAT_V4, TenantId::DEFAULT).unwrap();
    assert!(
        decoded.vector_pages.is_empty(),
        "v5 reader on v4 bytes must synthesize empty vector_pages; \
         got {} entries",
        decoded.vector_pages.len()
    );
    // Other fields intact.
    assert_eq!(decoded.commit_lsn, Lsn::new(42));
    assert_eq!(decoded.mvcc_writes.len(), 1);
    assert_eq!(decoded.sidechannel_writes.len(), 1);
    assert_eq!(decoded.staged_pages.len(), 1);
    assert_eq!(decoded.allocator_advances.len(), 1);
}

// ─── Test 3: v4 reader rejects v5 bytes (forward-compat) ─────────

#[test]
fn v4_reader_rejects_v5_bundle() {
    // Build a v5 bundle with non-empty vector_pages so the trailing
    // section is real (not just a `0u32` trailer that v4 would
    // tolerate as "trailing bytes after decode").
    let vector_pages = vec![mk_vec_entry(TenantId::DEFAULT.raw(), 1, 100, 0xAB)];
    let v5_bytes = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &vector_pages,
    );

    // A v4-only reader (decode_commit_bundle_v4 directly) on v5 bytes
    // surfaces WalCorruption: the v4 decoder's strict trailing-bytes
    // check rejects the trailing `n_vector_pages` u32 + entries.
    let err = decode_commit_bundle_v4(&v5_bytes, TenantId::DEFAULT).unwrap_err();
    assert!(
        matches!(err, ArcGraphError::WalCorruption { .. }),
        "v4 reader on v5 bytes must surface WalCorruption (trailing \
         bytes from v5 vector_pages section); got {err:?}"
    );
}

// ─── Test 4: 4 tenants — no cross-tenant leak ─────────────────────

/// Recording mock: stores every install_or_replace call in a vec
/// keyed by (tenant, page_id) so the test can assert per-tenant
/// fidelity post-replay.
#[derive(Default)]
struct RecordingVectorStore {
    calls: StdMutex<Vec<(TenantId, PageId, Vec<u8>)>>,
}

impl VectorPageStoreHandle for RecordingVectorStore {
    fn install_or_replace(
        &self,
        tenant: TenantId,
        page_id: PageId,
        bytes: &[u8],
    ) -> std::result::Result<(), VectorStoreError> {
        self.calls
            .lock()
            .unwrap()
            .push((tenant, page_id, bytes.to_vec()));
        Ok(())
    }
    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> std::result::Result<(), VectorStoreError> {
        Ok(())
    }
}

#[test]
fn v5_multi_tenant_4_tenants_no_cross_tenant_leak() {
    // 4 distinct tenants with 2 pages each = 8 entries.
    let tenants: Vec<TenantId> = (1..=4u64).map(TenantId::new).collect();
    let mut vector_pages: Vec<VectorPageEntry> = Vec::new();
    for (t_idx, tenant) in tenants.iter().enumerate() {
        for p in 0..2u64 {
            vector_pages.push(VectorPageEntry {
                tenant: *tenant,
                partition: PartitionId::ZERO,
                index_id: 0,
                page_id: PageId::new(1_000 * (t_idx as u64 + 1) + p),
                commit_lsn: Lsn::new(p + 1),
                // Per-tenant fill so a cross-tenant leak surfaces as a
                // bytes-mismatch.
                bytes: mk_page_bytes(0xA0u8 + t_idx as u8),
            });
        }
    }

    // Encode + decode.
    let encoded = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &vector_pages,
    );
    let decoded = decode_commit_bundle_v5(&encoded, TenantId::DEFAULT).unwrap();
    assert_eq!(decoded.vector_pages.len(), 8);

    // Replay-simulate: route every entry through a recording mock,
    // then assert per-tenant page count + bytes match exactly.
    let recorder = RecordingVectorStore::default();
    for entry in &decoded.vector_pages {
        recorder
            .install_or_replace(entry.tenant, entry.page_id, entry.bytes.as_ref())
            .unwrap();
    }
    let calls = recorder.calls.lock().unwrap();
    assert_eq!(calls.len(), 8);

    for (t_idx, tenant) in tenants.iter().enumerate() {
        let per_tenant: Vec<&(TenantId, PageId, Vec<u8>)> =
            calls.iter().filter(|(t, _, _)| *t == *tenant).collect();
        assert_eq!(
            per_tenant.len(),
            2,
            "tenant {:?} should have exactly 2 vector_pages; got {}",
            tenant,
            per_tenant.len()
        );
        for (_, _, bytes) in per_tenant {
            // Every byte must be the per-tenant fill — a cross-tenant
            // leak would put a different tenant's fill here.
            assert!(
                bytes.iter().all(|b| *b == 0xA0u8 + t_idx as u8),
                "tenant {tenant:?} page bytes corrupted (cross-tenant leak)"
            );
        }
    }
}

// ─── Test 5: Lemma I3 apply order — staged → vector → allocator ──

/// Recording mock for staged_pages installs. Records the order the
/// install_or_replace was called via a shared atomic clock.
struct OrderedStagedRecorder {
    clock: AtomicUsize,
    calls: StdMutex<Vec<usize>>,
}

impl arcgraph_storage::wal::PrimaryPageStoreHandle for OrderedStagedRecorder {
    fn install_or_replace(&self, _page_id: PageId, _page: Box<[u8; PAGE_SIZE]>) -> ArcResult<()> {
        let t = self.clock.fetch_add(1, Ordering::SeqCst);
        self.calls.lock().unwrap().push(t);
        Ok(())
    }
    fn contains(&self, _page_id: PageId) -> bool {
        false
    }
}

struct OrderedVectorRecorder {
    clock: std::sync::Arc<AtomicUsize>,
    calls: StdMutex<Vec<usize>>,
}

impl VectorPageStoreHandle for OrderedVectorRecorder {
    fn install_or_replace(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> std::result::Result<(), VectorStoreError> {
        let t = self.clock.fetch_add(1, Ordering::SeqCst);
        self.calls.lock().unwrap().push(t);
        Ok(())
    }
    fn restore_page_bytes(
        &self,
        _tenant: TenantId,
        _page_id: PageId,
        _bytes: &[u8],
    ) -> std::result::Result<(), VectorStoreError> {
        Ok(())
    }
}

struct OrderedAllocSeed {
    clock: std::sync::Arc<AtomicUsize>,
    calls: StdMutex<Vec<usize>>,
}

impl arcgraph_storage::wal::AllocatorSeedHandle for OrderedAllocSeed {
    fn seed_from_advance(&self, _advance: AllocatorAdvance) {
        let t = self.clock.fetch_add(1, Ordering::SeqCst);
        self.calls.lock().unwrap().push(t);
    }
}

#[test]
fn v5_mixed_sections_apply_in_lemma_i3_order() {
    use std::sync::Arc;

    use arcgraph_storage::transaction::TxnManager;
    use arcgraph_storage::wal::{
        AllocatorSeedHandle, PageStoreTarget, PrimaryPageStoreHandle, ReplayConfig, ReplayExecutor,
        WalConfig, WalRecordType, WalRecoveryReader, WalWriter,
    };

    // Shared monotone clock — every install bumps it. The post-replay
    // call sequences land in the recorders in the order they fired,
    // and we read off "first staged tick < first vector tick < first
    // allocator tick" to pin Lemma I3.
    let clock = Arc::new(AtomicUsize::new(0));
    let staged_clock = clock.clone();
    let vector_clock = clock.clone();
    let alloc_clock = clock.clone();

    let staged_recorder = Arc::new(OrderedStagedRecorder {
        clock: AtomicUsize::new(0), // local relay; real clock used from impl
        calls: StdMutex::new(Vec::new()),
    });
    // We can't easily share the global clock through the trait
    // because the impl owns its own clock. Build a wrapper that uses
    // the shared clock instead.
    struct SharedStaged {
        shared: Arc<AtomicUsize>,
        calls: StdMutex<Vec<usize>>,
    }
    impl PrimaryPageStoreHandle for SharedStaged {
        fn install_or_replace(
            &self,
            _page_id: PageId,
            _page: Box<[u8; PAGE_SIZE]>,
        ) -> ArcResult<()> {
            let t = self.shared.fetch_add(1, Ordering::SeqCst);
            self.calls.lock().unwrap().push(t);
            Ok(())
        }
        fn contains(&self, _page_id: PageId) -> bool {
            false
        }
    }
    drop(staged_recorder); // unused above; we use SharedStaged below

    let staged = Arc::new(SharedStaged {
        shared: staged_clock.clone(),
        calls: StdMutex::new(Vec::new()),
    });
    let vector = Arc::new(OrderedVectorRecorder {
        clock: vector_clock.clone(),
        calls: StdMutex::new(Vec::new()),
    });
    let alloc = Arc::new(OrderedAllocSeed {
        clock: alloc_clock.clone(),
        calls: StdMutex::new(Vec::new()),
    });

    // Wire the WAL stack and append a single v5 bundle with 1 staged
    // record page + 1 vector page + 1 allocator advance.
    let dir = tempfile::tempdir().unwrap();
    let writer = WalWriter::spawn(WalConfig {
        dir: dir.path().to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    })
    .unwrap();
    let staged_v5 = vec![mk_staged(
        BundlePageKind::Record,
        7,
        TenantId::DEFAULT,
        0xAA,
    )];
    let advances = vec![AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::PageNode,
        new_high_water: 99,
    }];
    let vector_pages = vec![mk_vec_entry(TenantId::DEFAULT.raw(), 11, 1, 0xBB)];
    // #1221 (ADR-218): this is the only WAL-write-then-recover test in
    // this file; it must encode at the CURRENT bundle version (v8) so the
    // v8-stamped segment decodes cleanly through the executor. The
    // pure-codec round-trips elsewhere stay on v5/v6 (those codecs are
    // still supported for reading pre-upgrade segments).
    let payload = encode_commit_bundle_v8(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &staged_v5,
        &advances,
        &vector_pages,
        &[], // no idempotency bindings in this fixture
        &[], // #1221: no acl_grants in this fixture
    );
    writer
        .handle()
        .append(
            WalRecordType::CommitBundle,
            1,
            0,
            TenantId::DEFAULT,
            payload,
        )
        .unwrap();
    writer.shutdown().unwrap();

    // Build a target that wires every store into the shared clock.
    let txn_mgr = Arc::new(TxnManager::new());
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> = Arc::clone(&staged) as _;
    let vector_handle: Arc<dyn VectorPageStoreHandle> = Arc::clone(&vector) as _;
    let alloc_handle: Arc<dyn AllocatorSeedHandle> = Arc::clone(&alloc) as _;

    // The record-store handle is what BundlePageKind::Record routes to;
    // since we didn't wire a record store, the staged_pages dispatch
    // will reject the Record entry. Use PrimaryIndex kind instead so
    // the install routes through the primary handle.
    let staged_v5_primary = vec![mk_staged(
        BundlePageKind::PrimaryIndex,
        7,
        TenantId::DEFAULT,
        0xAA,
    )];
    let payload2 = encode_commit_bundle_v8(
        Lsn::new(2),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &staged_v5_primary,
        &advances,
        &vector_pages,
        &[], // no idempotency bindings in this fixture
        &[], // #1221: no acl_grants in this fixture
    );
    let dir2 = tempfile::tempdir().unwrap();
    let writer2 = WalWriter::spawn(WalConfig {
        dir: dir2.path().to_path_buf(),
        segment_size_bytes: 64 * 1024 * 1024,
        group_commit_window: std::time::Duration::from_millis(2),
        group_commit_max_batch: 4,
        metrics_sink: None,
        encryption: None,

        inflight_budget_bytes: None,
    })
    .unwrap();
    writer2
        .handle()
        .append(
            WalRecordType::CommitBundle,
            1,
            0,
            TenantId::DEFAULT,
            payload2,
        )
        .unwrap();
    writer2.shutdown().unwrap();

    let target = PageStoreTarget::primary_only(primary_handle)
        .with_vector_store(vector_handle)
        .with_allocator_seed(alloc_handle);
    let reader = WalRecoveryReader::open(dir2.path()).unwrap();
    let mut exec = ReplayExecutor::new(
        ReplayConfig::default_with_temp_spill(),
        Arc::clone(&txn_mgr),
        target,
    );
    let _high = exec.run(reader).unwrap();

    // Read the per-store call ticks. Lemma I3 says
    // staged_pages → vector_pages → allocator_advances.
    let staged_ticks = staged.calls.lock().unwrap().clone();
    let vector_ticks = vector.calls.lock().unwrap().clone();
    let alloc_ticks = alloc.calls.lock().unwrap().clone();
    assert_eq!(staged_ticks.len(), 1, "expected 1 staged install");
    assert_eq!(vector_ticks.len(), 1, "expected 1 vector install");
    assert_eq!(alloc_ticks.len(), 1, "expected 1 allocator seed");
    assert!(
        staged_ticks[0] < vector_ticks[0],
        "Lemma I3 (a): staged_pages must apply before vector_pages; \
         staged_tick={} vector_tick={}",
        staged_ticks[0],
        vector_ticks[0]
    );
    assert!(
        vector_ticks[0] < alloc_ticks[0],
        "Lemma I3 (b): vector_pages must apply before allocator_advances; \
         vector_tick={} alloc_tick={}",
        vector_ticks[0],
        alloc_ticks[0]
    );
}

// ─── Test 6: empty vector_pages — wire shape pin ─────────────────

#[test]
fn v5_empty_vector_pages_round_trips() {
    // v5 with EMPTY vector_pages MUST equal:
    //   v4-equivalent prefix + 4-byte `n_vector_pages = 0` trailer.
    // This is the on-wire pin for the v4 → v5 backward-compat
    // contract: "v4 bundle is a v5 prefix; v5 prefix bytes match v4
    // exactly".
    let primary: HashMap<u64, Option<Bytes>> = [(1u64, Some(Bytes::from_static(b"data")))]
        .into_iter()
        .collect();
    let staged = vec![mk_staged(
        BundlePageKind::PrimaryIndex,
        100,
        TenantId::SYSTEM,
        0xAB,
    )];
    let advances = vec![AllocatorAdvance {
        tenant: TenantId::DEFAULT,
        kind: AllocatorKind::Node,
        new_high_water: 7,
    }];

    let v4 = encode_commit_bundle_v4(
        Lsn::new(42),
        TenantId::DEFAULT,
        &primary,
        &[],
        &staged,
        &advances,
    );
    let v5 = encode_commit_bundle_v5(
        Lsn::new(42),
        TenantId::DEFAULT,
        &primary,
        &[],
        &staged,
        &advances,
        &[],
    );

    // v5 bytes = v4 bytes + 4-byte zero trailer.
    assert_eq!(
        v5.len(),
        v4.len() + 4,
        "v5 with empty vector_pages must add exactly 4 bytes"
    );
    assert_eq!(&v5[..v4.len()], &v4[..], "v5 prefix MUST equal v4 bytes");
    assert_eq!(
        &v5[v4.len()..],
        &0u32.to_le_bytes(),
        "v5 trailer MUST be n_vector_pages=0"
    );

    // And decode round-trips with empty vector_pages.
    let decoded = decode_commit_bundle_v5(&v5, TenantId::DEFAULT).unwrap();
    assert!(decoded.vector_pages.is_empty());
    assert_eq!(decoded.commit_lsn, Lsn::new(42));
    assert_eq!(decoded.staged_pages.len(), 1);
    assert_eq!(decoded.allocator_advances.len(), 1);
}

// ─── Test 7: truncated vector_pages section ──────────────────────

#[test]
fn v5_truncated_vector_section_returns_truncated_bundle_error() {
    // Sub-test A: truncate after `n_vector_pages` but before any
    // entry — decoder needs to read the first entry's tenant_id u64
    // and overruns.
    let vector_pages = vec![mk_vec_entry(TenantId::DEFAULT.raw(), 1, 1, 0xAA)];
    let v5 = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &vector_pages,
    );
    // n_vector_pages u32 sits at v4_prefix_len; cut right after it
    // (so 0 entries' worth of data remains for the declared count of
    // 1).
    let v4_prefix = encode_commit_bundle_v4(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
    );
    let cut_after_n_vec = v4_prefix.len() + 4;
    assert!(cut_after_n_vec < v5.len());
    let mut truncated_a = v5[..cut_after_n_vec].to_vec();
    // Restore n_vector_pages = 1 (we cut at exactly that boundary).
    let n_offset = v4_prefix.len();
    truncated_a[n_offset..n_offset + 4].copy_from_slice(&1u32.to_le_bytes());
    let err_a = decode_commit_bundle_v5(&truncated_a, TenantId::DEFAULT).unwrap_err();
    assert!(
        matches!(err_a, ArcGraphError::WalCorruption { .. }),
        "truncate-after-count must surface WalCorruption (mid-entry overrun); \
         got {err_a:?}"
    );

    // Sub-test B: truncate mid-entry — knock out the trailing page
    // bytes inside the only entry.
    let mut truncated_b = v5.clone();
    truncated_b.truncate(truncated_b.len() - 100);
    let err_b = decode_commit_bundle_v5(&truncated_b, TenantId::DEFAULT).unwrap_err();
    assert!(
        matches!(err_b, ArcGraphError::WalCorruption { .. }),
        "truncate-mid-entry must surface WalCorruption (page bytes overrun payload); \
         got {err_b:?}"
    );
}

// ─── Regression guards ───────────────────────────────────────────

#[test]
fn vector_page_entry_partition_id_always_zero_at_v1() {
    // Local-only guard (mirrors
    // `allocator_advance_partition_id_always_zero_at_v1` and
    // `replay_partition_id_always_zero_at_v1`). The v1.0 wire format
    // for `VectorPageEntry` carries an 8-byte `partition_id u64 LE`
    // slot which MUST be zero. The invariant is pinned structurally (struct
    // construction with `PartitionId::ZERO`) and on-wire (the slot
    // bytes are 0u64 LE).
    let entry = VectorPageEntry {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 0,
        page_id: PageId::new(1),
        commit_lsn: Lsn::new(1),
        bytes: mk_page_bytes(0xAB),
    };
    let encoded = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        std::slice::from_ref(&entry),
    );
    // Locate the partition_id slot inside the trailing entry. The
    // entry starts at: len_of_v4_prefix + 4 (n_vector_pages u32).
    let v4_prefix = encode_commit_bundle_v4(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
    );
    let entry_off = v4_prefix.len() + 4;
    // partition_id is the SECOND 8-byte field in the entry (after
    // tenant_id).
    let partition_off = entry_off + 8;
    assert_eq!(
        &encoded[partition_off..partition_off + 8],
        &0u64.to_le_bytes(),
        "v1.0 invariant: partition_id slot MUST be 0u64 LE at the \
         on-wire offset"
    );
}

#[test]
fn vector_page_entry_index_id_always_zero_at_v1() {
    // Mirror partition_id pin for the index_id slot. The v1.0 wire
    // format carries an 8-byte `index_id u64 LE` slot reserved for
    // v1.1 multi-index lift. v1.0 invariant: always 0.
    let entry = VectorPageEntry {
        tenant: TenantId::DEFAULT,
        partition: PartitionId::ZERO,
        index_id: 0,
        page_id: PageId::new(1),
        commit_lsn: Lsn::new(1),
        bytes: mk_page_bytes(0xAB),
    };
    let encoded = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        std::slice::from_ref(&entry),
    );
    let v4_prefix = encode_commit_bundle_v4(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
    );
    let entry_off = v4_prefix.len() + 4;
    // index_id is the THIRD 8-byte field (after tenant + partition).
    let index_id_off = entry_off + 8 + 8;
    assert_eq!(
        &encoded[index_id_off..index_id_off + 8],
        &0u64.to_le_bytes(),
        "v1.0 invariant: index_id slot MUST be 0u64 LE at the \
         on-wire offset"
    );
}

#[test]
fn vector_page_entry_encoded_len_is_pinned() {
    // ENCODED_LEN MUST equal 8 + 8 + 8 + 8 + 8 + 4 + PAGE_SIZE.
    // A future field addition would silently grow the entry; this
    // test guards against drift. v1.1 partition / index_id widening
    // MUST bump format_version (BUNDLE_FORMAT_V6 or later), not grow
    // the v5 entry.
    assert_eq!(
        VectorPageEntry::ENCODED_LEN,
        8 + 8 + 8 + 8 + 8 + 4 + PAGE_SIZE,
        "v5 VectorPageEntry on-wire size MUST be \
         tenant(8) + partition(8) + index_id(8) + page_id(8) + \
         commit_lsn(8) + n_bytes(4) + PAGE_SIZE — total {}, \
         got {}",
        8 + 8 + 8 + 8 + 8 + 4 + PAGE_SIZE,
        VectorPageEntry::ENCODED_LEN,
    );

    // Also pin the difference-of-encoded-sizes between zero entries
    // and one entry (mirrors
    // `v4_advances_section_size_is_17_per_entry`).
    let zero = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &[],
    );
    let one = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        &[mk_vec_entry(TenantId::DEFAULT.raw(), 1, 1, 0xAB)],
    );
    assert_eq!(
        one.len() - zero.len(),
        VectorPageEntry::ENCODED_LEN,
        "v1.0 VectorPageEntry on-wire entries MUST be exactly \
         {} bytes — v1.1 partition_id / index_id / multi-bytes \
         MUST bump format_version, not silently grow this entry",
        VectorPageEntry::ENCODED_LEN,
    );
}

// ─── Helper: DecodedCommitBundle structural sanity ───────────────

#[test]
fn v5_decoded_bundle_struct_carries_vector_pages_field() {
    // Type-level pin: DecodedCommitBundle MUST carry a
    // `vector_pages: Vec<VectorPageEntry>` field (Slice G.4 contract).
    // Dropping this test would fail once a future refactor renames
    // the field — we want a load-bearing assertion to flag the rename.
    let bundle = DecodedCommitBundle {
        commit_lsn: Lsn::new(1),
        primary_tenant: TenantId::DEFAULT,
        mvcc_writes: HashMap::new(),
        sidechannel_writes: Vec::new(),
        staged_pages: Vec::new(),
        deltas: Vec::new(),
        allocator_advances: Vec::new(),
        vector_pages: Vec::new(),
        // #352 Part 2 (ADR-199): v6 idempotency_bindings field.
        idempotency_bindings: Vec::new(),
        // #1221 (ADR-218): v8 acl_grants field.
        acl_grants: Vec::new(),
    };
    assert!(bundle.vector_pages.is_empty());
    assert!(bundle.idempotency_bindings.is_empty());
}

// ─── Sanity: dispatcher routes V5 ────────────────────────────────

#[test]
fn dispatcher_routes_v5_bundle_to_v5_decoder() {
    let entry = mk_vec_entry(TenantId::DEFAULT.raw(), 1, 1, 0xCC);
    let encoded = encode_commit_bundle_v5(
        Lsn::new(1),
        TenantId::DEFAULT,
        &HashMap::new(),
        &[],
        &[],
        &[],
        std::slice::from_ref(&entry),
    );
    let decoded =
        decode_commit_bundle_for_version(&encoded, BUNDLE_FORMAT_V5, TenantId::DEFAULT).unwrap();
    assert_eq!(decoded.vector_pages.len(), 1);
    assert_eq!(decoded.vector_pages[0].page_id, PageId::new(1));
}
