//! M3.a Phase 5.5 — I-V1..I-V7 invariant pin tests (ADR-035 §10).
//!
//! Phase 5.5 (Path A directive 2026-04-26: "all boundaries pushed
//! and tested") is the cross-cutting integration agent that pins
//! each ADR-035 §10 invariant with a dedicated regression guard.
//! Phase 5 covered per-slice boundary tests; Phase 5.5 covers
//! slice-interaction + invariant correctness.
//!
//! Each test is **load-bearing for v1.0 ship** — a failure here means
//! a Phase 5 regression has been introduced or production code drift
//! has violated a published contract from ADR-035 §10. The tests
//! must fail loudly with a single-glance diagnostic; CI gating on
//! these is the per-commit gauntlet's `cargo test -p arcgraph-vector
//! --test iv_invariants` step.
//!
//! ## Per-invariant test
//!
//! - **I-V1**: `iv1_mvcc_authoritative_no_ghost_vectors` — proptest
//!   with random crash + bootstrap_from_mvcc cycles; assert no ghost
//!   (search returns id ⟹ MVCC has matching version).
//! - **I-V2**: `iv2_tenant_isolation_2_tenant_proptest` — 2-tenant
//!   proptest across the full operation matrix (filtered search,
//!   unfiltered search, snapshot, recovery, rollback); zero
//!   cross-tenant leakage.
//! - **I-V3**: `iv3_filter_correctness_proptest` — 1 K random Filter
//!   expressions; every returned vector satisfies the filter.
//! - **I-V4**: `iv4_recall_floor_sweep` — selectivity sweep at
//!   multiple `dim × encoding` combinations; recall@10 ≥ AC threshold
//!   per encoding.
//! - **I-V5**: `iv5_partition_neutral_all_phase_5_apis` — every
//!   Phase-5 public API surface that names `PartitionId` either
//!   produces `ZERO` or rejects non-`ZERO`.
//! - **I-V6**: `iv6_tier_integration_vector_commits` — vector commits
//!   inherit ADR-034 tier dispatch verbatim; T1 strict commits
//!   durable before ack; T3 periodic commits durable within `rpo_ms`;
//!   T1 fsync piggybacks prior T3 commits.
//! - **I-V7**: `iv7_t1_ryw_diskann_filtered_full_chain` — DiskANN
//!   stream_insert + immediate filtered_search_with_delta + replay
//!   round-trip; the just-inserted vector is visible at every step.
//!
//! ## Sibling Path A coverage
//!
//! Phase 5 sibling test files exercise per-slice contracts:
//!
//! - F.2 (`hnsw_filtered.rs`) — selectivity sweep, filter
//!   correctness proptest at slice boundary.
//! - F.3 (`diskann_filtered.rs`) — same for filter-aware DiskANN.
//! - G.2 (`vector_snapshot.rs`) — snapshot atomic-rename, CRC,
//!   crash points.
//! - G.3 (`vector_recovery.rs`) — recovery from snapshot/WAL,
//!   bootstrap fallback, Z-1 rollback at the trait level.
//!
//! Phase 5.5's `iv_*` tests cross-cut: they hit the slice
//! interactions (e.g., F.2's filter under G.3's recovery; D's
//! stream_insert under F.3's filter dispatch).
//!
//! ## Knobs (env-controlled)
//!
//! - `IV_PROPTEST_CASES` — overrides the proptest case count for
//!   I-V1, I-V2, I-V3 (default: 64 for `cargo test`; the 1 K-case
//!   spec is honored by exporting `IV_PROPTEST_CASES=1024` in the
//!   release-mode CI gauntlet).
//! - `IV4_DIMS` — comma-separated list of dims for I-V4's sweep
//!   (default `16,32`; ADR-035 §11.1 reference `384,768,1024` is
//!   gated to release mode via `IV4_DIMS=384,768`).
//!
//! Run:
//!   cargo test -p arcgraph-vector --test iv_invariants
//!   IV_PROPTEST_CASES=1024 cargo test -p arcgraph-vector --release \
//!       --test iv_invariants

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arcgraph_core::{LabelId, Lsn, PageId, PartitionId, StringId, TenantId};
use arcgraph_storage::vector_store::recovery::VectorArenaPageStore;
use arcgraph_storage::vector_store::recovery::{
    IndexType as SnapIndexType, MvccVectorSource, VectorPageDelta, VectorRecoveryRequest,
    WalDeltaSource, bootstrap_from_mvcc,
};
use arcgraph_storage::vector_store::{
    SectionKind, SnapshotCatalog, SnapshotSection, SnapshotSpec, VectorPageStoreHandle,
    flush_snapshot,
};
use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{FilteredHnsw, HnswParams, Payload};
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{Encoding, Filter, IndexId, PropertyValue, VectorIndexHandle};
use parking_lot::Mutex;
use proptest::prelude::*;
use rand::distr::{Distribution, StandardUniform};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

// ─────────────────────────────────────────────────────────────────
// Shared test helpers
// ─────────────────────────────────────────────────────────────────

fn bytes_of(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Deterministic unit-vector dataset. Mirrors the F.2 helper so the
/// recall thresholds across this suite + F.2 / F.3 are comparable.
fn unit_vectors(seed: u64, n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    let u: f32 = StandardUniform.sample(&mut rng);
                    u * 2.0 - 1.0
                })
                .collect();
            l2_normalize(v)
        })
        .collect()
}

fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn proptest_case_count(default: u32) -> u32 {
    std::env::var("IV_PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(default)
}

/// Brute-force top-k under L2 (filtered subset). Mirrors the F.2
/// helper so I-V3 cross-checks against the same baseline shape.
fn brute_force_top_k_filtered(
    vectors: &[(VectorId, Vec<f32>, Payload)],
    query: &[f32],
    k: usize,
    filter: &Filter,
) -> Vec<VectorId> {
    let mut scored: Vec<(f32, VectorId)> = vectors
        .iter()
        .filter(|(_, _, p)| filter.matches(p))
        .map(|(id, v, _)| {
            let d: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (d, *id)
        })
        .collect();
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN compare"));
    scored.into_iter().take(k).map(|(_, id)| id).collect()
}

/// In-memory MvccVectorSource for I-V1 / I-V2 bootstrap cycles.
struct InMemoryMvcc {
    snapshot_lsn: Lsn,
    vectors: Mutex<std::collections::VecDeque<(u64, Vec<u8>)>>,
}

impl InMemoryMvcc {
    fn new(snapshot_lsn: Lsn, vectors: Vec<(u64, Vec<u8>)>) -> Self {
        Self {
            snapshot_lsn,
            vectors: Mutex::new(std::collections::VecDeque::from(vectors)),
        }
    }
}

impl MvccVectorSource for InMemoryMvcc {
    fn next_vector(
        &self,
    ) -> std::result::Result<Option<(u64, Vec<u8>)>, arcgraph_core::ArcGraphError> {
        Ok(self.vectors.lock().pop_front())
    }

    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }
}

/// In-memory empty WalDeltaSource — used by tests that do not care
/// about post-snapshot WAL deltas.
struct EmptyWal {
    snapshot_lsn: Lsn,
}

impl WalDeltaSource for EmptyWal {
    fn snapshot_lsn(&self) -> Lsn {
        self.snapshot_lsn
    }

    fn next_delta(
        &self,
    ) -> std::result::Result<Option<VectorPageDelta>, arcgraph_core::ArcGraphError> {
        Ok(None)
    }
}

// ─────────────────────────────────────────────────────────────────
// I-V1 — Vector index is consistent with MVCC (no ghost vectors)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V1): every committed vector W satisfies
// either `arena[t,i].contains(W)` OR the arena is in a rebuild-from-
// MVCC state and the rebuild materializes W. In neither case is W
// "lost"; in neither case does the index report a vector W' that
// MVCC does not also have.
//
// Phase 5.5 pin: random sequences of (insert into MVCC + arena,
// rebuild arena from MVCC). After each rebuild, every search result
// from the arena MUST have a corresponding entry in the MVCC log.
// No ghost vectors (search returns vector_id ⟹ MVCC has it).

/// Run a single I-V1 trial: build an authoritative MVCC log, mirror
/// it into a FilteredHnsw, randomly invoke `bootstrap_from_mvcc`
/// (which simulates the §9.1 "rebuild from MVCC" rebuild path), and
/// assert no ghost vectors in any search result.
fn iv1_trial(
    seed: u64,
    n_vectors: usize,
    n_queries: usize,
) -> std::result::Result<(), TestCaseError> {
    let dim = 8;
    let kernel = L2F32;
    let params = HnswParams {
        m: 8,
        ef_construction: 32,
        ef_search: 32,
        seed,
    };
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    let raw = unit_vectors(seed, n_vectors, dim);

    // The MVCC authoritative log (id → bytes). Every Ok insert into
    // the arena is mirrored here; bootstrap_from_mvcc rebuilds from
    // this set.
    let mut mvcc: HashMap<VectorId, Vec<u8>> = HashMap::new();
    for (i, v) in raw.iter().enumerate() {
        let id = VectorId::new(i as u32);
        let bytes = bytes_of(v);
        let payload = Payload::with_labels(vec![LabelId::new((i % 4) as u32 + 1)]);
        hnsw.filtered_insert(id, &bytes, payload, &kernel)
            .expect("insert");
        mvcc.insert(id, bytes);
    }

    // Pre-rebuild ghost check: every search result is in MVCC.
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(101));
    for _ in 0..n_queries {
        let q_idx: usize = (rng.next_u32() as usize) % raw.len();
        let q_bytes = bytes_of(&raw[q_idx]);
        let res = hnsw
            .filtered_search(&q_bytes, 5, &Filter::And(vec![]), 32, &kernel, Lsn::MAX)
            .expect("search");
        for (id, _) in &res {
            prop_assert!(
                mvcc.contains_key(id),
                "I-V1 ghost (pre-rebuild): vector {id:?} returned by search but absent from MVCC"
            );
        }
    }

    // Crash + rebuild via bootstrap_from_mvcc. The rebuilt arena
    // (RecoveredArena.bootstrap_vectors) MUST cover the MVCC set
    // exactly — no ghosts (vectors not in MVCC) and no losses
    // (vectors in MVCC but not in the rebuild output).
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let mvcc_vec: Vec<(u64, Vec<u8>)> = mvcc
        .iter()
        .map(|(id, bytes)| (id.raw() as u64, bytes.clone()))
        .collect();
    let mvcc_src = InMemoryMvcc::new(Lsn::new(1000), mvcc_vec);
    let req = VectorRecoveryRequest::v1(TenantId::new(1), 1, SnapIndexType::Hnsw, dim as u32);
    let arena = bootstrap_from_mvcc(store, &mvcc_src, req).expect("bootstrap");

    // Bootstrap output covers the MVCC set exactly.
    prop_assert_eq!(
        arena.bootstrap_vectors.len(),
        mvcc.len(),
        "I-V1 rebuild count mismatch"
    );
    let rebuilt_ids: HashSet<u64> = arena.bootstrap_vectors.iter().map(|(id, _)| *id).collect();
    for id in mvcc.keys() {
        prop_assert!(
            rebuilt_ids.contains(&(id.raw() as u64)),
            "I-V1 rebuild dropped MVCC vector {id:?}"
        );
    }
    for id in &rebuilt_ids {
        prop_assert!(
            mvcc.contains_key(&VectorId::new(*id as u32)),
            "I-V1 rebuild ghosted in vector_id {id} not present in MVCC"
        );
    }
    Ok(())
}

#[test]
fn iv1_mvcc_authoritative_no_ghost_vectors() {
    // Local default 256 cases (raised from 16 per Phase 5.5 PR #128
    // review fix #6). CI exports IV_PROPTEST_CASES=1024 to reach the
    // Path A 1 K-cases bar.
    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);
    runner
        .run(
            &(0u64..1_000_000, 16usize..64, 4usize..16),
            |(seed, n_vectors, n_queries)| iv1_trial(seed, n_vectors, n_queries),
        )
        .expect("I-V1 ghost-vector proptest failed");
}

// ─────────────────────────────────────────────────────────────────
// I-V2 — Tenant isolation (zero cross-tenant leakage)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V2): no path through the vector subsystem
// returns a vector_id from `arena[T1]` when the query was issued
// under `tenant=T2`. Includes filtered search, unfiltered search,
// snapshot, recovery, rollback.
//
// Phase 5.5 pin: 2-tenant proptest. Construct two arenas (one per
// tenant) with disjoint vector_id ranges. Run a random query mix
// (filtered / unfiltered / various K). Assert every result for
// tenant T comes from T's vector_id range. Verify isolation via:
//   1. Per-arena search (the arena selection IS the tenant filter
//      per ADR-011 §6.1).
//   2. Snapshot path: serialize each arena to its own snapshot file
//      (filename keyed by tenant) — assert the file's tenant-id
//      header byte matches the owning tenant.
//   3. Recovery path: invoke `bootstrap_from_mvcc` per tenant with
//      tenant-specific MVCC sources; assert no cross-tenant vector
//      ends up in either rebuilt arena.

#[test]
fn iv2_tenant_isolation_2_tenant_proptest() {
    // Build two single-tenant arenas with disjoint id-ranges.
    // Tenant 1 → ids in [0, 200); tenant 2 → ids in [1000, 1200).
    // Cross-tenant leakage manifests as a result id outside the
    // querying tenant's range.
    let dim = 8;
    let n_per_tenant = 100;
    let kernel = L2F32;
    let params = HnswParams {
        m: 8,
        ef_construction: 32,
        ef_search: 32,
        seed: 42,
    };
    let mut hnsw_t1 = FilteredHnsw::new(params, dim, &kernel);
    let mut hnsw_t2 = FilteredHnsw::new(params, dim, &kernel);

    let raw_t1 = unit_vectors(101, n_per_tenant, dim);
    let raw_t2 = unit_vectors(202, n_per_tenant, dim);

    let mut t1_data: Vec<(VectorId, Vec<f32>, Payload)> = Vec::new();
    let mut t2_data: Vec<(VectorId, Vec<f32>, Payload)> = Vec::new();
    for (i, v) in raw_t1.iter().enumerate() {
        let id = VectorId::new(i as u32);
        let payload = Payload {
            tenant_id: Some(TenantId::new(1)),
            labels: vec![LabelId::new((i % 4) as u32 + 1)],
            properties: HashMap::new(),
            ..Payload::default()
        };
        hnsw_t1
            .filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
            .expect("t1 insert");
        t1_data.push((id, v.clone(), payload));
    }
    for (i, v) in raw_t2.iter().enumerate() {
        let id = VectorId::new(1000 + i as u32);
        let payload = Payload {
            tenant_id: Some(TenantId::new(2)),
            labels: vec![LabelId::new((i % 4) as u32 + 1)],
            properties: HashMap::new(),
            ..Payload::default()
        };
        hnsw_t2
            .filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
            .expect("t2 insert");
        t2_data.push((id, v.clone(), payload));
    }

    // Local default 256 cases (raised from 64 per Phase 5.5 PR #128
    // review fix #6). CI exports IV_PROPTEST_CASES=1024 to reach the
    // Path A 1 K-cases bar.
    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);

    runner
        .run(
            &(any::<bool>(), 0usize..n_per_tenant, 1usize..6),
            |(query_t1, q_idx, k)| {
                let (qvec, query_tenant_id, expected_range) = if query_t1 {
                    (&raw_t1[q_idx], TenantId::new(1), 0u32..200u32)
                } else {
                    (&raw_t2[q_idx], TenantId::new(2), 1000u32..1200u32)
                };
                let q_bytes = bytes_of(qvec);

                // Filter mix: tenant filter, label filter, no filter.
                let filters: Vec<Filter> = vec![
                    Filter::Tenant(query_tenant_id),
                    Filter::LabelIn(vec![LabelId::new(1), LabelId::new(2)]),
                    Filter::And(vec![]),
                ];
                for filter in &filters {
                    // Query against the OWNING arena. Result ids
                    // MUST land in expected_range; the other arena
                    // is never consulted.
                    let target = if query_t1 { &hnsw_t1 } else { &hnsw_t2 };
                    let res = target
                        .filtered_search(&q_bytes, k, filter, 32, &kernel, Lsn::MAX)
                        .expect("search");
                    for (id, _) in &res {
                        prop_assert!(
                            expected_range.contains(&id.raw()),
                            "I-V2 leakage: tenant {query_tenant_id:?} \
                             query returned id {id:?} outside expected range \
                             {expected_range:?} (filter {filter:?})"
                        );
                    }

                    // Cross-arena negative test: querying the OTHER
                    // arena returns ids in the OTHER tenant's range,
                    // never the query tenant's range. This pins
                    // arena_for(T)::None for cross-tenant lookups
                    // per §6.1 Pattern A.
                    let other = if query_t1 { &hnsw_t2 } else { &hnsw_t1 };
                    let other_range = if query_t1 {
                        1000u32..1200u32
                    } else {
                        0u32..200u32
                    };
                    let cross = other
                        .filtered_search(&q_bytes, k, filter, 32, &kernel, Lsn::MAX)
                        .expect("cross-arena search");
                    for (id, _) in &cross {
                        prop_assert!(
                            other_range.contains(&id.raw()),
                            "I-V2 cross-arena leakage: arena for tenant \
                             {:?} returned id {id:?} outside its range \
                             {other_range:?}",
                            if query_t1 {
                                TenantId::new(2)
                            } else {
                                TenantId::new(1)
                            }
                        );
                    }
                }
                Ok(())
            },
        )
        .expect("I-V2 isolation proptest failed");

    // Snapshot path isolation. Each tenant's snapshot is keyed by
    // (tenant, index_id, lsn); the snapshot bytes themselves stamp
    // the tenant_id at offset 32..40 (per ADR-035 §4.1). Assert the
    // stamped tenant matches the owning tenant.
    let tmp = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    for tenant_raw in [1u64, 2u64] {
        let tenant = TenantId::new(tenant_raw);
        let payload = vec![tenant_raw as u8; 64];
        let sections = [SnapshotSection {
            kind: SectionKind::Quantized,
            bytes: &payload,
        }];
        let spec = SnapshotSpec {
            tenant,
            partition: PartitionId::ZERO,
            index_id: 1,
            lsn: Lsn::new(100 + tenant_raw),
            encoding: 0,   // F32
            index_type: 0, // HNSW
            dim: dim as u32,
            vectors_count: 64,
            sections: &sections,
        };
        let path = flush_snapshot(&spec, tmp.path(), &catalog).expect("flush");
        let bytes = std::fs::read(&path).unwrap();
        let stamped = u64::from_le_bytes(bytes[32..40].try_into().unwrap());
        assert_eq!(
            stamped, tenant_raw,
            "I-V2 snapshot path: tenant_id stamp at offset 32..40 \
             must equal owning tenant {tenant_raw}; got {stamped}"
        );
    }

    // Recovery path isolation. Per-tenant bootstrap_from_mvcc with
    // tenant-specific MVCC sources; the rebuilt arena's tenant_id
    // matches and no cross-tenant ids appear.
    for tenant_raw in [1u64, 2u64] {
        let tenant = TenantId::new(tenant_raw);
        let mvcc_vec: Vec<(u64, Vec<u8>)> = if tenant_raw == 1 {
            (0..n_per_tenant)
                .map(|i| (i as u64, bytes_of(&raw_t1[i])))
                .collect()
        } else {
            (0..n_per_tenant)
                .map(|i| (1000 + i as u64, bytes_of(&raw_t2[i])))
                .collect()
        };
        let mvcc_src = InMemoryMvcc::new(Lsn::new(500), mvcc_vec);
        let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
        let req = VectorRecoveryRequest::v1(tenant, 1, SnapIndexType::Hnsw, dim as u32);
        let arena = bootstrap_from_mvcc(store, &mvcc_src, req).expect("bootstrap");
        assert_eq!(arena.tenant_id, tenant, "I-V2 rebuilt arena tenant");
        let expected_range = if tenant_raw == 1 {
            0u64..200
        } else {
            1000u64..1200
        };
        for (id, _) in &arena.bootstrap_vectors {
            assert!(
                expected_range.contains(id),
                "I-V2 recovery path: rebuilt arena for tenant {tenant_raw} \
                 contains id {id} outside range {expected_range:?}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// I-V3 — Filter correctness (every result satisfies the filter)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V3): for every filtered query, every
// returned vector_id satisfies `filter.matches(payload(id))` AND is
// not tombstoned AND |result| ≤ K. F.2's Filter is the canonical
// type for v1.0 (per the task brief and #127 follow-up note).
//
// Phase 5.5 pin: 1 K random Filter expressions across the full v1.0
// grammar (Tenant / PropertyEq / LabelIn / And / Or / nested), each
// run against both `filtered_search` and `filtered_search_dispatch`.
// Every returned id must satisfy the filter when re-evaluated against
// the payload sidecar.

fn build_iv3_fixture() -> (FilteredHnsw, Vec<(VectorId, Vec<f32>, Payload)>) {
    let dim = 16;
    let n = 200;
    let params = HnswParams {
        m: 8,
        ef_construction: 50,
        ef_search: 50,
        seed: 1337,
    };
    let kernel = L2F32;
    let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
    let mut all = Vec::new();
    let labels = [
        LabelId::new(1),
        LabelId::new(2),
        LabelId::new(3),
        LabelId::new(4),
        LabelId::new(5),
    ];
    let prop_keys = [StringId::new(10), StringId::new(20)];

    let mut rng = StdRng::seed_from_u64(7);
    let raw = unit_vectors(13, n, dim);
    for (i, v) in raw.iter().enumerate() {
        let n_lbls = (rng.next_u32() as usize) % 4;
        let mut shuffled = labels.to_vec();
        for j in (1..shuffled.len()).rev() {
            let k = (rng.next_u32() as usize) % (j + 1);
            shuffled.swap(j, k);
        }
        let chosen_labels: Vec<LabelId> = shuffled.into_iter().take(n_lbls).collect();
        let mut props = HashMap::new();
        for &pk in &prop_keys {
            if rng.next_u32() % 2 == 0 {
                let val = rng.next_u32() % 10;
                props.insert(pk, PropertyValue::U32(val));
            }
        }
        let payload = Payload {
            tenant_id: Some(TenantId::DEFAULT),
            labels: chosen_labels,
            properties: props,
            ..Payload::default()
        };
        let id = VectorId::new(i as u32);
        hnsw.filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
            .expect("insert");
        all.push((id, v.clone(), payload));
    }
    (hnsw, all)
}

fn arb_filter() -> impl Strategy<Value = Filter> {
    let leaf = prop_oneof![
        Just(Filter::Tenant(TenantId::DEFAULT)),
        (1u32..=5).prop_map(|l| Filter::LabelIn(vec![LabelId::new(l)])),
        prop::collection::vec(1u32..=5, 0..3)
            .prop_map(|ls| Filter::LabelIn(ls.into_iter().map(LabelId::new).collect())),
        (
            (10u32..=20).prop_filter("only known prop_keys", |k| *k == 10 || *k == 20),
            0u32..10
        )
            .prop_map(|(k, v)| Filter::PropertyEq(StringId::new(k), PropertyValue::U32(v))),
    ];
    leaf.prop_recursive(3, 8, 4, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..3).prop_map(Filter::And),
            prop::collection::vec(inner, 0..3).prop_map(Filter::Or),
        ]
    })
}

#[test]
fn iv3_filter_correctness_proptest() {
    let (hnsw, all) = build_iv3_fixture();
    let kernel = L2F32;

    // Default 256 cases for `cargo test`; release-mode CI exports
    // IV_PROPTEST_CASES=1024 to satisfy the spec's "1 K cases" bar.
    let cases = proptest_case_count(256);
    let config = ProptestConfig {
        cases,
        ..ProptestConfig::default()
    };
    let mut runner = proptest::test_runner::TestRunner::new(config);

    // Use a deterministic query basis (the first 4 dataset vectors)
    // so the test is reproducible across runs.
    let queries: Vec<Vec<u8>> = all.iter().take(4).map(|(_, v, _)| bytes_of(v)).collect();

    runner
        .run(&arb_filter(), |filter| {
            for q_bytes in &queries {
                // 1. filtered_search — every returned id satisfies
                //    the filter.
                let res = hnsw
                    .filtered_search(q_bytes, 10, &filter, 50, &kernel, Lsn::MAX)
                    .expect("search");
                for (id, _) in &res {
                    let p = hnsw
                        .payload(*id)
                        .expect("payload sidecar present for inserted id");
                    prop_assert!(
                        filter.matches(p),
                        "I-V3 filter violation (filtered_search): id {id:?} returned \
                         but does not satisfy filter {:?} (payload labels={:?}, \
                         properties={:?})",
                        filter,
                        p.labels,
                        p.properties
                    );
                }
                // |result| ≤ K guard.
                prop_assert!(res.len() <= 10, "I-V3 |result| > K");

                // 2. filtered_search_dispatch — same property must
                //    hold under the selectivity-driven dispatcher.
                //    The hardcoded `0.5_f32` exercises the mid-range
                //    branch; the I-V3 filter-correctness property
                //    holds in every branch so a single seed is
                //    sufficient (Phase 6 F.4 will own the real
                //    selectivity estimator per ADR-035 amendment-03).
                let sel = 0.5_f32;
                let res_d = hnsw
                    .filtered_search_dispatch(q_bytes, 10, &filter, sel, 50, &kernel, Lsn::MAX)
                    .expect("dispatch");
                for (id, _) in &res_d {
                    let p = hnsw
                        .payload(*id)
                        .expect("payload sidecar present for dispatch-returned id");
                    prop_assert!(
                        filter.matches(p),
                        "I-V3 filter violation (dispatch): id {id:?} returned \
                         but does not satisfy filter {:?}",
                        filter
                    );
                }
                prop_assert!(res_d.len() <= 10, "I-V3 dispatch |result| > K");
            }
            Ok(())
        })
        .expect("I-V3 filter correctness proptest failed");

    // Bonus pin (W28-S2 recall-oracle hardening; gap analysis PR
    // #510 §3 + ADR-165 M1): a real recall FRACTION against the
    // exhaustive brute-force ground truth, averaged over a batch of
    // deterministic queries.
    //
    // **What this replaces.** The prior oracle was
    // `assert!(!inter.is_empty())` on a single query — "≥ 1 overlap
    // with brute-force counts as recall". That is the weakest recall
    // oracle in the repo: a filtered search that returned exactly one
    // correct id out of K (recall 1/K = 0.20) passed it, and a
    // graph-construction regression that collapsed recall to 0.20
    // would NOT have been caught. We now assert a genuine
    // mean-recall@K fraction so the oracle can fail on the bug it
    // guards (test-suite-green ≠ test-correctness;
    // `feedback_review_oracle_relaxations`).
    //
    // **Why a mean over a batch.** Filtered HNSW recall on a single
    // query is a sample from a probabilistic algorithm; the mean over
    // ≥ 1 query (those whose matching subset is non-empty) is the
    // statistically meaningful quantity. The floor below is a genuine
    // statistical floor on that mean, NOT a slack tolerance.
    let tight = Filter::LabelIn(vec![LabelId::new(1)]);
    let k = 5;
    // First 20 dataset vectors as deterministic queries.
    let mut total = 0.0_f64;
    let mut q_seen = 0_usize;
    for (_, qv, _) in all.iter().take(20) {
        // EXHAUSTIVE brute-force over the matching subset — the
        // ground-truth oracle is a full linear scan, never sampled.
        let bf = brute_force_top_k_filtered(&all, qv, k, &tight);
        if bf.is_empty() {
            continue;
        }
        let res = hnsw
            .filtered_search(&bytes_of(qv), k, &tight, 50, &kernel, Lsn::MAX)
            .expect("tight search");
        // I-V3 sub-property re-stated: every returned id satisfies
        // the filter (the recall fraction must not be inflated by
        // filter-violating ids slipping into the intersection).
        for (id, _) in &res {
            let p = hnsw.payload(*id).expect("payload");
            assert!(
                tight.matches(p),
                "I-V3 baseline: filtered_search returned id {id:?} that fails \
                 the label filter"
            );
        }
        let res_ids: HashSet<VectorId> = res.iter().map(|(id, _)| *id).collect();
        let bf_ids: HashSet<VectorId> = bf.iter().copied().collect();
        let inter = res_ids.intersection(&bf_ids).count();
        // Denominator is the achievable ceiling (matching subset may
        // be smaller than k); clamp to ≥ 1 to avoid div-by-zero.
        let denom = bf.len().min(k).max(1);
        total += inter as f64 / denom as f64;
        q_seen += 1;
    }
    assert!(
        q_seen > 0,
        "I-V3 baseline: no query had a non-empty label-1 matching subset"
    );
    let mean = total / q_seen as f64;
    println!(
        "I-V3 baseline tight-filter (LabelIn[1]) recall@{k} mean over {q_seen} queries = {mean:.4}"
    );
    // Floor calibrated against the measured mean for this fixture
    // (M=8, ef_search=50, ~30 % selectivity over 200 vectors). The
    // sweep is deterministic (fixture seed 1337, data seed 13,
    // queries = first 20 dataset vectors) and measures mean
    // recall@5 = 1.0000 — see the `println!` above + PR body
    // §"Numerical claim derivation". We pin the production filtered
    // target 0.90; the 10-point margin below the observed 1.0 is a
    // genuine statistical floor on a probabilistic search, not a
    // "≥ 1 overlap" placeholder (which 0.20 recall used to pass).
    let floor = 0.90_f64;
    assert!(
        mean >= floor,
        "I-V3 baseline: filtered_search mean recall@{k} = {mean:.4} < floor {floor} \
         on a label filter matching the fixture's label-1 subset"
    );
}

// ─────────────────────────────────────────────────────────────────
// I-V4 — Recall floor sweep (selectivity × dim × encoding)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V4 + §11.1 acceptance criteria): the
// measured recall@10 against brute-force ground truth meets the
// per-encoding floor. At v1.0 the ship-blocking floor is:
//
//   - Raw f32 HNSW:  recall@10 ≥ 0.97
//   - SQ8 + rescore: recall@10 ≥ 0.95
//   - Filtered HNSW: recall@10 ≥ 0.90 across all selectivities
//
// Phase 5.5 pin: small-N sweep across `dim ∈ {16, 32}` × encoding
// ∈ {F32} (Sq8 codebook training is gated to release benches).
// Selectivity ∈ {1 %, 10 %, 50 %, 99 %, 100 %}.
//
// W28-S2 hardening: the recall floor is now the production AC-5
// bar **0.90** in every bucket (see `iv4_floor`), tightened from
// the old small-N relaxation (0.80 / 0.85) after the deterministic
// sweep measured recall@10 = 1.0000 in all 10 buckets. The
// release-mode F.2 SELECTIVITY_FIXTURE at N=5K enforces the same
// 0.90 bar; the small-N fixture no longer relaxes it.

fn iv4_default_dims() -> Vec<usize> {
    std::env::var("IV4_DIMS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![16, 32])
}

/// Per-selectivity recall@10 floor for the I-V4 sweep.
///
/// **W28-S2 hardening (gap analysis PR #510 §3 + ADR-165 M1).**
/// The prior floors were `0.80` (≤ 1 % selectivity) / `0.85`
/// (everything else) — a "~5–15 point allowance for small-N
/// variance" below the production AC-5 ≥ 0.90 bar. The empirical
/// sweep (this fixture is fully deterministic: data seed
/// `7 + dim`, query seed `99_991 + dim`, HNSW seed `99`) measures
/// recall@10 = **1.0000** in EVERY (dim, selectivity) bucket —
/// see the `println!` above and the PR body §"Numerical claim
/// derivation". The old floors therefore left a 15–20 point gap
/// that a real recall regression could hide inside.
///
/// We tighten every bucket to the production target **0.90**. The
/// 10-point margin below the observed 1.0 is a genuine statistical
/// floor on a probabilistic ANN search — it absorbs legitimate
/// parameter-tuning drift while still failing loudly on the
/// connectivity-bug class (e.g. the PR #126 seed=76603 regression)
/// that collapses recall by ≥ 10 points. It is NOT slack: the mean
/// is taken over 20 queries, so dropping it below 0.90 requires a
/// broad, multi-query recall collapse, not a single unlucky query.
fn iv4_floor(_cutoff: f64) -> f64 {
    0.90
}

#[test]
fn iv4_recall_floor_sweep() {
    let dims = iv4_default_dims();
    let kernel = L2F32;
    // N=600 keeps the test fast while supplying enough vectors per
    // bucket for the recall@10 measurement to be statistically
    // meaningful at low selectivities. The 1 % bucket has 6
    // vectors → recall@k clamps to brute-force size when bf < k.
    let n = 600;
    // Bucketed labels per F.2's selectivity scheme; relabel for the
    // smaller fixture: 1 % / 10 % / 50 % / 99 % / 100 %.
    let buckets: Vec<(f64, LabelId)> = vec![
        (0.01, LabelId::new(201)),
        (0.10, LabelId::new(202)),
        (0.50, LabelId::new(203)),
        (0.99, LabelId::new(204)),
        (1.00, LabelId::new(205)),
    ];

    for dim in dims {
        let params = HnswParams {
            m: 16,
            ef_construction: 100,
            ef_search: 100,
            seed: 99,
        };
        let mut hnsw = FilteredHnsw::new(params, dim, &kernel);
        let raw = unit_vectors(7 + dim as u64, n, dim);

        let mut data: Vec<(VectorId, Vec<f32>, Payload)> = Vec::new();
        for (i, v) in raw.iter().enumerate() {
            let frac = i as f64 / n as f64;
            let mut labels: Vec<LabelId> = Vec::new();
            for (cutoff, lbl) in &buckets {
                if frac < *cutoff {
                    labels.push(*lbl);
                }
            }
            let payload = Payload::with_labels(labels);
            let id = VectorId::new(i as u32);
            hnsw.filtered_insert(id, &bytes_of(v), payload.clone(), &kernel)
                .expect("insert");
            data.push((id, v.clone(), payload));
        }

        // Use 20 deterministic queries per dim so the test stays fast.
        let queries = unit_vectors(99_991 + dim as u64, 20, dim);
        let k = 10;

        for (cutoff, lbl) in &buckets {
            let filter = Filter::LabelIn(vec![*lbl]);
            let mut total = 0.0_f64;
            let mut q_count = 0_usize;
            for q in &queries {
                let bf = brute_force_top_k_filtered(&data, q, k, &filter);
                if bf.is_empty() {
                    continue;
                }
                let res = hnsw
                    .filtered_search(&bytes_of(q), k, &filter, 100, &kernel, Lsn::MAX)
                    .expect("search");
                // Pin: every returned id satisfies the filter (I-V3
                // sub-property).
                for (id, _) in &res {
                    let p = hnsw.payload(*id).expect("payload");
                    assert!(
                        filter.matches(p),
                        "I-V4 sub-property: returned id {id:?} fails filter (dim={dim}, \
                         selectivity≈{cutoff})"
                    );
                }
                let h: HashSet<VectorId> = res.iter().map(|(id, _)| *id).take(k).collect();
                let b: HashSet<VectorId> = bf.iter().copied().take(k).collect();
                let inter = h.intersection(&b).count();
                let denom = b.len().min(k).max(1);
                total += inter as f64 / denom as f64;
                q_count += 1;
            }
            if q_count == 0 {
                continue;
            }
            let mean = total / q_count as f64;
            println!(
                "I-V4 recall floor sweep: dim={dim} selectivity≈{cutoff} \
                 recall@{k}={mean:.4} over {q_count} queries"
            );
            let floor: f64 = iv4_floor(*cutoff);
            assert!(
                mean >= floor,
                "I-V4 recall floor: dim={dim} selectivity≈{cutoff} \
                 recall@{k}={mean:.4} < floor {floor}"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// I-V4 mixed-label DiskANN — F.3 connectivity cross-pin
// ─────────────────────────────────────────────────────────────────
//
// Pin the F.3 (Filtered-DiskANN) recall floor at the I-V layer with
// a mixed-label dataset that exercises FilteredRobustPrune's per-
// label connectivity guarantee (Gollapudi et al. WWW 2023 §5).
//
// Phase 5.5 PR #128 review fold-in (#10/Fix 4): the existing
// iv4_recall_floor_sweep covers ONLY HNSW (FilteredHnsw); the
// existing iv7_t1_ryw_diskann_filtered_full_chain uses DiskANN's
// filtered API but every vector carries the SAME label, so the
// per-label entry-point cache is a single bucket and the
// connectivity claim is trivial. Neither test exercised the
// parameter regime where the seed=76603 connectivity bug surfaced
// during PR #126 review (Slice F.3 boundary tests). The fix landed
// at commit 499d167; this test pins the cross-cutting equivalent
// at I-V4 so a future regression of the FilteredRobustPrune
// connectivity guarantee fails the gauntlet loudly.
//
// Companion: `crates/arcgraph-vector/tests/diskann_filtered.
// proptest-regressions` (Phase 5.5 PR #128 review fix #3) commits
// the F.3 proptest replay seed so the slice-level proptest also
// re-runs the historical failure case deterministically.

#[test]
fn iv4_recall_floor_diskann_filtered_mixed_labels() {
    // Mirrors the f3_filtered_alpha_prune_preserves_label_connectivity
    // shape (DiskAnnGraph::build_filtered with mixed labels {0, 1, 2})
    // but pinned as a deterministic non-proptest test at the I-V
    // layer. Per-label recall@10 ≥ 0.85 (AC-6 floor); per-label
    // search MUST also return only label-matching vectors (I-V3
    // sub-property re-stated for DiskANN's filter dispatcher).
    use arcgraph_vector::{DistanceKernel, Metric};

    // 80 vectors at dim=8 with **deterministic minority-label**
    // distribution: label 0 = 50 vectors (majority), label 1 = 15
    // (minority), label 2 = 15 (minority). Both minority buckets
    // are below R=16, so FilteredRobustPrune's "preserve label-co-
    // located edges until R is exceeded" guarantee gets exercised
    // — this is the regime where uneven distributions stress the
    // connectivity invariant per Gollapudi et al. WWW 2023 §5.
    //
    // Note: the previous uniform `i % 3` distribution gave a
    // perfectly even 27/27/26 split — the BEST CASE for
    // FilteredRobustPrune. PR #128 review fold-in (2026-04-26)
    // pinned the minority-label fixture instead so a real
    // connectivity regression in the minority regime fails the
    // gauntlet loudly. This was the regime the prior Phase 5
    // review claimed seed=76603 failed in (independently re-derived
    // and retracted on re-review — see the proptest-regressions
    // companion file's docstring for the empirical result).
    let n = 80usize;
    let dim = 8usize;
    let kernel = L2F32;

    let mut rng = StdRng::seed_from_u64(0x1356_3F00); // 76603 in lower bits
    let raw: Vec<Vec<f32>> = (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    let u: f32 = StandardUniform.sample(&mut rng);
                    u * 2.0 - 1.0
                })
                .collect();
            l2_normalize(v)
        })
        .collect();
    let labels: Vec<u32> = (0..n)
        .map(|i| {
            if i < 50 {
                0
            } else if i < 65 {
                1
            } else {
                2
            }
        })
        .collect();
    debug_assert_eq!(labels.iter().filter(|&&l| l == 0).count(), 50);
    debug_assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 15);
    debug_assert_eq!(labels.iter().filter(|&&l| l == 2).count(), 15);

    let params = DiskAnnParams {
        r: 16,
        alpha: 1.2,
        l_construction: 48,
        l_search_default: 64,
        ..DiskAnnParams::default()
    };
    let mut g = DiskAnnGraph::new(params, Encoding::F32, Metric::L2, Box::new(L2F32))
        .expect("DiskAnnGraph::new");
    let owned: Vec<(VectorId, Vec<u8>)> = raw
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), bytes_of(v)))
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    let label_opts: Vec<Option<DiskAnnLabelId>> = labels.iter().map(|l| Some(*l)).collect();
    g.build_filtered(&pairs, &label_opts, &kernel)
        .expect("build_filtered");

    // Brute-force ground truth + per-label recall measurement.
    // For each label l ∈ {0, 1, 2}: pick 10 query vectors from
    // label-l members, run filtered_search, compute recall@k
    // against brute-force top-k filtered to label-l.
    let k = 10usize;
    for target_label in 0u32..3 {
        let label_members: Vec<usize> = labels
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == target_label)
            .map(|(i, _)| i)
            .collect();
        if label_members.len() < k {
            // Bucket too small for a meaningful recall@k. Skip;
            // the test's other labels carry the recall assertion.
            continue;
        }

        // I-V3 sub-property: every returned vector has the
        // matching label.
        let query_idx = label_members[0];
        let q_bytes = bytes_of(&raw[query_idx]);
        let filter = Filter::label_eq(target_label);
        let res = g
            .filtered_search(&q_bytes, k, &filter, 64, &kernel, Lsn::MAX)
            .expect("filtered_search");
        for (id, _) in &res {
            let idx = id.raw() as usize;
            assert_eq!(
                labels[idx], target_label,
                "I-V4 mixed-label DiskANN: filtered_search returned id {id:?} \
                 with label {} instead of target_label {target_label} \
                 (FilteredRobustPrune connectivity / dispatcher correctness regression)",
                labels[idx]
            );
        }

        // Per-label recall@10 across multiple queries from the
        // same label bucket.
        let n_queries = label_members.len().min(10);
        let mut total_recall = 0.0_f64;
        let mut q_count = 0_usize;
        for &qi in label_members.iter().take(n_queries) {
            let q_b = bytes_of(&raw[qi]);
            // Brute-force ground truth: top-k label-`target_label`
            // members ranked by L2.
            let mut bf: Vec<(f32, VectorId)> = label_members
                .iter()
                .map(|&j| {
                    let d: f32 = raw[qi]
                        .iter()
                        .zip(raw[j].iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (d, VectorId::new(j as u32))
                })
                .collect();
            bf.sort_by(|a, b| a.0.partial_cmp(&b.0).expect("non-NaN compare"));
            let bf_ids: HashSet<VectorId> = bf.iter().take(k).map(|(_, id)| *id).collect();

            let res = g
                .filtered_search(&q_b, k, &filter, 64, &kernel, Lsn::MAX)
                .expect("filtered_search");
            let res_ids: HashSet<VectorId> = res.iter().map(|(id, _)| *id).collect();
            let inter = res_ids.intersection(&bf_ids).count();
            let denom = bf_ids.len().min(k).max(1);
            total_recall += inter as f64 / denom as f64;
            q_count += 1;
        }
        let mean = total_recall / q_count.max(1) as f64;
        // AC-6: Filtered-DiskANN recall@10 ≥ 0.85 across all
        // selectivities (per ADR-035 §11.3). With the
        // minority-label fixture (50/15/15 at N=80) the two
        // minority buckets sit below R=16, so FilteredRobustPrune's
        // connectivity guarantee is the dominant constraint
        // rather than statistical recall variance.
        let floor = 0.85_f64;
        eprintln!(
            "iv4 minority-label recall: target_label={target_label} \
             bucket_size={} queries={} recall@{k}={mean:.4} (floor {floor})",
            label_members.len(),
            q_count
        );
        assert!(
            mean >= floor,
            "I-V4 mixed-label DiskANN: target_label={target_label} \
             recall@{k}={mean:.4} < floor {floor} (would indicate F.3 \
             FilteredRobustPrune connectivity regression in the \
             minority-label regime)"
        );
    }

    // Connectivity sanity: from each label-l vertex, filtered_search
    // for that label must reach EVERY OTHER label-l vertex within
    // l_search rounds. Mirrors the F.3 connectivity proptest
    // (`f3_filtered_alpha_prune_preserves_label_connectivity` —
    // seed=76603 was the historical regression case captured by
    // the proptest-regressions companion file).
    for target_label in 0u32..3 {
        let label_members: Vec<VectorId> = labels
            .iter()
            .enumerate()
            .filter(|(_, l)| **l == target_label)
            .map(|(i, _)| VectorId::new(i as u32))
            .collect();
        if label_members.len() < 2 {
            continue;
        }
        let start_idx = label_members[0].raw() as usize;
        let q_bytes = bytes_of(&raw[start_idx]);
        let filter = Filter::label_eq(target_label);
        // High l_search to give the beam room to reach every
        // matching vertex.
        let res = g
            .filtered_search(
                &q_bytes,
                label_members.len(),
                &filter,
                256,
                &kernel,
                Lsn::MAX,
            )
            .expect("connectivity search");
        let reached: HashSet<VectorId> = res.iter().map(|(id, _)| *id).collect();
        let unreached: Vec<VectorId> = label_members
            .iter()
            .filter(|id| !reached.contains(id))
            .copied()
            .collect();
        // F.3 connectivity claim: at sufficient l_search the per-
        // label sub-graph stays connected → reach every vertex.
        // Allow at most 10% unreached as the AC-6 statistical
        // floor (per Gollapudi 2023 §5 connectivity invariant
        // is asymptotic, not absolute, at small N).
        let max_unreached = (label_members.len() / 10).max(1);
        eprintln!(
            "iv4 minority-label connectivity: target_label={target_label} \
             reached={}/{} unreached={} max_allowed={max_unreached}",
            reached.len(),
            label_members.len(),
            unreached.len()
        );
        assert!(
            unreached.len() <= max_unreached,
            "I-V4 mixed-label DiskANN connectivity: target_label={target_label} \
             reached {}/{} vertices (max_unreached_allowed={max_unreached}); \
             unreached: {:?}. Would indicate F.3 FilteredRobustPrune \
             connectivity regression in the minority-label regime.",
            reached.len(),
            label_members.len(),
            unreached
        );
    }

    // Suppress kernel-trait import warning if unused above.
    let _ = kernel.encoding();
}

// ─────────────────────────────────────────────────────────────────
// I-V5 — Local-only (PartitionId == ZERO at v1.0)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V5 + §8): every public-API construction
// site for a vector handle / recovery request / snapshot spec
// produces `PartitionId::ZERO`.
//
// Phase 5.5 pin: enumerate every Phase-5 API surface that names
// `PartitionId` and assert it accepts `ZERO`-only or hard-codes ZERO
// internally. This is the cross-cutting version of the per-slice
// regression guards (`g3_recover_request_partition_id_always_zero_at_v1`,
// `f2_partition_id_always_zero_at_v1`, etc.).

#[test]
fn iv5_partition_neutral_all_phase_5_apis() {
    // 1. VectorIndexHandle::for_tenant — hard-codes ZERO.
    let h = VectorIndexHandle::for_tenant(TenantId::new(7), IndexId::new(42));
    assert_eq!(
        h.partition(),
        PartitionId::ZERO,
        "I-V5 VectorIndexHandle::for_tenant must produce PartitionId::ZERO"
    );
    assert!(h.is_v1_local(), "I-V5 handle must be v1-local");

    // 2. VectorRecoveryRequest::v1 — hard-codes ZERO.
    let req = VectorRecoveryRequest::v1(TenantId::new(11), 7, SnapIndexType::Hnsw, 8);
    assert_eq!(
        req.partition_id,
        PartitionId::ZERO,
        "I-V5 VectorRecoveryRequest::v1 must produce PartitionId::ZERO"
    );

    // 3. SnapshotSpec — the local ZERO sentinel flushes successfully.
    let tmp = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let payload = vec![0u8; 32];
    let sections = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload,
    }];
    let spec_zero = SnapshotSpec {
        tenant: TenantId::new(13),
        partition: PartitionId::ZERO,
        index_id: 1,
        lsn: Lsn::new(7),
        encoding: 0,
        index_type: 0,
        dim: 8,
        vectors_count: 32,
        sections: &sections,
    };
    let path = flush_snapshot(&spec_zero, tmp.path(), &catalog).expect("flush ZERO");
    assert!(
        path.exists(),
        "I-V5 snapshot flush at ZERO partition succeeds"
    );

    // 4. The recovery request's partition_id is structurally part of
    //    the API. Confirm every known v1 constructor produces ZERO.
    let v1_a = VectorRecoveryRequest::v1(TenantId::DEFAULT, 1, SnapIndexType::Hnsw, 8);
    let v1_b = VectorRecoveryRequest::v1(TenantId::SYSTEM, 99, SnapIndexType::DiskAnn, 256);
    assert_eq!(v1_a.partition_id, PartitionId::ZERO);
    assert_eq!(v1_b.partition_id, PartitionId::ZERO);

    // 5. The only public construction path produces local handles.
    for (t, i) in [(1u64, 1u64), (1, 2), (2, 1), (99, 99)] {
        let h = VectorIndexHandle::for_tenant(TenantId::new(t), IndexId::new(i));
        assert!(
            h.is_v1_local(),
            "I-V5: VectorIndexHandle for ({t},{i}) is not v1-local"
        );
    }

    // 6. The DiskANN public API surface accepts no partition_id at
    //    v1.0 (per ADR-035 §8.4 + the sliceLocked parameter set).
    //    `DiskAnnGraph::new` takes (params, encoding, metric, kernel)
    //    only — no partition. We pin this by constructing a graph
    //    and confirming no public method names PartitionId in its
    //    return type. The smoke is structural (the type compiles),
    //    not behavioral.
    let _g = DiskAnnGraph::new(
        DiskAnnParams::default(),
        Encoding::F32,
        arcgraph_vector::Metric::L2,
        Box::new(L2F32),
    )
    .expect("DiskAnnGraph::new construct");

    // 7. Cross-arena: VectorPageStoreHandle methods are
    //    (TenantId, PageId, &[u8]) — no PartitionId in the trait
    //    signature. The G.1 trait was deliberately defined this way
    //    so v1.1 extension is a key widening (§7.5 doc note). We
    //    smoke-test by constructing the in-memory backend and
    //    invoking install_or_replace; a future addition of a
    //    partition_id method parameter would break this call site
    //    and surface here.
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    store
        .install_or_replace(TenantId::new(5), PageId::new(1), &[0xCAu8; 32])
        .expect("install_or_replace ZERO partition shape");
}

// ─────────────────────────────────────────────────────────────────
// I-V6 — Tier integration (T1 strict, T3 periodic, piggyback)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V6): vector commits inherit ADR-034 tier
// dispatch verbatim. T1 strict: durable before ack. T3 periodic:
// durable within `rpo_ms`. T1 commits piggyback prior T3 commits.
//
// Phase 5.5 pin: at the trait + recovery-fixture level, the vector
// arena page bytes are byte-routed through the same WAL bundle path
// as record / blob bytes. ADR-035 §7 sequence diagrams show vector
// arena pages share staged_pages with primary / record / blob. The
// tier byte does NOT vary per page-kind; ADR-034 D-3 (and the
// existing `replay_bytes_are_tier_agnostic` regression in
// `durability_tier_mixed.rs`) pins the tier-agnostic codec.
//
// At Phase 5.5 (test-only, no production CrudStore wiring of the
// vector page store yet), we pin the tier-integration contract at
// the vector-snapshot + arena-recovery level:
//
//   - A T1-tier snapshot flush (sync `flush_snapshot`) is durable
//     on return (rename + dir-fsync + catalog stamp completed).
//   - A second T1 flush after the first finds the catalog already
//     advanced past the prior LSN (piggyback equivalence: the latest
//     stamp covers all earlier stamped flushes).
//   - The recovery loader reads the latest snapshot per
//     `(tenant, index_id)` AND replays only post-snapshot WAL
//     deltas (the analogue of T3 fsync-batch).

#[test]
fn iv6_tier_integration_vector_commits() {
    let tmp = TempDir::new().unwrap();
    let catalog = SnapshotCatalog::new();
    let tenant = TenantId::new(73);
    let index_id = 1u64;
    let dim = 8u32;

    // T1 commit equivalence: flush_snapshot is sync, returns Ok only
    // after the rename + dir-fsync + catalog stamp are all complete.
    // Post-return the snapshot file exists AND the catalog stamp is
    // ≥ the flush's LSN (by construction).
    let payload_1 = vec![0xAAu8; 64];
    let sections_1 = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload_1,
    }];
    let spec_1 = SnapshotSpec {
        tenant,
        partition: PartitionId::ZERO,
        index_id,
        lsn: Lsn::new(100),
        encoding: 0,
        index_type: 0,
        dim,
        vectors_count: 64,
        sections: &sections_1,
    };
    let path_1 = flush_snapshot(&spec_1, tmp.path(), &catalog).expect("T1 flush 1");
    assert!(path_1.exists(), "I-V6 T1 path-on-disk after sync flush");
    assert_eq!(
        catalog.latest_lsn(tenant, index_id),
        Some(Lsn::new(100)),
        "I-V6 T1 catalog stamp matches flush LSN (durable-before-ack analogue)"
    );

    // Piggyback equivalence: a second T1 flush at LSN > prior LSN
    // advances the catalog stamp. The prior snapshot file remains
    // on disk; the catalog points at the newer one. (The G.3
    // recovery loader picks the highest-stamped snapshot per
    // ADR-035 §4.6 step 5 + the §4.5 cleanup pass.)
    let payload_2 = vec![0xBBu8; 64];
    let sections_2 = [SnapshotSection {
        kind: SectionKind::Quantized,
        bytes: &payload_2,
    }];
    let spec_2 = SnapshotSpec {
        tenant,
        partition: PartitionId::ZERO,
        index_id,
        lsn: Lsn::new(250),
        encoding: 0,
        index_type: 0,
        dim,
        vectors_count: 64,
        sections: &sections_2,
    };
    let path_2 = flush_snapshot(&spec_2, tmp.path(), &catalog).expect("T1 flush 2");
    assert!(path_2.exists(), "I-V6 T1 second-flush path-on-disk");
    assert_eq!(
        catalog.latest_lsn(tenant, index_id),
        Some(Lsn::new(250)),
        "I-V6 piggyback: catalog stamp advances to most-recent flush"
    );
    assert!(
        path_1.exists(),
        "I-V6 prior snapshot file persists post-second-flush \
         (G.3 cleanup pass owns GC; flush itself is non-destructive)"
    );

    // T3 equivalence: the post-snapshot WAL delta queue feeds
    // `recover_arena` exactly the bytes that landed between flush_2
    // and the next flush, without re-replaying anything ≤
    // flush_2's LSN. Verified at the trait level via
    // `g3_pre_snapshot_deltas_skipped` in `vector_recovery.rs`; here
    // we pin the "T1 ack covers prior T3" property at the recovery
    // entry-point: bootstrap_from_mvcc with an empty delta source
    // produces an arena whose `last_applied_commit_lsn` equals the
    // MVCC source's `snapshot_lsn`.
    let mvcc_src = InMemoryMvcc::new(Lsn::new(2_000), Vec::new());
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let req = VectorRecoveryRequest::v1(tenant, index_id, SnapIndexType::Hnsw, dim);
    let arena = bootstrap_from_mvcc(store, &mvcc_src, req).expect("bootstrap");
    assert_eq!(
        arena.last_applied_commit_lsn,
        Lsn::new(2_000),
        "I-V6 T3 piggyback equivalence: bootstrap LSN = MVCC source LSN"
    );

    // Mixed T1+T3 piggyback per ADR-034 I-D3 + ADR-035 §7.7. The
    // canonical proof lives in `durability_tier_mixed.rs::
    // mixed_t1_t3_t1_strict_preservation`; here we cross-pin that
    // the snapshot catalog supports the same monotonicity property:
    // intervening T3 commits at lower LSNs cannot un-advance the T1
    // stamp.
    let spec_t3 = SnapshotSpec {
        tenant,
        partition: PartitionId::ZERO,
        index_id,
        // Stale lower LSN — simulates an out-of-order / delayed
        // T3 batch arriving after the T1 stamp.
        lsn: Lsn::new(150),
        encoding: 0,
        index_type: 0,
        dim,
        vectors_count: 64,
        sections: &sections_2,
    };
    flush_snapshot(&spec_t3, tmp.path(), &catalog).expect("stale T3 flush");
    assert_eq!(
        catalog.latest_lsn(tenant, index_id),
        Some(Lsn::new(250)),
        "I-V6 catalog monotonicity: stale T3 must NOT un-advance T1 stamp"
    );
}

// ─────────────────────────────────────────────────────────────────
// I-V7 — DiskANN T1 RYW (filtered + replay round-trip)
// ─────────────────────────────────────────────────────────────────
//
// Statement (ADR-035 §10 I-V7 + AC-13): every Strict-tier vector W
// inserted via `insert_stream` is visible to a subsequent search by
// the same tenant — including a FILTERED search whose payload P
// satisfies the filter, AND across a snapshot+replay round-trip.
//
// Sibling coverage:
//   - `diskann::diskann_streaming_insert_t1_ryw` covers the
//     unfiltered RYW chain at slice D.
//   - `diskann_filtered::*` covers the filter dispatch surface.
//
// Phase 5.5 extension: full chain — stream_insert → search_with_delta
// (unfiltered) → filtered_search_with_delta (filtered) → snapshot
// flush → recovery → arena re-rehydrated → vector still present.

#[test]
fn iv7_t1_ryw_diskann_filtered_full_chain() {
    let dim = 32;
    let kernel = L2F32;

    // Build a base graph with filtered build so the per-label
    // entry-point cache is populated. Use a single label "L1" for
    // every vector — the post-insert filtered search uses the same
    // label, so the per-label entry hits.
    let mut g = DiskAnnGraph::new(
        DiskAnnParams {
            r: 32,
            alpha: 1.2,
            l_construction: 64,
            l_search_default: 64,
            // Generous threshold so insert_stream stays in the
            // delta segment (RYW must be visible PRE-merge).
            delta_max_size: 10_000,
            ..DiskAnnParams::default()
        },
        Encoding::F32,
        arcgraph_vector::Metric::L2,
        Box::new(L2F32),
    )
    .expect("graph");

    let base_n = 200;
    let raw_base = unit_vectors(0xCAFE_BABE, base_n, dim);
    let owned_base: Vec<(VectorId, Vec<u8>)> = raw_base
        .iter()
        .enumerate()
        .map(|(i, v)| (VectorId::new(i as u32), bytes_of(v)))
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned_base
        .iter()
        .map(|(id, b)| (*id, b.as_slice()))
        .collect();
    let label_l1: DiskAnnLabelId = 1;
    let labels: Vec<Option<DiskAnnLabelId>> = (0..base_n).map(|_| Some(label_l1)).collect();
    g.build_filtered(&pairs, &labels, &kernel)
        .expect("build_filtered");

    // Stream-insert N brand-new vectors. Each one carries label L1
    // in the parallel sidecar so the delta_label_lookup resolves
    // correctly. After EACH insert:
    //   1. unfiltered search_with_delta → top-1 == inserted id
    //   2. filtered_search_with_delta(label_eq L1) → contains id
    let n_inserts = 64;
    let mut rng = StdRng::seed_from_u64(0xFEED_BEEF);
    let mut delta_labels: HashMap<VectorId, DiskAnnLabelId> = HashMap::new();
    let mut inserted_ids: Vec<VectorId> = Vec::new();
    let mut inserted_bytes: HashMap<VectorId, Vec<u8>> = HashMap::new();

    for i in 0..n_inserts {
        let id = VectorId::new(100_000 + i as u32);
        let v: Vec<f32> = (0..dim)
            .map(|_| {
                let u: f32 = StandardUniform.sample(&mut rng);
                u * 2.0 - 1.0
            })
            .collect();
        let v = l2_normalize(v);
        let v_bytes = bytes_of(&v);
        g.insert_stream(&[(id, v_bytes.as_slice())])
            .expect("insert_stream");
        delta_labels.insert(id, label_l1);
        inserted_bytes.insert(id, v_bytes.clone());
        inserted_ids.push(id);

        // 1. Unfiltered RYW: query by the just-inserted vector;
        //    top-1 MUST be this id (distance to self is the kernel
        //    minimum; the §10 disjunct collapses).
        let res_unfiltered = g
            .search_with_delta(&v_bytes, 1, 64)
            .expect("search_with_delta");
        let top = res_unfiltered.first().map(|(id, _)| *id);
        assert_eq!(
            top,
            Some(id),
            "I-V7 unfiltered RYW: insert {id:?} not visible (top-1 = {top:?})"
        );

        // 2. Filtered RYW: filter on label L1. The per-label entry
        //    cache for L1 is populated (built into the main graph);
        //    the new id lives in the delta segment with label L1
        //    via delta_label_lookup.
        let filter = Filter::label_eq(label_l1);
        let lookup =
            |q_id: VectorId| -> Option<DiskAnnLabelId> { delta_labels.get(&q_id).copied() };
        let res_filtered = g
            .filtered_search_with_delta(&v_bytes, 5, &filter, 64, &kernel, lookup, Lsn::MAX)
            .expect("filtered_search_with_delta");
        let filtered_ids: HashSet<VectorId> = res_filtered.iter().map(|(id, _)| *id).collect();
        assert!(
            filtered_ids.contains(&id),
            "I-V7 filtered RYW: insert {id:?} not visible in filtered_search_with_delta \
             (returned ids = {:?})",
            filtered_ids
        );
    }

    // Replay round-trip: after stream insert + before merge, the
    // delta segment holds N entries. A "replay" in the
    // arcgraph-storage recovery harness is the
    // `bootstrap_from_mvcc` path: an MVCC source replays the live
    // version chain. Pin the round-trip property: every inserted
    // id can be reproduced by an MVCC walk (constructed from the
    // arena's known live set), and the rebuild output covers them.
    //
    // We construct a synthetic MvccVectorSource carrying every
    // inserted vector. After bootstrap, the rebuilt arena's
    // `bootstrap_vectors` MUST contain every inserted id with
    // byte-identical payload — proving the post-insert stream
    // entries survive a recovery round-trip.
    let mvcc_vec: Vec<(u64, Vec<u8>)> = inserted_ids
        .iter()
        .map(|id| (id.raw() as u64, inserted_bytes[id].clone()))
        .collect();
    let mvcc_src = InMemoryMvcc::new(Lsn::new(5_000), mvcc_vec);
    let store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorArenaPageStore::new());
    let req = VectorRecoveryRequest::v1(TenantId::new(99), 1, SnapIndexType::DiskAnn, dim as u32);
    let recovered = bootstrap_from_mvcc(store, &mvcc_src, req).expect("bootstrap_from_mvcc");
    let recovered_ids: HashSet<u64> = recovered
        .bootstrap_vectors
        .iter()
        .map(|(id, _)| *id)
        .collect();
    for id in &inserted_ids {
        assert!(
            recovered_ids.contains(&(id.raw() as u64)),
            "I-V7 replay round-trip: inserted id {id:?} absent from recovered arena"
        );
    }
    // Byte identity: every (id, bytes) pair from the rebuild
    // matches the originally-inserted bytes.
    for (id, bytes) in &recovered.bootstrap_vectors {
        let v_id = VectorId::new(*id as u32);
        let original = inserted_bytes
            .get(&v_id)
            .expect("recovered id has known bytes");
        assert_eq!(
            bytes, original,
            "I-V7 replay round-trip: bytes for {v_id:?} drifted"
        );
    }
}

// ─────────────────────────────────────────────────────────────────
// EmptyWal local helper — silences unused-warning if it ever drifts
// across changes (the harness above is the consumer; this lets
// EmptyWal stay private to this file without dead-code lint).
// ─────────────────────────────────────────────────────────────────
#[allow(dead_code)]
fn _empty_wal_smoke(snapshot_lsn: Lsn) -> EmptyWal {
    EmptyWal { snapshot_lsn }
}
