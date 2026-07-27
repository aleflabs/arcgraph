//! Cross-substrate MVCC visibility integration test per ADR-041.
//!
//! Closes codex retrospective F-1 (2026-05-03; CONCERN-hard):
//! BM25 had snapshot isolation per ADR-039 §D-3; vector +
//! community + lowering carrier had ZERO. ADR-041 unified the
//! contract; this test exercises the end-to-end shape.
//!
//! ## Scenario
//!
//! Construct three substrate handles:
//!
//! 1. A BM25 service holding documents for a single tenant.
//! 2. A FilteredHnsw vector index with payload-stamped
//!    `(commit_lsn, expired_lsn)` windows.
//! 3. A BTreeMembershipIndex with per-(tenant, level) install
//!    history.
//!
//! Stamp each substrate's data at distinct LSNs (10, 20, 30).
//! Then read at three different snapshot LSNs:
//!
//! - **`read_lsn = 5`** (predates every install): every
//!   substrate returns empty.
//! - **`read_lsn = 15`** (between install 10 and 20): only the
//!   first generation of each substrate's data is visible.
//! - **`read_lsn = 35`** (past every install): every substrate
//!   returns the latest snapshot.
//!
//! ## What this pins
//!
//! - The three substrate APIs uniformly accept `read_lsn` as
//!   the last positional argument (mirrors BM25 convention from
//!   ADR-039 §D-3).
//! - The visibility-filter shape is identical across substrates:
//!   `commit_lsn ≤ read_lsn ∧ read_lsn < expired_lsn` for vector +
//!   BM25 (per-row); install-history binary search for community
//!   (per-install).
//! - A hybrid query at `read_lsn = N` routes into all three
//!   substrates with the SAME N → no silent snapshot mixing.
//!
//! ## What this does NOT pin
//!
//! - The M4-05 / M4-06 executor wiring (out of scope; that's
//!   M4-05's slice). This test exercises the substrate APIs
//!   directly to validate the contract.
//! - `Expand` (graph traversal) visibility — that's deferred per
//!   ADR-041 "Open questions / follow-ups".
//!
//! Failure of any pin is a *contract* break, not a test bug —
//! the cross-substrate snapshot-isolation guarantee depends on
//! these invariants holding across vector + community + BM25
//! (ADR-041).

use std::sync::Arc;

use arcgraph_bm25::{Bm25Service, IndexId as Bm25IndexId};
use arcgraph_community::{BTreeMembershipIndex, CommunityId, Level, MembershipIndex};
use arcgraph_core::{LabelId, Lsn, NodeId, TenantId};
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use arcgraph_vector::Filter;
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{FilteredHnsw, HnswParams, Payload};
use arcgraph_vector::ids::VectorId;
use tempfile::TempDir;

const TENANT: TenantId = TenantId::DEFAULT;
const LEVEL: Level = Level::FINEST;

fn bytes_of(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

/// Build a tri-substrate fixture:
///
/// - BM25: 2 docs ("alpha", "beta") committed at LSN=10 and
///   LSN=20 respectively.
/// - Vector: 2 vectors (NodeId=1 at commit_lsn=10, NodeId=2 at
///   commit_lsn=20).
/// - Community: 2 installs at LSN=10 and LSN=30. Install A
///   places node 1 in community 100 + node 2 in community 100;
///   install B re-classifies into community 200.
fn build_fixture() -> (
    TempDir,
    Arc<Bm25Service>,
    Arc<arcgraph_bm25::Bm25IndexHandle>,
    FilteredHnsw,
    Arc<BTreeMembershipIndex>,
) {
    // ── BM25 ────────────────────────────────────────────────
    let tmp = TempDir::new().expect("tempdir");
    let svc = Bm25Service::new(tmp.path().to_path_buf());
    let bm = svc
        .handle(TENANT, Bm25IndexId::DEFAULT_BM25)
        .expect("bm25 handle");
    bm.upsert_document(NodeId::new(1), "alpha keyword", Lsn::new(10))
        .expect("bm25 upsert at lsn=10");
    bm.upsert_document(NodeId::new(2), "beta keyword", Lsn::new(20))
        .expect("bm25 upsert at lsn=20");
    let trait_obj: Arc<dyn Bm25IndexStoreHandle> = svc.clone();
    trait_obj
        .commit_pending(TENANT)
        .expect("bm25 commit_pending");

    // ── Vector ─────────────────────────────────────────────
    let mut hnsw = FilteredHnsw::new(HnswParams::default(), 4, &L2F32);
    let p_node1 = Payload {
        labels: vec![LabelId::new(1)],
        ..Payload::default()
    }
    .with_lsn_window(Lsn::new(10), Lsn::MAX);
    let p_node2 = Payload {
        labels: vec![LabelId::new(1)],
        ..Payload::default()
    }
    .with_lsn_window(Lsn::new(20), Lsn::MAX);
    hnsw.filtered_insert(
        VectorId::new(1),
        &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
        p_node1,
        &L2F32,
    )
    .expect("vector insert at lsn=10");
    hnsw.filtered_insert(
        VectorId::new(2),
        &bytes_of(&[0.99, 0.01, 0.0, 0.0]),
        p_node2,
        &L2F32,
    )
    .expect("vector insert at lsn=20");

    // ── Community ───────────────────────────────────────────
    let community = Arc::new(BTreeMembershipIndex::new());
    community.install_level(
        TENANT,
        LEVEL,
        Lsn::new(10),
        &[
            (NodeId::new(1), CommunityId::new(100)),
            (NodeId::new(2), CommunityId::new(100)),
        ],
    );
    community.install_level(
        TENANT,
        LEVEL,
        Lsn::new(30),
        &[
            (NodeId::new(1), CommunityId::new(200)),
            (NodeId::new(2), CommunityId::new(200)),
        ],
    );

    (tmp, svc, bm, hnsw, community)
}

/// PIN: ADR-041 §D-1, §D-2, §D-3 — at `read_lsn = 5` (predates
/// every install across substrates), every substrate returns
/// empty. The cross-substrate visibility contract holds: the
/// hybrid query at this snapshot routes into all three with
/// `read_lsn=5` and gets a uniform empty result.
#[test]
fn cross_substrate_pre_install_returns_empty_everywhere() {
    let (_tmp, _svc, bm, hnsw, community) = build_fixture();
    let read_lsn = Lsn::new(5);

    let bm_hits = bm
        .search("alpha", 10, read_lsn)
        .expect("bm25 search at stale LSN");
    assert!(
        bm_hits.is_empty(),
        "PIN: ADR-039 §D-3 — BM25 at read_lsn=5 (pre-commit) is empty",
    );

    let v_hits = hnsw
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            read_lsn,
        )
        .expect("vector search");
    assert!(
        v_hits.is_empty(),
        "PIN: ADR-041 §D-3a — vector at read_lsn=5 (pre-commit) is empty",
    );

    let c_lookup = community
        .lookup(TENANT, NodeId::new(1), LEVEL, read_lsn)
        .expect("community lookup");
    assert_eq!(
        c_lookup, None,
        "PIN: ADR-041 §D-3b — community at read_lsn=5 (pre-install) is None",
    );
    let c_members = community
        .members(TENANT, CommunityId::new(100), LEVEL, read_lsn)
        .expect("community members");
    assert!(
        c_members.is_empty(),
        "PIN: community.members at pre-install LSN is empty",
    );
}

/// PIN: ADR-041 — at `read_lsn = 15` (between substrate
/// generations), each substrate returns its FIRST-GENERATION
/// snapshot:
///
/// - BM25: only "alpha" doc (committed at LSN=10) is visible;
///   "beta" (LSN=20) is invisible.
/// - Vector: only NodeId=1 (commit_lsn=10) is visible;
///   NodeId=2 (commit_lsn=20) is invisible.
/// - Community: install A is visible (community 100); install B
///   (LSN=30) is invisible.
#[test]
fn cross_substrate_mid_window_sees_first_generation_only() {
    let (_tmp, _svc, bm, hnsw, community) = build_fixture();
    let read_lsn = Lsn::new(15);

    let bm_hits = bm
        .search("alpha", 10, read_lsn)
        .expect("bm25 search at lsn=15");
    assert_eq!(
        bm_hits.len(),
        1,
        "PIN: BM25 at read_lsn=15 sees only the LSN=10 doc",
    );
    assert_eq!(bm_hits[0].0, NodeId::new(1));

    let bm_hits_beta = bm
        .search("beta", 10, read_lsn)
        .expect("bm25 search at lsn=15 for beta");
    assert!(
        bm_hits_beta.is_empty(),
        "PIN: BM25 at read_lsn=15 does NOT see the LSN=20 'beta' doc",
    );

    let v_hits = hnsw
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            read_lsn,
        )
        .expect("vector search");
    assert_eq!(
        v_hits.len(),
        1,
        "PIN: vector at read_lsn=15 sees only the LSN=10 vector",
    );
    assert_eq!(v_hits[0].0, VectorId::new(1));

    // Community: install A places nodes 1 + 2 in community 100;
    // install B (at LSN=30) re-places into community 200.
    // At read_lsn=15 we see install A only.
    let c1 = community
        .lookup(TENANT, NodeId::new(1), LEVEL, read_lsn)
        .expect("community lookup")
        .expect("node 1 should be in install A's community");
    assert_eq!(
        c1,
        CommunityId::new(100),
        "PIN: community at read_lsn=15 returns install A's classification",
    );

    let c_members = community
        .members(TENANT, CommunityId::new(100), LEVEL, read_lsn)
        .expect("members");
    assert_eq!(
        c_members,
        vec![NodeId::new(1), NodeId::new(2)],
        "PIN: install A has both nodes in community 100",
    );

    // Install B's community (200) has no members at read_lsn=15.
    let c_b_members = community
        .members(TENANT, CommunityId::new(200), LEVEL, read_lsn)
        .expect("members");
    assert!(
        c_b_members.is_empty(),
        "PIN: install B's community 200 is invisible at read_lsn=15",
    );
}

/// PIN: ADR-041 — at `read_lsn = 35` (past every install), each
/// substrate returns its latest snapshot. Both BM25 docs visible,
/// both vectors visible, install B's community 200 visible.
#[test]
fn cross_substrate_post_window_sees_latest_generation() {
    let (_tmp, _svc, bm, hnsw, community) = build_fixture();
    let read_lsn = Lsn::new(35);

    let bm_hits = bm.search("alpha", 10, read_lsn).expect("bm25 alpha");
    assert_eq!(bm_hits.len(), 1);
    assert_eq!(bm_hits[0].0, NodeId::new(1));

    let bm_hits_beta = bm.search("beta", 10, read_lsn).expect("bm25 beta");
    assert_eq!(bm_hits_beta.len(), 1);
    assert_eq!(bm_hits_beta[0].0, NodeId::new(2));

    let v_hits = hnsw
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            read_lsn,
        )
        .expect("vector");
    assert_eq!(v_hits.len(), 2, "PIN: at read_lsn=35 both vectors visible",);

    // Community: install B re-placed nodes 1 + 2 into community
    // 200; community 100 is empty at this snapshot.
    assert_eq!(
        community
            .lookup(TENANT, NodeId::new(1), LEVEL, read_lsn)
            .expect("ok"),
        Some(CommunityId::new(200)),
        "PIN: at read_lsn=35 install B's classification wins",
    );
    assert!(
        community
            .members(TENANT, CommunityId::new(100), LEVEL, read_lsn)
            .expect("ok")
            .is_empty(),
    );
    assert_eq!(
        community
            .members(TENANT, CommunityId::new(200), LEVEL, read_lsn)
            .expect("ok"),
        vec![NodeId::new(1), NodeId::new(2)],
    );
}

/// PIN: ADR-041 §D-1 + §D-3 — uniform "read-latest" behavior at
/// the safe v1.0 boundary (`read_lsn = u64::MAX - 1`). The BM25
/// substrate's `debug_assert!(read != u64::MAX, ...)` at
/// `crates/arcgraph-bm25/src/mvcc.rs:57` (per ADR-039 §D-3 +
/// M3.b codex CONCERN-soft #6 retro review, 2026-05-03)
/// deliberately surfaces the `saturating_add(1)` semantic gap at
/// `read_lsn = u64::MAX` for v1.1 LSN-width changes (e.g.,
/// per-tenant 32-bit LSNs). The "uniform across substrates"
/// promise of ADR-041 §D-3 therefore holds at any `read_lsn`
/// within the safe v1.0 range (`read_lsn < u64::MAX`); this test
/// pins that range's upper edge.
///
/// Vector + community substrates ARE separately tested at the
/// absolute `Lsn::MAX` in their per-substrate visibility tests
/// (`vector_mvcc_visibility` + `community_mvcc_visibility`); the
/// asymmetry at exactly `u64::MAX` is pinned in the next test
/// (`cross_substrate_saturating_max_boundary_panics_in_bm25_…`).
///
/// Mirrors the BM25 amendment's
/// `expired_lsn_is_structurally_max_at_v1` pin which also reads
/// at `Lsn::new(u64::MAX - 1)` for the same reason.
#[test]
fn cross_substrate_read_at_safe_max_sees_latest_uniformly() {
    let (_tmp, _svc, bm, hnsw, community) = build_fixture();
    // The safe v1.0 boundary per ADR-039 §D-3 + M3.b CONCERN-soft #6
    // (the BM25 debug_assert excludes `u64::MAX` in debug builds).
    let read_lsn = Lsn::new(u64::MAX - 1);

    // BM25: both docs visible. The saturating_add(1) saturates
    // `expired_lower` to `u64::MAX`; every v1.0 doc has
    // `expired_lsn = u64::MAX` so the [MAX,MAX] expired-range
    // admits all live docs.
    let bm_alpha = bm.search("alpha", 10, read_lsn).expect("bm25 alpha");
    let bm_beta = bm.search("beta", 10, read_lsn).expect("bm25 beta");
    assert_eq!(bm_alpha.len(), 1);
    assert_eq!(bm_beta.len(), 1);

    // Vector: both visible.
    let v_hits = hnsw
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            read_lsn,
        )
        .expect("vector");
    assert_eq!(v_hits.len(), 2);

    // Community: latest install (B) wins.
    assert_eq!(
        community
            .lookup(TENANT, NodeId::new(1), LEVEL, read_lsn)
            .expect("ok"),
        Some(CommunityId::new(200)),
    );
}

/// PIN: ADR-041 §D-3a + §D-3b + ADR-039 §D-3 (per M3.b codex
/// CONCERN-soft #6 retro review, 2026-05-03) — saturating-add
/// boundary asymmetry across substrates at `read_lsn = u64::MAX`:
///
/// - **BM25** (`crates/arcgraph-bm25/src/mvcc.rs:57`):
///   `debug_assert!(read != u64::MAX, …)` deliberately PANICS in
///   debug builds to surface the `saturating_add(1)` semantic gap
///   for downstream LSN-width changes (e.g., per-tenant 32-bit
///   LSNs at v1.1+). In release builds the saturating-add path
///   silently saturates `expired_lower` to `u64::MAX`; the gap
///   exists without surfacing.
/// - **Vector** (`slot_visible_at`,
///   `crates/arcgraph-vector/src/diskann/graph.rs:760`) handles
///   `u64::MAX` cleanly: `expired_lower = u64::MAX` (saturated),
///   the check `expired >= expired_lower` is satisfied for every
///   v1.0 doc with `expired_lsn = MAX`. No debug_assert.
/// - **Community** (`BTreeMembershipIndex::lookup`,
///   per-install binary search on `Lsn`) handles `u64::MAX`
///   cleanly: returns the latest install's classification.
///
/// The "uniform across substrates" promise of ADR-041 §D-3 is
/// therefore qualified — uniform within the safe v1.0 range
/// (`read_lsn < u64::MAX`), with BM25 surfacing the boundary in
/// debug per the codex CONCERN-soft #6 retro for downstream
/// LSN-width follow-ups.
///
/// Pattern mirrors the M3.b codex V2 catch_unwind discipline in
/// `crates/arcgraph-vector/tests/multi_tenant_proptest.rs`.
#[test]
fn cross_substrate_saturating_max_boundary_panics_in_bm25_per_adr_039_amendment_01() {
    let (_tmp, _svc, bm, hnsw, community) = build_fixture();

    // ── Vector + community at `Lsn::MAX` work cleanly in BOTH
    //    debug + release builds (no debug_assert; the saturating
    //    upper-bound keeps the visibility predicate total).
    let v_hits_max = hnsw
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            10,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::MAX,
        )
        .expect("vector at u64::MAX must not panic (no debug_assert)");
    assert_eq!(
        v_hits_max.len(),
        2,
        "PIN: ADR-041 §D-3a — vector at Lsn::MAX returns all live docs",
    );

    let c_max = community
        .lookup(TENANT, NodeId::new(1), LEVEL, Lsn::MAX)
        .expect("community at u64::MAX must not panic");
    assert_eq!(
        c_max,
        Some(CommunityId::new(200)),
        "PIN: ADR-041 §D-3b — community at Lsn::MAX returns latest install",
    );

    // ── BM25 at `Lsn::MAX` — debug-vs-release split per
    //    `crates/arcgraph-bm25/src/mvcc.rs:57` debug_assert.
    if cfg!(debug_assertions) {
        // Debug build: BM25's debug_assert! fires before the
        // search returns. Suppress the panic-handler stderr noise
        // during the catch_unwind so the test output stays clean,
        // then assert the panic payload names the
        // `saturating_add(1) semantic gap` exactly as ADR-039 §D-3
        // requires.
        let prior_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bm.search("alpha", 10, Lsn::MAX)
        }));
        std::panic::set_hook(prior_hook);

        match result {
            Ok(_) => panic!(
                "PIN: ADR-039 §D-3 + M3.b CONCERN-soft #6 — BM25 debug_assert \
                 at read_lsn = u64::MAX MUST fire in debug builds",
            ),
            Err(payload) => {
                let msg: String = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| {
                        payload
                            .downcast_ref::<&'static str>()
                            .map(|s| (*s).to_owned())
                    })
                    .unwrap_or_default();
                assert!(
                    msg.contains("saturating_add(1) semantic gap"),
                    "PIN: ADR-039 §D-3 — expected debug_assert! message naming the \
                     saturating_add(1) semantic gap; got: {msg}",
                );
            }
        }
    } else {
        // Release build: BM25's debug_assert! is compiled out; the
        // saturating-add path saturates `expired_lower` to MAX
        // silently. v1.0 docs (expired_lsn = MAX) remain visible.
        let bm_alpha = bm
            .search("alpha", 10, Lsn::MAX)
            .expect("bm25 at u64::MAX in release: saturating-add must keep search total");
        assert!(
            !bm_alpha.is_empty(),
            "PIN: ADR-039 §D-3 — release-build BM25 at u64::MAX saturates and \
             returns v1.0 docs (expired_lsn = MAX); the gap exists silently per \
             the codex CONCERN-soft #6 retro for v1.1 LSN-width follow-ups",
        );
    }
}
