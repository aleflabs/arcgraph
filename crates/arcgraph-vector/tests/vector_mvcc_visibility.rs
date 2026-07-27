//! Path-A boundary tests for ADR-041 §D-3a: MVCC visibility
//! windowing on vector backends (HNSW + DiskANN).
//!
//! Mirrors `crates/arcgraph-bm25/tests/bm25_mvcc_visibility.rs`
//! — the BM25 reference implementation. Vector entries gain
//! `commit_lsn` / `expired_lsn` per-vector tracking (HNSW via
//! `Payload`, DiskANN via parallel slot-indexed arrays per
//! ADR-041 §D-3a). The visibility filter applied at search time:
//!
//!     visible iff `commit_lsn ≤ read_lsn ∧ read_lsn < expired_lsn`
//!
//! At v1.0 every live vector has `expired_lsn = Lsn::MAX` (in-
//! place upsert; no version chain — same posture as BM25 §D-2).
//! The test surface focuses on the `commit_lsn ≤ read_lsn` half;
//! the upper-bound clause's saturation behavior at `read_lsn =
//! Lsn::MAX` is also pinned.
//!
//! PINS (each exercised on BOTH HNSW and DiskANN paths):
//! - `*_visibility_max_expired_visible_at_any_read_lsn` —
//!   `expired_lsn = MAX` (live) is visible at any read_lsn,
//!   including `Lsn::MAX` (saturating-add boundary preserved).
//! - `*_visibility_pre_commit_invisible` — a vector at
//!   `commit_lsn = 10` MUST NOT be visible at `read_lsn = 5`.
//! - `*_visibility_at_expire_invisible` — a vector at
//!   `expired_lsn = 20` MUST NOT be visible at `read_lsn = 20`
//!   (exclusive upper bound — `read_lsn < expired_lsn`).
//! - `*_visibility_disjoint_snapshots` — vectors A and B
//!   committed at distinct LSNs N and M (with N < M); a read at
//!   `read_lsn = N` sees only A; a read at `read_lsn = M`
//!   sees both.
//!
//! Failure of any pin is a *contract* break, not a test bug —
//! the cross-substrate snapshot-isolation guarantee depends on
//! these invariants being preserved across vector + community +
//! BM25.

use arcgraph_core::{LabelId, Lsn, TenantId};
use arcgraph_vector::diskann::{DiskAnnGraph, DiskAnnLabelId, DiskAnnParams};
use arcgraph_vector::distance::L2F32;
use arcgraph_vector::hnsw::{FilteredHnsw, HnswParams, Payload};
use arcgraph_vector::ids::VectorId;
use arcgraph_vector::{Encoding, Filter, Metric};

fn bytes_of(v: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(v).to_vec()
}

// ─── HNSW path ───────────────────────────────────────────────────

fn build_hnsw(payloads: &[(VectorId, Vec<f32>, Payload)]) -> FilteredHnsw {
    let mut g = FilteredHnsw::new(HnswParams::default(), 4, &L2F32);
    for (id, v, p) in payloads {
        g.filtered_insert(*id, &bytes_of(v), p.clone(), &L2F32)
            .expect("insert");
    }
    g
}

/// PIN: ADR-041 §D-3a — `expired_lsn = MAX` (the v1.0
/// live-doc invariant) is visible at every read_lsn including
/// `Lsn::MAX`. The saturating_add on the upper bound stays
/// stable at the MAX-MAX boundary (mirror of BM25 §D-2 +
/// amendment-01 saturation semantic).
#[test]
fn hnsw_visibility_max_expired_visible_at_any_read_lsn() {
    let payload =
        Payload::with_labels(vec![LabelId::new(1)]).with_lsn_window(Lsn::new(10), Lsn::MAX);
    let g = build_hnsw(&[(VectorId::new(1), vec![1.0, 0.0, 0.0, 0.0], payload)]);

    // Read at every snapshot ≥ commit_lsn must see the doc.
    let read_lsns = [
        Lsn::new(10),           // exact commit
        Lsn::new(11),           // just after
        Lsn::new(1_000_000),    // far future
        Lsn::new(u64::MAX - 1), // near MAX
        Lsn::MAX,               // MAX boundary
    ];
    for read_lsn in read_lsns {
        let r = g
            .filtered_search(
                &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
                5,
                &Filter::LabelIn(vec![LabelId::new(1)]),
                10,
                &L2F32,
                read_lsn,
            )
            .expect("search");
        assert_eq!(
            r.len(),
            1,
            "PIN: expired_lsn=MAX must be visible at read_lsn={} (got {} hits)",
            read_lsn.raw(),
            r.len(),
        );
        assert_eq!(
            r[0].0,
            VectorId::new(1),
            "PIN: round-tripped id must equal the inserted id",
        );
    }
}

/// PIN: ADR-041 §D-3a — `commit_lsn ≤ read_lsn` is the lower-
/// bound clause. A vector with `commit_lsn = 10` MUST NOT
/// surface to a reader at `read_lsn = 5`. Mirror of BM25
/// §D-3 / `bm25_mvcc_visibility::reader_at_older_lsn_excludes_post_lsn_doc`.
#[test]
fn hnsw_visibility_pre_commit_invisible() {
    let payload =
        Payload::with_labels(vec![LabelId::new(1)]).with_lsn_window(Lsn::new(10), Lsn::MAX);
    let g = build_hnsw(&[(VectorId::new(1), vec![1.0, 0.0, 0.0, 0.0], payload)]);

    let r = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(5),
        )
        .expect("search at stale LSN");
    assert!(
        r.is_empty(),
        "PIN: ADR-041 §D-3a — vector with commit_lsn=10 MUST NOT be \
         visible at read_lsn=5 (got {} hits: {r:?})",
        r.len(),
    );
}

/// PIN: ADR-041 §D-3a — `read_lsn < expired_lsn` (EXCLUSIVE on
/// the expired side; the row at `expired_lsn = expire` is
/// invisible at `read_lsn = expire`).
#[test]
fn hnsw_visibility_at_expire_invisible() {
    let payload =
        Payload::with_labels(vec![LabelId::new(1)]).with_lsn_window(Lsn::new(10), Lsn::new(20));
    let g = build_hnsw(&[(VectorId::new(1), vec![1.0, 0.0, 0.0, 0.0], payload)]);

    // Visible at read_lsn = 19 (just before expiry).
    let r19 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(19),
        )
        .expect("search at lsn=19");
    assert_eq!(
        r19.len(),
        1,
        "PIN: at read_lsn=19 (before expire 20) the vector must be visible",
    );

    // INVISIBLE at read_lsn = 20 (exact expiry — exclusive upper).
    let r20 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(20),
        )
        .expect("search at lsn=20");
    assert!(
        r20.is_empty(),
        "PIN: ADR-041 §D-3a — at read_lsn=20 (== expired_lsn) the \
         vector MUST be invisible (got {} hits)",
        r20.len(),
    );

    // Stays invisible past the expiry.
    let r21 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(21),
        )
        .expect("search at lsn=21");
    assert!(
        r21.is_empty(),
        "PIN: ADR-041 §D-3a — past expiry stays invisible (got {} hits)",
        r21.len(),
    );
}

/// PIN: ADR-041 §D-3a — disjoint snapshots see disjoint result
/// sets. A vector A at `commit_lsn = 10` and B at `commit_lsn =
/// 20`. A read at `read_lsn = 15` sees only A. A read at
/// `read_lsn = 25` sees both A and B.
#[test]
fn hnsw_visibility_disjoint_snapshots() {
    let pa = Payload::with_labels(vec![LabelId::new(1)]).with_lsn_window(Lsn::new(10), Lsn::MAX);
    let pb = Payload::with_labels(vec![LabelId::new(1)]).with_lsn_window(Lsn::new(20), Lsn::MAX);
    let g = build_hnsw(&[
        (VectorId::new(1), vec![1.0, 0.0, 0.0, 0.0], pa),
        (VectorId::new(2), vec![0.99, 0.01, 0.0, 0.0], pb),
    ]);

    // Snapshot at LSN=15: only A is committed.
    let r_15 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(15),
        )
        .expect("search at lsn=15");
    assert_eq!(
        r_15.len(),
        1,
        "PIN: at read_lsn=15 only vector A is committed; got {} hits",
        r_15.len(),
    );
    assert_eq!(r_15[0].0, VectorId::new(1));

    // Snapshot at LSN=25: both A and B are committed.
    let r_25 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(25),
        )
        .expect("search at lsn=25");
    assert_eq!(
        r_25.len(),
        2,
        "PIN: at read_lsn=25 both vectors A and B are committed; got {} hits",
        r_25.len(),
    );

    // Snapshot at LSN=5 (before either commit): empty.
    let r_5 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::LabelIn(vec![LabelId::new(1)]),
            10,
            &L2F32,
            Lsn::new(5),
        )
        .expect("search at lsn=5");
    assert!(
        r_5.is_empty(),
        "PIN: read_lsn=5 predates every commit; result must be empty (got {} hits)",
        r_5.len(),
    );
}

// ─── DiskANN path ────────────────────────────────────────────────

/// Build a DiskANN graph with a vector + label per index, then
/// stamp `(commit_lsn, expired_lsn)` per `set_lsn_window`. All
/// vectors carry label 1 so a single `Filter::label_eq(1)` is
/// the universal predicate.
fn build_diskann(entries: &[(VectorId, Vec<f32>, Lsn, Lsn)]) -> DiskAnnGraph {
    let mut g = DiskAnnGraph::new(
        DiskAnnParams {
            r: 4,
            alpha: 1.2,
            l_construction: 16,
            l_search_default: 16,
            ..DiskAnnParams::default()
        },
        Encoding::F32,
        Metric::L2,
        Box::new(L2F32),
    )
    .expect("graph");
    let owned: Vec<(VectorId, Vec<u8>)> = entries
        .iter()
        .map(|(id, v, _, _)| (*id, bytes_of(v)))
        .collect();
    let pairs: Vec<(VectorId, &[u8])> = owned.iter().map(|(id, b)| (*id, b.as_slice())).collect();
    let labels: Vec<Option<DiskAnnLabelId>> = entries.iter().map(|_| Some(1u32)).collect();
    g.build_filtered(&pairs, &labels, &L2F32).expect("build");
    for (id, _, commit, expired) in entries {
        g.set_lsn_window(*id, *commit, *expired)
            .expect("set_lsn_window");
    }
    g
}

/// PIN (DiskANN mirror of HNSW): `expired_lsn = MAX` is
/// visible at every read_lsn including `Lsn::MAX`.
#[test]
fn diskann_visibility_max_expired_visible_at_any_read_lsn() {
    let g = build_diskann(&[(
        VectorId::new(1),
        vec![1.0, 0.0, 0.0, 0.0],
        Lsn::new(10),
        Lsn::MAX,
    )]);

    let read_lsns = [
        Lsn::new(10),
        Lsn::new(11),
        Lsn::new(1_000_000),
        Lsn::new(u64::MAX - 1),
        Lsn::MAX,
    ];
    for read_lsn in read_lsns {
        let r = g
            .filtered_search(
                &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
                5,
                &Filter::label_eq(1u32),
                32,
                &L2F32,
                read_lsn,
            )
            .expect("search");
        assert_eq!(
            r.len(),
            1,
            "PIN (DiskANN): expired_lsn=MAX must be visible at read_lsn={} (got {} hits)",
            read_lsn.raw(),
            r.len(),
        );
        assert_eq!(r[0].0, VectorId::new(1));
    }
}

/// PIN (DiskANN mirror): `commit_lsn ≤ read_lsn` is the lower
/// bound. A vector at `commit_lsn = 10` is invisible at
/// `read_lsn = 5`.
#[test]
fn diskann_visibility_pre_commit_invisible() {
    let g = build_diskann(&[(
        VectorId::new(1),
        vec![1.0, 0.0, 0.0, 0.0],
        Lsn::new(10),
        Lsn::MAX,
    )]);

    let r = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::label_eq(1u32),
            32,
            &L2F32,
            Lsn::new(5),
        )
        .expect("search at stale LSN");
    assert!(
        r.is_empty(),
        "PIN (DiskANN): vector at commit_lsn=10 MUST NOT be visible \
         at read_lsn=5 (got {} hits)",
        r.len(),
    );
}

/// PIN (DiskANN mirror): `read_lsn < expired_lsn` is EXCLUSIVE
/// on the expired side.
#[test]
fn diskann_visibility_at_expire_invisible() {
    let g = build_diskann(&[(
        VectorId::new(1),
        vec![1.0, 0.0, 0.0, 0.0],
        Lsn::new(10),
        Lsn::new(20),
    )]);

    let r19 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::label_eq(1u32),
            32,
            &L2F32,
            Lsn::new(19),
        )
        .expect("search lsn=19");
    assert_eq!(r19.len(), 1, "PIN (DiskANN): visible at read_lsn=19");

    let r20 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::label_eq(1u32),
            32,
            &L2F32,
            Lsn::new(20),
        )
        .expect("search lsn=20");
    assert!(
        r20.is_empty(),
        "PIN (DiskANN): at read_lsn=20 (== expired_lsn) MUST be invisible (got {} hits)",
        r20.len(),
    );
}

/// PIN (DiskANN mirror): disjoint snapshots return disjoint
/// vector sets.
#[test]
fn diskann_visibility_disjoint_snapshots() {
    let g = build_diskann(&[
        (
            VectorId::new(1),
            vec![1.0, 0.0, 0.0, 0.0],
            Lsn::new(10),
            Lsn::MAX,
        ),
        (
            VectorId::new(2),
            vec![0.99, 0.01, 0.0, 0.0],
            Lsn::new(20),
            Lsn::MAX,
        ),
    ]);

    // Snapshot at LSN=15: only A is committed.
    let r_15 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::label_eq(1u32),
            32,
            &L2F32,
            Lsn::new(15),
        )
        .expect("search at lsn=15");
    assert_eq!(r_15.len(), 1, "PIN (DiskANN): only A committed at lsn=15");
    assert_eq!(r_15[0].0, VectorId::new(1));

    // Snapshot at LSN=25: both committed.
    let r_25 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::label_eq(1u32),
            32,
            &L2F32,
            Lsn::new(25),
        )
        .expect("search at lsn=25");
    assert_eq!(
        r_25.len(),
        2,
        "PIN (DiskANN): both A and B committed at lsn=25",
    );

    // Snapshot at LSN=5: empty.
    let r_5 = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::label_eq(1u32),
            32,
            &L2F32,
            Lsn::new(5),
        )
        .expect("search at lsn=5");
    assert!(
        r_5.is_empty(),
        "PIN (DiskANN): read_lsn=5 predates every commit (got {} hits)",
        r_5.len(),
    );
}

// ─── Cross-cut: tenant + visibility composition ──────────────────

/// PIN: the visibility filter composes correctly with the
/// user-supplied predicate (`Filter::Tenant`). A vector
/// committed but not matching the tenant filter is invisible;
/// a vector matching the tenant but uncommitted at read_lsn is
/// also invisible. Mirror of BM25's tenant-isolation tests but
/// pinning the visibility-and-filter conjunction.
#[test]
fn hnsw_visibility_composes_with_tenant_filter() {
    let pa = Payload {
        tenant_id: Some(TenantId::new(1)),
        labels: vec![LabelId::new(1)],
        properties: Default::default(),
        commit_lsn: Lsn::new(10),
        expired_lsn: Lsn::MAX,
    };
    let pb = Payload {
        tenant_id: Some(TenantId::new(2)),
        labels: vec![LabelId::new(1)],
        properties: Default::default(),
        commit_lsn: Lsn::new(10),
        expired_lsn: Lsn::MAX,
    };
    let g = build_hnsw(&[
        (VectorId::new(1), vec![1.0, 0.0, 0.0, 0.0], pa),
        (VectorId::new(2), vec![0.99, 0.01, 0.0, 0.0], pb),
    ]);

    // Tenant 1 + read_lsn=15 → only vector 1 is visible.
    let r = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::Tenant(TenantId::new(1)),
            10,
            &L2F32,
            Lsn::new(15),
        )
        .expect("search");
    assert_eq!(
        r.len(),
        1,
        "PIN: tenant + commit composes; got {} hits",
        r.len()
    );
    assert_eq!(r[0].0, VectorId::new(1));

    // Tenant 1 + read_lsn=5 (pre-commit) → empty.
    let r_pre = g
        .filtered_search(
            &bytes_of(&[1.0, 0.0, 0.0, 0.0]),
            5,
            &Filter::Tenant(TenantId::new(1)),
            10,
            &L2F32,
            Lsn::new(5),
        )
        .expect("search pre-commit");
    assert!(
        r_pre.is_empty(),
        "PIN: tenant matches but pre-commit → empty"
    );
}
