//! Round-trip integration test for [`BTreeMembershipIndex`].
//!
//! Per the M3.d-1 task #3 prompt: install an assignment for tenant
//! A, verify membership() matches, members() returns sorted,
//! rank_by_seeds() returns correct ordering. Replace install for
//! same (tenant, level) and verify the LATEST snapshot reflects
//! the new assignment when read at `Lsn::MAX`.
//!
//! Per ADR-041 §D-3b each install is tagged with a monotonic
//! `install_lsn`; lookups consult the most-recent install ≤ the
//! query's `read_lsn`. The tests in this file pin behavior at the
//! latest snapshot via `Lsn::MAX`; the per-snapshot history shape
//! is exercised in `tests/community_mvcc_visibility.rs`.

use arcgraph_community::{
    BTreeMembershipIndex, CommunityId, CommunityIndexHandle, CommunityIndexId, Level,
    MembershipIndex,
};
use arcgraph_core::{Lsn, NodeId, TenantId};
use std::sync::Arc;

fn pairs(input: &[(u64, u64)]) -> Vec<(NodeId, CommunityId)> {
    input
        .iter()
        .map(|&(n, c)| (NodeId::new(n), CommunityId::new(c)))
        .collect()
}

#[test]
fn install_lookup_members_round_trip() {
    let idx: Arc<dyn MembershipIndex> = Arc::new(BTreeMembershipIndex::new());

    // Two communities at level 0:
    //   community 0: {0, 1, 2}
    //   community 1: {3, 4, 5, 6}
    let assignment = pairs(&[(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1), (6, 1)]);

    // Cast back to the concrete type to call the bulk install
    // entry point (which is not part of the trait).
    let concrete = idx.clone();
    // We installed via the concrete type before wrapping; but
    // since we wrapped via Arc::new we need the concrete first.
    drop(concrete);
    let concrete = BTreeMembershipIndex::new();
    concrete.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(1), &assignment);
    let idx: Arc<dyn MembershipIndex> = Arc::new(concrete);

    // Forward (membership) lookups.
    for &(n, c) in &assignment {
        let got = idx
            .lookup(TenantId::DEFAULT, n, Level::FINEST, Lsn::MAX)
            .expect("lookup ok");
        assert_eq!(got, Some(c), "node {} → community {}", n.raw(), c.raw());
    }
    // A missing node returns None, not an error.
    let missing = idx
        .lookup(TenantId::DEFAULT, NodeId::new(999), Level::FINEST, Lsn::MAX)
        .expect("lookup ok");
    assert_eq!(missing, None);

    // Reverse (members) returns ascending NodeId order.
    let m0 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(0),
            Level::FINEST,
            Lsn::MAX,
        )
        .expect("members ok");
    assert_eq!(m0, vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)]);
    let m1 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(1),
            Level::FINEST,
            Lsn::MAX,
        )
        .expect("members ok");
    assert_eq!(
        m1,
        vec![
            NodeId::new(3),
            NodeId::new(4),
            NodeId::new(5),
            NodeId::new(6)
        ]
    );

    // Rank by seeds: 2 seeds in community 0 (size 3), 1 in
    // community 1 (size 4). Scores: 2/3 ≈ 0.667 vs 1/4 = 0.25.
    let seeds = [NodeId::new(0), NodeId::new(1), NodeId::new(3)];
    let ranking = idx
        .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 5, Lsn::MAX)
        .expect("rank ok");
    assert_eq!(ranking.len(), 2);
    assert_eq!(ranking[0].0, CommunityId::new(0));
    assert!((ranking[0].1 - 2.0 / 3.0).abs() < 1e-6);
    assert_eq!(ranking[1].0, CommunityId::new(1));
    assert!((ranking[1].1 - 0.25).abs() < 1e-6);
}

#[test]
fn install_versions_each_partition() {
    // Per ADR-041 §D-3b, successive installs at distinct LSNs
    // are versioned: a `read_lsn` between them sees the older
    // snapshot, a `read_lsn` after the second sees the newer.
    // The pre-ADR-041 "replace in place" contract is gone.
    let idx = BTreeMembershipIndex::new();
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 0), (1, 0), (2, 0)]),
    );

    // Re-install at a later LSN: community 0 shrinks; node 2
    // moves to community 1; node 3 (new) joins community 0.
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(20),
        &pairs(&[(0, 0), (1, 0), (2, 1), (3, 0)]),
    );

    // At Lsn::MAX (the latest snapshot is visible).
    assert_eq!(
        idx.lookup(TenantId::DEFAULT, NodeId::new(2), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(1))
    );
    assert_eq!(
        idx.lookup(TenantId::DEFAULT, NodeId::new(3), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(0))
    );

    // Reverse reflects the new assignment.
    let m0 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(0),
            Level::FINEST,
            Lsn::MAX,
        )
        .expect("ok");
    assert_eq!(m0, vec![NodeId::new(0), NodeId::new(1), NodeId::new(3)]);
    let m1 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(1),
            Level::FINEST,
            Lsn::MAX,
        )
        .expect("ok");
    assert_eq!(m1, vec![NodeId::new(2)]);

    // ADR-041 §D-3b: at read_lsn=15 (between the two installs)
    // the OLDER snapshot is visible.
    assert_eq!(
        idx.lookup(
            TenantId::DEFAULT,
            NodeId::new(2),
            Level::FINEST,
            Lsn::new(15)
        )
        .expect("ok"),
        Some(CommunityId::new(0)),
        "PIN: read at lsn=15 sees the lsn=10 snapshot's assignment for node 2",
    );
    let m0_old = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(0),
            Level::FINEST,
            Lsn::new(15),
        )
        .expect("ok");
    assert_eq!(
        m0_old,
        vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
        "PIN: lsn=15 community 0 = {{0, 1, 2}} (lsn=10 snapshot)",
    );
}

#[test]
fn round_trip_via_handle() {
    // Same shape but routed through the public CommunityIndexHandle
    // API to confirm the handle's three methods delegate correctly
    // to the BTreeMembershipIndex.
    let idx = Arc::new(BTreeMembershipIndex::new());
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(1),
        &pairs(&[(0, 0), (1, 1), (2, 1)]),
    );
    let handle = CommunityIndexHandle::for_tenant(
        TenantId::DEFAULT,
        CommunityIndexId::new(1),
        idx as Arc<dyn MembershipIndex>,
    );

    assert_eq!(
        handle
            .membership(NodeId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(0))
    );
    assert_eq!(
        handle
            .members(CommunityId::new(1), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        vec![NodeId::new(1), NodeId::new(2)]
    );
    let r = handle
        .rank_by_seeds(&[NodeId::new(1)], Level::FINEST, 1, Lsn::MAX)
        .expect("ok");
    assert_eq!(r.len(), 1);
    assert_eq!(r[0].0, CommunityId::new(1));
}

#[test]
fn round_trip_multi_level() {
    let idx = BTreeMembershipIndex::new();
    // Level 0: 4 leaf communities, 8 nodes.
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(1),
        &pairs(&[
            (0, 0),
            (1, 0),
            (2, 1),
            (3, 1),
            (4, 2),
            (5, 2),
            (6, 3),
            (7, 3),
        ]),
    );
    // Level 1: leaves agglomerate pairwise into 2 communities.
    idx.install_level(
        TenantId::DEFAULT,
        Level::new(1),
        Lsn::new(2),
        &pairs(&[
            (0, 10),
            (1, 10),
            (2, 10),
            (3, 10),
            (4, 11),
            (5, 11),
            (6, 11),
            (7, 11),
        ]),
    );

    // Both levels are independently queryable.
    assert_eq!(
        idx.lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(0))
    );
    assert_eq!(
        idx.lookup(TenantId::DEFAULT, NodeId::new(0), Level::new(1), Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(10))
    );

    // Members at the upper level pulls across the lower-level
    // sub-communities.
    let upper = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(10),
            Level::new(1),
            Lsn::MAX,
        )
        .expect("ok");
    assert_eq!(
        upper,
        vec![
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
        ]
    );
}
