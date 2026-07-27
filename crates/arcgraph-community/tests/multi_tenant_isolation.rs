//! Multi-tenant isolation test for [`BTreeMembershipIndex`].
//!
//! Per ADR-040 §D-3 + §D-8 Q3: community state is keyed
//! `(tenant_id, partition_id, …)`. A query for tenant A MUST
//! NOT surface community state from tenant B; this is the
//! I-V2-equivalent invariant for community detection.
//!
//! This integration test installs **identical**
//! `(NodeId, CommunityId)` tuples for two tenants A and B and
//! verifies:
//!
//! 1. A's `lookup` only returns A's communities.
//! 2. A's `members` only returns A's nodes.
//! 3. A's `rank_by_seeds` only counts A's nodes — even when the
//!    seed list collides with B's nodes 1:1.
//! 4. `clear_tenant(A)` does not affect B.
//!
//! Per ADR-041 §D-3b each install carries an `install_lsn`; the
//! tenant-isolation contract holds at any `read_lsn` ≥ the
//! latest install for the relevant tenant. Tests use `Lsn::MAX`
//! to read the latest installs.

use arcgraph_community::{BTreeMembershipIndex, CommunityId, Level, MembershipIndex};
use arcgraph_core::{Lsn, NodeId, TenantId};

const TENANT_A: TenantId = TenantId::new(100);
const TENANT_B: TenantId = TenantId::new(200);

fn pairs(input: &[(u64, u64)]) -> Vec<(NodeId, CommunityId)> {
    input
        .iter()
        .map(|&(n, c)| (NodeId::new(n), CommunityId::new(c)))
        .collect()
}

#[test]
fn lookups_do_not_cross_tenants() {
    let idx = BTreeMembershipIndex::new();
    // Tenant A: nodes 0..3 in community 100.
    idx.install_level(
        TENANT_A,
        Level::FINEST,
        Lsn::new(1),
        &pairs(&[(0, 100), (1, 100), (2, 100)]),
    );
    // Tenant B: same node ids, different community id.
    idx.install_level(
        TENANT_B,
        Level::FINEST,
        Lsn::new(2),
        &pairs(&[(0, 200), (1, 200), (2, 200)]),
    );

    // A's view sees community 100.
    for n in 0..3 {
        assert_eq!(
            idx.lookup(TENANT_A, NodeId::new(n), Level::FINEST, Lsn::MAX)
                .expect("ok"),
            Some(CommunityId::new(100)),
            "tenant A lookup must return community 100, not B's 200"
        );
    }

    // B's view sees community 200.
    for n in 0..3 {
        assert_eq!(
            idx.lookup(TENANT_B, NodeId::new(n), Level::FINEST, Lsn::MAX)
                .expect("ok"),
            Some(CommunityId::new(200)),
            "tenant B lookup must return community 200, not A's 100"
        );
    }

    // A querying B's community id returns no members (the id 200
    // is unknown in A's tenant).
    assert!(
        idx.members(TENANT_A, CommunityId::new(200), Level::FINEST, Lsn::MAX)
            .expect("ok")
            .is_empty(),
        "tenant A must not see community 200 (B's only)"
    );
    // B querying A's community id returns no members.
    assert!(
        idx.members(TENANT_B, CommunityId::new(100), Level::FINEST, Lsn::MAX)
            .expect("ok")
            .is_empty(),
        "tenant B must not see community 100 (A's only)"
    );
}

#[test]
fn members_do_not_cross_tenants() {
    let idx = BTreeMembershipIndex::new();
    // Both tenants have a community 0 with overlapping node ids.
    idx.install_level(
        TENANT_A,
        Level::FINEST,
        Lsn::new(1),
        &pairs(&[(0, 0), (1, 0), (2, 0)]),
    );
    idx.install_level(
        TENANT_B,
        Level::FINEST,
        Lsn::new(2),
        &pairs(&[(10, 0), (11, 0), (12, 0)]),
    );

    let a_members = idx
        .members(TENANT_A, CommunityId::new(0), Level::FINEST, Lsn::MAX)
        .expect("ok");
    let b_members = idx
        .members(TENANT_B, CommunityId::new(0), Level::FINEST, Lsn::MAX)
        .expect("ok");

    assert_eq!(
        a_members,
        vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)]
    );
    assert_eq!(
        b_members,
        vec![NodeId::new(10), NodeId::new(11), NodeId::new(12)]
    );
}

#[test]
fn rank_by_seeds_only_counts_target_tenant() {
    let idx = BTreeMembershipIndex::new();
    // Tenant A: nodes 0,1 in community 5.
    idx.install_level(
        TENANT_A,
        Level::FINEST,
        Lsn::new(1),
        &pairs(&[(0, 5), (1, 5)]),
    );
    // Tenant B: nodes 0,1 in community 5 (same ids!).
    idx.install_level(
        TENANT_B,
        Level::FINEST,
        Lsn::new(2),
        &pairs(&[(0, 5), (1, 5)]),
    );

    // Seeds that exist in both tenants — the rank must use only A.
    let seeds = [NodeId::new(0), NodeId::new(1)];
    let a_rank = idx
        .rank_by_seeds(TENANT_A, &seeds, Level::FINEST, 5, Lsn::MAX)
        .expect("ok");
    assert_eq!(a_rank.len(), 1);
    // 2 hits / size 2 = 1.0 (both A's seeds map to A's community 5).
    assert!((a_rank[0].1 - 1.0).abs() < 1e-6);
}

#[test]
fn clear_tenant_does_not_affect_other_tenants() {
    let idx = BTreeMembershipIndex::new();
    idx.install_level(
        TENANT_A,
        Level::FINEST,
        Lsn::new(1),
        &pairs(&[(0, 0), (1, 0)]),
    );
    idx.install_level(
        TENANT_B,
        Level::FINEST,
        Lsn::new(2),
        &pairs(&[(0, 0), (1, 0)]),
    );

    idx.clear_tenant(TENANT_A);

    // A is gone …
    assert_eq!(
        idx.lookup(TENANT_A, NodeId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        None
    );
    assert!(
        idx.members(TENANT_A, CommunityId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok")
            .is_empty()
    );

    // … but B remains.
    assert_eq!(
        idx.lookup(TENANT_B, NodeId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(0))
    );
    let b_members = idx
        .members(TENANT_B, CommunityId::new(0), Level::FINEST, Lsn::MAX)
        .expect("ok");
    assert_eq!(b_members, vec![NodeId::new(0), NodeId::new(1)]);
}

#[test]
fn unknown_level_per_tenant_independent() {
    // Tenant A has 2 levels (0 and 1); tenant B has only 1 (0).
    // Querying level 1 against A succeeds; against B errors.
    let idx = BTreeMembershipIndex::new();
    idx.install_level(TENANT_A, Level::FINEST, Lsn::new(1), &pairs(&[(0, 0)]));
    idx.install_level(TENANT_A, Level::new(1), Lsn::new(2), &pairs(&[(0, 0)]));
    idx.install_level(TENANT_B, Level::FINEST, Lsn::new(3), &pairs(&[(0, 0)]));

    assert!(
        idx.lookup(TENANT_A, NodeId::new(0), Level::new(1), Lsn::MAX)
            .is_ok(),
        "A has level 1"
    );
    let err = idx
        .lookup(TENANT_B, NodeId::new(0), Level::new(1), Lsn::MAX)
        .expect_err("B should fail at level 1");
    assert!(
        format!("{err}").contains("level"),
        "expected UnknownLevel, got {err}"
    );
}
