//! Path-A boundary tests for ADR-041 §D-3b: MVCC visibility
//! windowing on the community substrate.
//!
//! Mirrors `crates/arcgraph-bm25/tests/bm25_mvcc_visibility.rs`
//! and `crates/arcgraph-vector/tests/vector_mvcc_visibility.rs`
//! — the community substrate uses **per-install LSN history**
//! rather than per-row LSNs (per ADR-041 §D-3b: install_level is
//! the unit of update). The visibility filter applied at lookup
//! time:
//!
//!     visible install at read_lsn := latest install with
//!         install_lsn ≤ read_lsn for the (tenant, level)
//!
//! PINS:
//! - `pre_install_read_returns_empty` — a `read_lsn` strictly
//!   less than the earliest install for a (tenant, level)
//!   returns empty / None. Mirrors BM25 §D-3
//!   `reader_at_older_lsn_excludes_post_lsn_doc`.
//! - `at_install_read_sees_install` — `read_lsn = install_lsn`
//!   (inclusive lower bound) sees the install. Mirrors BM25
//!   `reader_at_exact_commit_lsn_includes_doc`.
//! - `subsequent_install_does_not_perturb_prior_snapshot` —
//!   pre-install snapshots are immutable; a new install at a
//!   later LSN does not change what an older `read_lsn` sees.
//! - `disjoint_snapshots_per_tenant_level` — distinct
//!   `(tenant, level)` pairs maintain independent histories;
//!   one install does not leak into the other.
//! - `read_at_max_sees_latest_install` — `read_lsn = Lsn::MAX`
//!   sees the most recent install (the "read-latest" default
//!   for callers without snapshot context).
//!
//! Failure of any pin is a *contract* break, not a test bug —
//! the cross-substrate snapshot-isolation guarantee depends on
//! these invariants holding across vector + community + BM25
//! (ADR-041).

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

/// PIN: ADR-041 §D-3b — `read_lsn` strictly less than the
/// earliest install returns empty / None. The read predates
/// every refresh for the (tenant, level).
#[test]
fn pre_install_read_returns_empty() {
    let idx = BTreeMembershipIndex::new();
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 0), (1, 0), (2, 1)]),
    );

    // Reads at LSN < 10 see nothing — pre-install territory.
    for stale in [Lsn::new(0), Lsn::new(1), Lsn::new(9)] {
        let r = idx
            .lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, stale)
            .expect("ok");
        assert_eq!(
            r,
            None,
            "PIN: ADR-041 §D-3b — read_lsn={} predates install at 10; lookup MUST be None (got {r:?})",
            stale.raw(),
        );

        let m = idx
            .members(TenantId::DEFAULT, CommunityId::ZERO, Level::FINEST, stale)
            .expect("ok");
        assert!(
            m.is_empty(),
            "PIN: ADR-041 §D-3b — read_lsn={} predates install; members MUST be empty (got {m:?})",
            stale.raw(),
        );
    }
}

/// PIN: ADR-041 §D-3b — `read_lsn = install_lsn` is INCLUSIVE
/// on the lower bound (mirror of BM25 §D-3
/// `reader_at_exact_commit_lsn_includes_doc`).
#[test]
fn at_install_read_sees_install() {
    let idx = BTreeMembershipIndex::new();
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 5), (1, 5), (2, 7)]),
    );

    // At read_lsn = exact install_lsn the snapshot is visible.
    let r = idx
        .lookup(
            TenantId::DEFAULT,
            NodeId::new(0),
            Level::FINEST,
            Lsn::new(10),
        )
        .expect("ok");
    assert_eq!(
        r,
        Some(CommunityId::new(5)),
        "PIN: ADR-041 §D-3b — read_lsn=10 = install_lsn=10 (inclusive) MUST see install (got {r:?})",
    );

    let m5 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(5),
            Level::FINEST,
            Lsn::new(10),
        )
        .expect("ok");
    assert_eq!(m5, vec![NodeId::new(0), NodeId::new(1)]);

    // Above the install_lsn — same data visible.
    let r_after = idx
        .lookup(
            TenantId::DEFAULT,
            NodeId::new(0),
            Level::FINEST,
            Lsn::new(11),
        )
        .expect("ok");
    assert_eq!(r_after, Some(CommunityId::new(5)));
}

/// PIN: ADR-041 §D-3b — pre-install snapshots are immutable.
/// A subsequent install at a later LSN does not perturb what a
/// `read_lsn` between the two installs sees.
#[test]
fn subsequent_install_does_not_perturb_prior_snapshot() {
    let idx = BTreeMembershipIndex::new();

    // Install A at LSN=10.
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 100), (1, 100), (2, 100)]),
    );

    // Install B at LSN=20 with completely different community ids.
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(20),
        &pairs(&[(0, 200), (1, 200), (2, 200)]),
    );

    // A read at LSN=15 sees install A (community 100).
    for n in 0..3u64 {
        let r = idx
            .lookup(
                TenantId::DEFAULT,
                NodeId::new(n),
                Level::FINEST,
                Lsn::new(15),
            )
            .expect("ok");
        assert_eq!(
            r,
            Some(CommunityId::new(100)),
            "PIN: ADR-041 §D-3b — at read_lsn=15 (between installs) node {n} \
             MUST see install A (community 100); got {r:?}",
        );
    }

    // A read at LSN=25 sees install B (community 200).
    for n in 0..3u64 {
        let r = idx
            .lookup(
                TenantId::DEFAULT,
                NodeId::new(n),
                Level::FINEST,
                Lsn::new(25),
            )
            .expect("ok");
        assert_eq!(
            r,
            Some(CommunityId::new(200)),
            "PIN: ADR-041 §D-3b — at read_lsn=25 (after install B) node {n} \
             MUST see install B (community 200); got {r:?}",
        );
    }

    // members(community 100) at LSN=15 returns A's snapshot.
    let m_a = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(100),
            Level::FINEST,
            Lsn::new(15),
        )
        .expect("ok");
    assert_eq!(
        m_a,
        vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
        "PIN: at read_lsn=15 community 100 has A's members"
    );

    // members(community 100) at LSN=25 is empty (install B
    // wholesale replaced; community 100 has no members at the
    // visible snapshot).
    let m_a_at_25 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(100),
            Level::FINEST,
            Lsn::new(25),
        )
        .expect("ok");
    assert!(
        m_a_at_25.is_empty(),
        "PIN: at read_lsn=25 community 100 has no members (install B replaced); got {m_a_at_25:?}",
    );

    // members(community 200) at LSN=25 returns B's snapshot.
    let m_b_at_25 = idx
        .members(
            TenantId::DEFAULT,
            CommunityId::new(200),
            Level::FINEST,
            Lsn::new(25),
        )
        .expect("ok");
    assert_eq!(
        m_b_at_25,
        vec![NodeId::new(0), NodeId::new(1), NodeId::new(2)],
    );
}

/// PIN: ADR-041 §D-3b — distinct (tenant, level) pairs maintain
/// independent histories. An install on (tenant_a, level_0)
/// does not leak into (tenant_b, level_0) or
/// (tenant_a, level_1).
#[test]
fn disjoint_snapshots_per_tenant_level() {
    let idx = BTreeMembershipIndex::new();
    let tenant_a = TenantId::new(100);
    let tenant_b = TenantId::new(200);

    // tenant_a, level 0 at LSN=10
    idx.install_level(
        tenant_a,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 1), (1, 1)]),
    );
    // tenant_a, level 1 at LSN=20
    idx.install_level(
        tenant_a,
        Level::new(1),
        Lsn::new(20),
        &pairs(&[(0, 11), (1, 11)]),
    );
    // tenant_b, level 0 at LSN=30
    idx.install_level(
        tenant_b,
        Level::FINEST,
        Lsn::new(30),
        &pairs(&[(0, 99), (1, 99)]),
    );

    // tenant_a level 0 at LSN=15 sees its install.
    let r = idx
        .lookup(tenant_a, NodeId::new(0), Level::FINEST, Lsn::new(15))
        .expect("ok");
    assert_eq!(r, Some(CommunityId::new(1)));

    // tenant_a level 1 at LSN=15 sees nothing yet (install at 20).
    let r = idx
        .lookup(tenant_a, NodeId::new(0), Level::new(1), Lsn::new(15))
        .expect("ok");
    assert_eq!(
        r, None,
        "PIN: tenant_a level 1 install was at LSN=20; LSN=15 predates it",
    );

    // tenant_b level 0 at LSN=15 sees nothing (install at 30).
    let r = idx
        .lookup(tenant_b, NodeId::new(0), Level::FINEST, Lsn::new(15))
        .expect("ok");
    assert_eq!(
        r, None,
        "PIN: tenant_b level 0 install was at LSN=30; LSN=15 predates it",
    );

    // At LSN=Lsn::MAX, all three (tenant, level) pairs see their
    // respective latest install — no cross-leak.
    assert_eq!(
        idx.lookup(tenant_a, NodeId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(1)),
    );
    assert_eq!(
        idx.lookup(tenant_a, NodeId::new(0), Level::new(1), Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(11)),
    );
    assert_eq!(
        idx.lookup(tenant_b, NodeId::new(0), Level::FINEST, Lsn::MAX)
            .expect("ok"),
        Some(CommunityId::new(99)),
    );
}

/// PIN: `read_lsn = Lsn::MAX` sees the most recent install.
/// This is the default "read-latest" semantic for callers
/// without snapshot context.
#[test]
fn read_at_max_sees_latest_install() {
    let idx = BTreeMembershipIndex::new();
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 1)]),
    );
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(20),
        &pairs(&[(0, 2)]),
    );
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(30),
        &pairs(&[(0, 3)]),
    );

    let r = idx
        .lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("ok");
    assert_eq!(
        r,
        Some(CommunityId::new(3)),
        "PIN: Lsn::MAX must see the latest install (lsn=30 → community 3)",
    );
}

/// PIN: `rank_by_seeds` honors the visibility filter — the
/// scoring is computed against the visible snapshot, not the
/// latest. Disjoint installs ⇒ disjoint score outputs.
#[test]
fn rank_by_seeds_honors_read_lsn() {
    let idx = BTreeMembershipIndex::new();

    // Install at LSN=10: 4 nodes split between communities 1+2.
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 1), (1, 1), (2, 2), (3, 2)]),
    );
    // Install at LSN=20: same 4 nodes ALL in community 5.
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(20),
        &pairs(&[(0, 5), (1, 5), (2, 5), (3, 5)]),
    );

    let seeds = [
        NodeId::new(0),
        NodeId::new(1),
        NodeId::new(2),
        NodeId::new(3),
    ];

    // At LSN=15: sees the LSN=10 install. 2 communities (1, 2),
    // each with 2 members and 2 seeds. Score = 2/2 = 1.0.
    let r_15 = idx
        .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 10, Lsn::new(15))
        .expect("ok");
    assert_eq!(r_15.len(), 2, "PIN: at LSN=15 expect 2 communities");
    let comm_set_15: std::collections::HashSet<u64> = r_15.iter().map(|(c, _)| c.raw()).collect();
    assert_eq!(comm_set_15, [1, 2].iter().copied().collect());
    for (_c, score) in &r_15 {
        assert!((score - 1.0).abs() < 1e-6);
    }

    // At LSN=25: sees the LSN=20 install. 1 community (5), 4
    // members, 4 seeds. Score = 4/4 = 1.0.
    let r_25 = idx
        .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 10, Lsn::new(25))
        .expect("ok");
    assert_eq!(r_25.len(), 1, "PIN: at LSN=25 expect 1 community");
    assert_eq!(r_25[0].0, CommunityId::new(5));
    assert!((r_25[0].1 - 1.0).abs() < 1e-6);

    // At LSN=5 (pre-install): empty.
    let r_5 = idx
        .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 10, Lsn::new(5))
        .expect("ok");
    assert!(
        r_5.is_empty(),
        "PIN: at LSN=5 (pre-install) rank_by_seeds returns empty",
    );
}

/// PIN: the `CommunityIndexHandle` public API correctly
/// propagates `read_lsn` through to the membership index. The
/// integration boundary the M4 query layer consumes.
#[test]
fn handle_propagates_read_lsn_to_membership() {
    let idx = Arc::new(BTreeMembershipIndex::new());
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(10),
        &pairs(&[(0, 1)]),
    );
    idx.install_level(
        TenantId::DEFAULT,
        Level::FINEST,
        Lsn::new(20),
        &pairs(&[(0, 2)]),
    );

    let handle = CommunityIndexHandle::for_tenant(
        TenantId::DEFAULT,
        CommunityIndexId::new(7),
        idx as Arc<dyn MembershipIndex>,
    );

    // At LSN=15 → community 1 (the older install).
    let r_old = handle
        .membership(NodeId::new(0), Level::FINEST, Lsn::new(15))
        .expect("ok");
    assert_eq!(r_old, Some(CommunityId::new(1)));

    // At LSN=25 → community 2 (the newer install).
    let r_new = handle
        .membership(NodeId::new(0), Level::FINEST, Lsn::new(25))
        .expect("ok");
    assert_eq!(r_new, Some(CommunityId::new(2)));

    // At LSN=Lsn::MAX → community 2 (latest).
    let r_max = handle
        .membership(NodeId::new(0), Level::FINEST, Lsn::MAX)
        .expect("ok");
    assert_eq!(r_max, Some(CommunityId::new(2)));
}
