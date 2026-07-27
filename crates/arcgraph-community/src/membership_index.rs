//! B-tree-backed [`MembershipIndex`] per ADR-040 §D-4.
//!
//! Per ADR-041 §D-3b, the membership index maintains a
//! **per-install LSN history** so cross-substrate snapshot
//! isolation (vector + community + BM25) is enforced. Each call
//! to [`BTreeMembershipIndex::install_level`] appends a new
//! `(install_lsn, snapshot)` tuple to the history vec for the
//! `(tenant, level)`. Lookups binary-search the history for the
//! latest `install_lsn ≤ read_lsn` and answer from THAT
//! snapshot.
//!
//! Each snapshot keeps the original B-tree shape:
//!
//! - **Forward** (a `BTreeSet`) keyed by `(CommunityId, NodeId)`
//!   for O(community_size) [`MembershipIndex::members`] range
//!   scans (§D-4 "B-tree range scan order").
//! - **Reverse** (a `BTreeMap`) keyed by `NodeId` mapping to
//!   `CommunityId` for <200 ns P50 [`MembershipIndex::lookup`]
//!   point lookups (§D-9 perf budget).
//!
//! Per-snapshot keying drops the `(TenantId, Level)` columns
//! from the B-tree key — the outer `BTreeMap<(TenantId, Level),
//! Vec<Snapshot>>` already partitions by them, so the inner
//! key shape is just the per-snapshot pair.
//!
//! All methods are **per-tenant scoped**: every signature
//! accepts `tenant: TenantId` and the index never returns rows
//! from a different tenant. This is the I-V2-equivalent
//! invariant for community detection (ADR-011 + ADR-040 §D-3 +
//! §D-8).
//!
//! # Latency / memory budget
//!
//! - `lookup` is one binary-search over the install history (vec
//!   of `Vec<(Lsn, Snapshot)>`; daily-refresh × 1 year ≈ 365
//!   entries → ~9 comparisons ~30 ns) plus one `BTreeMap::get`
//!   on the resolved snapshot. The standard library's B-tree
//!   node fan-out is 6, so a 100 K-node tenant traverses
//!   `log_6(100_000) ≈ 7` levels (~170 ns). Total: ~200 ns P50,
//!   within the ADR-040 §D-9 budget. The
//!   `membership_lookup_latency` integration test asserts this
//!   against a 100 K populated index.
//! - `members` is one binary-search + one `BTreeSet::range`,
//!   reading `community_size` entries; the B-tree pulls these
//!   out in sorted order without an additional sort.
//!   `O(log_history + community_size)` per ADR-040 §D-4.
//! - `rank_by_seeds` is `O(log_history + |seeds| log N +
//!   |candidate_communities|)` per ADR-040 §D-9. The
//!   `|members(c)|` divisor in the ADR-040 §D-3 score formula
//!   is served from a per-community count cell on each snapshot
//!   (see `Snapshot::member_count`) so the cost per candidate
//!   community is `O(log N)`, not `O(|members(c)|)`. Per codex
//!   M3.d retro 2026-05-03 F2.
//! - Memory: per `(tenant, level)`, one history vec of
//!   `Snapshot`s. Each `Snapshot` is two B-trees + a per-
//!   community count `BTreeMap` + the install LSN. For one
//!   tenant with 100 K nodes the B-trees + count cell occupy
//!   ~3 MB total; daily-refresh × 1 year × 100 K nodes ≈
//!   1 GB. v1.1 GC pass (`clear_tenant_below_floor`) bounds
//!   this; at v1.0 the history grows monotonically —
//!   operator-tractable for v1.0 deployments. Note that
//!   [`BTreeMembershipIndex::clear_tenant`] erases ALL history
//!   for the tenant (test-side helper).
//!
//! # Score formula for `rank_by_seeds`
//!
//! ADR-040 §D-3 specifies the score function unchanged from the
//! pre-history shape:
//!
//! ```text
//! score(c, S, level, read_lsn) =
//!     | { v ∈ S : membership_at(read_lsn)(v, level) = c } |
//!     ──────────────────────────────────────────────────────
//!                  | members_at(read_lsn)(c, level) |
//! ```
//!
//! Ties on `score` are broken by ascending `CommunityId` for
//! determinism.

use std::collections::{BTreeMap, BTreeSet};

use arcgraph_core::{Lsn, NodeId, TenantId};
use parking_lot::RwLock;

use crate::error::CommunityError;
use crate::ids::{CommunityId, Level};
use crate::index::MembershipIndex;

/// One install of a `(tenant, level)` partition. Carries the
/// install LSN + the per-install B-tree state.
///
/// Populated by [`BTreeMembershipIndex::install_level`]; consumed
/// by lookups via the per-`(tenant, level)` history vec's binary
/// search.
struct InstalledSnapshot {
    /// LSN at which this install was committed (per ADR-041
    /// §D-3b). Lookups consult `commit_at_install ≤ read_lsn`
    /// to pick the visible snapshot.
    install_lsn: Lsn,
    /// `(community_id, node_id)` pairs sorted ascending. Reverse
    /// range scan within `(community_id)` yields all members in
    /// ascending `NodeId` order.
    forward: BTreeSet<(CommunityId, NodeId)>,
    /// `node_id → community_id` point-lookup hot path.
    reverse: BTreeMap<NodeId, CommunityId>,
    /// Per-community member count for this snapshot. Maintained
    /// alongside `forward` so [`MembershipIndex::rank_by_seeds`]
    /// can compute the ADR-040 §D-3 score `hits / |members(c)|`
    /// in `O(log N)` per candidate community instead of
    /// `O(|members(c)|)`. Per codex M3.d retro 2026-05-03 F2.
    ///
    /// **Invariant:** for every key `c` present here,
    /// `member_count[c]
    ///   == forward.range((c, NodeId::ZERO)..=(c, NodeId::MAX)).count()`.
    /// Conversely, `c` is absent from this map iff `forward` has
    /// no rows for that community in this snapshot. The count
    /// cell is built from scratch on every install (the F-1 per-
    /// install snapshot construction) so the lockstep invariant
    /// holds by construction; the
    /// `membership_index_count_cell_consistency_*` tests verify
    /// it.
    member_count: BTreeMap<CommunityId, u32>,
}

/// Inner mutable state guarded by the [`BTreeMembershipIndex`]'s
/// single `RwLock`. All lookups take a shared read lock; only
/// [`BTreeMembershipIndex::install_level`] (and the tests-only
/// [`BTreeMembershipIndex::clear_tenant`]) take the write lock.
#[derive(Default)]
struct Inner {
    /// Per `(tenant, level)`, a chronological vec of installs
    /// sorted ascending by `install_lsn`. Lookups binary-search
    /// for the latest `install_lsn ≤ read_lsn`. Each
    /// [`InstalledSnapshot`] carries its own
    /// `member_count` cell so `rank_by_seeds` is `O(log N)` per
    /// candidate community per the M3.d retro F2 contract;
    /// per-install duplication is bounded by the same per-snapshot
    /// memory budget as the rest of the snapshot.
    history: BTreeMap<(TenantId, Level), Vec<InstalledSnapshot>>,
    /// Per-tenant tracked maximum level present. Bumped by
    /// [`BTreeMembershipIndex::install_level`]; consulted by all
    /// three retrieval methods to surface
    /// [`CommunityError::UnknownLevel`] when the caller queries
    /// past the hierarchy depth.
    max_level_per_tenant: BTreeMap<TenantId, Level>,
}

/// B-tree-backed membership index. The single concrete
/// [`MembershipIndex`] implementor at v1.0 per ADR-040 §D-4 +
/// ADR-041 §D-3b.
///
/// Constructed empty via [`Self::new`] (or [`Default::default`]);
/// populated via [`Self::install_level`].
#[derive(Default)]
pub struct BTreeMembershipIndex {
    inner: RwLock<Inner>,
}

impl BTreeMembershipIndex {
    /// Construct an empty index.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner::default()),
        }
    }

    /// Bulk-install a `(tenant, level)` partition at the given
    /// `install_lsn`.
    ///
    /// Per ADR-041 §D-3b, each install is **appended** to the
    /// `(tenant, level)` history rather than replacing in place.
    /// The history vec stays sorted by `install_lsn` ascending;
    /// a future `read_lsn ≥ install_lsn` resolves to this
    /// install.
    ///
    /// `assignment` is interpreted as `(node_id, community_id)`
    /// pairs; duplicate `node_id` entries within an assignment
    /// are last-write-wins per `BTreeMap`'s `insert` semantics
    /// (preserves the original snapshot-shape invariant).
    ///
    /// # Panics
    ///
    /// Panics if `install_lsn` is non-monotonic — i.e., strictly
    /// less than the latest install for the same
    /// `(tenant, level)`. The scheduler / commit pipeline is
    /// expected to allocate LSNs monotonically per ADR-031;
    /// non-monotonic install is a bug.
    pub fn install_level(
        &self,
        tenant: TenantId,
        level: Level,
        install_lsn: Lsn,
        assignment: &[(NodeId, CommunityId)],
    ) {
        let mut g = self.inner.write();

        // Build the snapshot from scratch. Per ADR-041 §D-3b the
        // history is append-only — prior installs are NOT mutated.
        // The per-install `member_count` cell is built in lockstep
        // with `forward`: we bump only on a genuinely new
        // `(community, node)` tuple (matches the M3.d retro F2
        // contract that the count tracks the unique-row population
        // of `forward`).
        let mut forward: BTreeSet<(CommunityId, NodeId)> = BTreeSet::new();
        let mut reverse: BTreeMap<NodeId, CommunityId> = BTreeMap::new();
        let mut member_count: BTreeMap<CommunityId, u32> = BTreeMap::new();
        for &(node, community) in assignment {
            let was_new = forward.insert((community, node));
            if was_new {
                *member_count.entry(community).or_insert(0) += 1;
            }
            reverse.insert(node, community);
        }

        // Append to the history vec. The history MUST stay sorted
        // ascending by install_lsn; the scheduler's monotonic LSN
        // allocator guarantees this. We assert defensively.
        let entry = g.history.entry((tenant, level)).or_default();
        if let Some(last) = entry.last() {
            // W17δ #335 closure: under hostile-repro (~3 hrs wall-time,
            // 4× parallel workspace test contention) the BACKWARD
            // install_lsn observation from PR #330 did NOT reproduce.
            // Analysis of the LSN allocation + install path:
            //
            // - `scheduler.rs::do_refresh` allocates via
            //   `AtomicU64::fetch_add(AcqRel)` (monotonic by construction).
            // - `run_one_tick` drains the pending queue sequentially
            //   (one tenant at a time); no parallel installs within a
            //   single scheduler instance.
            // - Cross-test scheduler instances have independent
            //   `next_install_lsn` counters (no shared state).
            //
            // The assertion fires with enriched diagnostic context
            // (tenant, level, history depth, thread name) so a future
            // recurrence captures enough information to identify the
            // race window in a single observation rather than requiring
            // multi-worktree reproduction.
            assert!(
                last.install_lsn.raw() < install_lsn.raw(),
                "ADR-041 §D-3b: install_lsn must be strictly monotonic per (tenant, level); \
                 last install was {} but new is {} (tenant={}, level={}, history_depth={}, \
                 thread={:?})",
                last.install_lsn.raw(),
                install_lsn.raw(),
                tenant.raw(),
                level.raw(),
                entry.len(),
                std::thread::current().name().unwrap_or("<unnamed>"),
            );
        }
        entry.push(InstalledSnapshot {
            install_lsn,
            forward,
            reverse,
            member_count,
        });

        // Maintain the per-tenant max-level. We never lower the
        // max — uninstall is unsupported at v1.0 — so a fresh
        // tenant starts at FINEST and grows monotonically.
        let max = g
            .max_level_per_tenant
            .entry(tenant)
            .or_insert(Level::FINEST);
        if level.raw() > max.raw() {
            *max = level;
        }
    }

    /// Convenience for tests: drop ALL installs for a tenant
    /// across every `(tenant, level)`. Production callers
    /// re-install via [`Self::install_level`] instead — there is
    /// no public uninstall path at v1.0.
    pub fn clear_tenant(&self, tenant: TenantId) {
        let mut g = self.inner.write();
        // Drop every history vec keyed by this tenant. Each
        // dropped snapshot takes its per-snapshot member_count
        // cell with it, so the count cell stays consistent with
        // `forward` by construction.
        let keys: Vec<_> = g
            .history
            .range((tenant, Level::FINEST)..=(tenant, Level::MAX))
            .map(|(k, _)| *k)
            .collect();
        for k in keys {
            g.history.remove(&k);
        }
        g.max_level_per_tenant.remove(&tenant);
    }

    /// Locate the snapshot for `(tenant, level)` visible at
    /// `read_lsn`: the LATEST install with `install_lsn ≤
    /// read_lsn`. Returns `None` when no install has fired yet
    /// for the `(tenant, level)` OR when `read_lsn` predates
    /// every install (e.g., the read snapshot was allocated
    /// before the first refresh ran).
    fn snapshot_at(history: &[InstalledSnapshot], read_lsn: Lsn) -> Option<&InstalledSnapshot> {
        // Binary search by install_lsn — we want the rightmost
        // install with install_lsn ≤ read_lsn. The history is
        // sorted ascending by install_lsn (monotonic install
        // contract; see `install_level`).
        let read = read_lsn.raw();
        let i = history
            .binary_search_by(|s| s.install_lsn.raw().cmp(&read))
            // Ok(i) — exact match. Visible at read_lsn=install_lsn
            // (inclusive lower bound — same as ADR-039 §D-3 BM25).
            // Err(i) — the index where read_lsn would be inserted.
            // The visible install is at index `i - 1` (the
            // predecessor); when i == 0, there is no predecessor.
            .map(Some)
            .unwrap_or_else(|i| if i == 0 { None } else { Some(i - 1) })?;
        Some(&history[i])
    }
}

impl MembershipIndex for BTreeMembershipIndex {
    fn lookup(
        &self,
        tenant: TenantId,
        node_id: NodeId,
        level: Level,
        read_lsn: Lsn,
    ) -> Result<Option<CommunityId>, CommunityError> {
        let g = self.inner.read();
        // Validate the level if the tenant has any data. A
        // tenant with no data returns `Ok(None)` (orphan / not
        // populated yet) rather than UnknownLevel — surfacing an
        // error on every pre-refresh query would noise up the
        // common "first refresh pending" path.
        if let Some(max) = g.max_level_per_tenant.get(&tenant)
            && level.raw() > max.raw()
        {
            return Err(CommunityError::UnknownLevel {
                tenant,
                level,
                max_level: *max,
            });
        }
        let Some(history) = g.history.get(&(tenant, level)) else {
            return Ok(None);
        };
        let Some(snap) = Self::snapshot_at(history, read_lsn) else {
            // read_lsn predates every install for this
            // (tenant, level) — equivalent to the "first refresh
            // pending" path. Return None per the orphan-node
            // contract.
            return Ok(None);
        };
        Ok(snap.reverse.get(&node_id).copied())
    }

    fn members(
        &self,
        tenant: TenantId,
        community_id: CommunityId,
        level: Level,
        read_lsn: Lsn,
    ) -> Result<Vec<NodeId>, CommunityError> {
        let g = self.inner.read();
        if let Some(max) = g.max_level_per_tenant.get(&tenant)
            && level.raw() > max.raw()
        {
            return Err(CommunityError::UnknownLevel {
                tenant,
                level,
                max_level: *max,
            });
        }
        let Some(history) = g.history.get(&(tenant, level)) else {
            return Ok(Vec::new());
        };
        let Some(snap) = Self::snapshot_at(history, read_lsn) else {
            return Ok(Vec::new());
        };
        let lo = (community_id, NodeId::ZERO);
        let hi = (community_id, NodeId::MAX);
        Ok(snap.forward.range(lo..=hi).map(|(_, n)| *n).collect())
    }

    fn rank_by_seeds(
        &self,
        tenant: TenantId,
        seeds: &[NodeId],
        level: Level,
        k: usize,
        read_lsn: Lsn,
    ) -> Result<Vec<(CommunityId, f32)>, CommunityError> {
        if seeds.is_empty() {
            return Err(CommunityError::EmptySeeds);
        }
        let g = self.inner.read();
        if let Some(max) = g.max_level_per_tenant.get(&tenant)
            && level.raw() > max.raw()
        {
            return Err(CommunityError::UnknownLevel {
                tenant,
                level,
                max_level: *max,
            });
        }
        let Some(history) = g.history.get(&(tenant, level)) else {
            return Ok(Vec::new());
        };
        let Some(snap) = Self::snapshot_at(history, read_lsn) else {
            return Ok(Vec::new());
        };

        // Count seed memberships per community via the reverse
        // index. Seeds that are not present at this level are
        // silently ignored (matches the orphan-node semantics of
        // `lookup`).
        let mut hits: BTreeMap<CommunityId, u32> = BTreeMap::new();
        for &seed in seeds {
            if let Some(c) = snap.reverse.get(&seed) {
                *hits.entry(*c).or_insert(0) += 1;
            }
        }
        if hits.is_empty() {
            return Ok(Vec::new());
        }

        // ADR-040 §D-3 score: hits / |members(c, level)|.
        // We read |members(c, level)| from the per-snapshot count
        // cell on the visible snapshot — `O(log N)` per candidate
        // community via [`BTreeMap::get`]. This restores the
        // ADR-040 §D-9 analytical bound of
        // `O(k_seeds · log|V| + k_out)`: the size term is no
        // longer a per-candidate forward-range count scan
        // (`O(|members(c)|)` per call) but a constant-factor
        // B-tree lookup. Per codex M3.d retro 2026-05-03 F2.
        //
        // Invariant: `snap.member_count[c]` is in lockstep with
        // the snapshot's forward-range count for the same
        // community — see the doc-comment on
        // `InstalledSnapshot::member_count` for the contract.
        // The `membership_index_count_cell_consistency_*` tests
        // verify the invariant.
        let mut scored: Vec<(CommunityId, f32)> = hits
            .into_iter()
            .map(|(c, hit_count)| {
                let size = snap.member_count.get(&c).copied().unwrap_or(0);
                let denom = (size as f32).max(1.0);
                (c, hit_count as f32 / denom)
            })
            .collect();

        // Sort by score descending; ties broken by ascending
        // `CommunityId` for determinism.
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        scored.truncate(k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pairs(input: &[(u64, u64)]) -> Vec<(NodeId, CommunityId)> {
        input
            .iter()
            .map(|&(n, c)| (NodeId::new(n), CommunityId::new(c)))
            .collect()
    }

    #[test]
    fn new_index_is_empty() {
        let idx = BTreeMembershipIndex::new();
        assert_eq!(
            idx.lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
                .expect("lookup on empty"),
            None
        );
        assert!(
            idx.members(
                TenantId::DEFAULT,
                CommunityId::new(0),
                Level::FINEST,
                Lsn::MAX
            )
            .expect("members on empty")
            .is_empty()
        );
    }

    #[test]
    fn install_level_round_trip_lookup() {
        let idx = BTreeMembershipIndex::new();
        let assignment = pairs(&[(0, 0), (1, 0), (2, 1), (3, 1)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        for &(n, c) in &assignment {
            assert_eq!(
                idx.lookup(TenantId::DEFAULT, n, Level::FINEST, Lsn::MAX)
                    .expect("ok"),
                Some(c),
                "node {} should map to community {}",
                n.raw(),
                c.raw()
            );
        }
    }

    #[test]
    fn install_level_round_trip_members_sorted() {
        let idx = BTreeMembershipIndex::new();
        // Insert out-of-order to verify the BTreeSet returns sorted.
        let assignment = pairs(&[(7, 1), (1, 1), (4, 1), (2, 1)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        let members = idx
            .members(
                TenantId::DEFAULT,
                CommunityId::new(1),
                Level::FINEST,
                Lsn::MAX,
            )
            .expect("ok");
        assert_eq!(
            members,
            vec![
                NodeId::new(1),
                NodeId::new(2),
                NodeId::new(4),
                NodeId::new(7)
            ]
        );
    }

    /// PIN: ADR-041 §D-3b — successive installs ARE versioned.
    /// A read at the older install_lsn returns the older
    /// snapshot; a read at the newer install_lsn returns the
    /// newer snapshot. The replace-in-place semantic of the
    /// pre-ADR-041 contract is gone.
    #[test]
    fn install_level_versions_history() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0), (1, 0)]),
        );
        // Re-install at lsn=20: node 0 moves to community 1, node
        // 1 disappears.
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(20),
            &pairs(&[(0, 1)]),
        );

        // At lsn=15 (between the two installs), the OLDER
        // snapshot is visible.
        assert_eq!(
            idx.lookup(
                TenantId::DEFAULT,
                NodeId::new(0),
                Level::FINEST,
                Lsn::new(15)
            )
            .expect("ok"),
            Some(CommunityId::new(0)),
            "PIN: read at lsn=15 sees the lsn=10 snapshot",
        );
        assert_eq!(
            idx.lookup(
                TenantId::DEFAULT,
                NodeId::new(1),
                Level::FINEST,
                Lsn::new(15)
            )
            .expect("ok"),
            Some(CommunityId::new(0)),
            "PIN: node 1 was in the lsn=10 snapshot",
        );

        // At lsn=25 (after the second install), the NEWER
        // snapshot is visible.
        assert_eq!(
            idx.lookup(
                TenantId::DEFAULT,
                NodeId::new(0),
                Level::FINEST,
                Lsn::new(25)
            )
            .expect("ok"),
            Some(CommunityId::new(1)),
            "PIN: read at lsn=25 sees the lsn=20 snapshot",
        );
        assert_eq!(
            idx.lookup(
                TenantId::DEFAULT,
                NodeId::new(1),
                Level::FINEST,
                Lsn::new(25)
            )
            .expect("ok"),
            None,
            "PIN: node 1 was evicted in the lsn=20 snapshot",
        );

        // members() reflects the visible snapshot too.
        let members_pre = idx
            .members(
                TenantId::DEFAULT,
                CommunityId::new(0),
                Level::FINEST,
                Lsn::new(15),
            )
            .expect("ok");
        assert_eq!(members_pre, vec![NodeId::new(0), NodeId::new(1)]);

        let members_post = idx
            .members(
                TenantId::DEFAULT,
                CommunityId::new(0),
                Level::FINEST,
                Lsn::new(25),
            )
            .expect("ok");
        assert!(
            members_post.is_empty(),
            "post-replacement community 0 is empty"
        );
    }

    /// PIN: ADR-041 §D-3b — read at an LSN earlier than every
    /// install returns empty (the read predates every refresh).
    #[test]
    fn read_before_first_install_returns_empty() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );

        assert_eq!(
            idx.lookup(
                TenantId::DEFAULT,
                NodeId::new(0),
                Level::FINEST,
                Lsn::new(5)
            )
            .expect("ok"),
            None,
            "PIN: read at lsn=5 predates first install (lsn=10)",
        );
    }

    #[test]
    fn install_level_does_not_affect_other_levels() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0), (1, 0)]),
        );
        idx.install_level(
            TenantId::DEFAULT,
            Level::new(1),
            Lsn::new(20),
            &pairs(&[(0, 7), (1, 7)]),
        );
        // Re-install level 0 at a later LSN but keep level 1.
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(30),
            &pairs(&[(0, 1)]),
        );
        assert_eq!(
            idx.lookup(TenantId::DEFAULT, NodeId::new(0), Level::new(1), Lsn::MAX)
                .expect("ok"),
            Some(CommunityId::new(7))
        );
        assert_eq!(
            idx.lookup(TenantId::DEFAULT, NodeId::new(1), Level::new(1), Lsn::MAX)
                .expect("ok"),
            Some(CommunityId::new(7)),
            "level 1 row for node 1 should survive a level-0 reinstall"
        );
    }

    #[test]
    fn lookup_unknown_level_errors() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        let err = idx
            .lookup(TenantId::DEFAULT, NodeId::new(0), Level::new(5), Lsn::MAX)
            .expect_err("should error on level beyond max");
        match err {
            CommunityError::UnknownLevel {
                level, max_level, ..
            } => {
                assert_eq!(level, Level::new(5));
                assert_eq!(max_level, Level::FINEST);
            }
            other => panic!("expected UnknownLevel, got {other:?}"),
        }
    }

    #[test]
    fn members_unknown_level_errors() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        let err = idx
            .members(
                TenantId::DEFAULT,
                CommunityId::ZERO,
                Level::new(9),
                Lsn::MAX,
            )
            .expect_err("should error");
        assert!(matches!(err, CommunityError::UnknownLevel { .. }));
    }

    #[test]
    fn rank_by_seeds_empty_seeds_errors() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        let err = idx
            .rank_by_seeds(TenantId::DEFAULT, &[], Level::FINEST, 5, Lsn::MAX)
            .expect_err("empty seeds errors");
        assert!(matches!(err, CommunityError::EmptySeeds));
    }

    #[test]
    fn rank_by_seeds_size_aware_score() {
        // Community 0 has 2 members, both seeds → score 2/2 = 1.0.
        // Community 1 has 4 members, 2 are seeds → score 2/4 = 0.5.
        // Community 0 should rank first despite tied seed counts.
        let idx = BTreeMembershipIndex::new();
        let assignment = pairs(&[(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        let seeds = [
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
        ];
        let ranking = idx
            .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 10, Lsn::MAX)
            .expect("ok");
        assert_eq!(ranking.len(), 2);
        assert_eq!(ranking[0].0, CommunityId::new(0));
        assert!(
            (ranking[0].1 - 1.0).abs() < 1e-6,
            "community 0 score = {}",
            ranking[0].1
        );
        assert_eq!(ranking[1].0, CommunityId::new(1));
        assert!(
            (ranking[1].1 - 0.5).abs() < 1e-6,
            "community 1 score = {}",
            ranking[1].1
        );
    }

    #[test]
    fn rank_by_seeds_truncates_to_k() {
        let idx = BTreeMembershipIndex::new();
        // Three communities, one seed each.
        let assignment = pairs(&[(0, 0), (1, 1), (2, 2)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        let seeds = [NodeId::new(0), NodeId::new(1), NodeId::new(2)];
        let r = idx
            .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 2, Lsn::MAX)
            .expect("ok");
        assert_eq!(r.len(), 2);
    }

    #[test]
    fn rank_by_seeds_empty_when_no_seeds_match() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        // Seed 99 is not in the index — ranking is empty.
        let r = idx
            .rank_by_seeds(
                TenantId::DEFAULT,
                &[NodeId::new(99)],
                Level::FINEST,
                10,
                Lsn::MAX,
            )
            .expect("ok");
        assert!(r.is_empty());
    }

    #[test]
    fn rank_by_seeds_tiebreak_is_ascending_community_id() {
        let idx = BTreeMembershipIndex::new();
        // Three communities each with size 2 and one seed:
        // identical scores 0.5 each. Ranking must be deterministic
        // by ascending CommunityId.
        let assignment = pairs(&[(0, 5), (1, 5), (2, 3), (3, 3), (4, 7), (5, 7)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        let seeds = [NodeId::new(0), NodeId::new(2), NodeId::new(4)];
        let r = idx
            .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 5, Lsn::MAX)
            .expect("ok");
        assert_eq!(r.len(), 3);
        let ids: Vec<u64> = r.iter().map(|(c, _)| c.raw()).collect();
        assert_eq!(ids, vec![3, 5, 7], "tiebreak is ascending community id");
    }

    #[test]
    fn clear_tenant_drops_only_target() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::new(10),
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        idx.install_level(
            TenantId::new(20),
            Level::FINEST,
            Lsn::new(20),
            &pairs(&[(0, 0)]),
        );
        idx.clear_tenant(TenantId::new(10));
        assert!(
            idx.lookup(TenantId::new(10), NodeId::new(0), Level::FINEST, Lsn::MAX)
                .expect("ok")
                .is_none()
        );
        assert_eq!(
            idx.lookup(TenantId::new(20), NodeId::new(0), Level::FINEST, Lsn::MAX)
                .expect("ok"),
            Some(CommunityId::new(0))
        );
    }

    #[test]
    fn lookup_returns_none_for_missing_node() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        assert_eq!(
            idx.lookup(TenantId::DEFAULT, NodeId::new(999), Level::FINEST, Lsn::MAX)
                .expect("ok"),
            None
        );
    }

    #[test]
    fn members_for_unknown_community_returns_empty() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0)]),
        );
        let members = idx
            .members(
                TenantId::DEFAULT,
                CommunityId::new(99),
                Level::FINEST,
                Lsn::MAX,
            )
            .expect("ok");
        assert!(members.is_empty());
    }

    // ───────────────────────────────────────────────────────────
    // Per-community count cell consistency (codex F2 fix-up; F-1
    // adapted to the per-install snapshot model).
    //
    // The per-snapshot `InstalledSnapshot::member_count` cell is
    // the load-bearing data structure that lets `rank_by_seeds`
    // honour the ADR-040 §D-9
    // `O(k_seeds · log|V| + k_out)` complexity claim — *iff* it
    // stays in lockstep with the snapshot's `forward` population.
    // The tests below pin that invariant under both
    // `install_level` (which appends a fresh snapshot) and
    // `clear_tenant` (which drops the entire history vec for the
    // tenant) on representative scenarios.
    //
    // The invariant (per InstalledSnapshot s) is:
    //   for every c in s.member_count:
    //     s.member_count[c]
    //       == s.forward.range((c, NodeId::ZERO)..=(c, NodeId::MAX)).count()
    //   AND no c is absent from s.member_count when s.forward has
    //   rows for it.
    //
    // The PR #188 tests originally addressed a flat-keyed
    // `Inner::member_count`. Under the F-1 snapshot model, a
    // re-install no longer overwrites — it appends — so the
    // helpers here look only at the LATEST snapshot per
    // `(tenant, level)`, which is what `rank_by_seeds` reads at
    // `Lsn::MAX`.
    // ───────────────────────────────────────────────────────────

    /// Re-derive the per-community member count from the forward
    /// index of the latest snapshot for each `(tenant, level)`,
    /// keyed identically to the PR #188 helper for compatibility.
    /// Used by the consistency tests below to compare against the
    /// per-snapshot `member_count` cell — if the cell drifts,
    /// this re-derivation diverges.
    fn forward_count_groundtruth(
        idx: &BTreeMembershipIndex,
    ) -> BTreeMap<(TenantId, Level, CommunityId), u32> {
        let g = idx.inner.read();
        let mut out: BTreeMap<(TenantId, Level, CommunityId), u32> = BTreeMap::new();
        for ((t, l), history) in &g.history {
            let Some(snap) = history.last() else {
                continue;
            };
            for (c, _n) in &snap.forward {
                *out.entry((*t, *l, *c)).or_insert(0) += 1;
            }
        }
        out
    }

    /// Snapshot the latest-per-`(tenant, level)` `member_count`
    /// cell across all (tenant, level) pairs into a flat map for
    /// cross-checking against `forward_count_groundtruth`.
    fn member_count_snapshot(
        idx: &BTreeMembershipIndex,
    ) -> BTreeMap<(TenantId, Level, CommunityId), u32> {
        let g = idx.inner.read();
        let mut out: BTreeMap<(TenantId, Level, CommunityId), u32> = BTreeMap::new();
        for ((t, l), history) in &g.history {
            let Some(snap) = history.last() else {
                continue;
            };
            for (c, count) in &snap.member_count {
                out.insert((*t, *l, *c), *count);
            }
        }
        out
    }

    #[test]
    fn count_cell_matches_after_install_level() {
        let idx = BTreeMembershipIndex::new();
        // Mixed assignment: 4 nodes in community 0, 2 in community 1,
        // 1 in community 7 — 3 distinct communities at one level.
        let assignment = pairs(&[(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 1), (9, 7)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        assert_eq!(member_count_snapshot(&idx), forward_count_groundtruth(&idx));
    }

    #[test]
    fn count_cell_matches_after_reinstall_replaces_prior() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0), (1, 0), (2, 1), (3, 1)]),
        );
        // Reinstall (later install_lsn): drop community 1 entirely;
        // promote node 0 into a new community 5. Under the snapshot
        // model the prior install is preserved in history; the
        // *latest* snapshot is what `rank_by_seeds` reads at
        // `Lsn::MAX`, so the count cell helpers focus on it.
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(20),
            &pairs(&[(0, 5), (1, 0)]),
        );
        let truth = forward_count_groundtruth(&idx);
        let snap = member_count_snapshot(&idx);
        assert_eq!(snap, truth);
        // Concrete invariants for human review of the fixture (latest
        // snapshot only): community 0 has just node 1; community 5
        // has just node 0; community 1 is absent from this install.
        assert_eq!(
            snap.get(&(TenantId::DEFAULT, Level::FINEST, CommunityId::new(0))),
            Some(&1)
        );
        assert_eq!(
            snap.get(&(TenantId::DEFAULT, Level::FINEST, CommunityId::new(5))),
            Some(&1)
        );
        assert!(
            !snap.contains_key(&(TenantId::DEFAULT, Level::FINEST, CommunityId::new(1))),
            "community 1 should be absent from the latest snapshot's count cell"
        );
    }

    #[test]
    fn count_cell_matches_across_levels() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0), (1, 0), (2, 1)]),
        );
        idx.install_level(
            TenantId::DEFAULT,
            Level::new(1),
            Lsn::new(20),
            &pairs(&[(0, 7), (1, 7), (2, 7)]),
        );
        // Reinstall just level 0 — level 1 must survive untouched
        // in the count cell.
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(30),
            &pairs(&[(0, 0)]),
        );
        let truth = forward_count_groundtruth(&idx);
        let snap = member_count_snapshot(&idx);
        assert_eq!(snap, truth);
        assert_eq!(
            snap.get(&(TenantId::DEFAULT, Level::new(1), CommunityId::new(7))),
            Some(&3),
            "level 1 community 7 should still have 3 members"
        );
    }

    #[test]
    fn count_cell_matches_after_clear_tenant() {
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::new(10),
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 0), (1, 0), (2, 1)]),
        );
        idx.install_level(
            TenantId::new(20),
            Level::FINEST,
            Lsn::new(20),
            &pairs(&[(0, 5), (1, 5), (2, 5), (3, 9)]),
        );
        idx.clear_tenant(TenantId::new(10));
        let truth = forward_count_groundtruth(&idx);
        let snap = member_count_snapshot(&idx);
        assert_eq!(snap, truth);
        // Tenant 10 is wiped from the count cell.
        assert!(
            snap.keys().all(|(t, _, _)| *t != TenantId::new(10)),
            "no tenant-10 keys should remain"
        );
        // Tenant 20 is intact: community 5 still has 3 members,
        // community 9 still has 1.
        assert_eq!(
            snap.get(&(TenantId::new(20), Level::FINEST, CommunityId::new(5))),
            Some(&3)
        );
        assert_eq!(
            snap.get(&(TenantId::new(20), Level::FINEST, CommunityId::new(9))),
            Some(&1)
        );
    }

    #[test]
    fn count_cell_handles_duplicate_node_in_assignment() {
        // Pre-existing edge case: install_level's contract
        // documents that duplicate node_id entries are
        // last-write-wins for the reverse map (BTreeMap::insert
        // semantics) but BTreeSet::insert leaves both forward rows.
        // The count cell must reflect the actual forward
        // population, not the reverse population.
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            // Node 0 appears twice with two different communities;
            // forward gets BOTH (community 5 and community 7) for
            // node 0. The count cell should bump both.
            &pairs(&[(0, 5), (0, 7)]),
        );
        let truth = forward_count_groundtruth(&idx);
        let snap = member_count_snapshot(&idx);
        assert_eq!(snap, truth);
        assert_eq!(
            snap.get(&(TenantId::DEFAULT, Level::FINEST, CommunityId::new(5))),
            Some(&1)
        );
        assert_eq!(
            snap.get(&(TenantId::DEFAULT, Level::FINEST, CommunityId::new(7))),
            Some(&1)
        );
        // Reverse map is last-write-wins — node 0 maps to 7.
        assert_eq!(
            idx.lookup(TenantId::DEFAULT, NodeId::new(0), Level::FINEST, Lsn::MAX)
                .expect("ok"),
            Some(CommunityId::new(7))
        );
    }

    #[test]
    fn count_cell_handles_duplicate_pair_idempotent() {
        // Same (node, community) twice: BTreeSet::insert returns
        // false on the second; the count should reflect a single
        // entry, not double-count.
        let idx = BTreeMembershipIndex::new();
        idx.install_level(
            TenantId::DEFAULT,
            Level::FINEST,
            Lsn::new(10),
            &pairs(&[(0, 5), (0, 5), (1, 5)]),
        );
        let truth = forward_count_groundtruth(&idx);
        let snap = member_count_snapshot(&idx);
        assert_eq!(snap, truth);
        assert_eq!(
            snap.get(&(TenantId::DEFAULT, Level::FINEST, CommunityId::new(5))),
            Some(&2),
            "duplicate (0, 5) should be idempotent — community 5 has 2 unique members"
        );
    }

    #[test]
    fn rank_by_seeds_unchanged_with_count_cell() {
        // Regression sanity: rank_by_seeds reads the count cell
        // for the size denominator; the score formula and tiebreak
        // are unchanged. This re-asserts the
        // `rank_by_seeds_size_aware_score` fixture exactly to pin
        // that the cell-based path produces identical output.
        let idx = BTreeMembershipIndex::new();
        let assignment = pairs(&[(0, 0), (1, 0), (2, 1), (3, 1), (4, 1), (5, 1)]);
        idx.install_level(TenantId::DEFAULT, Level::FINEST, Lsn::new(10), &assignment);
        let seeds = [
            NodeId::new(0),
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
        ];
        let ranking = idx
            .rank_by_seeds(TenantId::DEFAULT, &seeds, Level::FINEST, 10, Lsn::MAX)
            .expect("ok");
        assert_eq!(ranking.len(), 2);
        assert_eq!(ranking[0].0, CommunityId::new(0));
        assert!((ranking[0].1 - 1.0).abs() < 1e-6);
        assert_eq!(ranking[1].0, CommunityId::new(1));
        assert!((ranking[1].1 - 0.5).abs() < 1e-6);
    }

    proptest::proptest! {
        /// Property: under a randomised sequence of `install_level`
        /// and `clear_tenant` operations, the per-snapshot
        /// `member_count` cell on the latest snapshot for every
        /// `(tenant, level)` always equals the per-community count
        /// derived independently from that snapshot's `forward`
        /// index. Per codex M3.d retro 2026-05-03 F2: the count
        /// cell is the sole source of truth for `rank_by_seeds`'s
        /// size denominator and must not drift.
        ///
        /// LSN allocation: monotonically increasing per op so the
        /// scheduler-monotonic install_lsn contract holds.
        #[test]
        fn count_cell_consistency_under_random_ops(
            ops in proptest::collection::vec(
                (
                    0u64..3,                   // tenant id (kept small to force cross-tenant interaction)
                    0u8..3,                    // level (kept small to force level-shadowing)
                    proptest::collection::vec(
                        (0u64..16, 0u64..6),   // (node, community)
                        0..12,
                    ),
                    proptest::bool::ANY,        // clear-tenant flag
                ),
                1..40,
            ),
        ) {
            let idx = BTreeMembershipIndex::new();
            // Per-(tenant, level) running install_lsn; install_level
            // requires strict monotonicity per (tenant, level). We
            // also monotonically advance a global counter as a
            // cheap source of fresh LSNs.
            let mut next_lsn: BTreeMap<(TenantId, Level), u64> = BTreeMap::new();
            let mut global: u64 = 1;
            for (t_raw, l_raw, assignment, clear) in ops {
                let t = TenantId::new(t_raw);
                let l = Level::new(l_raw);
                if clear {
                    idx.clear_tenant(t);
                    // Wipe per-(tenant, level) LSN tracking so a
                    // subsequent install on the wiped tenant can
                    // start from a fresh LSN.
                    let keys: Vec<_> = next_lsn
                        .keys()
                        .copied()
                        .filter(|(tt, _)| *tt == t)
                        .collect();
                    for k in keys {
                        next_lsn.remove(&k);
                    }
                } else {
                    let asg: Vec<_> = assignment
                        .into_iter()
                        .map(|(n, c)| (NodeId::new(n), CommunityId::new(c)))
                        .collect();
                    let prior = next_lsn.get(&(t, l)).copied().unwrap_or(0);
                    global = global.max(prior) + 1;
                    next_lsn.insert((t, l), global);
                    idx.install_level(t, l, Lsn::new(global), &asg);
                }
                proptest::prop_assert_eq!(
                    member_count_snapshot(&idx),
                    forward_count_groundtruth(&idx),
                    "member_count drifted from forward-derived ground truth"
                );
            }
        }
    }

    /// PIN: snapshot_at binary-search returns the rightmost
    /// install with `install_lsn ≤ read_lsn`.
    #[test]
    fn snapshot_at_returns_rightmost_le_install() {
        let history = vec![
            InstalledSnapshot {
                install_lsn: Lsn::new(10),
                forward: BTreeSet::new(),
                reverse: BTreeMap::new(),
                member_count: BTreeMap::new(),
            },
            InstalledSnapshot {
                install_lsn: Lsn::new(20),
                forward: BTreeSet::new(),
                reverse: BTreeMap::new(),
                member_count: BTreeMap::new(),
            },
            InstalledSnapshot {
                install_lsn: Lsn::new(30),
                forward: BTreeSet::new(),
                reverse: BTreeMap::new(),
                member_count: BTreeMap::new(),
            },
        ];
        // read_lsn before any install
        assert!(BTreeMembershipIndex::snapshot_at(&history, Lsn::new(5)).is_none());
        // exact-match boundary
        assert_eq!(
            BTreeMembershipIndex::snapshot_at(&history, Lsn::new(10)).map(|s| s.install_lsn.raw()),
            Some(10)
        );
        // between installs
        assert_eq!(
            BTreeMembershipIndex::snapshot_at(&history, Lsn::new(15)).map(|s| s.install_lsn.raw()),
            Some(10)
        );
        // exact-match middle
        assert_eq!(
            BTreeMembershipIndex::snapshot_at(&history, Lsn::new(20)).map(|s| s.install_lsn.raw()),
            Some(20)
        );
        // far future
        assert_eq!(
            BTreeMembershipIndex::snapshot_at(&history, Lsn::MAX).map(|s| s.install_lsn.raw()),
            Some(30)
        );
    }
}
