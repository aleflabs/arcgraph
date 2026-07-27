//! M4-04e (issue #210) cross-key snapshot invariant proptest.
//!
//! Property: for any random sequence of (increment-commit, snapshot)
//! operations against a single `CatalogStats` instance, EVERY snapshot
//! satisfies the cross-key invariants
//!
//!   sum(label_cards) ≤ total_nodes  (when total_nodes is Some)
//!   sum(rel_type_cards) ≤ total_rels (when total_rels is Some)
//!
//! This is the core M4-04e correctness claim — the snapshot mechanism
//! exists to deliver this invariant for the M4-05 cost planner. The
//! Relaxed-ordering per-counter accessors do NOT satisfy it; this
//! proptest is therefore the regression pin against any future
//! refactor that "simplifies" the snapshot back to per-counter loads.
//!
//! 256 cases is the project's standard proptest case count (matches
//! `binding_proptest`, `type_check_proptest`, `multi_tenant_tier_proptest`,
//! `catalog_stats_proptest`).

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;

use arcgraph_core::{LabelId, TypeId};
use arcgraph_storage::CatalogStats;
use proptest::prelude::*;

/// Apply one "commit"-shaped batch to `stats`: bracket the per-record
/// increments with `begin_commit_observation` + `observe_commit` per
/// the M4-04e two-marker SeqLock protocol. Mirrors the production
/// commit-pipeline shape from `crud::commit`.
fn apply_commit(stats: &CatalogStats, label_incs: &[u32], rel_incs: &[u32]) {
    stats.begin_commit_observation();
    for &l in label_incs {
        stats.increment_label(LabelId::new(l));
        stats.increment_total_nodes();
    }
    for &t in rel_incs {
        stats.increment_rel_type(TypeId::new(t));
        stats.increment_total_rels();
    }
    stats.observe_commit();
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        ..ProptestConfig::default()
    })]

    /// Cross-key invariant on a quiescent `CatalogStats`: after a
    /// random batch of commits, the snapshot's
    /// `sum(label_cards)` MUST equal `total_nodes` (and similarly for
    /// rel-types). The `=` is the post-quiescent-commit invariant —
    /// at the level of the producer hook, every label increment is
    /// paired 1:1 with a `total_nodes` increment and that pairing is
    /// preserved across commits.
    ///
    /// The `≤` form (with no `=`) is the snapshot's contract under
    /// CONCURRENT commits — see the threaded test below for that.
    #[test]
    fn snapshot_quiescent_cross_key_invariant_holds(
        // 0..=8 commits, each commits 0..=4 labels and 0..=4 rel-types
        // drawn from a small 1..=4 id space (so commits collide on
        // labels and stress the per-key counter).
        commits in proptest::collection::vec(
            (
                proptest::collection::vec(1u32..=4u32, 0..=4),
                proptest::collection::vec(1u32..=4u32, 0..=4),
            ),
            0..=8,
        ),
    ) {
        let stats = CatalogStats::new();
        let mut total_label_incs: u64 = 0;
        let mut total_rel_incs: u64 = 0;
        let mut per_label: HashMap<u32, u64> = HashMap::new();
        let mut per_type: HashMap<u32, u64> = HashMap::new();

        for (label_incs, rel_incs) in commits.iter() {
            apply_commit(&stats, label_incs, rel_incs);
            total_label_incs += label_incs.len() as u64;
            total_rel_incs += rel_incs.len() as u64;
            for &l in label_incs {
                *per_label.entry(l).or_default() += 1;
            }
            for &t in rel_incs {
                *per_type.entry(t).or_default() += 1;
            }

            // Snapshot AFTER every commit; the invariant must hold at
            // every step.
            let snap = stats.snapshot();
            if commits.iter().any(|_| true) && total_label_incs > 0 {
                // At least one commit applied: totals are Some.
                prop_assert_eq!(snap.total_nodes(), Some(total_label_incs));
                prop_assert_eq!(snap.total_rels(), Some(total_rel_incs));
            }

            // Quiescent-state cross-key invariant: sum(label_cards) ==
            // total_nodes (the producer hook pairs label and total
            // increments 1:1; under sole-thread quiescence the
            // snapshot reflects that exactly).
            let sum_labels: u64 = snap.label_cards().iter().map(|(_, c)| *c).sum();
            let sum_rels: u64 = snap.rel_type_cards().iter().map(|(_, c)| *c).sum();
            if let Some(total_nodes) = snap.total_nodes() {
                prop_assert!(
                    sum_labels <= total_nodes,
                    "≤-invariant: sum(label_cards)={} > total_nodes={}",
                    sum_labels,
                    total_nodes,
                );
                prop_assert_eq!(
                    sum_labels, total_nodes,
                    "quiescent =-invariant: sum(label_cards) must equal total_nodes",
                );
            }
            if let Some(total_rels) = snap.total_rels() {
                prop_assert!(
                    sum_rels <= total_rels,
                    "≤-invariant: sum(rel_type_cards)={} > total_rels={}",
                    sum_rels,
                    total_rels,
                );
                prop_assert_eq!(
                    sum_rels, total_rels,
                    "quiescent =-invariant: sum(rel_type_cards) must equal total_rels",
                );
            }

            // Per-key invariant: snapshot's per-label / per-rel-type
            // counts match the test oracle.
            for (&l, &expected) in per_label.iter() {
                prop_assert_eq!(
                    snap.label_card(LabelId::new(l)),
                    Some(expected),
                    "per-label oracle mismatch on label {}",
                    l,
                );
            }
            for (&t, &expected) in per_type.iter() {
                prop_assert_eq!(
                    snap.rel_type_card(TypeId::new(t)),
                    Some(expected),
                    "per-rel-type oracle mismatch on rel_type {}",
                    t,
                );
            }
        }
    }

    /// ≤-invariant under concurrent writers: spawn N writer threads
    /// that each apply commits with random per-record increment
    /// counts; the main thread interleaves snapshots and asserts the
    /// ≤-invariant on each. This is the cross-key invariant the
    /// snapshot mechanism EXISTS to deliver — without the SeqLock
    /// retry pattern, a snapshot reading old `total_nodes` and new
    /// per-label increments could violate it.
    ///
    /// Per-thread workload is small (≤ 32 commits / thread, ≤ 4
    /// increments / commit) to keep proptest case time bounded; 4
    /// writer threads is sufficient to exercise the concurrent path.
    #[test]
    fn snapshot_concurrent_le_invariant_holds(
        // 4 writer threads; per thread, vec of (label_id, rel_id) pairs
        // representing one increment-and-commit each.
        per_thread_commits in proptest::collection::vec(
            proptest::collection::vec((1u32..=8u32, 1u32..=8u32), 1..=32),
            4..=4,
        ),
    ) {
        const SNAPSHOTS: usize = 100;
        let stats = Arc::new(CatalogStats::new());

        let mut writer_handles = Vec::new();
        for commits in per_thread_commits.iter() {
            let stats = Arc::clone(&stats);
            let commits = commits.clone();
            writer_handles.push(thread::spawn(move || {
                for (label_id, rel_id) in commits {
                    // 1 label inc, 1 rel inc per "commit", bracketed
                    // by begin/observe per M4-04e protocol.
                    stats.begin_commit_observation();
                    stats.increment_label(LabelId::new(label_id));
                    stats.increment_total_nodes();
                    stats.increment_rel_type(TypeId::new(rel_id));
                    stats.increment_total_rels();
                    stats.observe_commit();
                }
            }));
        }

        // Reader thread takes SNAPSHOTS snapshots and asserts the
        // ≤-invariant on each.
        let stats_reader = Arc::clone(&stats);
        let reader = thread::spawn(move || -> Result<(), String> {
            for _ in 0..SNAPSHOTS {
                let snap = stats_reader.snapshot();
                if let Some(total_nodes) = snap.total_nodes() {
                    let sum_labels: u64 =
                        snap.label_cards().iter().map(|(_, c)| *c).sum();
                    if sum_labels > total_nodes {
                        return Err(format!(
                            "≤-invariant violated: sum(label_cards)={} > total_nodes={} \
                             (commits_observed={})",
                            sum_labels, total_nodes, snap.commits_observed(),
                        ));
                    }
                }
                if let Some(total_rels) = snap.total_rels() {
                    let sum_rels: u64 =
                        snap.rel_type_cards().iter().map(|(_, c)| *c).sum();
                    if sum_rels > total_rels {
                        return Err(format!(
                            "≤-invariant violated: sum(rel_type_cards)={} > total_rels={} \
                             (commits_observed={})",
                            sum_rels, total_rels, snap.commits_observed(),
                        ));
                    }
                }
            }
            Ok(())
        });

        for h in writer_handles {
            h.join().expect("writer panicked");
        }
        let res = reader.join().expect("reader panicked");
        prop_assert!(res.is_ok(), "concurrent ≤-invariant: {:?}", res);

        // Final-state =-invariant: every commit landed; sum equals
        // total. This pins the producer-hook 1:1 pairing.
        let final_snap = stats.snapshot();
        let total_commits: u64 = per_thread_commits
            .iter()
            .map(|c| c.len() as u64)
            .sum();
        prop_assert_eq!(final_snap.commits_observed(), total_commits);
        let sum_labels: u64 = final_snap
            .label_cards()
            .iter()
            .map(|(_, c)| *c)
            .sum();
        let sum_rels: u64 = final_snap
            .rel_type_cards()
            .iter()
            .map(|(_, c)| *c)
            .sum();
        prop_assert_eq!(final_snap.total_nodes(), Some(sum_labels));
        prop_assert_eq!(final_snap.total_rels(), Some(sum_rels));
        prop_assert_eq!(final_snap.total_nodes(), Some(total_commits));
        prop_assert_eq!(final_snap.total_rels(), Some(total_commits));
    }
}
