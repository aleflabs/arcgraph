//! Executable single-marker SeqLock rejection rationale (PR #220 S6 / issue #227).
//!
//! PR #220 (M4-04e, issue #210) shipped a **two-marker SeqLock** —
//! `commits_started` (pre-write `Release`) + `commits_observed` (post-write
//! `Release`) — for [`CatalogStats::snapshot`]'s cross-key consistency
//! protocol. The module rustdoc at
//! `crates/arcgraph-storage/src/catalog/stats.rs` (§"Cross-key snapshot
//! mechanism") documents in prose that a **single-marker** alternative was
//! considered and rejected after a proptest empirically demonstrated the
//! `sum(label_cards) > total_nodes` torn-aggregate failure mode.
//!
//! This file converts that rejection rationale from prose to **executable
//! evidence**. It defines a standalone test-only `SingleMarkerSeqLockCatalogStats`
//! variant that mirrors `CatalogStats` but with ONE post-write `Release`
//! marker (no pre-write companion). The single test it carries —
//! [`single_marker_seqlock_demonstrates_torn_aggregate`] — runs concurrent
//! writers + a snapshot reader and asserts that **at least one snapshot
//! violates the cross-key invariant** `sum(label_cards) ≤ total_nodes` (or
//! the analogous rel-type invariant). Failure of the cross-key invariant
//! is the **expected outcome**: the test exists to prove the design's
//! incorrectness inline, future-proofing against any PR that proposes
//! "simplifying" the production design back to single-marker.
//!
//! # Why `#[ignore]`-gated by default
//!
//! The expected outcome is a torn-aggregate violation — that is, the test
//! exercises a counter-example, not a real-data invariant. Running it as
//! part of the default `cargo test` run would either (a) succeed (if the
//! race surfaces), making the test conceptually inverted from every other
//! test in the suite, or (b) fail (if timing suppresses the race),
//! breaking CI on a counter-example test. Neither is desirable. The test
//! is therefore `#[ignore]`-gated and intended for **explicit, on-demand
//! invocation** when:
//!
//! - A reviewer wants to empirically witness the rejection rationale.
//! - A future PR proposes simplifying back to single-marker, and the test
//!   author wants a fresh execution as proof-of-incorrectness.
//!
//! # How to run explicitly
//!
//! ```bash
//! cargo test -p arcgraph-storage --release \
//!   single_marker_seqlock_demonstrates_torn_aggregate \
//!   -- --ignored --nocapture
//! ```
//!
//! `--release` is recommended: the race window between `increment_label`
//! and `increment_total_nodes` is short, and `--release` keeps writer
//! throughput high enough for the reader to catch a mid-commit snapshot
//! within reasonable iteration counts.
//!
//! # Cross-references
//!
//! - PR #220 round-1 review packet, Concern S6 (`single-marker rejection
//!   rationale not executable`).
//! - Issue #227 — the follow-up that scoped this counter-test.
//! - Production `CatalogStats` module rustdoc § "Cross-key snapshot
//!   mechanism" — the prose rejection rationale this test makes
//!   executable. See in particular the bullet rejecting single-marker
//!   SeqLock and pointing to this file.
//! - `tests/catalog_stats_snapshot_proptest.rs` — the regression PIN that
//!   asserts the production two-marker design satisfies the same cross-
//!   key invariant under concurrent writers (the inverse of this test's
//!   assertion).

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;

use arcgraph_core::{LabelId, TypeId};
use dashmap::DashMap;

/// Test-only single-marker SeqLock variant of `CatalogStats`.
///
/// **DO NOT USE IN PRODUCTION.** This struct exists solely to demonstrate
/// the torn-aggregate failure mode of the rejected design (see module
/// rustdoc above). The production type is `arcgraph_storage::CatalogStats`,
/// which uses the two-marker SeqLock.
///
/// The simplification from two-marker to single-marker is:
/// - Drop `commits_started`.
/// - Drop `begin_commit_observation`.
/// - Snapshot retries iff the single `commits_observed` marker changes
///   between the Acquire load before the Relaxed reads and the Acquire
///   load after.
///
/// The bug: `commits_observed` is bumped only AFTER all per-counter
/// `Relaxed` writes complete. A writer that has applied SOME (but not all)
/// per-counter increments — e.g., `increment_label` done, but
/// `increment_total_nodes` not yet — leaves the marker at its
/// pre-commit value. A snapshot taken in that window sees both
/// `marker.Acquire` loads return the same value, so it returns without
/// retrying — even though it captured an internally inconsistent
/// (`sum(label_cards) > total_nodes`) view.
#[derive(Default)]
struct SingleMarkerSeqLockCatalogStats {
    label_counts: DashMap<LabelId, AtomicU64>,
    rel_type_counts: DashMap<TypeId, AtomicU64>,
    total_nodes: AtomicU64,
    total_rels: AtomicU64,
    /// Single post-write `Release` marker. No pre-write companion. This
    /// is the load-bearing simplification the production design rejects.
    commits_observed: AtomicU64,
}

impl SingleMarkerSeqLockCatalogStats {
    fn increment_label(&self, label: LabelId) {
        self.label_counts
            .entry(label)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    fn increment_rel_type(&self, rel_type: TypeId) {
        self.rel_type_counts
            .entry(rel_type)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    fn increment_total_nodes(&self) {
        self.total_nodes.fetch_add(1, Ordering::Relaxed);
    }

    fn increment_total_rels(&self) {
        self.total_rels.fetch_add(1, Ordering::Relaxed);
    }

    /// Bump the single marker AFTER the per-counter Relaxed writes.
    /// Mirrors `CatalogStats::observe_commit` but is NOT paired with a
    /// pre-write `begin_commit_observation` — that is the rejected
    /// simplification this struct embodies.
    fn observe_commit(&self) {
        self.commits_observed.fetch_add(1, Ordering::Release);
    }

    /// Single-marker snapshot:
    /// 1. `g1 = commits_observed.Acquire`
    /// 2. Read totals (Relaxed) and DashMap entries (Relaxed)
    /// 3. `g2 = commits_observed.Acquire`
    /// 4. If `g1 == g2`, return; else retry
    ///
    /// The retry only catches commits that **completed** during the read
    /// window. It does NOT catch commits that **started** mid-read
    /// because `commits_observed` does not advance until the commit's
    /// final `observe_commit` call. The mid-commit Relaxed writes are
    /// visible through the per-counter loads, but the marker is unchanged
    /// — so step 4 reports `g1 == g2` and the snapshot returns torn.
    fn snapshot(&self) -> SingleMarkerSnapshot {
        loop {
            let g1 = self.commits_observed.load(Ordering::Acquire);
            let total_nodes = self.total_nodes.load(Ordering::Relaxed);
            let total_rels = self.total_rels.load(Ordering::Relaxed);
            let label_cards: Vec<(LabelId, u64)> = self
                .label_counts
                .iter()
                .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
                .collect();
            let rel_type_cards: Vec<(TypeId, u64)> = self
                .rel_type_counts
                .iter()
                .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
                .collect();
            let g2 = self.commits_observed.load(Ordering::Acquire);
            if g1 == g2 {
                return SingleMarkerSnapshot {
                    total_nodes,
                    total_rels,
                    label_cards,
                    rel_type_cards,
                };
            }
        }
    }
}

#[derive(Debug)]
struct SingleMarkerSnapshot {
    total_nodes: u64,
    total_rels: u64,
    label_cards: Vec<(LabelId, u64)>,
    rel_type_cards: Vec<(TypeId, u64)>,
}

/// Counter-test demonstrating that single-marker SeqLock fails the
/// cross-key invariant `sum(label_cards) ≤ total_nodes` (and the
/// analogous rel-type invariant) under concurrent commits.
///
/// **Failure of the cross-key invariant is the EXPECTED outcome.** This
/// test asserts that AT LEAST ONE snapshot violates the invariant — the
/// counter-example proof of the rejection rationale. If the assertion
/// fires (zero violations), it would mean the race was timing-suppressed;
/// the suggested remediation is to rerun on a multi-core machine in
/// `--release` with the recommended invocation in the module rustdoc.
///
/// Workload shape:
/// - 4 writer threads, each looping: `increment_label` → **widen
///   gap** → `increment_total_nodes` → **widen gap** → `increment_rel_type`
///   → **widen gap** → `increment_total_rels` → `observe_commit`. The
///   per-counter order mirrors the production commit-pipeline (and the
///   proptest harness in `apply_commit`); the inter-increment gaps are
///   the same intentional-torture instrumentation the production
///   `bench_snapshot_under_contention` uses to keep race windows open
///   long enough for the reader to observe them. Without the gaps the
///   race window is single-digit nanoseconds — too narrow for reliable
///   demonstration. The gaps make the race surface, NOT cause; they do
///   not change the architectural defect being exposed.
/// - 1 reader thread looping `snapshot()` and counting torn-aggregate
///   violations.
///
/// Iteration counts (5K+ commits / writer, 5K+ snapshots) plus the gap
/// instrumentation ensure the race window between `increment_label` and
/// `increment_total_nodes` is hit on every snapshot. The test runs in
/// well under a second on an Apple M3 Pro in `--release`.
#[ignore = "demonstrates rejected single-marker design; expected to surface \
            torn-aggregate violations. Run explicitly with `cargo test \
            -p arcgraph-storage --release \
            single_marker_seqlock_demonstrates_torn_aggregate -- --ignored \
            --nocapture` to witness the rejection rationale."]
#[test]
fn single_marker_seqlock_demonstrates_torn_aggregate() {
    const WRITER_THREADS: usize = 4;
    const COMMITS_PER_WRITER: u64 = 5_000;
    const SNAPSHOTS: usize = 5_000;
    const LABEL_SPACE: u32 = 8;
    const REL_TYPE_SPACE: u32 = 8;
    /// Spin iterations to widen the race window between per-counter
    /// `Relaxed` writes. Mirrors the production
    /// `bench_snapshot_under_contention` instrumentation (intentional
    /// torture; not steady-state). Tuned so each writer's mid-commit
    /// window is wide enough for the reader's per-counter Relaxed loads
    /// to interleave reliably.
    const INTER_INCREMENT_SPIN: u32 = 256;

    fn widen_race_window() {
        for _ in 0..INTER_INCREMENT_SPIN {
            std::hint::spin_loop();
        }
    }

    let stats = Arc::new(SingleMarkerSeqLockCatalogStats::default());
    let labels: Vec<LabelId> = (0..LABEL_SPACE).map(LabelId::new).collect();
    let rel_types: Vec<TypeId> = (0..REL_TYPE_SPACE).map(TypeId::new).collect();

    let mut writer_handles = Vec::new();
    for tid in 0..WRITER_THREADS {
        let stats = Arc::clone(&stats);
        let labels = labels.clone();
        let rel_types = rel_types.clone();
        writer_handles.push(thread::spawn(move || {
            for i in 0..COMMITS_PER_WRITER {
                let li = (tid as u64 + i) as usize % labels.len();
                let ri = (tid as u64 + i) as usize % rel_types.len();
                // Order matches the production crud.rs commit hook AND
                // the proptest's apply_commit helper. The mid-commit
                // window between increment_label and increment_total_nodes
                // is the load-bearing race — single-marker fails to
                // detect it because `commits_observed` does not move
                // until observe_commit() at the end of the commit.
                stats.increment_label(labels[li]);
                widen_race_window();
                stats.increment_total_nodes();
                widen_race_window();
                stats.increment_rel_type(rel_types[ri]);
                widen_race_window();
                stats.increment_total_rels();
                stats.observe_commit();
            }
        }));
    }

    let stats_reader = Arc::clone(&stats);
    let reader_handle = thread::spawn(move || {
        let mut violations_label: u64 = 0;
        let mut violations_rel: u64 = 0;
        let mut max_label_excess: u64 = 0;
        let mut max_rel_excess: u64 = 0;
        for _ in 0..SNAPSHOTS {
            let snap = stats_reader.snapshot();
            let sum_labels: u64 = snap.label_cards.iter().map(|(_, c)| *c).sum();
            let sum_rels: u64 = snap.rel_type_cards.iter().map(|(_, c)| *c).sum();
            if sum_labels > snap.total_nodes {
                violations_label += 1;
                let excess = sum_labels - snap.total_nodes;
                if excess > max_label_excess {
                    max_label_excess = excess;
                }
            }
            if sum_rels > snap.total_rels {
                violations_rel += 1;
                let excess = sum_rels - snap.total_rels;
                if excess > max_rel_excess {
                    max_rel_excess = excess;
                }
            }
        }
        (
            violations_label,
            violations_rel,
            max_label_excess,
            max_rel_excess,
        )
    });

    for h in writer_handles {
        h.join().expect("writer panicked");
    }
    let (violations_label, violations_rel, max_label_excess, max_rel_excess) =
        reader_handle.join().expect("reader panicked");

    // Print the diagnostic FIRST so the explicit-invocation reader sees
    // the violation count before the assertion result.
    eprintln!(
        "single-marker SeqLock surfaced {} sum(label_cards) > total_nodes \
         violations and {} sum(rel_type_cards) > total_rels violations \
         over {} snapshots (label excess up to {}, rel excess up to {}). \
         This is the EXPECTED outcome documenting the single-marker \
         design's torn-aggregate failure mode — see module rustdoc and \
         crates/arcgraph-storage/src/catalog/stats.rs §\"Cross-key \
         snapshot mechanism\".",
        violations_label, violations_rel, SNAPSHOTS, max_label_excess, max_rel_excess,
    );

    let total_violations = violations_label + violations_rel;
    assert!(
        total_violations > 0,
        "EXPECTED: single-marker SeqLock should surface ≥ 1 torn-aggregate \
         violation over {WRITER_THREADS} writers × {COMMITS_PER_WRITER} \
         commits + {SNAPSHOTS} snapshots, demonstrating the rejection \
         rationale documented in stats.rs §\"Cross-key snapshot \
         mechanism\". Got 0 violations — the race may be timing-\
         suppressed on this hardware. Rerun with --release on a multi-\
         core machine; if violations still do not surface, this counter-\
         test no longer demonstrates the rejection rationale and the \
         module rustdoc should be revisited.",
    );
}
