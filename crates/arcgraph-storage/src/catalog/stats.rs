//! Per-tenant catalog stats — M4-41 (M4-04a) per ADR-038 §2 D-25.
//!
//! # Budget
//!
//! Under the performance-budget discipline (back-of-envelope before
//! implementation). `CatalogStats` is the producer feeding the
//! `SelectivityEstimator` hot path consumed by the M4-05 cost planner.
//!
//! - **Atomic loads p99 ≤ 5ns** under uncontended access; `Relaxed`
//!   ordering chosen for performance on the per-counter accessors.
//!   Cross-key consistency is NOT maintained on the per-counter
//!   accessors (see "Cross-key consistency" section below); consumers
//!   that need a coherent multi-key view call [`CatalogStats::snapshot`].
//! - **CAS-loop decrements** under contention can degrade to O(N
//!   retries); v1.0 commit-serialisation invariant (per ADR-031 /
//!   ADR-034) keeps N ≤ 1 in practice. Bench TODO for v1.1 pipelined-
//!   commit promotion.
//! - **`snapshot()` cost** is `O(label_count + rel_type_count)` —
//!   three Acquire loads on the SeqLock counters + iterate-and-
//!   Relaxed-load both DashMaps + two `Vec` allocations + two
//!   `sort_unstable_by_key` calls. At v1.0 tenant sizes (≤ 100 labels,
//!   ≤ 100 rel-types per tenant) the per-call cost is sub-microsecond
//!   in benches and is paid ONCE per plan in the M4-05 cost planner;
//!   well inside the 5 ms plan-build budget per ADR-036 §D-25.
//! - **Per-tenant `CatalogStats` instances are independent**; no
//!   cross-tenant coordination cost.
//!
//! [`CatalogStats`] is the in-memory backing store for the
//! `CatalogProvider` trait's M4-41 cardinality methods
//! (`label_cardinality`, `rel_type_cardinality`, `total_node_count`,
//! `total_rel_count`). Each instance owns the stats for **one tenant**;
//! multi-tenant isolation is structural (each tenant's stats live in
//! a separate `CatalogStats` instance, typically held by a
//! per-tenant catalog handle on `CrudStore`).
//!
//! # Concurrency
//!
//! - Per-label and per-rel-type counters live in [`DashMap`] entries
//!   wrapping [`AtomicU64`]; sharded locking on the DashMap key set
//!   plus lock-free RMW on the counter avoids any global mutex.
//! - Tenant-wide totals are bare `AtomicU64`s — there is no per-key
//!   sharding to worry about.
//! - Within a tenant, commits are serialized by the MVCC kernel
//!   (per ADR-031 / ADR-034 commit ordering), so there is no
//!   correctness-relevant interleaving inside the increment helpers.
//!   The atomics + DashMap are belt-and-braces against future
//!   pipelined-commit work and against tests that hammer the helper
//!   from many threads.
//!
//! # Cross-key consistency
//!
//! Per-counter accessors ([`CatalogStats::label_cardinality`],
//! [`CatalogStats::total_node_count`], etc.) use `Relaxed` ordering and
//! do NOT maintain cross-key invariants. Two reads on a per-counter
//! accessor pair (e.g., `label_cardinality` followed by
//! `total_node_count`) can straddle a concurrent commit, producing an
//! inconsistent ratio (`label_card / total > 1.0` or two labels whose
//! `card / total` ratios sum to > 1.0). Consumers of the per-counter
//! accessors MUST clamp downstream — see
//! `arcgraph_query::semantic::selectivity::clamp_unit` for the
//! canonical defense-in-depth pattern.
//!
//! For plan-time use cases that need a coherent multi-key view —
//! specifically the M4-05 cost planner's join-cardinality estimation,
//! which composes per-label selectivities and depends on
//! `sum(label_cards) ≤ total_nodes` for monotonic cost estimates —
//! call [`CatalogStats::snapshot`]. The snapshot returns a
//! [`CatalogSnapshot`] capturing all counters under a single Acquire
//! barrier (see "Cross-key snapshot mechanism" below).
//!
//! # Cross-key snapshot mechanism (M4-04e per issue #210)
//!
//! [`CatalogStats::snapshot`] uses a two-marker SeqLock-style read
//! pattern keyed on a pair of coordinating atomics:
//!
//! - `commits_started` — bumped with `Release` ordering at the START
//!   of a commit's stats updates by `Self::begin_commit_observation`.
//! - `commits_observed` — bumped with `Release` ordering at the END
//!   of a commit's stats updates by `Self::observe_commit` (the
//!   pre-existing field).
//!
//! Invariants:
//! - `commits_started ≥ commits_observed` always.
//! - `commits_started == commits_observed` ⟺ no commit in-flight on
//!   any thread.
//!
//! The commit pipeline contract is **two-phase per commit, per
//! tenant**: callers MUST invoke `begin_commit_observation()` BEFORE
//! issuing any per-counter `increment_*` / `decrement_*` calls for
//! that tenant in the commit, and MUST invoke `observe_commit()`
//! exactly once after all per-counter updates are issued. The two
//! markers bracket the per-counter writes; the reader uses them to
//! detect mid-commit interleaving.
//!
//! Reader protocol:
//!
//! 1. `o1 = commits_observed.load(Acquire)`. The Acquire pairs with
//!    the most recent `observe_commit` Release; all per-counter
//!    writes from commits with index ≤ `o1` are guaranteed visible.
//! 2. `s1 = commits_started.load(Acquire)`. If `s1 != o1`, a commit
//!    is currently in flight on some thread — retry from step 1.
//! 3. Read totals + iterate per-label / per-rel-type DashMaps with
//!    `Relaxed` ordering.
//! 4. `s2 = commits_started.load(Acquire)`. If `s2 != s1`, a new
//!    commit started during the read window — retry. (We do NOT need
//!    to re-load `commits_observed`: if `s2 == s1`, no commit
//!    started, therefore no commit could have completed either —
//!    `commits_observed` cannot move ahead of `commits_started`.)
//! 5. Return the snapshot. The Relaxed reads in step 3 are guaranteed
//!    to see ONLY values from commits with index ≤ `o1`.
//!
//! Under v1.0 commit serialization (per ADR-031 / ADR-034) at most
//! one writer is in-flight at any moment, so the retry rate is bounded
//! by the snapshot duration vs. inter-commit gap; in practice ≤ 1
//! retry. The retry loop makes monotonic progress because
//! `commits_started` strictly increases.
//!
//! This mechanism preserves the lock-free per-counter hot path —
//! commits do NOT take any lock; only the twice-per-commit markers
//! (`begin_commit_observation` + `observe_commit`) upgrade from
//! `Relaxed` to `Release`. Alternative designs considered and
//! rejected:
//!
//! - **Lock-based** (RwLock around all counters): adds a lock to the
//!   commit hot path, contradicts the lock-free design ADR-038 §D-25
//!   commits to.
//! - **Single-marker SeqLock** (`Acquire`/`Release` on a single
//!   counter, retry on torn read): rejected after empirical proptest
//!   failure. With only the post-write Release marker, the reader's
//!   Relaxed loads can see partial mid-commit state because the
//!   counter increments themselves are Relaxed and unordered with the
//!   final Release; the SeqLock retry only catches commits that
//!   COMPLETED during the read, not commits that STARTED mid-read.
//!   The `tests/catalog_stats_snapshot_proptest.rs` regression pin
//!   guards against any future refactor reverting to single-marker;
//!   `tests/catalog_stats_single_marker_counter_test.rs` is the
//!   `#[ignore]`-gated executable counter-test (PR #220 S6 / issue
//!   #227) that constructs the rejected design inline and demonstrates
//!   the torn-aggregate failure mode under concurrent commits.
//!
//! [`CatalogSnapshot`] is `Clone`-able and `Send`-able; M4-05 cost
//! planner takes one snapshot per plan and reads from it through many
//! predicates without re-paying the snapshot cost or interleaving with
//! a concurrent commit.
//!
//! ## Cold-start rebuild contract (M4-41 implementation forward-note)
//!
//! When the future M4-41 stats-persistence implementation slice (per
//! ADR-038 amendment-06 §D-25.1) lands, its cold-start rebuild path MUST
//! honor the two-marker SeqLock contract documented above:
//!
//! 1. For each tenant being rebuilt, call `begin_commit_observation()`
//!    ONCE before the per-record increment walk.
//! 2. Apply per-record `increment_label` / `increment_rel_type` /
//!    `increment_total_*` calls for every MVCC record visible at the
//!    recovered LSN.
//! 3. Call `observe_commit()` ONCE after the walk completes.
//! 4. If the walk panics mid-flight, `observe_commit()` MUST still run
//!    (mirror the `crud::commit` panic-safety pattern at lines
//!    ~2927-3015 of `crud.rs`) — otherwise post-recovery `snapshot()`
//!    callers will spin retrying because `commits_started >
//!    commits_observed`.
//!
//! See ADR-038 amendment-06 §D-25.1 for the rebuild architecture +
//! K-1a R1 acceptance criteria. The rebuild path's snapshot consistency
//! is the cross-PR coherence point flagged by codex review of PR #220
//! (S4) + PR #217 (H-2) — both findings closed by this forward-note +
//! the matching amendment-06 §D-25.1 step 2 amendment.
//!
//! # v1.0 scope (per amendment-03 M4-04 decomposition)
//!
//! - **M4-41 (this module):** label cardinality + rel-type cardinality
//!   + per-tenant totals. Exact counts only.
//! - **M4-42 (next sub-slice):** selectivity estimators per predicate
//!   class. Builds on `CatalogStats`'s cardinalities.
//! - **M4-04e (this slice; issue #210):** [`CatalogSnapshot`] +
//!   [`CatalogStats::snapshot`] for plan-time cross-key consistency.
//!   The M4-05 (M4-51) cost planner — queued as the next post-M4-04
//!   slice — consumes `snapshot()` once per plan; consumer-side wiring
//!   ships in M4-51 (this slice ships only the producer-side API).
//! - **M4-04c (deferred to v1.1):** HyperLogLog approximations and
//!   bucketed property-value sketches. The trait surface stays
//!   `Option<u64>` so the v1.1 swap is additive.
//! - **M4-71 feedback loop (later in M4):** observed runtime row
//!   counts feed back into `CatalogStats` (per ADR-038 amendment-03
//!   §"Implicit dependency edges" item 4); the public increment
//!   surface is the receiver of that closed loop.
//!
//! # Saturating decrement
//!
//! `decrement_*` helpers saturate at zero. Within a single commit the
//! collector cannot decrement a label that was not previously
//! incremented — every `Delete` must have observed a `Create` at some
//! prior LSN. The saturation is defensive: it protects against the
//! pathological case where stats infrastructure is bolted onto an
//! existing tenant whose pre-stats commits installed records but did
//! not run the increment hook (e.g., upgrade path; not a v1.0 scenario,
//! but cheap to defend now). The trait contract documents `Some(0)` as
//! "observed-then-fully-deleted" rather than "never observed"; readers
//! tell them apart by checking whether the label key is present in the
//! map.
//!
//! # Local partition discipline
//!
//! `CatalogStats` carries no `PartitionId` field. v1.0 deployments are
//! single-partition (`PartitionId::ZERO`), and the per-tenant catalog
//! handle that owns this `CatalogStats` already inherits its local
//! partition identity from the tenant.
//!
//! # Production `CatalogProvider` wiring (M4-31+ deferred)
//!
//! `CatalogStats` is the data substrate. The
//! `CatalogProvider::label_cardinality` /
//! `CatalogProvider::rel_type_cardinality` /
//! `CatalogProvider::total_node_count` /
//! `CatalogProvider::total_rel_count` trait methods (defined in
//! `arcgraph-query`) read from a `CatalogStats` instance through a
//! production `CatalogProvider` impl that delegates `label_cardinality(label)`
//! to `stats.label_cardinality(label)`, etc.
//!
//! That production impl does NOT live in `arcgraph-storage`: per
//! `docs/bounded-contexts.md`, the dependency direction is
//! `arcgraph-query → arcgraph-storage`, so a `CatalogProvider` impl
//! in storage would require an inverted dependency. The production
//! impl lands at the M4-31+ executor wiring slice (when
//! `arcgraph-query` gains the `arcgraph-storage` dep per
//! `bounded-contexts.md` §"arcgraph-query"); it consumes the
//! [`crate::crud::CrudStore::catalog_stats`] accessor exported here.
//! M4-41 ships the substrate; M4-31+ wires the trait. See ADR-038
//! §2 D-25 for the contract.

use arcgraph_core::{LabelId, NodeId, TypeId};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_OUT_DEGREE_SKETCH_CAP: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct OutDegreeKey {
    label: LabelId,
    rel_type: TypeId,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MaxOutDegreeEntry {
    pub label: LabelId,
    pub rel_type: TypeId,
    pub vertex: NodeId,
    pub degree: u64,
}

#[derive(Debug, Default)]
struct OutDegreeSketch {
    degrees: HashMap<NodeId, u64>,
}

impl OutDegreeSketch {
    fn record_increment(&mut self, vertex: NodeId) {
        if let Some(degree) = self.degrees.get_mut(&vertex) {
            *degree = degree.saturating_add(1);
            return;
        }
        if self.degrees.len() < MAX_OUT_DEGREE_SKETCH_CAP {
            self.degrees.insert(vertex, 1);
            return;
        }

        if let Some((victim, min_degree)) = self
            .degrees
            .iter()
            .min_by(|(left_vertex, left_degree), (right_vertex, right_degree)| {
                left_degree
                    .cmp(right_degree)
                    .then_with(|| left_vertex.raw().cmp(&right_vertex.raw()))
            })
            .map(|(victim, degree)| (*victim, *degree))
        {
            self.degrees.remove(&victim);
            self.degrees.insert(vertex, min_degree.saturating_add(1));
        }
    }

    fn entries(&self, key: OutDegreeKey) -> Vec<MaxOutDegreeEntry> {
        let mut entries: Vec<_> = self
            .degrees
            .iter()
            .map(|(vertex, degree)| MaxOutDegreeEntry {
                label: key.label,
                rel_type: key.rel_type,
                vertex: *vertex,
                degree: *degree,
            })
            .collect();
        entries.sort_unstable_by(|a, b| {
            b.degree
                .cmp(&a.degree)
                .then_with(|| a.label.raw().cmp(&b.label.raw()))
                .then_with(|| a.rel_type.raw().cmp(&b.rel_type.raw()))
                .then_with(|| a.vertex.raw().cmp(&b.vertex.raw()))
        });
        entries
    }
}

/// Per-tenant catalog stats — M4-41 backing store.
///
/// Owned by a per-tenant catalog handle. Instances are independent —
/// two tenants' [`CatalogStats`] never share state.
///
/// # Thread safety
///
/// `Send + Sync` by virtue of [`DashMap`] + [`AtomicU64`]. Cloning is
/// not provided: callers wrap the instance in `Arc<CatalogStats>` to
/// share across threads / builders / handlers.
#[derive(Debug, Default)]
pub struct CatalogStats {
    /// Per-label exact node cardinality. A key absent from the map
    /// means "no node with this label has been observed since the
    /// stats started"; readers translate that to `None` at the
    /// `CatalogProvider` boundary.
    label_counts: DashMap<LabelId, AtomicU64>,
    /// Per-rel-type exact relationship cardinality.
    rel_type_counts: DashMap<TypeId, AtomicU64>,
    /// ADR-025 §5 max_out_degree_sketch[label, rel_type].
    ///
    /// Perf budget: O(100) space per `(label, rel_type)` key, O(1)
    /// amortized update while a tracked vertex remains in the bounded
    /// exact-top set, and O(100 log 100) snapshot serialization per key.
    max_out_degree: DashMap<OutDegreeKey, Mutex<OutDegreeSketch>>,
    /// Tenant-wide total node count. Read as 0 before the first
    /// commit; the `CatalogProvider` boundary translates the
    /// `commits_observed = 0` state to `None` via [`Self::has_observed_any`].
    total_nodes: AtomicU64,
    /// Tenant-wide total relationship count. Mirrors `total_nodes`.
    total_rels: AtomicU64,
    /// Number of commits that have STARTED applying stats updates to
    /// this instance. Bumped with `Release` ordering by
    /// [`Self::begin_commit_observation`] BEFORE the per-counter
    /// `increment_*` / `decrement_*` calls fire.
    ///
    /// Invariant: `commits_started >= commits_observed`. Equality
    /// holds iff no commit is in-flight on any thread.
    ///
    /// M4-04e (issue #210): paired with `commits_observed` to provide
    /// the two-marker SeqLock used by [`Self::snapshot`] to detect
    /// mid-commit interleaving. See module docs §"Cross-key snapshot
    /// mechanism" for the full protocol.
    commits_started: AtomicU64,
    /// Number of commits that have FINISHED applying stats updates
    /// to this instance. Bumped with `Release` ordering by
    /// [`Self::observe_commit`] AFTER the per-counter
    /// `increment_*` / `decrement_*` calls fire.
    ///
    /// Used by the `CatalogProvider` boundary to distinguish
    /// "fresh tenant, never observed any commit" (return `None` for
    /// totals) from "totals observed and now equal zero" (return
    /// `Some(0)`).
    commits_observed: AtomicU64,
}

impl CatalogStats {
    /// Construct an empty stats instance with all counters at 0.
    /// Equivalent to [`Default`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Increment the label counter by 1. Used by the commit-pipeline
    /// hook on every newly-created node.
    pub fn increment_label(&self, label: LabelId) {
        self.label_counts
            .entry(label)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the label counter by 1, saturating at 0. Used by
    /// the commit-pipeline hook on every node deletion. See module
    /// docs for the saturation rationale.
    pub fn decrement_label(&self, label: LabelId) {
        if let Some(entry) = self.label_counts.get(&label) {
            // Saturating CAS loop: load → max(prev-1, 0) → CAS, retry
            // on contention. At most one retry under the v1.0 commit
            // serialization invariant.
            let cell = entry.value();
            let mut current = cell.load(Ordering::Relaxed);
            loop {
                if current == 0 {
                    return;
                }
                match cell.compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(actual) => current = actual,
                }
            }
        }
        // Absent key: defensive no-op. Decrementing a label that was
        // never incremented is a logic bug at the call site, but
        // not a correctness violation here.
    }

    /// Increment the per-rel-type counter by 1.
    pub fn increment_rel_type(&self, rel_type: TypeId) {
        self.rel_type_counts
            .entry(rel_type)
            .or_insert_with(|| AtomicU64::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record one outgoing relationship for the ADR-025 §5
    /// `max_out_degree_sketch[label, rel_type]` supernode detector.
    ///
    /// This is called from the same post-commit stats seam as
    /// [`Self::increment_rel_type`], after the relationship record is
    /// durably committed.
    pub fn record_out_degree(&self, label: LabelId, rel_type: TypeId, vertex: NodeId) {
        let key = OutDegreeKey { label, rel_type };
        self.max_out_degree
            .entry(key)
            .or_default()
            .lock()
            .record_increment(vertex);
    }

    /// Decrement the per-rel-type counter by 1, saturating at 0.
    pub fn decrement_rel_type(&self, rel_type: TypeId) {
        if let Some(entry) = self.rel_type_counts.get(&rel_type) {
            let cell = entry.value();
            let mut current = cell.load(Ordering::Relaxed);
            loop {
                if current == 0 {
                    return;
                }
                match cell.compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return,
                    Err(actual) => current = actual,
                }
            }
        }
    }

    /// Increment the tenant-wide total node count by 1.
    pub fn increment_total_nodes(&self) {
        self.total_nodes.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the tenant-wide total node count by 1, saturating
    /// at 0. Same saturating CAS loop as [`Self::decrement_label`].
    pub fn decrement_total_nodes(&self) {
        let mut current = self.total_nodes.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                return;
            }
            match self.total_nodes.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// Increment the tenant-wide total relationship count by 1.
    pub fn increment_total_rels(&self) {
        self.total_rels.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement the tenant-wide total relationship count by 1,
    /// saturating at 0.
    pub fn decrement_total_rels(&self) {
        let mut current = self.total_rels.load(Ordering::Relaxed);
        loop {
            if current == 0 {
                return;
            }
            match self.total_rels.compare_exchange_weak(
                current,
                current - 1,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return,
                Err(actual) => current = actual,
            }
        }
    }

    /// M4-04e (issue #210): mark the start of a commit's stats
    /// updates. Called once per commit, per tenant, BEFORE issuing
    /// any [`Self::increment_label`] / [`Self::increment_total_nodes`]
    /// / etc. calls for that tenant. Paired with [`Self::observe_commit`]
    /// at the end of the commit.
    ///
    /// The two markers bracket the commit's per-counter writes so that
    /// [`Self::snapshot`]'s reader can detect mid-commit interleaving
    /// — a snapshot taken between `begin_commit_observation` and
    /// `observe_commit` would see partial state (some labels
    /// incremented but `total_nodes` not yet updated, or vice-versa)
    /// without this marker.
    ///
    /// # Memory ordering
    ///
    /// Uses `Release` ordering. The Acquire loads in
    /// [`Self::snapshot`] use this to detect that a commit is
    /// currently in flight (`commits_started > commits_observed`)
    /// and retry the snapshot read.
    ///
    /// # Pairing contract
    ///
    /// Every `begin_commit_observation` MUST be paired with exactly
    /// one matching `observe_commit` for the same tenant. The
    /// commit-pipeline hook in `crud::commit` enforces this pairing
    /// per touched tenant. A panic between begin and end leaves
    /// `commits_started > commits_observed`; subsequent
    /// `snapshot()` calls will retry until a future commit closes
    /// the gap. (The current commit-pipeline catches such panics and
    /// logs them; see PR #170 reviewer Finding 2 in `crud.rs`.)
    pub fn begin_commit_observation(&self) {
        self.commits_started.fetch_add(1, Ordering::Release);
    }

    /// Mark that a commit has finished updating this stats instance.
    /// Called once per commit by the pipeline hook AFTER all
    /// increments / decrements for that tenant have been applied,
    /// paired with a prior [`Self::begin_commit_observation`] call.
    /// The `CatalogProvider` boundary uses this counter to distinguish
    /// "no commits ever observed" (return `None` for totals) from
    /// "commits observed and totals equal 0".
    ///
    /// # Memory ordering
    ///
    /// Uses `Release` ordering. This pairs with the `Acquire` loads in
    /// [`Self::snapshot`] so that the snapshot reader sees ALL of this
    /// commit's prior `Relaxed` increments to label / rel-type / total
    /// counters when it observes the post-`fetch_add` value of
    /// `commits_observed`. The cost is one extra synchronization
    /// fence per commit (a once-per-commit operation, NOT on the
    /// per-record increment hot path). See module docs §"Cross-key
    /// snapshot mechanism" for the full protocol.
    pub fn observe_commit(&self) {
        self.commits_observed.fetch_add(1, Ordering::Release);
    }

    /// Returns `true` once at least one commit has called
    /// [`Self::observe_commit`]. Used by the `CatalogProvider`
    /// boundary's totals translation: `None` before any commit,
    /// `Some(count)` after.
    ///
    /// # Recovery-vs-cold-start distinction
    ///
    /// This `None` / `Some(0)` distinction is load-bearing for
    /// `arcgraph_query::semantic::selectivity::SelectivityEstimator`
    /// graceful degradation:
    /// - `None` (this method returns `false`) → "never observed" →
    ///   cold-start path; the estimator falls back to
    ///   `DEFAULT_*_SELECTIVITY` constants.
    /// - `Some(0)` (this method returns `true` but the underlying
    ///   counter is `0`) → "observed, currently zero" → real-data
    ///   path; the estimator honors the zero (and divides by total
    ///   carefully per `clamp_unit`).
    ///
    /// This matters at recovery: a cold-restart with empty stats
    /// SHOULD return `None` (so the cost planner uses defaults), not
    /// `Some(0)` (which would over-prune all plans). Persistence /
    /// recovery semantics for `commits_observed` are wired by
    /// M4-04a's deferral options (a)–(d) — see ADR-038 §2 D-25.
    #[must_use]
    pub fn has_observed_any(&self) -> bool {
        self.commits_observed.load(Ordering::Relaxed) > 0
    }

    /// Read the per-tenant commit counter directly. Mirrors the
    /// `(label_count, total_node_count, total_rel_count)` accessors
    /// — this is the cardinality primitive the K-1 recovery oracle
    /// compares against the rebuilt-from-ledger commit count to
    /// detect M4-41 stats-persistence drift. Returns the raw count
    /// (0 if `observe_commit` has never been called for this
    /// tenant); the `None` / `Some` boundary translation is the
    /// caller's responsibility via [`Self::has_observed_any`].
    ///
    /// **Post-recovery semantics (M4-41):** after a cold-start rebuild
    /// (per ADR-038 amendment-06 §D-25.1 / [`crate::recovery::stats_rebuild`])
    /// this counter equals exactly `1` for every tenant whose MVCC
    /// slice was walked during the rebuild, regardless of how many
    /// commits the tenant accumulated pre-crash. The single coalesced
    /// rebuild bracket represents "all pre-crash commits observed"
    /// as one observation event. Consumers that depend on a
    /// monotonically-non-decreasing commit count across restarts
    /// should NOT use this counter as the source of truth.
    #[must_use]
    pub fn commits_observed_count(&self) -> u64 {
        self.commits_observed.load(Ordering::Relaxed)
    }

    /// Read the per-label cardinality. `None` if the label has never
    /// been observed; `Some(0)` if observed-then-fully-deleted.
    #[must_use]
    pub fn label_cardinality(&self, label: LabelId) -> Option<u64> {
        self.label_counts
            .get(&label)
            .map(|entry| entry.value().load(Ordering::Relaxed))
    }

    /// Read the per-rel-type cardinality. Same `None` / `Some(0)`
    /// semantics as [`Self::label_cardinality`].
    #[must_use]
    pub fn rel_type_cardinality(&self, rel_type: TypeId) -> Option<u64> {
        self.rel_type_counts
            .get(&rel_type)
            .map(|entry| entry.value().load(Ordering::Relaxed))
    }

    /// Read the tenant-wide total node count. `None` until the first
    /// commit lands; `Some(count)` thereafter.
    #[must_use]
    pub fn total_node_count(&self) -> Option<u64> {
        if self.has_observed_any() {
            Some(self.total_nodes.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    /// Read the tenant-wide total relationship count. Same semantics
    /// as [`Self::total_node_count`].
    #[must_use]
    pub fn total_rel_count(&self) -> Option<u64> {
        if self.has_observed_any() {
            Some(self.total_rels.load(Ordering::Relaxed))
        } else {
            None
        }
    }

    /// M4-04e (issue #210): capture a plan-time snapshot of all
    /// counters under a single Acquire barrier so consumers — primarily
    /// the M4-05 (M4-51) cost planner — see a coherent point-in-time
    /// view across totals + per-label + per-rel-type cardinalities.
    ///
    /// The returned [`CatalogSnapshot`] satisfies the cross-key
    /// invariants
    ///
    /// - `sum(label_cards) ≤ total_nodes`,
    /// - `sum(rel_type_cards) ≤ total_rels`,
    ///
    /// for every snapshot instance. The per-counter accessors
    /// ([`Self::label_cardinality`], [`Self::total_node_count`], etc.)
    /// do NOT provide this guarantee — they use `Relaxed` ordering for
    /// the lock-free fast path; reading them in sequence can straddle
    /// a commit and produce an inconsistent view (this is the bug that
    /// the M4-2x light-pass retro Recommendation #4 / issue #210
    /// surfaced).
    ///
    /// See module docs §"Cross-key snapshot mechanism" for the full
    /// protocol; the high-level shape is a two-marker SeqLock-style
    /// retry loop keyed on `Self::commits_started` +
    /// `Self::commits_observed`:
    ///
    /// 1. Acquire-load `commits_observed` (`o1`).
    /// 2. Acquire-load `commits_started` (`s1`); if `s1 != o1`, a
    ///    commit is in-flight on some thread — retry.
    /// 3. Read all interior counters with `Relaxed`.
    /// 4. Acquire-load `commits_started` again (`s2`); if `s2 != s1`,
    ///    a new commit started during the read — retry. (No need to
    ///    re-load `commits_observed`: `commits_started ≥
    ///    commits_observed` always, so if no commit started, none
    ///    completed either.)
    /// 5. Return the snapshot.
    ///
    /// Under v1.0 commit serialization (per ADR-031 / ADR-034) the
    /// retry rate is bounded; in practice ≤ 1 retry. The retry loop
    /// makes monotonic progress because [`Self::begin_commit_observation`]
    /// strictly increases `commits_started`.
    ///
    /// # Performance
    ///
    /// `O(label_count + rel_type_count)` per call. At v1.0 tenant
    /// sizes (≤ 100 labels per tenant typical) this is sub-microsecond
    /// in benches and well inside the M4-05 plan-build 5 ms budget
    /// (ADR-036 §D-25). See the `catalog_stats_snapshot` Criterion
    /// bench in `benches/catalog_stats_snapshot.rs` for numbers.
    ///
    /// Snapshot capture is per tenant.
    #[must_use]
    pub fn snapshot(&self) -> CatalogSnapshot {
        // Two-marker SeqLock retry loop. Module docs §"Cross-key
        // snapshot mechanism" has the full protocol justification.
        loop {
            // 1. Acquire-load commits_observed. Pairs with the most
            //    recent observe_commit Release; all per-counter writes
            //    from commits with index ≤ o1 are visible.
            let o1 = self.commits_observed.load(Ordering::Acquire);
            // 2. Acquire-load commits_started. If a commit is
            //    currently in flight, started > observed; retry.
            let s1 = self.commits_started.load(Ordering::Acquire);
            if s1 != o1 {
                continue;
            }

            // 3. Read interior counters with Relaxed. The two Acquire
            //    loads above guarantee that no commit is mid-flight
            //    at this instant on any thread; the Relaxed reads
            //    therefore see only state from completed commits ≤ o1.
            let total_nodes = self.total_nodes.load(Ordering::Relaxed);
            let total_rels = self.total_rels.load(Ordering::Relaxed);

            let mut label_cards: Vec<(LabelId, u64)> = self
                .label_counts
                .iter()
                .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
                .collect();
            let mut rel_type_cards: Vec<(TypeId, u64)> = self
                .rel_type_counts
                .iter()
                .map(|entry| (*entry.key(), entry.value().load(Ordering::Relaxed)))
                .collect();
            let mut max_out_degree: Vec<MaxOutDegreeEntry> = self
                .max_out_degree
                .iter()
                .flat_map(|entry| entry.value().lock().entries(*entry.key()))
                .collect();

            // 4. Acquire-load commits_started again. If a new commit
            //    started during the read window, retry — the counters
            //    we just read may include partial mid-commit values.
            //    No need to re-load commits_observed: commits_started
            //    ≥ commits_observed is invariant; if no commit started,
            //    none completed either.
            let s2 = self.commits_started.load(Ordering::Acquire);
            if s2 != s1 {
                continue;
            }

            // Sort for deterministic iteration order; binary-search
            // lookup is downstream-helpful for the M4-05 planner.
            label_cards.sort_unstable_by_key(|(label, _)| label.raw());
            rel_type_cards.sort_unstable_by_key(|(rel_type, _)| rel_type.raw());
            max_out_degree.sort_unstable_by(|a, b| {
                a.label
                    .raw()
                    .cmp(&b.label.raw())
                    .then_with(|| a.rel_type.raw().cmp(&b.rel_type.raw()))
                    .then_with(|| b.degree.cmp(&a.degree))
                    .then_with(|| a.vertex.raw().cmp(&b.vertex.raw()))
            });

            // Mirror the per-counter accessors' None / Some(0)
            // distinction: pre-first-commit, totals are None (use
            // selectivity defaults); post-first-commit, totals are
            // Some(_). Per-label / per-rel-type entries that are
            // absent simply do not appear in the cards Vec; consumers
            // call `label_card(label)` / `rel_type_card(rel_type)`
            // and treat None as the "never observed" sentinel.
            let observed_any = o1 > 0;
            return CatalogSnapshot {
                total_nodes: if observed_any {
                    Some(total_nodes)
                } else {
                    None
                },
                total_rels: if observed_any { Some(total_rels) } else { None },
                commits_observed: o1,
                label_cards,
                rel_type_cards,
                max_out_degree,
            };
        }
    }
}

/// Plan-time snapshot of [`CatalogStats`] providing cross-key
/// consistency for the M4-05 (M4-51) cost planner — issue #210.
///
/// Captured by [`CatalogStats::snapshot`]. Every snapshot satisfies
/// the cross-key invariants
///
/// - `sum(label_cards) ≤ total_nodes` (when `total_nodes` is `Some`),
/// - `sum(rel_type_cards) ≤ total_rels` (when `total_rels` is `Some`),
///
/// because all counters were read under a single Acquire barrier (see
/// [`CatalogStats::snapshot`] rustdoc and module docs §"Cross-key
/// snapshot mechanism" for the protocol).
///
/// `CatalogSnapshot` is `Clone` + `Send` + `Sync`; the M4-05 cost
/// planner takes ONE snapshot per plan and reads from it through many
/// predicates without re-paying the snapshot cost or interleaving with
/// a concurrent commit.
///
/// # `None` vs `Some(0)` semantics
///
/// Mirrors the per-counter accessors:
///
/// - `total_nodes() == None` → no commit has been observed yet; the
///   M4-05 cost planner SHOULD use `DEFAULT_*_SELECTIVITY` constants
///   per `arcgraph_query::semantic::selectivity::SelectivityEstimator`.
/// - `total_nodes() == Some(0)` → commits observed; current count is
///   zero (e.g., observed-then-fully-deleted). The planner divides by
///   total only after handling this case.
/// - `label_card(label) == None` → that label has never been observed
///   by the commit pipeline.
/// - `label_card(label) == Some(0)` → label observed; current count
///   is zero (observed-then-fully-deleted).
///
/// `DEFAULT_*_SELECTIVITY`: arcgraph_query::semantic::selectivity
#[derive(Debug, Clone)]
pub struct CatalogSnapshot {
    /// Tenant-wide total node count at snapshot capture. `None` if no
    /// commit has been observed yet.
    total_nodes: Option<u64>,
    /// Tenant-wide total relationship count at snapshot capture.
    /// `None` if no commit has been observed yet.
    total_rels: Option<u64>,
    /// Commit-counter value at snapshot capture. `0` for fresh-tenant
    /// snapshots (no commit observed); used as the SeqLock generation
    /// for debugging / future tracing hooks.
    commits_observed: u64,
    /// Per-label cardinalities at snapshot capture. Sorted by
    /// [`LabelId`] for deterministic iteration and binary-search
    /// lookup. Labels never observed by the commit pipeline are NOT
    /// present in this Vec; [`Self::label_card`] translates absence to
    /// `None`.
    label_cards: Vec<(LabelId, u64)>,
    /// Per-rel-type cardinalities at snapshot capture. Sorted by
    /// [`TypeId`]; same absence semantics as `label_cards`.
    rel_type_cards: Vec<(TypeId, u64)>,
    /// ADR-025 §5 bounded top out-degree entries, sorted by
    /// `(label, rel_type, degree desc, vertex)`.
    max_out_degree: Vec<MaxOutDegreeEntry>,
}

impl CatalogSnapshot {
    /// Tenant-wide total node count at snapshot capture. `None` until
    /// the first commit has been observed; `Some(_)` thereafter.
    #[must_use]
    pub fn total_nodes(&self) -> Option<u64> {
        self.total_nodes
    }

    /// Tenant-wide total relationship count at snapshot capture.
    /// Mirrors [`Self::total_nodes`] semantics.
    #[must_use]
    pub fn total_rels(&self) -> Option<u64> {
        self.total_rels
    }

    /// Number of commits observed up to and including this snapshot.
    /// `0` for a fresh-tenant snapshot. Useful for debugging and for
    /// the K-1 recovery oracle (compares against the rebuilt-from-
    /// ledger count). Distinct from [`Self::has_observed_any`] in
    /// that this returns the raw count, not just the boolean.
    ///
    /// **Post-recovery semantics (M4-41):** after a cold-start rebuild
    /// (per ADR-038 amendment-06 §D-25.1 / [`crate::recovery::stats_rebuild`])
    /// this value equals exactly `1` for every tenant whose MVCC
    /// slice was walked during the rebuild, regardless of how many
    /// commits the tenant accumulated pre-crash. The single coalesced
    /// rebuild bracket represents "all pre-crash commits observed"
    /// as one observation event. Consumers that depend on a
    /// monotonically-non-decreasing commit count across restarts
    /// should NOT use this counter as the source of truth.
    #[must_use]
    pub fn commits_observed(&self) -> u64 {
        self.commits_observed
    }

    /// `true` iff at least one commit has been observed before this
    /// snapshot was captured. Mirrors
    /// [`CatalogStats::has_observed_any`].
    #[must_use]
    pub fn has_observed_any(&self) -> bool {
        self.commits_observed > 0
    }

    /// Per-label cardinality at snapshot capture. Returns `None` if
    /// `label` has never been observed by the commit pipeline (the
    /// "never observed" sentinel — distinct from `Some(0)` =
    /// "observed-then-fully-deleted").
    ///
    /// O(log n) — binary search over the sorted `label_cards` Vec.
    #[must_use]
    pub fn label_card(&self, label: LabelId) -> Option<u64> {
        self.label_cards
            .binary_search_by_key(&label.raw(), |(l, _)| l.raw())
            .ok()
            .map(|idx| self.label_cards[idx].1)
    }

    /// Per-rel-type cardinality at snapshot capture. Same `None` /
    /// `Some(0)` semantics as [`Self::label_card`].
    #[must_use]
    pub fn rel_type_card(&self, rel_type: TypeId) -> Option<u64> {
        self.rel_type_cards
            .binary_search_by_key(&rel_type.raw(), |(t, _)| t.raw())
            .ok()
            .map(|idx| self.rel_type_cards[idx].1)
    }

    /// All per-label cardinalities, sorted by [`LabelId`]. Yields the
    /// canonical iteration order for cost-planner code that walks the
    /// full label set (e.g., bulk per-label selectivity ratios).
    #[must_use]
    pub fn label_cards(&self) -> &[(LabelId, u64)] {
        &self.label_cards
    }

    /// All per-rel-type cardinalities, sorted by [`TypeId`]. Mirrors
    /// [`Self::label_cards`].
    #[must_use]
    pub fn rel_type_cards(&self) -> &[(TypeId, u64)] {
        &self.rel_type_cards
    }

    /// ADR-025 §5 `max_out_degree_sketch[label, rel_type]` degree
    /// estimate entries. Degrees are overestimates after deletes and
    /// under space-saving eviction, which is safe for the planner's
    /// conservative supernode detector.
    #[must_use]
    pub fn max_out_degree_entries(&self) -> &[MaxOutDegreeEntry] {
        &self.max_out_degree
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn max_out_degree_space_saving_evicts_min_preserves_hub_and_overestimates() {
        let stats = CatalogStats::new();
        let label = LabelId::new(1);
        let rel_type = TypeId::new(2);
        let hub = NodeId::new(42);

        stats.begin_commit_observation();
        for _ in 0..25 {
            stats.record_out_degree(label, rel_type, hub);
        }
        for raw in 1_000..1_100 {
            stats.record_out_degree(label, rel_type, NodeId::new(raw));
        }
        let evicted_min = NodeId::new(1_000);
        stats.observe_commit();

        let entries = stats.snapshot().max_out_degree_entries().to_vec();
        assert_eq!(entries.len(), MAX_OUT_DEGREE_SKETCH_CAP);
        assert!(
            entries
                .iter()
                .all(|entry| entry.degree >= if entry.vertex == hub { 25 } else { 1 }),
            "space-saving entries must overestimate true counts: {entries:?}"
        );
        assert!(
            entries
                .iter()
                .any(|entry| entry.vertex == hub && entry.degree >= 25),
            "high-degree hub must survive churn: {entries:?}"
        );
        assert!(
            !entries.iter().any(|entry| entry.vertex == evicted_min),
            "the deterministic minimum victim should be evicted: {entries:?}"
        );
    }

    #[test]
    fn stats_increment_label_then_read_returns_count() {
        // M4-41: simplest contract — N increments produce Some(N).
        let stats = CatalogStats::new();
        let person = LabelId::new(1);

        // Pre-increment: never-observed label returns None.
        assert_eq!(stats.label_cardinality(person), None);

        for _ in 0..5 {
            stats.increment_label(person);
        }
        assert_eq!(stats.label_cardinality(person), Some(5));

        // Different label remains untouched.
        let doc = LabelId::new(2);
        assert_eq!(stats.label_cardinality(doc), None);

        // Symmetric for rel-types.
        let knows = TypeId::new(1);
        assert_eq!(stats.rel_type_cardinality(knows), None);
        for _ in 0..3 {
            stats.increment_rel_type(knows);
        }
        assert_eq!(stats.rel_type_cardinality(knows), Some(3));

        // Totals require an observe_commit before they surface.
        assert_eq!(stats.total_node_count(), None);
        assert_eq!(stats.total_rel_count(), None);
        stats.observe_commit();
        assert_eq!(stats.total_node_count(), Some(0));
        assert_eq!(stats.total_rel_count(), Some(0));
    }

    #[test]
    fn stats_decrement_below_zero_saturates_at_zero() {
        // Defensive saturation: a delete should never produce a
        // negative count, even under buggy upstream call patterns.
        // The trait surface is `u64`, so an underflow would wrap
        // to ~2^64-1 — catastrophic for a cost planner.
        let stats = CatalogStats::new();
        let person = LabelId::new(1);

        stats.increment_label(person);
        stats.increment_label(person); // count = 2
        stats.decrement_label(person);
        stats.decrement_label(person);
        stats.decrement_label(person); // attempt to go below 0
        stats.decrement_label(person); // and again
        assert_eq!(stats.label_cardinality(person), Some(0));

        // Decrement on never-incremented label is a defensive no-op.
        let phantom = LabelId::new(99);
        stats.decrement_label(phantom);
        assert_eq!(stats.label_cardinality(phantom), None);

        // Same shape for rel-types and totals.
        let knows = TypeId::new(1);
        stats.increment_rel_type(knows);
        stats.decrement_rel_type(knows);
        stats.decrement_rel_type(knows);
        assert_eq!(stats.rel_type_cardinality(knows), Some(0));

        stats.observe_commit();
        stats.increment_total_nodes();
        stats.decrement_total_nodes();
        stats.decrement_total_nodes();
        assert_eq!(stats.total_node_count(), Some(0));

        stats.increment_total_rels();
        stats.decrement_total_rels();
        stats.decrement_total_rels();
        assert_eq!(stats.total_rel_count(), Some(0));
    }

    #[test]
    fn stats_per_tenant_isolation() {
        // Multi-tenant invariant: each tenant's CatalogStats is an
        // independent instance. Mutating one MUST NOT be visible
        // through the other. This test exercises the structural
        // isolation — there is no shared global state to leak across.
        let tenant_a = CatalogStats::new();
        let tenant_b = CatalogStats::new();
        let label = LabelId::new(1);
        let rel = TypeId::new(1);

        for _ in 0..7 {
            tenant_a.increment_label(label);
            tenant_a.increment_rel_type(rel);
            tenant_a.increment_total_nodes();
            tenant_a.increment_total_rels();
        }
        tenant_a.observe_commit();

        // Tenant B is fully untouched.
        assert_eq!(tenant_a.label_cardinality(label), Some(7));
        assert_eq!(tenant_b.label_cardinality(label), None);
        assert_eq!(tenant_a.rel_type_cardinality(rel), Some(7));
        assert_eq!(tenant_b.rel_type_cardinality(rel), None);
        assert_eq!(tenant_a.total_node_count(), Some(7));
        assert_eq!(tenant_b.total_node_count(), None);
        assert_eq!(tenant_a.total_rel_count(), Some(7));
        assert_eq!(tenant_b.total_rel_count(), None);
    }

    #[test]
    fn stats_commits_observed_count_returns_observed_total() {
        // K-1 oracle: the commit counter is the cardinality primitive
        // the K-1 recovery oracle compares against the rebuilt-from-
        // ledger count. `commits_observed_count` returns 0 before any
        // commit; equals the number of `observe_commit` calls thereafter.
        let stats = CatalogStats::new();
        assert_eq!(stats.commits_observed_count(), 0);
        stats.observe_commit();
        assert_eq!(stats.commits_observed_count(), 1);
        for _ in 0..4 {
            stats.observe_commit();
        }
        assert_eq!(stats.commits_observed_count(), 5);
        // has_observed_any agrees with > 0 invariant.
        assert!(stats.has_observed_any());
    }

    #[test]
    fn stats_concurrent_increments_eventually_consistent() {
        // DashMap + AtomicU64 thread safety pin: N threads each
        // perform M increments; final counter equals N*M. The
        // `Relaxed` ordering on the counters is sufficient because
        // we only care about the cumulative count — there is no
        // happens-before edge between increments and other state.
        const N_THREADS: usize = 8;
        const M_PER_THREAD: u64 = 1_000;

        let stats = Arc::new(CatalogStats::new());
        let label = LabelId::new(1);
        let rel = TypeId::new(1);

        let mut handles = Vec::new();
        for _ in 0..N_THREADS {
            let stats = Arc::clone(&stats);
            handles.push(thread::spawn(move || {
                for _ in 0..M_PER_THREAD {
                    stats.increment_label(label);
                    stats.increment_rel_type(rel);
                    stats.increment_total_nodes();
                    stats.increment_total_rels();
                }
            }));
        }
        for h in handles {
            h.join().expect("thread panicked");
        }
        stats.observe_commit();

        let expected = N_THREADS as u64 * M_PER_THREAD;
        assert_eq!(stats.label_cardinality(label), Some(expected));
        assert_eq!(stats.rel_type_cardinality(rel), Some(expected));
        assert_eq!(stats.total_node_count(), Some(expected));
        assert_eq!(stats.total_rel_count(), Some(expected));
    }

    // ─── M4-04e (issue #210) snapshot() tests ─────────────────────

    #[test]
    fn snapshot_pre_first_commit_returns_none_totals() {
        // Per the per-counter accessor contract, totals are `None`
        // until the first `observe_commit` lands. The snapshot mirrors
        // that contract: a fresh-tenant snapshot has both totals
        // `None`, an empty `label_cards` Vec, and `commits_observed = 0`.
        let stats = CatalogStats::new();
        let snap = stats.snapshot();
        assert_eq!(snap.total_nodes(), None);
        assert_eq!(snap.total_rels(), None);
        assert_eq!(snap.commits_observed(), 0);
        assert!(!snap.has_observed_any());
        assert!(snap.label_cards().is_empty());
        assert!(snap.rel_type_cards().is_empty());

        // Per-key lookup on an absent label / rel-type returns None.
        assert_eq!(snap.label_card(LabelId::new(1)), None);
        assert_eq!(snap.rel_type_card(TypeId::new(1)), None);
    }

    #[test]
    fn snapshot_after_n_commits_returns_some_n_and_expected_totals() {
        // Per-commit pattern: begin_commit_observation + increments +
        // observe_commit. After N commits, snapshot should report:
        // - commits_observed = N
        // - total_nodes / total_rels = Some(<commit-derived count>)
        // - per-label / per-rel-type cards reflecting the increments
        //   applied since stats started.
        let stats = CatalogStats::new();
        let person = LabelId::new(1);
        let knows = TypeId::new(1);

        // 3 commits, each adds 2 nodes and 1 rel. begin/observe
        // bracket each commit per the M4-04e two-marker SeqLock
        // protocol.
        for _ in 0..3 {
            stats.begin_commit_observation();
            stats.increment_label(person);
            stats.increment_label(person);
            stats.increment_total_nodes();
            stats.increment_total_nodes();
            stats.increment_rel_type(knows);
            stats.increment_total_rels();
            stats.observe_commit();
        }

        let snap = stats.snapshot();
        assert_eq!(snap.commits_observed(), 3);
        assert!(snap.has_observed_any());
        assert_eq!(snap.total_nodes(), Some(6));
        assert_eq!(snap.total_rels(), Some(3));
        assert_eq!(snap.label_card(person), Some(6));
        assert_eq!(snap.rel_type_card(knows), Some(3));

        // Cards Vec is sorted-by-id and contains exactly the observed
        // labels.
        assert_eq!(snap.label_cards(), &[(person, 6)]);
        assert_eq!(snap.rel_type_cards(), &[(knows, 3)]);
    }

    #[test]
    fn snapshot_label_cards_sorted_by_label_id() {
        // The snapshot's `label_cards` Vec is sorted by `LabelId::raw()`
        // for deterministic iteration and binary-search lookup. Insert
        // labels out of order; the snapshot must still return them sorted.
        let stats = CatalogStats::new();
        let labels = [
            LabelId::new(7),
            LabelId::new(2),
            LabelId::new(11),
            LabelId::new(4),
        ];
        stats.begin_commit_observation();
        for (i, l) in labels.iter().enumerate() {
            for _ in 0..(i + 1) {
                stats.increment_label(*l);
                stats.increment_total_nodes();
            }
        }
        stats.observe_commit();

        let snap = stats.snapshot();
        let raws: Vec<u32> = snap.label_cards().iter().map(|(l, _)| l.raw()).collect();
        let mut expected_sorted = raws.clone();
        expected_sorted.sort();
        assert_eq!(
            raws, expected_sorted,
            "label_cards must be sorted by LabelId::raw()"
        );

        // Same shape for rel-types.
        let rel_types = [TypeId::new(5), TypeId::new(1), TypeId::new(9)];
        stats.begin_commit_observation();
        for (i, t) in rel_types.iter().enumerate() {
            for _ in 0..(i + 1) {
                stats.increment_rel_type(*t);
                stats.increment_total_rels();
            }
        }
        stats.observe_commit();

        let snap = stats.snapshot();
        let raws: Vec<u32> = snap.rel_type_cards().iter().map(|(t, _)| t.raw()).collect();
        let mut expected_sorted = raws.clone();
        expected_sorted.sort();
        assert_eq!(
            raws, expected_sorted,
            "rel_type_cards must be sorted by TypeId::raw()"
        );
    }

    #[test]
    fn snapshot_clone_yields_independent_consistent_view() {
        // CatalogSnapshot is Clone — the M4-05 cost planner takes one
        // snapshot per plan and may clone it across the planner sub-
        // tasks. Cloning produces an independent value with the same
        // observed state.
        let stats = CatalogStats::new();
        let person = LabelId::new(1);
        stats.begin_commit_observation();
        for _ in 0..4 {
            stats.increment_label(person);
            stats.increment_total_nodes();
        }
        stats.observe_commit();

        let snap = stats.snapshot();
        let cloned = snap.clone();
        assert_eq!(cloned.total_nodes(), snap.total_nodes());
        assert_eq!(cloned.commits_observed(), snap.commits_observed());
        assert_eq!(cloned.label_card(person), snap.label_card(person));
    }

    #[test]
    fn snapshot_internal_consistency_under_concurrent_commits() {
        // Cross-key invariant under concurrent commits: the snapshot
        // mechanism (SeqLock-style retry on `commits_observed`) must
        // ALWAYS return a snapshot satisfying:
        //   sum(label_cards) ≤ total_nodes
        //   sum(rel_type_cards) ≤ total_rels
        //
        // This is the core M4-04e correctness pin. Without the
        // mechanism, a snapshot reading old `total_nodes` and new
        // per-label increments could violate the invariant.
        //
        // 4 writer threads each do 500 commit-shaped sequences
        // (increment label + increment total_nodes + observe_commit).
        // 1 snapshot thread takes 200 snapshots and asserts the
        // invariant on each. The test asserts the cross-key
        // invariant holds on EVERY snapshot.
        const WRITER_THREADS: usize = 4;
        const COMMITS_PER_WRITER: u64 = 500;
        const SNAPSHOTS: usize = 200;

        let stats = Arc::new(CatalogStats::new());
        // Use a label space of 8 labels; writers cycle through them
        // to stress per-label / total interaction.
        let labels: Vec<LabelId> = (0..8).map(LabelId::new).collect();
        let rel_types: Vec<TypeId> = (0..8).map(TypeId::new).collect();

        let mut writer_handles = Vec::new();
        for tid in 0..WRITER_THREADS {
            let stats = Arc::clone(&stats);
            let labels = labels.clone();
            let rel_types = rel_types.clone();
            writer_handles.push(thread::spawn(move || {
                for i in 0..COMMITS_PER_WRITER {
                    let li = (tid as u64 + i) as usize % labels.len();
                    let ri = (tid as u64 + i) as usize % rel_types.len();
                    // M4-04e commit shape: begin_commit_observation
                    // → increments → observe_commit.
                    stats.begin_commit_observation();
                    stats.increment_label(labels[li]);
                    stats.increment_total_nodes();
                    stats.increment_rel_type(rel_types[ri]);
                    stats.increment_total_rels();
                    stats.observe_commit();
                }
            }));
        }

        let stats_reader = Arc::clone(&stats);
        let reader_handle = thread::spawn(move || {
            for _ in 0..SNAPSHOTS {
                let snap = stats_reader.snapshot();
                // Skip the pre-first-commit window — totals are None
                // and per-label sums are trivially empty.
                if let Some(total_nodes) = snap.total_nodes() {
                    let sum_labels: u64 = snap.label_cards().iter().map(|(_, c)| *c).sum();
                    assert!(
                        sum_labels <= total_nodes,
                        "cross-key invariant violated: sum(label_cards)={} > total_nodes={}",
                        sum_labels,
                        total_nodes
                    );
                }
                if let Some(total_rels) = snap.total_rels() {
                    let sum_rels: u64 = snap.rel_type_cards().iter().map(|(_, c)| *c).sum();
                    assert!(
                        sum_rels <= total_rels,
                        "cross-key invariant violated: sum(rel_type_cards)={} > total_rels={}",
                        sum_rels,
                        total_rels
                    );
                }
            }
        });

        for h in writer_handles {
            h.join().expect("writer panicked");
        }
        reader_handle.join().expect("reader panicked");

        // Final state is a quiescent invariant: every commit landed,
        // so sum(label_cards) MUST equal total_nodes and similarly
        // for rels.
        let final_snap = stats.snapshot();
        let total_commits = WRITER_THREADS as u64 * COMMITS_PER_WRITER;
        assert_eq!(final_snap.commits_observed(), total_commits);
        let sum_labels: u64 = final_snap.label_cards().iter().map(|(_, c)| *c).sum();
        let sum_rels: u64 = final_snap.rel_type_cards().iter().map(|(_, c)| *c).sum();
        assert_eq!(final_snap.total_nodes(), Some(sum_labels));
        assert_eq!(final_snap.total_rels(), Some(sum_rels));
        assert_eq!(final_snap.total_nodes(), Some(total_commits));
    }

    #[test]
    fn snapshot_some_zero_after_increment_then_full_decrement() {
        // The `Some(0)` ("observed-then-fully-deleted") sentinel is
        // distinct from `None` ("never observed"); the snapshot
        // preserves the distinction.
        let stats = CatalogStats::new();
        let person = LabelId::new(1);
        let knows = TypeId::new(1);

        stats.begin_commit_observation();
        stats.increment_label(person);
        stats.increment_rel_type(knows);
        stats.increment_total_nodes();
        stats.increment_total_rels();
        stats.observe_commit();
        stats.begin_commit_observation();
        stats.decrement_label(person);
        stats.decrement_rel_type(knows);
        stats.decrement_total_nodes();
        stats.decrement_total_rels();
        stats.observe_commit();

        let snap = stats.snapshot();
        assert_eq!(snap.commits_observed(), 2);
        assert_eq!(snap.total_nodes(), Some(0));
        assert_eq!(snap.total_rels(), Some(0));
        // Per-label / per-rel-type entries persist as Some(0) — the
        // map key is sticky once observed (saturating-decrement
        // invariant).
        assert_eq!(snap.label_card(person), Some(0));
        assert_eq!(snap.rel_type_card(knows), Some(0));
        // Never-touched entries remain None.
        assert_eq!(snap.label_card(LabelId::new(99)), None);
    }
}
