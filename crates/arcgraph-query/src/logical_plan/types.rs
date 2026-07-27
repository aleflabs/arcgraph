//! M4-31 / M4-32 / M4-33 LogicalPlan tree types.
//!
//! [`LogicalPlan`] is a parallel tree to [`crate::semantic::bound_ast`]
//! produced by [`crate::logical_plan::lowering::LogicalPlanLoweringVisitor`].
//! It is purely "what to compute"; the M4-05 cost-based planner
//! consumes [`LogicalPlan`] and produces a `PhysicalPlan` carrying
//! cost-model + operator-substitution decisions.
//!
//! # Variant scope at M4-31 + M4-32 + M4-33
//!
//! M4-31 shipped the SIMPLE operators:
//! - [`LogicalScan`] — node-pattern scan (label-filtered if present);
//! - [`LogicalExpand`] — relationship-pattern traversal;
//! - [`LogicalFilter`] — WHERE / WITH WHERE predicate;
//! - [`LogicalProject`] — RETURN / WITH projection;
//! - [`LogicalJoin`] — implicit join from multiple MATCH patterns
//!   sharing a variable;
//! - [`LogicalLimit`] — RETURN ... LIMIT N;
//! - [`LogicalSkip`] — RETURN ... SKIP N;
//! - [`LogicalEmpty`] — sentinel for the degenerate empty-clauses case.
//!
//! M4-32 added the HYBRID-retrieval + OPTIONAL MATCH operators
//! additively:
//! - [`LogicalRankByHybrid`] — top-level hybrid retrieval orchestration;
//! - [`LogicalFusion`] — RRF fusion (CombSUM/CombMnz/Ltr remain
//!   reserved);
//! - [`LogicalCommunityLookup`] — community-membership filter (closes
//!   PR #154 reviewer Finding 5; lowers IN-COMMUNITY ↔ canonical
//!   `community(n) = $cid` to identical trees);
//! - [`LogicalVectorNear`] — vector ANN retrieval;
//! - [`LogicalTextMatch`] — BM25 text search;
//! - [`LogicalLeftOuterJoin`] — OPTIONAL MATCH lowering per ADR-006
//!   amendment-01 §A-2.
//!
//! M4-33 (this slice) closes the M4-03 substrate by adding the
//! aggregation + sort + DISTINCT + UNWIND + named-path + dynamic-LIMIT
//! operators additively:
//! - [`LogicalAggregate`] — GROUP BY + aggregation functions
//!   (`count` / `sum` / `avg` / `min` / `max` / `collect`) per
//!   openCypher 9 §6.4;
//! - [`LogicalSort`] — ORDER BY (replacing both the tail-clause and
//!   return-clause M4-31 deferral sites) per Cypher 9 §6.6;
//! - [`LogicalDistinct`] — RETURN DISTINCT;
//! - [`LogicalUnwind`] — `UNWIND <list> AS <var>` per Cypher 9 §6.7;
//! - [`LogicalNamedPath`] — named path `p = (a)-[..]->(b)` plus
//!   `p = SHORTEST_PATH(...)` per Cypher 9 §6.5;
//! - [`LogicalDynamicLimit`] — non-literal LIMIT / SKIP backed by a
//!   parameter or expression.
//!
//! UNION (multi-statement composition) defers to M4-08.
//!
//! Each unsupported surface raises
//! [`crate::logical_plan::error::LogicalPlanError::NotImplementedAtM4_31`]
//! with the `target_slice` slot naming the future slice.
//!
//! # Span discipline
//!
//! Every [`LogicalPlan`] node carries [`Span`] for IDE-grade error
//! reporting + future M4-91 EXPLAIN output. Mirrors the M4-22 / M4-23
//! span discipline.
//!
//! # Exhaustive-match contract
//!
//! `LogicalPlan` is **NOT** `#[non_exhaustive]`. Per the M4-21 / M4-22
//! / M4-23 / M4-31 / M4-32 / M4-33 surface convention, the variant
//! set is each slice's public contract for downstream M4-05 / M4-06
//! consumption; every consumer MUST exhaustively match. New variants
//! land via amendment alongside future slices.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 — logical-plan-types contract (M4-31 baseline).
//! - ADR-038 §2 D-26 — hybrid retrieval lowering + OPTIONAL MATCH
//!   contract (M4-32 baseline).
//! - ADR-038 §2 D-28 — aggregation + sort + path operators contract
//!   (this slice's primary spec).
//! - ADR-038 §2 D-23 — visitor-trait discipline lock (M4-31, M4-32,
//!   M4-33 inherit; the lowering walker is a CUSTOM struct, not a
//!   trait).
//! - ADR-036 §D-28 — plan-tree operator taxonomy (the M4-32 / M4-33
//!   variants land via this taxonomy).
//! - ADR-006 amendment-01 §A-2 — OPTIONAL MATCH at v1.0 lowers to a
//!   left-outer join per Cypher 9 §6.5.

use arcgraph_core::{LabelId, Lsn, TypeId};

use crate::ast::{CreateRelDirection, LengthRange, RelDirection};
use crate::error::Span;
use crate::semantic::bound_ast::{BindingId, BoundExpression, BoundProjectionItem};

/// Direction of a [`LogicalExpand`] relationship traversal.
///
/// Re-export of [`crate::ast::RelDirection`] is intentionally avoided —
/// the AST type carries openCypher-specific semantics ("source vs.
/// target syntactic ordering"); the LogicalPlan variant carries
/// execution-relevant semantics ("which endpoint binds first").
/// At M4-31 the two are isomorphic, so we map directly; if M4-05
/// introduces direction-rewriting (e.g., flipping an undirected
/// edge for cost reasons) the LogicalPlan-side type can diverge
/// without touching the AST contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// `(from)-[r]->(to)`. Bind `from` first; expand to `to`.
    LeftToRight,
    /// `(from)<-[r]-(to)`. Bind `from` first; expand to `to`
    /// (reversed-edge traversal).
    RightToLeft,
    /// `(from)-[r]-(to)`. Either direction admissible.
    Undirected,
}

impl From<&RelDirection> for Direction {
    fn from(d: &RelDirection) -> Self {
        match d {
            RelDirection::LeftToRight => Direction::LeftToRight,
            RelDirection::RightToLeft => Direction::RightToLeft,
            RelDirection::Undirected => Direction::Undirected,
        }
    }
}

/// Join condition for [`LogicalJoin`].
///
/// `JoinCondition::SharedBindings(vec)` encodes both Cartesian (vec is
/// empty — no shared variables, full cross-product) and equi-join (vec
/// is non-empty — natural-join on shared variables) semantics.
/// Theta-joins (custom predicate) are NOT shipped at v1.0; M4-05
/// cost-planner derives plan-shape from the SharedBindings vec.
///
/// At M4-31 the lowering pass introduces both shapes naturally:
/// multi-pattern MATCH with shared variables (e.g.,
/// `MATCH (a), (a)-[:R]->(b)` — the second pattern's `(a)` shares
/// `binding_id` with the first pattern's `(a)`) yields a non-empty vec
/// (equi-join); multi-pattern MATCH with disjoint variables (e.g.,
/// `MATCH (a), (b)`) yields an empty vec (Cartesian).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JoinCondition {
    /// Equi-join on a list of shared bindings (Cartesian when the list
    /// is empty). Both `left` and `right` inputs MUST produce a
    /// binding for each id in the list.
    SharedBindings(Vec<BindingId>),
}

/// Physical join algorithm picked for a [`LogicalJoin`] per ADR-097.
///
/// W25-M4-61b adds a second join executor flavor ([`crate::executor::ops::merge_join::MergeJoinOp`])
/// alongside the W17α hash-join. The M4-31 lowering ships
/// [`JoinAlgorithm::Auto`] as the default; the M4-51 cost walker +
/// [`crate::planner::pick_join_algorithms`] resolve it to a concrete
/// algorithm based on per-side cardinality estimates from the
/// M4-42 [`crate::semantic::SelectivityEstimator`].
///
/// Tests + EXPLAIN consumers that want to pin a specific algorithm
/// (e.g., the cross-substrate equivalence proptest) construct
/// `LogicalJoin` directly with the desired variant; the picker is a
/// no-op when the algorithm is already concrete.
///
/// # Auto vs. Cartesian
///
/// `JoinCondition::SharedBindings(vec![])` (Cartesian) ALWAYS resolves
/// to [`JoinAlgorithm::HashJoin`] regardless of cost — merge-join is
/// undefined without join keys. The picker enforces this.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JoinAlgorithm {
    /// Cost-based picker has NOT yet resolved the algorithm. The M4-51
    /// cost walker's join-cost function returns
    /// `min(hash_cost, merge_cost)` for `Auto`; the
    /// [`crate::planner::pick_join_algorithms`] pass rewrites `Auto`
    /// → concrete variant before the executor consumes the plan.
    /// Pipeline build defaults `Auto` → `HashJoin` defensively (it
    /// preserves W17α behavior when the picker was not invoked, e.g.,
    /// in tests that call `Pipeline::build` directly without going
    /// through `execute()` / `QueryEngine::execute`).
    #[default]
    Auto,
    /// In-memory hash join (build = LEFT, probe = RIGHT). Cheaper
    /// when one side is small enough to fit the BUILD bucket map
    /// in the per-tenant byte budget. Per ADR-038 §2 D-24 — the
    /// W17α default.
    HashJoin,
    /// Pipeline-aware sort-merge join. Cheaper when BOTH sides arrive
    /// pre-sorted on the join key (e.g., a Scan whose underlying
    /// storage iteration order matches the join key) OR when the
    /// hash-side build would exceed the per-tenant memory cap. Per
    /// ADR-097 cost-model: merge-join cost = sort(L) + sort(R) + merge
    /// (sort terms collapse to zero when input is already sorted).
    MergeJoin,
}

/// Logical-plan node.
///
/// Each variant carries a [`Span`] for IDE error reporting + future
/// M4-91 EXPLAIN output.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalPlan {
    /// Node-pattern scan (label-filtered if `label` is `Some`).
    Scan(LogicalScan),
    /// O(1) counts-store lookup for exact unfiltered `count(*)` /
    /// `count(n)` / `count(r)` queries over a single bare MATCH.
    CountStore(LogicalCountStore),
    /// Relationship-pattern traversal.
    Expand(LogicalExpand),
    /// WHERE / WITH WHERE predicate.
    Filter(LogicalFilter),
    /// RETURN / WITH projection.
    Project(LogicalProject),
    /// Implicit equi-join from multiple MATCH patterns sharing a
    /// variable.
    Join(LogicalJoin),
    /// Left-outer-join from OPTIONAL MATCH per ADR-006 amendment-01
    /// §A-2.
    LeftOuterJoin(LogicalLeftOuterJoin),
    /// `LIMIT N`.
    Limit(LogicalLimit),
    /// `SKIP N`.
    Skip(LogicalSkip),
    /// `RANK BY HYBRID(VECTOR(...), TEXT(...))` orchestration node.
    RankByHybrid(LogicalRankByHybrid),
    /// `WITH FUSION = RRF(k = N)` (only `Rrf` lit at v1.0).
    Fusion(LogicalFusion),
    /// Community-membership lookup. Surfaces:
    /// `n IN COMMUNITY($cid)` (predicate form per ADR-038
    /// amendment-01) AND `community(n) = $cid` (canonical D-4 form)
    /// lower to IDENTICAL `LogicalCommunityLookup` trees per ADR-038
    /// §2 D-26.
    CommunityLookup(LogicalCommunityLookup),
    /// Vector ANN retrieval (`vector_distance(...)` /
    /// `<expr> NEAR <expr>`).
    VectorNear(LogicalVectorNear),
    /// BM25 text search (`text_match(...)` / `<expr> MATCH <expr>`).
    TextMatch(LogicalTextMatch),
    /// GROUP BY + aggregation per openCypher 9 §6.4 (M4-33).
    Aggregate(LogicalAggregate),
    /// ORDER BY (M4-33). Replaces both the tail-clause + return-clause
    /// M4-31 deferral emissions per ADR-038 §2 D-28.
    Sort(LogicalSort),
    /// `RETURN DISTINCT` (M4-33).
    Distinct(LogicalDistinct),
    /// `UNION ALL` set-op concat per ADR-185 (#649-A1, W28 —
    /// openCypher v9 §8). A1 lowers only the keep-duplicates form;
    /// bare `UNION` (distinct) composes a [`LogicalDistinct`] over this
    /// node and lands in #649-A2.
    Union(LogicalUnion),
    /// `UNWIND <list> AS <var>` (M4-33) per Cypher 9 §6.7.
    Unwind(LogicalUnwind),
    /// **ADR-197 (#802)** — `CALL <proc>(args) [YIELD …]` schema-
    /// introspection procedure call, OR a `SHOW …` command. A
    /// generating operator: produces rows from the catalog/intern-table
    /// (no children beyond a leading unit row), each row binding the
    /// YIELD'd / SHOW columns. See [`LogicalProcedureCall`].
    ProcedureCall(LogicalProcedureCall),
    /// Named path `p = (a)-[..]->(b)` or `p = SHORTEST_PATH(...)`
    /// (M4-33) per Cypher 9 §6.5.
    NamedPath(LogicalNamedPath),
    /// Non-literal LIMIT / SKIP backed by a parameter or expression
    /// (M4-33). Replaces the M4-31 `non-literal SKIP` /
    /// `non-literal LIMIT` deferral emissions.
    DynamicLimit(LogicalDynamicLimit),
    /// ADR-147 W26-θ Phase 1 — node-shape `CREATE` write op.
    /// Multi-item `CREATE (a), (b)` lowers to a left-deep chain of
    /// `CreateNode` operators (one per item).
    CreateNode(LogicalCreateNode),
    /// ADR-148 W26-θ Phase 2 — path-shape `CREATE` write op.
    /// `CREATE (a)-[r:R]->(b)` lowers to a sequence of operators in
    /// the left-deep chain: CreateNode(source) → CreateNode(target)
    /// → CreateRel. The CreateRel consumes the source + target
    /// `BindingId`s from the row produced by the upstream chain.
    CreateRel(LogicalCreateRel),
    /// ADR-149 W26-θ Phase 3 — `DELETE` / `DETACH DELETE` write op.
    /// Single-input operator over the prior MATCH's row stream;
    /// per-row deletion of each item's resolved Node / Rel id via
    /// the substrate's `delete_node` / `delete_rel`.
    Delete(LogicalDelete),
    /// ADR-150 W26-θ Phase 4 — `SET` write op. Single-input operator
    /// over the prior MATCH's row stream; per-row dispatch of each
    /// item's mutation (property assign / merge / replace / label-
    /// add) via the substrate's `set_node` / `set_rel`.
    Set(LogicalSet),
    /// ADR-150 W26-θ Phase 4 — `REMOVE` write op. Single-input
    /// operator over the prior MATCH's row stream; per-row dispatch
    /// of each item's removal (property / label) via the substrate's
    /// `remove_node` / `remove_rel`.
    Remove(LogicalRemove),
    /// ADR-151 W26-θ Phase 5 — `MERGE` write op (match-or-create).
    /// Wraps a match-branch sub-plan + a create-branch sub-plan +
    /// on_create / on_match action item vecs. The executor probes
    /// the match-branch; if non-empty, emits the matched rows + fires
    /// the on_match actions; if empty, fires the create-branch + emits
    /// its row + fires the on_create actions.
    Merge(LogicalMerge),
    /// **#830 / ADR-198 §OQ-7 / ADR-200** — `CREATE VECTOR INDEX <name>
    /// [IF NOT EXISTS] FOR (var:Label) ON var.prop [OPTIONS {…}]`
    /// accept-and-register write op. A LEAF (roots on its own unit
    /// trigger): on execute it parses the `OPTIONS` map for
    /// `vector.dimensions` + `vector.similarity_function` (resolving
    /// `$param` values against the per-query parameter bag), then
    /// registers a metadata entry in the per-tenant vector-index catalog
    /// via [`crate::executor::ExecutorSubstrate::register_vector_index`].
    /// Emits ZERO rows (a DDL has no result rows — Neo4j `CREATE VECTOR
    /// INDEX` returns an empty result). The served HNSW BUILD is
    /// auto-on-ingest (#765 PART-1); this op does NOT trigger a build.
    CreateVectorIndex(LogicalCreateVectorIndex),
    /// #1366 (task #248, Phase 1) — `CREATE INDEX <name> [IF NOT EXISTS]
    /// FOR (var:Label) ON (var.prop)`, the user-visible secondary
    /// node-property index DDL. The executor
    /// ([`crate::executor::ops::CreatePropertyIndexOp`]) registers the
    /// index in the durable property-index catalog as `Building`,
    /// backfills the MVCC-visible nodes once, and flips `Online`
    /// co-committed with the final backfill watermark (via
    /// [`crate::executor::ExecutorSubstrate::create_property_index`]).
    /// Emits ZERO rows. Distinct from `CreateVectorIndex` (no OPTIONS /
    /// no HNSW build). Phase 1 does NOT query-enable it (no planner
    /// `PropertyIndexScan`; that is Phase 2).
    CreatePropertyIndex(LogicalCreatePropertyIndex),
    /// #1366 (Phase 2, query-enable) — the indexed point-lookup leaf.
    /// The planner rewrites a `Scan(label) + Filter(prop = value)` into
    /// this leaf when the catalog reports an **Online** secondary
    /// property index on `(label, property)` (RC-6 planner-visible
    /// gate). At execute time
    /// ([`crate::executor::ops::PropertyIndexScanOp`]) it calls
    /// [`crate::executor::ExecutorSubstrate::property_index_lookup_with_context`]
    /// which does a B+tree candidate lookup then **MVCC-verifies each
    /// candidate** (hydrate through the txn snapshot + recheck label AND
    /// property equality) — the index NEVER determines visibility
    /// (ADR-023 candidate-then-verify). `residual` carries any OTHER
    /// predicates on the same binding (kept as a post-lookup filter). It
    /// is a LEAF (no `LogicalPlan` child): the anchor scan it replaces is
    /// gone, so `MATCH (n:User {email:"x"})` starts from `O(matches)`
    /// verified rows instead of an `O(node_high_water)` scan. Closes
    /// #1366's read-path OOM + the ~5183ms/~820× Neo4j point-lookup A/B
    /// lead (design §"Planner and executor wiring").
    PropertyIndexScan(LogicalPropertyIndexScan),
    /// ADR-192 (#623) — `CALL { <subquery> }` correlated brace-subquery
    /// (Cypher 25, beyond openCypher v9). Wraps a driving `input`
    /// sub-plan + the lowered subquery `body` sub-plan. The executor's
    /// `CallOp` (re-)executes `body` once per `input` row, seeding the
    /// `imported` bindings into the body's [`LogicalCorrelationSeed`]
    /// leaf, and emits `driving_row ++ body_output` (UNION-ALL across
    /// driving rows). This is the one operator with a `body` sub-plan
    /// DISTINCT from `input` (a correlated apply / lateral join).
    Call(LogicalCall),
    /// ADR-192 (#623) — the per-driving-row correlation seed: a one-row
    /// table carrying the `imported` bindings (the leading-clause `prev`
    /// of a `CALL { … }` body). The executor injects the current driving
    /// row's imported values per body build. Appears ONLY inside a
    /// [`LogicalCall::body`] (never at the top level).
    CorrelationSeed(LogicalCorrelationSeed),
    /// Sentinel for the degenerate empty-clauses case.
    Empty(LogicalEmpty),
}

impl LogicalPlan {
    /// Return the span covering this plan node.
    pub fn span(&self) -> &Span {
        match self {
            LogicalPlan::Scan(s) => &s.span,
            LogicalPlan::CountStore(c) => &c.span,
            LogicalPlan::Expand(e) => &e.span,
            LogicalPlan::Filter(f) => &f.span,
            LogicalPlan::Project(p) => &p.span,
            LogicalPlan::Join(j) => &j.span,
            LogicalPlan::LeftOuterJoin(j) => &j.span,
            LogicalPlan::Limit(l) => &l.span,
            LogicalPlan::Skip(s) => &s.span,
            LogicalPlan::RankByHybrid(r) => &r.span,
            LogicalPlan::Fusion(f) => &f.span,
            LogicalPlan::CommunityLookup(c) => &c.span,
            LogicalPlan::VectorNear(v) => &v.span,
            LogicalPlan::TextMatch(t) => &t.span,
            LogicalPlan::Aggregate(a) => &a.span,
            LogicalPlan::Sort(s) => &s.span,
            LogicalPlan::Distinct(d) => &d.span,
            LogicalPlan::Union(u) => &u.span,
            LogicalPlan::Unwind(u) => &u.span,
            LogicalPlan::ProcedureCall(p) => &p.span,
            LogicalPlan::NamedPath(n) => &n.span,
            LogicalPlan::DynamicLimit(l) => &l.span,
            LogicalPlan::CreateNode(c) => &c.span,
            LogicalPlan::CreateRel(c) => &c.span,
            LogicalPlan::Delete(d) => &d.span,
            LogicalPlan::Set(s) => &s.span,
            LogicalPlan::Remove(r) => &r.span,
            LogicalPlan::Merge(m) => &m.span,
            LogicalPlan::CreateVectorIndex(c) => &c.span,
            LogicalPlan::CreatePropertyIndex(c) => &c.span,
            LogicalPlan::PropertyIndexScan(p) => &p.span,
            LogicalPlan::Call(c) => &c.span,
            LogicalPlan::CorrelationSeed(s) => &s.span,
            LogicalPlan::Empty(e) => &e.span,
        }
    }

    /// **D-2 (ADR-147 §D-8 / W26-θ Phase 5) — statement-mutates predicate.**
    ///
    /// Returns `true` if this plan tree contains ANY write operator
    /// (`CREATE` node/rel, `DELETE`, `SET`, `REMOVE`, `MERGE`, `CREATE
    /// VECTOR INDEX`). Read-only plans (pure MATCH / RETURN / CALL-of-
    /// read-proc / SHOW) return `false`.
    ///
    /// # Why recursive
    ///
    /// The write operators wrap their driving input in the left-deep
    /// chain (`Delete(input: Scan)`, `Set(input: Filter(Scan))`,
    /// `CreateRel(input: CreateNode(...))`), and `MERGE` / `CALL` carry
    /// match/create/body sub-plans. A statement mutates iff ANY node in
    /// its tree is a write operator — so the walk visits every child.
    ///
    /// # Consumer (D-2 statement-scoped autocommit txn)
    ///
    /// `crate::materialize` calls this ONCE per statement to decide
    /// whether to open a statement-scoped transaction (begin-once,
    /// commit-once) around the pipeline drive. A read-only statement
    /// pays no transaction cost; a write statement's every substrate op
    /// stages into ONE txn committed once at statement end — closing the
    /// pre-D-2 hole where a multi-op `CREATE` spine committed per-op
    /// (3 durable commits for a 2-node-1-rel spine, non-atomic on a
    /// mid-statement crash). See the ingest memo
    /// `docs/perf/gap-neo4j-ingest-batch.md` §3 D-2.
    #[must_use]
    pub fn writes(&self) -> bool {
        match self {
            // Write operators — the statement mutates the graph.
            LogicalPlan::CreateNode(_)
            | LogicalPlan::CreateRel(_)
            | LogicalPlan::Delete(_)
            | LogicalPlan::Set(_)
            | LogicalPlan::Remove(_)
            | LogicalPlan::Merge(_)
            | LogicalPlan::CreateVectorIndex(_)
            | LogicalPlan::CreatePropertyIndex(_) => true,

            // Pure read leaves — no sub-plan can hide a write.
            // `Expand`, `RankByHybrid`, `VectorNear`, `TextMatch` are
            // retrieval leaves (endpoints / operands are pre-bound row
            // bindings, not nested plans).
            LogicalPlan::Scan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::CommunityLookup(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CorrelationSeed(_)
            | LogicalPlan::Empty(_) => false,

            // Read wrappers — recurse into the driving input (a write
            // may be nested below, though at v1.0-α the write op is
            // always at or near the tree root; the recursion is the
            // future-proof, clause-order-agnostic answer).
            LogicalPlan::Filter(p) => p.input.writes(),
            LogicalPlan::Project(p) => p.input.writes(),
            LogicalPlan::Limit(l) => l.input.writes(),
            LogicalPlan::Skip(s) => s.input.writes(),
            LogicalPlan::Aggregate(a) => a.input.writes(),
            LogicalPlan::Sort(s) => s.input.writes(),
            LogicalPlan::Distinct(d) => d.input.writes(),
            LogicalPlan::Unwind(u) => u.input.writes(),
            LogicalPlan::ProcedureCall(p) => p.input.writes(),
            LogicalPlan::NamedPath(n) => n.input.writes(),
            LogicalPlan::DynamicLimit(l) => l.input.writes(),
            LogicalPlan::Fusion(f) => f.inputs.iter().any(|i| i.writes()),
            LogicalPlan::Join(j) => j.left.writes() || j.right.writes(),
            LogicalPlan::LeftOuterJoin(j) => j.left.writes() || j.right.writes(),
            LogicalPlan::Union(u) => u.arms.iter().any(LogicalPlan::writes),
            LogicalPlan::Call(c) => c.input.writes() || c.body.writes(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalScan {
    /// Resolved label, if the source pattern carried one. `None` for
    /// label-free patterns `MATCH (n) ...`. Multi-label patterns
    /// (openCypher 9 `:A:B`) are not yet supported at v1.0; the M4-22
    /// reserved-variant rejection rejects them upstream of M4-31.
    pub label: Option<LabelId>,
    /// The binding produced by this scan (the node-pattern variable).
    pub var: BindingId,
    /// MVCC visibility key per ADR-041 §D-4. The executor passes
    /// this through to the storage substrate at scan time so the
    /// scan's emitted rows are filtered to the active read
    /// snapshot. v1.0 default is `current_lsn()` from the
    /// executor's transaction context (read-latest); v1.1 lifts
    /// to a `BEGIN AT SNAPSHOT lsn=N` parser surface that lets
    /// the caller pin a historical snapshot.
    pub read_lsn: Lsn,
    /// Span of the source node pattern.
    pub span: Span,
}

/// #1366 (Phase 2) — indexed point-lookup leaf. See the
/// [`LogicalPlan::PropertyIndexScan`] variant doc for the full contract.
///
/// The planner produces this ONLY when the catalog reports an
/// **Online** secondary index on `(label, property)` (RC-6
/// planner-visible gate — a `Building` index is never routed here).
/// The executor treats [`Self::value`] as a candidate KEY (not a
/// visibility authority): it hydrates each B+tree candidate through the
/// txn snapshot and re-checks label + property equality; stale / dup /
/// invisible candidates are DROPPED (candidate-then-verify, ADR-023).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalPropertyIndexScan {
    /// The label the index is declared on. ALWAYS `Some` at the logical
    /// level — an unlabelled `MATCH (n {email:"x"})` is NOT routed here
    /// (the label-agnostic union scan is out of RC scope; the planner
    /// keeps the full-scan path for it — design §Planner selection).
    pub label: LabelId,
    /// The indexed property name (display + executor recheck key).
    pub property: String,
    /// The exact-equality lookup value expression. A literal or a
    /// parameter reference; the executor resolves it against the
    /// per-query parameter bag at first-batch time to a concrete
    /// [`crate::executor::value::Value`] key.
    pub value: BoundExpression,
    /// The binding produced by this lookup (the node-pattern variable).
    pub var: BindingId,
    /// MVCC visibility key (mirrors [`LogicalScan::read_lsn`]).
    pub read_lsn: Lsn,
    /// OTHER predicates on the same binding kept as a post-lookup
    /// filter (e.g. `MATCH (n:User {email:"x"}) WHERE n.age > 30` keeps
    /// `n.age > 30` here). `None` when the index predicate is the whole
    /// filter. Applied AFTER the MVCC-verify so a residual match is over
    /// a live, snapshot-visible node.
    pub residual: Option<BoundExpression>,
    /// Span of the source node pattern / WHERE clause.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalExpand {
    /// The pre-bound endpoint (the `from` end of the traversal in
    /// source order).
    pub from: BindingId,
    /// The post-bound endpoint (the `to` end of the traversal).
    pub to: BindingId,
    /// Direction of the traversal.
    pub direction: Direction,
    /// Resolved relationship-type filter, if the source pattern
    /// carried one. `None` for type-free patterns. Multi-type
    /// patterns (openCypher 9 `:A|:B`) are not yet supported at v1.0;
    /// the M4-22 reserved-variant rejection rejects them upstream.
    pub rel_type: Option<TypeId>,
    /// Length range for variable-length patterns (openCypher `*N..M`).
    /// `None` for fixed-length single-hop traversals.
    pub length_range: Option<LengthRange>,
    /// Optional binding for the relationship itself
    /// (`(a)-[r:KNOWS]->(b)` — `r` is bound).
    pub rel_var: Option<BindingId>,
    /// Span of the source relationship pattern.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalFilter {
    /// Input plan whose output rows are filtered.
    pub input: Box<LogicalPlan>,
    /// The predicate expression. M4-31 trusts the M4-22 type-check
    /// pass to have validated this is a Boolean (or Null per Cypher
    /// 3VL D-20).
    pub predicate: BoundExpression,
    /// Span of the source WHERE / WITH WHERE clause.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalProject {
    /// Input plan whose output rows are projected.
    pub input: Box<LogicalPlan>,
    /// The projection items in declared order. Mirrors the
    /// `BoundProjectionItem` shape (a single item may be `Wildcard`
    /// or a bound expression).
    pub items: Vec<BoundProjectionItem>,
    /// Span of the source RETURN / WITH clause.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalJoin {
    /// Left input plan.
    pub left: Box<LogicalPlan>,
    /// Right input plan.
    pub right: Box<LogicalPlan>,
    /// Join condition.
    pub on: JoinCondition,
    /// Physical algorithm picked by the M4-51 cost walker +
    /// [`crate::planner::pick_join_algorithms`] (W25-M4-61b /
    /// ADR-097). Default at lowering time is [`JoinAlgorithm::Auto`];
    /// the picker pass rewrites it to a concrete variant before the
    /// executor consumes the plan. Tests + EXPLAIN consumers that
    /// want to pin a specific algorithm construct `LogicalJoin`
    /// directly with the desired variant.
    pub algorithm: JoinAlgorithm,
    /// Span of the source MATCH clause that introduced the join.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalLimit {
    /// Input plan whose output rows are limited.
    pub input: Box<LogicalPlan>,
    /// The literal `LIMIT N` count. M4-31 admits literal-integer LIMIT
    /// expressions only; parameter-driven LIMIT (`LIMIT $n`) defers
    /// to M4-33 alongside aggregation + sort.
    pub count: u64,
    /// Span of the source LIMIT clause.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSkip {
    /// Input plan whose first `count` rows are dropped.
    pub input: Box<LogicalPlan>,
    /// The literal `SKIP N` count. Same rationale as
    /// [`LogicalLimit::count`] for the literal-only restriction.
    pub count: u64,
    /// Span of the source SKIP clause.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalEmpty {
    /// Span of the source query (the whole input).
    pub span: Span,
}

/// ADR-192 (#623) — `CALL { <subquery> }` correlated brace-subquery.
///
/// `input` is the driving (outer) sub-plan; `body` is the lowered
/// subquery sub-plan whose leaf is a [`LogicalCorrelationSeed`] (the
/// leading-clause `prev` carrying the imported bindings). The executor
/// runs `body` once per `input` row, seeding the driving row's
/// `imported` values into the body's correlation seed, and emits
/// `input_row ++ body_output` (the body output relabeled positionally
/// to `returned`). See [`crate::executor::ops::CallOp`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCall {
    /// The driving (outer) input sub-plan. A LEADING `CALL { … }` (no
    /// preceding clause) roots this on the leading-clause [`LogicalEmpty`]
    /// unit row (ADR-192 D-5a) so the body runs exactly once.
    pub input: Box<LogicalPlan>,
    /// The lowered subquery body sub-plan (leaf = [`LogicalCorrelationSeed`]).
    pub body: Box<LogicalPlan>,
    /// Outer bindings imported into the subquery (ADR-192 D-3).
    pub imported: Vec<BindingId>,
    /// The body's terminal-RETURN output columns (binding-ids declared
    /// in the OUTER scope, ADR-192 D-4). The executor relabels the body's
    /// output columns to these positionally.
    pub returned: Vec<BindingId>,
    /// Span of the source `CALL` clause.
    pub span: Span,
}

/// ADR-192 (#623) — the per-driving-row correlation seed: a single-row
/// table whose columns are the imported bindings. Lowered as the
/// leading-clause `prev` of a [`LogicalCall::body`]; the executor's
/// [`crate::executor::ops::CorrelationSeedOp`] emits one row carrying the
/// current driving row's imported values (injected per body build). For
/// an empty `imported` (an uncorrelated `CALL { … }`) it degenerates to a
/// zero-column unit row (equivalent to the leading-clause [`LogicalEmpty`]
/// unit row).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalCorrelationSeed {
    /// The imported bindings this seed provides (its output schema, in
    /// column order). Empty ⇒ a zero-column unit row.
    pub imported: Vec<BindingId>,
    /// Span of the source `CALL` clause.
    pub span: Span,
}

// =====================================================================
// M4-32 — Hybrid retrieval + OPTIONAL MATCH operators (per ADR-038 §2 D-26)
// =====================================================================

/// `RANK BY HYBRID(VECTOR(...), TEXT(...))` orchestration node.
///
/// Carries the operand list (each a [`HybridOperand`]) plus an optional
/// fusion spec (when the source query had a directly-attached `WITH
/// FUSION = ...` clause). When the fusion clause is parsed as a
/// separate `BoundClause::WithFusion` (the v1.0 grammar shape per
/// ADR-038 §2 D-3), this node ships `fusion = None` and the fusion
/// step is rendered as a separate [`LogicalFusion`] sibling per the
/// pipeline order. M4-05 cost-planner is free to merge / split.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalRankByHybrid {
    /// Operand list (VECTOR / TEXT / EXPAND, in source order).
    pub operands: Vec<HybridOperand>,
    /// Optional result binding carrying the fused score.
    pub score_binding: Option<BindingId>,
    /// Fusion specification attached by a following `WITH FUSION`
    /// clause. Keeping it on the retrieval node ensures the requested
    /// smoothing constant reaches the executor even when a candidate
    /// `MATCH` plan wraps the retrieval in a join.
    pub fusion: Option<FusionSpec>,
    /// Span of the source `RANK BY` keyword + body.
    pub span: Span,
}

/// One operand of a `RANK BY HYBRID(...)` clause.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridOperand {
    /// Operand kind (vector retrieval vs. BM25 text search vs.
    /// graph-traversal expand). Per ADR-038 §2 D-3 v1.0 admits the
    /// VECTOR + TEXT operands and reserves EXPAND for v1.1.
    pub kind: HybridOperandKind,
    /// The bound variable the field is rooted at (the `field`'s
    /// [`crate::semantic::bound_ast::BoundFieldRef::base`] binding ID).
    pub var: BindingId,
    /// The property name the operand reads (e.g., `"embedding"` for
    /// VECTOR, `"content"` for TEXT). For `Expand` operands the
    /// property is empty (the operand is pattern-shaped, not
    /// field-shaped).
    pub property: String,
    /// The query expression — `$q` parameter for VECTOR / TEXT, or
    /// the seed expression for `Expand`.
    pub query: BoundExpression,
    /// The required `K = N` parameter (M4-23 cross-substrate
    /// validation rejects bare operands without K).
    pub k: u64,
    /// MVCC visibility key per ADR-041 §D-4. The executor passes
    /// this through to the substrate (vector / BM25) at retrieval
    /// time so each operand's hits are filtered to the active
    /// read snapshot. Closes codex retro F-1: without this
    /// carrier, a hybrid `BEGIN AT SNAPSHOT lsn=N` query would
    /// silently mix snapshot-isolated text hits with read-latest
    /// vector hits.
    pub read_lsn: Lsn,
    /// Span of the operand.
    pub span: Span,
}

/// Kind of a [`HybridOperand`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridOperandKind {
    /// `VECTOR(field, query, K = N)` — vector ANN retrieval.
    Vector,
    /// `TEXT(field, query, K = N)` — BM25 text search.
    Text,
}

/// `WITH FUSION = <kind>` rank-fusion node.
///
/// At v1.0 only `Rrf` is lit; the M4-22 type-checker rejects
/// CombSUM-Norm / CombMnz-Norm / Ltr (per ADR-038 §2 D-9 + D-10), so
/// they never reach lowering. The variant is kept narrow to match.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalFusion {
    /// Fusion kind + parameters.
    pub spec: FusionSpec,
    /// Inputs to fuse. The v1.0 shape is `inputs = [hybrid_root]`
    /// where `hybrid_root` is a [`LogicalPlan::RankByHybrid`] node
    /// or a wrapper around it. The vec is kept variadic so M4-05 +
    /// M4-33 may rewrite (e.g., split a hybrid into per-substrate
    /// retrievers and re-fuse) without changing the type shape.
    pub inputs: Vec<Box<LogicalPlan>>,
    /// Span of the source `WITH FUSION = ...` clause.
    pub span: Span,
}

/// Fusion kind + parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FusionSpec {
    /// Algorithm (only `Rrf` lit at v1.0).
    pub kind: FusionKind,
    /// The required `k` parameter (smoothing constant for RRF;
    /// Cormack SIGIR 2009 default is 60).
    pub k: u64,
    /// Span of the fusion algorithm token.
    pub span: Span,
}

/// Fusion algorithm (D-9). v1.0 admits `Rrf` only; CombSUM / CombMnz /
/// Ltr stay rejected at the M4-22 type-check layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionKind {
    /// Reciprocal-Rank Fusion. Cormack SIGIR 2009.
    Rrf,
}

/// Community-membership lookup (closes PR #154 reviewer Finding 5 +
/// ADR-038 amendment-01 §A-2 commitment).
///
/// Both the predicate surface `n IN COMMUNITY($cid)` and the canonical
/// surface `community(n) = $cid` (or `$cid = community(n)`) lower to
/// IDENTICAL trees rooted at this node — same `node_var`, same
/// `community_id` payload, modulo span coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCommunityLookup {
    /// Input plan whose output rows are filtered by community
    /// membership (typically a [`LogicalScan`] producing the
    /// node-pattern variable).
    pub input: Box<LogicalPlan>,
    /// The bound variable being tested for community membership
    /// (i.e., the `n` in `n IN COMMUNITY($cid)` or `community(n)`).
    pub node_var: BindingId,
    /// The community identifier expression (e.g., `Parameter($cid)`).
    pub community_id: BoundExpression,
    /// MVCC visibility key per ADR-041 §D-4. The executor passes
    /// this to `CommunityIndexHandle::membership` at evaluation
    /// time so the membership lookup resolves against the visible
    /// install (per ADR-041 §D-3b history-binary-search).
    pub read_lsn: Lsn,
    /// Span of the source community-membership construct.
    pub span: Span,
}

/// Vector ANN retrieval node (lowered from `vector_distance(...)` /
/// `<expr> NEAR <expr>` per ADR-038 §2 D-5).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalVectorNear {
    /// The bound variable whose property is the index field.
    pub var: BindingId,
    /// The property name (e.g., `"embedding"`).
    pub property: String,
    /// The query vector expression (typically a `$q` parameter).
    pub query_vector: BoundExpression,
    /// Top-K cap. Defaults to `0` (= no cap) when the source surface
    /// did not carry an explicit K (e.g., a bare `n.embedding NEAR $q`
    /// expression). The M4-32 hybrid lowering for `RANK BY HYBRID`
    /// always populates K from the operand's required parameter.
    pub k: u64,
    /// MVCC visibility key per ADR-041 §D-4. Threaded into the
    /// vector substrate's `FilteredVectorIndex::filtered_search`
    /// at execution time.
    pub read_lsn: Lsn,
    /// Span of the source vector predicate.
    pub span: Span,
}

/// BM25 text-match node (lowered from `text_match(...)` /
/// `<expr> MATCH <expr>` per ADR-038 §2 D-6).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalTextMatch {
    /// The bound variable whose property is the index field.
    pub var: BindingId,
    /// The property name (e.g., `"content"`).
    pub property: String,
    /// The query-text expression (typically a `$q` parameter or a
    /// string literal).
    pub query_text: BoundExpression,
    /// Top-K cap. `None` when the source surface was a bare
    /// expression (`<expr> MATCH <expr>`); `Some(k)` when surfaced
    /// from a `RANK BY HYBRID` `TEXT(field, query, K = N)` operand.
    pub k: Option<u64>,
    /// MVCC visibility key per ADR-041 §D-4. Threaded into the
    /// BM25 substrate's `Bm25IndexHandle::search` at execution
    /// time (the BM25 substrate already accepted `read_lsn`
    /// pre-ADR-041 per ADR-039 §D-3; this carrier closes the
    /// asymmetry by giving the vector + community + lowering
    /// path the same plumbing).
    pub read_lsn: Lsn,
    /// Span of the source text predicate.
    pub span: Span,
}

/// Left-outer join (per ADR-006 amendment-01 §A-2 + Cypher 9 §6.5).
///
/// Lowered from `OPTIONAL MATCH` clauses. `right`'s rows that do not
/// satisfy the join condition produce NULLs for fresh `right`-side
/// bindings; re-references of pre-existing bindings stay
/// non-nullable per the M4-22b binding-time `may_be_null`
/// propagation rule. M4-32 reads the binding-time flag from
/// [`crate::semantic::bound_ast::BoundVariable::may_be_null`] and
/// does NOT recompute nullability at lowering time.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalLeftOuterJoin {
    /// Left input (preserved entirely).
    pub left: Box<LogicalPlan>,
    /// Right input (rows nulled out when the join condition fails).
    pub right: Box<LogicalPlan>,
    /// Join condition. v1.0 ships `SharedBindings` only — same shape
    /// as [`LogicalJoin::on`].
    pub on: JoinCondition,
    /// Span of the source `OPTIONAL MATCH` clause.
    pub span: Span,
}

// =====================================================================
// M4-33 — Aggregation + ORDER BY + DISTINCT + UNWIND + path operators
// =====================================================================

/// `GROUP BY` + aggregation per openCypher 9 §6.4 (M4-33).
///
/// Emitted when a RETURN / WITH clause contains at least one
/// aggregation function call. The `group_by` slot carries the
/// non-aggregation projection items (the implicit GROUP BY per
/// openCypher 9 §6.4: every non-aggregate item in RETURN / WITH is a
/// grouping key); `aggregations` carries the resolved aggregation
/// function calls.
///
/// When all RETURN / WITH items are aggregations, `group_by` is
/// empty — the result is a single row per the openCypher 9 §6.4
/// "single-row aggregate" semantics.
///
/// # NULL handling
///
/// Aggregation functions handle NULL per openCypher 9 §6.4 (and the
/// 3VL discipline locked at ADR-038 §2 D-20):
/// - `count(expr)` excludes rows where `expr` is NULL;
/// - `sum`, `avg`, `min`, `max` ignore NULL operands;
/// - `collect` drops NULL elements from the resulting list.
///
/// When the input is a [`LogicalLeftOuterJoin`] (lowered from
/// `OPTIONAL MATCH`), null-flagged bindings flow through unchanged —
/// the per-aggregation NULL semantics apply at execution time.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalAggregate {
    /// Input plan whose rows are grouped.
    pub input: Box<LogicalPlan>,
    /// Implicit GROUP BY keys: the non-aggregation projection items
    /// (preserved as [`BoundProjectionItem`] so aliases survive). The
    /// list is empty for the single-row aggregate case (`MATCH (n)
    /// RETURN count(n)`).
    pub group_by: Vec<BoundProjectionItem>,
    /// Resolved aggregation function calls (declared order matches
    /// the source RETURN / WITH item order, NOT the
    /// `group_by`-then-aggregations split).
    pub aggregations: Vec<AggregationSpec>,
    /// Span of the source RETURN / WITH clause that introduced the
    /// aggregation.
    pub span: Span,
}

/// Counts-store source selected only for exact unfiltered count queries.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCountStore {
    /// Which tenant-wide counter to read.
    pub source: CountStoreSource,
    /// Binding under which the single count value is emitted.
    pub output_id: BindingId,
    /// Source aggregate span.
    pub span: Span,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CountStoreSource {
    /// Tenant-wide node count (`MATCH (n) RETURN count(n)`).
    Nodes,
    /// Tenant-wide relationship count (`MATCH ()-->() RETURN count(*)`).
    Relationships,
    /// Per-label node count (`MATCH (n:Label) RETURN count(n)`). F1
    /// (#1356 §F1) lowers this to the EXISTING
    /// `CatalogStats::label_counts` counter (an O(1) read) instead of a
    /// full label scan. `LabelId` is `Copy`, so the enum keeps its
    /// `Copy`/`Eq`/`Hash` derives.
    NodesWithLabel(LabelId),
    /// Per-type relationship count (`MATCH ()-[:TYPE]->() RETURN
    /// count(*)`). F1 (#1356 §F1) lowers this to the EXISTING
    /// `CatalogStats::rel_type_counts` counter instead of a full scan +
    /// per-row expand.
    RelsWithType(TypeId),
}

/// Resolved aggregation function call.
#[derive(Debug, Clone, PartialEq)]
pub struct AggregationSpec {
    /// Aggregation function (resolved from the source function name).
    pub function: AggregationKind,
    /// Argument expression (the inner expression of the function
    /// call — e.g., `n` for `count(n)`, `n.price` for
    /// `sum(n.price)`).
    pub arg: BoundExpression,
    /// Output binding-id (#746). The [`BindingId`] under which the
    /// `AggregateOp` emits this aggregation's result column, and the id
    /// the layered-over `Project` references (the lowering rewrites the
    /// projection item from `count(n)` to `VariableRef(output_id)` so
    /// the `Project` passes the precomputed aggregate value through
    /// instead of re-evaluating the aggregate function). Sourced from
    /// the projection item's
    /// [`crate::semantic::bound_ast::BoundProjectionItem::output_id`]
    /// so a downstream `WITH count(n) AS c … RETURN c` resolves `c` to
    /// the SAME id.
    pub output_id: BindingId,
    /// Optional output alias (`count(n) AS c` → `Some("c")`).
    pub alias: Option<String>,
    /// `count(DISTINCT expr)` / `collect(DISTINCT expr)` — deduplicate
    /// the aggregated (non-NULL) values before the count/collect/fold
    /// (#773 G5; openCypher v9 §3). Threaded from the source
    /// [`crate::semantic::bound_ast::BoundExpression::FunctionCall::distinct`]
    /// by [`super::lowering`]'s `try_lift_aggregation`; consumed by the
    /// executor's per-group dedup set.
    pub distinct: bool,
    /// `count(*)` — count input ROWS rather than non-NULL `arg` values
    /// (#773 G4). When `true`, `arg` is a placeholder (the star form has
    /// no expression); the executor folds a non-NULL sentinel per row so
    /// the count includes all-NULL rows. Mutually exclusive with
    /// `distinct` (`count(DISTINCT *)` is not an openCypher form).
    pub star: bool,
    /// Span of the source function-call site.
    pub span: Span,
}

/// Aggregation function kind. Matches the v1.0 [`crate::semantic::functions`]
/// aggregation-registry entries (`count` / `sum` / `avg` / `min` /
/// `max` / `collect`) per openCypher 9 §3.
///
/// `count(*)` (#773 G4) is represented as `Count` + the
/// [`AggregationSpec::star`] flag — NOT a dedicated `AggregationKind`
/// variant. The star form counts ROWS (the executor folds a non-NULL
/// sentinel per row); `Count` without `star` counts non-NULL `arg`
/// values. `DISTINCT` (#773 G5) is likewise orthogonal to the kind — it
/// rides [`AggregationSpec::distinct`] and applies a per-group dedup
/// before any kind's fold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationKind {
    /// `count(expr)` — count rows with non-NULL `expr`.
    Count,
    /// `sum(expr)` — sum non-NULL numeric `expr`.
    Sum,
    /// `avg(expr)` — arithmetic mean of non-NULL numeric `expr`.
    Avg,
    /// `min(expr)` — minimum of non-NULL `expr`.
    Min,
    /// `max(expr)` — maximum of non-NULL `expr`.
    Max,
    /// `collect(expr)` — accumulate non-NULL `expr` values into a
    /// list.
    Collect,
}

impl AggregationKind {
    /// Resolve a function-call name to an [`AggregationKind`] (case-
    /// insensitive). Returns `None` for non-aggregation functions —
    /// the same set covered by the v1.0 [`crate::semantic::functions::BUILTINS`]
    /// aggregation entries (see lines 172–177).
    pub fn from_function_name(name: &str) -> Option<Self> {
        match () {
            _ if name.eq_ignore_ascii_case("count") => Some(Self::Count),
            _ if name.eq_ignore_ascii_case("sum") => Some(Self::Sum),
            _ if name.eq_ignore_ascii_case("avg") => Some(Self::Avg),
            _ if name.eq_ignore_ascii_case("min") => Some(Self::Min),
            _ if name.eq_ignore_ascii_case("max") => Some(Self::Max),
            _ if name.eq_ignore_ascii_case("collect") => Some(Self::Collect),
            _ => None,
        }
    }
}

/// `ORDER BY` per Cypher 9 §6.6 (M4-33).
///
/// Both the tail-clause `... ORDER BY x` and the return-clause
/// `RETURN ... ORDER BY x` source surfaces lower to a [`LogicalSort`]
/// — modulo the position of the node in the resulting tree. This
/// closes the two M4-31 ORDER BY deferral sites with a single
/// [`LogicalPlan`] variant per ADR-038 §2 D-28.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSort {
    /// Input plan whose rows are sorted.
    pub input: Box<LogicalPlan>,
    /// Order-by keys (declared order = sort-key precedence: the
    /// first key is the primary, the second key is the tie-breaker,
    /// etc. per Cypher 9 §6.6).
    pub order_by: Vec<OrderByItem>,
    /// Span of the source ORDER BY construct.
    pub span: Span,
}

/// Single ORDER BY key.
#[derive(Debug, Clone, PartialEq)]
pub struct OrderByItem {
    /// Sort-key expression (e.g., `n.age`).
    pub expr: BoundExpression,
    /// Sort direction.
    pub direction: SortDirection,
    /// Span of the source ORDER BY item.
    pub span: Span,
}

/// Sort direction. Cypher 9 §6.6 specifies `Asc` as the default;
/// [`SortDirection::Asc`] mirrors that default. The
/// `OrderDirection::Default` AST variant is collapsed to `Asc` at
/// lowering time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    Asc,
    Desc,
}

/// `RETURN DISTINCT` (M4-33).
///
/// The `on` slot carries the bindings whose tuple identifies a
/// duplicate row. M4-33 populates this from the projection items'
/// shared bindings; M4-05 cost-planner is free to specialize (e.g.,
/// hash-distinct vs. sort-distinct).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalDistinct {
    /// Input plan whose rows are deduplicated.
    pub input: Box<LogicalPlan>,
    /// Bindings whose joint tuple identifies a row for
    /// deduplication. Empty list = "every row is a duplicate of
    /// every other row" (semantically the single-row case).
    pub on: Vec<BindingId>,
    /// Span of the source `DISTINCT` keyword.
    pub span: Span,
}

/// `UNION ALL` set-op concatenation per ADR-185 (#649-A1, W28 —
/// openCypher v9 §8 "Set operations").
///
/// The executor's [`crate::executor::ops::UnionOp`] concatenates the
/// arms' row streams in arm order — pure streaming, O(1) extra memory
/// (NOT a materialization point; that is [`LogicalDistinct`], which a
/// bare `UNION` composes OVER this node in #649-A2). The union's output
/// schema is arm 0's output schema; `column_orders` realigns each arm's
/// columns to arm 0's column order so arms exposing the same column
/// NAME set in a different ORDER still concatenate correctly (the §8
/// order-independent result-column rule).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalUnion {
    /// The union arms (≥2) as fully-lowered sub-plans, left-to-right.
    pub arms: Vec<LogicalPlan>,
    /// Per-arm column permutation: `column_orders[i][j]` is the source
    /// position in arm `i`'s output row supplying the union's canonical
    /// output column `j` (canonical = arm 0's order). `column_orders[0]`
    /// is the identity. Carried verbatim from
    /// [`crate::semantic::bound_ast::BoundUnionQuery::column_orders`].
    pub column_orders: Vec<Vec<usize>>,
    /// Span covering the whole union.
    pub span: Span,
}

/// `UNWIND <list_expr> AS <var>` per Cypher 9 §6.7 (M4-33).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalUnwind {
    /// Input plan whose rows are expanded by the UNWIND. May be
    /// [`LogicalEmpty`] for a top-level UNWIND with no preceding
    /// MATCH (e.g., `UNWIND [1,2,3] AS x RETURN x`).
    pub input: Box<LogicalPlan>,
    /// List expression to unwind.
    pub list_expr: BoundExpression,
    /// Binding for the per-element variable.
    pub var: BindingId,
    /// Span of the source UNWIND clause.
    pub span: Span,
}

/// **ADR-197 (#802)** — the source of a [`LogicalProcedureCall`]'s
/// rows: a schema-introspection procedure or a SHOW command.
#[derive(Debug, Clone, PartialEq)]
pub enum ProcedureSource {
    /// `CALL <proc>(…) [YIELD …]` — the resolved procedure.
    Procedure(crate::semantic::bound_ast::ProcedureKind),
    /// `SHOW CONSTRAINTS | INDEXES | DATABASES`.
    Show(crate::ast::ShowKind),
}

/// **ADR-197 (#802)** — `CALL <proc>(args) [YIELD …]` / `SHOW …`
/// generating operator. Produces rows from the live catalog /
/// intern-table; each row binds the projected columns to their
/// [`BindingId`]s so following operators (Filter / Project) consume
/// them. Roots on a leading [`LogicalEmpty`] unit row (same idiom as
/// [`LogicalUnwind`]) when it is the first clause.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalProcedureCall {
    /// Input plan (a leading [`LogicalEmpty`] unit row for a top-level
    /// CALL / SHOW; a prior pipeline otherwise — the procedure runs
    /// once per driving row, but at v1.0-α the only consumer is a
    /// leading CALL so the input is the unit row).
    pub input: Box<LogicalPlan>,
    /// Procedure or SHOW source.
    pub source: ProcedureSource,
    /// Argument expressions (procedure only; empty for SHOW). Accepted
    /// + carried but not interpreted at v1.0-α.
    pub args: Vec<BoundExpression>,
    /// Projected columns: `(source_column_name, binding_id)` in output
    /// order. The executor produces a row per catalog entry, placing
    /// each column's value at its binding slot.
    pub columns: Vec<(String, BindingId)>,
    /// Span of the source clause.
    pub span: Span,
}

/// Named path per Cypher 9 §6.5 (M4-33).
///
/// Lowered from `MATCH p = (a)-[..]->(b)` (Plain) and `MATCH p =
/// SHORTEST_PATH(...)` (ShortestPath). The path subtree (Scan +
/// Expand chain) lives in `input`; `path_var` is the binding for the
/// path variable; `algorithm` selects between full path enumeration
/// (Plain) and the SSSP-shaped traversal (ShortestPath). M4-05
/// cost-planner consumes the marker to pick a physical traversal
/// implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalNamedPath {
    /// The lowered path subtree (Scan + Expand chain).
    pub input: Box<LogicalPlan>,
    /// Binding for the path variable (the `p` in `p = ...`).
    pub path_var: BindingId,
    /// Algorithm marker.
    pub algorithm: PathAlgorithm,
    /// **ADR-193 D-4/D-5.** Ordered element-binding sequence for a
    /// `Plain` path, captured at lowering time so the executor can
    /// materialize a `Value::Path` from the bound MATCH row in pattern
    /// order. `Some` for `PathAlgorithm::Plain`; `None` for
    /// `PathAlgorithm::ShortestPath` (whose executor re-traverses the
    /// substrate and does not consume the shape). The bindings index
    /// into the `input` subtree's output schema.
    pub plain_shape: Option<PlainPathShape>,
    /// **ADR-194 D-3a.** Binding of the pattern's HEAD (source) node —
    /// the `a` in `(a)-[..]->(b)` — captured at lowering so the
    /// `ShortestPath` executor reads the source endpoint from a STABLE
    /// binding rather than the child schema's first slot. The schema-first
    /// heuristic is fragile: when the tail node carries a label, lowering
    /// joins the pattern subtree with a tail-label `Scan`, and the join
    /// can reorder the output schema so the first slot is the TAIL, not
    /// the head (e.g. `[b, a]`). `Some(head)` for a named head endpoint;
    /// `None` for an anonymous head `()-[..]->(..)` (degenerate for
    /// shortest-path), in which case the pipeline falls back to the
    /// legacy schema-first slot. The binding indexes into the `input`
    /// subtree's output schema. Consumed only by
    /// `PathAlgorithm::ShortestPath`.
    pub source: Option<BindingId>,
    /// **ADR-194 D-3a.** Binding of the pattern's TAIL-endpoint node —
    /// the `b` in `(a)-[..]->(b)` — captured at lowering so the
    /// `ShortestPath` executor can run bidirectional source→target BFS
    /// (one path per `(source, target)` pair) instead of single-source
    /// BFS (one row per reachable node). `Some(tail)` for a named tail
    /// endpoint (and for the degenerate single-node pattern `(a)`, where
    /// `target == source`, yielding a zero-length path); `None` for an
    /// anonymous tail endpoint `(a)-[..]->()`, which falls back to the
    /// efficient single-source enumeration (shortest path to every
    /// reachable node). The binding indexes into the `input` subtree's
    /// output schema. Consumed only by `PathAlgorithm::ShortestPath`;
    /// inert for `Plain` (whose executor materializes from `plain_shape`).
    pub target: Option<BindingId>,
    /// Span of the source named-path construct.
    pub span: Span,
}

/// **ADR-193 D-4/D-5.** The ordered element-binding sequence of a `Plain`
/// named path, mirroring the AST pattern `head + tail[(rel, node)]`.
///
/// `start` is the head node's binding; each segment carries the
/// relationship binding + the node it lands on (the AST tail node). For
/// a named path EVERY element is bound — anonymous relationships (e.g.
/// the `[:CALLS]` in `p = (a)-[:CALLS]->(b)`) are FORCE-BOUND a synthetic
/// binding at lowering so the path can materialize them (unlike a plain
/// MATCH, where an anonymous rel is left unbound). All bindings index
/// into the `LogicalNamedPath::input` output schema.
#[derive(Debug, Clone, PartialEq)]
pub struct PlainPathShape {
    /// Binding of the head (start) node.
    pub start: BindingId,
    /// Ordered segments in pattern (traversal) order.
    pub segments: Vec<PlainPathSegmentShape>,
}

/// One segment of a [`PlainPathShape`].
#[derive(Debug, Clone, PartialEq)]
pub struct PlainPathSegmentShape {
    /// Binding of the relationship column.
    pub rel: BindingId,
    /// Binding of the node this segment lands on (the AST tail node).
    pub end: BindingId,
    /// `true` when this segment is a var-length expand (`*N..M`): the
    /// `rel` column carries a `Value::List(Vec<Value::Relationship>)` in
    /// traversal order (ADR-186 RC-2), expanding to `rels.len()`
    /// path-segments. `false` for a single-hop segment whose `rel`
    /// column is a scalar `Value::Relationship`.
    pub var_length: bool,
}

/// Named-path traversal algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathAlgorithm {
    /// `p = (a)-[..]->(b)` — full path enumeration.
    Plain,
    /// `p = SHORTEST_PATH((a)-[..*]->(b))` (macro) or canonical
    /// `p = shortestPath((a)-[..*]->(b))` per ADR-038 §2 D-7, ADR-194 D-3,
    /// and Cypher 9 §6.5. ONE minimum-length source→target path.
    ShortestPath,
    /// `p = allShortestPaths((a)-[..*]->(b))` per ADR-194 D-2/D-4 and
    /// openCypher §allShortestPaths. ALL equal-minimum-length
    /// source→target paths (cardinality = #min-length paths; one
    /// `Value::Path` row each). Intrinsically src→dst — REQUIRES a bound
    /// target endpoint (the pipeline rejects an anonymous tail).
    AllShortestPaths,
}

/// Non-literal LIMIT / SKIP backed by a parameter or expression
/// (M4-33). Replaces the M4-31 `non-literal SKIP` /
/// `non-literal LIMIT` deferral emissions.
///
/// Both LIMIT and SKIP source surfaces lower to the same variant
/// shape — the `kind` slot disambiguates. The `count_expr` is a
/// parameter / expression that evaluates to a non-negative integer
/// at runtime. Negative-runtime-value handling is M4-05 / M4-06's
/// concern.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalDynamicLimit {
    /// Input plan whose rows are limited / skipped.
    pub input: Box<LogicalPlan>,
    /// LIMIT vs SKIP (which way the count applies).
    pub kind: DynamicLimitKind,
    /// Count expression (typically a `Parameter { name: "n", ... }`,
    /// but any well-typed integer expression is admissible per Cypher
    /// 9 §6.6).
    pub count_expr: BoundExpression,
    /// Span of the source LIMIT / SKIP construct.
    pub span: Span,
}

/// Dynamic-LIMIT direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicLimitKind {
    /// `LIMIT <expr>` — keep only the first N rows.
    Limit,
    /// `SKIP <expr>` — drop the first N rows.
    Skip,
}

// =====================================================================
// ADR-147 W26-θ Phase 1 — CREATE node logical plan operator
// =====================================================================

/// Node-shape `CREATE` write op (ADR-147 W26-θ Phase 1).
///
/// Multi-item `CREATE (a), (b)` lowers to a left-deep chain of
/// `LogicalCreateNode` operators (one per item), each consuming the
/// prior op's row output and emitting one new row per CREATE.
///
/// # Properties
///
/// At Phase 1 each property value's `BoundExpression` is restricted
/// to `BoundExpression::Literal { value, .. }` per ADR-147 §D-4 (the
/// M4-22 type-check pass enforces). The executor reads
/// `entry.value.literal()` at execute-time without recursing into
/// expression evaluation.
///
/// # Span
///
/// Carries the span of the source `CREATE` keyword + item — used by
/// future M4-91 EXPLAIN output to annotate the create-op node.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCreateNode {
    /// The optional variable binding for the new node (None for
    /// anonymous CREATEs like `CREATE (:User)`).
    pub var: Option<BindingId>,
    /// The optional LABEL NAME for the new node (None for label-less
    /// CREATEs like `CREATE (n {a:1})`). Resolved or interned at
    /// substrate-execute time per ADR-147 §D-7.
    pub label: Option<String>,
    /// Phase 1 literal-only property values per ADR-147 §D-4.
    /// Carried as `(key, literal-bearing BoundExpression)`; the
    /// executor extracts the inner literal at execute-time.
    pub properties: Vec<(String, BoundExpression)>,
    /// Optional upstream row stream this CREATE-node chains onto
    /// (issue #832 — silent multi-pattern data loss).
    ///
    /// For a multi-item leading `CREATE (a),(b),(c)` the lowering
    /// builds a left-deep chain: item N's `input` is the sub-plan for
    /// items `1..N-1`, so EVERY item executes. Previously
    /// `lower_create` overwrote the accumulator (`current = Some(op)`)
    /// each iteration, keeping ONLY the last item and silently
    /// dropping the rest — a `CREATE (:T{n:1}),(:T{n:2}),(:T{n:3})`
    /// persisted only `{n:3}`.
    ///
    /// `None` marks a chain leaf (the first item) and the endpoint
    /// sub-plans of a [`LogicalCreateRel`] / the create-branch of a
    /// [`LogicalMerge`], which are driven by their parent op. The
    /// executor [`crate::executor::ops::CreateNodeOp`] performs one
    /// create per upstream row and emits the row EXTENDED with the
    /// new binding (left-deep chain semantic).
    pub input: Option<Box<LogicalPlan>>,
    /// Span of the source CREATE-item construct.
    pub span: Span,
}

// =====================================================================
// #830 / ADR-198 §OQ-7 / ADR-200 — CREATE VECTOR INDEX logical op
// =====================================================================

/// `CREATE VECTOR INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
/// var.prop [OPTIONS {…}]` accept-and-register write op (#830 / ADR-200).
///
/// Lowered from [`crate::ast::IndexDdlStatement::CreateVector`]. A LEAF
/// op (no input child). On execute, the [`crate::executor::ops::CreateVectorIndexOp`]:
/// 1. Resolves the index `name` — a `$param` name (a common
///    Neo4j-compatible client form) is resolved against the per-query
///    parameter bag; a literal name is used verbatim.
/// 2. Extracts `vector.dimensions` + `vector.similarity_function` from
///    the raw `OPTIONS` map (resolving `$param` / `toInteger($param)`
///    values against the parameter bag — the real client passes
///    `toInteger($dimensions)` + `$similarity_fn`).
/// 3. Validates dimensions > 0 when present.
/// 4. Registers the metadata entry in the per-tenant vector-index
///    catalog via [`crate::executor::ExecutorSubstrate::register_vector_index`]
///    (honoring `IF NOT EXISTS` idempotency).
///
/// Emits ZERO rows (Neo4j `CREATE VECTOR INDEX` returns an empty
/// result). The served HNSW index BUILD is auto-on-ingest (#765
/// PART-1); this op is metadata-only and does NOT trigger a build.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCreateVectorIndex {
    /// The index name (`$param` or literal — resolved at execute-time).
    pub name: crate::ast::IndexNameRef,
    /// `IF NOT EXISTS` present (idempotent create).
    pub if_not_exists: bool,
    /// The node label in `FOR (var:Label)` (e.g. `CzChunk`).
    pub label: String,
    /// The indexed vector property in `ON var.prop` (e.g. `embedding`).
    pub property: String,
    /// The raw `OPTIONS { … }` map, captured verbatim (ADR-198 §OQ-7).
    /// The executor extracts `vector.dimensions` +
    /// `vector.similarity_function` at execute-time (the values may be
    /// `$param`s, so resolution is deferred to where the parameter bag
    /// is available). `None` when no `OPTIONS` clause.
    pub options: Option<crate::ast::Expression>,
    /// Span of the source `CREATE VECTOR INDEX` statement.
    pub span: Span,
}

/// #1366 (task #248, Phase 1) — lowered `CREATE INDEX … FOR (var:Label)
/// ON (var.prop)` for the user-visible property index. Mirror of
/// [`LogicalCreateVectorIndex`] minus the OPTIONS clause.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCreatePropertyIndex {
    /// The index name (`$param` or literal — resolved at execute-time).
    pub name: crate::ast::IndexNameRef,
    /// `IF NOT EXISTS` present (idempotent create).
    pub if_not_exists: bool,
    /// The node label in `FOR (var:Label)` (e.g. `User`).
    pub label: String,
    /// The indexed property in `ON (var.prop)` (e.g. `email`).
    pub property: String,
    /// Span of the source `CREATE INDEX` statement.
    pub span: Span,
}

// =====================================================================
// ADR-148 W26-θ Phase 2 — CREATE rel logical plan operator
// =====================================================================

/// Path-shape `CREATE rel` write op (ADR-148 W26-θ Phase 2).
///
/// Lowered from `CreateItem::Path` after the source + target
/// [`LogicalCreateNode`] operators. The executor reads the
/// `source` + `target` `BindingId`s from the upstream row to resolve
/// the new edge's endpoint NodeIds, then writes via
/// [`crate::executor::substrate::ExecutorSubstrate::create_rel`].
///
/// # Properties
///
/// Phase 2 inherits Phase 1's literal-only narrowing per ADR-147 §D-4;
/// the type-check pass enforces this on every property bag inside a
/// CREATE-path item (source / rel / target).
///
/// # Span
///
/// Carries the span of the source CREATE-rel construct — used by
/// future M4-91 EXPLAIN output to annotate the create-rel node.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalCreateRel {
    /// The optional variable binding for the new rel (None for
    /// anonymous CREATE rels like `(a)-[:R]->(b)`).
    pub var: Option<BindingId>,
    /// Mandatory rel-type NAME (Phase 2 per ADR-148 §D-1; grammar
    /// rejects label-less rel detail). Interned at substrate-execute
    /// time via `InternTable::intern_type` per ADR-148 §D-7.
    pub label: String,
    /// Phase 2 literal-only property values per ADR-147 §D-4
    /// (inherited from Phase 1).
    pub properties: Vec<(String, BoundExpression)>,
    /// Sub-plan producing the source NodeId. At Phase 2 always a
    /// [`LogicalPlan::CreateNode`] (inline-CREATE source); future
    /// Phase 5 lights MATCH→CREATE by allowing this to be any
    /// node-producing plan (e.g., [`LogicalPlan::Scan`]).
    pub source_plan: Box<LogicalPlan>,
    /// The binding inside `source_plan`'s schema that carries the
    /// source NodeId. The executor reads `source_plan`'s output row
    /// and projects the cell at this schema index to get the NodeId.
    pub source: BindingId,
    /// True when `source` came from a user-visible CREATE variable
    /// (e.g. `CREATE (a)-[:R]->()`), false when lowering synthesized it
    /// only to route an anonymous endpoint's NodeId.
    pub source_visible: bool,
    /// Whether the source endpoint is produced by `source_plan` or
    /// referenced from the current input row.
    pub source_endpoint: LogicalCreateEndpoint,
    /// Sub-plan producing the target NodeId. Same shape as
    /// `source_plan`.
    pub target_plan: Box<LogicalPlan>,
    /// The binding inside `target_plan`'s schema that carries the
    /// target NodeId.
    pub target: BindingId,
    /// True when `target` came from a user-visible CREATE variable,
    /// false when it is a lowering-only anonymous endpoint binding.
    pub target_visible: bool,
    /// Whether the target endpoint is produced by `target_plan` or
    /// referenced from the current input row.
    pub target_endpoint: LogicalCreateEndpoint,
    /// Rel direction. Per ADR-148 §D-1 only `LeftToRight` and
    /// `RightToLeft` are admissible at Phase 2; the substrate write
    /// converts to canonical (src, dst) form before calling
    /// `crud::create_rel`.
    pub direction: CreateRelDirection,
    /// Optional upstream row stream this CREATE-rel chains onto
    /// (issue #832). Mirrors [`LogicalCreateNode::input`]: for a
    /// multi-item `CREATE (a)-[:R]->(b),(c)-[:R]->(d)` item N's
    /// `input` carries items `1..N-1`, so every path executes
    /// (previously only the last path's nodes + rel persisted). This
    /// is distinct from `source_plan` / `target_plan` (the endpoint
    /// node producers, which the op pulls internally). `None` marks
    /// the chain leaf.
    pub input: Option<Box<LogicalPlan>>,
    /// Span of the source CREATE-rel construct.
    pub span: Span,
}

/// Endpoint source for a [`LogicalCreateRel`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogicalCreateEndpoint {
    /// Resolve the endpoint by pulling the corresponding endpoint
    /// sub-pipeline and reading the binding from that row.
    Fresh,
    /// Resolve the endpoint from the current input row's schema.
    RowBinding(BindingId),
}

// =====================================================================
// ADR-149 W26-θ Phase 3 — DELETE logical plan operator
// =====================================================================

/// `DELETE` / `DETACH DELETE` write op (ADR-149 W26-θ Phase 3).
///
/// Lowered from `BoundClause::Delete` over the prior MATCH-produced
/// row stream. The operator pulls rows from `input` exhaustively;
/// per row, it dispatches each item to the substrate's `delete_node`
/// or `delete_rel` (per [`LogicalDeleteItem::kind`]) using the row's
/// cell at the item's `binding` schema slot to resolve the
/// NodeId / RelId being tombstoned.
///
/// # Schema
///
/// The output schema is EMPTY at Phase 3 — DELETE is a terminal
/// clause and produces no downstream rows. RETURN-after-DELETE
/// (openCypher v9 §6's `DELETE n RETURN n` shape) is forward-pinned
/// to Phase 4+ per ADR-149 §"Forward-deferred".
///
/// # DETACH semantic
///
/// When `detach = true`, every Node-typed item in `items` triggers
/// the substrate's cascade-rel-tombstone path BEFORE the node-
/// tombstone is staged. When `detach = false`, a Node-typed item
/// over a node with attached rels surfaces an `ExecutionError::Eval`
/// with the openCypher v9 §6 "relationships attached" message
/// (a dedicated `RelationshipsAttached` variant lights at v1.1+
/// per the 7-slice 3-strike rule).
///
/// # Span
///
/// Carries the span of the source DELETE keyword + items — used by
/// future M4-91 EXPLAIN output to annotate the delete-op node.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalDelete {
    /// Upstream sub-plan producing rows that bind the variables in
    /// `items`. At Phase 3 this is typically the LogicalPlan produced
    /// by the prior MATCH; the executor consumes the upstream
    /// exhaustively and deletes one item-id per row.
    pub input: Box<LogicalPlan>,
    /// One entry per item in the source `DELETE var (, var)*`.
    pub items: Vec<LogicalDeleteItem>,
    /// `true` if the source was `DETACH DELETE ...` — the executor
    /// cascade-tombstones attached rels FIRST for each Node-typed
    /// item before staging the node tombstone.
    pub detach: bool,
    /// Span of the source DELETE construct.
    pub span: Span,
}

/// One bound DELETE item: the upstream-bound binding + the
/// substrate-dispatch discriminator. The executor reads the
/// upstream row's cell at the item's `binding` schema slot to
/// resolve the NodeId or RelId at execute-time; dispatches to
/// `delete_node` (for `Node`) or `delete_rel` (for `Rel`) per
/// `kind`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogicalDeleteItem {
    /// The upstream binding carrying the NodeId / RelId.
    pub binding: BindingId,
    /// Node vs Rel — derived from the bound type_info per ADR-149
    /// §D-6.
    pub kind: DeleteKind,
    /// Span of the source `DELETE var` reference.
    pub span: Span,
}

/// Discriminator for a DELETE item — Node vs Relationship — driving
/// the substrate dispatch (`delete_node` vs `delete_rel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteKind {
    Node,
    Rel,
}

// ─────────────────────────────────────────────────────────────────
// ADR-150 W26-θ Phase 4: SET / REMOVE
// ─────────────────────────────────────────────────────────────────

/// `SET` write op (ADR-150 W26-θ Phase 4).
///
/// Lowered from `BoundClause::Set` over the prior MATCH-produced row
/// stream. The operator pulls rows from `input` exhaustively; per row
/// it dispatches each item to the substrate's `set_node` /
/// `set_rel` (per [`LogicalSetItem::kind`]) using the row's cell at
/// the item's `binding` schema slot to resolve the NodeId / RelId
/// being mutated.
///
/// # Schema + emission (#709 fix, R1-narrowed)
///
/// The output schema EQUALS the input schema — SET binds no new columns;
/// it mutates the substrate. The physical [`crate::executor::ops::SetOp`]
/// emission is **terminal-vs-stacked** (set at
/// [`crate::executor::Pipeline::build`] time):
/// - A **stacked** SET — the inner clause of `SET … SET …` /
///   `SET … REMOVE …` (the `Set(v=1, Set(v=0, Scan))` lowering of
///   `SET n.a = 0 SET n.a = 1`) — PASSES its mutated rows THROUGH so the
///   outer write-op composes (#709: pre-fix only the innermost clause
///   ran, persisting the first write).
/// - A **terminal** SET — the pipeline root / no write-op consumer above
///   it — DRAINS the upstream and emits **0 rows** (the RETURN-less
///   terminal-write contract per openCypher v9 / ADR-149/150 §D /
///   ADR-182; emitting rows from a terminal write breaks the openCypher
///   TCK write-op RowSet conformance gate).
///
/// # Span
///
/// Carries the span of the source SET keyword + items.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSet {
    /// Upstream sub-plan producing rows that bind the variables in
    /// `items`.
    pub input: Box<LogicalPlan>,
    /// One entry per item in the source `SET <item> (, <item>)*`.
    pub items: Vec<LogicalSetItem>,
    /// Span of the source SET construct.
    pub span: Span,
}

/// One bound SET item: the upstream-bound binding, the substrate-
/// dispatch discriminator (Node vs Rel), and the mutation shape.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalSetItem {
    pub binding: BindingId,
    pub kind: SetTargetKind,
    pub mutation: LogicalSetMutation,
    pub span: Span,
}

/// Discriminator for a SET / REMOVE item — Node vs Relationship —
/// driving the substrate dispatch (`set_node` / `set_rel` /
/// `remove_node` / `remove_rel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetTargetKind {
    Node,
    Rel,
}

/// The four bound mutation shapes a `SET` item can take per
/// ADR-150 §D-6.
///
/// `BoundExpression` values in `PropertyAssign` / `PropertyReplace` /
/// `PropertyMerge` are constrained to literals at the type-check pass
/// (ADR-150 §D-4 inherited from ADR-147 §D-4); the bound AST
/// preserves general expression shape so the lowering can carry the
/// `BoundExpression::Literal` through to executor-eval time without
/// re-typing.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalSetMutation {
    PropertyAssign {
        name: String,
        value: BoundExpression,
    },
    PropertyReplace(Vec<(String, BoundExpression)>),
    PropertyMerge(Vec<(String, BoundExpression)>),
    LabelAdd(Vec<String>),
}

/// `REMOVE` write op (ADR-150 W26-θ Phase 4; #709 fix, R1-narrowed).
///
/// Symmetric to [`LogicalSet`] — single-input operator over the
/// MATCH-produced row stream; per-row dispatch of each item's removal
/// (property or label) via the substrate. Output schema = input schema
/// (binds no columns); the physical [`crate::executor::ops::RemoveOp`]
/// emission is **terminal-vs-stacked** like [`LogicalSet`]: a stacked
/// REMOVE passes its rows through so a stacked outer write-op composes
/// (#709); a terminal REMOVE drains the upstream and emits **0 rows**
/// (the RETURN-less terminal-write contract).
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalRemove {
    pub input: Box<LogicalPlan>,
    pub items: Vec<LogicalRemoveItem>,
    pub span: Span,
}

/// One bound REMOVE item.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalRemoveItem {
    pub binding: BindingId,
    pub kind: SetTargetKind,
    pub mutation: LogicalRemoveMutation,
    pub span: Span,
}

/// The two bound removal shapes a `REMOVE` item can take per
/// ADR-150 §D-6.
#[derive(Debug, Clone, PartialEq)]
pub enum LogicalRemoveMutation {
    Property(String),
    LabelRemove(Vec<String>),
}

// =====================================================================
// ADR-151 W26-θ Phase 5 — MERGE logical plan operator
// =====================================================================

/// **NN-4 (#1384) — the MERGE get-or-create serialization key.**
///
/// Captures the merge pattern's UNIQUE IDENTITY so the executor can
/// serialize concurrent `MERGE` on the same key (see
/// [`crate::executor::ops::MergeOp`]). Built at lowering time from the
/// node-shape merge pattern's label + inline property map.
///
/// The property VALUES are carried as `BoundExpression` (NOT resolved
/// literals) because a MERGE property may be a **parameter**
/// (`MERGE (u:User {email:$e})`) — resolved only at execute time against
/// the query's bound parameter bag. The executor evaluates them at
/// `next_batch` time (via `crate::executor::eval::evaluate`) to build the
/// concrete, injection-safe lock key.
///
/// # Key canonicalization (NN-4 #1384 re-spin, Fix 2)
///
/// The lock-key string is order-independent AND int/float-coercion-safe
/// so two MERGEs that the match-filter would treat as the SAME identity
/// lock IDENTICALLY:
/// - **Property order** — the resolver SORTS `properties` by name before
///   rendering, so `{a:1,b:2}` and `{b:2,a:1}` produce the SAME key (the
///   match filter is an order-insensitive AND-conjunction, so a
///   verbatim-order key would false-split into two mutexes → both create).
/// - **Integral Float → Integer** — the resolver normalizes a Float that
///   equals an integer (`1.0`, `42.0`) to that Integer before rendering,
///   mirroring the `=`-operator's `(x as f64) == y` numeric coercion
///   (`eval.rs` `values_equal_3vl`). So `{v:1}` (an Integer) and `{v:1.0}`
///   (a Float) — which the match filter treats as EQUAL — lock on the
///   same key rather than false-splitting.
///
/// # Which merges carry a spec
///
/// - **Node-shape** (`MERGE (n:Label {props})`) → one `MergeKeySpec`
///   (label + property set).
/// - **Path-shape** (`MERGE (a:A {..})-[r:R]->(b:B {..})`) → TWO specs
///   (source endpoint + target endpoint), acquired in canonical total
///   order (Fix 3) so two identical path-MERGEs get exactly one path.
/// - **Anonymous** (`MERGE (:Label)`) → no spec (no read-then-create
///   idempotency contract to protect on a specific key).
#[derive(Debug, Clone, PartialEq)]
pub struct MergeKeySpec {
    /// The merge pattern node's label NAME (`None` for a label-agnostic
    /// `MERGE (n {id:42})`). Part of the key identity so
    /// `MERGE (a:User {id:1})` and `MERGE (a:Account {id:1})` never
    /// contend on the same lock.
    pub label: Option<String>,
    /// The merge pattern node's inline property `(key, value-expression)`
    /// pairs, in pattern order. Evaluated at execute time to resolve
    /// parameters into the concrete key, then SORTED by name +
    /// int/float-normalized so the key is order-independent and
    /// coercion-safe (see the type-level docs, Fix 2).
    pub properties: Vec<(String, BoundExpression)>,
}

/// `MERGE` (match-or-create) write op (ADR-151 W26-θ Phase 5).
///
/// Lowered from `BoundClause::Merge`. The operator wraps a
/// match-branch sub-plan (a Scan / Filter chain that probes the
/// merge pattern's literal property bag in the current snapshot) +
/// a create-branch sub-plan (a CreateNode / CreateRel chain) +
/// optional on_create / on_match action item vecs.
///
/// # Execution semantics
///
/// On first `next_batch`:
/// 1. Pull `match_branch` to exhaustion → collect matched rows.
/// 2. If matched rows is non-empty: emit them as the MERGE's output;
///    fire `on_match` actions per row.
/// 3. If matched rows is empty: pull `create_branch` (one row out);
///    emit that row; fire `on_create` actions on it.
///
/// # Schema
///
/// The MERGE op's output schema is driven by [`Self::output_binding`]
/// (ADR-151-amendment-01 §D-1): `Some(b)` ⇒ `[b]` and each emitted row
/// is `[Value::Node(NodeView)]` (RETURN-after-MERGE, node-shape named);
/// `None` ⇒ empty schema and the op stays terminal (path-shape +
/// anonymous merges — RC-3 boundary). Pre-amendment the op was
/// unconditionally terminal (ADR-151 §D-9); the amendment lifts the
/// node-shape named case to v1.0-α.
///
/// # Span
///
/// Carries the span of the source MERGE keyword + pattern.
#[derive(Debug, Clone, PartialEq)]
pub struct LogicalMerge {
    /// Sub-plan that probes for the merge pattern. Typically a
    /// `Scan + Filter` (Node-shape) or `Scan + Filter + Expand +
    /// Filter + Scan + Filter` (Path-shape) chain.
    pub match_branch: Box<LogicalPlan>,
    /// Sub-plan that creates the merge pattern when match_branch is
    /// empty. Typically a `CreateNode` (Node-shape) or `CreateRel`
    /// wrapping `CreateNode` source + `CreateNode` target (Path-shape).
    pub create_branch: Box<LogicalPlan>,
    /// Items to fire on the create branch (per ADR-150 §D-6
    /// `LogicalSetItem` shape).
    pub on_create: Vec<LogicalSetItem>,
    /// Items to fire on the match branch.
    pub on_match: Vec<LogicalSetItem>,
    /// **ADR-151-amendment-01 §D-1/§D-3** — explicit RETURN-after-MERGE
    /// emission discriminator. `Some(binding)` for a **node-shape
    /// NAMED** merge (`MERGE (n:Label {…})`): the executor emits the
    /// matched/created binding row(s) carrying `Value::Node(NodeView)`
    /// at column 0, so a downstream `Project` can resolve `n` / `n.id`
    /// (the output schema is `[binding]`). `None` for **path-shape**
    /// merges (inconsistent match `[source, rel, target]` vs create
    /// `[rel]` schemas — un-unionable) and **anonymous** node merges
    /// (`MERGE (:Label)`): the op stays terminal (empty schema, empty
    /// batch).
    ///
    /// This is a deliberate *explicit discriminator*, NOT a
    /// `match_branch ∪ create_branch` schema union — the union is wrong
    /// for the path / anonymous shapes (design review RC-3). For a
    /// node-shape named merge both branches already produce the SAME
    /// binding id (the binder's single `bind_create_node_spec` mints it
    /// once; `lower_merge_node_scan` + `lower_merge_create_branch` both
    /// thread it), so `[binding]` equals both branch schemas.
    pub output_binding: Option<BindingId>,
    /// **NN-4 (#1384)** — the get-or-create serialization keys. When
    /// non-empty, the executor serializes the match→create critical
    /// section on these keys (acquired in canonical total order) so two
    /// concurrent `MERGE` on the SAME identity cannot both create (the
    /// SI + OCC double-create hole). See [`MergeKeySpec`].
    ///
    /// - **Node-shape** `MERGE (n:Label {props})` → ONE key (the node's
    ///   label + property set).
    /// - **Path-shape** `MERGE (a)-[r]->(b)` → TWO keys (source + target
    ///   endpoints — Fix 3 of the NN-4 re-spin). The executor acquires
    ///   them sorted so two path-MERGEs naming the same endpoints in
    ///   opposite pattern order cannot deadlock.
    /// - **Anonymous** `MERGE (:Label)` → EMPTY (no key; runs
    ///   unserialized, byte-identical to the pre-NN-4 path).
    ///
    /// (Pre-respin this was a single `Option<MergeKeySpec>` — node-only.
    /// The `Vec` generalizes to the path-endpoint set without a second
    /// discriminator field.)
    pub merge_keys: Vec<MergeKeySpec>,
    pub span: Span,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_accessor_returns_node_span() {
        let plan = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(3, 7),
        });
        assert_eq!(plan.span(), &Span::point(3, 7));
    }

    #[test]
    fn direction_from_reldirection_is_isomorphic_at_m4_31() {
        assert_eq!(
            Direction::from(&RelDirection::LeftToRight),
            Direction::LeftToRight,
        );
        assert_eq!(
            Direction::from(&RelDirection::RightToLeft),
            Direction::RightToLeft,
        );
        assert_eq!(
            Direction::from(&RelDirection::Undirected),
            Direction::Undirected,
        );
    }

    #[test]
    fn join_condition_carries_shared_bindings() {
        let cond = JoinCondition::SharedBindings(vec![BindingId::new(0), BindingId::new(2)]);
        match cond {
            JoinCondition::SharedBindings(ids) => assert_eq!(ids.len(), 2),
        }
    }

    #[test]
    fn logical_plan_is_clone() {
        let plan = LogicalPlan::Scan(LogicalScan {
            label: None,
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: Span::point(1, 7),
        });
        let cloned = plan.clone();
        assert_eq!(plan, cloned);
    }

    // -----------------------------------------------------------------
    // D-2 (ADR-147 §D-8) — LogicalPlan::writes() statement-mutates gate
    // -----------------------------------------------------------------

    fn empty() -> LogicalPlan {
        LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(0, 0),
        })
    }

    fn scan() -> LogicalPlan {
        LogicalPlan::Scan(LogicalScan {
            label: None,
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: Span::point(0, 0),
        })
    }

    #[test]
    fn writes_is_false_for_read_only_plans() {
        // Pure MATCH / RETURN plans do not mutate → no statement txn.
        assert!(!scan().writes(), "a bare Scan is read-only");
        assert!(!empty().writes(), "Empty is read-only");
        let projected = LogicalPlan::Project(LogicalProject {
            input: Box::new(scan()),
            items: Vec::new(),
            span: Span::point(0, 0),
        });
        assert!(!projected.writes(), "Project(Scan) is read-only");
    }

    #[test]
    fn writes_is_true_for_create_node() {
        let create = LogicalPlan::CreateNode(LogicalCreateNode {
            var: Some(BindingId::new(0)),
            label: Some("User".into()),
            properties: Vec::new(),
            input: None,
            span: Span::point(0, 0),
        });
        assert!(create.writes(), "CREATE node mutates the graph");
    }

    #[test]
    fn writes_recurses_into_write_below_a_read_wrapper() {
        // `MATCH (n) DELETE n` lowers to `Delete(input: Scan)`; the write
        // is at the root here, but a read wrapper ABOVE a write (e.g. a
        // future Project over a Delete's returned rows) must still report
        // the statement as mutating. Confirms the recursion.
        let delete = LogicalPlan::Delete(LogicalDelete {
            input: Box::new(scan()),
            items: Vec::new(),
            detach: false,
            span: Span::point(0, 0),
        });
        assert!(delete.writes(), "DELETE mutates");
        let projected_over_delete = LogicalPlan::Project(LogicalProject {
            input: Box::new(delete),
            items: Vec::new(),
            span: Span::point(0, 0),
        });
        assert!(
            projected_over_delete.writes(),
            "a read wrapper over a write still reports the statement as mutating"
        );
    }

    #[test]
    fn community_lookup_span_routes_through_enum_accessor() {
        let plan = LogicalPlan::CommunityLookup(LogicalCommunityLookup {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty {
                span: Span::point(1, 1),
            })),
            node_var: BindingId::new(0),
            community_id: BoundExpression::Parameter {
                name: "cid".into(),
                span: Span::point(2, 5),
                type_info: None,
            },
            read_lsn: Lsn::MAX,
            span: Span::point(2, 1),
        });
        assert_eq!(plan.span(), &Span::point(2, 1));
    }

    #[test]
    fn left_outer_join_carries_shared_bindings_condition() {
        let l = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(1, 1),
        });
        let r = LogicalPlan::Empty(LogicalEmpty {
            span: Span::point(2, 1),
        });
        let plan = LogicalPlan::LeftOuterJoin(LogicalLeftOuterJoin {
            left: Box::new(l),
            right: Box::new(r),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0)]),
            span: Span::point(3, 1),
        });
        match &plan {
            LogicalPlan::LeftOuterJoin(j) => match &j.on {
                JoinCondition::SharedBindings(ids) => {
                    assert_eq!(ids, &vec![BindingId::new(0)]);
                }
            },
            _ => panic!("expected LeftOuterJoin"),
        }
    }

    #[test]
    fn fusion_kind_only_admits_rrf_at_v1_0() {
        let s = FusionSpec {
            kind: FusionKind::Rrf,
            k: 60,
            span: Span::point(1, 1),
        };
        assert_eq!(s.kind, FusionKind::Rrf);
    }

    #[test]
    fn hybrid_operand_kind_round_trips() {
        assert_eq!(HybridOperandKind::Vector, HybridOperandKind::Vector);
        assert_ne!(HybridOperandKind::Vector, HybridOperandKind::Text);
    }

    // -----------------------------------------------------------------
    // M4-33 — type-level smoke tests for the new variants
    // -----------------------------------------------------------------

    #[test]
    fn aggregation_kind_resolves_v1_0_function_names() {
        assert_eq!(
            AggregationKind::from_function_name("count"),
            Some(AggregationKind::Count)
        );
        assert_eq!(
            AggregationKind::from_function_name("SUM"),
            Some(AggregationKind::Sum),
            "case-insensitive match per openCypher 9 §3"
        );
        assert_eq!(
            AggregationKind::from_function_name("collect"),
            Some(AggregationKind::Collect)
        );
        assert_eq!(
            AggregationKind::from_function_name("avg"),
            Some(AggregationKind::Avg),
            "avg lowercase must resolve to AggregationKind::Avg"
        );
        assert_eq!(
            AggregationKind::from_function_name("min"),
            Some(AggregationKind::Min),
            "min lowercase must resolve to AggregationKind::Min"
        );
        assert_eq!(
            AggregationKind::from_function_name("max"),
            Some(AggregationKind::Max),
            "max lowercase must resolve to AggregationKind::Max"
        );
        assert_eq!(
            AggregationKind::from_function_name("not_an_aggregation"),
            None
        );
    }

    #[test]
    fn sort_direction_distinguishes_asc_desc() {
        assert_ne!(SortDirection::Asc, SortDirection::Desc);
    }

    #[test]
    fn dynamic_limit_kind_distinguishes_limit_skip() {
        assert_ne!(DynamicLimitKind::Limit, DynamicLimitKind::Skip);
    }

    #[test]
    fn path_algorithm_distinguishes_plain_shortest() {
        assert_ne!(PathAlgorithm::Plain, PathAlgorithm::ShortestPath);
    }

    #[test]
    fn logical_aggregate_span_routes_through_enum_accessor() {
        let plan = LogicalPlan::Aggregate(LogicalAggregate {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty {
                span: Span::point(1, 1),
            })),
            group_by: Vec::new(),
            aggregations: Vec::new(),
            span: Span::point(2, 1),
        });
        assert_eq!(plan.span(), &Span::point(2, 1));
    }

    #[test]
    fn logical_named_path_carries_algorithm() {
        let plan = LogicalPlan::NamedPath(LogicalNamedPath {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty {
                span: Span::point(1, 1),
            })),
            path_var: BindingId::new(0),
            algorithm: PathAlgorithm::ShortestPath,
            plain_shape: None,
            source: None,
            target: None,
            span: Span::point(2, 1),
        });
        match &plan {
            LogicalPlan::NamedPath(np) => {
                assert_eq!(np.algorithm, PathAlgorithm::ShortestPath);
            }
            _ => panic!("expected NamedPath"),
        }
    }
}
