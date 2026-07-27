//! [`PlanTree`] — the cost-annotated, agent-renderable projection of
//! [`crate::planner::cost::CostedPlan`] returned by EXPLAIN.
//!
//! # Slice scope (M4-91)
//!
//! `PlanTree` is the EXPLAIN-side public type the M5-07 (`graph.search`)
//! / M5-11 (`graph.raw_query`) / M5-13 (Bolt) MCP surfaces serialize to
//! JSON / TOON for agent consumption per ADR-038 §2 D-19 + amendment-03
//! §M5↔M4 contract surface. It is a one-way projection of [`CostedPlan`]:
//! cost numbers + bindings + operator-shape are preserved verbatim;
//! [`crate::semantic::bound_ast::BoundExpression`] payloads (filter
//! predicates, projection items, etc.) are NOT re-serialized — the
//! `annotations` slot carries human-grade summary strings instead.
//!
//! # Round-trip discipline
//!
//! Per the M4-91 proptest pin (`tests/m4_91_explain_proptest.rs`):
//! - Tree shape (`children.len()` per node) MUST match the source
//!   [`CostedPlan`]'s [`crate::planner::cost::CostedTree`] shape exactly.
//! - `estimated_cost.total()` and `estimated_card.rows()` MUST equal
//!   the corresponding [`CostedTree`] node's `subtree_cost` /
//!   `output_card`.
//! - `annotations` is a `BTreeMap` (NOT `HashMap`) so `Display`
//!   output is byte-identical across runs — `Display` stability is
//!   load-bearing for the snapshot-test pin in
//!   `tests/m4_91_explain_integration.rs`.
//!
//! # ADR provenance
//! - ADR-038 §2 D-19 — EXPLAIN/PROFILE return shape.
//! - ADR-038 amendment-03 §TIER-1 GAP B — M4-91 sub-slice scope.
//! - ADR-038 amendment-03 §M5↔M4 contract surface — `PlanTree` /
//!   `ExecutionMetrics` typing.
//! - ADR-036 §D-25 — 5 ms M4-05 plan-build budget; the EXPLAIN walk is
//!   `O(plan-nodes)` post-cost-walker — well inside budget.

use std::collections::BTreeMap;

use crate::logical_plan::types::{
    AggregationKind, DynamicLimitKind, FusionKind, HybridOperandKind, JoinCondition, LogicalPlan,
    PathAlgorithm, SortDirection,
};
use crate::planner::cost::{COST_HINT_HIGH, Cardinality, Cost, CostedPlan, CostedTree};
use crate::semantic::bound_ast::BindingId;

/// Operator kind for a [`PlanTree`] node.
///
/// Mirrors the [`LogicalPlan`] taxonomy but carries no nested data:
/// every operator-specific datum the EXPLAIN consumer needs lives in
/// the parent [`PlanTree`]'s `bindings` / `annotations` slots. This
/// keeps the EXPLAIN-side enum payload-free + cheap to clone +
/// trivially serializable.
///
/// # Exhaustive-match contract
///
/// `PlanTreeOp` is **NOT** `#[non_exhaustive]`. The variant set mirrors
/// [`LogicalPlan`]; new [`LogicalPlan`] variants force a compile error
/// here, which is the design intent (a new operator MUST be wired into
/// the EXPLAIN renderer at the same time).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlanTreeOp {
    Scan,
    /// **#1366 (Phase 2).** Indexed property point-lookup — the EXPLAIN
    /// marker operators use to confirm the index path is live (not a
    /// full scan). Renders `PropertyIndexScan(label=…, property=…,
    /// residual=…)`.
    PropertyIndexScan,
    CountStore,
    Expand,
    Filter,
    Project,
    Join,
    LeftOuterJoin,
    Limit,
    Skip,
    RankByHybrid,
    Fusion,
    CommunityLookup,
    VectorNear,
    TextMatch,
    Aggregate,
    Sort,
    Distinct,
    /// **ADR-185 (#649-A1, W28).** UNION ALL set-op concat.
    Union,
    Unwind,
    NamedPath,
    DynamicLimit,
    /// **ADR-147 W26-θ Phase 1.** CREATE node write op.
    CreateNode,
    /// **#830 / ADR-200.** CREATE VECTOR INDEX accept-and-register
    /// write op (metadata-only catalog registration).
    CreateVectorIndex,
    /// **#1366 (task #248, Phase 1).** CREATE INDEX property-index
    /// register + backfill + Online-flip write op.
    CreatePropertyIndex,
    /// **ADR-148 W26-θ Phase 2.** CREATE rel write op.
    CreateRel,
    /// **ADR-149 W26-θ Phase 3.** DELETE / DETACH DELETE write op.
    Delete,
    /// **ADR-150 W26-θ Phase 4.** SET write op.
    Set,
    /// **ADR-150 W26-θ Phase 4.** REMOVE write op.
    Remove,
    /// **ADR-151 W26-θ Phase 5.** MERGE (match-or-create) write op.
    Merge,
    /// **ADR-192 (#623).** `CALL { <subquery> }` correlated subquery
    /// (Cypher 25, beyond openCypher v9).
    Call,
    /// **ADR-192 (#623).** The one-row correlation seed feeding a
    /// `CALL { … }` body.
    CorrelationSeed,
    /// **ADR-197 (#802).** `CALL <proc>(…) [YIELD …]` / `SHOW …`
    /// schema-introspection generating operator.
    ProcedureCall,
    Empty,
}

impl PlanTreeOp {
    /// Stable, agent-readable name for the operator. Used by
    /// [`crate::explain::format`] and by JSON / TOON serializers.
    ///
    /// Names are CamelCase (matching the `LogicalPlan` variant names)
    /// so `git diff` between an EXPLAIN snapshot and its
    /// [`LogicalPlan`] source remains readable.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            PlanTreeOp::Scan => "Scan",
            PlanTreeOp::PropertyIndexScan => "PropertyIndexScan",
            PlanTreeOp::CountStore => "CountStore",
            PlanTreeOp::Expand => "Expand",
            PlanTreeOp::Filter => "Filter",
            PlanTreeOp::Project => "Project",
            PlanTreeOp::Join => "Join",
            PlanTreeOp::LeftOuterJoin => "LeftOuterJoin",
            PlanTreeOp::Limit => "Limit",
            PlanTreeOp::Skip => "Skip",
            PlanTreeOp::RankByHybrid => "RankByHybrid",
            PlanTreeOp::Fusion => "Fusion",
            PlanTreeOp::CommunityLookup => "CommunityLookup",
            PlanTreeOp::VectorNear => "VectorNear",
            PlanTreeOp::TextMatch => "TextMatch",
            PlanTreeOp::Aggregate => "Aggregate",
            PlanTreeOp::Sort => "Sort",
            PlanTreeOp::Distinct => "Distinct",
            PlanTreeOp::Union => "Union",
            PlanTreeOp::Unwind => "Unwind",
            PlanTreeOp::NamedPath => "NamedPath",
            PlanTreeOp::DynamicLimit => "DynamicLimit",
            PlanTreeOp::CreateNode => "CreateNode",
            PlanTreeOp::CreateVectorIndex => "CreateVectorIndex",
            PlanTreeOp::CreatePropertyIndex => "CreatePropertyIndex",
            PlanTreeOp::CreateRel => "CreateRel",
            PlanTreeOp::Delete => "Delete",
            PlanTreeOp::Set => "Set",
            PlanTreeOp::Remove => "Remove",
            PlanTreeOp::Merge => "Merge",
            PlanTreeOp::Call => "Call",
            PlanTreeOp::CorrelationSeed => "CorrelationSeed",
            PlanTreeOp::ProcedureCall => "ProcedureCall",
            PlanTreeOp::Empty => "Empty",
        }
    }
}

/// Cost-annotated plan tree returned by EXPLAIN per ADR-038 §2 D-19 +
/// amendment-03 §TIER-1 GAP B.
///
/// Construct via [`PlanTree::from_costed_plan`] (consumes a
/// [`CostedPlan`] from the M4-51 cost walker). The resulting tree is
/// `Send + Sync + Clone`, deterministic across runs, and carries all
/// data the M5-07 / M5-11 / M5-13 surfaces need to serialize EXPLAIN
/// output.
///
/// # Field semantics
///
/// - `op` — operator kind (see [`PlanTreeOp`]).
/// - `bindings` — source-meaningful variables touched by this operator
///   (rendered as `b{raw}` so the rendering is deterministic regardless
///   of the binding pass's id-allocation order). Empty for operators
///   that do not introduce or test bindings (Filter, Limit, Skip,
///   Sort, Fusion, Empty).
/// - `estimated_cost` — root-of-subtree cost (`subtree_cost` from the
///   M4-51 walker). EXPLAIN consumers compare costs across alternative
///   plans by reading the root node's `estimated_cost`.
/// - `estimated_card` — output cardinality flowing OUT of this
///   operator (consumed by the parent's input-cardinality slot
///   conceptually).
/// - `children` — child sub-trees in source order. Pre-order walk
///   matches [`CostedPlan::plan`]'s pre-order walk by construction.
/// - `annotations` — operator-specific human-grade fields. Always a
///   `BTreeMap` so `Display` order is stable.
#[derive(Debug, Clone, PartialEq)]
pub struct PlanTree {
    /// Operator kind — see [`PlanTreeOp`].
    pub op: PlanTreeOp,
    /// Bindings the operator introduces or tests, rendered as
    /// `b{raw}` (e.g., `b0`, `b1`). Order matches the underlying
    /// [`LogicalPlan`] variant's natural order (head before tail; from
    /// before to; left before right; etc.).
    pub bindings: Vec<String>,
    /// Subtree-cumulative cost from the M4-51 walker. Equivalent to
    /// `costed_tree.cost.subtree_cost`.
    pub estimated_cost: Cost,
    /// Output cardinality from the M4-51 walker. Equivalent to
    /// `costed_tree.cost.output_card`.
    pub estimated_card: Cardinality,
    /// Child sub-trees in source order.
    pub children: Vec<PlanTree>,
    /// Operator-specific annotations (label, K, fusion-k, direction,
    /// etc.). Always a `BTreeMap` so `Display` iteration order is
    /// deterministic.
    pub annotations: BTreeMap<String, String>,
}

impl PlanTree {
    /// Project a [`CostedPlan`] into an EXPLAIN-renderable [`PlanTree`].
    ///
    /// The walk is a single pre-order traversal of the
    /// (LogicalPlan, CostedTree) pair. Per the M4-51 walker contract,
    /// the two trees have identical shape; this function asserts that
    /// invariant via paired iteration.
    ///
    /// # Determinism
    ///
    /// Same input [`CostedPlan`] → same `PlanTree` byte-for-byte
    /// (modulo `Cost` / `Cardinality` which are `f64` and so equal-up-
    /// to-bit-pattern only for the exact same input — saturation in
    /// the cost walker keeps these well-behaved).
    #[must_use]
    pub fn from_costed_plan(costed: &CostedPlan) -> Self {
        let mut tree = Self::build(costed.plan(), costed.costs());
        if !costed.diagnostics().is_empty() {
            tree.annotations
                .insert("cost_hint".into(), COST_HINT_HIGH.into());
            let mut diagnostics = costed
                .diagnostics()
                .iter()
                .take(8)
                .cloned()
                .collect::<Vec<_>>();
            if costed.diagnostics().len() > diagnostics.len() {
                diagnostics.push(format!(
                    "... {} more diagnostics omitted",
                    costed.diagnostics().len() - diagnostics.len()
                ));
            }
            tree.annotations
                .insert("diagnostics".into(), diagnostics.join(" | "));
        }
        tree
    }

    /// Recursive lockstep walk over `(LogicalPlan, CostedTree)`. Per
    /// the M4-51 walker contract the two have identical structure; if
    /// they ever diverge we'd produce a malformed tree, but the
    /// proptest pin would catch the regression immediately.
    fn build(plan: &LogicalPlan, costs: &CostedTree) -> Self {
        let (op, bindings, annotations) = describe_node(plan);
        let children = match plan {
            // Leaves with NO LogicalPlan children. The cost walker
            // also emits a `CostedTree::leaf` here, so `children`
            // is empty by construction. Defense-in-depth: if a
            // future LogicalPlan widening adds a child slot we'd
            // mismatch, and the proptest catches it.
            LogicalPlan::Scan(_)
            // #1366 (Phase 2): the indexed point-lookup is a LEAF.
            | LogicalPlan::PropertyIndexScan(_)
            | LogicalPlan::CountStore(_)
            | LogicalPlan::Empty(_)
            | LogicalPlan::Expand(_)
            | LogicalPlan::RankByHybrid(_)
            | LogicalPlan::VectorNear(_)
            | LogicalPlan::TextMatch(_)
            // ADR-192 (#623): the correlation seed is a LEAF (its one
            // row is the imported bindings; no LogicalPlan children).
            | LogicalPlan::CorrelationSeed(_)
            // #830 / ADR-200: CREATE VECTOR INDEX is a leaf DDL — no
            // input child, so it STAYS in the zero-children leaf group
            // (unlike CreateNode below, which #832 pulled out to recurse
            // its chain `input`). Lockstep with the cost + row-count
            // walkers (both emit zero children for it).
            | LogicalPlan::CreateVectorIndex(_)
            // #1366: CREATE INDEX (property index) is a leaf DDL too.
            | LogicalPlan::CreatePropertyIndex(_) => Vec::new(),

            // #832: CreateNode is a LEAF only at the chain bottom. A
            // multi-item `CREATE (a),(b),(c)` lowers to a left-deep
            // chain via `input`; EXPLAIN recurses so it shows EVERY
            // create, not just the top one.
            LogicalPlan::CreateNode(c) => c
                .input
                .as_ref()
                .map(|i| vec![Self::build(i, child_at(costs, 0))])
                .unwrap_or_default(),

            LogicalPlan::Filter(f) => {
                vec![Self::build(&f.input, child_at(costs, 0))]
            }
            LogicalPlan::Project(p) => {
                vec![Self::build(&p.input, child_at(costs, 0))]
            }
            LogicalPlan::Limit(l) => {
                vec![Self::build(&l.input, child_at(costs, 0))]
            }
            LogicalPlan::Skip(s) => {
                vec![Self::build(&s.input, child_at(costs, 0))]
            }
            LogicalPlan::DynamicLimit(d) => {
                vec![Self::build(&d.input, child_at(costs, 0))]
            }
            LogicalPlan::Sort(s) => {
                vec![Self::build(&s.input, child_at(costs, 0))]
            }
            LogicalPlan::Distinct(d) => {
                vec![Self::build(&d.input, child_at(costs, 0))]
            }
            LogicalPlan::Unwind(u) => {
                vec![Self::build(&u.input, child_at(costs, 0))]
            }
            LogicalPlan::ProcedureCall(p) => {
                vec![Self::build(&p.input, child_at(costs, 0))]
            }
            LogicalPlan::Aggregate(a) => {
                vec![Self::build(&a.input, child_at(costs, 0))]
            }
            LogicalPlan::CommunityLookup(c) => {
                vec![Self::build(&c.input, child_at(costs, 0))]
            }
            LogicalPlan::NamedPath(np) => {
                vec![Self::build(&np.input, child_at(costs, 0))]
            }

            LogicalPlan::Join(j) => vec![
                Self::build(&j.left, child_at(costs, 0)),
                Self::build(&j.right, child_at(costs, 1)),
            ],
            LogicalPlan::LeftOuterJoin(j) => vec![
                Self::build(&j.left, child_at(costs, 0)),
                Self::build(&j.right, child_at(costs, 1)),
            ],

            LogicalPlan::Fusion(fu) => fu
                .inputs
                .iter()
                .enumerate()
                .map(|(i, child_plan)| Self::build(child_plan, child_at(costs, i)))
                .collect(),

            // ADR-185 (#649-A1, W28) — UNION ALL: one child per arm, in
            // source order (lockstep with the cost walker's n-ary
            // children).
            LogicalPlan::Union(u) => u
                .arms
                .iter()
                .enumerate()
                .map(|(i, arm)| Self::build(arm, child_at(costs, i)))
                .collect(),

            // ADR-148 W26-θ Phase 2: CreateRel has source + target
            // sub-plans (in source order); #832 adds an optional chain
            // `input` child so EXPLAIN shows a prior CREATE item too.
            LogicalPlan::CreateRel(cr) => {
                let mut kids = vec![
                    Self::build(&cr.source_plan, child_at(costs, 0)),
                    Self::build(&cr.target_plan, child_at(costs, 1)),
                ];
                if let Some(input) = &cr.input {
                    kids.push(Self::build(input, child_at(costs, 2)));
                }
                kids
            }

            // ADR-149 W26-θ Phase 3: Delete has ONE child (the input
            // sub-plan — typically the prior MATCH's lowered plan).
            LogicalPlan::Delete(d) => {
                vec![Self::build(&d.input, child_at(costs, 0))]
            }
            // ADR-150 W26-θ Phase 4: Set / Remove each have ONE child
            // (the input sub-plan — typically the prior MATCH's
            // lowered plan).
            LogicalPlan::Set(s) => {
                vec![Self::build(&s.input, child_at(costs, 0))]
            }
            LogicalPlan::Remove(r) => {
                vec![Self::build(&r.input, child_at(costs, 0))]
            }
            // ADR-151 W26-θ Phase 5: Merge has TWO children (match-
            // branch + create-branch, in source order — the executor
            // pulls match first; create fires only on probe miss).
            LogicalPlan::Merge(m) => vec![
                Self::build(&m.match_branch, child_at(costs, 0)),
                Self::build(&m.create_branch, child_at(costs, 1)),
            ],
            // ADR-192 (#623): CALL{} has TWO children (the driving
            // `input` + the subquery `body`, in source order — lockstep
            // with the cost walker's child order).
            LogicalPlan::Call(c) => vec![
                Self::build(&c.input, child_at(costs, 0)),
                Self::build(&c.body, child_at(costs, 1)),
            ],
        };

        let cost_node = costs.cost;
        Self {
            op,
            bindings,
            estimated_cost: cost_node.subtree_cost,
            estimated_card: cost_node.output_card,
            children,
            annotations,
        }
    }
}

/// Project a [`BindingId`] to the deterministic `b{raw}` string.
fn render_binding(id: BindingId) -> String {
    format!("b{}", id.raw())
}

/// Look up a child of a [`CostedTree`] by index. Defensive against a
/// mismatch between the LogicalPlan and CostedTree shapes — falls back
/// to a default leaf node so the build can complete (the mismatch
/// surfaces via the proptest pin).
fn child_at(parent: &CostedTree, idx: usize) -> &CostedTree {
    parent
        .children
        .get(idx)
        .unwrap_or_else(|| panic_costed_shape_mismatch(parent.children.len(), idx))
}

#[cold]
fn panic_costed_shape_mismatch(have: usize, want: usize) -> ! {
    // M4-51 walker contract violation. This is unreachable in
    // production code paths — the walker guarantees lockstep
    // pre-order shape per its module docs §"Determinism" — but a
    // future walker bug would surface here loudly rather than as a
    // silently-corrupted EXPLAIN output. The proptest pin in
    // `tests/m4_91_explain_proptest.rs` covers structural validity
    // (i.e., this branch is never taken on a properly-emitted
    // CostedPlan).
    //
    // # F-5 forward (W9b LOW; track for v1.1 multi-tenant)
    //
    // At v1.1 + `TenantHandle` per ADR-037, a panic in the EXPLAIN
    // path aborts the process serving every other tenant's queries.
    // Convert to `Result<PlanTree, ArcQLError>` propagation BEFORE
    // multi-tenant ships. Until then v1.0 single-tenant is bounded:
    // a panic crashes the single-tenant process which the supervisor
    // restarts. Sister-cite: round-1 PR #240 review FIND-4.
    //
    // Loud structured-event surface added in W9d M4-52b (F-5 partial
    // closure); the panic is preserved but observability is bumped
    // so a future M4-71 ops layer can capture the event pre-abort.
    tracing::error!(
        target: "arcgraph_query::explain::plan_tree",
        have,
        want,
        "PlanTree::build: CostedTree shape mismatch — M4-51 walker contract violation; \
         track for v1.1 multi-tenant Result-propagation cutover (W9b F-5)"
    );
    panic!(
        "PlanTree build: CostedTree shape mismatch — child index {want} requested but \
         CostedTree has only {have} children. This violates the M4-51 walker contract \
         (LogicalPlan and CostedTree must have identical pre-order shape)."
    )
}

/// Describe a single [`LogicalPlan`] node — its `op` kind, declared
/// bindings, and operator-specific annotations.
///
/// Pulled out as a free function so the [`PlanTree::build`] recursion
/// stays focused on the shape walk.
#[allow(clippy::too_many_lines)] // exhaustive match over 20 variants is the design.
fn describe_node(plan: &LogicalPlan) -> (PlanTreeOp, Vec<String>, BTreeMap<String, String>) {
    let mut anns: BTreeMap<String, String> = BTreeMap::new();
    let (op, bindings) = match plan {
        // -------------- Leaves --------------
        LogicalPlan::Scan(s) => {
            anns.insert(
                "label".into(),
                match s.label {
                    Some(l) => format!("L{}", l.raw()),
                    None => "<any>".into(),
                },
            );
            anns.insert("read_lsn".into(), s.read_lsn.raw().to_string());
            (PlanTreeOp::Scan, vec![render_binding(s.var)])
        }
        // #1366 (Phase 2): the indexed point-lookup. The annotations let
        // operators confirm the index path is live: which label +
        // property the index covers, and whether a residual filter runs
        // over the verified rows. The label id is rendered `L{raw}` (the
        // query crate has the interned id, not the catalog name — the
        // Bolt / MCP layer reverse-resolves names at render time).
        LogicalPlan::PropertyIndexScan(p) => {
            anns.insert("label".into(), format!("L{}", p.label.raw()));
            anns.insert("property".into(), p.property.clone());
            anns.insert("residual".into(), p.residual.is_some().to_string());
            anns.insert("read_lsn".into(), p.read_lsn.raw().to_string());
            (PlanTreeOp::PropertyIndexScan, vec![render_binding(p.var)])
        }
        LogicalPlan::CountStore(c) => {
            anns.insert("source".into(), format!("{:?}", c.source));
            (PlanTreeOp::CountStore, vec![render_binding(c.output_id)])
        }
        LogicalPlan::Empty(_) => (PlanTreeOp::Empty, Vec::new()),

        LogicalPlan::Expand(e) => {
            anns.insert(
                "direction".into(),
                match e.direction {
                    crate::logical_plan::types::Direction::LeftToRight => "->".into(),
                    crate::logical_plan::types::Direction::RightToLeft => "<-".into(),
                    crate::logical_plan::types::Direction::Undirected => "-".into(),
                },
            );
            anns.insert(
                "rel_type".into(),
                match e.rel_type {
                    Some(t) => format!("T{}", t.raw()),
                    None => "<any>".into(),
                },
            );
            if let Some(lr) = &e.length_range {
                anns.insert("length".into(), format!("{lr}"));
            }
            let mut bs = vec![render_binding(e.from), render_binding(e.to)];
            if let Some(rv) = e.rel_var {
                bs.push(render_binding(rv));
            }
            (PlanTreeOp::Expand, bs)
        }

        LogicalPlan::RankByHybrid(r) => {
            anns.insert("operands".into(), r.operands.len().to_string());
            // Operand kinds in source order — useful for agent
            // routing decisions ("VECTOR before TEXT" vs the reverse).
            let kinds: Vec<&str> = r
                .operands
                .iter()
                .map(|o| match o.kind {
                    HybridOperandKind::Vector => "VECTOR",
                    HybridOperandKind::Text => "TEXT",
                })
                .collect();
            anns.insert("operand_kinds".into(), kinds.join(","));
            let mut bs: Vec<String> = r.operands.iter().map(|o| render_binding(o.var)).collect();
            if let Some(score) = r.score_binding {
                bs.push(render_binding(score));
            }
            (PlanTreeOp::RankByHybrid, bs)
        }
        LogicalPlan::VectorNear(v) => {
            anns.insert("property".into(), v.property.clone());
            anns.insert("k".into(), v.k.to_string());
            anns.insert("read_lsn".into(), v.read_lsn.raw().to_string());
            (PlanTreeOp::VectorNear, vec![render_binding(v.var)])
        }
        LogicalPlan::TextMatch(t) => {
            anns.insert("property".into(), t.property.clone());
            if let Some(k) = t.k {
                anns.insert("k".into(), k.to_string());
            }
            anns.insert("read_lsn".into(), t.read_lsn.raw().to_string());
            (PlanTreeOp::TextMatch, vec![render_binding(t.var)])
        }

        // -------------- Unary --------------
        LogicalPlan::Filter(_) => (PlanTreeOp::Filter, Vec::new()),
        LogicalPlan::Project(p) => {
            anns.insert("items".into(), p.items.len().to_string());
            (PlanTreeOp::Project, Vec::new())
        }
        LogicalPlan::Limit(l) => {
            anns.insert("count".into(), l.count.to_string());
            (PlanTreeOp::Limit, Vec::new())
        }
        LogicalPlan::Skip(s) => {
            anns.insert("count".into(), s.count.to_string());
            (PlanTreeOp::Skip, Vec::new())
        }
        LogicalPlan::DynamicLimit(d) => {
            anns.insert(
                "kind".into(),
                match d.kind {
                    DynamicLimitKind::Limit => "LIMIT".into(),
                    DynamicLimitKind::Skip => "SKIP".into(),
                },
            );
            (PlanTreeOp::DynamicLimit, Vec::new())
        }
        LogicalPlan::Sort(s) => {
            anns.insert("keys".into(), s.order_by.len().to_string());
            // Direction summary — the per-key direction is not in
            // bindings but is informational ("are we sorting ASC or
            // DESC overall"). Lists 1 letter per key in source order.
            let dirs: String = s
                .order_by
                .iter()
                .map(|k| match k.direction {
                    SortDirection::Asc => 'A',
                    SortDirection::Desc => 'D',
                })
                .collect();
            anns.insert("directions".into(), dirs);
            (PlanTreeOp::Sort, Vec::new())
        }
        LogicalPlan::Distinct(d) => {
            anns.insert("on_count".into(), d.on.len().to_string());
            let bs: Vec<String> = d.on.iter().copied().map(render_binding).collect();
            (PlanTreeOp::Distinct, bs)
        }
        // ADR-185 (#649-A1, W28) — UNION ALL: annotate the arm count so
        // an agent can read the fan-in without re-parsing the query.
        LogicalPlan::Union(u) => {
            anns.insert("arm_count".into(), u.arms.len().to_string());
            (PlanTreeOp::Union, Vec::new())
        }
        LogicalPlan::Unwind(u) => (PlanTreeOp::Unwind, vec![render_binding(u.var)]),
        LogicalPlan::ProcedureCall(p) => {
            let name = match &p.source {
                crate::logical_plan::types::ProcedureSource::Procedure(k) => format!("{k:?}"),
                crate::logical_plan::types::ProcedureSource::Show(k) => format!("SHOW {k:?}"),
            };
            anns.insert("procedure".into(), name);
            (
                PlanTreeOp::ProcedureCall,
                p.columns
                    .iter()
                    .map(|(_, bid)| render_binding(*bid))
                    .collect(),
            )
        }
        LogicalPlan::Aggregate(a) => {
            anns.insert("group_by_count".into(), a.group_by.len().to_string());
            anns.insert("aggregations".into(), a.aggregations.len().to_string());
            // List the aggregation function names so an agent can
            // tell `count` vs `sum` apart without re-parsing the
            // source query.
            let kinds: Vec<&str> = a
                .aggregations
                .iter()
                .map(|spec| match spec.function {
                    AggregationKind::Count => "count",
                    AggregationKind::Sum => "sum",
                    AggregationKind::Avg => "avg",
                    AggregationKind::Min => "min",
                    AggregationKind::Max => "max",
                    AggregationKind::Collect => "collect",
                })
                .collect();
            anns.insert("aggregation_kinds".into(), kinds.join(","));
            (PlanTreeOp::Aggregate, Vec::new())
        }
        LogicalPlan::CommunityLookup(c) => {
            anns.insert("read_lsn".into(), c.read_lsn.raw().to_string());
            (
                PlanTreeOp::CommunityLookup,
                vec![render_binding(c.node_var)],
            )
        }
        LogicalPlan::NamedPath(np) => {
            anns.insert(
                "algorithm".into(),
                match np.algorithm {
                    PathAlgorithm::Plain => "Plain".into(),
                    PathAlgorithm::ShortestPath => "ShortestPath".into(),
                    PathAlgorithm::AllShortestPaths => "AllShortestPaths".into(),
                },
            );
            (PlanTreeOp::NamedPath, vec![render_binding(np.path_var)])
        }

        // -------------- Binary --------------
        LogicalPlan::Join(j) => {
            anns.insert("condition".into(), describe_join_cond(&j.on));
            // W25-M4-61b / ADR-097: surface the picked algorithm so
            // EXPLAIN consumers (M4-91) can see whether the planner
            // resolved Hash vs Merge. `Auto` shows up when the picker
            // has not yet run (typically: direct `estimate_costs`
            // calls in tests).
            anns.insert(
                "algorithm".into(),
                match j.algorithm {
                    crate::logical_plan::JoinAlgorithm::Auto => "auto".into(),
                    crate::logical_plan::JoinAlgorithm::HashJoin => "hash".into(),
                    crate::logical_plan::JoinAlgorithm::MergeJoin => "merge".into(),
                },
            );
            let bs = match &j.on {
                JoinCondition::SharedBindings(ids) => {
                    ids.iter().copied().map(render_binding).collect()
                }
            };
            (PlanTreeOp::Join, bs)
        }
        LogicalPlan::LeftOuterJoin(j) => {
            anns.insert("condition".into(), describe_join_cond(&j.on));
            let bs = match &j.on {
                JoinCondition::SharedBindings(ids) => {
                    ids.iter().copied().map(render_binding).collect()
                }
            };
            (PlanTreeOp::LeftOuterJoin, bs)
        }

        // -------------- N-ary --------------
        LogicalPlan::Fusion(f) => {
            anns.insert(
                "kind".into(),
                match f.spec.kind {
                    FusionKind::Rrf => "RRF".into(),
                },
            );
            anns.insert("k".into(), f.spec.k.to_string());
            anns.insert("input_count".into(), f.inputs.len().to_string());
            (PlanTreeOp::Fusion, Vec::new())
        }
        // -------------- ADR-147 W26-θ Phase 1 — CREATE node ---------
        LogicalPlan::CreateNode(c) => {
            if let Some(l) = &c.label {
                anns.insert("label".into(), l.clone());
            }
            anns.insert("property_count".into(), c.properties.len().to_string());
            let bs: Vec<String> = c.var.iter().copied().map(render_binding).collect();
            (PlanTreeOp::CreateNode, bs)
        }
        // -------------- #830 / ADR-200 — CREATE VECTOR INDEX --------
        LogicalPlan::CreateVectorIndex(c) => {
            anns.insert("label".into(), c.label.clone());
            anns.insert("property".into(), c.property.clone());
            anns.insert("if_not_exists".into(), c.if_not_exists.to_string());
            // A DDL leaf — no output bindings (returns 0 rows).
            (PlanTreeOp::CreateVectorIndex, Vec::new())
        }
        // -------------- #1366 (task #248) — CREATE property INDEX ---
        LogicalPlan::CreatePropertyIndex(c) => {
            anns.insert("label".into(), c.label.clone());
            anns.insert("property".into(), c.property.clone());
            anns.insert("if_not_exists".into(), c.if_not_exists.to_string());
            // A DDL leaf — no output bindings (returns 0 rows).
            (PlanTreeOp::CreatePropertyIndex, Vec::new())
        }
        // -------------- ADR-148 W26-θ Phase 2 — CREATE rel ---------
        LogicalPlan::CreateRel(c) => {
            anns.insert("label".into(), c.label.clone());
            anns.insert("property_count".into(), c.properties.len().to_string());
            anns.insert(
                "direction".into(),
                match c.direction {
                    crate::ast::CreateRelDirection::LeftToRight => "->".into(),
                    crate::ast::CreateRelDirection::RightToLeft => "<-".into(),
                },
            );
            anns.insert("source_binding".into(), render_binding(c.source));
            anns.insert("target_binding".into(), render_binding(c.target));
            let bs: Vec<String> = c.var.iter().copied().map(render_binding).collect();
            (PlanTreeOp::CreateRel, bs)
        }
        // -------------- ADR-149 W26-θ Phase 3 — DELETE ---------
        LogicalPlan::Delete(d) => {
            anns.insert("item_count".into(), d.items.len().to_string());
            anns.insert("detach".into(), d.detach.to_string());
            let bs: Vec<String> = d
                .items
                .iter()
                .map(|it| render_binding(it.binding))
                .collect();
            (PlanTreeOp::Delete, bs)
        }
        // -------------- ADR-150 W26-θ Phase 4 — SET ---------
        LogicalPlan::Set(s) => {
            anns.insert("item_count".into(), s.items.len().to_string());
            let bs: Vec<String> = s
                .items
                .iter()
                .map(|it| render_binding(it.binding))
                .collect();
            (PlanTreeOp::Set, bs)
        }
        // -------------- ADR-150 W26-θ Phase 4 — REMOVE ---------
        LogicalPlan::Remove(r) => {
            anns.insert("item_count".into(), r.items.len().to_string());
            let bs: Vec<String> = r
                .items
                .iter()
                .map(|it| render_binding(it.binding))
                .collect();
            (PlanTreeOp::Remove, bs)
        }
        // -------------- ADR-151 W26-θ Phase 5 — MERGE ----------
        LogicalPlan::Merge(m) => {
            anns.insert("on_create_item_count".into(), m.on_create.len().to_string());
            anns.insert("on_match_item_count".into(), m.on_match.len().to_string());
            // Bindings the MERGE touches: union of on_create + on_match
            // item bindings (each item carries a binding into the
            // pattern's fresh scope; the executor reads the cell at
            // that binding's row slot for action dispatch).
            let mut bs: Vec<String> = Vec::new();
            for item in m.on_create.iter().chain(m.on_match.iter()) {
                let render = render_binding(item.binding);
                if !bs.contains(&render) {
                    bs.push(render);
                }
            }
            (PlanTreeOp::Merge, bs)
        }
        // -------------- ADR-192 (#623) — CALL { … } subquery --------
        LogicalPlan::Call(c) => {
            anns.insert("imported_count".into(), c.imported.len().to_string());
            anns.insert("returned_count".into(), c.returned.len().to_string());
            // Bindings surfaced: the body's returned columns (the only
            // subquery vars that escape the scoping fence — D-4).
            let bs: Vec<String> = c.returned.iter().copied().map(render_binding).collect();
            (PlanTreeOp::Call, bs)
        }
        LogicalPlan::CorrelationSeed(s) => {
            anns.insert("imported_count".into(), s.imported.len().to_string());
            let bs: Vec<String> = s.imported.iter().copied().map(render_binding).collect();
            (PlanTreeOp::CorrelationSeed, bs)
        }
    };
    (op, bindings, anns)
}

fn describe_join_cond(on: &JoinCondition) -> String {
    match on {
        JoinCondition::SharedBindings(ids) if ids.is_empty() => "Cartesian".into(),
        JoinCondition::SharedBindings(ids) => {
            let names: Vec<String> = ids.iter().copied().map(render_binding).collect();
            format!("shared:[{}]", names.join(","))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;
    use crate::logical_plan::types::*;
    use crate::planner::cost::estimate_costs;
    use crate::semantic::StubCatalogProvider;
    use crate::semantic::bound_ast::BoundExpression;
    use arcgraph_core::{LabelId, Lsn};

    fn span() -> Span {
        Span::point(1, 1)
    }

    fn lit_true() -> BoundExpression {
        BoundExpression::Literal {
            value: crate::ast::Literal::Bool(true),
            span: span(),
            type_info: None,
        }
    }

    #[test]
    fn plan_tree_op_name_is_stable_for_every_variant() {
        // Canary: if a new variant is added, this list shrinks at
        // compile-time via the exhaustive `name()` match — keeping
        // the EXPLAIN renderer in lockstep with the LogicalPlan
        // taxonomy.
        let names: &[(PlanTreeOp, &str)] = &[
            (PlanTreeOp::Scan, "Scan"),
            (PlanTreeOp::Expand, "Expand"),
            (PlanTreeOp::Filter, "Filter"),
            (PlanTreeOp::Project, "Project"),
            (PlanTreeOp::Join, "Join"),
            (PlanTreeOp::LeftOuterJoin, "LeftOuterJoin"),
            (PlanTreeOp::Limit, "Limit"),
            (PlanTreeOp::Skip, "Skip"),
            (PlanTreeOp::RankByHybrid, "RankByHybrid"),
            (PlanTreeOp::Fusion, "Fusion"),
            (PlanTreeOp::CommunityLookup, "CommunityLookup"),
            (PlanTreeOp::VectorNear, "VectorNear"),
            (PlanTreeOp::TextMatch, "TextMatch"),
            (PlanTreeOp::Aggregate, "Aggregate"),
            (PlanTreeOp::Sort, "Sort"),
            (PlanTreeOp::Distinct, "Distinct"),
            (PlanTreeOp::Unwind, "Unwind"),
            (PlanTreeOp::NamedPath, "NamedPath"),
            (PlanTreeOp::DynamicLimit, "DynamicLimit"),
            (PlanTreeOp::CreateNode, "CreateNode"),
            (PlanTreeOp::CreateVectorIndex, "CreateVectorIndex"),
            (PlanTreeOp::CreateRel, "CreateRel"),
            (PlanTreeOp::Delete, "Delete"),
            (PlanTreeOp::Set, "Set"),
            (PlanTreeOp::Remove, "Remove"),
            (PlanTreeOp::Merge, "Merge"),
            (PlanTreeOp::ProcedureCall, "ProcedureCall"),
            (PlanTreeOp::Empty, "Empty"),
        ];
        for (op, expected) in names {
            assert_eq!(op.name(), *expected, "op name drift for {op:?}");
        }
    }

    #[test]
    fn from_costed_plan_preserves_tree_shape_scan_filter_project() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 1_000);
        let scan = LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        };
        let filter = LogicalFilter {
            input: Box::new(LogicalPlan::Scan(scan)),
            predicate: lit_true(),
            span: span(),
        };
        let project = LogicalProject {
            input: Box::new(LogicalPlan::Filter(filter)),
            items: Vec::new(),
            span: span(),
        };
        let plan = LogicalPlan::Project(project);
        let costed = estimate_costs(plan, &cat);
        let pt = PlanTree::from_costed_plan(&costed);

        assert_eq!(pt.op, PlanTreeOp::Project);
        assert_eq!(pt.children.len(), 1);
        assert_eq!(pt.children[0].op, PlanTreeOp::Filter);
        assert_eq!(pt.children[0].children.len(), 1);
        assert_eq!(pt.children[0].children[0].op, PlanTreeOp::Scan);
        assert_eq!(pt.children[0].children[0].children.len(), 0);
        assert_eq!(pt.children[0].children[0].bindings, vec!["b0".to_string()]);
    }

    #[test]
    fn scan_carries_label_and_read_lsn_annotation() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(7)),
            var: BindingId::new(3),
            read_lsn: Lsn::MAX,
            span: span(),
        });
        let costed = estimate_costs(plan, &cat);
        let pt = PlanTree::from_costed_plan(&costed);
        assert_eq!(pt.op, PlanTreeOp::Scan);
        assert_eq!(pt.bindings, vec!["b3".to_string()]);
        assert_eq!(pt.annotations.get("label").map(String::as_str), Some("L7"));
        assert!(pt.annotations.contains_key("read_lsn"));
    }

    #[test]
    fn join_condition_renders_shared_bindings() {
        let cat = StubCatalogProvider::new();
        let l = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let r = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let plan = LogicalPlan::Join(LogicalJoin {
            left: Box::new(l),
            right: Box::new(r),
            on: JoinCondition::SharedBindings(vec![BindingId::new(0), BindingId::new(2)]),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        });
        let costed = estimate_costs(plan, &cat);
        let pt = PlanTree::from_costed_plan(&costed);
        assert_eq!(pt.op, PlanTreeOp::Join);
        assert_eq!(pt.bindings, vec!["b0".to_string(), "b2".to_string()]);
        assert_eq!(
            pt.annotations.get("condition").map(String::as_str),
            Some("shared:[b0,b2]"),
        );
    }

    #[test]
    fn cartesian_join_renders_as_cartesian() {
        let cat = StubCatalogProvider::new();
        let l = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let r = LogicalPlan::Empty(LogicalEmpty { span: span() });
        let plan = LogicalPlan::Join(LogicalJoin {
            left: Box::new(l),
            right: Box::new(r),
            on: JoinCondition::SharedBindings(Vec::new()),
            algorithm: JoinAlgorithm::Auto,
            span: span(),
        });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        assert_eq!(
            pt.annotations.get("condition").map(String::as_str),
            Some("Cartesian"),
        );
        assert!(pt.bindings.is_empty());
    }

    #[test]
    fn estimated_cost_and_card_match_costed_plan_root() {
        let cat = StubCatalogProvider::new()
            .with_total_node_count(10_000)
            .with_label_cardinality(LabelId::new(1), 1_000);
        let plan = LogicalPlan::Scan(LogicalScan {
            label: Some(LabelId::new(1)),
            var: BindingId::new(0),
            read_lsn: Lsn::MAX,
            span: span(),
        });
        let costed = estimate_costs(plan, &cat);
        let expected_cost = costed.total_cost();
        let expected_card = costed.output_card();
        let pt = PlanTree::from_costed_plan(&costed);
        assert_eq!(pt.estimated_cost.total(), expected_cost.total());
        assert_eq!(pt.estimated_card.rows(), expected_card.rows());
    }

    #[test]
    fn annotations_are_btreemap_so_iteration_order_is_stable() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Limit(LogicalLimit {
            input: Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() })),
            count: 10,
            span: span(),
        });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        let keys: Vec<&str> = pt.annotations.keys().map(String::as_str).collect();
        assert_eq!(keys, vec!["count"]);
    }

    #[test]
    fn fusion_carries_kind_k_and_input_count() {
        let cat = StubCatalogProvider::new();
        let plan = LogicalPlan::Fusion(LogicalFusion {
            spec: FusionSpec {
                kind: FusionKind::Rrf,
                k: 60,
                span: span(),
            },
            inputs: vec![Box::new(LogicalPlan::Empty(LogicalEmpty { span: span() }))],
            span: span(),
        });
        let pt = PlanTree::from_costed_plan(&estimate_costs(plan, &cat));
        assert_eq!(pt.op, PlanTreeOp::Fusion);
        assert_eq!(pt.annotations.get("kind").map(String::as_str), Some("RRF"));
        assert_eq!(pt.annotations.get("k").map(String::as_str), Some("60"));
        assert_eq!(
            pt.annotations.get("input_count").map(String::as_str),
            Some("1"),
        );
        assert_eq!(pt.children.len(), 1);
        assert_eq!(pt.children[0].op, PlanTreeOp::Empty);
    }
}
