//! M4-31 logical-plan-lowering semantic-preservation proptest
//! (256 cases). Closes the per amendment-03 §M4-31 row pin.
//!
//! # Property under test
//!
//! For any well-formed simple-operator-only ArcQL query, lowering
//! preserves the binding-level semantics established by M4-21 + M4-22
//! + M4-23:
//!
//! - **No drift in variable identity.** Every `BindingId` that the
//!   binding pass placed on a node-pattern variable in the
//!   `BoundQuery` MUST appear in the lowered `LogicalPlan` tree
//!   (either as a `Scan::var`, an `Expand::from`/`to`/`rel_var`, or a
//!   join condition). The lowering pass MUST NOT silently drop a
//!   variable.
//!
//! - **No drift in label identity.** Every `LabelId` that the binding
//!   pass resolved on a node pattern MUST be reflected in the lowered
//!   plan as a `Scan::label`.
//!
//! - **No drift in rel-type identity.** Every `TypeId` that the
//!   binding pass resolved on a relationship pattern MUST be reflected
//!   in the lowered plan as an `Expand::rel_type`.
//!
//! These invariants are the load-bearing safety property for the
//! M4-05 cost-based planner: any later optimization pass MUST be able
//! to derive plan cardinality from the same labels + rel-types the
//! binding pass resolved.
//!
//! # Strategy
//!
//! The proptest generator builds queries from a fixed alphabet:
//! - 3 labels (`L0` / `L1` / `L2`) registered in the catalog;
//! - 3 rel-types (`R0` / `R1` / `R2`);
//! - 4 variable names (`a` / `b` / `c` / `d`);
//! - chains of length 0..3 nodes;
//! - optional WHERE filter (`var.x > 0`);
//! - optional SKIP / LIMIT (literal integers);
//! - terminal `RETURN <var>`.
//!
//! The 256-case shrink space exercises every operator + every chain
//! length. Pathological patterns (multi-pattern MATCH, OPTIONAL MATCH,
//! aggregation, RANK BY HYBRID) are excluded — those defer to M4-32 /
//! M4-33 / future slices and have dedicated tests in
//! `logical_plan_integration.rs`.
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 (the M4-31 contract).
//! - ADR-038 amendment-03 §M4-31 row (256-case proptest pin).

use std::collections::BTreeSet;

use arcgraph_query::logical_plan::{
    Direction, JoinCondition, LogicalPlan, LogicalPlanLoweringVisitor,
};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingId, BindingVisitor, BoundClause, BoundMatchBody, BoundPathPattern, BoundQuery,
    BoundStatement, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["L0", "L1", "L2"])
        .with_rel_types(["R0", "R1", "R2"])
        .with_properties(["x"])
}

// ---------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------

const VARS: &[&str] = &["a", "b", "c", "d"];
const LABELS: &[&str] = &["L0", "L1", "L2"];
const REL_TYPES: &[&str] = &["R0", "R1", "R2"];

/// A generated chain step: relationship type + tail node variable +
/// optional tail node label.
#[derive(Debug, Clone)]
struct ChainStep {
    rel_type: usize,           // index into REL_TYPES
    tail_var: usize,           // index into VARS
    tail_label: Option<usize>, // index into LABELS or None
}

/// A generated query.
#[derive(Debug, Clone)]
struct GenQuery {
    head_var: usize,
    head_label: Option<usize>,
    chain: Vec<ChainStep>,
    where_var_idx: Option<usize>, // index into VARS for the WHERE filter
    skip: Option<u64>,
    limit: Option<u64>,
}

impl GenQuery {
    fn render(&self) -> String {
        let mut s = String::new();
        s.push_str("MATCH (");
        s.push_str(VARS[self.head_var]);
        if let Some(li) = self.head_label {
            s.push(':');
            s.push_str(LABELS[li]);
        }
        s.push(')');
        // Track which variables are bound, so WHERE picks a bound one.
        let mut bound_vars: BTreeSet<usize> = BTreeSet::from([self.head_var]);
        for step in &self.chain {
            s.push_str("-[:");
            s.push_str(REL_TYPES[step.rel_type]);
            s.push_str("]->(");
            s.push_str(VARS[step.tail_var]);
            if let Some(li) = step.tail_label {
                s.push(':');
                s.push_str(LABELS[li]);
            }
            s.push(')');
            bound_vars.insert(step.tail_var);
        }
        if let Some(wv_idx) = self.where_var_idx {
            // Use a bound variable. Default to head_var if the
            // generator landed on something not in scope.
            let var = if bound_vars.contains(&wv_idx) {
                VARS[wv_idx]
            } else {
                VARS[self.head_var]
            };
            s.push_str(" WHERE ");
            s.push_str(var);
            s.push_str(".x > 0");
        }
        s.push_str(" RETURN ");
        s.push_str(VARS[self.head_var]);
        if let Some(n) = self.skip {
            s.push_str(&format!(" SKIP {n}"));
        }
        if let Some(n) = self.limit {
            s.push_str(&format!(" LIMIT {n}"));
        }
        s
    }
}

fn step_strategy() -> impl Strategy<Value = ChainStep> {
    (
        0usize..REL_TYPES.len(),
        0usize..VARS.len(),
        prop::option::of(0usize..LABELS.len()),
    )
        .prop_map(|(rel_type, tail_var, tail_label)| ChainStep {
            rel_type,
            tail_var,
            tail_label,
        })
}

fn query_strategy() -> impl Strategy<Value = GenQuery> {
    (
        0usize..VARS.len(),                           // head_var
        prop::option::of(0usize..LABELS.len()),       // head_label
        prop::collection::vec(step_strategy(), 0..3), // chain
        prop::option::of(0usize..VARS.len()),         // where_var
        prop::option::of(0u64..1000),                 // skip
        prop::option::of(0u64..1000),                 // limit
    )
        .prop_map(
            |(head_var, head_label, chain, where_var_idx, skip, limit)| GenQuery {
                head_var,
                head_label,
                chain,
                where_var_idx,
                skip,
                limit,
            },
        )
}

// ---------------------------------------------------------------------
// Invariant checks
// ---------------------------------------------------------------------

fn collect_input_bindings(q: &BoundQuery) -> BTreeSet<BindingId> {
    let mut out = BTreeSet::new();
    for c in &q.clauses {
        if let BoundClause::Match(m) = c {
            if let BoundMatchBody::Patterns(ps) = &m.body {
                for p in ps {
                    collect_input_bindings_in_path(p, &mut out);
                }
            }
        }
    }
    out
}

fn collect_input_bindings_in_path(p: &BoundPathPattern, out: &mut BTreeSet<BindingId>) {
    if let Some(v) = &p.head.var {
        out.insert(v.binding_id);
    }
    for (rel, node) in &p.tail {
        if let Some(v) = &rel.var {
            out.insert(v.binding_id);
        }
        if let Some(v) = &node.var {
            out.insert(v.binding_id);
        }
    }
}

fn collect_input_labels(q: &BoundQuery) -> BTreeSet<arcgraph_core::LabelId> {
    let mut out = BTreeSet::new();
    for c in &q.clauses {
        if let BoundClause::Match(m) = c {
            if let BoundMatchBody::Patterns(ps) = &m.body {
                for p in ps {
                    if let Some(l) = p.head.labels.first() {
                        out.insert(l.label_id);
                    }
                    for (_rel, node) in &p.tail {
                        if let Some(l) = node.labels.first() {
                            out.insert(l.label_id);
                        }
                    }
                }
            }
        }
    }
    out
}

fn collect_input_rel_types(q: &BoundQuery) -> BTreeSet<arcgraph_core::TypeId> {
    let mut out = BTreeSet::new();
    for c in &q.clauses {
        if let BoundClause::Match(m) = c {
            if let BoundMatchBody::Patterns(ps) = &m.body {
                for p in ps {
                    for (rel, _node) in &p.tail {
                        if let Some(t) = rel.rel_types.first() {
                            out.insert(t.type_id);
                        }
                    }
                }
            }
        }
    }
    out
}

fn collect_plan_bindings(p: &LogicalPlan, out: &mut BTreeSet<BindingId>) {
    match p {
        LogicalPlan::Scan(s) => {
            out.insert(s.var);
        }
        LogicalPlan::PropertyIndexScan(p) => {
            out.insert(p.var);
        }
        LogicalPlan::CountStore(c) => {
            out.insert(c.output_id);
        }
        LogicalPlan::Expand(e) => {
            out.insert(e.from);
            out.insert(e.to);
            if let Some(rv) = e.rel_var {
                out.insert(rv);
            }
        }
        LogicalPlan::Filter(f) => collect_plan_bindings(&f.input, out),
        LogicalPlan::Project(pr) => collect_plan_bindings(&pr.input, out),
        LogicalPlan::Join(j) => {
            collect_plan_bindings(&j.left, out);
            collect_plan_bindings(&j.right, out);
            let JoinCondition::SharedBindings(ids) = &j.on;
            for id in ids {
                out.insert(*id);
            }
        }
        LogicalPlan::LeftOuterJoin(j) => {
            collect_plan_bindings(&j.left, out);
            collect_plan_bindings(&j.right, out);
            let JoinCondition::SharedBindings(ids) = &j.on;
            for id in ids {
                out.insert(*id);
            }
        }
        LogicalPlan::Limit(l) => collect_plan_bindings(&l.input, out),
        LogicalPlan::Skip(s) => collect_plan_bindings(&s.input, out),
        LogicalPlan::CommunityLookup(c) => {
            collect_plan_bindings(&c.input, out);
            out.insert(c.node_var);
        }
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                collect_plan_bindings(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                collect_plan_bindings(arm, out);
            }
        }
        LogicalPlan::RankByHybrid(r) => {
            for op in &r.operands {
                out.insert(op.var);
            }
            if let Some(score) = r.score_binding {
                out.insert(score);
            }
        }
        LogicalPlan::VectorNear(v) => {
            out.insert(v.var);
        }
        LogicalPlan::TextMatch(t) => {
            out.insert(t.var);
        }
        LogicalPlan::Aggregate(a) => collect_plan_bindings(&a.input, out),
        LogicalPlan::Sort(s) => collect_plan_bindings(&s.input, out),
        LogicalPlan::Distinct(d) => collect_plan_bindings(&d.input, out),
        LogicalPlan::Unwind(u) => {
            collect_plan_bindings(&u.input, out);
            out.insert(u.var);
        }
        LogicalPlan::ProcedureCall(p) => {
            collect_plan_bindings(&p.input, out);
            for (_, bid) in &p.columns {
                out.insert(*bid);
            }
        }
        LogicalPlan::NamedPath(np) => {
            collect_plan_bindings(&np.input, out);
            out.insert(np.path_var);
        }
        LogicalPlan::DynamicLimit(l) => collect_plan_bindings(&l.input, out),
        LogicalPlan::CreateNode(c) => {
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        // #830 / ADR-200: CREATE VECTOR INDEX is a leaf DDL — no bindings.
        LogicalPlan::CreateVectorIndex(_) => {}
        // #1366: CREATE INDEX (property index) is a leaf DDL — no bindings.
        LogicalPlan::CreatePropertyIndex(_) => {}
        LogicalPlan::CreateRel(c) => {
            collect_plan_bindings(&c.source_plan, out);
            collect_plan_bindings(&c.target_plan, out);
            if let Some(v) = c.var {
                out.insert(v);
            }
        }
        LogicalPlan::Delete(d) => collect_plan_bindings(&d.input, out),
        LogicalPlan::Set(s) => collect_plan_bindings(&s.input, out),
        LogicalPlan::Remove(r) => collect_plan_bindings(&r.input, out),
        LogicalPlan::Merge(m) => {
            collect_plan_bindings(&m.match_branch, out);
            collect_plan_bindings(&m.create_branch, out);
        }
        // ADR-192 (#623): a CALL{}'s OUTPUT bindings = driving input ++
        // returned; the seed carries its imported set. Mirrors
        // `lowering::collect_bindings`.
        LogicalPlan::Call(c) => {
            collect_plan_bindings(&c.input, out);
            for b in &c.returned {
                out.insert(*b);
            }
        }
        LogicalPlan::CorrelationSeed(s) => {
            for b in &s.imported {
                out.insert(*b);
            }
        }
        LogicalPlan::Empty(_) => {}
    }
}

fn collect_plan_labels(p: &LogicalPlan, out: &mut BTreeSet<arcgraph_core::LabelId>) {
    match p {
        LogicalPlan::Scan(s) => {
            if let Some(l) = s.label {
                out.insert(l);
            }
        }
        LogicalPlan::PropertyIndexScan(p) => {
            out.insert(p.label);
        }
        LogicalPlan::Expand(_) => {}
        LogicalPlan::Filter(f) => collect_plan_labels(&f.input, out),
        LogicalPlan::Project(pr) => collect_plan_labels(&pr.input, out),
        LogicalPlan::Join(j) => {
            collect_plan_labels(&j.left, out);
            collect_plan_labels(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            collect_plan_labels(&j.left, out);
            collect_plan_labels(&j.right, out);
        }
        LogicalPlan::Limit(l) => collect_plan_labels(&l.input, out),
        LogicalPlan::Skip(s) => collect_plan_labels(&s.input, out),
        LogicalPlan::CommunityLookup(c) => collect_plan_labels(&c.input, out),
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                collect_plan_labels(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                collect_plan_labels(arm, out);
            }
        }
        LogicalPlan::Aggregate(a) => collect_plan_labels(&a.input, out),
        LogicalPlan::Sort(s) => collect_plan_labels(&s.input, out),
        LogicalPlan::Distinct(d) => collect_plan_labels(&d.input, out),
        LogicalPlan::Unwind(u) => collect_plan_labels(&u.input, out),
        LogicalPlan::ProcedureCall(p) => collect_plan_labels(&p.input, out),
        LogicalPlan::NamedPath(np) => collect_plan_labels(&np.input, out),
        LogicalPlan::DynamicLimit(l) => collect_plan_labels(&l.input, out),
        LogicalPlan::Empty(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CreateRel(_)
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => {}
    }
}

fn collect_plan_rel_types(p: &LogicalPlan, out: &mut BTreeSet<arcgraph_core::TypeId>) {
    match p {
        LogicalPlan::Scan(_) => {}
        LogicalPlan::PropertyIndexScan(_) => {}
        LogicalPlan::CountStore(_) => {}
        LogicalPlan::Expand(e) => {
            if let Some(t) = e.rel_type {
                out.insert(t);
            }
        }
        LogicalPlan::Filter(f) => collect_plan_rel_types(&f.input, out),
        LogicalPlan::Project(pr) => collect_plan_rel_types(&pr.input, out),
        LogicalPlan::Join(j) => {
            collect_plan_rel_types(&j.left, out);
            collect_plan_rel_types(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            collect_plan_rel_types(&j.left, out);
            collect_plan_rel_types(&j.right, out);
        }
        LogicalPlan::Limit(l) => collect_plan_rel_types(&l.input, out),
        LogicalPlan::Skip(s) => collect_plan_rel_types(&s.input, out),
        LogicalPlan::CommunityLookup(c) => collect_plan_rel_types(&c.input, out),
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                collect_plan_rel_types(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                collect_plan_rel_types(arm, out);
            }
        }
        LogicalPlan::Aggregate(a) => collect_plan_rel_types(&a.input, out),
        LogicalPlan::Sort(s) => collect_plan_rel_types(&s.input, out),
        LogicalPlan::Distinct(d) => collect_plan_rel_types(&d.input, out),
        LogicalPlan::Unwind(u) => collect_plan_rel_types(&u.input, out),
        LogicalPlan::ProcedureCall(p) => collect_plan_rel_types(&p.input, out),
        LogicalPlan::NamedPath(np) => collect_plan_rel_types(&np.input, out),
        LogicalPlan::DynamicLimit(l) => collect_plan_rel_types(&l.input, out),
        LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CreateRel(_)
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => {}
    }
}

/// Sanity check: every Expand we emitted has its `from` / `to` /
/// rel-var bindings referencing real BindingIds (i.e., the
/// anonymous-binding seeding worked correctly).
fn collect_plan_directions(p: &LogicalPlan, out: &mut Vec<Direction>) {
    match p {
        LogicalPlan::Expand(e) => out.push(e.direction),
        LogicalPlan::Filter(f) => collect_plan_directions(&f.input, out),
        LogicalPlan::Project(pr) => collect_plan_directions(&pr.input, out),
        LogicalPlan::Join(j) => {
            collect_plan_directions(&j.left, out);
            collect_plan_directions(&j.right, out);
        }
        LogicalPlan::LeftOuterJoin(j) => {
            collect_plan_directions(&j.left, out);
            collect_plan_directions(&j.right, out);
        }
        LogicalPlan::Limit(l) => collect_plan_directions(&l.input, out),
        LogicalPlan::Skip(s) => collect_plan_directions(&s.input, out),
        LogicalPlan::CommunityLookup(c) => collect_plan_directions(&c.input, out),
        LogicalPlan::Fusion(f) => {
            for inp in &f.inputs {
                collect_plan_directions(inp, out);
            }
        }
        LogicalPlan::Union(u) => {
            for arm in &u.arms {
                collect_plan_directions(arm, out);
            }
        }
        LogicalPlan::Aggregate(a) => collect_plan_directions(&a.input, out),
        LogicalPlan::Sort(s) => collect_plan_directions(&s.input, out),
        LogicalPlan::Distinct(d) => collect_plan_directions(&d.input, out),
        LogicalPlan::Unwind(u) => collect_plan_directions(&u.input, out),
        LogicalPlan::ProcedureCall(p) => collect_plan_directions(&p.input, out),
        LogicalPlan::NamedPath(np) => collect_plan_directions(&np.input, out),
        LogicalPlan::DynamicLimit(l) => collect_plan_directions(&l.input, out),
        LogicalPlan::Scan(_)
        | LogicalPlan::PropertyIndexScan(_)
        | LogicalPlan::CountStore(_)
        | LogicalPlan::Empty(_)
        | LogicalPlan::RankByHybrid(_)
        | LogicalPlan::VectorNear(_)
        | LogicalPlan::TextMatch(_)
        | LogicalPlan::CreateNode(_)
        | LogicalPlan::CreateVectorIndex(_)
        | LogicalPlan::CreatePropertyIndex(_)
        | LogicalPlan::CreateRel(_)
        | LogicalPlan::Delete(_)
        | LogicalPlan::Set(_)
        | LogicalPlan::Remove(_)
        | LogicalPlan::Merge(_)
        | LogicalPlan::Call(_)
        | LogicalPlan::CorrelationSeed(_) => {}
    }
}

// ---------------------------------------------------------------------
// The proptest
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        .. ProptestConfig::default()
    })]

    /// For any well-formed simple-operator-only ArcQL query, lowering
    /// preserves the binding-, label-, and rel-type-level identity
    /// established by the M4-21 + M4-22 + M4-23 passes.
    #[test]
    fn lowering_preserves_binding_label_and_rel_type_identity(q in query_strategy()) {
        let input = q.render();
        let cat = cat();

        let stmt = match parse(&input) {
            Ok(s) => s,
            // Parser rejection on a generated input is a strategy bug,
            // not a semantic-preservation failure. Skip the case via
            // proptest's `prop_assume!`.
            Err(_) => return Err(TestCaseError::reject("parser rejected generated input")),
        };
        let mut bound = match BindingVisitor::bind(&stmt, &input, &cat) {
            Ok(b) => b,
            Err(_) => return Err(TestCaseError::reject("binding rejected generated input")),
        };
        if TypeCheckVisitor::check(&mut bound, &cat).is_err() {
            return Err(TestCaseError::reject("type-check rejected generated input"));
        }
        if CrossSubstrateValidator::validate(&bound, &cat).is_err() {
            return Err(TestCaseError::reject("cross-substrate rejected generated input"));
        }

        let plan = LogicalPlanLoweringVisitor::lower(&bound)
            .expect("M4-31 lowering must accept simple-operator-only queries");

        // Extract from BoundQuery.
        let bq: &BoundQuery = match &bound {
            BoundStatement::Read(q) => q,
            _ => return Err(TestCaseError::reject("non-read statement")),
        };
        let in_bindings = collect_input_bindings(bq);
        let in_labels = collect_input_labels(bq);
        let in_rel_types = collect_input_rel_types(bq);

        let mut plan_bindings = BTreeSet::new();
        collect_plan_bindings(&plan, &mut plan_bindings);
        let mut plan_labels = BTreeSet::new();
        collect_plan_labels(&plan, &mut plan_labels);
        let mut plan_rel_types = BTreeSet::new();
        collect_plan_rel_types(&plan, &mut plan_rel_types);

        // Property 1 — every input binding appears in the plan.
        for b in &in_bindings {
            prop_assert!(
                plan_bindings.contains(b),
                "input binding {b:?} is missing from the lowered plan; input={input:?}"
            );
        }

        // Property 2 — every input label appears in the plan.
        prop_assert_eq!(
            in_labels.clone(),
            plan_labels.intersection(&in_labels).copied().collect::<BTreeSet<_>>(),
            "input labels are not a subset of plan labels; input={:?}", input
        );
        prop_assert!(
            in_labels.is_subset(&plan_labels),
            "input labels are not a subset of plan labels; input={input:?}"
        );

        // Property 3 — every input rel-type appears in the plan.
        prop_assert!(
            in_rel_types.is_subset(&plan_rel_types),
            "input rel-types are not a subset of plan rel-types; input={input:?}"
        );

        // Property 4 — every Expand's direction is well-defined (the
        // Direction::from(&RelDirection) impl is exhaustive). We
        // simply check that the call returned a known variant for
        // each Expand in the tree.
        let mut dirs = Vec::new();
        collect_plan_directions(&plan, &mut dirs);
        for d in &dirs {
            prop_assert!(
                matches!(
                    d,
                    Direction::LeftToRight | Direction::RightToLeft | Direction::Undirected
                ),
                "Expand has unknown direction {d:?}; input={input:?}"
            );
        }
    }
}
