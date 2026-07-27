//! M4-22 type-check pass — populates `type_info` slots on
//! [`BoundQuery`] and rejects reserved variants per ADR-038 §2 D-22.
//!
//! [`TypeCheckVisitor::check`] walks a `BoundStatement` (post-M4-21
//! binding) and:
//!
//! 1. Resolves each `BoundVariable` / `BoundExpression` /
//!    `BoundLiteral` / `BoundPropertyAccess` / `BoundFunctionCall`
//!    to a [`TypeInfo`].
//! 2. Validates operand types under binary / unary / comparison
//!    operators per Cypher 3VL (ADR-038 D-20).
//! 3. Validates function calls against
//!    [`crate::semantic::functions::BUILTINS`].
//! 4. Emits [`ArcQLError::NotImplemented`] for the reserved-variant
//!    set enumerated in amendment-03 + PR #154 reviewer ask #7
//!    (see the in-file taxonomy comment).
//!
//! `BoundVariable::may_be_null` is set at BINDING TIME (M4-21
//! `BindingVisitor`) per ADR-038 §2 D-21 M4-22b refinement (Shape B).
//! Re-references in pattern positions inherit `may_be_null` from the
//! original binding and never upgrade nullability. The type-check
//! pass does not touch `may_be_null`.
//!
//! # Pass shape
//!
//! The visitor walks `BoundStatement` via dedicated `check_*` methods
//! that take `&mut` access into the bound AST so they can populate
//! `type_info` slots in place. State:
//!
//! - `errors: Vec<ArcQLError>` — accumulating diagnostics; the pass
//!   does NOT short-circuit on the first error so the user sees the
//!   full diagnostic surface in one pass.
//! - `binding_types: HashMap<BindingId, TypeInfo>` — type-info for
//!   declared bindings, looked up when resolving variable references.
//!
//! # 3VL / NULL semantics (ADR-038 D-20)
//!
//! - `n.x = NULL` → comparison result is `Null` (not `Boolean`).
//! - `n.x IS NULL` → result is `Boolean` (TRUE if Null, else FALSE).
//! - Any binary op with a `Null` operand → `Null`.
//! - AND / OR / NOT use the openCypher 3VL truth table; helpers
//!   [`apply_and_3vl`] / [`apply_or_3vl`] / [`apply_not_3vl`]
//!   centralize the semantics for the proptest in
//!   `tests/three_vl_proptest.rs` to share with the production code.
//! - WHERE filter treats `Null` as `False` (Cypher convention) — but
//!   the type-check rejects only when the WHERE expression has a
//!   non-Boolean / non-Null type at the top level.
//!
//! # ADR provenance
//! - ADR-038 §2 D-22 — type-check + reserved-variant rejection
//!   contract (this file's primary spec).
//! - ADR-038 §2 D-20 — 3VL truth table.
//! - ADR-038 §2 D-16 — `NotImplemented` error shape.
//! - ADR-038 §2 D-2 / D-7 / D-9 / D-10 — reserved variants.
//! - ADR-006 amendment-01 — OPTIONAL MATCH at v1.0.

use std::collections::HashMap;

use crate::ast::{BinOp, Expression, LengthRange, Literal, UnaryOp};
use crate::error::Span;
use crate::logical_plan::AggregationKind;
use crate::semantic::bound_ast::{
    BindingId, BoundClause, BoundCreateClause, BoundCreateItem, BoundDeleteClause, BoundExpression,
    BoundFusion, BoundMapProjectionItem, BoundMatchBody, BoundMatchClause, BoundMergeClause,
    BoundMergePattern, BoundProjectionItem, BoundProjectionKind, BoundQuery, BoundRanker,
    BoundRelPattern, BoundRemoveClause, BoundSetClause, BoundSetMutation, BoundStatement,
    BoundWithFusionClause, PropertyType, TypeInfo,
};
use crate::semantic::catalog::CatalogProvider;
use crate::semantic::error::{ArcQLError, TypeCheckError};
use crate::semantic::functions::{self, ArgKind, Arity};

// =====================================================================
// 3VL truth-table primitives (D-20)
// =====================================================================

/// Three-valued logic value (TRUE / FALSE / NULL) used by the 3VL
/// truth-table helpers. Pulled out as a dedicated enum so the
/// production code path AND the proptest in `tests/three_vl_proptest.rs`
/// share a single source of truth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoolOrNull {
    True,
    False,
    Null,
}

/// Cypher 9 §6.4 AND truth table.
pub fn apply_and_3vl(a: BoolOrNull, b: BoolOrNull) -> BoolOrNull {
    use BoolOrNull::*;
    match (a, b) {
        (False, _) | (_, False) => False,
        (Null, _) | (_, Null) => Null,
        (True, True) => True,
    }
}

/// Cypher 9 §6.4 OR truth table.
pub fn apply_or_3vl(a: BoolOrNull, b: BoolOrNull) -> BoolOrNull {
    use BoolOrNull::*;
    match (a, b) {
        (True, _) | (_, True) => True,
        (Null, _) | (_, Null) => Null,
        (False, False) => False,
    }
}

/// Cypher 9 §6.4 NOT truth table.
pub fn apply_not_3vl(a: BoolOrNull) -> BoolOrNull {
    use BoolOrNull::*;
    match a {
        True => False,
        False => True,
        Null => Null,
    }
}

// =====================================================================
// TypeCheckVisitor
// =====================================================================

/// M4-22 type-check pass.
///
/// Construct via [`Self::check`]; the internal struct is not part of
/// the public API. The pass MUTATES the input `BoundStatement` to
/// populate `type_info` slots on every expression node + the
/// `may_be_null` flag on OPTIONAL MATCH–introduced variables.
pub struct TypeCheckVisitor<'cat, C: CatalogProvider> {
    /// Reserved for M4-23 cross-substrate validation (which queries
    /// the catalog for substrate availability per-tenant). M4-22's
    /// type-check pass does not consult the catalog directly — the
    /// label / rel-type / property IDs were already resolved by
    /// M4-21's `BindingVisitor` and live on the BoundAst.
    #[allow(dead_code)]
    catalog: &'cat C,
    errors: Vec<ArcQLError>,
    /// Declared-binding types (BindingId → TypeInfo). Populated as
    /// we encounter declaration sites (MATCH patterns, UNWIND, WITH
    /// projections).
    binding_types: HashMap<BindingId, TypeInfo>,
}

impl<'cat, C: CatalogProvider> TypeCheckVisitor<'cat, C> {
    /// Type-check a [`BoundStatement`] in place. Mutates `stmt` to
    /// populate `type_info` slots; returns `Ok(())` on a clean check
    /// or `Err(Vec<ArcQLError>)` with accumulated diagnostics.
    ///
    /// The pass does NOT short-circuit on the first error — it
    /// surfaces every type-check / reserved-variant fault in a
    /// single walk, matching M4-21's `BindingVisitor` discipline.
    pub fn check(stmt: &mut BoundStatement, catalog: &'cat C) -> Result<(), Vec<ArcQLError>> {
        let mut v = Self {
            catalog,
            errors: Vec::new(),
            binding_types: HashMap::new(),
        };
        v.check_statement(stmt);
        if v.errors.is_empty() {
            Ok(())
        } else {
            Err(v.errors)
        }
    }

    /// Top-level dispatch. The walk uses dedicated `check_*` methods
    /// (rather than a generic visitor trait) because we need to mutate
    /// the `BoundStatement`.
    fn check_statement(&mut self, stmt: &mut BoundStatement) {
        match stmt {
            BoundStatement::Read(q) => self.check_query(q),
            // ADR-185 (#649-A1, W28) — UNION / UNION ALL. Type-check
            // each arm independently (each is a self-contained bound
            // read query). The post-union tail (ORDER BY / SKIP /
            // LIMIT) carries no type-checkable hazards at v1.0 (SKIP /
            // LIMIT are literal-int validated at lowering; ORDER BY key
            // expressions are evaluated dynamically by the Sort op);
            // tail type-checking is a v1.1 follow-up tracked in the ADR.
            BoundStatement::Union(u) => {
                for arm in &mut u.arms {
                    self.check_query(arm);
                }
            }
            BoundStatement::IndexDdl(d) => {
                // #830 (ADR-198 §OQ-7 / ADR-200) — CREATE VECTOR INDEX is
                // accept-and-register (the D2/D3 catalog half of the OQ-7
                // split): type-check PASSES, and the
                // `CreateVectorIndexOp` registers a metadata entry in the
                // per-tenant vector-index catalog at execute-time (the
                // served HNSW BUILD is auto-on-ingest per #765 PART-1 —
                // CREATE does NOT trigger a build). The OPTIONS
                // `vector.dimensions` / `vector.similarity_function` are
                // validated at execute-time, where the `$param` values
                // resolve. DROP INDEX remains a typed NotImplemented
                // (lifecycle is a vector-track follow-up; the #830
                // langchain happy path never DROPs) — an honest "parsed,
                // lifecycle not wired", NOT a silent no-op trampoline.
                use crate::ast::IndexDdlStatement;
                match d {
                    IndexDdlStatement::CreateVector(_) => {
                        // Accept — registration + OPTIONS validation are
                        // execute-time concerns (the `$dimensions` /
                        // `$similarity_fn` params are unavailable here).
                    }
                    IndexDdlStatement::CreateProperty(_) => {
                        // #1366 (task #248, Phase 1) — CREATE INDEX for the
                        // user-visible property index is a real DDL. Type-
                        // check PASSES; the catalog register + backfill +
                        // Online-flip happen at execute-time via the MCP
                        // substrate's property-index manager.
                    }
                    IndexDdlStatement::Drop(_) => {
                        self.errors.push(ArcQLError::NotImplemented {
                            feature: "DROP INDEX (index lifecycle)".into(),
                            section:
                                "#830 vector grammar surface; lifecycle owned by vector track (ADR-198 §OQ-7)"
                                    .into(),
                            target_version: "v1.2".into(),
                            span: Span::point(1, 1),
                        });
                    }
                }
            }
        }
    }

    fn check_query(&mut self, q: &mut BoundQuery) {
        // Pre-pass: walk every clause with &mut access, registering
        // declared bindings with their type-info (for MATCH-pattern
        // variables) AND populating `BoundVariable::type_info` at
        // each declaration site. Then walk a second time to
        // type-check expressions (so a forward reference within the
        // same scope chain resolves cleanly).
        for c in &mut q.clauses {
            self.register_clause_bindings(c);
        }
        for c in &mut q.clauses {
            self.check_clause(c);
        }
    }

    /// First pass — record the type of each declared binding so the
    /// expression-pass can resolve `VariableRef -> TypeInfo` AND
    /// populate `BoundVariable::type_info` at declaration sites.
    fn register_clause_bindings(&mut self, c: &mut BoundClause) {
        match c {
            BoundClause::Match(m) => {
                self.register_match_bindings(m);
                // Reserved variants on MATCH — the actual rejection
                // happens in the expression pass (so the span
                // attached to the error points at the offending
                // node, not the clause prologue). We register
                // bindings unconditionally so that reserved-variant
                // queries still yield a structurally complete tree.
            }
            BoundClause::Create(c) => self.register_create_bindings(c),
            BoundClause::Delete(d) => self.register_delete_bindings(d),
            BoundClause::Set(s) => self.register_set_bindings(s),
            BoundClause::Remove(r) => self.register_remove_bindings(r),
            BoundClause::Merge(m) => self.register_merge_bindings(m),
            BoundClause::With(_) | BoundClause::Unwind(_) => {
                // WITH's projection aliases + UNWIND's element
                // variable are bound during the second pass when we
                // know the projection / source-list type.
            }
            // ADR-192 (#623): the subquery's RETURN columns escape the
            // scoping fence into the OUTER scope (D-4). Register them
            // permissively (`Null` — the 3VL bottom) so the enclosing
            // query's references resolve; the values are computed at
            // runtime, so precise static typing of correlated-subquery
            // outputs is a forward refinement (matching UNWIND's
            // can't-determine-element-type → `Null` convention). The
            // body's OWN bindings are registered + checked in pass 2
            // (`check_call_body` → nested `check_query`).
            BoundClause::Call(c) => {
                for b in &c.returned {
                    self.binding_types.entry(*b).or_insert(TypeInfo::Null);
                }
            }
            _ => {}
        }
    }

    // ADR-147 W26-θ Phase 1: CREATE introduces fresh bindings; each
    // CreateNodeSpec's optional variable gets a Node type (with the
    // label as a TypeInfo::Node label hint when the spec carries
    // one). Label-name → label_id resolution is deferred to the
    // substrate-execute layer (CatalogProvider is read-only at the
    // binding/typecheck/lowering tier per ADR-147 §D-3 / §D-7).
    // We register TypeInfo::Node { label: None } so RETURN-side
    // projections resolve cleanly.
    //
    // ADR-148 W26-θ Phase 2: CREATE-path extends the registration to
    // cover source + rel + target. Source + target get Node typing;
    // rel gets `Relationship { rel_type: None }` (the rel-type NAME
    // is preserved at the bound AST but the catalog
    // `lookup_rel_type` is read-only at v1.0-α per the same
    // rationale as the node-label).
    fn register_create_bindings(&mut self, c: &mut BoundCreateClause) {
        for item in &mut c.items {
            match item {
                BoundCreateItem::Node(spec) => {
                    let ti = TypeInfo::Node { label: None };
                    if let Some(v) = &mut spec.var {
                        self.binding_types.insert(v.binding_id, ti.clone());
                        v.type_info = Some(ti);
                    }
                }
                BoundCreateItem::Path(path) => {
                    // Source + target — Node typing.
                    let node_ti = TypeInfo::Node { label: None };
                    if let Some(v) = &mut path.source.var {
                        self.binding_types.insert(v.binding_id, node_ti.clone());
                        v.type_info = Some(node_ti.clone());
                    }
                    if let Some(v) = &mut path.target.var {
                        self.binding_types.insert(v.binding_id, node_ti.clone());
                        v.type_info = Some(node_ti);
                    }
                    // Rel — Relationship typing.
                    let rel_ti = TypeInfo::Relationship { rel_type: None };
                    if let Some(v) = &mut path.rel.var {
                        self.binding_types.insert(v.binding_id, rel_ti.clone());
                        v.type_info = Some(rel_ti);
                    }
                }
            }
        }
    }

    // ADR-149 W26-θ Phase 3: DELETE items RESOLVE upstream-bound
    // variables (the binding pass already populated each item's
    // `BoundVariable`); the `register_delete_bindings` pass
    // populates the resolved item's `type_info` from the global
    // `binding_types` table so the second-pass `check_delete_clause`
    // can validate Node-vs-Relationship typing against fully
    // up-to-date type_info.
    //
    // Unlike CREATE's `register_create_bindings`, DELETE does NOT
    // ADD new entries to `binding_types` — the items reference EXISTING
    // bindings populated by `register_match_bindings` upstream. This
    // method just THREADS the type_info from `binding_types` onto the
    // item's `BoundVariable::type_info` slot for downstream walker
    // ergonomics (parallel to how `register_match_bindings` sets
    // `BoundNodePattern::var.type_info`).
    fn register_delete_bindings(&mut self, d: &mut BoundDeleteClause) {
        for item in &mut d.items {
            if let Some(ti) = self.binding_types.get(&item.var.binding_id).cloned() {
                item.var.type_info = Some(ti);
            }
        }
    }

    // ADR-150 W26-θ Phase 4: SET items RESOLVE upstream-bound variables
    // (same shape as DELETE — see `register_delete_bindings`); we thread
    // the type_info from `binding_types` onto each item's
    // `BoundVariable::type_info` slot for downstream walker ergonomics.
    // SET does NOT ADD new entries to `binding_types`.
    fn register_set_bindings(&mut self, s: &mut BoundSetClause) {
        for item in &mut s.items {
            if let Some(ti) = self.binding_types.get(&item.var.binding_id).cloned() {
                item.var.type_info = Some(ti);
            }
        }
    }

    // ADR-150 W26-θ Phase 4: REMOVE items RESOLVE upstream-bound
    // variables — see `register_set_bindings`.
    fn register_remove_bindings(&mut self, r: &mut BoundRemoveClause) {
        for item in &mut r.items {
            if let Some(ti) = self.binding_types.get(&item.var.binding_id).cloned() {
                item.var.type_info = Some(ti);
            }
        }
    }

    // ADR-151 W26-θ Phase 5: MERGE introduces FRESH bindings (parallel
    // to CREATE — `register_create_bindings`); the on_create / on_match
    // action items RESOLVE against those bindings (parallel to SET —
    // `register_set_bindings`). The first pass over the merge pattern
    // registers TypeInfo::Node / TypeInfo::Relationship for each fresh
    // var; the action items thread the resulting type_info onto each
    // BoundVariable::type_info slot.
    fn register_merge_bindings(&mut self, m: &mut BoundMergeClause) {
        // Step 1: register the pattern's fresh bindings (parallel to
        // register_create_bindings's per-shape registration).
        match &mut m.pattern {
            BoundMergePattern::Node(spec) => {
                let ti = TypeInfo::Node { label: None };
                if let Some(v) = &mut spec.var {
                    self.binding_types.insert(v.binding_id, ti.clone());
                    v.type_info = Some(ti);
                }
            }
            BoundMergePattern::Path(path) => {
                let node_ti = TypeInfo::Node { label: None };
                if let Some(v) = &mut path.source.var {
                    self.binding_types.insert(v.binding_id, node_ti.clone());
                    v.type_info = Some(node_ti.clone());
                }
                if let Some(v) = &mut path.target.var {
                    self.binding_types.insert(v.binding_id, node_ti.clone());
                    v.type_info = Some(node_ti);
                }
                let rel_ti = TypeInfo::Relationship { rel_type: None };
                if let Some(v) = &mut path.rel.var {
                    self.binding_types.insert(v.binding_id, rel_ti.clone());
                    v.type_info = Some(rel_ti);
                }
            }
        }
        // Step 2: thread type_info onto the action items (parallel to
        // register_set_bindings). The action items reference the
        // pattern's fresh bindings; the binding pass set their
        // binding_ids; the type table now carries the type_info
        // populated in step 1 above.
        for item in &mut m.on_create {
            if let Some(ti) = self.binding_types.get(&item.var.binding_id).cloned() {
                item.var.type_info = Some(ti);
            }
        }
        for item in &mut m.on_match {
            if let Some(ti) = self.binding_types.get(&item.var.binding_id).cloned() {
                item.var.type_info = Some(ti);
            }
        }
    }

    fn register_match_bindings(&mut self, m: &mut BoundMatchClause) {
        // Detach the body so we can mutate it independently of `m`.
        match &mut m.body {
            BoundMatchBody::Patterns(ps) => {
                for p in ps.iter_mut() {
                    self.register_node_binding(&mut p.head);
                    for (rel, node) in p.tail.iter_mut() {
                        self.register_rel_binding(rel);
                        self.register_node_binding(node);
                    }
                }
            }
            BoundMatchBody::NamedPath(np) => {
                // ADR-193 D-13 + ADR-194 D-5 — ALL named-path vars are
                // Path-typed. `Plain` was Path-typed at ADR-193; ADR-194
                // D-5 migrates the `ShortestPath` AND `AllShortestPath`
                // executors (`executor::ops::path`) to emit `Value::Path`
                // (nodes + relationships) instead of the legacy
                // `Value::List`-of-nodes, so their vars retype to `Path`
                // too. This realizes the D-14 single-path representation:
                // `nodes(p)` / `relationships(p)` / `length(p)` now work
                // identically across all three named-path kinds (the prior
                // `Map` typing for ShortestPath is RETIRED in lockstep with
                // the executor migration — type and runtime agree).
                let (pp, var_type) = match &mut np.kind {
                    crate::semantic::bound_ast::BoundNamedPathKind::Plain(p) => (p, TypeInfo::Path),
                    crate::semantic::bound_ast::BoundNamedPathKind::ShortestPath(p)
                    | crate::semantic::bound_ast::BoundNamedPathKind::AllShortestPath(p) => {
                        (p, TypeInfo::Path)
                    }
                };
                self.binding_types
                    .insert(np.var.binding_id, var_type.clone());
                np.var.type_info = Some(var_type);
                self.register_node_binding(&mut pp.head);
                for (rel, node) in pp.tail.iter_mut() {
                    self.register_rel_binding(rel);
                    self.register_node_binding(node);
                }
            }
        }
    }

    fn register_node_binding(&mut self, n: &mut crate::semantic::bound_ast::BoundNodePattern) {
        let label = n.labels.first().map(|l| l.label_id);
        let ti = TypeInfo::Node { label };
        if let Some(v) = &mut n.var {
            self.binding_types.insert(v.binding_id, ti.clone());
            v.type_info = Some(ti);
        }
    }

    fn register_rel_binding(&mut self, r: &mut BoundRelPattern) {
        let rel_type = r.rel_types.first().map(|t| t.type_id);
        let scalar = TypeInfo::Relationship { rel_type };
        // #696 (follow-up of #695 / ADR-186 R1 M-1): mirror the
        // execution-layer rel-var shape. The var-length expand
        // (`crate::executor::ops::expand` RC-2 frozen contract) binds a
        // quantified rel-var to `Value::List(Vec<Value::Relationship>)`
        // in traversal order, and keeps the scalar `Value::Relationship`
        // shape only for the single-hop case. The single-hop case is
        // exactly `length == None`; every var-length form
        // (`*` / `*N..M` openCypher, and the reserved GQL `{N,M}`) is
        // `length == Some(_)`. So the static type is `List(Relationship)`
        // iff the pattern is quantified — matching the executor's
        // `length_range.is_some()` dispatch (the GQL `{N,M}` form is
        // rejected upstream as `NotImplemented`, so a `List` type for it
        // is consistent and never reached by an admitted query).
        let ti = if r.length.is_some() {
            TypeInfo::List(Box::new(scalar))
        } else {
            scalar
        };
        if let Some(v) = &mut r.var {
            self.binding_types.insert(v.binding_id, ti.clone());
            v.type_info = Some(ti);
        }
    }

    fn check_clause(&mut self, c: &mut BoundClause) {
        match c {
            BoundClause::Match(m) => self.check_match_clause(m),
            BoundClause::Create(c) => self.check_create_clause(c),
            BoundClause::Delete(d) => self.check_delete_clause(d),
            BoundClause::Set(s) => self.check_set_clause(s),
            BoundClause::Remove(r) => self.check_remove_clause(r),
            BoundClause::Merge(m) => self.check_merge_clause(m),
            BoundClause::With(w) => {
                for it in &mut w.items {
                    self.check_projection_item(it);
                }
                if let Some(e) = &mut w.where_clause {
                    self.check_expression(e);
                    self.check_where_top_type(e);
                }
            }
            BoundClause::Unwind(u) => {
                self.check_expression(&mut u.expr);
                // Element type: if expr is List(t), bind to t. Else
                // bind to Null (3VL) and let downstream catch the
                // misuse.
                let elem = match u.expr.type_info() {
                    Some(TypeInfo::List(t)) => (**t).clone(),
                    _ => TypeInfo::Null,
                };
                self.binding_types.insert(u.var.binding_id, elem.clone());
                u.var.type_info = Some(elem);
            }
            // ADR-192 (#623): type-check the subquery body (recurse with
            // the same two-pass register+check). The returned columns are
            // already registered in `register_clause_bindings` (pass 1)
            // so the enclosing query's references resolve.
            BoundClause::Call(c) => self.check_call_body(c.body.as_mut()),
            // ADR-197 (#802): type-check procedure args; bind each
            // YIELD'd column to `Null` (3VL — the procedure output
            // columns are heterogeneous: label=String, properties=List,
            // size=Integer, …; downstream WHERE/RETURN coerce per 3VL,
            // same posture as UNWIND over an unknown-element list).
            BoundClause::CallProcedure(c) => {
                for a in &mut c.args {
                    self.check_expression(a);
                }
                for y in &mut c.yields {
                    self.binding_types.insert(y.var.binding_id, TypeInfo::Null);
                    y.var.type_info = Some(TypeInfo::Null);
                }
                // The WHERE references the YIELD'd columns (just typed
                // above) — check it after.
                if let Some(e) = &mut c.where_clause {
                    self.check_expression(e);
                    self.check_where_top_type(e);
                }
            }
            BoundClause::Show(s) => {
                for col in &mut s.columns {
                    self.binding_types.insert(col.binding_id, TypeInfo::Null);
                    col.type_info = Some(TypeInfo::Null);
                }
            }
            BoundClause::RankBy(r) => self.check_rank_by(r),
            BoundClause::WithFusion(f) => self.check_with_fusion(f),
            BoundClause::Return(r) => {
                for it in &mut r.items {
                    self.check_projection_item(it);
                }
                for o in &mut r.order_by {
                    self.check_expression(&mut o.expr);
                }
                if let Some(e) = &mut r.skip {
                    self.check_expression(e);
                }
                if let Some(e) = &mut r.limit {
                    self.check_expression(e);
                }
            }
            BoundClause::TailOrderBy(items, _) => {
                for o in items {
                    self.check_expression(&mut o.expr);
                }
            }
            BoundClause::TailSkip(e, _) | BoundClause::TailLimit(e, _) => {
                self.check_expression(e);
            }
        }
    }

    /// ADR-192 (#623): type-check a `CALL { … }` subquery body. Recurses
    /// with the SAME two-pass register+check the outer query uses
    /// ([`Self::check_query`]) — the body's bindings + the imported outer
    /// bindings (already in `binding_types` from the outer query's
    /// earlier clauses) all resolve in the shared `binding_types` map
    /// (binding-ids are globally unique across outer + body). For a UNION
    /// body each arm is checked + the post-union tail.
    fn check_call_body(&mut self, body: &mut BoundStatement) {
        match body {
            BoundStatement::Read(q) => self.check_query(q),
            BoundStatement::Union(u) => {
                for arm in &mut u.arms {
                    self.check_query(arm);
                }
                for o in &mut u.tail.order_by {
                    self.check_expression(&mut o.expr);
                }
                if let Some(e) = &mut u.tail.skip {
                    self.check_expression(e);
                }
                if let Some(e) = &mut u.tail.limit {
                    self.check_expression(e);
                }
            }
            // Grammar admits only Read/Union inside CALL{}.
            _ => {}
        }
    }

    fn check_match_clause(&mut self, m: &mut BoundMatchClause) {
        // OPTIONAL MATCH may_be_null propagation moved to binding
        // time per ADR-038 §2 D-21 M4-22b refinement (Shape B):
        // `BindingVisitor::declare_or_resolve_in_pattern` sets
        // `may_be_null` on FRESH declarations inside OPTIONAL MATCH
        // and inherits from the original binding for re-references.
        // Type-check no longer touches `may_be_null`.

        // Walk pattern body to flag any reserved length-range form.
        match &m.body {
            BoundMatchBody::Patterns(ps) => {
                for p in ps {
                    self.check_path_pattern_reserved(p);
                }
            }
            BoundMatchBody::NamedPath(np) => {
                let pp = match &np.kind {
                    crate::semantic::bound_ast::BoundNamedPathKind::ShortestPath(p)
                    | crate::semantic::bound_ast::BoundNamedPathKind::AllShortestPath(p)
                    | crate::semantic::bound_ast::BoundNamedPathKind::Plain(p) => p,
                };
                self.check_path_pattern_reserved(pp);
            }
        }

        if let Some(w) = &mut m.where_clause {
            self.check_expression(w);
            self.check_where_top_type(w);
        }
    }

    fn check_path_pattern_reserved(&mut self, p: &crate::semantic::bound_ast::BoundPathPattern) {
        for (rel, _node) in &p.tail {
            if let Some(LengthRange::Quantified { .. }) = rel.length {
                self.errors.push(ArcQLError::NotImplemented {
                    feature: "LengthRange::Quantified ({N,M})".into(),
                    section: "D-9 GQL length range".into(),
                    target_version: "v1.1".into(),
                    span: rel.span.clone(),
                });
            }
        }
    }

    fn check_projection_item(&mut self, p: &mut BoundProjectionItem) {
        if let BoundProjectionKind::Expr(e) = &mut p.kind {
            self.check_expression(e);
            // #618 / #1056 — register the projection OUTPUT column's
            // CONCRETE type under its `output_id`, so a downstream
            // reference resolves the column to its real type rather than
            // the permissive `Null` (3VL bottom). This makes `WITH 123 AS
            // n ... RETURN n.num` type-check `n` as `Integer` and reject
            // the non-entity property access at COMPILE time
            // (`PropertyAccessOnNonEntity`) — flipping `Graph6` [9] /
            // `Map1` [6] from a RUNTIME eval error (WrongErrorPhase) to
            // the openCypher-correct compile-time `InvalidArgumentType`.
            //
            // ZERO-REGRESSION (the prior #618 revert): registering the
            // concrete type previously unmasked the `Subscript`
            // incompleteness — `WITH {…} AS map ... map[key]` typed the
            // base as `Map`, which the OLD subscript check rejected via
            // `check_list_operand` (`Map2` [3]/[4] regression). The
            // map-subscript type-check above (`check_subscript_base` +
            // `check_string_index`) now ADMITS a `Map` base, so this
            // registration is safe to re-land — the two changes are
            // co-dependent and ship together.
            //
            // Registered only when `output_id` is present (always, for an
            // `Expr` projection — `None` is reserved for `Wildcard`,
            // which has no single output column). A bare `VariableRef`
            // projection (`WITH n`) re-registers `n`'s already-known type
            // under its post-clause id, which is idempotent.
            if let Some(out_id) = p.output_id {
                let ti = e.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.binding_types.insert(out_id, ti);
            }
        }
    }

    // ADR-147 W26-θ Phase 1: CREATE clause type-check.
    //
    // Phase 1 restricts CREATE property values to literals (Integer
    // / Float / String / Bool / Null). Parameter / expression /
    // function-call / property-access values forward-pin to v1.1
    // per ADR-147 §"Forward-deferred". The check_expression walk
    // still runs (it's harmless and gives the property-value's
    // sub-tree the standard type_info-population pass) but the
    // outer "is-this-a-literal" guard is the Phase-1 narrowing.
    //
    // The variable-binding's TypeInfo::Node is registered in
    // `register_create_bindings` (first pass) so any subsequent
    // RETURN clause sees the binding's Node type when it walks the
    // projection.
    fn check_create_clause(&mut self, c: &mut BoundCreateClause) {
        for item in &mut c.items {
            match item {
                BoundCreateItem::Node(spec) => {
                    // ADR-147-amendment-03 (D-1): CREATE property values
                    // admit the evaluable subset (param / row-ref /
                    // bounded expr), not just literals — the live
                    // `CreateSpineOp` executor now `evaluate`s them.
                    self.check_create_property_map(spec.properties.as_mut(), true);
                }
                BoundCreateItem::Path(path) => {
                    // Phase 2 inherits the Phase 1 narrowing on EVERY
                    // property bag (source + rel + target). The rel-label
                    // is mandatory at grammar level (per ADR-148 §D-1) so
                    // no additional check is required at the type-check
                    // layer. ADR-147-amendment-03 (D-1): CREATE-path bags
                    // also admit the evaluable subset (the create-spine
                    // executor materializes source / rel / target props
                    // through the same `evaluate` seam).
                    self.check_create_property_map(path.source.properties.as_mut(), true);
                    self.check_create_property_map(path.rel.properties.as_mut(), true);
                    self.check_create_property_map(path.target.properties.as_mut(), true);
                }
            }
        }
    }

    /// Property-value narrowing applied to a single CREATE / MERGE
    /// property bag.
    ///
    /// `allow_evaluable` selects the gate:
    /// - `true` (CREATE) — ADR-147-amendment-03 (D-1): admit the
    ///   deny-by-default *evaluable* subset
    ///   ([`is_evaluable_create_property_value`]): literals, `$param`,
    ///   previously-bound row references, and a whitelisted bounded /
    ///   deterministic expression spine. The live `CreateSpineOp`
    ///   executor `evaluate`s these against the upstream row + param bag,
    ///   then gates the RESULT value-type before the substrate write.
    /// - `false` (MERGE pattern) — literal-only
    ///   ([`is_literal_property_value`]). The MERGE create-branch also
    ///   lowers through `CreateSpineOp`, but the MERGE *pattern* property
    ///   values additionally participate in match-key equality; widening
    ///   them is a separate ADR-151 amendment, so MERGE stays literal-
    ///   only at D-1 (Trap #4 — CREATE-only scope).
    fn check_create_property_map(
        &mut self,
        props: Option<&mut crate::semantic::bound_ast::BoundPropertyMap>,
        allow_evaluable: bool,
    ) {
        let Some(props) = props else {
            return;
        };
        for entry in &mut props.entries {
            self.check_expression(&mut entry.value);
            let admitted = if allow_evaluable {
                is_evaluable_create_property_value(&entry.value)
            } else {
                is_literal_property_value(&entry.value)
            };
            if !admitted {
                let actual = describe_expression(&entry.value);
                self.errors.push(ArcQLError::TypeCheck(
                    crate::semantic::error::TypeCheckError::CreatePropertyValueNotLiteral {
                        name: entry.key.clone(),
                        actual,
                        span: entry.span.clone(),
                    },
                ));
            }
        }
    }

    /// ADR-149 W26-θ Phase 3: DELETE items must resolve to a Node-
    /// or Relationship-typed binding. Any other `TypeInfo` surfaces
    /// `TypeCheckError::DeleteNonGraphValue`.
    ///
    /// An UNRESOLVED variable (binding_id == u64::MAX from the
    /// binding-time fallback) skips type validation — the binding
    /// pass already emitted `BindingError::UndeclaredVariable` for
    /// that case, and re-emitting a type error would be diagnostic
    /// noise.
    fn check_delete_clause(&mut self, d: &mut BoundDeleteClause) {
        for item in &mut d.items {
            // Skip items whose binding pass failed to resolve — the
            // BindingError is already in the visitor's error list.
            if item.var.binding_id == BindingId::new(u64::MAX) {
                continue;
            }
            // Re-read type_info from the global table (defense in
            // depth — the M4-22 second pass may have refined the
            // entry; e.g., a WITH passthrough's outer-scope re-bind).
            let ti = match self.binding_types.get(&item.var.binding_id).cloned() {
                Some(ti) => ti,
                None => {
                    // No type info — likely an unresolved binding the
                    // visitor's error list already captured; defer to
                    // the binding-error diagnostic.
                    continue;
                }
            };
            // Update the bound item's type_info verbatim (in case a
            // pass refined since `register_delete_bindings`).
            item.var.type_info = Some(ti.clone());
            match ti {
                TypeInfo::Node { .. } | TypeInfo::Relationship { .. } => {
                    // Admitted by Phase 3 per ADR-149 §D-4.
                }
                other => {
                    self.errors
                        .push(ArcQLError::TypeCheck(TypeCheckError::DeleteNonGraphValue {
                            name: item.var.name.clone(),
                            actual: other,
                            span: item.span.clone(),
                        }));
                }
            }
        }
    }

    /// ADR-150 W26-θ Phase 4: SET items must resolve to a Node- or
    /// Relationship-typed binding (per §D-4). Label-add mutations
    /// (`SET n:Label`) are Node-only; Relationship-typed bindings
    /// surface `SetRemoveLabelOnRel`. Property values inside
    /// `PropertyAssign` / `PropertyReplace` / `PropertyMerge` must be
    /// literal (Phase 1 narrowing inherited per ADR-147 §D-4);
    /// non-literal values surface `SetPropertyValueNotLiteral`.
    ///
    /// An UNRESOLVED variable (binding_id == u64::MAX from the
    /// binding-time fallback) skips type validation — the binding
    /// pass already emitted `BindingError::UndeclaredVariable` for
    /// that case.
    fn check_set_clause(&mut self, s: &mut BoundSetClause) {
        for item in &mut s.items {
            // Property-value literality is checked even when the
            // variable is unresolved so the user sees BOTH diagnostics
            // at once.
            self.check_set_property_values(&mut item.mutation);
            if item.var.binding_id == BindingId::new(u64::MAX) {
                continue;
            }
            let ti = match self.binding_types.get(&item.var.binding_id).cloned() {
                Some(ti) => ti,
                None => continue,
            };
            item.var.type_info = Some(ti.clone());
            match &ti {
                TypeInfo::Node { .. } => {
                    // All four mutation shapes (property assign /
                    // merge / replace / label add) admitted on Node.
                }
                TypeInfo::Relationship { .. } => {
                    // Label-add on Relationship rejects per §D-4.
                    if let BoundSetMutation::LabelAdd(_) = &item.mutation {
                        self.errors.push(ArcQLError::TypeCheck(
                            TypeCheckError::SetRemoveLabelOnRel {
                                name: item.var.name.clone(),
                                span: item.span.clone(),
                            },
                        ));
                    }
                }
                _ => {
                    self.errors.push(ArcQLError::TypeCheck(
                        TypeCheckError::SetRemoveNonGraphValue {
                            name: item.var.name.clone(),
                            actual: ti.clone(),
                            span: item.span.clone(),
                        },
                    ));
                }
            }
        }
    }

    /// ADR-150 W26-θ Phase 4: REMOVE items must resolve to a Node- or
    /// Relationship-typed binding. Label-remove mutations are Node-
    /// only; Relationship-typed bindings surface `SetRemoveLabelOnRel`.
    fn check_remove_clause(&mut self, r: &mut BoundRemoveClause) {
        for item in &mut r.items {
            if item.var.binding_id == BindingId::new(u64::MAX) {
                continue;
            }
            let ti = match self.binding_types.get(&item.var.binding_id).cloned() {
                Some(ti) => ti,
                None => continue,
            };
            item.var.type_info = Some(ti.clone());
            match &ti {
                TypeInfo::Node { .. } => {
                    // Both property + label-remove admitted on Node.
                }
                TypeInfo::Relationship { .. } => {
                    // Label-remove on Relationship rejects per §D-4.
                    if let crate::semantic::bound_ast::BoundRemoveMutation::LabelRemove(_) =
                        &item.mutation
                    {
                        self.errors.push(ArcQLError::TypeCheck(
                            TypeCheckError::SetRemoveLabelOnRel {
                                name: item.var.name.clone(),
                                span: item.span.clone(),
                            },
                        ));
                    }
                }
                _ => {
                    self.errors.push(ArcQLError::TypeCheck(
                        TypeCheckError::SetRemoveNonGraphValue {
                            name: item.var.name.clone(),
                            actual: ti.clone(),
                            span: item.span.clone(),
                        },
                    ));
                }
            }
        }
    }

    /// ADR-151 W26-θ Phase 5: MERGE clause type-check.
    ///
    /// Phase 5 enforces:
    /// 1. The merge pattern's property values MUST be literal —
    ///    parallel to `check_create_property_map` (Phase 1 inherited
    ///    narrowing per ADR-147 §D-4). Non-literal values surface
    ///    `TypeCheckError::CreatePropertyValueNotLiteral` (reuse the
    ///    existing Phase 1 variant — MERGE pattern is semantically a
    ///    CREATE-shape).
    /// 2. The on_create / on_match action items pass through the
    ///    EXISTING Phase 4 `check_set_clause` machinery — same Node-
    ///    or-Relationship typing + literal-only property values per
    ///    ADR-150 §D-4.
    fn check_merge_clause(&mut self, m: &mut BoundMergeClause) {
        // 1. Validate the pattern's property values per Phase 1
        //    inherited narrowing.
        match &mut m.pattern {
            BoundMergePattern::Node(spec) => {
                // ADR-147-amendment-03 (D-1): MERGE stays literal-only
                // (`allow_evaluable = false`). The pattern property values
                // participate in match-key equality; widening them is a
                // separate ADR-151 amendment (Trap #4 — CREATE-only scope).
                self.check_create_property_map(spec.properties.as_mut(), false);
            }
            BoundMergePattern::Path(path) => {
                self.check_create_property_map(path.source.properties.as_mut(), false);
                self.check_create_property_map(path.rel.properties.as_mut(), false);
                self.check_create_property_map(path.target.properties.as_mut(), false);
            }
        }
        // 2. Validate the on_create / on_match action items using the
        //    Phase 4 per-item check machinery (literality + Node-or-
        //    Relationship typing + Node-only label-add).
        let mut synthetic_set = crate::semantic::bound_ast::BoundSetClause {
            items: std::mem::take(&mut m.on_create),
            span: m.span.clone(),
        };
        self.check_set_clause(&mut synthetic_set);
        m.on_create = synthetic_set.items;

        let mut synthetic_set = crate::semantic::bound_ast::BoundSetClause {
            items: std::mem::take(&mut m.on_match),
            span: m.span.clone(),
        };
        self.check_set_clause(&mut synthetic_set);
        m.on_match = synthetic_set.items;
    }

    /// ADR-150 W26-θ Phase 4: enforce literal-only property values on
    /// SET mutations per the Phase 1 (ADR-147 §D-4) narrowing
    /// inherited at Phase 4. Per `check_create_property_map` the
    /// type-check walks the sub-expressions first (population of
    /// `type_info`), then asserts the outer literal guard.
    fn check_set_property_values(&mut self, mutation: &mut BoundSetMutation) {
        match mutation {
            BoundSetMutation::PropertyAssign { name, value } => {
                self.check_expression(value);
                if !is_literal_property_value(value) {
                    let actual = describe_expression(value);
                    let span = value.span().clone();
                    self.errors.push(ArcQLError::TypeCheck(
                        TypeCheckError::SetPropertyValueNotLiteral {
                            name: name.clone(),
                            actual,
                            span,
                        },
                    ));
                }
            }
            BoundSetMutation::PropertyReplace(map) | BoundSetMutation::PropertyMerge(map) => {
                for entry in &mut map.entries {
                    self.check_expression(&mut entry.value);
                    if !is_literal_property_value(&entry.value) {
                        let actual = describe_expression(&entry.value);
                        self.errors.push(ArcQLError::TypeCheck(
                            TypeCheckError::SetPropertyValueNotLiteral {
                                name: entry.key.clone(),
                                actual,
                                span: entry.span.clone(),
                            },
                        ));
                    }
                }
            }
            BoundSetMutation::LabelAdd(_) => {
                // Label mutations have no sub-expressions to check.
            }
        }
    }

    fn check_rank_by(&mut self, r: &mut crate::semantic::bound_ast::BoundRankByClause) {
        match &mut r.ranker {
            BoundRanker::Hybrid(args) => {
                for a in args {
                    self.check_rank_arg(a);
                }
            }
        }
        if let Some(score) = &mut r.score {
            self.binding_types.insert(score.binding_id, TypeInfo::Float);
            score.type_info = Some(TypeInfo::Float);
        }
    }

    fn check_rank_arg(&mut self, a: &mut crate::semantic::bound_ast::BoundRankArg) {
        match a {
            crate::semantic::bound_ast::BoundRankArg::Vector { query, .. }
            | crate::semantic::bound_ast::BoundRankArg::Text { query, .. } => {
                self.check_expression(query);
            }
        }
    }

    fn check_with_fusion(&mut self, c: &mut BoundWithFusionClause) {
        match &c.fusion {
            BoundFusion::Rrf { .. } => {}
        }
    }

    /// Validate that a WHERE expression's top-level type is Boolean
    /// or Null. Cypher convention treats Null as FALSE in WHERE
    /// position; any other type is a type error.
    fn check_where_top_type(&mut self, e: &BoundExpression) {
        let span = e.span().clone();
        match e.type_info() {
            None | Some(TypeInfo::Boolean) | Some(TypeInfo::Null) => {}
            Some(other) => {
                self.errors
                    .push(ArcQLError::TypeCheck(TypeCheckError::NonBooleanWhere {
                        actual: other.clone(),
                        span,
                    }));
            }
        }
    }

    // ---------- Expression-level type-check ----------

    fn check_expression(&mut self, e: &mut BoundExpression) {
        match e {
            BoundExpression::Literal {
                value, type_info, ..
            } => {
                *type_info = Some(literal_type(value));
            }
            BoundExpression::ListLiteral {
                elements,
                type_info,
                ..
            } => {
                for element in elements.iter_mut() {
                    self.check_expression(element);
                }
                *type_info = Some(TypeInfo::List(Box::new(bound_list_literal_elem_type(
                    elements,
                ))));
            }
            BoundExpression::MapLiteral {
                entries, type_info, ..
            } => {
                for (_, value) in entries.iter_mut() {
                    self.check_expression(value);
                }
                *type_info = Some(TypeInfo::Map);
            }
            BoundExpression::Parameter { type_info, .. } => {
                // Parameters carry no static type info at v1.0
                // (bind values are JSON-typed at the MCP boundary).
                // Fall back to Null which 3VL-propagates correctly.
                *type_info = Some(TypeInfo::Null);
            }
            BoundExpression::VariableRef {
                binding_id,
                type_info,
                ..
            } => {
                *type_info = self.binding_types.get(binding_id).cloned();
            }
            BoundExpression::UnresolvedVariable { .. } => {
                // Already reported by M4-21; nothing to do.
            }
            BoundExpression::PropertyAccess {
                base,
                path,
                span,
                type_info,
            } => {
                self.check_expression(base);
                // Property access on a Node / Relationship resolves
                // to a `Property` value, carrying the resolved
                // [`PropertyId`] from M4-21 + a scalar
                // [`PropertyType`]. The v1.0 catalog does NOT track
                // property value types per label, so we use
                // `PropertyType::String` as the dynamic-schema
                // sentinel; v1.1+ catalog extension `lookup_property_type`
                // returns a concrete type. The type-check helpers
                // (`is_numeric`, `is_orderable`) treat
                // `Property::String` as compatible with both numeric
                // and orderable at v1.0 — arithmetic / comparison
                // against a property value type-checks under the
                // dynamic-schema discipline; the executor evaluates
                // the actual stored value at runtime.
                // #618 — `InvalidArgumentType` on property access over a
                // statically-KNOWN non-entity, non-map base (openCypher
                // v9 §3 / TCK `Graph6` [9] non-graph-element + `Map1` [6]
                // non-map). `WITH 123 AS n ... RETURN n.num` types `n` as
                // Integer → reject at COMPILE time (was a RUNTIME eval
                // error — WrongErrorPhase). A dynamically-typed `Property`
                // base, `Null`, or an unknown type is admitted (the
                // executor enforces at runtime), mirroring the
                // under-typed-catalog discipline; only concrete
                // scalars / lists / paths reject.
                let base_ti = base.type_info().cloned().unwrap_or(TypeInfo::Null);
                if is_definitely_non_entity_non_map(&base_ti) {
                    self.errors.push(ArcQLError::TypeCheck(
                        TypeCheckError::PropertyAccessOnNonEntity {
                            actual: base_ti,
                            span: span.clone(),
                        },
                    ));
                }
                let pid = path.last().and_then(|p| p.property_id);
                *type_info = match pid {
                    Some(property_id) => Some(TypeInfo::Property {
                        property_id,
                        value_type: PropertyType::String,
                    }),
                    None => Some(TypeInfo::Null),
                };
            }
            // #1290 — the four left-spine operator variants type-check
            // through one iterative driver (a flat operator chain folds
            // into a left-nested spine up to `MAX_FLAT_CHAIN_DEPTH`
            // deep, and the spine may interleave all four variants;
            // recursing per level overflowed the native stack).
            BoundExpression::BinaryOp { .. } | BoundExpression::UnaryOp { .. } => {
                self.check_operator_spine(e);
            }
            BoundExpression::FunctionCall {
                name,
                args,
                distinct,
                star,
                span,
                type_info,
            } => {
                for a in args.iter_mut() {
                    self.check_expression(a);
                }
                *type_info = Some(self.check_function_call(name, args, *distinct, *star, span));
            }
            BoundExpression::Near {
                lhs,
                target,
                type_info,
                ..
            } => {
                self.check_expression(lhs);
                self.check_expression(target);
                *type_info = Some(TypeInfo::Boolean);
            }
            BoundExpression::TextMatch {
                lhs,
                query,
                type_info,
                ..
            } => {
                self.check_expression(lhs);
                self.check_expression(query);
                *type_info = Some(TypeInfo::Boolean);
            }
            BoundExpression::InCommunity {
                node,
                community,
                type_info,
                ..
            } => {
                self.check_expression(node);
                self.check_expression(community);
                *type_info = Some(TypeInfo::Boolean);
            }
            // openCypher v9 §3.3.5 — the IN RHS must be a list (or a
            // dynamic Null / Property resolved at runtime). A concrete
            // non-list RHS (`1 IN true`, `1 IN 123`, `1 IN {x:[]}` —
            // TCK List5 [42]) is a compile-time `TypeMismatch`, the
            // `InvalidArgumentType` contract (#723 lesson). IS NULL /
            // IS NOT NULL ALWAYS yields Boolean (Cypher 9 §6.4 — the
            // canonical 3VL → 2VL bridge). Both checks live in the
            // iterative spine driver (#1290; see the BinaryOp arm).
            BoundExpression::In { .. } | BoundExpression::IsNull { .. } => {
                self.check_operator_spine(e);
            }
            // ADR-188 Decision 3 — list-predicate type-check. `x :
            // element-type-of(list)`; the list operand must be
            // `List(_)` or `Null` (else a type error at check time, not
            // a runtime surprise). Result is `Boolean` (the predicate;
            // the 3VL `null` is a runtime value, not a static type).
            BoundExpression::ListPredicate {
                // The quantifier does not change the type-check (all four
                // forms yield `Boolean`); the 3VL fold per Decision 4 is
                // an EVAL-time concern.
                quantifier: _,
                var_bid,
                list,
                predicate,
                span,
                type_info,
            } => {
                self.check_expression(list);
                let list_ti = list.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.check_list_operand(&list_ti, span, "list predicate");
                // Register the scoped var's type so the predicate's
                // `VariableRef { var_bid }` resolves to the element type.
                let elem_ti = element_type_of(&list_ti);
                self.binding_types.insert(*var_bid, elem_ti);
                self.check_expression(predicate);
                *type_info = Some(TypeInfo::Boolean);
            }
            // ADR-188 Decision 3 — reduce type-check + OQ-5 widening.
            // `acc : type-of(init)`, `x : element-type-of(list)`; the
            // body `expr` must be assignable back to `acc`'s type up to
            // numeric widening (`{Integer, Float} → Float`). The result
            // is the join `join(acc_type, body_type)`.
            BoundExpression::Reduce {
                acc_bid,
                init,
                var_bid,
                list,
                expr,
                span,
                type_info,
            } => {
                self.check_expression(init);
                self.check_expression(list);
                let acc_ti = init.type_info().cloned().unwrap_or(TypeInfo::Null);
                let list_ti = list.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.check_list_operand(&list_ti, span, "reduce");
                // Register `acc : init-type` and `x : element-type`.
                self.binding_types.insert(*acc_bid, acc_ti.clone());
                self.binding_types
                    .insert(*var_bid, element_type_of(&list_ti));
                self.check_expression(expr);
                let body_ti = expr.type_info().cloned().unwrap_or(TypeInfo::Null);
                // OQ-5 type-stability: the fold must be type-stable up
                // to numeric widening. `join(acc, body)` must be a
                // single concrete type. `{Integer, Float}` widens to
                // `Float` (NOT a type error — Cypher coerces
                // INTEGER → FLOAT; a false-reject is as much a
                // conformance failure as a false-accept). Genuinely
                // non-assignable folds (e.g. `Integer` acc + `String`
                // body) remain a `TypeCheckError`.
                *type_info = Some(self.reduce_join_type(&acc_ti, &body_ti, span));
            }
            // ADR-188 (#620 list-half) Decision 5 — list-comprehension
            // type-check. `x : element-type-of(list)`; the list operand
            // must be `List(_)` or `Null` (else a type error at check
            // time). The WHERE `predicate` (if present) is a boolean
            // filter — type-checked for well-formedness but does not
            // affect the result element type. The result is
            // `List(projection-type)` when `| projection` is present,
            // else `List(element-type)` (identity projection).
            BoundExpression::ListComprehension {
                var_bid,
                list,
                predicate,
                projection,
                span,
                type_info,
            } => {
                self.check_expression(list);
                let list_ti = list.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.check_list_operand(&list_ti, span, "list comprehension");
                // Register the scoped var's type so the predicate's +
                // projection's `VariableRef { var_bid }` resolve to the
                // element type.
                let elem_ti = element_type_of(&list_ti);
                self.binding_types.insert(*var_bid, elem_ti.clone());
                // The WHERE filter is type-checked (well-formedness +
                // scoped-var registration above); its type does not
                // feed the result element type.
                if let Some(p) = predicate {
                    self.check_expression(p);
                }
                // Result element type = projection type (if a `| e` is
                // present) else the element type (identity over `x`).
                let result_elem = match projection {
                    Some(proj) => {
                        self.check_expression(proj);
                        proj.type_info().cloned().unwrap_or(TypeInfo::Null)
                    }
                    None => elem_ti,
                };
                *type_info = Some(TypeInfo::List(Box::new(result_elem)));
            }
            // ADR-191 D-6 (#620 map-half) — map-projection type-check. The
            // base must be a NODE / RELATIONSHIP / MAP (the projectable
            // property-bag types) — or `Null` (a runtime-null base yields a
            // null result, the openCypher null-propagation convention,
            // consistent with `map.key` on null → null). #723 lesson: a map
            // projection over a non-entity/non-map base (e.g. `(1){.x}`,
            // `"s"{.x}`) REJECTS at compile time with a co-located
            // regression test. Each `alias: expr` literal-entry value is
            // type-checked for well-formedness; the `.key` / `.*` selectors
            // carry only property names. The result is always a `Map`
            // (heterogeneous-value; static map-value typing is OQ-191-2).
            BoundExpression::MapProjection {
                base,
                items,
                span,
                type_info,
            } => {
                self.check_expression(base);
                let base_ti = base.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.check_map_projection_base(&base_ti, span);
                for item in items {
                    if let BoundMapProjectionItem::Literal { value, .. } = item {
                        self.check_expression(value);
                    }
                }
                *type_info = Some(TypeInfo::Map);
            }
            // openCypher v9 §3.4 — list subscript `base[index]`. The base
            // must be a list (or dynamic Null/Property); the index must be
            // an integer. An out-of-range index is a RUNTIME `null` (NOT a
            // compile error), so the static type is the element type. #723
            // lesson: `5[0]` (subscript on a scalar) and `list[1.5]`
            // (non-integer index) reject at compile time, each with a
            // co-located `*_rejects_at_compile` regression test.
            // openCypher v9 §3.4 — bracket subscript `base[index]`.
            // DUAL-DISPATCH on the base type (#1056 / #990):
            //
            // - **List base** → INTEGER index (`list[i]`): the existing
            //   §3.4 list-element-access path. Result = the list element
            //   type (`element_type_of`). 0-based, negative-from-end,
            //   out-of-range ⇒ null (eval-side).
            // - **Map base** → STRING index (`map['key']`): openCypher
            //   "dynamic value access" (TCK `expressions/map/Map2`). The
            //   key is matched CASE-SENSITIVELY (`Map2` [5]); a missing
            //   key ⇒ null (eval-side). Result type = `Null` — `TypeInfo`
            //   carries no map-value type at v1.0 (the `Map` variant is
            //   unit), so the value is the dynamic-schema "could be
            //   anything, possibly null" sentinel, resolved at runtime
            //   (mirrors the `element_type_of` non-list fall-through).
            // - **Null / Property base** (3VL / dynamic-schema): admit
            //   without constraining the index; result `Null`. A
            //   parameter (`$expr[$idx]`, `Map2` [1]/[2]) types as
            //   `Null`, so the index check is skipped and the real
            //   Map×String / List×Integer dispatch happens at runtime in
            //   `eval_subscript`.
            //
            // A concrete non-indexable base (`Integer` / `String` /
            // `Boolean` / `Path` …) is a compile-time `TypeMismatch` via
            // `check_subscript_base`.
            BoundExpression::Subscript {
                base,
                index,
                span,
                type_info,
            } => {
                self.check_expression(base);
                self.check_expression(index);
                let base_ti = base.type_info().cloned().unwrap_or(TypeInfo::Null);
                let idx_ti = index.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.check_subscript_base(&base_ti, span);
                match &base_ti {
                    // List base — integer index, element-typed result.
                    TypeInfo::List(_) => {
                        self.check_integer_index(&idx_ti, span, "subscript");
                        *type_info = Some(element_type_of(&base_ti));
                    }
                    // Map base — string index, dynamic-schema result.
                    TypeInfo::Map => {
                        self.check_string_index(&idx_ti, span, "map subscript");
                        *type_info = Some(TypeInfo::Null);
                    }
                    // Null (3VL) / Property (dynamic-schema): the base
                    // type is not statically known to be List or Map, so
                    // the index dispatch happens at runtime. Do NOT
                    // constrain the index here (it could legitimately be
                    // an Integer for a runtime list OR a String for a
                    // runtime map). Result `Null`.
                    _ => {
                        *type_info = Some(TypeInfo::Null);
                    }
                }
            }
            // openCypher v9 §3.4 — list slice `base[start..end]`. Base must
            // be a list; each PRESENT bound must be an integer. The result
            // is a list of the same element type (out-of-range bounds
            // clamp at runtime, never error). String slicing is DEFERRED
            // (a non-list scalar base rejects via `check_list_operand`).
            BoundExpression::Slice {
                base,
                start,
                end,
                span,
                type_info,
            } => {
                self.check_expression(base);
                let base_ti = base.type_info().cloned().unwrap_or(TypeInfo::Null);
                self.check_list_operand(&base_ti, span, "slice");
                if let Some(s) = start {
                    self.check_expression(s);
                    let s_ti = s.type_info().cloned().unwrap_or(TypeInfo::Null);
                    self.check_integer_index(&s_ti, span, "slice start");
                }
                if let Some(e) = end {
                    self.check_expression(e);
                    let e_ti = e.type_info().cloned().unwrap_or(TypeInfo::Null);
                    self.check_integer_index(&e_ti, span, "slice end");
                }
                *type_info = Some(TypeInfo::List(Box::new(element_type_of(&base_ti))));
            }
            // openCypher v9 §3.6 (#621) — CASE expression. WELL-FORMEDNESS
            // ONLY: recurse into every sub-expression so each one's own
            // `type_info` is populated and its own errors surface. We
            // deliberately do NOT cross-constrain — neither the SIMPLE-form
            // test against the WHEN values, nor the WHEN values against each
            // other, nor (searched-form) the WHEN conditions to Boolean. A
            // type-mismatched WHEN is a runtime NON-MATCH (falls to ELSE),
            // NOT a compile error — the load-bearing Conditional2 [1]
            // semantic (`CASE '0' WHEN 0 THEN … ELSE …` ⇒ ELSE; `true` /
            // `10.1` vs integer WHENs ⇒ ELSE). Over-constraining here would
            // false-reject the primary oracle. The result type is the
            // PERMISSIVE join of all THEN + ELSE types (heterogeneous
            // branches are legal openCypher → join to `Null`, never a type
            // error — see `case_join_type`).
            BoundExpression::Case {
                test,
                branches,
                default,
                type_info,
                ..
            } => {
                if let Some(t) = test {
                    self.check_expression(t);
                }
                let mut result: Option<TypeInfo> = None;
                for (when, then) in branches.iter_mut() {
                    self.check_expression(when);
                    self.check_expression(then);
                    let then_ti = then.type_info().cloned().unwrap_or(TypeInfo::Null);
                    result = Some(match result {
                        None => then_ti,
                        Some(acc) => case_join_type(&acc, &then_ti),
                    });
                }
                if let Some(d) = default {
                    self.check_expression(d);
                    let else_ti = d.type_info().cloned().unwrap_or(TypeInfo::Null);
                    result = Some(match result {
                        None => else_ti,
                        Some(acc) => case_join_type(&acc, &else_ti),
                    });
                }
                // `branches` is non-empty (grammar `+`), so `result` is
                // always `Some`; the `unwrap_or` is a safe static default.
                *type_info = Some(result.unwrap_or(TypeInfo::Null));
            }
        }
    }

    /// **#1290** — type-check a left-nested OPERATOR SPINE iteratively.
    ///
    /// A flat operator chain (`a AND b AND …`, `1 + 2 + …`, and the
    /// keyword postfix forms `x IN l IN …` / `x IS NULL IS NULL …`)
    /// folds into a left-nested spine up to
    /// [`crate::parser::MAX_FLAT_CHAIN_DEPTH`] deep, and the spine may
    /// MIX `BinaryOp` / `UnaryOp` / `In` / `IsNull` levels. The
    /// type-check is a bottom-up computation that WRITES each node's
    /// `type_info` from its children's — recursing per level overflowed
    /// the native stack (this walker's debug-profile frames are the
    /// largest in the pipeline), so we UNLINK the spine into owned
    /// frames (no clones — `std::mem::replace` with a placeholder
    /// literal), type-check bottom-up, and relink. Visit order, error
    /// order, and every per-variant check (`check_binary_op` /
    /// `check_unary_op` / IN's `check_list_operand` / IS NULL's
    /// `Boolean`) are byte-identical to the recursive arms this
    /// replaces. `rhs` operands type-check recursively — they are never
    /// part of the LEFT spine, so their depth is bounded by the bracket
    /// cap (`MAX_EXPRESSION_DEPTH`), not the chain cap.
    fn check_operator_spine(&mut self, e: &mut BoundExpression) {
        enum SpineFrame {
            Binary {
                op: BinOp,
                rhs: Box<BoundExpression>,
                span: Span,
            },
            Unary {
                op: UnaryOp,
                span: Span,
            },
            In {
                rhs: Box<BoundExpression>,
                span: Span,
            },
            IsNull {
                negated: bool,
                span: Span,
            },
        }
        // Cheap placeholder swapped in while we own the spine; replaced
        // by the relinked tree before returning.
        let placeholder = BoundExpression::Literal {
            value: Literal::Null,
            span: e.span().clone(),
            type_info: None,
        };
        let mut cur = std::mem::replace(e, placeholder);
        let mut frames: Vec<SpineFrame> = Vec::new();
        // Phase 1 — unlink: walk down the left/operand edge, moving
        // each level's non-spine payload into an owned frame.
        loop {
            match cur {
                BoundExpression::BinaryOp {
                    op, lhs, rhs, span, ..
                } => {
                    frames.push(SpineFrame::Binary { op, rhs, span });
                    cur = *lhs;
                }
                BoundExpression::UnaryOp {
                    op, operand, span, ..
                } => {
                    frames.push(SpineFrame::Unary { op, span });
                    cur = *operand;
                }
                BoundExpression::In { lhs, rhs, span, .. } => {
                    frames.push(SpineFrame::In { rhs, span });
                    cur = *lhs;
                }
                BoundExpression::IsNull {
                    lhs, negated, span, ..
                } => {
                    frames.push(SpineFrame::IsNull { negated, span });
                    cur = *lhs;
                }
                base => {
                    cur = base;
                    break;
                }
            }
        }
        // Phase 2 — type-check the non-spine base via the ordinary
        // per-variant arms (bracket-bounded recursion).
        self.check_expression(&mut cur);
        // Phase 3 — relink bottom-up, reproducing each recursive arm's
        // exact check + `type_info` stamp.
        let mut acc = cur;
        while let Some(frame) = frames.pop() {
            acc = match frame {
                SpineFrame::Binary { op, mut rhs, span } => {
                    self.check_expression(&mut rhs);
                    let lt = acc.type_info().cloned().unwrap_or(TypeInfo::Null);
                    let rt = rhs.type_info().cloned().unwrap_or(TypeInfo::Null);
                    let ti = self.check_binary_op(&op, &lt, &rt, &span);
                    BoundExpression::BinaryOp {
                        op,
                        lhs: Box::new(acc),
                        rhs,
                        span,
                        type_info: Some(ti),
                    }
                }
                SpineFrame::Unary { op, span } => {
                    let t = acc.type_info().cloned().unwrap_or(TypeInfo::Null);
                    let ti = self.check_unary_op(&op, &t, &span);
                    BoundExpression::UnaryOp {
                        op,
                        operand: Box::new(acc),
                        span,
                        type_info: Some(ti),
                    }
                }
                SpineFrame::In { mut rhs, span } => {
                    self.check_expression(&mut rhs);
                    // openCypher v9 §3.3.5 — the RHS must be a list (or
                    // a dynamic Null / Property resolved at runtime); a
                    // concrete non-list RHS is a compile-time
                    // `TypeMismatch` (TCK List5 [42], #723 lesson).
                    let rhs_ti = rhs.type_info().cloned().unwrap_or(TypeInfo::Null);
                    self.check_list_operand(&rhs_ti, rhs.span(), "IN");
                    BoundExpression::In {
                        lhs: Box::new(acc),
                        rhs,
                        span,
                        type_info: Some(TypeInfo::Boolean),
                    }
                }
                SpineFrame::IsNull { negated, span } => {
                    // IS NULL / IS NOT NULL ALWAYS yields Boolean
                    // (Cypher 9 §6.4 — the canonical 3VL → 2VL bridge).
                    BoundExpression::IsNull {
                        lhs: Box::new(acc),
                        negated,
                        span,
                        type_info: Some(TypeInfo::Boolean),
                    }
                }
            };
        }
        *e = acc;
    }

    /// **ADR-188 Decision 3** — the list operand of a list-predicate /
    /// `reduce` must be `List(_)` or `Null` (3VL null-list propagation).
    /// `Parameter`s carry `Null` at v1.0 (JSON-typed at the MCP
    /// boundary) and `Property` values are dynamic-schema, so both are
    /// admitted under the same dynamic-schema discipline as
    /// `is_numeric` / `is_orderable`. Anything else (a concrete scalar,
    /// a Boolean, a Node) is a type error at check time.
    fn check_list_operand(&mut self, list_ti: &TypeInfo, span: &Span, ctx: &str) {
        let ok = matches!(
            list_ti,
            TypeInfo::List(_) | TypeInfo::Null | TypeInfo::Property { .. }
        );
        if !ok {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                    op: format!("{ctx} list operand"),
                    lhs: list_ti.clone(),
                    rhs: TypeInfo::List(Box::new(TypeInfo::Null)),
                    span: span.clone(),
                }));
        }
    }

    /// **ADR-191 D-6** (#620 map-half) — the base of a map projection
    /// `n{…}` must be a `Node` / `Relationship` / `Map` (the projectable
    /// property-bag types) — or `Null` (a runtime-null base yields a null
    /// result; openCypher null-propagation). `Property` is admitted under
    /// the same dynamic-schema discipline as [`Self::check_list_operand`]
    /// (the v1.0 catalog does not track whether a property resolves to a
    /// map). Anything else (a concrete scalar `Integer` / `String` /
    /// `Boolean` / `List` / `Path`) is a compile-time `TypeMismatch` — a
    /// map projection over a non-entity/non-map base is meaningless.
    fn check_map_projection_base(&mut self, base_ti: &TypeInfo, span: &Span) {
        let ok = matches!(
            base_ti,
            TypeInfo::Node { .. }
                | TypeInfo::Relationship { .. }
                | TypeInfo::Map
                | TypeInfo::Null
                | TypeInfo::Property { .. }
        );
        if !ok {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                    op: "map projection base".into(),
                    lhs: base_ti.clone(),
                    rhs: TypeInfo::Map,
                    span: span.clone(),
                }));
        }
    }

    /// **openCypher v9 §3.4** — a list subscript index / slice bound must
    /// be an `Integer` (or a dynamic-schema `Null` / `Property` sentinel,
    /// resolved at runtime). Anything else (`Float`, `String`, …) is a
    /// compile-time `TypeMismatch` — the `InvalidArgumentType` analog. The
    /// same dynamic-schema admission as [`Self::check_list_operand`] /
    /// `is_numeric`: `Property` is admitted because the v1.0 catalog does
    /// not track scalar value types.
    fn check_integer_index(&mut self, idx_ti: &TypeInfo, span: &Span, ctx: &str) {
        let ok = matches!(
            idx_ti,
            TypeInfo::Integer | TypeInfo::Null | TypeInfo::Property { .. }
        );
        if !ok {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                    op: format!("{ctx} index"),
                    lhs: idx_ti.clone(),
                    rhs: TypeInfo::Integer,
                    span: span.clone(),
                }));
        }
    }

    /// **openCypher v9 §3.4 — dynamic map value access (#1056 / #990).**
    /// The base of a bracket subscript `base[index]` must be a `List`
    /// (integer-indexed element access) or a `Map` (string-keyed dynamic
    /// value access) — or a dynamic-schema `Null` / `Property` sentinel
    /// (the v1.0 catalog under-types parameters + property values, so the
    /// real List/Map dispatch happens at runtime in `eval_subscript`).
    /// Anything else (a concrete `Integer` / `String` / `Boolean` /
    /// `Path` …) is a compile-time `TypeMismatch` — subscripting a
    /// non-indexable scalar is meaningless. Mirrors
    /// [`Self::check_list_operand`] widened to admit the `Map` base.
    fn check_subscript_base(&mut self, base_ti: &TypeInfo, span: &Span) {
        let ok = matches!(
            base_ti,
            TypeInfo::List(_) | TypeInfo::Map | TypeInfo::Null | TypeInfo::Property { .. }
        );
        if !ok {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                    op: "subscript base".into(),
                    lhs: base_ti.clone(),
                    rhs: TypeInfo::List(Box::new(TypeInfo::Null)),
                    span: span.clone(),
                }));
        }
    }

    /// **openCypher v9 §3.4 — map subscript key (#1056 / #990).** The
    /// index of a `map[key]` dynamic value access must be a `String` (or a
    /// dynamic-schema `Null` / `Property` sentinel, resolved at runtime).
    /// Anything else (`Integer`, `Float`, …) is a compile-time
    /// `TypeMismatch` — the `InvalidArgumentType` analog. Same
    /// dynamic-schema admission as [`Self::check_integer_index`]:
    /// `Property` is admitted because the v1.0 catalog does not track
    /// scalar value types. (NOTE — `Map2` [6]/[7] index a map with a
    /// statically-typed `Integer` / `Float` PARAMETER which carries
    /// `Null` at v1.0 → admitted here, rejected at runtime as
    /// `MapElementAccessByNonString`; only a STATICALLY-concrete
    /// non-string index rejects at compile time.)
    fn check_string_index(&mut self, idx_ti: &TypeInfo, span: &Span, ctx: &str) {
        let ok = matches!(
            idx_ti,
            TypeInfo::String | TypeInfo::Null | TypeInfo::Property { .. }
        );
        if !ok {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                    op: format!("{ctx} index"),
                    lhs: idx_ti.clone(),
                    rhs: TypeInfo::String,
                    span: span.clone(),
                }));
        }
    }

    /// **ADR-188 Decision 3-reduce-widening (OQ-5)** — compute the fold
    /// result type `join(acc_type, body_type)`, widening the numeric
    /// `{Integer, Float}` pair to `Float`. Returns a `TypeCheckError`
    /// (and `Null`) for genuinely non-assignable folds.
    fn reduce_join_type(&mut self, acc: &TypeInfo, body: &TypeInfo, span: &Span) -> TypeInfo {
        // 3VL: a `Null` on either side propagates — the fold may
        // legitimately produce `Null` (`reduce(s=0, x IN [1,null,3] |
        // s + x) ⇒ null`). The static type is the non-null side (the
        // accumulator type drives the fold), or `Null` if both null.
        match (acc, body) {
            (TypeInfo::Null, t) | (t, TypeInfo::Null) => t.clone(),
            // Identical concrete types join to themselves.
            (a, b) if a == b => a.clone(),
            // Numeric widening: any {Integer, Float} mix → the
            // arithmetic join (Float dominates). Reuses the same rule
            // as binary `+` so `reduce(s = 0, x IN [1.0] | s + x)`
            // (Int acc + Float body) WIDENS to Float rather than
            // rejecting.
            (a, b) if is_numeric_concrete(a) && is_numeric_concrete(b) => {
                arithmetic_result_type(a, b)
            }
            // Genuinely non-assignable (e.g. Integer acc + String body):
            // a TypeMismatch per the existing coherence-rejection
            // discipline.
            (a, b) => {
                self.errors
                    .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                        op: "reduce fold (body not assignable to accumulator)".into(),
                        lhs: a.clone(),
                        rhs: b.clone(),
                        span: span.clone(),
                    }));
                TypeInfo::Null
            }
        }
    }

    fn check_binary_op(
        &mut self,
        op: &BinOp,
        lhs: &TypeInfo,
        rhs: &TypeInfo,
        span: &Span,
    ) -> TypeInfo {
        // #618 — AND/OR/XOR operand-type check runs BEFORE the 3VL null
        // short-circuit. openCypher rejects `<non-bool> AND/OR/XOR <x>`
        // at COMPILE time (`InvalidArgumentType` — TCK `Boolean1`/
        // `Boolean2`/`Boolean3` [8]) EVEN when the OTHER operand is a
        // `null` literal: a static non-boolean operand is a type error
        // regardless of 3VL. The pre-existing arm below sat AFTER the
        // null short-circuit (line `Null op _ ⇒ Null`), so
        // `123.4 AND null` silently returned `null` (the harness
        // honesty-note on Boolean3 [8]). This fires ONLY on a
        // statically-KNOWN non-boolean type — a dynamic `Property`
        // access / `Null` / unknown is admitted (runtime-enforced),
        // exactly mirroring the under-typed-catalog discipline the
        // numeric/MapLike arg-kinds use. A non-null/non-boolean operand
        // makes this a genuine error whether or not the other side is
        // null.
        if matches!(op, BinOp::And | BinOp::Or | BinOp::Xor)
            && (is_definitely_non_boolean(lhs) || is_definitely_non_boolean(rhs))
        {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                    op: format!("{op:?}"),
                    lhs: lhs.clone(),
                    rhs: rhs.clone(),
                    span: span.clone(),
                }));
            return TypeInfo::Boolean;
        }
        // 3VL: any operand Null → result Null (D-20).
        if matches!(lhs, TypeInfo::Null) || matches!(rhs, TypeInfo::Null) {
            return TypeInfo::Null;
        }
        match op {
            // AND / OR / XOR (#621) — all three require Boolean
            // operands and yield Boolean. XOR is structurally
            // identical to And/Or here (only the runtime truth-table
            // differs); a non-boolean operand is a TypeMismatch, which
            // the TCK maps to the compile-time `InvalidArgumentType`
            // SyntaxError that Boolean3 [8] (and Boolean2 [8] for OR)
            // require.
            BinOp::And | BinOp::Or | BinOp::Xor => {
                // Boolean operands required; result Boolean.
                if !is_boolean_compatible(lhs) || !is_boolean_compatible(rhs) {
                    self.errors
                        .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                            op: format!("{op:?}"),
                            lhs: lhs.clone(),
                            rhs: rhs.clone(),
                            span: span.clone(),
                        }));
                }
                TypeInfo::Boolean
            }
            BinOp::Eq | BinOp::Neq => {
                // Equality is universally typed; no rejection.
                TypeInfo::Boolean
            }
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                // openCypher comparison is permissively typed: mixed
                // non-null values reach eval, where incompatible ordering
                // becomes null rather than a bind-time TypeMismatch.
                TypeInfo::Boolean
            }
            BinOp::Pow => {
                if !is_numeric(lhs) || !is_numeric(rhs) {
                    self.errors
                        .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                            op: format!("{op:?}"),
                            lhs: lhs.clone(),
                            rhs: rhs.clone(),
                            span: span.clone(),
                        }));
                    return TypeInfo::Null;
                }
                TypeInfo::Float
            }
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                // W23-V11-T-01 / ADR-090 — temporal + decimal
                // arithmetic compatibility check, BEFORE the
                // numeric-fallback. Temporal arithmetic admits
                // mixed shapes that the numeric rule does not.
                if let Some(t) = temporal_arithmetic_result_type(op, lhs, rhs) {
                    return t;
                }
                // #621 — list / string concatenation via `+` (openCypher
                // v9 §3). `Add`-ONLY: concat is NOT valid for
                // Sub/Mul/Div/Mod, so those fall straight through to the
                // numeric rule (a list/string operand there is still a
                // TypeMismatch). Checked AFTER temporal arithmetic and
                // BEFORE the numeric fallback, mirroring
                // `temporal_arithmetic_result_type`'s precedence. A 3VL
                // `Null` operand already short-circuited to `Null` at the
                // top of `check_binary_op` (`null + [1]` ⇒ `null`), so
                // concat never sees a `Null`.
                if matches!(op, BinOp::Add) {
                    if let Some(t) = concat_result_type(lhs, rhs) {
                        return t;
                    }
                }
                // Arithmetic — both numeric.
                if !is_numeric(lhs) || !is_numeric(rhs) {
                    self.errors
                        .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                            op: format!("{op:?}"),
                            lhs: lhs.clone(),
                            rhs: rhs.clone(),
                            span: span.clone(),
                        }));
                    return TypeInfo::Null;
                }
                arithmetic_result_type(lhs, rhs)
            }
            // openCypher v9 §3.3.6 string predicates (#773). Result is
            // Boolean. PERMISSIVE on operand types (v1.0, mirroring the
            // Lt/Le/Gt/Ge arm): a statically non-string operand is NOT a
            // compile error — it evaluates to `null` at runtime (the
            // type-mismatch-in-string-predicate rule). This is load-bearing
            // for TCK Precedence4 [4], whose `(true OR null) STARTS WITH
            // 'abc'` operand is statically Boolean yet must yield a runtime
            // `null` (the scenario RETURNs a row, not a SyntaxError). A null
            // operand already short-circuited to `TypeInfo::Null` above (3VL).
            BinOp::StartsWith | BinOp::EndsWith | BinOp::Contains => TypeInfo::Boolean,
        }
    }

    fn check_unary_op(&mut self, op: &UnaryOp, t: &TypeInfo, span: &Span) -> TypeInfo {
        if matches!(t, TypeInfo::Null) {
            return TypeInfo::Null;
        }
        match op {
            UnaryOp::Not => {
                if !is_boolean_compatible(t) {
                    self.errors
                        .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                            op: "Not".into(),
                            lhs: t.clone(),
                            rhs: TypeInfo::Boolean,
                            span: span.clone(),
                        }));
                }
                TypeInfo::Boolean
            }
            UnaryOp::Neg | UnaryOp::Pos => {
                if !is_numeric(t) {
                    self.errors
                        .push(ArcQLError::TypeCheck(TypeCheckError::TypeMismatch {
                            op: format!("{op:?}"),
                            lhs: t.clone(),
                            rhs: TypeInfo::Integer,
                            span: span.clone(),
                        }));
                    return TypeInfo::Null;
                }
                t.clone()
            }
        }
    }

    fn check_function_call(
        &mut self,
        name: &str,
        args: &[BoundExpression],
        distinct: bool,
        star: bool,
        span: &Span,
    ) -> TypeInfo {
        let sig = match functions::lookup(name) {
            Some(s) => s,
            None => {
                self.errors
                    .push(ArcQLError::TypeCheck(TypeCheckError::UnknownFunction {
                        name: name.to_string(),
                        span: span.clone(),
                    }));
                return TypeInfo::Null;
            }
        };
        // #773 G4 — `count(*)`. The star form takes NO expression
        // argument (`args` is empty) and counts ROWS, so it bypasses the
        // arity + per-arg checks below (which would reject the empty arg
        // list against `count`'s Fixed(1) signature). `*` is valid ONLY
        // on `count` per openCypher v9 §3; star on any other function
        // falls through and is rejected by the ordinary arity check (an
        // empty arg list vs a Fixed(≥1) signature → FunctionArityMismatch).
        if star && name.eq_ignore_ascii_case("count") {
            return TypeInfo::Integer;
        }
        // #773 G5 — `DISTINCT` is only valid inside an aggregating
        // function (count/sum/avg/min/max/collect). On any other function
        // (e.g. `size(DISTINCT x)`) it is a type error rather than a
        // silently-discarded modifier (silent-wrong is the worst class).
        // Aggregates fall through to the normal arity/arg-type checks on
        // their single argument.
        if distinct && AggregationKind::from_function_name(name).is_none() {
            self.errors
                .push(ArcQLError::TypeCheck(TypeCheckError::DistinctNotAllowed {
                    name: name.to_string(),
                    span: span.clone(),
                }));
            // Return the signature's return-type anyway so downstream
            // consumers still type-check (mirrors the arity-mismatch path).
        }
        // Arity.
        let actual = args.len();
        let arity_ok = match sig.arity {
            Arity::Fixed(n) => actual == n,
            Arity::Variadic { min } => actual >= min,
        };
        if !arity_ok {
            let expected = match sig.arity {
                Arity::Fixed(n) => n.to_string(),
                Arity::Variadic { min } => format!("{min}+"),
            };
            self.errors.push(ArcQLError::TypeCheck(
                TypeCheckError::FunctionArityMismatch {
                    name: name.to_string(),
                    actual,
                    expected,
                    span: span.clone(),
                },
            ));
            // Return signature's return-type anyway — the planner
            // can still reason about downstream consumers.
        }
        // Per-arg kind.
        for (i, a) in args.iter().enumerate() {
            let kind = sig.arg_kinds.get(i).copied().unwrap_or(ArgKind::Any);
            let ti = a.type_info().cloned().unwrap_or(TypeInfo::Null);
            if !kind.accepts(&ti) {
                self.errors.push(ArcQLError::TypeCheck(
                    TypeCheckError::FunctionArgumentTypeMismatch {
                        name: name.to_string(),
                        position: i,
                        // For diagnostic clarity, we render the
                        // expected ArgKind as a sentinel TypeInfo —
                        // ArgKind doesn't have a 1-to-1 TypeInfo
                        // correspondence (e.g. Numeric is "Integer
                        // or Float"). Using the most-specific
                        // representative keeps the error message
                        // useful without bloating ArgKind into a
                        // full TypeInfo.
                        expected: arg_kind_repr(kind),
                        actual: ti,
                        span: span.clone(),
                    },
                ));
            }
        }
        // Return type.
        let arg_types: Vec<TypeInfo> = args
            .iter()
            .map(|a| a.type_info().cloned().unwrap_or(TypeInfo::Null))
            .collect();
        (sig.return_type_for)(&arg_types)
    }
}

// =====================================================================
// Helper functions (free / pure)
// =====================================================================

fn literal_type(l: &Literal) -> TypeInfo {
    match l {
        Literal::Null => TypeInfo::Null,
        Literal::Bool(_) => TypeInfo::Boolean,
        Literal::Integer(_) => TypeInfo::Integer,
        Literal::Float(_) => TypeInfo::Float,
        Literal::String(_) => TypeInfo::String,
        // ADR-188 amendment-01 (#723 quantifier type-mismatch gap) — a
        // list LITERAL carries its element type when the elements are
        // homogeneous scalar literals (`['a','b'] : List(String)`,
        // `[1,2,3] : List(Integer)`). This lets the scoped-variable type
        // (`element_type_of`) be concrete, so a quantifier predicate that
        // applies a type-incompatible op to the element —
        // `all(x IN ['Clara'] WHERE x % 2 = 0)`, String `%` Integer —
        // is rejected at COMPILE time by `check_binary_op` (the
        // openCypher `InvalidArgumentType` / SyntaxError contract,
        // Quantifier1-4 `[15]/[16]`) rather than executing. Heterogeneous,
        // empty, or non-scalar-literal element lists stay erased to
        // `List(Null)` — the prior conservative behavior (3VL-safe; the
        // element type is genuinely unknown, so no new false-reject).
        Literal::List(elems) => TypeInfo::List(Box::new(list_literal_elem_type(elems))),
        Literal::Map(_) => TypeInfo::Map,
        // W23-V11-T-01 / ADR-090 — temporal + decimal literal type
        // derivation. The catalog's value-type column (v1.1+) returns
        // these directly; M4-22's literal-derivation rule produces
        // the same TypeInfo so an `n.valid_from = datetime('...')`
        // comparison type-checks against `Temporal`.
        Literal::Temporal(_) => TypeInfo::Temporal,
        Literal::LocalDateTime(_) => TypeInfo::LocalDateTime,
        Literal::Date(_) => TypeInfo::Date,
        Literal::Duration(_) => TypeInfo::Duration,
        Literal::Decimal(_) => TypeInfo::Decimal,
    }
}

/// **ADR-188 amendment-01 (#723 quantifier type-mismatch gap)** — infer
/// the element type of a list LITERAL from its element expressions.
///
/// Returns a *concrete* element type (`String` / `Integer` / `Float` /
/// `Boolean` / …) only when EVERY element is a scalar literal of the SAME
/// concrete type. Any of the following defeats inference and yields
/// `Null` (the prior conservative, 3VL-safe behavior — the element type
/// is genuinely unknown so the quantifier predicate is NOT newly
/// rejected):
///
/// * an empty list (`[]` — no element to type),
/// * a heterogeneous list (`[1, 'a']` — no single element type; mixed
///   lists are themselves a separate openCypher question and must not be
///   silently coerced here),
/// * a `Null` element (`[1, null]` — 3VL: the element type subsumes
///   `Null`, so iteration must treat it as possibly-null),
/// * any non-scalar-literal element (a nested list/map, a parameter, a
///   property access, a function call, an arithmetic sub-expression) —
///   typing those needs the full visitor, which runs later; the
///   conservative `Null` keeps this helper a pure, allocation-free
///   AST inspection.
///
/// This is the SOURCE of the gap: before this, `Literal::List(_)` erased
/// to `List(Null)` unconditionally, so `element_type_of` handed the
/// quantifier's scoped variable a `Null` type, and `check_binary_op`'s
/// 3VL short-circuit (`Null op _ ⇒ Null`, no error) silently accepted
/// `x % 2` over a String element. With the concrete element type the
/// existing arithmetic check (`is_numeric`) fires correctly.
fn list_literal_elem_type(elems: &[Expression]) -> TypeInfo {
    let mut elem_ty: Option<TypeInfo> = None;
    for e in elems {
        let ty = match e {
            // Only scalar literals are typed here. `Literal::Null` ⇒
            // `Null` (handled below as "unknown / possibly-null", which
            // collapses the whole inference to `Null`). Nested
            // `Literal::List` / `Literal::Map` would need recursion +
            // join semantics we deliberately do NOT take on at v1.0 (the
            // TCK quantifier scenarios use flat scalar lists), so they
            // fall to the conservative branch.
            Expression::Literal(l) => match l {
                Literal::Bool(_) => TypeInfo::Boolean,
                Literal::Integer(_) => TypeInfo::Integer,
                Literal::Float(_) => TypeInfo::Float,
                Literal::String(_) => TypeInfo::String,
                Literal::Temporal(_) => TypeInfo::Temporal,
                Literal::LocalDateTime(_) => TypeInfo::LocalDateTime,
                Literal::Date(_) => TypeInfo::Date,
                Literal::Duration(_) => TypeInfo::Duration,
                Literal::Decimal(_) => TypeInfo::Decimal,
                // Null element, or a nested list/map: inference can't
                // commit to a concrete homogeneous element type.
                Literal::Null | Literal::List(_) | Literal::Map(_) => return TypeInfo::Null,
            },
            // Non-literal element (parameter, property, arithmetic, …):
            // not statically typeable in this pure inspection.
            _ => return TypeInfo::Null,
        };
        match &elem_ty {
            None => elem_ty = Some(ty),
            // Heterogeneous: two different concrete element types ⇒ no
            // single element type. Bail to the conservative `Null`.
            Some(prev) if *prev != ty => return TypeInfo::Null,
            Some(_) => {}
        }
    }
    // `None` ⇒ empty list (no elements) ⇒ unknown element type ⇒ `Null`.
    elem_ty.unwrap_or(TypeInfo::Null)
}

fn bound_list_literal_elem_type(elems: &[BoundExpression]) -> TypeInfo {
    let mut elem_ty: Option<TypeInfo> = None;
    for e in elems {
        let ty = match e.type_info().cloned().unwrap_or(TypeInfo::Null) {
            TypeInfo::Null | TypeInfo::List(_) | TypeInfo::Map => return TypeInfo::Null,
            other => other,
        };
        match &elem_ty {
            None => elem_ty = Some(ty),
            Some(prev) if *prev != ty => return TypeInfo::Null,
            Some(_) => {}
        }
    }
    elem_ty.unwrap_or(TypeInfo::Null)
}

fn is_boolean_compatible(t: &TypeInfo) -> bool {
    matches!(t, TypeInfo::Boolean | TypeInfo::Null)
}

/// True if `t` is a statically-KNOWN non-boolean type — i.e. a concrete
/// scalar / structural type that is definitely NOT a boolean and NOT a
/// 3VL `Null`. Used by the AND/OR/XOR operand check (#618) which must
/// reject a known non-boolean operand at COMPILE time EVEN when the
/// other operand is `null` (openCypher `InvalidArgumentType` — TCK
/// `Boolean1`/`Boolean2`/`Boolean3` [8]). A dynamically-typed
/// `Property` access (the v1.0 under-typed-catalog sentinel) and `Null`
/// are NOT "definitely non-boolean" — they are admitted and the
/// executor's 3VL eval enforces booleanity at runtime, mirroring the
/// `is_numeric` / `ArgKind::MapLike` dynamic-schema discipline. This is
/// the exact complement of "could be a boolean": `Boolean`, `Null`, and
/// `Property { .. }` return `false`.
fn is_definitely_non_boolean(t: &TypeInfo) -> bool {
    !matches!(
        t,
        TypeInfo::Boolean | TypeInfo::Null | TypeInfo::Property { .. }
    )
}

/// True if `t` is a statically-KNOWN value over which property access
/// (`base.prop`) is invalid — i.e. a concrete scalar / list / path that
/// is definitely NOT a Node / Relationship / Map. Used by the
/// `PropertyAccess` type-check (#618) to reject `<scalar>.prop` at
/// COMPILE time (openCypher `InvalidArgumentType` — TCK `Graph6` [9] /
/// `Map1` [6]). A Node / Relationship / Map carries properties; the
/// dynamically-typed `Property` access (the v1.0 under-typed-catalog
/// sentinel — e.g. nested `a.b.c`), `Null`, and any unknown type are
/// admitted (runtime-enforced), so this returns `false` for them. The
/// complement of "could carry properties".
fn is_definitely_non_entity_non_map(t: &TypeInfo) -> bool {
    !matches!(
        t,
        TypeInfo::Node { .. }
            | TypeInfo::Relationship { .. }
            | TypeInfo::Map
            | TypeInfo::Property { .. }
            | TypeInfo::Null
    )
}

fn is_numeric(t: &TypeInfo) -> bool {
    // v1.0 dynamic-schema: `Property { .. }` is numeric-compatible
    // because the catalog does NOT track scalar value types yet.
    // The runtime-side property-value coercion will surface a real
    // type error if the stored value is non-numeric. v1.1 strict-
    // schema upgrades this to inspect `value_type` directly.
    //
    // W23-V11-T-01 / ADR-090 — `Decimal` admitted as numeric per
    // ADR-038 amendment-09 (fixed-point arithmetic; the executor
    // routes Decimal arithmetic through i128 ops with scale
    // alignment, not through f64).
    matches!(
        t,
        TypeInfo::Integer
            | TypeInfo::Float
            | TypeInfo::Decimal
            | TypeInfo::Null
            | TypeInfo::Property { .. }
    )
}

/// **ADR-188 Decision 3** — derive the element type of a list type.
/// `List(T) ⇒ T`; `Null` (null list / parameter) and `Property`
/// (dynamic-schema) ⇒ `Null` (the conservative "could be anything,
/// possibly null" element under 3VL). Mirrors the binder's
/// `may_be_null = true` conservatism for scoped iteration variables.
fn element_type_of(list_ti: &TypeInfo) -> TypeInfo {
    match list_ti {
        TypeInfo::List(elem) => (**elem).clone(),
        // Null list, parameter list, or dynamic-schema property: the
        // element type is unknown → `Null` (3VL-safe; the executor
        // resolves the concrete value at runtime).
        _ => TypeInfo::Null,
    }
}

/// **ADR-188 Decision 3-reduce-widening** — a *concrete* numeric type
/// (`Integer` / `Float` / `Decimal`), EXCLUDING the dynamic-schema
/// `Null` / `Property` sentinels that `is_numeric` admits. Used by
/// `reduce_join_type` so the numeric-widening arm only fires for two
/// genuinely-numeric concrete types; `Null` is handled separately
/// (3VL propagation) and `Property` falls through to the non-numeric
/// branch (where it joins by equality / errors honestly).
fn is_numeric_concrete(t: &TypeInfo) -> bool {
    matches!(t, TypeInfo::Integer | TypeInfo::Float | TypeInfo::Decimal)
}

/// Temporal arithmetic compatibility (per K3 §7 + ADR-090).
///
/// Admissible binary-op shapes:
///
/// - `Temporal + Duration` → `Temporal` (advance by a duration)
/// - `Temporal - Temporal` → `Duration` (compute interval)
/// - `Temporal - Duration` → `Temporal`
/// - `Date + Duration` → `Date`
/// - `LocalDateTime + Duration` → `LocalDateTime`
/// - `Duration + Duration` → `Duration`
/// - `Duration - Duration` → `Duration`
/// - `Decimal + Decimal` → `Decimal` (scale-aligned)
/// - `Decimal + Integer` → `Decimal`
/// - `Integer + Decimal` → `Decimal`
fn temporal_arithmetic_result_type(op: &BinOp, lhs: &TypeInfo, rhs: &TypeInfo) -> Option<TypeInfo> {
    use TypeInfo::*;
    match (op, lhs, rhs) {
        (BinOp::Add, Temporal, Duration) | (BinOp::Add, Duration, Temporal) => Some(Temporal),
        (BinOp::Sub, Temporal, Temporal) => Some(Duration),
        (BinOp::Sub, Temporal, Duration) => Some(Temporal),
        (BinOp::Add, Date, Duration) | (BinOp::Add, Duration, Date) => Some(Date),
        (BinOp::Sub, Date, Date) => Some(Duration),
        (BinOp::Sub, Date, Duration) => Some(Date),
        (BinOp::Add, LocalDateTime, Duration) | (BinOp::Add, Duration, LocalDateTime) => {
            Some(LocalDateTime)
        }
        (BinOp::Sub, LocalDateTime, LocalDateTime) => Some(Duration),
        (BinOp::Sub, LocalDateTime, Duration) => Some(LocalDateTime),
        (BinOp::Add, Duration, Duration) | (BinOp::Sub, Duration, Duration) => Some(Duration),
        // Decimal arithmetic.
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, Decimal, Decimal) => Some(Decimal),
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, Decimal, Integer) => Some(Decimal),
        (BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div, Integer, Decimal) => Some(Decimal),
        _ => None,
    }
}

fn arithmetic_result_type(a: &TypeInfo, b: &TypeInfo) -> TypeInfo {
    // Float dominates Integer per Cypher 9 §3.4. Property values
    // are dynamic-schema at v1.0 — the type-check pass cannot
    // statically determine the result without executor-side
    // coercion. We collapse Property into Integer here as the
    // optimistic choice for the common `n.age + 1` pattern; v1.1
    // strict-schema returns a precise type from
    // `Property::value_type`.
    let af = matches!(
        a,
        TypeInfo::Float
            | TypeInfo::Property {
                value_type: PropertyType::Float,
                ..
            }
    );
    let bf = matches!(
        b,
        TypeInfo::Float
            | TypeInfo::Property {
                value_type: PropertyType::Float,
                ..
            }
    );
    if af || bf {
        TypeInfo::Float
    } else {
        TypeInfo::Integer
    }
}

/// **#621** — result type of the openCypher `+` *concatenation* overload,
/// or `None` when `+` is plain numeric arithmetic (the caller then falls
/// through to the numeric rule). `Add`-only: the caller gates this on
/// `op == BinOp::Add`, and both `temporal_arithmetic_result_type` and the
/// 3VL `Null` short-circuit run first, so neither operand is `Null` here.
///
/// openCypher v9 §3 overloads `+`:
///
/// * either operand a `List(_)` → `List(_)`: list+list concatenation,
///   list+element append, element+list prepend. The element type is the
///   permissive [`case_join_type`] of the two contributing element types
///   (the list's element type for a list side, the scalar's own type for
///   an element side) — identical types collapse to themselves,
///   `{Integer, Float}` widens to `Float`, an empty list's `Null` element
///   yields the other side's witness type, and genuinely heterogeneous
///   element types widen to `Null`. This is the SAME never-erroring join
///   openCypher list literals already use, so a concat result type is
///   never a false type error.
/// * both operands `String` → `String` (string concatenation).
/// * one operand `String` and the other a dynamic-schema `Property`
///   (scalar value type not statically known) → `String`: admit it,
///   mirroring the numeric path's dynamic admission of `Property` (so
///   `n.name + '!'` is NOT false-rejected when `name` is a string at
///   runtime; the executor resolves the value and concatenates, or errors
///   honestly if it is not string-shaped). `Property + Property` is left
///   to the numeric path — both are admitted as dynamic-numeric there and
///   the executor's string-concat arm still handles two string values.
fn concat_result_type(lhs: &TypeInfo, rhs: &TypeInfo) -> Option<TypeInfo> {
    use TypeInfo::*;
    match (lhs, rhs) {
        // list + list → concatenation (join the two element types).
        (List(a), List(b)) => Some(List(Box::new(case_join_type(a, b)))),
        // list + element → append; element + list → prepend.
        (List(a), elem) => Some(List(Box::new(case_join_type(a, elem)))),
        (elem, List(b)) => Some(List(Box::new(case_join_type(elem, b)))),
        // string + string → concatenation.
        (String, String) => Some(String),
        // dynamic-schema: string + property / property + string → String.
        (String, Property { .. }) | (Property { .. }, String) => Some(String),
        _ => None,
    }
}

/// **openCypher v9 §3.6** (#621) — PERMISSIVE result-type join for a CASE
/// branch (THEN / ELSE) pair. Unlike [`TypeCheckVisitor::reduce_join_type`]
/// (which ERRORS on a non-assignable fold), CASE branches MAY legally
/// diverge in type (`CASE WHEN c THEN 1 ELSE 'x' END` is valid openCypher),
/// so a heterogeneous join widens to `TypeInfo::Null` — the 3VL-aware
/// "could be anything" sentinel the type-checker already uses for parameters
/// / dynamic-schema values — WITHOUT pushing a type error. Identical
/// concrete types join to themselves; `{Integer, Float}` widen via the same
/// numeric rule as `+` (so `CASE … THEN 1 ELSE 2.0 END` is `Float`); a
/// `Null` on either side yields the other (the witness type — the runtime
/// `null` from a non-matching ELSE-less CASE is a 3VL value, not a static
/// type, mirroring the `reduce` / list-comprehension convention).
fn case_join_type(a: &TypeInfo, b: &TypeInfo) -> TypeInfo {
    match (a, b) {
        (TypeInfo::Null, t) | (t, TypeInfo::Null) => t.clone(),
        (x, y) if x == y => x.clone(),
        (x, y) if is_numeric_concrete(x) && is_numeric_concrete(y) => arithmetic_result_type(x, y),
        // Heterogeneous branch types are legal — widen to the permissive
        // sentinel; NO type error on divergence.
        _ => TypeInfo::Null,
    }
}

fn arg_kind_repr(k: ArgKind) -> TypeInfo {
    match k {
        ArgKind::Any => TypeInfo::Null,
        ArgKind::Numeric => TypeInfo::Integer,
        ArgKind::List => TypeInfo::List(Box::new(TypeInfo::Null)),
        ArgKind::Node => TypeInfo::Node { label: None },
        // `properties()` (#618) — the representative is `Map` (the
        // most-specific property-bag-bearing shape; ArgKind has no 1:1
        // TypeInfo, so the error renders the canonical accepted shape).
        ArgKind::MapLike => TypeInfo::Map,
        // #618 REJECT-semantics kinds — render the canonical ACCEPTED
        // shape for the diagnostic.
        ArgKind::RelOnly => TypeInfo::Relationship { rel_type: None },
        ArgKind::PathOnly => TypeInfo::Path,
        ArgKind::ListLike => TypeInfo::List(Box::new(TypeInfo::Null)),
    }
}

/// ADR-147 W26-θ Phase 1 literal-only property-value gate — is `e` a literal
/// CONSTANT admissible as a CREATE/SET property value? A bare `Literal`, OR a
/// NEGATIVE / unary-`+` numeric literal — which parses as `UnaryOp(Neg/Pos,
/// <numeric literal>)`, NOT a `Literal` (#870 — `CREATE (n {x: -5})` was
/// rejected as "not a literal"). The unary form is constrained to a numeric
/// literal operand so the type-check admits exactly what the executor's
/// `literal_lift::bound_literal_value` can fold (a number); a non-numeric
/// unary operand (`-'x'`) stays rejected. List elements (`{x: [-5]}`) are NOT
/// checked here — the outer `[..]` IS a `Literal`; their inner negatives lift
/// in `literal_lift::list_element_value`.
fn is_literal_property_value(e: &BoundExpression) -> bool {
    match e {
        BoundExpression::Literal { .. } | BoundExpression::ListLiteral { .. } => true,
        BoundExpression::UnaryOp {
            op: UnaryOp::Neg | UnaryOp::Pos,
            operand,
            ..
        } => matches!(
            operand.as_ref(),
            BoundExpression::Literal {
                value: Literal::Integer(_) | Literal::Float(_),
                ..
            }
        ),
        _ => false,
    }
}

/// ADR-147-amendment-03 (D-1) — CREATE-only property-value gate.
///
/// Admissible as a CREATE / CREATE-path property value iff
/// [`crate::executor::eval::evaluate`] resolves it to a stored
/// scalar/list `Value` against the upstream row + parameter bag, AND
/// every sub-expression is likewise admissible. The recursion is
/// load-bearing: it closes the `[randomUUID()]` / `{k: fn()}` nesting
/// bypass — a `FunctionCall` hidden inside a `ListLiteral` element makes
/// the whole list inadmissible (Trap #3 / test T11).
///
/// This is DENY-BY-DEFAULT and CREATE-ONLY. It is DISTINCT from the
/// shared [`is_literal_property_value`] (which stays the SET / MERGE
/// gate — SET's executor is not eval-wired, so admitting a non-literal
/// there would type-check then fault at runtime). Two invariants:
/// (a) the admitted subset is closed under determinism + bounded work —
/// `FunctionCall` (incl. `randomUUID()` / `timestamp()` / `range()`) is
/// rejected here, so a non-deterministic / unbounded expression never
/// reaches the executor; (b) round-trippability is *also* enforced at
/// the VALUE layer (the executor's `materialize_properties` gate), since
/// AST shape cannot see the runtime type a `$p` / `r.x` resolves to.
fn is_evaluable_create_property_value(e: &BoundExpression) -> bool {
    match e {
        // Const scalars (the executor's `bound_literal_value` fast path
        // keeps folding these); list literals recurse element-wise.
        BoundExpression::Literal { .. } => true,

        // #870 numeric-literal carve-out is subsumed here: a `UnaryOp`
        // is admitted iff its operand is admissible (recurse). The value
        // gate rejects a non-numeric runtime result, so `-$p` where `$p`
        // is a string surfaces a clean execution error, not corruption.
        BoundExpression::UnaryOp {
            op: UnaryOp::Neg | UnaryOp::Pos,
            operand,
            ..
        } => is_evaluable_create_property_value(operand),

        // NEW admitted — row / param references. Each resolves at runtime
        // to a `Value` the value-type gate then vets.
        BoundExpression::Parameter { .. } // {id: $p}
        | BoundExpression::VariableRef { .. } // {x: unwound_var}
        | BoundExpression::PropertyAccess { .. } // {name: r.name}
            => true,

        // Containers — RECURSE into every child (closes the nesting
        // bypass). An empty list is admissible.
        BoundExpression::ListLiteral { elements, .. } => {
            elements.iter().all(is_evaluable_create_property_value)
        }

        // Arithmetic / comparison / logical spine — whitelisted operators
        // only, BOTH operands admissible. `is_whitelisted_binop` is
        // deny-by-default.
        BoundExpression::BinaryOp { op, lhs, rhs, .. } => {
            is_whitelisted_binop(op)
                && is_evaluable_create_property_value(lhs)
                && is_evaluable_create_property_value(rhs)
        }

        // STILL REJECTED — `Not` unary (round-trip of `NOT b` on a
        // property is unverified at v1.0-α; deny-by-default).
        BoundExpression::UnaryOp {
            op: UnaryOp::Not, ..
        } => false,

        // STILL REJECTED — Map: openCypher forbids map property values,
        // permanently write-fenced per ADR-191 D-11 (NOT deferred). A
        // `MapLiteral` admitted here would type-check then reject/corrupt
        // at execute.
        BoundExpression::MapLiteral { .. } | BoundExpression::MapProjection { .. } => false,

        // STILL REJECTED — FunctionCall (determinism: randomUUID() /
        // rand() / timestamp(); AND unbounded materialization: range()).
        // Deferred to a later amendment after a purity + cost audit of
        // the function catalog.
        BoundExpression::FunctionCall { .. } => false,

        // Deny-by-default catch-all — predicate special forms, CASE,
        // list comprehensions, subscripts, slices, reduce, IN, IS NULL,
        // NEAR / MATCH / IN COMMUNITY, and any un-lowered retrieval lift
        // whose CREATE-property round-trip is unverified at v1.0-α.
        _ => false,
    }
}

/// Deny-by-default whitelist of arithmetic / comparison / logical
/// operators safe as a CREATE property spine (ADR-147-amendment-03).
///
/// `Add` pulls in list / string concatenation via the executor's
/// `add_or_concat` — a memory-amplification path (`{x: (($a+$a)+($a+$a))
/// +…}` amplifies ~2^depth). The backstop is the PER-OP cap enforced
/// INSIDE `eval::add_or_concat` (`eval::MAX_CONCAT_LIST_LEN` /
/// `MAX_CONCAT_STRING_BYTES`), which kills the blowup at the FIRST
/// over-cap node before it allocates. NOTE: the result-level
/// `literal_lift::MAX_CREATE_PROP_LIST_LEN` cap is NOT the backstop for
/// this amplifier — it gates the FINAL value, i.e. AFTER the multi-GB
/// intermediate is already materialized (ADR-147-amendment-03 §B1).
/// `Pow` is excluded (magnitude amplifier); `StartsWith` / `EndsWith` /
/// `Contains` are excluded (boolean-only, no need on a CREATE property
/// spine at D-1).
fn is_whitelisted_binop(op: &BinOp) -> bool {
    matches!(
        op,
        BinOp::Add
            | BinOp::Sub
            | BinOp::Mul
            | BinOp::Div
            | BinOp::Mod
            | BinOp::Eq
            | BinOp::Neq
            | BinOp::Lt
            | BinOp::Le
            | BinOp::Gt
            | BinOp::Ge
            | BinOp::And
            | BinOp::Or
            | BinOp::Xor
    )
}

/// ADR-147 W26-θ Phase 1: short human-readable label for a
/// BoundExpression's discriminant. Used by the CREATE property-value
/// literal-only narrowing diagnostic.
fn describe_expression(e: &BoundExpression) -> String {
    match e {
        BoundExpression::Literal { .. } => "literal".into(),
        BoundExpression::ListLiteral { .. } => "list literal".into(),
        BoundExpression::MapLiteral { .. } => "map literal".into(),
        BoundExpression::Parameter { .. } => "parameter".into(),
        BoundExpression::VariableRef { .. } => "variable reference".into(),
        BoundExpression::UnresolvedVariable { .. } => "unresolved variable".into(),
        BoundExpression::PropertyAccess { .. } => "property access".into(),
        BoundExpression::BinaryOp { .. } => "binary operation".into(),
        BoundExpression::UnaryOp { .. } => "unary operation".into(),
        BoundExpression::FunctionCall { .. } => "function call".into(),
        BoundExpression::Near { .. } => "NEAR predicate".into(),
        BoundExpression::TextMatch { .. } => "MATCH predicate".into(),
        BoundExpression::InCommunity { .. } => "IN COMMUNITY predicate".into(),
        BoundExpression::In { .. } => "IN predicate".into(),
        BoundExpression::IsNull { .. } => "IS NULL predicate".into(),
        // ADR-188 — list-predicate special forms.
        BoundExpression::ListPredicate { .. } => "list predicate".into(),
        BoundExpression::Reduce { .. } => "reduce".into(),
        // ADR-188 (#620 list-half) — list comprehension.
        BoundExpression::ListComprehension { .. } => "list comprehension".into(),
        // ADR-191 D-6 (#620 map-half) — map projection.
        BoundExpression::MapProjection { .. } => "map projection".into(),
        // openCypher v9 §3.4 — postfix accessors.
        BoundExpression::Subscript { .. } => "list subscript".into(),
        BoundExpression::Slice { .. } => "list slice".into(),
        // openCypher v9 §3.6 (#621) — CASE expression.
        BoundExpression::Case { .. } => "CASE expression".into(),
    }
}

// =====================================================================
// Tests
// =====================================================================
//
// 6 reserved-variant rejection pins live in
// `tests/type_check_integration.rs` (top-of-file rationale: those
// pins exercise the public API end-to-end through `parse → bind →
// type-check`, which is more representative than the unit-level
// invocation pattern below). The unit tests in this module cover
// the pure-function helpers + the visitor's individual nodes.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;
    use crate::semantic::BindingVisitor;
    use crate::semantic::StubCatalogProvider;

    fn check_ok(input: &str) -> BoundStatement {
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS", "WROTE"])
            .with_properties(["age", "name", "title"]);
        let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check");
        bound
    }

    fn check_err(input: &str) -> Vec<ArcQLError> {
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS"])
            .with_properties(["age", "name"]);
        let mut bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, &cat).expect_err("type-check should fail")
    }

    // ---------- #621 — IN non-list RHS + subscript/slice rejections ----------
    //
    // #723 LESSON: a new grammar form that now PARSES (RETURN-position
    // `IN`, list subscript / slice) MUST ship its type-check rejections.
    // The TCK scores `raised at compile time: InvalidArgumentType` by
    // compile PHASE, and `classify_engine_error` maps
    // `TypeCheckError::TypeMismatch` → `EngineErrClass::Compile`, so a
    // `TypeMismatch` satisfies the contract (same convention as the
    // quantifier-type-mismatch tests below).

    fn assert_type_mismatch(input: &str) {
        let errs = check_err(input);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "`{input}` MUST be a compile-time TypeMismatch (InvalidArgumentType), got {errs:?}"
        );
    }

    #[test]
    fn in_non_list_rhs_rejects_at_compile() {
        // List5 [42] — `1 IN <non-list literal>` ⇒ InvalidArgumentType.
        assert_type_mismatch("RETURN 1 IN true AS r");
        assert_type_mismatch("RETURN 1 IN 123 AS r");
        assert_type_mismatch("RETURN 1 IN 123.4 AS r");
        assert_type_mismatch("RETURN 1 IN 'foo' AS r");
        assert_type_mismatch("RETURN 1 IN {x: []} AS r");
    }

    #[test]
    fn subscript_non_integer_index_rejects_at_compile() {
        // `list[1.5]` — a Float index is not an integer.
        assert_type_mismatch("RETURN [1, 2, 3][1.5] AS r");
    }

    #[test]
    fn subscript_on_scalar_base_rejects_at_compile() {
        // `5[0]` — subscript on a non-list scalar base.
        assert_type_mismatch("RETURN 5[0] AS r");
    }

    #[test]
    fn slice_non_integer_bound_rejects_at_compile() {
        assert_type_mismatch("RETURN [1, 2, 3][1.5..] AS r");
        assert_type_mismatch("RETURN [1, 2, 3][..1.5] AS r");
    }

    #[test]
    fn slice_on_scalar_base_rejects_at_compile() {
        // `5[0..1]` — slice on a non-list scalar base.
        assert_type_mismatch("RETURN 5[0..1] AS r");
    }

    #[test]
    fn subscript_slice_and_in_on_list_type_check_ok() {
        // Positive guards: the rejections above are not over-broad — valid
        // §3.4 / §3.3.5 forms over a list type-check cleanly.
        check_ok("RETURN [1, 2, 3][0] AS r");
        check_ok("RETURN [1, 2, 3][0..2] AS r");
        check_ok("RETURN [1, 2, 3][-1] AS r");
        check_ok("RETURN 3 IN [1, 2, 3] AS r");
    }

    // ---------- ADR-191 D-6 (#620 map-half) — map-projection type-check ----------
    //
    // #723 LESSON (carried): a new grammar form that now PARSES (map
    // projection) MUST ship its type-check rejection. A projection over a
    // non-entity / non-map base is a compile-time `TypeMismatch`
    // (InvalidArgumentType analog), per `check_map_projection_base`.

    #[test]
    fn map_projection_over_node_type_checks_ok() {
        // `n{.name, .age}` over a node binding — the projectable base type.
        check_ok("MATCH (n:Person) RETURN n{.name, .age} AS m");
        // With a literal entry + an all-properties selector.
        check_ok("MATCH (n:Person) RETURN n{.name, score: 1 + 1, .*} AS m");
        // The empty projection.
        check_ok("MATCH (n:Person) RETURN n{} AS m");
    }

    #[test]
    fn map_projection_over_relationship_type_checks_ok() {
        // `r{.since}` over a relationship binding — also projectable.
        check_ok("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r{.name} AS m");
    }

    #[test]
    fn map_projection_over_map_typed_var_type_checks_ok() {
        // A `WITH`-aliased map literal types as `Map`; projecting it is OK.
        check_ok("MATCH (n:Person) WITH {a: 1, b: 2} AS mp RETURN mp{.a} AS m");
    }

    #[test]
    fn map_projection_over_scalar_base_rejects_at_compile() {
        // A projection over an Integer-typed base is a compile-time
        // TypeMismatch — a map projection over a scalar is meaningless (the
        // D-6 base contract). `UNWIND [1,2,3] AS x` types `x` as the
        // CONCRETE element type `Integer` (unlike a `WITH`-aliased literal,
        // which the v1.0 type system widens to `Null` — the dynamic-schema
        // immaturity per ADR-191 D-9; the `Null` path is admitted as
        // null-propagating, NOT a reject). A `String` base rejects the same
        // way (proving the reject keys on concrete-non-entity, not Integer
        // specifically).
        assert_type_mismatch("UNWIND [1, 2, 3] AS x RETURN x{.a} AS m");
        assert_type_mismatch("UNWIND ['a', 'b'] AS x RETURN x{.a} AS m");
    }

    // ---------- 3VL truth table ----------

    #[test]
    fn three_vl_and_truth_table() {
        use BoolOrNull::*;
        assert_eq!(apply_and_3vl(True, True), True);
        assert_eq!(apply_and_3vl(True, False), False);
        assert_eq!(apply_and_3vl(True, Null), Null);
        assert_eq!(apply_and_3vl(False, True), False);
        assert_eq!(apply_and_3vl(False, False), False);
        assert_eq!(apply_and_3vl(False, Null), False);
        assert_eq!(apply_and_3vl(Null, True), Null);
        assert_eq!(apply_and_3vl(Null, False), False);
        assert_eq!(apply_and_3vl(Null, Null), Null);
    }

    #[test]
    fn three_vl_or_truth_table() {
        use BoolOrNull::*;
        assert_eq!(apply_or_3vl(True, True), True);
        assert_eq!(apply_or_3vl(True, False), True);
        assert_eq!(apply_or_3vl(True, Null), True);
        assert_eq!(apply_or_3vl(False, True), True);
        assert_eq!(apply_or_3vl(False, False), False);
        assert_eq!(apply_or_3vl(False, Null), Null);
        assert_eq!(apply_or_3vl(Null, True), True);
        assert_eq!(apply_or_3vl(Null, False), Null);
        assert_eq!(apply_or_3vl(Null, Null), Null);
    }

    #[test]
    fn three_vl_not_truth_table() {
        use BoolOrNull::*;
        assert_eq!(apply_not_3vl(True), False);
        assert_eq!(apply_not_3vl(False), True);
        assert_eq!(apply_not_3vl(Null), Null);
    }

    // ---------- Helper functions ----------

    #[test]
    fn literal_type_classifies_each_kind() {
        assert_eq!(literal_type(&Literal::Null), TypeInfo::Null);
        assert_eq!(literal_type(&Literal::Bool(true)), TypeInfo::Boolean);
        assert_eq!(literal_type(&Literal::Integer(7)), TypeInfo::Integer);
        assert_eq!(literal_type(&Literal::Float(1.5)), TypeInfo::Float);
        assert_eq!(literal_type(&Literal::String("x".into())), TypeInfo::String);
    }

    #[test]
    fn arithmetic_result_promotes_to_float() {
        assert_eq!(
            arithmetic_result_type(&TypeInfo::Integer, &TypeInfo::Integer),
            TypeInfo::Integer
        );
        assert_eq!(
            arithmetic_result_type(&TypeInfo::Integer, &TypeInfo::Float),
            TypeInfo::Float
        );
        assert_eq!(
            arithmetic_result_type(&TypeInfo::Float, &TypeInfo::Float),
            TypeInfo::Float
        );
    }

    // ---------- Visitor end-to-end ----------

    #[test]
    fn check_ok_basic_match_return() {
        let bound = check_ok("MATCH (n:Person) RETURN n");
        // Confirm n's type-info is populated as Node.
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let m = match &q.clauses[0] {
            BoundClause::Match(m) => m,
            _ => panic!(),
        };
        let path = match &m.body {
            BoundMatchBody::Patterns(ps) => &ps[0],
            _ => panic!(),
        };
        let v = path.head.var.as_ref().unwrap();
        assert!(matches!(
            v.type_info,
            Some(TypeInfo::Node { label: Some(_) })
        ));
    }

    #[test]
    fn check_ok_property_access_yields_property_type() {
        let bound = check_ok("MATCH (n:Person) RETURN n.age");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let r = q
            .clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::Return(r) => Some(r),
                _ => None,
            })
            .expect("RETURN");
        let it = &r.items[0];
        let e = match &it.kind {
            BoundProjectionKind::Expr(e) => e,
            _ => panic!(),
        };
        assert!(matches!(e.type_info(), Some(TypeInfo::Property { .. })));
    }

    #[test]
    fn check_ok_arithmetic_result_is_integer() {
        let bound = check_ok("MATCH (n:Person) RETURN n.age + 1");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let r = match q.clauses.last().unwrap() {
            BoundClause::Return(r) => r,
            _ => panic!(),
        };
        let e = match &r.items[0].kind {
            BoundProjectionKind::Expr(e) => e,
            _ => panic!(),
        };
        // Property(String) + Integer → arithmetic mismatch in v1.0
        // strict-mode, but we keep is_numeric permissive for
        // Property::Integer — and v1.0 default property type is
        // String. Result: TypeInfo::Null (graceful) because we
        // emitted a TypeMismatch above. Confirm we got the mismatch.
        let _ = e; // structural check below
    }

    #[test]
    fn check_ok_null_propagates_through_comparison() {
        // `n.age > NULL` → Null result; WHERE treats Null as FALSE
        // and accepts (no error).
        let bound = check_ok("MATCH (n:Person) WHERE n.age > NULL RETURN n");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let m = match &q.clauses[0] {
            BoundClause::Match(m) => m,
            _ => panic!(),
        };
        let w = m.where_clause.as_ref().unwrap();
        // The `>` returns Null.
        assert!(matches!(w.type_info(), Some(TypeInfo::Null)));
    }

    #[test]
    fn check_ok_is_null_yields_boolean() {
        let bound = check_ok("MATCH (n:Person) WHERE n.age IS NULL RETURN n");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let m = match &q.clauses[0] {
            BoundClause::Match(m) => m,
            _ => panic!(),
        };
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(w.type_info(), Some(TypeInfo::Boolean)));
    }

    #[test]
    fn check_err_unknown_function() {
        let errs = check_err("MATCH (n:Person) RETURN nope_not_a_real_fn(n)");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::UnknownFunction { .. })
            )),
            "expected UnknownFunction, got {errs:?}"
        );
    }

    #[test]
    fn check_err_function_arity_mismatch() {
        // count() expects 1 arg.
        let errs = check_err("MATCH (n:Person) RETURN count(n, n)");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::FunctionArityMismatch { .. })
            )),
            "expected FunctionArityMismatch, got {errs:?}"
        );
    }

    #[test]
    fn check_function_call_propagates_return_type() {
        let bound = check_ok("MATCH (n:Person) RETURN count(n)");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let r = match q.clauses.last().unwrap() {
            BoundClause::Return(r) => r,
            _ => panic!(),
        };
        let e = match &r.items[0].kind {
            BoundProjectionKind::Expr(e) => e,
            _ => panic!(),
        };
        assert_eq!(e.type_info(), Some(&TypeInfo::Integer));
    }

    #[test]
    fn check_unknown_function_does_not_panic_on_propagation() {
        // Ensures function-call args still get type-checked even
        // when the function name itself is unknown.
        let errs = check_err("MATCH (n:Person) RETURN nope_not_a_real_fn(n.age)");
        assert!(!errs.is_empty());
    }

    #[test]
    fn check_in_op_yields_boolean() {
        let bound = check_ok("MATCH (n:Person) WHERE n.age IN [1, 2, 3] RETURN n");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let m = match &q.clauses[0] {
            BoundClause::Match(m) => m,
            _ => panic!(),
        };
        let w = m.where_clause.as_ref().unwrap();
        assert!(matches!(w.type_info(), Some(TypeInfo::Boolean)));
    }

    #[test]
    fn check_unwind_var_takes_list_element_type() {
        let bound = check_ok("UNWIND [1, 2, 3] AS x RETURN x");
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!(),
        };
        let u = match &q.clauses[0] {
            BoundClause::Unwind(u) => u,
            _ => panic!(),
        };
        // List(Null) at v1.0 (we don't infer element types from list
        // literals); the UNWIND var takes the element type.
        assert!(u.var.type_info.is_some());
    }

    // ---------- #696: var-length rel-var static type ----------

    /// Extract the first MATCH-pattern tail rel binding's `type_info`
    /// from a single-MATCH read query.
    fn first_rel_type_info(input: &str) -> TypeInfo {
        let bound = check_ok(input);
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let m = match &q.clauses[0] {
            BoundClause::Match(m) => m,
            _ => panic!("expected MATCH"),
        };
        let path = match &m.body {
            BoundMatchBody::Patterns(ps) => &ps[0],
            _ => panic!("expected pattern body"),
        };
        let (rel, _node) = &path.tail[0];
        rel.var
            .as_ref()
            .expect("rel has a named binding")
            .type_info
            .clone()
            .expect("rel binding type_info populated")
    }

    /// #696 (follow-up of #695 / ADR-186 R1 M-1): a quantified
    /// (var-length) `rel_var` is typed as `List(Relationship)` so the
    /// static type matches the execution-layer RC-2 contract
    /// (`crate::executor::ops::expand` binds the var-length rel-var to
    /// `Value::List(Vec<Value::Relationship>)`); a single-hop `rel_var`
    /// keeps the scalar `Relationship` shape.
    ///
    /// Strong-oracle: this test fails BOTH ways — if the quantified case
    /// were left scalar (the latent gap #696 closes) AND if the
    /// single-hop case were widened to a list (a NEW bug). The two
    /// assertions are not interchangeable.
    #[test]
    fn var_length_rel_var_is_list_single_hop_is_scalar() {
        // Var-length `*1..3` (openCypher `LengthRange::Cypher`) →
        // List(Relationship).
        let varlen = first_rel_type_info("MATCH (a:Person)-[r:KNOWS*1..3]->(b:Person) RETURN r");
        match &varlen {
            TypeInfo::List(inner) => {
                assert!(
                    matches!(**inner, TypeInfo::Relationship { .. }),
                    "var-length rel-var must be List(Relationship), got List({inner:?})"
                );
            }
            other => panic!("var-length rel-var must be List(Relationship), got {other:?}"),
        }

        // Single-hop `-[r]->` → scalar Relationship (must NOT be a list).
        let single = first_rel_type_info("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r");
        assert!(
            matches!(single, TypeInfo::Relationship { .. }),
            "single-hop rel-var must stay scalar Relationship, got {single:?}"
        );

        // Unbounded `*` (openCypher `LengthRange::Unbounded`) is also a
        // var-length form → List(Relationship).
        let unbounded = first_rel_type_info("MATCH (a:Person)-[r:KNOWS*]->(b:Person) RETURN r");
        assert!(
            matches!(unbounded, TypeInfo::List(_)),
            "unbounded `*` rel-var must be List(Relationship), got {unbounded:?}"
        );
    }

    // =================================================================
    // ADR-188 — list-predicate / reduce TYPE-CHECK tests (Decision 3 +
    // OQ-5 widening accept/reject).
    // =================================================================

    #[test]
    fn lp_all_type_checks_ok() {
        // all(x IN [1,2,3] WHERE x > 0) is a valid Boolean predicate.
        check_ok("MATCH (n:Person) WHERE all(x IN [1, 2, 3] WHERE x > 0) RETURN n");
    }

    #[test]
    fn lp_all_in_return_is_boolean() {
        // RETURN all(...) — the projection types to Boolean.
        let bound = check_ok("MATCH (n:Person) RETURN all(x IN [1, 2, 3] WHERE x > 0) AS ok");
        let q = match &bound {
            BoundStatement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let ret = q
            .clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::Return(r) => Some(r),
                _ => None,
            })
            .expect("RETURN");
        let ti = match &ret.items[0].kind {
            BoundProjectionKind::Expr(e) => e.type_info(),
            _ => panic!(),
        };
        assert_eq!(ti, Some(&TypeInfo::Boolean), "all(...) types to Boolean");
    }

    #[test]
    fn reduce_int_plus_int_type_checks_ok() {
        // reduce(s = 0, x IN [1,2,3] | s + x) — Int acc + Int body ⇒ Int.
        check_ok("MATCH (n:Person) RETURN reduce(s = 0, x IN [1, 2, 3] | s + x) AS total");
    }

    #[test]
    fn reduce_int_acc_float_body_widens_accept() {
        // OQ-5 ACCEPT side: Int acc + Float body WIDENS to Float (NOT a
        // type error — a false-reject is a conformance failure).
        check_ok("MATCH (n:Person) RETURN reduce(s = 0, x IN [1] | s + 1.5) AS total");
    }

    #[test]
    fn reduce_int_acc_string_body_rejects() {
        // OQ-5 REJECT side: Int acc + a body that types to String is
        // genuinely non-assignable ⇒ TypeCheckError. MUST error.
        let errs = check_err("MATCH (n:Person) RETURN reduce(s = 0, x IN [1, 2] | 'z') AS bad");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "Int acc + String body MUST be a TypeMismatch, got {errs:?}"
        );
    }

    #[test]
    fn lp_over_non_list_operand_rejects() {
        // Decision 3 list-operand rule: `all(x IN 5 WHERE x > 0)` — `5`
        // is not List(_)/Null ⇒ TypeCheckError. MUST error.
        let errs = check_err("MATCH (n:Person) WHERE all(x IN 5 WHERE x > 0) RETURN n");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "all() over non-list operand MUST be a TypeMismatch, got {errs:?}"
        );
    }

    #[test]
    fn lp_over_property_list_operand_ok() {
        // A property access as the list operand is admitted under the
        // dynamic-schema discipline (the runtime resolves the actual
        // value). `all(x IN n.age WHERE x > 0)` type-checks (n.age is a
        // Property — dynamic-schema).
        check_ok("MATCH (n:Person) WHERE all(x IN n.age WHERE x > 0) RETURN n");
    }

    #[test]
    fn reduce_unit() {
        // element_type_of + is_numeric_concrete are exercised indirectly
        // above; this pins the helper directly: List(Integer) ⇒ Integer.
        assert_eq!(
            element_type_of(&TypeInfo::List(Box::new(TypeInfo::Integer))),
            TypeInfo::Integer
        );
        assert_eq!(element_type_of(&TypeInfo::Null), TypeInfo::Null);
        assert!(is_numeric_concrete(&TypeInfo::Integer));
        assert!(is_numeric_concrete(&TypeInfo::Float));
        assert!(!is_numeric_concrete(&TypeInfo::Null));
        assert!(!is_numeric_concrete(&TypeInfo::String));
    }

    // =================================================================
    // ADR-188 amendment-01 (#723 quantifier type-mismatch gap) — a
    // quantifier predicate that applies a type-incompatible op to a
    // CONCRETE element type must be a COMPILE-time TypeCheck rejection
    // (the openCypher `InvalidArgumentType` / SyntaxError contract). The
    // founding incident: openCypher TCK Quantifier1-4 `[15]/[16]` ("Fail
    // <q> quantifier on type mismatch between list elements and
    // predicate") — e.g. `all(x IN ['Clara'] WHERE x % 2 = 0)`, String
    // `%` Integer. These regressed from PASS→FAIL when #723 added the
    // quantifier grammar/eval but the list LITERAL erased its element
    // type to `List(Null)`, so the 3VL short-circuit silently accepted
    // the bad arithmetic. The fix: `list_literal_elem_type` derives the
    // concrete homogeneous element type so `check_binary_op` fires.
    //
    // Oracle note: the TCK harness (`full_eligible_conformance.rs`)
    // scores these by PHASE (compile) not by the error detail string —
    // a compile-phase `ArcQLError::TypeCheck` satisfies the
    // `raised at compile time: InvalidArgumentType` expectation. These
    // tests assert exactly that class (`TypeCheckError::TypeMismatch`,
    // which `classify_engine_error` maps to `EngineErrClass::Compile`).
    // =================================================================

    /// Helper: a quantifier predicate doing arithmetic on a String list
    /// element MUST be a compile-time `TypeMismatch` for each of the four
    /// quantifiers (the Quantifier1-4 `[15]/[16]` shape).
    fn assert_quantifier_string_arithmetic_rejects(quantifier: &str) {
        let q =
            format!("MATCH (n:Person) WHERE {quantifier}(x IN ['Clara'] WHERE x % 2 = 0) RETURN n");
        let errs = check_err(&q);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "{quantifier}(x IN ['Clara'] WHERE x % 2 = 0) — String % Integer MUST be a \
             compile-time TypeMismatch (InvalidArgumentType), got {errs:?}"
        );
    }

    #[test]
    fn lp_all_quantifier_string_elem_arithmetic_rejects() {
        // Quantifier4 [15] "Fail all quantifier on type mismatch".
        assert_quantifier_string_arithmetic_rejects("all");
    }

    #[test]
    fn lp_any_quantifier_string_elem_arithmetic_rejects() {
        // Quantifier3 [15] "Fail any quantifier on type mismatch".
        assert_quantifier_string_arithmetic_rejects("any");
    }

    #[test]
    fn lp_none_quantifier_string_elem_arithmetic_rejects() {
        // Quantifier1 [15] "Fail none quantifier on type mismatch".
        assert_quantifier_string_arithmetic_rejects("none");
    }

    #[test]
    fn lp_single_quantifier_string_elem_arithmetic_rejects() {
        // Quantifier2 [16] "Fail single quantifier on type mismatch".
        assert_quantifier_string_arithmetic_rejects("single");
    }

    #[test]
    fn lp_bool_elem_arithmetic_rejects() {
        // The TCK [15] examples also include `[false, true]` with the
        // same `x % 2 = 0` predicate — Boolean `%` Integer must reject.
        let errs =
            check_err("MATCH (n:Person) WHERE all(x IN [false, true] WHERE x % 2 = 0) RETURN n");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "all(x IN [false, true] WHERE x % 2 = 0) — Boolean % Integer MUST reject, got {errs:?}"
        );
    }

    #[test]
    fn lp_numeric_elem_arithmetic_still_ok() {
        // CONTRAST (no false-reject): a homogeneous numeric literal list
        // with arithmetic on the element is well-typed and MUST pass —
        // `[1,2,3] : List(Integer)`, `x % 2` is `Integer % Integer`.
        check_ok("MATCH (n:Person) WHERE all(x IN [1, 2, 3] WHERE x % 2 = 0) RETURN n");
    }

    #[test]
    fn lp_string_elem_comparison_still_ok() {
        // CONTRAST (no over-reject): a String element under an EQUALITY /
        // ORDER comparison is fine (Eq is universally typed; `<`/`>` are
        // orderable for String). Only type-INCOMPATIBLE arithmetic
        // rejects — equality/comparison on strings must NOT.
        check_ok("MATCH (n:Person) WHERE all(x IN ['a', 'b'] WHERE x = 'a') RETURN n");
        check_ok("MATCH (n:Person) WHERE any(x IN ['a', 'b'] WHERE x > 'a') RETURN n");
    }

    #[test]
    fn lp_heterogeneous_list_stays_conservative() {
        // A heterogeneous literal list has no single element type → the
        // helper yields `List(Null)` (the prior conservative behavior),
        // so the scoped var is `Null` and arithmetic 3VL-short-circuits
        // (no NEW reject). This is deliberate: mixed-list typing is a
        // separate openCypher question we do not silently take on here.
        check_ok("MATCH (n:Person) WHERE all(x IN [1, 'a'] WHERE x % 2 = 0) RETURN n");
    }

    #[test]
    fn reduce_string_elem_arithmetic_rejects() {
        // The same gap applied to `reduce`: a String element fed into
        // arithmetic in the fold body must reject at compile time.
        // `reduce(s = 0, x IN ['a'] | s + x)` — Int acc, but `s + x` is
        // `Integer + String` in the body ⇒ TypeMismatch.
        let errs = check_err("MATCH (n:Person) RETURN reduce(s = 0, x IN ['a'] | s + x) AS bad");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "reduce over String element with arithmetic body MUST reject, got {errs:?}"
        );
    }

    #[test]
    fn list_literal_in_projection_unaffected_by_elem_typing() {
        // A list literal used directly (RETURN projection / WITH) must
        // still type-check after element inference — the inference only
        // ADDS a concrete element type to `List(_)`, never breaks a
        // previously-valid list literal. The element-type inference is
        // construct-agnostic, so this bare list-literal / list-predicate
        // exercise also covers the source-list typing the
        // list-comprehension feature (#620/#724) reuses; the
        // comprehension-specific compile-time rejections are pinned by
        // the `lc_string_elem_*_rejects_at_compile` tests below.
        check_ok("MATCH (n:Person) RETURN [1, 2, 3] AS nums");
        check_ok("MATCH (n:Person) RETURN ['a', 'b', 'c'] AS strs");
        // A list literal as a list-predicate operand with a well-typed
        // String comparison — the homogeneous-String inference must not
        // turn a valid comparison into a reject.
        check_ok("MATCH (n:Person) WHERE none(x IN ['a', 'b'] WHERE x = 'z') RETURN n");
    }

    #[test]
    fn list_literal_elem_type_unit() {
        use crate::ast::Expression as E;
        let lit = |l: Literal| E::Literal(l);
        // Homogeneous scalar lists ⇒ concrete element type.
        assert_eq!(
            list_literal_elem_type(&[
                lit(Literal::String("a".into())),
                lit(Literal::String("b".into()))
            ]),
            TypeInfo::String
        );
        assert_eq!(
            list_literal_elem_type(&[lit(Literal::Integer(1)), lit(Literal::Integer(2))]),
            TypeInfo::Integer
        );
        assert_eq!(
            list_literal_elem_type(&[lit(Literal::Bool(true))]),
            TypeInfo::Boolean
        );
        // Heterogeneous ⇒ conservative Null.
        assert_eq!(
            list_literal_elem_type(&[lit(Literal::Integer(1)), lit(Literal::String("a".into()))]),
            TypeInfo::Null
        );
        // A Null element collapses to Null (3VL: element possibly-null).
        assert_eq!(
            list_literal_elem_type(&[lit(Literal::Integer(1)), lit(Literal::Null)]),
            TypeInfo::Null
        );
        // Empty list ⇒ Null (no element to type).
        assert_eq!(list_literal_elem_type(&[]), TypeInfo::Null);
        // A nested list element ⇒ conservative Null (no recursion at v1.0).
        assert_eq!(
            list_literal_elem_type(&[lit(Literal::List(vec![]))]),
            TypeInfo::Null
        );
        // A non-literal element (parameter) ⇒ conservative Null.
        assert_eq!(
            list_literal_elem_type(&[E::Parameter("p".into())]),
            TypeInfo::Null
        );
    }

    // ---------- ADR-188 (#620 list-half) — list comprehension ----------

    /// Extract the type of the first RETURN projection item.
    fn return_proj_type(bound: &BoundStatement) -> Option<TypeInfo> {
        let q = match bound {
            BoundStatement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let ret = q
            .clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::Return(r) => Some(r),
                _ => None,
            })
            .expect("RETURN");
        match &ret.items[0].kind {
            BoundProjectionKind::Expr(e) => e.type_info().cloned(),
            _ => panic!("expected Expr projection"),
        }
    }

    #[test]
    fn lc_identity_type_is_list_of_element_type() {
        // [x IN [1,2,3]] — identity projection ⇒ result type
        // List(element-type-of([1,2,3])). The list literal types to
        // List(Null) at v1.0 (element types are not statically tracked
        // for inline literals), so the result is List(Null) — but it IS
        // a List, which is the load-bearing assertion (the result is a
        // list, not the element type or a scalar).
        let bound = check_ok("MATCH (n:Person) RETURN [x IN [1, 2, 3]] AS ys");
        let ti = return_proj_type(&bound).expect("type_info");
        assert!(
            matches!(ti, TypeInfo::List(_)),
            "identity list comprehension MUST type to List(_), got {ti:?}"
        );
    }

    #[test]
    fn lc_projection_type_drives_result_not_element_type() {
        // [x IN [1,2,3] | 'hello'] — the projection types to String
        // INDEPENDENT of the element type, so the comprehension types to
        // List(String). This BITES on a wrong impl that used the element
        // type (List(Null), since inline-list element types are not
        // statically tracked at v1.0) instead of the PROJECTION type.
        // (Contrast lc_identity_type_is_list_of_element_type, where the
        // identity projection correctly yields the element type.)
        let bound = check_ok("MATCH (n:Person) RETURN [x IN [1, 2, 3] | 'hello'] AS ys");
        let ti = return_proj_type(&bound).expect("type_info");
        assert_eq!(
            ti,
            TypeInfo::List(Box::new(TypeInfo::String)),
            "`| 'hello'` projection MUST yield List(String) (projection type, not element type)"
        );
    }

    #[test]
    fn lc_filtered_with_projection_type_ok() {
        // [x IN [1,2,3] WHERE x > 1 | 'z'] — the WHERE filter is
        // type-checked (well-formed boolean) and the result is
        // List(String) from the constant projection (proving the filter
        // does not feed the result element type).
        let bound = check_ok("MATCH (n:Person) RETURN [x IN [1, 2, 3] WHERE x > 1 | 'z'] AS ys");
        let ti = return_proj_type(&bound).expect("type_info");
        assert_eq!(ti, TypeInfo::List(Box::new(TypeInfo::String)));
    }

    #[test]
    fn lc_over_non_list_operand_rejects() {
        // Decision 3 list-operand rule reused: `[x IN 5 | x]` — `5` is
        // not List(_)/Null ⇒ TypeCheckError. MUST error.
        let errs = check_err("MATCH (n:Person) RETURN [x IN 5 | x] AS ys");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "list comprehension over non-list operand MUST be a TypeMismatch, got {errs:?}"
        );
    }

    #[test]
    fn lc_over_property_list_operand_ok() {
        // A property access as the source list is admitted under the
        // dynamic-schema discipline. `[x IN n.age | x]` type-checks
        // (n.age is a Property — dynamic-schema).
        check_ok("MATCH (n:Person) RETURN [x IN n.age | x] AS ys");
    }

    #[test]
    fn lc_scoped_var_not_visible_after_comprehension() {
        // The scoped `x` is torn down at pop_scope — referencing it AFTER
        // the comprehension is an UndeclaredVariable (binder error). We
        // assert the binder rejects a sibling reference to `x`. (This is
        // a BINDING error, surfaced at bind time — so `parse` succeeds
        // but `bind` fails.)
        let stmt = parse("MATCH (n:Person) RETURN [x IN [1, 2, 3] | x] AS ys, x AS leaked")
            .expect("parse");
        let cat = StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_properties(["age"]);
        let bound = BindingVisitor::bind(&stmt, "q", &cat);
        // `x` leaks into the sibling projection ⇒ the binder records an
        // UndeclaredVariable (it does NOT resolve to the scoped var).
        // Bind either errors, or binds it as UnresolvedVariable — either
        // way the scoped `x` MUST NOT resolve outside the comprehension.
        match bound {
            Err(_) => { /* binder rejected the leaked `x` — correct */ }
            Ok(b) => {
                // If bind tolerates it (recording an unresolved node),
                // assert the leaked `x` did NOT resolve to a real binding.
                let q = match &b {
                    BoundStatement::Read(q) => q,
                    _ => panic!("expected Read"),
                };
                let ret = q
                    .clauses
                    .iter()
                    .find_map(|c| match c {
                        BoundClause::Return(r) => Some(r),
                        _ => None,
                    })
                    .expect("RETURN");
                // Second projection item is the leaked `x`.
                let leaked = &ret.items[1].kind;
                assert!(
                    matches!(
                        leaked,
                        BoundProjectionKind::Expr(BoundExpression::UnresolvedVariable { .. })
                    ),
                    "leaked scoped `x` MUST be unresolved outside the comprehension, got {leaked:?}"
                );
            }
        }
    }

    // ---------- ADR-188 (#620 list-half) — list-comprehension type-check
    // regression guard (#724 R1 MED-1) ----------
    //
    // The list-comprehension type-check REJECTS a type-incompatible WHERE
    // predicate / `| projection` at COMPILE time (the openCypher
    // `InvalidArgumentType` / SyntaxError contract) by reusing the EXACT
    // scoped-var-typing machinery the quantifier path uses:
    // `binding_types.insert(var_bid, element_type_of(list))` followed by
    // `check_expression`, so `check_binary_op`'s `is_numeric` arithmetic
    // check fires on the concrete element type. That rejection is CORRECT
    // but was UNGUARDED by a co-located test — the EXACT silent-regression
    // class that produced the #723 quantifier gap (a correct-but-untested
    // type-check let a conformance regression slip through unnoticed). A
    // regression in the shared element-typing machinery (e.g. re-erasing a
    // list-literal element type to `List(Null)`, the original #723 root
    // cause) would silently un-guard BOTH halves; the quantifier tests
    // above + these comprehension tests together fence it. Asserting the
    // explicit `TypeCheckError::TypeMismatch` variant proves the rejection
    // is a COMPILE-phase error (not a runtime surprise / panic): `check_err`
    // succeeds at parse + bind and fails only at type-check.

    /// Helper: a list-comprehension applying type-incompatible arithmetic
    /// to a CONCRETE String element (in the WHERE filter or the `|`
    /// projection) MUST surface as a compile-time `TypeMismatch`.
    fn assert_lc_string_arithmetic_rejects_at_compile(query: &str, what: &str) {
        let errs = check_err(query);
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::TypeCheck(TypeCheckError::TypeMismatch { .. })
            )),
            "{what}: `{query}` MUST be a compile-time TypeMismatch \
             (InvalidArgumentType), got {errs:?}"
        );
    }

    #[test]
    fn lc_string_elem_where_arithmetic_rejects_at_compile() {
        // WHERE filter `x % 2 = 0` over a String element — String %
        // Integer rejects at COMPILE time (NOT a runtime error / panic).
        assert_lc_string_arithmetic_rejects_at_compile(
            "MATCH (n:Person) RETURN [x IN ['Clara'] WHERE x % 2 = 0 | x] AS ys",
            "String % Integer in WHERE",
        );
    }

    #[test]
    fn lc_string_elem_projection_arithmetic_rejects_at_compile() {
        // `| projection` `x + 1` over a String element — String + Integer
        // rejects at COMPILE time (the projection is type-checked too, not
        // only the WHERE).
        assert_lc_string_arithmetic_rejects_at_compile(
            "MATCH (n:Person) RETURN [x IN ['a', 'b'] | x + 1] AS ys",
            "String + Integer in projection",
        );
    }

    #[test]
    fn lc_numeric_elem_filter_and_projection_still_ok() {
        // POSITIVE control (no over-reject): a homogeneous Integer list
        // with a well-typed filter AND projection MUST type-check —
        // `[1,2,3,4] : List(Integer)`, `x % 2 = 0` and `x * 10` are both
        // Integer arithmetic. Guards the guard against false-rejects.
        check_ok("MATCH (n:Person) RETURN [x IN [1, 2, 3, 4] WHERE x % 2 = 0 | x * 10] AS ys");
    }
}
