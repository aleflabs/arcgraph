//! M4-23 cross-substrate validation pass.
//!
//! [`CrossSubstrateValidator::validate`] consumes a type-checked
//! [`BoundStatement`] (post-M4-22) and verifies two contracts:
//!
//! 1. **Substrate availability.** Surfaces that require the vector,
//!    BM25, or community substrate are admitted only when the
//!    per-tenant catalog reports the substrate is attached. Surfaces
//!    checked:
//!    - `<expr> NEAR <expr>` and `vector_distance(...)` →
//!      [`CatalogProvider::has_vector_index`].
//!    - `<expr> MATCH <expr>` and `text_match(...)` →
//!      [`CatalogProvider::has_bm25_index`].
//!    - `n IN COMMUNITY(<expr>)` and the `community(...)` /
//!      `community_members(...)` / `community_rank_by_seeds(...)`
//!      function family → [`CatalogProvider::has_community_index`].
//!    - `RANK BY HYBRID(VECTOR(...), TEXT(...))` requires both vector
//!      AND BM25 substrates per ADR-038 §2 D-3.
//!
//! 2. **`RANK BY HYBRID` + `WITH FUSION = RRF` semantic shape.** Per
//!    ADR-038 §2 D-3 + D-9:
//!    - The hybrid clause MUST contain at least one `VECTOR(...)`
//!      operand AND at least one `TEXT(...)` operand. A clause that
//!      ships only VECTOR or only TEXT operands is rejected with
//!      [`CrossSubstrateError::HybridMissingOperand`].
//!    - Each `VECTOR(...)` / `TEXT(...)` operand MUST carry an
//!      explicit `K = N` parameter (no implicit defaults at v1.0).
//!      A bare `VECTOR(field, query)` without `K = ...` is rejected
//!      with [`CrossSubstrateError::HybridMissingK`].
//!    - `WITH FUSION = RRF(k = N)` MUST carry an explicit `k`. The
//!      v1.0 grammar already requires `k` at parse time
//!      (`ParseError::AstConstruction`), so
//!      [`CrossSubstrateError::FusionMissingK`] is defensive — it
//!      exists for symmetry with `HybridMissingK` and for any
//!      programmatic constructor of `BoundFusion::Rrf` that
//!      bypasses the parser.
//!
//! # Walker shape — CUSTOM, not a trait
//!
//! `CrossSubstrateValidator` walks `BoundQuery` through dedicated
//! `walk_*` methods. **There is no trait abstraction.** The M4-21 +
//! M4-22 history (PR #164 reviewer ask + PR #165 reviewer Finding 1)
//! established that speculative visitor traits with no second consumer
//! produce 400+ LOC of dead surface that gets deleted in the next
//! slice. The custom-walker pattern is the established precedent
//! ([`crate::semantic::binding::BindingVisitor`],
//! [`crate::semantic::type_check::TypeCheckVisitor`], and now
//! `CrossSubstrateValidator`). M4-31 (logical plan generator) MUST
//! ship its own custom walker; it MUST NOT inherit any abstraction
//! from this module — see `feedback_avoid_speculative_scaffolding.md`.
//!
//! # Error accumulation
//!
//! The pass does NOT short-circuit on the first error: it surfaces
//! every cross-substrate fault in a single walk, matching M4-21 +
//! M4-22 discipline. Multiple substrate misses, missing operands, and
//! missing K parameters are all reported together.
//!
//! # ADR provenance
//! - ADR-038 §2 D-23 — cross-substrate validation contract (this
//!   file's primary spec).
//! - ADR-038 §2 D-3 — `RANK BY HYBRID(VECTOR(...), TEXT(...))` shape.
//! - ADR-038 §2 D-4 — `community(...)` family + `IN COMMUNITY(...)`
//!   alternate predicate (per amendment-01).
//! - ADR-038 §2 D-9 — `RRF(k = N)` fusion explicit-k requirement.
//! - ADR-035 D-7 / ADR-039 D-4 / ADR-040 D-3 — per-tenant substrate
//!   keying.

use crate::error::Span;
use crate::semantic::bound_ast::{
    BoundClause, BoundCreateItem, BoundExpression, BoundFusion, BoundMapProjectionItem,
    BoundMatchBody, BoundMatchClause, BoundProjectionItem, BoundProjectionKind, BoundQuery,
    BoundRankArg, BoundRankByClause, BoundRanker, BoundReturnClause, BoundStatement,
    BoundWithClause, BoundWithFusionClause,
};
use crate::semantic::catalog::CatalogProvider;
use crate::semantic::error::{ArcQLError, CrossSubstrateError, SubstrateKind};

/// M4-23 cross-substrate validation pass.
///
/// Construct via [`Self::validate`]; the internal struct is not part
/// of the public API. The pass does NOT mutate the input
/// `BoundStatement` — it walks read-only and accumulates errors.
pub struct CrossSubstrateValidator<'cat, C: CatalogProvider> {
    catalog: &'cat C,
    errors: Vec<CrossSubstrateError>,
}

impl<'cat, C: CatalogProvider> CrossSubstrateValidator<'cat, C> {
    /// Validate a [`BoundStatement`]. Returns `Ok(())` on a clean
    /// pass or `Err(Vec<ArcQLError>)` with accumulated cross-substrate
    /// diagnostics.
    ///
    /// Errors are returned as [`ArcQLError::CrossSubstrate`] for
    /// uniform downstream handling alongside the binding +
    /// type-check error taxonomies.
    pub fn validate(stmt: &BoundStatement, catalog: &'cat C) -> Result<(), Vec<ArcQLError>> {
        let mut v = Self {
            catalog,
            errors: Vec::new(),
        };
        v.walk_statement(stmt);
        if v.errors.is_empty() {
            Ok(())
        } else {
            Err(v.errors.into_iter().map(ArcQLError::from).collect())
        }
    }

    // ---------- Statement / query / clause dispatch ----------

    fn walk_statement(&mut self, stmt: &BoundStatement) {
        match stmt {
            BoundStatement::Read(q) => self.walk_query(q),
            // ADR-185 (#649-A1, W28) — UNION / UNION ALL: cross-
            // substrate-validate each arm independently (each arm's
            // MATCH patterns may touch different substrates; validation
            // is per-arm just like a standalone read query).
            BoundStatement::Union(u) => {
                for arm in &u.arms {
                    self.walk_query(arm);
                }
            }
            BoundStatement::IndexDdl(_) => {}
        }
    }

    fn walk_query(&mut self, q: &BoundQuery) {
        for c in &q.clauses {
            self.walk_clause(c);
        }
    }

    fn walk_clause(&mut self, c: &BoundClause) {
        match c {
            BoundClause::Match(m) => self.walk_match(m),
            // ADR-147 W26-θ Phase 1: CREATE node touches only the
            // CRUD store — no vector / BM25 / community substrate
            // dependence. Walk property-value sub-expressions for
            // defense-in-depth (an embedded function call inside a
            // property expression could surface a substrate fault),
            // but no cross-substrate constraint applies to the
            // CREATE clause itself.
            BoundClause::Create(c) => {
                for item in &c.items {
                    match item {
                        BoundCreateItem::Node(spec) => {
                            if let Some(props) = &spec.properties {
                                for entry in &props.entries {
                                    self.walk_expression(&entry.value);
                                }
                            }
                        }
                        // ADR-148 W26-θ Phase 2: CREATE-path is
                        // semantically a no-op for cross-substrate
                        // (same as Phase 1 — CRUD store only). Walk
                        // every property bag for defense-in-depth
                        // (an embedded function call in source / rel /
                        // target property bag could surface a substrate
                        // fault).
                        BoundCreateItem::Path(path) => {
                            if let Some(props) = &path.source.properties {
                                for entry in &props.entries {
                                    self.walk_expression(&entry.value);
                                }
                            }
                            if let Some(props) = &path.rel.properties {
                                for entry in &props.entries {
                                    self.walk_expression(&entry.value);
                                }
                            }
                            if let Some(props) = &path.target.properties {
                                for entry in &props.entries {
                                    self.walk_expression(&entry.value);
                                }
                            }
                        }
                    }
                }
            }
            // ADR-149 W26-θ Phase 3: DELETE touches only the CRUD
            // store (no vector / BM25 / community substrate
            // dependence). The items are bare variable references —
            // no sub-expressions to walk for defense-in-depth (no
            // property bag, no filter predicate). Pure no-op.
            BoundClause::Delete(_) => {}
            // ADR-150 W26-θ Phase 4: SET touches only the CRUD store.
            // The property-value sub-expressions of PropertyAssign /
            // PropertyReplace / PropertyMerge are walked for any
            // embedded substrate-bearing function calls (parallel to
            // the Phase 1 + Phase 2 CREATE walk-property-bag pattern);
            // LabelAdd has no expression sub-trees.
            BoundClause::Set(s) => {
                for item in &s.items {
                    match &item.mutation {
                        crate::semantic::bound_ast::BoundSetMutation::PropertyAssign {
                            value,
                            ..
                        } => {
                            self.walk_expression(value);
                        }
                        crate::semantic::bound_ast::BoundSetMutation::PropertyReplace(map)
                        | crate::semantic::bound_ast::BoundSetMutation::PropertyMerge(map) => {
                            for entry in &map.entries {
                                self.walk_expression(&entry.value);
                            }
                        }
                        crate::semantic::bound_ast::BoundSetMutation::LabelAdd(_) => {}
                    }
                }
            }
            // ADR-150 W26-θ Phase 4: REMOVE touches only the CRUD
            // store; the items are bare references — pure no-op
            // (parallel to DELETE).
            BoundClause::Remove(_) => {}
            // ADR-151 W26-θ Phase 5: MERGE touches only the CRUD store
            // (match-branch reuses scan_nodes / expand; create-branch
            // reuses create_node / create_rel; action items reuse
            // update_node / update_rel). Walk the pattern's property-
            // value sub-expressions for defense-in-depth (parallel to
            // the Phase 1 + 2 CREATE walk-property-bag pattern); walk
            // the on_create / on_match action items' property-value
            // sub-expressions for defense-in-depth (parallel to the
            // Phase 4 SET walk).
            BoundClause::Merge(m) => {
                match &m.pattern {
                    crate::semantic::bound_ast::BoundMergePattern::Node(spec) => {
                        if let Some(props) = &spec.properties {
                            for entry in &props.entries {
                                self.walk_expression(&entry.value);
                            }
                        }
                    }
                    crate::semantic::bound_ast::BoundMergePattern::Path(path) => {
                        if let Some(props) = &path.source.properties {
                            for entry in &props.entries {
                                self.walk_expression(&entry.value);
                            }
                        }
                        if let Some(props) = &path.rel.properties {
                            for entry in &props.entries {
                                self.walk_expression(&entry.value);
                            }
                        }
                        if let Some(props) = &path.target.properties {
                            for entry in &props.entries {
                                self.walk_expression(&entry.value);
                            }
                        }
                    }
                }
                for item in m.on_create.iter().chain(m.on_match.iter()) {
                    match &item.mutation {
                        crate::semantic::bound_ast::BoundSetMutation::PropertyAssign {
                            value,
                            ..
                        } => {
                            self.walk_expression(value);
                        }
                        crate::semantic::bound_ast::BoundSetMutation::PropertyReplace(map)
                        | crate::semantic::bound_ast::BoundSetMutation::PropertyMerge(map) => {
                            for entry in &map.entries {
                                self.walk_expression(&entry.value);
                            }
                        }
                        crate::semantic::bound_ast::BoundSetMutation::LabelAdd(_) => {}
                    }
                }
            }
            BoundClause::With(w) => self.walk_with(w),
            BoundClause::Unwind(u) => self.walk_expression(&u.expr),
            // ADR-192 (#623): recurse into the CALL{} subquery body so
            // its clauses' expressions are cross-substrate-validated too
            // (a substrate-bearing predicate inside a subquery must be
            // caught the same as in the outer query).
            BoundClause::Call(c) => self.walk_call_body(c.body.as_ref()),
            // ADR-197 (#802): the schema-introspection procedures + SHOW
            // touch only the catalog/intern-table — no vector / BM25 /
            // community substrate dependence. Walk the procedure args
            // for defense-in-depth (an embedded function-call inside an
            // arg could surface a substrate fault); SHOW carries no
            // expressions.
            //
            // #830 (D4): `db.index.vector.queryNodes` DOES depend on the
            // vector substrate, but — unlike `RANK BY vector(...)` — it
            // is NOT gated here at plan-time. Its substrate-availability
            // is checked inside the proc-body at execution, where it
            // surfaces a structured `SubstrateAccessError` (never a
            // silent-empty); see `executor::ops::procedure_call`. So the
            // arg-walk below is the only cross-substrate concern at this
            // clause (an embedded substrate-bearing arg expression).
            BoundClause::CallProcedure(c) => {
                for a in &c.args {
                    self.walk_expression(a);
                }
                if let Some(w) = &c.where_clause {
                    self.walk_expression(w);
                }
            }
            BoundClause::Show(_) => {}
            BoundClause::RankBy(r) => self.walk_rank_by(r),
            BoundClause::WithFusion(f) => self.walk_with_fusion(f),
            BoundClause::Return(r) => self.walk_return(r),
            BoundClause::TailOrderBy(items, _) => {
                for o in items {
                    self.walk_expression(&o.expr);
                }
            }
            BoundClause::TailSkip(e, _) | BoundClause::TailLimit(e, _) => {
                self.walk_expression(e);
            }
        }
    }

    /// ADR-192 (#623): cross-substrate-validate a `CALL { … }` subquery
    /// body (Read → walk its clauses; Union → walk each arm + the
    /// post-union tail).
    fn walk_call_body(&mut self, body: &BoundStatement) {
        match body {
            BoundStatement::Read(q) => self.walk_query(q),
            BoundStatement::Union(u) => {
                for arm in &u.arms {
                    self.walk_query(arm);
                }
                for o in &u.tail.order_by {
                    self.walk_expression(&o.expr);
                }
                if let Some(e) = &u.tail.skip {
                    self.walk_expression(e);
                }
                if let Some(e) = &u.tail.limit {
                    self.walk_expression(e);
                }
            }
            // Grammar admits only Read/Union inside CALL{}.
            _ => {}
        }
    }

    fn walk_match(&mut self, m: &BoundMatchClause) {
        // Pattern-internal expressions (property maps) — walk for any
        // embedded substrate-bearing function calls.
        match &m.body {
            BoundMatchBody::Patterns(ps) => {
                for p in ps {
                    self.walk_property_map_in_node(&p.head.properties);
                    for (rel, node) in &p.tail {
                        self.walk_property_map_in_rel(&rel.properties);
                        self.walk_property_map_in_node(&node.properties);
                    }
                }
            }
            BoundMatchBody::NamedPath(np) => {
                let pp = match &np.kind {
                    crate::semantic::bound_ast::BoundNamedPathKind::ShortestPath(p)
                    | crate::semantic::bound_ast::BoundNamedPathKind::AllShortestPath(p)
                    | crate::semantic::bound_ast::BoundNamedPathKind::Plain(p) => p,
                };
                self.walk_property_map_in_node(&pp.head.properties);
                for (rel, node) in &pp.tail {
                    self.walk_property_map_in_rel(&rel.properties);
                    self.walk_property_map_in_node(&node.properties);
                }
            }
        }
        if let Some(w) = &m.where_clause {
            self.walk_expression(w);
        }
    }

    fn walk_property_map_in_node(
        &mut self,
        pm: &Option<crate::semantic::bound_ast::BoundPropertyMap>,
    ) {
        if let Some(map) = pm {
            for entry in &map.entries {
                self.walk_expression(&entry.value);
            }
        }
    }

    fn walk_property_map_in_rel(
        &mut self,
        pm: &Option<crate::semantic::bound_ast::BoundPropertyMap>,
    ) {
        self.walk_property_map_in_node(pm);
    }

    fn walk_with(&mut self, w: &BoundWithClause) {
        for it in &w.items {
            self.walk_projection_item(it);
        }
        if let Some(e) = &w.where_clause {
            self.walk_expression(e);
        }
    }

    fn walk_return(&mut self, r: &BoundReturnClause) {
        for it in &r.items {
            self.walk_projection_item(it);
        }
        for o in &r.order_by {
            self.walk_expression(&o.expr);
        }
        if let Some(e) = &r.skip {
            self.walk_expression(e);
        }
        if let Some(e) = &r.limit {
            self.walk_expression(e);
        }
    }

    fn walk_projection_item(&mut self, p: &BoundProjectionItem) {
        if let BoundProjectionKind::Expr(e) = &p.kind {
            self.walk_expression(e);
        }
    }

    // ---------- RANK BY HYBRID + WITH FUSION ----------

    fn walk_rank_by(&mut self, r: &BoundRankByClause) {
        match &r.ranker {
            BoundRanker::Hybrid(args) => self.check_hybrid(args, &r.span),
        }
    }

    /// Validate a `RANK BY HYBRID(...)` operand list:
    ///
    /// 1. Both VECTOR(...) AND TEXT(...) operands MUST be present.
    /// 2. Each VECTOR / TEXT operand MUST carry an explicit K param.
    /// 3. The vector + BM25 substrates MUST be attached to the tenant.
    fn check_hybrid(&mut self, args: &[BoundRankArg], clause_span: &Span) {
        let mut has_vector = false;
        let mut has_text = false;
        for a in args {
            match a {
                BoundRankArg::Vector { k, query, span, .. } => {
                    has_vector = true;
                    if k.is_none() {
                        self.errors
                            .push(CrossSubstrateError::HybridMissingK { span: span.clone() });
                    }
                    self.require_substrate(SubstrateKind::Vector, span);
                    self.walk_expression(query);
                }
                BoundRankArg::Text { k, query, span, .. } => {
                    has_text = true;
                    if k.is_none() {
                        self.errors
                            .push(CrossSubstrateError::HybridMissingK { span: span.clone() });
                    }
                    self.require_substrate(SubstrateKind::Bm25, span);
                    self.walk_expression(query);
                }
            }
        }
        if !has_vector {
            self.errors.push(CrossSubstrateError::HybridMissingOperand {
                kind: "VECTOR",
                span: clause_span.clone(),
            });
        }
        if !has_text {
            self.errors.push(CrossSubstrateError::HybridMissingOperand {
                kind: "TEXT",
                span: clause_span.clone(),
            });
        }
    }

    fn walk_with_fusion(&mut self, c: &BoundWithFusionClause) {
        match &c.fusion {
            BoundFusion::Rrf { k } => {
                // The grammar enforces `k` at parse time; defensive
                // check for any programmatic constructor of
                // `BoundFusion::Rrf` that bypasses the parser. We treat
                // `k <= 0` as "missing": the RRF formula
                // `1 / (k + rank)` is degenerate at k = 0 (smoothing
                // property lost; Cormack SIGIR 2009 default is k = 60)
                // and undefined for negative k where `rank_i = -k`
                // (literal division by zero).
                if *k <= 0 {
                    self.errors.push(CrossSubstrateError::FusionMissingK {
                        span: c.span.clone(),
                    });
                }
            }
        }
    }

    // ---------- Expression-level substrate gating ----------

    fn walk_expression(&mut self, e: &BoundExpression) {
        match e {
            BoundExpression::Literal { .. }
            | BoundExpression::Parameter { .. }
            | BoundExpression::VariableRef { .. }
            | BoundExpression::UnresolvedVariable { .. } => {}
            BoundExpression::ListLiteral { elements, .. } => {
                for element in elements {
                    self.walk_expression(element);
                }
            }
            BoundExpression::MapLiteral { entries, .. } => {
                for (_, value) in entries {
                    self.walk_expression(value);
                }
            }
            BoundExpression::PropertyAccess { base, .. } => {
                self.walk_expression(base);
            }
            // #1290 — left-nested operator SPINE walked iteratively
            // (the spine can be `MAX_FLAT_CHAIN_DEPTH` deep and may
            // interleave BinaryOp / UnaryOp / In / IsNull levels;
            // recursing per level overflowed the native stack). Base
            // subtree first, then each level's rhs innermost→outermost
            // — the same error-emission order as the recursion this
            // replaces. Non-spine children recurse (bracket-bounded).
            BoundExpression::BinaryOp { .. }
            | BoundExpression::UnaryOp { .. }
            | BoundExpression::In { .. }
            | BoundExpression::IsNull { .. } => {
                let mut rhs_stack: Vec<&BoundExpression> = Vec::new();
                let mut cur = e;
                loop {
                    match cur {
                        BoundExpression::BinaryOp { lhs, rhs, .. }
                        | BoundExpression::In { lhs, rhs, .. } => {
                            rhs_stack.push(rhs);
                            cur = lhs;
                        }
                        BoundExpression::UnaryOp { operand, .. } => cur = operand,
                        BoundExpression::IsNull { lhs, .. } => cur = lhs,
                        other => {
                            self.walk_expression(other);
                            break;
                        }
                    }
                }
                while let Some(rhs) = rhs_stack.pop() {
                    self.walk_expression(rhs);
                }
            }
            BoundExpression::FunctionCall {
                name, args, span, ..
            } => {
                self.check_function_call(name, span);
                for a in args {
                    self.walk_expression(a);
                }
            }
            BoundExpression::Near {
                lhs, target, span, ..
            } => {
                self.require_substrate(SubstrateKind::Vector, span);
                self.walk_expression(lhs);
                self.walk_expression(target);
            }
            BoundExpression::TextMatch {
                lhs, query, span, ..
            } => {
                self.require_substrate(SubstrateKind::Bm25, span);
                self.walk_expression(lhs);
                self.walk_expression(query);
            }
            BoundExpression::InCommunity {
                node,
                community,
                span,
                ..
            } => {
                self.require_substrate(SubstrateKind::Community, span);
                self.walk_expression(node);
                self.walk_expression(community);
            }
            // ADR-188 — a list-predicate / reduce sub-expression may
            // contain a substrate-gated call (e.g.
            // `all(x IN community_members($c) WHERE …)`); walk every
            // child so the substrate requirement is surfaced.
            BoundExpression::ListPredicate {
                list, predicate, ..
            } => {
                self.walk_expression(list);
                self.walk_expression(predicate);
            }
            BoundExpression::Reduce {
                init, list, expr, ..
            } => {
                self.walk_expression(init);
                self.walk_expression(list);
                self.walk_expression(expr);
            }
            // ADR-188 (#620 list-half) — a list-comprehension
            // sub-expression may contain a substrate-gated call (e.g.
            // `[x IN community_members($c) WHERE … | x]`); walk every
            // child (list + optional predicate + optional projection) so
            // the substrate requirement is surfaced.
            BoundExpression::ListComprehension {
                list,
                predicate,
                projection,
                ..
            } => {
                self.walk_expression(list);
                if let Some(p) = predicate {
                    self.walk_expression(p);
                }
                if let Some(e) = projection {
                    self.walk_expression(e);
                }
            }
            // ADR-191 D-6 (#620 map-half) — a map projection's base or a
            // literal-entry value may contain a substrate-gated call (e.g.
            // `n{rank: community_rank_by_seeds($s)}`); walk the base + every
            // literal value so the substrate requirement is surfaced. The
            // `.key` / `.*` selectors carry only property names.
            BoundExpression::MapProjection { base, items, .. } => {
                self.walk_expression(base);
                for item in items {
                    if let BoundMapProjectionItem::Literal { value, .. } = item {
                        self.walk_expression(value);
                    }
                }
            }
            // openCypher v9 §3.4 — postfix accessors: walk the base + the
            // index / bounds so a substrate-gated call in any operand
            // (e.g. `community_members($c)[0]`) is surfaced.
            BoundExpression::Subscript { base, index, .. } => {
                self.walk_expression(base);
                self.walk_expression(index);
            }
            BoundExpression::Slice {
                base, start, end, ..
            } => {
                self.walk_expression(base);
                if let Some(s) = start {
                    self.walk_expression(s);
                }
                if let Some(e) = end {
                    self.walk_expression(e);
                }
            }
            // openCypher v9 §3.6 (#621) — a CASE sub-expression (the test,
            // any WHEN / THEN, or the ELSE) may contain a substrate-gated
            // call (e.g. `CASE WHEN community_members($c) … END`); walk every
            // child so the substrate requirement is surfaced.
            BoundExpression::Case {
                test,
                branches,
                default,
                ..
            } => {
                if let Some(t) = test {
                    self.walk_expression(t);
                }
                for (when, then) in branches {
                    self.walk_expression(when);
                    self.walk_expression(then);
                }
                if let Some(d) = default {
                    self.walk_expression(d);
                }
            }
        }
    }

    /// Substrate-gate function-call expressions.
    ///
    /// Per ADR-038 §2 D-4 + D-5 + D-6, the following function families
    /// require the corresponding substrate:
    ///
    /// - `vector_distance` → vector
    /// - `text_match` → BM25
    /// - `community` / `community_members` / `community_rank_by_seeds`
    ///   → community
    fn check_function_call(&mut self, name: &str, span: &Span) {
        match name {
            "vector_distance" => self.require_substrate(SubstrateKind::Vector, span),
            "text_match" => self.require_substrate(SubstrateKind::Bm25, span),
            "community" | "community_members" | "community_rank_by_seeds" => {
                self.require_substrate(SubstrateKind::Community, span)
            }
            _ => {}
        }
    }

    fn require_substrate(&mut self, kind: SubstrateKind, span: &Span) {
        let available = match kind {
            SubstrateKind::Vector => self.catalog.has_vector_index(),
            SubstrateKind::Bm25 => self.catalog.has_bm25_index(),
            SubstrateKind::Community => self.catalog.has_community_index(),
        };
        if !available {
            self.errors.push(CrossSubstrateError::SubstrateUnavailable {
                kind,
                tenant: self.catalog.tenant(),
                span: span.clone(),
            });
        }
    }
}

// =====================================================================
// Tests
// =====================================================================
//
// 8 unit tests below cover the validator's core paths against
// hand-constructed `BoundQuery` trees. End-to-end pins (parse → bind →
// type-check → cross-substrate) live in
// `tests/cross_substrate_integration.rs`; the IN-COMMUNITY ↔
// canonical `community(...)` lowering equivalence proptest lives in
// `tests/in_community_equivalence_proptest.rs`.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;
    use crate::semantic::{BindingVisitor, StubCatalogProvider, TypeCheckVisitor};

    fn cat_full() -> StubCatalogProvider {
        StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS"])
            .with_properties(["age", "name", "embedding", "content"])
            .with_vector_index()
            .with_bm25_index()
            .with_community_index()
    }

    fn cat_bare() -> StubCatalogProvider {
        StubCatalogProvider::new()
            .with_labels(["Person", "Doc"])
            .with_rel_types(["KNOWS"])
            .with_properties(["age", "name", "embedding", "content"])
    }

    fn run<C: CatalogProvider>(input: &str, cat: &C) -> Result<(), Vec<ArcQLError>> {
        let stmt = parse(input).expect("parse");
        let mut bound = BindingVisitor::bind(&stmt, input, cat).expect("bind");
        TypeCheckVisitor::check(&mut bound, cat).expect("type-check");
        CrossSubstrateValidator::validate(&bound, cat)
    }

    // ----- substrate-availability gating -----

    #[test]
    fn unit_near_predicate_requires_vector_substrate() {
        let errs = run(
            "MATCH (n:Doc) WHERE n.embedding NEAR $q RETURN n",
            &cat_bare(),
        )
        .expect_err("vector substrate missing");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                    kind: SubstrateKind::Vector,
                    ..
                })
            )),
            "expected SubstrateUnavailable(Vector), got {errs:?}"
        );
    }

    #[test]
    fn unit_text_match_predicate_requires_bm25_substrate() {
        let errs = run(
            r#"MATCH (n:Doc) WHERE n.content MATCH "needle" RETURN n"#,
            &cat_bare(),
        )
        .expect_err("bm25 substrate missing");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                    kind: SubstrateKind::Bm25,
                    ..
                })
            )),
            "expected SubstrateUnavailable(Bm25), got {errs:?}"
        );
    }

    #[test]
    fn unit_in_community_predicate_requires_community_substrate() {
        let errs = run(
            "MATCH (n:Person) WHERE n IN COMMUNITY($cid) RETURN n",
            &cat_bare(),
        )
        .expect_err("community substrate missing");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                    kind: SubstrateKind::Community,
                    ..
                })
            )),
            "expected SubstrateUnavailable(Community), got {errs:?}"
        );
    }

    #[test]
    fn unit_community_function_requires_community_substrate() {
        let errs = run("MATCH (n:Person) RETURN community(n)", &cat_bare())
            .expect_err("community substrate missing");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::SubstrateUnavailable {
                    kind: SubstrateKind::Community,
                    ..
                })
            )),
            "expected SubstrateUnavailable(Community) for community(...), got {errs:?}"
        );
    }

    // ----- RANK BY HYBRID semantic shape -----

    #[test]
    fn unit_hybrid_missing_text_operand_is_rejected() {
        // Only VECTOR(...) — no TEXT(...). Both substrates available
        // so the only fault is the missing operand.
        let errs = run(
            "MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20)) WITH FUSION = RRF(k = 60) RETURN n",
            &cat_full(),
        )
        .expect_err("missing TEXT operand");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingOperand {
                    kind: "TEXT",
                    ..
                })
            )),
            "expected HybridMissingOperand(TEXT), got {errs:?}"
        );
    }

    #[test]
    fn unit_hybrid_missing_vector_operand_is_rejected() {
        let errs = run(
            r#"MATCH (n:Doc) RANK BY HYBRID(TEXT(n.content, "x", K = 20)) WITH FUSION = RRF(k = 60) RETURN n"#,
            &cat_full(),
        )
        .expect_err("missing VECTOR operand");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingOperand {
                    kind: "VECTOR",
                    ..
                })
            )),
            "expected HybridMissingOperand(VECTOR), got {errs:?}"
        );
    }

    #[test]
    fn unit_hybrid_missing_k_on_vector_operand_is_rejected() {
        // VECTOR(field, query) without the `K = ...` parameter.
        let errs = run(
            r#"MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q), TEXT(n.content, "x", K = 20)) WITH FUSION = RRF(k = 60) RETURN n"#,
            &cat_full(),
        )
        .expect_err("missing K on VECTOR");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                ArcQLError::CrossSubstrate(CrossSubstrateError::HybridMissingK { .. })
            )),
            "expected HybridMissingK, got {errs:?}"
        );
    }

    #[test]
    fn unit_full_hybrid_with_all_substrates_validates_clean() {
        // Vector + BM25 substrates attached, both operands present
        // with K, RRF k = 60. Should validate cleanly.
        run(
            r#"MATCH (n:Doc) RANK BY HYBRID(VECTOR(n.embedding, $q, K = 20), TEXT(n.content, "x", K = 20)) WITH FUSION = RRF(k = 60) RETURN n"#,
            &cat_full(),
        )
        .expect("full hybrid with all substrates available should validate clean");
    }
}
