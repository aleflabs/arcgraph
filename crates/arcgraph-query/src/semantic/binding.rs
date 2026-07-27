//! M4-21 binding pass — symbol resolution + scope chain.
//!
//! [`BindingVisitor::bind`] walks an AST [`Statement`] and produces a
//! [`BoundStatement`] (parallel structure preserving the M4-01
//! frozen AST surface). The visitor maintains a scope chain
//! (`Vec<Scope>`) and a source-string cursor for span computation.
//!
//! # Span computation strategy
//!
//! The AST does not carry spans (M4-01 contract — see PR #154
//! reviewer ask #7); the parser is FROZEN and cannot be modified
//! to thread spans into the AST. Therefore `BindingVisitor` re-
//! tokenizes the source on the fly via `SourceCursor`:
//!
//! For each AST node it visits, it advances the cursor past a
//! token that uniquely identifies the node (variable name, label,
//! rel-type, property name, keyword) using `str::find` from the
//! current cursor position. The cursor is monotonically advancing,
//! so repeated tokens (e.g. variable `n` appearing both at
//! declaration and at reference) get distinct spans corresponding
//! to source order.
//!
//! ## Invariants the heuristic relies on
//!
//! 1. **Visit order matches source order.** The `bind_*` walkers
//!    visit children in the same order they appear in the source
//!    (head node before tail; head before relationship before
//!    next-node; LHS expression before RHS).
//! 2. **Identifiers are not embedded substrings of other tokens
//!    visited earlier.** A label named `Person` followed by a
//!    variable `personnel` is not a problem because the visitor
//!    walks in source order; a label named `Person` after a
//!    variable `personnel` (declaration) IS shaky in pathological
//!    inputs. The integration test inputs (and the proptest's
//!    `[a-z][a-z0-9]?` strategy) are constructed to avoid such
//!    overlap. Real-world ArcQL inputs that exercise this corner
//!    will surface as "wrong span" diagnostics rather than wrong
//!    binding semantics — the binding logic itself is independent
//!    of the cursor.
//! 3. **No backtick-escaped identifiers in span tokens.** The
//!    parser's `identifier_text` strips backticks from the AST
//!    name; the source still carries the backticked form. The
//!    cursor searches for the un-backticked name, so backtick-
//!    escaped identifiers may produce a "not found" cursor result
//!    and fall back to a `Span::point(1, 1)` placeholder. The
//!    binding logic is unaffected — only the span coordinate is
//!    degraded for the rare backtick-escape case.
//!
//! ## Failure modes (acceptable trade-offs at M4-21)
//!
//! - Overlapping names (substring or backtick-escape): produce a
//!   degraded span, never a wrong binding.
//! - Source-order skew (e.g., a parser pass that re-orders
//!   children — none in M4-01): would produce wrong-but-non-empty
//!   spans; not a current risk.
//!
//! ## Future-link to M4-22
//!
//! M4-22 (type-check) consumes BoundAst's spans for diagnostics. If
//! the cursor heuristic proves insufficient (e.g., for IDE
//! integration), M4-22 may refine span tracking by exposing pest
//! pairs through a parser-internal `parse_with_spans` API. That
//! refinement is OUT of M4-21 scope — the heuristic is sufficient
//! for the M4-21 acceptance tests.
//!
//! # Scope-management semantics (openCypher)
//!
//! - Each MATCH-chain (between query start / WITH boundaries) is a
//!   single scope frame. Multiple MATCHes share the frame.
//! - WHERE within MATCH uses the current frame.
//! - WITH closes the current frame and opens a new frame containing
//!   only the projected names. Variables not in WITH's projection
//!   are dropped.
//! - RETURN sees the current frame.
//!
//! Variable references resolve via lexical lookup (nearest-
//! enclosing-scope wins). Multiple-MATCH-same-name within a single
//! frame emits [`BindingError::DuplicateBinding`].

use std::collections::HashMap;

use crate::ast::{
    BinOp, CallClause, Clause, CreateClause, CreateItem, CreateNodeSpec, CreatePathSpec,
    CreateRelSpec, DeleteClause, DeleteItem, Expression, FieldRef, Fusion, Literal,
    MapProjectionItem, MatchBody, MatchClause, MergeClause, MergePattern, NamedPath, NamedPathKind,
    NodePattern, OrderItem, PathPattern, ProjectionItem, ProjectionKind, PropertyMap, RankArg,
    RankByClause, Ranker, ReadQuery, RelPattern, RemoveClause, RemoveItem, RemoveMutation,
    ReturnClause, SetClause, SetItem, SetMutation, Statement, UnaryOp, UnionQuery, UnionTail,
    UnwindClause, WithClause, WithFusionClause,
};
use crate::error::Span;
use crate::semantic::bound_ast::*;
use crate::semantic::catalog::CatalogProvider;
use crate::semantic::error::BindingError;

/// v1.0-α dynamic-schema "unresolved label" sentinel (ADR-038
/// amendment-12, #796). An unknown label binds to this reserved id rather
/// than raising [`BindingError::UnknownLabel`], so the node pattern matches
/// NOTHING — no node carries [`arcgraph_core::LabelId::MAX`] — which is
/// exactly openCypher's "unknown label ⇒ empty match" semantics. Label ids
/// are allocated from 1 upward by every `CatalogProvider`, so `LabelId::MAX`
/// is permanently unallocated and safe as the never-matches marker. This
/// aligns labels with the property dynamic-schema fallback (ADR-038 §"Schema-id
/// resolution"); the `UnknownLabel` variant is retained for the v1.1+
/// strict-schema mode, not removed.
const UNRESOLVED_LABEL: arcgraph_core::LabelId = arcgraph_core::LabelId::MAX;
/// v1.0-α dynamic-schema "unresolved rel-type" sentinel — the relationship
/// analogue of [`UNRESOLVED_LABEL`] (ADR-038 amendment-12, #796). An unknown
/// rel-type binds here so an expand matches NO relationship (empty), per
/// openCypher "unknown rel-type ⇒ empty match".
const UNRESOLVED_REL_TYPE: arcgraph_core::TypeId = arcgraph_core::TypeId::MAX;

/// Cross-statement carry-over binding extracted from a previous
/// statement's RETURN clause per ADR-038 §5.4.1 closure (M4-83).
///
/// Mirrors the openCypher `WITH` projection-emission contract: an
/// aliased projection (`RETURN expr AS x`) emits `x`; a bare passthrough
/// (`RETURN n`) emits `n`; an unaliased non-passthrough projection
/// (`RETURN n.x`) emits nothing (no name to carry forward).
#[derive(Debug, Clone)]
pub(crate) struct CarryOverBinding {
    pub name: String,
    /// Mirrors the openCypher convention: aliased non-passthrough
    /// projections are conservatively non-nullable at v1.0; bare
    /// passthrough inherits the source binding's nullability via the
    /// emitting statement's own [`BindingVisitor::lookup_may_be_null`].
    pub may_be_null: bool,
}

// =====================================================================
// Source cursor for span computation
// =====================================================================

/// Monotonically advancing byte-position cursor into the source
/// string. Used to compute spans for AST nodes that do not carry
/// position information.
struct SourceCursor<'src> {
    source: &'src str,
    cursor: usize,
}

impl<'src> SourceCursor<'src> {
    fn new(source: &'src str) -> Self {
        Self { source, cursor: 0 }
    }

    /// Find the next occurrence of `needle` at or after the current
    /// cursor; advance the cursor past it; return the (start, end)
    /// byte range. Returns `None` when not found.
    fn advance_to(&mut self, needle: &str) -> Option<(usize, usize)> {
        if needle.is_empty() {
            return None;
        }
        let rest = self.source.get(self.cursor..)?;
        let local = rest.find(needle)?;
        let start = self.cursor + local;
        let end = start + needle.len();
        self.cursor = end;
        Some((start, end))
    }

    /// Compute a span for a (start, end) byte range by walking the
    /// source counting `\n` characters. 1-indexed line:col.
    fn span_for_range(&self, start: usize, end: usize) -> Span {
        let (sl, sc) = self.byte_to_line_col(start);
        let (el, ec) = self.byte_to_line_col(end);
        Span {
            start_line: sl,
            start_col: sc,
            end_line: el,
            end_col: ec,
        }
    }

    fn byte_to_line_col(&self, byte: usize) -> (usize, usize) {
        let mut line = 1usize;
        let mut line_start = 0usize;
        let bytes = self.source.as_bytes();
        let target = byte.min(bytes.len());
        for (i, b) in bytes.iter().enumerate().take(target) {
            if *b == b'\n' {
                line += 1;
                line_start = i + 1;
            }
        }
        let col = target - line_start + 1;
        (line, col)
    }
}

// =====================================================================
// Scope chain
// =====================================================================

#[derive(Debug)]
struct Scope {
    id: ScopeId,
    bindings: HashMap<String, BindingInfo>,
    /// Span of the clause that opened this scope. Surfaced by
    /// `BindingError::ScopeViolation` when the visitor emits one
    /// (M4-21 emits `UndeclaredVariable` instead — see
    /// `BindingError` for the rationale).
    #[allow(dead_code)]
    opened_at: Span,
}

#[derive(Debug, Clone)]
struct BindingInfo {
    binding_id: BindingId,
    declared_at: Span,
    /// Whether this binding may be NULL at runtime. Set to `true`
    /// for FRESH declarations inside an OPTIONAL MATCH clause (per
    /// ADR-006 amendment-01 + ADR-038 §2 D-21 M4-22b refinement —
    /// Shape B: binding-time may_be_null with re-reference
    /// inheritance). Re-references in pattern positions inherit the
    /// flag from the original binding via
    /// [`BindingVisitor::declare_or_resolve_in_pattern`] and never
    /// upgrade nullability.
    may_be_null: bool,
    /// The KIND of value this variable is bound to (node / relationship
    /// / path / value). openCypher v9 §2 forbids re-using a variable as
    /// a different kind (`VariableTypeConflict` / `VariableAlreadyBound`)
    /// and forbids re-using a relationship variable at all
    /// (`RelationshipUniquenessViolation`). Tracked here so the pattern
    /// binders can enforce both at COMPILE (bind) time. #618 GA Lane
    /// BINDER-VALIDATIONS.
    kind: BindingKind,
}

/// The kind of value a variable is bound to, for openCypher v9 §2
/// variable-type-conflict + relationship-uniqueness enforcement (#618).
///
/// A variable's kind is fixed at its first binding; a later use as a
/// DIFFERENT kind is a [`BindingError::VariableTypeConflict`]. A
/// relationship variable is additionally NON-re-referenceable (a second
/// relationship binding of the same name is a
/// [`BindingError::RelationshipUniquenessViolation`]); only NODE
/// variables admit re-reference (the same node may legally appear in
/// multiple pattern positions).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingKind {
    /// Bound by a node pattern `(x)`. Re-referenceable as a node.
    Node,
    /// Bound by a relationship pattern `[r]`. NOT re-referenceable.
    Rel,
    /// Bound by a named-path variable `p = (...)`. Not re-referenceable
    /// as a node/relationship.
    Path,
    /// Bound by a non-pattern position — a `WITH`/`UNWIND` projection,
    /// a `CALL {}` returned column, an expression-scoped iteration
    /// variable, etc. Matching it as a node/relationship/path is a
    /// type conflict.
    Value,
    /// A non-pattern projection whose value is the STATICALLY-NULL literal
    /// (`WITH null AS x`). Distinct from [`Value`] for ONE reason: the
    /// `null` type UNIFIES with every type, including NODE, so referencing
    /// such a variable as a node pattern anchor (`OPTIONAL MATCH (x)-...`)
    /// is NOT a static [`BindingError::VariableTypeConflict`] — it is a
    /// well-typed pattern that null-extends at runtime (openCypher 9 §6.5;
    /// TCK `expressions/path/Path1[1]` + `Path2[3]`). A NON-null value
    /// (`WITH 123 AS x`) keeps [`Value`] kind and STILL conflicts when used
    /// as an anchor (TCK `clauses/match/Match1[11]` — the conflict is the
    /// STATIC type signature `(x): NODE` and does not depend on
    /// OPTIONAL-ness). In every OTHER respect a `NullValue` behaves exactly
    /// like a `Value` (rel-position conflict, `as_str` diagnostics, etc.).
    NullValue,
    /// A cross-statement CARRY-OVER prelude binding — the name a prior
    /// statement's `RETURN` emitted, injected into the next statement's
    /// prelude scope (the implicit-`WITH`-between-statements model per
    /// ADR-038 §5.4.1, M4-83). Unlike a true `Value` binding, a
    /// carry-over does NOT block a fresh pattern binding in the new
    /// statement: `MATCH (a) RETURN a; MATCH (a) RETURN a` is two
    /// INDEPENDENT statements, so statement 2's `MATCH (a)` shadows the
    /// carried `a` with a FRESH node binding rather than conflicting.
    /// (#618 — fixes the multi-statement over-rejection the kind-conflict
    /// check would otherwise introduce; the carried kind is unknown here
    /// because the producing statement's scope is already popped, so the
    /// safe model is "shadowable prelude", matching the pre-#618
    /// behaviour exactly.)
    CarryOver,
}

impl BindingKind {
    /// Human-readable name for diagnostics.
    fn as_str(self) -> &'static str {
        match self {
            BindingKind::Node => "node",
            BindingKind::Rel => "relationship",
            BindingKind::Path => "path",
            // `NullValue` reports as "value" — to a downstream conflict
            // diagnostic it is indistinguishable from any other value.
            BindingKind::Value | BindingKind::NullValue => "value",
            BindingKind::CarryOver => "carry-over",
        }
    }
}

// =====================================================================
// BindingVisitor
// =====================================================================

/// Symbol-resolution + scope-chain visitor producing a
/// [`BoundStatement`] from an AST [`Statement`].
///
/// Construct via [`Self::bind`]; the visitor is stateful and the
/// internal struct is not part of the public API.
pub struct BindingVisitor<'cat, 'src, C: CatalogProvider> {
    catalog: &'cat C,
    cursor: SourceCursor<'src>,
    scope_chain: Vec<Scope>,
    next_binding_id: u64,
    next_scope_id: u32,
    errors: Vec<BindingError>,
    /// #836 — RETURN projection expressions paired with their minted
    /// `output_id`, set by [`Self::bind_return_clause`] and consumed by
    /// [`Self::bind_order_item`] so a RETURN-clause `ORDER BY` over a
    /// PROJECTED EXPRESSION (`RETURN p.name ORDER BY p.name`) resolves to
    /// the projected column (openCypher v9 §6.6: the sort sees the
    /// projection OUTPUT). The output-NAME scope (#618) already covers an
    /// alias / bare-identifier passthrough; this covers the UNALIASED
    /// expression case, which has no output name. Each entry is
    /// `(ast_expr, output_id, display_name)`; the AST [`Expression`]
    /// equality is span-free (see `impl PartialEq for ProjectionItem`) so
    /// the match is structural. Valid ONLY for the tail `ORDER BY` /
    /// `SKIP` / `LIMIT` immediately following the RETURN that built it —
    /// [`Self::bind_clause`] clears it at the start of every non-tail
    /// clause and [`Self::bind_union_tail`] clears it (union tails keep
    /// their unchanged name-scope behavior).
    return_tail_outputs: Vec<(Expression, BindingId, String)>,
    /// #1053 — a DEFERRED removal of an aggregating `WITH`'s pre-projection
    /// (input) scope frame. openCypher v9 §6.6 lets the `ORDER BY`
    /// immediately following an AGGREGATING `WITH` (`WITH me.age AS age,
    /// count(you.age) AS cnt ORDER BY me.age + count(you.age)`) contain an
    /// aggregate whose ARGUMENT (`you.age`) references a pre-projection
    /// variable — that aggregate is computed at the `WITH` boundary, where
    /// the input variable is still live. The `WITH` fence normally removes
    /// the input frame as the clause closes (so the next reading clause sees
    /// only the projected outputs); for an AGGREGATING `WITH` we DEFER that
    /// removal so the immediately-following tail `ORDER BY` / `SKIP` /
    /// `LIMIT` can still resolve the aggregate's input-scoped argument. The
    /// removal is flushed (`flush_pending_with_fence`) at the start of every
    /// NON-tail clause, so the fence is preserved against the next reading
    /// clause. `Some(input_scope_index)` records the frame's position in
    /// `scope_chain` at the time the `WITH` closed.
    pending_with_fence: Option<usize>,
    /// #1053 — the implicit GROUP BY keys of the immediately-preceding
    /// AGGREGATING `RETURN` / `WITH`, set by [`Self::bind_return_clause`] /
    /// [`Self::bind_with_clause`] and consumed by [`Self::bind_order_item`].
    /// openCypher v9 §6.6 lets the tail `ORDER BY` of an aggregating
    /// projection contain an aggregate (`ORDER BY me.age + count(you.age)`):
    /// that aggregate is computed alongside the projection's aggregates, and
    /// every NON-aggregated leaf in the sort key must itself be a grouping key
    /// — the SAME rule [`Self::check_aggregation_grouping`] enforces on the
    /// projection. `None` ⇒ the preceding projection was NOT aggregating (or
    /// there is none), so ANY aggregate in `ORDER BY` is rejected
    /// (`ReturnOrderBy2` [14] / `WithOrderBy2` [25] `InvalidAggregation`).
    /// `Some(keys)` carries the grouping-key AST expressions for the
    /// non-grouping-leaf validation. Cleared (set `None`) at the start of
    /// every NON-tail clause, exactly like [`Self::return_tail_outputs`].
    tail_grouping_context: Option<Vec<Expression>>,
}

impl<'cat, 'src, C: CatalogProvider> BindingVisitor<'cat, 'src, C> {
    /// Bind a parsed [`Statement`].
    ///
    /// `source` is the original input string — required because the
    /// AST does not carry spans (see top-of-file comment for span
    /// computation strategy).
    ///
    /// Returns `Ok(BoundStatement)` on success (no binding errors)
    /// or `Err(Vec<BindingError>)` with one or more errors. The
    /// visitor accumulates ALL errors found in a single pass; it
    /// does NOT short-circuit on the first error.
    pub fn bind(
        stmt: &Statement,
        source: &'src str,
        catalog: &'cat C,
    ) -> Result<BoundStatement, Vec<BindingError>> {
        let mut v = Self {
            catalog,
            cursor: SourceCursor::new(source),
            scope_chain: Vec::new(),
            next_binding_id: 0,
            next_scope_id: 0,
            errors: Vec::new(),
            return_tail_outputs: Vec::new(),
            pending_with_fence: None,
            tail_grouping_context: None,
        };
        let bound = v.bind_statement(stmt);
        if v.errors.is_empty() {
            Ok(bound)
        } else {
            Err(v.errors)
        }
    }

    /// Bind a multi-statement query per ADR-038 §5.4.1 closure (M4-83).
    ///
    /// Each statement is bound left-to-right; the RETURN-emitted
    /// bindings of statement N flow into statement N+1's outer scope as
    /// a "prelude" frame so cross-statement variable references resolve
    /// (`MATCH (n) RETURN n.name AS pname; MATCH (m) WHERE m.name = pname RETURN m`).
    /// The semantics mirror an implicit `WITH` between statements:
    ///
    /// - Aliased projection (`RETURN expr AS x`) → `x` visible next.
    /// - Bare passthrough (`RETURN n`) → `n` visible next.
    /// - Unaliased non-passthrough (`RETURN n.x`) → nothing emitted.
    ///
    /// All errors from all statements are accumulated; the function
    /// does NOT short-circuit on the first error so the caller surfaces
    /// the complete diagnostic set in one shot.
    ///
    /// # Snapshot LSN
    ///
    /// Per amendment-03 §TIER-1 GAP E rule 2 the snapshot LSN is shared
    /// across all statements at execute time; the bind layer reserves
    /// the [`BoundQuery::snapshot_lsn`] field as `None` on every
    /// statement (just as the single-statement [`Self::bind`] path
    /// does). The shared-LSN load lives on
    /// [`crate::materialize::materialize_multi`] (executor primitive) /
    /// [`crate::QueryEngine::execute_multi`] (M5↔M4 surface) via a
    /// single [`crate::executor::ExecutionContext`] threaded through
    /// every statement's materialize call.
    ///
    /// # Errors
    ///
    /// On any binding failure across any statement, returns the
    /// concatenated error vec; the partial bound vec is discarded. This
    /// matches the single-statement [`Self::bind`] convention.
    pub fn bind_multi(
        stmts: &[Statement],
        source: &'src str,
        catalog: &'cat C,
    ) -> Result<Vec<BoundStatement>, Vec<BindingError>> {
        let mut v = Self {
            catalog,
            cursor: SourceCursor::new(source),
            scope_chain: Vec::new(),
            next_binding_id: 0,
            next_scope_id: 0,
            errors: Vec::new(),
            return_tail_outputs: Vec::new(),
            pending_with_fence: None,
            tail_grouping_context: None,
        };
        let mut bound_stmts: Vec<BoundStatement> = Vec::with_capacity(stmts.len());
        let mut carry_over: Vec<CarryOverBinding> = Vec::new();

        for stmt in stmts {
            // Inject prior-statement RETURN-emitted names as a prelude
            // scope BEFORE the statement's own root scope opens.
            // bind_read_query opens its root scope on top, so lexical
            // lookups walk root → prelude → resolved-or-undeclared.
            //
            // Even when carry_over is empty (first statement, or a
            // prior statement with no RETURN-emitted names), we always
            // push an empty prelude so the iteration is symmetric and
            // the chain depth at the start of every bind_read_query is
            // identical (defense-in-depth — bind_read_query's defensive
            // pop-everything cleanup zeroes the chain at the end of
            // each iteration regardless).
            let prelude_span = v.whole_source_span();
            v.push_scope(prelude_span);
            // Span for prelude declarations is a 1:1 placeholder — the
            // real declaration site is the prior statement's RETURN
            // clause; the cross-statement span pin is forward-deferred
            // to v1.1 (matches the M4-21 cursor heuristic's "degraded
            // span never wrong binding" trade-off doc'd at the top of
            // this file).
            for entry in &carry_over {
                // Declare carry-over names with `BindingKind::CarryOver`
                // (NOT `Value`) so a subsequent statement's pattern can
                // SHADOW them with a fresh binding rather than tripping
                // the #618 kind-conflict check (multi-statement
                // independence per ADR-038 §5.4.1).
                let _ = v.declare_with_kind(
                    &entry.name,
                    Span::point(1, 1),
                    entry.may_be_null,
                    BindingKind::CarryOver,
                );
            }

            let bound = v.bind_statement(stmt);
            // Pop any remaining prelude frame defensively.
            while !v.scope_chain.is_empty() {
                v.pop_scope();
            }
            // Capture this statement's RETURN-emitted bindings as the
            // next iteration's carry-over.
            carry_over = extract_returned_bindings(&bound);
            bound_stmts.push(bound);
        }

        if v.errors.is_empty() {
            Ok(bound_stmts)
        } else {
            Err(v.errors)
        }
    }

    // ---------- Scope helpers ----------

    fn fresh_binding_id(&mut self) -> BindingId {
        let id = BindingId::new(self.next_binding_id);
        self.next_binding_id += 1;
        id
    }

    fn fresh_scope_id(&mut self) -> ScopeId {
        let id = ScopeId::new(self.next_scope_id);
        self.next_scope_id += 1;
        id
    }

    fn push_scope(&mut self, opened_at: Span) -> ScopeId {
        let id = self.fresh_scope_id();
        self.scope_chain.push(Scope {
            id,
            bindings: HashMap::new(),
            opened_at,
        });
        id
    }

    fn pop_scope(&mut self) {
        self.scope_chain.pop();
    }

    /// #1053 — flush a DEFERRED aggregating-`WITH` input-scope removal (see
    /// [`Self::pending_with_fence`]). Removes the recorded pre-projection
    /// frame from `scope_chain` IF it is still present (bounds-checked: the
    /// index was captured before the frame was popped/reordered; a defensive
    /// `< len` guard makes a stale index inert). Idempotent — a no-op when
    /// nothing is pending. Called at the start of every non-tail clause (so
    /// the next reading clause sees only the `WITH` outputs) and in
    /// [`Self::bind_read_query`]'s defensive end-of-clauses cleanup.
    fn flush_pending_with_fence(&mut self) {
        if let Some(idx) = self.pending_with_fence.take() {
            if idx < self.scope_chain.len() {
                self.scope_chain.remove(idx);
            }
        }
    }

    fn current_scope_id(&self) -> ScopeId {
        self.scope_chain
            .last()
            .map(|s| s.id)
            .unwrap_or_else(|| ScopeId::new(0))
    }

    /// Declare a new binding in the innermost scope. Emits
    /// [`BindingError::DuplicateBinding`] if the name is already
    /// bound there. Always returns a fresh [`BindingId`] (even on
    /// duplicate — the BoundAst still gets a unique id so
    /// downstream walkers don't trip on shared ids).
    ///
    /// `may_be_null` is recorded in the resulting [`BindingInfo`]
    /// (per ADR-038 §2 D-21 M4-22b — Shape B). For non-pattern
    /// positions (named-path var, UNWIND var, fresh WITH-projected
    /// names of non-passthrough projections), pass `false`. WITH
    /// passthrough (`WITH n` or `WITH n AS x`) inherits the flag
    /// from the pre-WITH binding via
    /// [`BindingVisitor::lookup_may_be_null`].
    fn declare(&mut self, name: &str, span: Span, may_be_null: bool) -> BindingId {
        // Non-pattern declarations default to `BindingKind::Value`. The
        // pattern/path binders pass the precise kind via
        // [`Self::declare_with_kind`].
        self.declare_with_kind(name, span, may_be_null, BindingKind::Value)
    }

    /// As [`Self::declare`], but records the binding's [`BindingKind`].
    /// Emits [`BindingError::DuplicateBinding`] on a same-scope name
    /// collision (the existing M4-21 contract). #618.
    fn declare_with_kind(
        &mut self,
        name: &str,
        span: Span,
        may_be_null: bool,
        kind: BindingKind,
    ) -> BindingId {
        let id = self.fresh_binding_id();
        if let Some(scope) = self.scope_chain.last_mut() {
            if let Some(prior) = scope.bindings.get(name) {
                self.errors.push(BindingError::DuplicateBinding {
                    name: name.to_string(),
                    span: span.clone(),
                    prior_span: prior.declared_at.clone(),
                    reason: String::new(),
                });
            } else {
                scope.bindings.insert(
                    name.to_string(),
                    BindingInfo {
                        binding_id: id,
                        declared_at: span,
                        may_be_null,
                        kind,
                    },
                );
            }
        }
        id
    }

    /// Pattern-position binding with openCypher v9 §2 kind enforcement
    /// (#618). `position_kind` is the kind of the CURRENT pattern
    /// position ([`BindingKind::Node`] for `(x)`, [`BindingKind::Rel`]
    /// for `[r]`).
    ///
    /// Resolution rules (per openCypher v9 §2):
    /// - **Re-reference (existing kind == position_kind == Node):** the
    ///   same node may appear in multiple positions — return the
    ///   existing `BindingId` + `may_be_null` verbatim (re-references
    ///   INHERIT nullability and never upgrade it; the M4-22b Shape-B
    ///   refinement of ADR-038 §2 D-21).
    /// - **Relationship re-binding (position_kind == Rel and the name is
    ///   already bound — to ANY kind):** a relationship variable cannot
    ///   be re-used. If the prior is also a relationship →
    ///   [`BindingError::RelationshipUniquenessViolation`]; otherwise →
    ///   [`BindingError::VariableTypeConflict`].
    /// - **Kind mismatch (existing kind != position_kind):** e.g. a
    ///   relationship/path/value variable re-used as a node, or a
    ///   node/path/value variable re-used as a relationship →
    ///   [`BindingError::VariableTypeConflict`].
    /// - **Fresh:** declare in the innermost scope with the position's
    ///   kind; nullability follows the enclosing clause's optional-ness.
    ///
    /// Returns `(binding_id, may_be_null)`. On a conflict the error is
    /// pushed and a FRESH id is returned (the accumulate-all-errors
    /// convention — the BoundAst still gets a unique id so downstream
    /// walkers do not trip on shared ids).
    fn declare_or_resolve_in_pattern(
        &mut self,
        name: &str,
        span: Span,
        in_optional_clause: bool,
        position_kind: BindingKind,
    ) -> (BindingId, bool) {
        // Scope-chain lookup mirrors `resolve` (nearest-enclosing wins).
        if let Some((prior_kind, prior_span, prior_id, prior_nullable)) =
            self.scope_chain.iter().rev().find_map(|scope| {
                scope
                    .bindings
                    .get(name)
                    .map(|i| (i.kind, i.declared_at.clone(), i.binding_id, i.may_be_null))
            })
        {
            // A cross-statement CARRY-OVER prelude binding does NOT block
            // a fresh pattern binding: each `;`-separated statement is
            // INDEPENDENT, so a pattern position SHADOWS the carried name
            // with a fresh binding in the current scope (no conflict).
            // This preserves the pre-#618 multi-statement semantics
            // (`MATCH (a) RETURN a; MATCH (a) RETURN a`). #618.
            if prior_kind == BindingKind::CarryOver {
                let id = self.declare_with_kind(name, span, in_optional_clause, position_kind);
                return (id, in_optional_clause);
            }
            // A relationship position can NEVER re-reference an existing
            // binding (relationship-uniqueness).
            if position_kind == BindingKind::Rel {
                if prior_kind == BindingKind::Rel {
                    self.errors
                        .push(BindingError::RelationshipUniquenessViolation {
                            name: name.to_string(),
                            span,
                            prior_span,
                        });
                } else {
                    self.errors.push(BindingError::VariableTypeConflict {
                        name: name.to_string(),
                        new_kind: BindingKind::Rel.as_str(),
                        prior_kind: prior_kind.as_str(),
                        span,
                        prior_span,
                    });
                }
                return (self.fresh_binding_id(), in_optional_clause);
            }
            // A node position re-references ONLY an existing node.
            if position_kind == BindingKind::Node && prior_kind == BindingKind::Node {
                return (prior_id, prior_nullable);
            }
            // A node position anchored on a prior STATICALLY-NULL binding
            // (`WITH null AS a`, kind [`BindingKind::NullValue`]) inside an
            // OPTIONAL MATCH is NOT a type conflict: the `null` type unifies
            // with NODE, so `(a)` is well-typed (anchor `a: NODE` is
            // satisfied by a null value). At runtime a null anchor matches
            // no node, so the OPTIONAL MATCH null-extends (openCypher 9
            // §6.5). This is the path that makes `WITH null AS a OPTIONAL
            // MATCH p = (a)-[r]->() RETURN nodes(p)` emit one `[null]` row
            // (TCK `expressions/path/Path1[1]` + `Path2[3]`). We resolve to
            // the PRIOR binding id (so the LeftOuterJoin's shared-binding
            // join pivots on `a`; the null `a` equals no scanned node, so
            // the right side is empty and the join null-extends) and mark
            // the binding nullable.
            //
            // CRITICAL — this is gated on `NullValue`, NOT `Value`. A
            // NON-null value bound as a node anchor (`WITH 123 AS a`,
            // `WITH [1,2] AS a`, `WITH {x:1} AS a`) keeps `Value` kind and
            // STILL raises a `VariableTypeConflict` — in OPTIONAL just as
            // in a required MATCH — because the conflict is the STATIC type
            // signature `(a): NODE` (a non-null scalar/list/map does not
            // unify with NODE) and does NOT depend on OPTIONAL-ness.
            // OPTIONAL changes the RUNTIME null-extension behaviour, not the
            // static type check. TCK `clauses/match/Match1[11]` pins the
            // non-null `true,123,123.4,'foo',[],[10],{x:1},{x:[]}` matrix as
            // a compile-time conflict; the required-MATCH form is enforced
            // by the catch-all below, and the OPTIONAL form is now covered
            // by an adversarial e2e (`null_anchor_optional_named_path_e2e`).
            //
            // `in_optional_clause` is retained as a guard so the (untested,
            // TCK-silent) `WITH null AS a MATCH (a)` required form continues
            // to take the catch-all conflict arm rather than silently
            // null-extending to zero rows — a strictly conservative choice
            // that ships only the behaviour the Path1/Path2 scenarios pin.
            if position_kind == BindingKind::Node
                && prior_kind == BindingKind::NullValue
                && in_optional_clause
            {
                return (prior_id, true);
            }
            // Any other combination is a kind conflict (rel/path/value
            // re-used as a node; node/path/value re-used differently).
            self.errors.push(BindingError::VariableTypeConflict {
                name: name.to_string(),
                new_kind: position_kind.as_str(),
                prior_kind: prior_kind.as_str(),
                span,
                prior_span,
            });
            return (self.fresh_binding_id(), in_optional_clause);
        }
        // Fresh declaration: nullability follows the enclosing
        // clause's optional-ness.
        let id = self.declare_with_kind(name, span, in_optional_clause, position_kind);
        (id, in_optional_clause)
    }

    /// Resolve `name` via lexical lookup (nearest-enclosing wins).
    /// Emits [`BindingError::UndeclaredVariable`] on miss.
    fn resolve(&mut self, name: &str, ref_span: Span) -> Option<BindingId> {
        for scope in self.scope_chain.iter().rev() {
            if let Some(info) = scope.bindings.get(name) {
                return Some(info.binding_id);
            }
        }
        self.errors.push(BindingError::UndeclaredVariable {
            name: name.to_string(),
            span: ref_span,
        });
        None
    }

    /// Look up `name`'s `may_be_null` in the current scope chain
    /// without emitting an error on miss. Used by
    /// [`Self::bind_with_clause`] to thread nullability through
    /// passthrough projections (`WITH n` / `WITH n AS x`).
    fn lookup_may_be_null(&self, name: &str) -> Option<bool> {
        for scope in self.scope_chain.iter().rev() {
            if let Some(info) = scope.bindings.get(name) {
                return Some(info.may_be_null);
            }
        }
        None
    }

    /// Look up `name`'s [`BindingKind`] in the current scope chain
    /// without emitting an error on miss. Used by
    /// [`Self::bind_with_clause`] / [`Self::bind_unwind_clause`] to
    /// thread the kind through passthrough projections so a later
    /// pattern position correctly re-references a node (`WITH n`) or
    /// conflicts on a value (`WITH 1 AS n`). #618.
    fn lookup_kind(&self, name: &str) -> Option<BindingKind> {
        for scope in self.scope_chain.iter().rev() {
            if let Some(info) = scope.bindings.get(name) {
                return Some(info.kind);
            }
        }
        None
    }

    fn lookup_binding_info(&self, name: &str) -> Option<BindingInfo> {
        self.scope_chain
            .iter()
            .rev()
            .find_map(|scope| scope.bindings.get(name).cloned())
    }

    // ---------- Span helpers ----------

    fn span_for_token(&mut self, token: &str) -> Span {
        match self.cursor.advance_to(token) {
            Some((s, e)) => self.cursor.span_for_range(s, e),
            None => Span::point(1, 1),
        }
    }

    fn whole_source_span(&self) -> Span {
        let end = self.cursor.source.len();
        let (el, ec) = self.cursor.byte_to_line_col(end);
        Span {
            start_line: 1,
            start_col: 1,
            end_line: el,
            end_col: ec,
        }
    }

    // ---------- Top-level ----------

    fn bind_statement(&mut self, s: &Statement) -> BoundStatement {
        match s {
            Statement::Read(q) => BoundStatement::Read(self.bind_read_query(q)),
            // #830 (ADR-198 §OQ-7) — Neo4j-compatible index DDL. Bound
            // as a pass-through (the parsed label / property are carried
            // through verbatim); the OPTIONS map is NOT bound as a query
            // expression (it is index-build config for the vector track,
            // not query semantics). The type-checker surfaces the typed
            Statement::IndexDdl(d) => BoundStatement::IndexDdl(d.clone()),
            // EXPLAIN / PROFILE are planner-control wrappers per
            // ADR-038 §2 D-19 + amendment-03 §TIER-1 GAP B (M4-91).
            // The wrapper bit is consumed by the
            // [`crate::explain`] entry points; the binding pass
            // strips it and produces the same `BoundStatement::Read`
            // shape so type-check / cross-substrate / lowering remain
            // wrapper-agnostic.
            Statement::Explain(q) | Statement::Profile(q) => {
                BoundStatement::Read(self.bind_read_query(q))
            }
            // ADR-185 (#649-A1, W28) — UNION / UNION ALL. Each arm is
            // bound independently; the column-compatibility + no-mixing
            // rules (openCypher v9 §8) are enforced inside.
            Statement::Union(u) => BoundStatement::Union(Box::new(self.bind_union_query(u))),
        }
    }

    fn bind_read_query(&mut self, q: &ReadQuery) -> BoundQuery {
        let query_span = self.whole_source_span();
        let root_scope = self.push_scope(query_span.clone());

        let mut clauses = Vec::with_capacity(q.clauses.len());
        for c in &q.clauses {
            clauses.push(self.bind_clause(c));
        }

        // #1053 — clear any DEFERRED aggregating-`WITH` input frame that the
        // final clause's tail left pending (a query ending in `WITH … ORDER
        // BY` over an aggregate), so the marker does not leak across the
        // defensive scope teardown below or (in `bind_multi`) into the next
        // statement.
        self.pending_with_fence = None;

        // Defensive cleanup: pop any scopes we left open.
        while !self.scope_chain.is_empty() {
            self.pop_scope();
        }

        BoundQuery {
            clauses,
            root_scope,
            span: query_span,
            tenant: self.catalog.tenant(),
            partition: self.catalog.partition(),
            // Per ADR-038 amendment-03 §TIER-1 GAP E (D-18 rule 1):
            // M4-61 acquires the snapshot LSN at execute-time, before
            // the first operator pulls a batch. M4-21 reserves the
            // field; the value is always None at bind time.
            snapshot_lsn: None,
        }
    }

    /// Bind a `UNION` / `UNION ALL` query per ADR-185 (#649-A1, W28 —
    /// openCypher v9 §8). Each arm is bound in its OWN fresh root scope
    /// (arms are independent — no cross-arm variable visibility). The
    /// post-union tail (ORDER BY / SKIP / LIMIT) binds against the
    /// FIRST arm's terminal scope: §8 requires every arm to expose the
    /// same column names, so arm-0 is representative, and the union's
    /// output schema is arm-0's projection. Binding the tail before
    /// arm-0's scope cleanup makes `ORDER BY <col>` resolve EXACTLY as
    /// it would for a standalone arm-0 query + tail.
    ///
    /// The two §8 structural rules — column-compatibility (same name
    /// set across arms) and no-mixing (`UNION` vs `UNION ALL` cannot be
    /// combined) — are enforced here; on violation the corresponding
    /// [`BindingError`] is pushed and `bind()` returns `Err`.
    fn bind_union_query(&mut self, u: &UnionQuery) -> BoundUnionQuery {
        let union_span = self.whole_source_span();
        let mut arms: Vec<BoundQuery> = Vec::with_capacity(u.arms.len());
        let mut tail = BoundUnionTail::default();

        for (idx, arm) in u.arms.iter().enumerate() {
            let query_span = self.whole_source_span();
            let root_scope = self.push_scope(query_span.clone());
            let mut clauses = Vec::with_capacity(arm.clauses.len());
            for c in &arm.clauses {
                clauses.push(self.bind_clause(c));
            }
            // Whole-union tail binds in arm-0's terminal scope (before
            // the defensive cleanup pop below).
            if idx == 0 {
                tail = self.bind_union_tail(&u.tail);
            }
            // Defensive cleanup: pop any scopes this arm left open
            // (mirrors `bind_read_query`). Arms do NOT leak scope into
            // one another.
            while !self.scope_chain.is_empty() {
                self.pop_scope();
            }
            arms.push(BoundQuery {
                clauses,
                root_scope,
                span: query_span,
                tenant: self.catalog.tenant(),
                partition: self.catalog.partition(),
                snapshot_lsn: None,
            });
        }

        // --- openCypher v9 §8 structural validation ---------------
        self.check_union_no_mixing(u, &union_span);
        let column_orders = self.check_union_column_compat(u, &union_span);

        BoundUnionQuery {
            arms,
            all: u.all.clone(),
            column_orders,
            tail,
            span: union_span,
        }
    }

    /// Bind the post-union tail (ORDER BY / SKIP / LIMIT) using the
    /// SAME per-item binders as the read-query tail, so the bound shape
    /// and downstream lowering are identical (only the BINDING locus,
    /// whole-union vs last-clause, differs). Caller binds this in
    /// arm-0's terminal scope.
    fn bind_union_tail(&mut self, t: &UnionTail) -> BoundUnionTail {
        // #836 — a union tail ORDER BY resolves against the union's output
        // COLUMN NAMES (arm-0's projection), handled by the established
        // name-scope. Clear the per-RETURN projected-expression map (the
        // last arm's RETURN set it) so union ordering keeps its unchanged
        // behavior — the #836 expression rewrite targets only the
        // single-query RETURN tail.
        self.return_tail_outputs.clear();
        // #1053 — a union tail keeps its unchanged behavior (an aggregate in
        // a union ORDER BY stays rejected); no grouping context is exposed.
        self.tail_grouping_context = None;
        let order_by: Vec<BoundOrderItem> =
            t.order_by.iter().map(|o| self.bind_order_item(o)).collect();
        // #618 — SKIP/LIMIT constant-ness validation on the union tail
        // (openCypher v9 §6.4), mirroring the standalone tail clauses.
        if let Some(e) = t.skip.as_ref() {
            let span = self.span_for_token("SKIP");
            self.check_skip_limit_expr(e, "SKIP", span);
        }
        if let Some(e) = t.limit.as_ref() {
            let span = self.span_for_token("LIMIT");
            self.check_skip_limit_expr(e, "LIMIT", span);
        }
        let skip = t.skip.as_ref().map(|e| self.bind_expression(e));
        let limit = t.limit.as_ref().map(|e| self.bind_expression(e));
        BoundUnionTail {
            order_by,
            skip,
            limit,
        }
    }

    /// Reject mixing `UNION` and `UNION ALL` in one union (openCypher
    /// v9 §8 — TCK `Union3` → `InvalidClauseComposition`). Emits
    /// [`BindingError::UnionMixedSetOps`] if the per-boundary flags are
    /// not all equal.
    fn check_union_no_mixing(&mut self, u: &UnionQuery, span: &Span) {
        let mixed = u.all.windows(2).any(|w| w[0] != w[1]);
        if mixed {
            self.errors
                .push(BindingError::UnionMixedSetOps { span: span.clone() });
        }
    }

    /// Enforce union column-compatibility (openCypher v9 §8 — every arm
    /// projects the same result-column NAME set, order-independent) AND
    /// compute the per-arm column permutation used by the executor to
    /// realign differently-ordered arms. Emits
    /// [`BindingError::UnionColumnMismatch`] for the first arm whose
    /// name set differs from arm 1's (maps to TCK
    /// `DifferentColumnsInUnion`); on mismatch the returned permutation
    /// is unused (the bind returns `Err`). The canonical column order
    /// is arm 1's RETURN order; `result[i][j]` is the position in arm
    /// `i` that supplies canonical column `j`.
    fn check_union_column_compat(&mut self, u: &UnionQuery, span: &Span) -> Vec<Vec<usize>> {
        let Some(first_arm) = u.arms.first() else {
            return Vec::new();
        };
        let canonical = terminal_return_column_names(first_arm);
        let canonical_set: std::collections::BTreeSet<&String> = canonical.iter().collect();
        let mut orders: Vec<Vec<usize>> = Vec::with_capacity(u.arms.len());
        // Arm 0 is the canonical order → identity permutation.
        orders.push((0..canonical.len()).collect());
        for (i, arm) in u.arms.iter().enumerate().skip(1) {
            let names = terminal_return_column_names(arm);
            let set: std::collections::BTreeSet<&String> = names.iter().collect();
            if set != canonical_set {
                let mut a = canonical.clone();
                a.sort();
                let mut b = names.clone();
                b.sort();
                self.errors.push(BindingError::UnionColumnMismatch {
                    first: a,
                    mismatching: b,
                    arm_index: i + 1,
                    span: span.clone(),
                });
                // Report only the first mismatch — one structured
                // diagnostic is the IDE-grade signal. Return the
                // identity-padded permutation (unused: bind returns
                // Err on the pushed error).
                orders.push((0..canonical.len()).collect());
                return orders;
            }
            // Compatible: map canonical column j → this arm's source
            // position. Names are unique within an arm at v1.0 (dup
            // result-column names are a separate RETURN-binding
            // concern); `position` takes the first match defensively.
            let perm: Vec<usize> = canonical
                .iter()
                .map(|cname| names.iter().position(|n| n == cname).unwrap_or(0))
                .collect();
            orders.push(perm);
        }
        orders
    }

    fn bind_clause(&mut self, c: &Clause) -> BoundClause {
        // #836 — the RETURN→ORDER-BY projected-expression map is valid
        // ONLY for the tail `ORDER BY` / `SKIP` / `LIMIT` clauses the
        // parser emits immediately after a RETURN. Clear it at the start
        // of every NON-tail clause so a subquery / earlier RETURN's
        // outputs can't leak into an unrelated sort key. `bind_return_clause`
        // re-populates it (the RETURN arm clears here first, then sets).
        if !matches!(
            c,
            Clause::TailOrderBy(_) | Clause::TailSkip(_) | Clause::TailLimit(_)
        ) {
            self.return_tail_outputs.clear();
            // #1053 — the aggregating-projection grouping context is valid
            // ONLY for the immediately-following tail `ORDER BY`; clear it on
            // every non-tail clause (mirrors `return_tail_outputs`).
            self.tail_grouping_context = None;
            // #1053 — flush any DEFERRED aggregating-`WITH` input frame
            // (`pending_with_fence`) before the next reading clause, so the
            // `WITH` pipeline fence is preserved: only the tail `ORDER BY` /
            // `SKIP` / `LIMIT` immediately following the aggregating `WITH`
            // sees the pre-projection scope; this MATCH / WITH / RETURN does
            // not.
            self.flush_pending_with_fence();
        }
        match c {
            Clause::Match(m) => BoundClause::Match(self.bind_match_clause(m, false)),
            Clause::OptionalMatch(m) => BoundClause::Match(self.bind_match_clause(m, true)),
            Clause::Create(c) => BoundClause::Create(self.bind_create_clause(c)),
            Clause::Delete(d) => BoundClause::Delete(self.bind_delete_clause(d)),
            Clause::Set(s) => BoundClause::Set(self.bind_set_clause(s)),
            Clause::Remove(r) => BoundClause::Remove(self.bind_remove_clause(r)),
            Clause::Merge(m) => BoundClause::Merge(self.bind_merge_clause(m)),
            Clause::With(w) => BoundClause::With(self.bind_with_clause(w)),
            Clause::Unwind(u) => BoundClause::Unwind(self.bind_unwind_clause(u)),
            Clause::Call(c) => BoundClause::Call(self.bind_call_clause(c)),
            Clause::CallProcedure(c) => {
                BoundClause::CallProcedure(self.bind_call_procedure_clause(c))
            }
            Clause::Show(s) => BoundClause::Show(self.bind_show_clause(s)),
            Clause::RankBy(r) => BoundClause::RankBy(self.bind_rank_by_clause(r)),
            Clause::WithFusion(f) => BoundClause::WithFusion(self.bind_with_fusion_clause(f)),
            Clause::Return(r) => BoundClause::Return(self.bind_return_clause(r)),
            Clause::TailOrderBy(items) => {
                let span = self.span_for_token("ORDER");
                let bound: Vec<BoundOrderItem> =
                    items.iter().map(|o| self.bind_order_item(o)).collect();
                BoundClause::TailOrderBy(bound, span)
            }
            Clause::TailSkip(e) => {
                let span = self.span_for_token("SKIP");
                // #618 — SKIP constant-ness / non-negative / integer
                // validation (openCypher v9 §6.4) BEFORE binding, so it
                // pre-empts the executor SKIP-NotImplemented.
                self.check_skip_limit_expr(e, "SKIP", span.clone());
                BoundClause::TailSkip(self.bind_expression(e), span)
            }
            Clause::TailLimit(e) => {
                let span = self.span_for_token("LIMIT");
                // #618 — LIMIT constant-ness / non-negative / integer
                // validation (openCypher v9 §6.4).
                self.check_skip_limit_expr(e, "LIMIT", span.clone());
                BoundClause::TailLimit(self.bind_expression(e), span)
            }
        }
    }

    // ---------- CREATE (ADR-147 W26-θ Phase 1) ----------

    fn bind_create_clause(&mut self, c: &CreateClause) -> BoundCreateClause {
        let span = self.span_for_token("CREATE");
        let items = c
            .items
            .iter()
            .map(|item| self.bind_create_item(item))
            .collect();
        BoundCreateClause { items, span }
    }

    fn bind_create_item(&mut self, item: &CreateItem) -> BoundCreateItem {
        match item {
            CreateItem::Node(spec) => BoundCreateItem::Node(self.bind_create_node_spec(spec)),
            CreateItem::Path(path) => BoundCreateItem::Path(self.bind_create_path_spec(path)),
        }
    }

    // ADR-148 W26-θ Phase 2 + Phase-5 forward-pin: CREATE-path endpoints
    // bind in source order. A bare endpoint variable that is already a
    // node binding references the incoming row; an endpoint with labels
    // or properties remains a fresh declaration site and cannot
    // re-declare an existing variable.
    fn bind_create_path_spec(&mut self, path: &CreatePathSpec) -> BoundCreatePathSpec {
        let source = self.bind_create_path_endpoint_spec(&path.source);
        let rel = self.bind_create_rel_spec(&path.rel);
        let target = self.bind_create_path_endpoint_spec(&path.target);
        // Span: prefer the source node's span; the source is bound
        // first so its span is the path's left edge.
        let span = source.span.clone();
        BoundCreatePathSpec {
            source,
            rel,
            target,
            span,
        }
    }

    fn bind_create_rel_spec(&mut self, spec: &CreateRelSpec) -> BoundCreateRelSpec {
        // CREATE-rel binding mirrors CREATE-node: FRESH non-nullable
        // declaration. Per ADR-148 §D-3 the rel-var is a fresh
        // declaration (re-using a prior MATCH-bound variable is
        // illegal at Phase 2).
        let var = spec.var.as_ref().map(|name| {
            let span = self.span_for_token(name);
            let binding_id = self.declare(name, span.clone(), false);
            BoundVariable {
                name: name.clone(),
                binding_id,
                may_be_null: false,
                span,
                type_info: None,
            }
        });
        let label = spec.label.clone();
        let properties = spec.properties.as_ref().map(|m| self.bind_property_map(m));
        let span = var
            .as_ref()
            .map(|v| v.span.clone())
            .unwrap_or_else(|| self.span_for_token(&spec.label));
        BoundCreateRelSpec {
            var,
            label,
            properties,
            direction: spec.direction.clone(),
            span,
        }
    }

    fn bind_create_node_spec(&mut self, spec: &CreateNodeSpec) -> BoundCreateNodeSpec {
        // CREATE introduces a FRESH binding for the variable per
        // openCypher v9 §6: re-using an existing variable from a
        // prior MATCH is illegal at Phase 1 (CreateRel may relax this
        // at Phase 2). `declare` emits `BindingError::DuplicateBinding`
        // on conflict — exactly the openCypher v9 contract.
        let var = spec.var.as_ref().map(|name| {
            let span = self.span_for_token(name);
            // CREATE-introduced bindings are non-nullable: a CREATE
            // ALWAYS produces a row carrying the new node-id.
            let binding_id = if let Some(info) = self.lookup_binding_info(name) {
                let reason = if spec.label.is_some() || spec.properties.is_some() {
                    format!(
                        "; variable `{name}` already bound — cannot re-declare with labels/properties in CREATE"
                    )
                } else {
                    String::new()
                };
                self.errors.push(BindingError::DuplicateBinding {
                    name: name.clone(),
                    span: span.clone(),
                    prior_span: info.declared_at,
                    reason,
                });
                self.fresh_binding_id()
            } else {
                self.declare_with_kind(name, span.clone(), false, BindingKind::Node)
            };
            BoundVariable {
                name: name.clone(),
                binding_id,
                may_be_null: false,
                span,
                type_info: None,
            }
        });
        // Label NAME flows through to the substrate — see the
        // `BoundCreateNodeSpec::label` rustdoc for the read-only-
        // catalog rationale.
        let label = spec.label.clone();
        // ADR-152-amendment-01 §D-1 — None-tolerant match-side label
        // resolution. `lookup_label` is a pure, side-effect-free read
        // (`CatalogProvider` contract: "Returns `None` for unknown
        // names") — we do NOT push `BindingError::UnknownLabel` (that
        // is MATCH's erroring site, `bind_node_pattern`); MERGE may
        // legitimately name a label that no node carries yet, in which
        // case the lowering emits a provably-empty match-branch and the
        // create-branch mints the label. Resolved uniformly for every
        // create-node spec; only the MERGE lowering reads it (CREATE
        // mints by name at execute-time).
        let match_label_id = label
            .as_deref()
            .and_then(|name| self.catalog.lookup_label(name));
        let properties = spec.properties.as_ref().map(|m| self.bind_property_map(m));
        // Span: prefer the var span; fall back to the cursor's
        // current position.
        let span = if let Some(v) = &var {
            v.span.clone()
        } else if let Some(p_map) = &properties {
            p_map
                .entries
                .first()
                .map(|e| e.span.clone())
                .unwrap_or_else(|| Span::point(1, 1))
        } else {
            Span::point(1, 1)
        };
        BoundCreateNodeSpec {
            var,
            label,
            match_label_id,
            properties,
            endpoint_binding: CreateEndpointBinding::Fresh,
            span,
        }
    }

    fn bind_create_path_endpoint_spec(&mut self, spec: &CreateNodeSpec) -> BoundCreateNodeSpec {
        let label = spec.label.clone();
        let properties = spec.properties.as_ref().map(|m| self.bind_property_map(m));
        let match_label_id = label
            .as_deref()
            .and_then(|name| self.catalog.lookup_label(name));
        let is_bare = label.is_none() && properties.is_none();
        let (var, endpoint_binding) = match spec.var.as_ref() {
            Some(name) => {
                let span = self.span_for_token(name);
                if let Some(info) = self.lookup_binding_info(name) {
                    if is_bare && info.kind == BindingKind::Node {
                        let binding_id = info.binding_id;
                        let var = BoundVariable {
                            name: name.clone(),
                            binding_id,
                            may_be_null: info.may_be_null,
                            span,
                            type_info: None,
                        };
                        (Some(var), CreateEndpointBinding::RowBinding(binding_id))
                    } else if info.kind != BindingKind::Node {
                        self.errors.push(BindingError::VariableTypeConflict {
                            name: name.clone(),
                            new_kind: BindingKind::Node.as_str(),
                            prior_kind: info.kind.as_str(),
                            span: span.clone(),
                            prior_span: info.declared_at.clone(),
                        });
                        let var = BoundVariable {
                            name: name.clone(),
                            binding_id: self.fresh_binding_id(),
                            may_be_null: false,
                            span,
                            type_info: None,
                        };
                        (Some(var), CreateEndpointBinding::Fresh)
                    } else {
                        self.errors.push(BindingError::DuplicateBinding {
                            name: name.clone(),
                            span: span.clone(),
                            prior_span: info.declared_at.clone(),
                            reason: format!(
                                "; variable `{name}` already bound — cannot re-declare with labels/properties in CREATE"
                            ),
                        });
                        let var = BoundVariable {
                            name: name.clone(),
                            binding_id: self.fresh_binding_id(),
                            may_be_null: false,
                            span,
                            type_info: None,
                        };
                        (Some(var), CreateEndpointBinding::Fresh)
                    }
                } else {
                    let binding_id =
                        self.declare_with_kind(name, span.clone(), false, BindingKind::Node);
                    let var = BoundVariable {
                        name: name.clone(),
                        binding_id,
                        may_be_null: false,
                        span,
                        type_info: None,
                    };
                    (Some(var), CreateEndpointBinding::Fresh)
                }
            }
            None => (None, CreateEndpointBinding::Fresh),
        };
        let span = if let Some(v) = &var {
            v.span.clone()
        } else if let Some(p_map) = &properties {
            p_map
                .entries
                .first()
                .map(|e| e.span.clone())
                .unwrap_or_else(|| Span::point(1, 1))
        } else {
            Span::point(1, 1)
        };
        BoundCreateNodeSpec {
            var,
            label,
            match_label_id,
            properties,
            endpoint_binding,
            span,
        }
    }

    // ---------- DELETE (ADR-149 W26-θ Phase 3) ----------

    // Bind a `DELETE` clause body. Each `DeleteItem` RESOLVES against
    // the current scope chain (parallel to RETURN's identifier-
    // projection resolution); type-check enforces that the resolved
    // binding is Node-typed or Relationship-typed (ADR-149 §D-4).
    //
    // An unresolved item surfaces `BindingError::UndeclaredVariable`
    // via `self.resolve`. We still emit a `BoundDeleteItem` with a
    // FRESH (declarative) BoundVariable for that name so the bound
    // tree remains structurally complete — the type-check pass will
    // see the already-accumulated binding error and short-circuit
    // before emitting type-related diagnostics. This mirrors the
    // RETURN-side handling where an unresolved identifier produces
    // `BoundExpression::UnresolvedVariable` rather than a hole.
    fn bind_delete_clause(&mut self, d: &DeleteClause) -> BoundDeleteClause {
        // Advance the cursor past `DETACH` first (if present), then
        // `DELETE`, so subsequent identifier spans land at the right
        // source positions.
        if d.detach {
            let _ = self.span_for_token("DETACH");
        }
        let span = self.span_for_token("DELETE");
        let items = d
            .items
            .iter()
            .map(|item| self.bind_delete_item(item))
            .collect();
        BoundDeleteClause {
            items,
            detach: d.detach,
            span,
        }
    }

    fn bind_delete_item(&mut self, item: &DeleteItem) -> BoundDeleteItem {
        let span = self.span_for_token(&item.var);
        // Resolve the variable against the current scope chain.
        // `resolve` emits `BindingError::UndeclaredVariable` on miss;
        // we still construct a `BoundVariable` with a synthesized
        // binding-id (`u64::MAX` is reserved as an unresolved-marker)
        // so downstream walkers don't need a separate hole-shape.
        let binding_id = self
            .resolve(&item.var, span.clone())
            .unwrap_or(BindingId::new(u64::MAX));
        // `may_be_null`: look up the original declaration's flag.
        // For an unresolved variable the lookup returns None; default
        // to false (the type-check pass will see the binding error
        // and surface the canonical diagnostic).
        let may_be_null = self.lookup_may_be_null(&item.var).unwrap_or(false);
        let var = BoundVariable {
            name: item.var.clone(),
            binding_id,
            may_be_null,
            span: span.clone(),
            type_info: None,
        };
        BoundDeleteItem { var, span }
    }

    // ---------- SET / REMOVE (ADR-150 W26-θ Phase 4) ----------

    // Bind a `SET` clause body. Each `SetItem` RESOLVES its `var`
    // against the current scope chain (parallel to DELETE-side
    // resolution per ADR-149 §D-3). Each item's mutation is bound:
    // PropertyAssign + PropertyReplace + PropertyMerge bind the
    // value expressions / maps; LabelAdd passes through.
    //
    // An unresolved `var` surfaces `BindingError::UndeclaredVariable`
    // via `self.resolve`. We still emit a `BoundSetItem` with a
    // FRESH (declarative) BoundVariable so the bound tree remains
    // structurally complete — the type-check pass observes the
    // already-accumulated binding error and short-circuits before
    // emitting type-related diagnostics.
    fn bind_set_clause(&mut self, s: &SetClause) -> BoundSetClause {
        let span = self.span_for_token("SET");
        let items = s
            .items
            .iter()
            .map(|item| self.bind_set_item(item))
            .collect();
        BoundSetClause { items, span }
    }

    fn bind_set_item(&mut self, item: &SetItem) -> BoundSetItem {
        let span = self.span_for_token(&item.var);
        let binding_id = self
            .resolve(&item.var, span.clone())
            .unwrap_or(BindingId::new(u64::MAX));
        let may_be_null = self.lookup_may_be_null(&item.var).unwrap_or(false);
        let var = BoundVariable {
            name: item.var.clone(),
            binding_id,
            may_be_null,
            span: span.clone(),
            type_info: None,
        };
        let mutation = match &item.mutation {
            SetMutation::PropertyAssign { name, value } => BoundSetMutation::PropertyAssign {
                name: name.clone(),
                value: self.bind_expression(value),
            },
            SetMutation::PropertyReplace(map) => {
                BoundSetMutation::PropertyReplace(self.bind_property_map(map))
            }
            SetMutation::PropertyMerge(map) => {
                BoundSetMutation::PropertyMerge(self.bind_property_map(map))
            }
            SetMutation::LabelAdd(labels) => BoundSetMutation::LabelAdd(labels.clone()),
        };
        BoundSetItem {
            var,
            mutation,
            span,
        }
    }

    // Bind a `REMOVE` clause body. Each `RemoveItem` RESOLVES its
    // `var` against the current scope chain. Each item's removal
    // mutation passes through verbatim (no sub-expressions to bind).
    fn bind_remove_clause(&mut self, r: &RemoveClause) -> BoundRemoveClause {
        let span = self.span_for_token("REMOVE");
        let items = r
            .items
            .iter()
            .map(|item| self.bind_remove_item(item))
            .collect();
        BoundRemoveClause { items, span }
    }

    fn bind_remove_item(&mut self, item: &RemoveItem) -> BoundRemoveItem {
        let span = self.span_for_token(&item.var);
        let binding_id = self
            .resolve(&item.var, span.clone())
            .unwrap_or(BindingId::new(u64::MAX));
        let may_be_null = self.lookup_may_be_null(&item.var).unwrap_or(false);
        let var = BoundVariable {
            name: item.var.clone(),
            binding_id,
            may_be_null,
            span: span.clone(),
            type_info: None,
        };
        let mutation = match &item.mutation {
            RemoveMutation::Property(name) => BoundRemoveMutation::Property(name.clone()),
            RemoveMutation::LabelRemove(labels) => BoundRemoveMutation::LabelRemove(labels.clone()),
        };
        BoundRemoveItem {
            var,
            mutation,
            span,
        }
    }

    // ---------- MERGE (ADR-151 W26-θ Phase 5) ----------

    // Bind a `MERGE` clause body per ADR-151 §D-3:
    //
    // 1. Bind the merge pattern (Node or Path) by DECLARING fresh
    //    bindings for each variable — reuses Phase 1 / Phase 2 CREATE-
    //    side binding helpers (`bind_create_node_spec` /
    //    `bind_create_path_spec`). Re-using an existing variable from
    //    a prior MATCH is illegal at Phase 5 (same `DuplicateBinding`
    //    error surface as CREATE; MATCH→MERGE cross-clause binding
    //    flow forward-pinned to v1.1 per ADR-151 §"Forward-deferred").
    // 2. Bind the optional `on_create` / `on_match` action items by
    //    RESOLVING each item's `var` against the now-extended scope
    //    chain (the merge pattern's freshly-declared variables are in
    //    scope). Each action's mutation is bound via the SAME helper
    //    Phase 4 SET items use (`bind_set_item`).
    fn bind_merge_clause(&mut self, m: &MergeClause) -> BoundMergeClause {
        let span = self.span_for_token("MERGE");
        let pattern = match &m.pattern {
            MergePattern::Node(spec) => BoundMergePattern::Node(self.bind_create_node_spec(spec)),
            MergePattern::Path(path) => BoundMergePattern::Path(self.bind_create_path_spec(path)),
        };
        let on_create: Vec<BoundSetItem> = m
            .on_create
            .iter()
            .map(|item| self.bind_set_item(item))
            .collect();
        let on_match: Vec<BoundSetItem> = m
            .on_match
            .iter()
            .map(|item| self.bind_set_item(item))
            .collect();
        BoundMergeClause {
            pattern,
            on_create,
            on_match,
            span,
        }
    }

    // ---------- MATCH ----------

    fn bind_match_clause(&mut self, m: &MatchClause, is_optional: bool) -> BoundMatchClause {
        // OPTIONAL MATCH cursor advance: skip past the `OPTIONAL`
        // keyword first so the subsequent `MATCH` token finds the
        // right occurrence in the source. This keeps span
        // computation in source order for the rest of the clause.
        if is_optional {
            let _ = self.span_for_token("OPTIONAL");
        }
        let span = self.span_for_token("MATCH");
        let scope = self.current_scope_id();

        let body = match &m.body {
            MatchBody::Patterns(ps) => {
                let bound: Vec<BoundPathPattern> = ps
                    .iter()
                    .map(|p| self.bind_path_pattern(p, is_optional))
                    .collect();
                BoundMatchBody::Patterns(bound)
            }
            MatchBody::NamedPath(np) => {
                BoundMatchBody::NamedPath(self.bind_named_path(np, is_optional))
            }
        };

        let where_clause = m.where_clause.as_ref().map(|e| self.bind_expression(e));
        // #618 — aggregation is forbidden in WHERE (openCypher v9 §6.4 /
        // TCK `MatchWhere1.feature` [15] `InvalidAggregation`). Pushes
        // `InvalidAggregation` so it pre-empts the executor's
        // aggregation-NotImplemented (which would otherwise surface as
        // WrongErrorPhase).
        if let (Some(ast), Some(bound)) = (m.where_clause.as_ref(), where_clause.as_ref()) {
            self.check_no_aggregate(ast, "WHERE", bound.span().clone());
        }

        BoundMatchClause {
            body,
            where_clause,
            scope,
            span,
            // ADR-006 amendment-01: `is_optional` carries the OPTIONAL
            // MATCH discriminant. M4-22's `TypeCheckVisitor`
            // propagates `may_be_null = true` to every variable
            // declared in this clause when `is_optional == true`.
            is_optional,
        }
    }

    fn bind_named_path(&mut self, np: &NamedPath, in_optional: bool) -> BoundNamedPath {
        // The named-path's outer var (`p` in `MATCH p = (..)-[..]->(..)`)
        // is itself a fresh declaration (named-path vars don't admit
        // re-reference at v1.0 — `MATCH p = (...) MATCH p = (...)`
        // would be a duplicate path binding, not a re-reference). Use
        // pure `declare`. nullability follows the enclosing clause.
        let span = self.span_for_token(&np.var);
        let binding_id =
            self.declare_with_kind(&np.var, span.clone(), in_optional, BindingKind::Path);
        let var = BoundVariable {
            name: np.var.clone(),
            binding_id,
            may_be_null: in_optional,
            span,
            type_info: None,
        };
        let kind = match &np.kind {
            NamedPathKind::ShortestPath(p) => {
                BoundNamedPathKind::ShortestPath(self.bind_path_pattern(p, in_optional))
            }
            NamedPathKind::AllShortestPath(p) => {
                BoundNamedPathKind::AllShortestPath(self.bind_path_pattern(p, in_optional))
            }
            NamedPathKind::Plain(p) => {
                BoundNamedPathKind::Plain(self.bind_path_pattern(p, in_optional))
            }
        };
        BoundNamedPath { var, kind }
    }

    fn bind_path_pattern(&mut self, p: &PathPattern, in_optional: bool) -> BoundPathPattern {
        let head = self.bind_node_pattern(&p.head, in_optional);
        let tail = p
            .tail
            .iter()
            .map(|(rel, node)| {
                let rel_b = self.bind_rel_pattern(rel, in_optional);
                let node_b = self.bind_node_pattern(node, in_optional);
                (rel_b, node_b)
            })
            .collect();
        BoundPathPattern { head, tail }
    }

    fn bind_node_pattern(&mut self, p: &NodePattern, in_optional: bool) -> BoundNodePattern {
        // Pattern-position variable binding: declare-or-resolve. A
        // re-reference of an existing name shares the binding_id and
        // INHERITS may_be_null from the original (never upgrades).
        // See ADR-038 §2 D-21 M4-22b refinement (Shape B).
        let var = p.var.as_ref().map(|name| {
            let span = self.span_for_token(name);
            let (binding_id, may_be_null) = self.declare_or_resolve_in_pattern(
                name,
                span.clone(),
                in_optional,
                BindingKind::Node,
            );
            BoundVariable {
                name: name.clone(),
                binding_id,
                may_be_null,
                span,
                type_info: None,
            }
        });

        let labels = p
            .labels
            .iter()
            .map(|name| {
                let span = self.span_for_token(name);
                match self.catalog.lookup_label(name) {
                    Some(label_id) => BoundLabelRef {
                        name: name.clone(),
                        label_id,
                        span,
                    },
                    None => {
                        // ADR-038 amendment-12 (#796): permissive dynamic-schema
                        // binding. An unknown label is NOT an error — it binds to
                        // the [`UNRESOLVED_LABEL`] sentinel (`LabelId::MAX`), which
                        // no node carries, so the pattern matches NOTHING (empty)
                        // per openCypher "unknown label ⇒ empty match". (Previously
                        // raised `UnknownLabel` → `-32005`, breaking cold-start /
                        // empty-db reads — #796.)
                        BoundLabelRef {
                            name: name.clone(),
                            label_id: UNRESOLVED_LABEL,
                            span,
                        }
                    }
                }
            })
            .collect();

        let properties = p.properties.as_ref().map(|m| self.bind_property_map(m));
        // Compute a span covering the node pattern (best-effort —
        // we use the var/label span if available, else cursor pos).
        let span = if let Some(v) = &var {
            v.span.clone()
        } else if let Some(p_map) = &properties {
            p_map
                .entries
                .first()
                .map(|e| e.span.clone())
                .unwrap_or_else(|| Span::point(1, 1))
        } else {
            Span::point(1, 1)
        };
        BoundNodePattern {
            var,
            labels,
            properties,
            span,
        }
    }

    fn bind_rel_pattern(&mut self, r: &RelPattern, in_optional: bool) -> BoundRelPattern {
        // Pattern-position variable binding: declare-or-resolve.
        // See `bind_node_pattern` for rationale.
        let var = r.var.as_ref().map(|name| {
            let span = self.span_for_token(name);
            let (binding_id, may_be_null) = self.declare_or_resolve_in_pattern(
                name,
                span.clone(),
                in_optional,
                BindingKind::Rel,
            );
            BoundVariable {
                name: name.clone(),
                binding_id,
                may_be_null,
                span,
                type_info: None,
            }
        });

        let rel_types = r
            .rel_types
            .iter()
            .map(|name| {
                let span = self.span_for_token(name);
                match self.catalog.lookup_rel_type(name) {
                    Some(type_id) => BoundRelTypeRef {
                        name: name.clone(),
                        type_id,
                        span,
                    },
                    None => {
                        // ADR-038 amendment-12 (#796): permissive dynamic-schema
                        // binding — an unknown rel-type binds to the
                        // [`UNRESOLVED_REL_TYPE`] sentinel (`TypeId::MAX`), which no
                        // relationship carries, so the expand matches NOTHING
                        // (empty) per openCypher "unknown rel-type ⇒ empty match".
                        BoundRelTypeRef {
                            name: name.clone(),
                            type_id: UNRESOLVED_REL_TYPE,
                            span,
                        }
                    }
                }
            })
            .collect();

        let properties = r.properties.as_ref().map(|m| self.bind_property_map(m));

        // `LengthRange::Quantified` (GQL `{N,M}`) is reserved at
        // v1.0 (D-9). Pass through here; M4-22's `TypeCheckVisitor`
        // emits `ArcQLError::NotImplemented`.
        let span = var
            .as_ref()
            .map(|v| v.span.clone())
            .unwrap_or_else(|| Span::point(1, 1));
        BoundRelPattern {
            var,
            rel_types,
            direction: r.direction.clone(),
            length: r.length.clone(),
            properties,
            span,
        }
    }

    fn bind_property_map(&mut self, m: &PropertyMap) -> BoundPropertyMap {
        let entries = m
            .entries
            .iter()
            .map(|(key, value)| {
                let span = self.span_for_token(key);
                let property_id = self.catalog.lookup_property(key);
                let value = self.bind_expression(value);
                BoundPropertyEntry {
                    key: key.clone(),
                    property_id,
                    value,
                    span,
                }
            })
            .collect();
        BoundPropertyMap { entries }
    }

    // ---------- WITH / UNWIND ----------

    fn bind_with_clause(&mut self, w: &WithClause) -> BoundWithClause {
        let span = self.span_for_token("WITH");

        // ADR-038 amendment-12 (#796 companion) — openCypher v9 §6.4
        // implicit-grouping-key validation (`AmbiguousAggregationExpression`).
        self.check_aggregation_grouping(&w.items, &span);
        // #1053 — record the implicit GROUP BY keys for the tail `ORDER BY`'s
        // aggregate-in-sort-key validation (`Some` iff this WITH aggregates).
        self.set_tail_grouping_context(&w.items);

        // Resolve all RHS expressions in the CURRENT scope, BEFORE
        // we close it. Collect the (alias-or-derived-name, span,
        // nullable) triples that will become the new scope's
        // bindings. Nullability rule (Shape B):
        // - Bare identifier passthrough (`WITH n` or `WITH n AS x`):
        //   inherit `may_be_null` from the pre-WITH binding so
        //   downstream re-references see the correct nullability.
        // - Other projections (arithmetic, function calls, literals,
        //   property access): conservatively false at v1.0. Transitive
        //   nullability through arbitrary expressions is out of scope
        //   for M4-22b — the executor's NULL semantics handle the
        //   3VL evaluation at runtime.
        let mut bound_items = Vec::with_capacity(w.items.len());
        // (name, decl_span, nullable, item_index). The trailing
        // `item_index` lets us back-patch the post-WITH `declare()`d id
        // onto the corresponding projection item's `output_id` (#746),
        // so the column the executor's `ProjectOp` emits carries the
        // SAME id a downstream `resolve()` returns.
        // (name, decl_span, nullable, kind, item_index). `kind` (#618):
        // a bare/aliased passthrough of an identifier INHERITS the
        // source binding's kind (so `WITH n` keeps `n` a node and a
        // later `MATCH (n)` re-references it); every other projection
        // (literal / arithmetic / function-call / property access /
        // list) is a [`BindingKind::Value`], so a later pattern position
        // (`MATCH (n)` after `WITH 1 AS n`) is a `VariableTypeConflict`.
        let mut new_bindings: Vec<(String, Span, bool, BindingKind, usize)> = Vec::new();
        // #802 / ADR-197: snapshot the in-scope bindings a `WITH *`
        // carries through, BEFORE the old scope is popped (with their
        // ORIGINAL binding ids — see `current_in_scope_named`).
        let wildcard_carries: Vec<(String, BindingInfo)> = if w
            .items
            .iter()
            .any(|i| matches!(i.kind, ProjectionKind::Wildcard))
        {
            self.current_in_scope_named()
        } else {
            Vec::new()
        };
        for (idx, item) in w.items.iter().enumerate() {
            let bound_item = self.bind_projection_item(item);
            // #618 — `NoExpressionAlias` (openCypher v9 §6 / TCK
            // `With4.feature` [5]): a WITH projection that is NOT a bare
            // variable reference MUST be aliased. `WITH *` (wildcard) and
            // `WITH n` (bare identifier) are exempt; `WITH count(*)` /
            // `WITH a.b` / `WITH 1 + 2` with no `AS` reject at bind time.
            if item.alias.is_none()
                && !matches!(
                    &item.kind,
                    ProjectionKind::Wildcard | ProjectionKind::Expr(Expression::Identifier(_))
                )
            {
                self.errors.push(BindingError::NoExpressionAlias {
                    span: bound_item.span.clone(),
                });
            }
            let visible: Option<(String, bool, BindingKind)> = match (&item.alias, &item.kind) {
                (Some(a), ProjectionKind::Expr(Expression::Identifier(n))) => {
                    // Aliased passthrough: `WITH n AS x`. Inherit
                    // nullability + kind of `n`.
                    let nullable = self.lookup_may_be_null(n).unwrap_or(false);
                    let kind = self.lookup_kind(n).unwrap_or(BindingKind::Value);
                    Some((a.clone(), nullable, kind))
                }
                (Some(a), ProjectionKind::Expr(expr)) if is_static_null_literal(expr) => {
                    // `WITH null AS a`: the value is the STATICALLY-NULL
                    // literal. Tagged `NullValue` (nullable) so a later
                    // OPTIONAL node anchor `(a)` is well-typed (null unifies
                    // with NODE) — see `declare_or_resolve_in_pattern`. Any
                    // other expression stays `Value` (and a non-null value
                    // anchor still conflicts). TCK Path1[1]/Path2[3].
                    Some((a.clone(), true, BindingKind::NullValue))
                }
                (Some(a), _) => {
                    // Aliased non-passthrough projection: derived
                    // value, conservatively non-nullable + Value-kind.
                    Some((a.clone(), false, BindingKind::Value))
                }
                (None, ProjectionKind::Expr(Expression::Identifier(n))) => {
                    // Bare passthrough: `WITH n`. Inherit nullability +
                    // kind.
                    let nullable = self.lookup_may_be_null(n).unwrap_or(false);
                    let kind = self.lookup_kind(n).unwrap_or(BindingKind::Value);
                    Some((n.clone(), nullable, kind))
                }
                _ => None,
            };
            if let Some((name, nullable, kind)) = visible {
                let alias_span = bound_item.span.clone();
                new_bindings.push((name, alias_span, nullable, kind, idx));
            }
            bound_items.push(bound_item);
        }

        // Push the projected-output scope above the still-live input
        // scope. WITH is still a pipeline fence after this clause, but
        // openCypher lets WITH-WHERE predicates see both the projected
        // outputs and the pre-projection inputs; nearest-first lookup
        // gives projected outputs the required shadowing behavior.
        let input_scope_idx = self.scope_chain.len().saturating_sub(1);
        let new_scope = self.push_scope(span.clone());
        let mut declared_names: std::collections::HashSet<String> =
            std::collections::HashSet::new();
        for (name, decl_span, nullable, kind, idx) in new_bindings {
            declared_names.insert(name.clone());
            let declared = self.declare_with_kind(&name, decl_span, nullable, kind);
            // #746: the projected column carries the post-WITH binding
            // id so downstream references (a following RETURN / MATCH /
            // UNWIND) resolve to the SAME id the `ProjectOp` emits.
            bound_items[idx].output_id = Some(declared);
        }
        // #802 / ADR-197: carry the `WITH *` passthrough bindings into
        // the new scope PRESERVING their original binding ids (insert the
        // BindingInfo verbatim, NOT a fresh declare) so the Filter above
        // the wildcard ProjectOp resolves to the same id the Project's
        // `extend_from_slice(child_schema)` emits. Skip names an explicit
        // item already declared (`WITH *, x AS x`).
        for (name, info) in wildcard_carries {
            if !declared_names.insert(name.clone()) {
                continue;
            }
            if let Some(scope) = self.scope_chain.last_mut() {
                scope.bindings.entry(name).or_insert(info);
            }
        }
        // #746: every `Expr` projection item MUST carry an output id
        // (the binder↔ProjectOp contract). A non-visible `Expr` item —
        // e.g. an unaliased non-passthrough projection, which
        // openCypher rejects but which we still bind for a complete
        // tree — gets a fresh id even though nothing downstream
        // references it.
        for item in &mut bound_items {
            if matches!(item.kind, BoundProjectionKind::Expr(_)) && item.output_id.is_none() {
                item.output_id = Some(self.fresh_binding_id());
            }
        }

        // WITH-WHERE resolves against projection OUTPUTS ∪ pre-WITH
        // INPUTS, with OUTPUTS shadowing. This preserves #773's HAVING
        // fix (`WITH a, count(*) AS relCount WHERE relCount > 1` resolves
        // `relCount` to the output-only aggregate alias) and restores the
        // openCypher dropped-input case (`WITH c WHERE r IS NULL` resolves
        // `r` through the input fallback). After binding this predicate we
        // remove the pre-WITH frame so the next clause sees only the WITH
        // outputs; lowering decides whether the filter must run below the
        // projection to keep dropped-input ids in the row schema.
        let where_clause = w.where_clause.as_ref().map(|e| self.bind_expression(e));
        if input_scope_idx < self.scope_chain.len().saturating_sub(1) {
            // #1053 — DEFER the pre-WITH input-frame removal for an
            // AGGREGATING `WITH`. openCypher v9 §6.6 lets the tail `ORDER BY`
            // immediately following an aggregating `WITH` carry an aggregate
            // (`ORDER BY me.age + count(you.age)`) whose argument references a
            // pre-projection variable (`you`); that aggregate is computed at
            // the `WITH` boundary, where `you` is still live. Keeping the
            // frame until the next NON-tail clause (flushed by
            // `flush_pending_with_fence`) lets the tail resolve the argument
            // WITHOUT widening the fence: the input frame is removed before
            // any subsequent reading clause, so a following MATCH / WITH /
            // RETURN still sees ONLY the projected outputs. A NON-aggregating
            // `WITH` keeps the original immediate removal (no aggregate can
            // appear in its tail `ORDER BY` — `ReturnOrderBy2`/`WithOrderBy2`
            // reject an aggregate over a non-aggregating projection), so the
            // dropped-input error (`WITH a AS x ORDER BY a` → UndeclaredVariable)
            // is preserved.
            let with_is_aggregating = w
                .items
                .iter()
                .any(|i| matches!(&i.kind, ProjectionKind::Expr(e) if expr_contains_aggregate(e)));
            if with_is_aggregating {
                self.pending_with_fence = Some(input_scope_idx);
            } else {
                self.scope_chain.remove(input_scope_idx);
            }
        }

        BoundWithClause {
            // #842 part B — thread DISTINCT through to the BoundAST so the
            // lowering can compose `LogicalDistinct` over the projection.
            distinct: w.distinct,
            items: bound_items,
            where_clause,
            scope: new_scope,
            span,
        }
    }

    fn bind_unwind_clause(&mut self, u: &UnwindClause) -> BoundUnwindClause {
        // UNWIND's element variable is not a pattern position per
        // ADR-038 §2 D-21 — it's a fresh declaration (re-declaration
        // emits `DuplicateBinding`). Element nullability follows the
        // list expression's nullability; v1.0 conservatively keeps it
        // non-nullable here (the type-check pass derives the precise
        // element type from the source list).
        let span = self.span_for_token("UNWIND");
        let expr = self.bind_expression(&u.expr);
        let var_span = self.span_for_token(&u.var);
        let binding_id = self.declare(&u.var, var_span.clone(), false);
        let var = BoundVariable {
            name: u.var.clone(),
            binding_id,
            may_be_null: false,
            span: var_span,
            type_info: None,
        };
        BoundUnwindClause { expr, var, span }
    }

    // ---------- CALL <proc>(args) [YIELD …] (ADR-197 / #802) ----------

    /// Bind a `CALL <proc>(args) [YIELD col [AS alias], …]` schema-
    /// introspection procedure call (ADR-197, #802).
    ///
    /// 1. Resolve the dotted name to a [`ProcedureKind`]
    ///    ([`BindingError::UnknownProcedure`] on miss).
    /// 2. Bind the argument expressions (accepted but not interpreted
    ///    at v1.0-α — sampling is a no-op).
    /// 3. For each YIELD item, validate the column against the
    ///    procedure's fixed output set
    ///    ([`BindingError::InvalidYieldColumn`] on miss) + DECLARE the
    ///    aliased variable into scope (like UNWIND's output binding) so
    ///    the following WHERE / RETURN can reference it. An empty YIELD
    ///    declares nothing (standalone `CALL proc()` — its result IS
    ///    the query result; the executor yields all output columns).
    fn bind_call_procedure_clause(
        &mut self,
        c: &crate::ast::CallProcedureClause,
    ) -> BoundCallProcedureClause {
        let span = self.span_for_token("CALL");
        let kind = match ProcedureKind::from_name(&c.name) {
            Some(k) => k,
            None => {
                self.errors.push(BindingError::UnknownProcedure {
                    name: c.name.clone(),
                    span: span.clone(),
                });
                // Recover with a benign procedure so binding continues;
                // bind returns Err on the pushed error regardless.
                ProcedureKind::DbLabels
            }
        };
        let args: Vec<BoundExpression> = c.args.iter().map(|a| self.bind_expression(a)).collect();
        let valid_cols = kind.output_columns();
        // Standalone `CALL proc()` with NO YIELD yields ALL output
        // columns as the result (Neo4j semantic); synthesize an
        // all-columns YIELD list so the columns are declared + flow to
        // the result. With an explicit YIELD, use exactly those.
        let effective_items: Vec<(String, Option<String>)> = if c.yield_items.is_empty() {
            valid_cols
                .iter()
                .map(|c| ((*c).to_string(), None))
                .collect()
        } else {
            c.yield_items.clone()
        };
        let mut yields: Vec<BoundProcedureYield> = Vec::with_capacity(effective_items.len());
        for (column, alias) in &effective_items {
            if !valid_cols.iter().any(|vc| vc == column) {
                self.errors.push(BindingError::InvalidYieldColumn {
                    proc: c.name.clone(),
                    column: column.clone(),
                    span: span.clone(),
                });
                // Skip declaring an invalid column (bind already Err).
                continue;
            }
            let binding_name = alias.clone().unwrap_or_else(|| column.clone());
            let var_span = self.span_for_token(&binding_name);
            let binding_id = self.declare(&binding_name, var_span.clone(), true);
            yields.push(BoundProcedureYield {
                column: column.clone(),
                var: BoundVariable {
                    name: binding_name,
                    binding_id,
                    may_be_null: true,
                    span: var_span,
                    type_info: None,
                },
            });
        }
        // Bind the optional WHERE AFTER the yields are declared so the
        // predicate can reference the YIELD'd columns
        // (`… YIELD type … WHERE NOT type = 'RELATIONSHIP'`).
        let where_clause = c.where_clause.as_ref().map(|e| self.bind_expression(e));
        BoundCallProcedureClause {
            kind,
            args,
            yields,
            where_clause,
            span,
        }
    }

    /// Bind a `SHOW CONSTRAINTS | INDEXES | DATABASES` clause (ADR-197,
    /// #802). Declares the fixed output columns into scope so a
    /// following RETURN / WHERE can reference them. v1.0-α surfaces a
    /// minimal column set (langchain consumes `SHOW CONSTRAINTS` via
    /// `r.data()` over whatever columns are returned).
    fn bind_show_clause(&mut self, s: &crate::ast::ShowClause) -> BoundShowClause {
        use crate::ast::ShowKind;
        let span = self.span_for_token("SHOW");
        // Valid output columns per SHOW kind (Neo4j's SHOW output has
        // many columns; we surface the subset common vector clients
        // YIELD). `SHOW VECTOR INDEXES` (#830)
        // adds `options` (the real vector path yields `name,
        // labelsOrTypes, properties, options`).
        let valid_cols: &[&str] = match s.kind {
            ShowKind::Constraints => &["name", "type", "entityType", "labelsOrTypes", "properties"],
            ShowKind::Indexes | ShowKind::VectorIndexes => &[
                "name",
                "type",
                "entityType",
                "labelsOrTypes",
                "properties",
                "options",
            ],
            ShowKind::Databases => &["name", "address", "role", "currentStatus"],
        };
        // No YIELD → the full column set is the result (bare `SHOW`
        // behaviour, preserved). Explicit `YIELD a [AS b], …` (#830) →
        // exactly those, aliased. Mirrors `bind_call_procedure_clause`.
        let kind_label = match s.kind {
            ShowKind::Constraints => "SHOW CONSTRAINTS",
            ShowKind::Indexes => "SHOW INDEXES",
            ShowKind::VectorIndexes => "SHOW VECTOR INDEXES",
            ShowKind::Databases => "SHOW DATABASES",
        };
        let effective_items: Vec<(String, Option<String>)> = if s.yield_items.is_empty() {
            valid_cols
                .iter()
                .map(|c| ((*c).to_string(), None))
                .collect()
        } else {
            s.yield_items.clone()
        };
        let mut columns: Vec<BoundVariable> = Vec::with_capacity(effective_items.len());
        for (column, alias) in &effective_items {
            if !valid_cols.iter().any(|vc| vc == column) {
                self.errors.push(BindingError::InvalidYieldColumn {
                    proc: kind_label.to_string(),
                    column: column.clone(),
                    span: span.clone(),
                });
                // Skip declaring an invalid column (bind already Err).
                continue;
            }
            let binding_name = alias.clone().unwrap_or_else(|| column.clone());
            let var_span = self.span_for_token(&binding_name);
            let binding_id = self.declare(&binding_name, var_span.clone(), true);
            columns.push(BoundVariable {
                name: binding_name,
                binding_id,
                may_be_null: true,
                span: var_span,
                type_info: None,
            });
        }
        // Bind the optional WHERE AFTER the columns are declared so the
        // predicate can reference the YIELD'd columns (#830 —
        // `… YIELD name … WHERE name = $index_name`).
        let where_clause = s.where_clause.as_ref().map(|e| self.bind_expression(e));
        BoundShowClause {
            kind: s.kind,
            columns,
            where_clause,
            span,
        }
    }

    // ---------- CALL { <subquery> } (ADR-192 / #623) ----------

    /// Bind a `CALL { <subquery> }` correlated brace-subquery (ADR-192,
    /// #623 — Cypher 25, beyond openCypher v9).
    ///
    /// The four load-bearing semantics:
    /// - **D-3 implicit import.** The body binds in a CHILD scope pushed
    ///   ATOP the current chain, so [`Self::resolve`] (nearest-first)
    ///   finds the outer in-scope variables WITHOUT a mandatory
    ///   importing-`WITH`. `imported` records the outer in-scope binding
    ///   set at the `CALL` point (drives the executor's per-driving-row
    ///   correlation seed).
    /// - **D-4 scoping fence.** Variables declared INSIDE the body do
    ///   NOT escape — we truncate the scope chain back to the pre-body
    ///   depth after binding, dropping the body's frames. Only the
    ///   body's terminal-RETURN columns escape: we RE-DECLARE them in the
    ///   outer scope so the enclosing query can reference them.
    /// - **D-4 RETURN-collision.** A body RETURN column name that
    ///   collides with an outer in-scope name is a bind error — the
    ///   outer-scope [`Self::declare`] emits
    ///   [`BindingError::DuplicateBinding`] (reused per the ADR's
    ///   R1-verified decision).
    /// - **D-9 read-only fence.** A write clause inside the body is
    ///   rejected with
    ///   [`BindingError::WriteInCallSubqueryNotSupported`].
    fn bind_call_clause(&mut self, c: &CallClause) -> BoundCallClause {
        let span = self.span_for_token("CALL");

        // D-9: read-only fence. A write clause anywhere in the immediate
        // body (or any union arm) is rejected. Nested `CALL { … }`
        // bodies are scanned when each nested clause is bound (recursion
        // through `bind_call_clause`), so a single-level scan here +
        // recursion covers the whole tree.
        if call_body_has_write_clause(&c.body) {
            self.errors
                .push(BindingError::WriteInCallSubqueryNotSupported { span: span.clone() });
            // Accumulate-all-errors convention: keep binding so other
            // diagnostics in the body still surface.
        }

        // D-3: snapshot the OUTER in-scope binding set BEFORE pushing the
        // child scope. At v1.0-α `imported` is the FULL outer in-scope
        // set (a superset of the strictly-referenced set); seeding
        // unreferenced bindings is correctness-neutral (the body reads
        // only what it references; the RETURN projects the rest away).
        // Precise referenced-only subsetting is a forward optimization
        // tied to D-10 uncorrelated-subquery caching (OQ-192-3).
        let imported = self.in_scope_binding_ids();

        // Bind the body in a CHILD scope. We do NOT call
        // `bind_read_query` / `bind_union_query` (their defensive cleanup
        // pops the ENTIRE chain, destroying the outer scopes). Record the
        // pre-body depth + truncate back to it (the D-4 fence).
        let base_depth = self.scope_chain.len();
        let bound_body = self.bind_call_body(&c.body);
        while self.scope_chain.len() > base_depth {
            self.pop_scope();
        }

        // D-4: the body's terminal-RETURN columns escape the fence —
        // RE-DECLARE them in the (now-current) OUTER scope so the
        // enclosing query can reference them. A name colliding with an
        // outer in-scope variable surfaces `DuplicateBinding` here.
        let returned_names = call_body_returned_names(&c.body);
        let mut returned = Vec::with_capacity(returned_names.len());
        for name in &returned_names {
            let id = self.declare(name, span.clone(), false);
            returned.push(id);
        }

        BoundCallClause {
            body: Box::new(bound_body),
            imported,
            returned,
            span,
        }
    }

    /// Bind a `CALL { … }` body (a `Statement::Read` or
    /// `Statement::Union` — the only two shapes the grammar admits).
    /// The body's scopes are left on the chain; [`Self::bind_call_clause`]
    /// truncates them after (the D-4 fence). For a UNION body each arm
    /// binds independently (no cross-arm visibility) but EACH sees the
    /// outer imports.
    fn bind_call_body(&mut self, body: &Statement) -> BoundStatement {
        match body {
            Statement::Read(q) => BoundStatement::Read(self.bind_call_read_arm(q)),
            Statement::Union(u) => BoundStatement::Union(Box::new(self.bind_call_union(u))),
            // The `call_clause` grammar admits ONLY `union_query` /
            // `read_query`, so the parser never produces another
            // `Statement` variant here. Bind defensively as an empty read
            // query so downstream passes see a well-formed node (this arm
            // is unreachable in practice).
            other => {
                debug_assert!(
                    false,
                    "CALL body must be Read/Union (grammar-enforced), got {other:?}"
                );
                BoundStatement::Read(self.bind_call_read_arm(&ReadQuery {
                    clauses: Vec::new(),
                }))
            }
        }
    }

    /// Bind one `CALL { … }` body arm (a read query) in a fresh child
    /// scope pushed atop the outer chain. Mirrors [`Self::bind_read_query`]
    /// WITHOUT the defensive whole-chain cleanup (the caller manages the
    /// chain depth so the outer scopes survive — implicit import).
    fn bind_call_read_arm(&mut self, q: &ReadQuery) -> BoundQuery {
        let query_span = self.whole_source_span();
        let root_scope = self.push_scope(query_span.clone());
        let mut clauses = Vec::with_capacity(q.clauses.len());
        for c in &q.clauses {
            clauses.push(self.bind_clause(c));
        }
        BoundQuery {
            clauses,
            root_scope,
            span: query_span,
            tenant: self.catalog.tenant(),
            partition: self.catalog.partition(),
            snapshot_lsn: None,
        }
    }

    /// Bind a UNION body inside `CALL { … }`. Mirrors
    /// [`Self::bind_union_query`] (per-arm independent scoping + the §8
    /// no-mixing / column-compat checks) but each arm's child scope sits
    /// ATOP the outer chain (so every arm sees the outer imports) and is
    /// truncated back to the pre-body depth between arms (no cross-arm
    /// visibility). The caller truncates any residual after the last arm.
    fn bind_call_union(&mut self, u: &UnionQuery) -> BoundUnionQuery {
        let union_span = self.whole_source_span();
        let mut arms: Vec<BoundQuery> = Vec::with_capacity(u.arms.len());
        let mut tail = BoundUnionTail::default();
        let pre_depth = self.scope_chain.len();
        for (idx, arm) in u.arms.iter().enumerate() {
            let bq = self.bind_call_read_arm(arm);
            if idx == 0 {
                tail = self.bind_union_tail(&u.tail);
            }
            // Arms are independent: pop this arm's frame(s) back to the
            // pre-body depth before the next arm (each arm re-sees the
            // outer imports, never the sibling arm's bindings).
            while self.scope_chain.len() > pre_depth {
                self.pop_scope();
            }
            arms.push(bq);
        }
        self.check_union_no_mixing(u, &union_span);
        let column_orders = self.check_union_column_compat(u, &union_span);
        BoundUnionQuery {
            arms,
            all: u.all.clone(),
            column_orders,
            tail,
            span: union_span,
        }
    }

    /// All binding-ids currently in scope (the union across every frame
    /// in the chain). The seed-import set for a `CALL { … }` body
    /// (ADR-192 D-3). Sorted for determinism. In this scope model a
    /// `WITH` POPS the prior frame (see [`Self::bind_with_clause`]), so
    /// the chain holds exactly the live in-scope bindings — dropped
    /// variables are not present.
    fn in_scope_binding_ids(&self) -> Vec<BindingId> {
        let mut ids: std::collections::BTreeSet<BindingId> = std::collections::BTreeSet::new();
        for scope in &self.scope_chain {
            for info in scope.bindings.values() {
                ids.insert(info.binding_id);
            }
        }
        ids.into_iter().collect()
    }

    /// **#802 / ADR-197 fix.** Enumerate every CURRENTLY in-scope
    /// binding as `(name, BindingInfo)`, nearest-scope-wins for shadowed
    /// names, sorted by name. Used to carry a `WITH *` wildcard's
    /// passthrough bindings into the post-projection scope WITH THEIR
    /// ORIGINAL binding ids preserved — so a following `WHERE` / clause
    /// resolves them to the SAME id the ProjectOp's wildcard passthrough
    /// emits (the ProjectOp `Wildcard` arm `extend_from_slice(child_schema)`
    /// carries the CHILD ids verbatim; re-`declare`ing fresh ids would
    /// desync the Filter's binding ref from the row schema → the
    /// "BindingId missing from row schema" runtime fault). Closes the
    /// latent `WITH * WHERE …` bind bug langchain's `… YIELD … UNWIND …
    /// WITH * WHERE …` REL_QUERY surfaced.
    fn current_in_scope_named(&self) -> Vec<(String, BindingInfo)> {
        let mut by_name: std::collections::BTreeMap<String, BindingInfo> =
            std::collections::BTreeMap::new();
        for scope in &self.scope_chain {
            for (name, info) in &scope.bindings {
                by_name.insert(name.clone(), info.clone());
            }
        }
        by_name.into_iter().collect()
    }

    fn bind_projection_item(&mut self, p: &ProjectionItem) -> BoundProjectionItem {
        match &p.kind {
            ProjectionKind::Wildcard => {
                let span = self.span_for_token("*");
                // openCypher v9 §6.1 — `RETURN *` / `WITH *` returns
                // every in-scope variable in ALPHABETICAL order by name.
                // `current_in_scope_named()` is a BTreeMap iteration, so
                // it is already name-sorted; project the ids in that order
                // so the wildcard passthrough emits columns alphabetically
                // (not in pipeline-declaration order). Anonymous pattern
                // bindings carry no name (they are never `declare`d into
                // the scope map — see `declare_or_resolve_in_pattern`), so
                // they are correctly EXCLUDED from `*` (Cypher 9 §6.1).
                let order: Vec<BindingId> = self
                    .current_in_scope_named()
                    .into_iter()
                    .map(|(_, info)| info.binding_id)
                    .collect();
                BoundProjectionItem {
                    kind: BoundProjectionKind::Wildcard { order },
                    alias: p.alias.clone(),
                    // `*` passes the child schema through; no fresh
                    // output id (#746).
                    output_id: None,
                    // `*` has no single source expression (#353).
                    source_text: None,
                    span,
                }
            }
            ProjectionKind::Expr(e) => {
                let bound = self.bind_expression(e);
                let span = bound.span().clone();
                // If aliased, advance cursor past the alias name so
                // the subsequent projection's variable refs (and a
                // following WITH's new-binding span) point at the
                // right occurrence.
                if let Some(a) = &p.alias {
                    let _ = self.span_for_token(a);
                }
                BoundProjectionItem {
                    kind: BoundProjectionKind::Expr(bound),
                    alias: p.alias.clone(),
                    // The output binding-id is stamped by the caller
                    // (`bind_with_clause` back-patches the post-WITH
                    // `declare()`d id; `bind_return_clause` mints a
                    // fresh id) so it AGREES with what downstream
                    // references resolve to (#746).
                    output_id: None,
                    // #353 — thread the parser-captured implicit
                    // column-name source text through verbatim. Used by
                    // `BoundProjectionItem::display_name` only when there
                    // is no explicit alias.
                    source_text: p.source_text.clone(),
                    span,
                }
            }
        }
    }

    // ---------- RANK BY / WITH FUSION ----------

    fn bind_rank_by_clause(&mut self, r: &RankByClause) -> BoundRankByClause {
        let span = self.span_for_token("RANK");
        let ranker = match &r.ranker {
            Ranker::Hybrid(args) => {
                let bound: Vec<BoundRankArg> = args.iter().map(|a| self.bind_rank_arg(a)).collect();
                BoundRanker::Hybrid(bound)
            }
        };
        let score = r.score_alias.as_ref().map(|name| {
            let score_span = self.span_for_token(name);
            let binding_id = self.declare(name, score_span.clone(), false);
            BoundVariable {
                name: name.clone(),
                binding_id,
                may_be_null: false,
                span: score_span,
                type_info: None,
            }
        });
        BoundRankByClause {
            ranker,
            score,
            span,
        }
    }

    fn bind_rank_arg(&mut self, a: &RankArg) -> BoundRankArg {
        match a {
            RankArg::Vector { field, query, k } => {
                let field = self.bind_field_ref(field);
                let span = field.span.clone();
                let query = self.bind_expression(query);
                BoundRankArg::Vector {
                    field,
                    query,
                    k: *k,
                    span,
                }
            }
            RankArg::Text { field, query, k } => {
                let field = self.bind_field_ref(field);
                let span = field.span.clone();
                let query = self.bind_expression(query);
                BoundRankArg::Text {
                    field,
                    query,
                    k: *k,
                    span,
                }
            }
        }
    }

    fn bind_field_ref(&mut self, f: &FieldRef) -> BoundFieldRef {
        let base_span = self.span_for_token(&f.base);
        let binding_id = self.resolve(&f.base, base_span.clone()).unwrap_or_else(|| {
            // Resolution emitted an UndeclaredVariable; we still
            // fabricate a binding_id sentinel for downstream tree
            // shape. M4-22's type-checker will see the BoundAst is
            // structurally complete; the error-vec is the source of
            // truth.
            BindingId::new(u64::MAX)
        });
        let base = BoundVariable {
            name: f.base.clone(),
            binding_id,
            may_be_null: false,
            span: base_span.clone(),
            type_info: None,
        };
        let path = f
            .path
            .iter()
            .map(|name| {
                let span = self.span_for_token(name);
                let property_id = self.catalog.lookup_property(name);
                BoundPropertyRef {
                    name: name.clone(),
                    property_id,
                    span,
                }
            })
            .collect();
        BoundFieldRef {
            base,
            path,
            span: base_span,
        }
    }

    fn bind_with_fusion_clause(&mut self, c: &WithFusionClause) -> BoundWithFusionClause {
        let span = self.span_for_token("FUSION");
        let fusion = match &c.fusion {
            Fusion::Rrf { k } => BoundFusion::Rrf { k: *k },
        };
        BoundWithFusionClause { fusion, span }
    }

    // ---------- RETURN ----------

    fn bind_return_clause(&mut self, r: &ReturnClause) -> BoundReturnClause {
        let span = self.span_for_token("RETURN");

        // ADR-038 amendment-12 (#796 companion) — openCypher v9 §6.4
        // implicit-grouping-key validation (`AmbiguousAggregationExpression`).
        self.check_aggregation_grouping(&r.items, &span);
        // #1053 — record the implicit GROUP BY keys for the tail `ORDER BY`'s
        // aggregate-in-sort-key validation (`Some` iff this RETURN aggregates).
        self.set_tail_grouping_context(&r.items);

        // #836 — start a fresh projected-expression map for THIS RETURN's
        // tail `ORDER BY` (self-contained; `bind_clause` also clears on
        // the non-tail RETURN arm, but a directly-constructed clause must
        // not inherit a stale map).
        self.return_tail_outputs.clear();

        // #618 — `NoVariablesInScope` (openCypher v9 §6 / TCK
        // `Return7.feature` [2]): `RETURN *` requires at least one
        // in-scope variable to expand. `MATCH () RETURN *` (the
        // anonymous node binds nothing) has an empty scope → reject.
        // A `RETURN *` with in-scope variables, or a `*` alongside
        // explicit columns, is fine.
        if r.items
            .iter()
            .any(|i| matches!(i.kind, ProjectionKind::Wildcard))
            && self.in_scope_binding_ids().is_empty()
        {
            self.errors
                .push(BindingError::NoVariablesInScope { span: span.clone() });
        }

        // #618 — `ColumnNameConflict` (openCypher v9 §6 / TCK
        // `Return4.feature` [10]): two result columns with the same
        // name (`RETURN 1 AS a, 2 AS a`). The column name is the
        // explicit alias, else the canonical expression rendering (the
        // same derivation `terminal_return_column_names` uses for UNION
        // column-compat); `*` is exempt (it expands at runtime).
        let mut seen_columns: std::collections::HashSet<String> = std::collections::HashSet::new();
        for i in &r.items {
            // `safe_expr_display` (not `format!("{e}")`) — the
            // `Display for Expression` impl returns `Err` for a
            // non-finite float literal (`1.34E999` → `inf`), and
            // `format!` PANICS on that; the safe variant yields `None`
            // instead. A non-finite literal is independently rejected by
            // the `FloatingPointOverflow` check in `bind_expression`, so
            // skipping it here only forgoes the (irrelevant) dup-name
            // comparison for an already-invalid query. #618.
            let col = match (&i.alias, &i.kind) {
                (Some(alias), _) => Some(alias.clone()),
                (None, ProjectionKind::Expr(e)) => safe_expr_display(e),
                (None, ProjectionKind::Wildcard) => None,
            };
            if let Some(col) = col {
                if !seen_columns.insert(col.clone()) {
                    self.errors.push(BindingError::ColumnNameConflict {
                        name: col,
                        span: span.clone(),
                    });
                }
            }
        }

        // Bind each projection item, minting its fresh output binding-id
        // (#746 binder↔`ProjectOp` contract — the executor's `ProjectOp`
        // emits the column under this id rather than an executor-local
        // synthetic, and a `Project` layered over an `Aggregate`
        // references the SAME id; see the aggregate-output wiring in
        // lowering). `Wildcard` keeps `output_id = None` (it passes the
        // child schema through). While binding, collect the NAME by which
        // an ORDER BY can reference each output COLUMN — an alias
        // (`expr AS x` → `x`) or an unaliased passthrough variable
        // (`RETURN ints` → `ints`) — paired with its `output_id`; this
        // drives the ORDER-BY-over-projection scope built below (#618).
        let mut items: Vec<BoundProjectionItem> = Vec::with_capacity(r.items.len());
        let mut output_bindings: Vec<(String, Span, bool, BindingId)> = Vec::new();
        for i in &r.items {
            let mut item = self.bind_projection_item(i);
            if matches!(item.kind, BoundProjectionKind::Expr(_)) {
                let output_id = self.fresh_binding_id();
                item.output_id = Some(output_id);
                if let Some((name, nullable)) = self.return_output_name(i) {
                    output_bindings.push((name, item.span.clone(), nullable, output_id));
                }
                // #836 — register the projected expression so a tail
                // `ORDER BY` over the SAME expression (aliased or not)
                // resolves to THIS column's output id. Covers the
                // unaliased-expression gap the output-NAME scope above
                // cannot (an unaliased `RETURN p.name` has no output
                // name). `return_output_name` (name scope) and this
                // (expression map) are complementary: the name scope wins
                // for an `ORDER BY <alias>`; the expression map wins for an
                // `ORDER BY <projected expr>`.
                if let ProjectionKind::Expr(ast_expr) = &i.kind {
                    let display = item
                        .alias
                        .clone()
                        .or_else(|| item.source_text.clone())
                        .unwrap_or_default();
                    self.return_tail_outputs
                        .push((ast_expr.clone(), output_id, display));
                }
            }
            items.push(item);
        }

        // openCypher: ORDER BY / SKIP / LIMIT are evaluated AFTER the
        // projection and resolve against the projection's OUTPUT columns
        // — a returned alias or passthrough variable is orderable by that
        // output column (clauses/return-orderby, with-orderBy). Lowering
        // places `Sort` OVER `Project`, so the Sort key's binding-id MUST
        // be the Project output column's id (`output_id`), NOT the
        // pre-projection source id — else the key is "missing from row
        // schema" at runtime (`UNWIND [1,3,2] AS ints RETURN ints ORDER
        // BY ints` was the founding failure, #618).
        //
        // Mirror `bind_with_clause`'s #746 back-patch: push a scope that
        // maps each RETURN output NAME → its `output_id` ATOP the current
        // (pre-projection) scope, then bind ORDER BY against it. Unlike
        // WITH (a pipeline fence that POPS the prior frame), RETURN keeps
        // the pre-projection scope underneath, so an ORDER BY reference to
        // an in-scope-but-not-returned expression still resolves
        // (nearest-first: outputs shadow, the pre-projection scope is the
        // fall-back — the deferred non-aggregating "order by a
        // non-projected in-scope var" sub-case stays #618 follow-up).
        // RETURN is terminal, so the standalone `Clause::TailOrderBy` /
        // `TailSkip` / `TailLimit` the parser emits AFTER this RETURN bind
        // in this SAME scope; `bind_read_query`'s defensive cleanup (and
        // the CALL D-4 depth truncation) pops it at clause-list end.
        self.push_scope(span.clone());
        if let Some(scope) = self.scope_chain.last_mut() {
            for (name, decl_span, nullable, output_id) in &output_bindings {
                // First-wins on duplicate output names (`RETURN a, a` or
                // duplicate aliases): insert WITHOUT a `DuplicateBinding`
                // diagnostic — this transient scope only steers ORDER-BY
                // resolution; the RETURN projection list itself defines
                // column identity.
                scope.bindings.entry(name.clone()).or_insert(BindingInfo {
                    binding_id: *output_id,
                    declared_at: decl_span.clone(),
                    may_be_null: *nullable,
                    // RETURN is terminal — this transient scope only
                    // steers ORDER-BY resolution and never feeds a
                    // pattern position, so the kind is immaterial here;
                    // `Value` is the safe default (#618).
                    kind: BindingKind::Value,
                });
            }
        }

        // The embedded `r.order_by` path is dormant for PARSED queries
        // (the parser emits ORDER BY / SKIP / LIMIT as standalone
        // `Clause::Tail*`), but binding it here in the projection-output
        // scope keeps a directly-constructed `ReturnClause` correct.
        let order_by: Vec<BoundOrderItem> =
            r.order_by.iter().map(|o| self.bind_order_item(o)).collect();
        // #618 — SKIP/LIMIT constant-ness validation on the embedded
        // RETURN tail (dormant for parsed queries — the parser emits
        // standalone `Clause::Tail*` — but keeps a directly-constructed
        // `ReturnClause` correct, mirroring the standalone-clause check).
        if let Some(e) = r.skip.as_ref() {
            self.check_skip_limit_expr(e, "SKIP", span.clone());
        }
        if let Some(e) = r.limit.as_ref() {
            self.check_skip_limit_expr(e, "LIMIT", span.clone());
        }
        let skip = r.skip.as_ref().map(|e| self.bind_expression(e));
        let limit = r.limit.as_ref().map(|e| self.bind_expression(e));
        BoundReturnClause {
            distinct: r.distinct,
            items,
            order_by,
            skip,
            limit,
            span,
        }
    }

    /// The name an ORDER BY can use to reference a RETURN projection
    /// item's OUTPUT column: the alias if present (`expr AS x` → `x`),
    /// else the passthrough variable of a bare-identifier projection
    /// (`RETURN ints` → `ints`). Returns the name plus its nullability (a
    /// passthrough inherits the source binding's `may_be_null`; an
    /// aliased non-passthrough projection is conservatively non-nullable —
    /// mirrors `bind_with_clause`'s Shape-B rule). An unaliased
    /// NON-identifier expression (`RETURN a + b`) yields `None` — ordering
    /// by its rendered column name is a deferred #618 sub-case. MUST be
    /// called in the PRE-projection scope (before the output scope is
    /// pushed) so the passthrough nullability lookup sees the source.
    fn return_output_name(&self, item: &ProjectionItem) -> Option<(String, bool)> {
        match (&item.alias, &item.kind) {
            (Some(a), ProjectionKind::Expr(Expression::Identifier(n))) => {
                Some((a.clone(), self.lookup_may_be_null(n).unwrap_or(false)))
            }
            (Some(a), _) => Some((a.clone(), false)),
            (None, ProjectionKind::Expr(Expression::Identifier(n))) => {
                Some((n.clone(), self.lookup_may_be_null(n).unwrap_or(false)))
            }
            _ => None,
        }
    }

    fn bind_order_item(&mut self, o: &OrderItem) -> BoundOrderItem {
        // Bind the key in the current scope FIRST — this advances the
        // span cursor identically whether or not the #836 rewrite fires,
        // so downstream token spans stay correct. The bound result is
        // discarded only when the key matches a projected expression.
        let expr = self.bind_expression(&o.expr);
        let span = expr.span().clone();
        // #618 / #1053 — aggregation in ORDER BY is CONDITIONALLY allowed
        // (openCypher v9 §6.6). Run BEFORE the #836 rewrite so the sort key's
        // shape is examined as-written.
        //
        // - Sort key WITHOUT an aggregate → unchanged (no constraint here).
        // - Sort key WITH an aggregate over a NON-aggregating projection
        //   (no `tail_grouping_context`) → `InvalidAggregation`: the
        //   aggregate has no group to fold over and would silently collapse a
        //   non-aggregating result (`ReturnOrderBy2` [14] `RETURN n.num1 ORDER
        //   BY max(n.num2)`; `WithOrderBy2` [25] `WITH n.num1 AS foo ORDER BY
        //   count(1)`).
        // - Sort key WITH an aggregate over an AGGREGATING projection
        //   (`tail_grouping_context = Some(keys)`) → the aggregate is computed
        //   alongside the projection's aggregates; every NON-aggregated leaf
        //   in the sort key must itself be a grouping key — the SAME rule
        //   `check_aggregation_grouping` enforces on a projection. A
        //   non-grouping leaf ⇒ `AmbiguousAggregationExpression` (a
        //   compile-time error; `ReturnOrderBy6` [4]/[5], `WithOrderBy4`
        //   [19]/[20]). All-grouping leaves ⇒ ACCEPT (`ReturnOrderBy6` [2]/[3],
        //   `WithOrderBy4` [17]/[18]); the lowering lifts the inline aggregate
        //   into the projection's `Aggregate` node and points the `Sort` at
        //   the computed column.
        if expr_contains_aggregate(&o.expr) {
            match self.tail_grouping_context.clone() {
                None => {
                    self.errors.push(BindingError::InvalidAggregation {
                        position: "ORDER BY",
                        span: span.clone(),
                    });
                }
                Some(grouping_keys) => {
                    let keys: Vec<&Expression> = grouping_keys.iter().collect();
                    if agg_has_nongrouping_ref(&o.expr, &keys) {
                        self.errors
                            .push(BindingError::AmbiguousAggregationExpression {
                                span: span.clone(),
                            });
                    }
                }
            }
        }
        // #836 — RETURN-clause ORDER BY over a PROJECTED EXPRESSION. If
        // the (non-aggregate) sort key is structurally the SAME COMPOUND
        // expression a RETURN item projected, resolve it to that projected
        // column's `output_id` — the id the `ProjectOp` emits and the
        // `Sort` (which lowering places OVER the `Project`) sees. Otherwise
        // the key references the PRE-projection source binding, which
        // `Project` drops → "binding … missing from row schema" at runtime
        // (openCypher v9 §6.6: the sort reads the projection OUTPUT).
        //
        // A BARE IDENTIFIER key (`RETURN n ORDER BY n`, `RETURN x AS y
        // ORDER BY y`) is deliberately EXCLUDED: it is already resolved
        // correctly by the output-NAME scope (#618) that
        // `bind_return_clause` pushes — `bind_expression` above returns a
        // VariableRef to the right output id for it — so re-routing it
        // through the map would only risk a cosmetic name drift. The map
        // therefore handles EXACTLY the gap the name scope cannot: an
        // unaliased / aliased COMPOUND expression (`p.name`, `a + b`).
        // Aggregates are also excluded — their ORDER BY handling is
        // unchanged (and rejected above).
        let expr = match self.match_return_tail_output(&o.expr) {
            Some((output_id, name))
                if !expr_contains_aggregate(&o.expr)
                    && !matches!(o.expr, Expression::Identifier(_)) =>
            {
                BoundExpression::VariableRef {
                    name,
                    binding_id: output_id,
                    span: span.clone(),
                    type_info: None,
                }
            }
            _ => expr,
        };
        BoundOrderItem {
            expr,
            direction: o.direction.clone(),
            span,
        }
    }

    /// #836 — if `order_expr` (an AST `ORDER BY` key) is structurally
    /// equal to a RETURN projection expression registered by the
    /// preceding [`Self::bind_return_clause`], return that projected
    /// column's `output_id` + display name. openCypher v9 §6.6: a
    /// RETURN-clause `ORDER BY` resolves against the projection OUTPUT, so
    /// ordering by the SAME expression that was projected targets the
    /// projected column. AST [`Expression`] equality is span-free (see
    /// `impl PartialEq for ProjectionItem`), so this is a structural match
    /// robust to span / source-slice differences. First match wins on a
    /// duplicate projected expression (`RETURN p.x, p.x` — both carry the
    /// same value, so either column is a correct sort key). Returns `None`
    /// for a union tail / WITH ORDER BY (the map is cleared for those), so
    /// only the single-query RETURN tail is affected.
    fn match_return_tail_output(&self, order_expr: &Expression) -> Option<(BindingId, String)> {
        self.return_tail_outputs
            .iter()
            .find(|(expr, _, _)| expr == order_expr)
            .map(|(_, id, name)| (*id, name.clone()))
    }

    /// Push a [`BindingError::InvalidAggregation`] if `e` contains an
    /// aggregating function (openCypher v9 §6.4 — aggregation is confined
    /// to RETURN/WITH projection terms). `position` names the illegal
    /// position for the diagnostic. #618 GA Lane BINDER-VALIDATIONS.
    fn check_no_aggregate(&mut self, e: &Expression, position: &'static str, span: Span) {
        if expr_contains_aggregate(e) {
            self.errors
                .push(BindingError::InvalidAggregation { position, span });
        }
    }

    /// openCypher v9 §6.4 — implicit-grouping-key validation for a RETURN /
    /// WITH projection list (`AmbiguousAggregationExpression`; ADR-038
    /// amendment-12, the #796 permissive-binding companion). The
    /// NON-aggregating projection expressions form the implicit grouping key.
    /// Within an AGGREGATING projection, every variable/property reference
    /// OUTSIDE an aggregate-function argument must itself be a grouping key —
    /// a COMPLEX grouping key (`a + b`) does NOT make its leaves grouping
    /// keys (it must be aliased and the alias referenced). Passes TCK
    /// `Return6` [18]/[19] + `With6` [6]/[7] (simple grouping-key reference);
    /// rejects [20]/[21] + [8]/[9] (no key / complex key recomputed inside an
    /// aggregating expression). One error per violating projection.
    /// #1053 — record the implicit GROUP BY keys of an AGGREGATING projection
    /// into [`Self::tail_grouping_context`] so the immediately-following tail
    /// `ORDER BY` can validate an inline aggregate against them (openCypher v9
    /// §6.6). Sets `None` when the projection is NOT aggregating — then ANY
    /// aggregate in the tail `ORDER BY` is rejected (`ReturnOrderBy2` [14] /
    /// `WithOrderBy2` [25]). Called by `bind_return_clause` / `bind_with_clause`
    /// AFTER `check_aggregation_grouping`. The keys are the non-aggregating
    /// projection expressions, cloned into owned [`Expression`]s (so the
    /// context outlives the borrow of `items`), PLUS the ALIAS identifier of
    /// each non-aggregating projection (`me.age AS age` contributes both
    /// `me.age` and `age`): the tail `ORDER BY` sees the projection OUTPUT, so
    /// `ORDER BY age + count(...)` references the grouping key by its alias —
    /// the alias is as valid a grouping reference as the underlying expression
    /// (`ReturnOrderBy6` [2], `WithOrderBy4` [17]).
    fn set_tail_grouping_context(&mut self, items: &[ProjectionItem]) {
        let any_aggregating = items
            .iter()
            .any(|i| matches!(&i.kind, ProjectionKind::Expr(e) if expr_contains_aggregate(e)));
        self.tail_grouping_context = if any_aggregating {
            let mut keys: Vec<Expression> = Vec::new();
            for i in items {
                if let ProjectionKind::Expr(e) = &i.kind {
                    if !expr_contains_aggregate(e) {
                        // The grouping-key expression itself (`me.age`).
                        keys.push(e.clone());
                        // Its alias as a bare identifier (`age`), so an
                        // ORDER BY referencing the projected OUTPUT column by
                        // alias counts as a grouping reference.
                        if let Some(alias) = &i.alias {
                            keys.push(Expression::Identifier(alias.clone()));
                        }
                    }
                }
            }
            Some(keys)
        } else {
            None
        };
    }

    fn check_aggregation_grouping(&mut self, items: &[ProjectionItem], span: &Span) {
        let grouping_keys: Vec<&Expression> = items
            .iter()
            .filter_map(|i| match &i.kind {
                ProjectionKind::Expr(e) if !expr_contains_aggregate(e) => Some(e),
                _ => None,
            })
            .collect();
        let any_aggregating = items
            .iter()
            .any(|i| matches!(&i.kind, ProjectionKind::Expr(e) if expr_contains_aggregate(e)));
        if !any_aggregating {
            // No aggregation in this projection list ⇒ no implicit grouping ⇒
            // no constraint.
            return;
        }
        for i in items {
            if let ProjectionKind::Expr(e) = &i.kind {
                if expr_contains_aggregate(e) && agg_has_nongrouping_ref(e, &grouping_keys) {
                    self.errors
                        .push(BindingError::AmbiguousAggregationExpression { span: span.clone() });
                }
            }
        }
    }

    /// Validate a `SKIP` / `LIMIT` expression at COMPILE (bind) time per
    /// openCypher v9 §6.4 (`clause` = `"SKIP"` / `"LIMIT"`):
    /// - a negative integer literal → [`BindingError::NegativeIntegerArgument`]
    ///   (`ReturnSkipLimit1` [11] / `ReturnSkipLimit2` [12]);
    /// - a non-integer constant (float / string / bool / list / map) →
    ///   [`BindingError::NonIntegerSkipLimit`] (`ReturnSkipLimit2` [16]
    ///   `LIMIT 1.7`);
    /// - an expression that references a bound variable →
    ///   [`BindingError::NonConstantExpression`] (`ReturnSkipLimit1`
    ///   [5]/[10] / `ReturnSkipLimit2` [9] `SKIP n.count`).
    ///
    /// A non-negative integer LITERAL and a PARAMETER are valid (a
    /// parameter is a query-constant; its non-negativity is a RUNTIME
    /// check per the TCK's "raised at runtime" negative-parameter
    /// scenarios, which are out of this compile-time slice's scope). The
    /// check runs on the AST expression BEFORE lowering, so it pre-empts
    /// the executor's `SKIP`/dynamic-`LIMIT` `NotImplemented` (which
    /// would otherwise surface the invalid forms as WrongErrorPhase).
    /// #618 GA Lane BINDER-VALIDATIONS.
    fn check_skip_limit_expr(&mut self, e: &Expression, clause: &'static str, span: Span) {
        match e {
            Expression::Literal(Literal::Integer(n)) => {
                if *n < 0 {
                    self.errors.push(BindingError::NegativeIntegerArgument {
                        clause,
                        value: *n,
                        span,
                    });
                }
            }
            Expression::Literal(Literal::Float(_)) => {
                self.errors.push(BindingError::NonIntegerSkipLimit {
                    clause,
                    actual: "float",
                    span,
                });
            }
            Expression::Literal(Literal::Bool(_)) => {
                self.errors.push(BindingError::NonIntegerSkipLimit {
                    clause,
                    actual: "boolean",
                    span,
                });
            }
            Expression::Literal(Literal::String(_)) => {
                self.errors.push(BindingError::NonIntegerSkipLimit {
                    clause,
                    actual: "string",
                    span,
                });
            }
            Expression::Literal(Literal::List(_)) => {
                self.errors.push(BindingError::NonIntegerSkipLimit {
                    clause,
                    actual: "list",
                    span,
                });
            }
            Expression::Literal(Literal::Map(_)) => {
                self.errors.push(BindingError::NonIntegerSkipLimit {
                    clause,
                    actual: "map",
                    span,
                });
            }
            // `SKIP -1` parses as a unary negation of an integer literal
            // (NOT a `Literal::Integer(-1)`), so the negative-value check
            // lives here: a negated integer literal is a negative count
            // (`ReturnSkipLimit1` [11] / `ReturnSkipLimit2` [12]); a
            // negated float literal is non-integer (degenerate
            // `LIMIT -1.7`). Unary-plus of an integer (`+1`) is a valid
            // non-negative constant.
            Expression::UnaryOp {
                op: UnaryOp::Neg,
                operand,
            } => match operand.as_ref() {
                Expression::Literal(Literal::Integer(n)) => {
                    self.errors.push(BindingError::NegativeIntegerArgument {
                        clause,
                        value: -*n,
                        span,
                    });
                }
                Expression::Literal(Literal::Float(_)) => {
                    self.errors.push(BindingError::NonIntegerSkipLimit {
                        clause,
                        actual: "float",
                        span,
                    });
                }
                // Negation of a variable-bearing expression is
                // non-constant; otherwise leave to runtime.
                other => {
                    if expr_references_variable(other) {
                        self.errors
                            .push(BindingError::NonConstantExpression { clause, span });
                    }
                }
            },
            Expression::UnaryOp {
                op: UnaryOp::Pos,
                operand,
            } => {
                // `+<expr>` is just `<expr>` for count purposes — recurse.
                self.check_skip_limit_expr(operand, clause, span);
            }
            // A parameter is a query-constant (valid at compile time);
            // `NULL` and other literals are degenerate but not in the
            // TCK's compile-time set — leave them to runtime.
            Expression::Parameter(_) | Expression::Literal(_) => {}
            // Any other expression shape: if it references a bound
            // variable it is non-constant; a pure-constant arithmetic
            // expression (`1 + 1`) is permitted (it folds to a constant).
            other => {
                if expr_references_variable(other) {
                    self.errors
                        .push(BindingError::NonConstantExpression { clause, span });
                }
            }
        }
    }

    // ---------- Expressions ----------

    fn bind_expression(&mut self, e: &Expression) -> BoundExpression {
        match e {
            Expression::Literal(l) => {
                // Literals — best-effort span: cursor at current
                // position. We don't try to find the literal's
                // exact source slice (numeric/string-literal
                // re-tokenization is fragile); the span is a
                // single-point at the cursor.
                let (line, col) = self.cursor.byte_to_line_col(self.cursor.cursor);
                let span = Span::point(line, col);
                // #618 — `FloatingPointOverflow` (openCypher v9 §3 / TCK
                // `Literals5.feature` [27]): a float literal that
                // overflowed to infinity at parse time (`1.34E999`) is an
                // out-of-range literal. A non-finite float cannot arise
                // from any other source in a parsed query (the grammar
                // has no `inf`/`NaN` bareword), so this fires ONLY on the
                // overflow case — a valid finite float is untouched.
                if let Literal::Float(x) = l {
                    if x.is_infinite() {
                        self.errors
                            .push(BindingError::FloatingPointOverflow { span: span.clone() });
                    }
                }
                match l {
                    Literal::List(items) => BoundExpression::ListLiteral {
                        elements: items
                            .iter()
                            .map(|item| self.bind_expression(item))
                            .collect(),
                        span,
                        type_info: None,
                    },
                    Literal::Map(entries) => BoundExpression::MapLiteral {
                        entries: entries
                            .iter()
                            .map(|(k, v)| (k.clone(), self.bind_expression(v)))
                            .collect(),
                        span,
                        type_info: None,
                    },
                    _ => BoundExpression::Literal {
                        value: l.clone(),
                        span,
                        type_info: None,
                    },
                }
            }
            Expression::Parameter(name) => {
                // Parameter identifier follows the `$`.
                let span = self.span_for_token(name);
                BoundExpression::Parameter {
                    name: name.clone(),
                    span,
                    type_info: None,
                }
            }
            Expression::Identifier(name) => {
                let span = self.span_for_token(name);
                match self.resolve(name, span.clone()) {
                    Some(binding_id) => BoundExpression::VariableRef {
                        name: name.clone(),
                        binding_id,
                        span,
                        type_info: None,
                    },
                    None => BoundExpression::UnresolvedVariable {
                        name: name.clone(),
                        span,
                    },
                }
            }
            Expression::PropertyAccess { base, path } => {
                let base = Box::new(self.bind_expression(base));
                let path: Vec<BoundPropertyRef> = path
                    .iter()
                    .map(|name| {
                        let span = self.span_for_token(name);
                        let property_id = self.catalog.lookup_property(name);
                        BoundPropertyRef {
                            name: name.clone(),
                            property_id,
                            span,
                        }
                    })
                    .collect();
                let span = path
                    .last()
                    .map(|p| p.span.clone())
                    .unwrap_or_else(|| base.span().clone());
                BoundExpression::PropertyAccess {
                    base,
                    path,
                    span,
                    type_info: None,
                }
            }
            Expression::BinaryOp { .. } | Expression::UnaryOp { .. } => self.bind_operator_spine(e),
            Expression::FunctionCall {
                name,
                args,
                distinct,
                star,
            } => {
                // M4-22 (D-22): function name + arity + arg-type
                // resolution lives in `type_check::TypeCheckVisitor`
                // against `crate::semantic::functions::FunctionRegistry`.
                // The `distinct` / `star` flags (#773 G4/G5) thread through
                // verbatim — type-check gates their validity (count(*) /
                // DISTINCT-on-aggregate) and the lowering lifts them into
                // the `AggregationSpec`.
                let span = self.span_for_token(name);
                // #618 — `NestedAggregation` (openCypher v9 §6.4 / TCK
                // `Return6.feature` [14] `count(count(*))`): an
                // aggregating function whose argument itself contains an
                // aggregate. Checked on the AST args so it pre-empts the
                // executor's aggregation-NotImplemented (WrongErrorPhase).
                if is_aggregate_fn(name) && args.iter().any(expr_contains_aggregate) {
                    self.errors
                        .push(BindingError::NestedAggregation { span: span.clone() });
                }
                let args: Vec<BoundExpression> =
                    args.iter().map(|a| self.bind_expression(a)).collect();
                BoundExpression::FunctionCall {
                    name: name.clone(),
                    args,
                    distinct: *distinct,
                    star: *star,
                    span,
                    type_info: None,
                }
            }
            Expression::Near {
                lhs,
                target,
                vector_index,
            } => {
                let lhs = Box::new(self.bind_expression(lhs));
                let target = Box::new(self.bind_expression(target));
                let span = lhs.span().clone();
                BoundExpression::Near {
                    lhs,
                    target,
                    vector_index: vector_index.clone(),
                    span,
                    type_info: None,
                }
            }
            Expression::TextMatch { lhs, query } => {
                let lhs = Box::new(self.bind_expression(lhs));
                let query = Box::new(self.bind_expression(query));
                let span = lhs.span().clone();
                BoundExpression::TextMatch {
                    lhs,
                    query,
                    span,
                    type_info: None,
                }
            }
            // TODO(M4-23): unify with community(...) function-call form.
            Expression::InCommunity { node, community } => {
                let node = Box::new(self.bind_expression(node));
                let community = Box::new(self.bind_expression(community));
                let span = node.span().clone();
                BoundExpression::InCommunity {
                    node,
                    community,
                    span,
                    type_info: None,
                }
            }
            Expression::In { .. } | Expression::IsNull { .. } => self.bind_operator_spine(e),
            // ADR-188 Decision 3 — list-predicate scoped binding. Bind
            // the list in the OUTER scope, then open a CHILD scope for
            // the iteration variable so it resolves ONLY inside the
            // predicate and is torn down immediately after (the
            // lexical-lifetime bug class is structurally impossible —
            // the scope is gone before any sibling expression binds).
            Expression::ListPredicate {
                quantifier,
                var,
                list,
                predicate,
            } => {
                let span = self.span_for_token(var);
                let bound_list = Box::new(self.bind_expression(list));
                // Child scope opens. The iteration variable may be NULL
                // (lists can contain NULL elements); we pass
                // `may_be_null = true` conservatively so a NULL element
                // 3VL-propagates correctly through the predicate.
                self.push_scope(span.clone());
                let var_span = self.span_for_token(var);
                let var_bid = self.declare(var, var_span, true);
                let bound_pred = Box::new(self.bind_expression(predicate));
                self.pop_scope(); // child scope closes; `var` gone.
                BoundExpression::ListPredicate {
                    quantifier: *quantifier,
                    var_bid,
                    list: bound_list,
                    predicate: bound_pred,
                    span,
                    type_info: None,
                }
            }
            // ADR-188 Decision 3 — reduce scoped binding. Bind `init` +
            // `list` in the outer scope; open ONE child scope declaring
            // both `acc` and `x` (LIFO with the fold body). Both resolve
            // ONLY inside the body. Reverse scope-walk gives the correct
            // inner-shadows-outer semantics for nested reduces.
            Expression::Reduce {
                acc_var,
                init,
                var,
                list,
                expr,
            } => {
                let span = self.span_for_token(acc_var);
                let bound_init = Box::new(self.bind_expression(init));
                let bound_list = Box::new(self.bind_expression(list));
                self.push_scope(span.clone());
                let acc_span = self.span_for_token(acc_var);
                // The accumulator's nullability follows `init` — a
                // non-null init keeps the fold non-null until the body
                // introduces one. We pass `false` (init-typed,
                // conservatively non-null); a NULL produced by the body
                // flows as an ordinary value (Decision 4 pure-fold) and
                // the type-check carries `Null`-propagation separately.
                let acc_bid = self.declare(acc_var, acc_span, false);
                let var_span = self.span_for_token(var);
                let var_bid = self.declare(var, var_span, true);
                let bound_expr = Box::new(self.bind_expression(expr));
                self.pop_scope();
                BoundExpression::Reduce {
                    acc_bid,
                    init: bound_init,
                    var_bid,
                    list: bound_list,
                    expr: bound_expr,
                    span,
                    type_info: None,
                }
            }
            // ADR-188 (#620 list-half) Decision 5 — list-comprehension
            // scoped binding. Bind `list` in the outer scope; open ONE
            // child scope declaring the iteration variable so it
            // resolves ONLY inside the WHERE `predicate` AND the
            // `| projection`, and is torn down immediately after (the
            // lexical-lifetime bug class is structurally impossible —
            // the scope is gone before any sibling expression binds).
            // Identical scoped-var lifetime to `ListPredicate`; the only
            // difference is that BOTH the (optional) predicate and the
            // (optional) projection bind inside the child scope.
            Expression::ListComprehension {
                var,
                list,
                predicate,
                projection,
            } => {
                let span = self.span_for_token(var);
                // #618 — aggregation is forbidden inside a list
                // comprehension's WHERE / `| projection` (openCypher v9
                // §6.4 / TCK `List12.feature` [7] `InvalidAggregation`).
                // The `list` source is also a non-aggregating position
                // here (a comprehension is an expression, not a
                // projection term).
                if predicate.as_deref().is_some_and(expr_contains_aggregate)
                    || projection.as_deref().is_some_and(expr_contains_aggregate)
                {
                    self.errors.push(BindingError::InvalidAggregation {
                        position: "list comprehension",
                        span: span.clone(),
                    });
                }
                let bound_list = Box::new(self.bind_expression(list));
                // Child scope opens. `may_be_null = true` — lists can
                // contain NULL elements, so a NULL element must
                // 3VL-propagate correctly through the WHERE filter (only
                // `true` keeps the element; null/false filter out).
                self.push_scope(span.clone());
                let var_span = self.span_for_token(var);
                let var_bid = self.declare(var, var_span, true);
                let bound_pred = predicate
                    .as_ref()
                    .map(|p| Box::new(self.bind_expression(p)));
                let bound_proj = projection
                    .as_ref()
                    .map(|e| Box::new(self.bind_expression(e)));
                self.pop_scope(); // child scope closes; `var` gone.
                BoundExpression::ListComprehension {
                    var_bid,
                    list: bound_list,
                    predicate: bound_pred,
                    projection: bound_proj,
                    span,
                    type_info: None,
                }
            }
            // ADR-191 D-6 (#620 map-half) — map projection `n{.k, alias: e,
            // .*}`. NO scoped variable (unlike the comprehensions): the
            // base is an OUTER row variable and the `alias: expr` value
            // expressions evaluate in the CURRENT scope. We bind the base by
            // resolving it as an ordinary variable reference (reusing the
            // `Identifier` resolution + `UnresolvedVariable` error path) and
            // bind each literal-entry value in the current scope; the `.key`
            // / `.*` selectors carry only property names (no sub-expression
            // to bind). The base binds FIRST (source order) so the cursor
            // advances past it before the item values bind.
            Expression::MapProjection { base, items } => {
                let span = self.span_for_token(base);
                let bound_base =
                    Box::new(self.bind_expression(&Expression::Identifier(base.clone())));
                let bound_items = items
                    .iter()
                    .map(|item| match item {
                        MapProjectionItem::Property(k) => {
                            BoundMapProjectionItem::Property(k.clone())
                        }
                        MapProjectionItem::AllProperties => BoundMapProjectionItem::AllProperties,
                        MapProjectionItem::Literal { alias, value } => {
                            BoundMapProjectionItem::Literal {
                                alias: alias.clone(),
                                value: Box::new(self.bind_expression(value)),
                            }
                        }
                    })
                    .collect();
                BoundExpression::MapProjection {
                    base: bound_base,
                    items: bound_items,
                    span,
                    type_info: None,
                }
            }
            // openCypher v9 §3.4 — postfix accessors. No scoped variable;
            // bind the base + index / bounds as ordinary sub-expressions.
            Expression::Subscript { base, index } => {
                let base = Box::new(self.bind_expression(base));
                let index = Box::new(self.bind_expression(index));
                let span = base.span().clone();
                BoundExpression::Subscript {
                    base,
                    index,
                    span,
                    type_info: None,
                }
            }
            Expression::Slice { base, start, end } => {
                let base = Box::new(self.bind_expression(base));
                let start = start.as_ref().map(|s| Box::new(self.bind_expression(s)));
                let end = end.as_ref().map(|e| Box::new(self.bind_expression(e)));
                let span = base.span().clone();
                BoundExpression::Slice {
                    base,
                    start,
                    end,
                    span,
                    type_info: None,
                }
            }
            // openCypher v9 §3.6 (#621) — CASE expression. NO scoped variable
            // (unlike the comprehensions): the test, every WHEN / THEN, and
            // the ELSE all bind in the CURRENT scope. We bind each
            // sub-expression as an ordinary child; the span is the test's
            // span (simple form) or the first WHEN's (searched form) —
            // `branches` is non-empty by the grammar `+` and the parser
            // arity guard.
            Expression::Case {
                test,
                branches,
                default,
            } => {
                let bound_test = test.as_ref().map(|t| Box::new(self.bind_expression(t)));
                let bound_branches: Vec<(BoundExpression, BoundExpression)> = branches
                    .iter()
                    .map(|(when, then)| (self.bind_expression(when), self.bind_expression(then)))
                    .collect();
                let bound_default = default.as_ref().map(|d| Box::new(self.bind_expression(d)));
                let span = bound_test
                    .as_ref()
                    .map(|t| t.span().clone())
                    .unwrap_or_else(|| bound_branches[0].0.span().clone());
                BoundExpression::Case {
                    test: bound_test,
                    branches: bound_branches,
                    default: bound_default,
                    span,
                    type_info: None,
                }
            }
        }
    }

    /// #1290 — bind a left-nested OPERATOR SPINE iteratively.
    ///
    /// Flat operator chains (`a AND b AND …`, `1 + 2 + …`, and the
    /// keyword postfix forms `x IN l IN …` / `x IS NULL IS NULL …`)
    /// fold into a left-nested spine up to
    /// [`crate::parser::MAX_FLAT_CHAIN_DEPTH`] deep. The spine may MIX
    /// `BinaryOp` / `UnaryOp` / `In` / `IsNull` levels (the grammar's
    /// `( comparison_op ~ add_expr | special_pred )*` repetition
    /// interleaves them freely), so all four variants despine through
    /// this one driver: walk down the left/operand edge collecting one
    /// frame per level, bind the non-spine base via the ordinary
    /// [`Self::bind_expression`] arms, then fold the frames back up.
    /// Left-associativity, operand order, and error-emission order are
    /// identical to the recursive binding this replaces (base subtree
    /// first, then each level's rhs from innermost to outermost). The
    /// `rhs` operands are bound recursively — they are never part of
    /// the LEFT spine, so their depth is bounded by the bracket cap
    /// (`MAX_EXPRESSION_DEPTH`), not the chain cap.
    fn bind_operator_spine(&mut self, e: &Expression) -> BoundExpression {
        enum SpineFrame<'a> {
            Binary { op: BinOp, rhs: &'a Expression },
            Unary { op: UnaryOp },
            In { rhs: &'a Expression },
            IsNull { negated: bool },
        }
        let mut frames: Vec<SpineFrame<'_>> = Vec::new();
        let mut base = e;
        loop {
            match base {
                Expression::BinaryOp { op, lhs, rhs } => {
                    frames.push(SpineFrame::Binary {
                        op: op.clone(),
                        rhs,
                    });
                    base = lhs;
                }
                Expression::UnaryOp { op, operand } => {
                    frames.push(SpineFrame::Unary { op: op.clone() });
                    base = operand;
                }
                Expression::In { lhs, rhs } => {
                    frames.push(SpineFrame::In { rhs });
                    base = lhs;
                }
                Expression::IsNull { lhs, negated } => {
                    frames.push(SpineFrame::IsNull { negated: *negated });
                    base = lhs;
                }
                _ => break,
            }
        }

        let mut acc = self.bind_expression(base);
        while let Some(frame) = frames.pop() {
            let span = acc.span().clone();
            acc = match frame {
                SpineFrame::Binary { op, rhs } => BoundExpression::BinaryOp {
                    op,
                    lhs: Box::new(acc),
                    rhs: Box::new(self.bind_expression(rhs)),
                    span,
                    type_info: None,
                },
                SpineFrame::Unary { op } => BoundExpression::UnaryOp {
                    op,
                    operand: Box::new(acc),
                    span,
                    type_info: None,
                },
                SpineFrame::In { rhs } => BoundExpression::In {
                    lhs: Box::new(acc),
                    rhs: Box::new(self.bind_expression(rhs)),
                    span,
                    type_info: None,
                },
                SpineFrame::IsNull { negated } => BoundExpression::IsNull {
                    lhs: Box::new(acc),
                    negated,
                    span,
                    type_info: None,
                },
            };
        }
        acc
    }
}

/// Extract the cross-statement carry-over bindings from a freshly-bound
/// [`BoundStatement`] per ADR-038 §5.4.1 closure (M4-83). Walks the
/// statement's last RETURN clause (innermost, since RETURN is always
/// the trailing read-query clause when present) and emits one
/// [`CarryOverBinding`] per projection that has an in-scope name:
///
/// - `RETURN expr AS x` → `(x, may_be_null=false)`. Aliased non-
///   passthrough projections are conservatively non-nullable at v1.0
///   (matches the `bind_with_clause` convention for the same shape).
/// - `RETURN n` → `(n, may_be_null=...)`. Bare passthrough — we don't
///   have direct access to the source binding's `may_be_null` here
///   (the visitor's scope chain has been popped), so we conservatively
///   default to `false` and pin the precise nullability to v1.1+.
///   Forward-pin: issue #NEW W13γ fix-up NIT-2 — M4-83 v1.1 carry-over
///   nullability inheritance via `lookup_may_be_null` (closes
///   review-pr-285-final.md NIT-2; under the dependency and artifact policy every TODO carries
///   an issue link).
/// - `RETURN n.x` (no alias) → emits nothing.
/// - `RETURN *` (wildcard) → emits nothing at v1.0; the wildcard is
///   resolved at lowering time, not at bind time. v1.1 may emit the
///   full surrounding scope's names — pinned forward in the M4-83
///   integration test.
///
/// Non-read variants emit no returned bindings.
fn extract_returned_bindings(stmt: &BoundStatement) -> Vec<CarryOverBinding> {
    let q = match stmt {
        BoundStatement::Read(q) => q,
        _ => return Vec::new(),
    };
    // RETURN is always the last clause when present per the M4-01
    // grammar (`read_query = clause+`); a query may have a tail
    // ORDER BY / SKIP / LIMIT clause AFTER the RETURN, but those
    // do not emit names. Walk from the end and stop at the first
    // RETURN.
    for clause in q.clauses.iter().rev() {
        if let BoundClause::Return(r) = clause {
            return r
                .items
                .iter()
                .filter_map(|item| match (&item.alias, &item.kind) {
                    // `RETURN expr AS x` — aliased, conservatively
                    // non-nullable.
                    (Some(alias), _) => Some(CarryOverBinding {
                        name: alias.clone(),
                        may_be_null: false,
                    }),
                    // `RETURN n` (bare passthrough). Inheriting
                    // may_be_null from the live scope is not possible
                    // here (the scope chain has been popped); default
                    // false and pin the precise nullability inheritance
                    // to v1.1.
                    (
                        None,
                        BoundProjectionKind::Expr(BoundExpression::VariableRef { name, .. }),
                    ) => Some(CarryOverBinding {
                        name: name.clone(),
                        may_be_null: false,
                    }),
                    // Wildcard / non-passthrough non-aliased — no name
                    // to carry forward at v1.0.
                    _ => None,
                })
                .collect();
        }
    }
    Vec::new()
}

/// Derive the openCypher result-column NAMES of a union arm from its
/// terminal `RETURN` clause (ADR-185; openCypher v9 §8 column
/// compatibility). The column name of a projection item is its
/// explicit `AS` alias if present, else the canonical rendering of the
/// projected expression (`format!("{expr}")` — e.g. `a.x`, `x`, `1`),
/// matching openCypher's implicit-column-name rule. A bare `RETURN *`
/// is rendered as the opaque name `"*"`: at v1.0 the wildcard's
/// concrete columns are scope-expanded only at execution time, so the
/// static compatibility check treats `*` as a single name (two `*`
/// arms compare equal; mixing `*` with explicit columns mismatches) —
/// full wildcard-union column expansion is a documented v1.1 follow-up
/// (issue tracked in the ADR). An arm with NO `RETURN` yields the
/// empty name set.
fn terminal_return_column_names(arm: &ReadQuery) -> Vec<String> {
    for clause in arm.clauses.iter().rev() {
        if let Clause::Return(r) = clause {
            return r
                .items
                .iter()
                .map(|item| match (&item.alias, &item.kind) {
                    (Some(alias), _) => alias.clone(),
                    // `safe_expr_display` guards the non-finite-float
                    // `Display`-panic (see its doc); an un-renderable
                    // expression falls back to the opaque `"?"` name so
                    // the column-compat comparison is total (#618).
                    (None, ProjectionKind::Expr(e)) => {
                        safe_expr_display(e).unwrap_or_else(|| "?".to_string())
                    }
                    (None, ProjectionKind::Wildcard) => "*".to_string(),
                })
                .collect();
        }
    }
    Vec::new()
}

/// Return `true` if a `CALL { … }` subquery body contains a WRITE clause
/// (`CREATE` / `DELETE` / `SET` / `REMOVE` / `MERGE`) at the immediate
/// level (across all union arms). Drives the ADR-192 D-9 read-only
/// fence. Nested `CALL { … }` bodies are NOT scanned here — they are
/// scanned when each nested clause is bound (recursion through
/// [`BindingVisitor::bind_call_clause`]).
fn call_body_has_write_clause(body: &Statement) -> bool {
    fn read_query_has_write(q: &ReadQuery) -> bool {
        q.clauses.iter().any(|c| {
            matches!(
                c,
                Clause::Create(_)
                    | Clause::Delete(_)
                    | Clause::Set(_)
                    | Clause::Remove(_)
                    | Clause::Merge(_)
            )
        })
    }
    match body {
        Statement::Read(q) => read_query_has_write(q),
        Statement::Union(u) => u.arms.iter().any(read_query_has_write),
        // Grammar admits only Read/Union inside CALL{}.
        _ => false,
    }
}

/// The terminal-RETURN column NAMES of a `CALL { … }` subquery body
/// (the columns that escape the scoping fence per ADR-192 D-4). For a
/// UNION body all arms expose the same name set (openCypher v9 §8,
/// enforced at bind), so arm-0 is representative. Reuses
/// [`terminal_return_column_names`] (the same column-name derivation the
/// UNION column-compat check uses).
fn call_body_returned_names(body: &Statement) -> Vec<String> {
    match body {
        Statement::Read(q) => terminal_return_column_names(q),
        Statement::Union(u) => u
            .arms
            .first()
            .map(terminal_return_column_names)
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// Render `e` to its canonical openCypher source string, returning
/// `None` if the `Display` impl fails (it returns `Err` for a non-finite
/// float literal — `1.34E999` → `inf` — which the bareword grammar
/// cannot round-trip). Unlike `format!("{e}")`, this NEVER panics. #618.
fn safe_expr_display(e: &Expression) -> Option<String> {
    use std::fmt::Write as _;
    let mut s = String::new();
    write!(s, "{e}").ok().map(|()| s)
}

/// True iff `e` is the STATICALLY-NULL literal (`null` / `NULL`).
///
/// Deliberately SYNTACTIC and narrow: only a bare `null` literal qualifies,
/// NOT an expression that merely evaluates to null at runtime (e.g.
/// `1 + null`, `head([])`, a null-valued parameter). Per openCypher the
/// `null` type unifies with NODE, so `WITH null AS a OPTIONAL MATCH (a)`
/// is well-typed (it null-extends at runtime). Generalising to
/// runtime-null expressions would require a full nullability/type
/// inference pass that the binder does not have at v1.0-alpha; the
/// conservative literal-only check ships exactly the
/// `expressions/path/Path1[1]` + `Path2[3]` surface and leaves every
/// non-literal value anchor a `VariableTypeConflict` (the correct default
/// — a non-null value does NOT unify with NODE).
fn is_static_null_literal(e: &Expression) -> bool {
    matches!(e, Expression::Literal(Literal::Null))
}

/// True if `name` is an aggregating function (`count`/`sum`/`avg`/`min`/
/// `max`/`collect`), case-insensitively. Single source of truth =
/// [`crate::logical_plan::types::AggregationKind::from_function_name`]
/// (the same predicate the lowering pass uses for the implicit GROUP
/// BY), so the bind-time position check and the lowering agree exactly.
/// #618.
fn is_aggregate_fn(name: &str) -> bool {
    crate::logical_plan::types::AggregationKind::from_function_name(name).is_some()
}

/// True if `e` references a bound variable ANYWHERE in its expression
/// tree — i.e. a bare `Identifier`, a `PropertyAccess` base, or a
/// `MapProjection` base. Used by the `SKIP` / `LIMIT` constant-ness
/// check (`NonConstantExpression`, openCypher v9 §6.4): `SKIP n.count`
/// references `n`. Comprehension-scoped variables count as references
/// here too (a `SKIP` that depends on a comprehension is non-constant).
/// #618 GA Lane BINDER-VALIDATIONS.
fn expr_references_variable(e: &Expression) -> bool {
    match e {
        Expression::Identifier(_) | Expression::MapProjection { .. } => true,
        Expression::PropertyAccess { .. } => true,
        Expression::Literal(Literal::List(items)) => items.iter().any(expr_references_variable),
        Expression::Literal(Literal::Map(entries)) => {
            entries.iter().any(|(_, v)| expr_references_variable(v))
        }
        Expression::Literal(_) | Expression::Parameter(_) => false,
        // #1290 — left-nested operator SPINE walked iteratively (the
        // spine can be `MAX_FLAT_CHAIN_DEPTH` deep and may interleave
        // all four operator variants; recursing per level overflowed
        // the native stack). Base subtree first, then each level's rhs
        // innermost→outermost — the same visit order as the recursion
        // this replaces. Non-spine children recurse (bracket-bounded).
        Expression::BinaryOp { .. }
        | Expression::UnaryOp { .. }
        | Expression::In { .. }
        | Expression::IsNull { .. } => {
            let mut rhs_stack: Vec<&Expression> = Vec::new();
            let mut cur = e;
            let base_hit = loop {
                match cur {
                    Expression::BinaryOp { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    Expression::In { lhs, rhs } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    Expression::UnaryOp { operand, .. } => cur = operand,
                    Expression::IsNull { lhs, .. } => cur = lhs,
                    other => break expr_references_variable(other),
                }
            };
            base_hit
                || rhs_stack
                    .iter()
                    .rev()
                    .any(|rhs| expr_references_variable(rhs))
        }
        Expression::FunctionCall { args, .. } => args.iter().any(expr_references_variable),
        Expression::Near { lhs, target, .. } => {
            expr_references_variable(lhs) || expr_references_variable(target)
        }
        Expression::TextMatch { lhs, query, .. } => {
            expr_references_variable(lhs) || expr_references_variable(query)
        }
        Expression::InCommunity {
            node, community, ..
        } => expr_references_variable(node) || expr_references_variable(community),
        Expression::ListPredicate {
            list, predicate, ..
        } => expr_references_variable(list) || expr_references_variable(predicate),
        Expression::Reduce {
            init, list, expr, ..
        } => {
            expr_references_variable(init)
                || expr_references_variable(list)
                || expr_references_variable(expr)
        }
        Expression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            expr_references_variable(list)
                || predicate.as_deref().is_some_and(expr_references_variable)
                || projection.as_deref().is_some_and(expr_references_variable)
        }
        Expression::Subscript { base, index } => {
            expr_references_variable(base) || expr_references_variable(index)
        }
        Expression::Slice { base, start, end } => {
            expr_references_variable(base)
                || start.as_deref().is_some_and(expr_references_variable)
                || end.as_deref().is_some_and(expr_references_variable)
        }
        Expression::Case {
            test,
            branches,
            default,
        } => {
            test.as_deref().is_some_and(expr_references_variable)
                || branches
                    .iter()
                    .any(|(w, t)| expr_references_variable(w) || expr_references_variable(t))
                || default.as_deref().is_some_and(expr_references_variable)
        }
    }
}

/// True if `e` contains an aggregating function call ANYWHERE in its
/// expression tree. Used to reject aggregation in illegal positions
/// (`WHERE` / `ORDER BY` / list comprehension) per openCypher v9 §6.4
/// (`InvalidAggregation`). The walk is total over the [`Expression`]
/// shape; any new variant defaults to recursing its sub-expressions.
/// #618 GA Lane BINDER-VALIDATIONS.
fn expr_contains_aggregate(e: &Expression) -> bool {
    match e {
        Expression::FunctionCall { name, args, .. } => {
            is_aggregate_fn(name) || args.iter().any(expr_contains_aggregate)
        }
        // List / map literals nest sub-expressions (`[1, count(*)]`,
        // `{k: count(*)}`); scalar literals do not.
        Expression::Literal(Literal::List(items)) => items.iter().any(expr_contains_aggregate),
        Expression::Literal(Literal::Map(entries)) => {
            entries.iter().any(|(_, v)| expr_contains_aggregate(v))
        }
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Identifier(_)
        // A map-projection base is a bare variable (no sub-expression
        // that could be an aggregate).
        | Expression::MapProjection { .. } => false,
        Expression::PropertyAccess { base, .. } => expr_contains_aggregate(base),
        // #1290 — left-nested operator SPINE walked iteratively (see
        // `expr_references_variable` for the pattern rationale).
        Expression::BinaryOp { .. }
        | Expression::UnaryOp { .. }
        | Expression::In { .. }
        | Expression::IsNull { .. } => {
            let mut rhs_stack: Vec<&Expression> = Vec::new();
            let mut cur = e;
            let base_hit = loop {
                match cur {
                    Expression::BinaryOp { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    Expression::In { lhs, rhs } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    Expression::UnaryOp { operand, .. } => cur = operand,
                    Expression::IsNull { lhs, .. } => cur = lhs,
                    other => break expr_contains_aggregate(other),
                }
            };
            base_hit
                || rhs_stack
                    .iter()
                    .rev()
                    .any(|rhs| expr_contains_aggregate(rhs))
        }
        Expression::Near { lhs, target, .. } => {
            expr_contains_aggregate(lhs) || expr_contains_aggregate(target)
        }
        Expression::TextMatch { lhs, query, .. } => {
            expr_contains_aggregate(lhs) || expr_contains_aggregate(query)
        }
        Expression::InCommunity { node, community, .. } => {
            expr_contains_aggregate(node) || expr_contains_aggregate(community)
        }
        Expression::ListPredicate { list, predicate, .. } => {
            expr_contains_aggregate(list) || expr_contains_aggregate(predicate)
        }
        Expression::Reduce { init, list, expr, .. } => {
            expr_contains_aggregate(init)
                || expr_contains_aggregate(list)
                || expr_contains_aggregate(expr)
        }
        Expression::ListComprehension { list, predicate, projection, .. } => {
            expr_contains_aggregate(list)
                || predicate.as_deref().is_some_and(expr_contains_aggregate)
                || projection.as_deref().is_some_and(expr_contains_aggregate)
        }
        Expression::Subscript { base, index } => {
            expr_contains_aggregate(base) || expr_contains_aggregate(index)
        }
        Expression::Slice { base, start, end } => {
            expr_contains_aggregate(base)
                || start.as_deref().is_some_and(expr_contains_aggregate)
                || end.as_deref().is_some_and(expr_contains_aggregate)
        }
        Expression::Case { test, branches, default } => {
            test.as_deref().is_some_and(expr_contains_aggregate)
                || branches
                    .iter()
                    .any(|(w, t)| expr_contains_aggregate(w) || expr_contains_aggregate(t))
                || default.as_deref().is_some_and(expr_contains_aggregate)
        }
    }
}

/// Append the `scoped` iteration-variable identifiers to `grouping_keys`,
/// producing the grouping-key set in force INSIDE an ADR-188 scoped-variable
/// body (`ListPredicate` / `ListComprehension` / `Reduce`). A scoped variable
/// is locally bound (not a free outer-scope reference), so a bare reference to
/// it inside the body is exempt from the openCypher v9 §6.4 grouping-key
/// requirement — modeled here by treating it AS a grouping key for the body
/// recursion only.
fn extend_keys<'a>(
    grouping_keys: &[&'a Expression],
    scoped: &'a [Expression],
) -> Vec<&'a Expression> {
    let mut keys: Vec<&'a Expression> = grouping_keys.to_vec();
    keys.extend(scoped.iter());
    keys
}

/// openCypher v9 §6.4 grouping-key walk — worker for
/// [`BindingVisitor::check_aggregation_grouping`] (ADR-038 amendment-12).
/// Returns `true` if the aggregating projection `e` contains a simple
/// variable/property reference that is OUTSIDE every aggregate-function
/// argument AND is not one of `grouping_keys`. A property access is checked
/// as a WHOLE (`me.age` — never descended into its base), so a simple
/// grouping key matches but a complex one (`a + b`) forces its leaves to be
/// checked individually. ADR-188 scoped-variable forms
/// (`ListPredicate`/`ListComprehension`/`Reduce`) add their iteration
/// variable(s) to the grouping-key set for the BODY recursion (via
/// [`extend_keys`]) — a locally-bound iteration variable is not a free
/// reference and so is exempt. Exhaustive over `Expression` (no wildcard) so a
/// new variant must be classified here, never silently skipped.
fn agg_has_nongrouping_ref(e: &Expression, grouping_keys: &[&Expression]) -> bool {
    match e {
        // Aggregate call: args are aggregated away (not governed). Non-
        // aggregate call: recurse into its arguments.
        Expression::FunctionCall { name, args, .. } => {
            !is_aggregate_fn(name)
                && args
                    .iter()
                    .any(|a| agg_has_nongrouping_ref(a, grouping_keys))
        }
        // The governed leaf: a simple variable / property reference must BE a
        // grouping key (span-free AST equality).
        Expression::Identifier(_) | Expression::PropertyAccess { .. } => {
            !grouping_keys.contains(&e)
        }
        Expression::Literal(Literal::List(items)) => items
            .iter()
            .any(|i| agg_has_nongrouping_ref(i, grouping_keys)),
        Expression::Literal(Literal::Map(entries)) => entries
            .iter()
            .any(|(_, v)| agg_has_nongrouping_ref(v, grouping_keys)),
        Expression::Literal(_) | Expression::Parameter(_) | Expression::MapProjection { .. } => {
            false
        }
        // #1290 — left-nested operator SPINE walked iteratively (see
        // `expr_references_variable` for the pattern rationale). The
        // spine variants never extend the grouping-key set (only the
        // ADR-188 scoped-variable forms below do), so the same
        // `grouping_keys` applies to every level.
        Expression::BinaryOp { .. }
        | Expression::UnaryOp { .. }
        | Expression::In { .. }
        | Expression::IsNull { .. } => {
            let mut rhs_stack: Vec<&Expression> = Vec::new();
            let mut cur = e;
            let base_hit = loop {
                match cur {
                    Expression::BinaryOp { lhs, rhs, .. } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    Expression::In { lhs, rhs } => {
                        rhs_stack.push(rhs);
                        cur = lhs;
                    }
                    Expression::UnaryOp { operand, .. } => cur = operand,
                    Expression::IsNull { lhs, .. } => cur = lhs,
                    other => break agg_has_nongrouping_ref(other, grouping_keys),
                }
            };
            base_hit
                || rhs_stack
                    .iter()
                    .rev()
                    .any(|rhs| agg_has_nongrouping_ref(rhs, grouping_keys))
        }
        Expression::Near { lhs, target, .. } => {
            agg_has_nongrouping_ref(lhs, grouping_keys)
                || agg_has_nongrouping_ref(target, grouping_keys)
        }
        Expression::TextMatch { lhs, query, .. } => {
            agg_has_nongrouping_ref(lhs, grouping_keys)
                || agg_has_nongrouping_ref(query, grouping_keys)
        }
        Expression::InCommunity {
            node, community, ..
        } => {
            agg_has_nongrouping_ref(node, grouping_keys)
                || agg_has_nongrouping_ref(community, grouping_keys)
        }
        // ADR-188 scoped-variable forms (`ListPredicate`,
        // `ListComprehension`, `Reduce`) bind an iteration variable (`var`,
        // and `acc_var` for `Reduce`) ONLY inside the body sub-expressions
        // (`predicate` / `projection` / fold `expr`); the `list` / `init`
        // sources are evaluated in the OUTER scope. A reference to a scoped
        // iteration variable is NOT a free outer-scope reference, so it must
        // NOT be required to be a grouping key (openCypher v9 §6.4 governs
        // FREE references only). Without this, `ALL(ok IN collect(...) WHERE
        // ok)` — a legal aggregating projection — wrongly raised
        // `AmbiguousAggregationExpression` because `ok` (the quantifier
        // variable) is not in the grouping-key set (TCK `List11` [3]). We
        // extend the grouping-key set with the scoped variable(s) for the
        // body recursion so a bare reference to them passes; `list` / `init`
        // stay checked against the original keys.
        Expression::ListPredicate {
            var,
            list,
            predicate,
            ..
        } => {
            agg_has_nongrouping_ref(list, grouping_keys) || {
                let scoped = [Expression::Identifier(var.clone())];
                agg_has_nongrouping_ref(predicate, &extend_keys(grouping_keys, &scoped))
            }
        }
        Expression::Reduce {
            acc_var,
            init,
            var,
            list,
            expr,
        } => {
            agg_has_nongrouping_ref(init, grouping_keys)
                || agg_has_nongrouping_ref(list, grouping_keys)
                || {
                    let scoped = [
                        Expression::Identifier(acc_var.clone()),
                        Expression::Identifier(var.clone()),
                    ];
                    agg_has_nongrouping_ref(expr, &extend_keys(grouping_keys, &scoped))
                }
        }
        Expression::ListComprehension {
            var,
            list,
            predicate,
            projection,
        } => {
            agg_has_nongrouping_ref(list, grouping_keys) || {
                let scoped = [Expression::Identifier(var.clone())];
                let body_keys = extend_keys(grouping_keys, &scoped);
                predicate
                    .as_deref()
                    .is_some_and(|p| agg_has_nongrouping_ref(p, &body_keys))
                    || projection
                        .as_deref()
                        .is_some_and(|p| agg_has_nongrouping_ref(p, &body_keys))
            }
        }
        Expression::Subscript { base, index } => {
            agg_has_nongrouping_ref(base, grouping_keys)
                || agg_has_nongrouping_ref(index, grouping_keys)
        }
        Expression::Slice { base, start, end } => {
            agg_has_nongrouping_ref(base, grouping_keys)
                || start
                    .as_deref()
                    .is_some_and(|s| agg_has_nongrouping_ref(s, grouping_keys))
                || end
                    .as_deref()
                    .is_some_and(|en| agg_has_nongrouping_ref(en, grouping_keys))
        }
        Expression::Case {
            test,
            branches,
            default,
        } => {
            test.as_deref()
                .is_some_and(|t| agg_has_nongrouping_ref(t, grouping_keys))
                || branches.iter().any(|(w, t)| {
                    agg_has_nongrouping_ref(w, grouping_keys)
                        || agg_has_nongrouping_ref(t, grouping_keys)
                })
                || default
                    .as_deref()
                    .is_some_and(|d| agg_has_nongrouping_ref(d, grouping_keys))
        }
    }
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;
    use crate::semantic::catalog::StubCatalogProvider;

    fn read(stmt: &BoundStatement) -> &BoundQuery {
        match stmt {
            BoundStatement::Read(q) => q,
            other => panic!("expected BoundStatement::Read, got {other:?}"),
        }
    }

    /// Walk all `BoundExpression` nodes and collect the binding-ids
    /// of `VariableRef` nodes (in source order).
    fn collect_var_ref_binding_ids(q: &BoundQuery) -> Vec<(String, BindingId)> {
        let mut acc = Vec::new();
        for c in &q.clauses {
            collect_clause_var_refs(c, &mut acc);
        }
        acc
    }

    fn collect_clause_var_refs(c: &BoundClause, acc: &mut Vec<(String, BindingId)>) {
        match c {
            BoundClause::Match(m) => {
                if let Some(w) = &m.where_clause {
                    collect_expr_var_refs(w, acc);
                }
            }
            BoundClause::With(w) => {
                for it in &w.items {
                    if let BoundProjectionKind::Expr(e) = &it.kind {
                        collect_expr_var_refs(e, acc);
                    }
                }
                if let Some(we) = &w.where_clause {
                    collect_expr_var_refs(we, acc);
                }
            }
            BoundClause::Return(r) => {
                for it in &r.items {
                    if let BoundProjectionKind::Expr(e) = &it.kind {
                        collect_expr_var_refs(e, acc);
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_expr_var_refs(e: &BoundExpression, acc: &mut Vec<(String, BindingId)>) {
        match e {
            BoundExpression::VariableRef {
                name, binding_id, ..
            } => acc.push((name.clone(), *binding_id)),
            BoundExpression::PropertyAccess { base, .. } => {
                collect_expr_var_refs(base, acc);
            }
            // #1290 — left-nested operator SPINE walked iteratively
            // (the spine can be `MAX_FLAT_CHAIN_DEPTH` deep and may
            // interleave all four operator variants; recursing per
            // level overflowed the native stack). Base subtree first,
            // then each level's rhs innermost→outermost — the same
            // in-order `acc` push sequence as the recursion this
            // replaces.
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
                            collect_expr_var_refs(other, acc);
                            break;
                        }
                    }
                }
                while let Some(rhs) = rhs_stack.pop() {
                    collect_expr_var_refs(rhs, acc);
                }
            }
            BoundExpression::FunctionCall { args, .. } => {
                for a in args {
                    collect_expr_var_refs(a, acc);
                }
            }
            BoundExpression::Near { lhs, target, .. } => {
                collect_expr_var_refs(lhs, acc);
                collect_expr_var_refs(target, acc);
            }
            BoundExpression::TextMatch { lhs, query, .. } => {
                collect_expr_var_refs(lhs, acc);
                collect_expr_var_refs(query, acc);
            }
            BoundExpression::InCommunity {
                node, community, ..
            } => {
                collect_expr_var_refs(node, acc);
                collect_expr_var_refs(community, acc);
            }
            _ => {}
        }
    }

    /// Walk to find the BindingId of a declaration with the given
    /// name (first occurrence in source order).
    fn find_decl_binding_id(q: &BoundQuery, name: &str) -> Option<BindingId> {
        for c in &q.clauses {
            if let BoundClause::Match(m) = c {
                if let BoundMatchBody::Patterns(ps) = &m.body {
                    for p in ps {
                        if let Some(v) = &p.head.var {
                            if v.name == name {
                                return Some(v.binding_id);
                            }
                        }
                        for (rel, node) in &p.tail {
                            if let Some(v) = &rel.var {
                                if v.name == name {
                                    return Some(v.binding_id);
                                }
                            }
                            if let Some(v) = &node.var {
                                if v.name == name {
                                    return Some(v.binding_id);
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }

    // ---- The 6 unit tests (5 from brief + 1 negative-case companion) ----

    #[test]
    fn bind_single_match_variable() {
        let input = "MATCH (n) RETURN n";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);
        assert_eq!(q.snapshot_lsn, None, "M4-21 reserves snapshot_lsn = None");

        let decl_id = find_decl_binding_id(q, "n").expect("n declared");
        let refs = collect_var_ref_binding_ids(q);
        assert_eq!(refs.len(), 1, "RETURN n is one VariableRef");
        assert_eq!(refs[0].0, "n");
        assert_eq!(
            refs[0].1, decl_id,
            "RETURN n must resolve to the MATCH (n) declaration"
        );
    }

    // ---- #842 part B — WITH DISTINCT threading (AST → BoundAST) ----

    #[test]
    fn cz842_bind_with_distinct_threaded() {
        let cat = StubCatalogProvider::new();
        let distinct_of = |input: &str| -> bool {
            let stmt = parse(input).expect("parse");
            let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
            let q = read(&bound);
            q.clauses
                .iter()
                .find_map(|c| match c {
                    BoundClause::With(w) => Some(w.distinct),
                    _ => None,
                })
                .expect("a WITH clause")
        };
        assert!(
            distinct_of("MATCH (n) WITH DISTINCT n AS m RETURN m"),
            "WITH DISTINCT threads distinct=true into BoundWithClause"
        );
        assert!(
            !distinct_of("MATCH (n) WITH n AS m RETURN m"),
            "plain WITH leaves distinct=false in BoundWithClause"
        );
    }

    // ---- #836 — RETURN-clause ORDER BY over a projected expression ----

    /// The output_id of the first RETURN projection item.
    fn first_return_output_id(q: &BoundQuery) -> BindingId {
        q.clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::Return(r) => r.items.first().and_then(|it| it.output_id),
                _ => None,
            })
            .expect("RETURN item has an output_id")
    }

    /// The first standalone tail `ORDER BY` key (bound expression).
    fn first_tail_order_key(q: &BoundQuery) -> BoundExpression {
        q.clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::TailOrderBy(items, _) => items.first().map(|o| o.expr.clone()),
                _ => None,
            })
            .expect("a TailOrderBy clause with one key")
    }

    #[test]
    fn cz836_orderby_projected_expr_resolves_to_output_id() {
        // `RETURN p.name ORDER BY p.name`: the tail ORDER BY key MUST
        // resolve to the RETURN item's projected `output_id` (the column
        // the `ProjectOp` emits + the `Sort` over it sees), NOT the
        // pre-projection `p` binding (which `Project` drops → runtime
        // "binding … missing from row schema"). Binder-level structural
        // proof, complementary to the e2e row-order oracle.
        let input = "MATCH (p) RETURN p.name ORDER BY p.name";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new().with_properties(["name"]);
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);

        let output_id = first_return_output_id(q);
        match first_tail_order_key(q) {
            BoundExpression::VariableRef { binding_id, .. } => assert_eq!(
                binding_id, output_id,
                "#836: ORDER BY p.name must resolve to the projected column's output_id"
            ),
            other => panic!(
                "#836: ORDER BY key must be rewritten to a VariableRef to the \
                 projected output column, got {other:?}"
            ),
        }
    }

    #[test]
    fn cz836_orderby_alias_still_resolves_via_name_scope() {
        // Control: `RETURN p.name AS n ORDER BY n` still resolves the
        // alias via the output-NAME scope (#618), unaffected by the #836
        // expression map. The key is a VariableRef to the alias's
        // output_id.
        let input = "MATCH (p) RETURN p.name AS n ORDER BY n";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new().with_properties(["name"]);
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);

        let output_id = first_return_output_id(q);
        match first_tail_order_key(q) {
            BoundExpression::VariableRef { binding_id, .. } => assert_eq!(
                binding_id, output_id,
                "ORDER BY <alias> resolves to the alias's output_id (unchanged)"
            ),
            other => panic!("expected a VariableRef to the alias output, got {other:?}"),
        }
    }

    #[test]
    fn cz836_orderby_nonprojected_inscope_var_is_not_rewritten_boundary() {
        // Documented BOUNDARY (deferred #618 follow-up): ORDER BY a
        // NON-projected in-scope expression (`RETURN p.name ORDER BY
        // p.age`) is NOT rewritten by #836 — only an expression that was
        // actually PROJECTED resolves to an output column. `p.age` stays a
        // PropertyAccess on `p`'s pre-projection binding (full support
        // needs `Project` to carry a hidden sort column — a structurally
        // larger change, out of scope for #836). This test pins the scope
        // so a future fix is a deliberate change, not an accident.
        let input = "MATCH (p) RETURN p.name ORDER BY p.age";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new().with_properties(["name", "age"]);
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);

        assert!(
            matches!(
                first_tail_order_key(q),
                BoundExpression::PropertyAccess { .. }
            ),
            "#836 boundary: ORDER BY a non-projected in-scope expr stays a \
             PropertyAccess (not rewritten to an output column)"
        );
    }

    #[test]
    fn cz836_orderby_multi_key_both_projected_exprs_resolve_to_outputs() {
        // Multi-key `RETURN a.x, a.y ORDER BY a.x, a.y`: BOTH keys resolve
        // to their respective projected output_ids (precedence preserved).
        let input = "MATCH (a) RETURN a.x, a.y ORDER BY a.x, a.y";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new().with_properties(["x", "y"]);
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);

        let outs: Vec<BindingId> = q
            .clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::Return(r) => {
                    Some(r.items.iter().filter_map(|it| it.output_id).collect())
                }
                _ => None,
            })
            .expect("RETURN output ids");
        assert_eq!(outs.len(), 2, "two projected columns");

        let keys: Vec<BindingId> = q
            .clauses
            .iter()
            .find_map(|c| match c {
                BoundClause::TailOrderBy(items, _) => Some(
                    items
                        .iter()
                        .map(|o| match &o.expr {
                            BoundExpression::VariableRef { binding_id, .. } => *binding_id,
                            other => panic!("#836: key not rewritten: {other:?}"),
                        })
                        .collect(),
                ),
                _ => None,
            })
            .expect("TailOrderBy with two keys");
        assert_eq!(
            keys, outs,
            "#836: ORDER BY a.x, a.y must resolve to [output_x, output_y] in order"
        );
    }

    #[test]
    fn reject_undeclared_variable() {
        let input = "MATCH (n) RETURN m";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        let errs = BindingVisitor::bind(&stmt, input, &cat).expect_err("expected errors");
        assert_eq!(errs.len(), 1);
        match &errs[0] {
            BindingError::UndeclaredVariable { name, .. } => assert_eq!(name, "m"),
            other => panic!("expected UndeclaredVariable, got {other:?}"),
        }
    }

    #[test]
    fn bind_through_with_chain() {
        let input = "MATCH (n) WITH n AS x RETURN x";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);
        assert!(
            q.clauses.iter().any(|c| matches!(c, BoundClause::With(_))),
            "expected a WITH clause"
        );
        // RETURN x must resolve. Find the x binding declared by WITH.
        let with_x_binding = q.clauses.iter().find_map(|c| match c {
            BoundClause::With(w) => w.items.iter().find_map(|i| {
                if i.alias.as_deref() == Some("x") {
                    Some(())
                } else {
                    None
                }
            }),
            _ => None,
        });
        assert!(with_x_binding.is_some(), "x is the WITH alias");

        let refs = collect_var_ref_binding_ids(q);
        // Two refs in source order: WITH's `n` projection, RETURN's `x`.
        assert!(
            refs.iter().any(|(n, _)| n == "x"),
            "RETURN x should resolve as a VariableRef"
        );
    }

    #[test]
    fn with_drops_unprojected_variables() {
        // Negative-case companion to bind_through_with_chain:
        // `n` is not visible in RETURN because WITH only projected
        // `x`. Expect UndeclaredVariable for `n`.
        let input = "MATCH (n) WITH n AS x RETURN n";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        let errs = BindingVisitor::bind(&stmt, input, &cat).expect_err("expected errors");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "n"
            )),
            "expected UndeclaredVariable for `n`, got {errs:?}"
        );
    }

    // =================================================================
    // ADR-188 scoped-variable forms inside an aggregating projection —
    // the implicit-grouping-key walk (`agg_has_nongrouping_ref`) must NOT
    // treat a comprehension / quantifier / reduce iteration variable as a
    // FREE reference that has to be a grouping key (openCypher v9 §6.4
    // governs free references only). TCK `List11` [3].
    // =================================================================

    /// The exact `List11` [3] tail shape: `collect(...)` nested inside an
    /// `ALL(ok IN ... WHERE ok)` quantifier. `ok` is the quantifier's
    /// scoped variable — NOT a grouping key — so this MUST bind cleanly
    /// (pre-fix it raised `AmbiguousAggregationExpression`).
    #[test]
    fn aggregate_inside_all_quantifier_scoped_var_is_exempt() {
        let input =
            "UNWIND [1, 2] AS x WITH x, x AS y RETURN ALL(ok IN collect(x = y) WHERE ok) AS okay";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        BindingVisitor::bind(&stmt, input, &cat)
            .expect("ALL(ok IN collect(...) WHERE ok) is a valid aggregating projection");
    }

    /// `reduce(acc = 0, e IN collect(x) | acc + e)` — both `acc` and `e`
    /// are reduce-scoped; neither needs to be a grouping key.
    #[test]
    fn aggregate_inside_reduce_scoped_vars_are_exempt() {
        let input = "UNWIND [1, 2] AS x RETURN reduce(acc = 0, e IN collect(x) | acc + e) AS total";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        BindingVisitor::bind(&stmt, input, &cat)
            .expect("reduce over collect(...) with scoped acc/elem is valid");
    }

    /// `[e IN collect(x) WHERE e > 0]` — the list-comprehension variable
    /// `e` is scoped to the WHERE / projection; exempt from grouping.
    #[test]
    fn aggregate_inside_list_comprehension_scoped_var_is_exempt() {
        let input = "UNWIND [1, 2] AS x RETURN [e IN collect(x) WHERE e > 0] AS pos";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        BindingVisitor::bind(&stmt, input, &cat)
            .expect("list comprehension over collect(...) with scoped elem is valid");
    }

    /// Regression guard — the exemption is NARROW: a FREE (non-scoped,
    /// non-grouping) reference inside a scoped-form body still raises
    /// `AmbiguousAggregationExpression`. Here `y` is a free outer variable
    /// referenced inside the `ALL` body but is NOT a grouping key (only `x`
    /// is projected as the implicit key alongside the aggregate), so the
    /// §6.4 rule must still fire.
    #[test]
    fn free_var_inside_scoped_form_still_rejected() {
        let input = "UNWIND [1, 2] AS x UNWIND [3, 4] AS y \
             RETURN x, ALL(ok IN collect(ok) WHERE ok = y) AS okay";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        let errs = BindingVisitor::bind(&stmt, input, &cat)
            .expect_err("free var `y` inside the ALL body is non-grouping");
        assert!(
            errs.iter()
                .any(|e| matches!(e, BindingError::AmbiguousAggregationExpression { .. })),
            "expected AmbiguousAggregationExpression for the free `y` ref, got {errs:?}"
        );
    }

    #[test]
    fn bind_across_match_chain_shares_scope() {
        let input = "MATCH (a) MATCH (b) RETURN a, b";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind(&stmt, input, &cat).expect("bind");
        let q = read(&bound);

        let a_id = find_decl_binding_id(q, "a").expect("a declared");
        let b_id = find_decl_binding_id(q, "b").expect("b declared");
        assert_ne!(a_id, b_id, "a and b are distinct bindings");

        let refs = collect_var_ref_binding_ids(q);
        let a_refs: Vec<_> = refs.iter().filter(|(n, _)| n == "a").collect();
        let b_refs: Vec<_> = refs.iter().filter(|(n, _)| n == "b").collect();
        assert_eq!(a_refs.len(), 1);
        assert_eq!(b_refs.len(), 1);
        assert_eq!(a_refs[0].1, a_id);
        assert_eq!(b_refs[0].1, b_id);
    }

    #[test]
    fn unknown_label_binds_permissively_to_sentinel() {
        // ADR-038 amendment-12 (#796): an unknown label is NO LONGER rejected
        // (was `reject_unknown_label`). It binds to the `UNRESOLVED_LABEL`
        // sentinel — no node carries it, so the pattern matches nothing per
        // openCypher "unknown label ⇒ empty match". The `UnknownLabel` variant
        // is retained for the v1.1+ strict-schema mode. The runtime
        // "matches nothing" semantics are pinned end-to-end by
        // `tests/permissive_label_binding_e2e.rs`.
        let input = "MATCH (n:Foo) RETURN n";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        // Binds WITHOUT error (the contract change) — pre-amendment this
        // returned `Err([UnknownLabel])`. The sentinel-id + runtime
        // "matches nothing" semantics are pinned end-to-end by
        // `tests/permissive_label_binding_e2e.rs` (unknown label ⇒ 0 rows,
        // known label unaffected).
        BindingVisitor::bind(&stmt, input, &cat)
            .expect("unknown label must bind permissively, not error (#796)");
    }

    #[test]
    fn unknown_rel_type_binds_permissively() {
        // Symmetric to `unknown_label_binds_permissively_to_sentinel` for an
        // unknown relationship type (#796 / ADR-038 amendment-12).
        let input = "MATCH (a)-[:GHOSTREL]->(b) RETURN b";
        let stmt = parse(input).expect("parse");
        let cat = StubCatalogProvider::new();
        BindingVisitor::bind(&stmt, input, &cat)
            .expect("unknown rel-type must bind permissively, not error (#796)");
    }

    // =================================================================
    // M4-83 multi-statement binding unit tests (ADR-038 §5.4.1 closure)
    // =================================================================

    use crate::parse_multi;

    #[test]
    fn bind_multi_three_statement_chain_returns_three_bound() {
        let q = "MATCH (a) RETURN a; MATCH (b) RETURN b; MATCH (c) RETURN c";
        let stmts = parse_multi(q).expect("parse_multi");
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind_multi(&stmts, q, &cat).expect("bind_multi");
        assert_eq!(bound.len(), 3, "one bound stmt per parsed stmt");
        for b in &bound {
            assert!(matches!(b, BoundStatement::Read(_)));
        }
    }

    #[test]
    fn bind_multi_aliased_projection_carries_over() {
        // M4-83 (cross-statement variable scoping): aliased RETURN
        // projection emits the alias into the next statement's scope.
        let q = "\
            MATCH (n) RETURN n.x AS pname;\n\
            MATCH (m) RETURN m, pname\
        ";
        let stmts = parse_multi(q).expect("parse_multi");
        let cat = StubCatalogProvider::new().with_properties(["x"]);
        let bound = BindingVisitor::bind_multi(&stmts, q, &cat).expect("bind_multi clean");
        assert_eq!(bound.len(), 2);
    }

    #[test]
    fn bind_multi_bare_passthrough_carries_over() {
        // M4-83 (statement-N-sees-statement-N-1-binding): bare
        // passthrough `RETURN n` emits `n` for the next statement.
        let q = "\
            MATCH (n) RETURN n;\n\
            MATCH (m) RETURN m, n\
        ";
        let stmts = parse_multi(q).expect("parse_multi");
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind_multi(&stmts, q, &cat)
            .expect("bind_multi: bare passthrough `n` resolves in stmt 2");
        assert_eq!(bound.len(), 2);
    }

    #[test]
    fn bind_multi_unaliased_property_does_not_carry() {
        // M4-83 (cross-statement scoping negative): an unaliased
        // non-passthrough projection (`RETURN n.x`) emits no name.
        let q = "\
            MATCH (n) RETURN n.x;\n\
            MATCH (m) WHERE m.x = x RETURN m\
        ";
        let stmts = parse_multi(q).expect("parse_multi");
        let cat = StubCatalogProvider::new().with_properties(["x"]);
        let errs = BindingVisitor::bind_multi(&stmts, q, &cat)
            .expect_err("bind_multi: bare `x` should be UndeclaredVariable");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "x"
            )),
            "expected UndeclaredVariable for `x`, got {errs:?}"
        );
    }

    #[test]
    fn bind_multi_error_in_later_statement_aborts_whole() {
        // M4-83 (cross-statement-error-aborts at bind layer): a binding error
        // in stmt 2 surfaces as an error returned to the caller. (Was an
        // `UnknownLabel`; since ADR-038 amendment-12 made unknown labels
        // permissive, this uses an `UndeclaredVariable` — an undeclared `x` in
        // stmt 2 — which remains a hard bind-time error.)
        let q = "MATCH (a) RETURN a; MATCH (n) RETURN x";
        let stmts = parse_multi(q).expect("parse_multi");
        let cat = StubCatalogProvider::new();
        let errs = BindingVisitor::bind_multi(&stmts, q, &cat).expect_err("expected errors");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "x"
            )),
            "expected UndeclaredVariable for x, got {errs:?}"
        );
    }

    #[test]
    fn bind_multi_carry_over_does_not_leak_match_only_bindings() {
        // M4-83 (LSN-shared-across-statements companion at bind layer):
        // MATCH-bound but NOT RETURN-emitted variables are NOT visible
        // in stmt 2 (drop-bindings-not-in-projection per WITH).
        let q = "\
            MATCH (n) RETURN n.x AS pname;\n\
            MATCH (m) RETURN m, n\
        ";
        let stmts = parse_multi(q).expect("parse_multi");
        let cat = StubCatalogProvider::new().with_properties(["x"]);
        let errs = BindingVisitor::bind_multi(&stmts, q, &cat)
            .expect_err("`n` MATCH-bound in stmt 1 must NOT leak past RETURN AS pname");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "n"
            )),
            "expected UndeclaredVariable for `n` in stmt 2, got {errs:?}"
        );
    }

    // =================================================================
    // ADR-188 — list-predicate / reduce SCOPED-BINDING tests. The
    // load-bearing structural invariant (Decision 3): the iteration
    // variable is live ONLY inside the predicate/body and can NEVER
    // resolve in a sibling expression or a later clause.
    // =================================================================

    #[test]
    fn lp_scoped_var_resolves_inside_predicate() {
        // all(x IN [1,2,3] WHERE x > 0) — the `x` inside the predicate
        // MUST bind cleanly (no UndeclaredVariable).
        let q = "MATCH (n) WHERE all(x IN [1, 2, 3] WHERE x > 0) RETURN n";
        let stmt = parse(q).expect("parse");
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind(&stmt, q, &cat);
        assert!(
            bound.is_ok(),
            "scoped `x` MUST resolve inside the predicate, got {:?}",
            bound.err()
        );
    }

    #[test]
    fn lp_scoped_var_does_not_leak_to_return() {
        // The scoped `x` is NOT visible in RETURN (a sibling of the
        // WHERE). `RETURN x` MUST be UndeclaredVariable — the structural
        // lifetime invariant. A naive implementation that declared `x`
        // in the enclosing scope would FAIL this (it'd bind).
        let q = "MATCH (n) WHERE all(x IN [1, 2, 3] WHERE x > 0) RETURN x";
        let stmt = parse(q).expect("parse");
        let cat = StubCatalogProvider::new();
        let errs =
            BindingVisitor::bind(&stmt, q, &cat).expect_err("scoped `x` MUST NOT leak to RETURN");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "x"
            )),
            "expected UndeclaredVariable for leaked scoped `x`, got {errs:?}"
        );
    }

    #[test]
    fn reduce_scoped_vars_do_not_leak() {
        // reduce's `acc`/`x` are NOT visible after the reduce. `RETURN s`
        // (the accumulator name) MUST be UndeclaredVariable.
        let q = "MATCH (n) RETURN reduce(s = 0, x IN [1, 2] | s + x) AS total, s";
        let stmt = parse(q).expect("parse");
        let cat = StubCatalogProvider::new();
        let errs = BindingVisitor::bind(&stmt, q, &cat).expect_err("reduce's `s` MUST NOT leak");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "s"
            )),
            "expected UndeclaredVariable for leaked `s`, got {errs:?}"
        );
    }

    #[test]
    fn lp_nested_inner_var_does_not_leak_to_outer_predicate() {
        // any(x IN [1] WHERE all(y IN [2] WHERE y > x)) is fine, BUT the
        // inner `y` MUST NOT leak to the OUTER predicate position. We
        // test the leak by referencing `y` in the outer predicate
        // alongside the inner `all` — `y` is only in the inner scope.
        let q = "MATCH (n) WHERE any(x IN [1] WHERE all(y IN [2] WHERE y > x) AND y > 0) RETURN n";
        let stmt = parse(q).expect("parse");
        let cat = StubCatalogProvider::new();
        let errs = BindingVisitor::bind(&stmt, q, &cat)
            .expect_err("inner `y` MUST NOT leak to the outer predicate");
        assert!(
            errs.iter().any(|e| matches!(
                e,
                BindingError::UndeclaredVariable { name, .. } if name == "y"
            )),
            "expected UndeclaredVariable for leaked inner `y`, got {errs:?}"
        );
    }

    #[test]
    fn lp_predicate_can_reference_outer_match_binding() {
        // all(x IN [1,2] WHERE x > 0) co-existing with an outer `n`
        // reference inside the predicate is fine: the predicate sees BOTH
        // the scoped `x` AND the outer-scope `n`.
        let q = "MATCH (n) WHERE all(x IN [1, 2] WHERE x > 0) RETURN n";
        let stmt = parse(q).expect("parse");
        let cat = StubCatalogProvider::new();
        assert!(
            BindingVisitor::bind(&stmt, q, &cat).is_ok(),
            "outer `n` + scoped `x` MUST co-exist"
        );
    }
}
