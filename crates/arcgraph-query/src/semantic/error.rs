//! Semantic-analysis errors (M4-21 binding pass + M4-22 type-check).
//!
//! Three error types, all `thiserror`-derived with `span_byte_range`
//! mirrors of [`crate::error::ParseError`]:
//!
//! - [`BindingError`] — M4-21 binding-pass failures (undeclared
//!   variable, unknown label / rel-type, duplicate binding). Each
//!   variant carries a primary [`Span`].
//! - [`TypeCheckError`] — M4-22 type-check failures (operand-type
//!   mismatch, function arity / argument-type mismatch, unknown
//!   function / property). Each variant carries a primary [`Span`].
//! - [`ArcQLError`] — the public-API umbrella that encompasses the
//!   above + the [`ArcQLError::NotImplemented`] variant for
//!   reserved-but-unimplemented variants per ADR-038 §2 D-16.
//!
//! # Why a separate taxonomy?
//!
//! Per ADR-038 §2 D-16, the parser emits **syntactic** errors
//! ([`crate::error::ParseError`]); the semantic analyzer emits
//! **binding / type / substrate** errors. Conflating them would
//! erase the v1.0 reserved-syntax discipline (a string can be
//! syntactically valid ArcQL but semantically un-bindable, e.g.
//! a label that does not exist in the catalog).
//!
//! M4-22 introduces [`TypeCheckError`] + [`ArcQLError::NotImplemented`]
//! for the type-check layer; M4-23 will add substrate-validation
//! errors. All three layers stay distinct.
//!
//! # ADR provenance
//! - ADR-038 §2 D-16 — error taxonomy split + reserved-variant
//!   `NotImplemented` shape.
//! - ADR-038 §2 D-21 — binding-error contract (M4-21).
//! - ADR-038 §2 D-22 — type-check error contract (M4-22).
//! - ADR-038 amendment-03 §TIER-1 GAP E — span-bearing error
//!   contract mirrors `ParseError::span_byte_range`.

use arcgraph_core::TenantId;

use crate::error::Span;
use crate::logical_plan::error::LogicalPlanError;
use crate::semantic::bound_ast::TypeInfo;

/// Faults surfaced by [`crate::semantic::BindingVisitor::bind`].
///
/// Each variant carries a `span` pointing at the offending token in
/// the original input. Variants with secondary spans (e.g.
/// `DuplicateBinding`'s `prior_span`) carry the cross-reference for
/// IDE-grade error rendering.
///
/// `#[non_exhaustive]` is **omitted** on the same rationale as
/// `ParseError`: the variant set is the binding pass's public
/// contract for M4-22 (type-check) and M4-23 (substrate validation)
/// consumption. New variants land via amendment alongside future
/// semantic-pass extensions.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BindingError {
    /// A variable is referenced (e.g. in `RETURN n`, in a `WHERE`
    /// expression, in a property-access base) but was never declared
    /// in any enclosing scope.
    #[error("undeclared variable `{name}` at {span}")]
    UndeclaredVariable {
        /// The variable name as it appears in the source.
        name: String,
        /// The span of the offending reference.
        span: Span,
    },

    /// A node-pattern label `:Foo` references a label that the
    /// catalog does not know about.
    #[error("unknown label `{name}` at {span}")]
    UnknownLabel { name: String, span: Span },

    /// A relationship-pattern type `:KNOWS` references a rel-type
    /// that the catalog does not know about.
    #[error("unknown relationship type `{name}` at {span}")]
    UnknownRelType { name: String, span: Span },

    /// **ADR-197 (#802)** — `CALL <proc>(…)` references a procedure
    /// outside the v1.0-α schema-introspection catalog
    /// ([`crate::semantic::bound_ast::ProcedureKind`]).
    #[error(
        "unknown procedure `{name}` at {span} (v1.0-α supports the schema-introspection procedures: apoc.meta.data, apoc.schema.nodes, db.labels, db.relationshipTypes, db.propertyKeys, db.schema.visualization)"
    )]
    UnknownProcedure { name: String, span: Span },

    /// **ADR-197 (#802)** — a `YIELD <col>` references a column the
    /// procedure does not produce.
    #[error("procedure `{proc}` does not yield column `{column}` at {span}")]
    InvalidYieldColumn {
        /// The procedure name.
        proc: String,
        /// The unknown yielded column.
        column: String,
        /// Source span.
        span: Span,
    },

    /// A variable is declared twice in the same scope (e.g.
    /// `MATCH (n) MATCH (n)` — the second `n` collides with the
    /// first within the shared MATCH-chain scope).
    #[error("duplicate binding `{name}` at {span} (prior at {prior_span}){reason}")]
    DuplicateBinding {
        /// The duplicated variable name.
        name: String,
        /// The span of the duplicate declaration.
        span: Span,
        /// The span of the prior declaration that this duplicates.
        prior_span: Span,
        /// Optional contextual detail for duplicate-binding sites whose
        /// openCypher rule benefits from a sharper diagnostic.
        reason: String,
    },

    /// A variable is referenced in a scope where it has been dropped
    /// by an intervening `WITH` clause (e.g.
    /// `MATCH (n) WITH n AS x RETURN n` — `n` is not in scope after
    /// the `WITH` because only `x` was projected).
    ///
    /// In M4-21 this is reported as
    /// [`Self::UndeclaredVariable`] (the pragmatic encoding — once
    /// the scope chain pops, the name is simply absent). The variant
    /// is reserved here for M4-22+ which may want a stronger
    /// diagnostic with the prior-scope span attached.
    #[error(
        "variable `{name}` at {span} is not in scope (introduced at {scope_span} but dropped by intervening WITH)"
    )]
    ScopeViolation {
        name: String,
        span: Span,
        /// The span of the clause where the variable was originally
        /// in scope.
        scope_span: Span,
    },

    /// Two arms of a `UNION` / `UNION ALL` expose different result-
    /// column name SETS. openCypher v9 §8 ("Set operations") requires
    /// every arm to project the SAME set of result-column names
    /// (by-name set-equality, ORDER-INDEPENDENT — the column ORDER may
    /// differ across arms, but the NAME set must match; the executor
    /// realigns columns by name). Per ADR-185 (#649-A1, W28). This is
    /// the load-bearing union-compatibility rule (PE FROZEN CONTRACT
    /// item 3) — it MUST fail loudly with a pinning test, NOT be
    /// silently skipped. Maps to the openCypher TCK error detail
    /// `DifferentColumnsInUnion` (`clauses/union/Union{1,2}.feature`
    /// scenario `5`).
    #[error(
        "UNION arms have incompatible result columns: arm 1 projects {first:?} but arm {arm_index} projects {mismatching:?} at {span}"
    )]
    UnionColumnMismatch {
        /// Arm 1's result-column names (sorted for stable rendering).
        first: Vec<String>,
        /// The mismatching arm's result-column names (sorted).
        mismatching: Vec<String>,
        /// 1-based index of the mismatching arm.
        arm_index: usize,
        /// Span of the union (the offending arm could not be pinned to
        /// a sub-span on the v1.0 cursor; the whole-union span is the
        /// IDE-grade anchor).
        span: Span,
    },

    /// A single `UNION` query MIXES the distinct (`UNION`) and
    /// keep-duplicates (`UNION ALL`) set operators. openCypher v9 §8
    /// forbids this: a union must be uniformly `UNION` or uniformly
    /// `UNION ALL` (mixing is ambiguous w.r.t. which boundary the
    /// deduplication applies to). Per ADR-185 (#649-A1, W28). Maps to
    /// the openCypher TCK error detail `InvalidClauseComposition`
    /// (`clauses/union/Union3.feature` scenarios `1`+`2`).
    #[error("UNION and UNION ALL cannot be mixed in a single union at {span}")]
    UnionMixedSetOps {
        /// Span of the union.
        span: Span,
    },

    /// A `CALL { … }` subquery body contains a WRITE clause
    /// (`CREATE` / `DELETE` / `SET` / `REMOVE` / `MERGE`). Per ADR-192
    /// D-9, v1.0-α admits READ-ONLY subquery bodies only; write-inside-
    /// `CALL{}` (per-row write-transaction + cardinality-side-effect
    /// semantics — a write subquery's "returns 0 rows" still performed
    /// its writes) is forward-deferred to the v1.1 write-in-`CALL{}`
    /// slice (a sub-task of #623). This is a deliberate scope narrowing,
    /// not an unimplemented-feature error.
    #[error(
        "write clauses (CREATE/DELETE/SET/REMOVE/MERGE) inside CALL {{ … }} are not supported at v1.0-alpha (read-only subqueries only; write-in-CALL is forward-deferred to v1.1 per ADR-192 D-9) at {span}"
    )]
    WriteInCallSubqueryNotSupported {
        /// Span of the offending `CALL` clause.
        span: Span,
    },

    /// A variable already bound as one kind (node / relationship / path /
    /// value) is re-used as a DIFFERENT kind. openCypher v9 §2 requires a
    /// variable to keep a single type across a query: a relationship
    /// variable cannot later appear as a node, a path variable cannot
    /// appear as a node/relationship, and a value-bound variable
    /// (`WITH 1 AS n`) cannot be matched as a node/relationship/path.
    /// Maps to the openCypher TCK error detail `VariableTypeConflict`
    /// (`clauses/match/Match1.feature` `7`/`8`/`9`/`10`/`11`;
    /// `Match2.feature` `9`/`10`/`11`/`12`/`13`; `Match3.feature` `30`)
    /// and `VariableAlreadyBound` (`Match6.feature` `21`-`25`, where the
    /// re-used variable is a named-PATH variable). Raised at COMPILE
    /// (bind) time. #618 GA Lane BINDER-VALIDATIONS.
    #[error(
        "variable `{name}` at {span} is already bound as {prior_kind} (prior at {prior_span}) and cannot be re-used as {new_kind}"
    )]
    VariableTypeConflict {
        /// The conflicting variable name.
        name: String,
        /// Human-readable kind the variable is now used as
        /// (`node` / `relationship` / `path`).
        new_kind: &'static str,
        /// Human-readable kind the variable was originally bound as.
        prior_kind: &'static str,
        /// The span of the conflicting re-use.
        span: Span,
        /// The span of the prior (original-kind) binding.
        prior_span: Span,
    },

    /// A single relationship variable appears MORE THAN ONCE in a
    /// pattern (e.g. `MATCH (a)-[r]->()-[r]->(a)`). openCypher v9 §2
    /// relationship-uniqueness forbids binding the same relationship
    /// variable twice (a relationship may not be traversed twice within
    /// one pattern). Distinct from [`Self::VariableTypeConflict`]: here
    /// BOTH uses are relationship positions. Maps to the openCypher TCK
    /// error detail `RelationshipUniquenessViolation`
    /// (`clauses/match/Match3.feature` `29`). Raised at COMPILE (bind)
    /// time. #618 GA Lane BINDER-VALIDATIONS.
    #[error(
        "relationship variable `{name}` at {span} is already bound (prior at {prior_span}); a relationship variable cannot be re-used"
    )]
    RelationshipUniquenessViolation {
        /// The re-used relationship variable name.
        name: String,
        /// The span of the offending second relationship binding.
        span: Span,
        /// The span of the first relationship binding.
        prior_span: Span,
    },

    /// A `RETURN` / `WITH` projection produces two result columns with
    /// the SAME name (e.g. `RETURN 1 AS a, 2 AS a`). openCypher v9 §6
    /// requires result-column names to be unique within a projection.
    /// Maps to the openCypher TCK error detail `ColumnNameConflict`
    /// (`clauses/return/Return4.feature` `10`). Raised at COMPILE (bind)
    /// time. #618 GA Lane BINDER-VALIDATIONS.
    #[error("result column name `{name}` is used more than once in a projection at {span}")]
    ColumnNameConflict {
        /// The duplicated result-column name.
        name: String,
        /// Span of the offending second column.
        span: Span,
    },

    /// `RETURN *` (or `WITH *`) appears with NO variables in scope to
    /// expand (e.g. `MATCH () RETURN *` — the anonymous node binds no
    /// variable). openCypher v9 §6 requires `*` to have at least one
    /// in-scope variable to project. Maps to the openCypher TCK error
    /// detail `NoVariablesInScope` (`clauses/return/Return7.feature`
    /// `2`). Raised at COMPILE (bind) time. #618 GA Lane
    /// BINDER-VALIDATIONS.
    #[error("RETURN * / WITH * has no variables in scope to project at {span}")]
    NoVariablesInScope {
        /// Span of the `*` projection.
        span: Span,
    },

    /// A `WITH` projection contains a non-trivial expression with no
    /// alias (e.g. `WITH a, count(*)` — the `count(*)` term needs an
    /// `AS <name>`). openCypher v9 §6 requires every `WITH` projection
    /// term that is not a bare variable reference to be aliased (a
    /// `WITH` fence must name every output column). Maps to the
    /// openCypher TCK error detail `NoExpressionAlias`
    /// (`clauses/with/With4.feature` `5`). Raised at COMPILE (bind)
    /// time. #618 GA Lane BINDER-VALIDATIONS.
    #[error("WITH projection expression must be aliased (add `AS <name>`) at {span}")]
    NoExpressionAlias {
        /// Span of the unaliased projection expression.
        span: Span,
    },

    /// An aggregating function (`count`/`sum`/`avg`/`min`/`max`/
    /// `collect`) appears in a position where aggregation is not
    /// permitted: a `WHERE` predicate, an `ORDER BY` sort key, or a list
    /// comprehension `| projection`. openCypher v9 §6.4 confines
    /// aggregation to `RETURN` / `WITH` projection terms (the implicit
    /// GROUP BY); a `WHERE count(a) > 10` or `ORDER BY max(x)` is a
    /// static error. Maps to the openCypher TCK error detail
    /// `InvalidAggregation` (`clauses/match-where/MatchWhere1.feature`
    /// `15`; `clauses/return-orderby/ReturnOrderBy2.feature` `14`;
    /// `clauses/with-orderBy/WithOrderBy2.feature` `25`;
    /// `expressions/list/List12.feature` `7`). Raised at COMPILE (bind)
    /// time. #618 GA Lane BINDER-VALIDATIONS.
    #[error("aggregation is not allowed in {position} at {span}")]
    InvalidAggregation {
        /// The illegal position (`WHERE` / `ORDER BY` / `list
        /// comprehension`).
        position: &'static str,
        /// Span of the offending clause/expression.
        span: Span,
    },

    /// A `RETURN` / `WITH` projection that CONTAINS an aggregating function
    /// also references a variable/property OUTSIDE the aggregate that is not
    /// an implicit grouping key (a bare non-aggregating projection). openCypher
    /// v9 §6.4 — the non-aggregating projections form the implicit GROUP BY,
    /// and a non-aggregated term inside an aggregating expression is ambiguous
    /// unless it is itself a (simple) grouping key. `RETURN me.age +
    /// count(you.age)` (no grouping key) and `RETURN a+b, a+b+count(*)` (the
    /// complex grouping key `a+b` must be aliased + referenced, not recomputed)
    /// both raise this. Maps to the openCypher TCK error detail
    /// `AmbiguousAggregationExpression` (`clauses/return/Return6.feature`
    /// `20`/`21`; `clauses/with/With6.feature` `8`/`9`). Raised at COMPILE
    /// (bind) time. ADR-038 amendment-12 (#796 companion — the permissive-
    /// binding lane unmasks these previously `UnknownLabel`-masked cases).
    #[error(
        "ambiguous aggregation expression at {span}: a non-aggregated term is not a grouping key"
    )]
    AmbiguousAggregationExpression {
        /// Span of the offending RETURN/WITH clause.
        span: Span,
    },

    /// An aggregating function is nested directly inside another
    /// aggregating function (e.g. `count(count(*))`). openCypher v9
    /// §6.4 forbids aggregation of an aggregation. Maps to the
    /// openCypher TCK error detail `NestedAggregation`
    /// (`clauses/return/Return6.feature` `14`). Raised at COMPILE
    /// (bind) time. #618 GA Lane BINDER-VALIDATIONS.
    #[error("aggregation function cannot be nested inside another aggregation at {span}")]
    NestedAggregation {
        /// Span of the offending nested aggregation.
        span: Span,
    },

    /// A floating-point literal is too large to represent — it
    /// overflowed to infinity at parse time (e.g. `1.34E999`).
    /// openCypher v9 §3 rejects an out-of-`f64`-range float literal at
    /// compile time. Maps to the openCypher TCK error detail
    /// `FloatingPointOverflow` (`expressions/literals/Literals5.feature`
    /// `27`). Raised at COMPILE (bind) time. #618 GA Lane
    /// BINDER-VALIDATIONS.
    #[error("floating-point literal is out of range (overflowed to infinity) at {span}")]
    FloatingPointOverflow {
        /// Span of the offending float literal (best-effort cursor
        /// position — literal re-tokenization is fragile at v1.0).
        span: Span,
    },

    /// A `SKIP` / `LIMIT` expression is NOT a constant — it references a
    /// bound variable (e.g. `SKIP n.count`). openCypher v9 §6.4 requires
    /// `SKIP` / `LIMIT` to be a constant expression (a literal or
    /// parameter) so the row window is known before evaluation. Maps to
    /// the openCypher TCK error detail `NonConstantExpression`
    /// (`clauses/return-skip-limit/ReturnSkipLimit1.feature` `5`/`10`;
    /// `ReturnSkipLimit2.feature` `9`). This REPLACES the prior
    /// `NotImplemented` for the dynamic case (WrongErrorPhase →
    /// compile-time). Raised at COMPILE (bind) time. #618.
    #[error("{clause} expression must be a constant (no variable references) at {span}")]
    NonConstantExpression {
        /// `SKIP` or `LIMIT`.
        clause: &'static str,
        /// Span of the offending clause.
        span: Span,
    },

    /// A `SKIP` / `LIMIT` integer literal is negative (e.g. `SKIP -1`).
    /// openCypher v9 §6.4 requires a non-negative row count. Maps to the
    /// openCypher TCK error detail `NegativeIntegerArgument`
    /// (`clauses/return-skip-limit/ReturnSkipLimit1.feature` `11`;
    /// `ReturnSkipLimit2.feature` `12`). Raised at COMPILE (bind) time.
    /// #618.
    #[error("{clause} value must be non-negative; got {value} at {span}")]
    NegativeIntegerArgument {
        /// `SKIP` or `LIMIT`.
        clause: &'static str,
        /// The offending negative value.
        value: i64,
        /// Span of the offending clause.
        span: Span,
    },

    /// A `SKIP` / `LIMIT` constant is not an integer (e.g. `LIMIT 1.7`, a
    /// float; or a string / boolean / list). openCypher v9 §6.4 requires
    /// an INTEGER row count. Maps to the openCypher TCK error detail
    /// `InvalidArgumentType` (`clauses/return-skip-limit/ReturnSkipLimit2.feature`
    /// `16` `LIMIT 1.7`). Raised at COMPILE (bind) time. #618.
    #[error("{clause} value must be an integer; got {actual} at {span}")]
    NonIntegerSkipLimit {
        /// `SKIP` or `LIMIT`.
        clause: &'static str,
        /// Human-readable description of the offending value's type.
        actual: &'static str,
        /// Span of the offending clause.
        span: Span,
    },
}

impl BindingError {
    /// Return the carried primary [`Span`].
    ///
    /// Every variant has a primary span (the offending token);
    /// secondary spans (e.g. `DuplicateBinding::prior_span`) are
    /// accessed via direct match on the variant.
    pub fn span(&self) -> &Span {
        match self {
            BindingError::UndeclaredVariable { span, .. }
            | BindingError::UnknownLabel { span, .. }
            | BindingError::UnknownRelType { span, .. }
            | BindingError::UnknownProcedure { span, .. }
            | BindingError::InvalidYieldColumn { span, .. }
            | BindingError::DuplicateBinding { span, .. }
            | BindingError::ScopeViolation { span, .. }
            | BindingError::UnionColumnMismatch { span, .. }
            | BindingError::UnionMixedSetOps { span, .. }
            | BindingError::WriteInCallSubqueryNotSupported { span, .. }
            | BindingError::VariableTypeConflict { span, .. }
            | BindingError::RelationshipUniquenessViolation { span, .. }
            | BindingError::ColumnNameConflict { span, .. }
            | BindingError::NoVariablesInScope { span, .. }
            | BindingError::NoExpressionAlias { span, .. }
            | BindingError::InvalidAggregation { span, .. }
            | BindingError::AmbiguousAggregationExpression { span, .. }
            | BindingError::NestedAggregation { span, .. }
            | BindingError::FloatingPointOverflow { span, .. }
            | BindingError::NonConstantExpression { span, .. }
            | BindingError::NegativeIntegerArgument { span, .. }
            | BindingError::NonIntegerSkipLimit { span, .. } => span,
        }
    }

    /// Translate the primary span (line:col coordinates) into a
    /// byte-offset range in the original input string.
    ///
    /// Returns `None` only on coordinate-system mismatch (defensive;
    /// should not happen for spans produced by `BindingVisitor`).
    /// Mirrors [`crate::error::ParseError::span_byte_range`].
    pub fn span_byte_range(&self, input: &str) -> Option<(usize, usize)> {
        let span = self.span();
        let start = line_col_to_byte(input, span.start_line, span.start_col)?;
        let end = line_col_to_byte(input, span.end_line, span.end_col)?;
        Some((start, end))
    }
}

/// Convert a 1-indexed `(line, col)` coordinate into a byte offset
/// into `input`. Clamps off-the-end lines/columns to `input.len()`.
///
/// **Defensive duplicate** of `crate::error::line_col_to_byte`.
/// `crate::error` is FROZEN by the M4-21 contract (PR #154 surface);
/// duplicating the 12-line helper avoids modifying it. If a future
/// slice unfreezes `crate::error`, fold this back to a `pub(crate)`
/// re-export.
fn line_col_to_byte(input: &str, line: usize, col: usize) -> Option<usize> {
    if line == 0 || col == 0 {
        return None;
    }
    let bytes = input.as_bytes();
    let mut current_line = 1usize;
    let mut line_start = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        if current_line == line {
            let offset = line_start + (col - 1);
            return Some(offset.min(bytes.len()));
        }
        if *b == b'\n' {
            current_line += 1;
            line_start = i + 1;
        }
    }
    if current_line == line {
        let offset = line_start + (col - 1);
        return Some(offset.min(bytes.len()));
    }
    Some(bytes.len())
}

// =====================================================================
// TypeCheckError (M4-22 — D-22)
// =====================================================================

/// Faults surfaced by [`crate::semantic::TypeCheckVisitor::check`].
///
/// Each variant carries a `span` pointing at the offending token in
/// the original input. Mirrors [`BindingError`]'s shape; both convert
/// into [`ArcQLError`] via `From`.
///
/// `#[non_exhaustive]` is **omitted** on the same rationale as
/// [`BindingError`]: the variant set is the type-check pass's public
/// contract for M4-23 (substrate validation) consumption. New
/// variants land via amendment alongside future type-check
/// extensions.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TypeCheckError {
    /// Operand types are incompatible with the operator. E.g.,
    /// `n.age > "thirty"` (Integer vs. String under `>`).
    #[error("type mismatch: cannot apply `{op}` to {lhs:?} and {rhs:?} at {span}")]
    TypeMismatch {
        op: String,
        lhs: TypeInfo,
        rhs: TypeInfo,
        span: Span,
    },

    /// A function call's argument count does not match its registry
    /// signature. `expected` is rendered as either `"N"` (exact arity)
    /// or `"N+"` (variadic minimum).
    #[error("function `{name}` called with {actual} args but expects {expected} at {span}")]
    FunctionArityMismatch {
        name: String,
        actual: usize,
        expected: String,
        span: Span,
    },

    /// A function call's argument has the wrong type at `position`
    /// (0-indexed). E.g., `length(42)` passes Integer where the
    /// signature expects List.
    #[error(
        "function `{name}` argument {position} expects {expected:?} but got {actual:?} at {span}"
    )]
    FunctionArgumentTypeMismatch {
        name: String,
        position: usize,
        expected: TypeInfo,
        actual: TypeInfo,
        span: Span,
    },

    /// The function name does not exist in the registry +
    /// ArcGraph extensions.
    #[error("unknown function `{name}` at {span}")]
    UnknownFunction { name: String, span: Span },

    /// A property access references a property name that the catalog
    /// does not associate with the variable's resolved label. v1.0
    /// dynamic-schema fallback admits unknown properties; this variant
    /// is reserved for v1.1+ strict-schema rejection.
    #[error("unknown property `{name}` on label `{label}` at {span}")]
    UnknownProperty {
        name: String,
        label: String,
        span: Span,
    },

    /// `WHERE expr` resolves to a non-Boolean (and non-Null) type.
    /// Cypher 3VL admits Null in WHERE position (treated as FALSE);
    /// any other non-Boolean type is a type error.
    #[error("WHERE filter must be Boolean or Null, got {actual:?} at {span}")]
    NonBooleanWhere { actual: TypeInfo, span: Span },

    /// **ADR-147 W26-θ Phase 1 + amendment-03 (D-1).** A property value
    /// inside a `CREATE (... { name: <expr> })` clause is not admissible.
    /// Amendment-03 (Phase 1.5) lifts the literal-only narrowing on
    /// CREATE: property values now admit literals, `$param`,
    /// previously-bound row references, and a bounded / deterministic
    /// expression subset over admissible operands. Still rejected:
    /// `FunctionCall` (determinism / unbounded materialization), map /
    /// entity shapes (openCypher forbids map property values, ADR-191
    /// D-11), and other non-evaluable forms. MERGE pattern property
    /// values remain literal-only (they reuse this variant); SET / REMOVE
    /// use `SetPropertyValueNotLiteral`. The message keeps the word
    /// "literal" — `bolt_param_binding_e2e.rs` asserts on it.
    #[error(
        "CREATE property `{name}` value must be a literal, parameter, or a bounded expression over bound rows/params (Phase 1.5 per ADR-147-amendment-03); got {actual} at {span}"
    )]
    CreatePropertyValueNotLiteral {
        name: String,
        actual: String,
        span: Span,
    },

    /// **ADR-149 W26-θ Phase 3.** A `DELETE` clause item resolves to
    /// a non-graph-typed binding. Phase 3 admits Node-typed and
    /// Relationship-typed bindings only (per ADR-149 §D-4); any
    /// other type (Integer / String / Map / Boolean / Path / etc.)
    /// is a type error.
    #[error(
        "DELETE item `{name}` must be a Node or Relationship binding (Phase 3 per ADR-149); got {actual:?} at {span}"
    )]
    DeleteNonGraphValue {
        name: String,
        actual: TypeInfo,
        span: Span,
    },

    /// **ADR-150 W26-θ Phase 4.** A `SET` or `REMOVE` clause item
    /// resolves to a non-graph-typed binding. Phase 4 admits Node-
    /// typed and Relationship-typed bindings only (per ADR-150 §D-4);
    /// any other type (Integer / String / Map / Boolean / Path /
    /// etc.) is a type error.
    #[error(
        "SET/REMOVE item `{name}` must be a Node or Relationship binding (Phase 4 per ADR-150); got {actual:?} at {span}"
    )]
    SetRemoveNonGraphValue {
        name: String,
        actual: TypeInfo,
        span: Span,
    },

    /// **ADR-150 W26-θ Phase 4.** A `SET n:Label` / `REMOVE n:Label`
    /// item targets a Relationship-typed binding. Per openCypher v9
    /// §6 + ADR-150 §D-4, label mutations apply only to Node-typed
    /// bindings; Relationship-typed bindings reject at type-check.
    #[error(
        "SET/REMOVE label mutation on `{name}` requires a Node binding (Phase 4 per ADR-150); rels do not carry labels at {span}"
    )]
    SetRemoveLabelOnRel { name: String, span: Span },

    /// **ADR-150 W26-θ Phase 4.** A property value inside a
    /// `SET n.prop = <expr>` / `SET n = {prop: <expr>}` /
    /// `SET n += {prop: <expr>}` clause is not a literal. Phase 4
    /// restricts SET property values to literals (Integer / Float /
    /// String / Bool / Null) per the Phase 1 (ADR-147 §D-4) inherited
    /// narrowing; parameter / expression-typed property values
    /// forward-pin to v1.1 per ADR-150 §"Forward-deferred".
    #[error(
        "SET property `{name}` value must be a literal at v1.0-α (Phase 4 per ADR-150); got {actual} at {span}"
    )]
    SetPropertyValueNotLiteral {
        name: String,
        actual: String,
        span: Span,
    },

    /// **#618 GA Lane BINDER-VALIDATIONS.** A property access
    /// (`base.prop`) targets a `base` whose static type is a concrete
    /// NON-graph-element, non-map value (e.g. `WITH 123 AS n RETURN
    /// n.num` — `n` is an Integer). openCypher v9 §3 admits property
    /// access only on a Node / Relationship / Map (plus the
    /// dynamically-typed `Property` / `Null` escapes the v1.0 catalog
    /// under-types); a concrete scalar / list / path base is an
    /// `InvalidArgumentType` at COMPILE time. Maps to the openCypher TCK
    /// error detail `InvalidArgumentType` (`expressions/graph/Graph6.feature`
    /// `9` non-graph-element; `expressions/map/Map1.feature` `6` non-map).
    /// This REPLACES the prior runtime `property access on non-entity
    /// value` eval error for the statically-known case (WrongErrorPhase →
    /// compile-time). Raised at COMPILE (type-check) time.
    #[error("property access requires a node, relationship, or map; got {actual:?} at {span}")]
    PropertyAccessOnNonEntity { actual: TypeInfo, span: Span },

    /// **#773 G5.** A `DISTINCT` modifier (`fn(DISTINCT x)`) was applied
    /// to a non-aggregating function. Per openCypher v9 §3, `DISTINCT`
    /// is only valid inside an aggregating function invocation
    /// (`count` / `sum` / `avg` / `min` / `max` / `collect`); any other
    /// function (e.g. `size(DISTINCT x)`) rejects at type-check rather
    /// than silently discarding the modifier. (`count(*)` star misuse —
    /// e.g. `sum(*)` — is rejected by the ordinary arity check, since
    /// the star form supplies zero expression arguments.)
    #[error(
        "DISTINCT is only valid inside an aggregating function (count/sum/avg/min/max/collect); `{name}` does not support it at {span}"
    )]
    DistinctNotAllowed { name: String, span: Span },
}

impl TypeCheckError {
    /// Return the carried primary [`Span`].
    pub fn span(&self) -> &Span {
        match self {
            TypeCheckError::TypeMismatch { span, .. }
            | TypeCheckError::FunctionArityMismatch { span, .. }
            | TypeCheckError::FunctionArgumentTypeMismatch { span, .. }
            | TypeCheckError::UnknownFunction { span, .. }
            | TypeCheckError::UnknownProperty { span, .. }
            | TypeCheckError::NonBooleanWhere { span, .. }
            | TypeCheckError::CreatePropertyValueNotLiteral { span, .. }
            | TypeCheckError::DeleteNonGraphValue { span, .. }
            | TypeCheckError::SetRemoveNonGraphValue { span, .. }
            | TypeCheckError::SetRemoveLabelOnRel { span, .. }
            | TypeCheckError::SetPropertyValueNotLiteral { span, .. }
            | TypeCheckError::PropertyAccessOnNonEntity { span, .. }
            | TypeCheckError::DistinctNotAllowed { span, .. } => span,
        }
    }

    /// Translate the primary span into a byte-offset range in
    /// `input`. Mirrors [`BindingError::span_byte_range`].
    pub fn span_byte_range(&self, input: &str) -> Option<(usize, usize)> {
        let span = self.span();
        let start = line_col_to_byte(input, span.start_line, span.start_col)?;
        let end = line_col_to_byte(input, span.end_line, span.end_col)?;
        Some((start, end))
    }
}

// =====================================================================
// CrossSubstrateError (M4-23 — D-23)
// =====================================================================

/// Substrate kind identifier carried in
/// [`CrossSubstrateError::SubstrateUnavailable`]. Mirrors the per-tenant
/// substrate-availability predicates on [`crate::semantic::CatalogProvider`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateKind {
    /// Vector substrate (HNSW). Required by `<expr> NEAR <expr>`,
    /// `vector_distance(...)`, and the `VECTOR(...)` operand of
    /// `RANK BY HYBRID`.
    Vector,
    /// BM25 (text-search) substrate (Tantivy). Required by
    /// `<expr> MATCH <expr>`, `text_match(...)`, and the `TEXT(...)`
    /// operand of `RANK BY HYBRID`.
    Bm25,
    /// Community-detection substrate. Required by
    /// `<expr> IN COMMUNITY(<expr>)` and the `community(...)`
    /// function family.
    Community,
}

impl std::fmt::Display for SubstrateKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubstrateKind::Vector => f.write_str("vector"),
            SubstrateKind::Bm25 => f.write_str("bm25"),
            SubstrateKind::Community => f.write_str("community"),
        }
    }
}

/// Faults surfaced by
/// [`crate::semantic::cross_substrate::CrossSubstrateValidator::validate`].
///
/// Each variant carries a `span` pointing at the offending token in the
/// original input. Mirrors [`BindingError`] / [`TypeCheckError`] shape;
/// converts into [`ArcQLError`] via `From`.
///
/// `#[non_exhaustive]` is **omitted** on the same rationale as the
/// other M4-2x error taxonomies: the variant set is M4-23's public
/// contract for downstream M4-31 (logical plan generator) consumption.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CrossSubstrateError {
    /// A surface that requires a substrate (VECTOR / TEXT / community)
    /// was used but the per-tenant catalog reports the substrate is
    /// not attached.
    #[error("substrate `{kind}` not attached to tenant {tenant:?} at {span}")]
    SubstrateUnavailable {
        kind: SubstrateKind,
        tenant: TenantId,
        span: Span,
    },

    /// `RANK BY HYBRID(...)` is missing a required operand kind. v1.0
    /// requires exactly one VECTOR(...) operand AND exactly one
    /// TEXT(...) operand (per ADR-038 §2 D-3 hybrid surface). The
    /// `kind` slot names the missing operand kind ("VECTOR" or
    /// "TEXT"); the `span` points at the offending RANK BY clause.
    #[error("RANK BY HYBRID requires {kind}(...) operand at {span}")]
    HybridMissingOperand { kind: &'static str, span: Span },

    /// A `RANK BY HYBRID` operand (`VECTOR(field, query)` or
    /// `TEXT(field, query)`) is missing its required `K = N` parameter.
    /// Each substrate operand needs an explicit K to bound retrieval
    /// fan-out (no implicit defaults at v1.0; ADR-038 §2 D-3 + D-9).
    #[error("RANK BY HYBRID operand requires K parameter at {span}")]
    HybridMissingK { span: Span },

    /// `WITH FUSION = RRF(...)` is missing the required `k = N`
    /// parameter. The unweighted RRF form is the only ratified
    /// fusion at v1.0; it requires an explicit `k` (no default —
    /// per ADR-038 §2 D-9).
    #[error("WITH FUSION = RRF requires k parameter at {span}")]
    FusionMissingK { span: Span },
}

impl CrossSubstrateError {
    /// Return the carried primary [`Span`].
    pub fn span(&self) -> &Span {
        match self {
            CrossSubstrateError::SubstrateUnavailable { span, .. }
            | CrossSubstrateError::HybridMissingOperand { span, .. }
            | CrossSubstrateError::HybridMissingK { span, .. }
            | CrossSubstrateError::FusionMissingK { span, .. } => span,
        }
    }

    /// Translate the primary span into a byte-offset range in
    /// `input`. Mirrors [`BindingError::span_byte_range`].
    pub fn span_byte_range(&self, input: &str) -> Option<(usize, usize)> {
        let span = self.span();
        let start = line_col_to_byte(input, span.start_line, span.start_col)?;
        let end = line_col_to_byte(input, span.end_line, span.end_col)?;
        Some((start, end))
    }
}

// =====================================================================
// ArcQLError (M4-22 umbrella)
// =====================================================================

/// Public-API error umbrella for the semantic analyzer.
///
/// Encompasses [`BindingError`] (M4-21), [`TypeCheckError`] (M4-22),
/// [`CrossSubstrateError`] (M4-23), [`LogicalPlanError`] (M4-31), and
/// the `NotImplemented` variant for reserved-but-unimplemented clauses
/// per ADR-038 §2 D-16.
///
/// `#[non_exhaustive]` is **omitted** on the same rationale as its
/// constituent M4-2x error taxonomies (`BindingError`,
/// `TypeCheckError`, `CrossSubstrateError`, `LogicalPlanError`,
/// `ParseError`): the variant set is the ArcQL public umbrella
/// contract that downstream M5 / M6 entry points (Bolt, MCP, gRPC)
/// pattern-match exhaustively for transport-layer error mapping per
/// ADR-038 §2 D-16. Adding a sibling variant is a SemVer event.
///
/// # W12α addition: `ResourceExhausted`
///
/// W12α (M4-64a per amendment-03 §Structural-1) adds the
/// [`ArcQLError::ResourceExhausted`] variant to surface per-tenant
/// memory-budget exhaustion at execute time. M5-12 rate-limit config
/// will additionally surface budget rejection at plan-time when an
/// estimated query cost exceeds the configured cap. The variant is the
/// shared error shape across both surfaces.
///
/// Per the M5/M6 transport-layer renderer contract above, the addition
/// is a deliberate SemVer event: M5-07 / M5-11 / M5-13 future renderers
/// MUST add a `ResourceExhausted` arm (parallel to the existing
/// `NotImplemented` arm) when they ship. v1.0-alpha has no in-tree
/// transport-layer pattern match on `ArcQLError` outside of the
/// `crate::explain::translate_execution_error` site (which forwards via
/// `ExplainError::ArcQL(_)` — a transparent passthrough), so the
/// addition is non-breaking at HEAD.
///
/// # W13β fix-up addition: `Internal`
///
/// PR #287 review M-1 + N-1 add the [`ArcQLError::Internal`] variant
/// for client-side / lifecycle-invariant violations the executor
/// detects at runtime — e.g. a `StreamingCursor::open` on an
/// `ExecutionContext` whose snapshot LSN was already released by a
/// prior cursor (close-then-reopen, ADR-038 amendment-03 §TIER-1 GAP E
/// rule 5), or a `next_batch` call on a closed cursor (M4-82 lifecycle
/// invariant). Distinct from `NotImplemented` ("deferred-feature
/// semantic") and `ResourceExhausted` ("capacity overrun") — the
/// renderer surfaces this as a "client misuse: `<feature>` — `<reason>`"
/// diagnostic.
///
/// Per the M5/M6 transport-layer renderer contract above, this
/// addition is a deliberate SemVer event: M5-07 / M5-11 / M5-13 future
/// renderers MUST add an `Internal` arm when they ship. v1.0-alpha has
/// the same transparent-passthrough posture as for `ResourceExhausted`
/// — the addition is non-breaking at HEAD.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ArcQLError {
    #[error("binding error: {0}")]
    Binding(#[from] BindingError),

    #[error("type-check error: {0}")]
    TypeCheck(#[from] TypeCheckError),

    #[error("cross-substrate error: {0}")]
    CrossSubstrate(#[from] CrossSubstrateError),

    /// M4-31 logical-plan-lowering fault. See
    /// [`crate::logical_plan::error::LogicalPlanError`] for the
    /// variant taxonomy.
    #[error("logical plan error: {0}")]
    LogicalPlan(#[from] LogicalPlanError),

    /// Reserved-but-unimplemented clause / variant per ADR-038 §2
    /// D-16. The `feature` slot names the construct
    /// (for example, an unsupported clause); the `section` slot cites
    /// the design section that owns it;
    /// the `target_version` slot states when the variant lights
    /// (e.g., `"v1.1"`); the `span` points at the offending token.
    #[error(
        "not implemented: {feature} (reserved per ADR-038 {section}, planned {target_version}) at {span}"
    )]
    NotImplemented {
        feature: String,
        section: String,
        target_version: String,
        span: Span,
    },

    /// Per-tenant memory budget exhausted (W12α / M4-64a per
    /// amendment-03 §Structural-1). Surfaces when an operator's
    /// per-batch reservation would push the tenant's total above the
    /// configured `cap_bytes`. M5-12 rate-limit config will additionally
    /// surface this at plan-time when an estimated query cost exceeds
    /// the configured cap.
    ///
    /// `feature` names the operator / surface that triggered exhaustion
    /// (e.g., `"ExpandOp spillover"`). The byte slots carry the actual
    /// numbers the renderer surfaces to the caller. `span` is the
    /// `Span::point(0, 0)` sentinel at execute-time (no source-text
    /// span maps to a runtime budget overflow); plan-time surfaces from
    /// M5-12 will carry the offending construct's span.
    #[error(
        "resource exhausted: {feature} would reserve {requested_bytes} bytes \
         pushing tenant total to {projected_bytes} bytes \
         past per-tenant cap {cap_bytes} bytes"
    )]
    ResourceExhausted {
        feature: String,
        requested_bytes: u64,
        cap_bytes: u64,
        projected_bytes: u64,
        span: Span,
    },

    /// W13β fix-up — client-side / lifecycle-invariant violation
    /// detected at runtime (PR #287 review M-1 + N-1).
    ///
    /// Distinct from [`Self::NotImplemented`] (which carries the
    /// "deferred-feature" semantic for reserved-but-unimplemented
    /// constructs per ADR-038 §2 D-16) and [`Self::ResourceExhausted`]
    /// (capacity overrun). `Internal` covers cases where the executor
    /// detects misuse at runtime that the planner cannot statically
    /// reject:
    ///
    /// - **`StreamingCursor::open` on a consumed
    ///   `ExecutionContext`** — close-then-reopen would re-acquire
    ///   a fresh snapshot LSN, violating ADR-038 amendment-03
    ///   §TIER-1 GAP E rule 5. The fix-up REJECTS rather than
    ///   silently re-acquiring.
    /// - **`StreamingCursor::next_batch` on a closed cursor** —
    ///   M4-82 lifecycle invariant. Was previously surfaced as
    ///   `NotImplemented` (wrong taxonomy — review NIT-1).
    ///
    /// `feature` names the surface that detected the violation
    /// (e.g., `"StreamingCursor::open"` /
    /// `"StreamingCursor::next_batch"`); `reason` is a human-readable
    /// description of the invariant violated; `span` is
    /// [`Span::point(0, 0)`] at v1.0-alpha (no source-text span maps
    /// to a runtime lifecycle violation).
    #[error("internal: {feature} — {reason}")]
    Internal {
        feature: String,
        reason: String,
        span: Span,
    },
}

impl ArcQLError {
    /// Return the carried primary [`Span`].
    pub fn span(&self) -> &Span {
        match self {
            ArcQLError::Binding(e) => e.span(),
            ArcQLError::TypeCheck(e) => e.span(),
            ArcQLError::CrossSubstrate(e) => e.span(),
            ArcQLError::LogicalPlan(e) => e.span(),
            ArcQLError::NotImplemented { span, .. }
            | ArcQLError::ResourceExhausted { span, .. }
            | ArcQLError::Internal { span, .. } => span,
        }
    }

    /// Translate the primary span into a byte-offset range in
    /// `input`. Mirrors [`BindingError::span_byte_range`].
    pub fn span_byte_range(&self, input: &str) -> Option<(usize, usize)> {
        match self {
            ArcQLError::Binding(e) => e.span_byte_range(input),
            ArcQLError::TypeCheck(e) => e.span_byte_range(input),
            ArcQLError::CrossSubstrate(e) => e.span_byte_range(input),
            ArcQLError::LogicalPlan(e) => e.span_byte_range(input),
            ArcQLError::NotImplemented { span, .. }
            | ArcQLError::ResourceExhausted { span, .. }
            | ArcQLError::Internal { span, .. } => {
                let start = line_col_to_byte(input, span.start_line, span.start_col)?;
                let end = line_col_to_byte(input, span.end_line, span.end_col)?;
                Some((start, end))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_accessor_returns_primary_span() {
        let e = BindingError::UnknownLabel {
            name: "Foo".into(),
            span: Span::point(1, 9),
        };
        assert_eq!(e.span(), &Span::point(1, 9));
    }

    #[test]
    fn span_byte_range_translates_single_line() {
        let input = "MATCH (n:Foo) RETURN n";
        let e = BindingError::UnknownLabel {
            name: "Foo".into(),
            // col 10 is the `F` in `Foo` (1-indexed).
            span: Span {
                start_line: 1,
                start_col: 10,
                end_line: 1,
                end_col: 13,
            },
        };
        let (s, eoff) = e.span_byte_range(input).expect("translation");
        assert_eq!(&input[s..eoff], "Foo");
    }

    #[test]
    fn duplicate_binding_carries_prior_span() {
        let e = BindingError::DuplicateBinding {
            name: "n".into(),
            span: Span::point(2, 8),
            prior_span: Span::point(1, 8),
            reason: String::new(),
        };
        let s = format!("{e}");
        assert!(s.contains("duplicate binding"));
        assert!(s.contains("`n`"));
        assert!(s.contains("2:8"));
        assert!(s.contains("1:8"));
    }

    #[test]
    fn equality_is_structural() {
        let a = BindingError::UndeclaredVariable {
            name: "x".into(),
            span: Span::point(1, 1),
        };
        let b = BindingError::UndeclaredVariable {
            name: "x".into(),
            span: Span::point(1, 1),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn implements_std_error_trait() {
        fn assert_impls_error<E: std::error::Error>(_: E) {}
        assert_impls_error(BindingError::UndeclaredVariable {
            name: "x".into(),
            span: Span::point(1, 1),
        });
    }

    // ----- CrossSubstrateError (M4-23) -----

    #[test]
    fn substrate_kind_display_names_match_catalog_predicates() {
        assert_eq!(SubstrateKind::Vector.to_string(), "vector");
        assert_eq!(SubstrateKind::Bm25.to_string(), "bm25");
        assert_eq!(SubstrateKind::Community.to_string(), "community");
    }

    #[test]
    fn cross_substrate_error_span_accessor_returns_primary_span() {
        let e = CrossSubstrateError::SubstrateUnavailable {
            kind: SubstrateKind::Vector,
            tenant: TenantId::DEFAULT,
            span: Span::point(2, 14),
        };
        assert_eq!(e.span(), &Span::point(2, 14));
    }

    #[test]
    fn cross_substrate_error_lifts_into_arcql_error() {
        let inner = CrossSubstrateError::FusionMissingK {
            span: Span::point(3, 5),
        };
        let lifted: ArcQLError = inner.clone().into();
        match lifted {
            ArcQLError::CrossSubstrate(c) => assert_eq!(c, inner),
            other => panic!("expected ArcQLError::CrossSubstrate, got {other:?}"),
        }
    }
}
