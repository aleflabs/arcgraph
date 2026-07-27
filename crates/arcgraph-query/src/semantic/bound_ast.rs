//! Typed AST after the binding + type-check passes.
//!
//! `BoundAst` is a **parallel** structure to the syntactic AST in
//! [`crate::ast`] — not an in-place modification. Each AST node has
//! a `Bound*` mirror that adds:
//!
//! - a [`Span`] field pointing into the original source (the AST
//!   does NOT carry spans by design; the M4-01 100 ratified tests
//!   pin that contract — see PR #154 reviewer ask #7);
//! - a [`BindingId`] for nodes that introduce a binding;
//! - a resolved schema-id (`Option<LabelId>` / `Option<TypeId>` /
//!   `Option<PropertyId>`) for nodes that reference the catalog;
//! - a [`TypeInfo`] payload (M4-22 onwards): `None` after the
//!   binding pass, `Some(...)` after type-check;
//! - the OPTIONAL-MATCH `may_be_null` / `is_optional` flags per
//!   ADR-006 amendment-01.
//!
//! # ADR provenance
//! - ADR-038 §2 D-1 — openCypher binding semantics.
//! - ADR-038 §2 D-21 — variable-binding scopes (M4-21).
//! - ADR-038 §2 D-22 — type-checking + reserved-variant rejection
//!   (M4-22).
//! - ADR-038 amendment-03 §TIER-1 GAP E — `BoundQuery::snapshot_lsn`
//!   field reservation for M4-61 execute-time handoff.
//! - ADR-006 amendment-01 — OPTIONAL MATCH at v1.0; M4-22 sets
//!   `BoundVariable::may_be_null = true` for variables introduced
//!   in OPTIONAL MATCH.

use arcgraph_core::{LabelId, Lsn, PartitionId, PropertyId, TenantId, TypeId};

use crate::ast::{
    BinOp, CreateRelDirection, LengthRange, Literal, OrderDirection, Quantifier, RelDirection,
    UnaryOp,
};
use crate::error::Span;

// =====================================================================
// 0. Identifiers
// =====================================================================

/// Identifier for a binding (a variable declaration site).
///
/// Two `BoundVariable`s with equal `binding_id` refer to the same
/// declaration. Variable references (in expressions, RETURN
/// projections) carry the `binding_id` of the declaration they
/// resolve to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BindingId(pub u64);

impl BindingId {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Identifier for a lexical-scope frame.
///
/// Each `MATCH`-chain (between two `WITH` clauses, or between query
/// start and the first `WITH`) introduces a single scope frame.
/// Each `WITH` opens a new frame containing only the projected
/// names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ScopeId(pub u32);

impl ScopeId {
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }
    pub const fn raw(self) -> u32 {
        self.0
    }
}

// =====================================================================
// 1. Top-level
// =====================================================================

/// A bound ArcQL statement.
///
/// Read queries are fully bound; index DDL passes through to lowering.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundStatement {
    /// A bound read query.
    Read(BoundQuery),
    /// Neo4j-compatible index DDL pass-through (#830, ADR-198 §OQ-7):
    /// `CREATE VECTOR INDEX …` / `DROP INDEX …`. Bound here (label /
    /// property captured), then rejected by
    /// [`crate::semantic::type_check::TypeCheckVisitor`] with
    /// `ArcQLError::NotImplemented` (parsed-but-not-built) — the index
    /// BUILD is the vector track's follow-up.
    IndexDdl(crate::ast::IndexDdlStatement),
    /// A bound `UNION` / `UNION ALL` set-operation query per ADR-185
    /// (#649-A1, W28 — openCypher v9 §8). Each arm is a fully-bound
    /// [`BoundQuery`]; the post-union tail binds against arm-0's
    /// terminal scope (all arms expose the same column names, so arm-0
    /// is representative). See [`BoundUnionQuery`].
    ///
    /// `Box`ed because [`BoundUnionQuery`] (a `Vec<BoundQuery>` + tail +
    /// per-arm permutation) is much larger than the `Read(BoundQuery)`
    /// variant; boxing keeps `BoundStatement` small (clippy
    /// `large_enum_variant`) without bloating the common read path.
    Union(Box<BoundUnionQuery>),
}

/// A bound `UNION` / `UNION ALL` query per ADR-185 (#649-A1, W28).
///
/// The column-compatibility rule (openCypher v9 §8 — every arm
/// projects the same column-name set) + the no-mixing rule are
/// enforced at bind time; if either fails the bind returns
/// [`crate::semantic::error::BindingError::UnionColumnMismatch`] /
/// [`crate::semantic::error::BindingError::UnionMixedSetOps`] and this
/// struct is never constructed for the failing input.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundUnionQuery {
    /// The bound union arms (≥2), left-to-right.
    pub arms: Vec<BoundQuery>,
    /// Per-boundary `ALL` flag; `all.len() == arms.len() - 1`. All
    /// elements are equal (the no-mixing rule); `all[0]` is the union
    /// kind (`true` = UNION ALL / keep dupes, `false` = UNION /
    /// distinct).
    pub all: Vec<bool>,
    /// Per-arm column permutation for openCypher v9 §8 ORDER-INDEPENDENT
    /// column alignment. `column_orders[i][j]` = the source position in
    /// arm `i`'s output row that supplies the union's canonical output
    /// column `j` (canonical order = arm 0's RETURN order).
    /// `column_orders[0]` is always the identity. The executor's
    /// `UnionOp` applies the permutation so two arms exposing the same
    /// column NAME set in a DIFFERENT order still concatenate
    /// correctly (the §8 "same columns, order-independent" rule applies
    /// to the RESULT, not only the compatibility check). Derived at
    /// bind time (where the AST column names are reliable).
    pub column_orders: Vec<Vec<usize>>,
    /// Post-union ORDER BY / SKIP / LIMIT, bound against arm-0's
    /// terminal scope and applied to the COMBINED result.
    pub tail: BoundUnionTail,
    /// Span covering the whole union.
    pub span: Span,
}

/// The bound post-union tail (ORDER BY / SKIP / LIMIT) per ADR-185.
/// An all-empty tail is the no-tail case.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BoundUnionTail {
    /// Bound ORDER BY items (empty when absent).
    pub order_by: Vec<BoundOrderItem>,
    /// Bound SKIP expression (when present).
    pub skip: Option<BoundExpression>,
    /// Bound LIMIT expression (when present).
    pub limit: Option<BoundExpression>,
}

/// A bound read query.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundQuery {
    /// Bound clauses in source order.
    pub clauses: Vec<BoundClause>,
    /// The root scope (entered before the first MATCH).
    pub root_scope: ScopeId,
    /// Span covering the entire query text.
    pub span: Span,
    /// The query's tenant identity (stamped from
    /// [`crate::semantic::CatalogProvider::tenant`] at bind time).
    pub tenant: TenantId,
    /// The query's partition identity (stamped from
    /// [`crate::semantic::CatalogProvider::partition`] at bind
    /// time). v1.0 invariant: [`PartitionId::ZERO`] per ADR-024
    /// amendment-02.
    pub partition: PartitionId,
    /// **Reserved for execute-time binding handoff.** Per ADR-038
    /// amendment-03 §TIER-1 GAP E (D-18), the snapshot LSN is
    /// acquired at execute-time, before the first operator pulls a
    /// batch. M4-21 always sets this to `None`; M4-61 (execution
    /// context + batch cursor) populates it pre-first-batch and
    /// holds until query-end.
    pub snapshot_lsn: Option<Lsn>,
}

// `BoundMatchClause` is the largest variant because it carries the
// pattern body (head + tail Vec) AND the WHERE expression AND span
// metadata. Boxing it would change the public API ergonomics for
// every downstream walker (M4-22 type-check, M4-23 substrate
// validation) — the pattern-match arm becomes
// `BoundClause::Match(boxed) => walk(&**boxed)` instead of
// `BoundClause::Match(m) => walk(m)`. The size delta is acceptable
// because:
//
// - `BoundQuery` is built once per query (not in a hot path);
// - the enum is moved by `Vec` into `BoundQuery::clauses` exactly
//   once and then read-only;
// - boxing the largest variant trades a heap allocation per clause
//   for a smaller stack footprint, which is a worse cache-locality
//   trade for downstream walkers.
//
// If a future slice profiles bind-allocation overhead and the
// enum-size cost dominates, M4-22 can revisit; the
// `#[non_exhaustive]`-omission convention from M4-01 means a
// representation change does not require a breaking-API marker.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum BoundClause {
    Match(BoundMatchClause),
    /// `CREATE …` write-op clause per ADR-147 (W26-θ Phase 1 —
    /// CREATE node only; CREATE rel + DELETE + SET / REMOVE /
    /// MERGE forward-pinned).
    Create(BoundCreateClause),
    /// `DELETE …` / `DETACH DELETE …` write-op clause per ADR-149
    /// (W26-θ Phase 3). Items RESOLVE to upstream-bound variables
    /// (parallel to RETURN-side projection resolution); type-check
    /// constrains each resolved binding's `TypeInfo` to
    /// `Node { .. }` or `Relationship { .. }`.
    Delete(BoundDeleteClause),
    /// `SET …` write-op clause per ADR-150 (W26-θ Phase 4). Items
    /// RESOLVE to upstream-bound variables; type-check enforces
    /// Node-or-Relationship typing (Node-only for label-add per
    /// ADR-150 §D-4) and literal-only property values per the Phase 1
    /// (ADR-147 §D-4) inherited narrowing.
    Set(BoundSetClause),
    /// `REMOVE …` write-op clause per ADR-150 (W26-θ Phase 4). Items
    /// RESOLVE to upstream-bound variables; type-check enforces
    /// Node-or-Relationship typing (Node-only for label-remove per
    /// ADR-150 §D-4).
    Remove(BoundRemoveClause),
    /// `MERGE …` write-op clause per ADR-151 (W26-θ Phase 5). The
    /// merge pattern's variables are FRESH declarations (parallel to
    /// Phase 1-2 CREATE bindings); the on_create / on_match action
    /// items RESOLVE against the pattern's fresh scope (parallel to
    /// Phase 4 SET items resolving against prior MATCH bindings).
    Merge(BoundMergeClause),
    With(BoundWithClause),
    Unwind(BoundUnwindClause),
    /// `CALL { <subquery> }` correlated brace-subquery per ADR-192
    /// (#623) — Cypher 25, a beyond-openCypher-v9 capability extension.
    /// See [`BoundCallClause`].
    Call(BoundCallClause),
    /// **ADR-197 (#802)** — `CALL <proc>(args) [YIELD …]` schema-
    /// introspection procedure call. See [`BoundCallProcedureClause`].
    CallProcedure(BoundCallProcedureClause),
    /// **ADR-197 (#802)** — `SHOW CONSTRAINTS | INDEXES | DATABASES`.
    Show(BoundShowClause),
    RankBy(BoundRankByClause),
    WithFusion(BoundWithFusionClause),
    Return(BoundReturnClause),
    /// Standalone `ORDER BY …` tail clause.
    TailOrderBy(Vec<BoundOrderItem>, Span),
    /// Standalone `SKIP …` tail clause.
    TailSkip(BoundExpression, Span),
    /// Standalone `LIMIT …` tail clause.
    TailLimit(BoundExpression, Span),
}

// =====================================================================
// 1a. CREATE (ADR-147 W26-θ Phase 1)
// =====================================================================

/// Bound `CREATE` clause body — Phase 1 admits CREATE-node items only.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCreateClause {
    pub items: Vec<BoundCreateItem>,
    pub span: Span,
}

/// One bound item inside a `CREATE` clause.
///
/// `#[allow(clippy::large_enum_variant)]` — the `Path` variant is
/// roughly 544 bytes (three `BoundCreateNodeSpec`s plus a
/// `BoundCreateRelSpec` plus a `Span`); the `Node` variant is roughly
/// 168 bytes. The size delta is acceptable for the same reason as
/// `BoundClause` (rustdoc'd at [`BoundClause`]): the enum is built
/// once per query, moved into `BoundCreateClause::items` by `Vec`,
/// and then read-only. Boxing would trade a heap allocation per item
/// for a smaller stack footprint, which is a worse cache-locality
/// trade for the downstream walkers (BindingVisitor /
/// TypeCheckVisitor / CrossSubstrateValidator /
/// LogicalPlanLoweringVisitor all walk each item exhaustively per
/// the standard match dispatch).
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum BoundCreateItem {
    Node(BoundCreateNodeSpec),
    /// Phase 2 (ADR-148) — `(source)-[rel:LABEL {props}]->(target)`
    /// path-shape. The lowering emits the source + target CREATE-node
    /// operators FIRST, then the CREATE-rel operator that consumes
    /// their NodeId bindings via the row-binding row produced by the
    /// upstream CREATE-node operators.
    Path(BoundCreatePathSpec),
}

/// Bound CREATE-path: `(source)-[rel]->(target)` per ADR-148 §D-3.
///
/// At Phase 2 both endpoints are FRESH inline-CREATE node specs (no
/// MATCH-binding resolution); the rel-direction is `LeftToRight` or
/// `RightToLeft` (undirected forward-pinned to Phase 4 per ADR-148
/// §"Forward-deferred").
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCreatePathSpec {
    pub source: BoundCreateNodeSpec,
    pub rel: BoundCreateRelSpec,
    pub target: BoundCreateNodeSpec,
    pub span: Span,
}

/// How a CREATE-path endpoint obtains its node.
///
/// `Fresh` preserves the existing CREATE behavior: the endpoint is a new
/// node produced by a `CreateNode` sub-pipeline. `RowBinding` is the
/// Phase-5 MATCH→CREATE composition case: a bare endpoint variable already
/// bound in the incoming row references that row's node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CreateEndpointBinding {
    Fresh,
    RowBinding(BindingId),
}

/// Bound CREATE-rel: optional binding + MANDATORY label-NAME +
/// optional property bag + direction.
///
/// The label NAME is forwarded verbatim (not resolved to a `TypeId`)
/// for the same read-only-catalog rationale as
/// [`BoundCreateNodeSpec::label`] (see ADR-147 §D-3). The substrate's
/// `create_rel` interns the type-name via
/// `arcgraph_storage::intern::InternTable::intern_type(tenant, name)`
/// at create-time.
///
/// At Phase 2 the property bag's value expressions are restricted to
/// `BoundExpression::Literal { value, .. }` per ADR-147 §D-4 (the
/// Phase 1 type-check narrowing is INHERITED unchanged by Phase 2).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCreateRelSpec {
    /// Allocated binding for the optional variable (None when the
    /// CREATE-rel was anonymous like `-[:KNOWS]->`).
    pub var: Option<BoundVariable>,
    /// Mandatory label NAME (Phase 2 per ADR-148 §D-1; grammar
    /// rejects label-less rel detail). Interned at substrate-execute
    /// time per the rustdoc above.
    pub label: String,
    /// Bound property values (Phase 2 = literal-only per ADR-147 §D-4
    /// inherited).
    pub properties: Option<BoundPropertyMap>,
    /// Rel direction (no Undirected at Phase 2 per ADR-148 §D-1).
    pub direction: CreateRelDirection,
    pub span: Span,
}

/// Bound CREATE-node spec: allocated optional binding for the variable,
/// optional LABEL NAME (resolved or interned at executor-substrate
/// time, NOT at binding time), and bound property values.
///
/// At Phase 1 the property bag's value expressions are restricted to
/// `BoundExpression::Literal { value, .. }` — see ADR-147 §D-4.
/// `TypeCheckVisitor` enforces this restriction.
///
/// # Why `label: Option<String>` (NOT `BoundLabelRef`)
///
/// The `CatalogProvider` trait is READ-ONLY at v1.0-α (`lookup_label`
/// returns `Option<LabelId>` — `None` for unknown names). CREATE
/// semantically WRITES a node carrying a possibly-new label name;
/// rejecting at the binding layer would conflict with the openCypher
/// v9 §6 dynamic-schema convention. The label NAME flows through to
/// the substrate, which routes through
/// `arcgraph_storage::intern::InternTable::intern_label(tenant, name)`
/// at create-time. Multi-label forward-pinned to a v1.1 amendment
/// per ADR-147 §"Forward-deferred".
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCreateNodeSpec {
    /// Allocated binding for the optional variable (None when the
    /// CREATE spec was anonymous like `CREATE (:User)`).
    pub var: Option<BoundVariable>,
    /// Optional LABEL NAME (None when the spec carried no label like
    /// `CREATE (n {...})`). Interned at substrate-execute time per
    /// the rustdoc above.
    pub label: Option<String>,
    /// Binder-resolved `LabelId` for [`Self::label`] **if that name is
    /// already interned in the catalog at bind time** — `None` when
    /// `label` is `None` OR the label name has never been interned
    /// (no live node can carry an un-interned label).
    ///
    /// This is the **match-side** resolution used by the MERGE lowering
    /// (ADR-152-amendment-01 §D-1/§D-2/§D-3) to enforce the
    /// match-branch label: `Some(id)` → `Scan{label: Some(id)}` (the
    /// same proven path MATCH uses); label-present-but-`None` → a
    /// provably-empty (`LogicalEmpty`) match-branch so the create-branch
    /// fires and mints the label. It is populated via the **None-
    /// tolerant** `lookup_label` read (NOT MATCH's erroring
    /// `BindingError::UnknownLabel` site — MERGE legitimately may name a
    /// not-yet-interned label).
    ///
    /// CREATE lowering IGNORES this field: a CREATE never matches; it
    /// mints the label by NAME at substrate-execute time
    /// ([`Self::label`]). The field is resolved uniformly for every
    /// `BoundCreateNodeSpec` (the `lookup_label` read is pure /
    /// side-effect-free per `CatalogProvider`), and only the MERGE
    /// lowering consumes it.
    pub match_label_id: Option<LabelId>,
    /// Bound property values (Phase 1 = literal-only per ADR-147 §D-4).
    pub properties: Option<BoundPropertyMap>,
    /// Endpoint binding mode for CREATE-path node positions.
    pub endpoint_binding: CreateEndpointBinding,
    pub span: Span,
}

// =====================================================================
// 1b. DELETE (ADR-149 W26-θ Phase 3)
// =====================================================================

/// Bound `DELETE` clause body — Phase 3 admits one or more
/// resolved-variable items per ADR-149 §D-3.
///
/// `detach` carries the AST `DeleteClause::detach` flag verbatim;
/// the executor consumes it at run-time to decide whether to
/// cascade-tombstone attached rels FIRST (DETACH=true) before
/// tombstoning the node, OR surface a runtime
/// `ExecutionError::Eval("relationships attached")` when a
/// node-item has attached rels (DETACH=false).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundDeleteClause {
    pub items: Vec<BoundDeleteItem>,
    pub detach: bool,
    pub span: Span,
}

/// Bound DELETE item: a RESOLVED reference to an upstream-bound
/// variable. Unlike CREATE-side bindings (which DECLARE fresh names),
/// DELETE-side bindings RESOLVE prior names — the executor reads the
/// upstream row's cell at this binding's schema slot to get the
/// NodeId / RelId to tombstone.
///
/// The variable's `BoundVariable::type_info` (populated by the M4-22
/// type-check pass on the original declaration site) carries the
/// `Node { .. }` vs `Relationship { .. }` discriminator that drives
/// the substrate dispatch at execute-time.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundDeleteItem {
    /// The resolved variable (parallel to a RETURN projection's
    /// variable-reference resolution at M4-21).
    pub var: BoundVariable,
    /// Span of the source `DELETE var` reference.
    pub span: Span,
}

// =====================================================================
// 1c. SET (ADR-150 W26-θ Phase 4)
// =====================================================================

/// Bound `SET` clause body — Phase 4 admits one or more resolved-
/// variable items per ADR-150 §D-1. Each item carries the resolved
/// `BoundVariable` + the mutation shape (per-key property assign,
/// property merge, property replace, label add).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundSetClause {
    pub items: Vec<BoundSetItem>,
    pub span: Span,
}

/// Bound SET item: a RESOLVED reference to an upstream-bound variable
/// plus the mutation. Unlike CREATE-side bindings (which DECLARE fresh
/// names), SET-side bindings RESOLVE prior names per ADR-150 §D-3
/// (parallel to ADR-149 DELETE-side resolution).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundSetItem {
    /// The resolved variable.
    pub var: BoundVariable,
    /// The mutation shape (property assign / merge / replace / label
    /// add).
    pub mutation: BoundSetMutation,
    /// Span of the source `SET <item>` reference.
    pub span: Span,
}

/// The four bound mutation shapes a `SET` item can take per ADR-150
/// §D-2.
///
/// `BoundExpression` values inside `PropertyAssign` /
/// `PropertyReplace` / `PropertyMerge` are constrained to literals at
/// the type-check pass per ADR-147 §D-4 inherited narrowing (the bound
/// AST preserves general expression shape; the type-check enforces
/// the literal-only rule and surfaces `SetPropertyValueNotLiteral` on
/// violation).
#[derive(Debug, Clone, PartialEq)]
pub enum BoundSetMutation {
    /// `SET n.prop = expr`
    PropertyAssign {
        name: String,
        value: BoundExpression,
    },
    /// `SET n = {map}` — full bag overwrite per ADR-150 §D-1.
    PropertyReplace(BoundPropertyMap),
    /// `SET n += {map}` — additive merge per ADR-150 §D-1.
    PropertyMerge(BoundPropertyMap),
    /// `SET n:L1:L2` — label add (Node-only per ADR-150 §D-4).
    LabelAdd(Vec<String>),
}

/// Bound `REMOVE` clause body — Phase 4 admits one or more resolved-
/// variable items per ADR-150 §D-1. Each item carries the resolved
/// `BoundVariable` + the removal shape (per-key property clear,
/// label remove).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundRemoveClause {
    pub items: Vec<BoundRemoveItem>,
    pub span: Span,
}

/// Bound REMOVE item: a RESOLVED reference to an upstream-bound
/// variable + the removal shape.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundRemoveItem {
    pub var: BoundVariable,
    pub mutation: BoundRemoveMutation,
    pub span: Span,
}

/// The two bound removal shapes a `REMOVE` item can take per ADR-150
/// §D-2.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundRemoveMutation {
    /// `REMOVE n.prop`
    Property(String),
    /// `REMOVE n:L1:L2` — label remove (Node-only per ADR-150 §D-4).
    LabelRemove(Vec<String>),
}

// =====================================================================
// 1d. MERGE (ADR-151 W26-θ Phase 5)
// =====================================================================

/// Bound `MERGE` clause body — Phase 5 admits a single pattern +
/// optional `on_create` / `on_match` action item vecs.
///
/// REUSES [`BoundCreateNodeSpec`] (Phase 1) / [`BoundCreatePathSpec`]
/// (Phase 2) for the pattern via [`BoundMergePattern`]; REUSES
/// [`BoundSetItem`] (Phase 4) for the action items. The pattern's
/// variables are FRESH declarations (binding-pass DECLARES them at
/// `BindingVisitor::bind_merge_clause` time, parallel to CREATE-side
/// binding); the action item `var`s RESOLVE against the now-extended
/// scope chain after pattern binding.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundMergeClause {
    pub pattern: BoundMergePattern,
    /// Action items that fire on the create branch.
    pub on_create: Vec<BoundSetItem>,
    /// Action items that fire on the match branch.
    pub on_match: Vec<BoundSetItem>,
    pub span: Span,
}

/// The two bound pattern shapes a `MERGE` item can take per
/// ADR-151 §D-2.
///
/// `#[allow(clippy::large_enum_variant)]` — same rationale as
/// [`BoundCreateItem`]; the `Path` variant is larger but the enum is
/// built once per query, moved into `BoundMergeClause::pattern` by
/// value, then read-only.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum BoundMergePattern {
    /// Node-shape: `MERGE (n:Label {props})`.
    Node(BoundCreateNodeSpec),
    /// Path-shape: `MERGE (a)-[r:R {props}]->(b)`.
    Path(BoundCreatePathSpec),
}

// =====================================================================
// 2. MATCH and patterns
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct BoundMatchClause {
    pub body: BoundMatchBody,
    pub where_clause: Option<BoundExpression>,
    /// The scope frame this MATCH declares into.
    pub scope: ScopeId,
    /// Span of the `MATCH` keyword + body.
    pub span: Span,
    /// `true` for `OPTIONAL MATCH` per ADR-006 amendment-01 +
    /// amendment-03 §TIER-1 GAP D. M4-21 sets `false`; M4-22
    /// sets `true` when the parser yields `Clause::OptionalMatch`.
    /// Downstream: variables introduced in this clause get
    /// `BoundVariable::may_be_null = true`.
    pub is_optional: bool,
}

// Same rationale as `BoundClause` — boxing the `NamedPath` variant
// changes the API ergonomics for downstream walkers, and the build
// is one-shot per query. Acceptable size delta.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, PartialEq)]
pub enum BoundMatchBody {
    Patterns(Vec<BoundPathPattern>),
    NamedPath(BoundNamedPath),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundNamedPath {
    pub var: BoundVariable,
    pub kind: BoundNamedPathKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundNamedPathKind {
    /// `SHORTEST_PATH(<pattern>)` macro OR canonical `shortestPath(...)`
    /// (ADR-194 D-3 — one algorithm, two spellings). The single
    /// minimum-length source→target path.
    ShortestPath(BoundPathPattern),
    /// Canonical `allShortestPaths(<pattern>)` (ADR-194 D-2/D-4) — ALL
    /// equal-minimum-length source→target paths.
    AllShortestPath(BoundPathPattern),
    /// Plain named path: `p = (a)-[..]->(b)`.
    Plain(BoundPathPattern),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundPathPattern {
    pub head: BoundNodePattern,
    pub tail: Vec<(BoundRelPattern, BoundNodePattern)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundNodePattern {
    /// `None` for anonymous nodes `()`.
    pub var: Option<BoundVariable>,
    /// Resolved labels (paired with their span). UnknownLabel
    /// errors do not appear here — they're surfaced by the
    /// visitor's error vec.
    pub labels: Vec<BoundLabelRef>,
    pub properties: Option<BoundPropertyMap>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundLabelRef {
    pub name: String,
    pub label_id: LabelId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundRelPattern {
    pub var: Option<BoundVariable>,
    /// Resolved rel-types. UnknownRelType errors do not appear
    /// here — they're surfaced by the visitor's error vec.
    pub rel_types: Vec<BoundRelTypeRef>,
    pub direction: RelDirection,
    /// `LengthRange::Quantified` (GQL `{N,M}`) is reserved at v1.0.
    /// Rejected by M4-22 [`crate::semantic::type_check::TypeCheckVisitor`]
    /// with `ArcQLError::NotImplemented` per D-9 + D-16; the
    /// openCypher `*N..M` form is admitted.
    pub length: Option<LengthRange>,
    pub properties: Option<BoundPropertyMap>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundRelTypeRef {
    pub name: String,
    pub type_id: TypeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundPropertyMap {
    pub entries: Vec<BoundPropertyEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundPropertyEntry {
    pub key: String,
    /// Resolved property ID. `None` when the catalog returns `None`
    /// (v1.1+ strict-schema; v1.0 dynamic-schema typically returns
    /// `Some`).
    pub property_id: Option<PropertyId>,
    pub value: BoundExpression,
    pub span: Span,
}

/// A bound variable declaration or reference.
///
/// Two `BoundVariable`s with the same `binding_id` denote the same
/// declaration site. The `name` is preserved for diagnostics +
/// downstream RETURN-projection rendering.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundVariable {
    pub name: String,
    pub binding_id: BindingId,
    /// `true` if this binding was introduced as a FRESH declaration
    /// in an OPTIONAL MATCH clause (per openCypher 9 3VL conventions;
    /// ADR-006 amendment-01 + ADR-038 §2 D-21 M4-22b refinement —
    /// Shape B). The flag is set at BINDING TIME by the M4-21
    /// `BindingVisitor`'s `declare_or_resolve_in_pattern`, NOT by
    /// the M4-22 type-check pass.
    ///
    /// Re-references in pattern positions (e.g. `MATCH (a) OPTIONAL
    /// MATCH (a)-[:R]-(c)` — the second `(a)`) INHERIT `may_be_null`
    /// from the original binding and never upgrade nullability.
    /// WITH passthrough projections (`WITH n` / `WITH n AS x`) also
    /// inherit nullability; non-passthrough projections are
    /// conservatively non-nullable at v1.0.
    pub may_be_null: bool,
    pub span: Span,
    /// Type information populated by M4-22's
    /// [`crate::semantic::type_check::TypeCheckVisitor`].
    /// `None` after the M4-21 binding pass; `Some(TypeInfo::Node {..})`
    /// for node-pattern variables, `Some(TypeInfo::Relationship {..})`
    /// for rel-pattern variables, etc., after type-check.
    pub type_info: Option<TypeInfo>,
}

// =====================================================================
// 3. WITH / UNWIND
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct BoundWithClause {
    /// `WITH DISTINCT …` — threaded from [`crate::ast::WithClause::distinct`]
    /// (#842 part B). The lowering composes [`crate::logical_plan::LogicalDistinct`]
    /// over the WITH projection when set (the same operator `RETURN DISTINCT`
    /// lowers to).
    pub distinct: bool,
    pub items: Vec<BoundProjectionItem>,
    pub where_clause: Option<BoundExpression>,
    /// The scope frame this WITH opens (containing only the
    /// projected names).
    pub scope: ScopeId,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundUnwindClause {
    pub expr: BoundExpression,
    pub var: BoundVariable,
    pub span: Span,
}

/// **ADR-197 (#802)** — the fixed catalog of schema-introspection
/// procedures supported at v1.0-α. The binder resolves a dotted
/// procedure name to one of these (an unknown name is rejected at
/// bind); the executor dispatches on it. Each procedure has a fixed
/// output-column set (validated against the YIELD list).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcedureKind {
    /// `apoc.meta.data` — one row per (label/relType, property): YIELD
    /// columns `label, other, elementType, type, property`. The
    /// langchain-neo4j `refresh_schema` driver-critical procedure.
    ApocMetaData,
    /// `apoc.schema.nodes` — index/constraint rows: YIELD `label,
    /// properties, type, size, valuesSelectivity`. Returns empty at
    /// v1.0-α (no secondary-index catalog surfaced).
    ApocSchemaNodes,
    /// `db.labels` — YIELD `label` (one row per distinct node label).
    DbLabels,
    /// `db.relationshipTypes` — YIELD `relationshipType`.
    DbRelationshipTypes,
    /// `db.propertyKeys` — YIELD `propertyKey`.
    DbPropertyKeys,
    /// `db.schema.visualization` — YIELD `nodes, relationships`.
    /// Returns a single empty-structure row at v1.0-α (langchain calls
    /// it but tolerates a partial/empty result).
    DbSchemaVisualization,
    /// `dbms.components` — the driver / library version handshake the
    /// neo4j driver + langchain-neo4j `Neo4jVector` send on connect.
    /// YIELD columns `name, versions, edition`. STATIC, zero-arg.
    ///
    /// Neo4j-compatible clients parse `records[0]["versions"][0]` into
    /// an integer tuple and gate the vector
    /// surface on it: `has_vector_index_support` needs `>= (5, 11, 0)`
    /// and `is_version_5_23_or_above` needs `>= (5, 23, 0)` — the latter
    /// is what routes `db.index.vector.queryNodes` to the SUPPORTED
    /// vector path (below 5.23 langchain falls back to a legacy /
    /// unsupported path). The executor returns a `>= 5.23` version so
    /// the D4 KNN path is reached. **#830 (D1) / ADR-198 OQ-7.**
    DbmsComponents,
    /// `db.index.vector.queryNodes(indexName, k, queryVector)` — the
    /// langchain-neo4j `Neo4jVector` per-query KNN search. YIELD columns
    /// `node, score` (one row per ranked hit, score-descending). DYNAMIC
    /// — evaluates its three arguments and calls
    /// [`crate::executor::ExecutorSubstrate::vector_search`].
    ///
    /// `indexName` is ADVISORY at v1.0-α: it resolves to the served
    /// tenant's single vector property (the named-index catalog is the
    /// D2/D3 `CREATE VECTOR INDEX` DDL, grammar-gated and out of this
    /// slice). This is documented, NOT a silent no-op — see the
    /// executor proc-body. **#830 (D4) / ADR-198 OQ-7.**
    DbIndexVectorQueryNodes,
}

impl ProcedureKind {
    /// Resolve a dotted procedure name (case-insensitive on the `db.`/
    /// `apoc.` namespace per Neo4j convention) to a [`ProcedureKind`].
    /// Returns `None` for an unknown procedure.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "apoc.meta.data" => Some(Self::ApocMetaData),
            "apoc.schema.nodes" => Some(Self::ApocSchemaNodes),
            "db.labels" => Some(Self::DbLabels),
            "db.relationshiptypes" => Some(Self::DbRelationshipTypes),
            "db.propertykeys" => Some(Self::DbPropertyKeys),
            "db.schema.visualization" => Some(Self::DbSchemaVisualization),
            // #830 D1 + D4. Match is on the ASCII-lower-cased name, so
            // the mixed-case `db.index.vector.queryNodes` Neo4j spells
            // is matched here lower-cased (`…querynodes`).
            "dbms.components" => Some(Self::DbmsComponents),
            "db.index.vector.querynodes" => Some(Self::DbIndexVectorQueryNodes),
            _ => None,
        }
    }

    /// The procedure's fixed output columns (the names it YIELDs), in
    /// declaration order. A YIELD clause must reference a subset of
    /// these; an empty YIELD (standalone `CALL proc()`) yields all.
    #[must_use]
    pub fn output_columns(self) -> &'static [&'static str] {
        match self {
            Self::ApocMetaData => &["label", "other", "elementType", "type", "property"],
            Self::ApocSchemaNodes => &["label", "properties", "type", "size", "valuesSelectivity"],
            Self::DbLabels => &["label"],
            Self::DbRelationshipTypes => &["relationshipType"],
            Self::DbPropertyKeys => &["propertyKey"],
            Self::DbSchemaVisualization => &["nodes", "relationships"],
            // #830 D1: the Neo4j `dbms.components()` YIELD contract.
            Self::DbmsComponents => &["name", "versions", "edition"],
            // #830 D4: the Neo4j `db.index.vector.queryNodes` YIELD
            // contract — `(node, score)` per ranked hit.
            Self::DbIndexVectorQueryNodes => &["node", "score"],
        }
    }
}

/// **ADR-197 (#802)** — bound `CALL <proc>(args) [YIELD …]`. The
/// resolved [`ProcedureKind`], the (unevaluated) argument expressions,
/// and the YIELD'd output bindings (one [`BoundVariable`] per yielded
/// column, declared into the scope so the following WHERE / RETURN can
/// reference them — like UNWIND's output binding).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCallProcedureClause {
    /// Resolved procedure.
    pub kind: ProcedureKind,
    /// Bound argument expressions (e.g. `apoc.meta.data({sample: 1000})`).
    /// v1.0-α the args are accepted + bound but the procedure bodies do
    /// not interpret them (sampling is a no-op).
    pub args: Vec<BoundExpression>,
    /// One binding per YIELD'd column, in YIELD order. Each carries the
    /// source column name + the bound (aliased) variable. Empty YIELD
    /// (standalone `CALL proc()`) yields ALL output columns.
    pub yields: Vec<BoundProcedureYield>,
    /// Optional bound `WHERE <pred>` filtering the YIELD'd rows
    /// (lowered to a `Filter` over the procedure op).
    pub where_clause: Option<BoundExpression>,
    /// Source span.
    pub span: Span,
}

/// A single YIELD binding for a [`BoundCallProcedureClause`].
#[derive(Debug, Clone, PartialEq)]
pub struct BoundProcedureYield {
    /// The procedure output column this yields (one of
    /// [`ProcedureKind::output_columns`]).
    pub column: String,
    /// The bound (possibly aliased) variable the column flows into.
    pub var: BoundVariable,
}

/// **ADR-197 (#802)** — bound `SHOW CONSTRAINTS | INDEXES | DATABASES`.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundShowClause {
    /// The SHOW target.
    pub kind: crate::ast::ShowKind,
    /// The output bindings (column name → bound variable), in column
    /// order. With no YIELD this is the full column set; with an
    /// explicit `YIELD` (#830) it is the YIELD'd subset (aliased). SHOW
    /// commands have fixed columns; v1.0-α returns rows for
    /// `SHOW DATABASES` (one default db) and empty rowsets for
    /// `CONSTRAINTS`/`INDEXES`/`VECTOR INDEXES`.
    pub columns: Vec<BoundVariable>,
    /// Optional `WHERE <pred>` filtering the YIELD'd rows (#830). Bound
    /// AFTER `columns` are declared so the predicate can reference the
    /// YIELD'd columns — mirrors [`BoundCallProcedureClause`]. Lowered
    /// to a `Filter` wrapping the SHOW rows.
    pub where_clause: Option<BoundExpression>,
    /// Source span.
    pub span: Span,
}

/// Bound `CALL { <subquery> }` correlated brace-subquery per ADR-192
/// (#623) — Cypher 25, a beyond-openCypher-v9 capability extension.
///
/// The `body` is bound in a CHILD scope SEEDED with the outer in-scope
/// variables (implicit import, D-3) — so `resolve` inside the body
/// walks up the scope chain and finds the outer variables WITHOUT a
/// mandatory importing-`WITH`. `imported` records the outer bindings in
/// scope at the `CALL` point; the executor seeds those into the
/// subquery's per-driving-row environment (the correlation). `returned`
/// records the body's terminal-RETURN columns, RE-DECLARED in the OUTER
/// scope (the only subquery vars that escape the scoping fence, D-4) so
/// the enclosing query can reference them; the executor relabels the
/// body's output columns to these binding-ids POSITIONALLY (the body's
/// own projection op mints fresh synthetic ids — see
/// [`crate::executor::ops::ProjectOp`] — so the relabel is what makes
/// the outer reference resolve).
#[derive(Debug, Clone, PartialEq)]
pub struct BoundCallClause {
    /// The bound subquery body — [`BoundStatement::Read`] or
    /// [`BoundStatement::Union`] (the grammar admits only those two).
    pub body: Box<BoundStatement>,
    /// Outer in-scope bindings imported into the subquery (D-3). Drives
    /// the per-driving-row correlation seed in the executor. Empty for
    /// an uncorrelated subquery (e.g. a leading `CALL { … }`).
    pub imported: Vec<BindingId>,
    /// The body's terminal-RETURN output columns, declared in the OUTER
    /// scope (visible after `}`, D-4). The executor's `CallOp` relabels
    /// the body's output columns to these ids positionally + appends
    /// them to the driving-row columns.
    pub returned: Vec<BindingId>,
    /// Span of the `CALL` clause.
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundProjectionItem {
    pub kind: BoundProjectionKind,
    pub alias: Option<String>,
    /// Output binding-id contract (#746). The [`BindingId`] under which
    /// this projected column appears in the operator's output row
    /// schema — and the SAME id any downstream consumer (a 2nd
    /// `Project` over an `Aggregate`, a `MATCH`/`RETURN` after a
    /// `WITH`, an `UNWIND` of a WITH-projected list) resolves the
    /// column to.
    ///
    /// The binder assigns this during `bind_with_clause` /
    /// `bind_return_clause`: for a WITH-projected name it is the id the
    /// post-WITH scope `declare()`s (so a downstream `resolve()` returns
    /// the SAME id the column carries); for a RETURN item it is a fresh
    /// monotonic id. The executor's [`crate::executor::ops::ProjectOp`]
    /// (and the lowering's aggregate-output wiring) USE this id rather
    /// than minting an executor-local synthetic — closing the
    /// binder↔executor binding-id mismatch that blocked
    /// Project-over-Aggregate + WITH-projection execution.
    ///
    /// `None` for [`BoundProjectionKind::Wildcard`]: `*` passes the
    /// child schema through unchanged (each passed-through column keeps
    /// its original id), so there is no single fresh output id to mint.
    pub output_id: Option<BindingId>,
    /// #353 — verbatim (whitespace-normalized) source text of an
    /// un-aliased expression projection, threaded from
    /// [`crate::ast::ProjectionItem::source_text`] at bind time. The
    /// implicit result-column name openCypher/Neo4j surface for an
    /// un-aliased expression (`RETURN n.name` → `"n.name"`; `RETURN
    /// count(*)` → `"count(*)"`). `None` for a wildcard, and for a
    /// hand-built bound item (only the bind path threads it from the
    /// parser-captured AST field). Consumed by
    /// [`BoundProjectionItem::display_name`] (the alias takes
    /// precedence) to drive the user-meaningful column names on the
    /// `MaterializedResult` → MCP / Bolt wire surfaces.
    pub source_text: Option<String>,
    pub span: Span,
}

impl BoundProjectionItem {
    /// #353 — the user-meaningful result-column name for this
    /// projection item, matching openCypher / Neo4j implicit-column
    /// naming:
    ///
    /// - explicit `AS alias` → the alias (the user's chosen name);
    /// - bare variable reference (`RETURN n`) → the variable's name;
    /// - any other un-aliased expression → the verbatim
    ///   [`Self::source_text`] (`n.name`, `count(*)`, `a.x + 1`);
    /// - defensive fallback (no alias, no source text, not a bare var —
    ///   e.g. a hand-built test item) → `col_{index}` so the wire is
    ///   never empty.
    ///
    /// `index` is the column's zero-based position in the projection,
    /// used only for the defensive fallback label (so two un-nameable
    /// columns don't collide).
    ///
    /// A [`BoundProjectionKind::Wildcard`] has no single name — it
    /// expands to the child schema's columns; callers handle wildcard
    /// expansion separately (see
    /// [`crate::output_column_names`]). For a wildcard this returns the
    /// `col_{index}` fallback, but the column-name extractor never calls
    /// it on a wildcard item.
    #[must_use]
    pub fn display_name(&self, index: usize) -> String {
        if let Some(alias) = &self.alias {
            return alias.clone();
        }
        if let BoundProjectionKind::Expr(BoundExpression::VariableRef { name, .. }) = &self.kind {
            return name.clone();
        }
        if let Some(src) = &self.source_text {
            return src.clone();
        }
        format!("col_{index}")
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundProjectionKind {
    /// `RETURN *` / `WITH *` — expands to every CURRENTLY in-scope
    /// binding.
    ///
    /// `order` is the in-scope binding ids in **openCypher wildcard
    /// output order** — i.e. ASCENDING by the variable's NAME (Cypher 9
    /// §6.1: "`RETURN *` returns all variables, in alphabetical order").
    /// The physical row/schema columns carry insertion order (the order
    /// the variables were declared along the pipeline), so the wildcard
    /// passthrough must REORDER the child columns into this name-sorted
    /// order at projection time — otherwise a multi-variable `RETURN *`
    /// (e.g. `WITH … AS xs, ys, zs UNWIND … RETURN *`, Unwind1 `13`)
    /// would emit columns in declaration order (`xs, ys, zs, …`) rather
    /// than alphabetical (`x, xs, y, ys, z, zs`).
    ///
    /// An empty `order` (the default for a hand-built test item, or a
    /// wildcard the binder did not populate) means "fall back to verbatim
    /// child-schema passthrough" — preserving the pre-fix behavior for
    /// any path that constructs the variant directly.
    Wildcard { order: Vec<BindingId> },
    /// Bound expression projection.
    Expr(BoundExpression),
}

impl BoundProjectionKind {
    /// The unit `Wildcard` (empty name-sort order) — for hand-built test
    /// items and any caller that does not need the alphabetical reorder.
    /// An empty order makes the passthrough fall back to verbatim
    /// child-schema order (the pre-fix behavior).
    #[must_use]
    pub fn wildcard() -> Self {
        BoundProjectionKind::Wildcard { order: Vec::new() }
    }
}

// =====================================================================
// 4. RANK BY
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct BoundRankByClause {
    pub ranker: BoundRanker,
    /// Optional value binding carrying the fused score.
    pub score: Option<BoundVariable>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundRanker {
    Hybrid(Vec<BoundRankArg>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundRankArg {
    Vector {
        field: BoundFieldRef,
        query: BoundExpression,
        k: Option<i64>,
        span: Span,
    },
    Text {
        field: BoundFieldRef,
        query: BoundExpression,
        k: Option<i64>,
        span: Span,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundFieldRef {
    pub base: BoundVariable,
    /// Property path; each segment carries an optional resolved
    /// `PropertyId`.
    pub path: Vec<BoundPropertyRef>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundPropertyRef {
    pub name: String,
    pub property_id: Option<PropertyId>,
    pub span: Span,
}

// =====================================================================
// 5. WITH FUSION = …
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct BoundWithFusionClause {
    pub fusion: BoundFusion,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundFusion {
    Rrf { k: i64 },
}

// =====================================================================
// 6. RETURN
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct BoundReturnClause {
    pub distinct: bool,
    pub items: Vec<BoundProjectionItem>,
    pub order_by: Vec<BoundOrderItem>,
    pub skip: Option<BoundExpression>,
    pub limit: Option<BoundExpression>,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundOrderItem {
    pub expr: BoundExpression,
    pub direction: OrderDirection,
    pub span: Span,
}

// =====================================================================
// 7. Expressions
// =====================================================================

/// Bound expression form. Mirrors [`crate::ast::Expression`] with
/// resolved bindings + spans + optional type info.
///
/// Every variant carries a `type_info: Option<TypeInfo>` field
/// populated by M4-22's [`crate::semantic::type_check::TypeCheckVisitor`].
/// `None` after the M4-21 binding pass; `Some(...)` after type-check.
///
/// `Clone` is HAND-WRITTEN (not derived) with an iterative left-spine
/// walk — see the impl below (#1290).
#[derive(Debug, PartialEq)]
pub enum BoundExpression {
    Literal {
        value: Literal,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    ListLiteral {
        elements: Vec<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    MapLiteral {
        entries: Vec<(String, BoundExpression)>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    Parameter {
        name: String,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// A variable reference that resolved to a binding.
    VariableRef {
        name: String,
        binding_id: BindingId,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// A variable reference that did NOT resolve. The visitor emits
    /// [`crate::semantic::error::BindingError::UndeclaredVariable`]
    /// for this case AND records the unresolved node so downstream
    /// passes (M4-22 type-check) can still walk a complete tree.
    UnresolvedVariable { name: String, span: Span },
    /// `n.prop` / `n.prop.sub`. Each path segment carries a resolved
    /// `PropertyId` (or `None` if the catalog did not know it).
    PropertyAccess {
        base: Box<BoundExpression>,
        path: Vec<BoundPropertyRef>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    BinaryOp {
        op: BinOp,
        lhs: Box<BoundExpression>,
        rhs: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// Function call. M4-22's type-checker resolves the function
    /// name against [`crate::semantic::functions`] and arity-checks
    /// the call; on success the `type_info` slot carries the return
    /// type, on failure a `TypeCheckError` is recorded.
    FunctionCall {
        name: String,
        args: Vec<BoundExpression>,
        /// `count(DISTINCT x)` / `collect(DISTINCT x)` — dedup the
        /// aggregated values before the fold (#773 G5). Threaded from
        /// [`crate::ast::Expression::FunctionCall::distinct`] and read by
        /// the lowering's `try_lift_aggregation` into
        /// [`crate::logical_plan::AggregationSpec::distinct`].
        distinct: bool,
        /// `count(*)` — count ROWS (no expr argument; #773 G4). When
        /// `true`, `args` is empty.
        star: bool,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// `<expr> NEAR <expr>` (ADR-038 D-5 vector ANN predicate).
    Near {
        lhs: Box<BoundExpression>,
        target: Box<BoundExpression>,
        vector_index: Option<String>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// `<expr> MATCH <expr>` (ADR-038 D-6 BM25 text-match).
    TextMatch {
        lhs: Box<BoundExpression>,
        query: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// `n IN COMMUNITY($cid)` per ADR-038 amendment-01 (alternate
    /// surface; canonical D-4 shape uses `community(...)`
    /// function-calls, parsed as `FunctionCall`). M4-23 unifies the
    /// two surfaces.
    // TODO(M4-23): unify with community(...) function-call form.
    InCommunity {
        node: Box<BoundExpression>,
        community: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// `expr IN <list_or_param>`.
    In {
        lhs: Box<BoundExpression>,
        rhs: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    IsNull {
        lhs: Box<BoundExpression>,
        negated: bool,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **ADR-188** — bound list-predicate (`all`/`any`/`none`/`single`).
    /// `var_bid` is the iteration variable's [`BindingId`], declared in
    /// a child scope by [`crate::semantic::binding`]'s
    /// `push_scope`/`declare`/`pop_scope` and resolvable ONLY inside
    /// `predicate`. The evaluator appends the element value to the
    /// current row at a slot computed at eval time (= `row.len()`, the
    /// extended-row base) and wires a scoped schema closure mapping
    /// `var_bid → that slot` (ADR-188 Decision 1). `type_info` is
    /// `Boolean` (Cypher predicate; may yield `null` at runtime under
    /// 3VL — the static type is `Boolean`, the 3VL `null` is a runtime
    /// value).
    ListPredicate {
        quantifier: Quantifier,
        var_bid: BindingId,
        list: Box<BoundExpression>,
        predicate: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **ADR-188** — bound `reduce(acc = init, x IN list | expr)`.
    /// `acc_bid` + `var_bid` are declared in one child scope (LIFO with
    /// the fold body). At eval time the accumulator occupies the
    /// extended-row base slot (`row.len()`) and the element the next
    /// (`row.len() + 1`); `acc`'s slot is overwritten with the running
    /// accumulator each iteration (ADR-188 Decision 1 + Decision 4
    /// pure-fold). `type_info` is the fold result type (= the
    /// type-widening join of `acc` and the body per Decision
    /// 3-reduce-widening / OQ-5).
    Reduce {
        acc_bid: BindingId,
        init: Box<BoundExpression>,
        var_bid: BindingId,
        list: Box<BoundExpression>,
        expr: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **ADR-188** (Decision 5 — #620 list-half) — bound list
    /// comprehension `[x IN list WHERE p | e]`. `var_bid` is the
    /// iteration variable's [`BindingId`], declared in a child scope by
    /// [`crate::semantic::binding`]'s `push_scope`/`declare`/`pop_scope`
    /// and resolvable ONLY inside `predicate` + `projection`. The
    /// evaluator binds each element at the extended-row base slot
    /// (= `row.len()`) via the SAME per-element extended-row synthesis
    /// as [`BoundExpression::ListPredicate`] (ADR-188 Decision 1),
    /// applies the optional `predicate` as a 3VL filter (only `true`
    /// keeps the element), and projects `projection` (identity over
    /// `var` when absent), collecting into a `Value::List`. `predicate`
    /// and `projection` are both `Option` (the four openCypher v9 §3.5
    /// combinations). `type_info` is `List(element-or-projection type)`.
    ListComprehension {
        var_bid: BindingId,
        list: Box<BoundExpression>,
        predicate: Option<Box<BoundExpression>>,
        projection: Option<Box<BoundExpression>>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **ADR-191 D-6** (#620 map-half) — bound map projection
    /// `n{.key, .other, alias: expr, .*}` (openCypher v9 §3.5). `base` is
    /// the projected variable bound to a node / relationship / map (a
    /// [`BoundExpression::VariableRef`], or
    /// [`BoundExpression::UnresolvedVariable`] if the name did not resolve
    /// — surfaced at type-check). Each [`BoundMapProjectionItem`] carries
    /// the D-6 null-handling split: a `Property` selector DROPS a
    /// null/absent value; a `Literal` entry KEEPS its key even when the
    /// value is null; `AllProperties` (`.*`) copies every property of the
    /// base. The evaluator builds a [`crate::executor::Value::Map`]
    /// (ADR-191). `type_info` is `Map`. NOTE: the base is the ONLY
    /// row-binding reference (no expression-internal scoped var, unlike the
    /// comprehensions) — `collect_referenced_bindings` keeps it.
    MapProjection {
        base: Box<BoundExpression>,
        items: Vec<BoundMapProjectionItem>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **openCypher v9 §3.4** — bound list element access `base[index]`.
    /// `type_info` is the element type of `base` (`element_type_of`) —
    /// an out-of-range index yields `null` at runtime (the static type
    /// is the element type; the 3VL `null` is a runtime value, mirroring
    /// the `ListComprehension` / `In` treatment).
    Subscript {
        base: Box<BoundExpression>,
        index: Box<BoundExpression>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **openCypher v9 §3.4** — bound list slice `base[start..end]`.
    /// `type_info` is `base`'s list type (slicing a list yields a list of
    /// the same element type). Either bound may be absent (open form).
    Slice {
        base: Box<BoundExpression>,
        start: Option<Box<BoundExpression>>,
        end: Option<Box<BoundExpression>>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
    /// **openCypher v9 §3.6** (#621) — bound conditional `CASE` expression.
    /// `test` is `Some` for the SIMPLE form (compared by openCypher value
    /// equality against each branch's WHEN value), `None` for the SEARCHED
    /// form (each WHEN is a standalone 3VL boolean condition). `branches` are
    /// the `(WHEN, THEN)` arms in source order (non-empty — the grammar
    /// requires ≥1); `default` is the optional ELSE. No expression-internal
    /// scoped variable (unlike the comprehensions) — every sub-expression
    /// evaluates in the CURRENT scope, so `collect_referenced_bindings`
    /// keeps every binding the sub-expressions reference. `type_info` is the
    /// permissive join of all THEN + ELSE types (a heterogeneous CASE — legal
    /// openCypher — joins to `Null`, the 3VL-aware "could be anything"
    /// sentinel; it never type-errors on branch-type divergence).
    Case {
        test: Option<Box<BoundExpression>>,
        branches: Vec<(BoundExpression, BoundExpression)>,
        default: Option<Box<BoundExpression>>,
        span: Span,
        type_info: Option<TypeInfo>,
    },
}

/// **#1290** — HAND-WRITTEN `Clone` with an iterative left-spine walk.
///
/// The derived `Clone` recursed once per tree level; a flat operator
/// chain folds into a left-nested spine up to
/// [`crate::parser::MAX_FLAT_CHAIN_DEPTH`] deep, and deep bound-tree
/// clones are endemic on the pipeline (lowering clones projection
/// items and filter predicates), so the derive's per-level frames
/// overflowed the native stack on legitimate wide filters. Mirrors
/// the [`crate::ast::Expression`] `Clone` impl: walk down the
/// `BinaryOp`/`UnaryOp`/`In`/`IsNull` spine collecting one frame per
/// level, clone the non-spine base per-variant, rebuild bottom-up.
/// Non-spine children clone recursively (bracket-bounded by
/// `MAX_EXPRESSION_DEPTH`).
impl Clone for BoundExpression {
    fn clone(&self) -> Self {
        enum SpineFrame<'a> {
            Binary {
                op: BinOp,
                rhs: &'a BoundExpression,
                span: &'a Span,
                type_info: &'a Option<TypeInfo>,
            },
            Unary {
                op: UnaryOp,
                span: &'a Span,
                type_info: &'a Option<TypeInfo>,
            },
            In {
                rhs: &'a BoundExpression,
                span: &'a Span,
                type_info: &'a Option<TypeInfo>,
            },
            IsNull {
                negated: bool,
                span: &'a Span,
                type_info: &'a Option<TypeInfo>,
            },
        }
        let mut frames: Vec<SpineFrame<'_>> = Vec::new();
        let mut cur = self;
        loop {
            match cur {
                BoundExpression::BinaryOp {
                    op,
                    lhs,
                    rhs,
                    span,
                    type_info,
                } => {
                    frames.push(SpineFrame::Binary {
                        op: op.clone(),
                        rhs,
                        span,
                        type_info,
                    });
                    cur = lhs;
                }
                BoundExpression::UnaryOp {
                    op,
                    operand,
                    span,
                    type_info,
                } => {
                    frames.push(SpineFrame::Unary {
                        op: op.clone(),
                        span,
                        type_info,
                    });
                    cur = operand;
                }
                BoundExpression::In {
                    lhs,
                    rhs,
                    span,
                    type_info,
                } => {
                    frames.push(SpineFrame::In {
                        rhs,
                        span,
                        type_info,
                    });
                    cur = lhs;
                }
                BoundExpression::IsNull {
                    lhs,
                    negated,
                    span,
                    type_info,
                } => {
                    frames.push(SpineFrame::IsNull {
                        negated: *negated,
                        span,
                        type_info,
                    });
                    cur = lhs;
                }
                _ => break,
            }
        }
        let mut acc = clone_non_spine_bound_expression(cur);
        while let Some(frame) = frames.pop() {
            acc = match frame {
                SpineFrame::Binary {
                    op,
                    rhs,
                    span,
                    type_info,
                } => BoundExpression::BinaryOp {
                    op,
                    lhs: Box::new(acc),
                    rhs: Box::new(rhs.clone()),
                    span: span.clone(),
                    type_info: type_info.clone(),
                },
                SpineFrame::Unary {
                    op,
                    span,
                    type_info,
                } => BoundExpression::UnaryOp {
                    op,
                    operand: Box::new(acc),
                    span: span.clone(),
                    type_info: type_info.clone(),
                },
                SpineFrame::In {
                    rhs,
                    span,
                    type_info,
                } => BoundExpression::In {
                    lhs: Box::new(acc),
                    rhs: Box::new(rhs.clone()),
                    span: span.clone(),
                    type_info: type_info.clone(),
                },
                SpineFrame::IsNull {
                    negated,
                    span,
                    type_info,
                } => BoundExpression::IsNull {
                    lhs: Box::new(acc),
                    negated,
                    span: span.clone(),
                    type_info: type_info.clone(),
                },
            };
        }
        acc
    }
}

/// The per-variant clone for every NON-spine [`BoundExpression`] form —
/// what the derive would have generated, minus the four left-spine
/// operator arms, which the iterative driver in `Clone::clone` handles.
/// Those arms remain here as a total-match fallback that re-enters
/// `Clone` (the driver despines, so no unbounded recursion is
/// possible); they are unreachable from the driver itself.
fn clone_non_spine_bound_expression(e: &BoundExpression) -> BoundExpression {
    use BoundExpression as BE;
    match e {
        BE::BinaryOp { .. } | BE::UnaryOp { .. } | BE::In { .. } | BE::IsNull { .. } => e.clone(),
        BE::Literal {
            value,
            span,
            type_info,
        } => BE::Literal {
            value: value.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::ListLiteral {
            elements,
            span,
            type_info,
        } => BE::ListLiteral {
            elements: elements.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::MapLiteral {
            entries,
            span,
            type_info,
        } => BE::MapLiteral {
            entries: entries.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Parameter {
            name,
            span,
            type_info,
        } => BE::Parameter {
            name: name.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::VariableRef {
            name,
            binding_id,
            span,
            type_info,
        } => BE::VariableRef {
            name: name.clone(),
            binding_id: *binding_id,
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::UnresolvedVariable { name, span } => BE::UnresolvedVariable {
            name: name.clone(),
            span: span.clone(),
        },
        BE::PropertyAccess {
            base,
            path,
            span,
            type_info,
        } => BE::PropertyAccess {
            base: base.clone(),
            path: path.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::FunctionCall {
            name,
            args,
            distinct,
            star,
            span,
            type_info,
        } => BE::FunctionCall {
            name: name.clone(),
            args: args.clone(),
            distinct: *distinct,
            star: *star,
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Near {
            lhs,
            target,
            vector_index,
            span,
            type_info,
        } => BE::Near {
            lhs: lhs.clone(),
            target: target.clone(),
            vector_index: vector_index.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::TextMatch {
            lhs,
            query,
            span,
            type_info,
        } => BE::TextMatch {
            lhs: lhs.clone(),
            query: query.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::InCommunity {
            node,
            community,
            span,
            type_info,
        } => BE::InCommunity {
            node: node.clone(),
            community: community.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::ListPredicate {
            quantifier,
            var_bid,
            list,
            predicate,
            span,
            type_info,
        } => BE::ListPredicate {
            quantifier: *quantifier,
            var_bid: *var_bid,
            list: list.clone(),
            predicate: predicate.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Reduce {
            acc_bid,
            init,
            var_bid,
            list,
            expr,
            span,
            type_info,
        } => BE::Reduce {
            acc_bid: *acc_bid,
            init: init.clone(),
            var_bid: *var_bid,
            list: list.clone(),
            expr: expr.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::ListComprehension {
            var_bid,
            list,
            predicate,
            projection,
            span,
            type_info,
        } => BE::ListComprehension {
            var_bid: *var_bid,
            list: list.clone(),
            predicate: predicate.clone(),
            projection: projection.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::MapProjection {
            base,
            items,
            span,
            type_info,
        } => BE::MapProjection {
            base: base.clone(),
            items: items.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Subscript {
            base,
            index,
            span,
            type_info,
        } => BE::Subscript {
            base: base.clone(),
            index: index.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Slice {
            base,
            start,
            end,
            span,
            type_info,
        } => BE::Slice {
            base: base.clone(),
            start: start.clone(),
            end: end.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
        BE::Case {
            test,
            branches,
            default,
            span,
            type_info,
        } => BE::Case {
            test: test.clone(),
            branches: branches.clone(),
            default: default.clone(),
            span: span.clone(),
            type_info: type_info.clone(),
        },
    }
}

/// **ADR-191 D-6** (#620 map-half) — one bound element of a map projection.
/// Mirrors [`crate::ast::MapProjectionItem`]; the `Literal` entry's value
/// is lowered to a [`BoundExpression`]. The D-6 null-handling split is the
/// variant choice (NOT a runtime flag): the evaluator DROPS a `Property`
/// selector whose value is null/absent and KEEPS a `Literal` entry's key
/// even when its value is null.
#[derive(Debug, Clone, PartialEq)]
pub enum BoundMapProjectionItem {
    /// `.key` — include `key` with the base's value; DROP if null/absent.
    Property(String),
    /// `.*` — include every property of the base entity / map.
    AllProperties,
    /// `alias: expr` — include `alias` with `expr`'s value; KEEP if null.
    Literal {
        alias: String,
        value: Box<BoundExpression>,
    },
}

impl BoundExpression {
    /// Return the span covering this expression.
    pub fn span(&self) -> &Span {
        match self {
            BoundExpression::Literal { span, .. }
            | BoundExpression::ListLiteral { span, .. }
            | BoundExpression::MapLiteral { span, .. }
            | BoundExpression::Parameter { span, .. }
            | BoundExpression::VariableRef { span, .. }
            | BoundExpression::UnresolvedVariable { span, .. }
            | BoundExpression::PropertyAccess { span, .. }
            | BoundExpression::BinaryOp { span, .. }
            | BoundExpression::UnaryOp { span, .. }
            | BoundExpression::FunctionCall { span, .. }
            | BoundExpression::Near { span, .. }
            | BoundExpression::TextMatch { span, .. }
            | BoundExpression::InCommunity { span, .. }
            | BoundExpression::In { span, .. }
            | BoundExpression::IsNull { span, .. }
            | BoundExpression::ListPredicate { span, .. }
            | BoundExpression::Reduce { span, .. }
            | BoundExpression::ListComprehension { span, .. }
            | BoundExpression::MapProjection { span, .. }
            | BoundExpression::Subscript { span, .. }
            | BoundExpression::Slice { span, .. }
            | BoundExpression::Case { span, .. } => span,
        }
    }

    /// Return the `type_info` carried by this expression. `None` for
    /// unresolved variables (which never type-check) or before the
    /// M4-22 type-check pass has run.
    pub fn type_info(&self) -> Option<&TypeInfo> {
        match self {
            BoundExpression::Literal { type_info, .. }
            | BoundExpression::ListLiteral { type_info, .. }
            | BoundExpression::MapLiteral { type_info, .. }
            | BoundExpression::Parameter { type_info, .. }
            | BoundExpression::VariableRef { type_info, .. }
            | BoundExpression::PropertyAccess { type_info, .. }
            | BoundExpression::BinaryOp { type_info, .. }
            | BoundExpression::UnaryOp { type_info, .. }
            | BoundExpression::FunctionCall { type_info, .. }
            | BoundExpression::Near { type_info, .. }
            | BoundExpression::TextMatch { type_info, .. }
            | BoundExpression::InCommunity { type_info, .. }
            | BoundExpression::In { type_info, .. }
            | BoundExpression::IsNull { type_info, .. }
            | BoundExpression::ListPredicate { type_info, .. }
            | BoundExpression::Reduce { type_info, .. }
            | BoundExpression::ListComprehension { type_info, .. }
            | BoundExpression::MapProjection { type_info, .. }
            | BoundExpression::Subscript { type_info, .. }
            | BoundExpression::Slice { type_info, .. }
            | BoundExpression::Case { type_info, .. } => type_info.as_ref(),
            BoundExpression::UnresolvedVariable { .. } => None,
        }
    }

    /// In-place mutator for `type_info`. Used by the M4-22
    /// [`crate::semantic::type_check::TypeCheckVisitor`] to
    /// populate the slot post-bind.
    pub fn set_type_info(&mut self, ti: TypeInfo) {
        match self {
            BoundExpression::Literal { type_info, .. }
            | BoundExpression::ListLiteral { type_info, .. }
            | BoundExpression::MapLiteral { type_info, .. }
            | BoundExpression::Parameter { type_info, .. }
            | BoundExpression::VariableRef { type_info, .. }
            | BoundExpression::PropertyAccess { type_info, .. }
            | BoundExpression::BinaryOp { type_info, .. }
            | BoundExpression::UnaryOp { type_info, .. }
            | BoundExpression::FunctionCall { type_info, .. }
            | BoundExpression::Near { type_info, .. }
            | BoundExpression::TextMatch { type_info, .. }
            | BoundExpression::InCommunity { type_info, .. }
            | BoundExpression::In { type_info, .. }
            | BoundExpression::IsNull { type_info, .. }
            | BoundExpression::ListPredicate { type_info, .. }
            | BoundExpression::Reduce { type_info, .. }
            | BoundExpression::ListComprehension { type_info, .. }
            | BoundExpression::MapProjection { type_info, .. }
            | BoundExpression::Subscript { type_info, .. }
            | BoundExpression::Slice { type_info, .. }
            | BoundExpression::Case { type_info, .. } => *type_info = Some(ti),
            BoundExpression::UnresolvedVariable { .. } => {
                // No-op — unresolved variables don't carry a type.
            }
        }
    }
}

// =====================================================================
// 8. Type info (M4-22)
// =====================================================================

/// Resolved type of a bound expression / variable / property access.
///
/// Populated by M4-22's [`crate::semantic::type_check::TypeCheckVisitor`].
/// The variant set covers the openCypher / ArcQL value taxonomy at v1.0:
///
/// - **Graph values** — `Node` (with optional resolved [`LabelId`]),
///   `Relationship` (with optional resolved [`TypeId`]).
/// - **Property values** — `Property` (resolved [`PropertyId`] +
///   scalar [`PropertyType`]).
/// - **Scalar values** — `Boolean`, `Integer`, `Float`, `String`.
/// - **Null** — the "could be NULL" type per ADR-038 D-20 3VL.
/// - **Composite** — `List(elem)`, `Map`.
///
/// # NULL semantics (ADR-038 D-20)
///
/// `TypeInfo::Null` represents the openCypher 3VL "could be NULL"
/// value. NULL propagates through comparisons and arithmetic per the
/// truth table in D-20; the type-checker carries `Null` rather than
/// failing on operand-type-mismatch when one side is NULL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeInfo {
    /// A node value (e.g., a variable bound by `MATCH (n:Person)`).
    /// `label` is `Some` if the catalog resolved the label name;
    /// `None` for label-free patterns `MATCH (n) ...`.
    Node {
        label: Option<arcgraph_core::LabelId>,
    },
    /// A relationship value (e.g., a variable bound by `[r:KNOWS]`).
    Relationship {
        rel_type: Option<arcgraph_core::TypeId>,
    },
    /// A resolved property value: catalog-interned [`PropertyId`]
    /// plus the scalar value type.
    Property {
        property_id: arcgraph_core::PropertyId,
        value_type: PropertyType,
    },
    Boolean,
    Integer,
    Float,
    String,
    /// "Could be NULL" — the 3VL-aware sentinel. ADR-038 D-20.
    Null,
    /// Homogeneous list of `elem`-typed values.
    List(Box<TypeInfo>),
    /// Map / property-bag value.
    Map,
    /// **ADR-193.** A path value (`MATCH p = (a)-[..]->(b)`) — the
    /// openCypher v9 §3 alternating node/relationship sequence. Carried
    /// by the `NamedPathKind::Plain` path variable. ORDERABLE
    /// (`ORDER BY <path>` is VALID and orders deterministically — paths
    /// sort FIRST in the global type-order, D-11/D-13); accepted by
    /// `nodes`/`relationships`/`length` (D-7).
    Path,
    /// **W23-V11-T-01 / ADR-090.** Wall-clock instant with timezone
    /// (`TIMESTAMPTZ` per ADR-007 amendment-01).
    Temporal,
    /// Wall-clock without zone. Per K3 §2.3 "Gap shape" item 1.
    LocalDateTime,
    /// Calendar date with no time / zone.
    Date,
    /// ISO-8601 duration. Per K3 §2.3 "Gap shape" item 1.
    Duration,
    /// Fixed-point decimal `(scale, units)`. Per V11-T-02 companion.
    Decimal,
}

/// Scalar property value type. Mirrors the openCypher property value
/// taxonomy at v1.1.
///
/// Carried by [`TypeInfo::Property`]; the catalog's
/// `lookup_property_type` (v1.1+) returns one of these. At v1.0 the
/// catalog did not track property-value types; v1.1 opens the enum
/// per ADR-038 amendment-09 + ADR-090 to admit temporal + decimal
/// variants. The type-checker uses the literal-derivation rule when
/// the catalog returns no concrete type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyType {
    Boolean,
    Integer,
    Float,
    String,
    /// **W23-V11-T-01 / ADR-090** — `TIMESTAMPTZ` wall-clock instant.
    Temporal,
    /// Wall-clock without zone.
    LocalDateTime,
    /// Calendar date.
    Date,
    /// ISO-8601 duration.
    Duration,
    /// Fixed-point decimal.
    Decimal,
}
