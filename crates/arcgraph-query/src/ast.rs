//! ArcQL AST — typed mirror of `grammar.pest` productions.
//!
//! # Design notes
//!
//! - **Owned strings.** v1.0 keeps strings owned (`String`) for
//!   simplicity; v1.1 may refactor to borrowed slices for zero-copy
//!   if profiling justifies. See README "v1.1 considerations".
//! - **`PartialEq` everywhere.** The 256-case round-trip property
//!   test (`tests/grammar_proptest.rs`) needs structural equality
//!   between the original AST and a re-parsed printed form.
//! - **`Display` is the round-trip pretty-printer.** `Display` MUST
//!   emit text that re-parses to a structurally equal AST; this is
//!   the property `roundtrip_parse_print_parse` checks.
//! - **No semantic resolution.** This is M4-01 scope; the AST
//!   carries syntactic shape only. Variable binding, label
//!   existence, type checking, and reserved-clause `NotImplemented`
//!   detection (ADR-038 D-16) all live in M4-02.
//!
//! # ADR provenance
//! - ADR-006 D-1 — openCypher subset baseline.
//! - ADR-038 §2 D-1..D-10 — every clause variant is justified by
//!   one of these decisions.
//! - ADR-038 §3.4 — the locked test names (`parser_accepts_…`,
//!   `executor_returns_not_implemented_…`) pin against drift; M4-02
//!   pins the executor side, M4-01 (here) pins the parser side.

use std::fmt;

// =====================================================================
// 1. Top-level
// =====================================================================

/// A complete ArcQL statement.
///
/// `Statement::Read` covers the supported openCypher subset and
/// ArcGraph's native search clauses.
///
/// `Statement::Explain` / `Statement::Profile` (M4-91; ADR-038 §2
/// D-19 + amendment-03 §TIER-1 GAP B) wrap a read query for
/// planner-only EXPLAIN (lit at v1.0) or execute-then-annotate
/// PROFILE (parser-lit at v1.0; executor binding deferred to M4-71
/// row-count observer + M4-61 execution layer). The wrapper carries
/// `ReadQuery` (NOT `Statement`) because D-19 restricts the body to a
/// full read query and the grammar enforces this — index DDL cannot be
/// EXPLAINed.
#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Read(ReadQuery),
    /// Neo4j-compatible index DDL (#830, ADR-198 §OQ-7):
    /// `CREATE VECTOR INDEX …` and `DROP INDEX … [IF EXISTS]`. Parsed +
    /// bound at this layer (label / property captured); the index
    /// BUILD / lifecycle is the vector track's follow-up, so the
    /// type-checker rejects with `ArcQLError::NotImplemented`
    /// (parsed-but-not-built) — distinct from a parse error. Kept
    IndexDdl(IndexDdlStatement),
    /// `EXPLAIN <read_query>` per ADR-038 §2 D-19. The
    /// [`crate::explain::explain`] entry point lowers to a
    /// planner-only path (no executor invocation, no snapshot LSN
    /// acquired per D-18 rule 1).
    Explain(ReadQuery),
    /// `PROFILE <read_query>` per ADR-038 §2 D-19. Parser-lit at
    /// v1.0 (M4-91); executor lowering deferred to M4-71
    /// (row-count observer) + M4-61 (execution layer). The
    /// [`crate::explain::profile`] entry point returns
    /// [`crate::semantic::ArcQLError::NotImplemented`] until those
    /// land.
    Profile(ReadQuery),
    /// `<body> UNION [ALL] <body> …` per ADR-185 (#649-A1, W28 —
    /// openCypher v9 §8 "Set operations"). The arms are tail-free
    /// read-query bodies (the grammar factors any ORDER BY / SKIP /
    /// LIMIT out to [`UnionQuery::tail`], bound to the WHOLE union —
    /// the RC-2 fix). See [`UnionQuery`].
    Union(UnionQuery),
}

/// A read query is a sequence of clauses applied in order.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadQuery {
    pub clauses: Vec<Clause>,
}

/// A `UNION` / `UNION ALL` set-operation query per ADR-185 (#649-A1,
/// W28) — openCypher v9 §8.
///
/// `arms` are the ≥2 tail-free read-query bodies, left-to-right.
/// `all` carries one flag PER BOUNDARY (`all.len() == arms.len() - 1`):
/// `all[i] == true` ⇔ the boundary between `arms[i]` and `arms[i+1]`
/// is `UNION ALL` (keep duplicates); `false` ⇔ bare `UNION` (distinct).
/// openCypher v9 §8 forbids MIXING the two within one union (TCK
/// `Union3` → `InvalidClauseComposition`); the all-flags-must-agree
/// rule is enforced at bind time
/// ([`crate::semantic::error::BindingError::UnionMixedSetOps`]) rather
/// than in the grammar.
///
/// `tail` is the post-union ORDER BY / SKIP / LIMIT, applied to the
/// COMBINED result (NOT the last arm — the RC-2 fix per the PE FROZEN
/// CONTRACT item 1).
#[derive(Debug, Clone, PartialEq)]
pub struct UnionQuery {
    /// The union arms (≥2), in source (left-to-right) order.
    pub arms: Vec<ReadQuery>,
    /// Per-boundary `ALL` flags; `all.len() == arms.len() - 1`.
    pub all: Vec<bool>,
    /// Post-union ORDER BY / SKIP / LIMIT (bound to the whole union).
    pub tail: UnionTail,
}

/// The post-union tail (ORDER BY / SKIP / LIMIT) bound to the WHOLE
/// union per openCypher v9 §8 (ADR-185). An all-empty `UnionTail`
/// (`order_by.is_empty() && skip.is_none() && limit.is_none()`) is the
/// no-tail case (the grammar's elided `union_tail?`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct UnionTail {
    /// ORDER BY items (empty when absent).
    pub order_by: Vec<OrderItem>,
    /// SKIP expression (when present).
    pub skip: Option<Expression>,
    /// LIMIT expression (when present).
    pub limit: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Clause {
    Match(MatchClause),
    /// `OPTIONAL MATCH …` per ADR-006 amendment-01 + ADR-038
    /// amendment-03 §TIER-1 GAP D. Same body shape as
    /// [`Clause::Match`]; the discriminant carries the OPTIONAL
    /// flag, which the M4-22 binding pass uses to set
    /// `BoundMatchClause::is_optional = true` + propagate
    /// `BoundVariable::may_be_null = true`.
    OptionalMatch(MatchClause),
    /// `CREATE …` write-op clause per ADR-147 (W26-θ Phase 1 —
    /// CREATE node only at this slice; CREATE rel forward-pinned
    /// to Phase 2). openCypher v9 §6 "Updating Clauses".
    Create(CreateClause),
    /// `DELETE …` / `DETACH DELETE …` write-op clause per ADR-149
    /// (W26-θ Phase 3). openCypher v9 §6 "Updating Clauses".
    /// Admits one or more comma-separated identifier arguments that
    /// MUST resolve to a Node-typed or Relationship-typed
    /// upstream-bound variable (type-checked at M4-22). The optional
    /// `DETACH` prefix admits deleting a node with attached rels
    /// (the rels are tombstoned FIRST); without `DETACH`, the
    /// executor surfaces a runtime "relationships attached" error.
    Delete(DeleteClause),
    /// `SET …` write-op clause per ADR-150 (W26-θ Phase 4).
    /// openCypher v9 §6 "Updating Clauses". Admits per-item
    /// mutations of four shapes: per-key property assign
    /// (`n.prop = expr`), property merge (`n += {map}`), property
    /// replace (`n = {map}`), and label add (`n:Label1:Label2`).
    /// Each item resolves its `var` against the upstream MATCH
    /// scope; type-check enforces Node or Relationship typing
    /// (Node-only for label-add).
    Set(SetClause),
    /// `REMOVE …` write-op clause per ADR-150 (W26-θ Phase 4).
    /// openCypher v9 §6 "Updating Clauses". Admits per-item
    /// mutations of two shapes: per-key property remove
    /// (`n.prop`) and label remove (`n:Label1:Label2`). Each item
    /// resolves its `var` against the upstream MATCH scope; type-
    /// check enforces Node or Relationship typing (Node-only for
    /// label-remove).
    Remove(RemoveClause),
    /// `MERGE …` write-op clause per ADR-151 (W26-θ Phase 5).
    /// openCypher v9 §6 "MERGE": match-or-create. The merge
    /// pattern reuses Phase 1 ([`CreateNodeSpec`]) + Phase 2
    /// ([`CreatePathSpec`]) shapes verbatim. Optional `ON CREATE
    /// SET …` / `ON MATCH SET …` action clauses each carry a Phase 4
    /// [`SetItem`] vec. The executor probes the pattern against the
    /// current snapshot; on match — bind matched rows + fire
    /// `on_match` actions; on miss — fire the create branch + bind
    /// the newly created rows + fire `on_create` actions.
    Merge(MergeClause),
    With(WithClause),
    Unwind(UnwindClause),
    /// `CALL { <subquery> }` correlated brace-subquery per ADR-192
    /// (#623). **Cypher 25 — a deliberate beyond-openCypher-v9
    /// capability extension** (the vendored v9 TCK has ZERO `CALL{}`
    /// scenarios; v9's `CALL` is a procedure call, which is scoped OUT).
    /// The subquery runs once per driving (outer) row, implicitly
    /// importing the outer in-scope variables, and concatenates its
    /// result rows back (UNION-ALL semantics) joined to the driving row.
    /// The implicit-import set is computed at BIND time (ADR-192 D-3),
    /// not parsed. v1.0-α admits READ-ONLY subquery bodies; a write
    /// clause inside the body is rejected at bind
    /// ([`crate::semantic::error::BindingError::WriteInCallSubqueryNotSupported`],
    /// ADR-192 D-9).
    Call(CallClause),
    /// **ADR-197 (#802)** — `CALL <proc>(args) [YIELD …]` procedure
    /// call (schema-introspection: `apoc.meta.data`, `db.labels`, …).
    /// The YIELD'd items become bindings flowing into the following
    /// clauses (like UNWIND's output binding).
    CallProcedure(CallProcedureClause),
    /// **ADR-197 (#802)** — `SHOW CONSTRAINTS | INDEXES | DATABASES`.
    Show(ShowClause),
    RankBy(RankByClause),
    WithFusion(WithFusionClause),
    Return(ReturnClause),
    /// Standalone `ORDER BY …` tail clause. ADR-038 D-3 / D-7
    /// examples chain ORDER BY after `RANK BY ... WITH FUSION ...`
    /// without an explicit `RETURN`. Modeled as its own clause so
    /// the AST does not have to fabricate a synthetic RETURN.
    TailOrderBy(Vec<OrderItem>),
    /// Standalone `SKIP …` tail clause; same rationale as above.
    TailSkip(Expression),
    /// Standalone `LIMIT …` tail clause; same rationale as above.
    TailLimit(Expression),
}

// =====================================================================
// 2a. CREATE (ADR-147 W26-θ Phase 1)
// =====================================================================

/// `CREATE` clause body: one or more items separated by `,`.
///
/// Phase 1 admits `CreateItem::Node` only. Phase 2 (W26-θ-2 sister
/// PR) adds `CreateItem::Rel`. Multi-statement chained CREATE with
/// references to MATCH-bound variables (`MATCH (a) CREATE (a)-[:R]->(b)`)
/// requires CREATE rel + cross-clause binding flow, both forward-pinned.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateClause {
    pub items: Vec<CreateItem>,
}

/// One item inside a `CREATE` clause body. Phase 1 ships
/// `CreateItem::Node`; Phase 2 (ADR-148) adds `CreateItem::Path`.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateItem {
    Node(CreateNodeSpec),
    /// Phase 2 (ADR-148) — `(source)-[rel:LABEL {props}]->(target)`
    /// path-shape. Both endpoints are inline-CREATE node specs at
    /// Phase 2; MATCH-bound endpoint resolution forward-pinned to
    /// Phase 5 (ArcQL-statement-scoped batch transaction).
    Path(CreatePathSpec),
}

/// Node-shape inside a `CREATE` clause: optional variable binding,
/// optional SINGLE label, optional property bag.
///
/// Multi-label `CREATE (n:A:B)` is forward-pinned to a v1.1 amendment
/// per ADR-147 §"Forward-deferred": v1.0 `NodeRecord` carries a single
/// `LabelId`, so multi-label support requires both a record-shape
/// extension and a v1.1 storage migration.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateNodeSpec {
    pub var: Option<String>,
    pub label: Option<String>,
    pub properties: Option<PropertyMap>,
}

/// Path-shape inside a `CREATE` clause: `(source)-[rel]->(target)`.
///
/// Both endpoints are inline-CREATE node specs at Phase 2 (ADR-148
/// §D-1). The rel-label is MANDATORY and the rel-direction is
/// `LeftToRight` or `RightToLeft` (undirected forward-pinned to
/// Phase 4 per ADR-148 §"Forward-deferred").
#[derive(Debug, Clone, PartialEq)]
pub struct CreatePathSpec {
    pub source: CreateNodeSpec,
    pub rel: CreateRelSpec,
    pub target: CreateNodeSpec,
}

/// Relationship-shape inside a `CREATE` path: optional variable
/// binding + MANDATORY label + optional property bag + direction.
///
/// Discipline: distinct from [`RelPattern`] so CREATE-rel grammar
/// constraints (mandatory label, no undirected at Phase 2, no
/// variable-length) stay separate from MATCH-rel admissibility (per
/// the [`CreateNodeSpec`] vs [`NodePattern`] rationale).
#[derive(Debug, Clone, PartialEq)]
pub struct CreateRelSpec {
    pub var: Option<String>,
    /// Mandatory at Phase 2 (ADR-148 §D-1); the grammar rejects
    /// label-less rel detail. Single label only at Phase 2; multi-
    /// rel-type (`:A|B`) forward-pinned to v1.1.
    pub label: String,
    pub properties: Option<PropertyMap>,
    pub direction: CreateRelDirection,
}

/// Direction of a CREATE-path rel.
///
/// Distinct enum from [`RelDirection`] so the CREATE-rel
/// Phase 2 narrowing (no undirected) is enforced at the type level.
/// `Undirected` lands at Phase 4 alongside the MATCH disambiguation
/// surface.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CreateRelDirection {
    /// `-[..]->`
    LeftToRight,
    /// `<-[..]-`
    RightToLeft,
    // Undirected forward-pinned to Phase 4 per ADR-148 §"Forward-deferred"
}

// =====================================================================
// 2b. DELETE (ADR-149 W26-θ Phase 3)
// =====================================================================

/// `DELETE …` / `DETACH DELETE …` clause body per ADR-149 §D-1.
///
/// Phase 3 admits one or more comma-separated identifier arguments;
/// each MUST resolve to a Node-typed or Relationship-typed
/// upstream-bound variable per ADR-149 §D-3 (binding) + §D-4
/// (type-check). The optional `detach` flag controls whether the
/// executor tombstones attached rels FIRST (DETACH=true) or surfaces
/// a runtime "relationships attached" Eval error (DETACH=false) when
/// a node-item has attached rels.
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteClause {
    pub items: Vec<DeleteItem>,
    /// `true` if the source was `DETACH DELETE ...`. Preserved
    /// verbatim from the grammar's optional `detach` production;
    /// drives the executor-side cascading-rel-tombstone behavior at
    /// per-tenant Transaction layer per ADR-149 §D-7.
    pub detach: bool,
}

/// One item inside a `DELETE` clause body. Phase 3 admits only the
/// identifier-argument shape (per ADR-149 §D-1); the parser-side
/// `delete_item` production constrains the syntactic surface to
/// `identifier`. The semantic constraint that the identifier resolve
/// to a Node-typed or Relationship-typed binding is enforced at the
/// type-check layer (ADR-149 §D-4).
#[derive(Debug, Clone, PartialEq)]
pub struct DeleteItem {
    pub var: String,
}

// =====================================================================
// 2c. SET (ADR-150 W26-θ Phase 4)
// =====================================================================

/// `SET …` clause body per ADR-150 §D-1.
///
/// Phase 4 admits one or more comma-separated items; each item carries
/// its target variable name + the mutation shape. The grammar-side
/// `set_clause` production constrains the syntactic surface to four
/// item shapes (property assign / merge / replace / label add); the
/// semantic constraint that each item resolve to a Node or Relationship
/// binding (Node-only for label add) is enforced at the type-check
/// layer per ADR-150 §D-4.
#[derive(Debug, Clone, PartialEq)]
pub struct SetClause {
    pub items: Vec<SetItem>,
}

/// One item inside a `SET` clause body. Each item targets a variable
/// (`var`) with a specific [`SetMutation`].
#[derive(Debug, Clone, PartialEq)]
pub struct SetItem {
    pub var: String,
    pub mutation: SetMutation,
}

/// The four mutation shapes a `SET` item can take per ADR-150 §D-1.
///
/// - `PropertyAssign { name, value }` — `n.name = "Alice"`
/// - `PropertyReplace(PropertyMap)` — `n = {name: "Bob"}` (full bag
///   overwrite; existing entries NOT in the map are cleared).
/// - `PropertyMerge(PropertyMap)` — `n += {name: "Bob"}` (additive;
///   existing entries outside the map are preserved).
/// - `LabelAdd(labels)` — `n:VIP:Premium` (multi-label add; Node-only
///   per ADR-150 §D-4).
#[derive(Debug, Clone, PartialEq)]
pub enum SetMutation {
    PropertyAssign { name: String, value: Expression },
    PropertyReplace(PropertyMap),
    PropertyMerge(PropertyMap),
    LabelAdd(Vec<String>),
}

/// `REMOVE …` clause body per ADR-150 §D-1.
///
/// Phase 4 admits one or more comma-separated items; each item carries
/// its target variable name + the removal shape. The grammar-side
/// `remove_clause` production constrains the syntactic surface to two
/// item shapes (property remove / label remove); the semantic
/// constraint that each item resolve to a Node or Relationship binding
/// (Node-only for label remove) is enforced at the type-check layer
/// per ADR-150 §D-4.
#[derive(Debug, Clone, PartialEq)]
pub struct RemoveClause {
    pub items: Vec<RemoveItem>,
}

/// One item inside a `REMOVE` clause body. Each item targets a
/// variable (`var`) with a specific [`RemoveMutation`].
#[derive(Debug, Clone, PartialEq)]
pub struct RemoveItem {
    pub var: String,
    pub mutation: RemoveMutation,
}

/// The two mutation shapes a `REMOVE` item can take per ADR-150 §D-1.
///
/// - `Property(name)` — `REMOVE n.age` (per-key property clear).
/// - `LabelRemove(labels)` — `REMOVE n:VIP` (multi-label remove;
///   Node-only per ADR-150 §D-4).
#[derive(Debug, Clone, PartialEq)]
pub enum RemoveMutation {
    Property(String),
    LabelRemove(Vec<String>),
}

// =====================================================================
// 2d. MERGE (ADR-151 W26-θ Phase 5)
// =====================================================================

/// `MERGE …` clause body per ADR-151 §D-2.
///
/// Phase 5 admits a single pattern (Node or Path) + optional `ON
/// CREATE SET …` / `ON MATCH SET …` action clauses. The grammar-side
/// `merge_clause` production constrains the syntactic surface; the
/// semantic constraints (literal-only property values per ADR-147 §D-4
/// inherited; action items resolve to Node-or-Relationship typing per
/// ADR-150 §D-4 inherited) are enforced at the type-check layer.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeClause {
    pub pattern: MergePattern,
    /// Action items that fire when the create branch is taken.
    /// Empty when `MERGE` has no `ON CREATE SET …` clause.
    pub on_create: Vec<SetItem>,
    /// Action items that fire when the match branch is taken.
    /// Empty when `MERGE` has no `ON MATCH SET …` clause.
    pub on_match: Vec<SetItem>,
}

/// The two pattern shapes a `MERGE` item can take per ADR-151 §D-2.
///
/// REUSES [`CreateNodeSpec`] (Phase 1) and [`CreatePathSpec`]
/// (Phase 2) verbatim — the source-text shape `(var:Label {props})`
/// (Node) and `(a)-[r:R {props}]->(b)` (Path) is identical to CREATE.
/// The match-or-create branching is encoded at the executor layer,
/// NOT at the AST.
#[derive(Debug, Clone, PartialEq)]
pub enum MergePattern {
    /// Node-shape: `MERGE (n:Label {props})`.
    Node(CreateNodeSpec),
    /// Path-shape: `MERGE (a)-[r:R {props}]->(b)`.
    Path(CreatePathSpec),
}

// =====================================================================
// 2. MATCH and patterns (D-1, D-7)
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct MatchClause {
    pub body: MatchBody,
    pub where_clause: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MatchBody {
    /// Plain `MATCH (a)-[r]->(b)` style.
    Patterns(Vec<PathPattern>),
    /// `MATCH p = SHORTEST_PATH(...)` (ADR-038 D-7) or
    /// `MATCH p = (a)-[..]->(b)` (named-path binding).
    NamedPath(NamedPath),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NamedPath {
    pub var: String,
    pub kind: NamedPathKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NamedPathKind {
    /// `SHORTEST_PATH(<pattern>)` (uppercase macro, ADR-038 §D-7) OR the
    /// canonical openCypher camelCase `shortestPath(<pattern>)` — TWO
    /// SPELLINGS of the SAME single-shortest-path algorithm (ADR-194 D-3).
    /// Both spellings parse to this one variant; the grammar maps them to
    /// the same `shortest_path_pattern` rule so the distinction is erased
    /// before this AST.
    ShortestPath(PathPattern),
    /// Canonical openCypher `allShortestPaths(<pattern>)` — ALL
    /// equal-minimum-length source→target paths (ADR-194 D-2/D-4). Sibling
    /// of `ShortestPath` (one min-length path) and `Plain`.
    AllShortestPath(PathPattern),
    /// Plain named path: `p = (a)-[..]->(b)`.
    Plain(PathPattern),
}

#[derive(Debug, Clone, PartialEq)]
pub struct PathPattern {
    pub head: NodePattern,
    /// Each element binds a relationship + the trailing node it
    /// connects to. Empty for a single-node pattern `(a)`.
    pub tail: Vec<(RelPattern, NodePattern)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodePattern {
    pub var: Option<String>,
    /// Multi-label syntax `(n:Label1:Label2)` is openCypher v9.
    pub labels: Vec<String>,
    pub properties: Option<PropertyMap>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RelPattern {
    pub var: Option<String>,
    /// Empty = match any rel type.
    pub rel_types: Vec<String>,
    pub direction: RelDirection,
    pub length: Option<LengthRange>,
    pub properties: Option<PropertyMap>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RelDirection {
    /// `-[..]->`
    LeftToRight,
    /// `<-[..]-`
    RightToLeft,
    /// `-[..]-`
    Undirected,
}

/// openCypher `*N..M` and GQL `{N,M}` length-range. Both are
/// representable; the M4-02 semantic analyzer rejects the GQL form
/// at v1.0 per ADR-038 D-9 ("v1.1 lights").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LengthRange {
    /// `*` — unbounded.
    Unbounded,
    /// `*N..M` (openCypher v9).
    Cypher { min: u32, max: Option<u32> },
    /// `{N,M}` (GQL ISO 39075:2024 §10.x).
    Quantified { min: u32, max: Option<u32> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PropertyMap {
    pub entries: Vec<(String, Expression)>,
}

// =====================================================================
// 3. WITH / UNWIND
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct WithClause {
    /// `WITH DISTINCT …` — mid-pipeline row dedup (#842 part B). Mirrors
    /// [`ReturnClause::distinct`]; lowers to the SAME [`crate::logical_plan::LogicalDistinct`]
    /// operator `RETURN DISTINCT` uses, composed over the WITH projection.
    pub distinct: bool,
    pub items: Vec<ProjectionItem>,
    pub where_clause: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnwindClause {
    pub expr: Expression,
    pub var: String,
}

/// `CALL { <subquery> }` body per ADR-192 (#623). The body is a full
/// READ query — a [`Statement::Read`] (MATCH/WITH/UNWIND/RETURN/…) or a
/// [`Statement::Union`] (UNION / UNION ALL arms, ADR-185). The grammar
/// admits only those two `Statement` shapes (EXPLAIN / PROFILE / DDL /
/// index DDL is excluded inside braces); the parser never produces
/// the other variants here. `Box`ed because [`Statement`] is large
/// (it embeds whole query bodies) and a subquery nests it inside a
/// clause of the enclosing query.
#[derive(Debug, Clone, PartialEq)]
pub struct CallClause {
    /// The subquery body — `Statement::Read` or `Statement::Union`.
    pub body: Box<Statement>,
}

/// **ADR-197 (#802)** — `CALL <proc>(args) [YIELD item [AS alias], …]`.
///
/// Schema-introspection procedure call (`apoc.meta.data`, `db.labels`,
/// …). At v1.0-α the supported procedures are a fixed catalog (see
/// `crate::logical_plan::ProcedureKind`); an unknown procedure name
/// is rejected at bind. The YIELD'd items become row bindings flowing
/// into the following WHERE / RETURN clauses.
#[derive(Debug, Clone, PartialEq)]
pub struct CallProcedureClause {
    /// Fully-qualified dotted procedure name, e.g. `"apoc.meta.data"`,
    /// `"db.labels"`.
    pub name: String,
    /// Positional argument expressions (e.g. `apoc.meta.data({sample: 1000})`).
    pub args: Vec<Expression>,
    /// YIELD items: `(yielded_column, optional_alias)`. Empty when no
    /// YIELD clause (a standalone `CALL proc()` whose result is the
    /// query result).
    pub yield_items: Vec<(String, Option<String>)>,
    /// Optional `WHERE <pred>` filtering the YIELD'd rows
    /// (`CALL apoc.meta.data(...) YIELD … WHERE NOT type = 'RELATIONSHIP'`
    /// — the langchain `refresh_schema` shape). `None` when absent.
    pub where_clause: Option<Expression>,
}

/// **ADR-197 (#802) · #830 (ADR-198 §OQ-7)** —
/// `SHOW CONSTRAINTS | INDEXES | DATABASES | VECTOR INDEXES`
/// `[YIELD item [AS alias], … [WHERE <pred>]]`.
///
/// The optional `YIELD … [WHERE …]` tail mirrors
/// [`CallProcedureClause`] EXACTLY (#830): Neo4j-compatible vector
/// clients send `SHOW VECTOR INDEXES YIELD name, labelsOrTypes,
/// properties, options WHERE name = $index_name RETURN …`, so the
/// bare-kind form alone does not unblock it. The YIELD'd items become
/// row bindings flowing into the WHERE here + a following RETURN
/// clause.
#[derive(Debug, Clone, PartialEq)]
pub struct ShowClause {
    /// The SHOW target kind.
    pub kind: ShowKind,
    /// YIELD items: `(yielded_column, optional_alias)`. Empty when no
    /// YIELD clause (a bare `SHOW <kind>` whose full column set is the
    /// result).
    pub yield_items: Vec<(String, Option<String>)>,
    /// Optional `WHERE <pred>` filtering the YIELD'd rows. `None` when
    /// absent.
    pub where_clause: Option<Expression>,
}

/// The kind of a [`ShowClause`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShowKind {
    /// `SHOW CONSTRAINTS`
    Constraints,
    /// `SHOW INDEXES`
    Indexes,
    /// `SHOW DATABASES`
    Databases,
    /// `SHOW VECTOR INDEXES` (#830). v1.0 surfaces an empty rowset with
    /// the Neo4j SHOW-VECTOR-INDEXES column schema (no vector indexes
    /// exist until the vector track wires the build per ADR-198 §OQ-7).
    VectorIndexes,
}

#[derive(Debug, Clone)]
pub struct ProjectionItem {
    pub kind: ProjectionKind,
    pub alias: Option<String>,
    /// Verbatim source text of an un-aliased expression projection
    /// (#353). openCypher / Neo4j use the expression's SOURCE TEXT as
    /// the implicit result-column name when no `AS alias` is given
    /// (`RETURN n.name` → column `"n.name"`; `RETURN count(*)` → column
    /// `"count(*)"`). The parser captures the expression rule's
    /// `Pair::as_str()` here so the column-name derivation (binder →
    /// `MaterializedResult` → MCP `RawQueryRows` / Bolt `RunOutcome`
    /// wire) emits user-meaningful names instead of synthesized
    /// `col_0..N` labels.
    ///
    /// `None` for [`ProjectionKind::Wildcard`] (`*` has no single
    /// expression) and, defensively, when source capture is
    /// unavailable. An explicit `AS alias` takes precedence over this
    /// source text in the display-name rule (the alias is the user's
    /// chosen column name); this field is the fallback for un-aliased
    /// items only.
    pub source_text: Option<String>,
}

/// `source_text` is DERIVED display metadata (the implicit column name
/// for an un-aliased expression, #353), NOT part of the projection's
/// logical identity: two projections that project the same expression
/// under the same (or no) alias are semantically equal regardless of
/// the verbatim source slice the parser happened to capture. Excluding
/// it from equality keeps the parse→print→parse round-trip property
/// (`grammar_proptest::roundtrip_parse_print_parse`) meaningful — the
/// AST printer (`Display`) renders from `kind` + `alias` only, so a
/// re-parse re-captures a `source_text` that need not byte-match the
/// strategy-built `None`, yet the two ASTs are logically identical.
/// (`#[derive(PartialEq)]` would compare the derived field and spuriously
/// fail that round-trip; this manual impl is the standard pattern for a
/// cache/derived field.)
impl PartialEq for ProjectionItem {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.alias == other.alias
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionKind {
    /// `RETURN *`
    Wildcard,
    /// `RETURN n`, `RETURN n.x`, `RETURN n.x + 1`, …
    Expr(Expression),
}

// =====================================================================
// 4. RANK BY
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct RankByClause {
    pub ranker: Ranker,
    /// Optional binding that exposes the ranker's fused score.
    pub score_alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Ranker {
    Hybrid(Vec<RankArg>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RankArg {
    Vector {
        field: FieldRef,
        query: Expression,
        k: Option<i64>,
    },
    Text {
        field: FieldRef,
        query: Expression,
        k: Option<i64>,
    },
}

// =====================================================================
// 5. WITH FUSION = …
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct WithFusionClause {
    pub fusion: Fusion,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Fusion {
    Rrf { k: i64 },
}

// =====================================================================
// 6. RETURN
// =====================================================================

#[derive(Debug, Clone, PartialEq)]
pub struct ReturnClause {
    pub distinct: bool,
    pub items: Vec<ProjectionItem>,
    pub order_by: Vec<OrderItem>,
    pub skip: Option<Expression>,
    pub limit: Option<Expression>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderItem {
    pub expr: Expression,
    pub direction: OrderDirection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
    Default,
}

// =====================================================================
// 7. Expressions (WHERE + projection)
// =====================================================================

#[derive(Debug, PartialEq)]
pub enum Expression {
    Literal(Literal),
    Parameter(String),
    Identifier(String),
    /// Property access chain: `n`, `n.prop`, `n.prop.sub`. The
    /// length-1 form `n.prop` is the canonical D-3 / D-5 / D-6
    /// `field` shape.
    PropertyAccess {
        base: Box<Expression>,
        path: Vec<String>,
    },
    BinaryOp {
        op: BinOp,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    UnaryOp {
        op: UnaryOp,
        operand: Box<Expression>,
    },
    FunctionCall {
        name: String,
        args: Vec<Expression>,
        /// `count(DISTINCT x)` / `collect(DISTINCT x)` — deduplicate the
        /// aggregated values before the fold (openCypher v9 §3; #773
        /// G5). `false` for the bare `fn(x)` / `fn(a, b, c)` forms. Only
        /// valid on aggregating functions (enforced at type-check).
        distinct: bool,
        /// `count(*)` — counts ROWS rather than non-NULL arg values
        /// (openCypher v9 §3; #773 G4). When `true`, `args` is empty
        /// (the star form takes no expression). Only valid on `count`
        /// (enforced at type-check).
        star: bool,
    },
    /// ADR-038 D-5 — vector ANN predicate.
    Near {
        lhs: Box<Expression>,
        target: Box<Expression>,
        vector_index: Option<String>,
    },
    /// ADR-038 D-6 — BM25 text-match predicate. The operator
    /// `MATCH` is overloaded with the clause keyword `MATCH`; the
    /// PEG disambiguates by position.
    TextMatch {
        lhs: Box<Expression>,
        query: Box<Expression>,
    },
    /// `n IN COMMUNITY($cid)` per ADR-038 amendment-01 (alternate
    /// surface; the canonical D-4 shape uses `community(...)`
    /// function-calls and is parsed as `FunctionCall` above).
    InCommunity {
        node: Box<Expression>,
        community: Box<Expression>,
    },
    /// `expr IN <list_or_param>`.
    In {
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    IsNull {
        lhs: Box<Expression>,
        negated: bool,
    },
    /// **ADR-188** — openCypher v9 list-predicate function
    /// (`all`/`any`/`none`/`single`): a quantifier over `list` with an
    /// iteration variable `var` bound ONLY inside `predicate`. Parsed
    /// from the dedicated `filter_expr` grammar production (NOT the
    /// function-call path — `x IN list WHERE p` cannot parse as a
    /// `FunctionCall` argument; see ADR-188 Decision 2). The scoped
    /// `var` is resolved via the binder's child-scope primitives and
    /// evaluated through per-element extended-row synthesis (ADR-188
    /// Decision 1).
    ListPredicate {
        quantifier: Quantifier,
        /// Iteration variable, bound only inside `predicate`.
        var: String,
        list: Box<Expression>,
        predicate: Box<Expression>,
    },
    /// **ADR-188** — openCypher v9 `reduce(acc = init, x IN list | expr)`
    /// list-reduction (left-fold with TWO scoped variables: the
    /// accumulator `acc_var` and the element `var`, both bound only
    /// inside `expr`). A pure fold — no 3VL short-circuit; `null`
    /// propagates as an ordinary value (ADR-188 Decision 4).
    Reduce {
        /// Accumulator variable, init-typed.
        acc_var: String,
        init: Box<Expression>,
        /// Element variable.
        var: String,
        list: Box<Expression>,
        /// Fold body; `acc_var` + `var` in scope.
        expr: Box<Expression>,
    },
    /// **ADR-188** (Decision 5 — #620 list-half) — openCypher v9 §3.5
    /// list comprehension `[x IN list WHERE predicate | projection]`.
    /// For each element `x` of `list` (in order) where `predicate(x)`
    /// is TRUE (3VL: only `true` passes — `null`/`false` filter the
    /// element out), the result list collects `projection(x)`. The
    /// iteration variable `var` is bound ONLY inside `predicate` and
    /// `projection` (the same expression-internal scoped-var lifetime
    /// as `ListPredicate`). Both the WHERE filter and the `| projection`
    /// are OPTIONAL:
    /// - `[x IN list]` (neither) ⇒ identity over the whole list;
    /// - `[x IN list WHERE p]` (no projection) ⇒ filter, project `x`
    ///   itself (identity);
    /// - `[x IN list | e]` (no WHERE) ⇒ map every element;
    /// - `[x IN list WHERE p | e]` (both) ⇒ filter then map.
    ///
    /// Parsed from the dedicated `list_comprehension` grammar
    /// production (NOT a `list_literal` — it is shape-ambiguous with
    /// `[...]` but the `identifier IN` prefix disambiguates; ADR-188
    /// Decision 2 places it BEFORE `literal` in `primary_atom`).
    /// Reuses the per-element extended-row synthesis + scoped binding
    /// of `ListPredicate` (ADR-188 Decision 1 + Decision 5). The
    /// map-comprehension half of #620 is DEFERRED — it needs a runtime
    /// `Value::Map` which does not exist (ADR-188 Decision 5; tracked
    /// as a separate `Value::Map` ADR/PR).
    ListComprehension {
        /// Iteration variable, bound only inside `predicate` +
        /// `projection`.
        var: String,
        list: Box<Expression>,
        /// Optional WHERE filter (3VL: only `true` keeps the element).
        predicate: Option<Box<Expression>>,
        /// Optional `| projection`; absent ⇒ identity (project `var`).
        projection: Option<Box<Expression>>,
    },
    /// **ADR-191 D-6** (#620 map-half) — openCypher v9 §3.5 map projection
    /// `n{.key, .other, alias: expr, .*}`. Builds a NEW map by selecting
    /// keys from the node / relationship / map bound to `base`. Each
    /// [`MapProjectionItem`] contributes (or omits) one key per the D-6
    /// null-handling split: a `.key` property selector DROPS the key when
    /// the base's value is `null`/absent; an `alias: expr` literal entry
    /// KEEPS the key even when `expr` is `null`; `.*` includes every
    /// property of the base. Parsed from the dedicated `map_projection`
    /// grammar production (placed in `primary_atom` BEFORE `function_call`
    /// — both open with an identifier but commit on `{` vs `(`). The result
    /// is a [`crate::executor::Value::Map`] (ADR-191).
    MapProjection {
        /// The projected variable — a node / relationship / map. Grammar
        /// restricts the base to a bare identifier (openCypher's
        /// `MapProjection` base is a `Variable`).
        base: String,
        /// Selectors / literal entries, in source order. An empty `items`
        /// (`n{}`) projects the empty map.
        items: Vec<MapProjectionItem>,
    },
    /// **openCypher v9 §3.4** — list/string element access `base[index]`.
    /// Parsed from the postfix `index_accessor` grammar production (an
    /// `accessor` on an atom). 0-based; a negative index counts from the
    /// end; an out-of-range index evaluates to `null` (NOT an error).
    Subscript {
        base: Box<Expression>,
        index: Box<Expression>,
    },
    /// **openCypher v9 §3.4** — list slice `base[start..end]` (end
    /// exclusive). Either bound may be absent (`[..end]` ⇒ from 0;
    /// `[start..]` ⇒ to len; `[..]` ⇒ whole list); negative bounds count
    /// from the end; out-of-range bounds clamp (no error). Parsed from
    /// the postfix `slice_accessor` grammar production.
    Slice {
        base: Box<Expression>,
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
    },
    /// **openCypher v9 §3.6** — conditional `CASE` expression, both forms.
    ///
    /// - **SIMPLE** (`test = Some(..)`): evaluate `test`, then compare it for
    ///   openCypher VALUE equality against each branch's WHEN value in source
    ///   order; return the first matching THEN. A type-mismatched or
    ///   null-involving comparison simply does NOT match (it is NOT an error —
    ///   Conditional2 `1` compares `'0'` / `true` / `10.1` against integer
    ///   WHENs and falls through to ELSE).
    /// - **SEARCHED** (`test = None`): evaluate each branch's WHEN as a boolean
    ///   condition in source order under Cypher 3VL; return the first THEN
    ///   whose condition is TRUE. A `null` / `false` / non-boolean condition
    ///   does NOT match (the WHERE-filter 3VL discipline).
    ///
    /// If no branch matches, return `default` (the ELSE arm) or `Value::Null`
    /// when ELSE is absent. Parsed from the dedicated `case_expr` grammar
    /// production (`primary_atom`, before `function_call`). Evaluation
    /// short-circuits: only the matching branch's THEN (or the ELSE) is
    /// evaluated — non-taken THEN/ELSE arms are never evaluated.
    Case {
        /// `Some` ⇒ SIMPLE form (the test compared by equality against each
        /// WHEN value); `None` ⇒ SEARCHED form (each WHEN is a boolean cond).
        test: Option<Box<Expression>>,
        /// The `(WHEN, THEN)` arms in source order. Non-empty — the grammar's
        /// `(…)+` requires ≥1. In the simple form each WHEN is a value
        /// compared to `test`; in the searched form each is a boolean cond.
        branches: Vec<(Expression, Expression)>,
        /// The optional `ELSE` default; `None` ⇒ a non-match yields `null`.
        default: Option<Box<Expression>>,
    },
}

/// **#1290** — HAND-WRITTEN `Clone` with an iterative left-spine walk.
///
/// The derived `Clone` recursed once per tree level; a flat operator
/// chain folds into a left-nested spine up to
/// [`crate::parser::MAX_FLAT_CHAIN_DEPTH`] deep, and deep clones are
/// endemic on the query pipeline (the plan-cache key clones the whole
/// `Statement`; lowering clones projection items and predicates), so
/// the derive's per-level frames overflowed the native stack on
/// legitimate wide filters. This impl walks down the
/// `BinaryOp`/`UnaryOp`/`In`/`IsNull` spine collecting one frame per
/// level, clones the non-spine base via the ordinary per-variant
/// clone, then rebuilds bottom-up. `rhs`/non-spine children clone
/// recursively — they are never part of the LEFT spine, so their depth
/// is bounded by the bracket cap (`MAX_EXPRESSION_DEPTH`).
impl Clone for Expression {
    fn clone(&self) -> Self {
        enum SpineFrame<'a> {
            Binary { op: BinOp, rhs: &'a Expression },
            Unary { op: UnaryOp },
            In { rhs: &'a Expression },
            IsNull { negated: bool },
        }
        let mut frames: Vec<SpineFrame<'_>> = Vec::new();
        let mut cur = self;
        loop {
            match cur {
                Expression::BinaryOp { op, lhs, rhs } => {
                    frames.push(SpineFrame::Binary {
                        op: op.clone(),
                        rhs,
                    });
                    cur = lhs;
                }
                Expression::UnaryOp { op, operand } => {
                    frames.push(SpineFrame::Unary { op: op.clone() });
                    cur = operand;
                }
                Expression::In { lhs, rhs } => {
                    frames.push(SpineFrame::In { rhs });
                    cur = lhs;
                }
                Expression::IsNull { lhs, negated } => {
                    frames.push(SpineFrame::IsNull { negated: *negated });
                    cur = lhs;
                }
                _ => break,
            }
        }
        let mut acc = clone_non_spine_expression(cur);
        while let Some(frame) = frames.pop() {
            acc = match frame {
                SpineFrame::Binary { op, rhs } => Expression::BinaryOp {
                    op,
                    lhs: Box::new(acc),
                    rhs: Box::new(rhs.clone()),
                },
                SpineFrame::Unary { op } => Expression::UnaryOp {
                    op,
                    operand: Box::new(acc),
                },
                SpineFrame::In { rhs } => Expression::In {
                    lhs: Box::new(acc),
                    rhs: Box::new(rhs.clone()),
                },
                SpineFrame::IsNull { negated } => Expression::IsNull {
                    lhs: Box::new(acc),
                    negated,
                },
            };
        }
        acc
    }
}

/// The per-variant clone for every NON-spine [`Expression`] form — what
/// the derive would have generated, minus the four left-spine operator
/// arms (`BinaryOp` / `UnaryOp` / `In` / `IsNull`), which the iterative
/// driver in `Clone::clone` handles. Those four arms remain here as a
/// total-match fallback that re-enters `Clone` (the driver despines, so
/// no unbounded recursion is possible); they are unreachable from the
/// driver itself.
fn clone_non_spine_expression(e: &Expression) -> Expression {
    match e {
        Expression::BinaryOp { .. }
        | Expression::UnaryOp { .. }
        | Expression::In { .. }
        | Expression::IsNull { .. } => e.clone(),
        Expression::Literal(l) => Expression::Literal(l.clone()),
        Expression::Parameter(p) => Expression::Parameter(p.clone()),
        Expression::Identifier(n) => Expression::Identifier(n.clone()),
        Expression::PropertyAccess { base, path } => Expression::PropertyAccess {
            base: base.clone(),
            path: path.clone(),
        },
        Expression::FunctionCall {
            name,
            args,
            distinct,
            star,
        } => Expression::FunctionCall {
            name: name.clone(),
            args: args.clone(),
            distinct: *distinct,
            star: *star,
        },
        Expression::Near {
            lhs,
            target,
            vector_index,
        } => Expression::Near {
            lhs: lhs.clone(),
            target: target.clone(),
            vector_index: vector_index.clone(),
        },
        Expression::TextMatch { lhs, query } => Expression::TextMatch {
            lhs: lhs.clone(),
            query: query.clone(),
        },
        Expression::InCommunity { node, community } => Expression::InCommunity {
            node: node.clone(),
            community: community.clone(),
        },
        Expression::ListPredicate {
            quantifier,
            var,
            list,
            predicate,
        } => Expression::ListPredicate {
            quantifier: *quantifier,
            var: var.clone(),
            list: list.clone(),
            predicate: predicate.clone(),
        },
        Expression::Reduce {
            acc_var,
            init,
            var,
            list,
            expr,
        } => Expression::Reduce {
            acc_var: acc_var.clone(),
            init: init.clone(),
            var: var.clone(),
            list: list.clone(),
            expr: expr.clone(),
        },
        Expression::ListComprehension {
            var,
            list,
            predicate,
            projection,
        } => Expression::ListComprehension {
            var: var.clone(),
            list: list.clone(),
            predicate: predicate.clone(),
            projection: projection.clone(),
        },
        Expression::MapProjection { base, items } => Expression::MapProjection {
            base: base.clone(),
            items: items.clone(),
        },
        Expression::Subscript { base, index } => Expression::Subscript {
            base: base.clone(),
            index: index.clone(),
        },
        Expression::Slice { base, start, end } => Expression::Slice {
            base: base.clone(),
            start: start.clone(),
            end: end.clone(),
        },
        Expression::Case {
            test,
            branches,
            default,
        } => Expression::Case {
            test: test.clone(),
            branches: branches.clone(),
            default: default.clone(),
        },
    }
}

/// **ADR-191 D-6** (#620 map-half) — one element of a map projection
/// `n{ … }` (openCypher v9 §3.5). The D-6 null-handling split is encoded
/// in the variant choice, NOT a runtime flag: `Property` selectors drop
/// null/absent values, `Literal` entries keep them.
#[derive(Debug, Clone, PartialEq)]
pub enum MapProjectionItem {
    /// `.key` — include `key` with the base's `.key` value. **D-6: DROP
    /// the key if the value is `null`/absent.**
    Property(String),
    /// `.*` — include EVERY property of the base entity / map.
    AllProperties,
    /// `alias: expr` — include `alias` with `expr`'s value. **D-6: KEEP
    /// the key even when `expr` evaluates to `null`.**
    Literal {
        alias: String,
        value: Box<Expression>,
    },
}

/// **ADR-188** — the four openCypher v9 list-predicate quantifiers.
/// `All` = universal, `Any` = existential, `None` = negated
/// existential, `Single` = exactly-one (uniqueness). Each folds
/// through `executor::ThreeValued` per the Decision 4 truth table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantifier {
    All,
    Any,
    None,
    Single,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinOp {
    Eq,
    Neq,
    Lt,
    Le,
    Gt,
    Ge,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    And,
    Or,
    /// openCypher v9 boolean exclusive-disjunction (#621). Binds
    /// tighter than `Or`, looser than `And` (precedence ladder
    /// `OR → XOR → AND`). Evaluates through the same 3VL truth-table
    /// machinery as `And`/`Or` — see [`crate::executor::ThreeValued::xor`].
    Xor,
    /// openCypher v9 §3.3.6 string-comparison operators (#773). All three
    /// are binary `(String, String) -> Boolean` predicates parsed at the
    /// comparison-precedence tier (the `special_pred` / `expr_special_pred`
    /// grammar suffixes). Modeled as `BinOp` rather than a dedicated
    /// `Expression` variant because they are structurally identical to the
    /// other scalar comparisons (`Eq`/`Lt`/…): two operand expressions, a
    /// Boolean result — so they reuse the generic `BinaryOp` binder + bound
    /// plumbing. Semantics: prefix / suffix / substring match; any-operand
    /// null ⇒ null (3VL); non-string operand ⇒ null. Case-SENSITIVE,
    /// codepoint-correct (Rust `str::{starts_with,ends_with,contains}` are
    /// byte-based but UTF-8-correct for substring matching).
    StartsWith,
    EndsWith,
    Contains,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
    Pos,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldRef {
    /// `n.embedding` → base = "n", path = ["embedding"].
    pub base: String,
    pub path: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    List(Vec<Expression>),
    Map(Vec<(String, Expression)>),
    /// **W23-V11-T-01 / ADR-090** — `datetime('2026-05-24T12:00:00Z')`
    /// per ADR-038 amendment-09 + ADR-007 amendment-01.
    Temporal(arcgraph_core::ZonedDateTime),
    /// `localdatetime('2026-05-24T12:00:00')`.
    LocalDateTime(arcgraph_core::LocalDateTime),
    /// `date('2026-05-24')`.
    Date(arcgraph_core::Date),
    /// `duration('PT1H30M')` per ISO-8601.
    Duration(arcgraph_core::Duration),
    /// `decimal('100.50')`.
    Decimal(arcgraph_core::Decimal),
}

/// Subset of `Literal` that fusion weights accept (so the weight
/// printer never emits a string / list).
#[derive(Debug, Clone, PartialEq)]
pub enum NumericLiteral {
    Integer(i64),
    Float(f64),
}

// =====================================================================
// 8. Index DDL
// =====================================================================

/// Neo4j-compatible index DDL emitted by vector clients (#830). The
/// grammar parses + the
/// binder binds these (label / property captured); the executor path
/// surfaces a typed [`crate::semantic::ArcQLError::NotImplemented`] —
/// the index BUILD / lifecycle is the vector track's follow-up per
/// ADR-198 §OQ-7. This is the mgr-dev half of the OQ-7 split (grammar +
/// proc-registration); the substrate binding is the vector track's.
#[derive(Debug, Clone, PartialEq)]
pub enum IndexDdlStatement {
    /// `CREATE VECTOR INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
    /// var.prop [OPTIONS { … }]`.
    CreateVector(CreateVectorIndexStatement),
    /// `CREATE INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
    /// (var.prop)` — the user-visible secondary node-property index
    /// (#1366, task #248). Distinct from `CreateVector`: no `VECTOR`
    /// keyword, no OPTIONS.
    CreateProperty(CreatePropertyIndexStatement),
    /// `DROP INDEX <name> [IF EXISTS]` — the generic Neo4j drop form
    /// (drops any index by name; Neo4j has no `DROP VECTOR INDEX`).
    Drop(DropIndexStatement),
}

/// An index-name reference: a literal name or a `$param`.
/// Neo4j-compatible clients may pass the index name as a query
/// parameter (`CREATE VECTOR INDEX $name …`); a literal name is admitted
/// (the issue #830 raw-Cypher repro + Neo4j docs form).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IndexNameRef {
    /// A literal index name (bare or backtick-escaped identifier).
    Literal(String),
    /// A `$param` index name (the value supplied at execution time).
    Param(String),
}

/// `CREATE VECTOR INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
/// var.prop [OPTIONS { … }]` (#830). Matches the common
/// Neo4j-compatible wire form.
#[derive(Debug, Clone, PartialEq)]
pub struct CreateVectorIndexStatement {
    /// The index name (`$param` or literal).
    pub name: IndexNameRef,
    /// `IF NOT EXISTS` present (idempotent create).
    pub if_not_exists: bool,
    /// The pattern variable in `FOR (var:Label)` (e.g. `n`).
    pub pattern_var: String,
    /// The node label in `FOR (var:Label)` (e.g. `Chunk`).
    pub label: String,
    /// The indexed property path in `ON var.prop` (e.g. `embedding`;
    /// the segment(s) after the pattern variable).
    pub property: String,
    /// The raw `OPTIONS { … }` map, captured verbatim as a parsed
    /// expression (index-build config for the vector track — NOT
    /// interpreted here per ADR-198 §OQ-7). `None` when no OPTIONS.
    pub options: Option<Expression>,
}

/// `CREATE INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON (var.prop)`
/// (#1366, task #248) — the user-visible secondary node-property index.
/// No OPTIONS, no `VECTOR` keyword.
#[derive(Debug, Clone, PartialEq)]
pub struct CreatePropertyIndexStatement {
    /// The index name (`$param` or literal).
    pub name: IndexNameRef,
    /// `IF NOT EXISTS` present (idempotent create).
    pub if_not_exists: bool,
    /// The pattern variable in `FOR (var:Label)` (e.g. `n`).
    pub pattern_var: String,
    /// The node label in `FOR (var:Label)` (e.g. `User`).
    pub label: String,
    /// The indexed property path in `ON (var.prop)` (the segment(s)
    /// after the pattern variable, e.g. `email`).
    pub property: String,
}

/// `DROP INDEX <name> [IF EXISTS]` (#830) — the generic Neo4j form
/// emitted by vector clients.
#[derive(Debug, Clone, PartialEq)]
pub struct DropIndexStatement {
    /// The index name to drop (`$param` or literal).
    pub name: IndexNameRef,
    /// `IF EXISTS` present (no error if the index is absent).
    pub if_exists: bool,
}

// =====================================================================
// 9. Display impls — round-trip pretty-printer
// =====================================================================
//
// Discipline: every Display impl emits text that re-parses to a
// structurally equal AST. The 256-case proptest in
// `tests/grammar_proptest.rs` enforces this. When in doubt, prefer
// a canonical form (e.g. ALWAYS upper-case keywords, ALWAYS double-
// quote strings) so the printer is deterministic.

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Read(q) => write!(f, "{q}"),
            Statement::IndexDdl(d) => write!(f, "{d}"),
            Statement::Explain(q) => write!(f, "EXPLAIN {q}"),
            Statement::Profile(q) => write!(f, "PROFILE {q}"),
            Statement::Union(u) => write!(f, "{u}"),
        }
    }
}

impl fmt::Display for UnionQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // arm0 UNION [ALL] arm1 UNION [ALL] arm2 … <tail>
        for (i, arm) in self.arms.iter().enumerate() {
            if i > 0 {
                write!(f, " UNION")?;
                // `all[i-1]` is the boundary BEFORE `arms[i]`.
                if self.all.get(i - 1).copied().unwrap_or(false) {
                    write!(f, " ALL")?;
                }
                write!(f, " ")?;
            }
            write!(f, "{arm}")?;
        }
        if !self.tail.order_by.is_empty() {
            write!(f, " ORDER BY ")?;
            for (i, o) in self.tail.order_by.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{o}")?;
            }
        }
        if let Some(skip) = &self.tail.skip {
            write!(f, " SKIP {skip}")?;
        }
        if let Some(limit) = &self.tail.limit {
            write!(f, " LIMIT {limit}")?;
        }
        Ok(())
    }
}

impl fmt::Display for ReadQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, c) in self.clauses.iter().enumerate() {
            if i > 0 {
                write!(f, " ")?;
            }
            write!(f, "{c}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Clause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Clause::Match(c) => write!(f, "{c}"),
            Clause::OptionalMatch(c) => write!(f, "OPTIONAL {c}"),
            Clause::Create(c) => write!(f, "{c}"),
            Clause::Delete(c) => write!(f, "{c}"),
            Clause::Set(c) => write!(f, "{c}"),
            Clause::Remove(c) => write!(f, "{c}"),
            Clause::Merge(c) => write!(f, "{c}"),
            Clause::With(c) => write!(f, "{c}"),
            Clause::Unwind(c) => write!(f, "{c}"),
            Clause::Call(c) => write!(f, "{c}"),
            Clause::CallProcedure(c) => {
                write!(f, "CALL {}(", c.name)?;
                for (i, a) in c.args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")?;
                if !c.yield_items.is_empty() {
                    write!(f, " YIELD ")?;
                    for (i, (col, alias)) in c.yield_items.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{col}")?;
                        if let Some(a) = alias {
                            write!(f, " AS {a}")?;
                        }
                    }
                    if let Some(w) = &c.where_clause {
                        write!(f, " WHERE {w}")?;
                    }
                }
                Ok(())
            }
            Clause::Show(c) => {
                let kind = match c.kind {
                    ShowKind::Constraints => "CONSTRAINTS",
                    ShowKind::Indexes => "INDEXES",
                    ShowKind::Databases => "DATABASES",
                    ShowKind::VectorIndexes => "VECTOR INDEXES",
                };
                write!(f, "SHOW {kind}")?;
                if !c.yield_items.is_empty() {
                    write!(f, " YIELD ")?;
                    for (i, (col, alias)) in c.yield_items.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        match alias {
                            Some(a) => write!(f, "{col} AS {a}")?,
                            None => write!(f, "{col}")?,
                        }
                    }
                }
                if let Some(w) = &c.where_clause {
                    write!(f, " WHERE {w}")?;
                }
                Ok(())
            }
            Clause::RankBy(c) => write!(f, "{c}"),
            Clause::WithFusion(c) => write!(f, "{c}"),
            Clause::Return(c) => write!(f, "{c}"),
            Clause::TailOrderBy(items) => {
                write!(f, "ORDER BY ")?;
                for (i, o) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{o}")?;
                }
                Ok(())
            }
            Clause::TailSkip(e) => write!(f, "SKIP {e}"),
            Clause::TailLimit(e) => write!(f, "LIMIT {e}"),
        }
    }
}

impl fmt::Display for CallClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CALL {{ {} }}", self.body)
    }
}

impl fmt::Display for CreateClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE ")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl fmt::Display for CreateItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CreateItem::Node(n) => write!(f, "{n}"),
            CreateItem::Path(p) => write!(f, "{p}"),
        }
    }
}

impl fmt::Display for CreatePathSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}{}", self.source, self.rel, self.target)
    }
}

impl fmt::Display for CreateRelSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (left, right) = match self.direction {
            CreateRelDirection::LeftToRight => ("-", "->"),
            CreateRelDirection::RightToLeft => ("<-", "-"),
        };
        write!(f, "{left}[")?;
        if let Some(v) = &self.var {
            write!(f, "{v}")?;
        }
        write!(f, ":{}", self.label)?;
        if let Some(p) = &self.properties {
            write!(f, " {p}")?;
        }
        write!(f, "]{right}")
    }
}

impl fmt::Display for DeleteClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.detach {
            write!(f, "DETACH ")?;
        }
        write!(f, "DELETE ")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl fmt::Display for DeleteItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.var)
    }
}

impl fmt::Display for SetClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SET ")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl fmt::Display for SetItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.mutation {
            SetMutation::PropertyAssign { name, value } => {
                write!(f, "{}.{} = {}", self.var, name, value)
            }
            SetMutation::PropertyReplace(map) => {
                write!(f, "{} = {}", self.var, map)
            }
            SetMutation::PropertyMerge(map) => {
                write!(f, "{} += {}", self.var, map)
            }
            SetMutation::LabelAdd(labels) => {
                write!(f, "{}", self.var)?;
                for l in labels {
                    write!(f, ":{l}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for RemoveClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "REMOVE ")?;
        for (i, item) in self.items.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{item}")?;
        }
        Ok(())
    }
}

impl fmt::Display for RemoveItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.mutation {
            RemoveMutation::Property(name) => write!(f, "{}.{name}", self.var),
            RemoveMutation::LabelRemove(labels) => {
                write!(f, "{}", self.var)?;
                for l in labels {
                    write!(f, ":{l}")?;
                }
                Ok(())
            }
        }
    }
}

impl fmt::Display for MergeClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MERGE {}", self.pattern)?;
        if !self.on_create.is_empty() {
            write!(f, " ON CREATE SET ")?;
            for (i, item) in self.on_create.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{item}")?;
            }
        }
        if !self.on_match.is_empty() {
            write!(f, " ON MATCH SET ")?;
            for (i, item) in self.on_match.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{item}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for MergePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MergePattern::Node(n) => write!(f, "{n}"),
            MergePattern::Path(p) => write!(f, "{p}"),
        }
    }
}

impl fmt::Display for CreateNodeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        if let Some(v) = &self.var {
            write!(f, "{v}")?;
        }
        if let Some(l) = &self.label {
            write!(f, ":{l}")?;
        }
        if let Some(p) = &self.properties {
            if self.var.is_some() || self.label.is_some() {
                write!(f, " ")?;
            }
            write!(f, "{p}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for MatchClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MATCH {}", self.body)?;
        if let Some(w) = &self.where_clause {
            write!(f, " WHERE {w}")?;
        }
        Ok(())
    }
}

impl fmt::Display for MatchBody {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatchBody::Patterns(ps) => {
                for (i, p) in ps.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{p}")?;
                }
                Ok(())
            }
            MatchBody::NamedPath(np) => write!(f, "{np}"),
        }
    }
}

impl fmt::Display for NamedPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            NamedPathKind::ShortestPath(p) => {
                write!(f, "{} = shortestPath({})", self.var, p)
            }
            NamedPathKind::AllShortestPath(p) => {
                write!(f, "{} = allShortestPaths({})", self.var, p)
            }
            NamedPathKind::Plain(p) => write!(f, "{} = {}", self.var, p),
        }
    }
}

impl fmt::Display for PathPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.head)?;
        for (rel, node) in &self.tail {
            write!(f, "{rel}{node}")?;
        }
        Ok(())
    }
}

impl fmt::Display for NodePattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "(")?;
        if let Some(v) = &self.var {
            write!(f, "{v}")?;
        }
        for l in &self.labels {
            write!(f, ":{l}")?;
        }
        if let Some(p) = &self.properties {
            if self.var.is_some() || !self.labels.is_empty() {
                write!(f, " ")?;
            }
            write!(f, "{p}")?;
        }
        write!(f, ")")
    }
}

impl fmt::Display for RelPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (left, right) = match self.direction {
            RelDirection::LeftToRight => ("-", "->"),
            RelDirection::RightToLeft => ("<-", "-"),
            RelDirection::Undirected => ("-", "-"),
        };
        write!(f, "{left}")?;

        // openCypher `*N..M` lives INSIDE the rel_detail brackets;
        // GQL `{N,M}` lives OUTSIDE, AFTER the trailing arrow.
        // We split the length-range printer accordingly.
        let length_inside = matches!(
            self.length,
            Some(LengthRange::Cypher { .. }) | Some(LengthRange::Unbounded)
        );
        let length_outside = matches!(self.length, Some(LengthRange::Quantified { .. }));

        let has_detail = self.var.is_some()
            || !self.rel_types.is_empty()
            || length_inside
            || self.properties.is_some();
        if has_detail {
            write!(f, "[")?;
            if let Some(v) = &self.var {
                write!(f, "{v}")?;
            }
            for (i, t) in self.rel_types.iter().enumerate() {
                if i == 0 {
                    write!(f, ":{t}")?;
                } else {
                    write!(f, "|{t}")?;
                }
            }
            if length_inside {
                if let Some(l) = &self.length {
                    write!(f, "{l}")?;
                }
            }
            if let Some(p) = &self.properties {
                write!(f, " {p}")?;
            }
            write!(f, "]")?;
        }
        write!(f, "{right}")?;
        if length_outside {
            if let Some(l) = &self.length {
                write!(f, "{l}")?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for LengthRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LengthRange::Unbounded => write!(f, "*"),
            LengthRange::Cypher { min, max } => match max {
                Some(m) => write!(f, "*{min}..{m}"),
                None => write!(f, "*{min}.."),
            },
            LengthRange::Quantified { min, max } => match max {
                Some(m) => write!(f, "{{{min},{m}}}"),
                None => write!(f, "{{{min},}}"),
            },
        }
    }
}

impl fmt::Display for PropertyMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        for (i, (k, v)) in self.entries.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{k}: {v}")?;
        }
        write!(f, "}}")
    }
}

impl fmt::Display for WithClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WITH ")?;
        if self.distinct {
            write!(f, "DISTINCT ")?;
        }
        write_projection_list(f, &self.items)?;
        if let Some(w) = &self.where_clause {
            write!(f, " WHERE {w}")?;
        }
        Ok(())
    }
}

impl fmt::Display for UnwindClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "UNWIND {} AS {}", self.expr, self.var)
    }
}

fn write_projection_list(f: &mut fmt::Formatter<'_>, items: &[ProjectionItem]) -> fmt::Result {
    for (i, p) in items.iter().enumerate() {
        if i > 0 {
            write!(f, ", ")?;
        }
        write!(f, "{p}")?;
    }
    Ok(())
}

impl fmt::Display for ProjectionItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.kind {
            ProjectionKind::Wildcard => write!(f, "*")?,
            ProjectionKind::Expr(e) => write!(f, "{e}")?,
        }
        if let Some(a) = &self.alias {
            write!(f, " AS {a}")?;
        }
        Ok(())
    }
}

impl fmt::Display for RankByClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RANK BY {}", self.ranker)?;
        if let Some(alias) = &self.score_alias {
            write!(f, " AS {alias}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Ranker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ranker::Hybrid(args) => {
                write!(f, "HYBRID(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for RankArg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RankArg::Vector { field, query, k } => {
                write!(f, "VECTOR({field}, {query}")?;
                if let Some(kv) = k {
                    write!(f, ", K = {kv}")?;
                }
                write!(f, ")")
            }
            RankArg::Text { field, query, k } => {
                write!(f, "TEXT({field}, {query}")?;
                if let Some(kv) = k {
                    write!(f, ", K = {kv}")?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for WithFusionClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "WITH FUSION = {}", self.fusion)
    }
}

impl fmt::Display for Fusion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Fusion::Rrf { k } => write!(f, "RRF(k = {k})"),
        }
    }
}

impl fmt::Display for NumericLiteral {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumericLiteral::Integer(i) => write!(f, "{i}"),
            NumericLiteral::Float(x) => write_float(f, *x),
        }
    }
}

impl fmt::Display for ReturnClause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RETURN ")?;
        if self.distinct {
            write!(f, "DISTINCT ")?;
        }
        write_projection_list(f, &self.items)?;
        if !self.order_by.is_empty() {
            write!(f, " ORDER BY ")?;
            for (i, o) in self.order_by.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{o}")?;
            }
        }
        if let Some(s) = &self.skip {
            write!(f, " SKIP {s}")?;
        }
        if let Some(l) = &self.limit {
            write!(f, " LIMIT {l}")?;
        }
        Ok(())
    }
}

impl fmt::Display for OrderItem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.expr)?;
        match self.direction {
            OrderDirection::Asc => write!(f, " ASC"),
            OrderDirection::Desc => write!(f, " DESC"),
            OrderDirection::Default => Ok(()),
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // #1290 — render the left-nested operator SPINE iteratively.
        // Flat operator chains (`a AND b AND …`, `1 + 2 + …`,
        // `x IN l IN …`, `x IS NULL IS NULL …`) fold into a left-nested
        // spine up to `parser::MAX_FLAT_CHAIN_DEPTH` deep; the previous
        // `write!(f, "({lhs} {op} {rhs})")` recursion burned several
        // native fmt-machinery frames per level and overflowed the
        // stack on legitimate wide filters (this Display feeds the
        // plan-cache key bytestream and UNION column names, so it is
        // reachable from `QueryEngine::execute`). We walk down the
        // spine emitting each level's opening prefix, render the
        // non-spine base via the ordinary per-variant formatter, then
        // pop each level's ` <op> <rhs>)` suffix — byte-identical to
        // the recursive rendering (the grammar round-trip proptest
        // pins it). `rhs` sub-expressions are rendered recursively;
        // they are never part of the LEFT spine, so their depth is
        // bounded by the bracket cap (`MAX_EXPRESSION_DEPTH`), not the
        // chain cap.
        enum SpineSuffix<'a> {
            /// ` <op> <rhs>)` for a binary operator level.
            Binary { op: &'a BinOp, rhs: &'a Expression },
            /// ` IN <rhs>)` for a list-membership level.
            In { rhs: &'a Expression },
            /// ` IS NULL)` / ` IS NOT NULL)` for a null-test level.
            IsNull { negated: bool },
            /// A bare `)` (unary levels emit their operator in the
            /// prefix).
            Close,
        }
        let mut suffixes: Vec<SpineSuffix<'_>> = Vec::new();
        let mut cur = self;
        loop {
            match cur {
                Expression::BinaryOp { op, lhs, rhs } => {
                    // Always parenthesize binary ops to keep the printer
                    // associativity-neutral. The `_` identifier wrapping
                    // is benign at parse time and preserved by Display.
                    f.write_str("(")?;
                    suffixes.push(SpineSuffix::Binary { op, rhs });
                    cur = lhs;
                }
                Expression::UnaryOp { op, operand } => {
                    f.write_str(match op {
                        UnaryOp::Not => "(NOT ",
                        UnaryOp::Neg => "(-",
                        UnaryOp::Pos => "(+",
                    })?;
                    suffixes.push(SpineSuffix::Close);
                    cur = operand;
                }
                Expression::In { lhs, rhs } => {
                    f.write_str("(")?;
                    suffixes.push(SpineSuffix::In { rhs });
                    cur = lhs;
                }
                Expression::IsNull { lhs, negated } => {
                    f.write_str("(")?;
                    suffixes.push(SpineSuffix::IsNull { negated: *negated });
                    cur = lhs;
                }
                other => {
                    fmt_expression_non_spine(other, f)?;
                    break;
                }
            }
        }
        while let Some(s) = suffixes.pop() {
            match s {
                SpineSuffix::Binary { op, rhs } => write!(f, " {op} {rhs})")?,
                SpineSuffix::In { rhs } => write!(f, " IN {rhs})")?,
                SpineSuffix::IsNull { negated: true } => f.write_str(" IS NOT NULL)")?,
                SpineSuffix::IsNull { negated: false } => f.write_str(" IS NULL)")?,
                SpineSuffix::Close => f.write_str(")")?,
            }
        }
        Ok(())
    }
}

/// The per-variant rendering for every NON-spine [`Expression`] form —
/// the body of the pre-#1290 recursive `Display` match minus the four
/// left-spine operator arms (`BinaryOp` / `UnaryOp` / `In` / `IsNull`),
/// which the iterative driver in `Display::fmt` handles. Those four
/// arms remain here as a total-match fallback that simply re-enters
/// `Display` (the driver despines, so no unbounded recursion is
/// possible); they are unreachable from the driver itself.
fn fmt_expression_non_spine(e: &Expression, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match e {
        // Spine variants — handled by the iterative driver in
        // `Display::fmt`; kept total (not `unreachable!`) per the
        // no-panic discipline.
        Expression::BinaryOp { .. }
        | Expression::UnaryOp { .. }
        | Expression::In { .. }
        | Expression::IsNull { .. } => write!(f, "{e}"),
        Expression::Literal(l) => write!(f, "{l}"),
        Expression::Parameter(p) => write!(f, "${p}"),
        Expression::Identifier(n) => write!(f, "{n}"),
        Expression::PropertyAccess { base, path } => {
            write!(f, "{base}")?;
            for p in path {
                write!(f, ".{p}")?;
            }
            Ok(())
        }
        Expression::FunctionCall {
            name,
            args,
            distinct,
            star,
        } => {
            // Round-trip faithful (the plan-cache key is this Display
            // rendering — `count(*)` / `count(DISTINCT x)` / `count(x)`
            // MUST render distinctly so they never collide in cache).
            write!(f, "{name}(")?;
            if *star {
                write!(f, "*")?;
            } else {
                if *distinct {
                    write!(f, "DISTINCT ")?;
                }
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
            }
            write!(f, ")")
        }
        Expression::Near {
            lhs,
            target,
            vector_index,
        } => {
            write!(f, "({lhs} NEAR {target}")?;
            if let Some(v) = vector_index {
                write!(f, " VECTOR_INDEX {v}")?;
            }
            write!(f, ")")
        }
        Expression::TextMatch { lhs, query } => {
            write!(f, "({lhs} MATCH {query})")
        }
        Expression::InCommunity { node, community } => {
            write!(f, "({node} IN COMMUNITY({community}))")
        }
        // ADR-188 — list-predicate special forms. Display emits the
        // canonical openCypher form so `parse(format!("{e}")) == e`
        // (the round-trip property the grammar proptest pins).
        Expression::ListPredicate {
            quantifier,
            var,
            list,
            predicate,
        } => {
            let kw = match quantifier {
                Quantifier::All => "all",
                Quantifier::Any => "any",
                Quantifier::None => "none",
                Quantifier::Single => "single",
            };
            write!(f, "{kw}({var} IN {list} WHERE {predicate})")
        }
        Expression::Reduce {
            acc_var,
            init,
            var,
            list,
            expr,
        } => {
            write!(f, "reduce({acc_var} = {init}, {var} IN {list} | {expr})")
        }
        // ADR-188 (#620 list-half) — list comprehension. Display
        // emits the canonical openCypher v9 §3.5 form so
        // `parse(format!("{e}")) == e` (the round-trip property the
        // grammar proptest pins), with the WHERE filter and the
        // `| projection` emitted only when present (each is
        // grammar-optional and the four combinations must all
        // round-trip).
        Expression::ListComprehension {
            var,
            list,
            predicate,
            projection,
        } => {
            write!(f, "[{var} IN {list}")?;
            if let Some(p) = predicate {
                write!(f, " WHERE {p}")?;
            }
            if let Some(e) = projection {
                write!(f, " | {e}")?;
            }
            write!(f, "]")
        }
        // ADR-191 D-6 (#620 map-half) — map projection. Display emits
        // the canonical openCypher v9 §3.5 form so
        // `parse(format!("{e}")) == e` (the round-trip property). Items
        // are comma-joined in source order; the base prefixes the brace
        // with no space (`n{.a, .b}`).
        Expression::MapProjection { base, items } => {
            write!(f, "{base}{{")?;
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                match item {
                    MapProjectionItem::Property(k) => write!(f, ".{k}")?,
                    MapProjectionItem::AllProperties => write!(f, ".*")?,
                    MapProjectionItem::Literal { alias, value } => write!(f, "{alias}: {value}")?,
                }
            }
            write!(f, "}}")
        }
        // openCypher v9 §3.4 — postfix accessors. Display emits the
        // canonical `base[..]` form so `parse(format!("{e}")) == e`
        // (the round-trip property the grammar proptest pins).
        Expression::Subscript { base, index } => {
            write!(f, "{base}[{index}]")
        }
        Expression::Slice { base, start, end } => {
            write!(f, "{base}[")?;
            if let Some(s) = start {
                write!(f, "{s}")?;
            }
            write!(f, "..")?;
            if let Some(e) = end {
                write!(f, "{e}")?;
            }
            write!(f, "]")
        }
        // openCypher v9 §3.6 — CASE expression. Display emits the
        // canonical form so `parse(format!("{e}")) == e` (the round-trip
        // property the grammar proptest pins). The leading `test` is
        // emitted only for the simple form (absent ⇒ searched); each
        // `(when, then)` arm + the optional `ELSE` follow in source
        // order, closed by `END`. The plan-cache key relies on this
        // bytestream to distinguish structurally-different CASE
        // expressions (#621 cache-key requirement).
        Expression::Case {
            test,
            branches,
            default,
        } => {
            write!(f, "CASE")?;
            if let Some(t) = test {
                write!(f, " {t}")?;
            }
            for (when, then) in branches {
                write!(f, " WHEN {when} THEN {then}")?;
            }
            if let Some(d) = default {
                write!(f, " ELSE {d}")?;
            }
            write!(f, " END")
        }
    }
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            BinOp::Eq => "=",
            BinOp::Neq => "<>",
            BinOp::Lt => "<",
            BinOp::Le => "<=",
            BinOp::Gt => ">",
            BinOp::Ge => ">=",
            BinOp::Add => "+",
            BinOp::Sub => "-",
            BinOp::Mul => "*",
            BinOp::Div => "/",
            BinOp::Mod => "%",
            BinOp::Pow => "^",
            BinOp::And => "AND",
            BinOp::Or => "OR",
            BinOp::Xor => "XOR",
            // openCypher v9 §3.3.6 string predicates (#773). Canonical
            // uppercase spelling (round-trips through the parser).
            BinOp::StartsWith => "STARTS WITH",
            BinOp::EndsWith => "ENDS WITH",
            BinOp::Contains => "CONTAINS",
        };
        write!(f, "{s}")
    }
}

impl fmt::Display for FieldRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base)?;
        for p in &self.path {
            write!(f, ".{p}")?;
        }
        Ok(())
    }
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Literal::Null => write!(f, "NULL"),
            Literal::Bool(b) => write!(f, "{}", if *b { "TRUE" } else { "FALSE" }),
            Literal::Integer(i) => write!(f, "{i}"),
            Literal::Float(x) => write_float(f, *x),
            Literal::String(s) => write!(f, "{}", quote_string(s)),
            Literal::List(xs) => {
                write!(f, "[")?;
                for (i, x) in xs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{x}")?;
                }
                write!(f, "]")
            }
            Literal::Map(es) => {
                write!(f, "{{")?;
                for (i, (k, v)) in es.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
            // W23-V11-T-01 / ADR-090 — temporal + decimal literal
            // round-trip discipline: Display emits the canonical
            // constructor form so `parse(format!("{l}")) == l`.
            // The 256-case proptest in `tests/grammar_proptest.rs`
            // pins this property — and now applies to temporal /
            // decimal literals as well.
            Literal::Temporal(t) => write!(f, "datetime('{t}')"),
            Literal::LocalDateTime(ldt) => write!(f, "localdatetime('{ldt}')"),
            Literal::Date(d) => write!(f, "date('{d}')"),
            Literal::Duration(d) => write!(f, "duration('{d}')"),
            Literal::Decimal(d) => write!(f, "decimal('{d}')"),
        }
    }
}

impl fmt::Display for IndexNameRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexNameRef::Literal(s) => write!(f, "{s}"),
            IndexNameRef::Param(p) => write!(f, "${p}"),
        }
    }
}

impl fmt::Display for IndexDdlStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndexDdlStatement::CreateVector(c) => write!(f, "{c}"),
            IndexDdlStatement::CreateProperty(c) => write!(f, "{c}"),
            IndexDdlStatement::Drop(d) => write!(f, "{d}"),
        }
    }
}

impl fmt::Display for CreatePropertyIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE INDEX {}", self.name)?;
        if self.if_not_exists {
            write!(f, " IF NOT EXISTS")?;
        }
        write!(
            f,
            " FOR ({}:{}) ON ({}.{})",
            self.pattern_var, self.label, self.pattern_var, self.property
        )
    }
}

impl fmt::Display for CreateVectorIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CREATE VECTOR INDEX {}", self.name)?;
        if self.if_not_exists {
            write!(f, " IF NOT EXISTS")?;
        }
        write!(
            f,
            " FOR ({}:{}) ON {}.{}",
            self.pattern_var, self.label, self.pattern_var, self.property
        )?;
        if let Some(opts) = &self.options {
            write!(f, " OPTIONS {opts}")?;
        }
        Ok(())
    }
}

impl fmt::Display for DropIndexStatement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DROP INDEX {}", self.name)?;
        if self.if_exists {
            write!(f, " IF EXISTS")?;
        }
        Ok(())
    }
}

// =====================================================================
// 10. Helpers
// =====================================================================

/// Escape and double-quote a string literal, matching the
/// `string_literal` grammar production. Always emits double quotes
/// for canonical form (re-parses cleanly via `inner_dq`).
fn quote_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            // Other control characters → \uXXXX
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Print a float in a form `float_literal` will re-tokenize. Pest's
/// `float_literal` requires a digit on each side of the decimal
/// point, so `0.0` (not `0.`) is the canonical form.
fn write_float(f: &mut fmt::Formatter<'_>, x: f64) -> fmt::Result {
    if x.is_nan() {
        // The grammar has no NaN literal; the printer would round-
        // trip a NaN as the bareword "NaN" which is illegal. We
        // refuse rather than silently mis-print. NaN cannot reach
        // here from a parsed query (it has no syntax); a downstream
        // analyzer that constructs an AST with NaN must handle the
        // print failure.
        return Err(fmt::Error);
    }
    if x.is_infinite() {
        // Same reasoning as NaN.
        return Err(fmt::Error);
    }
    let s = format!("{x:?}");
    // Rust's `Debug` for f64 always produces a `.` even for whole
    // numbers (`1.0` not `1`), which matches our `float_literal`.
    f.write_str(&s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quote_string_escapes_backslash_and_quote() {
        assert_eq!(quote_string("hi"), "\"hi\"");
        assert_eq!(quote_string("a\\b"), "\"a\\\\b\"");
        assert_eq!(quote_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(quote_string("a\nb"), "\"a\\nb\"");
    }

    #[test]
    fn float_print_is_round_trippable() {
        // The Display for Literal::Float must produce a string the
        // grammar will re-parse as a float (not an int).
        let cases = [0.0_f64, 1.5, -3.25, 1e10, -1e-10];
        for x in cases {
            let printed = format!("{}", Literal::Float(x));
            assert!(
                printed.contains('.') || printed.contains('e') || printed.contains('E'),
                "float {x} printed as `{printed}` — must contain `.` or exponent",
            );
        }
    }

    #[test]
    fn rel_pattern_prints_canonical_directed_form() {
        let r = RelPattern {
            var: Some("r".into()),
            rel_types: vec!["KNOWS".into()],
            direction: RelDirection::LeftToRight,
            length: Some(LengthRange::Cypher {
                min: 1,
                max: Some(3),
            }),
            properties: None,
        };
        let s = format!("{r}");
        assert_eq!(s, "-[r:KNOWS*1..3]->");
    }

    #[test]
    fn shortest_path_prints_named_form() {
        // ADR-194 D-1/D-3 — the canonical Display spelling is the
        // openCypher camelCase `shortestPath(...)` (the `SHORTEST_PATH`
        // macro remains an accepted INPUT alias, but canonical OUTPUT is
        // camelCase). Re-parsing this canonical form yields the SAME
        // `ShortestPath` AST (covered by `parser_smoke::round_trip_*`).
        let np = NamedPath {
            var: "p".into(),
            kind: NamedPathKind::ShortestPath(PathPattern {
                head: NodePattern {
                    var: Some("a".into()),
                    labels: vec![],
                    properties: None,
                },
                tail: vec![],
            }),
        };
        let s = format!("{np}");
        assert_eq!(s, "p = shortestPath((a))");
    }

    #[test]
    fn all_shortest_path_prints_named_form() {
        // ADR-194 D-2 — canonical `allShortestPaths(...)` Display.
        let np = NamedPath {
            var: "p".into(),
            kind: NamedPathKind::AllShortestPath(PathPattern {
                head: NodePattern {
                    var: Some("a".into()),
                    labels: vec![],
                    properties: None,
                },
                tail: vec![],
            }),
        };
        let s = format!("{np}");
        assert_eq!(s, "p = allShortestPaths((a))");
    }
}
