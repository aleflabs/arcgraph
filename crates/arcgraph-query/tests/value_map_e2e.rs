//! **ADR-191 / #620 (map-half foundation)** — `Value::Map` END-TO-END
//! tests through the FULL query pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute).
//!
//! These complement the direct-evaluator strong-oracle unit tests in
//! `src/executor/{eval,value,budget}.rs` and `src/executor/ops/*.rs`
//! (which pin the value-level semantics — D-2 literal eval, D-3 3VL
//! equality, D-4 comparability, D-5 orderability, D-7 JSON, D-8 access,
//! D-11 write fence, D-12 keying, D-13 memory — at the strongest oracle).
//! Here we prove the grammar production, binder, type-check, lowering,
//! and the executor all compose correctly through a real query.
//!
//! All oracles are STRONG `==` over the result rows.
//!
//! # Why MATCH-wrapped, not bare `RETURN {…}`
//!
//! At v1.0-alpha a `RETURN`-only statement lowers to `Project(Empty)` and
//! `EmptyOp` emits ZERO rows (a bare `RETURN 1` yields no rows — see
//! `w28_conformance_scalar_fns_e2e.rs`). So every projection is wrapped
//! in `MATCH (nd:X)` over an N-node fixture; the map expression in the
//! RETURN list is what we assert over. (`count(*)` is likewise spelled
//! `count(nd)` — the v1.0 surrogate per `aggregation_lowering_integration`.)
//!
//! # The bug fix this slice closes (D-2)
//!
//! `RETURN {a: 1, b: 2}` evaluated to `null` on `main` (the evaluator
//! threw away the parsed map — a silent wrong answer). After this slice
//! it returns a real map. `map_literal_returns_a_map` is the regression
//! that FAILS on `main`.
//!
//! # Out of scope (PR-B / sister PRs)
//!
//! - Map PROJECTION `n{.name, .age}` and map COMPREHENSION `{…}` consume
//!   `Value::Map`; they are PR-B (deferred).
//! - Dynamic subscript `m['key']` needs the `[expr]` accessor grammar,
//!   sister PR #621's scope (the grammar has only `.identifier` today).
//!   The `map.key` dot form (tested here) shares the same resolution.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, Pipeline, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

const LABEL_X: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["g", "p"])
}

fn node(id: u64) -> NodeView {
    NodeView::new(NodeId::new(id), Some(LabelId::new(LABEL_X)))
}

/// `n` nodes of label X (the MATCH-supplied driving rows).
fn n_nodes(n: u64) -> StubExecutorSubstrate {
    let mut s = StubExecutorSubstrate::new();
    for i in 1..=n {
        s = s.with_node(
            TenantId::DEFAULT,
            node(i).with_property("g", Value::Integer(i as i64)),
        );
    }
    s
}

/// Full read pipeline → rows. Asserts success at every stage.
fn run(query: &str, s: &StubExecutorSubstrate, c: &StubCatalogProvider) -> Vec<Vec<Value>> {
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, c).expect("bind");
    TypeCheckVisitor::check(&mut bound, c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, c).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    materialize::materialize(&plan, s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec()
}

fn vmap(entries: &[(&str, Value)]) -> Value {
    Value::Map(
        entries
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect(),
    )
}

// =====================================================================
// D-2 — the bug fix (a map literal returns a map, not null).
// =====================================================================

#[test]
fn map_literal_returns_a_map() {
    // REGRESSION: on `main` the `m` cell was `Null` (the latent bug).
    let s = n_nodes(1);
    let rows = run("MATCH (nd:X) RETURN {a: 1, b: 2} AS m", &s, &cat());
    assert_eq!(
        rows,
        vec![vec![vmap(&[
            ("a", Value::Integer(1)),
            ("b", Value::Integer(2))
        ])]],
    );
}

#[test]
fn empty_map_literal_returns_empty_map() {
    let s = n_nodes(1);
    let rows = run("MATCH (nd:X) RETURN {} AS m", &s, &cat());
    assert_eq!(
        rows,
        vec![vec![Value::Map(std::collections::BTreeMap::new())]]
    );
}

// =====================================================================
// D-8 — property access `map.key` (+ nested). Dynamic `m['key']` is #621.
// =====================================================================

#[test]
fn map_dot_access_present_key() {
    let s = n_nodes(1);
    assert_eq!(
        run("MATCH (nd:X) RETURN {a: 1}.a AS v", &s, &cat()),
        vec![vec![Value::Integer(1)]]
    );
}

#[test]
fn map_dot_access_missing_key_is_null() {
    let s = n_nodes(1);
    assert_eq!(
        run("MATCH (nd:X) RETURN {a: 1}.b AS v", &s, &cat()),
        vec![vec![Value::Null]]
    );
}

#[test]
fn map_dot_access_nested() {
    let s = n_nodes(1);
    assert_eq!(
        run("MATCH (nd:X) RETURN {x: {y: 5}}.x.y AS v", &s, &cat()),
        vec![vec![Value::Integer(5)]]
    );
}

// NOTE on `WITH {…} AS m RETURN m['b']` (the spec's active-verification
// (b) form): the `m['b']` DYNAMIC SUBSCRIPT needs the `[expr]` accessor
// grammar (sister PR #621). The `m.b` DOT form is proven by the
// `map_dot_access_*` tests above (on a literal map in the projection).
// Routing a constant map THROUGH a `WITH` column additionally trips an
// UNRELATED engine limitation (a pure-constant projected through `WITH`
// is dropped from the post-`WITH` row schema — `binding … missing from
// row schema`, reproducible for a constant scalar too), so the
// WITH-bound variant is deferred to #621/PR-B. Maps DO flow through
// RETURN columns end-to-end — see the UNION dedup tests below (maps
// carried through `Batch` columns + canonicalized).

// =====================================================================
// D-12 — DISTINCT / UNION over maps (shared canonical_row_key oracle).
// =====================================================================

#[test]
fn union_dedups_equal_maps_order_independent() {
    // `{a:1,b:2}` and `{b:2,a:1}` are the SAME map (order-independent) ⇒
    // UNION (distinct) collapses to ONE row.
    let s = n_nodes(1);
    let rows = run(
        "MATCH (nd:X) RETURN {a:1, b:2} AS m UNION MATCH (nd:X) RETURN {b:2, a:1} AS m",
        &s,
        &cat(),
    );
    assert_eq!(
        rows,
        vec![vec![vmap(&[
            ("a", Value::Integer(1)),
            ("b", Value::Integer(2))
        ])]],
        "equal maps must collapse under UNION"
    );
}

#[test]
fn union_keeps_distinct_maps_distinct() {
    let s = n_nodes(1);
    let rows = run(
        "MATCH (nd:X) RETURN {a:1} AS m UNION MATCH (nd:X) RETURN {a:2} AS m",
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        2,
        "distinct maps must stay distinct under UNION"
    );
}

#[test]
fn union_keeps_delimiter_injection_maps_distinct() {
    // #735 R1 (value-side) — `{a:"x", b:1}` and `{a:"x;b=I:1"}` are
    // DISTINCT maps a user can write directly. Pre-fix their canonical
    // keys collided (BOTH rendered `M{a=S:x;b=I:1;}`) so UNION silently
    // dedup'd them to ONE row (silent-wrong-answer). Post-fix the
    // length-prefixed key/string keeps them apart ⇒ TWO rows. This is
    // the END-TO-END proof (parse → bind → type-check → lower → execute
    // → UNION dedup, the shared `canonical_row_key` oracle) of the
    // unit-level `distinct_maps_never_collide_under_delimiter_injection`
    // test in `ops/mod.rs`.
    let s = n_nodes(1);
    let rows = run(
        r#"MATCH (nd:X) RETURN {a:"x", b:1} AS m UNION MATCH (nd:X) RETURN {a:"x;b=I:1"} AS m"#,
        &s,
        &cat(),
    );
    assert_eq!(
        rows.len(),
        2,
        "delimiter-injected distinct maps must NOT collapse under UNION (#735 R1)"
    );
}

#[test]
fn union_all_preserves_duplicate_maps() {
    let s = n_nodes(1);
    let rows = run(
        "MATCH (nd:X) RETURN {a:1} AS m UNION ALL MATCH (nd:X) RETURN {a:1} AS m",
        &s,
        &cat(),
    );
    assert_eq!(rows.len(), 2, "UNION ALL must NOT dedup");
}

// =====================================================================
// D-12 — DISTINCT over MULTIPLE map rows (EQUIVALENCE: equal maps = one
// group). The DISTINCT path shares `canonical_row_key` with GROUP BY +
// UNION; this proves the multi-row keying (3 identical map rows → 1).
// (A GROUP-BY `count` form is blocked by an unrelated engine quirk —
// `count(binding)` with a constant group key drops the binding from the
// post-group schema; the equivalence semantic is proven here + in the
// `canonical_row_key` unit test.)
// =====================================================================

#[test]
fn distinct_collapses_identical_maps() {
    // 3 nodes each project `{k:1}` ⇒ RETURN DISTINCT collapses to ONE row.
    let s = n_nodes(3);
    let rows = run("MATCH (nd:X) RETURN DISTINCT {k: 1} AS m", &s, &cat());
    assert_eq!(rows, vec![vec![vmap(&[("k", Value::Integer(1))])]]);
}

#[test]
fn distinct_map_with_null_value_is_one_group_equivalence() {
    // D-3/D-12 EQUIVALENCE — `{a:null}` rows dedup together (null ≡ null),
    // distinct from the `=`-operator's 3VL null. 3 nodes ⇒ ONE row.
    let s = n_nodes(3);
    let rows = run("MATCH (nd:X) RETURN DISTINCT {a: null} AS m", &s, &cat());
    assert_eq!(rows, vec![vec![vmap(&[("a", Value::Null)])]]);
}

// =====================================================================
// D-11 — write-op map-property FENCE (maps stay rejected; lists succeed).
// =====================================================================

/// Run a write query through the full pipeline over a 1-node fixture;
/// return Err(stage) if it is rejected at ANY stage (parse / bind /
/// type-check / lower / execute). A node IS present so a MATCH-driven
/// SET actually reaches the property-materialization step.
fn try_run_write(query: &str) -> Result<(), String> {
    let c = cat();
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let mut bound = BindingVisitor::bind(&stmt, query, &c).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &c).map_err(|e| format!("type-check: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &c).map_err(|e| format!("cross-substrate: {e:?}"))?;
    let plan = LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))?;
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    let s = n_nodes(1);
    let mut op = Pipeline::build(&plan).map_err(|e| format!("build: {e:?}"))?;
    // Drain to surface execution-time rejection.
    loop {
        let batch = op
            .next_batch(&ctx, &s)
            .map_err(|e| format!("execute: {e:?}"))?;
        if batch.is_empty() {
            break;
        }
    }
    Ok(())
}

#[test]
fn create_with_map_property_is_rejected() {
    // D-11 — a map MUST NOT persist as a property value (openCypher
    // forbids it). `literal_lift` keeps rejecting maps even though
    // `Value::Map` now exists; the rejection surfaces end-to-end.
    let result = try_run_write("CREATE (n {p: {a: 1}})");
    assert!(
        result.is_err(),
        "CREATE with a map property must be rejected, got {result:?}"
    );
}

#[test]
fn set_map_property_is_rejected() {
    let result = try_run_write("MATCH (nd:X) SET nd.p = {a: 1}");
    assert!(
        result.is_err(),
        "SET of a map property must be rejected, got {result:?}"
    );
}

#[test]
fn create_with_list_property_still_succeeds() {
    // The fence is map-SPECIFIC — the ADR-152-amendment-02 List lift is
    // preserved (lists ARE valid property values). A regression here
    // would mean the fence over-rejected.
    let result = try_run_write("CREATE (n {p: [1, 2]})");
    assert!(
        result.is_ok(),
        "CREATE with a list property must still succeed, got {result:?}"
    );
}
