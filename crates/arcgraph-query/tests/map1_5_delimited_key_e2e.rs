//! **openCypher TCK `expressions/map/Map1[5]`** — *"Statically access a
//! field with a delimited identifier"* — plus the sibling
//! `Map2[5]` last two rows, both of which depend on the SAME map literal
//! `{null: 'Mats', NULL: 'Pontus'}` parsing.
//!
//! # Reproduce-first diagnosis (the bug was NOT where the prompt guessed)
//!
//! Map1[5]'s query template is `RETURN map.<key>` where `<key>` is a
//! BACKTICK-delimited identifier (`` map.`name` ``, `` map.`null` ``).
//! That POST-`.` backtick property-access ALREADY parsed and evaluated
//! correctly on `main` (the `property_accessor` rule already admits
//! `backtick_ident`, and `identifier_text` already strips the backticks).
//!
//! The actual blocker is Examples rows 5-6, whose map is the LITERAL
//! `{null: 'Mats', NULL: 'Pontus'}`. The bare uppercase keyword key
//! `NULL` was rejected at `map_entry`'s `identifier` position (the
//! case-sensitive `keyword` exclusion), so the whole literal failed to
//! parse with *"expected backtick_ident"*. (Bare lowercase `null` parsed
//! fine — it is not matched by the case-sensitive `"NULL"` keyword.)
//!
//! # The fix
//!
//! A `map_key` rule (`@{ backtick_ident | identifier_inner }`) scoped to
//! the EXPRESSION-context map literal admits reserved-word keys WITHOUT
//! backticks. A map key is followed by a mandatory `:`, so there is no
//! clause-ambiguity — this is squarely OUTSIDE the post-`.` property-key
//! v1.1 deferral (ADR-038 amendment-04 §D-X.1 / issue #189), which is the
//! `n.MATCH`-without-backticks case left UNCHANGED here.
//!
//! All oracles are STRONG `==` over the result rows. Each projection is
//! `MATCH (nd:X)`-wrapped because a bare `RETURN` lowers to `EmptyOp`
//! (zero rows) at v1.0-alpha — see `value_map_e2e.rs` for the rationale.

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize, parse};

const LABEL_X: u32 = 1;

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["X"])
        .with_properties(["g"])
}

fn one_node() -> StubExecutorSubstrate {
    StubExecutorSubstrate::new().with_node(
        TenantId::DEFAULT,
        NodeView::new(NodeId::new(1), Some(LabelId::new(LABEL_X)))
            .with_property("g", Value::Integer(1)),
    )
}

/// Full read pipeline → single-cell result. Asserts success at every
/// stage and returns the first column of the first row.
fn run_one(query: &str) -> Value {
    let c = cat();
    let s = one_node();
    let stmt = parse(query).expect("parse");
    let mut bound = BindingVisitor::bind(&stmt, query, &c).expect("bind");
    TypeCheckVisitor::check(&mut bound, &c).expect("type-check");
    CrossSubstrateValidator::validate(&bound, &c).expect("cross-substrate");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    let ctx = ExecutionContext::new(c.tenant(), c.partition());
    let rows = materialize::materialize(&plan, &s, &ctx)
        .expect("materialize")
        .rows()
        .to_vec();
    rows.into_iter()
        .next()
        .and_then(|r| r.into_iter().next())
        .expect("one result cell")
}

fn s(v: &str) -> Value {
    Value::String(v.to_string())
}

// =====================================================================
// Map1[5] Examples table — backtick-delimited static (`.`) access.
//   | map                            | key    | result   |
//   | {name: 'Mats', nome: 'Pontus'} | `name` | 'Mats'   |
//   | {name: 'Mats', nome: 'Pontus'} | `nome` | 'Pontus' |
//   | {name: 'Mats', nome: 'Pontus'} | `Mats` | null     |
//   | {name: 'Mats', nome: 'Pontus'} | `null` | null     |
//   | {null: 'Mats', NULL: 'Pontus'} | `null` | 'Mats'   |
//   | {null: 'Mats', NULL: 'Pontus'} | `NULL` | 'Pontus' |
// =====================================================================

#[test]
fn map1_5_delimited_key_name_hits() {
    let q = "MATCH (nd:X) WITH {name: 'Mats', nome: 'Pontus'} AS map RETURN map.`name` AS result";
    assert_eq!(run_one(q), s("Mats"));
}

#[test]
fn map1_5_delimited_key_nome_hits() {
    let q = "MATCH (nd:X) WITH {name: 'Mats', nome: 'Pontus'} AS map RETURN map.`nome` AS result";
    assert_eq!(run_one(q), s("Pontus"));
}

#[test]
fn map1_5_delimited_key_value_text_misses() {
    // `Mats` is a VALUE in the map, never a key → property access misses.
    let q = "MATCH (nd:X) WITH {name: 'Mats', nome: 'Pontus'} AS map RETURN map.`Mats` AS result";
    assert_eq!(run_one(q), Value::Null);
}

#[test]
fn map1_5_delimited_key_absent_null_misses() {
    let q = "MATCH (nd:X) WITH {name: 'Mats', nome: 'Pontus'} AS map RETURN map.`null` AS result";
    assert_eq!(run_one(q), Value::Null);
}

#[test]
fn map1_5_bare_keyword_keys_in_literal_parse_and_resolve_lowercase() {
    // ROW 5 — the regression. `{null: ..., NULL: ...}` failed to PARSE on
    // `main` (bare uppercase `NULL` keyword-excluded at `map_entry`).
    // Case-sensitive resolution: `` map.`null` `` → 'Mats'.
    let q = "MATCH (nd:X) WITH {null: 'Mats', NULL: 'Pontus'} AS map RETURN map.`null` AS result";
    assert_eq!(run_one(q), s("Mats"));
}

#[test]
fn map1_5_bare_keyword_keys_in_literal_resolve_uppercase() {
    // ROW 6 — case-sensitive: `` map.`NULL` `` → 'Pontus' (distinct key
    // from `null`; the two coexist in one literal).
    let q = "MATCH (nd:X) WITH {null: 'Mats', NULL: 'Pontus'} AS map RETURN map.`NULL` AS result";
    assert_eq!(run_one(q), s("Pontus"));
}

// =====================================================================
// Sibling coverage — Map2[5] last two rows share the SAME literal and
// are unblocked by the SAME fix (dynamic `[ 'key' ]` access landed
// separately, #1250). Pin them so the literal-key fix is not silently
// reverted.
//   | {null: 'Mats', NULL: 'Pontus'} | 'null' | 'Mats'   |
//   | {null: 'Mats', NULL: 'Pontus'} | 'NULL' | 'Pontus' |
// =====================================================================

#[test]
fn map2_5_bare_keyword_literal_dynamic_subscript_lowercase() {
    let q = "MATCH (nd:X) WITH {null: 'Mats', NULL: 'Pontus'} AS map RETURN map['null'] AS result";
    assert_eq!(run_one(q), s("Mats"));
}

#[test]
fn map2_5_bare_keyword_literal_dynamic_subscript_uppercase() {
    let q = "MATCH (nd:X) WITH {null: 'Mats', NULL: 'Pontus'} AS map RETURN map['NULL'] AS result";
    assert_eq!(run_one(q), s("Pontus"));
}

// =====================================================================
// Confinement guards — the fix is scoped to the MAP-LITERAL-KEY position
// ONLY. Plain identifier keys and bare-form property access are
// unaffected; reserved words at the POST-`.` property-key position still
// require backticks (issue #189's v1.1 surface, deliberately untouched).
// =====================================================================

#[test]
fn confinement_plain_identifier_map_key_unaffected() {
    let q = "MATCH (nd:X) WITH {name: 'Mats'} AS map RETURN map.name AS result";
    assert_eq!(run_one(q), s("Mats"));
}

#[test]
fn confinement_post_dot_uppercase_keyword_still_requires_backtick() {
    // Bare `n.NULL` (no backticks) at property-access position must STILL
    // be rejected — that is issue #189's v1.1 scope, NOT touched here.
    let q = "MATCH (nd:X) WITH {x: 1} AS map RETURN map.NULL AS result";
    assert!(
        parse(q).is_err(),
        "bare uppercase keyword at post-`.` property-key position must \
         still parse-fail (issue #189 scope, untouched by this slice)"
    );
}
