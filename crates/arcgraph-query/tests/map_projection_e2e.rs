//! **ADR-191 D-6 / #620 (map-half, PR-B)** — map projection
//! `n{.key, .other, alias: expr, .*}` (openCypher v9 §3.5) END-TO-END
//! through the FULL query pipeline (parse → bind → type-check →
//! cross-substrate → lower → execute).
//!
//! These complement the direct-evaluator strong-oracle unit tests in
//! `src/executor/eval.rs::tests` (`mp_*`, which pin the D-6 value-level
//! semantics at the strongest oracle) + the parser tests
//! (`src/parser.rs::tests::mp_*`) + the type-check tests
//! (`src/semantic/type_check.rs::tests::map_projection_*`). Here we prove
//! the grammar production, binder, type-check, lowering, AND the executor
//! all compose correctly through a REAL `CREATE`-then-`MATCH` round-trip:
//! the projected properties are genuinely PERSISTED to the stub substrate
//! by `CREATE`, then RETRIEVED + projected by `MATCH … RETURN n{…}`.
//!
//! All oracles are STRONG `==` over the exact result rows / maps.
//!
//! # The D-6 null-handling split (the load-bearing semantic)
//!
//! - **`.key` property selector** DROPS a null/absent value
//!   (`n{.missing}` → `{}`).
//! - **`alias: expr` literal entry** KEEPS its key even when the value is
//!   null (`n{x: null}` → `{x: null}`).
//!
//! These two forms are the load-bearing contrast — `null_drop_vs_keep`
//! pins both in one query so a regression in either direction fails.
//!
//! # Why `CREATE`-then-`MATCH` (not a bare projection)
//!
//! Map projection over a node reads the node's PERSISTED property bag, so
//! the strongest oracle round-trips a real node through the substrate
//! (mirrors `create_then_match_by_property_smoke.rs`). The fixture node
//! carries `id` / `name` / `age` so `.name` / `.age` / `.*` have concrete
//! values to project.

use arcgraph_core::{LabelId, PartitionId, TenantId};
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::executor::{ExecutionContext, value::Value};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{Statement, executor::Pipeline, parse};
use std::collections::BTreeMap;

/// LabelId the StubExecutorSubstrate allocates for the first interned
/// label name (per `create_then_match_by_property_smoke.rs`). Pre-binding
/// the catalog to this id closes the catalog↔substrate id-divergence so
/// the MATCH-lowered Scan emits the SAME LabelId `create_node` assigns.
const STUB_FIRST_LABEL_ID: u32 = 1024;

fn lower(query: &str) -> LogicalPlan {
    let stmt = parse(query).expect("parse OK");
    let inner = match stmt {
        Statement::Read(_) => stmt,
        other => panic!("expected Read statement, got {other:?}"),
    };
    let cat = StubCatalogProvider::new().with_label_id("User", LabelId::new(STUB_FIRST_LABEL_ID));
    let mut bound = BindingVisitor::bind(&inner, query, &cat).expect("bind OK");
    TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
    CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
    LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK")
}

fn execute(
    plan: &LogicalPlan,
    substrate: &StubExecutorSubstrate,
    ctx: &ExecutionContext,
) -> Vec<Vec<Value>> {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    let mut out: Vec<Vec<Value>> = Vec::new();
    loop {
        let b = op.next_batch(ctx, substrate).expect("batch OK");
        if b.is_empty() {
            break;
        }
        for i in 0..b.row_count() {
            out.push(b.row(i).to_vec());
        }
    }
    out
}

/// `CREATE (n:User {id: 1, name: "Alice", age: 30})`, then run the given
/// projection query against the persisted node; return the rows.
fn create_then_project(projection_query: &str) -> Vec<Vec<Value>> {
    let substrate = StubExecutorSubstrate::new();
    let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let create_plan = lower(r#"CREATE (n:User {id: 1, name: "Alice", age: 30}) RETURN n"#);
    let _ = execute(&create_plan, &substrate, &ctx);

    let ctx2 = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
    let plan = lower(projection_query);
    execute(&plan, &substrate, &ctx2)
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
// Core — `.key` property selectors over a persisted node.
// =====================================================================

#[test]
fn property_selectors_project_persisted_values() {
    // `n{.name, .age}` over the persisted node → exactly `{name, age}`.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{.name, .age} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[
            ("name", Value::String("Alice".into())),
            ("age", Value::Integer(30)),
        ])]],
    );
}

// =====================================================================
// D-6 — `.key` DROPS null/absent; `alias: expr` KEEPS explicit null.
// =====================================================================

#[test]
fn missing_property_is_dropped() {
    // D-6 — `n{.name, .missing}` drops the absent `.missing` key → only
    // `{name}` survives.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{.name, .missing} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[("name", Value::String("Alice".into()))])]],
        "D-6: a .key selector over an absent property must DROP the key"
    );
}

#[test]
fn explicit_null_literal_entry_is_kept() {
    // D-6 — `n{x: null, y: 1}` KEEPS the explicit-null key → `{x: null, y: 1}`.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{x: null, y: 1} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[("x", Value::Null), ("y", Value::Integer(1))])]],
        "D-6: an explicit alias: null entry must KEEP the key"
    );
}

#[test]
fn null_drop_vs_keep_in_one_projection() {
    // The load-bearing D-6 contrast IN ONE QUERY: `.missing` (absent,
    // selector → DROP) alongside `present: null` (literal entry → KEEP).
    // A regression in EITHER direction fails this exact-map oracle.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{.missing, present: null} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[("present", Value::Null)])]],
        "D-6 split: selector-null DROPPED, literal-null KEPT — in one projection"
    );
}

// =====================================================================
// Literal entries with expressions + the `.*` all-properties selector.
// =====================================================================

#[test]
fn literal_entry_evaluates_expression() {
    // `n{.name, score: 1 + 1}` — the literal-entry value is a real
    // expression evaluated in the row scope.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{.name, score: 1 + 1} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[
            ("name", Value::String("Alice".into())),
            ("score", Value::Integer(2)),
        ])]],
    );
}

#[test]
fn all_properties_selector_copies_the_whole_bag() {
    // `n{.*}` copies EVERY persisted property of the node
    // (`id`, `name`, `age`).
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{.*} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[
            ("age", Value::Integer(30)),
            ("id", Value::Integer(1)),
            ("name", Value::String("Alice".into())),
        ])]],
    );
}

#[test]
fn all_properties_then_override_is_last_writer_wins() {
    // `n{.*, age: 99}` — `.*` copies, then the explicit `age: 99` overrides.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{.*, age: 99} AS m");
    assert_eq!(
        rows,
        vec![vec![vmap(&[
            ("age", Value::Integer(99)),
            ("id", Value::Integer(1)),
            ("name", Value::String("Alice".into())),
        ])]],
        "an explicit entry after .* overrides via last-writer-wins"
    );
}

#[test]
fn empty_projection_is_empty_map() {
    // `n{}` → the empty map.
    let rows = create_then_project("MATCH (n:User {id: 1}) RETURN n{} AS m");
    assert_eq!(rows, vec![vec![Value::Map(BTreeMap::new())]]);
}
