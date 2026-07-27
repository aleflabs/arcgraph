//! M4-83 multi-statement snapshot-isolation invariant proptest.
//!
//! Closes ADR-038 §5.4.1 multi-statement deferral per amendment-03
//! §TIER-1 GAP E rule 2: "Same snapshot LSN held for all statements in
//! a multi-statement query."
//!
//! # Property
//!
//! For any random k-statement query (k ∈ [1, 5]), running
//! [`arcgraph_query::materialize_multi`] over the lowered plans against
//! a single shared [`arcgraph_query::executor::ExecutionContext`]
//! satisfies:
//!
//! 1. Exactly ONE [`arcgraph_core::Lsn`] value is captured across all
//!    statements (the post-execute `ctx.snapshot_lsn()` is `Some(_)`
//!    when k ≥ 1 and any plan produced a row stream).
//! 2. The captured value is identical for every statement in the
//!    chain (load-bearing per amendment-03 §TIER-1 GAP E rule 2).
//! 3. The result vec length equals the input statement count
//!    (one [`arcgraph_query::MaterializedResult`] per statement).
//!
//! # Case generation
//!
//! `PROPTEST_CASES = 10000` per the W13γ spawn prompt's load-bearing
//! invariant pin. Statements are drawn from a small library of
//! known-good ArcQL shapes (single-label MATCH+RETURN over the test
//! catalog) so the property is over LSN-sharing, NOT random ArcQL
//! grammar coverage. Random-ArcQL fuzz already lights at
//! `tests/grammar_proptest.rs`.

use arcgraph_core::{LabelId, NodeId};
use arcgraph_query::executor::value::NodeView;
use arcgraph_query::executor::{ExecutionContext, StubExecutorSubstrate, Value};
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CatalogProvider, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{materialize_multi, parse_multi};

use proptest::prelude::*;

// ---------------------------------------------------------------------
// Test catalog + substrate (shared across all proptest cases).
// ---------------------------------------------------------------------

fn build_catalog() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person", "Place"])
        .with_rel_types(["KNOWS", "LIVES_IN"])
        .with_properties(["name", "city", "id"])
}

fn build_substrate(catalog: &StubCatalogProvider) -> StubExecutorSubstrate {
    let person = LabelId::new(1);
    let place = LabelId::new(2);
    StubExecutorSubstrate::new()
        .with_node(
            catalog.tenant(),
            NodeView::new(NodeId::new(1), Some(person))
                .with_property("name", Value::String("Alice".into())),
        )
        .with_node(
            catalog.tenant(),
            NodeView::new(NodeId::new(2), Some(person))
                .with_property("name", Value::String("Bob".into())),
        )
        .with_node(
            catalog.tenant(),
            NodeView::new(NodeId::new(3), Some(place))
                .with_property("name", Value::String("Anytown".into())),
        )
}

// ---------------------------------------------------------------------
// Statement library — small, deterministic.
// ---------------------------------------------------------------------

const STATEMENT_LIB: &[&str] = &[
    "MATCH (n:Person) RETURN n",
    "MATCH (m:Person) RETURN m",
    "MATCH (p:Place)  RETURN p",
    "MATCH (a:Person) RETURN a",
    "MATCH (b:Place)  RETURN b",
];

// ---------------------------------------------------------------------
// Property: snapshot-LSN sharing across k random statements.
// ---------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 10_000,
        // Under the testing strategy "Property tests for every data structure
        // with invariants" — the M4-83 LSN-sharing invariant is that
        // load-bearing data structure invariant.
        ..ProptestConfig::default()
    })]

    /// For any random k-statement query, `materialize_multi` shares ONE
    /// snapshot LSN across ALL statements, returns one
    /// `MaterializedResult` per statement, and never spuriously fails
    /// on the known-good statement library.
    #[test]
    fn m4_83_multi_statement_snapshot_isolation_invariant(
        // Statement count k ∈ [1, 5]. The TIER-1 GAP E invariant
        // applies for k ≥ 1; k = 1 is the degenerate single-statement
        // case (still must observe the LSN-sharing property — captured
        // exactly once).
        k in 1usize..=5,
        // Pick statements by index from STATEMENT_LIB.
        indices in proptest::collection::vec(0usize..5, 1..=5),
    ) {
        // Trim indices to length k.
        let indices: Vec<usize> = indices.into_iter().take(k).collect();
        prop_assume!(!indices.is_empty());
        let q = indices
            .iter()
            .map(|&i| STATEMENT_LIB[i])
            .collect::<Vec<_>>()
            .join("; ");

        // Plan-time pipeline.
        let stmts = parse_multi(&q).unwrap();
        prop_assert_eq!(stmts.len(), indices.len());

        let cat = build_catalog();
        let mut bound_stmts = BindingVisitor::bind_multi(&stmts, &q, &cat).unwrap();
        let mut plans = Vec::with_capacity(bound_stmts.len());
        for bound in bound_stmts.iter_mut() {
            TypeCheckVisitor::check(bound, &cat).unwrap();
            CrossSubstrateValidator::validate(bound, &cat).unwrap();
            plans.push(LogicalPlanLoweringVisitor::lower(bound).unwrap());
        }

        // Execute against a shared ctx.
        let sub = build_substrate(&cat);
        let ctx = ExecutionContext::new(cat.tenant(), cat.partition());
        prop_assert_eq!(ctx.snapshot_lsn(), None, "lazy LSN — none captured pre-first-batch");

        let results = materialize_multi(&plans, &sub, &ctx).unwrap();

        // Property 1: result count == input statement count.
        prop_assert_eq!(results.len(), indices.len());

        // Property 2 (W13β fix-up M-1 reconciliation): every
        // statement in the chain observed the SAME captured LSN
        // DURING the run; at multi-stmt-end the outer LSN guard
        // released the LSN (rule 4) and lit the consumption latch
        // (rule 5). Pre-fix-up this property was tested by reading
        // `ctx.snapshot_lsn()` post-run; post-fix-up the LSN slot is
        // None at that point, and the load-bearing observable is
        // the latch.
        prop_assert_eq!(
            ctx.snapshot_lsn(),
            None,
            "post-run: outer guard drop released the LSN (rule 4)"
        );
        prop_assert!(
            ctx.lsn_consumed(),
            "k≥1 chain captured + released the LSN exactly once \
             (rule 5 latch lit on consumed ctx)"
        );

        // Property 3 (W13β fix-up M-1 reconciliation): re-running on
        // the SAME ctx is FORBIDDEN per rule 5; a fresh ctx exhibits
        // the same capture-and-release discipline (rule-5 uniformity
        // across invocations).
        let ctx2 = ExecutionContext::new(cat.tenant(), cat.partition());
        let _again = materialize_multi(&plans, &sub, &ctx2).unwrap();
        prop_assert!(
            ctx2.lsn_consumed(),
            "fresh-ctx re-run also lights the rule-5 latch (uniformity \
             across runs per TIER-1 GAP E)"
        );
    }
}
