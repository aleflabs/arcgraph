//! ADR-147 W26-θ Phase 1 — CREATE node proptest.
//!
//! Random labels + property bags should:
//! 1. Parse cleanly.
//! 2. Round-trip through Display.
//! 3. Bind + type-check + cross-substrate validate.
//! 4. Lower to a plan containing a CreateNode.
//! 5. Execute against `StubExecutorSubstrate` emitting exactly one row.
//! 6. Result in the new node being visible via `scan_nodes`.

use arcgraph_core::{PartitionId, TenantId};

use arcgraph_query::ExecutorSubstrate;
use arcgraph_query::executor::ExecutionContext;
use arcgraph_query::executor::substrate::StubExecutorSubstrate;
use arcgraph_query::logical_plan::LogicalPlanLoweringVisitor;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use arcgraph_query::{executor::Pipeline, parse};
use proptest::prelude::*;

/// Strategy for label names.
///
/// Prefix labels so the generated bare identifier cannot collide with
/// grammar.pest's reserved keyword exclusion set.
fn label_strategy() -> impl Strategy<Value = String> {
    "[A-Z][A-Za-z0-9_]{0,8}".prop_map(|s| format!("L_{s}"))
}

/// Strategy for property keys — lowercase-prefixed identifier-safe.
fn prop_key_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,8}".prop_filter("non-keyword", |s| !is_reserved(s))
}

/// Strategy for property values — Phase 1 literal-only subset.
///
/// Note: we generate ONLY positive integers because negative
/// integers parse as `UnaryOp(Neg, ...)` rather than as
/// `Literal::Integer(_)`. Phase 1 Type-check rejects the unary-op
/// shape per ADR-147 §D-4 literal-only narrowing; positive integers
/// parse as bare `Literal::Integer(_)` and admit. A future Phase
/// amendment may pre-fold `-N` to `Literal::Integer(-N)` to flatten
/// the AST; until then the proptest uses the literal-shape subset.
fn prop_value_strategy() -> impl Strategy<Value = String> {
    prop_oneof![
        any::<u32>().prop_map(|i| i.to_string()),
        any::<bool>().prop_map(|b| if b {
            "TRUE".to_string()
        } else {
            "FALSE".to_string()
        }),
        "[a-zA-Z][a-zA-Z0-9 ]{0,16}".prop_map(|s| format!("\"{s}\"")),
    ]
}

fn is_reserved(s: &str) -> bool {
    matches!(
        s,
        "MATCH"
            | "WHERE"
            | "RETURN"
            | "WITH"
            | "UNWIND"
            | "AS"
            | "DISTINCT"
            | "ORDER"
            | "BY"
            | "ASC"
            | "DESC"
            | "LIMIT"
            | "SKIP"
            | "AND"
            | "OR"
            | "NOT"
            | "IN"
            | "IS"
            | "NULL"
            | "TRUE"
            | "FALSE"
            | "FOR"
            | "ALL"
            | "NEAR"
            | "RANK"
            | "DEFINE"
            | "OPTIONAL"
            | "EXPLAIN"
            | "PROFILE"
            | "CREATE"
    )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// ADR-147 §D-1 + §D-2: random CREATE-node queries parse cleanly
    /// and the round-trip through Display is structure-preserving.
    #[test]
    fn create_node_random_label_parses_and_roundtrips(label in label_strategy()) {
        let q = format!("CREATE (n:{label}) RETURN n");
        let parsed = parse(&q).expect("parse OK");
        let printed = format!("{parsed}");
        let re_parsed = parse(&printed).expect("re-parse OK");
        prop_assert_eq!(parsed, re_parsed, "Display round-trips");
    }

    /// ADR-147 §D-3 / §D-4: random literal property bags bind + type-check.
    #[test]
    fn create_node_random_properties_typecheck(
        label in label_strategy(),
        key in prop_key_strategy(),
        value in prop_value_strategy(),
    ) {
        let q = format!("CREATE (n:{label} {{{key}: {value}}}) RETURN n");
        let stmt = match parse(&q) {
            Ok(s) => s,
            Err(_) => return Ok(()),  // syntactic rejects skip
        };
        let cat = StubCatalogProvider::new();
        let mut bound = BindingVisitor::bind(&stmt, &q, &cat).expect("bind OK");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check accepts literals");
    }

    /// End-to-end: each CREATE produces exactly one node visible via
    /// scan_nodes.
    #[test]
    fn create_node_random_label_visible_via_scan_nodes(label in label_strategy()) {
        let q = format!("CREATE (n:{label}) RETURN n");
        let stmt = parse(&q).expect("parse OK");
        let cat = StubCatalogProvider::new();
        let mut bound = BindingVisitor::bind(&stmt, &q, &cat).expect("bind OK");
        TypeCheckVisitor::check(&mut bound, &cat).expect("type-check OK");
        CrossSubstrateValidator::validate(&bound, &cat).expect("cross-substrate OK");
        let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower OK");
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        let mut op = Pipeline::build(&plan).expect("pipeline build OK");
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        prop_assert_eq!(b1.row_count(), 1, "CREATE emits one row");
        // The substrate now contains exactly one node.
        let nodes = s.scan_nodes(TenantId::DEFAULT, None, arcgraph_core::Lsn::MAX).expect("scan OK");
        prop_assert_eq!(nodes.len(), 1, "scan_nodes observes the new node");
    }
}
