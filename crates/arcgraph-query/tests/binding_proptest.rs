//! M4-21 binding-pass property tests.
//!
//! Strategy: generate well-formed read queries of the shape
//! ```text
//! MATCH (var0) MATCH (var1) ... RETURN var0, var1, ...
//! ```
//! where every `varN` is a unique 1–2-char lowercase identifier
//! (avoids the binding pass's span-cursor heuristic edge cases — see
//! `crates/arcgraph-query/src/semantic/binding.rs` top-of-file
//! comment).
//!
//! Property: **binding preserves variable identity.** Every RETURN
//! reference resolves to the BindingId minted by the corresponding
//! MATCH declaration. This catches binding-pass regressions where
//! e.g. a fresh BindingId is minted at every reference (instead of
//! resolving via the scope chain).

use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingId, BindingVisitor, BoundClause, BoundExpression, BoundMatchBody, BoundProjectionKind,
    BoundStatement, StubCatalogProvider,
};
use proptest::prelude::*;
use std::collections::HashMap;

/// Walk a BoundQuery and collect a map: variable name → BindingId
/// of its declaration site (first occurrence wins; the strategy
/// guarantees uniqueness of declared names).
fn collect_decl_map(stmt: &BoundStatement) -> HashMap<String, BindingId> {
    let mut acc = HashMap::new();
    if let BoundStatement::Read(q) = stmt {
        for c in &q.clauses {
            if let BoundClause::Match(m) = c {
                if let BoundMatchBody::Patterns(ps) = &m.body {
                    for p in ps {
                        if let Some(v) = &p.head.var {
                            acc.entry(v.name.clone()).or_insert(v.binding_id);
                        }
                    }
                }
            }
        }
    }
    acc
}

/// Walk RETURN's projections and collect (name, binding_id) for
/// every resolved VariableRef, in source order.
fn collect_return_refs(stmt: &BoundStatement) -> Vec<(String, BindingId)> {
    let mut acc = Vec::new();
    if let BoundStatement::Read(q) = stmt {
        for c in &q.clauses {
            if let BoundClause::Return(r) = c {
                for it in &r.items {
                    if let BoundProjectionKind::Expr(e) = &it.kind {
                        collect_expr_refs(e, &mut acc);
                    }
                }
            }
        }
    }
    acc
}

fn collect_expr_refs(e: &BoundExpression, acc: &mut Vec<(String, BindingId)>) {
    if let BoundExpression::VariableRef {
        name, binding_id, ..
    } = e
    {
        acc.push((name.clone(), *binding_id));
    }
}

/// Strategy: generate (input, declared_var_names).
fn arbitrary_well_bound_query() -> impl Strategy<Value = (String, Vec<String>)> {
    prop::collection::vec("[a-z][a-z0-9]?", 1..=4)
        .prop_filter("unique vars", |v| {
            let mut s = v.clone();
            s.sort();
            s.dedup();
            s.len() == v.len()
        })
        .prop_map(|vars| {
            let matches: Vec<String> = vars.iter().map(|v| format!("MATCH ({v})")).collect();
            let returns = vars.join(", ");
            let input = format!("{} RETURN {}", matches.join(" "), returns);
            (input, vars)
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn binding_preserves_variable_identity(
        (input, declared) in arbitrary_well_bound_query()
    ) {
        let stmt = parse(&input)
            .map_err(|e| TestCaseError::reject(format!("parse failed for {input:?}: {e}")))?;
        let cat = StubCatalogProvider::new();
        let bound = BindingVisitor::bind(&stmt, &input, &cat)
            .map_err(|errs| {
                TestCaseError::reject(format!(
                    "bind failed for {input:?} (well-bound by construction): {errs:?}"
                ))
            })?;

        let decl_map = collect_decl_map(&bound);
        for v in &declared {
            prop_assert!(
                decl_map.contains_key(v),
                "declared `{v}` not found in MATCH declarations"
            );
        }

        let refs = collect_return_refs(&bound);
        prop_assert_eq!(
            refs.len(),
            declared.len(),
            "RETURN should have one VariableRef per declared name"
        );
        for (name, binding_id) in &refs {
            let expected = decl_map
                .get(name)
                .copied()
                .unwrap_or_else(|| panic!("RETURN ref `{name}` has no declaration"));
            prop_assert_eq!(
                *binding_id,
                expected,
                "RETURN ref `{}` must resolve to its MATCH declaration's BindingId",
                name
            );
        }
    }
}
