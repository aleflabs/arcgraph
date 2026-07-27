//! ADR-150 W26-θ Phase 4 — SET / REMOVE proptest.
//!
//! Random property names + label names + SET / REMOVE mutation
//! shapes must:
//! 1. Parse cleanly.
//! 2. Round-trip through Display.
//! 3. Bind + type-check + cross-substrate validate.
//! 4. Lower to a plan containing `LogicalPlan::Set` or
//!    `LogicalPlan::Remove`.

use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};
use proptest::prelude::*;

fn ident_strategy() -> impl Strategy<Value = String> {
    "[a-z][a-zA-Z0-9_]{0,8}".prop_filter("non-reserved", |s| !is_reserved(s))
}

fn label_strategy() -> impl Strategy<Value = String> {
    // The canonical reserved-word set lives in grammar.pest; keep this
    // self-maintaining by generating labels that cannot equal a bare
    // keyword.
    "[A-Z][A-Za-z0-9_]{0,8}".prop_map(|s| format!("L_{s}"))
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
            | "DELETE"
            | "DETACH"
            | "SET"
            | "REMOVE"
    )
}

fn lower(query: &str) -> Result<LogicalPlan, String> {
    let stmt = parse(query).map_err(|e| format!("parse: {e:?}"))?;
    let cat = StubCatalogProvider::new();
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).map_err(|e| format!("bind: {e:?}"))?;
    TypeCheckVisitor::check(&mut bound, &cat).map_err(|e| format!("typecheck: {e:?}"))?;
    CrossSubstrateValidator::validate(&bound, &cat)
        .map_err(|e| format!("cross-substrate: {e:?}"))?;
    LogicalPlanLoweringVisitor::lower(&bound).map_err(|e| format!("lower: {e:?}"))
}

fn has_set(p: &LogicalPlan) -> bool {
    matches!(p, LogicalPlan::Set(_))
        || match p {
            LogicalPlan::Filter(f) => has_set(&f.input),
            LogicalPlan::Project(pr) => has_set(&pr.input),
            LogicalPlan::Limit(l) => has_set(&l.input),
            LogicalPlan::Skip(s) => has_set(&s.input),
            _ => false,
        }
}

fn has_remove(p: &LogicalPlan) -> bool {
    matches!(p, LogicalPlan::Remove(_))
        || match p {
            LogicalPlan::Filter(f) => has_remove(&f.input),
            LogicalPlan::Project(pr) => has_remove(&pr.input),
            LogicalPlan::Limit(l) => has_remove(&l.input),
            LogicalPlan::Skip(s) => has_remove(&s.input),
            _ => false,
        }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// ADR-150 §D-1 / §D-2: random SET property-assign queries parse
    /// + round-trip through Display + lower to a plan with
    /// LogicalPlan::Set.
    #[test]
    fn set_property_assign_round_trips(
        label in label_strategy(),
        prop in ident_strategy(),
        value in 0i64..1_000_000,
    ) {
        let q = format!("CREATE (n:{label}) SET n.{prop} = {value}");
        let stmt = parse(&q).expect("parse OK");
        let printed = format!("{stmt}");
        let reparsed = parse(&printed).expect("reparse OK");
        prop_assert_eq!(stmt, reparsed, "Display round-trip failed for `{}`", q);
        let plan = lower(&q).expect("lower OK");
        prop_assert!(has_set(&plan), "expected LogicalPlan::Set in: {:?}", plan);
    }

    /// ADR-150 §D-1 / §D-2: random SET label-add queries round-trip
    /// + lower.
    #[test]
    fn set_label_add_round_trips(
        node_label in label_strategy(),
        add_label in label_strategy(),
    ) {
        let q = format!("CREATE (n:{node_label}) SET n:{add_label}");
        let stmt = parse(&q).expect("parse OK");
        let printed = format!("{stmt}");
        let reparsed = parse(&printed).expect("reparse OK");
        prop_assert_eq!(stmt, reparsed, "Display round-trip failed for `{}`", q);
        let plan = lower(&q).expect("lower OK");
        prop_assert!(has_set(&plan), "expected LogicalPlan::Set in: {:?}", plan);
    }

    /// ADR-150 §D-1 / §D-2: random REMOVE property queries round-trip
    /// + lower.
    #[test]
    fn remove_property_round_trips(
        node_label in label_strategy(),
        prop in ident_strategy(),
    ) {
        let q = format!("CREATE (n:{node_label}) REMOVE n.{prop}");
        let stmt = parse(&q).expect("parse OK");
        let printed = format!("{stmt}");
        let reparsed = parse(&printed).expect("reparse OK");
        prop_assert_eq!(stmt, reparsed, "Display round-trip failed for `{}`", q);
        let plan = lower(&q).expect("lower OK");
        prop_assert!(has_remove(&plan), "expected LogicalPlan::Remove in: {:?}", plan);
    }

    /// ADR-150 §D-1 / §D-2: random REMOVE label queries round-trip +
    /// lower.
    #[test]
    fn remove_label_round_trips(
        node_label in label_strategy(),
        rm_label in label_strategy(),
    ) {
        let q = format!("CREATE (n:{node_label}) REMOVE n:{rm_label}");
        let stmt = parse(&q).expect("parse OK");
        let printed = format!("{stmt}");
        let reparsed = parse(&printed).expect("reparse OK");
        prop_assert_eq!(stmt, reparsed, "Display round-trip failed for `{}`", q);
        let plan = lower(&q).expect("lower OK");
        prop_assert!(has_remove(&plan), "expected LogicalPlan::Remove in: {:?}", plan);
    }
}
