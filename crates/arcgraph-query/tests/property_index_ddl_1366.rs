//! **#1366 (task #248, Phase 1) — property-index DDL round-trip +
//! lowering.**
//!
//! Strong-oracle parser tests for `CREATE INDEX <name> [IF NOT EXISTS]
//! FOR (var:Label) ON (var.prop)` (the user-visible property index,
//! distinct from `CREATE VECTOR INDEX`): the exact AST shape, the
//! parse→print→re-parse round-trip, the type-check PASS, and the
//! lowering to `LogicalPlan::CreatePropertyIndex`.

use arcgraph_query::ast::{
    CreatePropertyIndexStatement, IndexDdlStatement, IndexNameRef, Statement,
};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{BindingVisitor, StubCatalogProvider, TypeCheckVisitor};

fn parse_property_ddl(src: &str) -> CreatePropertyIndexStatement {
    match parse(src).unwrap_or_else(|e| panic!("parse {src:?} failed: {e:?}")) {
        Statement::IndexDdl(IndexDdlStatement::CreateProperty(c)) => c,
        other => panic!("expected CreateProperty, got {other:?}"),
    }
}

#[test]
fn parse_create_index_exact_ast() {
    let c = parse_property_ddl("CREATE INDEX user_email FOR (n:User) ON (n.email)");
    assert_eq!(c.name, IndexNameRef::Literal("user_email".into()));
    assert!(!c.if_not_exists);
    assert_eq!(c.pattern_var, "n");
    assert_eq!(c.label, "User");
    assert_eq!(c.property, "email");
}

#[test]
fn parse_create_index_if_not_exists() {
    let c = parse_property_ddl("CREATE INDEX e IF NOT EXISTS FOR (n:User) ON (n.email)");
    assert!(c.if_not_exists, "IF NOT EXISTS parsed");
    assert_eq!(c.label, "User");
    assert_eq!(c.property, "email");
}

#[test]
fn parse_create_index_admits_unparenthesized_property() {
    // `ON n.email` (no parens) is also admitted (index_property covers
    // both forms).
    let c = parse_property_ddl("CREATE INDEX e FOR (n:User) ON n.email");
    assert_eq!(c.property, "email");
}

#[test]
fn create_index_round_trips_through_display() {
    // parse → Display → re-parse produces an equal AST (the grammar
    // round-trip property).
    let src = "CREATE INDEX user_email IF NOT EXISTS FOR (n:User) ON (n.email)";
    let stmt = parse(src).unwrap();
    let printed = format!("{stmt}");
    let reparsed = parse(&printed).unwrap_or_else(|e| panic!("re-parse {printed:?} failed: {e:?}"));
    assert_eq!(stmt, reparsed, "parse→print→parse must be stable");
}

#[test]
fn create_index_is_distinct_from_create_vector_index() {
    // A plain CREATE INDEX (no VECTOR) must parse as CreateProperty, NOT
    // CreateVector — the PEG ordered-choice divergence at the 2nd token.
    match parse("CREATE INDEX e FOR (n:User) ON (n.email)").unwrap() {
        Statement::IndexDdl(IndexDdlStatement::CreateProperty(_)) => {}
        other => panic!("plain CREATE INDEX must be CreateProperty, got {other:?}"),
    }
    // And CREATE VECTOR INDEX still parses as CreateVector.
    match parse("CREATE VECTOR INDEX v FOR (n:Doc) ON n.embedding").unwrap() {
        Statement::IndexDdl(IndexDdlStatement::CreateVector(_)) => {}
        other => panic!("CREATE VECTOR INDEX must be CreateVector, got {other:?}"),
    }
}

#[test]
fn create_index_type_checks_and_lowers_to_property_index_plan() {
    let query = "CREATE INDEX user_email FOR (n:User) ON (n.email)";
    let c = StubCatalogProvider::new();
    let stmt = parse(query).unwrap();
    // Binds + type-checks (a real Phase-1 DDL — must PASS, not
    // NotImplemented).
    let mut bound = BindingVisitor::bind(&stmt, query, &c).expect("bind");
    TypeCheckVisitor::check(&mut bound, &c).expect("type-check passes for CREATE INDEX");
    let plan = LogicalPlanLoweringVisitor::lower(&bound).expect("lower");
    match plan {
        LogicalPlan::CreatePropertyIndex(pc) => {
            assert_eq!(pc.label, "User");
            assert_eq!(pc.property, "email");
            assert!(!pc.if_not_exists);
        }
        other => panic!("expected CreatePropertyIndex plan, got {other:?}"),
    }
}
