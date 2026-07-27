//! #954 — openCypher `//` line comments parse through the real query path.

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::semantic::StubCatalogProvider;

fn execute(query: &str) -> Vec<Vec<Value>> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);

    engine.execute(query, &substrate).expect("execute").rows
}

#[test]
fn double_slash_comments_parse_in_query_positions() {
    let baseline = execute("MATCH (n) RETURN n");

    assert_eq!(
        execute("MATCH (n) RETURN n // trailing comment"),
        baseline,
        "trailing // comment must preserve query result"
    );
    execute("// leading comment\nMATCH (n) RETURN n");
    execute("MATCH (n) // c1\nWHERE true // c2\nRETURN n");
}

#[test]
fn existing_dash_and_block_comments_still_parse() {
    execute("MATCH (n) RETURN n -- dash comment");
    execute("MATCH (n) /* block */ RETURN n");
}

#[test]
fn double_slash_inside_string_literal_is_not_a_comment() {
    assert_eq!(
        execute("RETURN '//not a comment' AS s"),
        vec![vec![Value::String("//not a comment".into())]]
    );
}
