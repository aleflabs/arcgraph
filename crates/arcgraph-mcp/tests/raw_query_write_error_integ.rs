//! W27-β ADR-153 — `graph.raw_query` write-op error-path integration.
//!
//! ADR-153 §D-4 pin: a write-op runtime failure SURFACES through the
//! MCP error envelope (not silently as a partial result). The per-
//! tenant transaction's commit-or-rollback discipline per ADR-031 +
//! ADR-033 guarantees no partial side-effects: a failing DELETE on a
//! node with attached rels (and no DETACH) does NOT delete the node;
//! the WriteSummary counter stays at 0; the error envelope carries
//! the "relationships attached" diagnostic.

#![allow(clippy::unwrap_used)]

mod raw_query_write_common;
use arcgraph_mcp::CODE_QUERY_ERROR;
use raw_query_write_common::{fresh_dispatcher, parse_body, raw_query};

#[test]
fn bare_delete_on_attached_node_surfaces_relationships_attached_error() {
    // ADR-149 §D-7 + ADR-153 §D-4: `DELETE n` (no DETACH) on a node
    // with attached rels surfaces a runtime error through the MCP
    // error envelope. The substrate returns
    // `SubstrateAccessError::Io("relationships attached")`; the W13δ
    // codec-local error-translation maps Substrate errors to MCPError
    // ExecutionEval → JSON-RPC -32006.
    let d = fresh_dispatcher();
    let _ = parse_body(&raw_query(
        &d,
        "CREATE (a:User)-[r:FOLLOWS]->(b:User) RETURN r",
    ));
    // Bare DELETE (no DETACH) on the attached node.
    let resp = raw_query(&d, "MATCH (a:User)-[r:FOLLOWS]->(b:User) DELETE a");
    assert!(
        !resp["error"].is_null(),
        "bare DELETE on attached node must surface error envelope; resp={resp:?}"
    );
    let code = resp["error"]["code"].as_i64().expect("error code");
    // -32006 = ExecutionEval per the W13δ codec-local translation.
    assert_eq!(
        code, -32006,
        "expected -32006 ExecutionEval; got {code}: {resp:?}"
    );
    // Pin the diagnostic message — `data` carries the substrate's
    // detail; renderers consume this for the user-facing message.
    let data = resp["error"]["data"].as_str().unwrap_or("");
    assert!(
        data.contains("relationships attached")
            || data.contains("relationships")
            || data.contains("attached"),
        "error data must name the underlying cause; got: {data}"
    );

    // ADR-031 commit-or-rollback discipline: the state must be
    // unchanged after the failed DELETE. Both endpoints + the rel
    // remain observable.
    let post_nodes = parse_body(&raw_query(&d, "MATCH (n:User) RETURN n"));
    assert_eq!(
        post_nodes["row_count"], 2,
        "rollback discipline: both endpoints remain after failed DELETE"
    );
    let post_rels = parse_body(&raw_query(
        &d,
        "MATCH (a:User)-[r:FOLLOWS]->(b:User) RETURN r",
    ));
    assert_eq!(
        post_rels["row_count"], 1,
        "rollback discipline: rel remains after failed DELETE"
    );
}

#[test]
fn parse_error_in_write_op_surfaces_through_mcp_envelope() {
    // ADR-153 §D-4: a parse error on a write-op query surfaces through
    // the MCP error envelope. Per the W13δ codec-local taxonomy and
    // #945, ExplainError::Parse is a query-domain fault, not a
    // JSON-RPC envelope parse fault.
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "CREATE (n:User WITH bogus syntax)");
    assert!(
        !resp["error"].is_null(),
        "parse error must surface; resp={resp:?}"
    );
    let code = resp["error"]["code"].as_i64().expect("error code");
    assert_eq!(
        code,
        i64::from(CODE_QUERY_ERROR),
        "parse-class error must surface as -32005 QueryError; got {code}: {resp:?}"
    );
    let data = resp["error"]["data"].as_str().unwrap_or("");
    assert!(
        data.contains("parse") || data.contains("expected"),
        "parse error data must name the underlying cause; got: {data}"
    );
}

#[test]
fn empty_query_string_rejects_at_minus_32602_invalid_params() {
    // raw_query_tool empty-string guard fires BEFORE the executor
    // body runs (per raw_query.rs §"Validation order" step 4).
    let d = fresh_dispatcher();
    let resp = raw_query(&d, "");
    assert!(!resp["error"].is_null());
    assert_eq!(resp["error"]["code"], -32602);
}
