#![no_main]
//! W22-DB-ε: ArcQL parser fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_query::parse`] plus binding of successfully parsed ASTs.
//! Per ADR-038 D-1..D-10 the parser maps `pest::error::Error<Rule>` to
//! a `Rule`-free `ParseError` so the semantic phase does not pull in
//! grammar internals.
//!
//! # Assertion
//!
//! `parse(input)` and binding of valid parses MUST NOT panic on ANY
//! UTF-8 input — neither valid grammar, nor invalid grammar, nor
//! pathological inputs (deep nesting, oversized identifiers, embedded
//! NULs, mixed-script homoglyphs). Valid inputs return `Ok(Statement)`;
//! invalid inputs return `Err(ParseError)`. Binding may return semantic
//! errors, but it must return normally.
//!
//! The fuzz harness rejects non-UTF-8 input early (the parser is a
//! `&str` consumer; non-UTF-8 is a framing-layer concern). Input
//! length is capped at 16 KiB to bound per-iter wall time —
//! production ArcQL queries are bounded by the MCP transport's
//! [`MAX_MESSAGE_BYTES`](arcgraph_mcp::jsonrpc::MAX_MESSAGE_BYTES)
//! framing cap (16 MiB), but fuzz iter wall budget makes 16 KiB the
//! practical cap.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 16 * 1024;

fuzz_target!(|data: &[u8]| {
    // Reject inputs that exceed the per-iter cap. libFuzzer-sys
    // iterates over an arbitrary byte stream; we narrow to UTF-8 +
    // bounded length to bound wall time without losing coverage of
    // the parser's UTF-8 hot path.
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };
    // The contract: parse MUST return without panicking. If it
    // succeeds, the binder must also return without panicking/aborting
    // on the parsed AST.
    if let Ok(stmt) = arcgraph_query::parse(s) {
        let catalog = arcgraph_query::semantic::StubCatalogProvider::new();
        let _ = arcgraph_query::semantic::BindingVisitor::bind(&stmt, s, &catalog);
    }

    // Multi-statement parser also gets exercised — its grammar arc
    // is a superset of the single-statement form. Same no-panic
    // contract.
    if let Ok(stmts) = arcgraph_query::parse_multi(s) {
        let catalog = arcgraph_query::semantic::StubCatalogProvider::new();
        let _ = arcgraph_query::semantic::BindingVisitor::bind_multi(&stmts, s, &catalog);
    }
});
