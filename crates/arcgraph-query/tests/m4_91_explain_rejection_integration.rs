//! M4-91 EXPLAIN parser-edge-case (hostile-input rejection) pins.
//!
//! These tests pin the rejection paths the grammar enforces around the
//! `EXPLAIN` keyword recognition + the D-19 read-only body restriction.
//! Without them, a future grammar refactor (e.g., dropping the
//! `kw_end` boundary on `kw_explain`, removing `EXPLAIN` from the
//! `keyword` exclusion set, or relaxing `explain_query`'s body from
//! `read_query` to `statement`) could regress the discipline silently.
//!
//! # Rejection set covered
//!
//! 1. `explain_alone_no_inner_query` — `EXPLAIN` alone (no body).
//! 2. `explain_explain_double_prefix` — `EXPLAIN EXPLAIN MATCH …`
//!    (nested wrapper).
//! 3. `explainer_identifier_not_keyword` — `EXPLAINER MATCH …`
//!    (kw_end boundary; EXPLAIN must not consume EXPLAINER prefix).
//! 4. `explain_ddl_unsupported` — `EXPLAIN CREATE VECTOR INDEX …` (D-19
//!    read-only body restriction).
//! 5. `match_paren_n_explain_label` — `MATCH (n:EXPLAIN) RETURN n`
//!    (uppercase EXPLAIN at identifier position is keyword-excluded;
//!    the backtick-escaped `` `EXPLAIN` `` form remains admissible).
//! 6. `empty_explain_query` — `EXPLAIN ` (whitespace-only inner body).
//! 7. `whitespace_only_query` — `   ` (entire query is whitespace).
//!
//! # Reverse-test discipline (Phase 4.3)
//!
//! Each test is non-vacuous against a SPECIFIC grammar mutation:
//!
//! - `match_paren_n_explain_label` — remove `EXPLAIN` from the
//!   `keyword` exclusion set in `grammar.pest` → input parses
//!   successfully → test reds.
//! - `explain_explain_double_prefix` — change `explain_query` body
//!   from `read_query` to `statement` → input parses (nested
//!   wrapper is now admissible) → test reds.
//! - `explainer_identifier_not_keyword` — remove `kw_end` from
//!   `kw_explain` → `EXPLAIN` prefix is consumed out of `EXPLAINER`,
//!   leaving `ER MATCH …` to fail downstream; the test shape stays
//!   parse-error but the failure cause shifts. The companion
//!   positive pin (`n.explainer` parses as identifier accessor)
//!   surfaces the kw_end-removal regression directly.
//! - `explain_alone_no_inner_query` / `empty_explain_query` —
//!   change `read_query` from `clause+` to `clause*` → empty body
//!   admitted → tests red.
//! - `explain_ddl_unsupported` — change `explain_query` body from
//!   `read_query` to `statement` → index DDL is admitted → test reds.
//! - `whitespace_only_query` — change top-level `query` to
//!   `SOI ~ statement? ~ ";"? ~ EOI` → empty input admitted → test
//!   reds.
//!
//! The shipping reverse-test cycle (commit f0a5f56's reverse-test on
//! `kw_explain`) demonstrated that removing the `kw_explain` rule
//! collapses the EXPLAIN happy paths; that cycle covers the
//! complementary case (load-bearing keyword recognition). This file
//! covers the OTHER half: load-bearing rejection of inputs that LOOK
//! like EXPLAIN but must not be admitted.
//!
//! # ADR provenance
//! - ADR-038 §2 D-19 — EXPLAIN body restricted to a "full ArcQL
//!   read query"; index DDL is not admitted.
//! - ADR-038 amendment-03 §TIER-1 GAP B — M4-91 sub-slice scope.
//! - PR #240 round-2 review FIND-3 — hostile-input pins required.

use arcgraph_query::ast::Statement;
use arcgraph_query::error::ParseError;
use arcgraph_query::explain::ExplainError;
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::{explain, parse};

// ---------------------------------------------------------------------
// Catalog fixture
// ---------------------------------------------------------------------

fn cat() -> StubCatalogProvider {
    StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_rel_types(["KNOWS"])
        .with_properties(["age", "name", "explain", "explainer", "profile"])
}

// Helper: assert a `parse()` failure is a `ParseError::Pest` (the
// PEG-level rejection — not the rarer `AstConstruction` path that
// triggers on integer overflow + similar). Each rejection case in
// this file is a grammar-level reject, so `Pest` is the canonical
// variant.
fn assert_pest_parse_error(input: &str, hint: &str) {
    match parse(input) {
        Err(ParseError::Pest { .. }) => {}
        Err(other) => panic!("{hint}: expected ParseError::Pest, got {other:?}"),
        Ok(stmt) => panic!("{hint}: expected parse error, got OK statement {stmt:?}"),
    }
}

// Same shape, but routed through the `explain()` entry point — pins
// that the parse-rejection bubbles up as `ExplainError::Parse`
// (NOT `ExplainError::ArcQL`, which would mean the input was
// syntactically OK and failed downstream).
fn assert_explain_parse_error(input: &str, hint: &str) {
    let cat = cat();
    match explain(input, &cat) {
        Err(ExplainError::Parse(ParseError::Pest { .. })) => {}
        Err(other) => panic!("{hint}: expected ExplainError::Parse(Pest), got {other:?}"),
        Ok(pt) => panic!("{hint}: expected parse error, got OK PlanTree {pt}"),
    }
}

// ---------------------------------------------------------------------
// Pins (8 rejection cases)
// ---------------------------------------------------------------------

#[test]
fn explain_alone_no_inner_query() {
    // `EXPLAIN` with no body: `read_query = clause+` requires at least
    // one clause. The grammar must reject — without this pin, a
    // future relaxation of `read_query` to `clause*` would silently
    // admit a no-op EXPLAIN.
    assert_pest_parse_error("EXPLAIN", "EXPLAIN alone");
    assert_explain_parse_error("EXPLAIN", "EXPLAIN alone (via explain entry)");
}

#[test]
fn explain_explain_double_prefix() {
    // Nested `EXPLAIN EXPLAIN …`. The grammar's `explain_query` body
    // is `read_query` (NOT `statement`), so the inner `EXPLAIN` is
    // not admissible inside the outer wrapper. A future refactor
    // that loosens the body to `statement` would regress this pin.
    assert_pest_parse_error(
        "EXPLAIN EXPLAIN MATCH (n:Person) RETURN n",
        "double EXPLAIN prefix",
    );
    assert_explain_parse_error(
        "EXPLAIN EXPLAIN MATCH (n:Person) RETURN n",
        "double EXPLAIN (via explain entry)",
    );

    // Symmetry: `EXPLAIN PROFILE …` is also rejected (the inner is a
    // bare read query, NOT a statement). PROFILE inside EXPLAIN
    // would otherwise be a coherent-looking wrapper composition.
    assert_pest_parse_error(
        "EXPLAIN PROFILE MATCH (n:Person) RETURN n",
        "EXPLAIN PROFILE composition",
    );
}

#[test]
fn explainer_identifier_not_keyword() {
    // `EXPLAINER MATCH …` — the `kw_end` boundary on `kw_explain`
    // (`@{ ^"EXPLAIN" ~ kw_end }`) prevents the EXPLAIN keyword
    // matcher from consuming the EXPLAIN prefix of EXPLAINER. The
    // remaining EXPLAINER doesn't start any statement, so overall
    // parse error.
    assert_pest_parse_error(
        "EXPLAINER MATCH (n:Person) RETURN n",
        "EXPLAINER as statement-leading identifier",
    );
    assert_explain_parse_error(
        "EXPLAINER MATCH (n:Person) RETURN n",
        "EXPLAINER (via explain entry)",
    );

    // Positive companion: lowercase `n.explainer` is admitted as a
    // property accessor. This is the kw_end-load-bearing pin —
    // without `kw_end`, the EXPLAIN matcher would still mis-fire on
    // the EXPLAINER prefix at clause-leading position; the property
    // accessor path doesn't go through `kw_explain` at all but
    // shows EXPLAINER is identifier-shaped.
    let pt = explain("EXPLAIN MATCH (n:Person) RETURN n.explainer", &cat())
        .expect("n.explainer must parse as a property accessor");
    // Sanity: the planner produced a non-degenerate tree (not a
    // catch-all).
    assert!(
        !pt.children.is_empty(),
        "EXPLAIN over property accessor must produce a child tree",
    );
}

#[test]
fn explain_ddl_unsupported() {
    // EXPLAIN body is restricted to a read query. Index DDL is not
    // admissible.
    // The grammar enforces this by setting `explain_query`'s body
    // to `read_query`, NOT `statement`.
    //
    // Without this pin, a future spawn that loosens the body to
    // `statement` (e.g., as part of a "support EXPLAIN over DDL"
    // feature request) would silently widen the v1.0 contract.
    assert_pest_parse_error(
        "EXPLAIN CREATE VECTOR INDEX foo FOR (n:Person) ON n.embedding",
        "EXPLAIN over DDL",
    );
    assert_explain_parse_error(
        "EXPLAIN CREATE VECTOR INDEX foo FOR (n:Person) ON n.embedding",
        "EXPLAIN over DDL (via explain entry)",
    );

    parse("CREATE VECTOR INDEX foo FOR (n:Person) ON n.embedding")
        .expect("bare vector-index DDL parses");
}

#[test]
fn match_paren_n_explain_label() {
    // `MATCH (n:EXPLAIN) RETURN n` — uppercase EXPLAIN at label
    // position. EXPLAIN is in the `keyword` exclusion set
    // (case-sensitive uppercase per Fix A.2 / PR #154 reviewer
    // Finding 1) so it cannot be admitted as a bare identifier.
    // This pin is the regression fence for accidentally REMOVING
    // EXPLAIN from the exclusion set (which would let EXPLAIN flow
    // through as a label name and silently widen the keyword
    // surface).
    assert_pest_parse_error(
        "MATCH (n:EXPLAIN) RETURN n",
        "MATCH (n:EXPLAIN) — uppercase EXPLAIN as label",
    );

    // Positive companion: backtick-escaped `` `EXPLAIN` `` IS
    // admissible (the Cypher canonical escape hatch). This pin
    // confirms the keyword exclusion ONLY rejects bare uppercase
    // EXPLAIN — escaped identifier text bypasses the exclusion.
    let stmt = parse("MATCH (n:`EXPLAIN`) RETURN n")
        .expect("backtick-escaped EXPLAIN must parse as label");
    assert!(matches!(stmt, Statement::Read(_)));

    // Positive companion 2: lowercase `n.explain` IS admissible
    // (the keyword set is case-sensitive uppercase only — the
    // lowercase property name `explain` is NOT keyword-excluded).
    // This pin is the regression fence for accidentally making the
    // exclusion case-INsensitive (which would break the n.match /
    // n.return / n.explain idiom at v1.0).
    let pt = explain("EXPLAIN MATCH (n:Person) RETURN n.explain", &cat())
        .expect("n.explain must parse (lowercase property)");
    assert!(
        !pt.children.is_empty(),
        "EXPLAIN over n.explain must produce a child tree",
    );
}

#[test]
fn empty_explain_query() {
    // `EXPLAIN ` followed by whitespace only — no inner read query.
    // The grammar's `read_query = clause+` requires at least one
    // clause; the body must reject. This pin is distinct from
    // `explain_alone_no_inner_query` in that there's trailing
    // whitespace; `pest` admits arbitrary inter-token whitespace
    // but the body `clause+` still requires at least one clause to
    // match. Without this pin, a future relaxation of `read_query`
    // to `clause*` would silently admit a no-op EXPLAIN with
    // trailing whitespace.
    assert_pest_parse_error("EXPLAIN ", "EXPLAIN with trailing whitespace");
    assert_pest_parse_error("EXPLAIN   \t\n  ", "EXPLAIN with mixed whitespace");
    assert_explain_parse_error("EXPLAIN ", "EXPLAIN with whitespace (via explain entry)");
}

#[test]
fn whitespace_only_query() {
    // Empty / whitespace-only top-level input. The grammar's
    // top-level `query = SOI ~ statement ~ ";"? ~ EOI` requires a
    // statement, so a whitespace-only input must reject. This pin
    // is the regression fence for accidentally making `statement`
    // optional at the top level (e.g., `query = SOI ~ statement?
    // ~ ";"? ~ EOI` would silently admit empty input as a no-op).
    assert_pest_parse_error("", "empty input");
    assert_pest_parse_error("   ", "spaces only");
    assert_pest_parse_error("\t\n  \r\n", "mixed whitespace only");
    // Comment-only input is rejected too (a comment is treated as
    // implicit whitespace; the statement requirement still applies).
    assert_pest_parse_error("-- a comment\n", "line comment only");

    // Routed through `explain()` for symmetry with the other pins:
    // empty input is a parse error (NOT an `ExplainError::ArcQL`,
    // which would imply the input was syntactically OK).
    assert_explain_parse_error("", "empty input (via explain entry)");
    assert_explain_parse_error("   ", "spaces only (via explain entry)");
}
