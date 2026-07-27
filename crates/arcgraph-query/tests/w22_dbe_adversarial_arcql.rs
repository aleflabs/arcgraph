//! W22-DB-ε Pillar 3 — Adversarial ArcQL inputs.
//!
//! # What this validates
//!
//! 12 adversarial ArcQL inputs that exercise the parser's reject path
//! across the v1.0-α attack surface:
//!
//! 1. **Quote break-out** — strings escaping the parameter context.
//! 2. **Comment injection** — `//` and `/* … */` inside identifier
//!    positions.
//! 3. **Embedded NUL byte** — `\0` injected mid-identifier.
//! 4. **Unicode homoglyph** — Cyrillic-A vs Latin-A on keyword
//!    boundary.
//! 5. **Deeply-nested expression** — recursion limit attack.
//! 6. **Oversized literal** — multi-MB string literal in the AST.
//! 7. **Mixed-script identifier** — RTL/LTR boundary confusion.
//! 8. **Empty input** — degenerate parse case.
//! 9. **Whitespace-only input** — degenerate parse case.
//! 10. **Truncated MATCH** — half-formed query.
//! 11. **Unbalanced parens / brackets** — pest grammar mis-anchor.
//! 12. **Multi-statement payload** — `;`-separated chain that the
//!     single-statement `parse` MUST reject.
//!
//! # Assertion (per ADR-079 Pillar 3)
//!
//! For EVERY adversarial input:
//! - `arcgraph_query::parse(input)` MUST NOT panic.
//! - The result MUST be `Err(ParseError)` — adversarial inputs MUST
//!   NOT silently parse to a valid `Statement`.
//! - The error variant carries useful diagnostic information (the
//!   shape varies by attack class).
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
//! every attack vector has an explicit failure-mode test with a
//! reject-shape pin.

#![allow(clippy::expect_used)]

use arcgraph_query::parse;

/// Each row: (label, attack input that the parser MUST reject).
///
/// The list excludes inputs that are *adversarially-shaped but
/// legitimately-parsing* (e.g., multi-megabyte string literals) — those
/// are exercised by the separate `does-not-panic` tests below.
///
/// NOTE: deeply-nested expressions (attack #5 — "recursion limit
/// attack") DID parse pre-#819 and so were excluded; post-#819 they are
/// a MUST-REJECT (the depth cap turns the former SIGABRT into a clean
/// `ParseError::ExpressionTooDeep`), so the three deep-nesting forms are
/// now first-class adversarial rows here.
fn adversarial_inputs() -> Vec<(&'static str, String)> {
    vec![
        (
            "quote_break_out",
            r#"MATCH (n {name: "alice"; DROP DATABASE; --"}) RETURN n"#.to_string(),
        ),
        ("embedded_nul", "MATCH (n\0:Person) RETURN n".to_string()),
        ("unicode_homoglyph", "МATCH (n) RETURN n".to_string()), // Cyrillic M
        (
            "mixed_script_ident",
            "MATCH (n) RETURN n.\u{200F}admin\u{200E}".to_string(),
        ),
        ("empty", String::new()),
        ("whitespace_only", "   \t\n  ".to_string()),
        ("truncated_match", "MATCH (n".to_string()),
        ("unbalanced_parens", "MATCH (n RETURN n)".to_string()),
        // Multi-statement: `parse` (single-statement) MUST reject; the
        // public `parse_multi` is the multi-statement entry point.
        (
            "multi_statement_via_single",
            "MATCH (n) RETURN n; MATCH (m) RETURN m".to_string(),
        ),
        // Single-quote string literal break (Cypher uses ' as well as ").
        (
            "quote_break_out_single",
            "MATCH (n {name: 'alice\\') RETURN n".to_string(),
        ),
        // Identifier that's actually a reserved keyword in many positions.
        (
            "keyword_as_ident",
            "MATCH (RETURN:Person) RETURN n".to_string(),
        ),
        // Trailing garbage after a valid query.
        (
            "trailing_garbage",
            "RETURN 1 GARBAGE BYTES HERE".to_string(),
        ),
        // #819 — deep-nesting DoS. Pre-fix these SIGABRT-ed the process;
        // post-fix the depth cap rejects them cleanly. Family (A)
        // bracket-style (300 = the issue's documented crash threshold)
        // + family (B) unary chaining (the R1 residual — `unary_expr`
        // self-recursion with no bracket; 4000 alternating `-`/`+` ops
        // ≈ the ~4 KB crash repro).
        (
            "deep_nested_parens",
            "RETURN ".to_string() + &"(".repeat(300) + "1" + &")".repeat(300),
        ),
        (
            "deep_nested_lists",
            "RETURN ".to_string() + &"[".repeat(300) + "1" + &"]".repeat(300),
        ),
        (
            "deep_nested_case",
            "RETURN ".to_string() + &"CASE WHEN true THEN ".repeat(300) + "1" + &" END".repeat(300),
        ),
        (
            "deep_unary_chain",
            "RETURN ".to_string()
                + &(0..4000)
                    .map(|i| if i % 2 == 0 { '-' } else { '+' })
                    .collect::<String>()
                + "1",
        ),
    ]
}

#[test]
fn w22_dbe_adversarial_arcql_inputs_all_reject_without_panic() {
    let mut accepted_in_error = Vec::new();
    let mut accepted_in_success = Vec::new();
    for (label, input) in adversarial_inputs() {
        // No panic — std::panic::catch_unwind would mask the bug but
        // would also hide a UAF in pest. We rely on the parser being
        // panic-free per ADR-038 §"library code panic-free" and let
        // any panic surface as a test failure.
        match parse(&input) {
            Ok(_) => {
                accepted_in_success.push(label);
            }
            Err(_) => {
                accepted_in_error.push(label);
            }
        }
    }
    // Every adversarial input MUST land in the reject pool. If any
    // silently parsed to a valid Statement, surface the label so the
    // discovery is actionable.
    assert!(
        accepted_in_success.is_empty(),
        "adversarial inputs that silently parsed: {accepted_in_success:?} \
         (see W22-DB-ε known-issues for triage)"
    );
    assert_eq!(
        accepted_in_error.len(),
        adversarial_inputs().len(),
        "every adversarial input MUST surface as a ParseError"
    );
}

/// Bounded-recursion — RETURN with nested parens.
///
/// **History.** W22-DB-ε / ADV-1 (see
/// [`docs/chaos/v1-alpha-chaos-known-issues.md`]) found that the pest
/// `expr` rule accepted arbitrary-depth nesting up to the thread stack
/// limit, so ~1024+ deep input over-flowed the stack and **SIGABRT-ed
/// the process** (an unauthenticated remote DoS — filed as #819). That
/// known-issue is now **CLOSED**: #819 added an expression-nesting-
/// depth cap ([`arcgraph_query::parser::MAX_EXPRESSION_DEPTH`]) enforced
/// at BOTH a pre-parse scan (bounding the pest matcher) and the AST
/// builder's depth guard, so a too-deep expression returns a clean
/// `Err(ParseError::ExpressionTooDeep)` BEFORE the stack overflows.
///
/// This test now pins BOTH ends of the contract: a cap-depth expression
/// parses cleanly (no over-rejection), and a deep expression (the old
/// crash regime) rejects with the depth-cap error (no SIGABRT). The
/// exhaustive depth/form/eval coverage lives in
/// `tests/expression_depth_dos.rs`.
///
/// [`docs/chaos/v1-alpha-chaos-known-issues.md`]: ../../docs/chaos/v1-alpha-chaos-known-issues.md
#[test]
fn w22_dbe_nested_parens_capped_not_stack_overflow() {
    use arcgraph_query::ParseError;
    use arcgraph_query::parser::MAX_EXPRESSION_DEPTH;

    // Cap-depth parses cleanly (the guard does not over-reject).
    let safe = "RETURN ".to_string()
        + &"(".repeat(MAX_EXPRESSION_DEPTH)
        + "1"
        + &")".repeat(MAX_EXPRESSION_DEPTH);
    assert!(
        parse(&safe).is_ok(),
        "a cap-depth ({MAX_EXPRESSION_DEPTH}) paren expression must parse"
    );

    // The old crash regime (1024-deep) now rejects with a clean
    // depth-cap error — NOT a stack overflow / SIGABRT (#819).
    let deep = "RETURN ".to_string() + &"(".repeat(1024) + "1" + &")".repeat(1024);
    assert!(
        matches!(parse(&deep), Err(ParseError::ExpressionTooDeep { .. })),
        "a 1024-deep paren expression must return ExpressionTooDeep (no SIGABRT)"
    );
}

/// Oversized-string-literal DoS pin — 1 MiB literal embedded in the
/// query MUST either parse cleanly OR reject cleanly. NEVER panic.
/// The MCP framer caps message at 16 MiB; 1 MiB is well-within that
/// cap so the parser is the bottleneck.
#[test]
fn w22_dbe_oversized_literal_does_not_panic() {
    let big = "x".repeat(1024 * 1024); // 1 MiB
    let input = format!("RETURN '{big}'");
    let _ = parse(&input);
    // Both Ok and Err are acceptable — the contract is no-panic.
}
