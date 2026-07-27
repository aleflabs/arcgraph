//! W26-γ-3 / ADR-136 §D5 — malformed-input corpus regression for
//! the openCypher parser.
//!
//! # Surface
//!
//! `arcgraph_query::parse(input)`. Extends the 40-row fuzz corpus
//! at `fuzz/corpus/arcql_parser_fuzz/` (post-W26-γ-3 D2) into the
//! `cargo test`-replayable regression layer: every adversarial
//! pattern that has historically broken the parser gets a named
//! test case here, so a future regression manifests as a test
//! failure rather than a fuzz finding.
//!
//! # Adversarial classes covered (extension of W26-γ-3 D5)
//!
//! 1. Empty input.
//! 2. Whitespace-only input.
//! 3. Comments-only input.
//! 4. Single-token inputs (bare keywords).
//! 5. Mismatched brackets / parens / braces.
//! 6. Huge integer literals (> i64::MAX).
//! 7. Malformed string escapes.
//! 8. Mixed-case keywords (Cypher convention: case-insensitive at
//!    clause level).
//! 9. Unicode identifiers / strings.
//! 10. Backtick-escaped identifiers with adversarial content.
//! 11. Deep parenthesis nesting.
//! 12. Deep MATCH path chains.
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
//! every fuzz corpus entry has a `cargo test` regression sibling.

use arcgraph_query::parse;

/// Run a list of adversarial inputs through the parser. Every input
/// must result in a structured Result — never a panic. Both Ok and
/// Err outcomes are acceptable.
fn assert_no_panic_on_corpus(rows: &[(&str, &str)]) {
    for (name, input) in rows {
        let result = std::panic::catch_unwind(|| parse(input));
        match result {
            Ok(_) => (),
            Err(_) => panic!("corpus row '{name}' panicked on input: {input:?}"),
        }
    }
}

#[test]
fn corpus_empty_and_whitespace() {
    assert_no_panic_on_corpus(&[
        ("empty", ""),
        ("space", " "),
        ("tab", "\t"),
        ("newline", "\n"),
        ("crlf", "\r\n"),
        ("mixed-ws", " \t\n\r \t"),
    ]);
}

#[test]
fn corpus_comments_only() {
    assert_no_panic_on_corpus(&[
        ("line-comment", "-- a comment"),
        ("block-comment", "/* a block comment */"),
        ("block-no-close", "/* unclosed"),
        ("nested-block", "/* /* nested */ */"),
        ("comment-then-eof", "-- end"),
    ]);
}

#[test]
fn corpus_bare_keywords() {
    assert_no_panic_on_corpus(&[
        ("match", "MATCH"),
        ("return", "RETURN"),
        ("where", "WHERE"),
        ("with", "WITH"),
        ("limit", "LIMIT"),
        ("create", "CREATE"),
        ("optional", "OPTIONAL"),
    ]);
}

#[test]
fn corpus_mismatched_brackets() {
    assert_no_panic_on_corpus(&[
        ("paren-open-only", "MATCH ("),
        ("paren-close-only", "MATCH )"),
        ("bracket-open-only", "MATCH (n)-["),
        ("bracket-close-only", "MATCH (n)-]"),
        ("brace-open-only", "MATCH (n {"),
        ("brace-close-only", "MATCH (n })"),
        ("double-paren", "MATCH ((n))"),
        ("double-paren-mismatch", "MATCH ((n)"),
    ]);
}

#[test]
fn corpus_huge_int_literals() {
    assert_no_panic_on_corpus(&[
        (
            "i64-max",
            "MATCH (n) WHERE n.x = 9223372036854775807 RETURN n",
        ),
        (
            "i64-max-plus-one",
            "MATCH (n) WHERE n.x = 9223372036854775808 RETURN n",
        ),
        (
            "u64-max",
            "MATCH (n) WHERE n.x = 18446744073709551615 RETURN n",
        ),
        (
            "huge",
            "MATCH (n) WHERE n.x = 99999999999999999999999999999999 RETURN n",
        ),
        (
            "negative-huge",
            "MATCH (n) WHERE n.x = -99999999999999999999999999999999 RETURN n",
        ),
        ("zero-padded", "MATCH (n) WHERE n.x = 000000123 RETURN n"),
    ]);
}

#[test]
fn corpus_string_escapes() {
    assert_no_panic_on_corpus(&[
        ("escape-quote", "MATCH (n) WHERE n.x = 'don\\'t' RETURN n"),
        (
            "escape-newline",
            "MATCH (n) WHERE n.x = 'line1\\nline2' RETURN n",
        ),
        ("escape-tab", "MATCH (n) WHERE n.x = 'col1\\tcol2' RETURN n"),
        (
            "escape-backslash",
            "MATCH (n) WHERE n.x = 'path\\\\to\\\\file' RETURN n",
        ),
        ("escape-hex", "MATCH (n) WHERE n.x = '\\u00FF' RETURN n"),
        (
            "unclosed-string",
            "MATCH (n) WHERE n.x = 'unclosed RETURN n",
        ),
        (
            "double-quote-string",
            "MATCH (n) WHERE n.x = \"double\" RETURN n",
        ),
    ]);
}

#[test]
fn corpus_mixed_case_keywords() {
    assert_no_panic_on_corpus(&[
        ("lower-match", "match (n) return n"),
        ("mixed-match", "MaTcH (n) ReTuRn n"),
        ("upper-everything", "MATCH (N) WHERE N.X > 0 RETURN N"),
        ("lower-everything", "match (n) where n.x > 0 return n"),
    ]);
}

#[test]
fn corpus_unicode_identifiers_and_strings() {
    assert_no_panic_on_corpus(&[
        ("chinese-string", "MATCH (n) WHERE n.name = '你好' RETURN n"),
        (
            "japanese-string",
            "MATCH (n) WHERE n.name = 'こんにちは' RETURN n",
        ),
        ("emoji-string", "MATCH (n) WHERE n.name = '🚀' RETURN n"),
        ("arabic-string", "MATCH (n) WHERE n.name = 'مرحبا' RETURN n"),
        ("backtick-unicode", "MATCH (n) WHERE n.`你好` = 1 RETURN n"),
        ("backtick-emoji", "MATCH (n) WHERE n.`🚀` = 1 RETURN n"),
    ]);
}

#[test]
fn corpus_backtick_adversarial() {
    assert_no_panic_on_corpus(&[
        ("bt-newline", "MATCH (n) WHERE n.`a\\nb` = 1 RETURN n"),
        ("bt-null", "MATCH (n) WHERE n.`a\\0b` = 1 RETURN n"),
        ("bt-cr", "MATCH (n) WHERE n.`a\\rb` = 1 RETURN n"),
        ("bt-empty", "MATCH (n) WHERE n.`` = 1 RETURN n"),
        ("bt-only-bt", "MATCH (n) WHERE n.````` = 1 RETURN n"),
        ("bt-space", "MATCH (n) WHERE n.`a b c` = 1 RETURN n"),
    ]);
}

#[test]
fn corpus_deep_parenthesis_nesting() {
    let mut q = "MATCH (n) WHERE ".to_string();
    for _ in 0..50 {
        q.push('(');
    }
    q.push_str("n.x = 1");
    for _ in 0..50 {
        q.push(')');
    }
    q.push_str(" RETURN n");
    assert_no_panic_on_corpus(&[("50-deep-parens", &q)]);
}

#[test]
fn corpus_long_path_chains() {
    let mut q = "MATCH ".to_string();
    for i in 0..30 {
        if i == 0 {
            q.push_str("(n0)");
        } else {
            q.push_str(&format!("-[:R]->(n{i})"));
        }
    }
    q.push_str(" RETURN n0, n29");
    assert_no_panic_on_corpus(&[("30-hop-chain", &q)]);
}

#[test]
fn corpus_pathological_combinations() {
    assert_no_panic_on_corpus(&[
        (
            "all-or-and-not",
            "MATCH (n) WHERE NOT NOT NOT (n.a = 1 AND (n.b = 2 OR n.c = 3 AND NOT n.d = 4)) RETURN n",
        ),
        (
            "many-projections",
            "MATCH (n) RETURN n.a, n.b, n.c, n.d, n.e, n.f, n.g, n.h, n.i, n.j, n.k LIMIT 1",
        ),
        (
            "many-order-by",
            "MATCH (n) RETURN n ORDER BY n.a, n.b DESC, n.c ASC, n.d DESC LIMIT 10",
        ),
        ("trailing-semicolon", "MATCH (n) RETURN n;"),
        ("multiple-semicolons", "MATCH (n) RETURN n;;;"),
        ("multi-statement", "MATCH (n) RETURN n; MATCH (m) RETURN m"),
    ]);
}

#[test]
fn corpus_determinism_across_repeated_parse() {
    // Every entry MUST yield the same Ok/Err outcome across repeated
    // parse calls — no hidden state in the parser.
    let rows: &[&str] = &[
        "MATCH (n) RETURN n",
        "INVALID INPUT",
        "MATCH (n:Person {age: 30}) RETURN n LIMIT 10",
        "",
        "MATCH (n)-[:R*1..3]->(b) RETURN n, b",
    ];
    for q in rows {
        let r1 = parse(q).is_ok();
        let r2 = parse(q).is_ok();
        let r3 = parse(q).is_ok();
        assert_eq!(r1, r2);
        assert_eq!(r2, r3);
    }
}
