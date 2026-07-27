//! **#819 (HIGH, security/DoS)** — expression-nesting-depth guard.
//!
//! # The vulnerability (closed by this slice)
//!
//! A query containing a deeply-nested expression drove unbounded
//! native-stack recursion in **both** the pest PEG matcher (building
//! the `Pairs` tree) **and** the recursive-descent AST builder,
//! overflowing the thread stack and aborting the WHOLE process with
//! **SIGABRT**. Bolt auth is accepted-but-not-enforced today, so a
//! ~600-byte–to–~4 KB query was an **unauthenticated remote DoS** that
//! took ArcGraph down for every tenant. A stack overflow is UNCATCHABLE
//! (the runtime `abort()`s — `catch_unwind` cannot recover it), so the
//! fix is a depth **check that returns a clean `Err` BEFORE the stack
//! is exhausted**, at both recursion sites:
//!   - a pre-parse depth scan (`check_pre_parse_nesting_depth`) that
//!     bounds the pest matcher, and
//!   - the AST builder's RAII depth guard (defense-in-depth).
//!
//! There are TWO recursion families, BOTH covered here:
//!   - **(A) bracket-style** — each level re-enters the precedence
//!     ladder once, carrying a literal bracket / keyword: nested parens
//!     `((((1))))`, list `[[[[1]]]]`, map / subscript / function-call
//!     args, and `CASE WHEN … THEN CASE WHEN … 1 END END`.
//!   - **(B) unary-operator chaining** — `unary_expr = ("-"|"+") ~
//!     unary_expr` self-recurses once per prefix operator with NO
//!     bracket: `RETURN -+-+-+ … 1`. The bracket scan scored this 0, so
//!     it bypassed the first cut of the guard and still SIGABRT-ed
//!     (the #819 **R1 residual**, closed here). The pre-scan now counts
//!     consecutive unary-prefix operators toward the same depth budget;
//!     comments (`--` / `/* */`) and string literals stay opaque.
//!
//! See the `parser.rs` module-level budget comment for the empirical
//! derivation of the cap ([`MAX_EXPRESSION_DEPTH`] = 64): measured on a
//! 2 MiB Tokio worker stack, the pest matcher overflows at nesting
//! depth 100; depth 64 survives even a 512 KiB stack (⇒ ~4× margin).
//!
//! # Why these are the load-bearing assertions
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`: a
//! green test that can't fail on the bug is worthless. The
//! discriminating fact is that on `origin/main` (pre-fix) the 300/1000-
//! deep inputs **SIGABRT the process** (the issue documents the
//! `250 < depth ≤ 300` crash threshold + the `fatal runtime error:
//! stack overflow` server log). A crash kills the unit-test runner, so
//! we cannot assert "did not SIGABRT" directly in-process; instead we
//! assert the post-fix contract that is only reachable BECAUSE the
//! crash no longer happens — the deep input returns a clean
//! `Err(ParseError::ExpressionTooDeep)` — plus the just-under-cap
//! expression still parses AND evaluates. The dedicated server-stays-up
//! subprocess proof lives in
//! `crates/arcgraph-mcp/tests/bolt_e2e_depth_dos.rs` (sends the deep
//! query over Bolt, asserts a clean FAILURE + the server still answers
//! a subsequent `RETURN 1`).

#![allow(clippy::expect_used)]

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::eval::evaluate;
use arcgraph_query::executor::value::Value;
use arcgraph_query::parser::{MAX_EXPRESSION_DEPTH, MAX_FLAT_CHAIN_DEPTH};
use arcgraph_query::semantic::StubCatalogProvider;
use arcgraph_query::semantic::bound_ast::{
    BoundClause, BoundExpression, BoundProjectionKind, BoundStatement,
};
use arcgraph_query::{ParseError, parse};

// ---------------------------------------------------------------------
// Deep-input generators — the three nesting forms the issue documents.
// ---------------------------------------------------------------------

/// `RETURN (((( … d … 1 … ))))` — nested parens.
fn nested_parens(d: usize) -> String {
    "RETURN ".to_string() + &"(".repeat(d) + "1" + &")".repeat(d)
}

/// `RETURN [[[[ … d … 1 … ]]]]` — nested list literal.
fn nested_lists(d: usize) -> String {
    "RETURN ".to_string() + &"[".repeat(d) + "1" + &"]".repeat(d)
}

/// `RETURN CASE WHEN true THEN … d … 1 … END END` — nested CASE.
/// CASE has NO literal brackets, so it exercises the pre-scan's
/// keyword-based depth accounting (and was a distinct pest-overflow
/// form pre-fix).
fn nested_cases(d: usize) -> String {
    "RETURN ".to_string() + &"CASE WHEN true THEN ".repeat(d) + "1" + &" END".repeat(d)
}

/// `RETURN -+-+-+ … n … 1` — an alternating unary-prefix chain of `n`
/// operators (family (B), the #819 R1 residual). Alternating `-`/`+`
/// avoids the adjacent `--` that would open a line comment, so the
/// whole run reaches the deep `unary_expr` self-recursion in pest.
/// Each operator is one `unary_expr` recursion level with NO bracket —
/// the form the bracket/CASE scan under-counted to 0 (→ SIGABRT) before
/// this fix.
fn unary_chain(ops: usize) -> String {
    let chain: String = (0..ops)
        .map(|i| if i % 2 == 0 { '-' } else { '+' })
        .collect();
    format!("RETURN {chain}1")
}

/// `MATCH (n) WHERE n.p1 = 1 AND ... n.pN = N RETURN n`.
fn wide_where_predicates(predicates: usize) -> String {
    let mut input = String::from("MATCH (n) WHERE ");
    for i in 1..=predicates {
        if i > 1 {
            input.push_str(" AND ");
        }
        input.push_str(&format!("n.p{i} = {i}"));
    }
    input.push_str(" RETURN n");
    input
}

// This larger thread is ONLY for the at-cap ACCEPT-side boundary test
// (`depth == MAX_EXPRESSION_DEPTH` parses successfully). It does not
// relax the #819 DoS rejection bound: the O(n) iterative pre-scan in
// `check_pre_parse_nesting_depth` is unchanged, and every over-cap
// rejection test still runs on the default libtest stacks.
//
// Measured per subprocess at depth 64, deterministic across 3 repeats:
// in the debug profile, `lists@64` aborts at 2 MiB and succeeds at
// 4 MiB. The AST-builder operand restructure for the IS NULL / IN
// precedence fix grew debug-profile per-level stack usage past the
// 2 MiB default libtest thread; debug `parens@64` and `cases@64`
// still fit at 2 MiB. In the release profile, production at-cap stack
// consumption is <= 512 KiB (at least a 4x margin on a 2 MiB worker)
// and actually improves on main, which needed 768 KiB. The 8 MiB
// thread here accommodates only the debug-profile boundary measurement.
const BOUNDARY_ACCEPT_STACK_BYTES: usize = 8 * 1024 * 1024;

fn parses_ok_on_boundary_stack(input: String, form: &str, depth: usize) -> bool {
    std::thread::Builder::new()
        .name(format!("expression-depth-boundary-{form}-{depth}"))
        .stack_size(BOUNDARY_ACCEPT_STACK_BYTES)
        .spawn(move || parse(&input).is_ok())
        .expect("spawn expression-depth boundary parser")
        .join()
        .expect("join expression-depth boundary parser")
}

/// Assert `input` is rejected with the depth-cap error (NOT a panic,
/// NOT a stack overflow, NOT a silent parse). The returned error must
/// be exactly `ExpressionTooDeep` carrying the configured `max`.
fn assert_too_deep(input: &str, form: &str, depth: usize) {
    match parse(input) {
        Err(ParseError::ExpressionTooDeep { depth: d, max }) => {
            assert_eq!(
                max, MAX_EXPRESSION_DEPTH,
                "{form}@{depth}: error must report the authoritative cap"
            );
            assert!(
                d > MAX_EXPRESSION_DEPTH,
                "{form}@{depth}: reported depth {d} must exceed the cap {MAX_EXPRESSION_DEPTH}"
            );
        }
        Err(other) => panic!(
            "{form}@{depth}: expected ParseError::ExpressionTooDeep, got {other:?} \
             (the input must be rejected by the depth guard, not another parse path)"
        ),
        Ok(_) => panic!(
            "{form}@{depth}: a {depth}-deep nested expression MUST be rejected — \
             it silently parsed (the DoS guard did not fire)"
        ),
    }
}

fn assert_flat_chain_too_deep(input: &str, form: &str, depth: usize) {
    match parse(input) {
        Err(ParseError::ExpressionTooDeep { depth: d, max }) => {
            assert_eq!(
                max, MAX_FLAT_CHAIN_DEPTH,
                "{form}@{depth}: flat-chain overflow must report the flat-chain cap"
            );
            assert!(
                d > MAX_FLAT_CHAIN_DEPTH,
                "{form}@{depth}: reported depth {d} must exceed the flat-chain cap \
                 {MAX_FLAT_CHAIN_DEPTH}"
            );
        }
        Err(other) => {
            panic!("{form}@{depth}: expected flat-chain ExpressionTooDeep, got {other:?}")
        }
        Ok(_) => panic!("{form}@{depth}: over-cap flat chain must be rejected"),
    }
}

// ---------------------------------------------------------------------
// PART A — the issue's exact repro depths reject cleanly (all 3 forms).
// On origin/main these SIGABRT the process; post-fix they return Err.
// ---------------------------------------------------------------------

#[test]
fn d819_parens_300_deep_rejected_cleanly() {
    assert_too_deep(&nested_parens(300), "parens", 300);
}

#[test]
fn d819_lists_300_deep_rejected_cleanly() {
    assert_too_deep(&nested_lists(300), "lists", 300);
}

#[test]
fn d819_cases_300_deep_rejected_cleanly() {
    assert_too_deep(&nested_cases(300), "cases", 300);
}

/// 1000-deep proves the cap scales — it fires early regardless of how
/// deep the input is, and crucially does so BEFORE pest recurses
/// (pre-fix, 1000-deep overflowed even an 8 MiB main-thread stack).
#[test]
fn d819_parens_1000_deep_rejected_cleanly() {
    assert_too_deep(&nested_parens(1000), "parens", 1000);
}

#[test]
fn d819_lists_1000_deep_rejected_cleanly() {
    assert_too_deep(&nested_lists(1000), "lists", 1000);
}

#[test]
fn d819_cases_1000_deep_rejected_cleanly() {
    assert_too_deep(&nested_cases(1000), "cases", 1000);
}

/// An extreme depth (100_000 — a ~200 KB query) must STILL reject
/// cleanly and cheaply (the O(n) pre-scan rejects on the first byte
/// past the cap; it does not recurse). Guards against any accidental
/// reintroduction of recursion before the cap check.
#[test]
fn d819_pathological_100k_deep_rejected_cleanly() {
    assert_too_deep(&nested_parens(100_000), "parens", 100_000);
    assert_too_deep(&nested_lists(100_000), "lists", 100_000);
    assert_too_deep(&nested_cases(100_000), "cases", 100_000);
}

// ---------------------------------------------------------------------
// PART B — the cap does NOT over-reject valid moderately-nested input.
// just-under-cap parses; exactly-at-cap parses; one-over rejects.
// (This is the risk the issue calls out: a cap set too low breaks
//  legitimate ORM-/query-builder-emitted nested boolean filters.)
// ---------------------------------------------------------------------

#[test]
fn d819_just_under_cap_parens_parse_ok() {
    // CAP-1 nested parens must parse without error.
    let d = MAX_EXPRESSION_DEPTH - 1;
    assert!(
        parse(&nested_parens(d)).is_ok(),
        "a {d}-deep (just-under-cap) paren expression must parse"
    );
}

#[test]
fn d819_at_cap_all_forms_parse_ok() {
    // Exactly CAP nested levels must parse (the boundary is inclusive:
    // depth == MAX is accepted, depth == MAX+1 is rejected).
    let d = MAX_EXPRESSION_DEPTH;
    assert!(
        parses_ok_on_boundary_stack(nested_parens(d), "parens", d),
        "parens@{d} must parse"
    );
    assert!(
        parses_ok_on_boundary_stack(nested_lists(d), "lists", d),
        "lists@{d} must parse"
    );
    assert!(
        parses_ok_on_boundary_stack(nested_cases(d), "cases", d),
        "cases@{d} must parse"
    );
}

#[test]
fn d819_one_over_cap_all_forms_reject() {
    // depth == MAX+1 is the first rejected level, for every form.
    let d = MAX_EXPRESSION_DEPTH + 1;
    assert_too_deep(&nested_parens(d), "parens", d);
    assert_too_deep(&nested_lists(d), "lists", d);
    assert_too_deep(&nested_cases(d), "cases", d);
}

/// A realistic ORM-style moderately-nested boolean filter (depth ~12)
/// MUST parse — this is the false-positive the cap must avoid.
#[test]
fn d819_realistic_nested_filter_parses() {
    let input = "MATCH (n) WHERE ".to_string()
        + &"(".repeat(12)
        + "n.score > 0.5 AND n.active = true"
        + &")".repeat(12)
        + " RETURN n";
    assert!(
        parse(&input).is_ok(),
        "a depth-12 ORM-style boolean filter must parse (no over-rejection)"
    );
}

#[test]
fn d1290_wide_500_predicate_filter_parses_and_binds() {
    let input = wide_where_predicates(500);
    let stmt = parse(&input).expect("500-predicate flat WHERE must parse");
    let catalog = StubCatalogProvider::new();
    arcgraph_query::semantic::BindingVisitor::bind(&stmt, &input, &catalog)
        .expect("500-predicate flat WHERE must bind");
}

// ---------------------------------------------------------------------
// PART C — string-literal safety: brackets / CASE / END inside a string
// literal must NOT count toward nesting depth (else a valid query with
// bracket-heavy string content would be false-rejected).
// ---------------------------------------------------------------------

#[test]
fn d819_brackets_inside_string_literal_do_not_count() {
    // A single short expression whose STRING content has far more
    // brackets than the cap — must parse OK (the brackets are opaque
    // string bytes, not nesting).
    let many = "(".repeat(MAX_EXPRESSION_DEPTH * 10);
    let single = format!("RETURN '{many}'");
    assert!(
        parse(&single).is_ok(),
        "{}+ open-parens INSIDE a string literal must not trip the depth guard",
        MAX_EXPRESSION_DEPTH * 10
    );

    // Same for double-quoted strings with brackets.
    let brackets = "[".repeat(MAX_EXPRESSION_DEPTH * 10);
    let dq = format!("RETURN \"{brackets}\"");
    assert!(
        parse(&dq).is_ok(),
        "brackets inside a double-quoted string literal must not count"
    );
}

/// Adversarial string-desync: a string that CLOSES followed by a real
/// deep nest MUST still be rejected — the closed string's brackets are
/// skipped, but the genuine deep nest after it is counted. This pins
/// the under-count-direction safety: the scanner must NOT keep treating
/// bytes as "in string" past a real close quote (which would let the
/// trailing brackets slip past uncounted and reach pest → overflow).
/// ArcQL escapes only via backslash (`''` is two strings, not one
/// escaped quote), so a bare quote always closes.
#[test]
fn d819_closed_string_then_deep_nest_still_rejected() {
    // `'<many parens>' + <deep parens>`: the parens inside the closed
    // string don't count; the deep nest after `+` does.
    let in_string = "(".repeat(MAX_EXPRESSION_DEPTH * 5);
    let deep = "(".repeat(300);
    let close = ")".repeat(300);
    let input = format!("RETURN '{in_string}' + {deep}1{close}");
    assert_too_deep(&input, "closed_string_then_deep", 300);

    // The empty-string `''` prefix (NOT a doubled-quote escape in ArcQL)
    // followed by a deep nest must also reject — `''` is the empty
    // string, then the parens are real.
    let input2 = format!("RETURN ''{}1{}", "(".repeat(300), ")".repeat(300));
    assert_too_deep(&input2, "empty_string_then_deep", 300);
}

#[test]
fn d819_case_keyword_inside_string_and_identifier_do_not_count() {
    // `CASE` / `END` as STRING content — must not count.
    let casey = "CASE WHEN ".repeat(MAX_EXPRESSION_DEPTH * 5);
    let in_string = format!("RETURN '{casey} done'");
    assert!(
        parse(&in_string).is_ok(),
        "the literal text 'CASE WHEN …' inside a string must not count as nesting"
    );

    // `CASE` as a SUBSTRING of an identifier (word-boundary check) —
    // `mycaseval` must not be read as the CASE keyword.
    let identy = "mycaseval + ".repeat(MAX_EXPRESSION_DEPTH) + "1";
    let q = format!("RETURN {identy}");
    assert!(
        parse(&q).is_ok(),
        "an identifier containing 'case' as a substring must not count as a CASE keyword"
    );
}

// ---------------------------------------------------------------------
// PART D — END-TO-END: a cap-depth VALID expression EVALUATES correctly
// (parse → bind → type-check → lower → eval), proving (a) no over-
// rejection and (b) the evaluator is transitively protected (it never
// sees a tree deeper than the parser admits). Over-cap propagates as a
// clean engine error, NOT a crash.
//
// This is the ADR-133 §D-4 "Query" active-verification path
// (`QueryEngine::execute`) — the EXACT pipeline the TCK ratchet drives.
// ---------------------------------------------------------------------

fn execute(cypher: &str) -> Result<Vec<Vec<Value>>, String> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine
        .execute(cypher, &substrate)
        .map(|r| r.rows)
        .map_err(|e| e.to_string())
}

fn bind_return_expression(cypher: &str) -> BoundExpression {
    let stmt = parse(cypher).expect("query must parse");
    let catalog = StubCatalogProvider::new();
    let bound =
        arcgraph_query::semantic::BindingVisitor::bind(&stmt, cypher, &catalog).expect("bind");
    let BoundStatement::Read(query) = bound else {
        panic!("expected read query");
    };
    let Some(BoundClause::Return(ret)) = query.clauses.last() else {
        panic!("expected trailing RETURN clause");
    };
    let Some(item) = ret.items.first() else {
        panic!("expected first RETURN item");
    };
    let BoundProjectionKind::Expr(expr) = &item.kind else {
        panic!("expected expression projection");
    };
    expr.clone()
}

#[test]
fn d819_cap_depth_paren_expression_evaluates_end_to_end() {
    // CAP nested parens around a literal core that evaluates. This keeps
    // the assertion scoped to the bracket/CASE parser-recursion cap; flat
    // operator folds have their own higher cap below.
    let d = MAX_EXPRESSION_DEPTH;
    let cypher = "RETURN ".to_string() + &"(".repeat(d) + "1" + &")".repeat(d);
    let rows = execute(&cypher).expect("cap-depth paren expression must evaluate");
    assert_eq!(
        rows,
        vec![vec![Value::Integer(1)]],
        "((…1…)) at depth {d} must evaluate to 1"
    );
}

#[test]
fn d819_cap_depth_case_expression_evaluates_end_to_end() {
    let d = MAX_EXPRESSION_DEPTH;
    let cypher =
        "RETURN ".to_string() + &"CASE WHEN true THEN ".repeat(d) + "42" + &" ELSE 0 END".repeat(d);
    let rows = execute(&cypher).expect("cap-depth CASE expression must evaluate");
    assert_eq!(
        rows,
        vec![vec![Value::Integer(42)]],
        "nested CASE at depth {d} must evaluate to 42"
    );
}

#[test]
fn d819_over_cap_propagates_clean_engine_error_not_crash() {
    // Over-cap input surfaces as a clean engine error through the FULL
    // pipeline — the server stays up (in a real deployment this becomes
    // a Bolt FAILURE, not a SIGABRT).
    let d = MAX_EXPRESSION_DEPTH + 100;
    let err = execute(&nested_parens(d)).expect_err("over-cap must error, not crash");
    assert!(
        err.contains("nests too deep"),
        "over-cap engine error must be the depth-cap error, got: {err}"
    );
}

// ---------------------------------------------------------------------
// PART E — pin the cap constant + safety band (mirrors the W22-DB-ε
// `DEFAULT_MAX_DEPTH` pin pattern in `w22_dbe_depth_limit_dos.rs`).
// A future "make it configurable" refactor that defaults to a huge
// value (removing the protection) must surface as a test failure here.
// ---------------------------------------------------------------------

/// Binding: the expression-depth cap is the #819 DoS-protection floor.
/// Lifting it requires re-measuring the pest-overflow cliff on the
/// target worker stack (see the `parser.rs` budget comment) — do NOT
/// raise it without that evidence.
#[test]
fn d819_max_expression_depth_pinned_at_64() {
    assert_eq!(
        MAX_EXPRESSION_DEPTH, 64,
        "MAX_EXPRESSION_DEPTH MUST stay at 64 (measured: pest overflows a 2 MiB Tokio \
         worker at depth 100; 64 survives even a 512 KiB stack ⇒ ~4× margin). \
         Raising it requires re-measuring the on-box pest-overflow cliff."
    );
}

/// Bound-class invariant: the cap must sit in `[16, 100)`. Below 16 it
/// would block legitimately-nested boolean filters; at/above 100 it
/// stops protecting the pest matcher on a 2 MiB worker (the measured
/// overflow cliff). Const-evaluated so a future non-`const`
/// (configurable) refactor forces an explicit re-pin at the config
/// boundary.
#[test]
#[allow(clippy::assertions_on_constants)]
fn d819_max_expression_depth_within_safety_band() {
    const _: () = assert!(
        MAX_EXPRESSION_DEPTH >= 16,
        "MAX_EXPRESSION_DEPTH too low: legitimate nested boolean filters need >= 16"
    );
    const _: () = assert!(
        MAX_EXPRESSION_DEPTH < 100,
        "MAX_EXPRESSION_DEPTH too high: the pest matcher overflows a 2 MiB worker at depth 100"
    );
    // Runtime mirror so the band shows in `cargo test` output.
    assert!((16..100).contains(&MAX_EXPRESSION_DEPTH));
}

// ---------------------------------------------------------------------
// PART F — UNARY-OPERATOR CHAINING (family (B), the #819 R1 residual).
//
// `unary_expr = ("-"|"+") ~ unary_expr` self-recurses once per prefix
// operator with NO literal bracket, so the bracket/CASE pre-scan scored
// it 0 and a ~4 KB `RETURN -+-+ … 1` STILL drove pest to SIGABRT after
// the first cut of this guard (R1 PROBE: parsed OK at 1000 ops,
// SIGABRT at ~3500 ops on a 2 MiB worker; the Bolt `parse_multi` path
// crashed at 4000 ops). The pre-scan now counts consecutive unary
// prefixes toward the same depth budget. These are the load-bearing
// assertions for the residual vector: the deep chain returns a clean
// `Err` (was SIGABRT) and the just-under-cap chain still parses.
// ---------------------------------------------------------------------

#[test]
fn d819_unary_chain_300_ops_rejected_cleanly() {
    // 300 prefix operators — well over the cap. On the pre-fix build
    // this passed the (bracket-only) scan and recursed in pest; now it
    // is a clean depth-cap Err.
    assert_too_deep(&unary_chain(300), "unary", 300);
}

#[test]
fn d819_unary_chain_4000_ops_rejected_cleanly() {
    // The R1 repro depth (`-+`×2000 = 4000 operators, ~4 KB on the
    // wire) that SIGABRT-ed the server. Must reject cleanly and cheaply
    // (the O(n) pre-scan bails on the first over-cap operator).
    assert_too_deep(&unary_chain(4000), "unary", 4000);
}

#[test]
fn d819_pathological_unary_100k_ops_rejected_cleanly() {
    // 100k operators (~100 KB) must STILL reject cheaply, not recurse.
    assert_too_deep(&unary_chain(100_000), "unary", 100_000);
}

#[test]
fn d819_just_under_cap_unary_chain_parses_ok() {
    // A short unary chain (60 operators — well under the cap) must parse
    // without error: the cap must not over-reject legitimate `-`/`+`
    // prefixing. (60 ≤ CAP-ish even after the leading-operator infix
    // mis-classification, which only ever UNDER-counts a keyword-prefixed
    // chain by one — see the parser unit tests.)
    let q = unary_chain(60);
    assert!(
        parse(&q).is_ok(),
        "a 60-operator unary chain (well under cap {MAX_EXPRESSION_DEPTH}) must parse"
    );
    // Small canonical forms must parse too (no over-rejection of the
    // everyday cases). `RETURN -(-(-1))` mixes unary with parens.
    assert!(parse("RETURN -1").is_ok(), "`RETURN -1` must parse");
    assert!(parse("RETURN -+1").is_ok(), "`RETURN -+1` must parse");
    assert!(
        parse("RETURN -(-(-1))").is_ok(),
        "`RETURN -(-(-1))` must parse"
    );
    assert!(
        parse("RETURN [-1, -2, -3]").is_ok(),
        "sibling negatives must parse (unary discharges between commas)"
    );
}

#[test]
fn d819_over_cap_unary_chain_rejects() {
    // Generously over the cap (600 operators) — rejects regardless of
    // the documented ±1 keyword-position imprecision.
    assert_too_deep(&unary_chain(600), "unary", 600);
}

/// Interleaved `-+(` per level stacks unary frames ON TOP of bracket
/// depth (`-(-(-( … )))` is depth 2/level). This is the anti-bypass
/// case: only ~22 brackets (far under the 64 bracket cap — a separate
/// bracket-only cap would ACCEPT it) but a total native depth > cap.
/// Must reject — proving unary and bracket depth are summed, not capped
/// independently (a naive two-cap fix would let this through to pest).
#[test]
fn d819_interleaved_unary_and_brackets_rejected() {
    let interleaved = "RETURN ".to_string() + &"-+(".repeat(40) + "1" + &")".repeat(40);
    assert_too_deep(&interleaved, "interleaved_unary_bracket", 40);
}

// ---------------------------------------------------------------------
// PART G — comment + string opacity at the PARSE level. pest strips
// `--` line and `/* */` block comments, so bracket-/operator-heavy
// comment text must NOT trip the depth guard (the unary counter made
// comment-skipping load-bearing: `-- … +++++` would otherwise count).
// ---------------------------------------------------------------------

#[test]
fn d819_bracket_and_operator_heavy_comment_does_not_false_reject() {
    // A trailing line comment crammed with parens and `+` operators —
    // pest discards it, so it must parse exactly like `RETURN 1`.
    let line = format!(
        "RETURN 1 -- {} {}",
        "(".repeat(MAX_EXPRESSION_DEPTH * 8),
        "+".repeat(MAX_EXPRESSION_DEPTH * 8)
    );
    assert!(
        parse(&line).is_ok(),
        "a `--` comment full of brackets/operators must not trip the depth guard"
    );
    // Block comment in the middle of an expression.
    let block = format!(
        "RETURN 1 /* {} {} */ + 2",
        "[".repeat(MAX_EXPRESSION_DEPTH * 8),
        "-".repeat(MAX_EXPRESSION_DEPTH * 8)
    );
    assert!(
        parse(&block).is_ok(),
        "a `/* */` comment full of brackets/operators must not trip the depth guard"
    );
}

// ---------------------------------------------------------------------
// PART H — END-TO-END eval of a cap-depth unary chain (parse → bind →
// type-check → lower → eval). Proves (a) a deep-but-legal unary chain
// is admitted AND evaluates, and (b) the evaluator (which recurses on
// `UnaryOp { operand }`) never sees a tree deeper than the parser
// admits. This is the ADR-133 §D-4 "Query" active-verification path.
// ---------------------------------------------------------------------

#[test]
fn d819_cap_depth_unary_chain_evaluates_end_to_end() {
    // 64 alternating operators (`-+`×32) applied to 1. There are 32
    // `Neg`s ⇒ value = (-1)^32 = +1. A chain this length parses (it is
    // ~46× under the pest cliff) and must evaluate without overflowing
    // the evaluator's own recursion.
    let cypher = unary_chain(64);
    let rows = execute(&cypher).expect("cap-depth unary chain must evaluate");
    assert_eq!(
        rows,
        vec![vec![Value::Integer(1)]],
        "`-+-+…`×64 applied to 1 (32 negations) must evaluate to 1"
    );
}

#[test]
fn d819_over_cap_unary_chain_propagates_clean_engine_error_not_crash() {
    // Over-cap unary surfaces as a clean engine error through the FULL
    // pipeline (the server stays up — a Bolt FAILURE, not a SIGABRT).
    let err = execute(&unary_chain(4000)).expect_err("over-cap unary must error, not crash");
    assert!(
        err.contains("nests too deep"),
        "over-cap unary engine error must be the depth-cap error, got: {err}"
    );
}

#[test]
fn d1290_flat_chain_at_cap_evaluates_end_to_end() {
    let cypher = format!("RETURN {}true", "true AND ".repeat(MAX_FLAT_CHAIN_DEPTH));
    let expr = bind_return_expression(&cypher);
    let value = evaluate(&expr, &[], &|_| None, &Default::default())
        .expect("flat chain at cap must evaluate");
    assert_eq!(
        value,
        Value::Boolean(true),
        "{MAX_FLAT_CHAIN_DEPTH}-operator flat AND chain must evaluate safely"
    );
}

#[test]
fn d1290_flat_chain_cap_plus_one_rejected_cleanly() {
    let cypher = format!(
        "RETURN {}true",
        "true AND ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)
    );
    assert_flat_chain_too_deep(&cypher, "flat_and", MAX_FLAT_CHAIN_DEPTH + 1);
}

#[test]
fn d1290_pathological_flat_chain_100k_rejected_cleanly() {
    let cypher = format!("RETURN {}true", "true AND ".repeat(100_000));
    assert_flat_chain_too_deep(&cypher, "flat_and", 100_000);
}

// ---------------------------------------------------------------------
// PART I — COMPLETENESS: every OTHER recursive expression-parse form is
// depth-counted too (they all carry a literal `( [ {`, so the existing
// bracket balance bounds them). Together with the unary coverage above,
// this makes the depth bound COMPLETE across every recursive
// `primary_atom` / `accessor` / `unary_expr` path — there is no
// remaining unbounded nesting form (so the #819 / W22-DB-ε ADV-1
// known-issue is fully closed, not partially). Verified per the R1
// follow-up directive ("verify each recursive expr-parse path is
// depth-counted").
// ---------------------------------------------------------------------

#[test]
fn d819_all_bracketed_recursive_forms_are_depth_counted() {
    let d = MAX_EXPRESSION_DEPTH + 1;
    // Nested function-call args: f(f(f(…1…))) — each `f(` is one `(`.
    let fcalls = "RETURN ".to_string() + &"f(".repeat(d) + "1" + &")".repeat(d);
    assert_too_deep(&fcalls, "nested_function_calls", d);
    // Nested map literals: {a:{a:{…1…}}} — each `{` is counted.
    let maps = "RETURN ".to_string() + &"{a:".repeat(d) + "1" + &"}".repeat(d);
    assert_too_deep(&maps, "nested_maps", d);
    // Nested subscript / index: a[[[…0…]]] — each `[` is counted.
    let subs = "RETURN a".to_string() + &"[".repeat(d) + "0" + &"]".repeat(d);
    assert_too_deep(&subs, "nested_subscripts", d);
    // Nested list comprehensions still open `[` → counted.
    let comp = "RETURN ".to_string() + &"[x IN ".repeat(d) + "[1]" + &"]".repeat(d);
    assert_too_deep(&comp, "nested_list_comprehensions", d);
}

// ---------------------------------------------------------------------
// PART J — #1290 fix-4: EVERY pipeline spine walk is ITERATIVE, so a
// legitimate wide flat operator chain executes END-TO-END through
// `QueryEngine::execute` (parse → bind → type-check → cross-substrate →
// lower → enumerate → cost-pick → execute → render). Pre-fix, the
// recursive spine walkers (type-check / lowering / cost / cache-key /
// Display / eval residuals) overflowed the native stack at ~28
// predicates in the debug profile — a SIGABRT that killed the whole
// server. These tests run on DEFAULT libtest stacks ON PURPOSE: the
// point is that no thread-size accommodation is needed anymore.
// ---------------------------------------------------------------------

use arcgraph_core::{LabelId, NodeId, TenantId};
use arcgraph_query::executor::value::NodeView;

/// A substrate with exactly ONE node whose properties `p1..pN = 1..N`
/// satisfy every conjunct of [`wide_where_predicates`], plus one node
/// whose `p1` mismatches (so the filter chain provably RAN — the E2E
/// assertion is `1 row`, not merely "no crash").
fn wide_filter_substrate(predicates: usize) -> StubExecutorSubstrate {
    let mut matching = NodeView::new(NodeId::new(1), Some(LabelId::new(1)));
    let mut failing = NodeView::new(NodeId::new(2), Some(LabelId::new(1)));
    for i in 1..=predicates {
        matching = matching.with_property(format!("p{i}"), Value::Integer(i as i64));
        let v = if i == 1 { -1 } else { i as i64 };
        failing = failing.with_property(format!("p{i}"), Value::Integer(v));
    }
    StubExecutorSubstrate::new()
        .with_node(TenantId::DEFAULT, matching)
        .with_node(TenantId::DEFAULT, failing)
}

fn execute_on(cypher: &str, substrate: &StubExecutorSubstrate) -> Result<Vec<Vec<Value>>, String> {
    let catalog = StubCatalogProvider::new();
    let engine = QueryEngine::new(&catalog);
    engine
        .execute(cypher, substrate)
        .map(|r| r.rows)
        .map_err(|e| e.to_string())
}

/// The load-bearing E2E assertion: an N-predicate flat WHERE executes
/// through `QueryEngine::execute` and FILTERS (1 of 2 nodes passes).
/// N = 200 / 500 / 1000 — the pre-fix pipeline SIGABRT-ed at ~28.
#[test]
fn d1290_wide_flat_where_executes_e2e_200() {
    let rows = execute_on(&wide_where_predicates(200), &wide_filter_substrate(200))
        .expect("200-predicate flat WHERE must execute E2E");
    assert_eq!(rows.len(), 1, "exactly the all-match node passes at N=200");
}

#[test]
fn d1290_wide_flat_where_executes_e2e_500() {
    let rows = execute_on(&wide_where_predicates(500), &wide_filter_substrate(500))
        .expect("500-predicate flat WHERE must execute E2E");
    assert_eq!(rows.len(), 1, "exactly the all-match node passes at N=500");
}

#[test]
fn d1290_wide_flat_where_executes_e2e_1000() {
    let rows = execute_on(&wide_where_predicates(1000), &wide_filter_substrate(1000))
        .expect("1000-predicate flat WHERE must execute E2E");
    assert_eq!(rows.len(), 1, "exactly the all-match node passes at N=1000");
}

/// Every flat-chain operator family executes E2E at 200 operators.
/// (A bare `NOT NOT …` chain is a DIFFERENT recursion family — pest
/// genuinely recurses per prefix, so it is bounded by
/// `MAX_EXPRESSION_DEPTH` = 64 at parse time per #819; the NOT family
/// is exercised here the way a real query uses it wide — one negation
/// per conjunct — plus an at-parse-cap 60-deep bare chain.)
#[test]
fn d1290_all_operator_families_200_execute_e2e() {
    let substrate = StubExecutorSubstrate::new();
    let cases: Vec<(String, Value, &str)> = vec![
        (
            format!("RETURN {}true", "true AND ".repeat(200)),
            Value::Boolean(true),
            "AND",
        ),
        (
            format!("RETURN {}true", "false OR ".repeat(200)),
            Value::Boolean(true),
            "OR",
        ),
        // 201 TRUE terms; a left-folded XOR of n TRUEs is (n odd) —
        // 201 is odd ⇒ TRUE.
        (
            format!("RETURN {}true", "true XOR ".repeat(200)),
            Value::Boolean(true),
            "XOR",
        ),
        (
            format!("RETURN {}1", "1 + ".repeat(200)),
            Value::Integer(201),
            "additive",
        ),
        (
            format!("RETURN {}1", "1 * ".repeat(200)),
            Value::Integer(1),
            "multiplicative",
        ),
        // Comparison family — an `=`-chain is type-coherent at every
        // level ((true = true) = true …).
        (
            format!("RETURN {}true", "true = ".repeat(200)),
            Value::Boolean(true),
            "comparison",
        ),
        // NOT — one negation per conjunct (the realistic wide form).
        (
            format!("RETURN {}NOT false", "NOT false AND ".repeat(199)),
            Value::Boolean(true),
            "NOT-per-conjunct",
        ),
        // Bare NOT-chain at 60 (under the #819 parse cap of 64).
        (
            format!("RETURN {}true", "NOT ".repeat(60)),
            Value::Boolean(true),
            "NOT-chain-at-parse-cap",
        ),
        // Keyword postfix families (#1290 — previously uncounted AND
        // recursively walked): IN / IS NULL / string predicates.
        (
            format!("RETURN 1 IN [1]{}", " IN [true]".repeat(199)),
            Value::Boolean(true),
            "IN-chain",
        ),
        // null IS NULL ⇒ true; true IS NULL ⇒ false; false IS NULL ⇒
        // false thereafter.
        (
            format!("RETURN null{}", " IS NULL".repeat(200)),
            Value::Boolean(false),
            "IS-NULL-chain",
        ),
        (
            format!(
                "RETURN {}{}{}true",
                "'ab' STARTS WITH 'a' AND ".repeat(100),
                "'ab' ENDS WITH 'b' AND ".repeat(50),
                "'abc' CONTAINS 'b' AND ".repeat(50)
            ),
            Value::Boolean(true),
            "string-predicates-per-conjunct",
        ),
    ];
    for (cypher, expected, family) in cases {
        let rows = execute_on(&cypher, &substrate)
            .unwrap_or_else(|e| panic!("{family} chain at 200 must execute E2E, got: {e}"));
        assert_eq!(
            rows,
            vec![vec![expected]],
            "{family} chain at 200 must evaluate to the pinned value"
        );
    }
}

/// Semantics unchanged by the iterativization: associativity,
/// precedence, 3VL null handling, and MIXED-variant spine order
/// (`BinaryOp` / `In` / `IsNull` interleaved on one left spine) are
/// pinned E2E.
#[test]
fn d1290_spine_semantics_pinned_e2e() {
    let substrate = StubExecutorSubstrate::new();
    let cases: Vec<(&str, Value)> = vec![
        // Left associativity of subtraction / division.
        ("RETURN 1 - 2 - 3", Value::Integer(-4)),
        ("RETURN 100 / 10 / 5", Value::Integer(2)),
        // Precedence — `*` binds tighter than `-`.
        ("RETURN 2 - 3 * 4", Value::Integer(-10)),
        // 3VL.
        ("RETURN null AND false", Value::Boolean(false)),
        ("RETURN null OR true", Value::Boolean(true)),
        ("RETURN null XOR true", Value::Null),
        ("RETURN NOT NOT true", Value::Boolean(true)),
        // MIXED left spine: ((1 = 1) IN [true]) IS NULL ⇒ false.
        ("RETURN 1 = 1 IN [true] IS NULL", Value::Boolean(false)),
        // Mixed families on one spine: comparison over an additive
        // chain result.
        ("RETURN 1 + 2 + 3 = 6", Value::Boolean(true)),
    ];
    for (cypher, expected) in cases {
        let rows =
            execute_on(cypher, &substrate).unwrap_or_else(|e| panic!("`{cypher}` errored: {e}"));
        assert_eq!(rows, vec![vec![expected]], "`{cypher}` semantics pinned");
    }
}

/// The plan-cache key derivation (AST clone → canonicalize walk →
/// `Display` render — all spine-iterative post-#1290) survives a deep
/// flat chain: EXPLAIN through an engine WITH a plan cache attached
/// derives the key for a 500-predicate WHERE without overflow.
#[test]
fn d1290_plan_cache_key_derivation_survives_deep_chain() {
    use arcgraph_query::planner::cache::PlanCache;
    use std::sync::Arc;
    let catalog = StubCatalogProvider::new();
    let engine = QueryEngine::new(&catalog).with_cache(Arc::new(PlanCache::new()));
    let substrate = StubExecutorSubstrate::new();
    let cypher = format!("EXPLAIN {}", wide_where_predicates(500));
    let result = engine
        .execute(&cypher, &substrate)
        .expect("EXPLAIN over a 500-predicate WHERE must derive a plan-cache key and render");
    assert!(
        !result.rows.is_empty(),
        "EXPLAIN must return plan rows for the deep-chain query"
    );
}

/// Over-cap chains reject with a clean `ExpressionTooDeep` THROUGH
/// `QueryEngine::execute` — for every family, including the #1290
/// keyword postfix forms that were previously uncounted. No SIGABRT.
#[test]
fn d1290_over_cap_chains_reject_gracefully_via_execute() {
    let substrate = StubExecutorSubstrate::new();
    let over = MAX_FLAT_CHAIN_DEPTH + 1;
    let cases = [
        format!("RETURN {}true", "true AND ".repeat(over)),
        format!("RETURN {}1", "1 + ".repeat(over)),
        format!("RETURN 1{}", " IN [1]".repeat(over)),
        format!("RETURN null{}", " IS NULL".repeat(over)),
        format!("RETURN 'a'{}", " STARTS WITH 'a'".repeat(over)),
        format!("RETURN 'a'{}", " ENDS WITH 'a'".repeat(over)),
        format!("RETURN 'a'{}", " CONTAINS 'a'".repeat(over)),
    ];
    for cypher in &cases {
        let err = execute_on(cypher, &substrate)
            .expect_err("an over-cap flat chain must error, not crash");
        assert!(
            err.contains("nests too deep"),
            "over-cap chain must surface the depth-cap error, got: {err}"
        );
    }
}

/// Keyword postfix chains draw from the SAME flat-chain budget as the
/// symbol operators (pre-#1290 they were uncounted — an unbounded
/// bypass): the parse-time reject reports the flat-chain cap.
#[test]
fn d1290_keyword_postfix_chains_count_toward_flat_cap() {
    let over = MAX_FLAT_CHAIN_DEPTH + 1;
    assert_flat_chain_too_deep(
        &format!("RETURN 1{}", " IN [1]".repeat(over)),
        "in_chain",
        over,
    );
    assert_flat_chain_too_deep(
        &format!("RETURN null{}", " IS NULL".repeat(over)),
        "is_null_chain",
        over,
    );
    assert_flat_chain_too_deep(
        &format!("RETURN 'a'{}", " STARTS WITH 'a'".repeat(over)),
        "starts_with_chain",
        over,
    );
    // Pathological 100k-op keyword chains reject cheaply (O(n) scan).
    assert_flat_chain_too_deep(
        &format!("RETURN 1{}", " IN [1]".repeat(100_000)),
        "in_chain",
        100_000,
    );
    assert_flat_chain_too_deep(
        &format!("RETURN null{}", " IS NULL".repeat(100_000)),
        "is_null_chain",
        100_000,
    );
}

/// Bracket nesting cannot MULTIPLY the flat-chain budget (#1290 R1
/// composition residual): the cap is cumulative along the bracket
/// path, so K bracket levels × (cap/K + slack) ops compose to a
/// reject, while every single level is far under the cap.
#[test]
fn d1290_bracket_composed_chains_cannot_multiply_flat_budget() {
    // 8 levels × 1024 ops = 8192 total parked+current > 4096 cap,
    // though each level alone (1024) is only a quarter of the cap.
    let levels = 8;
    let per_level = MAX_FLAT_CHAIN_DEPTH / 4;
    let mut q = String::from("RETURN ");
    for _ in 0..levels {
        q.push_str(&"true AND ".repeat(per_level));
        q.push('(');
    }
    q.push_str("true");
    q.push_str(&")".repeat(levels));
    match parse(&q) {
        Err(ParseError::ExpressionTooDeep { max, .. }) => {
            assert_eq!(
                max, MAX_FLAT_CHAIN_DEPTH,
                "composed bracket×chain overflow must report the flat-chain cap"
            );
        }
        other => panic!(
            "8×{per_level}-op bracket-composed chain must reject as ExpressionTooDeep, got {:?}",
            other.map(|_| "Ok(..)")
        ),
    }

    // The accept side: a REALISTIC composition (parenthesized
    // sub-expressions inside a wide filter) stays admitted — parking
    // is cumulative, not a blanket reject.
    let mut ok = String::from("RETURN ");
    for _ in 0..8 {
        ok.push_str(&"true AND ".repeat(10));
        ok.push('(');
    }
    ok.push_str("true");
    ok.push_str(&")".repeat(8));
    assert!(
        parse(&ok).is_ok(),
        "an 8-level × 10-op composition (well under cap) must parse"
    );
}

/// The cap-margin measurement harness: a NEAR-CAP (4000-predicate)
/// wide WHERE — the deepest legitimate shape the raised cap admits —
/// executes E2E on a DEFAULT test stack in both debug and release.
/// This bounds the stack cost of the RESIDUAL recursive passes at the
/// cap (derived `Clone`/`Drop` glue on the 4000-deep spine + the
/// 4000-deep nested-`LogicalFilter` plan chain through plan compile /
/// cost / executor pull). If a future change fattens any of those
/// per-level frames past the margin, THIS test SIGABRTs before a
/// production server does.
#[test]
fn d1290_near_cap_4000_predicate_filter_executes_e2e() {
    let rows = execute_on(&wide_where_predicates(4000), &wide_filter_substrate(4000))
        .expect("4000-predicate (near-cap) flat WHERE must execute E2E");
    assert_eq!(rows.len(), 1, "exactly the all-match node passes at N=4000");
}

/// A near-cap RETURN-position chain (no plan-depth involvement —
/// isolates the expression-pipeline half of the margin measurement).
#[test]
fn d1290_at_cap_return_chain_executes_e2e() {
    let substrate = StubExecutorSubstrate::new();
    let cypher = format!("RETURN {}true", "true AND ".repeat(MAX_FLAT_CHAIN_DEPTH));
    let rows = execute_on(&cypher, &substrate).expect("at-cap RETURN chain must execute E2E");
    assert_eq!(rows, vec![vec![Value::Boolean(true)]]);
}

/// Pin the raised backstop cap. The primary defense post-#1290 is the
/// ITERATIVE spine walks; this cap bounds the residual recursive
/// passes (derived `Clone`/`Drop`/`PartialEq` glue + the per-conjunct
/// nested-`LogicalFilter` plan chain). Raising it further requires
/// re-measuring those residuals' stack cost at the new cap (the E2E
/// at-cap tests here are the measurement harness).
#[test]
fn d1290_max_flat_chain_depth_pinned_at_4096() {
    assert_eq!(
        MAX_FLAT_CHAIN_DEPTH, 4096,
        "MAX_FLAT_CHAIN_DEPTH must stay at 4096 — high enough for any legitimate \
         generated filter (hundreds to low-thousands of predicates), low enough \
         that the residual recursive passes keep stack margin (measured via the \
         PART J E2E tests on default test stacks)"
    );
}
