//! ArcQL parser — `pest` pairs to typed AST.
//!
//! # Scope (M4-01 / ADR-038 §2 D-1)
//!
//! `parse(input)` MUST:
//! - Accept the v1.0 lit grammar and the v1.0 reserved-but-
//!   unimplemented grammar (ADR-038 D-1..D-10) without
//!   distinction. Reservation-vs-lit is M4-02 territory.
//! - Reject **only** syntactic ill-formedness. No variable binding,
//!   no type checking, no label-existence check.
//! - Map `pest::error::Error<Rule>` to a `Rule`-free
//!   [`crate::error::ParseError::Pest`] so the semantic analyzer
//!   does not pull in grammar-internal symbols.
//!
//! # Architecture
//!
//! The flow is two-pass per pest discipline:
//!
//! 1. `Grammar::parse(Rule::query, input)` produces a
//!    `Pairs<Rule>` tree.
//! 2. Recursive-descent transformers in this file fold the tree
//!    into the typed AST in `ast.rs`.
//!
//! The transformer functions are pure (no globals, no shared
//! mutable state per no-shared-mutable-state rule). All
//! errors thread through the `Result` chain — no panics in
//! library code.
//!
//! # ADR provenance
//! - ADR-006 D-1; ADR-038 §2 D-1..D-10; ADR-038 D-16 (M4-02 owns
//!   the executor-side `NotImplemented` half).

use pest::Parser;
use pest::iterators::{Pair, Pairs};
use pest_derive::Parser;

use crate::ast::*;
use crate::error::{ParseError, Span};

#[derive(Parser)]
#[grammar = "grammar.pest"]
struct ArcQLGrammar;

// =====================================================================
// Expression-nesting-depth guard (#819 — DoS hardening)
// =====================================================================
//
// LATENCY/MEMORY BUDGET.
//
// THE THREAT (#819). A deeply-nested expression drives unbounded
// native-stack recursion → Rust runtime `abort()` (SIGABRT) → the
// WHOLE server dies for all tenants. Bolt auth is accepted-but-not-
// enforced today, so this is an unauthenticated remote DoS via a
// ~600-byte–to–~4 KB query. A stack overflow is UNCATCHABLE
// (`catch_unwind` cannot recover an abort), so the fix MUST be a depth
// *check that returns `Err`* BEFORE the stack is exhausted — never a
// panic-catcher.
//
// There are TWO families of unbounded-recursion form, BOTH bounded
// here (the second was the #819 R1 residual — a unary chain bypassed
// the first cut of this guard):
//   (A) BRACKET-style nesting — each level re-enters the precedence
//       ladder once: parens `((((1))))`, list `[[[[1]]]]`, map
//       `{a:{a:{…}}}`, subscript `x[x[x[…]]]`, function-call args
//       `f(f(f(…)))`, and the keyword-bracketed `CASE WHEN … THEN
//       CASE WHEN … 1 END END`. ALL of these carry a literal bracket
//       (`( [ {`) or the `CASE`/`END` keyword pair, so the pre-parse
//       scan counts them by bracket/keyword balance.
//   (B) UNARY-operator chaining — `unary_expr = ("-"|"+") ~ unary_expr`
//       self-recurses once per prefix operator, with NO bracket:
//       `RETURN -+-+-+ … 1` (alternating, ~4 KB on the wire). This
//       carries no bracket, so the bracket balance under-counts it to
//       0 — the residual crash vector the R1 found. The scan now
//       counts consecutive unary-prefix operators toward the same
//       depth budget (see `check_pre_parse_nesting_depth`).
//   (Bare `NOT NOT …` and `----` / adjacent `--` do NOT recurse:
//    `kw_not?` is non-repeating so `NOT NOT x` is a parse error, and
//    `--` opens a line comment — both verified on-box. `NOT`-via-parens
//    is bracket-counted. So `-`/`+` is the only un-bracketed recursive
//    form, and counting it makes the bound COMPLETE across every
//    recursive expression-parse path.)
//
// TWO RECURSION SITES — both must be bounded (this is the subtle part
// the initial single-guard attempt missed; verified empirically, see
// below):
//
//   (1) The pest PEG matcher inside `ArcQLGrammar::parse(Rule::query,
//       …)` recurses on the grammar's `expression` rule nesting and
//       overflows the stack BUILDING the `Pairs` tree — BEFORE any of
//       our AST-walk code runs. This is the DOMINANT site on a real
//       Tokio worker stack.
//   (2) Our recursive-descent AST builder re-enters `parse_expression`
//       / `parse_where_expr` once per BRACKET nesting level; each level
//       expands into the 11-frame precedence ladder (`parse_expression →
//       parse_or_expr → parse_xor_expr → parse_and_expr → parse_not_expr
//       → parse_comparison_expr → parse_add_expr → parse_mul_expr →
//       parse_unary_expr → parse_atom → parse_primary_atom`) before
//       recursing into the nested sub-expression. SEPARATELY,
//       `parse_unary_expr` self-recurses once per unary-prefix operator
//       (family (B)) — its own RAII guard bounds that.
//
// So site (1) is guarded by a cheap PRE-PARSE depth SCAN
// (`check_pre_parse_nesting_depth`, run BEFORE `ArcQLGrammar::parse`)
// and site (2) by the RAII [`DepthGuard`] threaded through the two
// ladder funnels AND `parse_unary_expr`'s prefix arm. The pre-scan is
// load-bearing for pest (which overflows FIRST, before any AST-walk
// code runs); the AST guards are defense-in-depth. The pre-scan counts
// THREE things by a single O(n) running-balance pass: bracket balance
// (`( [ {` vs `) ] }`), the `CASE … END` keyword pair, AND consecutive
// unary-prefix `-`/`+` operators (family (B)). It skips `'…'`/`"…"`
// string literals and `--` line / `/* */` block comments (pest strips
// comments, so a `-- … +++++` comment must not be miscounted).
//
// MEASURED on this box (Mac M-series), RELEASE build, on a 2 MiB
// stack (== Tokio's default worker-thread stack size — the server
// configures no custom stack, so 2 MiB is the binding constraint):
//
//   | nesting form        | last depth OK | first depth SIGABRT |
//   | paren  `(…)`         |      99       |        100          |
//   | list   `[…]`         |      99       |        100          |
//   | CASE … END          |      99       |        100          |
//   | unary  `-+-+…`       |  ~3000–3500   |     ~3000–3500      |
//
//   The unary cliff is ~30× DEEPER than the bracket cliff because a
//   unary level costs ~1 pest `unary_expr` frame vs the bracket
//   ladder's ~11–12 frames/level (R1 repro: OK at 1500 `-+` pairs =
//   3000 ops, SIGABRT at 1750 pairs = 3500 ops). On a 512 KiB stack the
//   bracket cliff drops to depth 64 and the unary cliff to ~500–1000
//   ops. ONE cap (64) covers BOTH families with margin: ~64 % of the
//   bracket cliff, ~2 % of the unary cliff on 2 MiB; ~8–15× under the
//   unary cliff even on a 512 KiB stack.
//   (The issue reported the bracket crash at depth 250–300, which
//    implies its server worker ran with a larger-than-2 MiB stack; we
//    size the cap against the WORST realistic case — the 2 MiB default
//    worker. The unary residual crashed at ~4 KB / 4000 ops.)
//
// CAP = 64. Rationale:
//   • The deepest ACCEPTED input (depth 64) parses cleanly even on a
//     512 KiB stack — i.e. ~4× stack headroom on the real 2 MiB worker
//     for pest (site 1) PLUS the AST walk (site 2) PLUS the evaluator's
//     own recursion over the bound AST PLUS `#[tracing::instrument]` /
//     async frame bloat PLUS the caller's stack budget, all of which
//     stack on top under real load.
//   • Pest overflows at depth 100 (bracket) / ~3000 (unary) on a 2 MiB
//     worker; 64 sits well below BOTH cliffs, so pest NEVER overflows on
//     accepted input — for either recursion family.
//   • 64 matches the crate's two other network-reachable recursion
//     bounds — `MAX_JSON_DECODE_DEPTH = 64` (`executor::value`, the
//     JSON→Value decode guard) and the traversal `DEFAULT_MAX_DEPTH =
//     64` — and is far under the crate `#![recursion_limit = "256"]`.
//   • 64 is far deeper than any legitimate query nests (no eligible
//     openCypher-TCK scenario nests anywhere near it; ORM- /
//     query-builder-emitted boolean filters nest a handful of levels).
//   We prove we sit well INSIDE the bound, not merely at it (doctrine
//   §"EXCEED-THE-SPEC"). The known-issues doc (W22-DB-ε / ADV-1)
//   suggested 256 — but #819's on-box repro shows the pest matcher
//   overflows a 2 MiB worker at depth 100, so 256 would be a FALSE-safe
//   cap that still SIGABRTs. On-box measurement (doctrine §"Active
//   verification") is exactly what catches this.
//
// Why the EVALUATOR needs no separate guard: the parser rejects depth
// > CAP before a single `BoundExpression` node is built, so `evaluate`
// (`executor/eval.rs`, recursing on `BinaryOp { lhs, rhs }` / `UnaryOp
// { operand }` / nested list / `CASE`) can never observe a tree deeper
// than CAP. eval depth ≤ parse depth ≤ CAP = 64, well under its own
// stack budget. (Verified: `tests/expression_depth_dos.rs` evaluates
// both a depth-`CAP` paren expression AND a `CAP`-deep unary chain
// end-to-end without overflow.)
//
/// Maximum expression-nesting depth accepted by the parser. A user
/// expression nesting deeper than this is rejected with
/// [`ParseError::ExpressionTooDeep`] **before** the native stack
/// overflows. This is the AUTHORITATIVE, user-visible cap, enforced by
/// the pre-parse depth scan (`check_pre_parse_nesting_depth`, which
/// guards the pest PEG matcher — the dominant overflow site). The
/// recursive-descent AST builder's RAII `DepthGuard` is a looser
/// internal backstop (see `AST_GUARD_DEPTH_LIMIT`).
///
/// See the module-level budget comment above for the empirical
/// derivation (measured: pest overflows a 2 MiB Tokio worker at depth
/// 100; this cap of 64 survives even a 512 KiB stack ⇒ ~4× margin).
/// This is a runtime DoS guard (#819), distinct from code-quality policy's
/// compile-time `#![recursion_limit]`.
pub const MAX_EXPRESSION_DEPTH: usize = 64;

/// Maximum folded flat-operator chain depth accepted by the parser.
///
/// Flat chains (`a AND b AND c`, `1 + 2 + 3`, repeated comparisons,
/// string predicates, `IN` / `IS NULL` postfix chains, etc.) are parsed
/// by pest as iterative `*` repetitions, not recursive bracket/CASE
/// nesting. They later fold into left-nested AST nodes, so every
/// pipeline stage that walks the tree still needs a DoS guard — but
/// since #1290 the pipeline's spine walks (bind → type-check →
/// cross-substrate → lower → plan-cache key → cost → eval → Display)
/// are ITERATIVE (explicit worklist, O(1) native stack per chain
/// level), so this cap is a BACKSTOP against the residual recursive
/// passes (the compiler-derived `Clone` / `Drop` / `PartialEq` glue on
/// the left-nested tree and the nested-`LogicalFilter` plan chain the
/// per-conjunct WHERE push-down builds), not the primary defense.
///
/// `4096` admits wide but legitimate generated filters (a 4000-predicate
/// `WHERE` uses a 3999-operator boolean chain plus single-comparison
/// operands — far above any observed ORM/query-builder output) and
/// still rejects pathological 100k-wide chains before pest, binder, or
/// evaluator work. A chain over this cap is DoS-intent, not a real
/// query. Empirical margin for the residual recursive passes at the
/// cap: see `tests/expression_depth_dos.rs` (the at-cap E2E execute
/// proof runs on a default 2 MiB test stack in BOTH debug and release).
///
/// The cap is CUMULATIVE along a bracket-nesting path (the open-bracket
/// parking in `DepthScan` carries the outer chain totals forward
/// rather than granting each bracket level a fresh budget), so
/// `(…512 ops… ( …512 ops… ( … )))` cannot multiply the admitted
/// left-spine depth past the cap by composing bracket levels with
/// per-level chains (#1290 R1 composition residual).
pub const MAX_FLAT_CHAIN_DEPTH: usize = 4096;

/// Internal backstop limit for the recursive-descent AST builder's RAII
/// [`DepthGuard`]. Set to `2 × MAX_EXPRESSION_DEPTH` so it never
/// pre-empts the authoritative pre-parse scan on input that scan
/// accepted (the AST walk counts a couple of extra `parse_expression`
/// frames the bracket/`CASE` scan does not — the top-level projection
/// expression + ladder framing — so an equal limit would make the AST
/// guard reject a hair before the pre-scan, blurring the user-visible
/// boundary). Because every path into `parse_expression` is reached
/// via `parse` / `parse_multi` (which run the pre-scan first) and
/// `parse_expression` is private with no external callers, this guard
/// can only fire if a FUTURE internal recursion path bypasses the
/// pre-scan or the pre-scan under-counts a future grammar form — pure
/// defense-in-depth. `2×64 = 128` is still far under the 2 MiB-worker
/// pest cliff (depth 100 for bracket forms) only because the pre-scan
/// already rejects bracket/`CASE` nesting at 64; a bracket-LESS future
/// form reaching 128 ladder frames (≈ the JSON-decode `recursion_limit`
/// regime) remains well within a 2 MiB stack for the AST walk alone
/// (the walk's frames are far cheaper than pest's).
const AST_GUARD_DEPTH_LIMIT: usize = MAX_EXPRESSION_DEPTH * 2;

thread_local! {
    /// Current expression-nesting depth for the in-flight parse on
    /// this thread. Incremented on entry to `parse_expression` /
    /// `parse_where_expr` (the two — and only two — funnels into the
    /// precedence ladder) and decremented on scope exit via the RAII
    /// [`DepthGuard`]. Reset to 0 at each top-level `parse` /
    /// `parse_multi` entry so a prior parse that unwound mid-flight
    /// (e.g. a panic from a different code path) cannot leak a
    /// non-zero baseline into the next query on a pooled thread.
    ///
    /// This is thread-LOCAL (never shared across threads) and always
    /// returns to 0 between top-level parses, so the transformer
    /// functions remain functionally pure w.r.t. their AST output
    /// (the module-doc "no shared mutable state" invariant is about
    /// cross-query / cross-thread state affecting *results*; this
    /// counter affects only whether an adversarially-deep input is
    /// rejected, and is invisible to every well-formed parse).
    static EXPRESSION_DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// RAII guard that increments [`EXPRESSION_DEPTH`] on construction and
/// decrements it on drop. Drop runs on BOTH the `Ok` return path and
/// any `?`-propagated `Err` / unwind, so the counter is always
/// balanced — panic-safe by construction (the decrement cannot be
/// skipped). Construction returns `Err(ExpressionTooDeep)` when the
/// incremented depth would exceed the internal backstop limit
/// [`AST_GUARD_DEPTH_LIMIT`], so the caller bails BEFORE recursing one
/// frame deeper. This is the SECOND-layer / defense-in-depth guard; the
/// authoritative user-visible cap is enforced earlier by the pre-parse
/// scan (see [`MAX_EXPRESSION_DEPTH`] / [`check_pre_parse_nesting_depth`]).
struct DepthGuard;

impl DepthGuard {
    /// Enter one expression-nesting level. Returns `Err` (without
    /// incrementing past the limit) when the new depth would exceed the
    /// backstop [`AST_GUARD_DEPTH_LIMIT`]; otherwise increments the
    /// counter and returns a guard whose `Drop` decrements it.
    #[inline]
    fn enter() -> Result<Self, ParseError> {
        EXPRESSION_DEPTH.with(|d| {
            let next = d.get() + 1;
            if next > AST_GUARD_DEPTH_LIMIT {
                // Do NOT store `next` — we reject without descending,
                // so the counter reflects only frames we actually
                // entered. The error reports the authoritative
                // user-visible cap (`MAX_EXPRESSION_DEPTH`), not the
                // internal backstop limit — what the user must stay
                // under is the documented cap.
                return Err(ParseError::ExpressionTooDeep {
                    depth: next,
                    max: MAX_EXPRESSION_DEPTH,
                });
            }
            d.set(next);
            Ok(DepthGuard)
        })
    }
}

impl Drop for DepthGuard {
    #[inline]
    fn drop(&mut self) {
        EXPRESSION_DEPTH.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Reset the per-thread expression-depth counter to 0. Called at each
/// top-level parse entry so a prior unwound parse cannot leak a
/// non-zero baseline onto a pooled worker thread (defense-in-depth —
/// balanced `DepthGuard` drops already return it to 0 on every normal
/// path, but a panic from an unrelated code path between an increment
/// and its guard's drop is conceivable).
#[inline]
fn reset_expression_depth() {
    EXPRESSION_DEPTH.with(|d| d.set(0));
}

/// Pre-parse nesting-depth bound for the pest PEG matcher (#819).
///
/// **Why a pre-parse scan?** The pest matcher in
/// `ArcQLGrammar::parse(Rule::query, …)` recurses on the grammar's
/// `expression` / `unary_expr` rule nesting and overflows the native
/// stack BUILDING the `Pairs` tree — BEFORE any of our AST-walk code
/// (and its RAII [`DepthGuard`]) runs. On a 2 MiB Tokio worker stack
/// pest overflows at bracket-nesting depth 100 / unary-chain depth
/// ~3000 (measured; see the module budget comment). So we MUST bound
/// the depth *before* invoking pest. This O(n) byte pass does exactly
/// that, via the [`DepthScan`] running-balance accumulator.
///
/// **What it counts — a COMPLETE upper bound across EVERY recursive
/// expression-parse form** (the residual #819 R1 finding was that the
/// first cut counted only brackets/`CASE` and MISSED unary chains):
///   1. **Bracket balance** — `( [ {` (openers) vs `) ] }` (closers).
///      Covers parens, list / map literals, subscripts / slices,
///      function-call arg lists, list comprehensions, map projections,
///      and filter/reduce: every form that re-enters `parse_expression`
///      / `parse_where_expr` carries one of these literal brackets.
///   2. **`CASE … END`** — the one bracket-less re-entry into the
///      ladder (keyword-bracketed, word-boundary matched).
///   3. **Unary-prefix `-`/`+` / `NOT` operators** — `unary_expr = ("-"|"+") ~
///      unary_expr` self-recurses once per prefix operator with NO
///      bracket, and `kw_not*` folds into nested `UnaryOp::Not`, so a
///      chain (`-+-+ … 1`, `NOT NOT … x`) is real AST depth the bracket
///      balance scores as 0. Each prefix operator adds one to the
///      running depth (via [`DepthScan::push_unary`]) and is discharged
///      when its operand atom completes, so a long FLAT infix add
///      (`1+1+ … +1`) does not look like parser recursion but still
///      becomes a left-nested AST in the iterative fold.
///   4. **Flat binary operator chains** — boolean (`AND`/`OR`/`XOR`),
///      comparison, arithmetic, and string/list predicate operators are
///      grammar `*` repetitions. pest parses them iteratively, but the
///      parser folds them into left-nested `BinaryOp` / predicate ASTs,
///      so each operator adds one frame of later binder/evaluator
///      recursion. The scanner counts the current expression's operator
///      chain and discharges it at expression separators (`comma`,
///      semicolon, colon, or a closing bracket).
///
/// Unary frames persist across an enclosing
///      bracket and stack additively with bracket depth, so the running
///      total is the true native-recursion depth (`-(-(-( … )))` is
///      depth 2 per level).
///
/// Returns `Ok(())` iff the max running depth stays
/// `≤ MAX_EXPRESSION_DEPTH`.
///
/// **String + comment opacity.** Bytes inside `'…'` / `"…"` string
/// literals AND inside `--` line / `/* */` block comments are opaque:
/// neither brackets nor `CASE` nor `-`/`+` inside them count. pest
/// strips comments and treats string content as data, so counting them
/// would FALSE-REJECT a valid query (e.g. `RETURN 1 -- ((((  +++++` or
/// `RETURN '(((( CASE'`). String escapes honor backslash (`\'`, `\"`, …
/// per `escape_seq`); ArcQL has no doubled-quote escape (`''` is two
/// adjacent strings), so a bare quote always closes — matching that
/// exactly keeps the scan from UNDER-counting. Adjacent `--` opens a
/// line comment (so it is NOT two unary minuses), while spaced `- -`
/// stays two operators (both verified on-box).
///
/// **Direction of imprecision.** The unary prefix/infix split is decided
/// by an `expecting_operand` flag that is exact after brackets, commas,
/// and the arithmetic/comparison operators, and may mis-classify the
/// FIRST operator after an unrecognized keyword (`AND`/`OR`/`WHEN`/…) —
/// but only ever toward UNDER-counting a unary run by at most ONE, never
/// toward letting a stack-overflowing input through: the 2nd…Nth
/// operators of a run are still counted (a unary prefix keeps
/// `expecting_operand` true), and keyword-separated runs are independent
/// FLAT operands that do not stack. Bracket nesting (capped at CAP)
/// dominates any mixed form, so the off-by-one cannot bypass the cap.
/// The scan deliberately does NOT validate balance or grammar (pest's
/// job); unbalanced closers saturate at 0 (never go negative).
///
/// On rejection, returns `Err(ExpressionTooDeep { depth, max })` where
/// `depth` is the first running depth that exceeded the cap.
fn check_pre_parse_nesting_depth(input: &str) -> Result<(), ParseError> {
    // In-string state: `None` outside any literal; `Some(quote)` inside
    // a literal opened by `quote` (`'` or `"`).
    let mut in_string: Option<u8> = None;
    let mut escaped = false; // previous byte was an unescaped backslash, inside a string
    let mut scan = DepthScan::new();
    // `true` when the next `+`/`-` is a UNARY prefix (an operand is
    // expected here): at expression start, after an opener, a comma, a
    // colon, or another operator. `false` after an operand byte (atom
    // char, closing bracket, or closing quote) — there a `+`/`-` is the
    // INFIX add/sub operator. See the fn doc "Direction of imprecision".
    let mut expecting_operand = true;

    // Byte-oriented scan: every construct we care about (`( ) [ ] { }`,
    // quotes, backslash, the comment markers `--` / `/*` / `*/`, the
    // unary `+`/`-`, and the ASCII `CASE` / `END` keywords) is pure
    // ASCII, so a byte scan is correct even with multi-byte UTF-8
    // identifiers / string contents (non-ASCII bytes are all `>= 0x80`
    // and never collide with the ASCII bytes we match).
    let bytes = input.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if let Some(q) = in_string {
            // Inside a string literal — skip all accounting. ArcQL
            // strings escape ONLY via backslash (`escape_seq` in the
            // grammar); a bare quote always CLOSES the string (there is
            // no `''` doubled-quote escape — `''` is the empty string
            // followed by a new string). Closing on every unescaped
            // quote is the conservative choice: it never treats a
            // closed string as still-open, so brackets after a string
            // are always counted. A closed string is a completed operand.
            if escaped {
                // This byte is escaped; consume it literally.
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == q {
                in_string = None;
                expecting_operand = false;
            }
            i += 1;
            continue;
        }

        // Comments (outside strings) — pest strips them, so their
        // contents must NOT count toward depth (else `RETURN 1 -- ((((`
        // / `RETURN 1 /* +++++ */` would FALSE-REJECT). `--` (adjacent)
        // opens a line comment to EOL; `/*` opens a block comment to
        // `*/`. A comment is whitespace-like and does NOT change
        // `expecting_operand`. (This also disambiguates `--` as a
        // comment from two unary minuses; `- -` spaced stays two ops.)
        if b == b'-' && bytes.get(i + 1) == Some(&b'-') {
            i += 2;
            while i < bytes.len() && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
            i += 2;
            while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                i += 1;
            }
            i = (i + 2).min(bytes.len()); // skip the closing `*/`
            continue;
        }

        // Outside string + comment. Each opener (`( [ {` or `CASE`)
        // enters one nesting level; each closer (`) ] }` or `END`)
        // leaves one; a unary-prefix `+`/`-` enters one transient level
        // discharged when its operand completes. `CASE`/`END` are
        // ASCII-case-insensitive + word-boundary matched so `MYCASE` /
        // `CASES` / `ENDPOINT` / `SUSPEND` do not trip the counter.
        match b {
            b' ' | b'\t' | b'\r' | b'\n' => {
                // whitespace: no state change.
            }
            b'\'' | b'"' => {
                in_string = Some(b);
                escaped = false;
                expecting_operand = false; // a string literal is an operand
            }
            b'(' | b'[' | b'{' => {
                scan.open_bracket()?;
                expecting_operand = true;
            }
            b')' | b']' | b'}' => {
                scan.close_bracket();
                expecting_operand = false;
            }
            b'+' | b'-' => {
                if expecting_operand {
                    // UNARY prefix — opens one `unary_expr` recursion
                    // frame; the operand is still pending, so we stay
                    // expecting (a chain `-+-+…` keeps pushing).
                    scan.push_unary()?;
                } else {
                    // INFIX add/sub — the left operand (and any unary
                    // frames wrapping it) is complete; an operand
                    // follows. The iterative parser fold will nest one
                    // more BinaryOp on the left, so count it.
                    scan.discharge_unary();
                    scan.push_chain_operator(ChainFamily::Additive)?;
                    expecting_operand = true;
                }
            }
            b'*' | b'/' | b'%' | b'^' => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Multiplicative)?;
                expecting_operand = true;
            }
            b'<' | b'>' | b'=' => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Comparison)?;
                let is_two_byte_comparison = ((b == b'<' || b == b'>')
                    && bytes.get(i + 1) == Some(&b'='))
                    || (b == b'<' && bytes.get(i + 1) == Some(&b'>'));
                if is_two_byte_comparison {
                    i += 1;
                }
                expecting_operand = true;
            }
            b',' | b':' | b';' => {
                // Expression separators: the current operand completes
                // and the current flat operator chain ends.
                scan.discharge_unary();
                scan.discharge_chain_operators();
                expecting_operand = true;
            }
            b'A' | b'a' if matches_keyword_ci(bytes, i, b"AND") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Boolean)?;
                expecting_operand = true;
                i += 3;
                continue;
            }
            b'O' | b'o' if matches_keyword_ci(bytes, i, b"OR") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Boolean)?;
                expecting_operand = true;
                i += 2;
                continue;
            }
            b'X' | b'x' if matches_keyword_ci(bytes, i, b"XOR") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Boolean)?;
                expecting_operand = true;
                i += 3;
                continue;
            }
            // #1290 — keyword-spelled comparison-family postfix operators.
            // The grammar's `( comparison_op ~ add_expr | special_pred )*`
            // repetition admits unbounded `IN` / `IS [NOT] NULL` /
            // `STARTS WITH` / `ENDS WITH` / `CONTAINS` chains that fold
            // into the SAME left-nested spine the symbol comparisons do,
            // so they must draw from the same flat-chain budget (they
            // were uncounted before this fix — an unbounded-depth
            // bypass). Word-boundary matched like AND/OR/XOR, so
            // identifiers merely CONTAINING these words don't count; a
            // bare identifier that IS one of these words (e.g. a
            // property named `contains`) over-counts by one — benign at
            // the cap's magnitude (false-reject-only direction).
            b'I' | b'i' if matches_keyword_ci(bytes, i, b"IN") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Comparison)?;
                expecting_operand = true;
                i += 2;
                continue;
            }
            b'I' | b'i' if matches_keyword_ci(bytes, i, b"IS") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Comparison)?;
                expecting_operand = true;
                i += 2;
                continue;
            }
            b'S' | b's' if matches_keyword_ci(bytes, i, b"STARTS") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Comparison)?;
                expecting_operand = true;
                i += 6;
                continue;
            }
            b'E' | b'e' if matches_keyword_ci(bytes, i, b"ENDS") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Comparison)?;
                expecting_operand = true;
                i += 4;
                continue;
            }
            b'C' | b'c' if matches_keyword_ci(bytes, i, b"CONTAINS") => {
                scan.discharge_unary();
                scan.push_chain_operator(ChainFamily::Comparison)?;
                expecting_operand = true;
                i += 8;
                continue;
            }
            b'N' | b'n' if matches_keyword_ci(bytes, i, b"NOT") => {
                scan.push_unary()?;
                expecting_operand = true;
                i += 3;
                continue;
            }
            b'C' | b'c' if matches_keyword_ci(bytes, i, b"CASE") => {
                scan.open_bracket()?;
                expecting_operand = true; // the WHEN / branch operand follows
                i += 4; // skip the matched keyword
                continue;
            }
            b'E' | b'e' if matches_keyword_ci(bytes, i, b"END") => {
                scan.close_bracket();
                expecting_operand = false;
                i += 3; // skip the matched keyword
                continue;
            }
            _ => {
                // identifier / number / parameter / any other atom byte:
                // an operand is present (a following `+`/`-` is infix).
                expecting_operand = false;
            }
        }
        i += 1;
    }
    Ok(())
}

/// Running-balance accumulator for [`check_pre_parse_nesting_depth`].
///
/// `depth` is the open parser-recursion frames at the current scan
/// position — bracket / `CASE` frames PLUS unary-prefix frames — i.e. a
/// conservative upper bound on how deep pest will recurse at the
/// deepest atom under this position. `max_depth` is its running max; the
/// scan bails the instant it would exceed [`MAX_EXPRESSION_DEPTH`].
///
/// `chain_operators` is deliberately separate: flat binary/postfix
/// chains are not pest-recursive, but they fold into left-nested AST
/// nodes later walked by binding/evaluation. Those cheaper frames are
/// capped by [`MAX_FLAT_CHAIN_DEPTH`], not by the 64-deep parser
/// recursion cap.
///
/// Unary frames must persist across an enclosing bracket (`-(-(-( … )))`
/// is depth 2 per level) and discharge when their operand atom
/// completes, so each open bracket parks the unary count that wrapped it
/// (`saved_unary`) and restores it when the bracket closes. `saved_unary`
/// holds one entry per OPEN bracket, which is bounded by `depth ≤ CAP`
/// (the scan returns `Err` before `depth` exceeds the cap), so it never
/// holds more than CAP entries — O(CAP) memory, O(n) total work.
struct DepthScan {
    /// Total open frames (brackets + `CASE` + active unary prefixes).
    depth: usize,
    /// Running maximum of `depth`.
    max_depth: usize,
    /// Consecutive unary-prefix frames at the TOP of the frame stack
    /// (opened since the last bracket-open / discharge) — the run that
    /// discharges when the current operand completes.
    trailing_unary: usize,
    /// Per open bracket, the `trailing_unary` that wrapped it (parked on
    /// open, restored on close).
    saved_unary: Vec<usize>,
    /// Binary / postfix operators in the current flat expression chain,
    /// split by precedence family. These are not parser-recursive, but
    /// they fold into left-nested AST nodes later walked by
    /// binding/evaluation.
    chain_operators: [usize; ChainFamily::COUNT],
    /// Per open bracket, the outer expression's chain counts (parked on
    /// open, restored on close).
    saved_chain_operators: Vec<[usize; ChainFamily::COUNT]>,
    /// Running SUM of every parked `saved_chain_operators` entry — the
    /// chain operators contributed by ENCLOSING bracket levels along the
    /// current nesting path. The flat-chain cap is enforced on
    /// `parked + current` so bracket nesting cannot multiply the
    /// admitted left-spine depth (each open bracket would otherwise
    /// grant a fresh per-level budget, letting
    /// `a AND … AND ( b AND … AND ( … ))` compose bracket depth × chain
    /// cap into an AST path far deeper than the cap — the #1290 R1
    /// composition residual).
    parked_chain_operators: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ChainFamily {
    Boolean,
    Comparison,
    Additive,
    Multiplicative,
}

impl ChainFamily {
    const COUNT: usize = 4;

    const fn index(self) -> usize {
        match self {
            Self::Boolean => 0,
            Self::Comparison => 1,
            Self::Additive => 2,
            Self::Multiplicative => 3,
        }
    }
}

impl DepthScan {
    fn new() -> Self {
        Self {
            depth: 0,
            max_depth: 0,
            trailing_unary: 0,
            saved_unary: Vec::new(),
            chain_operators: [0; ChainFamily::COUNT],
            saved_chain_operators: Vec::new(),
            parked_chain_operators: 0,
        }
    }

    /// Raise `max_depth` to `depth` if exceeded and return
    /// `Err(ExpressionTooDeep)` the instant it passes the cap (so the
    /// scan bails on the FIRST over-cap frame — O(1), never recursing).
    #[inline]
    fn note_max(&mut self) -> Result<(), ParseError> {
        if self.depth > self.max_depth {
            self.max_depth = self.depth;
            if self.max_depth > MAX_EXPRESSION_DEPTH {
                return Err(ParseError::ExpressionTooDeep {
                    depth: self.max_depth,
                    max: MAX_EXPRESSION_DEPTH,
                });
            }
        }
        Ok(())
    }

    /// Raise the folded flat-chain maximum and reject once it exceeds
    /// the separate flat-chain cap. `count` already includes the parked
    /// chain operators of every enclosing bracket level (see
    /// [`Self::chain_operators_in_scope`]), so the cap bounds the
    /// DEEPEST root-to-leaf left-spine an accepted statement can fold
    /// to (within the documented per-family under-count — a looser
    /// family resets tighter counts, so a maxed chain per family can
    /// stack to a small-constant multiple of the cap; the backstop
    /// margin measurement in `tests/expression_depth_dos.rs` covers
    /// that worst case).
    #[inline]
    fn note_chain_max(&self, count: usize) -> Result<(), ParseError> {
        if count > MAX_FLAT_CHAIN_DEPTH {
            return Err(ParseError::ExpressionTooDeep {
                depth: count,
                max: MAX_FLAT_CHAIN_DEPTH,
            });
        }
        Ok(())
    }

    /// Flat chain operators along the current bracket-nesting PATH: the
    /// current expression's per-family counts plus every enclosing
    /// (parked) level's counts. This is what the flat-chain cap bounds.
    #[inline]
    fn chain_operators_in_scope(&self) -> usize {
        self.parked_chain_operators + self.chain_operators.iter().sum::<usize>()
    }

    /// Enter one bracket / `CASE` nesting level. Parks the trailing
    /// unary run (it wraps this group and must outlive it).
    #[inline]
    fn open_bracket(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        self.note_max()?;
        self.saved_unary.push(self.trailing_unary);
        self.saved_chain_operators.push(self.chain_operators);
        // The parked chain operators stay part of the in-scope total:
        // an inner bracket's chain extends the SAME root-to-leaf AST
        // path the outer chain sits on, so the flat-chain budget is
        // cumulative across bracket levels, not per-level.
        self.parked_chain_operators += self.chain_operators.iter().sum::<usize>();
        self.trailing_unary = 0;
        self.chain_operators = [0; ChainFamily::COUNT];
        Ok(())
    }

    /// Leave one bracket / `CASE` level: discharge any unary frames
    /// trailing INSIDE this group, pop the bracket frame, then restore
    /// the unary run that wrapped the group (so a following infix
    /// operator discharges it). All decrements saturate at 0 so excess
    /// closers never underflow.
    #[inline]
    fn close_bracket(&mut self) {
        self.depth = self.depth.saturating_sub(self.trailing_unary);
        self.depth = self.depth.saturating_sub(1);
        self.trailing_unary = self.saved_unary.pop().unwrap_or(0);
        self.chain_operators = self
            .saved_chain_operators
            .pop()
            .unwrap_or([0; ChainFamily::COUNT]);
        // Un-park the restored level's contribution (it is `current`
        // again, not `parked`). Saturating so excess closers on
        // unbalanced input never underflow (mirrors the depth
        // decrements above; pest rejects the input later anyway).
        self.parked_chain_operators = self
            .parked_chain_operators
            .saturating_sub(self.chain_operators.iter().sum::<usize>());
    }

    /// Enter one unary-prefix (`-`/`+`) frame.
    #[inline]
    fn push_unary(&mut self) -> Result<(), ParseError> {
        self.depth += 1;
        self.trailing_unary += 1;
        self.note_max()
    }

    /// Discharge the trailing unary run (its operand just completed).
    #[inline]
    fn discharge_unary(&mut self) {
        self.depth = self.depth.saturating_sub(self.trailing_unary);
        self.trailing_unary = 0;
    }

    /// Enter one flat binary/postfix operator frame in the current
    /// expression chain.
    #[inline]
    fn push_chain_operator(&mut self, family: ChainFamily) -> Result<(), ParseError> {
        let idx = family.index();
        for tighter in (idx + 1)..ChainFamily::COUNT {
            self.chain_operators[tighter] = 0;
        }
        self.chain_operators[idx] += 1;
        self.note_chain_max(self.chain_operators_in_scope())
    }

    /// Discharge the current expression's flat operator chain at a
    /// separator or group boundary.
    #[inline]
    fn discharge_chain_operators(&mut self) {
        self.chain_operators = [0; ChainFamily::COUNT];
    }
}

/// `true` iff `kw` (ASCII-uppercase) matches `bytes` at offset `i`
/// case-insensitively AND on a word boundary — i.e. the byte BEFORE `i`
/// and the byte AFTER the match are not identifier characters
/// (ASCII-alphanumeric or `_`). Mirrors the grammar's `kw_end` boundary
/// discipline (`!( ASCII_ALPHANUMERIC | "_" )`) so the pre-scan's
/// keyword detection agrees with how pest tokenizes `CASE` / `END`.
#[inline]
fn matches_keyword_ci(bytes: &[u8], i: usize, kw: &[u8]) -> bool {
    let end = i + kw.len();
    if end > bytes.len() {
        return false;
    }
    // Body match, ASCII-case-insensitive.
    for (off, &kb) in kw.iter().enumerate() {
        if !bytes[i + off].eq_ignore_ascii_case(&kb) {
            return false;
        }
    }
    // Left boundary: byte before `i` must not be an identifier char.
    if i > 0 && is_ident_byte(bytes[i - 1]) {
        return false;
    }
    // Right boundary: byte after the match must not be an identifier char.
    if end < bytes.len() && is_ident_byte(bytes[end]) {
        return false;
    }
    true
}

/// `true` for ASCII bytes that can appear inside an identifier
/// (alphanumeric or underscore). Matches the grammar's `kw_end`
/// boundary class.
#[inline]
fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// Parse an ArcQL source string into a typed [`Statement`] AST.
///
/// On success, returns the AST. On failure, returns the structured
/// [`ParseError`]. The parser is whitespace- and case-insensitive
/// at keywords (per `^"…"` PEG tokens) and case-sensitive at
/// identifiers (per openCypher v9 §1.3).
///
/// # Single-statement contract
///
/// `parse()` enforces single-statement input via the grammar's
/// `query = SOI ~ statement ~ ";"? ~ EOI` rule. Multi-statement input
/// (`stmt1; stmt2`) surfaces as `ParseError::Pest` because the EOI
/// fails after the first statement+optional-semicolon. Callers that
/// need multi-statement support route through [`parse_multi`] per
/// ADR-038 §5.4.1 closure (M4-83).
///
/// # Examples
///
/// ```
/// use arcgraph_query::parse;
/// let q = parse("MATCH (n) RETURN n").unwrap();
/// // …
/// # let _ = q;
/// ```
pub fn parse(input: &str) -> Result<Statement, ParseError> {
    // DoS guard, site (1) — bound the nesting depth BEFORE pest, whose
    // PEG matcher would otherwise overflow the native stack building the
    // `Pairs` tree on adversarially-deep input (#819; see the parser
    // module budget comment). This O(n) scan runs first so a ~600-byte
    // deep-nested query fails cleanly instead of SIGABRT-ing the server.
    check_pre_parse_nesting_depth(input)?;
    // Clear any leaked baseline from a prior unwound parse on this
    // (possibly pooled) thread before counting this query's nesting
    // depth (#819). Normal paths already return the counter to 0 via
    // balanced `DepthGuard` drops; this is belt-and-suspenders.
    reset_expression_depth();
    let mut top = ArcQLGrammar::parse(Rule::query, input).map_err(map_pest_err)?;
    let query_pair = top.next().ok_or_else(|| ParseError::AstConstruction {
        message: "pest parse produced no pairs".into(),
        span: None,
    })?;
    if query_pair.as_rule() != Rule::query {
        return Err(ParseError::AstConstruction {
            message: format!("expected Rule::query, got {:?}", query_pair.as_rule()),
            span: Some(span_of(&query_pair)),
        });
    }

    // `query = SOI ~ statement ~ ";"? ~ EOI`. Walk to the inner
    // `statement` pair.
    let stmt_pair = query_pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::statement)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "Rule::query missing inner Rule::statement".into(),
            span: None,
        })?;
    parse_statement(stmt_pair)
}

/// Parse an ArcQL source string into a `Vec<Statement>` admitting a
/// semicolon-separated multi-statement chain per ADR-038 §5.4.1 closure
/// (M4-83). Single-statement input is admissible (a degenerate chain
/// of length 1) so callers can route both shapes through this entry
/// point uniformly.
///
/// # Multi-statement semantics
///
/// The grammar admits the chain at the syntactic layer; semantic-layer
/// invariants (cross-statement variable scoping, shared snapshot LSN
/// per amendment-03 §TIER-1 GAP E rule 2) flow through
/// [`crate::semantic::BindingVisitor::bind_multi`] (binding) →
/// [`crate::materialize::materialize_multi`] (executor primitive) /
/// [`crate::QueryEngine::execute_multi`] (M5↔M4 surface).
///
/// # Errors
///
/// Same taxonomy as [`parse`]: [`ParseError::Pest`] for syntactic
/// rejection, [`ParseError::AstConstruction`] for tree-walk failures.
/// An empty input string produces a `Pest` error (the grammar requires
/// at least one statement); the success path always returns a Vec of
/// length ≥ 1.
///
/// # Examples
///
/// ```
/// use arcgraph_query::parse_multi;
/// let stmts = parse_multi("MATCH (n) RETURN n; MATCH (m) RETURN m").unwrap();
/// assert_eq!(stmts.len(), 2);
/// // Single-statement input is also admissible.
/// let one = parse_multi("MATCH (n) RETURN n").unwrap();
/// assert_eq!(one.len(), 1);
/// ```
pub fn parse_multi(input: &str) -> Result<Vec<Statement>, ParseError> {
    // DoS guard, site (1) — pre-parse depth bound before pest (#819;
    // see `parse`). NOTE: the pre-scan's running bracket/CASE balance
    // returns to 0 at each `;` boundary between statements (a complete
    // statement is balanced), so this whole-input scan correctly bounds
    // the DEEPEST single statement's nesting — it does NOT sum depth
    // across a long shallow `;`-chain. Only a single deeply-nested
    // expression trips the cap.
    check_pre_parse_nesting_depth(input)?;
    // See `parse` — reset the per-thread depth baseline (#819). The
    // depth cap is PER-STATEMENT: each statement's expression tree
    // re-enters `parse_expression` from a fresh 0 baseline (the
    // counter returns to 0 between statements via balanced guard
    // drops), so a long `;`-chain of shallow statements is fine; only
    // a single deeply-nested expression trips the cap.
    reset_expression_depth();
    let mut top = ArcQLGrammar::parse(Rule::multi_query, input).map_err(map_pest_err)?;
    let query_pair = top.next().ok_or_else(|| ParseError::AstConstruction {
        message: "pest parse produced no pairs".into(),
        span: None,
    })?;
    if query_pair.as_rule() != Rule::multi_query {
        return Err(ParseError::AstConstruction {
            message: format!("expected Rule::multi_query, got {:?}", query_pair.as_rule()),
            span: Some(span_of(&query_pair)),
        });
    }
    // `multi_query = SOI ~ statement ~ (";" ~ statement)* ~ ";"? ~ EOI`.
    // Walk every inner `statement` pair and parse it. The grammar
    // guarantees at least one — defense-in-depth empty check.
    let mut stmts = Vec::new();
    for child in query_pair.into_inner() {
        if child.as_rule() == Rule::statement {
            stmts.push(parse_statement(child)?);
        }
    }
    if stmts.is_empty() {
        return Err(ParseError::AstConstruction {
            message: "Rule::multi_query produced no Rule::statement children".into(),
            span: None,
        });
    }
    Ok(stmts)
}

// =====================================================================
// Pest error mapping
// =====================================================================

fn map_pest_err(e: pest::error::Error<Rule>) -> ParseError {
    let span = match &e.line_col {
        pest::error::LineColLocation::Pos((l, c)) => Span::point(*l, *c),
        pest::error::LineColLocation::Span((l1, c1), (l2, c2)) => Span {
            start_line: *l1,
            start_col: *c1,
            end_line: *l2,
            end_col: *c2,
        },
    };
    // `pest::error::Error::Display` includes a multi-line marker
    // pointing at the offending position; we keep the first line
    // (the human-readable summary) and discard the indicator art.
    let message = e.variant.message().to_string();
    ParseError::Pest { message, span }
}

fn span_of(p: &Pair<'_, Rule>) -> Span {
    let (sl, sc) = p.as_span().start_pos().line_col();
    let (el, ec) = p.as_span().end_pos().line_col();
    Span {
        start_line: sl,
        start_col: sc,
        end_line: el,
        end_col: ec,
    }
}

/// Extract the canonical identifier text from a `Rule::identifier`
/// pair, stripping surrounding backticks if the matched alternative
/// was the `backtick_ident` form (Cypher canonical escape hatch —
/// `` `MATCH` ``, `` `order by` ``, `` `n.x = 1` ``).
///
/// PR #154 reviewer Finding 1 / Fix A.1. Backtick-escape is parser-
/// internal: the `Identifier(String)` AST variant carries the
/// inner text verbatim, indistinguishable from an unescaped form.
/// The grammar guarantees a backtick-escaped identifier has at
/// least one inner byte (the `+` in `( !"`" ~ ANY )+`).
///
/// Caller MUST pass a `Pair` whose rule is `Rule::identifier`. The
/// helper handles both the `backtick_ident` and the bare-form
/// alternatives transparently.
fn identifier_text(pair: &Pair<'_, Rule>) -> String {
    debug_assert_eq!(
        pair.as_rule(),
        Rule::identifier,
        "identifier_text: expected Rule::identifier, got {:?}",
        pair.as_rule()
    );
    let raw = pair.as_str();
    // Two shapes per `grammar.pest`:
    //   1. `` `inner` `` — strip the leading and trailing backtick.
    //   2. bare `identifier_inner` — return verbatim.
    // We detect the backtick form by the literal first byte rather
    // than walking the inner pairs, because `backtick_ident` is
    // declared as a compound-atomic (`${ ... }`) — its inner
    // content is a single non-emitting `(!"`" ~ ANY)+` repetition.
    if let Some(rest) = raw.strip_prefix('`') {
        // Trailing backtick is guaranteed by the grammar.
        rest.strip_suffix('`')
            .unwrap_or(rest) // defensive — should not happen
            .to_string()
    } else {
        raw.to_string()
    }
}

/// Extract the canonical key text from a `Rule::map_key` pair (the
/// EXPRESSION-context map-literal key class, `{null: ..., NULL: ...}`).
///
/// `map_key = @{ backtick_ident | identifier_inner }` (grammar.pest) —
/// unlike `Rule::identifier`, it admits reserved-word spellings (`NULL`,
/// `MATCH`, …) WITHOUT backticks, because a map key is followed by a
/// mandatory `:` and so carries no clause-ambiguity. The backtick form
/// strips its surrounding backticks exactly as `identifier_text` does;
/// the bare form is returned verbatim. See the `map_key` rationale block
/// in `grammar.pest` (ADR-038 amendment-04 §D-X.1 scope note).
fn map_key_text(pair: &Pair<'_, Rule>) -> String {
    debug_assert_eq!(
        pair.as_rule(),
        Rule::map_key,
        "map_key_text: expected Rule::map_key, got {:?}",
        pair.as_rule()
    );
    let raw = pair.as_str();
    if let Some(rest) = raw.strip_prefix('`') {
        rest.strip_suffix('`').unwrap_or(rest).to_string()
    } else {
        raw.to_string()
    }
}

// =====================================================================
// Top-level statement
// =====================================================================

fn parse_statement(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::ddl_statement => parse_ddl_statement(inner),
        Rule::read_query => parse_read_query(inner).map(Statement::Read),
        // UNION / UNION ALL set-op query per ADR-185 (#649-A1, W28 —
        // openCypher v9 §8). Routed ABOVE `read_query` in `statement`
        // (grammar.pest); the inner `union_query` carries ≥2 tail-free
        // `union_body` arms + an optional post-union `union_tail`.
        Rule::union_query => parse_union_query(inner).map(Statement::Union),
        // EXPLAIN / PROFILE wrappers per ADR-038 §2 D-19 +
        // amendment-03 §TIER-1 GAP B (M4-91). The grammar enforces
        // that the inner is a `read_query` (NOT a full `statement`);
        // we mirror that on the AST side by carrying `ReadQuery`
        // directly rather than `Box<Statement>`.
        Rule::explain_query => parse_explain_query(inner),
        Rule::profile_query => parse_profile_query(inner),
        r => unexpected("statement", r, &inner),
    }
}

fn parse_explain_query(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = span_of(&pair);
    let inner = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::read_query)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "EXPLAIN missing read_query body".into(),
            span: Some(span),
        })?;
    let q = parse_read_query(inner)?;
    Ok(Statement::Explain(q))
}

fn parse_profile_query(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let span = span_of(&pair);
    let inner = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::read_query)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "PROFILE missing read_query body".into(),
            span: Some(span),
        })?;
    let q = parse_read_query(inner)?;
    Ok(Statement::Profile(q))
}

fn parse_read_query(pair: Pair<'_, Rule>) -> Result<ReadQuery, ParseError> {
    let mut clauses = Vec::new();
    for c in pair.into_inner() {
        if c.as_rule() == Rule::clause {
            clauses.push(parse_clause(c)?);
        }
    }
    Ok(ReadQuery { clauses })
}

// =====================================================================
// UNION / UNION ALL (ADR-185 §8 — openCypher v9 §8 Set operations)
// =====================================================================

/// Parse a `union_query` pair into a [`UnionQuery`].
///
/// Grammar shape: `union_body (kw_union kw_all? union_body)+
/// union_tail?`. The `kw_union` / `kw_all` keyword rules are
/// non-silent atomics (`@{}`), so they appear in the pair stream; we
/// walk it left-to-right, pushing each `union_body` as an arm and
/// recording one `all` flag per `kw_union` boundary (true ⇔ a `kw_all`
/// follows). The optional trailing `union_tail` (bound to the WHOLE
/// union — the RC-2 fix) lands in [`UnionQuery::tail`]. Left-assoc is
/// implicit in the flat arm vector + per-boundary flags.
fn parse_union_query(pair: Pair<'_, Rule>) -> Result<UnionQuery, ParseError> {
    let mut arms: Vec<ReadQuery> = Vec::new();
    let mut all: Vec<bool> = Vec::new();
    let mut tail = UnionTail::default();
    let mut it = pair.into_inner().peekable();
    while let Some(p) = it.next() {
        match p.as_rule() {
            Rule::union_body => arms.push(parse_union_body(p)?),
            Rule::kw_union => {
                // The boundary's `ALL` modifier is the OPTIONAL
                // `kw_all` immediately following this `kw_union`.
                let is_all = matches!(it.peek().map(|x| x.as_rule()), Some(Rule::kw_all));
                if is_all {
                    it.next(); // consume the kw_all
                }
                all.push(is_all);
            }
            Rule::union_tail => tail = parse_union_tail(p)?,
            // kw_all is consumed in the kw_union arm above; nothing
            // else can appear (WHITESPACE/COMMENT are silent).
            _ => {}
        }
    }
    Ok(UnionQuery { arms, all, tail })
}

/// Parse a single tail-free `union_body` into a [`ReadQuery`]. Because
/// `core_clause` is a SILENT grammar rule (ADR-185), the body's
/// children ARE the real clause pairs (`match_clause`, `return_clause`,
/// …) — never wrapped in a `clause` — so each dispatches through
/// [`parse_clause_inner`] directly. The standalone ORDER BY / SKIP /
/// LIMIT tail clauses are excluded by `core_clause` and thus cannot
/// appear inside an arm.
fn parse_union_body(pair: Pair<'_, Rule>) -> Result<ReadQuery, ParseError> {
    let mut clauses = Vec::new();
    for c in pair.into_inner() {
        clauses.push(parse_clause_inner(c)?);
    }
    Ok(ReadQuery { clauses })
}

/// Parse the post-union `union_tail` into a [`UnionTail`]. Reuses the
/// same `order_by` / `skip` / `limit` sub-productions the
/// `tail_*_clause`s use, so the ORDER BY / SKIP / LIMIT parse shape is
/// identical to the read-query tail — only the BINDING locus differs
/// (whole-union vs last-clause).
fn parse_union_tail(pair: Pair<'_, Rule>) -> Result<UnionTail, ParseError> {
    let mut tail = UnionTail::default();
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::order_by => tail.order_by = parse_order_by(p)?,
            Rule::skip => {
                let e = parse_expression(
                    p.into_inner()
                        .find(|q| q.as_rule() == Rule::expression)
                        .ok_or_else(|| ParseError::AstConstruction {
                            message: "union_tail SKIP missing expression".into(),
                            span: None,
                        })?,
                )?;
                tail.skip = Some(e);
            }
            Rule::limit => {
                let e = parse_expression(
                    p.into_inner()
                        .find(|q| q.as_rule() == Rule::expression)
                        .ok_or_else(|| ParseError::AstConstruction {
                            message: "union_tail LIMIT missing expression".into(),
                            span: None,
                        })?,
                )?;
                tail.limit = Some(e);
            }
            _ => {}
        }
    }
    Ok(tail)
}

fn parse_clause(pair: Pair<'_, Rule>) -> Result<Clause, ParseError> {
    let inner = first_inner(pair)?;
    parse_clause_inner(inner)
}

/// Parse a `call_clause` pair into a [`CallClause`] per ADR-192 (#623).
///
/// Grammar shape: `kw_call "{" ( union_query | read_query ) "}"`. The
/// `kw_call` keyword is a non-silent atomic (`@{}`) so it appears in the
/// pair stream; the `{` / `}` are bare string literals and do NOT. We
/// find the single body pair (a `union_query` or a bare `read_query`)
/// and wrap it as a [`Statement::Union`] / [`Statement::Read`] — the
/// only two `Statement` shapes a subquery body admits (EXPLAIN / PROFILE
/// and DDL are excluded by the grammar). Implicit import,
/// scoping, and the read-only write fence are BIND-time concerns
/// (ADR-192 D-3 / D-4 / D-9), not parse-time.
fn parse_call_clause(pair: Pair<'_, Rule>) -> Result<CallClause, ParseError> {
    let span = span_of(&pair);
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::union_query => {
                let u = parse_union_query(p)?;
                return Ok(CallClause {
                    body: Box::new(Statement::Union(u)),
                });
            }
            Rule::read_query => {
                let q = parse_read_query(p)?;
                return Ok(CallClause {
                    body: Box::new(Statement::Read(q)),
                });
            }
            // kw_call (atomic, consumed) + WHITESPACE/COMMENT (silent).
            _ => {}
        }
    }
    Err(ParseError::AstConstruction {
        message: "CALL { … } missing a read_query / union_query subquery body".into(),
        span: Some(span),
    })
}

/// **ADR-197 (#802)** — parse `CALL <proc>(args) [YIELD item [AS alias], …]`
/// into a [`CallProcedureClause`].
///
/// Grammar shape: `kw_call qualified_proc_name "(" proc_arg_list? ")"
/// (kw_yield yield_item_list)?`. The `kw_call`/`kw_yield` atomics +
/// `qualified_proc_name` appear in the pair stream; the parens are bare
/// literals (silent).
fn parse_call_procedure_clause(pair: Pair<'_, Rule>) -> Result<CallProcedureClause, ParseError> {
    let span = span_of(&pair);
    let mut name: Option<String> = None;
    let mut args: Vec<Expression> = Vec::new();
    let mut yield_items: Vec<(String, Option<String>)> = Vec::new();
    let mut where_clause: Option<Expression> = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::qualified_proc_name => name = Some(p.as_str().to_string()),
            Rule::where_sub => where_clause = Some(parse_where_sub(p)?),
            Rule::proc_arg_list => {
                for arg in p.into_inner() {
                    if arg.as_rule() == Rule::expression {
                        args.push(parse_expression(arg)?);
                    }
                }
            }
            Rule::yield_item_list => {
                for item in p.into_inner() {
                    if item.as_rule() == Rule::yield_item {
                        let mut idents = item
                            .into_inner()
                            .filter(|i| i.as_rule() == Rule::identifier);
                        let col = idents.next().map(|i| identifier_text(&i)).ok_or_else(|| {
                            ParseError::AstConstruction {
                                message: "YIELD item missing column name".into(),
                                span: Some(span.clone()),
                            }
                        })?;
                        let alias = idents.next().map(|i| identifier_text(&i));
                        yield_items.push((col, alias));
                    }
                }
            }
            // kw_call / kw_yield (atomic, consumed) + WS/COMMENT (silent).
            _ => {}
        }
    }
    let name = name.ok_or_else(|| ParseError::AstConstruction {
        message: "CALL <proc>(…) missing a qualified procedure name".into(),
        span: Some(span),
    })?;
    Ok(CallProcedureClause {
        name,
        args,
        yield_items,
        where_clause,
    })
}

/// **ADR-197 (#802)** — parse `SHOW CONSTRAINTS | INDEXES | DATABASES`
/// into a [`ShowClause`].
fn parse_show_clause(pair: Pair<'_, Rule>) -> Result<ShowClause, ParseError> {
    let span = span_of(&pair);
    let mut kind: Option<ShowKind> = None;
    let mut yield_items: Vec<(String, Option<String>)> = Vec::new();
    let mut where_clause: Option<Expression> = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            // `SHOW VECTOR INDEXES` (#830) — the multi-word kind, matched
            // via the `show_vector_indexes` sub-rule (`kw_vector ~
            // kw_indexes`), distinct from the atomic single-word
            // `show_kind`.
            Rule::show_vector_indexes => kind = Some(ShowKind::VectorIndexes),
            Rule::show_kind => {
                kind = Some(match p.as_str().to_ascii_uppercase().as_str() {
                    "CONSTRAINTS" => ShowKind::Constraints,
                    "INDEXES" => ShowKind::Indexes,
                    "DATABASES" => ShowKind::Databases,
                    other => {
                        return Err(ParseError::AstConstruction {
                            message: format!("unsupported SHOW kind: {other}"),
                            span: Some(span.clone()),
                        });
                    }
                });
            }
            // `YIELD item [AS alias], …` (#830) — mirrors
            // `parse_call_procedure_clause`'s yield parsing.
            Rule::yield_item_list => {
                for item in p.into_inner() {
                    if item.as_rule() == Rule::yield_item {
                        let mut idents = item
                            .into_inner()
                            .filter(|i| i.as_rule() == Rule::identifier);
                        let col = idents.next().map(|i| identifier_text(&i)).ok_or_else(|| {
                            ParseError::AstConstruction {
                                message: "YIELD item missing column name".into(),
                                span: Some(span.clone()),
                            }
                        })?;
                        let alias = idents.next().map(|i| identifier_text(&i));
                        yield_items.push((col, alias));
                    }
                }
            }
            // `WHERE <pred>` filtering the YIELD'd rows (#830).
            Rule::where_sub => where_clause = Some(parse_where_sub(p)?),
            // kw_show / kw_yield (atomic, consumed) + WS/COMMENT (silent).
            _ => {}
        }
    }
    let kind = kind.ok_or_else(|| ParseError::AstConstruction {
        message: "SHOW missing a kind (CONSTRAINTS / INDEXES / DATABASES / VECTOR INDEXES)".into(),
        span: Some(span),
    })?;
    Ok(ShowClause {
        kind,
        yield_items,
        where_clause,
    })
}

/// Dispatch a single already-unwrapped clause pair (the inner of a
/// `clause`, OR a direct child of `union_body` — `core_clause` is a
/// SILENT grammar rule per ADR-185, so a `union_body`'s children ARE
/// the real `match_clause` / `return_clause` / … pairs, never wrapped).
/// Factored out of [`parse_clause`] so [`parse_union_body`] reuses the
/// exact same dispatch without re-implementing it (the union arm clause
/// set is `core_clause`, i.e. `clause` MINUS the three tail arms; those
/// tail arms cannot appear here for a union body but the match handles
/// them harmlessly if they ever do).
fn parse_clause_inner(inner: Pair<'_, Rule>) -> Result<Clause, ParseError> {
    match inner.as_rule() {
        Rule::match_clause => parse_match_clause(inner).map(Clause::Match),
        Rule::optional_match_clause => parse_match_clause(inner).map(Clause::OptionalMatch),
        Rule::create_clause => parse_create_clause(inner).map(Clause::Create),
        Rule::delete_clause => parse_delete_clause(inner).map(Clause::Delete),
        Rule::set_clause => parse_set_clause(inner).map(Clause::Set),
        Rule::remove_clause => parse_remove_clause(inner).map(Clause::Remove),
        Rule::merge_clause => parse_merge_clause(inner).map(Clause::Merge),
        Rule::with_clause => parse_with_clause(inner).map(Clause::With),
        Rule::unwind_clause => parse_unwind_clause(inner).map(Clause::Unwind),
        Rule::call_clause => parse_call_clause(inner).map(Clause::Call),
        Rule::call_procedure_clause => {
            parse_call_procedure_clause(inner).map(Clause::CallProcedure)
        }
        Rule::show_clause => parse_show_clause(inner).map(Clause::Show),
        Rule::rank_by_clause => parse_rank_by_clause(inner).map(Clause::RankBy),
        Rule::with_fusion_clause => parse_with_fusion(inner).map(Clause::WithFusion),
        Rule::return_clause => parse_return_clause(inner).map(Clause::Return),
        Rule::tail_order_by_clause => {
            let inner_ob = first_inner(inner)?;
            Ok(Clause::TailOrderBy(parse_order_by(inner_ob)?))
        }
        Rule::tail_skip_clause => {
            let skip = first_inner(inner)?;
            let e = parse_expression(
                skip.into_inner()
                    .find(|q| q.as_rule() == Rule::expression)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "tail SKIP missing expression".into(),
                        span: None,
                    })?,
            )?;
            Ok(Clause::TailSkip(e))
        }
        Rule::tail_limit_clause => {
            let lim = first_inner(inner)?;
            let e = parse_expression(
                lim.into_inner()
                    .find(|q| q.as_rule() == Rule::expression)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "tail LIMIT missing expression".into(),
                        span: None,
                    })?,
            )?;
            Ok(Clause::TailLimit(e))
        }
        r => unexpected("clause", r, &inner),
    }
}

// =====================================================================
// MATCH
// =====================================================================

fn parse_match_clause(pair: Pair<'_, Rule>) -> Result<MatchClause, ParseError> {
    let span = span_of(&pair);
    let mut body = None;
    let mut where_clause = None;

    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::match_body => body = Some(parse_match_body(child)?),
            Rule::where_sub => where_clause = Some(parse_where_sub(child)?),
            _ => {} // keywords / punctuation — ignored.
        }
    }

    Ok(MatchClause {
        body: body.ok_or_else(|| ParseError::AstConstruction {
            message: "MATCH without body".into(),
            span: Some(span),
        })?,
        where_clause,
    })
}

fn parse_match_body(pair: Pair<'_, Rule>) -> Result<MatchBody, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::named_path_match => parse_named_path_match(inner).map(MatchBody::NamedPath),
        Rule::pattern_list => parse_pattern_list(inner).map(MatchBody::Patterns),
        r => unexpected("match_body", r, &inner),
    }
}

fn parse_named_path_match(pair: Pair<'_, Rule>) -> Result<NamedPath, ParseError> {
    let span = span_of(&pair);
    let mut iter = pair.into_inner();
    let var_pair = expect_rule(&mut iter, Rule::identifier, "named_path_match var", &span)?;
    let var = identifier_text(&var_pair);
    let next = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "named_path_match missing rhs".into(),
        span: Some(span.clone()),
    })?;
    let kind = match next.as_rule() {
        // ADR-194 D-1/D-3 — both the `SHORTEST_PATH` macro and the
        // canonical camelCase `shortestPath` collapse to this single rule
        // (one algorithm, two spellings) → one `ShortestPath` variant.
        Rule::shortest_path_pattern => {
            let p = parse_path_fn_pattern(next)?;
            NamedPathKind::ShortestPath(p)
        }
        // ADR-194 D-2 — canonical `allShortestPaths(...)` → the net-new
        // all-equal-min-length variant.
        Rule::all_shortest_path_pattern => {
            let p = parse_path_fn_pattern(next)?;
            NamedPathKind::AllShortestPath(p)
        }
        Rule::path_pattern => NamedPathKind::Plain(parse_path_pattern(next)?),
        r => {
            return Err(ParseError::AstConstruction {
                message: format!("unexpected named_path_match rhs: Rule::{r:?}"),
                span: Some(span_of(&next)),
            });
        }
    };
    Ok(NamedPath { var, kind })
}

/// Extract the inner `path_pattern` from a path-function pattern
/// (`shortest_path_pattern` or `all_shortest_path_pattern`). Both rules
/// are `<keyword> ~ "(" ~ path_pattern ~ ")"`, so the keyword token is
/// skipped and the inner `path_pattern` parsed uniformly (ADR-194 D-1).
fn parse_path_fn_pattern(pair: Pair<'_, Rule>) -> Result<PathPattern, ParseError> {
    let inner = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::path_pattern)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "shortestPath/allShortestPaths missing inner path_pattern".into(),
            span: None,
        })?;
    parse_path_pattern(inner)
}

fn parse_pattern_list(pair: Pair<'_, Rule>) -> Result<Vec<PathPattern>, ParseError> {
    let mut out = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::path_pattern {
            out.push(parse_path_pattern(p)?);
        }
    }
    Ok(out)
}

fn parse_path_pattern(pair: Pair<'_, Rule>) -> Result<PathPattern, ParseError> {
    let mut iter = pair.into_inner();
    let head_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "path_pattern missing head node".into(),
        span: None,
    })?;
    let head = parse_node_pattern(head_pair)?;
    let mut tail = Vec::new();
    while let Some(rel_pair) = iter.next() {
        let rel = parse_rel_pattern(rel_pair)?;
        let node_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
            message: "path_pattern: relationship not followed by node".into(),
            span: None,
        })?;
        let node = parse_node_pattern(node_pair)?;
        tail.push((rel, node));
    }
    Ok(PathPattern { head, tail })
}

fn parse_node_pattern(pair: Pair<'_, Rule>) -> Result<NodePattern, ParseError> {
    let mut var = None;
    let mut labels = Vec::new();
    let mut properties = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::node_var => {
                // `node_var = { identifier }`; descend to the inner
                // identifier so backtick-escape stripping applies.
                let id_pair = p
                    .into_inner()
                    .find(|q| q.as_rule() == Rule::identifier)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "node_var missing identifier".into(),
                        span: None,
                    })?;
                var = Some(identifier_text(&id_pair));
            }
            Rule::label_spec => {
                for li in p.into_inner() {
                    if li.as_rule() == Rule::identifier {
                        labels.push(identifier_text(&li));
                    }
                }
            }
            Rule::property_map => properties = Some(parse_property_map(p)?),
            _ => {}
        }
    }
    Ok(NodePattern {
        var,
        labels,
        properties,
    })
}

fn parse_rel_pattern(pair: Pair<'_, Rule>) -> Result<RelPattern, ParseError> {
    let raw = pair.as_str();
    let span = span_of(&pair);
    // The grammar makes `rel_pattern` a single atomic token — see
    // the comment in `grammar.pest`. We re-tokenize the matched
    // text in Rust to extract var / rel_types / direction / length
    // / properties.
    parse_rel_pattern_text(raw, &span)
}

/// Re-tokenize a `rel_pattern` raw text slice. Splits the slice
/// into an optional `[..]` body (rel_detail) and an optional
/// trailing `{N,M}` quantifier suffix; the leading and trailing
/// arrow shape determines direction.
fn parse_rel_pattern_text(raw: &str, span: &Span) -> Result<RelPattern, ParseError> {
    let raw = raw.trim();
    let direction = if raw.starts_with("<-") {
        RelDirection::RightToLeft
    } else if raw.contains("->") {
        RelDirection::LeftToRight
    } else {
        RelDirection::Undirected
    };

    // Strip leading `<-`/`-` and trailing `->`/`-` to get the
    // optional `[…]` body and any `{N,M}` quantifier suffix.
    let mut rest = raw;
    if let Some(s) = rest.strip_prefix("<-") {
        rest = s;
    } else if let Some(s) = rest.strip_prefix('-') {
        rest = s;
    }

    // Now split the trailing arrow / dash. We first check for a
    // `{N,M}` suffix; that always lives at the very end.
    let mut quant: Option<&str> = None;
    if let Some(open) = rest.rfind('{') {
        if rest[open..].ends_with('}') && rest[open..].contains(',') {
            quant = Some(&rest[open..]);
            rest = &rest[..open];
        }
    }
    rest = rest.trim_end();

    // Strip trailing arrow/dash.
    if let Some(s) = rest.strip_suffix("->") {
        rest = s;
    } else if let Some(s) = rest.strip_suffix('-') {
        rest = s;
    }
    rest = rest.trim();

    // What remains is an optional `[…]` body.
    let mut var: Option<String> = None;
    let mut rel_types: Vec<String> = Vec::new();
    let mut length: Option<LengthRange> = None;
    let mut properties: Option<PropertyMap> = None;
    if !rest.is_empty() {
        let inner = rest
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| ParseError::AstConstruction {
                message: format!("rel_pattern body must be `[…]`, got `{rest}`"),
                span: Some(span.clone()),
            })?;
        let (v, t, l, p) = parse_rel_detail_text(inner, span)?;
        var = v;
        rel_types = t;
        length = l;
        properties = p;
    }

    if let Some(q) = quant {
        // GQL `{N,M}` length-range — parser-reserved at v1.0.
        let trimmed = q
            .strip_prefix('{')
            .and_then(|s| s.strip_suffix('}'))
            .ok_or_else(|| ParseError::AstConstruction {
                message: format!("malformed quantified suffix `{q}`"),
                span: Some(span.clone()),
            })?;
        let parts: Vec<&str> = trimmed.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(ParseError::AstConstruction {
                message: format!("GQL quantifier needs comma: `{q}`"),
                span: Some(span.clone()),
            });
        }
        let min: u32 = parts[0]
            .trim()
            .parse()
            .map_err(|e: std::num::ParseIntError| ParseError::AstConstruction {
                message: format!("GQL quantifier min: {e}"),
                span: Some(span.clone()),
            })?;
        let max_str = parts[1].trim();
        let max = if max_str.is_empty() {
            None
        } else {
            Some(max_str.parse().map_err(|e: std::num::ParseIntError| {
                ParseError::AstConstruction {
                    message: format!("GQL quantifier max: {e}"),
                    span: Some(span.clone()),
                }
            })?)
        };
        length = Some(LengthRange::Quantified { min, max });
    }

    Ok(RelPattern {
        var,
        rel_types,
        direction,
        length,
        properties,
    })
}

/// 4-tuple shape returned by [`parse_rel_detail_text`].
type RelDetailParts = (
    Option<String>,
    Vec<String>,
    Option<LengthRange>,
    Option<PropertyMap>,
);

/// Re-tokenize the body of a `[…]` rel_detail — split into var /
/// type-list / cypher-length-range / property-map. Whitespace-
/// tolerant. The grammar's overall `rel_pattern` rule is atomic,
/// so the inner text is not pre-tokenized by pest.
fn parse_rel_detail_text(body: &str, span: &Span) -> Result<RelDetailParts, ParseError> {
    use pest::Parser;
    // We re-parse the body via two helper rules: `rel_detail_body`
    // captures the structured shape. Since `rel_pattern` is atomic,
    // we couldn't rely on pest to recurse here; we hand-scan the
    // body instead.

    let body = body.trim();
    let mut cursor = body;
    let mut var: Option<String> = None;
    let mut rel_types: Vec<String> = Vec::new();
    let mut length: Option<LengthRange> = None;
    let mut properties: Option<PropertyMap> = None;

    // 1) Variable name.
    if let Some(c) = cursor.chars().next() {
        if c.is_alphabetic() || c == '_' {
            let end = cursor
                .char_indices()
                .find(|(_, ch)| !(ch.is_alphanumeric() || *ch == '_'))
                .map(|(i, _)| i)
                .unwrap_or(cursor.len());
            var = Some(cursor[..end].to_string());
            cursor = cursor[end..].trim_start();
        }
    }

    // 2) Type list `:T1|T2|…`.
    if let Some(rest) = cursor.strip_prefix(':') {
        cursor = rest.trim_start();
        loop {
            let end = cursor
                .char_indices()
                .find(|(_, ch)| !(ch.is_alphanumeric() || *ch == '_'))
                .map(|(i, _)| i)
                .unwrap_or(cursor.len());
            if end == 0 {
                return Err(ParseError::AstConstruction {
                    message: "rel_pattern type missing identifier".into(),
                    span: Some(span.clone()),
                });
            }
            rel_types.push(cursor[..end].to_string());
            cursor = cursor[end..].trim_start();
            if let Some(rest) = cursor.strip_prefix('|') {
                cursor = rest.trim_start();
                continue;
            }
            break;
        }
    }

    // 3) openCypher length range `*N..M` or `*`.
    if cursor.starts_with('*') {
        let end_idx = cursor[1..]
            .char_indices()
            .find(|(_, ch)| !(ch.is_ascii_digit() || *ch == '.'))
            .map(|(i, _)| i + 1)
            .unwrap_or(cursor.len());
        let chunk = &cursor[..end_idx];
        length = Some(parse_cypher_length_range_text(chunk, span)?);
        cursor = cursor[end_idx..].trim_start();
    }

    // 4) Property map `{ k: v, … }`. We delegate back to pest by
    // building a `property_map` parse on the substring.
    if cursor.starts_with('{') {
        let map_end = find_matching_brace(cursor).ok_or_else(|| ParseError::AstConstruction {
            message: "rel_pattern property map missing closing brace".into(),
            span: Some(span.clone()),
        })?;
        let map_text = &cursor[..map_end + 1];
        // Parse via pest's `property_map` rule.
        let mut pairs = ArcQLGrammar::parse(Rule::property_map, map_text).map_err(map_pest_err)?;
        let pm_pair = pairs.next().ok_or_else(|| ParseError::AstConstruction {
            message: "rel_pattern property_map produced no pairs".into(),
            span: Some(span.clone()),
        })?;
        properties = Some(parse_property_map(pm_pair)?);
        cursor = cursor[map_end + 1..].trim_start();
    }

    if !cursor.is_empty() {
        return Err(ParseError::AstConstruction {
            message: format!("trailing text in rel_pattern body: `{cursor}`"),
            span: Some(span.clone()),
        });
    }

    Ok((var, rel_types, length, properties))
}

fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_cypher_length_range_text(raw: &str, span: &Span) -> Result<LengthRange, ParseError> {
    if raw == "*" {
        return Ok(LengthRange::Unbounded);
    }
    let rest = raw
        .strip_prefix('*')
        .ok_or_else(|| ParseError::AstConstruction {
            message: format!("cypher length-range missing leading `*`: `{raw}`"),
            span: Some(span.clone()),
        })?;
    let parts: Vec<&str> = rest.splitn(2, "..").collect();
    let min: u32 =
        parts[0]
            .parse()
            .map_err(|e: std::num::ParseIntError| ParseError::AstConstruction {
                message: format!("cypher length range min: {e}"),
                span: Some(span.clone()),
            })?;
    let max = if parts.len() == 1 {
        // #939: openCypher `*N` is shorthand for `*N..N`, i.e. exactly N hops.
        Some(min)
    } else if parts[1].is_empty() {
        None
    } else {
        Some(parts[1].parse().map_err(|e: std::num::ParseIntError| {
            ParseError::AstConstruction {
                message: format!("cypher length range max: {e}"),
                span: Some(span.clone()),
            }
        })?)
    };
    Ok(LengthRange::Cypher { min, max })
}

fn parse_property_map(pair: Pair<'_, Rule>) -> Result<PropertyMap, ParseError> {
    let mut entries = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::prop_entry {
            let mut iter = p.into_inner();
            let key = iter.next().ok_or_else(|| ParseError::AstConstruction {
                message: "prop_entry missing key".into(),
                span: None,
            })?;
            let val = iter.next().ok_or_else(|| ParseError::AstConstruction {
                message: "prop_entry missing value".into(),
                span: None,
            })?;
            // `prop_entry = ${ identifier ~ ":" ~ expression }` — the
            // key is always a `Rule::identifier`, so the
            // backtick-aware extractor applies.
            entries.push((identifier_text(&key), parse_expression(val)?));
        }
    }
    Ok(PropertyMap { entries })
}

fn parse_where_sub(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let inner = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::where_expr)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "WHERE missing expression".into(),
            span: None,
        })?;
    parse_where_expr(inner)
}

// =====================================================================
// CREATE (ADR-147 W26-θ Phase 1)
// =====================================================================

fn parse_create_clause(pair: Pair<'_, Rule>) -> Result<CreateClause, ParseError> {
    let mut items = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::create_item {
            items.push(parse_create_item(child)?);
        }
    }
    Ok(CreateClause { items })
}

fn parse_create_item(pair: Pair<'_, Rule>) -> Result<CreateItem, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::create_path => parse_create_path(inner).map(CreateItem::Path),
        Rule::create_node => parse_create_node(inner).map(CreateItem::Node),
        r => unexpected("create_item", r, &inner),
    }
}

// ADR-148 W26-θ Phase 2: `(source)-[rel:LABEL {props}]->(target)`.
// The grammar's `create_path` rule sequences `create_node ~ create_rel
// ~ create_node`. We extract each in source order.
fn parse_create_path(pair: Pair<'_, Rule>) -> Result<CreatePathSpec, ParseError> {
    let span = span_of(&pair);
    let mut source: Option<CreateNodeSpec> = None;
    let mut rel: Option<CreateRelSpec> = None;
    let mut target: Option<CreateNodeSpec> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::create_node => {
                if source.is_none() {
                    source = Some(parse_create_node(child)?);
                } else if target.is_none() {
                    target = Some(parse_create_node(child)?);
                } else {
                    return Err(ParseError::AstConstruction {
                        message: "create_path: extra create_node beyond source + target".into(),
                        span: Some(span.clone()),
                    });
                }
            }
            Rule::create_rel => {
                rel = Some(parse_create_rel(child)?);
            }
            _ => {}
        }
    }
    Ok(CreatePathSpec {
        source: source.ok_or_else(|| ParseError::AstConstruction {
            message: "create_path missing source node".into(),
            span: Some(span.clone()),
        })?,
        rel: rel.ok_or_else(|| ParseError::AstConstruction {
            message: "create_path missing relationship".into(),
            span: Some(span.clone()),
        })?,
        target: target.ok_or_else(|| ParseError::AstConstruction {
            message: "create_path missing target node".into(),
            span: Some(span),
        })?,
    })
}

// `create_rel = ${ ("<" ~ "-" ~ create_rel_detail ~ "-") | ("-" ~
// create_rel_detail ~ "->") }`. Direction is determined by the raw
// text's leading characters (`<-` → RightToLeft; otherwise
// LeftToRight). The grammar's atomic-shape wrapping means the inner
// pair is a `create_rel_detail`; we descend.
fn parse_create_rel(pair: Pair<'_, Rule>) -> Result<CreateRelSpec, ParseError> {
    let raw = pair.as_str();
    let span = span_of(&pair);
    let direction = if raw.trim_start().starts_with("<-") {
        CreateRelDirection::RightToLeft
    } else {
        CreateRelDirection::LeftToRight
    };
    let detail = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::create_rel_detail)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "create_rel missing create_rel_detail".into(),
            span: Some(span.clone()),
        })?;
    let (var, label, properties) = parse_create_rel_detail(detail, &span)?;
    Ok(CreateRelSpec {
        var,
        label,
        properties,
        direction,
    })
}

// `create_rel_detail = ${ "[" ~ create_rel_var? ~ create_rel_label ~
// property_map? ~ "]" }`. The label is mandatory per ADR-148 §D-1;
// the parser surfaces a clean error when the grammar admits a malformed
// detail (defense-in-depth — the grammar already rejects the no-label
// shape).
fn parse_create_rel_detail(
    pair: Pair<'_, Rule>,
    parent_span: &Span,
) -> Result<(Option<String>, String, Option<PropertyMap>), ParseError> {
    let mut var: Option<String> = None;
    let mut label: Option<String> = None;
    let mut properties: Option<PropertyMap> = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::create_rel_var => {
                let id_pair = p
                    .into_inner()
                    .find(|q| q.as_rule() == Rule::identifier)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "create_rel_var missing identifier".into(),
                        span: Some(parent_span.clone()),
                    })?;
                var = Some(identifier_text(&id_pair));
            }
            Rule::create_rel_label => {
                let id_pair = p
                    .into_inner()
                    .find(|q| q.as_rule() == Rule::identifier)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "create_rel_label missing identifier".into(),
                        span: Some(parent_span.clone()),
                    })?;
                label = Some(identifier_text(&id_pair));
            }
            Rule::property_map => properties = Some(parse_property_map(p)?),
            _ => {}
        }
    }
    Ok((
        var,
        label.ok_or_else(|| ParseError::AstConstruction {
            message: "create_rel_detail missing mandatory label (Phase 2 per ADR-148 §D-1)".into(),
            span: Some(parent_span.clone()),
        })?,
        properties,
    ))
}

fn parse_create_node(pair: Pair<'_, Rule>) -> Result<CreateNodeSpec, ParseError> {
    let mut var = None;
    let mut label = None;
    let mut properties = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::create_node_var => {
                // `create_node_var = { identifier }`; descend to the
                // inner identifier so backtick-escape stripping applies.
                let id_pair = p
                    .into_inner()
                    .find(|q| q.as_rule() == Rule::identifier)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "create_node_var missing identifier".into(),
                        span: None,
                    })?;
                var = Some(identifier_text(&id_pair));
            }
            Rule::create_label => {
                // `create_label = ${ ":" ~ ws_opt ~ identifier }`;
                // single label at Phase 1 (multi-label forward-pinned
                // to v1.1 per ADR-147).
                let id_pair = p
                    .into_inner()
                    .find(|q| q.as_rule() == Rule::identifier)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "create_label missing identifier".into(),
                        span: None,
                    })?;
                label = Some(identifier_text(&id_pair));
            }
            Rule::property_map => properties = Some(parse_property_map(p)?),
            _ => {}
        }
    }
    Ok(CreateNodeSpec {
        var,
        label,
        properties,
    })
}

// =====================================================================
// DELETE (ADR-149 W26-θ Phase 3)
// =====================================================================

// `delete_clause = { detach? ~ kw_delete ~ delete_item ~ ("," ~
// delete_item)* }`. The grammar admits an optional `DETACH` prefix
// followed by the `DELETE` keyword and one-or-more
// comma-separated identifier arguments. We thread the `detach`
// flag verbatim from the grammar's optional `detach` production.
fn parse_delete_clause(pair: Pair<'_, Rule>) -> Result<DeleteClause, ParseError> {
    let mut detach = false;
    let mut items = Vec::new();
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::detach => detach = true,
            Rule::delete_item => items.push(parse_delete_item(child)?),
            _ => {} // kw_delete + punctuation — ignored.
        }
    }
    if items.is_empty() {
        // Defense-in-depth: the grammar requires ≥1 `delete_item` so
        // this branch should be unreachable in practice. Surface a
        // clean error if a future grammar edit ever loosens the
        // requirement.
        return Err(ParseError::AstConstruction {
            message: "delete_clause missing items (grammar guarantees ≥1)".into(),
            span: None,
        });
    }
    Ok(DeleteClause { items, detach })
}

// `delete_item = { identifier }`. We descend to the inner identifier
// so backtick-escape stripping applies (parallel to `create_node_var`
// + `create_rel_var` discipline).
fn parse_delete_item(pair: Pair<'_, Rule>) -> Result<DeleteItem, ParseError> {
    let id_pair = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::identifier)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "delete_item missing identifier".into(),
            span: None,
        })?;
    Ok(DeleteItem {
        var: identifier_text(&id_pair),
    })
}

// =====================================================================
// SET (ADR-150 W26-θ Phase 4)
// =====================================================================

// `set_clause = { kw_set ~ set_item ~ ("," ~ set_item)* }`. Threads
// the kw_set + punctuation tokens through; collects `set_item`s into
// the AST `SetClause::items` vec in source order.
fn parse_set_clause(pair: Pair<'_, Rule>) -> Result<SetClause, ParseError> {
    let mut items = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::set_item {
            items.push(parse_set_item(child)?);
        }
    }
    if items.is_empty() {
        return Err(ParseError::AstConstruction {
            message: "set_clause missing items (grammar guarantees ≥1)".into(),
            span: None,
        });
    }
    Ok(SetClause { items })
}

// `set_item = { set_property_assign | set_property_merge |
// set_property_replace | set_label_add }`. Each alternative carries
// its target identifier as its FIRST child; the SetItem's `var` is
// that identifier's text, and the `mutation` discriminates on the
// alternative.
fn parse_set_item(pair: Pair<'_, Rule>) -> Result<SetItem, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::set_property_assign => parse_set_property_assign(inner),
        Rule::set_property_merge => parse_set_property_merge(inner),
        Rule::set_property_replace => parse_set_property_replace(inner),
        Rule::set_label_add => parse_set_label_add(inner),
        r => unexpected("set_item", r, &inner),
    }
}

// `set_property_assign = { identifier ~ "." ~ identifier ~ "=" ~
// expression }`. Children: identifier (var), identifier (prop name),
// expression (rhs). Punctuation tokens (`.`, `=`) are silent at the
// pest level.
fn parse_set_property_assign(pair: Pair<'_, Rule>) -> Result<SetItem, ParseError> {
    let mut var: Option<String> = None;
    let mut name: Option<String> = None;
    let mut value: Option<Expression> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::identifier => {
                if var.is_none() {
                    var = Some(identifier_text(&child));
                } else if name.is_none() {
                    name = Some(identifier_text(&child));
                }
            }
            Rule::expression => {
                value = Some(parse_expression(child)?);
            }
            _ => {}
        }
    }
    let var = var.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_assign missing target identifier".into(),
        span: None,
    })?;
    let name = name.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_assign missing property name".into(),
        span: None,
    })?;
    let value = value.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_assign missing value expression".into(),
        span: None,
    })?;
    Ok(SetItem {
        var,
        mutation: SetMutation::PropertyAssign { name, value },
    })
}

// `set_property_merge = { identifier ~ "+=" ~ property_map }`.
fn parse_set_property_merge(pair: Pair<'_, Rule>) -> Result<SetItem, ParseError> {
    let mut var: Option<String> = None;
    let mut map: Option<PropertyMap> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::identifier if var.is_none() => {
                var = Some(identifier_text(&child));
            }
            Rule::property_map => {
                map = Some(parse_property_map(child)?);
            }
            _ => {}
        }
    }
    let var = var.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_merge missing target identifier".into(),
        span: None,
    })?;
    let map = map.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_merge missing property map".into(),
        span: None,
    })?;
    Ok(SetItem {
        var,
        mutation: SetMutation::PropertyMerge(map),
    })
}

// `set_property_replace = { identifier ~ "=" ~ property_map }`.
fn parse_set_property_replace(pair: Pair<'_, Rule>) -> Result<SetItem, ParseError> {
    let mut var: Option<String> = None;
    let mut map: Option<PropertyMap> = None;
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::identifier if var.is_none() => {
                var = Some(identifier_text(&child));
            }
            Rule::property_map => {
                map = Some(parse_property_map(child)?);
            }
            _ => {}
        }
    }
    let var = var.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_replace missing target identifier".into(),
        span: None,
    })?;
    let map = map.ok_or_else(|| ParseError::AstConstruction {
        message: "set_property_replace missing property map".into(),
        span: None,
    })?;
    Ok(SetItem {
        var,
        mutation: SetMutation::PropertyReplace(map),
    })
}

// `set_label_add = { identifier ~ (":" ~ identifier)+ }`. The first
// identifier is the target variable; each subsequent identifier is a
// label to add.
fn parse_set_label_add(pair: Pair<'_, Rule>) -> Result<SetItem, ParseError> {
    let mut idents: Vec<String> = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::identifier {
            idents.push(identifier_text(&child));
        }
    }
    if idents.len() < 2 {
        return Err(ParseError::AstConstruction {
            message: "set_label_add requires ≥1 label after target identifier".into(),
            span: None,
        });
    }
    let var = idents.remove(0);
    Ok(SetItem {
        var,
        mutation: SetMutation::LabelAdd(idents),
    })
}

// =====================================================================
// REMOVE (ADR-150 W26-θ Phase 4)
// =====================================================================

// `remove_clause = { kw_remove ~ remove_item ~ ("," ~ remove_item)* }`.
fn parse_remove_clause(pair: Pair<'_, Rule>) -> Result<RemoveClause, ParseError> {
    let mut items = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::remove_item {
            items.push(parse_remove_item(child)?);
        }
    }
    if items.is_empty() {
        return Err(ParseError::AstConstruction {
            message: "remove_clause missing items (grammar guarantees ≥1)".into(),
            span: None,
        });
    }
    Ok(RemoveClause { items })
}

// `remove_item = { remove_property | remove_label }`.
fn parse_remove_item(pair: Pair<'_, Rule>) -> Result<RemoveItem, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::remove_property => parse_remove_property(inner),
        Rule::remove_label => parse_remove_label(inner),
        r => unexpected("remove_item", r, &inner),
    }
}

// `remove_property = { identifier ~ "." ~ identifier }`.
fn parse_remove_property(pair: Pair<'_, Rule>) -> Result<RemoveItem, ParseError> {
    let mut idents: Vec<String> = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::identifier {
            idents.push(identifier_text(&child));
        }
    }
    if idents.len() != 2 {
        return Err(ParseError::AstConstruction {
            message: "remove_property requires exactly two identifiers (var.prop)".into(),
            span: None,
        });
    }
    let mut it = idents.into_iter();
    let var = it.next().expect("checked len above");
    let name = it.next().expect("checked len above");
    Ok(RemoveItem {
        var,
        mutation: RemoveMutation::Property(name),
    })
}

// `remove_label = { identifier ~ (":" ~ identifier)+ }`.
fn parse_remove_label(pair: Pair<'_, Rule>) -> Result<RemoveItem, ParseError> {
    let mut idents: Vec<String> = Vec::new();
    for child in pair.into_inner() {
        if child.as_rule() == Rule::identifier {
            idents.push(identifier_text(&child));
        }
    }
    if idents.len() < 2 {
        return Err(ParseError::AstConstruction {
            message: "remove_label requires ≥1 label after target identifier".into(),
            span: None,
        });
    }
    let var = idents.remove(0);
    Ok(RemoveItem {
        var,
        mutation: RemoveMutation::LabelRemove(idents),
    })
}

// =====================================================================
// MERGE (ADR-151 W26-θ Phase 5)
// =====================================================================

// `merge_clause = { kw_merge ~ merge_pattern ~ merge_action* }`. The
// merge_pattern reuses Phase 1 / Phase 2 create_node / create_path
// shapes verbatim; the optional merge_action* lift the `ON CREATE SET`
// / `ON MATCH SET` action bodies into vecs of SetItem.
fn parse_merge_clause(pair: Pair<'_, Rule>) -> Result<MergeClause, ParseError> {
    let mut pattern: Option<MergePattern> = None;
    let mut on_create: Vec<SetItem> = Vec::new();
    let mut on_match: Vec<SetItem> = Vec::new();
    for child in pair.into_inner() {
        match child.as_rule() {
            Rule::merge_pattern => pattern = Some(parse_merge_pattern(child)?),
            Rule::merge_action => {
                let inner = first_inner(child)?;
                match inner.as_rule() {
                    Rule::merge_on_create => {
                        on_create.extend(parse_merge_action_items(inner)?);
                    }
                    Rule::merge_on_match => {
                        on_match.extend(parse_merge_action_items(inner)?);
                    }
                    r => return unexpected("merge_action", r, &inner),
                }
            }
            _ => {} // kw_merge — ignored.
        }
    }
    let pattern = pattern.ok_or_else(|| ParseError::AstConstruction {
        message: "merge_clause missing merge_pattern (grammar guarantees ≥1)".into(),
        span: None,
    })?;
    Ok(MergeClause {
        pattern,
        on_create,
        on_match,
    })
}

// `merge_pattern = { create_path | create_node }` — reuses Phase 1 +
// Phase 2 source-text shapes verbatim.
fn parse_merge_pattern(pair: Pair<'_, Rule>) -> Result<MergePattern, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::create_path => parse_create_path(inner).map(MergePattern::Path),
        Rule::create_node => parse_create_node(inner).map(MergePattern::Node),
        r => unexpected("merge_pattern", r, &inner),
    }
}

// Extract the `SetItem` vec from a `merge_on_create` /
// `merge_on_match` pair — both wrap a `set_clause` body. The
// `set_clause` body's leading `kw_set` token is part of the action's
// grammar (`ON CREATE SET …`); we delegate to the existing
// `parse_set_clause` parser and extract the resulting items vec.
fn parse_merge_action_items(pair: Pair<'_, Rule>) -> Result<Vec<SetItem>, ParseError> {
    // `merge_on_create = { kw_on ~ ^"CREATE" ~ kw_end ~ set_clause }` /
    // `merge_on_match  = { kw_on ~ ^"MATCH"  ~ kw_end ~ set_clause }`.
    // Find the inner set_clause and reuse the existing parser.
    let set_clause = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::set_clause)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "merge_action missing set_clause body".into(),
            span: None,
        })?;
    let sc = parse_set_clause(set_clause)?;
    Ok(sc.items)
}

// =====================================================================
// WITH / UNWIND
// =====================================================================

fn parse_with_clause(pair: Pair<'_, Rule>) -> Result<WithClause, ParseError> {
    // Conservative DISTINCT detector — IDENTICAL idiom to
    // `parse_return_clause`: the keyword, if present, is the token
    // immediately after `WITH`. A bounded prefix scan (not a full
    // uppercase search) so a projection like `WITH distinct_col AS c`
    // (the `kw_end` word-boundary in the grammar already prevents the
    // keyword from matching an identifier prefix) cannot false-positive.
    let raw = pair.as_str();
    let upper_prefix = raw
        .get(..raw.len().min(64))
        .map(str::to_ascii_uppercase)
        .unwrap_or_default();
    let distinct = upper_prefix
        .split_whitespace() // #926: collapse runs of whitespace before DISTINCT.
        .nth(1)
        .map(|s| s.trim().eq_ignore_ascii_case("DISTINCT"))
        .unwrap_or(false);
    let mut items = Vec::new();
    let mut where_clause = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::projection_list => items = parse_projection_list(p)?,
            Rule::where_sub => where_clause = Some(parse_where_sub(p)?),
            _ => {}
        }
    }
    Ok(WithClause {
        distinct,
        items,
        where_clause,
    })
}

fn parse_unwind_clause(pair: Pair<'_, Rule>) -> Result<UnwindClause, ParseError> {
    let span = span_of(&pair);
    let mut iter = pair.into_inner();
    let expr_pair = expect_rule(&mut iter, Rule::expression, "UNWIND expr", &span)?;
    let var_pair = expect_rule(&mut iter, Rule::identifier, "UNWIND var", &span)?;
    Ok(UnwindClause {
        expr: parse_expression(expr_pair)?,
        var: identifier_text(&var_pair),
    })
}

fn parse_projection_list(pair: Pair<'_, Rule>) -> Result<Vec<ProjectionItem>, ParseError> {
    let mut out = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::projection_item {
            out.push(parse_projection_item(p)?);
        }
    }
    Ok(out)
}

fn parse_projection_item(pair: Pair<'_, Rule>) -> Result<ProjectionItem, ParseError> {
    let mut kind = None;
    let mut alias = None;
    // #353 — capture the verbatim source text of an expression
    // projection BEFORE consuming the pair into `parse_expression`. This
    // is the implicit result-column name openCypher/Neo4j use for an
    // un-aliased expression (`RETURN n.name` → column `"n.name"`;
    // `RETURN count(*)` → `"count(*)"`). `Pair::as_str()` returns the
    // exact slice of the source query the `expression` rule matched (no
    // re-rendering — byte-for-byte what the user wrote). Whitespace is
    // normalized below so `n.name` and `n . name` both yield `"n.name"`
    // (Neo4j collapses insignificant whitespace in implicit column
    // names). `None` for the wildcard (no single expression).
    let mut source_text = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::wildcard => kind = Some(ProjectionKind::Wildcard),
            Rule::expression => {
                source_text = Some(normalize_projection_source(p.as_str()));
                kind = Some(ProjectionKind::Expr(parse_expression(p)?));
            }
            Rule::identifier => alias = Some(identifier_text(&p)),
            _ => {}
        }
    }
    let kind = kind.ok_or_else(|| ParseError::AstConstruction {
        message: "projection_item without expression or `*`".into(),
        span: None,
    })?;
    // Source text is meaningful only for an expression projection; a
    // wildcard never carries one (defensive — the loop already leaves it
    // `None` for `*`).
    if matches!(kind, ProjectionKind::Wildcard) {
        source_text = None;
    }
    Ok(ProjectionItem {
        kind,
        alias,
        source_text,
    })
}

/// #353 — normalize the captured source text of an un-aliased
/// projection expression into the implicit column name Neo4j surfaces.
///
/// Neo4j collapses insignificant whitespace in the implicit column name
/// derived from an un-aliased expression: `RETURN n . name` and
/// `RETURN n.name` both expose the column `"n.name"`; `RETURN a.x  +  1`
/// exposes `"a.x + 1"`. We mirror that by trimming the slice and
/// collapsing internal runs of ASCII whitespace (including embedded
/// newlines from a multi-line query) to a single space.
///
/// This is a DISPLAY-NAME normalization only — it never affects parsing
/// or evaluation (the already-parsed `Expression` carries the real
/// semantics). String-literal contents inside an expression are not
/// specially preserved here because the v1.0-α column-name surface is a
/// best-effort implicit name; an explicit `AS alias` (which takes
/// precedence) is the path for callers that need an exact column label.
fn normalize_projection_source(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_was_space = false;
    for ch in raw.trim().chars() {
        if ch.is_ascii_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            out.push(ch);
            last_was_space = false;
        }
    }
    out
}

// =====================================================================
// RANK BY / WITH FUSION =
// =====================================================================

fn parse_rank_by_clause(pair: Pair<'_, Rule>) -> Result<RankByClause, ParseError> {
    let mut ranker = None;
    let mut score_alias = None;
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::ranker => ranker = Some(inner),
            Rule::identifier => score_alias = Some(inner.as_str().to_string()),
            _ => {}
        }
    }
    let inner = ranker.ok_or_else(|| ParseError::AstConstruction {
        message: "RANK BY missing ranker".into(),
        span: None,
    })?;
    let ranker_inner = first_inner(inner)?;
    let r = match ranker_inner.as_rule() {
        Rule::hybrid_ranker => Ranker::Hybrid(parse_hybrid_ranker(ranker_inner)?),
        r => return unexpected("ranker", r, &ranker_inner),
    };
    Ok(RankByClause {
        ranker: r,
        score_alias,
    })
}

fn parse_hybrid_ranker(pair: Pair<'_, Rule>) -> Result<Vec<RankArg>, ParseError> {
    let mut args = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::rank_arg {
            args.push(parse_rank_arg(p)?);
        }
    }
    Ok(args)
}

fn parse_rank_arg(pair: Pair<'_, Rule>) -> Result<RankArg, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::vector_rank_arg => parse_vector_rank_arg(inner),
        Rule::text_rank_arg => parse_text_rank_arg(inner),
        r => unexpected("rank_arg", r, &inner),
    }
}

fn parse_vector_rank_arg(pair: Pair<'_, Rule>) -> Result<RankArg, ParseError> {
    let span = span_of(&pair);
    let mut iter = pair.into_inner();
    let field_pair = expect_rule(&mut iter, Rule::field_ref, "VECTOR field", &span)?;
    let query_pair = expect_rule(&mut iter, Rule::expression, "VECTOR query", &span)?;
    let k = parse_optional_k(&mut iter)?;
    Ok(RankArg::Vector {
        field: parse_field_ref(field_pair)?,
        query: parse_expression(query_pair)?,
        k,
    })
}

fn parse_text_rank_arg(pair: Pair<'_, Rule>) -> Result<RankArg, ParseError> {
    let span = span_of(&pair);
    let mut iter = pair.into_inner();
    let field_pair = expect_rule(&mut iter, Rule::field_ref, "TEXT field", &span)?;
    let query_pair = expect_rule(&mut iter, Rule::expression, "TEXT query", &span)?;
    let k = parse_optional_k(&mut iter)?;
    Ok(RankArg::Text {
        field: parse_field_ref(field_pair)?,
        query: parse_expression(query_pair)?,
        k,
    })
}

fn parse_optional_k(iter: &mut Pairs<'_, Rule>) -> Result<Option<i64>, ParseError> {
    for p in iter.by_ref() {
        if p.as_rule() == Rule::k_assign {
            let lit = p
                .into_inner()
                .find(|q| q.as_rule() == Rule::int_literal)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "K = … missing integer literal".into(),
                    span: None,
                })?;
            let v: i64 = lit.as_str().parse().map_err(|e: std::num::ParseIntError| {
                ParseError::AstConstruction {
                    message: format!("K = …: {e}"),
                    span: Some(span_of(&lit)),
                }
            })?;
            return Ok(Some(v));
        }
    }
    Ok(None)
}

fn parse_with_fusion(pair: Pair<'_, Rule>) -> Result<WithFusionClause, ParseError> {
    let inner = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::fusion_expr)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "WITH FUSION = missing fusion_expr".into(),
            span: None,
        })?;
    let fusion = parse_fusion_expr(inner)?;
    Ok(WithFusionClause { fusion })
}

fn parse_fusion_expr(pair: Pair<'_, Rule>) -> Result<Fusion, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::rrf_fusion => parse_rrf_fusion(inner),
        r => unexpected("fusion_expr", r, &inner),
    }
}

fn parse_rrf_fusion(pair: Pair<'_, Rule>) -> Result<Fusion, ParseError> {
    let span = span_of(&pair);
    let mut k: Option<i64> = None;
    for p in pair.into_inner() {
        if p.as_rule() == Rule::rrf_arg {
            let raw = p.as_str();
            let trimmed = raw.trim_start();
            if trimmed.starts_with("k") || trimmed.starts_with("K") {
                let lit = p
                    .into_inner()
                    .find(|q| q.as_rule() == Rule::int_literal)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "RRF k = … missing integer".into(),
                        span: Some(span.clone()),
                    })?;
                k = Some(lit.as_str().parse().map_err(|e: std::num::ParseIntError| {
                    ParseError::AstConstruction {
                        message: format!("RRF k: {e}"),
                        span: Some(span_of(&lit)),
                    }
                })?);
            }
        }
    }
    let k = k.ok_or_else(|| ParseError::AstConstruction {
        message: "RRF requires k = …".into(),
        span: Some(span),
    })?;
    Ok(Fusion::Rrf { k })
}

// =====================================================================
// RETURN
// =====================================================================

fn parse_return_clause(pair: Pair<'_, Rule>) -> Result<ReturnClause, ParseError> {
    let raw = pair.as_str();
    // Conservative DISTINCT detector: the keyword (if present)
    // appears between `RETURN` and the projection list. Use a
    // bounded prefix scan rather than a full uppercase-search so
    // queries like `RETURN distinct_score AS s` (where `distinct`
    // appears as part of an identifier — though the keyword
    // exclusion guards against it) cannot false-positive.
    let upper_prefix = raw
        .get(..raw.len().min(64))
        .map(str::to_ascii_uppercase)
        .unwrap_or_default();
    let distinct = upper_prefix
        .split_whitespace() // #926: collapse runs of whitespace before DISTINCT.
        .nth(1)
        .map(|s| s.trim().eq_ignore_ascii_case("DISTINCT"))
        .unwrap_or(false);
    let mut items = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::projection_list {
            items = parse_projection_list(p)?;
        }
    }
    Ok(ReturnClause {
        distinct,
        items,
        // ORDER BY / SKIP / LIMIT now arrive as separate
        // `Clause::TailOrderBy` / `Clause::TailSkip` /
        // `Clause::TailLimit` clauses. M4-02 (semantic analyzer)
        // is the layer that folds them back into ReturnClause.
        order_by: Vec::new(),
        skip: None,
        limit: None,
    })
}

fn parse_order_by(pair: Pair<'_, Rule>) -> Result<Vec<OrderItem>, ParseError> {
    let mut out = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::order_item {
            let raw = p.as_str().to_uppercase();
            let direction = if raw.ends_with("DESC") {
                OrderDirection::Desc
            } else if raw.ends_with("ASC") {
                OrderDirection::Asc
            } else {
                OrderDirection::Default
            };
            let expr_pair = p
                .into_inner()
                .find(|q| q.as_rule() == Rule::expression)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "order_item missing expression".into(),
                    span: None,
                })?;
            out.push(OrderItem {
                expr: parse_expression(expr_pair)?,
                direction,
            });
        }
    }
    Ok(out)
}

// =====================================================================
// Expressions
// =====================================================================

fn parse_where_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    // Expression-nesting-depth guard (#819). `parse_where_expr` and
    // `parse_expression` are the two — and only two — funnels into the
    // precedence ladder (`parse_or_expr` has no other callers), and
    // EVERY nested sub-expression (parens, list element, CASE branch,
    // map value, subscript / slice index) re-enters through one of
    // them. So bumping the depth here + in `parse_expression` bounds
    // the recursion at exactly one increment per nesting level. The
    // guard is held for the whole descent of this level and dropped
    // (decrementing) on return — including the `?`-propagated error
    // path below. Held in a binding (NOT `let _ =`, which would drop
    // it immediately).
    let _depth_guard = DepthGuard::enter()?;
    let inner = first_inner(pair)?;
    parse_or_expr(inner)
}

fn parse_expression(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    // See `parse_where_expr` for the depth-guard rationale (#819).
    let _depth_guard = DepthGuard::enter()?;
    let inner = first_inner(pair)?;
    parse_or_expr(inner)
}

// `or_expr = xor_expr (kw_or xor_expr)*`. With the kw_* boundary
// rules now atomic-non-silent, `into_inner()` yields:
// `[xor_expr, kw_or, xor_expr, kw_or, xor_expr, ...]`. We filter
// out the kw_or markers via `is_kw` and fold the remainder.
// Left-associative. The inner level is `parse_xor_expr` (#621 —
// XOR binds tighter than OR; the precedence ladder is
// `OR → XOR → AND`). This one function handles BOTH the
// `where_or_expr` and `expr_or_expr` grammar rules — it operates
// structurally on `into_inner()`, blind to which ladder produced
// the pair.
fn parse_or_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut acc: Option<Expression> = None;
    for p in pair.into_inner().filter(|p| !is_kw(p.as_rule())) {
        let rhs = parse_xor_expr(p)?;
        acc = Some(match acc {
            None => rhs,
            Some(lhs) => Expression::BinaryOp {
                op: BinOp::Or,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
        });
    }
    acc.ok_or_else(|| ParseError::AstConstruction {
        message: "or_expr empty".into(),
        span: None,
    })
}

// `xor_expr = and_expr (kw_xor and_expr)*` (#621). Mirrors the
// OR/AND fold EXACTLY (left-associative, kw-marker filtered) — only
// the operator (`BinOp::Xor`) and the inner level (`parse_and_expr`)
// differ. Like `parse_or_expr`/`parse_and_expr`, this single
// function handles BOTH the `where_xor_expr` and `expr_xor_expr`
// grammar rules (structural on `into_inner()`).
fn parse_xor_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut acc: Option<Expression> = None;
    for p in pair.into_inner().filter(|p| !is_kw(p.as_rule())) {
        let rhs = parse_and_expr(p)?;
        acc = Some(match acc {
            None => rhs,
            Some(lhs) => Expression::BinaryOp {
                op: BinOp::Xor,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
        });
    }
    acc.ok_or_else(|| ParseError::AstConstruction {
        message: "xor_expr empty".into(),
        span: None,
    })
}

fn parse_and_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut acc: Option<Expression> = None;
    for p in pair.into_inner().filter(|p| !is_kw(p.as_rule())) {
        let rhs = parse_not_expr(p)?;
        acc = Some(match acc {
            None => rhs,
            Some(lhs) => Expression::BinaryOp {
                op: BinOp::And,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            },
        });
    }
    acc.ok_or_else(|| ParseError::AstConstruction {
        message: "and_expr empty".into(),
        span: None,
    })
}

fn parse_not_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    // `not_expr = kw_not* ~ comparison_expr`. The rule fires from
    // both the `where_*` and `expr_*` precedence ladders (which
    // have identical shape but admit different `special_pred`
    // surfaces — see grammar.pest comment). The grammar admits a RUN
    // of `kw_not` (double-negation `NOT NOT x`; #1050 / TCK Boolean4
    // [2]), so we COUNT the `kw_not` occurrences and fold the inner
    // comparison in that many nested `UnaryOp::Not` layers. This
    // builds a genuine `Not(Not(x))` AST (NOT a parity-bool collapse):
    // each layer is independently type-checked (binding.rs recurses)
    // and 3VL-evaluated (eval.rs `apply_unop` Not arm recurses —
    // `Not(Not(true))` = `Not(false)` = true; null propagates), so the
    // identity for even counts and negation for odd counts both fall
    // out naturally with no eval/binding change.
    let inners: Vec<_> = pair.into_inner().collect();
    let not_count = inners
        .iter()
        .filter(|p| matches!(p.as_rule(), Rule::kw_not))
        .count();
    let cmp_pair = inners
        .into_iter()
        .find(|p| !is_kw(p.as_rule()))
        .ok_or_else(|| ParseError::AstConstruction {
            message: "not_expr missing comparison_expr".into(),
            span: None,
        })?;
    let mut inner = parse_comparison_expr(cmp_pair)?;
    for _ in 0..not_count {
        inner = Expression::UnaryOp {
            op: UnaryOp::Not,
            operand: Box::new(inner),
        };
    }
    Ok(inner)
}

fn parse_comparison_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    // comparison_expr = add_expr ( comparison_op add_expr | special_pred )*
    // Pest emits a flat stream to preserve the #819 expression-depth stack
    // margin, but openCypher gives postfix predicates tighter precedence than
    // binary comparison. Consume predicates immediately after a comparison RHS
    // before building the binary expression, so `a IS NULL = b IS NULL` builds
    // `(a IS NULL) = (b IS NULL)`.
    let mut iter = pair.into_inner();
    let first = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "comparison_expr empty".into(),
        span: None,
    })?;
    let mut acc = parse_add_expr(first)?;
    let mut pending = None;
    while let Some(p) = pending.take().or_else(|| iter.next()) {
        match p.as_rule() {
            Rule::comparison_op => {
                let op = match p.as_str() {
                    "=" => BinOp::Eq,
                    "<>" => BinOp::Neq,
                    "<" => BinOp::Lt,
                    "<=" => BinOp::Le,
                    ">" => BinOp::Gt,
                    ">=" => BinOp::Ge,
                    other => {
                        return Err(ParseError::AstConstruction {
                            message: format!("unknown comparison_op `{other}`"),
                            span: Some(span_of(&p)),
                        });
                    }
                };
                let rhs_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
                    message: "comparison_op without rhs".into(),
                    span: None,
                })?;
                let mut rhs = parse_add_expr(rhs_pair)?;
                for next in iter.by_ref() {
                    match next.as_rule() {
                        Rule::special_pred | Rule::expr_special_pred => {
                            rhs = apply_special_pred(rhs, next)?;
                        }
                        _ => {
                            pending = Some(next);
                            break;
                        }
                    }
                }
                acc = Expression::BinaryOp {
                    op,
                    lhs: Box::new(acc),
                    rhs: Box::new(rhs),
                };
            }
            // A predicate seen outside a comparison_op RHS belongs to the
            // current accumulated operand. With the AST-side operand grouping,
            // this can occur only before the first comparison operator.
            Rule::special_pred | Rule::expr_special_pred => {
                acc = apply_special_pred(acc, p)?;
            }
            r => {
                return Err(ParseError::AstConstruction {
                    message: format!("unexpected pair in comparison_expr: {r:?}"),
                    span: Some(span_of(&p)),
                });
            }
        }
    }
    Ok(acc)
}

fn apply_special_pred(lhs: Expression, pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::near_op => {
            // near_op = kw_near ~ add_expr ~ ( kw_vector_index ~ identifier )?
            let mut iter = inner.into_inner().filter(|p| !is_kw(p.as_rule()));
            let target_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
                message: "NEAR without target".into(),
                span: None,
            })?;
            let target = parse_add_expr(target_pair)?;
            let vector_index = iter
                .next()
                .filter(|p| p.as_rule() == Rule::identifier)
                .map(|p| identifier_text(&p));
            Ok(Expression::Near {
                lhs: Box::new(lhs),
                target: Box::new(target),
                vector_index,
            })
        }
        Rule::text_match_op => {
            let target_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::add_expr)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "MATCH (operator) without rhs".into(),
                    span: None,
                })?;
            Ok(Expression::TextMatch {
                lhs: Box::new(lhs),
                query: Box::new(parse_add_expr(target_pair)?),
            })
        }
        Rule::community_membership_op => {
            let community_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::expression)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "IN COMMUNITY missing expression".into(),
                    span: None,
                })?;
            Ok(Expression::InCommunity {
                node: Box::new(lhs),
                community: Box::new(parse_expression(community_pair)?),
            })
        }
        Rule::in_op => {
            // openCypher v9 §3.3.5 — RHS broadened to a full `add_expr`
            // (was the restricted `list_or_param`) so a subscript / slice
            // RHS parses (`3 IN list[0]`, `3 IN [1,2,3][0..1]`). The
            // type-checker enforces the list-typed-RHS contract via
            // `check_list_operand` (rejects `1 IN true` — List5 [42]).
            let rhs_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::add_expr)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "IN missing rhs expression".into(),
                    span: None,
                })?;
            Ok(Expression::In {
                lhs: Box::new(lhs),
                rhs: Box::new(parse_add_expr(rhs_pair)?),
            })
        }
        Rule::is_null_pred => {
            let raw = inner.as_str().to_uppercase();
            let negated = raw.contains("NOT");
            Ok(Expression::IsNull {
                lhs: Box::new(lhs),
                negated,
            })
        }
        // openCypher v9 §3.3.6 string predicates → `BinaryOp` (#773). All
        // three carry RHS = `add_expr` (the `starts_with_op` / `ends_with_op`
        // rules also hold leading `kw_starts`/`kw_with` etc. pairs, which the
        // `find(add_expr)` skips). The `_` arm can only be `contains_op` — the
        // outer match guards exactly these three rules — so no panic/`unreachable!`.
        Rule::starts_with_op | Rule::ends_with_op | Rule::contains_op => {
            let op = match inner.as_rule() {
                Rule::starts_with_op => BinOp::StartsWith,
                Rule::ends_with_op => BinOp::EndsWith,
                _ => BinOp::Contains,
            };
            let rhs_pair = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::add_expr)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "string predicate missing rhs expression".into(),
                    span: None,
                })?;
            Ok(Expression::BinaryOp {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(parse_add_expr(rhs_pair)?),
            })
        }
        r => unexpected("special_pred", r, &inner),
    }
}

fn parse_add_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut iter = pair.into_inner();
    let mut acc = parse_mul_expr(iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "add_expr empty".into(),
        span: None,
    })?)?;
    while let Some(op_pair) = iter.next() {
        let op = match op_pair.as_str() {
            "+" => BinOp::Add,
            "-" => BinOp::Sub,
            other => {
                return Err(ParseError::AstConstruction {
                    message: format!("unknown add_op `{other}`"),
                    span: Some(span_of(&op_pair)),
                });
            }
        };
        let rhs = parse_mul_expr(iter.next().ok_or_else(|| ParseError::AstConstruction {
            message: "add_op without rhs".into(),
            span: None,
        })?)?;
        acc = Expression::BinaryOp {
            op,
            lhs: Box::new(acc),
            rhs: Box::new(rhs),
        };
    }
    Ok(acc)
}

fn parse_mul_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut iter = pair.into_inner();
    let first = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "mul_expr empty".into(),
        span: None,
    })?;
    let mut operands = vec![MulOperand {
        expr: parse_unary_expr(first)?,
    }];
    let mut ops = Vec::new();
    while let Some(op_pair) = iter.next() {
        let op = op_pair.as_str().to_string();
        if !matches!(op.as_str(), "*" | "/" | "%" | "^") {
            return Err(ParseError::AstConstruction {
                message: format!("unknown mul_op `{op}`"),
                span: Some(span_of(&op_pair)),
            });
        }
        let rhs_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
            message: "mul_op without rhs".into(),
            span: None,
        })?;
        ops.push(op);
        operands.push(MulOperand {
            expr: parse_unary_expr(rhs_pair)?,
        });
    }
    fold_mul_expr(operands, ops)
}

struct MulOperand {
    expr: Expression,
}

fn fold_mul_expr(
    mut operands: Vec<MulOperand>,
    mut ops: Vec<String>,
) -> Result<Expression, ParseError> {
    let mut i = 0;
    while i < ops.len() {
        if ops[i] != "^" {
            i += 1;
            continue;
        }
        let lhs = operands.remove(i).expr;
        let rhs = operands.remove(i).expr;
        let expr = Expression::BinaryOp {
            op: BinOp::Pow,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        };
        operands.insert(i, MulOperand { expr });
        ops.remove(i);
    }

    let mut acc = operands.remove(0).expr;
    for (op, rhs) in ops.into_iter().zip(operands) {
        let op = match op.as_str() {
            "*" => BinOp::Mul,
            "/" => BinOp::Div,
            "%" => BinOp::Mod,
            other => {
                return Err(ParseError::AstConstruction {
                    message: format!("unknown mul_op `{other}`"),
                    span: None,
                });
            }
        };
        acc = Expression::BinaryOp {
            op,
            lhs: Box::new(acc),
            rhs: Box::new(rhs.expr),
        };
    }
    Ok(acc)
}

fn parse_unary_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let raw = pair.as_str();
    let trimmed = raw.trim_start();
    if trimmed.starts_with('-') || trimmed.starts_with('+') {
        // Expression-nesting-depth guard (#819 R1 follow-up). UNLIKE the
        // bracket forms (which funnel through `parse_expression` /
        // `parse_where_expr`), a unary chain `-+-+ … 1` self-recurses
        // HERE — `unary_expr = ("-"|"+") ~ unary_expr` — so this arm is
        // the AST-side site for family (B). The pre-parse scan
        // ([`check_pre_parse_nesting_depth`]) is the load-bearing guard
        // (pest overflows first), but counting each prefix frame here too
        // makes the AST walk self-bounding: defense-in-depth if the
        // pre-scan ever under-counts a future grammar form. Guarded only
        // in the prefix arm (the no-prefix pass-through below adds no
        // frame), so it does NOT double-count bracket nesting — the
        // at-cap paren depth still parses. Held across the recursive
        // descent + dropped on every return path (incl. the i64::MIN
        // fold) via the RAII binding.
        let _depth_guard = DepthGuard::enter()?;
        let op = if trimmed.starts_with('-') {
            UnaryOp::Neg
        } else {
            UnaryOp::Pos
        };
        // Find the inner unary_expr pair
        let inner_pair = pair
            .into_inner()
            .find(|p| p.as_rule() == Rule::unary_expr)
            .ok_or_else(|| ParseError::AstConstruction {
                message: "unary missing operand".into(),
                span: None,
            })?;
        // i64::MIN constant-fold (#618 GA Lane C). A unary `-` directly
        // applied to a bare integer literal whose POSITIVE magnitude is
        // exactly 2^63 (`-9223372036854775808` / `-0x8000000000000000` /
        // `-0o1000000000000000000000`) cannot be built as `Integer(2^63)`
        // then negated — 2^63 overflows a positive i64 at `parse_int_lit`
        // BEFORE the negation runs. Fold the sign into the literal so the
        // value is `Integer(i64::MIN)` directly. ONLY the overflow boundary
        // folds; every other `-<expr>` (including in-range `-5`) keeps the
        // canonical `UnaryOp{Neg, ..}` shape, so no existing AST is
        // perturbed.
        if op == UnaryOp::Neg {
            if let Some(lit_str) = bare_int_literal_str(&inner_pair) {
                if let Some(folded) = fold_min_boundary_int(&lit_str) {
                    return Ok(Expression::Literal(Literal::Integer(folded)));
                }
            }
        }
        let operand = parse_unary_expr(inner_pair)?;
        return Ok(Expression::UnaryOp {
            op,
            operand: Box::new(operand),
        });
    }
    // No unary prefix; descend to atom. Exponentiation is recognized in the
    // existing multiplicative token stream and folded by `parse_mul_expr`.
    let atom_pair = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::atom)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "unary_expr missing atom".into(),
            span: None,
        })?;
    parse_atom(atom_pair)
}

fn parse_atom(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut iter = pair.into_inner();
    let primary_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "atom missing primary".into(),
        span: None,
    })?;
    let mut base = parse_primary_atom(primary_pair)?;
    // openCypher v9 §3.4 — postfix accessors apply left-to-right.
    // Consecutive PROPERTY accessors batch into one
    // `PropertyAccess { path: [..] }` (the canonical multi-segment shape
    // the binder expects); a subscript / slice flushes the pending batch
    // first, then wraps the accumulated base.
    let mut path: Vec<String> = Vec::new();
    for accessor in iter {
        if accessor.as_rule() != Rule::accessor {
            continue;
        }
        let inner = first_inner(accessor)?;
        match inner.as_rule() {
            Rule::property_accessor => {
                let id = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::identifier)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "property accessor missing identifier".into(),
                        span: None,
                    })?;
                path.push(identifier_text(&id));
            }
            Rule::index_accessor => {
                base = flush_property_path(base, &mut path);
                let idx_pair = inner
                    .into_inner()
                    .find(|p| p.as_rule() == Rule::expression)
                    .ok_or_else(|| ParseError::AstConstruction {
                        message: "index accessor missing expression".into(),
                        span: None,
                    })?;
                base = Expression::Subscript {
                    base: Box::new(base),
                    index: Box::new(parse_expression(idx_pair)?),
                };
            }
            Rule::slice_accessor => {
                base = flush_property_path(base, &mut path);
                // `slice_lo` / `slice_hi` are distinct named rules so the
                // present bound is unambiguous in the open forms.
                let mut start = None;
                let mut end = None;
                for sp in inner.into_inner() {
                    match sp.as_rule() {
                        Rule::slice_lo => {
                            start = Some(Box::new(parse_expression(first_inner(sp)?)?))
                        }
                        Rule::slice_hi => end = Some(Box::new(parse_expression(first_inner(sp)?)?)),
                        _ => {}
                    }
                }
                base = Expression::Slice {
                    base: Box::new(base),
                    start,
                    end,
                };
            }
            r => {
                return Err(ParseError::AstConstruction {
                    message: format!("unexpected accessor inner: {r:?}"),
                    span: None,
                });
            }
        }
    }
    Ok(flush_property_path(base, &mut path))
}

/// Wrap `base` in a `PropertyAccess` if a property path has accumulated,
/// then clear the path. A no-op when the path is empty (so a pure
/// subscript / slice chain does not synthesize an empty `PropertyAccess`).
fn flush_property_path(base: Expression, path: &mut Vec<String>) -> Expression {
    if path.is_empty() {
        base
    } else {
        Expression::PropertyAccess {
            base: Box::new(base),
            path: std::mem::take(path),
        }
    }
}

fn parse_primary_atom(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::parameter => Ok(parse_parameter(inner)),
        // openCypher v9 §3.6 (#621) — CASE expression, matched EARLY in the
        // grammar's ordered choice (right after `parameter`, before
        // `function_call` / `identifier`).
        Rule::case_expr => parse_case_expr(inner),
        // ADR-188 — list-predicate special forms, matched BEFORE
        // `function_call` in the grammar's ordered choice (see
        // `primary_atom` in grammar.pest).
        Rule::filter_expr => parse_filter_expr(inner),
        Rule::reduce_expr => parse_reduce_expr(inner),
        // ADR-188 (#620 list-half) — list comprehension, matched BEFORE
        // `literal` (which contains `list_literal`) in the grammar's
        // ordered choice (see `primary_atom` in grammar.pest).
        Rule::list_comprehension => parse_list_comprehension(inner),
        // ADR-191 D-6 (#620 map-half) — map projection `n{.k, alias: e, .*}`,
        // matched BEFORE `function_call` (both open with an identifier;
        // commit on `{` vs `(`).
        Rule::map_projection => parse_map_projection(inner),
        Rule::function_call => parse_function_call(inner),
        Rule::literal => Ok(Expression::Literal(parse_literal(inner)?)),
        Rule::expression | Rule::where_expr => parse_where_expr(inner),
        Rule::identifier => Ok(Expression::Identifier(identifier_text(&inner))),
        r => unexpected("primary_atom", r, &inner),
    }
}

/// **openCypher v9 §3.6** (#621) — build an [`Expression::Case`] from a
/// `case_expr` parse pair. The grammar is
/// `kw_case ~ ( !kw_when ~ expression )? ~ ( kw_when ~ expression ~ kw_then
/// ~ expression )+ ~ ( kw_else ~ expression )? ~ kw_case_end`, so the inner
/// pairs (in source order) interleave the atomic keyword markers with the
/// operand `expression`s:
/// `[ kw_case, expression? (test), (kw_when, expression, kw_then,
/// expression)+, (kw_else, expression)?, kw_case_end ]`.
///
/// We classify each `expression` by the most-recent keyword marker (a small
/// position state-machine): an `expression` seen BEFORE any `kw_when` is the
/// SIMPLE-form test (`test = Some`; absent ⇒ SEARCHED, `test = None`); after
/// `kw_when` it is a WHEN value (buffered); after `kw_then` it pairs with the
/// buffered WHEN into a `(when, then)` branch; after `kw_else` it is the
/// default. The grammar's `(…)+` guarantees ≥1 branch — we re-assert it
/// (defensive, matches the `parse_reduce_expr` / `parse_list_comprehension`
/// arity guards).
fn parse_case_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let span = span_of(&pair);

    // Which operand slot the NEXT `expression` pair fills, driven by the
    // most-recent keyword marker. Starts at `BeforeWhen` (the optional
    // simple-form test sits before the first WHEN).
    #[derive(Clone, Copy)]
    enum Slot {
        BeforeWhen,
        WhenValue,
        ThenResult,
        ElseDefault,
    }

    let mut slot = Slot::BeforeWhen;
    let mut test: Option<Box<Expression>> = None;
    let mut branches: Vec<(Expression, Expression)> = Vec::new();
    let mut default: Option<Box<Expression>> = None;
    // The WHEN value awaiting its THEN result (buffered between the
    // `kw_when` operand and the `kw_then` operand of one arm).
    let mut pending_when: Option<Expression> = None;

    for p in pair.into_inner() {
        match p.as_rule() {
            // Start / end markers carry no operand.
            Rule::kw_case | Rule::kw_case_end => {}
            Rule::kw_when => slot = Slot::WhenValue,
            Rule::kw_then => slot = Slot::ThenResult,
            Rule::kw_else => slot = Slot::ElseDefault,
            Rule::expression => {
                let e = parse_expression(p)?;
                match slot {
                    Slot::BeforeWhen => test = Some(Box::new(e)),
                    Slot::WhenValue => pending_when = Some(e),
                    Slot::ThenResult => {
                        // The grammar emits WHEN…THEN strictly paired, so a
                        // THEN operand always has a buffered WHEN.
                        let when =
                            pending_when
                                .take()
                                .ok_or_else(|| ParseError::AstConstruction {
                                    message: "CASE: THEN result with no preceding WHEN value"
                                        .into(),
                                    span: Some(span.clone()),
                                })?;
                        branches.push((when, e));
                    }
                    Slot::ElseDefault => default = Some(Box::new(e)),
                }
            }
            // No other rule shapes appear inside `case_expr`.
            _ => {}
        }
    }

    if branches.is_empty() {
        return Err(ParseError::AstConstruction {
            message: "CASE expression requires at least one `WHEN … THEN …` arm".into(),
            span: Some(span),
        });
    }

    Ok(Expression::Case {
        test,
        branches,
        default,
    })
}

/// **ADR-188** — build a [`Expression::ListPredicate`] from a
/// `filter_expr` parse pair. The pest pairs are (with `kw_*` boundary
/// markers interleaved): `[ kw_<quantifier>, identifier, kw_in,
/// expression, (kw_where, where_expr)? ]`. The quantifier keyword pair
/// is `is_kw`-filtered by the navigation helpers, so we read the
/// quantifier from the keyword rule FIRST (before filtering), then
/// collect the non-keyword pairs in order: identifier (the iteration
/// var), the list `expression`, and the optional WHERE `where_expr`.
fn parse_filter_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let span = span_of(&pair);
    let inners: Vec<_> = pair.into_inner().collect();
    // The leading keyword pair carries the quantifier. It is the first
    // `kw_all` / `kw_any` / `kw_none` / `kw_single` rule.
    let quantifier = inners
        .iter()
        .find_map(|p| match p.as_rule() {
            Rule::kw_all => Some(Quantifier::All),
            Rule::kw_any => Some(Quantifier::Any),
            Rule::kw_none => Some(Quantifier::None),
            Rule::kw_single => Some(Quantifier::Single),
            _ => None,
        })
        .ok_or_else(|| ParseError::AstConstruction {
            message: "filter_expr missing quantifier keyword".into(),
            span: Some(span.clone()),
        })?;
    // Non-keyword pairs, in source order: [identifier, expression
    // (list), where_expr?]. The `identifier` is `Rule::identifier`; the
    // list is `Rule::expression`; the optional predicate is
    // `Rule::where_expr`.
    let var = inners
        .iter()
        .find(|p| p.as_rule() == Rule::identifier)
        .map(|p| identifier_text(p))
        .ok_or_else(|| ParseError::AstConstruction {
            message: "filter_expr missing iteration variable".into(),
            span: Some(span.clone()),
        })?;
    let list_pair = inners
        .iter()
        .find(|p| p.as_rule() == Rule::expression)
        .cloned()
        .ok_or_else(|| ParseError::AstConstruction {
            message: "filter_expr missing list expression".into(),
            span: Some(span.clone()),
        })?;
    let list = Box::new(parse_expression(list_pair)?);
    // WHERE predicate. The grammar makes it optional, but the four
    // quantifiers REQUIRE it — a bare `all(x IN l)` has no predicate to
    // fold (ADR-188 Decision 4 tables key on the predicate). Reject at
    // parse time with a precise message rather than defaulting to a
    // silent `true` (honesty-gate: a weak default would mask the
    // user's error).
    let predicate_pair = inners
        .iter()
        .find(|p| p.as_rule() == Rule::where_expr)
        .cloned()
        .ok_or_else(|| ParseError::AstConstruction {
            message: format!(
                "{quantifier:?}(...) list-predicate requires a WHERE clause \
                 (e.g. `{}(x IN list WHERE <predicate>)`)",
                quantifier_keyword(quantifier)
            ),
            span: Some(span.clone()),
        })?;
    let predicate = Box::new(parse_where_expr(predicate_pair)?);
    Ok(Expression::ListPredicate {
        quantifier,
        var,
        list,
        predicate,
    })
}

/// Lowercase openCypher spelling of a [`Quantifier`] for diagnostics.
fn quantifier_keyword(q: Quantifier) -> &'static str {
    match q {
        Quantifier::All => "all",
        Quantifier::Any => "any",
        Quantifier::None => "none",
        Quantifier::Single => "single",
    }
}

/// **ADR-188** — build a [`Expression::Reduce`] from a `reduce_expr`
/// parse pair. The non-keyword pairs, in source order, are:
/// `[ identifier (acc_var), expression (init), identifier (var),
/// expression (list), expression (fold body) ]`. The `=`, `,`, `|`
/// literal tokens carry no pairs; `kw_reduce` / `kw_in` are
/// `is_kw`-filtered.
fn parse_reduce_expr(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let span = span_of(&pair);
    // Collect the non-keyword pairs in order. Two `identifier`s
    // (acc_var, then var) and three `expression`s (init, list, body).
    let mut idents: Vec<Pair<'_, Rule>> = Vec::new();
    let mut exprs: Vec<Pair<'_, Rule>> = Vec::new();
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::identifier => idents.push(p),
            Rule::expression => exprs.push(p),
            _ => {} // kw_reduce / kw_in boundary markers — skip.
        }
    }
    if idents.len() != 2 || exprs.len() != 3 {
        return Err(ParseError::AstConstruction {
            message: format!(
                "reduce(...) expects `reduce(acc = init, x IN list | expr)` \
                 (got {} identifiers, {} expressions)",
                idents.len(),
                exprs.len()
            ),
            span: Some(span),
        });
    }
    let acc_var = identifier_text(&idents[0]);
    let var = identifier_text(&idents[1]);
    let init = Box::new(parse_expression(exprs[0].clone())?);
    let list = Box::new(parse_expression(exprs[1].clone())?);
    let expr = Box::new(parse_expression(exprs[2].clone())?);
    Ok(Expression::Reduce {
        acc_var,
        init,
        var,
        list,
        expr,
    })
}

/// **ADR-188** (#620 list-half) — build a
/// [`Expression::ListComprehension`] from a `list_comprehension` parse
/// pair. The non-keyword pairs, in source order, are:
/// `[ identifier (var), expression (list), where_expr? (predicate),
/// expression? (projection) ]`. The `[`, `]`, `|` literal tokens carry
/// no pairs; `kw_in` / `kw_where` are `is_kw`-filtered.
///
/// Both the WHERE `predicate` and the `| projection` are OPTIONAL (the
/// four openCypher v9 §3.5 combinations). The `list` and the
/// `projection` are BOTH `Rule::expression`; we disambiguate by SOURCE
/// ORDER — the FIRST `expression` is the list, the SECOND (if present)
/// is the projection — exactly as `parse_reduce_expr` orders its three
/// `expression`s. The `predicate` is the distinct `Rule::where_expr`.
fn parse_list_comprehension(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let span = span_of(&pair);
    let mut var: Option<String> = None;
    // `expression` pairs in source order: [0] = list, [1] = projection.
    let mut exprs: Vec<Pair<'_, Rule>> = Vec::new();
    let mut where_pair: Option<Pair<'_, Rule>> = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            // The single `identifier` is the iteration variable. (It is
            // the FIRST non-keyword pair; there is only one.)
            Rule::identifier if var.is_none() => var = Some(identifier_text(&p)),
            Rule::expression => exprs.push(p),
            Rule::where_expr => where_pair = Some(p),
            _ => {} // `[` / `]` / `|` literals + kw_in / kw_where markers.
        }
    }
    let var = var.ok_or_else(|| ParseError::AstConstruction {
        message: "list comprehension missing iteration variable".into(),
        span: Some(span.clone()),
    })?;
    if exprs.is_empty() {
        return Err(ParseError::AstConstruction {
            message: "list comprehension missing list expression \
                      (`[x IN <list> …]`)"
                .into(),
            span: Some(span.clone()),
        });
    }
    if exprs.len() > 2 {
        return Err(ParseError::AstConstruction {
            message: format!(
                "list comprehension expects `[x IN list (WHERE p)? (| e)?]` \
                 (got {} list/projection expressions)",
                exprs.len()
            ),
            span: Some(span),
        });
    }
    // First `expression` = the list; second (if any) = the projection.
    let list = Box::new(parse_expression(exprs[0].clone())?);
    let projection = match exprs.get(1) {
        Some(p) => Some(Box::new(parse_expression(p.clone())?)),
        None => None,
    };
    let predicate = match where_pair {
        Some(p) => Some(Box::new(parse_where_expr(p)?)),
        None => None,
    };
    Ok(Expression::ListComprehension {
        var,
        list,
        predicate,
        projection,
    })
}

/// **ADR-191 D-6** (#620 map-half) — build an [`Expression::MapProjection`]
/// from a `map_projection` parse pair. The pairs are: the leading
/// `identifier` (the base variable), then zero or more
/// `map_projection_item`s (each wrapping exactly one of
/// `all_properties_selector` / `property_selector` / `literal_entry`). The
/// `{`, `}`, `,` literal tokens carry no pairs. The base is the FIRST
/// `identifier`; every subsequent item is read in source order so the
/// projected map preserves declaration order (last-writer-wins on a
/// duplicate key is the executor's `BTreeMap::insert` concern, matching
/// the map-literal carrier).
fn parse_map_projection(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let span = span_of(&pair);
    let mut base: Option<String> = None;
    let mut items: Vec<MapProjectionItem> = Vec::new();
    for p in pair.into_inner() {
        match p.as_rule() {
            // The FIRST `identifier` is the base variable. (The grammar
            // emits the base identifier before any item; a `literal_entry`'s
            // alias identifier is nested INSIDE a `map_projection_item`
            // pair, so it never reaches this top-level loop.)
            Rule::identifier if base.is_none() => base = Some(identifier_text(&p)),
            Rule::map_projection_item => items.push(parse_map_projection_item(p)?),
            _ => {} // `{` / `}` / `,` literal tokens.
        }
    }
    let base = base.ok_or_else(|| ParseError::AstConstruction {
        message: "map projection missing base variable".into(),
        span: Some(span),
    })?;
    Ok(Expression::MapProjection { base, items })
}

/// **ADR-191 D-6** — build one [`MapProjectionItem`] from a
/// `map_projection_item` parse pair. The inner pair is exactly one of:
/// `all_properties_selector` (`.*`), `property_selector` (`.key`), or
/// `literal_entry` (`alias: expr`).
fn parse_map_projection_item(pair: Pair<'_, Rule>) -> Result<MapProjectionItem, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::all_properties_selector => Ok(MapProjectionItem::AllProperties),
        Rule::property_selector => {
            // `.key` — the inner `identifier` is the property name.
            let id = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::identifier)
                .ok_or_else(|| ParseError::AstConstruction {
                    message: "map-projection property selector missing identifier".into(),
                    span: None,
                })?;
            Ok(MapProjectionItem::Property(identifier_text(&id)))
        }
        Rule::literal_entry => {
            // `alias: expr` — the FIRST `identifier` is the alias; the
            // `expression` is the value.
            let mut alias: Option<String> = None;
            let mut value: Option<Expression> = None;
            for p in inner.into_inner() {
                match p.as_rule() {
                    Rule::identifier if alias.is_none() => alias = Some(identifier_text(&p)),
                    Rule::expression => value = Some(parse_expression(p)?),
                    _ => {}
                }
            }
            let alias = alias.ok_or_else(|| ParseError::AstConstruction {
                message: "map-projection literal entry missing alias".into(),
                span: None,
            })?;
            let value = value.ok_or_else(|| ParseError::AstConstruction {
                message: "map-projection literal entry missing value expression".into(),
                span: None,
            })?;
            Ok(MapProjectionItem::Literal {
                alias,
                value: Box::new(value),
            })
        }
        r => unexpected("map_projection_item", r, &inner),
    }
}

fn parse_parameter(pair: Pair<'_, Rule>) -> Expression {
    let inner = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::identifier_inner);
    let name = match inner {
        Some(p) => p.as_str().to_string(),
        None => String::new(),
    };
    Expression::Parameter(name)
}

fn parse_function_call(pair: Pair<'_, Rule>) -> Result<Expression, ParseError> {
    let mut iter = pair.into_inner();
    let name_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "function_call missing name".into(),
        span: None,
    })?;
    // `function_call = identifier ~ "(" ~ ...` — the leading pair is
    // always `Rule::identifier`, so backtick-stripping applies.
    let name = identifier_text(&name_pair);
    let mut args = Vec::new();
    let mut distinct = false;
    let mut star = false;
    for p in iter {
        match p.as_rule() {
            // `count(*)` — star form: no expression argument.
            Rule::star_arg => star = true,
            // `count(DISTINCT x)` — the single inner expression is the
            // aggregated value; flag `distinct` for the dedup path.
            Rule::distinct_arg => {
                distinct = true;
                for inner in p.into_inner() {
                    if inner.as_rule() == Rule::expression {
                        args.push(parse_expression(inner)?);
                    }
                }
            }
            // Bare `fn(x)` / `fn(a, b, c)` expression list (unchanged).
            Rule::expression => args.push(parse_expression(p)?),
            _ => {}
        }
    }
    Ok(Expression::FunctionCall {
        name,
        args,
        distinct,
        star,
    })
}

fn parse_field_ref(pair: Pair<'_, Rule>) -> Result<FieldRef, ParseError> {
    let mut iter = pair.into_inner();
    let base_pair = iter.next().ok_or_else(|| ParseError::AstConstruction {
        message: "field_ref missing base".into(),
        span: None,
    })?;
    let base = identifier_text(&base_pair);
    let mut path = Vec::new();
    for p in iter {
        if p.as_rule() == Rule::identifier {
            path.push(identifier_text(&p));
        }
    }
    Ok(FieldRef { base, path })
}

fn parse_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::null_literal => Ok(Literal::Null),
        Rule::bool_literal => {
            let v = inner.as_str().eq_ignore_ascii_case("TRUE");
            Ok(Literal::Bool(v))
        }
        Rule::float_literal => Ok(Literal::Float(parse_float_lit(inner)?)),
        Rule::int_literal => Ok(Literal::Integer(parse_int_lit(inner)?)),
        Rule::string_literal => Ok(Literal::String(parse_string_literal(inner)?)),
        Rule::list_literal => parse_list_literal(inner),
        Rule::map_literal => parse_map_literal(inner),
        // W23-V11-T-01 / ADR-090 — temporal + decimal constructor
        // literals. Inner-string parsing routes through
        // arcgraph_core::datetime::parse_* which surfaces precise
        // diagnostics via TemporalError; the parse-error path wraps
        // them as ParseError::AstConstruction with the literal's span
        // so the user sees the offending position.
        Rule::datetime_literal => parse_datetime_literal(inner),
        Rule::localdatetime_literal => parse_localdatetime_literal(inner),
        Rule::date_literal => parse_date_literal(inner),
        Rule::duration_literal => parse_duration_literal(inner),
        Rule::decimal_literal => parse_decimal_literal(inner),
        r => unexpected("literal", r, &inner),
    }
}

/// Extract the `string_literal` payload from a temporal-constructor
/// pair (`datetime('...')` / `date('...')` / `duration('...')` /
/// `localdatetime('...')` / `decimal('...')`). The constructor
/// keyword is the first inner; the string literal is the second.
fn temporal_inner_string(pair: Pair<'_, Rule>) -> Result<(String, Span), ParseError> {
    let span = span_of(&pair);
    let s = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::string_literal)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "temporal constructor missing string literal".into(),
            span: Some(span.clone()),
        })?;
    let inner = parse_string_literal(s)?;
    Ok((inner, span))
}

fn parse_datetime_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let (s, span) = temporal_inner_string(pair)?;
    arcgraph_core::parse_zoned_datetime(&s)
        .map(Literal::Temporal)
        .map_err(|e| ParseError::AstConstruction {
            message: format!("datetime literal: {e}"),
            span: Some(span),
        })
}

fn parse_localdatetime_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let (s, span) = temporal_inner_string(pair)?;
    arcgraph_core::parse_local_datetime(&s)
        .map(Literal::LocalDateTime)
        .map_err(|e| ParseError::AstConstruction {
            message: format!("localdatetime literal: {e}"),
            span: Some(span),
        })
}

fn parse_date_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let (s, span) = temporal_inner_string(pair)?;
    arcgraph_core::parse_date(&s)
        .map(Literal::Date)
        .map_err(|e| ParseError::AstConstruction {
            message: format!("date literal: {e}"),
            span: Some(span),
        })
}

fn parse_duration_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let (s, span) = temporal_inner_string(pair)?;
    arcgraph_core::parse_duration(&s)
        .map(Literal::Duration)
        .map_err(|e| ParseError::AstConstruction {
            message: format!("duration literal: {e}"),
            span: Some(span),
        })
}

fn parse_decimal_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let (s, span) = temporal_inner_string(pair)?;
    arcgraph_core::parse_decimal(&s)
        .map(Literal::Decimal)
        .map_err(|e| ParseError::AstConstruction {
            message: format!("decimal literal: {e}"),
            span: Some(span),
        })
}

/// Decode an integer-literal token — decimal, hexadecimal (`0x…`/`0X…`),
/// or octal (`0o…`/`0O…`), with an optional leading `-`/`+` — into an
/// `i64` (openCypher v9 §3 integer literals; #618 GA Lane C).
///
/// The radix prefix is stripped and the sign is re-attached to the bare
/// digits BEFORE handing to `i64::from_str_radix`, which itself honors a
/// leading sign. That ordering is what makes the i64::MIN boundary parse
/// without an unsigned-then-negate overflow: `-9223372036854775808`'s
/// magnitude (2^63) is NOT representable as a positive i64, but the SIGNED
/// string `-9223372036854775808` (radix 10) / `-8000000000000000` (radix
/// 16) / `-1000000000000000000000` (radix 8) maps straight to `i64::MIN`.
/// An over-range magnitude (e.g. `0xFFFFFFFFFFFFFFFF`, `9223372036854775808`)
/// returns a `ParseIntError` here — a clean parse-time rejection, never a
/// panic.
fn parse_radix_i64(s: &str) -> Result<i64, std::num::ParseIntError> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (radix, digits) =
        if let Some(h) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
            (16u32, h)
        } else if let Some(o) = body.strip_prefix("0o").or_else(|| body.strip_prefix("0O")) {
            (8u32, o)
        } else {
            (10u32, body)
        };
    if neg {
        i64::from_str_radix(&format!("-{digits}"), radix)
    } else {
        i64::from_str_radix(digits, radix)
    }
}

fn parse_int_lit(pair: Pair<'_, Rule>) -> Result<i64, ParseError> {
    let span = span_of(&pair);
    parse_radix_i64(pair.as_str()).map_err(|e| ParseError::AstConstruction {
        message: format!("integer literal: {e}"),
        span: Some(span),
    })
}

/// If `inner` is structurally a BARE integer literal — i.e. the chain
/// `unary_expr → atom → primary_atom → literal → int_literal` with NO `^`,
/// accessors, or nested unary prefix — return the `int_literal` token text.
/// Used by the i64::MIN constant-fold in `parse_unary_expr`; any other shape
/// (`-x`, `-(1+2)`, `-3.5`, `-n.prop`, `-2^63`) returns `None` and keeps the
/// canonical `UnaryOp{Neg, ..}` path.
fn bare_int_literal_str(inner: &Pair<'_, Rule>) -> Option<String> {
    // `inner` is the post-sign `unary_expr`; its first child is `atom`.
    let mut inner_pairs = inner.clone().into_inner();
    let atom = inner_pairs.next()?;
    // A bare literal carries NO exponentiation suffix.
    if inner_pairs.next().is_some() {
        return None;
    }
    if atom.as_rule() != Rule::atom {
        return None;
    }
    let mut atom_inner = atom.into_inner();
    let primary = atom_inner.next()?;
    if primary.as_rule() != Rule::primary_atom {
        return None;
    }
    // A bare literal carries NO accessors (`.prop` / `[i]` / `[lo..hi]`).
    if atom_inner.next().is_some() {
        return None;
    }
    let lit = primary.into_inner().next()?;
    if lit.as_rule() != Rule::literal {
        return None;
    }
    let int_lit = lit.into_inner().next()?;
    if int_lit.as_rule() == Rule::int_literal {
        Some(int_lit.as_str().to_string())
    } else {
        None
    }
}

/// Constant-fold a unary `-` applied to a bare integer literal AT THE
/// i64::MIN BOUNDARY ONLY. Returns `Some(value)` iff the literal's POSITIVE
/// magnitude overflows i64 but the NEGATED value fits (the sole such value
/// being `i64::MIN`); returns `None` when the positive magnitude already
/// fits (caller keeps `UnaryOp{Neg, Integer(mag)}` — no AST perturbation)
/// OR when even the negated value overflows (caller's normal descent then
/// surfaces the clean "integer literal" parse error for genuinely
/// out-of-range input like `-9223372036854775809`). `lit_str` is the bare
/// (sign-free) `int_literal` text — the grammar's unary `-` already
/// consumed the sign.
fn fold_min_boundary_int(lit_str: &str) -> Option<i64> {
    // Positive magnitude representable ⇒ do NOT fold (preserve UnaryOp).
    if parse_radix_i64(lit_str).is_ok() {
        return None;
    }
    // Magnitude overflowed positive i64; the negated form fits only at
    // i64::MIN. `.ok()` drops the genuinely-too-small case to the normal
    // path's clean error.
    parse_radix_i64(&format!("-{lit_str}")).ok()
}

fn parse_float_lit(pair: Pair<'_, Rule>) -> Result<f64, ParseError> {
    let span = span_of(&pair);
    pair.as_str()
        .parse()
        .map_err(|e: std::num::ParseFloatError| ParseError::AstConstruction {
            message: format!("float literal: {e}"),
            span: Some(span),
        })
}

fn parse_string_literal(pair: Pair<'_, Rule>) -> Result<String, ParseError> {
    let raw = pair.as_str();
    if raw.len() < 2 {
        return Err(ParseError::AstConstruction {
            message: "string literal too short".into(),
            span: Some(span_of(&pair)),
        });
    }
    let inner = &raw[1..raw.len() - 1];
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => out.push('"'),
                Some('\'') => out.push('\''),
                Some('\\') => out.push('\\'),
                Some('/') => out.push('/'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('0') => out.push('\0'),
                Some('b') => out.push('\u{0008}'),
                Some('f') => out.push('\u{000C}'),
                Some('u') => {
                    let mut hex = String::with_capacity(4);
                    for _ in 0..4 {
                        if let Some(h) = chars.next() {
                            hex.push(h);
                        }
                    }
                    let code =
                        u32::from_str_radix(&hex, 16).map_err(|e| ParseError::AstConstruction {
                            message: format!("\\u{hex}: {e}"),
                            span: Some(span_of(&pair)),
                        })?;
                    if let Some(c) = char::from_u32(code) {
                        out.push(c);
                    } else {
                        return Err(ParseError::AstConstruction {
                            message: format!("invalid unicode \\u{hex}"),
                            span: Some(span_of(&pair)),
                        });
                    }
                }
                Some(other) => {
                    return Err(ParseError::AstConstruction {
                        message: format!("unknown escape \\{other}"),
                        span: Some(span_of(&pair)),
                    });
                }
                None => {
                    return Err(ParseError::AstConstruction {
                        message: "trailing backslash in string literal".into(),
                        span: Some(span_of(&pair)),
                    });
                }
            }
        } else {
            out.push(c);
        }
    }
    Ok(out)
}

fn parse_list_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let mut out = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::expression {
            out.push(parse_expression(p)?);
        }
    }
    Ok(Literal::List(out))
}

fn parse_map_literal(pair: Pair<'_, Rule>) -> Result<Literal, ParseError> {
    let mut out = Vec::new();
    for p in pair.into_inner() {
        if p.as_rule() == Rule::map_entry {
            let mut iter = p.into_inner();
            let k = iter.next().ok_or_else(|| ParseError::AstConstruction {
                message: "map_entry missing key".into(),
                span: None,
            })?;
            let v = iter.next().ok_or_else(|| ParseError::AstConstruction {
                message: "map_entry missing value".into(),
                span: None,
            })?;
            // `map_entry = { map_key ~ ":" ~ expression }` — the key is a
            // `Rule::map_key` (the expression-context property-key class
            // admitting reserved words like `NULL` without backticks;
            // openCypher Map1[5]/Map2[5]). Backtick-stripping applies in
            // the escaped form just as for `identifier`.
            out.push((map_key_text(&k), parse_expression(v)?));
        }
    }
    Ok(Literal::Map(out))
}

// =====================================================================
// Index DDL
// =====================================================================

/// Dispatch a `ddl_statement` to its concrete index DDL form.
fn parse_ddl_statement(pair: Pair<'_, Rule>) -> Result<Statement, ParseError> {
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::create_vector_index_ddl => parse_create_vector_index(inner)
            .map(|c| Statement::IndexDdl(IndexDdlStatement::CreateVector(c))),
        Rule::create_property_index_ddl => parse_create_property_index(inner)
            .map(|c| Statement::IndexDdl(IndexDdlStatement::CreateProperty(c))),
        Rule::drop_index_ddl => {
            parse_drop_index(inner).map(|d| Statement::IndexDdl(IndexDdlStatement::Drop(d)))
        }
        other => Err(ParseError::AstConstruction {
            message: format!("unexpected DDL statement rule: {other:?}"),
            span: Some(span_of(&inner)),
        }),
    }
}

/// Extract an `index_name` (`parameter | identifier`) into an
/// [`IndexNameRef`]. Neo4j-compatible clients may pass the name as a
/// `$param`; a literal identifier is also admitted (#830).
fn parse_index_name(pair: Pair<'_, Rule>) -> Result<IndexNameRef, ParseError> {
    let span = span_of(&pair);
    let inner = first_inner(pair)?;
    match inner.as_rule() {
        Rule::parameter => {
            // `parameter = ${ "$" ~ identifier_inner }` — the name is the
            // `identifier_inner` child (matches `parse_parameter`).
            let name = inner
                .into_inner()
                .find(|p| p.as_rule() == Rule::identifier_inner)
                .map(|p| p.as_str().to_string())
                .unwrap_or_default();
            Ok(IndexNameRef::Param(name))
        }
        Rule::identifier => Ok(IndexNameRef::Literal(identifier_text(&inner))),
        other => Err(ParseError::AstConstruction {
            message: format!("index name expected parameter or identifier, got {other:?}"),
            span: Some(span),
        }),
    }
}

/// Extract the indexed property path from an `index_property`
/// (`("(" field_ref ")") | field_ref`, where `field_ref = identifier
/// ("." identifier)+`). Returns the segment(s) AFTER the leading
/// pattern variable (`n.embedding` → `"embedding"`; `n.a.b` → `"a.b"`).
fn parse_index_property(pair: Pair<'_, Rule>) -> Result<String, ParseError> {
    let span = span_of(&pair);
    let field_ref = pair
        .into_inner()
        .find(|p| p.as_rule() == Rule::field_ref)
        .ok_or_else(|| ParseError::AstConstruction {
            message: "ON <var>.<property> missing a property reference".into(),
            span: Some(span.clone()),
        })?;
    let idents: Vec<String> = field_ref
        .into_inner()
        .filter(|p| p.as_rule() == Rule::identifier)
        .map(|p| identifier_text(&p))
        .collect();
    if idents.len() < 2 {
        return Err(ParseError::AstConstruction {
            message: "ON <var>.<property> requires a `var.property` reference".into(),
            span: Some(span),
        });
    }
    // idents[0] is the pattern variable; the remainder is the property
    // path (single segment for the common `n.embedding` shape).
    Ok(idents[1..].join("."))
}

/// Parse `CREATE VECTOR INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
/// var.prop [OPTIONS { … }]` (#830). Matches the common
/// Neo4j-compatible wire form.
fn parse_create_vector_index(
    pair: Pair<'_, Rule>,
) -> Result<CreateVectorIndexStatement, ParseError> {
    let span = span_of(&pair);
    let mut name: Option<IndexNameRef> = None;
    let mut if_not_exists = false;
    let mut pattern_var: Option<String> = None;
    let mut label: Option<String> = None;
    let mut property: Option<String> = None;
    let mut options: Option<Expression> = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::index_name => name = Some(parse_index_name(p)?),
            Rule::if_not_exists => if_not_exists = true,
            Rule::index_pattern_var => pattern_var = Some(identifier_text(&first_inner(p)?)),
            Rule::index_label => label = Some(identifier_text(&first_inner(p)?)),
            Rule::index_property => property = Some(parse_index_property(p)?),
            // OPTIONS reuses the map literal (backtick keys, nested map,
            // function-call / parameter values all already supported);
            // captured verbatim, NOT interpreted here (ADR-198 §OQ-7).
            Rule::map_literal => options = Some(Expression::Literal(parse_map_literal(p)?)),
            _ => {}
        }
    }
    Ok(CreateVectorIndexStatement {
        name: name.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE VECTOR INDEX missing an index name".into(),
            span: Some(span.clone()),
        })?,
        if_not_exists,
        pattern_var: pattern_var.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE VECTOR INDEX missing FOR (var:Label) pattern variable".into(),
            span: Some(span.clone()),
        })?,
        label: label.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE VECTOR INDEX missing FOR (var:Label) label".into(),
            span: Some(span.clone()),
        })?,
        property: property.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE VECTOR INDEX missing ON var.property".into(),
            span: Some(span),
        })?,
        options,
    })
}

/// Parse `CREATE INDEX <name> [IF NOT EXISTS] FOR (var:Label) ON
/// (var.prop)` (#1366, task #248) — the user-visible property index.
/// Same shape as the vector form minus the `VECTOR` keyword + OPTIONS.
fn parse_create_property_index(
    pair: Pair<'_, Rule>,
) -> Result<CreatePropertyIndexStatement, ParseError> {
    let span = span_of(&pair);
    let mut name: Option<IndexNameRef> = None;
    let mut if_not_exists = false;
    let mut pattern_var: Option<String> = None;
    let mut label: Option<String> = None;
    let mut property: Option<String> = None;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::index_name => name = Some(parse_index_name(p)?),
            Rule::if_not_exists => if_not_exists = true,
            Rule::index_pattern_var => pattern_var = Some(identifier_text(&first_inner(p)?)),
            Rule::index_label => label = Some(identifier_text(&first_inner(p)?)),
            Rule::index_property => property = Some(parse_index_property(p)?),
            _ => {}
        }
    }
    Ok(CreatePropertyIndexStatement {
        name: name.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE INDEX missing an index name".into(),
            span: Some(span.clone()),
        })?,
        if_not_exists,
        pattern_var: pattern_var.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE INDEX missing FOR (var:Label) pattern variable".into(),
            span: Some(span.clone()),
        })?,
        label: label.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE INDEX missing FOR (var:Label) label".into(),
            span: Some(span.clone()),
        })?,
        property: property.ok_or_else(|| ParseError::AstConstruction {
            message: "CREATE INDEX missing ON (var.property)".into(),
            span: Some(span),
        })?,
    })
}

/// Parse `DROP INDEX <name> [IF EXISTS]` (#830) — the generic Neo4j
/// form emitted by vector clients (Neo4j has no `DROP VECTOR INDEX`;
/// this drops any index by name).
fn parse_drop_index(pair: Pair<'_, Rule>) -> Result<DropIndexStatement, ParseError> {
    let span = span_of(&pair);
    let mut name: Option<IndexNameRef> = None;
    let mut if_exists = false;
    for p in pair.into_inner() {
        match p.as_rule() {
            Rule::index_name => name = Some(parse_index_name(p)?),
            Rule::if_exists => if_exists = true,
            _ => {}
        }
    }
    Ok(DropIndexStatement {
        name: name.ok_or_else(|| ParseError::AstConstruction {
            message: "DROP INDEX missing an index name".into(),
            span: Some(span),
        })?,
        if_exists,
    })
}

// =====================================================================
// Tiny helpers
// =====================================================================

/// Check if a pair is a keyword-shape rule (`kw_*`). Keyword rules
/// in `grammar.pest` are atomic helpers that match a case-
/// insensitive keyword token plus the non-word-character boundary
/// per ADR-038 OQ-38-5 mitigation. They surface in the pair tree
/// but carry no AST-relevant data; the parser skips them when
/// walking inner pairs.
fn is_kw(rule: Rule) -> bool {
    let name = format!("{rule:?}");
    name.starts_with("kw_") || name == "WHITESPACE" || name == "COMMENT"
}

fn first_inner(pair: Pair<'_, Rule>) -> Result<Pair<'_, Rule>, ParseError> {
    pair.into_inner()
        .find(|p| !is_kw(p.as_rule()))
        .ok_or_else(|| ParseError::AstConstruction {
            message: "expected inner pair, got empty".into(),
            span: None,
        })
}

fn expect_rule<'a>(
    iter: &mut Pairs<'a, Rule>,
    expected: Rule,
    label: &str,
    span: &Span,
) -> Result<Pair<'a, Rule>, ParseError> {
    // Skip keyword-shape pairs that the grammar emits as atomic
    // boundary-asserted markers (kw_*). They carry no AST data.
    let p = loop {
        let p = iter.next().ok_or_else(|| ParseError::AstConstruction {
            message: format!("expected {label}, got nothing"),
            span: Some(span.clone()),
        })?;
        if !is_kw(p.as_rule()) {
            break p;
        }
    };
    if p.as_rule() != expected {
        return Err(ParseError::AstConstruction {
            message: format!(
                "expected {label} (Rule::{expected:?}), got Rule::{:?}",
                p.as_rule()
            ),
            span: Some(span_of(&p)),
        });
    }
    Ok(p)
}

fn unexpected<T>(context: &str, r: Rule, pair: &Pair<'_, Rule>) -> Result<T, ParseError> {
    Err(ParseError::AstConstruction {
        message: format!("unexpected {context}: Rule::{r:?}"),
        span: Some(span_of(pair)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_match_return() {
        let q = parse("MATCH (n) RETURN n").expect("parse");
        match q {
            Statement::Read(r) => {
                assert_eq!(r.clauses.len(), 2);
            }
            _ => panic!("expected Statement::Read"),
        }
    }

    #[test]
    fn cz842_parse_with_distinct_sets_flag() {
        // #842 part B — `WITH DISTINCT …` parses (was a `-32700` parse
        // error: no `kw_distinct?` in `with_clause`) and sets the AST
        // `distinct` flag; plain `WITH …` leaves it false.
        fn with_flag(query: &str) -> bool {
            let q = parse(query).expect("parse");
            let Statement::Read(r) = q else {
                panic!("expected Statement::Read")
            };
            r.clauses
                .iter()
                .find_map(|c| match c {
                    Clause::With(w) => Some(w.distinct),
                    _ => None,
                })
                .expect("a WITH clause")
        }
        assert!(
            with_flag("MATCH (n) WITH DISTINCT n.x AS x RETURN x"),
            "WITH DISTINCT sets distinct=true"
        );
        assert!(
            with_flag("MATCH (n) WITH  DISTINCT n.x AS x RETURN x"),
            "WITH double-space DISTINCT sets distinct=true"
        );
        assert!(
            with_flag("MATCH (n) WITH\tDISTINCT n.x AS x RETURN x"),
            "WITH tab DISTINCT sets distinct=true"
        );
        assert!(
            !with_flag("MATCH (n) WITH n.x AS x RETURN x"),
            "plain WITH leaves distinct=false"
        );
        // A column whose name merely STARTS with 'distinct' must not
        // false-positive — the `kw_end` word boundary guards it (same
        // protection `RETURN distinct_col` relies on).
        assert!(
            !with_flag("MATCH (n) WITH n.distinctish AS distinctish RETURN distinctish"),
            "an identifier with a 'distinct' prefix is not the DISTINCT keyword"
        );
        assert!(
            !with_flag("WITH distinct_col AS c RETURN c"),
            "WITH distinct_col does not false-positive as DISTINCT"
        );
    }

    #[test]
    fn c926_parse_return_distinct_after_multi_whitespace_sets_flag() {
        fn return_flag(query: &str) -> bool {
            let q = parse(query).expect("parse");
            let Statement::Read(r) = q else {
                panic!("expected Statement::Read")
            };
            r.clauses
                .iter()
                .find_map(|c| match c {
                    Clause::Return(r) => Some(r.distinct),
                    _ => None,
                })
                .expect("a RETURN clause")
        }

        assert!(
            return_flag("MATCH (n) RETURN DISTINCT n.x"),
            "RETURN DISTINCT sets distinct=true"
        );
        assert!(
            return_flag("MATCH (n) RETURN  DISTINCT n.x"),
            "RETURN double-space DISTINCT sets distinct=true"
        );
        assert!(
            return_flag("MATCH (n) RETURN\tDISTINCT n.x"),
            "RETURN tab DISTINCT sets distinct=true"
        );
        assert!(
            !return_flag("MATCH (n) RETURN n.x"),
            "plain RETURN leaves distinct=false"
        );
    }

    #[test]
    fn cz842_with_distinct_display_round_trips() {
        // Display renders `WITH DISTINCT …` (cache-key / EXPLAIN parity
        // with `RETURN DISTINCT`), and the rendered form re-parses with
        // the flag preserved.
        let q = parse("MATCH (n) WITH DISTINCT n.x AS x RETURN x").expect("parse");
        let Statement::Read(r) = &q else {
            panic!("expected Statement::Read")
        };
        let with = r
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::With(w) => Some(w),
                _ => None,
            })
            .expect("a WITH clause");
        let rendered = format!("{with}");
        assert!(
            rendered.contains("WITH DISTINCT "),
            "Display must render WITH DISTINCT, got: {rendered}"
        );
    }

    #[test]
    fn parse_rejects_match_without_return_returns_err() {
        // M4-01 syntactic well-formedness: bare "MATCH (n)" with no
        // RETURN/WITH/UNWIND/RANK BY clause IS valid syntax (a
        // read_query is `clause+`); only an empty input or one with
        // a non-clause production fails. So this test pins the
        // *actual* parser-level failure: incomplete input like a
        // trailing comma on a property map.
        let r = parse("MATCH (n {prop:})");
        assert!(r.is_err(), "expected parse error, got {r:?}");
    }

    /// Concrete-oracle companion to the F-02 mutation proptest.
    ///
    /// The proptest in `tests/grammar_proptest.rs` exercises span
    /// correctness over random ASTs, but its oracle is a tolerance
    /// band (±SLACK), and the case generation is dominated by long
    /// well-formed queries with whitespace insertion points. Codex
    /// M4-01 retro F-02 (HIGH): a wide oracle would let genuine span
    /// misalignment by 50-100 bytes pass.
    ///
    /// This test pins the EXACT byte-range coordinates the parser
    /// emits for three KNOWN syntactic faults — a stricter oracle
    /// than the proptest can offer. Any drift in pest's span
    /// reporting (e.g., a future grammar refactor that changes which
    /// rule the parser was attempting at the failure point) shows up
    /// here as an exact-match assertion failure.
    ///
    /// The fault classes covered:
    ///   1. Missing close paren after a node pattern — parser was
    ///      mid-`label_spec_or_property_map`, fails at the unexpected
    ///      `WHERE` token.
    ///   2. Unexpected duplicate `=` in a comparison — parser was
    ///      at `primary_atom` position after the first `=`, fails on
    ///      the second.
    ///   3. Malformed property access — `n.` followed by end-of-input
    ///      where the parser expected an identifier (or backtick-
    ///      escape).
    #[test]
    fn parser_error_span_points_at_offending_token() {
        // (1) Missing close paren after `n`. Byte 9 = 'W' (of WHERE),
        //     which is exactly where the parser realized it expected
        //     `)` instead.
        let bad_paren = "MATCH (n WHERE n.x = 1 RETURN n";
        let err = parse(bad_paren).expect_err("missing close paren must error");
        let span = err
            .span_byte_range(bad_paren)
            .expect("Pest variant always carries a span");
        assert_eq!(
            span,
            (9, 9),
            "missing-close-paren span MUST point at byte 9 ('W' of WHERE), got {span:?}; full err: {err}"
        );

        // (2) Duplicate `=` in WHERE comparison. Byte 21 = the second
        //     `=`, where pest's `primary_atom` expectation fails.
        let bad_eq = "MATCH (n) WHERE n.x == 1 RETURN n";
        let err = parse(bad_eq).expect_err("`==` must error");
        let span = err
            .span_byte_range(bad_eq)
            .expect("Pest variant always carries a span");
        assert_eq!(
            span,
            (21, 21),
            "unexpected-`==` span MUST point at byte 21 (second `=`), got {span:?}; full err: {err}"
        );

        // (3) Malformed property access — `n.` with nothing after.
        //     Byte 19 = end of input, the position the parser expected
        //     an identifier or backtick-escape.
        let bad_prop = "MATCH (n) RETURN n.";
        let err = parse(bad_prop).expect_err("trailing `.` must error");
        let span = err
            .span_byte_range(bad_prop)
            .expect("Pest variant always carries a span");
        assert_eq!(
            span,
            (19, 19),
            "trailing-dot span MUST point at byte 19 (end of input), got {span:?}; full err: {err}"
        );
    }

    // =================================================================
    // M4-83 multi-statement parser unit tests (ADR-038 §5.4.1 closure)
    // =================================================================

    #[test]
    fn parse_multi_single_statement_admitted() {
        // M4-83 unit #1: parse_multi admits a single-statement input
        // (degenerate chain of length 1) so callers route both shapes
        // through one entry point.
        let stmts = parse_multi("MATCH (n) RETURN n").expect("parse_multi");
        assert_eq!(stmts.len(), 1);
        assert!(matches!(stmts[0], Statement::Read(_)));
    }

    #[test]
    fn parse_multi_three_statement_chain_admitted() {
        // M4-83 unit #2: parse_multi admits a 3-statement chain with
        // semicolon separators.
        let q = "MATCH (a) RETURN a; MATCH (b) RETURN b; MATCH (c) RETURN c";
        let stmts = parse_multi(q).expect("parse_multi");
        assert_eq!(stmts.len(), 3);
        for s in &stmts {
            assert!(matches!(s, Statement::Read(_)));
        }
    }

    #[test]
    fn parse_rejects_multi_statement() {
        // M4-83 unit #3: backward-compat — `parse()` (single-statement
        // entry) rejects a multi-statement input. Callers must route
        // through `parse_multi`.
        let r = parse("MATCH (a) RETURN a; MATCH (b) RETURN b");
        assert!(
            r.is_err(),
            "parse() must reject multi-statement input; got {r:?}"
        );
    }

    #[test]
    fn parse_multi_admits_trailing_semicolon() {
        // M4-83 unit #4: trailing `;` is admissible in a multi-
        // statement chain (matches the single-statement
        // `";"?` discipline in the canonical grammar).
        let stmts = parse_multi("MATCH (a) RETURN a; MATCH (b) RETURN b;")
            .expect("parse_multi with trailing semi");
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn parse_multi_empty_input_rejected() {
        // M4-83 unit #5: the empty string fails — the grammar requires
        // at least one statement.
        let r = parse_multi("");
        assert!(r.is_err());
    }

    #[test]
    fn parse_multi_preserves_per_statement_shapes() {
        // M4-83 unit #6: per-statement Statement-variant fidelity.
        // Mixed Read + EXPLAIN-wrapped Read in the same chain land as
        // distinct AST variants (the EXPLAIN wrapper is preserved at
        // parse time; the QueryEngine layer strips it).
        let q = "MATCH (a) RETURN a; EXPLAIN MATCH (b) RETURN b";
        let stmts = parse_multi(q).expect("parse_multi");
        assert_eq!(stmts.len(), 2);
        assert!(matches!(stmts[0], Statement::Read(_)));
        assert!(matches!(stmts[1], Statement::Explain(_)));
    }

    // =================================================================
    // W23-V11-T-01 / ADR-090 — temporal + decimal literal parser pins
    // =================================================================

    fn extract_return_literal(input: &str) -> Literal {
        let stmt = parse(input).expect("parse");
        let q = match stmt {
            Statement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let ret = q
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::Return(r) => Some(r),
                _ => None,
            })
            .expect("RETURN clause");
        let it = &ret.items[0];
        let e = match &it.kind {
            ProjectionKind::Expr(e) => e,
            _ => panic!("expected Expr projection"),
        };
        match e {
            Expression::Literal(l) => l.clone(),
            _ => panic!("expected literal, got {e:?}"),
        }
    }

    #[test]
    fn parse_datetime_literal_utc_z() {
        let lit = extract_return_literal("RETURN datetime('2026-05-24T12:00:00Z')");
        match lit {
            Literal::Temporal(t) => assert_eq!(t.offset_seconds(), 0),
            other => panic!("expected Temporal, got {other:?}"),
        }
    }

    #[test]
    fn parse_datetime_literal_positive_offset() {
        let lit = extract_return_literal("RETURN datetime('2026-05-24T13:00:00+01:00')");
        match lit {
            Literal::Temporal(t) => assert_eq!(t.offset_seconds(), 3600),
            other => panic!("expected Temporal, got {other:?}"),
        }
    }

    #[test]
    fn parse_date_literal() {
        let lit = extract_return_literal("RETURN date('2026-05-24')");
        match lit {
            Literal::Date(d) => {
                assert_eq!(d.year, 2026);
                assert_eq!(d.ordinal, 144);
            }
            other => panic!("expected Date, got {other:?}"),
        }
    }

    #[test]
    fn parse_localdatetime_literal() {
        let lit = extract_return_literal("RETURN localdatetime('2026-05-24T08:00:00')");
        match lit {
            Literal::LocalDateTime(ldt) => assert_eq!(ldt.year, 2026),
            other => panic!("expected LocalDateTime, got {other:?}"),
        }
    }

    #[test]
    fn parse_duration_literal() {
        let lit = extract_return_literal("RETURN duration('PT1H30M')");
        match lit {
            Literal::Duration(d) => {
                assert_eq!(d.months, 0);
                assert_eq!(d.nanos, (3600 + 1800) * 1_000_000_000);
            }
            other => panic!("expected Duration, got {other:?}"),
        }
    }

    #[test]
    fn parse_decimal_literal() {
        let lit = extract_return_literal("RETURN decimal('100.50')");
        match lit {
            Literal::Decimal(d) => {
                assert_eq!(d.scale, 2);
                assert_eq!(d.units, 10050);
            }
            other => panic!("expected Decimal, got {other:?}"),
        }
    }

    #[test]
    fn parse_temporal_literal_lowercase_keyword() {
        // Per the rest of the grammar, keywords are case-insensitive.
        let lit = extract_return_literal("RETURN DATETIME('2026-05-24T12:00:00Z')");
        assert!(matches!(lit, Literal::Temporal(_)));
    }

    #[test]
    fn parse_temporal_literal_rejects_malformed_iso_string() {
        // The grammar admits `datetime('...')` syntactically; the
        // arcgraph_core::parse_zoned_datetime call within
        // parse_datetime_literal surfaces a TemporalError that wraps
        // as ParseError::AstConstruction.
        let r = parse("RETURN datetime('not-a-date')");
        assert!(r.is_err(), "malformed datetime literal must error");
    }

    #[test]
    fn parse_temporal_literal_in_property_map() {
        // datetime(...) literal as a value in a node property map.
        let stmt =
            parse("MATCH (n {created: datetime('2026-05-24T12:00:00Z')}) RETURN n").expect("parse");
        let q = match stmt {
            Statement::Read(q) => q,
            _ => panic!(),
        };
        let m = match &q.clauses[0] {
            Clause::Match(m) => m,
            _ => panic!(),
        };
        let MatchBody::Patterns(ps) = &m.body else {
            panic!("expected Patterns");
        };
        let props = ps[0].head.properties.as_ref().expect("node has props");
        let val = &props.entries[0].1;
        let lit = match val {
            Expression::Literal(l) => l,
            other => panic!("expected literal, got {other:?}"),
        };
        assert!(matches!(lit, Literal::Temporal(_)));
    }

    #[test]
    fn parse_round_trip_for_temporal_literal() {
        // Display → re-parse round-trip for the temporal-constructor
        // surface. Mirrors the F-01 256-case proptest discipline (pins
        // the Display/parse symmetry per ADR-038 amendment-09).
        for input in [
            "RETURN datetime('2026-05-24T12:00:00.000000000Z')",
            "RETURN localdatetime('2026-05-24T12:00:00.000000000')",
            "RETURN date('2026-05-24')",
            "RETURN duration('PT1H30M')",
            "RETURN decimal('100.5')",
        ] {
            let lit = extract_return_literal(input);
            let printed = format!("{lit}");
            // The Display form is the canonical constructor. Re-parse
            // it (as a RETURN-clause expression).
            let round_trip = parse(&format!("RETURN {printed}"));
            assert!(
                round_trip.is_ok(),
                "round-trip failed for {input}: printed='{printed}' err={round_trip:?}"
            );
        }
    }

    #[test]
    fn parse_temporal_literal_does_not_shadow_property_access() {
        // The temporal-keyword names are NOT in the `keyword` exclusion
        // set, so `n.datetime` (property access) still parses cleanly.
        let stmt = parse("MATCH (n) RETURN n.datetime").expect("parse");
        let q = match stmt {
            Statement::Read(q) => q,
            _ => panic!(),
        };
        let r = q
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::Return(r) => Some(r),
                _ => None,
            })
            .expect("RETURN");
        let e = match &r.items[0].kind {
            ProjectionKind::Expr(e) => e,
            _ => panic!(),
        };
        // Should be a property access, not a function call.
        assert!(
            matches!(e, Expression::PropertyAccess { .. }),
            "expected PropertyAccess; got {e:?}"
        );
    }

    // =================================================================
    // ADR-188 — list-predicate (`all`/`any`/`none`/`single`) + `reduce`
    // PARSER tests: AST shape + Display round-trip.
    // =================================================================

    /// Extract the first RETURN projection expression from a query.
    fn return_expr(query: &str) -> Expression {
        let stmt = parse(query).expect("parse");
        let q = match stmt {
            Statement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let r = q
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::Return(r) => Some(r),
                _ => None,
            })
            .expect("RETURN");
        match &r.items[0].kind {
            ProjectionKind::Expr(e) => e.clone(),
            _ => panic!("expected Expr projection"),
        }
    }

    // ----------------------------------------------------------------
    // openCypher v9 §3.3.6 (#773) — string-predicate parser tests.
    // STARTS WITH / ENDS WITH / CONTAINS parse to `BinaryOp` at the
    // comparison-precedence tier, in BOTH RETURN and WHERE position.
    // ----------------------------------------------------------------

    #[test]
    fn string_predicate_return_position_ast_shape() {
        for (q, want) in [
            ("RETURN 'abc' STARTS WITH 'a'", BinOp::StartsWith),
            ("RETURN 'abc' ENDS WITH 'c'", BinOp::EndsWith),
            ("RETURN 'abc' CONTAINS 'b'", BinOp::Contains),
        ] {
            match return_expr(q) {
                Expression::BinaryOp { op, lhs, rhs } => {
                    assert_eq!(op, want, "op for `{q}`");
                    assert!(
                        matches!(*lhs, Expression::Literal(Literal::String(ref s)) if s == "abc"),
                        "lhs for `{q}`"
                    );
                    assert!(
                        matches!(*rhs, Expression::Literal(Literal::String(_))),
                        "rhs for `{q}`"
                    );
                }
                other => panic!("expected BinaryOp for `{q}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn string_predicate_case_insensitive_keywords() {
        // Keyword wrappers are case-insensitive `^"…"`, like every other kw.
        for q in [
            "RETURN 'abc' starts with 'a'",
            "RETURN 'abc' Ends With 'c'",
            "RETURN 'abc' contains 'b'",
        ] {
            assert!(
                matches!(return_expr(q), Expression::BinaryOp { .. }),
                "case-insensitive `{q}` must parse as BinaryOp"
            );
        }
    }

    #[test]
    fn string_predicate_binds_tighter_than_or() {
        // `'a' STARTS WITH 'a' OR false` ⇒ `('a' STARTS WITH 'a') OR false`:
        // the top operator is OR and its LHS is the StartsWith BinaryOp
        // (the discriminating precedence oracle — Precedence4 [4]).
        match return_expr("RETURN 'a' STARTS WITH 'a' OR false") {
            Expression::BinaryOp {
                op: BinOp::Or, lhs, ..
            } => {
                assert!(
                    matches!(
                        *lhs,
                        Expression::BinaryOp {
                            op: BinOp::StartsWith,
                            ..
                        }
                    ),
                    "STARTS WITH must bind tighter than OR"
                );
            }
            other => panic!("expected top-level OR, got {other:?}"),
        }
    }

    #[test]
    fn string_predicate_in_where_position() {
        // The WHERE ladder (`special_pred`) admits the operator too.
        let stmt = parse("MATCH (n) WHERE n.name CONTAINS 'x' RETURN n").expect("parse");
        let q = match stmt {
            Statement::Read(q) => q,
            _ => panic!("expected Read"),
        };
        let where_pred = q
            .clauses
            .iter()
            .find_map(|c| match c {
                Clause::Match(m) => m.where_clause.clone(),
                _ => None,
            })
            .expect("WHERE clause");
        assert!(
            matches!(
                where_pred,
                Expression::BinaryOp {
                    op: BinOp::Contains,
                    ..
                }
            ),
            "WHERE-position CONTAINS must parse as BinaryOp{{Contains}}, got {where_pred:?}"
        );
    }

    #[test]
    fn string_op_keywords_do_not_swallow_lowercase_identifiers() {
        // Case-sensitive keyword exclusion: lowercase `starts`/`ends`/
        // `contains` remain valid property names (parity with `n.in`/`n.is`).
        for prop in ["starts", "ends", "contains"] {
            let q = format!("MATCH (n) RETURN n.{prop}");
            assert!(
                parse(&q).is_ok(),
                "`n.{prop}` (lowercase) must parse as a property access, not an operator"
            );
        }
    }

    // ----------------------------------------------------------------
    // openCypher v9 §3.6 (#621) — CASE expression parser tests.
    // ----------------------------------------------------------------

    #[test]
    fn case_parse_simple_form() {
        // `CASE x WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END` — SIMPLE:
        // leading test present, 2 arms, ELSE present.
        let e = return_expr("RETURN CASE x WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END");
        match e {
            Expression::Case {
                test,
                branches,
                default,
            } => {
                assert!(
                    matches!(test.as_deref(), Some(Expression::Identifier(v)) if v == "x"),
                    "simple form ⇒ test = Some(Identifier(x))"
                );
                assert_eq!(branches.len(), 2, "two WHEN…THEN arms");
                assert!(matches!(
                    &branches[0].0,
                    Expression::Literal(Literal::Integer(1))
                ));
                assert!(matches!(
                    &branches[0].1,
                    Expression::Literal(Literal::String(s)) if s == "a"
                ));
                assert!(matches!(
                    &branches[1].0,
                    Expression::Literal(Literal::Integer(2))
                ));
                assert!(
                    matches!(default.as_deref(), Some(Expression::Literal(Literal::String(s))) if s == "c"),
                    "ELSE present"
                );
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn case_parse_searched_form() {
        // `CASE WHEN x > 0 THEN 'pos' ELSE 'np' END` — SEARCHED: NO leading
        // test (`test = None`), the WHEN is a boolean condition. The
        // `!kw_when` grammar guard is what makes the leading test elide
        // cleanly here (no greedy consumption of `WHEN` as an identifier).
        let e = return_expr("RETURN CASE WHEN x > 0 THEN 'pos' ELSE 'np' END");
        match e {
            Expression::Case {
                test,
                branches,
                default,
            } => {
                assert!(test.is_none(), "searched form ⇒ test = None");
                assert_eq!(branches.len(), 1);
                assert!(
                    matches!(&branches[0].0, Expression::BinaryOp { op: BinOp::Gt, .. }),
                    "the WHEN is a `>` comparison condition"
                );
                assert!(default.is_some(), "ELSE present");
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn case_parse_simple_no_else() {
        // `CASE x WHEN 1 THEN 'a' END` — SIMPLE, no ELSE ⇒ default = None.
        let e = return_expr("RETURN CASE x WHEN 1 THEN 'a' END");
        match e {
            Expression::Case { test, default, .. } => {
                assert!(test.is_some());
                assert!(default.is_none(), "no ELSE ⇒ default None");
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn case_parse_searched_no_else() {
        // `CASE WHEN true THEN 'a' END` — SEARCHED, no ELSE.
        let e = return_expr("RETURN CASE WHEN true THEN 'a' END");
        match e {
            Expression::Case {
                test,
                branches,
                default,
            } => {
                assert!(test.is_none());
                assert_eq!(branches.len(), 1);
                assert!(default.is_none());
            }
            other => panic!("expected Case, got {other:?}"),
        }
    }

    #[test]
    fn case_soft_keywords_stay_property_names() {
        // SOFT-KEYWORD posture: `case` / `when` / `then` / `else` / `end`
        // (lowercase) keep parsing as property names — they are NOT in the
        // reserved `keyword` set (parallel to `n.and` / `n.or`).
        for q in [
            "RETURN n.case",
            "RETURN n.when",
            "RETURN n.then",
            "RETURN n.else",
            "RETURN n.end",
        ] {
            let e = return_expr(q);
            assert!(
                matches!(e, Expression::PropertyAccess { .. }),
                "`{q}` should parse as a property access, got {e:?}"
            );
        }
    }

    #[test]
    fn case_display_round_trips() {
        // `parse(format!("{e}")) == e` for both forms + the no-ELSE variants
        // (the round-trip property the grammar proptest pins for every
        // Expression). The Display bytestream is ALSO what the plan-cache key
        // hashes to distinguish structurally-different CASE expressions.
        for q in [
            "RETURN CASE x WHEN 1 THEN 'a' WHEN 2 THEN 'b' ELSE 'c' END",
            "RETURN CASE WHEN x > 0 THEN 'pos' ELSE 'np' END",
            "RETURN CASE x WHEN 1 THEN 'a' END",
            "RETURN CASE WHEN x THEN 'a' END",
            "RETURN CASE x + 1 WHEN 2 THEN 10 * 2 ELSE 0 END",
        ] {
            let e = return_expr(q);
            let rendered = format!("{e}");
            let reparsed = return_expr(&format!("RETURN {rendered}"));
            assert_eq!(e, reparsed, "round-trip failed for `{rendered}`");
        }
    }

    #[test]
    fn lp_parse_all_shape() {
        let e = return_expr("MATCH (n) RETURN all(x IN [1, 2, 3] WHERE x > 0)");
        match e {
            Expression::ListPredicate {
                quantifier,
                var,
                list,
                predicate,
            } => {
                assert_eq!(quantifier, Quantifier::All);
                assert_eq!(var, "x");
                assert!(matches!(*list, Expression::Literal(Literal::List(_))));
                assert!(matches!(
                    *predicate,
                    Expression::BinaryOp { op: BinOp::Gt, .. }
                ));
            }
            other => panic!("expected ListPredicate, got {other:?}"),
        }
    }

    #[test]
    fn lp_parse_each_quantifier() {
        for (kw, want) in [
            ("all", Quantifier::All),
            ("any", Quantifier::Any),
            ("none", Quantifier::None),
            ("single", Quantifier::Single),
        ] {
            let q = format!("MATCH (n) RETURN {kw}(x IN [1] WHERE x = 1)");
            match return_expr(&q) {
                Expression::ListPredicate { quantifier, .. } => {
                    assert_eq!(quantifier, want, "quantifier for `{kw}`");
                }
                other => panic!("expected ListPredicate for `{kw}`, got {other:?}"),
            }
        }
    }

    #[test]
    fn lp_parse_case_insensitive_keyword() {
        // `ALL(...)` / `Any(...)` parse the same as lowercase (keyword
        // wrappers are case-insensitive `^"…"`).
        for kw in ["ALL", "Any", "NONE", "Single"] {
            let q = format!("MATCH (n) RETURN {kw}(x IN [1] WHERE x = 1)");
            assert!(
                matches!(return_expr(&q), Expression::ListPredicate { .. }),
                "case-insensitive `{kw}(` MUST parse as ListPredicate"
            );
        }
    }

    #[test]
    fn lp_parse_reduce_shape() {
        let e = return_expr("MATCH (n) RETURN reduce(s = 0, x IN [1, 2, 3] | s + x)");
        match e {
            Expression::Reduce {
                acc_var,
                init,
                var,
                list,
                expr,
            } => {
                assert_eq!(acc_var, "s");
                assert_eq!(var, "x");
                assert!(matches!(*init, Expression::Literal(Literal::Integer(0))));
                assert!(matches!(*list, Expression::Literal(Literal::List(_))));
                assert!(matches!(*expr, Expression::BinaryOp { op: BinOp::Add, .. }));
            }
            other => panic!("expected Reduce, got {other:?}"),
        }
    }

    #[test]
    fn lp_parse_all_without_where_rejected() {
        // The four quantifiers REQUIRE a WHERE (Decision 4 tables key on
        // the predicate); a bare `all(x IN l)` is rejected at parse time
        // (no silent default to `true`).
        assert!(
            parse("MATCH (n) RETURN all(x IN [1, 2, 3])").is_err(),
            "all(x IN list) without WHERE MUST fail to parse"
        );
        assert!(parse("MATCH (n) RETURN single(x IN [1])").is_err());
    }

    #[test]
    fn lp_parse_function_call_still_works() {
        // A regular function call `size([1,2,3])` must STILL parse as a
        // FunctionCall (the special-form productions are tried first but
        // PEG backtracks when the `IN` doesn't follow).
        let e = return_expr("MATCH (n) RETURN size([1, 2, 3])");
        assert!(
            matches!(e, Expression::FunctionCall { ref name, .. } if name == "size"),
            "size(...) MUST still parse as a FunctionCall, got {e:?}"
        );
    }

    // ----------------------------------------------------------------
    // #773 G4/G5 — count(*) (star_arg) + count/collect(DISTINCT x)
    // (distinct_arg) aggregate-argument forms (openCypher v9 §3).
    // ----------------------------------------------------------------

    #[test]
    fn cz773_parse_count_star() {
        // `count(*)` → star=true, no args, distinct=false.
        let e = return_expr("MATCH (i) RETURN count(*)");
        match e {
            Expression::FunctionCall {
                name,
                args,
                distinct,
                star,
            } => {
                assert_eq!(name, "count");
                assert!(star, "count(*) MUST set star");
                assert!(!distinct);
                assert!(args.is_empty(), "count(*) has no expression args");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn cz773_parse_count_distinct() {
        // `count(DISTINCT a.country)` → distinct=true, args=[a.country],
        // star=false.
        let e = return_expr("MATCH (a) RETURN count(DISTINCT a.country)");
        match e {
            Expression::FunctionCall {
                name,
                args,
                distinct,
                star,
            } => {
                assert_eq!(name, "count");
                assert!(distinct, "count(DISTINCT x) MUST set distinct");
                assert!(!star);
                assert_eq!(args.len(), 1, "the single DISTINCT arg is preserved");
            }
            other => panic!("expected FunctionCall, got {other:?}"),
        }
    }

    #[test]
    fn cz773_parse_collect_distinct() {
        // `collect(DISTINCT x)` → distinct=true.
        let e = return_expr("MATCH (a) RETURN collect(DISTINCT a.country)");
        assert!(
            matches!(
                e,
                Expression::FunctionCall { ref name, distinct: true, star: false, ref args }
                    if name == "collect" && args.len() == 1
            ),
            "collect(DISTINCT x) MUST set distinct, got {e:?}"
        );
    }

    #[test]
    fn cz773_parse_count_var_and_multiarg_unchanged() {
        // Regression: `count(i)` (var arg), `fn(a, b, c)` (multi-arg
        // list), and `range(1, 3)` keep star=false + distinct=false.
        let count_var = return_expr("MATCH (i) RETURN count(i)");
        assert!(matches!(
            count_var,
            Expression::FunctionCall { ref name, distinct: false, star: false, ref args }
                if name == "count" && args.len() == 1
        ));
        let multi = return_expr("MATCH (n) RETURN range(1, 3)");
        assert!(matches!(
            multi,
            Expression::FunctionCall { ref name, distinct: false, star: false, ref args }
                if name == "range" && args.len() == 2
        ));
        // `size([1,2,3])` (single composite arg) unchanged.
        let size = return_expr("MATCH (n) RETURN size([1, 2, 3])");
        assert!(matches!(
            size,
            Expression::FunctionCall {
                distinct: false,
                star: false,
                ..
            }
        ));
    }

    #[test]
    fn cz773_count_star_distinct_display_round_trips() {
        // The plan-cache key is this Display rendering, so `count(*)` /
        // `count(DISTINCT x)` MUST round-trip parse→print→parse stably
        // (and distinctly from `count(x)`).
        for query in [
            "MATCH (i) RETURN count(*)",
            "MATCH (a) RETURN count(DISTINCT a.country)",
            "MATCH (a) RETURN collect(DISTINCT a.country)",
            "MATCH (i) RETURN count(i)",
        ] {
            let e = return_expr(query);
            let printed = format!("{e}");
            let reparsed = return_expr(&format!("MATCH (n) RETURN {printed}"));
            assert_eq!(
                e, reparsed,
                "Display round-trip MUST be stable for `{query}` (printed `{printed}`)"
            );
        }
        // The three forms print distinctly (cache-key discrimination).
        assert_eq!(
            format!("{}", return_expr("MATCH (i) RETURN count(*)")),
            "count(*)"
        );
        assert_eq!(
            format!(
                "{}",
                return_expr("MATCH (a) RETURN count(DISTINCT a.country)")
            ),
            "count(DISTINCT a.country)"
        );
        assert_eq!(
            format!("{}", return_expr("MATCH (i) RETURN count(i)")),
            "count(i)"
        );
    }

    #[test]
    fn lp_display_round_trips() {
        // `parse(format!("{e}")) == e` for each special form — the
        // round-trip property (parallel to the temporal-literal
        // round-trip the grammar proptest pins).
        for query in [
            "MATCH (n) RETURN all(x IN [1, 2, 3] WHERE (x > 0))",
            "MATCH (n) RETURN any(y IN [4, 5] WHERE (y = 5))",
            "MATCH (n) RETURN none(z IN [6] WHERE (z < 0))",
            "MATCH (n) RETURN single(w IN [7, 8] WHERE (w = 7))",
            "MATCH (n) RETURN reduce(s = 0, x IN [1, 2, 3] | (s + x))",
        ] {
            let e = return_expr(query);
            let printed = format!("{e}");
            // Re-parse the printed expression (wrapped in a RETURN).
            let reparsed = return_expr(&format!("MATCH (n) RETURN {printed}"));
            assert_eq!(
                e, reparsed,
                "Display round-trip MUST be stable for `{query}` (printed: `{printed}`)"
            );
        }
    }

    #[test]
    fn lp_parse_nested_all_inside_any() {
        // Nested special forms parse: the inner `all` is the predicate
        // of the outer `any`.
        let e = return_expr("MATCH (n) RETURN any(x IN [1, 2] WHERE all(y IN [3, 4] WHERE y > x))");
        match e {
            Expression::ListPredicate {
                quantifier: Quantifier::Any,
                predicate,
                ..
            } => {
                assert!(
                    matches!(
                        *predicate,
                        Expression::ListPredicate {
                            quantifier: Quantifier::All,
                            ..
                        }
                    ),
                    "inner predicate MUST be an `all` ListPredicate"
                );
            }
            other => panic!("expected outer Any ListPredicate, got {other:?}"),
        }
    }

    // ----------------------------------------------------------------
    // ADR-188 (#620 list-half) — list-comprehension parser tests.
    // ----------------------------------------------------------------

    #[test]
    fn lc_parse_filter_then_map() {
        // `[x IN [1,2,3] WHERE x > 1 | x * 10]` — full form: both WHERE
        // and `| projection` present.
        let e = return_expr("MATCH (n) RETURN [x IN [1, 2, 3] WHERE x > 1 | x * 10]");
        match e {
            Expression::ListComprehension {
                var,
                list,
                predicate,
                projection,
            } => {
                assert_eq!(var, "x");
                assert!(matches!(*list, Expression::Literal(Literal::List(_))));
                assert!(matches!(
                    predicate.as_deref(),
                    Some(Expression::BinaryOp { op: BinOp::Gt, .. })
                ));
                assert!(matches!(
                    projection.as_deref(),
                    Some(Expression::BinaryOp { op: BinOp::Mul, .. })
                ));
            }
            other => panic!("expected ListComprehension, got {other:?}"),
        }
    }

    #[test]
    fn lc_parse_map_only_no_where() {
        // `[x IN [1,2,3] | x * 10]` — no WHERE, projection present.
        let e = return_expr("MATCH (n) RETURN [x IN [1, 2, 3] | x * 10]");
        match e {
            Expression::ListComprehension {
                predicate,
                projection,
                ..
            } => {
                assert!(predicate.is_none(), "no WHERE ⇒ predicate None");
                assert!(
                    matches!(
                        projection.as_deref(),
                        Some(Expression::BinaryOp { op: BinOp::Mul, .. })
                    ),
                    "projection MUST be present"
                );
            }
            other => panic!("expected ListComprehension, got {other:?}"),
        }
    }

    #[test]
    fn lc_parse_filter_only_no_projection() {
        // `[x IN [1,2,3] WHERE x > 1]` — WHERE present, no `| projection`
        // (identity).
        let e = return_expr("MATCH (n) RETURN [x IN [1, 2, 3] WHERE x > 1]");
        match e {
            Expression::ListComprehension {
                predicate,
                projection,
                ..
            } => {
                assert!(
                    matches!(
                        predicate.as_deref(),
                        Some(Expression::BinaryOp { op: BinOp::Gt, .. })
                    ),
                    "predicate MUST be present"
                );
                assert!(projection.is_none(), "no `|` ⇒ projection None (identity)");
            }
            other => panic!("expected ListComprehension, got {other:?}"),
        }
    }

    #[test]
    fn lc_parse_identity_no_where_no_projection() {
        // `[x IN [1,2,3]]` — bareword identity comprehension (neither
        // WHERE nor `|`). MUST parse as ListComprehension, NOT a list
        // literal (the `x IN` prefix disambiguates).
        let e = return_expr("MATCH (n) RETURN [x IN [1, 2, 3]]");
        match e {
            Expression::ListComprehension {
                var,
                predicate,
                projection,
                ..
            } => {
                assert_eq!(var, "x");
                assert!(predicate.is_none());
                assert!(projection.is_none());
            }
            other => panic!("expected ListComprehension, got {other:?}"),
        }
    }

    #[test]
    fn lc_parse_list_literal_still_works() {
        // CRITICAL disambiguation: a plain `[1, 2, 3]` MUST still parse
        // as a list LITERAL (the comprehension production is tried first
        // but PEG backtracks when `identifier ~ IN` does not follow the
        // `[`).
        let e = return_expr("MATCH (n) RETURN [1, 2, 3]");
        assert!(
            matches!(e, Expression::Literal(Literal::List(_))),
            "[1,2,3] MUST still parse as a list literal, got {e:?}"
        );
    }

    #[test]
    fn lc_parse_list_of_var_refs_still_a_literal() {
        // A list literal of bare identifiers `[a, b, c]` must STILL be a
        // list literal — there is no `IN` after the first identifier, so
        // the comprehension production backtracks.
        let e = return_expr("MATCH (n) RETURN [a, b, c]");
        assert!(
            matches!(e, Expression::Literal(Literal::List(_))),
            "[a, b, c] MUST still parse as a list literal, got {e:?}"
        );
    }

    #[test]
    fn lc_parse_inner_list_literal_as_source() {
        // `[x IN [1,2,3] | x]` — the inner `[1,2,3]` is the SOURCE list
        // (a list literal); the outer `[...]` is the comprehension. This
        // exercises the comprehension-wrapping-a-list-literal nesting.
        let e = return_expr("MATCH (n) RETURN [x IN [1, 2, 3] | x]");
        match e {
            Expression::ListComprehension { list, .. } => {
                assert!(
                    matches!(*list, Expression::Literal(Literal::List(_))),
                    "source list MUST be the inner list literal"
                );
            }
            other => panic!("expected ListComprehension, got {other:?}"),
        }
    }

    #[test]
    fn lc_parse_nested_comprehension() {
        // `[x IN [1,2] | [y IN [3,4] | x + y]]` — the projection of the
        // outer comprehension is an INNER comprehension.
        let e = return_expr("MATCH (n) RETURN [x IN [1, 2] | [y IN [3, 4] | x + y]]");
        match e {
            Expression::ListComprehension {
                var, projection, ..
            } => {
                assert_eq!(var, "x");
                assert!(
                    matches!(
                        projection.as_deref(),
                        Some(Expression::ListComprehension { .. })
                    ),
                    "outer projection MUST be an inner ListComprehension"
                );
            }
            other => panic!("expected outer ListComprehension, got {other:?}"),
        }
    }

    #[test]
    fn lc_display_round_trips() {
        // `parse(format!("{e}")) == e` for each of the four §3.5
        // combinations (the round-trip property the grammar proptest
        // pins).
        for query in [
            "MATCH (n) RETURN [x IN [1, 2, 3] WHERE (x > 1) | (x * 10)]",
            "MATCH (n) RETURN [x IN [1, 2, 3] | (x * 10)]",
            "MATCH (n) RETURN [x IN [1, 2, 3] WHERE (x > 1)]",
            "MATCH (n) RETURN [x IN [1, 2, 3]]",
            "MATCH (n) RETURN [x IN [1, 2] | [y IN [3, 4] | (x + y)]]",
        ] {
            let e = return_expr(query);
            let printed = format!("{e}");
            let reparsed = return_expr(&format!("MATCH (n) RETURN {printed}"));
            assert_eq!(
                e, reparsed,
                "Display round-trip MUST be stable for `{query}` (printed: `{printed}`)"
            );
        }
    }

    // ----------------------------------------------------------------
    // ADR-191 D-6 (#620 map-half) — map-projection parser tests.
    // ----------------------------------------------------------------

    #[test]
    fn mp_parse_property_selectors() {
        // `n{.name, .age}` — two `.key` property selectors.
        let e = return_expr("MATCH (n) RETURN n{.name, .age}");
        match e {
            Expression::MapProjection { base, items } => {
                assert_eq!(base, "n");
                assert_eq!(
                    items,
                    vec![
                        MapProjectionItem::Property("name".into()),
                        MapProjectionItem::Property("age".into()),
                    ]
                );
            }
            other => panic!("expected MapProjection, got {other:?}"),
        }
    }

    #[test]
    fn mp_parse_literal_entry_with_expression() {
        // `n{.name, alias: 1 + 1}` — a `.key` selector + an `alias: expr`
        // literal entry (the value is a full expression).
        let e = return_expr("MATCH (n) RETURN n{.name, alias: 1 + 1}");
        match e {
            Expression::MapProjection { base, items } => {
                assert_eq!(base, "n");
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], MapProjectionItem::Property("name".into()));
                match &items[1] {
                    MapProjectionItem::Literal { alias, value } => {
                        assert_eq!(alias, "alias");
                        assert!(matches!(
                            **value,
                            Expression::BinaryOp { op: BinOp::Add, .. }
                        ));
                    }
                    other => panic!("expected Literal entry, got {other:?}"),
                }
            }
            other => panic!("expected MapProjection, got {other:?}"),
        }
    }

    #[test]
    fn mp_parse_single_missing_selector() {
        // `n{.missing}` — single property selector (D-6 null-drop is an
        // EXECUTOR concern; the parser just records the selector).
        let e = return_expr("MATCH (n) RETURN n{.missing}");
        match e {
            Expression::MapProjection { base, items } => {
                assert_eq!(base, "n");
                assert_eq!(items, vec![MapProjectionItem::Property("missing".into())]);
            }
            other => panic!("expected MapProjection, got {other:?}"),
        }
    }

    #[test]
    fn mp_parse_all_properties_selector() {
        // `n{.*}` — the all-properties selector.
        let e = return_expr("MATCH (n) RETURN n{.*}");
        match e {
            Expression::MapProjection { base, items } => {
                assert_eq!(base, "n");
                assert_eq!(items, vec![MapProjectionItem::AllProperties]);
            }
            other => panic!("expected MapProjection, got {other:?}"),
        }
    }

    #[test]
    fn mp_parse_mixed_all_props_and_entry() {
        // `n{.*, score: 99}` — `.*` followed by a literal entry.
        let e = return_expr("MATCH (n) RETURN n{.*, score: 99}");
        match e {
            Expression::MapProjection { items, .. } => {
                assert_eq!(items.len(), 2);
                assert_eq!(items[0], MapProjectionItem::AllProperties);
                assert!(matches!(items[1], MapProjectionItem::Literal { .. }));
            }
            other => panic!("expected MapProjection, got {other:?}"),
        }
    }

    #[test]
    fn mp_parse_empty_projection() {
        // `n{}` — an empty projection (the empty map). Distinct from the
        // bare `{}` map literal (which has no leading identifier).
        let e = return_expr("MATCH (n) RETURN n{}");
        match e {
            Expression::MapProjection { base, items } => {
                assert_eq!(base, "n");
                assert!(items.is_empty());
            }
            other => panic!("expected MapProjection, got {other:?}"),
        }
    }

    #[test]
    fn mp_parse_whitespace_between_var_and_brace() {
        // openCypher's `MapProjection = Variable { SP } '{'` admits
        // whitespace between the base and the `{` (`n {.name}`).
        let e = return_expr("MATCH (n) RETURN n {.name}");
        assert!(
            matches!(e, Expression::MapProjection { .. }),
            "`n {{.name}}` (space before brace) MUST parse as a map projection, got {e:?}"
        );
    }

    #[test]
    fn mp_parse_does_not_break_map_literal() {
        // CRITICAL disambiguation: a bare `{a: 1}` (no leading identifier)
        // MUST still parse as a map LITERAL — the map_projection production
        // requires `identifier ~ {` and backtracks on a bare `{`.
        let e = return_expr("MATCH (n) RETURN {a: 1, b: 2}");
        assert!(
            matches!(e, Expression::Literal(Literal::Map(_))),
            "bare `{{a:1, b:2}}` MUST still parse as a map literal, got {e:?}"
        );
    }

    #[test]
    fn mp_parse_does_not_break_function_call() {
        // CRITICAL disambiguation: a function call `count(n)` MUST still
        // parse as a FunctionCall — map_projection and function_call both
        // open with an identifier but commit on `{` vs `(`.
        let e = return_expr("MATCH (n) RETURN count(n)");
        assert!(
            matches!(e, Expression::FunctionCall { .. }),
            "`count(n)` MUST still parse as a function call, got {e:?}"
        );
    }

    #[test]
    fn mp_parse_does_not_break_bare_identifier() {
        // A bare `n` (no `{`) MUST still parse as an Identifier — the
        // map_projection production backtracks on a non-`{` follower.
        let e = return_expr("MATCH (n) RETURN n");
        assert!(
            matches!(e, Expression::Identifier(_)),
            "bare `n` MUST still parse as an identifier, got {e:?}"
        );
    }

    #[test]
    fn mp_display_round_trips() {
        // `parse(format!("{e}")) == e` for each map-projection form.
        for query in [
            "MATCH (n) RETURN n{.name, .age}",
            "MATCH (n) RETURN n{.*}",
            "MATCH (n) RETURN n{x: 1}",
            "MATCH (n) RETURN n{.name, y: 2}",
            "MATCH (n) RETURN n{.*, score: 99}",
            "MATCH (n) RETURN n{}",
        ] {
            let e = return_expr(query);
            let printed = format!("{e}");
            let reparsed = return_expr(&format!("MATCH (n) RETURN {printed}"));
            assert_eq!(
                e, reparsed,
                "Display round-trip MUST be stable for `{query}` (printed: `{printed}`)"
            );
        }
    }

    // ----------------------------------------------------------------
    // #618 GA Lane C — openCypher v9 §3 number literals: hex / octal /
    // leading-dot float / i64::MIN. AST-shape oracles (the e2e VALUE
    // oracles live in `tests/number_literal_e2e.rs`).
    // ----------------------------------------------------------------

    #[test]
    fn hex_octal_decimal_int_ast() {
        for (q, want) in [
            ("RETURN 0x1A", 26i64),
            ("RETURN 0X1a", 26),
            ("RETURN 0xFF", 255),
            ("RETURN 0o17", 15),
            ("RETURN 0O17", 15),
            ("RETURN 0o777", 511),
            ("RETURN 0", 0),
            ("RETURN 00", 0),
            ("RETURN 10", 10),
            ("RETURN 0x7FFFFFFFFFFFFFFF", i64::MAX),
            ("RETURN 0o777777777777777777777", i64::MAX),
        ] {
            match return_expr(q) {
                Expression::Literal(Literal::Integer(n)) => {
                    assert_eq!(n, want, "`{q}` => Integer({n}), want {want}")
                }
                other => panic!("`{q}` => {other:?}, want Integer({want})"),
            }
        }
    }

    #[test]
    fn leading_dot_float_ast() {
        for (q, want) in [
            ("RETURN .5", 0.5f64),
            ("RETURN .0", 0.0),
            ("RETURN .3405892687", 0.3405892687),
            ("RETURN .1e-5", 0.000_001),
        ] {
            match return_expr(q) {
                Expression::Literal(Literal::Float(f)) => {
                    assert!((f - want).abs() < 1e-12, "`{q}` => Float({f}), want {want}")
                }
                other => panic!("`{q}` => {other:?}, want Float({want})"),
            }
        }
    }

    #[test]
    fn i64_min_folds_to_integer_literal_not_unaryop() {
        // The load-bearing fold: `-9223372036854775808` (and its hex/octal
        // spellings) cannot be `UnaryOp{Neg, Integer(2^63)}` — 2^63 is not a
        // positive i64. The parser folds the sign into the literal.
        for q in [
            "RETURN -9223372036854775808",
            "RETURN -0x8000000000000000",
            "RETURN -0o1000000000000000000000",
        ] {
            match return_expr(q) {
                Expression::Literal(Literal::Integer(n)) => {
                    assert_eq!(n, i64::MIN, "`{q}` => Integer({n}), want i64::MIN")
                }
                other => panic!("`{q}` => {other:?}, want Integer(i64::MIN)"),
            }
        }
    }

    #[test]
    fn in_range_negative_int_keeps_unaryop_shape() {
        // The fold is SURGICAL — it triggers ONLY at the i64::MIN overflow
        // boundary. Every in-range negative literal keeps the canonical
        // `UnaryOp{Neg, Integer(mag)}` AST, so no existing tree is
        // perturbed. (This is the discriminator proving the fold is not a
        // blanket constant-fold.)
        for (q, mag) in [
            ("RETURN -5", 5i64),
            ("RETURN -1", 1),
            ("RETURN -0x1", 1),
            ("RETURN -0o17", 15),
            ("RETURN -9223372036854775807", i64::MAX),
        ] {
            match return_expr(q) {
                Expression::UnaryOp {
                    op: UnaryOp::Neg,
                    operand,
                } => match *operand {
                    Expression::Literal(Literal::Integer(n)) => {
                        assert_eq!(
                            n, mag,
                            "`{q}` => UnaryOp(Neg, Integer({n})), want mag {mag}"
                        )
                    }
                    other => panic!("`{q}` operand => {other:?}, want Integer({mag})"),
                },
                other => panic!("`{q}` => {other:?}, want UnaryOp{{Neg, Integer({mag})}}"),
            }
        }
    }

    #[test]
    fn leading_dot_float_does_not_steal_dot_accessor_or_range() {
        // Property access `.prop` MUST remain an accessor (a leading-dot
        // float is `.`+DIGIT; property access is `.`+identifier).
        match return_expr("MATCH (n) RETURN n.prop") {
            Expression::PropertyAccess { path, .. } => {
                assert_eq!(path, vec!["prop".to_string()])
            }
            other => panic!("`n.prop` => {other:?}, want PropertyAccess"),
        }
        // List-slice `..` MUST remain the range token (`.`+`.`, never
        // `.`+DIGIT), not be consumed by a leading-dot float.
        match return_expr("RETURN [1, 2, 3][0..2]") {
            Expression::Slice { .. } => {}
            other => panic!("`[1,2,3][0..2]` => {other:?}, want Slice"),
        }
    }

    #[test]
    fn over_range_integer_is_parse_error() {
        // Magnitudes outside i64 reject cleanly at the AST int build — a
        // ParseError, never a panic (openCypher Literals2 [9]/[10],
        // Literals4 [9]/[10]).
        for q in [
            "RETURN 9223372036854775808",      // i64::MAX + 1
            "RETURN -9223372036854775809",     // i64::MIN - 1
            "RETURN 0xFFFFFFFFFFFFFFFF",       // u64::MAX
            "RETURN 0o1000000000000000000000", // octal too large
        ] {
            assert!(parse(q).is_err(), "`{q}` MUST be a parse error");
        }
    }

    // ─────────────────────────────────────────────────────────────
    // #819 — pre-parse depth-scan internals (direct unit coverage of
    // the security-critical conservative-counting + word-boundary
    // logic that the integration tests exercise only indirectly).
    // ─────────────────────────────────────────────────────────────

    #[test]
    fn pre_scan_counts_each_bracket_form_as_one_level() {
        // At-cap accepted; one-over rejected, for `(`, `[`, `{`.
        for (open, close) in [('(', ')'), ('[', ']'), ('{', '}')] {
            let at_cap: String = format!(
                "{}{}",
                String::from(open).repeat(MAX_EXPRESSION_DEPTH),
                String::from(close).repeat(MAX_EXPRESSION_DEPTH)
            );
            assert!(
                check_pre_parse_nesting_depth(&at_cap).is_ok(),
                "{open}×{MAX_EXPRESSION_DEPTH} must pass the pre-scan (== cap)"
            );
            let over: String = format!(
                "{}{}",
                String::from(open).repeat(MAX_EXPRESSION_DEPTH + 1),
                String::from(close).repeat(MAX_EXPRESSION_DEPTH + 1)
            );
            assert!(
                matches!(
                    check_pre_parse_nesting_depth(&over),
                    Err(ParseError::ExpressionTooDeep { .. })
                ),
                "{open}×{} must trip the pre-scan (cap+1)",
                MAX_EXPRESSION_DEPTH + 1
            );
        }
    }

    #[test]
    fn pre_scan_brackets_inside_string_are_opaque() {
        // Brackets inside a string literal do not count.
        let s = format!("'{}'", "(".repeat(MAX_EXPRESSION_DEPTH * 4));
        assert!(
            check_pre_parse_nesting_depth(&s).is_ok(),
            "brackets inside a string must not count toward depth"
        );
        // ...but a closed string followed by a real deep nest DOES count
        // (the under-count-direction safety: a closed quote must close).
        let mixed = format!(
            "'{}' {}",
            "(".repeat(MAX_EXPRESSION_DEPTH * 4),
            "(".repeat(MAX_EXPRESSION_DEPTH + 1)
        );
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&mixed),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "a real deep nest after a closed string must still trip the scan"
        );
    }

    #[test]
    fn pre_scan_backslash_escaped_quote_stays_in_string() {
        // `'\\''` — backslash escapes the quote, so the string is NOT
        // closed there; the brackets that follow are inside the string.
        let s = format!("'a\\'{}'", "(".repeat(MAX_EXPRESSION_DEPTH * 4));
        assert!(
            check_pre_parse_nesting_depth(&s).is_ok(),
            "backslash-escaped quote must keep the brackets in-string"
        );
    }

    #[test]
    fn pre_scan_case_end_keyword_word_boundary() {
        // `CASE` / `END` count only on word boundaries.
        let deep_case = "CASE ".repeat(MAX_EXPRESSION_DEPTH + 1);
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&deep_case),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "{}+ CASE keywords must trip the scan",
            MAX_EXPRESSION_DEPTH + 1
        );
        // `MYCASE` / `CASES` / `ENDPOINT` are NOT the keywords.
        let identy = "mycase casey endpoint ".repeat(MAX_EXPRESSION_DEPTH + 5);
        assert!(
            check_pre_parse_nesting_depth(&identy).is_ok(),
            "identifiers containing case/end as substrings must not count"
        );
    }

    #[test]
    fn matches_keyword_ci_boundary_and_case() {
        // case-insensitive body match
        assert!(matches_keyword_ci(b"case", 0, b"CASE"));
        assert!(matches_keyword_ci(b"CaSe", 0, b"CASE"));
        assert!(matches_keyword_ci(b"END", 0, b"END"));
        // word-boundary: left neighbor is an ident char ⇒ no match
        assert!(!matches_keyword_ci(b"mycase", 2, b"CASE"));
        // right neighbor is an ident char ⇒ no match
        assert!(!matches_keyword_ci(b"cases", 0, b"CASE"));
        assert!(!matches_keyword_ci(b"endpoint", 0, b"END"));
        // bounded by non-ident (space/paren) ⇒ match
        assert!(matches_keyword_ci(b" case ", 1, b"CASE"));
        assert!(matches_keyword_ci(b"(case)", 1, b"CASE"));
        // truncated (kw runs past end) ⇒ no match
        assert!(!matches_keyword_ci(b"cas", 0, b"CASE"));
    }

    #[test]
    fn pre_scan_unbalanced_closers_saturate_at_zero() {
        // Excess closers must not underflow / wrap; depth saturates at 0
        // so a following bracket group still counts from 0.
        let s = format!("))))){}", "(".repeat(MAX_EXPRESSION_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&s),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "leading excess closers must saturate at 0, not mask a later deep nest"
        );
    }

    // ─────────────────────────────────────────────────────────────
    // #819 R1 follow-up — pre-scan UNARY-chain counting + comment
    // opacity (family (B): `unary_expr = ("-"|"+") ~ unary_expr`, the
    // residual crash vector the bracket/CASE scan under-counted to 0).
    // ─────────────────────────────────────────────────────────────

    /// `n` alternating unary-prefix operators (`-+-+…`), avoiding the
    /// adjacent `--` that would open a line comment.
    fn unary_ops(n: usize) -> String {
        (0..n).map(|i| if i % 2 == 0 { '-' } else { '+' }).collect()
    }

    #[test]
    fn pre_scan_unary_chain_counts_each_prefix_as_one_level() {
        // A bracket-LESS unary chain is real `unary_expr` recursion: the
        // pre-scan must count it (pre-fix it scored 0 → pest SIGABRT).
        // Tested in TRUE prefix position (a bare chain, where the scan
        // starts `expecting_operand`) so the count is EXACT: at-cap (CAP
        // ops) accepted, one-over (CAP+1 ops) rejected.
        let at_cap = unary_ops(MAX_EXPRESSION_DEPTH) + "1";
        assert!(
            check_pre_parse_nesting_depth(&at_cap).is_ok(),
            "a {MAX_EXPRESSION_DEPTH}-operator unary chain (== cap) must pass the pre-scan"
        );
        let over = unary_ops(MAX_EXPRESSION_DEPTH + 1) + "1";
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&over),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "a {}-operator unary chain (cap+1) must trip the pre-scan",
            MAX_EXPRESSION_DEPTH + 1
        );
        // Pathological 8000-operator chain (~4 KB on the wire — the R1
        // repro) must reject cheaply (O(n), bails on the first over-cap
        // operator), NOT recurse. Tested in real RETURN position too.
        let huge = format!("RETURN {}1", unary_ops(8000));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&huge),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "an 8000-operator RETURN unary chain must reject cleanly"
        );
    }

    #[test]
    fn pre_scan_unary_after_keyword_counts_folded_operator_load() {
        // #1290: after a keyword consumed as identifier bytes (`RETURN`,
        // `WHEN`, ...), the first `-`/`+` is still seen by the scanner as
        // an operator. The rest of the run is counted as unary parser
        // recursion, so the old #819 one-token unary under-count no
        // longer buys extra depth budget.
        let cap_plus_two = format!("RETURN {}1", unary_ops(MAX_EXPRESSION_DEPTH + 2));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&cap_plus_two),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "CAP+2 operators after a keyword must reject once unary depth is counted"
        );
    }

    #[test]
    fn pre_scan_flat_operator_chains_count_folded_ast_depth() {
        // #1290 — these grammar forms parse as flat `*` repetitions but
        // fold into left-nested ASTs. Binding/evaluation then recurse on
        // that AST, so the pre-parse guard must reject once the folded
        // depth exceeds MAX_FLAT_CHAIN_DEPTH.
        let flat_add = format!("RETURN {}1", "1+".repeat(MAX_FLAT_CHAIN_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_add),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 infix `+` operators must count as folded BinaryOp depth"
        );
        let flat_sub = format!("RETURN 1{}", "-1".repeat(MAX_FLAT_CHAIN_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_sub),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 infix `-` operators must count as folded BinaryOp depth"
        );
        let flat_mul = format!("RETURN {}1", "1*".repeat(MAX_FLAT_CHAIN_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_mul),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 infix `*` operators must count as folded BinaryOp depth"
        );
        let flat_and = format!(
            "RETURN {}true",
            "true AND ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)
        );
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_and),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 AND operators must count as folded BinaryOp depth"
        );
        let flat_or = format!("RETURN {}true", "true OR ".repeat(MAX_FLAT_CHAIN_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_or),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 OR operators must count as folded BinaryOp depth"
        );
        let flat_xor = format!(
            "RETURN {}true",
            "true XOR ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)
        );
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_xor),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 XOR operators must count as folded BinaryOp depth"
        );
        let flat_cmp = format!("RETURN {}1", "1 < ".repeat(MAX_FLAT_CHAIN_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_cmp),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-cap+1 comparison operators must count as folded BinaryOp depth"
        );
        let flat_not = format!("RETURN {}true", "NOT ".repeat(MAX_EXPRESSION_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&flat_not),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "cap+1 NOT operators must count as folded UnaryOp depth"
        );
    }

    #[test]
    fn parse_rejects_over_cap_flat_chains_gracefully() {
        for query in [
            format!(
                "MATCH (n) WHERE {}true RETURN n",
                "true AND ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)
            ),
            format!(
                "MATCH (n) WHERE {}true RETURN n",
                "true OR ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)
            ),
            format!(
                "MATCH (n) WHERE {}true RETURN n",
                "true XOR ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)
            ),
            format!("RETURN {}1", "1 < ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)),
            format!("RETURN {}1", "1 + ".repeat(MAX_FLAT_CHAIN_DEPTH + 1)),
        ] {
            assert!(
                matches!(parse(&query), Err(ParseError::ExpressionTooDeep { .. })),
                "`{}` must reject at parse time with ExpressionTooDeep",
                &query[..query.len().min(80)]
            );
        }
    }

    #[test]
    fn normal_depth_flat_chain_still_parses_and_binds() {
        let mut query = String::from("MATCH (n) WHERE ");
        for i in 1..=500 {
            if i > 1 {
                query.push_str(" AND ");
            }
            query.push_str(&format!("n.p{i} = {i}"));
        }
        query.push_str(" RETURN n");
        let stmt = parse(&query).expect("normal-depth flat chain parses");
        let catalog = crate::semantic::StubCatalogProvider::new();
        crate::semantic::BindingVisitor::bind(&stmt, &query, &catalog)
            .expect("normal-depth flat chain binds");
    }

    #[test]
    fn pre_scan_independent_unary_operands_discharge_and_do_not_accumulate() {
        // `RETURN -1, -1, …` — 1000 INDEPENDENT unary operands separated
        // by commas. Each `-` discharges at the next separator, so the
        // running depth stays 1 (NOT 1000). Pins the discharge logic that
        // prevents false-rejecting many sibling negatives.
        let many = format!("RETURN {}-1", "-1,".repeat(1000));
        assert!(
            check_pre_parse_nesting_depth(&many).is_ok(),
            "1000 comma-separated `-1` operands must not accumulate depth"
        );
        // `-(1) + -(1) + …` — #1290 counts the infix `+` chain because
        // the parser folds it into left-nested BinaryOp nodes. A small
        // chain still passes.
        let wrapped = format!("RETURN {}-(1)", "-(1)+".repeat(10));
        assert!(
            check_pre_parse_nesting_depth(&wrapped).is_ok(),
            "normal-depth repeated `-(1)+` must pass the pre-scan"
        );
        let wrapped_over_cap = format!("RETURN {}-(1)", "-(1)+".repeat(MAX_FLAT_CHAIN_DEPTH + 1));
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&wrapped_over_cap),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "flat-over-cap repeated `-(1)+` must reject as folded BinaryOp depth"
        );
    }

    #[test]
    fn pre_scan_unary_frames_stack_additively_with_brackets() {
        // THE anti-bypass case: unary frames PERSIST across an enclosing
        // bracket, so they stack with bracket depth. `-+(` per level is
        // depth +3 (2 unary + 1 bracket). 22 levels = 22 brackets (well
        // UNDER the 64 bracket cap — a separate bracket-only cap would
        // ACCEPT it) but total depth 66 > cap → must REJECT. If the scan
        // reset the unary run at each `(` (under-count), this would slip
        // through to pest and SIGABRT.
        let interleaved = "RETURN ".to_string() + &"-+(".repeat(22) + "1";
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&interleaved),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "22 levels of `-+(` (only 22 brackets, but depth 66) must reject — \
             unary frames stack on bracket depth"
        );
        // Just under: 21 levels = depth 63 ≤ cap → accepted.
        let under = "RETURN ".to_string() + &"-+(".repeat(21) + "1" + &")".repeat(21);
        assert!(
            check_pre_parse_nesting_depth(&under).is_ok(),
            "21 levels of `-+(` (depth 63) must pass the pre-scan"
        );
    }

    #[test]
    fn pre_scan_comment_contents_are_opaque() {
        // pest strips comments, so brackets / `+` / `CASE` inside a `--`
        // line comment or `/* */` block comment must NOT count (else a
        // valid query with a bracket-/operator-heavy comment would
        // false-reject). This was a latent gap the unary counter makes
        // load-bearing (`-- … +++++` would otherwise be counted).
        let line = format!(
            "RETURN 1 -- {} {} CASE CASE",
            "(".repeat(MAX_EXPRESSION_DEPTH * 4),
            unary_ops(MAX_EXPRESSION_DEPTH * 4)
        );
        assert!(
            check_pre_parse_nesting_depth(&line).is_ok(),
            "brackets / unary / CASE inside a `--` line comment must not count"
        );
        let block = format!(
            "RETURN 1 /* {} {} CASE */ + 2",
            "(".repeat(MAX_EXPRESSION_DEPTH * 4),
            unary_ops(MAX_EXPRESSION_DEPTH * 4)
        );
        assert!(
            check_pre_parse_nesting_depth(&block).is_ok(),
            "brackets / unary / CASE inside a `/* */` block comment must not count"
        );
        // A real deep nest AFTER a closed comment still counts (the
        // under-count-direction safety: a comment must not swallow the
        // rest of the line past its newline / `*/`).
        let after = format!(
            "RETURN 1 /* harmless */ + {}1{}",
            "(".repeat(MAX_EXPRESSION_DEPTH + 1),
            ")".repeat(MAX_EXPRESSION_DEPTH + 1)
        );
        assert!(
            matches!(
                check_pre_parse_nesting_depth(&after),
                Err(ParseError::ExpressionTooDeep { .. })
            ),
            "a real deep nest after a closed block comment must still trip the scan"
        );
    }

    #[test]
    fn pre_scan_adjacent_double_dash_is_comment_not_unary() {
        // `--` (adjacent) opens a line comment, so a run of dashes is a
        // comment, NOT a unary chain — must not count, must not be
        // mistaken for many unary minuses.
        let dashes = format!("RETURN 1 {}", "-".repeat(MAX_EXPRESSION_DEPTH * 8));
        assert!(
            check_pre_parse_nesting_depth(&dashes).is_ok(),
            "a run of adjacent dashes is a `--` comment, not a counted unary chain"
        );
    }
}
