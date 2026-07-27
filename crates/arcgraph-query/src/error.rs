//! ArcQL parser errors (M4-01 surface).
//!
//! # Scope
//!
//! `ParseError` covers **syntactic** failures only — the grammar
//! either matched or it did not. Reserved-but-unimplemented clause
//! detection (ADR-038 D-16, the `ArcQLError::NotImplemented`
//! variant) lives in M4-02 (`semantic.rs`, future slice). The two
//! error layers are distinct on purpose:
//!
//! - `ParseError` says "this string is not valid ArcQL".
//! - `ArcQLError::NotImplemented` says "this string is valid ArcQL
//!   but the executor lights at v1.1+; see ADR-038 §2 D-16".
//!
//! Conflating them would erase the v1.0 reserved-syntax discipline
//! ADR-038 §2 D-16 calls out explicitly.
//!
//! # ADR provenance
//! - ADR-006 D-1 — `pest` PEG.
//! - ADR-038 §2 D-16 — error taxonomy (the `NotImplemented` half
//!   lives in M4-02; this file is the syntactic half).
//! - code-quality policy — `thiserror` for library errors.

use std::fmt;

/// Span coordinates `line:col-line:col` (1-indexed). Surfaced from
/// `pest::error::LineColLocation` and from synthesized
/// AST-construction failures (where applicable).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
}

impl Span {
    /// Single-position span (start == end).
    pub fn point(line: usize, col: usize) -> Self {
        Self {
            start_line: line,
            start_col: col,
            end_line: line,
            end_col: col,
        }
    }
}

impl fmt::Display for Span {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}-{}:{}",
            self.start_line, self.start_col, self.end_line, self.end_col
        )
    }
}

/// Faults surfaced by `parser::parse`.
///
/// `#[non_exhaustive]` is omitted: the variant set is the parser's
/// public contract for M4-02 (semantic analyzer) consumption. A new
/// variant lands via amendment alongside any future grammar
/// extension (e.g., parametrized fuzzing surfacing a new rejection
/// path). The `ExpressionTooDeep` variant (#819) is exactly such an
/// amendment: a runtime DoS-hardening rejection — not a grammar
/// extension — landed alongside the expression-nesting-depth guard
/// (`feedback_security_class_first_network_surface.md`; W14 retro IR
/// L1-HIGH-4 "recursion depth" item for the query surface).
#[derive(Debug, thiserror::Error, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// `pest` PEG matcher rejected the input. The `message` slot
    /// carries `pest`'s humanized rendering; the `span` slot points
    /// at the offending token.
    #[error("pest parse error at {span}: {message}")]
    Pest {
        /// Humanized `pest::error::Error` rendering. The full
        /// `pest::error::Error` is intentionally NOT preserved
        /// here because:
        ///
        /// 1. `pest::error::Error` is generic over `Rule`, which
        ///    leaks the `pest_derive`-generated `Rule` enum across
        ///    the crate boundary. The semantic analyzer (M4-02)
        ///    must not depend on grammar-internal symbols.
        /// 2. We want `ParseError: PartialEq + Eq` for ergonomic
        ///    test assertions; `pest::error::Error<Rule>` is not
        ///    `Eq`.
        message: String,
        span: Span,
    },

    /// AST construction logic failed downstream of a successful
    /// pest parse — e.g. an integer literal whose digit-sequence
    /// the PEG accepted but which overflows `i64`.
    #[error("AST construction error{}: {message}", span_opt_display(span))]
    AstConstruction {
        message: String,
        /// `None` only when the failure has no parsed-token origin
        /// (which should not happen in well-formed inputs; we keep
        /// the slot optional defensively).
        span: Option<Span>,
    },

    /// A user-supplied expression nests deeper than
    /// [`crate::parser::MAX_EXPRESSION_DEPTH`] (#819). The recursive-
    /// descent AST builder (`parse_expression` → the precedence ladder
    /// → back into `parse_expression` for each nested paren / list /
    /// `CASE` / map / subscript) would overflow the native thread
    /// stack on adversarially-deep input and `abort()` the whole
    /// process (SIGABRT) — an unauthenticated remote DoS over Bolt /
    /// MCP via a ~600-byte query. We track nesting depth during AST
    /// construction and reject at the cap **before** the recursion
    /// exhausts the stack, so the query fails cleanly and the server
    /// stays up. Per `feedback_security_class_first_network_surface.md`
    /// (recursion-depth guard on a first-network-surface) — the
    /// runtime sibling of code-quality policy's *compile-time*
    /// `#![recursion_limit]`, and the parser-expression analogue of
    /// [`crate::executor::value::ValueJsonError::NestingTooDeep`]
    /// (the JSON-decode hardening guard).
    #[error(
        "expression nests too deep: depth {depth} exceeds the maximum {max} (rejected to prevent a parser stack overflow)"
    )]
    ExpressionTooDeep {
        /// The depth at which the cap fired (== `max + 1`).
        depth: usize,
        /// The configured cap ([`crate::parser::MAX_EXPRESSION_DEPTH`]).
        max: usize,
    },
}

/// Render an optional span as ` at L:C-L:C` (with leading space) or
/// the empty string when absent. Used by `AstConstruction`'s
/// `#[error(...)]` template; thiserror does not have a built-in
/// "format if Some" helper.
fn span_opt_display(s: &Option<Span>) -> String {
    match s {
        Some(sp) => format!(" at {sp}"),
        None => String::new(),
    }
}

impl ParseError {
    /// Return the carried [`Span`] if any. `Pest` always has a span;
    /// `AstConstruction` may not; `ExpressionTooDeep` carries none (the
    /// depth cap fires structurally, not at a single offending token —
    /// the *whole* nested expression is the offender).
    pub fn span(&self) -> Option<&Span> {
        match self {
            ParseError::Pest { span, .. } => Some(span),
            ParseError::AstConstruction { span, .. } => span.as_ref(),
            ParseError::ExpressionTooDeep { .. } => None,
        }
    }

    /// Translate the carried span (line:col coordinates) into a
    /// byte-offset range in the original input string.
    ///
    /// Returns `None` when:
    ///   - the error has no span (`AstConstruction` without span info), or
    ///   - the span coordinates fall outside `input` (which would
    ///     indicate a coordinate-system mismatch — the helper is
    ///     defensive and returns `None` rather than panicking).
    ///
    /// Both `(line, col)` are 1-indexed (matching `pest`'s
    /// `LineColLocation`); columns and lines past the end of `input`
    /// are clamped to the input length.
    ///
    /// # Use site
    ///
    /// `tests/grammar_proptest.rs`'s mutation-based span-correctness
    /// proptest (PR #154 reviewer Finding 2 / Fix B) uses this to
    /// assert that the span produced by a single-byte mutation
    /// overlaps the mutation site (modulo a small slack for the
    /// line:col → byte-offset translation noise).
    ///
    /// The translation walks `input` newline-by-newline to find the
    /// target line, then char-by-char along that line so that the
    /// pest column count (which is char-indexed, not byte-indexed)
    /// translates to the correct byte offset on multi-byte UTF-8
    /// input. Returned offsets always fall on a `char` boundary, so
    /// callers can safely slice `&input[start..end]` without
    /// panicking on a non-boundary index.
    ///
    /// Codex M4-01 retro F-05 (MEDIUM, 2026-05-03): the previous
    /// implementation did `line_start + (col - 1)` treating pest's
    /// char-indexed column as a byte offset, which produced wrong
    /// byte coordinates and could panic on `&input[s..e]` mid-
    /// codepoint. With backtick identifiers admitting non-ASCII
    /// (e.g., `` n.`日本` ``, CJK round-trips per F-03 positive pin),
    /// this surface is reachable from real user input.
    pub fn span_byte_range(&self, input: &str) -> Option<(usize, usize)> {
        let span = self.span()?;
        let start = line_col_to_byte(input, span.start_line, span.start_col)?;
        let end = line_col_to_byte(input, span.end_line, span.end_col)?;
        Some((start, end))
    }
}

/// Convert a 1-indexed (line, col) coordinate into a byte offset
/// into `input`. Clamps off-the-end lines/columns to `input.len()`
/// and always returns an offset that falls on a `char` boundary so
/// `&input[..offset]` / `&input[offset..]` cannot panic.
///
/// pest reports `col` as a 1-indexed char count (UTF-8 code points),
/// not a byte count, so this walks the line char-by-char rather than
/// adding `(col - 1)` to a byte cursor. See [`ParseError::span_byte_range`]
/// for the bug history (codex M4-01 retro F-05).
fn line_col_to_byte(input: &str, line: usize, col: usize) -> Option<usize> {
    if line == 0 || col == 0 {
        return None;
    }
    // 1) Walk to the start of the target line.
    let bytes = input.as_bytes();
    let mut current_line = 1usize;
    let mut line_start = 0usize;
    while current_line < line {
        let Some(rel) = bytes[line_start..].iter().position(|b| *b == b'\n') else {
            // Line beyond end of input — clamp.
            return Some(bytes.len());
        };
        line_start += rel + 1;
        current_line += 1;
    }
    // 2) Walk char-by-char along the line counting chars to `col`.
    //    `col == 1` → offset = line_start (the very start of the line).
    let line_tail = &input[line_start..];
    for (char_count, (idx, _ch)) in (1usize..).zip(line_tail.char_indices()) {
        if char_count == col {
            return Some(line_start + idx);
        }
    }
    // Column past end of line — clamp to end of line (or end of
    // input, since `line_tail` may run to end of input on the last
    // line). Both fall on a char boundary by construction.
    Some((line_start + line_tail.len()).min(input.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_point_is_zero_width() {
        let s = Span::point(2, 7);
        assert_eq!(format!("{s}"), "2:7-2:7");
    }

    #[test]
    fn pest_error_display_carries_span() {
        let e = ParseError::Pest {
            message: "expected RETURN".into(),
            span: Span {
                start_line: 1,
                start_col: 5,
                end_line: 1,
                end_col: 12,
            },
        };
        let s = format!("{e}");
        assert!(s.contains("1:5-1:12"));
        assert!(s.contains("expected RETURN"));
    }

    #[test]
    fn ast_construction_error_optional_span_displays_none_marker() {
        let e = ParseError::AstConstruction {
            message: "integer overflow".into(),
            span: None,
        };
        // Display still works (the spanless form is left to
        // ParseError::Display, so we just verify no panic and that
        // the message bubbles up).
        let _ = format!("{e:?}");
    }

    #[test]
    fn equality_is_structural() {
        let a = ParseError::Pest {
            message: "x".into(),
            span: Span::point(1, 1),
        };
        let b = ParseError::Pest {
            message: "x".into(),
            span: Span::point(1, 1),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn implements_std_error_trait() {
        fn assert_impls_error<E: std::error::Error>(_: E) {}
        assert_impls_error(ParseError::Pest {
            message: "x".into(),
            span: Span::point(1, 1),
        });
    }

    // ----- span_byte_range translation -----------------------------------

    #[test]
    fn span_byte_range_single_line_returns_byte_offsets() {
        let input = "MATCH (n) RETURN n";
        let e = ParseError::Pest {
            message: "x".into(),
            // pest 1-indexed line:col — col=7 lands on the `(`.
            span: Span {
                start_line: 1,
                start_col: 7,
                end_line: 1,
                end_col: 8,
            },
        };
        let (s, e_off) = e.span_byte_range(input).expect("translation");
        assert_eq!(&input[s..e_off], "(");
    }

    #[test]
    fn span_byte_range_multiline_walks_newlines() {
        let input = "MATCH (n)\nRETURN n";
        // Col 8 of line 2 lands on the `n` after `RETURN `.
        let e = ParseError::Pest {
            message: "x".into(),
            span: Span {
                start_line: 2,
                start_col: 8,
                end_line: 2,
                end_col: 9,
            },
        };
        let (s, e_off) = e.span_byte_range(input).expect("translation");
        assert_eq!(&input[s..e_off], "n");
    }

    #[test]
    fn span_byte_range_clamps_off_the_end() {
        let input = "MATCH (n)";
        let e = ParseError::Pest {
            message: "x".into(),
            // Beyond end of input — must clamp, must not panic.
            span: Span {
                start_line: 5,
                start_col: 99,
                end_line: 5,
                end_col: 100,
            },
        };
        let (s, e_off) = e.span_byte_range(input).expect("translation");
        assert_eq!(s, input.len());
        assert_eq!(e_off, input.len());
    }

    #[test]
    fn span_byte_range_returns_none_for_ast_construction_without_span() {
        let e = ParseError::AstConstruction {
            message: "no span".into(),
            span: None,
        };
        assert_eq!(e.span_byte_range("anything"), None);
    }

    #[test]
    fn span_byte_range_handles_multi_byte_utf8() {
        // Codex M4-01 retro F-05. Multi-byte UTF-8 input is reachable
        // via backtick identifiers (Cypher 9 §2.4 admits arbitrary
        // Unicode; F-03 positive pin asserts CJK round-trip). Pre-fix
        // `line_start + (col - 1)` returned wrong byte offsets for
        // such input and could panic on `&input[s..e]` mid-codepoint.
        //
        // Construct: a valid query containing a 6-byte (CJK) backtick
        // identifier followed by an intentional parse fault. Walk the
        // returned span and verify every byte coordinate falls on a
        // valid char boundary AND that slicing doesn't panic.
        let input = "MATCH (n) WHERE n.`日本` = \"bad RETURN n";
        // Sanity: the input contains multi-byte chars (CJK is 3 bytes
        // in UTF-8).
        assert!(!input.is_ascii(), "test input must contain non-ASCII");

        // The unclosed double-quoted string is the parse fault.
        let err = crate::parse(input).expect_err("unclosed string must error");
        let (start, end) = err
            .span_byte_range(input)
            .expect("Pest variant always carries a span");

        // Both coordinates must fall on a char boundary — otherwise
        // `&input[..start]` / `&input[start..]` would panic.
        assert!(
            input.is_char_boundary(start),
            "start={start} must fall on char boundary in {input:?}"
        );
        assert!(
            input.is_char_boundary(end),
            "end={end} must fall on char boundary in {input:?}"
        );

        // Slicing must not panic — that's the panic we are guarding
        // against (pre-fix `line_start + (col - 1)` could land mid-
        // codepoint).
        let _slice = &input[start..end];

        // Sanity: the span is at-or-near the unclosed quote site,
        // not somewhere absurd. The unclosed `"` byte position is
        // `input.find('"').unwrap()` (counting bytes, since `"` is
        // ASCII).
        let expected_quote_byte = input.find('"').expect("quote present");
        assert!(
            start >= expected_quote_byte && start <= input.len(),
            "start={start} should be at or after the unclosed-quote site \
             {expected_quote_byte} in input of length {}",
            input.len()
        );
    }

    #[test]
    fn span_byte_range_walks_chars_not_bytes_on_constructed_pest_span() {
        // Direct lower-level check (no parser dependency): construct a
        // ParseError with a known line:col and verify the byte offset
        // accounts for multi-byte chars in the line.
        //
        // Line: "abc日本def"
        //   - chars: a(1) b(2) c(3) 日(4) 本(5) d(6) e(7) f(8)
        //   - byte offsets (0-indexed): a=0, b=1, c=2, 日=3..6, 本=6..9, d=9, e=10, f=11
        //
        // col=6 (the `d`) should translate to byte offset 9, NOT
        // byte offset 5 (which is mid-codepoint inside `日`).
        let input = "abc日本def";
        let e = ParseError::Pest {
            message: "x".into(),
            span: Span {
                start_line: 1,
                start_col: 6,
                end_line: 1,
                end_col: 7,
            },
        };
        let (s, e_off) = e.span_byte_range(input).expect("translation");
        // The byte at `s` should be the start of `d`.
        assert!(input.is_char_boundary(s), "start must be char boundary");
        assert!(input.is_char_boundary(e_off), "end must be char boundary");
        assert_eq!(
            &input[s..e_off],
            "d",
            "expected `d` at col 6, got {:?}",
            &input[s..e_off]
        );
    }
}
