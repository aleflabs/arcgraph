//! M4-31 logical-plan-lowering errors.
//!
//! [`LogicalPlanError`] is the M4-31 error taxonomy surfaced by
//! [`crate::logical_plan::LogicalPlanLoweringVisitor::lower`]. It
//! mirrors the M4-21 [`crate::semantic::error::BindingError`] /
//! M4-22 [`crate::semantic::error::TypeCheckError`] / M4-23
//! [`crate::semantic::error::CrossSubstrateError`] shape:
//!
//! - `thiserror`-derived;
//! - every variant carries a primary [`Span`];
//! - the variant set is a closed contract for downstream M4-32 /
//!   M4-33 / M4-05 consumption (`#[non_exhaustive]` is **omitted**
//!   on the same rationale as the M4-2x error taxonomies);
//! - lifts into [`crate::semantic::error::ArcQLError`] via the
//!   `ArcQLError::LogicalPlan` variant added by M4-31.
//!
//! # Variant scope
//!
//! - [`LogicalPlanError::NotImplementedAtM4_31`] is the load-bearing
//!   "deferred to a future slice" marker. Per ADR-038 §2 D-24, M4-31
//!   ships the SIMPLE-operator subset (Scan / Expand / Filter / Project
//!   / Join / Limit / Skip / Empty); RANK BY HYBRID + the hybrid-fusion
//!   family defer to M4-32, aggregation + sort + path operators defer
//!   to M4-33. Every unsupported surface emits this variant with a
//!   `target_slice` slot naming the future slice.
//! - [`LogicalPlanError::InvalidPlanStructure`] is reserved for future
//!   structural violations the lowering pass cannot continue past
//!   (e.g., a query with no clauses at all reaching the lowering pass).
//! - [`LogicalPlanError::JoinConditionFailed`] is reserved for the
//!   multi-pattern MATCH join case where shared-variable resolution
//!   fails (shouldn't happen post-M4-21 binding, but we keep the
//!   variant defensively for any programmatic constructor of
//!   `BoundQuery` that bypasses the binding pass).
//!
//! # ADR provenance
//! - ADR-038 §2 D-24 — logical-plan-lowering contract (this file's
//!   primary spec).
//! - ADR-038 §2 D-23 — visitor-trait discipline lock (M4-31 inherits;
//!   this file's surface is consumed by a CUSTOM walker).
//! - ADR-038 §2 D-16 — error-taxonomy split (parser / semantic /
//!   logical-plan layers stay distinct).
//! - ADR-038 amendment-03 §M4-31 row — test-artifact pin: the
//!   `NotImplementedAtM4_31` markers cover RANK BY HYBRID + aggregation
//!   surfaces.

use crate::error::Span;

/// Faults surfaced by [`crate::logical_plan::LogicalPlanLoweringVisitor::lower`].
///
/// Each variant carries a `span` pointing at the offending token in
/// the original input. Mirrors [`crate::semantic::error::BindingError`]
/// / [`crate::semantic::error::TypeCheckError`] /
/// [`crate::semantic::error::CrossSubstrateError`] shape; lifts into
/// [`crate::semantic::error::ArcQLError`] via the
/// `ArcQLError::LogicalPlan` variant added by M4-31.
///
/// `#[non_exhaustive]` is **omitted** on the same rationale as the
/// M4-2x error taxonomies: the variant set is M4-31's public contract
/// for downstream M4-32 / M4-33 / M4-05 consumption.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LogicalPlanError {
    /// A surface that M4-31 deliberately defers to a future slice was
    /// encountered during lowering. The `surface` slot names the
    /// offending construct (e.g., `"RANK BY HYBRID"`,
    /// `"aggregation function"`); the `target_slice` slot names the
    /// future slice that lights it (e.g., `"M4-32"`, `"M4-33"`); the
    /// `span` points at the offending token.
    ///
    /// Per ADR-038 §2 D-24 the M4-32 / M4-33 implementations replace
    /// the emission sites with proper lowering rather than masking the
    /// marker.
    #[error(
        "logical plan lowering not implemented at M4-31 for `{surface}` (target slice: {target_slice}) at {span}"
    )]
    NotImplementedAtM4_31 {
        /// The offending construct (a stable `&'static str` literal so
        /// the error type stays `Eq` + cheap to clone).
        surface: &'static str,
        /// The future slice that will light this surface.
        target_slice: &'static str,
        /// Span of the offending token.
        span: Span,
    },

    /// A `BoundQuery` reached lowering with a structural shape the
    /// lowering pass cannot continue past. Reserved for defensive
    /// rejection of programmatic `BoundQuery` constructors that
    /// bypass the M4-21 binding pass.
    #[error("invalid logical plan structure at {span}: {reason}")]
    InvalidPlanStructure {
        /// Human-readable explanation of what is structurally wrong.
        reason: String,
        /// Span of the offending construct.
        span: Span,
    },

    /// A multi-pattern MATCH join's shared-variable resolution
    /// failed. Reserved defensively — post-M4-21 binding, shared
    /// variables across patterns within the same MATCH-chain scope
    /// are guaranteed to share `binding_id`. This variant exists for
    /// any programmatic `BoundQuery` constructor that bypasses the
    /// binding pass.
    #[error(
        "join condition resolution failed at {span}: shared variable `{var}` not bound in either input"
    )]
    JoinConditionFailed {
        /// The variable name that failed to resolve to either input.
        var: String,
        /// Span of the offending join site.
        span: Span,
    },
}

impl LogicalPlanError {
    /// Return the carried primary [`Span`].
    pub fn span(&self) -> &Span {
        match self {
            LogicalPlanError::NotImplementedAtM4_31 { span, .. }
            | LogicalPlanError::InvalidPlanStructure { span, .. }
            | LogicalPlanError::JoinConditionFailed { span, .. } => span,
        }
    }

    /// Translate the primary span (line:col coordinates) into a
    /// byte-offset range in the original input string.
    ///
    /// Returns `None` only on coordinate-system mismatch (defensive;
    /// should not happen for spans produced by `LogicalPlanLoweringVisitor`).
    /// Mirrors [`crate::semantic::error::BindingError::span_byte_range`].
    pub fn span_byte_range(&self, input: &str) -> Option<(usize, usize)> {
        let span = self.span();
        let start = line_col_to_byte(input, span.start_line, span.start_col)?;
        let end = line_col_to_byte(input, span.end_line, span.end_col)?;
        Some((start, end))
    }
}

/// Convert a 1-indexed `(line, col)` coordinate into a byte offset
/// into `input`. Clamps off-the-end lines/columns to `input.len()`.
///
/// **Defensive duplicate** of `crate::semantic::error::line_col_to_byte`
/// (which itself defensively duplicates `crate::error::line_col_to_byte`).
/// `crate::semantic::error::line_col_to_byte` is private to its module;
/// duplicating the 12-line helper avoids exposing it.
fn line_col_to_byte(input: &str, line: usize, col: usize) -> Option<usize> {
    if line == 0 || col == 0 {
        return None;
    }
    let bytes = input.as_bytes();
    let mut current_line = 1usize;
    let mut line_start = 0usize;
    for (i, b) in bytes.iter().enumerate() {
        if current_line == line {
            let offset = line_start + (col - 1);
            return Some(offset.min(bytes.len()));
        }
        if *b == b'\n' {
            current_line += 1;
            line_start = i + 1;
        }
    }
    if current_line == line {
        let offset = line_start + (col - 1);
        return Some(offset.min(bytes.len()));
    }
    Some(bytes.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_accessor_returns_primary_span() {
        let e = LogicalPlanError::NotImplementedAtM4_31 {
            surface: "RANK BY HYBRID",
            target_slice: "M4-32",
            span: Span::point(2, 14),
        };
        assert_eq!(e.span(), &Span::point(2, 14));
    }

    #[test]
    fn not_implemented_display_carries_surface_and_slice() {
        let e = LogicalPlanError::NotImplementedAtM4_31 {
            surface: "aggregation function",
            target_slice: "M4-33",
            span: Span::point(1, 8),
        };
        let s = format!("{e}");
        assert!(s.contains("aggregation function"), "got: {s}");
        assert!(s.contains("M4-33"), "got: {s}");
        assert!(s.contains("M4-31"), "got: {s}");
    }

    #[test]
    fn span_byte_range_translates_single_line() {
        let input = "MATCH (n:Doc) RETURN sum(n.x)";
        let e = LogicalPlanError::NotImplementedAtM4_31 {
            surface: "aggregation function",
            target_slice: "M4-33",
            // col 22 is the `s` in `sum` (1-indexed).
            span: Span {
                start_line: 1,
                start_col: 22,
                end_line: 1,
                end_col: 25,
            },
        };
        let (s, eoff) = e.span_byte_range(input).expect("translation");
        assert_eq!(&input[s..eoff], "sum");
    }

    #[test]
    fn equality_is_structural() {
        let a = LogicalPlanError::JoinConditionFailed {
            var: "n".into(),
            span: Span::point(1, 1),
        };
        let b = LogicalPlanError::JoinConditionFailed {
            var: "n".into(),
            span: Span::point(1, 1),
        };
        assert_eq!(a, b);
    }

    #[test]
    fn implements_std_error_trait() {
        fn assert_impls_error<E: std::error::Error>(_: E) {}
        assert_impls_error(LogicalPlanError::InvalidPlanStructure {
            reason: "x".into(),
            span: Span::point(1, 1),
        });
    }
}
