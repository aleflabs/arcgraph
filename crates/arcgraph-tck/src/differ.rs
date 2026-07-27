//! Row-set diff for the openCypher TCK dual-execute.
//!
//! Per `feedback_review_oracle_relaxations.md`: relaxing the diff
//! oracle is the EXACT opposite of the dual-execute's purpose. The
//! differ here uses **strict multiset equality** (or strict
//! ordered-list equality when the query has an `ORDER BY`) — no
//! approximate matchers, no row-count-only short-circuits.
//!
//! Per `feedback_noop_trampoline_anti_pattern.md`: the differ explicitly
//! fails on diverging row-sets and produces an actionable report.
//! NEVER silently passes when one side errors and the other succeeds.

use std::collections::BTreeMap;

use crate::executor::RowSet;

/// Structured row-set diff. Produced by [`assert_row_set_equal`] when
/// the two row-sets differ; consumed by test reporting code that
/// renders divergences for the CI log.
#[derive(Debug, Clone)]
pub struct RowSetDiff {
    /// Whether the differ was run in ordered-equality mode (true) or
    /// multiset-equality mode (false).
    pub ordered: bool,
    /// Number of rows on the lhs.
    pub lhs_row_count: usize,
    /// Number of rows on the rhs.
    pub rhs_row_count: usize,
    /// Rows present in lhs but missing from rhs (under the chosen
    /// equivalence relation).
    pub lhs_only: Vec<Vec<String>>,
    /// Rows present in rhs but missing from lhs.
    pub rhs_only: Vec<Vec<String>>,
    /// First mismatched row index in ordered mode; `None` in
    /// multiset mode.
    pub first_position_mismatch: Option<usize>,
}

impl std::fmt::Display for RowSetDiff {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "row-set diff ({}): lhs={} rows, rhs={} rows",
            if self.ordered { "ordered" } else { "multiset" },
            self.lhs_row_count,
            self.rhs_row_count,
        )?;
        if let Some(pos) = self.first_position_mismatch {
            writeln!(f, "  first position mismatch at index {pos}")?;
        }
        if !self.lhs_only.is_empty() {
            writeln!(
                f,
                "  lhs-only ({} rows): {}",
                self.lhs_only.len(),
                render_rows_preview(&self.lhs_only)
            )?;
        }
        if !self.rhs_only.is_empty() {
            writeln!(
                f,
                "  rhs-only ({} rows): {}",
                self.rhs_only.len(),
                render_rows_preview(&self.rhs_only)
            )?;
        }
        Ok(())
    }
}

fn render_rows_preview(rows: &[Vec<String>]) -> String {
    let preview: Vec<String> = rows
        .iter()
        .take(5)
        .map(|row| format!("[{}]", row.join(",")))
        .collect();
    if rows.len() > 5 {
        format!("{} … (+{} more)", preview.join(", "), rows.len() - 5)
    } else {
        preview.join(", ")
    }
}

/// Assert two row-sets are equivalent under the chosen equivalence
/// relation. Returns `Ok(())` on match, `Err(RowSetDiff)` on
/// divergence.
///
/// `ordered = true` is for queries carrying an `ORDER BY` clause; row
/// order MUST match exactly. `ordered = false` (the TCK default for
/// queries without `ORDER BY`) is multiset equality — order doesn't
/// matter but every row in lhs must appear the same number of times
/// in rhs.
pub fn assert_row_set_equal(lhs: &RowSet, rhs: &RowSet, ordered: bool) -> Result<(), RowSetDiff> {
    if lhs.rows == rhs.rows && ordered {
        return Ok(());
    }
    if ordered {
        let first_mismatch = lhs
            .rows
            .iter()
            .zip(rhs.rows.iter())
            .position(|(a, b)| a != b);
        let mut lhs_only = Vec::new();
        let mut rhs_only = Vec::new();
        for row in lhs.rows.iter() {
            if !rhs.rows.contains(row) {
                lhs_only.push(row.clone());
            }
        }
        for row in rhs.rows.iter() {
            if !lhs.rows.contains(row) {
                rhs_only.push(row.clone());
            }
        }
        if first_mismatch.is_some()
            || lhs.rows.len() != rhs.rows.len()
            || !lhs_only.is_empty()
            || !rhs_only.is_empty()
        {
            return Err(RowSetDiff {
                ordered: true,
                lhs_row_count: lhs.rows.len(),
                rhs_row_count: rhs.rows.len(),
                lhs_only,
                rhs_only,
                first_position_mismatch: first_mismatch,
            });
        }
        return Ok(());
    }
    // Multiset mode: per-row count comparison.
    let lhs_counts = row_counts(&lhs.rows);
    let rhs_counts = row_counts(&rhs.rows);
    if lhs_counts == rhs_counts {
        return Ok(());
    }
    let mut lhs_only = Vec::new();
    for (row, &count) in lhs_counts.iter() {
        let rhs_count = rhs_counts.get(row).copied().unwrap_or(0);
        for _ in 0..count.saturating_sub(rhs_count) {
            lhs_only.push(row.clone());
        }
    }
    let mut rhs_only = Vec::new();
    for (row, &count) in rhs_counts.iter() {
        let lhs_count = lhs_counts.get(row).copied().unwrap_or(0);
        for _ in 0..count.saturating_sub(lhs_count) {
            rhs_only.push(row.clone());
        }
    }
    Err(RowSetDiff {
        ordered: false,
        lhs_row_count: lhs.rows.len(),
        rhs_row_count: rhs.rows.len(),
        lhs_only,
        rhs_only,
        first_position_mismatch: None,
    })
}

fn row_counts(rows: &[Vec<String>]) -> BTreeMap<Vec<String>, usize> {
    let mut out: BTreeMap<Vec<String>, usize> = BTreeMap::new();
    for row in rows {
        *out.entry(row.clone()).or_insert(0) += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rs(rows: Vec<Vec<&str>>) -> RowSet {
        RowSet::from_rows(
            rows.into_iter()
                .map(|r| r.into_iter().map(String::from).collect())
                .collect(),
        )
    }

    #[test]
    fn multiset_equal_returns_ok_for_reordered_rows() {
        let lhs = rs(vec![vec!["1"], vec!["2"], vec!["3"]]);
        let rhs = rs(vec![vec!["3"], vec!["1"], vec!["2"]]);
        assert!(assert_row_set_equal(&lhs, &rhs, false).is_ok());
    }

    #[test]
    fn ordered_equal_rejects_reordered_rows() {
        let lhs = rs(vec![vec!["1"], vec!["2"], vec!["3"]]);
        let rhs = rs(vec![vec!["3"], vec!["1"], vec!["2"]]);
        assert!(assert_row_set_equal(&lhs, &rhs, true).is_err());
    }

    #[test]
    fn multiset_reports_lhs_only_rows() {
        let lhs = rs(vec![vec!["1"], vec!["2"], vec!["3"]]);
        let rhs = rs(vec![vec!["1"], vec!["2"]]);
        let diff = assert_row_set_equal(&lhs, &rhs, false).expect_err("must diff");
        assert_eq!(diff.lhs_only, vec![vec!["3".to_string()]]);
        assert!(diff.rhs_only.is_empty());
    }

    #[test]
    fn multiset_reports_rhs_only_rows() {
        let lhs = rs(vec![vec!["1"]]);
        let rhs = rs(vec![vec!["1"], vec!["2"]]);
        let diff = assert_row_set_equal(&lhs, &rhs, false).expect_err("must diff");
        assert!(diff.lhs_only.is_empty());
        assert_eq!(diff.rhs_only, vec![vec!["2".to_string()]]);
    }

    #[test]
    fn multiset_counts_duplicates() {
        let lhs = rs(vec![vec!["1"], vec!["1"], vec!["2"]]);
        let rhs = rs(vec![vec!["1"], vec!["2"]]);
        let diff = assert_row_set_equal(&lhs, &rhs, false).expect_err("must diff on multiplicity");
        assert_eq!(diff.lhs_only, vec![vec!["1".to_string()]]);
    }

    #[test]
    fn ordered_reports_first_position_mismatch() {
        let lhs = rs(vec![vec!["a"], vec!["b"], vec!["c"]]);
        let rhs = rs(vec![vec!["a"], vec!["c"], vec!["b"]]);
        let diff = assert_row_set_equal(&lhs, &rhs, true).expect_err("must diff");
        assert_eq!(diff.first_position_mismatch, Some(1));
    }

    #[test]
    fn display_renders_diff_summary() {
        let lhs = rs(vec![vec!["1"]]);
        let rhs = rs(vec![vec!["2"]]);
        let diff = assert_row_set_equal(&lhs, &rhs, false).expect_err("must diff");
        let rendered = format!("{diff}");
        assert!(rendered.contains("multiset"));
        assert!(rendered.contains("lhs-only"));
        assert!(rendered.contains("rhs-only"));
    }
}
