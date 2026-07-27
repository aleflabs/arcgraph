//! Dual-execute infrastructure for the openCypher TCK harness (W18δ
//! Task §2 + addendum item 4).
//!
//! # Architecture
//!
//! Two executors implement the [`TckExecutor`] trait:
//!
//! - [`ArcGraphExecutor`] — runs queries in-process against
//!   [`arcgraph_query::QueryEngine`] + an `ExecutorSubstrate`. Used
//!   for the always-on smoke-side of the dual-execute.
//! - [`Neo4jOracleExecutor`] — runs queries against a `neo4j-community:5`
//!   Docker container via a minimal Bolt 5.0 client. Used as the
//!   "ground truth" oracle. Env-gated; PANIC by default per
//!   `feedback_test_env_gate_panic_by_default.md`. Opt-out:
//!   `ARCGRAPH_TCK_SKIP_OK=1`.
//!
//! [`crate::differ::assert_row_set_equal`] compares two [`RowSet`]s
//! from the dual-execute and reports divergences.
//!
//! # Why "skeleton" not "fully wired"
//!
//! The W11 R-7 commitment names the harness skeleton. End-to-end live
//! diffing against a Docker neo4j container is the v1.1 next-wave
//! scope (the Bolt 5.0 client implementation here covers the BASE
//! handshake + HELLO + RUN + PULL but NOT every PackStream type the
//! TCK consumes — Date, Duration, Point, etc. are forward-pinned at
//! the codec-level error to the v1.1 expansion).

pub mod arcgraph;
pub mod neo4j_oracle;

pub use self::arcgraph::ArcGraphExecutor;
pub use self::neo4j_oracle::Neo4jOracleExecutor;

use std::fmt;

/// Wire-shape neutral row-set surfaced by both executors. Each row is
/// a Vec of column values; each column value is rendered as a string
/// so the row-set diff is platform-neutral (Neo4j's PackStream and
/// ArcGraph's `arcgraph_query::Value` do not have a common typed
/// surface today; the W18δ skeleton uses string-render parity as the
/// equivalence relation).
///
/// Forward-pin: v1.1 swaps to a typed `Value` union once both
/// executors share a canonical projection-cell encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowSet {
    /// Column names in projection order. `None` when the executor
    /// surfaces no projection metadata (W17α ArcGraph side).
    pub columns: Option<Vec<String>>,
    /// Rows in producer order. Row order matters only when the query
    /// carries an `ORDER BY`; the differ accepts either ordered or
    /// multiset comparison via [`crate::differ::assert_row_set_equal`]
    /// parameters.
    pub rows: Vec<Vec<String>>,
}

impl RowSet {
    /// Empty row-set.
    pub fn empty() -> Self {
        Self {
            columns: None,
            rows: Vec::new(),
        }
    }

    /// Row-set with no column metadata, just the row body.
    pub fn from_rows(rows: Vec<Vec<String>>) -> Self {
        Self {
            columns: None,
            rows,
        }
    }
}

/// Error type both executors surface. Variants are tagged so the
/// differ can distinguish "ArcGraph parse error" from
/// "Neo4j network error" etc.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutorError {
    /// Parse or planning failed at the executor (BindingError /
    /// ParseError / TypeCheckError).
    #[error("plan-build error: {0}")]
    PlanBuild(String),
    /// Execution started but did not complete cleanly.
    #[error("execution error: {0}")]
    Execution(String),
    /// I/O failure (network / file).
    #[error("I/O error: {0}")]
    Io(String),
    /// The oracle executor requires a Docker `neo4j:5` instance which
    /// is unavailable. The dual-execute harness gates on this to
    /// surface a structured panic rather than a silent skip per
    /// `feedback_test_env_gate_panic_by_default.md`.
    #[error("oracle unavailable: {0}")]
    OracleUnavailable(String),
}

/// Trait every TCK executor satisfies.
pub trait TckExecutor: fmt::Debug {
    /// Friendly name for logs / divergence reports (`"ArcGraph"`,
    /// `"Neo4jOracle"`).
    fn name(&self) -> &'static str;

    /// Execute `cypher` and return a row-set.
    fn execute(&self, cypher: &str) -> Result<RowSet, ExecutorError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rowset_empty_constructor() {
        let rs = RowSet::empty();
        assert!(rs.columns.is_none());
        assert!(rs.rows.is_empty());
    }

    #[test]
    fn rowset_from_rows() {
        let rs = RowSet::from_rows(vec![vec!["a".into()], vec!["b".into()]]);
        assert!(rs.columns.is_none());
        assert_eq!(rs.rows.len(), 2);
    }
}
