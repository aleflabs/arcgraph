//! Vectorized [`Batch`] cursor primitive.
//!
//! A [`Batch`] is the unit of work flowing between operators in the
//! M4-61 pipeline. The 2048-row cap is the forward-pin chosen so:
//!
//! - **M4-64a memory budget**: per-query memory tracking can be done
//!   per-batch (a tracker slot is bumped at `next_batch` boundary).
//! - **M4-64b SIMD**: 2048 rows is a common AVX-512 / NEON working
//!   set — large enough to amortize loop overhead, small enough to
//!   stay in L2.
//! - **M4-92 cancellation**: 2048 rows is well inside the cancel-
//!   latency budget per ADR-036 §D-24 (a single-batch read on a
//!   cold-cache substrate is < 1 ms; tripping the cancellation
//!   token at batch boundaries delivers cancel within a single
//!   batch's worth of work).
//!
//! # Layout
//!
//! v1.0-alpha ships a row-major `Vec<Vec<Value>>` per
//! "factorized intermediate" per ADR-038 amendment-03 Structural-1.
//! The outer `Vec` is the row dimension (≤ [`BATCH_ROWS`]); the inner
//! `Vec<Value>` is one cell per [`crate::semantic::BindingId`]
//! position (the operator's column schema).
//!
//! Future M4-64b SIMD specialization will re-shape into a
//! column-major `Vec<Column>` for vectorized predicate eval; the
//! [`Batch::push_row`] API stays stable across that change because
//! the per-row mutation surface is invariant under row-major-vs-
//! column-major.
//!
//! # ADR provenance
//! - **ADR-038 amendment-02 §M4.f** — primary M4-61 batch size cite.
//! - **ADR-038 amendment-03 Structural-1** — factorized intermediate
//!   forward-pin for M4-64a / M4-64b.
//! - **ADR-036 §D-24** — cancel-latency budget; 2048 rows fits.

use crate::executor::value::Value;

/// Row-count cap for a single [`Batch`].
///
/// 2048 = 2¹¹ — power-of-two for SIMD convenience; large enough to
/// amortize per-batch overhead; small enough to stay in L2 for typical
/// row widths (a 6-column row of `Value::Integer` is 48 bytes per
/// row × 2048 = 96 KiB, well under a typical 1 MiB L2 cache).
pub const BATCH_ROWS: usize = 2048;

/// A 2048-row factorized intermediate flowing between operators.
///
/// Each row is a `Vec<Value>` whose length matches the operator's
/// per-batch column schema. The schema is uniform within a batch
/// (every row in a batch has the same column count); a downstream
/// operator can rely on `batch.column_count()` to size its output.
///
/// # Invariants
///
/// - `rows.len() <= BATCH_ROWS` — checked by [`Self::push_row`].
/// - `rows[i].len() == column_count` for every row in the batch
///   (uniform-shape invariant; pinned by debug-assertion in
///   [`Self::push_row`]).
///
/// # Empty-batch sentinel
///
/// An empty batch (`rows.len() == 0`) is the "operator has no more
/// rows" sentinel returned from [`crate::executor::PhysicalOperator::next_batch`].
/// A single-batch query may emit 1 non-empty batch then 1 empty
/// batch; a multi-batch query emits N non-empty batches then 1
/// empty batch. The
/// [`crate::executor::execute`] driver loops until the empty sentinel.
#[derive(Debug, Clone, PartialEq)]
pub struct Batch {
    /// Per-row cell vectors. `rows[i][j]` is the cell at row `i`,
    /// column `j`. Inner Vec length is uniform within a batch.
    rows: Vec<Vec<Value>>,
    /// Cached column count (= `rows[0].len()` if `rows` non-empty,
    /// else the schema-derived hint set at construction). Cached so
    /// downstream operators can read column count without a length
    /// check.
    column_count: usize,
}

impl Batch {
    /// Construct an empty batch with the given column count.
    ///
    /// `column_count` is the per-row width every subsequent
    /// [`Self::push_row`] MUST honor; the debug-assertion in
    /// `push_row` traps a mismatched-row bug at the operator level.
    #[must_use]
    pub fn empty(column_count: usize) -> Self {
        Self {
            rows: Vec::new(),
            column_count,
        }
    }

    /// Construct an empty batch sized for [`BATCH_ROWS`] rows.
    ///
    /// Reserves capacity for [`BATCH_ROWS`] up-front so per-row
    /// pushes inside the operator's hot loop don't trigger reallocs.
    #[must_use]
    pub fn with_capacity(column_count: usize) -> Self {
        Self {
            rows: Vec::with_capacity(BATCH_ROWS),
            column_count,
        }
    }

    /// Construct from a pre-built row vector. Asserts row-shape
    /// uniformity in debug builds. Returns `None` if `rows.len() >
    /// BATCH_ROWS` (the caller mis-sized the batch).
    #[must_use]
    pub fn from_rows(rows: Vec<Vec<Value>>) -> Option<Self> {
        if rows.len() > BATCH_ROWS {
            return None;
        }
        let column_count = rows.first().map(Vec::len).unwrap_or(0);
        // Debug-only uniformity check — production hot-path skips.
        debug_assert!(
            rows.iter().all(|r| r.len() == column_count),
            "Batch::from_rows: rows have non-uniform widths"
        );
        Some(Self { rows, column_count })
    }

    /// Append one row. Asserts shape uniformity in debug builds.
    /// Returns `false` if the batch is at [`BATCH_ROWS`] capacity (the
    /// operator is responsible for spilling overflow into the next
    /// batch); returns `true` on successful push.
    pub fn push_row(&mut self, row: Vec<Value>) -> bool {
        if self.rows.len() >= BATCH_ROWS {
            return false;
        }
        debug_assert_eq!(
            row.len(),
            self.column_count,
            "Batch::push_row: row width {} mismatches column_count {}",
            row.len(),
            self.column_count,
        );
        self.rows.push(row);
        true
    }

    /// Number of rows currently in the batch.
    #[inline]
    #[must_use]
    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// Per-row column width.
    #[inline]
    #[must_use]
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// `true` if the batch holds zero rows. The empty-batch sentinel
    /// from `next_batch` per module docs.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// `true` if the batch is at [`BATCH_ROWS`] capacity (further
    /// pushes will return `false`). Operators check this at
    /// loop tops so they can flush the current batch and start a
    /// new one.
    #[inline]
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.rows.len() >= BATCH_ROWS
    }

    /// Borrow the rows for read-only iteration.
    #[inline]
    #[must_use]
    pub fn rows(&self) -> &[Vec<Value>] {
        &self.rows
    }

    /// Borrow a single row by index. Panics in debug builds if out
    /// of bounds; release builds return an empty-slice fallback (the
    /// debug check is the load-bearing guarantee).
    #[inline]
    #[must_use]
    pub fn row(&self, idx: usize) -> &[Value] {
        debug_assert!(
            idx < self.rows.len(),
            "Batch::row: index {idx} out of bounds (row_count={})",
            self.rows.len()
        );
        self.rows.get(idx).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Decompose into the row vector. Used by
    /// [`crate::execute_with_context`] to flatten the per-batch
    /// stream into the v1.0-alpha materialized result.
    #[inline]
    #[must_use]
    pub fn into_rows(self) -> Vec<Vec<Value>> {
        self.rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_rows_constant_is_2048() {
        // Pin: per ADR-038 amendment-02 §M4.f the batch size is 2048.
        // A future amendment that re-tunes this MUST update this test
        // alongside; load-bearing for M4-64a + M4-64b forward pins.
        assert_eq!(BATCH_ROWS, 2048);
    }

    #[test]
    fn empty_batch_is_empty_and_not_full() {
        let b = Batch::empty(1);
        assert!(b.is_empty());
        assert!(!b.is_full());
        assert_eq!(b.row_count(), 0);
        assert_eq!(b.column_count(), 1);
    }

    #[test]
    fn push_row_appends_until_capacity() {
        let mut b = Batch::with_capacity(1);
        for i in 0..10 {
            assert!(b.push_row(vec![Value::Integer(i)]));
        }
        assert_eq!(b.row_count(), 10);
        assert_eq!(b.row(3), &[Value::Integer(3)]);
    }

    #[test]
    fn push_row_returns_false_at_batch_rows_capacity() {
        let mut b = Batch::with_capacity(1);
        for _ in 0..BATCH_ROWS {
            assert!(b.push_row(vec![Value::Null]));
        }
        assert!(b.is_full());
        // Next push fails.
        assert!(!b.push_row(vec![Value::Null]));
        assert_eq!(b.row_count(), BATCH_ROWS);
    }

    #[test]
    fn from_rows_rejects_oversized() {
        let big = vec![vec![Value::Null]; BATCH_ROWS + 1];
        assert!(Batch::from_rows(big).is_none());
    }

    #[test]
    fn from_rows_accepts_at_or_below_capacity() {
        let rows = vec![vec![Value::Integer(1), Value::Boolean(true)]; 5];
        let b = Batch::from_rows(rows).expect("under capacity");
        assert_eq!(b.row_count(), 5);
        assert_eq!(b.column_count(), 2);
    }

    #[test]
    fn into_rows_returns_owned_vec() {
        let mut b = Batch::with_capacity(1);
        b.push_row(vec![Value::Integer(7)]);
        let rows = b.into_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0], vec![Value::Integer(7)]);
    }

    #[test]
    fn from_rows_empty_vec_keeps_caller_supplied_column_count_at_zero() {
        // Edge case: an empty rows vec doesn't carry per-row width;
        // we encode "unknown until first push" as `column_count = 0`.
        // Operators downstream MUST construct via `Batch::empty(N)` /
        // `Batch::with_capacity(N)` to set the schema width.
        let b = Batch::from_rows(Vec::new()).expect("empty");
        assert_eq!(b.column_count(), 0);
    }
}
