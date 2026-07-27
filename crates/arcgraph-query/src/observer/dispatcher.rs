//! Dispatcher hook bridging
//! [`crate::executor::PhysicalOperator::next_batch`] to the
//! [`crate::observer::RowCountObserver`] registered on the
//! [`crate::executor::ExecutionContext`].
//!
//! The hook is `pub(crate)` so the executor's
//! `crate::executor::ops::PhysicalOperator::next_batch` can call it
//! without exposing the bridge to external callers — observation is an
//! internal-only contract between the executor and the observer.
//!
//! # Why a dedicated module
//!
//! The dispatcher modification in `crate::executor::ops::mod` is kept
//! to a SINGLE call site: `record_dispatch(ctx, op_kind, &batch_result,
//! elapsed_ns)`. All observer-side logic (option-presence check, batch
//! introspection, structured-field tracing) lives here. This minimizes
//! the rebase surface against W12α (M4-63 + M4-64a) which adds new
//! operator variants in the same file.

use std::time::Instant;

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::observer::row_count::OperatorKind;

/// Record one batch dispatch event. Called by
/// [`crate::executor::PhysicalOperator::next_batch`] after the per-
/// operator inner match dispatch returns.
///
/// No-op if the [`ExecutionContext`] has no observer attached. When an
/// observer IS attached, this records the batch's row count, accumulated
/// wall-time, and high-water memory estimate via
/// [`crate::observer::RowCountObserver::record_dispatched_batch`].
///
/// # Tracing
///
/// Per amendment-03 §TIER-2-c, this hook does NOT add tracing events
/// itself — the observer's `record_batch` already emits at
/// `target = "arcgraph_query::observer::row_count"`. The dispatcher hook
/// is purely the wiring; observability concerns live in the observer
/// module proper.
///
/// # Panics
///
/// Will not panic. The batch_result and elapsed_ns are read defensively;
/// observation is best-effort (an Err result is silently skipped — the
/// inner match already returned the error to the caller, no need to
/// double-report).
#[inline]
pub fn record_dispatch(
    ctx: &ExecutionContext,
    op_kind: OperatorKind,
    batch_result: &Result<Batch, ExecutionError>,
    start: Instant,
) {
    let Some(observer) = ctx.observer() else {
        return;
    };
    let Ok(batch) = batch_result else {
        // The dispatcher returned an error. No batch to observe;
        // skipping is the right discipline (Err already surfaced to
        // caller).
        return;
    };
    let elapsed_ns = elapsed_ns(start);
    observer.record_dispatched_batch(op_kind, batch, elapsed_ns);
}

#[inline]
fn elapsed_ns(start: Instant) -> u64 {
    let elapsed = start.elapsed();
    // u128 → u64 saturating conversion. Per ADR-036 §D-24, no operator
    // batch should ever exceed u64::MAX nanoseconds (~584 years); the
    // saturating form is defense-in-depth.
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor::batch::Batch;
    use crate::executor::error::ExecutionError;
    use crate::observer::RowCountObserver;
    use arcgraph_core::{PartitionId, TenantId};
    use std::sync::Arc;

    /// Hook is no-op when no observer is attached — the recording
    /// goes nowhere; the function returns cleanly.
    #[test]
    fn record_dispatch_is_noop_without_observer() {
        use crate::executor::value::Value;
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        // 0-column batch with 0 rows is the EOS shape Batch::empty(0)
        // produces.
        let batch = Batch::empty(0);
        let result: Result<Batch, ExecutionError> = Ok(batch);
        // No panic — function returns without touching anything.
        record_dispatch(&ctx, OperatorKind::Scan, &result, Instant::now());
        // A non-empty 1-column batch likewise no-ops.
        let mut batch = Batch::with_capacity(1);
        for _ in 0..3 {
            assert!(batch.push_row(vec![Value::Null]));
        }
        let result: Result<Batch, ExecutionError> = Ok(batch);
        record_dispatch(&ctx, OperatorKind::Scan, &result, Instant::now());
    }

    /// Hook records to the observer when one is attached. Verifies the
    /// row count + observer state advancement.
    #[test]
    fn record_dispatch_writes_to_attached_observer() {
        use crate::executor::value::Value;
        let observer = Arc::new(RowCountObserver::new());
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
            .with_observer(Arc::clone(&observer));
        let mut batch = Batch::with_capacity(1);
        for _ in 0..5 {
            assert!(batch.push_row(vec![Value::Null]));
        }
        let result: Result<Batch, ExecutionError> = Ok(batch);
        record_dispatch(&ctx, OperatorKind::Scan, &result, Instant::now());
        let metrics = observer.metrics();
        assert_eq!(metrics.len(), 1);
        assert_eq!(metrics[0].op_kind, Some(OperatorKind::Scan));
        assert_eq!(metrics[0].observed_rows, 5);
    }

    /// Hook silently skips Err results — the executor caller already
    /// surfaced the error; we don't double-record.
    #[test]
    fn record_dispatch_skips_err_results() {
        let observer = Arc::new(RowCountObserver::new());
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO)
            .with_observer(Arc::clone(&observer));
        let result: Result<Batch, ExecutionError> = Err(ExecutionError::Cancelled);
        record_dispatch(&ctx, OperatorKind::Scan, &result, Instant::now());
        // Observer was not touched.
        assert!(observer.metrics().is_empty());
    }
}
