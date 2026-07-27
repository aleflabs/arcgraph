//! O(1) counts-store lookup operator for exact unfiltered count queries.

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::CountStoreSource;
use crate::semantic::bound_ast::BindingId;

#[derive(Debug)]
pub struct CountStoreOp {
    source: CountStoreSource,
    output_id: BindingId,
    schema: Vec<BindingId>,
    emitted: bool,
}

impl CountStoreOp {
    #[must_use]
    pub fn new(source: CountStoreSource, output_id: BindingId) -> Self {
        Self {
            source,
            output_id,
            schema: vec![output_id],
            emitted: false,
        }
    }

    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.emitted {
            return Ok(Batch::empty(self.schema.len()));
        }

        // The count-store total is a live catalog_stats aggregate, not an
        // MVCC snapshot-isolated read; still bind the statement snapshot LSN
        // before first output to preserve executor snapshot discipline.
        let _exec_lsn = ctx.ensure_snapshot_lsn();
        let count = substrate.count_store(ctx.tenant(), self.source)?;
        let count_i64 = i64::try_from(count).map_err(|_| {
            ExecutionError::Eval("CountStoreOp: count-store total exceeds i64::MAX".into())
        })?;
        let mut batch = Batch::with_capacity(self.schema.len());
        if !batch.push_row(vec![Value::Integer(count_i64)]) {
            return Err(ExecutionError::Eval(
                "CountStoreOp: batch overflow during single-row push".into(),
            ));
        }
        self.emitted = true;
        Ok(batch)
    }

    #[must_use]
    pub fn output_id(&self) -> BindingId {
        self.output_id
    }
}
