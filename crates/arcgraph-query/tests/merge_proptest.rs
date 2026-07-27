//! ADR-151 W26-θ Phase 5 + ADR-152-amendment-01 — MERGE proptest pin:
//! MERGE-then-MERGE is idempotent on the Stub substrate (no duplicate
//! creates when the match-branch fires on the previously-created node).
//!
//! ADR-152-amendment-01 §D-2: the match-branch now ENFORCES the
//! pattern label. For idempotency, the label must be interned so the
//! match-branch lowers to `Scan{label: Some(id)}` (NOT the §D-3
//! `LogicalEmpty` that fires on an un-interned label). The bind catalog
//! therefore maps the generated label name → `STUB_FIRST_LABEL_ID`
//! (`1024`), which is exactly the id the Stub's `create_node` interns
//! the first label to — so the lowered `Scan{Some(1024)}` finds the
//! node the first MERGE created. This models the production steady
//! state (the per-statement catalog rebuild + `commits_observed`
//! plan-cache watermark re-bind once the label is interned).

use proptest::prelude::*;

use arcgraph_core::{LabelId, Lsn, PartitionId, TenantId};

use arcgraph_query::executor::substrate::{ExecutorSubstrate, StubExecutorSubstrate};
use arcgraph_query::executor::{ExecutionContext, Pipeline};
use arcgraph_query::logical_plan::{LogicalPlan, LogicalPlanLoweringVisitor};
use arcgraph_query::parse;
use arcgraph_query::semantic::{
    BindingVisitor, CrossSubstrateValidator, StubCatalogProvider, TypeCheckVisitor,
};

/// The Stub's `create_node` interns the first label name to this id.
const STUB_FIRST_LABEL_ID: u32 = 1024;

fn lower(query: &str, label: &str) -> Option<LogicalPlan> {
    let stmt = parse(query).ok()?;
    // ADR-152-amendment-01 §D-1 — intern the label so the MERGE
    // match-branch lowers to the label-enforced `Scan{Some(id)}`.
    let cat = StubCatalogProvider::new().with_label_id(label, LabelId::new(STUB_FIRST_LABEL_ID));
    let mut bound = BindingVisitor::bind(&stmt, query, &cat).ok()?;
    TypeCheckVisitor::check(&mut bound, &cat).ok()?;
    CrossSubstrateValidator::validate(&bound, &cat).ok()?;
    LogicalPlanLoweringVisitor::lower(&bound).ok()
}

fn drain(plan: &LogicalPlan, stub: &StubExecutorSubstrate, ctx: &ExecutionContext) {
    let mut op = Pipeline::build(plan).expect("pipeline build OK");
    loop {
        let b = op.next_batch(ctx, stub).expect("batch OK");
        if b.is_empty() {
            break;
        }
    }
}

/// Strategy for valid label names.
///
/// There is no shared Rust helper for the grammar's `keyword`
/// exclusion set, and hand-maintained keyword filters drift. Prefixing
/// with `L_` keeps generated bare labels identifier-safe while making
/// them unable to equal a reserved keyword.
fn label_strategy() -> impl Strategy<Value = String> {
    "[A-Z][a-zA-Z0-9_]{0,7}".prop_map(|s| format!("L_{s}"))
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        .. ProptestConfig::default()
    })]

    /// MERGE-then-MERGE is idempotent on the Stub substrate.
    ///
    /// First call creates the node; second call finds it via the
    /// match-branch + fires the (empty) on_match action — no duplicate.
    #[test]
    fn merge_then_merge_idempotent(label in label_strategy()) {
        let query = format!("MERGE (n:{label})");
        let plan = lower(&query, &label).expect("lower OK");
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);

        // First MERGE — create branch fires.
        drain(&plan, &s, &ctx);
        let count_after_1 = s.scan_nodes(tenant, None, Lsn::MAX).unwrap().len();
        prop_assert_eq!(count_after_1, 1, "first MERGE creates 1 node");

        // Second MERGE — match branch fires; no new node.
        drain(&plan, &s, &ctx);
        let count_after_2 = s.scan_nodes(tenant, None, Lsn::MAX).unwrap().len();
        prop_assert_eq!(count_after_2, 1, "second MERGE is idempotent");
    }
}
