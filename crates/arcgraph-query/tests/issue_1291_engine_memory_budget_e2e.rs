//! **#1291** — `QueryEngine::with_memory_budget` threads the per-tenant
//! memory budget into every `ExecutionContext` the engine constructs.
//!
//! # The gap this pins
//!
//! The M4-64a `MemoryBudget` (ADR-038 amendment-03 §Structural-1) was
//! mechanically correct and threaded through every blocking operator,
//! but shipped DISABLED: `ExecutionContext` defaulted to an unbounded
//! budget and `QueryEngine` offered NO seam to attach a configured one
//! — so the served binary could not enable it, and the only guard for
//! unbudgeted tenants was `UNCAPPED_RUNAWAY_GUARD_ROWS` (`1 << 32` ≈
//! 4.29 B rows — effectively unbounded → OOM under a heavy query).
//!
//! # RED-on-revert
//!
//! `engine_budget_over_cap_query_surfaces_resource_exhausted` drives an
//! over-budget query through the FULL pipeline (`QueryEngine::execute`:
//! parse → bind → type-check → lower → execute). Reverting the #1291
//! threading (dropping `apply_memory_budget` from the engine's
//! `ExecutionContext` construction) makes the query succeed — the
//! `expect_err` fails RED. The sort operator is the enforcement point:
//! `SortOp::materialize_and_sort` charges `estimate_row_bytes` per
//! buffered row against the context budget when a cap is configured.

use arcgraph_query::executor::{MemoryBudget, StubExecutorSubstrate};
use arcgraph_query::semantic::error::ArcQLError;
use arcgraph_query::semantic::{CatalogProvider, StubCatalogProvider};
use arcgraph_query::{ExplainError, QueryEngine};

/// ~20k integer rows buffered by the ORDER BY sort: each row estimates
/// at ≥ 24 B Vec overhead + the `Value` stack size (≥ 56 B on 64-bit)
/// → ≥ 1.6 MB total, far above the 64 KiB test cap.
const OVER_BUDGET_QUERY: &str = "UNWIND range(1, 20000) AS x RETURN x ORDER BY x";

/// 10 rows ≈ < 1 KiB — comfortably under the 64 KiB test cap.
const UNDER_BUDGET_QUERY: &str = "UNWIND range(1, 10) AS x RETURN x ORDER BY x";

const TEST_CAP_BYTES: u64 = 64 * 1024;

fn engine_budget(catalog: &StubCatalogProvider) -> MemoryBudget {
    MemoryBudget::with_per_tenant_cap(catalog.tenant(), TEST_CAP_BYTES)
}

#[test]
fn engine_budget_over_cap_query_surfaces_resource_exhausted() {
    // The #1291 enablement seam: a budget attached to the ENGINE (not
    // hand-built contexts) must gate the execute path. This is the
    // exact shape the served binary wires.
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog).with_memory_budget(engine_budget(&catalog));
    let err = engine
        .execute(OVER_BUDGET_QUERY, &substrate)
        .expect_err("over-budget ORDER BY must trip the per-tenant byte cap");
    match err {
        ExplainError::ArcQL(ArcQLError::ResourceExhausted { cap_bytes, .. }) => {
            assert_eq!(
                cap_bytes, TEST_CAP_BYTES,
                "error reports the configured cap"
            );
        }
        other => panic!("expected ResourceExhausted; got {other:?}"),
    }
}

#[test]
fn engine_budget_under_cap_query_succeeds() {
    // A normal query under the cap is unaffected by the budget.
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog).with_memory_budget(engine_budget(&catalog));
    let result = engine
        .execute(UNDER_BUDGET_QUERY, &substrate)
        .expect("under-budget query succeeds with the budget attached");
    assert_eq!(result.rows().len(), 10);
}

#[test]
fn engine_without_budget_stays_opt_in_unbounded() {
    // Embedded / library posture pin: WITHOUT with_memory_budget, the
    // pre-#1291 behavior is byte-for-byte preserved — the same query
    // that trips the cap above completes (row-count runaway guard
    // only). This is the contrast leg that makes the RED-on-revert
    // direction of the first test unambiguous.
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    assert!(
        engine.memory_budget().is_none(),
        "default engine: no budget"
    );
    let result = engine
        .execute(OVER_BUDGET_QUERY, &substrate)
        .expect("uncapped engine admits the same query (opt-in posture)");
    assert_eq!(result.rows().len(), 20000);
}

#[test]
fn engine_budget_gates_the_multi_statement_path_too() {
    // #1291 NIT-2 — execute_multi_with_query_id_and_deadline constructs
    // its OWN context shared across the whole chain; the budget must
    // ride along there as well, making the with_memory_budget rustdoc's
    // "every ExecutionContext" claim true. RED-on-revert: dropping
    // apply_memory_budget from the multi path makes the chain succeed
    // and the expect_err below fails. A two-statement chain (under-cap
    // first, over-cap second) pins the MULTI path specifically — the
    // first statement proves the chain genuinely runs before the cap
    // trips on statement 2.
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog).with_memory_budget(engine_budget(&catalog));
    let chain = format!("{UNDER_BUDGET_QUERY}; {OVER_BUDGET_QUERY}");
    let err = engine
        .execute_multi(&chain, &substrate)
        .expect_err("over-budget statement in a multi chain must trip the per-tenant byte cap");
    match err {
        ExplainError::ArcQL(ArcQLError::ResourceExhausted { cap_bytes, .. }) => {
            assert_eq!(
                cap_bytes, TEST_CAP_BYTES,
                "error reports the configured cap"
            );
        }
        other => panic!("expected ResourceExhausted; got {other:?}"),
    }
}

#[test]
fn engine_budget_gates_the_explicit_txn_path_too() {
    // #1291 — execute_in_txn_with_parameters constructs its OWN
    // context; the budget must ride along there as well (Bolt
    // BEGIN…COMMIT parity with auto-commit RUN). A plan-only assertion
    // is not enough: drive the same over-budget statement through the
    // in-txn entry point. No substrate writes are staged — the
    // statement is read-shaped, so the returned held handle is
    // dropped without commit.
    use arcgraph_query::executor::eval::Parameters;
    use arcgraph_query::executor::substrate::HeldTxnHandle;

    #[derive(Debug)]
    struct NoopHeldTxn;
    impl HeldTxnHandle for NoopHeldTxn {
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
        fn snapshot_lsn(&self) -> arcgraph_core::Lsn {
            arcgraph_core::Lsn::MAX
        }
    }

    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog).with_memory_budget(engine_budget(&catalog));
    let (result, _held) = engine.execute_in_txn_with_parameters(
        OVER_BUDGET_QUERY,
        &substrate,
        Box::new(NoopHeldTxn),
        std::time::Duration::from_millis(arcgraph_query::cancel::DEFAULT_QUERY_TIMEOUT_MS),
        Parameters::new(),
    );
    let err = result.expect_err("over-budget in-txn statement must trip the cap");
    assert!(
        matches!(
            err,
            ExplainError::ArcQL(ArcQLError::ResourceExhausted { .. })
        ),
        "expected ResourceExhausted; got {err:?}"
    );
}
