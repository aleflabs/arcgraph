//! **#1291** — the served binary's per-tenant memory cap is ENFORCED:
//! an over-budget query through the production `graph.raw_query`
//! executor surfaces `-32009 BudgetExceeded`, NOT the ≈4.29 B-row
//! `UNCAPPED_RUNAWAY_GUARD_ROWS` fallback (→ OOM under a heavy query).
//!
//! # The gap this pins
//!
//! The M4-64a `MemoryBudget` shipped mechanically correct but DISABLED:
//! the byte cap defaulted to `None` and the served binary never called
//! `set_per_tenant_cap`, so unbudgeted tenants had NO real byte
//! ceiling. #1291 wires a default cap
//! (`arcgraph_cli::ops::resolve_per_tenant_memory_cap()` — 1 GiB,
//! `ARCGRAPH_TENANT_MEMORY_CAP_BYTES` overrides, `0` disables) through
//! `StorageRawQueryExecutor::with_per_tenant_memory_cap` +
//! `StorageBoltHandler::with_per_tenant_memory_cap` +
//! `QueryEngine::with_memory_budget`.
//!
//! # RED-on-revert
//!
//! `served_over_budget_query_gets_budget_error` builds the executor the
//! SAME way the served binary does (builder + resolved cap; the test
//! substitutes a small explicit cap so it doesn't need >1 GiB of rows)
//! and asserts the budget error class. Reverting the enablement at ANY
//! layer — the builder application in
//! `StorageRawQueryExecutor::execute`, or the `apply_memory_budget`
//! threading inside `QueryEngine` — makes the over-budget query
//! succeed and the `expect_err` fail RED.
//!
//! This drives the FULL served stack: `StorageRawQueryExecutor` →
//! per-call catalog → `QueryEngine::execute_with_deadline` →
//! `SortOp` budget reservation → `ArcQLError::ResourceExhausted` →
//! `MCPError::BudgetExceeded` (-32009) — the exact path a served
//! `graph.raw_query` takes.

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::ops::memory_cap::DEFAULT_PER_TENANT_MEMORY_CAP_BYTES;
use arcgraph_core::TenantId;
use arcgraph_mcp::MCPError;
use arcgraph_mcp::storage::{StorageBackend, StorageRawQueryExecutor};
use arcgraph_mcp::tools::raw_query::RawQueryExecutor;
use arcgraph_query::CancellationToken;

/// ~20k integer rows buffered by the ORDER BY sort — ≥ 1.6 MB of
/// estimated row bytes, far above [`TEST_CAP_BYTES`] but instant to
/// produce (no substrate rows needed).
const OVER_BUDGET_QUERY: &str = "UNWIND range(1, 20000) AS x RETURN x ORDER BY x";

/// 10 rows ≈ < 1 KiB — comfortably under every cap in this file.
const UNDER_BUDGET_QUERY: &str = "UNWIND range(1, 10) AS x RETURN x ORDER BY x";

/// Small explicit cap so the over-budget leg doesn't need to buffer
/// more than 1 GiB. The DEFAULT (1 GiB) value itself is pinned by the
/// `ops::memory_cap` unit tests; THIS file pins the enforcement seam.
const TEST_CAP_BYTES: u64 = 64 * 1024;

fn in_memory_backend() -> StorageBackend {
    let (backend, _guard) =
        bootstrap_storage_backend(&BootstrapMode::InMemory).expect("in-memory bootstrap");
    backend
}

#[test]
fn served_over_budget_query_gets_budget_error() {
    // The served wiring shape (builder-applied cap), with a small cap.
    let exec = StorageRawQueryExecutor::new(in_memory_backend())
        .with_per_tenant_memory_cap(TEST_CAP_BYTES);
    let cancel = CancellationToken::new();
    let err = exec
        .execute(TenantId::DEFAULT, OVER_BUDGET_QUERY, 100, &cancel)
        .expect_err("over-budget query must trip the per-tenant byte cap");
    // -32009 BudgetExceeded — the dedicated resource class (#980
    // Part 2), NOT the generic -32005 QueryError.
    match err {
        MCPError::BudgetExceeded { detail } => {
            assert!(
                detail.contains(&TEST_CAP_BYTES.to_string()),
                "detail names the configured cap: {detail}"
            );
        }
        other => panic!("expected BudgetExceeded (-32009); got {other:?}"),
    }
}

#[test]
fn served_under_budget_query_succeeds_with_the_default_cap() {
    // The ACTUAL served default (1 GiB) admits normal queries — the
    // cap is a runaway ceiling, not a workload throttle.
    let exec = StorageRawQueryExecutor::new(in_memory_backend())
        .with_per_tenant_memory_cap(DEFAULT_PER_TENANT_MEMORY_CAP_BYTES);
    let cancel = CancellationToken::new();
    let rows = exec
        .execute(TenantId::DEFAULT, UNDER_BUDGET_QUERY, 100, &cancel)
        .expect("normal query under the default cap succeeds");
    assert_eq!(rows.rows.len(), 10);
    // Sorted ascending — the query really executed through SortOp.
    let first = rows.rows[0].as_array().expect("row is an array")[0]
        .as_i64()
        .expect("integer cell");
    assert_eq!(first, 1);
}

#[test]
fn uncapped_executor_still_admits_the_over_budget_query() {
    // Contrast leg (embedded / opt-in posture pin): WITHOUT the
    // builder, the same query completes — which is exactly why the
    // served binary MUST apply the cap (#1291). If a revert removes
    // the enablement, `served_over_budget_query_gets_budget_error`
    // goes RED while this leg keeps passing — an unambiguous signal
    // that the SEAM (not the query) broke.
    let exec = StorageRawQueryExecutor::new(in_memory_backend());
    let cancel = CancellationToken::new();
    let rows = exec
        .execute(TenantId::DEFAULT, OVER_BUDGET_QUERY, 100, &cancel)
        .expect("uncapped executor admits the over-budget query");
    // max_rows=100 truncates the wire result; the query itself ran.
    assert_eq!(rows.rows.len(), 100);
}
