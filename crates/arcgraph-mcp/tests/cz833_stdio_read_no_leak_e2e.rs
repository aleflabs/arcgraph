//! **#833 (P0, CZ Stage-3 of #818)** — the MCP stdio read surface
//! silently returned EMPTY for every read after ~100 reads/process.
//!
//! # Root cause (MEASURED, not guessed — see the PR body)
//!
//! NOT an MVCC read-snapshot leak. A probe over a real production
//! dispatcher drove 250 sequential `graph.raw_query` reads and recorded
//! `TxnManager::active_count()` after each: it stayed **flat at 0** —
//! the per-read borrowed `Transaction` is dropped per call (`Drop` →
//! `abort_txn` → `active.remove`). The exhausted resource was the W14γ
//! M5-12 **per-tenant read rate-limiter** (capacity **100**,
//! refill ≈ 1.667/s): `graph.search` + `graph.raw_query` are BOTH
//! `OpClass::Read` (`op_class_for_method`), so they SHARE one
//! `(tenant, Read)` token bucket. After ~100 reads/min the bucket is
//! empty and every further read is rejected with `-32007` — which
//! agent clients (langchain, the #818 recall harness) coerce to an
//! EMPTY result-set, i.e. silently-wrong answers. The fix removes the
//! (multi-tenant NETWORK) rate-limiter from the TRUSTED-LOCAL stdio
//! dispatcher (`bin/arcgraph_mcp_stdio.rs` + `bin/arcgraph.rs`).
//!
//! # What THIS file pins (in-process dispatcher properties)
//!
//! 1. [`no_throttle_serves_unbounded_reads_without_mvcc_leak`] — a
//!    dispatcher built like the FIXED stdio surface
//!    (`with_session_scope`, no limiter) serves 250 sequential reads,
//!    EVERY one correct + non-empty (reads 101..250 explicitly), and
//!    `active_count()` stays 0 throughout (the "no MVCC leak" proof the
//!    Director asked for — a probe that goes RED if a future change
//!    leaks read snapshots).
//! 2. [`read_resource_exhaustion_is_a_loud_error_never_silent_empty`] —
//!    defect 2: an EXHAUSTED read returns an ERROR envelope (`-32007`,
//!    `result` absent), NEVER a success envelope with empty rows. Pins
//!    "fail loud" against a future regression that softens exhaustion
//!    to `Ok([])`.
//! 3. [`default_read_rate_limit_was_the_100_read_silent_cap`] — the
//!    DISCRIMINATING root-cause repro: a dispatcher built like the
//!    PRE-fix stdio surface (`with_session_scope_and_rate_limiter`,
//!    default policy) reproduces the bug — reads past the ~100-read
//!    boundary return `-32007` while `active_count()` stays 0 (proving
//!    the limiter, NOT a leak).
//!
//! The faithful end-to-end reproduction over the REAL stdio transport
//! (subprocess, `graph.search` + `graph.raw_query` interleaved, 250
//! reads, RED-on-revert) lives in
//! `crates/arcgraph-cli/tests/cz833_stdio_read_lifecycle_e2e.rs` — it
//! must be a subprocess test because the rate-limiter is wired in the
//! BINARY's dispatcher construction (same rationale as the #818
//! `served_vector_transport_818.rs` subprocess test).

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{Dispatcher, OpClass, RateLimiter, SessionScope};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::primary_index::PrimaryIndex;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use serde_json::{Value, json};

type TestDispatcher = Dispatcher<
    StorageSchemaProvider,
    StorageNodeInspector,
    StorageNeighborhoodExplorer,
    StorageHybridSearcher,
    StorageIngestProvider,
    StorageRawQueryExecutor,
>;

/// Fresh in-memory backend with `PrimaryIndex` wired so label-anchored
/// `MATCH (n:Person)` returns rows (mirrors the production bootstrap +
/// `tests/return_alias_columns_wire_e2e.rs::fresh_backend`).
fn fresh_backend() -> StorageBackend {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&mgr), Arc::clone(&allocator), None).expect("PrimaryIndex"),
    );
    let crud = Arc::new(CrudStore::new_with_index(None, primary, allocator));
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    StorageBackend::new(router, mgr, intern)
}

/// Build a dispatcher EXACTLY like the FIXED stdio surface: no rate
/// limiter (`with_session_scope`).
fn fixed_stdio_dispatcher(backend: &StorageBackend) -> TestDispatcher {
    Dispatcher::with_session_scope(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
    )
}

/// Build a dispatcher with a rate limiter (the PRE-fix stdio shape, or
/// a custom-capped one for the defect-2 test).
fn rate_limited_dispatcher(backend: &StorageBackend, limiter: RateLimiter) -> TestDispatcher {
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
        limiter,
    )
}

fn raw_query(d: &TestDispatcher, query: &str) -> Value {
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(1)),
        method: "graph.raw_query".into(),
        params: json!({ "tenant_id": 1, "query": query, "max_rows": 100, "format": "json" }),
    };
    d.dispatch(req).expect("dispatch returns Some(envelope)")
}

/// Parse the row_count out of a `graph.raw_query` success envelope.
/// Returns `None` if the envelope is an error (no `result`).
fn row_count(resp: &Value) -> Option<u64> {
    resp["result"]["body"]
        .as_str()
        .and_then(|b| serde_json::from_str::<Value>(b).ok())
        .and_then(|b| b["row_count"].as_u64())
}

const PERSONS: &[&str] = &["Ada", "Bob", "Cy", "Dot", "Eve"];

/// Seed the known graph via `CREATE` (raw_query). Returns nothing; the
/// caller drives reads afterward.
fn seed_persons(d: &TestDispatcher) {
    for name in PERSONS {
        let r = raw_query(d, &format!("CREATE (n:Person {{name: '{name}'}})"));
        assert!(r["error"].is_null(), "seed CREATE failed: {r:?}");
    }
}

// ─────────────────────────────────────────────────────────────────────
// 1. The FIX: unbounded reads, all correct, no MVCC snapshot leak.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn no_throttle_serves_unbounded_reads_without_mvcc_leak() {
    let backend = fresh_backend();
    let d = fixed_stdio_dispatcher(&backend);
    seed_persons(&d);

    let want = PERSONS.len() as u64;
    let q = "MATCH (n:Person) RETURN n.name AS name";

    // Drive 250 sequential reads — well past the old ~100 boundary.
    for i in 1..=250u32 {
        let resp = raw_query(&d, q);
        assert!(
            resp["error"].is_null(),
            "read #{i}: FIXED stdio surface must never reject a read \
             (the #833 bug returned -32007 here for i>~100): {resp:?}"
        );
        assert_eq!(
            row_count(&resp),
            Some(want),
            "read #{i}: must return all {want} Person rows (reads 101..250 \
             were the silently-empty zone in #833)"
        );
        // The Director's explicit "no MVCC read-snapshot leak" probe:
        // each read's borrowed Transaction is dropped per-call, so the
        // active-txn table never grows.
        let active = backend.txn_manager().active_count();
        assert_eq!(
            active, 0,
            "read #{i}: active_count() must stay 0 (a climbing count = a \
             leaked read snapshot); got {active}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// 2. Defect 2: an EXHAUSTED read fails LOUD, never returns Ok(empty).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn read_resource_exhaustion_is_a_loud_error_never_silent_empty() {
    let backend = fresh_backend();
    // Keep a clone so we can tighten the read cap AFTER seeding (the
    // clone shares state via `Arc<RateLimiterInner>`).
    let limiter = RateLimiter::new();
    let d = rate_limited_dispatcher(&backend, limiter.clone());

    // Seed one Person so the label binds (otherwise MATCH on an unseeded
    // graph returns a binding error, not the rate-limit error we want to
    // pin). The default 100-token read bucket easily covers one CREATE.
    let seed = raw_query(&d, "CREATE (n:Person {name: 'Ada'})");
    assert!(seed["error"].is_null(), "seed CREATE failed: {seed:?}");

    // Now tighten the READ class to 2 tokens / 0.0 refill so the 3rd
    // read is guaranteed exhausted within the test (no wall-clock
    // refill); set_per_tenant clamps the live bucket's tokens to 2.
    limiter.set_per_tenant(TenantId::DEFAULT, OpClass::Read, 2, 0.0);

    let q = "MATCH (n:Person) RETURN n.name AS name";
    // First two reads consume the 2 tokens (they return the seeded row).
    let r1 = raw_query(&d, q);
    assert!(r1["error"].is_null(), "read #1 should serve: {r1:?}");
    let r2 = raw_query(&d, q);
    assert!(r2["error"].is_null(), "read #2 should serve: {r2:?}");

    // Third read is exhausted. It MUST be an ERROR envelope, NOT a
    // success envelope with empty rows.
    let r3 = raw_query(&d, q);
    assert_eq!(
        r3["error"]["code"].as_i64(),
        Some(-32007),
        "exhausted read must surface -32007 (resource exhausted), not a \
         silent result: {r3:?}"
    );
    assert!(
        r3.get("result").is_none() || r3["result"].is_null(),
        "exhausted read must NOT carry a `result` (a DB that returns \
         wrong/empty answers silently is worse than one that errors): {r3:?}"
    );
    // And explicitly: it is not an empty-rows success.
    assert_eq!(
        row_count(&r3),
        None,
        "exhausted read must not render an (empty) row set: {r3:?}"
    );
}

// ─────────────────────────────────────────────────────────────────────
// 3. DISCRIMINATING root-cause repro: the default-policy limiter (the
//    PRE-fix stdio shape) caps reads at ~100 while active_count stays 0.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn default_read_rate_limit_was_the_100_read_silent_cap() {
    let backend = fresh_backend();
    // Exactly what the stdio binary did before the fix:
    let d = rate_limited_dispatcher(&backend, RateLimiter::new());
    seed_persons(&d); // 5 CREATEs are Read-class → consume 5 read tokens.

    let q = "MATCH (n:Person) RETURN n.name AS name";
    let mut ok = 0u32;
    let mut rate_limited = 0u32;
    for _ in 1..=250u32 {
        let resp = raw_query(&d, q);
        if resp["error"]["code"].as_i64() == Some(-32007) {
            rate_limited += 1;
        } else if row_count(&resp) == Some(PERSONS.len() as u64) {
            ok += 1;
        }
        // The active-txn table NEVER grows — proving the ~100 cap is the
        // rate limiter, not a leaked read snapshot.
        assert_eq!(
            backend.txn_manager().active_count(),
            0,
            "active_count must stay 0 even under the buggy limiter (the \
             cap is the rate-limiter, NOT an MVCC leak)"
        );
    }

    // ~95 reads succeed (100-token bucket minus 5 seed CREATEs), then the
    // rest are rate-limited. Assert the SHAPE of the bug robustly:
    assert!(
        (80..=100).contains(&ok),
        "expected ~95 reads to succeed before the 100-token read bucket \
         drained; got {ok}"
    );
    assert!(
        rate_limited >= 140,
        "expected the bulk of 250 reads (past the ~100 cap) to be \
         -32007 rate-limited under the PRE-fix limiter; got {rate_limited}"
    );
}
