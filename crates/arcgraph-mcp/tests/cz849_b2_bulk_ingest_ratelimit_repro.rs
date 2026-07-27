//! **#849 Blocker 2 (HIGH, usability)** — durable bulk-loading rejected
//! chunked `graph.ingest` calls with `-32007 rate limited` after the
//! ~10-write/min default cap, so no high-throughput bulk-ingest path
//! existed. This file is the WRITE-class twin of
//! `cz833_stdio_read_no_leak_e2e.rs` (which pinned the READ surface).
//!
//! # The finding this file PROVES (measured, not asserted)
//!
//! #849-B2 was filed on main `2e3c2fab` (2026-06-03), when the
//! trusted-local serve dispatcher still carried the W14γ M5-12
//! per-tenant rate-limiter (`with_session_scope_and_rate_limiter`,
//! default policy). **#838** (`0398a4ff`, 2026-06-04, Closes #833)
//! then DROPPED that limiter from BOTH trusted-local serve binaries
//! (`bin/arcgraph.rs` + `bin/arcgraph_mcp_stdio.rs`) — they now build
//! the dispatcher via `with_session_scope` (`rate_limiter: None`). The
//! removal was scoped to the whole limiter (read AND write buckets),
//! not just the read cap that motivated #833. As a direct consequence
//! the 10-write/min cap that hard-stopped bulk ingest in #849-B2 is no
//! longer enforced on any trusted-local transport.
//!
//! # What THIS file pins (in-process dispatcher properties)
//!
//! 1. [`bulk_ingest_unthrottled_on_trusted_local_dispatcher`] — a
//!    dispatcher built like the CURRENT (post-#838) serve surface
//!    (`with_session_scope`, no limiter, exactly the
//!    `build_default_dispatcher` rate-limit-relevant shape) accepts 15
//!    sequential `graph.ingest` write calls in well under a minute,
//!    EVERY one a success envelope (no `-32007`). This is the #849-B2
//!    capability proof: the trusted-local bulk-ingest path is no longer
//!    hard-stopped at 10/min. It goes RED if a future change re-wires a
//!    write limiter onto the trusted-local serve dispatcher (the
//!    regression #849-B2 guards against).
//!
//! 2. [`pre_fix_default_write_limit_was_the_10_ingest_cap`] — the
//!    DISCRIMINATING root-cause repro: a dispatcher built like the
//!    PRE-#838 serve surface (`with_session_scope_and_rate_limiter`,
//!    DEFAULT policy — capacity 10, refill 10/min) reproduces #849-B2
//!    exactly: the first 10 `graph.ingest` calls succeed (burst
//!    capacity) and the 11th returns `-32007`. This proves the limiter
//!    — specifically the design-v2 §9.4 / ADR-004 am-02 default write
//!    cap — was the blocker, and that #838's removal is what unblocked
//!    it (NOT some unrelated change).

use std::sync::Arc;

use arcgraph_core::TenantId;
use arcgraph_mcp::jsonrpc::JsonRpcRequest;
use arcgraph_mcp::storage::{
    StorageBackend, StorageHybridSearcher, StorageIngestProvider, StorageNeighborhoodExplorer,
    StorageNodeInspector, StorageRawQueryExecutor, StorageSchemaProvider,
};
use arcgraph_mcp::{Dispatcher, RateLimiter, SessionScope};
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

/// Fresh in-memory backend with `PrimaryIndex` wired (mirrors the
/// production bootstrap + `tests/raw_query_write_common::fresh_backend`).
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

/// Build a dispatcher EXACTLY like the CURRENT (post-#838) trusted-local
/// serve surface: no rate limiter (`with_session_scope`). This is the
/// rate-limit-relevant shape of `bin/arcgraph.rs::build_default_dispatcher`
/// (the `.with_subscribe_provider(..)` chain it adds is orthogonal to the
/// rate-limit gate).
fn current_serve_dispatcher(backend: &StorageBackend) -> TestDispatcher {
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

/// Build a dispatcher like the PRE-#838 serve surface: the W14γ M5-12
/// per-tenant rate-limiter at DEFAULT policy (capacity 10 / refill
/// 10-per-minute on the write bucket).
fn pre_fix_serve_dispatcher(backend: &StorageBackend) -> TestDispatcher {
    Dispatcher::with_session_scope_and_rate_limiter(
        TenantId::DEFAULT,
        SessionScope::Power,
        Arc::new(StorageSchemaProvider::new(backend.clone())),
        Arc::new(StorageNodeInspector::new(backend.clone())),
        Arc::new(StorageNeighborhoodExplorer::new(backend.clone())),
        Arc::new(StorageHybridSearcher::new(backend.clone())),
        Arc::new(StorageIngestProvider::new(backend.clone())),
        Arc::new(StorageRawQueryExecutor::new(backend.clone())),
        RateLimiter::new(),
    )
}

/// Drive one `graph.ingest` write call carrying two uniquely-keyed nodes
/// and one edge between them (so call `i` never collides with call `j`).
/// Returns the raw response envelope (caller inspects `error` / `result`).
fn ingest_chunk(d: &TestDispatcher, chunk: usize) -> Value {
    let a = format!("svc-{chunk}-a");
    let b = format!("svc-{chunk}-b");
    let req = JsonRpcRequest {
        jsonrpc: "2.0".into(),
        id: Some(json!(chunk as i64)),
        method: "graph.ingest".into(),
        params: json!({
            "tenant_id": 1,
            "nodes": [
                { "external_id": a, "label": "Service", "properties": {} },
                { "external_id": b, "label": "Service", "properties": {} }
            ],
            "relationships": [
                { "from_external_id": a, "to_external_id": b,
                  "rel_type": "DEPENDS_ON", "properties": {} }
            ],
            "format": "json"
        }),
    };
    d.dispatch(req)
        .expect("ingest dispatch returns Some(envelope)")
}

/// `-32007` (`MCPError::RateLimited`) — the wire code the issue quotes.
fn is_rate_limited(resp: &Value) -> bool {
    resp["error"]["code"].as_i64() == Some(-32007)
}

// ─────────────────────────────────────────────────────────────────────
// 1. The capability the #849-B2 fix (via #838) delivers: bulk ingest is
//    NOT throttled on the trusted-local serve path.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn bulk_ingest_unthrottled_on_trusted_local_dispatcher() {
    // 15 > the 10-token default write-bucket capacity. On the PRE-#838
    // shape (limiter present) the 11th would be `-32007`; on the CURRENT
    // shape (no limiter) all 15 succeed. The calls run back-to-back in
    // microseconds — well inside one minute — so no refill masks the
    // result: a limiter, if present, would still be drained at call 11.
    let backend = fresh_backend();
    let d = current_serve_dispatcher(&backend);

    for chunk in 0..15 {
        let resp = ingest_chunk(&d, chunk);
        assert!(
            !is_rate_limited(&resp),
            "chunk {chunk}: trusted-local bulk ingest must NOT be rate-limited \
             (the #849-B2 regression), got: {resp:?}",
        );
        assert!(
            resp["error"].is_null(),
            "chunk {chunk}: ingest must succeed, got error envelope: {resp:?}",
        );
        assert!(
            !resp["result"].is_null(),
            "chunk {chunk}: ingest must carry a result envelope, got: {resp:?}",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// 2. DISCRIMINATING repro: the DEFAULT write limiter (the PRE-#838
//    shape) was the #849-B2 blocker — 10 succeed, the 11th is -32007.
// ─────────────────────────────────────────────────────────────────────

#[test]
fn pre_fix_default_write_limit_was_the_10_ingest_cap() {
    // Faithful reproduction of #849-B2 on the PRE-#838 serve shape:
    // design-v2 §9.4 / ADR-004 am-02 default write policy = capacity 10,
    // refill 10/min. The first 10 `graph.ingest` calls drain the burst
    // pool; the 11th rejects with `-32007`. Refill over the ~microsecond
    // span of 11 dispatches is ≈ 0 tokens, so this is deterministic.
    let backend = fresh_backend();
    let d = pre_fix_serve_dispatcher(&backend);

    for chunk in 0..10 {
        let resp = ingest_chunk(&d, chunk);
        assert!(
            !is_rate_limited(&resp),
            "chunk {chunk}: first 10 writes are within the default burst \
             capacity, got: {resp:?}",
        );
        assert!(
            resp["error"].is_null(),
            "chunk {chunk}: first 10 writes must succeed, got: {resp:?}",
        );
    }

    // 11th call: bucket empty → the exact `-32007 rate limited` the
    // #849-B2 repro quotes. This is the load-bearing safety oracle: if
    // it stops being RED on the pre-fix shape, the limiter default has
    // been silently weakened (the #818-class regression class).
    let resp11 = ingest_chunk(&d, 10);
    assert!(
        is_rate_limited(&resp11),
        "the 11th write/min MUST hit the default 10-write/min cap (-32007) \
         on the PRE-#838 limiter shape — this is the #849-B2 root cause, \
         got: {resp11:?}",
    );
}
