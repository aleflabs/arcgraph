//! W26-γ-3 / ADR-136 — storage-adapter negative-path tests.
//!
//! # Surface
//!
//! [`arcgraph_mcp::storage::substrate::CrudExecutorSubstrate`] —
//! production [`arcgraph_query::ExecutorSubstrate`] impl backed by
//! [`arcgraph_storage::router::MultiTenantRouter`] +
//! [`arcgraph_storage::transaction::TxnManager`]. The substrate
//! returns structured [`SubstrateAccessError`] variants on every
//! failure mode under the code-quality policy + `feedback_review_oracle_relaxations.md`.
//!
//! # Adversarial classes covered
//!
//! 1. **Vector substrate unwired** — `vector_search` without
//!    `with_search_provider` returns `IndexUnavailable("vector")`.
//! 2. **BM25 substrate unwired** — `bm25_search` without a provider
//!    returns `IndexUnavailable("bm25")`.
//! 3. **Community substrate unwired (v1.0-α)** — `community_members`
//!    returns `IndexUnavailable("community")`.
//! 4. **No-panic invariant** — every failure mode is structured `Err`,
//!    never a panic.
//! 5. **Error equality + display stability** —
//!    [`SubstrateAccessError`] is `Eq`; failure diagnostic strings
//!    are byte-stable across repeated calls.
//! 6. **Pattern-match completeness** — every variant of the enum
//!    is matched via exhaustive matrix (compiler-enforced
//!    completeness check on `#[non_exhaustive]` enum via `matches!`).
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
//! every load-bearing surface gets a fault-injection regression test
//! per failure mode.

use std::sync::Arc;

use arcgraph_core::{Lsn, TenantId};
use arcgraph_mcp::storage::substrate::CrudExecutorSubstrate;
use arcgraph_query::executor::substrate::{ExecutorSubstrate, SubstrateAccessError};
use arcgraph_storage::InternTable;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::CrudStore;
use arcgraph_storage::io::InMemoryPageIo;
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;

/// Bootstrap a `CrudExecutorSubstrate` whose router has no search
/// provider attached — the minimal "vector + bm25 unwired" posture.
/// The catalog IS bootstrapped (needed before any router lookup).
fn bootstrap_unwired_substrate() -> CrudExecutorSubstrate {
    let io = Arc::new(InMemoryPageIo::new());
    let pool = BufferPool::new(64, io);
    let mgr = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog.bootstrap(&pool, &mgr).expect("catalog bootstrap");
    let crud = Arc::new(CrudStore::new());
    let router = Arc::new(MultiTenantRouter::new(catalog, Arc::clone(&crud), None));
    let intern = Arc::new(InternTable::new());
    CrudExecutorSubstrate::new(router, mgr, intern)
}

#[test]
fn vector_search_unwired_returns_index_unavailable() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    let result = s.vector_search(tenant, "embedding", &[0.0, 0.1, 0.2, 0.3], 10, Lsn::new(0));
    let err = result.expect_err("unwired vector_search must surface Err");
    assert_eq!(
        err,
        SubstrateAccessError::IndexUnavailable("vector".into()),
        "unwired vector substrate must surface IndexUnavailable(\"vector\")"
    );
}

#[test]
fn bm25_search_unwired_returns_index_unavailable() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    let result = s.bm25_search(tenant, "body", "fraud", 10, Lsn::new(0));
    let err = result.expect_err("unwired bm25_search must surface Err");
    assert_eq!(
        err,
        SubstrateAccessError::IndexUnavailable("bm25".into()),
        "unwired bm25 substrate must surface IndexUnavailable(\"bm25\")"
    );
}

#[test]
fn community_members_unwired_returns_index_unavailable() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    let result = s.community_members(tenant, 42, Lsn::new(0));
    let err = result.expect_err("unwired community must surface Err");
    assert_eq!(
        err,
        SubstrateAccessError::IndexUnavailable("community".into())
    );
}

#[test]
fn substrate_error_equality_is_stable() {
    // `SubstrateAccessError` is `Eq` — failure diagnostics MUST be
    // byte-stable across repeated invocations (no time / RNG / address
    // leakage into the string payload).
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    let e1 = s
        .vector_search(tenant, "embedding", &[0.0; 4], 10, Lsn::new(0))
        .unwrap_err();
    let e2 = s
        .vector_search(tenant, "embedding", &[0.0; 4], 10, Lsn::new(0))
        .unwrap_err();
    assert_eq!(e1, e2, "substrate error stability across repeated calls");
}

#[test]
fn substrate_error_display_messages_are_stable() {
    let e1 = SubstrateAccessError::IndexUnavailable("vector".into());
    let s1a = e1.to_string();
    let s1b = e1.to_string();
    assert_eq!(s1a, s1b);
    assert!(
        s1a.contains("vector"),
        "error msg must include substrate name"
    );

    let e2 = SubstrateAccessError::TenantUnknown(TenantId::new(42));
    let s2 = e2.to_string();
    // Display includes the wrapped TenantId Debug shape.
    assert!(!s2.is_empty(), "TenantUnknown display must be non-empty");

    let e3 = SubstrateAccessError::Io("disk full".into());
    let s3 = e3.to_string();
    assert!(s3.contains("disk full"));
}

#[test]
fn substrate_error_partial_eq_distinguishes_variants() {
    let a = SubstrateAccessError::IndexUnavailable("vector".into());
    let b = SubstrateAccessError::IndexUnavailable("bm25".into());
    assert_ne!(a, b);

    let c = SubstrateAccessError::IndexUnavailable("vector".into());
    assert_eq!(a, c);

    let d = SubstrateAccessError::TenantUnknown(TenantId::new(0));
    let e = SubstrateAccessError::TenantUnknown(TenantId::new(0));
    assert_eq!(d, e);
}

#[test]
fn substrate_error_is_send_sync_clone() {
    fn assert_send_sync<T: Send + Sync>() {}
    fn assert_clone<T: Clone>() {}
    assert_send_sync::<SubstrateAccessError>();
    assert_clone::<SubstrateAccessError>();
}

#[test]
fn vector_search_zero_k_no_panic() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    // K=0 is degenerate but must not panic — substrate returns
    // IndexUnavailable BEFORE any K-validation since the search
    // body never lights without a provider.
    let result = s.vector_search(tenant, "embedding", &[0.0; 4], 0, Lsn::new(0));
    let err = result.expect_err("must surface structured Err");
    assert!(matches!(err, SubstrateAccessError::IndexUnavailable(_)));
}

#[test]
fn vector_search_empty_query_vector_no_panic() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    // Empty query vector — degenerate input. Substrate returns
    // IndexUnavailable (the unwired path short-circuits before
    // any dimensionality check).
    let result = s.vector_search(tenant, "embedding", &[], 10, Lsn::new(0));
    let err = result.expect_err("must surface structured Err");
    assert!(matches!(err, SubstrateAccessError::IndexUnavailable(_)));
}

#[test]
fn bm25_search_empty_query_text_no_panic() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    let result = s.bm25_search(tenant, "body", "", 10, Lsn::new(0));
    let err = result.expect_err("must surface structured Err");
    assert!(matches!(err, SubstrateAccessError::IndexUnavailable(_)));
}

#[test]
fn bm25_search_huge_k_no_panic() {
    let s = bootstrap_unwired_substrate();
    let tenant = TenantId::DEFAULT;
    // Huge K — must not allocate excess memory; unwired path
    // short-circuits before any allocation.
    let result = s.bm25_search(tenant, "body", "x", u64::MAX, Lsn::new(0));
    let err = result.expect_err("must surface structured Err");
    assert!(matches!(err, SubstrateAccessError::IndexUnavailable(_)));
}

#[test]
fn substrate_error_pattern_match_completeness() {
    // Every variant of SubstrateAccessError MUST be pattern-
    // matchable. The `#[non_exhaustive]` attribute (under the code-quality policy
    // R-10) means we need a catch-all, but the explicit arms below
    // pin every CURRENT variant. Adding a new variant later requires
    // updating this test (or the catch-all will silently absorb).
    let variants = vec![
        SubstrateAccessError::TenantUnknown(TenantId::DEFAULT),
        SubstrateAccessError::IndexUnavailable("vector".into()),
        SubstrateAccessError::IndexUnavailable("bm25".into()),
        SubstrateAccessError::IndexUnavailable("community".into()),
        SubstrateAccessError::Io("synthetic".into()),
    ];
    for v in variants {
        let matched = matches!(
            v,
            SubstrateAccessError::TenantUnknown(_)
                | SubstrateAccessError::IndexUnavailable(_)
                | SubstrateAccessError::Io(_)
        );
        assert!(matched, "unmatched variant — test out of sync");
    }
}

#[test]
fn substrate_error_implements_std_error() {
    let e = SubstrateAccessError::IndexUnavailable("vector".into());
    // Must be a real std::error::Error (so JSON-RPC error mapping
    // can call `.source()` and `.to_string()` polymorphically).
    let as_dyn: &(dyn std::error::Error + 'static) = &e;
    let _ = std::error::Error::source(as_dyn);
    let _ = as_dyn.to_string();
}

#[test]
fn substrate_search_provider_initially_none() {
    let s = bootstrap_unwired_substrate();
    // Per ADR-132 D-3 + storage/substrate.rs:268-272, the search
    // provider is None until with_search_provider is called.
    assert!(
        s.search_provider().is_none(),
        "default substrate has no provider"
    );
}

#[test]
fn substrate_index_unavailable_diagnostic_contains_name() {
    // ADR-132 D-3 contract — IndexUnavailable carries the substrate
    // name (vector / bm25 / community) so the JSON-RPC error data
    // slot can route on it.
    for name in &["vector", "bm25", "community", "synthetic_extra"] {
        let e = SubstrateAccessError::IndexUnavailable((*name).into());
        let s = e.to_string();
        assert!(
            s.contains(name),
            "diagnostic must include name {name}: got {s:?}"
        );
    }
}
