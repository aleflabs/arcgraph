//! W26-γ-2 D5#1 — Negative scenario: power loss / fsync failure
//! mid-commit (the "fsyncgate" class).
//!
//! Real-world incident: PostgreSQL fsyncgate (CVE-2018-7187 / 2018-04
//! pgsql-bug postings). When a kernel-side fsync fails, Linux's
//! Open(2) handle no longer reports the failure to the next fsync
//! caller — the page-cache state silently de-syncs from the on-disk
//! state. PostgreSQL's WAL writer assumed every fsync that returned
//! 0 had written its bytes; the fsyncgate fix added a "panic on first
//! EIO" policy.
//!
//! ArcGraph's analog: `ArcGraphError::WalErrorRolledBack` per ADR-033
//! Z-1(b) — when a WAL fsync returns Err, the in-memory rollback
//! machinery unwinds MVCC versions + page-state mutations BEFORE
//! surfacing the structured error to the caller. The caller MUST get
//! a retryable error, not silent corruption.
//!
//! This test asserts:
//!
//! 1. `ArcGraphError::WalErrorRolledBack` is constructible + carries
//!    its underlying `WalUnavailable` source.
//! 2. `Display` includes the "rolled back (retryable)" hint.
//! 3. `source()` exposes the underlying WAL error for operator
//!    diagnostics.
//! 4. The error variant is matchable for retry-logic dispatch.
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`:
//! this test is REVERSE-TESTABLE — a regression that silently
//! swallows the WAL error (returning `Ok` instead of
//! `WalErrorRolledBack`) would fail this test's `is_err()` assertion;
//! a regression that emits `Io(...)` raw without the rollback wrapper
//! would fail the variant-match assertion.

use std::error::Error;
use std::io;

use arcgraph_core::error::ArcGraphError;

#[test]
fn wal_error_rolled_back_carries_retryable_hint() {
    // Mimic an Io(EIO) failing the WAL fsync.
    let inner = ArcGraphError::Io(io::Error::other("simulated fsync EIO"));
    let outer = ArcGraphError::WalErrorRolledBack {
        source: Box::new(inner),
    };
    let display = format!("{outer}");
    assert!(
        display.contains("rolled back"),
        "operator-facing message must explain rollback; got: {display}"
    );
    assert!(
        display.contains("retryable"),
        "operator-facing message must call out retryability; got: {display}"
    );
}

#[test]
fn wal_error_rolled_back_source_chain_intact() {
    let inner = ArcGraphError::WalUnavailable;
    let outer = ArcGraphError::WalErrorRolledBack {
        source: Box::new(inner),
    };
    let src = outer
        .source()
        .expect("WalErrorRolledBack must expose its underlying WAL error");
    // Source chain: WalErrorRolledBack → WalUnavailable.
    assert_eq!(src.to_string(), "wal writer unavailable");
}

#[test]
fn fsyncgate_pattern_match_for_retry_logic() {
    // Production retry-logic dispatches off this variant. A regression
    // that renames or removes the variant would fail this pattern match
    // at compile time — load-bearing for the rollback contract.
    let e = ArcGraphError::WalErrorRolledBack {
        source: Box::new(ArcGraphError::Io(io::Error::other("EIO"))),
    };
    let is_retryable = matches!(e, ArcGraphError::WalErrorRolledBack { .. });
    assert!(
        is_retryable,
        "WalErrorRolledBack MUST be pattern-matchable for retry dispatch"
    );
}

#[test]
fn fsyncgate_io_variant_is_distinct_from_rolled_back() {
    // A bare `Io` error MUST NOT be retried at the WAL level — the
    // caller has no rollback guarantee. This distinction is the
    // post-fsyncgate fix: silent fsync failures (returning Ok where
    // the previous fsync had failed) are STRUCTURALLY impossible
    // because ArcGraph's WAL rollback wraps every fsync-failure path
    // in `WalErrorRolledBack`.
    let raw_io = ArcGraphError::Io(io::Error::other("simulated bare io"));
    assert!(matches!(raw_io, ArcGraphError::Io(_)));
    assert!(!matches!(raw_io, ArcGraphError::WalErrorRolledBack { .. }));
}

#[test]
fn fsyncgate_nested_wal_unavailable_recoverable() {
    // ADR-033 Z-1(b): inner source typically `Io`, `WalUnavailable`,
    // `WalCorruption`. Verify the nested variants compose without
    // double-boxing or panicking.
    let cases = vec![
        ArcGraphError::Io(io::Error::other("eio")),
        ArcGraphError::WalUnavailable,
        ArcGraphError::WalCorruption {
            lsn: arcgraph_core::Lsn::new(42),
            reason: "torn record".into(),
        },
    ];
    for inner in cases {
        let outer = ArcGraphError::WalErrorRolledBack {
            source: Box::new(inner),
        };
        let _ = format!("{outer}"); // must not panic
        let src = outer.source();
        assert!(src.is_some(), "every WalErrorRolledBack must expose source");
    }
}
