//! W26-γ-2 D3 — Comprehensive error-translation tests for
//! `arcgraph-core::error::ArcGraphError`.
//!
//! Per ADR-134 forward-binding (test:prod ratio uplift) + W26-γ-2 D3
//! spec. The existing inline `#[cfg(test)]` block in `src/error.rs`
//! covers individual variant Display strings; this integration-test
//! file adds the cross-cutting invariants:
//!
//! - Every variant round-trips through `Debug` + `Display` + `Error`.
//! - `std::error::Error::source()` correctly delegates when the
//!   variant has a wrapped source.
//! - The taxonomy is fully matchable against `#[non_exhaustive]` (so
//!   downstream consumers can pattern-match on the variants we
//!   document as load-bearing).
//! - Operator-facing message conventions (hex-formatted magics,
//!   structured "delta=" field, "retryable" hint, etc.) are stable.

use std::error::Error as StdError;
use std::io;

use arcgraph_core::error::{ArcGraphError, Result};
use arcgraph_core::ids::{Lsn, PageId};

// ────────────────────── Display non-empty ──────────────────────

#[test]
fn every_variant_has_nonempty_display() {
    let variants: Vec<ArcGraphError> = sample_variants();
    for v in variants {
        let d = format!("{v}");
        assert!(!d.is_empty(), "Display empty for {v:?}");
    }
}

// ────────────────────── Debug + Display + Error trait round-trip ──────────────────────

#[test]
fn every_variant_has_debug_and_error_trait_impl() {
    let variants: Vec<ArcGraphError> = sample_variants();
    for v in variants {
        let _dbg = format!("{v:?}");
        let _disp = format!("{v}");
        // `&dyn std::error::Error` upcast must succeed for every
        // variant — the workspace's `?`-propagation depends on it.
        let _: &dyn StdError = &v;
    }
}

// ────────────────────── Source delegation ──────────────────────

#[test]
fn io_variant_exposes_underlying_source() {
    let io = io::Error::other("underlying");
    let e = ArcGraphError::Io(io);
    let src = e.source().expect("Io variant must expose its source");
    assert!(src.to_string().contains("underlying"));
}

#[test]
fn wal_error_rolled_back_exposes_underlying_source() {
    let inner = ArcGraphError::WalUnavailable;
    let outer = ArcGraphError::WalErrorRolledBack {
        source: Box::new(inner),
    };
    let s = outer.source().expect("rolled-back must expose source");
    assert_eq!(s.to_string(), "wal writer unavailable");
}

#[test]
fn non_io_variant_has_no_source() {
    let e = ArcGraphError::BufferPoolExhausted;
    // BufferPoolExhausted has no #[source]; it must have None.
    assert!(e.source().is_none());
}

// ────────────────────── ? operator translation from std::io ──────────────────────

#[test]
fn question_mark_lifts_io_error_to_arcgraph_error() {
    fn touch() -> Result<()> {
        let _ = std::fs::metadata("/definitely/does/not/exist/arcgraph-w26-test")?;
        Ok(())
    }
    let err = touch().unwrap_err();
    assert!(matches!(err, ArcGraphError::Io(_)));
}

// ────────────────────── Message-stability invariants ──────────────────────

#[test]
fn bad_page_magic_displays_lowercase_hex_canonical() {
    // The Display fmt uses `{:08x}` — must be 8-digit lowercase hex.
    let e = ArcGraphError::BadPageMagic {
        got: 0xDEAD_BEEF,
        expected: 0x4743_5241,
    };
    let s = format!("{e}");
    assert!(s.contains("0xdeadbeef"), "got: {s}");
    assert!(s.contains("0x47435241"), "got: {s}");
}

#[test]
fn vector_inconsistency_carries_delta_field() {
    let e = ArcGraphError::VectorIndexInconsistency {
        tenant_id: 1,
        index_id: 2,
        snapshot_lsn: 100,
        observed_vectors_count: 1024,
        observed_graph_node_count: 1023,
        wal_replay_high_lsn: 110,
        delta: 1,
    };
    let s = format!("{e}");
    // Operator-facing structured fields.
    for required in &[
        "tenant=1",
        "index=2",
        "snapshot_lsn=100",
        "vectors_count=1024",
        "graph_node_count=1023",
        "delta=1",
        "wal_replay_high_lsn=110",
        "bootstrap_from_mvcc",
    ] {
        assert!(s.contains(required), "missing {required:?} in: {s}");
    }
}

#[test]
fn wal_decryption_failure_names_key_version() {
    let e = ArcGraphError::WalDecryptionFailed {
        lsn: Lsn::new(42),
        key_version: 3,
        reason: "tag mismatch".to_owned(),
    };
    let s = format!("{e}");
    assert!(s.contains("key_version 3"), "got: {s}");
    // Operator hint to look at the historical-key-version path.
    assert!(
        s.contains("historical key version"),
        "operator hint missing: {s}"
    );
}

#[test]
fn unsafe_mount_options_names_mountpoint_and_reason() {
    let e = ArcGraphError::UnsafeMountOptions {
        mountpoint: "/data".to_owned(),
        reason: "ext4 with nobarrier".to_owned(),
    };
    let s = format!("{e}");
    assert!(s.contains("/data"), "got: {s}");
    assert!(s.contains("ext4"), "got: {s}");
}

#[test]
fn unrecoverable_orphans_carries_manual_recovery_hint() {
    let e = ArcGraphError::UnrecoverableOrphans {
        orphan_count: 7,
        reason: "bootstrap failed".to_owned(),
    };
    let s = format!("{e}");
    assert!(s.contains("manual recovery required"), "got: {s}");
}

// ────────────────────── Non-exhaustive matching ──────────────────────

#[test]
fn match_on_load_bearing_variants_compiles() {
    // Pattern-match on the variants that downstream code matches on.
    // If a refactor renames or removes any of these, this test fails
    // and surfaces the breaking-change at PR time.
    let cases = sample_variants();
    for e in cases {
        match e {
            ArcGraphError::Io(_) => {}
            ArcGraphError::PageCorruption { .. } => {}
            ArcGraphError::WalCorruption { .. } => {}
            ArcGraphError::WalRecordTypeReserved { .. } => {}
            ArcGraphError::MvccConflict { .. } => {}
            ArcGraphError::InvalidRecordLength { .. } => {}
            ArcGraphError::UnknownPageType(_) => {}
            ArcGraphError::UnsupportedRecordVersion(_) => {}
            ArcGraphError::BadPageMagic { .. } => {}
            ArcGraphError::BufferPoolExhausted => {}
            ArcGraphError::TransactionAborted { .. } => {}
            ArcGraphError::WalUnavailable => {}
            ArcGraphError::WalFormatMismatch { .. } => {}
            ArcGraphError::WalBadMagic { .. } => {}
            ArcGraphError::UnsafeMountOptions { .. } => {}
            ArcGraphError::WalErrorRolledBack { .. } => {}
            ArcGraphError::UnrecoverableOrphans { .. } => {}
            ArcGraphError::VectorIndexInconsistency { .. } => {}
            ArcGraphError::WalDecryptionFailed { .. } => {}
            ArcGraphError::PageDecryptionFailed { .. } => {}
            // `#[non_exhaustive]` requires a catch-all for downstream
            // builders; we name a wildcard for compile coverage.
            _ => {}
        }
    }
}

// ────────────────────── Display does not panic ──────────────────────

#[test]
fn display_never_panics_with_zero_lsn_and_zero_page() {
    // Edge-case payloads at sentinels.
    let cases = vec![
        ArcGraphError::PageCorruption {
            page_id: PageId::ZERO,
            reason: String::new(),
        },
        ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: String::new(),
        },
        ArcGraphError::WalDecryptionFailed {
            lsn: Lsn::ZERO,
            key_version: 0,
            reason: String::new(),
        },
    ];
    for c in cases {
        let _ = format!("{c}");
    }
}

#[test]
fn display_never_panics_with_max_lsn_and_max_page() {
    let cases = vec![
        ArcGraphError::PageCorruption {
            page_id: PageId::MAX,
            reason: "x".to_owned(),
        },
        ArcGraphError::WalCorruption {
            lsn: Lsn::MAX,
            reason: "x".to_owned(),
        },
    ];
    for c in cases {
        let _ = format!("{c}");
    }
}

// ────────────────────── Helper: sample every variant ──────────────────────

fn sample_variants() -> Vec<ArcGraphError> {
    vec![
        ArcGraphError::Io(io::Error::other("io")),
        ArcGraphError::PageCorruption {
            page_id: PageId::new(1),
            reason: "r".to_owned(),
        },
        ArcGraphError::WalCorruption {
            lsn: Lsn::new(2),
            reason: "r".to_owned(),
        },
        ArcGraphError::WalRecordTypeReserved { byte: 13 },
        ArcGraphError::MvccConflict {
            target: "vertex 1".to_owned(),
        },
        ArcGraphError::InvalidRecordLength {
            got: 10,
            expected: 20,
        },
        ArcGraphError::UnknownPageType(7),
        ArcGraphError::UnsupportedRecordVersion(9),
        ArcGraphError::BadPageMagic {
            got: 0xDEAD_BEEF,
            expected: 0x4743_5241,
        },
        ArcGraphError::BufferPoolExhausted,
        ArcGraphError::TransactionAborted {
            reason: "r".to_owned(),
        },
        ArcGraphError::WalUnavailable,
        ArcGraphError::WalFormatMismatch {
            found_version: 999,
            supported_versions: &[1],
        },
        ArcGraphError::WalBadMagic {
            got: *b"XXXX",
            expected: *b"AGWL",
        },
        ArcGraphError::UnsafeMountOptions {
            mountpoint: "/m".to_owned(),
            reason: "r".to_owned(),
        },
        ArcGraphError::WalErrorRolledBack {
            source: Box::new(ArcGraphError::WalUnavailable),
        },
        ArcGraphError::UnrecoverableOrphans {
            orphan_count: 1,
            reason: "r".to_owned(),
        },
        ArcGraphError::VectorIndexInconsistency {
            tenant_id: 1,
            index_id: 2,
            snapshot_lsn: 3,
            observed_vectors_count: 4,
            observed_graph_node_count: 5,
            wal_replay_high_lsn: 6,
            delta: -1,
        },
        ArcGraphError::WalDecryptionFailed {
            lsn: Lsn::new(7),
            key_version: 1,
            reason: "r".to_owned(),
        },
        ArcGraphError::PageDecryptionFailed {
            page_id: PageId::new(99),
            key_version: 5,
            reason: "r".to_owned(),
        },
    ]
}

// ────────────────────── Stable Display prefix (audit-grade) ──────────────────────

#[test]
fn display_prefix_is_stable_for_grep_audit() {
    // Operations + audit greps depend on the message prefixes.
    let cases = vec![
        (
            ArcGraphError::WalErrorRolledBack {
                source: Box::new(ArcGraphError::WalUnavailable),
            },
            "wal fsync failed",
        ),
        (
            ArcGraphError::UnsafeMountOptions {
                mountpoint: "/x".to_owned(),
                reason: "y".to_owned(),
            },
            "unsafe mount options",
        ),
        (
            ArcGraphError::UnrecoverableOrphans {
                orphan_count: 1,
                reason: "r".to_owned(),
            },
            "wal replay halted",
        ),
        (
            ArcGraphError::VectorIndexInconsistency {
                tenant_id: 1,
                index_id: 2,
                snapshot_lsn: 0,
                observed_vectors_count: 0,
                observed_graph_node_count: 0,
                wal_replay_high_lsn: 0,
                delta: 0,
            },
            "wal replay halted",
        ),
        (
            ArcGraphError::WalDecryptionFailed {
                lsn: Lsn::ZERO,
                key_version: 0,
                reason: "r".to_owned(),
            },
            "wal decryption failed",
        ),
        (
            ArcGraphError::PageDecryptionFailed {
                page_id: PageId::ZERO,
                key_version: 0,
                reason: "r".to_owned(),
            },
            "page decryption failed",
        ),
    ];
    for (e, prefix) in cases {
        let s = format!("{e}");
        assert!(
            s.starts_with(prefix),
            "expected prefix {prefix:?}, got: {s}"
        );
    }
}

// ────────────────────── Result type alias resolves ──────────────────────

#[test]
fn result_alias_resolves() {
    let ok: Result<u32> = Ok(7);
    let err: Result<u32> = Err(ArcGraphError::WalUnavailable);
    match ok {
        Ok(v) => assert_eq!(v, 7),
        Err(_) => panic!("ok branch must be Ok"),
    }
    assert!(err.is_err());
}
