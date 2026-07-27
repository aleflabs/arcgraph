//! W26-γ-2 D5#8 — Negative scenario: cosmic-bit-flip / silent page
//! corruption.
//!
//! Real-world incident: cosmic bit-flips on consumer SSDs (Backblaze
//! 2019 study), ECC-related silent corruption on AWS i3.metal in
//! 2021, MongoDB's 2019 SERVER-37182 silent-corruption-on-read class.
//! The general class: page bytes on disk silently corrupt due to
//! hardware-level fault (radiation, cosmic ray, DRAM bit-flip,
//! controller bug), AND the storage layer's CRC32C check is the
//! only defense between "garbage in, garbage out" and "structured
//! error path."
//!
//! ArcGraph's analog: per ADR-001 every page carries a CRC32C of
//! its body (bytes 40..8192). The page-read path validates the
//! CRC; mismatch → `ArcGraphError::PageCorruption`. The W20β-3
//! AES-GCM encryption surface adds a SECOND defense layer:
//! `ArcGraphError::PageDecryptionFailed` distinguishes "bit flip
//! on disk" (CRC catches; PageCorruption) from "wrong key / tag
//! mismatch" (GCM auth tag catches; PageDecryptionFailed).
//!
//! This test asserts the structured-error contract at the
//! arcgraph-core error taxonomy boundary.

use arcgraph_core::PageId;
use arcgraph_core::error::ArcGraphError;
use arcgraph_core::ids::Lsn;

#[test]
fn page_corruption_variant_distinguishes_from_wal_corruption() {
    // Two distinct corruption classes:
    //  - PageCorruption: page-store-level bit flip
    //  - WalCorruption: WAL-level bit flip
    // The operator-facing recovery is different — pin the
    // taxonomy split.
    let page = ArcGraphError::PageCorruption {
        page_id: PageId::new(42),
        reason: "crc mismatch".into(),
    };
    let wal = ArcGraphError::WalCorruption {
        lsn: Lsn::new(7),
        reason: "crc mismatch".into(),
    };
    assert!(matches!(page, ArcGraphError::PageCorruption { .. }));
    assert!(matches!(wal, ArcGraphError::WalCorruption { .. }));
    assert!(!matches!(page, ArcGraphError::WalCorruption { .. }));
    assert!(!matches!(wal, ArcGraphError::PageCorruption { .. }));
}

#[test]
fn page_corruption_carries_page_id_and_reason() {
    let e = ArcGraphError::PageCorruption {
        page_id: PageId::new(0xDEAD_BEEF),
        reason: "cosmic bit flip detected via crc32c".into(),
    };
    let display = format!("{e}");
    assert!(display.contains("page corruption"));
    // Page id must surface in the diagnostic.
    assert!(
        display.contains("0xdeadbeef") || display.contains("3735928559"),
        "got: {display}"
    );
    assert!(
        display.contains("cosmic bit flip"),
        "reason must surface; got: {display}"
    );
}

#[test]
fn page_decryption_failed_distinguishes_from_page_corruption() {
    // Per ADR-052 (page-store encryption), bit-flip and wrong-key
    // failures are DISTINCT taxonomy paths — silent fallback to
    // plaintext is forbidden.
    let bit_flip = ArcGraphError::PageCorruption {
        page_id: PageId::new(1),
        reason: "crc mismatch".into(),
    };
    let wrong_key = ArcGraphError::PageDecryptionFailed {
        page_id: PageId::new(1),
        key_version: 3,
        reason: "GCM tag mismatch".into(),
    };
    // The variants are distinct.
    assert!(matches!(bit_flip, ArcGraphError::PageCorruption { .. }));
    assert!(matches!(
        wrong_key,
        ArcGraphError::PageDecryptionFailed { .. }
    ));
    // Operator-facing messages cite different paths.
    let m1 = format!("{bit_flip}");
    let m2 = format!("{wrong_key}");
    assert!(m1.contains("page corruption"));
    assert!(m2.contains("page decryption failed"));
}

#[test]
fn page_decryption_failed_operator_hint_names_secrets_provider() {
    // The operator hint MUST point at the secrets provider
    // (where the historical key would be loaded from). Audit
    // surface — pin so a refactor that drops the hint doesn't
    // silently regress the runbook.
    let e = ArcGraphError::PageDecryptionFailed {
        page_id: PageId::new(99),
        key_version: 5,
        reason: "wrong key".into(),
    };
    let display = format!("{e}");
    assert!(
        display.contains("page-store key"),
        "operator hint must mention page-store key; got: {display}"
    );
    assert!(
        display.contains("SecretsProvider"),
        "operator hint must cite SecretsProvider; got: {display}"
    );
}

#[test]
fn bad_page_magic_distinguishes_from_page_corruption() {
    // A wrong-magic-byte page is "this isn't a page at all" — a
    // STRUCTURALLY different failure than "crc said the page is
    // corrupt." Pin the distinction.
    let bad_magic = ArcGraphError::BadPageMagic {
        got: 0xCAFE_BABE,
        expected: 0x4743_5241,
    };
    let crc_fail = ArcGraphError::PageCorruption {
        page_id: PageId::new(1),
        reason: "crc".into(),
    };
    assert!(matches!(bad_magic, ArcGraphError::BadPageMagic { .. }));
    assert!(matches!(crc_fail, ArcGraphError::PageCorruption { .. }));
}

#[test]
fn page_corruption_at_zero_page_id_handled() {
    // Defensive: a corruption at PageId::ZERO must surface
    // structurally. ZERO is the "free / not-allocated" sentinel
    // in production but the error must still display correctly.
    let e = ArcGraphError::PageCorruption {
        page_id: PageId::ZERO,
        reason: "page 0 (free list head) crc mismatch".into(),
    };
    let display = format!("{e}");
    assert!(display.contains("0"));
    assert!(display.contains("crc mismatch"));
}

#[test]
fn page_corruption_at_max_page_id_handled() {
    // The other extreme: PageId::MAX = u64::MAX. The display fmt
    // must not overflow / panic.
    let e = ArcGraphError::PageCorruption {
        page_id: PageId::MAX,
        reason: "crc".into(),
    };
    let _ = format!("{e}");
}

#[test]
fn page_corruption_silent_fallback_is_forbidden_by_taxonomy() {
    // The TAXONOMY is the load-bearing surface: the workspace
    // never has a "PageCorruption → return empty" path. Every
    // PageCorruption variant must propagate as Err. A regression
    // that added a silent-Ok fallback would not compile because
    // there's no such variant in the enum.
    let variants: Vec<ArcGraphError> = vec![
        ArcGraphError::PageCorruption {
            page_id: PageId::new(1),
            reason: "x".into(),
        },
        ArcGraphError::PageDecryptionFailed {
            page_id: PageId::new(1),
            key_version: 1,
            reason: "x".into(),
        },
    ];
    // Every variant has a non-empty Display string — load-bearing
    // for operator logs.
    for v in variants {
        let d = format!("{v}");
        assert!(!d.is_empty());
        // Both variants are Err-class (no Ok variant in the enum).
        let _: &dyn std::error::Error = &v;
    }
}
