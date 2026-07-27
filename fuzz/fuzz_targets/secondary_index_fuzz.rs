#![no_main]
//! W28-S604: secondary-index key/entry decoder fuzz target (testing strategy;
//! full-canonical-set audit Task #604).
//!
//! # What this fuzzes
//!
//! The on-disk binary codecs of the secondary B-tree at
//! `crates/arcgraph-index/src/secondary_btree.rs`:
//!
//! 1. [`arcgraph_index::secondary_btree::SecondaryKey::decode`] — the
//!    fixed 24-byte composite-key decoder (`tenant | label |
//!    property_key | tag | value`). It validates the `PropertyValue`
//!    variant tag (0=U32, 1=U64, 2=StringId; others rejected) and that
//!    the U32/StringId variants have zero-padded high bytes —
//!    hand-written validation distinct from the slotted-page frame
//!    parser (#559 `page_layout_fuzz`).
//! 2. [`arcgraph_index::secondary_btree::LeafEntry::decode`] — the
//!    64-byte leaf-entry decoder (`SecondaryKey` + inline `NodeId`
//!    slots + overflow-chain head). It transitively exercises
//!    `SecondaryKey::decode`.
//!
//! These read persisted index-page bytes; on-disk corruption feeds the
//! decoder directly, so they are genuine untrusted-input parsers. They
//! were previously UNCOVERED (audit Task #604 deliverable 1).
//!
//! # Size contract (mirrors `node_record_deser_fuzz`)
//!
//! Both `decode` functions take `&[u8]` but assert (`debug_assert_eq!`)
//! that the slice is EXACTLY their fixed size and then index directly —
//! i.e., the caller guarantees framing. `cargo fuzz` builds enable
//! debug-assertions, so the harness feeds exactly-sized windows
//! (`SecondaryKey::SIZE` = 24, `LEAF_ENTRY_SIZE` = 64); a wrong-length
//! call would be a caller-contract violation, not a parser bug.
//!
//! # Assertions
//!
//! - **No panic** on any 24-byte / 64-byte window.
//! - **Canonical round-trip.** When `decode` succeeds, `encode_into`
//!   followed by `decode` MUST reproduce an equal value. The encoders
//!   are total over decoded values (U64 decode reads only 7 bytes, so
//!   re-encode cannot overflow the 56-bit field), so any divergence is
//!   a codec-asymmetry bug.

use libfuzzer_sys::fuzz_target;

use arcgraph_index::secondary_btree::{LeafEntry, SecondaryKey, LEAF_ENTRY_SIZE};

fuzz_target!(|data: &[u8]| {
    // ── Dimension 1: 24-byte SecondaryKey decode + canonical round-trip.
    if data.len() >= SecondaryKey::SIZE {
        let key_window = &data[..SecondaryKey::SIZE];
        if let Ok(key) = SecondaryKey::decode(key_window) {
            let mut buf = [0u8; SecondaryKey::SIZE];
            key.encode_into(&mut buf)
                .expect("encode_into of a decoded SecondaryKey must succeed");
            let re = SecondaryKey::decode(&buf)
                .expect("re-decode of canonical-encoded SecondaryKey must succeed");
            assert_eq!(key, re, "SecondaryKey roundtrip diverged: {key:?} != {re:?}");
        }
    }

    // ── Dimension 2: 64-byte LeafEntry decode + canonical round-trip.
    //    Use a distinct window (offset by SIZE when available) so a
    //    single input can exercise both dimensions with independent
    //    bytes; fall back to the prefix when the input is short.
    if data.len() >= LEAF_ENTRY_SIZE {
        let off = if data.len() >= SecondaryKey::SIZE + LEAF_ENTRY_SIZE {
            SecondaryKey::SIZE
        } else {
            0
        };
        let entry_window = &data[off..off + LEAF_ENTRY_SIZE];
        if let Ok(entry) = LeafEntry::decode(entry_window) {
            let mut buf = [0u8; LEAF_ENTRY_SIZE];
            entry
                .encode_into(&mut buf)
                .expect("encode_into of a decoded LeafEntry must succeed");
            let re = LeafEntry::decode(&buf)
                .expect("re-decode of canonical-encoded LeafEntry must succeed");
            assert_eq!(
                entry, re,
                "LeafEntry roundtrip diverged: {entry:?} != {re:?}"
            );
        }
    }
});
