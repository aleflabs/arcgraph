#![no_main]
//! W28-S604: WAL segment-header decoder fuzz target (testing strategy;
//! full-canonical-set audit Task #604).
//!
//! # What this fuzzes
//!
//! [`arcgraph_storage::wal::SegmentHeader::decode`] — the 8-byte WAL
//! *segment* header parser at
//! `crates/arcgraph-storage/src/wal/segment.rs:159`. This is a distinct
//! surface from the WAL *record* decoder (`WalRecord::decode`, covered
//! by `wal_deserializer_fuzz`): the segment header is the first 8 bytes
//! of every on-disk WAL segment file and gates recovery — it validates
//! the segment magic (`WAL\x01`-class), the format version against
//! `SUPPORTED_WAL_FORMAT_VERSIONS`, and that the 2 reserved bytes are
//! zero. A corrupt/forged segment header on the recovery path is a
//! genuine untrusted-input surface, previously UNCOVERED (audit
//! Task #604 deliverable 1).
//!
//! # Assertions
//!
//! - **No panic.** `decode(bytes)` MUST NOT panic on ANY byte slice
//!   (including the empty slice — the decoder length-checks internally
//!   and returns `Err(WalFormatMismatch)` rather than indexing). Valid
//!   headers return `Ok(SegmentHeader)`; bad magic / unsupported
//!   version / non-zero reserved bytes / short input return
//!   `Err(ArcGraphError::*)`. Both outcomes are valid.
//! - **Canonical fixed point.** When `decode` succeeds, `encode()` then
//!   `decode` MUST reproduce an equal `SegmentHeader` (the encoding is
//!   canonical — reserved bytes are zeroed by the encoder, so a decoded
//!   header always re-encodes to an accepted byte sequence).

use libfuzzer_sys::fuzz_target;

use arcgraph_storage::wal::SegmentHeader;

fuzz_target!(|data: &[u8]| {
    // `decode` is total over `&[u8]` — it length-checks internally, so
    // no input-size guard is required (unlike the fixed-`&[u8; N]`
    // record decoders).
    match SegmentHeader::decode(data) {
        Ok(header) => {
            let re_encoded = header.encode();
            let re_decoded = SegmentHeader::decode(&re_encoded)
                .expect("re-decode of canonical-encoded SegmentHeader must succeed");
            assert_eq!(
                header, re_decoded,
                "SegmentHeader roundtrip diverged: {header:?} != {re_decoded:?}"
            );
        }
        Err(_) => {
            // Bad magic / unsupported version / non-zero reserved /
            // short input rejected cleanly — expected behaviour.
        }
    }
});
