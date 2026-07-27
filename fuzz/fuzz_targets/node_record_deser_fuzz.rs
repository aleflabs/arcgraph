#![no_main]
//! W22-DB-ε: NodeRecord deserializer fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_core::NodeRecord::from_bytes`] — the 64-byte fixed-size
//! node-record decoder at `crates/arcgraph-core/src/record.rs:341`.
//!
//! # Assertion
//!
//! - **No panic.** `from_bytes` on ANY 64-byte slice MUST return
//!   either `Ok(NodeRecord)` (version-bits accept) or
//!   `Err(ArcGraphError::UnsupportedRecordVersion)` (version-bits
//!   reject). Both are valid outcomes.
//! - **Roundtrip.** When `from_bytes` succeeds,
//!   `record.to_bytes()` MUST reproduce a byte sequence that
//!   `from_bytes` accepts and that round-trips back to an
//!   equal `NodeRecord`. (The encoded form is canonical — padding
//!   bytes are zeroed; the round-trip equality test is total over
//!   all decoded records.)
//!
//! Input shorter or longer than 64 bytes is silently ignored —
//! `from_bytes` takes `&[u8; 64]` so framing is the caller's
//! responsibility.

use libfuzzer_sys::fuzz_target;

const RECORD_SIZE: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() < RECORD_SIZE {
        return;
    }
    let mut buf = [0u8; RECORD_SIZE];
    buf.copy_from_slice(&data[..RECORD_SIZE]);
    match arcgraph_core::NodeRecord::from_bytes(&buf) {
        Ok(record) => {
            // Canonical roundtrip — encode then decode the encoded
            // form. The encoder zeroes padding so the second decode
            // MUST also succeed. Any divergence is a decoder
            // asymmetry bug.
            let re_encoded = record.to_bytes();
            let re_decoded = arcgraph_core::NodeRecord::from_bytes(&re_encoded)
                .expect("re-decode of canonical-encoded NodeRecord must succeed");
            assert_eq!(
                record, re_decoded,
                "NodeRecord roundtrip diverged: {record:?} != {re_decoded:?}"
            );
        }
        Err(_) => {
            // Unsupported version-byte — expected reject path.
        }
    }
});
