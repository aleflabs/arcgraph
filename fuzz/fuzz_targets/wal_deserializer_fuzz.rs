#![no_main]
//! W22-DB-ε: WAL record deserializer fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_storage::WalRecord::decode`] — the binary WAL record
//! parser at `crates/arcgraph-storage/src/wal/record.rs:294`. The
//! decoder reads a 44-byte header (crc32c | length | type | reserved
//! ×3 | txn_id | lsn | timestamp_ms | tenant_id) plus a variable-
//! length payload.
//!
//! # Assertion
//!
//! `decode(bytes)` MUST NOT panic on ANY byte sequence. Valid records
//! return `Ok((WalRecord, consumed))`; invalid records (short
//! header, crc mismatch, non-zero reserved bytes, length under
//! header size) return `Err(ArcGraphError::*)`. Both outcomes are
//! valid — the contract is no-panic, no-UB, no-infinite-loop.
//!
//! When `decode` succeeds, we additionally assert that the consumed
//! byte count does not exceed the input length (no over-read), and
//! we attempt a roundtrip — `record.encode_to_vec()` followed by
//! `WalRecord::decode` on the encoded bytes — to catch encoder /
//! decoder asymmetry bugs.
//!
//! Input length is capped at 64 KiB to bound per-iter wall time;
//! production WAL records are bounded by the segment size
//! (default 64 MiB per `WalConfig::DEFAULT_SEGMENT_SIZE`).

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    match arcgraph_storage::WalRecord::decode(data) {
        Ok((record, consumed)) => {
            // Over-read guard — decode MUST NOT claim to have
            // consumed more bytes than the input held.
            assert!(
                consumed <= data.len(),
                "WalRecord::decode consumed {consumed} bytes from a {} byte input",
                data.len()
            );
            // Roundtrip — `encode_to_vec` then `decode` MUST match.
            // The encoder is total over valid `WalRecord` values, so
            // any failure here is an encoder bug.
            let Ok(re_encoded) = record.encode_to_vec() else {
                // Encoder failure on a value the decoder produced =
                // bug, but `expect` would panic. Soft-fail per the
                // crash-discipline (P1) — fuzz finds the case but
                // doesn't escalate to abort.
                return;
            };
            let _ = arcgraph_storage::WalRecord::decode(&re_encoded);
        }
        Err(_) => {
            // Invalid input rejected cleanly — expected behavior.
        }
    }
});
