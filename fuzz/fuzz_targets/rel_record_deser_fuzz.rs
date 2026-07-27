#![no_main]
//! W22-DB-ε: RelRecord deserializer fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_core::RelRecord::from_bytes`] — the 96-byte fixed-size
//! relationship-record decoder at
//! `crates/arcgraph-core/src/record.rs:448`. Per ADR-014 §"Wire
//! shape" the byte layout is part of the v1.0 contract and feeds the
//! storage / replication / cross-substrate paths.
//!
//! # Assertion
//!
//! - **No panic.** `from_bytes` on ANY 96-byte slice MUST return
//!   either `Ok(RelRecord)` or `Err(ArcGraphError::UnsupportedRecordVersion)`.
//! - **Roundtrip.** When `from_bytes` succeeds,
//!   `record.to_bytes()` MUST reproduce a byte sequence that
//!   `from_bytes` accepts AND round-trips back to an equal
//!   `RelRecord`. Padding bytes are canonically zeroed; the
//!   equality is exact.
//! - **NaN weight tolerance.** `RelRecord::weight` is `f32`; bit
//!   patterns including NaN must round-trip without panicking.
//!   `RelRecord: Eq` is implemented manually (the `Eq` impl is part
//!   of the type's contract per ADR-014); the equality assertion
//!   may legitimately fail on NaN weights because `f32 != f32` when
//!   the value is NaN. We special-case NaN to avoid spurious
//!   failures while still asserting the bit-pattern roundtrip via
//!   `weight.to_bits()`.

use libfuzzer_sys::fuzz_target;

const RECORD_SIZE: usize = 96;

fuzz_target!(|data: &[u8]| {
    if data.len() < RECORD_SIZE {
        return;
    }
    let mut buf = [0u8; RECORD_SIZE];
    buf.copy_from_slice(&data[..RECORD_SIZE]);
    match arcgraph_core::RelRecord::from_bytes(&buf) {
        Ok(record) => {
            let re_encoded = record.to_bytes();
            let re_decoded = arcgraph_core::RelRecord::from_bytes(&re_encoded)
                .expect("re-decode of canonical-encoded RelRecord must succeed");
            // f32 NaN is not Eq with itself — special-case via
            // bit-pattern equality to avoid a spurious roundtrip
            // failure on NaN weights.
            if record.weight.is_nan() {
                assert_eq!(
                    record.weight.to_bits(),
                    re_decoded.weight.to_bits(),
                    "RelRecord weight bit-pattern diverged on NaN: {:#x} != {:#x}",
                    record.weight.to_bits(),
                    re_decoded.weight.to_bits()
                );
                // All non-weight fields MUST match.
                assert_eq!(record.id, re_decoded.id);
                assert_eq!(record.type_id, re_decoded.type_id);
                assert_eq!(record.flags, re_decoded.flags);
                assert_eq!(record.src_id, re_decoded.src_id);
                assert_eq!(record.dst_id, re_decoded.dst_id);
                assert_eq!(record.property_ref, re_decoded.property_ref);
                assert_eq!(record.inline_u32a, re_decoded.inline_u32a);
                assert_eq!(record.inline_u32b, re_decoded.inline_u32b);
                assert_eq!(record.created_lsn, re_decoded.created_lsn);
                assert_eq!(record.expired_lsn, re_decoded.expired_lsn);
            } else {
                assert_eq!(
                    record, re_decoded,
                    "RelRecord roundtrip diverged: {record:?} != {re_decoded:?}"
                );
            }
        }
        Err(_) => {
            // Unsupported version-byte — expected reject path.
        }
    }
});
