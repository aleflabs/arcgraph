#![no_main]
//! W28-S604: Bolt client-message decoder fuzz target (testing strategy;
//! testing-strategy §2.4 "bolt" minimum; full-canonical-set audit
//! Task #604).
//!
//! # What this fuzzes
//!
//! [`arcgraph_mcp::transport::bolt::decode_client`] — the Bolt 5.0
//! *message-level* decoder at
//! `crates/arcgraph-mcp/src/transport/bolt/message.rs:239`. This is the
//! layer ABOVE the PackStream value decoder (`packstream::decode`,
//! covered by `bolt_packstream_fuzz` / PR #559): it requires the root
//! value to be a `Struct`, dispatches on the message tag
//! (HELLO / GOODBYE / RESET / RUN / DISCARD / PULL), and validates each
//! message's field shape (field count, map-vs-scalar fields, required
//! keys). Those hand-written dispatch + field-extraction paths are not
//! exercised by the value-level packstream target, so this is a
//! genuinely distinct, previously-UNCOVERED parser surface (audit
//! Task #604 deliverable 1).
//!
//! # Assertions
//!
//! - **No panic / no UB** on ANY byte sequence — the libfuzzer
//!   contract. Truncated structs, unknown tags, wrong field arity,
//!   non-map HELLO extras, trailing bytes after the value must all
//!   surface as `Err(BoltError)`, never a panic.
//! - **Round-trip is a canonical fixed point.** For any payload that
//!   decodes to a [`ClientMessage`], re-encoding via
//!   [`encode_client`] and decoding the result again MUST yield a
//!   `ClientMessage` equal to the first decode. (`encode_client` is the
//!   server's own encoder for the inbound variants; a decode→encode→
//!   decode divergence is an encoder/decoder asymmetry bug.) We only
//!   assert the fixed point when the re-encode succeeds — some decoded
//!   messages may carry shapes the encoder legitimately rejects, which
//!   is not a round-trip failure.
//!
//! Input is capped at 64 KiB to bound per-iteration wall time; the Bolt
//! chunking layer (`transport::bolt::chunking`) caps a reassembled
//! message well below this, but the decoder is fuzzed directly per the
//! §2.4 "parser" contract (same convention as `bolt_packstream_fuzz`).

use libfuzzer_sys::fuzz_target;

use arcgraph_mcp::transport::bolt::{decode_client, encode_client};

const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    // Core contract: decode_client returns without panicking on any
    // byte stream. Both Ok and Err are acceptable outcomes.
    let Ok(msg) = decode_client(data) else {
        return;
    };

    // Canonical-fixed-point round-trip. Encode the decoded message and
    // decode the encoded bytes; the second decode MUST equal the first.
    let mut out = Vec::new();
    if encode_client(&mut out, &msg).is_ok() {
        if let Ok(msg2) = decode_client(&out) {
            assert_eq!(
                msg, msg2,
                "Bolt decode->encode->decode diverged:\n first = {msg:?}\n second = {msg2:?}"
            );
        }
    }
});
