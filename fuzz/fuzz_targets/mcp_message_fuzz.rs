#![no_main]
//! W22-DB-ε: MCP JSON-RPC envelope fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_mcp::jsonrpc::decode_request`] — the JSON-RPC 2.0
//! request-envelope decoder at
//! `crates/arcgraph-mcp/src/jsonrpc.rs:330`. The function consumes a
//! `serde_json::Value` and returns `Result<JsonRpcRequest, MCPError>`.
//!
//! # Assertion
//!
//! `decode_request` MUST NOT panic on ANY JSON value. Valid envelopes
//! ({"jsonrpc": "2.0", "method": "…", "params": …, "id": …}) return
//! `Ok(req)`; envelopes with wrong version / missing method / wrong
//! types return `Err(MCPError::InvalidRequest)`. Both are valid.
//!
//! The harness first parses the fuzzer-provided byte buffer as JSON
//! (skipping the iter when JSON parse fails — that's
//! `serde_json::from_slice`'s contract, not the MCP layer's). Input
//! length is capped at 1 MiB — the MCP framer caps at
//! [`MAX_MESSAGE_BYTES`](arcgraph_mcp::jsonrpc::MAX_MESSAGE_BYTES) =
//! 16 MiB; 1 MiB is wall-budget-friendly for fuzz iters.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    // Best-effort JSON parse — if the byte stream isn't JSON,
    // there's nothing to fuzz at the MCP layer.
    let Ok(value) = serde_json::from_slice::<serde_json::Value>(data) else {
        return;
    };
    // The contract: decode_request returns without panicking on any
    // JSON value. Both Ok and Err are acceptable outcomes.
    let _ = arcgraph_mcp::jsonrpc::decode_request(value);
});
