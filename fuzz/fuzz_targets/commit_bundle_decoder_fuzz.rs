#![no_main]
//! #1287 (security hardening; follow-up to #1285): CommitBundle payload
//! decoder fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_storage::wal::bundle::decode_commit_bundle_for_version`] —
//! the per-version `CommitBundle` payload dispatcher at
//! `crates/arcgraph-storage/src/wal/bundle.rs:1257`. It reads a
//! `format_version` (from the owning [`arcgraph_storage::wal::SegmentHeader`])
//! and dispatches to `decode_commit_bundle_v1..v8`; an unknown version
//! returns [`arcgraph_core::ArcGraphError::WalFormatMismatch`].
//!
//! These decoders parse **untrusted on-disk bytes** on two hot paths:
//!
//! 1. **crash recovery** — the WAL replay executor decodes each segment's
//!    committed `CommitBundle` payloads while rebuilding state, and
//! 2. **spill reload** — `load_one_spill_file` re-reads spilled bundles
//!    with the same version-dispatched decoder.
//!
//! A malformed / truncated section length, an over-long entry count, a bad
//! id byte, or a non-UTF-8 principal must NOT panic the recovering process
//! (that is a denial-of-service on startup) — it must be rejected with a
//! structured error. Prior to this target the bundle payload decoders had
//! NO fuzz coverage: `wal_deserializer_fuzz` (`WalRecord::decode`) and
//! `wal_segment_fuzz` (`SegmentHeader::decode`) stop at the record/segment
//! framing and never reach the bundle payload parsers.
//!
//! The v5..v8 decoders are the *sectioned* parsers (mvcc / staged_pages /
//! sidechannel / allocator_advances / vector_pages / idempotency_bindings /
//! **acl_grants**). The v8 `acl_grants` tail is the decode-side sibling of
//! the #1285 encode-side omission — it reads a `u32` count and per-entry
//! `u32` grant counts + UTF-8 principals straight from the payload, exactly
//! the shape (over-long count → huge `with_capacity`, length overrun,
//! invalid UTF-8) a fuzzer should probe. This target guarantees v8 (and
//! therefore `acl_grants`) is reached on every input.
//!
//! # Assertion
//!
//! **No panic.** For ANY byte sequence and ANY `format_version`,
//! `decode_commit_bundle_for_version` MUST return either
//! `Ok(DecodedCommitBundle)` (a well-formed payload) or a structured `Err`
//! (`WalCorruption` / decode error / `WalFormatMismatch` for unknown
//! versions). It must NOT `unwrap`-panic, index out of bounds, slice past
//! the payload, or overflow-panic on truncated / over-long / malformed
//! sections. We do NOT assert `Ok` — malformed input SHOULD `Err`; the
//! contract is purely no-panic (the closure returning normally), so a
//! panic anywhere in the decoder crashes the fuzzer = the finding.

use libfuzzer_sys::fuzz_target;

use arcgraph_core::ids::TenantId;
use arcgraph_storage::wal::bundle::decode_commit_bundle_for_version;

/// Cap per-iter input to bound wall time. A `CommitBundle` payload is
/// bounded in production by the WAL segment size; 64 KiB is ample to
/// exercise every section parser (each section length is a `u32`/`u16`
/// read against the *available* bytes, so the truncation / overrun logic
/// is fully reachable well under this cap).
const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }

    // ── Dimension 1: version driven from the first fuzz byte ──
    //
    // `*v as u16` ranges over 0..=255, which covers all 8 valid versions
    // (1..=8) AND many invalid ones (0, 9..=255 → WalFormatMismatch).
    // `rest` is the payload. This exercises the full dispatch table plus
    // the unknown-version reject path in one shot.
    if let Some((v, rest)) = data.split_first() {
        let version = u16::from(*v);
        // The decoder MUST NOT panic on any (payload, version) pair — a
        // panic here escapes the closure and crashes the fuzzer = finding.
        let _ = decode_commit_bundle_for_version(rest, version, TenantId::DEFAULT);
    }

    // ── Dimension 2: fixed v7 + v8 over the WHOLE input ──
    //
    // When the first-byte drive lands on a short/invalid-version payload
    // it may not reach deep section-parse coverage. Feeding the entire
    // `data` as payload with the two richest sectioned versions guarantees
    // the v7 (idempotency tail) and v8 (`acl_grants` tail — the #1285
    // decode sibling) parsers are reached on every input, maximizing
    // coverage of the over-long-count / length-overrun / non-UTF-8
    // principal branches. Both calls contract on no-panic only.
    let _ = decode_commit_bundle_for_version(data, 7, TenantId::DEFAULT);
    let _ = decode_commit_bundle_for_version(data, 8, TenantId::DEFAULT);
});
