#![no_main]
//! W26-γ-3 / ADR-136 §D-4 — arcql-smith libfuzzer harness.
//!
//! # What this fuzzes
//!
//! The `arcgraph_query::test_support::smith::Smith` generator at
//! `crates/arcgraph-query/src/test_support/smith.rs` (ADR-136). Per
//! ADR-136 §D-4 "Fuzz harness (nightly)" — this target is a workload
//! binary for nightly 24h libfuzzer runs.
//!
//! # Assertion (ADR-136 §D-3 invariants)
//!
//! The fuzz body asserts the SAME 5 invariants the smoke test checks:
//!
//! 1. **No-panic.** `Smith::gen_query()` MUST NOT panic for any
//!    fuzzer-supplied seed bytes.
//! 2. **No-panic on parse.** `arcgraph_query::parse(&q)` MUST NOT
//!    panic for the smith-generated `q`.
//! 3. **Round-trip.** If `parse(q) == Ok(s)`, then `parse(format!("{s}"))`
//!    MUST also return Ok and the two ASTs MUST be structurally equal.
//! 4. **Length-bounded output.** The generator MUST emit < 100 KiB per
//!    `gen_query()` call (depth=5, width=5 invariant).
//! 5. **Structured-error contract** — implicitly enforced by the
//!    `ParseError` type (the `Result<_, ParseError>` shape from
//!    `arcgraph_query::parse` rules out opaque-error contracts).
//!
//! Any panic anywhere in the assertion body is a libfuzzer crash; the
//! input bytes get checked in to `fuzz/artifacts/arcql_smith_fuzz/` as
//! the reproducer corpus per W22-DB-ε precedent.
//!
//! # Seed expansion
//!
//! libfuzzer hands the body `&[u8]`. We hash to a `u64` via FNV-1a
//! (avoid the std-lib `SipHash` non-determinism on differently-seeded
//! HashMaps) and use that as the smith seed. Same `&[u8]` → same seed
//! → same smith query (deterministic reproducibility per ADR-136 §D-1
//! #2).

use libfuzzer_sys::fuzz_target;

use arcgraph_query::test_support::smith::{Smith, SmithConfig};

/// FNV-1a hash — deterministic across stdlib versions; no environment
/// dependency. We don't need cryptographic strength here, only stable
/// `&[u8] → u64` seed expansion.
fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01B3);
    }
    h
}

fuzz_target!(|data: &[u8]| {
    // Seed expansion — empty input maps to seed=0 (the smith generator
    // handles this; no panic).
    let seed = fnv1a_64(data);

    let cfg = SmithConfig::stub();
    let mut smith = Smith::new(seed, cfg);

    // Invariant #1: smith generation MUST NOT panic.
    let q = smith.gen_query();

    // Invariant #4: length-bounded output.
    assert!(
        q.len() < 100 * 1024,
        "smith produced {}-byte query (cap 100 KiB) for seed={:#x}",
        q.len(),
        seed
    );

    // Invariant #2: parse MUST NOT panic.
    let parsed = arcgraph_query::parse(&q);

    // Invariant #3: round-trip on OK parses.
    if let Ok(stmt) = parsed {
        let printed = format!("{}", stmt);
        let reparsed = arcgraph_query::parse(&printed);
        // Round-trip failure on a smith-generated query is a finding
        // (either the generator emitted an over-printable form, or the
        // parser is non-idempotent on its own output).
        assert!(
            reparsed.is_ok(),
            "round-trip failed for seed={:#x}: original parsed OK but printed form failed to re-parse\nquery: {}\nprinted: {}",
            seed,
            q,
            printed
        );
        if let Ok(stmt2) = reparsed {
            assert_eq!(
                stmt, stmt2,
                "round-trip diverged for seed={:#x}\nquery: {}\nprinted: {}",
                seed, q, printed
            );
        }
    }
});
