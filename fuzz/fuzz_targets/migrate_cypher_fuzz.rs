#![no_main]
//! W28-S604: Neo4j Cypher-export migration parser fuzz target
//! (testing strategy; full-canonical-set audit Task #604).
//!
//! # What this fuzzes
//!
//! [`arcgraph_cli::migrate::parse_cypher_export_str`] — the W18δ
//! hand-written parser for `apoc.export.cypher.all()` output at
//! `crates/arcgraph-cli/src/migrate.rs:157`. It is the most
//! panic-surface-rich parser in the repo audit: a recursive-descent-
//! style text parser that splits on `;`, classifies CREATE/MATCH
//! statements, and drives a family of partial sub-parsers
//! (`parse_node_create`, `parse_rel_create`, `parse_bare_rel_create`,
//! `parse_match_endpoint_ids`, `parse_edge_arrow`, `parse_properties`,
//! `parse_value`, `parse_id_from_props`) that slice and char-index the
//! input. This is the `docs/testing-strategy.md` §3 "Neo4j Northwind
//! round-trip" ingestion surface — operators feed it externally-
//! generated export files, so it is a genuine untrusted-input parser.
//! It was previously UNCOVERED (audit Task #604 deliverable 1).
//!
//! # Assertions
//!
//! - **No panic.** `parse_cypher_export_str(s)` MUST NOT panic on ANY
//!   UTF-8 input. Recognised statements produce `Ok(Vec<IngestBatch>)`;
//!   unclassifiable statements return `Err(MigrateError::CypherParse)`.
//!   Both outcomes are valid — the contract is no-panic / no-UB /
//!   no-OOB-index on hostile input (truncated props, unbalanced braces
//!   / parens / quotes, embedded NULs, deep nesting, mixed-script
//!   identifiers).
//! - **Determinism.** The parser is pure over its `&str` input;
//!   parsing the same input twice MUST yield equal results (`Ok == Ok`
//!   structurally, or both `Err`). A divergence would indicate hidden
//!   nondeterministic state (e.g., iteration-order-dependent output).
//!
//! Input is capped at 64 KiB to bound per-iteration wall time; real
//! exports are larger but the per-statement parse paths are fully
//! exercised well below this cap.

use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 64 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_INPUT_BYTES {
        return;
    }
    let Ok(s) = std::str::from_utf8(data) else {
        return;
    };

    // Core contract: no panic on any input. Both Ok and Err are valid.
    let first = arcgraph_cli::migrate::parse_cypher_export_str(s);

    // Determinism oracle: a second parse of the same input must agree
    // with the first (Ok/Err arm + structural equality on Ok).
    let second = arcgraph_cli::migrate::parse_cypher_export_str(s);
    match (first, second) {
        (Ok(a), Ok(b)) => assert_eq!(
            a, b,
            "parse_cypher_export_str non-deterministic: two parses of the same input diverged"
        ),
        (Err(_), Err(_)) => {
            // Both rejected — consistent.
        }
        (a, b) => panic!(
            "parse_cypher_export_str non-deterministic Ok/Err arm: {:?} vs {:?}",
            a.is_ok(),
            b.is_ok()
        ),
    }
});
