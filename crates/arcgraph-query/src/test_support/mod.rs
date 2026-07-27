//! W26-γ-3 / ADR-136 — test-support utilities for `arcgraph-query`.
//!
//! This module is `pub` (not `#[cfg(test)]`) so it can be consumed by
//! the libfuzzer harness at `fuzz/fuzz_targets/arcql_smith_fuzz.rs`.
//! The submodules carry no production code path — every call site is
//! a test, a fuzz target, or a smoke-bench.
//!
//! # Submodules
//!
//! - `smith` — type/label-aware random Cypher generator per
//!   ADR-136. Used by:
//!     - `crates/arcgraph-query/tests/arcql_smith_smoke.rs` (CI smoke)
//!     - `fuzz/fuzz_targets/arcql_smith_fuzz.rs` (nightly fuzz)

pub mod smith;
