//! Shared integration-test helpers (per Rust testing convention —
//! `tests/common/` is reserved for shared modules and does NOT
//! compile as a separate test crate).
//!
//! Each consumer integration test file pulls in the helpers via
//! `mod common;` at the top level.

#![allow(dead_code)]
// Different integration-test files consume different subsets of the
// helpers exposed here; allow `dead_code` so each consumer crate's
// build sees its own subset cleanly.

pub mod ldbc_fixture;
pub mod m4_04d_person_tenant;
