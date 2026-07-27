#![no_main]
//! M10 stage-1 (ADR-207): catalog root page decoder fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_storage::catalog::decode_catalog_page`] — the fail-closed
//! decoder for the dedicated catalog root page
//! (`crates/arcgraph-storage/src/catalog/page.rs`). The decoder runs
//! over UNTRUSTED bytes in production: restored ADR-204 backups, torn
//! writes, zeroed pre-M10 pages, foreign files.
//!
//! # Assertion
//!
//! - **No panic.** `decode_catalog_page` on ANY `PAGE_SIZE` buffer MUST
//!   return either `Ok(Vec<TenantRecord>)` or a typed
//!   `CatalogPageError`. Both are valid outcomes.
//! - **Re-encode round-trip.** When decode succeeds, re-encoding the
//!   decoded records MUST succeed (the decoded form is within encode
//!   caps by construction) and decode back to an equal registry — the
//!   canonical-form property the attach verify step relies on.
//!
//! Inputs shorter than `PAGE_SIZE` are zero-padded into a page buffer
//! (framing is the buffer pool's responsibility — the decoder always
//! sees exactly one page); longer inputs are truncated.

use arcgraph_core::PAGE_SIZE;
use arcgraph_storage::catalog::{decode_catalog_page, encode_catalog_page};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let mut page = [0u8; PAGE_SIZE];
    let n = data.len().min(PAGE_SIZE);
    page[..n].copy_from_slice(&data[..n]);

    if let Ok(records) = decode_catalog_page(&page) {
        // Canonical-form round-trip: anything the decoder accepts must
        // re-encode + re-decode to the same registry.
        let re = encode_catalog_page(&records)
            .expect("decoded registry must be within encode caps");
        let back = decode_catalog_page(&re).expect("re-encoded page must decode");
        assert_eq!(back, records, "catalog page round-trip must be canonical");
    }
});
