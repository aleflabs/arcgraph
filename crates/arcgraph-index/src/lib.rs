//! Secondary indices for ArcGraph.
//!
//! Scope: HNSW vector index (with layered deletion strategies — see
//! ADR-003), BM25 / full-text via Tantivy, and secondary B-trees on
//! properties. Index lifecycle is tied to MVCC visibility rules.
//!
//! The primary B-tree (`NodeId → PageId`, `RelId → PageId`) lives in
//! `arcgraph-storage`, not here.

#![recursion_limit = "256"]

pub mod property_key;
pub mod secondary_btree;

pub use property_key::{IndexKeyInput, canonical_row_key};
pub use secondary_btree::{
    INLINE_NODEID_COUNT, INTERNAL_CAPACITY, INTERNAL_ENTRY_OFFSET, INTERNAL_ENTRY_SIZE,
    INTERNAL_FIRST_CHILD_OFFSET, InternalPageMut, InternalPageRef, LEAF_CAPACITY,
    LEAF_ENTRY_OFFSET, LEAF_ENTRY_SIZE, LeafEntry, LeafFindResult, LeafPageMut, LeafPageRef,
    OVERFLOW_FILLED_COUNT_OFFSET, OVERFLOW_NEXT_OFFSET, OVERFLOW_SLOTS_OFFSET,
    OVERFLOW_SLOTS_PER_PAGE, OverflowPageMut, OverflowPageRef, PageBuf, PageLatch, PropertyValue,
    SECONDARY_INDEX_ROOT_KEY, SecondaryIndex, SecondaryIndexError, SecondaryKey,
    SecondaryPageStore, SplitInfo, fresh_page_buf, hash_str_56,
};
