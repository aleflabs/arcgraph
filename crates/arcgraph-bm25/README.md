# `arcgraph-bm25`

Tantivy-backed BM25 indexing for the ArcGraph `v0.1.0-beta` bare database
engine.

## Responsibility

- `Bm25IndexHandle` provides tenant-local top-k search, filtered search,
  upsert, and delete operations.
- `Bm25Service` owns the per-tenant on-disk directories under
  `<data-dir>/bm25/` and opens indexes lazily.
- The storage commit adapter applies or rolls back pending BM25 mutations with
  the corresponding graph commit.
- MVCC visibility filtering prevents an index hit from bypassing storage
  visibility.

The served `graph.search` path indexes the conventional string property
`text`; see [`docs/search.md`](../../docs/search.md).

This crate does not generate text, choose an embedding model, or provide a
text-index DDL. Tantivy owns its own index files; graph records and transaction
visibility remain storage-owned.
