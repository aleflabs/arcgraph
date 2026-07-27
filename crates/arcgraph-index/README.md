# `arcgraph-index`

Secondary property-index primitives for the ArcGraph `v0.1.0-beta` bare
database engine.

The crate owns the secondary B-tree page layouts, latches, key
canonicalization, overflow pages, and insert/remove/lookup implementation.
Keys are tenant-, label-, property-, and value-scoped. Storage is supplied
through the `SecondaryPageStore` interface.

The primary node/relationship index lives in `arcgraph-storage`. Vector and
BM25 implementations live in `arcgraph-vector` and `arcgraph-bm25`
respectively; this crate does not wrap either one. Query planning belongs to
`arcgraph-query`.
