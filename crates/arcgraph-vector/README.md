# `arcgraph-vector`

Vector indexing for the ArcGraph `v0.1.0-beta` bare database engine.

## Responsibility

- tenant-local `VectorIndexHandle` instances;
- HNSW and DiskANN graph construction and filtered nearest-neighbor search;
- SIMD-backed L2, inner-product, cosine, and Hamming distance kernels;
- float, half, SQ8, binary, and RaBitQ encodings;
- tenant/index vector arenas and quantizer state;
- deletion repair and runtime backend dispatch.

Persistence, WAL integration, and recovery adapters live in
`arcgraph-storage`. Hybrid BM25/vector fusion is performed by the query/MCP
layers. This crate consumes vectors supplied by clients; it does not create
embeddings or invoke a model.

The served property convention and query dimensions are documented in
[`docs/search.md`](../../docs/search.md).
