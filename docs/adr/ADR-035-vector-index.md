# ADR-035: Tenant-local vector indexes

- **Status:** Accepted for `v0.1.0-beta`

The retained vector crate provides tenant-local HNSW and DiskANN search,
distance kernels, vector encodings, deletion repair, and storage adapters.
Clients supply vectors; ArcGraph does not generate embeddings or call a model.

The served convention is a numeric, non-empty `embedding` node property and a
same-dimension `query_vec`. Search request and ACL behavior are documented in
[`../search.md`](../search.md).
