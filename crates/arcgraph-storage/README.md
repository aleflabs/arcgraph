# `arcgraph-storage`

Durable storage for the ArcGraph `v0.1.0-beta` bare database engine.

## Responsibility

- fixed-size pages, buffer-pool pinning, eviction, and file-backed or in-memory
  page I/O;
- WAL writing, replay, segment reclamation, checkpoints, and doublewrite
  recovery;
- tenant routing, system catalogs, string interning, ID allocation, and
  idempotency records;
- MVCC transactions and current-snapshot visibility;
- node/relationship CRUD, TEL adjacency, property bags, blobs, and primary
  indexes;
- secondary/vector index persistence adapters, permission indexes, optional
  WAL encryption, and bounded spill storage.

The crate does not parse ArcQL, choose query plans, expose MCP/Bolt, or own the
Tantivy and vector-search algorithms. Those are composed at higher layers.

The record and ownership invariants are documented in
[`docs/records-semantics.md`](../../docs/records-semantics.md).
