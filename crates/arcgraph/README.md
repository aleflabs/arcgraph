# `arcgraph`

The embeddable facade for the ArcGraph `v0.1.0-beta` bare database engine.
It adds no database logic; it presents a curated API from eight implementation
crates under stable module paths.

| Facade module | Implementation crate | Surface |
|---|---|---|
| `arcgraph::core` | `arcgraph-core` | IDs, records, scalar values, durability, errors |
| `arcgraph::storage` | `arcgraph-storage` | pages, WAL, recovery, MVCC, CRUD |
| `arcgraph::index` | `arcgraph-index` | secondary property B-trees |
| `arcgraph::vector` | `arcgraph-vector` | HNSW/DiskANN vector indexing |
| `arcgraph::bm25` | `arcgraph-bm25` | Tantivy BM25 indexing |
| `arcgraph::community` | `arcgraph-community` | Leiden indexes and refresh scheduling |
| `arcgraph::query` | `arcgraph-query` | ArcQL parser, planner, executor |
| `arcgraph::mcp` | `arcgraph-mcp` | MCP, HTTPS, Bolt, ACL and storage adapters |

The facade deliberately curates names; an item becoming public in an
implementation crate does not automatically become public here. Test-only
storage helpers are omitted.

The compiled embedded example is
[`examples/embedded_quickstart.rs`](examples/embedded_quickstart.rs).
Server and maintenance binaries are provided by
[`arcgraph-cli`](../arcgraph-cli/).
