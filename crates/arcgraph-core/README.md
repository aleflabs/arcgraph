# `arcgraph-core`

Shared primitives for the ArcGraph `v0.1.0-beta` bare database engine.

This leaf crate owns:

- tenant, node, relationship, page, label, property, type, string, and LSN
  identifiers;
- node, relationship, TEL, page-header, and page-type records;
- the shared error taxonomy and result alias;
- scalar date, time, duration, and exact-decimal parsing;
- durability-tier and secret-provider interfaces;
- cache-line and optional fault-injection primitives.

It performs no database I/O and owns no query planning, indexing, transport,
or process orchestration. See
[`docs/bounded-contexts.md`](../../docs/bounded-contexts.md) for dependency
boundaries.
