# `arcgraph-mcp`

Agent and driver protocol surfaces for the ArcGraph `v0.1.0-beta` bare
database engine.

## MCP catalog

The catalog contains exactly six tools:

- `graph.schema`
- `graph.inspect`
- `graph.explore`
- `graph.search`
- `graph.ingest`
- `graph.raw_query`

There are no prediction, history, statistics, compliance, administration, or
connector tools in this distribution.

## Transports and boundaries

The crate implements MCP over stdio, MCP over HTTPS at `POST /mcp`, and a Bolt
5.0 server. It also owns JSON-RPC envelopes, JSON/TOON/YAML rendering, request
rate limiting, OAuth/JWT validation, TLS reload support, read-ACL enforcement,
and adapters into storage and ArcQL execution. It does not provide a WebSocket
or gRPC transport.

Storage, index, and query execution remain owned by their respective crates.
Start and connect to each network transport using
[`docs/transports.md`](../../docs/transports.md). Search and ACL request shapes
are documented in [`docs/search.md`](../../docs/search.md).

## JSON-RPC errors

Protocol errors use the standard `-32700` and `-32600` through `-32603`
codes. ArcGraph server errors use:

| Code | Meaning |
|---:|---|
| `-32001` | request cancelled |
| `-32002` | unauthorized |
| `-32003` | tenant unknown |
| `-32004` | required index unavailable |
| `-32005` | ArcQL query error |
| `-32006` | execution or substrate I/O error |
| `-32007` | rate limited |
| `-32008` | scope forbidden |

The public boundary translates internal errors into these envelopes; internal
crate error variants are not sent directly.
