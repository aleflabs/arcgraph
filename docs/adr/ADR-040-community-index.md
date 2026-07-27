# ADR-040: Community membership index

- **Status:** Accepted for `v0.1.0-beta`

The retained community crate implements Leiden-family static and incremental
membership indexing plus bounded refresh scheduling. Query planning can
consume that index through an adapter.

The bare server does not expose a separate community-analysis MCP tool. The
six-tool public catalog remains the boundary in
[`ADR-004`](ADR-004-mcp-tool-catalog.md).
