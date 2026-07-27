# ADR-004: Six-tool MCP catalog

- **Status:** Accepted for `v0.1.0-beta`
- **Scope:** Bare database distribution

The MCP catalog contains exactly:

1. `graph.schema`
2. `graph.inspect`
3. `graph.explore`
4. `graph.search`
5. `graph.ingest`
6. `graph.raw_query`

The catalog is intentionally a database surface. Prediction, history,
statistics, compliance, administration, connector, and distribution tools are
not registered in this build. Bolt provides a driver protocol for ArcQL; it
does not add MCP tools.

Adding a tool changes the public protocol and requires an explicit release
decision, implementation, authorization policy, and end-to-end test.
