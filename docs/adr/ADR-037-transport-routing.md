# ADR-037: Transport routing

- **Status:** Accepted for `v0.1.0-beta`

One `arcgraph serve` process selects one primary data transport: MCP over
stdio, MCP over HTTPS, or Bolt 5.0. All route to the same storage, query, and
permission adapters; none owns an alternate database implementation.

The exact startup, tenant, authentication, and client rules are documented in
[`../transports.md`](../transports.md).
