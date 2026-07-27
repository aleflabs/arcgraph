# Changelog

All notable public changes are recorded here.

## 0.1.0-beta

- Established the 13-crate public database workspace.
- Shipped durable page storage, WAL recovery, checkpoints, MVCC visibility,
  and snapshot isolation.
- Shipped ArcQL parsing and execution with ordinary date, time, duration,
  and exact-decimal scalar values.
- Shipped property, vector, BM25, traversal, and community-index support.
- Shipped the six-tool MCP catalog, Bolt transport, CLI, TCK runner, and
  reusable test harness.
- Pruned release automation, documentation, examples, and benchmarks that
  do not apply to the public database engine.
