# `arcgraph-test-harness`

Reusable workload and oracle helpers for testing the ArcGraph
`v0.1.0-beta` bare database engine.

The crate owns generic `Dataset`, `Workload`, `OracleAdapter`, and
`RegressionGate` interfaces, plus a deterministic synthetic LDBC SNB-shaped
fixture and workload. It is test support: it is not linked into production
request paths and does not add a product capability.

Workspace test commands and their external dependency gates are documented in
[`docs/testing-strategy.md`](../../docs/testing-strategy.md).
