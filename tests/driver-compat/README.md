# Bolt driver compatibility

ArcGraph `v0.1.0-beta` speaks Bolt 5.0. The externally validated client is the
official Neo4j Python driver pinned at `neo4j==6.2.0`.

The canonical setup is [`docs/transports.md`](../../docs/transports.md). It
seeds a durable ACL-granted graph, installs the exact driver in an isolated
environment, connects over loopback, and verifies node and relationship
properties using [`scripts/bolt-quickstart.py`](../../scripts/bolt-quickstart.py).

The Rust compatibility layers are:

- `crates/arcgraph-mcp/tests/bolt_e2e_python_driver_shape.rs`
- `crates/arcgraph-mcp/tests/bolt_e2e_full_session.rs`
- `crates/arcgraph-cli/tests/driver_compat_bolt_v5.rs`

The older Python files in `python/` are protocol-development fixtures, not the
public setup guide. They do not replace the property and ACL assertions in the
canonical quickstart.
