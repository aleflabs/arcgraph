# `arcgraph-cli`

Command-line and server composition for the ArcGraph `v0.1.0-beta` bare
database engine. Database algorithms remain in the subsystem crates.

The `arcgraph` binary has seven subcommands:

| Subcommand | Behavior |
|---|---|
| `serve` | Starts exactly one of MCP stdio, MCP over HTTPS, or Bolt 5.0 against durable or in-memory storage |
| `check` | Validates configuration and optionally cold-opens and samples a committed store |
| `dump` | Emits an empty no-store envelope; a storage-rooted logical dump refuses rather than exporting an incomplete graph |
| `health` | Probes a plain-HTTP `/healthz` URL; it does not support HTTPS |
| `migrate` | Upgrades an on-disk data directory or parses Neo4j exports into an ephemeral process store |
| `backup` | Creates or restores a verified cold backup |
| `load` | Bootstraps native newline-delimited JSON into a new durable data directory |

`serve` also wires the optional admin and metrics listeners, graceful
shutdown, storage locking, WAL recovery, and the retained community-index
refresh worker. Transport startup examples are in
[`docs/transports.md`](../../docs/transports.md); backup and load procedures
are in [`docs/operations/`](../../docs/operations/).
