# ArcGraph

ArcGraph `v0.1.0-beta` is the bare database engine: a durable property-graph
store written in Rust, with ArcQL execution, graph traversal, property/vector/
BM25 indexes, MCP access, Bolt access, and command-line tools. This distribution
does not contain the wider product's prediction, general analytics, compliance,
connector/distribution-service, Python-sidecar, or bi-temporal features.

The binary identifies itself as `arcgraph 0.1.0-beta`.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Rust 1.85+](https://img.shields.io/badge/rust-1.85%2B-orange.svg)](Cargo.toml)

## Install prerequisites

You need [Git](https://git-scm.com/downloads), [curl](https://curl.se/download.html),
and [Python 3](https://www.python.org/downloads/). The quickstart uses only the
Python standard library. Confirm they are on `PATH`:

```sh
git --version
curl --version
python3 --version
```

Install Rust with the official [rustup](https://rustup.rs/) installer, load
Cargo into the current shell's `PATH`, and check the compiler:

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
. "$HOME/.cargo/env"
rustc --version
cargo --version
```

The `rustc` command must report `rustc 1.85.0` or newer. A new rustup
installation selects the current stable toolchain and also installs Cargo.

The public repository is
[`https://github.com/aleflabs/arcgraph`](https://github.com/aleflabs/arcgraph).
After cloning or extracting the source archive, change into its root directory.

## Build

Build the 13-crate public workspace with its default features:

```bash
cargo build --workspace
```

The cold build measured on the OCI VM took **114.60 seconds**, and `target/`
occupied **4.4 GB**. These are the measured validation values, not estimates.
The complete all-features workspace test is much larger: it reached a
**34.99 GB** `target/` with only **3.6 GB** free while it was still linking
test binaries, before any test body ran. Provide **well over 35 GB free**;
the release test preflight requires at least **45 GB**. See
[`docs/testing-strategy.md`](docs/testing-strategy.md) before attempting it.

## Quickstart

Run the worked durable MCP example:

```bash
python3 scripts/agent-quickstart.py --bin target/debug/arcgraph
```

The script starts `arcgraph serve` non-interactively against a new durable
data directory, waits for the MCP `initialize` response, discovers the tool
catalog, ingests three nodes and one relationship, applies Bolt read grants for
principal `neo4j`, inspects the stored values, runs ArcQL, BM25, and vector
queries, shuts the server down cleanly, and restarts against the same directory.
It checks the data and both search indexes after restart. Every value below is
asserted before it is printed. The temporary directory is removed on exit; pass
`--data /path/to/empty-dir` to retain the store.

Expected output:

```text
agent-quickstart: READY server=arcgraph protocol=2025-06-18
agent-quickstart: TOOLS graph.schema,graph.inspect,graph.explore,graph.search,graph.ingest,graph.raw_query (6)
agent-quickstart: INGEST inserted=4 failed=0 node_ids=1,2,3 rel_id=1 acl_principal=neo4j
agent-quickstart: MCP_REQUEST {"id":4,"jsonrpc":"2.0","method":"tools/call","params":{"arguments":{"format":"json","node_id":1,"tenant_id":1},"name":"graph.inspect"}}
agent-quickstart: MCP_RESPONSE {"id":4,"jsonrpc":"2.0","result":{"content":[{"text":"{\"body\":\"{\\\"id\\\":1,\\\"label\\\":\\\"Person\\\",\\\"neighbors\\\":[{\\\"direction\\\":\\\"out\\\",\\\"label\\\":\\\"Person\\\",\\\"node_id\\\":2,\\\"rel_type\\\":\\\"KNOWS\\\"}],\\\"properties\\\":{\\\"embedding\\\":[1.0,0.0,0.0],\\\"language\\\":\\\"Analytical Engine\\\",\\\"name\\\":\\\"Ada Lovelace\\\",\\\"text\\\":\\\"Analytical Engine algorithm notes\\\"}}\",\"format\":\"json\"}","type":"text"}],"isError":false}}
agent-quickstart: READBACK {"id":1,"label":"Person","neighbors":[{"direction":"out","label":"Person","node_id":2,"rel_type":"KNOWS"}],"properties":{"embedding":[1.0,0.0,0.0],"language":"Analytical Engine","name":"Ada Lovelace","text":"Analytical Engine algorithm notes"}}
agent-quickstart: ARCQL {"columns":["a.name","r.since","b.name"],"row_count":1,"rows":[["Ada Lovelace",1952,"Grace Hopper"]],"truncated":false,"writes":{"labels_added":0,"labels_removed":0,"nodes_created":0,"nodes_deleted":0,"properties_removed":0,"properties_set":0,"rels_created":0,"rels_deleted":0}}
agent-quickstart: BM25_REQUEST {"id":6,"jsonrpc":"2.0","method":"tools/call","params":{"arguments":{"format":"json","k":2,"principal":"neo4j","query":"compiler","tenant_id":1},"name":"graph.search"}}
agent-quickstart: BM25 {"hits":[{"label":"Person","node_id":2,"score":1.04170823097229}],"k":2}
agent-quickstart: VECTOR_REQUEST {"id":7,"jsonrpc":"2.0","method":"tools/call","params":{"arguments":{"format":"json","k":2,"principal":"neo4j","query_vec":[1.0,0.0,0.0],"tenant_id":1},"name":"graph.search"}}
agent-quickstart: VECTOR {"hits":[{"label":"Person","node_id":1,"score":1.0},{"label":"Person","node_id":2,"score":0.3333333333333333}],"k":2}
agent-quickstart: STOP phase=initial exit=0
agent-quickstart: RESTART_READY server=arcgraph protocol=2025-06-18
agent-quickstart: DURABLE rows=[["Ada Lovelace",1952,"Grace Hopper"]] readback_name="Ada Lovelace"
agent-quickstart: STOP phase=restart exit=0
agent-quickstart: PASS all values survived restart
```

## Using ArcGraph from an agent

Point an MCP client at the built binary with `serve`, `--stdio-mcp`, and a
stable `--data` directory. For example, the client entry is:

```json
{
  "command": "/absolute/path/to/arcgraph/target/debug/arcgraph",
  "args": [
    "serve",
    "--stdio-mcp",
    "--data",
    "/absolute/path/to/arcgraph-data",
    "--admin-http",
    "",
    "--metrics-http",
    ""
  ]
}
```

The empty admin and metrics arguments keep an agent-owned stdio process from
opening optional side listeners. A server must receive exactly one storage
mode: durable `--data DIR` or ephemeral `--in-memory`. The catalog contains
exactly six tools:
`graph.schema`, `graph.inspect`, `graph.explore`, `graph.search`,
`graph.ingest`, and `graph.raw_query`.

After the standard MCP `initialize` and `tools/list` exchange, a concrete
inspection call is:

```json
{"id":4,"jsonrpc":"2.0","method":"tools/call","params":{"arguments":{"format":"json","node_id":1,"tenant_id":1},"name":"graph.inspect"}}
```

Its real response in the quickstart is:

```json
{"id":4,"jsonrpc":"2.0","result":{"content":[{"text":"{\"body\":\"{\\\"id\\\":1,\\\"label\\\":\\\"Person\\\",\\\"neighbors\\\":[{\\\"direction\\\":\\\"out\\\",\\\"label\\\":\\\"Person\\\",\\\"node_id\\\":2,\\\"rel_type\\\":\\\"KNOWS\\\"}],\\\"properties\\\":{\\\"embedding\\\":[1.0,0.0,0.0],\\\"language\\\":\\\"Analytical Engine\\\",\\\"name\\\":\\\"Ada Lovelace\\\",\\\"text\\\":\\\"Analytical Engine algorithm notes\\\"}}\",\"format\":\"json\"}","type":"text"}],"isError":false}}
```

For the exact `text`/`embedding` indexing conventions and the ACL behavior
demonstrated above, read [`docs/search.md`](docs/search.md). For HTTPS and Bolt
startup and client instructions, read
[`docs/transports.md`](docs/transports.md). The supported query language is
listed in [`docs/arcql-reference.md`](docs/arcql-reference.md).

## Bare database surface

- Durable page storage, WAL recovery, checkpoints, MVCC visibility, and
  snapshot isolation.
- ArcQL parse, bind, type-check, plan, and execution pipeline.
- Property, vector, and BM25 indexes.
- Directed graph traversal.
- Leiden-family community indexes and background refresh scheduling.
- Ordinary scalar values for `datetime`, `localdatetime`, `date`,
  `duration`, and exact `decimal` values.
- Bolt 5.0, MCP over stdio or HTTP/TLS, and an embeddable Rust API.

Temporal qualifiers such as `FOR VALID_TIME` and `AS OF` are **not in this
distribution's ArcQL grammar**. MVCC supplies current transaction visibility;
this build has no bi-temporal query surface.

## Workspace

The public engine is split into 13 crates:

```text
arcgraph
arcgraph-core
arcgraph-storage
arcgraph-index
arcgraph-query
arcgraph-vector
arcgraph-bm25
arcgraph-graph-traversal
arcgraph-community
arcgraph-mcp
arcgraph-cli
arcgraph-tck
arcgraph-test-harness
```

Crate responsibilities and dependency boundaries are documented in
[`docs/bounded-contexts.md`](docs/bounded-contexts.md).

## Development checks

CI enforces formatting, the default-feature workspace build, warning-free
linting, documentation, and tests. Run Cargo commands serially. Some scale and
environment-sensitive suites require explicit opt-in or skip flags; see
[`docs/testing-strategy.md`](docs/testing-strategy.md) before running the
complete workspace suite.

Operational guides live in [`docs/operations/`](docs/operations/), and the
security reporting policy is in [`SECURITY.md`](SECURITY.md).

## Known issues

- **Startup logging (#1640).** A healthy restart on a populated store emits
  `WARN` and `ERROR` messages about tenant-0 system records, including
  `id/label lookup will miss`. These messages are expected on a healthy start
  in this beta and do **not** indicate user-data loss. Cold validation passed
  every user-data assertion around them, including recovery after `SIGKILL`.
- **Bolt network scope.** The CLI starts plaintext Bolt only on loopback. A
  non-loopback Bolt bind is rejected because this build exposes no CLI flags
  for Bolt TLS certificate material. Use `127.0.0.1` or `::1`; do not treat
  `--allow-remote-bolt-bind` alone as sufficient.
- **Graph dump.** `arcgraph dump --data` deliberately refuses instead of
  producing an incomplete export. Use the verified cold-backup command for a
  durable store; the storage-rooted logical dump is not implemented.
- **Config file flag.** `arcgraph serve --config PATH` is accepted but ignored
  in this beta and emits a warning. Use explicit CLI flags; do not assume a
  TOML file changed server behavior.
- **Built-in health client.** `arcgraph health` accepts only `http://` URLs,
  while the MCP network listener is HTTPS-only. Use a TLS-aware client for
  MCP `/healthz`, or use the separate plain-HTTP admin `/livez` and `/readyz`
  endpoints.

## License

Licensed under the [Apache License 2.0](LICENSE).
