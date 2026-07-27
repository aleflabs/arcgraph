# Neo4j export parsing in `v0.1.0-beta`

The `arcgraph migrate from-neo4j-*` commands validate and ingest two Neo4j
export shapes:

| Command | Accepted input |
|---|---|
| `from-neo4j-cypher PATH` | Semicolon-separated APOC-style node `CREATE` and relationship stitch statements |
| `from-neo4j-csv --nodes PATH --rels PATH` | A node CSV with `:ID`/`:LABEL` and a relationship CSV with `:START_ID`/`:END_ID`/`:TYPE` |

The parsers accept string, integer, floating-point, boolean, and null property
values. Cypher constructors for dates, durations, points, nested maps, and
lists are rejected by this migration parser even though some scalar
constructors are available in ArcQL.

Cypher input supports one node label and relationships whose endpoint IDs can
be resolved from `neo4j_id` properties. CSV handling follows quoted RFC 4180
fields and removes Neo4j type suffixes from property headers.

## Important persistence limit

These two subcommands currently build an ephemeral in-process store, print
inserted/failed counts, and exit. They have no `--data` option. A successful
run proves that the export parsed and passed through the ingest adapter; it
does **not** create a durable ArcGraph data directory.

For a durable migration, convert the source rows into `graph.ingest` batches
and send them to a running `arcgraph serve --data DIR` process. Verify every
ingest response, including `failed_count` and `dropped_acl_grants`, then stop,
restart, and query the committed store. The repository's durable fixture in
[`../../scripts/agent-quickstart.py`](../../scripts/agent-quickstart.py)
demonstrates that ingest/restart/readback path.

Do not treat the temporary `migrate` command as an exporter, converter, or
durable import tool.
