# Offline native bulk load

`arcgraph load` builds a durable generation while the server is stopped. The
target must be virgin or an interrupted loader-owned generation; unrelated
state is refused without mutation.

The only accepted format is newline-delimited native JSON. Each record uses
hex strings for byte fields and raw IEEE-754 bits:

```json
{"kind":"node","external_id":"01","label_or_type":1,"float_bits":"3ff0000000000000","opaque":"616461"}
{"kind":"node","external_id":"02","label_or_type":1,"float_bits":"4000000000000000","opaque":"6772616365"}
{"kind":"relationship","external_id":"0a","source_id":"01","target_id":"02","label_or_type":1,"float_bits":"0000000000000000","opaque":"6b6e6f7773"}
```

Load the file into a non-system tenant:

```bash
target/debug/arcgraph load \
  --input ./native-load.ndjson \
  --data-dir ./arcgraph-loaded \
  --format native \
  --tenant 2
```

`external_id`, `source_id`, `target_id`, and `opaque` must contain an even
number of hexadecimal digits. `float_bits` must contain exactly 16.
Relationship endpoints must name nodes in the input.

The loader bounds individual record size and JSON depth, plans disk use before
publication, writes into a generation namespace, and publishes through one
atomic `CURRENT` swap. Re-running after a completed load is a no-op; rerunning
after an interrupted loader generation resumes its durable stages.

This native format is a storage-oriented bootstrap boundary, not the
`graph.ingest` JSON property format. Parquet is not accepted.
