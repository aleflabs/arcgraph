# ArcGraph fuzz harness

This directory contains `cargo-fuzz` targets for retained bare-database
parsers, codecs, protocols, page layouts, WAL records, indexes, and the native
loader. It is excluded from the default workspace, so libFuzzer and sanitizer
dependencies are not part of the production dependency graph or the README
build.

The target catalog is:

- `arcql_parser_fuzz`
- `arcql_smith_fuzz`
- `bolt_message_fuzz`
- `bolt_packstream_fuzz`
- `catalog_page_fuzz`
- `commit_bundle_decoder_fuzz`
- `m5_native_loader_fuzz`
- `mcp_message_fuzz`
- `mcp_raw_query_fuzz`
- `mcp_yaml_fuzz`
- `migrate_cypher_fuzz`
- `node_record_deser_fuzz`
- `page_layout_fuzz`
- `property_value_fuzz`
- `rel_record_deser_fuzz`
- `secondary_index_fuzz`
- `toon_serializer_fuzz`
- `value_map_json_fuzz`
- `wal_deserializer_fuzz`
- `wal_segment_fuzz`

`placeholder` remains as a build-shape sentinel and does not cover a product
surface. Seed corpora live under `corpus/`; generated discoveries and crash
artifacts are ignored unless a minimal reproducer is deliberately reviewed and
committed.

Fuzzing requires a nightly Rust toolchain and `cargo-fuzz`. It is an
additional developer campaign, not part of the default-feature public build or
the release commands in [`docs/testing-strategy.md`](../docs/testing-strategy.md).
