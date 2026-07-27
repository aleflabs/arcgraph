# `arcgraph-tck`

Query-conformance support for the ArcGraph `v0.1.0-beta` bare database
engine.

The crate contains:

- 220 vendored openCypher feature files and their upstream provenance;
- a static scenario classifier and Markdown scorecard generator;
- a cucumber harness for dispatching feature steps into ArcGraph;
- 50 curated read queries grouped across five language areas;
- ArcGraph and optional Neo4j-oracle executors plus strict row-set
  comparison.

Vendoring a feature does not claim that ArcGraph passes it. The generated
scorecard records supported, unsupported, not-yet-executed, and out-of-scope
families separately. The user-facing syntax contract is
[`docs/arcql-reference.md`](../../docs/arcql-reference.md); it takes
precedence over the presence of a fixture.

The upstream feature license and pin are recorded in
[`tck/PROVENANCE.md`](tck/PROVENANCE.md). General workspace test commands and
external-oracle opt-outs are documented in
[`docs/testing-strategy.md`](../../docs/testing-strategy.md).
