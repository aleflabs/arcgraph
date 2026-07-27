# openCypher TCK provenance

## Upstream pin

- **Source repository:** https://github.com/opencypher/openCypher
- **Pinned commit (upstream openCypher):** `583c1419651ae80ddf340d07163be1aa0386b4ea`
- **License:** Apache-2.0 (see `crates/arcgraph-tck/LICENSE-OPENCYPHER`).
- **Copyright notice:** see `crates/arcgraph-tck/NOTICE-OPENCYPHER`.

The TCK feature files under `tck/features/` and the named graphs
under `tck/graphs/` are vendored from the openCypher repository at
the commit pinned above.

## Local modifications

A single set of step-keyword fix-ups is applied. See
`tck/features/clauses/match/Match5.feature`
scenarios `[25]` and `[26]`: leading `And having executed:` →
`Given having executed:` (5 occurrences total). The substitution
is functionally a no-op under Cucumber semantics — the gherkin-rust
0.14 parser (which `cucumber 0.21+` depends on) does not inherit
step types across scenarios the way the reference Java parser does.
The fix lets the file load cleanly through the Rust harness.

If a future TCK refresh re-vendors from a newer upstream commit,
this fix-up must be re-applied OR the upstream resolution
(if any) carried in.

## Vendoring scope

Only the TCK subtree of the upstream openCypher repository is
vendored:

- `tck/features/` — 220 `.feature` files across `clauses/`,
  `expressions/`, `useCases/`.
- `tck/graphs/` — three named graph definitions
  (`binary-tree-1`, `binary-tree-2`, `yago`) consumed by
  `Given the <name> graph` step bindings.
- `tck/ASL-2-header.txt` — the upstream Apache-2.0 source-file
  header (recorded as the canonical attribution string).

Grammar XML, CIP documents, scalafmt config, and Maven build
artifacts are NOT vendored — they are irrelevant to the runtime
behaviour we want to verify.

## License preservation

The vendored material is Apache-2.0; the
upstream NOTICE file is preserved at
`crates/arcgraph-tck/NOTICE-OPENCYPHER` per Apache-2.0 §4(d).
`cargo deny check` does NOT scan vendored data files (it scans
crate manifests only); manual licensing audit is recorded here.

When refreshing the pin to a newer openCypher commit:

1. Update the **Pinned commit** line above and the
   `crates/arcgraph-tck/Cargo.toml` `description` field.
2. Re-vendor `tck/features/`, `tck/graphs/`, `tck/ASL-2-header.txt`,
   `LICENSE`, and `NOTICE` from the new commit.
3. Re-apply the local Match5 step-keyword fix-up (or delete it if
   upstream has resolved the gherkin-rust compatibility).
4. Re-run the harness count assertion in
   `tests/tck.rs::tck_features_detected`.
