//! Downstream-consumer compilation test for the umbrella's
//! "Embedded usage" doc-test block.
//!
//! # Why this test exists
//!
//! W15α PR #306 round-1 review §F-1 flagged that the `lib.rs:22-46`
//! doc-test imported `arcgraph_query::*` paths directly — which made
//! the doc-test pass under `cargo test --doc -p arcgraph` (the
//! doc-test crate is built in the arcgraph package's context, so
//! `arcgraph-query` is in scope as a direct dep) but FAIL for a real
//! downstream consumer with only `arcgraph` in their `Cargo.toml`.
//! The round-2 fix-up at this slice rewrote the doc-test to use
//! `arcgraph::query::*` paths exclusively. This test pins the
//! invariant: any future drift back to `arcgraph_*::*` imports in the
//! doc-test block — or any future rename that removes a name from the
//! curated umbrella surface — re-breaks this test mechanically.
//!
//! # What this test does
//!
//! 1. Synthesizes a sibling consumer crate under `target/doc-test-
//!    consumer-<unique>/` (sibling to the arcgraph workspace so the
//!    parent workspace's `[workspace]` table does NOT auto-include
//!    it). The crate's `Cargo.toml` declares `arcgraph` as the ONLY
//!    dependency, pinned via path to the umbrella crate's
//!    `CARGO_MANIFEST_DIR`.
//! 2. Writes `src/main.rs` with the exact code from the doc-test
//!    block (Embedded usage section in `crates/arcgraph/src/lib.rs`).
//! 3. Runs `cargo build` on the synthesized crate. If it exits 0 the
//!    doc-test is portable; if it fails — with E0432 / E0433 on
//!    `arcgraph_query` or any underlying crate — the test fails with
//!    the captured stderr.
//!
//! # Why this test exists
//!
//! This test has an immediate consumer: the doc-test block above.
//!
//! # Cost
//!
//! ~3-5 s on a warm cache (cargo only rebuilds the synthesized
//! consumer; the umbrella + its deps are already built). On cold
//! cache, ~30-60 s. The test runs in `--release` is OFF (default
//! `cargo build`) to keep cycle time small.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

/// The exact body of the `# Embedded usage` doc-test in
/// `crates/arcgraph/src/lib.rs`. Kept in sync MANUALLY: any change to
/// the doc-test block above (or to the canonical embedded-quickstart
/// example) MUST be mirrored here. A drift would not break correctness
/// — both versions are independently compile-checked — but would
/// invalidate the "this test pins the doc-test text" claim.
///
/// **v1.0-GA polish (per W15α R2-NEW-2):** replace the hardcoded
/// `DOC_TEST_BODY` with a runtime extractor that reads `lib.rs` from
/// `CARGO_MANIFEST_DIR`, parses the first ```rust block inside the
/// `# Embedded usage` section (Tantivy-style doctest parser), and
/// writes that to `src/main.rs`. Eliminates the manual-sync
/// requirement entirely — the test then mechanically pins lib.rs's
/// actual doc-test text rather than a parallel copy.
const DOC_TEST_BODY: &str = r#"use arcgraph::core::{LabelId, NodeId, TenantId};
use arcgraph::query::QueryEngine;
use arcgraph::query::executor::StubExecutorSubstrate;
use arcgraph::query::executor::value::{NodeView, Value};
use arcgraph::query::semantic::StubCatalogProvider;

fn main() {
    let mut substrate = StubExecutorSubstrate::new();
    for i in 1..=3 {
        substrate = substrate.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                .with_property("age", Value::Integer(i as i64 * 10)),
        );
    }
    let catalog = StubCatalogProvider::new()
        .with_labels(["Person"])
        .with_properties(["age"]);

    let engine = QueryEngine::new(&catalog);
    let result = engine
        .execute("MATCH (n:Person) RETURN n", &substrate)
        .expect("execute MATCH");
    assert_eq!(result.rows().len(), 3);
}
"#;

#[test]
fn doc_test_compiles_as_downstream_consumer() {
    // CARGO_MANIFEST_DIR is the arcgraph crate's directory at
    // compile time. The synthesized consumer crate lives under
    // CARGO_TARGET_TMPDIR (set by cargo for tests), or falls back to
    // a sibling of the workspace target dir.
    let arcgraph_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let tmp_root = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| arcgraph_path.join("../../target/doc-test-consumer-fallback"));

    let consumer = tmp_root.join("doc_test_consumer");
    // Clean slate — the test self-cleans so re-runs don't pick up
    // stale state.
    if consumer.exists() {
        fs::remove_dir_all(&consumer).expect("clean prior consumer dir");
    }
    fs::create_dir_all(consumer.join("src")).expect("create consumer src dir");

    // `[workspace]` empty table makes this crate a standalone — not a
    // member of any ambient workspace cargo might discover by walking
    // upward. This is the key trick that makes the test simulate a
    // real downstream `cargo new` consumer.
    let cargo_toml = format!(
        r#"[package]
name = "doc_test_consumer"
version = "0.0.0"
edition = "2024"

[workspace]

[dependencies]
arcgraph = {{ path = "{path}" }}
"#,
        path = arcgraph_path.to_string_lossy(),
    );
    fs::write(consumer.join("Cargo.toml"), cargo_toml).expect("write consumer Cargo.toml");
    fs::write(consumer.join("src/main.rs"), DOC_TEST_BODY).expect("write consumer main.rs");

    // Resolve cargo via the env var cargo sets when invoking tests.
    let cargo = env!("CARGO");

    // Build the consumer. Use `--offline=false` (cargo's default)
    // because the path-dep arcgraph is local. Use `--target-dir`
    // pointing inside the consumer so we don't poison the parent
    // workspace's target.
    let output = Command::new(cargo)
        .args(["build", "--quiet", "--manifest-path"])
        .arg(consumer.join("Cargo.toml"))
        .arg("--target-dir")
        .arg(consumer.join("target"))
        .output()
        .expect("spawn cargo build for downstream consumer");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "downstream consumer FAILED to compile against the umbrella. \
         This is the W15α F-1 regression: the doc-test text uses paths \
         not reachable through `arcgraph::*` alone. \
         stdout:\n{stdout}\nstderr:\n{stderr}",
    );

    // Hostile-pin: scan stderr for any E0432 / E0433 error citing one
    // of the underlying `arcgraph_*` crate names — those would mean
    // the doc-test body silently uses a direct `arcgraph_query::*`
    // path that compiled because the parent workspace had them in
    // scope. The output.status.success() above already covers the
    // common case; this is belt-and-braces.
    for crate_name in [
        "arcgraph_core",
        "arcgraph_storage",
        "arcgraph_query",
        "arcgraph_mcp",
        "arcgraph_index",
        "arcgraph_vector",
        "arcgraph_bm25",
        "arcgraph_community",
    ] {
        assert!(
            !stderr.contains(&format!(
                "use of unresolved module or unlinked crate `{crate_name}`"
            )),
            "downstream consumer stderr cites unresolved `{crate_name}` — \
             the doc-test body uses a direct `arcgraph_*` path that breaks \
             portability. stderr:\n{stderr}",
        );
    }

    // Best-effort cleanup. Failure to clean is non-fatal (CI usually
    // re-uses the consumer dir; local dev environments rerun the test
    // which cleans at the top of the function).
    let _ = fs::remove_dir_all(&consumer);
}
