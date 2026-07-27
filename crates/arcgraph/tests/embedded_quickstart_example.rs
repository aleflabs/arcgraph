//! Integration test for the M6-01 `embedded_quickstart` example.
//!
//! Spawns `cargo run --example embedded_quickstart` as a subprocess
//! and verifies (a) the example exits 0 and (b) its stdout pins the
//! expected row count + envelope. This is the integration-side smoke
//! that the umbrella's re-export surface stays live for embedded
//! callers (the doc-test in `lib.rs` covers the in-process path).

use std::process::Command;

#[test]
fn example_embedded_quickstart_runs_to_completion() {
    // `env!("CARGO")` is set by cargo when compiling tests — it
    // resolves to the cargo binary in use, so we don't need to
    // probe PATH. Same pattern Cargo's own integration tests use.
    let cargo = env!("CARGO");

    // `-p arcgraph` pins the package even though the test crate is
    // already in the arcgraph package — explicit-is-better-than-
    // implicit when the test runs in a workspace.
    let output = Command::new(cargo)
        .args([
            "run",
            "--quiet",
            "-p",
            "arcgraph",
            "--example",
            "embedded_quickstart",
        ])
        .output()
        .expect("spawn cargo run --example embedded_quickstart");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "example failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        output.status,
    );
    assert!(
        stdout.contains("50 rows materialized"),
        "expected '50 rows materialized' in stdout, got: {stdout}",
    );
    assert!(
        stdout.contains("ingested 100 Person nodes"),
        "expected ingestion line in stdout, got: {stdout}",
    );
}
