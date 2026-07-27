//! W18δ Task §3 — `arcgraph migrate` subprocess integration test.
//!
//! Drives the binary with both `from-neo4j-cypher` and `from-neo4j-csv`
//! input shapes and asserts the binary prints the expected
//! `inserted=` / `failed=` line.
//!
//! Per `feedback_test_env_gate_panic_by_default.md`: this test is NOT
//! env-gated — it exercises the in-process ingest path with a tempdir-
//! local fixture, no external dependencies. The Docker-Neo4j round-trip
//! test (W18δ addendum item 3 — Northwind) is env-gated and lives in
//! `tests/migrate_neo4j_round_trip.rs` (forward-pin: lands when the CI
//! Docker neo4j is provisioned).

use std::io::Write;
use std::process::Command;

use tempfile::NamedTempFile;

const BIN: &str = env!("CARGO_BIN_EXE_arcgraph");

#[test]
fn migrate_from_neo4j_cypher_subprocess_ingests_nodes_and_rels() {
    let mut script = NamedTempFile::new().expect("tmp");
    writeln!(
        script,
        r#"
            CREATE (n:Person {{name: 'Alice', neo4j_id: 1}});
            CREATE (n:Person {{name: 'Bob', neo4j_id: 2}});
            MATCH (a {{neo4j_id: 1}}), (b {{neo4j_id: 2}}) CREATE (a)-[:KNOWS {{since: 2020}}]->(b);
        "#,
    )
    .unwrap();
    script.flush().unwrap();
    let output = Command::new(BIN)
        .args([
            "migrate",
            "from-neo4j-cypher",
            script.path().to_str().unwrap(),
        ])
        .output()
        .expect("spawn arcgraph migrate");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "migrate exited non-zero: status={:?} stdout=\n{stdout}\nstderr=\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("inserted=3"),
        "expected 3 inserts (2 nodes + 1 rel); got: {stdout}",
    );
    assert!(
        stdout.contains("failed=0"),
        "expected zero failures: {stdout}"
    );
}

#[test]
fn migrate_from_neo4j_csv_subprocess_ingests_nodes_and_rels() {
    let mut nodes_file = NamedTempFile::new().expect("tmp");
    writeln!(nodes_file, ":ID,name:string,age:int,:LABEL").unwrap();
    writeln!(nodes_file, "1,Alice,30,Person").unwrap();
    writeln!(nodes_file, "2,Bob,25,Person").unwrap();
    nodes_file.flush().unwrap();
    let mut rels_file = NamedTempFile::new().expect("tmp");
    writeln!(rels_file, ":START_ID,:END_ID,:TYPE,since:int").unwrap();
    writeln!(rels_file, "1,2,KNOWS,2020").unwrap();
    rels_file.flush().unwrap();
    let output = Command::new(BIN)
        .args([
            "migrate",
            "from-neo4j-csv",
            "--nodes",
            nodes_file.path().to_str().unwrap(),
            "--rels",
            rels_file.path().to_str().unwrap(),
        ])
        .output()
        .expect("spawn arcgraph migrate");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "migrate exited non-zero: status={:?} stdout=\n{stdout}\nstderr=\n{stderr}",
        output.status
    );
    assert!(
        stdout.contains("inserted=3"),
        "expected 3 inserts (2 nodes + 1 rel); got: {stdout}"
    );
    assert!(
        stdout.contains("failed=0"),
        "expected zero failures: {stdout}"
    );
}

#[test]
fn migrate_from_neo4j_cypher_rejects_malformed_input() {
    let mut script = NamedTempFile::new().expect("tmp");
    // `date(...)` value not in the W18δ migrator taxonomy.
    writeln!(
        script,
        "CREATE (n:Person {{birthdate: date('2020-01-01')}});"
    )
    .unwrap();
    script.flush().unwrap();
    let output = Command::new(BIN)
        .args([
            "migrate",
            "from-neo4j-cypher",
            script.path().to_str().unwrap(),
        ])
        .output()
        .expect("spawn arcgraph migrate");
    assert!(
        !output.status.success(),
        "migrate must reject malformed input; got status {:?}",
        output.status
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported property value"),
        "expected unsupported-value error; got: {stderr}"
    );
}
