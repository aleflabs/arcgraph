//! W26-β-1 GA-BOOTSTRAP-WIRING — operator-path regression pin for the
//! shared [`arcgraph_cli::bootstrap::bootstrap_storage_backend`] helper.
//!
//! # What this pins
//!
//! The W18δ-flagged operator-visible gap (issue #439): `arcgraph serve`
//! ingest → `graph.schema` returns EMPTY labels because the catalog-stats
//! hook in `arcgraph_storage::crud::commit` early-returns when
//! `primary.is_none()`. ADR-087 D-2 ratifies the production posture
//! (`CrudStore::new_with_index` required for every deployment surface that
//! consumes `graph.raw_query` / Bolt RUN); the CLI lift was forward-pinned
//! to this v1.0-GA hardening slice.
//!
//! Before W26-β-1, `bootstrap_storage_backend` constructed `CrudStore`
//! via `CrudStore::new()` — no primary index → `crud::commit` skipped the
//! per-tenant `CatalogStats` hook → `graph.schema` always returned empty
//! labels even after a successful `graph.ingest`. This test enforces the
//! ratified D-2 wire: bootstrap → `StorageIngestProvider::ingest` →
//! `StorageSchemaProvider::schema` and assert the ingested labels +
//! rel-types surface in the schema response.
//!
//! # Why in-process (not subprocess)
//!
//! Per the W26-β-1 spawn prompt: "If `arcgraph serve` boot is too heavy
//! for an integration test, an in-process equivalent (programmatic
//! `bootstrap_storage_backend` + commit + schema call) is acceptable as
//! long as it exercises the `new_with_index` wire path." The shared
//! `arcgraph_cli::bootstrap::bootstrap_storage_backend` IS the canonical
//! wire-pattern call site for BOTH `arcgraph` and `arcgraph-mcp-stdio`
//! binaries (extracted in W26-β-1 from byte-identical bodies), so pinning
//! the helper directly pins both binaries simultaneously without the
//! subprocess fork + framed-JSON-RPC overhead.
//!
//! # Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`
//!
//! Load-bearing PRs require ≥1 fault-injection regression test per
//! failure mode (W17 #349 H-1/H-2 founding incidents; W18ε no-op
//! trampoline sweep). The failure mode here is: bootstrap reverts to
//! `CrudStore::new()` → schema returns empty labels. The assertion below
//! (schema labels include the ingested name "TestLabel") fails under
//! that posture — a future regression that drops `new_with_index` from
//! the shared helper surfaces immediately at this test.
//!
//! # ADR provenance
//!
//! - **ADR-087 D-2** — primary-index wiring requirement (this test ratifies
//!   the closure for the CLI lift surface).
//! - **W18δ writeup** — operator-visible gap documentation (forward-pinned
//!   to v1.0-GA deployment-hardening; closed by this slice).
//! - **Issue #439** — tracking issue for this closure.

use std::collections::BTreeMap;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_core::TenantId;
use arcgraph_mcp::storage::{StorageIngestProvider, StorageSchemaProvider};
use arcgraph_mcp::{IngestBatch, IngestProvider, NodeIngest, RelIngest, SchemaProvider};

/// Operator-path regression pin (ADR-087 D-2 + issue #439).
///
/// 1. Bootstrap the per-process storage substrate via the shared helper.
/// 2. Ingest 2 labels (`Person`, `Service`) + 1 rel-type (`USES`) under
///    `TenantId::DEFAULT`.
/// 3. Call `graph.schema` via [`StorageSchemaProvider`].
/// 4. Assert the response carries BOTH ingested labels AND the ingested
///    rel-type with non-zero cardinality — the operator-visible symptom
///    of the W18δ gap.
#[test]
fn bootstrap_storage_backend_wires_primary_index_for_catalog_stats_hook() {
    // 1. Bootstrap. This pin exercises the post-commit CatalogStats hook
    //    (ADR-087 D-2), independent of the durability substrate, so it uses
    //    the ephemeral in-memory mode (W28 / ADR-183).
    let (backend, _durability) = bootstrap_storage_backend(&BootstrapMode::InMemory)
        .expect("bootstrap_storage_backend succeeds");
    let tenant = TenantId::DEFAULT;

    // 2. Ingest. 2 labels (Person, Service) + 1 rel-type (USES).
    //    A minimal but operator-realistic shape; the symptom doesn't
    //    require scale — the catalog hook either fires (per-label
    //    cardinality surfaces) or it doesn't (empty labels list).
    let ingest = StorageIngestProvider::new(backend.clone());
    let batch = IngestBatch {
        nodes: vec![
            NodeIngest {
                external_id: Some("alice".into()),
                label: "Person".into(),
                properties: BTreeMap::new(),
            },
            NodeIngest {
                external_id: Some("bob".into()),
                label: "Person".into(),
                properties: BTreeMap::new(),
            },
            NodeIngest {
                external_id: Some("svc-a".into()),
                label: "Service".into(),
                properties: BTreeMap::new(),
            },
            NodeIngest {
                external_id: Some("svc-b".into()),
                label: "Service".into(),
                properties: BTreeMap::new(),
            },
        ],
        relationships: vec![
            RelIngest {
                external_id: Some("uses-1".into()),
                from_external_id: "alice".into(),
                to_external_id: "svc-a".into(),
                rel_type: "USES".into(),
                properties: BTreeMap::new(),
            },
            RelIngest {
                external_id: Some("uses-2".into()),
                from_external_id: "bob".into(),
                to_external_id: "svc-b".into(),
                rel_type: "USES".into(),
                properties: BTreeMap::new(),
            },
        ],
        acl_grants: vec![],
    };
    let summary = ingest.ingest(tenant, batch).expect("ingest succeeds");
    // Sanity: ingest itself succeeded — the W18δ gap is in the
    // post-commit schema-readback path, not the ingest path. If ingest
    // fails the harness breaks before we can assert the symptom.
    assert_eq!(
        summary.failed_count, 0,
        "ingest had per-record failures: {summary:?}"
    );
    assert_eq!(
        summary.inserted_count, 6,
        "expected 4 nodes + 2 rels inserted, got {summary:?}"
    );

    // 3. graph.schema readback.
    let schema_provider = StorageSchemaProvider::new(backend.clone());
    let schema = schema_provider
        .schema(tenant)
        .expect("schema readback succeeds");

    // 4. Load-bearing assertions — the W18δ operator-visible symptom.
    //    Under the pre-W26-β-1 `CrudStore::new()` posture each of these
    //    assertions fails because the labels + rel_types lists are empty.
    assert!(
        !schema.labels.is_empty(),
        "ADR-087 D-2 regression: bootstrap dropped primary-index wiring; \
         graph.schema returned empty labels list (W18δ operator-visible gap, \
         issue #439). Got: {schema:?}",
    );
    let label_names: Vec<&str> = schema.labels.iter().map(|l| l.name.as_str()).collect();
    assert!(
        label_names.contains(&"Person"),
        "schema.labels missing ingested 'Person'; got {label_names:?}",
    );
    assert!(
        label_names.contains(&"Service"),
        "schema.labels missing ingested 'Service'; got {label_names:?}",
    );
    // Per-label cardinality surfaces only when the CatalogStats hook
    // fires; the pre-fix posture leaves these at None.
    let person = schema
        .labels
        .iter()
        .find(|l| l.name == "Person")
        .expect("Person label present");
    assert_eq!(
        person.cardinality,
        Some(2),
        "Person cardinality should be 2 (alice + bob), got {:?}",
        person.cardinality
    );

    assert!(
        !schema.rel_types.is_empty(),
        "ADR-087 D-2 regression: graph.schema returned empty rel_types list \
         (W18δ operator-visible gap, issue #439). Got: {schema:?}",
    );
    let rel_names: Vec<&str> = schema.rel_types.iter().map(|r| r.name.as_str()).collect();
    assert!(
        rel_names.contains(&"USES"),
        "schema.rel_types missing ingested 'USES'; got {rel_names:?}",
    );
    let uses = schema
        .rel_types
        .iter()
        .find(|r| r.name == "USES")
        .expect("USES rel-type present");
    assert_eq!(
        uses.cardinality,
        Some(2),
        "USES cardinality should be 2 (alice→svc-a, bob→svc-b), got {:?}",
        uses.cardinality
    );

    // Totals also flow through the CatalogStats snapshot only when the
    // hook fires.
    assert_eq!(
        schema.total_node_count,
        Some(4),
        "total_node_count should be 4 (2 Person + 2 Service), got {:?}",
        schema.total_node_count
    );
    assert_eq!(
        schema.total_rel_count,
        Some(2),
        "total_rel_count should be 2 (2 USES), got {:?}",
        schema.total_rel_count
    );
}
