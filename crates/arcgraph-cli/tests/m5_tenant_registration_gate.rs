//! M5-D1b tenant-registration gates (#1513; `docs/design/M5D-REDESIGN-AMENDMENT.md`
//! §10 Risk-2 ruling; refs #1457, refs #1404 — M6-entry precondition).
//!
//! The #1513 finding: at cold open, a fresh-loaded generation's tenants are
//! servable (data on disk, INV-M5.20) but NOT catalog-listed — so the
//! PRODUCTION dispatch `route(tenant, PartitionId::ZERO)` (the exact call
//! `arcgraph-mcp` issues at `storage/adapters.rs` `crud_for` and
//! `storage/bolt.rs` `read_access`) returns `RoutingError::UnknownTenant`.
//! The ruled fix: cold open registers every tenant from the MANIFEST
//! `tenant_census` through the SAME catalog path production routing consults
//! (`SystemCatalog::register_tenant` → `list_tenants` → the router guard).
//!
//! **Dispatch honesty (memory: gates must exercise the ARM production
//! dispatches to):** every serving assertion below routes
//! `route(LOAD_TENANT, ZERO)` — the per-tenant guard the adapters hit —
//! and reads ONLY through the handle THAT dispatch returns. The D2 gates'
//! `route(TenantId::DEFAULT, …)` + shared-crud read is exactly the shape
//! that masked #1513 and is deliberately NOT used here.
//!
//! Gates (all RED-on-revert via cfg-gated seams in
//! `arcgraph-cli::bootstrap::register_census_tenants`, armed in child
//! processes of this binary; CI lane `arcgraph-cli-release-fault-injection`):
//!
//! 1. **route-resolves-loaded-tenant** — after production load + production
//!    cold open, `route(LOAD_TENANT, ZERO)` is Ok and
//!    `catalog.list_tenants()` includes the loaded tenant. RED-on-revert:
//!    `ARCGRAPH_M5_SKIP_CENSUS_REGISTRATION` (the register-nothing
//!    total-bypass mutant) → the child fails WITH the exact #1513 marker
//!    `unknown tenant: tenant_id=91` — the mutation-test of this gate.
//! 2. **serve-through-route** — property + adjacency reads for the loaded
//!    tenant, through the handle returned by the production dispatch.
//!    RED-on-revert: unregistered → the read fails AT ROUTING.
//! 3. **idempotent-recovery** — cold-open twice + crash-mid-registration
//!    (`ARCGRAPH_M5_CRASH_MID_CENSUS_REGISTRATION` child aborts between
//!    census entries) → next cold open completes the set; no duplicates.
//! 4. **census-authority** — the registered set EQUALS the MANIFEST
//!    `tenant_census` exactly. RED-on-revert:
//!    `ARCGRAPH_M5_CENSUS_REGISTRATION_SUBSET` (register a strict subset).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};
#[cfg(feature = "fault-injection")]
use std::process::Command;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::m5_load::{LoadFormat, LoadOutcome, canonical_property_bag, load_data_dir};
use arcgraph_core::{NodeId, PartitionId, TenantId, TypeId};
use arcgraph_storage::crud::{read_node_with_store, scan_in, scan_out};
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::read_data_dir_manifest;
use tempfile::tempdir;

const LOAD_TENANT: TenantId = TenantId::new(91);
const FINAL_GENERATION: &str = "gen-load-v6";

#[cfg(feature = "fault-injection")]
const SKIP_ENV: &str = "ARCGRAPH_M5_SKIP_CENSUS_REGISTRATION";
#[cfg(feature = "fault-injection")]
const SUBSET_ENV: &str = "ARCGRAPH_M5_CENSUS_REGISTRATION_SUBSET";
const CRASH_ENV: &str = "ARCGRAPH_M5_CRASH_MID_CENSUS_REGISTRATION";
/// Parent → crash-child handshake: the pre-loaded data dir the child must
/// cold-open (and abort inside).
const CRASH_CHILD_DIR_ENV: &str = "ARCGRAPH_M5_D1B_CRASH_CHILD_DIR";

#[cfg(feature = "fault-injection")]
fn armed(var: &str) -> bool {
    std::env::var_os(var).is_some()
}

#[cfg(feature = "fault-injection")]
fn any_seam_armed() -> bool {
    armed(SKIP_ENV) || armed(SUBSET_ENV) || armed(CRASH_ENV)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Fixture: small multi-property, multi-edge input (production loader)
// ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct FixtureNode {
    external: Vec<u8>,
    label: u32,
    float_bits: u64,
    opaque: Vec<u8>,
}

#[derive(Debug, Clone)]
struct FixtureRel {
    external: Vec<u8>,
    source: Vec<u8>,
    target: Vec<u8>,
    type_id: u32,
    float_bits: u64,
    opaque: Vec<u8>,
}

struct Fixture {
    /// Canonically ordered (external-id byte order == dense id order).
    nodes: Vec<FixtureNode>,
    rels: Vec<FixtureRel>,
}

type AdjacencyMap = BTreeMap<(u64, u32), Vec<(u64, u64)>>;

impl Fixture {
    fn node_id(&self, external: &[u8]) -> u64 {
        self.nodes
            .iter()
            .position(|node| node.external == external)
            .map(|index| index as u64 + 1)
            .expect("fixture node exists")
    }

    fn expected_node_bag(&self, id: u64) -> Vec<u8> {
        let node = &self.nodes[usize::try_from(id - 1).unwrap()];
        canonical_property_bag(node.float_bits, &node.opaque)
    }

    fn expected_adjacency(&self) -> (AdjacencyMap, AdjacencyMap) {
        let mut out: AdjacencyMap = BTreeMap::new();
        let mut inn: AdjacencyMap = BTreeMap::new();
        for (index, rel) in self.rels.iter().enumerate() {
            let rel_id = index as u64 + 1;
            let source = self.node_id(&rel.source);
            let target = self.node_id(&rel.target);
            out.entry((source, rel.type_id))
                .or_default()
                .push((target, rel_id));
            inn.entry((target, rel.type_id))
                .or_default()
                .push((source, rel_id));
        }
        for entries in out.values_mut().chain(inn.values_mut()) {
            entries.sort_unstable();
        }
        (out, inn)
    }
}

fn registration_fixture() -> Fixture {
    let mut nodes = Vec::new();
    for index in 0_u32..4 {
        nodes.push(FixtureNode {
            external: format!("m5d1b-node-{index:04}").into_bytes(),
            label: 7 + (index % 3),
            float_bits: 0x3ff0_0000_0000_0000 + u64::from(index), // 1.0 ± ULPs
            opaque: format!("payload-{index}").into_bytes(),
        });
    }
    nodes.sort_by(|left, right| left.external.cmp(&right.external));
    let mut rels = Vec::new();
    for (index, (source, target, ty)) in
        [(0_usize, 1_usize, 3_u32), (1, 2, 3), (2, 3, 9), (3, 0, 9)]
            .into_iter()
            .enumerate()
    {
        rels.push(FixtureRel {
            external: format!("m5d1b-rel-{index:04}").into_bytes(),
            source: nodes[source].external.clone(),
            target: nodes[target].external.clone(),
            type_id: ty,
            float_bits: 0x3fd5_5555_5555_5555,
            opaque: format!("edge-{index}").into_bytes(),
        });
    }
    rels.sort_by(|left, right| left.external.cmp(&right.external));
    Fixture { nodes, rels }
}

/// Independent native producer (literal schema formatting — deliberately
/// NOT the loader's encoders; the D2 oracle shape).
fn write_native_fixture(fixture: &Fixture, path: &Path) {
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(File::create(path).unwrap());
    for node in &fixture.nodes {
        writeln!(
            writer,
            "{{\"kind\":\"node\",\"external_id\":\"{}\",\"label_or_type\":{},\"float_bits\":\"{:016x}\",\"opaque\":\"{}\"}}",
            hex(&node.external),
            node.label,
            node.float_bits,
            hex(&node.opaque),
        )
        .unwrap();
    }
    for rel in &fixture.rels {
        writeln!(
            writer,
            "{{\"kind\":\"relationship\",\"external_id\":\"{}\",\"source_id\":\"{}\",\"target_id\":\"{}\",\"label_or_type\":{},\"float_bits\":\"{:016x}\",\"opaque\":\"{}\"}}",
            hex(&rel.external),
            hex(&rel.source),
            hex(&rel.target),
            rel.type_id,
            rel.float_bits,
            hex(&rel.opaque),
        )
        .unwrap();
    }
    writer.flush().unwrap();
}

/// Production load into a fresh dir; returns (tempdir guard, data root).
fn load_fixture(fixture: &Fixture) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.jsonl");
    write_native_fixture(fixture, &input);
    let root = dir.path().join("data");
    let outcome = load_data_dir(&input, LoadFormat::Native, &root, LOAD_TENANT)
        .expect("production load of the registration fixture");
    let LoadOutcome::Loaded(report) = outcome else {
        panic!("fresh load did not build: {outcome:?}");
    };
    assert_eq!(report.nodes as usize, fixture.nodes.len());
    assert_eq!(report.relationships as usize, fixture.rels.len());
    assert!(
        root.join(FINAL_GENERATION).is_dir(),
        "committed generation missing"
    );
    (dir, root)
}

/// The MANIFEST `tenant_census` of the committed generation (sorted raws).
fn manifest_census(root: &Path) -> Vec<u64> {
    let mut census = read_data_dir_manifest(&root.join(FINAL_GENERATION))
        .expect("read committed generation MANIFEST")
        .expect("committed generation has a MANIFEST")
        .tenant_census
        .expect("fresh-load MANIFEST carries a tenant census");
    census.sort_unstable();
    census
}

/// Sorted registered tenant raws, read from the SAME catalog the router
/// routes against (`MultiTenantRouter::catalog` → `list_tenants`).
fn registered_raws(backend: &arcgraph_mcp::storage::StorageBackend) -> Vec<u64> {
    let mut raws: Vec<u64> = backend
        .router()
        .catalog()
        .list_tenants()
        .into_iter()
        .map(|record| record.tenant_id.raw())
        .collect();
    raws.sort_unstable();
    raws
}

/// Re-run `test_name` in a child of this binary with `env_var` armed; the
/// child MUST fail (RED-on-revert / mutation-test), and when `marker` is
/// given the failure output MUST contain it — pinning WHERE the child
/// failed (e.g. the router's `UnknownTenant` guard), not just THAT it
/// failed.
#[cfg(feature = "fault-injection")]
fn assert_red_under(test_name: &str, env_var: &str, marker: Option<&str>) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env(env_var, "1")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "{test_name} stayed GREEN under armed {env_var} — the gate is a no-op\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    if let Some(marker) = marker {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stdout.contains(marker) || stderr.contains(marker),
            "{test_name} went red under {env_var} but WITHOUT the expected marker {marker:?} — \
             the red is not the routing-guard failure this gate pins\nstdout:\n{stdout}\nstderr:\n{stderr}",
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 1 — route-resolves-loaded-tenant (the #1513 contract)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn gate1_route_resolves_loaded_tenant_after_cold_open() {
    let fixture = registration_fixture();
    let (_dir, root) = load_fixture(&fixture);

    // Production cold open of the committed generation.
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: root.clone(),
    })
    .expect("loaded store must cold-open through production bootstrap");

    // THE production dispatch FIRST: route(loaded_tenant, ZERO) —
    // identical router type + method + arguments as arcgraph-mcp
    // storage/adapters.rs `crud_for` and storage/bolt.rs `read_access`.
    // Probed before any catalog-surface assertion so the armed
    // register-nothing mutant reds THIS dispatch with the exact #1513
    // probe result (`unknown tenant: tenant_id=91`), proving the gate
    // traces the router guard, not a sibling read.
    let routed = backend
        .router()
        .route(LOAD_TENANT, PartitionId::ZERO)
        .unwrap_or_else(|error| {
            panic!("#1513: production route(loaded_tenant, ZERO) dispatch failed: {error}")
        });
    assert_eq!(
        routed.tenant(),
        LOAD_TENANT,
        "routed handle is tenant-keyed"
    );

    // catalog.list_tenants() must include the loaded tenant — the SAME
    // catalog surface the router's UnknownTenant guard consults.
    let raws = registered_raws(&backend);
    assert!(
        raws.contains(&LOAD_TENANT.raw()),
        "loaded tenant {} not in catalog.list_tenants() after cold open (got {raws:?})",
        LOAD_TENANT.raw(),
    );

    drop(routed);
    drop(backend);
    drop(guard);

    // Mutation-test of this gate (CI fault-injection lane): the
    // register-nothing total-bypass mutant MUST red it, and the red MUST
    // be the router guard (the exact #1513 repro), not an unrelated
    // failure.
    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under(
            "gate1_route_resolves_loaded_tenant_after_cold_open",
            SKIP_ENV,
            Some("unknown tenant: tenant_id=91"),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 2 — serve-through-route (reads via the production dispatch)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn gate2_serve_through_route_returns_loaded_data() {
    let fixture = registration_fixture();
    let (_dir, root) = load_fixture(&fixture);

    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: root.clone(),
    })
    .expect("loaded store must cold-open through production bootstrap");

    // Every read below flows through the handle returned by the
    // PRODUCTION per-tenant dispatch — the MCP/Bolt shape. NO
    // route(DEFAULT) shared-handle fallback exists in this gate: if the
    // loaded tenant is unregistered, the gate fails HERE, at routing.
    let routed = backend
        .router()
        .route(LOAD_TENANT, PartitionId::ZERO)
        .unwrap_or_else(|error| {
            panic!("serve-through-route failed AT ROUTING (the #1513 class): {error}")
        });

    let reader = backend.txn_manager().begin(LOAD_TENANT);

    // Property reads: record -> property_ref -> BlobRef -> blob get_bag.
    for (index, node) in fixture.nodes.iter().enumerate() {
        let id = index as u64 + 1;
        let record = read_node_with_store(routed.crud(), &reader, NodeId::new(id))
            .expect("read loaded node through routed handle")
            .unwrap_or_else(|| panic!("loaded node {id} missing through route(loaded_tenant)"));
        assert_eq!(record.label_id, node.label, "node {id} label");
        let blob_ref = BlobRef::decode(record.property_ref)
            .unwrap_or_else(|| panic!("node {id} lost its property payload"));
        let bag = routed
            .crud()
            .blob_store()
            .get_bag(LOAD_TENANT, blob_ref)
            .expect("served bag readable through routed handle");
        assert_eq!(
            &*bag,
            &fixture.expected_node_bag(id)[..],
            "serve-through-route: node {id} property bag differs from input"
        );
    }

    // Adjacency reads (both directions), same routed handle.
    let (expected_out, expected_in) = fixture.expected_adjacency();
    for ((owner, type_id), entries) in &expected_out {
        let mut scanned: Vec<(u64, u64)> = scan_out(
            routed.crud(),
            &reader,
            NodeId::new(*owner),
            Some(TypeId::new(*type_id)),
        )
        .map(|entry| (entry.dst_id, entry.rel_id))
        .collect();
        scanned.sort_unstable();
        assert_eq!(
            &scanned, entries,
            "serve-through-route: out-adjacency of node {owner} type {type_id} differs"
        );
    }
    for ((owner, type_id), entries) in &expected_in {
        let mut scanned: Vec<(u64, u64)> = scan_in(
            routed.crud(),
            &reader,
            NodeId::new(*owner),
            Some(TypeId::new(*type_id)),
        )
        .expect("in-adjacency scan through routed handle")
        .into_iter()
        .map(|entry| (entry.dst_id, entry.rel_id))
        .collect();
        scanned.sort_unstable();
        assert_eq!(
            &scanned, entries,
            "serve-through-route: in-adjacency of node {owner} type {type_id} differs"
        );
    }

    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);

    // RED-on-revert: unregistered → this gate fails at routing.
    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under(
            "gate2_serve_through_route_returns_loaded_data",
            SKIP_ENV,
            Some("unknown tenant: tenant_id=91"),
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 3 — idempotent-recovery (re-open twice + crash mid-registration)
// ─────────────────────────────────────────────────────────────────────

/// Assert the registered set equals the census with NO duplicate catalog
/// entries, then return it.
fn assert_registered_exact(backend: &arcgraph_mcp::storage::StorageBackend, census: &[u64]) {
    let raws = registered_raws(backend);
    let mut deduped = raws.clone();
    deduped.dedup();
    assert_eq!(raws, deduped, "duplicate catalog entries after cold open");
    assert_eq!(
        raws, census,
        "registered tenant set differs from the MANIFEST census"
    );
}

#[test]
fn gate3_idempotent_recovery_reopen_and_crash_mid_registration() {
    let fixture = registration_fixture();
    let (_dir, root) = load_fixture(&fixture);
    let census = manifest_census(&root);

    // Cold open #1.
    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: root.clone(),
        })
        .expect("cold open #1");
        assert_registered_exact(&backend, &census);
        drop(backend);
        drop(guard);
    }

    // Cold open #2 over the same dir: identical set, no drift, no dupes,
    // and the production dispatch still resolves.
    {
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: root.clone(),
        })
        .expect("cold open #2");
        assert_registered_exact(&backend, &census);
        backend
            .router()
            .route(LOAD_TENANT, PartitionId::ZERO)
            .expect("route resolves on re-open");
        drop(backend);
        drop(guard);
    }

    // Crash-mid-registration: a child cold-opens with the abort seam
    // armed (dies between the first and second census entries — the
    // loaded tenant is NOT yet registered at the abort point). The next
    // cold open must complete the set.
    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "crash_mid_registration_child",
                "--nocapture",
                "--include-ignored",
            ])
            .env(CRASH_ENV, "1")
            .env(CRASH_CHILD_DIR_ENV, &root)
            .output()
            .unwrap();
        assert!(
            !output.status.success(),
            "crash child survived the armed abort seam\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("ARCGRAPH_M5_CRASH_MID_CENSUS_REGISTRATION"),
            "crash child died, but not at the mid-registration abort point\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr),
        );

        // Recovery: the next cold open completes registration.
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: root.clone(),
        })
        .expect("cold open after crash-mid-registration");
        assert_registered_exact(&backend, &census);
        backend
            .router()
            .route(LOAD_TENANT, PartitionId::ZERO)
            .expect("route resolves after crash-mid-registration recovery");
        drop(backend);
        drop(guard);
    }
}

/// Crash-fixture child body (spawned by gate 3 with the abort seam +
/// data-dir handshake armed; `#[ignore]` keeps it out of the normal
/// suite). Cold-opens the pre-loaded dir; the armed seam aborts the
/// process mid-registration.
#[test]
#[ignore = "gate3 crash-fixture child; runs only via the armed subprocess"]
fn crash_mid_registration_child() {
    let Some(dir) = std::env::var_os(CRASH_CHILD_DIR_ENV) else {
        panic!("{CRASH_CHILD_DIR_ENV} not set; this child only runs under gate 3");
    };
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: PathBuf::from(dir),
    })
    .expect("crash child cold open");
    // Unreachable under the armed seam (the process aborts inside
    // bootstrap). If reached, the seam is unwired — fail loud.
    drop(backend);
    drop(guard);
    panic!("crash seam did not fire: bootstrap completed under {CRASH_ENV}");
}

// ─────────────────────────────────────────────────────────────────────
// Gate 4 — census-authority (registered set == MANIFEST census exactly)
// ─────────────────────────────────────────────────────────────────────

#[test]
fn gate4_census_authority_registered_set_equals_manifest_census() {
    let fixture = registration_fixture();
    let (_dir, root) = load_fixture(&fixture);
    let census = manifest_census(&root);
    assert!(
        census.contains(&LOAD_TENANT.raw()) && census.contains(&TenantId::DEFAULT.raw()),
        "fixture census must carry DEFAULT + the loaded tenant (got {census:?})"
    );

    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: root.clone(),
    })
    .expect("loaded store must cold-open through production bootstrap");
    assert_registered_exact(&backend, &census);
    drop(backend);
    drop(guard);

    // RED-on-revert: a strict-subset registration MUST red this gate
    // (missing tenants); the total bypass reds it too.
    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under(
            "gate4_census_authority_registered_set_equals_manifest_census",
            SUBSET_ENV,
            Some("registered tenant set differs from the MANIFEST census"),
        );
        assert_red_under(
            "gate4_census_authority_registered_set_equals_manifest_census",
            SKIP_ENV,
            Some("registered tenant set differs from the MANIFEST census"),
        );
    }
}
