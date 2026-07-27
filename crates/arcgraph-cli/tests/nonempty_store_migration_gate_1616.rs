//! #1616 / #1622 — prevent cold-start reconcile from creating durable
//! record slots stamped `created_lsn = 0`.
//!
//! This gate builds a real non-empty v4 store through production CRUD,
//! cold-starts it twice, migrates it v4 → v5 → v6 through the shipped
//! binary, inspects the v5 record pages on disk, and reads the corpus back.
//! It is a prevention gate: the fixture is produced by the binary under
//! test and does not claim to repair a store already damaged by an older
//! binary.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::data_dir_migration::current_generation;
use arcgraph_core::{LabelId, NodeId, PAGE_SIZE, PartitionId, TenantId};
use arcgraph_storage::crud::{CrudStore, PropertyData, commit, create_node, read_node_with_store};
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::records::SlottedPageRef;
use tempfile::tempdir;

const BIN: &str = env!("CARGO_BIN_EXE_arcgraph");
const RECORDS: u32 = 24;
const PRE_MIGRATION_RESTARTS: usize = 2;

fn crud_for(backend: &arcgraph_mcp::storage::StorageBackend, tenant: TenantId) -> Arc<CrudStore> {
    backend
        .router()
        .route(tenant, PartitionId::ZERO)
        .expect("route tenant")
        .crud()
        .clone()
}

fn payload(i: u32) -> Vec<u8> {
    format!(
        "{{\"seq\":{i},\"pad\":\"{}\"}}",
        "x".repeat(48 + (i as usize % 17))
    )
    .into_bytes()
}

fn label_for(i: u32) -> LabelId {
    LabelId::new(i + 1)
}

fn migrate(dir: &Path) -> std::process::Output {
    Command::new(BIN)
        .args([
            "migrate",
            "upgrade-data-dir",
            "--data-dir",
            dir.to_str().expect("UTF-8 data dir"),
        ])
        .output()
        .expect("spawn arcgraph migrate upgrade-data-dir")
}

fn serve_once(dir: &Path) -> std::process::Output {
    Command::new(BIN)
        .args([
            "serve",
            "--stdio-mcp",
            "--data",
            dir.to_str().expect("UTF-8 data dir"),
            "--admin-http",
            "",
            "--metrics-http",
            "",
            "--drain-grace-seconds",
            "0",
        ])
        .stdin(Stdio::null())
        .output()
        .expect("spawn arcgraph serve")
}

fn print_process_output(label: &str, output: &std::process::Output) {
    eprintln!("{label}: status={}", output.status);
    eprintln!(
        "{label}: stdout:\n{}",
        String::from_utf8_lossy(&output.stdout)
    );
    eprintln!(
        "{label}: stderr:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_migrate_succeeded(leg: &str, output: &std::process::Output) {
    print_process_output(leg, output);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("panicked at"),
        "{leg}: migration panicked (issue #1616)\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(
        output.status.success(),
        "{leg}: migration exited {:?}\nstdout={stdout}\nstderr={stderr}",
        output.status,
    );
}

/// Inspect every durable v5 record page rather than inferring safety from
/// logs or a successful process exit.
fn assert_no_zero_created_lsn_slot(generation: &Path) {
    let tenants_root = generation.join("tenants");
    let mut scanned_pages = 0_u64;
    let mut scanned_records = 0_u64;
    let entries = std::fs::read_dir(&tenants_root)
        .unwrap_or_else(|error| panic!("read {}: {error}", tenants_root.display()));

    for entry in entries {
        let entry = entry.expect("tenant directory entry");
        let record_store = entry.path().join("record.store");
        if !record_store.is_file() {
            continue;
        }
        let mut file = std::fs::File::open(&record_store)
            .unwrap_or_else(|error| panic!("open {}: {error}", record_store.display()));
        let mut page_no = 0_u64;
        loop {
            let mut page = vec![0_u8; PAGE_SIZE];
            if file.read_exact(&mut page).is_err() {
                break;
            }
            if page.iter().all(|byte| *byte == 0) {
                page_no += 1;
                continue;
            }
            let view = SlottedPageRef::open(&page).unwrap_or_else(|error| {
                panic!(
                    "{} page {page_no} is not a valid slotted page: {error:?}",
                    record_store.display()
                )
            });
            scanned_pages += 1;
            for (slot, record) in view.iter_nodes() {
                scanned_records += 1;
                assert_ne!(
                    record.created_lsn,
                    0,
                    "#1616: {} page {page_no} slot {slot:?} holds node {} with created_lsn = 0",
                    record_store.display(),
                    record.id,
                );
            }
        }
    }

    assert!(
        scanned_pages > 0 && scanned_records > 0,
        "zero-LSN oracle scanned nothing ({scanned_pages} pages, {scanned_records} records) \
         under {}",
        tenants_root.display(),
    );
    eprintln!(
        "disk inspection: generation={} pages={} records={} zero_created_lsn_slots=0",
        generation.display(),
        scanned_pages,
        scanned_records,
    );
}

fn assert_corpus_intact(data_dir: &Path, ids: &[NodeId]) {
    let mode = BootstrapMode::Durable {
        data_dir: data_dir.to_path_buf(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).expect("reopen the migrated store");
    let crud = crud_for(&backend, TenantId::DEFAULT);
    let tx = backend.txn_manager().begin(TenantId::DEFAULT);

    for (i, id) in ids.iter().enumerate() {
        let i = u32::try_from(i).expect("corpus index fits u32");
        let record = read_node_with_store(&crud, &tx, *id)
            .unwrap_or_else(|error| panic!("read node {id:?} after migration: {error}"))
            .unwrap_or_else(|| panic!("node {id:?} vanished across migration"));
        assert_eq!(
            record.label_id,
            label_for(i).raw(),
            "node {id:?} label changed across migration"
        );
        let blob_ref = BlobRef::decode(record.property_ref)
            .unwrap_or_else(|| panic!("node {id:?} lost its property reference"));
        let bytes = crud
            .blob_store()
            .get(TenantId::DEFAULT, blob_ref)
            .unwrap_or_else(|error| panic!("property read for node {id:?}: {error}"));
        assert_eq!(
            bytes.as_ref(),
            payload(i).as_slice(),
            "node {id:?} property payload changed across migration"
        );
    }

    tx.abort();
    drop(crud);
    drop(backend);
    drop(guard);
}

fn build_populated_store(data_dir: &Path) -> Vec<NodeId> {
    let mode = BootstrapMode::Durable {
        data_dir: data_dir.to_path_buf(),
    };
    let mut ids = Vec::with_capacity(RECORDS as usize);

    {
        let (backend, guard) = bootstrap_storage_backend(&mode).expect("virgin durable bootstrap");
        let crud = crud_for(&backend, TenantId::DEFAULT);
        let mut created = 0_u32;
        while created < RECORDS {
            let batch = (RECORDS - created).min(5);
            let mut tx = backend.txn_manager().begin(TenantId::DEFAULT);
            for _ in 0..batch {
                let id = create_node(
                    &crud,
                    &mut tx,
                    TenantId::DEFAULT,
                    label_for(created),
                    &PropertyData::Blob(payload(created)),
                )
                .expect("create node");
                ids.push(id);
                created += 1;
            }
            commit(tx, &crud).expect("commit batch");
        }
        drop(crud);
        drop(backend);
        drop(guard);
    }

    for _ in 0..PRE_MIGRATION_RESTARTS {
        let (backend, guard) =
            bootstrap_storage_backend(&mode).expect("cold restart populated store");
        drop(backend);
        drop(guard);
    }

    assert_eq!(ids.len(), RECORDS as usize);
    ids
}

fn version_of(path: &Path) -> u16 {
    arcgraph_storage::check_or_stamp_data_dir(path, true, false).expect("read VERSION")
}

#[test]
fn nonempty_store_upgrades_through_v5_to_v6_and_reads_back_intact() {
    let tmp = tempdir().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let ids = build_populated_store(&data_dir);

    let serve = serve_once(&data_dir);
    print_process_output("serve after two cold restarts", &serve);
    assert!(
        serve.status.success(),
        "serve after two cold restarts exited {:?}",
        serve.status
    );
    assert!(
        !String::from_utf8_lossy(&serve.stderr).contains("panicked at"),
        "serve after two cold restarts panicked"
    );

    let v4_to_v5 = migrate(&data_dir);
    assert_migrate_succeeded("v4->v5", &v4_to_v5);
    let v5_generation = current_generation(&data_dir)
        .expect("read CURRENT")
        .expect("v4->v5 must commit a generation");
    assert_eq!(
        v5_generation.file_name().expect("generation name"),
        "gen-v9"
    );
    assert_eq!(version_of(&v5_generation), 5);
    assert_no_zero_created_lsn_slot(&v5_generation);

    let v5_to_v6 = migrate(&data_dir);
    assert_migrate_succeeded("v5->v6", &v5_to_v6);
    let v6_generation = current_generation(&data_dir)
        .expect("read CURRENT")
        .expect("v5->v6 must commit a generation");
    assert_eq!(
        v6_generation.file_name().expect("generation name"),
        "gen-v10"
    );
    assert_eq!(version_of(&v6_generation), 6);

    assert_corpus_intact(&data_dir, &ids);
}

#[test]
fn virgin_store_upgrade_control_still_passes() {
    let tmp = tempdir().expect("tempdir");
    let data_dir = tmp.path().join("db");
    let mode = BootstrapMode::Durable {
        data_dir: data_dir.clone(),
    };
    let (backend, guard) = bootstrap_storage_backend(&mode).expect("virgin durable bootstrap");
    drop(backend);
    drop(guard);

    assert_migrate_succeeded("virgin v4->v5", &migrate(&data_dir));
    assert_migrate_succeeded("virgin v5->v6", &migrate(&data_dir));
    let generation: PathBuf = current_generation(&data_dir)
        .expect("read CURRENT")
        .expect("virgin upgrade must commit a generation");
    assert_eq!(generation.file_name().expect("generation name"), "gen-v10");
    assert_eq!(version_of(&generation), 6);
}
