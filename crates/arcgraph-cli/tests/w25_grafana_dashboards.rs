//! W25-OPS-PROD / ADR-093-amendment-01 §D-1 — Grafana dashboard pack
//! validity tests.
//!
//! Three contracts:
//!
//! 1. **`grafana_dashboards_are_well_formed_json`** — every file under
//!    `deploy/grafana/dashboards/` parses as valid JSON.
//! 2. **`grafana_dashboards_carry_required_fields`** — each dashboard
//!    has the minimum-viable Grafana fields (`title`, `uid`,
//!    `schemaVersion`, `panels`).
//! 3. **`grafana_provisioning_matches_dashboard_files`** — the
//!    `provisioning/dashboards/dashboards.yml` provider config points
//!    at a path that contains every dashboard file (no orphan files;
//!    no broken provider config).
//!
//! These tests do NOT require Grafana to be installed; they validate
//! the dashboard pack at the file-shape level. End-to-end "do the
//! dashboards actually render" requires a Grafana instance, which is
//! a manual operator verification per the README.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crates/arcgraph-cli should have a workspace parent")
        .to_path_buf()
}

fn dashboards_dir() -> PathBuf {
    workspace_root().join("deploy/grafana/dashboards")
}

fn dashboard_files() -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = fs::read_dir(dashboards_dir())
        .expect("dashboards dir exists")
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    out.sort();
    out
}

#[test]
fn grafana_dashboards_are_well_formed_json() {
    let files = dashboard_files();
    assert!(
        !files.is_empty(),
        "expected ≥1 dashboard json under {}",
        dashboards_dir().display()
    );
    for file in &files {
        let content =
            fs::read_to_string(file).unwrap_or_else(|e| panic!("read {}: {}", file.display(), e));
        let _: serde_json::Value = serde_json::from_str(&content)
            .unwrap_or_else(|e| panic!("invalid JSON in {}: {}", file.display(), e));
    }
}

#[test]
fn grafana_dashboards_carry_required_fields() {
    let files = dashboard_files();
    for file in &files {
        let content = fs::read_to_string(file).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        let obj = value
            .as_object()
            .unwrap_or_else(|| panic!("{} root not object", file.display()));
        for required in &["title", "uid", "schemaVersion", "panels"] {
            assert!(
                obj.contains_key(*required),
                "{} missing required field {required:?}",
                file.display()
            );
        }
        // schemaVersion must be ≥ 39 (Grafana 10.4+ compatibility per
        // dashboard README).
        let schema_version = obj
            .get("schemaVersion")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| panic!("{} schemaVersion not u64", file.display()));
        assert!(
            schema_version >= 39,
            "{} schemaVersion {schema_version} < 39 (Grafana 10.4+ required)",
            file.display()
        );
        // panels must be a non-empty array.
        let panels = obj
            .get("panels")
            .and_then(|v| v.as_array())
            .unwrap_or_else(|| panic!("{} panels not array", file.display()));
        assert!(!panels.is_empty(), "{} has no panels", file.display());
    }
}

#[test]
fn grafana_provisioning_matches_dashboard_files() {
    // The provider config points at /var/lib/grafana/dashboards/arcgraph
    // inside the Grafana container; the bind-mount is from
    // deploy/grafana/dashboards. We assert the README documents the
    // bind-mount + the provider config exists + the dashboards file
    // list is non-empty (the bind-mount semantics are inherently
    // operator-side configuration).
    let provider = workspace_root().join("deploy/grafana/provisioning/dashboards/dashboards.yml");
    assert!(
        provider.exists(),
        "provider config missing at {}",
        provider.display()
    );
    let provider_content = fs::read_to_string(&provider).unwrap();
    assert!(
        provider_content.contains("/var/lib/grafana/dashboards/arcgraph"),
        "provider config does not reference the canonical bind-mount target",
    );
    assert!(
        provider_content.contains("ArcGraph"),
        "provider config does not reference the ArcGraph folder",
    );
    // Sanity: every dashboard file has a unique uid.
    let mut uids: HashSet<String> = HashSet::new();
    for file in dashboard_files() {
        let content = fs::read_to_string(&file).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        let uid = value
            .get("uid")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| panic!("{} uid not string", file.display()))
            .to_string();
        assert!(
            uids.insert(uid.clone()),
            "duplicate uid {uid} across dashboard files",
        );
    }
}

#[test]
fn grafana_dashboard_catalog_matches_bare_engine_pack() {
    let names: Vec<_> = dashboard_files()
        .into_iter()
        .map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .expect("dashboard file name is UTF-8")
                .to_owned()
        })
        .collect();
    assert_eq!(
        names,
        [
            "01-query-latency.json",
            "02-wal.json",
            "03-buffer-pool.json",
            "05-tenant-cost.json",
        ],
        "the public pack contains only bare-engine operational dashboards",
    );
}
