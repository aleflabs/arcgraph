//! W25-OPS-PROD / ADR-093-amendment-01 §D-4 — Helm chart shape tests.
//!
//! Three contracts:
//!
//! 1. **`helm_chart_required_files_present`** — Chart.yaml +
//!    values.yaml + at least 5 template files + NOTES.txt all exist.
//! 2. **`helm_chart_yaml_is_well_formed`** — Chart.yaml + values.yaml
//!    parse as YAML.
//! 3. **`helm_chart_lint_passes`** (env-gated) — `helm lint` succeeds
//!    against the chart directory. Skip-by-PANIC if `helm` not on
//!    PATH unless `ARCGRAPH_W25_SKIP_HELM_LINT=1` is set explicitly
//!    (per `feedback_test_env_gate_panic_by_default.md`).

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest_dir
        .parent()
        .and_then(Path::parent)
        .expect("crates/arcgraph-cli should have a workspace parent")
        .to_path_buf()
}

fn chart_dir() -> PathBuf {
    workspace_root().join("deploy/helm/arcgraph")
}

#[test]
fn helm_chart_required_files_present() {
    let chart = chart_dir();
    let must_exist = [
        "Chart.yaml",
        "values.yaml",
        "templates/NOTES.txt",
        "templates/_helpers.tpl",
        "templates/statefulset.yaml",
        "templates/service.yaml",
        "templates/configmap.yaml",
        "templates/serviceaccount.yaml",
        "templates/poddisruptionbudget.yaml",
        "templates/networkpolicy.yaml",
    ];
    for rel in &must_exist {
        let p = chart.join(rel);
        assert!(
            p.exists(),
            "required helm chart file missing: {}",
            p.display()
        );
    }
}

#[test]
fn helm_chart_yaml_is_well_formed() {
    let chart = chart_dir();
    let chart_yaml = fs::read_to_string(chart.join("Chart.yaml")).expect("read Chart.yaml");
    let values_yaml = fs::read_to_string(chart.join("values.yaml")).expect("read values.yaml");
    let _chart: serde_yaml::Value =
        serde_yaml::from_str(&chart_yaml).expect("Chart.yaml not valid YAML");
    let _values: serde_yaml::Value =
        serde_yaml::from_str(&values_yaml).expect("values.yaml not valid YAML");
    // Chart name + version + appVersion must be present.
    let chart_doc: serde_yaml::Value = serde_yaml::from_str(&chart_yaml).unwrap();
    for required in &["apiVersion", "name", "version", "appVersion", "type"] {
        assert!(
            chart_doc.get(*required).is_some(),
            "Chart.yaml missing required field {required:?}",
        );
    }
    let api_version = chart_doc
        .get("apiVersion")
        .and_then(|v| v.as_str())
        .unwrap();
    assert_eq!(api_version, "v2", "Helm Chart.yaml apiVersion must be v2");
}

/// Try to spawn `helm` directly; return Err on NotFound. Avoids the
/// `which` crate dependency by leveraging the OS's PATH search via
/// `Command::new`.
fn try_spawn_helm(args: &[&std::ffi::OsStr]) -> Result<std::process::Output, std::io::Error> {
    Command::new("helm").args(args).output()
}

/// Helm-lint integration test. PANICs by default if `helm` is not on
/// PATH (per `feedback_test_env_gate_panic_by_default.md` — soft-skip
/// would silently let the chart drift). Soft-skip via
/// `ARCGRAPH_W25_SKIP_HELM_LINT=1` (CI sets this when helm-cli is
/// unavailable; the gauntlet env can override per-host).
#[test]
fn helm_chart_lint_passes() {
    if std::env::var("ARCGRAPH_W25_SKIP_HELM_LINT").ok().as_deref() == Some("1") {
        eprintln!("ARCGRAPH_W25_SKIP_HELM_LINT=1 — skipping helm lint");
        return;
    }
    let chart = chart_dir();
    let chart_os = chart.as_os_str();
    let args: Vec<&std::ffi::OsStr> = vec![std::ffi::OsStr::new("lint"), chart_os];
    let out = match try_spawn_helm(&args) {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => panic!(
            "`helm` not on PATH; install Helm >=3.0 or set \
             ARCGRAPH_W25_SKIP_HELM_LINT=1 to skip explicitly. Soft-skipping \
             by default is the silent-bypass bug class per \
             `feedback_test_env_gate_panic_by_default.md`."
        ),
        Err(e) => panic!("helm spawn failed: {e}"),
    };
    if !out.status.success() {
        panic!(
            "helm lint failed:\nstdout:\n{}\nstderr:\n{}\n",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    }
}

/// Helm-template smoke test — `helm template` produces parseable YAML.
/// Same env-gate discipline as helm lint.
#[test]
fn helm_chart_template_renders_to_valid_yaml() {
    if std::env::var("ARCGRAPH_W25_SKIP_HELM_LINT").ok().as_deref() == Some("1") {
        eprintln!("ARCGRAPH_W25_SKIP_HELM_LINT=1 — skipping helm template");
        return;
    }
    let chart = chart_dir();
    let chart_os = chart.as_os_str();
    let args: Vec<&std::ffi::OsStr> = vec![
        std::ffi::OsStr::new("template"),
        std::ffi::OsStr::new("test-release"),
        chart_os,
    ];
    let out = match try_spawn_helm(&args) {
        Ok(o) => o,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            panic!("`helm` not on PATH; set ARCGRAPH_W25_SKIP_HELM_LINT=1 to skip.")
        }
        Err(e) => panic!("helm spawn failed: {e}"),
    };
    if !out.status.success() {
        panic!(
            "helm template failed:\nstderr:\n{}\n",
            String::from_utf8_lossy(&out.stderr),
        );
    }
    let rendered = String::from_utf8_lossy(&out.stdout);
    // Multi-document YAML — split + parse each one.
    let docs: Vec<&str> = rendered.split("\n---\n").collect();
    assert!(
        docs.len() >= 4,
        "expected ≥4 rendered yaml documents (statefulset + service + configmap + sa); got {}",
        docs.len()
    );
    for (i, doc) in docs.iter().enumerate() {
        let trimmed = doc.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        serde_yaml::from_str::<serde_yaml::Value>(trimmed)
            .unwrap_or_else(|e| panic!("rendered doc {i} not valid YAML: {e}\n{trimmed}"));
    }
}
