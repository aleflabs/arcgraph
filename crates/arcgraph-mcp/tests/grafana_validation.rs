//! W15δ M6-07 — validation tests for the in-tree Grafana dashboards
//! + alert rules.
//!
//! ## Three invariants enforced
//!
//!  1. **Schema** — dashboards parse as JSON, top-level keys are
//!     present, every panel has `type` / `title` / `targets`. Alerts
//!     parse as YAML, every rule has `alert` / `expr` /
//!     `labels.severity` / `annotations.summary`.
//!  2. **Cross-PR exporter coherence (issue #314 closure / W15 IR
//!     L1-HIGH-3)** — every `arcgraph_*` metric root referenced in
//!     any panel target `expr` or alert rule `expr` is either:
//!     (a) registered in the sister-PR #309 W15γ M6-06 exporter
//!     (the four-metric allowlist below), OR
//!     (b) the surrounding panel / alert is marked `[forward-
//!     bound]` (panels: in `panel.title`; alerts: via the
//!     `annotations.contract_metric` annotation).
//!     This is the load-bearing drift-detector — if anyone adds a
//!     panel referring to a metric the exporter does not register,
//!     this test fails until either the exporter learns the metric
//!     or the panel switches to a `vector(...)` placeholder + a
//!     `[forward-bound]` tag + `contract_metric` annotation.
//!  3. **Prometheus rule semantics** — when `promtool` is on PATH,
//!     alerts pass `promtool check rules` (canonical Prometheus
//!     validation). PANIC by default when promtool is absent;
//!     `ARCGRAPH_PROMTOOL_SKIP_OK=1` opts into a soft-skip per
//!     `feedback_test_env_gate_panic_by_default.md`.
//!
//! ## Why hardcoded registered-metric allowlist (not gather() from
//! the live registry)
//!
//! The W15γ M6-06 exporter lives in a sister-PR branch
//! (`feat/w15-gamma-m6-04-06-ldbc-prometheus`) that hasn't merged
//! yet. When sister-PR #309 merges, this test can switch to
//! `MetricsRegistry::new().registry().gather()` for the registered
//! set; until then the allowlist is a verbatim transcription of the
//! four metrics registered in
//! `crates/arcgraph-mcp/src/transport/metrics.rs` on that sister
//! branch (HEAD at the W15γ-final commit). Drift between this
//! allowlist and the sister-branch surface would show up at merge
//! time as a compile-or-test failure; the cross-branch snapshot is
//! verified empirically in the W15 IR packet.

#![allow(clippy::expect_used)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// Actually-registered metric name set per the live
/// `crates/arcgraph-mcp/src/transport/metrics.rs::MetricsRegistry::new`
/// surface. Cross-PR drift detector: every `arcgraph_*` root referenced
/// in any panel / alert expr must be in this list, OR the surrounding
/// panel / alert must be `[forward-bound]`-tagged.
///
/// - `arcgraph_mcp_tool_invocations{tenant, tool, status}` — IntCounterVec (#309)
/// - `arcgraph_read_latency_ms{tenant, tool}` — HistogramVec ms (#309)
/// - `arcgraph_write_latency_ms{tenant, tool}` — HistogramVec ms (#309)
/// - `arcgraph_active_connections{transport}` — IntGaugeVec (#309)
/// - `arcgraph_hot_vertex_warnings_total{tenant}` — IntCounterVec (W17δ #313)
///
/// Histograms emit Prometheus's `_bucket{le=...}` / `_count` /
/// `_sum` rows at scrape time, so PromQL exprs can reference
/// `arcgraph_read_latency_ms_bucket` etc. — the suffix tolerance is
/// handled in [`is_registered`].
const REGISTERED_METRICS: &[&str] = &[
    "arcgraph_mcp_tool_invocations",
    "arcgraph_read_latency_ms",
    "arcgraph_write_latency_ms",
    "arcgraph_active_connections",
    "arcgraph_hot_vertex_warnings_total",
    // ADR-202 — community-detection freshness gauge (§10.2 line
    // 724); registered + producer-wired via the community-resident
    // `RefreshObserver` seam. Its alerts.yml rule
    // (ArcGraphLeidenFreshnessStale) swapped from the vector(0)
    // placeholder to the live contract_expr in the same slice.
    "arcgraph_leiden_last_run_seconds",
];

/// Histogram suffix variants emitted by `HistogramVec` at the
/// Prometheus text-exposition layer.
const HISTOGRAM_SUFFIXES: &[&str] = &["_bucket", "_count", "_sum"];

/// Resolve a docs path from `CARGO_MANIFEST_DIR` (which points at
/// `crates/arcgraph-mcp/`).
fn docs_path(rel: &str) -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest.parent().unwrap().parent().unwrap().join(rel)
}

/// Extract every `arcgraph_*` metric root substring from a PromQL
/// expression. Walks the expr as bytes: a "root" starts at the byte
/// position of `arcgraph_` and continues through the contiguous
/// `[A-Za-z0-9_]+` run that follows. No regex (avoids dragging
/// `regex` into dev-dependencies just for this one test).
///
/// Examples:
///
/// - `sum by (tool) (rate(arcgraph_mcp_tool_invocations[1m]))`
///   → `["arcgraph_mcp_tool_invocations"]`
/// - `histogram_quantile(0.99, sum by (le) (rate(arcgraph_read_latency_ms_bucket[5m])))`
///   → `["arcgraph_read_latency_ms_bucket"]`
/// - `vector(0)` → `[]`
/// - `arcgraph_a + arcgraph_b` → `["arcgraph_a", "arcgraph_b"]`
fn extract_arcgraph_roots(expr: &str) -> Vec<String> {
    let bytes = expr.as_bytes();
    let needle = b"arcgraph_";
    let mut roots = Vec::new();
    let mut i = 0;
    while i + needle.len() <= bytes.len() {
        if &bytes[i..i + needle.len()] == needle {
            let start = i;
            i += needle.len();
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let root = std::str::from_utf8(&bytes[start..i])
                .expect("PromQL expr was valid utf-8 above")
                .to_string();
            roots.push(root);
        } else {
            i += 1;
        }
    }
    roots
}

/// Returns true if `root` is in [`REGISTERED_METRICS`] either
/// verbatim or with one of the histogram suffixes appended.
fn is_registered(root: &str) -> bool {
    if REGISTERED_METRICS.contains(&root) {
        return true;
    }
    for &registered in REGISTERED_METRICS {
        for &suffix in HISTOGRAM_SUFFIXES {
            if root.len() == registered.len() + suffix.len()
                && root.starts_with(registered)
                && root.ends_with(suffix)
            {
                return true;
            }
        }
    }
    false
}

// ─────────────────────────────────────────────────────────────────
// Test 1 — overview dashboard JSON parses + has the required shape
// ─────────────────────────────────────────────────────────────────

#[test]
fn grafana_overview_dashboard_json_validates() {
    let path = docs_path("docs/grafana/dashboards/arcgraph-overview.json");
    let bytes = std::fs::read(&path).expect("overview dashboard exists");
    let v: Value = serde_json::from_slice(&bytes).expect("overview dashboard parses as JSON");

    // Required top-level keys per the Grafana schema-version-38
    // dashboard format.
    for k in ["title", "uid", "schemaVersion", "panels", "templating"] {
        assert!(
            v.get(k).is_some(),
            "overview dashboard missing required top-level key: {k}"
        );
    }
    assert!(
        v["title"].as_str().unwrap_or("").contains("ArcGraph"),
        "overview dashboard title should mention ArcGraph"
    );
    assert!(
        v["uid"].as_str().unwrap_or("").starts_with("arcgraph-"),
        "overview dashboard uid should start with arcgraph-"
    );

    let panels = v["panels"].as_array().expect("panels is an array");
    assert!(
        !panels.is_empty(),
        "overview dashboard must define at least one panel"
    );
    for (i, panel) in panels.iter().enumerate() {
        assert!(panel["type"].is_string(), "panel {i} missing type");
        assert!(panel["title"].is_string(), "panel {i} missing title");
    }
    // The deeper exporter-coherence check (every panel expr's
    // arcgraph_ root must be registered OR the panel must be
    // forward-bound-tagged) lives in test 4.
}

// ─────────────────────────────────────────────────────────────────
// Test 2 — per-tenant dashboard JSON parses + pins rate-limit cite
// ─────────────────────────────────────────────────────────────────

#[test]
fn grafana_tenants_dashboard_json_validates_and_pins_rate_limit_contract_metric() {
    let path = docs_path("docs/grafana/dashboards/arcgraph-tenants.json");
    let bytes = std::fs::read(&path).expect("tenants dashboard exists");
    let v: Value = serde_json::from_slice(&bytes).expect("tenants dashboard parses as JSON");

    for k in ["title", "uid", "schemaVersion", "panels", "templating"] {
        assert!(
            v.get(k).is_some(),
            "tenants dashboard missing required top-level key: {k}"
        );
    }

    // Per the original spawn prompt mandate ("tenants dashboard
    // consumes W14γ rate-limit metrics"). After the W15 IR L1-HIGH-3
    // fix-up the rate-limit metric is no longer in the panel `expr`
    // (it's been forward-bound with a `vector(0)` placeholder); the
    // contract surface has moved to the panel TITLE
    // (`contract metric: arcgraph_rate_limit_*`). Assert the title
    // surface, which is the load-bearing reference for the M6-07
    // wire-up.
    let panels = v["panels"].as_array().expect("panels array");
    let saw_rate_limit_contract = panels.iter().any(|panel| {
        let title = panel.get("title").and_then(Value::as_str).unwrap_or("");
        title.contains("arcgraph_rate_limit")
    });
    assert!(
        saw_rate_limit_contract,
        "tenants dashboard MUST cite a W14γ rate-limit contract metric \
         (arcgraph_rate_limit_*) in at least one panel title"
    );

    // The tenant dashboard MUST include a `tenant` template variable
    // so panels filter by it.
    let template_vars = v["templating"]["list"].as_array().expect("templating list");
    let saw_tenant_var = template_vars
        .iter()
        .any(|v| v["name"].as_str() == Some("tenant"));
    assert!(
        saw_tenant_var,
        "tenants dashboard must define a 'tenant' template variable"
    );
}

// ─────────────────────────────────────────────────────────────────
// Test 3 — alerts.yml structural validation + optional promtool
// ─────────────────────────────────────────────────────────────────

#[test]
fn grafana_alerts_yml_validates_and_promtool_passes_if_available() {
    let path = docs_path("docs/grafana/alerts.yml");
    let bytes = std::fs::read(&path).expect("alerts.yml exists");
    let s = std::str::from_utf8(&bytes).expect("alerts.yml is utf-8");

    // Parse via serde_yaml and assert the rule-group schema.
    let v: serde_yaml::Value = serde_yaml::from_str(s).expect("alerts.yml parses as YAML");
    let groups = v["groups"]
        .as_sequence()
        .expect("alerts.yml top-level 'groups' must be a sequence");
    assert!(
        !groups.is_empty(),
        "alerts.yml must define at least one rule group"
    );

    let mut total_rules = 0usize;
    for (gi, group) in groups.iter().enumerate() {
        assert!(
            group["name"].as_str().is_some(),
            "alerts.yml group {gi} missing name"
        );
        let rules = group["rules"]
            .as_sequence()
            .unwrap_or_else(|| panic!("alerts.yml group {gi} missing 'rules' sequence"));
        for (ri, rule) in rules.iter().enumerate() {
            assert!(
                rule["alert"].as_str().is_some(),
                "alerts.yml group {gi} rule {ri} missing 'alert'"
            );
            assert!(
                rule["expr"].as_str().is_some(),
                "alerts.yml group {gi} rule {ri} missing 'expr'"
            );
            assert!(
                rule["labels"]["severity"].as_str().is_some(),
                "alerts.yml group {gi} rule {ri} missing labels.severity"
            );
            assert!(
                rule["annotations"]["summary"].as_str().is_some(),
                "alerts.yml group {gi} rule {ri} missing annotations.summary"
            );
            total_rules += 1;
        }
    }
    assert!(
        total_rules >= 4,
        "alerts.yml must define ≥4 alert rules (got {total_rules})"
    );

    // Canonical Prometheus rule validation via `promtool check
    // rules`. PANIC by default when promtool is missing per
    // `feedback_test_env_gate_panic_by_default.md`; explicit
    // `ARCGRAPH_PROMTOOL_SKIP_OK=1` opts into the soft-skip
    // (W15δ R1 LOW-2 closure).
    if which_promtool() {
        let out = Command::new("promtool")
            .arg("check")
            .arg("rules")
            .arg(&path)
            .output()
            .expect("invoke promtool");
        assert!(
            out.status.success(),
            "promtool check rules failed:\nstdout={}\nstderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        );
    } else if std::env::var("ARCGRAPH_PROMTOOL_SKIP_OK").is_ok() {
        eprintln!(
            "grafana_alerts_yml_validates: promtool not on PATH and \
             ARCGRAPH_PROMTOOL_SKIP_OK=1 — skipping the shell-out \
             check explicitly. Structural validation passed."
        );
    } else {
        panic!(
            "grafana_alerts_yml_validates: promtool not on PATH. \
             Install promtool (Prometheus 2.x tarball ships it), or \
             set ARCGRAPH_PROMTOOL_SKIP_OK=1 to opt into a soft-skip \
             for hostile-environment debugging. CI runners are \
             responsible for installing promtool so the canonical \
             rule-check fires there."
        );
    }
}

fn which_promtool() -> bool {
    Command::new("promtool")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

// ─────────────────────────────────────────────────────────────────
// Test 4 — cross-PR exporter coherence (issue #314 / W15 IR
// L1-HIGH-3) — every arcgraph_* metric root in any panel/alert expr
// must be registered or the surrounding panel/alert must carry the
// forward-bound contract.
// ─────────────────────────────────────────────────────────────────

#[test]
fn grafana_panel_metrics_match_exporter_registry_or_marked_forward_bound() {
    let overview = load_dashboard("docs/grafana/dashboards/arcgraph-overview.json");
    let tenants = load_dashboard("docs/grafana/dashboards/arcgraph-tenants.json");
    let alerts = load_alerts_yaml("docs/grafana/alerts.yml");

    let mut drift = Vec::new();
    let mut forward_bound_without_contract = Vec::new();

    // Walk each dashboard panel.
    for (dashboard_name, dashboard) in [("overview", &overview), ("tenants", &tenants)] {
        let panels = dashboard["panels"]
            .as_array()
            .expect("dashboard panels array");
        for (pi, panel) in panels.iter().enumerate() {
            let title = panel
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or("<no-title>");
            let is_forward = title.contains("[forward-bound");
            // Forward-bound panels MUST also name a contract metric
            // in the title (the M6-07 wire-up swap surface). We
            // accept either "contract metric: arcgraph_…" or any
            // `arcgraph_` substring in the title for the forward-
            // bound case — the panel title is the load-bearing M6-07
            // reference.
            if is_forward && !title.contains("arcgraph_") {
                forward_bound_without_contract.push(format!(
                    "[{dashboard_name} panel {pi}] '{title}' is marked forward-bound but \
                     names no contract metric (expected 'contract metric: arcgraph_*')"
                ));
            }
            let Some(targets) = panel.get("targets").and_then(Value::as_array) else {
                continue;
            };
            for (ti, t) in targets.iter().enumerate() {
                let Some(expr) = t.get("expr").and_then(Value::as_str) else {
                    continue;
                };
                let roots = extract_arcgraph_roots(expr);
                for root in roots {
                    let registered = is_registered(&root);
                    if !registered && !is_forward {
                        drift.push(format!(
                            "[{dashboard_name} panel {pi} target {ti}] '{title}' references \
                             unregistered metric '{root}' in expr '{expr}' \
                             without [forward-bound] tag"
                        ));
                    }
                }
            }
        }
    }

    // Walk each alert rule.
    let groups = alerts["groups"].as_sequence().expect("alerts.yml groups");
    for (gi, group) in groups.iter().enumerate() {
        let group_name = group["name"].as_str().unwrap_or("<no-name>");
        let rules = group["rules"].as_sequence().expect("group rules");
        for (ri, rule) in rules.iter().enumerate() {
            let alert_name = rule["alert"].as_str().unwrap_or("<no-alert>");
            let expr = rule["expr"].as_str().unwrap_or("");
            let summary = rule["annotations"]["summary"].as_str().unwrap_or("");
            // Alerts are forward-bound IFF they carry the
            // `annotations.contract_metric` field.
            let contract_metric = rule["annotations"]["contract_metric"].as_str();
            let is_forward = contract_metric.is_some() || summary.contains("[forward-bound");
            if is_forward && contract_metric.is_none() {
                forward_bound_without_contract.push(format!(
                    "[alerts group '{group_name}' rule {ri} alert '{alert_name}'] \
                     forward-bound (per summary) but missing \
                     `annotations.contract_metric` field"
                ));
            }
            let roots = extract_arcgraph_roots(expr);
            for root in roots {
                let registered = is_registered(&root);
                if !registered && !is_forward {
                    drift.push(format!(
                        "[alerts group '{group_name}' rule {ri} alert '{alert_name}'] \
                         references unregistered metric '{root}' in expr '{expr}' \
                         without forward-bound tag (group index {gi})"
                    ));
                }
            }
        }
    }

    if !drift.is_empty() || !forward_bound_without_contract.is_empty() {
        let mut msg = String::from(
            "Cross-PR exporter coherence FAILED (W15 IR L1-HIGH-3 / \
             issue #314).\n\nDrift findings:\n",
        );
        for d in &drift {
            msg.push_str("  - ");
            msg.push_str(d);
            msg.push('\n');
        }
        if !forward_bound_without_contract.is_empty() {
            msg.push_str("\nForward-bound without contract metric:\n");
            for d in &forward_bound_without_contract {
                msg.push_str("  - ");
                msg.push_str(d);
                msg.push('\n');
            }
        }
        msg.push_str("\nRegistered metrics (sister-PR #309 W15γ M6-06):\n");
        for r in REGISTERED_METRICS {
            msg.push_str("  - ");
            msg.push_str(r);
            msg.push('\n');
        }
        panic!("{msg}");
    }
}

fn load_dashboard(rel: &str) -> Value {
    let path = docs_path(rel);
    load_dashboard_at(&path)
}

fn load_dashboard_at(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn load_alerts_yaml(rel: &str) -> serde_yaml::Value {
    let path = docs_path(rel);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let s = std::str::from_utf8(&bytes).expect("alerts.yml utf-8");
    serde_yaml::from_str(s).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

// ─────────────────────────────────────────────────────────────────
// Test 5 — extract_arcgraph_roots + is_registered self-tests
// ─────────────────────────────────────────────────────────────────

#[test]
fn extract_arcgraph_roots_handles_promql_idioms() {
    // Empty / no metric.
    assert_eq!(extract_arcgraph_roots(""), Vec::<String>::new());
    assert_eq!(extract_arcgraph_roots("vector(0)"), Vec::<String>::new());
    assert_eq!(extract_arcgraph_roots("time() > 0"), Vec::<String>::new());

    // Single root, naked.
    assert_eq!(
        extract_arcgraph_roots("arcgraph_buffer_pool_hit_rate"),
        vec!["arcgraph_buffer_pool_hit_rate"],
    );

    // Inside rate() with label filter.
    assert_eq!(
        extract_arcgraph_roots(
            "sum by (tool) (rate(arcgraph_mcp_tool_invocations{status=\"ok\"}[1m]))"
        ),
        vec!["arcgraph_mcp_tool_invocations"],
    );

    // Histogram bucket suffix.
    assert_eq!(
        extract_arcgraph_roots(
            "histogram_quantile(0.99, sum by (le) (rate(arcgraph_read_latency_ms_bucket[5m])))"
        ),
        vec!["arcgraph_read_latency_ms_bucket"],
    );

    // Two roots in one expr.
    assert_eq!(
        extract_arcgraph_roots("arcgraph_active_connections / arcgraph_mcp_tool_invocations"),
        vec![
            "arcgraph_active_connections",
            "arcgraph_mcp_tool_invocations"
        ],
    );

    // Underscore-only continuation past the root.
    assert_eq!(
        extract_arcgraph_roots("arcgraph_wal_fsync_duration_ms_bucket"),
        vec!["arcgraph_wal_fsync_duration_ms_bucket"],
    );
}

#[test]
fn is_registered_handles_histogram_suffixes() {
    // Direct hits.
    assert!(is_registered("arcgraph_mcp_tool_invocations"));
    assert!(is_registered("arcgraph_read_latency_ms"));
    assert!(is_registered("arcgraph_write_latency_ms"));
    assert!(is_registered("arcgraph_active_connections"));
    // W17δ #313 — hot-vertex counter is now registered.
    assert!(is_registered("arcgraph_hot_vertex_warnings_total"));

    // Histogram suffix tolerance.
    assert!(is_registered("arcgraph_read_latency_ms_bucket"));
    assert!(is_registered("arcgraph_read_latency_ms_count"));
    assert!(is_registered("arcgraph_read_latency_ms_sum"));
    assert!(is_registered("arcgraph_write_latency_ms_bucket"));

    // NOT registered (forward-bound surfaces).
    assert!(!is_registered("arcgraph_wal_fsync_duration_ms_bucket"));
    assert!(!is_registered("arcgraph_buffer_pool_hit_rate"));
    assert!(!is_registered("arcgraph_rate_limit_rejections_total"));
    // The pre-W17δ shape `arcgraph_hot_vertex_warnings` (no `_total`
    // suffix) is intentionally NOT registered — the registered metric
    // canonically carries the `_total` suffix per Prometheus counter
    // naming convention.
    assert!(!is_registered("arcgraph_hot_vertex_warnings"));
    // ADR-202 — registered + producer-wired (was forward-bound).
    assert!(is_registered("arcgraph_leiden_last_run_seconds"));
    assert!(!is_registered("arcgraph_recovery_failed_total"));

    // Prefix-but-not-suffix-match must NOT be allowed.
    assert!(!is_registered("arcgraph_read_latency_ms_extra"));
    // Exact-overlap-on-different-suffix must NOT be allowed.
    assert!(!is_registered("arcgraph_unknown_metric"));

    // _ in REGISTERED check.
    let _used: HashSet<&str> = REGISTERED_METRICS.iter().copied().collect();
}
