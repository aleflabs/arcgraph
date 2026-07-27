//! W15β M6-03 — Docker + release-workflow shape tests.
//!
//! Pins four orthogonal contracts for the M6-03 deliverable:
//!
//! 1. **`m6_03_hadolint_dockerfile`** — the repo-root `Dockerfile`
//!    passes `hadolint` linting (no `error` findings; warnings tolerated
//!    so the test isn't brittle against hadolint version bumps).
//! 2. **`m6_03_docker_build`** — `docker build .` succeeds on the
//!    repo-root `Dockerfile`.
//! 3. **`m6_03_docker_run_smoke`** — the built image runs the
//!    `arcgraph` umbrella binary (W15α M6-02 / PR #306) and exits 0
//!    with a version string on `arcgraph --version`. The Dockerfile
//!    ENTRYPOINT is `["arcgraph", "serve", "--http", "0.0.0.0:8080"]`
//!    which currently bails on M6-08+ HTTPS wiring, so the smoke test
//!    overrides ENTRYPOINT via `docker run --entrypoint
//!    /usr/local/bin/arcgraph <tag> --version` to assert the binary
//!    is present and operationally sound independent of the M6-08+
//!    cert-resolver landing.
//! 4. **`m6_03_actionlint_release_workflow`** — the release workflow
//!    YAML passes `actionlint`.
//!
//! # Env-gate discipline
//!
//! Per `feedback_test_env_gate_panic_by_default.md` (W12δ HIGH-1):
//! environment-gated tests PANIC by default on a missing gate flag,
//! with a single explicit soft-skip env var. Soft-skipping silently
//! after `--ignored` bypass is the bug class that lets green-painted
//! tests pass without ever running.
//!
//! Each test:
//! - Is `#[ignore]`'d to keep it off the default `cargo test` gauntlet
//!   (these need `hadolint` / `actionlint` / `docker` on the host,
//!   which CI runners install conditionally).
//! - When run via `--ignored`, PANICS unless the test-specific
//!   `ARCGRAPH_*_TEST=1` flag is set OR the `ARCGRAPH_*_SKIP_OK=1`
//!   soft-skip flag is set.
//!
//! # Why this file lives in `arcgraph-cli/tests/`
//!
//! The Dockerfile + workflow are repo-root artifacts, but Cargo tests
//! must live inside a crate. `arcgraph-cli` owns the `arcgraph` umbrella
//! binary that the Dockerfile packages (W15α M6-02 / PR #306), so this
//! is the natural home — a Dockerfile regression that breaks the binary
//! build surfaces here alongside the existing
//! `arcgraph_cli_subprocess.rs` integration test.

use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate the workspace root by walking up from this test's
/// `CARGO_MANIFEST_DIR` until we find the workspace `Cargo.toml`
/// (the one declaring `[workspace]`). This keeps the test independent
/// of where `cargo test` is invoked from.
fn workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../crates/arcgraph-cli
    manifest_dir
        .parent() // .../crates
        .and_then(Path::parent) // .../<workspace-root>
        .expect("crates/arcgraph-cli should have a workspace parent")
        .to_path_buf()
}

/// Build a parallel-run-safe container name (N-4 from PR #308 round-1
/// review). Two concurrent test runs on the same host would collide on
/// a fixed `--name`; pid + ns-precision time defuses that. `--rm`
/// still cleans up the container on exit.
fn unique_container_name(prefix: &str) -> String {
    let pid = std::process::id();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{prefix}-{pid}-{nanos}")
}

/// Set `DOCKER_BUILDKIT=1` on `docker build` invocations (N-1 from
/// PR #308 round-1 review). The Dockerfile's `--mount=type=cache`
/// directives are BuildKit-only; modern engines (≥23.0) default to
/// BuildKit, but 18.09–22.x require this env var explicitly. Setting
/// it unconditionally is harmless on ≥23.0 (already-default) and
/// rejects gracefully on <18.09 (no `--mount` support means the test
/// surfaces the engine-too-old error rather than silently slow-pathing).
fn buildkit_env(mut cmd: Command) -> Command {
    cmd.env("DOCKER_BUILDKIT", "1");
    cmd
}

/// Convenience: enforce env-gate per
/// `feedback_test_env_gate_panic_by_default.md`. Returns `true` if the
/// test body should run; `false` if it should soft-skip (after
/// printing); panics if neither flag is set.
fn env_gate(test_name: &str, run_flag: &str, skip_flag: &str) -> bool {
    let run = std::env::var(run_flag).is_ok();
    let skip_ok = std::env::var(skip_flag).is_ok();
    if run {
        return true;
    }
    if skip_ok {
        eprintln!(
            "{test_name}: SKIPPING (opt-in via {skip_flag}=1) — set \
             {run_flag}=1 to run instead"
        );
        return false;
    }
    panic!(
        "{test_name}: required env flag {run_flag}=1 not set. This test \
         is `#[ignore]`'d to keep it off the default gauntlet; when \
         invoked via `--ignored` the gate flag must be set explicitly. \
         Set {run_flag}=1 to run, or {skip_flag}=1 to opt into a \
         soft-skip (hostile envs only, e.g. CI without the required \
         tool installed). Soft-skipping silently after `--ignored` \
         bypass is the W12δ HIGH-1 bug class \
         (`feedback_test_env_gate_panic_by_default.md`).",
    );
}

/// Convenience: assert the named tool is on PATH. Returns the success
/// `Output` so the caller can inspect stdout/stderr. Failure surfaces
/// as a panic with the underlying I/O error (typically "No such file
/// or directory" when the tool isn't installed).
fn ensure_tool_on_path(tool: &str) {
    let probe = Command::new(tool).arg("--version").output();
    match probe {
        Ok(out) => {
            if !out.status.success() {
                panic!(
                    "{tool} --version exited with status {:?}; stderr: {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr),
                );
            }
        }
        Err(e) => {
            panic!("could not exec `{tool}` (is it installed and on PATH?): {e}");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Test 1: hadolint
// ─────────────────────────────────────────────────────────────────────

const HADOLINT_RUN: &str = "ARCGRAPH_HADOLINT_TEST";
const HADOLINT_SKIP: &str = "ARCGRAPH_HADOLINT_SKIP_OK";

#[test]
#[ignore = "W15β M6-03 hadolint test; gated by ARCGRAPH_HADOLINT_TEST=1 \
            (panics if neither ARCGRAPH_HADOLINT_TEST=1 nor \
            ARCGRAPH_HADOLINT_SKIP_OK=1 is set; see \
            feedback_test_env_gate_panic_by_default.md)"]
fn m6_03_hadolint_dockerfile() {
    let test_name = "m6_03_hadolint_dockerfile";
    if !env_gate(test_name, HADOLINT_RUN, HADOLINT_SKIP) {
        return;
    }
    ensure_tool_on_path("hadolint");

    let dockerfile = workspace_root().join("Dockerfile");
    assert!(
        dockerfile.exists(),
        "Dockerfile missing at {}",
        dockerfile.display()
    );

    // `hadolint --no-fail` returns 0 even with warnings → we use
    // `--failure-threshold error` so the test fails only on `error`
    // findings (style warnings would make the test brittle against
    // hadolint releases).
    let output = Command::new("hadolint")
        .args(["--failure-threshold", "error"])
        .arg(&dockerfile)
        .output()
        .expect("failed to exec hadolint");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "hadolint reported error-level findings.\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 2: docker build
// ─────────────────────────────────────────────────────────────────────

const DOCKER_RUN: &str = "ARCGRAPH_DOCKER_TEST";
const DOCKER_SKIP: &str = "ARCGRAPH_DOCKER_SKIP_OK";

#[test]
#[ignore = "W15β M6-03 docker build test; gated by ARCGRAPH_DOCKER_TEST=1 \
            (panics if neither ARCGRAPH_DOCKER_TEST=1 nor \
            ARCGRAPH_DOCKER_SKIP_OK=1 is set; ~3-8 min wall first time)"]
fn m6_03_docker_build() {
    let test_name = "m6_03_docker_build";
    if !env_gate(test_name, DOCKER_RUN, DOCKER_SKIP) {
        return;
    }
    ensure_tool_on_path("docker");

    let root = workspace_root();
    // Tag the image with a test-specific suffix so a parallel test
    // run on the same host can't see a stale image.
    let tag = "arcgraph:m6-03-build-test";
    let mut cmd = Command::new("docker");
    cmd.arg("build")
        .args(["--tag", tag])
        .arg("--file")
        .arg(root.join("Dockerfile"))
        .arg(&root);
    let output = buildkit_env(cmd)
        .output()
        .expect("failed to exec docker build");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "docker build failed.\nstdout (tail):\n{}\nstderr (tail):\n{}",
        tail_lines(&stdout, 40),
        tail_lines(&stderr, 40),
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 3: docker run smoke (`arcgraph --version` exit-0)
// ─────────────────────────────────────────────────────────────────────
//
// The W15β prompt asks for `docker run arcgraph` to exit 0 on
// `--version`. Post-#306 (W15α M6-02) the umbrella `arcgraph` binary
// derives `--version` via clap (see
// `crates/arcgraph-cli/src/bin/arcgraph.rs` — `#[command(... version,
// ...)]`). The Dockerfile ENTRYPOINT is
// `["/usr/local/bin/arcgraph", "serve", "--http", "0.0.0.0:8080"]`,
// which currently bails out on the M6-08+ forward-bound HTTPS wiring,
// so this test overrides ENTRYPOINT via `--entrypoint
// /usr/local/bin/arcgraph` and passes `--version` as the only arg.
// The container exits 0 with a "arcgraph <version>" line on stdout.

#[test]
#[ignore = "W15β M6-03 docker run smoke test; gated by ARCGRAPH_DOCKER_TEST=1 \
            (panics if neither ARCGRAPH_DOCKER_TEST=1 nor \
            ARCGRAPH_DOCKER_SKIP_OK=1 is set; depends on m6_03_docker_build)"]
fn m6_03_docker_run_smoke() {
    let test_name = "m6_03_docker_run_smoke";
    if !env_gate(test_name, DOCKER_RUN, DOCKER_SKIP) {
        return;
    }
    ensure_tool_on_path("docker");

    // Sequential ordering: the smoke test depends on
    // `m6_03_docker_build` having already produced the image. We
    // re-run the build defensively (cheap on the build cache) so a
    // parallel test runner that schedules this test first still finds
    // an image to run.
    let root = workspace_root();
    let tag = "arcgraph:m6-03-build-test";
    let mut build_cmd = Command::new("docker");
    build_cmd
        .arg("build")
        .args(["--tag", tag])
        .arg("--file")
        .arg(root.join("Dockerfile"))
        .arg(&root);
    let build = buildkit_env(build_cmd)
        .output()
        .expect("failed to exec docker build (smoke prep)");
    assert!(
        build.status.success(),
        "docker build (smoke prep) failed; stderr (tail):\n{}",
        tail_lines(&String::from_utf8_lossy(&build.stderr), 30),
    );

    // N-6 (PR #308 round-1 review): pin the declared ENTRYPOINT path
    // so a build that accidentally packages the wrong binary surfaces
    // here as a path mismatch rather than as a downstream operational
    // regression. The `inspect_stdout.contains(...)` substring check
    // matches against the JSON-array rendering of the ENTRYPOINT
    // (i.e., `["/usr/local/bin/arcgraph","serve","--http","0.0.0.0:8080"]`)
    // and asserts the first element points at the umbrella binary.
    //
    // PR #337 R1 MED-1: the substring is **quote-bounded** —
    // `"\"/usr/local/bin/arcgraph\""` — so the JSON-array `"`
    // delimiters disambiguate. A regression to ENTRYPOINT
    // `["/usr/local/bin/arcgraph-mcp-stdio",…]` would render
    // `"/usr/local/bin/arcgraph-mcp-stdio"` inside the JSON array;
    // the trailing `"` after `arcgraph` would NOT appear in that
    // string, so the substring match would FAIL — which is the
    // discriminator N-6 originally promised.
    let expected_entrypoint = "\"/usr/local/bin/arcgraph\"";
    let inspect = Command::new("docker")
        .args(["inspect", "--format", "{{json .Config.Entrypoint}}", tag])
        .output()
        .expect("failed to exec docker inspect");
    assert!(
        inspect.status.success(),
        "docker inspect failed; stderr (tail):\n{}",
        tail_lines(&String::from_utf8_lossy(&inspect.stderr), 20),
    );
    let inspect_stdout = String::from_utf8_lossy(&inspect.stdout);
    assert!(
        inspect_stdout.contains(expected_entrypoint),
        "image ENTRYPOINT does not match expected binary path.\n\
         Expected to find quote-bounded: {expected_entrypoint}\n\
         docker inspect stdout: {inspect_stdout}",
    );

    // N-4 (PR #308 round-1 review): parallel-run-safe container name.
    // Two concurrent test runs would collide on `--name arcgraph-m6-03-smoke`;
    // pid + ns-precision time defuses that. `--rm` still cleans up.
    let container_name = unique_container_name("arcgraph-m6-03-smoke");

    // The Dockerfile ENTRYPOINT runs `arcgraph serve --http
    // 0.0.0.0:8080`, which currently bails on M6-08+ forward-bound
    // HTTPS wiring. Override ENTRYPOINT to the bare binary and pass
    // `--version` so the smoke test pins binary integrity
    // (clap-derived `--version` exits 0 with the crate version)
    // independently of the M6-08+ cert-resolver landing.
    let output = Command::new("docker")
        .args([
            "run",
            "--rm",
            "--name",
            &container_name,
            "--entrypoint",
            "/usr/local/bin/arcgraph",
            tag,
            "--version",
        ])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()
        .expect("failed to exec docker run --version");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "docker run --version exited non-zero: {:?}\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );
    // clap's derived `--version` prints `<bin-name> <version>` on
    // stdout, where `<bin-name>` is the `name = "arcgraph"` from
    // `#[command(name = "arcgraph", version, …)]` at the top of
    // `crates/arcgraph-cli/src/bin/arcgraph.rs`. So the umbrella
    // binary's `--version` stdout begins literally with `"arcgraph "`
    // (trailing space + semver).
    //
    // PR #337 R1 MED-2: use `starts_with("arcgraph ")` (with the
    // trailing space) rather than `contains("arcgraph")`. A sibling
    // `arcgraph-mcp-stdio --version` (clap-derived) would print
    // `"arcgraph-mcp-stdio <ver>"` — that contains `"arcgraph"` but
    // does NOT start with `"arcgraph "` (no space after `arcgraph` —
    // the next char is `-`). The trailing-space anchor therefore
    // discriminates sibling binaries whose name starts with
    // `arcgraph` from the umbrella binary itself.
    assert!(
        stdout.starts_with("arcgraph "),
        "expected `--version` stdout to begin with 'arcgraph ' (umbrella binary's clap-derived prefix); got:\n{stdout}",
    );
}

// ─────────────────────────────────────────────────────────────────────
// Test 4: actionlint
// ─────────────────────────────────────────────────────────────────────

const ACTIONLINT_RUN: &str = "ARCGRAPH_ACTIONLINT_TEST";
const ACTIONLINT_SKIP: &str = "ARCGRAPH_ACTIONLINT_SKIP_OK";

#[test]
#[ignore = "W15β M6-03 actionlint test; gated by ARCGRAPH_ACTIONLINT_TEST=1 \
            (panics if neither ARCGRAPH_ACTIONLINT_TEST=1 nor \
            ARCGRAPH_ACTIONLINT_SKIP_OK=1 is set; see \
            feedback_test_env_gate_panic_by_default.md)"]
fn m6_03_actionlint_release_workflow() {
    let test_name = "m6_03_actionlint_release_workflow";
    if !env_gate(test_name, ACTIONLINT_RUN, ACTIONLINT_SKIP) {
        return;
    }
    ensure_tool_on_path("actionlint");

    let workflow = workspace_root()
        .join(".github")
        .join("workflows")
        .join("release.yml");
    assert!(
        workflow.exists(),
        "release.yml missing at {}",
        workflow.display()
    );

    let output = Command::new("actionlint")
        .arg(&workflow)
        .output()
        .expect("failed to exec actionlint");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "actionlint reported findings on release.yml.\nstdout:\n{stdout}\nstderr:\n{stderr}",
    );
}

// ─────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────

/// Return the last `n` lines of `s` for failure-mode diagnostics.
fn tail_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let start = lines.len().saturating_sub(n);
    lines[start..].join("\n")
}
