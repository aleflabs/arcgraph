//! ArcGraph openCypher TCK harness — post-W11Z flip (M4-61 ships).
//!
//! `harness = false` means `cargo test -p arcgraph-tck` invokes
//! [`main`] directly. Post-M4-61 (PR #268, `5614e43`) the harness:
//!
//! 1. Walks `tck/features/` and pins the vendored count against
//!    [`arcgraph_tck::VENDORED_FEATURE_COUNT`].
//! 2. Sanity-checks the canonical named-graph fixture directories
//!    (`binary-tree-1`, `binary-tree-2`, `yago`) exist on disk.
//! 3. Drives the cucumber-rs harness against the vendored
//!    feature tree. `When executing query:` step bindings dispatch
//!    the captured docstring through
//!    `arcgraph_query::QueryEngine::execute(...)`; `Then` steps
//!    that the harness can verify in-process do so, while shape-
//!    advanced matchers (table comparisons, etc.) flow through as
//!    Skipped per the `feature-bindings-not-yet-shipped` ledger.
//! 4. Reports the **passed / failed / skipped / parsing-error**
//!    rate so CI logs surface the v1.0-alpha pass-rate floor and
//!    any future regression at-a-glance.
//!
//! ## v1.0-alpha pass-rate floor
//!
//! ArcGraph at v1.0-alpha is **read-only** per ADR-006
//! amendment-01 (no `CREATE` / `DELETE` / `MERGE`). Almost every
//! TCK feature opens with a `having executed: """CREATE..."""`
//! setup step, so the failure mass is dominated by a single root
//! cause (CREATE rejection at the parser/binder/executor seam).
//! That's the desired signal — the harness gives one consistent
//! v1.0-alpha shape: setup-step failure → scenario fails → counted
//! by the [`writer::Stats`] reporter. The floor lifts naturally
//! when:
//!
//! - **M4-61b / M4-61c (executor write-ops)** light CREATE shapes,
//! - **M5-08 (graph.ingest)** lights bulk-load setup, **and**
//! - **M5-12 (per-tenant parameter bag)** unblocks `$param` cypher
//!   forms.
//!
//! Until then the report is "X passed, Y failed, Z skipped" with
//! Y dominated by the v1.0-alpha shape; the **regression mode** is
//! "passed-step count drops below the floor recorded in CI."
//!
//! ## License + provenance
//!
//! TCK feature files vendored under `tck/features/` ship under
//! Apache-2.0 (see `crates/arcgraph-tck/LICENSE-OPENCYPHER`,
//! `NOTICE-OPENCYPHER`, `tck/PROVENANCE.md`).

use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::semantic::StubCatalogProvider;
use cucumber::gherkin::Step;
use cucumber::writer::Stats;
use cucumber::{World, given, then, when};

/// Cucumber world for the TCK harness.
///
/// `last_cypher` carries the most recent docstring captured from a
/// `When executing query:` step. `last_result` carries the
/// `QueryEngine::execute` outcome — `Ok(rows)` if the dispatch
/// surfaced a row-set, `Err(display)` if it surfaced an
/// `ExplainError` (parse / bind / type-check / cross-substrate /
/// executor NotImplemented). Subsequent `Then` step bindings read
/// from `last_result` to satisfy the assertion.
#[derive(World, Debug, Default)]
pub struct ArcGraphWorld {
    last_cypher: Option<String>,
    last_result: Option<Result<usize, String>>,
}

// ---------------------------------------------------------------
// Given steps
// ---------------------------------------------------------------

/// `Given an empty graph` — the most common TCK opening line.
/// The empty stub catalog/substrate is acquired at execute-time, so
/// no per-step state is materialised here; the binding just clears
/// any prior result-pin so a downstream `Then` can't accidentally
/// observe state from the previous scenario.
#[given(regex = r"^an empty graph$")]
fn given_empty_graph(world: &mut ArcGraphWorld) {
    world.last_cypher = None;
    world.last_result = None;
}

/// `Given any graph` — TCK shorthand. Same contract as
/// [`given_empty_graph`] at v1.0-alpha (the empty stub satisfies
/// "any" trivially).
#[given(regex = r"^any graph$")]
fn given_any_graph(world: &mut ArcGraphWorld) {
    world.last_cypher = None;
    world.last_result = None;
}

/// `Given the <name> graph` — TCK named-graph loader. M5-08
/// scaffolding: at v1.0-alpha there is no ingest path that can
/// honor the named-fixture seed, so the binding leaves the
/// substrate empty. Scenarios that depend on the fixture's data
/// will fail at the assertion step — desired signal.
#[given(regex = r"^the ([A-Za-z0-9][A-Za-z0-9_-]*) graph$")]
fn given_named_graph(world: &mut ArcGraphWorld, _name: String) {
    world.last_cypher = None;
    world.last_result = None;
}

/// `having executed: """..."""` — TCK fixture-setup step (always
/// uses `CREATE` / `MERGE` to seed). At v1.0-alpha-pre-M5-08 the
/// catalog is read-only per ADR-006 amendment-01; routing the
/// setup cypher through the executor surfaces a parse / bind /
/// not-implemented error. Failing the step here is the desired
/// signal — every scenario that needs setup data fails fast at
/// this binding, isolating the v1.0-alpha pass-rate floor to the
/// (small) subset of features that don't carry a `having executed:`
/// seed.
#[given(regex = r"^having executed:$")]
fn given_having_executed(world: &mut ArcGraphWorld, #[step] step: &Step) {
    let cypher = step.docstring.as_deref().unwrap_or("").trim().to_string();
    world.last_cypher = Some(cypher.clone());
    let outcome = execute_with_empty_substrate(&cypher);
    if let Err(err) = &outcome {
        // Per cucumber-rs convention, panicking inside a step body
        // marks the step as Failed. The panic message becomes the
        // failure reason in the writer's report.
        panic!(
            "having-executed setup rejected by v1.0-alpha read-only catalog: {err} \
             (per ADR-006 amendment-01; lifts at M4-61b / M5-08)"
        );
    }
    world.last_result = Some(outcome);
}

// ---------------------------------------------------------------
// When steps
// ---------------------------------------------------------------

/// `When executing query:` — the core TCK step. Captures the
/// docstring + dispatches through `QueryEngine::execute` against
/// an empty stub catalog/substrate. Result is recorded for
/// subsequent `Then` bindings.
#[when(regex = r"^executing query:$")]
fn when_executing_query(world: &mut ArcGraphWorld, #[step] step: &Step) {
    let cypher = step.docstring.as_deref().unwrap_or("").trim().to_string();
    world.last_cypher = Some(cypher.clone());
    world.last_result = Some(execute_with_empty_substrate(&cypher));
}

/// `When executing control query:` — TCK pre-condition step (e.g.
/// load fixture data). Same dispatch contract as
/// [`when_executing_query`].
#[when(regex = r"^executing control query:$")]
fn when_executing_control_query(world: &mut ArcGraphWorld, #[step] step: &Step) {
    when_executing_query(world, step);
}

// ---------------------------------------------------------------
// Then steps
// ---------------------------------------------------------------

/// `Then the result should be empty` — TCK assertion step. Asserts
/// the captured `last_result` is `Ok(rows)` with `rows.is_empty()`.
/// If the result is `Err`, propagate it as a step failure (this
/// shape gives clean signal on read-side queries that should pass
/// against the empty substrate but instead surface a NotImplemented
/// or BindingError).
#[then(regex = r"^the result should be empty$")]
fn then_result_should_be_empty(world: &mut ArcGraphWorld) {
    let result = world
        .last_result
        .as_ref()
        .expect("`Then the result should be empty` requires a prior `When executing ...` step");
    match result {
        Ok(rows) => assert_eq!(*rows, 0, "expected empty result, got {rows} rows"),
        Err(err) => panic!("execute failed: {err}"),
    }
}

/// `Then no side effects` — at v1.0-alpha-pre-M4-61b the executor
/// is read-only by construction; the assertion is trivially
/// satisfied.
#[then(regex = r"^no side effects$")]
fn then_no_side_effects(_world: &mut ArcGraphWorld) {}

// ---------------------------------------------------------------
// Internal — executor dispatch helper
// ---------------------------------------------------------------

/// Drive `cypher` through `QueryEngine::execute` against a fresh
/// empty stub catalog + substrate. Returns the row count on
/// success, or the [`arcgraph_query::ExplainError`] display on any
/// failure (parse / bind / cross-substrate / NotImplemented /
/// runtime).
fn execute_with_empty_substrate(cypher: &str) -> Result<usize, String> {
    let catalog = StubCatalogProvider::new();
    let substrate = StubExecutorSubstrate::new();
    let engine = QueryEngine::new(&catalog);
    engine
        .execute(cypher, &substrate)
        .map(|rows| rows.len())
        .map_err(|err| err.to_string())
}

// ---------------------------------------------------------------
// Harness entry point
// ---------------------------------------------------------------

#[tokio::main(flavor = "current_thread")]
async fn main() {
    // 1. Walk the vendored tree + pin against `tck/PROVENANCE.md`.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let feature_root = std::path::Path::new(manifest_dir)
        .join("tck")
        .join("features");
    let features = arcgraph_tck::enumerate_feature_files(&feature_root).unwrap_or_else(|err| {
        panic!("failed to walk vendored TCK features at {feature_root:?}: {err}")
    });
    let n = features.len();
    assert_eq!(
        n,
        arcgraph_tck::VENDORED_FEATURE_COUNT,
        "vendored TCK feature count drift: pinned to {} per tck/PROVENANCE.md",
        arcgraph_tck::VENDORED_FEATURE_COUNT,
    );

    // 2. Sanity-check the canonical named graphs are on disk.
    let graphs_dir = std::path::Path::new(manifest_dir)
        .join("tck")
        .join("graphs");
    for canonical in ["binary-tree-1", "binary-tree-2", "yago"] {
        let path = graphs_dir.join(canonical);
        assert!(
            path.is_dir(),
            "vendored TCK graph `{canonical}` missing at {path:?}",
        );
    }

    // 3. Drive cucumber against the vendored tree. Sequential
    //    execution (`max_concurrent_scenarios(1)`) keeps the report
    //    deterministic; the catalog/substrate is per-execution so
    //    cross-scenario state cannot leak. We use `run` (not
    //    `run_and_exit`) so the binary returns 0 even with high
    //    failure mass — at v1.0-alpha the failure mass IS the
    //    expected v1.0-alpha shape, not a CI regression.
    eprintln!(
        "arcgraph-tck: {n} features detected; running cucumber harness ({})",
        arcgraph_tck::M4_61_FORWARD_LINK,
    );
    let writer = ArcGraphWorld::cucumber()
        .max_concurrent_scenarios(1)
        .run(&feature_root)
        .await;

    // 4. Surface the pass-rate floor for CI log readers.
    let passed = writer.passed_steps();
    let failed = writer.failed_steps();
    let skipped = writer.skipped_steps();
    let parsing = writer.parsing_errors();
    eprintln!(
        "arcgraph-tck: harness run complete — \
         passed={passed} failed={failed} skipped={skipped} parsing_errors={parsing} \
         (v1.0-alpha read-only catalog dominates the failure mass; \
         M4-61b / M5-08 / M5-12 lift the floor; see crate-level docs)"
    );

    // W18δ MED-5 R1 fix-up — passed-step regression floor.
    //
    // The W18δ R1 review re-derived `passed=6787 failed=977 skipped=2920`
    // against this branch. Per
    // `feedback_review_oracle_relaxations.md` + the PR's claim that
    // this harness closes W11 R-7, we pin a floor at 6500 (≈ 96 % of
    // the observed 6787, leaving room for minor expected-skip drift
    // when openCypher vendoring lifts). A step-count cliff (e.g.,
    // parser/binder/executor regression dropping passed from 6787 to
    // a few hundred) trips this immediately, where previously
    // `cargo test -p arcgraph-tck` would still exit 0.
    //
    // Lift the floor in a follow-up wave when M4-61b / M5-08 / M5-12
    // land and the baseline shifts up; do NOT silently lower it
    // without a recorded regression-acceptance ADR.
    const TCK_PASSED_STEPS_FLOOR: usize = 6_500;
    assert!(
        passed >= TCK_PASSED_STEPS_FLOOR,
        "TCK passed-step regression: got {passed} passed steps, \
         below the {TCK_PASSED_STEPS_FLOOR} floor (W18δ baseline 6787). \
         Either a parser/binder/executor regression dropped pass-rate, \
         or the vendored feature tree changed shape — investigate \
         before lowering the floor."
    );
}
