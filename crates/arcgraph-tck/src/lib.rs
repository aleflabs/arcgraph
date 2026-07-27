//! openCypher TCK harness for ArcGraph.
//!
//! This crate is **test-binary-shaped**: the runtime body lives in
//! `tests/tck.rs`, where the cucumber-rs harness is wired up. The
//! lib target itself is intentionally minimal — it exists so
//! reusable helpers (e.g. forward-link cite strings, feature-file
//! enumeration) can be shared between the `tests/tck.rs` binary
//! and any future fuzz / property targets without re-vendoring.
//!
//! See `crates/arcgraph-tck/README.md` and
//! `tck/PROVENANCE.md` for the upstream-pin contract.

#![forbid(unsafe_code)]
#![recursion_limit = "256"]

pub mod curated;
pub mod differ;
pub mod executor;
pub mod scorecard;

pub use curated::{ALL_CURATED_QUERIES, CURATED_PER_CATEGORY, CuratedCategory, CuratedQuery};
pub use differ::{RowSetDiff, assert_row_set_equal};
pub use executor::{ArcGraphExecutor, ExecutorError, Neo4jOracleExecutor, RowSet, TckExecutor};
pub use scorecard::{
    FamilyCounts, STATIC_SNAPSHOT, STATIC_SNAPSHOT_TOTAL_SCENARIOS, ScenarioRecord,
    ScorecardSummary, Verdict, build_summary, categorize_feature_body, categorize_feature_file,
    format_markdown,
};

/// Forward-link cite string carried by harness logs as a load-bearing
/// breadcrumb for the M4-61 wave-level integration pin.
///
/// Wave-11β origin: every step binding that needed
/// `arcgraph_query::QueryEngine::execute` carried this cite while the
/// executor was unshipped. Post-M4-61 (PR #268, `5614e43`) the
/// bindings dispatch live and this constant remains as a log
/// breadcrumb so a `git log -SM4_61_FORWARD_LINK` walks back to the
/// flip — and to keep the producer/consumer name in lockstep with
/// the actual M4-61 type (`Batch`, per
/// `arcgraph_query::executor::batch::Batch`).
pub const M4_61_FORWARD_LINK: &str =
    "M4-61 (Slice γ — ExecutionContext + Batch) — wave-level executor seam (post-W11Z flip)";

/// Number of vendored TCK feature files at the upstream pin
/// (see `tck/PROVENANCE.md`).
///
/// This constant is the load-bearing pin for the harness's
/// "N features detected, 0 ran" report. If a future TCK refresh
/// changes the count, this constant moves and the
/// `tck_features_detected` test in `tests/tck.rs` updates the
/// expected number.
pub const VENDORED_FEATURE_COUNT: usize = 220;

/// Default feature path the harness runs against when
/// `TCK_FEATURES` is unset.
///
/// Match1 is the canonical openCypher introductory feature; running
/// it pre-M4-61 verifies the harness can parse and dispatch
/// `cucumber::Skip` decisions through the gherkin → cucumber
/// pipeline.
pub const DEFAULT_FEATURE_PATH: &str = "tck/features/clauses/match/Match1.feature";

/// Walk `tck/features/` and return every `.feature` path
/// discovered.
///
/// Used by the `tck_features_detected` test in `tests/tck.rs` to
/// pin the vendored count + report the "N features detected"
/// number. Pure filesystem-walk; no parsing.
pub fn enumerate_feature_files(
    root: impl AsRef<std::path::Path>,
) -> std::io::Result<Vec<std::path::PathBuf>> {
    use std::collections::VecDeque;
    let mut queue: VecDeque<std::path::PathBuf> = VecDeque::new();
    queue.push_back(root.as_ref().to_path_buf());
    let mut features = Vec::new();
    while let Some(dir) = queue.pop_front() {
        if !dir.is_dir() {
            continue;
        }
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                queue.push_back(path);
            } else if path
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("feature"))
                .unwrap_or(false)
            {
                features.push(path);
            }
        }
    }
    features.sort();
    Ok(features)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_link_cite_string_names_m4_61_and_batch_type() {
        // The cite string is load-bearing on the M4-61 forward-link
        // discipline. Future renames (e.g. M4-61 → M4-61a, or
        // `Batch` → `Cursor`) must also flip this pin so log
        // readers stay routed correctly per the W11Z sister-cite
        // sweep convention.
        assert!(M4_61_FORWARD_LINK.contains("M4-61"));
        assert!(
            M4_61_FORWARD_LINK.contains("Batch"),
            "constant must cite the actual M4-61 type name (`Batch`); \
             the W11β-era `BatchCursor` cite was retired in W11Z",
        );
    }

    #[test]
    fn vendored_count_pin_matches_provenance_doc() {
        // Pinned to the openCypher@583c1419 vendored snapshot count
        // recorded in `tck/PROVENANCE.md`. A change here MUST be
        // accompanied by a PROVENANCE update.
        assert_eq!(VENDORED_FEATURE_COUNT, 220);
    }

    #[test]
    fn default_feature_path_points_into_vendored_tree() {
        assert!(DEFAULT_FEATURE_PATH.starts_with("tck/features/"));
        assert!(DEFAULT_FEATURE_PATH.ends_with(".feature"));
    }
}
