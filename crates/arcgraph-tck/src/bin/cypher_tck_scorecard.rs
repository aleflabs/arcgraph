//! `cypher-tck-scorecard` — regenerate the openCypher TCK conformance
//! scorecard markdown by walking the vendored feature tree.
//!
//! Pure filesystem-walk + token-bounded keyword detection (no Cypher
//! parse; no executor dispatch). The actual markdown rendering lives
//! in [`arcgraph_tck::scorecard::format_markdown`] so the byte-stable
//! invariant test
//! `arcgraph_tck::scorecard::tests::scorecard_markdown_matches_in_tree_snapshot`
//! can re-derive the markdown without invoking this binary as a
//! subprocess.
//!
//! ## Usage
//!
//! ```bash
//! cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard \
//!   > docs/conformance/cypher-tck-scorecard.md
//! ```

use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let feature_root = Path::new(manifest_dir).join("tck").join("features");

    let summary = match arcgraph_tck::scorecard::build_summary(&feature_root) {
        Ok(s) => s,
        Err(err) => {
            eprintln!(
                "cypher-tck-scorecard: failed to walk vendored TCK feature tree \
                 at {feature_root:?}: {err}"
            );
            return ExitCode::from(2);
        }
    };

    // No trailing newline appended — `format_markdown` already ends
    // with a final `\n` after the last paragraph. `print!` would
    // suffice, but `print!` with a trailing `\n`-bearing String is
    // identical and reads more naturally.
    print!("{}", arcgraph_tck::scorecard::format_markdown(&summary));
    ExitCode::SUCCESS
}
