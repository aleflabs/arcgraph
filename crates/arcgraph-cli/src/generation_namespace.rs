//! INV-M5.22 — the generation-namespace registry (single source of truth).
//!
//! Every offline tool that creates, renames, or sweeps a generation directory
//! in a data-dir root resolves its directory names HERE. No other module in
//! this crate may contain a `gen-` path literal (enforced by
//! `tests::grep_lint_no_stray_generation_literals`, the same guardrail
//! pattern as the repo's other lexical lints), and prefix sets are pairwise
//! disjoint (enforced by `tests::namespaces_are_pairwise_disjoint`).
//!
//! Ownership table (`docs/design/M5D-REDESIGN-AMENDMENT.md` §2.4):
//!
//! ```text
//! gen-v9[.building]        owner = M3 migration        (data_dir_migration.rs)
//! gen-v10[.building]       owner = M4 v5→v6 migration  (data_dir_migration.rs)
//! gen-load-v6[.building]   owner = M5 fresh load       (m5_load.rs)
//! .gen-*.cleanup           owner = the tool that wrote the matching prefix
//! ```
//!
//! Rules (each carried by a test here or in the owning tool's gate):
//! 1. Every create/rename/sweep of a generation dir goes through these
//!    accessors.
//! 2. **A tool may sweep only its own prefix.** Cross-tool plant tests live in
//!    `tests/m5_load_attach_gate.rs` (a foreign `.building` survives the other
//!    tool's resume byte-identical).
//! 3. Prefix sets are pairwise disjoint (test below).
//! 4. Bootstrap remains `CURRENT`-driven and namespace-agnostic; its one
//!    explicit M4-fallback filter uses `GenerationTool::M4Migration`'s
//!    accessor instead of a string literal.
//!
//! Budget (performance-budget discipline): compile-time constants; zero runtime cost.

/// One offline generation-producing tool. Each owns exactly one
/// `building`/`final` directory-name pair in the data-dir root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationTool {
    /// The v4→v5 offline migration (`arcgraph migrate upgrade-data-dir`).
    M3Migration,
    /// The v5→v6 offline migration (same CLI entry point, second leg).
    M4Migration,
    /// The M5 leg-(c) offline bootstrap-load (`arcgraph load`).
    M5Load,
}

impl GenerationTool {
    /// Every registered tool, for disjointness sweeps.
    pub const ALL: [Self; 3] = [Self::M3Migration, Self::M4Migration, Self::M5Load];

    /// The unpublished beside-build directory name this tool owns.
    #[must_use]
    pub const fn building_dir(self) -> &'static str {
        match self {
            Self::M3Migration => "gen-v9.building",
            Self::M4Migration => "gen-v10.building",
            Self::M5Load => "gen-load-v6.building",
        }
    }

    /// The committed generation directory name this tool owns.
    #[must_use]
    pub const fn final_dir(self) -> &'static str {
        match self {
            Self::M3Migration => "gen-v9",
            Self::M4Migration => "gen-v10",
            Self::M5Load => "gen-load-v6",
        }
    }

    /// The retirement/cleanup tombstone name for this tool's *predecessor*
    /// reap, when the tool has one. Only the v5→v6 migration retires a prior
    /// generation today (`.gen-v9.cleanup`, owned by the tool that wrote the
    /// matching `gen-v9` prefix per the §2.4 table); the fresh-load leg starts
    /// from a virgin dir and has nothing to retire.
    #[must_use]
    pub const fn cleanup_dir(self) -> Option<&'static str> {
        match self {
            Self::M3Migration | Self::M5Load => None,
            Self::M4Migration => Some(".gen-v9.cleanup"),
        }
    }

    /// Every directory name this tool owns in a data-dir root.
    #[must_use]
    pub fn owned_names(self) -> Vec<&'static str> {
        let mut names = vec![self.building_dir(), self.final_dir()];
        if let Some(cleanup) = self.cleanup_dir() {
            names.push(cleanup);
        }
        names
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-M5.22 rule 3 — prefix sets are pairwise disjoint, so no tool can
    /// ever address (and therefore sweep or misread) another tool's work by
    /// name. RED-on-revert: point [`GenerationTool::M5Load`] at `gen-v10` and
    /// this test fails before any gate has to.
    #[test]
    fn namespaces_are_pairwise_disjoint() {
        let mut seen = std::collections::BTreeSet::new();
        for tool in GenerationTool::ALL {
            for name in tool.owned_names() {
                assert!(
                    seen.insert(name),
                    "generation name {name:?} is claimed by two tools"
                );
            }
        }
        // The loader prefix must not collide with either migration prefix
        // even by *prefixing* (a sweep of `gen-load-v6*` must never match
        // `gen-v10*` and vice versa).
        for a in GenerationTool::ALL {
            for b in GenerationTool::ALL {
                if a == b {
                    continue;
                }
                assert!(
                    !a.final_dir().starts_with(b.final_dir()) || a.final_dir() == b.final_dir(),
                    "{:?} final name is prefixed by {:?}'s",
                    a,
                    b
                );
                assert_ne!(a.building_dir(), b.building_dir());
            }
        }
    }

    /// INV-M5.22 rule 1 — grep-lint: no module in this crate other than the
    /// registry may contain a `gen-` generation-name string literal. This is
    /// the lexical guardrail that killed V-1b as a class (the closed PR #1504
    /// hardcoded `gen-v10` in `m5_load.rs` and collided with the migrate
    /// tool's namespace). Comments are exempt (the scan matches only inside
    /// string literals); tests/gates are exempt (oracle independence — gates
    /// assert on-disk names without consulting the registry).
    #[test]
    fn grep_lint_no_stray_generation_literals() {
        let src_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut offenders = Vec::new();
        scan_dir(&src_root, &mut offenders);
        assert!(
            offenders.is_empty(),
            "generation-name literals outside generation_namespace.rs \
             (route them through GenerationTool accessors): {offenders:#?}"
        );
    }

    fn scan_dir(dir: &std::path::Path, offenders: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).expect("read src dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                scan_dir(&path, offenders);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "rs")
                || path
                    .file_name()
                    .is_some_and(|name| name == "generation_namespace.rs")
            {
                continue;
            }
            let body = std::fs::read_to_string(&path).expect("read source file");
            for (idx, line) in body.lines().enumerate() {
                if line_has_generation_literal(line) {
                    offenders.push(format!("{}:{}: {}", path.display(), idx + 1, line.trim()));
                }
            }
        }
    }

    /// True when a double-quoted string literal on this line contains a
    /// generation directory name (`gen-v9`, `gen-v10`, `gen-load-v6`, or a
    /// `.gen-*.cleanup` tombstone). Plain-comment mentions do not match.
    fn line_has_generation_literal(line: &str) -> bool {
        let code = line.split("//").next().unwrap_or(line);
        let mut rest = code;
        while let Some(start) = rest.find('"') {
            let tail = &rest[start + 1..];
            let Some(end) = tail.find('"') else { break };
            let literal = &tail[..end];
            if literal.contains("gen-v9")
                || literal.contains("gen-v10")
                || literal.contains("gen-load-v6")
                || (literal.contains(".gen-") && literal.contains("cleanup"))
            {
                return true;
            }
            rest = &tail[end + 1..];
        }
        false
    }
}
