//! W24-CYPHER-TCK — openCypher conformance scorecard categorization.
//!
//! This module classifies vendored TCK scenarios into the four-bucket
//! conformance taxonomy ratified by ADR-095:
//!
//! * **PASS-eligible** — pure-read scenarios expected to execute through
//!   the ArcGraph query engine at v1.0-α. Whether they ACTUALLY pass
//!   depends on executor maturity; categorization here is the STATIC
//!   eligibility judgement, independent of the cucumber harness run.
//! * **NOT-APPLICABLE — write-op** — scenarios whose `having executed`
//!   setup, control query, or assertion docstring contains a Cypher
//!   write op (`CREATE` / `MERGE` / `DELETE` / `DETACH DELETE` / `SET`
//!   / `REMOVE`). Blocked at v1.0-α per ADR-006 amendment-01 (read-only
//!   catalog); lifts at M4-61b (executor write-ops) + M5-08 (graph.ingest
//!   bulk-load).
//! * **NOT-APPLICABLE — parameterized** — scenarios whose docstring uses
//!   `$param`-style parameter binding. Blocked pre-M5-12 (per-tenant
//!   parameter bag).
//! * **NOT-APPLICABLE — out-of-scope** — scenarios using stored
//!   procedures (`CALL ns.proc(...)` / `YIELD`) or other v1.1+ surface
//!   (subquery `CALL { … }`, `LOAD CSV`, `SHOW`).
//!
//! ## Heuristic boundary
//!
//! The categorization is REGEX-BASED and DELIBERATELY HEURISTIC. It
//! looks at literal Cypher tokens in scenario docstrings; it does NOT
//! tokenize or parse Cypher. Since #1300, string literals (`'…'` /
//! `"…"`) and comments (`// …` / `/* … */`) are STRIPPED before the
//! keyword checks run, so a keyword appearing ONLY inside a quoted
//! literal (`RETURN 'CREATE' AS x`) no longer mis-classifies a
//! genuinely-eligible scenario as NA — that was the one remaining
//! denominator-inflation vector (an eligible scenario counted as
//! blocked). The strip is CONSERVATIVE: on ambiguous input
//! (unterminated literal or block comment) it returns the text
//! unchanged, so the failure direction stays over-flag-NA — a real
//! write-op can never slip through as eligible. The pinned counts in
//! `STATIC_SNAPSHOT` below MUST remain stable across vendored-tree
//! refreshes — drift trips the freshness check in
//! `tests::scorecard_freshness_check`.
//!
//! ## What this module is NOT
//!
//! * NOT a cucumber `Writer` — it does not consume executor outcomes.
//!   The runtime dispatch counts are surfaced by `tests/tck.rs` via
//!   `cucumber::writer::Stats` and quoted in the scorecard separately.
//! * NOT a Cypher parser. A future v1.1 expansion could use the
//!   ArcGraph parser to re-categorize once write-ops + params lift, but
//!   the v1.0-α scorecard is the static heuristic only.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

/// Four-bucket conformance verdict for a single scenario.
///
/// Per ADR-095 §"Categorization scheme." `Eligible` does NOT guarantee
/// the scenario passes the cucumber harness — only that no v1.0-α
/// blocker has been detected by the static heuristic. The harness run
/// surfaces the true PASS/FAIL split among `Eligible` scenarios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Verdict {
    /// Static heuristic detects no v1.0-α blocker. Eligible for the
    /// cucumber harness run; the harness's `passed_steps` counter
    /// surfaces whether it actually passes.
    Eligible,
    /// Scenario depends on write-ops blocked by the v1.0-α read-only
    /// catalog (CREATE / MERGE / DELETE / DETACH DELETE / SET / REMOVE).
    NotApplicableWrite,
    /// Scenario depends on `$param` binding; lifts at M5-12.
    NotApplicableParameterized,
    /// Scenario depends on stored procedures (CALL ... YIELD) or other
    /// v1.1+ deferred surface (CALL { … } subqueries, LOAD CSV, SHOW).
    NotApplicableOutOfScope,
}

impl Verdict {
    /// Returns a kebab-case slug used in the scorecard markdown table
    /// column headers. Stable across renames (the column header is
    /// derived from this slug, not from the enum identifier).
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::NotApplicableWrite => "na-write",
            Self::NotApplicableParameterized => "na-param",
            Self::NotApplicableOutOfScope => "na-oos",
        }
    }

    /// Returns the human-readable bucket label for the scorecard
    /// summary table.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Eligible => "Eligible at v1.0-α",
            Self::NotApplicableWrite => "NOT-APPLICABLE (write-op)",
            Self::NotApplicableParameterized => "NOT-APPLICABLE (parameterized)",
            Self::NotApplicableOutOfScope => "NOT-APPLICABLE (out-of-scope)",
        }
    }
}

/// Categorize a single Cypher docstring body against the v1.0-α
/// blocker set. Returns the FIRST blocker hit; precedence is
/// `Write > Parameterized > OutOfScope > Eligible`.
///
/// "Write" wins precedence because the v1.0-α read-only ArcQL surface
/// (ADR-006 amendment-01) is the strictest blocker — a CREATE-then-
/// `$param` scenario would fail at the CREATE regardless of `$param`
/// support.
fn categorize_docstring(body: &str) -> Option<Verdict> {
    // #1300: strip string literals + comments FIRST so a keyword that
    // appears ONLY inside a quoted literal (`RETURN 'CREATE' AS x`) or
    // a comment cannot mis-fire a blocker check and eject a genuinely-
    // eligible scenario from the Eligible denominator. The strip fails
    // toward the pre-#1300 behavior on ambiguous input (see
    // `strip_string_literals`), so the conservative over-flag-NA
    // direction is preserved — a real write-op never slips through.
    let stripped = strip_string_literals(body);

    // The four checks are intentionally case-INSENSITIVE on
    // ASCII-uppercase keywords. The TCK fixtures consistently use
    // uppercase Cypher keywords, but a defensive `to_ascii_uppercase`
    // shield against a future refresh that lowers case is cheap.
    let upper = stripped.to_ascii_uppercase();

    if contains_write_op(&upper) {
        return Some(Verdict::NotApplicableWrite);
    }
    if contains_parameter_ref(&stripped) {
        return Some(Verdict::NotApplicableParameterized);
    }
    if contains_out_of_scope(&upper) {
        return Some(Verdict::NotApplicableOutOfScope);
    }
    None
}

/// Strip Cypher string literals (`'…'` / `"…"`) and comments
/// (`// …` to end-of-line, `/* … */`) from `body`, replacing each with
/// a single space so token boundaries around the removed span survive
/// (`MATCH (n) WHERE n.p = 'x' DELETE n` keeps `DELETE` as a
/// standalone token).
///
/// Escape handling: a backslash inside a literal consumes the next
/// byte, so `\'` / `\"` / `\\` never terminate the literal (the TCK
/// fixtures use `\'` heavily, e.g. `'Jerry O\'Connell'` in
/// `clauses/create/Create4.feature`). openCypher string literals use
/// backslash escapes only — there is no doubled-quote escape in the
/// openCypher 9 grammar; a doubled quote (`''`) degrades gracefully to
/// close-then-reopen, which strips the same spans and stays balanced.
///
/// **Conservative bail**: if the scan ends inside an unterminated
/// literal or block comment, literal parsing is ambiguous and the
/// input is returned UNCHANGED — the caller falls back to the
/// pre-#1300 raw-text keyword match, which over-flags NA rather than
/// letting a write-op slip into the eligible set. (This also covers a
/// hypothetical multi-line string literal reaching the per-line eager
/// check in [`categorize_feature_body`]: each half looks unterminated,
/// so no strip happens and the verdict stays conservative.)
///
/// Byte-level scan: the delimiters (`'`, `"`, `/`, `*`, `\`, `\n`) are
/// all ASCII, and UTF-8 continuation bytes can never equal an ASCII
/// byte, so scanning bytes while slicing only at delimiter positions
/// preserves UTF-8 validity.
fn strip_string_literals(body: &str) -> Cow<'_, str> {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    // Start of the pending verbatim (kept) run.
    let mut copied = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            quote @ (b'\'' | b'"') => {
                // Scan for the matching close quote, honoring `\`
                // escapes.
                let mut j = i + 1;
                let mut terminated = false;
                while j < bytes.len() {
                    if bytes[j] == b'\\' {
                        // Escape consumes the next byte; if the escape
                        // is the last byte the loop exits unterminated.
                        j += 2;
                    } else if bytes[j] == quote {
                        terminated = true;
                        break;
                    } else {
                        j += 1;
                    }
                }
                if !terminated {
                    return Cow::Borrowed(body);
                }
                out.push_str(&body[copied..i]);
                out.push(' ');
                i = j + 1;
                copied = i;
            }
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                // Line comment: drop to end-of-line (the `\n` itself is
                // kept via the next verbatim run).
                out.push_str(&body[copied..i]);
                out.push(' ');
                let mut j = i + 2;
                while j < bytes.len() && bytes[j] != b'\n' {
                    j += 1;
                }
                i = j;
                copied = i;
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                // Block comment: drop through the closing `*/`.
                let mut j = i + 2;
                let mut terminated = false;
                while j + 1 < bytes.len() {
                    if bytes[j] == b'*' && bytes[j + 1] == b'/' {
                        terminated = true;
                        break;
                    }
                    j += 1;
                }
                if !terminated {
                    return Cow::Borrowed(body);
                }
                out.push_str(&body[copied..i]);
                out.push(' ');
                i = j + 2;
                copied = i;
            }
            _ => i += 1,
        }
    }
    if copied == 0 {
        // Nothing was stripped — hand back the original borrow.
        return Cow::Borrowed(body);
    }
    out.push_str(&body[copied..]);
    Cow::Owned(out)
}

/// Detect Cypher write-op keywords with whitespace / paren boundary on
/// the LEFT side and whitespace / `(` / `:` on the RIGHT. Returns true
/// iff any write keyword appears as a structural token (NOT inside an
/// identifier like `CREATEDBY`).
fn contains_write_op(upper: &str) -> bool {
    // `DETACH DELETE` matches via the `DELETE` arm. The keyword set is
    // the v1.0-α read-only-catalog blocker enumerated in ADR-006
    // amendment-01 + ADR-095 §"Heuristic detail."
    const WRITE_KEYWORDS: &[&str] = &["CREATE", "MERGE", "DELETE", "SET", "REMOVE"];
    for kw in WRITE_KEYWORDS {
        if has_token(upper, kw) {
            return true;
        }
    }
    false
}

/// Detect Cypher `$param` references. Case-sensitive because Cypher
/// param names are case-sensitive at the protocol layer (`$user_id` ≠
/// `$USER_ID`); we look for a literal `$` followed by an identifier
/// start char.
fn contains_parameter_ref(body: &str) -> bool {
    // Search byte-wise to avoid UTF-8 string boundary surprises in
    // TCK fixtures that include unicode quoted-string literals.
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'$' && i + 1 < bytes.len() {
            let next = bytes[i + 1];
            if next.is_ascii_alphabetic() || next == b'_' {
                return true;
            }
        }
        i += 1;
    }
    false
}

/// Detect v1.1+ deferred surface — stored procedures (`CALL ... YIELD`)
/// and call-in-tx subqueries (`CALL { ... }`). The single `CALL` token
/// is the canonical marker; `YIELD` is the second-position confirmer.
fn contains_out_of_scope(upper: &str) -> bool {
    // `CALL` alone is the procedure-invocation prefix. The TCK file
    // tree has a `clauses/call/` subdir (~8 features) that we expect
    // to ALL fall into this bucket — confirmed by the freshness check
    // below.
    if has_token(upper, "CALL") {
        return true;
    }
    // `LOAD CSV` + `SHOW` are v1.1+ admin surface (no v1.0-α plan).
    if has_token(upper, "LOAD") || has_token(upper, "SHOW") {
        return true;
    }
    false
}

/// Match `kw` against `haystack` requiring a non-identifier boundary on
/// both sides. Equivalent to `\b<kw>\b` regex without taking a regex
/// dependency. ASCII-only (the TCK keyword set is ASCII).
fn has_token(haystack: &str, kw: &str) -> bool {
    let bytes = haystack.as_bytes();
    let kw_bytes = kw.as_bytes();
    if kw_bytes.is_empty() || bytes.len() < kw_bytes.len() {
        return false;
    }
    let mut i = 0;
    while i + kw_bytes.len() <= bytes.len() {
        if &bytes[i..i + kw_bytes.len()] == kw_bytes {
            let left_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
            let right_idx = i + kw_bytes.len();
            let right_ok = right_idx >= bytes.len() || !is_ident_byte(bytes[right_idx]);
            if left_ok && right_ok {
                return true;
            }
        }
        i += 1;
    }
    false
}

#[inline]
const fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// A single categorized scenario within a feature file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScenarioRecord {
    /// Display name (the text after `Scenario:` / `Scenario Outline:`).
    pub name: String,
    /// Static verdict produced by `categorize_docstring`.
    pub verdict: Verdict,
}

/// Categorize every scenario in a single `.feature` file.
///
/// Pure parser: opens `path`, walks the file line-by-line, tracks
/// scenario boundaries (`Scenario:` / `Scenario Outline:` lines) and
/// triple-quoted docstring blocks (`"""` delimiters). For each
/// scenario, the categorizer runs `categorize_docstring` against the
/// CONCATENATION of all docstring bodies in that scenario; the first
/// verdict hit wins, else `Eligible`.
pub fn categorize_feature_file(path: &Path) -> io::Result<Vec<ScenarioRecord>> {
    let body = std::fs::read_to_string(path)?;
    Ok(categorize_feature_body(&body))
}

/// Categorize a feature file body in-memory. Exposed for unit tests
/// that don't want to touch the filesystem.
#[must_use]
pub fn categorize_feature_body(body: &str) -> Vec<ScenarioRecord> {
    let mut records: Vec<ScenarioRecord> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_doc = String::new();
    let mut in_docstring = false;
    let mut current_verdict: Option<Verdict> = None;

    let flush = |name: &mut Option<String>,
                 doc: &mut String,
                 verdict: &mut Option<Verdict>,
                 records: &mut Vec<ScenarioRecord>| {
        if let Some(n) = name.take() {
            // First-line precedence already encoded in the verdict
            // update during the walk; if no docstring fired, fall
            // back to a final pass over the concatenated docstring
            // body. This catches scenarios where the only docstring
            // body fires AFTER an earlier check missed (no longer
            // possible with current heuristics, but defensive).
            let v = verdict.take().or_else(|| categorize_docstring(doc));
            records.push(ScenarioRecord {
                name: n,
                verdict: v.unwrap_or(Verdict::Eligible),
            });
            doc.clear();
        }
    };

    for raw_line in body.lines() {
        let line = raw_line.trim_start();

        if line.starts_with("\"\"\"") {
            in_docstring = !in_docstring;
            continue;
        }

        if in_docstring {
            current_doc.push_str(raw_line);
            current_doc.push('\n');
            // Update verdict eagerly so we get FIRST-blocker precedence
            // across multiple docstrings within a single scenario.
            if current_verdict.is_none() {
                if let Some(v) = categorize_docstring(raw_line) {
                    current_verdict = Some(v);
                }
            }
            continue;
        }

        if let Some(rest) = line.strip_prefix("Scenario Outline:") {
            flush(
                &mut current_name,
                &mut current_doc,
                &mut current_verdict,
                &mut records,
            );
            current_name = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("Scenario:") {
            flush(
                &mut current_name,
                &mut current_doc,
                &mut current_verdict,
                &mut records,
            );
            current_name = Some(rest.trim().to_string());
        }
    }

    flush(
        &mut current_name,
        &mut current_doc,
        &mut current_verdict,
        &mut records,
    );
    records
}

/// Aggregated counts per `(family, verdict)` cell + per-family totals.
#[derive(Debug, Default, Clone)]
pub struct ScorecardSummary {
    /// Per-family (e.g. `clauses/return`) per-verdict count.
    pub per_family: BTreeMap<String, FamilyCounts>,
    /// Sum across all families.
    pub total: FamilyCounts,
}

/// Verdict count tuple for a single family or the global total.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FamilyCounts {
    pub eligible: usize,
    pub na_write: usize,
    pub na_param: usize,
    pub na_oos: usize,
}

impl FamilyCounts {
    /// Total scenario count for this row.
    #[must_use]
    pub const fn total(&self) -> usize {
        self.eligible + self.na_write + self.na_param + self.na_oos
    }

    /// Eligible percentage (0-100), rounded to one decimal. Returns
    /// 0.0 when total is 0 (avoids divide-by-zero in empty families).
    #[must_use]
    pub fn eligible_pct(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.eligible as f64 / total as f64 * 1000.0).round() / 10.0
    }

    /// Increment the count for `verdict`.
    pub fn bump(&mut self, verdict: Verdict) {
        match verdict {
            Verdict::Eligible => self.eligible += 1,
            Verdict::NotApplicableWrite => self.na_write += 1,
            Verdict::NotApplicableParameterized => self.na_param += 1,
            Verdict::NotApplicableOutOfScope => self.na_oos += 1,
        }
    }
}

/// Build a [`ScorecardSummary`] by walking every `.feature` file under
/// `feature_root` and categorizing every scenario.
///
/// Family extraction: the family key is the path components between
/// `feature_root` and the `.feature` file, joined by `/`. E.g.
/// `tck/features/clauses/return/Return1.feature` under root
/// `tck/features` yields family `clauses/return`.
pub fn build_summary(feature_root: &Path) -> io::Result<ScorecardSummary> {
    let mut summary = ScorecardSummary::default();
    let feature_files = enumerate_feature_files_sorted(feature_root)?;
    for path in feature_files {
        let family = family_for(feature_root, &path);
        let records = categorize_feature_file(&path)?;
        let row = summary.per_family.entry(family).or_default();
        for r in records {
            row.bump(r.verdict);
            summary.total.bump(r.verdict);
        }
    }
    Ok(summary)
}

fn enumerate_feature_files_sorted(root: &Path) -> io::Result<Vec<PathBuf>> {
    crate::enumerate_feature_files(root)
}

fn family_for(root: &Path, file: &Path) -> String {
    let rel = file.strip_prefix(root).unwrap_or(file);
    let mut components: Vec<&str> = rel
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();
    // Drop the trailing filename so `clauses/return/Return1.feature`
    // becomes `clauses/return`.
    if components
        .last()
        .map(|s| s.ends_with(".feature"))
        .unwrap_or(false)
    {
        components.pop();
    }
    if components.is_empty() {
        "root".to_string()
    } else {
        components.join("/")
    }
}

/// Render the scorecard markdown for `summary` into a `String`.
///
/// Stable shape: 5 sections (title + provenance / summary table /
/// per-family breakdown / runtime harness observation / commitment
/// level + forward pin). Diff-friendly markdown: pipe-delimited tables
/// with single-space padding so reviewers spot drift on `git diff`
/// without column-realignment noise.
///
/// **Byte-stable invariant**: `format_markdown(&build_summary(root))`
/// MUST equal the in-tree `docs/conformance/cypher-tck-scorecard.md`
/// at every clean main HEAD. The
/// `tests::scorecard_markdown_matches_in_tree_snapshot` unit test
/// is the load-bearing assertion that prevents drift between the
/// generator and the checked-in report.
#[must_use]
pub fn format_markdown(summary: &ScorecardSummary) -> String {
    use std::fmt::Write;
    let mut out = String::new();

    // ----- 1. Title + provenance -----
    let _ = writeln!(
        &mut out,
        "# openCypher TCK Conformance Scorecard — ArcGraph v1.0-α"
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "> **Generated by:** `cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard`"
    );
    let _ = writeln!(
        &mut out,
        "> **Upstream pin:** openCypher@`583c1419` (see \
         `crates/arcgraph-tck/tck/PROVENANCE.md`)"
    );
    let _ = writeln!(
        &mut out,
        "> **Vendored:** {feat} features, {scen} scenarios total.",
        feat = crate::VENDORED_FEATURE_COUNT,
        scen = summary.total.total(),
    );
    let _ = writeln!(
        &mut out,
        "> **Categorization scheme:** ADR-095 §\"Categorization scheme\" \
         (Eligible / NA-Write / NA-Param / NA-OOS)."
    );
    let _ = writeln!(&mut out);

    // ----- 2. Summary table -----
    let _ = writeln!(&mut out, "## Summary");
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "| Bucket | Count | % of total |");
    let _ = writeln!(&mut out, "| --- | ---: | ---: |");
    let total = summary.total.total();
    for v in [
        Verdict::Eligible,
        Verdict::NotApplicableWrite,
        Verdict::NotApplicableParameterized,
        Verdict::NotApplicableOutOfScope,
    ] {
        let n = match v {
            Verdict::Eligible => summary.total.eligible,
            Verdict::NotApplicableWrite => summary.total.na_write,
            Verdict::NotApplicableParameterized => summary.total.na_param,
            Verdict::NotApplicableOutOfScope => summary.total.na_oos,
        };
        let pct = if total == 0 {
            0.0
        } else {
            (n as f64 / total as f64 * 1000.0).round() / 10.0
        };
        let _ = writeln!(&mut out, "| {} | {n} | {pct:.1}% |", v.label());
    }
    let _ = writeln!(&mut out, "| **Total** | **{total}** | **100.0%** |");
    let _ = writeln!(&mut out);

    // ----- 3. Per-family breakdown -----
    let _ = writeln!(&mut out, "## Per-family breakdown");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "Family path is the directory under `crates/arcgraph-tck/tck/features/`."
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "| Family | Eligible | NA-Write | NA-Param | NA-OOS | Total | Eligible % |"
    );
    let _ = writeln!(
        &mut out,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for (family, counts) in &summary.per_family {
        append_family_row(&mut out, family, counts);
    }
    append_family_row(&mut out, "**total**", &summary.total);
    let _ = writeln!(&mut out);

    // ----- 4. Runtime observation -----
    let _ = writeln!(&mut out, "## Runtime harness observation");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "Static categorization above is the LOAD-BEARING numerical claim of \
         this scorecard. Runtime harness output (the cucumber dispatch through \
         `arcgraph_query::QueryEngine::execute`) is reported separately by"
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "```bash");
    let _ = writeln!(&mut out, "cargo test --release -p arcgraph-tck");
    let _ = writeln!(&mut out, "```");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "with the harness writing a single summary line of the shape \
         `passed=N failed=N skipped=N parsing_errors=N` (steps, not scenarios)."
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "At v1.0-α the harness reports \
         `passed≈6787 failed≈977 skipped≈2920 parsing_errors=0`. The \
         `passed_steps >= 6500` floor in `tests/tck.rs::main` is the regression \
         gate — a parser/binder/executor regression dropping the pass count \
         below the floor trips the assertion."
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "**Feature-count reconciliation.** The static walk counts 220 \
         `.feature` files under `crates/arcgraph-tck/tck/features/` and \
         is asserted in `tests/tck.rs::main` against \
         `arcgraph_tck::VENDORED_FEATURE_COUNT`. The cucumber-rs runtime \
         may report a smaller internal `[Summary] N features` count when \
         it filters header-only or zero-scenario features at parse time \
         (with `parsing_errors=0` because they parse cleanly, they just \
         contribute no executable scenarios). The two counts measure \
         different things — disk presence vs. cucumber-internal feature \
         accounting — and the static walk count (220) is the load-bearing \
         one for this scorecard."
    );
    let _ = writeln!(&mut out);

    // ----- 5. Commitment level + forward pin -----
    let _ = writeln!(&mut out, "## Conformance commitment level (v1.0-α)");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "Under ADR-095 §\"Commitment level\", ArcGraph commits to:"
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "Every floor below is enforced by a named unit test or harness \
         assertion; the cite IS the load-bearing tripwire."
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "* **Per-family static categorization-shape floor**: \
         `expressions/literals` ≥ 100 Eligible with zero NA-Write / \
         NA-Param / NA-OOS contamination; `clauses/call` has 0 Eligible \
         with ≥ 40 total scenarios. Tripwires: \
         `commitment_level_floor_expressions_literals` + \
         `commitment_level_floor_clauses_call_is_oos` in \
         `crates/arcgraph-tck/src/scorecard.rs::tests`."
    );
    let _ = writeln!(
        &mut out,
        "* **Global static snapshot-stability floor**: every bucket count \
         in `STATIC_SNAPSHOT_*` (697 Eligible / 856 NA-Write / 22 NA-Param \
         / 40 NA-OOS = 1 615 scenarios) matches the live filesystem walk \
         under `crates/arcgraph-tck/tck/features/`. Tripwire: \
         `scorecard_freshness_check`."
    );
    let _ = writeln!(
        &mut out,
        "* **Global runtime aggregate step-pass floor**: \
         `passed_steps >= 6500` against the v1.0-α baseline \
         (`passed=6787 failed=977 skipped=2920 parsing_errors=0`). \
         Tripwire: `tests/tck.rs::main`. Aggregate step-level, not \
         per-family scenario-level."
    );
    let _ = writeln!(
        &mut out,
        "* **Write-clause runtime gate**: NOT-APPLICABLE at v1.0-α. \
         Documented as a categorization bucket per ADR-006 amendment-01 \
         (v1.0 ArcQL is read-only by design); the gate lifts at v1.0-GA \
         (M4-61b executor write-ops)."
    );
    let _ = writeln!(
        &mut out,
        "* **Per-family runtime pass-rate gate — KNOWN GAP, \
         forward-pinned to v1.0-GA amendment**. `cucumber::writer::Stats` \
         exposes only aggregate step counts (`passed_steps` / \
         `failed_steps` / `skipped_steps`); a custom cucumber `Writer` \
         impl is required for per-feature / per-scenario pass tracking. \
         Forward-pinned to ADR-095 amendment-01 (target v1.0-GA)."
    );
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "Lifting any floor in a follow-up wave requires a recorded \
         regression-acceptance ADR — see ADR-095 §\"Floor lift protocol\"."
    );
    let _ = writeln!(&mut out);

    let _ = writeln!(&mut out, "## v1.1+ forward-pin");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "* **NA-Write lifts** at M4-61b (executor write-ops) + M5-08 \
         (graph.ingest bulk-load). At full lift, the `clauses/create`, \
         `clauses/delete`, `clauses/merge`, `clauses/set`, `clauses/remove` \
         families re-enter the eligibility pool."
    );
    let _ = writeln!(
        &mut out,
        "* **NA-Param lifts** at M5-12 (per-tenant parameter bag). The \
         `$param`-bearing scenarios across `expressions/*` and \
         `clauses/match-where` re-enter the eligibility pool."
    );
    let _ = writeln!(
        &mut out,
        "* **NA-OOS** is split: `clauses/call` (~8 features, ~30-40 scenarios) \
         lifts at v1.1 (procedure surface); `SHOW`/`LOAD CSV` remain \
         out-of-scope through v1.2-GA per `docs/roadmap.md` §\"Notes for \
         engineering\" #3."
    );
    let _ = writeln!(&mut out);

    let _ = writeln!(&mut out, "## Re-deriving this scorecard");
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "```bash");
    let _ = writeln!(
        &mut out,
        "cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard \\"
    );
    let _ = writeln!(&mut out, "  > docs/conformance/cypher-tck-scorecard.md");
    let _ = writeln!(&mut out, "```");
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "The categorization counts above are pinned in \
         `crates/arcgraph-tck/src/scorecard.rs::STATIC_SNAPSHOT`. The \
         `scorecard_freshness_check` unit test re-derives them from the \
         live filesystem walk and asserts equality — drift trips the test \
         and the scorecard must be regenerated in lockstep."
    );

    out
}

fn append_family_row(out: &mut String, family: &str, counts: &FamilyCounts) {
    use std::fmt::Write;
    let _ = writeln!(
        out,
        "| {family} | {e} | {w} | {p} | {o} | {total} | {pct:.1}% |",
        e = counts.eligible,
        w = counts.na_write,
        p = counts.na_param,
        o = counts.na_oos,
        total = counts.total(),
        pct = counts.eligible_pct(),
    );
}

/// Pinned categorization snapshot at the openCypher@583c1419 vendoring.
///
/// This snapshot is the LOAD-BEARING numerical claim of the W24-CYPHER-
/// TCK conformance scorecard. The freshness check
/// `tests::scorecard_freshness_check` re-derives the categorization
/// from the live filesystem walk and asserts equality against this
/// snapshot. If a future vendored-tree refresh changes the counts, the
/// snapshot MUST move in lockstep with the scorecard markdown report
/// at `docs/conformance/cypher-tck-scorecard.md`.
///
/// **Provenance**: derived via
/// `cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard`
/// against the W24-CYPHER-TCK branch (post-W24-α main HEAD = be7e052).
pub const STATIC_SNAPSHOT: FamilyCounts = FamilyCounts {
    eligible: STATIC_SNAPSHOT_ELIGIBLE,
    na_write: STATIC_SNAPSHOT_NA_WRITE,
    na_param: STATIC_SNAPSHOT_NA_PARAM,
    na_oos: STATIC_SNAPSHOT_NA_OOS,
};

// Pinned at the W24-CYPHER-TCK landing (openCypher@583c1419 vendored
// snapshot, post-W24-α main HEAD = be7e052). Sum: 697 + 856 + 22 + 40
// = 1615 scenarios across 220 features. Derived by running
// `cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard` and
// reading the bottom-row total. The `scorecard_freshness_check` unit
// test re-derives these numbers from the live filesystem walk and
// asserts equality.
const STATIC_SNAPSHOT_ELIGIBLE: usize = 697;
const STATIC_SNAPSHOT_NA_WRITE: usize = 856;
const STATIC_SNAPSHOT_NA_PARAM: usize = 22;
const STATIC_SNAPSHOT_NA_OOS: usize = 40;

/// Total scenario count pinned by the snapshot. Equals
/// `STATIC_SNAPSHOT.total()` once the snapshot is populated.
pub const STATIC_SNAPSHOT_TOTAL_SCENARIOS: usize = STATIC_SNAPSHOT_ELIGIBLE
    + STATIC_SNAPSHOT_NA_WRITE
    + STATIC_SNAPSHOT_NA_PARAM
    + STATIC_SNAPSHOT_NA_OOS;

#[cfg(test)]
mod tests {
    use super::*;

    // ============================================================
    // Per-keyword detection (adversarial unit tests)
    //
    // Per W23-MFI-6 no-op trampoline anti-pattern sweep (memory:
    // `feedback_noop_trampoline_anti_pattern.md`; founding-incident
    // anchors W16 #332 nbf + W17 #349 inbound expand): every contract
    // claim in the doc-comment gets an adversarial test that proves
    // the contract holds under the hostile input the doc-comment
    // promises to reject.
    // ============================================================

    #[test]
    fn categorize_write_op_create() {
        assert_eq!(
            categorize_docstring("CREATE (n:Person)"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_lowercase() {
        // TCK fixtures use uppercase; defensive lowercase check.
        assert_eq!(
            categorize_docstring("create (n:Person)"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_delete() {
        assert_eq!(
            categorize_docstring("MATCH (n) DELETE n"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_detach_delete() {
        // `DETACH DELETE` is the DELETE arm.
        assert_eq!(
            categorize_docstring("MATCH (n) DETACH DELETE n"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_merge() {
        assert_eq!(
            categorize_docstring("MERGE (n:Person {id: 1})"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_set() {
        assert_eq!(
            categorize_docstring("MATCH (n) SET n.foo = 1"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_remove() {
        assert_eq!(
            categorize_docstring("MATCH (n) REMOVE n.foo"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn categorize_write_op_token_boundary_does_not_match_substring() {
        // `CREATEDBY` is an identifier, not the CREATE keyword.
        // (Synthetic — TCK fixtures don't actually contain this, but
        // the heuristic boundary check must hold for any future drift.)
        assert_eq!(
            categorize_docstring("RETURN n.CREATEDBY"),
            None,
            "CREATE keyword detector must respect identifier boundaries",
        );
    }

    #[test]
    fn categorize_parameterized_dollar_ident() {
        assert_eq!(
            categorize_docstring("MATCH (n {id: $id}) RETURN n"),
            Some(Verdict::NotApplicableParameterized)
        );
    }

    #[test]
    fn categorize_parameterized_underscore() {
        assert_eq!(
            categorize_docstring("RETURN $_my_param"),
            Some(Verdict::NotApplicableParameterized)
        );
    }

    #[test]
    fn categorize_parameterized_does_not_match_dollar_alone() {
        // Bare `$` (rare in Cypher; could appear in string literals)
        // does not count.
        assert_eq!(
            categorize_docstring("RETURN '$'"),
            None,
            "bare $ without identifier must not trigger NA-param",
        );
    }

    #[test]
    fn categorize_parameterized_does_not_match_dollar_digit() {
        // `$1` is positional binding in some dialects (not openCypher
        // TCK) — but per `contains_parameter_ref`, identifier-start
        // must be letter or `_`. Digit doesn't trigger.
        assert_eq!(
            categorize_docstring("RETURN $1"),
            None,
            "$<digit> must not trigger NA-param (not Cypher identifier start)",
        );
    }

    #[test]
    fn categorize_out_of_scope_call() {
        assert_eq!(
            categorize_docstring("CALL db.labels()"),
            Some(Verdict::NotApplicableOutOfScope)
        );
    }

    #[test]
    fn categorize_out_of_scope_call_yield() {
        assert_eq!(
            categorize_docstring("CALL db.labels() YIELD label"),
            Some(Verdict::NotApplicableOutOfScope)
        );
    }

    #[test]
    fn categorize_eligible_pure_read() {
        assert_eq!(categorize_docstring("MATCH (n) RETURN n"), None);
    }

    #[test]
    fn categorize_eligible_with_aggregation() {
        assert_eq!(categorize_docstring("MATCH (n) RETURN count(n)"), None);
    }

    // ============================================================
    // #1300 — string-literal / comment stripping.
    //
    // A keyword appearing ONLY inside a quoted string literal (or a
    // comment) must NOT trigger a blocker classification: that was the
    // one remaining "genuinely-eligible scenario counted as blocked"
    // denominator-inflation vector. RED-on-revert: removing the
    // `strip_string_literals` call in `categorize_docstring` flips
    // every `*_is_eligible` test below back to the NA mis-class.
    // ============================================================

    #[test]
    fn literal_keyword_create_is_eligible() {
        assert_eq!(
            categorize_docstring("RETURN 'CREATE' AS x"),
            None,
            "CREATE inside a single-quoted literal must not classify NA-write",
        );
    }

    #[test]
    fn literal_keyword_delete_double_quoted_is_eligible() {
        assert_eq!(
            categorize_docstring("RETURN \"DELETE\" AS x"),
            None,
            "DELETE inside a double-quoted literal must not classify NA-write",
        );
    }

    #[test]
    fn literal_param_ref_is_eligible() {
        assert_eq!(
            categorize_docstring("RETURN '$foo' AS x"),
            None,
            "$ident inside a string literal must not classify NA-param",
        );
    }

    #[test]
    fn literal_out_of_scope_keywords_are_eligible() {
        assert_eq!(
            categorize_docstring("RETURN 'CALL LOAD SHOW' AS x"),
            None,
            "OOS keywords inside a string literal must not classify NA-OOS",
        );
    }

    #[test]
    fn real_write_op_outside_literal_still_na_write() {
        // The strip must remove ONLY the literal spans — a genuine
        // keyword adjacent to (or between) literals still classifies.
        assert_eq!(
            categorize_docstring("MATCH (n) WHERE n.p = 'x' DELETE n"),
            Some(Verdict::NotApplicableWrite),
        );
        assert_eq!(
            categorize_docstring("MATCH (n) SET n.name = 'CREATE'"),
            Some(Verdict::NotApplicableWrite),
            "a real SET with a keyword-decoy literal must stay NA-write",
        );
    }

    #[test]
    fn escaped_quote_does_not_desync_literal_tracking() {
        // `\'` continues the literal (Create4.feature fixture shape).
        assert_eq!(
            categorize_docstring("RETURN 'it\\'s not a CREATE op' AS x"),
            None,
            "backslash-escaped quote must not terminate the literal early",
        );
        assert_eq!(
            categorize_docstring("CREATE (m {tagline: 'Don\\'t Breathe.'})"),
            Some(Verdict::NotApplicableWrite),
            "a real CREATE with an escaped-quote literal must stay NA-write",
        );
    }

    #[test]
    fn quote_inside_other_quote_kind_stays_in_literal() {
        assert_eq!(categorize_docstring("RETURN \"it's a CREATE\" AS x"), None);
        assert_eq!(
            categorize_docstring("RETURN 'she said \"CREATE\"' AS x"),
            None
        );
    }

    #[test]
    fn doubled_quote_degrades_to_close_reopen() {
        // openCypher 9 has no doubled-quote escape; `''` scans as
        // close-then-reopen, which strips the same spans and stays
        // balanced — the keyword is still inside a quoted region.
        assert_eq!(categorize_docstring("RETURN 'it''s CREATE' AS x"), None);
    }

    #[test]
    fn unterminated_literal_falls_back_conservative() {
        // Ambiguous literal parse → no strip → pre-#1300 raw-text
        // behavior (over-flag NA, never write-op-slips-through).
        assert_eq!(
            categorize_docstring("RETURN 'CREATE"),
            Some(Verdict::NotApplicableWrite),
        );
    }

    #[test]
    fn line_comment_keyword_is_eligible() {
        assert_eq!(
            categorize_docstring("MATCH (n) RETURN n // a CREATE remark"),
            None,
            "keyword inside a // comment must not classify NA-write",
        );
    }

    #[test]
    fn block_comment_keyword_is_eligible() {
        assert_eq!(
            categorize_docstring("/* CREATE */ MATCH (n) RETURN n"),
            None,
            "keyword inside a /* */ comment must not classify NA-write",
        );
    }

    #[test]
    fn unterminated_block_comment_falls_back_conservative() {
        assert_eq!(
            categorize_docstring("MATCH (n) RETURN n /* CREATE"),
            Some(Verdict::NotApplicableWrite),
        );
    }

    #[test]
    fn strip_preserves_token_boundaries_and_borrows_when_clean() {
        // Replacing a literal with a single space keeps surrounding
        // tokens apart (no accidental identifier fusion).
        assert_eq!(
            strip_string_literals("RETURN 'CREATE' AS x"),
            "RETURN   AS x"
        );
        // No strippable content → the original borrow comes back.
        assert!(matches!(
            strip_string_literals("MATCH (n) RETURN n"),
            Cow::Borrowed(_)
        ));
    }

    // ============================================================
    // Precedence (Write > Param > OOS > Eligible)
    // ============================================================

    #[test]
    fn precedence_write_over_param() {
        // A scenario that mixes a parameterized read with a CREATE
        // setup is NA-Write because the Write blocker is strictly
        // stronger.
        assert_eq!(
            categorize_docstring("CREATE (n {id: $id})"),
            Some(Verdict::NotApplicableWrite)
        );
    }

    #[test]
    fn precedence_param_over_oos() {
        // A parameterized CALL is NA-Param because Param precedes OOS
        // in the precedence chain (Param is also closer to lifting at
        // M5-12 than CALL procedures, which are v1.1+).
        assert_eq!(
            categorize_docstring("CALL db.proc($x)"),
            Some(Verdict::NotApplicableParameterized)
        );
    }

    // ============================================================
    // Feature-body parsing
    // ============================================================

    #[test]
    fn parse_simple_feature() {
        let body = "Feature: F1\n\n  Scenario: [1] read\n    Given an empty graph\n    When executing query:\n      \"\"\"\n      MATCH (n) RETURN n\n      \"\"\"\n";
        let records = categorize_feature_body(body);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "[1] read");
        assert_eq!(records[0].verdict, Verdict::Eligible);
    }

    #[test]
    fn parse_feature_with_write_setup() {
        let body = "Feature: F2\n\n  Scenario: [1] read after create\n    Given an empty graph\n    And having executed:\n      \"\"\"\n      CREATE (n:A)\n      \"\"\"\n    When executing query:\n      \"\"\"\n      MATCH (n) RETURN n\n      \"\"\"\n";
        let records = categorize_feature_body(body);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].verdict, Verdict::NotApplicableWrite);
    }

    #[test]
    fn parse_feature_with_scenario_outline() {
        let body = "Feature: F3\n\n  Scenario Outline: [1] outline read\n    Given an empty graph\n    When executing query:\n      \"\"\"\n      MATCH (n) RETURN <prop>\n      \"\"\"\n";
        let records = categorize_feature_body(body);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "[1] outline read");
        assert_eq!(records[0].verdict, Verdict::Eligible);
    }

    #[test]
    fn parse_feature_with_multiple_scenarios() {
        let body = "Feature: F4\n\n  Scenario: [1] read\n    When executing query:\n      \"\"\"\n      MATCH (n) RETURN n\n      \"\"\"\n\n  Scenario: [2] write\n    When executing query:\n      \"\"\"\n      CREATE (n)\n      \"\"\"\n";
        let records = categorize_feature_body(body);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].verdict, Verdict::Eligible);
        assert_eq!(records[1].verdict, Verdict::NotApplicableWrite);
    }

    // ============================================================
    // Family extraction
    // ============================================================

    #[test]
    fn family_for_typical_path() {
        let root = Path::new("/repo/tck/features");
        let file = Path::new("/repo/tck/features/clauses/return/Return1.feature");
        assert_eq!(family_for(root, file), "clauses/return");
    }

    #[test]
    fn family_for_top_level_path() {
        let root = Path::new("/repo/tck/features");
        let file = Path::new("/repo/tck/features/SomeTop.feature");
        assert_eq!(family_for(root, file), "root");
    }

    // ============================================================
    // FamilyCounts arithmetic
    // ============================================================

    #[test]
    fn family_counts_total_and_pct() {
        let c = FamilyCounts {
            eligible: 25,
            na_write: 70,
            na_param: 3,
            na_oos: 2,
        };
        assert_eq!(c.total(), 100);
        assert_eq!(c.eligible_pct(), 25.0);
    }

    #[test]
    fn family_counts_empty_pct() {
        let c = FamilyCounts::default();
        assert_eq!(c.total(), 0);
        // Avoid divide-by-zero in empty families.
        assert_eq!(c.eligible_pct(), 0.0);
    }

    // ============================================================
    // Verdict surface
    // ============================================================

    #[test]
    fn verdict_slugs_are_stable() {
        // Slugs are LOAD-BEARING on the scorecard markdown column
        // headers. Renaming a slug requires a scorecard regen.
        assert_eq!(Verdict::Eligible.slug(), "eligible");
        assert_eq!(Verdict::NotApplicableWrite.slug(), "na-write");
        assert_eq!(Verdict::NotApplicableParameterized.slug(), "na-param");
        assert_eq!(Verdict::NotApplicableOutOfScope.slug(), "na-oos");
    }

    // ============================================================
    // Adversarial: token-boundary on YIELD / LOAD / SHOW
    // ============================================================

    #[test]
    fn out_of_scope_load_csv() {
        assert_eq!(
            categorize_docstring("LOAD CSV FROM 'x.csv'"),
            Some(Verdict::NotApplicableOutOfScope)
        );
    }

    #[test]
    fn out_of_scope_show_databases() {
        assert_eq!(
            categorize_docstring("SHOW DATABASES"),
            Some(Verdict::NotApplicableOutOfScope)
        );
    }

    // ============================================================
    // Live freshness check — re-derives the static snapshot from the
    // vendored TCK tree and asserts equality with `STATIC_SNAPSHOT`.
    //
    // **Role**: CI smoke gate (per ADR-095 §"CI smoke gate"). This
    // test is the fast lane that catches categorization drift
    // BEFORE the cucumber harness body even loads. A failure here
    // means EITHER the vendored tree shifted shape (refresh) OR the
    // categorization heuristic changed (precedence / keyword set);
    // in both cases the scorecard markdown must regenerate in
    // lockstep via:
    //
    //     cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard \
    //       > docs/conformance/cypher-tck-scorecard.md
    //
    // and the `STATIC_SNAPSHOT_*` constants above must flip to
    // match the new derivation.
    //
    // Per `feedback_noop_trampoline_anti_pattern.md` (W23-MFI-6;
    // founding-incident anchors W16 #332 nbf + W17 #349 inbound
    // expand): the doc-comment claim that this check catches drift
    // is exercised in the test body — the assertion is the contract,
    // not the doc-comment.
    // ============================================================

    #[test]
    fn scorecard_freshness_check() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let feature_root = std::path::Path::new(manifest_dir)
            .join("tck")
            .join("features");
        let summary = build_summary(&feature_root).expect("vendored TCK feature tree walk failed");
        assert_eq!(
            summary.total.total(),
            STATIC_SNAPSHOT_TOTAL_SCENARIOS,
            "scorecard freshness drift: total scenario count {observed} \
             does not match pinned snapshot {pinned}. Either the vendored \
             tree shifted shape or categorization changed — regenerate the \
             scorecard markdown and flip the STATIC_SNAPSHOT_* constants.",
            observed = summary.total.total(),
            pinned = STATIC_SNAPSHOT_TOTAL_SCENARIOS,
        );
        assert_eq!(
            summary.total.eligible,
            STATIC_SNAPSHOT.eligible,
            "Eligible-bucket drift: live={live} pinned={pinned}",
            live = summary.total.eligible,
            pinned = STATIC_SNAPSHOT.eligible,
        );
        assert_eq!(
            summary.total.na_write,
            STATIC_SNAPSHOT.na_write,
            "NA-Write bucket drift: live={live} pinned={pinned}",
            live = summary.total.na_write,
            pinned = STATIC_SNAPSHOT.na_write,
        );
        assert_eq!(
            summary.total.na_param,
            STATIC_SNAPSHOT.na_param,
            "NA-Param bucket drift: live={live} pinned={pinned}",
            live = summary.total.na_param,
            pinned = STATIC_SNAPSHOT.na_param,
        );
        assert_eq!(
            summary.total.na_oos,
            STATIC_SNAPSHOT.na_oos,
            "NA-OOS bucket drift: live={live} pinned={pinned}",
            live = summary.total.na_oos,
            pinned = STATIC_SNAPSHOT.na_oos,
        );
    }

    // ============================================================
    // Conformance commitment level — pinned eligibility floors for
    // high-confidence read-path families. Per ADR-095 §"Commitment
    // level", these floors are the v1.0-α "best in class for what we
    // claim" surface. Lifting a floor requires a regression-
    // acceptance ADR.
    // ============================================================

    #[test]
    fn commitment_level_floor_expressions_literals() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let feature_root = std::path::Path::new(manifest_dir)
            .join("tck")
            .join("features");
        let summary = build_summary(&feature_root).expect("vendored tree walk failed");
        let lit = summary
            .per_family
            .get("expressions/literals")
            .copied()
            .unwrap_or_default();
        // expressions/literals must be 100% eligible at v1.0-α —
        // these are pure literal-evaluation scenarios that have
        // zero executor dependency.
        assert_eq!(
            lit.na_write, 0,
            "expressions/literals must not have write-op contamination"
        );
        assert_eq!(
            lit.na_param, 0,
            "expressions/literals must not have parameter contamination"
        );
        assert_eq!(
            lit.na_oos, 0,
            "expressions/literals must not have out-of-scope contamination"
        );
        assert!(
            lit.eligible >= 100,
            "expressions/literals eligibility regression: got {}, expected ≥ 100",
            lit.eligible,
        );
    }

    // ============================================================
    // Byte-stable markdown invariant
    //
    // `format_markdown(&build_summary(root))` MUST equal the in-tree
    // `docs/conformance/cypher-tck-scorecard.md` at every clean main
    // HEAD. This is the load-bearing assertion that prevents drift
    // between the markdown generator and the checked-in scorecard.
    //
    // If this test fails, the scorecard markdown is stale. Regenerate
    // via:
    //
    //     cargo run --quiet -p arcgraph-tck --bin cypher-tck-scorecard \
    //       > docs/conformance/cypher-tck-scorecard.md
    //
    // and re-run the workspace tests.
    // ============================================================

    #[test]
    fn scorecard_markdown_matches_in_tree_snapshot() {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let feature_root = std::path::Path::new(manifest_dir)
            .join("tck")
            .join("features");
        let summary = build_summary(&feature_root).expect("vendored tree walk failed");
        let regenerated = format_markdown(&summary);

        // Walk up from `CARGO_MANIFEST_DIR` to the workspace root
        // (`crates/arcgraph-tck` → `../..`) so the in-tree report
        // path resolves from any cargo invocation.
        let in_tree_path = std::path::Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("docs")
            .join("conformance")
            .join("cypher-tck-scorecard.md");
        let in_tree = std::fs::read_to_string(&in_tree_path).unwrap_or_else(|err| {
            panic!(
                "failed to read in-tree scorecard at {in_tree_path:?}: {err}. \
                 Regenerate via `cargo run --quiet -p arcgraph-tck --bin \
                 cypher-tck-scorecard > docs/conformance/cypher-tck-scorecard.md`"
            )
        });

        if in_tree != regenerated {
            // Print first 5 mismatching lines for fast localization on
            // CI log readers; full diff requires the suggested regen
            // command.
            let mut diff_lines = Vec::new();
            for (line_no, (a, b)) in in_tree.lines().zip(regenerated.lines()).enumerate() {
                if a != b {
                    diff_lines.push(format!("line {} in-tree:   {a}", line_no + 1));
                    diff_lines.push(format!("line {} regen:     {b}", line_no + 1));
                    if diff_lines.len() >= 10 {
                        break;
                    }
                }
            }
            panic!(
                "scorecard markdown drift detected — first mismatching lines:\n{}\n\
                 Regenerate via `cargo run --quiet -p arcgraph-tck --bin \
                 cypher-tck-scorecard > docs/conformance/cypher-tck-scorecard.md`",
                diff_lines.join("\n")
            );
        }
    }

    #[test]
    fn commitment_level_floor_clauses_call_is_oos() {
        // The `clauses/call` family must round-trip 100% to
        // NA-OutOfScope (the procedure-call surface is v1.1+).
        // If a future TCK refresh adds non-CALL scenarios under
        // `clauses/call/`, this floor catches the shape change.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let feature_root = std::path::Path::new(manifest_dir)
            .join("tck")
            .join("features");
        let summary = build_summary(&feature_root).expect("vendored tree walk failed");
        let call = summary
            .per_family
            .get("clauses/call")
            .copied()
            .unwrap_or_default();
        // 40 of 41 are NA-OOS; 1 was caught as NA-Write (CALL +
        // CREATE setup). That's the expected shape at the pin.
        assert!(
            call.eligible == 0,
            "clauses/call must not have any Eligible scenarios at v1.0-α \
             (procedure surface is v1.1+); got {} eligible",
            call.eligible,
        );
        assert!(
            call.total() >= 40,
            "clauses/call family must contain ≥ 40 scenarios; got {}",
            call.total(),
        );
    }
}
