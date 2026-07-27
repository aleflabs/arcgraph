#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# scripts/test_coverage_check.py — W28 Task #546 (Feature #529)
#
# Unit tests for scripts/coverage_check.py — the per-crate tarpaulin
# coverage-gate PARSER. Pure + quick: runs WITHOUT cargo-tarpaulin; the fixture below
# is a captured-shape cargo-tarpaulin LCOV report with adversarial edge cases.
#
# Run:  python3 -m unittest scripts.test_coverage_check    (from repo root)
#   or: python3 scripts/test_coverage_check.py
#
# Oracle discipline (ENGINEERING_DOCTRINE §3 "strong oracles"): assertions use
# exact `==` on computed counts / percentages / flag sets — not `>=` slack —
# because tarpaulin LCOV counts and the derived percentages are deterministic.
# ─────────────────────────────────────────────────────────────────────────────

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import coverage_check as cc  # noqa: E402


# Captured-shape tarpaulin LCOV fixture. Deterministic line counts so the
# expected percentages are exact. Edge cases are intentional:
#   - arcgraph-core/src: two source files in one crate (must sum)
#       lib.rs   DA: 4 lines, 3 hit
#       cache.rs DA: 6 lines, 5 hit   → crate total 10 found / 8 hit = 80.00%
#   - arcgraph-storage/src: 10 found / 7 hit = 70.00%  → BELOW 80% min
#   - arcgraph-query/src: 16 found / 16 hit = 100.00%
#   - arcgraph-core/tests/it.rs: under crates/<n>/ but NOT /src/ → EXCLUDED
#   - /home/runner/work/arcgraph/build.rs: outside crates/ entirely → EXCLUDED
#   - crates/arcgraph-empty/src/lib.rs: 0 DA records → 0 coverable → pct n/a
#   - a malformed DA line ("DA:bad") must be skipped, not crash
#   - a block with a missing end_of_record (the final one) must still flush
SAMPLE_LCOV = """\
TN:
SF:/home/runner/work/arcgraph/arcgraph/crates/arcgraph-core/src/lib.rs
DA:1,1
DA:2,1
DA:3,1
DA:7,0
LF:4
LH:3
end_of_record
TN:
SF:/home/runner/work/arcgraph/arcgraph/crates/arcgraph-core/src/cache.rs
DA:10,5
DA:11,2
DA:12,0
DA:13,9
DA:14,1
DA:15,4
end_of_record
SF:/home/runner/work/arcgraph/arcgraph/crates/arcgraph-storage/src/wal.rs
DA:1,3
DA:2,3
DA:3,0
DA:4,0
DA:5,0
DA:6,1
DA:7,1
DA:8,1
DA:9,1
DA:bad
DA:10,1
end_of_record
SF:/home/runner/work/arcgraph/arcgraph/crates/arcgraph-core/tests/it.rs
DA:1,1
DA:2,1
end_of_record
SF:/home/runner/work/arcgraph/arcgraph/build.rs
DA:1,1
DA:2,1
end_of_record
SF:/home/runner/work/arcgraph/arcgraph/crates/arcgraph-empty/src/lib.rs
end_of_record
SF:/home/runner/work/arcgraph/arcgraph/crates/arcgraph-query/src/planner.rs
DA:1,1
DA:2,1
DA:3,1
DA:4,1
DA:5,1
DA:6,1
DA:7,1
DA:8,1
DA:9,1
DA:10,1
DA:11,1
DA:12,1
DA:13,1
DA:14,1
DA:15,1
DA:16,1
"""


def _by_suffix(files, suffix):
    """Find the single FileCoverage whose path ends with `suffix`. Used to
    disambiguate basenames that collide (e.g., two `src/lib.rs`)."""
    matches = [f for f in files if f.path.endswith(suffix)]
    assert len(matches) == 1, f"expected exactly 1 match for {suffix}, got {len(matches)}"
    return matches[0]


class TestParseLcov(unittest.TestCase):
    def test_per_file_counts(self):
        files = cc.parse_lcov(SAMPLE_LCOV)
        core_lib = _by_suffix(files, "arcgraph-core/src/lib.rs")
        self.assertEqual((core_lib.lines_found, core_lib.lines_hit), (4, 3))
        cache = _by_suffix(files, "arcgraph-core/src/cache.rs")
        self.assertEqual((cache.lines_found, cache.lines_hit), (6, 5))
        # wal.rs: 10 valid DA records (the "DA:bad" line skipped); hits on
        # lines 1,2,6,7,8,9,10 → 7 hit.
        wal = _by_suffix(files, "arcgraph-storage/src/wal.rs")
        self.assertEqual((wal.lines_found, wal.lines_hit), (10, 7))

    def test_malformed_da_skipped_not_crashed(self):
        # The "DA:bad" line is present in wal.rs; parse must not raise.
        files = cc.parse_lcov(SAMPLE_LCOV)
        wal = next(f for f in files if f.path.endswith("wal.rs"))
        self.assertEqual(wal.lines_found, 10)

    def test_missing_trailing_end_of_record_flushed(self):
        # planner.rs is the last block and has no end_of_record — must flush.
        files = {f.path.rsplit("/", 1)[-1]: f for f in cc.parse_lcov(SAMPLE_LCOV)}
        self.assertIn("planner.rs", files)
        self.assertEqual(files["planner.rs"].lines_found, 16)

    def test_empty_input(self):
        self.assertEqual(cc.parse_lcov(""), [])


class TestCrateOf(unittest.TestCase):
    def test_src_path_maps_to_crate(self):
        self.assertEqual(
            cc.crate_of("/a/b/crates/arcgraph-core/src/lib.rs"), "arcgraph-core"
        )

    def test_tests_dir_excluded(self):
        self.assertIsNone(cc.crate_of("/a/b/crates/arcgraph-core/tests/it.rs"))

    def test_benches_dir_excluded(self):
        self.assertIsNone(cc.crate_of("/a/b/crates/arcgraph-core/benches/x.rs"))

    def test_non_crate_path_excluded(self):
        self.assertIsNone(cc.crate_of("/a/b/build.rs"))
        self.assertIsNone(cc.crate_of("/a/b/examples/demo.rs"))


class TestAggregate(unittest.TestCase):
    def setUp(self):
        self.crates = {c.crate: c for c in cc.aggregate_by_crate(cc.parse_lcov(SAMPLE_LCOV))}

    def test_core_sums_two_files(self):
        core = self.crates["arcgraph-core"]
        self.assertEqual((core.lines_found, core.lines_hit), (10, 8))
        self.assertEqual(core.pct, 80.0)  # exact

    def test_storage_below_min(self):
        st = self.crates["arcgraph-storage"]
        self.assertEqual((st.lines_found, st.lines_hit), (10, 7))
        self.assertEqual(st.pct, 70.0)

    def test_query_full(self):
        q = self.crates["arcgraph-query"]
        self.assertEqual((q.lines_found, q.lines_hit), (16, 16))
        self.assertEqual(q.pct, 100.0)

    def test_empty_crate_is_na(self):
        e = self.crates["arcgraph-empty"]
        self.assertEqual(e.lines_found, 0)
        self.assertIsNone(e.pct)

    def test_excluded_paths_not_present(self):
        # tests/it.rs and build.rs must not create crate buckets.
        self.assertNotIn("build.rs", self.crates)
        # arcgraph-core exists but its tests/it.rs (2 lines) must NOT be summed:
        self.assertEqual(self.crates["arcgraph-core"].lines_found, 10)

    def test_sorted_by_name(self):
        names = [c.crate for c in cc.aggregate_by_crate(cc.parse_lcov(SAMPLE_LCOV))]
        self.assertEqual(names, sorted(names))


class TestFlags(unittest.TestCase):
    def setUp(self):
        self.crates = cc.aggregate_by_crate(cc.parse_lcov(SAMPLE_LCOV))

    def test_below_min_flagged(self):
        flags = cc.compute_flags(self.crates, baseline={}, min_pct=80.0, max_drop_pp=1.0)
        below = {f.crate for f in flags if f.kind == "below-min"}
        # storage 70% and core 80% — core is exactly at min (not below); only
        # storage flagged. Strong oracle: exact set.
        self.assertEqual(below, {"arcgraph-storage"})

    def test_exact_min_not_flagged(self):
        # arcgraph-core at exactly 80.0% must NOT be flagged (>= passes).
        flags = cc.compute_flags(self.crates, baseline={}, min_pct=80.0)
        self.assertNotIn("arcgraph-core", {f.crate for f in flags})

    def test_empty_crate_never_flagged(self):
        flags = cc.compute_flags(self.crates, baseline={}, min_pct=80.0)
        self.assertNotIn("arcgraph-empty", {f.crate for f in flags})

    def test_regression_flagged_beyond_threshold(self):
        # core dropped 80.0 from base 81.5 → 1.5pp drop > 1.0pp → regression.
        baseline = {"arcgraph-core": 81.5, "arcgraph-query": 100.0}
        flags = cc.compute_flags(self.crates, baseline=baseline, min_pct=80.0, max_drop_pp=1.0)
        regr = {f.crate for f in flags if f.kind == "regression"}
        self.assertEqual(regr, {"arcgraph-core"})

    def test_regression_at_threshold_not_flagged(self):
        # Exactly 1.0pp drop is NOT > 1.0pp → not flagged (boundary).
        baseline = {"arcgraph-core": 81.0}  # 81.0 - 80.0 = 1.0pp exactly
        flags = cc.compute_flags(self.crates, baseline=baseline, max_drop_pp=1.0)
        self.assertEqual([f for f in flags if f.kind == "regression"], [])

    def test_improvement_not_flagged_as_regression(self):
        baseline = {"arcgraph-query": 50.0}  # improved to 100 → no regression
        flags = cc.compute_flags(self.crates, baseline=baseline)
        self.assertEqual([f for f in flags if f.kind == "regression"], [])

    def test_missing_baseline_skips_regression(self):
        flags = cc.compute_flags(self.crates, baseline={}, max_drop_pp=1.0)
        self.assertEqual([f for f in flags if f.kind == "regression"], [])


class TestBaselineIO(unittest.TestCase):
    def test_load_json_baseline(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"arcgraph-core": 81.5, "arcgraph-storage": 70.0}, fh)
            path = fh.name
        try:
            base = cc.load_baseline(path)
            self.assertEqual(base, {"arcgraph-core": 81.5, "arcgraph-storage": 70.0})
        finally:
            os.unlink(path)

    def test_load_lcov_baseline(self):
        with tempfile.NamedTemporaryFile("w", suffix=".info", delete=False) as fh:
            fh.write(SAMPLE_LCOV)
            path = fh.name
        try:
            base = cc.load_baseline(path)
            self.assertEqual(base["arcgraph-core"], 80.0)
            self.assertEqual(base["arcgraph-query"], 100.0)
            self.assertNotIn("arcgraph-empty", base)  # n/a excluded from baseline
        finally:
            os.unlink(path)

    def test_missing_baseline_returns_empty(self):
        self.assertEqual(cc.load_baseline("/no/such/baseline.json"), {})

    def test_garbled_json_returns_empty(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            fh.write("{not valid json")
            path = fh.name
        try:
            self.assertEqual(cc.load_baseline(path), {})
        finally:
            os.unlink(path)

    def test_emit_baseline_excludes_na(self):
        report = cc.build_report(SAMPLE_LCOV, baseline={})
        emitted = cc.emit_baseline(report)
        self.assertEqual(emitted["arcgraph-core"], 80.0)
        self.assertNotIn("arcgraph-empty", emitted)


class TestRender(unittest.TestCase):
    def test_table_contains_all_crates_and_total(self):
        report = cc.build_report(SAMPLE_LCOV, baseline={"arcgraph-core": 81.5})
        md = cc.render_markdown(report)
        self.assertIn("`arcgraph-core`", md)
        self.assertIn("`arcgraph-storage`", md)
        self.assertIn("`arcgraph-query`", md)
        self.assertIn("workspace total", md)
        # workspace total = (10+10+16+0) found, (8+7+16+0) hit = 31/36
        self.assertIn("31 / 36", md)
        # no-baseline notice absent when a baseline is present
        self.assertNotIn("No baseline available", md)

    def test_no_baseline_notice_present(self):
        report = cc.build_report(SAMPLE_LCOV, baseline={})
        md = cc.render_markdown(report)
        self.assertIn("No baseline available", md)


class TestMainExitCodes(unittest.TestCase):
    """The parser's exit code is the unit-tested contract that the CI workflow
    consumes in --warn-only mode. 0 = clean, 1 = flags present, 2 = error."""

    def _write(self, text, suffix=".info"):
        fh = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False)
        fh.write(text)
        fh.close()
        self.addCleanup(os.unlink, fh.name)
        return fh.name

    def test_exit_1_when_flags_present(self):
        lcov = self._write(SAMPLE_LCOV)
        rc = cc.main(["--lcov", lcov])  # storage 70% < 80% → flag
        self.assertEqual(rc, 1)

    def test_exit_0_with_warn_only_despite_flags(self):
        lcov = self._write(SAMPLE_LCOV)
        rc = cc.main(["--lcov", lcov, "--warn-only"])
        self.assertEqual(rc, 0)

    def test_exit_0_when_clean(self):
        # All crates at/above min, no baseline regression.
        clean = (
            "SF:/x/crates/arcgraph-query/src/a.rs\n"
            "DA:1,1\nDA:2,1\nDA:3,1\nDA:4,1\nDA:5,1\nend_of_record\n"
        )
        lcov = self._write(clean)
        rc = cc.main(["--lcov", lcov, "--min", "80"])
        self.assertEqual(rc, 0)

    def test_exit_2_when_lcov_missing(self):
        rc = cc.main(["--lcov", "/no/such/lcov.info"])
        self.assertEqual(rc, 2)

    def test_exit_0_when_lcov_missing_warn_only(self):
        rc = cc.main(["--lcov", "/no/such/lcov.info", "--warn-only"])
        self.assertEqual(rc, 0)

    def test_emit_baseline_and_summary_written(self):
        lcov = self._write(SAMPLE_LCOV)
        base_out = self._write("", suffix=".json")
        summ = self._write("", suffix=".md")
        rc = cc.main(
            ["--lcov", lcov, "--warn-only", "--emit-baseline", base_out, "--summary", summ]
        )
        self.assertEqual(rc, 0)
        with open(base_out) as fh:
            emitted = json.load(fh)
        self.assertEqual(emitted["arcgraph-core"], 80.0)
        with open(summ) as fh:
            self.assertIn("Per-crate line coverage", fh.read())


class TestCommittedFixture(unittest.TestCase):
    """The committed `scripts/testdata/sample_tarpaulin_lcov.info` is the
    "captured sample tarpaulin output" the CI self-test step runs against. It
    MUST stay byte-identical to the inline SAMPLE_LCOV so the two cannot drift
    (hard assert — never skip; soft-skip is the worst bug class per
    feedback_test_env_gate_panic_by_default.md)."""

    def test_committed_fixture_matches_inline(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "testdata",
            "sample_tarpaulin_lcov.info",
        )
        self.assertTrue(os.path.exists(path), f"committed fixture missing: {path}")
        with open(path, encoding="utf-8") as fh:
            self.assertEqual(fh.read(), SAMPLE_LCOV)


if __name__ == "__main__":
    unittest.main(verbosity=2)
