#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# scripts/coverage_check.py — W28 Task #546 (Feature #529)
#
# Per-crate line-coverage gate (WARN-mode) parser for cargo-tarpaulin output.
#
# Parses a cargo-tarpaulin LCOV report (`cargo tarpaulin --out Lcov` →
# `lcov.info`), aggregates line coverage per workspace crate
# (`crates/<name>/src/**`), and flags crates that are either:
#   - below the 80% line-coverage target (`docs/testing-strategy.md` §4), or
#   - regressed > 1.0pp vs an optional baseline (the same section's ">1pp drop").
#
# WARN-mode: in CI this runs with `--warn-only` so it reports per-crate
# coverage + flags but NEVER blocks merge (`docs/testing-strategy.md` §6 lists the
# `coverage` job as "Informational"). The flip to a BLOCKING required check is
# a Director decision affecting ALL tracks — see
# `docs/coverage-gate.md` §"Flip-to-blocking (Director sign-off)".
#
# Why LCOV (not tarpaulin's `--out Json`):
#   The LCOV text format (SF/DA/LF/LH/end_of_record) is stable across
#   tarpaulin versions and trivially parseable; tarpaulin's JSON schema has
#   churned across releases. Per-line counts are derived from the `DA:`
#   records (the exact oracle — one DA per coverable line), not the `LF:`/`LH:`
#   summary, so the parser is independent of the summary's correctness.
#
# This module is intentionally stdlib-only and its parse / aggregate / flag
# functions are PURE so the parser logic is unit-tested WITHOUT running
# tarpaulin (see `scripts/test_coverage_check.py`). Per coverage workflow policy
# "do NOT run workspace tarpaulin locally" — CI runs tarpaulin; locally we
# only validate this parser against a captured sample LCOV fixture.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field

# Default thresholds — the canonical numbers from `docs/testing-strategy.md`
# §4. Overridable via CLI for calibration (the flip-to-blocking
# Director decision may ratchet `--min` per-crate; see docs/coverage-gate.md).
DEFAULT_MIN_PCT = 80.0
DEFAULT_MAX_DROP_PP = 1.0

# A workspace crate's *production* coverable surface is `crates/<name>/src/**`.
# Anchoring on `/src/` deliberately EXCLUDES `crates/<name>/tests/**`,
# `crates/<name>/benches/**`, and `examples/**` — coverage is measured on
# non-generated production code (code-quality policy), not on the tests themselves.
_CRATE_SRC_RE = re.compile(r"(?:^|/)crates/([^/]+)/src/")


@dataclass(frozen=True)
class FileCoverage:
    """Per-source-file line counts derived from one LCOV `SF:` block."""

    path: str
    lines_found: int  # coverable lines (count of DA records)
    lines_hit: int  # covered lines (DA records with hit count > 0)


@dataclass
class CrateCoverage:
    """Aggregated per-crate line coverage."""

    crate: str
    lines_found: int = 0
    lines_hit: int = 0

    @property
    def pct(self) -> float | None:
        """Line coverage %, or None when the crate has 0 coverable lines.

        A crate with no coverable production lines (e.g., a pure re-export
        facade) is reported as `n/a` and never flagged below-min — flagging
        it would be a false positive (there is nothing to cover)."""
        if self.lines_found == 0:
            return None
        return 100.0 * self.lines_hit / self.lines_found


@dataclass
class Flag:
    """A single warn-mode finding against one crate."""

    crate: str
    kind: str  # "below-min" | "regression"
    detail: str


@dataclass
class Report:
    crates: list[CrateCoverage]
    flags: list[Flag]
    baseline: dict[str, float] = field(default_factory=dict)
    min_pct: float = DEFAULT_MIN_PCT
    max_drop_pp: float = DEFAULT_MAX_DROP_PP


# ─────────────────────────────── parsing ────────────────────────────────────


def parse_lcov(text: str) -> list[FileCoverage]:
    """Parse LCOV text into per-file coverage.

    Robust to interleaved/unknown records and missing trailing
    `end_of_record`. Per-line counts come from `DA:<line>,<hits>` records
    (the exact oracle); `LF:`/`LH:` summary records are ignored.
    """
    files: list[FileCoverage] = []
    cur_path: str | None = None
    found = 0
    hit = 0

    def flush() -> None:
        nonlocal cur_path, found, hit
        if cur_path is not None:
            files.append(FileCoverage(cur_path, found, hit))
        cur_path = None
        found = 0
        hit = 0

    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("SF:"):
            # New source block — flush any prior block first (tolerates a
            # missing end_of_record between blocks).
            flush()
            cur_path = line[3:].strip()
            found = 0
            hit = 0
        elif line.startswith("DA:") and cur_path is not None:
            payload = line[3:]
            # DA:<line>,<hits>[,<checksum>]
            parts = payload.split(",")
            if len(parts) < 2:
                continue  # malformed DA — skip, do not crash
            try:
                hits = int(parts[1])
            except ValueError:
                continue
            found += 1
            if hits > 0:
                hit += 1
        elif line == "end_of_record":
            flush()
        # all other records (TN:, LF:, LH:, FN:, BRDA:, …) are ignored.

    flush()  # final block if file lacked a trailing end_of_record
    return files


def crate_of(path: str) -> str | None:
    """Return the workspace crate name owning `path`, or None if `path` is
    not production crate source (`crates/<name>/src/**`)."""
    m = _CRATE_SRC_RE.search(path)
    return m.group(1) if m else None


def aggregate_by_crate(files: list[FileCoverage]) -> list[CrateCoverage]:
    """Aggregate per-file coverage into per-crate coverage, sorted by crate
    name. Files outside `crates/<name>/src/**` are dropped."""
    acc: dict[str, CrateCoverage] = {}
    for fc in files:
        name = crate_of(fc.path)
        if name is None:
            continue
        cc = acc.setdefault(name, CrateCoverage(name))
        cc.lines_found += fc.lines_found
        cc.lines_hit += fc.lines_hit
    return [acc[k] for k in sorted(acc)]


# ─────────────────────────────── flags ──────────────────────────────────────


def compute_flags(
    crates: list[CrateCoverage],
    baseline: dict[str, float],
    min_pct: float = DEFAULT_MIN_PCT,
    max_drop_pp: float = DEFAULT_MAX_DROP_PP,
) -> list[Flag]:
    """Compute warn-mode flags: below-min and >max-drop regression.

    `baseline` maps crate -> baseline line-% (e.g., the value committed/cached
    from a clean main run). A crate absent from `baseline` cannot be regression-
    checked (no base) and is only subject to the below-min check.
    """
    flags: list[Flag] = []
    for cc in crates:
        pct = cc.pct
        if pct is None:
            continue  # 0 coverable lines — never flagged (see CrateCoverage.pct)
        if pct < min_pct:
            flags.append(
                Flag(
                    cc.crate,
                    "below-min",
                    f"{pct:.2f}% < {min_pct:.1f}% target "
                    f"({cc.lines_hit}/{cc.lines_found} lines)",
                )
            )
        base = baseline.get(cc.crate)
        if base is not None:
            drop = base - pct
            if drop > max_drop_pp:
                flags.append(
                    Flag(
                        cc.crate,
                        "regression",
                        f"{pct:.2f}% is {drop:.2f}pp below base {base:.2f}% "
                        f"(> {max_drop_pp:.1f}pp)",
                    )
                )
    return flags


def build_report(
    lcov_text: str,
    baseline: dict[str, float],
    min_pct: float = DEFAULT_MIN_PCT,
    max_drop_pp: float = DEFAULT_MAX_DROP_PP,
) -> Report:
    crates = aggregate_by_crate(parse_lcov(lcov_text))
    flags = compute_flags(crates, baseline, min_pct, max_drop_pp)
    return Report(crates, flags, baseline, min_pct, max_drop_pp)


# ─────────────────────────────── rendering ──────────────────────────────────


def _fmt_pct(pct: float | None) -> str:
    return "n/a" if pct is None else f"{pct:.2f}%"


def render_markdown(report: Report) -> str:
    """Render a per-crate coverage markdown table + flag summary."""
    out: list[str] = []
    out.append("### Per-crate line coverage (cargo-tarpaulin, WARN-mode)")
    out.append("")
    out.append(
        f"Target **{report.min_pct:.0f}%** line coverage; "
        f"flag regressions **> {report.max_drop_pp:.1f}pp** vs base "
        f"(testing-strategy §4). Informational — does **not** block merge "
        f"(testing-strategy §6; flip-to-blocking = Director sign-off, "
        f"`docs/coverage-gate.md`)."
    )
    out.append("")
    out.append("| Crate | Covered / Coverable | Coverage | Δ vs base | Flag |")
    out.append("|-------|--------------------:|---------:|----------:|------|")

    tot_found = 0
    tot_hit = 0
    flagged = {f.crate for f in report.flags}
    flag_kinds: dict[str, list[str]] = {}
    for f in report.flags:
        flag_kinds.setdefault(f.crate, []).append(f.kind)

    for cc in report.crates:
        tot_found += cc.lines_found
        tot_hit += cc.lines_hit
        pct = cc.pct
        base = report.baseline.get(cc.crate)
        if base is None or pct is None:
            delta = "—"
        else:
            d = pct - base
            delta = f"{d:+.2f}pp"
        if cc.crate in flagged:
            kinds = "/".join(sorted(set(flag_kinds[cc.crate])))
            mark = f"⚠️ {kinds}"
        else:
            mark = "✅"
        out.append(
            f"| `{cc.crate}` | {cc.lines_hit} / {cc.lines_found} "
            f"| {_fmt_pct(pct)} | {delta} | {mark} |"
        )

    tot_pct = None if tot_found == 0 else 100.0 * tot_hit / tot_found
    out.append(
        f"| **workspace total** | **{tot_hit} / {tot_found}** "
        f"| **{_fmt_pct(tot_pct)}** | — | — |"
    )
    out.append("")

    if report.flags:
        out.append(f"**{len(report.flags)} warn-mode flag(s):**")
        out.append("")
        for f in report.flags:
            out.append(f"- ⚠️ `{f.crate}` — {f.kind}: {f.detail}")
    else:
        out.append("**No warn-mode flags** — all crates ≥ target, no regressions.")
    out.append("")
    if not report.baseline:
        out.append(
            "> ℹ️ No baseline available — regression (>1pp) check skipped this "
            "run. The baseline self-populates from `main` runs (see "
            "`docs/coverage-gate.md`)."
        )
        out.append("")
    return "\n".join(out)


def emit_baseline(report: Report) -> dict[str, float]:
    """Build the {crate: pct} baseline map for the current run (crates with
    coverable lines only)."""
    return {cc.crate: cc.pct for cc in report.crates if cc.pct is not None}


# ─────────────────────────────── baseline IO ────────────────────────────────


def load_baseline(path: str) -> dict[str, float]:
    """Load a baseline from a JSON `{crate: pct}` file, or from an LCOV file
    (auto-detected). Returns {} on any read/parse failure (warn-mode: a
    missing/garbled baseline degrades to "no regression check", never an
    error)."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}
    text_stripped = text.lstrip()
    if text_stripped.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return {}
        return {str(k): float(v) for k, v in data.items() if _is_number(v)}
    # Fall back to treating it as an LCOV report.
    crates = aggregate_by_crate(parse_lcov(text))
    return emit_baseline(Report(crates, []))


def _is_number(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# ─────────────────────────────── CLI ────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="coverage_check.py",
        description="Per-crate tarpaulin line-coverage gate (WARN-mode).",
    )
    p.add_argument("--lcov", required=True, help="path to tarpaulin lcov.info")
    p.add_argument(
        "--baseline",
        default=None,
        help="optional baseline ({crate: pct} JSON or LCOV) for >1pp regression check",
    )
    p.add_argument("--min", type=float, default=DEFAULT_MIN_PCT, help="min line %% target")
    p.add_argument(
        "--max-drop",
        type=float,
        default=DEFAULT_MAX_DROP_PP,
        help="max allowed pp drop vs base before flagging",
    )
    p.add_argument(
        "--summary",
        default=None,
        help="append the markdown table to this file (e.g. $GITHUB_STEP_SUMMARY)",
    )
    p.add_argument(
        "--emit-baseline",
        default=None,
        help="write the current run's {crate: pct} JSON baseline to this path",
    )
    p.add_argument("--json", action="store_true", help="print machine-readable JSON to stdout")
    p.add_argument(
        "--warn-only",
        action="store_true",
        help="always exit 0 even when flags are present (CI warn-mode)",
    )
    args = p.parse_args(argv)

    try:
        with open(args.lcov, encoding="utf-8") as fh:
            lcov_text = fh.read()
    except OSError as exc:
        # Warn-mode: a missing lcov (e.g., tarpaulin crashed) must not hard-fail
        # the informational job. Report and degrade.
        msg = f"coverage_check: could not read lcov '{args.lcov}': {exc}"
        print(msg, file=sys.stderr)
        if args.summary:
            _append(args.summary, f"### Per-crate line coverage\n\n> ⚠️ {msg}\n")
        return 0 if args.warn_only else 2

    baseline = load_baseline(args.baseline) if args.baseline else {}
    report = build_report(lcov_text, baseline, args.min, args.max_drop)

    md = render_markdown(report)
    print(md)
    if args.summary:
        _append(args.summary, md + "\n")
    if args.emit_baseline:
        with open(args.emit_baseline, "w", encoding="utf-8") as fh:
            json.dump(emit_baseline(report), fh, indent=2, sort_keys=True)
            fh.write("\n")
    if args.json:
        payload = {
            "min_pct": report.min_pct,
            "max_drop_pp": report.max_drop_pp,
            "crates": [
                {
                    "crate": cc.crate,
                    "lines_found": cc.lines_found,
                    "lines_hit": cc.lines_hit,
                    "pct": cc.pct,
                }
                for cc in report.crates
            ],
            "flags": [
                {"crate": f.crate, "kind": f.kind, "detail": f.detail}
                for f in report.flags
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))

    if report.flags and not args.warn_only:
        return 1
    return 0


def _append(path: str, text: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(text)


if __name__ == "__main__":
    raise SystemExit(main())
