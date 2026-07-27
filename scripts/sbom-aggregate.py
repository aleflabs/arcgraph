#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# scripts/sbom-aggregate.py — W19β #323
#
# Walks the current directory (skipping `target/` + `.git/`) for
# cargo-cyclonedx-emitted `*.cdx.json` (or `bom.json`) per-crate
# BOMs and merges them into ONE workspace-scope CycloneDX 1.5 BOM
# whose `metadata.component` names the workspace as a whole.
#
# Why this exists:
#   cargo-cyclonedx ≥ 0.5 emits one BOM per workspace member, not one
#   workspace-aggregate BOM. The release SBOM consumer story wants a
#   single signed artifact for `cosign verify-blob`, so this script
#   does the merge deterministically (sorted-by-(name, version)) so
#   re-runs produce identical bytes.
#
# Usage:
#   python3 scripts/sbom-aggregate.py --version v0.1.0-alpha.1 \
#       --output arcgraph-v0.1.0-alpha.1.cdx.json
#
# Per Prime Directive #6: this aggregator is the single source of
# truth for the workspace SBOM emission contract — it is referenced
# from `.github/workflows/release.yml` and exercisable locally so
# operators can `git diff` the emitted SBOM offline before releasing.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CYCLONEDX_SPEC_VERSION = "1.5"


def find_per_crate_boms(root: Path) -> list[Path]:
    """Return every *.cdx.json / bom.json under root, excluding target/ + .git/."""
    candidates: list[Path] = []
    for pattern in ("*.cdx.json", "bom.json"):
        for path in root.rglob(pattern):
            # Skip build artifacts and VCS internals.
            parts = path.parts
            if "target" in parts or ".git" in parts:
                continue
            candidates.append(path)
    return sorted(candidates)


def merge(boms: list[Path], workspace_name: str, version: str) -> dict:
    components: list[dict] = []
    seen: set[tuple[str, str]] = set()
    dependencies: list[dict] = []
    dep_seen: set[str] = set()

    for path in boms:
        try:
            with path.open() as f:
                bom = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"warn: skipping {path}: {e}", file=sys.stderr)
            continue

        for comp in bom.get("components", []) or []:
            key = (comp.get("name", ""), comp.get("version", ""))
            if key in seen or key == ("", ""):
                continue
            seen.add(key)
            components.append(comp)

        for dep in bom.get("dependencies", []) or []:
            ref = dep.get("ref", "")
            if ref in dep_seen or not ref:
                continue
            dep_seen.add(ref)
            dependencies.append(dep)

    components.sort(key=lambda c: (c.get("name", ""), c.get("version", "")))
    dependencies.sort(key=lambda d: d.get("ref", ""))

    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": workspace_name,
                "version": version,
            },
            "tools": [
                {
                    "vendor": "ArcGraph",
                    "name": "sbom-aggregate.py",
                    "version": "1.0.0",
                }
            ],
        },
        "components": components,
        "dependencies": dependencies,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        required=True,
        help="Workspace version (e.g. v0.1.0-alpha.1).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the aggregated SBOM.",
    )
    parser.add_argument(
        "--workspace-name",
        default="arcgraph",
        help="Workspace component name (default: arcgraph).",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Walk-root for per-crate BOMs (default: .).",
    )
    args = parser.parse_args(argv)

    boms = find_per_crate_boms(Path(args.root))
    if not boms:
        print(
            "error: no per-crate *.cdx.json / bom.json found; "
            "did `cargo cyclonedx --all` run first?",
            file=sys.stderr,
        )
        return 1

    aggregated = merge(boms, args.workspace_name, args.version)
    Path(args.output).write_text(json.dumps(aggregated, indent=2, sort_keys=True))
    print(
        f"wrote {args.output} ({len(aggregated['components'])} components, "
        f"{len(aggregated['dependencies'])} dependencies, "
        f"from {len(boms)} per-crate BOMs)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
