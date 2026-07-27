#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/refresh-docker-pins.sh — W19β #326
#
# Refresh the SHA-256 digest pins in every Dockerfile under the repo.
# Pin bumps are a deliberate human action (never wired to CI); this
# script just automates the digest-fetch + diff-emission so the
# reviewer's job is verifying the bump intent, not chasing registry
# manifest formats.
#
# Why this exists:
#   `FROM rust:1.85-slim-bookworm` is mutable — Docker Hub can
#   re-publish that tag with different bytes after a security patch
#   or a rebuild. SHA-pinning to `<tag>@sha256:<digest>` freezes the
#   build inputs; this script is the supported way to thaw + re-pin.
#
# Usage:
#   scripts/refresh-docker-pins.sh             # write new pins inline, leave diff
#   scripts/refresh-docker-pins.sh --check     # exit non-zero if any pin is stale
#   scripts/refresh-docker-pins.sh --dry-run   # print proposed pins, no edits
#
# Dependencies: curl, python3, sed. No docker / skopeo / crane required.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# Resolve repo root assuming this script lives in scripts/.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd)"

CHECK_MODE=0
DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --check)   CHECK_MODE=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --help|-h)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "error: unknown arg: $arg" >&2
            echo "usage: $0 [--check|--dry-run]" >&2
            exit 2
            ;;
    esac
done

# Pin table: parallel arrays indexed by image.
#
# IMAGES[i]  — registry-qualified image (registry/repository:tag)
# DIGESTS[i] — filled in by `fetch_digest` below
IMAGES=(
    "docker.io/library/rust:1.85-slim-bookworm"
    "docker.io/library/rust:1.85-bookworm"
    "docker.io/library/debian:bookworm-slim"
    "docker.io/library/python:3.12-slim"
    "gcr.io/distroless/cc-debian12:nonroot"
)

# Sister tag stripped of "docker.io/library/" namespace prefix — that
# prefix only matters at the registry-API level, NOT inside Dockerfiles.
DOCKERFILE_TAGS=(
    "rust:1.85-slim-bookworm"
    "rust:1.85-bookworm"
    "debian:bookworm-slim"
    "python:3.12-slim"
    "gcr.io/distroless/cc-debian12:nonroot"
)

DIGESTS=()

fetch_digest() {
    local image="$1"
    local registry repo tag
    # Split "<registry>/<repo>:<tag>" into pieces.
    if [[ "$image" == *"@"* ]]; then
        echo "error: pass tag, not digest: $image" >&2
        return 1
    fi
    # The first '/' separates registry from path; everything to the
    # right of the last ':' is the tag.
    registry="${image%%/*}"
    tag="${image##*:}"
    repo="${image#*/}"
    repo="${repo%:*}"

    local auth_arg=()
    local manifest_url

    if [ "$registry" = "docker.io" ]; then
        # Docker Hub requires anonymous-Bearer-token fetch.
        local token
        token=$(curl -fsSL \
                 "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${repo}:pull" \
                 | python3 -c "import sys, json; print(json.load(sys.stdin)['token'])")
        auth_arg=(-H "Authorization: Bearer ${token}")
        manifest_url="https://registry-1.docker.io/v2/${repo}/manifests/${tag}"
    else
        # gcr.io public images don't require auth.
        manifest_url="https://${registry}/v2/${repo}/manifests/${tag}"
    fi

    # Accept both Docker manifest-list v2 + OCI image-index v1.
    # `"${auth_arg[@]+...}"` expands to NOTHING when auth_arg is empty
    # (e.g. for gcr.io anonymous reads) and to the array contents
    # otherwise — set -u-safe.
    local digest
    digest=$(curl -fsSL -I \
             ${auth_arg[@]+"${auth_arg[@]}"} \
             -H "Accept: application/vnd.docker.distribution.manifest.list.v2+json" \
             -H "Accept: application/vnd.oci.image.index.v1+json" \
             "${manifest_url}" \
             | grep -i '^docker-content-digest:' \
             | awk '{print $2}' \
             | tr -d '\r')
    if [ -z "$digest" ]; then
        echo "error: no digest returned for $image" >&2
        return 1
    fi
    echo "$digest"
}

echo "Refreshing Docker FROM pins …"
for i in "${!IMAGES[@]}"; do
    img="${IMAGES[$i]}"
    digest=$(fetch_digest "$img")
    DIGESTS[$i]="$digest"
    printf "  %-50s → %s\n" "${DOCKERFILE_TAGS[$i]}" "$digest"
done

# Find every Dockerfile + sister build script under repo root.
# Use a while-read loop instead of `mapfile -t` for bash-3.2 compat
# (macOS ships bash 3.2; CI runs on bash 5.x).
DOCKERFILES=()
while IFS= read -r line; do
    DOCKERFILES+=("$line")
done < <(find "$REPO_ROOT" \
    -type f \
    \( -name 'Dockerfile' -o -name 'Dockerfile.*' \) \
    -not -path '*/target/*' \
    -not -path '*/.git/*' \
    -not -path '*/node_modules/*' \
    | sort)

echo ""
echo "Scanning ${#DOCKERFILES[@]} Dockerfile(s) …"

ANY_DRIFT=0
for dockerfile in "${DOCKERFILES[@]}"; do
    rel="${dockerfile#${REPO_ROOT}/}"
    file_drift=0
    for i in "${!DOCKERFILE_TAGS[@]}"; do
        tag="${DOCKERFILE_TAGS[$i]}"
        digest="${DIGESTS[$i]}"
        # Match existing `FROM <tag>` or `FROM <tag>@sha256:<old>` lines.
        if grep -E "^FROM ${tag}(@sha256:[0-9a-f]{64})?( AS .*)?$" "$dockerfile" >/dev/null 2>&1; then
            current=$(grep -oE "${tag}@sha256:[0-9a-f]{64}" "$dockerfile" | head -1 || true)
            want="${tag}@${digest}"
            if [ "$current" != "$want" ]; then
                file_drift=1
                ANY_DRIFT=1
                echo "  drift: ${rel}: ${tag}"
                echo "    have: ${current:-<unpinned>}"
                echo "    want: ${want}"
                if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_MODE" -eq 0 ]; then
                    # In-place rewrite, preserving the optional ` AS <stage>` suffix.
                    # Use python instead of sed for portable handling of `@`.
                    python3 - "$dockerfile" "$tag" "$digest" <<'PY'
import re, sys
path, tag, digest = sys.argv[1], sys.argv[2], sys.argv[3]
with open(path) as f:
    src = f.read()
pattern = re.compile(
    r'^(FROM\s+)' + re.escape(tag) + r'(@sha256:[0-9a-f]{64})?(\s+AS\s+\S+)?\s*$',
    re.MULTILINE,
)
new = pattern.sub(lambda m: f"{m.group(1)}{tag}@{digest}{m.group(3) or ''}", src)
if new != src:
    with open(path, 'w') as f:
        f.write(new)
    print(f"    wrote: {path}")
PY
                fi
            fi
        fi
    done
    if [ "$file_drift" -eq 0 ]; then
        echo "  ok:    ${rel}"
    fi
done

if [ "$CHECK_MODE" -eq 1 ] && [ "$ANY_DRIFT" -eq 1 ]; then
    echo ""
    echo "FAIL: pin drift detected. Run scripts/refresh-docker-pins.sh to refresh." >&2
    exit 1
fi

echo ""
echo "Done. Review the diff with: git diff Dockerfile docs/demo/Dockerfile"
