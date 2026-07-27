#!/usr/bin/env bash
set -euo pipefail

readonly GATE_PREFIX="agent-quickstart-gate"
CURRENT_STEP="locate repository"
TEMP_ROOT=""
DRIVER_PID=""

fail() {
  printf '%s: FAIL [%s] %s\n' "$GATE_PREFIX" "$CURRENT_STEP" "$*" >&2
  exit 1
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM HUP

  if [[ -n "$DRIVER_PID" ]] && kill -0 "$DRIVER_PID" 2>/dev/null; then
    kill -TERM "$DRIVER_PID" 2>/dev/null || true
    wait "$DRIVER_PID" 2>/dev/null || true
  fi

  if [[ -n "$TEMP_ROOT" && -d "$TEMP_ROOT" ]]; then
    case "$TEMP_ROOT" in
      "${TMPDIR:-/tmp}"/arcgraph-agent-gate.*)
        rm -rf -- "$TEMP_ROOT"
        ;;
      *)
        printf '%s: refusing to remove unexpected temp path %s\n' \
          "$GATE_PREFIX" "$TEMP_ROOT" >&2
        status=1
        ;;
    esac
  fi

  exit "$status"
}

unexpected_error() {
  local status=$?
  local line=$1
  printf '%s: FAIL [%s] unexpected command failure at line %s (exit %s)\n' \
    "$GATE_PREFIX" "$CURRENT_STEP" "$line" "$status" >&2
  exit "$status"
}

trap cleanup EXIT INT TERM HUP
trap 'unexpected_error "$LINENO"' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)" \
  || fail "expected this script to run from a Git checkout"
readonly REPO_ROOT

CURRENT_STEP="verify documented commands"
grep -Fq 'https://github.com/aleflabs/arcgraph' "$REPO_ROOT/README.md" \
  || fail 'README must identify the public repository'
grep -Fqx 'cargo build --workspace' "$REPO_ROOT/README.md" \
  || fail 'README must document `cargo build --workspace`'
grep -Fqx 'python3 scripts/agent-quickstart.py --bin target/debug/arcgraph' \
  "$REPO_ROOT/README.md" \
  || fail 'README has no executable agent walkthrough; expected `python3 scripts/agent-quickstart.py --bin target/debug/arcgraph`'
grep -Fq '## Quickstart' "$REPO_ROOT/README.md" \
  || fail 'README must contain a Quickstart section'
grep -Fq '## Using ArcGraph from an agent' "$REPO_ROOT/README.md" \
  || fail 'README must contain a "Using ArcGraph from an agent" section'

documented_bash_commands="$(
  awk '
    /^```bash$/ { in_bash = 1; next }
    in_bash && /^```$/ { in_bash = 0; next }
    in_bash { print }
  ' "$REPO_ROOT/README.md"
)"
expected_bash_commands="$(
  printf '%s\n' \
    'cargo build --workspace' \
    'python3 scripts/agent-quickstart.py --bin target/debug/arcgraph'
)"
[[ "$documented_bash_commands" == "$expected_bash_commands" ]] \
  || fail "expected every README bash command to be gate-executed; found: $documented_bash_commands"
printf '%s: README commands verified\n' "$GATE_PREFIX"

CURRENT_STEP="check prerequisites"
for command_name in git tar curl cargo rustc python3; do
  command -v "$command_name" >/dev/null 2>&1 \
    || fail "expected documented prerequisite \`$command_name\` on PATH"
done

TEMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/arcgraph-agent-gate.XXXXXX")" \
  || fail "could not create isolated temp directory"
readonly TEMP_ROOT
readonly CLEAN_TREE="$TEMP_ROOT/arcgraph"
mkdir -p "$CLEAN_TREE"

CURRENT_STEP="materialize clean published tree"
git -C "$REPO_ROOT" archive --format=tar HEAD \
  | env LC_ALL=C tar -xf - -C "$CLEAN_TREE"
[[ ! -e "$CLEAN_TREE/target" ]] \
  || fail "expected git archive HEAD to exclude build artifacts; found target/"
[[ -x "$CLEAN_TREE/scripts/agent-quickstart-gate.sh" ]] \
  || fail "expected scripts/agent-quickstart-gate.sh to be published as executable"
[[ -x "$CLEAN_TREE/scripts/agent-quickstart.py" ]] \
  || fail "expected scripts/agent-quickstart.py to be published as executable"
[[ -x "$CLEAN_TREE/scripts/bolt-quickstart.py" ]] \
  || fail "expected scripts/bolt-quickstart.py to be published as executable"
printf '%s: CLEAN_COPY git archive HEAD\n' "$GATE_PREFIX"

cd "$CLEAN_TREE"

CURRENT_STEP="build documented workspace"
readonly BUILD_LOG="$TEMP_ROOT/cargo-build.log"
if ! cargo build --workspace >"$BUILD_LOG" 2>&1; then
  printf '%s: cargo build output follows\n' "$GATE_PREFIX" >&2
  sed -n '1,240p' "$BUILD_LOG" >&2
  fail 'expected `cargo build --workspace` to succeed in the clean copy'
fi
printf '%s: BUILD workspace default-features\n' "$GATE_PREFIX"

CURRENT_STEP="run documented durable MCP walkthrough"
readonly DRIVER_LOG="$TEMP_ROOT/agent-quickstart.log"
python3 scripts/agent-quickstart.py --bin target/debug/arcgraph \
  >"$DRIVER_LOG" 2>&1 &
DRIVER_PID=$!
if wait "$DRIVER_PID"; then
  DRIVER_PID=""
else
  driver_status=$?
  DRIVER_PID=""
  sed -n '1,260p' "$DRIVER_LOG" >&2
  fail "expected the documented walkthrough to pass; driver exited $driver_status"
fi
sed -n '1,260p' "$DRIVER_LOG"

CURRENT_STEP="compare documented output"
readonly README_OUTPUT="$TEMP_ROOT/readme-expected-output.log"
awk '
  /^Expected output:$/ { found = 1; next }
  found && /^```text$/ { capture = 1; next }
  capture && /^```$/ { exit }
  capture { print }
' README.md >"$README_OUTPUT"
if ! diff -u "$README_OUTPUT" "$DRIVER_LOG"; then
  fail 'README expected output differs from the walkthrough output'
fi

actual_request="$(sed -n 's/^agent-quickstart: MCP_REQUEST //p' "$DRIVER_LOG")"
documented_request="$(
  awk '
    /^inspection call is:$/ { found = 1; next }
    found && /^```json$/ { capture = 1; next }
    capture && /^```$/ { exit }
    capture { print }
  ' README.md
)"
[[ "$documented_request" == "$actual_request" ]] \
  || fail 'README MCP request differs from the request executed by the walkthrough'

actual_response="$(sed -n 's/^agent-quickstart: MCP_RESPONSE //p' "$DRIVER_LOG")"
documented_response="$(
  awk '
    /^Its real response in the quickstart is:$/ { found = 1; next }
    found && /^```json$/ { capture = 1; next }
    capture && /^```$/ { exit }
    capture { print }
  ' README.md
)"
[[ "$documented_response" == "$actual_response" ]] \
  || fail 'README MCP response differs from the response returned by the walkthrough'

CURRENT_STEP="final assertions"
grep -Fqx 'agent-quickstart: PASS all values survived restart' "$DRIVER_LOG" \
  || fail 'expected the walkthrough durability PASS sentinel'
printf '%s: PASS clean-copy agent walkthrough\n' "$GATE_PREFIX"
