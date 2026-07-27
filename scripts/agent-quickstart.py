#!/usr/bin/env python3
"""Drive the public ArcGraph MCP quickstart and assert every returned value."""

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import time


PROTOCOL_VERSION = "2025-06-18"
RESPONSE_TIMEOUT_SECONDS = 30
SHUTDOWN_TIMEOUT_SECONDS = 30
TOOLS = [
    "graph.schema",
    "graph.inspect",
    "graph.explore",
    "graph.search",
    "graph.ingest",
    "graph.raw_query",
]
ARCQL = (
    "MATCH (a:Person)-[r:KNOWS]->(b:Person) "
    "RETURN a.name, r.since, b.name"
)
ZERO_WRITES = {
    "labels_added": 0,
    "labels_removed": 0,
    "nodes_created": 0,
    "nodes_deleted": 0,
    "properties_removed": 0,
    "properties_set": 0,
    "rels_created": 0,
    "rels_deleted": 0,
}
EXPECTED_INSPECTION = {
    "id": 1,
    "label": "Person",
    "neighbors": [
        {
            "direction": "out",
            "label": "Person",
            "node_id": 2,
            "rel_type": "KNOWS",
        }
    ],
    "properties": {
        "embedding": [1.0, 0.0, 0.0],
        "language": "Analytical Engine",
        "name": "Ada Lovelace",
        "text": "Analytical Engine algorithm notes",
    },
}
EXPECTED_QUERY = {
    "columns": ["a.name", "r.since", "b.name"],
    "row_count": 1,
    "rows": [["Ada Lovelace", 1952, "Grace Hopper"]],
    "truncated": False,
    "writes": ZERO_WRITES,
}


class WalkthroughFailure(RuntimeError):
    """A quickstart contract failed."""


ACTIVE_SESSIONS = []


def compact(value):
    """Stable JSON used both by the README and the gate output."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def expect_equal(step, actual, expected):
    if actual != expected:
        raise WalkthroughFailure(
            "{}: expected {}, got {}".format(step, compact(expected), compact(actual))
        )


def expect(step, condition, expectation, actual):
    if not condition:
        raise WalkthroughFailure(
            "{}: expected {}; got {}".format(step, expectation, compact(actual))
        )


class McpSession:
    """One newline-delimited stdio MCP session."""

    def __init__(self, binary, data_dir):
        self.stderr = tempfile.TemporaryFile(mode="w+b")
        self.process = subprocess.Popen(
            [
                str(binary),
                "serve",
                "--stdio-mcp",
                "--data",
                str(data_dir),
                "--admin-http",
                "",
                "--metrics-http",
                "",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            bufsize=0,
        )
        self.stdout_buffer = b""
        ACTIVE_SESSIONS.append(self)

    def diagnostics(self):
        if self.process.poll() is None:
            return "server is still running"
        self.stderr.flush()
        self.stderr.seek(0)
        text = self.stderr.read().decode("utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-30:]) or "(server wrote no diagnostics)"

    def send(self, request):
        payload = compact(request).encode("utf-8") + b"\n"
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise WalkthroughFailure(
                "send {}: server closed stdin: {}\n{}".format(
                    request.get("method"), error, self.diagnostics()
                )
            ) from error

    def read_response(self, step):
        deadline = time.monotonic() + RESPONSE_TIMEOUT_SECONDS
        stdout_fd = self.process.stdout.fileno()
        while b"\n" not in self.stdout_buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                diagnostics = self.force_close_with_diagnostics()
                raise WalkthroughFailure(
                    "{}: expected a JSON-RPC response within {}s; got timeout\n{}".format(
                        step, RESPONSE_TIMEOUT_SECONDS, diagnostics
                    )
                )
            readable, _, _ = select.select([stdout_fd], [], [], remaining)
            if not readable:
                continue
            chunk = os.read(stdout_fd, 4096)
            if not chunk:
                diagnostics = self.force_close_with_diagnostics()
                raise WalkthroughFailure(
                    "{}: expected a JSON-RPC response; got EOF\n{}".format(
                        step, diagnostics
                    )
                )
            self.stdout_buffer += chunk

        line, self.stdout_buffer = self.stdout_buffer.split(b"\n", 1)
        try:
            return json.loads(line.rstrip(b"\r").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise WalkthroughFailure(
                "{}: expected one newline-delimited JSON-RPC response; got {!r}".format(
                    step, line
                )
            ) from error

    def request(self, request, step):
        self.send(request)
        response = self.read_response(step)
        expect_equal("{} response id".format(step), response.get("id"), request["id"])
        expect_equal(
            "{} JSON-RPC version".format(step), response.get("jsonrpc"), "2.0"
        )
        expect(
            step,
            "error" not in response,
            "a success response",
            response,
        )
        return response

    def close_cleanly(self, step):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            status = self.process.wait(timeout=SHUTDOWN_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as error:
            diagnostics = self.force_close_with_diagnostics()
            raise WalkthroughFailure(
                "{}: expected clean exit within {}s; server did not stop\n{}".format(
                    step, SHUTDOWN_TIMEOUT_SECONDS, diagnostics
                )
            ) from error
        if status != 0:
            raise WalkthroughFailure(
                "{}: expected exit code 0; got {}\n{}".format(
                    step, status, self.diagnostics()
                )
            )
        self._release()

    def _stop_process(self):
        if self.process.poll() is None:
            if self.process.stdin and not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)

    def force_close_with_diagnostics(self):
        self._stop_process()
        diagnostics = self.diagnostics()
        self._release()
        return diagnostics

    def force_close(self):
        self._stop_process()
        self._release()

    def _release(self):
        if self in ACTIVE_SESSIONS:
            ACTIVE_SESSIONS.remove(self)
        if self.process.stdout:
            self.process.stdout.close()
        self.stderr.close()


def initialize(session, request_id, phase):
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "arcgraph-agent-quickstart", "version": "1"},
        },
    }
    response = session.request(request, "{} readiness".format(phase))
    result = response["result"]
    expect_equal(
        "{} protocol".format(phase), result.get("protocolVersion"), PROTOCOL_VERSION
    )
    expect_equal(
        "{} server name".format(phase),
        result.get("serverInfo", {}).get("name"),
        "arcgraph",
    )
    expect_equal(
        "{} server version".format(phase),
        result.get("serverInfo", {}).get("version"),
        "0.1.0-beta",
    )
    expect_equal(
        "{} tools capability".format(phase),
        result.get("capabilities", {}).get("tools", {}).get("listChanged"),
        False,
    )
    return result


def parse_tool_body(response, step):
    result = response.get("result")
    expect(step, isinstance(result, dict), "an MCP tool result object", response)
    expect_equal("{} isError".format(step), result.get("isError"), False)
    content = result.get("content")
    expect(
        step,
        isinstance(content, list) and len(content) == 1,
        "exactly one MCP content item",
        result,
    )
    expect_equal("{} content type".format(step), content[0].get("type"), "text")
    try:
        rendered = json.loads(content[0]["text"])
        body = json.loads(rendered["body"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise WalkthroughFailure(
            "{}: expected JSON text containing a JSON body; got {}".format(
                step, compact(content[0])
            )
        ) from error
    expect_equal("{} render format".format(step), rendered.get("format"), "json")
    return body


def tools_call(session, request_id, name, arguments, step):
    request = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }
    response = session.request(request, step)
    return request, response, parse_tool_body(response, step)


def run_walkthrough(binary, data_dir):
    first = McpSession(binary, data_dir)
    init = initialize(first, 1, "initial start")
    print(
        "agent-quickstart: READY server={} protocol={}".format(
            init["serverInfo"]["name"], init["protocolVersion"]
        )
    )

    first.send(
        {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
    )
    catalog_response = first.request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        },
        "MCP catalog",
    )
    descriptors = catalog_response["result"].get("tools")
    expect(
        "MCP catalog",
        isinstance(descriptors, list),
        "result.tools to be an array",
        catalog_response,
    )
    names = [tool.get("name") for tool in descriptors]
    expect_equal("MCP catalog names", names, TOOLS)
    for descriptor in descriptors:
        name = descriptor.get("name")
        expect(
            "MCP descriptor {}".format(name),
            bool(descriptor.get("description")),
            "a non-empty description",
            descriptor,
        )
        expect_equal(
            "MCP schema type {}".format(name),
            descriptor.get("inputSchema", {}).get("type"),
            "object",
        )
    raw_descriptor = next(tool for tool in descriptors if tool["name"] == "graph.raw_query")
    expect(
        "graph.raw_query schema",
        "query" in raw_descriptor["inputSchema"].get("required", []),
        "query to be required",
        raw_descriptor["inputSchema"],
    )
    print("agent-quickstart: TOOLS {} ({})".format(",".join(names), len(names)))

    _, _, ingest = tools_call(
        first,
        3,
        "graph.ingest",
        {
            "tenant_id": 1,
            "format": "json",
            "nodes": [
                {
                    "external_id": "ada",
                    "label": "Person",
                    "properties": {
                        "embedding": [1.0, 0.0, 0.0],
                        "name": "Ada Lovelace",
                        "language": "Analytical Engine",
                        "text": "Analytical Engine algorithm notes",
                    },
                },
                {
                    "external_id": "grace",
                    "label": "Person",
                    "properties": {
                        "embedding": [0.0, 1.0, 0.0],
                        "name": "Grace Hopper",
                        "language": "COBOL",
                        "text": "COBOL compiler systems",
                    },
                },
                {
                    "external_id": "katherine",
                    "label": "Person",
                    "properties": {
                        "embedding": [0.0, 0.0, 1.0],
                        "name": "Katherine Johnson",
                        "language": "FORTRAN",
                        "text": "orbital flight trajectory calculations",
                    },
                },
            ],
            "relationships": [
                {
                    "external_id": "ada-knows-grace",
                    "from_external_id": "ada",
                    "to_external_id": "grace",
                    "rel_type": "KNOWS",
                    "properties": {"since": 1952},
                }
            ],
            "acl_grants": [
                {"external_id": "ada", "read_principals": ["neo4j"]},
                {"external_id": "grace", "read_principals": ["neo4j"]},
                {"external_id": "katherine", "read_principals": ["neo4j"]},
            ],
        },
        "graph.ingest",
    )
    expect_equal("ingest inserted_count", ingest.get("inserted_count"), 4)
    expect_equal("ingest failed_count", ingest.get("failed_count"), 0)
    expect_equal("ingest dropped ACL grants", ingest.get("dropped_acl_grants"), None)
    expect(
        "ingest commit_lsn",
        isinstance(ingest.get("commit_lsn"), int) and ingest["commit_lsn"] > 0,
        "a positive durable commit LSN",
        ingest.get("commit_lsn"),
    )
    records = ingest.get("records")
    expect(
        "ingest records",
        isinstance(records, list) and len(records) == 4,
        "four per-record outcomes",
        records,
    )
    expect_equal(
        "ingest external ids",
        [record.get("external_id") for record in records],
        ["ada", "grace", "katherine", "ada-knows-grace"],
    )
    expect_equal(
        "ingest statuses",
        [record.get("status") for record in records],
        ["inserted", "inserted", "inserted", "inserted"],
    )
    expect_equal(
        "ingest internal ids",
        [record.get("internal_id") for record in records],
        [1, 2, 3, 1],
    )
    print(
        "agent-quickstart: INGEST inserted={} failed={} node_ids=1,2,3 rel_id=1 acl_principal=neo4j".format(
            ingest["inserted_count"], ingest["failed_count"]
        )
    )

    inspect_request, inspect_response, inspection = tools_call(
        first,
        4,
        "graph.inspect",
        {"tenant_id": 1, "node_id": 1, "format": "json"},
        "graph.inspect",
    )
    expect_equal("graph.inspect values", inspection, EXPECTED_INSPECTION)
    print("agent-quickstart: MCP_REQUEST {}".format(compact(inspect_request)))
    print("agent-quickstart: MCP_RESPONSE {}".format(compact(inspect_response)))
    print("agent-quickstart: READBACK {}".format(compact(inspection)))

    _, _, query = tools_call(
        first,
        5,
        "graph.raw_query",
        {"tenant_id": 1, "query": ARCQL, "format": "json"},
        "ArcQL query",
    )
    expect_equal("ArcQL values", query, EXPECTED_QUERY)
    print("agent-quickstart: ARCQL {}".format(compact(query)))

    bm25_request, _, bm25 = tools_call(
        first,
        6,
        "graph.search",
        {
            "tenant_id": 1,
            "query": "compiler",
            "k": 2,
            "principal": "neo4j",
            "format": "json",
        },
        "BM25 search",
    )
    expect_equal("BM25 honored k", bm25.get("k"), 2)
    expect(
        "BM25 hits",
        isinstance(bm25.get("hits"), list) and len(bm25["hits"]) >= 1,
        "at least one ranked hit",
        bm25,
    )
    expect_equal("BM25 rank 1 node", bm25["hits"][0].get("node_id"), 2)
    expect_equal("BM25 rank 1 label", bm25["hits"][0].get("label"), "Person")
    print("agent-quickstart: BM25_REQUEST {}".format(compact(bm25_request)))
    print("agent-quickstart: BM25 {}".format(compact(bm25)))

    vector_request, _, vector = tools_call(
        first,
        7,
        "graph.search",
        {
            "tenant_id": 1,
            "query_vec": [1.0, 0.0, 0.0],
            "k": 2,
            "principal": "neo4j",
            "format": "json",
        },
        "vector search",
    )
    expect_equal("vector honored k", vector.get("k"), 2)
    expect(
        "vector hits",
        isinstance(vector.get("hits"), list) and len(vector["hits"]) == 2,
        "two ranked hits",
        vector,
    )
    expect_equal("vector rank 1 node", vector["hits"][0].get("node_id"), 1)
    expect_equal("vector rank 1 label", vector["hits"][0].get("label"), "Person")
    print("agent-quickstart: VECTOR_REQUEST {}".format(compact(vector_request)))
    print("agent-quickstart: VECTOR {}".format(compact(vector)))

    first.close_cleanly("initial shutdown")
    print("agent-quickstart: STOP phase=initial exit=0")

    restarted = McpSession(binary, data_dir)
    restart_init = initialize(restarted, 8, "restart")
    print(
        "agent-quickstart: RESTART_READY server={} protocol={}".format(
            restart_init["serverInfo"]["name"], restart_init["protocolVersion"]
        )
    )
    _, _, durable_query = tools_call(
        restarted,
        9,
        "graph.raw_query",
        {"tenant_id": 1, "query": ARCQL, "format": "json"},
        "post-restart ArcQL query",
    )
    expect_equal("post-restart ArcQL values", durable_query, EXPECTED_QUERY)
    _, _, durable_inspection = tools_call(
        restarted,
        10,
        "graph.inspect",
        {"tenant_id": 1, "node_id": 1, "format": "json"},
        "post-restart graph.inspect",
    )
    expect_equal(
        "post-restart graph.inspect values",
        durable_inspection,
        EXPECTED_INSPECTION,
    )
    _, _, durable_bm25 = tools_call(
        restarted,
        11,
        "graph.search",
        {
            "tenant_id": 1,
            "query": "compiler",
            "k": 1,
            "principal": "neo4j",
            "format": "json",
        },
        "post-restart BM25 search",
    )
    expect_equal(
        "post-restart BM25 rank 1 node",
        durable_bm25.get("hits", [{}])[0].get("node_id"),
        2,
    )
    _, _, durable_vector = tools_call(
        restarted,
        12,
        "graph.search",
        {
            "tenant_id": 1,
            "query_vec": [1.0, 0.0, 0.0],
            "k": 1,
            "principal": "neo4j",
            "format": "json",
        },
        "post-restart vector search",
    )
    expect_equal(
        "post-restart vector rank 1 node",
        durable_vector.get("hits", [{}])[0].get("node_id"),
        1,
    )
    print(
        "agent-quickstart: DURABLE rows={} readback_name={}".format(
            compact(durable_query["rows"]),
            compact(durable_inspection["properties"]["name"]),
        )
    )
    restarted.close_cleanly("restart shutdown")
    print("agent-quickstart: STOP phase=restart exit=0")
    print("agent-quickstart: PASS all values survived restart")


def cleanup_sessions():
    for session in list(ACTIVE_SESSIONS):
        session.force_close()


def interrupted(signum, _frame):
    raise KeyboardInterrupt("received signal {}".format(signum))


def main():
    parser = argparse.ArgumentParser(
        description="Run the copy-pasteable ArcGraph durable MCP quickstart."
    )
    parser.add_argument(
        "--bin",
        default="target/debug/arcgraph",
        help="path to the built arcgraph binary",
    )
    parser.add_argument(
        "--data",
        help="optional empty data directory; a temporary directory is used by default",
    )
    args = parser.parse_args()

    binary = Path(args.bin).resolve()
    if not binary.is_file():
        raise WalkthroughFailure(
            "setup: expected built binary at {}; run "
            "`cargo build --workspace` first".format(binary)
        )

    if args.data:
        data_dir = Path(args.data).resolve()
        data_dir.mkdir(parents=True, exist_ok=True)
        run_walkthrough(binary, data_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="arcgraph-agent-quickstart-") as temp:
            run_walkthrough(binary, Path(temp))


if __name__ == "__main__":
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except WalkthroughFailure as error:
        print("agent-quickstart: FAIL {}".format(error), file=sys.stderr)
        cleanup_sessions()
        sys.exit(1)
    except KeyboardInterrupt as error:
        print("agent-quickstart: FAIL interrupted: {}".format(error), file=sys.stderr)
        cleanup_sessions()
        sys.exit(130)
    finally:
        cleanup_sessions()
