# syntax=docker/dockerfile:1.7
# ─────────────────────────────────────────────────────────────────────────────
# ArcGraph multi-stage distroless container — W15β M6-03
#
# Builder:  rust:1.85-slim-bookworm (matches workspace `rust-version`)
# Runtime:  gcr.io/distroless/cc-debian12:nonroot (glibc + libgcc; ~22MB)
#
# Minimum Docker version: 23.0 (BuildKit-by-default). The
# `--mount=type=cache` directives below are BuildKit-only; older
# engines (<18.09) reject them outright. Newer engines (18.09–22.x)
# require `DOCKER_BUILDKIT=1` explicitly. The smoke test in
# `crates/arcgraph-cli/tests/m6_03_docker_release.rs` sets the env
# var defensively so the test isn't engine-version-dependent.
#
# License gate (Prime Directive #1 — Apache-2.0/MIT throughout):
#   • rust:1.85-slim-bookworm    → Apache-2.0 OR MIT (Rust upstream)
#   • distroless/cc-debian12     → Apache-2.0 (Google distroless project)
#       https://github.com/GoogleContainerTools/distroless/blob/main/LICENSE
#   Both base images are PD-1 compatible.
#
# Roadmap pin: roadmap.md M6-03 — "Docker image (distroless, ~20MB).
# `docker run arcgraph` works." Depends on M6-02 (W15α umbrella binary
# `arcgraph serve|check|dump`); shipped in PR #306. The W16ζ post-#306
# binary swap (issue #311 / this PR) re-establishes the documented
# `arcgraph serve --http 0.0.0.0:8080` ENTRYPOINT.
#
# Structured deviations from W15β spawn prompt (per
# feedback_hard_boundary_deviation_protocol.md):
#   D1. Rust 1.82 → 1.85 (workspace `rust-version = "1.85"` per Cargo.toml).
#   D2. Prometheus EXPOSE port 9100 → 9090 (design-v2 §10.2 canonical
#       "Prometheus scrape endpoint on port 9090 by default"; cite-
#       CORRECTNESS verified per feedback_cite_correctness_not_just_resolution.md).
#   D3. HEALTHCHECK — RESOLVED in W16 M6-10 (#310). The `arcgraph
#       health` subcommand (added to the W15α M6-02 umbrella binary)
#       is the distroless-safe in-binary probe; it issues a plain
#       HTTP/1.1 GET against the `/healthz` path in
#       `crates/arcgraph-mcp/src/transport/http.rs` (the
#       `PATH_HEALTHZ` const) and exits 0 on 2xx / 1 otherwise. The
#       `HEALTHCHECK` directive below invokes it with the M6-10
#       defaults (`http://127.0.0.1:8080/healthz`, 2s timeout). The
#       probe is wire-functional now that #311 has swapped the
#       ENTRYPOINT binary to the umbrella `arcgraph` (D4 below);
#       `docker inspect --format '{{.State.Health}}'` will surface
#       the probe verdict once the M6-08+ HTTPS cert-resolver wires
#       a listener for `arcgraph serve --http` to bind against.
#   D4. RESOLVED in W16ζ / PR for issue #311 — `arcgraph-mcp-stdio`
#       interim ENTRYPOINT swapped to the umbrella `arcgraph` binary's
#       `serve --http 0.0.0.0:8080` invocation once M6-02 (PR #306)
#       landed. Note that `arcgraph serve --http <addr>` currently
#       bails at runtime ("HTTPS/TLS wiring is forward-bound to M6-08+",
#       per crates/arcgraph-cli/src/bin/arcgraph.rs `run_serve_http`)
#       — `docker run arcgraph` exits 1 with that forward-bind message
#       until M6-08+ wires the cert resolver. This is intentional: a
#       loud "feature not yet wired" failure beats the pre-#311 silent
#       stdin-EOF exit-0 that masqueraded as a working server. The
#       smoke test at crates/arcgraph-cli/tests/m6_03_docker_release.rs
#       overrides ENTRYPOINT to invoke `arcgraph --version` so the
#       container's binary integrity is still pinned independently of
#       the M6-08+ wiring.
# ─────────────────────────────────────────────────────────────────────────────

# ───────── Stage 1: builder ─────────
# W19β #326 — base images SHA-pinned by manifest digest. The
# `<tag>@sha256:<digest>` form is the OCI distribution spec's
# "immutable reference" syntax: a future tag-republish cannot move
# the bytes underneath us, and `docker build --pull` against this
# Dockerfile resolves to the exact same multi-arch manifest list we
# verified at W19β time.
#
# Refresh procedure: see `scripts/refresh-docker-pins.sh`. The
# refresh script re-queries the registry, writes the new digest
# back into this Dockerfile, and emits a diff for review — pin
# bumps are a deliberate human action, never automatic.
#
# Source verification (2026-05-17, W19β):
#   rust:1.85-slim-bookworm → sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36
#   Verified via:
#     curl -sI -H "Authorization: Bearer $(curl -s 'https://auth.docker.io/token?service=registry.docker.io&scope=repository:library/rust:pull' | jq -r .token)" \
#         -H "Accept: application/vnd.oci.image.index.v1+json" \
#         https://registry-1.docker.io/v2/library/rust/manifests/1.85-slim-bookworm
FROM rust:1.85-slim-bookworm@sha256:9f841bbe9e7d8e37ceb96ed907265a3a0df7f44e3737d0b100e7907a679acb36 AS builder

# `pkg-config` + `libssl-dev` are NOT needed (workspace pins
# `rustls` + `aws-lc-rs`, not OpenSSL). `ca-certificates` is needed at
# runtime, not build-time; distroless/cc already ships it.
#
# `--mount=type=cache` requires BuildKit (docker buildx, or
# DOCKER_BUILDKIT=1); falls back to a fresh apt index if unavailable.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifest + lock first so the `cargo fetch` registry
# cache survives source-only changes. The full source copy is in the
# next layer; .dockerignore strips target/, .git/, fuzz/, etc.
COPY Cargo.toml Cargo.lock ./
COPY deny.toml rustfmt.toml clippy.toml ./
# Apache 2.0 §4(a) recommends retaining LICENSE in derivative
# distributions; copying it through the builder lets the runtime stage
# `COPY --from=builder /build/LICENSE /LICENSE` populate the container
# image as well (the release tarball already includes it).
COPY LICENSE ./
COPY crates ./crates
COPY benches ./benches
# (top-level `tests/` is empty placeholder; .dockerignore skips it
# alongside fuzz/.)

# Build the binary. `--locked` enforces Cargo.lock as the dependency
# wire-of-record. `--release` activates `profile.release`
# settings (`lto = "thin"`, `codegen-units = 1`, `strip = "symbols"`)
# defined in workspace Cargo.toml; no further `strip` invocation needed.
RUN --mount=type=cache,target=/build/target,sharing=locked \
    --mount=type=cache,target=/usr/local/cargo/registry,sharing=locked \
    cargo build --release --locked --bin arcgraph && \
    cp /build/target/release/arcgraph /tmp/arcgraph

# ───────── Stage 2: runtime ─────────
# distroless/cc-debian12 ships glibc + libgcc + libssl + ca-certificates
# but no shell, no package manager, no other binaries — minimal attack
# surface per CIS Docker Benchmark §4.6. The `:nonroot` tag fixes UID/GID
# to 65532, which the binary inherits.
#
# W19β #326 — SHA-pinned by manifest digest. See the §"Refresh procedure"
# note in Stage 1 above. Source verification (2026-05-17, W19β):
#   gcr.io/distroless/cc-debian12:nonroot → sha256:e2d29aec8061843706b7e484c444f78fafb05bfe47745505252b1769a05d14f1
#   Verified via:
#     curl -sI -H "Accept: application/vnd.oci.image.index.v1+json" \
#         https://gcr.io/v2/distroless/cc-debian12/manifests/nonroot
FROM gcr.io/distroless/cc-debian12:nonroot@sha256:e2d29aec8061843706b7e484c444f78fafb05bfe47745505252b1769a05d14f1 AS runtime

# OCI annotations per opencontainers.org/image-spec — populated by the
# release workflow's docker/build-push-action invocation (see
# .github/workflows/release.yml).
LABEL org.opencontainers.image.title="ArcGraph"
LABEL org.opencontainers.image.description="Embeddable graph, vector, full-text, and traversal database engine with MCP access"
LABEL org.opencontainers.image.licenses="Apache-2.0"
LABEL org.opencontainers.image.source="https://github.com/aleflabs/arcgraph"
LABEL org.opencontainers.image.documentation="https://github.com/aleflabs/arcgraph/blob/main/README.md"
LABEL org.opencontainers.image.vendor="ArcGraph contributors"

COPY --from=builder --chown=nonroot:nonroot \
    /tmp/arcgraph /usr/local/bin/arcgraph

# Apache 2.0 §4(a) — include LICENSE inside the image so
# `docker run --rm arcgraph cat /LICENSE` is a working contract.
# (Distroless has no `cat`, but `docker cp <ctr>:/LICENSE -` works,
# and other tooling can extract it via `crane export`.)
COPY --from=builder --chown=nonroot:nonroot \
    /build/LICENSE /LICENSE

# Ports per design-v2 §M5 transport composition + §10.2 observability:
#   • 7687 — Bolt 5.0 (W14δ M5-13 transport scaffold; Neo4j-driver compat).
#   • 8080 — HTTP MCP (W14α M5-02b streamable-HTTP / TLS).
#   • 9090 — Prometheus scrape endpoint (design-v2 §10.2; M6-06 wires).
# These are declarative metadata — actual binds happen via the binary's
# CLI flags or config file.
EXPOSE 7687
EXPOSE 8080
EXPOSE 9090

USER nonroot:nonroot

# W16 M6-10 (#310) — distroless-safe in-binary HEALTHCHECK.
#
# `cc-debian12:nonroot` has no shell + no `curl` / `wget`, so a
# curl-shaped HEALTHCHECK is impossible. The `arcgraph health`
# subcommand (W15α M6-02 umbrella binary) ships a built-in HTTP/1.1
# GET probe against the `/healthz` endpoint and exits 0 (healthy) /
# 1 (unhealthy). The flags here track the design-v2 §M5 transport
# composition: `/healthz` on `127.0.0.1:8080` (the same port the
# `EXPOSE 8080` directive advertises), 2s per-attempt timeout, 30s
# poll interval, 10s start-period grace for the bind step, 3 retries
# before Docker marks the container unhealthy. The exec-form `CMD`
# array bypasses `/bin/sh` (which doesn't exist in distroless).
#
# The probe is wire-functional now that #311 has swapped the
# ENTRYPOINT binary to the umbrella `arcgraph` (see D3/D4 in this
# file's header). `docker inspect` will report the probe verdict
# once the M6-08+ HTTPS cert-resolver lands a listener for the
# `serve --http` ENTRYPOINT to bind against.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["/usr/local/bin/arcgraph", "health"]

# `arcgraph` is the W15α M6-02 umbrella binary (PR #306) exposing
# `serve`, `check`, and `dump` subcommands. The `serve --http <addr>`
# transport will bind once the cert-resolver wiring lands in M6-08+;
# until then the ENTRYPOINT exits 1 with a "HTTPS/TLS wiring is
# forward-bound to M6-08+" diagnostic from
# `crates/arcgraph-cli/src/bin/arcgraph.rs::run_serve_http`. To run
# the binary directly (e.g., `--version`, `check`, `dump`), override
# the ENTRYPOINT: `docker run --entrypoint /usr/local/bin/arcgraph
# <image> --version`. The M6-03 smoke test takes that same path.
ENTRYPOINT ["/usr/local/bin/arcgraph", "serve", "--http", "0.0.0.0:8080"]
