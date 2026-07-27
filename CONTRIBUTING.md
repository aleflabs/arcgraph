# Contributing to ArcGraph

ArcGraph `v0.1.0-beta` is the 13-crate bare database engine described in the
[README](README.md). Contributions must stay inside that distribution. Do not
add the removed prediction, anomaly, calibration, compliance, connector,
distribution, Python-sidecar, or bi-temporal surfaces as incidental work.

## Before changing code

1. Install the documented Rust toolchain and build the workspace using the
   [first-time setup](README.md#install-prerequisites).
2. Identify the owning crate in
   [`docs/bounded-contexts.md`](docs/bounded-contexts.md).
3. Check the public syntax and protocol contracts in
   [`docs/arcql-reference.md`](docs/arcql-reference.md),
   [`docs/search.md`](docs/search.md), and
   [`docs/transports.md`](docs/transports.md) when the change affects them.
4. Add or update a focused regression for behavior changes.

## Engineering rules

- Apache-2.0 throughout: do not add AGPL, GPL, SSPL, BUSL, or Commons Clause
  dependencies.
- Do not use `mmap` on the storage hot path.
- Every `unsafe` block requires an adjacent `// SAFETY:` explanation.
- Keep types and behavior in their bounded-context crate.
- Back performance claims with a reproducible measurement.
- Treat missing functionality honestly. A parser accepting syntax is not the
  same as an executable feature.

## Required checks

Run Cargo commands serially:

```bash
cargo build --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --check
cargo doc --workspace --no-deps
```

Run the focused tests for every crate you change. The complete workspace test
suite has large external and disk requirements; read
[`docs/testing-strategy.md`](docs/testing-strategy.md) before starting it.

## Change conventions

- Use a Conventional Commit prefix such as `feat:`, `fix:`, `perf:`,
  `refactor:`, `test:`, `docs:`, `chore:`, or `bench:`.
- Keep one logical change per commit and make reviewable checkpoints.
- Update public docs in the same change when behavior, flags, wire shapes, or
  supported syntax changes.
- Do not claim a test passed unless that exact command completed.

## Code of conduct

Be direct, rigorous, and respectful. Critique the change and its evidence, not
the person proposing it.
