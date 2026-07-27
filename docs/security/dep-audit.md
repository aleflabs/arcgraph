# Dependency audit

The repository keeps Rust dependency policy in `deny.toml` and RustSec audit
configuration in `.cargo/audit.toml`.

The regular CI workflow checks the configured license/source policy. The
scheduled security workflow refreshes advisory data so an unchanged
`Cargo.lock` is re-evaluated as new advisories appear.

An exception must be narrow, documented beside the relevant configuration,
and tied to a reviewable update. Do not infer that a clean dependency scan
proves database, protocol, authorization, or recovery correctness.
