//! Binary entrypoints and operator tooling for ArcGraph.
//!
//! Scope: the `serve`, `check`, `dump`, `health`, `migrate`, `backup`, and
//! `load` subcommands; storage startup; transport composition; and operator
//! listeners. Database algorithms remain in their subsystem crates.
//!
//! # Signal handling
//!
//! `arcgraph serve` wraps `shutdown_on_term()` with a
//! `CancellationRegistry::cancel_all()` call before the serve loop
//! terminates. This converts an operator's SIGTERM (or
//! Ctrl-C) into a graceful drain of every in-flight query — each
//! query's `CancellationToken` fires, the executor returns
//! `ExplainError::Cancelled` at the next batch boundary, and the
//! `CancellationRegistry` unregisters cleanly on its way out (no
//! token leak).
//!
//! Three binary wire sites carry this discipline (each owns its own
//! `CancellationRegistry` instance — there is no global registry):
//!
//! 1. `bin/arcgraph.rs::run_serve_stdio` (the umbrella `arcgraph
//!    serve --stdio-mcp` path).
//! 2. `bin/arcgraph.rs::run_serve_bolt` (the umbrella `arcgraph
//!    serve --bolt` path).
//! 3. `bin/arcgraph_mcp_stdio.rs::run` (the standalone
//!    `arcgraph-mcp-stdio` binary).
//!
//! All three share the same shutdown-future shape — `async move {
//! shutdown_on_term().await; cancel_registry_for_shutdown.cancel_all();
//! tracing::info!(...) }` — so a sister-cite enumeration discipline
//! catches any future fourth call site that forgets the wrap.
//!
//! The seam contract — `CancellationRegistry::cancel_all()` fires all
//! registered tokens—is pinned by
//! `crates/arcgraph-query/tests/cancel_integration.rs::sigterm_during_query_fires_token`
//! plus `::mcp_stdio_shutdown_sister_site_fires_cancel_all`.

#![recursion_limit = "256"]

pub mod bootstrap;
/// Offline data-directory generation upgrades.
pub mod data_dir_migration;
/// Exclusive advisory inter-process lock on a durable `--data` dir (#886).
pub mod data_lock;
/// Generation-namespace registry for `gen-*` directory names.
pub mod generation_namespace;
/// Offline native bootstrap-load implementation for `arcgraph load`.
pub mod m5_load;
pub mod m5_parallel;
pub mod migrate;
pub mod ops;
pub mod vector_search;
