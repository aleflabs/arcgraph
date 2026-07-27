//! W13ε M5-02 — SIGHUP-driven reload loop.
//!
//! Operators rotate certs by:
//!   1. Writing the new cert/key files atomically (e.g., `rename`d
//!      from a temp file or written by cert-manager + kubelet).
//!   2. Sending SIGHUP to the server (e.g., `kill -HUP $(pidof
//!      arcgraph)` or k8s `lifecycle.preStop`).
//!
//! The signal handler does NOT block: it simply triggers the
//! resolver's `reload()` which runs validation + atomic-swap. Failures
//! are logged at WARN level; the listener stays up with the previous
//! cert.
//!
//! ## Why SIGHUP and not mtime polling?
//!
//! ArcGraph uses a SIGHUP trigger rather than polling cert/key mtimes
//! because:
//!   1. It is push-driven (zero idle CPU vs. periodic polling).
//!   2. It composes with k8s pod-lifecycle hooks more naturally —
//!      `lifecycle.postStart` can re-emit SIGHUP after a Secret
//!      rotation without the operator needing to know the polling
//!      interval.
//!   3. Operators can manually rotate by `kill -HUP` for emergency
//!      response without waiting for the next polling tick.
//!
//! Future ACME / Vault providers may add their own internal
//! reload-trigger mechanisms (per-cert renewal callback) without
//! changing this loop's contract — the loop is the SIGHUP→`reload()`
//! glue, not the only reload trigger.
//!
//! ## Unix-only
//!
//! `tokio::signal::unix` is not available on non-unix targets. The
//! loop function is `cfg(unix)`-gated; non-unix builds get a
//! no-op stub at [`run_sighup_reload_loop`] that returns immediately
//! when shutdown is signaled, so the rest of the API surface compiles
//! on all platforms (the binary just never gets a SIGHUP-driven
//! reload — those operators use a future Windows ServiceControl-driven
//! variant or rely on the `HotReloadResolver::reload()` API directly).

use std::sync::Arc;

use super::error::{TlsResolverError, TlsResolverResult};
use super::resolver::HotReloadResolver;

/// Run the SIGHUP-driven reload loop until `shutdown` flips to `true`.
///
/// Spawn this on the operator's tokio runtime. The future resolves
/// `Ok(())` on clean shutdown; it only returns `Err` if the initial
/// signal-handler installation fails (rare — typically only when the
/// process is already in a state where Unix signal registration is
/// impossible, e.g., as a thread that lacks signal-handling privilege).
///
/// Per-reload errors do NOT propagate as the function's return value;
/// they are logged at WARN level and the loop continues. This is the
/// "listener never goes down" invariant from §5.11.
#[cfg(unix)]
pub async fn run_sighup_reload_loop(
    resolver: Arc<HotReloadResolver>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> TlsResolverResult<()> {
    use tokio::signal::unix::{SignalKind, signal};

    let mut hup = signal(SignalKind::hangup()).map_err(|source| TlsResolverError::Io {
        path: std::path::PathBuf::from("<sighup-handler>"),
        source,
    })?;

    let source = resolver.source_descriptor();
    tracing::info!(source = %source, "tls.reload.loop.started");

    loop {
        tokio::select! {
            biased;
            shutdown_changed = shutdown.changed() => {
                if shutdown_changed.is_err() || *shutdown.borrow() {
                    tracing::info!(source = %source, "tls.reload.loop.shutdown");
                    return Ok(());
                }
            }
            sig = hup.recv() => {
                if sig.is_none() {
                    // Signal stream closed — the reactor was dropped.
                    tracing::warn!(source = %source, "tls.reload.signal_stream_closed");
                    return Ok(());
                }
                drive_reload(&resolver);
            }
        }
    }
}

/// Non-unix stub: never observes a SIGHUP, but still resolves on
/// shutdown so the spawning task doesn't leak.
#[cfg(not(unix))]
pub async fn run_sighup_reload_loop(
    resolver: Arc<HotReloadResolver>,
    mut shutdown: tokio::sync::watch::Receiver<bool>,
) -> TlsResolverResult<()> {
    let source = resolver.source_descriptor();
    tracing::info!(source = %source, platform = "non-unix", "tls.reload.loop.started");
    let _ = shutdown.changed().await;
    tracing::info!(source = %source, "tls.reload.loop.shutdown");
    Ok(())
}

/// Single-shot reload driver, separated for direct testability and
/// for future reuse by non-SIGHUP triggers (e.g., a future
/// admin-tool-triggered reload via the Tier-2 MCP `admin.reload_tls`).
pub fn drive_reload(resolver: &HotReloadResolver) {
    let source = resolver.source_descriptor();
    match resolver.reload() {
        Ok(()) => {
            tracing::info!(source = %source, event = "tls.reload.success");
        }
        Err(err) => {
            // The "tls.reload.failed" structured field is the metric
            // name operators wire into Prometheus alert rules; keeping
            // the field name stable across reload paths is part of the
            // operational contract.
            tracing::warn!(
                source = %source,
                event = "tls.reload.failed",
                error = %err,
                "tls hot-reload failed; previous cert retained"
            );
        }
    }
}
