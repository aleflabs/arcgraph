//! Runtime configuration for storage-layer behavior.
//!
//! This module hosts operator-facing knobs that switch storage-layer
//! behavior between documented alternatives. The current set:
//!
//! - [`WalErrorPolicy`] — ADR-033 §8. Controls the response to a
//!   **foreground** WAL fsync failure during commit: default
//!   `Rollback` (Z-1 (b) in-memory rollback) or opt-in `Abort`
//!   (PostgreSQL-style fail-fast).
//! - [`crate::wal::BackgroundFsyncFailAction`] — ADR-034 §8.6 /
//!   D-7. Controls the response to a **background** (T3 scheduler)
//!   fsync failure. Default `Abort`; test-only
//!   `RollbackAndContinue` override exists but is NOT recommended
//!   for production.
//!
//! # ADR-033 × ADR-034 interaction matrix
//!
//! | Failure site                        | `WalErrorPolicy`  | `BackgroundFsyncFailAction` | Effective action         |
//! |-------------------------------------|-------------------|-----------------------------|--------------------------|
//! | Foreground `wal.append` (T1 or T3)  | `Rollback`        | —                           | Z-1 (b) in-memory unwind |
//! | Foreground `wal.append` (T1 or T3)  | `Abort`           | —                           | Process abort            |
//! | Foreground fsync (T1 batch)         | `Rollback`        | —                           | Z-1 (b) in-memory unwind |
//! | Foreground fsync (T1 batch)         | `Abort`           | —                           | Process abort            |
//! | Foreground fsync (mixed T1/T3 batch)| any               | —                           | **Process abort** (§6.2) |
//! | Background fsync (T3 scheduler)     | —                 | `Abort` (default)           | Process abort            |
//! | Background fsync (T3 scheduler)     | —                 | `RollbackAndContinue`       | Log + continue (TEST)    |
//!
//! All knobs are environment-variable driven at process startup.
//! See the individual enum's `from_env` for the variable name and
//! semantics. First-read-wins via `OnceLock` so a test-time
//! env-var change is NOT observed once a worker has read the
//! policy.

use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// WalCheckpointConfig — SVC-1 / #849 / ADR-229 checkpoint trigger
// ─────────────────────────────────────────────────────────────────────

/// Trigger policy for the ADR-229 WAL checkpoint producer.
///
/// A checkpoint durably snapshots the full reconstructed state + records
/// the frontier, so restart-recovery replays only WAL-since-checkpoint
/// (`O(WAL-since-checkpoint)`) instead of the entire history. This config
/// tunes how often a checkpoint fires on the background (Tokio
/// work-stealing) pool — NOT the thread-per-core hot path (design-v2
/// §4.1). A graceful shutdown ALSO fires a checkpoint regardless of these
/// thresholds.
///
/// # Back-of-envelope default (PD#5)
///
/// The rc-blocker (#849) was a 167 GB WAL that could not replay in
/// 8.5 min at 10M. Restart-recovery time ≈ (WAL bytes since checkpoint) /
/// (replay throughput). To keep restart bounded to a small budget, the
/// steady-state WAL-since-checkpoint must stay bounded:
///
///   `steady_state_wal_bytes ≈ interval_bytes × safety_factor`
///
/// Choosing `interval_bytes = 1 GiB`: a 167 GB workload checkpoints
/// ~167×/run, and at any restart the replay backlog is ≤ ~1 GiB (plus the
/// in-flight segment). At the observed replay rate (the whole 167 GB
/// stalled >8.5 min ⇒ ≲ 330 MB/s replay), ≤1 GiB replays in ≲ 3 s —
/// well inside a bounded-restart budget, and independent of total uptime.
/// The checkpoint I/O cost (one page-image write per live page + one
/// record per live MVCC row) is amortized over 1 GiB of commits, so the
/// background checkpoint never dominates foreground throughput.
///
/// A `_seconds` bound (default 5 min) is an upper wall-clock cap so a
/// low-write store still checkpoints periodically (bounding the segment
/// backlog by time even when the byte threshold is never reached).
/// Set `interval_bytes = 0` AND `interval_seconds = 0` to DISABLE the
/// interval trigger (shutdown checkpoint still fires) — e.g. for a
/// read-mostly or test deployment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WalCheckpointConfig {
    /// Fire a background checkpoint after this many WAL bytes have been
    /// appended since the last checkpoint. `0` disables the byte
    /// trigger. Default: 1 GiB (see the back-of-envelope above).
    pub interval_bytes: u64,
    /// Fire a background checkpoint at most this many seconds after the
    /// last one (upper wall-clock cap for low-write stores). `0`
    /// disables the time trigger. Default: 300 s (5 min).
    pub interval_seconds: u64,
    /// Whether to fire a checkpoint on graceful shutdown. Default
    /// `true` — a graceful shutdown that skips the checkpoint would
    /// force a full-WAL replay on the next restart. Kept configurable
    /// so a test can assert the no-shutdown-checkpoint fallback path.
    pub checkpoint_on_shutdown: bool,
}

impl WalCheckpointConfig {
    /// Default WAL-byte interval: 1 GiB (see the type's back-of-envelope).
    pub const DEFAULT_INTERVAL_BYTES: u64 = 1024 * 1024 * 1024;
    /// Default wall-clock interval cap: 5 minutes.
    pub const DEFAULT_INTERVAL_SECONDS: u64 = 300;

    /// Whether the interval (background) trigger is enabled at all. When
    /// both thresholds are `0`, only the shutdown checkpoint fires.
    #[must_use]
    pub fn interval_enabled(&self) -> bool {
        self.interval_bytes > 0 || self.interval_seconds > 0
    }

    /// Whether `wal_bytes_since_last` crosses the byte threshold.
    #[must_use]
    pub fn byte_threshold_reached(&self, wal_bytes_since_last: u64) -> bool {
        self.interval_bytes > 0 && wal_bytes_since_last >= self.interval_bytes
    }

    /// Whether `elapsed_secs` crosses the wall-clock threshold.
    #[must_use]
    pub fn time_threshold_reached(&self, elapsed_secs: u64) -> bool {
        self.interval_seconds > 0 && elapsed_secs >= self.interval_seconds
    }
}

impl Default for WalCheckpointConfig {
    fn default() -> Self {
        Self {
            interval_bytes: Self::DEFAULT_INTERVAL_BYTES,
            interval_seconds: Self::DEFAULT_INTERVAL_SECONDS,
            checkpoint_on_shutdown: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// ARCGRAPH_WAL_ERROR_POLICY — ADR-033 §8
// ─────────────────────────────────────────────────────────────────────

/// Policy for responding to a WAL fsync failure during commit.
///
/// In-memory rollback is the default because it restores the
/// transaction's MVCC versions and page-state mutations before
/// returning the fsync error, so a write that failed to become durable
/// cannot remain visible. Process abort remains an explicit opt-in.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum WalErrorPolicy {
    /// Z-1 (b) — unwind the transaction's MVCC versions AND
    /// in-memory page-state mutations, return `Err` to the caller
    /// who may retry. The default; matches FoundationDB's per-
    /// partition rollback shape.
    Rollback,

    /// PostgreSQL-style fail-fast — call [`std::process::abort`] on
    /// WAL fsync failure. No rollback runs; `visible` stays at
    /// pre-W LSN; the process dies. On restart, ADR-032's WAL
    /// replay executor rebuilds state from the durable prefix.
    /// Opt-in via `ARCGRAPH_WAL_ERROR_POLICY=abort`.
    Abort,
}

impl Default for WalErrorPolicy {
    /// Default is [`Self::Rollback`] per owner choice in #72.
    #[inline]
    fn default() -> Self {
        Self::Rollback
    }
}

impl WalErrorPolicy {
    /// Environment variable name this policy reads.
    pub const ENV_VAR: &'static str = "ARCGRAPH_WAL_ERROR_POLICY";

    /// Parse `ARCGRAPH_WAL_ERROR_POLICY` from the current process'
    /// environment.
    ///
    /// Recognized values (case-insensitive): `rollback` (default),
    /// `abort`. Unset or unrecognized values default to
    /// `Rollback` — a pathological env var like
    /// `ARCGRAPH_WAL_ERROR_POLICY=yolo` silently selects rollback,
    /// which is the safer choice. A future amendment may tighten
    /// this to reject unknown values with a startup error; deferred
    /// until the knob accumulates more variants.
    ///
    /// **`continue` / `continue-lossy`** is explicitly rejected at
    /// parse time — it would keep in-memory state after a WAL
    /// failure, violating ADR-023's "MVCC is authoritative"
    /// contract. Operators who think they want this are misreading
    /// fsync semantics; see ADR-033 §8.
    #[must_use]
    pub fn from_env() -> Self {
        match std::env::var(Self::ENV_VAR) {
            Ok(v) if v.eq_ignore_ascii_case("abort") => Self::Abort,
            Ok(v) if v.eq_ignore_ascii_case("rollback") => Self::Rollback,
            Ok(v)
                if v.eq_ignore_ascii_case("continue")
                    || v.eq_ignore_ascii_case("continue-lossy") =>
            {
                // `continue` violates ADR-023; parse-time rejection
                // is documented in the constant-arg branch. We still
                // must return A Policy — opt to `Rollback` (safe
                // default) and emit a warning via tracing.
                tracing::warn!(
                    "ADR-033 §8: {}={} rejected (violates ADR-023 MVCC-authoritative \
                     contract); falling back to rollback policy",
                    Self::ENV_VAR,
                    v,
                );
                Self::Rollback
            }
            _ => Self::Rollback,
        }
    }

    /// Process-wide cached policy, parsed once on first read.
    ///
    /// Tests that need to override the policy mid-process should
    /// use [`Self::from_env`] directly (re-parsing the env var each
    /// call) — `global()` uses a `OnceLock` whose value is fixed
    /// after the first read. Production callers want the cached
    /// value so they don't pay env-var lookup cost on every commit.
    pub fn global() -> Self {
        static POLICY: OnceLock<WalErrorPolicy> = OnceLock::new();
        *POLICY.get_or_init(Self::from_env)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // env-var tests serialize through this mutex so concurrent tests
    // don't race on setenv.
    use parking_lot::Mutex;
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn default_is_rollback() {
        assert_eq!(WalErrorPolicy::default(), WalErrorPolicy::Rollback);
    }

    #[test]
    fn from_env_unset_is_rollback() {
        let _guard = ENV_LOCK.lock();
        // SAFETY: serialized by ENV_LOCK; no concurrent env access.
        unsafe { std::env::remove_var(WalErrorPolicy::ENV_VAR) };
        assert_eq!(WalErrorPolicy::from_env(), WalErrorPolicy::Rollback);
    }

    #[test]
    fn from_env_abort_case_insensitive() {
        let _guard = ENV_LOCK.lock();
        for v in ["abort", "ABORT", "Abort", "aBoRt"] {
            // SAFETY: serialized by ENV_LOCK; no concurrent env access.
            unsafe { std::env::set_var(WalErrorPolicy::ENV_VAR, v) };
            assert_eq!(
                WalErrorPolicy::from_env(),
                WalErrorPolicy::Abort,
                "value {v} should parse to Abort",
            );
        }
        // SAFETY: serialized by ENV_LOCK; cleanup.
        unsafe { std::env::remove_var(WalErrorPolicy::ENV_VAR) };
    }

    #[test]
    fn from_env_rollback_explicit() {
        let _guard = ENV_LOCK.lock();
        // SAFETY: serialized by ENV_LOCK; no concurrent env access.
        unsafe { std::env::set_var(WalErrorPolicy::ENV_VAR, "rollback") };
        assert_eq!(WalErrorPolicy::from_env(), WalErrorPolicy::Rollback);
        unsafe { std::env::remove_var(WalErrorPolicy::ENV_VAR) };
    }

    #[test]
    fn from_env_continue_rejected() {
        let _guard = ENV_LOCK.lock();
        // SAFETY: serialized by ENV_LOCK; no concurrent env access.
        unsafe { std::env::set_var(WalErrorPolicy::ENV_VAR, "continue") };
        assert_eq!(
            WalErrorPolicy::from_env(),
            WalErrorPolicy::Rollback,
            "continue must fall back to rollback per ADR-033 §8",
        );
        unsafe { std::env::set_var(WalErrorPolicy::ENV_VAR, "continue-lossy") };
        assert_eq!(
            WalErrorPolicy::from_env(),
            WalErrorPolicy::Rollback,
            "continue-lossy must fall back to rollback per ADR-033 §8",
        );
        unsafe { std::env::remove_var(WalErrorPolicy::ENV_VAR) };
    }

    #[test]
    fn from_env_unknown_falls_back_to_rollback() {
        let _guard = ENV_LOCK.lock();
        // SAFETY: serialized by ENV_LOCK; no concurrent env access.
        unsafe { std::env::set_var(WalErrorPolicy::ENV_VAR, "yolo") };
        assert_eq!(WalErrorPolicy::from_env(), WalErrorPolicy::Rollback);
        unsafe { std::env::remove_var(WalErrorPolicy::ENV_VAR) };
    }
}
