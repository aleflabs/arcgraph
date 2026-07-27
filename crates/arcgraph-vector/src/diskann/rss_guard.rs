//! Process-RSS guard for the SSD-resident DiskANN serving tier (ADR-195 §2.2 / §4).
//!
//! ## Why this exists
//!
//! The 10M-vector GA-gate validation (ADR-189 §B) runs on a 19 GB box under a
//! hard RSS ceiling (`ARCGRAPH_VECTOR_RSS_CAP_MB`, default 14000). The bounded
//! [`arcgraph_storage::BufferPool`] frame-count PREVENTS steady-state breach and
//! the bounded-batch build (ADR-195 §3) prevents the build-time breach; this
//! guard is the **detect-and-abort backstop** for a transient spike. It is a
//! fail-CLEAN mechanism — it surfaces [`VectorIndexError::RssCapExceeded`] at the
//! next safe checkpoint rather than letting the process swap-thrash to an
//! OOM-kill (ADR-195 §2.2, "NO swap-thrash-to-death").
//!
//! "Validated at 10M" means *validated with the guard armed and never tripped
//! during the serving phase* — the guard tripping is itself the fault-injection
//! test (assert a clean abort, not an OOM-kill; doctrine §3).
//!
//! ## No `unsafe`
//!
//! The crate is `#![deny(unsafe_code)]`, so RSS is read through safe platform
//! paths: Linux `/proc/self/statm` (a plain file read); macOS `ps -o rss=`
//! (a subprocess). Neither uses FFI. The ~1 s sample cadence keeps the macOS
//! subprocess cost negligible and off the query hot path (the guard runs on its
//! own thread).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::{Result, VectorIndexError};

/// Default sampling cadence (ADR-195 §2.2: "~1 s").
pub const DEFAULT_SAMPLE_INTERVAL: Duration = Duration::from_millis(1000);

/// Default RSS cap in MB (`ARCGRAPH_VECTOR_RSS_CAP_MB` default per ADR-195 §2.2).
pub const DEFAULT_RSS_CAP_MB: u64 = 14000;

/// A background process-RSS sampler with a detect-and-abort contract.
///
/// Construct with [`RssGuard::spawn`] (arms the sampler) or [`RssGuard::disabled`]
/// (a no-op guard whose [`check`](RssGuard::check) never trips — for callers that
/// want the same code path without a ceiling). The owner polls
/// [`check`](RssGuard::check) at safe points (between build batches, between
/// queries); on a breach it returns [`VectorIndexError::RssCapExceeded`].
pub struct RssGuard {
    cap_mb: u64,
    tripped: Arc<AtomicBool>,
    observed_peak_mb: Arc<AtomicU64>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl RssGuard {
    /// Arm the guard: spawn a thread sampling process RSS every `interval`.
    /// When RSS exceeds `cap_mb` the guard latches `tripped`; subsequent
    /// [`check`](RssGuard::check) calls return [`VectorIndexError::RssCapExceeded`].
    ///
    /// The guard records the observed peak regardless of the cap so the caller
    /// can report the real high-water mark (the honest RSS number the Director
    /// wants, ADR-195 §6).
    #[must_use]
    pub fn spawn(cap_mb: u64, interval: Duration) -> Self {
        let tripped = Arc::new(AtomicBool::new(false));
        let observed_peak_mb = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));

        let t_tripped = Arc::clone(&tripped);
        let t_peak = Arc::clone(&observed_peak_mb);
        let t_stop = Arc::clone(&stop);

        let handle = std::thread::Builder::new()
            .name("arcgraph-vector-rss-guard".into())
            .spawn(move || {
                while !t_stop.load(Ordering::Relaxed) {
                    if let Some(rss) = current_rss_mb() {
                        // Monotonic peak (compare-and-set max).
                        let mut prev = t_peak.load(Ordering::Relaxed);
                        while rss > prev {
                            match t_peak.compare_exchange_weak(
                                prev,
                                rss,
                                Ordering::Relaxed,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(observed) => prev = observed,
                            }
                        }
                        if rss > cap_mb {
                            t_tripped.store(true, Ordering::SeqCst);
                            tracing::error!(
                                arcgraph.vector.rss_observed_mb = rss,
                                arcgraph.vector.rss_cap_mb = cap_mb,
                                "RSS guard tripped — process RSS exceeded the cap; \
                                 the run will abort cleanly at the next checkpoint \
                                 (ADR-195 §2.2)"
                            );
                        }
                    }
                    // Sleep in short slices so Drop's stop signal is honored
                    // promptly rather than after a full interval.
                    let mut slept = Duration::ZERO;
                    let slice = Duration::from_millis(50);
                    while slept < interval && !t_stop.load(Ordering::Relaxed) {
                        std::thread::sleep(slice.min(interval - slept));
                        slept += slice;
                    }
                }
            })
            .expect("spawn rss-guard thread");

        Self {
            cap_mb,
            tripped,
            observed_peak_mb,
            stop,
            handle: Some(handle),
        }
    }

    /// A guard that never trips (cap = `u64::MAX`, no sampler thread). Lets a
    /// caller run the identical guarded code path with the ceiling effectively
    /// removed (e.g., a small unit test).
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            cap_mb: u64::MAX,
            tripped: Arc::new(AtomicBool::new(false)),
            observed_peak_mb: Arc::new(AtomicU64::new(0)),
            stop: Arc::new(AtomicBool::new(true)),
            handle: None,
        }
    }

    /// Configured cap in MB.
    #[must_use]
    pub fn cap_mb(&self) -> u64 {
        self.cap_mb
    }

    /// Observed RSS peak in MB since the guard was armed (the honest
    /// high-water mark for the §6 report).
    #[must_use]
    pub fn peak_mb(&self) -> u64 {
        self.observed_peak_mb.load(Ordering::Relaxed)
    }

    /// `true` once the sampler has observed an over-cap RSS.
    #[must_use]
    pub fn is_tripped(&self) -> bool {
        self.tripped.load(Ordering::SeqCst)
    }

    /// Checkpoint: returns [`VectorIndexError::RssCapExceeded`] if the guard has
    /// tripped, else `Ok(())`. Callers poll this between build batches and
    /// between queries — the safe points at which a clean abort is possible.
    pub fn check(&self) -> Result<()> {
        if self.is_tripped() {
            return Err(VectorIndexError::RssCapExceeded {
                observed_mb: self.peak_mb(),
                cap_mb: self.cap_mb,
            });
        }
        Ok(())
    }
}

impl Drop for RssGuard {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl std::fmt::Debug for RssGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RssGuard")
            .field("cap_mb", &self.cap_mb)
            .field("peak_mb", &self.peak_mb())
            .field("tripped", &self.is_tripped())
            .finish()
    }
}

/// Current process resident-set size in MB, or `None` if the platform path is
/// unavailable. Safe (no FFI): Linux reads `/proc/self/statm`; macOS shells out
/// to `ps -o rss=`.
#[must_use]
pub fn current_rss_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        // `/proc/self/statm` field 1 (0-indexed) = resident set size in pages.
        // The system page size is 4096 on x86-64 / aarch64 Linux (the targets
        // arcgraph builds for); see the module note. Slight over/under-count
        // under huge pages is acceptable for a detect-and-abort backstop.
        let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
        let resident_pages: u64 = statm.split_whitespace().nth(1)?.parse().ok()?;
        Some(resident_pages * 4096 / (1024 * 1024))
    }
    #[cfg(target_os = "macos")]
    {
        // `ps -o rss= -p <pid>` prints the resident set size in KiB blocks.
        let pid = std::process::id();
        let out = std::process::Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok()?;
        if !out.status.success() {
            return None;
        }
        let kib: u64 = String::from_utf8_lossy(&out.stdout).trim().parse().ok()?;
        Some(kib / 1024)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_rss_is_plausible() {
        // The test process itself has a non-trivial RSS; the reader must return
        // Some(>0) on the supported platforms.
        if cfg!(any(target_os = "linux", target_os = "macos")) {
            let rss = current_rss_mb().expect("rss readable on this platform");
            assert!(rss > 0, "process RSS should be > 0 MB, got {rss}");
            // Sanity upper bound — a unit test process is well under 100 GB.
            assert!(rss < 100_000, "implausible RSS {rss} MB");
        }
    }

    #[test]
    fn disabled_guard_never_trips() {
        let g = RssGuard::disabled();
        assert!(!g.is_tripped());
        assert!(g.check().is_ok());
        assert_eq!(g.cap_mb(), u64::MAX);
    }

    #[test]
    fn guard_trips_on_tiny_cap_and_check_returns_clean_error() {
        // Fault injection (ADR-195 §2.2): a 0 MB cap is always exceeded by the
        // live process. The guard must latch + `check()` must return a CLEAN
        // RssCapExceeded error — NOT an OOM-kill / panic. This is the
        // load-bearing fault-injection assertion: the abort is observable and
        // recoverable, the process survives.
        let g = RssGuard::spawn(0, Duration::from_millis(20));
        // Poll for the trip (the sampler runs on its own thread).
        let mut tripped = false;
        for _ in 0..200 {
            if g.is_tripped() {
                tripped = true;
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(
            tripped,
            "guard with cap=0 must trip against the live process RSS"
        );
        match g.check() {
            Err(VectorIndexError::RssCapExceeded {
                observed_mb,
                cap_mb,
            }) => {
                assert_eq!(cap_mb, 0);
                assert!(observed_mb > 0, "observed peak should be the real RSS");
            }
            other => panic!("expected a clean RssCapExceeded, got {other:?}"),
        }
        // The process is still alive here — proving fail-CLEAN, not OOM-kill.
    }

    #[test]
    fn guard_records_peak() {
        let g = RssGuard::spawn(u64::MAX, Duration::from_millis(20));
        std::thread::sleep(Duration::from_millis(80));
        assert!(
            g.peak_mb() > 0,
            "peak should be recorded even under an infinite cap"
        );
        assert!(!g.is_tripped(), "infinite cap never trips");
    }
}
