//! M6.2 OOC-1 / INV-M6.9 — abort/restart scratch isolation plus the
//! measured data-volume headroom leg.
//!
//! This target is intentionally release-lane fault-injection coverage. The
//! hooks it consumes are bounded named-file retention and a deterministic
//! scratch/sweep rendezvous; stale rejection, epoch allocation, quota, and
//! headroom enforcement are production code.

#![cfg(feature = "fault-injection")]

use std::fs;

use arcgraph_core::TenantId;
use arcgraph_storage::spill::{
    SpillError, SpillManager, SpillManagerConfig, SpillQueryConfig, SpillRejectReason,
    VolumeHeadroom,
};

fn count_named_files(path: &std::path::Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .map(|entry| {
            let entry = entry.expect("read scratch entry");
            if entry.file_type().expect("scratch file type").is_dir() {
                count_named_files(&entry.path())
            } else {
                1
            }
        })
        .sum()
}

/// Deterministic RED-on-revert gate for the live-epoch scratch-directory
/// materialization race. The rendezvous puts the production periodic sweep
/// after tenant-directory inspection and before its permission check. Without
/// lifecycle serialization the sweep removes the empty tenant directory and
/// `create_run` fails with a spurious I/O error.
#[cfg(unix)]
#[test]
fn m6_spill_live_epoch_create_run_survives_concurrent_sweep() {
    let root = tempfile::tempdir().unwrap();
    let manager = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
    let query = manager
        .begin_query(SpillQueryConfig::new(
            TenantId::DEFAULT,
            0xC0FFEE,
            0,
            1024 * 1024,
        ))
        .unwrap();
    let barrier = manager.arm_sweep_create_barrier_for_test();

    let (create_result, sweep_result) = std::thread::scope(|scope| {
        let create = scope.spawn(|| query.create_run());
        barrier.wait_until_tenant_inspected();
        let sweep = scope.spawn(|| manager.periodic_sweep());
        (
            create.join().expect("create_run thread panicked"),
            sweep.join().expect("periodic_sweep thread panicked"),
        )
    });

    let writer = create_result.expect("live-epoch create_run must survive a concurrent sweep");
    sweep_result.expect("barrier-timed periodic sweep must succeed");
    drop(writer);
}

/// Decisive RED: if a rerun can pass its fresh epoch to an old attempt's
/// handle and read the old payload, this gate fails.
#[test]
fn m6_spill_abort_restart_isolated() {
    let root = tempfile::tempdir().unwrap();
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 2, false)
            .unwrap();

    let first = manager
        .begin_query(SpillQueryConfig::new(
            TenantId::DEFAULT,
            0xA11CE,
            0,
            1024 * 1024,
        ))
        .unwrap();
    let first_epoch = first.epoch();
    let mut writer = first.create_run().unwrap();
    writer.append_batch(b"attempt-zero-only").unwrap();
    let stale_run = writer.finish().unwrap();
    let stale_path = stale_run
        .retained_path_for_test()
        .expect("bounded retention must name this run")
        .to_path_buf();
    assert!(stale_path.exists());
    assert_eq!(stale_path.file_name().unwrap(), "run-0.spill");
    assert_eq!(
        stale_path.parent().unwrap().file_name().unwrap(),
        first_epoch.to_string().as_str()
    );
    assert_eq!(
        stale_path
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .file_name()
            .unwrap(),
        TenantId::DEFAULT.raw().to_string().as_str()
    );

    let mut ended_writer = first.create_run().unwrap();
    ended_writer.append_batch(b"ended-epoch-check").unwrap();
    let ended_run = ended_writer.finish().unwrap();
    let ended_path = ended_run
        .retained_path_for_test()
        .expect("second bounded retained run")
        .to_path_buf();

    // Abort ends the epoch and zeroizes any query key before the retry begins.
    drop(first);
    let retry = manager
        .begin_query(SpillQueryConfig::new(
            TenantId::DEFAULT,
            0xA11CE,
            1,
            1024 * 1024,
        ))
        .unwrap();
    assert!(retry.epoch().generation() > first_epoch.generation());
    assert_ne!(retry.epoch(), first_epoch);

    assert!(matches!(
        ended_run.into_reader(first_epoch),
        Err(SpillError::QueryEnded { epoch }) if epoch == first_epoch
    ));

    let error = stale_run.into_reader(retry.epoch()).unwrap_err();
    assert!(matches!(
        error,
        SpillError::StaleEpoch {
            active_epoch,
            run_epoch,
        } if active_epoch == retry.epoch() && run_epoch == first_epoch
    ));

    // The periodic fallback sweep preserves only the live retry epoch. It
    // deletes the retained old-attempt file instead of ever adopting it.
    let report = manager.periodic_sweep().unwrap();
    assert!(report.removed_files >= 2);
    assert!(!stale_path.exists());
    assert!(!ended_path.exists());
}

/// ENOSPC/headroom leg: set the configured floor to the volume's measured
/// currently-free bytes. Any positive reservation delta must be rejected
/// before O_EXCL creation; the typed error carries current spilled bytes.
#[test]
fn m6_spill_headroom_refuses_measured_delta_before_write() {
    let root = tempfile::tempdir().unwrap();
    let census = SpillManager::new(SpillManagerConfig::new(root.path())).unwrap();
    let census_space = census.volume_space().unwrap();
    let measured_free = census_space.available_bytes;
    drop(census);

    let mut config = SpillManagerConfig::new(root.path());
    config.volume_headroom = VolumeHeadroom::Bytes(measured_free);
    // Retention makes an accidental create-before-reserve observable as a
    // named file. The production order must leave zero files behind.
    let manager = SpillManager::new_with_fault_injection(config, 1, false).unwrap();
    let query = manager
        .begin_query(SpillQueryConfig::new(
            TenantId::DEFAULT,
            0xD15C,
            0,
            1024 * 1024,
        ))
        .unwrap();
    let error = query.create_run().unwrap_err();
    match error {
        SpillError::ResourceExhausted {
            reason: SpillRejectReason::VolumeHeadroom,
            requested_bytes,
            spilled_bytes,
            limit_bytes,
            available_bytes: Some(available_bytes),
            ..
        } => {
            assert!(
                requested_bytes > 0,
                "the real reservation delta is measured"
            );
            assert!(
                requested_bytes >= census_space.allocation_unit_bytes * 2,
                "headroom must charge scratch-directory allocation blocks"
            );
            assert_eq!(spilled_bytes, 0);
            assert_eq!(limit_bytes, measured_free);
            assert!(
                available_bytes.saturating_sub(requested_bytes) < limit_bytes,
                "observed free minus the requested real delta must breach the floor"
            );
        }
        other => panic!("expected typed volume-headroom rejection, got {other:?}"),
    }
    assert_eq!(
        count_named_files(manager.spill_root()),
        0,
        "reserve-before-write must reject without attempting a retained O_EXCL file"
    );
}

/// Disk-bomb regression: quota accounting covers the filesystem's measured
/// allocated blocks, rather than merely the serialized payload length.
#[cfg(unix)]
#[test]
fn m6_spill_accounting_covers_real_allocated_delta() {
    use std::os::unix::fs::MetadataExt;

    let root = tempfile::tempdir().unwrap();
    let manager =
        SpillManager::new_with_fault_injection(SpillManagerConfig::new(root.path()), 1, false)
            .unwrap();
    let unit = manager.volume_space().unwrap().allocation_unit_bytes;
    let mut config = SpillQueryConfig::new(TenantId::DEFAULT, 0xD15D, 0, 8 * 1024 * 1024);
    config.spill_quota_bytes = Some(8 * 1024 * 1024);
    let query = manager.begin_query(config).unwrap();
    let payload = vec![0x5A; 256 * 1024];
    let mut writer = query.create_run().unwrap();
    writer.append_batch(&payload).unwrap();
    let run = writer.finish().unwrap();
    let path = run.retained_path_for_test().unwrap().to_path_buf();
    let allocated = fs::metadata(&path).unwrap().blocks().saturating_mul(512);
    assert!(allocated >= payload.len() as u64);
    assert!(
        query.spilled_bytes() >= allocated.saturating_add(unit * 2),
        "charged bytes must cover real file blocks plus live query directories"
    );
    drop(run);
    drop(query);
    manager.periodic_sweep().unwrap();
    assert!(!path.exists());
}
