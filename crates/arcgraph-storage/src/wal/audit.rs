//! fsync barrier audit (roadmap M1-35, design-v2 §3.4).
//!
//! A classic durability trap: ext4 with `nobarrier` (or `barrier=0`)
//! and xfs with `nobarrier` silently disable write barriers. fsync
//! returns instantly, even though the data has not been flushed to
//! stable media. A crash at that point loses acknowledged writes.
//!
//! Startup runs [`audit_fsync_barriers`]; a nobarrier mount holding
//! the WAL directory is a hard refusal — the process exits with
//! [`ArcGraphError::UnsafeMountOptions`] rather than silently
//! risking committed data.
//!
//! On non-Linux platforms this is a no-op: macOS exposes
//! `F_FULLFSYNC` at the application level and does not offer a
//! mount-time barrier override; BSDs vary. We emit a
//! `tracing::debug!` rather than false-alarming a Linux mount option
//! on a platform that does not have it.

use std::path::Path;

use arcgraph_core::{ArcGraphError, Result};

/// Parsed view of one line from `/proc/mounts`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MountInfo {
    /// Device path (`/dev/nvme0n1p1`, etc.).
    pub device: String,
    /// Filesystem mount point.
    pub mountpoint: String,
    /// Filesystem type (`ext4`, `xfs`, `tmpfs`, ...).
    pub fstype: String,
    /// Comma-separated options.
    pub opts: String,
}

impl MountInfo {
    /// True if `opts` contains an option that matches `name` exactly,
    /// or starts with `name=`. `opts` is a comma-separated list.
    #[must_use]
    pub fn has_option(&self, name: &str) -> bool {
        self.opts.split(',').any(|o| {
            let head = o.split('=').next().unwrap_or(o);
            head == name
        })
    }

    /// True if `opts` contains `name=value` exactly.
    #[must_use]
    pub fn has_option_value(&self, name: &str, value: &str) -> bool {
        self.opts.split(',').any(|o| o == format!("{name}={value}"))
    }
}

/// Parse the content of `/proc/mounts` into a `Vec<MountInfo>`.
#[must_use]
pub fn parse_mounts(content: &str) -> Vec<MountInfo> {
    content
        .lines()
        .filter_map(|line| {
            let mut parts = line.split_whitespace();
            let device = parts.next()?.to_owned();
            let mountpoint = parts.next()?.to_owned();
            let fstype = parts.next()?.to_owned();
            let opts = parts.next()?.to_owned();
            Some(MountInfo {
                device,
                mountpoint,
                fstype,
                opts,
            })
        })
        .collect()
}

/// Find the mount point whose path is the longest prefix of
/// `canonical_path`. Exact match wins; otherwise the deepest ancestor.
#[must_use]
pub fn find_mount_for_path<'a>(
    canonical_path: &Path,
    mounts: &'a [MountInfo],
) -> Option<&'a MountInfo> {
    let target = canonical_path.to_string_lossy();
    let mut best: Option<&MountInfo> = None;
    for m in mounts {
        if target == m.mountpoint {
            return Some(m);
        }
        // `target` starts with `m.mountpoint` if m.mountpoint is a
        // parent directory of target. Require a path boundary to
        // avoid `/foo` matching `/foobar`.
        if target.starts_with(&m.mountpoint)
            && (m.mountpoint == "/" || target.as_bytes().get(m.mountpoint.len()) == Some(&b'/'))
            && best.is_none_or(|b| b.mountpoint.len() < m.mountpoint.len())
        {
            best = Some(m);
        }
    }
    best
}

/// Audit one mount's options for known unsafe settings. Returns
/// `Err(UnsafeMountOptions)` on a known trap; `Ok(())` otherwise.
/// Unknown filesystem types are accepted with a `tracing::debug!` —
/// we don't want to block startup on filesystems we haven't audited.
pub fn audit_mount(mount: &MountInfo) -> Result<()> {
    match mount.fstype.as_str() {
        "ext4" | "ext3" | "ext2" => {
            if mount.has_option("nobarrier") || mount.has_option_value("barrier", "0") {
                return Err(ArcGraphError::UnsafeMountOptions {
                    mountpoint: mount.mountpoint.clone(),
                    reason: format!(
                        "{} mounted with nobarrier / barrier=0 — fsync is a lie",
                        mount.fstype
                    ),
                });
            }
        }
        "xfs" => {
            if mount.has_option("nobarrier") {
                return Err(ArcGraphError::UnsafeMountOptions {
                    mountpoint: mount.mountpoint.clone(),
                    reason: "xfs mounted with nobarrier — fsync is a lie".to_owned(),
                });
            }
        }
        "btrfs" => {
            if mount.has_option("nobarrier") {
                return Err(ArcGraphError::UnsafeMountOptions {
                    mountpoint: mount.mountpoint.clone(),
                    reason: "btrfs mounted with nobarrier — fsync is a lie".to_owned(),
                });
            }
        }
        other => {
            tracing::debug!(fstype = other, "fsync audit: unknown fs type, skipped");
        }
    }
    Ok(())
}

/// Linux-only: audit the mount holding `path`. On non-Linux, this
/// function is a no-op that logs once and returns Ok.
#[cfg(target_os = "linux")]
pub fn audit_fsync_barriers(path: &Path) -> Result<()> {
    let canonical = path.canonicalize()?;
    let content = std::fs::read_to_string("/proc/mounts")?;
    let mounts = parse_mounts(&content);
    match find_mount_for_path(&canonical, &mounts) {
        Some(m) => audit_mount(m),
        None => {
            tracing::debug!(
                path = %canonical.display(),
                "fsync audit: no matching mount entry; skipping"
            );
            Ok(())
        }
    }
}

/// Non-Linux no-op. See module docs for the platform rationale.
#[cfg(not(target_os = "linux"))]
pub fn audit_fsync_barriers(_path: &Path) -> Result<()> {
    tracing::debug!("fsync barrier audit is a no-op on non-Linux");
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn mount(device: &str, mp: &str, fs: &str, opts: &str) -> MountInfo {
        MountInfo {
            device: device.to_owned(),
            mountpoint: mp.to_owned(),
            fstype: fs.to_owned(),
            opts: opts.to_owned(),
        }
    }

    // ---- parse_mounts ----

    #[test]
    fn parse_minimal_mounts_line() {
        let s = "/dev/nvme0n1p1 / ext4 rw,relatime,errors=remount-ro 0 0\n";
        let ms = parse_mounts(s);
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].device, "/dev/nvme0n1p1");
        assert_eq!(ms[0].mountpoint, "/");
        assert_eq!(ms[0].fstype, "ext4");
        assert_eq!(ms[0].opts, "rw,relatime,errors=remount-ro");
    }

    #[test]
    fn parse_skips_short_lines() {
        let s = "incomplete line\n\n/dev/sda1 /mnt ext4 rw 0 0\n";
        let ms = parse_mounts(s);
        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].mountpoint, "/mnt");
    }

    // ---- has_option ----

    #[test]
    fn has_option_matches_flag_and_kv() {
        let m = mount(
            "/dev/x",
            "/",
            "ext4",
            "rw,nobarrier,relatime,errors=remount-ro",
        );
        assert!(m.has_option("rw"));
        assert!(m.has_option("nobarrier"));
        assert!(m.has_option("errors"));
        assert!(!m.has_option("noexec"));
    }

    #[test]
    fn has_option_value_requires_exact_match() {
        let m = mount("/dev/x", "/", "ext4", "rw,barrier=0,errors=remount-ro");
        assert!(m.has_option_value("barrier", "0"));
        assert!(!m.has_option_value("barrier", "1"));
        assert!(!m.has_option_value("errors", "remount"));
    }

    // ---- audit_mount ----

    #[test]
    fn audit_ext4_nobarrier_fails() {
        let m = mount("/dev/x", "/data", "ext4", "rw,nobarrier");
        let err = audit_mount(&m).unwrap_err();
        assert!(matches!(err, ArcGraphError::UnsafeMountOptions { .. }));
    }

    #[test]
    fn audit_ext4_barrier_zero_fails() {
        let m = mount("/dev/x", "/data", "ext4", "rw,barrier=0");
        let err = audit_mount(&m).unwrap_err();
        assert!(matches!(err, ArcGraphError::UnsafeMountOptions { .. }));
    }

    #[test]
    fn audit_ext4_barrier_one_passes() {
        let m = mount("/dev/x", "/data", "ext4", "rw,barrier=1,relatime");
        assert!(audit_mount(&m).is_ok());
    }

    #[test]
    fn audit_xfs_nobarrier_fails() {
        let m = mount("/dev/x", "/data", "xfs", "rw,nobarrier");
        let err = audit_mount(&m).unwrap_err();
        assert!(matches!(err, ArcGraphError::UnsafeMountOptions { .. }));
    }

    #[test]
    fn audit_btrfs_nobarrier_fails() {
        let m = mount("/dev/x", "/data", "btrfs", "rw,nobarrier,subvol=/");
        let err = audit_mount(&m).unwrap_err();
        assert!(matches!(err, ArcGraphError::UnsafeMountOptions { .. }));
    }

    #[test]
    fn audit_unknown_fs_is_accepted() {
        let m = mount("none", "/proc", "proc", "rw,nosuid,nodev,noexec,relatime");
        assert!(audit_mount(&m).is_ok());
        let m = mount("tmp", "/tmp", "tmpfs", "rw,nosuid,nodev");
        assert!(audit_mount(&m).is_ok());
    }

    #[test]
    fn audit_clean_ext4_is_accepted() {
        let m = mount("/dev/sda1", "/var/lib/arcgraph", "ext4", "rw,relatime");
        assert!(audit_mount(&m).is_ok());
    }

    // ---- find_mount_for_path ----

    #[test]
    fn find_mount_matches_exact() {
        let mounts = vec![
            mount("/dev/sda1", "/", "ext4", "rw"),
            mount("/dev/sda2", "/var", "ext4", "rw"),
        ];
        let m = find_mount_for_path(&PathBuf::from("/var"), &mounts).unwrap();
        assert_eq!(m.mountpoint, "/var");
    }

    #[test]
    fn find_mount_picks_longest_prefix() {
        let mounts = vec![
            mount("/dev/sda1", "/", "ext4", "rw"),
            mount("/dev/sda2", "/var", "ext4", "rw"),
            mount("/dev/sda3", "/var/lib", "ext4", "rw"),
        ];
        let m = find_mount_for_path(&PathBuf::from("/var/lib/arcgraph/wal"), &mounts).unwrap();
        assert_eq!(m.mountpoint, "/var/lib");
    }

    #[test]
    fn find_mount_rejects_non_boundary_match() {
        // `/foo` must not match `/foobar`.
        let mounts = vec![
            mount("/dev/a", "/", "ext4", "rw"),
            mount("/dev/b", "/foo", "ext4", "rw"),
        ];
        let m = find_mount_for_path(&PathBuf::from("/foobar/x"), &mounts).unwrap();
        assert_eq!(m.mountpoint, "/", "must fall through to root, not /foo");
    }

    #[test]
    fn find_mount_handles_root() {
        let mounts = vec![mount("/dev/a", "/", "ext4", "rw")];
        let m = find_mount_for_path(&PathBuf::from("/anything/here"), &mounts).unwrap();
        assert_eq!(m.mountpoint, "/");
    }

    // ---- integration: simulated /proc/mounts ----

    #[test]
    fn integration_parse_and_audit_from_sample_mounts() {
        let sample = concat!(
            "proc /proc proc rw,nosuid,nodev,noexec,relatime 0 0\n",
            "/dev/nvme0n1p1 / ext4 rw,relatime,errors=remount-ro 0 0\n",
            "/dev/nvme0n1p2 /data ext4 rw,nobarrier,relatime 0 0\n",
            "/dev/nvme0n1p3 /fast xfs rw,nobarrier 0 0\n",
        );
        let mounts = parse_mounts(sample);
        // / is clean.
        let root = find_mount_for_path(&PathBuf::from("/usr/bin"), &mounts).unwrap();
        assert!(audit_mount(root).is_ok());
        // /data is ext4 with nobarrier.
        let data = find_mount_for_path(&PathBuf::from("/data/arcgraph/wal"), &mounts).unwrap();
        assert!(audit_mount(data).is_err());
        // /fast is xfs with nobarrier.
        let fast = find_mount_for_path(&PathBuf::from("/fast/db"), &mounts).unwrap();
        assert!(audit_mount(fast).is_err());
    }
}
