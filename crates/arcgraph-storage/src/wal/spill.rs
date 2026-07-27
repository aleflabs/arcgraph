//! ADR-032 §5 replay spill-to-disk.
//!
//! Spill files let the replay executor absorb pathological
//! commit_lsn ≠ WAL-LSN slack (§3 X-1) without using unbounded
//! memory. When the in-memory sorted buffer exceeds the configured
//! ceiling (default 8192 bundles / 1 GiB), the executor drains
//! the lowest-`commit_lsn` bundles into an append-only ARCGSPIL
//! file; the final drain re-reads the spill files and merges them
//! with the in-memory buffer in ascending `commit_lsn` order so
//! the apply stream is globally sorted regardless of arrival
//! order.
//!
//! # File format (ADR-032 §5)
//!
//! ```text
//!   0 .. 8   spill_header_magic: b"ARCGSPIL"
//!   8 ..10   spill_format_version: u16 LE = 1
//!  10 ..12   reserved: u16 = 0
//!  12 ..16   n_bundles: u32 LE
//!  16 ..24   min_commit_lsn: u64 LE
//!  24 ..32   max_commit_lsn: u64 LE
//!  32 ..36   reserved: u32 = 0
//!  36 ..     n_bundles × {
//!                u8 version                 — bundle codec version
//!                u64 LE primary_tenant      — header tenant used at decode time
//!                u32 LE payload_length
//!                [u8; payload_length]       — raw CommitBundle payload (no WAL framing)
//!            }
//!  last 4 bytes: crc32c over bytes 0 .. (size - 4)
//! ```
//!
//! The per-bundle `version` tag lets the reader pick the right bundle
//! decoder without having to recreate the owning WAL segment's
//! format_version. Spill writes the current WAL bundle payload so every
//! decoded bundle section survives the spill round-trip.
//!
//! # CRC-fail policy (§5 last paragraph)
//!
//! "A crash during spill produces a file whose trailing CRC does
//! not match; the next replay detects this and discards the
//! spill." We implement that: on load, a spill file whose CRC does
//! not match is skipped (with a `tracing::warn!`) — the buffered
//! bundles in the live WAL will be re-seen and re-buffered from
//! scratch on the next replay pass. No correctness risk: spill
//! is recoverable from WAL at any time.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use arcgraph_core::{ArcGraphError, Lsn, Result, TenantId};
use tracing::{debug, warn};

use crate::wal::bundle::{
    DecodedCommitBundle, VectorPageEntry, bounded_capacity, decode_commit_bundle_for_version,
    encode_commit_bundle_current,
};
use crate::wal::segment::{CURRENT_WAL_FORMAT_VERSION, fsync_dir};

// ─── On-disk format constants ────────────────────────────────────

/// ASCII magic bytes at spill-file offset 0. Disjoint from
/// [`crate::wal::WAL_SEGMENT_MAGIC`] (`b"AGWL"`).
const SPILL_MAGIC: &[u8; 8] = b"ARCGSPIL";

/// On-disk spill-file format version.
const SPILL_FORMAT_VERSION: u16 = 1;

/// Fixed-size file header in bytes (offsets 0..36 per module doc).
const SPILL_HEADER_SIZE: usize = 36;

/// Trailing CRC size (u32 LE, crc32c over bytes 0..(size-4)).
const SPILL_TRAILING_CRC_SIZE: usize = 4;

/// Extension stamped on spill files. `discard_spill_dir` removes
/// every entry with this suffix.
const SPILL_FILE_EXTENSION: &str = "spill";

/// Smallest on-wire size of ONE spilled bundle entry header, in bytes
/// (#1411): `u8 version (1) + u64 primary_tenant (8) + u32 payload_len
/// (4)` = 13. `payload_len` may be 0, so this is the floor of a single
/// entry. Matches the in-loop truncation guard `cursor + 13 > crc_offset`
/// in [`load_one_spill_file`]; used to bound the entry `Vec` capacity
/// hint against the untrusted `n_bundles` header field so a crafted count
/// cannot force an unbounded pre-alloc (OOM) before the first entry byte
/// is validated.
const MIN_BUNDLE_ELEM: usize = 1 + 8 + 4;

// ─── Writing ─────────────────────────────────────────────────────

/// Write a batch of decoded bundles to a fresh spill file.
///
/// Returns the path of the new file. The batch is sorted by
/// `commit_lsn` at write time so the load path can stream the
/// file without sorting; the caller-supplied slice MAY be
/// unordered.
///
/// The write path:
///
/// 1. Creates `dir` if missing.
/// 2. Writes the 36-byte header with the batch's min/max
///    `commit_lsn`.
/// 3. Serializes every bundle via [`encode_commit_bundle_current`], preserving
///    every section the replay executor may later need to apply.
/// 4. Writes a trailing crc32c over the full file contents.
/// 5. Fsyncs the file so a crash between step 4 and step 6
///    leaves a detectable torn file.
/// 6. Renames to the final `.spill` name (the "creation barrier":
///    a filename without the `.spill` extension is never visible
///    to the load path).
pub fn write_spill_batch(dir: &Path, bundles: &[DecodedCommitBundle]) -> Result<PathBuf> {
    if bundles.is_empty() {
        return Ok(dir.join(format!("empty.{SPILL_FILE_EXTENSION}")));
    }
    fs::create_dir_all(dir)?;

    // Sort by commit_lsn (ascending) so the on-disk entries are
    // in globally-sorted order. Cloning decoded bundles is O(N)
    // but the spill path is amortized.
    let mut sorted: Vec<&DecodedCommitBundle> = bundles.iter().collect();
    sorted.sort_by_key(|b| b.commit_lsn.raw());

    let min_lsn = sorted.first().map(|b| b.commit_lsn.raw()).unwrap_or(0);
    let max_lsn = sorted.last().map(|b| b.commit_lsn.raw()).unwrap_or(0);

    let tmp_name = format!(
        "arcgspil-{}-{}-{}.tmp",
        std::process::id(),
        min_lsn,
        max_lsn,
    );
    let final_name = format!("arcgspil-{min_lsn}-{max_lsn}.{SPILL_FILE_EXTENSION}");
    let tmp_path = dir.join(tmp_name);
    let final_path = dir.join(final_name);

    // Build the whole file in memory so the trailing CRC is
    // computed once over a consistent buffer. Spill batches are
    // bounded by the buffer byte ceiling (1 GiB default); at
    // realistic workload sizes the spill batch is a few MiB.
    let mut buf = Vec::with_capacity(SPILL_HEADER_SIZE + bundles.len() * 8192);
    // Header
    buf.extend_from_slice(SPILL_MAGIC);
    buf.extend_from_slice(&SPILL_FORMAT_VERSION.to_le_bytes());
    buf.extend_from_slice(&0u16.to_le_bytes()); // reserved
    let n_bundles = u32::try_from(sorted.len()).unwrap_or(u32::MAX);
    buf.extend_from_slice(&n_bundles.to_le_bytes());
    buf.extend_from_slice(&min_lsn.to_le_bytes());
    buf.extend_from_slice(&max_lsn.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
    debug_assert_eq!(buf.len(), SPILL_HEADER_SIZE);

    // Entries
    for bundle in &sorted {
        // Re-encode with the current bundle codec. Older hard-pins here
        // silently drop newer decoded sections for bundles that cross the
        // spill path.
        let staged: Vec<(
            crate::wal::bundle::BundlePageKind,
            arcgraph_core::PageId,
            arcgraph_core::TenantId,
            Box<[u8; arcgraph_core::PAGE_SIZE]>,
        )> = bundle
            .staged_pages
            .iter()
            .map(|p| (p.kind, p.page_id, p.tenant_id, p.bytes.clone()))
            .collect();
        let vector_pages: Vec<VectorPageEntry> = bundle.vector_pages.clone();
        let payload = encode_commit_bundle_current(
            bundle.commit_lsn,
            bundle.primary_tenant,
            &bundle.mvcc_writes,
            &bundle.sidechannel_writes,
            &staged,
            &bundle.allocator_advances,
            &vector_pages,
            &bundle.idempotency_bindings,
            &bundle.acl_grants,
        );

        // Per-entry framing.
        let version =
            u8::try_from(CURRENT_WAL_FORMAT_VERSION).map_err(|_| ArcGraphError::WalCorruption {
                lsn: bundle.commit_lsn,
                reason: format!(
                    "current WAL bundle format {} does not fit spill framing",
                    CURRENT_WAL_FORMAT_VERSION
                ),
            })?;
        buf.push(version);
        buf.extend_from_slice(&bundle.primary_tenant.raw().to_le_bytes());
        let len = u32::try_from(payload.len()).unwrap_or(u32::MAX);
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(&payload);
    }

    // Trailing CRC
    let crc = crc32c::crc32c(&buf);
    buf.extend_from_slice(&crc.to_le_bytes());

    // Write + fsync to the .tmp path; then rename atomically.
    {
        let mut f = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp_path)?;
        f.write_all(&buf)?;
        f.sync_all()?;
    }
    fs::rename(&tmp_path, &final_path)?;
    // Sync the directory so the rename is durable.
    fsync_dir(dir)?;

    debug!(
        path = ?final_path,
        n_bundles = n_bundles,
        bytes = buf.len(),
        min_lsn,
        max_lsn,
        "wrote spill file"
    );
    Ok(final_path)
}

// ─── Reading ────────────────────────────────────────────────────

/// Load every spill file in `dir` and return the decoded bundles.
///
/// Order is NOT guaranteed — callers merge-sort via the BTreeMap
/// in the replay buffer. Spill files with invalid magic, mismatched
/// CRC, or unreadable framing are **skipped with a warn** per §5
/// last paragraph (a crashed prior replay's spill file is
/// recoverable from WAL).
pub fn load_all_spill_bundles(dir: &Path) -> Result<Vec<DecodedCommitBundle>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some(SPILL_FILE_EXTENSION) {
            continue;
        }
        match load_one_spill_file(&path) {
            Ok(bundles) => out.extend(bundles),
            Err(e) => {
                warn!(
                    path = ?path,
                    error = ?e,
                    "skipping corrupt spill file (will re-buffer from WAL)",
                );
            }
        }
    }
    Ok(out)
}

/// Count spill files currently in `dir`. Used by the
/// `wal_replay_spill_files_reloaded` gauge when the executor
/// opens a WAL directory with pre-existing spill files from a
/// crashed prior replay.
pub fn count_spill_files(dir: &Path) -> Result<usize> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut n = 0;
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        if entry.path().extension().and_then(|s| s.to_str()) == Some(SPILL_FILE_EXTENSION) {
            n += 1;
        }
    }
    Ok(n)
}

/// Remove every spill file from `dir`, then remove the directory
/// itself if it is empty. Called on successful replay completion.
/// Non-fatal on failure (warn + continue).
pub fn discard_spill_dir(dir: &Path) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some(SPILL_FILE_EXTENSION)
            || path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.ends_with(".tmp"))
                .unwrap_or(false)
        {
            let _ = fs::remove_file(&path);
        }
    }
    // Best effort: remove empty dir.
    let _ = fs::remove_dir(dir);
    Ok(())
}

// ─── Internals ───────────────────────────────────────────────────

/// Load and decode a single spill file, returning its bundles or a
/// [`ArcGraphError::WalCorruption`] on any framing/CRC violation.
///
/// `pub` (not the private helper it once was) so the #1411 spill-OOM
/// regression test can assert the untrusted `n_bundles` header field is
/// bounded before pre-allocation — the caller [`load_all_spill_bundles`]
/// intentionally swallows this `Err` (warn + skip per §5), so the error
/// is not observable through the public batch entry.
pub fn load_one_spill_file(path: &Path) -> Result<Vec<DecodedCommitBundle>> {
    let mut f = File::open(path)?;
    // Read full file for CRC validation + parse. Spill files are
    // bounded by the executor's byte ceiling so in-memory parse is
    // reasonable at replay time.
    let meta = f.metadata()?;
    let total = meta.len() as usize;
    if total < SPILL_HEADER_SIZE + SPILL_TRAILING_CRC_SIZE {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!("spill {path:?}: too short ({total} bytes)"),
        });
    }
    let mut buf = vec![0u8; total];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut buf)?;

    // Verify trailing CRC.
    let crc_offset = total - SPILL_TRAILING_CRC_SIZE;
    let crc_stored = u32::from_le_bytes(buf[crc_offset..crc_offset + 4].try_into().unwrap());
    let crc_computed = crc32c::crc32c(&buf[..crc_offset]);
    if crc_stored != crc_computed {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!(
                "spill {path:?}: crc mismatch (stored {crc_stored:#x}, computed {crc_computed:#x})",
            ),
        });
    }

    // Parse header.
    if &buf[0..8] != SPILL_MAGIC {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!("spill {path:?}: bad magic"),
        });
    }
    let format_version = u16::from_le_bytes([buf[8], buf[9]]);
    if format_version != SPILL_FORMAT_VERSION {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!("spill {path:?}: unsupported format_version {format_version}"),
        });
    }
    let n_bundles = u32::from_le_bytes(buf[12..16].try_into().unwrap()) as usize;

    let mut cursor = SPILL_HEADER_SIZE;
    // #1411: `n_bundles` is an untrusted u32 read straight off the spill
    // header (bytes 12..16). Cap the capacity HINT at the number of
    // minimum-size entries the remaining (post-header, pre-crc) region
    // could hold — a crafted `n_bundles` (e.g. u32::MAX) is clamped
    // instead of forcing a multi-gigabyte pre-alloc. A valid file with
    // `n_bundles` genuine entries has `>= n_bundles * MIN_BUNDLE_ELEM`
    // remaining bytes, so the cap equals `n_bundles` (zero behavior
    // change; the Vec still grows on push). A malicious count hits the
    // existing in-loop `cursor + 13 > crc_offset` guard → WalCorruption,
    // without OOM first.
    let bundles_cap = bounded_capacity(
        n_bundles,
        crc_offset.saturating_sub(cursor),
        MIN_BUNDLE_ELEM,
    );
    let mut out = Vec::with_capacity(bundles_cap);
    for _ in 0..n_bundles {
        if cursor + 13 > crc_offset {
            return Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!("spill {path:?}: truncated entry header"),
            });
        }
        let version = buf[cursor];
        cursor += 1;
        let tenant_raw = u64::from_le_bytes(buf[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let payload_len = u32::from_le_bytes(buf[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;
        if cursor + payload_len > crc_offset {
            return Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!(
                    "spill {path:?}: entry payload overruns trailer ({} + {} > {})",
                    cursor, payload_len, crc_offset
                ),
            });
        }
        let payload_slice = &buf[cursor..cursor + payload_len];
        cursor += payload_len;
        let bundle = decode_commit_bundle_for_version(
            payload_slice,
            u16::from(version),
            TenantId::new(tenant_raw),
        )?;
        out.push(bundle);
    }
    if cursor != crc_offset {
        return Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!(
                "spill {path:?}: {} unparsed bytes before CRC",
                crc_offset - cursor,
            ),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wal::bundle::{
        AclGrantEntry, AclGrantOp, SideChannelWrite, StagedEmit, encode_commit_bundle_v2,
    };
    use arcgraph_core::{NodeId, PAGE_SIZE};
    use arcgraph_core::{PageId, TenantId};
    use bytes::Bytes;
    use std::collections::{BTreeSet, HashMap};
    use tempfile::tempdir;

    fn mk_bundle(
        commit_lsn: u64,
        primary_tenant: TenantId,
        writes: &[(u64, Option<&[u8]>)],
        sidechannel: &[SideChannelWrite],
        index_pages: &[(u64, TenantId, u8)],
    ) -> DecodedCommitBundle {
        // Build via the v2 encoder + decoder so we're round-tripping
        // through the same machinery the spill uses.
        let mvcc: HashMap<u64, Option<Bytes>> = writes
            .iter()
            .map(|(k, v)| (*k, v.map(Bytes::copy_from_slice)))
            .collect();
        let mut staged_emits: Vec<StagedEmit> = Vec::new();
        for (pid, _t, fill) in index_pages {
            staged_emits.push(StagedEmit {
                kind: crate::wal::bundle::BundlePageKind::PrimaryIndex,
                page_id: PageId::new(*pid),
                bytes: Box::new([*fill; PAGE_SIZE]),
            });
        }
        // Figure out a staged_tenant: if all index_pages share a
        // tenant, use it; otherwise use the first.
        let staged_tenant = index_pages
            .first()
            .map(|(_, t, _)| *t)
            .unwrap_or(primary_tenant);
        let payload = encode_commit_bundle_v2(
            Lsn::new(commit_lsn),
            primary_tenant,
            &mvcc,
            sidechannel,
            &staged_emits,
            staged_tenant,
        );
        crate::wal::bundle::decode_commit_bundle_v2(&payload, primary_tenant).unwrap()
    }

    fn grant_set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| (*s).to_owned()).collect()
    }

    #[test]
    fn round_trip_empty_spill_file() {
        let dir = tempdir().unwrap();
        // write + read an empty batch
        let path = write_spill_batch(dir.path(), &[]).unwrap();
        // empty-batch short-circuits: no file created.
        assert!(!path.exists());
        let reload = load_all_spill_bundles(dir.path()).unwrap();
        assert!(reload.is_empty());
    }

    #[test]
    fn round_trip_single_bundle() {
        let dir = tempdir().unwrap();
        let bundle = mk_bundle(
            42,
            TenantId::DEFAULT,
            &[(1u64, Some(b"hello"))],
            &[],
            &[(100u64, TenantId::SYSTEM, 0xAB)],
        );
        let path = write_spill_batch(dir.path(), std::slice::from_ref(&bundle)).unwrap();
        assert!(path.exists());
        let reload = load_all_spill_bundles(dir.path()).unwrap();
        assert_eq!(reload.len(), 1);
        assert_eq!(reload[0].commit_lsn, Lsn::new(42));
        assert_eq!(reload[0].primary_tenant, TenantId::DEFAULT);
        assert_eq!(reload[0].mvcc_writes.len(), 1);
        assert_eq!(reload[0].staged_pages.len(), 1);
        assert_eq!(reload[0].staged_pages[0].page_id.raw(), 100);
    }

    #[test]
    fn round_trip_multiple_bundles_unordered_input_sorted_on_disk() {
        let dir = tempdir().unwrap();
        let b1 = mk_bundle(10, TenantId::DEFAULT, &[(1u64, Some(b"a"))], &[], &[]);
        let b2 = mk_bundle(5, TenantId::DEFAULT, &[(2u64, Some(b"b"))], &[], &[]);
        let b3 = mk_bundle(20, TenantId::DEFAULT, &[(3u64, Some(b"c"))], &[], &[]);
        let _path = write_spill_batch(dir.path(), &[b1, b2, b3]).unwrap();
        let reload = load_all_spill_bundles(dir.path()).unwrap();
        assert_eq!(reload.len(), 3);
        // Load order reflects on-disk order (min-LSN first).
        assert_eq!(reload[0].commit_lsn, Lsn::new(5));
        assert_eq!(reload[1].commit_lsn, Lsn::new(10));
        assert_eq!(reload[2].commit_lsn, Lsn::new(20));
    }

    #[test]
    fn round_trip_bundle_with_sidechannel() {
        let dir = tempdir().unwrap();
        let bundle = mk_bundle(
            7,
            TenantId::DEFAULT,
            &[(1u64, Some(b"user"))],
            &[SideChannelWrite {
                tenant_id: TenantId::SYSTEM,
                key: 99,
                value: Some(Bytes::from_static(b"root-ptr")),
            }],
            &[],
        );
        let _path = write_spill_batch(dir.path(), std::slice::from_ref(&bundle)).unwrap();
        let reload = load_all_spill_bundles(dir.path()).unwrap();
        assert_eq!(reload.len(), 1);
        assert_eq!(reload[0].sidechannel_writes.len(), 1);
        assert_eq!(reload[0].sidechannel_writes[0].tenant_id, TenantId::SYSTEM);
        assert_eq!(reload[0].sidechannel_writes[0].key, 99);
    }

    #[test]
    fn round_trip_bundle_preserves_acl_apply_and_revoke_entries() {
        let dir = tempdir().unwrap();
        let mut bundle = mk_bundle(88, TenantId::DEFAULT, &[(1u64, Some(b"acl"))], &[], &[]);
        bundle.acl_grants = vec![
            AclGrantEntry {
                op: AclGrantOp::Apply,
                tenant: TenantId::DEFAULT,
                doc: NodeId::new(1001),
                grants: grant_set(&["principal:alice", "role:reader"]),
            },
            AclGrantEntry {
                op: AclGrantOp::Revoke,
                tenant: TenantId::DEFAULT,
                doc: NodeId::new(1002),
                grants: BTreeSet::new(),
            },
        ];
        let expected = bundle.acl_grants.clone();

        let _path = write_spill_batch(dir.path(), std::slice::from_ref(&bundle)).unwrap();
        let reload = load_all_spill_bundles(dir.path()).unwrap();

        assert_eq!(reload.len(), 1);
        assert_eq!(reload[0].acl_grants, expected);
    }

    #[test]
    fn crc_mismatch_is_skipped_not_errored() {
        let dir = tempdir().unwrap();
        let bundle = mk_bundle(1, TenantId::DEFAULT, &[(1u64, Some(b"x"))], &[], &[]);
        let path = write_spill_batch(dir.path(), std::slice::from_ref(&bundle)).unwrap();
        // Flip a byte inside the payload to produce a CRC
        // mismatch. The spill load path should warn + skip,
        // returning an empty Vec — NOT an error.
        use std::os::unix::fs::FileExt;
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut b = [0u8; 1];
        f.read_exact_at(&mut b, (SPILL_HEADER_SIZE + 4) as u64)
            .unwrap();
        b[0] ^= 0x80;
        f.write_all_at(&b, (SPILL_HEADER_SIZE + 4) as u64).unwrap();
        drop(f);

        let reload = load_all_spill_bundles(dir.path()).unwrap();
        assert!(reload.is_empty(), "CRC-fail spill must be silently skipped");
    }

    #[test]
    fn count_and_discard_spill_files() {
        let dir = tempdir().unwrap();
        let b1 = mk_bundle(1, TenantId::DEFAULT, &[(1u64, None)], &[], &[]);
        let b2 = mk_bundle(2, TenantId::DEFAULT, &[(2u64, None)], &[], &[]);
        write_spill_batch(dir.path(), std::slice::from_ref(&b1)).unwrap();
        write_spill_batch(dir.path(), std::slice::from_ref(&b2)).unwrap();
        assert_eq!(count_spill_files(dir.path()).unwrap(), 2);
        discard_spill_dir(dir.path()).unwrap();
        // Dir removed; count returns 0.
        assert_eq!(count_spill_files(dir.path()).unwrap(), 0);
    }

    #[test]
    fn empty_dir_returns_zero_spill_files() {
        let dir = tempdir().unwrap();
        assert_eq!(count_spill_files(dir.path()).unwrap(), 0);
        let reload = load_all_spill_bundles(dir.path()).unwrap();
        assert!(reload.is_empty());
    }

    #[test]
    fn nonexistent_dir_is_not_an_error() {
        let dir = tempdir().unwrap();
        let phantom = dir.path().join("does-not-exist");
        assert!(!phantom.exists());
        assert_eq!(count_spill_files(&phantom).unwrap(), 0);
        let reload = load_all_spill_bundles(&phantom).unwrap();
        assert!(reload.is_empty());
        discard_spill_dir(&phantom).unwrap();
    }
}
