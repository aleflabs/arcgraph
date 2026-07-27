//! WAL segment files (roadmap M1-33).
//!
//! The WAL is a sequence of fixed-size segment files on disk:
//!
//! ```text
//! wal-0000000000.log
//! wal-0000000001.log
//! wal-0000000002.log
//! ...
//! ```
//!
//! Segments are append-only. A segment rotates when the next batch
//! would exceed `max_bytes` (default 64 MiB, design-v2 §4.2). A single
//! record never spans two segments — if a batch wouldn't fit, we
//! rotate *first* and write the batch into the fresh segment. A single
//! record larger than `max_bytes` is written into a segment of its
//! own; we emit a `tracing::warn!` in that case.
//!
//! **Segment header (8 B, stamped at segment creation)** —
//! `[magic=b"AGWL" (4 B)] [format_version u16 LE (2 B)] [reserved 2 B = 0]`.
//! The header is written before any record; recovery validates it on
//! open and returns
//! [`arcgraph_core::ArcGraphError::WalFormatMismatch`] on an unknown
//! version so operators see "upgrade required" instead of "WAL
//! corrupt". Tracked by issue #39.
//!
//! Records following the header are self-describing via the format
//! defined in `wal::record`. Recovery (M1-34) validates the segment
//! header, then decodes records linearly from `SegmentHeader::SIZE`.

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use arcgraph_core::{ArcGraphError, Lsn, Result};
use tracing::warn;

/// Magic bytes at offset 0 of every WAL segment file. ASCII "AGWL".
/// Lets a reader distinguish "right file type, wrong version" from
/// "wrong file type entirely". See design note §2.
pub const WAL_SEGMENT_MAGIC: [u8; 4] = *b"AGWL";

/// On-disk format version this binary stamps into every fresh
/// segment header. Bumped when any WAL-breaking change lands
/// (record-framing change, new mandatory field, etc.).
///
/// **ADR-032 Slice 2 cutover:** flipped from `1` to `2` when the
/// commit path started emitting `encode_commit_bundle_v2` payloads.
///
/// **ADR-031 amendment-02 / PR #79 X-2 fold-in:** flipped from `2`
/// to `3`. v3 generalises the `index_pages` section of v2 to a
/// unified `staged_pages` section with a one-byte
/// [`crate::wal::BundlePageKind`] discriminator per entry, so
/// record + BLOB pages now travel in the bundle alongside index
/// pages. The commit path in `crud::commit` extends the builder
/// to collect record-page + blob-page entries into the
/// `staged_pages` vec; replay routes each entry into the matching
/// in-memory page store via `PageStoreTarget`. Pre-amendment v2
/// and pre-Slice-1 v1 segments remain decodable.
///
/// **Issue #129 P0 fix:** flipped from `3` to `4`. v4 extends v3
/// with a trailing `allocator_advances` section so per-tenant
/// allocator high-water marks (`NodeId`, `RelId`, per-page-type
/// `PageId`) are durified atomically with the commit that
/// consumed them. Pre-fix `PageAllocator` and
/// `CrudStore.next_node` / `next_rel` reset to zero on WAL
/// recovery, causing post-fault `create_node` to re-issue NodeIds
/// pre-fault commits already consumed (orphaning earlier T1
/// commits through the primary index — ADR-034 D-1 violated).
/// Replay seeds each (tenant, kind) counter via
/// `seed_from_advance` to `max(current, observed)` (Lemma I3).
/// Pre-amendment v3, v2, and v1 segments remain decodable.
///
/// **M3.a Slice G.4 (commit-bundle vector page staging):** flipped
/// from `4` to `5`. v5 extends v4 with a trailing `vector_pages`
/// section so vector arena page mutations are durified atomically
/// with the commit that wrote them. The new section is the FIRST
/// production source of vector page replay (pre-v5 the
/// `BundlePageKind::Vector` arm was a stub with no producer).
/// Replay applies entries via
/// `VectorPageStoreHandle::install_or_replace` AFTER `staged_pages`
/// and BEFORE `allocator_advances` (Lemma I3 — monotonic
/// idempotent replay). Per ADR-031 amendment-02 + ADR-035
/// §4.5/§4.6. Pre-amendment v4, v3, v2, and v1 segments remain
/// decodable.
///
/// **#352 Part 2 (ADR-199):** bumped to `6`. v6 segments carry
/// CommitBundle v6 payloads (the trailing `idempotency_bindings`
/// section). A v6 binary opening a data dir whose latest segment is an
/// older version rolls a fresh v6-stamped segment rather than appending
/// (see [`SegmentWriter::open`]) so no segment is ever
/// version-inhomogeneous — replay dispatches the bundle codec by each
/// segment's `format_version`.
///
/// **#1221 (ADR-218):** bumped to `8`. v8 segments carry CommitBundle v8
/// payloads (the trailing `acl_grants` section that durifies
/// `PermissionIndex` grant/revoke ops so a bare `serve --data` restart
/// replays ACLs instead of coming up deny-all). Same homogeneous-segment
/// rollover discipline: a v8 binary opening a data dir whose latest
/// segment is v7 (or older) rolls a fresh v8-stamped segment.
pub const CURRENT_WAL_FORMAT_VERSION: u16 = 8;

/// Versions this binary knows how to read. Newer binaries add older
/// versions here iff backward compatibility is explicit per-version.
///
/// **ADR-031 amendment-02 / PR #79 X-2 fold-in:** extended to
/// `[1, 2, 3]`. v3 segments carry CommitBundle v3 payloads
/// (staged_pages + BundlePageKind). Dispatch among v1 / v2 / v3
/// codec happens in
/// [`crate::wal::bundle::decode_commit_bundle_for_version`] using
/// the segment header's `format_version`.
///
/// **Issue #129 P0 fix:** extended to `[1, 2, 3, 4]`. v4 segments
/// carry CommitBundle v4 payloads (allocator_advances tail). The
/// dispatcher routes V4 to `decode_commit_bundle_v4`.
///
/// **M3.a Slice G.4:** extended to `[1, 2, 3, 4, 5]`. v5 segments
/// carry CommitBundle v5 payloads (vector_pages tail). The
/// dispatcher routes V5 to `decode_commit_bundle_v5`.
///
/// **#352 Part 2 (ADR-199):** extended to `[1, 2, 3, 4, 5, 6]`. v6
/// segments carry CommitBundle v6 payloads (idempotency_bindings tail).
/// The dispatcher routes V6 to `decode_commit_bundle_v6`; a v6 binary
/// reading an older segment still decodes it (the older codec yields an
/// empty `idempotency_bindings`).
///
/// **#1010 (ADR-199 amendment):** extended to `[1, 2, 3, 4, 5, 6, 7]`.
/// v7 segments carry idempotency binding ops (`Install` / `Release`).
///
/// **#1221 (ADR-218):** extended to `[1, 2, 3, 4, 5, 6, 7, 8]`. v8
/// segments carry CommitBundle v8 payloads (the `acl_grants` tail). The
/// dispatcher routes V8 to `decode_commit_bundle_v8`; a v8 binary
/// reading an older segment still decodes it (the older codec yields an
/// empty `acl_grants` ⇒ no ACL ops ⇒ fail-closed).
///
/// **M3 staging:** v9 is readable before it becomes the current writer
/// format. The explicit v4→v5 generation migration flips the writer only
/// after a complete v9 generation exists, preventing migrate-on-open.
///
/// **M4 owner-row cutover:** v10 keeps the v9 byte layout but expands the
/// declared DeltaOp legality set with kinds 8/9. v9 remains readable with its
/// original M3 reservation fence.
pub const SUPPORTED_WAL_FORMAT_VERSIONS: &[u16] = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

/// Fixed-size header stamped at offset 0 of every WAL segment file.
///
/// Layout:
///
/// ```text
/// offset  field           size  notes
/// 0       magic           4     b"AGWL"
/// 4       format_version  2     u16 little-endian
/// 6       reserved        2     must be 0
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentHeader {
    /// WAL on-disk format version stamped in the header.
    pub format_version: u16,
}

impl SegmentHeader {
    /// Size of the encoded header in bytes.
    pub const SIZE: usize = 8;

    /// Header stamped by the current binary onto every new segment.
    #[must_use]
    pub const fn current() -> Self {
        Self {
            format_version: CURRENT_WAL_FORMAT_VERSION,
        }
    }

    /// Encode into a fixed-size byte buffer.
    #[must_use]
    pub fn encode(&self) -> [u8; Self::SIZE] {
        let mut out = [0u8; Self::SIZE];
        out[0..4].copy_from_slice(&WAL_SEGMENT_MAGIC);
        out[4..6].copy_from_slice(&self.format_version.to_le_bytes());
        // out[6..8] = reserved (zero)
        out
    }

    /// Parse a header from the start of `bytes`. Returns the decoded
    /// header on success.
    ///
    /// Errors are fail-fast and structurally distinct so the operator
    /// can tell them apart:
    ///
    /// - Too few bytes → [`ArcGraphError::WalFormatMismatch`] with
    ///   `found_version = 0` (an unsupported placeholder).
    /// - Magic mismatch → [`ArcGraphError::WalBadMagic`].
    /// - Version not in [`SUPPORTED_WAL_FORMAT_VERSIONS`] →
    ///   [`ArcGraphError::WalFormatMismatch`].
    /// - Non-zero reserved bytes → [`ArcGraphError::WalFormatMismatch`]
    ///   (future versions may reclaim these; today they must be zero).
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(ArcGraphError::WalFormatMismatch {
                found_version: 0,
                supported_versions: SUPPORTED_WAL_FORMAT_VERSIONS,
            });
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        if magic != WAL_SEGMENT_MAGIC {
            return Err(ArcGraphError::WalBadMagic {
                got: magic,
                expected: WAL_SEGMENT_MAGIC,
            });
        }
        let format_version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if bytes[6] != 0 || bytes[7] != 0 {
            return Err(ArcGraphError::WalFormatMismatch {
                found_version: format_version,
                supported_versions: SUPPORTED_WAL_FORMAT_VERSIONS,
            });
        }
        if !SUPPORTED_WAL_FORMAT_VERSIONS.contains(&format_version) {
            return Err(ArcGraphError::WalFormatMismatch {
                found_version: format_version,
                supported_versions: SUPPORTED_WAL_FORMAT_VERSIONS,
            });
        }
        Ok(Self { format_version })
    }
}

/// Filename prefix shared by every segment.
pub const SEGMENT_FILENAME_PREFIX: &str = "wal-";
/// Filename suffix shared by every segment.
pub const SEGMENT_FILENAME_SUFFIX: &str = ".log";

/// Build the filename for segment `n`. Zero-padded to 10 digits so
/// lexicographic directory order matches numeric order.
#[must_use]
pub fn segment_filename(n: u64) -> String {
    format!("{SEGMENT_FILENAME_PREFIX}{n:010}{SEGMENT_FILENAME_SUFFIX}")
}

/// Parse a segment filename back to its number. Returns `None` if the
/// filename is not in the canonical `wal-NNNNNNNNNN.log` shape.
#[must_use]
pub fn parse_segment_filename(name: &str) -> Option<u64> {
    let mid = name
        .strip_prefix(SEGMENT_FILENAME_PREFIX)?
        .strip_suffix(SEGMENT_FILENAME_SUFFIX)?;
    mid.parse::<u64>().ok()
}

/// List the existing segment numbers in `dir`, sorted ascending.
pub fn list_segments(dir: &Path) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    let read = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(ArcGraphError::Io(e)),
    };
    for entry in read {
        let entry = entry?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if let Some(n) = parse_segment_filename(&name) {
            out.push(n);
        }
    }
    out.sort_unstable();
    Ok(out)
}

/// Fsync a directory after creating, renaming, or truncating WAL-adjacent files.
///
/// POSIX does not make a newly-created dirent durable through the file's
/// fdatasync alone. Segment creation/rotation is rare relative to append, so
/// surfacing this error synchronously is the right durability tradeoff.
pub fn fsync_dir(dir: &Path) -> Result<()> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn fsync_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fsync_dir(parent)?;
    }
    Ok(())
}

/// Append-only segment writer. Owns a single `File` at a time; rotates
/// to a fresh segment when the next batch would exceed `max_bytes`.
#[derive(Debug)]
pub struct SegmentWriter {
    dir: PathBuf,
    current_seg: u64,
    current_file: File,
    current_bytes: u64,
    max_bytes: u64,
    format_version: u16,
}

impl SegmentWriter {
    /// Open (or create) the WAL directory and attach to the highest
    /// existing segment. If no segments exist, creates segment 0 and
    /// stamps a [`SegmentHeader`] (at [`CURRENT_WAL_FORMAT_VERSION`]) at
    /// offset 0. Existing segments have their header validated; an
    /// unsupported version returns [`ArcGraphError::WalFormatMismatch`]
    /// (not `WalCorruption`), so the operator sees "upgrade required"
    /// rather than "corrupt".
    ///
    /// **#352 Part 2 (ADR-199) — version-homogeneous segments.** Legacy
    /// formats older than [`CURRENT_WAL_FORMAT_VERSION`] roll a fresh
    /// current-format segment rather than mixing codecs. An offline M3
    /// migration deliberately creates a v9 first segment, while M4 derives
    /// v10 from its generation MANIFEST; each generation keeps that version
    /// across reopen and rotation. A segment's header version drives the
    /// bundle codec for every record in it.
    pub fn open(dir: impl AsRef<Path>, max_bytes: u64) -> Result<Self> {
        assert!(max_bytes > 0, "max_bytes must be > 0");
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir)?;
        fsync_parent_dir(&dir)?;
        fsync_dir(&dir)?;
        let existing = list_segments(&dir)?;
        let opened = match existing.last().copied() {
            Some(n) => {
                let path = dir.join(segment_filename(n));
                let file = OpenOptions::new().read(true).write(true).open(&path)?;
                let len = file.metadata()?.len();
                if (len as usize) < SegmentHeader::SIZE {
                    // Torn creation tail: file exists but the header
                    // never reached disk. No records can have been
                    // durable here (writes go at offset >= header).
                    // Rewrite the header and start fresh — symmetric
                    // with `WalRecoveryReader`'s terminal-short-tail
                    // behaviour.
                    let format_version = match existing.iter().rev().nth(1) {
                        Some(previous) => {
                            let previous_path = dir.join(segment_filename(*previous));
                            let previous_file = File::open(previous_path)?;
                            let mut bytes = [0u8; SegmentHeader::SIZE];
                            previous_file.read_exact_at(&mut bytes, 0)?;
                            let previous_header = SegmentHeader::decode(&bytes)?;
                            if matches!(previous_header.format_version, 9 | 10) {
                                previous_header.format_version
                            } else {
                                CURRENT_WAL_FORMAT_VERSION
                            }
                        }
                        None => generation_wal_format(&dir)?,
                    };
                    file.write_all_at(&SegmentHeader { format_version }.encode(), 0)?;
                    file.set_len(SegmentHeader::SIZE as u64)?;
                    file.sync_data()?;
                    (n, file, 0u64, format_version)
                } else {
                    // Full header: decode strictly. Unknown version or
                    // wrong magic is always propagated — fail-fast.
                    let mut header_buf = [0u8; SegmentHeader::SIZE];
                    file.read_exact_at(&mut header_buf, 0)?;
                    let header = SegmentHeader::decode(&header_buf)?;
                    if matches!(header.format_version, CURRENT_WAL_FORMAT_VERSION | 9 | 10) {
                        // current_bytes excludes the header: it counts
                        // record-data bytes, so `max_bytes` retains its
                        // record-capacity meaning and the "empty segment"
                        // gate in `append` still fires correctly on a
                        // fresh segment.
                        let record_bytes = len - SegmentHeader::SIZE as u64;
                        (n, file, record_bytes, header.format_version)
                    } else {
                        // #352 Part 2 (ADR-199): the latest on-disk
                        // segment was written at a DIFFERENT format
                        // version (e.g. a pre-upgrade v5 binary; this
                        // binary writes v6). Appending current-format
                        // records into a v{header}-stamped segment would
                        // make it version-inhomogeneous — on the next
                        // recovery the segment header's version drives
                        // the bundle codec for ALL of its records, so the
                        // new-format records would mis-decode. Roll to a
                        // fresh segment stamped at CURRENT; the old
                        // segment stays intact + version-homogeneous, and
                        // replay dispatches each segment by its own header
                        // version. (`n` is the highest existing segment,
                        // so `n + 1` cannot already exist — `create_new`
                        // is safe.)
                        let next = n + 1;
                        let path = dir.join(segment_filename(next));
                        let new_file = OpenOptions::new()
                            .read(true)
                            .write(true)
                            .create_new(true)
                            .open(&path)?;
                        new_file.write_all_at(&SegmentHeader::current().encode(), 0)?;
                        new_file.sync_data()?;
                        fsync_dir(&dir)?;
                        (next, new_file, 0, CURRENT_WAL_FORMAT_VERSION)
                    }
                }
            }
            None => {
                let path = dir.join(segment_filename(0));
                let file = OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create_new(true)
                    .open(&path)?;
                // An offline-migrated generation deliberately publishes an
                // empty WAL directory so no pre-swap record can cross the
                // generation boundary. Derive the first segment's codec from
                // the already-durable generation MANIFEST instead of silently
                // falling back to the legacy process default on first open.
                let format_version = generation_wal_format(&dir)?;
                let header_bytes = SegmentHeader { format_version }.encode();
                file.write_all_at(&header_bytes, 0)?;
                file.sync_data()?;
                fsync_dir(&dir)?;
                (0, file, 0, format_version)
            }
        };
        let (current_seg, current_file, current_bytes, format_version) = opened;
        Ok(Self {
            dir,
            current_seg,
            current_file,
            current_bytes,
            max_bytes,
            format_version,
        })
    }

    /// Directory this writer operates in.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Current segment number.
    #[must_use]
    pub fn current_segment(&self) -> u64 {
        self.current_seg
    }

    /// Record-data bytes written to the current segment so far,
    /// excluding the fixed [`SegmentHeader`] prefix. Total file
    /// length is `SegmentHeader::SIZE + current_bytes()`.
    #[must_use]
    pub fn current_bytes(&self) -> u64 {
        self.current_bytes
    }

    /// WAL bundle format emitted into this writer's active generation.
    /// Fresh and legacy data directories use v8; an offline-migrated
    /// generation whose first segment is v9 remains v9 across reopen and
    /// rotation.
    #[must_use]
    pub fn format_version(&self) -> u16 {
        self.format_version
    }

    /// Segment number a non-empty batch of `batch_bytes` would land
    /// in if passed to [`Self::append`] now.
    #[must_use]
    pub fn landing_segment_for(&self, batch_bytes: usize) -> u64 {
        let would_exceed = self.current_bytes.saturating_add(batch_bytes as u64) > self.max_bytes;
        if would_exceed && self.current_bytes > 0 {
            self.current_seg + 1
        } else {
            self.current_seg
        }
    }

    /// Append one batch of pre-encoded bytes. Rotates the segment
    /// first if this write would exceed `max_bytes` (and the current
    /// segment already holds at least one record).
    pub fn append(&mut self, bytes: &[u8]) -> Result<()> {
        if bytes.is_empty() {
            return Ok(());
        }
        let would_exceed = self.current_bytes.saturating_add(bytes.len() as u64) > self.max_bytes;
        if would_exceed && self.current_bytes > 0 {
            self.rotate()?;
        }
        if (bytes.len() as u64) > self.max_bytes {
            warn!(
                size = bytes.len(),
                max_bytes = self.max_bytes,
                "WAL batch exceeds max_bytes; writing into dedicated segment"
            );
        }
        // Records land after the fixed segment header; current_bytes
        // counts record-data only, so the file offset is header+data.
        let file_offset = SegmentHeader::SIZE as u64 + self.current_bytes;
        self.current_file.write_all_at(bytes, file_offset)?;
        self.current_bytes += bytes.len() as u64;
        Ok(())
    }

    /// fdatasync the current segment. Safe to call with nothing
    /// pending — still forwards to the OS.
    pub fn fsync(&self) -> Result<()> {
        self.current_file.sync_data()?;
        Ok(())
    }

    /// Close the current segment and open segment number `current+1`.
    /// The old file is fdatasync'd before handoff so partial writes
    /// never surface on recovery. The fresh segment gets its
    /// [`SegmentHeader`] stamped at offset 0 before any record.
    pub fn rotate(&mut self) -> Result<()> {
        self.current_file.sync_data()?;
        let next_seg = self.current_seg + 1;
        let path = self.dir.join(segment_filename(next_seg));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        let header_bytes = SegmentHeader {
            format_version: self.format_version,
        }
        .encode();
        file.write_all_at(&header_bytes, 0)?;
        file.sync_data()?;
        fsync_dir(&self.dir)?;
        self.current_file = file;
        self.current_seg = next_seg;
        self.current_bytes = 0;
        Ok(())
    }
}

fn generation_wal_format(wal_dir: &Path) -> Result<u16> {
    let Some(generation) = wal_dir.parent() else {
        return Ok(CURRENT_WAL_FORMAT_VERSION);
    };
    let manifest = crate::manifest::read_data_dir_manifest(generation).map_err(|error| {
        ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!("cannot derive torn segment format from MANIFEST: {error}"),
        }
    })?;
    let Some(manifest) = manifest else {
        return Ok(CURRENT_WAL_FORMAT_VERSION);
    };
    match manifest.wal_format.as_str() {
        crate::manifest::WAL_FORMAT_PAGE_IMAGE => Ok(CURRENT_WAL_FORMAT_VERSION),
        crate::manifest::WAL_FORMAT_DELTA_V9 => Ok(9),
        crate::manifest::WAL_FORMAT_DELTA_V10 => Ok(10),
        other => Err(ArcGraphError::WalCorruption {
            lsn: Lsn::ZERO,
            reason: format!("MANIFEST carries unsupported wal_format {other:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    // ---------- segment header codec ----------

    #[test]
    fn header_size_is_8() {
        assert_eq!(SegmentHeader::SIZE, 8);
    }

    #[test]
    fn header_current_is_version_8() {
        // #1221 (ADR-218): the writer stamps v8 on every fresh segment;
        // the commit path emits v8 bundles (staged_pages + BundlePageKind
        // + allocator_advances + vector_pages + idempotency_bindings +
        // acl_grants). The reader accepts v1..=v8 via
        // SUPPORTED_WAL_FORMAT_VERSIONS — dispatch in
        // `decode_commit_bundle_for_version`. Per ADR-031 amendment-02 +
        // ADR-035 §4.5/§4.6 (v5) + ADR-199 (v6/v7) + ADR-218 (v8).
        assert_eq!(SegmentHeader::current().format_version, 8);
        assert_eq!(CURRENT_WAL_FORMAT_VERSION, 8);
        assert_eq!(
            SUPPORTED_WAL_FORMAT_VERSIONS,
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        );
    }

    #[test]
    fn header_encode_layout() {
        // #1221 (ADR-218): CURRENT_WAL_FORMAT_VERSION = 8.
        let bytes = SegmentHeader::current().encode();
        assert_eq!(&bytes[0..4], &WAL_SEGMENT_MAGIC);
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), 8);
        assert_eq!(bytes[6], 0);
        assert_eq!(bytes[7], 0);
    }

    #[test]
    fn header_roundtrip() {
        let h = SegmentHeader::current();
        let bytes = h.encode();
        let back = SegmentHeader::decode(&bytes).unwrap();
        assert_eq!(back, h);
    }

    #[test]
    fn header_decode_rejects_too_short() {
        let err = SegmentHeader::decode(&[0u8; 4]).unwrap_err();
        match err {
            ArcGraphError::WalFormatMismatch {
                found_version,
                supported_versions,
            } => {
                assert_eq!(found_version, 0);
                assert_eq!(supported_versions, SUPPORTED_WAL_FORMAT_VERSIONS);
            }
            other => panic!("expected WalFormatMismatch, got {other:?}"),
        }
    }

    #[test]
    fn header_decode_rejects_wrong_magic() {
        let mut bytes = SegmentHeader::current().encode();
        bytes[0..4].copy_from_slice(b"XXXX");
        let err = SegmentHeader::decode(&bytes).unwrap_err();
        match err {
            ArcGraphError::WalBadMagic { got, expected } => {
                assert_eq!(&got, b"XXXX");
                assert_eq!(&expected, b"AGWL");
            }
            other => panic!("expected WalBadMagic, got {other:?}"),
        }
    }

    #[test]
    fn header_decode_rejects_unknown_version() {
        let mut bytes = SegmentHeader::current().encode();
        bytes[4..6].copy_from_slice(&999u16.to_le_bytes());
        let err = SegmentHeader::decode(&bytes).unwrap_err();
        match err {
            ArcGraphError::WalFormatMismatch {
                found_version,
                supported_versions,
            } => {
                assert_eq!(found_version, 999);
                assert_eq!(supported_versions, SUPPORTED_WAL_FORMAT_VERSIONS);
            }
            other => panic!("expected WalFormatMismatch, got {other:?}"),
        }
    }

    #[test]
    fn header_decode_rejects_nonzero_reserved() {
        let mut bytes = SegmentHeader::current().encode();
        bytes[6] = 0xAB;
        let err = SegmentHeader::decode(&bytes).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalFormatMismatch { .. }));
    }

    // ---------- existing segment-writer tests ----------

    #[test]
    fn filename_roundtrip() {
        assert_eq!(segment_filename(0), "wal-0000000000.log");
        assert_eq!(segment_filename(42), "wal-0000000042.log");
        assert_eq!(parse_segment_filename("wal-0000000000.log"), Some(0));
        assert_eq!(parse_segment_filename("wal-0000000042.log"), Some(42));
        // Over-long numbers parse as u64 if they fit; we keep them as-is.
        assert_eq!(
            parse_segment_filename("wal-000000000000000042.log"),
            Some(42)
        );
        assert_eq!(parse_segment_filename("not-a-wal.log"), None);
        assert_eq!(parse_segment_filename("wal-abc.log"), None);
    }

    #[test]
    fn open_creates_segment_zero() {
        let dir = tempdir().unwrap();
        let w = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
        assert_eq!(w.current_segment(), 0);
        // current_bytes counts record-data only — a fresh segment is
        // "empty" from the caller's perspective even though the file
        // already contains the 8-byte header.
        assert_eq!(w.current_bytes(), 0);
        assert!(dir.path().join("wal-0000000000.log").exists());
    }

    #[test]
    fn open_stamps_header_on_fresh_segment() {
        let dir = tempdir().unwrap();
        let _w = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
        let bytes = std::fs::read(dir.path().join(segment_filename(0))).unwrap();
        assert!(bytes.len() >= SegmentHeader::SIZE);
        let header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        assert_eq!(header.format_version, CURRENT_WAL_FORMAT_VERSION);
    }

    #[test]
    fn rotate_stamps_header_on_fresh_segment() {
        let dir = tempdir().unwrap();
        let mut w = SegmentWriter::open(dir.path(), 8).unwrap();
        w.append(b"abcd").unwrap();
        // Next append would exceed max_bytes → rotate before writing.
        w.append(b"EFGHI").unwrap();
        assert_eq!(w.current_segment(), 1);
        let bytes = std::fs::read(dir.path().join(segment_filename(1))).unwrap();
        let header = SegmentHeader::decode(&bytes[..SegmentHeader::SIZE]).unwrap();
        assert_eq!(header.format_version, CURRENT_WAL_FORMAT_VERSION);
    }

    #[test]
    fn reopen_rejects_unknown_version() {
        let dir = tempdir().unwrap();
        // Craft a segment with valid magic but an unknown version.
        let mut header = SegmentHeader::current().encode();
        header[4..6].copy_from_slice(&999u16.to_le_bytes());
        std::fs::write(dir.path().join(segment_filename(0)), header).unwrap();
        let err = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap_err();
        match err {
            ArcGraphError::WalFormatMismatch {
                found_version,
                supported_versions,
            } => {
                assert_eq!(found_version, 999);
                assert_eq!(supported_versions, SUPPORTED_WAL_FORMAT_VERSIONS);
            }
            other => panic!("expected WalFormatMismatch, got {other:?}"),
        }
    }

    #[test]
    fn open_rolls_fresh_segment_on_format_version_mismatch_352() {
        // #352 Part 2 (ADR-199) upgrade-safety: a v6 binary opening a data
        // dir whose latest segment was written at an OLDER (here v5)
        // version must ROLL a fresh segment stamped at CURRENT rather than
        // append (which would make the old segment version-inhomogeneous
        // and mis-decode on the next recovery). The old segment is left
        // intact at its own version; replay dispatches per-segment.
        let dir = tempdir().unwrap();
        // Craft segment 0 with a VALID v5 header + some record-area bytes,
        // as a pre-upgrade binary would have left it.
        let mut header5 = SegmentHeader::current().encode();
        header5[4..6].copy_from_slice(&5u16.to_le_bytes());
        let seg0 = dir.path().join(segment_filename(0));
        let mut bytes0 = header5.to_vec();
        bytes0.extend_from_slice(&[0xAB_u8; 64]);
        std::fs::write(&seg0, &bytes0).unwrap();

        // Open with the CURRENT (v6) writer.
        let w = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
        assert_eq!(
            w.current_segment(),
            1,
            "version mismatch must position the writer at a fresh segment 1"
        );
        assert_eq!(
            list_segments(dir.path()).unwrap(),
            vec![0, 1],
            "the old v5 segment 0 stays; a fresh segment 1 is rolled"
        );

        // Segment 0 is untouched — still v5.
        let mut buf0 = [0u8; SegmentHeader::SIZE];
        std::fs::File::open(&seg0)
            .unwrap()
            .read_exact_at(&mut buf0, 0)
            .unwrap();
        assert_eq!(
            SegmentHeader::decode(&buf0).unwrap().format_version,
            5,
            "old segment must keep its v5 header (version-homogeneous)"
        );

        // Segment 1 is stamped at CURRENT (v6).
        let mut buf1 = [0u8; SegmentHeader::SIZE];
        std::fs::File::open(dir.path().join(segment_filename(1)))
            .unwrap()
            .read_exact_at(&mut buf1, 0)
            .unwrap();
        assert_eq!(
            SegmentHeader::decode(&buf1).unwrap().format_version,
            CURRENT_WAL_FORMAT_VERSION,
            "fresh segment must be stamped at the current version"
        );
    }

    #[test]
    fn open_appends_to_same_version_segment_no_roll() {
        // Control: when the latest segment IS the current version, open
        // attaches to it (no spurious roll) — the common steady-state path.
        let dir = tempdir().unwrap();
        let seg0 = dir.path().join(segment_filename(0));
        let mut bytes0 = SegmentHeader::current().encode().to_vec();
        bytes0.extend_from_slice(&[0xCD_u8; 32]);
        std::fs::write(&seg0, &bytes0).unwrap();

        let w = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
        assert_eq!(
            w.current_segment(),
            0,
            "same-version segment is appended to, not rolled"
        );
        assert_eq!(list_segments(dir.path()).unwrap(), vec![0]);
    }

    #[test]
    fn migrated_v9_writer_attaches_and_rotates_v9() {
        let dir = tempdir().unwrap();
        let seg0 = dir.path().join(segment_filename(0));
        let mut bytes0 = SegmentHeader { format_version: 9 }.encode().to_vec();
        bytes0.extend_from_slice(b"v9-data");
        std::fs::write(&seg0, bytes0).unwrap();

        let mut writer = SegmentWriter::open(dir.path(), 8).unwrap();
        assert_eq!(writer.current_segment(), 0, "v9 must not roll back to v8");
        assert_eq!(writer.format_version(), 9);
        writer.append(b"more-v9").unwrap();

        let bytes1 = std::fs::read(dir.path().join(segment_filename(1))).unwrap();
        let header1 = SegmentHeader::decode(&bytes1[..SegmentHeader::SIZE]).unwrap();
        assert_eq!(header1.format_version, 9, "v9 rotation must remain v9");
    }

    #[test]
    fn migrated_v9_writer_repairs_torn_rotation_as_v9() {
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join(segment_filename(0)),
            SegmentHeader { format_version: 9 }.encode(),
        )
        .unwrap();
        std::fs::write(dir.path().join(segment_filename(1)), b"AGW").unwrap();

        let writer = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
        assert_eq!(writer.current_segment(), 1);
        assert_eq!(writer.format_version(), 9);
        let repaired = std::fs::read(dir.path().join(segment_filename(1))).unwrap();
        let header = SegmentHeader::decode(&repaired[..SegmentHeader::SIZE]).unwrap();
        assert_eq!(header.format_version, 9);
    }

    #[test]
    fn sole_torn_segment_uses_generation_manifest_format() {
        let root = tempdir().unwrap();
        let generation = root.path().join("gen-v9");
        let wal = generation.join("wal");
        std::fs::create_dir_all(&wal).unwrap();
        crate::manifest::write_data_dir_manifest(
            &generation,
            &crate::manifest::DataDirManifest::m3_delta_from(
                &crate::manifest::DataDirManifest::m2_typed("before".to_owned()),
                "after".to_owned(),
                Lsn::new(9),
            ),
        )
        .unwrap();
        std::fs::write(wal.join(segment_filename(7)), b"AGW").unwrap();

        let writer = SegmentWriter::open(&wal, 64 * 1024 * 1024).unwrap();
        assert_eq!(writer.format_version(), 9);
        let bytes = std::fs::read(wal.join(segment_filename(7))).unwrap();
        assert_eq!(
            SegmentHeader::decode(&bytes[..SegmentHeader::SIZE])
                .unwrap()
                .format_version,
            9
        );
    }

    #[test]
    fn reopen_rejects_bad_magic() {
        let dir = tempdir().unwrap();
        // Segment file with wrong magic.
        let mut bogus = SegmentHeader::current().encode();
        bogus[..4].copy_from_slice(b"XXXX");
        std::fs::write(dir.path().join(segment_filename(0)), bogus).unwrap();
        let err = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap_err();
        assert!(
            matches!(err, ArcGraphError::WalBadMagic { .. }),
            "expected WalBadMagic, got {err:?}"
        );
    }

    #[test]
    fn list_segments_is_sorted() {
        let dir = tempdir().unwrap();
        for n in [7u64, 3, 5, 1] {
            std::fs::write(dir.path().join(segment_filename(n)), b"x").unwrap();
        }
        assert_eq!(list_segments(dir.path()).unwrap(), vec![1, 3, 5, 7]);
    }

    #[test]
    fn fsync_dir_reports_missing_directory() {
        let dir = tempdir().unwrap();
        let missing = dir.path().join("missing");
        assert!(
            fsync_dir(&missing).is_err(),
            "directory fsync helper must propagate failures"
        );
    }

    #[test]
    fn reopen_attaches_to_highest_segment() {
        let dir = tempdir().unwrap();
        // Pretend two prior segments exist, each with a valid header
        // prefix (post-#39: SegmentWriter validates the header on
        // reopen).
        let h = SegmentHeader::current().encode();
        let mut s0 = h.to_vec();
        s0.extend_from_slice(b"hello");
        let mut s1 = h.to_vec();
        s1.extend_from_slice(b"world!!");
        std::fs::write(dir.path().join(segment_filename(0)), &s0).unwrap();
        std::fs::write(dir.path().join(segment_filename(1)), &s1).unwrap();
        let w = SegmentWriter::open(dir.path(), 64 * 1024 * 1024).unwrap();
        assert_eq!(w.current_segment(), 1);
        // current_bytes counts record-data only (excludes the header).
        assert_eq!(w.current_bytes(), 7);
    }

    #[test]
    fn append_then_rotate_on_size() {
        let dir = tempdir().unwrap();
        // Max 8 bytes of record data per segment — tiny for testing.
        // max_bytes is record-data capacity, not file capacity.
        let mut w = SegmentWriter::open(dir.path(), 8).unwrap();
        w.append(b"abcd").unwrap();
        assert_eq!(w.current_segment(), 0);
        assert_eq!(w.current_bytes(), 4);
        // Next 5-byte batch would push to 9 > 8 → rotate first.
        w.append(b"EFGHI").unwrap();
        assert_eq!(w.current_segment(), 1);
        assert_eq!(w.current_bytes(), 5);
        // First segment persisted header + its 4 bytes exactly.
        let h = SegmentHeader::current().encode();
        let s0 = std::fs::read(dir.path().join(segment_filename(0))).unwrap();
        assert_eq!(&s0[..SegmentHeader::SIZE], &h);
        assert_eq!(&s0[SegmentHeader::SIZE..], b"abcd");
        let s1 = std::fs::read(dir.path().join(segment_filename(1))).unwrap();
        assert_eq!(&s1[..SegmentHeader::SIZE], &h);
        assert_eq!(&s1[SegmentHeader::SIZE..], b"EFGHI");
    }

    #[test]
    fn empty_batch_is_noop() {
        let dir = tempdir().unwrap();
        let mut w = SegmentWriter::open(dir.path(), 16).unwrap();
        w.append(&[]).unwrap();
        assert_eq!(w.current_bytes(), 0);
    }

    #[test]
    fn single_record_larger_than_max_fits_in_fresh_segment() {
        let dir = tempdir().unwrap();
        let mut w = SegmentWriter::open(dir.path(), 8).unwrap();
        // Current segment is empty; a 100-byte batch must NOT rotate.
        let big = vec![0x42u8; 100];
        w.append(&big).unwrap();
        assert_eq!(w.current_segment(), 0);
        assert_eq!(w.current_bytes(), 100);
    }

    #[test]
    fn fsync_is_safe_on_empty() {
        let dir = tempdir().unwrap();
        let w = SegmentWriter::open(dir.path(), 16).unwrap();
        w.fsync().unwrap();
    }
}
