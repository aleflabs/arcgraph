//! Immutable overflow payloads for fixed-size M4 owner rows.
//!
//! Common external ids and interned names stay inline in the 256-byte owner
//! row. Larger strings/grant sets are written and fsync'd here *before* the
//! row's WAL commit. A crash may leave an unreachable overflow image, but can
//! never leave a committed row pointing at bytes that were not durable first.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use thiserror::Error;

use crate::owner_row::OWNER_ROW_MAX_PAYLOAD;

const ROW_REF_MAGIC: &[u8; 4] = b"AGLP";
const OVERFLOW_MAGIC: &[u8; 4] = b"AGOV";
const ROW_REF_HEADER_BYTES: usize = 24;
const OVERFLOW_HEADER_BYTES: u64 = 16;
const MODE_INLINE: u8 = 0;
const MODE_OVERFLOW: u8 = 1;

/// Maximum payload bytes in one companion file.
pub const OWNER_PAYLOAD_DISK_CAP_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Typed immutable-payload failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OwnerPayloadError {
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Row reference or overflow record is malformed.
    #[error("owner payload is corrupt: {0}")]
    Corrupt(String),
    /// One logical value exceeds the supported u32 record length.
    #[error("owner logical payload length {0} exceeds u32")]
    PayloadTooLarge(usize),
    /// Append would exceed the bounded companion-file budget.
    #[error(
        "owner payload disk budget exceeded: current={current} additional={additional} cap={cap}"
    )]
    DiskBudgetExceeded {
        /// Current file bytes.
        current: u64,
        /// Record bytes requested.
        additional: u64,
        /// Hard ceiling.
        cap: u64,
    },
}

/// One class-scoped immutable payload companion.
#[derive(Debug)]
pub struct OwnerPayloadStore {
    file: parking_lot::Mutex<File>,
    disk_cap_bytes: u64,
}

impl OwnerPayloadStore {
    /// Create a new companion for an invisible generation build.
    pub fn create(path: &Path, disk_cap_bytes: u64) -> Result<Self, OwnerPayloadError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(path)?;
        file.sync_all()?;
        sync_parent(path)?;
        Ok(Self {
            file: parking_lot::Mutex::new(file),
            disk_cap_bytes,
        })
    }

    /// Open an already-published companion for the SERVE / incremental
    /// path. Missing files are corruption at the complete-generation
    /// boundary and surface as typed I/O errors.
    ///
    /// `disk_cap_bytes` is the ABSOLUTE ceiling on this companion's file
    /// size (pre-M5-D3, unchanged semantics — every recovery/boot/serve
    /// caller goes through this path via [`crate::owner_row::OwnerRowRegistry::open_logical`]).
    /// A companion already larger than `disk_cap_bytes` fails closed here;
    /// use [`Self::open_bulk`] for the bulk-load/migration build+verify
    /// path where a census-derived budget legitimately exceeds the
    /// incremental default (M5-D3 FIX 2 / #1518 skeptic review — the
    /// growth-above-published semantics must NOT leak into the serve path,
    /// where it would silently accept a companion of unbounded size).
    pub fn open(path: &Path, disk_cap_bytes: u64) -> Result<Self, OwnerPayloadError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let existing = file.metadata()?.len();
        if existing > disk_cap_bytes {
            return Err(OwnerPayloadError::DiskBudgetExceeded {
                current: existing,
                additional: 0,
                cap: disk_cap_bytes,
            });
        }
        Ok(Self {
            file: parking_lot::Mutex::new(file),
            disk_cap_bytes,
        })
    }

    /// Open an already-published companion for the BULK-LOAD / migration
    /// build+verify path only (M5-D3 amendment §5). `disk_cap_bytes` bounds
    /// GROWTH above the durably-published bytes: the effective ceiling is
    /// `existing_len + disk_cap_bytes`. A bulk-built companion (census-
    /// derived budget) may legally exceed the incremental default, and
    /// refusing to open it would make the loaded store silently unbootable
    /// (D-5); the incremental constant keeps governing exactly what it was
    /// sized for — churn-bounded growth above a ratified build-time budget.
    ///
    /// Do NOT call this from the serve/recovery/boot path — use
    /// [`Self::open`], whose ceiling is absolute.
    pub fn open_bulk(path: &Path, disk_cap_bytes: u64) -> Result<Self, OwnerPayloadError> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        let existing = file.metadata()?.len();
        Ok(Self {
            file: parking_lot::Mutex::new(file),
            disk_cap_bytes: existing.saturating_add(disk_cap_bytes),
        })
    }

    /// Encode logical bytes into an inline row payload or a durable overflow
    /// reference. Overflow bytes are synced before this method returns.
    pub fn encode(&self, logical: &[u8]) -> Result<Vec<u8>, OwnerPayloadError> {
        let len = u32::try_from(logical.len())
            .map_err(|_| OwnerPayloadError::PayloadTooLarge(logical.len()))?;
        let crc = crc32c::crc32c(logical);
        if logical.len() <= OWNER_ROW_MAX_PAYLOAD - ROW_REF_HEADER_BYTES {
            let mut out = Vec::with_capacity(ROW_REF_HEADER_BYTES + logical.len());
            write_ref_header(&mut out, MODE_INLINE, len, crc, 0);
            out.extend_from_slice(logical);
            return Ok(out);
        }

        let additional = OVERFLOW_HEADER_BYTES.saturating_add(u64::from(len));
        let mut file = self.file.lock();
        let offset = file.metadata()?.len();
        if offset.saturating_add(additional) > self.disk_cap_bytes {
            return Err(OwnerPayloadError::DiskBudgetExceeded {
                current: offset,
                additional,
                cap: self.disk_cap_bytes,
            });
        }
        file.seek(SeekFrom::Start(offset))?;
        file.write_all(OVERFLOW_MAGIC)?;
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&crc.to_le_bytes())?;
        file.write_all(&0_u32.to_le_bytes())?;
        file.write_all(logical)?;
        file.sync_data()?;
        let mut out = Vec::with_capacity(ROW_REF_HEADER_BYTES);
        write_ref_header(&mut out, MODE_OVERFLOW, len, crc, offset);
        Ok(out)
    }

    /// Resolve one row payload into its exact logical bytes.
    pub fn decode(&self, row_payload: &[u8]) -> Result<Vec<u8>, OwnerPayloadError> {
        if row_payload.len() < ROW_REF_HEADER_BYTES || &row_payload[..4] != ROW_REF_MAGIC {
            return Err(OwnerPayloadError::Corrupt(
                "row payload reference magic/length mismatch".to_owned(),
            ));
        }
        if row_payload[4] != 1 || row_payload[6..8] != [0, 0] {
            return Err(OwnerPayloadError::Corrupt(
                "row payload reference version/reserved mismatch".to_owned(),
            ));
        }
        let mode = row_payload[5];
        let len = u32::from_le_bytes(row_payload[8..12].try_into().map_err(|_| {
            OwnerPayloadError::Corrupt("row payload length field is malformed".to_owned())
        })?) as usize;
        let crc = u32::from_le_bytes(row_payload[12..16].try_into().map_err(|_| {
            OwnerPayloadError::Corrupt("row payload crc field is malformed".to_owned())
        })?);
        let offset = u64::from_le_bytes(row_payload[16..24].try_into().map_err(|_| {
            OwnerPayloadError::Corrupt("row payload offset field is malformed".to_owned())
        })?);
        let bytes = match mode {
            MODE_INLINE => {
                if offset != 0 || row_payload.len() != ROW_REF_HEADER_BYTES + len {
                    return Err(OwnerPayloadError::Corrupt(
                        "inline row payload length/offset mismatch".to_owned(),
                    ));
                }
                row_payload[ROW_REF_HEADER_BYTES..].to_vec()
            }
            MODE_OVERFLOW => {
                if row_payload.len() != ROW_REF_HEADER_BYTES {
                    return Err(OwnerPayloadError::Corrupt(
                        "overflow row reference carries trailing bytes".to_owned(),
                    ));
                }
                self.read_overflow(offset, len, crc)?
            }
            other => {
                return Err(OwnerPayloadError::Corrupt(format!(
                    "unknown row payload mode {other}"
                )));
            }
        };
        if crc32c::crc32c(&bytes) != crc {
            return Err(OwnerPayloadError::Corrupt(
                "logical payload checksum mismatch".to_owned(),
            ));
        }
        Ok(bytes)
    }

    /// Current file bytes for disk-budget gates.
    pub fn file_bytes(&self) -> Result<u64, OwnerPayloadError> {
        Ok(self.file.lock().metadata()?.len())
    }

    /// Hard byte ceiling.
    #[must_use]
    pub const fn disk_cap_bytes(&self) -> u64 {
        self.disk_cap_bytes
    }

    fn read_overflow(
        &self,
        offset: u64,
        expected_len: usize,
        expected_crc: u32,
    ) -> Result<Vec<u8>, OwnerPayloadError> {
        let mut file = self.file.lock();
        let len_u64 = u64::try_from(expected_len)
            .map_err(|_| OwnerPayloadError::PayloadTooLarge(expected_len))?;
        let end = offset
            .checked_add(OVERFLOW_HEADER_BYTES)
            .and_then(|value| value.checked_add(len_u64))
            .ok_or_else(|| OwnerPayloadError::Corrupt("overflow offset wraps".to_owned()))?;
        if end > file.metadata()?.len() {
            return Err(OwnerPayloadError::Corrupt(
                "overflow reference exceeds companion file".to_owned(),
            ));
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0_u8; OVERFLOW_HEADER_BYTES as usize];
        file.read_exact(&mut header)?;
        if &header[..4] != OVERFLOW_MAGIC || header[12..16] != [0; 4] {
            return Err(OwnerPayloadError::Corrupt(
                "overflow record header is invalid".to_owned(),
            ));
        }
        let stored_len = u32::from_le_bytes(header[4..8].try_into().map_err(|_| {
            OwnerPayloadError::Corrupt("overflow length field is malformed".to_owned())
        })?) as usize;
        let stored_crc = u32::from_le_bytes(header[8..12].try_into().map_err(|_| {
            OwnerPayloadError::Corrupt("overflow crc field is malformed".to_owned())
        })?);
        if stored_len != expected_len || stored_crc != expected_crc {
            return Err(OwnerPayloadError::Corrupt(
                "overflow record disagrees with row reference".to_owned(),
            ));
        }
        let mut bytes = vec![0_u8; expected_len];
        file.read_exact(&mut bytes)?;
        Ok(bytes)
    }
}

fn write_ref_header(out: &mut Vec<u8>, mode: u8, len: u32, crc: u32, offset: u64) {
    out.extend_from_slice(ROW_REF_MAGIC);
    out.push(1);
    out.push(mode);
    out.extend_from_slice(&0_u16.to_le_bytes());
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
}

fn sync_parent(path: &Path) -> Result<(), std::io::Error> {
    let parent = path.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "owner payload path has no parent",
        )
    })?;
    File::open(parent)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_and_overflow_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads");
        let store = OwnerPayloadStore::create(&path, 1024 * 1024).unwrap();
        for bytes in [vec![7; 12], vec![9; OWNER_ROW_MAX_PAYLOAD * 3]] {
            let encoded = store.encode(&bytes).unwrap();
            assert!(encoded.len() <= OWNER_ROW_MAX_PAYLOAD);
            assert_eq!(store.decode(&encoded).unwrap(), bytes);
        }
    }

    /// D-5 / M5-D3: a bulk-built companion larger than the incremental
    /// default must still OPEN via `open_bulk` (unbootability is the
    /// failure class the M3-r2 postmortem named), while the configured cap
    /// keeps bounding growth above the published bytes.
    #[test]
    fn open_bulk_treats_cap_as_growth_budget_above_published_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads");
        let big = vec![3_u8; OWNER_ROW_MAX_PAYLOAD * 4];
        let published = {
            let store = OwnerPayloadStore::create(&path, u64::MAX).unwrap();
            let reference = store.encode(&big).unwrap();
            (store.file_bytes().unwrap(), reference)
        };
        // Reopen with a tiny incremental cap: must open, must serve reads.
        let reopened = OwnerPayloadStore::open_bulk(&path, 64).unwrap();
        assert_eq!(reopened.decode(&published.1).unwrap(), big);
        assert_eq!(reopened.disk_cap_bytes(), published.0 + 64);
        // Growth beyond existing + cap still fails closed.
        let error = reopened.encode(&big).unwrap_err();
        assert!(matches!(
            error,
            OwnerPayloadError::DiskBudgetExceeded { .. }
        ));
    }

    /// FIX 2 (M5-D3 / #1518 skeptic review): the SERVE path (`open`, used by
    /// `OwnerRowRegistry::open_logical` on every boot/recovery) must keep
    /// its PRE-M5-D3 absolute-ceiling semantics — a companion already
    /// larger than `disk_cap_bytes` fails closed, exactly as before the
    /// bulk-load growth-cap change. RED-on-revert: if `open` regresses to
    /// growth-above-published semantics, this test goes green when it
    /// should fail (the reopen below would then succeed).
    #[test]
    fn open_serve_path_ceiling_is_absolute_and_unaffected_by_bulk_growth_cap() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads");
        let big = vec![3_u8; OWNER_ROW_MAX_PAYLOAD * 4];
        let published = {
            let store = OwnerPayloadStore::create(&path, u64::MAX).unwrap();
            let reference = store.encode(&big).unwrap();
            (store.file_bytes().unwrap(), reference)
        };
        // A companion already over the incremental cap MUST fail to open
        // via the serve path.
        let error = OwnerPayloadStore::open(&path, 64).unwrap_err();
        assert!(matches!(
            error,
            OwnerPayloadError::DiskBudgetExceeded { .. }
        ));
        // A cap that covers the published bytes opens fine and keeps an
        // ABSOLUTE ceiling (not existing + cap).
        let reopened = OwnerPayloadStore::open(&path, published.0 + 64).unwrap();
        assert_eq!(reopened.decode(&published.1).unwrap(), big);
        assert_eq!(reopened.disk_cap_bytes(), published.0 + 64);
    }

    #[test]
    fn overflow_disk_budget_is_hard() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("payloads");
        let store = OwnerPayloadStore::create(&path, 32).unwrap();
        let error = store
            .encode(&vec![1; OWNER_ROW_MAX_PAYLOAD * 2])
            .unwrap_err();
        assert!(matches!(
            error,
            OwnerPayloadError::DiskBudgetExceeded { cap: 32, .. }
        ));
    }
}
