//! WAL record wire format (roadmap M1-30, design-v2 §4.2; M1.5-03 adds tenant_id).
//!
//! ```text
//! offset  field        size  notes
//! 0       crc32c       4     CRC32C over bytes 4..length
//! 4       length       4     total length in bytes, including header
//! 8       record_type  1     see `WalRecordType`
//! 9       _reserved    3     must be 0 (forward-compat signal)
//! 12      txn_id       8
//! 20      lsn          8
//! 28      timestamp_ms 8     signed i64, UNIX millis
//! 36      tenant_id    8     logical database scope (ADR-011)
//! 44      payload      length - 44
//! ```
//!
//! Invariants guarded by the decoder:
//!
//! - `length >= HEADER_SIZE` (`ArcGraphError::InvalidRecordLength`)
//! - CRC32C of bytes `[4..length]` matches bytes `[0..4]`
//!   (`ArcGraphError::WalCorruption`)
//! - Reserved bytes 9..12 are zero (`ArcGraphError::WalCorruption`)
//! - `record_type` is either an encodable bare-engine variant, a reserved
//!   byte (`ArcGraphError::WalRecordTypeReserved`), or truly unknown
//!   (`ArcGraphError::WalCorruption`)
//!
//! The decoder is total on well-formed input and never panics; a
//! record it cannot materialize always produces a structured outcome so
//! the WAL recovery path (M1-34) can distinguish reserved format bytes
//! from corruption and decide whether to skip, stop replay, enter
//! "half-written tail" mode, or abort.

use arcgraph_core::{ArcGraphError, Lsn, Result, TenantId};

/// One-byte record-type tag.
///
/// **WAL FORMAT v1.** WAL compatibility across arcgraph versions is
/// not guaranteed before v1.0 GA. Adding or renumbering variants is
/// a breaking change; on upgrade, recovery must observe a version
/// magic in the segment header and fail fast on mismatch (M2.e
/// follow-up — tracked alongside the segment-header rework).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WalRecordType {
    /// Transaction begin marker.
    Begin = 1,
    /// Transaction commit marker.
    Commit = 2,
    /// Transaction abort marker.
    Abort = 3,
    /// Node create / upsert.
    PutNode = 4,
    /// Relationship create / upsert.
    PutRel = 5,
    /// Node delete (tombstone).
    DeleteNode = 6,
    /// Relationship delete (tombstone).
    DeleteRel = 7,
    /// Checkpoint marker (for recovery replay).
    Checkpoint = 8,
    /// Label/type intern binding — records a fresh `(tenant, StringId, name)`
    /// association (M2-32). Replay on startup rebuilds the
    /// [`crate::intern::InternTable`]. Payload format: 4 B `StringId`
    /// (little-endian u32) + name bytes (UTF-8; no length prefix — the
    /// record's `length` field bounds the tail).
    InternString = 9,
    /// BLOB property write (M2-31). Records a blob head-page id + the
    /// full byte payload so recovery can rebuild the
    /// [`crate::blob::BlobStore`] chain. Payload format:
    /// `8 B head_page_id (LE u64) + 4 B total_len (LE u32) + blob bytes`.
    /// `total_len` must equal the tail length; mismatches are rejected
    /// as WAL corruption.
    PutBlob = 10,
    /// Primary / secondary B-tree page-image write (M2-33 / M2-34).
    /// Physical page-write record: the full post-write page bytes
    /// keyed by page id. Covers both primary and secondary indices —
    /// discriminated on replay by the `PageType` byte inside the page
    /// header (DEC-11). Payload format:
    /// `8 B page_id (LE u64) + 8 B tenant_id (LE u64) + 8192 B page bytes`.
    /// WAL replay is an M2.e task; emission-only this milestone.
    ///
    /// **Legacy post-ADR-031.** The hot commit path no longer emits
    /// standalone IndexPage records; the bytes ride inside a
    /// [`CommitBundle`](Self::CommitBundle) instead. This variant is
    /// retained in the codec so recovery can decode pre-ADR-031 WAL
    /// segments and test fixtures.
    IndexPage = 11,
    /// Aggregated per-commit record (ADR-031, M2-E2 single-fire fold).
    /// Carries the MVCC commit payload (previously `Commit = 2`) plus
    /// N staged `IndexPage` snapshots in a single atomic length-
    /// prefixed + CRC-protected record. Every MVCC commit post-fix
    /// emits exactly one `CommitBundle`; replay applies the bundle
    /// atomically (MVCC writes + IndexPage entries). Payload format
    /// lives in [`super::bundle`]; see ADR-031 §Decision. Replay is
    /// an M2.e task (#38); emission-only this milestone.
    CommitBundle = 12,
}

impl WalRecordType {
    /// Parse a single byte back into an encodable bare-engine record type.
    ///
    /// Bytes 13–17 are reserved by the on-disk format. They return
    /// [`ArcGraphError::WalRecordTypeReserved`] so recovery can apply an
    /// explicit forward-compatibility policy without restoring the removed
    /// prediction-event variants. Bytes outside 1–17 remain WAL corruption.
    pub fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            1 => Self::Begin,
            2 => Self::Commit,
            3 => Self::Abort,
            4 => Self::PutNode,
            5 => Self::PutRel,
            6 => Self::DeleteNode,
            7 => Self::DeleteRel,
            8 => Self::Checkpoint,
            9 => Self::InternString,
            10 => Self::PutBlob,
            11 => Self::IndexPage,
            12 => Self::CommitBundle,
            13..=17 => {
                return Err(ArcGraphError::WalRecordTypeReserved { byte });
            }
            other => {
                return Err(ArcGraphError::WalCorruption {
                    lsn: Lsn::ZERO,
                    reason: format!("unknown wal record type byte: {other}"),
                });
            }
        })
    }

    /// Raw byte for on-disk storage.
    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }
}

/// A single WAL record in memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WalRecord {
    /// Record kind.
    pub record_type: WalRecordType,
    /// Owning transaction id (0 for records not bound to a txn, e.g. Checkpoint).
    pub txn_id: u64,
    /// Logical sequence number of this record.
    pub lsn: Lsn,
    /// Wall-clock timestamp in milliseconds since Unix epoch. Advisory
    /// only — MVCC visibility uses `lsn`, not timestamp.
    pub timestamp_ms: i64,
    /// Logical database scope. Every WAL record is scoped to a tenant so
    /// recovery can demultiplex entries per-tenant (ADR-011).
    pub tenant_id: TenantId,
    /// Record-type-specific payload.
    pub payload: Vec<u8>,
}

impl WalRecord {
    /// Size of the fixed header in bytes.
    pub const HEADER_SIZE: usize = 44;

    /// Maximum total record length (inclusive of header). A record
    /// cannot exceed the `u32` length field.
    pub const MAX_RECORD_LEN: usize = u32::MAX as usize;

    /// Maximum payload length. Encodes must return an error beyond this.
    pub const MAX_PAYLOAD_LEN: usize = Self::MAX_RECORD_LEN - Self::HEADER_SIZE;

    /// Total encoded length of this record.
    #[must_use]
    pub fn encoded_len(&self) -> usize {
        Self::HEADER_SIZE + self.payload.len()
    }

    /// Encode the record, appending to `out`. Returns the number of
    /// bytes appended. Fails if the payload is larger than
    /// [`Self::MAX_PAYLOAD_LEN`].
    pub fn encode(&self, out: &mut Vec<u8>) -> Result<usize> {
        if self.payload.len() > Self::MAX_PAYLOAD_LEN {
            return Err(ArcGraphError::InvalidRecordLength {
                got: self.payload.len(),
                expected: Self::MAX_PAYLOAD_LEN,
            });
        }
        let total_len = self.encoded_len();
        let start = out.len();
        out.resize(start + total_len, 0);

        let buf = &mut out[start..];
        // crc placeholder at [0..4]
        buf[4..8].copy_from_slice(
            &u32::try_from(total_len)
                .expect("bounded above")
                .to_le_bytes(),
        );
        buf[8] = self.record_type.as_byte();
        // buf[9..12] = reserved (zero)
        buf[12..20].copy_from_slice(&self.txn_id.to_le_bytes());
        buf[20..28].copy_from_slice(&self.lsn.raw().to_le_bytes());
        buf[28..36].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf[36..44].copy_from_slice(&self.tenant_id.raw().to_le_bytes());
        buf[44..total_len].copy_from_slice(&self.payload);

        let crc = crc32c::crc32c(&buf[4..total_len]);
        buf[0..4].copy_from_slice(&crc.to_le_bytes());
        Ok(total_len)
    }

    /// Allocate a `Vec` and encode into it.
    pub fn encode_to_vec(&self) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(self.encoded_len());
        self.encode(&mut out)?;
        Ok(out)
    }

    /// Parse one record from the start of `bytes`.
    ///
    /// Returns the decoded record and the number of bytes consumed. If
    /// `bytes.len()` is less than the expected record length, returns
    /// [`ArcGraphError::InvalidRecordLength`] so the reader can
    /// request more bytes from the WAL segment.
    pub fn decode(bytes: &[u8]) -> Result<(Self, usize)> {
        if bytes.len() < Self::HEADER_SIZE {
            return Err(ArcGraphError::InvalidRecordLength {
                got: bytes.len(),
                expected: Self::HEADER_SIZE,
            });
        }
        let length = u32::from_le_bytes(read_array::<4>(&bytes[4..8])) as usize;
        if length < Self::HEADER_SIZE {
            return Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!("record length {length} < header size"),
            });
        }
        if bytes.len() < length {
            return Err(ArcGraphError::InvalidRecordLength {
                got: bytes.len(),
                expected: length,
            });
        }
        let crc_stored = u32::from_le_bytes(read_array::<4>(&bytes[0..4]));
        let crc_computed = crc32c::crc32c(&bytes[4..length]);
        if crc_stored != crc_computed {
            return Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: format!(
                    "crc mismatch: stored 0x{crc_stored:08x}, computed 0x{crc_computed:08x}"
                ),
            });
        }
        // Reserved bytes must be zero for forward-compat.
        if bytes[9] != 0 || bytes[10] != 0 || bytes[11] != 0 {
            return Err(ArcGraphError::WalCorruption {
                lsn: Lsn::ZERO,
                reason: "non-zero reserved bytes".to_owned(),
            });
        }
        let record_type = WalRecordType::from_byte(bytes[8])?;
        let txn_id = u64::from_le_bytes(read_array::<8>(&bytes[12..20]));
        let lsn = Lsn::new(u64::from_le_bytes(read_array::<8>(&bytes[20..28])));
        let timestamp_ms = i64::from_le_bytes(read_array::<8>(&bytes[28..36]));
        let tenant_id = TenantId::new(u64::from_le_bytes(read_array::<8>(&bytes[36..44])));
        let payload = bytes[Self::HEADER_SIZE..length].to_vec();
        Ok((
            Self {
                record_type,
                txn_id,
                lsn,
                timestamp_ms,
                tenant_id,
                payload,
            },
            length,
        ))
    }
}

#[inline]
fn read_array<const N: usize>(slice: &[u8]) -> [u8; N] {
    let mut out = [0u8; N];
    out.copy_from_slice(&slice[..N]);
    out
}

#[cfg(test)]
mod tests {
    use arcgraph_core::TenantId;
    use proptest::prelude::*;

    use super::*;

    fn sample(ty: WalRecordType, txn_id: u64, lsn: u64, ts: i64, payload: Vec<u8>) -> WalRecord {
        WalRecord {
            record_type: ty,
            txn_id,
            lsn: Lsn::new(lsn),
            timestamp_ms: ts,
            tenant_id: TenantId::DEFAULT,
            payload,
        }
    }

    fn sample_tenant(
        ty: WalRecordType,
        txn_id: u64,
        lsn: u64,
        ts: i64,
        tenant_id: TenantId,
        payload: Vec<u8>,
    ) -> WalRecord {
        WalRecord {
            record_type: ty,
            txn_id,
            lsn: Lsn::new(lsn),
            timestamp_ms: ts,
            tenant_id,
            payload,
        }
    }

    // ---- record type ----

    #[test]
    fn record_type_roundtrip() {
        let encodable = [
            WalRecordType::Begin,
            WalRecordType::Commit,
            WalRecordType::Abort,
            WalRecordType::PutNode,
            WalRecordType::PutRel,
            WalRecordType::DeleteNode,
            WalRecordType::DeleteRel,
            WalRecordType::Checkpoint,
            WalRecordType::InternString,
            WalRecordType::PutBlob,
            WalRecordType::IndexPage,
            WalRecordType::CommitBundle,
        ];

        // Bytes 1..=12 are produced by this build and round-trip.
        for byte in 1u8..=12 {
            let ty = WalRecordType::from_byte(byte).unwrap();
            assert_eq!(ty.as_byte(), byte);
        }

        // Every constructible record type encodes below the reserved range.
        assert!(
            encodable
                .iter()
                .all(|record_type| (1..=12).contains(&record_type.as_byte()))
        );

        // Bytes 13..=17 remain valid reserved format discriminants, but the
        // bare engine cannot construct or encode a record carrying one.
        for byte in 13u8..=17 {
            let err = WalRecordType::from_byte(byte).unwrap_err();
            assert!(matches!(
                err,
                ArcGraphError::WalRecordTypeReserved { byte: observed }
                    if observed == byte
            ));
        }

        // Outside the post-W18α 1..=17 format range is genuine corruption.
        for byte in [0, 18, u8::MAX] {
            let err = WalRecordType::from_byte(byte).unwrap_err();
            assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
        }
    }

    #[test]
    fn intern_string_variant_byte_is_9() {
        assert_eq!(WalRecordType::InternString.as_byte(), 9);
        assert_eq!(
            WalRecordType::from_byte(9).unwrap(),
            WalRecordType::InternString
        );
    }

    #[test]
    fn put_blob_variant_byte_is_10() {
        assert_eq!(WalRecordType::PutBlob.as_byte(), 10);
        assert_eq!(
            WalRecordType::from_byte(10).unwrap(),
            WalRecordType::PutBlob
        );
    }

    #[test]
    fn index_page_variant_byte_is_11() {
        assert_eq!(WalRecordType::IndexPage.as_byte(), 11);
        assert_eq!(
            WalRecordType::from_byte(11).unwrap(),
            WalRecordType::IndexPage
        );
    }

    #[test]
    fn commit_bundle_variant_byte_is_12() {
        assert_eq!(WalRecordType::CommitBundle.as_byte(), 12);
        assert_eq!(
            WalRecordType::from_byte(12).unwrap(),
            WalRecordType::CommitBundle
        );
    }

    #[test]
    fn unknown_record_type_is_wal_corruption() {
        let err = WalRecordType::from_byte(42).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    // ---- encode / decode happy path ----

    #[test]
    fn empty_payload_roundtrip() {
        let r = sample(WalRecordType::Begin, 1, 100, 1_700_000_000_000, vec![]);
        let bytes = r.encode_to_vec().unwrap();
        assert_eq!(bytes.len(), WalRecord::HEADER_SIZE);
        let (back, consumed) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(back, r);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn small_payload_roundtrip() {
        let r = sample(WalRecordType::PutNode, 7, 200, 0, b"hello world".to_vec());
        let bytes = r.encode_to_vec().unwrap();
        assert_eq!(bytes.len(), WalRecord::HEADER_SIZE + 11);
        let (back, consumed) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(back, r);
        assert_eq!(consumed, bytes.len());
    }

    #[test]
    fn large_payload_roundtrip() {
        let payload = vec![0x5Au8; 128 * 1024]; // 128 KiB
        let r = sample(WalRecordType::PutRel, 42, 999, -1, payload);
        let bytes = r.encode_to_vec().unwrap();
        let (back, _) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(back, r);
    }

    #[test]
    fn encode_appends_to_existing_vec() {
        let mut buf = vec![0xFF_u8; 5];
        let r = sample(WalRecordType::Commit, 1, 1, 1, vec![0x01, 0x02]);
        let written = r.encode(&mut buf).unwrap();
        assert_eq!(written, r.encoded_len());
        assert_eq!(&buf[0..5], &[0xFF_u8; 5]);
        let (back, consumed) = WalRecord::decode(&buf[5..]).unwrap();
        assert_eq!(back, r);
        assert_eq!(consumed, written);
    }

    // ---- negative cases ----

    #[test]
    fn too_short_input_is_error() {
        let err = WalRecord::decode(&[0u8; 4]).unwrap_err();
        assert!(matches!(err, ArcGraphError::InvalidRecordLength { .. }));
    }

    #[test]
    fn length_smaller_than_header_is_corruption() {
        let mut bytes = vec![0u8; WalRecord::HEADER_SIZE];
        // Write length = 8 (smaller than HEADER_SIZE)
        bytes[4..8].copy_from_slice(&8u32.to_le_bytes());
        let err = WalRecord::decode(&bytes).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn reserved_nonzero_is_corruption() {
        let r = sample(WalRecordType::PutNode, 1, 1, 0, vec![0x01]);
        let mut bytes = r.encode_to_vec().unwrap();
        bytes[10] = 0xAB;
        // Rewrite CRC so the CRC check passes; the reserved-byte check
        // must still fire.
        let new_crc = crc32c::crc32c(&bytes[4..]);
        bytes[0..4].copy_from_slice(&new_crc.to_le_bytes());
        let err = WalRecord::decode(&bytes).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("reserved"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    #[test]
    fn unknown_type_byte_is_corruption() {
        let r = sample(WalRecordType::PutNode, 1, 1, 0, vec![]);
        let mut bytes = r.encode_to_vec().unwrap();
        bytes[8] = 99;
        let new_crc = crc32c::crc32c(&bytes[4..]);
        bytes[0..4].copy_from_slice(&new_crc.to_le_bytes());
        let err = WalRecord::decode(&bytes).unwrap_err();
        assert!(matches!(err, ArcGraphError::WalCorruption { .. }));
    }

    #[test]
    fn reserved_range_wal_record_is_not_reported_as_corruption() {
        let original = sample(WalRecordType::PutNode, 1, 7, 0, vec![])
            .encode_to_vec()
            .unwrap();

        for byte in 13u8..=17 {
            let mut bytes = original.clone();
            bytes[8] = byte;
            let new_crc = crc32c::crc32c(&bytes[4..]);
            bytes[0..4].copy_from_slice(&new_crc.to_le_bytes());

            let err = WalRecord::decode(&bytes).unwrap_err();
            assert!(
                matches!(
                    &err,
                    ArcGraphError::WalRecordTypeReserved { byte: observed }
                        if *observed == byte
                ),
                "reserved byte {byte} must have a typed non-corruption outcome; got {err:?}"
            );
            assert!(
                !err.to_string().contains("corruption"),
                "reserved byte {byte} was reported as corruption: {err}"
            );
        }
    }

    #[test]
    fn crc_mismatch_is_corruption() {
        let r = sample(WalRecordType::PutRel, 1, 1, 0, vec![0xDE, 0xAD]);
        let mut bytes = r.encode_to_vec().unwrap();
        // Flip a payload byte; leave CRC untouched.
        bytes[WalRecord::HEADER_SIZE] ^= 0x01;
        let err = WalRecord::decode(&bytes).unwrap_err();
        match err {
            ArcGraphError::WalCorruption { reason, .. } => {
                assert!(reason.contains("crc"), "got: {reason}");
            }
            other => panic!("expected WalCorruption, got {other:?}"),
        }
    }

    // ---- tenant_id roundtrips ----

    #[test]
    fn tenant_id_system_roundtrips() {
        let r = sample_tenant(WalRecordType::Begin, 0, 1, 0, TenantId::SYSTEM, vec![]);
        let bytes = r.encode_to_vec().unwrap();
        let (back, _) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(back.tenant_id, TenantId::SYSTEM);
    }

    #[test]
    fn tenant_id_default_roundtrips() {
        let r = sample_tenant(
            WalRecordType::Commit,
            1,
            2,
            0,
            TenantId::DEFAULT,
            vec![0xAB],
        );
        let bytes = r.encode_to_vec().unwrap();
        let (back, _) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(back.tenant_id, TenantId::DEFAULT);
    }

    #[test]
    fn tenant_id_large_value_roundtrips() {
        let large = TenantId::new(u64::MAX - 1);
        let r = sample_tenant(WalRecordType::PutNode, 42, 99, -1, large, vec![1, 2, 3]);
        let bytes = r.encode_to_vec().unwrap();
        let (back, _) = WalRecord::decode(&bytes).unwrap();
        assert_eq!(back.tenant_id, large);
    }

    #[test]
    fn header_size_is_44() {
        assert_eq!(WalRecord::HEADER_SIZE, 44);
    }

    #[test]
    fn tenant_id_at_bytes_36_to_44() {
        let tenant = TenantId::new(0xDEAD_BEEF_CAFE_1234);
        let r = sample_tenant(WalRecordType::PutRel, 1, 1, 0, tenant, vec![]);
        let bytes = r.encode_to_vec().unwrap();
        let stored = u64::from_le_bytes(bytes[36..44].try_into().unwrap());
        assert_eq!(stored, 0xDEAD_BEEF_CAFE_1234);
    }

    // ---- proptest: exhaustive roundtrip ----

    proptest! {
        #[test]
        fn property_roundtrip(
            type_byte in 1u8..=12, // proptest uses non-reserved variants only; reserved variants tested separately
            txn_id in any::<u64>(),
            lsn in any::<u64>(),
            timestamp in any::<i64>(),
            tenant_raw in any::<u64>(),
            payload in prop::collection::vec(any::<u8>(), 0..2048),
        ) {
            let ty = WalRecordType::from_byte(type_byte).unwrap();
            let r = sample_tenant(ty, txn_id, lsn, timestamp, TenantId::new(tenant_raw), payload);
            let bytes = r.encode_to_vec().unwrap();
            let (back, consumed) = WalRecord::decode(&bytes).unwrap();
            prop_assert_eq!(back, r);
            prop_assert_eq!(consumed, bytes.len());
        }

        #[test]
        fn property_single_bit_flip_never_silent(
            type_byte in 1u8..=12, // proptest uses non-reserved variants only; reserved variants tested separately
            txn_id in any::<u64>(),
            lsn in any::<u64>(),
            timestamp in any::<i64>(),
            tenant_raw in any::<u64>(),
            payload in prop::collection::vec(any::<u8>(), 0..256),
            flip_byte in 0usize..(44 + 256),
            flip_bit in 0u8..8,
        ) {
            let ty = WalRecordType::from_byte(type_byte).unwrap();
            let r = sample_tenant(ty, txn_id, lsn, timestamp, TenantId::new(tenant_raw), payload);
            let mut bytes = r.encode_to_vec().unwrap();
            prop_assume!(flip_byte < bytes.len());
            bytes[flip_byte] ^= 1 << flip_bit;
            // Decoder must either:
            //   (a) reject with a structured error, or
            //   (b) produce a record that differs from the original.
            // It must never silently accept the corrupted record as `r`.
            match WalRecord::decode(&bytes) {
                Ok((back, _)) => prop_assert_ne!(back, r),
                Err(e) => {
                    let ok = matches!(
                        e,
                        ArcGraphError::WalCorruption { .. }
                            | ArcGraphError::InvalidRecordLength { .. }
                    );
                    prop_assert!(ok, "unexpected error: {e:?}");
                }
            }
        }

        #[test]
        fn property_decode_rejects_truncated_records(
            payload in prop::collection::vec(any::<u8>(), 0..128),
            truncate_to in 0usize..44,
        ) {
            let r = sample(WalRecordType::PutNode, 1, 1, 0, payload);
            let bytes = r.encode_to_vec().unwrap();
            let truncated = &bytes[..truncate_to.min(bytes.len())];
            let err = WalRecord::decode(truncated).unwrap_err();
            let ok = matches!(err, ArcGraphError::InvalidRecordLength { .. });
            prop_assert!(ok);
        }

        #[test]
        fn property_encoded_len_matches_output(
            type_byte in 1u8..=12, // proptest uses non-reserved variants only; reserved variants tested separately
            payload in prop::collection::vec(any::<u8>(), 0..512),
        ) {
            let ty = WalRecordType::from_byte(type_byte).unwrap();
            let r = sample(ty, 1, 1, 0, payload);
            let bytes = r.encode_to_vec().unwrap();
            prop_assert_eq!(bytes.len(), r.encoded_len());
            prop_assert_eq!(bytes.len(), WalRecord::HEADER_SIZE + r.payload.len());
        }
    }
}
