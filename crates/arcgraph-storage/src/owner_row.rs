//! Durable direct-address substrate for M4 metadata owner rows.
//!
//! This module deliberately stops below the logical idempotency, intern, and
//! permission facades.  It owns only the stable class/address algebra, the
//! fixed self-identifying row envelope, and the production extent-backed read
//! path those facades use.  Keeping the layer narrow makes the durability
//! boundary explicit: an owner lookup faults the same bounded page store that
//! WAL redo and checkpoint use; there is no record-keyed in-memory mirror.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{ArcGraphError, Lsn, PageId, PageType, TenantId};
use bytes::Bytes;
use thiserror::Error;

use crate::extent::{EXTENT_BYTES, EXTENT_PAGES, ExtentAllocation, ExtentDataPageStore};
use crate::owner_index::{
    OWNER_INDEX_DISK_CAP_BYTES, OwnerForwardIndex, OwnerIndexError, str_hash_56,
};
use crate::owner_payload::{OWNER_PAYLOAD_DISK_CAP_BYTES, OwnerPayloadError, OwnerPayloadStore};
use crate::primary_index::RecordKind;
use crate::records::{
    PAGE_BODY_BYTES, PageError, SLOT_AREA_START, SLOT_SIZE, SlotId, SlottedPageRef,
};
use crate::redo::{DeltaPageStore, DirtyPageTable};
use crate::transaction::TxnManager;
use crate::wal::{
    AllocatorAdvance, BUNDLE_FORMAT_V10, DeltaIntent, DeltaOp, DeltaOpKind, STORE_GRANTS,
    STORE_INTERN, STORE_NODE_BINDINGS, STORE_RECORD, STORE_REL_BINDINGS, STORE_RELS, WalHandle,
};

/// Bytes occupied by every direct-address owner row.
///
/// A fixed envelope keeps slot arithmetic stable and makes sparse gaps
/// unambiguous: a zero directory entry is absent, while every live row has a
/// full 256-byte cell carrying its class and id.
pub const OWNER_ROW_BYTES: usize = 256;
/// Direct rows per slotted page. Changing this changes durable addressing.
pub const OWNER_ROWS_PER_PAGE: u64 = (PAGE_BODY_BYTES / (SLOT_SIZE + OWNER_ROW_BYTES)) as u64;
const _: () = assert!(OWNER_ROWS_PER_PAGE == 31);

const OWNER_ROW_MAGIC: &[u8; 8] = b"AGOWNR01";
const OWNER_ROW_STATE_LIVE: u8 = 1;
const OWNER_ROW_STATE_RETIRED: u8 = 2;
const OWNER_ROW_HEADER_BYTES: usize = 20;
/// Maximum payload retained directly in one substrate row.
///
/// Overflow ownership belongs to the logical facades in M4 Slice-3b-2; this
/// slice never introduces a second retirement or overflow encoding.
pub const OWNER_ROW_MAX_PAYLOAD: usize = OWNER_ROW_BYTES - OWNER_ROW_HEADER_BYTES;
/// Production direct-id capacity reserved for each class.
pub const OWNER_IDS_PER_CLASS: u64 = 64_000_000;
/// Reserved last id carrying a durable allocator high-water marker for
/// intern/class-id namespaces. Normal allocation never returns this id.
pub const OWNER_ALLOCATOR_MARKER_ID: u64 = OWNER_IDS_PER_CLASS - 1;

/// Every durable direct-row class exposed by the substrate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum OwnerRowClass {
    /// Node external-id binding, stored in `node-bindings.store`.
    NodeBinding = 1,
    /// Relationship external-id binding, stored in `rel-bindings.store`.
    RelBinding = 2,
    /// Interned string reverse row, stored in `intern.store`.
    InternedString = 3,
    /// ACL class-id row, stored in the second direct region of `intern.store`.
    ClassId = 4,
    /// Document grant row, stored in `grants.store`.
    Grant = 5,
}

impl OwnerRowClass {
    /// Class-complete substrate census used by gates and bootstrap wiring.
    pub const ALL: [Self; 5] = [
        Self::NodeBinding,
        Self::RelBinding,
        Self::InternedString,
        Self::ClassId,
        Self::Grant,
    ];

    /// Production extent store owning this class.
    #[must_use]
    pub const fn store_id(self) -> u16 {
        match self {
            Self::NodeBinding => STORE_NODE_BINDINGS,
            Self::RelBinding => STORE_REL_BINDINGS,
            Self::InternedString | Self::ClassId => STORE_INTERN,
            Self::Grant => STORE_GRANTS,
        }
    }

    /// Stable physical WAL operation used by this class.
    #[must_use]
    pub const fn delta_kind(self) -> DeltaOpKind {
        match self {
            Self::NodeBinding | Self::RelBinding | Self::InternedString => DeltaOpKind::InternBind,
            Self::ClassId | Self::Grant => DeltaOpKind::AclGrant,
        }
    }

    const fn region_ordinal(self) -> u64 {
        match self {
            Self::ClassId => 1,
            Self::NodeBinding | Self::RelBinding | Self::InternedString | Self::Grant => 0,
        }
    }

    fn from_byte(byte: u8) -> Result<Self, OwnerRowError> {
        match byte {
            1 => Ok(Self::NodeBinding),
            2 => Ok(Self::RelBinding),
            3 => Ok(Self::InternedString),
            4 => Ok(Self::ClassId),
            5 => Ok(Self::Grant),
            other => Err(OwnerRowError::InvalidEnvelope(format!(
                "unknown owner-row class {other}"
            ))),
        }
    }

    /// Closed-form page/slot address for one durable class id.
    pub fn address(self, id: u64) -> Result<OwnerRowAddress, OwnerRowError> {
        if id >= OWNER_IDS_PER_CLASS {
            return Err(OwnerRowError::IdOutOfRange {
                class: self,
                id,
                capacity: OWNER_IDS_PER_CLASS,
            });
        }
        let pages_per_class = OWNER_IDS_PER_CLASS.div_ceil(OWNER_ROWS_PER_PAGE);
        let page_no = self
            .region_ordinal()
            .checked_mul(pages_per_class)
            .and_then(|base| base.checked_add(id / OWNER_ROWS_PER_PAGE))
            .ok_or(OwnerRowError::AddressOverflow { class: self, id })?;
        Ok(OwnerRowAddress {
            page_no,
            slot: SlotId((id % OWNER_ROWS_PER_PAGE) as u16),
        })
    }
}

/// Companion payload file for one owner class.
#[must_use]
pub fn owner_payload_path(generation: &Path, tenant: TenantId, class: OwnerRowClass) -> PathBuf {
    let stem = match class {
        OwnerRowClass::NodeBinding => "node-bindings",
        OwnerRowClass::RelBinding => "rel-bindings",
        OwnerRowClass::InternedString => "intern-strings",
        OwnerRowClass::ClassId => "acl-classes",
        OwnerRowClass::Grant => "grants",
    };
    generation
        .join(crate::m3_migration::M3_TENANTS_DIR)
        .join(tenant.raw().to_string())
        .join(crate::extent::PRODUCTION_EXTENT_SUBDIR)
        .join(format!("{stem}.payload"))
}

/// Immutable sorted-run forward-index directory for a hash-addressed owner
/// class. Grant rows are direct-only and therefore have no forward index.
#[must_use]
pub fn owner_forward_index_path(
    generation: &Path,
    tenant: TenantId,
    class: OwnerRowClass,
) -> Option<PathBuf> {
    let stem = match class {
        OwnerRowClass::NodeBinding => "node-bindings",
        OwnerRowClass::RelBinding => "rel-bindings",
        OwnerRowClass::InternedString => "intern-strings",
        OwnerRowClass::ClassId => "acl-classes",
        OwnerRowClass::Grant => return None,
    };
    Some(
        generation
            .join(crate::m3_migration::M3_TENANTS_DIR)
            .join(tenant.raw().to_string())
            .join(crate::extent::PRODUCTION_EXTENT_SUBDIR)
            .join(format!("{stem}.forward")),
    )
}

/// One arithmetic direct-row address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerRowAddress {
    /// Logical page number inside the owning extent store.
    pub page_no: u64,
    /// Fixed row slot within the page.
    pub slot: SlotId,
}

/// Decoded live owner row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OwnerRow {
    class: OwnerRowClass,
    id: u64,
    payload: Vec<u8>,
    retired: bool,
}

impl OwnerRow {
    /// Construct one fixed direct row.
    pub fn new(
        class: OwnerRowClass,
        id: u64,
        payload: impl Into<Vec<u8>>,
    ) -> Result<Self, OwnerRowError> {
        class.address(id)?;
        let payload = payload.into();
        if payload.len() > OWNER_ROW_MAX_PAYLOAD {
            return Err(OwnerRowError::PayloadTooLarge {
                len: payload.len(),
                max: OWNER_ROW_MAX_PAYLOAD,
            });
        }
        Ok(Self {
            class,
            id,
            payload,
            retired: false,
        })
    }

    /// Construct a permanent binding retirement. The WAL carries this
    /// self-identifying envelope, while live/recovery apply materializes the
    /// exact PRE-B.4 `(offset=0, length=u16::MAX)` directory marker.
    pub fn retired(class: OwnerRowClass, id: u64) -> Result<Self, OwnerRowError> {
        if !matches!(
            class,
            OwnerRowClass::NodeBinding | OwnerRowClass::RelBinding
        ) {
            return Err(OwnerRowError::InvalidEnvelope(format!(
                "owner class {class:?} is not permanently retireable"
            )));
        }
        class.address(id)?;
        Ok(Self {
            class,
            id,
            payload: Vec::new(),
            retired: true,
        })
    }

    /// Durable class encoded in the row.
    #[must_use]
    pub const fn class(&self) -> OwnerRowClass {
        self.class
    }

    /// Direct id encoded in the row.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Exact caller payload.
    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.payload
    }

    /// Whether apply must materialize a permanent direct-slot tombstone.
    #[must_use]
    pub const fn is_retired(&self) -> bool {
        self.retired
    }

    /// Encode the canonical fixed-size durable cell.
    #[must_use]
    pub fn encode(&self) -> [u8; OWNER_ROW_BYTES] {
        let mut bytes = [0_u8; OWNER_ROW_BYTES];
        bytes[..8].copy_from_slice(OWNER_ROW_MAGIC);
        bytes[8] = self.class as u8;
        bytes[9] = if self.retired {
            OWNER_ROW_STATE_RETIRED
        } else {
            OWNER_ROW_STATE_LIVE
        };
        bytes[10..12].copy_from_slice(
            &u16::try_from(self.payload.len())
                .expect("owner-row payload is bounded")
                .to_le_bytes(),
        );
        bytes[12..20].copy_from_slice(&self.id.to_le_bytes());
        bytes[20..20 + self.payload.len()].copy_from_slice(&self.payload);
        bytes
    }

    fn decode(
        bytes: &[u8],
        expected_class: OwnerRowClass,
        expected_id: u64,
    ) -> Result<Self, OwnerRowError> {
        if bytes.len() != OWNER_ROW_BYTES {
            return Err(OwnerRowError::InvalidEnvelope(format!(
                "owner row has {} bytes, expected {OWNER_ROW_BYTES}",
                bytes.len()
            )));
        }
        if &bytes[..8] != OWNER_ROW_MAGIC {
            return Err(OwnerRowError::InvalidEnvelope(
                "owner row magic/version mismatch".to_owned(),
            ));
        }
        let class = Self::decode_class(bytes[8])?;
        if class != expected_class {
            return Err(OwnerRowError::WrongClass {
                expected: expected_class,
                got: class,
            });
        }
        let retired = match bytes[9] {
            OWNER_ROW_STATE_LIVE => false,
            OWNER_ROW_STATE_RETIRED => true,
            _ => {
                return Err(OwnerRowError::InvalidEnvelope(format!(
                    "owner row carries unknown state {}",
                    bytes[9]
                )));
            }
        };
        if retired
            && !matches!(
                class,
                OwnerRowClass::NodeBinding | OwnerRowClass::RelBinding
            )
        {
            return Err(OwnerRowError::InvalidEnvelope(format!(
                "owner class {class:?} carries an illegal permanent retirement"
            )));
        }
        let payload_len = usize::from(u16::from_le_bytes([bytes[10], bytes[11]]));
        if payload_len > OWNER_ROW_MAX_PAYLOAD {
            return Err(OwnerRowError::InvalidEnvelope(format!(
                "owner row payload length {payload_len} exceeds {OWNER_ROW_MAX_PAYLOAD}"
            )));
        }
        if retired && payload_len != 0 {
            return Err(OwnerRowError::InvalidEnvelope(
                "retired owner row carries a payload".to_owned(),
            ));
        }
        let id = u64::from_le_bytes(bytes[12..20].try_into().expect("fixed id field"));
        if id != expected_id {
            return Err(OwnerRowError::WrongId {
                expected: expected_id,
                got: id,
            });
        }
        if bytes[20 + payload_len..].iter().any(|byte| *byte != 0) {
            return Err(OwnerRowError::InvalidEnvelope(
                "owner row padding is non-zero".to_owned(),
            ));
        }
        Ok(Self {
            class,
            id,
            payload: bytes[20..20 + payload_len].to_vec(),
            retired,
        })
    }

    fn decode_class(byte: u8) -> Result<OwnerRowClass, OwnerRowError> {
        OwnerRowClass::from_byte(byte)
    }
}

/// Logical binding bytes carried by node/relationship direct rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingOwnerValue {
    /// Opaque node/relationship namespace byte.
    pub kind: u8,
    /// Complete external id used for mandatory collision verification.
    pub external_id: String,
    /// Optional same-process ingest payload hash.
    pub payload_hash: Option<u64>,
    /// False is a durable release tombstone.
    pub active: bool,
}

impl BindingOwnerValue {
    /// Stable logical encoding.
    pub fn encode(&self) -> Result<Vec<u8>, OwnerRowError> {
        let external = self.external_id.as_bytes();
        let len = u32::try_from(external.len()).map_err(|_| {
            OwnerRowError::InvalidEnvelope("binding external id exceeds u32".to_owned())
        })?;
        let mut out = Vec::with_capacity(24 + external.len());
        out.extend_from_slice(b"AGB1");
        out.push(u8::from(self.active));
        out.push(self.kind);
        out.push(u8::from(self.payload_hash.is_some()));
        out.push(0);
        out.extend_from_slice(&str_hash_56(&self.external_id).to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&self.payload_hash.unwrap_or(0).to_le_bytes());
        out.extend_from_slice(external);
        Ok(out)
    }

    /// Decode and validate a logical binding.
    pub fn decode(bytes: &[u8]) -> Result<Self, OwnerRowError> {
        if bytes.len() < 28 || &bytes[..4] != b"AGB1" || bytes[7] != 0 || bytes[4] > 1 {
            return Err(OwnerRowError::InvalidEnvelope(
                "binding logical envelope is malformed".to_owned(),
            ));
        }
        if bytes[6] > 1 {
            return Err(OwnerRowError::InvalidEnvelope(
                "binding payload-hash flag is not boolean".to_owned(),
            ));
        }
        let stored_hash = u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| {
            OwnerRowError::InvalidEnvelope("binding hash field is malformed".to_owned())
        })?);
        let len = u32::from_le_bytes(bytes[16..20].try_into().map_err(|_| {
            OwnerRowError::InvalidEnvelope("binding length field is malformed".to_owned())
        })?) as usize;
        if bytes.len() != 28 + len {
            return Err(OwnerRowError::InvalidEnvelope(
                "binding external id length mismatch".to_owned(),
            ));
        }
        let payload_hash = u64::from_le_bytes(bytes[20..28].try_into().map_err(|_| {
            OwnerRowError::InvalidEnvelope("binding payload hash is malformed".to_owned())
        })?);
        let external_id = String::from_utf8(bytes[28..].to_vec()).map_err(|_| {
            OwnerRowError::InvalidEnvelope("binding external id is not UTF-8".to_owned())
        })?;
        if stored_hash != str_hash_56(&external_id) {
            return Err(OwnerRowError::InvalidEnvelope(
                "binding stored hash disagrees with its full external id".to_owned(),
            ));
        }
        Ok(Self {
            kind: bytes[5],
            external_id,
            payload_hash: (bytes[6] == 1).then_some(payload_hash),
            active: bytes[4] == 1,
        })
    }
}

/// Logical intern row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternOwnerValue {
    /// Complete interned UTF-8 name.
    pub name: String,
}

impl InternOwnerValue {
    /// Stable logical encoding.
    pub fn encode(&self) -> Result<Vec<u8>, OwnerRowError> {
        let len = u32::try_from(self.name.len())
            .map_err(|_| OwnerRowError::InvalidEnvelope("intern name exceeds u32".to_owned()))?;
        let mut out = Vec::with_capacity(16 + self.name.len());
        out.extend_from_slice(b"AGI1");
        out.extend_from_slice(&str_hash_56(&self.name).to_le_bytes());
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        Ok(out)
    }

    /// Decode and validate a logical intern row.
    pub fn decode(bytes: &[u8]) -> Result<Self, OwnerRowError> {
        if bytes.len() < 16 || &bytes[..4] != b"AGI1" {
            return Err(OwnerRowError::InvalidEnvelope(
                "intern logical envelope is malformed".to_owned(),
            ));
        }
        let hash = u64::from_le_bytes(bytes[4..12].try_into().map_err(|_| {
            OwnerRowError::InvalidEnvelope("intern hash field is malformed".to_owned())
        })?);
        let len = u32::from_le_bytes(bytes[12..16].try_into().map_err(|_| {
            OwnerRowError::InvalidEnvelope("intern length field is malformed".to_owned())
        })?) as usize;
        if bytes.len() != 16 + len {
            return Err(OwnerRowError::InvalidEnvelope(
                "intern name length mismatch".to_owned(),
            ));
        }
        let name = String::from_utf8(bytes[16..].to_vec())
            .map_err(|_| OwnerRowError::InvalidEnvelope("intern name is not UTF-8".to_owned()))?;
        if hash != str_hash_56(&name) {
            return Err(OwnerRowError::InvalidEnvelope(
                "intern stored hash disagrees with its full name".to_owned(),
            ));
        }
        Ok(Self { name })
    }
}

/// Canonical grant-set bytes stored under a durable ACL class id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassOwnerValue {
    /// Sorted unique principal names.
    pub grants: BTreeSet<String>,
}

impl ClassOwnerValue {
    /// Canonical logical encoding. The encoded bytes themselves are the full
    /// collision-verification key for class interning.
    pub fn encode(&self) -> Result<Vec<u8>, OwnerRowError> {
        let count = u32::try_from(self.grants.len()).map_err(|_| {
            OwnerRowError::InvalidEnvelope("ACL grant count exceeds u32".to_owned())
        })?;
        let mut body = Vec::new();
        body.extend_from_slice(&count.to_le_bytes());
        for grant in &self.grants {
            let len = u32::try_from(grant.len()).map_err(|_| {
                OwnerRowError::InvalidEnvelope("ACL principal exceeds u32".to_owned())
            })?;
            body.extend_from_slice(&len.to_le_bytes());
            body.extend_from_slice(grant.as_bytes());
        }
        let mut out = Vec::with_capacity(12 + body.len());
        out.extend_from_slice(b"AGC1");
        out.extend_from_slice(&hash_bytes_56(&body).to_le_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Decode and validate canonical order/uniqueness.
    pub fn decode(bytes: &[u8]) -> Result<Self, OwnerRowError> {
        if bytes.len() < 16 || &bytes[..4] != b"AGC1" {
            return Err(OwnerRowError::InvalidEnvelope(
                "ACL class logical envelope is malformed".to_owned(),
            ));
        }
        let hash = u64::from_le_bytes(bytes[4..12].try_into().map_err(|_| {
            OwnerRowError::InvalidEnvelope("ACL class hash is malformed".to_owned())
        })?);
        let body = &bytes[12..];
        if hash != hash_bytes_56(body) {
            return Err(OwnerRowError::InvalidEnvelope(
                "ACL class hash disagrees with canonical grants".to_owned(),
            ));
        }
        let mut cursor = 0_usize;
        let count = take_u32(body, &mut cursor, "ACL class grant count")? as usize;
        let mut grants = BTreeSet::new();
        for _ in 0..count {
            let len = take_u32(body, &mut cursor, "ACL principal length")? as usize;
            let end = cursor.checked_add(len).ok_or_else(|| {
                OwnerRowError::InvalidEnvelope("ACL principal length wraps".to_owned())
            })?;
            let value = body.get(cursor..end).ok_or_else(|| {
                OwnerRowError::InvalidEnvelope("ACL principal overruns row".to_owned())
            })?;
            let value = std::str::from_utf8(value).map_err(|_| {
                OwnerRowError::InvalidEnvelope("ACL principal is not UTF-8".to_owned())
            })?;
            if !grants.insert(value.to_owned()) {
                return Err(OwnerRowError::InvalidEnvelope(
                    "ACL class contains a duplicate principal".to_owned(),
                ));
            }
            cursor = end;
        }
        if cursor != body.len() {
            return Err(OwnerRowError::InvalidEnvelope(
                "ACL class carries trailing bytes".to_owned(),
            ));
        }
        Ok(Self { grants })
    }

    /// Stable 56-bit hash of the canonical full grant key.
    pub fn hash(&self) -> Result<u64, OwnerRowError> {
        let encoded = self.encode()?;
        Ok(u64::from_le_bytes(encoded[4..12].try_into().map_err(
            |_| OwnerRowError::InvalidEnvelope("ACL class hash field is malformed".to_owned()),
        )?))
    }
}

/// Direct document→class mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrantOwnerValue {
    /// Durable class id.
    pub class_id: u32,
    /// False is a durable revoke tombstone.
    pub active: bool,
}

impl GrantOwnerValue {
    /// Stable logical encoding.
    #[must_use]
    pub fn encode(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(12);
        out.extend_from_slice(b"AGG1");
        out.push(u8::from(self.active));
        out.extend_from_slice(&[0; 3]);
        out.extend_from_slice(&self.class_id.to_le_bytes());
        out
    }

    /// Decode a direct grant mapping.
    pub fn decode(bytes: &[u8]) -> Result<Self, OwnerRowError> {
        if bytes.len() != 12 || &bytes[..4] != b"AGG1" || bytes[4] > 1 || bytes[5..8] != [0; 3] {
            return Err(OwnerRowError::InvalidEnvelope(
                "grant logical envelope is malformed".to_owned(),
            ));
        }
        Ok(Self {
            active: bytes[4] == 1,
            class_id: u32::from_le_bytes(bytes[8..12].try_into().map_err(|_| {
                OwnerRowError::InvalidEnvelope("grant class id is malformed".to_owned())
            })?),
        })
    }
}

/// Durable allocator marker stored at [`OWNER_ALLOCATOR_MARKER_ID`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerAllocatorMarker {
    /// Shared [`crate::wal::AllocatorKind`] byte.
    pub kind: u8,
    /// Highest id ever durably allocated.
    pub high_water: u64,
}

impl OwnerAllocatorMarker {
    /// Stable logical encoding.
    #[must_use]
    pub fn encode(self) -> Vec<u8> {
        let mut out = Vec::with_capacity(16);
        out.extend_from_slice(b"AGM1");
        out.push(self.kind);
        out.extend_from_slice(&[0; 3]);
        out.extend_from_slice(&self.high_water.to_le_bytes());
        out
    }

    /// Decode a marker.
    pub fn decode(bytes: &[u8]) -> Result<Self, OwnerRowError> {
        if bytes.len() != 16 || &bytes[..4] != b"AGM1" || bytes[5..8] != [0; 3] {
            return Err(OwnerRowError::InvalidEnvelope(
                "owner allocator marker is malformed".to_owned(),
            ));
        }
        Ok(Self {
            kind: bytes[4],
            high_water: u64::from_le_bytes(bytes[8..16].try_into().map_err(|_| {
                OwnerRowError::InvalidEnvelope("owner allocator high-water is malformed".to_owned())
            })?),
        })
    }
}

fn hash_bytes_56(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash & ((1_u64 << 56) - 1)
}

fn take_u32(bytes: &[u8], cursor: &mut usize, what: &str) -> Result<u32, OwnerRowError> {
    let end = cursor
        .checked_add(4)
        .ok_or_else(|| OwnerRowError::InvalidEnvelope(format!("{what} cursor wraps")))?;
    let field = bytes
        .get(*cursor..end)
        .ok_or_else(|| OwnerRowError::InvalidEnvelope(format!("{what} overruns row")))?;
    *cursor = end;
    Ok(u32::from_le_bytes(field.try_into().map_err(|_| {
        OwnerRowError::InvalidEnvelope(format!("{what} is malformed"))
    })?))
}

/// Owner-row substrate failure. Only a missing direct slot is represented as
/// `Ok(None)`; page identity, format, checksum, and store failures stay hard.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OwnerRowError {
    /// Direct record id could not be mapped into its arithmetic slot.
    #[error(transparent)]
    RecordAddress(#[from] crate::address::AddressError),
    /// Production extent I/O or directory validation failed.
    #[error(transparent)]
    Physical(#[from] ArcGraphError),
    /// Forward sorted-run index operation failed.
    #[error(transparent)]
    Index(#[from] OwnerIndexError),
    /// Immutable logical payload operation failed.
    #[error(transparent)]
    Payload(#[from] OwnerPayloadError),
    /// Slotted-page validation or decoding failed.
    #[error(transparent)]
    Page(#[from] PageError),
    /// The requested class is not owned by this store.
    #[error("owner class {class:?} belongs to store {expected}, not store {got}")]
    WrongStore {
        /// Requested class.
        class: OwnerRowClass,
        /// Store required by the class.
        expected: u16,
        /// Store backing this reader.
        got: u16,
    },
    /// The page header does not identify the requested owner home.
    #[error("owner page identity mismatch: {0}")]
    PageIdentity(String),
    /// A row at the arithmetic address names another class.
    #[error("owner row class mismatch: expected {expected:?}, got {got:?}")]
    WrongClass {
        /// Addressed class.
        expected: OwnerRowClass,
        /// Encoded class.
        got: OwnerRowClass,
    },
    /// A row at the arithmetic address names another id.
    #[error("owner row id mismatch: expected {expected}, got {got}")]
    WrongId {
        /// Addressed id.
        expected: u64,
        /// Encoded id.
        got: u64,
    },
    /// Row envelope bytes are malformed.
    #[error("invalid owner-row envelope: {0}")]
    InvalidEnvelope(String),
    /// Caller payload does not fit the direct-row substrate.
    #[error("owner-row payload has {len} bytes, max is {max}")]
    PayloadTooLarge {
        /// Supplied length.
        len: usize,
        /// Durable direct-row maximum.
        max: usize,
    },
    /// Direct id exceeds the class capacity.
    #[error("owner class {class:?} id {id} exceeds capacity {capacity}")]
    IdOutOfRange {
        /// Addressed class.
        class: OwnerRowClass,
        /// Supplied id.
        id: u64,
        /// Exclusive capacity.
        capacity: u64,
    },
    /// Direct address arithmetic overflowed.
    #[error("owner class {class:?} id {id} address overflow")]
    AddressOverflow {
        /// Addressed class.
        class: OwnerRowClass,
        /// Supplied id.
        id: u64,
    },
    /// Durable bootstrap has not attached the served transaction/WAL owners.
    #[error("owner-row commit runtime is unavailable")]
    CommitRuntimeUnavailable,
    /// A commit attempted to publish owner deltas into a non-v10 WAL.
    #[error("owner-row physical commit requires WAL bundle format v10, got {0}")]
    WrongWalFormat(u16),
    /// One physical target appeared twice in the same owner plan.
    #[error("owner-row plan repeats store {store_id} page {page_no} slot {slot}")]
    DuplicateTarget {
        /// Target store.
        store_id: u16,
        /// Target logical page.
        page_no: u64,
        /// Target direct slot.
        slot: u16,
    },
    /// The page header's `slot_count` claims a slot is unused, but the
    /// CRC-protected slot directory still addresses a live owner row there.
    ///
    /// `slot_count` (header bytes 36..38) is outside the page CRC (bytes 40..),
    /// so a bit-flip that lowers it keeps the checksum valid and would
    /// otherwise erase durable rows into a fail-open `NotFound`. Hard error:
    /// the substrate refuses to report a durable row as absent.
    #[error(
        "owner page header claims slot_count={header_slot_count} but the CRC-protected slot \
         directory still addresses a live owner row at slot {slot} (offset {row_offset}); \
         slot_count is outside the page CRC — refusing to report a durable row as absent"
    )]
    HeaderSlotCountUnderflow {
        /// The slot the reader addressed.
        slot: u16,
        /// The high-water mark the (unprotected) header claimed.
        header_slot_count: u16,
        /// In-page byte offset of the live row the directory still points at.
        row_offset: usize,
    },
    /// A physical owner commit must contain at least one row.
    #[error("owner-row commit cannot be empty")]
    EmptyCommit,
    /// A logical facade was invoked on a substrate-only registry without its
    /// published companion files.
    #[error("owner logical companion is unavailable for tenant {tenant:?} class {class:?}")]
    LogicalCompanionUnavailable {
        /// Tenant being served.
        tenant: TenantId,
        /// Owner class being served.
        class: OwnerRowClass,
    },
    /// A store's next dense physical extent offset overflowed.
    #[error("owner store {store_id} physical extent offset overflow")]
    PhysicalOffsetOverflow {
        /// Target owner store.
        store_id: u16,
    },
}

/// The single P1-b owner direct-row taxonomy seam.
///
/// An unused directory entry below the page high-water already arrives as
/// `Ok(None)`. A target at or above that high-water arrives as
/// [`PageError::SlotOutOfRange`] and is the only error mapped to NotFound.
/// Every other page error remains a hard failure.
///
/// # Why this seam needs the page bytes (the `slot_count`-outside-CRC hole)
///
/// The page CRC covers `bytes[PageHeader::SIZE..]` — bytes 40.. — but
/// `slot_count` lives in the header at bytes 36..38 (`core/record.rs`). It is
/// therefore **outside** integrity protection. A bit-flip that *lowers*
/// `slot_count` (say 6 → 2) leaves the CRC valid, so `SlottedPageRef::open`
/// succeeds; every live row at a slot ≥ the forged count then reports
/// [`PageError::SlotOutOfRange`], which this seam would map to `Ok(None)`.
/// Durable idempotency/intern/ACL rows would silently vanish: duplicate
/// ingests, never-released bindings, disappeared label and type names.
///
/// The slot **directory**, however, starts at byte 40 and *is* inside the CRC.
/// So a directory entry that still points at a live owner row (magic present)
/// is independent, integrity-protected evidence that the header is lying. A
/// corruptor cannot both lower `slot_count` and erase the directory entry
/// without breaking the CRC.
///
/// This seam therefore refuses to convert `SlotOutOfRange` into "absent" while
/// the CRC-protected body still holds a live row at that slot — it fails
/// **closed** with [`OwnerRowError::HeaderSlotCountUnderflow`] instead.
pub fn owner_direct_row_disposition<T>(
    result: Result<Option<T>, PageError>,
    page: &[u8],
    slot: SlotId,
) -> Result<Option<T>, OwnerRowError> {
    match result {
        Err(PageError::SlotOutOfRange { count, .. }) => {
            assert_slot_area_has_no_live_owner_row(page, slot, count)?;
            Ok(None)
        }
        Ok(value) => Ok(value),
        Err(error) => Err(OwnerRowError::Page(error)),
    }
}

/// Cross-check a header that claims `slot` is past the high-water mark against
/// the CRC-protected slot directory. See [`owner_direct_row_disposition`].
fn assert_slot_area_has_no_live_owner_row(
    page: &[u8],
    slot: SlotId,
    header_slot_count: u16,
) -> Result<(), OwnerRowError> {
    let index = usize::from(slot.0);
    // Owner pages are dense and fixed-stride: a slot at or beyond the class
    // capacity is genuinely unaddressable, not a forged count.
    if index >= OWNER_ROWS_PER_PAGE as usize {
        return Ok(());
    }
    let entry = SLOT_AREA_START + index * SLOT_SIZE;
    let Some(directory) = page.get(entry..entry + SLOT_SIZE) else {
        return Ok(());
    };
    let offset = usize::from(u16::from_le_bytes([directory[0], directory[1]]));
    let length = usize::from(u16::from_le_bytes([directory[2], directory[3]]));
    // offset==0 / length==0 is the tombstoned-or-never-written encoding: a
    // legitimately empty slot. Only a directory entry that still addresses a
    // live owner row contradicts the header.
    if offset == 0 || length == 0 {
        return Ok(());
    }
    let Some(row) = page.get(offset..offset.saturating_add(length)) else {
        return Ok(());
    };
    if !row.starts_with(OWNER_ROW_MAGIC) {
        return Ok(());
    }
    Err(OwnerRowError::HeaderSlotCountUnderflow {
        slot: slot.0,
        header_slot_count,
        row_offset: offset,
    })
}

/// Production reader for one `(tenant, owner store)` extent home.
#[derive(Clone)]
pub struct OwnerRowStore {
    data: Arc<ExtentDataPageStore>,
    read_faults: Arc<AtomicU64>,
}

impl std::fmt::Debug for OwnerRowStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnerRowStore")
            .field("tenant", &self.tenant())
            .field("store_id", &self.store_id())
            .finish_non_exhaustive()
    }
}

impl OwnerRowStore {
    /// Bind the reader to the same bounded extent data store used by redo.
    #[must_use]
    pub fn new(data: Arc<ExtentDataPageStore>) -> Self {
        Self {
            data,
            read_faults: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Tenant encoded by the durable extent owner.
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.data.directory().tenant()
    }

    /// Stable M4 store discriminator.
    #[must_use]
    pub fn store_id(&self) -> u16 {
        self.data.directory().store_id()
    }

    /// Read one direct row without creating a mapping, page, or mirror entry.
    pub fn read(&self, class: OwnerRowClass, id: u64) -> Result<Option<OwnerRow>, OwnerRowError> {
        if class.store_id() != self.store_id() {
            return Err(OwnerRowError::WrongStore {
                class,
                expected: class.store_id(),
                got: self.store_id(),
            });
        }
        let address = class.address(id)?;
        if self
            .data
            .directory()
            .mapping(address.page_no / EXTENT_PAGES)?
            .is_none()
        {
            return Ok(None);
        }
        self.read_faults.fetch_add(1, Ordering::AcqRel);
        let Some(page) = self
            .data
            .read_page_for_redo(self.tenant(), PageId::new(address.page_no))?
        else {
            return Ok(None);
        };
        if page.iter().all(|byte| *byte == 0) {
            return Ok(None);
        }
        let view = SlottedPageRef::open(page.as_ref())?;
        self.validate_page(&view, address.page_no)?;
        let Some(bytes) =
            owner_direct_row_disposition(view.read_bag(address.slot), page.as_ref(), address.slot)?
        else {
            return Ok(None);
        };
        OwnerRow::decode(bytes, class, id).map(Some)
    }

    /// Number of direct data-page faults issued by this store. Directory
    /// probes are bounded metadata and are intentionally not counted.
    #[must_use]
    pub fn read_fault_count(&self) -> u64 {
        self.read_faults.load(Ordering::Acquire)
    }

    /// Reset direct-fault instrumentation for one lookup gate.
    pub fn reset_read_fault_count(&self) {
        self.read_faults.store(0, Ordering::Release);
    }

    fn validate_page(&self, page: &SlottedPageRef<'_>, page_no: u64) -> Result<(), OwnerRowError> {
        let header = page.header();
        if header.page_id != page_no
            || header.tenant_id != self.tenant().raw()
            || header.page_type != PageType::PropSlotted.as_byte()
            || header.flags != self.store_id()
        {
            return Err(OwnerRowError::PageIdentity(format!(
                "requested tenant/store/page ({}, {}, {}), header has ({}, {}, {}, type={})",
                self.tenant().raw(),
                self.store_id(),
                page_no,
                header.tenant_id,
                header.flags,
                header.page_id,
                header.page_type,
            )));
        }
        Ok(())
    }
}

#[derive(Clone)]
struct OwnerCommitRuntime {
    txn: Arc<TxnManager>,
    wal: WalHandle,
}

impl std::fmt::Debug for OwnerCommitRuntime {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OwnerCommitRuntime")
            .field("wal_format", &self.wal.format_version())
            .finish_non_exhaustive()
    }
}

/// Shared production M4 extent registry with owner-row logical companions.
///
/// The map cardinality is tenant × the fixed M4 store census. Rows themselves
/// remain exclusively page-backed. Recovery, live Phase-3 apply, checkpoint,
/// and owner reads all share the exact [`ExtentDataPageStore`] handles
/// registered here. Only owner stores receive payload/index companions.
#[derive(Debug)]
pub struct OwnerRowRegistry {
    stores: BTreeMap<(TenantId, u16), Arc<OwnerRowStore>>,
    indexes: BTreeMap<(TenantId, OwnerRowClass), Arc<OwnerForwardIndex>>,
    payloads: BTreeMap<(TenantId, OwnerRowClass), Arc<OwnerPayloadStore>>,
    dpt: Arc<DirtyPageTable>,
    planning: parking_lot::Mutex<()>,
    commit_runtime: parking_lot::RwLock<Option<OwnerCommitRuntime>>,
}

impl OwnerRowRegistry {
    /// Construct the bounded registry over production extent homes.
    #[must_use]
    pub fn new(
        stores: impl IntoIterator<Item = Arc<OwnerRowStore>>,
        dpt: Arc<DirtyPageTable>,
    ) -> Self {
        Self {
            stores: stores
                .into_iter()
                .map(|store| ((store.tenant(), store.store_id()), store))
                .collect(),
            indexes: BTreeMap::new(),
            payloads: BTreeMap::new(),
            dpt,
            planning: parking_lot::Mutex::new(()),
            commit_runtime: parking_lot::RwLock::new(None),
        }
    }

    /// Open the field-complete logical companions for a published M4
    /// generation. Every class gets an immutable payload file; every forward
    /// owner gets a sorted-run candidate index. Missing files fail loudly.
    pub fn open_logical(
        generation: &Path,
        stores: impl IntoIterator<Item = Arc<OwnerRowStore>>,
        dpt: Arc<DirtyPageTable>,
    ) -> Result<Self, OwnerRowError> {
        let stores: BTreeMap<_, _> = stores
            .into_iter()
            .map(|store| ((store.tenant(), store.store_id()), store))
            .collect();
        let mut indexes = BTreeMap::new();
        let mut payloads = BTreeMap::new();
        for (tenant, store_id) in stores.keys() {
            for class in classes_for_store(*store_id) {
                payloads.insert(
                    (*tenant, class),
                    Arc::new(OwnerPayloadStore::open(
                        &owner_payload_path(generation, *tenant, class),
                        OWNER_PAYLOAD_DISK_CAP_BYTES,
                    )?),
                );
                if let Some(path) = owner_forward_index_path(generation, *tenant, class) {
                    indexes.insert(
                        (*tenant, class),
                        Arc::new(OwnerForwardIndex::open(&path, OWNER_INDEX_DISK_CAP_BYTES)?),
                    );
                }
            }
        }
        Ok(Self {
            stores,
            indexes,
            payloads,
            dpt,
            planning: parking_lot::Mutex::new(()),
            commit_runtime: parking_lot::RwLock::new(None),
        })
    }

    /// Whether every registered owner store has all of its field-complete
    /// logical companions attached.
    #[must_use]
    pub fn logical_companions_complete(&self) -> bool {
        self.stores.iter().all(|((tenant, store_id), _)| {
            classes_for_store(*store_id).into_iter().all(|class| {
                self.payloads.contains_key(&(*tenant, class))
                    && (class == OwnerRowClass::Grant
                        || self.indexes.contains_key(&(*tenant, class)))
            })
        })
    }

    /// Attach the exact transaction and WAL owners used by durable serving.
    /// Bootstrap calls this only after recovery and torn-tail truncation.
    pub fn attach_commit_runtime(&self, txn: Arc<TxnManager>, wal: WalHandle) {
        *self.commit_runtime.write() = Some(OwnerCommitRuntime { txn, wal });
    }

    /// Whether live physical commits are available after bootstrap.
    #[must_use]
    pub fn has_commit_runtime(&self) -> bool {
        self.commit_runtime.read().is_some()
    }

    /// Resolve one owner row through its production extent home.
    pub fn read(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
    ) -> Result<Option<OwnerRow>, OwnerRowError> {
        self.store(tenant, class.store_id())?.read(class, id)
    }

    /// Encode one logical value into its fixed owner row, spilling oversized
    /// bytes durably before returning the row.
    pub fn encode_logical_row(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
        logical: &[u8],
    ) -> Result<OwnerRow, OwnerRowError> {
        let payload = self.payload(tenant, class)?.encode(logical)?;
        OwnerRow::new(class, id, payload)
    }

    /// Direct-address fault followed by inline/overflow logical decode.
    pub fn read_logical(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
    ) -> Result<Option<Vec<u8>>, OwnerRowError> {
        let Some(row) = self.read(tenant, class, id)? else {
            return Ok(None);
        };
        self.payload(tenant, class)?
            .decode(row.payload())
            .map(Some)
            .map_err(Into::into)
    }

    /// Durably publish a forward candidate before the owner-row WAL commit,
    /// then commit the authoritative direct row. An uncommitted candidate is
    /// harmless because every lookup rechecks the full logical row.
    pub fn commit_indexed_logical_row(
        &self,
        tenant: TenantId,
        row: OwnerRow,
        hash: u64,
    ) -> Result<Lsn, OwnerRowError> {
        self.index(tenant, row.class())?
            .insert_batch([(hash, row.id())])?;
        self.commit_row(tenant, row)
    }

    /// Prepare a forward-indexed logical row for inclusion in a caller's
    /// existing v10 transaction. The candidate is fsync-published first; if
    /// the caller later aborts, full-row verification makes the orphan
    /// candidate inert.
    pub(crate) fn prepare_indexed_logical_row(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
        hash: u64,
        logical: &[u8],
    ) -> Result<OwnerRow, OwnerRowError> {
        let row = self.encode_logical_row(tenant, class, id, logical)?;
        self.index(tenant, class)?.insert_batch([(hash, id)])?;
        Ok(row)
    }

    /// Prepare a direct-only logical row for an existing v10 transaction.
    pub(crate) fn prepare_direct_logical_row(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
        logical: &[u8],
    ) -> Result<OwnerRow, OwnerRowError> {
        self.encode_logical_row(tenant, class, id, logical)
    }

    /// Prepare a permanent binding retirement for a caller's v10 commit.
    /// Apply uses the shared PRE-B.4 directory marker; no second tombstone
    /// encoding is introduced for owner bags.
    pub(crate) fn prepare_retired_binding_row(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        id: u64,
    ) -> Result<OwnerRow, OwnerRowError> {
        let _ = self.store(tenant, class.store_id())?;
        OwnerRow::retired(class, id)
    }

    /// Candidate-then-verify lookup. Hash equality is never authoritative;
    /// zero full-key matches is `Ok(None)`.
    pub fn find_verified(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
        hash: u64,
        mut verify: impl FnMut(u64, &[u8]) -> bool,
    ) -> Result<Option<(u64, Vec<u8>)>, OwnerRowError> {
        let index = self.index(tenant, class)?;
        let mut matched: Option<(u64, Vec<u8>)> = None;
        let mut callback_error: Option<OwnerRowError> = None;
        index.for_each_candidate(hash, |id| match self.read_logical(tenant, class, id) {
            Ok(Some(logical)) if verify(id, &logical) => {
                matched = Some((id, logical));
                Ok(true)
            }
            Ok(_) => Ok(false),
            Err(error) => {
                callback_error = Some(error);
                Ok(true)
            }
        })?;
        if let Some(error) = callback_error {
            return Err(error);
        }
        Ok(matched)
    }

    /// Reset and read direct-page fault instrumentation for a class's store.
    pub fn reset_read_fault_count(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
    ) -> Result<(), OwnerRowError> {
        self.store(tenant, class.store_id())?
            .reset_read_fault_count();
        Ok(())
    }

    /// Direct data-page faults since the last reset.
    pub fn read_fault_count(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
    ) -> Result<u64, OwnerRowError> {
        Ok(self.store(tenant, class.store_id())?.read_fault_count())
    }

    /// Registered tenant ids. Cardinality is tenant-count, never owner-row
    /// count; facades use this to seed their bounded allocator metadata.
    pub(crate) fn tenants(&self) -> impl Iterator<Item = TenantId> + '_ {
        self.stores
            .keys()
            .map(|(tenant, _)| *tenant)
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    /// Candidate count for a forward owner class. This reads only bounded run
    /// descriptors and is used by the residency census.
    pub fn candidate_count(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
    ) -> Result<u64, OwnerRowError> {
        Ok(self.index(tenant, class)?.candidate_count())
    }

    /// Flush every owner directory/data cache. Checkpoint callers establish
    /// the metadata frontier only after this returns successfully.
    pub fn flush_all(&self) -> Result<(), OwnerRowError> {
        for store in self.stores.values() {
            store.data.flush_all()?;
        }
        Ok(())
    }

    /// Commit one owner row through the existing transaction Phase-3 apply
    /// boundary. No page is published before the real v10 WAL append returns
    /// with exact durability proof.
    pub fn commit_row(&self, tenant: TenantId, row: OwnerRow) -> Result<Lsn, OwnerRowError> {
        self.commit_rows(tenant, [row])
    }

    /// Commit multiple owner rows under one WAL commit and one dense physical
    /// allocation plan.
    ///
    /// The plan latch extends the extent directory's apply-time dense-offset
    /// guard up to assignment time. It remains held until WAL outcome and
    /// Phase-3 apply resolve, so a concurrent plan can neither reuse a pending
    /// physical offset nor observe a process-local allocation that later
    /// rolls back. Direct reads do not take this latch.
    pub fn commit_rows(
        &self,
        tenant: TenantId,
        rows: impl IntoIterator<Item = OwnerRow>,
    ) -> Result<Lsn, OwnerRowError> {
        self.commit_rows_with_allocator_advances(tenant, rows, [])
    }

    /// Commit owner rows and their allocator high-water advances under the
    /// same v10 WAL record. Intern and ACL-class allocators use this so a
    /// durable reference can never outlive the allocator state that issued
    /// its id.
    pub fn commit_rows_with_allocator_advances(
        &self,
        tenant: TenantId,
        rows: impl IntoIterator<Item = OwnerRow>,
        advances: impl IntoIterator<Item = AllocatorAdvance>,
    ) -> Result<Lsn, OwnerRowError> {
        let rows: Vec<_> = rows.into_iter().collect();
        let advances: Vec<_> = advances.into_iter().collect();
        if rows.is_empty() {
            return Err(OwnerRowError::EmptyCommit);
        }
        let _plan_guard = self.planning.lock();
        let runtime = self
            .commit_runtime
            .read()
            .clone()
            .ok_or(OwnerRowError::CommitRuntimeUnavailable)?;
        let intents = self.plan_rows(tenant, &rows)?;
        let mut tx = runtime.txn.begin(tenant);
        let wal_format = tx.wal_format_version();
        if wal_format != BUNDLE_FORMAT_V10 {
            tx.abort();
            return Err(OwnerRowError::WrongWalFormat(wal_format));
        }
        tx.mutation_log_mut().delta_intents.extend(intents);
        #[cfg(any(test, feature = "fault-injection"))]
        let pause = owner_phase1_pause_requested(&rows);
        tx.commit_with_bundle_apply_and_rollback(
            move |_, _, allocator_advances, _, _| {
                allocator_advances.extend(advances);
                #[cfg(any(test, feature = "fault-injection"))]
                {
                    if let Some(path) = pause.as_deref() {
                        owner_phase1_pause(path)?;
                    }
                }
                Ok(Vec::new())
            },
            |deltas, commit_lsn| {
                if !runtime.wal.take_exact_durable(commit_lsn) {
                    return Err(corruption(
                        commit_lsn,
                        "owner-row Phase-3 apply lacks exact WAL durability proof",
                    ));
                }
                self.apply_committed(deltas, commit_lsn)
            },
            |_log| {},
        )
        .map_err(OwnerRowError::Physical)
    }

    fn store(&self, tenant: TenantId, store_id: u16) -> Result<&Arc<OwnerRowStore>, OwnerRowError> {
        self.stores.get(&(tenant, store_id)).ok_or_else(|| {
            OwnerRowError::PageIdentity(format!(
                "no production owner store for tenant {} store {store_id}",
                tenant.raw()
            ))
        })
    }

    fn payload(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
    ) -> Result<&Arc<OwnerPayloadStore>, OwnerRowError> {
        self.payloads
            .get(&(tenant, class))
            .ok_or(OwnerRowError::LogicalCompanionUnavailable { tenant, class })
    }

    fn index(
        &self,
        tenant: TenantId,
        class: OwnerRowClass,
    ) -> Result<&Arc<OwnerForwardIndex>, OwnerRowError> {
        self.indexes
            .get(&(tenant, class))
            .ok_or(OwnerRowError::LogicalCompanionUnavailable { tenant, class })
    }

    pub(crate) fn planning_guard(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.planning.lock()
    }

    pub(crate) fn plan_rows_locked(
        &self,
        tenant: TenantId,
        rows: &[OwnerRow],
    ) -> Result<Vec<DeltaIntent>, OwnerRowError> {
        self.plan_rows(tenant, rows)
    }

    /// Plan the durable extent mappings needed by already-reserved pages in
    /// one registered M4 store.
    ///
    /// Property pages are reserved in the bounded BlobStore scratch before
    /// the commit builder reaches the shared physical-store planner.  Their
    /// `PageAlloc` intents therefore need this companion pass when the first
    /// page in a new logical extent is touched.  The caller prepends the
    /// returned intents ahead of those page intents and holds
    /// [`Self::planning_guard`] through the WAL outcome and Phase-3 apply.
    pub(crate) fn plan_page_extents_locked(
        &self,
        tenant: TenantId,
        store_id: u16,
        page_nos: &[u64],
    ) -> Result<Vec<DeltaIntent>, OwnerRowError> {
        if page_nos.is_empty() {
            return Ok(Vec::new());
        }
        let store = self.store(tenant, store_id)?;
        let logical_extents: BTreeSet<_> = page_nos
            .iter()
            .map(|page_no| page_no / EXTENT_PAGES)
            .collect();
        let mut intents = Vec::new();
        let mut next_physical = None;
        for logical_extent in logical_extents {
            if store.data.directory().mapping(logical_extent)?.is_some() {
                continue;
            }
            let physical_offset = match next_physical {
                Some(offset) => offset,
                None => store.data.directory().recover_next_physical_offset()?,
            };
            next_physical = Some(
                physical_offset
                    .checked_add(EXTENT_BYTES)
                    .ok_or(OwnerRowError::PhysicalOffsetOverflow { store_id })?,
            );
            intents.push(DeltaIntent::extent_alloc(
                store_id,
                tenant,
                ExtentAllocation {
                    logical_extent,
                    physical_offset,
                    pairing: u32::try_from(logical_extent).unwrap_or(u32::MAX),
                },
            ));
        }
        Ok(intents)
    }

    /// Plan authoritative M4 node/relationship rows at `RecordKind::address`.
    /// The caller holds [`Self::planning_guard`] through WAL outcome and
    /// Phase-3 apply, extending the directory's dense-offset invariant back
    /// to assignment time exactly as for owner rows.
    pub(crate) fn plan_direct_records_locked(
        &self,
        tenant: TenantId,
        records: &[(RecordKind, u64, Option<Bytes>)],
    ) -> Result<Vec<DeltaIntent>, OwnerRowError> {
        let mut targets = BTreeSet::new();
        let mut extents = BTreeSet::new();
        let mut pages = BTreeMap::<(u16, u64), PageType>::new();
        for (kind, id, payload) in records {
            let store_id = match kind {
                RecordKind::Node => STORE_RECORD,
                RecordKind::Rel => STORE_RELS,
            };
            self.store(tenant, store_id)?;
            let (page_no, slot) = kind.address(*id)?;
            if !targets.insert((store_id, page_no, slot)) {
                return Err(OwnerRowError::DuplicateTarget {
                    store_id,
                    page_no,
                    slot,
                });
            }
            if let Some(payload) = payload {
                let expected = match kind {
                    RecordKind::Node => arcgraph_core::record::NodeRecord::SIZE,
                    RecordKind::Rel => arcgraph_core::record::RelRecord::SIZE,
                };
                if payload.len() != expected {
                    return Err(OwnerRowError::InvalidEnvelope(format!(
                        "direct {kind:?} record {id} has {} bytes, expected {expected}",
                        payload.len()
                    )));
                }
            }
            extents.insert((store_id, page_no / EXTENT_PAGES));
            pages.insert(
                (store_id, page_no),
                match kind {
                    RecordKind::Node => PageType::Node,
                    RecordKind::Rel => PageType::Rel,
                },
            );
        }

        let mut intents = Vec::new();
        let mut mapped_extents = BTreeMap::new();
        let mut next_physical = BTreeMap::<u16, u64>::new();
        for (store_id, logical_extent) in extents {
            let store = self.store(tenant, store_id)?;
            let mapped = store.data.directory().mapping(logical_extent)?.is_some();
            mapped_extents.insert((store_id, logical_extent), mapped);
            if mapped {
                continue;
            }
            let physical_offset = match next_physical.get(&store_id).copied() {
                Some(offset) => offset,
                None => store.data.directory().recover_next_physical_offset()?,
            };
            next_physical.insert(
                store_id,
                physical_offset
                    .checked_add(EXTENT_BYTES)
                    .ok_or(OwnerRowError::PhysicalOffsetOverflow { store_id })?,
            );
            intents.push(DeltaIntent::extent_alloc(
                store_id,
                tenant,
                ExtentAllocation {
                    logical_extent,
                    physical_offset,
                    pairing: u32::try_from(logical_extent).unwrap_or(u32::MAX),
                },
            ));
        }

        for ((store_id, page_no), page_type) in pages {
            let logical_extent = page_no / EXTENT_PAGES;
            let mapped = mapped_extents[&(store_id, logical_extent)];
            let formatted = if mapped {
                self.store(tenant, store_id)?
                    .data
                    .read_page_for_redo(tenant, PageId::new(page_no))?
                    .is_some_and(|page| page.iter().any(|byte| *byte != 0))
            } else {
                false
            };
            if !formatted {
                intents.push(DeltaIntent::page_alloc(
                    store_id, tenant, page_no, page_type, 1,
                ));
            }
        }

        for (kind, id, payload) in records {
            let store_id = match kind {
                RecordKind::Node => STORE_RECORD,
                RecordKind::Rel => STORE_RELS,
            };
            let (page_no, slot) = kind.address(*id)?;
            intents.push(DeltaIntent {
                kind: if payload.is_some() {
                    DeltaOpKind::PutRecord
                } else {
                    DeltaOpKind::TombstoneRecord
                },
                store_id,
                tenant_id: tenant,
                page_no,
                slot,
                payload: payload.clone().unwrap_or_default(),
            });
        }
        Ok(intents)
    }

    fn plan_rows(
        &self,
        tenant: TenantId,
        rows: &[OwnerRow],
    ) -> Result<Vec<DeltaIntent>, OwnerRowError> {
        let mut targets = BTreeSet::new();
        let mut extents = BTreeSet::new();
        let mut pages = BTreeSet::new();
        for row in rows {
            let class = row.class();
            self.store(tenant, class.store_id())?;
            let address = class.address(row.id())?;
            if !targets.insert((class.store_id(), address.page_no, address.slot.raw())) {
                return Err(OwnerRowError::DuplicateTarget {
                    store_id: class.store_id(),
                    page_no: address.page_no,
                    slot: address.slot.raw(),
                });
            }
            extents.insert((class.store_id(), address.page_no / EXTENT_PAGES));
            pages.insert((class.store_id(), address.page_no));
        }

        let mut intents = Vec::with_capacity(extents.len() + pages.len() + rows.len());
        let mut mapped_extents = BTreeMap::new();
        let mut next_physical = BTreeMap::<u16, u64>::new();
        for (store_id, logical_extent) in extents {
            let store = self.store(tenant, store_id)?;
            let mapped = store.data.directory().mapping(logical_extent)?.is_some();
            mapped_extents.insert((store_id, logical_extent), mapped);
            if mapped {
                continue;
            }
            let physical_offset = match next_physical.get(&store_id).copied() {
                Some(offset) => offset,
                None => store.data.directory().recover_next_physical_offset()?,
            };
            next_physical.insert(
                store_id,
                physical_offset
                    .checked_add(EXTENT_BYTES)
                    .ok_or(OwnerRowError::PhysicalOffsetOverflow { store_id })?,
            );
            intents.push(DeltaIntent::extent_alloc(
                store_id,
                tenant,
                ExtentAllocation {
                    logical_extent,
                    physical_offset,
                    pairing: u32::try_from(logical_extent).unwrap_or(u32::MAX),
                },
            ));
        }

        for (store_id, page_no) in pages {
            let logical_extent = page_no / EXTENT_PAGES;
            let mapped = mapped_extents[&(store_id, logical_extent)];
            let formatted = if mapped {
                self.store(tenant, store_id)?
                    .data
                    .read_page_for_redo(tenant, PageId::new(page_no))?
                    .is_some_and(|page| page.iter().any(|byte| *byte != 0))
            } else {
                false
            };
            if !formatted {
                intents.push(DeltaIntent::page_alloc(
                    store_id,
                    tenant,
                    page_no,
                    PageType::PropSlotted,
                    1,
                ));
            }
        }

        for row in rows {
            let class = row.class();
            let address = class.address(row.id())?;
            intents.push(DeltaIntent {
                kind: class.delta_kind(),
                store_id: class.store_id(),
                tenant_id: tenant,
                page_no: address.page_no,
                slot: address.slot.raw(),
                payload: Bytes::copy_from_slice(&row.encode()),
            });
        }
        Ok(intents)
    }

    pub(crate) fn apply_committed(
        &self,
        deltas: &[DeltaOp],
        commit_lsn: Lsn,
    ) -> arcgraph_core::Result<()> {
        for op in deltas {
            let Some(store) = self.stores.get(&(op.tenant_id, op.store_id)) else {
                continue;
            };
            if op.kind == DeltaOpKind::ExtentAlloc {
                store
                    .data
                    .directory()
                    .apply_extent_alloc(op, self.dpt.as_ref())?;
                continue;
            }
            crate::redo::apply_recovery_delta(
                store.data.as_ref(),
                store.data.as_ref(),
                self.dpt.as_ref(),
                op,
                commit_lsn,
            )?;
        }
        Ok(())
    }
}

/// Stable owner-store census shared by validation, replay, and bootstrap.
#[must_use]
pub const fn is_owner_store_id(store_id: u16) -> bool {
    matches!(
        store_id,
        STORE_NODE_BINDINGS | STORE_REL_BINDINGS | STORE_INTERN | STORE_GRANTS
    )
}

fn classes_for_store(store_id: u16) -> Vec<OwnerRowClass> {
    match store_id {
        STORE_NODE_BINDINGS => vec![OwnerRowClass::NodeBinding],
        STORE_REL_BINDINGS => vec![OwnerRowClass::RelBinding],
        STORE_INTERN => vec![OwnerRowClass::InternedString, OwnerRowClass::ClassId],
        STORE_GRANTS => vec![OwnerRowClass::Grant],
        _ => Vec::new(),
    }
}

/// Validate the complete owner delta shape without consulting process state.
pub(crate) fn validate_owner_delta(
    kind: DeltaOpKind,
    store_id: u16,
    page_no: u64,
    slot: u16,
    payload: &[u8],
    lsn: Lsn,
) -> arcgraph_core::Result<()> {
    if payload.len() != OWNER_ROW_BYTES {
        return Err(corruption(
            lsn,
            format!(
                "owner delta has {} payload bytes, expected {OWNER_ROW_BYTES}",
                payload.len()
            ),
        ));
    }
    let class = OwnerRowClass::from_byte(payload[8])
        .map_err(|error| corruption(lsn, format!("owner delta class is invalid: {error}")))?;
    let id = u64::from_le_bytes(payload[12..20].try_into().expect("fixed owner id"));
    let row = OwnerRow::decode(payload, class, id)
        .map_err(|error| corruption(lsn, format!("owner delta row is invalid: {error}")))?;
    let address = row
        .class()
        .address(row.id())
        .map_err(|error| corruption(lsn, format!("owner delta address is invalid: {error}")))?;
    if row.class().store_id() != store_id
        || row.class().delta_kind() != kind
        || address.page_no != page_no
        || address.slot.raw() != slot
    {
        return Err(corruption(
            lsn,
            "owner delta store/kind/page/slot disagrees with its self-identifying row",
        ));
    }
    Ok(())
}

/// The bundle validator has already checked the full owner envelope before
/// redo calls this; this helper keeps the stable state byte private to the
/// owner codec.
pub(crate) fn owner_delta_is_retirement(payload: &[u8]) -> bool {
    payload.get(9) == Some(&OWNER_ROW_STATE_RETIRED)
}

#[cfg(any(test, feature = "fault-injection"))]
fn owner_phase1_pause_requested(rows: &[OwnerRow]) -> Option<std::path::PathBuf> {
    let requested_id = std::env::var("ARCGRAPH_OWNER_ROW_PHASE1_PAUSE_ID")
        .ok()?
        .parse::<u64>()
        .ok()?;
    rows.iter()
        .any(|row| row.id() == requested_id)
        .then(|| std::env::var_os("ARCGRAPH_OWNER_ROW_PHASE1_READY"))
        .flatten()
        .map(Into::into)
}

#[cfg(any(test, feature = "fault-injection"))]
fn owner_phase1_pause(path: &std::path::Path) -> arcgraph_core::Result<()> {
    const TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);
    const POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(25);

    std::fs::write(path, b"phase1-ready")?;
    let started = std::time::Instant::now();
    while started.elapsed() < TIMEOUT {
        std::thread::park_timeout(POLL_INTERVAL.min(TIMEOUT.saturating_sub(started.elapsed())));
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::TimedOut,
        "owner-row Phase-1 fault rendezvous timed out after 10 seconds",
    )
    .into())
}

fn corruption(lsn: Lsn, reason: impl Into<String>) -> ArcGraphError {
    ArcGraphError::WalCorruption {
        lsn,
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    use arcgraph_core::NodeId;
    use proptest::prelude::*;

    use crate::extent::{ExtentDataPageStore, ExtentDirectory};
    use crate::idempotency::IdempotencyStore;
    use crate::intern::InternTable;
    use crate::io::{PageIo, PosixPageIo};
    use crate::permissions::PermissionIndex;
    use crate::redo::RedoLsnRange;

    static NEXT_TEST_LSN: AtomicU64 = AtomicU64::new(10_000);

    struct TestOwnerHome {
        root: tempfile::TempDir,
        tenant: TenantId,
        registry: Arc<OwnerRowRegistry>,
    }

    impl TestOwnerHome {
        fn create(tenant: TenantId) -> Self {
            let root = tempfile::tempdir().unwrap();
            for class in OwnerRowClass::ALL {
                let payload_path = owner_payload_path(root.path(), tenant, class);
                std::fs::create_dir_all(payload_path.parent().unwrap()).unwrap();
                drop(
                    OwnerPayloadStore::create(&payload_path, OWNER_PAYLOAD_DISK_CAP_BYTES).unwrap(),
                );
                if let Some(index_path) = owner_forward_index_path(root.path(), tenant, class) {
                    drop(
                        OwnerForwardIndex::create(&index_path, OWNER_INDEX_DISK_CAP_BYTES).unwrap(),
                    );
                }
            }
            let registry = Self::open_registry(root.path(), tenant, true);
            Self {
                root,
                tenant,
                registry,
            }
        }

        fn reopen(mut self) -> Self {
            self.registry.flush_all().unwrap();
            drop(self.registry);
            self.registry = Self::open_registry(self.root.path(), self.tenant, false);
            self
        }

        fn open_registry(root: &Path, tenant: TenantId, create: bool) -> Arc<OwnerRowRegistry> {
            let stores = [
                STORE_NODE_BINDINGS,
                STORE_REL_BINDINGS,
                STORE_INTERN,
                STORE_GRANTS,
            ]
            .into_iter()
            .map(|store_id| {
                let path = root.join(format!("owner-{store_id}.store"));
                let physical: Arc<dyn PageIo> = if create {
                    Arc::new(PosixPageIo::create(path).unwrap())
                } else {
                    Arc::new(PosixPageIo::open(path).unwrap())
                };
                let directory = Arc::new(ExtentDirectory::new(tenant, store_id, physical, 16));
                Arc::new(OwnerRowStore::new(Arc::new(ExtentDataPageStore::new(
                    directory, 32,
                ))))
            });
            Arc::new(
                OwnerRowRegistry::open_logical(root, stores, Arc::new(DirtyPageTable::new()))
                    .unwrap(),
            )
        }

        fn apply(&self, rows: &[OwnerRow]) {
            if rows.is_empty() {
                return;
            }
            let _planning = self.registry.planning_guard();
            let intents = self.registry.plan_rows_locked(self.tenant, rows).unwrap();
            let commit_raw = NEXT_TEST_LSN.fetch_add(100, Ordering::AcqRel) + 100;
            let commit_lsn = Lsn::new(commit_raw);
            let range = RedoLsnRange::ending_at(commit_lsn, intents.len()).unwrap();
            let ops: Vec<_> = intents
                .into_iter()
                .enumerate()
                .map(|(index, intent)| {
                    intent
                        .assign_for_format(
                            range.op_lsn(index).unwrap(),
                            commit_lsn,
                            BUNDLE_FORMAT_V10,
                        )
                        .unwrap()
                })
                .collect();
            self.registry.apply_committed(&ops, commit_lsn).unwrap();
        }

        fn indexed_row(
            &self,
            class: OwnerRowClass,
            id: u64,
            hash: u64,
            logical: &[u8],
        ) -> OwnerRow {
            self.registry
                .prepare_indexed_logical_row(self.tenant, class, id, hash, logical)
                .unwrap()
        }

        fn direct_row(&self, class: OwnerRowClass, id: u64, logical: &[u8]) -> OwnerRow {
            self.registry
                .prepare_direct_logical_row(self.tenant, class, id, logical)
                .unwrap()
        }
    }

    /// DEFERRAL-SAFETY gate for #1493 (owner forward index has no reclamation
    /// path). PR #1492 defers compaction of dead candidates; that deferral is
    /// only safe if an orphan candidate is provably INERT.
    ///
    /// `prepare_indexed_logical_row` fsync-publishes the forward candidate
    /// BEFORE the owner-row commit, and the index is append-only immutable runs
    /// — there is no un-publish. So an aborted v10 commit leaves a candidate
    /// `(hash -> id)` pointing at a row that never committed.
    ///
    /// The contract that makes this harmless is full-row verification: the
    /// production lookup (`find_verified`) reads the owner row for every
    /// candidate id and skips it when the row is absent. This gate pins that
    /// contract, with a DELIBERATELY PERMISSIVE `verify` closure — it returns
    /// `true` for anything it is handed — so the ONLY thing that can keep the
    /// orphan from resolving is the missing row itself. If someone ever
    /// "optimizes" the row read away and trusts the candidate, this fails.
    ///
    /// RED-on-revert: make `find_verified` trust a candidate without reading its
    /// row, and the orphan resolves.
    #[test]
    fn gate_orphan_index_candidate_from_aborted_commit_never_resolves() {
        let tenant = TenantId::new(73); // NON-DEFAULT
        let home = TestOwnerHome::create(tenant);
        let class = OwnerRowClass::NodeBinding;
        let external = "orphan-from-aborted-commit";
        let hash = str_hash_56(external);
        let orphan_id = 4_242_u64;

        let logical = BindingOwnerValue {
            kind: 0,
            external_id: external.to_owned(),
            payload_hash: None,
            active: true,
        }
        .encode()
        .unwrap();

        // Publish the candidate, then ABORT: never apply the row.
        let _row = home
            .registry
            .prepare_indexed_logical_row(tenant, class, orphan_id, hash, &logical)
            .unwrap();

        // NON-VACUITY: the orphan candidate really is durably in the index.
        // (Without this, the gate could pass simply because nothing was published.)
        let seen = std::cell::RefCell::new(Vec::new());
        home.registry
            .index(tenant, class)
            .unwrap()
            .for_each_candidate(hash, |id| {
                seen.borrow_mut().push(id);
                Ok(false) // keep walking; we only want to observe the candidate set
            })
            .unwrap();
        assert!(
            seen.borrow().contains(&orphan_id),
            "orphan candidate was never published — this gate would be vacuous"
        );

        // The production lookup must NOT resolve it, even though `verify` says
        // yes to everything: the owner row does not exist, so it is skipped.
        let resolved = home
            .registry
            .find_verified(tenant, class, hash, |_id, _logical| true)
            .unwrap();
        assert!(
            resolved.is_none(),
            "an orphan candidate from an aborted commit RESOLVED ({resolved:?}) — \
             full-row verification is what makes deferring index reclamation (#1493) \
             safe; without it the orphan is a live wrong answer"
        );

        // And once the row IS committed, the same candidate resolves normally —
        // proving the miss above came from the absent row, not a broken index.
        let row = home
            .registry
            .prepare_indexed_logical_row(tenant, class, orphan_id, hash, &logical)
            .unwrap();
        home.apply(&[row]);
        let resolved = home
            .registry
            .find_verified(tenant, class, hash, |_id, _logical| true)
            .unwrap();
        assert_eq!(
            resolved.map(|(id, _)| id),
            Some(orphan_id),
            "control: a COMMITTED row must resolve through the same candidate"
        );
    }

    /// ROOT CAUSE B gate — `slot_count` lives OUTSIDE the page CRC.
    ///
    /// The page CRC covers `bytes[PageHeader::SIZE..]` = bytes 40.., but
    /// `slot_count` sits in the header at bytes 36..38. A bit-flip that LOWERS
    /// it therefore keeps the checksum valid: `SlottedPageRef::open` succeeds,
    /// every live row at a slot >= the forged count reports
    /// `PageError::SlotOutOfRange`, and the owner taxonomy seam used to map
    /// that straight to `Ok(None)`. Live idempotency / intern / ACL rows
    /// silently VANISH — duplicate ingests, never-released bindings, and
    /// disappeared label + type names, with no error anywhere.
    ///
    /// The existing `owner_row_substrate` corruption cases all either recompute
    /// the CRC or flip a BODY byte, so they cannot see this class at all.
    ///
    /// The fix leans on the fact that the slot DIRECTORY (byte 40+) IS inside
    /// the CRC: a corruptor cannot lower `slot_count` and also erase the
    /// directory entry without breaking the checksum, so a directory entry
    /// still addressing a live owner row is integrity-protected proof that the
    /// header is lying.
    ///
    /// RED-on-revert: restore `Err(PageError::SlotOutOfRange { .. }) => Ok(None)`
    /// in [`owner_direct_row_disposition`] — the forged page then reports the
    /// live row as absent and this gate fails.
    #[test]
    fn gate_forged_slot_count_cannot_erase_live_owner_rows_into_notfound() {
        use crate::records::SlottedPage;
        use arcgraph_core::PageHeader;
        use arcgraph_core::record::PAGE_SIZE;

        let tenant = TenantId::new(73); // NON-DEFAULT
        let class = OwnerRowClass::InternedString;
        const LIVE_SLOTS: u16 = 6;
        let victim = SlotId(5); // live, and >= the count we are about to forge

        // Build a REAL owner page through the production writer (the same
        // `put_bag_at` the redo path uses to apply owner rows).
        let mut bytes = vec![0_u8; PAGE_SIZE];
        let header = PageHeader::new(PageId::new(1), PageType::PropSlotted, tenant);
        {
            let mut page = SlottedPage::init(&mut bytes, header).unwrap();
            for slot in 0..LIVE_SLOTS {
                let row = OwnerRow::new(
                    class,
                    u64::from(slot),
                    format!("owner-row-payload-{slot}").into_bytes(),
                )
                .unwrap();
                page.put_bag_at(SlotId(slot), &row.encode()).unwrap();
            }
        }

        // Control: the victim row reads back through the seam.
        {
            let view = SlottedPageRef::open(&bytes).unwrap();
            assert_eq!(view.slot_count(), LIVE_SLOTS);
            let got = owner_direct_row_disposition(view.read_bag(victim), &bytes, victim).unwrap();
            assert!(got.is_some(), "control: the live row must read back");
        }

        // TAMPER: lower slot_count 6 -> 2 in the header. Do NOT recompute the CRC.
        let forged: u16 = 2;
        bytes[36..38].copy_from_slice(&forged.to_le_bytes());

        // THE HOLE, asserted explicitly: the tampered page still passes checksum
        // verification, because the CRC does not cover the header. If this ever
        // starts failing, `slot_count` has been brought under the CRC and this
        // gate should be revisited (the fix would then be redundant, not wrong).
        let view = SlottedPageRef::open(&bytes).expect(
            "slot_count is outside the page CRC, so a forged count still verifies — \
             this is the defect under test",
        );
        assert_eq!(view.slot_count(), forged);
        assert!(
            matches!(view.read_bag(victim), Err(PageError::SlotOutOfRange { .. })),
            "the forged header must make the live slot look out of range"
        );

        // The seam must refuse to call a durable row absent.
        let error = owner_direct_row_disposition(view.read_bag(victim), &bytes, victim)
            .expect_err("a forged slot_count must NOT erase a live owner row into NotFound");
        assert!(
            matches!(
                error,
                OwnerRowError::HeaderSlotCountUnderflow { slot, header_slot_count, .. }
                    if slot == victim.0 && header_slot_count == forged
            ),
            "expected a HARD corruption error, got {error:?}"
        );

        // And a genuinely-unused slot above the high-water is still a clean miss:
        // the fix must not turn legitimate emptiness into a corruption error.
        let unused = SlotId(LIVE_SLOTS + 3);
        let got = owner_direct_row_disposition(view.read_bag(unused), &bytes, unused)
            .expect("an unused slot is a legitimate miss, not corruption");
        assert!(got.is_none());
    }

    #[test]
    fn fixed_row_round_trip_and_identity_checks() {
        for class in OwnerRowClass::ALL {
            let row = OwnerRow::new(class, 17, format!("{class:?}").into_bytes()).unwrap();
            assert_eq!(OwnerRow::decode(&row.encode(), class, 17).unwrap(), row);
        }
    }

    #[test]
    fn owner_binding_retirement_reuses_pre_b4_permanent_tombstone() {
        let tenant = TenantId::new(37);
        let home = TestOwnerHome::create(tenant);
        let logical = BindingOwnerValue {
            kind: 0,
            external_id: "retired-binding".to_owned(),
            payload_hash: None,
            active: true,
        }
        .encode()
        .unwrap();
        let live = home.indexed_row(
            OwnerRowClass::NodeBinding,
            5,
            str_hash_56("retired-binding"),
            &logical,
        );
        home.apply(&[live]);
        let retired = home
            .registry
            .prepare_retired_binding_row(tenant, OwnerRowClass::NodeBinding, 5)
            .unwrap();
        home.apply(&[retired]);
        assert!(
            home.registry
                .read(tenant, OwnerRowClass::NodeBinding, 5)
                .unwrap()
                .is_none()
        );

        let address = OwnerRowClass::NodeBinding.address(5).unwrap();
        let store = home
            .registry
            .store(tenant, OwnerRowClass::NodeBinding.store_id())
            .unwrap();
        let page = store
            .data
            .read_page_for_redo(tenant, PageId::new(address.page_no))
            .unwrap()
            .unwrap();
        let view = SlottedPageRef::open(page.as_ref()).unwrap();
        assert!(view.is_permanent_tombstone(address.slot).unwrap());
        assert!(
            view.permanent_tombstone_lsn(address.slot, OWNER_ROW_BYTES as u16)
                .unwrap()
                .is_some()
        );

        let home = home.reopen();
        assert!(
            home.registry
                .read(tenant, OwnerRowClass::NodeBinding, 5)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn class_regions_are_disjoint_within_shared_intern_store() {
        let intern = OwnerRowClass::InternedString.address(0).unwrap();
        let class = OwnerRowClass::ClassId.address(0).unwrap();
        assert_ne!(intern.page_no, class.page_no);
        assert_eq!(intern.slot, class.slot);
    }

    #[test]
    fn checkpoint_owner_census_field_complete_no_owner_dropped() {
        let tenant = TenantId::new(77);
        let home = TestOwnerHome::create(tenant);
        let node = BindingOwnerValue {
            kind: 0,
            external_id: "node-ext".to_owned(),
            payload_hash: Some(11),
            active: true,
        }
        .encode()
        .unwrap();
        let rel = BindingOwnerValue {
            kind: 1,
            external_id: "rel-ext".to_owned(),
            payload_hash: Some(12),
            active: true,
        }
        .encode()
        .unwrap();
        let intern = InternOwnerValue {
            name: "Customer".to_owned(),
        }
        .encode()
        .unwrap();
        let class = ClassOwnerValue {
            grants: ["alice".to_owned()].into_iter().collect(),
        };
        let class_bytes = class.encode().unwrap();
        let grant = GrantOwnerValue {
            class_id: 0,
            active: true,
        }
        .encode();
        let rows = vec![
            home.indexed_row(
                OwnerRowClass::NodeBinding,
                1,
                str_hash_56("node-ext"),
                &node,
            ),
            home.indexed_row(OwnerRowClass::RelBinding, 2, str_hash_56("rel-ext"), &rel),
            home.indexed_row(
                OwnerRowClass::InternedString,
                3,
                str_hash_56("Customer"),
                &intern,
            ),
            home.indexed_row(
                OwnerRowClass::ClassId,
                0,
                class.hash().unwrap(),
                &class_bytes,
            ),
            home.direct_row(OwnerRowClass::Grant, 4, &grant),
        ];
        home.apply(&rows);
        let home = home.reopen();
        assert!(home.registry.logical_companions_complete());
        for (class, id) in [
            (OwnerRowClass::NodeBinding, 1),
            (OwnerRowClass::RelBinding, 2),
            (OwnerRowClass::InternedString, 3),
            (OwnerRowClass::ClassId, 0),
            (OwnerRowClass::Grant, 4),
        ] {
            assert!(
                home.registry
                    .read_logical(tenant, class, id)
                    .unwrap()
                    .is_some()
            );
        }
        let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&home.registry)));
        let intern = InternTable::page_backed(Arc::clone(&home.registry)).unwrap();
        assert!(!idempotency.has_resident_owner_maps());
        let permissions =
            PermissionIndex::page_backed(Arc::clone(&home.registry), idempotency, tenant).unwrap();
        assert!(!intern.has_resident_owner_maps());
        assert!(!permissions.has_resident_owner_maps());
        assert_eq!(intern.name_count(), 0, "intern metadata capture retired");
        assert_eq!(
            permissions.doc_grant_count(),
            0,
            "grant metadata capture retired"
        );
    }

    #[test]
    #[allow(non_snake_case)] // Ratified gate name uses uppercase N.
    fn residency_census_no_owner_scales_with_N() {
        let tenant = TenantId::new(91);
        let home = TestOwnerHome::create(tenant);
        let class = ClassOwnerValue {
            grants: ["alice".to_owned()].into_iter().collect(),
        };
        let class_bytes = class.encode().unwrap();
        let mut rows = vec![home.indexed_row(
            OwnerRowClass::ClassId,
            0,
            class.hash().unwrap(),
            &class_bytes,
        )];
        for id in 1_u64..=64 {
            let external = format!("binding-{id}");
            rows.push(
                home.indexed_row(
                    OwnerRowClass::NodeBinding,
                    id,
                    str_hash_56(&external),
                    &BindingOwnerValue {
                        kind: 0,
                        external_id: external,
                        payload_hash: Some(id),
                        active: true,
                    }
                    .encode()
                    .unwrap(),
                ),
            );
            let name = format!("Interned-{id}");
            rows.push(home.indexed_row(
                OwnerRowClass::InternedString,
                id,
                str_hash_56(&name),
                &InternOwnerValue { name }.encode().unwrap(),
            ));
            rows.push(
                home.direct_row(
                    OwnerRowClass::Grant,
                    id,
                    &GrantOwnerValue {
                        class_id: 0,
                        active: true,
                    }
                    .encode(),
                ),
            );
        }
        home.apply(&rows);
        let home = home.reopen();
        let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&home.registry)));
        let intern = InternTable::page_backed(Arc::clone(&home.registry)).unwrap();
        let permissions = PermissionIndex::page_backed(
            Arc::clone(&home.registry),
            Arc::clone(&idempotency),
            tenant,
        )
        .unwrap();
        assert_eq!(
            idempotency
                .try_get(tenant, 0, "binding-64")
                .unwrap()
                .unwrap()
                .internal_id,
            64
        );
        assert_eq!(
            intern
                .try_probe(tenant, "Interned-64")
                .unwrap()
                .unwrap()
                .raw(),
            64
        );
        assert!(
            permissions
                .effective("alice")
                .try_is_visible(NodeId::new(64))
                .unwrap()
        );
        assert!(idempotency.is_page_backed());
        assert!(!idempotency.has_resident_owner_maps());
        assert!(!intern.has_resident_owner_maps());
        assert!(!permissions.has_resident_owner_maps());
        assert_eq!(idempotency.resident_len(), 0);
        assert_eq!(idempotency.resident_reverse_len(), 0);
        assert_eq!(intern.resident_map_cardinalities(), [0, 0, 0]);
        assert_eq!(permissions.resident_map_cardinalities(), [0, 0, 0, 0, 0]);
        let (cache_len, cache_cap) = permissions.physical_cache_census();
        assert_eq!(cache_len, 1);
        assert_eq!(cache_cap, 1_024);
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]
        #[test]
        fn strash_collision_mandatory_recheck(
            wrong in "[a-z]{1,24}",
            wanted in "[A-Z]{1,24}",
        ) {
            prop_assume!(wrong != wanted);
            let tenant = TenantId::new(123);
            let home = TestOwnerHome::create(tenant);
            let forced_collision = 0x00ab_cdef_u64;
            let wrong_value = BindingOwnerValue {
                kind: 0,
                external_id: wrong.clone(),
                payload_hash: None,
                active: true,
            }.encode().unwrap();
            let wanted_value = BindingOwnerValue {
                kind: 0,
                external_id: wanted.clone(),
                payload_hash: None,
                active: true,
            }.encode().unwrap();
            let rows = [
                home.indexed_row(OwnerRowClass::NodeBinding, 1, forced_collision, &wrong_value),
                home.indexed_row(OwnerRowClass::NodeBinding, 2, forced_collision, &wanted_value),
            ];
            home.apply(&rows);
            let found = home.registry.find_verified(
                tenant,
                OwnerRowClass::NodeBinding,
                forced_collision,
                |_id, logical| BindingOwnerValue::decode(logical)
                    .is_ok_and(|value| value.external_id == wanted),
            ).unwrap();
            prop_assert_eq!(found.map(|(id, _)| id), Some(2));
        }
    }

    #[test]
    fn binding_spill_durable_queryable() {
        let tenant = TenantId::new(456);
        let home = TestOwnerHome::create(tenant);
        let external = "durable-binding";
        let logical = BindingOwnerValue {
            kind: 0,
            external_id: external.to_owned(),
            payload_hash: Some(0x55),
            active: true,
        }
        .encode()
        .unwrap();
        let row = home.indexed_row(
            OwnerRowClass::NodeBinding,
            99,
            str_hash_56(external),
            &logical,
        );
        home.apply(&[row]);
        let home = home.reopen();
        let facade = IdempotencyStore::page_backed(Arc::clone(&home.registry));
        let found = facade.try_get(tenant, 0, external).unwrap().unwrap();
        assert_eq!(found.internal_id, 99);
        assert_eq!(found.payload_hash, Some(0x55));
        assert_eq!(facade.resident_len(), 0);
        assert_eq!(facade.resident_reverse_len(), 0);
    }

    #[test]
    fn reverse_lookup_single_fault() {
        let tenant = TenantId::new(789);
        let home = TestOwnerHome::create(tenant);
        let logical = BindingOwnerValue {
            kind: 1,
            external_id: "rel-r".to_owned(),
            payload_hash: None,
            active: true,
        }
        .encode()
        .unwrap();
        let row = home.indexed_row(
            OwnerRowClass::RelBinding,
            17,
            str_hash_56("rel-r"),
            &logical,
        );
        home.apply(&[row]);
        let facade = IdempotencyStore::page_backed(Arc::clone(&home.registry));
        home.registry
            .reset_read_fault_count(tenant, OwnerRowClass::RelBinding)
            .unwrap();
        assert_eq!(
            facade
                .try_external_id_for(tenant, 1, 17)
                .unwrap()
                .as_deref(),
            Some("rel-r")
        );
        assert_eq!(
            home.registry
                .read_fault_count(tenant, OwnerRowClass::RelBinding)
                .unwrap(),
            1
        );
    }

    #[test]
    fn is_visible_faults_grants_store_no_resident_docmap() {
        let tenant = TenantId::new(991);
        let home = TestOwnerHome::create(tenant);
        let class = ClassOwnerValue {
            grants: ["alice".to_owned()].into_iter().collect(),
        };
        let encoded = class.encode().unwrap();
        let rows = [
            home.indexed_row(OwnerRowClass::ClassId, 0, class.hash().unwrap(), &encoded),
            home.direct_row(
                OwnerRowClass::Grant,
                41,
                &GrantOwnerValue {
                    class_id: 0,
                    active: true,
                }
                .encode(),
            ),
        ];
        home.apply(&rows);
        let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&home.registry)));
        let permissions =
            PermissionIndex::page_backed(Arc::clone(&home.registry), idempotency, tenant).unwrap();
        home.registry
            .reset_read_fault_count(tenant, OwnerRowClass::Grant)
            .unwrap();
        assert!(
            permissions
                .effective("alice")
                .try_is_visible(NodeId::new(41))
                .unwrap()
        );
        assert_eq!(
            home.registry
                .read_fault_count(tenant, OwnerRowClass::Grant)
                .unwrap(),
            1
        );
        assert_eq!(permissions.resident_map_cardinalities(), [0, 0, 0, 0, 0]);
    }

    #[test]
    fn classid_durable_fetch_max_seeded() {
        let tenant = TenantId::new(222);
        let home = TestOwnerHome::create(tenant);
        let class = ClassOwnerValue {
            grants: ["alice".to_owned()].into_iter().collect(),
        };
        let encoded = class.encode().unwrap();
        let rows = [
            home.indexed_row(OwnerRowClass::ClassId, 7, class.hash().unwrap(), &encoded),
            home.direct_row(
                OwnerRowClass::ClassId,
                OWNER_ALLOCATOR_MARKER_ID,
                &OwnerAllocatorMarker {
                    kind: crate::wal::AllocatorKind::AclClass.as_byte(),
                    high_water: 7,
                }
                .encode(),
            ),
            home.direct_row(
                OwnerRowClass::Grant,
                88,
                &GrantOwnerValue {
                    class_id: 7,
                    active: true,
                }
                .encode(),
            ),
        ];
        home.apply(&rows);
        let home = home.reopen();
        let idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(&home.registry)));
        let permissions =
            PermissionIndex::page_backed(Arc::clone(&home.registry), idempotency, tenant).unwrap();
        assert_eq!(permissions.class_allocator_next(), 8);
        assert!(
            permissions
                .effective("alice")
                .try_is_visible(NodeId::new(88))
                .unwrap()
        );
    }
}
