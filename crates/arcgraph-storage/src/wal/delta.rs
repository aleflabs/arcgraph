//! M3+ physiological WAL operation wire format.

use arcgraph_core::{ArcGraphError, Lsn, PageType, Result, TenantId};
use bytes::Bytes;

use crate::extent::ExtentAllocation;
use crate::records::{MAX_SLOTS, PROP_BAG_MAX_BYTES};
use crate::wal::bundle::{BUNDLE_FORMAT_V9, BUNDLE_FORMAT_V10};

/// Recover the MVCC key/value carried by a `PutRecord` delta.
///
/// M3 logs record bytes once: the physical delta is also the authoritative
/// MVCC version payload. The v9 encoder uses this identity to omit a redundant
/// section-2 copy, and replay uses the same validated projection to rebuild
/// the version chain.
pub(crate) fn put_record_mvcc_write(op: &DeltaOp) -> Result<Option<(u64, Bytes)>> {
    if op.kind != DeltaOpKind::PutRecord {
        return Ok(None);
    }
    let key = match op.payload.len() {
        arcgraph_core::record::NodeRecord::SIZE => u64::from_le_bytes(
            op.payload[..8]
                .try_into()
                .expect("PutRecord node id is present"),
        ),
        arcgraph_core::record::RelRecord::SIZE => {
            let id = u64::from_le_bytes(
                op.payload[..8]
                    .try_into()
                    .expect("PutRecord relationship id is present"),
            );
            id | (1u64 << 63)
        }
        _ => {
            return Err(corruption(
                op.op_lsn,
                "PutRecord payload is not a node or relationship record",
            ));
        }
    };
    Ok(Some((key, op.payload.clone())))
}

/// `props.store` discriminator.
pub const STORE_PROPS: u16 = 0;
/// `record.store` discriminator.
pub const STORE_RECORD: u16 = 1;
/// Reserved M4 `tel.store` discriminator.
pub const STORE_TEL: u16 = 2;
/// Primary-index discriminator (page-image at M3).
pub const STORE_PRIMARY_INDEX: u16 = 3;
/// Secondary-index discriminator (page-image at M3).
pub const STORE_SECONDARY_INDEX: u16 = 4;
/// `blob.overflow` discriminator. Reserved for physical deltas until M4;
/// M3 carries this store as checkpoint page images.
pub const STORE_BLOB_OVERFLOW: u16 = 5;
/// M4 `rels.store` discriminator. Node records retain store id 1.
pub const STORE_RELS: u16 = 6;
/// M4 permanent node idempotency-binding store.
pub const STORE_NODE_BINDINGS: u16 = 7;
/// M4 permanent relationship idempotency-binding store.
pub const STORE_REL_BINDINGS: u16 = 8;
/// M4 permanent intern/name store.
pub const STORE_INTERN: u16 = 9;
/// M4 permanent document-grant store.
pub const STORE_GRANTS: u16 = 10;

/// Largest typed property block that fits in one `props.store` slot.
pub const MAX_PROP_BLOCK_PAYLOAD: usize = PROP_BAG_MAX_BYTES;

/// A physical redo operation before the commit owns an LSN range.
/// Producers build these while reserving targets, then assign one unique
/// sub-LSN to each intent after the exact operation count is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaIntent {
    pub kind: DeltaOpKind,
    pub store_id: u16,
    pub tenant_id: TenantId,
    pub page_no: u64,
    pub slot: u16,
    pub payload: Bytes,
}

impl DeltaIntent {
    #[must_use]
    pub fn page_alloc(
        store_id: u16,
        tenant_id: TenantId,
        page_no: u64,
        page_type: arcgraph_core::PageType,
        generation: u64,
    ) -> Self {
        let mut payload = Vec::with_capacity(9);
        payload.push(page_type.as_byte());
        payload.extend_from_slice(&generation.to_le_bytes());
        Self {
            kind: DeltaOpKind::PageAlloc,
            store_id,
            tenant_id,
            page_no,
            slot: 0,
            payload: Bytes::from(payload),
        }
    }

    /// Build the physical directory-page mutation for one new extent.
    #[must_use]
    pub fn extent_alloc(store_id: u16, tenant_id: TenantId, allocation: ExtentAllocation) -> Self {
        Self {
            kind: DeltaOpKind::ExtentAlloc,
            store_id,
            tenant_id,
            page_no: allocation
                .directory_page_no()
                .expect("validated logical extent has a directory page"),
            slot: 0,
            payload: Bytes::copy_from_slice(&allocation.encode()),
        }
    }

    pub fn assign(self, op_lsn: Lsn, commit_lsn: Lsn) -> Result<DeltaOp> {
        self.assign_for_format(op_lsn, commit_lsn, BUNDLE_FORMAT_V9)
    }

    /// Assign this intent under the declared delta-bundle format contract.
    pub fn assign_for_format(
        mut self,
        op_lsn: Lsn,
        commit_lsn: Lsn,
        format_version: u16,
    ) -> Result<DeltaOp> {
        if self.kind == DeltaOpKind::PutRecord {
            let created_lsn_offset = match self.payload.len() {
                64 => 56,
                96 => 48,
                _ => {
                    return Err(corruption(
                        op_lsn,
                        "PutRecord intent payload is not a node or relationship record",
                    ));
                }
            };
            let mut payload = self.payload.to_vec();
            payload[created_lsn_offset..created_lsn_offset + 8]
                .copy_from_slice(&commit_lsn.raw().to_le_bytes());
            self.payload = Bytes::from(payload);
        }
        DeltaOp::new_for_format(
            format_version,
            self.kind,
            self.store_id,
            self.tenant_id,
            self.page_no,
            self.slot,
            op_lsn,
            self.payload,
        )
    }
}

/// Stable M3+ physiological operation discriminants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[repr(u8)]
pub enum DeltaOpKind {
    PutRecord = 0,
    TombstoneRecord = 1,
    PutPropBlock = 2,
    TelAppend = 3,
    TelExpire = 4,
    IndexPut = 5,
    IndexDelete = 6,
    AllocAdvance = 7,
    InternBind = 8,
    AclGrant = 9,
    PageAlloc = 10,
    ExtentAlloc = 11,
    VectorDelta = 12,
    TelGrow = 13,
}

impl DeltaOpKind {
    /// Parse every wire-stable discriminant, including reserved ones.
    pub fn from_byte(byte: u8, lsn: Lsn) -> Result<Self> {
        Ok(match byte {
            0 => Self::PutRecord,
            1 => Self::TombstoneRecord,
            2 => Self::PutPropBlock,
            3 => Self::TelAppend,
            4 => Self::TelExpire,
            5 => Self::IndexPut,
            6 => Self::IndexDelete,
            7 => Self::AllocAdvance,
            8 => Self::InternBind,
            9 => Self::AclGrant,
            10 => Self::PageAlloc,
            11 => Self::ExtentAlloc,
            12 => Self::VectorDelta,
            13 => Self::TelGrow,
            other => return Err(corruption(lsn, format!("unknown DeltaOp kind {other}"))),
        })
    }

    #[must_use]
    pub const fn as_byte(self) -> u8 {
        self as u8
    }

    /// Kinds emitted and accepted by an M3/v9 bundle.
    #[must_use]
    pub const fn is_emitted_at_m3(self) -> bool {
        matches!(
            self,
            Self::PutRecord
                | Self::TombstoneRecord
                | Self::PutPropBlock
                | Self::AllocAdvance
                | Self::PageAlloc
                | Self::ExtentAlloc
        )
    }

    /// Whether this kind is legal in the bundle's declared WAL format.
    #[must_use]
    pub const fn is_emitted_for_format(self, format_version: u16) -> bool {
        match format_version {
            BUNDLE_FORMAT_V9 => self.is_emitted_at_m3(),
            BUNDLE_FORMAT_V10 => {
                self.is_emitted_at_m3() || matches!(self, Self::InternBind | Self::AclGrant)
            }
            _ => false,
        }
    }

    #[must_use]
    pub const fn is_physical(self) -> bool {
        matches!(
            self,
            Self::PutRecord
                | Self::TombstoneRecord
                | Self::PutPropBlock
                | Self::InternBind
                | Self::AclGrant
                | Self::PageAlloc
                | Self::ExtentAlloc
        )
    }
}

/// One v9+ physiological WAL op.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeltaOp {
    pub kind: DeltaOpKind,
    pub store_id: u16,
    pub tenant_id: TenantId,
    pub page_no: u64,
    pub slot: u16,
    pub op_lsn: Lsn,
    pub payload: Bytes,
}

impl DeltaOp {
    /// Fixed prefix: kind(1) + store(2) + reserved(1) + tenant(8) +
    /// page(8) + slot(2) + op_lsn(8) + payload_len(4).
    pub const FIXED_PREFIX_LEN: usize = 34;

    #[allow(clippy::too_many_arguments)] // mirrors the fixed wire prefix fields.
    pub fn new(
        kind: DeltaOpKind,
        store_id: u16,
        tenant_id: TenantId,
        page_no: u64,
        slot: u16,
        op_lsn: Lsn,
        payload: impl Into<Bytes>,
    ) -> Result<Self> {
        Self::new_for_format(
            BUNDLE_FORMAT_V9,
            kind,
            store_id,
            tenant_id,
            page_no,
            slot,
            op_lsn,
            payload,
        )
    }

    /// Construct and validate one op for a declared bundle format.
    #[allow(clippy::too_many_arguments)] // mirrors the fixed wire prefix fields.
    pub fn new_for_format(
        format_version: u16,
        kind: DeltaOpKind,
        store_id: u16,
        tenant_id: TenantId,
        page_no: u64,
        slot: u16,
        op_lsn: Lsn,
        payload: impl Into<Bytes>,
    ) -> Result<Self> {
        let op = Self {
            kind,
            store_id,
            tenant_id,
            page_no,
            slot,
            op_lsn,
            payload: payload.into(),
        };
        op.validate_for_format(format_version)?;
        Ok(op)
    }

    #[must_use]
    pub fn encoded_len(&self) -> usize {
        Self::FIXED_PREFIX_LEN + self.payload.len()
    }

    /// Validate the M3 algebra and scope fence.
    pub fn validate_m3(&self) -> Result<()> {
        self.validate_for_format(BUNDLE_FORMAT_V9)
    }

    /// Validate the delta algebra and the declared-version scope fence.
    pub fn validate_for_format(&self, format_version: u16) -> Result<()> {
        if !self.kind.is_emitted_for_format(format_version) {
            let milestone = match format_version {
                BUNDLE_FORMAT_V9 => "M3",
                BUNDLE_FORMAT_V10 => "M4",
                _ => "unknown",
            };
            return Err(corruption(
                self.op_lsn,
                format!(
                    "DeltaOp kind {:?} is reserved in declared WAL bundle format v{format_version} ({milestone})",
                    self.kind,
                ),
            ));
        }
        if self.kind == DeltaOpKind::PutRecord {
            let valid_home = match format_version {
                BUNDLE_FORMAT_V9 => {
                    self.store_id == STORE_RECORD && matches!(self.payload.len(), 64 | 96)
                }
                BUNDLE_FORMAT_V10 => {
                    (self.store_id == STORE_RECORD && self.payload.len() == 64)
                        || (self.store_id == STORE_RELS && self.payload.len() == 96)
                }
                _ => false,
            };
            if !valid_home {
                return Err(corruption(
                    self.op_lsn,
                    "PutRecord requires its format-specific node/relationship direct home",
                ));
            }
        }
        if self.kind == DeltaOpKind::TombstoneRecord {
            let valid_home = match format_version {
                BUNDLE_FORMAT_V9 => self.store_id == STORE_RECORD,
                BUNDLE_FORMAT_V10 => matches!(self.store_id, STORE_RECORD | STORE_RELS),
                _ => false,
            };
            if !valid_home || !self.payload.is_empty() {
                return Err(corruption(
                    self.op_lsn,
                    "TombstoneRecord requires its format-specific direct home and an empty payload",
                ));
            }
        }
        self.validate_shape()
    }

    /// Validate the implemented op algebra after the codec has enforced its
    /// declared-version fence.
    pub(crate) fn validate_shape(&self) -> Result<()> {
        if self.op_lsn == Lsn::ZERO {
            return Err(corruption(self.op_lsn, "DeltaOp op_lsn must be non-zero"));
        }
        if self.kind.is_physical() {
            if self.slot >= MAX_SLOTS
                && !matches!(self.kind, DeltaOpKind::PageAlloc | DeltaOpKind::ExtentAlloc)
            {
                return Err(corruption(
                    self.op_lsn,
                    format!("DeltaOp slot {} exceeds max {}", self.slot, MAX_SLOTS - 1),
                ));
            }
        } else if self.page_no != 0 || self.slot != 0 {
            return Err(corruption(
                self.op_lsn,
                "logical DeltaOp must carry page_no=slot=0",
            ));
        }

        match self.kind {
            DeltaOpKind::PutRecord => {
                if !((self.store_id == STORE_RECORD && matches!(self.payload.len(), 64 | 96))
                    || (self.store_id == STORE_RELS && self.payload.len() == 96))
                {
                    return Err(corruption(
                        self.op_lsn,
                        "PutRecord requires its format-specific node/relationship direct home",
                    ));
                }
            }
            DeltaOpKind::TombstoneRecord => {
                if !matches!(self.store_id, STORE_RECORD | STORE_RELS) || !self.payload.is_empty() {
                    return Err(corruption(
                        self.op_lsn,
                        "TombstoneRecord requires its format-specific direct home and an empty payload",
                    ));
                }
            }
            DeltaOpKind::PutPropBlock => {
                if self.store_id != STORE_PROPS
                    || self.payload.is_empty()
                    || self.payload.len() > MAX_PROP_BLOCK_PAYLOAD
                {
                    return Err(corruption(
                        self.op_lsn,
                        format!(
                            "PutPropBlock requires props.store and a 1..={MAX_PROP_BLOCK_PAYLOAD} B payload; blob.overflow store_id {STORE_BLOB_OVERFLOW} is reserved at M3"
                        ),
                    ));
                }
            }
            DeltaOpKind::AllocAdvance => {
                if self.payload.len() != 9 {
                    return Err(corruption(
                        self.op_lsn,
                        "AllocAdvance payload must be kind(1) + high_water(8)",
                    ));
                }
            }
            DeltaOpKind::InternBind | DeltaOpKind::AclGrant => {
                crate::owner_row::validate_owner_delta(
                    self.kind,
                    self.store_id,
                    self.page_no,
                    self.slot,
                    &self.payload,
                    self.op_lsn,
                )?;
            }
            DeltaOpKind::PageAlloc => {
                if !matches!(
                    self.store_id,
                    STORE_PROPS
                        | STORE_RECORD
                        | STORE_RELS
                        | STORE_TEL
                        | STORE_NODE_BINDINGS
                        | STORE_REL_BINDINGS
                        | STORE_INTERN
                        | STORE_GRANTS
                ) || self.slot != 0
                    || self.payload.len() != 9
                {
                    return Err(corruption(
                        self.op_lsn,
                        "PageAlloc requires an active delta store, slot=0, and type(1)+generation(8)",
                    ));
                }
                let page_type = PageType::from_byte(self.payload[0]).map_err(|error| {
                    corruption(
                        self.op_lsn,
                        format!("PageAlloc carries invalid page type: {error}"),
                    )
                })?;
                let type_matches_store = match self.store_id {
                    STORE_PROPS => page_type == PageType::PropSlotted,
                    STORE_RECORD => matches!(page_type, PageType::Node | PageType::Rel),
                    STORE_RELS => page_type == PageType::Rel,
                    STORE_TEL => page_type == PageType::Tel,
                    STORE_NODE_BINDINGS | STORE_REL_BINDINGS | STORE_INTERN | STORE_GRANTS => {
                        page_type == PageType::PropSlotted
                    }
                    _ => false,
                };
                if !type_matches_store {
                    return Err(corruption(
                        self.op_lsn,
                        format!(
                            "PageAlloc page type {page_type:?} does not belong to store_id {}",
                            self.store_id
                        ),
                    ));
                }
            }
            DeltaOpKind::ExtentAlloc => {
                if !matches!(
                    self.store_id,
                    STORE_PROPS
                        | STORE_RECORD
                        | STORE_RELS
                        | STORE_TEL
                        | STORE_BLOB_OVERFLOW
                        | STORE_NODE_BINDINGS
                        | STORE_REL_BINDINGS
                        | STORE_INTERN
                        | STORE_GRANTS
                ) || self.slot != 0
                {
                    return Err(corruption(
                        self.op_lsn,
                        "ExtentAlloc requires an extent-backed store and slot=0",
                    ));
                }
                let allocation = ExtentAllocation::decode(&self.payload, self.op_lsn)?;
                if allocation.directory_page_no()? != self.page_no {
                    return Err(corruption(
                        self.op_lsn,
                        "ExtentAlloc target is not its computed directory page",
                    ));
                }
            }
            _ => {
                return Err(corruption(
                    self.op_lsn,
                    format!(
                        "DeltaOp kind {:?} has no implemented delta algebra",
                        self.kind
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Append this op's exact v9 wire bytes.
    pub fn encode_into(&self, out: &mut Vec<u8>) -> Result<()> {
        self.encode_into_for_format(out, BUNDLE_FORMAT_V9)
    }

    /// Append this op's wire bytes under the declared bundle contract.
    pub fn encode_into_for_format(&self, out: &mut Vec<u8>, format_version: u16) -> Result<()> {
        self.validate_for_format(format_version)?;
        let payload_len = u32::try_from(self.payload.len())
            .map_err(|_| corruption(self.op_lsn, "DeltaOp payload length does not fit in u32"))?;
        out.reserve(self.encoded_len());
        out.push(self.kind.as_byte());
        out.extend_from_slice(&self.store_id.to_le_bytes());
        out.push(0);
        out.extend_from_slice(&self.tenant_id.raw().to_le_bytes());
        out.extend_from_slice(&self.page_no.to_le_bytes());
        out.extend_from_slice(&self.slot.to_le_bytes());
        out.extend_from_slice(&self.op_lsn.raw().to_le_bytes());
        out.extend_from_slice(&payload_len.to_le_bytes());
        out.extend_from_slice(&self.payload);
        Ok(())
    }

    /// Decode one op from the start of `bytes`, returning bytes consumed.
    pub fn decode_prefix(bytes: &[u8]) -> Result<(Self, usize)> {
        Self::decode_prefix_for_format(bytes, BUNDLE_FORMAT_V9)
    }

    /// Decode one op using the bundle's declared format-version fence.
    pub fn decode_prefix_for_format(bytes: &[u8], format_version: u16) -> Result<(Self, usize)> {
        if bytes.len() < Self::FIXED_PREFIX_LEN {
            return Err(corruption(Lsn::ZERO, "truncated DeltaOp fixed prefix"));
        }
        let kind_byte = bytes[0];
        let store_id = u16::from_le_bytes([bytes[1], bytes[2]]);
        if bytes[3] != 0 {
            return Err(corruption(
                Lsn::ZERO,
                format!("DeltaOp reserved byte must be 0, got {}", bytes[3]),
            ));
        }
        let tenant_id = TenantId::new(read_u64(&bytes[4..12]));
        let page_no = read_u64(&bytes[12..20]);
        let slot = u16::from_le_bytes([bytes[20], bytes[21]]);
        let op_lsn = Lsn::new(read_u64(&bytes[22..30]));
        let payload_len = u32::from_le_bytes(bytes[30..34].try_into().unwrap()) as usize;
        let end = Self::FIXED_PREFIX_LEN
            .checked_add(payload_len)
            .ok_or_else(|| corruption(op_lsn, "DeltaOp payload length overflow"))?;
        if end > bytes.len() {
            return Err(corruption(op_lsn, "DeltaOp payload overruns bundle"));
        }
        let kind = DeltaOpKind::from_byte(kind_byte, op_lsn)?;
        let op = Self {
            kind,
            store_id,
            tenant_id,
            page_no,
            slot,
            op_lsn,
            payload: Bytes::copy_from_slice(&bytes[Self::FIXED_PREFIX_LEN..end]),
        };
        op.validate_for_format(format_version)?;
        Ok((op, end))
    }
}

fn read_u64(bytes: &[u8]) -> u64 {
    u64::from_le_bytes(bytes.try_into().expect("caller passed 8 bytes"))
}

fn corruption(lsn: Lsn, reason: impl Into<String>) -> ArcGraphError {
    ArcGraphError::WalCorruption {
        lsn,
        reason: reason.into(),
    }
}
