//! Catalog root page codec (M10 stage-1, ADR-207).
//!
//! Encodes the tenant registry into the dedicated catalog root page
//! ([`super::CATALOG_PAGE_ID`] = `PageId::ZERO` — the well-known anchor
//! every `PageAllocator` counter starts above) and decodes it back,
//! fail-closed, for the attach-time read-back protocol in
//! [`super::SystemCatalog::attach_page_store`].
//!
//! # Layout (all integers little-endian, matching the WAL record convention)
//!
//! ```text
//! offset  field         size  notes
//! 0       magic         8     b"ARCGCAT1"
//! 8       version       u16   = 1
//! 10      reserved      u16   = 0 (flags; MUST be zero at v1 — fail-closed)
//! 12      payload_len   u32   byte length of the payload region
//! 16      crc32c        u32   crc32c over bytes [20 .. 20+payload_len)
//! 20      payload       …     tenant_count u32, then tenant_count records:
//!                               tenant_id   u64
//!                               created_lsn u64
//!                               tier_tag    u8   (0 = Strict, 1 = Periodic)
//!                               rpo_ms      u64  (0 when Strict; width matches
//!                                                 the ADR-034 MVCC tier encoding)
//!                               name_len    u16  (≤ 256, UTF-8)
//!                               name        name_len bytes
//! ```
//!
//! # Safety posture
//!
//! The decoder runs over UNTRUSTED bytes (restored ADR-204 backups, torn
//! writes, pre-M10 zeroed pages, foreign files): every read is
//! bounds-checked, the CRC is verified before any record parse, claimed
//! lengths never drive an allocation larger than the page, and malformed
//! input returns a typed [`CatalogPageError`] — never a panic. Errors are
//! codec-local per `docs/codec-error-translation.md`; the catalog boundary
//! translates. A fuzz target (`fuzz/fuzz_targets/catalog_page_fuzz.rs`)
//! drives the decoder per `docs/testing-strategy.md`.
//!
//! # Budget (boot / operator-mutation path; NOT a hot path)
//!
//! Encode and decode are one ≤ 8 KiB linear pass + one crc32c over
//! ≤ 8 KiB ≈ 1–2 µs (ADR-207 §Back-of-envelope). The commit-time tier
//! lookup never touches this codec — it stays the in-memory ≤ 50 ns Vec
//! scan (`catalog.rs` §Back-of-envelope).

use arcgraph_core::{DurabilityTier, Lsn, PAGE_SIZE, TenantId};
use thiserror::Error;

use super::TenantRecord;
use crate::io::PageBuf;

/// Magic bytes identifying a catalog root page.
///
/// `pub(crate)` per the #1058 R1 NIT-3 disposition: the format
/// constants are codec internals — the published contract is
/// [`encode_catalog_page`] / [`decode_catalog_page`] (all the fuzz
/// target consumes). Re-publish deliberately if a restore/inspection
/// tool ever needs raw-format access.
pub(crate) const CATALOG_PAGE_MAGIC: &[u8; 8] = b"ARCGCAT1";

/// Catalog page format version this binary writes and reads.
/// `pub(crate)` — see [`CATALOG_PAGE_MAGIC`].
pub(crate) const CATALOG_PAGE_VERSION: u16 = 1;

/// Header length: magic(8) + version(2) + reserved(2) + payload_len(4) +
/// crc32c(4).
const HEADER_LEN: usize = 20;

/// Maximum tenant-name byte length the on-page format accepts.
pub const MAX_NAME_LEN: usize = 256;

/// Fixed per-record byte length excluding the name:
/// tenant_id(8) + created_lsn(8) + tier_tag(1) + rpo_ms(8) + name_len(2).
const RECORD_FIXED_LEN: usize = 27;

/// Faults surfaced by the catalog page codec.
///
/// Codec-local per `docs/codec-error-translation.md`; translated at the
/// [`super::SystemCatalog::attach_page_store`] boundary.
#[derive(Debug, Error, PartialEq, Eq)]
#[non_exhaustive]
pub enum CatalogPageError {
    /// First 8 bytes are not `CATALOG_PAGE_MAGIC`. This is the NORMAL
    /// outcome for a zeroed / never-materialized page 0 (pre-M10 dirs).
    #[error("catalog page magic mismatch (not a catalog root page)")]
    BadMagic,
    /// Version field is not `CATALOG_PAGE_VERSION`.
    #[error("unsupported catalog page version {0} (this binary reads v{CATALOG_PAGE_VERSION})")]
    UnsupportedVersion(u16),
    /// Reserved/flags bytes are non-zero — written by a future writer
    /// whose semantics this binary does not understand. Fail-closed.
    #[error("catalog page reserved bytes non-zero ({0:#06x}); refusing to decode")]
    NonZeroReserved(u16),
    /// Declared payload length exceeds the page's payload capacity.
    #[error("catalog page payload length {len} exceeds capacity {cap}")]
    PayloadOverrun { len: usize, cap: usize },
    /// CRC over the payload region does not match the stored CRC.
    #[error("catalog page crc32c mismatch: stored {stored:#010x}, computed {computed:#010x}")]
    ChecksumMismatch { stored: u32, computed: u32 },
    /// Record region ended mid-field.
    #[error("catalog page truncated at payload byte {at} (need {need} more)")]
    Truncated { at: usize, need: usize },
    /// Unknown durability-tier tag.
    #[error("catalog page record carries unknown durability tier tag {0}")]
    UnknownTierTag(u8),
    /// A Strict-tier record must carry `rpo_ms == 0`.
    #[error("catalog page Strict-tier record carries non-zero rpo_ms {0}")]
    StrictWithRpo(u64),
    /// Tenant name exceeds [`MAX_NAME_LEN`].
    #[error("catalog page tenant-name length {len} exceeds cap {cap}")]
    NameTooLong { len: usize, cap: usize },
    /// Tenant name bytes are not valid UTF-8.
    #[error("catalog page tenant name is not valid UTF-8")]
    NameNotUtf8,
    /// Payload bytes remain after the declared record count was consumed.
    #[error("catalog page has {trailing} trailing payload byte(s) after {count} records")]
    TrailingBytes { trailing: usize, count: u32 },
    /// The registry does not fit a single page. Multi-page chaining is
    /// M10 stage-2 (ADR-207 §Forward-deferred).
    #[error(
        "encoded tenant registry needs {needed} bytes; single-page cap is {cap} \
         (multi-page catalog is M10 stage-2)"
    )]
    RegistryTooLarge { needed: usize, cap: usize },
}

/// Encode `records` into a fresh catalog root page.
///
/// # Errors
///
/// [`CatalogPageError::NameTooLong`] when a tenant name exceeds
/// [`MAX_NAME_LEN`]; [`CatalogPageError::RegistryTooLarge`] when the
/// encoded registry exceeds one page (M10 stage-2 territory).
pub fn encode_catalog_page(records: &[TenantRecord]) -> Result<Box<PageBuf>, CatalogPageError> {
    let mut payload: Vec<u8> =
        Vec::with_capacity(4 + records.len() * (RECORD_FIXED_LEN + MAX_NAME_LEN));
    let count = u32::try_from(records.len()).map_err(|_| CatalogPageError::RegistryTooLarge {
        needed: usize::MAX,
        cap: PAGE_SIZE,
    })?;
    payload.extend_from_slice(&count.to_le_bytes());
    for r in records {
        let name = r.name.as_bytes();
        if name.len() > MAX_NAME_LEN {
            return Err(CatalogPageError::NameTooLong {
                len: name.len(),
                cap: MAX_NAME_LEN,
            });
        }
        payload.extend_from_slice(&r.tenant_id.raw().to_le_bytes());
        payload.extend_from_slice(&r.created_lsn.raw().to_le_bytes());
        match r.tier {
            DurabilityTier::Strict => {
                payload.push(0);
                payload.extend_from_slice(&0u64.to_le_bytes());
            }
            DurabilityTier::Periodic { rpo_ms } => {
                payload.push(1);
                payload.extend_from_slice(&rpo_ms.to_le_bytes());
            }
        }
        // Cast is lossless: name.len() ≤ MAX_NAME_LEN = 256 ≤ u16::MAX.
        payload.extend_from_slice(&(name.len() as u16).to_le_bytes());
        payload.extend_from_slice(name);
    }
    let needed = HEADER_LEN + payload.len();
    if needed > PAGE_SIZE {
        return Err(CatalogPageError::RegistryTooLarge {
            needed,
            cap: PAGE_SIZE,
        });
    }
    let mut page: Box<PageBuf> = Box::new([0u8; PAGE_SIZE]);
    page[0..8].copy_from_slice(CATALOG_PAGE_MAGIC);
    page[8..10].copy_from_slice(&CATALOG_PAGE_VERSION.to_le_bytes());
    // bytes 10..12 stay zero (reserved).
    // Cast is lossless: payload.len() ≤ PAGE_SIZE - HEADER_LEN < u32::MAX.
    page[12..16].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    page[16..20].copy_from_slice(&crc32c::crc32c(&payload).to_le_bytes());
    page[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(&payload);
    Ok(page)
}

/// Decode a catalog root page back into tenant records. Fail-closed over
/// untrusted bytes — see the module-level §Safety posture.
///
/// # Errors
///
/// Every malformation maps to a typed [`CatalogPageError`]; a zeroed /
/// never-materialized page returns [`CatalogPageError::BadMagic`].
pub fn decode_catalog_page(page: &PageBuf) -> Result<Vec<TenantRecord>, CatalogPageError> {
    if &page[0..8] != CATALOG_PAGE_MAGIC {
        return Err(CatalogPageError::BadMagic);
    }
    let version = u16::from_le_bytes([page[8], page[9]]);
    if version != CATALOG_PAGE_VERSION {
        return Err(CatalogPageError::UnsupportedVersion(version));
    }
    let reserved = u16::from_le_bytes([page[10], page[11]]);
    if reserved != 0 {
        return Err(CatalogPageError::NonZeroReserved(reserved));
    }
    let payload_len = u32::from_le_bytes([page[12], page[13], page[14], page[15]]) as usize;
    if payload_len > PAGE_SIZE - HEADER_LEN {
        return Err(CatalogPageError::PayloadOverrun {
            len: payload_len,
            cap: PAGE_SIZE - HEADER_LEN,
        });
    }
    let stored = u32::from_le_bytes([page[16], page[17], page[18], page[19]]);
    let payload = &page[HEADER_LEN..HEADER_LEN + payload_len];
    let computed = crc32c::crc32c(payload);
    if stored != computed {
        return Err(CatalogPageError::ChecksumMismatch { stored, computed });
    }

    let mut cur = 0usize;
    let count = u32::from_le_bytes(read_array::<4>(payload, &mut cur)?);
    // Allocation guard: a (CRC-valid but) hostile count cannot drive the
    // Vec past what the payload could physically hold.
    let max_physical = payload_len / RECORD_FIXED_LEN + 1;
    let mut out: Vec<TenantRecord> = Vec::with_capacity((count as usize).min(max_physical));
    for _ in 0..count {
        let tenant_raw = u64::from_le_bytes(read_array::<8>(payload, &mut cur)?);
        let lsn_raw = u64::from_le_bytes(read_array::<8>(payload, &mut cur)?);
        let tag = read_array::<1>(payload, &mut cur)?[0];
        let rpo_ms = u64::from_le_bytes(read_array::<8>(payload, &mut cur)?);
        let tier = match tag {
            0 => {
                if rpo_ms != 0 {
                    return Err(CatalogPageError::StrictWithRpo(rpo_ms));
                }
                DurabilityTier::Strict
            }
            1 => DurabilityTier::Periodic { rpo_ms },
            t => return Err(CatalogPageError::UnknownTierTag(t)),
        };
        let name_len = u16::from_le_bytes(read_array::<2>(payload, &mut cur)?) as usize;
        if name_len > MAX_NAME_LEN {
            return Err(CatalogPageError::NameTooLong {
                len: name_len,
                cap: MAX_NAME_LEN,
            });
        }
        let name_bytes = read_slice(payload, &mut cur, name_len)?;
        let name = std::str::from_utf8(name_bytes)
            .map_err(|_| CatalogPageError::NameNotUtf8)?
            .to_string();
        out.push(TenantRecord {
            tenant_id: TenantId::new(tenant_raw),
            name,
            created_lsn: Lsn::new(lsn_raw),
            tier,
        });
    }
    if cur != payload_len {
        return Err(CatalogPageError::TrailingBytes {
            trailing: payload_len - cur,
            count,
        });
    }
    Ok(out)
}

/// Bounds-checked fixed-width read; advances `cur` on success.
fn read_array<const N: usize>(
    payload: &[u8],
    cur: &mut usize,
) -> Result<[u8; N], CatalogPageError> {
    let end = cur
        .checked_add(N)
        .ok_or(CatalogPageError::Truncated { at: *cur, need: N })?;
    if end > payload.len() {
        return Err(CatalogPageError::Truncated {
            at: *cur,
            need: end - payload.len(),
        });
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&payload[*cur..end]);
    *cur = end;
    Ok(arr)
}

/// Bounds-checked variable-width read; advances `cur` on success.
fn read_slice<'a>(
    payload: &'a [u8],
    cur: &mut usize,
    len: usize,
) -> Result<&'a [u8], CatalogPageError> {
    let end = cur.checked_add(len).ok_or(CatalogPageError::Truncated {
        at: *cur,
        need: len,
    })?;
    if end > payload.len() {
        return Err(CatalogPageError::Truncated {
            at: *cur,
            need: end - payload.len(),
        });
    }
    let out = &payload[*cur..end];
    *cur = end;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;

    fn rec(id: u64, name: &str, lsn: u64, tier: DurabilityTier) -> TenantRecord {
        TenantRecord {
            tenant_id: TenantId::new(id),
            name: name.to_string(),
            created_lsn: Lsn::new(lsn),
            tier,
        }
    }

    #[test]
    fn round_trip_empty_registry() {
        let page = encode_catalog_page(&[]).expect("encode");
        let back = decode_catalog_page(&page).expect("decode");
        assert!(back.is_empty());
    }

    #[test]
    fn round_trip_default_tenant_shape() {
        // The exact registry shape `build_durable` materializes at v1.
        let records = vec![rec(1, "default", 1, DurabilityTier::Strict)];
        let page = encode_catalog_page(&records).expect("encode");
        let back = decode_catalog_page(&page).expect("decode");
        assert_eq!(back, records);
    }

    #[test]
    fn round_trip_mixed_tiers_and_names() {
        let records = vec![
            rec(1, "default", 7, DurabilityTier::Strict),
            rec(
                42,
                "log-ingest",
                99,
                DurabilityTier::Periodic { rpo_ms: 500 },
            ),
            rec(7, "", 3, DurabilityTier::Strict), // empty name is legal
        ];
        let page = encode_catalog_page(&records).expect("encode");
        let back = decode_catalog_page(&page).expect("decode");
        assert_eq!(back, records);
    }

    #[test]
    fn zeroed_page_is_bad_magic() {
        // THE pre-M10 / fresh-page case: all-zero bytes must decode to
        // BadMagic (treated as "no prior page" by attach), not panic.
        let page: PageBuf = [0u8; PAGE_SIZE];
        assert_eq!(decode_catalog_page(&page), Err(CatalogPageError::BadMagic));
    }

    #[test]
    fn version_bump_is_rejected() {
        let mut page = *encode_catalog_page(&[]).expect("encode");
        page[8..10].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(
            decode_catalog_page(&page),
            Err(CatalogPageError::UnsupportedVersion(2))
        );
    }

    #[test]
    fn nonzero_reserved_is_rejected() {
        let mut page = *encode_catalog_page(&[]).expect("encode");
        page[10] = 0x01;
        assert_eq!(
            decode_catalog_page(&page),
            Err(CatalogPageError::NonZeroReserved(1))
        );
    }

    #[test]
    fn payload_overrun_is_rejected() {
        let mut page = *encode_catalog_page(&[]).expect("encode");
        page[12..16].copy_from_slice(&(PAGE_SIZE as u32).to_le_bytes());
        assert!(matches!(
            decode_catalog_page(&page),
            Err(CatalogPageError::PayloadOverrun { .. })
        ));
    }

    #[test]
    fn flipped_payload_byte_fails_crc() {
        let records = vec![rec(1, "default", 1, DurabilityTier::Strict)];
        let mut page = *encode_catalog_page(&records).expect("encode");
        page[HEADER_LEN + 4] ^= 0xFF; // first record byte
        assert!(matches!(
            decode_catalog_page(&page),
            Err(CatalogPageError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn truncated_record_region_is_rejected_not_panicking() {
        // Hand-build a CRC-valid payload that declares 1 record but
        // carries only 4 bytes of it.
        let mut payload = Vec::new();
        payload.extend_from_slice(&1u32.to_le_bytes());
        payload.extend_from_slice(&[0xAA; 4]);
        let mut page = [0u8; PAGE_SIZE];
        page[0..8].copy_from_slice(CATALOG_PAGE_MAGIC);
        page[8..10].copy_from_slice(&CATALOG_PAGE_VERSION.to_le_bytes());
        page[12..16].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        page[16..20].copy_from_slice(&crc32c::crc32c(&payload).to_le_bytes());
        page[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(&payload);
        assert!(matches!(
            decode_catalog_page(&page),
            Err(CatalogPageError::Truncated { .. })
        ));
    }

    #[test]
    fn unknown_tier_tag_is_rejected() {
        let records = vec![rec(1, "x", 1, DurabilityTier::Strict)];
        let mut page = *encode_catalog_page(&records).expect("encode");
        // tier_tag sits at payload offset 4 (count) + 16 (ids) = 20.
        let off = HEADER_LEN + 4 + 16;
        page[off] = 9;
        // Re-seal the CRC so the tag check (not the CRC) is what fires.
        let payload_len = u32::from_le_bytes([page[12], page[13], page[14], page[15]]) as usize;
        let crc = crc32c::crc32c(&page[HEADER_LEN..HEADER_LEN + payload_len]);
        page[16..20].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(
            decode_catalog_page(&page),
            Err(CatalogPageError::UnknownTierTag(9))
        );
    }

    #[test]
    fn strict_with_nonzero_rpo_is_rejected() {
        let records = vec![rec(1, "x", 1, DurabilityTier::Strict)];
        let mut page = *encode_catalog_page(&records).expect("encode");
        // rpo_ms sits right after the tag.
        let off = HEADER_LEN + 4 + 16 + 1;
        page[off] = 5;
        let payload_len = u32::from_le_bytes([page[12], page[13], page[14], page[15]]) as usize;
        let crc = crc32c::crc32c(&page[HEADER_LEN..HEADER_LEN + payload_len]);
        page[16..20].copy_from_slice(&crc.to_le_bytes());
        assert_eq!(
            decode_catalog_page(&page),
            Err(CatalogPageError::StrictWithRpo(5))
        );
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        // CRC-valid payload: 0 records + 3 stray bytes.
        let mut payload = Vec::new();
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&[1, 2, 3]);
        let mut page = [0u8; PAGE_SIZE];
        page[0..8].copy_from_slice(CATALOG_PAGE_MAGIC);
        page[8..10].copy_from_slice(&CATALOG_PAGE_VERSION.to_le_bytes());
        page[12..16].copy_from_slice(&(payload.len() as u32).to_le_bytes());
        page[16..20].copy_from_slice(&crc32c::crc32c(&payload).to_le_bytes());
        page[HEADER_LEN..HEADER_LEN + payload.len()].copy_from_slice(&payload);
        assert_eq!(
            decode_catalog_page(&page),
            Err(CatalogPageError::TrailingBytes {
                trailing: 3,
                count: 0
            })
        );
    }

    #[test]
    fn name_too_long_rejected_on_encode() {
        let records = vec![rec(
            1,
            &"n".repeat(MAX_NAME_LEN + 1),
            1,
            DurabilityTier::Strict,
        )];
        assert!(matches!(
            encode_catalog_page(&records),
            Err(CatalogPageError::NameTooLong { .. })
        ));
    }

    #[test]
    fn registry_too_large_rejected_on_encode() {
        // Worst-case names overflow one page well before 100 records.
        let records: Vec<TenantRecord> = (0..100)
            .map(|i| rec(i, &"n".repeat(MAX_NAME_LEN), i, DurabilityTier::Strict))
            .collect();
        assert!(matches!(
            encode_catalog_page(&records),
            Err(CatalogPageError::RegistryTooLarge { .. })
        ));
    }

    proptest! {
        /// Encode/decode round-trip over arbitrary registries — the
        /// `docs/testing-strategy.md` serializer property bar.
        #[test]
        fn property_round_trip(
            entries in prop::collection::vec(
                (
                    any::<u64>(),
                    "[a-z0-9_-]{0,32}",
                    any::<u64>(),
                    prop_oneof![
                        Just(None),
                        any::<u64>().prop_map(Some),
                    ],
                ),
                0..32,
            )
        ) {
            let records: Vec<TenantRecord> = entries
                .into_iter()
                .map(|(id, name, lsn, rpo)| TenantRecord {
                    tenant_id: TenantId::new(id),
                    name,
                    created_lsn: Lsn::new(lsn),
                    tier: match rpo {
                        None => DurabilityTier::Strict,
                        Some(rpo_ms) => DurabilityTier::Periodic { rpo_ms },
                    },
                })
                .collect();
            let page = encode_catalog_page(&records).expect("encode fits");
            let back = decode_catalog_page(&page).expect("decode");
            prop_assert_eq!(back, records);
        }

        /// Arbitrary header-region corruption of a valid page never
        /// panics — it decodes Ok (byte happened to be a no-op) or a
        /// typed error.
        #[test]
        fn property_corruption_never_panics(idx in 0usize..PAGE_SIZE, byte in any::<u8>()) {
            let records = vec![
                TenantRecord {
                    tenant_id: TenantId::new(1),
                    name: "default".to_string(),
                    created_lsn: Lsn::new(1),
                    tier: DurabilityTier::Strict,
                },
            ];
            let mut page = *encode_catalog_page(&records).expect("encode");
            page[idx] = byte;
            let _ = decode_catalog_page(&page); // must not panic
        }
    }
}
