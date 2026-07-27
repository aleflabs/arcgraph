#![no_main]
//! W28-S4: page-layout parser fuzz target (testing-strategy §2.4 minimum
//! target `page`; gap analysis PR #510 §1c; ADR-165 M6).
//!
//! # What this fuzzes
//!
//! The page-layout decode surface of the storage engine, in four
//! dimensions that mirror the `node_record_deser_fuzz.rs` /
//! `wal_deserializer_fuzz.rs` shape:
//!
//! 1. [`arcgraph_core::PageHeader::from_bytes`] — the fixed 40-byte
//!    page-header decoder at `crates/arcgraph-core/src/record.rs:194`.
//!    It validates magic (`ARCG`), header version, and `PageType`.
//! 2. [`arcgraph_storage::records::SlottedPageRef::open`] — the 8 KiB
//!    slotted-page *frame* validator at
//!    `crates/arcgraph-storage/src/records.rs:426`. It re-validates the
//!    header and checks the body CRC32C before admitting the frame.
//! 3. The full round-trip through [`arcgraph_storage::records::SlottedPage`]:
//!    `init` a typed page, insert records decoded from fuzz windows,
//!    re-`open` the resulting frame, and read every slot back.
//! 4. The **forged-frame** path (#592): a well-formed page whose header
//!    `slot_count` is hand-set to an arbitrary `u16`. The `slot_count`
//!    field lives at header bytes 36..38, *outside* the CRC body range
//!    (bytes 40..), so the forged frame still passes the checksum gate —
//!    exactly the adversarial shape that an attacker who can write a
//!    CRC-consistent page (or on-disk corruption that preserves the CRC)
//!    can produce. This drives `open` + `read_node`/`iter_*` against it.
//!
//! # Assertions (testing-strategy §2.4 oracle)
//!
//! - **(a) No panic / no UB** on ANY input — the libfuzzer contract.
//! - **(b) Valid frame ⇒ decode→encode→decode is idempotent.** A header
//!   that decodes re-encodes to the *exact* bytes it came from (the
//!   encoding is canonical — there are no lossy fields), and a frame
//!   built by the public constructor must be accepted by the validator
//!   and read back record-for-record.
//! - **(c) Malformed frames return a structured error**
//!   ([`Err(PageError)`] / [`Err(ArcGraphError)`]) — never a panic and
//!   never a silent wrong-accept (`open` re-derives the header rather
//!   than trusting the bytes).
//! - **(d) Forged `slot_count` never panics the reader (#592).** A frame
//!   whose `slot_count` exceeds the page's slot capacity is rejected by
//!   `open` (it is not a panic and not a silent accept); and for any
//!   frame `open` admits, `read_node`/`iter_*` over every claimed slot
//!   returns a structured result rather than an out-of-bounds slice
//!   panic.
//!
//! # Scope boundary
//!
//! This target fuzzes the page-*layout* parser: the header decoder, the
//! frame validator (`open`), the slot directory exercised through
//! legitimately-constructed pages, and — since #592 — the reader's
//! response to *forged* frames whose header `slot_count` is hand-set past
//! the page bound (dimension 4). The earlier revision of this target
//! deliberately excluded that forged-frame path; that exclusion was the
//! exact gap #592 exploited (an unbounded `slot_count` drove
//! `slot_entry` to index past the 8 KiB buffer and panic). The fix
//! validates `slot_count` at `open` time, and this target now pins it.

use libfuzzer_sys::fuzz_target;

use arcgraph_core::ids::{PageId, TenantId};
use arcgraph_core::{NodeRecord, PAGE_SIZE, PageHeader, PageType};
use arcgraph_storage::records::{SlotId, SlottedPage, SlottedPageRef};

/// `PageHeader::SIZE`. Re-stated as a literal so the harness does not
/// depend on the constant being `pub` (it is, but keep the contract local).
const HEADER_SIZE: usize = 40;

/// `arcgraph_core::PAGE_MAGIC` (`0x4743_5241`) in little-endian bytes —
/// ASCII `ARCG`. Used to force the constructed frame into the accepted
/// magic domain for dimension 3.
const PAGE_MAGIC_LE: [u8; 4] = [0x41, 0x52, 0x43, 0x47];

fuzz_target!(|data: &[u8]| {
    // ── Dimension 1: 40-byte PageHeader decode + canonical round-trip ──
    // Mirrors node_record_deser_fuzz: decode a fixed-size header window,
    // and on success prove the encoder is the decoder's exact inverse.
    if data.len() >= HEADER_SIZE {
        let mut hdr_buf = [0u8; HEADER_SIZE];
        hdr_buf.copy_from_slice(&data[..HEADER_SIZE]);
        match PageHeader::from_bytes(&hdr_buf) {
            Ok(header) => {
                // decode → encode → decode idempotence.
                let re = header.to_bytes();
                let header2 = PageHeader::from_bytes(&re)
                    .expect("re-decode of canonical-encoded PageHeader must succeed");
                assert_eq!(
                    header, header2,
                    "PageHeader roundtrip diverged: {header:?} != {header2:?}"
                );
                // The header has no lossy fields: a header that decoded
                // from `hdr_buf` MUST re-encode to those exact 40 bytes.
                // Any divergence is an encoder/decoder asymmetry.
                assert_eq!(
                    &hdr_buf[..],
                    &re[..],
                    "PageHeader encode is non-canonical for a decoded header"
                );
            }
            // Bad magic / unsupported version / unknown page-type byte —
            // the structured-reject path. Expected for most inputs.
            Err(_) => {}
        }
    }

    // ── Dimension 2: 8 KiB frame validator on arbitrary bytes ──
    // `open` validates length + header + body CRC32C. Arbitrary input is
    // almost always rejected (the CRC gate); both outcomes are valid. The
    // contract is no-panic and no silent wrong-accept.
    {
        let mut page = [0u8; PAGE_SIZE];
        let n = data.len().min(PAGE_SIZE);
        page[..n].copy_from_slice(&data[..n]);
        if let Ok(view) = SlottedPageRef::open(&page) {
            // Accept path: `open` having returned Ok means it re-derived a
            // valid header. Exercise it; the header must re-encode cleanly.
            let _ = view.header().to_bytes();
        }
    }

    // ── Dimension 3: build a valid frame, then prove the validator
    //    accepts it and reads back record-for-record (oracle (b)) ──
    //
    // `SlottedPage::init` writes a valid header and the matching body
    // CRC, so the resulting frame is well-formed by construction. This
    // is the legitimate producer path; driving the reader against it is
    // safe (slot_count stays in bounds because `insert_*` maintains it).
    if data.len() >= HEADER_SIZE {
        // Page type from a fuzz byte: init only admits Node / Rel.
        let pid = u64::from_le_bytes([
            data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7],
        ]);
        let page_type = if data[5] & 1 == 0 {
            PageType::Node
        } else {
            PageType::Rel
        };
        let header = PageHeader::new(PageId::new(pid), page_type, TenantId::DEFAULT);

        let mut page = [0u8; PAGE_SIZE];
        // Collect the records we successfully insert so we can verify the
        // read-back round-trip after re-opening the frame.
        let mut inserted: Vec<NodeRecord> = Vec::new();
        {
            let Ok(mut sp) = SlottedPage::init(&mut page, header) else {
                return;
            };
            // Only Node pages get a record round-trip in this dimension
            // (NodeRecord is the 64-byte fixed shape we decode from
            // windows); Rel pages still exercise init + reopen below.
            if page_type == PageType::Node {
                let mut off = HEADER_SIZE;
                while off + NodeRecord::SIZE <= data.len() {
                    let mut rbuf = [0u8; NodeRecord::SIZE];
                    rbuf.copy_from_slice(&data[off..off + NodeRecord::SIZE]);
                    off += NodeRecord::SIZE;
                    // Decode a candidate record; skip windows that aren't
                    // a valid NodeRecord (unsupported version byte, etc.).
                    let Ok(rec) = NodeRecord::from_bytes(&rbuf) else {
                        continue;
                    };
                    match sp.insert_node(&rec) {
                        Ok(_) => inserted.push(rec),
                        // Page full — stop inserting, keep what we have.
                        Err(_) => break,
                    }
                }
            }
            // `sp` (the &mut borrow) drops here, releasing `page`.
        }

        // The freshly-built frame MUST be accepted by the validator — a
        // false-reject here is a frame-decoder bug.
        let view = SlottedPageRef::open(&page)
            .expect("a page built by SlottedPage::init must be accepted by SlottedPageRef::open");

        // The constructed frame's magic must match the canonical magic
        // bytes (sanity pin on the encoder's magic write).
        assert_eq!(
            &page[0..4],
            &PAGE_MAGIC_LE[..],
            "init wrote a non-canonical page magic"
        );

        if page_type == PageType::Node {
            // Slot directory round-trip: every record we inserted reads
            // back byte-for-byte (decode→encode→decode idempotence at the
            // slotted-page layer).
            let mut read_back = 0usize;
            for (i, expected) in inserted.iter().enumerate() {
                let slot = arcgraph_storage::records::SlotId(i as u16);
                match view.read_node(slot) {
                    Ok(Some(got)) => {
                        assert_eq!(
                            &got, expected,
                            "slot {i} read-back diverged from inserted record"
                        );
                        read_back += 1;
                    }
                    other => panic!("slot {i} expected a live record, got {other:?}"),
                }
            }
            // The live-iterator must surface exactly the inserted set.
            let iter_count = view.iter_nodes().count();
            assert_eq!(
                iter_count, read_back,
                "iter_nodes count {iter_count} != read-back count {read_back}"
            );
        }
    }

    // ── Dimension 4 (#592): forged slot_count must not OOB-panic the reader ──
    //
    // Build a well-formed typed page, then forge its header `slot_count`
    // to an arbitrary u16 drawn from the input. Because `slot_count`
    // (header bytes 36..38) lies outside the CRC body range (bytes 40..),
    // the forged frame still passes `open`'s checksum gate. The contract:
    // `open` either rejects the over-capacity frame with a structured
    // error, or — for an admitted frame — every slot read across the
    // claimed `slot_count` returns a structured result, never a panic.
    if data.len() >= HEADER_SIZE + 2 {
        let page_type = if data[5] & 1 == 0 {
            PageType::Node
        } else {
            PageType::Rel
        };
        let header = PageHeader::new(PageId::new(0xABCD), page_type, TenantId::DEFAULT);
        let mut page = [0u8; PAGE_SIZE];
        if SlottedPage::init(&mut page, header).is_ok() {
            // Forge slot_count from two fuzz bytes (covers 0..=u16::MAX,
            // i.e. both in-bounds and far past the slot capacity). The CRC
            // stays valid because the field is outside the body range.
            let forged = u16::from_le_bytes([data[HEADER_SIZE], data[HEADER_SIZE + 1]]);
            page[36..38].copy_from_slice(&forged.to_le_bytes());

            if let Ok(view) = SlottedPageRef::open(&page) {
                // `open` admitted the frame ⇒ slot_count is within the
                // capacity bound; reading every claimed slot must not panic.
                let count = view.slot_count();
                for i in 0..count {
                    let slot = SlotId(i);
                    match page_type {
                        PageType::Node => {
                            let _ = view.read_node(slot);
                        }
                        PageType::Rel => {
                            let _ = view.read_rel(slot);
                        }
                        _ => unreachable!("init only admits Node/Rel pages"),
                    }
                }
                let _ = view.iter_nodes().count();
                let _ = view.iter_rels().count();
            }
            // Err(_) ⇒ the forged over-capacity frame was rejected — fine.
        }
    }
});
