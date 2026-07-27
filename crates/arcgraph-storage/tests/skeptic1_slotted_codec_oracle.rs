//! SKEPTIC-1 scratch (NOT FOR COMMIT) — adversarial independent-oracle
//! verification of the v2 M1 slotted prop-bag codec (`records.rs`).
//!
//! Oracle = a test-maintained model, never the codec's own reader:
//!   - `slots: Vec<Option<Vec<u8>>>` (slot id -> live payload bytes)
//!   - `free: usize` (spec: starts at PAGE_BODY_BYTES; insert of len L
//!     is legal iff free >= L + SLOT_SIZE; on success free -= L+4;
//!     tombstone NEVER reclaims)
//!     Mutations open the buffer per-op via `open_prop_trusted` — the exact
//!     production shape (`append_bag_to_image` opens per append). Sweeps
//!     re-validate through the FULL-CRC `SlottedPageRef::open` on a shared
//!     borrow, which catches any mutation path that forgets recompute_checksum.

use arcgraph_core::ids::{PageId, TenantId};
use arcgraph_core::record::{PAGE_SIZE, PageHeader, PageType};
use arcgraph_storage::records::{
    PAGE_BODY_BYTES, PROP_BAG_MAX_BYTES, PageError, SLOT_AREA_START, SLOT_SIZE, SlotId,
    SlottedPage, SlottedPageRef,
};

const MIN_NEEDED: usize = 1 + SLOT_SIZE; // smallest legal bag + its slot entry

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() as usize) % n.max(1)
    }
}

/// Independent model of one PropSlotted page.
struct Oracle {
    slots: Vec<Option<Vec<u8>>>,
    free: usize,
}

impl Oracle {
    fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: PAGE_BODY_BYTES,
        }
    }
    fn fits(&self, len: usize) -> bool {
        self.free >= len + SLOT_SIZE
    }
    fn record_insert(&mut self, bytes: &[u8]) -> u16 {
        let id = self.slots.len() as u16;
        self.free -= bytes.len() + SLOT_SIZE;
        self.slots.push(Some(bytes.to_vec()));
        id
    }
}

fn fresh_bag_page(buf: &mut [u8; PAGE_SIZE]) {
    let hdr = PageHeader::new(PageId::new(4242), PageType::PropSlotted, TenantId::DEFAULT);
    SlottedPage::init(&mut buf[..], hdr).expect("init PropSlotted");
}

/// Adversarial payload generator: sizes and byte patterns chosen to
/// stress boundaries and mimic structural bytes.
fn gen_payload(rng: &mut Lcg, free: usize) -> Vec<u8> {
    // Size classes: tiny dominates (deep slot counts); page-filling
    // classes fire ~1/16 combined so pages live long enough to stack
    // many slots before the boundary probes hit.
    let size = match rng.below(64) {
        0..=19 => 1,
        20..=31 => 2,
        32..=39 => SLOT_SIZE, // == slot entry size
        40..=43 => 5,
        44..=45 => 8,
        46..=47 => 33,
        48..=49 => 64,  // NodeRecord-ish
        50..=51 => 512, // pool threshold-ish
        52 => 1024,
        53 => 2048,
        54 => PROP_BAG_MAX_BYTES,     // 8148 — max legal
        55 => PROP_BAG_MAX_BYTES - 1, // 8147
        // exact remaining fit (free-4) and one-over (free-3):
        56 if free > SLOT_SIZE => free - SLOT_SIZE,
        57 if free > SLOT_SIZE + 1 => free - SLOT_SIZE + 1, // one too big
        _ => 1 + rng.below(512),
    };
    let size = size.clamp(1, PROP_BAG_MAX_BYTES);
    let pat = rng.below(5);
    let mut v = Vec::with_capacity(size);
    match pat {
        0 => v.resize(size, 0x00), // tombstone-mimic zeros
        1 => v.resize(size, 0xFF), // all ones
        2 => {
            // slot-entry mimic: LE u16 pairs that look like (offset,len)
            while v.len() < size {
                let off = (rng.next() as u16).to_le_bytes();
                v.extend_from_slice(&off);
            }
            v.truncate(size);
        }
        3 => {
            // PAGE_MAGIC-ish prefix then random
            v.extend_from_slice(b"ARCG");
            while v.len() < size {
                v.push(rng.next() as u8);
            }
            v.truncate(size);
        }
        _ => {
            while v.len() < size {
                v.push(rng.next() as u8);
            }
        }
    }
    v
}

/// Sweep every oracle slot through THREE independent views and assert
/// byte equality + header agreement.
fn full_sweep(buf: &[u8; PAGE_SIZE], oracle: &Oracle, ctx: &str) {
    // View 1: full-CRC validated read-only open (trust-boundary path).
    let crc_view = SlottedPageRef::open(&buf[..])
        .unwrap_or_else(|e| panic!("[{ctx}] full-CRC open failed: {e:?}"));
    // View 2: trusted read-only open (hot read path, get_slotted).
    let trusted = SlottedPageRef::open_prop_trusted(&buf[..])
        .unwrap_or_else(|e| panic!("[{ctx}] open_prop_trusted failed: {e:?}"));

    for view in [&crc_view, &trusted] {
        let hdr = view.header();
        assert_eq!(
            hdr.slot_count as usize,
            oracle.slots.len(),
            "[{ctx}] slot_count mismatch"
        );
        assert_eq!(
            hdr.free_space as usize, oracle.free,
            "[{ctx}] free_space mismatch"
        );
        for (i, want) in oracle.slots.iter().enumerate() {
            let got = view.read_bag(SlotId(i as u16));
            match want {
                Some(bytes) => {
                    let got = got
                        .unwrap_or_else(|e| panic!("[{ctx}] read_bag({i}) err: {e:?}"))
                        .unwrap_or_else(|| panic!("[{ctx}] slot {i} unexpectedly tombstoned"));
                    assert_eq!(got, &bytes[..], "[{ctx}] slot {i} byte mismatch");
                }
                None => {
                    assert!(
                        matches!(got, Ok(None)),
                        "[{ctx}] tombstoned slot {i} should read Ok(None), got {got:?}"
                    );
                }
            }
        }
        // One-past-the-end must be SlotOutOfRange.
        let oob = view.read_bag(SlotId(oracle.slots.len() as u16));
        assert!(
            matches!(oob, Err(PageError::SlotOutOfRange { .. })),
            "[{ctx}] read past slot_count must be SlotOutOfRange, got {oob:?}"
        );
    }
}

#[test]
fn oracle_adversarial_roundtrip_many_pages() {
    let mut total_inserts = 0u64;
    let mut total_tombstones = 0u64;
    let mut total_fulls = 0u64;
    let mut rng = Lcg(0x5EED_5EED_0001);
    for page_no in 0..1500u64 {
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        fresh_bag_page(&mut buf);
        let mut oracle = Oracle::new();

        let mut consecutive_full = 0u32;
        let mut op_no = 0u32;
        loop {
            op_no += 1;
            let ctx = format!("page {page_no} op {op_no}");
            match rng.below(10) {
                // ~70% inserts
                0..=6 => {
                    let payload = gen_payload(&mut rng, oracle.free);
                    let pre = *buf; // snapshot for Full-must-not-mutate check
                    let res = {
                        let mut page =
                            SlottedPage::open_prop_trusted(&mut buf[..]).expect("mut open");
                        page.insert_bag(&payload)
                    };
                    let should_fit = oracle.fits(payload.len());
                    match res {
                        Ok(slot) => {
                            assert!(
                                should_fit,
                                "[{ctx}] codec accepted len {} with oracle free {} (needed {}) — over-admission",
                                payload.len(),
                                oracle.free,
                                payload.len() + SLOT_SIZE
                            );
                            let want_id = oracle.record_insert(&payload);
                            assert_eq!(
                                slot.raw(),
                                want_id,
                                "[{ctx}] slot ids must be monotone appends (tombstones never reused)"
                            );
                            consecutive_full = 0;
                            total_inserts += 1;
                        }
                        Err(PageError::Full { needed, free }) => {
                            assert!(
                                !should_fit,
                                "[{ctx}] codec rejected len {} as Full{{needed:{needed},free:{free}}} but oracle free {} says it fits — under-admission (off-by-one)",
                                payload.len(),
                                oracle.free
                            );
                            assert_eq!(
                                free as usize, oracle.free,
                                "[{ctx}] Full.free disagrees with oracle"
                            );
                            assert_eq!(
                                needed as usize,
                                payload.len() + SLOT_SIZE,
                                "[{ctx}] Full.needed wrong"
                            );
                            assert_eq!(&pre[..], &buf[..], "[{ctx}] Full mutated the page");
                            consecutive_full += 1;
                            total_fulls += 1;
                        }
                        Err(other) => panic!("[{ctx}] unexpected insert error: {other:?}"),
                    }
                }
                // ~20% tombstones (incl. re-tombstone idempotency)
                7..=8 => {
                    if oracle.slots.is_empty() {
                        continue;
                    }
                    let i = rng.below(oracle.slots.len());
                    let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).expect("mut open");
                    page.tombstone(SlotId(i as u16))
                        .unwrap_or_else(|e| panic!("[{ctx}] tombstone({i}) err: {e:?}"));
                    oracle.slots[i] = None; // free NOT reclaimed — spec
                    total_tombstones += 1;
                }
                // ~10% mid-stream sweep
                _ => full_sweep(&buf, &oracle, &ctx),
            }
            if consecutive_full >= 4 || op_no > 3000 {
                break;
            }
        }
        full_sweep(&buf, &oracle, &format!("page {page_no} final"));
    }
    println!("volume: inserts={total_inserts} tombstones={total_tombstones} fulls={total_fulls}");
    assert!(
        total_inserts > 50_000,
        "coverage floor: expected > 50k successful inserts, got {total_inserts}"
    );
    assert!(
        total_fulls > 1_000,
        "coverage floor: Full arm underexercised"
    );
    assert!(
        total_tombstones > 10_000,
        "coverage floor: tombstone arm underexercised"
    );
}

/// Multi-slot EXACT FILL: drive free_space to exactly 0 with several
/// bags, so the LAST record's offset == slot-directory end with
/// count > 1 — the spot the `record_off < dir_end` relaxation governs
/// (in-tree test pins only the single max-bag shape). Must read back
/// through the FULL-CRC open.
#[test]
fn multi_slot_exact_fill_to_zero_free_space() {
    let mut rng = Lcg(0xFEED_BEEF);
    for round in 0..200u32 {
        let mut buf = Box::new([0u8; PAGE_SIZE]);
        fresh_bag_page(&mut buf);
        let mut oracle = Oracle::new();

        // Random fills until remaining < some threshold, then exact-fit.
        while oracle.free > 512 {
            let len = 1 + rng.below((oracle.free - SLOT_SIZE).min(2048));
            let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
            let payload: Vec<u8> = (0..len).map(|i| (i as u8) ^ (round as u8)).collect();
            match page.insert_bag(&payload) {
                Ok(s) => {
                    let want = oracle.record_insert(&payload);
                    assert_eq!(s.raw(), want);
                }
                Err(e) => panic!("round {round}: fill insert len {len} failed: {e:?}"),
            }
        }
        // The random fill can strand < 5 free bytes (no bag fits: min
        // needed = 1 + SLOT_SIZE). Only the fit-able rounds probe the
        // exact-fill boundary; stranded rounds probe the Full arm.
        if oracle.free < 1 + SLOT_SIZE {
            let r = {
                let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
                page.insert_bag(&[0xAB])
            };
            assert!(
                matches!(r, Err(PageError::Full { .. })),
                "round {round}: stranded free {} must reject 1-byte bag, got {r:?}",
                oracle.free
            );
            full_sweep(&buf, &oracle, &format!("stranded round {round}"));
            continue;
        }
        // Exact fit: len = free - SLOT_SIZE (>= 1 by the guard above).
        let final_len = oracle.free - SLOT_SIZE;
        let payload: Vec<u8> = (0..final_len).map(|i| (i as u8).wrapping_add(7)).collect();
        {
            let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
            let s = page.insert_bag(&payload).unwrap_or_else(|e| {
                panic!("round {round}: exact-fit len {final_len} rejected: {e:?}")
            });
            let want = oracle.record_insert(&payload);
            assert_eq!(s.raw(), want);
        }
        assert_eq!(oracle.free, 0, "round {round}: oracle free must be 0");

        // The last record must start exactly at the directory end.
        {
            let view = SlottedPageRef::open(&buf[..]).expect("full-CRC open of exact-full page");
            let hdr = view.header();
            assert_eq!(hdr.free_space, 0, "round {round}: header free_space");
            let dir_end = SLOT_AREA_START + (hdr.slot_count as usize) * SLOT_SIZE;
            // Read raw slot entry of the last slot to confirm off == dir_end.
            let last = (hdr.slot_count - 1) as usize;
            let e = SLOT_AREA_START + last * SLOT_SIZE;
            let off = u16::from_le_bytes([buf[e], buf[e + 1]]) as usize;
            assert_eq!(
                off, dir_end,
                "round {round}: exact-full page's last record must start at dir_end"
            );
        }

        // Nothing more fits — even a 1-byte bag.
        {
            let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
            let r = page.insert_bag(&[0xAB]);
            assert!(
                matches!(r, Err(PageError::Full { needed: 5, free: 0 })),
                "round {round}: 1-byte insert into full page must be Full{{5,0}}, got {r:?}"
            );
        }
        full_sweep(&buf, &oracle, &format!("exact-fill round {round}"));
    }
}

/// Max slot density: 1-byte bags until Full. Model says exactly
/// (PAGE_BODY_BYTES) / (1+SLOT_SIZE) = 1630 bags, free 2 left.
#[test]
fn max_slot_density_one_byte_bags() {
    let mut buf = Box::new([0u8; PAGE_SIZE]);
    fresh_bag_page(&mut buf);
    let mut oracle = Oracle::new();

    let mut n = 0usize;
    loop {
        let payload = [(n % 251) as u8 ^ 0x5A];
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        match page.insert_bag(&payload) {
            Ok(s) => {
                let want = oracle.record_insert(&payload);
                assert_eq!(s.raw(), want);
                n += 1;
            }
            Err(PageError::Full { needed, free }) => {
                assert_eq!(needed, MIN_NEEDED as u16);
                assert_eq!(free as usize, oracle.free);
                break;
            }
            Err(e) => panic!("unexpected: {e:?}"),
        }
        assert!(n <= 2038, "exceeded MAX_SLOTS worth of inserts?!");
    }
    assert_eq!(
        n,
        PAGE_BODY_BYTES / MIN_NEEDED,
        "expected exactly 1630 one-byte bags"
    );
    assert_eq!(
        oracle.free,
        PAGE_BODY_BYTES % MIN_NEEDED,
        "expected 2 bytes stranded"
    );
    full_sweep(&buf, &oracle, "max-density");
}

/// Tombstone semantics: neighbors intact byte-for-byte, idempotent
/// re-tombstone, freed space NEVER reused, slot ids never reused, and
/// the tombstone survives a full-CRC reopen (CRC recomputed).
#[test]
fn tombstone_no_reuse_no_leak_neighbors_intact() {
    let mut buf = Box::new([0u8; PAGE_SIZE]);
    fresh_bag_page(&mut buf);
    let mut oracle = Oracle::new();

    let a = vec![0x11u8; 100];
    let b = vec![0x22u8; 200];
    let c = vec![0x33u8; 300];
    for p in [&a, &b, &c] {
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        let s = page.insert_bag(p).unwrap();
        assert_eq!(s.raw(), oracle.record_insert(p));
    }
    let free_before = oracle.free;

    // Tombstone B twice (idempotent).
    for _ in 0..2 {
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        page.tombstone(SlotId(1)).unwrap();
    }
    oracle.slots[1] = None;
    full_sweep(&buf, &oracle, "post-tombstone-B");

    // Insert D: must take slot 3 (never reuse 1), free unchanged by tombstone.
    let d = vec![0x44u8; 150];
    {
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        assert_eq!(
            page.free_space() as usize,
            free_before,
            "tombstone must not reclaim"
        );
        let s = page.insert_bag(&d).unwrap();
        assert_eq!(s.raw(), 3, "tombstoned slot must not be reused");
        assert_eq!(s.raw(), oracle.record_insert(&d));
    }
    full_sweep(&buf, &oracle, "post-insert-D");

    // D's payload must not have been carved out of B's freed region in a
    // way that corrupts A/C — sweep already proves byte equality, but
    // also check D landed BELOW C's offset (heap keeps growing backward,
    // no in-place reuse of B's hole).
    {
        let _view = SlottedPageRef::open(&buf[..]).unwrap();
        let entry = |i: usize| {
            let e = SLOT_AREA_START + i * SLOT_SIZE;
            (
                u16::from_le_bytes([buf[e], buf[e + 1]]),
                u16::from_le_bytes([buf[e + 2], buf[e + 3]]),
            )
        };
        let (off_c, _) = entry(2);
        let (off_d, _) = entry(3);
        assert!(
            off_d < off_c,
            "heap must grow strictly backward; no hole reuse"
        );
    }
    // Tombstone everything; page must still full-CRC-open and read all-None.
    for i in 0..4u16 {
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        page.tombstone(SlotId(i)).unwrap();
        oracle.slots[i as usize] = None;
    }
    full_sweep(&buf, &oracle, "all-tombstoned");
}

/// insert_bag argument validation + the Full-vs-Format distinction that
/// stage_bag's fall-through relies on: an 8148 bag on a nearly-full page
/// must be Full (so stage_bag opens a fresh page), NEVER Format.
#[test]
fn insert_bag_rejects_and_full_vs_format_discrimination() {
    let mut buf = Box::new([0u8; PAGE_SIZE]);
    fresh_bag_page(&mut buf);

    {
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        // Empty bag: Format, not Full.
        assert!(matches!(page.insert_bag(&[]), Err(PageError::Format(_))));
        // One over max: Format, not Full.
        let over = vec![0u8; PROP_BAG_MAX_BYTES + 1];
        assert!(matches!(page.insert_bag(&over), Err(PageError::Format(_))));
        // Exactly max on a fresh page: fits, free_space -> 0.
        let max = vec![0xC3u8; PROP_BAG_MAX_BYTES];
        let s = page.insert_bag(&max).unwrap();
        assert_eq!(s.raw(), 0);
        assert_eq!(page.free_space(), 0);
    }
    // Reopen a SECOND fresh page, occupy 1 byte, then try the max bag:
    // must be Full (fall-through signal), not Format.
    let mut buf2 = Box::new([0u8; PAGE_SIZE]);
    fresh_bag_page(&mut buf2);
    {
        let mut page = SlottedPage::open_prop_trusted(&mut buf2[..]).unwrap();
        page.insert_bag(&[1]).unwrap();
        let max = vec![0xC3u8; PROP_BAG_MAX_BYTES];
        let r = page.insert_bag(&max);
        assert!(
            matches!(r, Err(PageError::Full { .. })),
            "max bag on non-empty page must be Full (stage_bag fall-through), got {r:?}"
        );
    }
    // Wrong page type: insert_bag/read_bag on a Node page.
    let mut nbuf = Box::new([0u8; PAGE_SIZE]);
    let hdr = PageHeader::new(PageId::new(1), PageType::Node, TenantId::DEFAULT);
    SlottedPage::init(&mut nbuf[..], hdr).unwrap();
    {
        let mut page = SlottedPage::open(&mut nbuf[..]).unwrap();
        assert!(matches!(
            page.insert_bag(&[1]),
            Err(PageError::WrongPageType { .. })
        ));
        let view = SlottedPageRef::open(&nbuf[..]).unwrap();
        assert!(matches!(
            view.read_bag(SlotId(0)),
            Err(PageError::WrongPageType { .. })
        ));
        // And open_prop_trusted must refuse a Node page outright.
        assert!(matches!(
            SlottedPageRef::open_prop_trusted(&nbuf[..]),
            Err(PageError::WrongPageType { .. })
        ));
    }
}

/// Half-tombstone corruption: a directory entry (0, len>0) must be a
/// LOUD Format error — neither treated as a tombstone (silent None)
/// nor as a valid slot (offset 0 is inside the header).
#[test]
fn half_tombstone_entry_is_loud_corruption() {
    let mut buf = Box::new([0u8; PAGE_SIZE]);
    fresh_bag_page(&mut buf);
    {
        let mut page = SlottedPage::open_prop_trusted(&mut buf[..]).unwrap();
        page.insert_bag(&[0xEE; 32]).unwrap();
    }
    // Corrupt slot 0's entry: offset -> 0, keep len 32. (open_prop_trusted
    // skips CRC so the tamper is visible to the hot-read path shape.)
    buf[SLOT_AREA_START..SLOT_AREA_START + 2].copy_from_slice(&0u16.to_le_bytes());
    let view = SlottedPageRef::open_prop_trusted(&buf[..]).unwrap();
    let r = view.read_bag(SlotId(0));
    assert!(
        matches!(r, Err(PageError::Format(_))),
        "(0, 32) entry must be Format corruption, got {r:?}"
    );
}
