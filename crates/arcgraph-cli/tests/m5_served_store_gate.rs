//! M5-D2 served-store-completeness gates (`docs/design/M5D-REDESIGN-AMENDMENT.md`
//! §3 + §7 M5-D2 row):
//!
//! - **INV-M5.20 `loaded_store_disk_complete`** — complete-store-set census
//!   over the loaded generation with NON-EMPTY STORE_PROPS / STORE_TEL for
//!   property/edge-bearing input, plus `cold_open_serves_props` (production
//!   bootstrap; every loaded property served back). This pair is RED against
//!   the superseded `dbf13a5a` shape by construction — the V-3 regression pin.
//! - **INV-M5.17 (hardened, anti-tautology)** — a DISK-LEVEL loader-vs-
//!   incremental differential over extent contents + record
//!   `property_ref`/`*_tel_ref` fields. The loader side is decoded from RAW
//!   EXTENT BYTES with no bootstrap and no rebuild; the oracle side is an
//!   independent production incremental ingest of the same logical content.
//!   Post-rebuild `scan_out`/`scan_in` oracles are deliberately NOT the
//!   verdict here (the #780 in-RAM rebuild satisfies them from rel records
//!   alone — the armed empty-TEL child below demonstrates exactly that).
//! - **INV-M5.12 (terminus fix)** — the ULP-adversarial fidelity oracle
//!   salvaged from closed PR #1504 (amendment §9), re-anchored per §3.3 to
//!   terminate at the SERVED store: bit-exact floats + byte-exact opaque
//!   embedder payloads read back through production bootstrap + the
//!   production property read path (the #1442 corpus; memory:
//!   serde_json's default float parse is ULP-lossy).
//!
//! Every gate proves RED-on-revert through cfg-gated fault seams
//! (`ARCGRAPH_M5_SHIP_EMPTY_PROPS`, `ARCGRAPH_M5_SHIP_EMPTY_TEL`,
//! `ARCGRAPH_M5_LOSSY_FLOAT_BITS`) armed in child processes of this same
//! binary, executed in the `arcgraph-cli-release-fault-injection` CI lane.
//!
//! Oracle-independence note: the TEL disk decode below is a hand-rolled
//! byte reader against the documented design-v2 §3.3 block layout — NOT the
//! production writer's own code — so a writer-side layout bug cannot
//! self-certify.

use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::Command;

use arcgraph_cli::bootstrap::{BootstrapMode, bootstrap_storage_backend};
use arcgraph_cli::m5_load::{
    FRESH_LOAD_MIGRATION_LSN, LoadFormat, LoadOutcome, canonical_property_bag, load_data_dir,
};
use arcgraph_core::{LabelId, NodeId, PAGE_SIZE, PageHeader, PageType, RelId, TenantId, TypeId};
use arcgraph_storage::crud::{
    CrudStore, PropertyData, commit, create_node, create_rel, read_node_with_store,
    read_rel_with_store, scan_in, scan_out,
};
use arcgraph_storage::extent::{EXTENT_PAGES, production_extent_store_path};
use arcgraph_storage::m4_migration::read_extent_ledger;
use arcgraph_storage::property::BlobRef;
use arcgraph_storage::records::{PROP_BAG_MAX_BYTES, SlotId, SlottedPageRef};
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::wal::{STORE_PROPS, STORE_RECORD, STORE_RELS, STORE_TEL};
use tempfile::tempdir;

const LOAD_TENANT: TenantId = TenantId::new(91);
const FINAL_GENERATION: &str = "gen-load-v6";

/// design-v2 §3.3 TEL block header size (asserted against the layout the
/// loader documents; deliberately a literal, not the producer's constant).
const TEL_BLOCK_HEADER: usize = 32;
const TEL_ENTRY: usize = 32;
const TEL_NO_PREV: u64 = u64::MAX;

// #1519 DENSIFY read-path constants — hand-rolled, deliberately NOT
// imported from `arcgraph_storage::m4_migration` (oracle independence:
// this file's whole point is that a writer-side layout bug cannot
// self-certify against its own constants/encoder).
/// `PageHeader::flags` value for a densified packed TEL page.
const TEL_FLAG_PACKED: u16 = 1;
/// Packed-page directory: 4 B block_count + 4 B reserved.
const TEL_DIR_HEADER: usize = 8;
/// Packed-page directory slot: owner(8) + type(4) + offset(2) + len(2).
const TEL_DIR_SLOT: usize = 16;

/// Decode an opaque #1519 TEL ref into `(page_no, slot)`: high 48 bits are
/// the physical `PageType::Tel` page id, low 16 bits are the directory
/// slot index within that page (always 0 for a non-packed/supernode
/// page). Hand-rolled inverse of `m4_migration::encode_tel_ref` — kept
/// independent for the same oracle-independence reason as the rest of
/// this decoder.
fn decode_ref(reference: u64) -> (u64, u16) {
    (reference >> 16, (reference & 0xFFFF) as u16)
}

#[cfg(feature = "fault-injection")]
const EMPTY_PROPS_ENV: &str = "ARCGRAPH_M5_SHIP_EMPTY_PROPS";
#[cfg(feature = "fault-injection")]
const EMPTY_TEL_ENV: &str = "ARCGRAPH_M5_SHIP_EMPTY_TEL";
#[cfg(feature = "fault-injection")]
const LOSSY_FLOAT_ENV: &str = "ARCGRAPH_M5_LOSSY_FLOAT_BITS";
/// #1519 RED-on-revert seam: force every TEL block down the pre-#1519
/// page-per-block path regardless of size (see `m4_migration.rs`
/// `flush_tel_block`'s `force_page_per_block`).
#[cfg(feature = "fault-injection")]
const TEL_PAGE_PER_BLOCK_ENV: &str = "ARCGRAPH_M5_TEL_PAGE_PER_BLOCK";

/// Serialize this file's parent tests (same idiom as
/// `m5_load_attach_gate.rs::serialize_gate`, M5-D1). Every test below
/// holds a `DataDirLock` — `load_fixture` acquires it through commit
/// (`m5_load` invariant 1), drops it, and the test re-acquires the SAME
/// dir via the production durable bootstrap (or an `arcgraph check`
/// child) — while SIBLING tests fork gate subprocesses
/// (`assert_red_under`, `CARGO_BIN_EXE_arcgraph`). On Unix, a
/// forked-but-not-yet-exec'd child momentarily shares the parent's open
/// file descriptions (`O_CLOEXEC` closes only AT exec), so a
/// concurrently spawned sibling child can extend a just-dropped `flock`
/// by that fork window — under machine load the window stretches and
/// the re-acquire sees `EWOULDBLOCK` ("data dir is already in use by
/// another `arcgraph serve` process", observed deterministically in the
/// full-workspace debug core shard on `inv_m5_12`). Holding one
/// file-scoped mutex across each test body means no sibling forks while
/// any test holds or hands over a lock, which closes the race by
/// construction instead of by retry. Child-process invocations (env
/// dispatch via `--exact`) each run alone in their own process — the
/// guard is uncontended there.
fn serialize_gate() -> std::sync::MutexGuard<'static, ()> {
    static GATE_SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GATE_SERIAL
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[cfg(feature = "fault-injection")]
fn armed(var: &str) -> bool {
    std::env::var_os(var).is_some()
}

#[cfg(feature = "fault-injection")]
fn any_seam_armed() -> bool {
    armed(EMPTY_PROPS_ENV)
        || armed(EMPTY_TEL_ENV)
        || armed(LOSSY_FLOAT_ENV)
        || armed(TEL_PAGE_PER_BLOCK_ENV)
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(DIGITS[usize::from(byte >> 4)] as char);
        out.push(DIGITS[usize::from(byte & 0x0f)] as char);
    }
    out
}

/// #1442 ULP-adversarial float corpus (salvaged from closed PR #1504):
/// signed zero, adjacent doubles at one, an f32-rounding boundary,
/// subnormal/normal edges, max finite — extended with NaN payload bit
/// patterns (quiet/signaling-shaped) that any decimal round-trip destroys.
const ULP_CORPUS: [u64; 11] = [
    0x0000_0000_0000_0000, // +0.0
    0x8000_0000_0000_0000, // -0.0
    0x3ff0_0000_0000_0001, // 1.0 + 1 ULP
    0x3fef_ffff_ffff_ffff, // 1.0 - 1 ULP
    0x3ff0_0000_1000_0000, // f32-rounding boundary
    0x0010_0000_0000_0000, // smallest normal
    0x0000_0000_0000_0001, // smallest subnormal
    0x7fef_ffff_ffff_ffff, // max finite
    0x7ff8_0000_0000_0000, // canonical quiet NaN
    0x7ff0_0000_0000_0001, // signaling-shaped NaN payload
    0xfff8_dead_beef_0001, // negative NaN with payload bits
];

/// One expected input record.
#[derive(Debug, Clone)]
struct FixtureNode {
    external: Vec<u8>,
    label: u32,
    float_bits: u64,
    opaque: Vec<u8>,
}

#[derive(Debug, Clone)]
struct FixtureRel {
    external: Vec<u8>,
    source: Vec<u8>,
    target: Vec<u8>,
    type_id: u32,
    float_bits: u64,
    opaque: Vec<u8>,
}

struct Fixture {
    /// Canonically ordered (external-id byte order == dense id order).
    nodes: Vec<FixtureNode>,
    rels: Vec<FixtureRel>,
}

impl Fixture {
    fn node_id(&self, external: &[u8]) -> u64 {
        self.nodes
            .iter()
            .position(|node| node.external == external)
            .map(|index| index as u64 + 1)
            .expect("fixture node exists")
    }

    fn expected_node_bag(&self, id: u64) -> Vec<u8> {
        let node = &self.nodes[usize::try_from(id - 1).unwrap()];
        canonical_property_bag(node.float_bits, &node.opaque)
    }

    fn expected_rel_bag(&self, id: u64) -> Vec<u8> {
        let rel = &self.rels[usize::try_from(id - 1).unwrap()];
        canonical_property_bag(rel.float_bits, &rel.opaque)
    }

    /// Expected per-node adjacency: `(node, type) -> sorted (neighbor,
    /// rel_id)` for each direction, derived from the raw input.
    #[cfg(feature = "fault-injection")]
    fn expected_adjacency(&self) -> (AdjacencyMap, AdjacencyMap) {
        let mut out: AdjacencyMap = BTreeMap::new();
        let mut inn: AdjacencyMap = BTreeMap::new();
        for (index, rel) in self.rels.iter().enumerate() {
            let rel_id = index as u64 + 1;
            let source = self.node_id(&rel.source);
            let target = self.node_id(&rel.target);
            out.entry((source, rel.type_id))
                .or_default()
                .push((target, rel_id));
            inn.entry((target, rel.type_id))
                .or_default()
                .push((source, rel_id));
        }
        for entries in out.values_mut().chain(inn.values_mut()) {
            entries.sort_unstable();
        }
        (out, inn)
    }
}

type AdjacencyMap = BTreeMap<(u64, u32), Vec<(u64, u64)>>;

/// Property + TEL-bearing fixture: the ULP corpus, distinct opaque
/// payloads (incl. the 0..=255 ladder and an oversized DEC-4-chained
/// payload), multiple relationship types per node, and a supernode whose
/// out-degree spans multiple on-disk TEL blocks.
fn served_fixture(oversized_bag: bool) -> Fixture {
    let mut nodes = Vec::new();
    for (index, bits) in ULP_CORPUS.iter().enumerate() {
        let mut opaque: Vec<u8> = (0_u8..=255).collect();
        opaque.extend_from_slice(&(index as u64).to_le_bytes());
        nodes.push(FixtureNode {
            external: format!("m5d2-node-{index:04}").into_bytes(),
            label: 7 + (index as u32 % 3),
            float_bits: *bits,
            opaque,
        });
    }
    if oversized_bag {
        // Oversized opaque embedder payload: bag = 8 + len > PROP_BAG_MAX,
        // exercising the production DEC-4 chain + first-checkpoint
        // page-image path end-to-end.
        let len = PROP_BAG_MAX_BYTES + 513;
        let opaque: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        nodes.push(FixtureNode {
            external: b"m5d2-node-big".to_vec(),
            label: 42,
            float_bits: 0x4009_21fb_5444_2d18, // pi
            opaque,
        });
    }
    // Supernode: out-degree > one on-disk TEL block (254 entries/page).
    let hub_external = b"m5d2-hub".to_vec();
    nodes.push(FixtureNode {
        external: hub_external.clone(),
        label: 5,
        float_bits: 0x3ff0_0000_0000_0000,
        opaque: b"hub".to_vec(),
    });
    nodes.sort_by(|left, right| left.external.cmp(&right.external));

    let mut rels = Vec::new();
    let first = nodes[0].external.clone();
    let second = nodes[1].external.clone();
    let third = nodes[2].external.clone();
    // Multi-type adjacency on one node pair (single-channel block split).
    for (suffix, ty) in [("t3", 3_u32), ("t9", 9)] {
        rels.push(FixtureRel {
            external: format!("m5d2-rel-{suffix}").into_bytes(),
            source: first.clone(),
            target: second.clone(),
            type_id: ty,
            float_bits: 0x3fd5_5555_5555_5555,
            opaque: b"edge".to_vec(),
        });
    }
    rels.push(FixtureRel {
        external: b"m5d2-rel-chain".to_vec(),
        source: second.clone(),
        target: third.clone(),
        type_id: 3,
        float_bits: 0x7ff0_0000_0000_0000, // +inf survives bit-exactly
        opaque: Vec::new(),
    });
    // 300 hub -> second edges of one type: 2 on-disk blocks for the hub's
    // out chain and for second's in chain.
    for index in 0..300_u32 {
        rels.push(FixtureRel {
            external: format!("m5d2-rel-hub-{index:04}").into_bytes(),
            source: hub_external.clone(),
            target: second.clone(),
            type_id: 5,
            float_bits: u64::from(index) << 1,
            opaque: index.to_le_bytes().to_vec(),
        });
    }
    rels.sort_by(|left, right| left.external.cmp(&right.external));
    Fixture { nodes, rels }
}

/// #1519 `tel_disk_size_is_dense` fixture: `n_owners` distinct source
/// nodes, each fanning out to `degree` distinct targets across `n_types`
/// relationship types (round-robin) — avg out-degree squarely in the
/// low-degree regime the M5-D3 100M rung STOP-report measured (avg 5 /
/// 7 types), every block WELL below
/// [`arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES`]
/// (126), so every block is a densify-packing candidate — the exact
/// regime that blew up ~200x under page-per-block.
fn dense_fixture(n_owners: u32, degree: u32, n_types: u32) -> Fixture {
    let mut nodes = Vec::new();
    for index in 0..n_owners {
        nodes.push(FixtureNode {
            external: format!("src-{index:05}").into_bytes(),
            label: 1,
            float_bits: 0,
            opaque: Vec::new(),
        });
    }
    for index in 0..degree {
        nodes.push(FixtureNode {
            external: format!("dst-{index:05}").into_bytes(),
            label: 2,
            float_bits: 0,
            opaque: Vec::new(),
        });
    }
    nodes.sort_by(|left, right| left.external.cmp(&right.external));

    let mut rels = Vec::new();
    let mut rel_index = 0_u64;
    for owner in 0..n_owners {
        for target in 0..degree {
            let type_id = (owner * degree + target) % n_types.max(1);
            rels.push(FixtureRel {
                external: format!("e-{rel_index:08}").into_bytes(),
                source: format!("src-{owner:05}").into_bytes(),
                target: format!("dst-{target:05}").into_bytes(),
                type_id,
                float_bits: 0,
                opaque: Vec::new(),
            });
            rel_index += 1;
        }
    }
    rels.sort_by(|left, right| left.external.cmp(&right.external));
    Fixture { nodes, rels }
}

/// Independent native producer (literal schema formatting — deliberately
/// NOT the loader's encoders; salvaged oracle shape from PR #1504).
fn write_native_fixture(fixture: &Fixture, path: &Path) {
    use std::io::Write;
    let mut writer = std::io::BufWriter::new(File::create(path).unwrap());
    for node in &fixture.nodes {
        writeln!(
            writer,
            "{{\"kind\":\"node\",\"external_id\":\"{}\",\"label_or_type\":{},\"float_bits\":\"{:016x}\",\"opaque\":\"{}\"}}",
            hex(&node.external),
            node.label,
            node.float_bits,
            hex(&node.opaque),
        )
        .unwrap();
    }
    for rel in &fixture.rels {
        writeln!(
            writer,
            "{{\"kind\":\"relationship\",\"external_id\":\"{}\",\"source_id\":\"{}\",\"target_id\":\"{}\",\"label_or_type\":{},\"float_bits\":\"{:016x}\",\"opaque\":\"{}\"}}",
            hex(&rel.external),
            hex(&rel.source),
            hex(&rel.target),
            rel.type_id,
            rel.float_bits,
            hex(&rel.opaque),
        )
        .unwrap();
    }
    writer.flush().unwrap();
}

fn load_fixture(fixture: &Fixture) -> (tempfile::TempDir, PathBuf) {
    let dir = tempdir().unwrap();
    let input = dir.path().join("input.jsonl");
    write_native_fixture(fixture, &input);
    let root = dir.path().join("data");
    let outcome = load_data_dir(&input, LoadFormat::Native, &root, LOAD_TENANT)
        .expect("production load of the served fixture");
    let LoadOutcome::Loaded(report) = outcome else {
        panic!("fresh load did not build: {outcome:?}");
    };
    assert_eq!(report.nodes as usize, fixture.nodes.len());
    assert_eq!(report.relationships as usize, fixture.rels.len());
    let generation = root.join(FINAL_GENERATION);
    assert!(generation.is_dir(), "committed generation missing");
    (dir, root)
}

// ─────────────────────────────────────────────────────────────────────
// Raw-disk decode (the anti-tautology side: no bootstrap, no rebuild)
// ─────────────────────────────────────────────────────────────────────

/// Read every mapped page of one extent store: `page_no -> PAGE bytes`.
fn read_store_pages(generation: &Path, store_id: u16) -> BTreeMap<u64, Box<[u8; PAGE_SIZE]>> {
    let path = production_extent_store_path(generation, LOAD_TENANT, store_id).unwrap();
    let ledger = read_extent_ledger(&path, LOAD_TENANT, store_id).unwrap();
    let mut file = File::open(&path).unwrap();
    let file_len = file.metadata().unwrap().len();
    let mut pages = BTreeMap::new();
    for extent in ledger {
        for within in 0..EXTENT_PAGES {
            let page_no = extent.logical_extent * EXTENT_PAGES + within;
            let physical = extent.physical_offset + within * PAGE_SIZE as u64;
            if physical + PAGE_SIZE as u64 > file_len {
                continue;
            }
            let mut bytes = Box::new([0_u8; PAGE_SIZE]);
            file.seek(SeekFrom::Start(physical)).unwrap();
            file.read_exact(bytes.as_mut()).unwrap();
            if bytes.iter().all(|byte| *byte == 0) {
                continue;
            }
            pages.insert(page_no, bytes);
        }
    }
    pages
}

struct DiskStore {
    nodes: BTreeMap<u64, arcgraph_core::record::NodeRecord>,
    rels: BTreeMap<u64, arcgraph_core::record::RelRecord>,
    prop_pages: BTreeMap<u64, Box<[u8; PAGE_SIZE]>>,
    tel_pages: BTreeMap<u64, Box<[u8; PAGE_SIZE]>>,
}

fn decode_disk_store(generation: &Path) -> DiskStore {
    let mut nodes = BTreeMap::new();
    for bytes in read_store_pages(generation, STORE_RECORD).values() {
        let view = SlottedPageRef::open(bytes.as_ref()).unwrap();
        for (_, record) in view.iter_nodes() {
            nodes.insert(record.id, record);
        }
    }
    let mut rels = BTreeMap::new();
    for bytes in read_store_pages(generation, STORE_RELS).values() {
        let view = SlottedPageRef::open(bytes.as_ref()).unwrap();
        for (_, record) in view.iter_rels() {
            rels.insert(record.id, record);
        }
    }
    DiskStore {
        nodes,
        rels,
        prop_pages: read_store_pages(generation, STORE_PROPS),
        tel_pages: read_store_pages(generation, STORE_TEL),
    }
}

impl DiskStore {
    /// Resolve one record's `property_ref` against the STORE_PROPS extent
    /// bytes (slotted bags only — the disk-differential fixture carries no
    /// chained bag; chains are covered at the served terminus by
    /// INV-M5.20/.12).
    fn disk_bag(&self, property_ref: u64) -> Vec<u8> {
        let blob_ref = BlobRef::decode(property_ref).expect("record carries an overflow ref");
        assert_ne!(
            blob_ref.slot_id, 0,
            "disk differential expects slotted bags"
        );
        let page = self
            .prop_pages
            .get(&blob_ref.page_id)
            .expect("property_ref names a mapped STORE_PROPS page");
        let view = SlottedPageRef::open(page.as_ref()).expect("valid prop page");
        assert_eq!(view.header().page_type, PageType::PropSlotted.as_byte());
        view.read_bag(SlotId(blob_ref.slot_id - 1))
            .expect("bag slot decodes")
            .expect("bag slot occupied")
            .to_vec()
    }

    /// Resolve one (page_no, slot) TEL ref to its block's raw bytes
    /// within that page's body: `(owner, block_size, entry_count, prev_ref,
    /// label, block_body)`. Dispatches on `PageHeader::flags` — `0` is the
    /// pre-#1519 supernode/chain shape (the sole block sits at body offset
    /// 0); `TEL_FLAG_PACKED` is the #1519 densified shape (an intra-page
    /// directory locates the block's byte range). This is the read-path
    /// contract's core: (owner,type) -> page_no -> [directory] ->
    /// block byte-range -> entries.
    fn resolve_tel_block(&self, page_no: u64, slot: u16) -> (u64, usize, usize, u64, u32, Vec<u8>) {
        let page = self
            .tel_pages
            .get(&page_no)
            .unwrap_or_else(|| panic!("TEL ref names unmapped page {page_no}"));
        let header =
            PageHeader::from_bytes(page[..PageHeader::SIZE].try_into().expect("header slice"))
                .expect("valid TEL page header");
        assert_eq!(header.page_type, PageType::Tel.as_byte(), "TEL page type");
        assert_eq!(header.tenant_id, LOAD_TENANT.raw(), "TEL page tenant");
        assert_eq!(header.page_id, page_no, "TEL page identity");
        assert_eq!(
            header.lsn,
            FRESH_LOAD_MIGRATION_LSN.raw(),
            "TEL page frontier stamp"
        );
        assert_eq!(
            crc32c::crc32c(&page[PageHeader::SIZE..]),
            header.checksum,
            "TEL page body checksum"
        );
        let body = &page[PageHeader::SIZE..];
        let (owner_from_dir, block_bytes): (Option<u64>, &[u8]) = if header.flags == TEL_FLAG_PACKED
        {
            // Densified packed page: body[0..4] = block_count, body[4..8]
            // reserved, then `block_count` 16 B directory slots, then the
            // packed blocks themselves.
            let block_count = u32::from_le_bytes(body[0..4].try_into().unwrap());
            assert!(
                (slot as u32) < block_count,
                "TEL ref names slot {slot} but packed page {page_no} only has \
                 {block_count} directory entries"
            );
            let slot_offset = TEL_DIR_HEADER + slot as usize * TEL_DIR_SLOT;
            let dir_owner =
                u64::from_le_bytes(body[slot_offset..slot_offset + 8].try_into().unwrap());
            let dir_type =
                u32::from_le_bytes(body[slot_offset + 8..slot_offset + 12].try_into().unwrap());
            let dir_block_offset =
                u16::from_le_bytes(body[slot_offset + 12..slot_offset + 14].try_into().unwrap())
                    as usize;
            let dir_block_len =
                u16::from_le_bytes(body[slot_offset + 14..slot_offset + 16].try_into().unwrap())
                    as usize;
            let _ = dir_type; // re-derived from the block body itself below
            (
                Some(dir_owner),
                &body[dir_block_offset..dir_block_offset + dir_block_len],
            )
        } else {
            assert_eq!(header.flags, 0, "unknown TEL page flags {}", header.flags);
            assert_eq!(
                slot, 0,
                "non-packed TEL page {page_no} referenced at nonzero slot {slot}"
            );
            // Supernode/chain page (pre-#1519 shape): the sole block sits
            // at body offset 0, sized to exactly `block_size` bytes — NOT
            // the whole page body (the body has trailing zero padding
            // after a sized-to-fit block).
            let block_size = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
            (None, &body[..block_size])
        };
        let src = u64::from_le_bytes(block_bytes[0..8].try_into().unwrap());
        if let Some(dir_owner) = owner_from_dir {
            assert_eq!(
                dir_owner, src,
                "packed directory owner must match the block's own owner field"
            );
        }
        let block_size = u32::from_le_bytes(block_bytes[8..12].try_into().unwrap()) as usize;
        let entry_count = u32::from_le_bytes(block_bytes[12..16].try_into().unwrap()) as usize;
        let prev = u64::from_le_bytes(block_bytes[16..24].try_into().unwrap());
        let label = u32::from_le_bytes(block_bytes[24..28].try_into().unwrap());
        assert_eq!(
            block_size,
            TEL_BLOCK_HEADER + entry_count * TEL_ENTRY,
            "sized-to-fit block"
        );
        assert_eq!(block_bytes.len(), block_size, "block byte-range length");
        (
            src,
            block_size,
            entry_count,
            prev,
            label,
            block_bytes.to_vec(),
        )
    }

    /// Walk one TEL chain from a record ref through the raw STORE_TEL
    /// bytes: hand-rolled design-v2 §3.3 block decode (oracle-independent
    /// of the producer), extended for #1519 densified page packing.
    /// Returns `(type, neighbor, rel_id)` entries in COMMITTED order
    /// (oldest block first, physical slot order within each block).
    ///
    /// PLACEMENT PIN (INV-M5.17/INV-M5.8, kills the order-blind MUT-3
    /// class): the producer appends strictly ascending `(type_id,
    /// rel_id)` per chain (`m4_migration.rs::append_tel_entry`) and
    /// places entry `i` BACKWARD at `block_size - (i+1)*32` (design-v2
    /// §3.3 — the exact arithmetic M6's `TelBlock::entry_bytes` reader
    /// faults these pages with). Decoding at the documented backward
    /// physical offsets must therefore reproduce the committed strictly
    /// ascending order WITHOUT SORTING; this is asserted below on the
    /// raw decoded sequence. A loader-only FORWARD-placement regression
    /// keeps the same entry byte-SET (checksum-consistent, set-blind
    /// oracles stay green) but reverses the per-block decoded order for
    /// backward readers — and reds here.
    fn walk_tel_chain(&self, owner: u64, head_ref: u64) -> Vec<(u32, u64, u64)> {
        // Blocks in walk order (newest block first via `prev_block_ptr`);
        // each block's entries in physical backward-offset order (slot 0
        // = oldest = the bytes nearest the block's end).
        let mut blocks: Vec<Vec<(u32, u64, u64)>> = Vec::new();
        let mut next = head_ref;
        while next != 0 {
            let (page_no, slot) = decode_ref(next);
            let (src, block_size, entry_count, prev, label, block_bytes) =
                self.resolve_tel_block(page_no, slot);
            assert_eq!(src, owner, "TEL block owner");
            let body = block_bytes.as_slice();
            // Entries are written BACKWARDS (design-v2 §3.3): entry[i]
            // occupies block bytes [size-(i+1)*32, size-i*32); entry 0 is
            // the oldest.
            let mut block = Vec::with_capacity(entry_count);
            for index in 0..entry_count {
                let end = block_size - index * TEL_ENTRY;
                let start = end - TEL_ENTRY;
                let raw = &body[start..end];
                let dst = u64::from_le_bytes(raw[0..8].try_into().unwrap());
                let rel = u64::from_le_bytes(raw[8..16].try_into().unwrap());
                let created = u64::from_le_bytes(raw[16..24].try_into().unwrap());
                let expired = u64::from_le_bytes(raw[24..32].try_into().unwrap());
                assert_eq!(created, FRESH_LOAD_MIGRATION_LSN.raw(), "entry created_lsn");
                assert_eq!(expired, u64::MAX, "loaded entries are alive");
                block.push((label, dst, rel));
            }
            blocks.push(block);
            next = if prev == TEL_NO_PREV { 0 } else { prev };
        }
        // Committed order: reverse the newest-first walk so blocks run
        // oldest → newest, keeping each block's raw physical decode order.
        let entries: Vec<(u32, u64, u64)> = blocks.into_iter().rev().flatten().collect();
        // PLACEMENT PIN: the UNSORTED decoded sequence at physical
        // offsets must be strictly ascending in (type_id, rel_id) — the
        // producer's committed append order. Do NOT sort before this
        // check; sorting is exactly what made the differential
        // order-blind (MUT-3: forward placement shipped green).
        for window in entries.windows(2) {
            let (prev_type, _, prev_rel) = window[0];
            let (next_type, _, next_rel) = window[1];
            assert!(
                (prev_type, prev_rel) < (next_type, next_rel),
                "INV-M5.17: TEL chain for owner {owner} decoded at the design-v2 \
                 §3.3 backward physical offsets is NOT in committed strictly \
                 ascending (type_id, rel_id) order — within-block entry \
                 PLACEMENT diverges from the format contract M6's backward \
                 reader depends on (got ({prev_type}, {prev_rel}) then \
                 ({next_type}, {next_rel}))"
            );
        }
        entries
    }

    /// Per-(owner, type) adjacency derived purely from disk bytes.
    fn disk_adjacency(
        &self,
        direction_ref: impl Fn(&arcgraph_core::record::NodeRecord) -> u64,
    ) -> AdjacencyMap {
        let mut map: AdjacencyMap = BTreeMap::new();
        for (id, record) in &self.nodes {
            let head = direction_ref(record);
            if head == 0 {
                continue;
            }
            for (type_id, neighbor, rel_id) in self.walk_tel_chain(*id, head) {
                map.entry((*id, type_id))
                    .or_default()
                    .push((neighbor, rel_id));
            }
        }
        // Set-membership canonicalization ONLY: entry PLACEMENT is pinned
        // UNSORTED inside `walk_tel_chain` (strictly ascending at the
        // physical backward offsets) BEFORE this sort, so this cannot
        // re-blind the differential to order (the MUT-3 hole). The sort
        // here canonicalizes for comparison against the incremental
        // oracle's scan output, whose iteration order is a live-writer
        // concern, not the loaded-format contract.
        for entries in map.values_mut() {
            entries.sort_unstable();
        }
        map
    }
}

// ─────────────────────────────────────────────────────────────────────
// Child spawn helper (RED-on-revert controls, CI-executed)
// ─────────────────────────────────────────────────────────────────────

#[cfg(feature = "fault-injection")]
fn assert_red_under(test_name: &str, env_var: &str) {
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test_name, "--nocapture"])
        .env(env_var, "1")
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "{test_name} stayed GREEN under armed {env_var} — the gate is a no-op\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

// ─────────────────────────────────────────────────────────────────────
// Gate 1 — INV-M5.20 loaded-store-disk-complete + cold_open_serves_props
// ─────────────────────────────────────────────────────────────────────

/// INV-M5.20: after attach, STORE_PROPS and STORE_TEL are non-empty on
/// disk for property/edge-bearing input, and a production cold open
/// serves EVERY loaded property value (bag = raw float bits + opaque,
/// byte-exact) — plus a full-store checksum over the hydrated views.
/// RED-on-revert: ship STORE_PROPS empty (`ARCGRAPH_M5_SHIP_EMPTY_PROPS`)
/// or STORE_TEL empty (`ARCGRAPH_M5_SHIP_EMPTY_TEL`).
#[test]
fn inv_m5_20_loaded_store_disk_complete_and_cold_open_serves_props() {
    let _serial = serialize_gate();
    let fixture = served_fixture(true);
    let (_dir, root) = load_fixture(&fixture);
    let generation = root.join(FINAL_GENERATION);

    // Disk census: the property/edge-bearing load ships NON-EMPTY
    // STORE_PROPS + STORE_TEL extent ledgers (this is the assertion that
    // is red against the dbf13a5a shape by construction).
    for (store_id, name) in [(STORE_PROPS, "STORE_PROPS"), (STORE_TEL, "STORE_TEL")] {
        let path = production_extent_store_path(&generation, LOAD_TENANT, store_id).unwrap();
        let ledger = read_extent_ledger(&path, LOAD_TENANT, store_id).unwrap();
        assert!(
            !ledger.is_empty(),
            "INV-M5.20: {name} shipped EMPTY for property/edge-bearing input"
        );
    }

    // cold_open_serves_props: production bootstrap, production read path
    // (record -> property_ref -> BlobRef -> blob get_bag), every record.
    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: root.clone(),
    })
    .expect("loaded store must cold-open through production bootstrap");
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .expect("route production partition");
    let reader = backend.txn_manager().begin(LOAD_TENANT);
    let mut hydrated_checksum = 0_u32;
    let mut expected_checksum = 0_u32;
    for (index, node) in fixture.nodes.iter().enumerate() {
        let id = index as u64 + 1;
        let record = read_node_with_store(routed.crud(), &reader, NodeId::new(id))
            .expect("read loaded node")
            .unwrap_or_else(|| panic!("loaded node {id} missing after cold open"));
        assert_eq!(record.label_id, node.label, "node {id} label");
        let blob_ref = BlobRef::decode(record.property_ref).unwrap_or_else(|| {
            panic!(
                "INV-M5.20: node {id} property_ref {:#x} is not an overflow ref — \
                 properties were dropped at materialization",
                record.property_ref
            )
        });
        let bag = routed
            .crud()
            .blob_store()
            .get_bag(LOAD_TENANT, blob_ref)
            .expect("served bag readable");
        let expected = fixture.expected_node_bag(id);
        assert_eq!(
            &*bag,
            &expected[..],
            "cold_open_serves_props: node {id} bag differs from input"
        );
        hydrated_checksum = crc32c::crc32c_append(hydrated_checksum, &bag);
        expected_checksum = crc32c::crc32c_append(expected_checksum, &expected);
    }
    for (index, _rel) in fixture.rels.iter().enumerate() {
        let id = index as u64 + 1;
        let record = read_rel_with_store(routed.crud(), &reader, RelId::new(id))
            .expect("read loaded relationship")
            .unwrap_or_else(|| panic!("loaded relationship {id} missing after cold open"));
        let blob_ref = BlobRef::decode(record.property_ref)
            .unwrap_or_else(|| panic!("INV-M5.20: relationship {id} lost its property payload"));
        let bag = routed
            .crud()
            .blob_store()
            .get_bag(LOAD_TENANT, blob_ref)
            .expect("served rel bag readable");
        let expected = fixture.expected_rel_bag(id);
        assert_eq!(&*bag, &expected[..], "relationship {id} bag differs");
        hydrated_checksum = crc32c::crc32c_append(hydrated_checksum, &bag);
        expected_checksum = crc32c::crc32c_append(expected_checksum, &expected);
    }
    assert_eq!(
        hydrated_checksum, expected_checksum,
        "full-store hydrated-view checksum differs from the input-derived checksum"
    );
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);

    // RED-on-revert (CI lane): both V-3 regression classes must redden
    // THIS gate. Skip when this process IS an armed child.
    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under(
            "inv_m5_20_loaded_store_disk_complete_and_cold_open_serves_props",
            EMPTY_PROPS_ENV,
        );
        assert_red_under(
            "inv_m5_20_loaded_store_disk_complete_and_cold_open_serves_props",
            EMPTY_TEL_ENV,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 2 — INV-M5.17 hardened: disk-level loader-vs-incremental
// ─────────────────────────────────────────────────────────────────────

/// INV-M5.17 (hardened): the loader's STORE_PROPS/STORE_TEL EXTENT BYTES
/// and on-disk record `property_ref`/`out_tel_ref`/`in_tel_ref` fields are
/// compared against an independent PRODUCTION incremental ingest of the
/// same logical content. The loader side never touches bootstrap or any
/// rebuild — a scan-based oracle is tautological here (the #780 in-RAM
/// rebuild satisfies scans from rel records alone) and is deliberately
/// NOT the verdict. RED-on-revert: `ARCGRAPH_M5_SHIP_EMPTY_TEL` ships
/// empty TEL extents — the armed child FIRST proves post-rebuild scans
/// still pass (the tautology exhibit), THEN this differential goes red.
#[test]
fn inv_m5_17_disk_differential_loader_vs_incremental() {
    let _serial = serialize_gate();
    // No oversized bag: the differential resolves every property_ref
    // against STORE_PROPS extent bytes (chained bags are covered at the
    // served terminus by INV-M5.20/.12).
    let fixture = served_fixture(false);
    let (_dir, root) = load_fixture(&fixture);
    let generation = root.join(FINAL_GENERATION);
    let disk = decode_disk_store(&generation);

    // Independent oracle: the production incremental write path over the
    // same logical content, ingested in canonical order so dense ids
    // coincide with the loader's assignment.
    let manager = TxnManager::new();
    let store = CrudStore::new();
    let mut txn = manager.begin(LOAD_TENANT);
    let mut incremental_nodes = Vec::new();
    for node in &fixture.nodes {
        let id = create_node(
            &store,
            &mut txn,
            LOAD_TENANT,
            LabelId::new(node.label),
            &PropertyData::Blob(canonical_property_bag(node.float_bits, &node.opaque)),
        )
        .expect("incremental node ingest");
        incremental_nodes.push(id);
    }
    for rel in &fixture.rels {
        let source = incremental_nodes[fixture.node_id(&rel.source) as usize - 1];
        let target = incremental_nodes[fixture.node_id(&rel.target) as usize - 1];
        create_rel(
            &store,
            &mut txn,
            LOAD_TENANT,
            source,
            target,
            TypeId::new(rel.type_id),
            &PropertyData::Blob(canonical_property_bag(rel.float_bits, &rel.opaque)),
        )
        .expect("incremental relationship ingest");
    }
    commit(txn, &store).expect("incremental commit");
    let oracle = manager.begin(LOAD_TENANT);

    // The incremental leg's ids must coincide (both sides assign densely
    // in canonical order); a drift here is a differential-setup bug.
    assert_eq!(
        incremental_nodes
            .iter()
            .map(|id| id.raw())
            .collect::<Vec<_>>(),
        (1..=fixture.nodes.len() as u64).collect::<Vec<_>>(),
        "incremental ingest ids drifted from the canonical dense assignment"
    );

    // (1) Record census + property payloads: what the incremental path
    // persists (read back through ITS production store) must equal what
    // the loader put in the extents, per id, byte-exact.
    assert_eq!(
        disk.nodes.len(),
        fixture.nodes.len(),
        "loader disk node census"
    );
    assert_eq!(
        disk.rels.len(),
        fixture.rels.len(),
        "loader disk rel census"
    );
    for (id, record) in &disk.nodes {
        let incremental = read_node_with_store(&store, &oracle, NodeId::new(*id))
            .expect("incremental node read")
            .expect("incremental node exists");
        assert_eq!(record.label_id, incremental.label_id, "node {id} label");
        let incremental_bag = store
            .blob_store()
            .get_bag(
                LOAD_TENANT,
                BlobRef::decode(incremental.property_ref).expect("incremental overflow ref"),
            )
            .expect("incremental bag");
        let disk_bag = disk.disk_bag(record.property_ref);
        assert_eq!(
            disk_bag, &*incremental_bag,
            "INV-M5.17: node {id} STORE_PROPS extent bytes differ from the \
             incremental-ingest persistence of the same logical content"
        );
    }
    for (id, record) in &disk.rels {
        let incremental = read_rel_with_store(&store, &oracle, RelId::new(*id))
            .expect("incremental rel read")
            .expect("incremental rel exists");
        assert_eq!(record.type_id, incremental.type_id, "rel {id} type");
        assert_eq!(record.src_id, incremental.src_id, "rel {id} source");
        assert_eq!(record.dst_id, incremental.dst_id, "rel {id} target");
        let incremental_bag = store
            .blob_store()
            .get_bag(
                LOAD_TENANT,
                BlobRef::decode(incremental.property_ref).expect("incremental rel overflow ref"),
            )
            .expect("incremental rel bag");
        let disk_bag = disk.disk_bag(record.property_ref);
        assert_eq!(disk_bag, &*incremental_bag, "rel {id} props differ on disk");
    }

    // Tautology exhibit (armed-child path): under SHIP_EMPTY_TEL the
    // POST-REBUILD scans of the loaded store still pass — which is exactly
    // why they must not be the verdict. Proven before the disk compare so
    // the armed child's failure is attributable to the DISK differential.
    #[cfg(feature = "fault-injection")]
    if armed(EMPTY_TEL_ENV) {
        let (backend, _guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
            data_dir: root.clone(),
        })
        .expect("armed child cold open");
        let routed = backend
            .router()
            .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
            .unwrap();
        let served = backend.txn_manager().begin(LOAD_TENANT);
        let (expected_out, _) = fixture.expected_adjacency();
        for ((owner, type_id), entries) in &expected_out {
            let scanned: Vec<u64> = scan_out(
                routed.crud(),
                &served,
                NodeId::new(*owner),
                Some(TypeId::new(*type_id)),
            )
            .map(|entry| entry.rel_id)
            .collect();
            assert_eq!(
                scanned.len(),
                entries.len(),
                "tautology exhibit setup drifted: the #780 rebuild should satisfy scans"
            );
        }
        eprintln!(
            "INV-M5.17 anti-tautology exhibit: post-rebuild scans PASSED over \
             an empty-STORE_TEL ship; only the disk differential below can catch it"
        );
        served.abort();
    }

    // (2) TEL: the loader's on-disk chains (walked from the records' own
    // on-disk refs through raw STORE_TEL bytes) must equal the incremental
    // store's committed adjacency, both directions, per (owner, type).
    let disk_out = disk.disk_adjacency(|record| record.out_tel_ref);
    let disk_in = disk.disk_adjacency(|record| record.in_tel_ref);
    let mut incremental_out: AdjacencyMap = BTreeMap::new();
    let mut incremental_in: AdjacencyMap = BTreeMap::new();
    let types: std::collections::BTreeSet<u32> =
        fixture.rels.iter().map(|rel| rel.type_id).collect();
    for id in 1..=fixture.nodes.len() as u64 {
        for type_id in &types {
            let out: Vec<(u64, u64)> = scan_out(
                &store,
                &oracle,
                NodeId::new(id),
                Some(TypeId::new(*type_id)),
            )
            .map(|entry| (entry.dst_id, entry.rel_id))
            .collect();
            if !out.is_empty() {
                // Set-membership canonicalization (placement is pinned
                // unsorted in `walk_tel_chain`, not here).
                let mut sorted = out;
                sorted.sort_unstable();
                incremental_out.insert((id, *type_id), sorted);
            }
            let inn: Vec<(u64, u64)> = scan_in(
                &store,
                &oracle,
                NodeId::new(id),
                Some(TypeId::new(*type_id)),
            )
            .expect("incremental reverse scan")
            .into_iter()
            .map(|entry| (entry.dst_id, entry.rel_id))
            .collect();
            if !inn.is_empty() {
                // Set-membership canonicalization (placement is pinned
                // unsorted in `walk_tel_chain`, not here).
                let mut sorted = inn;
                sorted.sort_unstable();
                incremental_in.insert((id, *type_id), sorted);
            }
        }
    }
    assert_eq!(
        disk_out, incremental_out,
        "INV-M5.17: loader STORE_TEL forward extent contents differ from the \
         incremental store's committed adjacency (empty-TEL ships red here \
         even though post-rebuild scans stay green)"
    );
    assert_eq!(
        disk_in, incremental_in,
        "INV-M5.17: loader STORE_TEL reverse extent contents differ from the \
         incremental store's committed adjacency"
    );
    oracle.abort();

    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under(
            "inv_m5_17_disk_differential_loader_vs_incremental",
            EMPTY_TEL_ENV,
        );
        assert_red_under(
            "inv_m5_17_disk_differential_loader_vs_incremental",
            EMPTY_PROPS_ENV,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 4 — INV-M5.12: ULP-adversarial fidelity, SERVED-store terminus
// ─────────────────────────────────────────────────────────────────────

/// INV-M5.12 (terminus fix, salvaged from closed PR #1504 per amendment
/// §9 + §3.3): the #1442 ULP-adversarial corpus goes input → `arcgraph
/// load` → production cold open → production property read, and every
/// float bit pattern (including NaN payloads and max-finite) and every
/// opaque embedder payload (including a DEC-4-chained oversized one)
/// returns bit-/byte-exact FROM THE SERVED STORE. The superseded #1504
/// oracle terminated at the parser/sidecars; those intermediates no
/// longer exist — this terminus is the served generation itself.
/// RED-on-revert: `ARCGRAPH_M5_LOSSY_FLOAT_BITS` (a 1-ULP-lossy
/// materialization) or `ARCGRAPH_M5_SHIP_EMPTY_PROPS`.
#[test]
fn inv_m5_12_ulp_fidelity_oracle_terminates_at_served_store() {
    let _serial = serialize_gate();
    let fixture = served_fixture(true);
    let (_dir, root) = load_fixture(&fixture);

    let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::Durable {
        data_dir: root.clone(),
    })
    .expect("ULP corpus store must cold-open");
    let routed = backend
        .router()
        .route(TenantId::DEFAULT, arcgraph_core::PartitionId::ZERO)
        .unwrap();
    let reader = backend.txn_manager().begin(LOAD_TENANT);
    for (index, node) in fixture.nodes.iter().enumerate() {
        let id = index as u64 + 1;
        let record = read_node_with_store(routed.crud(), &reader, NodeId::new(id))
            .expect("read ULP corpus node")
            .expect("ULP corpus node exists");
        let blob_ref = BlobRef::decode(record.property_ref)
            .unwrap_or_else(|| panic!("ULP node {id} lost its property payload"));
        let bag = routed
            .crud()
            .blob_store()
            .get_bag(LOAD_TENANT, blob_ref)
            .expect("served ULP bag readable");
        assert!(bag.len() >= 8, "served bag too short for float bits");
        let served_bits = u64::from_le_bytes(bag[..8].try_into().unwrap());
        if served_bits != node.float_bits {
            panic!(
                "INV-M5.12: node {id} ({}) float bits differ at the SERVED terminus: \
                 input {:#018x}, served {:#018x} (ULP-lossy materialization)",
                String::from_utf8_lossy(&node.external),
                node.float_bits,
                served_bits,
            );
        }
        if &bag[8..] != node.opaque.as_slice() {
            panic!(
                "INV-M5.12: node {id} opaque embedder payload differs at the SERVED \
                 terminus ({} input bytes, {} served bytes)",
                node.opaque.len(),
                bag.len() - 8,
            );
        }
    }
    reader.abort();
    drop(routed);
    drop(backend);
    drop(guard);

    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under(
            "inv_m5_12_ulp_fidelity_oracle_terminates_at_served_store",
            LOSSY_FLOAT_ENV,
        );
        assert_red_under(
            "inv_m5_12_ulp_fidelity_oracle_terminates_at_served_store",
            EMPTY_PROPS_ENV,
        );
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 3 — #1519 tel_disk_size_is_dense (the regression pin)
// ─────────────────────────────────────────────────────────────────────

/// #1519 THE regression pin: a low-degree, multi-type fixture (avg
/// out-degree well under [`arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES`],
/// every block a packing candidate) produces STORE_TEL bytes within a
/// SMALL constant factor of the dense minimum — NOT the ~200x page-per-
/// block blowup the M5-D3 100M rung measured. Measured via ACTUAL disk
/// pages touched (`read_store_pages`, extent-ledger-derived, no
/// bootstrap, no rebuild — same anti-tautology discipline as INV-M5.17),
/// not a post-load scan.
///
/// RED-on-revert: `ARCGRAPH_M5_TEL_PAGE_PER_BLOCK` forces every block down
/// the pre-#1519 dedicated-page path — the SAME layout the charter calls
/// out as the P0 (~200x blowup at avg out-degree 5 / 7 types).
#[test]
fn tel_disk_size_is_dense() {
    let _serial = serialize_gate();
    // 200 owners x out-degree 4 across 7 types, single direction of
    // interest is BOTH (loader always streams out+in) — comfortably in
    // the "avg out-degree 5 / 7 types" STOP-report regime, every block
    // well under the supernode threshold.
    const N_OWNERS: u32 = 200;
    const DEGREE: u32 = 4;
    const N_TYPES: u32 = 7;
    let fixture = dense_fixture(N_OWNERS, DEGREE, N_TYPES);
    let (_dir, root) = load_fixture(&fixture);
    let generation = root.join(FINAL_GENERATION);

    let tel_pages = read_store_pages(&generation, STORE_TEL);
    let tel_bytes = (tel_pages.len() * PAGE_SIZE) as u64;

    // Dense minimum: every (owner, type) block pays exactly its header +
    // entries, both directions, no page waste at all.
    let total_rels = fixture.rels.len() as u64;
    let out_groups: std::collections::BTreeSet<(u64, u32)> = fixture
        .rels
        .iter()
        .map(|rel| (fixture.node_id(&rel.source), rel.type_id))
        .collect();
    let in_groups: std::collections::BTreeSet<(u64, u32)> = fixture
        .rels
        .iter()
        .map(|rel| (fixture.node_id(&rel.target), rel.type_id))
        .collect();
    let groups = (out_groups.len() + in_groups.len()) as u64;
    let dense_min_bytes = groups * TEL_BLOCK_HEADER as u64 + total_rels * 2 * TEL_ENTRY as u64;

    // Anti-vacuousness: the fixture must actually exercise multiple
    // distinct blocks (otherwise "within a small factor of dense_min" is
    // trivially true of a single page).
    assert!(
        groups > 20,
        "fixture must produce enough distinct (owner,type) blocks to be a \
         meaningful densify signal, got {groups}"
    );

    const SMALL_FACTOR: u64 = 4;
    assert!(
        tel_bytes <= dense_min_bytes * SMALL_FACTOR,
        "tel_disk_size_is_dense: STORE_TEL used {tel_bytes} bytes ({} pages) \
         against a dense minimum of {dense_min_bytes} bytes ({groups} blocks, \
         {total_rels} rels x2 directions) — factor {:.1}x exceeds the \
         SMALL_FACTOR={SMALL_FACTOR}x bound (#1519 densify regression)",
        tel_pages.len(),
        tel_bytes as f64 / dense_min_bytes as f64,
    );
    eprintln!(
        "tel_disk_size_is_dense: {tel_bytes} B ({} pages) vs dense_min {dense_min_bytes} B \
         ({groups} blocks) — factor {:.2}x",
        tel_pages.len(),
        tel_bytes as f64 / dense_min_bytes as f64
    );

    #[cfg(feature = "fault-injection")]
    if !any_seam_armed() {
        assert_red_under("tel_disk_size_is_dense", TEL_PAGE_PER_BLOCK_ENV);
    }
}

// ─────────────────────────────────────────────────────────────────────
// Gate 6 — #1519 supernode threshold boundary
// ─────────────────────────────────────────────────────────────────────

/// #1519 gate 6: a fixture straddling
/// [`arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES`]
/// (126) — one owner with a block just BELOW the threshold (packs) and
/// one owner with a block just ABOVE it (chains via its own dedicated
/// page) — exercises both paths + the transition through the SAME
/// production reader (`walk_tel_chain`/`resolve_tel_block`).
#[test]
fn supernode_threshold_boundary_exercises_both_paths() {
    let _serial = serialize_gate();
    let below = arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES - 1;
    let above = arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES + 1;

    let mut nodes = vec![
        FixtureNode {
            external: b"owner-below".to_vec(),
            label: 1,
            float_bits: 0,
            opaque: Vec::new(),
        },
        FixtureNode {
            external: b"owner-above".to_vec(),
            label: 1,
            float_bits: 0,
            opaque: Vec::new(),
        },
    ];
    let max_targets = above.max(below);
    for index in 0..max_targets {
        nodes.push(FixtureNode {
            external: format!("target-{index:05}").into_bytes(),
            label: 2,
            float_bits: 0,
            opaque: Vec::new(),
        });
    }
    nodes.sort_by(|left, right| left.external.cmp(&right.external));

    let mut rels = Vec::new();
    let mut rel_index = 0_u64;
    for index in 0..below {
        rels.push(FixtureRel {
            external: format!("below-{rel_index:08}").into_bytes(),
            source: b"owner-below".to_vec(),
            target: format!("target-{index:05}").into_bytes(),
            type_id: 1,
            float_bits: 0,
            opaque: Vec::new(),
        });
        rel_index += 1;
    }
    for index in 0..above {
        rels.push(FixtureRel {
            external: format!("above-{rel_index:08}").into_bytes(),
            source: b"owner-above".to_vec(),
            target: format!("target-{index:05}").into_bytes(),
            type_id: 1,
            float_bits: 0,
            opaque: Vec::new(),
        });
        rel_index += 1;
    }
    rels.sort_by(|left, right| left.external.cmp(&right.external));
    let fixture = Fixture { nodes, rels };

    let (_dir, root) = load_fixture(&fixture);
    let generation = root.join(FINAL_GENERATION);
    let disk = decode_disk_store(&generation);

    let below_id = disk
        .nodes
        .iter()
        .find(|(_, record)| record.label_id == 1)
        .map(|(id, _)| *id);
    // Resolve both owners by walking node records in external-id order
    // (canonical order == dense id order per the fixture convention).
    let below_owner = fixture.node_id(b"owner-below");
    let above_owner = fixture.node_id(b"owner-above");
    let _ = below_id;

    let below_record = disk.nodes.get(&below_owner).expect("below-owner node");
    let above_record = disk.nodes.get(&above_owner).expect("above-owner node");
    assert_ne!(
        below_record.out_tel_ref, 0,
        "below-threshold owner has a chain"
    );
    assert_ne!(
        above_record.out_tel_ref, 0,
        "above-threshold owner has a chain"
    );

    // Below-threshold: the block's page must be a PACKED page (`flags =
    // TEL_PAGE_FLAG_PACKED`) — it shares its page with other blocks.
    let (below_page, _below_slot) = decode_ref(below_record.out_tel_ref);
    let below_page_bytes = disk
        .tel_pages
        .get(&below_page)
        .expect("below-threshold TEL page mapped");
    let below_header = PageHeader::from_bytes(
        below_page_bytes[..PageHeader::SIZE]
            .try_into()
            .expect("header slice"),
    )
    .unwrap();
    assert_eq!(
        below_header.flags, TEL_FLAG_PACKED,
        "below-threshold block ({below} entries) must land on a densified \
         packed page"
    );

    // Above-threshold: the block's page must be the pre-#1519 dedicated
    // chain shape (`flags = 0`) — a supernode gets its own page.
    let (above_page, above_slot) = decode_ref(above_record.out_tel_ref);
    let above_page_bytes = disk
        .tel_pages
        .get(&above_page)
        .expect("above-threshold TEL page mapped");
    let above_header = PageHeader::from_bytes(
        above_page_bytes[..PageHeader::SIZE]
            .try_into()
            .expect("header slice"),
    )
    .unwrap();
    assert_eq!(
        above_header.flags, 0,
        "above-threshold block ({above} entries) must land on a dedicated \
         supernode/chain page"
    );
    assert_eq!(
        above_slot, 0,
        "supernode/chain pages hold exactly one block at slot 0"
    );

    // Both paths must decode through the SAME production reader to the
    // exact entry set, in committed order.
    let below_entries = disk.walk_tel_chain(below_owner, below_record.out_tel_ref);
    assert_eq!(below_entries.len(), below as usize);
    let above_entries = disk.walk_tel_chain(above_owner, above_record.out_tel_ref);
    assert_eq!(above_entries.len(), above as usize);
}

// ─────────────────────────────────────────────────────────────────────
// #1519 BLOCK_FIX FIX 2 — chained (multi-block) supernode coverage
// ─────────────────────────────────────────────────────────────────────

/// #1519 BLOCK_FIX FIX 2: the report's "supernode pages byte-identical to
/// pre-#1519" claim is FALSE for a CHAINED supernode (only the
/// chain-TAIL, `prev = NO_PREV`, is byte-identical) — a chained flags=0
/// page's `prev_block_ptr` now carries an [`encode_tel_ref`]-encoded
/// value instead of a bare page id, changing that field's bytes (and
/// therefore the page's body CRC) even though the page SHAPE (flags = 0,
/// one block filling the body) is unchanged. No fixture before this test
/// reached a chained supernode: the boundary test above straddles
/// [`arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES`]
/// (126 ± 1) with single blocks, and `served_fixture`'s 300-edge hub
/// flushes a 253-entry flags=0 head chained to a 47-entry flags=PACKED
/// tail (47 < 126) — never two CONSECUTIVE flags=0 pages.
///
/// This fixture uses one owner with
/// `TEL_SUPERNODE_THRESHOLD_ENTRIES * 2 + 127` (= 379) same-type edges —
/// comfortably over 379, the charter's minimum for a real multi-block
/// supernode CHAIN: the first (oldest) block flushes at exactly
/// [`arcgraph_storage::m4_migration::FRESH_TEL_ENTRIES_PER_PAGE`] (253)
/// entries (>= threshold => flags=0, dedicated chain page), and the
/// remaining 126 entries flush at [`Self::finish_tel_chain`] as the
/// second (newest) block — also >= threshold (126 == threshold) => a
/// SECOND flags=0 dedicated chain page, linked to the first via
/// `prev_block_ptr`. Both pages must decode CORRECTLY through the
/// PRODUCTION chain walker (`prev_block_ptr` traversal, same
/// `walk_tel_chain`/`resolve_tel_block` reader every other gate uses),
/// returning every entry in committed order — the read-path contract
/// FIX 1's format-epoch discriminator exists to protect.
#[test]
fn chained_supernode_multi_block_decodes_through_chain_walker() {
    let _serial = serialize_gate();
    let threshold = arcgraph_storage::m4_migration::TEL_SUPERNODE_THRESHOLD_ENTRIES;
    let per_page = arcgraph_storage::m4_migration::FRESH_TEL_ENTRIES_PER_PAGE as u32;
    // Two full-size chain blocks: the first flushes mid-stream at exactly
    // `per_page` entries (block-full flush); the remainder flushes at
    // `finish_tel_chain` and must itself still be >= threshold so BOTH
    // blocks are dedicated (flags=0) chain pages, not a packed tail.
    let total_edges = per_page + threshold;
    assert!(
        total_edges >= 379,
        "fixture must meet the charter's >= 379 same-type-edge minimum for a \
         real multi-block supernode chain, got {total_edges}"
    );
    assert!(
        total_edges - per_page >= threshold,
        "the chain-tail block must itself be >= the supernode threshold \
         (both blocks flags=0) — got tail size {}",
        total_edges - per_page
    );

    let mut nodes = vec![FixtureNode {
        external: b"chained-hub".to_vec(),
        label: 1,
        float_bits: 0,
        opaque: Vec::new(),
    }];
    for index in 0..total_edges {
        nodes.push(FixtureNode {
            external: format!("chained-target-{index:05}").into_bytes(),
            label: 2,
            float_bits: 0,
            opaque: Vec::new(),
        });
    }
    nodes.sort_by(|left, right| left.external.cmp(&right.external));

    let mut rels = Vec::new();
    for index in 0..total_edges {
        rels.push(FixtureRel {
            external: format!("chained-rel-{index:08}").into_bytes(),
            source: b"chained-hub".to_vec(),
            target: format!("chained-target-{index:05}").into_bytes(),
            type_id: 1,
            float_bits: 0,
            opaque: Vec::new(),
        });
    }
    rels.sort_by(|left, right| left.external.cmp(&right.external));
    let fixture = Fixture { nodes, rels };

    let (_dir, root) = load_fixture(&fixture);
    let generation = root.join(FINAL_GENERATION);
    let disk = decode_disk_store(&generation);

    let hub = fixture.node_id(b"chained-hub");
    let hub_record = disk.nodes.get(&hub).expect("hub node");
    assert_ne!(hub_record.out_tel_ref, 0, "hub owner has a chain");

    // Walk page-by-page (bypassing `walk_tel_chain` for this structural
    // assertion) to confirm BOTH blocks in the chain are dedicated
    // flags=0 pages, and that the head's `prev_block_ptr` is a non-NO_PREV
    // ENCODED ref naming the tail page — i.e. a genuine multi-page chain,
    // not a single supernode page.
    let mut seen_pages = Vec::new();
    let mut next = hub_record.out_tel_ref;
    while next != 0 {
        let (page_no, slot) = decode_ref(next);
        assert_eq!(
            slot, 0,
            "dedicated chain pages hold exactly one block at slot 0"
        );
        let page_bytes = disk
            .tel_pages
            .get(&page_no)
            .unwrap_or_else(|| panic!("chain page {page_no} mapped"));
        let header = PageHeader::from_bytes(
            page_bytes[..PageHeader::SIZE]
                .try_into()
                .expect("header slice"),
        )
        .unwrap();
        assert_eq!(
            header.flags, 0,
            "every block in a >= threshold chain must be a dedicated \
             flags=0 supernode/chain page, got flags={} at page {page_no}",
            header.flags
        );
        seen_pages.push(page_no);
        let body = &page_bytes[PageHeader::SIZE..];
        let block_size = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
        let prev = u64::from_le_bytes(body[16..24].try_into().unwrap());
        next = if prev == TEL_NO_PREV { 0 } else { prev };
        // Sanity: the block fills (or nearly fills) the page body — the
        // pre-#1519 supernode/chain page shape.
        assert!(
            block_size <= PAGE_SIZE - PageHeader::SIZE,
            "block must fit within one page body"
        );
    }
    assert_eq!(
        seen_pages.len(),
        2,
        "expected exactly 2 chained dedicated pages (head + tail), got {}: {seen_pages:?}",
        seen_pages.len()
    );
    assert_ne!(
        seen_pages[0], seen_pages[1],
        "chain head and tail must be DISTINCT physical pages"
    );

    // The production chain walker must decode every entry, in committed
    // order, across the multi-page chain.
    let entries = disk.walk_tel_chain(hub, hub_record.out_tel_ref);
    assert_eq!(
        entries.len(),
        total_edges as usize,
        "chain walker must recover every entry across the chained \
         multi-block supernode"
    );
    let mut expected_targets: Vec<u64> = (0..total_edges)
        .map(|index| fixture.node_id(format!("chained-target-{index:05}").as_bytes()))
        .collect();
    expected_targets.sort_unstable();
    let mut got_targets: Vec<u64> = entries.iter().map(|(_, neighbor, _)| *neighbor).collect();
    got_targets.sort_unstable();
    assert_eq!(
        got_targets, expected_targets,
        "chain walker must recover the exact neighbor set through both \
         chained pages"
    );
}

// ─────────────────────────────────────────────────────────────────────
// CLI dress rehearsal, pinned — `arcgraph check` over a loaded store
// ─────────────────────────────────────────────────────────────────────

/// The operator-visible INV-M5.20 surface: `arcgraph check --data` over
/// a committed loaded generation COLD-OPENS through production bootstrap
/// and reports a served-property census for the loaded tenant (hydrated
/// through the production read path — bounded to the first 64 ids per
/// record class).
#[test]
fn arcgraph_check_cold_opens_and_serves_loaded_props() {
    let _serial = serialize_gate();
    let fixture = served_fixture(true);
    let (_dir, root) = load_fixture(&fixture);
    let output = Command::new(env!("CARGO_BIN_EXE_arcgraph"))
        .args(["check", "--data"])
        .arg(&root)
        .output()
        .expect("spawn arcgraph check over the loaded store");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "arcgraph check over a loaded store failed: {:?}\nstdout: {stdout}\nstderr: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );
    let sampled_nodes = fixture.nodes.len().min(64);
    let sampled_rels = fixture.rels.len().min(64);
    let expected = format!(
        "tenant={} sampled_nodes={sampled_nodes} sampled_rels={sampled_rels} hydrated_bags={}",
        LOAD_TENANT.raw(),
        sampled_nodes + sampled_rels,
    );
    assert!(
        stdout.contains(&expected),
        "check did not serve the loaded tenant's properties; expected \
         {expected:?} in:\n{stdout}"
    );
}

/// #1519 BLOCK_FIX FIX 1 (SILENT-M6-CORRUPTION), the end-to-end gate: a v6
/// generation whose MANIFEST names the PRE-#1519 `tel_ref_format`
/// (`TEL_REF_FORMAT_BARE_PAGE_ID` — the exact stamp a D2/D3-built
/// generation from before #1519 landed would carry, since #1519 changed
/// the STORE_TEL ref encoding without bumping the coarse `VERSION`
/// integer) must be REFUSED by the production attach path
/// (`arcgraph check --data`, which cold-opens through the same
/// `bootstrap_storage_backend` durable path `arcgraph serve` uses) —
/// never silently opened and served. RED-on-revert: this is exactly the
/// differential the TIER-1 gate's skeptic used — a bare page id decoded
/// through the new `decode_tel_ref` inverse is a plausible-looking but
/// WRONG `(page_no, slot)` pair (see
/// `arcgraph_storage::data_dir_version::tests::decode_tel_ref_misdecodes_a_bare_pre_1519_page_id_the_corruption_this_guards_against`);
/// removing the `check_tel_ref_format` call from `bootstrap.rs` (or
/// reverting the MANIFEST default to the current constant) makes this
/// test's refusal disappear and the stale generation opens successfully.
#[test]
fn stale_tel_ref_encoding_generation_is_refused_not_silently_served() {
    let _serial = serialize_gate();
    let fixture = dense_fixture(50, 3, 4);
    let (_dir, root) = load_fixture(&fixture);
    let generation = root.join(FINAL_GENERATION);
    let manifest_path = generation.join("MANIFEST");

    // Simulate a pre-#1519 generation: rewrite the just-loaded (current)
    // MANIFEST's `tel_ref_format` to the pre-#1519 bare-page-id sentinel —
    // the exact stamp a D2/D3-built generation from before #1519 landed
    // would carry (its binary never wrote this field with the CURRENT
    // value; a fully-absent field resolves to the same sentinel via
    // `#[serde(default)]`, exercised at the unit level in
    // `arcgraph-storage`'s `data_dir_version` tests). The on-disk STORE_TEL
    // bytes are UNCHANGED — this test isolates the discriminator/attach
    // guard, not the packer (already covered above).
    let mut manifest: serde_json::Value =
        serde_json::from_slice(&fs::read(&manifest_path).expect("read MANIFEST")).expect("parse");
    assert_eq!(
        manifest["tel_ref_format"],
        serde_json::json!(arcgraph_storage::TEL_REF_FORMAT_PAGE_SLOT_V1),
        "precondition: a freshly loaded generation must carry the CURRENT \
         tel_ref_format before this test corrupts it"
    );
    manifest["tel_ref_format"] = serde_json::json!(arcgraph_storage::TEL_REF_FORMAT_BARE_PAGE_ID);
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).expect("reserialize MANIFEST"),
    )
    .expect("write stale MANIFEST");

    let output = Command::new(env!("CARGO_BIN_EXE_arcgraph"))
        .args(["check", "--data"])
        .arg(&root)
        .output()
        .expect("spawn arcgraph check over the stale-encoding store");
    assert!(
        !output.status.success(),
        "arcgraph check MUST refuse a stale STORE_TEL ref encoding rather \
         than silently serving it; status={:?} stdout={} stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("tel_ref_format") || stderr.contains("STORE_TEL ref encoding"),
        "refusal must name the stale STORE_TEL ref-encoding class, not an \
         unrelated failure; stderr:\n{stderr}"
    );
    assert!(
        stderr.contains("1519"),
        "refusal must reference #1519 (the format-change issue) for \
         operator diagnosis; stderr:\n{stderr}"
    );
}
