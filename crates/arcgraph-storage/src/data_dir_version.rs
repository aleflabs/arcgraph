//! On-disk data-dir version stamp + boot-time compatibility guard
//! (SVC-2, #1302 — upgrade-safety).
//!
//! # Why (issue #1302, front-02 + front-10 grade-A gate)
//!
//! The WAL is version-guarded (`wal::segment::SUPPORTED_WAL_FORMAT_VERSIONS`
//! → [`arcgraph_core::ArcGraphError::WalFormatMismatch`] on an unknown
//! segment version) and the catalog root page is version-guarded
//! (`catalog::page::CATALOG_PAGE_VERSION` → `CatalogPageError::UnsupportedVersion`).
//! But the *primary page store* (`pages.db`) and the **data-dir as a
//! whole** carried no version stamp. So an operator who upgrades by
//! swapping the binary across a page-format change — a common mistake —
//! got *undefined behaviour / a misparse* instead of a loud "incompatible
//! on-disk version — run `arcgraph migrate`". Upgrade safety rested
//! entirely on the operator following `docs/operations/upgrade.md`, with no
//! runtime guard forcing it.
//!
//! This module closes that gap. At durable data-dir init the bootstrap
//! writes a tiny `VERSION` file (magic + a `DATA_DIR_FORMAT_VERSION` u16);
//! on every subsequent open it reads that file and **refuses to proceed**
//! before any page is parsed if the on-disk version is one this binary does
//! not support — with a clear, actionable error. Fail LOUD + fail-CLOSED.
//!
//! # Precedent mirrored (do not invent — evidence-over-intuition rule)
//!
//! The stamp shape follows the two existing on-disk version guards verbatim:
//! - **WAL:** `wal::segment` — `WAL_SEGMENT_MAGIC` (4 B) + `format_version`
//!   (u16 LE) + reserved (2 B) at the head of every segment; decode checks
//!   magic → version-in-supported-set → reserved-zero, returning a typed
//!   error listing the supported versions.
//! - **Catalog page:** `catalog::page` — `CATALOG_PAGE_MAGIC` (8 B) +
//!   `CATALOG_PAGE_VERSION` (u16 LE) at the head of the catalog root page;
//!   decode returns `CatalogPageError::UnsupportedVersion(found)`.
//!
//! The `VERSION` file uses the **same layout convention** as the WAL
//! segment header (little-endian integers, a fixed-size head): an 8-byte
//! ASCII magic, a u16 LE `format_version`, and 2 reserved bytes that MUST
//! be zero (a future version may reclaim them; today they fail closed).
//!
//! # On-disk layout
//!
//! `<data_dir>/VERSION`, a 12-byte fixed-size file:
//!
//! ```text
//! offset  field           size  notes
//! 0       magic           8     b"ARCGDDV1"
//! 8       format_version  2     u16 little-endian
//! 10      reserved        2     MUST be 0 (fail-closed at v1)
//! ```
//!
//! # Fresh vs. existing vs. legacy policy (#1302 §3 — the load-bearing call)
//!
//! [`check_or_stamp_data_dir`] takes `has_data`: whether the data dir
//! already holds a page store / WAL (the bootstrap's existing "is this a
//! restart?" signal — `pages.db` exists OR the WAL subdir is non-empty).
//! It also takes `adopt_legacy`: an EXPLICIT operator opt-in (the
//! `arcgraph serve --adopt-legacy-datadir` flag) asserting "this existing
//! dir IS the current format — stamp it". Cases:
//!
//! 1. **`VERSION` present** → read + check. Version in
//!    [`SUPPORTED_DATA_DIR_VERSIONS`] ⟹ proceed (a clean no-op re-open).
//!    Version NOT supported ⟹ [`DataDirVersionError::Incompatible`] —
//!    the loud, actionable error. `adopt_legacy` does NOT rescue this: an
//!    incompatible on-disk version can never be adopted (the format really
//!    differs); the operator needs a matching binary or a restore.
//! 2. **No `VERSION`, `has_data == false`** (a genuinely fresh dir) ⟹
//!    stamp the current version and proceed. This is first boot.
//! 3. **No `VERSION`, `has_data == true`** (a *legacy* / pre-stamp dir — a
//!    beta deployment created before this stamp existed):
//!    - `adopt_legacy == false` (default) ⟹
//!      [`DataDirVersionError::LegacyUnstamped`], NOT a silent proceed and
//!      NOT a silent brick. The error points at the REAL recovery path
//!      (`--adopt-legacy-datadir` / restore-from-backup) — NOT `arcgraph
//!      migrate`, which is the Neo4j-export IMPORT verb, not a version
//!      adopt (the #1345 R1 REQUIRED fix: pointing operators at `migrate`
//!      was a wrong/destructive dead-end that bricked beta→GA upgrades).
//!    - `adopt_legacy == true` (explicit operator opt-in) ⟹ stamp the
//!      current version and proceed. The operator is asserting the dir is
//!      the current format; we make that assertion durable + explicit
//!      rather than auto-stamping every unknown-provenance dir (which would
//!      defeat the guard). This is the ONLY path that writes a stamp onto a
//!      dir that already holds data. A recognized pre-M3 MANIFEST restores
//!      its exact v3/v4 stamp; a genuinely pre-stamp dir with no MANIFEST is
//!      stamped as chained v1 and migrated forward by the same boot. M3 v5,
//!      unknown, and malformed MANIFESTs remain refused.
//!
//! The default stays fail-CLOSED: an accidental binary-swap onto a legacy
//! dir (no `--adopt-legacy-datadir`) is still refused; only the explicit
//! opt-in adopts.
//!
//! # `has_data` scope (known edge — NIT-1, #1345 R1)
//!
//! `has_data` is derived by the bootstrap from `pages.db` + the WAL subdir
//! (the ADR-183 durable-substrate markers). It does NOT probe BM25 or
//! vector-arena subdirs. This is not reachable in normal operation: a
//! durable dir that has ingested anything has a non-empty WAL (every
//! commit — incl. the catalog bootstrap — writes a WAL segment), so
//! `has_data` is already true before any BM25/vector subdir exists. The
//! edge would only matter for a hand-constructed dir with a BM25/vector
//! subdir but an empty WAL and no `pages.db` — which the durable bootstrap
//! never produces. Documented as a known, not-normally-reachable edge
//! rather than widening the probe (which would couple this guard to every
//! future index subdir layout).
//!
//! # Budget (performance-budget discipline)
//!
//! The check is one 12-byte file read (existing dir) or one 12-byte
//! write+fsync (fresh dir or an explicit adopt), each once per process at
//! boot — a single `open` + `read`/`write` syscall pair over 12 bytes,
//! ≈ a few µs, utterly negligible against WAL recovery (O(WAL size)) that
//! follows it. It is NOT a hot path.

use std::fs;
use std::path::{Path, PathBuf};

use thiserror::Error;

/// Magic bytes at offset 0 of `<data_dir>/VERSION`. ASCII "ARCGDDV1"
/// (ARCGraph Data-Dir Version, format family 1). Lets a reader tell
/// "right file type, unknown version" from "wrong file entirely / not an
/// ArcGraph data dir". Mirrors `WAL_SEGMENT_MAGIC` / `CATALOG_PAGE_MAGIC`.
pub const DATA_DIR_VERSION_MAGIC: &[u8; 8] = b"ARCGDDV1";

/// On-disk data-dir format version this binary stamps into a fresh
/// `<data_dir>/VERSION` and requires on open.
///
/// Bump this when any breaking change to the data-dir's on-disk layout
/// lands that an older/newer binary cannot safely read — a page-format
/// change in `pages.db`, a new mandatory data-dir file, etc. Bumping it
/// makes a binary-swap across the change fail loud
/// ([`DataDirVersionError::Incompatible`]) instead of misparsing, exactly
/// as `CURRENT_WAL_FORMAT_VERSION` does for the WAL.
///
/// # Version ↔ design-doc numbering map (v2 M1/M2, ADR-230 / design §0.1)
///
/// | on-disk stamp | design `data_dir_version` | format |
/// |---|---|---|
/// | `1` | 2 (implicit) | pre-M1: DEC-4 always-chained property blobs |
/// | *(none)* | *(2 is never stamped on disk)* | — |
/// | `3` | 3 | M1: slotted small-blob packing, JSON payloads (`slotted-v1`) |
/// | `4` | 4 | M2: typed property blocks in the M1 slots (`typed-v1`) |
/// | `5` | 5 | M3: `CURRENT`-selected generation with delta-v9 WAL |
/// | `6` | 6 | M4: direct-addressed, extent-backed v6 generation with delta-v10 WAL |
///
/// The design treats un-stamped v1.0 stores as "implicit version 2";
/// this file's historical stamp for exactly that format is `1` (SVC-2
/// #1302 predates the design's numbering). Rather than re-meaning the
/// on-disk `1`, M1 stamps `3` — matching the design/ADR-230 counter
/// from M1 onward — and the value `2` never appears on disk. From M1
/// the `<data_dir>/MANIFEST` (`crate::manifest`, design §0.1) carries
/// the fine-grained per-substrate format strings; this `VERSION`
/// integer stays the coarse boot-refusal gate an OLD binary enforces
/// (an old binary knows nothing of the MANIFEST — it refuses on this
/// integer alone, which is what prevents the silent-empty-bag misread
/// of an M1 store by a pre-M1 binary, and equally the silent-JSON-
/// misparse of an M2 typed store by a pre-M2 binary: the M1 binary's
/// supported set is `[1, 3]` at merged HEAD `23228de0`, so a `4` dir
/// refuses there via [`DataDirVersionError::Incompatible`]).
pub const DATA_DIR_FORMAT_VERSION: u16 = 4;

/// The legacy pre-M1 stamp: DEC-4 chained-blob stores (v1.0 dirs, and
/// what `--adopt-legacy-datadir` truthfully stamps onto an unstamped
/// beta dir — the data IS this format; the same boot then runs the
/// M1 + M2 migrate-on-open sweeps to bring it to
/// [`DATA_DIR_FORMAT_VERSION`]).
pub const DATA_DIR_VERSION_CHAINED_V1: u16 = 1;

/// The v2 M1 stamp: slotted small-blob packing with JSON slot payloads
/// (`props_store_format: slotted-v1`). An M2 binary opens it via the
/// M2 migrate-on-open re-encode (design §M2.6, `data_dir_version`
/// 3 → 4).
pub const DATA_DIR_VERSION_SLOTTED_M1: u16 = 3;

/// The v2 M2 stamp: typed property blocks with the page-image v8 WAL.
pub const DATA_DIR_VERSION_TYPED_M2: u16 = 4;

/// The v2 M3 stamp: a `CURRENT`-selected generation with delta-v9 WAL.
pub const DATA_DIR_VERSION_DELTA_M3: u16 = 5;

/// The v2 M4 stamp: a `CURRENT`-selected direct-addressed v6 generation with
/// delta-v10 WAL.
pub const DATA_DIR_VERSION_DIRECT_M4: u16 = 6;

/// Data-dir versions this binary knows how to open. Mirrors
/// `SUPPORTED_WAL_FORMAT_VERSIONS`: a newer binary lists an older version
/// here *iff* it can still read that on-disk layout (explicit backward
/// compatibility, per-version). The M3 binary recognizes all four: `1`
/// opens via the M1-then-M2 migrate-on-open chain (chained blobs remain
/// readable — the DEC-4 read path is retained for large-payload
/// overflow anyway); `3` opens via the M2 migrate-on-open re-encode
/// (JSON slot payloads remain readable during the sweep — the mixed-
/// store read dispatch); `4` is the offline-upgrade source; `5` is selected
/// only through the explicit generation migration.
pub const SUPPORTED_DATA_DIR_VERSIONS: &[u16] = &[
    DATA_DIR_VERSION_CHAINED_V1,
    DATA_DIR_VERSION_SLOTTED_M1,
    DATA_DIR_FORMAT_VERSION,
    DATA_DIR_VERSION_DELTA_M3,
    DATA_DIR_VERSION_DIRECT_M4,
];

/// Name of the version-stamp file inside `<data_dir>`. Lives at the
/// data-dir root next to `LOCK` / `pages.db` (NOT inside `<data_dir>/wal`,
/// so it never perturbs the bootstrap's "is the WAL pre-existing?" restart
/// heuristic).
pub const VERSION_FILE: &str = "VERSION";

/// Fixed size of the encoded `VERSION` file: magic(8) + version(2) +
/// reserved(2).
const VERSION_FILE_LEN: usize = 12;

/// Faults surfaced by the data-dir version guard.
///
/// Fail-loud + fail-closed: every case that is not a clean, supported
/// re-open (or a legitimate fresh stamp) surfaces a typed error the
/// bootstrap propagates so the operator sees an actionable message rather
/// than a misparse downstream. Codec-local per
/// `docs/codec-error-translation.md`; the bootstrap boundary translates it
/// into `anyhow` with `.with_context(..)`.
///
/// `#[non_exhaustive]` under the code-quality policy — adding a variant is not a
/// breaking change.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DataDirVersionError {
    /// The on-disk `VERSION` file carries a `format_version` this binary
    /// does not support. THE core upgrade-safety guard: a binary-swap
    /// across a data-dir-format change fails here, loud, before any page is
    /// parsed — instead of misparsing `pages.db`. Lists the supported set
    /// (mirrors [`arcgraph_core::ArcGraphError::WalFormatMismatch`]).
    ///
    /// An incompatible version can NOT be adopted (the on-disk format really
    /// differs — `--adopt-legacy-datadir` only rescues an *unstamped legacy*
    /// dir, never a version mismatch). The remediation is a matching binary
    /// version or a restore from a compatible backup.
    #[error(
        "incompatible on-disk data-dir version {found} at {path}: this binary supports \
         v{supported:?}. Start the matching ArcGraph binary version for this data dir, or \
         restore from a backup taken with a compatible version. (An incompatible version \
         can NOT be adopted with `--adopt-legacy-datadir` — that only stamps an *unstamped* \
         legacy dir, not a version mismatch.) Refusing to open a data dir written by an \
         incompatible ArcGraph version rather than misparsing it (issue #1302, upgrade-safety)"
    )]
    Incompatible {
        /// Version stamped in the offending `VERSION` file.
        found: u16,
        /// Versions this binary knows how to open.
        supported: &'static [u16],
        /// Path of the `VERSION` file, for operator diagnostics.
        path: PathBuf,
    },

    /// The data dir holds data (`pages.db` / a non-empty WAL) but has NO
    /// `VERSION` file — a *legacy* dir created before this stamp existed
    /// (a pre-GA beta deployment). We refuse to silently proceed (its
    /// on-disk format is of unknown provenance and could be incompatible)
    /// and we refuse to silently auto-stamp it (that would mark a
    /// possibly-incompatible dir as current, defeating the guard).
    ///
    /// The REAL recovery path is the EXPLICIT operator opt-in
    /// `arcgraph serve --adopt-legacy-datadir --data <dir>`: if the operator
    /// confirms the dir IS the current format, that stamps it and proceeds.
    /// The message does NOT point at `arcgraph migrate` — that verb is the
    /// Neo4j-export IMPORT path (W18δ), NOT a version adopt; sending a beta
    /// operator there was a wrong/destructive dead-end (#1345 R1 REQUIRED).
    #[error(
        "data dir at {path} holds data but carries no `{version_file}` version stamp — it \
         predates the on-disk version guard (a pre-stamp beta deployment). If this dir IS \
         the current data-dir format v{current} (e.g. a beta→GA upgrade with no page-format \
         change), re-run with `arcgraph serve --adopt-legacy-datadir --data {path}` to adopt \
         it (this stamps the `{version_file}` file). Otherwise restore from a backup taken \
         with a compatible version, or point --data at a fresh dir. Refusing to open an \
         unstamped data dir of unknown format rather than misparsing it (issue #1302)"
    )]
    LegacyUnstamped {
        /// Data-dir path.
        path: PathBuf,
        /// The version-stamp file name (`VERSION`).
        version_file: &'static str,
        /// The format version this binary would stamp on an adopt.
        current: u16,
    },

    /// Explicit legacy adoption is unsafe when a MANIFEST identifies a format
    /// that cannot be recovered as a pre-M3 page-image store. In particular, a
    /// version-last M3 generation has MANIFEST=v5 before VERSION exists;
    /// stamping it as chained v1 would silently downgrade a complete delta-v9
    /// generation. A recognized v3/v4 page-image MANIFEST is different: it
    /// truthfully identifies the missing stamp and can be restored in place.
    #[error(
        "refusing `--adopt-legacy-datadir` for {path}: `{manifest_file}` already exists, so \
         this is a manifest-described ArcGraph generation, not an unstamped legacy store. \
         Resume the explicit migration with `arcgraph migrate upgrade-data-dir --data-dir \
         <operator-root>` (or open the operator root selected by CURRENT); never legacy-adopt \
         a generation directory"
    )]
    LegacyAdoptHasManifest {
        /// Generation path that must not be downgraded.
        path: PathBuf,
        /// Manifest file name used as the format authority.
        manifest_file: &'static str,
    },

    /// A v6/M4 generation's MANIFEST carries a `tel_ref_format` OLDER than
    /// [`crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1`] (#1519 BLOCK_FIX
    /// FIX 1, SILENT-M6-CORRUPTION).
    ///
    /// #1519 changed the STORE_TEL ref encoding (`NodeRecord::out_tel_ref`
    /// / `in_tel_ref` and a TEL block's `prev_block_ptr`) from a bare
    /// `PageType::Tel` page id to an opaque `encode_tel_ref(page_no << 16 |
    /// slot)` pair — but the coarse `VERSION` integer
    /// ([`crate::data_dir_version::DATA_DIR_VERSION_DIRECT_M4`]) is
    /// UNCHANGED, since both encodings are still v6/M4 generations at that
    /// granularity. Without this check, a pre-#1519 store (bare refs) is
    /// byte-indistinguishable from a post-#1519 store to the coarse
    /// `VERSION` guard: a bare page id ≥ 65536 decodes under the NEW
    /// `decode_tel_ref` inverse as a plausible-looking but WRONG
    /// `(page_no, slot)` pair — silent adjacency corruption at serve, with
    /// nothing refusing attach. This variant is the fine-grained,
    /// MANIFEST-level refusal that closes that gap (mirrors the
    /// `props_store_format` / `record_store_format` per-substrate
    /// discriminators the MANIFEST already carries for M1/M2/M4 —
    /// `crate::manifest` module docs).
    ///
    /// No migration path exists (or is needed) for this variant: D2/D3-built
    /// M4/M5 generations are ephemeral build/rung artifacts predating beta,
    /// not production data, so REFUSE-with-typed-error is the correct,
    /// simpler response — rebuild the generation with a binary that includes
    /// #1519 rather than attempting an in-place STORE_TEL re-encode.
    #[error(
        "v6/M4 generation at {path} has a stale STORE_TEL ref encoding \
         (tel_ref_format={found:?}, this binary requires {required:?}) — \
         #1519 changed the on-disk STORE_TEL ref encoding without bumping the \
         coarse data-dir VERSION, so an older generation's refs would silently \
         mis-decode as adjacency corruption if opened. Refusing to open a v6 \
         generation whose STORE_TEL predates #1519 rather than risk a silent \
         misread (issue #1519); rebuild the generation (offline migration or \
         `arcgraph load`) with a binary that includes #1519 to produce a \
         current tel_ref_format"
    )]
    StaleTelRefEncoding {
        /// Generation path carrying the stale MANIFEST.
        path: PathBuf,
        /// `tel_ref_format` found in the MANIFEST (or the implicit
        /// pre-field default, `TEL_REF_FORMAT_BARE_PAGE_ID`).
        found: String,
        /// `tel_ref_format` this binary requires
        /// (`TEL_REF_FORMAT_PAGE_SLOT_V1`).
        required: &'static str,
    },

    /// The `VERSION` file exists but its magic bytes are not
    /// [`DATA_DIR_VERSION_MAGIC`] — the dir is not an ArcGraph data dir, or
    /// the file was clobbered by an unrelated tool. Distinct from
    /// [`Self::Incompatible`] so the operator can tell "wrong file type
    /// entirely" from "right file, unsupported version" (mirrors
    /// `WalBadMagic` vs `WalFormatMismatch`).
    #[error(
        "`{version_file}` at {path} has bad magic (got {got:02x?}, expected {expected:02x?}) — \
         this does not look like an ArcGraph data dir, or the version stamp was clobbered. \
         Refusing to open (issue #1302)"
    )]
    BadMagic {
        /// Path of the `VERSION` file.
        path: PathBuf,
        /// Magic bytes read from the file.
        got: [u8; 8],
        /// Magic bytes expected (`b"ARCGDDV1"`).
        expected: &'static [u8; 8],
        /// The version-stamp file name (`VERSION`).
        version_file: &'static str,
    },

    /// The `VERSION` file exists but is not `VERSION_FILE_LEN` bytes, or
    /// its reserved bytes are non-zero (written by a future writer whose
    /// semantics this binary does not understand). Fail-closed.
    #[error(
        "`{version_file}` at {path} is malformed ({reason}) — refusing to open a data dir \
         with a corrupt or future-format version stamp (issue #1302)"
    )]
    Malformed {
        /// Path of the `VERSION` file.
        path: PathBuf,
        /// The version-stamp file name (`VERSION`).
        version_file: &'static str,
        /// Human-readable cause (too short / non-zero reserved).
        reason: String,
    },

    /// An I/O error reading or writing the `VERSION` file (permissions,
    /// `ENOSPC`, …). The stamp path failed for a filesystem reason, not a
    /// format reason.
    #[error("i/o error on `{version_file}` at {path}: {source}")]
    Io {
        /// Path of the `VERSION` file.
        path: PathBuf,
        /// The version-stamp file name (`VERSION`).
        version_file: &'static str,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Encode the fixed-size `VERSION` file body for `version`.
#[must_use]
fn encode_version_file(version: u16) -> [u8; VERSION_FILE_LEN] {
    let mut out = [0u8; VERSION_FILE_LEN];
    out[0..8].copy_from_slice(DATA_DIR_VERSION_MAGIC);
    out[8..10].copy_from_slice(&version.to_le_bytes());
    // out[10..12] = reserved (zero).
    out
}

/// Path of the `VERSION` file inside `data_dir`.
#[must_use]
pub fn version_file_path(data_dir: &Path) -> PathBuf {
    data_dir.join(VERSION_FILE)
}

/// Parse the `format_version` out of an on-disk `VERSION` file body,
/// validating magic + reserved bytes. Fail-closed over untrusted bytes.
fn decode_version_file(bytes: &[u8], path: &Path) -> Result<u16, DataDirVersionError> {
    if bytes.len() != VERSION_FILE_LEN {
        return Err(DataDirVersionError::Malformed {
            path: path.to_path_buf(),
            version_file: VERSION_FILE,
            reason: format!(
                "expected {VERSION_FILE_LEN} bytes, found {} — truncated or overwritten",
                bytes.len()
            ),
        });
    }
    let mut magic = [0u8; 8];
    magic.copy_from_slice(&bytes[0..8]);
    if &magic != DATA_DIR_VERSION_MAGIC {
        return Err(DataDirVersionError::BadMagic {
            path: path.to_path_buf(),
            got: magic,
            expected: DATA_DIR_VERSION_MAGIC,
            version_file: VERSION_FILE,
        });
    }
    if bytes[10] != 0 || bytes[11] != 0 {
        return Err(DataDirVersionError::Malformed {
            path: path.to_path_buf(),
            version_file: VERSION_FILE,
            reason: format!(
                "reserved bytes non-zero ({:#06x}) — written by a future format this binary \
                 does not understand",
                u16::from_le_bytes([bytes[10], bytes[11]])
            ),
        });
    }
    Ok(u16::from_le_bytes([bytes[8], bytes[9]]))
}

/// Stamp `<data_dir>/VERSION` at `version`, fsyncing the file so the
/// stamp is durable before the caller opens `pages.db` / the WAL. The
/// replacement is crash-atomic: a failed write can leave only the temporary
/// file torn while the prior `VERSION` remains intact.
/// `data_dir` MUST already exist (the bootstrap `create_dir_all`s it
/// first).
///
/// Used on a fresh dir (case 2, [`DATA_DIR_FORMAT_VERSION`]), on an
/// explicit legacy adopt (case 3 with `adopt_legacy`,
/// [`DATA_DIR_VERSION_CHAINED_V1`] — the dir's data IS the chained
/// format), and by the v2 M1 migrate-on-open to re-stamp `1 → 3`
/// BEFORE the first slotted byte can land durably (so a pre-M1 binary
/// refuses the dir from that moment on).
///
/// # Errors
///
/// [`DataDirVersionError::Io`] on a create/write/fsync failure.
pub fn stamp_data_dir(data_dir: &Path, version: u16) -> Result<(), DataDirVersionError> {
    stamp_data_dir_with_parent_sync(data_dir, version, |dir| fs::File::open(dir)?.sync_all())
}

/// Deterministic parent-directory fsync failure seam for crash gates.
///
/// This uses the exact implementation behind [`stamp_data_dir`]; only the
/// final directory-sync operation is replaced. It is public solely because
/// the CLI's subprocess migration gate lives in a separate crate.
#[doc(hidden)]
pub fn stamp_data_dir_with_parent_sync_error_for_test(
    data_dir: &Path,
    version: u16,
) -> Result<(), DataDirVersionError> {
    stamp_data_dir_with_parent_sync(data_dir, version, |_| {
        Err(std::io::Error::other(
            "injected VERSION parent-directory fsync failure",
        ))
    })
}

fn stamp_data_dir_with_parent_sync(
    data_dir: &Path,
    version: u16,
    sync_parent: impl FnOnce(&Path) -> std::io::Result<()>,
) -> Result<(), DataDirVersionError> {
    use std::io::Write;

    let path = version_file_path(data_dir);
    let tmp = data_dir.join(".VERSION.tmp");
    let body = encode_version_file(version);
    let io_err = |p: &Path, source: std::io::Error| DataDirVersionError::Io {
        path: p.to_path_buf(),
        version_file: VERSION_FILE,
        source,
    };
    {
        let mut file = fs::File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
        file.write_all(&body).map_err(|e| io_err(&tmp, e))?;
        file.sync_all().map_err(|e| io_err(&tmp, e))?;
    }
    fs::rename(&tmp, &path).map_err(|e| io_err(&path, e))?;
    sync_parent(data_dir).map_err(|e| io_err(data_dir, e))?;
    Ok(())
}

/// Boot-time data-dir version guard (SVC-2, #1302). Call this at durable
/// bootstrap, AFTER `<data_dir>` + its inter-process `LOCK` exist and
/// BEFORE `pages.db` / the WAL are opened, so an incompatible on-disk
/// version fails LOUD before any page is parsed.
///
/// `has_data` is the bootstrap's existing "does this dir already hold a
/// durable store?" signal (`pages.db` exists OR the WAL subdir is
/// non-empty; see the module-level §`has_data` scope note). `adopt_legacy`
/// is the EXPLICIT operator opt-in (`arcgraph serve --adopt-legacy-datadir`)
/// asserting an existing unstamped dir IS the current format. Together they
/// drive:
///
/// - `VERSION` present, supported version → clean no-op (proceed);
///   returns the found version so the bootstrap can run the v2
///   migrate-on-open when it reads [`DATA_DIR_VERSION_CHAINED_V1`].
/// - `VERSION` present, unsupported version → [`DataDirVersionError::Incompatible`]
///   (an incompatible version is NEVER adopted — `adopt_legacy` is ignored
///   here; the operator needs a matching binary or a restore).
/// - No `VERSION`, `has_data == false` (fresh dir) → stamp
///   [`DATA_DIR_FORMAT_VERSION`] (a store born at the current format).
/// - No `VERSION`, `has_data == true`, `adopt_legacy == false` (default) →
///   [`DataDirVersionError::LegacyUnstamped`] (fail loud, do not brick).
/// - No `VERSION`, `has_data == true`, `adopt_legacy == true` (explicit
///   opt-in) → restore the exact v3/v4 stamp from a recognized pre-M3
///   page-image MANIFEST, or stamp [`DATA_DIR_VERSION_CHAINED_V1`] when no
///   MANIFEST exists and proceed through the normal migrate-on-open chain.
///   M3 v5, unknown, and malformed MANIFESTs are refused rather than
///   downgraded through legacy adoption.
///
/// Returns the version the store is at when the check passes (found or
/// freshly stamped) so the caller can dispatch migrate-on-open.
///
/// See the module docs for the full policy rationale.
///
/// # Errors
///
/// Any [`DataDirVersionError`]: an incompatible / bad-magic / malformed
/// stamp, a legacy unstamped dir with data (without the adopt opt-in), or an
/// I/O failure.
pub fn check_or_stamp_data_dir(
    data_dir: &Path,
    has_data: bool,
    adopt_legacy: bool,
) -> Result<u16, DataDirVersionError> {
    let path = version_file_path(data_dir);
    match fs::read(&path) {
        Ok(bytes) => {
            // Case 1: a VERSION file exists — read + check. `adopt_legacy`
            // does NOT rescue an unsupported version: the on-disk format
            // really differs, so adopting (stamping current) would be a lie.
            let found = decode_version_file(&bytes, &path)?;
            if SUPPORTED_DATA_DIR_VERSIONS.contains(&found) {
                tracing::debug!(
                    target: "arcgraph_storage::data_dir_version",
                    found,
                    supported = ?SUPPORTED_DATA_DIR_VERSIONS,
                    "data-dir version stamp OK (#1302 boot guard)",
                );
                Ok(found)
            } else {
                Err(DataDirVersionError::Incompatible {
                    found,
                    supported: SUPPORTED_DATA_DIR_VERSIONS,
                    path,
                })
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            if has_data && !adopt_legacy {
                // Case 3a: legacy / pre-stamp dir with data, no explicit
                // adopt — fail loud, do not silently proceed and do not
                // silently auto-stamp (default stays fail-closed).
                Err(DataDirVersionError::LegacyUnstamped {
                    path: data_dir.to_path_buf(),
                    version_file: VERSION_FILE,
                    current: DATA_DIR_FORMAT_VERSION,
                })
            } else if has_data {
                if adopt_legacy {
                    match crate::manifest::read_data_dir_manifest(data_dir) {
                        Ok(Some(manifest))
                            if matches!(
                                manifest.data_dir_version,
                                DATA_DIR_VERSION_SLOTTED_M1 | DATA_DIR_VERSION_TYPED_M2
                            ) && manifest.wal_format
                                == crate::manifest::WAL_FORMAT_PAGE_IMAGE =>
                        {
                            // A pre-M3 MANIFEST is an authoritative, truthful
                            // recovery source for a missing coarse VERSION
                            // stamp. Restore that exact version instead of
                            // downgrading it to chained v1 and re-running
                            // migrations over already-migrated bytes.
                            stamp_data_dir(data_dir, manifest.data_dir_version)?;
                            tracing::info!(
                                target: "arcgraph_storage::data_dir_version",
                                version = manifest.data_dir_version,
                                adopted_legacy = true,
                                manifest_recovered = true,
                                "restored missing VERSION from pre-M3 MANIFEST (#1302)",
                            );
                            return Ok(manifest.data_dir_version);
                        }
                        Ok(None) => {}
                        Ok(Some(_)) | Err(_) => {
                            // M3 v5/delta-v9, unknown future formats, and
                            // malformed manifests are never legacy-adopted.
                            // Only an absent MANIFEST is genuinely pre-stamp;
                            // only known v3/v4 page-image manifests are safe
                            // missing-VERSION recovery sources.
                            return Err(DataDirVersionError::LegacyAdoptHasManifest {
                                path: data_dir.to_path_buf(),
                                manifest_file: crate::manifest::MANIFEST_FILE,
                            });
                        }
                    }
                }
                // Case 3b (has_data + explicit `adopt_legacy`): the only
                // format that can exist UNSTAMPED is the pre-stamp chained
                // v1.0 layout, so that is what the adopt truthfully stamps
                // — the ONLY path that stamps a dir that already holds
                // data. The caller then runs the same v2 migrate-on-open
                // any stamped chained dir gets.
                stamp_data_dir(data_dir, DATA_DIR_VERSION_CHAINED_V1)?;
                tracing::info!(
                    target: "arcgraph_storage::data_dir_version",
                    version = DATA_DIR_VERSION_CHAINED_V1,
                    adopted_legacy = true,
                    "adopted legacy data dir at the chained (pre-M1) on-disk format (#1302)",
                );
                Ok(DATA_DIR_VERSION_CHAINED_V1)
            } else {
                // Case 2 (fresh dir): born at the current format.
                stamp_data_dir(data_dir, DATA_DIR_FORMAT_VERSION)?;
                tracing::info!(
                    target: "arcgraph_storage::data_dir_version",
                    version = DATA_DIR_FORMAT_VERSION,
                    adopted_legacy = false,
                    "stamped fresh data dir with on-disk format version (#1302)",
                );
                Ok(DATA_DIR_FORMAT_VERSION)
            }
        }
        Err(source) => Err(DataDirVersionError::Io {
            path,
            version_file: VERSION_FILE,
            source,
        }),
    }
}

/// #1519 BLOCK_FIX FIX 1 (SILENT-M6-CORRUPTION) — refuse to open a v6/M4
/// generation whose MANIFEST names a STORE_TEL ref encoding older than
/// [`crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1`].
///
/// Call this at attach/open time for a generation the coarse `VERSION`
/// guard has already classified as
/// [`DATA_DIR_VERSION_DIRECT_M4`] — BEFORE any `PageType::Tel` page is
/// read — so a pre-#1519 generation (bare STORE_TEL refs) fails loud
/// instead of the new `decode_tel_ref` inverse silently misreading a bare
/// page id as a plausible-but-wrong `(page_no, slot)` pair.
///
/// `manifest` is `None` only for a v6/M4 generation with no MANIFEST at
/// all, which is already a distinct, earlier-caught inconsistency
/// (`verify_v6_generation` / the M4 migration's own invariants require a
/// MANIFEST for every v6 generation) — this function does not duplicate
/// that check and treats a missing manifest as "nothing to refuse here",
/// leaving the caller's existing MANIFEST-presence guard as the authority.
///
/// # Errors
///
/// [`DataDirVersionError::StaleTelRefEncoding`] when the MANIFEST's
/// `tel_ref_format` is not [`crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1`].
pub fn check_tel_ref_format(
    generation: &Path,
    manifest: Option<&crate::manifest::DataDirManifest>,
) -> Result<(), DataDirVersionError> {
    let Some(manifest) = manifest else {
        return Ok(());
    };
    if manifest.tel_ref_format != crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1 {
        return Err(DataDirVersionError::StaleTelRefEncoding {
            path: generation.to_path_buf(),
            found: manifest.tel_ref_format.clone(),
            required: crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Marker file standing in for `pages.db` — the bootstrap's "this dir
    /// holds data" signal in these unit tests.
    fn write_pages_db(dir: &Path) {
        fs::write(dir.join("pages.db"), b"fake page store").expect("write pages.db");
    }

    #[test]
    fn fresh_dir_stamps_current_version_and_reopen_is_noop() {
        // Test (2): fresh-dir round-trip. First open (no VERSION, no data)
        // stamps the current version; the file lands with the right magic +
        // version; a same-version re-open (Test 4) is a clean no-op.
        let tmp = TempDir::new().expect("tempdir");
        check_or_stamp_data_dir(tmp.path(), false, false).expect("fresh stamp must succeed");

        let path = version_file_path(tmp.path());
        assert!(path.exists(), "fresh open must write the VERSION file");
        let bytes = fs::read(&path).expect("read VERSION");
        assert_eq!(
            &bytes[0..8],
            DATA_DIR_VERSION_MAGIC,
            "magic must be stamped"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[8], bytes[9]]),
            DATA_DIR_FORMAT_VERSION,
            "current version must be stamped"
        );

        // Test (4): same-version re-open is a clean no-op (data now present).
        write_pages_db(tmp.path());
        check_or_stamp_data_dir(tmp.path(), true, false)
            .expect("same-version re-open must be a no-op");
    }

    #[test]
    fn boot_refuses_incompatible_version_the_oracle() {
        // THE ORACLE (Test 1): init at current version, tamper the VERSION
        // file to a DIFFERENT version, re-open → a clean, typed
        // incompatible-version error (NOT a misparse / panic).
        let tmp = TempDir::new().expect("tempdir");
        // Init at the current version + lay down data (a realistic restart).
        check_or_stamp_data_dir(tmp.path(), false, false).expect("initial stamp");
        write_pages_db(tmp.path());

        // Tamper: rewrite the VERSION file body at an unsupported version.
        let path = version_file_path(tmp.path());
        let tampered = DATA_DIR_FORMAT_VERSION.wrapping_add(7);
        assert!(
            !SUPPORTED_DATA_DIR_VERSIONS.contains(&tampered),
            "test precondition: tampered version must be unsupported"
        );
        fs::write(&path, encode_version_file(tampered)).expect("tamper VERSION");

        // Re-open MUST fail loud with the typed incompatible error.
        let err = check_or_stamp_data_dir(tmp.path(), true, false)
            .expect_err("re-open at an incompatible version MUST be refused");
        match &err {
            DataDirVersionError::Incompatible {
                found, supported, ..
            } => {
                assert_eq!(*found, tampered, "error must report the found version");
                assert_eq!(
                    *supported, SUPPORTED_DATA_DIR_VERSIONS,
                    "error must list the supported set"
                );
            }
            other => panic!("expected Incompatible, got {other:?}"),
        }
        // The message must be operator-actionable: point at the RIGHT
        // remediation (matching binary / restore), NOT `arcgraph migrate`
        // (the Neo4j-import verb) and NOT the adopt flag (an incompatible
        // version can never be adopted). #1345 R1 REQUIRED fix.
        let msg = err.to_string();
        assert!(
            !msg.contains("arcgraph migrate"),
            "incompatible error must NOT point at `arcgraph migrate` (Neo4j-import verb); got: {msg}"
        );
        assert!(
            msg.contains("matching ArcGraph binary") || msg.contains("restore"),
            "incompatible error must point at a matching binary / restore; got: {msg}"
        );
        assert!(
            msg.contains("can NOT be adopted"),
            "incompatible error must say an incompatible version can NOT be adopted; got: {msg}"
        );
        assert!(
            msg.contains(&tampered.to_string()),
            "incompatible error must name the found version; got: {msg}"
        );
    }

    #[test]
    fn legacy_unstamped_dir_with_data_fails_loud_not_silent() {
        // Test (3): a dir that holds data but has no VERSION file (a
        // pre-stamp beta deployment) and NO explicit adopt must NOT silently
        // proceed and must NOT silently auto-stamp — it fails loud pointing
        // at the REAL recovery path (the adopt flag), NOT `arcgraph migrate`.
        let tmp = TempDir::new().expect("tempdir");
        write_pages_db(tmp.path()); // data present, no VERSION file.

        let err = check_or_stamp_data_dir(tmp.path(), true, false)
            .expect_err("legacy unstamped dir with data MUST be refused");
        assert!(
            matches!(err, DataDirVersionError::LegacyUnstamped { .. }),
            "expected LegacyUnstamped, got {err:?}"
        );
        let msg = err.to_string();
        // #1345 R1 REQUIRED: must NOT point at `arcgraph migrate` (the
        // Neo4j-import verb — a wrong/destructive dead-end for beta ops).
        assert!(
            !msg.contains("arcgraph migrate"),
            "legacy error must NOT point at `arcgraph migrate` (Neo4j-import verb); got: {msg}"
        );
        assert!(
            msg.contains("--adopt-legacy-datadir"),
            "legacy error must point at the real adopt path; got: {msg}"
        );
        assert!(
            msg.contains("predates"),
            "legacy error must explain the dir predates the stamp; got: {msg}"
        );
        // Critically: the guard must NOT have written a VERSION file (no
        // silent auto-stamp of an unknown-provenance dir).
        assert!(
            !version_file_path(tmp.path()).exists(),
            "legacy dir must NOT be silently auto-stamped"
        );
    }

    #[test]
    fn legacy_unstamped_dir_with_explicit_adopt_stamps_and_proceeds() {
        // #1345 R1 REQUIRED adopt path (case 3b): a legacy dir with data +
        // the EXPLICIT operator opt-in stamps the dir and proceeds — the
        // real beta→GA recovery. This is the ONLY path that writes a stamp
        // onto a dir that already holds data. v2 M1: the adopt stamps the
        // CHAINED (pre-M1) version — the only format that can exist
        // unstamped — and the same boot then migrates it forward.
        let tmp = TempDir::new().expect("tempdir");
        write_pages_db(tmp.path()); // data present, no VERSION file.
        assert!(
            !version_file_path(tmp.path()).exists(),
            "precondition: legacy dir has no VERSION"
        );

        let adopted = check_or_stamp_data_dir(tmp.path(), true, true)
            .expect("explicit --adopt-legacy-datadir MUST adopt a legacy dir");
        assert_eq!(
            adopted, DATA_DIR_VERSION_CHAINED_V1,
            "adopt reports the chained (pre-M1) version"
        );

        // The chained version is now stamped, so a subsequent NORMAL boot
        // (no adopt flag) is a clean no-op — the adopt is durable.
        let vpath = version_file_path(tmp.path());
        assert!(vpath.exists(), "adopt must stamp the VERSION file");
        let bytes = fs::read(&vpath).expect("read VERSION");
        assert_eq!(
            &bytes[0..8],
            DATA_DIR_VERSION_MAGIC,
            "magic stamped on adopt"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[8], bytes[9]]),
            DATA_DIR_VERSION_CHAINED_V1,
            "adopt stamps the chained (pre-M1) version — the format the data actually is"
        );
        let reopened = check_or_stamp_data_dir(tmp.path(), true, false)
            .expect("after adopt, a normal boot is a clean no-op");
        assert_eq!(reopened, DATA_DIR_VERSION_CHAINED_V1);
    }

    #[test]
    fn legacy_adopt_refuses_v5_manifest_without_writing_version() {
        let tmp = TempDir::new().expect("tempdir");
        write_pages_db(tmp.path());
        let prior = crate::manifest::DataDirManifest::m2_typed("2026-07-12T00:00:00Z".to_owned());
        let manifest = crate::manifest::DataDirManifest::m3_delta_from(
            &prior,
            "2026-07-12T00:01:00Z".to_owned(),
            arcgraph_core::Lsn::new(41),
        );
        crate::manifest::write_data_dir_manifest(tmp.path(), &manifest).unwrap();

        let error = check_or_stamp_data_dir(tmp.path(), true, true)
            .expect_err("v5 manifest-described generation must never be legacy-adopted");
        assert!(matches!(
            error,
            DataDirVersionError::LegacyAdoptHasManifest { .. }
        ));
        assert!(error.to_string().contains("upgrade-data-dir"));
        assert!(!version_file_path(tmp.path()).exists());
    }

    #[test]
    fn legacy_adopt_restores_recognized_v4_manifest_version() {
        let tmp = TempDir::new().expect("tempdir");
        write_pages_db(tmp.path());
        let manifest =
            crate::manifest::DataDirManifest::m2_typed("2026-07-12T00:00:00Z".to_owned());
        crate::manifest::write_data_dir_manifest(tmp.path(), &manifest).unwrap();

        let adopted = check_or_stamp_data_dir(tmp.path(), true, true)
            .expect("recognized v4 MANIFEST must recover its missing VERSION");
        assert_eq!(adopted, DATA_DIR_VERSION_TYPED_M2);
        let bytes = fs::read(version_file_path(tmp.path())).expect("read restored VERSION");
        assert_eq!(
            u16::from_le_bytes([bytes[8], bytes[9]]),
            DATA_DIR_VERSION_TYPED_M2,
            "adopt must restore the version the MANIFEST truthfully describes"
        );
    }

    #[test]
    fn chained_v1_dir_is_supported_and_reported_for_migration() {
        // v2 M1: a stamped chained (v1) dir opens cleanly and the check
        // REPORTS the found version so the bootstrap dispatches the
        // migrate-on-open. After migration re-stamps 3, reopen reports 3.
        let tmp = TempDir::new().expect("tempdir");
        stamp_data_dir(tmp.path(), DATA_DIR_VERSION_CHAINED_V1).expect("stamp v1");
        write_pages_db(tmp.path());
        let found = check_or_stamp_data_dir(tmp.path(), true, false)
            .expect("chained v1 dir must be supported (migrate-on-open source)");
        assert_eq!(found, DATA_DIR_VERSION_CHAINED_V1);

        // The M1 migration's re-stamp: 1 → 3.
        stamp_data_dir(tmp.path(), DATA_DIR_FORMAT_VERSION).expect("re-stamp v3");
        let found = check_or_stamp_data_dir(tmp.path(), true, false).expect("v3 reopen");
        assert_eq!(found, DATA_DIR_FORMAT_VERSION);
    }

    #[test]
    fn incompatible_version_is_never_adopted_even_with_flag() {
        // #1345 R1 REQUIRED: adopt rescues ONLY an unstamped legacy dir —
        // an INCOMPATIBLE stamped version is refused even WITH the flag (the
        // format really differs; stamping current would be a lie).
        let tmp = TempDir::new().expect("tempdir");
        check_or_stamp_data_dir(tmp.path(), false, false).expect("initial stamp");
        write_pages_db(tmp.path());
        let path = version_file_path(tmp.path());
        let tampered = DATA_DIR_FORMAT_VERSION.wrapping_add(11);
        fs::write(&path, encode_version_file(tampered)).expect("tamper");

        // WITH the adopt flag set — still refused.
        let err = check_or_stamp_data_dir(tmp.path(), true, true)
            .expect_err("an incompatible version MUST be refused even with --adopt-legacy-datadir");
        assert!(
            matches!(err, DataDirVersionError::Incompatible { .. }),
            "adopt must NOT rescue an incompatible version; got {err:?}"
        );
        // The stamp is untouched (adopt did not overwrite the incompatible
        // version with the current one).
        let after = fs::read(&path).expect("read VERSION after refused adopt");
        assert_eq!(
            u16::from_le_bytes([after[8], after[9]]),
            tampered,
            "refused adopt must NOT overwrite the incompatible stamp"
        );
    }

    #[test]
    fn bad_magic_version_file_is_refused() {
        // A VERSION file with wrong magic (clobbered / not an ArcGraph dir)
        // is distinct from an unsupported version — fail closed.
        let tmp = TempDir::new().expect("tempdir");
        let path = version_file_path(tmp.path());
        let mut body = encode_version_file(DATA_DIR_FORMAT_VERSION);
        body[0..8].copy_from_slice(b"NOTARCG!");
        fs::write(&path, body).expect("write bad-magic VERSION");

        let err = check_or_stamp_data_dir(tmp.path(), true, false)
            .expect_err("bad-magic VERSION must be refused");
        assert!(
            matches!(err, DataDirVersionError::BadMagic { .. }),
            "expected BadMagic, got {err:?}"
        );
    }

    #[test]
    fn malformed_version_file_is_refused() {
        // Too-short and non-zero-reserved bodies both fail closed.
        let tmp = TempDir::new().expect("tempdir");
        let path = version_file_path(tmp.path());

        // Truncated body.
        fs::write(&path, b"ARCG").expect("write short VERSION");
        let err =
            check_or_stamp_data_dir(tmp.path(), true, false).expect_err("short VERSION refused");
        assert!(
            matches!(err, DataDirVersionError::Malformed { .. }),
            "expected Malformed (short), got {err:?}"
        );

        // Non-zero reserved bytes.
        let mut body = encode_version_file(DATA_DIR_FORMAT_VERSION);
        body[10] = 0xAB;
        fs::write(&path, body).expect("write reserved-nonzero VERSION");
        let err = check_or_stamp_data_dir(tmp.path(), true, false)
            .expect_err("non-zero reserved VERSION refused");
        assert!(
            matches!(err, DataDirVersionError::Malformed { .. }),
            "expected Malformed (reserved), got {err:?}"
        );
    }

    #[test]
    fn same_version_reopen_is_clean_noop_no_rewrite() {
        // Test (4) sharpened: a supported-version re-open must not rewrite
        // the stamp (idempotent, no churn) and must succeed.
        let tmp = TempDir::new().expect("tempdir");
        check_or_stamp_data_dir(tmp.path(), false, false).expect("initial stamp");
        let path = version_file_path(tmp.path());
        let before = fs::read(&path).expect("read VERSION before");
        write_pages_db(tmp.path());

        check_or_stamp_data_dir(tmp.path(), true, false).expect("re-open must succeed");
        let after = fs::read(&path).expect("read VERSION after");
        assert_eq!(
            before, after,
            "supported re-open must not rewrite the stamp"
        );
    }

    #[test]
    fn round_trip_encode_decode() {
        // Encode/decode symmetry for the fixed-size body.
        let body = encode_version_file(DATA_DIR_FORMAT_VERSION);
        assert_eq!(body.len(), VERSION_FILE_LEN);
        let path = Path::new("/tmp/does-not-matter/VERSION");
        let decoded = decode_version_file(&body, path).expect("decode own encoding");
        assert_eq!(decoded, DATA_DIR_FORMAT_VERSION);
    }

    // ─────────────────────────────────────────────────────────────────
    // #1519 BLOCK_FIX FIX 1 (SILENT-M6-CORRUPTION) — STORE_TEL
    // ref-encoding discriminator (`check_tel_ref_format`).
    // ─────────────────────────────────────────────────────────────────

    fn m4_manifest_with_tel_ref_format(
        now: &str,
        tel_ref_format: &str,
    ) -> crate::manifest::DataDirManifest {
        let prior = crate::manifest::DataDirManifest::m3_delta_from(
            &crate::manifest::DataDirManifest::m2_typed(now.to_owned()),
            now.to_owned(),
            arcgraph_core::Lsn::new(41),
        );
        let mut manifest = crate::manifest::DataDirManifest::m4_direct_from(
            &prior,
            now.to_owned(),
            arcgraph_core::Lsn::new(42),
            vec![1],
            "0".repeat(64),
        );
        manifest.tel_ref_format = tel_ref_format.to_string();
        manifest
    }

    #[test]
    fn check_tel_ref_format_accepts_current_encoding() {
        // The happy path: a v6/M4 generation whose MANIFEST names the
        // CURRENT #1519 encoding must be accepted (attach proceeds).
        let manifest = m4_manifest_with_tel_ref_format(
            "2026-07-17T00:00:00Z",
            crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1,
        );
        check_tel_ref_format(Path::new("/tmp/does-not-matter"), Some(&manifest))
            .expect("current tel_ref_format must be accepted");
    }

    #[test]
    fn check_tel_ref_format_refuses_explicit_old_encoding() {
        // A MANIFEST explicitly naming the pre-#1519 bare-page-id encoding
        // (e.g. a D2/D3-built generation from before #1519 landed, whose
        // MANIFEST was written by that older binary) must be REFUSED, not
        // silently opened.
        let manifest = m4_manifest_with_tel_ref_format(
            "2026-07-17T00:00:00Z",
            crate::manifest::TEL_REF_FORMAT_BARE_PAGE_ID,
        );
        let path = Path::new("/some/v6/generation");
        let err = check_tel_ref_format(path, Some(&manifest))
            .expect_err("pre-#1519 bare-page-id tel_ref_format must be refused");
        match &err {
            DataDirVersionError::StaleTelRefEncoding {
                path: err_path,
                found,
                required,
            } => {
                assert_eq!(err_path, path);
                assert_eq!(found, crate::manifest::TEL_REF_FORMAT_BARE_PAGE_ID);
                assert_eq!(*required, crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1);
            }
            other => panic!("expected StaleTelRefEncoding, got {other:?}"),
        }
        // Operator-actionable message: must not silently proceed and must
        // explain the class of risk (mirrors the #1345 R1 message-quality
        // bar the other DataDirVersionError variants meet).
        let msg = err.to_string();
        assert!(
            msg.contains("SILENT") || msg.contains("silent") || msg.contains("silently"),
            "message must name the silent-corruption risk; got: {msg}"
        );
        assert!(
            msg.contains("1519"),
            "message must reference the format-change issue; got: {msg}"
        );
    }

    #[test]
    fn check_tel_ref_format_refuses_a_manifest_predating_the_field_entirely() {
        // #1519 BLOCK_FIX FIX 1's core scenario: a manifest written by a
        // binary that PREDATES the `tel_ref_format` field's existence
        // entirely — the exact on-disk shape of every M4/M5 generation
        // built before #1519 landed (byte-for-byte; the field is simply
        // ABSENT from the JSON, not present-with-an-old-value). Deserializing
        // that JSON must resolve `tel_ref_format` to
        // `TEL_REF_FORMAT_BARE_PAGE_ID` via `#[serde(default)]` — an absent
        // field must mean "old store", never "current" — and
        // `check_tel_ref_format` must then refuse it.
        let now = "2026-07-17T00:00:00Z";
        let current =
            m4_manifest_with_tel_ref_format(now, crate::manifest::TEL_REF_FORMAT_PAGE_SLOT_V1);
        let json = serde_json::to_string(&current).expect("serialize manifest");
        // Simulate the pre-#1519 on-disk shape: strip the `tel_ref_format`
        // field out of the JSON entirely, exactly as an older binary's
        // MANIFEST would never have written it (`deny_unknown_fields` only
        // rejects EXTRA fields; a MISSING field is exactly what
        // `#[serde(default)]` exists to make safe).
        let value: serde_json::Value = serde_json::from_str(&json).expect("parse to Value");
        let mut object = value
            .as_object()
            .expect("manifest is a JSON object")
            .clone();
        assert!(
            object.remove("tel_ref_format").is_some(),
            "test precondition: tel_ref_format field must be present before stripping"
        );
        let stripped = serde_json::Value::Object(object);
        let stripped_json = serde_json::to_string(&stripped).expect("reserialize stripped JSON");

        let pre_1519_manifest: crate::manifest::DataDirManifest = serde_json::from_str(
            &stripped_json,
        )
        .expect("a manifest missing tel_ref_format must still deserialize (backward compat)");
        assert_eq!(
            pre_1519_manifest.tel_ref_format,
            crate::manifest::TEL_REF_FORMAT_BARE_PAGE_ID,
            "an ABSENT tel_ref_format field must default to the OLD bare-page-id \
             meaning, never be silently treated as current"
        );

        let path = Path::new("/some/pre-1519/generation");
        let err = check_tel_ref_format(path, Some(&pre_1519_manifest))
            .expect_err("a manifest predating the tel_ref_format field must be refused");
        assert!(
            matches!(err, DataDirVersionError::StaleTelRefEncoding { .. }),
            "expected StaleTelRefEncoding, got {err:?}"
        );
    }

    #[test]
    fn check_tel_ref_format_is_a_noop_without_a_manifest() {
        // A v6/M4 generation with literally no MANIFEST is an earlier,
        // distinct inconsistency this function does not duplicate-check
        // (the caller's MANIFEST-presence guard is the authority there).
        check_tel_ref_format(Path::new("/tmp/does-not-matter"), None)
            .expect("no manifest => nothing to refuse here");
    }

    #[test]
    fn decode_tel_ref_misdecodes_a_bare_pre_1519_page_id_the_corruption_this_guards_against() {
        // THE ADJUDICATOR DIFFERENTIAL (charter FIX 1 RED-on-revert
        // pattern): demonstrate the actual SILENT-M6-CORRUPTION the
        // discriminator exists to prevent — a pre-#1519 bare `PageType::Tel`
        // page id, read through the NEW `decode_tel_ref` inverse (what an
        // M6 reader would do without this guard), decodes as a
        // plausible-looking but WRONG `(page_no, slot)` pair whenever the
        // bare id is >= 65536 (2^16) — never the original page.
        let bare_page_id: u64 = 100_000; // a realistic large-store page id
        let (decoded_page, decoded_slot) = crate::m4_migration::decode_tel_ref(bare_page_id);
        assert_ne!(
            decoded_page, bare_page_id,
            "the whole point of the discriminator: an old bare page id must \
             NOT round-trip through the new decode_tel_ref — if it did, there \
             would be no silent-corruption class to guard against"
        );
        // Concretely: bare id 100_000 = 0x0001_86A0 decodes as page
        // 100_000 >> 16 = 1, slot 100_000 & 0xFFFF = 34464 — a
        // plausible-looking (page, slot) pair naming an ENTIRELY
        // DIFFERENT physical page than the original bare id 100_000. This
        // is the exact "valid-looking WRONG page" the charter names.
        assert_eq!(decoded_page, 1);
        assert_eq!(decoded_slot, 34_464);
    }
}
