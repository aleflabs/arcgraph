//! Shared per-process storage substrate bootstrap for the `arcgraph` +
//! `arcgraph-mcp-stdio` binaries (W26-β-1 GA-BOOTSTRAP-WIRING;
//! W28 GA durable-by-default per ADR-183).
//!
//! Both production CLI binaries (`bin/arcgraph.rs` + `bin/arcgraph_mcp_stdio.rs`)
//! previously hand-rolled byte-identical bootstrap bodies that constructed
//! the storage substrate via [`CrudStore::new`] — the W17α debugging-only
//! shape. Per ADR-087 D-2, every deployment surface that consumes the MCP
//! `graph.raw_query` / Bolt RUN paths MUST construct its [`CrudStore`] via
//! [`CrudStore::new_with_index`]; the W18δ-flagged operator-visible gap
//! (`arcgraph serve` ingest → `graph.schema` returns empty labels) was
//! forward-pinned to this v1.0-GA deployment-hardening slice + tracked at
//! issue #439.
//!
//! This module is the single canonical wire-pattern call site for both
//! binaries — extracting the bootstrap kills the duplication AND prevents
//! future drift between the two binaries' production postures.
//!
//! # Durable-by-default (W28 / ADR-183, GA blocker #659)
//!
//! The substrate is now selected by an explicit [`BootstrapMode`]:
//!
//! - [`BootstrapMode::Durable`] (the production default for `serve`):
//!   file-backed [`PosixPageIo`] page store + a real [`WalWriter`] over
//!   `<data_dir>/wal` + WAL recovery on startup. The `DEFAULT` tenant is
//!   bootstrapped at [`arcgraph_core::DurabilityTier::Strict`] (catalog
//!   default, ADR-034 §Slice A) → every acknowledged commit is fsync-durable
//!   *before* `commit()` returns (ADR-034 §I-D1, "T1 commit is durable before
//!   ack"), so committed records survive process restart. The crash-consistency
//!   boundary is the WAL writer's `committed_fsync_watermark` (ADR-034 §Slice B —
//!   the section that adds the watermark to `wal/writer.rs`): only commits at or
//!   below the watermark survive a crash.
//! - [`BootstrapMode::InMemory`] (explicit `--in-memory` opt-in only):
//!   [`InMemoryPageIo`] + no WAL — the prior v1.0-α posture. **NON-DURABLE**:
//!   all data is lost on process exit. For tests + ephemeral demos.
//!
//! [`BootstrapMode::from_flags`] enforces the **refuse-to-start** policy
//! (ADR-183 §Policy): `serve` with neither `--data` nor `--in-memory` exits
//! with an error naming both flags — a GA server never silently comes up
//! non-durable.
//!
//! The durable recover wiring mirrors the canonical
//! `crates/arcgraph-storage/tests/k1_smoke_30s.rs::recover_stack` /
//! `m4_41_cold_start_rebuild.rs::recover_stack` pattern (validated by the
//! K-3 10K-cycle crash campaign with post-recovery commits): build raw replay
//! targets, run `recover_from_wal` before any writer attaches, truncate any
//! torn terminal tail, then attach the writer/catalog/runtime wrappers and run
//! M4-41 cold-start [`rebuild_all_tenant_stats`] so `CatalogStats` is
//! populated. The fully-wired [`PageStoreTarget`] includes primary + record
//! store + blob store + allocator seed per ADR-183 R1 +
//! `wal/replay.rs:462-545` field contracts.
//!
//! # ADR provenance
//!
//! - **ADR-183 ("Durable-by-default server bootstrap")**: the durable
//!   substrate selection + refuse-to-start policy + the multi-tenant
//!   registry-recovery forward-pin (default-tenant durability is the GA
//!   scope; non-`DEFAULT` tenant *registry* entries do not yet survive
//!   restart — a built catalog-recover-from-pages path is M10 / "dedicated
//!   catalog page").
//! - **ADR-087 D-2 ("Primary-index wiring")**: every deployment surface
//!   that consumes the MCP `graph.raw_query` / Bolt RUN paths MUST
//!   construct its `CrudStore` via [`CrudStore::new_with_index`]. Without a
//!   primary index the [`arcgraph_storage::crud`]`::commit` path
//!   early-returns past the per-tenant `CatalogStats` update, so the
//!   catalog stays empty regardless of how many `graph.ingest` calls fire.
//! - **ADR-034 §Slice A ("DurabilityTier enum + TenantCatalog tier field")**:
//!   `DEFAULT` bootstraps at `Strict` (T1, fsync-per-commit) — the durability
//!   tier the GA guarantee rests on.
//! - **ADR-034 §Slice B ("WAL writer per-tier dispatch") + §I-D1 ("T1 commit
//!   is durable before ack")**: the `committed_fsync_watermark` lives on the
//!   WAL writer (added in §Slice B; `wal/writer.rs` attributes it there) and is
//!   the crash-consistency boundary — only commits at or below it survive a
//!   crash; §I-D1 is the matching durable-before-ack invariant.
//!
//! # Canonical wire-pattern references
//!
//! - `crates/arcgraph-storage/tests/k1_smoke_30s.rs::recover_stack`
//! - `crates/arcgraph-storage/tests/m4_41_cold_start_rebuild.rs::recover_stack`
//!
//! # Forward-deferred (post-S1)
//!
//! - Multi-tenant registry recovery (S0/M10 — see ADR-183 §Forward-pin).
//! - `arcgraph check` / `dump` over a durable dir (S3).
//! - WAL checkpoint/compaction to bound recovery cost (recovery is O(WAL
//!   size) at v1.0-GA — ADR-183 §Back-of-envelope).
//! - Secondary-index wiring via [`CrudStore::new_with_indices`] when the
//!   property→NodeId reverse index lands (M2-34+).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, bail, ensure};
// ADR-202 §D-8 serve-binary community-scheduler slice — community +
// engine scheduler primitives are pulled through the `arcgraph` umbrella
// facade (already a direct dep), so this slice adds NO new direct
// dependency on `arcgraph-community` (it is a workspace Apache-2.0 crate
// already transitive via `arcgraph-storage`). `start_community_scheduler`
// reuses these.
use arcgraph::community::{
    BTreeMembershipIndex, CommunityIndexId, CommunityRefreshScheduler, LeidenParams, RefreshHook,
    RefreshObserver, SchedulerConfig, SharedBTreeIndexProvider,
};
use arcgraph::storage::{CrudStoreGraphAdapter, ProductionRefreshHook};
use arcgraph_bm25::Bm25Service;
use arcgraph_core::{KekVersion, KeyScope, KeySource, Lsn, SecretsProvider, TenantId};
use arcgraph_mcp::storage::StorageBackend;
use arcgraph_storage::buffer::BufferPool;
use arcgraph_storage::catalog::SystemCatalog;
use arcgraph_storage::crud::{
    CrudAclWalSink, CrudStore, crud_allocator_seed_handle, crud_allocator_seed_handle_with_owners,
};
use arcgraph_storage::encryption::{
    SecretsProviderKeySource, WalEncryptionBootstrap, bootstrap_wal_encryption,
};
use arcgraph_storage::io::{InMemoryPageIo, PageIo, PosixPageIo};
use arcgraph_storage::metrics::MetricsSink;
use arcgraph_storage::mutation_log::Bm25IndexStoreHandle;
use arcgraph_storage::page_alloc::PageAllocator;
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PageStoreIdentity, PerTenantBufferPool, PerTenantBufferPoolConfig,
    TenantFilePageIo, TenantPageIo,
};
use arcgraph_storage::permissions::PermissionIndex;
use arcgraph_storage::primary_index::{PrimaryIndex, PrimaryPageStore};
use arcgraph_storage::record_store::RecordPageStore;
use arcgraph_storage::recovery::{
    rebuild_all_tenant_adjacency, rebuild_all_tenant_index, rebuild_all_tenant_stats,
};
use arcgraph_storage::router::MultiTenantRouter;
use arcgraph_storage::transaction::TxnManager;
use arcgraph_storage::vector_store::{VectorPageStore, VectorPageStoreHandle};
use arcgraph_storage::wal::{
    AllocatorSeedHandle, BlobStoreHandle, PageStoreTarget, PrimaryPageStoreHandle,
    RecordPageStoreHandle, WalConfig, WalWriter, recover_from_wal_encrypted_anchored,
    recover_from_wal_encrypted_incremental, truncate_torn_tail,
};
use arcgraph_storage::{
    BlobBoundConfig, BlobSpill, BlobStore, DirtyPageTable, IdempotencyBoundConfig,
    IdempotencySpill, IdempotencyStore, InternTable,
};

use crate::data_lock::DataDirLock;

/// Buffer-pool frame count for the catalog page store. The catalog is the
/// only consumer of the buffer pool at v1.0 (records + index live in the
/// MVCC stores rebuilt by WAL replay); 256 frames is the prior v1.0-α
/// default, preserved. M10 stage-1 (ADR-207): the catalog now PINS this
/// pool — `SystemCatalog::attach_page_store` materializes + read-back-
/// verifies the catalog root page at bootstrap and tier mutations write
/// through — so the pool is no longer a bootstrap-scoped local; it lives
/// inside the catalog for the server's lifetime.
const POOL_FRAMES: usize = 256;

/// File name of the [`PosixPageIo`]-backed page store inside `<data_dir>`.
const PAGES_FILE: &str = "pages.db";

/// Sub-directory of `<data_dir>` holding the WAL segment files.
const WAL_SUBDIR: &str = "wal";

/// Small bounded caches for each production Slice-2 extent owner. Directory
/// and data residency stays a function of active tenant/store owners, never
/// of the extent census.
const EXTENT_DIRECTORY_FRAMES: usize = 8;
const EXTENT_DATA_FRAMES: usize = 16;

/// Per-process discriminator for `--in-memory` BM25 temp roots.
static IN_MEMORY_BM25_BOOTSTRAP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct OpenedProductionExtents {
    stores: BTreeMap<(TenantId, u16), ProductionExtentRuntime>,
    affinity_allocators: BTreeMap<TenantId, arcgraph_storage::extent::PairedAffinityAllocator>,
}

fn open_production_extent_stores(
    generation: &Path,
    dpt: Arc<DirtyPageTable>,
    m4_generation: bool,
) -> Result<OpenedProductionExtents> {
    let mut tenants = BTreeSet::from([TenantId::DEFAULT]);
    let tenants_root = generation.join(arcgraph_storage::m3_migration::M3_TENANTS_DIR);
    match std::fs::read_dir(&tenants_root) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                if entry.file_type()?.is_dir()
                    && let Some(raw) = entry
                        .file_name()
                        .to_str()
                        .and_then(|name| name.parse().ok())
                {
                    tenants.insert(TenantId::new(raw));
                }
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }

    let mut stores = BTreeMap::new();
    for &tenant in &tenants {
        let store_ids: &[u16] = if m4_generation {
            arcgraph_storage::m4_migration::M4_EXTENT_STORE_IDS
        } else {
            &[
                arcgraph_storage::wal::STORE_PROPS,
                arcgraph_storage::wal::STORE_RECORD,
                arcgraph_storage::wal::STORE_TEL,
            ]
        };
        for &store_id in store_ids {
            let path = arcgraph_storage::extent::production_extent_store_path(
                generation, tenant, store_id,
            )
            .expect("production extent store id is supported");
            let physical: Arc<dyn PageIo> =
                if m4_generation {
                    // A v6 generation is complete before CURRENT selects it.
                    // Creating a missing store here would turn a half-built or
                    // corrupted generation into an apparently valid empty one.
                    Arc::new(PosixPageIo::open(&path).with_context(|| {
                        format!("open required v6 extent store {}", path.display())
                    })?)
                } else {
                    std::fs::create_dir_all(path.parent().expect("extent store path has parent"))?;
                    Arc::new(PosixPageIo::open_or_create(&path).with_context(|| {
                        format!("open production extent store {}", path.display())
                    })?)
                };
            let directory = Arc::new(arcgraph_storage::extent::ExtentDirectory::new(
                tenant,
                store_id,
                physical,
                EXTENT_DIRECTORY_FRAMES,
            ));
            let data = Arc::new(arcgraph_storage::extent::ExtentDataPageStore::new(
                Arc::clone(&directory),
                EXTENT_DATA_FRAMES,
            ));
            stores.insert(
                (tenant, store_id),
                ProductionExtentRuntime {
                    directory,
                    data,
                    dpt: Arc::clone(&dpt),
                },
            );
        }
    }
    let mut affinity_allocators = BTreeMap::new();
    for tenant in tenants {
        let props = stores
            .get(&(tenant, arcgraph_storage::wal::STORE_PROPS))
            .expect("every production extent set has props.store");
        let tel = stores
            .get(&(tenant, arcgraph_storage::wal::STORE_TEL))
            .expect("every production extent set has tel.store");
        let allocator = arcgraph_storage::extent::PairedAffinityAllocator::new_recovered(
            Arc::clone(&props.data),
            Arc::clone(&tel.data),
            Arc::clone(&dpt),
        )
        .with_context(|| {
            format!(
                "recover production extent counters for tenant {}",
                tenant.raw()
            )
        })?;
        affinity_allocators.insert(tenant, allocator);
    }
    Ok(OpenedProductionExtents {
        stores,
        affinity_allocators,
    })
}

fn in_memory_bm25_dir() -> PathBuf {
    let seq = IN_MEMORY_BM25_BOOTSTRAP_ID.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir()
        .join("arcgraph-bm25-in-memory")
        .join(format!("pid-{}-{seq}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Which storage substrate [`bootstrap_storage_backend`] constructs.
///
/// Resolved from the CLI flags via [`BootstrapMode::from_flags`], which
/// enforces the ADR-183 refuse-to-start policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootstrapMode {
    /// Durable substrate rooted at `data_dir`: [`PosixPageIo`] page store +
    /// WAL + recover-on-startup. The production default for `serve`.
    Durable {
        /// The `--data <dir>` directory. `<dir>/pages.db` holds the page
        /// store; `<dir>/wal/` holds the WAL segments. Created if absent.
        data_dir: PathBuf,
    },
    /// Ephemeral, **NON-DURABLE** substrate: [`InMemoryPageIo`] + no WAL.
    /// All data is lost on process exit. Explicit `--in-memory` opt-in
    /// only (tests + ephemeral demos).
    InMemory,
}

impl BootstrapMode {
    /// Resolve the `--data <dir>` / `--in-memory` CLI flags into a mode.
    ///
    /// Enforces the ADR-183 **refuse-to-start** policy: `serve` invoked
    /// with neither flag (`data_dir = None`, `in_memory = false`) returns
    /// an error naming both flags + documenting `--in-memory` as
    /// non-durable, so the operator's durability choice is informed and
    /// explicit. A GA server must never silently come up non-durable, and
    /// a CWD-relative default dir is a footgun.
    ///
    /// `--data` and `--in-memory` are mutually exclusive; passing both is
    /// an error. (The `arcgraph serve` clap surface also marks them
    /// `conflicts_with`, so the contradictory case is normally rejected at
    /// parse time; this method defends the contract for every caller.)
    ///
    /// # Errors
    ///
    /// Returns [`anyhow::Error`] when neither flag is set (refuse-to-start)
    /// or both are set (mutually exclusive).
    pub fn from_flags(data_dir: Option<&Path>, in_memory: bool) -> Result<Self> {
        match (data_dir, in_memory) {
            (Some(_), true) => bail!(
                "--data and --in-memory are mutually exclusive. Pass exactly one: \
                 --data <dir> for a durable store, XOR --in-memory for an ephemeral \
                 (non-durable) store."
            ),
            (Some(dir), false) => Ok(Self::Durable {
                data_dir: dir.to_path_buf(),
            }),
            (None, true) => Ok(Self::InMemory),
            (None, false) => bail!(
                "refusing to start without an explicit storage mode.\n  \
                 Pass --data <dir>   for a DURABLE store (file-backed pages + WAL; \
                 acknowledged commits survive restart), or\n  \
                 pass --in-memory    for an EPHEMERAL / NON-DURABLE store (ALL data is \
                 lost on process exit; intended for tests + demos).\n\
                 A GA server never silently comes up non-durable (ADR-183 §Policy)."
            ),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────
// WAL-encryption config + KeySource construction (ADR-216 §D-4 / #1180)
// ─────────────────────────────────────────────────────────────────────────

/// Which [`KeySource`] to construct for WAL encryption (ADR-216 §D-4).
///
/// `#[non_exhaustive]`: v1.1 adds `Cmk` / `Vault` / `AwsKms` / `GcpKms` /
/// `AzureKeyVault` variants behind the GA [`KeySource`] trait WITHOUT a
/// breaking change (per ADR-216 §D-2 — the named-not-stubbed v1.1 roadmap).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "kebab-case")]
pub enum KeySourceKind {
    /// The v1 source: a `SecretsProviderKeySource` over the OS keyring
    /// (prod) or env (dev). The only kind implemented at v1.0-α.
    #[default]
    SecretsProvider,
}

/// Which [`SecretsProvider`] backs the [`KeySourceKind::SecretsProvider`]
/// source (ADR-216 §D-4). Ignored for v1.1 KMS sources.
///
/// `#[non_exhaustive]`: forward-binds against any future provider variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[non_exhaustive]
#[serde(rename_all = "kebab-case")]
pub enum SecretsProviderKind {
    /// The OS keyring (macOS Keychain / Windows Credential Manager / Linux
    /// Secret Service) — the production default. Requires the
    /// `arcgraph-cli` `os-keyring` feature (which activates
    /// `arcgraph-core/os-keyring`); without it, a config selecting
    /// `os-keyring` fails closed at startup with a build-feature hint.
    #[default]
    OsKeyring,
    /// Environment-variable-backed provider — DEVELOPMENT ONLY (emits an
    /// `unsafe_for_prod=true` warning on construction).
    Env,
}

/// Operator config for WAL-at-rest encryption (ADR-216 §D-4, verbatim).
///
/// Deserialized from operator config (`#[serde(deny_unknown_fields)]` per
/// code-quality policy — misspellings reject at startup). When [`Self::enabled`],
/// `build_durable` constructs the selected [`KeySource`], does the
/// ADR-216 §D-2 bootstrap-sidecar dance, and wires the resulting
/// `WalEncryption` into BOTH the WAL writer AND the recovery readers.
#[derive(Debug, Clone, Default, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WalEncryptionConfig {
    /// Master switch. **Default `false` at v1.0-α** so existing deployments
    /// and tests are unaffected until an operator opts in.
    ///
    /// OQ-2: the GA default (secure-by-default `true` vs opt-in `false`) is
    /// an OPERATOR decision, flagged to the Director — v1.0-α stays
    /// `false`. Flipping it has a migration cost (existing plaintext WALs
    /// become mixed-WAL on upgrade — already supported, but operationally
    /// surprising), and "encryption at rest" compliance claims require an
    /// explicit opt-in. Do NOT flip this default without the operator's
    /// GA-prep ruling (ADR-216 §Open-questions OQ-2).
    #[serde(default)]
    pub enabled: bool,
    /// Which [`KeySource`] to construct. v1: `secrets-provider`.
    #[serde(default)]
    pub key_source: KeySourceKind,
    /// Provider sub-selection for the secrets-provider source
    /// (`os-keyring` | `env`); ignored for v1.1 KMS sources.
    #[serde(default)]
    pub secrets_provider: SecretsProviderKind,
}

// NOTE (OQ-2): `WalEncryptionConfig::default()` (derived) yields
// `enabled = false` — the v1.0-α opt-in default. The GA default
// (secure-by-default `true` vs opt-in `false`) is an OPERATOR decision,
// flagged to the Director; do NOT flip the `bool` default without the
// GA-prep ruling (ADR-216 §Open-questions OQ-2). The derive is equivalent
// to the prior hand-written impl (bool defaults to false; the kind enums
// carry `#[default]`); clippy::derivable_impls preferred the derive.

impl WalEncryptionConfig {
    /// Construct the selected [`KeySource`] (ADR-216 §D-4). Returns an
    /// `Arc<dyn KeySource>` ready for the bootstrap-sidecar dance.
    ///
    /// # Errors
    ///
    /// Returns [`anyhow::Error`] if the selected provider cannot be
    /// constructed (e.g. `os-keyring` selected without the build feature).
    fn build_key_source(&self) -> Result<Arc<dyn KeySource>> {
        match self.key_source {
            KeySourceKind::SecretsProvider => {
                let provider: Arc<dyn SecretsProvider> = self.build_secrets_provider()?;
                let provider_tag = match self.secrets_provider {
                    SecretsProviderKind::OsKeyring => "os-keyring",
                    SecretsProviderKind::Env => "env",
                };
                Ok(Arc::new(SecretsProviderKeySource::new(
                    provider,
                    provider_tag,
                    // First boot uses KEK v1; KEK rotation bumps this via
                    // the KeySource rotation path (ADR-216 §D-3).
                    KekVersion::ONE,
                )))
            }
        }
    }

    /// Construct the backing [`SecretsProvider`] for the secrets-provider
    /// key source.
    fn build_secrets_provider(&self) -> Result<Arc<dyn SecretsProvider>> {
        match self.secrets_provider {
            SecretsProviderKind::OsKeyring => build_os_keyring_provider(),
            SecretsProviderKind::Env => {
                // Dev provider — emits an `unsafe_for_prod=true` warning on
                // construction (the operator-facing pin).
                Ok(Arc::new(arcgraph_core::EnvSecretsProvider::new()))
            }
        }
    }
}

/// Construct the OS-keyring-backed provider. Feature-gated: the `keyring`
/// crate (Secret Service / D-Bus on Linux) is only compiled when the
/// `arcgraph-cli` `os-keyring` feature activates `arcgraph-core/os-keyring`.
/// Without it, selecting `os-keyring` fails CLOSED at startup with an
/// actionable build hint (NEVER a silent plaintext fallback, ADR-033).
#[cfg(feature = "os-keyring")]
fn build_os_keyring_provider() -> Result<Arc<dyn SecretsProvider>> {
    Ok(Arc::new(arcgraph_core::OsKeyringProvider::new()))
}

#[cfg(not(feature = "os-keyring"))]
fn build_os_keyring_provider() -> Result<Arc<dyn SecretsProvider>> {
    bail!(
        "WAL encryption selected `secrets_provider = os-keyring`, but this \
         binary was built WITHOUT the `os-keyring` feature. Rebuild with \
         `cargo build -p arcgraph-cli --features os-keyring`, or set \
         `secrets_provider = env` for development (UNSAFE FOR PRODUCTION). \
         Refusing to start rather than silently writing plaintext WAL \
         (ADR-033 fail-closed)."
    )
}

/// Process-lifetime owner of the durable substrate's [`WalWriter`] thread.
///
/// The WAL writer thread is owned by a [`WalWriter`] whose `Drop` drains
/// the pending batch, fsyncs, and joins the thread (graceful teardown). The
/// [`CrudStore`] + [`TxnManager`] hold cloneable [`arcgraph_storage::wal::WalHandle`]s
/// — but if the owning [`WalWriter`] is dropped, the writer thread shuts
/// down and every subsequent `append` fails with `WalUnavailable`.
///
/// Therefore the caller **MUST hold this guard for the full lifetime of the
/// server loop** (the binaries bind it to a scope-lived local that outlives
/// `serve_stdio` / the Bolt accept loop). Dropping it early would silently
/// make commits fail. In [`BootstrapMode::InMemory`] there is no WAL and
/// the guard holds `None`.
///
/// # Inter-process lock (#886, ADR-183 Strict-tier)
///
/// In durable mode the guard ALSO owns the [`DataDirLock`] — the exclusive
/// advisory lock on `<data_dir>/LOCK` taken at the top of `build_durable`
/// (before `pages.db` / the WAL are opened). A durable store is single-process:
/// a second `arcgraph serve --data <SAMEDIR>` is refused at bootstrap rather
/// than silently interleaving WAL appends and bricking the store on the next
/// restart (#886). Field order matters: `writer` is declared before `lock`, so
/// on `Drop` the WAL writer drains + fsyncs + joins FIRST and only then is the
/// inter-process lock released — the dir is never released to the next opener
/// while this process is still flushing. The OS also releases the lock on
/// process death (`flock` / `share_mode` semantics), so a crash never bricks the
/// dir. In-memory mode has no shared on-disk state → `lock` is `None`.
#[must_use = "the DurabilityGuard owns the WAL writer thread; dropping it early shuts the WAL down"]
pub struct DurabilityGuard {
    /// SVC-1 / #849 / ADR-229 — graceful-shutdown checkpoint. `Some` when
    /// durable AND `checkpoint_on_shutdown` is enabled. Fired FIRST on
    /// `Drop` (before the writer drains) so a graceful shutdown persists a
    /// full-state checkpoint — bounding the NEXT restart's recovery. A
    /// checkpoint failure on shutdown is logged, not propagated (the
    /// process is exiting; the WAL still holds the durable prefix, so the
    /// next restart falls back to a from-zero replay — correct, just
    /// slower). DROP ORDER LOAD-BEARING — checkpointer fires before
    /// writer drains, writer before lock releases.
    checkpointer: Option<DurableCheckpointer>,
    /// Operator root eligible for INV-M5.5 predecessor cleanup after this
    /// process establishes a post-swap successor checkpoint.
    generation_cleanup_root: Option<PathBuf>,
    /// `Some` in durable mode (owns the writer thread); `None` in-memory.
    /// DROP ORDER LOAD-BEARING — writer MUST precede lock; do not reorder
    /// (#886 shutdown-window durability).
    writer: Option<WalWriter>,
    /// Production M4 extent owners retained after replay so live reads,
    /// checkpoint routing, and diagnostics use the exact same directory and
    /// data caches that bootstrap populated.
    extent_stores: BTreeMap<(TenantId, u16), ProductionExtentRuntime>,
    /// Shared direct owner-row substrate used by recovery and live Phase-3
    /// publication. Present only for a v6/M4 generation.
    owner_rows: Option<Arc<arcgraph_storage::OwnerRowRegistry>>,
    /// Production property/TEL paired allocators, seeded from their durable
    /// directory ledgers on every open so no restart can reuse a live extent.
    affinity_allocators: BTreeMap<TenantId, arcgraph_storage::extent::PairedAffinityAllocator>,
    /// Immutable generation read epoch held for the complete served lifetime.
    /// The production cleanup reaper shares its registry and cannot unlink a
    /// predecessor while a pre-swap server still owns this pin.
    generation_pin: Option<crate::data_dir_migration::PinnedGenerationReader>,
    /// `Some` in durable mode (the exclusive inter-process lock on the data
    /// dir, #886); `None` in-memory.
    /// DROP ORDER LOAD-BEARING — writer MUST precede lock; do not reorder
    /// (#886 shutdown-window durability).
    lock: Option<DataDirLock>,
}

/// One production `(tenant, store)` extent owner opened by durable bootstrap.
///
/// This handle is intentionally small and cloneable: recovery, live I/O, and
/// the checkpointer all share the same directory/data/DPT owners. Tests use
/// it as a wiring oracle instead of constructing a parallel harness.
#[derive(Clone)]
pub struct ProductionExtentRuntime {
    directory: Arc<arcgraph_storage::extent::ExtentDirectory>,
    data: Arc<arcgraph_storage::extent::ExtentDataPageStore>,
    dpt: Arc<DirtyPageTable>,
}

impl ProductionExtentRuntime {
    /// Durable address directory used by the production replay path.
    #[must_use]
    pub fn directory(&self) -> &Arc<arcgraph_storage::extent::ExtentDirectory> {
        &self.directory
    }

    /// Data-page store whose live I/O resolves through [`Self::directory`].
    #[must_use]
    pub fn data(&self) -> &Arc<arcgraph_storage::extent::ExtentDataPageStore> {
        &self.data
    }

    /// Shared production DPT used by replay, live extent apply, and the
    /// write-behind checkpointer. Exposed as a cloneable diagnostic handle so
    /// crash gates can verify that aborted provisional placement made no
    /// flush-eligible mutation.
    #[must_use]
    pub fn dirty_page_table(&self) -> &Arc<DirtyPageTable> {
        &self.dpt
    }

    /// Apply a committed extent allocation to the same production owners
    /// that WAL recovery and the write-behind checkpointer use.
    pub fn apply_extent_alloc(
        &self,
        op: &arcgraph_storage::wal::DeltaOp,
    ) -> arcgraph_core::Result<arcgraph_storage::extent::ExtentApplyOutcome> {
        self.directory.apply_extent_alloc(op, self.dpt.as_ref())
    }

    /// Apply one committed page op through the production extent data store.
    pub fn apply_data_delta(
        &self,
        op: &arcgraph_storage::wal::DeltaOp,
        commit_lsn: Lsn,
    ) -> arcgraph_core::Result<arcgraph_storage::redo::RecoveryDeltaOutcome> {
        arcgraph_storage::redo::apply_recovery_delta(
            self.data.as_ref(),
            self.data.as_ref(),
            self.dpt.as_ref(),
            op,
            commit_lsn,
        )
    }
}

impl Drop for DurabilityGuard {
    fn drop(&mut self) {
        // SVC-1 / #849 / ADR-229 — graceful-shutdown checkpoint FIRST,
        // before the WAL writer drains + joins. Best-effort: a failure is
        // logged, never panics (the process is exiting; the WAL prefix is
        // still durable so the next restart recovers from-zero — correct).
        // The writer is still live here (dropped AFTER this via field
        // order), so `current_lsn` reflects every acked commit.
        let mut checkpoint_established = false;
        if let Some(cp) = self.checkpointer.take() {
            match cp.checkpoint() {
                Ok(lsn) => {
                    checkpoint_established = true;
                    tracing::info!(
                        target: "arcgraph_cli::bootstrap",
                        checkpoint_lsn = lsn.raw(),
                        "graceful-shutdown checkpoint established (ADR-229 #849)",
                    );
                }
                Err(e) => tracing::error!(
                    target: "arcgraph_cli::bootstrap",
                    error = %e,
                    "graceful-shutdown checkpoint FAILED — next restart falls back to \
                     from-zero WAL replay (durable prefix intact, no data loss)",
                ),
            }
        }
        if checkpoint_established
            && let Some(root) = &self.generation_cleanup_root
            && let Err(error) = crate::data_dir_migration::resume_generation_cleanup(
                root,
                crate::data_dir_migration::production_cleanup_fault(),
            )
        {
            tracing::error!(
                target: "arcgraph_cli::bootstrap",
                error = %error,
                "post-checkpoint old-generation cleanup interrupted; next boot will resume",
            );
        }
        // `writer` + `lock` drop after this via field declaration order
        // (checkpointer, writer, lock) — writer drains + fsyncs + joins,
        // then the lock releases. Do NOT reorder the fields (#886).
    }
}

impl DurabilityGuard {
    /// Durable guard owning the WAL writer thread + the data-dir lock (#886).
    /// `checkpointer` is `Some` when the ADR-229 shutdown checkpoint is
    /// enabled (`WalCheckpointConfig::checkpoint_on_shutdown`).
    #[allow(clippy::too_many_arguments)]
    fn durable(
        writer: WalWriter,
        lock: DataDirLock,
        checkpointer: Option<DurableCheckpointer>,
        generation_cleanup_root: Option<PathBuf>,
        extent_stores: BTreeMap<(TenantId, u16), ProductionExtentRuntime>,
        owner_rows: Option<Arc<arcgraph_storage::OwnerRowRegistry>>,
        affinity_allocators: BTreeMap<TenantId, arcgraph_storage::extent::PairedAffinityAllocator>,
        generation_pin: crate::data_dir_migration::PinnedGenerationReader,
    ) -> Self {
        Self {
            checkpointer,
            generation_cleanup_root,
            writer: Some(writer),
            extent_stores,
            owner_rows,
            affinity_allocators,
            generation_pin: Some(generation_pin),
            lock: Some(lock),
        }
    }

    /// Ephemeral guard (no WAL, no lock; `--in-memory`).
    fn ephemeral() -> Self {
        Self {
            checkpointer: None,
            generation_cleanup_root: None,
            writer: None,
            extent_stores: BTreeMap::new(),
            owner_rows: None,
            affinity_allocators: BTreeMap::new(),
            generation_pin: None,
            lock: None,
        }
    }

    /// SVC-1 / #849 / ADR-229 — a cloneable checkpointer handle for a
    /// background interval trigger, or `None` in-memory / when disabled.
    /// The serve loop can spawn a background task (Tokio work-stealing
    /// pool, NOT the hot path) that calls `checkpoint()` on the
    /// `WalCheckpointConfig` interval. See the module docs; the shutdown
    /// checkpoint is fired automatically on `Drop`.
    #[must_use]
    pub fn checkpointer(&self) -> Option<DurableCheckpointer> {
        self.checkpointer.clone()
    }

    /// `true` iff this is a durable substrate (owns a WAL writer).
    #[must_use]
    pub fn is_durable(&self) -> bool {
        self.writer.is_some()
    }

    /// Immutable generation selected for this durable served lifetime.
    #[must_use]
    pub fn generation(&self) -> Option<&Path> {
        self.generation_pin
            .as_ref()
            .map(crate::data_dir_migration::PinnedGenerationReader::generation)
    }

    /// Production extent owner registered for `(tenant, store_id)`, if any.
    /// The returned handle shares state with replay and checkpoint routing.
    #[must_use]
    pub fn extent_store(&self, tenant: TenantId, store_id: u16) -> Option<ProductionExtentRuntime> {
        self.extent_stores.get(&(tenant, store_id)).cloned()
    }

    /// Production direct owner-row substrate for a v6 generation.
    #[must_use]
    pub fn owner_rows(&self) -> Option<&Arc<arcgraph_storage::OwnerRowRegistry>> {
        self.owner_rows.as_ref()
    }

    /// Production paired allocator whose physical counters were recovered
    /// from this tenant's durable property/TEL extent directories at open.
    #[must_use]
    pub fn affinity_allocator(
        &self,
        tenant: TenantId,
    ) -> Option<arcgraph_storage::extent::PairedAffinityAllocator> {
        self.affinity_allocators.get(&tenant).cloned()
    }

    /// Clone the production WAL handle. This is primarily useful to fault
    /// inject a committed v9 bundle through the same writer a real server
    /// uses; ordinary callers should commit through `TxnManager`.
    #[must_use]
    pub fn wal_handle(&self) -> Option<arcgraph_storage::wal::WalHandle> {
        self.writer.as_ref().map(WalWriter::handle)
    }

    /// Path of the exclusive inter-process lock this guard holds on the data
    /// dir (`<data_dir>/LOCK`, #886), or `None` in [`BootstrapMode::InMemory`]
    /// mode (no shared on-disk state → no lock). Diagnostics + test oracle.
    #[must_use]
    pub fn data_dir_lock_path(&self) -> Option<&Path> {
        self.lock.as_ref().map(DataDirLock::path)
    }

    /// The WAL writer's committed-fsync watermark — the highest WAL LSN
    /// known durable on disk (ADR-034 §Slice B crash-consistency boundary).
    /// `None` in [`BootstrapMode::InMemory`] mode. Used by the durability
    /// regression tests to assert acked Strict commits are at or below the
    /// watermark before `commit()` returns.
    ///
    /// # Post-recovery divergence (ADR-183 advisory G)
    ///
    /// This watermark tracks the **current process's** WAL framing position,
    /// not a globally-monotonic logical clock that persists across restarts.
    /// `build_durable` uses plain [`WalWriter::spawn`] (not `spawn_from`), so
    /// after a restart the WAL framing-LSN counter starts fresh from `0` for the
    /// new process — whereas `recover_from_wal` has already advanced
    /// [`TxnManager::current_lsn`] to the recovered
    /// bundles' `commit_lsn`. So immediately post-recovery `last_durable_lsn()`
    /// (the in-process framing watermark) reads **below** the logical
    /// `commit_lsn` of records replayed from the prior process's WAL. Those
    /// replayed records are nonetheless fully durable — they live on the
    /// `pages.db` page store plus the replayed WAL segments. The watermark is
    /// therefore meaningful only as a crash-consistency boundary for commits
    /// issued *within this process's lifetime*; do **not** read it as the
    /// high-water mark of all durable data across restarts. Cross-restart record
    /// durability is proven separately by the K-3 10K-cycle crash campaign
    /// (`k3_10k_crash_cycle.rs`), not by this watermark. (See also the
    /// plain-`spawn`-vs-`spawn_from` rationale in `build_durable` §6.)
    #[must_use]
    pub fn last_durable_lsn(&self) -> Option<Lsn> {
        // `last_durable_lsn` lives on `WalHandle`; `handle()` is a cheap
        // clone that shares the same `committed_fsync_watermark` Arc.
        self.writer.as_ref().map(|w| w.handle().last_durable_lsn())
    }

    /// Quiesce an offline M3 migration while retaining the data-dir lock.
    /// Establishes the final full checkpoint, then shuts down the v8 WAL
    /// writer before returning. The caller owns the returned lock through the
    /// beside-build and `CURRENT` switch.
    pub fn quiesce_for_migration(mut self) -> Result<(DataDirLock, Lsn)> {
        let checkpointer = self
            .checkpointer
            .take()
            .context("offline migration requires the durable checkpointer")?;
        let migration_lsn = checkpointer.checkpoint()?;
        let writer = self
            .writer
            .take()
            .context("offline migration requires the durable WAL writer")?;
        writer
            .shutdown()
            .context("shutdown v8 WAL writer for migration")?;
        let lock = self
            .lock
            .take()
            .context("offline migration requires the data-dir lock")?;
        Ok((lock, migration_lsn))
    }
}

/// SVC-1 / #849 / ADR-229 — a durable-mode handle that can establish a
/// full-state WAL checkpoint on demand (graceful shutdown, or a
/// background interval trigger). Holds cloneable `Arc`s onto every
/// WAL-reconstructed owner + the catalog buffer pool + the data-dir, so
/// [`Self::checkpoint`] can snapshot the full state without re-plumbing
/// `build_durable`'s internals.
///
/// The frontier is `TxnManager::current_lsn()` (the highest committed +
/// visible LSN) at the moment of the snapshot — every effect at/below it
/// is durable in the WAL (Strict-tier commits fsync before `commit`
/// returns), so the snapshot + the frontier are a crash-consistent pair.
/// A concurrent commit that advances the watermark AFTER the frontier is
/// read is simply replayed from the WAL on the next restart (correct: it
/// is `> checkpoint_lsn`).
///
/// NOT the hot path (design-v2 §4.1): `checkpoint` runs on the graceful
/// shutdown thread or a background task, never a foreground commit.
#[derive(Clone)]
pub struct DurableCheckpointer {
    data_dir: PathBuf,
    buffer_pool: Arc<BufferPool>,
    txn_manager: Arc<TxnManager>,
    primary_pages: Arc<PrimaryPageStore>,
    record_pages: Arc<RecordPageStore>,
    blob_store: Arc<BlobStore>,
    allocator: Arc<PageAllocator>,
    crud: Arc<CrudStore>,
    allocator_seed: Arc<dyn AllocatorSeedHandle>,
    intern: Arc<InternTable>,
    idempotency: Arc<IdempotencyStore>,
    permissions: Arc<PermissionIndex>,
    wal_handle: arcgraph_storage::wal::WalHandle,
    m3_write_behind: Option<Arc<arcgraph_storage::WriteBehindCheckpointer>>,
    /// SVC-1 / #849 / ADR-229 BLOCK-3 — producer serialization mutex,
    /// SHARED across every clone (the interval task holds a clone; the
    /// shutdown `Drop` hook holds another). Held for the WHOLE `checkpoint`
    /// (capture + snapshot-write + sidecar-write), so two producers can
    /// never interleave and diverge the (snapshot, sidecar) pair — one
    /// establishes fully before the other begins. Paired with per-write
    /// unique tmp filenames (`snapshot.rs::unique_snapshot_tmp`) for the
    /// belt-and-suspenders no-clobber guarantee.
    producer_mutex: Arc<parking_lot::Mutex<()>>,
}

/// Exact live-owner observations used by the invisible-build isolation gate.
#[cfg(feature = "fault-injection")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuildIsolationCensus {
    pub generation: PathBuf,
    pub buffer_pool_pages: usize,
    pub dpt: Vec<arcgraph_storage::DirtyPageSnapshot>,
    pub doublewrite: Vec<arcgraph_storage::checkpoint::DoublewriteKey>,
    pub checkpointer_routes: BTreeSet<(Option<TenantId>, u16)>,
}

/// INV-M5.10 armed OQ-G negative control: `ARCGRAPH_M5_ROUTE_BUILD_OQG_LIVE`
/// simulates one invisible-build object entering the live retained-owner
/// census, but ONLY while a `*.building` sibling generation exists — the
/// gate's baseline checkpoints (no build in flight) stay unpolluted, so the
/// baseline/mid count-equality assertion is exactly what goes RED.
#[cfg(feature = "fault-injection")]
fn armed_oqg_build_leak(data_dir: &Path) -> bool {
    if std::env::var_os("ARCGRAPH_M5_ROUTE_BUILD_OQG_LIVE").is_none() {
        return false;
    }
    let Some(root) = data_dir.parent() else {
        return false;
    };
    std::fs::read_dir(root).is_ok_and(|entries| {
        entries
            .flatten()
            .any(|entry| entry.file_name().to_string_lossy().ends_with(".building"))
    })
}

/// SVC-1 / #849 / ADR-229 — spawn a background interval checkpointer on
/// the Tokio work-stealing pool (NOT the thread-per-core hot path,
/// design-v2 §4.1). Fires a full-state checkpoint every
/// `config.interval_seconds` (the wall-clock cap that bounds the segment
/// backlog by time). The actual checkpoint I/O runs on `spawn_blocking`
/// so it never blocks the async reactor.
///
/// Returns `None` (no task spawned) when the interval trigger is disabled
/// (`config.interval_seconds == 0`) or `checkpointer` is `None`
/// (in-memory). The returned [`tokio::task::JoinHandle`] runs until the
/// process exits; the serve loop should hold it for the server's
/// lifetime (dropping it does NOT cancel the task — bind it to a
/// scope-lived local, or `abort()` it on shutdown).
///
/// SVC-1 P2 (#1365): BOTH triggers are wired. A checkpoint fires when
/// EITHER (a) the WAL bytes appended since the last checkpoint cross
/// `interval_bytes` (the byte trigger — bounds steady-state WAL SIZE +
/// recovery backlog at high write rates), OR (b) `interval_seconds` has
/// elapsed since the last checkpoint (the time cap — bounds the segment
/// backlog by wall-clock even for a low-write store). The byte trigger is
/// the one that actually holds the #849 167 GB-at-10M line: at a high
/// ingest rate the 5-min time cap alone would let ~gigabytes accrete
/// between checkpoints, whereas the byte trigger fires as soon as a
/// configured budget of WAL is written.
///
/// Implementation: a single loop on a SHORT poll interval (min of the time
/// cap and a fixed poll granularity) evaluates both thresholds against the
/// WAL-bytes-appended gauge + the last-checkpoint timestamp, so the byte
/// trigger has bounded latency without a dedicated per-write hook on the hot
/// path. Each checkpoint runs on `spawn_blocking` (never the async reactor);
/// a successful checkpoint also reclaims WAL segments below the new frontier
/// + gc's MVCC versions (inside `DurableCheckpointer::checkpoint`).
///
/// Returns `None` (no task) when the interval trigger is disabled
/// (`interval_bytes == 0 && interval_seconds == 0`) or `checkpointer` is
/// `None` (in-memory). The returned handle runs for the server's lifetime.
#[must_use]
pub fn spawn_interval_checkpointer(
    checkpointer: Option<DurableCheckpointer>,
    config: arcgraph_storage::config::WalCheckpointConfig,
) -> Option<tokio::task::JoinHandle<()>> {
    let cp = checkpointer?;
    if !config.interval_enabled() {
        return None;
    }
    // Poll granularity: 1 s — fine enough that the byte trigger fires with
    // ≤1 s latency, coarse enough to be negligible overhead (one atomic load
    // + two comparisons per second). If a caller sets `interval_seconds` to
    // an even smaller value, cap the poll at it so the time trigger still
    // fires on schedule (a sub-second time cap is only meaningful in tests).
    let poll = std::time::Duration::from_secs(if config.interval_seconds > 0 {
        config.interval_seconds.min(1)
    } else {
        1
    });
    Some(tokio::spawn(async move {
        let mut ticker = tokio::time::interval(poll);
        // Skip the immediate first tick (t=0 checkpoint on an empty store is
        // wasteful).
        ticker.tick().await;
        // Byte + time baselines, updated after each successful checkpoint.
        let mut bytes_at_last_checkpoint: u64 = cp.wal_bytes_appended();
        let mut last_checkpoint = tokio::time::Instant::now();
        loop {
            ticker.tick().await;

            let wal_bytes_now = cp.wal_bytes_appended();
            let bytes_since = wal_bytes_now.saturating_sub(bytes_at_last_checkpoint);
            let elapsed_secs = last_checkpoint.elapsed().as_secs();
            let byte_fire = config.byte_threshold_reached(bytes_since);
            let time_fire = config.time_threshold_reached(elapsed_secs);
            if !byte_fire && !time_fire {
                continue;
            }
            let trigger = if byte_fire { "byte" } else { "time" };

            let cp_run = cp.clone();
            match tokio::task::spawn_blocking(move || cp_run.checkpoint()).await {
                Ok(Ok(lsn)) => {
                    tracing::info!(
                        target: "arcgraph_cli::bootstrap",
                        checkpoint_lsn = lsn.raw(),
                        trigger,
                        bytes_since,
                        elapsed_secs,
                        "interval checkpoint established (ADR-229 #849 P2)",
                    );
                    // Advance both baselines. Re-read the byte gauge AFTER the
                    // checkpoint so bytes written DURING the checkpoint count
                    // toward the NEXT interval (never dropped).
                    bytes_at_last_checkpoint = cp.wal_bytes_appended();
                    last_checkpoint = tokio::time::Instant::now();
                }
                Ok(Err(e)) => tracing::error!(
                    target: "arcgraph_cli::bootstrap",
                    error = %e,
                    trigger,
                    "interval checkpoint FAILED (WAL prefix intact; next restart from-zero)",
                ),
                Err(join_err) => tracing::error!(
                    target: "arcgraph_cli::bootstrap",
                    error = %join_err,
                    "interval checkpoint task panicked",
                ),
            }
        }
    }))
}

impl DurableCheckpointer {
    /// SVC-1 P2 (#1365 / ADR-229) — total WAL record-data bytes durably
    /// appended by this process's writer so far. The interval checkpointer's
    /// byte trigger reads this to decide when the WAL-since-checkpoint has
    /// grown past `interval_bytes`. Cheap (one atomic load off the shared
    /// `WalHandle`).
    #[must_use]
    pub fn wal_bytes_appended(&self) -> u64 {
        self.wal_handle.wal_bytes_appended()
    }

    /// Census every live physical checkpoint structure without consulting an
    /// invisible build directory. Compiled only for the release fault lane.
    #[cfg(feature = "fault-injection")]
    pub fn build_isolation_census(&self) -> Result<BuildIsolationCensus> {
        let (dpt, checkpointer_routes) = self.m3_write_behind.as_ref().map_or_else(
            || (Vec::new(), BTreeSet::new()),
            |write_behind| {
                (
                    write_behind.metadata_dpt_snapshot(),
                    write_behind.route_census(),
                )
            },
        );
        let doublewrite = arcgraph_storage::DoublewriteArea::new(&self.data_dir)
            .valid_batch_keys()
            .with_context(|| format!("census doublewrite at {}", self.data_dir.display()))?;
        Ok(BuildIsolationCensus {
            generation: self.data_dir.clone(),
            buffer_pool_pages: self.buffer_pool.mapped(),
            dpt,
            doublewrite,
            checkpointer_routes,
        })
    }

    /// Run the production full-state producer even for a v9 live generation.
    /// The gate follows it with the ordinary incremental producer before the
    /// builder is released, restoring the v9 sidecar posture.
    #[cfg(feature = "fault-injection")]
    pub fn full_checkpoint_for_build_isolation_gate(
        &self,
    ) -> Result<arcgraph_storage::checkpoint::CheckpointReport> {
        let _producer = self.producer_mutex.lock();
        let snap = arcgraph_storage::checkpoint::CheckpointSnapshot {
            txn: &self.txn_manager,
            primary_pages: &self.primary_pages,
            record_pages: &self.record_pages,
            blob: &self.blob_store,
            allocator_seed: self.allocator_seed.as_ref(),
            intern: &self.intern,
            idempotency: &self.idempotency,
            permissions: &self.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        let allocator = Arc::clone(&self.allocator);
        let crud = Arc::clone(&self.crud);
        let mut report = arcgraph_storage::checkpoint::checkpoint(
            &self.data_dir,
            &self.buffer_pool,
            &snap,
            move || {
                let mut advances = allocator.snapshot_advances();
                advances.extend(crud.snapshot_allocator_advances());
                advances
            },
            self.wal_handle.last_durable_lsn(),
        )
        .with_context(|| {
            format!(
                "establish full build-isolation checkpoint at {}",
                self.data_dir.display()
            )
        })?;
        if armed_oqg_build_leak(&self.data_dir) {
            report.counts.mvcc_records += 1;
        }
        Ok(report)
    }

    /// Run the normal production incremental producer and decode its complete
    /// retained-owner (OQ-G) census for the gate.
    #[cfg(feature = "fault-injection")]
    pub fn incremental_checkpoint_for_build_isolation_gate(
        &self,
    ) -> Result<arcgraph_storage::checkpoint::IncrementalCheckpointMetadata> {
        let checkpoint_lsn = self.checkpoint()?;
        let sidecar = arcgraph_storage::read_latest_sidecar(&self.data_dir)
            .context("read build-isolation incremental sidecar")?
            .context("build-isolation incremental sidecar is absent")?;
        ensure!(
            sidecar.incremental_metadata && sidecar.checkpoint_lsn == checkpoint_lsn,
            "ordinary v9 checkpointer did not restore incremental sidecar posture"
        );
        let snap = arcgraph_storage::checkpoint::CheckpointSnapshot {
            txn: &self.txn_manager,
            primary_pages: &self.primary_pages,
            record_pages: &self.record_pages,
            blob: &self.blob_store,
            allocator_seed: self.allocator_seed.as_ref(),
            intern: &self.intern,
            idempotency: &self.idempotency,
            permissions: &self.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        let mut metadata = arcgraph_storage::checkpoint::read_incremental_metadata(
            &self.data_dir,
            &snap,
            checkpoint_lsn,
            sidecar.metadata_generation,
        )
        .with_context(|| {
            format!(
                "decode build-isolation incremental census at {}",
                self.data_dir.display()
            )
        })?;
        if armed_oqg_build_leak(&self.data_dir) {
            metadata.counts.mvcc_records += 1;
        }
        Ok(metadata)
    }

    /// Gate-only negative control: bind one phantom page into the LIVE
    /// buffer pool, exactly as a leaked invisible-build page would appear to
    /// the `buffer_pool_pages` census (INV-M5.10).
    #[cfg(feature = "fault-injection")]
    pub fn build_isolation_gate_map_phantom_page(&self, page_no: u64) -> Result<()> {
        self.buffer_pool
            .map_phantom_page_for_build_isolation_gate(arcgraph_core::PageId::new(page_no))
            .context("map phantom build page into live buffer pool")
    }

    /// Gate-only negative control: register one synthetic route on the LIVE
    /// write-behind checkpointer so the INV-M5.10 route census differs.
    #[cfg(feature = "fault-injection")]
    pub fn build_isolation_gate_inject_route(
        &self,
        tenant: Option<TenantId>,
        store_id: u16,
    ) -> Result<()> {
        let write_behind = self
            .m3_write_behind
            .as_ref()
            .context("live checkpointer has no write-behind routes to pollute")?;
        write_behind.inject_route_for_build_isolation_gate((tenant, store_id));
        Ok(())
    }

    /// Establish a full-state checkpoint at the current committed
    /// frontier. Flushes the catalog buffer pool + writes the full-state
    /// snapshot + the frontier sidecar, all crash-atomically
    /// (both-or-neither; a crash mid-checkpoint leaves the PREVIOUS
    /// checkpoint valid). Returns the established frontier LSN.
    ///
    /// BLOCK-1/2 (consistency): the frontier read + full-state capture +
    /// allocator drain all run UNDER `TxnManager::checkpoint_freeze` inside
    /// `checkpoint::checkpoint` — no commit can allocate an id absent from
    /// the snapshot, nor leave a not-yet-WAL-durable page image in it.
    /// BLOCK-3 (no divergent pair): the `producer_mutex` (shared across
    /// clones) serializes the WHOLE producer, so the interval task and the
    /// shutdown Drop hook can never interleave their snapshot/sidecar
    /// writes.
    ///
    /// SVC-1 P2 (#1365 / ADR-229 §Segment reclamation): AFTER the checkpoint
    /// is durably established, the WAL segments fully below the new frontier
    /// are reclaimed (durably deleted) and MVCC versions below the oldest
    /// active snapshot are gc'd. Reclamation runs ONLY after the sidecar is
    /// durable (both-or-neither) — a segment we delete has its committed
    /// effects captured in the just-established snapshot. This is what
    /// actually BOUNDS WAL size (P1 bounded recovery TIME; P2 bounds the
    /// on-disk WAL + churn). A reclamation/gc failure is logged, not
    /// propagated — the checkpoint itself already succeeded, so the frontier
    /// LSN is returned; the next pass reclaims what this one missed.
    pub fn checkpoint(&self) -> Result<Lsn> {
        // BLOCK-3: serialize concurrent producers (interval + shutdown).
        let _producer = self.producer_mutex.lock();

        let snapshot_last_wal_lsn = self.wal_handle.last_durable_lsn();
        let snap = arcgraph_storage::checkpoint::CheckpointSnapshot {
            txn: &self.txn_manager,
            primary_pages: &self.primary_pages,
            record_pages: &self.record_pages,
            blob: &self.blob_store,
            allocator_seed: self.allocator_seed.as_ref(),
            intern: &self.intern,
            idempotency: &self.idempotency,
            permissions: &self.permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        // The allocator-advance drain closure runs UNDER the commit-freeze
        // (inside `checkpoint`), AFTER the frontier read (BLOCK-1). Union of
        // page-kind (PageAllocator) + Node/Rel (CrudStore) high-waters.
        let allocator = Arc::clone(&self.allocator);
        let crud = Arc::clone(&self.crud);
        if let Some(write_behind) = &self.m3_write_behind {
            let incremental_allocator = Arc::clone(&allocator);
            let incremental_crud = Arc::clone(&crud);
            let establish_txn = Arc::clone(&self.txn_manager);
            let establish_crud = Arc::clone(&crud);
            let establish_wal = self.wal_handle.clone();
            let report = arcgraph_storage::checkpoint::incremental_checkpoint(
                &self.data_dir,
                &self.buffer_pool,
                &snap,
                write_behind,
                move || {
                    let mut advances = incremental_allocator.snapshot_advances();
                    advances.extend(incremental_crud.snapshot_allocator_advances());
                    let deferred = incremental_crud.deferred_v9_boundary();
                    (advances, deferred)
                },
                move |horizon| {
                    establish_wal.flush().map_err(|error| {
                        arcgraph_storage::checkpoint::CheckpointError::Io(std::io::Error::other(
                            format!(
                                "flush WAL through v9 establishment horizon {}: {error}",
                                horizon.raw()
                            ),
                        ))
                    })?;
                    let durable = establish_wal.last_durable_lsn();
                    if durable < horizon {
                        return Err(arcgraph_storage::checkpoint::CheckpointError::Corrupt {
                            reason: format!(
                                "WAL flush stopped at {} below v9 establishment horizon {}",
                                durable.raw(),
                                horizon.raw()
                            ),
                        });
                    }
                    let _freeze = establish_txn.checkpoint_freeze();
                    establish_crud
                        .drain_deferred_v9_applies()
                        .map_err(|error| {
                            arcgraph_storage::checkpoint::CheckpointError::Io(
                                std::io::Error::other(format!(
                                    "drain durable v9 applies before establishment: {error}"
                                )),
                            )
                        })?;
                    Ok(durable)
                },
            )
            .with_context(|| {
                format!(
                    "establish M3 incremental checkpoint at {}",
                    self.data_dir.display()
                )
            })?;
            self.reclaim_and_gc(report.redo_lsn);
            return Ok(report.checkpoint_lsn);
        }
        let report = arcgraph_storage::checkpoint::checkpoint(
            &self.data_dir,
            &self.buffer_pool,
            &snap,
            move || {
                let mut advances = allocator.snapshot_advances();
                advances.extend(crud.snapshot_allocator_advances());
                advances
            },
            snapshot_last_wal_lsn,
        )
        .with_context(|| format!("establish checkpoint at {}", self.data_dir.display()))?;

        // ── SVC-1 P2: reclaim WAL + gc, ONLY after the checkpoint is durable ──
        // The checkpoint above is now fully established (snapshot + sidecar on
        // disk, both-or-neither). Every committed effect at/below
        // `report.checkpoint_lsn` is captured in the snapshot, so segments
        // fully below the frontier are safe to delete.
        self.reclaim_and_gc(report.checkpoint_lsn);

        Ok(report.checkpoint_lsn)
    }

    /// SVC-1 P2 (#1365 / ADR-229): reclaim WAL segments fully below
    /// `checkpoint_lsn` (durably deleting them) and gc MVCC versions below
    /// the oldest active snapshot. MUST be called only AFTER a checkpoint at
    /// `checkpoint_lsn` is durably established. Best-effort: a failure is
    /// logged and swallowed (the checkpoint already succeeded; the next pass
    /// retries). Never touches the active segment or one at/after the
    /// frontier (the reclaim module enforces that).
    fn reclaim_and_gc(&self, checkpoint_lsn: Lsn) {
        let wal_dir = self.data_dir.join(WAL_SUBDIR);
        match arcgraph_storage::wal::reclaim_segments_below(&wal_dir, checkpoint_lsn) {
            Ok(report) if !report.deleted_segments.is_empty() => tracing::info!(
                target: "arcgraph_cli::bootstrap",
                checkpoint_lsn = checkpoint_lsn.raw(),
                deleted_segments = report.deleted_segments.len(),
                bytes_freed = report.bytes_freed,
                "SVC-1 P2: reclaimed WAL segments below checkpoint frontier (WAL bounded)",
            ),
            Ok(_) => {} // nothing reclaimable this pass — normal
            Err(e) => tracing::error!(
                target: "arcgraph_cli::bootstrap",
                error = %e,
                checkpoint_lsn = checkpoint_lsn.raw(),
                "SVC-1 P2: WAL segment reclamation failed (segments intact; next pass retries)",
            ),
        }
        // Reclaim MVCC versions below the oldest active snapshot — the
        // `gc()` / `gc_with_prune_barrier` primitives had ZERO production
        // callers before this (#1005 churn leak). Same trigger as the
        // checkpoint. `gc()` derives its own anchor internally.
        let stats = self.txn_manager.gc();
        if stats.reclaimed > 0 {
            tracing::info!(
                target: "arcgraph_cli::bootstrap",
                reclaimed_versions = stats.reclaimed,
                pruned_keys = stats.pruned_keys,
                anchor = stats.anchor.raw(),
                "SVC-1 P2: gc'd MVCC versions below oldest active snapshot",
            );
        }
    }
}

/// #1404 M0.x — env var naming the MVCC drain's commit-count watermark: drive
/// a `gc()` pass every this-many commits on the durable serve path. Unset →
/// [`DEFAULT_GC_DRIVE_INTERVAL`].
const ENV_GC_DRIVE_INTERVAL: &str = "ARCGRAPH_GC_DRIVE_INTERVAL_COMMITS";

/// #1404 M0.x — default MVCC drain watermark (commits between driver-initiated
/// `gc()` passes). Sized so the drain fires frequently enough to keep the
/// resident superseded-version set bounded during sustained ingest, but rarely
/// enough that the `gc()` scan (one pass over the version-key set) is a
/// negligible amortized cost (~1 scan / 4096 commits). A value of `0` disables
/// the driver (legacy — `gc()` only on the checkpoint trigger).
const DEFAULT_GC_DRIVE_INTERVAL: u64 = 4096;

/// Resolve the MVCC drain watermark from [`ENV_GC_DRIVE_INTERVAL`], falling
/// back to [`DEFAULT_GC_DRIVE_INTERVAL`]. A parseable `0` explicitly disables
/// the driver (operator escape hatch); an unparseable value falls back to the
/// safe default (a bounded drain is better than none).
fn gc_drive_interval() -> u64 {
    std::env::var(ENV_GC_DRIVE_INTERVAL)
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(DEFAULT_GC_DRIVE_INTERVAL)
}

/// Bootstrap the per-process storage substrate the production adapters read
/// from, in the requested [`BootstrapMode`].
///
/// Returns the [`StorageBackend`] bundle the production adapters share and a
/// [`DurabilityGuard`] the caller MUST hold for the server loop's lifetime
/// (it owns the WAL writer thread in durable mode).
///
/// Construction order (durable mode):
/// 1. `<data_dir>` created, then an EXCLUSIVE inter-process advisory lock is
///    taken on `<data_dir>/LOCK` BEFORE any shared on-disk state is opened
///    (#886, ADR-183 Strict-tier): a second `arcgraph serve --data <SAMEDIR>`
///    is refused here rather than silently interleaving WAL appends and
///    bricking the store on the next restart. `<data_dir>/wal` created;
///    [`PosixPageIo`] over `<data_dir>/pages.db` backs the 256-frame catalog
///    [`BufferPool`].
/// 2. [`PosixPageIo`] over `<data_dir>/pages.db` backs the catalog
///    [`BufferPool`].
/// 3. Raw replay targets are allocated without a WAL writer: [`TxnManager`],
///    primary/record/blob page stores, allocator, intern table, idempotency
///    store.
/// 4. A fully-wired [`PageStoreTarget`] (ADR-183 R1) is built over those raw
///    handles.
/// 5. `recover_from_wal` replays committed bundles and any reported torn
///    terminal tail is durably truncated.
/// 6. A [`WalWriter`] spawns over `<data_dir>/wal`; its handle attaches to the
///    recovered [`TxnManager`], [`PrimaryIndex`], and [`CrudStore`].
/// 7. [`SystemCatalog::bootstrap`] registers
///    [`arcgraph_core::TenantId::DEFAULT`] after recovery/truncation
///    (re-registered idempotently on restart over an existing dir — a
///    redundant-but-harmless SYSTEM MVCC version; no page conflict, per
///    ADR-183 R3).
/// 8. Cold-start stats/TEL rebuilds run, then [`MultiTenantRouter`] +
///    [`InternTable`] + [`StorageBackend`] are returned.
///
/// In-memory mode (`--in-memory`) skips durable WAL/recovery steps and wires
/// [`InMemoryPageIo`] + `wal: None` (the prior v1.0-α posture).
///
/// # Errors
///
/// Returns [`anyhow::Error`] on data-dir / WAL-dir creation failure, a held
/// inter-process lock (another `arcgraph serve` already owns the durable
/// `--data` dir — #886; fail-fast before WAL replay / binding a listener),
/// [`PosixPageIo::open_or_create`] failure, [`WalWriter::spawn`] failure,
/// [`SystemCatalog::bootstrap`] failure, [`PrimaryIndex::new`] failure, or
/// WAL recovery failure (hard corruption — operator must intervene).
pub fn bootstrap_storage_backend(
    mode: &BootstrapMode,
) -> Result<(StorageBackend, DurabilityGuard)> {
    bootstrap_storage_backend_with_metrics(mode, None)
}

/// W28 Feature #582 (ADR-045) — [`bootstrap_storage_backend`] variant that
/// threads an observability sink into the [`CrudStore`] so the
/// `arcgraph_hot_vertex_warnings_total{tenant}` counter (design-v2 §10.2
/// line 721) fires on TEL/reverse-TEL overflow.
///
/// The `arcgraph serve` binary calls this with `Some(registry)` whenever
/// `--metrics-http` resolves to a non-empty bind and with `None` when
/// metrics are disabled. Per OBS-1 the `--metrics-http` default is ON
/// (`127.0.0.1:9090`), so `Some(registry)` is the DEFAULT path; the `None`
/// path is the explicit opt-out (`--metrics-http ""`), the zero-overhead
/// path identical to [`bootstrap_storage_backend`]. The sink is threaded
/// into the [`CrudStore`] on BOTH the durable (ADR-183) and the in-memory
/// bootstrap flows, so `--data` observes hot-vertex warnings just like
/// `--in-memory` (metrics on by default on both).
///
/// # Errors
///
/// See [`bootstrap_storage_backend`].
pub fn bootstrap_storage_backend_with_metrics(
    mode: &BootstrapMode,
    metrics_sink: Option<Arc<dyn MetricsSink>>,
) -> Result<(StorageBackend, DurabilityGuard)> {
    // ADR-216 §D-4: encryption defaults to OFF (OQ-2 — v1.0-α opt-in). The
    // serve binary calls the `_and_encryption` variant when an operator
    // config supplies a `WalEncryptionConfig`; legacy callers (tests, the
    // mcp-stdio binary) get the no-encryption posture unchanged.
    bootstrap_storage_backend_with_metrics_and_encryption(
        mode,
        metrics_sink,
        &WalEncryptionConfig::default(),
    )
}

/// ADR-216 §D-4 / #1180 — [`bootstrap_storage_backend_with_metrics`] variant
/// that ALSO threads a [`WalEncryptionConfig`] into the durable bootstrap.
///
/// When [`WalEncryptionConfig::enabled`], `build_durable` constructs the
/// selected [`KeySource`], performs the ADR-216 §D-2 bootstrap-sidecar
/// dance, and wires the resulting `WalEncryption` into BOTH the WAL writer
/// AND the recovery readers (encrypt-on-write without decrypt-on-recover is
/// unrecoverable WAL). When disabled (the v1.0-α default), the durable path
/// is byte-for-byte the prior plaintext-WAL posture. Encryption is a durable
/// concern only — `--in-memory` has no WAL, so the config is inert there.
///
/// # Errors
///
/// See [`bootstrap_storage_backend`]; additionally, a fail-closed startup
/// error if `enabled` but the KEK is unresolvable (ADR-033 — never serve
/// plaintext WAL silently).
pub fn bootstrap_storage_backend_with_metrics_and_encryption(
    mode: &BootstrapMode,
    metrics_sink: Option<Arc<dyn MetricsSink>>,
    wal_encryption: &WalEncryptionConfig,
) -> Result<(StorageBackend, DurabilityGuard)> {
    // SVC-2 / #1302: `adopt_legacy = false` — the default fail-closed posture.
    // Every existing caller (tests, mcp-stdio binary) keeps that posture; only
    // the `arcgraph serve --adopt-legacy-datadir` path opts in via the
    // `_and_adopt` entry below.
    bootstrap_storage_backend_with_metrics_encryption_and_adopt(
        mode,
        metrics_sink,
        wal_encryption,
        false,
    )
}

/// SVC-2 / #1302 — [`bootstrap_storage_backend_with_metrics_and_encryption`]
/// variant that ALSO threads the `adopt_legacy` operator opt-in
/// (`arcgraph serve --adopt-legacy-datadir`) into the durable data-dir
/// version guard.
///
/// When `adopt_legacy` is `true` AND the durable dir holds data but has no
/// `VERSION`, the guard restores a recognized pre-M3 MANIFEST's exact v3/v4
/// stamp, or stamps a genuinely pre-stamp (no-MANIFEST) beta dir as chained
/// v1 so the same boot migrates it forward. M3 v5 and unknown manifests are
/// never legacy-adopted. It also never rescues an *incompatible* stamped
/// version (the format really differs) — that still fails loud. When `false`
/// (the default for every path except the serve flag), the guard's fail-closed
/// posture is unchanged: an unstamped legacy dir is refused. Inert for
/// [`BootstrapMode::InMemory`] (no data dir).
///
/// # Errors
///
/// See [`bootstrap_storage_backend`]; additionally a
/// [`arcgraph_storage::DataDirVersionError`]-derived startup error if the
/// data-dir version guard refuses the dir.
pub fn bootstrap_storage_backend_with_metrics_encryption_and_adopt(
    mode: &BootstrapMode,
    metrics_sink: Option<Arc<dyn MetricsSink>>,
    wal_encryption: &WalEncryptionConfig,
    adopt_legacy: bool,
) -> Result<(StorageBackend, DurabilityGuard)> {
    let (backend, guard) = match mode {
        BootstrapMode::Durable { data_dir } => {
            build_durable(data_dir, metrics_sink, wal_encryption, adopt_legacy)?
        }
        BootstrapMode::InMemory => {
            // Encryption is a durable-WAL concern; `--in-memory` has no WAL,
            // so `wal_encryption` is intentionally not consulted here.
            // `adopt_legacy` is a durable-dir concern → inert here.
            build_in_memory(metrics_sink)?
        }
    };
    Ok((backend, guard))
}

/// #1513 (M5-D1b; `docs/design/M5D-REDESIGN-AMENDMENT.md` §10 Risk-2
/// ruling) — register every tenant in the MANIFEST `tenant_census` into
/// the served catalog at cold open, via the production
/// [`SystemCatalog::register_tenant`] path (the one the router's
/// `UnknownTenant` guard consults through `list_tenants`).
///
/// - **Idempotent / resumable:** `register_tenant` is a no-op for an
///   already-listed tenant (including DEFAULT, which
///   `SystemCatalog::bootstrap` installs), so re-running cold open —
///   or resuming after a crash mid-sweep — converges on the census set
///   with no duplicates.
/// - **Fail-loud:** a census carrying `TenantId::SYSTEM` is a corrupt
///   or foreign manifest; registration errors abort the boot.
/// - Names: the census (`Vec<u64>`, `manifest.rs`) carries ids only.
///   DEFAULT keeps its bootstrap name; other tenants use the loader's
///   `loaded-<raw>` convention (`m5_load::census_tenant_records`).
///
/// The `ARCGRAPH_M5_*` fault seams are cfg-gated to the
/// `fault-injection` feature (never shipped), bounded (skip / one-entry
/// subset / one-entry abort — no parking), and exist so the
/// `m5_tenant_registration_gate` RED-on-revert children can prove the
/// #1513 repro (register-nothing → `UnknownTenant`), the
/// census-authority subset red, and crash-mid-registration recovery.
fn register_census_tenants(
    catalog: &SystemCatalog,
    txn_manager: &TxnManager,
    census: &[u64],
) -> Result<()> {
    #[cfg(feature = "fault-injection")]
    if std::env::var_os("ARCGRAPH_M5_SKIP_CENSUS_REGISTRATION").is_some() {
        // Total-bypass mutant: the exact #1513 shape (loaded tenants
        // servable on disk, absent from the catalog).
        return Ok(());
    }
    let mut registered = 0_usize;
    for &raw in census {
        let tenant = TenantId::new(raw);
        if tenant == TenantId::SYSTEM {
            bail!(
                "MANIFEST tenant_census contains TenantId::SYSTEM ({raw}); SYSTEM is never a \
                 listable tenant — refusing to serve a corrupt census"
            );
        }
        let name = if tenant == TenantId::DEFAULT {
            "default".to_owned()
        } else {
            format!("loaded-{raw}")
        };
        catalog
            .register_tenant(txn_manager, tenant, &name)
            .with_context(|| format!("register census tenant {raw} into the served catalog"))?;
        registered += 1;
        #[cfg(feature = "fault-injection")]
        {
            if registered == 1
                && std::env::var_os("ARCGRAPH_M5_CRASH_MID_CENSUS_REGISTRATION").is_some()
            {
                // Crash-mid-registration fixture: die between the first
                // and second census entries; the next cold open must
                // complete the set (idempotent resume).
                eprintln!("ARCGRAPH_M5_CRASH_MID_CENSUS_REGISTRATION: aborting after 1 entry");
                std::process::abort();
            }
            if registered == 1
                && std::env::var_os("ARCGRAPH_M5_CENSUS_REGISTRATION_SUBSET").is_some()
            {
                // Subset mutant: register strictly fewer tenants than
                // the census — the census-authority gate must red.
                break;
            }
        }
    }
    tracing::info!(
        target: "arcgraph_cli::bootstrap",
        census_tenants = census.len(),
        registered,
        "durable bootstrap: MANIFEST tenant census registered into the served catalog (#1513)",
    );
    Ok(())
}

/// Build the **durable** substrate (ADR-183): [`PosixPageIo`] + WAL +
/// recover-on-startup. See [`bootstrap_storage_backend`] for the step list.
///
/// ADR-216 §D-4 / #1180: when `wal_encryption.enabled`, the KEK is
/// health-checked (fail-closed if absent), the `wal.dek` sidecar is
/// read-or-generated, the DEK is unwrapped, and the resulting
/// `WalEncryption` is wired into BOTH the recovery readers (§5) AND the WAL
/// writer (§6) — symmetric encrypt-on-write + decrypt-on-recover.
fn build_durable(
    data_dir: &Path,
    metrics_sink: Option<Arc<dyn MetricsSink>>,
    wal_encryption: &WalEncryptionConfig,
    adopt_legacy: bool,
) -> Result<(StorageBackend, DurabilityGuard)> {
    let operator_data_dir = data_dir.to_path_buf();
    // §1. Directory layout + EXCLUSIVE inter-process lock (#886, ADR-183
    //     Strict-tier). `create_dir_all` is idempotent — fresh start creates
    //     them; restart over an existing dir is a no-op.
    std::fs::create_dir_all(data_dir)
        .with_context(|| format!("create data dir {}", data_dir.display()))?;
    // Take the exclusive advisory lock on `<data_dir>/LOCK` BEFORE opening
    // `pages.db` / spawning the WAL, so a second `arcgraph serve --data
    // <SAMEDIR>` fails fast here instead of opening a second writer onto the
    // shared store (which interleaves WAL appends → `WalCorruption crc
    // mismatch` on the next restart, losing acknowledged Strict-tier commits —
    // #886). The lock is moved into the returned `DurabilityGuard` (held for
    // the server loop); the OS releases it on clean exit AND on process death
    // (`flock`/`share_mode`), so a crashed prior process does not brick the
    // dir for the next opener. `--data` is reachable with both transports via
    // the documented CLI (`--http` `conflicts_with` `--bolt`), so this guards a
    // real operator topology, not a contrived one.
    let data_lock = DataDirLock::acquire(data_dir)?;
    // INV-M5.5: resume only when a prior boot already established a durable
    // post-swap checkpoint. With only the migration checkpoint present this
    // is a no-op, preserving gen-v9 as M4 Slice-3a's recovery fallback.
    crate::data_dir_migration::resume_generation_cleanup(
        data_dir,
        crate::data_dir_migration::GenerationCleanupFault::None,
    )
    .context("resume interrupted old-generation cleanup")?;
    // M3's `CURRENT` rename is the generation commit point. Resolve it only
    // after holding the root lock, then direct every substrate below at the
    // selected generation while retaining the lock at the operator root.
    let generation_pin = crate::data_dir_migration::pin_current_generation(
        data_dir,
        crate::data_dir_migration::production_generation_pins(),
    )
    .context("pin CURRENT-selected production read epoch")?;
    let selected_generation = crate::data_dir_migration::current_generation(data_dir)?;
    ensure!(
        selected_generation.as_deref().unwrap_or(data_dir) == generation_pin.generation(),
        "CURRENT changed after production generation pin"
    );
    let generation_cleanup_root = selected_generation
        .as_deref()
        .filter(|generation| {
            generation.file_name()
                == Some(std::ffi::OsStr::new(
                    // INV-M5.22 rule 4: bootstrap stays CURRENT-driven; its
                    // one M4-fallback filter resolves the name through the
                    // generation-namespace registry, not a string literal.
                    crate::generation_namespace::GenerationTool::M4Migration.final_dir(),
                ))
        })
        .map(|_| operator_data_dir);
    let data_dir = generation_pin.generation();
    if selected_generation.is_some()
        && !arcgraph_storage::version_file_path(data_dir).exists()
        && let Some(manifest) = arcgraph_storage::read_data_dir_manifest(data_dir)
            .with_context(|| format!("read CURRENT-selected MANIFEST at {}", data_dir.display()))?
        && ((manifest.data_dir_version == arcgraph_storage::DATA_DIR_VERSION_DELTA_M3
            && manifest.wal_format == arcgraph_storage::manifest::WAL_FORMAT_DELTA_V9)
            || (manifest.data_dir_version == arcgraph_storage::DATA_DIR_VERSION_DIRECT_M4
                && manifest.wal_format == arcgraph_storage::manifest::WAL_FORMAT_DELTA_V10))
    {
        // Both offline generation migrations publish CURRENT before their
        // generation-local VERSION. M4 validates the complete selected store
        // set, checkpoint, empty WAL, MANIFEST, and LSN seed before it may
        // perform that last act; a half-built selected generation must fail
        // closed and remain unstamped.
        if manifest.data_dir_version == arcgraph_storage::DATA_DIR_VERSION_DIRECT_M4 {
            crate::data_dir_migration::resume_after_m4_swap(
                data_dir,
                crate::data_dir_migration::MigrationFault::None,
            )
            .context("resume validated VERSION=6 stamp for CURRENT-selected generation")?;
        } else {
            arcgraph_storage::stamp_data_dir(data_dir, manifest.data_dir_version).with_context(
                || {
                    format!(
                        "resume VERSION={} stamp for CURRENT-selected delta generation",
                        manifest.data_dir_version
                    )
                },
            )?;
        }
    }
    let wal_dir = data_dir.join(WAL_SUBDIR);
    std::fs::create_dir_all(&wal_dir)
        .with_context(|| format!("create WAL dir {}", wal_dir.display()))?;
    let pages_path = data_dir.join(PAGES_FILE);

    // Detect whether a prior process wrote to this WAL before §6 spawns this
    // session's writer (which would itself create a segment file). The
    // data-directory version guard below uses this as one signal that the
    // directory already contains durable state.
    let wal_preexisting = std::fs::read_dir(&wal_dir)
        .map(|mut entries| entries.next().is_some())
        .unwrap_or(false);

    // §1b. SVC-2 / #1302 — on-disk data-dir VERSION guard (upgrade-safety).
    //      Run BEFORE `pages.db` / the WAL are opened (§2 onward) so an
    //      operator who binary-swaps across a data-dir-format change fails
    //      LOUD here — with an actionable message pointing at the RIGHT
    //      recovery (`--adopt-legacy-datadir` for an unstamped legacy dir; a
    //      matching binary / restore for a version mismatch) — instead of
    //      misparsing `pages.db`. Mirrors the WAL-segment and catalog-page
    //      version guards (`SUPPORTED_WAL_FORMAT_VERSIONS` /
    //      `CATALOG_PAGE_VERSION`). Fail-closed by default.
    //
    //      `has_data` is the existing "does this dir already hold a durable
    //      store?" signal (`pages.db` present OR a non-empty WAL, the same
    //      `wal_preexisting` probe #843 uses; see the storage module's
    //      §`has_data` scope note re BM25/vector subdirs). It + `adopt_legacy`
    //      drive the policy in `check_or_stamp_data_dir`:
    //        - VERSION present + supported → clean no-op (proceed);
    //        - VERSION present + unsupported → loud incompatible error
    //          (never adopted, even with the flag);
    //        - no VERSION + fresh dir → stamp the current version;
    //        - no VERSION + has data + NO adopt (default) → loud legacy error;
    //        - no VERSION + has data + explicit `--adopt-legacy-datadir` →
    //          restore a recognized pre-M3 MANIFEST's v3/v4 stamp, or stamp
    //          a no-MANIFEST legacy dir as chained v1 and migrate forward;
    //          v5/unknown manifests remain refused.
    //      Placed after the `wal_preexisting` probe (so a fresh dir reads
    //      `false`) and before the encryption sidecar / page open, so a
    //      version-incompatible dir is refused before we touch any on-disk
    //      state we'd have to interpret.
    let has_data = pages_path.exists() || wal_preexisting;
    // v2 M1 (ADR-230): the check now RETURNS the store's on-disk version
    // — `DATA_DIR_VERSION_CHAINED_V1` (a pre-M1 chained store, incl. a
    // just-adopted legacy dir) dispatches the migrate-on-open sweep at
    // §11 below; `DATA_DIR_FORMAT_VERSION` (M1 slotted) proceeds.
    // A pre-fix 1→3 re-stamp could tear the old VERSION after the
    // migrating MANIFEST was durable. That state is unambiguously resumable:
    // restore the target stamp before applying the normal version guard.
    let m1_migration_in_flight = arcgraph_storage::read_data_dir_manifest(data_dir)
        .ok()
        .flatten()
        .is_some_and(|manifest| manifest.m1_migration_in_flight());
    let version_check = arcgraph_storage::check_or_stamp_data_dir(data_dir, has_data, adopt_legacy);
    let version_check = match version_check {
        Err(arcgraph_storage::DataDirVersionError::Malformed { .. }) if m1_migration_in_flight => {
            arcgraph_storage::stamp_data_dir(data_dir, arcgraph_storage::DATA_DIR_FORMAT_VERSION)
                .context("repair torn VERSION for in-flight M1 migration")?;
            arcgraph_storage::check_or_stamp_data_dir(data_dir, has_data, adopt_legacy)
        }
        other => other,
    };
    let data_dir_version = version_check.with_context(|| {
        format!(
            "on-disk data-dir version check failed for {} (SVC-2 upgrade-safety, #1302)",
            data_dir.display()
        )
    })?;
    let is_m3 = data_dir_version == arcgraph_storage::DATA_DIR_VERSION_DELTA_M3;
    let is_m4 = data_dir_version == arcgraph_storage::DATA_DIR_VERSION_DIRECT_M4;
    let is_delta_generation = is_m3 || is_m4;

    // #1519 BLOCK_FIX FIX 1 (SILENT-M6-CORRUPTION) — a v6/M4 generation's
    // STORE_TEL ref encoding is a MANIFEST-level discriminator distinct
    // from the coarse `data_dir_version` integer above (#1519 changed the
    // encoding without bumping `DATA_DIR_VERSION_DIRECT_M4`, since both
    // encodings are still v6/M4 generations at that granularity). Check it
    // HERE — before `pages.db` / the WAL are opened at all, let alone any
    // `PageType::Tel` page read — so a pre-#1519 generation (bare
    // STORE_TEL refs) is refused loud instead of silently misdecoded by
    // the new `decode_tel_ref` inverse.
    if is_m4 {
        let manifest = arcgraph_storage::read_data_dir_manifest(data_dir).with_context(|| {
            format!(
                "read v6/M4 MANIFEST for STORE_TEL ref-format check at {}",
                data_dir.display()
            )
        })?;
        arcgraph_storage::check_tel_ref_format(data_dir, manifest.as_ref()).with_context(|| {
            format!(
                "STORE_TEL ref-encoding check failed for {} (#1519 upgrade-safety)",
                data_dir.display()
            )
        })?;
    }

    // ADR-216 §D-4 / #1180 — WAL-encryption bootstrap. Computed AFTER the
    // `wal_preexisting` probe above so generating the `wal.dek` sidecar on
    // first boot does not flip the restart heuristic (the sidecar lives in
    // `wal_dir`). When `enabled`:
    //   (a) construct the selected `Arc<dyn KeySource>`;
    //   (b) `health_check(KeyScope::wal())` → fail-CLOSED startup error if
    //       the KEK is absent (ADR-033 — NEVER a silent plaintext fallback);
    //   (c) read-or-generate `wal.dek` + unwrap → a `WalEncryption`.
    // The resulting `WalEncryption` is threaded into BOTH §5 recovery AND
    // §6 the writer (encrypt-on-write WITHOUT decrypt-on-recover =
    // unrecoverable WAL). When disabled (v1.0-α default, OQ-2) this is
    // `None` and the durable path is the prior plaintext-WAL posture.
    let wal_encryption_bootstrap: Option<WalEncryptionBootstrap> = if wal_encryption.enabled {
        let key_source = wal_encryption.build_key_source()?;
        // (b) fail-fast KEK probe — refuse to serve plaintext WAL if the
        //     KEK is unresolvable (ADR-033 fail-closed).
        key_source.health_check(&KeyScope::wal()).with_context(|| {
            format!(
                "WAL encryption enabled but the KEK is unresolvable for scope {} — \
                 refusing to start rather than writing plaintext WAL (ADR-033 fail-closed). \
                 Provision the KEK in the selected secrets provider, or disable WAL encryption.",
                KeyScope::wal().namespace()
            )
        })?;
        // (c) read-or-generate the wrapped-DEK sidecar + unwrap → WalEncryption.
        let boot = bootstrap_wal_encryption(key_source.as_ref(), &wal_dir).with_context(|| {
            format!("WAL-encryption sidecar bootstrap at {}", wal_dir.display())
        })?;
        tracing::info!(
            target: "arcgraph_cli::bootstrap",
            key_source_id = key_source.key_source_id(),
            current_key_version = boot.current_key_version.raw(),
            freshly_generated = boot.freshly_generated,
            "durable bootstrap: WAL encryption ENABLED (ADR-216 §D-2 sidecar dance; #1180)",
        );
        Some(boot)
    } else {
        None
    };

    // §2. File-backed page IO (PD#2: PosixPageIo uses std `File`
    //     read/write + `sync_data()` for fdatasync/F_FULLFSYNC — NOT mmap).
    let io: Arc<dyn PageIo> = Arc::new(
        PosixPageIo::open_or_create(&pages_path)
            .with_context(|| format!("open page store {}", pages_path.display()))?,
    );
    // W28 Feature #582 (ADR-045) — feed the BufferPool's observability sink so
    // `arcgraph_storage_pages_total{kind}` (design-v2 §10.2 line 703 — the
    // hit-rate panel's PromQL source) is WIRED under `--data` + `--metrics-http`,
    // closing the §5.2 "reachable-pending-sink-wire" gap
    // (`docs/operations/v12-ga-exit-criteria.md`).
    //
    // M10 stage-1 (ADR-207) closed the producer half of the old HONESTY NOTE
    // here: the catalog now PINS this pool. `attach_page_store` (§7 below)
    // read-backs + materializes + verifies the catalog root page, so the
    // counter fires on REAL page reads on this path (≥1 Miss per boot; Hits on
    // restart read-back + tier-mutation write-through). The pool still backs
    // ONLY the catalog page store at v1 (records/index reads are served from
    // the in-memory MVCC stores rebuilt by WAL replay — their spill substrate
    // is ADR-140's separate track), so the hit-rate panel honestly reports
    // catalog-page traffic. `None` keeps the legacy zero-overhead path (the
    // pool's `metrics_sink` stays `None`).
    let mut buffer_pool = BufferPool::new(POOL_FRAMES, Arc::clone(&io));
    if let Some(sink) = &metrics_sink {
        buffer_pool = buffer_pool.with_metrics_sink(Arc::clone(sink));
    }
    // ADR-207: the pool moves into the catalog at §7 (attach_page_store) and
    // lives for the server's lifetime — no longer a bootstrap-scoped local.
    let buffer_pool = Arc::new(buffer_pool);

    // §3. Pre-writer replay scaffolding. Nothing in this section writes to the
    //     WAL: the writer MUST NOT attach until §5, after recovery has found
    //     and durably truncated any torn terminal tail (#1109). The replay
    //     target therefore starts from raw page-store handles plus an
    //     unlogged TxnManager. The same handles are wrapped for serving after
    //     recovery, so recovered pages/state are not discarded.
    //
    // #1404 M0.x — drive the frontier-advance MVCC drain on the durable serve
    // path. The reclaimer `gc()` was DRIVEN only at the ADR-229 checkpoint
    // trigger (`bootstrap.rs:801`, rare); between checkpoints, superseded
    // versions (update/delete churn + the REL-side adjacency updates the #1404
    // acceptance OOM'd on) accumulated resident with nothing driving
    // reclamation. `with_gc_drive_interval` runs `gc()` every N commits so the
    // resident superseded set stays bounded between checkpoints. INV-DRAIN is
    // unchanged (reclaim only `expired_lsn ≤ oldest_active_snapshot`).
    let txn_manager = Arc::new(TxnManager::new().with_gc_drive_interval(gc_drive_interval()));
    let allocator = Arc::new(PageAllocator::new());
    let primary_pages = Arc::new(PrimaryPageStore::new());
    let m3_record_store = if is_m3 {
        let record_io: Arc<dyn TenantPageIo> = Arc::new(TenantFilePageIo::new(
            data_dir,
            arcgraph_storage::m3_migration::M3_RECORD_STORE_FILE,
        ));
        let pools = Arc::new(PerTenantBufferPool::with_tenant_io(
            record_io,
            PerTenantBufferPoolConfig {
                frames_per_tenant: POOL_FRAMES,
                write_fraction: 0.5,
            },
        ));
        Some(Arc::new(BufferedRecordPageStore::with_identity(
            pools,
            PageStoreIdentity::for_generation(data_dir, arcgraph_storage::wal::STORE_RECORD),
        )))
    } else {
        None
    };
    // v9 incremental metadata does not capture record pages as a resident
    // owner. The served tenant-qualified tier is `m3_record_store`; this
    // empty legacy owner exists only to satisfy the v8 snapshot shape.
    let record_pages = Arc::new(RecordPageStore::new());
    // #1404 M0 — engage the BOUNDED resident blob-page tier on the durable
    // serve path. The #1414 heaptrack pinned the dominant ~8.2 KB/node OOM
    // term at `BlobStore.pages` (a never-drained in-memory DashMap of full
    // 8 KB blob pages, one per node's property bag). The bounded tier caps
    // the RESIDENT blob-page set at a byte watermark (default 0.5 × 4 GiB,
    // env-overridable via `ARCGRAPH_BLOB_RESIDENT_CAP_BYTES`), spilling
    // checkpoint-durable pages to `<data_dir>/blob-spill.db` and re-faulting
    // on read — so RSS is a function of the watermark, not of ingested node
    // count. The spill is process-local scratch (truncated on open,
    // discarded on restart; recovery rebuilds the store from WAL +
    // checkpoint), so it introduces no new durable format. INV-DURABLE
    // (evict only checkpoint-captured pages) is upheld inside the store.
    let blob_store = {
        let spill = BlobSpill::open(data_dir)
            .with_context(|| format!("open blob spill file in {}", data_dir.display()))?;
        Arc::new(BlobStore::with_bound(
            Arc::new(spill),
            BlobBoundConfig::from_env(),
        ))
    };
    let bm25_store: Arc<dyn Bm25IndexStoreHandle> = Bm25Service::new(data_dir.to_path_buf());
    let mut recovery_crud = match &m3_record_store {
        Some(records) => CrudStore::new_with_existing_buffered_page_store(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(records),
            Arc::clone(&blob_store),
        ),
        None => CrudStore::new_with_existing_page_stores(
            None,
            None,
            Arc::clone(&allocator),
            Arc::clone(&record_pages),
            Arc::clone(&blob_store),
        ),
    }
    .with_bm25_store(Arc::clone(&bm25_store));
    let m4_addressed_store = is_m4.then(|| Arc::new(arcgraph_storage::AddressedRecordStore::new()));
    if let Some(addressed) = &m4_addressed_store {
        recovery_crud =
            recovery_crud.with_authoritative_addressed_record_store(Arc::clone(addressed));
    }
    // W28 Feature #582 (ADR-045) — thread the observability sink into the
    // durable CrudStore so `arcgraph_hot_vertex_warnings_total{tenant}`
    // (design-v2 §10.2 line 721) fires under `--data` + `--metrics-http`.
    // `None` is the legacy zero-overhead path (metrics_sink stays `None`).
    if let Some(sink) = metrics_sink.clone() {
        recovery_crud = recovery_crud.with_metrics_sink(sink);
    }

    // §4. ADR-183 R1 — FULLY-WIRED PageStoreTarget. Production commits emit
    //     v4 CommitBundles with RecordPage + Blob + allocator_advances;
    //     `recover_from_wal` REJECTS RecordPage if record_store=None,
    //     rejects Blob if blob_store=None, and ZEROES allocator state if
    //     allocator_seed=None (a data-loss vector for T1/Strict commits per
    //     wal/replay.rs:462-545). All four are wired here; vector arenas are
    //     not present in the v1.0-α bootstrap (no vector index attached), so
    //     the plain `recover_from_wal` entry point is correct (the
    //     `recover_from_wal_with_vector_arenas` + snapshot-dir path lights
    //     up when a vector index is wired — out of S1 scope).
    //
    //     The 5th `PageStoreTarget` field — `bootstrap_from_mvcc` (the
    //     §Slice 3c orphan-IndexPage hook) — is intentionally left `None`.
    //     It fires ONLY when replay observes legacy `WalRecordType::IndexPage`
    //     records (pre-ADR-031); the post-ADR-031 hot path emits only v4
    //     `CommitBundle`s (`wal/replay.rs:1076-1084`), so a production WAL
    //     written by this binary never produces orphan pages. And even if a
    //     legacy fixture did, `None` makes replay WARN, not halt
    //     (`wal/replay.rs:1013-1023`) — so omitting the hook is safe, never a
    //     data-loss vector. (Do NOT "fix" this to `Some(...)`: there is no
    //     legacy-IndexPage source on the durable bootstrap path.)
    let primary_handle: Arc<dyn PrimaryPageStoreHandle> =
        Arc::clone(&primary_pages) as Arc<dyn PrimaryPageStoreHandle>;
    let records_handle: Arc<dyn RecordPageStoreHandle> = match &m3_record_store {
        Some(records) => Arc::clone(records) as Arc<dyn RecordPageStoreHandle>,
        None => Arc::clone(&record_pages) as Arc<dyn RecordPageStoreHandle>,
    };
    let blob_handle: Arc<dyn BlobStoreHandle> = Arc::clone(&blob_store) as Arc<dyn BlobStoreHandle>;
    let recovery_crud = Arc::new(recovery_crud);
    // P0 #776 — create the served InternTable BEFORE recovery and wire it
    // into the replay target so `recover_from_wal` reconstructs the label /
    // rel-type name↔id mapping from `WalRecordType::InternString` records.
    // The SAME `Arc` is handed to the `StorageBackend` at §9, so recovered
    // names reach `graph.schema` + the query binder. Pre-#776 a fresh empty
    // table was created at §9 AFTER recovery and nothing repopulated it, so
    // names came back as synthetic `label:N` and typed queries failed -32005.
    let mut intern = Arc::new(InternTable::new());
    // #352 Part 2 (ADR-199) — same discipline as the InternTable: create
    // the served IdempotencyStore BEFORE recovery and wire it into the
    // replay target so `recover_from_wal` rebuilds the `external_id →
    // internal_id` map from each v6 CommitBundle's `idempotency_bindings`
    // section. The SAME `Arc` is handed to the `StorageBackend` at §9, so a
    // post-restart `graph.ingest` resolves idempotently instead of minting
    // a duplicate (the #352 correctness bug).
    //
    // #1404 M0.x — engage the BOUNDED resident binding tier on the durable
    // serve path (the RE-2 freeze-capture term). Bindings are lookup-load-
    // bearing at-least-once ingest identity (~1/node, both node AND rel side),
    // and `iter_all()` materializes the WHOLE binding set into a `Vec` UNDER
    // `checkpoint_freeze` (`producer.rs:132`) — the RE-2 term that OOM'd the
    // 10M-nodes+20M-rels acceptance. The bounded tier caps the RESIDENT binding
    // set at a byte watermark (default 0.5 × 256 MiB, env-overridable via
    // `ARCGRAPH_IDEMPOTENCY_RESIDENT_CAP_BYTES`), spilling checkpoint-durable
    // bindings to `<data_dir>/idempotency-spill.db` — a durable, QUERYABLE-by-
    // key store — and faulting them back in on `get()`, so a re-ingest of a
    // spilled external_id STILL de-dupes (NEVER evict-to-nowhere). Process-
    // local scratch (truncated on open; bindings recover from WAL + checkpoint).
    let mut idempotency = if is_m4 {
        Arc::new(IdempotencyStore::new())
    } else {
        let spill = IdempotencySpill::open(data_dir)
            .with_context(|| format!("open idempotency spill file in {}", data_dir.display()))?;
        Arc::new(IdempotencyStore::with_bound(
            Arc::new(spill),
            IdempotencyBoundConfig::from_env(),
        ))
    };
    // #1221 (ADR-218) — same discipline as the InternTable / IdempotencyStore:
    // create the served per-tenant PermissionIndex BEFORE recovery and wire it
    // into the replay target so `recover_from_wal` re-drives every v8
    // CommitBundle's `acl_grants` section (apply_doc_acl / revoke_doc) into it.
    // The SAME `Arc` is handed to the router at §9 (via `.permissions(DEFAULT, …)`)
    // so the served principal-scoped `graph.search` enforces against the
    // recovered grants — closing the #1221 deny-all-on-bare-restart defect. At
    // v1.0 there is one user tenant (DEFAULT); the corpus seeds/ingests against
    // it. (The SYSTEM tenant carries no document ACLs.)
    let mut permissions = Arc::new(PermissionIndex::new());
    let m3_dpt = Arc::new(DirtyPageTable::new());
    let opened_production_extents = if is_delta_generation {
        open_production_extent_stores(data_dir, Arc::clone(&m3_dpt), is_m4)?
    } else {
        OpenedProductionExtents::default()
    };
    let production_extent_stores = opened_production_extents.stores;
    let production_affinity_allocators = opened_production_extents.affinity_allocators;
    let owner_rows = if is_m4 {
        Some(Arc::new(
            arcgraph_storage::OwnerRowRegistry::open_logical(
                data_dir,
                production_extent_stores.values().map(|runtime| {
                    Arc::new(arcgraph_storage::OwnerRowStore::new(Arc::clone(
                        &runtime.data,
                    )))
                }),
                Arc::clone(&m3_dpt),
            )
            .with_context(|| format!("open M4 logical owners in {}", data_dir.display()))?,
        ))
    } else {
        None
    };
    if let Some(owner) = &owner_rows {
        intern = Arc::new(
            InternTable::page_backed(Arc::clone(owner))
                .with_context(|| format!("open M4 intern owner in {}", data_dir.display()))?,
        );
        idempotency = Arc::new(IdempotencyStore::page_backed(Arc::clone(owner)));
        permissions = Arc::new(
            PermissionIndex::page_backed(
                Arc::clone(owner),
                Arc::clone(&idempotency),
                TenantId::DEFAULT,
            )
            .with_context(|| format!("open M4 permission owner in {}", data_dir.display()))?,
        );
    }
    // Built AFTER the M4 page-backing above, so replayed
    // `AllocatorAdvance{InternString | AclClass}` entries reach the SAME
    // page-backed counters the runtime will allocate from. Seeding them from
    // the checkpointed marker alone is not enough: post-checkpoint commits only
    // reach the in-RAM counters through their replayed advance, and dropping it
    // reissues durably-committed StringIds / AclClassIds after a crash.
    let allocator_seed: Arc<dyn AllocatorSeedHandle> = crud_allocator_seed_handle_with_owners(
        Arc::clone(&recovery_crud),
        Arc::clone(&allocator),
        intern.is_page_backed().then(|| Arc::clone(&intern)),
        permissions
            .is_page_backed()
            .then(|| Arc::clone(&permissions)),
    );
    let mut target = PageStoreTarget::primary_only(primary_handle)
        .with_record_store(records_handle)
        .with_blob_store(blob_handle)
        .with_allocator_seed(allocator_seed)
        .with_intern_table(Arc::clone(&intern))
        .with_idempotency_store(Arc::clone(&idempotency))
        .with_permission_index(Arc::clone(&permissions));
    if is_delta_generation {
        // M4 keeps `store.primary == None`, so its live CRUD path cannot emit
        // record-page deltas. Replay still requires both trait handles before
        // it will apply any physical delta, though, and ExtentAlloc needs the
        // same recovery DPT. Keep M3's real record owner; use the shared blob
        // store as a defensive, unreachable record-side handle on M4.
        let delta_records: Arc<dyn arcgraph_storage::DeltaPageStore> = match &m3_record_store {
            Some(records) => Arc::clone(records) as Arc<dyn arcgraph_storage::DeltaPageStore>,
            None => Arc::clone(&blob_store) as Arc<dyn arcgraph_storage::DeltaPageStore>,
        };
        target = target.with_delta_stores(
            Arc::clone(&blob_store) as Arc<dyn arcgraph_storage::DeltaPageStore>,
            delta_records,
            Arc::clone(&m3_dpt),
        );
    }
    for runtime in production_extent_stores.values() {
        target = target
            .with_extent_directory(Arc::clone(&runtime.directory))
            .with_extent_data_store(Arc::clone(&runtime.data));
    }

    // §5. ARIES-family WAL-replay-on-restart (PD#6), BEFORE writer attach.
    //     Replays every committed bundle into the MVCC stores before serving.
    //     On a fresh dir the WAL is empty → an empty report. Hard corruption
    //     surfaces as `Err` (fail-loud; operator intervenes). If recovery
    //     reports a torn terminal tail, truncate it to the last-valid offset
    //     and sync the segment + WAL dir before §6 opens SegmentWriter.
    //
    //     ADR-216 §D-4 RECOVERY PATH (load-bearing): when encryption is
    //     enabled, the SAME `WalEncryption` constructed in §1 MUST thread
    //     into the recovery readers — encrypt-on-write without
    //     decrypt-on-recover is unrecoverable WAL. `recover_from_wal_encrypted`
    //     opens its `WalRecoveryReader`s `with_encryption(..)` so encrypted
    //     payloads decrypt on yield (clear payloads pass through via the
    //     magic-peek, so mixed clear+encrypted WAL across an enable boundary
    //     is forward-safe). When disabled, `None` ⟹ the plaintext-recovery
    //     posture (byte-for-byte the prior `recover_from_wal`).
    // #1221 (ADR-218) forward-bind: durable recovery MUST run against a
    // fresh-at-ZERO `TxnManager` (created at §1; nothing has attached a
    // writer / catalog / PrimaryIndex yet — those land at §6/§7/§9 AFTER
    // recovery). The replay baseline (`applied_high_water`) is seeded from
    // `txn_manager.current_lsn()`; with it at `Lsn::ZERO` and
    // `LsnCounter::INITIAL == 1`, the skip-if-applied guard
    // (`commit_lsn <= baseline`) never skips the lowest real commit_lsn (1)
    // — so a #1221 `acl_grants`-only commit at the lowest LSN is always
    // replayed, never silently dropped (the deny-all defect). This assert
    // fails loud in debug/test builds if a future refactor pre-seeds the
    // recovery manager (advancing `current_lsn` before replay), which would
    // raise the baseline and reintroduce the skip-at-baseline risk.
    debug_assert_eq!(
        txn_manager.current_lsn(),
        Lsn::ZERO,
        "durable WAL recovery must run against a fresh-at-ZERO TxnManager \
         (current_lsn == 0); a pre-seeded baseline would skip-if-applied the \
         lowest-LSN bundle, incl. a #1221 acl_grants commit (ADR-218 deny-all risk)"
    );

    // §5a. SVC-1 / #849 / ADR-229 — checkpoint-anchored recovery (THE
    //      restart bound). BEFORE replaying the WAL, restore the latest
    //      valid full-state checkpoint (if any) into the SAME owner Arcs
    //      wired into the replay target above. `restore_latest_checkpoint`
    //      returns the frontier `checkpoint_lsn`; the WAL replay then
    //      SKIPS every record with `commit_lsn <= checkpoint_lsn` (already
    //      durable in the restored snapshot) and applies ONLY the
    //      post-frontier tail — bounding restart-recovery to
    //      O(WAL-since-checkpoint) instead of O(entire-history) (the #849
    //      rc-blocker: a 167 GB WAL that could not replay in 8.5 min at
    //      10M). No checkpoint (fresh/legacy dir, or a corrupt one) →
    //      `checkpoint_lsn == Lsn::ZERO` → a from-zero replay (exactly the
    //      pre-ADR-229 behaviour; back-compat + the SAFE direction).
    //
    //      The restore feeds every WAL-reconstructed owner (MVCC rows,
    //      primary/record/blob page images, allocator advances, intern
    //      names, idempotency bindings, permission grants) back through
    //      the SAME crash-campaign-proven replay entry points, then seeds
    //      the MVCC counter to the frontier — so the anchored replay's
    //      baseline (`max(current_lsn, checkpoint_floor)`) lands exactly
    //      at the frontier (defense-in-depth: the floor is ALSO passed
    //      explicitly, independent of the restore side-effect).
    let (checkpoint_frontier, incremental_redo_floor) = {
        // Build the restore-time allocator seed handle INSIDE this scope
        // so its `Arc<CrudStore>` clone drops at the end of the block —
        // BEFORE §7's `Arc::try_unwrap(recovery_crud)` (a lingering clone
        // would make try_unwrap fail with "recovery CrudStore still
        // shared"). The served checkpointer (§10) builds its own seed off
        // the SERVED `crud`, not `recovery_crud`.
        let restore_seed: Arc<dyn AllocatorSeedHandle> =
            crud_allocator_seed_handle(Arc::clone(&recovery_crud), Arc::clone(&allocator));
        let snap = arcgraph_storage::checkpoint::CheckpointSnapshot {
            txn: &txn_manager,
            primary_pages: &primary_pages,
            record_pages: &record_pages,
            blob: &blob_store,
            allocator_seed: restore_seed.as_ref(),
            intern: &intern,
            idempotency: &idempotency,
            permissions: &permissions,
            permissions_tenant: TenantId::DEFAULT,
        };
        if let Some(records) = &m3_record_store {
            arcgraph_storage::checkpoint::sweep_incremental_metadata_temps(data_dir).with_context(
                || format!("sweep orphan M3 checkpoint temps in {}", data_dir.display()),
            )?;
            let props_home: Arc<dyn PageIo> = Arc::new(
                PosixPageIo::open(
                    data_dir.join(arcgraph_storage::m3_migration::M3_PROPS_STORE_FILE),
                )
                .with_context(|| format!("open M3 props home in {}", data_dir.display()))?,
            );
            let mut home = arcgraph_storage::M3DoublewriteHome::with_tenant_records(
                props_home,
                records.pools().tenant_io(),
            );
            for runtime in production_extent_stores.values() {
                home = home.with_extent_directory(Arc::clone(&runtime.directory));
            }
            let dwb = arcgraph_storage::DoublewriteArea::new(data_dir);
            let dwb_report = dwb
                .restore(&mut home)
                .with_context(|| format!("restore M3 doublewrite at {}", data_dir.display()))?;
            let sidecar = arcgraph_storage::checkpoint::read_latest_sidecar(data_dir)
                .with_context(|| format!("read M3 checkpoint at {}", data_dir.display()))?
                .context("M3 generation has no checkpoint sidecar")?;
            if !sidecar.incremental_metadata {
                bail!("M3 generation checkpoint is not incremental metadata");
            }
            let base = arcgraph_storage::m3_migration::load_v9_physical_base(
                data_dir,
                sidecar.checkpoint_lsn,
                &txn_manager,
                records,
                &blob_store,
            )
            .with_context(|| format!("load M3 physical base at {}", data_dir.display()))?;
            let metadata = arcgraph_storage::checkpoint::read_incremental_metadata(
                data_dir,
                &snap,
                sidecar.checkpoint_lsn,
                sidecar.metadata_generation,
            )
            .with_context(|| format!("restore M3 metadata at {}", data_dir.display()))?;
            m3_dpt.restore(&metadata.dpt);
            let lsn_seed = crate::data_dir_migration::read_lsn_seed(data_dir)?;
            let migration_lsn = arcgraph_storage::read_data_dir_manifest(data_dir)
                .with_context(|| format!("read M3 MANIFEST at {}", data_dir.display()))?
                .and_then(|manifest| manifest.migration_lsn)
                .context("M3 MANIFEST is missing migration_lsn")?;
            if lsn_seed != migration_lsn.saturating_add(1) {
                bail!(
                    "M3 LSN_SEED {} is not migration frontier {} + 1",
                    lsn_seed,
                    migration_lsn
                );
            }
            tracing::info!(
                target: "arcgraph_cli::bootstrap",
                checkpoint_lsn = metadata.checkpoint_lsn.raw(),
                redo_lsn = metadata.redo_lsn.raw(),
                record_pages = base.record_pages,
                prop_pages = base.prop_pages,
                nodes = base.nodes,
                rels = base.rels,
                dwb_restored_pages = dwb_report.restored_pages,
                dwb_ignored_torn = dwb_report.ignored_torn_batch,
                "durable bootstrap: M3 physical base + incremental metadata restored",
            );
            (metadata.checkpoint_lsn, Some(metadata.redo_lsn))
        } else if is_m4 {
            arcgraph_storage::checkpoint::sweep_incremental_metadata_temps(data_dir).with_context(
                || format!("sweep orphan M4 checkpoint temps in {}", data_dir.display()),
            )?;
            let sidecar = arcgraph_storage::checkpoint::read_latest_sidecar(data_dir)
                .with_context(|| format!("read M4 checkpoint at {}", data_dir.display()))?
                .context("M4 generation has no checkpoint sidecar")?;
            if !sidecar.incremental_metadata || sidecar.full_state_snapshot {
                bail!("M4 generation checkpoint is not incremental metadata");
            }
            let manifest = arcgraph_storage::read_data_dir_manifest(data_dir)
                .with_context(|| format!("read M4 MANIFEST at {}", data_dir.display()))?
                .context("M4 generation has no MANIFEST")?;
            let migration_lsn = Lsn::new(
                manifest
                    .migration_lsn
                    .context("M4 MANIFEST is missing migration_lsn")?,
            );
            let lsn_seed = crate::data_dir_migration::read_lsn_seed(data_dir)?;
            if lsn_seed != migration_lsn.raw().saturating_add(1)
                || sidecar.checkpoint_lsn < migration_lsn
            {
                bail!(
                    "M4 generation frontier skew: LSN_SEED={}, migration_lsn={}, checkpoint_lsn={}",
                    lsn_seed,
                    migration_lsn.raw(),
                    sidecar.checkpoint_lsn.raw()
                );
            }
            let addressed = m4_addressed_store
                .as_ref()
                .context("M4 base load requires the direct-address store")?;
            let base = arcgraph_storage::m4_migration::load_v6_physical_base(
                data_dir,
                sidecar.checkpoint_lsn,
                &txn_manager,
                addressed,
                &blob_store,
            )
            .with_context(|| format!("load M4 physical base at {}", data_dir.display()))?;
            let metadata = arcgraph_storage::checkpoint::read_incremental_metadata(
                data_dir,
                &snap,
                sidecar.checkpoint_lsn,
                sidecar.metadata_generation,
            )
            .with_context(|| format!("restore M4 metadata at {}", data_dir.display()))?;
            m3_dpt.restore(&metadata.dpt);
            tracing::info!(
                target: "arcgraph_cli::bootstrap",
                checkpoint_lsn = metadata.checkpoint_lsn.raw(),
                redo_lsn = metadata.redo_lsn.raw(),
                record_pages = base.record_pages,
                prop_pages = base.prop_pages,
                nodes = base.nodes,
                rels = base.rels,
                "durable bootstrap: M4 direct extent base + incremental metadata restored",
            );
            (metadata.checkpoint_lsn, Some(metadata.redo_lsn))
        } else {
            match arcgraph_storage::checkpoint::restore_latest_checkpoint(data_dir, &snap)
                .with_context(|| format!("checkpoint restore at {}", data_dir.display()))?
            {
                Some(restore) => {
                    tracing::info!(
                        target: "arcgraph_cli::bootstrap",
                        checkpoint_lsn = restore.checkpoint_lsn.raw(),
                        mvcc_records = restore.counts.mvcc_records,
                        primary_pages = restore.counts.primary_pages,
                        record_pages = restore.counts.record_pages,
                        blob_pages = restore.counts.blob_pages,
                        "durable bootstrap: checkpoint restored — WAL replay anchored (ADR-229 #849)",
                    );
                    (restore.checkpoint_lsn, None)
                }
                None => (Lsn::ZERO, None),
            }
        }
    };

    let encryption = wal_encryption_bootstrap
        .as_ref()
        .map(|b| b.encryption.clone());
    let report = match incremental_redo_floor {
        Some(redo_floor) => recover_from_wal_encrypted_incremental(
            &wal_dir,
            Arc::clone(&txn_manager),
            target,
            None,
            encryption,
            checkpoint_frontier,
            redo_floor,
        ),
        None => recover_from_wal_encrypted_anchored(
            &wal_dir,
            Arc::clone(&txn_manager),
            target,
            None,
            encryption,
            checkpoint_frontier,
        ),
    }
    .with_context(|| format!("WAL recovery at {}", wal_dir.display()))?;
    if let Some(torn_tail) = report.torn_tail {
        truncate_torn_tail(&wal_dir, torn_tail).with_context(|| {
            format!(
                "truncate torn WAL tail at {}:{}",
                torn_tail.segment, torn_tail.offset
            )
        })?;
    }
    for (tenant, allocator) in &production_affinity_allocators {
        allocator.refresh_after_replay().with_context(|| {
            format!(
                "advance production affinity counters after replay for tenant {}",
                tenant.raw()
            )
        })?;
    }
    if is_m4 {
        let addressed = m4_addressed_store
            .as_ref()
            .context("M4 WAL recovery requires the direct-address store")?;
        let rebuilt = arcgraph_storage::m4_migration::rebuild_addressed_from_mvcc(
            &txn_manager,
            addressed,
            report.applied_commit_lsn,
        )
        .with_context(|| format!("rebuild M4 arithmetic slots at {}", data_dir.display()))?;
        tracing::info!(
            target: "arcgraph_cli::bootstrap",
            nodes = rebuilt.nodes,
            rels = rebuilt.rels,
            frontier = report.applied_commit_lsn.raw(),
            "durable bootstrap: M4 post-migration WAL heads republished by arithmetic address",
        );
    }

    // §6. WAL writer over `<data_dir>/wal`, opened only after §5 recovery and
    //     torn-tail truncation (#1109). The owning `WalWriter` is moved into
    //     the returned `DurabilityGuard`; its handle attaches in-place to the
    //     recovered TxnManager + runtime stores.
    //
    //     W28 Feature #582 (ADR-045) — feed the WAL writer's observability
    //     sink so `arcgraph_wal_fsync_duration_ms` (design-v2 §10.2 line 704)
    //     + `arcgraph_wal_writes_total{outcome}` fire under `--data` +
    //     `--metrics-http`, closing the §5.2 "reachable-pending-sink-wire"
    //     gap.
    let mut wal_config = WalConfig::new(&wal_dir);
    #[cfg(any(debug_assertions, feature = "fault-injection"))]
    if let Some(bytes) = std::env::var_os("ARCGRAPH_M3_TEST_WAL_SEGMENT_BYTES") {
        wal_config.segment_size_bytes = bytes
            .to_string_lossy()
            .parse::<u64>()
            .context("ARCGRAPH_M3_TEST_WAL_SEGMENT_BYTES must be a positive u64")?;
        if wal_config.segment_size_bytes == 0 {
            bail!("ARCGRAPH_M3_TEST_WAL_SEGMENT_BYTES must be greater than zero");
        }
    }
    if let Some(sink) = &metrics_sink {
        wal_config = wal_config.with_metrics_sink(Arc::clone(sink));
    }
    // ADR-216 §D-4 — wire encrypt-on-write. The SAME `WalEncryption` from §1
    // (and §5 recovery) is attached so every NEW WAL record's payload is
    // AES-256-GCM-wrapped at encode time. Symmetric with the recovery thread
    // above: the writer encrypts, the reader decrypts, under one DEK.
    if let Some(boot) = &wal_encryption_bootstrap {
        wal_config = wal_config.with_encryption(boot.encryption.clone());
    }
    let writer = WalWriter::spawn_from(wal_config, txn_manager.current_lsn())
        .with_context(|| format!("spawn WAL writer at {}", wal_dir.display()))?;
    let handle = writer.handle();
    txn_manager.attach_wal(handle.clone());
    if let Some(owner_rows) = &owner_rows {
        owner_rows.attach_commit_runtime(Arc::clone(&txn_manager), handle.clone());
    }

    // §7. Catalog + runtime wrappers. Catalog bootstrap now runs after
    //     recovery/truncation, so its idempotent SYSTEM sentinel cannot be
    //     appended after torn garbage. On restart over an existing dir, the
    //     re-bootstrap remains redundant-but-harmless (ADR-183 R3): the
    //     catalog root page lives at fixed `CATALOG_PAGE_ID`, and the attach
    //     below overwrites it idempotently.
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&buffer_pool, &txn_manager)
        .context("SystemCatalog::bootstrap failed")?;
    // #1513 (M5-D1b; amendment §10 Risk-2 ruling) — register the
    // generation's durable tenant census (MANIFEST `tenant_census`) into
    // the served catalog BEFORE the ADR-207 attach, through the SAME
    // registration path production routing consults
    // (`SystemCatalog::register_tenant` → `list_tenants` → the router's
    // `UnknownTenant` guard). Without this, a loaded generation's
    // tenants are servable on disk but `route(tenant, PartitionId::ZERO)`
    // — the exact dispatch the MCP/Bolt adapters issue (adapters.rs
    // `crud_for`, bolt.rs `read_access`) — returns `UnknownTenant`, so
    // M6's out-of-core pool could never serve them. ADR-207 stage-2
    // (registry-recovery-from-page) is NOT pulled forward: the source
    // here is the MANIFEST census, never the root page. Generations
    // without a census (pre-M4) register nothing. Registration is
    // idempotent per tenant, so a crash mid-sweep is completed by the
    // next cold open and re-opens never duplicate.
    let census = arcgraph_storage::read_data_dir_manifest(data_dir)
        .with_context(|| format!("read MANIFEST tenant census at {}", data_dir.display()))?
        .and_then(|manifest| manifest.tenant_census)
        .unwrap_or_default();
    register_census_tenants(&catalog, &txn_manager, &census).with_context(|| {
        format!(
            "register MANIFEST tenant census into the served catalog at {}",
            data_dir.display()
        )
    })?;
    let attach_report = catalog
        .attach_page_store(Arc::clone(&buffer_pool), Arc::clone(&io))
        .context("SystemCatalog::attach_page_store failed (ADR-207 catalog root page)")?;
    tracing::info!(
        target: "arcgraph_cli::bootstrap",
        prior_page = attach_report.prior_registry.is_some(),
        prior_diverged = attach_report.prior_diverged,
        healed_corruption = attach_report.healed_corruption,
        "durable bootstrap: catalog root page attached (ADR-207 M10 stage-1)",
    );
    txn_manager.attach_durability_lookup(catalog.clone());

    let primary = Arc::new(
        PrimaryIndex::with_page_store(
            Arc::clone(&txn_manager),
            Arc::clone(&allocator),
            (!is_delta_generation).then(|| handle.clone()),
            Arc::clone(&primary_pages),
        )
        .context("PrimaryIndex::new failed")?,
    );
    if is_m3 {
        // Rebuilding the derivative primary tree allocates real TxnManager
        // LSNs. Attach the v9 WAL first so those advances can never outrun
        // the durable watermark used by the final migration checkpoint.
        primary.attach_wal(handle.clone());
        let records = m3_record_store
            .as_ref()
            .context("M3 primary bootstrap requires the redone record cache")?;
        let stats = arcgraph_storage::m3_migration::bootstrap_primary_from_v9_base(
            records.as_ref(),
            &primary,
            txn_manager.current_lsn(),
        )
        .with_context(|| format!("bootstrap M3 primary index at {}", data_dir.display()))?;
        tracing::info!(
            target: "arcgraph_cli::bootstrap",
            indexed = stats.indexed,
            skipped = stats.skipped,
            "durable bootstrap: M3 primary index reconciled from record.store",
        );
    }
    let mut crud_store = match Arc::try_unwrap(recovery_crud) {
        Ok(store) => store,
        Err(_) => bail!("durable bootstrap internal error: recovery CrudStore still shared"),
    };
    crud_store.attach_wal(handle.clone());
    if !is_m4 {
        crud_store.attach_primary_index(Arc::clone(&primary));
    }
    if is_delta_generation {
        crud_store.attach_m3_dirty_page_table(Arc::clone(&m3_dpt));
    }
    if let Some(owner) = &owner_rows {
        crud_store.set_owner_rows(Arc::clone(owner));
    }
    let crud = Arc::new(crud_store);

    // §7c. #1221 (ADR-218) — wire the durable ACL WAL sink into the served
    //      PermissionIndex so post-startup write-through `apply_doc_acl` /
    //      `revoke_doc` (seed corpus + live `graph.ingest`, #1185) durify into
    //      the WAL's v8 `acl_grants` section and survive the NEXT bare restart.
    //      Done AFTER the CrudStore's WAL handle is attached (the sink's
    //      dedicated single-op commits must fire real WAL records). The sink
    //      issues its own transaction per op via the same `txn_manager` + `crud`
    //      the served path uses, so the op is durable iff its commit is
    //      (both-or-neither). At v1.0 the user tenant is DEFAULT.
    if !is_m4 {
        permissions.set_wal_sink(Arc::new(CrudAclWalSink::new(
            Arc::clone(&txn_manager),
            Arc::clone(&crud),
            TenantId::DEFAULT,
        )));
    }

    // §8. M4-41 cold-start stats rebuild (ADR-038 amendment-06 §D-25.1):
    //     replay populates the MVCC + record stores; this repopulates the
    //     per-tenant `CatalogStats` so `graph.schema` reflects recovered
    //     cardinalities. Per-tenant failures are captured (not propagated)
    //     and logged so a regression surfaces in stderr.
    let rebuild = rebuild_all_tenant_stats(report.applied_commit_lsn, &txn_manager, &crud);
    if !rebuild.failed.is_empty() {
        tracing::error!(
            target: "arcgraph_cli::bootstrap",
            failed = ?rebuild.failed,
            "per-tenant CatalogStats rebuild reported failures during durable bootstrap recover",
        );
    }
    // §8b. P0 #780 cold-start TEL ADJACENCY rebuild. WAL replay (§5)
    //      reinstated the relationship RECORDS into the MVCC + record
    //      stores, but the in-memory TEL adjacency chains that
    //      `scan_out` / `scan_in` walk for `MATCH ()-[r]->()` do NOT
    //      participate in the CommitBundle (the MVCC↔TEL atomicity gap,
    //      issue #20) and `tel_append` had no replay caller — so without
    //      this, traversal counts read 0 of N durably-committed rels
    //      after a restart (#780). This repopulates the forward + reverse
    //      adjacency from the recovered rel records, mirroring §8's stats
    //      rebuild (same per-tenant MVCC walk at `applied_commit_lsn`,
    //      same per-tenant fault isolation). Drives the SAME served `crud`
    //      so post-restart queries traverse the reinstated chains.
    let adj_rebuild = rebuild_all_tenant_adjacency(report.applied_commit_lsn, &txn_manager, &crud);
    if !adj_rebuild.failed.is_empty() {
        tracing::error!(
            target: "arcgraph_cli::bootstrap",
            failed = ?adj_rebuild.failed,
            "per-tenant TEL adjacency rebuild reported failures during durable bootstrap recover (#780)",
        );
    }
    // §8c. #1380 cold-start PRIMARY + SECONDARY index reconciliation. The
    //      live commit's dual-write index install DEGRADES on failure
    //      (warn-and-continue per ADR-023): the MVCC record commits but its
    //      primary-id (and any secondary-label) index entry can be MISSING,
    //      leaving a node SCAN-visible yet unreachable by id/label lookup.
    //      Recovery previously rebuilt only stats (§8) + TEL adjacency
    //      (§8b) from MVCC — NOT the primary/secondary index — so that
    //      split-brain SURVIVED restart forever (including on existing
    //      corrupt data-dirs). This reconciles every MVCC-visible record's
    //      missing primary + secondary entry from the authoritative MVCC
    //      store, mirroring §8 / §8b (same per-tenant walk at
    //      `applied_commit_lsn`, same per-tenant fault isolation). A
    //      no-op for every non-degraded record; heals + stays healed.
    let index_rebuild = rebuild_all_tenant_index(report.applied_commit_lsn, &txn_manager, &crud);
    if !index_rebuild.failed.is_empty() {
        tracing::error!(
            target: "arcgraph_cli::bootstrap",
            failed = ?index_rebuild.failed,
            "per-tenant primary/secondary index reconcile reported failures during durable bootstrap recover (#1380)",
        );
    }
    tracing::info!(
        target: "arcgraph_cli::bootstrap",
        data_dir = %data_dir.display(),
        applied_commit_lsn = report.applied_commit_lsn.raw(),
        last_wal_lsn = report.last_wal_lsn.raw(),
        nodes_walked = rebuild.total_nodes_walked(),
        rels_walked = rebuild.total_rels_walked(),
        // P0 #780 — number of relationships reinstated into the TEL
        // adjacency. A traversal-recovery regression surfaces as 0 here
        // on a dir whose workload committed ≥1 relationship.
        rels_reinstated = adj_rebuild.total_rels_reinstated(),
        // #1380 — number of records whose primary/secondary index entry
        // was reinstated from MVCC (the healed dual-write split-brain
        // population). Non-zero ONLY on a data-dir that suffered a
        // warn-and-continue index degrade; a positive value here means a
        // pre-existing corrupt dir was healed on this recovery.
        index_records_reinstated = index_rebuild.total_records_reinstated(),
        // P0 #776 — number of label / rel-type name↔id bindings replayed
        // into `intern`. A name-recovery regression surfaces as 0 here on
        // a dir whose workload created ≥1 named label / rel-type.
        interns_recovered = report.metrics.interns_recovered,
        "durable bootstrap: WAL recovery complete (ADR-183; #776; #780; #1380)",
    );

    // §9. Router + backend. The `intern` table was created at §4 and
    // populated by WAL replay (§5) — hand the SAME `Arc` to the backend so
    // recovered label / rel-type names are served (P0 #776). Do NOT create
    // a fresh table here (that was the pre-#776 bug: a fresh empty table
    // discarded every recovered name).
    //
    // #765 PART-1 / #1023 — attach an empty vector store and a BM25 store
    // (empty on fresh data dir; reopens persisted index on durable re-open) so
    // `TenantHandle::{vector,bm25}()` report the served search substrates
    // available. BM25 is also wired into `CrudStore` above so ingest commits
    // populate Tantivy and rollback can discard pending documents.
    //
    // #765 PART-1 — attach an (empty) VectorPageStore so `TenantHandle::vector()`
    // reports the vector substrate available (the `graph.schema` + `graph.search`
    // availability gates read this handle). The SEARCHABLE index is the
    // `SubstrateSearchProvider` bound at the dispatcher wiring site. Router-only
    // attach (NOT `CrudStore::with_vector_store`) → no WAL vector pages, recovery
    // path (§4/§5) unchanged. PART-2 populates this store for durable persistence.
    let vector_store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorPageStore::new());
    // #1221 (ADR-218): build via the builder so the WAL-replay-populated
    // PermissionIndex (the SAME `Arc` wired into the replay target at §4 and
    // the ACL WAL sink at §7c) is adopted by the served DEFAULT
    // `TenantHandle::permissions()` — closing the replay-vs-serve shared-index
    // requirement so recovered ACLs are actually enforced by `graph.search`.
    let router = Arc::new(
        MultiTenantRouter::builder(Arc::clone(&catalog), Arc::clone(&crud))
            .vector(vector_store)
            .bm25(bm25_store)
            .permissions(TenantId::DEFAULT, Arc::clone(&permissions))
            .build(),
    );

    // §10. SVC-1 / #849 / ADR-229 — build the graceful-shutdown / interval
    //      checkpointer over the SAME owner Arcs the served path uses, so a
    //      shutdown checkpoint (fired on `DurabilityGuard::drop`) snapshots
    //      the fully-committed state + bounds the NEXT restart's recovery.
    //      Constructed BEFORE the `StorageBackend::new` move of `txn_manager`
    //      / `intern` / `idempotency`. `WalCheckpointConfig::default()` is
    //      the v1.0 policy (1 GiB / 5 min interval + shutdown-checkpoint on);
    //      no server config struct is user-deserialized yet, so the default
    //      is the effective policy (a future server config landing threads a
    //      user-tuned `WalCheckpointConfig` here — the type is already
    //      `#[serde(deny_unknown_fields)]`).
    let checkpoint_config = arcgraph_storage::config::WalCheckpointConfig::default();
    let write_behind = if let Some(records) = &m3_record_store {
        let props_home: Arc<dyn PageIo> = Arc::new(
            PosixPageIo::open(data_dir.join(arcgraph_storage::m3_migration::M3_PROPS_STORE_FILE))
                .with_context(|| format!("open M3 props.store in {}", data_dir.display()))?,
        );
        let props: Arc<dyn arcgraph_storage::PageFlushTarget> = Arc::new(
            arcgraph_storage::BlobPageFlushTarget::new(Arc::clone(&blob_store), props_home),
        );
        let records_target: Arc<dyn arcgraph_storage::PageFlushTarget> = records.clone();
        Some(
            arcgraph_storage::WriteBehindCheckpointer::new(
                Arc::clone(&m3_dpt),
                props,
                records_target,
            )
            .with_doublewrite_area(Arc::new(
                arcgraph_storage::checkpoint::DoublewriteArea::new(data_dir),
            )),
        )
    } else if is_m4 {
        Some(
            arcgraph_storage::WriteBehindCheckpointer::new_extent_only(Arc::clone(&m3_dpt))
                .with_doublewrite_area(Arc::new(
                    arcgraph_storage::checkpoint::DoublewriteArea::new(data_dir),
                )),
        )
    } else {
        None
    };
    let m3_write_behind = write_behind.map(|mut write_behind| {
        for runtime in production_extent_stores.values() {
            write_behind = write_behind
                .with_extent_data_target(Arc::clone(&runtime.data))
                .with_extent_directory_target(Arc::clone(&runtime.directory));
        }
        Arc::new(write_behind)
    });
    // M4 uses the same bounded incremental producer with all direct-address
    // extent homes registered above. Owners 6-8 encode stable-zero retirement
    // sections, so the sidecar can advance only after DWB + every direct home
    // is durable; no legacy logical owner is recaptured.
    let checkpointer = if checkpoint_config.checkpoint_on_shutdown {
        // The checkpointer's allocator seed is built off the SERVED
        // `crud` (post-§7 `try_unwrap`), so it never keeps `recovery_crud`
        // shared. Node/Rel advances seed via `crud`; Page* via `allocator`.
        let checkpoint_allocator_seed: Arc<dyn AllocatorSeedHandle> =
            crud_allocator_seed_handle(Arc::clone(&crud), Arc::clone(&allocator));
        Some(DurableCheckpointer {
            data_dir: data_dir.to_path_buf(),
            buffer_pool: Arc::clone(&buffer_pool),
            txn_manager: Arc::clone(&txn_manager),
            primary_pages: Arc::clone(&primary_pages),
            record_pages: Arc::clone(&record_pages),
            blob_store: Arc::clone(&blob_store),
            allocator: Arc::clone(&allocator),
            crud: Arc::clone(&crud),
            allocator_seed: checkpoint_allocator_seed,
            intern: Arc::clone(&intern),
            idempotency: Arc::clone(&idempotency),
            permissions: Arc::clone(&permissions),
            wal_handle: handle.clone(),
            m3_write_behind,
            // BLOCK-3: ONE mutex shared across every clone (interval task +
            // shutdown Drop) so the two producers cannot interleave.
            producer_mutex: Arc::new(parking_lot::Mutex::new(())),
        })
    } else {
        None
    };

    // §11. v2 M1 (ADR-230 / design §0.1-§0.2/§M1.4) — MANIFEST stamp +
    //      migrate-on-open. Runs AFTER every rebuild (§5 recovery, §8
    //      stats, §8b TEL, §8c index reconcile) so the sweep's normal
    //      `update_node`/`update_rel` transactions operate on the fully
    //      recovered store, and BEFORE the server accepts connections
    //      (still under the §1 data-dir LOCK, no concurrent snapshots).
    //
    //      Dispatch (see `arcgraph_storage::migrate` for the §0.2
    //      crash-contract adaptation):
    //      - v1 (chained) store, OR a v3 store whose MANIFEST is absent /
    //        still `slotted-v1-migrating` (a crash-mid-sweep resume, or
    //        the self-healing anomalous-state path) → write the
    //        migrating MANIFEST, re-stamp VERSION=3 (pre-M1 binaries are
    //        locked out BEFORE the first slotted byte can land durably),
    //        run the idempotent sweep, fire a best-effort checkpoint
    //        (drops the reclaimed chains from the durable image + bounds
    //        next recovery), then rewrite the MANIFEST to `slotted-v1` —
    //        the single commit point.
    //      - v3 store with a final MANIFEST → nothing to do.
    let manifest = arcgraph_storage::read_data_dir_manifest(data_dir)
        .with_context(|| format!("read data-dir MANIFEST at {} (v2 M1)", data_dir.display()))?;
    let needs_m1_migration = data_dir_version == arcgraph_storage::DATA_DIR_VERSION_CHAINED_V1
        || manifest.is_none()
        || manifest
            .as_ref()
            .is_some_and(arcgraph_storage::DataDirManifest::m1_migration_in_flight);
    if needs_m1_migration {
        let now = arcgraph_storage::manifest::now_rfc3339_utc();
        // (a) The migrating marker lands FIRST (crash ⟹ resume on next
        //     open), then (b) the VERSION re-stamp locks out pre-M1
        //     binaries before any slotted byte can hit the WAL. Both are
        //     idempotent for the resume path.
        arcgraph_storage::write_data_dir_manifest(
            data_dir,
            &arcgraph_storage::DataDirManifest::m1_migrating(now.clone()),
        )
        .with_context(|| format!("write migrating MANIFEST at {}", data_dir.display()))?;
        arcgraph_storage::stamp_data_dir(data_dir, arcgraph_storage::DATA_DIR_VERSION_SLOTTED_M1)
            .with_context(|| format!("re-stamp data-dir VERSION 1→3 at {}", data_dir.display()))?;
        // (c) The sweep proper — forward-only, idempotent per record,
        //     batched through the normal transactional write path.
        let report = arcgraph_storage::run_m1_migrate_on_open(
            &txn_manager,
            &crud,
            &arcgraph_storage::M1MigrateOptions::from_env(),
        )
        .with_context(|| format!("v2 M1 migrate-on-open sweep at {}", data_dir.display()))?;
        // (d) Best-effort checkpoint: captures the migrated state (incl.
        //     the chain reclaim) into the durable image + bounds the next
        //     recovery. A failure is logged, NOT fatal — recovery
        //     correctness never depends on it (the sweep's commits are
        //     WAL-durable), so the MANIFEST flip below still proceeds.
        if report.batches_committed > 0 {
            match &checkpointer {
                Some(cp) => match cp.checkpoint() {
                    Ok(lsn) => tracing::info!(
                        target: "arcgraph_cli::bootstrap",
                        checkpoint_lsn = lsn.raw(),
                        "v2 M1 post-migration checkpoint established",
                    ),
                    Err(e) => tracing::warn!(
                        target: "arcgraph_cli::bootstrap",
                        error = %e,
                        "v2 M1 post-migration checkpoint FAILED (non-fatal; the sweep's \
                         commits are WAL-durable and the interval checkpointer will cover)",
                    ),
                },
                None => tracing::info!(
                    target: "arcgraph_cli::bootstrap",
                    "v2 M1 post-migration checkpoint skipped (checkpointer disabled)",
                ),
            }
        }
        // (e) The single commit point (§0.2): the crash-atomic MANIFEST
        //     rewrite to the final `slotted-v1`.
        arcgraph_storage::write_data_dir_manifest(
            data_dir,
            &arcgraph_storage::DataDirManifest::m1_slotted(
                arcgraph_storage::manifest::now_rfc3339_utc(),
            ),
        )
        .with_context(|| format!("write final MANIFEST at {}", data_dir.display()))?;
        tracing::info!(
            target: "arcgraph_cli::bootstrap",
            nodes = report.nodes_migrated,
            rels = report.rels_migrated,
            chains_removed = report.chains_removed,
            already_slotted = report.already_slotted,
            kept_chained_large = report.kept_chained_large,
            was_noop = report.was_noop(),
            "v2 M1 migrate-on-open complete — data dir at version 3 (slotted-v1)",
        );
    }

    // §11b. v2 M2 (ADR-230 / design §0.1-§0.2/§M2.6) — the JSON →
    //       typed-block migrate-on-open, `data_dir_version` 3 → 4.
    //       Runs AFTER the M1 leg (a v1 store chains 1→3→4 in one
    //       boot), same posture: under the §1 LOCK, before the server
    //       accepts connections, no concurrent snapshots.
    //
    //       Dispatch: any store whose MANIFEST is not final `typed-v1`
    //       — a fresh-from-M1 `slotted-v1`, a crash-mid-M2-sweep
    //       `typed-v1-migrating` resume, or an anomalous absent
    //       MANIFEST — enters the sweep. The mcp bridge's
    //       `reencode_json_bag_to_typed` is the injected re-encoder
    //       (storage never parses JSON — ADR-089 §D-1).
    let manifest_after_m1 = arcgraph_storage::read_data_dir_manifest(data_dir)
        .with_context(|| format!("read data-dir MANIFEST at {} (v2 M2)", data_dir.display()))?;
    let needs_m2_migration = !manifest_after_m1
        .as_ref()
        .is_some_and(arcgraph_storage::DataDirManifest::props_fully_typed);
    if needs_m2_migration {
        let now = arcgraph_storage::manifest::now_rfc3339_utc();
        // (a) The M2 migrating marker FIRST (crash ⟹ resume), then
        // (b) VERSION 4 locks out pre-M2 binaries BEFORE the first
        //     typed byte can land durably (they would silently
        //     mis-read a typed payload as a corrupt JSON bag → empty —
        //     the exact misread class the VERSION gate prevents).
        arcgraph_storage::write_data_dir_manifest(
            data_dir,
            &arcgraph_storage::DataDirManifest::m2_migrating(now),
        )
        .with_context(|| format!("write M2 migrating MANIFEST at {}", data_dir.display()))?;
        arcgraph_storage::stamp_data_dir(data_dir, arcgraph_storage::DATA_DIR_FORMAT_VERSION)
            .with_context(|| format!("re-stamp data-dir VERSION 3→4 at {}", data_dir.display()))?;
        // (c) The sweep — forward-only, idempotent per record, batched
        //     through the production transactional write path. The
        //     re-encoder closure captures the intern table + the WAL
        //     handle (key_id InternString records precede each batch's
        //     commit — the label-intern durability ordering).
        let intern_for_m2 = Arc::clone(&intern);
        let wal_for_m2 = crud.wal().cloned();
        let reencode = move |tenant: arcgraph_core::TenantId,
                             bag: &[u8]|
              -> Result<
            Option<arcgraph_storage::prop_block::TypedBagParts>,
            String,
        > {
            arcgraph_mcp::storage::property_payload::reencode_json_bag_to_typed(
                bag,
                &intern_for_m2,
                wal_for_m2.as_ref(),
                tenant,
            )
            .map_err(|e| e.to_string())
        };
        let report = arcgraph_storage::run_m2_migrate_on_open(
            &txn_manager,
            &crud,
            &reencode,
            &arcgraph_storage::M2MigrateOptions::from_env(),
        )
        .with_context(|| format!("v2 M2 migrate-on-open sweep at {}", data_dir.display()))?;
        // (d) Best-effort checkpoint (same rationale as the M1 leg —
        //     non-fatal; the sweep's commits are WAL-durable).
        if report.batches_committed > 0 {
            match &checkpointer {
                Some(cp) => match cp.checkpoint() {
                    Ok(lsn) => tracing::info!(
                        target: "arcgraph_cli::bootstrap",
                        checkpoint_lsn = lsn.raw(),
                        "v2 M2 post-migration checkpoint established",
                    ),
                    Err(e) => tracing::warn!(
                        target: "arcgraph_cli::bootstrap",
                        error = %e,
                        "v2 M2 post-migration checkpoint FAILED (non-fatal; the sweep's \
                         commits are WAL-durable and the interval checkpointer will cover)",
                    ),
                },
                None => tracing::info!(
                    target: "arcgraph_cli::bootstrap",
                    "v2 M2 post-migration checkpoint skipped (checkpointer disabled)",
                ),
            }
        }
        // (e) The single commit point (§0.2): the crash-atomic MANIFEST
        //     rewrite to the final `typed-v1`.
        arcgraph_storage::write_data_dir_manifest(
            data_dir,
            &arcgraph_storage::DataDirManifest::m2_typed(
                arcgraph_storage::manifest::now_rfc3339_utc(),
            ),
        )
        .with_context(|| format!("write final M2 MANIFEST at {}", data_dir.display()))?;
        tracing::info!(
            target: "arcgraph_cli::bootstrap",
            nodes = report.nodes_migrated,
            rels = report.rels_migrated,
            chains_removed = report.chains_removed,
            already_typed = report.already_typed,
            skipped_opaque = report.skipped_opaque,
            was_noop = report.was_noop(),
            "v2 M2 migrate-on-open complete — data dir at version 4 (typed-v1)",
        );
    }

    Ok((
        // #352 Part 2 (ADR-199): hand the backend the SAME idempotency
        // store wired into the replay target above, so recovered bindings
        // are visible to the served ingest path.
        StorageBackend::new(router, txn_manager, intern).with_idempotency_store(idempotency),
        // #886 — hand the guard the exclusive data-dir lock acquired at §1, so
        // it is held for the server loop and released (by the OS) on guard drop
        // or process death. #849 — plus the ADR-229 shutdown checkpointer.
        DurabilityGuard::durable(
            writer,
            data_lock,
            checkpointer,
            generation_cleanup_root,
            production_extent_stores,
            owner_rows,
            production_affinity_allocators,
            generation_pin,
        ),
    ))
}

/// Build the **ephemeral** substrate (`--in-memory`): [`InMemoryPageIo`] +
/// `wal: None`. Byte-for-byte the prior v1.0-α posture (the W26-β-1
/// GA-BOOTSTRAP-WIRING shape) — preserved so tests + demos keep working.
/// **NON-DURABLE**: all data is lost on process exit.
fn build_in_memory(
    metrics_sink: Option<Arc<dyn MetricsSink>>,
) -> Result<(StorageBackend, DurabilityGuard)> {
    let io: Arc<dyn PageIo> = Arc::new(InMemoryPageIo::new());
    // W28 Feature #582 (ADR-045) — feed the BufferPool sink for `--in-memory`
    // + `--metrics-http`, symmetric with the durable path so the observability
    // posture does not silently differ between the two modes. M10 stage-1
    // (ADR-207): the catalog pins this pool too (attach below), so
    // `arcgraph_storage_pages_total` fires symmetrically — the old "reads 0
    // samples until the page-backed read path lands" caveat is closed on BOTH
    // paths. `--in-memory` has NO WAL (`wal: None`,
    // `DurabilityGuard::ephemeral`), so `arcgraph_wal_fsync_duration_ms` is
    // durable-mode-only and is intentionally NOT wired here — there is no WAL
    // writer to observe (firing it on a non-existent WAL would be meaningless).
    // `None` keeps the legacy zero-overhead path.
    let mut buffer_pool = BufferPool::new(POOL_FRAMES, Arc::clone(&io));
    if let Some(sink) = &metrics_sink {
        buffer_pool = buffer_pool.with_metrics_sink(Arc::clone(sink));
    }
    let buffer_pool = Arc::new(buffer_pool);
    let txn_manager = Arc::new(TxnManager::new());
    let catalog = Arc::new(SystemCatalog::new());
    catalog
        .bootstrap(&buffer_pool, &txn_manager)
        .context("SystemCatalog::bootstrap failed")?;
    // M10 stage-1 (ADR-207) — symmetric with `build_durable` §4: pin the
    // catalog root page (over `InMemoryPageIo` here; ephemeral by
    // definition) so the metric posture matches durable mode.
    let attach_report = catalog
        .attach_page_store(Arc::clone(&buffer_pool), Arc::clone(&io))
        .context("SystemCatalog::attach_page_store failed (ADR-207 catalog root page)")?;
    tracing::debug!(
        target: "arcgraph_cli::bootstrap",
        prior_page = attach_report.prior_registry.is_some(),
        "in-memory bootstrap: catalog root page attached (ADR-207 M10 stage-1)",
    );
    // ADR-087 D-2 — wire the PrimaryIndex so `crud::commit`'s per-tenant
    // CatalogStats hook fires (without it `graph.ingest` succeeds but
    // `graph.schema` returns empty labels — the W18δ-flagged gap closed by
    // W26-β-1 / issue #439). No WAL in this mode (`wal: None`).
    let allocator = Arc::new(PageAllocator::new());
    let primary = Arc::new(
        PrimaryIndex::new(Arc::clone(&txn_manager), Arc::clone(&allocator), None)
            .context("PrimaryIndex::new failed")?,
    );
    // W28 Feature #582 (ADR-045) — chain `.with_metrics_sink(...)` when the
    // operator wired `--metrics-http`. The CRUD TEL overflow path then fires
    // `record_hot_vertex_warning(tenant)` (§10.2 line 721). When `None`, the
    // CrudStore's metrics_sink stays `None` (no-op zero-overhead path).
    let bm25_store: Arc<dyn Bm25IndexStoreHandle> = Bm25Service::new(in_memory_bm25_dir());
    let mut crud_store = CrudStore::new_with_index(None, primary, allocator)
        .with_bm25_store(Arc::clone(&bm25_store));
    if let Some(sink) = metrics_sink {
        crud_store = crud_store.with_metrics_sink(sink);
    }
    let crud = Arc::new(crud_store);
    // #765 PART-1 / #1023 — attach empty vector + BM25 stores so
    // `TenantHandle::{vector,bm25}()` report the served search substrates
    // available (symmetric with the durable path). The vector searchable index
    // is the bound `SubstrateSearchProvider`; BM25 is also wired into
    // `CrudStore` above so served ingest populates Tantivy.
    let vector_store: Arc<dyn VectorPageStoreHandle> = Arc::new(VectorPageStore::new());
    let router = Arc::new(MultiTenantRouter::new_with_bm25(
        Arc::clone(&catalog),
        Arc::clone(&crud),
        Some(vector_store),
        Some(bm25_store),
    ));
    let intern = Arc::new(InternTable::new());
    Ok((
        StorageBackend::new(router, txn_manager, intern),
        DurabilityGuard::ephemeral(),
    ))
}

/// ADR-202 §D-8 + §Open-questions — the serve-binary community-scheduler
/// slice: start a [`CommunityRefreshScheduler`] over the SAME `catalog` /
/// `crud` / `txn_manager` the served [`StorageBackend`] reads, wired with
/// the process `MetricsRegistry` as the [`RefreshObserver`] so every
/// successful per-tenant Leiden refresh fires
/// `arcgraph_leiden_last_run_seconds{tenant}` (design-v2 §10.2, the eighth
/// metric) into the SAME registry the `--metrics-http` listener scrapes.
///
/// # Why a CLI helper and not `bootstrap_engine`
///
/// [`arcgraph_storage::engine::bootstrap_engine`] is the canonical engine
/// composition (ADR-202 §D-4 wires the observer there), but it *also*
/// builds its own [`MultiTenantRouter`] from scratch via the builder.
/// The serve binary's router is already fully composed by
/// `build_durable` / `build_in_memory` with idempotency + metrics-sink
/// plus BM25 and vector wiring that `bootstrap_engine` does not
/// replicate, and the [`CrudStore`] is constructed before any router
/// exists. So the serve binary cannot simply *call* `bootstrap_engine`
/// without losing that wiring. Instead this helper reuses
/// `bootstrap_engine`'s **scheduler-composition primitives** verbatim
/// (the §1–§4 sequence: shared provider → production hook → register
/// tenants → `start_with_observer` → register with scheduler) over the
/// already-built handles. It is the single CLI-side composition site for
/// the scheduler — no duplication beyond the engine's own primitive calls.
///
/// # Scope (ADR-202 §D-8, deliberate)
///
/// This wires the **producer** (scheduler success → observer) so the
/// metric becomes real on the served binary. It does NOT rewire the
/// already-built router's `CommunityIndexProvider` (the router was
/// composed without one in `build_durable`/`build_in_memory`, so
/// `TenantHandle::community()` stays `None` and `graph.community`-style
/// reads remain unserved at v1.0-α). The scheduler here builds its own
/// [`SharedBTreeIndexProvider`] / [`BTreeMembershipIndex`] purely as the
/// `install_into` target the refresh needs; the gauge fires off the
/// install-success notification, which is exactly the §10.2 freshness
/// signal. Serving community *queries* from the refreshed index is a
/// separate router-recomposition slice (it would require threading the
/// provider into `build_*` before the router is built).
///
/// # Lifecycle
///
/// The returned `Arc<CommunityRefreshScheduler>` owns a dedicated OS
/// thread named `arcgraph-community-refresh`. The caller (`run_serve`)
/// MUST call [`CommunityRefreshScheduler::shutdown`] on it after the
/// transport returns and before the `MetricsRegistry` is dropped, so the
/// thread is joined cleanly (mirrors the `DurabilityGuard` WAL-thread
/// ownership discipline).
///
/// `interval` is the refresh cadence. Production passes the ADR-040 §D-7
/// daily default ([`SchedulerConfig::default`]'s 24 h); tests pass a short
/// interval (or drive [`CommunityRefreshScheduler::tick`] directly) to
/// observe the gauge within a bounded window.
#[must_use]
pub fn start_community_scheduler(
    catalog: Arc<SystemCatalog>,
    crud: Arc<CrudStore>,
    txn_manager: Arc<TxnManager>,
    observer: Arc<dyn RefreshObserver>,
    scheduler_config: SchedulerConfig,
) -> Arc<CommunityRefreshScheduler> {
    // §1 — shared community provider (its own BTreeMembershipIndex is the
    // scheduler's install target; see the "Scope" rustdoc). v1.0 supports
    // a single community index per deployment (ADR-040 §D-3); id(1) matches
    // `EngineConfig::new`'s convention.
    let community_provider = Arc::new(SharedBTreeIndexProvider::new(CommunityIndexId::new(1)));
    let membership_index: Arc<BTreeMembershipIndex> = Arc::clone(community_provider.index());

    // §2 — ProductionRefreshHook over the SAME crud + txn_manager the
    // served write path commits into, so each tick materialises the
    // tenant's real graph (per-tick re-mat, ADR-040 amendment-05). Register
    // every catalog-listed tenant (the served DEFAULT today; multi-tenant
    // when the catalog grows). LeidenParams default is sufficient for v1.0
    // (ADR-040 §D-7), matching `EngineConfig::new`.
    let adapter = CrudStoreGraphAdapter::new(Arc::clone(&crud), Arc::clone(&txn_manager));
    let refresh_hook = Arc::new(ProductionRefreshHook::new(
        adapter,
        Arc::clone(&membership_index),
        LeidenParams::default(),
    ));
    let tenants: Vec<TenantId> = catalog
        .list_tenants()
        .into_iter()
        .map(|r| r.tenant_id)
        .collect();
    for tenant in &tenants {
        refresh_hook.register_tenant(*tenant);
    }
    let refresh_hook_dyn: Arc<dyn RefreshHook> = refresh_hook as Arc<dyn RefreshHook>;

    // §3 — start the scheduler WITH the observer (ADR-202 §D-4): the
    // success arm of `refresh_one_tenant` notifies the observer, which the
    // process `MetricsRegistry` implements to set the freshness gauge.
    let scheduler = CommunityRefreshScheduler::start_with_observer(
        scheduler_config,
        refresh_hook_dyn,
        Some(observer),
    );

    // §4 — register every tenant with the scheduler so ticks pick them up
    // (after `start`, mirroring `bootstrap_engine`).
    for tenant in &tenants {
        scheduler.register(*tenant);
    }

    tracing::info!(
        target: "arcgraph_cli::bootstrap",
        registered_tenants = tenants.len(),
        interval = ?scheduler_config.interval,
        "community refresh scheduler started with metrics observer \
         (ADR-202 §D-8 serve-binary slice; arcgraph_leiden_last_run_seconds wired)",
    );

    scheduler
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─────────────────────────────────────────────────────────────────
    // BootstrapMode::from_flags — refuse-to-start policy (ADR-183 §Policy).
    // The byte-durability round-trip + fault-injection + multi-tenant
    // forward-pin live in the integration test
    // `tests/durable_bootstrap_restart.rs` (the ADR-133 active verification).
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn from_flags_neither_refuses_to_start() {
        let err = BootstrapMode::from_flags(None, false)
            .expect_err("neither --data nor --in-memory must refuse to start");
        let msg = format!("{err}");
        assert!(
            msg.contains("--data") && msg.contains("--in-memory"),
            "refuse-to-start error must name BOTH flags; got: {msg}"
        );
        assert!(
            msg.contains("non-durable") || msg.contains("NON-DURABLE"),
            "refuse-to-start error must document --in-memory as non-durable; got: {msg}"
        );
    }

    #[test]
    fn from_flags_data_only_is_durable() {
        let mode = BootstrapMode::from_flags(Some(Path::new("/tmp/arcgraph-x")), false)
            .expect("--data alone resolves to durable");
        assert_eq!(
            mode,
            BootstrapMode::Durable {
                data_dir: PathBuf::from("/tmp/arcgraph-x"),
            }
        );
    }

    #[test]
    fn from_flags_in_memory_only_is_ephemeral() {
        let mode =
            BootstrapMode::from_flags(None, true).expect("--in-memory alone resolves to ephemeral");
        assert_eq!(mode, BootstrapMode::InMemory);
    }

    #[test]
    fn from_flags_both_is_mutually_exclusive() {
        let err = BootstrapMode::from_flags(Some(Path::new("/tmp/x")), true)
            .expect_err("--data + --in-memory must be rejected");
        assert!(
            format!("{err}").contains("mutually exclusive"),
            "both-flags error must say 'mutually exclusive'; got: {err}"
        );
    }

    #[test]
    fn in_memory_bootstrap_succeeds_and_is_not_durable() {
        // Pin: --in-memory bootstraps without panic + the guard reports
        // non-durable (no WAL watermark).
        let (backend, guard) = bootstrap_storage_backend(&BootstrapMode::InMemory)
            .expect("in-memory bootstrap succeeds");
        assert!(!guard.is_durable());
        assert!(guard.last_durable_lsn().is_none());
        // The DEFAULT tenant routes after bootstrap.
        let handle = backend
            .router()
            .route(
                arcgraph_core::TenantId::DEFAULT,
                arcgraph_core::PartitionId::ZERO,
            )
            .expect("route DEFAULT");
        assert_eq!(handle.tenant(), arcgraph_core::TenantId::DEFAULT);
    }
}
