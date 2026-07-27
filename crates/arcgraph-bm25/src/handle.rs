//! Per-tenant search-side handle (ADR-039 §D-8).
//!
//! [`Bm25IndexHandle`] is the read-side query surface for BM25 text
//! search. M4 (ArcQL executor) and M5 (MCP tool) consumers obtain
//! handles via [`crate::Bm25Service::handle`]; the commit-side
//! [`arcgraph_storage::mutation_log::Bm25IndexStoreHandle`] trait
//! lives separately on `TenantHandle` (ADR-039 §D-9).
//!
//! # Local-only keying
//!
//! Every handle is keyed by `(TenantId, PartitionId, IndexId)`.
//! v1.0 invariants:
//! - `partition_id == PartitionId::ZERO`.
//! - `index_id == IndexId::DEFAULT_BM25` (= `IndexId::ZERO`).
//!
//! # Buffered semantics
//!
//! `upsert_document` / `delete_document` are NOT durable until the
//! surrounding CRUD txn's WAL fsync succeeds and the post-fsync
//! `commit_pending(tenant)` fires (ADR-039 §D-5). Producers MUST
//! call [`arcgraph_storage::TxnMutationLog::note_bm25_tenant`] for
//! every BM25 mutation so the rollback closure can dispatch
//! `rollback_pending` on WAL fsync failure.

use std::sync::Arc;

use arcgraph_core::{Lsn, NodeId, PartitionId, TenantId};
use parking_lot::Mutex;
use tantivy::collector::TopDocs;
use tantivy::query::{BooleanQuery, EmptyQuery, Occur, PhraseQuery, Query, TermQuery};
use tantivy::schema::{IndexRecordOption, Value};
use tantivy::{Index, IndexReader, IndexWriter, TantivyDocument, Term, doc};

use crate::error::Bm25Error;
use crate::eviction::IdleTracker;
use crate::mvcc::build_visibility_filter;
use crate::pool::{WriterPermit, WriterPool};
use crate::segment::Bm25Schema;

/// Eviction closure forwarded to [`WriterPool::acquire`] on every
/// cache-miss writer allocation. Two are captured by
/// [`Bm25IndexHandle`] at construction time, one per tier of ADR-039
/// amendment-02 §D-14's admission contract:
///
/// - the **eager** `on_full` sweep
///   ([`crate::Bm25Service::idle_sweeper`] → `evict_idle`, strict-idle
///   orphan reclamation — data-safe to run the instant the pool is
///   full); and
/// - the **timeout-gated** `on_block_timeout` evictor
///   ([`crate::Bm25Service::orphan_evictor`] → `evict_one_lru`, forced
///   LRU eviction — reached only after the admission block elapses with
///   no natural release; data-safe for any writer that commits within
///   [`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`], but a slower
///   in-flight writer (gap > the timeout) is reclaimed as an orphan and
///   loses its buffer — the accepted #575 residual per ADR-039
///   amendment-03 §D-18, tracked to #627).
///
/// The pool itself is unaware of the eviction policy; it merely
/// invokes the eager callback once, then (after the block timeout) the
/// forced callback. The shape is `Arc<dyn Fn() -> usize + Send + Sync>`
/// (returns the count freed) so each closure can be cloned cheaply
/// across calls and survive across `Bm25IndexHandle` Arc-clones.
pub(crate) type Sweeper = Arc<dyn Fn() -> usize + Send + Sync>;

/// Local v1.0 BM25 index identifier (ADR-039 §D-4).
///
/// Defined locally in `arcgraph-bm25` rather than in `arcgraph-core`
/// because the core IDs module is owned by the M3.b file boundary
/// and adding a new ID newtype there would cross the line. v1.1's
/// per-property index lift centralises the type into `arcgraph-core`
/// as part of M7 / OPEN-Q-3 unification.
///
/// The numeric domain (`u64`) matches the reservation shape in ADR-039
/// §D-4 so the centralisation lift is a rename / re-export, not a
/// representation change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct IndexId(pub u64);

impl IndexId {
    /// The single default BM25 index handle for v1.0 single-tenant
    /// deployments (ADR-039 §D-4). Every CRUD txn that buffers a
    /// BM25 upsert/delete routes to this `IndexId` for its tenant.
    pub const DEFAULT_BM25: Self = Self(0);

    /// Numeric zero — alias for [`Self::DEFAULT_BM25`] at v1.0.
    pub const ZERO: Self = Self(0);

    /// Construct from a raw u64.
    #[inline]
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Raw u64 representation.
    #[inline]
    #[must_use]
    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Filter shape consumed by [`Bm25IndexHandle::filtered_search`].
///
/// **v1.0 minimum.** Only [`Self::Any`] is supported; other variants
/// would require label / tenant FAST fields not present in the v1.0
/// schema (ADR-039 §D-2). The local definition exists rather than
/// re-exporting `arcgraph_vector::query::Filter` because
/// `arcgraph-bm25` deliberately does NOT depend on `arcgraph-vector`
/// (bounded-context discipline). v1.1 unifies via OPEN-Q-3:
/// `arcgraph-core::query::Filter` becomes the canonical type and
/// both `arcgraph-vector` and `arcgraph-bm25` re-export.
///
/// Variants are kept minimal at v1.0 (Any + Tenant) so the future
/// unification is additive — a v1.1 amendment that maps existing
/// callers from local Filter to canonical Filter does not break the
/// v1.0 wire shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Filter {
    /// No filtering — every match passes. The only variant supported
    /// at v1.0.
    Any,

    /// Tenant-scoped filter. v1.0 surfaces this as `FilterNotSupported`
    /// because the schema has no tenant FAST field; tenant isolation
    /// is enforced by per-tenant directory layout instead. The
    /// variant exists so future v1.1+ schema additions can opt
    /// callers into tenant-as-a-field without breaking the variant
    /// taxonomy.
    Tenant(TenantId),
}

/// One active per-tenant `IndexWriter` plus the [`WriterPermit`]
/// that admits it under the shared pool's capacity bound.
///
/// Construction: [`Bm25IndexHandle::ensure_writer`] allocates the
/// writer via `Index::writer(heap_bytes)` after acquiring a permit
/// from [`WriterPool::acquire`]. Drop: the permit is released
/// back to the pool, the writer's heap is freed by Tantivy, and the
/// directory lock is released.
pub(crate) struct ActiveWriter {
    pub(crate) writer: IndexWriter,
    /// Held alongside the writer for its entire lifetime. Dropping
    /// the [`ActiveWriter`] releases the permit back to the pool.
    /// The leading underscore appeases `dead_code` because the
    /// permit is intentionally only consulted via its `Drop`.
    _permit: WriterPermit,
}

impl std::fmt::Debug for ActiveWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveWriter")
            .field("writer", &"<tantivy::IndexWriter>")
            .field("permit", &"<WriterPermit>")
            .finish()
    }
}

/// The per-tenant Tantivy index state.
///
/// Held inside [`Bm25IndexHandle`] via [`Arc`] so multiple handles
/// returned by [`crate::Bm25Service::handle`] (one per `(tenant,
/// index)` key) share the same underlying index. The writer is
/// LAZY (per ADR-039 amendment-01 §D-11(b) / amendment-02 §D-13):
/// `None` when no writer is currently allocated for this tenant;
/// `Some(ActiveWriter)` between first write and eviction. The
/// `parking_lot::Mutex` serialises both the slot transition and the
/// per-tenant Tantivy mutation surface (ADR-039 OPEN-Q-2 closure;
/// see amendment-02 §D-15 contention measurement).
pub(crate) struct TantivyIndexInner {
    pub(crate) index: Index,
    /// Lazy `IndexWriter`. `None` after construction or eviction;
    /// `Some` after the first `upsert_document` /
    /// `delete_document` re-allocates via
    /// [`Bm25IndexHandle::ensure_writer`].
    pub(crate) writer: Mutex<Option<ActiveWriter>>,
    pub(crate) reader: IndexReader,
    pub(crate) schema: Bm25Schema,
    /// Shared admission pool (ADR-039 amendment-01 §D-11(c)). Held
    /// by `Arc` so every per-tenant `TantivyIndexInner` shares the
    /// SAME pool with the parent `Bm25Service`.
    pub(crate) pool: Arc<WriterPool>,
    /// Heap budget passed to `Index::writer(_)` on every (re-)
    /// allocation. Captured at construction so eviction-recreate
    /// cycles use the same sizing as the original allocation.
    pub(crate) heap_bytes: usize,
    /// Per-tenant idle counters consulted by
    /// [`crate::Bm25Service::evict_idle`].
    pub(crate) idle: IdleTracker,
}

impl std::fmt::Debug for TantivyIndexInner {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let writer_present = self.writer.lock().is_some();
        f.debug_struct("TantivyIndexInner")
            .field("index", &"<tantivy::Index>")
            .field("writer_present", &writer_present)
            .field("reader", &"<tantivy::IndexReader>")
            .field("schema", &self.schema)
            .field("pool", &self.pool)
            .field("heap_bytes", &self.heap_bytes)
            .field("idle", &self.idle)
            .finish()
    }
}

impl TantivyIndexInner {
    /// Drop the active writer, releasing its [`WriterPermit`] and
    /// freeing the Tantivy heap allocation. Returns `true` if a
    /// writer was actually evicted, `false` if the slot was already
    /// empty.
    ///
    /// Uses a blocking `lock()` — appropriate when called from
    /// outside the writer's hot path (e.g., the LRU eviction tail
    /// of [`crate::Bm25Service::evict_to_make_room`] after a
    /// candidate has been picked under `try_lock`).
    pub(crate) fn evict_writer(&self) -> bool {
        let mut guard = self.writer.lock();
        if guard.is_some() {
            // Drop the ActiveWriter — its `_permit` field's `Drop`
            // releases the pool permit and notifies one waiter.
            *guard = None;
            true
        } else {
            false
        }
    }

    /// Same as [`Self::evict_writer`] but uses `try_lock` — returns
    /// `false` if the writer mutex is currently held (by either
    /// the current thread or another thread). Used by the
    /// idle-sweep and LRU-iteration paths to avoid reentrant
    /// deadlock when the sweep is invoked from inside
    /// [`crate::Bm25IndexHandle::ensure_writer`].
    pub(crate) fn try_evict_writer(&self) -> bool {
        let Some(mut guard) = self.writer.try_lock() else {
            return false;
        };
        if guard.is_some() {
            *guard = None;
            true
        } else {
            false
        }
    }
}

/// Per-tenant BM25 search-side handle (ADR-039 §D-8).
///
/// Returned by [`crate::Bm25Service::handle`]; multiple calls for the
/// same `(tenant, index)` return the SAME `Arc<Bm25IndexHandle>`
/// (cache append-only at v1.0 per ADR-037 §D-6).
pub struct Bm25IndexHandle {
    tenant_id: TenantId,
    partition_id: PartitionId,
    index_id: IndexId,
    pub(crate) inner: Arc<TantivyIndexInner>,
    /// Eager, data-safe on-full sweep ([`WriterPool::acquire`]'s
    /// `on_full`): strict-idle orphan reclamation only
    /// ([`crate::Bm25Service::idle_sweeper`] → `evict_idle`). Captures
    /// a `Weak<Bm25Service>` so it can sweep without a strong reference
    /// cycle. Safe to run the instant the pool is found full — a
    /// strict-idle writer holds no committed-intent buffer.
    pub(crate) on_full_idle_sweep: Sweeper,
    /// Timeout-gated forced eviction ([`WriterPool::acquire`]'s
    /// `on_block_timeout`): forced LRU eviction to break an orphan-
    /// induced deadlock ([`crate::Bm25Service::orphan_evictor`] →
    /// `evict_one_lru`). Reached ONLY after the admission block elapses
    /// ([`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`]) with no natural
    /// permit release. A writer that commits within the timeout is
    /// never the victim; a slower in-flight writer (`upsert → commit`
    /// gap > the timeout) is indistinguishable from an orphan by timing
    /// alone and IS reclaimed, dropping its committed-intent buffer —
    /// the accepted #575 residual (ADR-039 amendment-03 §D-18; genuine
    /// close tracked to #627).
    pub(crate) on_block_timeout_evict: Sweeper,
}

impl std::fmt::Debug for Bm25IndexHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bm25IndexHandle")
            .field("tenant_id", &self.tenant_id)
            .field("partition_id", &self.partition_id)
            .field("index_id", &self.index_id)
            .field("inner", &self.inner)
            .field("on_full_idle_sweep", &"<Sweeper>")
            .field("on_block_timeout_evict", &"<Sweeper>")
            .finish()
    }
}

impl Bm25IndexHandle {
    /// Construct a handle from the freshly-opened Tantivy state.
    /// `pub(crate)` — only [`crate::Bm25Service`] materialises
    /// handles per ADR-039 §D-4 (cache invariant).
    pub(crate) fn new(
        tenant_id: TenantId,
        partition_id: PartitionId,
        index_id: IndexId,
        inner: Arc<TantivyIndexInner>,
        on_full_idle_sweep: Sweeper,
        on_block_timeout_evict: Sweeper,
    ) -> Self {
        debug_assert_eq!(
            partition_id,
            PartitionId::ZERO,
            "BM25 must be at PartitionId::ZERO"
        );
        debug_assert_eq!(
            index_id,
            IndexId::DEFAULT_BM25,
            "v1.0 BM25 must use IndexId::DEFAULT_BM25; per-property indexes \
             are v1.1+ per ADR-039 §D-4"
        );
        Self {
            tenant_id,
            partition_id,
            index_id,
            inner,
            on_full_idle_sweep,
            on_block_timeout_evict,
        }
    }

    /// Ensure an [`ActiveWriter`] exists for this tenant, allocating
    /// one (with a fresh [`WriterPermit`]) if the slot is empty.
    ///
    /// Returns the writer-mutex guard so callers can do their
    /// mutations under the same lock that admitted the writer (no
    /// Tantivy reentrancy hazard). The guard's `as_ref().unwrap()`
    /// is safe by construction — we either found `Some` or just
    /// stored `Some` above.
    ///
    /// On the slow path (writer slot empty), the pool's
    /// [`WriterPool::acquire`] is invoked with this handle's two
    /// eviction callbacks: [`Self::on_full_idle_sweep`] (eager,
    /// data-safe strict-idle reclamation) and
    /// [`Self::on_block_timeout_evict`] (timeout-gated forced LRU
    /// eviction for orphans). Per ADR-039 amendment-02 §D-14 the eager
    /// sweep runs before blocking; the forced eviction fires only after
    /// the block elapses with no natural permit release, so a writer
    /// that commits within [`crate::pool::WRITER_ACQUIRE_BLOCK_TIMEOUT`]
    /// is never its victim — but a slower in-flight writer (gap > the
    /// timeout) is reclaimed as an orphan and loses its buffer (the
    /// accepted #575 residual; ADR-039 amendment-03 §D-18, tracked to
    /// #627).
    pub(crate) fn ensure_writer(
        &self,
    ) -> Result<parking_lot::MutexGuard<'_, Option<ActiveWriter>>, Bm25Error> {
        let mut guard = self.inner.writer.lock();
        if guard.is_none() {
            let idle_sweep = Arc::clone(&self.on_full_idle_sweep);
            let orphan_evict = Arc::clone(&self.on_block_timeout_evict);
            let permit = self
                .inner
                .pool
                .acquire(move || idle_sweep(), move || orphan_evict());
            // Tantivy 0.26: `Index::writer(heap_bytes)` allocates a
            // fresh IndexWriter. On eviction-rewrite this re-opens
            // the on-disk segments via the same shared `Index`
            // instance — `Index::writer` is the natural entry-point
            // for both first-touch and post-eviction allocation.
            let writer = self
                .inner
                .index
                .writer(self.inner.heap_bytes)
                .map_err(|e| Bm25Error::Tantivy {
                    message: e.to_string(),
                })?;
            *guard = Some(ActiveWriter {
                writer,
                _permit: permit,
            });
        }
        Ok(guard)
    }

    /// Whether an active writer is currently allocated for this
    /// tenant. Test / observability helper; not used on the hot
    /// path.
    #[must_use]
    pub fn has_active_writer(&self) -> bool {
        self.inner.writer.lock().is_some()
    }

    /// Commit any buffered writes, reload the reader, and **drop
    /// the writer slot** (releasing the pool permit) per ADR-039
    /// amendment-02 §D-14 request-scoped semantics. Mirrors the
    /// per-tenant arm of
    /// [`arcgraph_storage::mutation_log::Bm25IndexStoreHandle::commit_pending`]
    /// for direct consumers (tests, ad-hoc tooling) that do not go
    /// through the kernel commit pipeline.
    ///
    /// If the writer slot is empty on entry (post-eviction or
    /// never-written), the commit is a no-op and the reader is
    /// still reloaded so any prior committed segments remain
    /// observable. The idle tracker is bumped on the commit axis
    /// regardless.
    ///
    /// # Errors
    ///
    /// - [`Bm25Error::Tantivy`] on commit / reload failure.
    pub fn commit(&self) -> Result<(), Bm25Error> {
        // Pattern-A early-take so the `WriterPermit` drops
        // unconditionally — the slot is set to `None` BEFORE the
        // fallible Tantivy commit runs, and the taken `ActiveWriter`
        // drops at the end of the match arm regardless of whether
        // `commit()` returned `Ok` or `Err`. Codex PR #221 F1
        // regression pin: prevents pool exhaustion under sustained
        // Tantivy I/O failure.
        let commit_result = {
            let mut guard = self.inner.writer.lock();
            match guard.take() {
                Some(mut active) => {
                    active
                        .writer
                        .commit()
                        .map(|_| ())
                        .map_err(|e| Bm25Error::Tantivy {
                            message: e.to_string(),
                        })
                }
                None => Ok(()),
            }
            // `active` drops here — `_permit.drop()` returns the
            // pool permit even when `commit()` returned `Err`.
        };
        commit_result?;
        self.inner.reader.reload().map_err(|e| Bm25Error::Tantivy {
            message: e.to_string(),
        })?;
        self.inner.idle.note_commit();
        Ok(())
    }

    /// Rollback any buffered writes and **drop the writer slot**
    /// per ADR-039 amendment-02 §D-14 request-scoped semantics.
    /// Mirrors the per-tenant arm of
    /// [`arcgraph_storage::mutation_log::Bm25IndexStoreHandle::rollback_pending`].
    /// If the writer slot is empty on entry, the rollback is a
    /// no-op.
    ///
    /// # Errors
    ///
    /// - [`Bm25Error::Tantivy`] on rollback failure.
    pub fn rollback(&self) -> Result<(), Bm25Error> {
        // Same Pattern-A early-take as `commit`. Codex PR #221 F1.
        // `active` (and its `_permit`) drops at the match arm's end,
        // releasing the pool permit even on the `Err` path.
        let mut guard = self.inner.writer.lock();
        match guard.take() {
            Some(mut active) => {
                active
                    .writer
                    .rollback()
                    .map(|_| ())
                    .map_err(|e| Bm25Error::Tantivy {
                        message: e.to_string(),
                    })
            }
            None => Ok(()),
        }
    }

    /// The tenant this handle is bound to.
    #[inline]
    #[must_use]
    pub fn tenant(&self) -> TenantId {
        self.tenant_id
    }

    /// The partition this handle is bound to. Always
    /// [`PartitionId::ZERO`] at v1.0.
    #[inline]
    #[must_use]
    pub fn partition(&self) -> PartitionId {
        self.partition_id
    }

    /// The index this handle is bound to. Always
    /// [`IndexId::DEFAULT_BM25`] at v1.0.
    #[inline]
    #[must_use]
    pub fn index(&self) -> IndexId {
        self.index_id
    }

    /// Top-K BM25 search at `read_lsn`.
    ///
    /// Composes the user-parsed query with the MVCC visibility filter
    /// (ADR-039 §D-3) and runs `Searcher::search` with
    /// `TopDocs::with_limit(k)`. Returns `(NodeId, score)` tuples
    /// sorted by descending score.
    ///
    /// # Query interpretation (ADR-039 §D-8 + #1220 demo-killer fix)
    ///
    /// The `query` string is treated as **free text**, NOT as Tantivy
    /// query-DSL syntax. It is run through the `body` field's own
    /// tokenizer (the same analyzer used at index time), and the
    /// resulting terms are OR-combined into a `BooleanQuery` of
    /// `TermQuery`s over `body`. This means metacharacters that the
    /// Tantivy `QueryParser` would otherwise interpret as operators —
    /// `:` (field-prefix), `[ ]` `{ }` (range), `^` (boost), `+ - !`
    /// (occur), `( )` (group), `* ?` (wildcard), `~` (fuzzy/slop),
    /// `"` (phrase), `\` `/`, and the bare keywords `AND OR NOT TO` —
    /// are NOT parsed as syntax. The tokenizer either folds them into
    /// ordinary tokens or strips them as inter-token separators, so a
    /// natural-language question like
    /// `"What is the status: open or closed? [URGENT] AND review"`
    /// searches as a bag of words and returns relevant hits instead of
    /// crashing with `QueryParse`.
    ///
    /// Relevance is preserved: OR-of-terms over a single field is
    /// exactly the shape the default `QueryParser` lowered a multi-word
    /// query to, so BM25 ranking (IDF × TF-saturation) is unchanged for
    /// ordinary questions. A query whose tokens are all stop-words /
    /// punctuation (or an empty string) yields zero terms → a
    /// match-nothing query → `Ok(vec![])` (unchanged v1.0 empty-query
    /// contract).
    ///
    /// The MVCC visibility / permission Must-clause
    /// ([`build_visibility_filter`]) is composed exactly as before — only
    /// the user-query construction changed, never the ACL/visibility
    /// filtering.
    ///
    /// # Errors
    ///
    /// - [`Bm25Error::Tantivy`] for any underlying engine error
    ///   (including the body tokenizer lookup).
    /// - [`Bm25Error::SchemaViolation`] if a matched doc lacks a
    ///   stored `node_id` (should never happen for v1.0-produced
    ///   docs).
    pub fn search(
        &self,
        query: &str,
        k: usize,
        read_lsn: Lsn,
    ) -> Result<Vec<(NodeId, f32)>, Bm25Error> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let schema = &self.inner.schema;
        let user_q = self.build_user_query(query)?;
        let visibility = build_visibility_filter(schema, read_lsn);
        let combined: Box<dyn Query> = Box::new(BooleanQuery::new(vec![
            (Occur::Must, user_q),
            (Occur::Must, visibility),
        ]));

        let searcher = self.inner.reader.searcher();
        // Tantivy 0.26: `TopDocs::with_limit(k)` does not directly
        // implement `Collector` for score-ordered results; call
        // `.order_by_score()` per the upstream usage pattern (see
        // `tantivy::lib.rs` doctest at module top).
        let top = searcher.search(&combined, &TopDocs::with_limit(k).order_by_score())?;

        let mut hits = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            let node_id = doc
                .get_first(schema.node_id)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Bm25Error::SchemaViolation {
                    detail: "matched doc has no stored node_id field".into(),
                })?;
            hits.push((NodeId::new(node_id), score));
        }
        Ok(hits)
    }

    /// Build the user-query clause as free text over the `body` field
    /// (#1220 demo-killer fix), faithfully mirroring how Tantivy's
    /// default `QueryParser` lowered an unquoted user query — but
    /// WITHOUT ever interpreting the input as query-DSL syntax.
    ///
    /// Construction:
    /// 1. Split the input on ASCII whitespace into whitespace-delimited
    ///    "words" (so `status:` `[URGENT]` `a^2` `AND` are each just a
    ///    word, never an operator).
    /// 2. Run each word through the `body` field's own analyzer.
    ///    - A word that tokenizes to a SINGLE token → a `TermQuery`.
    ///    - A word that tokenizes to MULTIPLE tokens (e.g. an
    ///      underscore-joined `first_doc_unique` → `first doc unique`,
    ///      or punctuation-split `open/closed`) → a `PhraseQuery` over
    ///      those adjacent tokens. This is exactly the QueryParser's
    ///      single-word-multi-token → phrase behaviour, so relevance
    ///      (notably the v1.0 `bm25_rollback_z1b` underscore fixture) is
    ///      preserved: `first_doc_unique` does NOT spuriously OR-match a
    ///      doc that merely shares `doc`/`unique`.
    /// 3. OR-combine the per-word clauses (`Occur::Should`) — the
    ///    default parser's disjunction-by-default bag-of-words shape.
    ///
    /// Because the input is never handed to the query-DSL parser, the
    /// special characters `: [ ] { } ^ + - ! ( ) ~ * ? " \ /` and the
    /// bare keywords `AND OR NOT TO` are treated as ordinary text —
    /// folded into tokens or dropped as separators by the tokenizer —
    /// and can no longer crash search with a `QueryParse` error.
    ///
    /// Returns a match-nothing [`EmptyQuery`] when no usable terms
    /// survive tokenization (empty string, all-punctuation, or
    /// all-stop-word input), preserving the v1.0 empty-query contract
    /// (`search("") -> Ok(vec![])`).
    fn build_user_query(&self, query: &str) -> Result<Box<dyn Query>, Bm25Error> {
        let schema = &self.inner.schema;
        // The body field's index-time analyzer; reusing it here keeps
        // query-time tokenization identical to index-time tokenization
        // so the produced terms line up with the posting lists.
        let mut analyzer = self.inner.index.tokenizer_for_field(schema.body)?;

        let mut clauses: Vec<(Occur, Box<dyn Query>)> = Vec::new();
        for word in query.split_whitespace() {
            // Tokenize THIS word in isolation so its tokens form a
            // contiguous run — the phrase grouping the default parser
            // applied to a single unquoted word.
            let mut token_stream = analyzer.token_stream(word);
            let mut word_terms: Vec<Term> = Vec::new();
            while token_stream.advance() {
                word_terms.push(Term::from_field_text(
                    schema.body,
                    &token_stream.token().text,
                ));
            }
            let clause: Box<dyn Query> = match word_terms.len() {
                0 => continue, // pure-punctuation / stop-word word: skip.
                // Single token → plain TermQuery. `WithFreqs` is enough
                // for BM25 scoring (no positions needed for a lone term).
                1 => Box::new(TermQuery::new(
                    word_terms.pop().expect("len==1 has one term"),
                    IndexRecordOption::WithFreqs,
                )),
                // Multi-token word → PhraseQuery over the adjacent tokens
                // (positions required; the `body` field is indexed
                // WithFreqsAndPositions via TEXT). Mirrors QueryParser.
                _ => Box::new(PhraseQuery::new(word_terms)),
            };
            // OR semantics across words — the disjunction-by-default
            // bag-of-words shape the default QueryParser produced.
            clauses.push((Occur::Should, clause));
        }

        if clauses.is_empty() {
            // No usable terms (empty / all-punctuation / all-stopword
            // input): match nothing, returning Ok(vec![]) to the caller.
            return Ok(Box::new(EmptyQuery));
        }
        Ok(Box::new(BooleanQuery::new(clauses)))
    }

    /// Top-K BM25 search with a filter applied (ADR-039 §D-8 / F.4
    /// dispatcher pattern).
    ///
    /// At v1.0 only [`Filter::Any`] is supported. Other variants
    /// surface as [`Bm25Error::FilterNotSupported`]; the consumer
    /// (M4 query layer) is responsible for fall-through dispatch
    /// per ADR-035 amendment-04 §D-3 escalation contract.
    pub fn filtered_search(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
        read_lsn: Lsn,
    ) -> Result<Vec<(NodeId, f32)>, Bm25Error> {
        match filter {
            Filter::Any => self.search(query, k, read_lsn),
            other => Err(Bm25Error::FilterNotSupported {
                variant: format!("{other:?}"),
            }),
        }
    }

    /// Buffer an upsert (`delete_term + add_document`) into the
    /// per-tenant `IndexWriter`.
    ///
    /// The writer is allocated lazily on first call (or on first
    /// call after eviction); the [`WriterPool`] admits at most
    /// [`crate::pool::WRITER_POOL_SIZE`] concurrently active
    /// writers across the whole `Bm25Service`. If the pool is full,
    /// this call invokes the eviction sweeper before falling
    /// through to a `Condvar` wait. The writer mutex still
    /// serialises all writes for this tenant per ADR-039 OPEN-Q-2.
    ///
    /// Writes are NOT durable until the kernel commit pipeline
    /// calls
    /// [`arcgraph_storage::mutation_log::Bm25IndexStoreHandle::commit_pending`]
    /// after WAL fsync success — producers MUST register the tenant
    /// via
    /// [`arcgraph_storage::TxnMutationLog::note_bm25_tenant`] so the
    /// rollback drain has a target on WAL fsync failure.
    pub fn upsert_document(
        &self,
        node_id: NodeId,
        text: &str,
        commit_lsn: Lsn,
    ) -> Result<(), Bm25Error> {
        let schema = &self.inner.schema;
        let guard = self.ensure_writer()?;
        // Safe by construction — `ensure_writer` returns a guard
        // whose slot is `Some` (it just stored, or was already
        // populated). The `as_ref().unwrap()` reflects that
        // post-condition.
        let active = guard
            .as_ref()
            .expect("ensure_writer post-condition: slot is Some");
        // Tantivy 0.26: `delete_term` / `add_document` are `&self`
        // (the IndexWriter is interior-mutable). The
        // `parking_lot::Mutex` on the slot still serialises per-
        // tenant writers (ADR-039 OPEN-Q-2 / amendment-02 §D-15).
        let term = Term::from_field_u64(schema.node_id, node_id.raw());
        active.writer.delete_term(term);
        active.writer.add_document(doc!(
            schema.node_id => node_id.raw(),
            schema.commit_lsn => commit_lsn.raw(),
            schema.expired_lsn => u64::MAX,
            schema.body => text,
        ))?;
        // Bump the idle tracker AFTER the Tantivy call so a failed
        // upsert leaves the tracker untouched — eviction policy
        // sees only successful writes as "not idle".
        self.inner.idle.note_write();
        Ok(())
    }

    /// Buffer a delete-by-term into the per-tenant `IndexWriter`.
    ///
    /// At v1.0 the `commit_lsn` is captured for audit / future v1.1
    /// doc-version tracking but unused (delete-term applies to all
    /// versions of `node_id`, of which v1.0 has at most one).
    /// Same buffered-semantics + lazy-writer contract as
    /// [`Self::upsert_document`].
    pub fn delete_document(&self, node_id: NodeId, _commit_lsn: Lsn) -> Result<(), Bm25Error> {
        let schema = &self.inner.schema;
        let guard = self.ensure_writer()?;
        let active = guard
            .as_ref()
            .expect("ensure_writer post-condition: slot is Some");
        let term = Term::from_field_u64(schema.node_id, node_id.raw());
        active.writer.delete_term(term);
        self.inner.idle.note_write();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_id_default_bm25_is_zero() {
        // ADR-039 §D-4: v1.0's single default BM25 index is
        // `IndexId(0)`. Pinned so a future renumbering surfaces here
        // rather than silently breaking on-disk directory layout.
        assert_eq!(IndexId::DEFAULT_BM25.raw(), 0);
        assert_eq!(IndexId::DEFAULT_BM25, IndexId::ZERO);
    }

    #[test]
    fn filter_any_is_default_v1() {
        let f = Filter::Any;
        match f {
            Filter::Any => {}
            other => panic!("Filter::Any must match itself, got {other:?}"),
        }
    }

    #[test]
    fn filter_tenant_renders_with_tenant_id() {
        let f = Filter::Tenant(TenantId::DEFAULT);
        let dbg = format!("{f:?}");
        assert!(dbg.contains("Tenant"), "{dbg}");
    }
}
