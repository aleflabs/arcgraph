//! ArcGraph — graph, vector, full-text, and community database engine.
//!
//! This crate is the **umbrella facade**: a single `cargo add arcgraph`
//! pulls in the full public API of the workspace via stable module
//! paths. The implementation lives in the nine bounded-context crates:
//!
//! - [`core`] — shared primitives (ID newtypes, error taxonomy, record
//!   layouts, cache-line primitives). Re-export of
//!   [`arcgraph_core`].
//! - [`storage`] — buffer pool, WAL, page format, TEL adjacency, MVCC.
//!   Re-export of [`arcgraph_storage`].
//! - [`index`] — secondary B-tree indices (per ADR-030 DEC-21
//!   relaxation). Re-export of [`arcgraph_index`].
//! - [`vector`] — HNSW + DiskANN vector retrieval, SIMD-backed distance
//!   kernels (per ADR-035). Re-export of [`arcgraph_vector`].
//! - [`bm25`] — Tantivy-backed BM25 full-text search (per ADR-039).
//!   Re-export of [`arcgraph_bm25`].
//! - [`community`] — GVE-Leiden + DF Leiden community detection (per
//!   ADR-040). Re-export of [`arcgraph_community`].
//! - [`query`] — ArcQL parser, planner, executor, [`query::QueryEngine`].
//!   Re-export of [`arcgraph_query`].
//! - [`mcp`] — MCP server, JSON-RPC dispatch, Tier-1 tools, stdio /
//!   HTTP / Bolt transports. Re-export of [`arcgraph_mcp`].
//!
//! # Embedded usage
//!
//! The minimal "open a catalog, run a query, inspect results" loop
//! (this is the same code that ships in `examples/embedded_quickstart.rs`):
//!
//! ```rust
//! use arcgraph::core::{LabelId, NodeId, TenantId};
//! use arcgraph::query::QueryEngine;
//! use arcgraph::query::executor::StubExecutorSubstrate;
//! use arcgraph::query::executor::value::{NodeView, Value};
//! use arcgraph::query::semantic::StubCatalogProvider;
//!
//! let mut substrate = StubExecutorSubstrate::new();
//! for i in 1..=3 {
//!     substrate = substrate.with_node(
//!         TenantId::DEFAULT,
//!         NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
//!             .with_property("age", Value::Integer(i as i64 * 10)),
//!     );
//! }
//! let catalog = StubCatalogProvider::new()
//!     .with_labels(["Person"])
//!     .with_properties(["age"]);
//!
//! let engine = QueryEngine::new(&catalog);
//! let result = engine
//!     .execute("MATCH (n:Person) RETURN n", &substrate)
//!     .expect("execute MATCH");
//! assert_eq!(result.rows().len(), 3);
//! ```
//!
//! All imports above route through `arcgraph::*` so a downstream
//! consumer with only `arcgraph` in their `Cargo.toml` compiles the
//! snippet verbatim. The integration test
//! `tests/doc_test_is_portable.rs` synthesizes a sibling consumer crate
//! and `cargo build`s it against just the umbrella to pin this
//! invariant.
//!
//! # Server / CLI usage
//!
//! The `arcgraph` binary (built from `crates/arcgraph-cli`) wraps the
//! [`mcp::serve_stdio`] / [`mcp::serve_http`] / [`mcp::serve_bolt_listener`]
//! entry points re-exported here so an operator can run
//! `arcgraph serve --stdio-mcp` (or `--http <addr>` / `--bolt <addr>`)
//! without depending on the individual crates.
//!
//! # Versioning + stability
//!
//! At v1.0-alpha the workspace ships a single `0.0.0` synchronized
//! version (see `Cargo.toml` `[workspace.package]`). The first crates.io
//! release will establish the stability commitments alongside a real semver.
//!
//! The umbrella's surface is **curated** (not blanket-re-exported): each
//! `pub mod` block enumerates the load-bearing types from the underlying
//! crate. Adding a `pub` name to an underlying crate does NOT
//! automatically widen the umbrella's surface — a follow-up curation
//! edit here is required. This insulates downstream consumers from
//! intra-workspace refactors of internal-only `pub` names.
//!
//! # Prime Directives
//!
//! Per the repository core design principles:
//! - Apache-2.0 throughout — every transitive dep verified at the
//!   workspace root.
//! - No `unsafe` here; this is a pure re-export facade.
//! - No `mmap` on the hot path; the storage crate's discipline is
//!   inherited.

#![recursion_limit = "256"]

/// Shared primitives: IDs, errors, records, cache-line types.
///
/// Curated re-export of [`arcgraph_core`]; mirrors the
/// `bounded-contexts.md` §`arcgraph-core` "Owns" list (ID newtypes +
/// error taxonomy + record layouts + cache-line primitives). Submodule
/// paths (`core::ids`, `core::error`, `core::record`, …) remain
/// reachable so deep callers can still navigate.
pub mod core {
    pub use arcgraph_core::{
        AlwaysStrict, ArcGraphError, CacheAligned, DurabilityTier, DurabilityTierError, LabelId,
        Lsn, NodeId, NodeRecord, PAGE_SIZE, PageHeader, PageId, PageType, PartitionId, PropertyId,
        RelId, RelRecord, Result, StringId, TelEntry, TenantDurabilityLookup, TenantId, TypeId,
    };
    pub use arcgraph_core::{cache_aligned, durability, error, ids, record};
}

/// Storage engine: buffer pool, WAL, TEL, MVCC, CRUD.
///
/// Curated re-export of [`arcgraph_storage`]. The crate's `test_harness`
/// submodule is **deliberately omitted** from this surface per
/// `bounded-contexts.md` §`arcgraph-storage` line 54: `test_harness::*`
/// is `pub` so integration tests in `crates/arcgraph-storage/tests/*.rs`
/// can consume it, NOT for downstream embedded users. Integration tests
/// reach it via the direct `arcgraph-storage` dep; downstream consumers
/// do not get it through the umbrella. (W15α F-3 fix-up; closes the
/// W11 R-12 stability-commitment hole flagged at PR #306 round-1.)
pub mod storage {
    pub use arcgraph_storage::{
        AllocatorAdvance, AllocatorKind, AllocatorSeedHandle, BLOB_CHUNK_BYTES, BLOB_MAX_BYTES,
        BLOB_PAGE_HEADER, BUNDLE_FORMAT_V1, BUNDLE_FORMAT_V2, BUNDLE_FORMAT_V3, BUNDLE_FORMAT_V4,
        BackgroundFsyncFailAction, BackgroundFsyncMetrics, BackgroundFsyncScheduler, BlobError,
        BlobPageSnapshot, BlobRef, BlobStore, BlobStoreHandle, BufferPool, CATALOG_PAGE_ID,
        CatalogStats, CrudAllocatorSeedHandle, CrudStoreGraphAdapter, DEFAULT_WRITE_FRACTION,
        DecodedCommitBundle, DecodedIndexPage, EngineConfig, EngineError, EngineHandles, Frame,
        FrameId, FrameReadGuard, FrameWriteGuard, GraphAdapterError, InMemoryPageIo, IndexHandle,
        InlineShape, InternTable, MultiTenantRouter, OVERFLOW_BIT, OVERFLOW_PAGE_BITS,
        OVERFLOW_PAGE_MASK, OVERFLOW_SLOT_BITS, OVERFLOW_SLOT_MASK, PageAllocator, PageIo,
        PageStoreKind, PageTable, PosixPageIo, ProductionRefreshHook, ProductionRefreshHookError,
        PropertyReadout, RecordPageStore, RecordStoreError, RoutingError, STRINGID_SENTINEL,
        SecondaryIndexHandle, SecondaryIndexHandleError, SecondaryIndexValue, SideChannelWrite,
        StagedBlob, StagedEmit, SystemCatalog, TenantHandle, TenantRecord, TornTail,
        TxnMutationLog, VectorPageStore, VectorPageStoreArc, VectorPageStoreHandle,
        VectorStoreError, WalConfig, WalErrorPolicy, WalFireMetrics, WalHandle, WalRecord,
        WalRecordType, WalRecoveryReader, WalWriter, audit_fsync_barriers, bootstrap_engine,
        crud_allocator_seed_handle, decode_commit_bundle, decode_commit_bundle_for_version,
        decode_commit_bundle_v1, decode_commit_bundle_v2, decode_commit_bundle_v3,
        decode_commit_bundle_v4, decode_durability_tier, decode_intern_payload,
        decode_property_node, decode_property_rel, decode_put_blob_payload, encode_commit_bundle,
        encode_commit_bundle_v2, encode_commit_bundle_v3, encode_commit_bundle_v4,
        encode_inline_node, encode_inline_rel, encode_intern_payload, encode_overflow_node,
        encode_overflow_rel, encode_put_blob_payload, intern_logged,
    };
    pub use arcgraph_storage::{
        blob, buffer, catalog, config, crud, engine, intern, io, mutation_log, page_alloc,
        primary_index, property, record_store, records, recovery, router, secondary_handle, tel,
        transaction, vector_store, wal,
    };
}

/// Secondary B-tree indices (per ADR-030 DEC-21 relaxation — index
/// pages are pre-durable read accelerators governed by the staged-
/// emit / drain-outside-locks contract).
///
/// Curated re-export of [`arcgraph_index`]. v1.0-α covers
/// the secondary B-tree surface; HNSW + DiskANN substrates live in
/// [`vector`] (per ADR-035); BM25 / full-text lives in [`bm25`]
/// (per ADR-039). (W15α F-2 fix-up; closes the umbrella's
/// substrate-omission gap flagged at PR #306 round-1 against the
/// graph, vector, full-text, and traversal surface in Cargo.toml.
/// W15α R2-NEW-1 fix-up retargets the cite from ADR-003 — which
/// governs HNSW deletion strategy, not secondary B-tree lifecycle
/// — to ADR-030, whose §Decision names `secondary_btree.rs` as a
/// load-bearing subject. Mirrors `bounded-contexts.md:252`
/// canonical "ADR-030 DEC-21 relaxation" cite-form.)
pub mod index {
    pub use arcgraph_index::secondary_btree;
    pub use arcgraph_index::{
        INLINE_NODEID_COUNT, INTERNAL_CAPACITY, INTERNAL_ENTRY_OFFSET, INTERNAL_ENTRY_SIZE,
        INTERNAL_FIRST_CHILD_OFFSET, InternalPageMut, InternalPageRef, LEAF_CAPACITY,
        LEAF_ENTRY_OFFSET, LEAF_ENTRY_SIZE, LeafEntry, LeafFindResult, LeafPageMut, LeafPageRef,
        OVERFLOW_FILLED_COUNT_OFFSET, OVERFLOW_NEXT_OFFSET, OVERFLOW_SLOTS_OFFSET,
        OVERFLOW_SLOTS_PER_PAGE, OverflowPageMut, OverflowPageRef, PageBuf, PageLatch,
        PropertyValue, SECONDARY_INDEX_ROOT_KEY, SecondaryIndex, SecondaryIndexError, SecondaryKey,
        SecondaryPageStore, SplitInfo, fresh_page_buf,
    };
}

/// Vector index engine: HNSW + DiskANN, SIMD-backed distance kernels.
///
/// Curated re-export of [`arcgraph_vector`] (per ADR-035). (W15α F-2
/// fix-up; closes the umbrella's vector-substrate omission gap flagged
/// at PR #306 round-1.)
pub mod vector {
    pub use arcgraph_vector::{
        ArenaLabelsRef, ArenaSliceRef, BackendKind, BackendSet, DispatchPreference, DistanceKernel,
        Encoding, Filter, FilteredVectorIndex, IndexId, IndexType, Metric, PropertyKey,
        PropertyValue, QuantizerState, Result, Sq8Params, VectorArena, VectorArenaRegistry,
        VectorId, VectorIndexError, VectorIndexHandle, dispatch_preference,
    };
    pub use arcgraph_vector::{
        arena, diskann, dispatcher, distance, encoding, error, handle, hnsw, ids, quantizer, query,
    };
}

/// BM25 text search via Tantivy.
///
/// Curated re-export of [`arcgraph_bm25`] (per ADR-036 D-2 + ADR-039).
/// (W15α F-2 fix-up; closes the umbrella's text-substrate omission gap
/// flagged at PR #306 round-1.)
pub mod bm25 {
    pub use arcgraph_bm25::{
        Bm25Error, Bm25IndexHandle, Bm25Schema, Bm25Service, Filter,
        IDLE_EVICTION_COMMIT_THRESHOLD, IDLE_EVICTION_WALL_CLOCK_THRESHOLD_SECS, IndexId,
        WRITER_POOL_SIZE, build_visibility_filter,
    };
    pub use arcgraph_bm25::{error, eviction, handle, mvcc, pool, segment, service, store};
}

/// Community detection: GVE-Leiden static + DF Leiden incremental.
///
/// Curated re-export of [`arcgraph_community`] (per ADR-040). (W15α F-2
/// fix-up; closes the umbrella's community-substrate omission gap
/// flagged at PR #306 round-1.)
pub mod community {
    pub use arcgraph_community::{
        BTreeMembershipIndex, CommunityError, CommunityId, CommunityIndexHandle, CommunityIndexId,
        CommunityIndexProvider, CommunityRefreshScheduler, EdgeUpdate, Graph, GveLeiden,
        IncrementalResult, LeidenIncremental, LeidenParams, LeidenResult, Level, MembershipIndex,
        OwnedRefreshInputs, RefreshHook, RefreshObserver, Result, SchedulerConfig, SchedulerHealth,
        SharedBTreeIndexProvider, modularity,
    };
    pub use arcgraph_community::{
        error, graph, handle, ids, index, leiden_incremental, leiden_static, membership_index,
        provider, scheduler,
    };
}

/// Query engine: ArcQL parser, planner, executor, `QueryEngine`.
///
/// Curated re-export of [`arcgraph_query`]. Surface mirrors the spawn-
/// prompt example list plus types the embedded-quickstart example +
/// CLI binary consume. Submodule paths (`query::executor::value`,
/// `query::semantic`, …) remain reachable so deep callers can still
/// navigate.
pub mod query {
    pub use arcgraph_query::{
        BATCH_ROWS, BUDGET_FALLBACK_ROWS, Batch, BinOp, BoundEdge, BoundNode, BreachDirection,
        CancellationError, CancellationRegistry, CancellationToken, Clause,
        DEFAULT_MAX_ENTRIES_PER_TENANT, DEFAULT_QUERY_TIMEOUT_MS, DEFAULT_THRESHOLD_FACTOR,
        DeadlineHandle, ExecutionContext, ExecutionError, ExecutionMetrics, ExecutorSubstrate,
        ExplainError, Expression, FieldRef, Fusion, LengthRange, Literal, LookupOutcome, MatchBody,
        MatchClause, MaterializedResult, MemoryBudget, MemoryReservation, NamedPath, NamedPathKind,
        NodePattern, NumericLiteral, ObservedStatsOverrides, OperatorKind, OperatorMetrics,
        OrderDirection, OrderItem, ParseError, PathPattern, PhysicalOperator, Pipeline, PlanCache,
        PlanCacheEntry, PlanCacheKey, PlanTree, PlanTreeOp, PlanWalkEntry, ProjectionItem,
        ProjectionKind, PropertyMap, QueryEngine, QueryId, RankArg, RankByClause, RankedHit,
        Ranker, ReadQuery, RelDirection, RelPattern, ReplanController, ReplanError, ReplanOutcome,
        ReplanReason, ReturnClause, RowCountObserver, SnapshotLsnGuard, Span, Statement,
        StreamingCursor, StubExecutorSubstrate, SubstrateAccessError, ThreeValued, ThresholdBreach,
        UnaryOp, UnwindClause, Value, ValueJsonError, WithClause, WithFusionClause,
        apply_overrides_to_stub_catalog, estimate_row_bytes, estimate_value_bytes, execute,
        execute_with_context, explain, explain_with_cache, materialize, materialize_multi, parse,
        parse_multi, profile, walk_plan_and_costs,
    };
    // `explain` and `materialize` are intentionally absent here: both
    // appear in the first `pub use arcgraph_query::{…}` block above
    // because each is BOTH a function (value namespace) and a module
    // (type namespace) in `arcgraph_query`. A single `pub use` imports
    // from both namespaces, so re-listing them here would E0252 with
    // "name defined multiple times".
    pub use arcgraph_query::{
        ast, cancel, cursor, error, executor, logical_plan, observer, parser, planner, semantic,
    };
}

/// MCP server surface: dispatch, Tier-1 tools, stdio / HTTP / Bolt
/// transports.
///
/// Curated re-export of [`arcgraph_mcp`]. Submodule paths
/// (`mcp::transport::bolt`, `mcp::tools::search`, `mcp::serializers`,
/// …) remain reachable so the umbrella CLI binary + deep callers can
/// navigate.
pub mod mcp {
    pub use arcgraph_mcp::{
        AvailableSubstrates, BoltError, BoltQueryHandler, BoltServeStats, BoltServerConfig,
        BoltVersion, CODE_CANCELLED, CODE_EXECUTION_EVAL, CODE_INDEX_UNAVAILABLE,
        CODE_INTERNAL_ERROR, CODE_INVALID_PARAMS, CODE_INVALID_REQUEST, CODE_METHOD_NOT_FOUND,
        CODE_PARSE_ERROR, CODE_QUERY_ERROR, CODE_RATE_LIMITED, CODE_TENANT_UNKNOWN,
        CODE_UNAUTHORIZED, ClassPolicy, ClientMessage, ConnFsm, ConnState, DEFAULT_EXPLORE_DEPTH,
        DEFAULT_EXPLORE_LIMIT, DEFAULT_READ_CAPACITY, DEFAULT_READ_REFILL_PER_SEC,
        DEFAULT_REQUEST_DEADLINE, DEFAULT_SEARCH_K, DEFAULT_WRITE_CAPACITY,
        DEFAULT_WRITE_REFILL_PER_SEC, DispatchBulkhead, Dispatcher, ExitReason, ExploreRequest,
        GraphSchema, HEADER_ORIGIN, HEADER_TENANT, HandlerOutcome, HttpExitReason, HttpServeStats,
        HttpServerConfig, HybridSearcher, IndexDescriptor, IndexKind, IngestBatch, IngestError,
        IngestProvider, IngestRecordOutcome, IngestRequest, IngestSummary, InspectRequest,
        JSONRPC_VERSION, JsonRpcErrorObject, JsonRpcErrorResponse, JsonRpcRequest, JsonRpcResponse,
        LabelInfo, MAX_EXPLORE_DEPTH, MAX_EXPLORE_LIMIT, MAX_MESSAGE_BYTES, MAX_SEARCH_K, MCPError,
        METHOD_GRAPH_EXPLORE, METHOD_GRAPH_INGEST, METHOD_GRAPH_INSPECT, METHOD_GRAPH_SCHEMA,
        METHOD_GRAPH_SEARCH, MetricsRegistry, NeighborDirection, NeighborInfo, Neighborhood,
        NeighborhoodEdge, NeighborhoodExplorer, NeighborhoodNode, NodeIngest, NodeInspection,
        NodeInspector, OpClass, PATH_HEALTHZ, PATH_MCP, PackValue, PropertyDescriptor,
        RateLimitConfig, RateLimitError, RateLimiter, RawQueryExecutor, RawQueryRequest,
        RawQueryRows, RelIngest, RelTypeInfo, ResponseFormat, RunOutcome, SUBSTRATE_SLUG_BM25,
        SUBSTRATE_SLUG_VECTOR, SchemaProvider, SchemaRequest, SearchHit, SearchRequest,
        SearchResult, ServeStats, ServerMessage, SessionScope, StubBoltHandler, StubFault,
        TenantPolicy, TenantStrategy, TransportError, client_verifier_for_roots, decode_request,
        explore_tool, handle_pair, handle_raw_envelope, ingest_tool, inspect_tool, read_message,
        render_response, schema_tool, search_tool, serve_bolt_listener, serve_http, serve_stdio,
        shutdown_on_term, substrate_kinds, write_message,
    };
    pub use arcgraph_mcp::{
        error, jsonrpc, rate_limit, serializers, storage, tls, tools, transport,
    };
}

#[cfg(test)]
mod tests {
    // Each test resolves a name from each underlying crate via the
    // umbrella's module-path. The test passes by compiling — the
    // assertions are belt-and-braces.

    #[test]
    fn core_reexports_resolve() {
        // ID newtypes + error type re-exported via `crate::core`.
        let _: crate::core::TenantId = crate::core::TenantId::DEFAULT;
        let _: crate::core::NodeId = crate::core::NodeId::new(1);
        let _: crate::core::RelId = crate::core::RelId::new(1);
        let _: crate::core::Lsn = crate::core::Lsn::ZERO;
        let _: crate::core::LabelId = crate::core::LabelId::new(1);
        let _: Result<(), crate::core::ArcGraphError> = Ok(());
    }

    #[test]
    fn storage_reexports_resolve() {
        // Storage surface — these are types, not constructors; the
        // existence-check is the test (each `_:` is a compile-time
        // resolve).
        let _: Option<crate::storage::EngineConfig> = None;
        let _: Option<crate::storage::PageAllocator> = None;
        let _: Option<crate::storage::WalConfig> = None;
        let _: Option<crate::storage::BackgroundFsyncMetrics> = None;
    }

    #[test]
    fn query_reexports_resolve() {
        // Public surface needed by an embedded caller: parser,
        // engine, executor primitives, stub substrate.
        let _ = crate::query::parse("MATCH (n) RETURN n");
        let _: crate::query::executor::StubExecutorSubstrate =
            crate::query::executor::StubExecutorSubstrate::new();
        let _: crate::query::semantic::StubCatalogProvider =
            crate::query::semantic::StubCatalogProvider::new();
        // MaterializedResult constructs via Default at re-export path.
        let _: crate::query::MaterializedResult = crate::query::MaterializedResult::default();
        // QueryEngine resolves at type level — full construction
        // (which requires a borrowed catalog) is exercised by
        // `umbrella_doc_test_smoke` below.
        let catalog = crate::query::semantic::StubCatalogProvider::new();
        let _: crate::query::QueryEngine<'_, _> = crate::query::QueryEngine::new(&catalog);
    }

    #[test]
    fn mcp_reexports_resolve() {
        // Error taxonomy + JSON-RPC envelopes re-export.
        let _: crate::mcp::MCPError = crate::mcp::MCPError::InvalidRequest("smoke".into());
        // Transport surface constants resolve (the symbols exist at
        // the re-export path); each is `&'static str` / `Duration`.
        let _: &str = crate::mcp::PATH_MCP;
        let _: &str = crate::mcp::PATH_HEALTHZ;
        let _: &str = crate::mcp::HEADER_TENANT;
        // JSON-RPC envelope shape + version.
        let _: &str = crate::mcp::JSONRPC_VERSION;
        // Bolt transport names (W14δ M5-13) resolve.
        let _: crate::mcp::BoltVersion = crate::mcp::BoltVersion::V5_0;
        // Tier-1 method names resolve.
        let _: &str = crate::mcp::METHOD_GRAPH_SCHEMA;
        let _: &str = crate::mcp::METHOD_GRAPH_INSPECT;
        let _: &str = crate::mcp::METHOD_GRAPH_EXPLORE;
        let _: &str = crate::mcp::METHOD_GRAPH_SEARCH;
        let _: &str = crate::mcp::METHOD_GRAPH_INGEST;
    }

    #[test]
    fn substrate_reexports_resolve() {
        // W15α F-2 fix-up: each newly-added substrate module
        // re-exports at least one load-bearing type. Compile-time
        // resolve is the assertion.
        let _: Option<crate::index::SecondaryIndexError> = None;
        let _: Option<crate::vector::VectorIndexError> = None;
        let _: Option<crate::bm25::Bm25Error> = None;
        let _: Option<crate::community::CommunityError> = None;
        // Submodule paths reachable too.
        let _: Option<crate::index::secondary_btree::SecondaryIndexError> = None;
        let _: Option<crate::vector::handle::VectorIndexHandle> = None;
        let _: Option<crate::bm25::handle::Bm25IndexHandle> = None;
        let _: Option<crate::community::handle::CommunityIndexHandle> = None;
    }

    #[test]
    fn umbrella_doc_test_smoke() {
        // Mirror the doc-test loop without the `#[doc]` extraction
        // path so the smoke is part of the regular test target too.
        use crate::core::{LabelId, NodeId, TenantId};
        use crate::query::QueryEngine;
        use crate::query::executor::StubExecutorSubstrate;
        use crate::query::executor::value::{NodeView, Value};
        use crate::query::semantic::StubCatalogProvider;

        let mut substrate = StubExecutorSubstrate::new();
        for i in 1..=3 {
            substrate = substrate.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("age", Value::Integer(i as i64 * 10)),
            );
        }
        let catalog = StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_properties(["age"]);

        let engine = QueryEngine::new(&catalog);
        let result = engine
            .execute("MATCH (n:Person) RETURN n", &substrate)
            .expect("execute MATCH");
        assert_eq!(result.rows().len(), 3);
    }

    #[test]
    fn umbrella_executes_match_with_filter_and_projection() {
        // Second behavior pin: WHERE + RETURN projection. Sanity
        // that the umbrella's re-export of executor/value/Value
        // covers the predicate path. (Closes M6-01 unit-test
        // coverage on the embedded-quickstart surface.)
        use crate::core::{LabelId, NodeId, TenantId};
        use crate::query::QueryEngine;
        use crate::query::executor::StubExecutorSubstrate;
        use crate::query::executor::value::{NodeView, Value};
        use crate::query::semantic::StubCatalogProvider;

        let mut substrate = StubExecutorSubstrate::new();
        for i in 1..=5u64 {
            substrate = substrate.with_node(
                TenantId::DEFAULT,
                NodeView::new(NodeId::new(i), Some(LabelId::new(1)))
                    .with_property("age", Value::Integer(i as i64 * 5)),
            );
        }
        let catalog = StubCatalogProvider::new()
            .with_labels(["Person"])
            .with_properties(["age"]);

        let engine = QueryEngine::new(&catalog);
        let result = engine
            .execute("MATCH (n:Person) WHERE n.age > 10 RETURN n.age", &substrate)
            .expect("execute MATCH+WHERE+RETURN");
        // age = 5*i for i in 1..=5 → ages 5,10,15,20,25. Predicate
        // > 10 keeps 15,20,25 → 3 rows.
        assert_eq!(result.rows().len(), 3);
    }
}
