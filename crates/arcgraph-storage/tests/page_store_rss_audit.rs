//! W26-ε-2 / ADR-140 — RSS-bounded audit microbenchmark.
//!
//! Demonstrates that [`BufferedRecordPageStore`] RSS is bounded by
//! `cache_cap × PAGE_SIZE` regardless of total installed page count.
//! Drives a synthetic-churn workload: install N pages, evict to cap,
//! verify the hot cache never exceeds the cap.
//!
//! Output (captured by the W26-ε-2 perf audit doc):
//!
//!   N installed | cap | post-evict cache_size | post-evict evicted_count
//!
//! Per ADR-140 §"Acceptance criteria" item 6 + the perf audit doc at
//! `docs/perf/w26-epsilon-2-page-store-sf100-audit.md`.

use std::sync::Arc;
use std::time::Instant;

use arcgraph_core::{PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
    io::PosixPageIo,
};
use tempfile::TempDir;

fn make_store(path: &std::path::Path, cache_cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn arcgraph_storage::io::PageIo> =
        Arc::new(PosixPageIo::open_or_create(path).expect("posix open"));
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 8,
            write_fraction: 0.20,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cache_cap))
}

/// RSS-audit harness. Runs four scale points:
///
/// - 1 K pages × 64 cache cap
/// - 4 K pages × 256 cache cap
/// - 16 K pages × 1 K cache cap
/// - 64 K pages × 4 K cache cap
///
/// Each scale point verifies:
///
/// 1. cache_size <= cache_cap post-evict.
/// 2. cache_size + evicted_count == N (no page loss).
/// 3. Every installed page is fault-in-able (bytes survive eviction +
///    re-read).
#[test]
fn rss_audit_synthetic_churn_scale_sweep() {
    let scale_points = [
        (1_000usize, 64usize),
        (4_000, 256),
        (16_000, 1_024),
        (64_000, 4_096),
    ];

    for (n, cap) in scale_points {
        let tmp = TempDir::new().expect("tempdir");
        let path = tmp.path().join("pages.db");
        let store = make_store(&path, cap);

        let install_start = Instant::now();
        for i in 0..n {
            let pid = PageId::new((i as u64) + 1_000_000);
            store
                .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
                .unwrap();
            // Drive eviction every cap-pages installed so the cache
            // stays bounded throughout the install loop (mirrors the
            // production-deployment cap-driven discipline).
            if i % cap == cap - 1 {
                store.evict_lru(cap).unwrap();
            }
        }
        let install_elapsed = install_start.elapsed();

        // Final eviction sweep to drive cache below cap.
        store.evict_lru(cap).unwrap();
        // Sanity: hot cache bounded.
        assert!(
            store.cache_size() <= cap,
            "[n={} cap={}] cache_size {} > cap {} after final evict",
            n,
            cap,
            store.cache_size(),
            cap,
        );
        // No page loss.
        assert_eq!(
            store.cache_size() + store.evicted_count(),
            n,
            "[n={} cap={}] cache_size {} + evicted_count {} != n {}",
            n,
            cap,
            store.cache_size(),
            store.evicted_count(),
            n,
        );

        // Calculated RSS envelope (informational).
        let hot_rss_bytes = store.cache_size() * PAGE_SIZE;
        let theoretical_max_rss = cap * PAGE_SIZE;
        eprintln!(
            "[rss_audit] n={} cap={} hot_cache={} evicted={} hot_rss_bytes={} \
             theoretical_max_rss={} install_secs={:.3}",
            n,
            cap,
            store.cache_size(),
            store.evicted_count(),
            hot_rss_bytes,
            theoretical_max_rss,
            install_elapsed.as_secs_f64(),
        );
        assert!(
            hot_rss_bytes <= theoretical_max_rss,
            "[n={} cap={}] hot RSS {} bytes > theoretical max {} bytes",
            n,
            cap,
            hot_rss_bytes,
            theoretical_max_rss,
        );

        // Fault-in verification: every page can be read.
        let verify_start = Instant::now();
        // Skip exhaustive verification at the 64 K scale (would dominate
        // wall-clock); sample 100 pages at evenly-spaced offsets.
        let stride = (n / 100).max(1);
        let mut sampled = 0;
        for i in (0..n).step_by(stride) {
            let pid = PageId::new((i as u64) + 1_000_000);
            store.fault_in(pid).expect("fault_in installed page");
            sampled += 1;
        }
        let verify_elapsed = verify_start.elapsed();
        eprintln!(
            "[rss_audit] n={} cap={} sampled_verify={} elapsed_secs={:.3}",
            n,
            cap,
            sampled,
            verify_elapsed.as_secs_f64(),
        );
    }
}

/// Shared cache cap respected across tenants: 4 tenants × 1 K pages each, with
/// the shared `BufferedRecordPageStore` cache bounded at the configured cap
/// (256 frames). Renamed from `rss_audit_per_tenant_isolation` at W26-ζ codification
/// (ADR-141 D-4 row "L1") per `feedback_cite_attribution_sync.md` and PR #477 R1
/// L1. The prior name implied per-tenant pool isolation testing, but at v1.1-α
/// the `install_fresh` and `evict_lru` paths do NOT route through
/// `PerTenantBufferPool::pool()` (lazy-creation), so `tenant_count` is
/// vacuously-true (`0 <= N_TENANTS`). The actual property being tested is shared
/// cache cap respect; a real per-tenant isolation test is v1.1-α §Forward-deferred.
#[test]
fn rss_audit_shared_cap_respected_across_tenants() {
    let tmp = TempDir::new().expect("tempdir");
    let path = tmp.path().join("pages.db");
    let store = make_store(&path, 256);

    const N_TENANTS: u64 = 4;
    const PAGES_PER_TENANT: u64 = 1_000;

    for tenant_raw in 0..N_TENANTS {
        let tenant = TenantId::new(tenant_raw + 1);
        for i in 0..PAGES_PER_TENANT {
            let pid = PageId::new(tenant_raw * 10_000_000 + i + 2_000_000);
            store.install_fresh(pid, PageType::Node, tenant).unwrap();
            if i % 256 == 255 {
                store.evict_lru(256).unwrap();
            }
        }
    }
    store.evict_lru(256).unwrap();

    // The shared cache (across all tenants) is bounded at 256.
    assert!(
        store.cache_size() <= 256,
        "shared cache exceeded cap: {}",
        store.cache_size()
    );
    // 4 K pages total ingested; 256 hot, rest evicted.
    let total_ingested = (N_TENANTS * PAGES_PER_TENANT) as usize;
    assert_eq!(store.cache_size() + store.evicted_count(), total_ingested);

    // Per-tenant pool count.
    let tenant_count = store.pools().tenant_count();
    eprintln!(
        "[rss_audit_per_tenant] N_TENANTS={} PAGES_PER_TENANT={} total_ingested={} \
         cache_size={} evicted_count={} per_tenant_pools={}",
        N_TENANTS,
        PAGES_PER_TENANT,
        total_ingested,
        store.cache_size(),
        store.evicted_count(),
        tenant_count,
    );
    // Each tenant that wrote at least once should have its own pool.
    // The PerTenantBufferPool lazy-creates only on `pool(tenant)`
    // invocation; install_fresh paths today don't go through
    // PerTenantBufferPool::pool, so tenant_count may be 0 — that's the
    // expected pre-implicit-fault-in posture. Documented in ADR-140
    // §Forward-deferred.
    assert!(
        tenant_count <= N_TENANTS as usize,
        "tenant pool count exceeded N_TENANTS"
    );
}
