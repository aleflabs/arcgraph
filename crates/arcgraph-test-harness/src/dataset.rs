//! [`Dataset`] trait + supporting types per ADR-042.
//!
//! A [`Dataset`] names an input fixture that a
//! [`Workload`](crate::Workload) consumes. The retained
//! [`workloads::ldbc_snb`](crate::workloads::ldbc_snb) module
//! synthesizes an SBM fixture for integration tests.

use crate::HarnessResult;

/// Coarse description of a dataset's footprint.
///
/// Used by the harness to decide whether a loader fits a CI budget
/// or should be opted-in to via `ARCGRAPH_TEST_HARNESS_DATA_DIR`
/// (per ADR-042 OQ-42-5). Numbers are upper-bound estimates from
/// the per-domain catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DatasetScale {
    /// Approximate node count.
    pub nodes: u64,
    /// Approximate edge count (undirected for symmetric domains;
    /// directed otherwise — per the dataset's native semantics).
    pub edges: u64,
}

/// An input dataset for a database workload.
///
/// Loaders are stateless trait impls: callers pass the impl into
/// [`Workload::run`](crate::Workload::run) which materialises a
/// [`DatasetHandle`] via [`Dataset::load`].
pub trait Dataset {
    /// Stable identifier (e.g. `"ldbc-snb-sf1"`).
    fn id(&self) -> &'static str;
    /// Domain bucket (PRD §2.2 column tag — `"social-network"`,
    /// `"biomedical"`, `"security"`, `"fraud"`, `"knowledge-graph"`).
    fn domain(&self) -> &'static str;
    /// Upstream source URL, recorded for provenance audit (per
    /// ADR-042 §"Per-domain dataset inventory").
    fn upstream_source(&self) -> &'static str;
    /// SPDX-style license tag for the dataset's content (NOT its
    /// schema — schema licensing is tracked separately).
    fn license(&self) -> &'static str;
    /// Coarse scale estimate (informational; not load-bearing on
    /// the trait contract).
    fn approximate_scale(&self) -> DatasetScale;
    /// Materialise the dataset. Most loaders return
    /// `HarnessError::NotImplementedAtV1` pre-M5-08
    /// (`graph.ingest()`); the LDBC SNB loader returns a synthetic
    /// SBM fixture wrapped in a [`DatasetHandle`].
    fn load(&self) -> HarnessResult<DatasetHandle>;
}

/// Materialised dataset handle.
///
/// Wave-11β shape is intentionally narrow: it carries the
/// originating dataset's `id` + an enum payload that today only has
/// the [`DatasetHandle::SbmGraph`] variant (LDBC SNB Tier-1 fixture).
/// Post-M5-08, this enum gains a `RealEngine` variant carrying a
/// `MultiTenantRouter` handle from `arcgraph-storage` so workloads
/// can dispatch real queries.
#[derive(Debug)]
#[non_exhaustive]
pub enum DatasetHandle {
    /// Synthetic SBM fixture used by the LDBC SNB Tier-1 loader.
    /// Pinned here at Wave 11β so the integration test can
    /// demonstrate the trait round-trip without M5-08.
    SbmGraph(arcgraph_community::Graph),
}

impl DatasetHandle {
    /// Best-effort approximate node count, for diagnostics.
    pub fn node_count(&self) -> u64 {
        match self {
            DatasetHandle::SbmGraph(graph) => u64::from(graph.n()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HarnessError;

    /// Minimal trait-shape exercise: a custom Dataset can be defined
    /// outside the crate and `load()` returns the documented
    /// `NotImplementedAtV1` shape pre-M5-08.
    struct ExternalLoaderProbe;

    impl Dataset for ExternalLoaderProbe {
        fn id(&self) -> &'static str {
            "external-probe"
        }
        fn domain(&self) -> &'static str {
            "scaffold-probe"
        }
        fn upstream_source(&self) -> &'static str {
            "(probe — no upstream)"
        }
        fn license(&self) -> &'static str {
            "Apache-2.0"
        }
        fn approximate_scale(&self) -> DatasetScale {
            DatasetScale { nodes: 0, edges: 0 }
        }
        fn load(&self) -> HarnessResult<DatasetHandle> {
            Err(HarnessError::NotImplementedAtV1 {
                feature: "M5-08",
                reason: "external probe does not ship a loader at v1.0-alpha".into(),
            })
        }
    }

    #[test]
    fn dataset_trait_is_object_safe() {
        // If `Dataset` is not object-safe a future caller's
        // `&dyn Dataset` argument breaks at compile time. The bound
        // is checked at impl-site so this is a structural pin.
        let probe: &dyn Dataset = &ExternalLoaderProbe;
        assert_eq!(probe.id(), "external-probe");
        assert_eq!(probe.domain(), "scaffold-probe");
        assert_eq!(probe.license(), "Apache-2.0");
    }

    #[test]
    fn dataset_handle_node_count_for_sbm_variant() {
        let edges: Vec<(u32, u32, f32)> = (0..5).map(|u| (u, u + 1, 1.0)).collect();
        let graph = arcgraph_community::Graph::from_edges_undirected(8, &edges);
        let handle = DatasetHandle::SbmGraph(graph);
        // node_count surfaces the underlying graph's `n()` so
        // workloads can sanity-check fixture scale before
        // dispatching a query.
        assert_eq!(handle.node_count(), 8);
    }

    #[test]
    fn placeholder_load_surfaces_milestone_tag() {
        // The HarnessError::NotImplementedAtV1 contract requires the
        // `feature` tag to identify the gating milestone so log
        // readers can route stub-hits. The probe above tags M5-08;
        // pin that here so future renames (e.g. M5-08 → M5-08a) do
        // not silently flip the tag without a compile error.
        match ExternalLoaderProbe.load() {
            Err(HarnessError::NotImplementedAtV1 { feature, .. }) => {
                assert_eq!(feature, "M5-08");
            }
            other => panic!("expected NotImplementedAtV1, got {other:?}"),
        }
    }
}
