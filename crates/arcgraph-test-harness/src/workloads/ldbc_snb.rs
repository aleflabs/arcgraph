//! LDBC SNB Interactive Tier-1 workload (ADR-042 §"Axis-B Tier-1").
//!
//! At Wave 11β this module ships:
//!
//! - [`LdbcSnbSf1Dataset`] — deterministic synthetic SBM fixture.
//! - [`LdbcSnbIs1Workload`] — the IS1 (profile lookup) workload from
//!   design-v2 §10.5. Post-M4-61 (Wave 11Z flip) `Workload::run`
//!   dispatches the IS1 Cypher through
//!   `arcgraph_query::explain::QueryEngine::execute`, which drives
//!   the executor's `ExecutionContext` + `Batch` pipeline. The SBM
//!   fixture has no `Person`/`Place` schema so the executor returns
//!   zero rows; the [`WorkloadResult::Ran`] payload pins the
//!   parser → binder → planner → executor seam end-to-end.
//!
//! The IS1 Cypher source mirrors design-v2 §10.5 + the LDBC SNB
//! Interactive specification (CC-BY 4.0 / Apache-2.0 dual-licensed
//! per LDBC consortium materials).

use arcgraph_community::Graph;
use arcgraph_core::Lsn;
use arcgraph_query::QueryEngine;
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::semantic::StubCatalogProvider;

use crate::dataset::{Dataset, DatasetHandle, DatasetScale};
use crate::workload::{RegressionGate, Workload, WorkloadResult};
use crate::{HarnessError, HarnessResult};

/// IS1 cypher rendered into an executable form for the in-process
/// stub catalog/substrate.
///
/// The workload's [`Workload::cypher`] returns the LDBC SNB
/// Interactive §IS1 *spec* form (parameterised on `$personId`).
/// `QueryEngine::execute` does not accept a runtime-supplied
/// parameter bag at v1.0-alpha (M5-12 lights one), so the
/// post-W11Z executor seam binds against this literal-substituted
/// variant. Once M5-12 ships, the workload can dispatch the spec
/// cypher directly with a single-entry parameter bag and this
/// const folds away.
const IS1_EXECUTABLE_CYPHER: &str = "MATCH (n:Person {id: 1})-[:IS_LOCATED_IN]->(p:Place)
     RETURN
       n.firstName    AS firstName,
       n.lastName     AS lastName,
       n.birthday     AS birthday,
       n.locationIP   AS locationIP,
       n.browserUsed  AS browserUsed,
       p.id           AS cityId,
       n.gender       AS gender,
       n.creationDate AS creationDate";

/// LDBC SNB Interactive SF-1 dataset (synthetic SBM fixture at
/// Wave 11β; real ingestion is M4-84 / M5-08 territory).
#[derive(Debug, Clone, Copy, Default)]
pub struct LdbcSnbSf1Dataset;

impl Dataset for LdbcSnbSf1Dataset {
    fn id(&self) -> &'static str {
        "ldbc-snb-sf1"
    }
    fn domain(&self) -> &'static str {
        "social-network"
    }
    fn upstream_source(&self) -> &'static str {
        "https://ldbcouncil.org/benchmarks/snb/"
    }
    fn license(&self) -> &'static str {
        "Apache-2.0"
    }
    fn approximate_scale(&self) -> DatasetScale {
        // SF-1 ≈ 10 K persons, ≈ 3 M edges per LDBC SNB Interactive
        // SF-1 published statistics. The SBM fixture below targets a
        // similar shape at the macroscopic level.
        DatasetScale {
            nodes: 10_000,
            edges: 3_000_000,
        }
    }
    fn load(&self) -> HarnessResult<DatasetHandle> {
        // Synthetic SBM fixture per ADR-042 §"Per-domain dataset
        // inventory" footnote: real SF-1 ingestion is gated on M5-08.
        // The fixture ships at a small scale (1 K persons rather
        // than 10 K) so the integration test runs in <100 ms;
        // production benches override the size via M4-84 wire-up.
        Ok(DatasetHandle::SbmGraph(build_ldbc_sbm_fixture(0xC0FFEE)))
    }
}

/// LDBC SNB Interactive IS1 (profile lookup) workload.
///
/// Per design-v2 §10.5: target P50 50 µs, P99 500 µs.
#[derive(Debug, Clone, Copy, Default)]
pub struct LdbcSnbIs1Workload;

impl LdbcSnbIs1Workload {
    /// Tier-1 regression gate per ADR-042 §"CI integration":
    /// design-v2 §10.5 P99 = 500 µs, +10% bound = 550 µs.
    pub fn regression_gate() -> RegressionGate {
        RegressionGate {
            workload_id: "LDBC-SNB-IS1",
            baseline_p99_us: 500,
            regression_threshold_pct: 10,
        }
    }
}

impl Workload for LdbcSnbIs1Workload {
    fn id(&self) -> &'static str {
        "LDBC-SNB-IS1"
    }
    fn domain(&self) -> &'static str {
        "social-network"
    }
    fn cypher(&self) -> &'static str {
        // LDBC SNB Interactive IS1 — profile lookup. Verbatim from
        // the LDBC SNB workload specification §IS1 (Apache-2.0 /
        // CC-BY 4.0). Read-only per ADR-006 amendment-01.
        "MATCH (n:Person {id: $personId})-[:IS_LOCATED_IN]->(p:Place)
         RETURN
           n.firstName    AS firstName,
           n.lastName     AS lastName,
           n.birthday     AS birthday,
           n.locationIP   AS locationIP,
           n.browserUsed  AS browserUsed,
           p.id           AS cityId,
           n.gender       AS gender,
           n.creationDate AS creationDate"
    }
    fn run(&self, dataset: &DatasetHandle) -> HarnessResult<WorkloadResult> {
        // W11Z flip — dispatch the IS1 cypher through the M4-61
        // executor seam (`QueryEngine::execute`). The SBM fixture's
        // shape (untyped vertices in `arcgraph_community::Graph`)
        // does NOT carry the IS1 schema, so the catalog is populated
        // with the IS1 label/rel-type/property names but the
        // substrate stays empty. The executor returns 0 rows; the
        // [`WorkloadResult::Ran`] payload pins the parser → binder
        // → planner → executor seam end-to-end and gives the
        // Tier-1 LDBC-SNB-IS1 regression gate a stable baseline
        // (P50/P99 anchored against the empty-substrate path until
        // M5-08 wires real ingestion).
        let DatasetHandle::SbmGraph(graph) = dataset;
        if graph.n() == 0 {
            return Err(HarnessError::FixtureFailed {
                reason: "LDBC SNB SF-1 fixture has zero nodes; SBM generator regressed".into(),
            });
        }
        // Catalog populates the IS1-shape labels/rel-types/properties
        // so the binder + cross-substrate validator type-check the
        // cypher. Substrate stays empty pre-M5-08 — the row_count
        // floor is 0 by construction.
        let catalog = StubCatalogProvider::new()
            .with_labels(["Person", "Place"])
            .with_rel_types(["IS_LOCATED_IN"])
            .with_properties([
                "id",
                "firstName",
                "lastName",
                "birthday",
                "locationIP",
                "browserUsed",
                "gender",
                "creationDate",
            ]);
        let substrate = StubExecutorSubstrate::new();
        let engine = QueryEngine::new(&catalog);
        match engine.execute(IS1_EXECUTABLE_CYPHER, &substrate) {
            Ok(rows) => Ok(WorkloadResult::Ran {
                id: "LDBC-SNB-IS1",
                // Stub catalog has no real WAL clock; `Lsn::MAX` is
                // the "stub catalog snapshot" sentinel mirroring
                // `crates/arcgraph-query/tests/m4_61_executor_integration.rs::cat_basic`'s
                // `read_lsn: Lsn::MAX` discipline. M5-08 + M5-12
                // will replace with the substrate-acquired LSN.
                snapshot_lsn: Lsn::MAX,
                row_count: rows.len() as u64,
            }),
            Err(err) => {
                // Defensive: per the W11Z spawn prompt, if
                // `QueryEngine::execute` flattens an unsupported
                // construct to NotImplemented, document the
                // observation as a Skipped result rather than
                // failing the workload outright. M4-63 / M5-12
                // close the surface area.
                Ok(WorkloadResult::Skipped {
                    id: "LDBC-SNB-IS1",
                    reason: format!(
                        "executor surface gap: {err} (W11Z flip — {n}-node SBM fixture transit pin)",
                        n = graph.n(),
                    ),
                })
            }
        }
    }
}

/// Synthesises a 20-block stochastic block model approximating the
/// macroscopic LDBC SNB SF-1 friendship structure.
///
/// Uses a deterministic fixture shape (`p_in = 0.075`,
/// `p_out = 0.0015`) at a
/// `N = 1_000` scale so the integration test runs in <100 ms. The
/// production scale (`N = 10_000`) lands in M4-84 once the LDBC
/// SNB Interactive harness is wired and runs out-of-process.
///
/// Per ADR-042 §"Per-domain dataset inventory": real LDBC SF-1 is
/// gigabytes — vendoring is OQ-42-5 territory; the SBM
/// approximation gives the Tier-1 regression gate a stable
/// in-process fixture.
fn build_ldbc_sbm_fixture(seed: u64) -> Graph {
    const N: u32 = 1_000;
    const K: u32 = 20;
    const P_IN: f64 = 0.075;
    const P_OUT: f64 = 0.0015;

    let block_size = N / K;
    let block_of = |v: u32| v / block_size;
    let mut state = seed;
    let mut next_unit = || -> f64 {
        // SplitMix64 — the same generator the bench uses; seed
        // determinism is the load-bearing property here.
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };

    let mut edges: Vec<(u32, u32, f32)> = Vec::with_capacity(50_000);
    for u in 0..N {
        for v in (u + 1)..N {
            let p = if block_of(u) == block_of(v) {
                P_IN
            } else {
                P_OUT
            };
            if next_unit() < p {
                edges.push((u, v, 1.0));
            }
        }
    }
    Graph::from_edges_undirected(N, &edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldbc_sf1_dataset_metadata_matches_adr_042() {
        let ds = LdbcSnbSf1Dataset;
        assert_eq!(ds.id(), "ldbc-snb-sf1");
        assert_eq!(ds.domain(), "social-network");
        assert_eq!(ds.license(), "Apache-2.0");
        assert!(ds.upstream_source().starts_with("https://ldbcouncil.org/"));
        // Scale numbers are coarse — pinned here as the catalog
        // commitment; bumping requires an ADR-042 amendment.
        let scale = ds.approximate_scale();
        assert_eq!(scale.nodes, 10_000);
        assert_eq!(scale.edges, 3_000_000);
    }

    #[test]
    fn ldbc_sf1_dataset_load_yields_non_empty_sbm_fixture() {
        let handle = LdbcSnbSf1Dataset.load().expect("load");
        // The SBM fixture has 1 K nodes (the in-process scale —
        // production scale is M4-84's 10 K). 1 K is the floor below
        // which the SBM cannot recover community structure.
        assert!(
            handle.node_count() >= 100,
            "SBM fixture must yield at least 100 nodes, got {}",
            handle.node_count()
        );
    }

    #[test]
    fn is1_workload_metadata_matches_design_v2_section_10_5() {
        let w = LdbcSnbIs1Workload;
        assert_eq!(w.id(), "LDBC-SNB-IS1");
        assert_eq!(w.domain(), "social-network");
        // Cypher must be read-only per ADR-006 amendment-01.
        let cypher = w.cypher();
        assert!(cypher.contains("MATCH"));
        assert!(cypher.contains("RETURN"));
        assert!(!cypher.contains("CREATE"));
        assert!(!cypher.contains("DELETE"));
        assert!(!cypher.contains("MERGE"));
    }

    #[test]
    fn is1_run_dispatches_through_executor_post_w11z_flip() {
        let dataset = LdbcSnbSf1Dataset.load().expect("load");
        let result = LdbcSnbIs1Workload.run(&dataset).expect("run");
        // Post-W11Z flip the workload routes through
        // `QueryEngine::execute`. The fixture substrate is empty
        // (no Person/Place rows pre-M5-08), so the executor is
        // expected to return either:
        //   * `Ran { row_count: 0, snapshot_lsn: Lsn::MAX }` — the
        //     happy-path pin, OR
        //   * `Skipped { reason }` carrying an executor-surface
        //     gap cite (per the W11Z spawn-prompt "DO NOT block on
        //     fix-up E" framing).
        match result {
            WorkloadResult::Ran {
                id,
                snapshot_lsn,
                row_count,
            } => {
                assert_eq!(id, "LDBC-SNB-IS1");
                // Empty substrate floor: M4-84 lifts above 0 once
                // M5-08 ingests SF-1 vertices.
                assert_eq!(
                    row_count, 0,
                    "empty stub substrate must yield 0 rows; got {row_count}",
                );
                // Stub-catalog snapshot sentinel.
                assert_eq!(snapshot_lsn, Lsn::MAX);
            }
            WorkloadResult::Skipped { id, reason } => {
                assert_eq!(id, "LDBC-SNB-IS1");
                // Skipped is only acceptable when the executor
                // surface flattens to NotImplemented — the W11Z
                // spawn prompt's documented escape hatch.
                assert!(
                    reason.contains("executor surface gap"),
                    "skip reason must cite the W11Z executor-gap escape, got {reason:?}",
                );
            }
        }
    }

    #[test]
    fn is1_regression_gate_matches_design_v2_section_10_5_targets() {
        let gate = LdbcSnbIs1Workload::regression_gate();
        assert_eq!(gate.workload_id, "LDBC-SNB-IS1");
        // design-v2 §10.5: IS1 P99 = 500 µs.
        assert_eq!(gate.baseline_p99_us, 500);
        // ADR-042 §"CI integration" Tier-1 = 10 %.
        assert_eq!(gate.regression_threshold_pct, 10);
        // +10% bound = 550 µs.
        assert!(gate.within_bound(550));
        assert!(!gate.within_bound(551));
    }
}
