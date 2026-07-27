//! Proptest for the MVCC visibility filter inside `TelScan`.
//!
//! The concurrent-append gate (`tel_scan_concurrent.rs`) runs with
//! `snapshot_lsn = u64::MAX - 1`, so every entry is visible — the
//! filter predicate `created_lsn <= snapshot < expired_lsn` is never
//! actually exercised there. This test closes that gap: it appends a
//! mixed bag of alive / tombstoned / future entries and checks that
//! `scan(Lsn)` yields exactly the subset that `is_visible_at` admits,
//! in original insertion order.
//!
//! Pure `TelBlock` test — no MVCC pipeline, no threads.
//!
//! Gate: **5 000 release cases, zero failures.** Debug builds cap at
//! 256 to keep default `cargo test` fast. Override with
//! `PROPTEST_CASES=<n>`.

use arcgraph_core::{LabelId, Lsn, NodeId, TelEntry, TenantId};
use arcgraph_storage::tel::{MAX_BLOCK_BYTES, TelBlock};
use proptest::prelude::*;
use proptest::test_runner::Config as ProptestConfig;

/// One entry specification used to build the block.
#[derive(Debug, Clone, Copy)]
struct EntrySpec {
    seed: u64,
    created_lsn: u64,
    expired_lsn: u64,
}

impl EntrySpec {
    fn to_entry(self) -> TelEntry {
        TelEntry {
            dst_id: self.seed,
            rel_id: self.seed,
            created_lsn: self.created_lsn,
            expired_lsn: self.expired_lsn,
        }
    }

    fn visible_at(self, snapshot: u64) -> bool {
        self.created_lsn <= snapshot && snapshot < self.expired_lsn
    }
}

/// Strategy: `created_lsn in 1..=1000`, `expired_lsn` either
/// `u64::MAX` (alive) or in `created_lsn..=2000` (tombstoned).
fn entry_spec() -> impl Strategy<Value = EntrySpec> {
    (any::<u64>(), 1u64..=1_000).prop_flat_map(|(seed, created)| {
        prop_oneof![
            Just((seed, created, u64::MAX)),
            (created..=2_000u64).prop_map(move |expired| (seed, created, expired)),
        ]
        .prop_map(|(seed, created_lsn, expired_lsn)| EntrySpec {
            seed,
            created_lsn,
            expired_lsn,
        })
    })
}

fn run_case(specs: &[EntrySpec], snapshots: &[u64]) -> Result<(), String> {
    let block = TelBlock::new(
        NodeId::new(1),
        LabelId::new(1),
        MAX_BLOCK_BYTES,
        TenantId::DEFAULT,
    )
    .map_err(|e| format!("block alloc failed: {e:?}"))?;
    for spec in specs {
        block
            .append(spec.to_entry())
            .map_err(|e| format!("append failed: {e:?}"))?;
    }

    for &snapshot in snapshots {
        let expected: Vec<TelEntry> = specs
            .iter()
            .copied()
            .filter(|s| s.visible_at(snapshot))
            .map(EntrySpec::to_entry)
            .collect();
        let got: Vec<TelEntry> = block.scan(Lsn::new(snapshot)).collect();
        if got != expected {
            return Err(format!(
                "scan@{snapshot} mismatch: got {got:?}, expected {expected:?}"
            ));
        }
    }
    Ok(())
}

fn config() -> ProptestConfig {
    let cases: u32 = if cfg!(debug_assertions) { 256 } else { 5_000 };
    ProptestConfig {
        cases,
        // Filter mismatches are easy to root-cause from the raw seed;
        // proptest shrinking on vec-of-struct strategies is slow and
        // not especially illuminating for this predicate.
        max_shrink_iters: 0,
        ..ProptestConfig::default()
    }
}

proptest! {
    #![proptest_config(config())]

    /// Scan output == `entries.filter(visible_at(snapshot))` under
    /// arbitrary mixes of alive / tombstoned / future entries.
    #[test]
    fn tel_scan_mvcc_filter_matches_oracle(
        specs in prop::collection::vec(entry_spec(), 1..=200),
        snapshots in prop::collection::vec(0u64..=2_000, 1..=10),
    ) {
        run_case(&specs, &snapshots)
            .map_err(proptest::test_runner::TestCaseError::fail)?;
    }
}

/// Regression-proofing sanity case for when the proptest is disabled
/// (e.g. cases = 0 in CI fast-path).
#[test]
fn tel_scan_mvcc_filter_sanity() {
    let specs = [
        EntrySpec {
            seed: 10,
            created_lsn: 5,
            expired_lsn: u64::MAX,
        }, // alive
        EntrySpec {
            seed: 11,
            created_lsn: 1,
            expired_lsn: 3,
        }, // tombstoned at 3
        EntrySpec {
            seed: 12,
            created_lsn: 100,
            expired_lsn: u64::MAX,
        }, // future at snapshot=50
    ];
    let snapshots = [0u64, 2, 5, 50, 200];
    run_case(&specs, &snapshots).expect("sanity case must pass");
}
