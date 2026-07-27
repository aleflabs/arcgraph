//! Census-derived owner-substrate disk budgets (M5-D3, amendment §5).
//!
//! `OWNER_INDEX_DISK_CAP_BYTES` / `OWNER_PAYLOAD_DISK_CAP_BYTES` stop being
//! fixed constants on the BULK path: the loader's pass-1 census (exact entry
//! counts + exact `Σ|external_id|`) is known before any substrate write, so
//! the caps are derived from it here and passed to
//! [`crate::OwnerForwardIndex::create`] / [`crate::OwnerPayloadStore::create`]
//! through [`crate::m4_migration::FreshV6Builder::create_with_budgets`]. The
//! fixed constants remain ONLY as the incremental-path defaults (churn-bounded
//! growth, the regime they were sized for — Director ruling D-5).
//!
//! Fail-closed is retained deliberately: the cap exists to catch runaway/leak
//! classes. The fix is deriving the number, not deleting the guard — reaching
//! `DiskBudgetExceeded` mid-build on well-formed input is a projection bug and
//! a gate FAILURE (INV-M5.25), never an accepted outcome.
//!
//! All arithmetic is u128-intermediate integer math (no floats): the formulas
//! must be exactly reproducible by the INV-M5.25 arithmetic gates.

/// Minimum derived cap. Derived budgets below this are rounded up so tiny
/// bulk loads keep slack for run-header/manifest fixed overheads that the
/// per-entry formulas do not model.
pub const OWNER_BUDGET_FLOOR_BYTES: u64 = 64 * 1024 * 1024;

/// Pass-1 census for ONE binding class (nodes or relationships).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BulkClassCensus {
    /// Exact entry count (rows) for this class.
    pub entries: u64,
    /// Exact `Σ|external_id|` bytes across the class.
    pub external_id_bytes: u64,
}

/// Derived disk caps for one binding class's owner substrate pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerSubstrateBudget {
    /// Cap for the class's forward candidate index directory.
    pub index_cap_bytes: u64,
    /// Cap for the class's immutable payload companion file.
    pub payload_cap_bytes: u64,
}

impl OwnerSubstrateBudget {
    /// Amendment §5 formulas, exactly:
    ///
    /// - **index**: `entries × 8192/509` (16.1 B/entry: 509-entry leaves of
    ///   8 KiB pages) `× 1.01` (internal pages) `× 2.2` (largest-merge
    ///   transient: old runs + intermediates + one replacement run)
    ///   `× 1.25` (safety), floored at [`OWNER_BUDGET_FLOOR_BYTES`].
    /// - **payload**: `(Σ|external_id| + entries × 16 B overflow envelope)`
    ///   `× 1.5` (transient) `× 1.25` (safety), floored likewise. Treating
    ///   every row as overflow is a deliberate upper bound — common ids stay
    ///   inline in the 256-byte owner row and never reach the companion.
    #[must_use]
    pub fn derive(census: BulkClassCensus) -> Self {
        // entries × 8192/509 × 101/100 × 22/10 × 125/100
        let index_need = u128::from(census.entries) * 8192 / 509;
        let index_cap = index_need * 101 * 22 * 125 / (100 * 10 * 100);
        // (Σ|external_id| + entries × 16) × 15/10 × 125/100
        let payload_need = u128::from(census.external_id_bytes) + u128::from(census.entries) * 16;
        let payload_cap = payload_need * 15 * 125 / (10 * 100);
        Self {
            index_cap_bytes: saturate(index_cap).max(OWNER_BUDGET_FLOOR_BYTES),
            payload_cap_bytes: saturate(payload_cap).max(OWNER_BUDGET_FLOOR_BYTES),
        }
    }

    /// Exact (un-inflated) space the formulas project the built substrate
    /// pair to need — the plan-time projection table's per-class row.
    #[must_use]
    pub fn projected_need_bytes(census: BulkClassCensus) -> u64 {
        let index_need = u128::from(census.entries) * 8192 / 509;
        let payload_need = u128::from(census.external_id_bytes) + u128::from(census.entries) * 16;
        saturate(index_need + payload_need)
    }
}

/// Derived budgets for both bulk binding classes. Non-binding classes
/// (intern/class-id/grant) keep the incremental defaults on every path —
/// the bulk loader writes exactly two rows into them per tenant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OwnerBulkBudgets {
    /// `OwnerRowClass::NodeBinding` caps.
    pub node_bindings: OwnerSubstrateBudget,
    /// `OwnerRowClass::RelBinding` caps.
    pub rel_bindings: OwnerSubstrateBudget,
}

impl OwnerBulkBudgets {
    /// Derive both classes from their pass-1 censuses.
    #[must_use]
    pub fn derive(nodes: BulkClassCensus, rels: BulkClassCensus) -> Self {
        Self {
            node_bindings: OwnerSubstrateBudget::derive(nodes),
            rel_bindings: OwnerSubstrateBudget::derive(rels),
        }
    }
}

fn saturate(value: u128) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::owner_index::OWNER_INDEX_DISK_CAP_BYTES;
    use crate::owner_payload::OWNER_PAYLOAD_DISK_CAP_BYTES;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn tiny_census_floors_at_minimum() {
        let budget = OwnerSubstrateBudget::derive(BulkClassCensus {
            entries: 10,
            external_id_bytes: 320,
        });
        assert_eq!(budget.index_cap_bytes, OWNER_BUDGET_FLOOR_BYTES);
        assert_eq!(budget.payload_cap_bytes, OWNER_BUDGET_FLOOR_BYTES);
    }

    /// Amendment §5 table, 100M+500M rung: the derived caps exceed the
    /// landed fixed constants by the ~7× / ~5× factors the vet computed —
    /// i.e. the fixed constants CANNOT govern the bulk path.
    #[test]
    fn rung_100m_derived_caps_exceed_fixed_constants() {
        let rels = OwnerSubstrateBudget::derive(BulkClassCensus {
            entries: 500_000_000,
            external_id_bytes: 500_000_000 * 32,
        });
        // 500M × 16.1 × 2.7775 ≈ 20.8 GiB > 3 GiB fixed.
        assert!(rels.index_cap_bytes > OWNER_INDEX_DISK_CAP_BYTES);
        assert!(rels.index_cap_bytes > 20 * GIB && rels.index_cap_bytes < 22 * GIB);
        // (16 GB + 8 GB) × 1.875 = 45 GB > 8 GiB fixed.
        assert!(rels.payload_cap_bytes > OWNER_PAYLOAD_DISK_CAP_BYTES);
    }

    /// The V-2 regression pin: at the 1B+5B census the LANDED constants are
    /// exceeded ~71× (index) — a plan built on the fixed caps MUST refuse.
    #[test]
    fn rung_1b_landed_constants_would_refuse_at_plan_time() {
        let census = BulkClassCensus {
            entries: 5_000_000_000,
            external_id_bytes: 5_000_000_000 * 32,
        };
        let need = OwnerSubstrateBudget::projected_need_bytes(census);
        assert!(
            need > OWNER_INDEX_DISK_CAP_BYTES + OWNER_PAYLOAD_DISK_CAP_BYTES,
            "1B census must not fit under the landed fixed constants"
        );
        let derived = OwnerSubstrateBudget::derive(census);
        assert!(derived.index_cap_bytes as u128 >= u128::from(census.entries) * 8192 / 509);
        assert!(
            u128::from(derived.payload_cap_bytes)
                >= u128::from(census.external_id_bytes) + u128::from(census.entries) * 16
        );
    }

    #[test]
    fn derived_caps_always_cover_projected_need() {
        for entries in [0u64, 1, 509, 1 << 20, 100_000_000, 5_000_000_000] {
            for id_bytes in [0u64, entries.saturating_mul(8), entries.saturating_mul(64)] {
                let census = BulkClassCensus {
                    entries,
                    external_id_bytes: id_bytes,
                };
                let budget = OwnerSubstrateBudget::derive(census);
                assert!(
                    budget
                        .index_cap_bytes
                        .saturating_add(budget.payload_cap_bytes)
                        >= OwnerSubstrateBudget::projected_need_bytes(census),
                    "caps must cover need at entries={entries} id_bytes={id_bytes}"
                );
            }
        }
    }
}
