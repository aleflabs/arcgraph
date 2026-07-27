//! Proptest #2 of 5.
//!
//! *No lost updates*: of N concurrent writers that buffer a write to
//! the same key (each `begin`-ing before any `commit` lands), exactly
//! one commits successfully. All others return
//! [`ArcGraphError::MvccConflict`]. No silent last-writer-wins.
//!
//! The prompt phrasing is "two writers"; we test 2 and also the
//! N-writer generalisation.
//!
//! Gate: 5,000 cases in `--release`.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_no_lost_updates --nocapture

use arcgraph_core::{ArcGraphError, TenantId};
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 5_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn two_writers_exactly_one_commits(
        key in 0u64..1024,
        v1 in any::<u8>(),
        v2 in any::<u8>(),
    ) {
        let m = TxnManager::new();
        let mut a = m.begin(TenantId::DEFAULT);
        let mut b = m.begin(TenantId::DEFAULT);
        a.write(key, Bytes::copy_from_slice(&[v1]));
        b.write(key, Bytes::copy_from_slice(&[v2]));

        // First-committer-wins: `a` goes first and succeeds.
        let a_commit = a.commit();
        let b_commit = b.commit();
        prop_assert!(a_commit.is_ok(), "first writer must commit");
        prop_assert!(
            matches!(b_commit, Err(ArcGraphError::MvccConflict { .. })),
            "second writer must get MvccConflict, got {b_commit:?}",
        );

        // Committed value is exactly the winner's value.
        let lsn = m.current_lsn();
        prop_assert_eq!(m.read_at(TenantId::DEFAULT, key, lsn).map(|b| b[0]), Some(v1));
    }

    #[test]
    fn n_concurrent_writers_exactly_one_commits(
        key in 0u64..1024,
        values in prop::collection::vec(any::<u8>(), 2..8),
    ) {
        let m = TxnManager::new();
        let mut txns: Vec<_> = values
            .iter()
            .map(|v| {
                let mut t = m.begin(TenantId::DEFAULT);
                t.write(key, Bytes::copy_from_slice(&[*v]));
                t
            })
            .collect();

        let mut successes = 0usize;
        let mut failures = 0usize;
        // Drain in FIFO order — the serial commit order decides who
        // wins. No matter the order, exactly one commits.
        while let Some(t) = txns.pop() {
            match t.commit() {
                Ok(_) => successes += 1,
                Err(ArcGraphError::MvccConflict { .. }) => failures += 1,
                Err(e) => prop_assert!(false, "unexpected error {:?}", e),
            }
        }

        prop_assert_eq!(successes, 1, "exactly one writer must commit");
        prop_assert_eq!(failures, values.len() - 1);
    }

    #[test]
    fn disjoint_concurrent_writers_all_commit(
        keys in prop::collection::vec(0u64..2048, 2..8)
            .prop_filter("distinct", |v| {
                let mut seen = std::collections::HashSet::new();
                v.iter().all(|k| seen.insert(*k))
            }),
    ) {
        let m = TxnManager::new();
        let mut txns: Vec<_> = keys
            .iter()
            .map(|k| {
                let mut t = m.begin(TenantId::DEFAULT);
                t.write(*k, Bytes::from_static(b"v"));
                t
            })
            .collect();
        while let Some(t) = txns.pop() {
            prop_assert!(t.commit().is_ok(), "disjoint write-sets must not conflict");
        }
    }
}
