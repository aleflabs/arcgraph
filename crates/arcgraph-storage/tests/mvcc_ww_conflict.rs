//! Proptest #4 of 5.
//!
//! *Write-write conflict detection*: a transaction that buffered a
//! write to key K must abort with [`ArcGraphError::MvccConflict`] if
//! another transaction committed a version of K after its snapshot.
//! Symmetrically, no conflict is signaled when the other write-sets
//! are disjoint from K.
//!
//! Gate: 5,000 cases in `--release`.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_ww_conflict --nocapture

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
    fn post_snapshot_commit_forces_conflict(
        conflict_key in 0u64..16,
        pad_keys in prop::collection::vec(1024u64..4096, 0..4),
        writers in 1usize..4,
    ) {
        let m = TxnManager::new();

        // A begins before anyone writes `conflict_key`.
        let mut a = m.begin(TenantId::DEFAULT);
        a.write(conflict_key, Bytes::from_static(b"from_a"));
        for k in &pad_keys {
            a.write(*k, Bytes::from_static(b"pad"));
        }

        // Someone else commits a write to the same key.
        for _ in 0..writers {
            let mut w = m.begin(TenantId::DEFAULT);
            w.write(conflict_key, Bytes::from_static(b"from_w"));
            // Only the first writer in each loop iteration commits
            // successfully (subsequent ones overlap with `a` AND with
            // the previous winner). This is fine — we only need one
            // successful post-snapshot commit for `a` to be forced.
            let _ = w.commit();
        }

        // A must now fail.
        let res = a.commit();
        prop_assert!(
            matches!(res, Err(ArcGraphError::MvccConflict { .. })),
            "expected MvccConflict, got {res:?}",
        );
    }

    #[test]
    fn disjoint_writes_never_conflict(
        my_keys in prop::collection::vec(0u64..128, 1..8),
        other_keys in prop::collection::vec(128u64..256, 0..8),
    ) {
        let m = TxnManager::new();

        let mut a = m.begin(TenantId::DEFAULT);
        for k in &my_keys {
            a.write(*k, Bytes::from_static(b"a"));
        }

        for k in &other_keys {
            let mut w = m.begin(TenantId::DEFAULT);
            w.write(*k, Bytes::from_static(b"w"));
            w.commit().unwrap();
        }

        prop_assert!(a.commit().is_ok(), "disjoint write-sets must not conflict");
    }

    #[test]
    fn no_conflict_when_other_commits_before_my_begin(
        key in 0u64..64,
        v_early in any::<u8>(),
        v_late in any::<u8>(),
    ) {
        let m = TxnManager::new();

        // Early commit that predates our snapshot.
        let mut pre = m.begin(TenantId::DEFAULT);
        pre.write(key, Bytes::copy_from_slice(&[v_early]));
        pre.commit().unwrap();

        // Now we begin — snapshot already includes the early commit.
        let mut a = m.begin(TenantId::DEFAULT);
        a.write(key, Bytes::copy_from_slice(&[v_late]));
        prop_assert!(a.commit().is_ok(), "pre-snapshot writes must not cause conflict");
    }
}
