//! Proptest #1 of 5 (testing-strategy §2.2 MVCC row, mvcc-lead prompt).
//!
//! *Snapshot isolation*: a reader transaction sees exactly the
//! committed prefix at its snapshot LSN. Writes installed after the
//! reader's snapshot are invisible to it, regardless of how many
//! commits race in during its lifetime.
//!
//! Gate: 5,000 cases in `--release`.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_snapshot_isolation --nocapture

use std::collections::HashMap;

use arcgraph_core::TenantId;
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;
use proptest::prelude::*;

const KEY_SPACE: u64 = 64;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 5_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn reader_sees_only_pre_snapshot_prefix(
        initial in prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 0..20),
        later in prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 0..20),
    ) {
        let m = TxnManager::new();

        // Build the initial committed state. `expected` models the
        // snapshot the reader will capture.
        let mut expected: HashMap<u64, u8> = HashMap::new();
        for (k, v) in &initial {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
            expected.insert(*k, *v);
        }

        // Begin the reader at the current snapshot.
        let reader = m.begin(TenantId::DEFAULT);

        // Race an arbitrary number of commits past the reader's snapshot.
        for (k, v) in &later {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
        }

        // Invariant: reader's view matches `expected` on the full key
        // space, ignoring anything installed after its snapshot.
        for k in 0..KEY_SPACE {
            let got = reader.read(k).map(|b| b[0]);
            let want = expected.get(&k).copied();
            prop_assert_eq!(got, want, "snapshot violated at key {}", k);
        }
    }

    #[test]
    fn multiple_readers_each_see_their_own_snapshot(
        batches in prop::collection::vec(
            prop::collection::vec((0u64..KEY_SPACE, any::<u8>()), 1..5),
            2..8
        ),
    ) {
        let m = TxnManager::new();
        let mut state: HashMap<u64, u8> = HashMap::new();
        let mut readers = Vec::new();
        let mut expectations = Vec::new();

        for batch in &batches {
            for (k, v) in batch {
                let mut t = m.begin(TenantId::DEFAULT);
                t.write(*k, Bytes::copy_from_slice(&[*v]));
                t.commit().unwrap();
                state.insert(*k, *v);
            }
            // Snapshot current state for the reader we begin here.
            expectations.push(state.clone());
            readers.push(m.begin(TenantId::DEFAULT));
        }

        // After more writes are layered in, every reader is still
        // pinned to the state that existed at its own begin().
        for (k, v) in [(0u64, 0u8), (KEY_SPACE - 1, 255)] {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(k, Bytes::copy_from_slice(&[v]));
            t.commit().unwrap();
        }

        for (reader, want) in readers.iter().zip(expectations.iter()) {
            for k in 0..KEY_SPACE {
                let got = reader.read(k).map(|b| b[0]);
                let exp = want.get(&k).copied();
                prop_assert_eq!(got, exp, "reader snapshot drifted at key {}", k);
            }
        }
    }
}
