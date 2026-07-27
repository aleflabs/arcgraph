//! Proptest #3 of 5.
//!
//! *Read-your-writes* (RYW): a transaction reads its own buffered
//! writes before falling through to the committed version store. A
//! buffered tombstone shadows a previously committed value.
//!
//! Gate: 5,000 cases in `--release`.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_ryw --nocapture

use std::collections::HashMap;

use arcgraph_core::TenantId;
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;
use proptest::prelude::*;

#[derive(Debug, Clone)]
enum Op {
    Write(u64, u8),
    Delete(u64),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        3 => (0u64..8, any::<u8>()).prop_map(|(k, v)| Op::Write(k, v)),
        1 => (0u64..8).prop_map(Op::Delete),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 5_000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn ryw_matches_deterministic_shadow_map(
        preload in prop::collection::vec((0u64..8, any::<u8>()), 0..8),
        ops in prop::collection::vec(op_strategy(), 1..32),
    ) {
        let m = TxnManager::new();
        // Seed some committed state so RYW has something to shadow.
        for (k, v) in &preload {
            let mut t = m.begin(TenantId::DEFAULT);
            t.write(*k, Bytes::copy_from_slice(&[*v]));
            t.commit().unwrap();
        }

        let mut t = m.begin(TenantId::DEFAULT);

        // `shadow` is the ground-truth RYW view: starts from committed
        // state, then applies each op locally.
        let mut shadow: HashMap<u64, Option<u8>> = HashMap::new();
        for (k, v) in &preload {
            shadow.insert(*k, Some(*v));
        }

        for op in &ops {
            match *op {
                Op::Write(k, v) => {
                    t.write(k, Bytes::copy_from_slice(&[v]));
                    shadow.insert(k, Some(v));
                }
                Op::Delete(k) => {
                    t.delete(k);
                    shadow.insert(k, None);
                }
            }
            // After every op, read back every key in the shadow and
            // compare.
            for (k, want) in &shadow {
                let got = t.read(*k).map(|b| b[0]);
                let expected = want.as_ref().copied();
                prop_assert_eq!(got, expected, "RYW mismatch at key {}", k);
            }
            // Keys never touched in shadow must read as the preloaded
            // committed value (or None).
            for k in 0u64..8 {
                if !shadow.contains_key(&k) {
                    let got = t.read(k);
                    prop_assert!(got.is_none(), "unseeded key {k} leaked a value");
                }
            }
        }

        // On commit, every shadow entry becomes observable at the new
        // snapshot.
        let commit_lsn = t.commit().unwrap();
        for (k, want) in &shadow {
            let got = m.read_at(TenantId::DEFAULT, *k, commit_lsn).map(|b| b[0]);
            prop_assert_eq!(got, want.as_ref().copied());
        }
    }
}
