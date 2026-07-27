//! Thread-spawned commit-race tests (M2.b correctness gate companion).
//!
//! The five proptest files in this directory exercise MVCC invariants
//! single-threaded and rely on the `commit_gate: parking_lot::Mutex`
//! inside `TxnManager` being sound. These two tests close that gap by
//! running real OS threads through the validate-then-install window.
//!
//!   cargo test -p arcgraph-storage --release \
//!       -- mvcc_commit_race --nocapture

use arcgraph_core::{ArcGraphError, TenantId};
use arcgraph_storage::transaction::TxnManager;
use bytes::Bytes;

#[test]
fn concurrent_same_key_writers_only_one_commits() {
    use std::sync::{Arc, Barrier};
    for trial in 0..200u32 {
        let m = Arc::new(TxnManager::new());
        let n_threads = 2 + (trial % 7); // 2..=8
        // Barrier forces every thread to acquire its snapshot before
        // any thread enters commit. Without it, short-lived txns may
        // serialize (begin → commit → begin → commit) and all succeed
        // legitimately — which would test scheduling, not commit_gate.
        let barrier = Arc::new(Barrier::new(n_threads as usize));
        let handles: Vec<_> = (0..n_threads)
            .map(|tid| {
                let m = Arc::clone(&m);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let mut t = m.begin(TenantId::DEFAULT);
                    barrier.wait();
                    t.write(42, Bytes::copy_from_slice(&[tid as u8]));
                    t.commit()
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let commits: Vec<_> = results.iter().filter(|r| r.is_ok()).collect();
        let conflicts: Vec<_> = results
            .iter()
            .filter(|r| matches!(r, Err(ArcGraphError::MvccConflict { .. })))
            .collect();
        assert_eq!(
            commits.len(),
            1,
            "trial {trial}: expected 1 commit, got {}",
            commits.len()
        );
        assert_eq!(conflicts.len() as u32, n_threads - 1);
        let reader = m.begin(TenantId::DEFAULT);
        assert!(reader.read(42).is_some());
    }
}

#[test]
fn concurrent_disjoint_writers_all_commit() {
    use std::sync::Arc;
    for _ in 0..100 {
        let m = Arc::new(TxnManager::new());
        let handles: Vec<_> = (0..8u64)
            .map(|k| {
                let m = Arc::clone(&m);
                std::thread::spawn(move || {
                    let mut t = m.begin(TenantId::DEFAULT);
                    t.write(k, Bytes::from_static(b"v"));
                    t.commit()
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap().expect("disjoint must commit");
        }
    }
}
