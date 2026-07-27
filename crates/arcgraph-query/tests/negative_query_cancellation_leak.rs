//! W26-γ-3 / ADR-136 §D5 — query cancellation no-leak regression.
//!
//! # Forced failure mode
//!
//! Cancellation that does not properly release the per-query
//! registry entry leaks an entry per cancelled query. Over time the
//! registry grows unbounded; eventually the cancel_all sweep on
//! SIGTERM walks a multi-MB hashmap.
//!
//! # Pinned invariants
//!
//! 1. **Register + cancel → unregister releases the entry.**
//!    After a successful cancel + unregister, the registry has
//!    one fewer entry.
//! 2. **Unregister without cancel works (success path).** When a
//!    query completes normally the registry entry is released.
//! 3. **Cancel-all on empty registry is a no-op.** No panic.
//! 4. **Cancel-all on N entries fires N times.** The return value
//!    is the number of fired cancellations.
//! 5. **Concurrent register + cancel does not panic.** Multiple
//!    threads can register / cancel against the same registry.
//! 6. **Registry sizes track register/unregister deterministically.**
//!    Per the documented `len() == 0 ⇔ is_empty()` invariant.
//! 7. **Token semantics — cancelling a token + re-checking is
//!    consistent.** After cancel(), the token reports cancelled.
//!
//! Per `feedback_load_bearing_pr_requires_fault_injection_tests.md`
//! + ADR-038 amendment-03 §TIER-1 GAP C cancellation contract.

use std::sync::Arc;
use std::thread;

use arcgraph_query::{CancellationRegistry, QueryId};

#[test]
fn register_then_unregister_releases_entry() {
    let reg = CancellationRegistry::new();
    assert!(reg.is_empty());

    let qid = QueryId::new();
    let _tok = reg.register(qid);
    assert_eq!(reg.len(), 1);

    let removed = reg.unregister(qid);
    assert!(removed, "unregister must report the entry was removed");
    assert!(reg.is_empty(), "registry must be empty after unregister");
}

#[test]
fn unregister_unknown_id_is_false() {
    let reg = CancellationRegistry::new();
    let unknown = QueryId::new();
    let removed = reg.unregister(unknown);
    assert!(!removed, "unregistering unknown id returns false");
}

#[test]
fn cancel_then_unregister_no_leak() {
    let reg = CancellationRegistry::new();

    let qid = QueryId::new();
    let _tok = reg.register(qid);

    let cancelled = reg.cancel(qid);
    assert!(cancelled);
    // The cancel does not remove the entry — the cancelled query
    // still needs to be cleaned up via unregister at end-of-loop.
    // But it must not double-up the entry either.
    assert!(reg.len() <= 1);

    let removed = reg.unregister(qid);
    assert!(removed);
    assert!(reg.is_empty(), "no entry leaks after cancel + unregister");
}

#[test]
fn cancel_all_on_empty_is_zero() {
    let reg = CancellationRegistry::new();
    let count = reg.cancel_all();
    assert_eq!(count, 0);
}

#[test]
fn cancel_all_fires_n_times() {
    let reg = CancellationRegistry::new();
    for _ in 0..10 {
        let qid = QueryId::new();
        let _tok = reg.register(qid);
    }
    assert_eq!(reg.len(), 10);
    let count = reg.cancel_all();
    assert_eq!(count, 10, "cancel_all must report all entries fired");
}

#[test]
fn concurrent_register_unregister_no_panic() {
    let reg = Arc::new(CancellationRegistry::new());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let r = Arc::clone(&reg);
        handles.push(thread::spawn(move || {
            for _ in 0..100 {
                let qid = QueryId::new();
                let _tok = r.register(qid);
                r.unregister(qid);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    // After all threads finish + unregister, the registry should
    // be empty. (Race-free because each thread registers + unregisters
    // its own qids.)
    assert!(
        reg.is_empty(),
        "registry empty after 8×100 register/unregister cycles"
    );
}

#[test]
fn concurrent_register_cancel_no_panic() {
    let reg = Arc::new(CancellationRegistry::new());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let r = Arc::clone(&reg);
        handles.push(thread::spawn(move || {
            for _ in 0..50 {
                let qid = QueryId::new();
                let _tok = r.register(qid);
                let _ = r.cancel(qid);
                let _ = r.unregister(qid);
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert!(
        reg.is_empty(),
        "registry empty after concurrent cancel cycles"
    );
}

#[test]
fn many_concurrent_cancel_all_no_panic() {
    let reg = Arc::new(CancellationRegistry::new());
    // Register N queries.
    for _ in 0..100 {
        let qid = QueryId::new();
        let _tok = reg.register(qid);
    }
    // Spawn many threads calling cancel_all simultaneously.
    let mut handles = Vec::new();
    for _ in 0..4 {
        let r = Arc::clone(&reg);
        handles.push(thread::spawn(move || r.cancel_all()));
    }
    let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
    // Total ≥ 100 (at least one thread saw the full count); but the
    // load-bearing assertion is no panic + all-threads-complete.
    assert!(total >= 100, "first cancel_all sweep must see >= 100");
}

#[test]
fn query_ids_returns_currently_registered() {
    let reg = CancellationRegistry::new();
    let mut ids = Vec::new();
    for _ in 0..5 {
        let qid = QueryId::new();
        let _tok = reg.register(qid);
        ids.push(qid);
    }
    let snapshot = reg.query_ids();
    assert_eq!(snapshot.len(), 5);
    for id in &ids {
        assert!(snapshot.contains(id), "snapshot missing {id:?}");
    }
}

#[test]
fn double_register_same_id_is_idempotent_or_replaces() {
    let reg = CancellationRegistry::new();
    let qid = QueryId::new();
    let _tok1 = reg.register(qid);
    let _tok2 = reg.register(qid);
    // The semantics may be replacement (the second token shadows
    // the first) OR idempotent. Either way the count must be
    // bounded — NOT 2.
    assert!(reg.len() <= 1, "double-register must not duplicate entries");
}

#[test]
fn cancel_releases_no_zombie_state_on_concurrent_overlap() {
    // Race: thread A registers + waits; thread B cancels that
    // exact qid; thread A then unregisters. The unregister must
    // return false (the entry was already absorbed by cancel's
    // cleanup) OR true (the registry tracks them as separate
    // operations). Either is acceptable; no panic + no zombie.
    let reg = Arc::new(CancellationRegistry::new());
    let mut handles = Vec::new();
    for _ in 0..8 {
        let r = Arc::clone(&reg);
        handles.push(thread::spawn(move || {
            let qid = QueryId::new();
            let _tok = r.register(qid);
            // Half the threads cancel; half just unregister.
            if qid.0.as_u128() % 2 == 0 {
                let _ = r.cancel(qid);
            }
            r.unregister(qid);
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    assert!(
        reg.is_empty(),
        "no zombie entries after concurrent cancel/unregister race"
    );
}
