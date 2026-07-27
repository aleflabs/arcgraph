//! W13ε M5-02 — rotation atomicity proptest.
//!
//! Invariant: at any point during a series of concurrent
//! `reload()` + `current()` calls, every observer must see ONE of the
//! valid `Arc<CertifiedKey>` snapshots — never a half-rotated state.
//!
//! Implementation note: rustls' `CertifiedKey` is not `PartialEq`, so
//! we identify each snapshot by its end-entity DER bytes (a unique
//! fingerprint per rotation generation since each generation gets a
//! fresh keypair → fresh SPKI → fresh signed certificate).

mod tls_common;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use arcgraph_mcp::tls::{FileSystemCertProvider, HotReloadResolver};
use proptest::prelude::*;
use tls_common::CertFixture;

/// Observe the resolver under simulated load. The reader thread spins
/// `current()` calls while the writer thread issues `reload()`s; the
/// observed DER set must always be a subset of the produced DERs
/// (i.e., no torn-rotation observation).
fn drive_concurrent_rotation(rotations: usize) -> (Vec<Vec<u8>>, HashSet<Vec<u8>>) {
    let fixture = CertFixture::fresh_localhost();
    let provider = Arc::new(FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("initial load");
    let stop = Arc::new(AtomicBool::new(false));

    // Reader thread captures every DER it observes.
    let reader_resolver = Arc::clone(&resolver);
    let reader_stop = Arc::clone(&stop);
    let reader = thread::spawn(move || {
        let mut observed: Vec<Vec<u8>> = Vec::new();
        while !reader_stop.load(Ordering::Acquire) {
            let snap = reader_resolver.current();
            observed.push(snap.cert[0].as_ref().to_vec());
            // Light spin — let the writer make forward progress.
            std::thread::yield_now();
        }
        // Drain one final reading after stop so the last rotation is
        // included in the observed set.
        observed.push(reader_resolver.current().cert[0].as_ref().to_vec());
        observed
    });

    // Writer rotates `rotations` times.
    let mut produced: HashSet<Vec<u8>> = HashSet::new();
    // Initial cert is the producer's first ground-truth value.
    produced.insert(resolver.current().cert[0].as_ref().to_vec());
    for _ in 0..rotations {
        fixture.rotate_with_san(&["localhost".into()]);
        resolver.reload().expect("reload");
        produced.insert(resolver.current().cert[0].as_ref().to_vec());
        // Yield so the reader sees the rotation.
        std::thread::yield_now();
    }

    stop.store(true, Ordering::Release);
    let observed = reader.join().expect("reader thread");

    (observed, produced)
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Concurrent IO + cert generation is heavy — keep cases low.
        cases: 8,
        ..ProptestConfig::default()
    })]

    /// Every cert observed by the reader thread must be one of the
    /// certs the writer produced. Equivalently: no observation is a
    /// "phantom" cert never installed by the resolver.
    #[test]
    fn rotation_atomicity_observed_subset_of_produced(
        rotations in 1usize..6,
    ) {
        let (observed, produced) = drive_concurrent_rotation(rotations);
        prop_assert!(
            !observed.is_empty(),
            "reader thread must capture at least one observation"
        );
        for o in &observed {
            prop_assert!(
                produced.contains(o),
                "observed cert was never produced — possible torn rotation"
            );
        }
    }
}
