//! W13ε M5-02 integration suite for the TLS hot-reload resolver.
//!
//! Three test surfaces:
//!   1. End-to-end `FileSystemCertProvider::load_validated` happy + sad
//!      paths — exercises PEM parse → validation → CertifiedKey.
//!   2. SIGHUP-driven reload — drives the unix-only signal loop with a
//!      synthetic SIGHUP and asserts the resolver picks up the new cert.
//!   3. Concurrent rotation — the proptest in `tls_resolver_proptest.rs`
//!      pins the no-half-swap invariant.

mod tls_common;

use std::sync::Arc;

use arcgraph_mcp::tls::{
    Clock, FileSystemCertProvider, HotReloadResolver, TlsResolverError, drive_reload,
};
use time::{Duration as TimeDuration, OffsetDateTime};
use tls_common::{CertFixture, write_malformed_pem};

#[derive(Debug, Clone, Copy)]
struct FixedClock(i64);
impl Clock for FixedClock {
    fn now_unix(&self) -> i64 {
        self.0
    }
}

#[test]
fn integ_load_validated_happy_path_with_localhost_san() {
    let fixture = CertFixture::fresh_localhost();
    let provider = FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    );
    let cert_key = provider.load_validated().expect("happy path");
    assert!(!cert_key.cert.is_empty(), "cert chain should be populated");
}

#[test]
fn integ_load_validated_rejects_hostname_mismatch() {
    let fixture = CertFixture::fresh_localhost();
    let provider = FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("not.example.org".into()),
    );
    let err = provider.load_validated().expect_err("mismatch");
    assert!(
        matches!(err, TlsResolverError::HostnameMismatch { .. }),
        "expected HostnameMismatch, got {err:?}"
    );
}

#[test]
fn integ_load_validated_falls_back_to_cn_when_san_absent() {
    let fixture = CertFixture::cn_only("api.internal");
    let provider = FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("api.internal".into()),
    );
    provider.load_validated().expect("CN fallback should match");
}

#[test]
fn integ_load_validated_rejects_expired_cert() {
    // Cert that expired 1 day ago.
    let now = OffsetDateTime::now_utc();
    let fixture = CertFixture::new_with_san_and_validity(
        &["localhost".into()],
        now - TimeDuration::days(7),
        now - TimeDuration::days(1),
        None,
    );
    // Use SystemClock (real clock) — cert is already expired so it
    // should fail without needing a fixed clock.
    let provider = FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    );
    let err = provider.load_validated().expect_err("expired");
    assert!(
        matches!(
            err,
            TlsResolverError::ValidityWindow {
                phase: "not_after",
                ..
            }
        ),
        "expected not_after window error, got {err:?}"
    );
}

#[test]
fn integ_load_validated_rejects_not_yet_valid_cert() {
    // Cert valid in the future (not_before > now).
    let now = OffsetDateTime::now_utc();
    let fixture = CertFixture::new_with_san_and_validity(
        &["localhost".into()],
        now + TimeDuration::days(7),
        now + TimeDuration::days(30),
        None,
    );
    let provider = FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    );
    let err = provider.load_validated().expect_err("not_before");
    assert!(
        matches!(
            err,
            TlsResolverError::ValidityWindow {
                phase: "not_before",
                ..
            }
        ),
        "expected not_before window error, got {err:?}"
    );
}

#[test]
fn integ_load_validated_rejects_malformed_pem() {
    let fixture = CertFixture::fresh_localhost();
    write_malformed_pem(&fixture.cert_path);
    let provider = FileSystemCertProvider::new(&fixture.cert_path, &fixture.key_path, None);
    let err = provider.load_validated().expect_err("malformed");
    // The PEM block has a CERTIFICATE armor but invalid base64; the
    // parser yields either MalformedPem or NoCertificatesFound
    // depending on whether base64 errors are surfaced as parse errors
    // or filtered out. Both indicate "no usable cert" so the test
    // accepts either.
    assert!(
        matches!(
            err,
            TlsResolverError::MalformedPem { .. } | TlsResolverError::NoCertificatesFound { .. }
        ),
        "expected MalformedPem or NoCertificatesFound, got {err:?}"
    );
}

#[test]
fn integ_load_validated_rejects_missing_cert_file() {
    let fixture = CertFixture::fresh_localhost();
    let missing_cert = fixture.dir.path().join("does-not-exist.crt");
    let provider = FileSystemCertProvider::new(&missing_cert, &fixture.key_path, None);
    let err = provider.load_validated().expect_err("missing");
    assert!(
        matches!(err, TlsResolverError::Io { .. }),
        "expected Io, got {err:?}"
    );
}

#[test]
fn integ_load_validated_rejects_mismatched_key() {
    // Two fixtures with different keys — copy fixture A's cert next to
    // fixture B's key. The signing_key/cert public-key SPKI compare
    // will fail.
    let a = CertFixture::fresh_localhost();
    let b = CertFixture::fresh_localhost();
    let provider = FileSystemCertProvider::new(&a.cert_path, &b.key_path, Some("localhost".into()));
    let err = provider.load_validated().expect_err("key mismatch");
    assert!(
        matches!(err, TlsResolverError::KeyMismatch { .. }),
        "expected KeyMismatch, got {err:?}"
    );
}

#[test]
fn integ_resolver_rotation_swaps_to_new_cert_atomically() {
    let fixture = CertFixture::fresh_localhost();
    let provider = Arc::new(FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("initial load");
    let before = resolver.current();
    // Capture the end-entity cert DER bytes — these uniquely identify
    // the cert (different keypair → different SPKI → different DER).
    let before_der = before.cert[0].as_ref().to_vec();

    // Rotate: write new cert/key into the same paths.
    fixture.rotate_with_san(&["localhost".into()]);
    resolver.reload().expect("reload");

    let after = resolver.current();
    let after_der = after.cert[0].as_ref().to_vec();
    assert_ne!(
        before_der, after_der,
        "rotated cert DER must differ from initial"
    );
    // Also confirm the resolver is still serviceable post-rotation
    // (i.e., reload didn't put us in a degraded state).
    assert!(!after.cert.is_empty());
}

#[test]
fn integ_resolver_keeps_old_when_rotation_writes_corrupt_pem() {
    let fixture = CertFixture::fresh_localhost();
    let provider = Arc::new(FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("initial load");
    let before = resolver.current();

    // Corrupt the cert file mid-rotation.
    write_malformed_pem(&fixture.cert_path);

    let err = resolver.reload().expect_err("corrupted");
    assert!(matches!(
        err,
        TlsResolverError::MalformedPem { .. } | TlsResolverError::NoCertificatesFound { .. }
    ));

    // The resolver MUST still serve the original cert.
    let after = resolver.current();
    assert!(
        Arc::ptr_eq(&before, &after),
        "resolver dropped old cert despite reload failure"
    );
}

/// Partial-rotation TOCTOU regression test.
///
/// `FileSystemCertProvider::load_validated` opens cert + key files in
/// two separate `File::open` calls. If the operator atomically replaces
/// both via two `rename` syscalls (cert first, key second, or vice
/// versa), there is a window where the provider sees the new cert but
/// the old key — i.e., a CertifiedKey wrapping a public-key SPKI that
/// doesn't match the private key SPKI. The validation pipeline catches
/// this in `keys_match()`; this test pins that behavior so a future
/// refactor cannot silently drop the keys-match step (which would
/// install a partial-rotation cert that fails handshakes downstream
/// while logging "tls.reload.success", a hard-to-diagnose failure
/// shape).
#[test]
fn integ_resolver_keeps_old_when_cert_rotated_but_key_lags() {
    let fixture = CertFixture::fresh_localhost();
    let provider = Arc::new(FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("initial load");
    let before = resolver.current();

    // Simulate the partial-rotation race: replace cert with a NEW
    // generation's cert, but leave the key file as the OLD generation.
    // This is what the operator's two-step rename hits in the window
    // between rename(new.cert -> server.crt) and rename(new.key ->
    // server.key).
    let new = CertFixture::fresh_localhost();
    std::fs::copy(&new.cert_path, &fixture.cert_path).expect("rotate cert in-place");
    // (key file deliberately untouched — still old generation)

    let err = resolver.reload().expect_err("partial rotation must fail");
    assert!(
        matches!(err, TlsResolverError::KeyMismatch { .. }),
        "expected KeyMismatch from partial rotation, got {err:?}"
    );
    assert!(
        Arc::ptr_eq(&resolver.current(), &before),
        "resolver must keep old cert on partial-rotation TOCTOU"
    );
}

#[test]
fn integ_drive_reload_logs_failure_without_panicking() {
    let fixture = CertFixture::fresh_localhost();
    let provider = Arc::new(FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("initial load");
    let before_ptr = Arc::as_ptr(&resolver.current());

    // Corrupt; drive_reload should NOT propagate the error or panic.
    write_malformed_pem(&fixture.cert_path);
    drive_reload(&resolver);

    let after_ptr = Arc::as_ptr(&resolver.current());
    assert_eq!(
        before_ptr, after_ptr,
        "drive_reload installed a cert despite validation failure"
    );
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn integ_sighup_loop_triggers_reload_end_to_end() {
    use std::time::Duration;
    use tokio::sync::watch;

    let fixture = CertFixture::fresh_localhost();
    let provider = Arc::new(FileSystemCertProvider::new(
        &fixture.cert_path,
        &fixture.key_path,
        Some("localhost".into()),
    ));
    let resolver = HotReloadResolver::new(provider).expect("initial load");
    let before_der = resolver.current().cert[0].as_ref().to_vec();

    let (shutdown_tx, shutdown_rx) = watch::channel(false);
    let resolver_clone = Arc::clone(&resolver);
    let loop_handle = tokio::spawn(async move {
        arcgraph_mcp::tls::run_sighup_reload_loop(resolver_clone, shutdown_rx)
            .await
            .expect("loop ok")
    });

    // Give the loop a moment to install the SIGHUP handler.
    tokio::time::sleep(Duration::from_millis(50)).await;

    // Rotate certs on disk + send ourselves SIGHUP.
    fixture.rotate_with_san(&["localhost".into()]);

    let pid = std::process::id();
    // SAFETY: `libc::kill` is FFI; we pass our own PID
    // (`std::process::id()` cast to `pid_t = i32`) and a standard
    // signal number (`SIGHUP`). With `pid > 0`, POSIX kill(2)
    // delivers the signal to the calling process specifically — NOT
    // the calling process group (that would be `pid == 0` semantics,
    // which we deliberately do NOT use to avoid signaling the test
    // runner's parent shell). Tokio's `signal::unix` handler picks
    // it up on its dedicated reactor thread. No memory or aliasing
    // invariant to uphold; the `libc::kill` ABI is stable and
    // well-defined, the args are scalar.
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGHUP);
    }

    // Poll for the rotation to take effect (loop is async; the swap
    // happens after `hup.recv()` resolves + `drive_reload` runs).
    let mut rotated = false;
    for _ in 0..50 {
        tokio::time::sleep(Duration::from_millis(20)).await;
        let current_der = resolver.current().cert[0].as_ref().to_vec();
        if current_der != before_der {
            rotated = true;
            break;
        }
    }

    shutdown_tx.send(true).expect("shutdown");
    let _ = tokio::time::timeout(Duration::from_secs(2), loop_handle).await;

    assert!(
        rotated,
        "resolver did not pick up rotated cert after SIGHUP"
    );
}

/// Validity-window expiry-edge test using a deterministic clock so
/// the assertion is independent of wall-clock skew.
#[test]
fn integ_validity_window_expiry_at_exact_boundary_uses_clock() {
    use arcgraph_mcp::tls::build_certified_key;
    use rustls_pki_types::pem::PemObject;
    use rustls_pki_types::{CertificateDer, PrivateKeyDer};

    let now = OffsetDateTime::now_utc();
    let fixture = CertFixture::new_with_san_and_validity(
        &["localhost".into()],
        now - TimeDuration::days(1),
        now + TimeDuration::days(1),
        None,
    );

    // Re-parse cert + key from disk (mirroring what the provider does).
    let cert_chain: Vec<_> = CertificateDer::pem_file_iter(&fixture.cert_path)
        .expect("open cert file")
        .collect::<Result<Vec<_>, _>>()
        .expect("parse cert chain");
    let key = PrivateKeyDer::from_pem_file(&fixture.key_path).expect("parse key");

    // FixedClock 10 years in the future → past `not_after`.
    let future_clock = FixedClock(now.unix_timestamp() + 10 * 365 * 86_400);
    let err = build_certified_key(cert_chain, key, Some("localhost"), &future_clock)
        .expect_err("future clock past not_after");
    assert!(matches!(
        err,
        TlsResolverError::ValidityWindow {
            phase: "not_after",
            ..
        }
    ));
}
