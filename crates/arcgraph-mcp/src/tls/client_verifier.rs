//! W20β-1 — mTLS client-cert verifier surface.
//!
//! Three concerns:
//!
//! 1. **PEM ingestion** ([`client_verifier_from_ca_pem`]). Accepts a PEM
//!    bundle of trusted client-CA roots, returns an `Arc<dyn
//!    ClientCertVerifier>` ready to install on `rustls::ServerConfig`.
//!    The `client_cert_required` flag selects the rustls posture:
//!    - `true`  → `WebPkiClientVerifier::builder(...).build()` — every
//!      accepted connection MUST present a chain-validating client cert;
//!      handshake fails otherwise.
//!    - `false` → `WebPkiClientVerifier::builder(...).allow_unauthenticated().build()`
//!      — operator offers mTLS but does not enforce; per-request handler
//!      decides whether the absence of a peer cert is acceptable for the
//!      method.
//!
//! 2. **Identity extraction** ([`parse_client_cert_identity`]). Returns
//!    a [`ClientCertIdentity`] carrying the X.500 CN (from the cert's
//!    subject distinguished name) + the SAN DNSName entries. The
//!    per-request handler invokes this on the peer's end-entity DER so
//!    the dispatcher / tracing layer can route on CN or SAN without
//!    re-parsing the cert.
//!
//! 3. **Reload-safe wrapper** ([`HotReloadClientVerifier`]). Mirrors the
//!    [`super::resolver::HotReloadResolver`] pattern: holds the current
//!    `Arc<dyn ClientCertVerifier>` in an `ArcSwap` so SIGHUP-driven CA
//!    rotation is observed by NEW handshakes without restart. Existing
//!    handshakes complete against the prior verifier per RFC 8446 §4.6.3
//!    (no in-band rekey).
//!
//! # Latency / memory budget
//!
//! - `parse_client_cert_identity` per-handshake: 1 X.509 DER parse +
//!   1 SAN walk + 1 CN scan. ≈ 10-50 μs on a 1-2 KB end-entity cert
//!   (dominated by ASN.1 BER decode); the TLS handshake itself is
//!   1-10 ms so the parse is ≤ 1% overhead.
//! - `HotReloadClientVerifier::verify_client_cert`: 1 ArcSwap load
//!   (≈ 5-15 ns) + 1 underlying-verifier delegate. The wrapper adds
//!   ≤ 0.01% overhead on top of the rustls chain-verify cost (which is
//!   itself bounded by RSA / ECDSA signature verify ≈ 50-300 μs).
//! - Reload: one ArcSwap store; the previous `Arc<dyn
//!   ClientCertVerifier>` is dropped when no in-flight handshake holds
//!   it.
//!
//! # ADR provenance
//!
//! - **design-v2 §9.4** — mandates HTTPS for non-stdio transports but
//!   does not specify client-cert verification at v1.0-α. W20β-1 ADDS
//!   mTLS as an additive identity surface on top of the OAuth bearer
//!   path the section enumerates (per `feedback_cite_correctness_not_just_resolution.md`:
//!   cited section must say what the claim attributes to it; §9.4 does
//!   not mention mTLS explicitly).
//! - **W14ε TLS hot-reload** — the [`super::resolver::HotReloadResolver`]
//!   pattern this module mirrors for the client-CA bundle.
//! - **code-quality policy** — `#[non_exhaustive]` on the surfaced error
//!   surface; `#[serde(deny_unknown_fields)]` is N/A (this module
//!   surfaces no `pub struct *Config`).

use std::fmt;
use std::sync::Arc;

use arc_swap::ArcSwap;
use rustls::DigitallySignedStruct;
use rustls::DistinguishedName;
use rustls::Error;
use rustls::SignatureScheme;
use rustls::client::danger::HandshakeSignatureValid;
use rustls::server::WebPkiClientVerifier;
use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
use rustls_pki_types::pem::PemObject;
use rustls_pki_types::{CertificateDer, UnixTime};

use super::error::TlsResolverError;

/// W20β-1 — peer identity extracted from a presented client cert.
///
/// The transport layer surfaces this struct on the per-request context
/// after the TLS handshake's peer-cert chain has been validated. The
/// dispatcher / tracing layer routes on it without re-parsing the DER.
///
/// At v1.0-β the dispatcher does NOT consume the CN for authorization
/// (the SAN-based `tenant-<N>` strategy is the canonical tenant pin per
/// [`crate::transport::http::TenantStrategy::PeerCertSan`]); the CN is
/// emitted on the per-request tracing span for operator audit.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClientCertIdentity {
    /// X.500 Common Name from the cert's subject DN. `None` when the
    /// subject DN carries no `CN=...` attribute (rare in practice; CAs
    /// typically emit a CN even on machine certs).
    pub cn: Option<String>,
    /// SubjectAltName DNSName entries, in the order they appear in the
    /// X.509 extension. Other SAN types (IP, URI, RFC822) are ignored
    /// at v1.0-β; v1.1+ may surface SAN.IP for IP-pinned mTLS.
    pub sans: Vec<String>,
}

impl ClientCertIdentity {
    /// `true` when neither CN nor any SAN was extracted (e.g., the cert
    /// has a degenerate subject DN AND no SAN extension — pathological,
    /// but rustls would have already rejected such a cert at the
    /// chain-verify stage).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cn.is_none() && self.sans.is_empty()
    }
}

/// Parse the X.500 CN + SubjectAltName DNSName entries from a DER-
/// encoded end-entity X.509 cert.
///
/// Returns an empty [`ClientCertIdentity`] (no CN, no SANs) when the
/// cert parses but carries neither attribute — this is degenerate but
/// not an error at the verifier surface (rustls's chain-validate is the
/// real gate).
///
/// # Errors
///
/// Returns [`TlsResolverError::MalformedPem`] (re-used for X.509 DER
/// parse failures; the variant message field names "client cert
/// parse" so operators can distinguish from server-cert PEM errors)
/// when the input is not a valid X.509 cert.
pub fn parse_client_cert_identity(der: &[u8]) -> Result<ClientCertIdentity, TlsResolverError> {
    use x509_parser::extensions::GeneralName;
    use x509_parser::prelude::*;

    let (_, cert) = X509Certificate::from_der(der).map_err(|e| TlsResolverError::MalformedPem {
        path: std::path::PathBuf::from("<peer-client-cert>"),
        detail: format!("client cert X.509 decode: {e}"),
    })?;

    let mut cn: Option<String> = None;
    for rdn in cert.subject().iter_common_name() {
        if let Ok(s) = rdn.as_str() {
            cn = Some(s.to_string());
            break;
        }
    }

    let mut sans = Vec::new();
    if let Ok(Some(san_ext)) = cert.subject_alternative_name() {
        for name in &san_ext.value.general_names {
            if let GeneralName::DNSName(d) = name {
                sans.push((*d).to_string());
            }
        }
    }

    Ok(ClientCertIdentity { cn, sans })
}

/// W20β-1 — build a [`ClientCertVerifier`] from a PEM bundle of trusted
/// client-CA roots.
///
/// `client_cert_required` selects the rustls posture:
/// - `true`  → handshake REJECTS clients that present no cert or whose
///   cert does not chain to a trusted root (`WebPkiClientVerifier::builder(...).build()`).
/// - `false` → handshake completes for clients with NO cert; clients
///   that DO present a cert are still chain-validated and rejected on
///   chain failure (`WebPkiClientVerifier::builder(...).allow_unauthenticated().build()`).
///
/// The `false` posture is the canonical "mTLS-optional" deployment shape
/// (e.g., embedded use where the operator wants OAuth as the primary
/// auth but lets mTLS-capable clients additionally pin via SAN).
///
/// # Revocation (CRL / OCSP) — v1.1+ deferral
///
/// **Client-cert revocation is NOT enforced at v1.0-β.** Neither the
/// `client_cert_required = true` nor the `allow_unauthenticated()` path
/// invokes `WebPkiClientVerifier::builder(...).with_crls(...)`, so a
/// revoked client cert that still chain-validates against a trusted CA
/// is admitted. This mirrors the server-side
/// `super::validation::validate_cert_chain` documented deferral
/// (lines 51–52 of `tls/validation.rs`): the v1.1+ integration point
/// stages CRL bundles operator-side and reloads via the same SIGHUP
/// path as the trust-store, analogous to [`HotReloadClientVerifier`].
/// OCSP stapling has no rustls v1.x integration yet.
///
/// Operational implication: a client cert issued legitimately and later
/// compromised cannot be revoked between issuance + cert expiry at
/// v1.0-β. Operators relying on mTLS as a sole identity surface should
/// keep cert TTLs short and/or layer OAuth on top so a compromised cert
/// is bounded by token expiry as well.
///
/// # Errors
///
/// - [`TlsResolverError::MalformedPem`] if the PEM bytes don't decode
///   into ≥ 1 CERTIFICATE block, or if any individual cert is malformed.
/// - [`TlsResolverError::NoCertificatesFound`] if the PEM parses but
///   yields zero certs (degenerate input — operator most likely meant
///   to disable mTLS by passing `None` for the verifier).
/// - [`TlsResolverError::Validation`] if rustls's builder rejects the
///   trust store (e.g., a cert was admissible-DER but unable to serve as
///   a trust anchor — bad key usage extension).
pub fn client_verifier_from_ca_pem(
    pem: &[u8],
    client_cert_required: bool,
) -> Result<Arc<dyn ClientCertVerifier>, TlsResolverError> {
    let mut store = rustls::RootCertStore::empty();
    let mut added = 0usize;
    for cert_res in CertificateDer::pem_slice_iter(pem) {
        let cert = cert_res.map_err(|e| TlsResolverError::MalformedPem {
            path: std::path::PathBuf::from("<client-ca-pem>"),
            detail: format!("client-CA PEM parse: {e}"),
        })?;
        store.add(cert).map_err(|e| TlsResolverError::Validation {
            detail: format!("client-CA trust anchor add: {e}"),
        })?;
        added += 1;
    }
    if added == 0 {
        return Err(TlsResolverError::NoCertificatesFound {
            path: std::path::PathBuf::from("<client-ca-pem>"),
        });
    }

    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let builder = WebPkiClientVerifier::builder_with_provider(Arc::new(store), provider);
    let verifier = if client_cert_required {
        builder.build().map_err(|e| TlsResolverError::Validation {
            detail: format!("client verifier build (required): {e}"),
        })?
    } else {
        builder
            .allow_unauthenticated()
            .build()
            .map_err(|e| TlsResolverError::Validation {
                detail: format!("client verifier build (optional): {e}"),
            })?
    };
    Ok(verifier)
}

// ─────────────────────────────────────────────────────────────────────
// Hot-reload wrapper
// ─────────────────────────────────────────────────────────────────────

/// W20β-1 — hot-reloading client-cert verifier.
///
/// Holds the current `Arc<dyn ClientCertVerifier>` in an `ArcSwap` so
/// SIGHUP-driven CA rotation is observed by NEW handshakes without a
/// listener restart. Mirrors the [`super::resolver::HotReloadResolver`]
/// pattern.
///
/// Construct via [`HotReloadClientVerifier::new`] — the initial verifier
/// is mandatory so the wrapper always has a chain validator from the
/// moment it's installed on `rustls::ServerConfig`.
pub struct HotReloadClientVerifier {
    current: ArcSwap<Arc<dyn ClientCertVerifier>>,
}

impl fmt::Debug for HotReloadClientVerifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HotReloadClientVerifier")
            .finish_non_exhaustive()
    }
}

impl HotReloadClientVerifier {
    /// Construct a hot-reloadable wrapper rooted at `initial`.
    #[must_use]
    pub fn new(initial: Arc<dyn ClientCertVerifier>) -> Arc<Self> {
        Arc::new(Self {
            current: ArcSwap::from(Arc::new(initial)),
        })
    }

    /// Swap to a new verifier. On success, the new `Arc<dyn
    /// ClientCertVerifier>` replaces the previous one atomically; the
    /// previous verifier is dropped when no in-flight handshake holds
    /// it. Returns the prior verifier so callers can hand it back if
    /// they want a stack-based rollback.
    pub fn reload(&self, next: Arc<dyn ClientCertVerifier>) -> Arc<dyn ClientCertVerifier> {
        let prev = self.current.load_full();
        self.current.store(Arc::new(next));
        (*prev).clone()
    }

    /// Snapshot the currently-installed verifier as an `Arc` clone.
    #[must_use]
    pub fn current(&self) -> Arc<dyn ClientCertVerifier> {
        (*self.current.load_full()).clone()
    }
}

impl ClientCertVerifier for HotReloadClientVerifier {
    fn offer_client_auth(&self) -> bool {
        (*self.current.load()).offer_client_auth()
    }

    fn client_auth_mandatory(&self) -> bool {
        (*self.current.load()).client_auth_mandatory()
    }

    fn root_hint_subjects(&self) -> &[DistinguishedName] {
        // Returns an empty hint slice unconditionally. RFC 8446 §4.2.4
        // admits this ("certificate_authorities" is OPTIONAL): an empty
        // list signals "no hints provided" and clients fall back to
        // presenting whichever client cert they have. The wrapper trades
        // hint-list optimization for reload-safety because the ArcSwap-
        // Guard borrow cannot outlive `&self` — re-loading per call to
        // delegate the inner slice would dangle. Operators that need
        // hint-driven cert selection should install the non-reload-
        // wrapped verifier returned by `client_verifier_from_ca_pem`
        // directly (the inner WebPkiClientVerifier carries the real
        // hint list); the wrapper is opt-in for reload paths only.
        const EMPTY: &[DistinguishedName] = &[];
        EMPTY
    }

    fn verify_client_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        now: UnixTime,
    ) -> Result<ClientCertVerified, Error> {
        let v = self.current.load_full();
        v.verify_client_cert(end_entity, intermediates, now)
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        let v = self.current.load_full();
        v.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, Error> {
        let v = self.current.load_full();
        v.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        (*self.current.load()).supported_verify_schemes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rcgen::{CertificateParams, DistinguishedName, DnType, KeyPair};
    use time::OffsetDateTime;

    /// Synthesize a self-signed cert + return (DER, CN, SAN-list).
    ///
    /// `cn = Some(_)` pushes a CN attribute; `cn = None` constructs an
    /// EMPTY DistinguishedName so the resulting cert truly has no CN
    /// (rcgen's default DN otherwise inserts `CN="rcgen self signed cert"`
    /// which would defeat the None-branch test).
    fn synth_cert(cn: Option<&str>, sans: &[&str]) -> (Vec<u8>, Option<String>, Vec<String>) {
        let san_vec: Vec<String> = sans.iter().map(|s| (*s).to_string()).collect();
        let mut params = CertificateParams::new(san_vec.clone()).expect("rcgen params");
        params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        match cn {
            Some(c) => {
                // Start with an empty DN so we control exactly what's in
                // the subject — pushing on top of rcgen's default would
                // leave the rcgen-default CN around, which would defeat
                // the test's "first CN wins" semantic.
                params.distinguished_name = DistinguishedName::new();
                params.distinguished_name.push(DnType::CommonName, c);
            }
            None => {
                params.distinguished_name = DistinguishedName::new();
            }
        }
        let kp = KeyPair::generate().expect("keypair");
        let cert = params.self_signed(&kp).expect("self-signed");
        (
            cert.der().as_ref().to_vec(),
            cn.map(|c| c.to_string()),
            san_vec,
        )
    }

    /// Synthesize a self-signed cert as a PEM blob.
    fn synth_cert_pem(cn: Option<&str>, sans: &[&str]) -> Vec<u8> {
        let san_vec: Vec<String> = sans.iter().map(|s| (*s).to_string()).collect();
        let mut params = CertificateParams::new(san_vec).expect("rcgen params");
        params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        if let Some(c) = cn {
            params.distinguished_name.push(DnType::CommonName, c);
        }
        let kp = KeyPair::generate().expect("keypair");
        let cert = params.self_signed(&kp).expect("self-signed");
        cert.pem().into_bytes()
    }

    #[test]
    fn parse_client_cert_identity_extracts_cn_and_sans() {
        let (der, expected_cn, expected_sans) = synth_cert(
            Some("alice.example.com"),
            &["alice.example.com", "tenant-7"],
        );
        let id = parse_client_cert_identity(&der).expect("parse");
        assert_eq!(id.cn, expected_cn);
        assert_eq!(id.sans, expected_sans);
    }

    #[test]
    fn parse_client_cert_identity_handles_san_only_cert() {
        let (der, _, expected_sans) = synth_cert(None, &["bob.example.com"]);
        let id = parse_client_cert_identity(&der).expect("parse");
        assert_eq!(id.cn, None);
        assert_eq!(id.sans, expected_sans);
    }

    #[test]
    fn parse_client_cert_identity_handles_cn_only_cert() {
        // rcgen requires at least one SAN to construct CertificateParams;
        // the SAN goes in but we verify CN extraction independently.
        let (der, expected_cn, _) = synth_cert(Some("carol.example.com"), &["x.example.com"]);
        let id = parse_client_cert_identity(&der).expect("parse");
        assert_eq!(id.cn, expected_cn);
    }

    #[test]
    fn parse_client_cert_identity_rejects_malformed_der() {
        let bogus = b"\x00\x01\x02not-a-cert\x03\x04\x05";
        let err = parse_client_cert_identity(bogus).expect_err("must reject");
        assert!(matches!(err, TlsResolverError::MalformedPem { .. }));
    }

    #[test]
    fn client_verifier_from_ca_pem_round_trips_required_posture() {
        let pem = synth_cert_pem(Some("Test CA"), &["ca.example.com"]);
        let verifier = client_verifier_from_ca_pem(&pem, true).expect("verifier build");
        assert!(verifier.client_auth_mandatory());
        assert!(verifier.offer_client_auth());
    }

    #[test]
    fn client_verifier_from_ca_pem_round_trips_optional_posture() {
        let pem = synth_cert_pem(Some("Test CA"), &["ca.example.com"]);
        let verifier = client_verifier_from_ca_pem(&pem, false).expect("verifier build");
        // `allow_unauthenticated` flips mandatory off but leaves offer on.
        assert!(!verifier.client_auth_mandatory());
        assert!(verifier.offer_client_auth());
    }

    #[test]
    fn client_verifier_from_ca_pem_rejects_empty_pem() {
        let err = client_verifier_from_ca_pem(b"", true).expect_err("must reject empty");
        assert!(
            matches!(err, TlsResolverError::NoCertificatesFound { .. }),
            "expected NoCertificatesFound, got {err:?}",
        );
    }

    #[test]
    fn client_verifier_from_ca_pem_rejects_malformed_pem() {
        let bogus = b"-----BEGIN CERTIFICATE-----\nnot-base64-data\n-----END CERTIFICATE-----\n";
        let err = client_verifier_from_ca_pem(bogus, true).expect_err("must reject malformed");
        assert!(matches!(err, TlsResolverError::MalformedPem { .. }));
    }

    #[test]
    fn hot_reload_client_verifier_round_trips_initial() {
        let pem = synth_cert_pem(Some("Test CA"), &["ca.example.com"]);
        let v1 = client_verifier_from_ca_pem(&pem, true).expect("v1 build");
        let wrapper = HotReloadClientVerifier::new(v1.clone());
        let snapshot = wrapper.current();
        assert!(snapshot.client_auth_mandatory());
        assert!(wrapper.client_auth_mandatory());
    }

    /// Synthesize a CA + a leaf signed by it. Returns (CA PEM, leaf DER).
    fn synth_ca_and_leaf() -> (Vec<u8>, Vec<u8>) {
        use rcgen::{BasicConstraints, ExtendedKeyUsagePurpose, IsCa, Issuer, KeyUsagePurpose};

        // CA
        let mut ca_params = CertificateParams::new(vec!["Test CA".to_string()]).expect("ca params");
        ca_params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        ca_params.not_after = OffsetDateTime::now_utc() + time::Duration::days(365);
        ca_params
            .distinguished_name
            .push(DnType::CommonName, "Test CA");
        ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        ca_params.key_usages = vec![
            KeyUsagePurpose::KeyCertSign,
            KeyUsagePurpose::DigitalSignature,
            KeyUsagePurpose::CrlSign,
        ];
        let ca_kp = KeyPair::generate().expect("ca kp");
        let ca_cert = ca_params.clone().self_signed(&ca_kp).expect("ca self-sign");

        // Leaf
        let mut leaf_params =
            CertificateParams::new(vec!["leaf.example.com".to_string()]).expect("leaf params");
        leaf_params.not_before = OffsetDateTime::now_utc() - time::Duration::days(1);
        leaf_params.not_after = OffsetDateTime::now_utc() + time::Duration::days(180);
        leaf_params
            .distinguished_name
            .push(DnType::CommonName, "leaf.example.com");
        leaf_params.extended_key_usages = vec![ExtendedKeyUsagePurpose::ClientAuth];
        let leaf_kp = KeyPair::generate().expect("leaf kp");
        let issuer = Issuer::new(ca_params, &ca_kp);
        let leaf_cert = leaf_params.signed_by(&leaf_kp, &issuer).expect("sign leaf");

        (
            ca_cert.pem().into_bytes(),
            leaf_cert.der().as_ref().to_vec(),
        )
    }

    #[test]
    fn client_verifier_accepts_leaf_signed_by_trusted_ca() {
        use rustls::pki_types::{CertificateDer, UnixTime};
        let (ca_pem, leaf_der) = synth_ca_and_leaf();
        let verifier = client_verifier_from_ca_pem(&ca_pem, true).expect("verifier");
        let leaf = CertificateDer::from(leaf_der);
        let now = UnixTime::now();
        verifier
            .verify_client_cert(&leaf, &[], now)
            .expect("trusted CA's leaf must verify");
    }

    #[test]
    fn client_verifier_rejects_leaf_signed_by_untrusted_ca() {
        // Build TWO CAs; verifier trusts CA-A only; present a leaf signed
        // by CA-B. Verifier MUST reject — the canonical "wrong CA"
        // adversarial case the W20β-1 mTLS surface is supposed to
        // defend against.
        use rustls::pki_types::{CertificateDer, UnixTime};
        let (ca_a_pem, _leaf_a_der) = synth_ca_and_leaf();
        let (_ca_b_pem, leaf_b_der) = synth_ca_and_leaf();
        let verifier = client_verifier_from_ca_pem(&ca_a_pem, true).expect("verifier");
        let leaf_b = CertificateDer::from(leaf_b_der);
        let now = UnixTime::now();
        let err = verifier
            .verify_client_cert(&leaf_b, &[], now)
            .expect_err("untrusted CA's leaf MUST reject");
        // The variant text varies by rustls minor; the canonical
        // rejection is `Error::InvalidCertificate(*)`. Match on the
        // outer enum variant only.
        let err_msg = format!("{err}");
        assert!(
            err_msg.contains("invalid")
                || err_msg.contains("Unknown")
                || err_msg.contains("issuer")
                || err_msg.contains("trust"),
            "expected chain-verify rejection text, got: {err_msg}",
        );
    }

    #[test]
    fn hot_reload_client_verifier_swaps_atomically() {
        let pem_a = synth_cert_pem(Some("Test CA A"), &["a.example.com"]);
        let pem_b = synth_cert_pem(Some("Test CA B"), &["b.example.com"]);
        let v_a = client_verifier_from_ca_pem(&pem_a, true).expect("v_a");
        let v_b = client_verifier_from_ca_pem(&pem_b, false).expect("v_b");
        let wrapper = HotReloadClientVerifier::new(v_a);
        assert!(wrapper.client_auth_mandatory(), "v_a was required");
        let _prev = wrapper.reload(v_b);
        assert!(
            !wrapper.client_auth_mandatory(),
            "after reload, v_b (optional) is observed",
        );
    }
}
