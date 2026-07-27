//! W13ε M5-02 — TLS hot-reload resolver.
//!
//! This module implements production-grade TLS cert/key rotation
//! without a server restart. It is shared by the HTTP and Bolt
//! transports and supports the operational TLS rotation runbook.
//!
//! ## Components
//!
//! - [`error::TlsResolverError`] — `#[non_exhaustive]` taxonomy of
//!   failure modes across the load → validate → swap pipeline.
//! - [`provider::CertProvider`] — trait abstracting cert sources
//!   (file system at v1.0; ACME / Vault forward-pin for v1.1+).
//!   Borderline trait-shape per the M5-02 spawn prompt + see
//!   `feedback_avoid_speculative_scaffolding.md` §"Trait-shape sub-rule".
//! - [`provider::FileSystemCertProvider`] — v1.0 default impl;
//!   reads PEM cert chain + private key from configured paths and
//!   runs the validation pipeline.
//! - [`validation::build_certified_key`] — `cert + key → Arc<CertifiedKey>`
//!   with all validation gates (key match, validity window, hostname).
//! - [`resolver::HotReloadResolver`] — `rustls::ResolvesServerCert`
//!   impl backed by `ArcSwap<CertifiedKey>` for wait-free reads + atomic
//!   rotation.
//! - [`reload::run_sighup_reload_loop`] — async loop wiring SIGHUP
//!   into the resolver's `reload()` for operator-driven rotation.
//!
//! ## Out of scope (W14+ / v1.1+)
//!
//! - HTTPS / Bolt / gRPC transport wiring — the resolver is the
//!   surface; the transport sub-slices consume it via
//!   `rustls::ServerConfig::builder().with_cert_resolver(resolver)`.
//! - ACME / Let's Encrypt — the [`provider::CertProvider`] trait
//!   is the natural integration point; v1.1+ adds an `AcmeCertProvider`
//!   impl.
//! - Tonic gRPC integration (tracked in issue #289) — tonic 0.12's
//!   `ServerTlsConfig` does NOT expose `ResolvesServerCert`, so ArcGraph
//!   will need a custom `tokio_rustls::TlsAcceptor` when it adds gRPC.

pub mod client_verifier;
pub mod error;
pub mod provider;
pub mod reload;
pub mod resolver;
pub mod validation;

pub use client_verifier::{
    ClientCertIdentity, HotReloadClientVerifier, client_verifier_from_ca_pem,
    parse_client_cert_identity,
};
pub use error::{TlsResolverError, TlsResolverResult};
pub use provider::{CertProvider, FileSystemCertProvider};
pub use reload::{drive_reload, run_sighup_reload_loop};
pub use resolver::HotReloadResolver;
pub use validation::{Clock, SystemClock, build_certified_key};
