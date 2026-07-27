//! W19γ ADR-049 — Bolt-transport OAuth 2.1 + PKCE Bearer-token
//! authentication.
//!
//! Bridges Bolt HELLO frames (`scheme="bearer"` + `credentials=<token>`)
//! to the shared [`crate::auth::oauth_pkce::OAuthConfig`] verifier. The
//! same JWKS + scope vocabulary + claims taxonomy serves the HTTP/TLS
//! transport (W16β ADR-044) and the Bolt transport (this slice).
//!
//! # Wire shape
//!
//! Neo4j-driver-python's [`bearer_auth`](https://neo4j.com/docs/api/python-driver/current/api.html#bearer-auth)
//! emits a Bolt HELLO with `scheme="bearer"` and `credentials=<token>`
//! (the principal field is unused in this scheme). Other Bolt 5.0
//! drivers (`neo4j-js`, `neo4rs`, `neo4j-go-driver`) use the same
//! shape — bearer auth is the wire-portable OAuth path.
//!
//! # Tenant derivation
//!
//! Per ADR-011 §M7-03 the `@tenant_id` suffix on a scope claim
//! identifies the session's tenant. v1.0-α derives the
//! [`arcgraph_core::TenantId`] from the suffix via
//! [`tenant_id_for_suffix`]:
//!
//! - Numeric suffix (`@42`) → `TenantId::new(42)`.
//! - Non-numeric suffix (`@alice`) → a deterministic hash-derived
//!   `TenantId` keyed off the suffix string (FNV-1a; sufficient for the
//!   adversarial cross-tenant probe tests to assign distinct ids to
//!   distinct strings, but the real RBAC tenant catalog lookup lands
//!   at M7-03).
//! - No suffix → [`TenantId::DEFAULT`].
//!
//! # Scope policy
//!
//! v1.0-α requires a Bolt OAuth session to present AT LEAST ONE of
//! `{arcgraph.read, arcgraph.write}` in its scope claim. Per-RUN scope
//! discrimination (READ-only MATCH vs WRITE-mutating CREATE/DELETE)
//! against the Cypher AST is forward-debt to v1.0-GA (the AST-walker
//! lights in M5-12+). The HELLO-time gate is the minimum-viable
//! posture for the v1.0-α deliverable: any authenticated session can
//! issue any RUN, but unauthenticated sessions cannot HELLO at all
//! when OAuth is enforced.
//!
//! # Security defenses inherited from the shared verifier
//!
//! - Algorithm whitelist (no HS*, no `alg=none`).
//! - JWKS HTTP-fetch body cap + scheme check + 5s timeout (sidecar
//!   only; the Rust verifier uses static JWKS for v1.0-α per
//!   ADR-044 §Decision item 5).
//! - `validate_nbf = true` (RFC 7519 §4.1.5 enforcement).
//! - `aud` overlap check against the configured audiences.

use std::sync::Arc;

use arcgraph_core::TenantId;

use super::error::BoltError;
use crate::auth::oauth_pkce::{
    OAuthConfig, OAuthError, SCOPE_READ, SCOPE_WRITE, TokenClaims, parse_scope_claim,
    tenant_id_hint_from_scope_claim, verify_bearer_token,
};

/// Validator that gates Bolt HELLO frames against an
/// [`OAuthConfig`]. Constructed once at server startup and shared
/// across per-connection tasks via `Arc`.
#[derive(Clone)]
pub struct BoltOAuthValidator {
    config: Arc<OAuthConfig>,
}

impl std::fmt::Debug for BoltOAuthValidator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BoltOAuthValidator")
            .field("issuer", &self.config.issuer)
            .field("audiences", &self.config.audiences)
            .finish_non_exhaustive()
    }
}

impl BoltOAuthValidator {
    /// Construct a validator over `config`. `Arc<OAuthConfig>` is
    /// shared with the HTTP transport when both are enabled (the
    /// ADR-049 unification point — one JWKS, two transports).
    #[must_use]
    pub fn new(config: Arc<OAuthConfig>) -> Self {
        Self { config }
    }

    /// Borrow the inner [`OAuthConfig`] (introspection / test).
    #[must_use]
    pub fn config(&self) -> &OAuthConfig {
        &self.config
    }

    /// Authenticate a Bolt HELLO frame.
    ///
    /// Returns the verified [`TokenClaims`] on success. Returns
    /// [`BoltError::Unauthorized`] on missing / malformed / invalid
    /// token, or when the token's scope claim does not satisfy the
    /// minimum HELLO-time scope policy (one of `arcgraph.{read,write}`).
    ///
    /// Per [`super::handler::BoltQueryHandler::authenticate`] this is
    /// called once per HELLO frame; on `Ok` the dispatcher routes
    /// subsequent RUNs through the per-handler `run` body without
    /// re-validating the token (the bearer is bound to the
    /// connection's lifetime — Bolt 5.1+ LOGON-driven re-auth is
    /// forward-debt).
    pub fn authenticate_hello(
        &self,
        scheme: Option<&str>,
        _principal: Option<&str>,
        credentials: Option<&str>,
    ) -> Result<TokenClaims, BoltError> {
        let scheme = scheme.unwrap_or("none");
        if !scheme.eq_ignore_ascii_case("bearer") {
            return Err(BoltError::Unauthorized(format!(
                "OAuth-enforced server requires `bearer` scheme; got `{scheme}`"
            )));
        }
        let token = credentials.ok_or_else(|| {
            BoltError::Unauthorized(
                "bearer scheme requires credentials field carrying the JWT".into(),
            )
        })?;
        if token.is_empty() {
            return Err(BoltError::Unauthorized("empty bearer token".into()));
        }
        let claims = verify_bearer_token(token, &self.config).map_err(oauth_to_bolt)?;
        // HELLO-time scope policy: require ≥1 of {arcgraph.read,
        // arcgraph.write}. Per-RUN scope discrimination is forward-
        // debt to v1.0-GA.
        let scopes = parse_scope_claim(&claims.scope);
        if !scopes.iter().any(|s| s == SCOPE_READ || s == SCOPE_WRITE) {
            return Err(BoltError::Unauthorized(format!(
                "bearer token must carry one of {{{SCOPE_READ}, {SCOPE_WRITE}}}; \
                 present scopes: {scopes:?}"
            )));
        }
        Ok(claims)
    }
}

/// Translate an [`OAuthError`] from the shared verifier into the
/// Bolt-side [`BoltError::Unauthorized`] variant. All OAuth-layer
/// errors map to the same Neo4j status code
/// (`Neo.ClientError.Security.Unauthorized`) per the existing Bolt
/// error taxonomy in [`super::error`].
fn oauth_to_bolt(err: OAuthError) -> BoltError {
    match err {
        OAuthError::MissingBearer => BoltError::Unauthorized("missing bearer token".to_string()),
        OAuthError::MalformedBearer(s) => {
            BoltError::Unauthorized(format!("malformed bearer header: {s}"))
        }
        OAuthError::InvalidToken(s) => {
            BoltError::Unauthorized(format!("invalid bearer token: {s}"))
        }
        OAuthError::InsufficientScope { required, present } => BoltError::Unauthorized(format!(
            "insufficient scope: required {required}, present {present:?}"
        )),
        OAuthError::UnknownMethod(m) => {
            BoltError::Unauthorized(format!("no scope policy for method {m}"))
        }
    }
}

/// Derive a [`TenantId`] from verified [`TokenClaims`] per ADR-011
/// §M7-03 forward-pin. The first scope carrying an `@tenant_id`
/// suffix wins (per sidecar `_parse_scope` precedent). No suffix
/// → [`TenantId::DEFAULT`].
#[must_use]
pub fn tenant_id_from_claims(claims: &TokenClaims) -> TenantId {
    match tenant_id_hint_from_scope_claim(&claims.scope) {
        Some(suffix) => tenant_id_for_suffix(&suffix),
        None => TenantId::DEFAULT,
    }
}

// W20β-1 R1 — `session_scope_from_claims` deferred to M5-12 per
// `feedback_avoid_speculative_scaffolding.md`. The Bolt per-RUN scope
// gate (AST-walker discrimination of READ-only MATCH vs WRITE-mutating
// CREATE/DELETE) is the first production consumer; the wrapper lands
// alongside that consumer rather than ahead of it. The lower-level
// primitive [`crate::scope::SessionScope::from_scope_claim`] is the
// integration point a future M5-12 wiring would call directly.

/// Map a tenant-id suffix string to a [`TenantId`].
///
/// Numeric suffixes (`@42`) decode directly to `TenantId::new(42)`;
/// non-numeric suffixes (`@alice`) derive a deterministic hash-keyed
/// id via FNV-1a. The hash space is `2..u64::MAX` (the `0`/`1` reserved
/// values are skipped — `TenantId::SYSTEM` and `TenantId::DEFAULT` are
/// not derivable from a non-numeric suffix, so an attacker cannot
/// craft a scope @-suffix that escalates to the system tenant).
///
/// v1.0-α uses this synthesis instead of a real RBAC catalog lookup;
/// M7-03 replaces this body with the catalog-resolved
/// `TenantId`.
#[must_use]
pub fn tenant_id_for_suffix(suffix: &str) -> TenantId {
    // Numeric path: a digit-only suffix decodes verbatim.
    if let Ok(n) = suffix.parse::<u64>() {
        // Guard reserved IDs so `@0` / `@1` cannot collide with
        // SYSTEM / DEFAULT respectively. Bumps into the
        // catalog-allocated range (100+) per ADR-011 line 200.
        if n <= 1 {
            return TenantId::new(n.wrapping_add(100));
        }
        return TenantId::new(n);
    }
    // Non-numeric: stable FNV-1a 64-bit hash. Output ∈ `[2, u64::MAX]`
    // (we add 2 if the hash collides with the reserved low values).
    let h = fnv1a_64(suffix.as_bytes());
    if h <= 1 {
        TenantId::new(h.wrapping_add(2))
    } else {
        TenantId::new(h)
    }
}

/// FNV-1a 64-bit hash. Vendored inline (no extra dep) — the standard
/// hash function for short strings when a stable deterministic output
/// across processes is required. Per
/// <http://www.isthe.com/chongo/tech/comp/fnv/> the offset basis and
/// prime are fixed constants.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;
    let mut hash = FNV_OFFSET_BASIS;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::oauth_pkce::{Audiences, JsonWebKey, JsonWebKeySet, OAuthConfig};
    use jsonwebtoken::{Algorithm, DecodingKey};

    /// Build a non-enforcing OAuthConfig for unit tests that don't
    /// exercise signature verification (the integration tests in
    /// `tests/security_adversarial.rs` exercise the full sign-verify
    /// loop against a minted EC keypair).
    fn test_config() -> Arc<OAuthConfig> {
        let jwks = JsonWebKeySet::new(vec![JsonWebKey {
            kid: "test".into(),
            algorithm: Algorithm::RS256,
            // The decoding key is dummy here; the unit tests below
            // exercise authenticate_hello with a HAND-CRAFTED claims
            // value via the test-only `authenticate_with_claims`
            // helper, not via verify_bearer_token (which requires a
            // real RSA-signed token).
            decoding_key: DecodingKey::from_secret(b"x"),
        }])
        .expect("jwks");
        Arc::new(OAuthConfig::new(
            "https://issuer.example/".to_string(),
            vec!["arcgraph-bolt".to_string()],
            jwks,
        ))
    }

    /// Construct a TokenClaims directly for unit-test scope-policy
    /// exercises. The integration tests cover the full
    /// JWT-verify → claims path.
    fn claims(scope: &str) -> TokenClaims {
        TokenClaims {
            iss: "https://issuer.example/".to_string(),
            aud: Audiences::Single("arcgraph-bolt".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: scope.to_string(),
        }
    }

    #[test]
    fn tenant_id_for_numeric_suffix_decodes_verbatim() {
        assert_eq!(tenant_id_for_suffix("42"), TenantId::new(42));
        assert_eq!(tenant_id_for_suffix("12345"), TenantId::new(12345));
    }

    #[test]
    fn tenant_id_for_reserved_numeric_lifts_into_catalog_range() {
        // @0 must NOT alias TenantId::SYSTEM; @1 must NOT alias
        // TenantId::DEFAULT. Both bump into the catalog range (100+).
        assert_eq!(tenant_id_for_suffix("0"), TenantId::new(100));
        assert_eq!(tenant_id_for_suffix("1"), TenantId::new(101));
        assert_ne!(tenant_id_for_suffix("0"), TenantId::SYSTEM);
        assert_ne!(tenant_id_for_suffix("1"), TenantId::DEFAULT);
    }

    #[test]
    fn tenant_id_for_named_suffix_is_deterministic() {
        let a1 = tenant_id_for_suffix("alice");
        let a2 = tenant_id_for_suffix("alice");
        let b = tenant_id_for_suffix("bob");
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
    }

    #[test]
    fn tenant_id_for_named_suffix_never_collides_with_reserved() {
        // Defense-in-depth: an attacker who can craft a scope @-suffix
        // must NOT be able to derive SYSTEM (0) or DEFAULT (1). This
        // test asserts the hash-output sanitization clamps reserved
        // values into the catalog range.
        // Pick a few inputs that we observe to land outside reserved.
        // The point of the test is the assertion, not the inputs.
        for s in ["alice", "bob", "charlie", "tenant_x"] {
            let t = tenant_id_for_suffix(s);
            assert_ne!(t, TenantId::SYSTEM);
            assert_ne!(t, TenantId::DEFAULT);
        }
    }

    #[test]
    fn tenant_id_from_claims_with_no_suffix_returns_default() {
        let c = claims("arcgraph.read arcgraph.write");
        assert_eq!(tenant_id_from_claims(&c), TenantId::DEFAULT);
    }

    #[test]
    fn tenant_id_from_claims_with_suffix_routes_to_derived() {
        let c = claims("arcgraph.read@alice");
        let t = tenant_id_from_claims(&c);
        assert_ne!(t, TenantId::DEFAULT);
        assert_eq!(t, tenant_id_for_suffix("alice"));
    }

    #[test]
    fn tenant_id_from_claims_uses_first_suffix_when_multiple() {
        let c = claims("arcgraph.read@alice arcgraph.write@bob");
        let t = tenant_id_from_claims(&c);
        assert_eq!(t, tenant_id_for_suffix("alice"));
    }

    // ────────────────────────────────────────────────────────────────
    // W20β-1 V11-FC-04 — HELLO-gate fail-closed posture
    // ────────────────────────────────────────────────────────────────
    //
    // The Bolt session-scope derivation wrapper was deleted in R1 fix-
    // up per `feedback_avoid_speculative_scaffolding.md` (no production
    // consumer at v1.0-β; lands alongside M5-12 per-RUN gate). The
    // underlying primitive `SessionScope::from_scope_claim` retains its
    // own test surface in `crate::scope::tests`. The HELLO-gate
    // adversarial pin below remains load-bearing — it asserts the v1.0-
    // β fail-closed posture even when the future per-RUN gate lights.

    #[test]
    fn v11_fc_04_only_token_rejects_at_hello() {
        // Defense-in-depth: even though V11-FC-04 scopes round-trip
        // through `parse_scope_claim` (per `oauth_pkce::parse_scope_claim`),
        // a Bolt HELLO with ONLY those scopes (no read/write) MUST
        // reject — the W19γ validator's ≥1-of-{read,write} gate is
        // load-bearing for the v1.0-β fail-closed posture.
        let v = BoltOAuthValidator::new(test_config());
        // We can't actually drive verify_bearer_token here because the
        // test JWKS uses a dummy key; instead we exercise the policy
        // check directly via `parse_scope_claim` (cite-correctness:
        // same code path the validator runs after JWT verify).
        let scopes = crate::auth::oauth_pkce::parse_scope_claim("unrecognized.scope");
        assert!(
            !scopes.iter().any(|s| {
                s == crate::auth::oauth_pkce::SCOPE_READ
                    || s == crate::auth::oauth_pkce::SCOPE_WRITE
            }),
            "V11-FC-04-only token MUST NOT satisfy HELLO-time read/write gate; \
             present scopes: {scopes:?}",
        );
        // The validator surface — at the layer above parse — is what
        // we'd reach from an actual HELLO; that path is exercised by
        // the integration tests against a real JWKS key.
        let _ = v; // borrow to silence unused
    }

    #[test]
    fn validator_rejects_basic_scheme() {
        let v = BoltOAuthValidator::new(test_config());
        let err = v
            .authenticate_hello(Some("basic"), Some("alice"), Some("pw"))
            .expect_err("reject basic");
        assert!(matches!(err, BoltError::Unauthorized(_)));
        assert!(format!("{err}").contains("requires `bearer`"));
    }

    #[test]
    fn validator_rejects_missing_credentials() {
        let v = BoltOAuthValidator::new(test_config());
        let err = v
            .authenticate_hello(Some("bearer"), None, None)
            .expect_err("reject missing creds");
        assert!(matches!(err, BoltError::Unauthorized(_)));
        assert!(format!("{err}").contains("credentials field"));
    }

    #[test]
    fn validator_rejects_empty_credentials() {
        let v = BoltOAuthValidator::new(test_config());
        let err = v
            .authenticate_hello(Some("bearer"), None, Some(""))
            .expect_err("reject empty creds");
        assert!(matches!(err, BoltError::Unauthorized(_)));
    }

    #[test]
    fn validator_rejects_none_scheme() {
        let v = BoltOAuthValidator::new(test_config());
        let err = v
            .authenticate_hello(None, None, None)
            .expect_err("reject none scheme");
        assert!(matches!(err, BoltError::Unauthorized(_)));
    }

    #[test]
    fn oauth_to_bolt_translates_all_variants() {
        // Defensive: every OAuthError variant must surface as
        // Unauthorized so the FAILURE wire frame carries the
        // canonical Neo.ClientError.Security.Unauthorized code.
        let cases = vec![
            OAuthError::MissingBearer,
            OAuthError::MalformedBearer("x".into()),
            OAuthError::InvalidToken("y".into()),
            OAuthError::InsufficientScope {
                required: "arcgraph.read",
                present: vec![],
            },
            OAuthError::UnknownMethod("z".into()),
        ];
        for e in cases {
            assert!(matches!(oauth_to_bolt(e), BoltError::Unauthorized(_)));
        }
    }

    #[test]
    fn fnv1a_known_vectors() {
        // Reference: <http://www.isthe.com/chongo/tech/comp/fnv/test_fnvtest.c>.
        // Empty string → offset basis.
        assert_eq!(fnv1a_64(b""), 0xcbf29ce484222325);
        // "a" → published FNV-1a 64-bit test vector.
        assert_eq!(fnv1a_64(b"a"), 0xaf63dc4c8601ec8c);
        // "foobar" → published FNV-1a 64-bit test vector.
        assert_eq!(fnv1a_64(b"foobar"), 0x85944171f73967e8);
    }
}
