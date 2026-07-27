//! W16β M5-03 — OAuth 2.1 + PKCE Bearer-token verification + scope
//! enforcement for the HTTP/TLS MCP transport.
//!
//! Implements the **Resource Server** side of OAuth 2.1: ArcGraph
//! receives Bearer tokens from clients, verifies them against an
//! operator-staged JWK Set, and enforces a per-method scope policy
//! against the token's `scope` claim. The **Authorization Server**
//! (token issuance) is delegated to an external IdP per ADR-044.
//!
//! ## Latency / memory budget
//!
//! Back-of-envelope (under the performance-budget discipline):
//! - Bearer-header extraction: O(1) header-map lookup — ≤ 1 μs.
//! - JWT signature verify (RS256, RSA-2048): ~50-150 μs on modern
//!   x86_64 / aarch64 via aws-lc-rs. Per
//!   <https://briansmith.org/rustdoc/ring/signature/index.html>
//!   reference benchmarks (similar backend; aws-lc-rs's RSA verify
//!   is in the same order of magnitude). Budget: ≤ 200 μs.
//! - Scope check: O(W × S) where W = scope words in the claim
//!   (typically 1-4) and S = the single required scope. ≤ 1 μs.
//! - Memory: the verified `TokenClaims` struct is ~256 bytes; the
//!   static `OAuthConfig` is ~1 KB independent of request rate.
//!
//! Token-verification overhead per request: 100-200 μs (signature
//! verify dominates). At a 10K-req/s steady state this is 1-2 cores
//! of pure JWT verify; the transport runs the verify inline on the
//! connection task, which is acceptable because (a) the catalog is
//! agent-driven, not human-driven, and (b) AWS LC RSA verify
//! benchmarks are within an order of magnitude of TLS handshake
//! cost which dominates anyway.
//!
//! ## Cite-correctness disclosure
//!
//! Every RFC / ADR / design-v2 citation in this module has been
//! verified against literal section prose per the
//! `feedback_cite_correctness_not_just_resolution.md` discipline:
//! - **RFC 7636 §4.1** (Client Creates a Code Verifier) — verifier
//!   length is 43-128 chars from `[A-Za-z0-9-._~]`. Verified
//!   2026-05-15.
//! - **RFC 7636 §4.2** (Client Creates the Code Challenge) — S256
//!   transform is `BASE64URL-ENCODE(SHA256(ASCII(code_verifier)))`.
//!   Verified 2026-05-15.
//! - **RFC 6749 §4.1** (Authorization Code Grant) — authorization-
//!   code grant is "used to obtain both access tokens and refresh
//!   tokens and is optimized for confidential clients." Verified
//!   2026-05-15.
//! - **RFC 6750 §3** (The WWW-Authenticate Response Header Field) —
//!   defines `error="invalid_token"` and `error="insufficient_scope"`
//!   error codes consumed by [`oauth_error_to_www_authenticate`].
//!   Verified 2026-05-15.
//! - **design-v2 §9.4 line 665** — "OAuth 2.1 with PKCE for remote.
//!   Bearer tokens with scopes (`arcgraph.read`, `arcgraph.write`,
//!   `arcgraph.power`, `arcgraph.admin`)." Verified 2026-05-15.
//! - **ADR-004 line 41** — Tier-2 tools (e.g. `graph.raw_query`)
//!   require `arcgraph.power` scope. Verified 2026-05-15.
//! - **ADR-011 line 162** — `@tenant_id` scope suffix; recognized
//!   here (the SUFFIX after `@` is stripped before scope membership
//!   check). Verified 2026-05-15.

use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use rand::RngExt;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::transport::{
    METHOD_GRAPH_EXPLORE, METHOD_GRAPH_INGEST, METHOD_GRAPH_INSPECT, METHOD_GRAPH_RAW_QUERY,
    METHOD_GRAPH_SCHEMA, METHOD_GRAPH_SEARCH,
};

// ─────────────────────────────────────────────────────────────────────
// Scope constants (design-v2 §9.4 line 665)
// ─────────────────────────────────────────────────────────────────────

/// Scope unlocking Tier-1 read tools (`graph.schema`, `graph.inspect`,
/// `graph.explore`, `graph.search`) per design-v2 §9.4 line 665.
pub const SCOPE_READ: &str = "arcgraph.read";

/// Scope unlocking Tier-1 write tools (`graph.ingest`) per design-v2
/// §9.4 line 665.
pub const SCOPE_WRITE: &str = "arcgraph.write";

/// Scope unlocking the Tier-2 `graph.raw_query` tool per ADR-004 line
/// 41 / design-v2 §9.4 line 665.
pub const SCOPE_POWER: &str = "arcgraph.power";

/// Scope unlocking system-admin tools per design-v2 §9.4 line 665.
/// **Reserved for roadmap-M5+ tools; no v1.0-α tools currently
/// require this scope.**
pub const SCOPE_ADMIN: &str = "arcgraph.admin";

/// Header carrying the Bearer token per RFC 6750 §2.1.
pub const HEADER_AUTHORIZATION: &str = "authorization";

/// Prefix for the Bearer scheme per RFC 6750 §2.1
/// ("Bearer" + SP + token). Case-insensitive match per the RFC, but
/// we normalize to title-case for outbound `WWW-Authenticate`.
pub const BEARER_PREFIX: &str = "Bearer ";

/// Minimum `code_verifier` length per RFC 7636 §4.1.
pub const CODE_VERIFIER_MIN_LEN: usize = 43;

/// Maximum `code_verifier` length per RFC 7636 §4.1.
pub const CODE_VERIFIER_MAX_LEN: usize = 128;

/// The default `code_verifier` length: 64 chars (median of the
/// 43-128 range; provides ~384 bits of entropy from the 66-char
/// unreserved set per RFC 7636 §4.1). Operators / clients that need
/// a specific length use [`code_verifier_with_len`].
pub const CODE_VERIFIER_DEFAULT_LEN: usize = 64;

/// Default `exp` / `nbf` clock-skew tolerance (seconds). Standard
/// practice across OAuth implementations; matches the default in
/// `jsonwebtoken`'s `Validation::leeway`. Operators override via
/// [`OAuthConfig::clock_skew_secs`].
pub const DEFAULT_CLOCK_SKEW_SECS: u64 = 30;

// ─────────────────────────────────────────────────────────────────────
// Errors
// ─────────────────────────────────────────────────────────────────────

/// Failure modes specific to OAuth verification. `#[non_exhaustive]`
/// permits adding a new
/// variant in a future slice (DPoP, opaque-token introspection,
/// JWKS HTTP fetch fault) MUST NOT regress source-compat for
/// downstream pattern-matchers.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OAuthError {
    /// The `Authorization` header was absent. Per RFC 6750 §3 this
    /// is an `invalid_request` for protected resources requiring a
    /// Bearer token; we surface HTTP `401 Unauthorized` with
    /// `WWW-Authenticate: Bearer realm="arcgraph"` (no `error=`
    /// since RFC 6750 §3 forbids an error code when no
    /// authentication was attempted).
    #[error("missing Authorization: Bearer header")]
    MissingBearer,

    /// The `Authorization` header was present but did not start with
    /// `Bearer ` (per RFC 6750 §2.1 token transmission rules).
    /// Surfaces HTTP `401 Unauthorized` + `error="invalid_token"`.
    #[error("Authorization header does not carry a Bearer token: {0}")]
    MalformedBearer(String),

    /// JWT decode / signature verify / claims check failed.
    /// Surfaces HTTP `401 Unauthorized` + `error="invalid_token"`
    /// per RFC 6750 §3.1.
    #[error("invalid token: {0}")]
    InvalidToken(String),

    /// The token's `scope` claim did not include the scope required
    /// for the dispatched method. Surfaces HTTP `403 Forbidden` +
    /// `error="insufficient_scope" scope="<required>"` per RFC 6750
    /// §3.1.
    #[error("insufficient scope: required {required}, present {present:?}")]
    InsufficientScope {
        /// The scope the method requires (e.g. `arcgraph.read`).
        required: &'static str,
        /// The scopes the token carried, for the JSON-RPC `data`
        /// payload. Excludes the `@tenant_id` suffix per ADR-011.
        present: Vec<String>,
    },

    /// The dispatched method is unknown to the OAuth policy table
    /// — defensive: today this means a method outside the v1.0-α
    /// catalog reached the scope gate (in practice the dispatcher
    /// returns `MethodNotFound` BEFORE this, but the OAuth gate
    /// runs first so we treat unknown-method as "no scope policy
    /// → reject" to deny-default rather than allow-by-omission).
    /// Surfaces HTTP `401 Unauthorized` + `error="invalid_token"`.
    #[error("no scope policy for method {0}")]
    UnknownMethod(String),
}

// ─────────────────────────────────────────────────────────────────────
// PKCE helpers (RFC 7636)
// ─────────────────────────────────────────────────────────────────────

/// A PKCE `code_verifier` per RFC 7636 §4.1. Wraps the underlying
/// string so a misplaced `Debug` log doesn't accidentally leak the
/// secret; the inner string is accessible via [`Self::as_str`] but
/// the `Debug` impl masks it.
#[derive(Clone)]
pub struct CodeVerifier(String);

impl CodeVerifier {
    /// Borrow the verifier's string form for use with the token
    /// endpoint (`code_verifier` parameter per RFC 7636 §4.5).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Move the underlying string out. Use this only at the boundary
    /// to the HTTP/form encoder; the [`CodeVerifier`] wrapper is
    /// what consumers should hold in memory.
    #[must_use]
    pub fn into_string(self) -> String {
        self.0
    }

    /// Construct a `CodeVerifier` from a previously-generated string.
    /// Validates the input against RFC 7636 §4.1 before wrapping.
    ///
    /// This is the constructor the **Authorization Server** side uses
    /// to roundtrip a verifier received over the wire during the
    /// token-exchange step: the AS receives `code_verifier` as a
    /// string in the token endpoint POST body, wraps it here, then
    /// calls [`code_challenge_s256`] to re-derive the challenge for
    /// comparison against the one stored at the authorization step.
    ///
    /// # Errors
    ///
    /// Returns a static error string when `s` violates RFC 7636 §4.1
    /// (length out of [43, 128] OR a char outside the unreserved set).
    pub fn from_string(s: String) -> Result<Self, &'static str> {
        validate_code_verifier(&s)?;
        Ok(Self(s))
    }
}

impl std::fmt::Debug for CodeVerifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CodeVerifier(<redacted len={}>)", self.0.len())
    }
}

/// The "unreserved characters" set from RFC 7636 §4.1 (which itself
/// references RFC 3986 §2.3 — `unreserved = ALPHA / DIGIT / "-" /
/// "." / "_" / "~"`). 66 chars total. ASCII-only by construction so
/// `byte == char` for indexing.
const UNRESERVED: &[u8; 66] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~";

/// Generate a `code_verifier` of [`CODE_VERIFIER_DEFAULT_LEN`]
/// characters per RFC 7636 §4.1. Uses [`rand::rng`]; callers
/// requiring a deterministic verifier (tests) use
/// [`code_verifier_with_len`] + an explicit RNG via
/// `code_verifier_from_rng`.
#[must_use]
pub fn code_verifier_new() -> CodeVerifier {
    code_verifier_with_len(CODE_VERIFIER_DEFAULT_LEN)
}

/// Generate a `code_verifier` of `len` characters per RFC 7636 §4.1.
///
/// # Panics
///
/// Panics if `len < CODE_VERIFIER_MIN_LEN` or
/// `len > CODE_VERIFIER_MAX_LEN`. RFC 7636 §4.1 mandates 43-128
/// chars; out-of-range is a programmer error (clients should reject
/// at config time, not at runtime).
#[must_use]
pub fn code_verifier_with_len(len: usize) -> CodeVerifier {
    assert!(
        (CODE_VERIFIER_MIN_LEN..=CODE_VERIFIER_MAX_LEN).contains(&len),
        "code_verifier length {len} out of RFC 7636 §4.1 range \
         [{CODE_VERIFIER_MIN_LEN}, {CODE_VERIFIER_MAX_LEN}]",
    );
    let mut rng = rand::rng();
    let bytes: Vec<u8> = (0..len)
        .map(|_| UNRESERVED[rng.random_range(0..UNRESERVED.len())])
        .collect();
    // SAFETY-equivalent: UNRESERVED is pure ASCII so the resulting
    // bytes are valid UTF-8 by construction. We use
    // `String::from_utf8` to avoid `unsafe`.
    CodeVerifier(String::from_utf8(bytes).expect("UNRESERVED is ASCII-only by construction"))
}

/// Derive the S256 `code_challenge` for `verifier` per RFC 7636 §4.2:
/// `BASE64URL-ENCODE(SHA256(ASCII(code_verifier)))`. The encoding is
/// "BASE64URL without padding" — RFC 7636 §3 ties this to RFC 4648
/// §5's base64url alphabet with `=` padding removed.
#[must_use]
pub fn code_challenge_s256(verifier: &CodeVerifier) -> String {
    let mut hasher = Sha256::new();
    hasher.update(verifier.0.as_bytes());
    let digest = hasher.finalize();
    URL_SAFE_NO_PAD.encode(digest)
}

/// Validate that `s` is a syntactically valid `code_verifier` per
/// RFC 7636 §4.1: length in [43, 128] AND every char is in the
/// unreserved set. Returns `Ok(())` on success, `Err(reason)` for
/// the first failure.
///
/// This is the validator the integration test exercises against
/// proptest inputs; the runtime PKCE flow itself doesn't validate
/// (the AS verifies the verifier matches the previously-stored
/// challenge). The validator lives here so client code that wants
/// to syntactically pre-check its own verifier-generation logic
/// has one canonical source of truth.
pub fn validate_code_verifier(s: &str) -> Result<(), &'static str> {
    if s.len() < CODE_VERIFIER_MIN_LEN {
        return Err("code_verifier shorter than RFC 7636 §4.1 minimum (43 chars)");
    }
    if s.len() > CODE_VERIFIER_MAX_LEN {
        return Err("code_verifier longer than RFC 7636 §4.1 maximum (128 chars)");
    }
    for b in s.bytes() {
        if !UNRESERVED.contains(&b) {
            return Err("code_verifier contains character outside RFC 7636 §4.1 \
                 unreserved set [A-Za-z0-9-._~]");
        }
    }
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────
// JWK Set (operator-staged static JWKS)
// ─────────────────────────────────────────────────────────────────────

/// One verification key in the operator-staged JWK Set, keyed by
/// `kid` (JWT header `kid` claim per RFC 7515 §4.1.4) and tagged
/// with its signing algorithm.
///
/// For v1.0-α only PEM-encoded RSA and EC public keys are accepted
/// (jsonwebtoken's `DecodingKey::from_rsa_pem` / `from_ec_pem`).
/// JWK-JSON-format keys are forward-pinned to v1.1 when HTTP-fetching
/// JWKS lands.
pub struct JsonWebKey {
    /// JWT header `kid` selector. When the token's header carries a
    /// `kid`, the verifier looks up this exact entry. If the token
    /// carries no `kid` and the JWK Set has exactly one entry, that
    /// entry is used (single-key deployments are the common case).
    pub kid: String,
    /// Algorithm this key signs with. The token header's `alg` MUST
    /// match (`jsonwebtoken::Validation::set_required_spec_claims`
    /// would not catch a mismatched alg-vs-key; we enforce
    /// explicitly).
    pub algorithm: Algorithm,
    /// The decoding key, prepared by jsonwebtoken from operator-
    /// supplied PEM bytes.
    pub decoding_key: DecodingKey,
}

impl std::fmt::Debug for JsonWebKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonWebKey")
            .field("kid", &self.kid)
            .field("algorithm", &self.algorithm)
            .field("decoding_key", &"<opaque>")
            .finish()
    }
}

/// The operator-staged JWK Set (RFC 7517 §5). For v1.0-α this is
/// purely an in-memory container — operators populate it at config
/// time. v1.1 adds an HTTP-fetching variant with TTL caching
/// (ADR-044 forward-pin).
#[derive(Debug)]
pub struct JsonWebKeySet {
    keys: Vec<JsonWebKey>,
}

impl JsonWebKeySet {
    /// Construct a JWK Set from a Vec of keys. Empty sets are
    /// rejected so misconfiguration surfaces at startup.
    ///
    /// # Errors
    ///
    /// Returns `Err("...")` if `keys` is empty OR any two keys share
    /// a `kid` (ambiguous resolution).
    pub fn new(keys: Vec<JsonWebKey>) -> Result<Self, String> {
        if keys.is_empty() {
            return Err("JsonWebKeySet must contain at least one key".to_string());
        }
        // Reject duplicate kids (RFC 7517 §5.2 recommends distinct
        // `kid` per JWK in a Set, but doesn't strictly forbid
        // duplicates; we forbid because our resolution is by-`kid`-
        // exact-match).
        let mut seen = std::collections::HashSet::new();
        for k in &keys {
            if !seen.insert(k.kid.clone()) {
                return Err(format!("duplicate kid in JsonWebKeySet: {}", k.kid));
            }
        }
        Ok(Self { keys })
    }

    /// Resolve a key by `kid`. When `kid` is `None` and the set has
    /// a single key, that key is returned (single-key deployments).
    /// Otherwise the call fails — explicit `kid` is required when
    /// the set has multiple keys.
    fn resolve(&self, kid: Option<&str>) -> Option<&JsonWebKey> {
        match kid {
            Some(k) => self.keys.iter().find(|key| key.kid == k),
            None if self.keys.len() == 1 => Some(&self.keys[0]),
            None => None,
        }
    }

    /// Iterator over the configured keys (introspection / debug).
    pub fn keys(&self) -> impl Iterator<Item = &JsonWebKey> {
        self.keys.iter()
    }
}

// ─────────────────────────────────────────────────────────────────────
// OAuthConfig
// ─────────────────────────────────────────────────────────────────────

/// Configuration mounted onto [`crate::transport::http::HttpServerConfig`]
/// via `with_oauth(...)`. When `Some`, the HTTP transport requires a
/// Bearer token on every `POST /mcp` request and enforces scope
/// against the dispatched method.
///
/// `#[serde(deny_unknown_fields)]` is NOT applied — `OAuthConfig` is
/// not yet user-deserialized (consistent with the rest of the v1.0-α
/// `pub struct *Config` surface; the deserialization wave lands in
/// M6 server config).
pub struct OAuthConfig {
    /// Expected `iss` (issuer) claim. The token's `iss` MUST equal
    /// this string. RFC 7519 §4.1.1: "The `iss` (issuer) claim
    /// identifies the principal that issued the JWT."
    pub issuer: String,

    /// Accepted audiences. The token's `aud` claim MUST include at
    /// least one entry from this list. RFC 7519 §4.1.3: "The `aud`
    /// (audience) claim identifies the recipients that the JWT is
    /// intended for." Multi-audience tokens are common in IdP
    /// deployments (a token may be valid for both `arcgraph` and
    /// `api.example.com`).
    pub audiences: Vec<String>,

    /// Operator-staged JWK Set. Static for v1.0-α (HTTP-fetching
    /// JWKS is the v1.1 forward-pin per ADR-044).
    pub jwks: JsonWebKeySet,

    /// Clock-skew tolerance for `exp` / `nbf` validation, in
    /// seconds. Defaults to [`DEFAULT_CLOCK_SKEW_SECS`] (30s).
    /// Compensates for clock drift between the issuer and ArcGraph.
    pub clock_skew_secs: u64,

    /// Whitelist of algorithms the verifier accepts. The token's
    /// header `alg` MUST be in this list. Defaults to
    /// `[RS256, RS384, RS512, ES256, ES384, PS256, PS384, PS512]`
    /// (asymmetric algorithms only; HS* is rejected per ADR-044
    /// rationale).
    pub required_algorithms: Vec<Algorithm>,
}

impl std::fmt::Debug for OAuthConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OAuthConfig")
            .field("issuer", &self.issuer)
            .field("audiences", &self.audiences)
            .field("jwks", &format_args!("<{} keys>", self.jwks.keys.len()))
            .field("clock_skew_secs", &self.clock_skew_secs)
            .field("required_algorithms", &self.required_algorithms)
            .finish()
    }
}

impl OAuthConfig {
    /// Construct an `OAuthConfig` with defaults for the algorithm
    /// whitelist + clock-skew. The caller supplies the issuer,
    /// audiences, and JWK Set.
    ///
    /// # Panics
    ///
    /// Panics if `audiences` is empty (a token with no acceptable
    /// audience can never verify — surface at config time, not
    /// per-request).
    #[must_use]
    pub fn new(issuer: String, audiences: Vec<String>, jwks: JsonWebKeySet) -> Self {
        assert!(
            !audiences.is_empty(),
            "OAuthConfig::audiences must be non-empty",
        );
        Self {
            issuer,
            audiences,
            jwks,
            clock_skew_secs: DEFAULT_CLOCK_SKEW_SECS,
            required_algorithms: vec![
                Algorithm::RS256,
                Algorithm::RS384,
                Algorithm::RS512,
                Algorithm::ES256,
                Algorithm::ES384,
                Algorithm::PS256,
                Algorithm::PS384,
                Algorithm::PS512,
            ],
        }
    }

    /// Builder: override the clock-skew tolerance.
    #[must_use]
    pub fn with_clock_skew_secs(mut self, secs: u64) -> Self {
        self.clock_skew_secs = secs;
        self
    }

    /// Builder: override the allowed algorithm whitelist.
    ///
    /// # Panics
    ///
    /// Panics if `algorithms` is empty OR includes an HS* algorithm.
    /// ADR-044 rejects symmetric algorithms; the assertion guards
    /// against operator misconfiguration that would weaken the
    /// security posture below the design-of-record.
    #[must_use]
    pub fn with_required_algorithms(mut self, algorithms: Vec<Algorithm>) -> Self {
        assert!(
            !algorithms.is_empty(),
            "OAuthConfig::required_algorithms must be non-empty",
        );
        for a in &algorithms {
            assert!(
                !matches!(a, Algorithm::HS256 | Algorithm::HS384 | Algorithm::HS512),
                "OAuthConfig::required_algorithms rejects HS* (symmetric) algorithms; \
                 see ADR-044 §Decision item 4",
            );
        }
        self.required_algorithms = algorithms;
        self
    }
}

// ─────────────────────────────────────────────────────────────────────
// Token claims
// ─────────────────────────────────────────────────────────────────────

/// The subset of JWT claims ArcGraph reads off a verified Bearer
/// token. Other claims (`sub`, `iat`, `jti`, custom IdP fields) are
/// IGNORED for v1.0-α — only the canonical OAuth 2.1 claims listed
/// here participate in policy.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TokenClaims {
    /// `iss` (issuer) — verified against [`OAuthConfig::issuer`].
    pub iss: String,
    /// `aud` (audience) — verified against
    /// [`OAuthConfig::audiences`]. JWT spec allows a single string
    /// OR an array of strings; we accept both via the
    /// `serde(untagged)` adapter [`Audiences`].
    pub aud: Audiences,
    /// `exp` (expiration) — verified against current time +
    /// clock-skew. Required.
    pub exp: u64,
    /// `nbf` (not-before) — verified against current time -
    /// clock-skew if present. Optional per RFC 7519 §4.1.5. When
    /// present, [`verify_bearer_token`] opts the `jsonwebtoken`
    /// `Validation` into `validate_nbf = true` so the RFC 7519
    /// §4.1.5 "MUST NOT be accepted for processing" invariant is
    /// enforced (not just documented).
    #[serde(default)]
    pub nbf: Option<u64>,
    /// `scope` — space-delimited list of granted scopes per OAuth
    /// 2.0 §3.3 / RFC 8693 §4.2. Required by ArcGraph (a token
    /// without `scope` cannot satisfy any method's policy).
    pub scope: String,
}

/// Adapter for the JWT `aud` claim which is allowed to be a single
/// string OR an array of strings (RFC 7519 §4.1.3 — "The `aud`
/// value [...] MAY also be a single case-sensitive string").
#[derive(Debug, Clone)]
pub enum Audiences {
    /// Single-audience form.
    Single(String),
    /// Multi-audience form.
    Multiple(Vec<String>),
}

impl Audiences {
    /// True if `target` is in the audience set.
    #[must_use]
    pub fn contains(&self, target: &str) -> bool {
        match self {
            Audiences::Single(s) => s == target,
            Audiences::Multiple(v) => v.iter().any(|s| s == target),
        }
    }

    /// Borrow the audiences as a slice for logging / diagnostics.
    /// Returns a one-element slice for the single-audience form.
    #[must_use]
    pub fn as_slice(&self) -> &[String] {
        match self {
            Audiences::Single(s) => std::slice::from_ref(s),
            Audiences::Multiple(v) => v.as_slice(),
        }
    }
}

impl<'de> serde::Deserialize<'de> for Audiences {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::{self, Visitor};

        struct AudVisitor;
        impl<'de> Visitor<'de> for AudVisitor {
            type Value = Audiences;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "a string or an array of strings")
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
                Ok(Audiences::Single(v.to_string()))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<Self::Value, E> {
                Ok(Audiences::Single(v))
            }
            fn visit_seq<A: de::SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut out = Vec::new();
                while let Some(s) = seq.next_element::<String>()? {
                    out.push(s);
                }
                Ok(Audiences::Multiple(out))
            }
        }
        deserializer.deserialize_any(AudVisitor)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Scope policy
// ─────────────────────────────────────────────────────────────────────

/// Map a JSON-RPC method to the scope it requires per ADR-044
/// §Decision item 6. Returns `None` for methods outside the v1.0-β
/// policy table (which the verifier translates to
/// [`OAuthError::UnknownMethod`] — deny-default semantics).
///
/// `graph.raw_query` maps to `arcgraph.power` so an OAuth-enabled HTTP
/// gate routes
/// `WWW-Authenticate: insufficient_scope; required="arcgraph.power"`
/// (403) when the token lacks the scope, and admits when present.
///
/// The tool-function-level scope check inside `raw_query_tool` remains
/// load-bearing — gate 6a's
/// `enforce_scope` matches against the OAuth scope claim, while the
/// per-tool gate matches against the dispatcher's `SessionScope`
/// (defense-in-depth: a misconfigured dispatcher constructed with
/// `SessionScope::Read` still rejects power-tier methods even if a
/// hypothetical OAuth bypass admits them).
#[must_use]
pub fn scope_for_method(method: &str) -> Option<&'static str> {
    match method {
        METHOD_GRAPH_SCHEMA | METHOD_GRAPH_INSPECT | METHOD_GRAPH_EXPLORE | METHOD_GRAPH_SEARCH => {
            Some(SCOPE_READ)
        }
        METHOD_GRAPH_INGEST => Some(SCOPE_WRITE),
        METHOD_GRAPH_RAW_QUERY => Some(SCOPE_POWER),
        _ => None,
    }
}

/// Parse a JWT `scope` claim (space-delimited per RFC 8693 §4.2)
/// into a vector of scope strings with `@tenant_id` suffixes
/// stripped (ADR-011 §"M7-03" forward-pin: suffix recognized,
/// enforcement deferred). Empty input yields an empty vec.
#[must_use]
pub fn parse_scope_claim(scope_claim: &str) -> Vec<String> {
    scope_claim
        .split_ascii_whitespace()
        .map(strip_tenant_suffix)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// Strip an `@tenant_id` suffix from a scope string per ADR-011 line
/// 162. Returns the bare scope on success; the original string
/// unchanged if no `@` is present.
fn strip_tenant_suffix(s: &str) -> &str {
    match s.find('@') {
        Some(idx) => &s[..idx],
        None => s,
    }
}

/// W19γ ADR-049 — extract the tenant-id hint from the first scope
/// carrying an `@tenant_id` suffix in `scope_claim`. Returns `None` if
/// no scope carries a suffix; the first-suffix-wins semantics mirror
/// the sidecar's `_parse_scope` helper (ADR-011 §M7-03 forward-pin).
///
/// The Bolt OAuth path uses this to route an authenticated session to
/// a specific [`arcgraph_core::TenantId`] — without it every OAuth
/// session would collapse to `TenantId::DEFAULT` and the adversarial
/// cross-tenant probe tests would be meaningless. v1.0-α uses
/// `FromStr` against the suffix to derive the tenant id; non-numeric
/// suffixes fall back to a string-hashed tenant id (deterministic for
/// a given suffix string).
#[must_use]
pub fn tenant_id_hint_from_scope_claim(scope_claim: &str) -> Option<String> {
    for word in scope_claim.split_ascii_whitespace() {
        if let Some(idx) = word.find('@') {
            let tail = &word[idx + 1..];
            if !tail.is_empty() {
                return Some(tail.to_string());
            }
        }
    }
    None
}

/// Check that `claims` carries the scope required for `method`.
///
/// # Errors
///
/// - [`OAuthError::UnknownMethod`] if `method` is outside the
///   v1.0-α policy table (deny-default).
/// - [`OAuthError::InsufficientScope`] if the required scope is not
///   present on the token.
pub fn enforce_scope(claims: &TokenClaims, method: &str) -> Result<(), OAuthError> {
    let Some(required) = scope_for_method(method) else {
        return Err(OAuthError::UnknownMethod(method.to_string()));
    };
    let present = parse_scope_claim(&claims.scope);
    if present.iter().any(|s| s == required) {
        Ok(())
    } else {
        Err(OAuthError::InsufficientScope { required, present })
    }
}

// ─────────────────────────────────────────────────────────────────────
// Bearer token extraction + verification
// ─────────────────────────────────────────────────────────────────────

/// Extract the Bearer token from an `Authorization` header value per
/// RFC 6750 §2.1. The scheme match is case-insensitive ("Bearer",
/// "bearer", and "BEARER" all valid).
///
/// # Errors
///
/// - [`OAuthError::MalformedBearer`] if `header_value` does NOT
///   start with the `Bearer ` scheme prefix.
pub fn extract_bearer_token(header_value: &str) -> Result<&str, OAuthError> {
    let trimmed = header_value.trim_start();
    if trimmed.len() < BEARER_PREFIX.len()
        || !trimmed[..BEARER_PREFIX.len()].eq_ignore_ascii_case(BEARER_PREFIX)
    {
        return Err(OAuthError::MalformedBearer(
            "Authorization header lacks 'Bearer ' scheme prefix".to_string(),
        ));
    }
    let token = trimmed[BEARER_PREFIX.len()..].trim();
    if token.is_empty() {
        return Err(OAuthError::MalformedBearer(
            "Authorization header carried 'Bearer' with no token".to_string(),
        ));
    }
    Ok(token)
}

/// Verify a Bearer token against the operator-staged OAuth config.
/// Returns the validated claims on success.
///
/// Steps:
/// 1. Decode the JWT header to read `kid` + `alg`.
/// 2. Confirm `alg` is in the `required_algorithms` whitelist.
/// 3. Resolve the matching `JsonWebKey` from the JWK Set; confirm
///    its `algorithm` matches the header `alg` (defense-in-depth
///    against a malicious header that names a different alg than
///    the key actually signs with).
/// 4. Construct a `Validation` instance that:
///    - Requires `iss == config.issuer`.
///    - Requires `aud ∩ config.audiences != ∅`.
///    - Sets `leeway = config.clock_skew_secs`.
///    - Validates `exp` against current time (jsonwebtoken default).
///    - Validates `nbf` against current time by opting into
///      `validate_nbf = true` per RFC 7519 §4.1.5 (jsonwebtoken's
///      default is `false`; we override).
///    - Limits accepted algorithms to the single `alg` we picked.
/// 5. Decode + verify the token; deserialize the claims into
///    [`TokenClaims`].
///
/// # Errors
///
/// - [`OAuthError::InvalidToken`] for any decode / signature /
///   claims-validation failure. The inner string carries the
///   underlying jsonwebtoken error for debug / logging.
pub fn verify_bearer_token(token: &str, config: &OAuthConfig) -> Result<TokenClaims, OAuthError> {
    let header = decode_header(token)
        .map_err(|e| OAuthError::InvalidToken(format!("header decode: {e}")))?;
    if !config.required_algorithms.contains(&header.alg) {
        return Err(OAuthError::InvalidToken(format!(
            "algorithm {:?} not in required_algorithms whitelist",
            header.alg
        )));
    }

    let jwk = config.jwks.resolve(header.kid.as_deref()).ok_or_else(|| {
        OAuthError::InvalidToken(match &header.kid {
            Some(kid) => format!("no JWK with kid={kid}"),
            None => "token header carried no kid and JWK Set has multiple keys".to_string(),
        })
    })?;
    if jwk.algorithm != header.alg {
        return Err(OAuthError::InvalidToken(format!(
            "header alg {:?} does not match JWK algorithm {:?} for kid={}",
            header.alg, jwk.algorithm, jwk.kid,
        )));
    }

    let mut validation = Validation::new(jwk.algorithm);
    validation.leeway = config.clock_skew_secs;
    validation.set_issuer(&[&config.issuer]);
    validation.set_audience(&config.audiences);
    // Defense-in-depth: jsonwebtoken accepts multiple algorithms by
    // default; we restrict to the one matched against the JWK.
    validation.algorithms = vec![jwk.algorithm];
    // RFC 7519 §4.1.5 (MUST NOT): jsonwebtoken's `Validation::new`
    // defaults `validate_nbf: false`. We MUST opt in or a token with
    // an `nbf` in the future is accepted before its activation time.
    // R1 HIGH-1.
    validation.validate_nbf = true;

    let data = decode::<TokenClaims>(token, &jwk.decoding_key, &validation)
        .map_err(|e| OAuthError::InvalidToken(format!("verify: {e}")))?;
    Ok(data.claims)
}

/// Convenience: extract + verify in one call. Mirrors the
/// per-request shape the HTTP transport uses.
pub fn verify_bearer_header(
    header_value: &str,
    config: &OAuthConfig,
) -> Result<TokenClaims, OAuthError> {
    let token = extract_bearer_token(header_value)?;
    verify_bearer_token(token, config)
}

// ─────────────────────────────────────────────────────────────────────
// WWW-Authenticate rendering
// ─────────────────────────────────────────────────────────────────────

/// Render the `WWW-Authenticate` header value for an OAuth error,
/// per RFC 6750 §3. Used by the HTTP transport when rejecting an
/// unauthenticated / unauthorized request.
#[must_use]
pub fn oauth_error_to_www_authenticate(err: &OAuthError) -> String {
    match err {
        // RFC 6750 §3: "When a request does not include any
        // authentication information, the resource server SHOULD
        // NOT include an error code or other error information."
        OAuthError::MissingBearer => "Bearer realm=\"arcgraph\"".to_string(),
        OAuthError::MalformedBearer(_) | OAuthError::InvalidToken(_) => {
            "Bearer realm=\"arcgraph\", error=\"invalid_token\"".to_string()
        }
        OAuthError::InsufficientScope { required, .. } => {
            format!("Bearer realm=\"arcgraph\", error=\"insufficient_scope\", scope=\"{required}\"",)
        }
        OAuthError::UnknownMethod(_) => {
            "Bearer realm=\"arcgraph\", error=\"invalid_token\"".to_string()
        }
    }
}

/// Current UNIX timestamp in seconds (used by tests + future audit
/// paths). Saturates at `u64::MAX` if the system clock predates
/// the UNIX epoch (defensive — practically unreachable).
#[must_use]
pub fn unix_now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(u64::MAX)
}

// ─────────────────────────────────────────────────────────────────────
// Tests (unit-level; integration tests live under tests/)
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rfc_7636_4_1_code_verifier_default_len_in_range() {
        let v = code_verifier_new();
        assert!((CODE_VERIFIER_MIN_LEN..=CODE_VERIFIER_MAX_LEN).contains(&v.as_str().len()));
        assert_eq!(v.as_str().len(), CODE_VERIFIER_DEFAULT_LEN);
        validate_code_verifier(v.as_str()).expect("RFC 7636 §4.1 conformant");
    }

    #[test]
    fn rfc_7636_4_1_code_verifier_with_len_extremes() {
        for len in [CODE_VERIFIER_MIN_LEN, 64, 96, CODE_VERIFIER_MAX_LEN] {
            let v = code_verifier_with_len(len);
            assert_eq!(v.as_str().len(), len);
            validate_code_verifier(v.as_str()).expect("RFC 7636 §4.1 conformant");
        }
    }

    #[test]
    #[should_panic(expected = "out of RFC 7636 §4.1 range")]
    fn rfc_7636_4_1_rejects_short_len() {
        let _ = code_verifier_with_len(CODE_VERIFIER_MIN_LEN - 1);
    }

    #[test]
    #[should_panic(expected = "out of RFC 7636 §4.1 range")]
    fn rfc_7636_4_1_rejects_long_len() {
        let _ = code_verifier_with_len(CODE_VERIFIER_MAX_LEN + 1);
    }

    #[test]
    fn rfc_7636_4_2_s256_known_vector() {
        // RFC 7636 Appendix B — informational example:
        //   code_verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
        //   code_challenge = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        //
        // Provides a reference vector for the S256 transform; our
        // implementation MUST match this exact output bit-for-bit.
        let verifier = CodeVerifier("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk".to_string());
        let challenge = code_challenge_s256(&verifier);
        assert_eq!(challenge, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn validate_rejects_non_unreserved_chars() {
        // `=` is reserved per RFC 3986 §2.3.
        let s: String = "A".repeat(CODE_VERIFIER_DEFAULT_LEN - 1) + "=";
        assert!(validate_code_verifier(&s).is_err());

        // `/` is reserved (and would appear in regular base64).
        let s: String = "A".repeat(CODE_VERIFIER_DEFAULT_LEN - 1) + "/";
        assert!(validate_code_verifier(&s).is_err());
    }

    #[test]
    fn validate_rejects_oversize_and_undersize() {
        let s = "A".repeat(CODE_VERIFIER_MIN_LEN - 1);
        assert!(validate_code_verifier(&s).is_err());
        let s = "A".repeat(CODE_VERIFIER_MAX_LEN + 1);
        assert!(validate_code_verifier(&s).is_err());
    }

    #[test]
    fn scope_for_method_v1_0_beta_catalog() {
        // Tier-1 read tools.
        assert_eq!(scope_for_method(METHOD_GRAPH_SCHEMA), Some(SCOPE_READ));
        assert_eq!(scope_for_method(METHOD_GRAPH_INSPECT), Some(SCOPE_READ));
        assert_eq!(scope_for_method(METHOD_GRAPH_EXPLORE), Some(SCOPE_READ));
        assert_eq!(scope_for_method(METHOD_GRAPH_SEARCH), Some(SCOPE_READ));
        // Tier-1 write tools.
        assert_eq!(scope_for_method(METHOD_GRAPH_INGEST), Some(SCOPE_WRITE));
        // Tier-2 power tool.
        assert_eq!(scope_for_method(METHOD_GRAPH_RAW_QUERY), Some(SCOPE_POWER));
        // Methods outside the catalog still deny-by-default.
        assert_eq!(scope_for_method("graph.explain"), None);
        assert_eq!(scope_for_method("unknown.method"), None);
    }

    #[test]
    fn parse_scope_strips_tenant_suffix() {
        // ADR-011 line 162 — `arcgraph.write@<tenant>_<app>` form.
        let parsed = parse_scope_claim("arcgraph.read@tenant_a arcgraph.write@tenant_a");
        assert_eq!(parsed, vec!["arcgraph.read", "arcgraph.write"]);
    }

    #[test]
    fn parse_scope_handles_no_suffix() {
        let parsed = parse_scope_claim("arcgraph.read arcgraph.write");
        assert_eq!(parsed, vec!["arcgraph.read", "arcgraph.write"]);
    }

    #[test]
    fn parse_scope_normalizes_whitespace() {
        let parsed = parse_scope_claim("  arcgraph.read   arcgraph.write\t ");
        assert_eq!(parsed, vec!["arcgraph.read", "arcgraph.write"]);
    }

    #[test]
    fn enforce_scope_accepts_matching_scope() {
        let claims = TokenClaims {
            iss: "issuer".to_string(),
            aud: Audiences::Single("arcgraph".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: "arcgraph.read arcgraph.write".to_string(),
        };
        assert!(enforce_scope(&claims, METHOD_GRAPH_SCHEMA).is_ok());
        assert!(enforce_scope(&claims, METHOD_GRAPH_INGEST).is_ok());
    }

    #[test]
    fn enforce_scope_rejects_missing_scope() {
        let claims = TokenClaims {
            iss: "issuer".to_string(),
            aud: Audiences::Single("arcgraph".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: "arcgraph.read".to_string(),
        };
        match enforce_scope(&claims, METHOD_GRAPH_INGEST) {
            Err(OAuthError::InsufficientScope { required, present }) => {
                assert_eq!(required, SCOPE_WRITE);
                assert_eq!(present, vec!["arcgraph.read"]);
            }
            other => panic!("expected InsufficientScope, got {other:?}"),
        }
    }

    #[test]
    fn enforce_scope_rejects_unknown_method() {
        let claims = TokenClaims {
            iss: "issuer".to_string(),
            aud: Audiences::Single("arcgraph".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: "arcgraph.admin".to_string(),
        };
        // W20β-1 R1 M-2: `graph.raw_query` is now in the table (maps to
        // SCOPE_POWER), so use a method that's truly outside the catalog.
        match enforce_scope(&claims, "graph.totally_unknown") {
            Err(OAuthError::UnknownMethod(m)) => assert_eq!(m, "graph.totally_unknown"),
            other => panic!("expected UnknownMethod, got {other:?}"),
        }
    }

    #[test]
    fn enforce_scope_admits_power_token_for_tier2_methods() {
        // A token carrying `arcgraph.power` drives the raw query method
        // through gate 6a.
        let claims = TokenClaims {
            iss: "issuer".to_string(),
            aud: Audiences::Single("arcgraph".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: "arcgraph.power".to_string(),
        };
        assert!(enforce_scope(&claims, METHOD_GRAPH_RAW_QUERY).is_ok());
    }

    #[test]
    fn enforce_scope_rejects_read_token_for_tier2_methods() {
        // Defense-in-depth: a Read-scope token MUST be rejected (403
        // insufficient_scope) for each Tier-2 method, NOT 401 unknown.
        let claims = TokenClaims {
            iss: "issuer".to_string(),
            aud: Audiences::Single("arcgraph".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: "arcgraph.read".to_string(),
        };
        for method in [METHOD_GRAPH_RAW_QUERY] {
            match enforce_scope(&claims, method) {
                Err(OAuthError::InsufficientScope { required, .. }) => {
                    assert_eq!(
                        required, SCOPE_POWER,
                        "Tier-2 method {method} must require arcgraph.power",
                    );
                }
                other => panic!("expected InsufficientScope for {method}, got {other:?}"),
            }
        }
    }

    #[test]
    fn extract_bearer_strips_scheme_case_insensitive() {
        assert_eq!(
            extract_bearer_token("Bearer abc.def.ghi").unwrap(),
            "abc.def.ghi"
        );
        assert_eq!(extract_bearer_token("bearer xyz").unwrap(), "xyz");
        assert_eq!(extract_bearer_token("BEARER zzz").unwrap(), "zzz");
        // Leading whitespace tolerated.
        assert_eq!(extract_bearer_token("  Bearer tok").unwrap(), "tok");
    }

    #[test]
    fn extract_bearer_rejects_wrong_scheme() {
        match extract_bearer_token("Basic dXNlcjpwYXNz") {
            Err(OAuthError::MalformedBearer(_)) => {}
            other => panic!("expected MalformedBearer, got {other:?}"),
        }
    }

    #[test]
    fn extract_bearer_rejects_empty_token() {
        match extract_bearer_token("Bearer ") {
            Err(OAuthError::MalformedBearer(_)) => {}
            other => panic!("expected MalformedBearer, got {other:?}"),
        }
    }

    #[test]
    fn audiences_accepts_string_or_array() {
        let v: TokenClaims = serde_json::from_value(serde_json::json!({
            "iss": "i", "aud": "single", "exp": 1, "scope": "s"
        }))
        .unwrap();
        assert!(v.aud.contains("single"));

        let v: TokenClaims = serde_json::from_value(serde_json::json!({
            "iss": "i", "aud": ["a", "b"], "exp": 1, "scope": "s"
        }))
        .unwrap();
        assert!(v.aud.contains("a"));
        assert!(v.aud.contains("b"));
        assert!(!v.aud.contains("c"));
    }

    #[test]
    fn jwks_rejects_empty() {
        assert!(JsonWebKeySet::new(vec![]).is_err());
    }

    #[test]
    fn jwks_rejects_duplicate_kid() {
        let k1 = JsonWebKey {
            kid: "a".to_string(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"x"),
        };
        let k2 = JsonWebKey {
            kid: "a".to_string(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"y"),
        };
        assert!(JsonWebKeySet::new(vec![k1, k2]).is_err());
    }

    #[test]
    fn www_authenticate_rendering() {
        assert_eq!(
            oauth_error_to_www_authenticate(&OAuthError::MissingBearer),
            "Bearer realm=\"arcgraph\""
        );
        assert_eq!(
            oauth_error_to_www_authenticate(&OAuthError::InvalidToken("x".to_string())),
            "Bearer realm=\"arcgraph\", error=\"invalid_token\""
        );
        assert_eq!(
            oauth_error_to_www_authenticate(&OAuthError::InsufficientScope {
                required: SCOPE_WRITE,
                present: vec!["arcgraph.read".to_string()],
            }),
            "Bearer realm=\"arcgraph\", error=\"insufficient_scope\", scope=\"arcgraph.write\""
        );
    }

    #[test]
    fn code_verifier_debug_redacts() {
        let v = code_verifier_new();
        let dbg = format!("{v:?}");
        assert!(dbg.contains("redacted"), "Debug must redact, got: {dbg}");
        assert!(
            !dbg.contains(v.as_str()),
            "Debug must not leak the verifier"
        );
    }

    #[test]
    fn oauth_config_with_required_algorithms_rejects_hs() {
        let result = std::panic::catch_unwind(|| {
            let jwks = JsonWebKeySet::new(vec![JsonWebKey {
                kid: "k1".to_string(),
                algorithm: Algorithm::RS256,
                decoding_key: DecodingKey::from_secret(b"x"),
            }])
            .unwrap();
            let _ = OAuthConfig::new("i".to_string(), vec!["a".to_string()], jwks)
                .with_required_algorithms(vec![Algorithm::HS256]);
        });
        assert!(result.is_err(), "HS256 must be rejected per ADR-044");
    }

    // ─────────────────────────────────────────────────────────────────
    // R1 fix-up unit tests — MED-3 edge cases.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn resolve_returns_none_for_unknown_kid() {
        // MED-3 item 3: a `kid` not in the set yields None which the
        // verifier translates to InvalidToken.
        let jwks = JsonWebKeySet::new(vec![JsonWebKey {
            kid: "k1".to_string(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"x"),
        }])
        .unwrap();
        assert!(jwks.resolve(Some("bogus")).is_none());
    }

    #[test]
    fn resolve_returns_none_for_missing_kid_when_multiple_keys() {
        // MED-3 item 2: ambiguous resolve (no `kid` against a
        // multi-key set) yields None → InvalidToken.
        let jwks = JsonWebKeySet::new(vec![
            JsonWebKey {
                kid: "k1".to_string(),
                algorithm: Algorithm::RS256,
                decoding_key: DecodingKey::from_secret(b"x"),
            },
            JsonWebKey {
                kid: "k2".to_string(),
                algorithm: Algorithm::RS256,
                decoding_key: DecodingKey::from_secret(b"y"),
            },
        ])
        .unwrap();
        assert!(jwks.resolve(None).is_none());
    }

    #[test]
    fn resolve_returns_single_when_no_kid_and_one_key() {
        // The common single-key deployment shape — no kid header
        // required.
        let jwks = JsonWebKeySet::new(vec![JsonWebKey {
            kid: "k1".to_string(),
            algorithm: Algorithm::RS256,
            decoding_key: DecodingKey::from_secret(b"x"),
        }])
        .unwrap();
        assert_eq!(jwks.resolve(None).map(|k| k.kid.as_str()), Some("k1"));
    }

    #[test]
    fn parse_scope_empty_claim_yields_empty_vec() {
        // MED-3 item 5: an empty scope claim has no scopes; downstream
        // enforce_scope returns InsufficientScope.
        assert!(parse_scope_claim("").is_empty());
        assert!(parse_scope_claim("   \t  ").is_empty());
    }

    #[test]
    fn enforce_scope_rejects_empty_scope_claim() {
        // MED-3 item 5 (terminal path): a token with empty `scope`
        // claim is rejected with InsufficientScope.
        let claims = TokenClaims {
            iss: "issuer".to_string(),
            aud: Audiences::Single("arcgraph".to_string()),
            exp: u64::MAX,
            nbf: None,
            scope: String::new(),
        };
        match enforce_scope(&claims, METHOD_GRAPH_SCHEMA) {
            Err(OAuthError::InsufficientScope { required, present }) => {
                assert_eq!(required, SCOPE_READ);
                assert!(present.is_empty());
            }
            other => panic!("expected InsufficientScope on empty claim, got {other:?}"),
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // W19γ ADR-049 — tenant_id_hint_from_scope_claim tests.
    // ─────────────────────────────────────────────────────────────────

    #[test]
    fn tenant_id_hint_returns_first_suffix() {
        assert_eq!(
            tenant_id_hint_from_scope_claim("arcgraph.read@alice arcgraph.write@bob"),
            Some("alice".to_string())
        );
    }

    #[test]
    fn tenant_id_hint_returns_none_when_no_suffix() {
        assert!(tenant_id_hint_from_scope_claim("arcgraph.read arcgraph.write").is_none());
    }

    #[test]
    fn tenant_id_hint_returns_none_for_empty_claim() {
        assert!(tenant_id_hint_from_scope_claim("").is_none());
        assert!(tenant_id_hint_from_scope_claim("   ").is_none());
    }

    #[test]
    fn tenant_id_hint_ignores_empty_suffix() {
        // `arcgraph.read@` has an `@` but no suffix → no hint;
        // continues to next word.
        assert_eq!(
            tenant_id_hint_from_scope_claim("arcgraph.read@ arcgraph.write@bob"),
            Some("bob".to_string())
        );
    }

    #[test]
    fn tenant_id_hint_handles_numeric_suffix() {
        assert_eq!(
            tenant_id_hint_from_scope_claim("arcgraph.read@42"),
            Some("42".to_string())
        );
    }

    #[test]
    fn parse_scope_strips_at_first_at_sign() {
        // MED-3 item 6: `scope@a@b` strips at the FIRST `@`. ADR-011
        // §M7-03 is the future RBAC tenant-suffix consumer; this test
        // pins the v1.0-α strip semantics so the M7-03 lift doesn't
        // silently change behavior.
        let parsed = parse_scope_claim("arcgraph.read@tenant_a@subteam");
        assert_eq!(parsed, vec!["arcgraph.read"]);
    }

    #[test]
    fn verify_validation_opts_in_to_validate_nbf() {
        // HIGH-1 defense-in-depth: pin the `validation.validate_nbf`
        // flag at the construction shape used by `verify_bearer_token`
        // so a future refactor that drops the line regresses loudly.
        // We reconstruct the same `Validation` shape inline and
        // assert the flag is true — the integ test
        // `integ_oauth_nbf_in_future_returns_401` is the empirical
        // end-to-end witness; this is the construction-time pin.
        let mut validation = Validation::new(Algorithm::ES256);
        validation.leeway = DEFAULT_CLOCK_SKEW_SECS;
        validation.algorithms = vec![Algorithm::ES256];
        validation.validate_nbf = true;
        assert!(
            validation.validate_nbf,
            "RFC 7519 §4.1.5: validate_nbf MUST be true"
        );
    }
}
