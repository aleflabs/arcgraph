//! W16β M5-03 — Authentication primitives for the MCP transports.
//!
//! Today the only auth surface is [`oauth_pkce`], which implements
//! OAuth 2.1 + PKCE Bearer-token verification + scope enforcement for
//! the HTTP/TLS transport (design-v2 §9.4 line 665). Future auth
//! mechanisms (DPoP per RFC 9449, mTLS-bound tokens, opaque-token
//! introspection per RFC 7662) land alongside `oauth_pkce` as
//! additional submodules without disturbing the existing surface.
//!
//! ## ADR provenance
//!
//! - **ADR-044** (OAuth 2.1 + PKCE for the HTTP/TLS MCP Transport) —
//!   the design-of-record for this slice; codifies PKCE-required,
//!   S256-only, JWT-with-asymmetric-signature-only,
//!   static-JWKS-for-v1.0-α, and the
//!   `arcgraph.{read,write,power,admin}` scope vocabulary.
//!   (Originally numbered ADR-043; renumbered to 044 on R1 fix-up
//!   per orchestrator adjudication of a three-way W16 ADR collision.)
//! - **design-v2 §9.4 line 665** — "OAuth 2.1 with PKCE for remote.
//!   Bearer tokens with scopes (...)" — the source of the scope
//!   vocabulary.
//! - **ADR-004 line 41** — Tier-2 tools require `arcgraph.power`
//!   scope; forward-pinned in this slice (no Tier-2 tools land
//!   v1.0-α).
//! - **ADR-011 line 162** — `@tenant_id` scope suffix; recognized
//!   (stripped during scope check) but not enforced in v1.0-α; the
//!   M7-03 row in ADR-011's phased rollout table is the landing
//!   forward-pin.

pub mod oauth_pkce;
