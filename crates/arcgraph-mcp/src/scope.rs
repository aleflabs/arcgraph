//! W16ζ M5-11 — MCP session-scope plumbing per ADR-004 amendment-03.
//!
//! At v1.0-alpha there is no real OAuth (M5-03 is forward-deferred);
//! [`SessionScope`] is a dispatcher field set by the surrounding
//! deployment. The canonical fail-closed entry point
//! [`crate::Dispatcher::with_session_scope`] defaults a new session to
//! [`SessionScope::Read`] — non-power sessions reject
//! [`crate::tools::raw_query::raw_query_tool`] as
//! [`crate::MCPError::Forbidden`] (-32008).
//!
//! When M5-03 lands, [`SessionScope`] becomes a Bearer-token-driven
//! derivation; the [`SessionScope::from_scope_claim`] helper handles
//! the derivation from an OAuth `scope` claim string. The OAuth-empty
//! session continues to default to [`SessionScope::Read`].
//!
//! # ADR provenance
//! - **ADR-004 amendment-03 §D-1, §D-2** — power-scope-gated raw_query +
//!   stub-auth posture.
//! - **design-v2 §9.4** — "Bearer tokens with scopes (`arcgraph.read`,
//!   `arcgraph.write`, `arcgraph.power`, `arcgraph.admin`)".
//! - **design-v2 §9.2** entry 7 — `graph.raw_query` "Requires elevated
//!   permission (OAuth scope `arcgraph.power`)".

/// MCP session scope per design-v2 §9.4.
///
/// `#[non_exhaustive]` permits adding `Write` / `Admin` variants in a
/// future amendment (alongside
/// the M5-03 OAuth slice + the M6+ admin-tool slice per ADR-004
/// amendment-02 §D-3 "Admin op class deferred to M6+") MUST not regress
/// source-compat for downstream pattern-matchers; the `_ => …` catch-
/// all in renderers stays valid.
///
/// At v1.0-alpha only two variants are admitted, per
/// `feedback_avoid_speculative_scaffolding.md`:
///
/// - [`SessionScope::Read`] — the default for non-power sessions; permits
///   Tier-1 read tools. Maps to `arcgraph.read` in the design-v2 §9.4
///   scope set.
/// - [`SessionScope::Power`] — required for [`crate::tools::raw_query`].
///   Maps to `arcgraph.power` in the design-v2 §9.4 scope set.
///
/// `arcgraph.write` + `arcgraph.admin` are forward-pinned: the v1.0-alpha
/// dispatcher does NOT route Tier-1 write tools (`graph.ingest`) through
/// a scope check — `graph.ingest`'s admission is governed by the
/// per-tenant rate-limit (W14γ M5-12) + cross-tenant guard, not by a
/// `Write` scope. When M5-03 OAuth lands, every Tier-1 tool routes
/// through a scope check; the `Write` variant lands then.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum SessionScope {
    /// `arcgraph.read` per design-v2 §9.4. Default for non-power
    /// sessions. Tier-1 read tools (`graph.schema`, `graph.inspect`,
    /// `graph.explore`, `graph.search`) admit; Tier-2 power tools
    /// (`graph.raw_query`) reject.
    #[default]
    Read,
    /// `arcgraph.power` per design-v2 §9.4 + §9.2 entry 7. Required
    /// for `graph.raw_query`. v1.0-alpha legacy constructors
    /// (`Dispatcher::new`, `Dispatcher::with_rate_limiter`) default to
    /// this variant for backward-compat with W13δ / W14β / W14γ test
    /// fixtures; the fail-closed posture activates at M5-03 OAuth.
    Power,
}

impl SessionScope {
    /// Returns the canonical design-v2 §9.4 scope slug for
    /// this variant. Used by the [`crate::MCPError::Forbidden`] data
    /// field so MCP clients can route on the slug without parsing the
    /// message string.
    #[must_use]
    pub const fn slug(self) -> &'static str {
        match self {
            SessionScope::Read => "arcgraph.read",
            SessionScope::Power => "arcgraph.power",
        }
    }

    /// `true` when this scope is sufficient for power-tier tools
    /// (`graph.raw_query`).
    ///
    /// Per ADR-004 amendment-03 §D-1 the power-tier admission is
    /// strict equality on `Power`; future scope variants (`Admin`)
    /// MAY also admit power-tier tools at their landing amendment.
    ///
    #[must_use]
    pub const fn admits_power(self) -> bool {
        matches!(self, SessionScope::Power)
    }

    /// Parse a single scope slug into a [`SessionScope`].
    /// Returns `None` for unknown slugs (round-trip parity: every
    /// non-`None` variant satisfies `from_slug(s.slug()) == Some(s)`).
    ///
    /// Used by [`Self::from_scope_claim`] to derive a session's max-
    /// privilege scope from a space-delimited OAuth `scope` claim.
    /// Unknown slugs are deliberately silent (not an error) — they
    /// might be a future v1.1+ slug that v1.0-β doesn't yet recognize.
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        match slug {
            "arcgraph.read" => Some(SessionScope::Read),
            "arcgraph.power" => Some(SessionScope::Power),
            _ => None,
        }
    }

    /// Derive a [`SessionScope`] from an OAuth `scope` claim
    /// (space-delimited per RFC 8693 §4.2). Returns the MAXIMUM
    /// privilege observed:
    ///
    /// - If `arcgraph.power` is present → [`SessionScope::Power`].
    /// - Otherwise if `arcgraph.read` (or `arcgraph.write`, treated as
    ///   read-equivalent at v1.0-β scope-derivation surface — Write is
    ///   a separate enum variant forward-pinned for v1.1+ OAuth slice)
    ///   is present → [`SessionScope::Read`].
    /// - Otherwise [`SessionScope::Read`] (fail-closed default per
    ///   ADR-004 amendment-03 §D-1).
    ///
    /// `@tenant_id` suffixes per ADR-011 §M7-03 are stripped before the
    /// slug match (`arcgraph.read@7` matches `arcgraph.read`).
    #[must_use]
    pub fn from_scope_claim(scope_claim: &str) -> Self {
        // We scan for `arcgraph.power` first (it dominates) and then for
        // any read-class scope (read/write). `has_read_class` is tracked
        // for readability + so the v1.1+ amendment that adds
        // `SessionScope::Write` only needs to flip the branch return,
        // not re-structure the loop. At v1.0-β both branches yield
        // `Read` — the deliberately-tautological return is documented
        // as the v1.1+ inflection point.
        let mut has_read_class = false;
        for word in scope_claim.split_ascii_whitespace() {
            // Strip `@tenant_id` suffix per ADR-011 §M7-03.
            let slug = match word.find('@') {
                Some(idx) => &word[..idx],
                None => word,
            };
            // Power dominates everything; short-circuit.
            if slug == "arcgraph.power" {
                return SessionScope::Power;
            }
            if slug == "arcgraph.read" || slug == "arcgraph.write" {
                has_read_class = true;
            }
        }
        // v1.0-β: both branches yield `Read`. v1.1+ flips the
        // `has_read_class = true` arm to `SessionScope::Write` per
        // design-v2 §9.4 line 681's full ladder (read, write, power,
        // admin). The `let _ = has_read_class;` keeps the variable
        // load-bearing under the existing compile against `Read` so a
        // future maintainer cannot accidentally drop the read-class
        // scan when adding the Write variant.
        let _ = has_read_class;
        SessionScope::Read
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_read() {
        // Pin the design-v2 §9.4 fail-closed posture: a fresh
        // SessionScope defaults to the lowest-privilege variant.
        assert_eq!(SessionScope::default(), SessionScope::Read);
    }

    #[test]
    fn slugs_match_design_v2_9_4() {
        // design-v2 §9.4 line 665: "Bearer tokens with scopes
        // (`arcgraph.read`, `arcgraph.write`, `arcgraph.power`,
        // `arcgraph.admin`)". The slug returned MUST literally match
        // the design-v2 spelling so MCP clients template against it
        // without surprises.
        assert_eq!(SessionScope::Read.slug(), "arcgraph.read");
        assert_eq!(SessionScope::Power.slug(), "arcgraph.power");
    }

    #[test]
    fn admits_power_is_strict_on_power() {
        // ADR-004 amendment-03 §D-1: power-tier admission is strict
        // equality on Power. Read does NOT admit power-tier tools.
        assert!(SessionScope::Power.admits_power());
        assert!(!SessionScope::Read.admits_power());
    }

    #[test]
    fn admits_power_is_const_fn() {
        // Pin the const-fn property — useful for `static` callers
        // that want to declare a compile-time constant. Without
        // const-fn, the call would not compile in a `const` context.
        // The const-eval itself is the assertion: a non-const fn would
        // not compile the body below.
        const ADMITS: bool = SessionScope::Power.admits_power();
        const { assert!(ADMITS) };
    }

    #[test]
    fn from_scope_claim_derives_power_when_present() {
        assert_eq!(
            SessionScope::from_scope_claim("arcgraph.read arcgraph.power"),
            SessionScope::Power,
        );
    }

    #[test]
    fn from_scope_claim_derives_read_for_read_only_claim() {
        assert_eq!(
            SessionScope::from_scope_claim("arcgraph.read"),
            SessionScope::Read,
        );
    }

    #[test]
    fn from_scope_claim_handles_tenant_suffix() {
        // ADR-011 §M7-03: `@tenant_id` suffix is stripped before slug
        // match. The numerical suffix routes elsewhere (bolt::tenant_id_from_claims)
        // but the privilege derivation MUST ignore it.
        assert_eq!(
            SessionScope::from_scope_claim("arcgraph.power@42"),
            SessionScope::Power,
        );
        assert_eq!(
            SessionScope::from_scope_claim("arcgraph.read@alice"),
            SessionScope::Read,
        );
    }

    #[test]
    fn from_scope_claim_empty_returns_read_default() {
        // Fail-closed: an empty claim derives Read, NOT Power.
        assert_eq!(SessionScope::from_scope_claim(""), SessionScope::Read);
    }

    #[test]
    fn from_scope_claim_ignores_unknown_slugs() {
        // Unknown slugs (e.g., a future v1.5 capability not yet in v1.0-β)
        // are silently ignored at the derivation surface. The derivation
        // honors only the v1.0-β privilege ladder.
        assert_eq!(
            SessionScope::from_scope_claim("arcgraph.read unknown.future.slug"),
            SessionScope::Read,
        );
        assert_eq!(
            SessionScope::from_scope_claim("unknown.future.slug arcgraph.power"),
            SessionScope::Power,
        );
    }
}
