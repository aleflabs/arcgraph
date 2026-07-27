//! W13δ M5-01 proptest — every `ExecutionError` variant maps to a
//! defined MCP error code (no `_ => …` catch-all behaviour).
//!
//! Per the spawn prompt's hard requirement: "1 proptest: every
//! ExecutionError variant maps to a defined MCP error code (no
//! `_ => ...` catch-all)". The proptest generates a random
//! [`ExecutionError`] (covering every public variant) and asserts:
//!
//! 1. The mapping returns one of the defined codes
//!    (-32001, -32003, -32004, -32005, -32006, plus -32602 for the
//!    client-param substrate variants `DimensionMismatch` (#786) and
//!    `IndexAlreadyExists` (#830 / ADR-200) — invalid-params, a CLIENT
//!    fault, not the reserved protocol-layer parse/invalid-request/
//!    internal codes in point 2).
//! 2. The mapping never returns the JSON-RPC parse-error /
//!    invalid-request / internal-error codes for an executor-side
//!    error (those are reserved for protocol-layer faults).
//! 3. The mapping is total (every input produces a non-panicking
//!    output).
//!
//! # PR #286 review MED-2 — `ExplainError` + `SubstrateAccessError`
//!
//! Both `ExplainError` and `SubstrateAccessError` carry a wildcard
//! arm in their `From<…> for MCPError` impls (gated on
//! `#[non_exhaustive]` source-compat). Today's variants are
//! exhaustively matched + map to one of the defined codes; a future
//! variant addition would silently route to `InternalError(-32603)` /
//! `ExecutionEval(-32006)` respectively. The proptest pins the
//! contract: every CURRENT variant of either enum maps to the
//! defined-codes set. A future variant addition that reaches the
//! wildcard arm without an explicit mapping update will turn the
//! `arb_explain_error` / `arb_substrate` strategy stale (build-clean,
//! but the wildcard arm is uncovered) — pair the proptest with the
//! strategy fns: a new variant added to either enum REQUIRES adding
//! a generator branch here.

use arcgraph_core::TenantId;
use arcgraph_mcp::{
    CODE_CANCELLED, CODE_EXECUTION_EVAL, CODE_INDEX_UNAVAILABLE, CODE_INVALID_PARAMS,
    CODE_QUERY_ERROR, CODE_TENANT_UNKNOWN, MCPError,
};
use arcgraph_query::error::{ParseError, Span};
use arcgraph_query::executor::{ExecutionError, SubstrateAccessError};
use arcgraph_query::explain::ExplainError;
use arcgraph_query::semantic::error::ArcQLError;
use proptest::prelude::*;

fn arb_substrate() -> impl Strategy<Value = SubstrateAccessError> {
    prop_oneof![
        any::<u64>().prop_map(|t| SubstrateAccessError::TenantUnknown(TenantId::new(t))),
        prop::sample::select(vec!["vector", "bm25", "community"])
            .prop_map(|s| SubstrateAccessError::IndexUnavailable(s.into())),
        any::<String>().prop_map(SubstrateAccessError::Io),
        // #786 — the client-param dimension-mismatch variant. Maps to
        // -32602 invalid params (NOT the -32006 wildcard). Previously
        // omitted from this generator, so the exhaustiveness proptest
        // below never actually exercised it; added at R1 #872 F1 (the
        // sibling client-error variant of `IndexAlreadyExists`).
        ("[a-z_]{1,16}", 1usize..4097, 1usize..4097).prop_map(
            |(property, query_dim, index_dim)| SubstrateAccessError::DimensionMismatch {
                property,
                query_dim,
                index_dim,
            }
        ),
        // #830 / ADR-200 — the CREATE-VECTOR-INDEX duplicate variant.
        // Maps to -32602 invalid params via the dedicated `substrate_to_mcp`
        // arm (R1 #872 F1). Pairs this generator with the enum per the
        // module-doc convention so the wildcard arm stays covered.
        "[a-zA-Z0-9_]{1,32}".prop_map(|name| SubstrateAccessError::IndexAlreadyExists { name }),
    ]
}

fn arb_arcql() -> impl Strategy<Value = ArcQLError> {
    prop_oneof![
        (
            "[a-zA-Z_][a-zA-Z0-9_]{0,15}",
            "[a-zA-Z0-9_ §-]{0,32}",
            "[a-z0-9_]{0,8}"
        )
            .prop_map(|(feature, section, target)| ArcQLError::NotImplemented {
                feature,
                section,
                target_version: target,
                span: arcgraph_query::error::Span::point(1, 1),
            }),
    ]
}

fn arb_execution_error() -> impl Strategy<Value = ExecutionError> {
    prop_oneof![
        Just(ExecutionError::Cancelled),
        arb_substrate().prop_map(ExecutionError::Substrate),
        arb_arcql().prop_map(ExecutionError::Plan),
        (
            "[a-zA-Z_]{1,20}",
            "M[1-9]-[0-9]{1,3}",
            "ADR-038 §[A-Z][0-9]+"
        )
            .prop_map(|(feature, slice, section)| ExecutionError::NotImplemented {
                feature,
                target_slice: slice,
                section,
            }),
        any::<String>().prop_map(ExecutionError::Eval),
    ]
}

/// Generator covering EVERY CURRENT [`ParseError`] variant. Sister of
/// [`arb_arcql`] / [`arb_substrate`] for the MED-2 closure proptests.
fn arb_parse_error() -> impl Strategy<Value = ParseError> {
    prop_oneof![
        ("[a-zA-Z0-9_ §-]{0,32}", 1usize..50, 1usize..50).prop_map(|(message, line, col)| {
            ParseError::Pest {
                message,
                span: Span::point(line, col),
            }
        }),
        ("[a-zA-Z0-9_ §-]{0,32}", 1usize..50, 1usize..50).prop_map(|(message, line, col)| {
            ParseError::AstConstruction {
                message,
                span: Some(Span::point(line, col)),
            }
        }),
        "[a-zA-Z0-9_ §-]{0,32}".prop_map(|message| ParseError::AstConstruction {
            message,
            span: None,
        }),
        // #819 — the expression-nesting-depth DoS-guard variant. Pins
        // that its error-code mapping lands in `DEFINED_CODES` like every
        // other ParseError variant (it must not silently route via a
        // wildcard to an undefined code).
        (65usize..1000).prop_map(|depth| ParseError::ExpressionTooDeep {
            depth,
            max: arcgraph_query::parser::MAX_EXPRESSION_DEPTH,
        }),
    ]
}

/// Generator covering EVERY CURRENT [`ExplainError`] variant. The
/// proptest pins that every variant lands in the [`DEFINED_CODES`]
/// set — closes PR #286 review MED-2 (forward-bind: a new variant
/// added without an explicit mapping update would silently route to
/// the `_` wildcard `InternalError`; this generator MUST be extended
/// in lockstep).
fn arb_explain_error() -> impl Strategy<Value = ExplainError> {
    prop_oneof![
        arb_parse_error().prop_map(ExplainError::Parse),
        arb_arcql().prop_map(ExplainError::ArcQL),
        Just(ExplainError::Cancelled),
        arb_substrate().prop_map(ExplainError::Substrate),
        any::<String>().prop_map(ExplainError::ExecutionEval),
    ]
}

const DEFINED_CODES: &[i32] = &[
    CODE_CANCELLED,
    CODE_TENANT_UNKNOWN,
    CODE_INDEX_UNAVAILABLE,
    CODE_QUERY_ERROR,
    CODE_EXECUTION_EVAL,
    // -32602: the client-error substrate variants (#786 `DimensionMismatch`,
    // #830 / ADR-200 `IndexAlreadyExists`) map here via their dedicated
    // `substrate_to_mcp` arms — a CLIENT param fault, not a server-side
    // execution fault. Added at R1 #872 F1 alongside the two `arb_substrate`
    // branches so the exhaustiveness proptest genuinely covers all 5 variants.
    CODE_INVALID_PARAMS,
];

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 2048,
        // No need for a custom timeout — every case completes in O(μs).
        ..ProptestConfig::default()
    })]

    /// Every `ExecutionError` variant maps to one of the defined MCP
    /// codes per the W13δ M5↔M4 contract surface table.
    #[test]
    fn execution_error_always_maps_to_defined_code(e in arb_execution_error()) {
        let mcp: MCPError = e.into();
        let code = mcp.code();
        prop_assert!(
            DEFINED_CODES.contains(&code),
            "code {code} not in defined set {:?}; mapped variant: {mcp:?}",
            DEFINED_CODES
        );
    }

    /// Cancelled invariant: the only path that produces -32001 is
    /// `ExecutionError::Cancelled`. Any other input must NOT produce
    /// -32001 (otherwise a SIGTERM drain would be confused with a
    /// per-query cancellation in the M5-side response framing).
    #[test]
    fn only_cancelled_maps_to_minus_32001(e in arb_execution_error()) {
        let cancelled = matches!(e, ExecutionError::Cancelled);
        let mcp: MCPError = e.into();
        if cancelled {
            prop_assert_eq!(mcp.code(), -32001);
        } else {
            prop_assert_ne!(mcp.code(), -32001);
        }
    }

    /// `Plan(ArcQLError)` AND `NotImplemented` (executor-side
    /// forward-deferred operators) MUST both map to -32005 per the
    /// M5↔M4 contract surface table. They are user-visible "your
    /// query asked for something we couldn't run" — same bucket.
    #[test]
    fn arcql_and_not_implemented_map_to_minus_32005(e in arb_execution_error()) {
        let is_query_error = matches!(
            e,
            ExecutionError::Plan(_) | ExecutionError::NotImplemented { .. }
        );
        let mcp: MCPError = e.into();
        if is_query_error {
            prop_assert_eq!(mcp.code(), -32005);
        }
    }

    /// `Substrate(IndexUnavailable)` MUST map to -32004 (a routing
    /// availability issue, not a runtime fault). Pin this distinct
    /// from -32006 (eval / I/O fault) so MCP clients can render
    /// "this tenant doesn't have the bm25 substrate" differently
    /// from "the substrate threw an I/O error".
    #[test]
    fn index_unavailable_maps_to_minus_32004(s in arb_substrate()) {
        let is_unavail = matches!(s, SubstrateAccessError::IndexUnavailable(_));
        let mcp: MCPError = ExecutionError::Substrate(s).into();
        if is_unavail {
            prop_assert_eq!(mcp.code(), -32004);
        }
    }

    /// Tenant-unknown maps to -32003 — distinct from -32004 (index
    /// unavailable) so the MCP client can render "you don't have a
    /// catalog for that tenant" differently from "your tenant has
    /// no vector index attached".
    #[test]
    fn tenant_unknown_maps_to_minus_32003(s in arb_substrate()) {
        let is_unknown = matches!(s, SubstrateAccessError::TenantUnknown(_));
        let mcp: MCPError = ExecutionError::Substrate(s).into();
        if is_unknown {
            prop_assert_eq!(mcp.code(), -32003);
        }
    }

    /// Mapping is TOTAL — no input produces a panic, NO matter how
    /// pathological the substrate / arcql / eval message.
    #[test]
    fn mapping_never_panics(e in arb_execution_error()) {
        // Just constructing + reading the code/message/data must
        // succeed for every variant. (`panic` from a `From` impl
        // would surface here as a proptest fail.)
        let mcp: MCPError = e.into();
        let _ = mcp.code();
        let _ = mcp.message();
        let _ = mcp.data();
    }

    // ─────────────────────────────────────────────────────────────────
    // PR #286 review MED-2 closures — pin `ExplainError` +
    // `SubstrateAccessError` mapping exhaustiveness against the
    // wildcard arms in `error.rs`.
    // ─────────────────────────────────────────────────────────────────

    /// Every `ExplainError` variant maps to a defined code (NOT the
    /// wildcard `InternalError(-32603)` arm). If a future variant is
    /// added to `ExplainError` without (a) updating the strategy AND
    /// (b) updating the `From<ExplainError>` mapping, this test stays
    /// green while the code drifts — pair the strategy with the enum
    /// at every variant landing.
    #[test]
    fn explain_error_always_maps_to_defined_code(e in arb_explain_error()) {
        let mcp: MCPError = e.into();
        let code = mcp.code();
        prop_assert!(
            DEFINED_CODES.contains(&code),
            "code {code} not in defined set {:?} (wildcard arm in From<ExplainError>?); mapped variant: {mcp:?}",
            DEFINED_CODES
        );
    }

    /// `ExplainError::Cancelled` → -32001 — pinned distinct from the
    /// generic mapping invariant so a future variant addition
    /// renaming `Cancelled` doesn't silently drift into `QueryError`.
    #[test]
    fn explain_cancelled_maps_to_minus_32001(e in arb_explain_error()) {
        let cancelled = matches!(e, ExplainError::Cancelled);
        let mcp: MCPError = e.into();
        if cancelled {
            prop_assert_eq!(mcp.code(), CODE_CANCELLED);
        } else {
            prop_assert_ne!(mcp.code(), CODE_CANCELLED);
        }
    }

    /// `ExplainError::Parse` and `ExplainError::ArcQL` BOTH map to
    /// the user-visible `QueryError(-32005)` bucket. Pin both arms so
    /// a future variant rename (e.g., splitting Parse into PestError
    /// + AstError) preserves the wire-shape semantics.
    #[test]
    fn explain_parse_and_arcql_map_to_minus_32005(e in arb_explain_error()) {
        let is_query_layer = matches!(e, ExplainError::Parse(_) | ExplainError::ArcQL(_));
        let mcp: MCPError = e.into();
        if is_query_layer {
            prop_assert_eq!(mcp.code(), CODE_QUERY_ERROR);
        }
    }

    /// `ExplainError::ExecutionEval` → -32006. Mirror of the
    /// `ExecutionError::Eval` invariant on the wider mapping.
    #[test]
    fn explain_execution_eval_maps_to_minus_32006(e in arb_explain_error()) {
        let is_eval = matches!(e, ExplainError::ExecutionEval(_));
        let mcp: MCPError = e.into();
        if is_eval {
            prop_assert_eq!(mcp.code(), CODE_EXECUTION_EVAL);
        }
    }

    /// Every `SubstrateAccessError` variant maps to a defined code
    /// (NOT the wildcard `ExecutionEval` route in
    /// `substrate_to_mcp`). Pairs with `arb_substrate` at every
    /// variant landing.
    #[test]
    fn substrate_access_error_always_maps_to_defined_code(s in arb_substrate()) {
        let mcp: MCPError = s.into();
        let code = mcp.code();
        prop_assert!(
            DEFINED_CODES.contains(&code),
            "code {code} not in defined set {:?} (wildcard arm in substrate_to_mcp?); mapped variant: {mcp:?}",
            DEFINED_CODES
        );
    }

    /// Direct `From<SubstrateAccessError>` and indirect-through-
    /// `ExecutionError::Substrate` produce the SAME code. This pins
    /// the convention that `substrate_to_mcp` is the single source of
    /// truth — a future contributor changing one path would need to
    /// change the other to keep this invariant.
    #[test]
    fn substrate_direct_and_via_execution_agree(s in arb_substrate()) {
        let direct: MCPError = s.clone().into();
        let via_exec: MCPError = ExecutionError::Substrate(s).into();
        prop_assert_eq!(direct.code(), via_exec.code());
    }

    /// `Substrate(IndexUnavailable)` → -32004 invariant, but on the
    /// direct `From<SubstrateAccessError>` path. Mirrors the
    /// `index_unavailable_maps_to_minus_32004` invariant on the
    /// `ExecutionError`-wrapped path.
    #[test]
    fn substrate_index_unavailable_direct_maps_to_minus_32004(s in arb_substrate()) {
        let is_unavail = matches!(s, SubstrateAccessError::IndexUnavailable(_));
        let mcp: MCPError = s.into();
        if is_unavail {
            prop_assert_eq!(mcp.code(), CODE_INDEX_UNAVAILABLE);
        }
    }

    /// `Substrate(TenantUnknown)` → -32003 invariant on the direct
    /// path. Mirrors `tenant_unknown_maps_to_minus_32003`.
    #[test]
    fn substrate_tenant_unknown_direct_maps_to_minus_32003(s in arb_substrate()) {
        let is_unknown = matches!(s, SubstrateAccessError::TenantUnknown(_));
        let mcp: MCPError = s.into();
        if is_unknown {
            prop_assert_eq!(mcp.code(), CODE_TENANT_UNKNOWN);
        }
    }
}
