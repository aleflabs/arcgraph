#![no_main]
//! W27-ξ / ADR-164 §D-4 — MCP `graph.raw_query` dispatcher fuzz target.
//!
//! # What this fuzzes
//!
//! [`arcgraph_mcp::raw_query_tool`] — the synchronous Tier-2 power-user
//! dispatcher at `crates/arcgraph-mcp/src/tools/raw_query.rs` (W16ζ
//! M5-11, ADR-004 amendment-03). This is a DISTINCT surface from the
//! existing `mcp_message_fuzz` target (which fuzzes the JSON-RPC
//! *envelope* decoder `decode_request`): `raw_query_tool` is the
//! *dispatcher* that runs the 6-step validation order
//! (cross-tenant guard → scope check → byte cap → empty guard →
//! row-cap clamp → executor invocation → result serialization).
//!
//! Per the W27-β scope ("fuzz the parser + the dispatcher together"),
//! the harness wires a [`RawQueryExecutor`] adapter
//! ([`FuzzExecutor`]) that invokes the REAL
//! [`arcgraph_query::parse`] on the query string — so the parser is
//! exercised THROUGH the dispatcher, not stubbed out. (The
//! arcgraph-query bounded-context production wiring at M4-08+ routes
//! through `QueryEngine::execute_with_deadline`; the fuzz harness
//! substitutes the parse-only adapter to keep the target hermetic +
//! synchronous.)
//!
//! # Assertions (ADR-164 §D-4)
//!
//! 1. **No-panic.** `raw_query_tool` MUST return `Ok`/`Err` — never
//!    panic / OOM / hang — for ANY control byte + ANY UTF-8 query
//!    string + ANY render format (the JSON / TOON / YAML
//!    `render_response` path is exercised on the `Ok` arm).
//! 2. **Tenant-isolation invariant** (reverse-test per
//!    `feedback_load_bearing_pr_requires_fault_injection_tests.md`):
//!    when the request `tenant_id` ≠ the fixed session tenant, the
//!    dispatcher MUST reject with [`MCPError::Unauthorized`] (-32002)
//!    AND the wired executor MUST NOT have run (the cross-tenant guard
//!    is step 1 — no cross-tenant data path is reachable). A breach of
//!    either clause is a libFuzzer crash.
//!
//! # Input encoding
//!
//! `data[0]` is a control byte (request-tenant match/mismatch, session
//! scope, `max_rows` shape, render format); `data[1..]` is the query
//! string (UTF-8; non-UTF-8 inputs are skipped — the dispatcher is a
//! `String` consumer and non-UTF-8 framing is the JSON-RPC layer's
//! concern, already covered by `mcp_message_fuzz`). Input length is
//! capped at 64 KiB to bound per-iter wall time (the dispatcher's own
//! `MAX_RAW_QUERY_BYTES` = 1 MiB cap path is unit-tested; oversized
//! 1 MiB+ inputs are wall-budget-hostile for fuzz iters).

use std::sync::atomic::{AtomicBool, Ordering};

use libfuzzer_sys::fuzz_target;

use arcgraph_core::TenantId;
use arcgraph_mcp::{
    MCPError, RawQueryExecutor, RawQueryRequest, RawQueryRows, ResponseFormat, SessionScope,
    raw_query_tool,
};
use arcgraph_query::CancellationToken;

const MAX_INPUT_BYTES: usize = 64 * 1024;

/// The fixed session tenant the harness authorizes.
const SESSION_TENANT: u64 = 0x5151_5151_5151_5151;
/// A distinct tenant used to exercise the cross-tenant rejection path.
const OTHER_TENANT: u64 = 0x9999_9999_9999_9999;

/// Parse-backed executor: records whether it ran + invokes the real
/// ArcQL parser so the parser is fuzzed through the dispatcher.
struct FuzzExecutor {
    tenant: TenantId,
    called: AtomicBool,
}

impl FuzzExecutor {
    fn new(tenant: TenantId) -> Self {
        Self {
            tenant,
            called: AtomicBool::new(false),
        }
    }
    fn was_called(&self) -> bool {
        self.called.load(Ordering::SeqCst)
    }
}

impl RawQueryExecutor for FuzzExecutor {
    fn execute(
        &self,
        tenant: TenantId,
        query: &str,
        _max_rows: u32,
        cancel: &CancellationToken,
    ) -> Result<RawQueryRows, MCPError> {
        // Record the call FIRST — the tenant-isolation invariant asserts
        // this NEVER flips on a cross-tenant request (the dispatcher's
        // step-1 guard must short-circuit before we get here).
        self.called.store(true, Ordering::SeqCst);
        if cancel.is_cancelled() {
            return Err(MCPError::Cancelled);
        }
        // Defensive belt-and-suspenders — the dispatcher already enforces
        // `tenant == session_tenant` before calling us.
        if tenant != self.tenant {
            return Err(MCPError::TenantUnknown(format!("{tenant:?}")));
        }
        // Fuzz the parser THROUGH the dispatcher. A parse error maps to
        // the QueryError (-32005) surface the production M5↔M4 bridge
        // uses; a parse success returns an empty (well-formed) row set.
        match arcgraph_query::parse(query) {
            Ok(_) => Ok(RawQueryRows {
                columns: None,
                rows: Vec::new(),
                row_count: 0,
                truncated: false,
                writes: arcgraph_mcp::tools::raw_query::WriteSummary::default(),
            }),
            Err(e) => Err(MCPError::QueryError(format!("{e}"))),
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.is_empty() || data.len() > MAX_INPUT_BYTES {
        return;
    }
    let ctrl = data[0];
    let Ok(query) = std::str::from_utf8(&data[1..]) else {
        return;
    };

    let session_tenant = TenantId::new(SESSION_TENANT);

    // Derive the request shape from the control byte so the mutator can
    // steer across the dispatcher's validation arcs.
    let cross_tenant = ctrl & 0b0000_0001 != 0;
    let req_tenant_id = if cross_tenant {
        OTHER_TENANT
    } else {
        SESSION_TENANT
    };
    let scope = if ctrl & 0b0000_0010 != 0 {
        SessionScope::Read
    } else {
        SessionScope::Power
    };
    let max_rows = match ctrl & 0b0000_1100 {
        0b0000_0000 => None,                      // -> DEFAULT_RAW_QUERY_MAX_ROWS
        0b0000_0100 => Some(0),                   // -> InvalidParams (>= 1)
        0b0000_1000 => Some(u32::from(ctrl) + 1), // small valid clamp
        _ => Some(u32::from(ctrl) << 24),         // likely above the hard cap -> InvalidParams
    };
    let format = match ctrl & 0b0011_0000 {
        0b0000_0000 => None, // -> default (JSON for raw_query)
        0b0001_0000 => Some(ResponseFormat::Json),
        0b0010_0000 => Some(ResponseFormat::Toon),
        _ => Some(ResponseFormat::Yaml),
    };
    // Steer the explain:true verb-consolidation branch (operator-ruled —
    // stays at the ADR-004 10-tool cap). FuzzExecutor uses the default
    // `explain` trait impl (returns MethodNotFound) so an explain:true
    // power session must reject cleanly without panicking.
    let explain = ctrl & 0b0100_0000 != 0;

    let executor = FuzzExecutor::new(session_tenant);
    let cancel = CancellationToken::new();
    let req = RawQueryRequest {
        tenant_id: req_tenant_id,
        query: query.to_string(),
        max_rows,
        format,
        explain,
    };

    // Assertion #1: the dispatcher returns without panicking.
    let result = raw_query_tool(&executor, session_tenant, scope, &cancel, req);

    // Assertion #2: tenant-isolation. A cross-tenant request MUST reject
    // with Unauthorized BEFORE the executor runs.
    if cross_tenant {
        assert!(
            matches!(result, Err(MCPError::Unauthorized)),
            "tenant-isolation breach: cross-tenant raw_query did not reject \
             with Unauthorized (got {result:?})"
        );
        assert!(
            !executor.was_called(),
            "tenant-isolation breach: cross-tenant raw_query reached the executor"
        );
    }
});
