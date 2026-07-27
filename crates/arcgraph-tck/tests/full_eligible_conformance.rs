//! W28 EXCEED-SPEC — full per-eligible-scenario openCypher TCK conformance
//! harness (Quality full-coverage layer; independent part of WBS #571,
//! Feature #570).
//!
//! # Why this exists (the gap it closes)
//!
//! The shipped TCK gate (`tests/tck.rs` + `src/scorecard.rs`, ADR-095) is
//! two-pronged and **neither prong proves per-eligible-scenario
//! conformance**:
//!
//! 1. A **static categorization** snapshot (`STATIC_SNAPSHOT`: 697 Eligible
//!    / 856 NA-Write / 22 NA-Param / 40 NA-OOS = 1 615 scenarios). This is
//!    a heuristic *eligibility* judgement — it says nothing about whether
//!    an Eligible scenario actually *passes*.
//! 2. A **runtime AGGREGATE step-floor** (`passed_steps >= 6500`,
//!    `tests/tck.rs::main`). The cucumber `writer::Stats` surface exposes
//!    only aggregate STEP counts; a single scenario contributes several
//!    steps, most `Then` assertions flow through as `Skipped`, and a
//!    half-broken scenario still contributes passing `Given`/`When` steps.
//!    The aggregate count can stay green while individual scenarios
//!    silently regress. The scorecard itself flags this as a KNOWN GAP:
//!    *"a custom cucumber `Writer` impl is required for per-feature /
//!    per-scenario pass tracking. Forward-pinned to ADR-095 amendment-01."*
//!
//! This harness is that per-scenario layer, built to the **EXCEED-SPEC
//! bar** (ENGINEERING_DOCTRINE §3): it runs the **FULL** set of currently
//! Eligible scenarios (not a 50-curated sample — *representative is a red
//! flag*) through the same engine path the cucumber harness uses
//! (`arcgraph_query::QueryEngine::execute` against an empty stub
//! substrate), and tracks **per-scenario PASS/FAIL** under a **strong
//! oracle**: a scenario passes only when its declared outcome — the parsed
//! `Then` block — *actually matches* the engine's observed outcome. This
//! is a strict superset of the aggregate step-floor.
//!
//! # The strong oracle (no inflation, ENGINEERING_DOCTRINE §2/§3)
//!
//! Per-scenario PASS is **real conformance**, never "didn't error":
//!
//! * **Result-rows expectation** (`Then the result should be[, in any
//!   order | in order]:` + table): the engine must return `Ok(rows)` AND
//!   those rows, rendered into TCK-canonical cell strings, must equal the
//!   expected table's data rows under the crate's strict
//!   [`arcgraph_tck::assert_row_set_equal`] differ — multiset equality by
//!   default, ordered equality when the step says `in order`.
//! * **Empty expectation** (`Then the result should be empty`, or a
//!   header-only result table): engine must return `Ok` with zero rows.
//! * **Error expectation** (`Then a <X>Error should be raised at compile
//!   time | runtime | any time: <detail>`): the engine must return `Err`
//!   that is a **genuine rejection at the matching phase**. Crucially,
//!   `ArcQLError::NotImplemented` and `::Internal` (and `Cancelled`) do
//!   **NOT** count — a query the engine cannot *run* has not been *proven
//!   invalid*. Counting "unsupported" as "correctly rejected" would
//!   inflate the number, which the doctrine forbids.
//!
//! Where the oracle cannot be sure it is **conservative** (counts FAIL),
//! so the reported number is an honest *lower bound* on conformance. It is
//! never inflated.
//!
//! # Substrate choice
//!
//! Eligible scenarios carry no `CREATE`/`MERGE`/`SET`/… in their own steps
//! (that is what makes them Eligible vs NA-Write). Their data comes from a
//! `Given an empty graph` / `Given any graph` opening, or — for a minority
//! — a named-graph fixture (`Given the binary-tree-1 graph`) that v1.0-α
//! cannot yet load. We therefore execute against a **fresh empty
//! `StubExecutorSubstrate` + `StubCatalogProvider`** per scenario — exactly
//! `tests/tck.rs::execute_with_empty_substrate`. Scenarios that genuinely
//! need fixture data (named graphs; the lone `Match5` background `CREATE`)
//! produce empty results and FAIL their assertion — honestly counted as a
//! gap, not silently skipped.
//!
//! # The ratchet (regression-proof, honest)
//!
//! The denominator is pinned to the scorecard's own `STATIC_SNAPSHOT`
//! eligible count (697); the harness independently re-derives the eligible
//! set from the SAME categorization (`categorize_feature_file`) and asserts
//! the two agree, so the headline `passed_eligible / 697` cannot drift from
//! the scorecard. The gate is a **floor at the CURRENT measured pass
//! count** — a ratchet that trips on regression while permitting
//! improvement. See `PASSED_ELIGIBLE_RATCHET_FLOOR` for the measured value
//! and the `TODO(#571)` gap-to-100%.
//!
//! # Frozen-contract note (Director cross-track ruling)
//!
//! Development owns the TCK write-op-eligibility substrate (#536) and the
//! TCK-plugin / scorecard changes (#506). This file is the Quality
//! full-coverage layer in a NEW FILE ONLY — it READS
//! `src/scorecard.rs`'s public surface (`categorize_feature_file`,
//! `Verdict`, `STATIC_SNAPSHOT`) and the public `differ` /
//! `RowSet`; it modifies nothing shared. This PR sequences to merge AFTER
//! Development's TCK PRs.

use std::path::Path;

use arcgraph_core::{LabelId, NodeId, RelId, TenantId, TypeId};
use arcgraph_query::executor::StubExecutorSubstrate;
use arcgraph_query::executor::value::{NodeView, RelView, Value};
use arcgraph_query::semantic::{ArcQLError, StubCatalogProvider};
use arcgraph_query::{ExplainError, QueryEngine};
use arcgraph_tck::scorecard::{STATIC_SNAPSHOT, Verdict, categorize_feature_file};
use arcgraph_tck::{RowSet, assert_row_set_equal};
use cucumber::gherkin::{Background, Feature, GherkinEnv, Scenario, Step};

// ===================================================================
// Pinned numbers — both derivable from a single source the reviewer can
// reproduce (per W19-MFI-6 numerical-claim derivation discipline).
// ===================================================================

/// Denominator for the headline `passed_eligible / N`, pinned to the
/// scorecard's own Eligible-bucket count. Re-derive:
/// `grep STATIC_SNAPSHOT_ELIGIBLE crates/arcgraph-tck/src/scorecard.rs`
/// (697 at the openCypher@583c1419 vendoring). The harness also asserts
/// its own independently-counted eligible set equals this, so the two
/// can never silently diverge.
const ELIGIBLE_DENOMINATOR: usize = STATIC_SNAPSHOT.eligible;

/// **Ratchet floor — the CURRENT measured per-eligible-scenario pass
/// count.** Set to the value the harness reports against the
/// openCypher@583c1419 vendored tree + the v1.0-α read-only stub engine.
///
/// This is a regression gate: a parser/binder/executor regression that
/// drops the pass count below the floor trips the assertion; a future
/// improvement (more eligible scenarios passing) keeps the gate green and
/// is the cue to RAISE this floor in lockstep.
///
/// TODO(#571): raise this floor toward `ELIGIBLE_DENOMINATOR` (697/697 =
/// 100% read-side conformance). The current gap is
/// `ELIGIBLE_DENOMINATOR - PASSED_ELIGIBLE_RATCHET_FLOOR` = 117 scenarios —
/// see the summary line emitted by the gate for the live breakdown of WHY
/// each failing scenario fails. At the measured baseline that gap is:
/// `UnexpectedError=93` (engine errored where rows/empty expected —
/// dominated by executor maturity / `NotImplemented` + un-loadable
/// named-graph fixtures), `RowsMismatch=45` (rows returned but unequal),
/// `ExpectedErrorGotRows=1` (engine did not reject an invalid query),
/// `WrongErrorPhase=2` (errored, but `NotImplemented`/wrong-phase rather
/// than a genuine rejection). The dominant blocker at v1.0-α is the
/// read-only stub substrate + executor maturity; the gap closes as the
/// executor matures, NOT by relaxing this oracle.
///
/// Measured value: 556/697 = 79.8% read-side per-scenario conformance,
/// against the openCypher@583c1419 vendored tree + v1.0-α read-only stub
/// engine. Reproduce:
/// `cargo test -p arcgraph-tck --test full_eligible_conformance -- --nocapture`
/// (reads the `passed_eligible=N/697` headline line).
///
/// **#746 floor bump (287 → 354).** This floor had drifted STALE at 70
/// (10.0%): the foundation train (#733 UNWIND / #735 Value::Map / #734
/// IN+subscript / #737 Value::Path) raised the actual measured count to
/// 287 without bumping the floor in lockstep. #746 then established the
/// binder↔`ProjectOp` output-binding-id contract — unblocking
/// Project-over-Aggregate + WITH-projection execution end-to-end — which
/// flipped a further +67 scenarios (aggregation + WITH-chained buckets),
/// taking the measured count 287 → 354. The new passes land WITH this
/// floor bump in the SAME PR (no main-RED window), and the floor is
/// re-synced to the true measured value to restore the lockstep
/// discipline. (The +85 drop in `UnexpectedError` split 67→pass + 18→
/// `RowsMismatch`: 18 scenarios that previously TOTALLY failed now
/// EXECUTE but need further row-semantic work — a net improvement, not a
/// regression; no previously-passing scenario regressed.)
///
/// **#621 floor bump (354 → 368, net +14).** The `XOR` boolean operator
/// (openCypher v9 §boolean / 3-valued logic) was unimplemented
/// end-to-end (grammar + AST + evaluator); PR-A added it (`kw_xor` +
/// the dual `*_xor_expr` precedence level + `BinOp::Xor` + the
/// `ThreeValued::xor` 3VL truth-table + the type-check arm). The XOR
/// scenarios live in `expressions/boolean/Boolean3` (XOR truth-table /
/// 3VL-null / commutativity / associativity) and
/// `expressions/precedence/Precedence1` + `…/Precedence4` (the
/// `OR < XOR < AND` ladder; Precedence2/3 contain no XOR). The change
/// is purely additive at the AST level — a single-child `*_xor_expr`
/// returns its child unwrapped, so XOR-free queries parse to an
/// IDENTICAL AST (1 796 arcgraph-query tests stay green; no
/// XOR-free / legitimately-passing scenario regressed).
///
/// **Honesty note on Boolean3 [8] (the non-boolean-fail scenario).**
/// It does NOT pass and is NOT part of the +14: `<non-bool> XOR null`
/// returns `null` rather than the spec's compile-time
/// `InvalidArgumentType` error, because the shared AND/OR/XOR
/// type-check arm 3VL-short-circuits to `Null` when EITHER operand is
/// a null literal (verified: `123.4 OR null` and `123.4 AND null`
/// return `null` too). This is the SAME pre-existing behavior as OR's
/// `Boolean2 [8]` (which also does not pass); XOR mirrors AND/OR
/// exactly per the #621 PR-A scope (do NOT diverge XOR's type-check
/// from AND/OR). Boolean3 [8] previously registered as a FALSE pass
/// (the pre-XOR parse error satisfied the "compile-time error"
/// expectation); now that XOR parses, it correctly exposes that gap
/// (`ExpectedErrorGotRows` 26 → 27). Net of that one false-pass
/// correction, the +14 is: Boolean3 [1]-[7] + the Precedence1/4
/// XOR-precedence scenarios. The new passes land WITH this floor bump
/// in the SAME PR (no main-RED window).
///
/// **#621 PR-B floor bump (368 → 369, net +1).** The `CASE` expression
/// (openCypher v9 §3.6 — BOTH the simple `CASE x WHEN v THEN r … END`
/// and searched `CASE WHEN cond THEN r … END` forms) was unimplemented
/// end-to-end (no grammar rule, no `Expression::Case` AST variant, no
/// eval arm — a `RETURN CASE … END` died with a parse error at the
/// `CASE` token). PR-B adds it (`case_expr` in `primary_atom` + the 5
/// soft keywords `kw_case`/`kw_when`/`kw_then`/`kw_else`/`kw_case_end` +
/// `Expression::Case` / `BoundExpression::Case` + the simple-form
/// equality dispatch reusing `values_equal_3vl` + the searched-form 3VL
/// truthiness reusing `ThreeValued`). The +1 is the single
/// `expressions/conditional/Conditional2.feature` scenario `[1] Simple
/// cases over integers` — a Scenario Outline with 12 example rows. The
/// ratchet counts a Scenario Outline as ONE scenario that passes IFF
/// EVERY example conforms (`run_scenario` — cucumber semantics), so the
/// +1 means ALL 12 rows now conform, INCLUDING the type-mismatch
/// discriminators (`'0'` / `true` / `10.1` compared against integer
/// WHENs ⇒ `ELSE 'something else'`, NOT an error — the load-bearing
/// simple-CASE semantic). The change is purely additive at the AST
/// level (a new `primary_atom` alternative committing on the `CASE`
/// keyword; a non-CASE query never commits to it and parses to an
/// IDENTICAL AST — no CASE-free / legitimately-passing scenario
/// regressed; `ExpectedErrorGotRows` stays at 27).
///
/// **Honesty note on Quantifier9-12 (the searched-CASE consumers).**
/// They do NOT pass and are NOT part of the +1: CASE is their FIRST
/// blocker (their queries now PARSE), but they ALSO need `rand()` + list
/// `+`-concatenation, which are out of scope for this slice — they
/// remain failing (`UnexpectedError` / `RowsMismatch`) pending a
/// follow-up. The new pass lands WITH this floor bump in the SAME PR
/// (no main-RED window).
///
/// **#621 PR-C floor bump (369 → 378, net +9).** The `+` operator was
/// numeric-ONLY end-to-end: the `BinOp::Add` arm rejected every
/// non-numeric operand at type-check (`semantic::type_check`'s
/// `is_numeric` fallback), and the executor's `Add` arm routed straight
/// to `arithmetic(..)` which errors on non-numeric — a SINGLE chokepoint
/// blocking BOTH list and string concatenation. PR-C adds the openCypher
/// v9 §3 `+` concat overload (`concat_result_type` in the type-check
/// `Add` arm + `add_or_concat` in the eval `Add` arm): list+list concat,
/// list+element append, element+list prepend, string+string concat. The
/// change is `Add`-ONLY — `Sub`/`Mul`/`Div`/`Mod` over a list/string
/// still type-error — and the numeric path is byte-identical (`arithmetic`
/// is reused verbatim for the non-concat fall-through), so no numeric /
/// concat-free scenario regressed (verified by a before/after passing-set
/// diff: the 9 below are the ONLY delta; zero regressions). NULL
/// propagation is unchanged — `apply_binop`'s `is_null` short-circuit and
/// the type-check 3VL guard already collapse `null + x` to `null` BEFORE
/// the concat dispatch. The +9 are ALL LIST-concatenation consumers
/// (every one previously died `UnexpectedError` at the chokepoint;
/// `UnexpectedError` 231 → 222, every other reason bucket unchanged):
/// `List4 [1]` (list+list same type), `List4 [2]` (list+scalar append),
/// `List6 [3]` (`size()` of concatenated literal lists), `Precedence3
/// [1]-[5]` (the list element-access / slice / append / concat /
/// containment precedence ladder), and `Unwind1 [3]` (UNWIND over a
/// list concatenation). NO eligible TCK scenario exercises a PURE string
/// `+` concat (the string-concat scenarios live in NA-Write / graph-data
/// contexts), so string concat flips 0 here despite being implemented +
/// e2e-proven (`list_string_concat_e2e.rs`) — this partially closes the
/// PR-B Quantifier9-12 forward-dep note above (those consumers also need
/// `rand()`, so they stay failing). The new passes land WITH this floor
/// bump in the SAME PR (no main-RED window).
///
/// **#618 (GA Lane A) floor bump (378 → 387, net +9).** `ORDER BY` could
/// not resolve a variable that the RETURN projection re-emits. The RETURN
/// binder mints a FRESH `output_id` per projection item (the #746
/// binder↔`ProjectOp` contract), but `ORDER BY` (a standalone
/// `Clause::TailOrderBy` the parser emits after RETURN) resolved its
/// variable refs in the PRE-projection scope → the original source id, NOT
/// the projected `output_id`. The lowered plan is `Sort[key=src_id](
/// Project[emits output_id]( … ) )`, and the Sort runs over the Project's
/// OUTPUT schema (which carries `output_id`), so a key of `src_id` died at
/// runtime with `binding … missing from row schema` (and ordering by an
/// alias died at bind with `undeclared variable`). The fix mirrors
/// `bind_with_clause`'s #746 back-patch: `bind_return_clause` pushes a
/// projection-output scope mapping each RETURN output NAME (alias OR
/// passthrough variable) → its `output_id` atop the pre-projection scope,
/// so `ORDER BY` resolves to the id the `ProjectOp` emits (the
/// pre-projection scope stays underneath as the fall-back, preserving
/// resolution of an in-scope-but-not-returned ordering expression). The
/// change is binder-only (`semantic/binding.rs`); lowering is unchanged.
/// The +9 are ALL of `clauses/return/ReturnOrderBy1.feature` — the
/// canonical single-key ordering scenarios `[1]`-`[9]`: `ORDER BY` and
/// `ORDER BY DESC` over booleans (`[1]`/`[2]`), strings (`[3]`/`[4]`),
/// ints (`[5]`/`[6]`), floats (`[7]`/`[8]`), and lists (`[9]`) — each a
/// passthrough/alias-orderable `UNWIND … RETURN x ORDER BY x` shape that
/// previously died `UnexpectedError` at the binder. The fix drained 10
/// scenarios out of `UnexpectedError` (222 → 212): 9 flipped to PASS, and
/// 1 flipped to `RowsMismatch` (62 → 63) — a scenario that now EXECUTES
/// (ORDER BY binds) but needs further row-semantic work (a net
/// improvement, not a regression); `ExpectedErrorGotRows` (27) and
/// `WrongErrorPhase` (8) are unchanged. A before/after passing-set diff
/// confirms those 9 are the ENTIRE pass delta with ZERO regressions (no
/// previously-passing scenario dropped — the #749 binder↔`ProjectOp`
/// contract for the projection/aggregate path stays intact). Other
/// `ReturnOrderBy*` scenarios stay failing on still-absent
/// features (multi-key sort, `ORDER BY` over a MATCH-bound property
/// needing graph fixtures, `SKIP`/dynamic-`LIMIT` execution — the
/// non-aggregating "order by a non-projected in-scope var" sub-case is a
/// deferred #618 follow-up). The new passes land WITH this floor bump in
/// the SAME PR (no main-RED window).
///
/// **#618 (GA Lane C) floor bump (387 → 415, net +28 cumulative atop Lane A).** GA Lane C added the
/// openCypher v9 §3 number-literal lexer forms that were unimplemented:
/// hexadecimal `0x…`/`0X…` + octal `0o…`/`0O…` integers (`int_literal`
/// was decimal-only), leading-dot floats `.5` / `.1e-5` (`float_literal`
/// required a leading digit), and the i64::MIN boundary
/// `-9223372036854775808` (the unary `-` over the magnitude 2^63
/// overflowed at the AST int build — now FOLDED at parse time into
/// `Integer(i64::MIN)`). The change is grammar + parser ONLY
/// (`grammar.pest` int/float rules + `parser.rs::parse_radix_i64` radix
/// decode + the `parse_unary_expr` i64::MIN fold); the `Literal::Integer`
/// / `Literal::Float` values + every eval / type-check arm are
/// byte-identical, and the fold is SURGICAL (it triggers ONLY at the
/// overflow boundary — every in-range `-N` keeps its `UnaryOp{Neg, ..}`
/// AST), so no number-free / in-range-numeric scenario regressed. The +28
/// are `expressions/literals/Literals2-5` scenarios that previously died
/// `UnexpectedError` at the decimal-only lexer: hexadecimal (`Literals3`),
/// octal (`Literals4`), the decimal i64::MIN (`Literals2 [8]`), and the
/// leading-dot floats whose canonical render matches. `UnexpectedError`
/// dropped 222 → 183 (−39); of those 39, 28 flipped to PASS and 11 moved
/// to `RowsMismatch` (62 → 73) — scenarios that now PARSE + EXECUTE but
/// whose ROW differs on (a) a float-toString render gap (`.5`→`0.5`,
/// `1e-5`→`0.00001`) owned by the TCK harness's `render_tck`, NOT this
/// lane, or (b) a list-literal NEGATED-element eval gap
/// (`[-0x162CD4F6]`→`[Null]`) that is an EXECUTOR gap independent of radix
/// (`[-5]` / `[1+2]` yield `[Null]` too on `b1540e4f`), NOT a number-lexer
/// gap. No previously-passing scenario regressed: the ONLY movement is OUT
/// of `UnexpectedError` (`ExpectedErrorGotRows`=27 + `WrongErrorPhase`=8
/// are byte-unchanged). The new passes land WITH this floor bump in the
/// SAME PR (no main-RED window). Refs #618.
///
/// **#618 (GA Lane D) floor bump (415 → 418, net +3 cumulative atop Lanes A+C).** The function registry was
/// (a) case-SENSITIVE on lookup — `RANGE(1,3)` failed type-check with
/// `unknown function RANGE` even though the evaluator already lower-cases
/// before dispatch (openCypher functions are case-INSENSITIVE, v9 §3) —
/// and (b) missing `properties()` + (c) rejecting a Map argument to
/// `keys()`. This slice case-folds the registry `lookup`
/// (`eq_ignore_ascii_case`), registers `properties(node|rel|map) -> Map`
/// (with a new compile-time `ArgKind::MapLike` constraint matching
/// openCypher's `InvalidArgumentType`-at-compile-time for scalar/list
/// args), and extends the `keys()` eval arm to accept a `Value::Map`. The
/// +3 are: `Map3 [4]` (`keys()` over maps-with-null-values — sorted key
/// output matches the alphabetical `k`/`l`/`m` examples), `Map3 [5]`
/// (`keys()` + `IN` field-existence — boolean output, no list-order
/// dependency), and `Unwind1 [4]` (the case-fold win — `UNWIND RANGE(1,2)`
/// now resolves `RANGE` -> `range`). `Graph9 [5]/[6]/[7]`
/// (`properties(1)`/`properties('x')`/`properties([..])`) STAY passing —
/// `MapLike` rejects the scalar/list literal at COMPILE time, the phase
/// the scenarios require; an `ArgKind::Any` registration would have
/// REGRESSED them (runtime error → `WrongErrorPhase`), so the compile-time
/// constraint is load-bearing (verified by a before/after passing-set
/// diff: zero regressions). `UnexpectedError` 222 → 217 (5 scenarios:
/// the +3 above plus `Map3 [1]` + `Graph9 [4]`, which moved to
/// `RowsMismatch`, see below); `RowsMismatch` 62 → 64.
///
/// **Honesty note — `Map3 [1]` + `Graph9 [4]` (keys/properties on a map)
/// produce CORRECT values but the HARNESS cannot observe them as passes.**
/// They flip `UnexpectedError` -> `RowsMismatch`, NOT -> pass:
/// `keys({name,age,address})` returns `['address','age','name']` (the
/// correct SET, BTreeMap-sorted) but the scenario says "ignoring element
/// order for lists" — a comparison mode this harness does NOT implement
/// (`parse_expectation` folds it to `Multiset`, which is ROW-order-
/// insensitive only; list CELLS are compared as exact rendered strings),
/// so the sorted order ≠ the TCK insertion-order `['name','age','address']`
/// mismatches. `Graph9 [4]` (`properties(map)`) returns the right map but
/// `render_tck` has no `Value::Map` arm (renders `<unrenderable:…>`). Both
/// are HARNESS-fidelity limitations orthogonal to this engine slice (the
/// engine is proven correct end-to-end in `ga618_fn_registry_e2e.rs`); a
/// list-element-order-insensitive differ + a Map renderer are a separate
/// harness slice that would unlock them. The new passes land WITH this
/// floor bump in the SAME PR (no main-RED window).
///
/// **#773 (Customer-Zero AML; umbrella #649) floor bump (418 → 420, net
/// +2 cumulative).** Post-WITH `WHERE` — the openCypher pipeline filter
/// (G1) + `HAVING`-over-aggregate-alias (G2) — bound in the PRE-WITH
/// scope instead of the WITH projection-OUTPUT scope (`bind_with_clause`,
/// `semantic/binding.rs`): a passthrough `WITH a WHERE a.balance > …`
/// keyed the `Filter` (lowered ABOVE the `Project`) on the pre-WITH Scan
/// id → runtime "missing from row schema" (-32006), and a HAVING
/// `WITH d, sum(t.amount) AS s WHERE s > …` could not even BIND the
/// aggregate alias `s` (-32005 `UndeclaredVariable`). The fix resolves
/// the WITH `WHERE` against the projection output (mirroring the #767
/// ORDER-BY-to-projection-output rule), so the filter runs over the
/// Project / Aggregate OUTPUT schema. A companion `ArgKind::Numeric`
/// type-check fix (`semantic/functions.rs`) admits the `Property{..}`
/// dynamic-schema sentinel so `sum(prop)` / `avg(prop)` type-check at all
/// (the v1.0 catalog under-types every property access as
/// `Property::String`; the prior `Property{Integer|Float}`-only rule
/// false-positived EVERY `sum(prop)`). `UnexpectedError` 168 → 159 (−9):
/// 9 WITH-WHERE / HAVING / `sum(prop)` scenarios stopped erroring; 2
/// flipped to PASS (the +2) and 7 moved to `RowsMismatch` (76 → 83) —
/// they now PARSE + EXECUTE but the ROW differs (row-semantics / harness-
/// render gap, NOT a regression — the same "now executes, needs row-
/// semantic work" movement as the Lane A/D bumps above).
/// `ExpectedErrorGotRows` (27) + `WrongErrorPhase` (8) are BYTE-unchanged
/// → no previously-passing scenario regressed and no should-error
/// scenario now wrongly returns rows (verified by a before/after
/// full-histogram diff). The new passes land WITH this floor bump in the
/// SAME PR (no main-RED window). Refs #773 #618.
///
/// **#773 G4/G5 (Customer-Zero AML count) floor bump (420 → 421, net +1 cumulative atop post-WITH-WHERE).** `count(*)` and
/// `count(DISTINCT x)` / `collect(DISTINCT x)` were PARSE-FAILS: the
/// `function_call` grammar admitted `expression`-only arguments (no `*`,
/// no `DISTINCT`). This slice adds the `star_arg` / `distinct_arg`
/// grammar productions, threads `distinct` / `star` through the AST →
/// bound AST → `AggregationSpec` → `AggregateCall`, makes `count(*)`
/// count ROWS (a non-NULL sentinel folded per row — including all-NULL
/// rows) and `<agg>(DISTINCT x)` dedup the per-group non-NULL values
/// before the fold. The engine is proven correct end-to-end in
/// `arcgraph-query/tests/cz773_count_star_distinct_e2e.rs` (count(*)
/// total + grouped + counts-NULL-property-rows; count/collect(DISTINCT);
/// no-regression on count(var)/collect(x)/RETURN DISTINCT).
///
/// **Why only +1 (honesty note).** `count(*)` is the PARSE gate, not the
/// whole unblock. Most eligible TCK `count(*)` scenarios COMPOSE it with
/// features outside this slice's scope — `WITH`-clause aggregation
/// (`With*`/`WithOrderBy*`), `SKIP`/dynamic-`LIMIT` (`ReturnSkipLimit2`),
/// or write clauses (`Merge5`/`Delete4`, which are NA-Write and not even
/// eligible) — so they STILL fail (on `RowsMismatch` / `UnexpectedError`)
/// for those orthogonal reasons; the Quantifier9-12 scenarios that parse
/// `count(*)` additionally need `rand()` + list-concat (a separate slice),
/// so they stay failing too. Exactly one eligible pure-aggregation
/// scenario flips fail → pass. **No regression is structurally possible:**
/// every construct this slice changed (`count(*)`, `<agg>(DISTINCT x)`,
/// and `<non-agg>(DISTINCT x)` — now a type-check `DistinctNotAllowed`
/// rejection) PREVIOUSLY failed to parse (`-32700`), so those scenarios
/// were already in the failing set and can only move fail → pass or
/// fail → fail-differently, never pass → fail. The new pass lands WITH
/// this floor bump in the SAME PR (no main-RED window).
///
/// **#618 (GA Lane BINDER-VALIDATIONS) floor bump (421 → 457, net +36
/// cumulative).** The engine EXECUTED a class of queries it should
/// REJECT at compile/bind time — MISSING SEMANTIC VALIDATIONS — and
/// rejected a second class at the WRONG PHASE (runtime / `NotImplemented`
/// instead of compile). This slice adds the missing COMPILE-time
/// validations in `arcgraph-query/src/semantic/{binding,type_check,
/// functions}.rs` (the binder is consumed by the planner / executor; the
/// new `BindingError` / `TypeCheckError` variants thread through every
/// pattern-match). The +36 close ALL 29 `ExpectedErrorGotRows` + 7 of 9
/// `WrongErrorPhase` by VALIDATION CLASS:
/// - **duplicate / rebound variable** (openCypher v9 §2
///   `VariableTypeConflict` / `RelationshipUniquenessViolation` /
///   `VariableAlreadyBound`): a variable now carries a KIND (node / rel /
///   path / value) in the binder's scope chain; re-using it as a
///   different kind, or re-using a relationship variable at all, is a
///   bind error (`Match1` [7]/[8]/[9]/[11], `Match2` [9]/[10]/[11]/[13],
///   `Match3` [29]/[30], `Match6` [23]/[24] — 12 scenarios);
/// - **aggregation in an illegal position** (openCypher v9 §6.4
///   `InvalidAggregation` / `NestedAggregation`): a `contains-aggregate`
///   walker rejects aggregation in `WHERE` / `ORDER BY` / nested-in-
///   aggregate / list-comprehension (`MatchWhere1` [14]/[15],
///   `ReturnOrderBy2` [14], `WithOrderBy2` [25], `Return6` [14],
///   `List12` [7] — 6 scenarios);
/// - **projection / scope** (`ColumnNameConflict` / `NoVariablesInScope`
///   / `NoExpressionAlias`): duplicate result-column name, `RETURN *`
///   with no in-scope variables, unaliased `WITH` expression (`Return4`
///   [10], `Return7` [2], `With4` [5] — 3 scenarios);
/// - **`SKIP` / `LIMIT` constant-ness** (`NonConstantExpression` /
///   `NegativeIntegerArgument` / `InvalidArgumentType`): a variable-
///   referencing / negative / float `SKIP`/`LIMIT` now rejects at bind
///   time BEFORE the executor's `SKIP`/dynamic-`LIMIT` `NotImplemented`
///   (`ReturnSkipLimit1` [5]/[10]/[11], `ReturnSkipLimit2` [9]/[12]/[16]
///   — 6 scenarios, all previously `WrongErrorPhase`);
/// - **operator / function arg types** (`InvalidArgumentType`): AND/OR/XOR
///   non-boolean operand (checked BEFORE the 3VL null short-circuit so
///   `<non-bool> AND null` rejects); `type()` on a node, `length()` on a
///   node/rel, `size()` on a path (new REJECT-semantics `ArgKind`s that
///   reject the concrete graph-element mismatch while admitting scalars /
///   `Property` / `Null` / unknown — eval-enforced); a too-large float
///   literal (`FloatingPointOverflow`); an unquoted identifier in a map
///   literal (`{k1: k2}` → `UndefinedVariable`) (`Boolean1`/`2`/`3` [8],
///   `Graph4` [7], `Path3` [2]/[3], `List6` [5], `Literals5` [27],
///   `Literals8` [22] — 9 scenarios).
///
/// **DEFERRED (honest):** `Graph6` [9] + `Map1` [6] (property access on a
/// non-graph-element / non-map, the remaining 2 `WrongErrorPhase`) need
/// the WITH projection-OUTPUT type registered under its `output_id` so
/// the downstream `n.prop` sees a concrete base type. That was prototyped
/// but unmasked an UNRELATED `Subscript` incompleteness — `map[key]`
/// dynamic field access (`Map2` [3]/[4]) rejects a `Map` base via
/// `check_list_operand` (map-subscript admission is unwired) — which
/// would regress 2 previously-passing scenarios. Closing them cleanly
/// requires FIRST completing map-subscript type-check; a separate slice.
///
/// **ZERO over-rejection (the load-bearing guard for this lane).** A
/// before/after PASSING-SET diff (each side = the exact set of passing
/// eligible scenarios under the strong oracle) shows the regression set
/// (passing-before ∧ not-passing-after) is EMPTY; the 36 new passes are
/// the ONLY delta. Confirmed by the full histogram: `RowsMismatch` (83)
/// and `UnexpectedError` (155) are BYTE-UNCHANGED — no previously-passing
/// scenario flipped into a different failure bucket (the over-rejection
/// signature). Every validation rejects EXACTLY the invalid form: each
/// new check ships per-class e2e tests pairing the invalid query (→ the
/// expected compile error) with a VALID neighbour (→ still succeeds). The
/// new passes land WITH this floor bump in the SAME PR (no main-RED
/// window). Refs #618.
///
/// **#618 (GA Lane HARNESS-RENDER) floor bump (cumulative with the
/// binder lane above; render contributes +25 disjoint RowsMismatch).** This
/// is a HARNESS-FIDELITY slice, NOT an executor change — it credits
/// already-correct engine output that the harness's `render_tck` comparator
/// could not render. `render_tck` had NO `Value::Map` arm (every map result
/// rendered `<unrenderable:…>`), rendered floats via Rust `{:?}` (Debug —
/// wrong canonical form for the small-exponent band + `-0.0`), and did not
/// escape strings. This slice adds the `Value::Map` arm (`{key: value, …}`,
/// bare keys, RECURSIVE values, `{}` empty), the openCypher float canonical
/// form (`render_tck_float` — plain `0.00001`/`0.000001`, `-0.0`→`0.0`,
/// `.0` on whole values, scientific kept for magnitude extremes), and the
/// string escape canonical form (`render_tck_string` — `\`→`\\`, `'`→`\'`).
/// The List arm already recursed through `render_tck`, so list-of-map works.
/// All forms are byte-tested against the vendored corpus
/// (`Literals5`/`6`/`8` expected tables) in `render_tck_maps_match_tck_form`
/// / `render_tck_float_matches_tck_form` / `render_tck_string_escapes_match_tck_form`.
/// The +25 are: `Literals8` [1]-[8]/[10]/[12]-[17] (maps), `Literals7` [13]
/// (`[{}]`), `Literals6` [4] (escaped `'`), `Literals5` [9]/[10] (`-0.0`) +
/// [16]/[17]/[18]/[22]/[23]/[24] (small-exponent floats).
///
/// **Honesty note — what did NOT flip, and why (out of scope for a render
/// lane).** The histogram moved ONLY in `RowsMismatch` (83 → 58, −25);
/// `UnexpectedError`/`ExpectedErrorGotRows`/`WrongErrorPhase` are byte-
/// unchanged (this slice touches only the row renderer). Several map/float
/// scenarios stay `RowsMismatch` because the ENGINE produces the wrong
/// `Value` (not a render gap): a negative/hex/float numeric literal INSIDE a
/// list/map literal currently parses to `Null` (`Literals8` [9] `{F: -0x…}`,
/// [11] `{k: -.1e-5}`; `Literals7` [5] `[-0x…]`, [7] `[-.1e-5]`, [14] the
/// `-2` element) — a query-crate lexer/parser bug, out of scope for this
/// harness lane (sibling lanes own arcgraph-query). `Literals8` [18] /
/// `Literals7` [18] (complex literals with NON-sorted declared key order)
/// stay failing because `Value::Map` is a `BTreeMap` — the declaration order
/// is already lost in the engine representation (a representation
/// limitation, not a render gap). `Literals6` [5] stays failing because the
/// cucumber gherkin table parser does NOT un-escape `\\` in the `.feature`
/// cell, so the expected cell carries un-normalized quadruple-backslashes
/// that the (correct) openCypher render does not — a table-parser gap, not a
/// render gap; forcing `render_tck` to match it would make it emit
/// non-canonical output. `Map3` [1] (`keys()` returns sorted not declared
/// order, multiset comparison does not normalize list-internal order) and
/// `Graph6` [3]/[7] (`OPTIONAL MATCH` on the empty stub substrate returns 0
/// rows, not 1 null row) / `Graph9` [4] are likewise engine/substrate
/// behavior, not render. None are forced. The new passes land WITH this
/// floor bump in the SAME PR (no main-RED window); a before/after
/// passing-set diff confirms ZERO previously-passing scenario regressed.
///
/// **Cumulative floor after merging the binder lane (+36) and this render
/// lane (+25) — both disjoint deltas: 421 + 36 + 25 = 482.** Re-measured on
/// the rebased tree: `passed_eligible=482/697` (69.2%); `RowsMismatch` 83→58
/// (render), `ExpectedErrorGotRows` 29→0 + `WrongErrorPhase` 9→2 (binder),
/// `UnexpectedError` 155 unchanged; 482+58+155+2 = 697 = `ELIGIBLE_DENOMINATOR`.
///
/// **GA-rand lane (+18 → 500; #618).** Registering the `rand()` builtin
/// (`arcgraph-query` `semantic::functions` BUILTINS + `executor::eval`)
/// unblocks the `Quantifier9`–`Quantifier12` random-INDEPENDENT invariant
/// scenarios, which previously errored at type-check
/// (`UnknownFunction { rand }`) and so sat in `UnexpectedError`. Re-measured
/// (STABLE across 3 consecutive runs — the generator is non-deterministic, so
/// stability across runs is the proof the unblocked scenarios are genuinely
/// random-independent): `passed_eligible=500/697` (71.7%); `UnexpectedError`
/// 155→136 (−19 = the rand-using scenarios that left the bucket);
/// `ExpectedErrorGotRows` 0→1 (+1: `Return6` [15] `RETURN count(rand())` now
/// executes — its expected `SyntaxError: NonConstantExpression`
/// aggregate-position validation is a SEPARATE unimplemented gap, not
/// rand-closeable; it was already failing as `UnexpectedError`, so this is a
/// failure-bucket reshuffle, NOT a regression); `RowsMismatch` 58 +
/// `WrongErrorPhase` 2 byte-unchanged. `rand()` is purely additive (reachable
/// only by queries that call it, all of which failed before), so no
/// previously-passing scenario can change — the +18 is a strict superset
/// gain. The `Quantifier1`–`Quantifier4` `RowsMismatch` cases do NOT use
/// `rand()` (the quantifier eval already returns correct openCypher values
/// for boolean/integer/null/3VL inputs) and stay put — a separate
/// render/edge matter, deferred. 500+58+136+2+1 = 697 = `ELIGIBLE_DENOMINATOR`.
///
/// **String-predicate lane (+1 → 501; #773).** Implementing the openCypher
/// v9 §3.3.6 string-comparison operators `STARTS WITH` / `ENDS WITH` /
/// `CONTAINS` in `arcgraph-query` (grammar `special_pred` + `expr_special_pred`
/// suffixes → `BinOp::{StartsWith,EndsWith,Contains}` → permissive type-check
/// → `apply_binop` string kernel with 3VL null-propagation + non-string⇒null)
/// flips exactly ONE eligible scenario:
/// `expressions/precedence/Precedence4` **[4]** "String predicate takes
/// precedence over binary boolean operator" (golden `a=true, b=null, c=true,
/// d=null` — exercises BOTH that the operator binds tighter than `OR` AND the
/// non-string-operand⇒null rule). It previously sat in `UnexpectedError` (the
/// operator was not in the grammar → parse error). Re-measured (before→after
/// failing-set diff confirms a SINGLE flip + ZERO regressions):
/// `passed_eligible=500→501/697` (71.7%→71.9%); `UnexpectedError` 136→135
/// (−1, the Precedence4 [4] parse error); `RowsMismatch` 58 +
/// `ExpectedErrorGotRows` 1 + `WrongErrorPhase` 2 byte-unchanged.
/// 501+58+135+2+1 = 697 = `ELIGIBLE_DENOMINATOR`.
///
/// **Honesty note — what did NOT flip, and why.** The bulk of the string-
/// predicate TCK surface (`expressions/string/String8`–`11`, ~29 scenarios)
/// is NOT-APPLICABLE-write (each CREATEs its fixture) and so is INELIGIBLE at
/// v1.0-α — blocked on the write-op substrate (#536), NOT on grammar support.
/// String predicates are a hard PREREQUISITE for those: they become eligible
/// (and should pass) the moment #536 lands and expands the denominator (see
/// the forward-pin below). The immediate read-side delta is therefore exactly
/// +1; the latent value is large but gated elsewhere. Floor lift +1 (500→501)
/// per the normal ratchet (new pass lands WITH the bump in the SAME PR — no
/// main-RED window); this is a RAISE, so the ADR-095 §"Floor lift protocol"
/// (which governs LOWERING) does not apply.
///
/// **Permissive label/rel-type binding + aggregation-position validation lane
/// (+9 → 510; #796, ADR-038 amendment-12).** Two coupled `arcgraph-query`
/// changes: (a) an unknown label/rel-type no longer raises
/// `UnknownLabel`/`UnknownRelType` (`-32005`) — the binder resolves it to the
/// reserved `LabelId::MAX`/`TypeId::MAX` "unresolved" sentinel, so the pattern
/// matches NOTHING (openCypher "unknown ⇒ empty match"; aligns labels/types
/// with the property dynamic-schema fallback per ADR-038 §"Schema-id
/// resolution"); (b) the companion `AmbiguousAggregationExpression` binder
/// validation (openCypher v9 §6.4 implicit-grouping-key rule) so the
/// previously-`UnknownLabel`-MASKED "Fail if … aggregation …" scenarios are
/// caught correctly instead of regressing. Re-measured (before→after
/// failing-set diff: exactly 9 flips, ZERO regressions):
/// `passed_eligible=501→510/697` (71.9%→73.2%); `UnexpectedError` 135→107
/// (−28: the unknown-label/type binder errors), `RowsMismatch` 58→77 (+19:
/// scenarios that stop erroring but still need #536 fixture data),
/// `ExpectedErrorGotRows` 1→1 + `WrongErrorPhase` 2→2 (the 4 aggregation
/// "Fail if" cases are caught by validation (b), so they do NOT regress).
/// 510+77+107+2+1 = 697 = `ELIGIBLE_DENOMINATOR`. The 9 flips:
/// `Match5` [11]/[12] (empty-interval var-length, now bind+return empty),
/// `Return6` [6]/[18]/[19], `With6` [6]/[7], `With1` [5] (the #796
/// `OPTIONAL MATCH (a:Start)` "Forwarding null" shape), `ReturnSkipLimit2` [5].
///
/// **Honesty note.** 28 scenarios stopped erroring but only 9 reached PASS;
/// the other ~19 became `RowsMismatch` — they expect POPULATED rows and remain
/// gated on the #536 write-op substrate (the dominant read-side TCK lever), NOT
/// on binding. Floor lift +9 (501→510), new passes land WITH the bump (no
/// main-RED window); a RAISE, so ADR-095 §"Floor lift protocol" (LOWERING) does
/// not apply. Closes #796; advances #773.
///
/// **Negative-numeric-literal lane (+13 → 523; #870 companion).** A negative
/// numeric literal parses as `UnaryOp(Neg, <numeric literal>)`, NOT a bare
/// `Literal` (`[-5]` ⇒ element is `UnaryOp(Neg, 5)`). The read-path collection
/// lift (`executor::eval::literal_expression_to_value`) dropped that `UnaryOp`
/// to `Null` (`[-5]` ⇒ `[null]`); the fix folds the unary-`-`/`+` on a numeric
/// literal to its constant value (shared `negate_const_value`). Re-measured
/// (before→after failing-set diff: exactly 13 flips, ZERO regressions):
/// `passed_eligible=510→523/697` (73.2%→75.0%); `RowsMismatch` −13 (the fix
/// produces the correct value, not `null`); all other buckets byte-unchanged.
/// The 13 flips (the bug is general to ANY negative number in a collection,
/// so it reached past `Literals`): `Literals7` [5]/[7]/[14] + `Literals8`
/// [9]/[11] (negative hex/float/int in list/map); `Quantifier1`/`2`/`3`/`4`
/// [3]/[4] (none/single/any/all over a list literal of negative ints/floats);
/// `Aggregation2` [2] (`min()` over negative integers). The SAME root cause
/// fix also lands the write-path (`CREATE (n {x:-5})` / `SET n.x=-3` —
/// `executor::ops::literal_lift::bound_literal_value` + the type-check
/// literal-only gate), which Closes #870 but flips no TCK scenario (CREATE/SET
/// are write-op, #536-gated in this read-only harness). Floor lift +13 per the
/// normal ratchet (new passes WITH the bump). Advances #773; #870 companion.
///
/// **ORDER BY by a non-projected in-scope expression lane (+1 → 524; #864).**
/// `RETURN e.id ORDER BY e.n` is valid openCypher — the sort key need NOT be a
/// RETURN column. #857 handled the key that IS a projected column (the binder
/// rewrites it to a `VariableRef` to that output); the genuinely non-projected
/// key failed at `SortOp` ("binding … missing from row schema") because
/// `Project` (#746) dropped its binding. The fix (`lowering`) carries the
/// non-projected key as a HIDDEN `Project` column (its bindings are still live
/// in the projection's input), sorts by it, then a trim `Project` drops it —
/// the standard openCypher hidden-sort-column form. Plus a root-cause fix to
/// the synthetic-id seed (`max_in_clause` now observes projection `output_id`s,
/// per its "never collide with binding-pass ids" contract) so the hidden
/// column never aliases a real output column. Re-measured (before→after diff:
/// exactly 1 flip, ZERO regressions): `passed_eligible=523→524/697`
/// (75.0%→75.2%); the flip is `clauses/return-orderby/ReturnOrderBy4` [1]
/// (`RETURN p ORDER BY rng`, where `rng` came from a prior WITH and is not
/// projected). The DISTINCT / aggregation cases correctly STAY an error
/// (openCypher forbids ordering a DISTINCT/grouped result by a non-output
/// value). Most ORDER BY scenarios need a populated graph (#536-gated), so the
/// read-side delta is +1; the fix is a real correctness/customer fix (#864 MED
/// CZ) verified end-to-end. Floor lift +1 per the normal ratchet. Closes #864.
///
/// **SKIP execution lane (+1 → 525; #842 part A).** Literal `SKIP N` (offset
/// pagination) errored `-32005` in EVERY position (the `LogicalPlan::Skip(_) =>
/// NotImplemented` arm at `executor/pipeline.rs`) while `LIMIT` worked — so
/// `SKIP n LIMIT m` offset pagination was impossible. The fix lights a `SkipOp`
/// (mirrors `LimitOp`: discard the first N rows, then pass through) consuming
/// the already-existing literal `LogicalSkip` lowering; no parser/binder change.
/// Re-measured (before→after failing-set diff: exactly 1 flip, ZERO
/// regressions): `passed_eligible=524→525/697` (75.2%→75.3%); `UnexpectedError`
/// 106→105 (the −1 is the now-executing SKIP) — every other bucket byte-
/// unchanged. The single flip is `clauses/return-skip-limit/ReturnSkipLimit1`
/// [4] (`Accept skip zero` — `MATCH (n) WHERE 1 = 0 RETURN n SKIP 0`,
/// read-only). The other SKIP scenarios in the corpus carry CREATE setup
/// (write-op) and stay #536-gated in this read-only harness, so the read-side
/// delta is only +1 — but the fix unblocks the entire pagination surface once
/// #536 lands. Floor lift +1 per the normal ratchet. Advances #842 (part A);
/// advances #773.
///
/// **WITH DISTINCT lane (+1 → 526; #842 part B).** `RETURN DISTINCT` worked but
/// `WITH DISTINCT …` was a `-32700` PARSE error — the `with_clause` grammar had
/// no `kw_distinct?`, so mid-pipeline dedup (dedup THEN continue the pipeline)
/// was impossible. The fix adds `kw_distinct?` to `with_clause`, threads a
/// `distinct: bool` AST→BoundAST (parser + binder), and the lowering composes
/// the SAME `LogicalDistinct` operator `RETURN DISTINCT` lowers to
/// (`lower_distinct`; #622/#649) over the WITH projection — no new dedup op.
/// Re-measured (before→after failing-set diff vs the SKIP-lane state: exactly 1
/// flip, ZERO regressions): `passed_eligible=525→526/697` (75.3%→75.5%);
/// `UnexpectedError` 105→104 (the now-parsing+executing WITH DISTINCT) — every
/// other bucket byte-unchanged. The single flip is
/// `clauses/with-orderBy/WithOrderBy1` [44] (`UNWIND [0,2,1,2,0,1] AS x WITH
/// DISTINCT x ORDER BY x LIMIT 1 RETURN x`, read-only). The other WITH DISTINCT
/// corpus scenarios (With5 [1]/[2], WithWhere1 [2], WithOrderBy2) all carry
/// CREATE setup (write-op) and stay #536-gated; WithOrderBy1 [45] is a
/// plain-WITH list-comprehension scenario (NOT DISTINCT-gated) and correctly
/// stays failing. Read-side delta +1; the fix unblocks the whole
/// mid-pipeline-dedup surface once #536 lands. Floor lift +1 per the normal
/// ratchet. Closes #842 (part B; part A landed the SKIP lane above); advances
/// #773.
///
/// **Aggregation nested in an expression lane (+1 → 527; #910).** An
/// aggregation used as a SUB-EXPRESSION of a projection (`count(n)*2`,
/// `size(collect(x))`, `sum(x)+1`, `toString(count(n))`, `100.0*count(a)/count(b)`,
/// `collect(x)[i]`, `count(a) > 0`) previously errored at row-eval with the
/// MISLEADING `-32005` `NotImplemented` (`aggregation function … reserved`):
/// the lowering's `try_lift_aggregation` matched only a BARE aggregate, so the
/// OUTER expression fell through to the implicit-GROUP-BY-key path and the
/// embedded aggregate was (mis)evaluated row-wise. The fix (`arcgraph-query`
/// `logical_plan::lowering`) lifts each embedded aggregate into the `Aggregate`
/// node under a fresh HIDDEN binding id (reusing the #746/#864 Aggregate→Project
/// hidden-column tunnel) and rewrites the outer expression to read those hidden
/// columns — composing the existing `AggregateOp` + `ProjectOp` (NO new
/// operator). Re-measured (before→after failing-set diff vs the WITH DISTINCT
/// lane state: exactly 1 flip, ZERO regressions):
/// `passed_eligible=526→527/697` (75.5%→75.6%); `RowsMismatch` 64→63 (−1),
/// all other buckets (`UnexpectedError` 104, `ExpectedErrorGotRows` 1,
/// `WrongErrorPhase` 2) BYTE-UNCHANGED. 527+63+104+2+1 = 697 =
/// `ELIGIBLE_DENOMINATOR`. The flip is `clauses/return/Return2` [10] "Return
/// count aggregation over an empty graph" (`MATCH (a) RETURN count(a) > 0` →
/// `false`): a `count` nested in a `>` comparison that, over the empty stub
/// substrate, previously produced 0 rows (the nested expression treated as a
/// group key) instead of the single `false` row.
///
/// **Honesty note — why +1, not more.** The Return6/Aggregation TCK scenarios
/// that exercise nested aggregation with graph DATA (`Return6` [2]/[4]/[5]/[9],
/// `Aggregation*`) use `CREATE` in `having executed` ⇒ `NotApplicableWrite`
/// (INELIGIBLE until the #536 write-op substrate lands), or a `$param`
/// (`Return6` [17], `With6` [5]) ⇒ `NotApplicableParameterized` — none are in
/// the 697 denominator. The eligible nested-agg scenarios over the empty stub
/// either expect an empty result (already passing, e.g. `Return6` [18]/[19]
/// grouping-key-reference) or expect populated rows (stay `RowsMismatch`,
/// #536-gated). The nested-aggregation CAPABILITY is proven end-to-end by 13
/// hermetic exact-value oracles in
/// `arcgraph-query/tests/aggregation_lowering_integration.rs`
/// (`count(x)*2`=10, `size(collect(x))`=3, `100.0*count(x)/count(*)`=75.0,
/// `collect(x)[1]`=20, grouping-key-reference WITH rows, …). **No regression is
/// structurally possible:** every changed item previously errored (`-32005`,
/// never passing) or, over empty input with a surviving group key, produced 0
/// rows (which the fix preserves) — so each can only move fail→pass or
/// fail→fail-differently, never pass→fail (the empty regression set confirms
/// it). Floor lift +1 per the normal ratchet (new pass lands WITH the bump — no
/// main-RED window); a RAISE, so ADR-095 §"Floor lift protocol" (LOWERING) does
/// not apply. Closes #910; advances #773.
///
/// **WITH-WHERE dropped-var scope lane (+1 → 528; #773 correction).**
/// openCypher lets `WITH ... WHERE` resolve against projection OUTPUTS ∪
/// pre-WITH INPUTS, with OUTPUTS shadowing. The prior #773 fence was too
/// strict: it kept HAVING aliases working but rejected dropped-input refs such
/// as `WITH c WHERE r IS NULL` with `UndeclaredVariable`. The fix binds
/// WITH-WHERE in an output-over-input scope and lowers input-only predicates
/// below the projection so dropped ids are still present in the row schema,
/// while output/HAVING predicates stay above the project/aggregate. Re-measured:
/// `passed_eligible=527→528/697` (75.6%→75.8%); `UnexpectedError` 104→98,
/// `RowsMismatch` 63→68 (several scenarios now execute far enough to compare
/// rows), `ExpectedErrorGotRows` 1 and `WrongErrorPhase` 2 unchanged. The
/// `clauses/with-where/WithWhere*` eligible surface has no remaining dump
/// failures, and `TriadicSelection1` now clears one scenario; the remaining
/// TriadicSelection1 residual is 16 `RowsMismatch` plus 2 `UnexpectedError`
/// cases, so this measured read-side floor lift is +1 rather than the expected
/// +~18. Floor lift +1 per the normal ratchet. Corrects #773's output-only
/// fence over-strictness; advances TriadicSelection1 / WithWhere1 / WithWhere6.
///
/// **OPTIONAL MATCH both-bound anti-join lane (+10 → 538; #996 correction).**
/// `OPTIONAL MATCH (a)-[r]->(c)` with both endpoints already bound now
/// correlates on the full shared `(a,c)` pair and NULL-extends exactly once
/// when the specific relationship is absent. The TCK harness also seeds the
/// vendored binary-tree-1/2 named fixtures for full-eligible runs, so the
/// TriadicSelection1 scenarios exercise the real graph instead of an empty
/// substrate. Re-measured: `passed_eligible=528→538/697` (75.8%→77.2%);
/// `RowsMismatch` 68→58, `UnexpectedError` 98 unchanged, `ExpectedErrorGotRows`
/// 1 and `WrongErrorPhase` 2 unchanged. This is a measured +10, not the
/// original +~18 estimate: TriadicSelection1 still has two multi-rel-type
/// `UnexpectedError` cases and six binary-tree-2 label-filtered
/// `RowsMismatch` cases. Closes #996; advances TriadicSelection1.
///
/// **Precedence exponentiation + Boolean order lane (+6 → 544; #1006).**
/// The expression grammar now admits openCypher `^` with the correct
/// precedence shape: tighter than multiplication, left-associative, and looser
/// than unary prefix so `-3 ^ 2` evaluates as `(-3) ^ 2`. The runtime
/// implements `Pow` as numeric-to-f64 exponentiation, always returning Float;
/// this flips `expressions/precedence/Precedence2` [2], [3], and [4], plus the
/// `^` tokenization scenario `Return2` [1]. Boolean order comparisons
/// (`false < true`) are admitted at type-check/eval and flip
/// `Precedence1` [6] plus `Quantifier7` [3]. Re-measured:
/// `passed_eligible=538→544/697` (77.2%→78.0%).
///
/// **IS NULL / IN postfix comparison-precedence lane (+12 → 556).**
/// Postfix predicates now bind to each comparison operand before binary
/// comparison operators chain them, so `a IS NULL = b IS NULL` parses as
/// `(a IS NULL) = (b IS NULL)`. This flips the Boolean1/Boolean2/Boolean5
/// NULL-ness oracle cluster plus related single-sided and `IN` comparison
/// precedence shapes. Re-measured: `passed_eligible=544→556/697`
/// (78.0%→79.8%); failure breakdown is now `UnexpectedError=93`,
/// `RowsMismatch=45`, `ExpectedErrorGotRows=1`, `WrongErrorPhase=2`.
///
/// **Comparison type-semantics lane (+5 → 561; #1016).**
/// Incompatible non-null equality now yields definite false/true for `=`/`<>`,
/// while incompatible ordering yields null instead of an "incomparable types"
/// eval error; list ordering is element-wise with prefix and null-propagation,
/// and float `0.0 / 0.0` can construct NaN for the comparison corpus. This
/// flips `Comparison2` [4], [5], [6], `Precedence3` [6], and `Comparison1`
/// [8]. `Comparison2` [3] semantics are also shipped, but that scenario is
/// #536-gated (`NotApplicableWrite`) and does not count toward this floor yet.
/// Re-measured: `passed_eligible=556→561/697` (79.8%→80.5%); failure
/// breakdown is now `UnexpectedError=87`, `RowsMismatch=46`,
/// `ExpectedErrorGotRows=1`, `WrongErrorPhase=2`.
///
/// **List orderability lane (+4 → 565; OQ-191-1 sibling of #1021).**
/// `ORDER BY` and MIN/MAX now route list operands through the openCypher
/// orderability total order: list comparison is element-wise, heterogeneous
/// elements use the global type rank, and prefix length breaks ties. The dump
/// diff flips `WithOrderBy1` [10], `ReturnOrderBy1` [10], and `Aggregation2`
/// [9], [12]. The same dump shows `WithOrderBy1` [45] advancing from a list
/// `RowsMismatch` to a later temporal `UnexpectedError`, so it is not counted
/// as a pass. Re-measured: `passed_eligible=561→565/697` (80.5%→81.1%);
/// failure breakdown is now `UnexpectedError=88`, `RowsMismatch=41`,
/// `ExpectedErrorGotRows=1`, `WrongErrorPhase=2`.
///
/// **Leading OPTIONAL MATCH null-extension lane (+8 → 573; #996-followup,
/// residual of the #771/#996 OPTIONAL-MATCH cluster).** A query whose first
/// reading clause is an `OPTIONAL MATCH` over a no-match graph now emits one
/// null-extended row (openCypher 9 §6.5) instead of zero. The lowering roots
/// the leading clause on a unit-row `LogicalEmpty` driving table under a
/// `LeftOuterJoin` (the same idiom leading UNWIND / CALL{} use), and the
/// pipeline builds the empty-shared (Cartesian) `OptionalExpandOp` whose
/// null-extension fires when the right side is empty. The before/after
/// passing-set diff flips exactly `Match7` [1], [10]; `Graph6` [3], [7];
/// `Graph3` [7]; `Graph9` [3]; `Null1` [3]; `Null2` [3], with NO
/// previously-passing scenario regressed (the AFTER failure set is a strict
/// subset of the BEFORE set). Re-measured: `passed_eligible=565→573/697`
/// (81.1%→82.2%); failure breakdown is now `UnexpectedError=88`,
/// `RowsMismatch=33`, `ExpectedErrorGotRows=1`, `WrongErrorPhase=2`.
///
/// **Aggregate-in-ORDER-BY lane (+4 → 577; #1053).** An aggregate inside an
/// `ORDER BY` sort key (`ORDER BY me.age + count(you.age)`) is now accepted
/// when the preceding RETURN / WITH projection is itself AGGREGATING
/// (openCypher 9 §6.6): the binder resolves the sort key's inline aggregate
/// against the post-aggregation output scope (deferring an aggregating
/// `WITH`'s input-frame removal so the aggregate's argument still binds, and
/// validating every non-aggregated leaf is a grouping key — the same rule
/// `check_aggregation_grouping` enforces on a projection), and the lowering
/// lifts the inline aggregate into the SAME `Aggregate` node (reusing the
/// #910 hidden-aggregate tunnel) so the `Sort` orders by the computed column.
/// An aggregate in `ORDER BY` over a NON-aggregating projection stays rejected
/// (`ReturnOrderBy2` [14], `WithOrderBy2` [25] `InvalidAggregation`), as do the
/// non-grouping-leaf forms (`ReturnOrderBy6` [4]/[5], `WithOrderBy4` [19]/[20]).
/// The before/after passing-set diff flips EXACTLY `ReturnOrderBy6` [2], [3]
/// and `WithOrderBy4` [17], [18] (all `UnexpectedError` → PASS), with NO
/// previously-passing scenario regressed (the AFTER failure set is a strict
/// subset of the BEFORE set; 124 → 120 failures). Re-measured:
/// `passed_eligible=573→577/697` (82.2%→82.8%); failure breakdown is now
/// `UnexpectedError=84`, `RowsMismatch=33`, `ExpectedErrorGotRows=1`,
/// `WrongErrorPhase=2`.
///
/// **Null-anchored OPTIONAL MATCH named-path lane (+2 → 579; #1051/#1243
/// sibling, residual of the OPTIONAL-MATCH path-accessor cluster).** A named
/// path whose anchor is a STATICALLY-NULL binding
/// (`WITH null AS a OPTIONAL MATCH p = (a)-[r]->()`) now NULL-extends instead
/// of erroring: the path variable `p` binds to `null`, so `nodes(p)` /
/// `relationships(p)` are `null` (openCypher 9 §6.5 + the path-accessor
/// contract). The KEY is that the `null` TYPE unifies with NODE, so `(a)` is
/// well-typed; it is NOT that OPTIONAL relaxes the type check. Two seams:
/// (1) the binder tags a `WITH null AS x` projection with a distinct
/// null-typed binding kind and, inside an OPTIONAL clause, resolves a node
/// anchor on it to the prior binding (nullable) rather than a
/// `VariableTypeConflict`. A NON-null value anchor (`WITH 123 AS a`,
/// `WITH [1,2] AS a`, `WITH {x:1} AS a`) STILL conflicts — in OPTIONAL just
/// as in a required MATCH — because a non-null scalar/list/map does not unify
/// with NODE (`clauses/match/Match1[11]` pins the non-null matrix; an
/// adversarial e2e extends it to the OPTIONAL form). (2) the OPTIONAL-MATCH
/// pipeline's `build_right_with_singleton_root` learns to descend a Plain
/// `NamedPath` right side (re-wrapping the singleton-rooted pattern with the
/// same `PlainPathOp` the main build uses), with the null left-row's
/// non-`Value::Node` anchor short-circuiting to the `EmptyOp` null-extend
/// branch at execution time. The before/after passing-set diff flips EXACTLY
/// `expressions/path/Path1` [1] and `expressions/path/Path2` [3] (both
/// `UnexpectedError` → PASS), with NO previously-passing scenario regressed
/// (the AFTER failure set is a strict subset of the BEFORE set; 120 → 118
/// failures). Re-measured: `passed_eligible=577→579/697` (82.8%→83.1%);
/// failure breakdown is now `UnexpectedError=82`, `RowsMismatch=33`,
/// `ExpectedErrorGotRows=1`, `WrongErrorPhase=2`.
///
/// **NOT NOT double-negation lane (+1 → 580; #1050).** `NOT NOT <x>` now
/// parses and folds to a genuinely-nested `Not(Not(x))` AST (openCypher 9
/// §boolean; TCK `expressions/boolean/Boolean4` [2]:
/// `RETURN NOT NOT true AS nnt, NOT NOT false AS nnf, NOT NOT null AS nnn`
/// → `true, false, null`). Two seams: (1) the grammar capped `kw_not` at a
/// single optional occurrence (`kw_not?`) on both the `where_not_expr` and
/// `expr_not_expr` rules, so `NOT NOT x` PARSE-FAILED ("expected
/// primary_atom"); lifting both to `kw_not*` admits a run of NOTs (the
/// `IS NOT NULL` construct at `is_null_pred`'s `kw_is ~ kw_not? ~ kw_null`
/// is a DISTINCT single-optional NOT and is left untouched). (2) the parser
/// `parse_not_expr` collapsed N NOTs to ONE boolean (`inners.iter().any(..)`),
/// which — given the lifted grammar — would have made `NOT NOT true`
/// evaluate to `false` (a LATENT eval-parity bug masked by the parse-fail);
/// it now COUNTS the `kw_not` occurrences and folds the inner comparison in
/// that many nested `UnaryOp::Not` layers. Each layer is independently
/// type-checked (binding.rs recurses) and 3VL-evaluated (eval.rs
/// `apply_unop` Not arm recurses — `Not(Not(true))` = `Not(false)` = true;
/// null propagates), so the identity for even NOT-counts and negation for
/// odd counts both fall out with no eval/binding/lowering change. The
/// before/after passing-set diff flips EXACTLY
/// `expressions/boolean/Boolean4` [2] (`UnexpectedError` → PASS), with NO
/// previously-passing scenario regressed (the AFTER failure set is a strict
/// subset of the BEFORE set; 118 → 117 failures). Re-measured:
/// `passed_eligible=579→580/697` (83.1%→83.2%); failure breakdown is now
/// `UnexpectedError=81`, `RowsMismatch=33`, `ExpectedErrorGotRows=1`,
/// `WrongErrorPhase=2`.
///
/// **Map-subscript keystone lane (+2 → 582; #1056 / #990).** The dynamic
/// map subscript `map['key']` is now supported END-TO-END via three
/// coupled changes (executor + type-check only — no grammar/parser): (1)
/// the `Subscript` type-check arm DUAL-DISPATCHES on the base type —
/// `List` base × `Integer` index (existing) OR `Map` base × `String`
/// index (new, `check_subscript_base` + `check_string_index`); (2)
/// `eval_subscript` handles `Value::Map` × `Value::String` with a
/// CASE-SENSITIVE key lookup (missing key ⇒ null); (3) the #618
/// projection-output-type registration is RE-LANDED in
/// `check_projection_item` (`WITH <expr> AS n` registers `n`'s CONCRETE
/// type under its `output_id`), so a downstream property access on a
/// known non-entity/non-map base (`WITH 123 AS n RETURN n.num`) rejects
/// at COMPILE time (`PropertyAccessOnNonEntity`) instead of at runtime.
/// Change (3) was PREVIOUSLY reverted because making the projected type
/// concrete unmasked the map-subscript incompleteness (the old
/// `check_list_operand` over-rejected a `Map` base, regressing
/// `expressions/map/Map2` [3]/[4]); changes (1)+(2) make a `Map` base
/// type-check, so re-landing (3) is now SAFE — the three are co-dependent
/// and ship as one slice. The before/after passing-set diff flips EXACTLY
/// `expressions/graph/Graph6` [9] (`Fail when performing property access
/// on a non-graph element`, `WrongErrorPhase` → PASS) and
/// `expressions/map/Map1` [6] (`Fail when performing property access on a
/// non-map`, `WrongErrorPhase` → PASS), with NO previously-passing
/// scenario regressed — the AFTER failure set is a strict subset of the
/// BEFORE set (`Map2` [3]/[4] STILL PASS — the zero-regression guard
/// holds; 117 → 115 failures). `expressions/map/Map2` [5] (`Dynamically
/// access a field is case-sensitive`) is the DIRECT map-subscript flip
/// target but does NOT flip: its Scenario Outline includes example rows
/// over `{null: 'Mats', NULL: 'Pontus'}`, a map with the reserved
/// keywords `null`/`NULL` as BARE (unquoted) keys, which the grammar
/// rejects (`map_entry = { identifier ~ ":" ~ expression }`, and
/// `identifier` excludes keywords) — a PRE-EXISTING grammar limitation
/// independent of this slice (the map-subscript SEMANTICS for `Map2` [5]
/// are correct, verified in `map_subscript_e2e::map_subscript_is_case_sensitive`;
/// the keyword-map-key grammar gap is a separate parser slice). Re-measured:
/// `passed_eligible=580→582/697` (83.2%→83.5%); failure breakdown is now
/// `UnexpectedError=81`, `RowsMismatch=33`, `ExpectedErrorGotRows=1`,
/// `WrongErrorPhase=0`.
///
/// **`RETURN *` alphabetical column order (+1 → 583; Unwind1 [13]).**
/// `clauses/unwind/Unwind1` [13] ("Multiple unwinds after each other")
/// failed `RowsMismatch` on column ORDER only: sequential `UNWIND`s
/// already composed the correct 2×2×2 cartesian product and `RETURN *`
/// already carried every in-scope binding, but the wildcard passthrough
/// emitted columns in pipeline-DECLARATION order (`xs, ys, zs, x, y, z`)
/// rather than the openCypher rule — ALPHABETICAL by variable name
/// (Cypher 9 §6.1), i.e. `x, xs, y, ys, z, zs`. The fix carries the
/// name-sorted in-scope binding order on `BoundProjectionKind::Wildcard`
/// (the binder's `current_in_scope_named()` BTreeMap iteration is already
/// name-sorted) and reorders the child columns into that order at
/// projection time (`ProjectOp` wildcard arm + `derive_schema`). Anonymous
/// pattern bindings (never `declare`d into the named scope) are correctly
/// excluded from `*`. The before/after passing-set diff flips EXACTLY
/// `Unwind1` [13] (`RowsMismatch` → PASS), with NO previously-passing
/// scenario regressed (`Unwind1` [11], whose `| list | x |` order is both
/// alphabetical AND declaration order, still passes — the zero-regression
/// guard holds). Re-measured: `passed_eligible=582→583/697`
/// (83.5%→83.6%); `RowsMismatch=33→32`, every other reason bucket
/// unchanged.
///
/// **Scoped iteration variable exempt from grouping-key rule (+2 → 585;
/// List11 [3] + List12 [3]).** Both scenarios nest an AGGREGATE inside an
/// ADR-188 scoped-variable form — `ALL(ok IN collect(...) WHERE ok)`
/// (List11 [3]) and `[x IN collect(r) WHERE x <> null]` (List12 [3]) — so
/// the implicit-grouping-key walk (`agg_has_nongrouping_ref`) classified
/// the whole projection as AGGREGATING and then flagged the scoped
/// iteration variable (`ok` / `x`, referenced in the WHERE body) as a FREE
/// non-grouping reference, raising `AmbiguousAggregationExpression` at BIND
/// time (surfaced as `UnexpectedError`). But a `ListPredicate` /
/// `ListComprehension` / `Reduce` iteration variable is LOCALLY bound (not a
/// free outer-scope reference), and openCypher v9 §6.4 governs FREE
/// references only. The fix extends the grouping-key set with each scoped
/// form's iteration variable(s) (`var`, plus `acc_var` for `Reduce`) for the
/// BODY recursion via `extend_keys` — `list` / `init` (outer-scope sources)
/// stay checked against the original keys. The exemption is NARROW: a free
/// non-grouping reference elsewhere in the body still fires (regression-guard
/// `free_var_inside_scoped_form_still_rejected`). The `range()` EVAL was
/// already correct on both axes (an inconsistent step direction already
/// yielded `[]`); the only defect was the binder's grouping-key walk. The
/// before/after passing-set diff flips EXACTLY List11 [3] + List12 [3]
/// (`UnexpectedError` → PASS) with NO previously-passing scenario regressed.
/// Re-measured: `passed_eligible=583→585/697` (83.6%→83.9%);
/// `UnexpectedError=81→79`, every other reason bucket unchanged.
///
/// **Bare reserved-word keys in expression-context map literals (+2 → 587;
/// Map1 [5] + Map2 [5]).** Both scenarios' last two Examples rows use the
/// literal `{null: 'Mats', NULL: 'Pontus'}`. The bare uppercase keyword key
/// `NULL` was keyword-excluded at `map_entry`'s `identifier` position (the
/// case-sensitive `keyword` rule), so the WHOLE literal failed to PARSE
/// (*"expected backtick_ident"*) — surfaced as `UnexpectedError`. (Bare
/// lowercase `null` already parsed; the case-sensitive `"NULL"` keyword does
/// not match it.) The backtick-delimited PROPERTY ACCESS the scenario title
/// names (`` map.`null` ``) already worked — `property_accessor` admits
/// `backtick_ident` and `identifier_text` strips the ticks; the defect was
/// purely the literal-key lexical class. The fix adds a `map_key` rule
/// (`@{ backtick_ident | identifier_inner }`) scoped to the EXPRESSION-context
/// `map_entry`, admitting reserved-word keys without backticks (a map key is
/// always followed by `:`, so there is ZERO clause-ambiguity). This is
/// NARROWER than — and does not pre-empt — the post-`.` property-key v1.1
/// split (ADR-038 amendment-04 §D-X.1 / issue #189): bare `n.NULL` at
/// property-access position STILL parse-fails (pinned by
/// `confinement_post_dot_uppercase_keyword_still_requires_backtick` in
/// `arcgraph-query/tests/map1_5_delimited_key_e2e.rs`). The before/after
/// passing-set diff flips EXACTLY Map1 [5] + Map2 [5] (`UnexpectedError` →
/// PASS) with NO previously-passing scenario regressed (the literal-only
/// `prop_entry` node-pattern map and the post-`.` accessor are untouched).
/// Re-measured: `passed_eligible=585→587/697` (83.9%→84.2%);
/// `UnexpectedError=79→77`, every other reason bucket unchanged.
const PASSED_ELIGIBLE_RATCHET_FLOOR: usize = 587;

// ===================================================================
// FORWARD-PIN (do NOT implement here) — write-op-eligible expansion.
//
// Of the 1 615 vendored scenarios, 856 are NOT-APPLICABLE (write-op) at
// v1.0-α (`STATIC_SNAPSHOT.na_write`): they carry CREATE / MERGE / DELETE
// / SET / REMOVE in setup or query and are blocked by the read-only
// catalog (ADR-006 amendment-01). When Development's TCK write-op
// substrate (#536) lands, those 856 become Eligible and this harness's
// denominator expands from 697 -> 697 + 856 = 1 553 read+write-eligible
// scenarios. That expansion is sequenced AFTER Development (Director
// cross-track ruling) and is deliberately NOT implemented in this PR —
// this comment is the documented forward-pin. The mechanism is identical:
// the only change is including `Verdict::NotApplicableWrite` alongside
// `Verdict::Eligible` in `is_in_scope()` below, plus a substrate that can
// honor the `having executed` write setup.
// ===================================================================

/// Verdicts in scope for THIS PR's gate. v1.0-α read-side only.
/// Forward-pin: `NotApplicableWrite` joins this set once #536 lands.
fn is_in_scope(verdict: Verdict) -> bool {
    matches!(verdict, Verdict::Eligible)
}

// ===================================================================
// Expectation model — parsed from a scenario's `Then` block.
// ===================================================================

/// How the engine's row-set is compared against the expected table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareMode {
    /// `Then the result should be, in any order:` — multiset equality.
    Multiset,
    /// `Then the result should be, in order:` — ordered list equality.
    Ordered,
}

/// The phase at which a TCK scenario expects an error to be raised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorPhase {
    /// `... should be raised at compile time: <detail>`
    Compile,
    /// `... should be raised at runtime: <detail>`
    Runtime,
    /// `... should be raised at any time: *`
    Any,
}

/// The declared expected outcome of a scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Expectation {
    /// Expect `Ok` rows equal to `data` (header stripped) under `mode`.
    Rows {
        mode: CompareMode,
        data: Vec<Vec<String>>,
    },
    /// Expect `Ok` with zero rows.
    Empty,
    /// Expect a genuine `Err` at the given phase.
    Error(ErrorPhase),
    /// No recognizable result/error expectation could be parsed. Counted
    /// as FAIL (conservative) and reported separately for transparency.
    Indeterminate,
}

/// How the engine's `Err` is classified for phase-matching against an
/// error expectation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EngineErrClass {
    /// Genuine compile-phase rejection (parse / bind / type-check /
    /// cross-substrate / logical-plan).
    Compile,
    /// Genuine runtime-phase fault (eval / substrate / resource).
    Runtime,
    /// NOT a conformance signal — the engine could not run the query
    /// (`NotImplemented` / `Internal` / `Cancelled`). Never counts as a
    /// correct rejection.
    Unsupported,
}

/// Reason a scenario was counted as FAIL — surfaced in the summary so the
/// gap-to-697 is explainable, not opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum FailReason {
    /// Engine returned rows but they did not match the expected table.
    RowsMismatch,
    /// Engine returned an error where rows/empty were expected.
    UnexpectedError,
    /// Engine returned rows where an error was expected.
    ExpectedErrorGotRows,
    /// Engine errored, but not at the expected phase (e.g. NotImplemented
    /// where a genuine compile-time rejection was required).
    WrongErrorPhase,
    /// No `When executing query` step, or no parseable expectation.
    Indeterminate,
}

// ===================================================================
// TCK-canonical Value rendering.
//
// TCK expected-result tables render Cypher values in a canonical textual
// form. We render the engine's `Value` cells to the SAME form so the
// strict differ can compare them. The renderer is deliberately faithful
// for the value kinds that pure-read scenarios against an empty substrate
// actually produce (scalars + lists); structural kinds (Node /
// Relationship / temporal / decimal) cannot arise from an empty substrate
// in the read-only v1.0-α path, so they fall through to a clearly-tagged
// debug form that will not coincidentally match a TCK cell (no false
// PASS — at worst an honest under-count).
// ===================================================================

/// Render a single `Value` into its TCK-canonical cell string.
///
/// The string forms here mirror the openCypher canonical value rendering
/// the `.feature` `Then the result should be` tables are written in (read
/// the expected cells in `tck/features/expressions/literals/Literals{5..8}`
/// for the source-of-truth). This is the HARNESS comparator — it must
/// faithfully render what the engine already computed; a missing/wrong arm
/// is a harness-fidelity gap (a spurious `RowsMismatch`), not an executor
/// bug. The scalar arms below are byte-tested against the corpus in
/// [`render_tck_scalars_match_tck_form`] / [`render_tck_float_matches_tck_form`]
/// / [`render_tck_string_escapes_match_tck_form`] / [`render_tck_maps_match_tck_form`].
fn render_tck(v: &Value) -> String {
    match v {
        Value::Null => "null".to_string(),
        Value::Boolean(b) => b.to_string(),
        Value::Integer(i) => i.to_string(),
        // openCypher float canonical form (see `render_tck_float`).
        Value::Float(f) => render_tck_float(*f),
        // TCK string cells are single-quoted with `\\`/`\'` escaping
        // (see `render_tck_string`).
        Value::String(s) => render_tck_string(s),
        Value::List(items) => {
            // Recurses through `render_tck` so list-of-map / nested-list
            // elements render in canonical form (Literals7 [13]).
            let inner: Vec<String> = items.iter().map(render_tck).collect();
            format!("[{}]", inner.join(", "))
        }
        // openCypher map cell: `{key: value, …}` — bare (unquoted) keys,
        // values rendered RECURSIVELY (so nested maps + list-of-map +
        // arbitrary depth work — Literals8 [14]/[15]/[16]), `{}` for empty.
        // `Value::Map` is a `BTreeMap`, so iteration is in sorted-key order;
        // single-key maps (the bulk of Literals8) and already-sorted
        // multi-key maps (Literals8 [17] `{a,c,d}`) render identically to
        // the TCK's expected order. (A literal authored in NON-sorted key
        // order — e.g. Literals8 [18] `{id, type, name, ppu, …}` — has
        // already lost its declaration order in the engine's `BTreeMap`
        // representation; that is an engine-representation limitation, not a
        // render gap, and is out of scope for this harness lane.)
        Value::Map(entries) => {
            let inner: Vec<String> = entries
                .iter()
                .map(|(k, val)| format!("{k}: {}", render_tck(val)))
                .collect();
            format!("{{{}}}", inner.join(", "))
        }
        // Not producible from an empty substrate in the read-only path;
        // tagged so it can never coincidentally equal a TCK cell.
        other => format!("<unrenderable:{other:?}>"),
    }
}

/// openCypher canonical `f64 -> String` for the TCK comparator.
///
/// Derived to reproduce EXACTLY the float forms pinned by the vendored
/// corpus `tck/features/expressions/literals/Literals5.feature` (the only
/// float-rendering oracle the TCK provides — all 26 scenarios are covered
/// by [`render_tck_float_matches_tck_form`]):
///
/// * Non-finite -> `NaN` / `Infinity` / `-Infinity`.
/// * Signed zero normalizes to `0.0` (Literals5 [9]/[10]: `-0.0` -> `0.0`).
/// * The small-magnitude band stays in PLAIN decimal: Rust's `{:?}`
///   switches to scientific notation for `|x| < 1e-4`, but openCypher keeps
///   `0.00001` / `0.000001` plain (Literals5 [16]/[17]/[18]/[22]/[23]/[24]);
///   for a Debug-scientific value with exponent in `[-6, -1]` we re-expand
///   via `{}` (Display never uses scientific) and re-attach a `.0` if whole.
/// * The magnitude EXTREMES keep Rust's Debug scientific form, which
///   already matches the corpus: `1e-305` / `1e308` / `1.2635418652381264e305`
///   / `1.23456789e308` (Literals5 [5]/[6]/[11]/[12]/[25]/[26]). The high
///   extreme (`|x| >= 1e16`) needs no override — Debug is already canonical.
/// * Otherwise Debug is already plain decimal and carries the trailing `.0`
///   for whole values (`1000000000.0` — Literals5 [13]/[14]/[15]) and the
///   shortest round-trip for fractionals (`0.55`, `3985764.3405892686`).
///
/// The `[-6, -1]` re-expansion floor is the smallest plain exponent present
/// in the corpus (`1e-6`); the next corpus magnitude down is `1e-305`, which
/// stays scientific. The boundary is corpus-derived, not guessed — there are
/// no corpus cases in `(-305, -6)`, and we deliberately do not over-reach.
fn render_tck_float(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    // openCypher renders both +0.0 and -0.0 as "0.0" (IEEE `-0.0 == 0.0`).
    let f = if f == 0.0 { 0.0 } else { f };
    let dbg = format!("{f:?}");
    if let Some(e_idx) = dbg.find(['e', 'E']) {
        // Rust Debug chose scientific notation. Parse the decimal exponent.
        let exp: i32 = dbg[e_idx + 1..].parse().unwrap_or(0);
        // Small negative-exponent band -> openCypher plain decimal form.
        if (-6..0).contains(&exp) {
            let plain = format!("{f}"); // Display never uses scientific.
            return if plain.contains('.') {
                plain
            } else {
                format!("{plain}.0")
            };
        }
        // Magnitude extreme -> keep the Debug scientific form (canonical).
        return dbg;
    }
    // Debug was already plain decimal: `.0` for whole, shortest for fraction.
    dbg
}

/// openCypher canonical string cell: single-quoted, with `\` escaped to
/// `\\` and `'` escaped to `\'` (double-quotes are NOT escaped). Mirrors
/// the expected cells in `Literals6.feature` [4]/[5] (the gherkin table
/// parser un-escapes `\\` -> `\` in the `.feature` source, so the rendered
/// string must carry the doubled backslash + escaped single-quote to
/// match). Covered by [`render_tck_string_escapes_match_tck_form`].
fn render_tck_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('\'');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\'' => out.push_str("\\'"),
            other => out.push(other),
        }
    }
    out.push('\'');
    out
}

/// Classify an engine error for phase-matching. `NotImplemented` /
/// `Internal` / `Cancelled` are NOT conformance signals.
fn classify_engine_error(e: &ExplainError) -> EngineErrClass {
    match e {
        ExplainError::Parse(_) => EngineErrClass::Compile,
        ExplainError::ArcQL(a) => match a {
            ArcQLError::Binding(_)
            | ArcQLError::TypeCheck(_)
            | ArcQLError::CrossSubstrate(_)
            | ArcQLError::LogicalPlan(_) => EngineErrClass::Compile,
            ArcQLError::ResourceExhausted { .. } => EngineErrClass::Runtime,
            // NotImplemented / Internal (and any future variant): the
            // engine could not run/validate the query — not a rejection.
            _ => EngineErrClass::Unsupported,
        },
        ExplainError::ExecutionEval(_) | ExplainError::Substrate(_) => EngineErrClass::Runtime,
        // A cancelled query (deadline / explicit) is not a conformance
        // signal — it says nothing about query validity.
        ExplainError::Cancelled => EngineErrClass::Unsupported,
        // `#[non_exhaustive]`: future variants default to "not a signal"
        // until classified explicitly.
        _ => EngineErrClass::Unsupported,
    }
}

/// Does an engine error class satisfy the expected error phase?
fn phase_satisfied(expected: ErrorPhase, observed: EngineErrClass) -> bool {
    match (expected, observed) {
        (_, EngineErrClass::Unsupported) => false,
        (ErrorPhase::Any, _) => true,
        (ErrorPhase::Compile, EngineErrClass::Compile) => true,
        (ErrorPhase::Runtime, EngineErrClass::Runtime) => true,
        _ => false,
    }
}

// ===================================================================
// Gherkin-step helpers.
// ===================================================================

/// True if `step` is a fixture-setup step (`having executed: """..."""`).
fn is_setup_query(step: &Step) -> bool {
    let v = step.value.trim_start().to_ascii_lowercase();
    v.starts_with("having executed")
}

/// True if `step` is a query-under-test step (`executing query:` /
/// `executing control query:`).
fn is_when_query(step: &Step) -> bool {
    let v = step.value.trim_start().to_ascii_lowercase();
    v.starts_with("executing query") || v.starts_with("executing control query")
}

/// Parse a `Then`/`And` step into an [`Expectation`], if it carries one.
/// Returns `None` for non-expectation steps (`no side effects`, etc.).
fn parse_expectation(step: &Step) -> Option<Expectation> {
    let v = step.value.trim().to_ascii_lowercase();

    if v.contains("should be raised at compile time") {
        return Some(Expectation::Error(ErrorPhase::Compile));
    }
    if v.contains("should be raised at runtime") {
        return Some(Expectation::Error(ErrorPhase::Runtime));
    }
    if v.contains("should be raised at any time") {
        return Some(Expectation::Error(ErrorPhase::Any));
    }
    if v.contains("result should be empty") {
        return Some(Expectation::Empty);
    }
    if v.contains("result should be") {
        // Mode: ordered iff the step says "in order" (but NOT "in any
        // order"). "(ignoring element order for lists)" is treated as the
        // strict multiset default — a conservative under-count for the 16
        // such scenarios (no list-internal normalization → no false PASS).
        let mode = if v.contains("in any order") {
            CompareMode::Multiset
        } else if v.contains("in order") {
            CompareMode::Ordered
        } else {
            CompareMode::Multiset
        };
        // Strip the header row; remaining rows are the expected data.
        let data: Vec<Vec<String>> = match &step.table {
            Some(t) if t.rows.len() > 1 => t
                .rows
                .iter()
                .skip(1)
                .map(|row| row.iter().map(|c| c.trim().to_string()).collect())
                .collect(),
            // Header-only or absent table = zero expected rows.
            _ => Vec::new(),
        };
        if data.is_empty() {
            return Some(Expectation::Empty);
        }
        return Some(Expectation::Rows { mode, data });
    }
    None
}

/// Substitute `<placeholder>` tokens in `s` using the example header→value
/// mapping. Used to expand a `Scenario Outline` example row into a
/// concrete runnable scenario.
fn substitute(s: &str, headers: &[String], values: &[String]) -> String {
    let mut out = s.to_string();
    for (h, val) in headers.iter().zip(values.iter()) {
        out = out.replace(&format!("<{}>", h.trim()), val.trim());
    }
    out
}

/// Apply example substitution to a single step, producing a concrete step.
fn substitute_step(step: &Step, headers: &[String], values: &[String]) -> Step {
    let mut concrete = step.clone();
    concrete.value = substitute(&step.value, headers, values);
    concrete.docstring = step
        .docstring
        .as_ref()
        .map(|d| substitute(d, headers, values));
    if let Some(t) = &step.table {
        let mut t2 = t.clone();
        t2.rows = t
            .rows
            .iter()
            .map(|row| row.iter().map(|c| substitute(c, headers, values)).collect())
            .collect();
        concrete.table = Some(t2);
    }
    concrete
}

/// Expand a scenario into one-or-more concrete step lists.
///
/// * Plain scenario (no `Examples:`): a single step list (background steps
///   prepended).
/// * Scenario Outline: one step list per example data row, across every
///   `Examples:` block, with `<placeholder>` substitution applied.
fn expand_runs(background: Option<&Background>, scenario: &Scenario) -> Vec<Vec<Step>> {
    let bg: Vec<Step> = background.map(|b| b.steps.clone()).unwrap_or_default();

    if scenario.examples.is_empty() {
        let mut steps = bg;
        steps.extend(scenario.steps.iter().cloned());
        return vec![steps];
    }

    let mut runs = Vec::new();
    for examples in &scenario.examples {
        let Some(table) = &examples.table else {
            continue;
        };
        if table.rows.len() < 2 {
            continue; // header only / empty — no example rows
        }
        let headers = &table.rows[0];
        for values in table.rows.iter().skip(1) {
            let mut steps = bg.clone();
            for st in &scenario.steps {
                steps.push(substitute_step(st, headers, values));
            }
            runs.push(steps);
        }
    }
    runs
}

// ===================================================================
// Engine path — fresh substrate per scenario, with selected vendored
// named-graph fixtures seeded when a `Given the <name> graph` step is
// present.
// ===================================================================

fn execute_with_fixture(
    cypher: &str,
    catalog: &StubCatalogProvider,
    substrate: &StubExecutorSubstrate,
) -> Result<Vec<Vec<Value>>, ExplainError> {
    let engine = QueryEngine::new(catalog);
    engine.execute(cypher, substrate).map(|res| res.into_rows())
}

fn named_graph_step(step: &Step) -> Option<&str> {
    let value = step.value.trim();
    value
        .strip_prefix("the ")
        .and_then(|rest| rest.strip_suffix(" graph"))
}

fn named_graph_fixture(name: &str) -> Option<(StubCatalogProvider, StubExecutorSubstrate)> {
    match name {
        "binary-tree-1" => Some(binary_tree_fixture(false)),
        "binary-tree-2" => Some(binary_tree_fixture(true)),
        _ => None,
    }
}

fn binary_tree_fixture(split_leaf_labels: bool) -> (StubCatalogProvider, StubExecutorSubstrate) {
    let catalog = StubCatalogProvider::new()
        .with_labels(["A", "X", "Y"])
        .with_rel_types(["KNOWS", "FOLLOWS", "FRIEND"])
        .with_properties(["name"]);

    let mut substrate = StubExecutorSubstrate::new();
    let nodes = [
        (1, "a", 1),
        (2, "b1", 2),
        (3, "b2", 2),
        (4, "b3", 2),
        (5, "b4", 2),
        (6, "c11", 2),
        (7, "c12", if split_leaf_labels { 3 } else { 2 }),
        (8, "c21", 2),
        (9, "c22", if split_leaf_labels { 3 } else { 2 }),
        (10, "c31", 2),
        (11, "c32", if split_leaf_labels { 3 } else { 2 }),
        (12, "c41", 2),
        (13, "c42", if split_leaf_labels { 3 } else { 2 }),
    ];
    for (id, name, label) in nodes {
        substrate = substrate.with_node(
            TenantId::DEFAULT,
            NodeView::new(NodeId::new(id), Some(LabelId::new(label)))
                .with_property("name", Value::String(name.to_owned())),
        );
    }

    let edges = [
        (1, 1, 2, 1),
        (2, 1, 3, 1),
        (3, 1, 4, 2),
        (4, 1, 5, 2),
        (5, 2, 6, 3),
        (6, 2, 7, 3),
        (7, 3, 8, 3),
        (8, 3, 9, 3),
        (9, 4, 10, 3),
        (10, 4, 11, 3),
        (11, 5, 12, 3),
        (12, 5, 13, 3),
        (13, 2, 3, 3),
        (14, 3, 4, 3),
        (15, 4, 5, 3),
        (16, 5, 2, 3),
    ];
    for (id, from, to, rel_type) in edges {
        substrate = substrate.with_edge(
            TenantId::DEFAULT,
            RelView::new(
                RelId::new(id),
                NodeId::new(from),
                NodeId::new(to),
                Some(TypeId::new(rel_type)),
            ),
        );
    }

    (catalog, substrate)
}

/// Run ONE concrete step list (a plain scenario or a single outline
/// example expansion) and return whether it conforms under the strong
/// oracle, plus the FailReason on failure.
fn run_one(steps: &[Step]) -> Result<(), FailReason> {
    // 1. Parse the expectation (first matching Then/And step).
    let expectation = steps
        .iter()
        .find_map(parse_expectation)
        .unwrap_or(Expectation::Indeterminate);

    // 2. Execute setup + queries in document order against one fresh
    //    substrate; remember the LAST query-under-test outcome.
    //    (Setup `having executed` writes fail on the read-only stub; that
    //    is expected and only matters insofar as it leaves data unseeded —
    //    handled honestly by the comparison below.)
    let mut catalog = StubCatalogProvider::new();
    let mut substrate = StubExecutorSubstrate::new();
    let mut last_query_outcome: Option<Result<Vec<Vec<Value>>, ExplainError>> = None;
    for step in steps {
        if let Some(name) = named_graph_step(step) {
            if let Some((seed_catalog, seed_substrate)) = named_graph_fixture(name) {
                catalog = seed_catalog;
                substrate = seed_substrate;
            }
        } else if is_setup_query(step) {
            if let Some(doc) = &step.docstring {
                let _ = execute_with_fixture(doc.trim(), &catalog, &substrate);
            }
        } else if is_when_query(step) {
            if let Some(doc) = &step.docstring {
                last_query_outcome = Some(execute_with_fixture(doc.trim(), &catalog, &substrate));
            }
        }
    }

    // 3. Compare observed outcome against the expectation.
    match expectation {
        Expectation::Indeterminate => Err(FailReason::Indeterminate),
        Expectation::Error(phase) => match last_query_outcome {
            Some(Err(e)) => {
                if phase_satisfied(phase, classify_engine_error(&e)) {
                    Ok(())
                } else {
                    Err(FailReason::WrongErrorPhase)
                }
            }
            Some(Ok(_)) => Err(FailReason::ExpectedErrorGotRows),
            None => Err(FailReason::Indeterminate),
        },
        Expectation::Empty => match last_query_outcome {
            Some(Ok(rows)) if rows.is_empty() => Ok(()),
            Some(Ok(_)) => Err(FailReason::RowsMismatch),
            Some(Err(_)) => Err(FailReason::UnexpectedError),
            None => Err(FailReason::Indeterminate),
        },
        Expectation::Rows { mode, data } => match last_query_outcome {
            Some(Ok(rows)) => {
                let actual = RowSet::from_rows(
                    rows.iter()
                        .map(|row| row.iter().map(render_tck).collect())
                        .collect(),
                );
                let expected = RowSet::from_rows(data);
                let ordered = mode == CompareMode::Ordered;
                if assert_row_set_equal(&actual, &expected, ordered).is_ok() {
                    Ok(())
                } else {
                    Err(FailReason::RowsMismatch)
                }
            }
            Some(Err(_)) => Err(FailReason::UnexpectedError),
            None => Err(FailReason::Indeterminate),
        },
    }
}

/// A scenario PASSES iff EVERY concrete expansion passes (cucumber
/// Scenario-Outline semantics: all examples must conform). Returns the
/// first FailReason on any failing expansion.
fn run_scenario(background: Option<&Background>, scenario: &Scenario) -> Result<(), FailReason> {
    let runs = expand_runs(background, scenario);
    if runs.is_empty() {
        return Err(FailReason::Indeterminate);
    }
    for steps in &runs {
        run_one(steps)?;
    }
    Ok(())
}

// ===================================================================
// The gate.
// ===================================================================

#[test]
fn full_eligible_scenario_conformance_ratchet() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let feature_root = Path::new(manifest_dir).join("tck").join("features");

    let files = arcgraph_tck::enumerate_feature_files(&feature_root)
        .unwrap_or_else(|err| panic!("failed to walk vendored TCK tree {feature_root:?}: {err}"));

    let mut eligible_total = 0usize;
    let mut passed = 0usize;
    // FailReason histogram for the gap-to-697 breakdown.
    let mut fail_hist: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();
    // Opt-in per-scenario failure dump (triage only; OFF by default so CI is
    // never spammed). `ARCGRAPH_TCK_DUMP=1` prints `TCK_FAIL {path}::{name} ->
    // {reason}` for each failing eligible scenario — used to attribute the gap
    // to specific feature clusters (e.g. string-predicate work).
    let dump_failures = std::env::var("ARCGRAPH_TCK_DUMP")
        .map(|v| v == "1")
        .unwrap_or(false);

    for path in &files {
        let records = categorize_feature_file(path)
            .unwrap_or_else(|err| panic!("categorize_feature_file failed for {path:?}: {err}"));
        let feature = Feature::parse_path(path, GherkinEnv::default())
            .unwrap_or_else(|err| panic!("gherkin parse failed for {path:?}: {err}"));

        // Structural alignment: the scorecard parser and gherkin must
        // agree on the per-file scenario count, else our positional
        // verdict↔scenario zip would miscount. A drift here is a real
        // signal (a parser divergence) and trips loudly.
        assert_eq!(
            records.len(),
            feature.scenarios.len(),
            "scenario-count drift in {path:?}: scorecard={} gherkin={} — the two \
             parsers disagree on scenario boundaries; positional verdict mapping \
             would miscount the eligible set",
            records.len(),
            feature.scenarios.len(),
        );

        for (record, scenario) in records.iter().zip(feature.scenarios.iter()) {
            if !is_in_scope(record.verdict) {
                continue;
            }
            eligible_total += 1;
            match run_scenario(feature.background.as_ref(), scenario) {
                Ok(()) => passed += 1,
                Err(reason) => {
                    if dump_failures {
                        eprintln!(
                            "TCK_FAIL {}::{} -> {reason:?}",
                            path.display(),
                            scenario.name
                        );
                    }
                    *fail_hist.entry(format!("{reason:?}")).or_insert(0) += 1;
                }
            }
        }
    }

    let failed = eligible_total - passed;
    let pct = if eligible_total == 0 {
        0.0
    } else {
        (passed as f64 / eligible_total as f64) * 100.0
    };

    // -------- Headline summary (the load-bearing report line) --------
    // Surfaced to CI log readers. Run with `--nocapture` to see it on a
    // green run; it always prints on a red (ratchet-trip) run.
    let gap = ELIGIBLE_DENOMINATOR.saturating_sub(passed);
    eprintln!(
        "\nW28 full-eligible TCK conformance:\n  \
         passed_eligible={passed}/{ELIGIBLE_DENOMINATOR} \
         ({pct:.1}% read-side per-scenario conformance) \
         failed={failed} ratchet_floor={PASSED_ELIGIBLE_RATCHET_FLOOR}\n  \
         gap_to_100%={gap} scenarios"
    );
    eprintln!("  failure breakdown (why the gap exists):");
    for (reason, count) in &fail_hist {
        eprintln!("    {reason:<22} {count}");
    }
    eprintln!(
        "  forward-pin: {} NA-Write scenarios become eligible once Dev #536 lands \
         (denominator 697 -> {}).",
        STATIC_SNAPSHOT.na_write,
        ELIGIBLE_DENOMINATOR + STATIC_SNAPSHOT.na_write,
    );

    // -------- Denominator integrity --------
    // Our independently-counted eligible set MUST equal the scorecard's
    // pinned Eligible bucket; otherwise the `/697` headline is dishonest.
    assert_eq!(
        eligible_total, ELIGIBLE_DENOMINATOR,
        "eligible-set drift: harness counted {eligible_total} Eligible scenarios but \
         scorecard STATIC_SNAPSHOT pins {ELIGIBLE_DENOMINATOR}. The harness and the \
         scorecard categorization have diverged — investigate before trusting the \
         conformance number."
    );
    assert_eq!(
        ELIGIBLE_DENOMINATOR, 697,
        "STATIC_SNAPSHOT.eligible moved off 697 (vendored-tree refresh?); re-measure \
         the ratchet floor against the new eligible set."
    );

    // -------- The ratchet (regression gate) --------
    assert!(
        passed >= PASSED_ELIGIBLE_RATCHET_FLOOR,
        "TCK full-eligible conformance REGRESSION: passed_eligible dropped to {passed}, \
         below the ratchet floor {PASSED_ELIGIBLE_RATCHET_FLOOR}. A parser/binder/\
         executor regression reduced per-scenario conformance. Investigate before \
         lowering the floor (lowering requires a recorded regression-acceptance ADR \
         per ADR-095 §\"Floor lift protocol\")."
    );
}

// ===================================================================
// Adversarial unit coverage for the oracle helpers themselves.
//
// Per `feedback_noop_trampoline_anti_pattern.md` (W23-MFI-6): every
// contract claim in the oracle is exercised under hostile input so the
// oracle cannot silently pass on its own bug — a green conformance number
// is only trustworthy if the comparator that produced it is itself tested.
// ===================================================================

#[test]
fn render_tck_scalars_match_tck_form() {
    assert_eq!(render_tck(&Value::Null), "null");
    assert_eq!(render_tck(&Value::Boolean(true)), "true");
    assert_eq!(render_tck(&Value::Boolean(false)), "false");
    assert_eq!(render_tck(&Value::Integer(42)), "42");
    assert_eq!(render_tck(&Value::Integer(-7)), "-7");
    // Integral floats keep the trailing `.0` (Cypher float toString).
    assert_eq!(render_tck(&Value::Float(2.0)), "2.0");
    assert_eq!(render_tck(&Value::Float(0.5)), "0.5");
    // Strings are single-quoted.
    assert_eq!(render_tck(&Value::String("foo".into())), "'foo'");
}

#[test]
fn render_tck_lists_are_bracketed_and_comma_separated() {
    let list = Value::List(vec![
        Value::Integer(1),
        Value::String("a".into()),
        Value::Boolean(true),
    ]);
    assert_eq!(render_tck(&list), "[1, 'a', true]");
    assert_eq!(render_tck(&Value::List(vec![])), "[]");
    // Nested.
    let nested = Value::List(vec![Value::List(vec![Value::Integer(1)])]);
    assert_eq!(render_tck(&nested), "[[1]]");
}

// Build a `Value::Map` from `(key, value)` pairs for the unit tests below.
fn map_of(pairs: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Map(
        pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect::<std::collections::BTreeMap<_, _>>(),
    )
}

#[test]
fn render_tck_maps_match_tck_form() {
    // Empty map (Literals8 [1]).
    assert_eq!(render_tck(&map_of([])), "{}");
    // Single entry, bare key, `: ` separator (Literals8 [2]/[8]).
    assert_eq!(
        render_tck(&map_of([("abc", Value::Integer(1))])),
        "{abc: 1}"
    );
    // Scalar value kinds inside a map (Literals8 [6]/[7]).
    assert_eq!(
        render_tck(&map_of([("k", Value::Boolean(false))])),
        "{k: false}"
    );
    assert_eq!(render_tck(&map_of([("k", Value::Null)])), "{k: null}");
    // String value renders recursively (single-quoted) (Literals8 [12]).
    assert_eq!(
        render_tck(&map_of([("k", Value::String("ab: c".into()))])),
        "{k: 'ab: c'}"
    );
    // Multi-entry: BTreeMap sorted-key order, `, ` between entries
    // (Literals8 [17] keys `a`,`c`,`d` are already sorted).
    let multi = map_of([
        ("a", Value::String(" { b : ".into())),
        ("c", map_of([("d", Value::String(" ".into()))])),
        ("d", Value::String(" } ".into())),
    ]);
    assert_eq!(render_tck(&multi), "{a: ' { b : ', c: {d: ' '}, d: ' } '}");
    // Nested map recursion, 3-deep (Literals8 [13]/[14] family).
    let nested = map_of([("a1", map_of([("a2", map_of([("a3", map_of([]))]))]))]);
    assert_eq!(render_tck(&nested), "{a1: {a2: {a3: {}}}}");
    // List-of-map + map-containing-list both recurse through render_tck
    // (Literals7 [13] `[{}]`; Literals8 [18]-shape `{data: [...]}`).
    assert_eq!(render_tck(&Value::List(vec![map_of([])])), "[{}]");
    assert_eq!(
        render_tck(&map_of([(
            "data",
            Value::List(vec![map_of([("id", Value::String("0001".into()))])]),
        )])),
        "{data: [{id: '0001'}]}"
    );
}

#[test]
fn render_tck_float_matches_tck_form() {
    // Whole-valued floats carry `.0` (Literals5 [1]/[7]/[8]/[13]/[14]/[15]).
    assert_eq!(render_tck(&Value::Float(1.0)), "1.0");
    assert_eq!(render_tck(&Value::Float(0.0)), "0.0");
    assert_eq!(render_tck(&Value::Float(1e9)), "1000000000.0");
    assert_eq!(render_tck(&Value::Float(1e8)), "100000000.0");
    assert_eq!(render_tck(&Value::Float(-1e9)), "-1000000000.0");
    // Signed zero normalizes to `0.0` (Literals5 [9]/[10]).
    assert_eq!(render_tck(&Value::Float(-0.0)), "0.0");
    // Small negative-exponent band stays PLAIN decimal — Rust Debug would
    // scientific-ize these (Literals5 [16]/[17]/[18]/[22]/[23]/[24]).
    assert_eq!(render_tck(&Value::Float(1e-5)), "0.00001");
    assert_eq!(render_tck(&Value::Float(1e-6)), "0.000001");
    assert_eq!(render_tck(&Value::Float(-1e-5)), "-0.00001");
    assert_eq!(render_tck(&Value::Float(-1e-6)), "-0.000001");
    // Fractional floats: shortest round-trip (Literals5 [2]/[3]/[4]).
    assert_eq!(render_tck(&Value::Float(0.1)), "0.1");
    assert_eq!(render_tck(&Value::Float(0.55)), "0.55");
    assert_eq!(
        render_tck(&Value::Float(3985764.3405892686)),
        "3985764.3405892686"
    );
    // Magnitude EXTREMES keep the scientific form (Literals5 [5]/[6]/[25]/[26]).
    assert_eq!(render_tck(&Value::Float(1e-305)), "1e-305");
    assert_eq!(render_tck(&Value::Float(-1e-305)), "-1e-305");
    assert_eq!(
        render_tck(&Value::Float(1.2635418652381264e305)),
        "1.2635418652381264e305"
    );
    assert_eq!(render_tck(&Value::Float(1e308)), "1e308");
    assert_eq!(render_tck(&Value::Float(1.23456789e308)), "1.23456789e308");
    // Non-finite (defensive; not in the literal corpus but the arm exists).
    assert_eq!(render_tck(&Value::Float(f64::NAN)), "NaN");
    assert_eq!(render_tck(&Value::Float(f64::INFINITY)), "Infinity");
    assert_eq!(render_tck(&Value::Float(f64::NEG_INFINITY)), "-Infinity");
}

#[test]
fn render_tck_string_escapes_match_tck_form() {
    // No escaping needed.
    assert_eq!(render_tck(&Value::String("foo".into())), "'foo'");
    assert_eq!(render_tck(&Value::String(String::new())), "''");
    // Single-quote escaped to `\'` (Literals6 [4]: engine value is `'`).
    assert_eq!(render_tck(&Value::String("'".into())), "'\\''");
    // Backslash escaped to `\\`.
    assert_eq!(render_tck(&Value::String("\\".into())), "'\\\\'");
    // Mixed backslash + single-quote; double-quotes are NOT escaped.
    // engine value `a\b'"c` -> `'a\\b\'"c'`.
    assert_eq!(
        render_tck(&Value::String("a\\b'\"c".into())),
        "'a\\\\b\\'\"c'"
    );
}

#[test]
fn not_implemented_is_not_a_correct_rejection() {
    // The honesty-critical case: a query the engine cannot RUN must NOT
    // count as a correctly-REJECTED error scenario.
    let err = ExplainError::ArcQL(ArcQLError::NotImplemented {
        feature: "execute: synthetic".into(),
        section: "test".into(),
        target_version: "v1.1".into(),
        span: arcgraph_query::Span::point(1, 1),
    });
    assert_eq!(classify_engine_error(&err), EngineErrClass::Unsupported);
    assert!(!phase_satisfied(
        ErrorPhase::Compile,
        EngineErrClass::Unsupported
    ));
    assert!(!phase_satisfied(
        ErrorPhase::Runtime,
        EngineErrClass::Unsupported
    ));
    assert!(!phase_satisfied(
        ErrorPhase::Any,
        EngineErrClass::Unsupported
    ));
}

#[test]
fn parse_expectation_recognizes_each_then_shape() {
    let mk = |value: &str| Step {
        keyword: "Then ".into(),
        ty: cucumber::gherkin::StepType::Then,
        value: value.into(),
        docstring: None,
        table: None,
        span: Default::default(),
        position: Default::default(),
    };
    assert_eq!(
        parse_expectation(&mk("the result should be empty")),
        Some(Expectation::Empty)
    );
    assert_eq!(
        parse_expectation(&mk(
            "a SyntaxError should be raised at compile time: UndefinedVariable"
        )),
        Some(Expectation::Error(ErrorPhase::Compile))
    );
    assert_eq!(
        parse_expectation(&mk(
            "a TypeError should be raised at runtime: InvalidArgumentValue"
        )),
        Some(Expectation::Error(ErrorPhase::Runtime))
    );
    assert_eq!(
        parse_expectation(&mk("an Error should be raised at any time: *")),
        Some(Expectation::Error(ErrorPhase::Any))
    );
    // `no side effects` is not an expectation.
    assert_eq!(parse_expectation(&mk("no side effects")), None);
}

#[test]
fn parse_expectation_rows_strips_header_and_picks_mode() {
    let table = cucumber::gherkin::Table {
        rows: vec![vec!["literal".to_string()], vec!["true".to_string()]],
        span: Default::default(),
        position: Default::default(),
    };
    let step = Step {
        keyword: "Then ".into(),
        ty: cucumber::gherkin::StepType::Then,
        value: "the result should be, in any order:".into(),
        docstring: None,
        table: Some(table),
        span: Default::default(),
        position: Default::default(),
    };
    match parse_expectation(&step) {
        Some(Expectation::Rows { mode, data }) => {
            assert_eq!(mode, CompareMode::Multiset);
            assert_eq!(data, vec![vec!["true".to_string()]]);
        }
        other => panic!("expected Rows, got {other:?}"),
    }
}

#[test]
fn header_only_result_table_is_empty_expectation() {
    let table = cucumber::gherkin::Table {
        rows: vec![vec!["n".to_string()]], // header only, no data rows
        span: Default::default(),
        position: Default::default(),
    };
    let step = Step {
        keyword: "Then ".into(),
        ty: cucumber::gherkin::StepType::Then,
        value: "the result should be, in any order:".into(),
        docstring: None,
        table: Some(table),
        span: Default::default(),
        position: Default::default(),
    };
    assert_eq!(parse_expectation(&step), Some(Expectation::Empty));
}

#[test]
fn outline_substitution_replaces_placeholders() {
    let headers = vec!["pattern".to_string()];
    let values = vec!["()-[r]-()".to_string()];
    assert_eq!(
        substitute("MATCH <pattern>\nRETURN r", &headers, &values),
        "MATCH ()-[r]-()\nRETURN r"
    );
}
