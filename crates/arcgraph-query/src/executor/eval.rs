//! BoundExpression evaluator under Cypher 3VL.
//!
//! [`evaluate`] walks a [`BoundExpression`] against the current row +
//! the per-query parameter bag, returning a [`Value`]. Predicates
//! lift via [`crate::executor::ThreeValued::from_value`] at the
//! WHERE / JOIN-ON / OPTIONAL-MATCH-WHERE consumer sites.
//!
//! # Scope
//!
//! v1.0-alpha admits the subset of [`BoundExpression`] variants the
//! M4-31 / M4-32 / M4-33 lowering can route through MATCH / WHERE /
//! WITH / RETURN at the four simple operators (Scan / Expand /
//! Filter / Project) PLUS the M4-62 hybrid + OPTIONAL MATCH 3VL
//! sites. Variants that the planner emits but the executor does not
//! evaluate yet (e.g., aggregation function calls — those are M4-63)
//! return [`crate::executor::ExecutionError::NotImplemented`] from
//! the operator that would have driven them — the eval routine
//! itself doesn't know those forward-deferral edges; the operator
//! does.
//!
//! # Schema indirection
//!
//! Each row is a `Vec<Value>` indexed by a CONSECUTIVE column index;
//! the executor's `Schema` maps `BindingId` → column index. The
//! [`evaluate`] entrypoint takes a schema lookup closure so callers
//! can supply different schemas (per-batch in Filter / Project,
//! pre-merge in Join).
//!
//! # ADR provenance
//! - **ADR-038 §2 D-20** — 3VL truth tables.
//! - **ADR-038 amendment-03 §TIER-2-b** — M4-62 3VL implementation.

use std::collections::{BTreeMap, HashMap};

use crate::ast::{BinOp, Expression, Literal, Quantifier, UnaryOp};
use crate::executor::error::ExecutionError;
use crate::executor::three_vl::ThreeValued;
use crate::executor::value::Value;
use crate::semantic::bound_ast::{BindingId, BoundExpression, BoundMapProjectionItem};

/// Schema lookup closure type. Maps `BindingId` → column-index in
/// the per-row `Vec<Value>`. Returns `None` if the binding is not
/// represented in the row (which would be a planner bug — every
/// referenced binding must have a schema slot).
pub type SchemaLookup<'a> = &'a dyn Fn(BindingId) -> Option<usize>;

/// Per-query parameter bag. Maps parameter name → value (string-keyed
/// so it's stable across compilation runs). Constructed by callers
/// that drive `EXPLAIN(stmt) WITH PARAMETERS(...)` style entry points
/// (M5-12 forward); v1.0-alpha tests pass an empty bag for queries
/// that don't reference parameters and pre-populate the bag for
/// queries that do.
pub type Parameters = HashMap<String, Value>;

/// Evaluate a [`BoundExpression`] against the current row.
///
/// Returns the cell value. NULL propagates per Cypher 3VL: any NULL
/// operand reaching a comparison / arithmetic site yields
/// [`Value::Null`]; AND/OR/NOT route through
/// [`crate::executor::ThreeValued`].
pub fn evaluate(
    expr: &BoundExpression,
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    match expr {
        // Literals.
        BoundExpression::Literal { value, .. } => Ok(literal_to_value(value)),
        BoundExpression::ListLiteral { elements, .. } => elements
            .iter()
            .map(|element| evaluate(element, row, schema, params))
            .collect::<Result<Vec<_>, _>>()
            .map(Value::List),
        BoundExpression::MapLiteral { entries, .. } => {
            let mut map = BTreeMap::new();
            for (key, value_expr) in entries {
                map.insert(key.clone(), evaluate(value_expr, row, schema, params)?);
            }
            Ok(Value::Map(map))
        }

        // Parameters (#797). Resolved against the per-query bag the
        // entry point installed on the ExecutionContext. A missing bind
        // is a CLIENT fault → typed `MissingParameter` (lifted to a
        // client-class wire error), never a silent NULL.
        BoundExpression::Parameter { name, .. } => params
            .get(name)
            .cloned()
            .ok_or_else(|| ExecutionError::MissingParameter { name: name.clone() }),

        // Variable references.
        BoundExpression::VariableRef { binding_id, .. } => {
            let idx = schema(*binding_id).ok_or_else(|| {
                ExecutionError::Eval(format!("binding {:?} missing from row schema", binding_id))
            })?;
            Ok(row.get(idx).cloned().unwrap_or(Value::Null))
        }
        BoundExpression::UnresolvedVariable { name, .. } => Err(ExecutionError::Eval(format!(
            "unresolved variable `{name}` reached executor"
        ))),

        // Property access: `n.prop` / `r.prop`.
        BoundExpression::PropertyAccess { base, path, .. } => {
            let base_val = evaluate(base, row, schema, params)?;
            // Walk the property path. v1.0 lit only single-segment
            // access reliably; multi-segment is admitted but resolves
            // to NULL on first miss.
            let mut current = base_val;
            for segment in path {
                current = match &current {
                    Value::Node(n) => n
                        .properties
                        .get(&segment.name)
                        .cloned()
                        .unwrap_or(Value::Null),
                    Value::Relationship(r) => r
                        .properties
                        .get(&segment.name)
                        .cloned()
                        .unwrap_or(Value::Null),
                    // ADR-191 D-8 — `map.key` resolves the key; a missing
                    // key yields `null` (matching the Node/Rel arms).
                    // Nested access (`{x: {y: 5}}.x.y`) walks one segment
                    // per loop iteration. The dynamic `map[expr]` subscript
                    // form shares this resolution but needs the `[expr]`
                    // accessor grammar (deferred to #621 / PR-B).
                    Value::Map(m) => m.get(&segment.name).cloned().unwrap_or(Value::Null),
                    Value::Null => Value::Null,
                    _ => {
                        return Err(ExecutionError::Eval(
                            "property access on non-entity value".into(),
                        ));
                    }
                };
            }
            Ok(current)
        }

        // Binary / unary / IN / IS NULL operator spines — evaluated
        // through one iterative driver (#1290): a flat operator chain
        // folds into a left-nested spine up to `MAX_FLAT_CHAIN_DEPTH`
        // deep, and the spine may interleave all four variants
        // (`a = 1 IN [true] IS NULL …`), so recursing per level — or
        // despining only `BinaryOp` as the first cut of this fix did —
        // overflowed the native stack.
        BoundExpression::BinaryOp { .. }
        | BoundExpression::UnaryOp { .. }
        | BoundExpression::In { .. }
        | BoundExpression::IsNull { .. } => {
            evaluate_left_nested_operator_spine(expr, row, schema, params)
        }

        // Hybrid retrieval predicate-lifts — these are normally
        // lowered to [`LogicalRankByHybrid`] / [`LogicalVectorNear`] /
        // [`LogicalTextMatch`] in the plan, so reaching them here in
        // an expression context means a planner edge case left them
        // un-lowered. v1.0-alpha returns `NotImplemented` so the bug
        // surfaces loudly.
        BoundExpression::Near { .. } => Err(ExecutionError::NotImplemented {
            feature: "Near (un-lowered NEAR predicate in expression)".into(),
            target_slice: "M4-32 / M4-62 lowering".into(),
            section: "ADR-038 §2 D-5".into(),
        }),
        BoundExpression::TextMatch { .. } => Err(ExecutionError::NotImplemented {
            feature: "TextMatch (un-lowered MATCH predicate in expression)".into(),
            target_slice: "M4-32 / M4-62 lowering".into(),
            section: "ADR-038 §2 D-6".into(),
        }),
        BoundExpression::InCommunity { .. } => Err(ExecutionError::NotImplemented {
            feature: "InCommunity (un-lowered IN COMMUNITY predicate)".into(),
            target_slice: "M4-32 / M4-62 lowering".into(),
            section: "ADR-038 amendment-01 §A-1".into(),
        }),

        // Function calls — v1.0-alpha lights only a small subset
        // sufficient for executor smoke (id, exists, length, ...).
        // Extend conservatively in M4-63.
        BoundExpression::FunctionCall { name, args, .. } => {
            apply_function(name, args, row, schema, params)
        }

        // ADR-188 Decision 1 + 4 — list-predicate (`all`/`any`/`none`/
        // `single`) via per-element extended-row synthesis + 3VL fold.
        BoundExpression::ListPredicate {
            quantifier,
            var_bid,
            list,
            predicate,
            ..
        } => eval_list_predicate(*quantifier, *var_bid, list, predicate, row, schema, params),

        // ADR-188 Decision 1 + 4 — `reduce` pure left-fold (TWO scoped
        // slots; no 3VL short-circuit; null propagates as an ordinary
        // value).
        BoundExpression::Reduce {
            acc_bid,
            init,
            var_bid,
            list,
            expr,
            ..
        } => eval_reduce(*acc_bid, init, *var_bid, list, expr, row, schema, params),

        // ADR-188 Decision 1 + 5 — list comprehension (#620 list-half)
        // `[x IN list WHERE p | e]` via the SAME per-element extended-row
        // synthesis: filter each element by the optional 3VL predicate
        // (only `true` keeps it), project the optional expression
        // (identity when absent), and collect into a `Value::List`.
        BoundExpression::ListComprehension {
            var_bid,
            list,
            predicate,
            projection,
            ..
        } => eval_list_comprehension(
            *var_bid,
            list,
            predicate.as_deref(),
            projection.as_deref(),
            row,
            schema,
            params,
        ),

        // ADR-191 D-6 (#620 map-half) — map projection `n{.k, alias: e, .*}`:
        // build a NEW `Value::Map` by selecting keys from the base entity /
        // map per the D-6 null-handling split.
        BoundExpression::MapProjection { base, items, .. } => {
            eval_map_projection(base, items, row, schema, params)
        }

        // openCypher v9 §3.4 — list subscript `base[index]`.
        BoundExpression::Subscript { base, index, .. } => {
            let base_v = evaluate(base, row, schema, params)?;
            let index_v = evaluate(index, row, schema, params)?;
            eval_subscript(base_v, index_v)
        }

        // openCypher v9 §3.4 — list slice `base[start..end]`.
        BoundExpression::Slice {
            base, start, end, ..
        } => {
            let base_v = evaluate(base, row, schema, params)?;
            let start_v = start
                .as_deref()
                .map(|s| evaluate(s, row, schema, params))
                .transpose()?;
            let end_v = end
                .as_deref()
                .map(|e| evaluate(e, row, schema, params))
                .transpose()?;
            eval_slice(base_v, start_v, end_v)
        }

        // openCypher v9 §3.6 (#621) — CASE expression (simple + searched),
        // short-circuiting on the first matching branch.
        BoundExpression::Case {
            test,
            branches,
            default,
            ..
        } => eval_case(
            test.as_deref(),
            branches,
            default.as_deref(),
            row,
            schema,
            params,
        ),
    }
}

/// #1290 — evaluate a left-nested OPERATOR SPINE iteratively.
///
/// Walks down the left/operand edge collecting one frame per spine
/// level (`BinaryOp` / `UnaryOp` / `In` / `IsNull` — the grammar's
/// comparison repetition interleaves them freely), evaluates the
/// non-spine base via the ordinary [`evaluate`] arms, then folds the
/// frames back up applying each level's operator — the same
/// left-associative evaluation order (and identical 3VL semantics per
/// operator) as the recursive arms this replaces. `rhs` operands
/// evaluate recursively — they are never part of the LEFT spine, so
/// their depth is bounded by the bracket cap (`MAX_EXPRESSION_DEPTH`),
/// not the chain cap.
fn evaluate_left_nested_operator_spine(
    expr: &BoundExpression,
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    enum SpineFrame<'a> {
        Binary { op: BinOp, rhs: &'a BoundExpression },
        Unary { op: UnaryOp },
        In { rhs: &'a BoundExpression },
        IsNull { negated: bool },
    }
    let mut spine: Vec<SpineFrame<'_>> = Vec::new();
    let mut base = expr;
    loop {
        match base {
            BoundExpression::BinaryOp { op, lhs, rhs, .. } => {
                spine.push(SpineFrame::Binary {
                    op: op.clone(),
                    rhs,
                });
                base = lhs;
            }
            BoundExpression::UnaryOp { op, operand, .. } => {
                spine.push(SpineFrame::Unary { op: op.clone() });
                base = operand;
            }
            BoundExpression::In { lhs, rhs, .. } => {
                spine.push(SpineFrame::In { rhs });
                base = lhs;
            }
            BoundExpression::IsNull { lhs, negated, .. } => {
                spine.push(SpineFrame::IsNull { negated: *negated });
                base = lhs;
            }
            _ => break,
        }
    }

    let mut acc = evaluate(base, row, schema, params)?;
    while let Some(frame) = spine.pop() {
        acc = match frame {
            SpineFrame::Binary { op, rhs } => {
                let r = evaluate(rhs, row, schema, params)?;
                apply_binop(op, acc, r)?
            }
            SpineFrame::Unary { op } => apply_unop(op, acc)?,
            SpineFrame::In { rhs } => {
                let haystack = evaluate(rhs, row, schema, params)?;
                apply_in_list(acc, haystack)?
            }
            SpineFrame::IsNull { negated } => {
                // IS NULL / IS NOT NULL — tunnel through 3VL per
                // Cypher 9 §6.2.5. IS NOT NULL flips the result (still
                // 2-valued — IS NULL never returns Unknown).
                let result = ThreeValued::is_null(&acc);
                let final_tv = if negated { result.not() } else { result };
                threevalued_to_value(final_tv)
            }
        };
    }
    Ok(acc)
}

/// IN list — Cypher 9 §3.3.5 3VL list-membership over openCypher
/// VALUE equality ([`values_equal_3vl`], NOT Rust `PartialEq`): the
/// needle is compared to each element via DEEP 3VL equality, then
/// folded — ANY definite match ⇒ TRUE (short-circuit); else ANY
/// `Unknown` (a comparison that is null, whether top-level or
/// nested, e.g. `[1,2] = [null,2]`) ⇒ NULL; else FALSE. A NULL
/// needle ⇒ NULL; an empty list ⇒ FALSE. The full TCK `List5`
/// oracle (`[1,2] IN [[null,2],[1,3]]` ⇒ null, `[1,2,null] IN
/// [1,[1,2,null]]` ⇒ null, type-mismatch ⇒ false-not-null, …) is
/// pinned by the unit tests + the conformance ratchet.
fn apply_in_list(needle: Value, haystack: Value) -> Result<Value, ExecutionError> {
    if needle.is_null() {
        return Ok(Value::Null);
    }
    match haystack {
        Value::List(elems) => {
            let mut saw_unknown = false;
            for elem in &elems {
                match values_equal_3vl(&needle, elem) {
                    Some(true) => return Ok(Value::Boolean(true)),
                    None => saw_unknown = true,
                    Some(false) => {}
                }
            }
            if saw_unknown {
                Ok(Value::Null)
            } else {
                Ok(Value::Boolean(false))
            }
        }
        Value::Null => Ok(Value::Null),
        _ => Err(ExecutionError::Eval("IN rhs must be a list".into())),
    }
}

/// **openCypher v9 §3.6** (#621) — evaluate a CASE expression in BOTH forms,
/// SHORT-CIRCUITING on the first matching branch. Only the taken branch's
/// THEN (or the ELSE, on no match) is ever evaluated — non-taken THEN/ELSE
/// arms are never touched, so `CASE WHEN true THEN 1 ELSE 1/0 END` does NOT
/// divide by zero.
///
/// - **SIMPLE** (`test = Some`): evaluate `test` once, then for each branch
///   in order compare its WHEN value to the test via openCypher VALUE
///   equality ([`values_equal_3vl`] — the SAME helper `=` / `IN` use). A
///   branch matches IFF the comparison is `Some(true)`. A definite mismatch
///   (`Some(false)` — e.g. the type-mismatched `'0' = 0`, or `true = 1`) and
///   a null-involving comparison (`None` — e.g. a null `test`) are BOTH
///   non-matches that fall through to the next branch / ELSE. This is the
///   load-bearing Conditional2 [1] semantic: `'0'` / `true` / `10.1`
///   compared against integer WHENs land on ELSE, NOT an error; and a null
///   test matches no WHEN (the safe openCypher behaviour).
/// - **SEARCHED** (`test = None`): evaluate each WHEN as a Cypher 3VL boolean
///   condition; a branch matches IFF the condition is `Boolean(true)`. A
///   `null` / `false` / non-boolean condition does NOT match — the SAME
///   WHERE-filter discipline via [`ThreeValued::from_value`] +
///   [`ThreeValued::passes_filter`] (`Unknown`/`False` ⇒ not taken).
///
/// No branch matches ⇒ the ELSE (`default`), or [`Value::Null`] when absent.
fn eval_case(
    test: Option<&BoundExpression>,
    branches: &[(BoundExpression, BoundExpression)],
    default: Option<&BoundExpression>,
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    match test {
        // SIMPLE form — equality dispatch over the evaluated test value.
        Some(test_expr) => {
            let test_val = evaluate(test_expr, row, schema, params)?;
            for (when, then) in branches {
                let when_val = evaluate(when, row, schema, params)?;
                // A match is ONLY a definite `Some(true)`; `Some(false)`
                // (type/value mismatch) and `None` (null-involving) fall
                // through — REUSING the `=` / `IN` value-equality helper, NOT
                // a new equality.
                if values_equal_3vl(&test_val, &when_val) == Some(true) {
                    return evaluate(then, row, schema, params);
                }
            }
        }
        // SEARCHED form — 3VL truthiness over each WHEN condition.
        None => {
            for (when, then) in branches {
                let cond = evaluate(when, row, schema, params)?;
                // Only `Boolean(true)` matches; `null`/`false`/non-boolean
                // are NOT taken (the WHERE-filter 3VL discipline).
                if ThreeValued::from_value(&cond).passes_filter() {
                    return evaluate(then, row, schema, params);
                }
            }
        }
    }
    // No branch matched → the ELSE default, or `null` when ELSE is absent.
    match default {
        Some(d) => evaluate(d, row, schema, params),
        None => Ok(Value::Null),
    }
}

/// **ADR-188 Decision 1 + 4** — evaluate a list-predicate
/// (`all`/`any`/`none`/`single`) over `list`, binding each element to
/// `var_bid` in a synthesized extended row and folding the per-element
/// predicate result through Cypher 3VL.
///
/// # Mechanism (per-element extended-row synthesis, reused scratch buffer)
///
/// We build the *extended row* = the current row with one appended
/// slot at `slot = row.len()` for the iteration variable, and a
/// *scoped schema closure* that returns `slot` for `var_bid` and
/// delegates every other binding to the outer `schema`. The extended
/// row is allocated **once** (a single `Vec<Value>` grown to
/// `row.len() + 1`) **before** the element loop; per element we
/// **overwrite** `ext[slot] = element.clone()` in place — there is
/// **no per-element heap allocation** (ADR-188 Decision 1 §BoE proviso
/// (2): "no per-element heap allocation in the scalar case" — the
/// element clone is a `memcpy` of a ≤32-byte scalar `Value`; the
/// backing `Vec` is reused across all N elements, not re-allocated).
/// The inner predicate is evaluated against that extended row via the
/// unchanged [`evaluate`] entrypoint — every variable (row-bound or
/// scoped) resolves through the single `binding_id → column-index →
/// row[idx]` path. Nested predicates `all(x IN l1 WHERE any(y IN l2
/// …))` allocate their OWN buffer (one fresh `Vec` per nesting-level
/// *invocation*, not per element) extending the already-extended row,
/// so the inner scoped var `y` lands at `ext.len()` = one past the
/// outer slot `x` for free, and the binder's reverse scope-walk gives
/// inner-shadows-outer. Short-circuit on the first definite witness
/// bounds the common case below N.
#[allow(clippy::too_many_arguments)]
fn eval_list_predicate(
    quantifier: Quantifier,
    var_bid: BindingId,
    list: &BoundExpression,
    predicate: &BoundExpression,
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    // Null list ⇒ null (Decision 4, "null L (any form)").
    let elems = match evaluate(list, row, schema, params)? {
        Value::Null => return Ok(Value::Null),
        Value::List(xs) => xs,
        other => {
            return Err(ExecutionError::Eval(format!(
                "list predicate over non-list value: {other:?}"
            )));
        }
    };

    // Empty-list results (Decision 4): all/none = true, any = false,
    // single = false (the standard vacuous-quantifier results).
    if elems.is_empty() {
        return Ok(Value::Boolean(match quantifier {
            Quantifier::All | Quantifier::None => true,
            Quantifier::Any | Quantifier::Single => false,
        }));
    }

    let slot = row.len();
    // The scoped schema closure: `var_bid` → the appended slot, else
    // delegate to the outer schema. Nested predicates re-wrap over the
    // already-extended row, so an inner closure delegates to THIS one
    // for the outer scoped var.
    let scoped =
        move |b: BindingId| -> Option<usize> { if b == var_bid { Some(slot) } else { schema(b) } };

    // ADR-188 Decision 1 — REUSED scratch buffer: allocate the extended
    // row ONCE (= `row` + one placeholder slot for the scoped var), then
    // OVERWRITE `ext[slot]` per element. No per-element heap allocation
    // (BoE proviso (2)); the only per-element cost is the scalar clone.
    let mut ext = extended_row(row, 1);

    match quantifier {
        // `all` = universal: false if any P_i=false; else null if any
        // P_i=null; else true. Short-circuit on the first definite
        // false.
        Quantifier::All => {
            let mut saw_null = false;
            for x in &elems {
                ext[slot] = x.clone();
                match three_valued_predicate(predicate, &ext, &scoped, params)? {
                    ThreeValued::False => return Ok(Value::Boolean(false)),
                    ThreeValued::Unknown => saw_null = true,
                    ThreeValued::True => {}
                }
            }
            Ok(if saw_null {
                Value::Null
            } else {
                Value::Boolean(true)
            })
        }
        // `any` = existential: true if any P_i=true; else null if any
        // P_i=null; else false. Short-circuit on the first definite
        // true.
        Quantifier::Any => {
            let mut saw_null = false;
            for x in &elems {
                ext[slot] = x.clone();
                match three_valued_predicate(predicate, &ext, &scoped, params)? {
                    ThreeValued::True => return Ok(Value::Boolean(true)),
                    ThreeValued::Unknown => saw_null = true,
                    ThreeValued::False => {}
                }
            }
            Ok(if saw_null {
                Value::Null
            } else {
                Value::Boolean(false)
            })
        }
        // `none` = negated existential (≡ NOT any): false if any
        // P_i=true; else null if any P_i=null; else true. Short-circuit
        // on the first definite true.
        Quantifier::None => {
            let mut saw_null = false;
            for x in &elems {
                ext[slot] = x.clone();
                match three_valued_predicate(predicate, &ext, &scoped, params)? {
                    ThreeValued::True => return Ok(Value::Boolean(false)),
                    ThreeValued::Unknown => saw_null = true,
                    ThreeValued::False => {}
                }
            }
            Ok(if saw_null {
                Value::Null
            } else {
                Value::Boolean(true)
            })
        }
        // `single` = exactly-one, with the PE-corrected NULL semantics
        // (Decision 4-single — BINDING):
        //   - false  if TWO OR MORE P_i=true (definitely not unique —
        //     a definite witness count ≥2 dominates any nulls);
        //   - true   if EXACTLY ONE P_i=true (a definite single witness
        //     dominates any nulls — the null does NOT speculatively
        //     count as a second match; cf. `2 IN [1,2,null] ⇒ true`),
        //     regardless of how many other elements are null/false;
        //   - null   if ZERO P_i=true AND at least one P_i=null (no
        //     definite witness; an unknown could be the single true);
        //   - false  if ZERO P_i=true and all others false (non-empty).
        // We CANNOT short-circuit on the first true (we must distinguish
        // exactly-one from two-or-more), but we CAN short-circuit on the
        // second true (≥2 ⇒ definitely false, dominates any nulls).
        Quantifier::Single => {
            let mut true_count: usize = 0;
            let mut saw_null = false;
            for x in &elems {
                ext[slot] = x.clone();
                match three_valued_predicate(predicate, &ext, &scoped, params)? {
                    ThreeValued::True => {
                        true_count += 1;
                        if true_count >= 2 {
                            // Two definite witnesses dominate any nulls.
                            return Ok(Value::Boolean(false));
                        }
                    }
                    ThreeValued::Unknown => saw_null = true,
                    ThreeValued::False => {}
                }
            }
            Ok(match (true_count, saw_null) {
                // Exactly one definite witness dominates any nulls.
                (1, _) => Value::Boolean(true),
                // Zero definite witnesses + a null could be THE single
                // match ⇒ genuinely unknown.
                (0, true) => Value::Null,
                // Zero witnesses, no nulls ⇒ definitely not exactly-one.
                (0, false) => Value::Boolean(false),
                // (≥2 already returned false above.)
                _ => unreachable!("true_count ≥ 2 short-circuits to false"),
            })
        }
    }
}

/// **ADR-188 Decision 1 + 4** — evaluate `reduce(acc = init, x IN list
/// | expr)` as a PURE left-fold. No 3VL short-circuit; a `null`
/// produced by the body is an ordinary value that flows on. Null list
/// ⇒ null; empty list ⇒ init.
///
/// Two extended-row slots on a REUSED scratch buffer (ADR-188 Decision
/// 1): `acc` at `row.len()` and `x` at `row.len() + 1`. The extended
/// row is allocated **once** before the fold (a single `Vec<Value>`
/// grown to `row.len() + 2`); each iteration **overwrites**
/// `ext[acc_slot] = acc.clone()` (the running accumulator) and
/// `ext[x_slot] = x.clone()` (the current element) in place — **no
/// per-element heap allocation** (BoE proviso (2)).
#[allow(clippy::too_many_arguments)]
fn eval_reduce(
    acc_bid: BindingId,
    init: &BoundExpression,
    var_bid: BindingId,
    list: &BoundExpression,
    expr: &BoundExpression,
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    // Null list ⇒ null (Decision 4). Evaluate the list FIRST (before
    // init side effects) — though eval is pure, this matches the
    // table's "null L ⇒ null" precedence.
    let elems = match evaluate(list, row, schema, params)? {
        Value::Null => return Ok(Value::Null),
        Value::List(xs) => xs,
        other => {
            return Err(ExecutionError::Eval(format!(
                "reduce over non-list value: {other:?}"
            )));
        }
    };

    // acc starts at init; empty list ⇒ init unchanged (Decision 4).
    let mut acc = evaluate(init, row, schema, params)?;

    let acc_slot = row.len();
    let x_slot = row.len() + 1;
    let scoped = move |b: BindingId| -> Option<usize> {
        if b == acc_bid {
            Some(acc_slot)
        } else if b == var_bid {
            Some(x_slot)
        } else {
            schema(b)
        }
    };

    // ADR-188 Decision 1 — REUSED scratch buffer: allocate the extended
    // row ONCE (= `row` + two placeholder slots for `acc` and `x`), then
    // OVERWRITE both slots in place each iteration. `acc`'s slot carries
    // the running fold value; `x`'s slot the current element. No
    // per-iteration heap allocation (BoE proviso (2)).
    let mut ext = extended_row(row, 2);

    for x in &elems {
        ext[acc_slot] = acc.clone();
        ext[x_slot] = x.clone();
        acc = evaluate(expr, &ext, &scoped, params)?;
    }
    Ok(acc)
}

/// **ADR-188 Decision 1 + 5** (#620 list-half) — evaluate a list
/// comprehension `[x IN list WHERE predicate | projection]` (openCypher
/// v9 §3.5). For each element `x` of `list`, in order: bind `x` to
/// `var_bid` in the synthesized extended row (the SAME per-element
/// extended-row synthesis as [`eval_list_predicate`] — one slot at
/// `row.len()`, a REUSED scratch buffer overwritten per element, no
/// per-element heap allocation), apply the optional `predicate` as a
/// **3VL filter** (only `ThreeValued::True` keeps the element — a
/// `null` or `false` predicate result filters it OUT, per openCypher v9
/// §3.5's "where the predicate holds" with Cypher 3VL), and push the
/// optional `projection`'s value (or the element itself — identity —
/// when `projection` is absent) into the result list.
///
/// Edge cases (ADR-188 Decision 4 + openCypher v9 §3.5):
/// - **null list ⇒ `Value::Null`** (the null-list rule shared with
///   `ListPredicate` / `reduce`).
/// - **empty list ⇒ empty `Value::List`** (map/filter over no elements).
/// - **`predicate = null` for an element ⇒ that element is filtered
///   OUT** (only definite `true` passes — the 3VL `Unknown` does not
///   keep the element).
/// - **nested** `[x IN l1 | [y IN l2 | …]]` works for free — the inner
///   comprehension allocates its own buffer extending the
///   already-extended row, so `y` lands one slot past `x` and the
///   binder's reverse scope-walk gives inner-shadows-outer.
#[allow(clippy::too_many_arguments)]
fn eval_list_comprehension(
    var_bid: BindingId,
    list: &BoundExpression,
    predicate: Option<&BoundExpression>,
    projection: Option<&BoundExpression>,
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    // Null list ⇒ null (Decision 4, "null L (any form)").
    let elems = match evaluate(list, row, schema, params)? {
        Value::Null => return Ok(Value::Null),
        Value::List(xs) => xs,
        other => {
            return Err(ExecutionError::Eval(format!(
                "list comprehension over non-list value: {other:?}"
            )));
        }
    };

    // Empty list ⇒ empty list (map/filter over no elements).
    if elems.is_empty() {
        return Ok(Value::List(Vec::new()));
    }

    let slot = row.len();
    // The scoped schema closure: `var_bid` → the appended slot, else
    // delegate to the outer schema. A nested inner comprehension
    // re-wraps over the already-extended row, delegating to THIS closure
    // for the outer scoped var.
    let scoped =
        move |b: BindingId| -> Option<usize> { if b == var_bid { Some(slot) } else { schema(b) } };

    // ADR-188 Decision 1 — REUSED scratch buffer: allocate the extended
    // row ONCE (= `row` + one placeholder slot for the scoped var), then
    // OVERWRITE `ext[slot]` per element. No per-element heap allocation
    // for the buffer; the only per-element cost is the scalar clone (and
    // the projection eval).
    let mut ext = extended_row(row, 1);
    // The result preserves source order (openCypher v9 §3.5). We do not
    // pre-size to `elems.len()` because the optional filter may drop
    // elements; the `Vec` grows amortized-O(1).
    let mut out: Vec<Value> = Vec::new();

    for x in &elems {
        ext[slot] = x.clone();
        // Optional WHERE filter: keep the element ONLY if the predicate
        // is definitely TRUE (3VL — `null`/`false` filter it out).
        if let Some(pred) = predicate {
            match three_valued_predicate(pred, &ext, &scoped, params)? {
                ThreeValued::True => {}
                ThreeValued::False | ThreeValued::Unknown => continue,
            }
        }
        // Project: the `| projection` value if present, else the element
        // itself (identity). The projection sees the SAME extended row
        // (so `var` resolves to the current element + any outer
        // bindings resolve through the delegated schema).
        let projected = match projection {
            Some(proj) => evaluate(proj, &ext, &scoped, params)?,
            None => ext[slot].clone(),
        };
        out.push(projected);
    }
    Ok(Value::List(out))
}

/// **ADR-191 D-6** (#620 map-half) — evaluate a map projection
/// `base{.key, .other, alias: expr, .*}` (openCypher v9 §3.5). Builds a
/// NEW [`Value::Map`] by selecting keys from `base` (a node /
/// relationship / map). Per the **D-6 null-handling split**:
///
/// - **`.key` property selector** — look up `key` in the base's property
///   bag; include it ONLY when present AND non-null. A `null`/absent value
///   DROPS the key (`n{.missing}` → `{}`).
/// - **`.*` all-properties selector** — copy EVERY property of the base.
///   (Stored node/rel properties are never null; for a map base this is a
///   bulk copy. A later explicit entry overrides via last-writer-wins.)
/// - **`alias: expr` literal entry** — evaluate `expr` in the CURRENT row
///   scope (NOT a scoped var — the base's bag is not in scope) and insert
///   `alias`; the key is KEPT even when `expr` is `null` (`n{x: null}` →
///   `{x: null}`).
///
/// Edge cases:
/// - **null base ⇒ `Value::Null`** (openCypher null-propagation; consistent
///   with `map.key` on a null base — `eval` `PropertyAccess` `Null` arm).
/// - **empty `base{}` ⇒ empty `Value::Map`**.
/// - **duplicate output key ⇒ last-writer-wins** (`BTreeMap::insert`),
///   matching the map-literal carrier.
fn eval_map_projection(
    base: &BoundExpression,
    items: &[BoundMapProjectionItem],
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    let base_val = evaluate(base, row, schema, params)?;
    // The base must be a property-bag-bearing value. A `null` base yields
    // a `null` projection (NOT an empty map) per openCypher null-propagation.
    let bag: &BTreeMap<String, Value> = match &base_val {
        Value::Node(n) => &n.properties,
        Value::Relationship(r) => &r.properties,
        Value::Map(m) => m,
        Value::Null => return Ok(Value::Null),
        other => {
            return Err(ExecutionError::Eval(format!(
                "map projection base is not a node, relationship, or map: {other:?}"
            )));
        }
    };
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    for item in items {
        match item {
            // D-6 — `.key` DROPS a null/absent value.
            BoundMapProjectionItem::Property(key) => {
                if let Some(v) = bag.get(key) {
                    if !matches!(v, Value::Null) {
                        out.insert(key.clone(), v.clone());
                    }
                }
                // absent key ⇒ drop (no insert).
            }
            // `.*` — copy every property of the base.
            BoundMapProjectionItem::AllProperties => {
                for (k, v) in bag {
                    out.insert(k.clone(), v.clone());
                }
            }
            // D-6 — `alias: expr` KEEPS the key even when the value is null.
            BoundMapProjectionItem::Literal { alias, value } => {
                let v = evaluate(value, row, schema, params)?;
                out.insert(alias.clone(), v);
            }
        }
    }
    Ok(Value::Map(out))
}

/// **ADR-188 Decision 1** — allocate the REUSED scratch buffer for a
/// scoped-eval level: a single `Vec<Value>` = `row` plus `n_slots`
/// trailing placeholder slots (`Value::Null`) for the level's scoped
/// variable(s) (`n_slots = 1` for `all`/`any`/`none`/`single`; `2` for
/// `reduce`'s `acc` + `x`). Allocated **once** per level invocation;
/// the caller then **overwrites** the trailing slot(s) in place per
/// element — so there is **no per-element heap allocation** (ADR-188
/// Decision 1 §BoE proviso (2)). The trailing slots sit at
/// `row.len() .. row.len() + n_slots`, exactly the planner-assigned
/// scoped-var column indices; the placeholder value is overwritten
/// before the first read, so its initial `Null` is never observed.
#[inline]
fn extended_row(row: &[Value], n_slots: usize) -> Vec<Value> {
    let mut ext = Vec::with_capacity(row.len() + n_slots);
    ext.extend_from_slice(row);
    ext.resize(row.len() + n_slots, Value::Null);
    ext
}

/// **ADR-188 Decision 4** — evaluate a list-predicate's inner predicate
/// against an extended row and lift the result to [`ThreeValued`]. The
/// predicate is a Cypher boolean expression; `from_value` maps
/// `Boolean(true)/Boolean(false)/Null` → `True/False/Unknown` (the
/// 3VL bridge). A non-boolean, non-null predicate result is a planner/
/// type-check escape — `from_value` maps any other `Value` to
/// `Unknown` per its documented coercion (the type-check pass should
/// have rejected a non-boolean predicate upstream; `Unknown` is the
/// conservative "could be anything" 3VL choice).
#[inline]
fn three_valued_predicate(
    predicate: &BoundExpression,
    ext_row: &[Value],
    scoped: &dyn Fn(BindingId) -> Option<usize>,
    params: &Parameters,
) -> Result<ThreeValued, ExecutionError> {
    let v = evaluate(predicate, ext_row, scoped, params)?;
    Ok(ThreeValued::from_value(&v))
}

/// Lift an [`crate::ast::Literal`] to a [`Value`].
///
/// `Literal::List` and `Literal::Map` carry inner [`Expression`]s
/// (NOT pre-lowered `BoundExpression`s — the parser emits AST shape
/// directly). v1.0-alpha lifts these to [`Value::List`] / [`Value::Map`]
/// by lifting each inner expression via [`literal_expression_to_value`]
/// (pure-literal inner shapes; a non-literal inner expression lifts to
/// `Value::Null`, matching the list carrier's documented v1.0 limit).
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::Integer(n) => Value::Integer(*n),
        Literal::Float(f) => Value::Float(*f),
        Literal::String(s) => Value::String(s.clone()),
        Literal::Bool(b) => Value::Boolean(*b),
        Literal::Null => Value::Null,
        Literal::List(elems) => Value::List(
            elems
                .iter()
                .map(literal_expression_to_value)
                .collect::<Vec<_>>(),
        ),
        // ADR-191 D-2 (THE BUG FIX) — evaluate each value-expression in
        // declaration order into a deterministic `BTreeMap`; a duplicate
        // key is last-writer-wins (`BTreeMap::insert` overwrites),
        // matching Cypher's parse-time dedup. Was `=> Value::Null`, the
        // silent-wrong-answer no-op the parser fed (the evaluator threw
        // away the parsed map). Inner values lift via
        // `literal_expression_to_value` (pure-literal — same v1.0 limit
        // as the `Literal::List` carrier above).
        Literal::Map(entries) => {
            let mut m = BTreeMap::new();
            for (k, v_expr) in entries {
                m.insert(k.clone(), literal_expression_to_value(v_expr));
            }
            Value::Map(m)
        }
        // W23-V11-T-01 / ADR-090 — temporal + decimal literal lift.
        // Each temporal literal type wraps its corresponding
        // arcgraph_core wire type; the executor `Value` variant
        // takes the same wire type so the lift is a direct copy.
        Literal::Temporal(t) => Value::Temporal(*t),
        Literal::LocalDateTime(ldt) => Value::LocalDateTime(*ldt),
        Literal::Date(d) => Value::Date(*d),
        Literal::Duration(d) => Value::Duration(*d),
        Literal::Decimal(d) => Value::Decimal(*d),
    }
}

/// Helper: lift a literal-shaped `ast::Expression` to a `Value`.
/// Returns `Value::Null` for non-literal inner shapes (a defensive
/// fallback; the planner SHOULD pre-lower these, but the
/// `Literal::List` carrier admits arbitrary expressions per the
/// grammar).
fn literal_expression_to_value(e: &Expression) -> Value {
    match e {
        Expression::Literal(lit) => literal_to_value(lit),
        // #870 / TCK `Literals7`/`8` — a NEGATIVE (or unary-`+`) numeric
        // literal parses as `UnaryOp { Neg/Pos, <numeric literal> }`, NOT a
        // bare `Literal` (`RETURN [-5]` ⇒ element is `UnaryOp(Neg, 5)`). It IS
        // a constant; fold it here instead of dropping it to `Null` (the
        // silent-wrong-answer the `_` arm produced — `[-5]` ⇒ `[null]`).
        // Non-numeric / overflow folds to `Null`, matching the carrier's
        // documented v1.0 limit for genuinely non-constant inner shapes.
        Expression::UnaryOp {
            op: UnaryOp::Neg,
            operand,
        } => negate_const_value(literal_expression_to_value(operand)).unwrap_or(Value::Null),
        Expression::UnaryOp {
            op: UnaryOp::Pos,
            operand,
        } => literal_expression_to_value(operand),
        // Other expression shapes inside a literal list are not
        // reduced at v1.0-alpha. The MVP test fixtures use pure-
        // literal lists.
        _ => Value::Null,
    }
}

/// Negate a CONSTANT numeric [`Value`] (the operand of a unary-minus on a
/// numeric literal — `-5`, `-.1e-5`, `-0x1f`). Returns `None` for a
/// non-numeric value (unary minus on a non-number is a type error, not a
/// constant) or an `i64` negation overflow. Shared by the read-path
/// (`literal_expression_to_value`) and the write-path property/list lifts
/// (`ops::literal_lift`) so the #870 fix has one definition. #870.
pub(crate) fn negate_const_value(v: Value) -> Option<Value> {
    match v {
        Value::Integer(n) => n.checked_neg().map(Value::Integer),
        Value::Float(f) => Some(Value::Float(-f)),
        _ => None,
    }
}

/// Lift a [`ThreeValued`] back to a [`Value`] for use in a
/// non-predicate expression context (e.g., `RETURN n IS NULL` —
/// returns Boolean, not 3VL).
fn threevalued_to_value(tv: ThreeValued) -> Value {
    match tv {
        ThreeValued::True => Value::Boolean(true),
        ThreeValued::False => Value::Boolean(false),
        ThreeValued::Unknown => Value::Null,
    }
}

/// Apply a binary operator with NULL propagation.
fn apply_binop(op: BinOp, l: Value, r: Value) -> Result<Value, ExecutionError> {
    use BinOp::*;
    // Boolean ops route through 3VL.
    match op {
        And => {
            let lt = ThreeValued::from_value(&l);
            let rt = ThreeValued::from_value(&r);
            return Ok(threevalued_to_value(lt.and(rt)));
        }
        Or => {
            let lt = ThreeValued::from_value(&l);
            let rt = ThreeValued::from_value(&r);
            return Ok(threevalued_to_value(lt.or(rt)));
        }
        // XOR (#621) — same 3VL routing as And/Or; only the truth-table
        // differs (`ThreeValued::xor`). `_ XOR null = null` falls out of
        // the 3VL Unknown-propagation, so XOR is INTENTIONALLY handled
        // here (BEFORE the NULL-short-circuit below) — exactly like
        // And/Or, whose `False AND null`/`True OR null` results are NOT
        // null. Non-boolean operands are rejected at type-check time
        // (see `semantic::type_check::check_binary_op`), mirroring And/Or.
        Xor => {
            let lt = ThreeValued::from_value(&l);
            let rt = ThreeValued::from_value(&r);
            return Ok(threevalued_to_value(lt.xor(rt)));
        }
        _ => {}
    }
    // Other binops: NULL operand → NULL result.
    if l.is_null() || r.is_null() {
        return Ok(Value::Null);
    }
    match op {
        // Eq / Neq route through the UNIFIED openCypher 3VL value
        // equality (`values_equal_3vl`): a NESTED null comparison (a list
        // `[1,2] = [null,2]` OR a map `{a:null} = {a:null}` per ADR-191
        // D-3) yields `Unknown` ⇒ `Null`, NOT a definite `false`. Maps
        // delegate to #735's `map_equality_3vl` inside `values_equal_3vl`,
        // so the #735 map-equality semantics are preserved AND lists now
        // fold 3VL too (this slice's `=`/`<>` fix). Top-level null operands
        // already short-circuited above, so a `None` here is a nested null.
        Eq => Ok(match values_equal_3vl(&l, &r) {
            Some(b) => Value::Boolean(b),
            None => Value::Null,
        }),
        Neq => Ok(match values_equal_3vl(&l, &r) {
            Some(b) => Value::Boolean(!b),
            None => Value::Null,
        }),
        Lt => compare_op(&l, &r, |o| matches!(o, std::cmp::Ordering::Less)),
        Le => compare_op(&l, &r, |o| {
            matches!(o, std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        }),
        Gt => compare_op(&l, &r, |o| matches!(o, std::cmp::Ordering::Greater)),
        Ge => compare_op(&l, &r, |o| {
            matches!(o, std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        }),
        // #621 — `+` is overloaded: list / string concatenation OR numeric
        // addition. `add_or_concat` dispatches; the numeric fall-through is
        // IDENTICAL to the pre-#621 `arithmetic(..)` path. Concat is
        // `Add`-ONLY (Sub/Mul/Div/Mod below stay numeric-only).
        Add => add_or_concat(l, r),
        Sub => arithmetic(&l, &r, |a, b| Ok(a - b), |a, b| a.checked_sub(b)),
        Mul => arithmetic(&l, &r, |a, b| Ok(a * b), |a, b| a.checked_mul(b)),
        Div => arithmetic_div(&l, &r),
        Mod => arithmetic(
            &l,
            &r,
            |a, b| {
                if b == 0.0 {
                    Err(ExecutionError::Eval("modulus by zero".into()))
                } else {
                    Ok(a % b)
                }
            },
            |a, b| if b == 0 { None } else { a.checked_rem(b) },
        ),
        Pow => pow(&l, &r),
        // openCypher v9 §3.3.6 string predicates (#773). Both operands are
        // non-null here (the null short-circuit above already returned Null
        // for any null operand — 3VL). A NON-string operand yields Null
        // (NOT an error, NOT `false`) — matching the TCK Precedence4 [4]
        // oracle (`true STARTS WITH 'abc'` ⇒ null). Case-SENSITIVE; Rust's
        // byte-based `str` predicates are UTF-8-correct for prefix / suffix /
        // substring matching (they operate on encoded-byte boundaries, which
        // for valid UTF-8 coincide with codepoint-aligned matches).
        StartsWith => Ok(string_predicate(&l, &r, |hay, needle| {
            hay.starts_with(needle)
        })),
        EndsWith => Ok(string_predicate(&l, &r, |hay, needle| {
            hay.ends_with(needle)
        })),
        Contains => Ok(string_predicate(&l, &r, |hay, needle| hay.contains(needle))),
        And | Or | Xor => unreachable!("handled above"),
    }
}

/// openCypher v9 §3.3.6 string-predicate kernel (STARTS WITH / ENDS WITH /
/// CONTAINS). Caller guarantees both operands are non-null (the
/// null-short-circuit in [`apply_binop`] already fired). A non-string
/// operand ⇒ `Null` (the openCypher type-mismatch-in-string-predicate rule —
/// e.g. `true STARTS WITH 'abc'` ⇒ null, TCK Precedence4 [4]); never an
/// error. `f` is the byte-level `str` predicate (`starts_with` / `ends_with`
/// / `contains`), which is codepoint-correct for valid UTF-8.
fn string_predicate(l: &Value, r: &Value, f: impl Fn(&str, &str) -> bool) -> Value {
    match (l, r) {
        (Value::String(hay), Value::String(needle)) => Value::Boolean(f(hay, needle)),
        _ => Value::Null,
    }
}

fn apply_unop(op: UnaryOp, v: Value) -> Result<Value, ExecutionError> {
    match op {
        UnaryOp::Not => {
            let tv = ThreeValued::from_value(&v);
            Ok(threevalued_to_value(tv.not()))
        }
        UnaryOp::Neg => {
            if v.is_null() {
                return Ok(Value::Null);
            }
            match v {
                Value::Integer(n) => {
                    Ok(Value::Integer(n.checked_neg().ok_or_else(|| {
                        ExecutionError::Eval("integer negation overflow".into())
                    })?))
                }
                Value::Float(f) => Ok(Value::Float(-f)),
                _ => Err(ExecutionError::Eval(
                    "unary negation on non-numeric value".into(),
                )),
            }
        }
        UnaryOp::Pos => {
            // Cypher 9 §3.4: unary + is identity on numerics.
            if v.is_null() {
                return Ok(Value::Null);
            }
            match v {
                Value::Integer(_) | Value::Float(_) => Ok(v),
                _ => Err(ExecutionError::Eval(
                    "unary plus on non-numeric value".into(),
                )),
            }
        }
    }
}

/// openCypher VALUE equality under 3VL (Cypher 9 §3.3.5 + §3.4),
/// returning `None` for `Unknown` (the `NULL`-comparison outcome) and
/// `Some(bool)` for a definite result. Shared by `IN` membership and
/// the `=` / `<>` operators (which share openCypher value-equality
/// semantics).
///
/// Rules:
/// - any operand is `NULL` ⇒ `None` (Unknown). This fires for NESTED
///   nulls during list recursion (`[1,2] = [null,2]` ⇒ Unknown);
///   top-level nulls are short-circuited by the `apply_binop` / `In`
///   callers before this is reached, but handling them here keeps the
///   helper self-contained (and correct for recursion).
/// - same-variant scalars compare by value; `Integer`/`Float` cross-
///   compare numerically.
/// - **lists**: unequal length ⇒ `Some(false)`; else a 3VL pairwise
///   fold — ANY pair `Some(false)` ⇒ `Some(false)` (a definite mismatch
///   DOMINATES); else ANY pair `None` ⇒ `None`; else `Some(true)`.
/// - heterogeneous non-null variants (e.g. `1 = '1'`, a list vs a
///   scalar) are a TYPE MISMATCH ⇒ `Some(false)` (definitely-false, NOT
///   null — Cypher 9 §3.3.5).
fn values_equal_3vl(a: &Value, b: &Value) -> Option<bool> {
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) => None,
        (Value::Boolean(x), Value::Boolean(y)) => Some(x == y),
        (Value::Integer(x), Value::Integer(y)) => Some(x == y),
        (Value::Float(x), Value::Float(y)) => Some(x == y),
        (Value::Integer(x), Value::Float(y)) | (Value::Float(y), Value::Integer(x)) => {
            Some((*x as f64) == *y)
        }
        (Value::String(x), Value::String(y)) => Some(x == y),
        (Value::Node(x), Value::Node(y)) => Some(x.id == y.id),
        (Value::Relationship(x), Value::Relationship(y)) => Some(x.id == y.id),
        (Value::List(x), Value::List(y)) => {
            if x.len() != y.len() {
                return Some(false);
            }
            let mut saw_unknown = false;
            for (ax, bx) in x.iter().zip(y) {
                match values_equal_3vl(ax, bx) {
                    Some(false) => return Some(false), // definite mismatch dominates
                    None => saw_unknown = true,
                    Some(true) => {}
                }
            }
            if saw_unknown { None } else { Some(true) }
        }
        // ADR-191 D-3 (#735) — maps fold under 3VL via `map_equality_3vl`
        // (same-key-set check + a 3VL pairwise value fold that recurses
        // back through THIS function, so a list/map nested in a map value
        // also gets 3VL). `{a:null} = {a:null}` ⇒ Unknown. A map vs a
        // non-map falls through to the type-mismatch arm below.
        (Value::Map(x), Value::Map(y)) => map_equality_3vl(x, y),
        // ADR-193 D-10 — two paths are equal iff their node sequences AND
        // relationship sequences are element-wise equal by IDENTITY (node
        // IDs in order, rel IDs in order). Direction is captured by the
        // node sequence (start + each segment's `end`). Reuses the by-ID
        // node/rel identity equality above (NOT structural property
        // equality). A path contains no NULLs (nodes/rels are always
        // bound), so path equality is TWO-VALUED — wrapped in `Some`;
        // `path = <non-path>` falls to the `_ => Some(false)` type-
        // mismatch arm below (FALSE, NOT null/error — openCypher §3.2).
        (Value::Path(x), Value::Path(y)) => Some(
            x.start.id == y.start.id
                && x.segments.len() == y.segments.len()
                && x.segments
                    .iter()
                    .zip(&y.segments)
                    .all(|(a, b)| a.rel.id == b.rel.id && a.end.id == b.end.id),
        ),
        // Heterogeneous non-null variants: type mismatch is definitely
        // FALSE under Cypher value-equality (NOT null). Same-variant
        // Temporal / Decimal equality is out of scope for this slice and
        // preserves the prior `_ => false` behavior.
        _ => Some(false),
    }
}

/// 3VL equality of two maps (ADR-191 D-3 / #735): `Some(true)` /
/// `Some(false)` for a definite verdict, `None` for the openCypher
/// Unknown (→ null).
///
/// Maps with a DIFFERENT key SET are definitely unequal (`Some(false)`),
/// NOT Unknown — `{a:null}` ≠ `{}`. With an identical key set, the result
/// is the 3VL fold over the pairwise value comparisons — which recurse
/// through [`values_equal_3vl`], so a list/map nested in a map value also
/// folds under 3VL (`{a:[1,null]}={a:[1,null]}` → Unknown): any
/// definite-`false` short-circuits to `Some(false)`; otherwise any
/// Unknown makes the whole comparison Unknown (`{a:null}={a:null}` →
/// `None`); only an all-`true` fold is `Some(true)`. Order-independence
/// is free — `BTreeMap` keys/values iterate in sorted order. (Merged with
/// #621: the pairwise comparison now routes through the unified
/// `values_equal_3vl` rather than #735's `value_eq_3vl`, so list-valued
/// map entries get 3VL too; the map-equality verdicts #735's
/// `map_equality_3vl_helper_table` pins are unchanged.)
fn map_equality_3vl(x: &BTreeMap<String, Value>, y: &BTreeMap<String, Value>) -> Option<bool> {
    if x.len() != y.len() || !x.keys().eq(y.keys()) {
        return Some(false);
    }
    let mut unknown = false;
    for (k, xv) in x {
        let yv = y.get(k).expect("key sets verified equal above");
        match values_equal_3vl(xv, yv) {
            Some(false) => return Some(false),
            Some(true) => {}
            None => unknown = true,
        }
    }
    if unknown { None } else { Some(true) }
}

/// openCypher v9 §3.4 — bracket subscript `base[index]`, dual-mode on
/// the runtime base value (#1056 / #990):
///
/// - **List × Integer** (`list[i]`): 0-based; a negative index counts
///   from the end (`list[-1]` = last element); an out-of-range index ⇒
///   `null` (NOT an error).
/// - **Map × String** (`map['key']`): CASE-SENSITIVE key lookup; a
///   MISSING key ⇒ `null` (openCypher dynamic value access — TCK
///   `expressions/map/Map2`). `{name:'a'}['NAME']` ⇒ `null` (`Map2` [5]).
///
/// `null` base or `null` index ⇒ `null` (3VL short-circuit). A mismatched
/// base/index combination (a non-list base with an integer index, a
/// non-map base with a string index, a Map indexed by a non-string, a
/// List indexed by a non-integer) only reaches here for a DYNAMIC value
/// (a `Property` / parameter that resolved to the wrong runtime type) —
/// the type-check rejects statically-known cases — so it is an honest
/// eval error.
fn eval_subscript(base: Value, index: Value) -> Result<Value, ExecutionError> {
    if base.is_null() || index.is_null() {
        return Ok(Value::Null);
    }
    // Map × String — dynamic value access (`map['key']`). Case-sensitive
    // key lookup; missing key ⇒ null (NOT an error), mirroring the
    // out-of-range list-index ⇒ null semantic.
    if let Value::Map(entries) = &base {
        return match index {
            Value::String(key) => Ok(entries.get(&key).cloned().unwrap_or(Value::Null)),
            _ => Err(ExecutionError::Eval(
                "map subscript key must be a string".into(),
            )),
        };
    }
    let elems = match base {
        Value::List(elems) => elems,
        _ => {
            return Err(ExecutionError::Eval("subscript base must be a list".into()));
        }
    };
    let i = match index {
        Value::Integer(i) => i,
        _ => {
            return Err(ExecutionError::Eval(
                "subscript index must be an integer".into(),
            ));
        }
    };
    let len = elems.len() as i64;
    let resolved = if i < 0 { len + i } else { i };
    if resolved < 0 || resolved >= len {
        return Ok(Value::Null); // out-of-range index ⇒ null
    }
    // `resolved` is in `[0, len)`; `nth` yields `Some` (the `unwrap_or`
    // is a defensive no-panic fallback, never taken).
    Ok(elems
        .into_iter()
        .nth(resolved as usize)
        .unwrap_or(Value::Null))
}

/// openCypher v9 §3.4 — list slice `base[start..end]` (end exclusive).
/// Open bounds default to `0` / `len`; negative bounds count from the
/// end; out-of-range bounds CLAMP (no error). `null` base ⇒ `null`; a
/// PRESENT bound that evaluates to `null` makes the whole slice `null`.
/// A non-list base only reaches here for a dynamic value (string slicing
/// out of scope), so it is an honest eval error (mirrors
/// [`eval_subscript`]).
fn eval_slice(
    base: Value,
    start: Option<Value>,
    end: Option<Value>,
) -> Result<Value, ExecutionError> {
    if base.is_null() {
        return Ok(Value::Null);
    }
    // A present-but-null bound ⇒ the whole slice is null (3VL).
    if matches!(start, Some(Value::Null)) || matches!(end, Some(Value::Null)) {
        return Ok(Value::Null);
    }
    let elems = match base {
        Value::List(elems) => elems,
        _ => return Err(ExecutionError::Eval("slice base must be a list".into())),
    };
    let len = elems.len() as i64;
    let resolve = |bound: Option<Value>, default: i64| -> Result<i64, ExecutionError> {
        match bound {
            None => Ok(default),
            // Negative bound counts from the end (pre-clamp).
            Some(Value::Integer(i)) => Ok(if i < 0 { len + i } else { i }),
            Some(_) => Err(ExecutionError::Eval(
                "slice bound must be an integer".into(),
            )),
        }
    };
    let lo = resolve(start, 0)?.clamp(0, len);
    let hi = resolve(end, len)?.clamp(0, len);
    if lo >= hi {
        return Ok(Value::List(Vec::new()));
    }
    Ok(Value::List(
        elems
            .into_iter()
            .skip(lo as usize)
            .take((hi - lo) as usize)
            .collect(),
    ))
}

fn compare_op<F: Fn(std::cmp::Ordering) -> bool>(
    a: &Value,
    b: &Value,
    pred: F,
) -> Result<Value, ExecutionError> {
    match compare_ordering_3vl(a, b) {
        CompareOrdering::Known(ord) => Ok(Value::Boolean(pred(ord))),
        CompareOrdering::Unknown => Ok(Value::Null),
        // IEEE-754 NaN has no numeric ordering. The Comparison2 [5]
        // corpus pins every ordering predicate involving numeric NaN to
        // false, including <= and >=.
        CompareOrdering::UnorderedNumber => Ok(Value::Boolean(false)),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompareOrdering {
    Known(std::cmp::Ordering),
    Unknown,
    UnorderedNumber,
}

/// openCypher order comparison kernel for `<`, `>`, `<=`, `>=`.
///
/// Non-null incompatible operands are Unknown (`null` at the operator
/// layer), not an evaluator error. Numbers compare across integer/float
/// variants; numeric NaN is a separate unordered outcome that the TCK
/// maps to `false` for all four ordering predicates. Lists compare
/// lexicographically and stop at the first decisive element; a null or
/// incompatible element reached before any decisive difference makes the
/// list comparison Unknown.
fn compare_ordering_3vl(a: &Value, b: &Value) -> CompareOrdering {
    match (a, b) {
        (Value::Null, _) | (_, Value::Null) => CompareOrdering::Unknown,
        (Value::Integer(x), Value::Integer(y)) => CompareOrdering::Known(x.cmp(y)),
        (Value::Float(x), Value::Float(y)) => x
            .partial_cmp(y)
            .map(CompareOrdering::Known)
            .unwrap_or(CompareOrdering::UnorderedNumber),
        (Value::Integer(x), Value::Float(y)) => (*x as f64)
            .partial_cmp(y)
            .map(CompareOrdering::Known)
            .unwrap_or(CompareOrdering::UnorderedNumber),
        (Value::Float(x), Value::Integer(y)) => x
            .partial_cmp(&(*y as f64))
            .map(CompareOrdering::Known)
            .unwrap_or(CompareOrdering::UnorderedNumber),
        (Value::String(x), Value::String(y)) => CompareOrdering::Known(x.cmp(y)),
        (Value::Boolean(x), Value::Boolean(y)) => CompareOrdering::Known(x.cmp(y)),
        (Value::List(xs), Value::List(ys)) => compare_lists_ordering_3vl(xs, ys),
        _ => CompareOrdering::Unknown,
    }
}

fn compare_lists_ordering_3vl(xs: &[Value], ys: &[Value]) -> CompareOrdering {
    use std::cmp::Ordering;

    for (x, y) in xs.iter().zip(ys) {
        match compare_ordering_3vl(x, y) {
            CompareOrdering::Known(Ordering::Equal) => {}
            other @ CompareOrdering::Known(_) => return other,
            CompareOrdering::Unknown => return CompareOrdering::Unknown,
            CompareOrdering::UnorderedNumber => return CompareOrdering::UnorderedNumber,
        }
    }
    CompareOrdering::Known(xs.len().cmp(&ys.len()))
}

fn arithmetic<FF, FI>(a: &Value, b: &Value, f_float: FF, f_int: FI) -> Result<Value, ExecutionError>
where
    FF: Fn(f64, f64) -> Result<f64, ExecutionError>,
    FI: Fn(i64, i64) -> Option<i64>,
{
    match (a, b) {
        (Value::Integer(x), Value::Integer(y)) => {
            Ok(Value::Integer(f_int(*x, *y).ok_or_else(|| {
                ExecutionError::Eval("integer overflow".into())
            })?))
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(f_float(*x, *y)?)),
        (Value::Integer(x), Value::Float(y)) => Ok(Value::Float(f_float(*x as f64, *y)?)),
        (Value::Float(x), Value::Integer(y)) => Ok(Value::Float(f_float(*x, *y as f64)?)),
        _ => Err(ExecutionError::Eval(
            "arithmetic on non-numeric value".into(),
        )),
    }
}

/// ADR-147-amendment-03 (D-1, §B1 write-path OOM backstop) — per-op cap
/// on the ELEMENT COUNT any single list-concat may produce. Enforced
/// INSIDE [`add_or_concat`] BEFORE the `extend`/`push` allocation, so a
/// bracketed doubling tree `{x: ((($a+$a)+($a+$a))+…)}` (which the
/// CREATE-property whitelist admits via `BinOp::Add`) dies at the FIRST
/// over-cap node instead of materializing every intermediate up to
/// ~2^depth elements and OOM-ing the process. This is the REAL backstop:
/// the result-only cap in `literal_lift::MAX_CREATE_PROP_LIST_LEN` runs
/// AFTER the multi-GB intermediate is already allocated, so it gates the
/// result, not the amplifying intermediates.
///
/// Matches `literal_lift::MAX_CREATE_PROP_LIST_LEN` (1M) so a value that
/// clears every per-op cap also clears the result cap — the two caps
/// agree, and a legitimate concat producing ≤1M elements is unaffected.
/// Back-of-envelope: 1M `Value`s ≈ 24 MB in-memory before
/// the JSON-blob encode — abusive for one expression, ~3 orders of
/// magnitude above any legitimate concat. Applies to ALL callers (read +
/// write); the read path was already exposed to the same amplifier
/// (`RETURN [0] + [0] + …`), so capping here contains both.
pub(crate) const MAX_CONCAT_LIST_LEN: usize = 1_000_000;

// ADR-147-amendment-03 §B1: the per-op list-concat cap MUST equal the
// result-level cap (`literal_lift::MAX_CREATE_PROP_LIST_LEN`) so a value
// that clears every intermediate also clears the result gate — else a
// value in the (per-op, result) gap would per-op-pass then result-reject
// (or vice versa). Compiler-enforced (const assertion, not a runtime
// test) so a future edit to EITHER const that breaks the equality fails
// the build.
const _: () = assert!(
    MAX_CONCAT_LIST_LEN == crate::executor::ops::MAX_CREATE_PROP_LIST_LEN,
    "eval::MAX_CONCAT_LIST_LEN must equal literal_lift::MAX_CREATE_PROP_LIST_LEN (ADR-147-amendment-03 §B1)"
);

/// ADR-147-amendment-03 (D-1, §B1) — per-op cap on the BYTE LENGTH any
/// single string-concat may produce. Unlike the list case there is NO
/// result-level string cap in `literal_lift`, so this per-op check is the
/// SOLE backstop against a doubling `'x' + 'x' + …` string amplifier on
/// the CREATE property path. Back-of-envelope: 16 MiB is
/// far above any legitimate single string property (a UTF-8 document is
/// KBs–low-MBs) yet dies well before the process OOMs; the doubling tree
/// reaches it in ~24 levels from a 1-byte base, and the first over-cap
/// node clean-errors before its `push_str` allocation.
pub(crate) const MAX_CONCAT_STRING_BYTES: usize = 16 * 1024 * 1024;

/// **#621** — the openCypher `+` dispatch: collection / string
/// concatenation, else numeric addition.
///
/// `+` is overloaded per openCypher v9 §3:
///
/// * `list + list`     → concatenation (`a` then `b`)
/// * `list + element`  → append (`element` after `a`)
/// * `element + list`  → prepend (`element` before `b`)
/// * `string + string` → concatenation
/// * otherwise         → numeric addition via [`arithmetic`] — the
///   numeric path is BYTE-IDENTICAL to the pre-#621 `Add` arm (a
///   non-numeric, non-concat operand still errors "arithmetic on
///   non-numeric value")
///
/// NULL operands NEVER reach here: [`apply_binop`] short-circuits
/// `l.is_null() || r.is_null()` to `Value::Null` BEFORE the op match, so
/// openCypher's null-propagating `+` (`null + [1] = null`,
/// `'a' + null = null`, `null + 1 = null`) is already handled upstream —
/// this helper does NOT re-handle null (no duplicate null logic).
///
/// Concat is `Add`-ONLY: `Sub`/`Mul`/`Div`/`Mod` never call this; a
/// list/string operand there flows through [`arithmetic`] /
/// [`arithmetic_div`] and errors, matching the type-checker's `+`-only
/// concat admission in `concat_result_type`.
///
/// # OOM backstop (ADR-147-amendment-03 §B1)
///
/// Delegates to [`checked_add_or_concat`] with the production caps
/// ([`MAX_CONCAT_LIST_LEN`] / [`MAX_CONCAT_STRING_BYTES`]). The cap is
/// enforced BEFORE the allocation on EVERY intermediate, so an admitted
/// `BinOp::Add` doubling tree amplifier dies at the first over-cap node.
fn add_or_concat(l: Value, r: Value) -> Result<Value, ExecutionError> {
    checked_add_or_concat(l, r, MAX_CONCAT_LIST_LEN, MAX_CONCAT_STRING_BYTES)
}

/// [`add_or_concat`] with EXPLICIT caps, so the OOM-backstop invariant is
/// unit-testable at a small cap (fast + deterministic, no actual OOM) —
/// the production entrypoint passes [`MAX_CONCAT_LIST_LEN`] /
/// [`MAX_CONCAT_STRING_BYTES`]. Every concat arm checks
/// `a.len() + b.len() > cap` BEFORE the `extend`/`push`/`push_str`
/// allocation and returns a clean typed [`ExecutionError::Eval`] (mirrors
/// `fn_range`'s over-cap error), so the amplifying intermediate is never
/// allocated.
fn checked_add_or_concat(
    l: Value,
    r: Value,
    list_cap: usize,
    str_cap: usize,
) -> Result<Value, ExecutionError> {
    match (l, r) {
        // list ++ list — concatenate (`a` then `b`).
        (Value::List(mut a), Value::List(b)) => {
            check_concat_len(a.len(), b.len(), list_cap, "list")?;
            a.extend(b);
            Ok(Value::List(a))
        }
        // list + element — append the scalar after the list.
        (Value::List(mut a), elem) => {
            check_concat_len(a.len(), 1, list_cap, "list")?;
            a.push(elem);
            Ok(Value::List(a))
        }
        // element + list — prepend the scalar before the list.
        (elem, Value::List(b)) => {
            check_concat_len(1, b.len(), list_cap, "list")?;
            let mut out = Vec::with_capacity(b.len() + 1);
            out.push(elem);
            out.extend(b);
            Ok(Value::List(out))
        }
        // string + string — concatenate.
        (Value::String(mut a), Value::String(b)) => {
            check_concat_len(a.len(), b.len(), str_cap, "string")?;
            a.push_str(&b);
            Ok(Value::String(a))
        }
        // Numeric (Integer / Float) addition — UNCHANGED from pre-#621.
        // Numeric `+` produces a single scalar, so it cannot amplify and
        // is deliberately NOT capped.
        (l, r) => arithmetic(&l, &r, |a, b| Ok(a + b), |a, b| a.checked_add(b)),
    }
}

/// ADR-147-amendment-03 (D-1, §B1) — reject a concat whose result would
/// exceed `cap` BEFORE the allocation. `a`/`b` are the two operand sizes
/// (list element counts or string byte lengths); `kind` labels the error.
/// Sums with `saturating_add` so a `usize` overflow on the sum itself
/// saturates to `usize::MAX` (> any real cap) rather than silently
/// wrapping — an overflowing sum is by definition over-cap.
fn check_concat_len(a: usize, b: usize, cap: usize, kind: &str) -> Result<(), ExecutionError> {
    let total = a.saturating_add(b);
    if total > cap {
        return Err(ExecutionError::Eval(format!(
            "{kind} concatenation would materialize {total} {unit}, exceeding cap {cap} \
             (ADR-147-amendment-03 §B1 write-path OOM backstop)",
            unit = if kind == "string" {
                "bytes"
            } else {
                "elements"
            }
        )));
    }
    Ok(())
}

fn arithmetic_div(a: &Value, b: &Value) -> Result<Value, ExecutionError> {
    // Cypher 9 §3.4: integer division when both operands are integer
    // (truncating toward zero per Rust semantics); float otherwise.
    match (a, b) {
        (Value::Integer(_), Value::Integer(0)) => {
            Err(ExecutionError::Eval("division by zero".into()))
        }
        (Value::Integer(x), Value::Integer(y)) => {
            Ok(Value::Integer(x.checked_div(*y).ok_or_else(|| {
                ExecutionError::Eval("integer division overflow".into())
            })?))
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x / y)),
        (Value::Integer(x), Value::Float(y)) => Ok(Value::Float((*x as f64) / *y)),
        (Value::Float(x), Value::Integer(y)) => {
            if *y == 0 {
                Err(ExecutionError::Eval("division by zero".into()))
            } else {
                Ok(Value::Float(*x / (*y as f64)))
            }
        }
        _ => Err(ExecutionError::Eval(
            "arithmetic on non-numeric value".into(),
        )),
    }
}

fn pow(a: &Value, b: &Value) -> Result<Value, ExecutionError> {
    let lhs = numeric_as_f64(a)?;
    let rhs = numeric_as_f64(b)?;
    Ok(Value::Float(lhs.powf(rhs)))
}

fn numeric_as_f64(v: &Value) -> Result<f64, ExecutionError> {
    match v {
        Value::Integer(i) => Ok(*i as f64),
        Value::Float(f) => Ok(*f),
        Value::Decimal(d) => Ok((d.units as f64) / 10_f64.powi(i32::from(d.scale))),
        _ => Err(ExecutionError::Eval(
            "exponentiation on non-numeric value".into(),
        )),
    }
}

/// Function-call dispatch. v1.0-alpha lights a small whitelist
/// sufficient for executor smoke; M4-63 expands.
fn apply_function(
    name: &str,
    args: &[BoundExpression],
    row: &[Value],
    schema: SchemaLookup<'_>,
    params: &Parameters,
) -> Result<Value, ExecutionError> {
    let evaled: Result<Vec<Value>, ExecutionError> = args
        .iter()
        .map(|a| evaluate(a, row, schema, params))
        .collect();
    let evaled = evaled?;
    match name.to_ascii_lowercase().as_str() {
        "id" => match evaled.as_slice() {
            [Value::Node(n)] => Ok(Value::Integer(n.id.raw() as i64)),
            [Value::Relationship(r)] => Ok(Value::Integer(r.id.raw() as i64)),
            [Value::Null] => Ok(Value::Null),
            _ => Err(ExecutionError::Eval("id() expects 1 entity arg".into())),
        },
        // #871 — `labels(node)` returns the node's label NAMES as a
        // list of strings (openCypher v9 §3). v1.0 is single-label per
        // M4-31, so the list is a singleton when the node is labeled, or
        // EMPTY for an unlabeled node. The name is the catalog-resolved
        // [`NodeView::label_name`] (populated at materialization / by the
        // CREATE op) — NEVER the opaque `LabelId`. `labels(null)` → null
        // per the standard's NULL-propagation. The type-checker
        // (`ArgKind::Node`) already rejects non-node arguments; the
        // catch-all arm is a defensive guard, not a reachable user path.
        "labels" => match evaled.as_slice() {
            [Value::Node(n)] => Ok(Value::List(
                n.label_name
                    .as_ref()
                    .map(|name| vec![Value::String(name.clone())])
                    .unwrap_or_default(),
            )),
            [Value::Null] => Ok(Value::Null),
            _ => Err(ExecutionError::Eval("labels() expects 1 node arg".into())),
        },
        // #871 — `type(rel)` returns the relationship-type NAME as a
        // string (openCypher v9 §3). The name is the catalog-resolved
        // [`RelView::rel_type_name`] — NEVER the opaque `TypeId`.
        // `type(null)` → null. The type-checker (`ArgKind::RelOnly`,
        // #618) already rejects node / path arguments; an unresolved
        // rel-type name surfaces as null rather than leaking the id.
        "type" => match evaled.as_slice() {
            [Value::Relationship(r)] => Ok(r
                .rel_type_name
                .as_ref()
                .map(|name| Value::String(name.clone()))
                .unwrap_or(Value::Null)),
            [Value::Null] => Ok(Value::Null),
            _ => Err(ExecutionError::Eval(
                "type() expects 1 relationship arg".into(),
            )),
        },
        "exists" => match evaled.as_slice() {
            [v] => Ok(Value::Boolean(!v.is_null())),
            _ => Err(ExecutionError::Eval("exists() expects 1 arg".into())),
        },

        // ---- W28 conformance scalar built-ins (Task #652) ----
        // Each arm asserts the openCypher-correct result incl. NULL
        // propagation + type-coercion edges (see the helper docs +
        // the unit tests). The match scrutinee is already
        // lower-cased, so the canonical camelCase registry names
        // (`toInteger`, `lTrim`, ...) dispatch here in lower form.

        // String (TCK expressions/string/*).
        "toupper" => str_fn(unary_arg(&evaled, name)?, name, |s| s.to_uppercase()),
        "tolower" => str_fn(unary_arg(&evaled, name)?, name, |s| s.to_lowercase()),
        "trim" => str_fn(unary_arg(&evaled, name)?, name, |s| s.trim().to_string()),
        "ltrim" => str_fn(unary_arg(&evaled, name)?, name, |s| {
            s.trim_start().to_string()
        }),
        "rtrim" => str_fn(unary_arg(&evaled, name)?, name, |s| {
            s.trim_end().to_string()
        }),
        "reverse" => fn_reverse(unary_arg(&evaled, name)?, name),
        "substring" => fn_substring(&evaled, name),
        "replace" => fn_replace(&evaled, name),
        "split" => fn_split(&evaled, name),
        "left" => fn_left_right(&evaled, name, true),
        "right" => fn_left_right(&evaled, name, false),

        // Math (TCK expressions/mathematical/*).
        "abs" => fn_abs(unary_arg(&evaled, name)?, name),
        "sign" => fn_sign(unary_arg(&evaled, name)?, name),
        "ceil" => num_to_float(unary_arg(&evaled, name)?, name, f64::ceil),
        "floor" => num_to_float(unary_arg(&evaled, name)?, name, f64::floor),
        "round" => num_to_float(unary_arg(&evaled, name)?, name, f64::round),
        "sqrt" => num_to_float(unary_arg(&evaled, name)?, name, f64::sqrt),
        "exp" => num_to_float(unary_arg(&evaled, name)?, name, f64::exp),
        "log" => num_to_float(unary_arg(&evaled, name)?, name, f64::ln),
        "log10" => num_to_float(unary_arg(&evaled, name)?, name, f64::log10),
        "sin" => num_to_float(unary_arg(&evaled, name)?, name, f64::sin),
        "cos" => num_to_float(unary_arg(&evaled, name)?, name, f64::cos),
        "tan" => num_to_float(unary_arg(&evaled, name)?, name, f64::tan),
        "e" => nullary(&evaled, name).map(|()| Value::Float(std::f64::consts::E)),
        "pi" => nullary(&evaled, name).map(|()| Value::Float(std::f64::consts::PI)),
        // `rand()` — nullary uniform `[0, 1)` generator (#618). Same
        // arity guard as `e`/`pi`; the value comes from the inline
        // per-thread PRNG (`next_rand_f64`, below). Non-deterministic by
        // design (openCypher v9 §3); consumed by the random-INDEPENDENT
        // `Quantifier9`..`Quantifier12` invariant scenarios.
        "rand" => nullary(&evaled, name).map(|()| Value::Float(next_rand_f64())),

        // Type conversion (TCK expressions/typeConversion/*).
        "tointeger" => to_integer(unary_arg(&evaled, name)?, name),
        "tofloat" => to_float(unary_arg(&evaled, name)?, name),
        "toboolean" => to_boolean(unary_arg(&evaled, name)?, name),
        "tostring" => to_string_value(unary_arg(&evaled, name)?, name),

        // Scalar / list.
        "coalesce" => Ok(fn_coalesce(&evaled)),
        "range" => fn_range(&evaled, name),

        // List / scalar accessors registered since M4-22 whose eval
        // was previously a NotImplemented fall-through.
        "size" => fn_size(unary_arg(&evaled, name)?, name),
        "length" => fn_length(unary_arg(&evaled, name)?, name),
        "head" => fn_head(unary_arg(&evaled, name)?, name),
        "last" => fn_last(unary_arg(&evaled, name)?, name),
        "tail" => fn_tail(unary_arg(&evaled, name)?, name),
        "keys" => fn_keys(unary_arg(&evaled, name)?, name),
        // #618 — `properties(node|rel|map)` → the property map. The match
        // scrutinee is already lower-cased, so `PROPERTIES`/`Properties`
        // dispatch here too (case-fold parity with the registry lookup).
        "properties" => fn_properties(unary_arg(&evaled, name)?, name),

        // Path functions (ADR-193 D-7). The match scrutinee is already
        // lower-cased; `nodes`/`relationships` dispatch in lower form.
        "nodes" => fn_nodes(unary_arg(&evaled, name)?, name),
        "relationships" => fn_relationships(unary_arg(&evaled, name)?, name),

        // Aggregations are M4-63.
        "count" | "sum" | "avg" | "min" | "max" | "collect" => {
            Err(ExecutionError::NotImplemented {
                feature: format!("aggregation function `{name}`"),
                target_slice: "M4-63".into(),
                section: "ADR-038 §2 D-28".into(),
            })
        }
        // The community(...) family is the planner-side surface for
        // IN-COMMUNITY; reaching it un-lowered is a planner bug.
        "community" | "community_id" => Err(ExecutionError::NotImplemented {
            feature: format!("function `{name}` (un-lowered community lookup)"),
            target_slice: "M4-32 / M4-62 lowering".into(),
            section: "ADR-038 amendment-01 §A-1".into(),
        }),
        // Other named functions: surface NotImplemented loudly so
        // tests catch missing wirings.
        other => Err(ExecutionError::NotImplemented {
            feature: format!("function `{other}`"),
            target_slice: "M4-63 / M4-71".into(),
            section: "ADR-038 §2 D-29".into(),
        }),
    }
}

// =====================================================================
// W28 conformance scalar built-in helpers (Task #652)
//
// Each helper realizes the openCypher v9 §3 semantics the vendored
// TCK expression corpus pins, with strict NULL propagation + runtime
// type enforcement. The eval arms in `apply_function` dispatch here.
// =====================================================================

/// Extract the single argument of a unary built-in. Defensive — the
/// type-checker already enforced arity, but the evaluator must never
/// panic on a malformed plan.
fn unary_arg<'a>(evaled: &'a [Value], name: &str) -> Result<&'a Value, ExecutionError> {
    match evaled {
        [v] => Ok(v),
        _ => Err(ExecutionError::Eval(format!(
            "{name}() expects exactly 1 argument, got {}",
            evaled.len()
        ))),
    }
}

/// Verify a nullary built-in (`e()` / `pi()` / `rand()`) was called
/// with no args.
fn nullary(evaled: &[Value], name: &str) -> Result<(), ExecutionError> {
    if evaled.is_empty() {
        Ok(())
    } else {
        Err(ExecutionError::Eval(format!(
            "{name}() expects no arguments, got {}",
            evaled.len()
        )))
    }
}

// =====================================================================
// `rand()` — nullary uniform `[0, 1)` generator (GA-rand slice, #618)
//
// openCypher v9 §3: "Returns a random floating point number in the
// range from 0 (inclusive) to 1 (exclusive); i.e. [0,1). The numbers
// returned follow an approximate uniform distribution."
//
// Implementation: a per-thread `xorshift64*` PRNG (Marsaglia / Vigna),
// lazily seeded from OS entropy. The seed is drawn from
// `std::collections::hash_map::RandomState`, whose hash keys are seeded
// from the platform RNG (getrandom / arc4random / RtlGenRandom). This
// is deliberately a **std-only entropy source**:
//
//   * NO third-party crate — Prime Directive #1 requires every
//     dependency be Apache-2.0 / MIT, and an inline generator sidesteps
//     the new-dep + architect-approval gate (dependency and artifact policy) entirely.
//   * NO time source — `Instant::now()` / `SystemTime::now()` are
//     avoided on purpose (entropy, not wall-clock), so there is no
//     hidden time-determinism coupling for the test harness.
//
// `rand()` is INHERENTLY non-deterministic, so there is no oracle for
// its VALUE. The conformance scenarios that consume it
// (`Quantifier9`..`Quantifier12`) assert random-INDEPENDENT invariants
// (a sublist built via `[y IN list WHERE rand() > 0.5 | y]` and an
// assertion that holds for ANY sublist), and the per-PR stability
// re-run proves that independence empirically.
// =====================================================================

thread_local! {
    /// Per-thread `xorshift64*` state. Lazily seeded from OS entropy on
    /// first use; the seed's `| 1` guarantees a nonzero state (zero is
    /// xorshift64's fixed point), and the transform is a bijection on
    /// the nonzero 64-bit words, so the state stays nonzero thereafter.
    static RAND_STATE: std::cell::Cell<u64> = std::cell::Cell::new(rand_seed());
}

/// Draw a nonzero per-thread seed from the platform RNG without a
/// third-party crate or a time source. `RandomState::new()` seeds its
/// hash keys from the OS RNG on construction; finishing a hasher over a
/// fixed salt surfaces that entropy as a `u64`.
fn rand_seed() -> u64 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(0x9E37_79B9_7F4A_7C15); // golden-ratio salt
    h.finish() | 1
}

/// One `xorshift64*` step → a uniform `Float` in `[0, 1)`.
///
/// The top 53 bits of the scrambled word are scaled by `2^-53` (the
/// canonical `u64 -> f64 ∈ [0, 1)` construction): the numerator is in
/// `[0, 2^53)` and `2^53` is exactly representable in `f64`, so the
/// quotient is in `[0, 1)` with uniform spacing — `1.0` is never
/// returned (max numerator `2^53 - 1`), `0.0` is reachable.
fn next_rand_f64() -> f64 {
    RAND_STATE.with(|s| {
        let mut x = s.get();
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        s.set(x);
        let scrambled = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        (scrambled >> 11) as f64 / ((1u64 << 53) as f64)
    })
}

/// Human-readable runtime type name for diagnostics.
fn value_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "Null",
        Value::Boolean(_) => "Boolean",
        Value::Integer(_) => "Integer",
        Value::Float(_) => "Float",
        Value::String(_) => "String",
        Value::Node(_) => "Node",
        Value::Relationship(_) => "Relationship",
        Value::List(_) => "List",
        Value::Map(_) => "Map",
        Value::Path(_) => "Path",
        Value::Temporal(_) => "Temporal",
        Value::LocalDateTime(_) => "LocalDateTime",
        Value::Date(_) => "Date",
        Value::Duration(_) => "Duration",
        Value::Decimal(_) => "Decimal",
    }
}

/// Extract a `&str` from a String value or surface a runtime type
/// error. NULL is the caller's responsibility (propagated before
/// this is reached).
fn expect_string<'a>(v: &'a Value, name: &str) -> Result<&'a str, ExecutionError> {
    match v {
        Value::String(s) => Ok(s),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a String argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// Extract an `i64` from an Integer value or surface a runtime type
/// error (the openCypher `InvalidArgumentType` analog — e.g. a Float
/// index is rejected per the TCK `List11` Scenario [5]).
fn expect_integer(v: &Value, name: &str) -> Result<i64, ExecutionError> {
    match v {
        Value::Integer(n) => Ok(*n),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires an Integer argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// Apply a `&str -> String` transform under NULL propagation. Errors
/// on a non-string, non-NULL argument.
fn str_fn(v: &Value, name: &str, f: impl Fn(&str) -> String) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::String(s) => Ok(Value::String(f(s))),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a String argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// Apply an `f64 -> f64` transform, widening Integer to Float and
/// propagating NULL. Errors on a non-numeric, non-NULL argument.
/// Used by `ceil`/`floor`/`round`/`sqrt`/`exp`/`log`/`log10`/`sin`/
/// `cos`/`tan` (all return Float per openCypher v9 §3).
fn num_to_float(v: &Value, name: &str, f: impl Fn(f64) -> f64) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Integer(n) => Ok(Value::Float(f(*n as f64))),
        Value::Float(x) => Ok(Value::Float(f(*x))),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a numeric argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `abs(n)` — preserves the numeric type (Integer -> Integer,
/// Float -> Float); errors on `i64::MIN` (no two's-complement
/// representation, consistent with the unary-`-` overflow handling
/// in [`apply_unop`]). (TCK `Mathematical11`: `abs(-1) = 1`.)
fn fn_abs(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Integer(n) => Ok(Value::Integer(n.checked_abs().ok_or_else(|| {
            ExecutionError::Eval(format!("{name}() integer overflow on i64::MIN"))
        })?)),
        Value::Float(x) => Ok(Value::Float(x.abs())),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a numeric argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `sign(n)` — returns an Integer in {-1, 0, 1}. `0.0` / `-0.0` / NaN
/// map to 0 (openCypher leaves NaN's sign unspecified; 0 is the safe
/// total choice). NOT `f64::signum` (which returns ±1.0 for ±0.0 and
/// NaN for NaN).
fn fn_sign(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    let s = match v {
        Value::Null => return Ok(Value::Null),
        Value::Integer(n) => (*n).signum(),
        Value::Float(x) => {
            if *x > 0.0 {
                1
            } else if *x < 0.0 {
                -1
            } else {
                0
            }
        }
        other => {
            return Err(ExecutionError::Eval(format!(
                "{name}() requires a numeric argument, got {}",
                value_type_name(other)
            )));
        }
    };
    Ok(Value::Integer(s))
}

/// `reverse(x)` — polymorphic over String (reverse Unicode codepoints)
/// and List (reverse elements). NULL propagates. (TCK `String3`:
/// `reverse('raksO') = 'Oskar'`.)
fn fn_reverse(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
        Value::List(l) => Ok(Value::List(l.iter().rev().cloned().collect())),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a String or List argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `substring(original, start[, length])` — 0-based char index at
/// Unicode-codepoint granularity. NULL in any position -> NULL.
/// Negative `start` / `length` raise a runtime error; indices beyond
/// the string clamp to the end (Neo4j / openCypher semantics). (TCK
/// `String1`: `substring('0123456789', 1) = '123456789'`.)
fn fn_substring(args: &[Value], name: &str) -> Result<Value, ExecutionError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(ExecutionError::Eval(format!(
            "{name}() expects 2 or 3 arguments, got {}",
            args.len()
        )));
    }
    if args.iter().any(Value::is_null) {
        return Ok(Value::Null);
    }
    let s = expect_string(&args[0], name)?;
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len() as i64;
    let start = expect_integer(&args[1], name)?;
    if start < 0 {
        return Err(ExecutionError::Eval(format!(
            "{name}() start index must be non-negative, got {start}"
        )));
    }
    let start = start.min(len);
    let take = match args.get(2) {
        Some(v) => {
            let l = expect_integer(v, name)?;
            if l < 0 {
                return Err(ExecutionError::Eval(format!(
                    "{name}() length must be non-negative, got {l}"
                )));
            }
            l
        }
        None => len - start,
    };
    let end = start.saturating_add(take).min(len);
    // start <= end <= len, all non-negative — the cast + slice is safe.
    let out: String = chars[start as usize..end as usize].iter().collect();
    Ok(Value::String(out))
}

/// `left(original, length)` / `right(original, length)` — first / last
/// `length` Unicode chars; `length` beyond the string returns the
/// whole string. NULL propagates; negative `length` errors.
///
/// Neo4j extension (Neo4j Cypher manual, String functions), not core
/// openCypher — the TCK `String8`/`String9` cover `STARTS WITH`/
/// `ENDS WITH`, not `left`/`right`.
fn fn_left_right(args: &[Value], name: &str, left: bool) -> Result<Value, ExecutionError> {
    if args.len() != 2 {
        return Err(ExecutionError::Eval(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    }
    if args.iter().any(Value::is_null) {
        return Ok(Value::Null);
    }
    let s = expect_string(&args[0], name)?;
    let n = expect_integer(&args[1], name)?;
    if n < 0 {
        return Err(ExecutionError::Eval(format!(
            "{name}() length must be non-negative, got {n}"
        )));
    }
    let chars: Vec<char> = s.chars().collect();
    let n = (n as usize).min(chars.len());
    let out: String = if left {
        chars[..n].iter().collect()
    } else {
        chars[chars.len() - n..].iter().collect()
    };
    Ok(Value::String(out))
}

/// `replace(original, search, replacement)` — replace ALL
/// non-overlapping occurrences. NULL in any position -> NULL.
fn fn_replace(args: &[Value], name: &str) -> Result<Value, ExecutionError> {
    if args.len() != 3 {
        return Err(ExecutionError::Eval(format!(
            "{name}() expects 3 arguments, got {}",
            args.len()
        )));
    }
    if args.iter().any(Value::is_null) {
        return Ok(Value::Null);
    }
    let s = expect_string(&args[0], name)?;
    let search = expect_string(&args[1], name)?;
    let repl = expect_string(&args[2], name)?;
    Ok(Value::String(s.replace(search, repl)))
}

/// `split(original, delimiter)` — split on a substring delimiter,
/// returning a List of String. An empty delimiter returns the whole
/// string as the single element (avoids the surprising per-char split
/// of `str::split("")`). NULL in either position -> NULL. (TCK
/// `String4`: `split('one1two', '1')` has size 2.)
fn fn_split(args: &[Value], name: &str) -> Result<Value, ExecutionError> {
    if args.len() != 2 {
        return Err(ExecutionError::Eval(format!(
            "{name}() expects 2 arguments, got {}",
            args.len()
        )));
    }
    if args[0].is_null() || args[1].is_null() {
        return Ok(Value::Null);
    }
    let s = expect_string(&args[0], name)?;
    let delim = expect_string(&args[1], name)?;
    let parts: Vec<Value> = if delim.is_empty() {
        vec![Value::String(s.to_string())]
    } else {
        s.split(delim)
            .map(|p| Value::String(p.to_string()))
            .collect()
    };
    Ok(Value::List(parts))
}

/// `toInteger(x)` — Integer pass-through, Float truncates toward zero,
/// String parses (Integer, then Float-truncate) returning NULL on
/// failure, NULL -> NULL. Other types raise an `InvalidArgumentValue`
/// analog. (TCK `TypeConversion2`: `toInteger(82.9) = 82`,
/// `toInteger('1.7') = 1`, `toInteger('foo') = null`.)
fn to_integer(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Integer(n) => Ok(Value::Integer(*n)),
        Value::Float(f) => Ok(float_to_integer(*f)),
        Value::String(s) => Ok(parse_integer(s)),
        other => Err(ExecutionError::Eval(format!(
            "{name}() invalid argument type: {}",
            value_type_name(other)
        ))),
    }
}

/// Truncate a finite, in-range `f64` toward zero to an Integer; an
/// out-of-range / non-finite value yields NULL (openCypher returns
/// NULL rather than overflowing).
fn float_to_integer(f: f64) -> Value {
    let t = f.trunc();
    if t.is_finite() && t >= i64::MIN as f64 && t <= i64::MAX as f64 {
        Value::Integer(t as i64)
    } else {
        Value::Null
    }
}

/// Parse a string to an Integer: exact `i64` first, then a
/// Float-truncate fallback (`'1.7' -> 1`), else NULL. Leading /
/// trailing whitespace is trimmed (Neo4j-compatible).
fn parse_integer(s: &str) -> Value {
    let t = s.trim();
    if let Ok(n) = t.parse::<i64>() {
        return Value::Integer(n);
    }
    if let Ok(f) = t.parse::<f64>() {
        return float_to_integer(f);
    }
    Value::Null
}

/// `toFloat(x)` — Integer widens, Float pass-through, String parses
/// (NULL on failure), NULL -> NULL. Boolean is a TCK-confirmed invalid
/// type (`TypeConversion3` Scenario [6]); List / Node / Relationship
/// likewise raise an `InvalidArgumentValue` analog.
fn to_float(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Integer(n) => Ok(Value::Float(*n as f64)),
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::String(s) => Ok(match s.trim().parse::<f64>() {
            Ok(f) => Value::Float(f),
            Err(_) => Value::Null,
        }),
        other => Err(ExecutionError::Eval(format!(
            "{name}() invalid argument type: {}",
            value_type_name(other)
        ))),
    }
}

/// `toBoolean(x)` — Boolean pass-through; String `"true"`/`"false"`
/// (case-insensitive, whitespace-trimmed) else NULL; NULL -> NULL.
/// Any other type (incl. Float, TCK-confirmed by `TypeConversion1`
/// Scenario [5]) raises an `InvalidArgumentValue` analog. (TCK
/// `TypeConversion1`: `toBoolean('true') = true`, invalid strings
/// like `' tru '` / `'f alse'` / `''` -> null.)
fn to_boolean(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Boolean(b) => Ok(Value::Boolean(*b)),
        Value::String(s) => Ok(match s.trim().to_ascii_lowercase().as_str() {
            "true" => Value::Boolean(true),
            "false" => Value::Boolean(false),
            _ => Value::Null,
        }),
        other => Err(ExecutionError::Eval(format!(
            "{name}() invalid argument type: {}",
            value_type_name(other)
        ))),
    }
}

/// `toString(x)` — Integer / Float / Boolean / String render to their
/// canonical openCypher string form; NULL -> NULL. List / Node /
/// Relationship / temporal raise an `InvalidArgumentValue` analog
/// (`TypeConversion4` Scenario [10]; temporal `toString` is owned by
/// the separate temporal surface, not this slice). (TCK
/// `TypeConversion4`: `toString(42) = '42'`, `toString(true) =
/// 'true'`, `toString(2.3) = '2.3'`.)
fn to_string_value(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::String(s) => Ok(Value::String(s.clone())),
        Value::Integer(n) => Ok(Value::String(n.to_string())),
        Value::Float(f) => Ok(Value::String(cypher_float_string(*f))),
        Value::Boolean(b) => Ok(Value::String(b.to_string())),
        other => Err(ExecutionError::Eval(format!(
            "{name}() invalid argument type: {}",
            value_type_name(other)
        ))),
    }
}

/// openCypher-style `f64 -> String`: whole-valued finite floats carry
/// an explicit `.0` (e.g. `toString(3.0) = "3.0"`); fractional floats
/// use the shortest round-trip form (`toString(2.3) = "2.3"`).
/// Non-finite floats render as `NaN` / `Infinity` / `-Infinity`. The
/// `< 1e15` guard keeps whole floats out of scientific-notation
/// territory where the `.1`-format would balloon.
///
/// v1.0 divergence: for `|f| >= 1e15` this emits the Rust `{}` form
/// (e.g. `toString(1e15) = "1000000000000000"`), whereas Neo4j renders
/// the scientific form `"1.0E15"`. Out of v1.0 scope — the max float
/// magnitude across the vendored openCypher TCK corpus is small.
fn cypher_float_string(f: f64) -> String {
    if f.is_nan() {
        return "NaN".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if f == f.trunc() && f.abs() < 1e15 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// `coalesce(...)` — the first non-NULL argument, or NULL if all are
/// NULL. (Exercised by TCK `TypeConversion4` Scenarios [8]/[9].)
fn fn_coalesce(args: &[Value]) -> Value {
    for v in args {
        if !v.is_null() {
            return v.clone();
        }
    }
    Value::Null
}

/// ADR-147-amendment-03 (D-1, §2b-iv) — hard cap on the element count a
/// single `range(...)` call may materialize. Defense-in-depth against a
/// pre-existing read-path DoS: `range(1, 9e18)` would otherwise allocate
/// ~9 quintillion `Value::Integer`s and OOM the process. `range()` is
/// ALREADY rejected as a CREATE property value (`FunctionCall` denied at
/// type-check), so this guards the read/expression path where `range()`
/// remains reachable. The cap matches
/// `create_spine::MAX_CREATE_PROP_LIST_LEN` (1M elements ≈ 24 MB of
/// in-memory `Value::Integer`s) — abusive for a single expression, three-
/// plus orders of magnitude above any legitimate `range()` use.
const MAX_RANGE_LEN: u64 = 1_000_000;

/// `range(start, end[, step])` — inclusive integer range. All
/// arguments MUST be Integer (Float / String / Boolean / List raise
/// the `InvalidArgumentType` analog, per the TCK `List11` Scenario
/// [5]); `step` defaults to 1 and MUST be non-zero (`NumberOutOfRange`,
/// Scenario [4]). A NULL argument propagates to NULL. An inconsistent
/// range/step direction yields an empty list (Scenario [3]). (TCK
/// `List11`: `range(0, 10) = [0..10]`, `range(10, -10, -3) =
/// [10,7,4,1,-2,-5,-8]`.)
fn fn_range(args: &[Value], name: &str) -> Result<Value, ExecutionError> {
    if args.len() < 2 || args.len() > 3 {
        return Err(ExecutionError::Eval(format!(
            "{name}() expects 2 or 3 arguments, got {}",
            args.len()
        )));
    }
    if args.iter().any(Value::is_null) {
        return Ok(Value::Null);
    }
    let start = expect_integer(&args[0], name)?;
    let end = expect_integer(&args[1], name)?;
    let step = match args.get(2) {
        Some(v) => expect_integer(v, name)?,
        None => 1,
    };
    if step == 0 {
        return Err(ExecutionError::Eval(format!(
            "{name}() step argument must be non-zero (NumberOutOfRange)"
        )));
    }
    // ADR-147-amendment-03 (§2b-iv): reject a range whose element count
    // would exceed the cap BEFORE materializing it. Element count for an
    // inclusive range is `floor(|end - start| / |step|) + 1` when the
    // direction agrees with `step`; a direction-mismatched range is
    // empty. Compute the span with `i128` to avoid `i64` overflow on
    // `range(i64::MIN, i64::MAX)`.
    let span = (i128::from(end) - i128::from(start)).unsigned_abs();
    let stride = i128::from(step).unsigned_abs();
    let directed = (end >= start) == (step > 0);
    if directed {
        let count = span / stride + 1;
        if count > u128::from(MAX_RANGE_LEN) {
            return Err(ExecutionError::Eval(format!(
                "{name}() would materialize {count} elements, exceeding cap {MAX_RANGE_LEN} \
                 (NumberOutOfRange)"
            )));
        }
    }
    let mut out = Vec::new();
    let mut cur = start;
    if step > 0 {
        while cur <= end {
            out.push(Value::Integer(cur));
            match cur.checked_add(step) {
                Some(n) => cur = n,
                None => break,
            }
        }
    } else {
        while cur >= end {
            out.push(Value::Integer(cur));
            match cur.checked_add(step) {
                Some(n) => cur = n,
                None => break,
            }
        }
    }
    Ok(Value::List(out))
}

/// `size(x)` — element count for a List, Unicode-char count for a
/// String, NULL -> NULL. (TCK `List6`: `size([1,2,3]) = 3`,
/// `size(null) = null`.)
fn fn_size(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::List(l) => Ok(Value::Integer(l.len() as i64)),
        Value::String(s) => Ok(Value::Integer(s.chars().count() as i64)),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a List or String argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `length(x)` — ADR-193 D-7: for a [`Value::Path`], the hop-count
/// (`#rels = segments.len()`). For backward compatibility it ALSO
/// evaluates the legacy List / String count form via [`fn_size`] (a
/// harmless superset of `size`, preserved for regression). NULL -> NULL;
/// any other type -> the `InvalidArgumentType` analog (via `fn_size`).
fn fn_length(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Path(p) => Ok(Value::Integer(p.hop_count() as i64)),
        _ => fn_size(v, name),
    }
}

/// `nodes(path)` — ADR-193 D-7: the path's nodes in TRAVERSAL order as a
/// `List(Node)` (always `length(p) + 1` nodes). NULL -> NULL (3VL);
/// any non-path, non-NULL argument -> the openCypher `InvalidArgumentType`
/// analog (`ExecutionError::Eval`), co-located reject test per the #723
/// lesson.
fn fn_nodes(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Path(p) => Ok(Value::List(
            p.nodes().into_iter().map(Value::Node).collect(),
        )),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a Path argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `relationships(path)` — ADR-193 D-7: the path's relationships in
/// TRAVERSAL order as a `List(Relationship)` (relationship IDENTITY /
/// stored orientation preserved). NULL -> NULL (3VL); any non-path,
/// non-NULL argument -> the `InvalidArgumentType` analog.
fn fn_relationships(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Path(p) => Ok(Value::List(
            p.relationships()
                .into_iter()
                .map(Value::Relationship)
                .collect(),
        )),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a Path argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `head(list)` — first element, or NULL for an empty list; NULL -> NULL.
fn fn_head(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::List(l) => Ok(l.first().cloned().unwrap_or(Value::Null)),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a List argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `last(list)` — last element, or NULL for an empty list; NULL -> NULL.
fn fn_last(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::List(l) => Ok(l.last().cloned().unwrap_or(Value::Null)),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a List argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `tail(list)` — all but the first element (an empty list for a
/// 0/1-element list); NULL -> NULL. (TCK `List9`:
/// `tail(tail([1,2,3,4,5])) = [3,4,5]`.)
fn fn_tail(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::List(l) => Ok(Value::List(l.iter().skip(1).cloned().collect())),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a List argument, got {}",
            value_type_name(other)
        ))),
    }
}

/// `keys(node|rel|map)` — the property / entry keys as a List of String
/// (BTreeMap-sorted; the TCK ignores list order for `keys`); NULL -> NULL.
/// openCypher v9 §3 admits a MAP argument (#618) alongside node/rel — a
/// literal / parameter / projected map yields its key set (`keys({a:1,
/// b:2}) = ['a','b']`, `keys({}) = []`). (TCK `Map3` Scenario [3] backs
/// `keys(null) = null`; `Map3`'s map scenarios exercise literal /
/// parameter MAPS, now first-class here.)
fn fn_keys(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    let props = match v {
        Value::Null => return Ok(Value::Null),
        Value::Node(n) => &n.properties,
        Value::Relationship(r) => &r.properties,
        // #618 — `keys(map)` returns the map's keys (BTreeMap-sorted).
        Value::Map(m) => m,
        other => {
            return Err(ExecutionError::Eval(format!(
                "{name}() requires a Node, Relationship, or Map argument, got {}",
                value_type_name(other)
            )));
        }
    };
    Ok(Value::List(
        props.keys().map(|k| Value::String(k.clone())).collect(),
    ))
}

/// `properties(node|rel|map)` — the property map of the argument as a
/// [`Value::Map`] (openCypher v9 §3; #618); NULL -> NULL. For a node /
/// relationship, a copy of its property bag; for a MAP argument it is the
/// IDENTITY (`properties(m) == m`). The returned map is BTreeMap-backed
/// (deterministic key order), matching the map-literal carrier + `keys`.
/// A scalar / list LITERAL is rejected at COMPILE time by the
/// `ArgKind::MapLike` registry constraint (openCypher raises
/// `InvalidArgumentType`); this eval arm is the RUNTIME BACKSTOP for a
/// dynamically-typed `Property` access (admitted at compile time to
/// avoid a false-positive on the under-typed v1.0 catalog) that resolves
/// to a non-entity, non-map value.
fn fn_properties(v: &Value, name: &str) -> Result<Value, ExecutionError> {
    match v {
        Value::Null => Ok(Value::Null),
        Value::Node(n) => Ok(Value::Map(n.properties.clone())),
        Value::Relationship(r) => Ok(Value::Map(r.properties.clone())),
        // `properties(map)` is the identity (openCypher v9 §3).
        Value::Map(m) => Ok(Value::Map(m.clone())),
        other => Err(ExecutionError::Eval(format!(
            "{name}() requires a Node, Relationship, or Map argument, got {}",
            value_type_name(other)
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Span;

    fn lit_int(n: i64) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Integer(n),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn lit_null() -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Null,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn lit_bool(b: bool) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Bool(b),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// openCypher v9 §3.3.6 string predicates — kernel-level oracle on
    /// `apply_binop`, INDEPENDENT of the grammar/binder (the `tests/
    /// string_predicates_e2e.rs` suite proves the full path). Pins the
    /// prefix/suffix/substring truth, the non-string⇒null + null-propagation
    /// 3VL rules, UTF-8 codepoint-correctness, case-sensitivity, and the
    /// empty-needle edge — exact VALUES, not "no error" (#773).
    #[test]
    fn string_predicate_kernel_apply_binop() {
        let st = |s: &str| Value::String(s.to_string());
        // prefix / suffix / substring — true and false.
        assert_eq!(
            apply_binop(BinOp::StartsWith, st("hello"), st("he")).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            apply_binop(BinOp::StartsWith, st("hello"), st("x")).unwrap(),
            Value::Boolean(false)
        );
        assert_eq!(
            apply_binop(BinOp::EndsWith, st("hello"), st("lo")).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            apply_binop(BinOp::EndsWith, st("hello"), st("x")).unwrap(),
            Value::Boolean(false)
        );
        assert_eq!(
            apply_binop(BinOp::Contains, st("hello"), st("ell")).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            apply_binop(BinOp::Contains, st("hello"), st("xyz")).unwrap(),
            Value::Boolean(false)
        );
        // null propagation — either operand null ⇒ null.
        assert_eq!(
            apply_binop(BinOp::StartsWith, Value::Null, st("he")).unwrap(),
            Value::Null
        );
        assert_eq!(
            apply_binop(BinOp::Contains, st("hello"), Value::Null).unwrap(),
            Value::Null
        );
        // non-string operand ⇒ null (NOT an error, NOT false).
        assert_eq!(
            apply_binop(BinOp::StartsWith, Value::Boolean(true), st("abc")).unwrap(),
            Value::Null
        );
        assert_eq!(
            apply_binop(BinOp::Contains, Value::Integer(5), st("a")).unwrap(),
            Value::Null
        );
        // UTF-8 codepoint-correct substring/prefix/suffix.
        assert_eq!(
            apply_binop(BinOp::Contains, st("héllo"), st("éll")).unwrap(),
            Value::Boolean(true)
        );
        assert_eq!(
            apply_binop(BinOp::StartsWith, st("héllo"), st("hé")).unwrap(),
            Value::Boolean(true)
        );
        // case-SENSITIVE.
        assert_eq!(
            apply_binop(BinOp::StartsWith, st("Hello"), st("h")).unwrap(),
            Value::Boolean(false)
        );
        // empty needle ⇒ true (every string contains/starts-with/ends-with "").
        assert_eq!(
            apply_binop(BinOp::Contains, st("hello"), st("")).unwrap(),
            Value::Boolean(true)
        );
    }

    fn no_schema() -> Box<dyn Fn(BindingId) -> Option<usize>> {
        Box::new(|_| None)
    }

    // =================================================================
    // #621 — `values_equal_3vl` (openCypher v9 §3.3.5 / §3.4 3VL value
    // equality, shared by `IN` membership and `=` / `<>`).
    // =================================================================

    fn vi(n: i64) -> Value {
        Value::Integer(n)
    }
    fn vl(xs: Vec<Value>) -> Value {
        Value::List(xs)
    }

    #[test]
    fn veq3vl_scalar_type_mismatch_is_definite_false() {
        // `1 = '1'` ⇒ false (type mismatch is DEFINITELY false, NOT null).
        assert_eq!(
            values_equal_3vl(&vi(1), &Value::String("1".into())),
            Some(false)
        );
        assert_eq!(values_equal_3vl(&vi(1), &vi(2)), Some(false));
        assert_eq!(values_equal_3vl(&vi(1), &vi(1)), Some(true));
        // Integer/Float numeric cross-compare.
        assert_eq!(values_equal_3vl(&vi(2), &Value::Float(2.0)), Some(true));
    }

    #[test]
    fn veq3vl_null_operand_is_unknown() {
        assert_eq!(values_equal_3vl(&Value::Null, &vi(1)), None);
        assert_eq!(values_equal_3vl(&vi(1), &Value::Null), None);
        assert_eq!(values_equal_3vl(&Value::Null, &Value::Null), None);
    }

    #[test]
    fn veq3vl_list_length_mismatch_is_false() {
        // Unequal length ⇒ definite false, even with a null present.
        assert_eq!(
            values_equal_3vl(&vl(vec![vi(1)]), &vl(vec![vi(1), Value::Null])),
            Some(false)
        );
    }

    #[test]
    fn veq3vl_nested_null_propagates_to_unknown() {
        // [1,2] = [null,2] ⇒ Unknown (pos0 unknown, pos1 true, no false).
        assert_eq!(
            values_equal_3vl(&vl(vec![vi(1), vi(2)]), &vl(vec![Value::Null, vi(2)])),
            None
        );
        // [1,2,null] = [1,2,null] ⇒ Unknown (null=null at pos2).
        assert_eq!(
            values_equal_3vl(
                &vl(vec![vi(1), vi(2), Value::Null]),
                &vl(vec![vi(1), vi(2), Value::Null])
            ),
            None
        );
    }

    #[test]
    fn veq3vl_definite_mismatch_dominates_nested_null() {
        // [1,2] = [null,3] ⇒ false (pos1 `2=3` is a definite mismatch,
        // which DOMINATES the pos0 unknown).
        assert_eq!(
            values_equal_3vl(&vl(vec![vi(1), vi(2)]), &vl(vec![Value::Null, vi(3)])),
            Some(false)
        );
    }

    #[test]
    fn veq3vl_empty_and_list_vs_scalar() {
        assert_eq!(values_equal_3vl(&vl(vec![]), &vl(vec![])), Some(true));
        // list vs scalar ⇒ definite false (type mismatch).
        assert_eq!(values_equal_3vl(&vl(vec![vi(1)]), &vi(1)), Some(false));
    }

    // =================================================================
    // #621 — `eval_subscript` (openCypher v9 §3.4 list indexing).
    // =================================================================

    #[test]
    fn subscript_basic_negative_and_out_of_range() {
        let l = || vl(vec![vi(10), vi(20), vi(30)]);
        assert_eq!(eval_subscript(l(), vi(0)).unwrap(), vi(10));
        assert_eq!(eval_subscript(l(), vi(2)).unwrap(), vi(30));
        assert_eq!(eval_subscript(l(), vi(-1)).unwrap(), vi(30)); // from end
        assert_eq!(eval_subscript(l(), vi(-3)).unwrap(), vi(10));
        // Out of range ⇒ null (NOT an error).
        assert_eq!(eval_subscript(l(), vi(3)).unwrap(), Value::Null);
        assert_eq!(eval_subscript(l(), vi(-4)).unwrap(), Value::Null);
    }

    #[test]
    fn subscript_null_propagation() {
        assert_eq!(eval_subscript(Value::Null, vi(0)).unwrap(), Value::Null);
        assert_eq!(
            eval_subscript(vl(vec![vi(1)]), Value::Null).unwrap(),
            Value::Null
        );
    }

    // =================================================================
    // #621 — `eval_slice` (openCypher v9 §3.4 list slicing).
    // =================================================================

    #[test]
    fn slice_basic_open_and_negative_bounds() {
        let l = || vl(vec![vi(10), vi(20), vi(30), vi(40)]);
        assert_eq!(
            eval_slice(l(), Some(vi(0)), Some(vi(2))).unwrap(),
            vl(vec![vi(10), vi(20)])
        );
        // Open bounds.
        assert_eq!(
            eval_slice(l(), None, Some(vi(2))).unwrap(),
            vl(vec![vi(10), vi(20)])
        );
        assert_eq!(
            eval_slice(l(), Some(vi(2)), None).unwrap(),
            vl(vec![vi(30), vi(40)])
        );
        assert_eq!(eval_slice(l(), None, None).unwrap(), l());
        // Negative bounds count from the end.
        assert_eq!(
            eval_slice(l(), Some(vi(1)), Some(vi(-1))).unwrap(),
            vl(vec![vi(20), vi(30)])
        );
        // lo >= hi ⇒ empty; out-of-range bounds clamp.
        assert_eq!(
            eval_slice(l(), Some(vi(2)), Some(vi(1))).unwrap(),
            vl(vec![])
        );
        assert_eq!(eval_slice(l(), Some(vi(0)), Some(vi(99))).unwrap(), l());
    }

    #[test]
    fn slice_null_base_and_present_null_bound() {
        assert_eq!(eval_slice(Value::Null, None, None).unwrap(), Value::Null);
        // A PRESENT null bound ⇒ the whole slice is null (3VL).
        assert_eq!(
            eval_slice(vl(vec![vi(1)]), Some(Value::Null), None).unwrap(),
            Value::Null
        );
        assert_eq!(
            eval_slice(vl(vec![vi(1)]), None, Some(Value::Null)).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn arithmetic_integer_add() {
        let e = BoundExpression::BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(lit_int(2)),
            rhs: Box::new(lit_int(3)),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Integer(5));
    }

    #[test]
    fn arithmetic_null_propagates() {
        let e = BoundExpression::BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(lit_int(2)),
            rhs: Box::new(lit_null()),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn integer_division_by_zero_errors() {
        let e = BoundExpression::BinaryOp {
            op: BinOp::Div,
            lhs: Box::new(lit_int(10)),
            rhs: Box::new(lit_int(0)),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let r = evaluate(&e, &[], &*s, &Parameters::new());
        assert!(matches!(r, Err(ExecutionError::Eval(_))));
    }

    #[test]
    fn comparison_null_returns_null_3vl() {
        // 3VL: `null > 5` → Unknown → Value::Null.
        let e = BoundExpression::BinaryOp {
            op: BinOp::Gt,
            lhs: Box::new(lit_null()),
            rhs: Box::new(lit_int(5)),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Null);
    }

    #[test]
    fn is_null_tunnels_through_3vl() {
        // IS NULL on a NULL operand → True. Crucially returns
        // Boolean(true), NOT Null. Cypher 9 §6.2.5.
        let e = BoundExpression::IsNull {
            lhs: Box::new(lit_null()),
            negated: false,
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Boolean(true));

        // IS NOT NULL on a NULL operand → False.
        let e = BoundExpression::IsNull {
            lhs: Box::new(lit_null()),
            negated: true,
            span: Span::point(1, 1),
            type_info: None,
        };
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Boolean(false));
    }

    #[test]
    fn boolean_and_routes_through_3vl() {
        // True AND Null → Null (not Boolean(false)).
        let e = BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs: Box::new(lit_bool(true)),
            rhs: Box::new(lit_null()),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Null);

        // False AND Null → False (NULL absorbs into False).
        let e = BoundExpression::BinaryOp {
            op: BinOp::And,
            lhs: Box::new(lit_bool(false)),
            rhs: Box::new(lit_null()),
            span: Span::point(1, 1),
            type_info: None,
        };
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Boolean(false));
    }

    #[test]
    fn parameter_resolution_threads_through_bag() {
        let e = BoundExpression::Parameter {
            name: "k".into(),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let mut params = Parameters::new();
        params.insert("k".into(), Value::Integer(42));
        let v = evaluate(&e, &[], &*s, &params).unwrap();
        assert_eq!(v, Value::Integer(42));

        // #797 — a missing bind surfaces as the typed `MissingParameter`
        // (NOT the generic `Eval`), carrying the offending name so the
        // wire layer can render a client error (Bolt ParameterMissing /
        // MCP -32602) rather than a server-fault bucket.
        let r = evaluate(&e, &[], &*s, &Parameters::new());
        assert!(
            matches!(&r, Err(ExecutionError::MissingParameter { name }) if name == "k"),
            "expected MissingParameter {{ name: \"k\" }}, got {r:?}"
        );
    }

    #[test]
    fn variable_ref_indexes_via_schema() {
        let bid = BindingId::new(7);
        let row = vec![Value::Integer(11), Value::String("hi".into())];
        let s: Box<dyn Fn(BindingId) -> Option<usize>> =
            Box::new(move |id| if id == bid { Some(0) } else { None });
        let e = BoundExpression::VariableRef {
            name: "x".into(),
            binding_id: bid,
            span: Span::point(1, 1),
            type_info: None,
        };
        let v = evaluate(&e, &row, &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Integer(11));
    }

    #[test]
    fn unary_not_routes_through_3vl() {
        // NOT Null → Null (not Boolean(true)).
        let e = BoundExpression::UnaryOp {
            op: UnaryOp::Not,
            operand: Box::new(lit_null()),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        let v = evaluate(&e, &[], &*s, &Parameters::new()).unwrap();
        assert_eq!(v, Value::Null);
    }

    // ================================================================
    // W28 conformance scalar built-ins (Task #652)
    //
    // Strong oracles: each asserts the openCypher-correct VALUE (incl.
    // NULL propagation + type-coercion edges the TCK pins), NOT merely
    // "no error". Most go through `evaluate(FunctionCall)` so they pin
    // the registry-name -> eval-arm wiring together with the semantics;
    // helper-direct tests cover Values (Node) that aren't literal-
    // constructible. Full engine coverage lives in
    // `tests/w28_conformance_scalar_fns_e2e.rs`.
    // ================================================================

    fn lit_str(s: &str) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::String(s.into()),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn lit_float(f: f64) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Float(f),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// A literal list of integers (evaluates to `Value::List` of
    /// `Value::Integer` via `literal_to_value`).
    fn lit_list_ints(vs: &[i64]) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::List(
                vs.iter()
                    .map(|n| Expression::Literal(Literal::Integer(*n)))
                    .collect(),
            ),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn call(name: &str, args: Vec<BoundExpression>) -> BoundExpression {
        BoundExpression::FunctionCall {
            name: name.into(),
            args,
            distinct: false,
            star: false,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Evaluate a function-call expr with no row / schema / params;
    /// expect success.
    fn ev(expr: &BoundExpression) -> Value {
        let s = no_schema();
        evaluate(expr, &[], &*s, &Parameters::new()).expect("eval")
    }

    /// Evaluate a function-call expr, returning the `Result` (for the
    /// runtime-error edge tests).
    fn ev_res(expr: &BoundExpression) -> Result<Value, ExecutionError> {
        let s = no_schema();
        evaluate(expr, &[], &*s, &Parameters::new())
    }

    fn vlist(vs: Vec<Value>) -> Value {
        Value::List(vs)
    }

    fn vints(vs: &[i64]) -> Value {
        Value::List(vs.iter().copied().map(Value::Integer).collect())
    }

    #[test]
    fn w28_string_functions_dispatch() {
        assert_eq!(
            ev(&call("toUpper", vec![lit_str("aBc")])),
            Value::String("ABC".into())
        );
        assert_eq!(
            ev(&call("toLower", vec![lit_str("aBc")])),
            Value::String("abc".into())
        );
        assert_eq!(
            ev(&call("trim", vec![lit_str("  hi  ")])),
            Value::String("hi".into())
        );
        assert_eq!(
            ev(&call("lTrim", vec![lit_str("  hi  ")])),
            Value::String("hi  ".into())
        );
        assert_eq!(
            ev(&call("rTrim", vec![lit_str("  hi  ")])),
            Value::String("  hi".into())
        );
        // TCK String3.
        assert_eq!(
            ev(&call("reverse", vec![lit_str("raksO")])),
            Value::String("Oskar".into())
        );
        // reverse is polymorphic over List.
        assert_eq!(
            ev(&call("reverse", vec![lit_list_ints(&[1, 2, 3])])),
            vints(&[3, 2, 1])
        );
        // TCK String1 — substring 2-arg (to end) + 3-arg.
        assert_eq!(
            ev(&call("substring", vec![lit_str("0123456789"), lit_int(1)])),
            Value::String("123456789".into())
        );
        assert_eq!(
            ev(&call(
                "substring",
                vec![lit_str("hello"), lit_int(1), lit_int(3)]
            )),
            Value::String("ell".into())
        );
        // replace + left/right (clamp beyond length).
        assert_eq!(
            ev(&call(
                "replace",
                vec![lit_str("hello"), lit_str("l"), lit_str("L")]
            )),
            Value::String("heLLo".into())
        );
        assert_eq!(
            ev(&call("left", vec![lit_str("hello"), lit_int(3)])),
            Value::String("hel".into())
        );
        assert_eq!(
            ev(&call("right", vec![lit_str("hello"), lit_int(3)])),
            Value::String("llo".into())
        );
        assert_eq!(
            ev(&call("left", vec![lit_str("hi"), lit_int(10)])),
            Value::String("hi".into())
        );
        // NULL propagation across the string family.
        assert_eq!(ev(&call("toUpper", vec![lit_null()])), Value::Null);
        assert_eq!(
            ev(&call("substring", vec![lit_null(), lit_int(1)])),
            Value::Null
        );
        assert_eq!(
            ev(&call(
                "replace",
                vec![lit_str("x"), lit_null(), lit_str("y")]
            )),
            Value::Null
        );
        assert_eq!(
            ev(&call("left", vec![lit_str("x"), lit_null()])),
            Value::Null
        );
        // Type error on non-String argument.
        assert!(ev_res(&call("toUpper", vec![lit_int(1)])).is_err());
    }

    #[test]
    fn w28_split_dispatch_tck_string4() {
        // TCK String4: split('one1two','1') -> ['one','two'] (size 2).
        assert_eq!(
            ev(&call("split", vec![lit_str("one1two"), lit_str("1")])),
            vlist(vec![
                Value::String("one".into()),
                Value::String("two".into())
            ])
        );
        // Consecutive delimiters preserve empty segments.
        assert_eq!(
            ev(&call("split", vec![lit_str("a,,b"), lit_str(",")])),
            vlist(vec![
                Value::String("a".into()),
                Value::String(String::new()),
                Value::String("b".into()),
            ])
        );
        assert_eq!(
            ev(&call("split", vec![lit_null(), lit_str(",")])),
            Value::Null
        );
    }

    #[test]
    fn w28_substring_unicode_and_negative_errors() {
        // Unicode-codepoint granularity (not bytes).
        assert_eq!(
            ev(&call(
                "substring",
                vec![lit_str("héllo"), lit_int(1), lit_int(3)]
            )),
            Value::String("éll".into())
        );
        // Negative start / length error at runtime.
        assert!(ev_res(&call("substring", vec![lit_str("abc"), lit_int(-1)])).is_err());
        assert!(
            ev_res(&call(
                "substring",
                vec![lit_str("abc"), lit_int(0), lit_int(-1)]
            ))
            .is_err()
        );
        // Start beyond length clamps to empty string.
        assert_eq!(
            ev(&call("substring", vec![lit_str("abc"), lit_int(10)])),
            Value::String(String::new())
        );
    }

    #[test]
    fn w28_math_functions_dispatch() {
        // TCK Mathematical11 — abs(-1) = 1 (type-preserving).
        assert_eq!(ev(&call("abs", vec![lit_int(-1)])), Value::Integer(1));
        assert_eq!(ev(&call("abs", vec![lit_float(-1.5)])), Value::Float(1.5));
        // TCK Mathematical13 — sqrt(12.96) == 3.6 EXACTLY in f64.
        assert_eq!(ev(&call("sqrt", vec![lit_float(12.96)])), Value::Float(3.6));
        // sign -> Integer in {-1,0,1}; 0.0 maps to 0.
        assert_eq!(ev(&call("sign", vec![lit_int(-7)])), Value::Integer(-1));
        assert_eq!(ev(&call("sign", vec![lit_int(0)])), Value::Integer(0));
        assert_eq!(ev(&call("sign", vec![lit_float(2.5)])), Value::Integer(1));
        assert_eq!(ev(&call("sign", vec![lit_float(0.0)])), Value::Integer(0));
        // ceil/floor/round -> Float; Integer widens.
        assert_eq!(ev(&call("ceil", vec![lit_float(0.1)])), Value::Float(1.0));
        assert_eq!(ev(&call("floor", vec![lit_float(0.9)])), Value::Float(0.0));
        assert_eq!(ev(&call("ceil", vec![lit_int(3)])), Value::Float(3.0));
        // round: half away from zero.
        assert_eq!(ev(&call("round", vec![lit_float(2.5)])), Value::Float(3.0));
        assert_eq!(
            ev(&call("round", vec![lit_float(-2.5)])),
            Value::Float(-3.0)
        );
        assert_eq!(ev(&call("round", vec![lit_float(2.4)])), Value::Float(2.0));
        // transcendental exact anchors.
        assert_eq!(ev(&call("exp", vec![lit_int(0)])), Value::Float(1.0));
        assert_eq!(ev(&call("log", vec![lit_int(1)])), Value::Float(0.0));
        assert_eq!(ev(&call("log10", vec![lit_int(1000)])), Value::Float(3.0));
        assert_eq!(ev(&call("sin", vec![lit_int(0)])), Value::Float(0.0));
        assert_eq!(ev(&call("cos", vec![lit_int(0)])), Value::Float(1.0));
        assert_eq!(ev(&call("tan", vec![lit_int(0)])), Value::Float(0.0));
        // nullary constants.
        assert_eq!(ev(&call("e", vec![])), Value::Float(std::f64::consts::E));
        assert_eq!(ev(&call("pi", vec![])), Value::Float(std::f64::consts::PI));
        // NULL propagation.
        assert_eq!(ev(&call("sqrt", vec![lit_null()])), Value::Null);
        assert_eq!(ev(&call("abs", vec![lit_null()])), Value::Null);
        assert_eq!(ev(&call("sign", vec![lit_null()])), Value::Null);
        // Type error on non-numeric.
        assert!(ev_res(&call("sqrt", vec![lit_str("x")])).is_err());
    }

    #[test]
    fn w28_abs_integer_min_overflows() {
        assert!(ev_res(&call("abs", vec![lit_int(i64::MIN)])).is_err());
    }

    #[test]
    fn ga_rand_eval_uniform_unit_interval_and_arity() {
        // GA-rand (#618) — `rand()` returns a `Float` in `[0, 1)`. There
        // is no VALUE oracle (non-deterministic); the oracle is the
        // RANGE, that the generator actually VARIES (a stuck/constant
        // PRNG would collapse to one value), and the nullary arity guard.
        let mut seen = std::collections::HashSet::new();
        for _ in 0..1000 {
            match ev(&call("rand", vec![])) {
                Value::Float(f) => {
                    assert!((0.0..1.0).contains(&f), "rand() must be in [0, 1), got {f}");
                    seen.insert(f.to_bits());
                }
                other => panic!("rand() must return Float, got {other:?}"),
            }
        }
        // A working uniform `[0, 1)` generator yields ~1000 distinct
        // values over 1000 draws; a constant (broken state) collapses to
        // 1. The `>= 100` floor is loose enough to never flake yet still
        // catches a stuck generator.
        assert!(
            seen.len() >= 100,
            "rand() should vary across calls; only {} distinct in 1000 draws",
            seen.len()
        );
        // Nullary arity guard — `rand(1)` is a runtime `Eval` error
        // (mirrors `e`/`pi`). The type-checker rejects arity at compile
        // time; this exercises the eval-arm defensive guard directly.
        assert!(
            ev_res(&call("rand", vec![lit_int(1)])).is_err(),
            "rand() with an argument must error"
        );
    }

    #[test]
    fn w28_to_integer_tck_typeconversion2() {
        assert_eq!(
            ev(&call("toInteger", vec![lit_float(82.9)])),
            Value::Integer(82)
        );
        assert_eq!(ev(&call("toInteger", vec![lit_str("foo")])), Value::Null);
        assert_eq!(ev(&call("toInteger", vec![lit_str("")])), Value::Null);
        assert_eq!(
            ev(&call("toInteger", vec![lit_str("2")])),
            Value::Integer(2)
        );
        assert_eq!(
            ev(&call("toInteger", vec![lit_str("2.9")])),
            Value::Integer(2)
        );
        // TCK Scenario [4]: '1.7' -> 1 (string parses as float, truncates).
        assert_eq!(
            ev(&call("toInteger", vec![lit_str("1.7")])),
            Value::Integer(1)
        );
        assert_eq!(ev(&call("toInteger", vec![lit_int(5)])), Value::Integer(5));
        assert_eq!(ev(&call("toInteger", vec![lit_null()])), Value::Null);
        // Invalid type (List) errors.
        assert!(ev_res(&call("toInteger", vec![lit_list_ints(&[1])])).is_err());
    }

    #[test]
    fn w28_to_float_tck_typeconversion3() {
        assert_eq!(ev(&call("toFloat", vec![lit_int(3)])), Value::Float(3.0));
        assert_eq!(
            ev(&call("toFloat", vec![lit_float(3.4)])),
            Value::Float(3.4)
        );
        assert_eq!(ev(&call("toFloat", vec![lit_str("5")])), Value::Float(5.0));
        assert_eq!(ev(&call("toFloat", vec![lit_str("foo")])), Value::Null);
        assert_eq!(ev(&call("toFloat", vec![lit_str("")])), Value::Null);
        assert_eq!(ev(&call("toFloat", vec![lit_null()])), Value::Null);
        // Boolean is TCK-confirmed invalid for toFloat (Scenario [6]).
        assert!(ev_res(&call("toFloat", vec![lit_bool(true)])).is_err());
    }

    #[test]
    fn w28_to_boolean_tck_typeconversion1() {
        assert_eq!(
            ev(&call("toBoolean", vec![lit_bool(true)])),
            Value::Boolean(true)
        );
        assert_eq!(
            ev(&call("toBoolean", vec![lit_bool(false)])),
            Value::Boolean(false)
        );
        assert_eq!(
            ev(&call("toBoolean", vec![lit_str("true")])),
            Value::Boolean(true)
        );
        assert_eq!(
            ev(&call("toBoolean", vec![lit_str("false")])),
            Value::Boolean(false)
        );
        // case-insensitive.
        assert_eq!(
            ev(&call("toBoolean", vec![lit_str("TRUE")])),
            Value::Boolean(true)
        );
        // TCK Scenario [4] invalid strings -> null.
        assert_eq!(ev(&call("toBoolean", vec![lit_str("")])), Value::Null);
        assert_eq!(ev(&call("toBoolean", vec![lit_str(" tru ")])), Value::Null);
        assert_eq!(ev(&call("toBoolean", vec![lit_str("f alse")])), Value::Null);
        assert_eq!(ev(&call("toBoolean", vec![lit_null()])), Value::Null);
        // Float is TCK-confirmed invalid (Scenario [5]).
        assert!(ev_res(&call("toBoolean", vec![lit_float(1.0)])).is_err());
    }

    #[test]
    fn w28_to_string_tck_typeconversion4() {
        assert_eq!(
            ev(&call("toString", vec![lit_int(42)])),
            Value::String("42".into())
        );
        assert_eq!(
            ev(&call("toString", vec![lit_bool(true)])),
            Value::String("true".into())
        );
        assert_eq!(
            ev(&call("toString", vec![lit_bool(false)])),
            Value::String("false".into())
        );
        assert_eq!(
            ev(&call("toString", vec![lit_float(2.3)])),
            Value::String("2.3".into())
        );
        assert_eq!(
            ev(&call("toString", vec![lit_str("apa")])),
            Value::String("apa".into())
        );
        // whole-valued float carries explicit ".0".
        assert_eq!(
            ev(&call("toString", vec![lit_float(3.0)])),
            Value::String("3.0".into())
        );
        // NULL -> NULL (TCK Scenario [8]: coalesce(toString(null),'x')='x').
        assert_eq!(ev(&call("toString", vec![lit_null()])), Value::Null);
        // List invalid (Scenario [10]).
        assert!(ev_res(&call("toString", vec![lit_list_ints(&[1])])).is_err());
    }

    #[test]
    fn w28_coalesce() {
        // TCK TypeConversion4 Scenarios [8]/[9] shape.
        assert_eq!(
            ev(&call("coalesce", vec![lit_null(), lit_str("x")])),
            Value::String("x".into())
        );
        assert_eq!(
            ev(&call("coalesce", vec![lit_str("male"), lit_str("x")])),
            Value::String("male".into())
        );
        assert_eq!(
            ev(&call("coalesce", vec![lit_null(), lit_null()])),
            Value::Null
        );
        assert_eq!(ev(&call("coalesce", vec![lit_int(7)])), Value::Integer(7));
    }

    #[test]
    fn w28_range_tck_list11() {
        // default step 1, inclusive.
        assert_eq!(
            ev(&call("range", vec![lit_int(0), lit_int(10)])),
            vints(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        );
        assert_eq!(
            ev(&call("range", vec![lit_int(6), lit_int(10)])),
            vints(&[6, 7, 8, 9, 10])
        );
        assert_eq!(
            ev(&call("range", vec![lit_int(1234), lit_int(1234)])),
            vints(&[1234])
        );
        // start > end with positive step -> empty.
        assert_eq!(
            ev(&call("range", vec![lit_int(0), lit_int(-1)])),
            vints(&[])
        );
        // explicit negative step (TCK List11 Scenario [2] row).
        assert_eq!(
            ev(&call("range", vec![lit_int(10), lit_int(-10), lit_int(-3)])),
            vints(&[10, 7, 4, 1, -2, -5, -8])
        );
        // step 2 truncates to [0].
        assert_eq!(
            ev(&call("range", vec![lit_int(0), lit_int(1), lit_int(2)])),
            vints(&[0])
        );
        // inconsistent direction -> empty (Scenario [3] shape).
        assert_eq!(
            ev(&call("range", vec![lit_int(0), lit_int(1), lit_int(-1)])),
            vints(&[])
        );
    }

    #[test]
    fn w28_range_errors_tck_list11_scenario_4_5() {
        // step 0 -> NumberOutOfRange (Scenario [4]).
        assert!(ev_res(&call("range", vec![lit_int(2), lit_int(8), lit_int(0)])).is_err());
        // non-integer arg -> InvalidArgumentType (Scenario [5]).
        assert!(ev_res(&call("range", vec![lit_float(0.0), lit_int(1)])).is_err());
        assert!(ev_res(&call("range", vec![lit_int(0), lit_str("xyz")])).is_err());
        assert!(ev_res(&call("range", vec![lit_int(0), lit_int(1), lit_bool(true)])).is_err());
        // NULL propagation.
        assert_eq!(
            ev(&call("range", vec![lit_null(), lit_int(5)])),
            Value::Null
        );
    }

    /// NN-3 (CZ range read-path DoS) — `range(1, 9e18)` MUST clean-error
    /// (element count over `MAX_RANGE_LEN`) BEFORE the build loop, never
    /// OOM. The cap is computed in `i128` so the span itself does not
    /// overflow; the guard fires on the COUNT, so no ~9-quintillion-
    /// element `Vec` is ever allocated. Fast + deterministic (the cap
    /// check returns before a single `push`), no actual OOM. This is the
    /// read-path leg of the unified builder-cap family (§B1 caps the
    /// write-path `+` concat amplifier; this caps the `range()` producer).
    #[test]
    fn nn3_range_over_cap_is_clean_error_not_oom() {
        // The canonical DoS: `RETURN range(1, 9e18)` — 9 quintillion
        // elements. Capped, not built.
        let huge = 9_000_000_000_000_000_000_i64;
        let err = ev_res(&call("range", vec![lit_int(1), lit_int(huge)]))
            .expect_err("an over-cap range must be a clean typed error, not OOM");
        match err {
            ExecutionError::Eval(m) => assert!(
                m.contains("exceeding cap") && m.contains("elements"),
                "range DoS error names the element cap; got {m}"
            ),
            other => panic!("expected Eval error, got {other:?}"),
        }
        // Extreme span endpoints (`i64::MIN`..`i64::MAX`) must NOT overflow
        // the span computation — the `i128` widening + count-cap holds.
        let err2 = ev_res(&call("range", vec![lit_int(i64::MIN), lit_int(i64::MAX)]))
            .expect_err("the full i64 span is over-cap and must clean-error");
        assert!(
            matches!(err2, ExecutionError::Eval(ref m) if m.contains("exceeding cap")),
            "full-i64-span range clean-errors on the count cap; got {err2:?}"
        );
        // A range AT the cap (exactly `MAX_RANGE_LEN` elements) still
        // succeeds — the cap rejects only strictly over-cap counts.
        let at_cap = ev_res(&call(
            "range",
            vec![lit_int(1), lit_int(MAX_RANGE_LEN as i64)],
        ))
        .expect("a range of exactly MAX_RANGE_LEN elements is allowed");
        match at_cap {
            Value::List(items) => assert_eq!(
                items.len() as u64,
                MAX_RANGE_LEN,
                "range(1, MAX_RANGE_LEN) materializes exactly MAX_RANGE_LEN elements"
            ),
            other => panic!("expected a List, got {other:?}"),
        }
    }

    #[test]
    fn w28_list_accessors_tck_list6_7_8_9() {
        let l = lit_list_ints(&[1, 2, 3]);
        // TCK List6: size([1,2,3]) = 3.
        assert_eq!(ev(&call("size", vec![l.clone()])), Value::Integer(3));
        // size on a String -> Unicode-char count.
        assert_eq!(ev(&call("size", vec![lit_str("hello")])), Value::Integer(5));
        // size(null) -> null (TCK List6 Scenario [4]).
        assert_eq!(ev(&call("size", vec![lit_null()])), Value::Null);
        // head / last.
        assert_eq!(ev(&call("head", vec![l.clone()])), Value::Integer(1));
        assert_eq!(ev(&call("last", vec![l.clone()])), Value::Integer(3));
        // tail (TCK List9: tail(tail([1..5])) = [3,4,5]; here single tail).
        assert_eq!(
            ev(&call("tail", vec![lit_list_ints(&[1, 2, 3, 4, 5])])),
            vints(&[2, 3, 4, 5])
        );
        // empty-list edges: head/last -> null, tail -> [].
        assert_eq!(ev(&call("head", vec![lit_list_ints(&[])])), Value::Null);
        assert_eq!(ev(&call("last", vec![lit_list_ints(&[])])), Value::Null);
        assert_eq!(ev(&call("tail", vec![lit_list_ints(&[])])), vints(&[]));
        // length wired as list/string count (path-length deferred — no Value::Path).
        assert_eq!(ev(&call("length", vec![l])), Value::Integer(3));
        assert_eq!(
            ev(&call("length", vec![lit_str("hello")])),
            Value::Integer(5)
        );
    }

    #[test]
    fn w28_tail_nested_tck_list9() {
        // TCK List9 exact: tail(tail([1,2,3,4,5])) = [3,4,5].
        let inner = call("tail", vec![lit_list_ints(&[1, 2, 3, 4, 5])]);
        assert_eq!(ev(&call("tail", vec![inner])), vints(&[3, 4, 5]));
    }

    #[test]
    fn w28_keys_on_node_and_null() {
        use arcgraph_core::{LabelId, NodeId};

        use crate::executor::value::NodeView;
        let node = Value::Node(
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("name", Value::String("Alice".into()))
                .with_property("age", Value::Integer(38)),
        );
        // BTreeMap-sorted keys; the TCK ignores list order for keys().
        assert_eq!(
            fn_keys(&node, "keys").unwrap(),
            vlist(vec![
                Value::String("age".into()),
                Value::String("name".into())
            ])
        );
        // keys(null) -> null (TCK Map3 Scenario [3]).
        assert_eq!(fn_keys(&Value::Null, "keys").unwrap(), Value::Null);
        // keys() on a non-entity, non-map errors.
        assert!(fn_keys(&Value::Integer(1), "keys").is_err());
    }

    #[test]
    fn ga618_keys_and_properties_on_map() {
        use std::collections::BTreeMap;

        use arcgraph_core::{LabelId, NodeId};

        use crate::executor::value::NodeView;

        // Build `{b:2, a:1}` — insertion order is irrelevant (BTreeMap
        // sorts), so `keys` is `['a','b']`.
        let mut m = BTreeMap::new();
        m.insert("b".to_string(), Value::Integer(2));
        m.insert("a".to_string(), Value::Integer(1));
        let map_val = Value::Map(m);

        // #618 — keys(map) -> sorted key list.
        assert_eq!(
            fn_keys(&map_val, "keys").unwrap(),
            vlist(vec![Value::String("a".into()), Value::String("b".into())])
        );
        // keys({}) -> [].
        assert_eq!(
            fn_keys(&Value::Map(BTreeMap::new()), "keys").unwrap(),
            vlist(vec![])
        );

        // #618 — properties(map) is the IDENTITY.
        assert_eq!(
            fn_properties(&map_val, "properties").unwrap(),
            map_val.clone()
        );
        // properties(null) -> null.
        assert_eq!(
            fn_properties(&Value::Null, "properties").unwrap(),
            Value::Null
        );
        // properties(node) -> its property bag as a Map.
        let node = Value::Node(
            NodeView::new(NodeId::new(1), Some(LabelId::new(1)))
                .with_property("name", Value::String("Alice".into()))
                .with_property("age", Value::Integer(38)),
        );
        let mut expect = BTreeMap::new();
        expect.insert("name".to_string(), Value::String("Alice".into()));
        expect.insert("age".to_string(), Value::Integer(38));
        assert_eq!(
            fn_properties(&node, "properties").unwrap(),
            Value::Map(expect)
        );
        // properties() on a scalar errors (non-entity, non-map).
        assert!(fn_properties(&Value::Integer(1), "properties").is_err());
    }

    #[test]
    fn ga618_function_dispatch_is_case_insensitive() {
        // #618 — the eval dispatch lower-cases the call name, so a
        // mis-cased function call dispatches to the same impl. (Paired
        // with the case-insensitive registry `lookup` at type-check.)
        let schema = no_schema();
        let ev_fn = |name: &str, args: Vec<BoundExpression>| {
            apply_function(name, &args, &[], &*schema, &Parameters::new())
        };
        // RANGE / Range / range all build [1,2,3].
        let want = Value::List(vec![
            Value::Integer(1),
            Value::Integer(2),
            Value::Integer(3),
        ]);
        for spelling in ["range", "RANGE", "Range"] {
            assert_eq!(
                ev_fn(spelling, vec![lit_int(1), lit_int(3)]).unwrap(),
                want,
                "`{spelling}(1,3)` dispatches to range"
            );
        }
        // ABS / Abs / abs all compute |−5| = 5.
        for spelling in ["abs", "ABS", "Abs"] {
            assert_eq!(
                ev_fn(spelling, vec![lit_int(-5)]).unwrap(),
                Value::Integer(5),
                "`{spelling}(-5)` dispatches to abs"
            );
        }
    }

    #[test]
    fn w28_cypher_float_string_formatting() {
        assert_eq!(cypher_float_string(2.3), "2.3");
        assert_eq!(cypher_float_string(3.0), "3.0");
        assert_eq!(cypher_float_string(-1.5), "-1.5");
        assert_eq!(cypher_float_string(0.0), "0.0");
        assert_eq!(cypher_float_string(f64::NAN), "NaN");
        assert_eq!(cypher_float_string(f64::INFINITY), "Infinity");
        assert_eq!(cypher_float_string(f64::NEG_INFINITY), "-Infinity");
    }

    #[test]
    fn w28_arity_guards_are_defensive() {
        // e()/pi() reject extra args (the type-checker also guards this).
        assert!(ev_res(&call("e", vec![lit_int(1)])).is_err());
        // substring below its min arity errors at eval.
        assert!(ev_res(&call("substring", vec![lit_str("x")])).is_err());
    }

    // =================================================================
    // ADR-188 — list-predicate (`all`/`any`/`none`/`single`) + `reduce`
    // DIRECT-EVALUATOR unit tests. The strongest possible oracle for
    // the Decision 4 3VL truth table: each test constructs the
    // `BoundExpression` node directly and calls `evaluate()` against an
    // empty input row, so the result depends ONLY on the per-element
    // fold logic (no operator-pipeline indirection). Every
    // `single`-with-NULL row of the PE-corrected Decision 4-single is
    // pinned with a STRONG `==` oracle that BITES on the wrong
    // semantics.
    // =================================================================

    use crate::ast::{Expression, Quantifier};

    // The scoped iteration variable's binding id used by all the
    // list-predicate unit tests. With an empty input row, the
    // evaluator appends the element at slot 0 and the scoped closure
    // maps THIS id → slot 0; no outer bindings exist.
    const X_BID: BindingId = BindingId::new(0);
    // reduce reserves two: acc at slot 0, x at slot 1.
    const ACC_BID: BindingId = BindingId::new(0);
    const RX_BID: BindingId = BindingId::new(1);

    /// Build a list literal `BoundExpression` from `Option<i64>`
    /// elements (`Some(n)` → integer, `None` → NULL element). Evaluates
    /// to a `Value::List`.
    fn bound_list(elems: &[Option<i64>]) -> BoundExpression {
        let inner: Vec<Expression> = elems
            .iter()
            .map(|e| match e {
                Some(n) => Expression::Literal(Literal::Integer(*n)),
                None => Expression::Literal(Literal::Null),
            })
            .collect();
        BoundExpression::Literal {
            value: Literal::List(inner),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// `x = needle` predicate over the scoped variable `X_BID`.
    fn pred_x_eq(needle: i64) -> BoundExpression {
        BoundExpression::BinaryOp {
            op: BinOp::Eq,
            lhs: Box::new(BoundExpression::VariableRef {
                name: "x".into(),
                binding_id: X_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(lit_int(needle)),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// `x > threshold` predicate over the scoped variable `X_BID`.
    fn pred_x_gt(threshold: i64) -> BoundExpression {
        BoundExpression::BinaryOp {
            op: BinOp::Gt,
            lhs: Box::new(BoundExpression::VariableRef {
                name: "x".into(),
                binding_id: X_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(lit_int(threshold)),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn list_pred(q: Quantifier, list: BoundExpression, pred: BoundExpression) -> BoundExpression {
        BoundExpression::ListPredicate {
            quantifier: q,
            var_bid: X_BID,
            list: Box::new(list),
            predicate: Box::new(pred),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Evaluate a list-predicate against an empty input row.
    fn ev_lp(q: Quantifier, list: &[Option<i64>], pred: BoundExpression) -> Value {
        let e = list_pred(q, bound_list(list), pred);
        let s = no_schema();
        evaluate(&e, &[], &*s, &Parameters::new()).unwrap()
    }

    // ---------- all() ----------

    #[test]
    fn lp_all_all_true() {
        // all(x IN [1,2,3] WHERE x > 0) ⇒ true.
        assert_eq!(
            ev_lp(Quantifier::All, &[Some(1), Some(2), Some(3)], pred_x_gt(0)),
            Value::Boolean(true)
        );
    }

    #[test]
    fn lp_all_one_false_dominates() {
        // all(x IN [1,2,3] WHERE x > 1) ⇒ false (1 is a definite false).
        assert_eq!(
            ev_lp(Quantifier::All, &[Some(1), Some(2), Some(3)], pred_x_gt(1)),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_all_null_no_definite_false_is_null() {
        // all(x IN [2,null,3] WHERE x > 1) ⇒ null (no false, a null).
        assert_eq!(
            ev_lp(Quantifier::All, &[Some(2), None, Some(3)], pred_x_gt(1)),
            Value::Null
        );
    }

    #[test]
    fn lp_all_definite_false_dominates_null() {
        // all(x IN [0,null,3] WHERE x > 1) ⇒ false (0 is a definite
        // false; the definite witness dominates the null).
        assert_eq!(
            ev_lp(Quantifier::All, &[Some(0), None, Some(3)], pred_x_gt(1)),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_all_empty_is_true() {
        // all([] WHERE …) ⇒ true (vacuous). MUST assert true.
        assert_eq!(
            ev_lp(Quantifier::All, &[], pred_x_gt(0)),
            Value::Boolean(true)
        );
    }

    // ---------- any() ----------

    #[test]
    fn lp_any_one_true() {
        // any(x IN [1,2,3] WHERE x > 2) ⇒ true.
        assert_eq!(
            ev_lp(Quantifier::Any, &[Some(1), Some(2), Some(3)], pred_x_gt(2)),
            Value::Boolean(true)
        );
    }

    #[test]
    fn lp_any_none_true_is_false() {
        // any(x IN [1,2,3] WHERE x > 100) ⇒ false.
        assert_eq!(
            ev_lp(
                Quantifier::Any,
                &[Some(1), Some(2), Some(3)],
                pred_x_gt(100)
            ),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_any_null_no_definite_true_is_null() {
        // any(x IN [1,null,3] WHERE x > 100) ⇒ null (no true, a null).
        assert_eq!(
            ev_lp(Quantifier::Any, &[Some(1), None, Some(3)], pred_x_gt(100)),
            Value::Null
        );
    }

    #[test]
    fn lp_any_definite_true_dominates_null() {
        // any(x IN [5,null,3] WHERE x > 100)? No true here. Use x > 2:
        // any(x IN [5,null,1] WHERE x > 2) ⇒ true (5 is a definite true,
        // dominates the null).
        assert_eq!(
            ev_lp(Quantifier::Any, &[Some(5), None, Some(1)], pred_x_gt(2)),
            Value::Boolean(true)
        );
    }

    #[test]
    fn lp_any_empty_is_false() {
        // any([] WHERE …) ⇒ false (vacuous).
        assert_eq!(
            ev_lp(Quantifier::Any, &[], pred_x_gt(0)),
            Value::Boolean(false)
        );
    }

    // ---------- none() ----------

    #[test]
    fn lp_none_no_true_is_true() {
        // none(x IN [1,2,3] WHERE x > 100) ⇒ true.
        assert_eq!(
            ev_lp(
                Quantifier::None,
                &[Some(1), Some(2), Some(3)],
                pred_x_gt(100)
            ),
            Value::Boolean(true)
        );
    }

    #[test]
    fn lp_none_one_true_is_false() {
        // none(x IN [1,2,3] WHERE x > 2) ⇒ false (3 matches).
        assert_eq!(
            ev_lp(Quantifier::None, &[Some(1), Some(2), Some(3)], pred_x_gt(2)),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_none_null_no_definite_true_is_null() {
        // none(x IN [1,null,3] WHERE x > 100) ⇒ null.
        assert_eq!(
            ev_lp(Quantifier::None, &[Some(1), None, Some(3)], pred_x_gt(100)),
            Value::Null
        );
    }

    #[test]
    fn lp_none_definite_true_dominates_null() {
        // none(x IN [5,null,1] WHERE x > 2) ⇒ false (5 is a definite
        // true; the definite witness dominates the null).
        assert_eq!(
            ev_lp(Quantifier::None, &[Some(5), None, Some(1)], pred_x_gt(2)),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_none_empty_is_true() {
        // none([] WHERE …) ⇒ true (vacuous).
        assert_eq!(
            ev_lp(Quantifier::None, &[], pred_x_gt(0)),
            Value::Boolean(true)
        );
    }

    // ---------- single() — the PE-corrected NULL semantics (BINDING) --
    // Each of these is a Decision 4-single row from the ADR. They MUST
    // bite on the exact value; a naive "count trues, ignore nulls"
    // implementation would FAIL `lp_single_zero_true_with_null_is_null`.

    #[test]
    fn lp_single_exactly_one_true() {
        // single(x IN [1,2,3] WHERE x = 2) ⇒ true (exactly one).
        assert_eq!(
            ev_lp(
                Quantifier::Single,
                &[Some(1), Some(2), Some(3)],
                pred_x_eq(2)
            ),
            Value::Boolean(true)
        );
    }

    #[test]
    fn lp_single_two_true_is_false() {
        // single(x IN [1,2,2] WHERE x = 2) ⇒ false (two matches).
        assert_eq!(
            ev_lp(
                Quantifier::Single,
                &[Some(1), Some(2), Some(2)],
                pred_x_eq(2)
            ),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_single_one_definite_true_plus_null_yields_true() {
        // single(x IN [2,null,3] WHERE x = 2) ⇒ TRUE.
        // PE-CORRECTED, LOAD-BEARING: one definite witness dominates the
        // null — Cypher does NOT speculate the null is a second match
        // (cf. `2 IN [1,2,null] ⇒ true`). A naive implementation that
        // demoted a definite single witness to null on seeing a null
        // would FAIL this assertion.
        assert_eq!(
            ev_lp(Quantifier::Single, &[Some(2), None, Some(3)], pred_x_eq(2)),
            Value::Boolean(true),
            "single([2,null,3] WHERE x=2) MUST be TRUE (one definite \
             witness dominates the null; ADR-188 Decision 4-single)"
        );
    }

    #[test]
    fn lp_single_zero_true_with_null_yields_null() {
        // single(x IN [1,null,3] WHERE x = 2) ⇒ NULL.
        // LOAD-BEARING: zero definite trues + a null could be the single
        // match ⇒ genuinely unknown. A naive count-based implementation
        // would return FALSE here and PASS a weak oracle while being
        // WRONG — this assertion forecloses that honesty-gate failure.
        assert_eq!(
            ev_lp(Quantifier::Single, &[Some(1), None, Some(3)], pred_x_eq(2)),
            Value::Null,
            "single([1,null,3] WHERE x=2) MUST be NULL (zero definite \
             trues + a null could be the single match; ADR-188 \
             Decision 4-single)"
        );
    }

    #[test]
    fn lp_single_two_true_plus_null_is_false() {
        // single(x IN [2,2,null] WHERE x = 2) ⇒ false (two definite
        // trues dominate the null).
        assert_eq!(
            ev_lp(Quantifier::Single, &[Some(2), Some(2), None], pred_x_eq(2)),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_single_zero_true_no_null_is_false() {
        // single(x IN [1,3,4] WHERE x = 2) ⇒ false (zero matches, no
        // nulls, non-empty).
        assert_eq!(
            ev_lp(
                Quantifier::Single,
                &[Some(1), Some(3), Some(4)],
                pred_x_eq(2)
            ),
            Value::Boolean(false)
        );
    }

    #[test]
    fn lp_single_empty_is_false() {
        // single([] WHERE …) ⇒ false (vacuously not-exactly-one).
        assert_eq!(
            ev_lp(Quantifier::Single, &[], pred_x_eq(2)),
            Value::Boolean(false)
        );
    }

    // ---------- null list (every quantifier) ----------

    #[test]
    fn lp_null_list_is_null_all_quantifiers() {
        for q in [
            Quantifier::All,
            Quantifier::Any,
            Quantifier::None,
            Quantifier::Single,
        ] {
            let e = list_pred(q, lit_null(), pred_x_eq(2));
            let s = no_schema();
            assert_eq!(
                evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
                Value::Null,
                "null list ⇒ null for {q:?}"
            );
        }
    }

    // ---------- nested (all inside any) ----------

    #[test]
    fn lp_nested_all_inside_any() {
        // any(x IN [1,2] WHERE all(y IN [10,20] WHERE y > x))
        // For x=1: all(y>1) over [10,20] ⇒ true ⇒ the OUTER any is true.
        // Tests that the inner scoped var `y` (slot 1, over the
        // already-extended row carrying `x` at slot 0) resolves
        // correctly AND that the outer `x` is still visible inside the
        // inner predicate (y > x references BOTH scoped vars).
        let y_bid = BindingId::new(1);
        let inner_pred = BoundExpression::BinaryOp {
            op: BinOp::Gt,
            lhs: Box::new(BoundExpression::VariableRef {
                name: "y".into(),
                binding_id: y_bid,
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::VariableRef {
                name: "x".into(),
                binding_id: X_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        };
        let inner = BoundExpression::ListPredicate {
            quantifier: Quantifier::All,
            var_bid: y_bid,
            list: Box::new(bound_list(&[Some(10), Some(20)])),
            predicate: Box::new(inner_pred),
            span: Span::point(1, 1),
            type_info: None,
        };
        let outer = BoundExpression::ListPredicate {
            quantifier: Quantifier::Any,
            var_bid: X_BID,
            list: Box::new(bound_list(&[Some(1), Some(2)])),
            predicate: Box::new(inner),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        assert_eq!(
            evaluate(&outer, &[], &*s, &Parameters::new()).unwrap(),
            Value::Boolean(true),
            "any(x IN [1,2] WHERE all(y IN [10,20] WHERE y > x)) ⇒ true"
        );
    }

    #[test]
    fn lp_nested_with_outer_row_binding() {
        // With an OUTER row binding present: row = [Integer(15)] at slot
        // 0 bound to id 99 (the outer `n.threshold`). The scoped var
        // appends at slot 1 (row.len()=1). Predicate: x > n_threshold.
        // all(x IN [20,30] WHERE x > 15) ⇒ true.
        let n_bid = BindingId::new(99);
        let pred = BoundExpression::BinaryOp {
            op: BinOp::Gt,
            lhs: Box::new(BoundExpression::VariableRef {
                name: "x".into(),
                // scoped var appended at slot = row.len() = 1.
                binding_id: BindingId::new(1),
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::VariableRef {
                name: "n".into(),
                binding_id: n_bid,
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        };
        let e = BoundExpression::ListPredicate {
            quantifier: Quantifier::All,
            var_bid: BindingId::new(1),
            list: Box::new(bound_list(&[Some(20), Some(30)])),
            predicate: Box::new(pred),
            span: Span::point(1, 1),
            type_info: None,
        };
        let row = [Value::Integer(15)];
        // Outer schema: id 99 → slot 0. The scoped closure overlays
        // id 1 → slot 1 (the appended element).
        let outer_schema =
            move |b: BindingId| -> Option<usize> { if b == n_bid { Some(0) } else { None } };
        assert_eq!(
            evaluate(&e, &row, &outer_schema, &Parameters::new()).unwrap(),
            Value::Boolean(true),
            "all(x IN [20,30] WHERE x > n.threshold=15) ⇒ true"
        );
    }

    // ---------- reduce() ----------

    fn reduce_sum(init: i64, list: &[Option<i64>]) -> BoundExpression {
        // reduce(s = init, x IN list | s + x)
        let body = BoundExpression::BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(BoundExpression::VariableRef {
                name: "s".into(),
                binding_id: ACC_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::VariableRef {
                name: "x".into(),
                binding_id: RX_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        };
        BoundExpression::Reduce {
            acc_bid: ACC_BID,
            init: Box::new(lit_int(init)),
            var_bid: RX_BID,
            list: Box::new(bound_list(list)),
            expr: Box::new(body),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    #[test]
    fn reduce_accumulates_sum() {
        // reduce(s = 0, x IN [1,2,3,4] | s + x) ⇒ 10.
        let e = reduce_sum(0, &[Some(1), Some(2), Some(3), Some(4)]);
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Integer(10)
        );
    }

    #[test]
    fn reduce_nonzero_init() {
        // reduce(s = 100, x IN [1,2,3] | s + x) ⇒ 106.
        let e = reduce_sum(100, &[Some(1), Some(2), Some(3)]);
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Integer(106)
        );
    }

    #[test]
    fn reduce_empty_list_is_init() {
        // reduce(s = 42, x IN [] | s + x) ⇒ 42 (empty ⇒ init).
        let e = reduce_sum(42, &[]);
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Integer(42)
        );
    }

    #[test]
    fn reduce_null_in_list_propagates() {
        // reduce(s = 0, x IN [1,null,3] | s + x) ⇒ null.
        // 0+1=1; 1+null=null; null+3=null. The null is an ORDINARY value
        // that propagates through the fold (Decision 4 pure-fold).
        let e = reduce_sum(0, &[Some(1), None, Some(3)]);
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Null,
            "reduce(s=0, x IN [1,null,3] | s+x) ⇒ null (null propagates)"
        );
    }

    #[test]
    fn reduce_null_list_is_null() {
        // reduce(s = 0, x IN null | s + x) ⇒ null.
        let body = lit_int(0);
        let e = BoundExpression::Reduce {
            acc_bid: ACC_BID,
            init: Box::new(lit_int(0)),
            var_bid: RX_BID,
            list: Box::new(lit_null()),
            expr: Box::new(body),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Null
        );
    }

    #[test]
    fn reduce_float_body_widens() {
        // reduce(s = 0, x IN [1.5, 2.5] | s + x) ⇒ 4.0 (Float).
        // The Int accumulator widens to Float at runtime as the body
        // produces Float (Decision 3-reduce-widening / OQ-5 — the
        // runtime arithmetic already does Int+Float→Float).
        let body = BoundExpression::BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(BoundExpression::VariableRef {
                name: "s".into(),
                binding_id: ACC_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            rhs: Box::new(BoundExpression::VariableRef {
                name: "x".into(),
                binding_id: RX_BID,
                span: Span::point(1, 1),
                type_info: None,
            }),
            span: Span::point(1, 1),
            type_info: None,
        };
        let list = BoundExpression::Literal {
            value: Literal::List(vec![
                Expression::Literal(Literal::Float(1.5)),
                Expression::Literal(Literal::Float(2.5)),
            ]),
            span: Span::point(1, 1),
            type_info: None,
        };
        let e = BoundExpression::Reduce {
            acc_bid: ACC_BID,
            init: Box::new(lit_int(0)),
            var_bid: RX_BID,
            list: Box::new(list),
            expr: Box::new(body),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Float(4.0)
        );
    }

    #[test]
    fn lp_non_list_value_errors() {
        // all(x IN 5 WHERE x > 0) — a non-list, non-null operand is a
        // runtime eval error (the type-check should reject it, but the
        // evaluator is defensive).
        let e = list_pred(Quantifier::All, lit_int(5), pred_x_gt(0));
        let s = no_schema();
        assert!(matches!(
            evaluate(&e, &[], &*s, &Parameters::new()),
            Err(ExecutionError::Eval(_))
        ));
    }

    // =================================================================
    // ADR-188 (#620 list-half) — list-comprehension eval unit tests.
    //
    // openCypher v9 §3.5 `[x IN list WHERE p | e]`. Each test drives the
    // evaluator DIRECTLY against a hand-built `BoundExpression` with an
    // EMPTY input row (so the result depends ONLY on the per-element
    // filter/project logic — no operator-pipeline indirection). Strong
    // `==` oracles that BITE on the wrong semantics (3VL filter: only
    // `true` keeps; null-list ⇒ null; empty ⇒ empty; order preserved).
    // =================================================================

    /// A `VariableRef` to the scoped iteration variable `X_BID` (slot 0
    /// with an empty input row). Used as the identity projection and
    /// inside projection arithmetic.
    fn var_x() -> BoundExpression {
        BoundExpression::VariableRef {
            name: "x".into(),
            binding_id: X_BID,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// `x * k` projection over the scoped variable `X_BID`.
    fn proj_x_times(k: i64) -> BoundExpression {
        BoundExpression::BinaryOp {
            op: BinOp::Mul,
            lhs: Box::new(var_x()),
            rhs: Box::new(lit_int(k)),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Build a list-comprehension node over `X_BID` with optional
    /// predicate + projection.
    fn list_comp(
        list: BoundExpression,
        predicate: Option<BoundExpression>,
        projection: Option<BoundExpression>,
    ) -> BoundExpression {
        BoundExpression::ListComprehension {
            var_bid: X_BID,
            list: Box::new(list),
            predicate: predicate.map(Box::new),
            projection: projection.map(Box::new),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Evaluate a list-comprehension over `list` (with `None` NULL
    /// elements) against an empty input row.
    fn ev_lc(
        list: &[Option<i64>],
        predicate: Option<BoundExpression>,
        projection: Option<BoundExpression>,
    ) -> Value {
        let e = list_comp(bound_list(list), predicate, projection);
        let s = no_schema();
        evaluate(&e, &[], &*s, &Parameters::new()).unwrap()
    }

    /// Convenience: a `Value::List` of integers, for oracle comparison.
    fn vlist_ints(vs: &[i64]) -> Value {
        Value::List(vs.iter().map(|n| Value::Integer(*n)).collect())
    }

    #[test]
    fn lc_filter_then_map() {
        // [x IN [1,2,3] WHERE x > 1 | x * 10] ⇒ [20, 30]
        assert_eq!(
            ev_lc(
                &[Some(1), Some(2), Some(3)],
                Some(pred_x_gt(1)),
                Some(proj_x_times(10)),
            ),
            vlist_ints(&[20, 30]),
        );
    }

    #[test]
    fn lc_map_only_no_where() {
        // [x IN [1,2,3] | x * 10] ⇒ [10, 20, 30] (map every element)
        assert_eq!(
            ev_lc(&[Some(1), Some(2), Some(3)], None, Some(proj_x_times(10))),
            vlist_ints(&[10, 20, 30]),
        );
    }

    #[test]
    fn lc_filter_only_identity_projection() {
        // [x IN [1,2,3,4] WHERE x > 2] ⇒ [3, 4] (filter, project x itself)
        assert_eq!(
            ev_lc(
                &[Some(1), Some(2), Some(3), Some(4)],
                Some(pred_x_gt(2)),
                None
            ),
            vlist_ints(&[3, 4]),
        );
    }

    #[test]
    fn lc_identity_no_where_no_projection() {
        // [x IN [1,2,3]] ⇒ [1, 2, 3] (identity over the whole list)
        assert_eq!(
            ev_lc(&[Some(1), Some(2), Some(3)], None, None),
            vlist_ints(&[1, 2, 3]),
        );
    }

    #[test]
    fn lc_empty_list_yields_empty_list() {
        // [x IN [] WHERE x > 1 | x * 10] ⇒ [] (NOT null)
        assert_eq!(
            ev_lc(&[], Some(pred_x_gt(1)), Some(proj_x_times(10))),
            Value::List(Vec::new()),
        );
        // Empty also for the no-filter / no-projection forms.
        assert_eq!(ev_lc(&[], None, None), Value::List(Vec::new()));
    }

    #[test]
    fn lc_null_list_yields_null() {
        // [x IN null | x * 10] ⇒ null (the null-list rule, shared with
        // ListPredicate / reduce).
        let e = list_comp(lit_null(), None, Some(proj_x_times(10)));
        let s = no_schema();
        assert_eq!(
            evaluate(&e, &[], &*s, &Parameters::new()).unwrap(),
            Value::Null,
        );
    }

    #[test]
    fn lc_null_predicate_filters_element_out() {
        // [x IN [1, null, 3] WHERE x > 1 | x] — the NULL element makes
        // `null > 1 ⇒ null` (3VL Unknown), so it is FILTERED OUT (only
        // `true` keeps). `1 > 1 ⇒ false` (out). `3 > 1 ⇒ true` (in,
        // identity-projected). ⇒ [3]. This BITES on a wrong impl that
        // would keep the null element or surface a null in the result.
        assert_eq!(
            ev_lc(&[Some(1), None, Some(3)], Some(pred_x_gt(1)), None),
            vlist_ints(&[3]),
        );
    }

    #[test]
    fn lc_null_element_passes_when_no_filter() {
        // [x IN [1, null, 3] | x] — no WHERE, so EVERY element is kept,
        // including the NULL (identity projection preserves it).
        // ⇒ [1, null, 3]. The null is a VALUE in the result, not a
        // filtered-out element (contrast lc_null_predicate_filters...).
        assert_eq!(
            ev_lc(&[Some(1), None, Some(3)], None, None),
            Value::List(vec![Value::Integer(1), Value::Null, Value::Integer(3)]),
        );
    }

    #[test]
    fn lc_order_preserved_when_filtering() {
        // [x IN [5,1,4,2,3] WHERE x > 2 | x] ⇒ [5, 4, 3] — SOURCE order
        // preserved (NOT sorted); only elements ≤ 2 dropped.
        assert_eq!(
            ev_lc(
                &[Some(5), Some(1), Some(4), Some(2), Some(3)],
                Some(pred_x_gt(2)),
                None,
            ),
            vlist_ints(&[5, 4, 3]),
        );
    }

    #[test]
    fn lc_projection_can_introduce_constant() {
        // [x IN [1,2,3] | 7] ⇒ [7, 7, 7] — projection need not reference
        // x; a constant projection maps every kept element to the
        // constant (no filter ⇒ all 3 kept).
        assert_eq!(
            ev_lc(&[Some(1), Some(2), Some(3)], None, Some(lit_int(7))),
            vlist_ints(&[7, 7, 7]),
        );
    }

    #[test]
    fn lc_nested_comprehension() {
        // Nested: [x IN [1,2] | [y IN [10,20] | x + y]] using DISTINCT
        // binding ids for the inner scoped var (slot 1, one past the
        // outer x at slot 0). Outer x ∈ {1,2}; inner y ∈ {10,20}.
        //   x=1 ⇒ [11, 21]; x=2 ⇒ [12, 22]
        // ⇒ [[11,21],[12,22]]. Exercises the nested-slot mechanism:
        // the inner comprehension extends the already-extended row, so
        // y lands at slot 1 and the outer x (slot 0) stays resolvable.
        const Y_BID: BindingId = BindingId::new(1);
        let y_ref = BoundExpression::VariableRef {
            name: "y".into(),
            binding_id: Y_BID,
            span: Span::point(1, 1),
            type_info: None,
        };
        // inner projection body: x + y
        let inner_body = BoundExpression::BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(var_x()),
            rhs: Box::new(y_ref),
            span: Span::point(1, 1),
            type_info: None,
        };
        let inner = BoundExpression::ListComprehension {
            var_bid: Y_BID,
            list: Box::new(bound_list(&[Some(10), Some(20)])),
            predicate: None,
            projection: Some(Box::new(inner_body)),
            span: Span::point(1, 1),
            type_info: None,
        };
        let outer = BoundExpression::ListComprehension {
            var_bid: X_BID,
            list: Box::new(bound_list(&[Some(1), Some(2)])),
            predicate: None,
            projection: Some(Box::new(inner)),
            span: Span::point(1, 1),
            type_info: None,
        };
        let s = no_schema();
        assert_eq!(
            evaluate(&outer, &[], &*s, &Parameters::new()).unwrap(),
            Value::List(vec![vlist_ints(&[11, 21]), vlist_ints(&[12, 22])]),
        );
    }

    #[test]
    fn lc_non_list_value_errors() {
        // [x IN 5 | x] — a non-list, non-null operand is a runtime eval
        // error (the type-check should reject it; the evaluator is
        // defensive — same posture as the list-predicate path).
        let e = list_comp(lit_int(5), None, None);
        let s = no_schema();
        assert!(matches!(
            evaluate(&e, &[], &*s, &Parameters::new()),
            Err(ExecutionError::Eval(_))
        ));
    }

    // =================================================================
    // ADR-191 — Value::Map (#620 foundation). Direct-evaluator strong
    // oracles for the value-level semantics (D-2 literal eval, D-3 3VL
    // equality, D-4 comparability, D-8 property access). Pipeline-level
    // proofs (GROUP BY / DISTINCT / UNION / ORDER BY / write-fence) live
    // in `tests/value_map_e2e.rs`.
    // =================================================================

    fn xint(n: i64) -> Expression {
        Expression::Literal(Literal::Integer(n))
    }
    fn xnull() -> Expression {
        Expression::Literal(Literal::Null)
    }
    fn xmap(entries: &[(&str, Expression)]) -> Expression {
        Expression::Literal(Literal::Map(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        ))
    }
    /// A `Literal::Map` BoundExpression for the evaluator.
    fn bmap(entries: &[(&str, Expression)]) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Map(
                entries
                    .iter()
                    .map(|(k, v)| (k.to_string(), v.clone()))
                    .collect(),
            ),
            span: Span::point(1, 1),
            type_info: None,
        }
    }
    fn eval_ok(e: &BoundExpression) -> Value {
        let s = no_schema();
        evaluate(e, &[], &*s, &Parameters::new()).expect("eval")
    }
    fn mk_binop(op: BinOp, l: BoundExpression, r: BoundExpression) -> BoundExpression {
        BoundExpression::BinaryOp {
            op,
            lhs: Box::new(l),
            rhs: Box::new(r),
            span: Span::point(1, 1),
            type_info: None,
        }
    }
    /// Build `{...}` directly as a `Value::Map`.
    fn vmap(entries: &[(&str, Value)]) -> Value {
        Value::Map(
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        )
    }

    // ---- D-2: literal evaluation (THE bug fix) ----

    #[test]
    fn map_literal_evaluates_to_map_not_null() {
        // THE BUG FIX: `{a:1, b:2}` evaluated to `null` on main; now it
        // is a real map. (A test that FAILS on main per D-14 item 1.)
        assert_eq!(
            eval_ok(&bmap(&[("a", xint(1)), ("b", xint(2))])),
            vmap(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]),
        );
    }

    #[test]
    fn map_literal_empty() {
        assert_eq!(eval_ok(&bmap(&[])), Value::Map(BTreeMap::new()));
    }

    #[test]
    fn map_literal_nested_and_composite() {
        // Nested map, list-in-map, map-in-list (D-14 item 2).
        assert_eq!(
            eval_ok(&bmap(&[("x", xmap(&[("y", xint(5))]))])),
            vmap(&[("x", vmap(&[("y", Value::Integer(5))]))]),
        );
        assert_eq!(
            eval_ok(&bmap(&[(
                "xs",
                Expression::Literal(Literal::List(vec![xint(1), xint(2)])),
            )])),
            vmap(&[(
                "xs",
                Value::List(vec![Value::Integer(1), Value::Integer(2)])
            )]),
        );
        // map-in-list
        let list_of_map = BoundExpression::Literal {
            value: Literal::List(vec![xmap(&[("a", xint(1))])]),
            span: Span::point(1, 1),
            type_info: None,
        };
        assert_eq!(
            eval_ok(&list_of_map),
            Value::List(vec![vmap(&[("a", Value::Integer(1))])]),
        );
    }

    #[test]
    fn map_literal_duplicate_key_last_writer_wins() {
        // D-2 — `{a:1, a:2}` ⇒ `{a:2}` (BTreeMap::insert overwrites).
        assert_eq!(
            eval_ok(&bmap(&[("a", xint(1)), ("a", xint(2))])),
            vmap(&[("a", Value::Integer(2))]),
        );
    }

    #[test]
    fn map_literal_explicit_null_value_retained() {
        // D-2 / D-6 — an explicit `null` value KEEPS the key (distinct
        // from projection's null-drop, which is PR-B).
        assert_eq!(
            eval_ok(&bmap(&[("a", xnull())])),
            vmap(&[("a", Value::Null)]),
        );
    }

    // ---- D-3: equality (the `=` / `<>` operators, 3VL) ----

    #[test]
    fn map_equality_order_independent_true() {
        // `{a:1,b:2} = {b:2,a:1}` ⇒ true (BTreeMap normalizes key order).
        let e = mk_binop(
            BinOp::Eq,
            bmap(&[("a", xint(1)), ("b", xint(2))]),
            bmap(&[("b", xint(2)), ("a", xint(1))]),
        );
        assert_eq!(eval_ok(&e), Value::Boolean(true));
    }

    #[test]
    fn map_equality_different_key_set_is_false() {
        // Different key SET ⇒ definitely false (NOT unknown).
        for (l, r) in [
            (
                bmap(&[("a", xint(1))]),
                bmap(&[("a", xint(1)), ("b", xint(2))]),
            ),
            (bmap(&[("a", xint(1))]), bmap(&[("b", xint(1))])),
            (bmap(&[("a", xnull())]), bmap(&[])), // {a:null} ≠ {} (D-3)
        ] {
            let e = mk_binop(BinOp::Eq, l, r);
            assert_eq!(eval_ok(&e), Value::Boolean(false));
        }
    }

    #[test]
    fn map_equality_null_value_is_unknown_not_true() {
        // THE subtle D-3 case (PE-verified against CIP2016-06-14):
        // `{a:null} = {a:null}` ⇒ null (Unknown), NOT true.
        let e = mk_binop(BinOp::Eq, bmap(&[("a", xnull())]), bmap(&[("a", xnull())]));
        assert_eq!(eval_ok(&e), Value::Null);
        // `<>` of the same is ALSO null (Unknown propagates through NOT).
        let ne = mk_binop(BinOp::Neq, bmap(&[("a", xnull())]), bmap(&[("a", xnull())]));
        assert_eq!(eval_ok(&ne), Value::Null);
        // But a definite-false pair short-circuits PAST the unknown:
        // `{a:1,b:null} = {a:2,b:null}` ⇒ false (a:1≠a:2 is definite).
        let def = mk_binop(
            BinOp::Eq,
            bmap(&[("a", xint(1)), ("b", xnull())]),
            bmap(&[("a", xint(2)), ("b", xnull())]),
        );
        assert_eq!(eval_ok(&def), Value::Boolean(false));
    }

    #[test]
    fn map_equality_definite_values_and_nesting() {
        // All-definite values ⇒ definite verdict; nesting recurses.
        let eq = mk_binop(
            BinOp::Eq,
            bmap(&[("a", xmap(&[("b", xint(1))]))]),
            bmap(&[("a", xmap(&[("b", xint(1))]))]),
        );
        assert_eq!(eval_ok(&eq), Value::Boolean(true));
        let neq = mk_binop(
            BinOp::Eq,
            bmap(&[("a", xmap(&[("b", xint(1))]))]),
            bmap(&[("a", xmap(&[("b", xint(2))]))]),
        );
        assert_eq!(eval_ok(&neq), Value::Boolean(false));
        // `{a:1} <> {a:2}` ⇒ true.
        let ne = mk_binop(BinOp::Neq, bmap(&[("a", xint(1))]), bmap(&[("a", xint(2))]));
        assert_eq!(eval_ok(&ne), Value::Boolean(true));
    }

    #[test]
    fn map_vs_scalar_equality_is_false() {
        // A map and a scalar are definitely unequal.
        let e = mk_binop(BinOp::Eq, bmap(&[("a", xint(1))]), lit_int(1));
        assert_eq!(eval_ok(&e), Value::Boolean(false));
        let ne = mk_binop(BinOp::Neq, bmap(&[("a", xint(1))]), lit_int(1));
        assert_eq!(eval_ok(&ne), Value::Boolean(true));
    }

    // ---- D-4: comparability (`<`,`>`,`<=`,`>=` → null) ----

    #[test]
    fn map_comparability_is_null_not_error() {
        // `{a:1} < {a:2}` (and <=, >, >=) ⇒ null (Unknown), NOT an error,
        // NOT a boolean (D-4).
        for op in [BinOp::Lt, BinOp::Le, BinOp::Gt, BinOp::Ge] {
            let e = mk_binop(op.clone(), bmap(&[("a", xint(1))]), bmap(&[("a", xint(2))]));
            assert_eq!(eval_ok(&e), Value::Null, "map {op:?} should be null");
        }
        // map vs scalar operand also → null (a map operand on either side).
        let mixed = mk_binop(BinOp::Lt, bmap(&[("a", xint(1))]), lit_int(5));
        assert_eq!(eval_ok(&mixed), Value::Null);
    }

    #[test]
    fn scalar_incomparable_ordering_is_null_not_error() {
        // #1016 / Comparison2 [3]: incompatible non-null ordering is
        // Unknown (`null`), not an evaluator "incomparable types" error.
        let e = mk_binop(
            BinOp::Lt,
            lit_int(1),
            BoundExpression::Literal {
                value: Literal::String("a".into()),
                span: Span::point(1, 1),
                type_info: None,
            },
        );
        assert_eq!(eval_ok(&e), Value::Null);
    }

    // ---- D-8: property access `map.key` ----

    fn prop_access(base: BoundExpression, segs: &[&str]) -> BoundExpression {
        BoundExpression::PropertyAccess {
            base: Box::new(base),
            path: segs
                .iter()
                .map(|s| crate::semantic::bound_ast::BoundPropertyRef {
                    name: (*s).to_string(),
                    property_id: None,
                    span: Span::point(1, 1),
                })
                .collect(),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    #[test]
    fn map_property_access_present_and_missing() {
        // `{a:1}.a` ⇒ 1; `{a:1}.b` ⇒ null (missing key → null, D-8).
        assert_eq!(
            eval_ok(&prop_access(bmap(&[("a", xint(1))]), &["a"])),
            Value::Integer(1),
        );
        assert_eq!(
            eval_ok(&prop_access(bmap(&[("a", xint(1))]), &["b"])),
            Value::Null,
        );
    }

    #[test]
    fn map_property_access_nested() {
        // `{x:{y:5}}.x.y` ⇒ 5 (multi-segment walk, D-8).
        assert_eq!(
            eval_ok(&prop_access(
                bmap(&[("x", xmap(&[("y", xint(5))]))]),
                &["x", "y"]
            )),
            Value::Integer(5),
        );
    }

    // ---- direct 3VL-helper oracle ----

    #[test]
    fn map_equality_3vl_helper_table() {
        use std::collections::BTreeMap as BM;
        let m = |entries: &[(&str, Value)]| -> BM<String, Value> {
            entries
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect()
        };
        // Definite true / false / unknown.
        assert_eq!(
            map_equality_3vl(
                &m(&[("a", Value::Integer(1))]),
                &m(&[("a", Value::Integer(1))])
            ),
            Some(true)
        );
        assert_eq!(
            map_equality_3vl(
                &m(&[("a", Value::Integer(1))]),
                &m(&[("a", Value::Integer(2))])
            ),
            Some(false)
        );
        assert_eq!(
            map_equality_3vl(&m(&[("a", Value::Null)]), &m(&[("a", Value::Null)])),
            None
        );
        // Different key set ⇒ definite false (not None).
        assert_eq!(
            map_equality_3vl(&m(&[("a", Value::Null)]), &m(&[])),
            Some(false)
        );
    }

    // -----------------------------------------------------------------
    // ADR-193 — path equality (D-10, test 7) + path-fn eval (D-7, test 11)
    // direct unit tests (paths are not literal-constructible, so we build
    // `Value::Path` directly and call the private helpers).
    // -----------------------------------------------------------------

    fn tpath(start: u64, segs: &[(u64, u64, u64)]) -> Value {
        // segs: (rel_id, rel_from, rel_to) — `end` = rel_to (by id).
        use crate::executor::value::{NodeView, PathView, RelView};
        use arcgraph_core::{LabelId, NodeId, RelId, TypeId};
        let mut p = PathView::new(NodeView::new(NodeId::new(start), Some(LabelId::new(1))));
        for &(rid, from, to) in segs {
            p = p.with_segment(
                RelView::new(
                    RelId::new(rid),
                    NodeId::new(from),
                    NodeId::new(to),
                    Some(TypeId::new(1)),
                ),
                NodeView::new(NodeId::new(to), None),
            );
        }
        Value::Path(p)
    }

    #[test]
    fn t7_path_equality_by_identity() {
        // Paths contain no NULLs, so equality is TWO-VALUED — the #734 3VL
        // `values_equal_3vl` returns `Some(true)`/`Some(false)`, never
        // `None`, for path operands.
        // Two structurally-identical paths (same node-seq + rel-seq IDs)
        // are equal.
        let a = tpath(1, &[(10, 1, 2)]);
        let b = tpath(1, &[(10, 1, 2)]);
        assert_eq!(
            values_equal_3vl(&a, &b),
            Some(true),
            "identical paths are equal (D-10)"
        );

        // Differing rel id ⇒ unequal.
        let c = tpath(1, &[(99, 1, 2)]);
        assert_eq!(values_equal_3vl(&a, &c), Some(false), "differing rel id");

        // Differing end-node id ⇒ unequal.
        let d = tpath(1, &[(10, 1, 3)]);
        assert_eq!(values_equal_3vl(&a, &d), Some(false), "differing end node");

        // Differing length ⇒ unequal.
        let e = tpath(1, &[(10, 1, 2), (11, 2, 3)]);
        assert_eq!(values_equal_3vl(&a, &e), Some(false), "differing length");

        // path = non-path ⇒ Some(false) (NOT null/error) per openCypher §3.2.
        assert_eq!(
            values_equal_3vl(&a, &Value::List(vec![])),
            Some(false),
            "path = list ⇒ false (type-mismatch equality, not error)"
        );
        assert_eq!(values_equal_3vl(&a, &Value::Integer(1)), Some(false));
    }

    #[test]
    fn t11_fn_nodes_relationships_length_eval_arms() {
        let p = tpath(1, &[(10, 1, 2), (11, 2, 3)]);

        // length(path) = hop count (D-7).
        assert_eq!(fn_length(&p, "length").unwrap(), Value::Integer(2));
        // length(list) regression.
        assert_eq!(
            fn_length(&Value::List(vec![Value::Null; 3]), "length").unwrap(),
            Value::Integer(3)
        );

        // nodes(path) → List(Node) in traversal order.
        match fn_nodes(&p, "nodes").unwrap() {
            Value::List(xs) => {
                let ids: Vec<u64> = xs
                    .iter()
                    .map(|v| match v {
                        Value::Node(n) => n.id.raw(),
                        other => panic!("expected Node, got {other:?}"),
                    })
                    .collect();
                assert_eq!(ids, vec![1, 2, 3]);
            }
            other => panic!("expected List, got {other:?}"),
        }

        // relationships(path) → List(Relationship) in traversal order.
        match fn_relationships(&p, "relationships").unwrap() {
            Value::List(xs) => {
                let ids: Vec<u64> = xs
                    .iter()
                    .map(|v| match v {
                        Value::Relationship(r) => r.id.raw(),
                        other => panic!("expected Relationship, got {other:?}"),
                    })
                    .collect();
                assert_eq!(ids, vec![10, 11]);
            }
            other => panic!("expected List, got {other:?}"),
        }

        // NULL → NULL (3VL).
        assert_eq!(fn_nodes(&Value::Null, "nodes").unwrap(), Value::Null);
        assert_eq!(
            fn_relationships(&Value::Null, "relationships").unwrap(),
            Value::Null
        );

        // Non-path → InvalidArgumentType analog (ExecutionError::Eval).
        assert!(matches!(
            fn_nodes(&Value::Integer(42), "nodes"),
            Err(ExecutionError::Eval(_))
        ));
        assert!(matches!(
            fn_relationships(&Value::String("x".into()), "relationships"),
            Err(ExecutionError::Eval(_))
        ));
    }

    // =================================================================
    // ADR-191 D-6 (#620 map-half) — `eval_map_projection` strong-oracle.
    // The D-6 null-handling split is the load-bearing semantic: a `.key`
    // selector DROPS null/absent; an `alias: expr` entry KEEPS even null.
    // =================================================================

    /// The base binding for the projected variable in these tests.
    const BASE_BID: BindingId = BindingId::new(0);

    /// A `VariableRef` to the projected base at slot 0.
    fn base_ref() -> BoundExpression {
        BoundExpression::VariableRef {
            name: "n".into(),
            binding_id: BASE_BID,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    /// Schema closure mapping the base binding to row slot 0.
    fn base_schema() -> Box<dyn Fn(BindingId) -> Option<usize>> {
        Box::new(|b: BindingId| if b == BASE_BID { Some(0) } else { None })
    }

    /// Build a node `Value` with the given string-keyed properties.
    fn node_with(props: &[(&str, Value)]) -> Value {
        use crate::executor::value::NodeView;
        use arcgraph_core::{LabelId, NodeId};
        let mut n = NodeView::new(NodeId::new(7), Some(LabelId::new(1)));
        for (k, v) in props {
            n.properties.insert((*k).to_string(), v.clone());
        }
        Value::Node(n)
    }

    /// `MapProjection { base = n, items }`.
    fn proj(items: Vec<BoundMapProjectionItem>) -> BoundExpression {
        BoundExpression::MapProjection {
            base: Box::new(base_ref()),
            items,
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn lit_entry(alias: &str, value: BoundExpression) -> BoundMapProjectionItem {
        BoundMapProjectionItem::Literal {
            alias: alias.into(),
            value: Box::new(value),
        }
    }

    /// Evaluate a projection over a single base value at row slot 0.
    fn ev_proj(base: Value, items: Vec<BoundMapProjectionItem>) -> Value {
        let row = vec![base];
        let s = base_schema();
        evaluate(&proj(items), &row, &*s, &Parameters::new()).expect("eval map projection")
    }

    #[test]
    fn mp_property_selectors_over_node() {
        // n{.name, .age} over a node with both → both keys included.
        let base = node_with(&[
            ("name", Value::String("ada".into())),
            ("age", Value::Integer(36)),
            ("ignored", Value::Integer(99)),
        ]);
        assert_eq!(
            ev_proj(
                base,
                vec![
                    BoundMapProjectionItem::Property("name".into()),
                    BoundMapProjectionItem::Property("age".into()),
                ]
            ),
            vmap(&[
                ("name", Value::String("ada".into())),
                ("age", Value::Integer(36)),
            ]),
        );
    }

    #[test]
    fn mp_missing_property_is_dropped() {
        // D-6 — `.missing` (absent key) DROPS the key (`n{.missing}` → `{}`).
        let base = node_with(&[("name", Value::String("ada".into()))]);
        assert_eq!(
            ev_proj(
                base,
                vec![BoundMapProjectionItem::Property("missing".into())]
            ),
            Value::Map(BTreeMap::new()),
        );
    }

    #[test]
    fn mp_null_valued_property_is_dropped() {
        // D-6 — a `.key` whose stored value is NULL DROPS the key too (the
        // selector-form null-drop applies to present-but-null, not just
        // absent).
        let base = node_with(&[("name", Value::String("ada".into())), ("nick", Value::Null)]);
        assert_eq!(
            ev_proj(
                base,
                vec![
                    BoundMapProjectionItem::Property("name".into()),
                    BoundMapProjectionItem::Property("nick".into()),
                ]
            ),
            vmap(&[("name", Value::String("ada".into()))]),
            "a null-valued .key must be dropped (D-6)"
        );
    }

    #[test]
    fn mp_literal_entry_keeps_explicit_null() {
        // D-6 — `n{x: null, y: 1}` KEEPS the explicit-null key
        // (`{x: null, y: 1}`). This is the load-bearing contrast with the
        // `.key` null-DROP above.
        let base = node_with(&[("name", Value::String("ada".into()))]);
        assert_eq!(
            ev_proj(
                base,
                vec![lit_entry("x", lit_null()), lit_entry("y", lit_int(1)),]
            ),
            vmap(&[("x", Value::Null), ("y", Value::Integer(1))]),
            "an explicit alias: null must be KEPT (D-6)"
        );
    }

    #[test]
    fn mp_literal_entry_evaluates_expression() {
        // `n{alias: 1 + 1}` evaluates the value expression → `{alias: 2}`.
        let base = node_with(&[]);
        let one_plus_one = BoundExpression::BinaryOp {
            op: BinOp::Add,
            lhs: Box::new(lit_int(1)),
            rhs: Box::new(lit_int(1)),
            span: Span::point(1, 1),
            type_info: None,
        };
        assert_eq!(
            ev_proj(base, vec![lit_entry("alias", one_plus_one)]),
            vmap(&[("alias", Value::Integer(2))]),
        );
    }

    #[test]
    fn mp_all_properties_selector() {
        // n{.*} copies EVERY property of the base.
        let base = node_with(&[("a", Value::Integer(1)), ("b", Value::String("two".into()))]);
        assert_eq!(
            ev_proj(base, vec![BoundMapProjectionItem::AllProperties]),
            vmap(&[("a", Value::Integer(1)), ("b", Value::String("two".into()))]),
        );
    }

    #[test]
    fn mp_all_properties_then_override_is_last_writer_wins() {
        // n{.*, a: 99} — `.*` copies, then the explicit `a: 99` overrides.
        let base = node_with(&[("a", Value::Integer(1)), ("b", Value::Integer(2))]);
        assert_eq!(
            ev_proj(
                base,
                vec![
                    BoundMapProjectionItem::AllProperties,
                    lit_entry("a", lit_int(99)),
                ]
            ),
            vmap(&[("a", Value::Integer(99)), ("b", Value::Integer(2))]),
            "explicit entry after .* overrides (last-writer-wins)"
        );
    }

    #[test]
    fn mp_over_map_base() {
        // A projection over a `Value::Map` base resolves the same way as a
        // node base (the bag is the map itself).
        let base = vmap(&[("a", Value::Integer(1)), ("b", Value::Null)]);
        assert_eq!(
            ev_proj(
                base,
                vec![
                    BoundMapProjectionItem::Property("a".into()),
                    BoundMapProjectionItem::Property("b".into()), // null → dropped
                ]
            ),
            vmap(&[("a", Value::Integer(1))]),
        );
    }

    #[test]
    fn mp_null_base_is_null() {
        // A null base ⇒ null projection (openCypher null-propagation).
        assert_eq!(
            ev_proj(
                Value::Null,
                vec![BoundMapProjectionItem::Property("a".into())]
            ),
            Value::Null,
        );
    }

    #[test]
    fn mp_empty_projection_is_empty_map() {
        // n{} ⇒ the empty map.
        let base = node_with(&[("a", Value::Integer(1))]);
        assert_eq!(ev_proj(base, vec![]), Value::Map(BTreeMap::new()));
    }

    #[test]
    fn mp_non_entity_base_errors() {
        // A projection over a scalar base (only reachable past the
        // type-check via a hand-built bound tree) is a runtime Eval error.
        let row = vec![Value::Integer(5)];
        let s = base_schema();
        let e = proj(vec![BoundMapProjectionItem::Property("a".into())]);
        assert!(matches!(
            evaluate(&e, &row, &*s, &Parameters::new()),
            Err(ExecutionError::Eval(_))
        ));
    }

    // =================================================================
    // ADR-147-amendment-03 §B1 — the per-op concat OOM backstop, tested
    // at the eval kernel with a SMALL explicit cap so the amplification
    // is fast + deterministic (no actual OOM) and genuinely RED-on-revert
    // (deleting the `check_concat_len` calls in `checked_add_or_concat`
    // makes these build the full result instead of erroring).
    // =================================================================

    /// A list of `n` `Integer(0)` cells (local to the B1 concat tests;
    /// the module already has a `vlist(Vec<Value>)` / `vlist_ints(&[i64])`
    /// with different signatures).
    fn vzeros(n: usize) -> Value {
        Value::List(vec![Value::Integer(0); n])
    }

    /// A single over-cap list concat clean-errors BEFORE the allocation.
    #[test]
    fn checked_concat_list_over_cap_errors() {
        // cap=4; [0,0,0] ++ [0,0,0] would be 6 > 4 → clean typed error.
        let e = checked_add_or_concat(vzeros(3), vzeros(3), 4, 64)
            .expect_err("6 elements over a cap-4 list concat must error");
        match e {
            ExecutionError::Eval(m) => assert!(
                m.contains("exceeding cap 4") && m.contains("elements"),
                "error names the element cap; got {m}"
            ),
            other => panic!("expected Eval error, got {other:?}"),
        }
    }

    /// A list concat AT the cap succeeds (boundary — cap is inclusive of
    /// `== cap`, rejects only `> cap`).
    #[test]
    fn checked_concat_list_at_cap_ok() {
        let v = checked_add_or_concat(vzeros(2), vzeros(2), 4, 64).expect("4 == cap-4 is allowed");
        assert_eq!(
            v,
            vzeros(4),
            "at-cap concat produces the full 4-element list"
        );
    }

    /// A single over-cap string concat clean-errors BEFORE the push_str.
    #[test]
    fn checked_concat_string_over_cap_errors() {
        let e = checked_add_or_concat(
            Value::String("abcde".into()),
            Value::String("fghij".into()),
            64,
            8,
        )
        .expect_err("10 bytes over a cap-8 string concat must error");
        match e {
            ExecutionError::Eval(m) => assert!(
                m.contains("exceeding cap 8") && m.contains("bytes"),
                "error names the byte cap; got {m}"
            ),
            other => panic!("expected Eval error, got {other:?}"),
        }
    }

    /// The amplification kill: a nested doubling tree folded through
    /// `checked_add_or_concat` at a small cap dies at the FIRST over-cap
    /// node — it never allocates the exponential blowup. Fold the tree
    /// bottom-up exactly as `evaluate` would, threading the error, and
    /// assert it clean-errors WELL BEFORE the full depth (proving the
    /// intermediate is never materialized). WITHOUT the per-op cap this
    /// loop would build a 2^depth list and hang/OOM.
    #[test]
    fn checked_concat_doubling_tree_dies_at_first_over_cap_node() {
        const CAP: usize = 1024;
        // Base leaf = a 1-element list. Level k folds `acc + acc`, so the
        // element count doubles each level: 1,2,4,…,2^k. At CAP=1024 the
        // first over-cap fold is level 11 (512+512=1024 is AT cap; the
        // NEXT fold 1024+1024=2048 > 1024 errors) — long before a depth
        // that would OOM at the production cap.
        let mut acc = Ok(vzeros(1));
        let mut folds = 0usize;
        loop {
            folds += 1;
            let cur = match acc {
                Ok(v) => v,
                Err(e) => {
                    // Errored at a moderate fold count → the blowup was
                    // killed early (RED-on-revert: no cap ⇒ this never
                    // errors and instead builds 2^folds elements).
                    assert!(
                        folds <= 20,
                        "the doubling tree must die at a SHALLOW node, not deep; folds={folds}"
                    );
                    match e {
                        ExecutionError::Eval(m) => {
                            assert!(m.contains("exceeding cap"), "got {m}");
                        }
                        other => panic!("expected Eval error, got {other:?}"),
                    }
                    return;
                }
            };
            // Fold `cur + cur` — the doubling amplifier.
            acc = checked_add_or_concat(cur.clone(), cur, CAP, CAP);
            assert!(
                folds < 40,
                "must have errored before 40 folds (blowup uncontained)"
            );
        }
    }

    /// A flat chain `[0] + [0] + … + [0]` folded left-associatively also
    /// dies once the running accumulator crosses the cap — the flat-chain
    /// analog of the amplification (each `+` appends one element).
    #[test]
    fn checked_concat_flat_chain_over_cap_errors() {
        const CAP: usize = 8;
        let mut acc = vzeros(1);
        for i in 0..100 {
            match checked_add_or_concat(acc.clone(), vzeros(1), CAP, CAP) {
                Ok(v) => acc = v,
                Err(ExecutionError::Eval(m)) => {
                    assert!(m.contains("exceeding cap 8"), "got {m}");
                    // 1 base + i appends; crosses cap-8 at the 8th append.
                    assert!(
                        i < 20,
                        "flat chain must error near the cap, not far past it"
                    );
                    return;
                }
                Err(other) => panic!("expected Eval error, got {other:?}"),
            }
        }
        panic!("flat chain never hit the cap");
    }
}
