//! [`MergeOp`] — write-op operator for `MERGE <pattern> [ON CREATE
//! SET …]* [ON MATCH SET …]*` per ADR-151 (W26-θ Phase 5).
//!
//! Lowers from `crate::logical_plan::LogicalMerge`. The operator
//! holds a match-branch sub-pipeline (a Scan / Filter chain for the
//! merge pattern's probe) + a create-branch sub-pipeline (a CreateNode
//! / CreateRel chain for the pattern's atomic write) + on_create /
//! on_match action item vecs.
//!
//! # Execution semantics
//!
//! On `next_batch` (first call):
//!
//! 1. Pull `match_op` to exhaustion → collect matched rows.
//! 2. **If matched rows non-empty:** for each row, fire `on_match`
//!    actions per item (substrate.set_node / set_rel per item kind),
//!    mirroring each mutation onto the row's in-memory entity view.
//! 3. **If matched rows empty:** pull `create_op` (one row out per
//!    Phase 1-2 CREATE semantics); for each row, fire `on_create`
//!    actions per item (with the same mirror).
//! 4. **If `MergeOp::output_binding` is `Some`** (node-shape named
//!    merge): emit the matched/created binding row(s). Otherwise emit
//!    an empty batch (terminal — path-shape + anonymous merges).
//! 5. Transition to EOS; subsequent calls return an empty batch.
//!
//! # Schema (ADR-151-amendment-01 §D-1)
//!
//! The output schema is driven by `MergeOp::output_binding`:
//!
//! - **`Some(binding)` — node-shape NAMED merge** (`MERGE (n:Label
//!   {…}) RETURN n`): schema is `[binding]`; each emitted row is
//!   `[Value::Node(NodeView)]`. RETURN-after-MERGE (node-shape) is
//!   lifted to v1.0-α per ADR-151-amendment-01, which reconciles the
//!   shipped code with ADR-151 §D-7's OWN pseudocode (it always
//!   specified row emission) and lifts the §D-9 "RETURN-after-MERGE"
//!   forward-pin row. The single-statement projection reads this
//!   in-memory binding row — it does NOT re-read the substrate — so it
//!   needs no statement-scoped batch tx (that remains pinned only for
//!   *multi-statement* read-your-writes; see §D-9).
//! - **`None` — path-shape OR anonymous merge**: schema is empty and
//!   the op stays **terminal** (emits an empty batch). Path-shape
//!   match `[source, rel, target]` vs create `[rel]` schemas are
//!   un-unionable (design review RC-3); anonymous merges have no
//!   binding to project.
//!
//! The emission discriminator is an EXPLICIT binding
//! (`crate::logical_plan::LogicalMerge::output_binding`), NOT a
//! `match_op ∪ create_op` schema union (the union is wrong for the
//! path / anonymous shapes).
//!
//! # Action-firing schema constraints (Phase 5 narrowing)
//!
//! - Match-branch (Node-shape): row carries `[n_binding]`; actions
//!   referencing `n_binding` work.
//! - Match-branch (Path-shape): row carries `[source, ?rel, target]`;
//!   actions referencing any of these bindings work.
//! - Create-branch (Node-shape; via CreateNodeOp): row carries
//!   `[n_binding]`; actions referencing `n_binding` work.
//! - Create-branch (Path-shape; via CreateRelOp): row carries
//!   `[rel_binding]` ONLY (the CreateRelOp emits only the rel-row);
//!   actions referencing `source` / `target` bindings surface a
//!   clean `ExecutionError::Eval` ("binding not in upstream
//!   schema"). This is the Phase 5 narrowing for Path-shape MERGE
//!   action clauses; the v1.1 MATCH→MERGE composition row in
//!   ADR-151 §"Forward-deferred" closes the gap.
//!
//! # ADR provenance
//! - **ADR-151** — primary spec (W26-θ Phase 5).
//! - **ADR-147** §D-7 — production-substrate convention (per-tenant
//!   Transaction; default trait impl returns `IndexUnavailable`).
//! - **ADR-148** §D-7 — CreateRel substrate convention.
//! - **ADR-150** §D-7 — SET-side action firing convention (reused
//!   via [`super::set`] helpers `build_node_mutation` /
//!   `build_rel_mutation` / `node_id_from_value` /
//!   `rel_id_from_value`).
//! - **ADR-031** + **ADR-033** — per-tenant `Transaction` discipline
//!   (commit + rollback).
//! - **ADR-018** — MVCC version-chain semantics for the match-branch
//!   probe (consistent snapshot over read + create + set).

use crate::executor::batch::Batch;
use crate::executor::context::ExecutionContext;
use crate::executor::error::ExecutionError;
use crate::executor::eval::evaluate;
use crate::executor::ops::set::{
    apply_node_mutation_to_view, apply_rel_mutation_to_view, build_node_mutation,
    build_rel_mutation, node_id_from_value, rel_id_from_value,
};
use crate::executor::ops::{PhysicalOperator, canonical_row_key, schema_index};
use crate::executor::substrate::ExecutorSubstrate;
use crate::executor::value::Value;
use crate::logical_plan::{LogicalSetMutation, MergeKeySpec, SetTargetKind};
use crate::semantic::bound_ast::BindingId;

/// One bound MERGE-side action item — the binding identifier + the
/// Node-vs-Rel substrate-dispatch discriminator + the materialized
/// mutation, captured at pipeline-build time. SAME SHAPE as
/// [`super::set::SetItemSpec`] (the Phase 4 action item shape).
#[derive(Debug, Clone)]
pub struct MergeActionSpec {
    pub binding: BindingId,
    pub kind: SetTargetKind,
    pub mutation: LogicalSetMutation,
}

/// MERGE executor op (ADR-151 W26-θ Phase 5).
#[derive(Debug)]
pub struct MergeOp {
    /// Sub-pipeline that probes the merge pattern in the current
    /// snapshot.
    match_op: Box<PhysicalOperator>,
    /// Sub-pipeline that creates the merge pattern when match_op
    /// returns no rows.
    create_op: Box<PhysicalOperator>,
    /// Actions to fire when the create branch is taken.
    on_create: Vec<MergeActionSpec>,
    /// Actions to fire when the match branch is taken.
    on_match: Vec<MergeActionSpec>,
    /// **ADR-151-amendment-01 §D-1** — RETURN-after-MERGE emission
    /// discriminator. `Some(binding)` ⇒ node-shape named merge: emit
    /// the matched/created binding row(s). `None` ⇒ path-shape /
    /// anonymous merge: stay terminal (emit empty).
    output_binding: Option<BindingId>,
    /// Cached output schema — `[output_binding]` for a node-shape named
    /// merge, empty otherwise (ADR-151-amendment-01 §D-1).
    schema: Vec<BindingId>,
    /// EOS flag — set after the match-or-create decision + action
    /// firing has run.
    eos: bool,
}

impl MergeOp {
    /// Construct a fresh [`MergeOp`] from a
    /// `crate::logical_plan::LogicalMerge`.
    ///
    /// `output_binding` is the
    /// `crate::logical_plan::LogicalMerge::output_binding`
    /// discriminator (ADR-151-amendment-01 §D-1): `Some(b)` for a
    /// node-shape NAMED merge (schema `[b]`, rows emitted); `None` for
    /// path-shape / anonymous merges (terminal).
    ///
    /// NN-4 (#1384) re-spin: the MERGE serialization guard(s) are acquired
    /// by the QUERY DRIVER (before the statement's snapshot pin), NOT by
    /// this op — see `crate::executor::ops::acquire_merge_guards`. The op
    /// therefore carries no key spec; it relies on the driver-held guard to
    /// have serialized its match→create window.
    #[must_use]
    pub fn new(
        match_op: PhysicalOperator,
        create_op: PhysicalOperator,
        on_create: Vec<MergeActionSpec>,
        on_match: Vec<MergeActionSpec>,
        output_binding: Option<BindingId>,
    ) -> Self {
        let schema = match output_binding {
            Some(b) => vec![b],
            None => Vec::new(),
        };
        Self {
            match_op: Box::new(match_op),
            create_op: Box::new(create_op),
            on_create,
            on_match,
            output_binding,
            schema,
            eos: false,
        }
    }

    /// Output schema — `[output_binding]` for a node-shape named merge
    /// (RETURN-after-MERGE), empty for the terminal path-shape /
    /// anonymous cases (ADR-151-amendment-01 §D-1).
    pub fn schema(&self) -> &[BindingId] {
        &self.schema
    }

    /// Pull the next batch — runs the match-or-create decision +
    /// action firing on first call; EOS thereafter.
    ///
    /// For a node-shape named merge (`Self::output_binding` = `Some`)
    /// the first call emits the matched/created binding row(s) with
    /// `ON CREATE`/`ON MATCH SET` mutations mirrored onto the emitted
    /// view (ADR-151-amendment-01 §D-1/§D-2). For path-shape /
    /// anonymous merges the first call emits an empty batch (terminal;
    /// the rows are still consumed for action-firing).
    pub fn next_batch<S: ExecutorSubstrate>(
        &mut self,
        ctx: &ExecutionContext,
        substrate: &S,
    ) -> Result<Batch, ExecutionError> {
        ctx.cancellation().check()?;
        if self.eos {
            return Ok(Batch::empty(self.schema.len()));
        }

        // NN-4 (#1384) re-spin — get-or-create critical section.
        //
        // The per-`(tenant, key)` serialization guard(s) for this MERGE are
        // acquired by the QUERY DRIVER (`materialize::materialize` /
        // `execute_with_context`) BEFORE this pipeline runs — see
        // [`crate::executor::acquire_merge_guards`] — and STASHED on the
        // `ExecutionContext`, held until AFTER the statement commits.
        //
        // WHY THE DRIVER, NOT HERE (Fix 1 — the ultracode-verify REJECT
        // catch + its deeper snapshot-pin root cause): under the D-2
        // statement-scoped autocommit wrap (`materialize::materialize`, the
        // SHIPPED Bolt/MCP path) `begin_statement` PINS the read snapshot
        // (installs a `BoltHeldTxn` whose LSN the match probe reads at) —
        // and that pin happens BEFORE this `next_batch`. If the guard were
        // acquired HERE (inside `next_batch`, after `begin_statement`), the
        // loser would already have pinned its snapshot at `begin_statement`
        // — BEFORE the winner committed — so even after blocking on the
        // guard its re-probe would read the stale pre-commit snapshot and
        // still double-create. Acquiring the guard in the driver BEFORE
        // `begin_statement` means the loser blocks BEFORE pinning its
        // snapshot; once it proceeds the winner has already committed, so
        // the loser's fresh snapshot sees the winner's node → match branch.
        // The match probe below therefore runs AFTER the lock is held
        // (probe-after-lock is load-bearing; probe-before-lock is the bug).
        //
        // The guard(s) are dropped by the driver only AFTER
        // `commit_statement`/`rollback_statement` (D-2 path) or after the
        // eager loop (`execute_with_context`, where each op auto-commits
        // inside `next_batch`, so the create is already durable). The guard
        // lifetime is thus [guard-acquire … statement commit], strictly
        // enclosing the match→create span.
        //
        // This `MergeOp` therefore holds NO lock itself; it simply relies on
        // the driver-held guard to have serialized the match→create window.
        // A keyless / anonymous merge (empty `merge_keys`) or a
        // read-only / stub substrate acquires no guard — byte-identical to
        // the pre-NN-4 path.
        let _exec_lsn = ctx.ensure_snapshot_lsn();

        // Phase 1: pull match-branch to exhaustion.
        let mut match_rows: Vec<Vec<Value>> = Vec::new();
        let match_schema = self.match_op.schema().to_vec();
        loop {
            let b = self.match_op.next_batch(ctx, substrate)?;
            if b.is_empty() {
                break;
            }
            for i in 0..b.row_count() {
                match_rows.push(b.row(i).to_vec());
            }
        }

        // The binding row(s) this MERGE will emit as its output. Only
        // populated for a node-shape named merge (RC-1/RC-3); left
        // empty for path-shape / anonymous merges (terminal).
        let emit = self.output_binding.is_some();
        let mut emitted: Vec<Vec<Value>> = Vec::new();

        if !match_rows.is_empty() {
            // Match branch taken — fire on_match actions per row,
            // mirroring each mutation onto the row's entity view so the
            // emitted post-SET state is correct (RC-2).
            for row in &mut match_rows {
                fire_actions(ctx, substrate, row, &match_schema, &self.on_match)?;
            }
            if emit {
                emitted = match_rows;
            }
        } else {
            // Create branch taken — pull create_op (one row out per
            // Phase 1-2 CREATE semantics).
            let create_schema = self.create_op.schema().to_vec();
            let b = self.create_op.next_batch(ctx, substrate)?;
            let mut create_rows: Vec<Vec<Value>> = Vec::with_capacity(b.row_count());
            for i in 0..b.row_count() {
                create_rows.push(b.row(i).to_vec());
            }
            for row in &mut create_rows {
                fire_actions(ctx, substrate, row, &create_schema, &self.on_create)?;
            }
            // Drain to EOS (Phase 1-2 CREATE ops require a second
            // next_batch call to settle their `emitted=true` flag).
            let _eos = self.create_op.next_batch(ctx, substrate)?;
            if emit {
                emitted = create_rows;
            }
        }

        self.eos = true;

        if !emit {
            // Path-shape / anonymous MERGE stays terminal (RC-3).
            return Ok(Batch::empty(self.schema.len()));
        }

        // Node-shape named MERGE — emit the matched/created binding
        // row(s) (RC-1), post-SET state already mirrored in (RC-2).
        let mut batch = Batch::with_capacity(self.schema.len());
        for row in emitted {
            if !batch.push_row(row) {
                return Err(ExecutionError::Eval(
                    "MergeOp: batch push overflow on RETURN-after-MERGE emission".into(),
                ));
            }
        }
        Ok(batch)
    }
}

/// **NN-4 (#1384) re-spin** — resolve a MERGE's [`MergeKeySpec`] list into
/// the concrete, injection-safe lock-key STRINGS the substrate's
/// [`ExecutorSubstrate::merge_guard`] serializes on, returned in CANONICAL
/// TOTAL ORDER (sorted + de-duplicated — Fix 3).
///
/// Returns an empty `Vec` when the merge carries no key (anonymous /
/// keyless) — the caller then runs without serialization. Returns ONE key
/// for a node-shape merge and up to TWO (source + target) for a path-shape
/// merge.
///
/// # Canonicalization (Fix 2)
///
/// Each key is a deterministic rendering of `(label, [(prop, value)]…)`
/// where:
/// - the properties are SORTED by name (order-independence: `{a:1,b:2}`
///   and `{b:2,a:1}` render identically — the match filter is an
///   order-insensitive AND-conjunction, so a verbatim-order key would
///   false-split into two mutexes and BOTH would create);
/// - an integral Float value is NORMALIZED to its Integer
///   ([`canonicalize_key_value`]), mirroring the `=`-operator's
///   `(x as f64) == y` coercion (`eval::values_equal_3vl`) — so `{v:1}`
///   (Integer) and `{v:1.0}` (Float), which the match filter treats as
///   EQUAL, lock on the SAME key.
///
/// # Total order (Fix 3 — no inter-path deadlock)
///
/// The resulting key STRINGS are sorted + de-duplicated so two path-MERGEs
/// naming the same source+target endpoints in OPPOSITE pattern order
/// acquire the two per-key mutexes in the SAME order → no ABBA lock
/// inversion; and a self-loop path (`(a:X{id:1})-[:R]->(b:X{id:1})`) whose
/// endpoints resolve to the SAME key acquires that ONE mutex ONCE (a second
/// `lock_arc()` on the same thread would self-deadlock).
///
/// The property VALUE expressions are evaluated NOW (execute time) against
/// the query's bound parameter bag so `MERGE (u:User {email:$e})` keys on
/// the resolved value of `$e`, not the literal text `$e`. Each value
/// renders through [`canonical_row_key`] (the same length-prefixed,
/// injection-safe encoding DISTINCT / GROUP BY use — #735 R1), so two
/// merges key IDENTICALLY iff their (label, resolved-property-set) are
/// equal, and never collide across distinct keys even under delimiter
/// injection in a string property.
///
/// Property expressions here are literals / parameters only (ADR-147 §D-4
/// — MERGE inline properties are literal-only), so evaluation needs no
/// upstream row: an empty row + a schema closure that resolves nothing is
/// passed; a stray variable-ref surfaces a clean `ExecutionError::Eval`
/// rather than a silent mis-key.
pub(crate) fn resolve_merge_keys(
    specs: &[MergeKeySpec],
    ctx: &ExecutionContext,
) -> Result<Vec<String>, ExecutionError> {
    let mut keys = Vec::with_capacity(specs.len());
    for spec in specs {
        keys.push(resolve_one_key(ctx, spec)?);
    }
    // Fix 3 — deterministic total order + de-dup (see the type doc above).
    keys.sort();
    keys.dedup();
    Ok(keys)
}

/// Resolve a single [`MergeKeySpec`] into its canonical, injection-safe
/// lock-key string (properties sorted by name + integral-Float→Integer
/// normalized — Fix 2).
fn resolve_one_key(ctx: &ExecutionContext, spec: &MergeKeySpec) -> Result<String, ExecutionError> {
    // Frame the key with the tenant + label so distinct tenants / labels
    // never share a lock. The tenant is ALSO passed to `merge_guard` (the
    // lock table is per-tenant), but including it here keeps the string
    // self-describing for diagnostics.
    let mut key = String::from("MK\u{1e}"); // record separator sentinel
    key.push_str(&format!("{:?}\u{1e}", ctx.tenant()));
    match &spec.label {
        Some(l) => {
            key.push_str("L:");
            key.push_str(&l.len().to_string());
            key.push(':');
            key.push_str(l);
        }
        None => key.push_str("L:-"),
    }
    // Empty row + no-op schema: MERGE inline property values are literals /
    // parameters (ADR-147 §D-4), never upstream variables.
    let empty_row: [Value; 0] = [];
    let no_schema = |_: BindingId| None;
    // Fix 2 — evaluate each property, then SORT by name so the key is
    // order-independent (the match filter is an order-insensitive
    // AND-conjunction). The canonical value is int/float-normalized
    // (`canonicalize_key_value`) so `{v:1}` and `{v:1.0}` lock alike.
    let mut resolved: Vec<(&str, Value)> = Vec::with_capacity(spec.properties.len());
    for (name, value_expr) in &spec.properties {
        let value = evaluate(value_expr, &empty_row, &no_schema, ctx.parameters())?;
        resolved.push((name.as_str(), canonicalize_key_value(value)));
    }
    resolved.sort_by(|a, b| a.0.cmp(b.0));
    for (name, value) in &resolved {
        key.push('\u{1e}');
        key.push_str(&name.len().to_string());
        key.push(':');
        key.push_str(name);
        key.push('=');
        key.push_str(&canonical_row_key(std::slice::from_ref(value)));
    }
    Ok(key)
}

/// `2^53` — the exact boundary of the contiguous integer range an `f64`
/// mantissa can represent losslessly. For a magnitude `< 2^53` every
/// integer has a UNIQUE `f64` image (spacing ≤ 1); at `2^53` the `f64`
/// spacing widens to 2, so `2^53` and `2^53 + 1` share the image
/// `2^53.0`. This is exactly where the `=`-operator's `(x as f64) == y`
/// coercion (see [`canonicalize_key_value`]) starts to merge DISTINCT
/// integers — so `|n| >= 2^53` is the routing threshold below.
const F64_EXACT_INT_BOUND: i64 = 1_i64 << 53; // 9_007_199_254_740_992

/// **NN-4 (#1384) re-spin, Fix 2** — normalize a MERGE key property value
/// so integer / float forms that the `=`-operator treats as EQUAL render
/// to the SAME [`canonical_row_key`] byte string.
///
/// The match filter compares a numeric key against an existing node's
/// property via `eval::values_equal_3vl`, whose cross-type arm coerces
/// `(x as f64) == y` (`x: i64`, `y: f64`) — LOSSY on the integer side. So
/// `Integer(1)` matches `Float(1.0)`, AND — above `2^53` — `Integer(2^53+1)`
/// matches `Float(2^53.0)` (because `(2^53+1) as f64 == 2^53.0`). But
/// `canonical_row_key` renders `Integer(n)` as `I:<n>` and `Float(f)` as
/// `F:<f.to_bits()>` — DIFFERENT strings → two mutexes → both create.
///
/// The invariant this function upholds: **two values the match filter calls
/// EQUAL MUST render to the SAME key.** We split at the `f64`-exact integer
/// boundary [`F64_EXACT_INT_BOUND`] (`2^53`):
///
/// - **`|n| < 2^53` (Integer) and `|f| < 2^53` integral (Float):** the
///   value has a UNIQUE `f64` image and, for a Float, a unique round-trip
///   integer. Render through the INTEGER bucket (`I:<n>`) — the Integer
///   verbatim, the integral Float normalized to its `Integer`. So `{v:1}`
///   and `{v:1.0}` lock identically, matching `(x as f64) == y`. No two
///   distinct integers below `2^53` collide, so `I:<n>` is exact + safe.
/// - **`|n| >= 2^53` (Integer) OR `|f| >= 2^53` integral (Float):** the
///   `(x as f64)` coercion is where the filter starts merging distinct
///   integers (`2^53` and `2^53+1` both → `2^53.0`). Render BOTH the
///   Integer and the integral Float through the FLOAT bucket
///   (`F:<(v as f64).to_bits()>`) — the SAME lossy coercion the filter
///   uses. `Integer(2^53)`, `Integer(2^53+1)`, and `Float(2^53.0)` all
///   collapse to `F:<(2^53.0).to_bits()>` and lock as ONE mutex, so the
///   filter's `Integer(2^53+1) ≡ Float(2^53.0) ≡ Integer(2^53)` overlap
///   (which is NOT transitive across the type boundary — both integers
///   match the float but not each other) is serialized under a single
///   get-or-create guard. Symmetric on the negative side (`-2^53`,
///   `-2^53-1`, `Float(-2^53.0)` → one key).
///
/// **Boundary decision (`>=` not `>`):** `2^53` ITSELF must go to the float
/// bucket. Although `Integer(2^53)` round-trips exactly, it shares its `f64`
/// image `2^53.0` with `Integer(2^53+1)` (the first lossy integer). If
/// `2^53` stayed in the `I:` bucket while `2^53+1` went to `F:`, the two —
/// both filter-equal to `Float(2^53.0)` — would SPLIT. Routing at `>= 2^53`
/// (i.e. `unsigned_abs() >= 2^53`) makes `2^53`, `2^53+1`, and `2^53.0` all
/// key identically. `unsigned_abs()` (not `abs()`) avoids the `i64::MIN`
/// overflow.
///
/// A fractional Float (`1.5`), a non-finite Float (`NaN`, `±inf`), or a
/// Float outside `i64` range keeps its `Float` encoding (there is no
/// Integer the match filter would coerce it to). Non-numeric values pass
/// through unchanged.
fn canonicalize_key_value(value: Value) -> Value {
    match value {
        // Integer side: exact + f64-unique below the boundary keeps `I:<n>`;
        // at/above the boundary route through the FLOAT bucket via the SAME
        // `as f64` coercion the match filter applies, so a lossy Integer
        // collides with the Float (and the other Integer) it is filter-equal
        // to.
        Value::Integer(n) => {
            if n.unsigned_abs() >= F64_EXACT_INT_BOUND as u64 {
                Value::Float(n as f64)
            } else {
                Value::Integer(n)
            }
        }
        // Float side: an integral, `i64`-round-trippable Float BELOW the
        // boundary normalizes to its `Integer` (so `{v:1.0}` locks with
        // `{v:1}`). At/above the boundary — where the filter merges distinct
        // integers onto one float — it STAYS in the float bucket so the
        // lossy Integer(s) routed to `F:<bits>` collide with it. A
        // fractional / non-finite / out-of-`i64`-range Float keeps `F:<bits>`.
        Value::Float(f) => {
            // `f.fract() == 0.0` catches integral floats; the `i64`
            // round-trip guard rejects magnitudes `i64` cannot represent
            // (and NaN/inf, whose `fract()` is NaN/NaN so `== 0.0` is false).
            if f.fract() == 0.0 && f >= i64::MIN as f64 && f <= i64::MAX as f64 {
                let as_int = f as i64;
                // Only normalize to `Integer` when (a) the cast round-trips
                // exactly AND (b) the magnitude is below `2^53` (the
                // f64-exact contiguous range). At/above `2^53` the float
                // shares its image with a distinct integer, so it must stay
                // in the float bucket to collide with that integer.
                if as_int as f64 == f && as_int.unsigned_abs() < F64_EXACT_INT_BOUND as u64 {
                    return Value::Integer(as_int);
                }
            }
            Value::Float(f)
        }
        other => other,
    }
}

/// **NN-4 (#1384) re-spin, Fix 1** — resolve + acquire the MERGE
/// get-or-create serialization guard(s) for every `MERGE` in `plan`, and
/// STASH them on `ctx` so they SPAN the statement commit.
///
/// The query driver ([`crate::materialize::materialize`] /
/// [`crate::executor::execute_with_context`]) calls this BEFORE it pins the
/// statement's read snapshot — i.e. BEFORE `begin_statement` on the D-2
/// auto-commit path. Acquiring the guard before the snapshot pin is
/// load-bearing: under D-2 `begin_statement` installs a `BoltHeldTxn` whose
/// pinned snapshot the MERGE match probe reads at, so a guard acquired only
/// INSIDE `MergeOp::next_batch` (after `begin_statement`) would let the
/// loser pin a stale pre-commit snapshot and still double-create. By
/// acquiring here first, the loser BLOCKS before pinning its snapshot;
/// once it proceeds the winner has committed, so the loser's fresh snapshot
/// sees the winner's node → match branch → get-or-create uniqueness.
///
/// The guard(s) are released by the driver AFTER `commit_statement` /
/// `rollback_statement` (D-2 path) or after the eager materialize loop
/// (`execute_with_context`, where each op auto-commits inside `next_batch`).
///
/// # Determinism (Fix 3 — no deadlock)
///
/// The keys of ALL merges in the plan are resolved, then globally SORTED +
/// de-duplicated, and the guards acquired in that ONE canonical total
/// order. So two concurrent statements naming the same key set — even
/// across two path endpoints in opposite pattern order — acquire the
/// per-key mutexes in the SAME order (no ABBA inversion), and a repeated
/// key (path self-loop) locks once.
///
/// A read plan, an anonymous / keyless merge, or a stub / read-only
/// substrate acquires nothing (the resolved key set is empty OR
/// `merge_guard` returns `None`) — byte-identical to the pre-NN-4 path.
///
/// # Errors
///
/// Surfaces an [`ExecutionError`] from key evaluation (a malformed merge
/// key expression) or from the substrate's `merge_guard`. The shipped
/// `merge_guard` (`CrudExecutorSubstrate::merge_guard`, a `parking_lot`
/// `lock_arc`) is INFALLIBLE, so on the production path only the key-eval
/// arm can error, and it errors BEFORE any guard is stashed → nothing to
/// leak. (A hypothetical FALLIBLE substrate that Err'd mid-multi-key
/// acquire would leak the guards stashed before the error: the driver's
/// `MergeGuardDrain` is bound AFTER this call, so it does NOT cover an
/// error from this function. Such a substrate must bind the drain before
/// the acquire, or have this function hold guards in a local `Vec`
/// transferred to `ctx` only on `Ok`. See the acquire-site comment in
/// `materialize::materialize_with_outer_lsn_held`.)
pub(crate) fn acquire_merge_guards<S: ExecutorSubstrate>(
    plan: &crate::logical_plan::LogicalPlan,
    substrate: &S,
    ctx: &ExecutionContext,
) -> Result<(), ExecutionError> {
    // Collect every merge's key specs, resolve to canonical strings, then
    // globally sort + de-dup so acquisition order is a single total order.
    let mut specs: Vec<MergeKeySpec> = Vec::new();
    collect_merge_key_specs(plan, &mut specs);
    if specs.is_empty() {
        return Ok(());
    }
    // `resolve_merge_keys` already sorts + de-dups WITHIN this spec list;
    // since we pass EVERY merge's specs at once, that yields the single
    // global total order across all merges in the plan.
    let keys = resolve_merge_keys(&specs, ctx)?;
    for key in keys {
        if let Some(guard) = substrate
            .merge_guard(ctx.tenant(), &key)
            .map_err(ExecutionError::Substrate)?
        {
            ctx.stash_merge_guard(guard);
        }
    }
    Ok(())
}

/// Recursively collect the `merge_keys` specs of every [`LogicalMerge`] in
/// `plan`. Walks the read-wrapper + write-nesting children (mirroring
/// [`crate::logical_plan::LogicalPlan::writes`]'s recursion) so a MERGE
/// under a `Project` (RETURN-after-MERGE), a join rewrite, or a `CALL`
/// body is found. At v1.0-α a MERGE is at / near the root, but the walk is
/// clause-order-agnostic + future-proof.
fn collect_merge_key_specs(plan: &crate::logical_plan::LogicalPlan, out: &mut Vec<MergeKeySpec>) {
    use crate::logical_plan::LogicalPlan as LP;
    match plan {
        LP::Merge(m) => {
            out.extend(m.merge_keys.iter().cloned());
            collect_merge_key_specs(&m.match_branch, out);
            collect_merge_key_specs(&m.create_branch, out);
        }
        // Single-input read/write wrappers — recurse into the driving input.
        LP::Filter(p) => collect_merge_key_specs(&p.input, out),
        LP::Project(p) => collect_merge_key_specs(&p.input, out),
        LP::Limit(l) => collect_merge_key_specs(&l.input, out),
        LP::Skip(s) => collect_merge_key_specs(&s.input, out),
        LP::Aggregate(a) => collect_merge_key_specs(&a.input, out),
        LP::Sort(s) => collect_merge_key_specs(&s.input, out),
        LP::Distinct(d) => collect_merge_key_specs(&d.input, out),
        LP::Unwind(u) => collect_merge_key_specs(&u.input, out),
        LP::ProcedureCall(p) => collect_merge_key_specs(&p.input, out),
        LP::NamedPath(n) => collect_merge_key_specs(&n.input, out),
        LP::DynamicLimit(l) => collect_merge_key_specs(&l.input, out),
        LP::Delete(d) => collect_merge_key_specs(&d.input, out),
        LP::Set(s) => collect_merge_key_specs(&s.input, out),
        LP::Remove(r) => collect_merge_key_specs(&r.input, out),
        // Multi-input nodes — recurse into every child.
        LP::Fusion(f) => f
            .inputs
            .iter()
            .for_each(|i| collect_merge_key_specs(i, out)),
        LP::Join(j) => {
            collect_merge_key_specs(&j.left, out);
            collect_merge_key_specs(&j.right, out);
        }
        LP::LeftOuterJoin(j) => {
            collect_merge_key_specs(&j.left, out);
            collect_merge_key_specs(&j.right, out);
        }
        LP::Union(u) => u.arms.iter().for_each(|a| collect_merge_key_specs(a, out)),
        LP::Call(c) => {
            collect_merge_key_specs(&c.input, out);
            collect_merge_key_specs(&c.body, out);
        }
        // CreateNode's optional `input` (a MATCH→CREATE chain) can nest a
        // prior write; recurse defensively.
        LP::CreateNode(c) => {
            if let Some(input) = &c.input {
                collect_merge_key_specs(input, out);
            }
        }
        LP::CreateRel(c) => {
            if let Some(input) = &c.input {
                collect_merge_key_specs(input, out);
            }
        }
        // Pure read / retrieval leaves — no nested merge.
        LP::Scan(_)
        | LP::PropertyIndexScan(_)
        | LP::CountStore(_)
        | LP::Expand(_)
        | LP::CommunityLookup(_)
        | LP::VectorNear(_)
        | LP::TextMatch(_)
        | LP::RankByHybrid(_)
        | LP::CorrelationSeed(_)
        | LP::CreateVectorIndex(_)
        | LP::CreatePropertyIndex(_)
        | LP::Empty(_) => {}
    }
}

/// Fire the given action items against the row + schema. Each item
/// resolves its `binding` to a cell in the row via [`schema_index`];
/// the cell's `NodeId` / `RelId` is extracted and dispatched to the
/// substrate's `set_node` / `set_rel` per the item's `kind`.
///
/// Takes the row by `&mut` so that — after each substrate write — the
/// SAME materialized mutation is mirrored onto the cell's in-memory
/// `NodeView` / `RelView` via [`apply_node_mutation_to_view`] /
/// [`apply_rel_mutation_to_view`] (ADR-151-amendment-01 §D-2, RC-2).
/// This keeps the emitted RETURN-after-MERGE row's post-SET state in
/// lock-step with the substrate, with the property-apply logic living
/// in ONE place (`set.rs`). For the terminal path-shape / anonymous
/// case the row is discarded after firing, so the mirror is a harmless
/// no-op cost.
///
/// Returns the first error encountered (defense-in-depth — the
/// Phase 5 narrowing per ADR-151 §D-9 ("Forward-deferred — MERGE
/// on path with MATCH-bound source / target" row; documented
/// locally in this module's `# Action-firing schema constraints`
/// heading) surfaces cleanly via `ExecutionError::Eval` when an
/// action item references a binding not in the executed branch's
/// output schema).
fn fire_actions<S: ExecutorSubstrate>(
    ctx: &ExecutionContext,
    substrate: &S,
    row: &mut [Value],
    schema: &[BindingId],
    items: &[MergeActionSpec],
) -> Result<(), ExecutionError> {
    for item in items {
        let idx = schema_index(schema, item.binding).ok_or_else(|| {
            ExecutionError::Eval(format!(
                "MergeOp: action item binding {:?} not in branch schema {:?} (Phase 5 narrowing \
                 per ADR-151 §D-9 (Forward-deferred — MERGE on path with MATCH-bound \
                 source / target) — Path-shape create-branch emits only the rel binding; \
                 actions referencing source/target bindings on the create branch \
                 forward-pinned to v1.1 per ADR-151 §D-9)",
                item.binding, schema
            ))
        })?;
        let cell = row.get_mut(idx).ok_or_else(|| {
            ExecutionError::Eval(format!("MergeOp: row missing cell at index {idx}"))
        })?;
        match item.kind {
            SetTargetKind::Node => {
                let node_id = node_id_from_value(cell)?;
                let mutation = build_node_mutation(&item.mutation)?;
                substrate
                    .set_node(ctx.tenant(), node_id, &mutation, ctx)
                    .map_err(ExecutionError::Substrate)?;
                // RC-2 — mirror the SAME mutation onto the emitted view
                // (single source of truth with set.rs); the create /
                // match cell is `Value::Node` for node-shape merges.
                if let Value::Node(view) = cell {
                    apply_node_mutation_to_view(view, &mutation);
                }
            }
            SetTargetKind::Rel => {
                let rel_id = rel_id_from_value(cell)?;
                let mutation = build_rel_mutation(&item.mutation)?;
                substrate
                    .set_rel(ctx.tenant(), rel_id, &mutation, ctx)
                    .map_err(ExecutionError::Substrate)?;
                if let Value::Relationship(view) = cell {
                    apply_rel_mutation_to_view(view, &mutation);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use arcgraph_core::{LabelId, Lsn, NodeId, PartitionId, TenantId};

    use super::*;
    use crate::ast::Literal;
    use crate::error::Span;
    use crate::executor::ops::{CreateNodeOp, EmptyOp, ScanOp};
    use crate::executor::substrate::StubExecutorSubstrate;
    use crate::executor::value::NodeView;
    use crate::semantic::bound_ast::BoundExpression;

    const N0: BindingId = BindingId::new(0);

    fn mk_create_node(var: BindingId, label: &str) -> PhysicalOperator {
        PhysicalOperator::CreateNode(CreateNodeOp::new(
            Some(var),
            Some(label.to_string()),
            Vec::new(),
        ))
    }

    fn mk_empty() -> PhysicalOperator {
        PhysicalOperator::Empty(EmptyOp::new())
    }

    fn lit_str(s: &str) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::String(s.into()),
            span: Span::point(1, 1),
            type_info: None,
        }
    }

    fn assign(name: &str, value: BoundExpression) -> LogicalSetMutation {
        LogicalSetMutation::PropertyAssign {
            name: name.into(),
            value,
        }
    }

    // Extract the single `NodeView` from a 1-row, 1-column batch.
    fn sole_node(b: &Batch) -> &NodeView {
        assert_eq!(b.row_count(), 1, "expected exactly one emitted row");
        assert_eq!(b.column_count(), 1, "expected the [binding] schema width");
        match &b.row(0)[0] {
            Value::Node(n) => n,
            other => panic!("expected Value::Node at column 0, got {other:?}"),
        }
    }

    #[test]
    fn merge_op_create_branch_emits_binding_row() {
        // RC-1/RC-4: a node-shape NAMED MERGE on an empty store —
        // match-branch returns 0 rows, create-branch fires, AND the op
        // now EMITS the created binding row (was empty pre-amendment;
        // ADR-151-amendment-01 lifts §D-9 RETURN-after-MERGE node-shape).
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let match_op = mk_empty();
        let create_op = mk_create_node(N0, "User");
        let mut op = MergeOp::new(match_op, create_op, Vec::new(), Vec::new(), Some(N0));
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        // RC-1 — emits the created binding row (Value::Node) at [n].
        let _node = sole_node(&b1);
        // Verify create-branch fired (substrate side-effect).
        let nodes = s.scan_nodes(tenant, None, Lsn::MAX).unwrap();
        assert_eq!(nodes.len(), 1, "create-branch should have created 1 node");
        let b2 = op.next_batch(&ctx, &s).expect("second batch OK");
        assert!(b2.is_empty(), "subsequent batches EOS");
    }

    #[test]
    fn merge_op_match_branch_emits_matched_row() {
        // RC-1/RC-4: pre-bake a User node. MERGE (n:User) — match-branch
        // scans + finds it; create-branch is NOT pulled (no new node);
        // the op EMITS the matched binding row carrying the node id.
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(7);
        let pre = NodeView::new(NodeId::new(1), Some(label));
        let s = StubExecutorSubstrate::new().with_node(tenant, pre);
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let match_op = PhysicalOperator::Scan(ScanOp::new(N0, Some(label), Lsn::MAX));
        let create_op = mk_create_node(N0, "User");
        let mut op = MergeOp::new(match_op, create_op, Vec::new(), Vec::new(), Some(N0));
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        // RC-1 — emits the matched binding row (the pre-baked node id).
        assert_eq!(sole_node(&b1).id, NodeId::new(1), "emits the matched node");
        // Verify match-branch fired, create-branch did NOT.
        let nodes = s.scan_nodes(tenant, None, Lsn::MAX).unwrap();
        assert_eq!(nodes.len(), 1, "no new node should have been created");
    }

    #[test]
    fn merge_op_terminal_when_output_binding_none() {
        // RC-3: an anonymous (or path-shape) MERGE — `output_binding`
        // is None — stays TERMINAL: the create-branch still fires its
        // side-effect, but the op emits an EMPTY batch (no row to
        // project; the path-shape match/create schemas are un-unionable).
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        // Anonymous create-branch: CreateNodeOp with no var.
        let create_op =
            PhysicalOperator::CreateNode(CreateNodeOp::new(None, Some("User".into()), Vec::new()));
        let mut op = MergeOp::new(mk_empty(), create_op, Vec::new(), Vec::new(), None);
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert!(b1.is_empty(), "terminal MERGE emits an empty batch");
        // The create-branch side-effect still fired.
        let nodes = s.scan_nodes(tenant, None, Lsn::MAX).unwrap();
        assert_eq!(nodes.len(), 1, "anonymous create-branch still creates");
        assert!(
            op.next_batch(&ctx, &s).expect("second batch OK").is_empty(),
            "EOS"
        );
    }

    #[test]
    fn merge_op_schema_reflects_output_binding() {
        // `Some(binding)` ⇒ schema `[binding]`; `None` ⇒ empty schema.
        let named = MergeOp::new(
            mk_empty(),
            mk_create_node(N0, "User"),
            vec![],
            vec![],
            Some(N0),
        );
        assert_eq!(named.schema(), &[N0], "named node-shape schema is [n]");
        let terminal = MergeOp::new(mk_empty(), mk_create_node(N0, "User"), vec![], vec![], None);
        assert!(terminal.schema().is_empty(), "terminal schema is empty");
    }

    #[test]
    fn merge_op_pre_cancellation_short_circuits() {
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        ctx.cancellation().cancel();
        let match_op = mk_empty();
        let create_op = mk_create_node(N0, "User");
        let mut op = MergeOp::new(match_op, create_op, Vec::new(), Vec::new(), Some(N0));
        let r = op.next_batch(&ctx, &s);
        assert_eq!(r, Err(ExecutionError::Cancelled));
    }

    #[test]
    fn merge_op_on_create_action_fires_on_create_branch() {
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let match_op = mk_empty();
        let create_op = mk_create_node(N0, "User");
        let on_create = vec![MergeActionSpec {
            binding: N0,
            kind: SetTargetKind::Node,
            mutation: assign("name", lit_str("Alice")),
        }];
        let mut op = MergeOp::new(match_op, create_op, on_create, Vec::new(), Some(N0));
        let _ = op.next_batch(&ctx, &s).expect("first batch OK");
        // The Stub's create_node allocates the NodeId; on_create
        // action sets the property bag (substrate side-effect).
        let node_id = NodeId::new((1u64 << 32) + 1);
        let bag = s
            .node_properties(tenant, node_id)
            .expect("on_create SET should have recorded a property bag");
        assert_eq!(
            bag.get("name"),
            Some(&Value::String("Alice".into())),
            "on_create action should have set the `name` property"
        );
    }

    #[test]
    fn merge_op_on_create_mirror_reflects_post_set_in_emitted_row() {
        // RC-2 (create-fires-on_create path): the EMITTED row's NodeView
        // reflects the `ON CREATE SET` mutation — not the pre-SET bag.
        let tenant = TenantId::DEFAULT;
        let s = StubExecutorSubstrate::new();
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let on_create = vec![MergeActionSpec {
            binding: N0,
            kind: SetTargetKind::Node,
            mutation: assign("name", lit_str("Alice")),
        }];
        let mut op = MergeOp::new(
            mk_empty(),
            mk_create_node(N0, "User"),
            on_create,
            vec![],
            Some(N0),
        );
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        assert_eq!(
            sole_node(&b1).properties.get("name"),
            Some(&Value::String("Alice".into())),
            "emitted create-branch row reflects ON CREATE SET (RC-2)"
        );
    }

    #[test]
    fn merge_op_on_match_mirror_reflects_post_set_in_emitted_row() {
        // RC-2 (match-fires-on_match path, the make-or-break): pre-bake a
        // node with id=1; MERGE matches; ON MATCH SET n.x = 2; the
        // EMITTED row's NodeView carries x=2 (the substrate write is
        // mirrored onto the in-memory row — without the mirror this
        // would be stale/Null).
        let tenant = TenantId::DEFAULT;
        let label = LabelId::new(7);
        let pre = NodeView::new(NodeId::new(1), Some(label)).with_property("id", Value::Integer(1));
        let s = StubExecutorSubstrate::new().with_node(tenant, pre);
        let ctx = ExecutionContext::new(tenant, PartitionId::ZERO);
        let match_op = PhysicalOperator::Scan(ScanOp::new(N0, Some(label), Lsn::MAX));
        let on_match = vec![MergeActionSpec {
            binding: N0,
            kind: SetTargetKind::Node,
            mutation: LogicalSetMutation::PropertyAssign {
                name: "x".into(),
                value: BoundExpression::Literal {
                    value: Literal::Integer(2),
                    span: Span::point(1, 1),
                    type_info: None,
                },
            },
        }];
        let mut op = MergeOp::new(
            match_op,
            mk_create_node(N0, "User"),
            vec![],
            on_match,
            Some(N0),
        );
        let b1 = op.next_batch(&ctx, &s).expect("first batch OK");
        let node = sole_node(&b1);
        assert_eq!(
            node.properties.get("x"),
            Some(&Value::Integer(2)),
            "emitted match-branch row reflects ON MATCH SET n.x = 2 (RC-2)"
        );
        // The pre-existing `id` property is preserved (PropertyAssign is
        // per-key, not a full-bag replace).
        assert_eq!(node.properties.get("id"), Some(&Value::Integer(1)));
    }

    // =================================================================
    // NN-4 (#1384) re-spin, Fix 2 — canonical lock-key resolution.
    //
    // These tests exercise `resolve_merge_keys` directly (the pure
    // key-resolution surface) so they need no substrate / race. They
    // are RED against the pre-respin `resolve_merge_key`, which
    // concatenated properties in verbatim pattern order (order-swap
    // false-split) and encoded `Integer(1)` vs `Float(1.0)` distinctly
    // (int/float false-split).
    // =================================================================

    fn lit_int(n: i64) -> BoundExpression {
        BoundExpression::Literal {
            value: Literal::Integer(n),
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

    /// Resolve a list of [`MergeKeySpec`]s to canonical lock-key strings
    /// via the free `resolve_merge_keys` (the surface the query driver
    /// calls). Panics on eval error (all these tests pass literals).
    fn resolve(specs: Vec<MergeKeySpec>) -> Vec<String> {
        let ctx = ExecutionContext::new(TenantId::DEFAULT, PartitionId::ZERO);
        resolve_merge_keys(&specs, &ctx).expect("resolve merge keys")
    }

    fn key_spec(label: &str, props: Vec<(&str, BoundExpression)>) -> MergeKeySpec {
        MergeKeySpec {
            label: Some(label.into()),
            properties: props.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        }
    }

    #[test]
    fn resolve_merge_key_is_property_order_independent_nn4() {
        // Fix 2 (order): `{a:1,b:2}` and `{b:2,a:1}` MUST resolve to the
        // SAME lock key — the match filter is an order-insensitive
        // AND-conjunction, so a verbatim-order key would false-split into
        // two mutexes and BOTH clients would create.
        let ka = resolve(vec![key_spec(
            "User",
            vec![("a", lit_int(1)), ("b", lit_int(2))],
        )]);
        let kb = resolve(vec![key_spec(
            "User",
            vec![("b", lit_int(2)), ("a", lit_int(1))],
        )]);
        assert_eq!(
            ka, kb,
            "order-swapped composite keys MUST canonicalize identically \
             (RED against verbatim-pattern-order concatenation)"
        );
        // And DISTINCT property sets must NOT collide (guard the sort did
        // not collapse everything to a constant).
        let ac = resolve(vec![key_spec(
            "User",
            vec![("a", lit_int(1)), ("b", lit_int(3))],
        )]);
        assert_ne!(ka, ac, "distinct property values must key distinctly");
    }

    #[test]
    fn resolve_merge_key_normalizes_integral_float_to_integer_nn4() {
        // Fix 2 (int/float): `{v:1}` (Integer) and `{v:1.0}` (Float) —
        // which the `=`-operator treats as EQUAL via `(x as f64)==y` —
        // MUST resolve to the SAME lock key (RED against `I:1` vs `F:bits`
        // distinct encoding).
        let int_key = resolve(vec![key_spec("User", vec![("v", lit_int(1))])]);
        let float_key = resolve(vec![key_spec("User", vec![("v", lit_float(1.0))])]);
        assert_eq!(
            int_key, float_key,
            "integral Float `1.0` must normalize to Integer `1` and lock \
             identically with Integer `1` (mirrors eval `(x as f64)==y`)"
        );
        // A FRACTIONAL float is NOT an integer — it must NOT collapse onto
        // any integer key (there is no integer the match filter coerces
        // `1.5` to).
        let frac_key = resolve(vec![key_spec("User", vec![("v", lit_float(1.5))])]);
        assert_ne!(
            int_key, frac_key,
            "fractional Float 1.5 must NOT normalize onto Integer 1's key"
        );
    }

    #[test]
    fn resolve_merge_key_canonicalizes_both_axes_together_nn4() {
        // Fix 2 (combined): `{a:1, b:2.0}` in one order vs `{b:2, a:1.0}`
        // in the other — order-swap AND int/float normalization together —
        // MUST resolve identically (the composite `{tenant_id,external_id}`
        // + driver-float shape the verdict flagged).
        let one = resolve(vec![key_spec(
            "User",
            vec![("a", lit_int(1)), ("b", lit_float(2.0))],
        )]);
        let two = resolve(vec![key_spec(
            "User",
            vec![("b", lit_int(2)), ("a", lit_float(1.0))],
        )]);
        assert_eq!(
            one, two,
            "order-swapped + int/float-mixed composite keys canonicalize \
             identically across BOTH axes"
        );
    }

    #[test]
    fn resolve_merge_key_large_int_vs_rounded_float_collide_above_2p53_nn4() {
        // Fix 4 (>2^53 residual) — the match filter's `values_equal_3vl`
        // numeric arm coerces `(x as f64) == y` (LOSSY on the i64 side), so
        // above 2^53 DISTINCT integers share a float image:
        //   `9007199254740993` (2^53+1, Integer) and `9007199254740992.0`
        //   (2^53, Float) are EQUAL to the match filter (`(2^53+1) as f64
        //   == 2^53.0`), yet the pre-Fix-4 canonicalize rendered the
        //   Integer verbatim (`I:...993`) and NORMALIZED the Float to
        //   `Integer(2^53)` (`I:...992`) → the two lock keys SPLIT → two
        //   mutexes → both create → duplicate. Fix 4 routes BOTH through the
        //   FLOAT bucket (`F:<(v as f64).to_bits()>`) so they collide.
        const P53: i64 = 1_i64 << 53; // 9_007_199_254_740_992
        let int_2p53p1 = resolve(vec![key_spec("N", vec![("v", lit_int(P53 + 1))])]);
        let float_2p53 = resolve(vec![key_spec("N", vec![("v", lit_float(P53 as f64))])]);
        assert_eq!(
            int_2p53p1, float_2p53,
            "Integer(2^53+1) and Float(2^53.0) — which the match filter calls \
             EQUAL via `(x as f64)==y` — MUST resolve to the SAME lock key \
             (RED against `I:...993` vs normalized `I:...992` split → double \
             create). This is the F4 >2^53 residual the verdict reproduced."
        );

        // 2^53 ITSELF (the boundary) must ALSO collide with 2^53+1 and the
        // Float — it shares its float image `2^53.0` with 2^53+1 (the first
        // lossy integer). If `2^53` stayed in the `I:` bucket while `2^53+1`
        // went to `F:`, the two (both filter-equal to Float(2^53.0)) would
        // SPLIT — this is exactly why the boundary is `>=` not `>`.
        let int_2p53 = resolve(vec![key_spec("N", vec![("v", lit_int(P53))])]);
        assert_eq!(
            int_2p53, int_2p53p1,
            "Integer(2^53) and Integer(2^53+1) share the float image 2^53.0 \
             → MUST key identically (boundary is `>= 2^53`, not `> 2^53`)"
        );
        assert_eq!(
            int_2p53, float_2p53,
            "Integer(2^53) and Float(2^53.0) MUST key identically"
        );

        // 2^54 (also above the boundary, and exactly representable) — the
        // Integer and its Float image collide too.
        const P54: i64 = 1_i64 << 54; // 18_014_398_509_481_984
        let int_2p54 = resolve(vec![key_spec("N", vec![("v", lit_int(P54))])]);
        let float_2p54 = resolve(vec![key_spec("N", vec![("v", lit_float(P54 as f64))])]);
        assert_eq!(
            int_2p54, float_2p54,
            "Integer(2^54) and Float(2^54.0) MUST key identically above 2^53"
        );

        // NEGATIVE mirror — `-2^53-1` (Integer) and `-2^53.0` (Float) are
        // filter-equal (`(-2^53-1) as f64 == -2^53.0`) and MUST collide; so
        // must `-2^53` (Integer). `unsigned_abs()` handles the sign without
        // an `abs()` overflow at `i64::MIN`.
        let int_neg = resolve(vec![key_spec("N", vec![("v", lit_int(-P53 - 1))])]);
        let float_neg = resolve(vec![key_spec("N", vec![("v", lit_float(-(P53 as f64)))])]);
        let int_neg_boundary = resolve(vec![key_spec("N", vec![("v", lit_int(-P53))])]);
        assert_eq!(
            int_neg, float_neg,
            "negative mirror: Integer(-2^53-1) and Float(-2^53.0) MUST key \
             identically (verdict verified the symmetric negative split)"
        );
        assert_eq!(
            int_neg_boundary, float_neg,
            "negative boundary: Integer(-2^53) and Float(-2^53.0) MUST key \
             identically"
        );

        // Below the boundary stays EXACT — `2^53-1` is f64-unique, so the
        // Integer keeps `I:...` and a distinct integer does NOT collide.
        let int_below = resolve(vec![key_spec("N", vec![("v", lit_int(P53 - 1))])]);
        let float_below = resolve(vec![key_spec(
            "N",
            vec![("v", lit_float((P53 - 1) as f64))],
        )]);
        assert_eq!(
            int_below, float_below,
            "below 2^53 (2^53-1) the Integer and its exact Float still key \
             identically (existing {{v:1}}=={{v:1.0}} equivalence, extended)"
        );
        let int_below2 = resolve(vec![key_spec("N", vec![("v", lit_int(P53 - 2))])]);
        assert_ne!(
            int_below, int_below2,
            "distinct integers BELOW 2^53 (f64-unique) MUST NOT collide — the \
             float-bucket routing must not over-collapse the exact range"
        );

        // A genuinely-fractional float must NEVER collide with any integer
        // key (no integer the filter coerces it to). Above 2^53 there ARE no
        // representable fractional floats (spacing is >= 2, every f64 is
        // integral), so use a sub-boundary fraction (`2.5`) — its bits differ
        // from any integer's `I:` key, and it is not integral so it stays in
        // the float bucket rather than normalizing onto `Integer(2)`.
        let two = resolve(vec![key_spec("N", vec![("v", lit_int(2))])]);
        let frac = resolve(vec![key_spec("N", vec![("v", lit_float(2.5))])]);
        assert_ne!(
            two, frac,
            "a fractional float (2.5) must not collide with an integer key"
        );
    }

    #[test]
    fn resolve_merge_keys_path_endpoints_sorted_total_order_nn4() {
        // Fix 3 — a path-shape MERGE carries TWO endpoint keys (source +
        // target); `resolve_merge_keys` returns them in a canonical TOTAL
        // ORDER (sorted) so two path-MERGEs naming the same endpoints in
        // OPPOSITE pattern order acquire the two mutexes in the SAME order
        // (no inter-path deadlock).
        // Pattern A: (User{id:1})-[:R]->(Org{id:2})
        let ab = resolve(vec![
            key_spec("User", vec![("id", lit_int(1))]),
            key_spec("Org", vec![("id", lit_int(2))]),
        ]);
        // Pattern B: endpoints supplied in the OPPOSITE order.
        let ba = resolve(vec![
            key_spec("Org", vec![("id", lit_int(2))]),
            key_spec("User", vec![("id", lit_int(1))]),
        ]);
        assert_eq!(
            ab, ba,
            "path endpoint keys MUST be returned in canonical total order \
             regardless of pattern order (deadlock-free acquisition)"
        );
        assert_eq!(ab.len(), 2, "two distinct endpoints → two keys");
    }

    #[test]
    fn resolve_merge_keys_dedups_identical_endpoints_nn4() {
        // Fix 3 self-deadlock guard — a path whose source and target
        // resolve to the SAME key must acquire that ONE mutex ONCE
        // (acquiring the same per-key mutex twice on one thread would
        // self-deadlock on the second `lock_arc()`).
        let keys = resolve(vec![
            key_spec("User", vec![("id", lit_int(1))]),
            key_spec("User", vec![("id", lit_int(1))]),
        ]);
        assert_eq!(
            keys.len(),
            1,
            "identical source+target endpoints dedup to ONE lock key \
             (prevents same-thread re-lock self-deadlock)"
        );
    }

    #[test]
    fn resolve_merge_keys_empty_when_no_specs_nn4() {
        // Anonymous / keyless merge → no key → runs unserialized.
        let keys = resolve(Vec::new());
        assert!(keys.is_empty(), "no specs → no lock keys");
    }
}
