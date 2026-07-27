//! Built-in function signature registry consumed by M4-22's
//! [`crate::semantic::type_check::TypeCheckVisitor`].
//!
//! # Scope
//!
//! The registry covers the openCypher v9 §3 built-ins admitted at
//! v1.0 plus the ArcGraph-specific extensions (`vector_distance`,
//! `text_match`). The W28 conformance slice (Task #652, Feature #648)
//! expanded the standard scalar surface with the
//! string / mathematical / list / type-conversion built-ins the
//! vendored openCypher TCK expression corpus
//! (`crates/arcgraph-tck/tck/features/expressions/{string,mathematical,list,typeConversion}`)
//! defines — `toUpper`/`toLower`/`trim`/`lTrim`/`rTrim`/`substring`/
//! `replace`/`split`/`reverse`/`toString` (string),
//! `abs`/`ceil`/`floor`/`round`/`sign`/`sqrt`/`exp`/`log`/`log10`/
//! `sin`/`cos`/`tan`/`e`/`pi` (math), `toInteger`/`toFloat`/
//! `toBoolean` (conversion), `coalesce`/`range` (scalar/list) — plus
//! the eval wiring for the already-registered `size`/`length`/`head`/
//! `last`/`tail`/`keys`. `left`/`right` are registered alongside the
//! string set but are **Neo4j extensions** (Neo4j Cypher manual,
//! String functions), NOT core openCypher — the openCypher TCK
//! `String8`/`String9` features specify `STARTS WITH`/`ENDS WITH`,
//! not `left`/`right`. Predicate functions (`all`/`any`/`none`/
//! `single`), `reduce`, comprehensions, and aggregations are
//! deliberately OUT of this slice. `rand()` (nullary, returns a
//! `Float` in `[0,1)`; openCypher v9 §3 scalar) IS registered
//! (GA-rand slice, #618): it is non-deterministic, so there is no
//! oracle for its VALUE, but the openCypher `Quantifier9`..`Quantifier12`
//! invariant scenarios consume it only via
//! `[y IN list WHERE rand() > 0.5 | y]` to build an arbitrary
//! sublist, then assert results that hold for ANY sublist
//! (random-INDEPENDENT), so registration + a uniform `[0,1)` eval
//! (`executor::eval`) unblocks them without a value oracle.
//! (`properties` — node/rel/map → property map — was
//! added in the GA function-registry slice (#618) once `Value::Map`
//! landed; `keys` likewise now accepts a map.) Each entry pins:
//!
//! - the function name; openCypher functions are CASE-INSENSITIVE
//!   (`toInteger`/`TOINTEGER`/`tointeger` all denote the same
//!   function), so [`lookup`] case-folds and the canonical camelCase
//!   spelling is preserved only as `sig.name` (#618);
//! - the arity (exact `Fixed(n)` or variadic `Variadic { min }`);
//! - a return-type producer `fn(&[TypeInfo]) -> TypeInfo` so
//!   parametric returns (e.g., `head(List(elem)) -> elem`) are
//!   first-class;
//! - per-argument type predicates (the simplest "any" / "specific"
//!   constraint at v1.0; v1.1 may extend with subtyping / coercion
//!   rules).
//!
//! # Why a custom mini-registry (vs. `phf` / `lazy_static`)
//!
//! v1.0 has ~17 functions. A linear scan against a `&'static` slice
//! is well within the parse-budget (D-12); a derived perfect hash
//! function would be over-engineering for the slice size. v1.1's
//! function-set growth (LIST / DATE / DURATION builtins) may justify
//! a switch; the surface here keeps that refactor possible without
//! breaking callers.
//!
//! # ADR provenance
//! - ADR-038 §2 D-22 — type-check + reserved-variant rejection (the
//!   M4-22 contract that consumes this registry).
//! - openCypher v9 §3.1–§3.3 — built-in function set.

use crate::semantic::bound_ast::TypeInfo;

/// Function arity declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// Exactly `n` arguments.
    Fixed(usize),
    /// At least `min` arguments; no upper bound.
    Variadic { min: usize },
}

/// A single per-argument type predicate. Kept simple at v1.0 — the
/// type-checker uses these only to reject obvious mismatches; it
/// does NOT do subtyping or coercion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// Accepts any type. Most v1.0 built-ins accept `Any` because
    /// the openCypher spec leans on dynamic typing.
    Any,
    /// Accepts only the listed types (Boolean / Integer / Float /
    /// String / Null). Used by arithmetic / string-ops where a
    /// stronger constraint catches obvious errors.
    Numeric,
    /// Accepts only types that are list-shaped.
    List,
    /// Accepts only types that are node-shaped.
    Node,
    /// Accepts property-bag-bearing shapes — Node / Relationship / Map
    /// (plus the dynamically-typed `Property` escape + universal `Null`).
    /// Used by `properties()` (#618). Unlike the scalar/math family's
    /// `Any` + runtime-enforcement posture, openCypher rejects
    /// `properties(<scalar|list>)` at COMPILE time (`InvalidArgumentType`
    /// — TCK `Graph9` `5`/`6`/`7`: `properties(1)` / `properties('x')` /
    /// `properties([true,false])` "raised at compile time"). A concrete
    /// scalar / list LITERAL carries its type (`literal_type`), so the
    /// static check fires; a dynamically-typed `Property` access is
    /// admitted (the catalog under-types properties at v1.0, so a
    /// compile-time reject there would false-positive) and the eval arm
    /// enforces it at runtime.
    MapLike,
    /// **#618** — `type()` argument: relationship-only. REJECT-semantics
    /// (see [`ArgKind::accepts`]): a concrete Node / Path argument is a
    /// COMPILE-time `InvalidArgumentType` (TCK `Graph4` `7`); scalars /
    /// `Property` / `Null` / unknown are admitted (eval-enforced).
    RelOnly,
    /// **#618** — `length()` argument: path-only. REJECT-semantics: a
    /// concrete Node / Relationship argument is a COMPILE-time
    /// `InvalidArgumentType` (TCK `Path3` `2`/`3`); scalars / `Property`
    /// / `Null` / unknown are admitted (eval-enforced).
    PathOnly,
    /// **#618** — `size()` argument: list/string-like. REJECT-semantics:
    /// a concrete Node / Relationship / Path argument is a COMPILE-time
    /// `InvalidArgumentType` (TCK `List6` `5` `size(path)`); scalars /
    /// `Property` / `Null` / unknown are admitted (eval-enforced).
    ListLike,
}

impl ArgKind {
    /// Return `true` if `ti` satisfies this kind.
    pub fn accepts(self, ti: &TypeInfo) -> bool {
        // Null is admissible everywhere — 3VL propagation per D-20.
        if matches!(ti, TypeInfo::Null) {
            return true;
        }
        match self {
            ArgKind::Any => true,
            ArgKind::Numeric => matches!(
                ti,
                TypeInfo::Integer
                    | TypeInfo::Float
                    // Dynamic-schema discipline (#773): the v1.0 catalog
                    // under-types every property access as the
                    // `Property::String` sentinel (type_check.rs assigns
                    // it WITHOUT per-label type info), so restricting to
                    // `Property{Integer|Float}` here false-positives EVERY
                    // `sum(prop)` / `avg(prop)` — the literal Customer-Zero
                    // AML "mule-by-volume" `sum(t.amount)` HAVING included.
                    // Admit ANY property access at compile time (the
                    // aggregate eval arm enforces numericity at runtime),
                    // matching `is_numeric` (type_check.rs) and
                    // `ArgKind::MapLike` below — both deliberately admit
                    // `Property { .. }` for exactly this under-typed-catalog
                    // reason.
                    | TypeInfo::Property { .. }
            ),
            ArgKind::List => matches!(ti, TypeInfo::List(_)),
            ArgKind::Node => matches!(ti, TypeInfo::Node { .. }),
            // Node / Relationship / Map accept; a dynamically-typed
            // `Property` access is admitted (runtime-enforced) to avoid a
            // false-positive on the under-typed v1.0 catalog. Scalars /
            // lists fall through to `false` → compile-time reject.
            ArgKind::MapLike => matches!(
                ti,
                TypeInfo::Node { .. }
                    | TypeInfo::Relationship { .. }
                    | TypeInfo::Map
                    | TypeInfo::Property { .. }
            ),
            // #618 — REJECT-semantics kinds (the complement of the
            // accept-list kinds above). `type()`/`length()`/`size()` are
            // restricted to a specific value kind by openCypher, but the
            // v1.0 catalog under-types scalars / properties, so an
            // accept-list (like `ArgKind::List`) would false-positive
            // (`size(n.numbers)` is `Property`, `size(x)` is unknown).
            // Instead we reject ONLY the concrete graph-element kinds the
            // function definitely cannot take (the cases the openCypher
            // TCK rejects at COMPILE time), and admit scalars / `Property`
            // / `Null` / unknown — the eval arm enforces the finer
            // type at runtime, preserving the existing W28
            // runtime-enforcement posture for everything else.
            //
            // `RelOnly` (`type()`): a Node / Path argument is a static
            // error (`Graph4` [7] `type(node)`).
            ArgKind::RelOnly => !matches!(ti, TypeInfo::Node { .. } | TypeInfo::Path),
            // `PathOnly` (`length()`): a Node / Relationship argument is a
            // static error (`Path3` [2]/[3] `length(node)`/`length(rel)`).
            ArgKind::PathOnly => {
                !matches!(ti, TypeInfo::Node { .. } | TypeInfo::Relationship { .. })
            }
            // `ListLike` (`size()`): a Node / Relationship / Path argument
            // is a static error (`List6` [5] `size(path)`).
            ArgKind::ListLike => !matches!(
                ti,
                TypeInfo::Node { .. } | TypeInfo::Relationship { .. } | TypeInfo::Path
            ),
        }
    }
}

/// A function signature.
///
/// `return_type_for` is a function pointer rather than a concrete
/// `TypeInfo` so parametric returns (e.g. `head` / `last` / `tail`
/// over a `List(elem)`) can yield the element type at call site.
#[derive(Debug, Clone, Copy)]
pub struct FunctionSig {
    pub name: &'static str,
    pub arity: Arity,
    /// Per-argument constraints. For `Variadic { min }` signatures,
    /// `arg_kinds[..min]` apply to the first `min` arguments and the
    /// last entry (`arg_kinds[min - 1]` if length == min, or
    /// `arg_kinds[min]` if length > min) repeats for the variadic
    /// tail.
    pub arg_kinds: &'static [ArgKind],
    pub return_type_for: fn(&[TypeInfo]) -> TypeInfo,
}

// ---------- return-type producers ----------

fn ret_integer(_args: &[TypeInfo]) -> TypeInfo {
    TypeInfo::Integer
}

fn ret_float(_args: &[TypeInfo]) -> TypeInfo {
    TypeInfo::Float
}

fn ret_string(_args: &[TypeInfo]) -> TypeInfo {
    TypeInfo::String
}

fn ret_boolean(_args: &[TypeInfo]) -> TypeInfo {
    TypeInfo::Boolean
}

fn ret_list_of_string(_args: &[TypeInfo]) -> TypeInfo {
    TypeInfo::List(Box::new(TypeInfo::String))
}

fn ret_list_of_integer(_args: &[TypeInfo]) -> TypeInfo {
    // `range(start, end[, step]) -> List(Integer)`. The element type
    // is always `Integer` (openCypher v9 §3 — `range` produces an
    // integer list; non-integer arguments are a runtime
    // `InvalidArgumentType` per the TCK `List11` Scenario [5]).
    TypeInfo::List(Box::new(TypeInfo::Integer))
}

fn ret_collect_list(args: &[TypeInfo]) -> TypeInfo {
    // `collect(x)` builds List(elem-type). If we have type-info on
    // the argument, propagate; else `List(Any)` is approximated as
    // `List(Null)` (a v1.0 placeholder).
    let elem = args.first().cloned().unwrap_or(TypeInfo::Null);
    TypeInfo::List(Box::new(elem))
}

fn ret_first_arg_or_null(args: &[TypeInfo]) -> TypeInfo {
    args.first().cloned().unwrap_or(TypeInfo::Null)
}

fn ret_list_element(args: &[TypeInfo]) -> TypeInfo {
    // `head(list) -> elem`, `last(list) -> elem`.
    match args.first() {
        Some(TypeInfo::List(elem)) => (**elem).clone(),
        _ => TypeInfo::Null,
    }
}

fn ret_list_same_shape(args: &[TypeInfo]) -> TypeInfo {
    // `tail(List(t)) -> List(t)`.
    match args.first() {
        Some(t @ TypeInfo::List(_)) => t.clone(),
        _ => TypeInfo::List(Box::new(TypeInfo::Null)),
    }
}

fn ret_list_of_node(_args: &[TypeInfo]) -> TypeInfo {
    // ADR-193 D-7 — `nodes(path) -> List(Node)`. The element label is
    // unknown at the function-return level (a path may traverse mixed
    // labels), so `Node { label: None }`.
    TypeInfo::List(Box::new(TypeInfo::Node { label: None }))
}

fn ret_list_of_relationship(_args: &[TypeInfo]) -> TypeInfo {
    // ADR-193 D-7 — `relationships(path) -> List(Relationship)`. The
    // element rel-type is unknown at the function-return level.
    TypeInfo::List(Box::new(TypeInfo::Relationship { rel_type: None }))
}

fn ret_map(_args: &[TypeInfo]) -> TypeInfo {
    // #618 — `properties(node|rel|map) -> Map` (the property bag). The
    // key/value shape is not tracked at the function-return level, so a
    // bare `Map` is returned (matching the map-literal / map-projection
    // carriers' `TypeInfo::Map`).
    TypeInfo::Map
}

// ---------- The v1.0 built-in registry ----------

/// Static registry of built-in function signatures.
///
/// Order matches openCypher v9 §3 documentation order for human
/// readability; lookup is a linear scan (`O(n)` over ~50 entries —
/// still well within the D-12 parse-budget; a `phf`/perfect-hash
/// switch is a v1.1 option per the module doc if the set keeps
/// growing).
///
/// Names use the openCypher canonical spelling (camelCase for the
/// `to*` family + `lTrim`/`rTrim`); [`lookup`] is CASE-INSENSITIVE
/// (#618 — openCypher functions are case-insensitive, so `toInteger`/
/// `TOINTEGER`/`tointeger` all resolve to the same signature) and the
/// evaluator (`executor::eval::apply_function`) independently
/// lower-cases before dispatch so the eval arms match `tointeger`/
/// `toupper`/`ltrim` etc. No two builtin names are case-variants of
/// one another, so the case-fold never aliases distinct functions.
#[rustfmt::skip]
pub const BUILTINS: &[FunctionSig] = &[
    // Identity / metadata
    FunctionSig { name: "id",     arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],     return_type_for: ret_integer },
    FunctionSig { name: "labels", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Node],    return_type_for: ret_list_of_string },
    // `type()` is relationship-only (#618): `type(node)` rejects at
    // COMPILE time (`Graph4` [7]). REJECT-semantics admit scalars /
    // `Property` / `Null` / unknown (eval-enforced).
    FunctionSig { name: "type",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::RelOnly], return_type_for: ret_string },
    FunctionSig { name: "keys",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],  return_type_for: ret_list_of_string },
    // `properties(node|rel|map) -> Map` (#618). `MapLike` arg-kind —
    // openCypher rejects `properties(<scalar|list>)` at COMPILE time
    // (`InvalidArgumentType`, TCK `Graph9` [5]/[6]/[7]), so a concrete
    // scalar / list LITERAL is statically rejected here; Node / Rel / Map
    // / Null / (dynamic) Property are admitted and the eval arm
    // (`fn_properties`) enforces the runtime split (NULL → NULL;
    // `properties(map)` identity; node/rel → their property bag).
    FunctionSig { name: "properties", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::MapLike], return_type_for: ret_map },
    FunctionSig { name: "exists", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],  return_type_for: ret_boolean },

    // Aggregations
    FunctionSig { name: "count",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],     return_type_for: ret_integer },
    FunctionSig { name: "sum",     arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Numeric], return_type_for: ret_first_arg_or_null },
    FunctionSig { name: "avg",     arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Numeric], return_type_for: ret_float },
    FunctionSig { name: "min",     arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],     return_type_for: ret_first_arg_or_null },
    FunctionSig { name: "max",     arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],     return_type_for: ret_first_arg_or_null },
    FunctionSig { name: "collect", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any],     return_type_for: ret_collect_list },

    // Lists / sizes. `length()` is path-only (#618): `length(node)` /
    // `length(rel)` reject at COMPILE time (`Path3` [2]/[3]). `size()` is
    // list/string-like (#618): `size(path)` rejects at COMPILE time
    // (`List6` [5]). Both use REJECT-semantics arg-kinds (admit scalars /
    // `Property` / `Null` / unknown — eval-enforced) so the under-typed
    // v1.0 catalog does not false-positive (`size(n.numbers)` /
    // `size(x)`).
    FunctionSig { name: "length", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::PathOnly], return_type_for: ret_integer },
    FunctionSig { name: "size",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::ListLike], return_type_for: ret_integer },
    FunctionSig { name: "head",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::List], return_type_for: ret_list_element },
    FunctionSig { name: "last",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::List], return_type_for: ret_list_element },
    FunctionSig { name: "tail",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::List], return_type_for: ret_list_same_shape },

    // Path functions (ADR-193 D-7). `arg_kinds: Any` — the eval arm owns
    // the path-vs-non-path enforcement (NULL → NULL 3VL; non-path →
    // InvalidArgumentType), matching the W28 runtime-enforcement posture
    // for the `length`/`head`/… family. `length` is ALREADY registered
    // (Any → Integer) above and admits a path arg unchanged; only its
    // EVAL arm is net-new for paths. `nodes`/`relationships` are net-new.
    FunctionSig { name: "nodes",         arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_list_of_node },
    FunctionSig { name: "relationships", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_list_of_relationship },

    // ---- W28 conformance additions (Task #652) ----
    // openCypher v9 §3 standard scalar built-ins. `arg_kinds` are
    // `Any` across the board so the *runtime* eval arm owns type
    // enforcement: openCypher raises these argument-type errors AT
    // RUNTIME (the TCK is explicit — `range` -> "raised at runtime:
    // InvalidArgumentType" (`List11` Scenario [5]); `toBoolean`/
    // `toInteger`/`toFloat`/`toString` -> "raised at runtime:
    // InvalidArgumentValue" (`TypeConversion1/2/3/4`)). A compile-time
    // `Numeric` constraint would FALSE-POSITIVE reject a valid query
    // whenever the static type is imprecise — e.g. a node property
    // whose type the v1.0 catalog does not track (it defaults to
    // `String`), a parameter, or a `coalesce(...)` result — rejecting
    // `abs(n.age)` even though `n.age` is an Integer at runtime. The
    // eval arms (`num_to_float` / `expect_integer` / ...) enforce the
    // real types + propagate NULL. The evaluator lower-cases the name
    // before dispatch.

    // String (TCK expressions/string/*). `reverse` is polymorphic
    // (String -> String, List -> List) so it preserves the arg type.
    // `left`/`right` are Neo4j extensions (Neo4j Cypher manual, String
    // functions), NOT core openCypher: the TCK `String8`/`String9`
    // features specify `STARTS WITH`/`ENDS WITH`, not `left`/`right`.
    FunctionSig { name: "toUpper",   arity: Arity::Fixed(1),        arg_kinds: &[ArgKind::Any],                          return_type_for: ret_string },
    FunctionSig { name: "toLower",   arity: Arity::Fixed(1),        arg_kinds: &[ArgKind::Any],                          return_type_for: ret_string },
    FunctionSig { name: "trim",      arity: Arity::Fixed(1),        arg_kinds: &[ArgKind::Any],                          return_type_for: ret_string },
    FunctionSig { name: "lTrim",     arity: Arity::Fixed(1),        arg_kinds: &[ArgKind::Any],                          return_type_for: ret_string },
    FunctionSig { name: "rTrim",     arity: Arity::Fixed(1),        arg_kinds: &[ArgKind::Any],                          return_type_for: ret_string },
    FunctionSig { name: "substring", arity: Arity::Variadic { min: 2 }, arg_kinds: &[ArgKind::Any, ArgKind::Any, ArgKind::Any], return_type_for: ret_string },
    FunctionSig { name: "replace",   arity: Arity::Fixed(3),        arg_kinds: &[ArgKind::Any, ArgKind::Any, ArgKind::Any], return_type_for: ret_string },
    FunctionSig { name: "split",     arity: Arity::Fixed(2),        arg_kinds: &[ArgKind::Any, ArgKind::Any],            return_type_for: ret_list_of_string },
    FunctionSig { name: "left",      arity: Arity::Fixed(2),        arg_kinds: &[ArgKind::Any, ArgKind::Any],            return_type_for: ret_string },
    FunctionSig { name: "right",     arity: Arity::Fixed(2),        arg_kinds: &[ArgKind::Any, ArgKind::Any],            return_type_for: ret_string },
    FunctionSig { name: "reverse",   arity: Arity::Fixed(1),        arg_kinds: &[ArgKind::Any],                          return_type_for: ret_first_arg_or_null },

    // Math (TCK expressions/mathematical/*). `abs` preserves the
    // numeric type (Integer -> Integer, Float -> Float); `sign`
    // returns Integer; the transcendental / rounding family returns
    // Float. `e`/`pi` are nullary constants. `Any` arg-kind — the eval
    // arm enforces "numeric or NULL" at runtime (see the block comment
    // above on FALSE-POSITIVE rejection of dynamically-typed args).
    FunctionSig { name: "abs",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_first_arg_or_null },
    FunctionSig { name: "ceil",  arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "floor", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "round", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "sign",  arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_integer },
    FunctionSig { name: "sqrt",  arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "exp",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "log",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "log10", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "sin",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "cos",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "tan",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "e",     arity: Arity::Fixed(0), arg_kinds: &[],             return_type_for: ret_float },
    FunctionSig { name: "pi",    arity: Arity::Fixed(0), arg_kinds: &[],             return_type_for: ret_float },
    // `rand()` — nullary uniform `[0,1)` generator (openCypher v9 §3
    // scalar). Same Fixed(0) → Float shape as the `e`/`pi` constants;
    // the eval arm (`executor::eval`) draws the value. Non-deterministic
    // (no VALUE oracle), registered for the `Quantifier9`..`Quantifier12`
    // random-INDEPENDENT invariant scenarios (#618) — see the module doc.
    FunctionSig { name: "rand",  arity: Arity::Fixed(0), arg_kinds: &[],             return_type_for: ret_float },

    // Type conversion (TCK expressions/typeConversion/*). Each accepts
    // `Any` and returns its target type (or NULL at runtime for
    // un-parseable strings, per the TCK null-return scenarios).
    FunctionSig { name: "toInteger", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_integer },
    FunctionSig { name: "toFloat",   arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "toBoolean", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_boolean },
    FunctionSig { name: "toString",  arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_string },

    // Scalar / list (TCK expressions/{typeConversion,list}/*).
    // `coalesce` returns the first non-NULL argument's type (approx
    // first-arg); `range` builds an integer list.
    FunctionSig { name: "coalesce", arity: Arity::Variadic { min: 1 }, arg_kinds: &[ArgKind::Any],                          return_type_for: ret_first_arg_or_null },
    FunctionSig { name: "range",    arity: Arity::Variadic { min: 2 }, arg_kinds: &[ArgKind::Any, ArgKind::Any, ArgKind::Any], return_type_for: ret_list_of_integer },

    // ArcGraph extensions (D-5, D-6 — also expressible as operators)
    FunctionSig { name: "vector_distance", arity: Arity::Fixed(2), arg_kinds: &[ArgKind::Any, ArgKind::Any], return_type_for: ret_float },
    FunctionSig { name: "text_match",      arity: Arity::Fixed(2), arg_kinds: &[ArgKind::Any, ArgKind::Any], return_type_for: ret_boolean },

    // ArcGraph extensions (D-4 — community membership)
    FunctionSig { name: "community", arity: Arity::Fixed(1), arg_kinds: &[ArgKind::Any], return_type_for: ret_integer },
];

/// Look up a function signature by name. Linear scan over [`BUILTINS`].
///
/// **Case-insensitive** (#618): openCypher functions are case-insensitive
/// (`range`/`RANGE`/`Range`, `toInteger`/`TOINTEGER` all denote the same
/// function — openCypher v9 §3), so the comparison case-folds via
/// `eq_ignore_ascii_case` (every builtin name is ASCII). The matched
/// signature's `name` preserves the registry's canonical spelling; only
/// the COMPARISON ignores case. Pre-#618 this was an exact match, so
/// `RANGE(1,3)` failed type-check with `unknown function RANGE` even
/// though the evaluator (which already lower-cases before dispatch) would
/// have computed it. No two builtin names differ only by case, so the
/// first (and only) case-fold match is unambiguous.
pub fn lookup(name: &str) -> Option<&'static FunctionSig> {
    BUILTINS.iter().find(|s| s.name.eq_ignore_ascii_case(name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_returns_known_function() {
        let sig = lookup("count").expect("count exists");
        assert_eq!(sig.name, "count");
        assert_eq!(sig.arity, Arity::Fixed(1));
    }

    #[test]
    fn lookup_returns_none_for_unknown() {
        assert!(lookup("not_a_real_function").is_none());
    }

    #[test]
    fn arg_kind_accepts_null_universally() {
        assert!(ArgKind::Numeric.accepts(&TypeInfo::Null));
        assert!(ArgKind::List.accepts(&TypeInfo::Null));
        assert!(ArgKind::Node.accepts(&TypeInfo::Null));
    }

    #[test]
    fn arg_kind_numeric_accepts_numeric_scalars_and_any_property() {
        use crate::semantic::bound_ast::PropertyType;
        use arcgraph_core::PropertyId;
        assert!(ArgKind::Numeric.accepts(&TypeInfo::Integer));
        assert!(ArgKind::Numeric.accepts(&TypeInfo::Float));
        // A BARE non-numeric scalar is still rejected at compile time.
        assert!(!ArgKind::Numeric.accepts(&TypeInfo::Boolean));
        assert!(!ArgKind::Numeric.accepts(&TypeInfo::String));
        // #773 dynamic-schema discipline: ANY property access is admitted
        // (runtime-enforced). The v1.0 type-checker types EVERY property
        // access as the `Property::String` sentinel (it has no per-label
        // type info), so `sum(t.amount)` / `avg(t.amount)` MUST type-check
        // here — the Customer-Zero AML `sum`-over-amount HAVING. Mirrors
        // `is_numeric` + `ArgKind::MapLike`.
        for vt in [
            PropertyType::String,
            PropertyType::Integer,
            PropertyType::Float,
        ] {
            assert!(
                ArgKind::Numeric.accepts(&TypeInfo::Property {
                    property_id: PropertyId::new(1),
                    value_type: vt,
                }),
                "Numeric must admit Property{{{vt:?}}} (dynamic-schema, runtime-enforced)"
            );
        }
    }

    #[test]
    fn return_type_collect_propagates_element() {
        let ti = ret_collect_list(&[TypeInfo::Integer]);
        assert_eq!(ti, TypeInfo::List(Box::new(TypeInfo::Integer)));
    }

    #[test]
    fn return_type_head_yields_element_type() {
        let arg = TypeInfo::List(Box::new(TypeInfo::String));
        let ti = ret_list_element(&[arg]);
        assert_eq!(ti, TypeInfo::String);
    }

    // ---- W28 conformance additions (Task #652) ----

    #[test]
    fn w28_registry_includes_all_added_builtins() {
        // Every function the W28 slice wires (registry + eval) MUST be
        // resolvable by its canonical openCypher spelling.
        for name in [
            // string
            "toUpper",
            "toLower",
            "trim",
            "lTrim",
            "rTrim",
            "substring",
            "replace",
            "split",
            "left",
            "right",
            "reverse",
            // math
            "abs",
            "ceil",
            "floor",
            "round",
            "sign",
            "sqrt",
            "exp",
            "log",
            "log10",
            "sin",
            "cos",
            "tan",
            "e",
            "pi",
            // conversion
            "toInteger",
            "toFloat",
            "toBoolean",
            "toString",
            // scalar / list
            "coalesce",
            "range",
        ] {
            assert!(
                lookup(name).is_some(),
                "builtin `{name}` must be registered"
            );
        }
    }

    #[test]
    fn w28_arity_pins() {
        // Variadic + fixed arities the type-checker enforces.
        assert_eq!(
            lookup("substring").unwrap().arity,
            Arity::Variadic { min: 2 }
        );
        assert_eq!(lookup("range").unwrap().arity, Arity::Variadic { min: 2 });
        assert_eq!(
            lookup("coalesce").unwrap().arity,
            Arity::Variadic { min: 1 }
        );
        assert_eq!(lookup("replace").unwrap().arity, Arity::Fixed(3));
        assert_eq!(lookup("split").unwrap().arity, Arity::Fixed(2));
        assert_eq!(lookup("left").unwrap().arity, Arity::Fixed(2));
        assert_eq!(lookup("abs").unwrap().arity, Arity::Fixed(1));
        assert_eq!(lookup("e").unwrap().arity, Arity::Fixed(0));
        assert_eq!(lookup("pi").unwrap().arity, Arity::Fixed(0));
    }

    #[test]
    fn w28_return_types_pin() {
        // `abs` preserves the numeric type; `sign` -> Integer; the
        // transcendental family -> Float; `range` -> List(Integer);
        // `toString` -> String.
        let abs = lookup("abs").unwrap();
        assert_eq!(
            (abs.return_type_for)(&[TypeInfo::Integer]),
            TypeInfo::Integer
        );
        assert_eq!((abs.return_type_for)(&[TypeInfo::Float]), TypeInfo::Float);
        assert_eq!(
            (lookup("sign").unwrap().return_type_for)(&[TypeInfo::Float]),
            TypeInfo::Integer
        );
        assert_eq!(
            (lookup("sqrt").unwrap().return_type_for)(&[TypeInfo::Integer]),
            TypeInfo::Float
        );
        assert_eq!(
            (lookup("range").unwrap().return_type_for)(&[TypeInfo::Integer, TypeInfo::Integer]),
            TypeInfo::List(Box::new(TypeInfo::Integer))
        );
        assert_eq!(
            (lookup("toString").unwrap().return_type_for)(&[TypeInfo::Integer]),
            TypeInfo::String
        );
        assert_eq!(
            (lookup("split").unwrap().return_type_for)(&[TypeInfo::String, TypeInfo::String]),
            TypeInfo::List(Box::new(TypeInfo::String))
        );
    }

    #[test]
    fn adr193_path_functions_registered_with_list_returns() {
        // ADR-193 D-7 — `nodes`/`relationships` are registered (arity 1,
        // Any arg) returning List(Node) / List(Relationship); `length`
        // stays Any → Integer (admits a path arg unchanged).
        let nodes = lookup("nodes").expect("nodes registered");
        assert_eq!(nodes.arity, Arity::Fixed(1));
        assert_eq!(
            (nodes.return_type_for)(&[TypeInfo::Null]),
            TypeInfo::List(Box::new(TypeInfo::Node { label: None }))
        );
        let rels = lookup("relationships").expect("relationships registered");
        assert_eq!(rels.arity, Arity::Fixed(1));
        assert_eq!(
            (rels.return_type_for)(&[TypeInfo::Null]),
            TypeInfo::List(Box::new(TypeInfo::Relationship { rel_type: None }))
        );
        assert_eq!(
            (lookup("length").unwrap().return_type_for)(&[TypeInfo::Null]),
            TypeInfo::Integer
        );
    }

    #[test]
    fn lookup_is_case_insensitive() {
        // #618 — openCypher functions are CASE-INSENSITIVE: every casing
        // of a name resolves to the SAME signature, whose `name` field
        // preserves the canonical spelling. (Pre-#618 `lookup` was
        // case-sensitive — `RANGE(1,3)` failed `unknown function RANGE`
        // even though the engine computes `range`.)
        for spelling in ["toInteger", "tointeger", "TOINTEGER", "ToInteger"] {
            let sig = lookup(spelling).unwrap_or_else(|| panic!("`{spelling}` must resolve"));
            assert_eq!(
                sig.name, "toInteger",
                "all casings of `{spelling}` resolve to the canonical name"
            );
        }
        // An all-lowercase canonical name resolves from a mixed/upper call.
        assert_eq!(lookup("RANGE").expect("RANGE resolves").name, "range");
        assert_eq!(lookup("Abs").expect("Abs resolves").name, "abs");
        assert_eq!(lookup("Range").expect("Range resolves").name, "range");
        // A camelCase canonical name resolves from an all-caps call.
        assert_eq!(lookup("TOUPPER").expect("TOUPPER resolves").name, "toUpper");
        // Case-fold does NOT admit non-builtins.
        assert!(lookup("not_a_real_function").is_none());
        assert!(lookup("NOT_A_REAL_FUNCTION").is_none());
    }

    #[test]
    fn ga618_properties_registered_returns_map() {
        // #618 — `properties(x)` is registered (arity 1, `Any` arg)
        // returning a `Map` (the property bag of a node / rel / map). The
        // eval arm owns the node/rel/map-vs-other runtime enforcement.
        let sig = lookup("properties").expect("properties registered");
        assert_eq!(sig.arity, Arity::Fixed(1));
        assert_eq!(sig.name, "properties");
        assert_eq!((sig.return_type_for)(&[TypeInfo::Null]), TypeInfo::Map);
        assert_eq!((sig.return_type_for)(&[TypeInfo::Map]), TypeInfo::Map);
        assert_eq!(
            (sig.return_type_for)(&[TypeInfo::Node { label: None }]),
            TypeInfo::Map
        );
        // Case-insensitive resolution holds for the new builtin too.
        assert_eq!(
            lookup("PROPERTIES").expect("PROPERTIES resolves").name,
            "properties"
        );
    }

    #[test]
    fn ga_rand_registered_nullary_float() {
        // GA-rand (#618) — `rand()` is registered as a nullary (`Fixed(0)`,
        // no arg-kinds) builtin returning `Float`, the same shape as the
        // `e`/`pi` constants. The eval arm draws the value; the registry
        // entry is what makes the type-checker admit `rand()` (pre-GA-rand
        // a `rand()` call failed type-check with `UnknownFunction`).
        let sig = lookup("rand").expect("rand registered");
        assert_eq!(sig.name, "rand");
        assert_eq!(sig.arity, Arity::Fixed(0));
        assert!(sig.arg_kinds.is_empty(), "rand() takes no arguments");
        assert_eq!((sig.return_type_for)(&[]), TypeInfo::Float);
        // Case-insensitive resolution holds for `rand` too.
        assert_eq!(lookup("RAND").expect("RAND resolves").name, "rand");
        assert_eq!(lookup("Rand").expect("Rand resolves").name, "rand");
    }
}
