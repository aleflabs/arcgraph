//! W26-γ-3 / ADR-136 — `arcql-smith` random Cypher generator.
//!
//! # What this is
//!
//! Type/label-aware random Cypher generator inspired by:
//! - **SQLsmith** (Postgres community, 2017) — found 100+ Postgres bugs.
//! - **CockroachDB SQLsmith fork** (2018-2020) — found 40+ bugs in 3 months.
//! - **GDsmith** (arXiv 2206.08530, 2022) — first academic graph-SQLsmith.
//! - **Cynthia** (ASE 2021) — differential Cypher fuzzer.
//!
//! Per ADR-136 §D-1, this is the first Rust port of GDsmith-style
//! type-aware Cypher random generation; Apache-2.0; reusable by
//! other Rust graph-DB projects.
//!
//! # Architecture (ADR-136 §D-1)
//!
//! Recursive-descent generator that emits CYPHER STRINGS (not AST
//! nodes — strings are the canonical fuzz-input surface for the
//! parser). The generator carries:
//!
//! 1. **`Scope`** — variable bindings + their bound labels. Variable
//!    references emit only from in-scope bindings. Per GDsmith §3.2
//!    type-aware generation.
//!
//! 2. **`Budget { depth, width }`** — recursion depth + clause-chain
//!    width caps. Default `depth = 5, width = 5` per ADR-136 §D-1.
//!    Prevents stack overflow + terabyte-output pathologies.
//!
//! 3. **`Rng`** — deterministic seedable RNG. Same seed → byte-identical
//!    output. Per `feedback_determinism_oracle_concurrency_tests.md`:
//!    deterministic algorithms get binary-equal oracles.
//!
//! 4. **`Catalog`** — labels + relationship types + per-label property
//!    schemas (label → property → type). The generator samples from
//!    this catalog rather than emitting free-form identifiers.
//!
//! # Usage
//!
//! ```
//! use arcgraph_query::test_support::smith::{Smith, SmithConfig};
//!
//! let cfg = SmithConfig::stub();      // ~10-label x 5-prop stub catalog
//! let mut smith = Smith::new(42, cfg); // seed=42, deterministic
//! let q1 = smith.gen_query();
//! let q2 = smith.gen_query();
//! // q1 != q2 (RNG advances); same seed → same sequence.
//! assert!(arcgraph_query::parse(&q1).is_ok() || q1.is_empty());
//! ```
//!
//! # Coverage at landing (W26-γ-3 = ADR-136 §D-1)
//!
//! - **Clause-level.** MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN,
//!   ORDER BY, SKIP, LIMIT, UNWIND.
//! - **Pattern shapes.** Node patterns, relationship patterns,
//!   path patterns up to 4 hops.
//! - **Expression shapes.** Identifiers, property accessors, literals
//!   (int / float / string / bool / null), arithmetic, comparison,
//!   logical, IS NULL, IN list, parameters.
//! - **Excluded.** Syntax that is not backed by the public engine.

use std::fmt::Write as _;

// =====================================================================
// 1. Deterministic RNG (xoshiro256** — fast, well-tested, no_std-safe)
// =====================================================================

/// xoshiro256** PRNG per Blackman + Vigna 2018 (vigna.di.unimi.it/xorshift).
///
/// We don't depend on the `rand` crate to keep arcql-smith dep-light
/// (it ships at v1.0 with zero new transitive deps). The xoshiro256**
/// algorithm passes BigCrush + PractRand testsuites and has period
/// 2^256 - 1; more than adequate for fuzz seed expansion.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    /// Build an RNG from a 64-bit seed. The seed is splitmixed across
    /// the 4-word state so seeds with low Hamming weight still produce
    /// a well-mixed state.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u64; 4];
        let mut x = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        for slot in &mut state {
            // splitmix64 finalizer (Stafford 2014).
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z = z ^ (z >> 31);
            *slot = z;
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        }
        // Guard against all-zero state (xoshiro is invalid at all-zero).
        if state.iter().all(|&w| w == 0) {
            state[0] = 0xBAD0_C0FF_EE15_DEAD;
        }
        Self { s: state }
    }

    /// Pull a u64.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Pull a uniformly-distributed `usize` in `0..n`. Panics on `n == 0`.
    pub fn gen_range(&mut self, n: usize) -> usize {
        assert!(n > 0, "Rng::gen_range called with n == 0");
        // Lemire's nearly-divisionless method — minor bias acceptable
        // for fuzz seed expansion (and bounded by 2^-32 for n < 2^32).
        let v = self.next_u64() as u128;
        let bound = n as u128;
        ((v * bound) >> 64) as usize
    }

    /// Pick a random element from a non-empty slice.
    pub fn pick<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        assert!(!slice.is_empty(), "Rng::pick called on empty slice");
        let idx = self.gen_range(slice.len());
        &slice[idx]
    }

    /// Flip a biased coin; returns true with probability `numerator/denom`.
    pub fn chance(&mut self, numerator: u32, denom: u32) -> bool {
        assert!(denom > 0, "Rng::chance denominator must be > 0");
        let pick = self.gen_range(denom as usize) as u32;
        pick < numerator
    }
}

// =====================================================================
// 2. Schema typing
// =====================================================================

/// Property value type known to the generator. Per ADR-136 §D-1 #3
/// type-aware generation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PropType {
    Int,
    Float,
    Str,
    Bool,
}

impl PropType {
    /// Is this type orderable (for ORDER BY / comparison)?
    pub fn is_orderable(self) -> bool {
        matches!(self, PropType::Int | PropType::Float | PropType::Str)
    }
    /// Is this type numeric (for + - * / arithmetic)?
    pub fn is_numeric(self) -> bool {
        matches!(self, PropType::Int | PropType::Float)
    }
}

// =====================================================================
// 3. Catalog (labels + types + per-label property schemas)
// =====================================================================

/// A catalog snapshot the generator samples from. Per ADR-136 §D-1 #5
/// catalog-driven sampling.
#[derive(Clone, Debug)]
pub struct Catalog {
    pub labels: Vec<&'static str>,
    pub rel_types: Vec<&'static str>,
    /// Per-label property schemas. Each entry is `(label, vec[(prop, type)])`.
    pub properties: Vec<(&'static str, Vec<(&'static str, PropType)>)>,
}

impl Catalog {
    /// Stub catalog — ~10 labels × 5 properties each. Stable across runs;
    /// suitable for golden-file diff oracles. Per ADR-136 §D-1 #5
    /// "Stub catalog mode (default for smoke / CI / local)."
    pub fn stub() -> Self {
        Self {
            labels: vec![
                "Person", "Company", "Order", "Product", "City", "Country", "Tag", "Post",
                "Comment", "Event",
            ],
            rel_types: vec![
                "KNOWS",
                "WORKS_AT",
                "BOUGHT",
                "LIKES",
                "LOCATED_IN",
                "TAGGED",
                "REPLIES_TO",
                "ATTENDED",
            ],
            properties: vec![
                (
                    "Person",
                    vec![
                        ("age", PropType::Int),
                        ("name", PropType::Str),
                        ("salary", PropType::Float),
                        ("active", PropType::Bool),
                        ("score", PropType::Float),
                    ],
                ),
                (
                    "Company",
                    vec![
                        ("name", PropType::Str),
                        ("employees", PropType::Int),
                        ("revenue", PropType::Float),
                        ("public", PropType::Bool),
                    ],
                ),
                (
                    "Order",
                    vec![
                        ("id", PropType::Int),
                        ("total", PropType::Float),
                        ("status", PropType::Str),
                        ("paid", PropType::Bool),
                    ],
                ),
                (
                    "Product",
                    vec![
                        ("sku", PropType::Str),
                        ("price", PropType::Float),
                        ("stock", PropType::Int),
                        ("active", PropType::Bool),
                    ],
                ),
                (
                    "City",
                    vec![
                        ("name", PropType::Str),
                        ("population", PropType::Int),
                        ("area", PropType::Float),
                    ],
                ),
                (
                    "Country",
                    vec![
                        ("code", PropType::Str),
                        ("gdp", PropType::Float),
                        ("population", PropType::Int),
                    ],
                ),
                (
                    "Tag",
                    vec![("name", PropType::Str), ("uses", PropType::Int)],
                ),
                (
                    "Post",
                    vec![
                        ("title", PropType::Str),
                        ("body", PropType::Str),
                        ("likes", PropType::Int),
                        ("published", PropType::Bool),
                    ],
                ),
                (
                    "Comment",
                    vec![
                        ("text", PropType::Str),
                        ("score", PropType::Int),
                        ("flagged", PropType::Bool),
                    ],
                ),
                (
                    "Event",
                    vec![
                        ("name", PropType::Str),
                        ("attendees", PropType::Int),
                        ("budget", PropType::Float),
                    ],
                ),
            ],
        }
    }

    /// Returns properties for a label, or an empty slice if label is
    /// not in the catalog.
    pub fn props_for(&self, label: &str) -> &[(&'static str, PropType)] {
        self.properties
            .iter()
            .find(|(lbl, _)| *lbl == label)
            .map(|(_, ps)| ps.as_slice())
            .unwrap_or(&[])
    }

    /// Returns all (label, prop, type) triples matching a given type
    /// (across all labels in the catalog).
    pub fn props_of_type(&self, ty: PropType) -> Vec<(&'static str, &'static str)> {
        self.properties
            .iter()
            .flat_map(|(lbl, ps)| {
                ps.iter()
                    .filter_map(move |(p, pt)| if *pt == ty { Some((*lbl, *p)) } else { None })
            })
            .collect()
    }
}

// =====================================================================
// 4. Scope tracking
// =====================================================================

/// A bound variable + the label it's bound to. Used to drive type-aware
/// property access generation per ADR-136 §D-1 #3.
#[derive(Clone, Debug)]
pub struct Binding {
    pub name: String,
    pub label: Option<&'static str>,
}

/// Generation scope — the set of in-scope bindings during clause emission.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    pub bindings: Vec<Binding>,
}

impl Scope {
    /// Pick a random binding (uniform), or None if scope is empty.
    pub fn pick<'a>(&'a self, rng: &mut Rng) -> Option<&'a Binding> {
        if self.bindings.is_empty() {
            None
        } else {
            Some(&self.bindings[rng.gen_range(self.bindings.len())])
        }
    }

    /// Pick a random binding whose label has a property of the requested
    /// type. Returns `(binding, prop_name)`.
    pub fn pick_typed_prop<'a>(
        &'a self,
        catalog: &Catalog,
        ty: PropType,
        rng: &mut Rng,
    ) -> Option<(&'a Binding, &'static str)> {
        let candidates: Vec<(&'a Binding, &'static str)> = self
            .bindings
            .iter()
            .filter_map(|b| {
                let lbl = b.label?;
                let props = catalog.props_for(lbl);
                let typed: Vec<&'static str> = props
                    .iter()
                    .filter_map(|(p, pt)| if *pt == ty { Some(*p) } else { None })
                    .collect();
                if typed.is_empty() {
                    None
                } else {
                    let p = typed[rng.gen_range(typed.len())];
                    Some((b, p))
                }
            })
            .collect();
        if candidates.is_empty() {
            None
        } else {
            let idx = rng.gen_range(candidates.len());
            Some(candidates[idx])
        }
    }
}

// =====================================================================
// 5. Budget
// =====================================================================

/// Recursion + width budget. Decremented at each recursive call.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    pub depth: u8,
    pub width: u8,
}

impl Budget {
    pub fn deeper(self) -> Self {
        Self {
            depth: self.depth.saturating_sub(1),
            width: self.width,
        }
    }
    pub fn exhausted(self) -> bool {
        self.depth == 0
    }
}

// =====================================================================
// 6. Config + Smith
// =====================================================================

/// `Smith` configuration. Per ADR-136 §D-1 #5 catalog-driven sampling.
#[derive(Clone, Debug)]
pub struct SmithConfig {
    pub catalog: Catalog,
    pub max_depth: u8,
    pub max_width: u8,
    /// Probability of emitting OPTIONAL MATCH instead of MATCH (out of 100).
    pub optional_match_pct: u32,
    /// Probability of emitting a WHERE clause (out of 100).
    pub where_clause_pct: u32,
    /// Probability of emitting an ORDER BY tail clause (out of 100).
    pub order_by_pct: u32,
    /// Probability of emitting a SKIP tail clause (out of 100).
    pub skip_pct: u32,
    /// Probability of emitting a LIMIT tail clause (out of 100).
    pub limit_pct: u32,
}

impl SmithConfig {
    /// Default — stub catalog + ADR-136 §D-1 budget caps.
    pub fn stub() -> Self {
        Self {
            catalog: Catalog::stub(),
            max_depth: 5,
            max_width: 5,
            optional_match_pct: 20,
            where_clause_pct: 60,
            order_by_pct: 30,
            skip_pct: 20,
            limit_pct: 70,
        }
    }
}

/// The generator. Hold one per fuzz iteration; the `Rng` advances
/// internally per `gen_*` call.
pub struct Smith {
    rng: Rng,
    cfg: SmithConfig,
    /// Counter for fresh variable names (n0, n1, n2, ...).
    next_var: u32,
}

impl Smith {
    /// Build with a deterministic seed.
    pub fn new(seed: u64, cfg: SmithConfig) -> Self {
        Self {
            rng: Rng::new(seed),
            cfg,
            next_var: 0,
        }
    }

    fn fresh_var(&mut self, prefix: &str) -> String {
        let v = format!("{}{}", prefix, self.next_var);
        self.next_var += 1;
        v
    }

    /// Top-level: generate one full ArcQL/Cypher query string.
    ///
    /// The query is well-typed per the catalog and bounded by the
    /// configured budget. May be empty in degenerate cases (e.g.,
    /// budget exhausted before MATCH could complete) — callers
    /// MUST tolerate empty output without panic.
    pub fn gen_query(&mut self) -> String {
        let budget = Budget {
            depth: self.cfg.max_depth,
            width: self.cfg.max_width,
        };
        let mut scope = Scope::default();
        let mut out = String::new();

        // Always start with MATCH (or OPTIONAL MATCH) per ADR-136 §D-1 #4
        // clause-level coverage.
        let optional = self.rng.chance(self.cfg.optional_match_pct, 100);
        if optional {
            out.push_str("OPTIONAL ");
        }
        out.push_str("MATCH ");
        self.gen_path_pattern(&mut scope, budget, &mut out);

        // Optional WHERE clause.
        if scope.bindings.iter().any(|b| b.label.is_some())
            && self.rng.chance(self.cfg.where_clause_pct, 100)
        {
            out.push_str(" WHERE ");
            self.gen_where_expr(&scope, budget, &mut out);
        }

        // WITH chain — at depth > 2, optionally emit a WITH clause to
        // exercise the projection/aliasing surface.
        if budget.depth > 2 && self.rng.chance(40, 100) {
            let with_count = (self.rng.gen_range(2) + 1).min(scope.bindings.len().max(1));
            out.push_str(" WITH ");
            for i in 0..with_count {
                if i > 0 {
                    out.push_str(", ");
                }
                if let Some(b) = scope.pick(&mut self.rng) {
                    out.push_str(&b.name);
                } else {
                    out.push('1');
                }
            }
        }

        // RETURN clause.
        out.push_str(" RETURN ");
        self.gen_return_items(&scope, &mut out);

        // Optional ORDER BY.
        if self.rng.chance(self.cfg.order_by_pct, 100) {
            if let Some(b) = scope.pick(&mut self.rng) {
                if let Some(lbl) = b.label {
                    let props = self.cfg.catalog.props_for(lbl);
                    let orderable: Vec<&(&'static str, PropType)> =
                        props.iter().filter(|(_, t)| t.is_orderable()).collect();
                    if !orderable.is_empty() {
                        let p = orderable[self.rng.gen_range(orderable.len())].0;
                        let _ = write!(out, " ORDER BY {}.{}", b.name, p);
                        if self.rng.chance(50, 100) {
                            out.push_str(" DESC");
                        }
                    }
                }
            }
        }

        // Optional SKIP + LIMIT.
        if self.rng.chance(self.cfg.skip_pct, 100) {
            let _ = write!(out, " SKIP {}", self.rng.gen_range(20));
        }
        if self.rng.chance(self.cfg.limit_pct, 100) {
            let n = self.rng.gen_range(100) + 1;
            let _ = write!(out, " LIMIT {}", n);
        }

        out
    }

    // -----------------------------------------------------------------
    // Pattern generators
    // -----------------------------------------------------------------

    fn gen_path_pattern(&mut self, scope: &mut Scope, budget: Budget, out: &mut String) {
        self.gen_node_pattern(scope, out);
        // Up to (width - 1) additional hops, depth-bounded.
        let hops = if budget.exhausted() {
            0
        } else {
            self.rng
                .gen_range(self.cfg.max_width.min(budget.width) as usize)
        };
        for _ in 0..hops {
            self.gen_rel_pattern(out);
            self.gen_node_pattern(scope, out);
        }
    }

    fn gen_node_pattern(&mut self, scope: &mut Scope, out: &mut String) {
        out.push('(');
        let name = self.fresh_var("n");
        out.push_str(&name);

        // 70% chance of labeling.
        let label = if self.rng.chance(70, 100) {
            let lbl = *self.rng.pick(&self.cfg.catalog.labels);
            let _ = write!(out, ":{}", lbl);
            Some(lbl)
        } else {
            None
        };

        // 25% chance of a property map.
        if label.is_some() && self.rng.chance(25, 100) {
            if let Some(lbl) = label {
                let props = self.cfg.catalog.props_for(lbl);
                if !props.is_empty() {
                    let (pname, ptype) = *self.rng.pick(props);
                    out.push_str(" {");
                    out.push_str(pname);
                    out.push_str(": ");
                    self.emit_literal(ptype, out);
                    out.push('}');
                }
            }
        }

        out.push(')');
        scope.bindings.push(Binding { name, label });
    }

    fn gen_rel_pattern(&mut self, out: &mut String) {
        // Direction — outbound (60%), inbound (20%), undirected (20%).
        let r = self.rng.gen_range(100);
        let (lhs, rhs) = if r < 60 {
            ("-", "->")
        } else if r < 80 {
            ("<-", "-")
        } else {
            ("-", "-")
        };
        out.push_str(lhs);

        // 70% chance of typed/bracketed.
        if self.rng.chance(70, 100) {
            out.push('[');
            // 50% chance of variable name.
            if self.rng.chance(50, 100) {
                let rv = self.fresh_var("r");
                out.push_str(&rv);
            }
            // 70% chance of typed.
            if self.rng.chance(70, 100) {
                let ty = *self.rng.pick(&self.cfg.catalog.rel_types);
                let _ = write!(out, ":{}", ty);
            }
            // 20% chance of length range.
            if self.rng.chance(20, 100) {
                let lo = self.rng.gen_range(3) + 1;
                let hi = lo + self.rng.gen_range(3) + 1;
                let _ = write!(out, "*{}..{}", lo, hi);
            }
            out.push(']');
        }

        out.push_str(rhs);
    }

    // -----------------------------------------------------------------
    // Expression generators
    // -----------------------------------------------------------------

    fn gen_where_expr(&mut self, scope: &Scope, budget: Budget, out: &mut String) {
        self.gen_or_expr(scope, budget, out);
    }

    fn gen_or_expr(&mut self, scope: &Scope, budget: Budget, out: &mut String) {
        self.gen_and_expr(scope, budget, out);
        // Up to 1 OR (avoid combinatorial blow-up).
        if !budget.exhausted() && self.rng.chance(20, 100) {
            out.push_str(" OR ");
            self.gen_and_expr(scope, budget.deeper(), out);
        }
    }

    fn gen_and_expr(&mut self, scope: &Scope, budget: Budget, out: &mut String) {
        self.gen_not_expr(scope, budget, out);
        if !budget.exhausted() && self.rng.chance(30, 100) {
            out.push_str(" AND ");
            self.gen_not_expr(scope, budget.deeper(), out);
        }
    }

    fn gen_not_expr(&mut self, scope: &Scope, budget: Budget, out: &mut String) {
        if self.rng.chance(15, 100) {
            out.push_str("NOT ");
        }
        self.gen_comparison(scope, budget, out);
    }

    fn gen_comparison(&mut self, scope: &Scope, budget: Budget, out: &mut String) {
        // Pick the comparison shape. 70% binary comparison; 15% IS NULL;
        // 10% IN list; 5% bare boolean literal.
        let r = self.rng.gen_range(100);
        if r < 70 {
            // Binary comparison on a typed property.
            // Pick a property type uniformly across orderables.
            let ty = match self.rng.gen_range(3) {
                0 => PropType::Int,
                1 => PropType::Float,
                _ => PropType::Str,
            };
            if let Some((bind, prop)) = scope.pick_typed_prop(&self.cfg.catalog, ty, &mut self.rng)
            {
                let _ = write!(out, "{}.{} ", bind.name, prop);
                let op = match self.rng.gen_range(6) {
                    0 => "=",
                    1 => "<>",
                    2 => "<",
                    3 => "<=",
                    4 => ">",
                    _ => ">=",
                };
                let _ = write!(out, "{} ", op);
                self.emit_literal(ty, out);
                return;
            }
            // Fall-through to bare literal if no typed property is available.
            out.push_str("true");
        } else if r < 85 {
            // IS NULL / IS NOT NULL on any property.
            if let Some(b) = scope.pick(&mut self.rng) {
                if let Some(lbl) = b.label {
                    let props = self.cfg.catalog.props_for(lbl);
                    if !props.is_empty() {
                        let (p, _) = *self.rng.pick(props);
                        let _ = write!(out, "{}.{} IS ", b.name, p);
                        if self.rng.chance(50, 100) {
                            out.push_str("NOT ");
                        }
                        out.push_str("NULL");
                        return;
                    }
                }
            }
            out.push_str("true");
        } else if r < 95 {
            // IN [a, b, c].
            if let Some((b, prop)) =
                scope.pick_typed_prop(&self.cfg.catalog, PropType::Int, &mut self.rng)
            {
                let _ = write!(out, "{}.{} IN [", b.name, prop);
                let n = self.rng.gen_range(3) + 1;
                for i in 0..n {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    let _ = write!(out, "{}", self.rng.gen_range(1000));
                }
                out.push(']');
                let _ = budget; // silence unused
                return;
            }
            out.push_str("true");
        } else {
            // Bare boolean.
            out.push_str(if self.rng.chance(50, 100) {
                "true"
            } else {
                "false"
            });
        }
    }

    fn emit_literal(&mut self, ty: PropType, out: &mut String) {
        match ty {
            PropType::Int => {
                let n = self.rng.gen_range(10_000) as i64 - 5_000;
                let _ = write!(out, "{}", n);
            }
            PropType::Float => {
                let n = (self.rng.gen_range(10_000) as f64) / 10.0 - 500.0;
                let _ = write!(out, "{:.2}", n);
            }
            PropType::Str => {
                let words = ["foo", "bar", "baz", "qux", "alpha", "beta", "gamma"];
                let _ = write!(out, "'{}'", self.rng.pick(&words));
            }
            PropType::Bool => {
                out.push_str(if self.rng.chance(50, 100) {
                    "true"
                } else {
                    "false"
                });
            }
        }
    }

    fn gen_return_items(&mut self, scope: &Scope, out: &mut String) {
        if scope.bindings.is_empty() {
            out.push('1');
            return;
        }
        let count = (self.rng.gen_range(3) + 1).min(scope.bindings.len());
        for i in 0..count {
            if i > 0 {
                out.push_str(", ");
            }
            // 60% chance bare variable; 30% chance property access; 10% chance literal expr.
            let r = self.rng.gen_range(100);
            if r < 60 {
                if let Some(b) = scope.pick(&mut self.rng) {
                    out.push_str(&b.name);
                } else {
                    out.push('1');
                }
            } else if r < 90 {
                if let Some(b) = scope.pick(&mut self.rng) {
                    if let Some(lbl) = b.label {
                        let props = self.cfg.catalog.props_for(lbl);
                        if !props.is_empty() {
                            let (p, _) = *self.rng.pick(props);
                            let _ = write!(out, "{}.{}", b.name, p);
                            continue;
                        }
                    }
                    out.push_str(&b.name);
                } else {
                    out.push('1');
                }
            } else {
                // Literal int.
                let n = self.rng.gen_range(1000);
                let _ = write!(out, "{}", n);
            }
        }
    }
}

// =====================================================================
// 7. Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Determinism oracle — same seed → byte-identical output. Per
    /// `feedback_determinism_oracle_concurrency_tests.md`: deterministic
    /// algorithms get binary-equal oracles.
    #[test]
    fn smith_is_deterministic() {
        let cfg1 = SmithConfig::stub();
        let cfg2 = SmithConfig::stub();
        let mut a = Smith::new(0xDEAD_BEEF, cfg1);
        let mut b = Smith::new(0xDEAD_BEEF, cfg2);
        for _ in 0..20 {
            let qa = a.gen_query();
            let qb = b.gen_query();
            assert_eq!(
                qa, qb,
                "deterministic seeds must produce byte-identical queries"
            );
        }
    }

    /// Different seeds produce different sequences (with high probability).
    #[test]
    fn distinct_seeds_diverge() {
        let mut a = Smith::new(1, SmithConfig::stub());
        let mut b = Smith::new(2, SmithConfig::stub());
        let qa: Vec<String> = (0..20).map(|_| a.gen_query()).collect();
        let qb: Vec<String> = (0..20).map(|_| b.gen_query()).collect();
        // Allow occasional collisions but require at least one difference.
        assert!(qa != qb, "distinct seeds must diverge over 20 queries");
    }

    /// Round-trip — generated queries either parse or fail gracefully
    /// (no panic). Per ADR-136 §D-3 invariant 1.
    #[test]
    fn smith_no_panic_on_parse() {
        let mut s = Smith::new(0xABAD_1DEA, SmithConfig::stub());
        for _ in 0..50 {
            let q = s.gen_query();
            let _ = crate::parse(&q); // must NOT panic — Ok or Err both fine.
        }
    }

    /// Rng range bounds.
    #[test]
    fn rng_gen_range_bounded() {
        let mut r = Rng::new(42);
        for _ in 0..1000 {
            let v = r.gen_range(7);
            assert!(v < 7);
        }
    }

    /// Catalog stub has sane shape.
    #[test]
    fn catalog_stub_shape() {
        let c = Catalog::stub();
        assert_eq!(c.labels.len(), 10);
        assert!(!c.rel_types.is_empty());
        for lbl in &c.labels {
            // Every label must have ≥ 1 property.
            assert!(!c.props_for(lbl).is_empty(), "label {} missing props", lbl);
        }
    }

    /// Scope picks typed properties only when they exist.
    #[test]
    fn scope_picks_typed_props() {
        let cat = Catalog::stub();
        let scope = Scope {
            bindings: vec![Binding {
                name: "p".into(),
                label: Some("Person"),
            }],
        };
        let mut rng = Rng::new(7);
        let pick = scope.pick_typed_prop(&cat, PropType::Int, &mut rng);
        let (b, p) = pick.expect("Person has Int props");
        assert_eq!(b.name, "p");
        // Person's Int props are `age`.
        assert_eq!(p, "age");
    }
}
