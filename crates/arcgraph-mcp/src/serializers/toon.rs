//! M5-09 — TOON (Token-Oriented Object Notation) serializer.
//!
//! Implements the canonical TOON spec at <https://github.com/toon-format/spec>
//! (commit `main`, fetched 2026-05-10) targeted at the design-v2 §9.3
//! "uniform tabular rows" use-case for `graph.explore` / `graph.search`
//! result sets.
//!
//! ## Why TOON
//!
//! Per design-v2 §9.3, TOON delivers 40-60% fewer tokens than JSON on
//! uniform-shape arrays (the dominant `graph.explore` shape — N rows of
//! the same property set). The savings come from emitting the field
//! header once (`{id,name,...}`) instead of repeating quoted key names
//! per row. The bench at `benches/serializers_toon.rs` pins the
//! ≥30% acceptance bar from the spawn prompt against a 100-row LDBC
//! SNB Person dataset.
//!
//! All performance figures (token-savings ratios, encode latency)
//! ultimately come from `benches/serializers_toon.rs`. Absolute encode
//! latency in microseconds is hardware-, build-mode-, and contention-
//! sensitive; cite numbers ONLY from a fresh `cargo bench -p arcgraph-mcp
//! --bench serializers_toon` run on the publishing host, and pair them
//! with hardware + load context (e.g., "M3 Pro, isolated, --quick").
//! The token-savings ratio is the only load-bearing claim and reproduces
//! deterministically since it is a pure function of the encoder output.
//!
//! ## Encoding strategy
//!
//! The public API is `T: Serialize` → TOON text. Internally the encoder
//! pivots through `serde_json::Value` for two reasons:
//!   1. `Value` already represents JSON's primitive lattice
//!      (null/bool/number/string/array/object), which is exactly the
//!      lattice TOON normalizes onto per spec §3.
//!   2. The roundtrip oracle (`encode → decode → Value::eq`) is the same
//!      shape regardless of caller-provided `T`, so the proptest at
//!      `tests/toon_proptest.rs` exercises the full structural surface
//!      without needing a per-`T` strategy.
//!
//! The encoder emits canonical TOON per spec §2:
//!   - Numbers normalized: no exponent for finite floats, no leading
//!     zeros, integer-valued floats emitted as integers, `-0` → `0`,
//!     non-finite (`NaN`, ±Inf) coerced to `null`.
//!   - Strings quoted only when required by spec §7.2 (empty,
//!     leading/trailing whitespace, structural chars, control chars,
//!     numeric-looking, equals one of `true`/`false`/`null`, equals or
//!     starts with `-`).
//!   - Arrays rendered in three forms by uniformity / element-type
//!     analysis:
//!       * **Tabular** (`key[N]{f1,f2,...}:` then N comma-rows) — used
//!         when every element is an object with the same primitive-only
//!         field set. This is the token-savings path.
//!       * **Inline** (`key[N]: v1,v2,...`) — used when every element is
//!         a primitive (no body lines, compact for scalar lists).
//!       * **Block list** (`key[N]:` then `- ...` items) — used for
//!         heterogeneous arrays, nested-container element values, or
//!         any array we encode at a list-item-first position (see
//!         §"List-item position" below).
//!   - Objects rendered as indented `key: value` lines (2-space
//!     indent — spec §2.3 default).
//!
//! ## List-item position (spec §10 special case)
//!
//! Spec §10 describes a depth+2 rule for tabular arrays appearing as
//! the **first field** of a list-item object (and analogously for the
//! list-item being a tabular array directly). To keep the parser
//! uniform — body items always at header_depth+1 — this encoder skips
//! tabular form whenever the array is the first field of a list-item
//! object or the list-item itself, falling back to block-list form.
//! The cost is some tabular-token-savings on deeply-nested structures;
//! the benefit is a simpler, easier-to-audit parser surface (under the
//! W11 LOC budget) and zero §10-special-case bugs. The bench shape
//! (top-level uniform array) is unaffected since it is at the regular
//! object-field position.
//!
//! ## Quoted keys
//!
//! Keys outside the `^[A-Za-z_][A-Za-z0-9_.]*$` unquoted-key syntax
//! (spec §7.3) are NOT supported in this slice — encoder returns
//! `ToonError::Unencodable` rather than risking a non-conforming
//! emission. Future slices may add quoted-key support; the wire-level
//! format is stable so adding it is non-breaking. The proptest's key
//! strategy intentionally generates only valid-unquoted identifiers.
//!
//! ## Decoding strategy
//!
//! The decoder is a hand-rolled line-based recursive-descent parser
//! (TOON is indentation-sensitive in the YAML sense; pest grammars
//! handle significant whitespace poorly). It tokenizes the input into
//! `(depth, content)` pairs in `tokenize_lines`, then walks them in
//! `Parser` via three mutually-recursive entry points (`parse_object`,
//! `parse_array_body`, `parse_list_item`). Strict-mode errors per
//! spec §14 are surfaced as `ToonError` variants with line-number
//! context.

use std::fmt::Write;

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};

use super::error::ToonError;

/// Indentation in spaces per depth level. Spec §2 default; not exposed
/// as a parameter in this slice (encoder + decoder pin to 2).
const INDENT: usize = 2;

// ─────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────

/// Encode a Serde-compatible `value` as canonical TOON.
///
/// Pivots `value` through `serde_json::Value` (see module docs). The
/// returned string has no trailing newline (spec §2.3), uses 2-space
/// indentation, and uses comma as both document and active-array
/// delimiter.
///
/// # Errors
///
/// - `ToonError::SerdePivot` if `value`'s `Serialize` impl fails.
/// - `ToonError::Unencodable` if a string contains a control character
///   outside `{\n, \r, \t}` (spec §7.1 only defines five escape
///   sequences, so other control chars would be unrepresentable
///   losslessly).
/// - `ToonError::Unencodable` if a map key isn't a valid unquoted
///   identifier (see §"Quoted keys" in the module docs).
pub fn to_toon<T: Serialize>(value: &T) -> Result<String, ToonError> {
    let v = serde_json::to_value(value)?;
    let mut out = String::new();
    encode_root(&v, &mut out)?;
    if out.ends_with('\n') {
        out.pop();
    }
    Ok(out)
}

/// Decode a canonical TOON document into `T`.
///
/// Strict mode: rejects tab indentation, non-multiple-of-2 indents,
/// unknown escape sequences, count mismatches, and tabular-row width
/// mismatches per spec §14. Tab/pipe array delimiters are not
/// supported in this slice — encoder emits comma only and decoder
/// rejects non-comma delimiter declarations.
///
/// # Errors
///
/// Returns `ToonError` per the spec §14 enumeration. The
/// `ToonError::DecodeTarget` arm wraps a `serde_json::Error` from the
/// final `from_value::<T>` step (i.e., the TOON parsed cleanly but
/// the resulting JSON shape didn't match `T`).
pub fn from_toon<T: DeserializeOwned>(s: &str) -> Result<T, ToonError> {
    let v = parse_toon(s)?;
    serde_json::from_value(v).map_err(ToonError::DecodeTarget)
}

// ─────────────────────────────────────────────────────────────────────
// Encoder
// ─────────────────────────────────────────────────────────────────────

fn encode_root(v: &Value, out: &mut String) -> Result<(), ToonError> {
    match v {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            encode_scalar(v, out)?;
            out.push('\n');
        }
        Value::Array(arr) => {
            // Root array: no leading key. Spec §5 root form rule (1):
            // valid array header at depth-0 means root array.
            emit_array_with_optional_key(None, arr, 0, /*list_item_inhibit=*/ false, out)?;
        }
        Value::Object(obj) => {
            // Spec §5 root form: if every line is kv, document is an
            // object. Empty object → empty document.
            if !obj.is_empty() {
                emit_object_fields(obj, 0, out)?;
            }
        }
    }
    Ok(())
}

fn emit_object_fields(
    obj: &Map<String, Value>,
    depth: usize,
    out: &mut String,
) -> Result<(), ToonError> {
    for (k, v) in obj {
        emit_field(k, v, depth, out)?;
    }
    Ok(())
}

fn emit_field(key: &str, value: &Value, depth: usize, out: &mut String) -> Result<(), ToonError> {
    push_indent(out, depth);
    push_unquoted_key(out, key)?;
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            out.push_str(": ");
            encode_scalar(value, out)?;
            out.push('\n');
        }
        Value::Object(inner) => {
            // "key:" followed by either nothing (empty object) or the
            // nested fields at depth+1. The decoder distinguishes by
            // peeking at the next line's depth.
            out.push_str(":\n");
            if !inner.is_empty() {
                emit_object_fields(inner, depth + 1, out)?;
            }
        }
        Value::Array(arr) => {
            // The key has been written without a trailing colon; the
            // array emitter writes "[N]...:" plus body.
            emit_array_after_key(arr, depth, /*list_item_inhibit=*/ false, out)?;
        }
    }
    Ok(())
}

/// Emit an array with an optional leading key (root-array case has
/// no key, object-field case does). `depth` is the depth of the
/// header LINE.
fn emit_array_with_optional_key(
    key: Option<&str>,
    arr: &[Value],
    depth: usize,
    list_item_inhibit: bool,
    out: &mut String,
) -> Result<(), ToonError> {
    push_indent(out, depth);
    if let Some(k) = key {
        push_unquoted_key(out, k)?;
    }
    emit_array_after_key(arr, depth, list_item_inhibit, out)
}

/// Emit `[N]...:<body>` after the (already-written) key. `depth` is the
/// depth of the header line. `list_item_inhibit=true` suppresses
/// tabular form to avoid the spec §10 depth+2 special case (see
/// module docs §"List-item position").
fn emit_array_after_key(
    arr: &[Value],
    depth: usize,
    list_item_inhibit: bool,
    out: &mut String,
) -> Result<(), ToonError> {
    write!(out, "[{}]", arr.len()).expect("write to String is infallible");

    if arr.is_empty() {
        out.push_str(":\n");
        return Ok(());
    }

    // Tabular form: only allowed at non-list-item-inhibited positions.
    if !list_item_inhibit {
        if let Some(fields) = detect_uniform_tabular(arr) {
            out.push('{');
            for (i, f) in fields.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                push_unquoted_key(out, f)?;
            }
            out.push_str("}:\n");
            for row in arr {
                push_indent(out, depth + 1);
                let obj = row
                    .as_object()
                    .expect("detect_uniform_tabular ensures object");
                for (i, f) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    encode_cell(obj.get(f).expect("uniform fields"), out)?;
                }
                out.push('\n');
            }
            return Ok(());
        }
    }

    // Inline form: every element is primitive, no body lines.
    if arr.iter().all(is_primitive) {
        out.push_str(": ");
        for (i, v) in arr.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            encode_cell(v, out)?;
        }
        out.push('\n');
        return Ok(());
    }

    // Block list form.
    out.push_str(":\n");
    for item in arr {
        emit_list_item(item, depth + 1, out)?;
    }
    Ok(())
}

fn emit_list_item(item: &Value, depth: usize, out: &mut String) -> Result<(), ToonError> {
    push_indent(out, depth);
    out.push('-');
    match item {
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
            out.push(' ');
            encode_scalar(item, out)?;
            out.push('\n');
        }
        Value::Array(arr) => {
            // List-item is itself an array. To avoid spec §10 depth+2
            // for tabular-as-list-item-direct, suppress tabular form.
            out.push(' ');
            emit_array_after_key(arr, depth, /*list_item_inhibit=*/ true, out)?;
        }
        Value::Object(obj) => {
            if obj.is_empty() {
                // Spec §10: empty list-item object emitted as bare "-".
                out.push('\n');
                return Ok(());
            }
            // First field stays on the dash line; subsequent fields go
            // at depth+1 from the dash line per spec §10 standard form.
            //
            // Disambiguation rule for the first field's nested-object
            // value: the dash-line consumes one indent level visually,
            // so the first field's KEY is logically at depth+1 (= where
            // siblings live). Its nested children therefore go at
            // depth+2 — one deeper than siblings — so the parser can
            // tell `- key:\n      nested: 1\n    sibling: 2` apart from
            // `- key:\n    sibling: 2` (where `sibling` is at depth+1
            // and means "first field is empty + sibling field follows").
            //
            // The proptest at `tests/toon_proptest.rs` regressed on
            // exactly this ambiguity (`{"_": {}, "a": null}` shrank to
            // `[{"_": {"a": null}}]` on decode) before we adopted the
            // depth+2 convention.
            let mut first = true;
            for (k, v) in obj {
                if first {
                    first = false;
                    out.push(' ');
                    push_unquoted_key(out, k)?;
                    match v {
                        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_) => {
                            out.push_str(": ");
                            encode_scalar(v, out)?;
                            out.push('\n');
                        }
                        Value::Object(inner) => {
                            out.push_str(":\n");
                            if !inner.is_empty() {
                                emit_object_fields(inner, depth + 2, out)?;
                            }
                        }
                        Value::Array(inner) => {
                            // First field is array — suppress tabular form
                            // to avoid §10 depth+2 special case for rows.
                            // Block-list bodies still go at depth+1; the
                            // `- ` prefix prevents ambiguity with siblings.
                            emit_array_after_key(
                                inner, depth, /*list_item_inhibit=*/ true, out,
                            )?;
                        }
                    }
                } else {
                    emit_field(k, v, depth + 1, out)?;
                }
            }
        }
    }
    Ok(())
}

/// Write a JSON scalar as a TOON scalar token (used in `key: value`
/// position; assumes the caller has already emitted the leading
/// indent / dash / `:` separator).
fn encode_scalar(v: &Value, out: &mut String) -> Result<(), ToonError> {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => encode_number(n, out),
        Value::String(s) => encode_string(s, out)?,
        _ => unreachable!("encode_scalar called with composite value"),
    }
    Ok(())
}

/// Cells (tabular rows + inline-array values) follow the same scalar
/// encoding as `encode_scalar`. Kept as a thin alias so the bench can
/// instrument cell-vs-scalar paths separately if needed later.
fn encode_cell(v: &Value, out: &mut String) -> Result<(), ToonError> {
    encode_scalar(v, out)
}

fn encode_number(n: &serde_json::Number, out: &mut String) {
    // Spec §2 canonical form:
    //   - No exponent for finite values that can be represented as decimal.
    //   - No leading zeros (the encoder cannot produce them — `n` is a
    //     well-formed Number).
    //   - Trailing-zero fraction trimmed; integer-valued floats emit as
    //     integers.
    //   - `-0.0` collapsed to `0`.
    //   - NaN / ±Inf coerced to `null`.
    if let Some(i) = n.as_i64() {
        let _ = write!(out, "{i}");
        return;
    }
    if let Some(u) = n.as_u64() {
        let _ = write!(out, "{u}");
        return;
    }
    if let Some(f) = n.as_f64() {
        if !f.is_finite() {
            out.push_str("null");
            return;
        }
        if f == 0.0 {
            // Covers +0.0 and -0.0.
            out.push('0');
            return;
        }
        if f.fract() == 0.0 && f.abs() < 1e16 {
            let as_int = f as i64;
            let _ = write!(out, "{as_int}");
            return;
        }
        // Fall through: Rust's `Display` for f64 is shortest-roundtrip
        // decimal for typical magnitudes; spec §4 accepts both decimal
        // and exponent forms on decode, so `f64::to_string` output is
        // always parseable back. We do not aggressively avoid exponent
        // for sub-normal magnitudes (rare in graph workloads); strict
        // canonical-form tightening is reserved for a follow-up.
        let _ = write!(out, "{f}");
        return;
    }
    out.push_str("null");
}

/// Encode a string per spec §7. Quotes when required, escapes the five
/// allowed escape sequences, errors on un-escapable control chars.
fn encode_string(s: &str, out: &mut String) -> Result<(), ToonError> {
    if !must_quote(s) {
        out.push_str(s);
        return Ok(());
    }
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {
                return Err(ToonError::Unencodable(format!(
                    "control char U+{:04X} not in TOON escape set",
                    c as u32
                )));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    Ok(())
}

/// Spec §7.2 must-quote enumeration. Active delimiter is comma in this
/// implementation; document delimiter is also comma. The two rule-sets
/// collapse so we never need to distinguish them.
fn must_quote(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    let first = s
        .chars()
        .next()
        .expect("non-empty: empty-string guarded above");
    let last = s
        .chars()
        .next_back()
        .expect("non-empty: empty-string guarded above");
    if first.is_whitespace() || last.is_whitespace() {
        return true;
    }
    if matches!(s, "true" | "false" | "null") {
        return true;
    }
    if looks_numeric(s) {
        return true;
    }
    if s == "-" {
        return true;
    }
    if first == '-' {
        return true;
    }
    // Numeric-with-leading-zero per spec §7.2: matches /^0\d+$/
    if first == '0' && s.len() > 1 && s.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) {
        return true;
    }
    s.chars().any(|c| {
        c == ':'
            || c == '"'
            || c == '\\'
            || c == '['
            || c == ']'
            || c == '{'
            || c == '}'
            || c == ','
            || c.is_control()
    })
}

/// Spec §7.2 numeric regex `/^-?\d+(?:\.\d+)?(?:e[+-]?\d+)?$/i`,
/// ALSO disallowing multi-digit leading zero (so `"05"` is NOT
/// numeric-looking, matching spec §4 decoder rule that forbids leading
/// zeros).
fn looks_numeric(s: &str) -> bool {
    let bytes = s.as_bytes();
    let n = bytes.len();
    if n == 0 {
        return false;
    }
    let mut i = 0;
    if bytes[0] == b'-' {
        i = 1;
        if i >= n {
            return false;
        }
    }
    let int_start = i;
    let first = bytes[i];
    if !first.is_ascii_digit() {
        return false;
    }
    i += 1;
    if first == b'0' {
        // "0" alone, or followed by '.' / 'e'. "05" rejected.
        if i < n && bytes[i].is_ascii_digit() {
            return false;
        }
    } else {
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
    }
    if i == int_start {
        return false;
    }
    if i < n && bytes[i] == b'.' {
        i += 1;
        let frac_start = i;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == frac_start {
            return false;
        }
    }
    if i < n && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < n && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < n && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            return false;
        }
    }
    i == n
}

/// Spec §7.3 unquoted-key regex `^[A-Za-z_][A-Za-z0-9_.]*$`.
fn is_valid_unquoted_key(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().expect("non-empty: empty-string guarded above");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.')
}

fn push_unquoted_key(out: &mut String, key: &str) -> Result<(), ToonError> {
    if !is_valid_unquoted_key(key) {
        return Err(ToonError::Unencodable(format!(
            "key {key:?} is not a valid unquoted identifier; \
             quoted keys are not supported in this slice"
        )));
    }
    out.push_str(key);
    Ok(())
}

fn push_indent(out: &mut String, depth: usize) {
    for _ in 0..(depth * INDENT) {
        out.push(' ');
    }
}

fn is_primitive(v: &Value) -> bool {
    matches!(
        v,
        Value::Null | Value::Bool(_) | Value::Number(_) | Value::String(_)
    )
}

/// Detect a tabular-eligible array: every element is an object, every
/// object has identical (key-sorted, since `serde_json::Map` is a
/// `BTreeMap` by default) field set, and every value is a primitive.
/// Returns the canonical field ordering on success.
fn detect_uniform_tabular(arr: &[Value]) -> Option<Vec<String>> {
    if arr.is_empty() {
        return None;
    }
    let first = arr.first()?.as_object()?;
    if first.is_empty() {
        return None;
    }
    let fields: Vec<String> = first.keys().cloned().collect();
    if fields.iter().any(|k| !is_valid_unquoted_key(k)) {
        return None;
    }
    if !first.values().all(is_primitive) {
        return None;
    }
    for v in &arr[1..] {
        let obj = v.as_object()?;
        if obj.len() != fields.len() {
            return None;
        }
        for f in &fields {
            match obj.get(f) {
                Some(val) if is_primitive(val) => {}
                _ => return None,
            }
        }
    }
    Some(fields)
}

// ─────────────────────────────────────────────────────────────────────
// Decoder
// ─────────────────────────────────────────────────────────────────────

/// Internal entry: parse text into `Value` (caller does the
/// `from_value::<T>` step).
fn parse_toon(s: &str) -> Result<Value, ToonError> {
    let lines = tokenize_lines(s)?;
    let mut p = Parser {
        lines: &lines,
        pos: 0,
    };
    let v = p.parse_root()?;
    if p.pos != p.lines.len() {
        return Err(ToonError::Malformed {
            line: p.lines[p.pos].no,
            reason: "unexpected trailing content".into(),
        });
    }
    Ok(v)
}

#[derive(Debug)]
struct PhysicalLine<'a> {
    no: usize,
    depth: usize,
    content: &'a str,
}

fn tokenize_lines(s: &str) -> Result<Vec<PhysicalLine<'_>>, ToonError> {
    let mut out = Vec::new();
    for (i, raw) in s.split('\n').enumerate() {
        let line_no = i + 1;
        // Tolerate CRLF.
        let line = raw.strip_suffix('\r').unwrap_or(raw);
        if line.is_empty() {
            continue;
        }
        // Spec §14.3 strict mode rejects tabs in indentation. We scan the
        // leading run of whitespace and reject any tab.
        let bytes = line.as_bytes();
        let mut indent = 0;
        for &b in bytes {
            match b {
                b' ' => indent += 1,
                b'\t' => {
                    return Err(ToonError::Indent {
                        line: line_no,
                        indent_size: INDENT,
                    });
                }
                _ => break,
            }
        }
        // Spec §14.3 indent must be multiple of indentSize.
        if indent % INDENT != 0 {
            return Err(ToonError::Indent {
                line: line_no,
                indent_size: INDENT,
            });
        }
        let depth = indent / INDENT;
        let content = &line[indent..];
        if content.is_empty() {
            // Whitespace-only line — skip (treated like blank).
            continue;
        }
        out.push(PhysicalLine {
            no: line_no,
            depth,
            content,
        });
    }
    Ok(out)
}

struct Parser<'a> {
    lines: &'a [PhysicalLine<'a>],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&PhysicalLine<'a>> {
        self.lines.get(self.pos)
    }

    fn parse_root(&mut self) -> Result<Value, ToonError> {
        // Spec §5 root-form determination.
        if self.lines.is_empty() {
            return Ok(Value::Object(Map::new()));
        }
        let first = &self.lines[0];
        if first.depth != 0 {
            return Err(ToonError::Indent {
                line: first.no,
                indent_size: INDENT,
            });
        }
        // (1) Array header. Without a key it's a root array; with a
        // key it's an object's first field — fall through to (3).
        if let Some(header) = ArrayHeader::parse(first.content, first.no, false)? {
            if header.key.is_none() {
                self.pos += 1;
                return self.parse_array_body(&header, 0);
            }
            return self.parse_object(0);
        }
        // (2) kv line → root object.
        if looks_like_kv(first.content)? {
            return self.parse_object(0);
        }
        // (3) single bare scalar line → root primitive.
        if self.lines.len() == 1 {
            self.pos += 1;
            return parse_scalar_token(first.content, first.no);
        }
        Err(ToonError::Malformed {
            line: first.no,
            reason: "multiple root primitives".into(),
        })
    }

    fn parse_object(&mut self, depth: usize) -> Result<Value, ToonError> {
        let mut map = Map::new();
        while let Some(line) = self.peek() {
            if line.depth < depth {
                break;
            }
            if line.depth > depth {
                return Err(ToonError::Indent {
                    line: line.no,
                    indent_size: INDENT,
                });
            }
            // List items can't start an object field line.
            if is_list_item_line(line.content) {
                return Err(ToonError::Malformed {
                    line: line.no,
                    reason: "unexpected list-item line at object position".into(),
                });
            }
            // Try array header (with key required at object position).
            if let Some(header) = ArrayHeader::parse(line.content, line.no, true)? {
                let key = header.key.clone().expect("with_key=true ensures Some");
                let header_depth = line.depth;
                self.pos += 1;
                let arr = self.parse_array_body(&header, header_depth)?;
                map.insert(key, arr);
                continue;
            }
            // Try kv split.
            match parse_kv_token(line.content, line.no)? {
                Some((key, rest)) => {
                    let line_no = line.no;
                    self.pos += 1;
                    if rest.is_empty() {
                        // "key:" - empty object or nested object based on depth-peek.
                        let inner = if matches!(self.peek(), Some(l) if l.depth == depth + 1) {
                            self.parse_object(depth + 1)?
                        } else {
                            Value::Object(Map::new())
                        };
                        map.insert(key, inner);
                    } else {
                        let val = parse_scalar_token(rest, line_no)?;
                        map.insert(key, val);
                    }
                }
                None => {
                    return Err(ToonError::Malformed {
                        line: line.no,
                        reason: "unrecognized object-field line".into(),
                    });
                }
            }
        }
        Ok(Value::Object(map))
    }

    fn parse_array_body(
        &mut self,
        header: &ArrayHeader,
        header_depth: usize,
    ) -> Result<Value, ToonError> {
        // Inline values present on the header line — values follow ":
        // " on the header, no body lines.
        if let Some(inline) = &header.inline_values {
            let cells = split_cells(inline, header.line)?;
            check_count_eq(cells.len(), header.count, header.line)?;
            let arr: Result<Vec<Value>, ToonError> = cells
                .into_iter()
                .map(|c| parse_scalar_token(c, header.line))
                .collect();
            return Ok(Value::Array(arr?));
        }

        let body_depth = header_depth + 1;

        // Tabular form.
        if let Some(fields) = &header.fields {
            let mut arr = Vec::with_capacity(header.count);
            for _ in 0..header.count {
                let line = self.peek().ok_or(ToonError::CountMismatch {
                    line: header.line,
                    declared: header.count,
                    observed: arr.len(),
                })?;
                if line.depth != body_depth {
                    return Err(ToonError::CountMismatch {
                        line: header.line,
                        declared: header.count,
                        observed: arr.len(),
                    });
                }
                let cells = split_cells(line.content, line.no)?;
                if cells.len() != fields.len() {
                    return Err(ToonError::RowWidthMismatch {
                        line: line.no,
                        header: fields.len(),
                        row: cells.len(),
                    });
                }
                let mut row = Map::new();
                for (f, c) in fields.iter().zip(cells.iter()) {
                    row.insert(f.clone(), parse_scalar_token(c, line.no)?);
                }
                arr.push(Value::Object(row));
                self.pos += 1;
            }
            return Ok(Value::Array(arr));
        }

        // Block list form.
        if header.count == 0 {
            return Ok(Value::Array(Vec::new()));
        }
        let mut arr = Vec::with_capacity(header.count);
        for _ in 0..header.count {
            let line = self.peek().ok_or(ToonError::CountMismatch {
                line: header.line,
                declared: header.count,
                observed: arr.len(),
            })?;
            if line.depth != body_depth {
                return Err(ToonError::CountMismatch {
                    line: header.line,
                    declared: header.count,
                    observed: arr.len(),
                });
            }
            if !is_list_item_line(line.content) {
                return Err(ToonError::Malformed {
                    line: line.no,
                    reason: "expected list-item line ('-' or '- ...') in block array".into(),
                });
            }
            let item = self.parse_list_item(body_depth)?;
            arr.push(item);
        }
        Ok(Value::Array(arr))
    }

    fn parse_list_item(&mut self, depth: usize) -> Result<Value, ToonError> {
        let line = &self.lines[self.pos];
        let line_no = line.no;
        let content = line.content;
        // Strip "- " or accept bare "-".
        let after_dash = if content == "-" {
            ""
        } else if let Some(rest) = content.strip_prefix("- ") {
            rest
        } else {
            return Err(ToonError::Malformed {
                line: line_no,
                reason: "list item must start with '- ' or be bare '-'".into(),
            });
        };
        self.pos += 1;
        if after_dash.is_empty() {
            // Spec §10 empty list-item object form.
            return Ok(Value::Object(Map::new()));
        }
        // Try array header (key optional — `- [N]:...` or `- key[N]:...`).
        if let Some(header) = ArrayHeader::parse(after_dash, line_no, false)? {
            // The header lives on the dash line; its body items are at
            // depth + 1 (this implementation's encoder skips the §10
            // depth+2 tabular special case).
            if header.key.is_some() {
                let key = header
                    .key
                    .clone()
                    .expect("header.key.is_some() checked above");
                let arr = self.parse_array_body(&header, depth)?;
                let mut map = Map::new();
                map.insert(key, arr);
                self.collect_remaining_obj_fields(&mut map, depth + 1)?;
                return Ok(Value::Object(map));
            }
            return self.parse_array_body(&header, depth);
        }
        // Try kv (object with first field on the dash line).
        if let Some((key, rest)) = parse_kv_token(after_dash, line_no)? {
            let mut map = Map::new();
            if rest.is_empty() {
                // Mirror the encoder's depth+2 convention for nested
                // objects in list-item-first position — see encoder
                // commentary in `emit_list_item`. Lines at depth+1 are
                // outer-object SIBLINGS; lines at depth+2 are nested
                // children of the first field. Anything else (or no
                // next line) means the first field's value is an empty
                // object.
                let first_val = if matches!(self.peek(), Some(l) if l.depth == depth + 2)
                    && !is_list_item_line(
                        self.peek()
                            .expect("matches!(Some(_)) above guarantees Some")
                            .content,
                    ) {
                    self.parse_object(depth + 2)?
                } else {
                    Value::Object(Map::new())
                };
                map.insert(key, first_val);
            } else {
                let val = parse_scalar_token(rest, line_no)?;
                map.insert(key, val);
            }
            self.collect_remaining_obj_fields(&mut map, depth + 1)?;
            return Ok(Value::Object(map));
        }
        // Bare scalar list-item.
        parse_scalar_token(after_dash, line_no)
    }

    /// After the first field of a list-item-object has been parsed,
    /// continue collecting subsequent fields at `depth` (= dash-line
    /// depth + 1) until we see a depth-change or another list-item.
    fn collect_remaining_obj_fields(
        &mut self,
        map: &mut Map<String, Value>,
        depth: usize,
    ) -> Result<(), ToonError> {
        while let Some(line) = self.peek() {
            if line.depth != depth {
                break;
            }
            if is_list_item_line(line.content) {
                break;
            }
            // Array header (key required).
            if let Some(header) = ArrayHeader::parse(line.content, line.no, true)? {
                let key = header.key.clone().expect("require_key=true ensures Some");
                let header_depth = line.depth;
                self.pos += 1;
                let arr = self.parse_array_body(&header, header_depth)?;
                map.insert(key, arr);
                continue;
            }
            match parse_kv_token(line.content, line.no)? {
                Some((key, rest)) => {
                    let line_no = line.no;
                    self.pos += 1;
                    if rest.is_empty() {
                        let inner = if matches!(self.peek(), Some(l) if l.depth == depth + 1)
                            && !is_list_item_line(
                                self.peek()
                                    .expect("matches!(Some(_)) above guarantees Some")
                                    .content,
                            ) {
                            self.parse_object(depth + 1)?
                        } else {
                            Value::Object(Map::new())
                        };
                        map.insert(key, inner);
                    } else {
                        let val = parse_scalar_token(rest, line_no)?;
                        map.insert(key, val);
                    }
                }
                None => {
                    return Err(ToonError::Malformed {
                        line: line.no,
                        reason: "unrecognized line in list-item-object continuation".into(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ArrayHeader {
    line: usize,
    key: Option<String>,
    count: usize,
    fields: Option<Vec<String>>,
    /// Values appearing inline after `: ` on the header line (the
    /// inline-array form). Mutually exclusive with body lines.
    inline_values: Option<String>,
}

impl ArrayHeader {
    /// Try to parse `content` as an array header. Returns `Ok(None)`
    /// if `content` doesn't look like one (caller falls back to other
    /// productions); returns `Err` if it looks like a header but is
    /// malformed.
    fn parse(content: &str, line_no: usize, require_key: bool) -> Result<Option<Self>, ToonError> {
        // Find first unquoted '['. The unquoted-only walk avoids
        // confusing a string like `"with [bracket]"` for a header.
        let bracket_open = match find_unquoted_byte(content, b'[') {
            Some(i) => i,
            None => return Ok(None),
        };
        // Key is content[..bracket_open].
        let key_str = &content[..bracket_open];
        let key = if key_str.is_empty() {
            None
        } else if is_valid_unquoted_key(key_str) {
            Some(key_str.to_string())
        } else {
            // Looks header-ish but key isn't a valid identifier; fall
            // back to other productions (caller will report a malformed
            // line if no other production matches).
            return Ok(None);
        };
        if require_key && key.is_none() {
            return Ok(None);
        }
        let after_open = &content[bracket_open + 1..];
        let close_rel = after_open.find(']').ok_or(ToonError::Malformed {
            line: line_no,
            reason: "missing ']' in array header".into(),
        })?;
        let count_str = &after_open[..close_rel];
        // Reject tab/pipe delimiter declarations.
        if count_str.ends_with('\t') {
            return Err(ToonError::Malformed {
                line: line_no,
                reason: "tab-delimited arrays not supported in this slice".into(),
            });
        }
        if count_str.ends_with('|') {
            return Err(ToonError::Malformed {
                line: line_no,
                reason: "pipe-delimited arrays not supported in this slice".into(),
            });
        }
        let count: usize = count_str.parse().map_err(|_| ToonError::Malformed {
            line: line_no,
            reason: format!("invalid array count {count_str:?}"),
        })?;
        let mut rest = &after_open[close_rel + 1..];
        let fields = if let Some(after_brace) = rest.strip_prefix('{') {
            let close = after_brace.find('}').ok_or(ToonError::Malformed {
                line: line_no,
                reason: "missing '}' in tabular header".into(),
            })?;
            let fields_str = &after_brace[..close];
            rest = &after_brace[close + 1..];
            // Tabular field names are comma-separated unquoted-key
            // identifiers. Empty fields rejected.
            let fields: Vec<String> = fields_str.split(',').map(|s| s.to_string()).collect();
            if fields.iter().any(|f| !is_valid_unquoted_key(f)) {
                return Err(ToonError::Malformed {
                    line: line_no,
                    reason: "tabular field name is not a valid unquoted identifier".into(),
                });
            }
            Some(fields)
        } else {
            None
        };
        // Now expect `:`, possibly followed by ` <inline values>`.
        let rest = rest.strip_prefix(':').ok_or(ToonError::Malformed {
            line: line_no,
            reason: "array header missing ':'".into(),
        })?;
        let inline_values = if rest.is_empty() {
            None
        } else if let Some(after_space) = rest.strip_prefix(' ') {
            Some(after_space.to_string())
        } else {
            return Err(ToonError::Malformed {
                line: line_no,
                reason: "expected ' ' after ':' in array header".into(),
            });
        };
        // Tabular + inline is not a spec-allowed combination.
        if fields.is_some() && inline_values.is_some() {
            return Err(ToonError::Malformed {
                line: line_no,
                reason: "tabular array header cannot have inline values".into(),
            });
        }
        Ok(Some(ArrayHeader {
            line: line_no,
            key,
            count,
            fields,
            inline_values,
        }))
    }
}

fn parse_kv_token(content: &str, line_no: usize) -> Result<Option<(String, &str)>, ToonError> {
    let colon = match find_unquoted_byte(content, b':') {
        Some(i) => i,
        None => return Ok(None),
    };
    let key_str = &content[..colon];
    if key_str.is_empty() || !is_valid_unquoted_key(key_str) {
        return Ok(None);
    }
    let after = &content[colon + 1..];
    if after.is_empty() {
        return Ok(Some((key_str.to_string(), "")));
    }
    if let Some(rest) = after.strip_prefix(' ') {
        Ok(Some((key_str.to_string(), rest)))
    } else {
        Err(ToonError::Malformed {
            line: line_no,
            reason: "expected ' ' or end-of-line after ':' in key:value line".into(),
        })
    }
}

/// Lightweight check the root parser uses to disambiguate "single bare
/// scalar at root" from "object with kv lines". Returns Err only if a
/// quoted-key probe sees an unterminated string (defensive — unreachable
/// for well-formed root scalars).
fn looks_like_kv(content: &str) -> Result<bool, ToonError> {
    let Some(colon) = find_unquoted_byte(content, b':') else {
        return Ok(false);
    };
    let key_str = &content[..colon];
    Ok(!key_str.is_empty() && is_valid_unquoted_key(key_str))
}

fn is_list_item_line(content: &str) -> bool {
    content == "-" || content.starts_with("- ")
}

fn parse_scalar_token(s: &str, line_no: usize) -> Result<Value, ToonError> {
    if s.is_empty() {
        // Strict spec rejects unquoted-empty; allow (decoder leniency)
        // and produce empty string. The encoder never emits this shape.
        return Ok(Value::String(String::new()));
    }
    if s == "null" {
        return Ok(Value::Null);
    }
    if s == "true" {
        return Ok(Value::Bool(true));
    }
    if s == "false" {
        return Ok(Value::Bool(false));
    }
    if s.starts_with('"') {
        return parse_quoted_string(s, line_no);
    }
    if looks_numeric(s) {
        return parse_number_token(s, line_no);
    }
    Ok(Value::String(s.to_string()))
}

fn parse_quoted_string(s: &str, line_no: usize) -> Result<Value, ToonError> {
    // s starts with `"`. Walk chars unescaping per §7.1; the closing
    // unescaped `"` must be the LAST char (we never expect trailing
    // content in a scalar position for this slice).
    debug_assert!(s.starts_with('"'));
    let mut chars = s.chars();
    chars.next(); // consume leading "
    let mut out = String::with_capacity(s.len());
    while let Some(c) = chars.next() {
        if c == '"' {
            // Must be EOS.
            if chars.next().is_some() {
                return Err(ToonError::InvalidString { line: line_no });
            }
            return Ok(Value::String(out));
        }
        if c == '\\' {
            match chars.next() {
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                _ => return Err(ToonError::InvalidString { line: line_no }),
            }
        } else if c.is_control() {
            // Unescaped control char inside quoted literal — reject.
            return Err(ToonError::InvalidString { line: line_no });
        } else {
            out.push(c);
        }
    }
    Err(ToonError::InvalidString { line: line_no })
}

fn parse_number_token(s: &str, line_no: usize) -> Result<Value, ToonError> {
    if !s.contains('.') && !s.contains(['e', 'E']) {
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Value::Number(i.into()));
        }
        if let Ok(u) = s.parse::<u64>() {
            return Ok(Value::Number(u.into()));
        }
    }
    if let Ok(f) = s.parse::<f64>() {
        if !f.is_finite() {
            return Err(ToonError::Malformed {
                line: line_no,
                reason: "non-finite number".into(),
            });
        }
        if let Some(n) = serde_json::Number::from_f64(f) {
            return Ok(Value::Number(n));
        }
    }
    Err(ToonError::Malformed {
        line: line_no,
        reason: format!("unparseable number {s:?}"),
    })
}

/// Split a comma-delimited cell list, respecting unquoted-only commas
/// (commas inside `"..."` are literal).
fn split_cells(s: &str, line_no: usize) -> Result<Vec<&str>, ToonError> {
    let bytes = s.as_bytes();
    let mut cells = Vec::new();
    let mut start = 0;
    let mut i = 0;
    let mut in_quote = false;
    let mut escaping = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_quote {
            if escaping {
                escaping = false;
            } else if b == b'\\' {
                escaping = true;
            } else if b == b'"' {
                in_quote = false;
            }
        } else if b == b',' {
            cells.push(&s[start..i]);
            start = i + 1;
        } else if b == b'"' {
            in_quote = true;
        }
        i += 1;
    }
    if in_quote {
        return Err(ToonError::InvalidString { line: line_no });
    }
    cells.push(&s[start..]);
    Ok(cells)
}

/// Find the first occurrence of `target` (single ASCII byte) in `s`
/// that is NOT inside a quoted string.
fn find_unquoted_byte(s: &str, target: u8) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut in_quote = false;
    let mut escaping = false;
    while i < bytes.len() {
        let b = bytes[i];
        if in_quote {
            if escaping {
                escaping = false;
            } else if b == b'\\' {
                escaping = true;
            } else if b == b'"' {
                in_quote = false;
            }
        } else if b == target {
            return Some(i);
        } else if b == b'"' {
            in_quote = true;
        }
        i += 1;
    }
    None
}

fn check_count_eq(observed: usize, declared: usize, line_no: usize) -> Result<(), ToonError> {
    if observed != declared {
        Err(ToonError::CountMismatch {
            line: line_no,
            declared,
            observed,
        })
    } else {
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Convenience: encode + decode a Value, return the decoded Value.
    fn roundtrip(v: &Value) -> Value {
        let s = to_toon(v).expect("encode");
        let back: Value = from_toon(&s).expect("decode");
        back
    }

    #[test]
    fn primitive_scalars_roundtrip() {
        // Spec §7.2 + §4 primitive lattice: null, bool, integer,
        // unquoted string. Float is exercised separately in
        // `float_canonical_form_is_lossy_but_consistent`.
        for v in [
            json!(null),
            json!(true),
            json!(false),
            json!(0_i64),
            json!(42_i64),
            json!(-7_i64),
            json!("hello"),
            json!("an_identifier_42"),
        ] {
            assert_eq!(roundtrip(&v), v, "roundtrip mismatch on {v:?}");
        }
    }

    #[test]
    fn string_must_quote_classes_roundtrip() {
        // Spec §7.2 must-quote enumeration. Each branch is a
        // quoting-rule trigger; encoding must escape, decoding must
        // restore byte-for-byte.
        for v in [
            json!(""),
            json!(" leading_space"),
            json!("trailing_space "),
            json!("true"),
            json!("false"),
            json!("null"),
            json!("123"), // numeric-like
            json!("-7"),  // numeric-like + leading hyphen
            json!("05"),  // leading-zero
            json!("-"),   // bare hyphen
            json!("with: colon"),
            json!("with, comma"),
            json!("with \"quote\""),
            json!("with \\ backslash"),
            json!("with [brackets]"),
            json!("with {braces}"),
            json!("with\nnewline"),
            json!("with\rcr"),
            json!("with\ttab"),
        ] {
            assert_eq!(roundtrip(&v), v, "must-quote roundtrip on {v:?}");
        }
    }

    #[test]
    fn unicode_text_roundtrips_unquoted_when_allowed() {
        let v = json!("emoji 🎉 and 漢字");
        let toon = to_toon(&v).expect("encode");
        // Spec §7 allows valid UTF-8 in unquoted strings provided no
        // structural / control / numeric-like trigger fires.
        assert!(
            !toon.contains('"'),
            "unicode text shouldn't trigger quotes: {toon:?}"
        );
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn empty_object_is_empty_document_at_root() {
        let s = to_toon(&json!({})).expect("encode");
        assert_eq!(s, "");
        let back: Value = from_toon("").expect("decode");
        assert_eq!(back, json!({}));
    }

    #[test]
    fn empty_array_emits_zero_header() {
        // Per spec §9 empty-array form: `[0]:` with no body.
        let s = to_toon(&json!({"xs": []})).expect("encode");
        assert!(s.contains("xs[0]:"), "missing zero-header: {s:?}");
        assert_eq!(roundtrip(&json!({"xs": []})), json!({"xs": []}));
    }

    #[test]
    fn nested_object_roundtrips() {
        let v = json!({
            "outer": {
                "inner": {
                    "leaf": 1,
                    "name": "deep",
                },
                "sibling": "value",
            },
            "top": "x",
        });
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn tabular_array_emits_compact_header_and_rows() {
        // The token-savings shape: uniform array of primitive-only
        // objects. Mirrors the LDBC SNB Person bench fixture.
        let v = json!({
            "people": [
                {"id": 1, "name": "Ada"},
                {"id": 2, "name": "Bob"},
                {"id": 3, "name": "Cay"},
            ],
        });
        let s = to_toon(&v).expect("encode");
        assert!(
            s.contains("people[3]{id,name}:"),
            "expected tabular header, got:\n{s}"
        );
        assert!(s.contains("1,Ada"), "missing tabular row 1: {s:?}");
        assert!(s.contains("2,Bob"), "missing tabular row 2: {s:?}");
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn inline_array_for_primitive_only_lists() {
        let v = json!({"tags": ["alpha", "beta", "gamma"]});
        let s = to_toon(&v).expect("encode");
        assert!(
            s.contains("tags[3]: alpha,beta,gamma"),
            "expected inline form, got: {s:?}"
        );
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn block_list_for_heterogeneous_arrays() {
        // Mixed types → cannot use tabular or inline; falls back to
        // block list with `- ` markers.
        let v = json!({"mix": [1, "two", null, true]});
        let s = to_toon(&v).expect("encode");
        assert!(
            s.contains("mix[4]: 1,two,null,true"),
            "primitives still go inline, got: {s:?}"
        );
        // Now force block by adding a non-primitive element.
        let v = json!({"mix": [1, {"k": "v"}, "three"]});
        let s = to_toon(&v).expect("encode");
        assert!(s.contains("mix[3]:\n"), "expected block header, got: {s:?}");
        assert!(s.contains("- 1"), "missing scalar list-item: {s:?}");
        assert!(s.contains("- k: v"), "missing object list-item: {s:?}");
        assert!(s.contains("- three"), "missing trailing scalar: {s:?}");
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn deeply_nested_structure_roundtrips() {
        // Stress the depth bookkeeping: 4 levels of object + arrays.
        let v = json!({
            "l0": {
                "l1": {
                    "l2": {
                        "l3": [1, 2, 3],
                        "l3b": {"leaf": "deep"},
                    },
                },
            },
        });
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn tabular_array_with_string_cells_roundtrips() {
        // Cells must be quoting-aware: comma-bearing values inside
        // tabular rows trigger quotes.
        let v = json!({
            "rows": [
                {"id": 1, "label": "Alice, A"},
                {"id": 2, "label": "Bob"},
            ],
        });
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn large_string_roundtrips() {
        let big: String = "x".repeat(8_000);
        let v = json!({"big": big});
        assert_eq!(roundtrip(&v), v);
    }

    #[test]
    fn float_canonical_form_is_lossy_but_consistent() {
        // Spec §2 canonicalizes 1.0 → "1" (integer). Round-trip via
        // the Value lattice promotes the float to integer; this test
        // pins that behavior so a future change doesn't silently
        // regress the canonical-form contract.
        let s = to_toon(&json!(1.0_f64)).expect("encode");
        assert_eq!(s, "1");
        let back: Value = from_toon(&s).expect("decode");
        assert_eq!(back, json!(1_i64));
        // Non-integer floats keep their decimal form.
        let s = to_toon(&json!(1.5_f64)).expect("encode");
        assert_eq!(s, "1.5");
        let back: Value = from_toon(&s).expect("decode");
        assert_eq!(back, json!(1.5_f64));
    }

    #[test]
    fn unencodable_key_surfaces_error() {
        // Quoted keys not supported in this slice; encoder errors.
        let v = json!({"with space": 1});
        let err = to_toon(&v).unwrap_err();
        assert!(matches!(err, ToonError::Unencodable(_)), "got {err:?}");
    }

    #[test]
    fn tab_indent_is_rejected() {
        let bad = "key:\n\tnested: 1\n";
        let err = from_toon::<Value>(bad).unwrap_err();
        assert!(matches!(err, ToonError::Indent { .. }), "got {err:?}");
    }

    #[test]
    fn count_mismatch_surfaces_error() {
        // Header declares 3 rows, only 2 supplied.
        let bad = "rows[3]{id}:\n  1\n  2\n";
        let err = from_toon::<Value>(bad).unwrap_err();
        assert!(
            matches!(
                err,
                ToonError::CountMismatch {
                    declared: 3,
                    observed: 2,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn row_width_mismatch_surfaces_error() {
        let bad = "rows[1]{id,name}:\n  1\n";
        let err = from_toon::<Value>(bad).unwrap_err();
        assert!(
            matches!(
                err,
                ToonError::RowWidthMismatch {
                    header: 2,
                    row: 1,
                    ..
                }
            ),
            "got {err:?}"
        );
    }

    #[test]
    fn root_array_inline_decodes() {
        let s = "[3]: 1,2,3\n";
        let v: Value = from_toon(s).expect("decode");
        assert_eq!(v, json!([1, 2, 3]));
    }

    #[test]
    fn root_array_tabular_decodes() {
        let s = "[2]{id,name}:\n  1,Ada\n  2,Bob\n";
        let v: Value = from_toon(s).expect("decode");
        assert_eq!(
            v,
            json!([{"id": 1, "name": "Ada"}, {"id": 2, "name": "Bob"}])
        );
    }

    #[test]
    fn root_bare_scalar_decodes() {
        for (input, expected) in [
            ("null", json!(null)),
            ("true", json!(true)),
            ("42", json!(42_i64)),
            ("\"a string\"", json!("a string")),
        ] {
            assert_eq!(from_toon::<Value>(input).expect("decode"), expected);
        }
    }

    // ─── ToonError taxonomy coverage (W11Z fix-up — MED-3 + NIT-3 + NIT-4) ───
    //
    // Pins each `ToonError` variant the prior happy-path proptest envelope
    // didn't exercise. The proptest at `tests/toon_proptest.rs` rejects on
    // any encode/decode failure but does NOT branch on the specific variant,
    // so a regression that swaps `Malformed` for `InvalidString` (or
    // similar) would slip through. These unit tests assert on the variant
    // class only — `reason` strings are debug-only.

    #[test]
    fn non_finite_floats_coerce_to_null_on_encode() {
        // Spec §2 canonicalization (encoder docs §"Encoding strategy"):
        // NaN / ±Inf are coerced to TOON `null` because the format has no
        // in-band representation for non-finite floats. Pin so a future
        // encoder change (e.g., emitting "nan" or returning Unencodable)
        // is caught instead of silently breaking downstream contract.
        assert_eq!(to_toon(&json!(f64::NAN)).expect("encode"), "null");
        assert_eq!(to_toon(&json!(f64::INFINITY)).expect("encode"), "null");
        assert_eq!(to_toon(&json!(f64::NEG_INFINITY)).expect("encode"), "null");
    }

    #[test]
    fn unencodable_in_string_control_char_surfaces_error() {
        // Spec §7.1 defines five escape sequences (\\, \", \n, \r, \t).
        // Other control chars in a string VALUE must surface as
        // Unencodable rather than silently emit a non-conforming literal.
        // Mirrors `unencodable_key_surfaces_error` for the in-key path.
        for c in ['\x00', '\x07', '\x1F'] {
            let v = json!({ "k": c.to_string() });
            let err = to_toon(&v).unwrap_err();
            assert!(
                matches!(err, ToonError::Unencodable(_)),
                "got {err:?} for U+{:04X}",
                c as u32
            );
        }
    }

    #[test]
    fn non_multiple_of_two_indent_is_rejected() {
        // Spec §14.3 strict mode: indent must be a multiple of indentSize=2.
        // Both 1-space and 3-space leads are rejected. Tab indentation is
        // exercised separately by `tab_indent_is_rejected`.
        for bad in ["k:\n nested: 1\n", "k:\n   nested: 1\n"] {
            let err = from_toon::<Value>(bad).unwrap_err();
            assert!(
                matches!(err, ToonError::Indent { .. }),
                "got {err:?} for {bad:?}"
            );
        }
    }

    #[test]
    fn malformed_array_header_variants_surface_errors() {
        // Spec §14.1 strict mode rejects malformed array headers. Each
        // input below trips a distinct `Malformed { reason }` arm in
        // `ArrayHeader::parse`.
        for bad in [
            "xs[3:\n",          // missing ']'
            "xs[1]{id:\n  1\n", // missing '}' in tabular header
            "xs[abc]:\n",       // non-numeric array count
            "xs[1\t]:\n",       // tab-delimiter declaration
            "xs[1|]:\n",        // pipe-delimiter declaration
            "xs[1]{id}: 1\n",   // tabular + inline (forbidden combo)
        ] {
            let err = from_toon::<Value>(bad).unwrap_err();
            assert!(
                matches!(err, ToonError::Malformed { .. }),
                "got {err:?} for {bad:?}"
            );
        }
    }

    #[test]
    fn list_item_line_at_object_position_surfaces_malformed() {
        // Object-position parser rejects `- ...` as a field line — pins
        // the "unexpected list-item line at object position" Malformed arm.
        let err = from_toon::<Value>("k: 1\n- listitem\n").unwrap_err();
        assert!(matches!(err, ToonError::Malformed { .. }), "got {err:?}");
    }

    #[test]
    fn trailing_content_after_root_array_surfaces_malformed() {
        // After parse_root consumes the count-1 array, the unconsumed
        // tail line trips parse_toon's pos != lines.len() check.
        let err = from_toon::<Value>("[1]:\n  - a\n  - b\n").unwrap_err();
        assert!(matches!(err, ToonError::Malformed { .. }), "got {err:?}");
    }

    #[test]
    fn invalid_string_variants_surface_errors() {
        // Spec §7.1: invalid escape, unterminated literal, control char
        // inside a quoted literal — all three trip InvalidString.
        for bad in [
            "k: \"\\x\"\n",        // unrecognized escape
            "k: \"unterminated\n", // never closes
            "k: \"\x07\"\n",       // unescaped control char in quoted string
        ] {
            let err = from_toon::<Value>(bad).unwrap_err();
            assert!(
                matches!(err, ToonError::InvalidString { .. }),
                "got {err:?} for {bad:?}"
            );
        }
    }

    #[test]
    fn serde_pivot_failure_surfaces_error() {
        // A `Serialize` impl that errors mid-stream surfaces as
        // ToonError::SerdePivot — distinct from any encoder-side error.
        use serde::ser::Error as _;

        struct AlwaysFails;
        impl serde::Serialize for AlwaysFails {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(S::Error::custom("intentional test failure"))
            }
        }

        let err = to_toon(&AlwaysFails).unwrap_err();
        assert!(matches!(err, ToonError::SerdePivot(_)), "got {err:?}");
    }

    #[test]
    fn decode_target_mismatch_surfaces_error() {
        // The TOON parses cleanly to a Value, but the requested target
        // type rejects the resulting JSON shape — surface as DecodeTarget,
        // not Malformed / InvalidString.
        #[derive(serde::Deserialize, Debug)]
        #[allow(dead_code)]
        struct Strict {
            n: i64,
        }
        let err = from_toon::<Strict>("n: not-a-number\n").unwrap_err();
        assert!(matches!(err, ToonError::DecodeTarget(_)), "got {err:?}");
    }
}
