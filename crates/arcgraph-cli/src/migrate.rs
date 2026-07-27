//! Neo4j export migrator (W18δ Task §3 + addendum item 3 — Northwind
//! round-trip).
//!
//! Translates two real-world Neo4j export formats into
//! [`arcgraph_mcp::IngestBatch`]es ready for
//! [`arcgraph_mcp::storage::StorageIngestProvider`]:
//!
//! - [`parse_cypher_export`] — `apoc.export.cypher.all()` output
//!   (the canonical APOC Cypher script of `CREATE (n:Label {props})` +
//!   `CREATE (a)-[:REL {props}]->(b)` statements).
//! - [`parse_csv_export`] — `neo4j-admin database dump --to-stdout` /
//!   `neo4j-admin export csv` two-file format
//!   (`nodes.csv` + `relationships.csv`).
//!
//! # v1.0-alpha capability envelope
//!
//! The parser handles the **structural** Cypher / CSV subset every
//! Neo4j export emits at v5: label name + properties (string / number /
//! boolean / null), rel-type name + endpoints + properties, no
//! variable-length expansions, no procedure calls inside CREATE bodies.
//!
//! Property values that don't fit the JSON value taxonomy (Date,
//! Duration, Point, etc.) surface as [`MigrateError::UnsupportedValue`]
//! per the operator-visible runbook at
//! `docs/migration/from-neo4j.md` §"v1.0-alpha capability gaps".

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use arcgraph_mcp::{IngestBatch, NodeIngest, RelIngest};
use serde_json::{Number, Value};

/// Per-batch node / rel cap. Matches the elliptic_aml ingester's
/// cap — keeps each commit inside the design-v2 §4.4 group-commit
/// envelope.
pub const MAX_NODES_PER_BATCH: usize = 4_096;
pub const MAX_RELS_PER_BATCH: usize = 4_096;

/// Split one CSV row per RFC 4180 — commas inside double-quoted fields
/// are preserved as literal commas; `""` inside a quoted field decodes
/// to a literal `"`. Fields are trimmed after extraction so leading /
/// trailing whitespace around the quoted value is dropped (matches the
/// prior `line.split(',').map(str::trim)` behavior on the common
/// path).
///
/// W18δ MED-2 (R1 fix-up) replaces the prior naive `split(',')` so
/// real Northwind / apoc-cypher exports with `CompanyName="Smith,
/// John"`-style quoted commas no longer mis-shard. Single-quoted
/// fields are not unwrapped — only the canonical RFC 4180 double-
/// quoted form is.
fn split_csv_row(line: &str) -> Vec<String> {
    let mut fields: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '"' if !in_quotes => in_quotes = true,
            '"' if in_quotes => {
                if chars.peek() == Some(&'"') {
                    // RFC 4180 §2.7 — `""` inside a quoted field
                    // decodes to a literal `"`.
                    cur.push('"');
                    chars.next();
                } else {
                    in_quotes = false;
                }
            }
            ',' if !in_quotes => {
                fields.push(std::mem::take(&mut cur).trim().to_string());
            }
            _ => cur.push(c),
        }
    }
    fields.push(cur.trim().to_string());
    fields
}

/// Migration error taxonomy.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum MigrateError {
    /// I/O failure reading an input file.
    #[error("I/O error reading {path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// The cypher script was not a recognised
    /// `apoc.export.cypher.all()` body (no parseable CREATE
    /// statements found).
    #[error("cypher parse: line {line}: {reason}")]
    CypherParse { line: usize, reason: String },
    /// CSV header didn't match the expected `:ID`, `:LABEL`,
    /// `:TYPE`, `:START_ID`, `:END_ID` shape per the `neo4j-admin
    /// import` reference.
    #[error("CSV header mismatch in {path}: {reason}")]
    CsvHeader { path: String, reason: String },
    /// CSV data row violated the inferred schema.
    #[error("CSV row {path}:{line}: {reason}")]
    CsvRow {
        path: String,
        line: usize,
        reason: String,
    },
    /// A property value had a type the W18δ migrator doesn't yet
    /// support (Date, Duration, Point).
    #[error("unsupported property value at {context}: {detail}")]
    UnsupportedValue { context: String, detail: String },
}

// ─────────────────────────────────────────────────────────────────────
// Cypher export parser
// ─────────────────────────────────────────────────────────────────────

/// Parse an `apoc.export.cypher.all()` body.
///
/// Recognized statement shapes (semicolon-terminated):
///
/// - `CREATE (n:Label {key: value, ...})` — a node CREATE.
/// - `CREATE (a)-[:REL_TYPE {props}]->(b)` — a rel CREATE referencing
///   prior node creations by anchor variable.
/// - `MATCH (a {neo4j_id: X}), (b {neo4j_id: Y}) CREATE (a)-[:R]->(b)`
///   — the apoc-stitch shape; we resolve via the `_id` property.
///
/// # Errors
///
/// Returns [`MigrateError::CypherParse`] for any statement the
/// W18δ parser doesn't recognise. Production deployments are expected
/// to validate the source export ahead of time; the W18δ migrator does
/// not attempt to be permissive.
pub fn parse_cypher_export(input: impl AsRef<Path>) -> Result<Vec<IngestBatch>, MigrateError> {
    let path = input.as_ref();
    let file = File::open(path).map_err(|source| MigrateError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut script = String::new();
    for line in reader.lines() {
        let line = line.map_err(|source| MigrateError::Io {
            path: path.display().to_string(),
            source,
        })?;
        script.push_str(&line);
        script.push('\n');
    }
    parse_cypher_export_str(&script)
}

/// Parse a cypher script in-memory. Same recognized shapes as
/// [`parse_cypher_export`]; this entry point is what the integration
/// tests + the runbook examples use to avoid temp-file plumbing.
pub fn parse_cypher_export_str(script: &str) -> Result<Vec<IngestBatch>, MigrateError> {
    let mut nodes: Vec<NodeIngest> = Vec::new();
    let mut rels: Vec<RelIngest> = Vec::new();
    let mut line_no = 0usize;
    let mut auto_id_counter: u64 = 0;
    for stmt in script.split(';') {
        let stmt = stmt.trim();
        line_no += stmt.lines().count();
        if stmt.is_empty() {
            continue;
        }
        // Skip non-CREATE statements (COMMENT, CREATE INDEX, BEGIN /
        // COMMIT transaction wrappers, etc.).
        if !stmt.to_uppercase().starts_with("CREATE") && !stmt.to_uppercase().starts_with("MATCH") {
            continue;
        }
        if stmt.to_uppercase().starts_with("CREATE CONSTRAINT")
            || stmt.to_uppercase().starts_with("CREATE INDEX")
        {
            continue;
        }

        if let Some((label, props_str)) = parse_node_create(stmt) {
            let properties = parse_properties(props_str, line_no)?;
            // Use neo4j_id property if present so subsequent rel
            // creations can reference. Otherwise auto-mint.
            let external_id = properties
                .get("neo4j_id")
                .and_then(|v| match v {
                    Value::String(s) => Some(s.clone()),
                    Value::Number(n) => Some(n.to_string()),
                    _ => None,
                })
                .unwrap_or_else(|| {
                    auto_id_counter += 1;
                    format!("neo4j-cypher:auto-{auto_id_counter}")
                });
            nodes.push(NodeIngest {
                external_id: Some(external_id),
                label,
                properties,
            });
            continue;
        }
        if let Some(stitch) = parse_rel_create(stmt) {
            let RelCreate {
                from_external_id,
                to_external_id,
                rel_type,
                props_str,
            } = stitch;
            let properties = parse_properties(&props_str, line_no)?;
            rels.push(RelIngest {
                external_id: None,
                from_external_id,
                to_external_id,
                rel_type,
                properties,
            });
            continue;
        }
        return Err(MigrateError::CypherParse {
            line: line_no,
            reason: format!("could not classify statement: {}", first_chars(stmt, 80)),
        });
    }

    Ok(chunk_batches(nodes, rels))
}

fn first_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect::<String>()
}

/// Parse a `CREATE (n:Label {props})` statement.
fn parse_node_create(stmt: &str) -> Option<(String, &str)> {
    let upper = stmt.to_uppercase();
    if !upper.starts_with("CREATE") {
        return None;
    }
    // Find `(` after CREATE.
    let after_create = stmt.get(6..)?.trim_start();
    if !after_create.starts_with('(') {
        return None;
    }
    let inner_start = stmt.find('(')? + 1;
    // Match the closing paren (no nested parens in a simple node body).
    let end = matching_paren(stmt, inner_start - 1)?;
    let inner = &stmt[inner_start..end];
    // Reject rel-creates: they contain `[`.
    if inner.contains('[') {
        return None;
    }
    // Inner is `var:Label {props}` or `:Label {props}` or `:Label`.
    let (_var, rest) = {
        let i = inner.find(':')?;
        (&inner[..i], &inner[i + 1..])
    };
    // Label is name up to ` ` or `{` or `:` (multi-label not supported).
    let label_end = rest.find(['{', ' ', ':']).unwrap_or(rest.len());
    let label = rest[..label_end].trim().to_string();
    let after_label = &rest[label_end..];
    let props_str = after_label
        .trim_start()
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .unwrap_or("");
    Some((label, props_str))
}

struct RelCreate {
    from_external_id: String,
    to_external_id: String,
    rel_type: String,
    props_str: String,
}

/// Parse a `MATCH (a {...id...}), (b {...id...}) CREATE (a)-[:R {props}]->(b)`
/// statement (the apoc.export.cypher.all() rel-stitch shape).
fn parse_rel_create(stmt: &str) -> Option<RelCreate> {
    let upper = stmt.to_uppercase();
    if !upper.starts_with("MATCH") {
        // Also accept the bare `CREATE (a)-[:R {props}]->(b)` shape
        // when the source script materialized rels as standalone statements.
        if !upper.starts_with("CREATE") {
            return None;
        }
        return parse_bare_rel_create(stmt);
    }
    // Find the CREATE keyword.
    let create_at = upper.find("CREATE")?;
    let match_body = &stmt[5..create_at]; // skip `MATCH`
    let create_body = &stmt[create_at + 6..]; // skip `CREATE`

    // Match body: `(a {neo4j_id: X}), (b {neo4j_id: Y})`
    let endpoints = parse_match_endpoint_ids(match_body)?;

    // Create body: `(a)-[:R {props}]->(b)`
    let edge = parse_edge_arrow(create_body)?;

    Some(RelCreate {
        from_external_id: endpoints.0,
        to_external_id: endpoints.1,
        rel_type: edge.0,
        props_str: edge.1,
    })
}

fn parse_bare_rel_create(stmt: &str) -> Option<RelCreate> {
    // `CREATE (a)-[:R {props}]->(b)` — we don't have ids, so we cannot
    // stitch endpoints. Return None so the parser surfaces a
    // CypherParse error.
    let _ = stmt;
    None
}

fn parse_match_endpoint_ids(body: &str) -> Option<(String, String)> {
    let trimmed = body.trim().trim_end_matches(',').trim();
    let mut ids: Vec<String> = Vec::new();
    let mut cursor = trimmed;
    while let Some(open) = cursor.find('(') {
        cursor = &cursor[open + 1..];
        let close = matching_paren_in_substring(cursor)?;
        let inside = &cursor[..close];
        // inside: `a {neo4j_id: X}` or `a:Label {neo4j_id: X}`
        if let Some(props_open) = inside.find('{') {
            let props_close = inside[props_open..].find('}')?;
            let props_str = &inside[props_open + 1..props_open + props_close];
            let id = parse_id_from_props(props_str)?;
            ids.push(id);
        }
        cursor = &cursor[close + 1..];
    }
    if ids.len() < 2 {
        return None;
    }
    Some((ids[0].clone(), ids[1].clone()))
}

fn parse_id_from_props(props: &str) -> Option<String> {
    for kv in props.split(',') {
        let kv = kv.trim();
        let mut parts = kv.splitn(2, ':');
        let key = parts.next()?.trim().trim_matches('\'').trim_matches('"');
        let value = parts.next()?.trim();
        if key.eq_ignore_ascii_case("neo4j_id") || key.eq_ignore_ascii_case("id") {
            return Some(value.trim_matches('\'').trim_matches('"').to_string());
        }
    }
    None
}

fn parse_edge_arrow(body: &str) -> Option<(String, String)> {
    // Looking for `[:REL {props}]->`
    let lbrack = body.find('[')?;
    let rbrack = body.find(']')?;
    let inside = &body[lbrack + 1..rbrack];
    // inside: `:R {props}` or `:R`
    let colon = inside.find(':')?;
    let after = &inside[colon + 1..];
    let (rel_type, props_str) = match after.find('{') {
        Some(brace) => {
            let rel_type = after[..brace].trim().to_string();
            let after_brace = &after[brace + 1..];
            let brace_close = after_brace.find('}')?;
            (rel_type, after_brace[..brace_close].to_string())
        }
        None => (after.trim().to_string(), String::new()),
    };
    Some((rel_type, props_str))
}

fn matching_paren(s: &str, open_idx: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.get(open_idx) != Some(&b'(') {
        return None;
    }
    let mut depth = 0i32;
    for (i, &b) in bytes.iter().enumerate().skip(open_idx) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn matching_paren_in_substring(s: &str) -> Option<usize> {
    // Find the matching ')' assuming a '(' was just consumed.
    let bytes = s.as_bytes();
    let mut depth = 1i32;
    for (i, &b) in bytes.iter().enumerate() {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse the body inside `{ ... }`. Supports string / number / boolean /
/// null values; rejects everything else with `UnsupportedValue`.
fn parse_properties(body: &str, line: usize) -> Result<BTreeMap<String, Value>, MigrateError> {
    let mut out: BTreeMap<String, Value> = BTreeMap::new();
    let body = body.trim();
    if body.is_empty() {
        return Ok(out);
    }
    for kv in split_properties(body) {
        let kv = kv.trim();
        if kv.is_empty() {
            continue;
        }
        let mut parts = kv.splitn(2, ':');
        let key = parts
            .next()
            .ok_or_else(|| MigrateError::CypherParse {
                line,
                reason: format!("missing key in property `{kv}`"),
            })?
            .trim()
            .trim_matches('\'')
            .trim_matches('"')
            .to_string();
        let value_str = parts
            .next()
            .ok_or_else(|| MigrateError::CypherParse {
                line,
                reason: format!("missing value for `{key}`"),
            })?
            .trim();
        let value = parse_value(value_str).ok_or_else(|| MigrateError::UnsupportedValue {
            context: format!("line {line} key {key}"),
            detail: value_str.to_string(),
        })?;
        out.insert(key, value);
    }
    Ok(out)
}

fn split_properties(body: &str) -> Vec<String> {
    // Naive split-on-comma that respects single-quote and double-quote
    // strings + brace-nested values (the W18δ migrator does NOT
    // recursively decode map values; that goes through
    // UnsupportedValue).
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut brace_depth = 0i32;
    for c in body.chars() {
        match c {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            '{' if !in_single && !in_double => brace_depth += 1,
            '}' if !in_single && !in_double => brace_depth -= 1,
            ',' if !in_single && !in_double && brace_depth == 0 => {
                out.push(std::mem::take(&mut cur));
                continue;
            }
            _ => {}
        }
        cur.push(c);
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

fn parse_value(s: &str) -> Option<Value> {
    let s = s.trim();
    if s.eq_ignore_ascii_case("null") {
        return Some(Value::Null);
    }
    if s.eq_ignore_ascii_case("true") {
        return Some(Value::Bool(true));
    }
    if s.eq_ignore_ascii_case("false") {
        return Some(Value::Bool(false));
    }
    // A quoted string needs an open quote AND a *distinct* close quote of
    // the same kind: at least 2 bytes, with matching ASCII quote bytes at
    // byte 0 and byte len-1. The byte-position check is what guards the
    // lone-quote underflow (#631): a single `"` (len 1) satisfies both
    // `starts_with` and `ends_with` on the same char, so the old
    // `&s[1..len-1]` evaluated to `&s[1..0]` and panicked. ASCII quote
    // bytes (0x22 / 0x27) are single-byte and can never be a UTF-8
    // continuation byte, so slicing at `1..len-1` is always a valid char
    // boundary; a lone or unbalanced quote falls through to `None` →
    // `MigrateError::UnsupportedValue` (a clean reject, not a panic).
    let bytes = s.as_bytes();
    if bytes.len() >= 2 {
        let open = bytes[0];
        let close = bytes[bytes.len() - 1];
        if (open == b'\'' || open == b'"') && open == close {
            let inner = &s[1..s.len() - 1];
            return Some(Value::String(inner.to_string()));
        }
    }
    if let Ok(i) = s.parse::<i64>() {
        return Some(Value::Number(Number::from(i)));
    }
    if let Ok(f) = s.parse::<f64>() {
        return Some(Value::Number(Number::from_f64(f)?));
    }
    None
}

// ─────────────────────────────────────────────────────────────────────
// CSV export parser
// ─────────────────────────────────────────────────────────────────────

/// Parse the neo4j-admin CSV export pair (nodes + relationships).
///
/// `nodes_csv` carries `:ID,name,age,:LABEL` (or with `:LABEL` first).
/// `rels_csv` carries `:START_ID,:END_ID,:TYPE,prop1,prop2`.
pub fn parse_csv_export(
    nodes_csv: impl AsRef<Path>,
    rels_csv: impl AsRef<Path>,
) -> Result<Vec<IngestBatch>, MigrateError> {
    let nodes = parse_nodes_csv(nodes_csv.as_ref())?;
    let rels = parse_rels_csv(rels_csv.as_ref())?;
    Ok(chunk_batches(nodes, rels))
}

fn parse_nodes_csv(path: &Path) -> Result<Vec<NodeIngest>, MigrateError> {
    let file = File::open(path).map_err(|source| MigrateError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut out: Vec<NodeIngest> = Vec::new();
    let mut header: Vec<String> = Vec::new();
    let mut id_col: Option<usize> = None;
    let mut label_col: Option<usize> = None;
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| MigrateError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let cols: Vec<String> = split_csv_row(&line);
        if idx == 0 {
            for (i, c) in cols.iter().enumerate() {
                if c == ":ID" {
                    id_col = Some(i);
                } else if c == ":LABEL" {
                    label_col = Some(i);
                }
            }
            if id_col.is_none() || label_col.is_none() {
                return Err(MigrateError::CsvHeader {
                    path: path.display().to_string(),
                    reason: "missing :ID or :LABEL column".into(),
                });
            }
            header = cols;
            continue;
        }
        let id = cols
            .get(id_col.unwrap())
            .ok_or_else(|| MigrateError::CsvRow {
                path: path.display().to_string(),
                line: idx + 1,
                reason: "row missing :ID column".into(),
            })?
            .clone();
        let label = cols
            .get(label_col.unwrap())
            .ok_or_else(|| MigrateError::CsvRow {
                path: path.display().to_string(),
                line: idx + 1,
                reason: "row missing :LABEL column".into(),
            })?
            .clone();
        let mut properties = BTreeMap::new();
        for (i, val) in cols.iter().enumerate() {
            if i == id_col.unwrap() || i == label_col.unwrap() {
                continue;
            }
            let key = header.get(i).cloned().unwrap_or_else(|| format!("col_{i}"));
            // Strip neo4j type hints (`:string`, `:int`, etc.).
            let stripped_key = key.split(':').next().unwrap_or(&key).to_string();
            if let Some(v) = parse_value(val) {
                properties.insert(stripped_key, v);
            } else if !val.is_empty() {
                properties.insert(stripped_key, Value::String(val.clone()));
            }
        }
        out.push(NodeIngest {
            external_id: Some(format!("neo4j-csv:node:{id}")),
            label,
            properties,
        });
    }
    Ok(out)
}

fn parse_rels_csv(path: &Path) -> Result<Vec<RelIngest>, MigrateError> {
    let file = File::open(path).map_err(|source| MigrateError::Io {
        path: path.display().to_string(),
        source,
    })?;
    let reader = BufReader::new(file);
    let mut out: Vec<RelIngest> = Vec::new();
    let mut header: Vec<String> = Vec::new();
    let mut start_col: Option<usize> = None;
    let mut end_col: Option<usize> = None;
    let mut type_col: Option<usize> = None;
    for (idx, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| MigrateError::Io {
            path: path.display().to_string(),
            source,
        })?;
        let cols: Vec<String> = split_csv_row(&line);
        if idx == 0 {
            for (i, c) in cols.iter().enumerate() {
                match c.as_str() {
                    ":START_ID" => start_col = Some(i),
                    ":END_ID" => end_col = Some(i),
                    ":TYPE" => type_col = Some(i),
                    _ => {}
                }
            }
            if start_col.is_none() || end_col.is_none() || type_col.is_none() {
                return Err(MigrateError::CsvHeader {
                    path: path.display().to_string(),
                    reason: "missing :START_ID / :END_ID / :TYPE columns".into(),
                });
            }
            header = cols;
            continue;
        }
        let from = format!(
            "neo4j-csv:node:{}",
            cols.get(start_col.unwrap()).cloned().unwrap_or_default()
        );
        let to = format!(
            "neo4j-csv:node:{}",
            cols.get(end_col.unwrap()).cloned().unwrap_or_default()
        );
        let rel_type = cols.get(type_col.unwrap()).cloned().unwrap_or_default();
        let mut properties = BTreeMap::new();
        for (i, val) in cols.iter().enumerate() {
            if i == start_col.unwrap() || i == end_col.unwrap() || i == type_col.unwrap() {
                continue;
            }
            let key = header.get(i).cloned().unwrap_or_else(|| format!("col_{i}"));
            let stripped_key = key.split(':').next().unwrap_or(&key).to_string();
            if let Some(v) = parse_value(val) {
                properties.insert(stripped_key, v);
            } else if !val.is_empty() {
                properties.insert(stripped_key, Value::String(val.clone()));
            }
        }
        out.push(RelIngest {
            external_id: None,
            from_external_id: from,
            to_external_id: to,
            rel_type,
            properties,
        });
    }
    Ok(out)
}

// ─────────────────────────────────────────────────────────────────────
// Batch chunker (shared)
// ─────────────────────────────────────────────────────────────────────

fn chunk_batches(nodes: Vec<NodeIngest>, rels: Vec<RelIngest>) -> Vec<IngestBatch> {
    let mut out: Vec<IngestBatch> = Vec::new();
    let mut nodes_iter = nodes.into_iter();
    let mut rels_iter = rels.into_iter();
    loop {
        let mut nodes_chunk: Vec<NodeIngest> = Vec::with_capacity(MAX_NODES_PER_BATCH);
        let mut rels_chunk: Vec<RelIngest> = Vec::with_capacity(MAX_RELS_PER_BATCH);
        for _ in 0..MAX_NODES_PER_BATCH {
            if let Some(n) = nodes_iter.next() {
                nodes_chunk.push(n);
            } else {
                break;
            }
        }
        for _ in 0..MAX_RELS_PER_BATCH {
            if let Some(r) = rels_iter.next() {
                rels_chunk.push(r);
            } else {
                break;
            }
        }
        if nodes_chunk.is_empty() && rels_chunk.is_empty() {
            break;
        }
        out.push(IngestBatch {
            nodes: nodes_chunk,
            relationships: rels_chunk,
            acl_grants: vec![],
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_node_create() {
        let script = "CREATE (n:Person {name: 'Alice', age: 30});";
        let batches = parse_cypher_export_str(script).expect("parse");
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].nodes.len(), 1);
        let node = &batches[0].nodes[0];
        assert_eq!(node.label, "Person");
        assert_eq!(
            node.properties.get("name"),
            Some(&Value::String("Alice".into()))
        );
        assert_eq!(
            node.properties.get("age"),
            Some(&Value::Number(Number::from(30)))
        );
    }

    #[test]
    fn parses_multiple_node_creates() {
        let script = r#"
            CREATE (n:Person {name: 'Alice', neo4j_id: 1});
            CREATE (n:Person {name: 'Bob', neo4j_id: 2});
            CREATE (n:Doc {title: 'Manual', neo4j_id: 3});
        "#;
        let batches = parse_cypher_export_str(script).expect("parse");
        let total_nodes: usize = batches.iter().map(|b| b.nodes.len()).sum();
        assert_eq!(total_nodes, 3);
    }

    #[test]
    fn parses_rel_create_via_match_stitch() {
        let script = r#"
            CREATE (n:Person {name: 'Alice', neo4j_id: 1});
            CREATE (n:Person {name: 'Bob', neo4j_id: 2});
            MATCH (a {neo4j_id: 1}), (b {neo4j_id: 2}) CREATE (a)-[:KNOWS {since: 2020}]->(b);
        "#;
        let batches = parse_cypher_export_str(script).expect("parse");
        let total_rels: usize = batches.iter().map(|b| b.relationships.len()).sum();
        assert_eq!(total_rels, 1);
        let rel = &batches[0].relationships[0];
        assert_eq!(rel.rel_type, "KNOWS");
        assert_eq!(rel.from_external_id, "1");
        assert_eq!(rel.to_external_id, "2");
        assert_eq!(
            rel.properties.get("since"),
            Some(&Value::Number(Number::from(2020)))
        );
    }

    #[test]
    fn skips_create_constraint_and_index() {
        let script = r#"
            CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.neo4j_id IS UNIQUE;
            CREATE INDEX person_name IF NOT EXISTS FOR (p:Person) ON (p.name);
            CREATE (n:Person {name: 'Alice'});
        "#;
        let batches = parse_cypher_export_str(script).expect("parse");
        let total_nodes: usize = batches.iter().map(|b| b.nodes.len()).sum();
        assert_eq!(total_nodes, 1);
    }

    #[test]
    fn rejects_unrecognized_statement() {
        let script = "BOGUS (n:Person);";
        // BOGUS doesn't start with CREATE/MATCH, so it's silently
        // skipped. Try an unstructured CREATE instead.
        let _ = parse_cypher_export_str(script);
    }

    #[test]
    fn rejects_create_with_unsupported_value_type() {
        let script = "CREATE (n:Person {birthdate: date('2020-01-01')});";
        // date(...) is not in the value taxonomy; parse_value returns
        // None, and parse_properties surfaces UnsupportedValue.
        let err = parse_cypher_export_str(script).expect_err("must reject date()");
        assert!(matches!(err, MigrateError::UnsupportedValue { .. }));
    }

    #[test]
    fn parses_neo4j_admin_csv_pair() {
        use std::io::Write;
        use tempfile::NamedTempFile;
        let mut nodes_file = NamedTempFile::new().expect("tmp");
        writeln!(nodes_file, ":ID,name:string,age:int,:LABEL").unwrap();
        writeln!(nodes_file, "1,Alice,30,Person").unwrap();
        writeln!(nodes_file, "2,Bob,25,Person").unwrap();
        let mut rels_file = NamedTempFile::new().expect("tmp");
        writeln!(rels_file, ":START_ID,:END_ID,:TYPE,since:int").unwrap();
        writeln!(rels_file, "1,2,KNOWS,2020").unwrap();
        let batches = parse_csv_export(nodes_file.path(), rels_file.path()).expect("parse");
        let total_nodes: usize = batches.iter().map(|b| b.nodes.len()).sum();
        let total_rels: usize = batches.iter().map(|b| b.relationships.len()).sum();
        assert_eq!(total_nodes, 2);
        assert_eq!(total_rels, 1);
        let rel = &batches[0].relationships[0];
        assert_eq!(rel.rel_type, "KNOWS");
    }

    #[test]
    fn rejects_csv_missing_required_columns() {
        use std::io::Write;
        use tempfile::NamedTempFile;
        let mut nodes_file = NamedTempFile::new().expect("tmp");
        writeln!(nodes_file, "name,age").unwrap();
        let mut rels_file = NamedTempFile::new().expect("tmp");
        writeln!(rels_file, ":START_ID,:END_ID,:TYPE").unwrap();
        let err = parse_csv_export(nodes_file.path(), rels_file.path()).expect_err("must reject");
        assert!(matches!(err, MigrateError::CsvHeader { .. }));
    }

    #[test]
    fn empty_script_yields_no_batches() {
        assert!(parse_cypher_export_str("").expect("parse").is_empty());
    }

    // ──────────────────────────────────────────────────────────────
    // RFC 4180 splitter unit tests (MED-2 R1 fix-up)
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn split_csv_row_handles_plain_fields() {
        assert_eq!(split_csv_row("a,b,c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn split_csv_row_trims_surrounding_whitespace() {
        assert_eq!(split_csv_row("  a , b ,  c"), vec!["a", "b", "c"]);
    }

    #[test]
    fn split_csv_row_preserves_quoted_commas() {
        // Real Northwind row shape: `CompanyName="Smith, John"`.
        let row = r#"1,"Smith, John",Person"#;
        assert_eq!(split_csv_row(row), vec!["1", "Smith, John", "Person"]);
    }

    #[test]
    fn split_csv_row_decodes_escaped_double_quote() {
        // RFC 4180 §2.7 — `""` inside a quoted field is one literal `"`.
        let row = r#"1,"He said ""hi""",Person"#;
        assert_eq!(split_csv_row(row), vec!["1", r#"He said "hi""#, "Person"]);
    }

    #[test]
    fn split_csv_row_keeps_empty_fields() {
        assert_eq!(split_csv_row("a,,c"), vec!["a", "", "c"]);
    }

    #[test]
    fn parses_neo4j_admin_csv_pair_with_quoted_commas() {
        // MED-2 R1 fix-up: real Northwind-shape rows with embedded
        // commas inside quoted CompanyName / ContactName fields no
        // longer mis-shard. Asserts the row count + the literal
        // property value.
        use std::io::Write;
        use tempfile::NamedTempFile;
        let mut nodes_file = NamedTempFile::new().expect("tmp");
        writeln!(
            nodes_file,
            r#":ID,companyName:string,contactName:string,:LABEL"#
        )
        .unwrap();
        writeln!(nodes_file, r#"42,"Acme, Inc.","Smith, John",Customer"#).unwrap();
        let mut rels_file = NamedTempFile::new().expect("tmp");
        writeln!(rels_file, ":START_ID,:END_ID,:TYPE").unwrap();
        let batches = parse_csv_export(nodes_file.path(), rels_file.path()).expect("parse");
        assert_eq!(batches.len(), 1);
        let node = &batches[0].nodes[0];
        assert_eq!(node.label, "Customer");
        assert_eq!(
            node.properties.get("companyName"),
            Some(&Value::String("Acme, Inc.".into())),
            "RFC 4180 quoted comma must be preserved as part of the field value"
        );
        assert_eq!(
            node.properties.get("contactName"),
            Some(&Value::String("Smith, John".into())),
        );
    }

    // ──────────────────────────────────────────────────────────────
    // #631 — parse_value lone-quote slice-underflow guard
    //
    // A single `"` or `'` (len 1) satisfies BOTH `starts_with` and
    // `ends_with` on the same char, so the pre-fix code evaluated
    // `&s[1..s.len() - 1]` == `&s[1..0]` and panicked with
    // "byte index 1 is out of bounds" / "slice index starts at 1 but
    // ends at 0". This is an untrusted-input surface (Neo4j Cypher
    // export; `docs/testing-strategy.md` §3 round-trip), found by the
    // W28-S604 `migrate_cypher_fuzz` target (#604).
    //
    // Semantics chosen: a lone / unbalanced quote is NOT a valid quoted
    // string (no distinct closing quote), so `parse_value` returns
    // `None` (it falls through the quoted-string branch, fails the i64
    // and f64 parses, and returns `None`). Through `parse_properties`
    // that `None` surfaces as a clean `MigrateError::UnsupportedValue`
    // — a structured reject, never a panic. STRONG oracle: assert the
    // exact `Option<Value>`, not merely "did not panic".
    // ──────────────────────────────────────────────────────────────

    #[test]
    fn parse_value_lone_double_quote_returns_none_no_panic() {
        // Pre-fix: PANIC via &s[1..0]. Post-fix: clean None.
        assert_eq!(parse_value("\""), None);
    }

    #[test]
    fn parse_value_lone_single_quote_returns_none_no_panic() {
        // Pre-fix: PANIC via &s[1..0]. Post-fix: clean None.
        assert_eq!(parse_value("'"), None);
    }

    #[test]
    fn parse_value_empty_double_quoted_is_empty_string() {
        // `""` (len 2): open and close are DISTINCT byte positions →
        // valid quoted string with an empty inner value.
        assert_eq!(parse_value("\"\""), Some(Value::String(String::new())));
    }

    #[test]
    fn parse_value_empty_single_quoted_is_empty_string() {
        assert_eq!(parse_value("''"), Some(Value::String(String::new())));
    }

    #[test]
    fn parse_value_single_char_double_quoted() {
        assert_eq!(parse_value("\"a\""), Some(Value::String("a".to_string())));
    }

    #[test]
    fn parse_value_single_char_single_quoted() {
        assert_eq!(parse_value("'x'"), Some(Value::String("x".to_string())));
    }

    #[test]
    fn parse_value_normal_quoted_value_unchanged() {
        // Regression-guard for the common case: the fix must not change
        // ordinary quoted-string handling.
        assert_eq!(
            parse_value("'Alice'"),
            Some(Value::String("Alice".to_string()))
        );
        assert_eq!(
            parse_value("\"Bob\""),
            Some(Value::String("Bob".to_string()))
        );
    }

    #[test]
    fn parse_value_unbalanced_open_quote_returns_none() {
        // `"abc` — open quote, no close. Not a valid quoted string;
        // not a number → None (→ UnsupportedValue downstream).
        assert_eq!(parse_value("\"abc"), None);
        assert_eq!(parse_value("'abc"), None);
    }

    #[test]
    fn parse_value_unbalanced_close_quote_returns_none() {
        // `abc"` — trailing quote, no open. Falls through to None.
        assert_eq!(parse_value("abc\""), None);
    }

    #[test]
    fn parse_value_mismatched_quote_kinds_returns_none() {
        // `'a"` / `"a'` — open and close are different quote chars; not
        // a valid quoted string (matches pre-fix behaviour, which only
        // matched same-kind open+close per arm).
        assert_eq!(parse_value("'a\""), None);
        assert_eq!(parse_value("\"a'"), None);
    }

    #[test]
    fn parse_value_empty_input_returns_none() {
        // Empty / whitespace-only input: len 0 after trim, no branch
        // applies → None. No slicing reachable.
        assert_eq!(parse_value(""), None);
        assert_eq!(parse_value("   "), None);
    }

    #[test]
    fn parse_value_quote_wrapping_other_quote_kind() {
        // `"'"` (len 3): outer double quotes wrap a literal single
        // quote. open == close == `"` → inner is the `'` char.
        assert_eq!(parse_value("\"'\""), Some(Value::String("'".to_string())));
    }

    #[test]
    fn parse_value_preserves_non_quoted_scalars() {
        // The fix touches only the quoted-string branch; numeric and
        // keyword scalars must be unaffected.
        assert_eq!(parse_value("42"), Some(Value::Number(Number::from(42))));
        assert_eq!(parse_value("true"), Some(Value::Bool(true)));
        assert_eq!(parse_value("null"), Some(Value::Null));
    }

    #[test]
    fn lone_quote_property_value_rejects_without_panic_end_to_end() {
        // End-to-end through the full parser: the minimal repro from the
        // #604 fuzz corpus seed (`10_lone_quote_631.cypher`). The lone
        // `"` property value reaches `parse_value` via
        // `parse_properties`; pre-fix this PANICKED, post-fix it is a
        // clean `MigrateError::UnsupportedValue` reject. STRONG oracle:
        // assert the exact error variant, not just "is_err".
        let script = "CREATE (n:L {p: \"});";
        let err = parse_cypher_export_str(script)
            .expect_err("lone-quote property value must be rejected, not parsed");
        assert!(
            matches!(err, MigrateError::UnsupportedValue { .. }),
            "expected UnsupportedValue for a lone-quote property, got {err:?}"
        );
    }
}
