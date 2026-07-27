//! Error taxonomy for the M5-09 / M5-10 serializers.
//!
//! Two private error types (`ToonError`, `YamlError`) live next to their
//! respective modules; this module owns the public `SerializerError`
//! umbrella that callers in tools / transports unify against.
//!
//! Both inner types carry a line-number-precision context where the
//! upstream parser supplies one — TOON's hand-rolled lexer attaches
//! `line` to every diagnostic, and `serde_yaml` ships its own
//! `Location` we forward verbatim.

use thiserror::Error;

/// TOON-specific encode/decode failures.
///
/// Per spec §14 (Strict Mode), every recognized failure mode gets its
/// own variant so callers can branch on it without re-parsing the error
/// message. New variants may be added without breaking SemVer thanks to
/// `#[non_exhaustive]`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ToonError {
    /// `serde_json::to_value(value)` failed during the
    /// `T: Serialize` → `Value` pivot. Always wraps an internal
    /// serialization issue (e.g., a `Serialize` impl that returns an
    /// error mid-stream). Inner `serde_json::Error` is preserved.
    #[error("serde pivot failed: {0}")]
    SerdePivot(#[from] serde_json::Error),

    /// `serde_json::from_value(value)` failed when materializing the
    /// decoded `Value` into the caller's `DeserializeOwned` type. The
    /// TOON text parsed cleanly but the resulting JSON shape didn't
    /// match `T`'s expectations.
    #[error("decode-target mismatch: {0}")]
    DecodeTarget(serde_json::Error),

    /// A line's leading-whitespace count was not a multiple of the
    /// active `indentSize` (=2 in this implementation), or contained a
    /// tab character. Per spec §14.3 strict mode rejects this.
    #[error("line {line}: indentation must be a multiple of {indent_size} spaces (no tabs)")]
    Indent { line: usize, indent_size: usize },

    /// A TOON line didn't match any of the recognized productions
    /// (header / kv / list-item / bare-scalar). The `reason` field
    /// gives the most-specific cause the lexer was able to identify.
    #[error("line {line}: malformed TOON ({reason})")]
    Malformed { line: usize, reason: String },

    /// Declared array length `[N]` did not match the number of rows /
    /// inline values present. Spec §14.1 strict-mode array width
    /// mismatch.
    #[error("line {line}: array count mismatch — declared {declared}, observed {observed}")]
    CountMismatch {
        line: usize,
        declared: usize,
        observed: usize,
    },

    /// A tabular row's cell count didn't match the field-header width.
    /// Spec §14.1 strict-mode row width mismatch.
    #[error("line {line}: tabular row width mismatch — header={header}, row={row}")]
    RowWidthMismatch {
        line: usize,
        header: usize,
        row: usize,
    },

    /// A quoted string contained an escape outside the spec's allowed
    /// set (`\\`, `\"`, `\n`, `\r`, `\t`) or terminated mid-string.
    #[error("line {line}: invalid string escape or unterminated literal")]
    InvalidString { line: usize },

    /// Encoder rejected an input it cannot represent in canonical TOON
    /// (e.g., a JSON object key the spec forbids — currently unused;
    /// reserved for future canonical-form tightening).
    #[error("cannot encode value: {0}")]
    Unencodable(String),
}

/// YAML-specific encode/decode failures.
///
/// Currently both arms wrap `serde_yaml::Error` — kept distinct so
/// callers can tell encode-side from decode-side problems without
/// inspecting the underlying error string. `#[non_exhaustive]` reserves
/// room for future YAML-specific surface (e.g., explicit indent/style
/// budget violations) per the same SemVer policy as `ToonError`.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum YamlError {
    /// `serde_yaml::to_string(value)` failed during encode.
    #[error("yaml encode failed: {0}")]
    Encode(serde_yaml::Error),

    /// `serde_yaml::from_str(text)` failed during decode.
    #[error("yaml decode failed: {0}")]
    Decode(serde_yaml::Error),
}

/// Public umbrella the MCP surface unifies against.
///
/// Exists so consumers (tools / transports) can pattern-match on
/// "format-class" without depending on the concrete TOON / YAML error
/// shapes. Per the M5 contract surface this will gain a `Json` arm
/// when M5-08 / serializer-fallback wiring lands.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SerializerError {
    #[error("TOON: {0}")]
    Toon(#[from] ToonError),

    #[error("YAML: {0}")]
    Yaml(#[from] YamlError),
}
