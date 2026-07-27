//! Result-set serializers for the MCP surface.
//!
//! Per design-v2 §9.3, agent-facing query results select a wire format
//! based on shape:
//!   - **TOON** for uniform tabular row sets (≥30% token-savings vs JSON
//!     on the LDBC SNB Person bench shape; design-v2 §9.3 cites 40-60%
//!     vs JSON across the upstream TOON benchmarks).
//!   - **YAML** for nested results that don't fit a tabular schema (the
//!     `graph.schema()` Tier-1 tool returns YAML per design-v2 §9.1
//!     bullet 6, "return the full graph schema (labels, types, property
//!     ranges) as YAML").
//!   - **JSON** as a fallback (delegated to `serde_json::to_string`;
//!     this module does not re-export it — call-sites use it directly).
//!
//! Both encoders pivot through `serde_json::Value` so the `T: Serialize`
//! contract stays uniform across formats and the roundtrip oracle
//! (`encode → decode → Value::eq`) is the same shape.
//!
//! Wave 11 slice ε ships only the M5-09 (TOON) and M5-10 (YAML)
//! sub-tasks of M5; transports / tools / Bolt land in later slices.

pub mod error;
pub mod toon;
pub mod yaml;

pub use error::{SerializerError, ToonError, YamlError};
pub use toon::{from_toon, to_toon};
pub use yaml::{from_yaml, to_yaml};
