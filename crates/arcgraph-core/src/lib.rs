//! Shared primitives for ArcGraph.
//!
//! Scope: ID newtypes, error taxonomy, record layouts, cache-line
//! primitives. No I/O, no locking, no async. Strictly a
//! dependency-free-ish crate that every other crate builds on.
//!
//! See `docs/bounded-contexts.md` for what this crate must not own.

#![recursion_limit = "256"]
// Register the `arcgraph_sim` cfg flag (per ADR-135 D-3 BUGGIFY).
// Production builds (no `--cfg arcgraph_sim` rustflag) elide BUGGIFY
// sites; simulation builds (DST runtime per ADR-134) enable them.
#![cfg_attr(
    arcgraph_sim,
    doc = "arcgraph-core compiled with simulation faults enabled (BUGGIFY active)"
)]

pub mod buggify;
pub mod cache_aligned;
pub mod cost_telemetry;
pub mod datetime;
pub mod durability;
pub mod error;
pub mod ids;
pub mod record;
pub mod secrets;

pub use cache_aligned::CacheAligned;
pub use cost_telemetry::{CostAccumulator, CostSnapshot, PerTenantCostRegistry};
pub use datetime::{
    Date, Decimal, Duration, LocalDateTime, MAX_OFFSET_SECONDS, TemporalError, ZonedDateTime,
    parse_date, parse_decimal, parse_duration, parse_local_datetime, parse_zoned_datetime,
};
pub use durability::{AlwaysStrict, DurabilityTier, DurabilityTierError, TenantDurabilityLookup};
pub use error::{ArcGraphError, Result};
pub use ids::{
    LabelId, Lsn, NodeId, PageId, PartitionId, PropertyId, RelId, StringId, TenantId, TypeId,
};
pub use record::{NodeRecord, PAGE_SIZE, PageHeader, PageType, RelRecord, TelEntry};
#[cfg(feature = "os-keyring")]
pub use secrets::OsKeyringProvider;
pub use secrets::{
    ENCRYPTION_KEY_NAMESPACE_WAL, EnvSecretsProvider, KekVersion, KeyScope, KeySource,
    KeySourceError, KeyVersion, SECRET_VALUE_LEN, SecretValue, SecretsError, SecretsProvider,
    WrapAlg, WrappedDek,
};
