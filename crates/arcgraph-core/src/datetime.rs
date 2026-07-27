//! Ordinary date, time, duration, and decimal value primitives.
//!
//! These types are scalar query values. They do not model entity
//! history, version reconstruction, or transaction visibility.
//!
//! # Type taxonomy
//!
//! - [`ZonedDateTime`] — UTC nanoseconds-since-epoch (`i64`) + UTC
//!   offset (`i32` seconds; `-50_400 ..= 50_400`, i.e. ±14h). Closed
//!   under serialization. Total ordering by UTC instant.
//! - [`LocalDateTime`] — wall-clock with NO zone information. Stored
//!   as a `(year, ordinal, nano_of_day)` tuple to avoid mis-fold
//!   semantics across DST boundaries. These are property values that
//!   don't carry a zone.
//! - [`Date`] — calendar date as `(year, ordinal)`. ISO-8601 proleptic
//!   Gregorian.
//! - [`Duration`] — ISO-8601 duration. Stored as `(months, nanos)`
//!   because calendar-aware durations (PT1M30S vs P1M for "1 month")
//!   are NOT equivalent in nanos — months are calendar-resolved at
//!   apply time, not at construction.
//! - [`Decimal`] — fixed-point decimal as `(scale: i8, units: i128)`.
//!   `value = units / 10^scale`. Companion to V11-T-02 atomic landing.
//!
//! # Why not chrono / jiff?
//!
//! - **chrono** is dual-MIT/Apache-2.0 (license-clean per
//!   `deny.toml`). It IS in `Cargo.lock` as a transitive dep but no
//!   workspace crate consumes it directly today. Adding it as a
//!   direct `arcgraph-core` dep would (a) bloat the bounded-context
//!   surface, (b) make every dependent crate inherit its
//!   `iana-time-zone` chain, and (c) tie the database wire format
//!   to chrono's evolving `DateTime<Tz>` representation. A thin
//!   newtype around `i64` provides a stable wire representation.
//! - **jiff** is the modern alternative (per K3 §7) but introduces a
//!   3rd-party time-zone DB at build time; deferred until the
//!   date/time surface needs IANA-aware time-zone
//!   conversions (intervals like `P1M` evaluated against
//!   `America/New_York` cross DST). The current wire surface uses
//!   fixed-offset zones only.

use std::fmt;

use thiserror::Error;

// =====================================================================
// 1. ZonedDateTime — wall-clock + offset
// =====================================================================

/// Maximum admissible UTC offset, in seconds (±14h per ISO-8601).
///
/// Real-world max is +14:00 (Line Islands, Kiribati) and -12:00
/// (Baker Island); ISO-8601 permits ±14:00 explicitly. We accept the
/// symmetric ±14:00 envelope.
pub const MAX_OFFSET_SECONDS: i32 = 14 * 3600;

/// A wall-clock instant + UTC offset. The on-wire representation is
/// the canonical "UTC nanoseconds since Unix epoch" + "offset in
/// seconds" pair; downstream consumers reconstruct the local
/// wall-clock as `utc_nanos + offset_seconds * 1_000_000_000`.
///
/// Equality compares the underlying UTC instant (NOT the offset);
/// `2026-05-24T12:00:00Z` and `2026-05-24T13:00:00+01:00` are equal.
/// Ordering is by UTC instant.
///
/// # Range
///
/// - `utc_nanos` is `i64` → range `[1677-09-21T00:12:43.145224192Z,
///   2262-04-11T23:47:16.854775807Z]`.
/// - `offset_seconds` ∈ `[-50_400, 50_400]` (±14h).
#[derive(Debug, Clone, Copy, PartialOrd, Ord)]
pub struct ZonedDateTime {
    /// UTC nanoseconds since Unix epoch (1970-01-01T00:00:00Z).
    utc_nanos: i64,
    /// Offset from UTC, in seconds. `+3600` = UTC+01:00.
    offset_seconds: i32,
}

impl ZonedDateTime {
    /// Construct a ZonedDateTime from a UTC instant + offset.
    ///
    /// # Errors
    ///
    /// - `TemporalError::OffsetOutOfRange` if `|offset_seconds| > 14*3600`.
    pub const fn from_utc_nanos_and_offset(
        utc_nanos: i64,
        offset_seconds: i32,
    ) -> Result<Self, TemporalError> {
        if offset_seconds < -MAX_OFFSET_SECONDS || offset_seconds > MAX_OFFSET_SECONDS {
            return Err(TemporalError::OffsetOutOfRange { offset_seconds });
        }
        Ok(Self {
            utc_nanos,
            offset_seconds,
        })
    }

    /// Construct a ZonedDateTime at UTC (`Z` offset).
    #[must_use]
    pub const fn from_utc_nanos(utc_nanos: i64) -> Self {
        Self {
            utc_nanos,
            offset_seconds: 0,
        }
    }

    /// UTC instant, in nanoseconds since Unix epoch.
    #[inline]
    #[must_use]
    pub const fn utc_nanos(self) -> i64 {
        self.utc_nanos
    }

    /// UTC offset, in seconds.
    #[inline]
    #[must_use]
    pub const fn offset_seconds(self) -> i32 {
        self.offset_seconds
    }
}

impl PartialEq for ZonedDateTime {
    fn eq(&self, other: &Self) -> bool {
        // Equality is by UTC instant only — offset is presentation.
        self.utc_nanos == other.utc_nanos
    }
}

impl Eq for ZonedDateTime {}

impl std::hash::Hash for ZonedDateTime {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Mirror Eq: hash by UTC instant only.
        self.utc_nanos.hash(state);
    }
}

impl fmt::Display for ZonedDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (y, ord, h, m, s, ns) =
            utc_breakdown(self.utc_nanos + (self.offset_seconds as i64) * 1_000_000_000);
        let (mo, day) = ordinal_to_month_day(y, ord);
        if self.offset_seconds == 0 {
            write!(
                f,
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
                y, mo, day, h, m, s, ns
            )
        } else {
            let sign = if self.offset_seconds >= 0 { '+' } else { '-' };
            let off_abs = self.offset_seconds.unsigned_abs() as i32;
            let off_h = off_abs / 3600;
            let off_m = (off_abs % 3600) / 60;
            write!(
                f,
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}{}{:02}:{:02}",
                y, mo, day, h, m, s, ns, sign, off_h, off_m
            )
        }
    }
}

// =====================================================================
// 2. LocalDateTime — wall-clock with no zone
// =====================================================================

/// Local wall-clock with NO zone. Stored as a structured tuple
/// (`year`, `ordinal`, `nano_of_day`) to avoid DST-fold ambiguity.
///
/// Per K3 §2.3 "Gap shape" item 1: `LocalDateTime` is the
/// wall-clock-without-zone counterpart to `ZonedDateTime` (the
/// Cypher-9-aligned taxonomy). Used for catalog rows where the
/// zone is implicit (e.g., business-hours configuration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalDateTime {
    /// Proleptic Gregorian year. Range: `i32::MIN ..= i32::MAX`
    /// in principle; v1.1 narrows to `[-9999, 9999]` per the ISO-8601
    /// 4-digit-year convention (extended forms via `±YYYYYY` deferred).
    pub year: i32,
    /// Day-of-year, 1-indexed (1..=366).
    pub ordinal: u16,
    /// Nanoseconds since midnight, `0..=86_399_999_999_999` (incl. a
    /// 60-th leap second admission for ISO-8601 conformity; we DON'T
    /// admit ≥60 second values — leap-seconds are smeared per the
    /// IETF recommendation).
    pub nano_of_day: u64,
}

impl LocalDateTime {
    /// Construct, validating ordinal + nano_of_day envelopes.
    pub fn new(year: i32, ordinal: u16, nano_of_day: u64) -> Result<Self, TemporalError> {
        let max_ord = if is_leap_year(year) { 366 } else { 365 };
        if ordinal == 0 || ordinal > max_ord {
            return Err(TemporalError::OrdinalOutOfRange { year, ordinal });
        }
        if nano_of_day >= 86_400_000_000_000 {
            return Err(TemporalError::NanoOfDayOutOfRange { nano_of_day });
        }
        Ok(Self {
            year,
            ordinal,
            nano_of_day,
        })
    }
}

impl fmt::Display for LocalDateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (mo, day) = ordinal_to_month_day(self.year, self.ordinal);
        let secs = self.nano_of_day / 1_000_000_000;
        let ns = self.nano_of_day % 1_000_000_000;
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}",
            self.year, mo, day, h, m, s, ns
        )
    }
}

// =====================================================================
// 3. Date — calendar date, no time / zone
// =====================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Date {
    pub year: i32,
    /// Day-of-year, 1-indexed.
    pub ordinal: u16,
}

impl Date {
    pub fn new(year: i32, ordinal: u16) -> Result<Self, TemporalError> {
        let max_ord = if is_leap_year(year) { 366 } else { 365 };
        if ordinal == 0 || ordinal > max_ord {
            return Err(TemporalError::OrdinalOutOfRange { year, ordinal });
        }
        Ok(Self { year, ordinal })
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (mo, day) = ordinal_to_month_day(self.year, self.ordinal);
        write!(f, "{:04}-{:02}-{:02}", self.year, mo, day)
    }
}

// =====================================================================
// 4. Duration — ISO-8601 duration
// =====================================================================

/// ISO-8601 duration. Months are NOT canonicalized to nanos because
/// calendar-aware durations resolve at apply-time, not at
/// construction. `P1M` applied to `2026-01-31` yields `2026-02-28`
/// (or `2026-02-29` in a leap year) — the underlying month count is
/// stable; the resolved nanos depend on the base date.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Duration {
    /// Months component, signed. Can be negative (for `-P1M` etc.).
    pub months: i32,
    /// Nanoseconds component, signed. Includes days × 86_400 × 1e9 +
    /// hours / minutes / seconds / sub-second nanos.
    pub nanos: i64,
}

impl Duration {
    pub const fn new(months: i32, nanos: i64) -> Self {
        Self { months, nanos }
    }

    pub const fn from_nanos(nanos: i64) -> Self {
        Self { months: 0, nanos }
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Canonical ISO-8601 form: `P[nM][nDTnHnMnS]`. We emit the
        // sign-prefixed form when either component is negative.
        let sign = if self.months < 0 || self.nanos < 0 {
            "-"
        } else {
            ""
        };
        let m = self.months.unsigned_abs();
        let n = self.nanos.unsigned_abs();
        let secs = n / 1_000_000_000;
        let ns = n % 1_000_000_000;
        let days = secs / 86_400;
        let rem = secs % 86_400;
        let h = rem / 3600;
        let r2 = rem % 3600;
        let mi = r2 / 60;
        let se = r2 % 60;
        write!(f, "{sign}P")?;
        if m > 0 {
            write!(f, "{m}M")?;
        }
        if days > 0 {
            write!(f, "{days}D")?;
        }
        if h > 0 || mi > 0 || se > 0 || ns > 0 {
            write!(f, "T")?;
            if h > 0 {
                write!(f, "{h}H")?;
            }
            if mi > 0 {
                write!(f, "{mi}M")?;
            }
            if se > 0 || ns > 0 {
                if ns > 0 {
                    let mut buf = format!("{ns:09}");
                    while buf.ends_with('0') {
                        buf.pop();
                    }
                    write!(f, "{se}.{buf}S")?;
                } else {
                    write!(f, "{se}S")?;
                }
            }
        }
        if m == 0 && days == 0 && h == 0 && mi == 0 && se == 0 && ns == 0 {
            // Degenerate zero duration — emit `PT0S` per ISO-8601.
            write!(f, "T0S")?;
        }
        Ok(())
    }
}

// =====================================================================
// 5. Decimal — fixed-point decimal (V11-T-02 companion landing)
// =====================================================================

/// Fixed-point decimal. `value = units / 10^scale`. Scale `[0, 38]`
/// per the openCypher / GQL DECIMAL surface; the query value accepts
/// up to 38 (matching ANSI SQL / Snowflake / BigQuery; deeper scales
/// admitted at v1.2+ when the executor's arithmetic kernel pulls in
/// a 256-bit fixed-point library).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Decimal {
    /// Number of decimal digits after the point. Range `[0, 38]`.
    pub scale: i8,
    /// Unscaled integer value.
    pub units: i128,
}

impl Decimal {
    pub fn new(scale: i8, units: i128) -> Result<Self, TemporalError> {
        if !(0..=38).contains(&scale) {
            return Err(TemporalError::DecimalScaleOutOfRange { scale });
        }
        Ok(Self { scale, units })
    }
}

impl fmt::Display for Decimal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scale == 0 {
            return write!(f, "{}", self.units);
        }
        let sign = if self.units < 0 { "-" } else { "" };
        let mag = self.units.unsigned_abs();
        let divisor: u128 = 10u128.pow(self.scale as u32);
        let int_part = mag / divisor;
        let frac_part = mag % divisor;
        let mut frac_str = format!("{frac_part:0width$}", width = self.scale as usize);
        while frac_str.ends_with('0') && frac_str.len() > 1 {
            frac_str.pop();
        }
        if frac_str == "0" {
            write!(f, "{sign}{int_part}")
        } else {
            write!(f, "{sign}{int_part}.{frac_str}")
        }
    }
}

// =====================================================================
// 6. Error taxonomy
// =====================================================================

/// Errors surfaced at temporal-value construction + parsing.
///
/// `#[non_exhaustive]` permits adding a new variant (e.g., a future
/// `LeapSecondAmbiguity` when the
/// library's leap-second policy moves from "smear" to "explicit")
/// is not a SemVer break.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum TemporalError {
    #[error("UTC offset out of range: {offset_seconds}s (envelope: ±50_400s = ±14h)")]
    OffsetOutOfRange { offset_seconds: i32 },

    #[error("ordinal out of range for year {year}: {ordinal} (valid: 1..=365 / 366)")]
    OrdinalOutOfRange { year: i32, ordinal: u16 },

    #[error(
        "nano-of-day out of range: {nano_of_day} (valid: 0..86_400_000_000_000; \
         leap-seconds smeared per date/time policy)"
    )]
    NanoOfDayOutOfRange { nano_of_day: u64 },

    #[error("decimal scale out of range: {scale} (valid: 0..=38)")]
    DecimalScaleOutOfRange { scale: i8 },

    #[error("malformed temporal literal: {message}")]
    Malformed { message: String },
}

// =====================================================================
// 7. ISO-8601 parsers (RFC-3339 subset; production callers route via
//    grammar.pest's datetime(...) / date(...) / etc. literal arms)
// =====================================================================

/// Parse an RFC-3339 / ISO-8601 zoned datetime: `YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]`.
///
/// Accepted forms:
/// - `2026-05-24T12:00:00Z`
/// - `2026-05-24T12:00:00+02:00`
/// - `2026-05-24T12:00:00.123Z` (fractional seconds, 1-9 digits)
///
/// # Errors
///
/// - `TemporalError::Malformed` for any structural violation.
pub fn parse_zoned_datetime(s: &str) -> Result<ZonedDateTime, TemporalError> {
    let (local_part, tz_part) = split_tz_suffix(s).ok_or_else(|| TemporalError::Malformed {
        message: format!("zoned datetime missing timezone suffix: {s}"),
    })?;
    let ldt = parse_local_datetime(local_part)?;
    let offset_seconds = parse_tz_offset(tz_part)?;
    let utc_nanos = local_to_utc_nanos(&ldt) - (offset_seconds as i64) * 1_000_000_000;
    ZonedDateTime::from_utc_nanos_and_offset(utc_nanos, offset_seconds)
}

/// Parse a local-datetime literal: `YYYY-MM-DDTHH:MM:SS[.fff]` (no zone).
pub fn parse_local_datetime(s: &str) -> Result<LocalDateTime, TemporalError> {
    let (date_part, time_part) = s.split_once('T').ok_or_else(|| TemporalError::Malformed {
        message: format!("local datetime missing 'T': {s}"),
    })?;
    let d = parse_date(date_part)?;
    let nano_of_day = parse_time_of_day(time_part)?;
    LocalDateTime::new(d.year, d.ordinal, nano_of_day)
}

/// Parse a date literal: `YYYY-MM-DD`.
pub fn parse_date(s: &str) -> Result<Date, TemporalError> {
    let parts: Vec<&str> = s.split('-').collect();
    // Negative-year forms ("-0001-01-01") split into 4 parts: ["", "0001", "01", "01"].
    let (year, mo, day) = if parts.len() == 3 {
        (
            parts[0]
                .parse::<i32>()
                .map_err(|_| TemporalError::Malformed {
                    message: format!("date year: {s}"),
                })?,
            parts[1]
                .parse::<u8>()
                .map_err(|_| TemporalError::Malformed {
                    message: format!("date month: {s}"),
                })?,
            parts[2]
                .parse::<u8>()
                .map_err(|_| TemporalError::Malformed {
                    message: format!("date day: {s}"),
                })?,
        )
    } else if parts.len() == 4 && parts[0].is_empty() {
        (
            -parts[1]
                .parse::<i32>()
                .map_err(|_| TemporalError::Malformed {
                    message: format!("date year: {s}"),
                })?,
            parts[2]
                .parse::<u8>()
                .map_err(|_| TemporalError::Malformed {
                    message: format!("date month: {s}"),
                })?,
            parts[3]
                .parse::<u8>()
                .map_err(|_| TemporalError::Malformed {
                    message: format!("date day: {s}"),
                })?,
        )
    } else {
        return Err(TemporalError::Malformed {
            message: format!("date should be YYYY-MM-DD: {s}"),
        });
    };
    if !(1..=12).contains(&mo) {
        return Err(TemporalError::Malformed {
            message: format!("date month out of range: {mo}"),
        });
    }
    let dim = days_in_month(year, mo);
    if !(1..=dim).contains(&day) {
        return Err(TemporalError::Malformed {
            message: format!("date day out of range: {day}/{mo}/{year}"),
        });
    }
    let ord = month_day_to_ordinal(year, mo, day);
    Date::new(year, ord)
}

/// Parse an ISO-8601 duration: `P[nM][nDTnHnMnS]`. Sign prefix `-`
/// admitted for negative durations.
pub fn parse_duration(s: &str) -> Result<Duration, TemporalError> {
    let (negative, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else {
        (false, s)
    };
    let rest = rest
        .strip_prefix('P')
        .ok_or_else(|| TemporalError::Malformed {
            message: format!("duration must start with 'P': {s}"),
        })?;
    let (date_part, time_part) = match rest.split_once('T') {
        Some((d, t)) => (d, t),
        None => (rest, ""),
    };
    let mut months: i64 = 0;
    let mut days: i64 = 0;
    parse_duration_date_part(date_part, &mut months, &mut days, s)?;
    let mut nanos: i128 = (days as i128) * 86_400 * 1_000_000_000;
    parse_duration_time_part(time_part, &mut nanos, s)?;
    let nanos_i64: i64 = nanos.try_into().map_err(|_| TemporalError::Malformed {
        message: format!("duration nanos overflow i64: {s}"),
    })?;
    let months_i32: i32 = months.try_into().map_err(|_| TemporalError::Malformed {
        message: format!("duration months overflow i32: {s}"),
    })?;
    let d = Duration::new(months_i32, nanos_i64);
    if negative {
        Ok(Duration::new(-d.months, -d.nanos))
    } else {
        Ok(d)
    }
}

fn parse_duration_date_part(
    s: &str,
    months: &mut i64,
    days: &mut i64,
    full: &str,
) -> Result<(), TemporalError> {
    let mut acc = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() {
            acc.push(c);
        } else {
            let n: i64 = acc.parse().map_err(|_| TemporalError::Malformed {
                message: format!("duration date number: {full}"),
            })?;
            acc.clear();
            match c {
                'Y' => *months += n * 12,
                'M' => *months += n,
                'D' => *days += n,
                'W' => *days += n * 7,
                _ => {
                    return Err(TemporalError::Malformed {
                        message: format!("duration unknown date unit '{c}': {full}"),
                    });
                }
            }
        }
    }
    if !acc.is_empty() {
        return Err(TemporalError::Malformed {
            message: format!("duration date part trailing digits: {full}"),
        });
    }
    Ok(())
}

fn parse_duration_time_part(s: &str, nanos: &mut i128, full: &str) -> Result<(), TemporalError> {
    if s.is_empty() {
        return Ok(());
    }
    let mut acc = String::new();
    for c in s.chars() {
        if c.is_ascii_digit() || c == '.' {
            acc.push(c);
        } else {
            let f: f64 = acc.parse().map_err(|_| TemporalError::Malformed {
                message: format!("duration time number: {full}"),
            })?;
            acc.clear();
            match c {
                'H' => *nanos += (f * 3600.0 * 1_000_000_000.0) as i128,
                'M' => *nanos += (f * 60.0 * 1_000_000_000.0) as i128,
                'S' => *nanos += (f * 1_000_000_000.0) as i128,
                _ => {
                    return Err(TemporalError::Malformed {
                        message: format!("duration unknown time unit '{c}': {full}"),
                    });
                }
            }
        }
    }
    if !acc.is_empty() {
        return Err(TemporalError::Malformed {
            message: format!("duration time part trailing digits: {full}"),
        });
    }
    Ok(())
}

/// Parse a decimal literal: `[-]<int>[.<frac>]`. The scale is the
/// number of fractional digits; `units = int*10^scale + frac` (sign-
/// adjusted).
pub fn parse_decimal(s: &str) -> Result<Decimal, TemporalError> {
    let (negative, rest) = if let Some(r) = s.strip_prefix('-') {
        (true, r)
    } else {
        (false, s)
    };
    let (int_part, frac_part) = rest.split_once('.').unwrap_or((rest, ""));
    if int_part.is_empty() || !int_part.chars().all(|c| c.is_ascii_digit()) {
        return Err(TemporalError::Malformed {
            message: format!("decimal int part: {s}"),
        });
    }
    if !frac_part.chars().all(|c| c.is_ascii_digit()) {
        return Err(TemporalError::Malformed {
            message: format!("decimal frac part: {s}"),
        });
    }
    let scale: i8 = frac_part
        .len()
        .try_into()
        .map_err(|_| TemporalError::Malformed {
            message: format!("decimal scale: {s}"),
        })?;
    if scale > 38 {
        return Err(TemporalError::DecimalScaleOutOfRange { scale });
    }
    let mut concat = String::with_capacity(int_part.len() + frac_part.len());
    concat.push_str(int_part);
    concat.push_str(frac_part);
    let mag: i128 = concat.parse().map_err(|_| TemporalError::Malformed {
        message: format!("decimal magnitude overflow i128: {s}"),
    })?;
    let units = if negative { -mag } else { mag };
    Decimal::new(scale, units)
}

// =====================================================================
// 8. Pure helpers (calendar math)
// =====================================================================

const fn is_leap_year(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

const fn days_in_month(year: i32, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap_year(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

fn month_day_to_ordinal(year: i32, month: u8, day: u8) -> u16 {
    let mut ord: u16 = 0;
    for m in 1..month {
        ord += days_in_month(year, m) as u16;
    }
    ord + day as u16
}

fn ordinal_to_month_day(year: i32, ordinal: u16) -> (u8, u8) {
    let mut remaining = ordinal;
    for m in 1u8..=12 {
        let dim = days_in_month(year, m) as u16;
        if remaining <= dim {
            return (m, remaining as u8);
        }
        remaining -= dim;
    }
    (12, 31)
}

/// Convert a UTC LocalDateTime (i.e., a wall-clock at offset 0) to
/// UTC nanoseconds since 1970-01-01.
fn local_to_utc_nanos(ldt: &LocalDateTime) -> i64 {
    // Days since 1970-01-01.
    let days_to_year = days_since_epoch_to_year_start(ldt.year);
    let days = days_to_year + (ldt.ordinal as i64) - 1;
    days * 86_400 * 1_000_000_000 + ldt.nano_of_day as i64
}

/// Days from 1970-01-01 to the start of `year` (1970-01-01 itself).
fn days_since_epoch_to_year_start(year: i32) -> i64 {
    if year >= 1970 {
        let mut d: i64 = 0;
        for y in 1970..year {
            d += if is_leap_year(y) { 366 } else { 365 };
        }
        d
    } else {
        let mut d: i64 = 0;
        for y in year..1970 {
            d -= if is_leap_year(y) { 366 } else { 365 };
        }
        d
    }
}

/// Reverse: UTC nanos → (year, ordinal, h, m, s, ns).
fn utc_breakdown(utc_nanos: i64) -> (i32, u16, u8, u8, u8, u32) {
    let secs_total = utc_nanos.div_euclid(1_000_000_000);
    let ns = utc_nanos.rem_euclid(1_000_000_000) as u32;
    let mut days = secs_total.div_euclid(86_400);
    let secs_of_day = secs_total.rem_euclid(86_400);
    let h = (secs_of_day / 3600) as u8;
    let m = ((secs_of_day % 3600) / 60) as u8;
    let s = (secs_of_day % 60) as u8;
    // Find year by counting days.
    let mut y: i32 = 1970;
    if days >= 0 {
        loop {
            let dy = if is_leap_year(y) { 366 } else { 365 };
            if days < dy {
                break;
            }
            days -= dy;
            y += 1;
        }
    } else {
        while days < 0 {
            y -= 1;
            let dy = if is_leap_year(y) { 366 } else { 365 };
            days += dy;
        }
    }
    let ordinal = (days + 1) as u16;
    (y, ordinal, h, m, s, ns)
}

fn split_tz_suffix(s: &str) -> Option<(&str, &str)> {
    if let Some(pos) = s.rfind('Z') {
        if pos == s.len() - 1 {
            return Some((&s[..pos], "Z"));
        }
    }
    // Find last '+' or '-' that is positioned after the 'T' (so we
    // don't mistake the date-component separator '-' for the offset).
    let t_pos = s.find('T').unwrap_or(0);
    for (i, c) in s.char_indices().rev() {
        if i > t_pos && (c == '+' || c == '-') {
            return Some((&s[..i], &s[i..]));
        }
    }
    None
}

fn parse_tz_offset(s: &str) -> Result<i32, TemporalError> {
    if s == "Z" {
        return Ok(0);
    }
    let bytes = s.as_bytes();
    if bytes.len() != 6 || (bytes[0] != b'+' && bytes[0] != b'-') || bytes[3] != b':' {
        return Err(TemporalError::Malformed {
            message: format!("tz offset not ±HH:MM: {s}"),
        });
    }
    let sign: i32 = if bytes[0] == b'+' { 1 } else { -1 };
    let h: i32 = std::str::from_utf8(&bytes[1..3])
        .map_err(|_| TemporalError::Malformed {
            message: format!("tz offset hours: {s}"),
        })?
        .parse()
        .map_err(|_| TemporalError::Malformed {
            message: format!("tz offset hours: {s}"),
        })?;
    let m: i32 = std::str::from_utf8(&bytes[4..6])
        .map_err(|_| TemporalError::Malformed {
            message: format!("tz offset minutes: {s}"),
        })?
        .parse()
        .map_err(|_| TemporalError::Malformed {
            message: format!("tz offset minutes: {s}"),
        })?;
    let secs = sign * (h * 3600 + m * 60);
    if secs.abs() > MAX_OFFSET_SECONDS {
        return Err(TemporalError::OffsetOutOfRange {
            offset_seconds: secs,
        });
    }
    Ok(secs)
}

fn parse_time_of_day(s: &str) -> Result<u64, TemporalError> {
    let parts: Vec<&str> = s.split(':').collect();
    if parts.len() != 3 {
        return Err(TemporalError::Malformed {
            message: format!("time-of-day not HH:MM:SS[.f]: {s}"),
        });
    }
    let h: u64 = parts[0].parse().map_err(|_| TemporalError::Malformed {
        message: format!("time hour: {s}"),
    })?;
    let m: u64 = parts[1].parse().map_err(|_| TemporalError::Malformed {
        message: format!("time minute: {s}"),
    })?;
    let (sec_part, frac_part) = parts[2].split_once('.').unwrap_or((parts[2], ""));
    let sec: u64 = sec_part.parse().map_err(|_| TemporalError::Malformed {
        message: format!("time second: {s}"),
    })?;
    let ns: u64 = if frac_part.is_empty() {
        0
    } else if frac_part.len() > 9 {
        return Err(TemporalError::Malformed {
            message: format!("time fractional > 9 digits: {s}"),
        });
    } else {
        let pad = 9 - frac_part.len();
        let mut padded = String::with_capacity(9);
        padded.push_str(frac_part);
        for _ in 0..pad {
            padded.push('0');
        }
        padded.parse().map_err(|_| TemporalError::Malformed {
            message: format!("time fractional: {s}"),
        })?
    };
    if h >= 24 || m >= 60 || sec >= 60 {
        return Err(TemporalError::Malformed {
            message: format!("time component out of range: {s}"),
        });
    }
    Ok(h * 3600 * 1_000_000_000 + m * 60 * 1_000_000_000 + sec * 1_000_000_000 + ns)
}

// =====================================================================
// 9. Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoned_datetime_parses_utc_z() {
        let zdt = parse_zoned_datetime("2026-05-24T12:00:00Z").expect("parse");
        assert_eq!(zdt.offset_seconds(), 0);
        // 2026-05-24 is ordinal 144 (Jan=31, Feb=28, Mar=31, Apr=30 = 120; + 24 = 144).
        let (y, ord, h, m, s, ns) = utc_breakdown(zdt.utc_nanos());
        assert_eq!(y, 2026);
        assert_eq!(ord, 144);
        assert_eq!((h, m, s, ns), (12, 0, 0, 0));
    }

    #[test]
    fn zoned_datetime_parses_positive_offset() {
        let zdt = parse_zoned_datetime("2026-05-24T13:00:00+01:00").expect("parse");
        // Equivalent to UTC noon.
        let utc = parse_zoned_datetime("2026-05-24T12:00:00Z").expect("parse");
        assert_eq!(zdt, utc, "+01:00 13:00 should equal UTC 12:00");
        assert_eq!(zdt.offset_seconds(), 3600);
    }

    #[test]
    fn zoned_datetime_parses_negative_offset() {
        let zdt = parse_zoned_datetime("2026-05-24T07:00:00-05:00").expect("parse");
        // Equivalent to UTC noon.
        let utc = parse_zoned_datetime("2026-05-24T12:00:00Z").expect("parse");
        assert_eq!(zdt, utc, "-05:00 07:00 should equal UTC 12:00");
    }

    #[test]
    fn zoned_datetime_equality_is_by_utc_instant() {
        let a = parse_zoned_datetime("2026-05-24T13:00:00+01:00").unwrap();
        let b = parse_zoned_datetime("2026-05-24T12:00:00Z").unwrap();
        let c = parse_zoned_datetime("2026-05-24T07:00:00-05:00").unwrap();
        assert_eq!(a, b);
        assert_eq!(b, c);
        assert_eq!(a, c);
    }

    #[test]
    fn zoned_datetime_offset_out_of_range_rejected() {
        let result = ZonedDateTime::from_utc_nanos_and_offset(0, 15 * 3600);
        assert!(matches!(
            result,
            Err(TemporalError::OffsetOutOfRange { .. })
        ));
    }

    #[test]
    fn zoned_datetime_fractional_seconds_admitted() {
        let zdt = parse_zoned_datetime("2026-05-24T12:00:00.123456789Z").expect("parse");
        let (_, _, _, _, _, ns) = utc_breakdown(zdt.utc_nanos());
        assert_eq!(ns, 123_456_789);
    }

    #[test]
    fn date_parses_and_round_trips() {
        let d = parse_date("2026-05-24").expect("parse");
        assert_eq!(d.year, 2026);
        assert_eq!(d.ordinal, 144);
        assert_eq!(format!("{d}"), "2026-05-24");
    }

    #[test]
    fn date_february_29_admitted_in_leap_year() {
        let d = parse_date("2024-02-29").expect("parse leap-day");
        assert_eq!(d.year, 2024);
    }

    #[test]
    fn date_february_29_rejected_in_non_leap_year() {
        let result = parse_date("2026-02-29");
        assert!(result.is_err());
    }

    #[test]
    fn local_datetime_parses() {
        let ldt = parse_local_datetime("2026-05-24T12:00:00").expect("parse");
        assert_eq!(ldt.year, 2026);
        assert_eq!(ldt.ordinal, 144);
        assert_eq!(ldt.nano_of_day, 12 * 3600 * 1_000_000_000);
    }

    #[test]
    fn duration_parses_iso_8601() {
        let d = parse_duration("PT1H30M").expect("parse");
        assert_eq!(d.months, 0);
        assert_eq!(d.nanos, (3600 + 30 * 60) * 1_000_000_000);
    }

    #[test]
    fn duration_parses_with_months() {
        let d = parse_duration("P1Y6M").expect("parse");
        assert_eq!(d.months, 18);
        assert_eq!(d.nanos, 0);
    }

    #[test]
    fn duration_parses_with_days_and_time() {
        let d = parse_duration("P1DT2H3M4.5S").expect("parse");
        assert_eq!(d.months, 0);
        let expected = 86_400 * 1_000_000_000
            + 2 * 3600 * 1_000_000_000
            + 3 * 60 * 1_000_000_000
            + 4 * 1_000_000_000
            + 500_000_000;
        assert_eq!(d.nanos, expected);
    }

    #[test]
    fn duration_parses_negative() {
        let d = parse_duration("-PT1H").expect("parse");
        assert_eq!(d.nanos, -3600 * 1_000_000_000);
    }

    #[test]
    fn duration_round_trips_display() {
        let d = parse_duration("PT1H30M").expect("parse");
        // Display canonical form.
        assert_eq!(format!("{d}"), "PT1H30M");
    }

    #[test]
    fn decimal_parses_simple() {
        let d = parse_decimal("100.50").expect("parse");
        assert_eq!(d.scale, 2);
        assert_eq!(d.units, 10050);
        assert_eq!(format!("{d}"), "100.5");
    }

    #[test]
    fn decimal_parses_negative() {
        let d = parse_decimal("-3.14").expect("parse");
        assert_eq!(d.scale, 2);
        assert_eq!(d.units, -314);
    }

    #[test]
    fn decimal_parses_no_fraction() {
        let d = parse_decimal("42").expect("parse");
        assert_eq!(d.scale, 0);
        assert_eq!(d.units, 42);
    }

    #[test]
    fn decimal_scale_too_high_rejected() {
        // 39 fractional digits.
        let d = parse_decimal("0.123456789012345678901234567890123456789");
        assert!(matches!(
            d,
            Err(TemporalError::DecimalScaleOutOfRange { .. })
        ));
    }

    #[test]
    fn decimal_constructor_validates_scale() {
        assert!(Decimal::new(39, 0).is_err());
        assert!(Decimal::new(-1, 0).is_err());
        assert!(Decimal::new(38, 0).is_ok());
    }

    #[test]
    fn zoned_datetime_ord_total() {
        // 2026-05-24T11:00:00Z < 2026-05-24T12:00:00Z
        let a = parse_zoned_datetime("2026-05-24T11:00:00Z").unwrap();
        let b = parse_zoned_datetime("2026-05-24T12:00:00Z").unwrap();
        assert!(a < b);
    }

    #[test]
    fn calendar_helpers_are_consistent() {
        assert!(is_leap_year(2024));
        assert!(!is_leap_year(2026));
        assert!(is_leap_year(2000));
        assert!(!is_leap_year(1900));
        // Round-trip: month/day → ordinal → month/day.
        for year in [2024i32, 2026, 2000, 1970] {
            for m in 1u8..=12 {
                let dim = days_in_month(year, m);
                for d in 1u8..=dim {
                    let ord = month_day_to_ordinal(year, m, d);
                    let (m2, d2) = ordinal_to_month_day(year, ord);
                    assert_eq!((m, d), (m2, d2), "{year}-{m:02}-{d:02} round-trip");
                }
            }
        }
    }

    #[test]
    fn utc_breakdown_recovers_epoch() {
        let (y, ord, h, m, s, ns) = utc_breakdown(0);
        assert_eq!((y, ord, h, m, s, ns), (1970, 1, 0, 0, 0, 0));
    }
}
