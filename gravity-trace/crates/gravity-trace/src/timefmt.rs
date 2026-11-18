//! Wall-clock times as UTC nanoseconds since the Unix epoch (the unit of every timestamp column and file name), to
//! and from text.
use std::time::{SystemTime, UNIX_EPOCH};

const NS: i64 = 1_000_000_000;

pub fn now_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)
}

/// Days since 1970-01-01 -> (year, month, day), proleptic Gregorian (H. Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// (year, month, day) -> days since 1970-01-01.
fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let m = i64::from(m);
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + i64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `2026-11-16T10:00:00.123Z`
pub fn fmt_utc(ns: i64) -> String {
    let (secs, sub) = (ns.div_euclid(NS), ns.rem_euclid(NS));
    let (days, sod) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}.{:03}Z", sod / 3600, sod % 3600 / 60, sod % 60, sub / 1_000_000)
}

/// `950ms`, `12.3s`, `4m05s`, `2h03m`.
pub fn fmt_span(ns: i64) -> String {
    let s = ns as f64 / 1e9;
    if s < 1.0 {
        format!("{:.0}ms", s * 1e3)
    } else if s < 120.0 {
        format!("{s:.1}s")
    } else if s < 7200.0 {
        format!("{}m{:02}s", (s / 60.0) as u64, (s % 60.0) as u64)
    } else {
        format!("{}h{:02}m", (s / 3600.0) as u64, (s % 3600.0 / 60.0) as u64)
    }
}

/// A point in time given on the command line, as UTC ns:
///
/// - `now`, `now-10m`, `now+30s`, `-10m` (= now-10m); with `base` (the start of a range) also `+5m` (= base+5m)
/// - Unix epoch time, unit by size: `1763280000` (s, fractions allowed), `1763280000123` (ms), 16 digits (us),
///   19 digits (ns)
/// - `2026-11-16 10:00`, `2026-11-16T10:00:00.5`, `2026-11-16` (midnight), `10:00:00` (today): local time of this
///   machine (`TZ`), or UTC / an offset with a `Z` / `+08:00` suffix
pub fn parse_time(s: &str, now: i64, base: Option<i64>) -> Option<i64> {
    let s = s.trim();
    let dur = |d: &str| crate::parse_duration(d).ok().and_then(|d| i64::try_from(d.as_nanos()).ok());
    if s.eq_ignore_ascii_case("now") {
        return Some(now);
    }
    if let Some(rest) = s.strip_prefix("now").or_else(|| s.strip_prefix("NOW")) {
        return match rest.split_at_checked(1)? {
            ("+", d) => now.checked_add(dur(d)?),
            ("-", d) => now.checked_sub(dur(d)?),
            _ => None,
        };
    }
    if let Some(d) = s.strip_prefix('+') {
        return base?.checked_add(dur(d)?);
    }
    if let Some(d) = s.strip_prefix('-') {
        return now.checked_sub(dur(d)?);
    }
    parse_epoch(s).or_else(|| parse_datetime(s, now))
}

fn all_digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit())
}

fn parse_epoch(s: &str) -> Option<i64> {
    let (int, frac) = s.split_once('.').unwrap_or((s, ""));
    if !all_digits(int) || !(frac.is_empty() || all_digits(frac)) {
        return None;
    }
    let v: i128 = int.parse().ok()?;
    // seconds until the year 5138, then ms, us, ns
    let (scale, digits) = if v < 100_000_000_000 {
        (1_000_000_000, 9)
    } else if v < 100_000_000_000_000 {
        (1_000_000, 6)
    } else if v < 100_000_000_000_000_000 {
        (1_000, 3)
    } else {
        (1, 0)
    };
    let mut ns = v * scale;
    if digits > 0 && !frac.is_empty() {
        let f: String = frac.chars().chain(std::iter::repeat('0')).take(digits).collect();
        ns += f.parse::<i128>().ok()?;
    }
    i64::try_from(ns).ok()
}

/// `YYYY-MM-DD[( |T)HH:MM[:SS[.f]]][Z|±HH[:MM]]` or `HH:MM[:SS[.f]][Z|±HH[:MM]]` (today).
fn parse_datetime(s: &str, now: i64) -> Option<i64> {
    let b = s.as_bytes();
    let has_date = b.len() >= 10 && b[4] == b'-' && b[7] == b'-';
    let (date, rest) = if has_date { (Some(&s[..10]), &s[10..]) } else { (None, s) };
    let rest = match (date, rest.chars().next()) {
        (Some(_), Some('T' | 't' | ' ')) => &rest[1..],
        (Some(_), Some('Z' | 'z' | '+' | '-')) | (_, None) | (None, _) => rest,
        (Some(_), Some(_)) => return None,
    };
    let (time, zone) = split_zone(rest)?;
    if date.is_none() && time.is_empty() {
        return None;
    }
    let (h, mi, sec, nanos) = if time.is_empty() { (0, 0, 0, 0) } else { parse_hms(time)? };
    let (y, mo, d) = match (date, zone) {
        (Some(d), _) => parse_ymd(d)?,
        (None, Some(off)) => civil_from_days((now.div_euclid(NS) + off).div_euclid(86_400)),
        (None, None) => local_today(now)?,
    };
    let secs = match zone {
        Some(off) => days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + sec - off,
        None => local_to_utc(y, mo, d, h, mi, sec)?,
    };
    secs.checked_mul(NS)?.checked_add(nanos)
}

/// Split a trailing `Z` / `±HH[:MM]` off; the offset in seconds east of UTC, `None` for local time.
fn split_zone(s: &str) -> Option<(&str, Option<i64>)> {
    if let Some(t) = s.strip_suffix(['Z', 'z']) {
        return Some((t, Some(0)));
    }
    let Some(i) = s.rfind(['+', '-']) else { return Some((s, None)) };
    let (t, z) = (&s[..i], &s[i + 1..]);
    let sign = if s.as_bytes()[i] == b'-' { -1 } else { 1 };
    let (hh, mm) = match z.split_once(':') {
        Some((h, m)) => (h, m),
        None if z.len() == 4 => z.split_at(2),
        None => (z, "0"),
    };
    if !all_digits(hh) || !all_digits(mm) {
        return None;
    }
    let (hh, mm): (i64, i64) = (hh.parse().ok()?, mm.parse().ok()?);
    if hh > 14 || mm > 59 {
        return None;
    }
    Some((t, Some(sign * (hh * 3600 + mm * 60))))
}

fn parse_ymd(s: &str) -> Option<(i64, u32, u32)> {
    let mut it = s.split('-');
    let (y, m, d) = (it.next()?, it.next()?, it.next()?);
    if it.next().is_some() || y.len() != 4 || !all_digits(y) || !all_digits(m) || !all_digits(d) {
        return None;
    }
    let (y, m, d): (i64, u32, u32) = (y.parse().ok()?, m.parse().ok()?, d.parse().ok()?);
    ((1..=12).contains(&m) && (1..=31).contains(&d)).then_some((y, m, d))
}

fn parse_hms(s: &str) -> Option<(i64, i64, i64, i64)> {
    let mut it = s.split(':');
    let (h, m, sec) = (it.next()?, it.next()?, it.next().unwrap_or("0"));
    if it.next().is_some() || !all_digits(h) || !all_digits(m) {
        return None;
    }
    let (sec, frac) = sec.split_once('.').unwrap_or((sec, ""));
    if !all_digits(sec) || !(frac.is_empty() || all_digits(frac)) {
        return None;
    }
    let (h, m, sec): (i64, i64, i64) = (h.parse().ok()?, m.parse().ok()?, sec.parse().ok()?);
    if h > 23 || m > 59 || sec > 60 {
        return None;
    }
    let nanos = if frac.is_empty() {
        0
    } else {
        frac.chars().chain(std::iter::repeat('0')).take(9).collect::<String>().parse().ok()?
    };
    Some((h, m, sec, nanos))
}

/// Local wall-clock time (this machine's `TZ`) -> Unix seconds.
fn local_to_utc(y: i64, mo: u32, d: u32, h: i64, mi: i64, s: i64) -> Option<i64> {
    // SAFETY: an all-zero `tm` is a valid value (null tm_zone); mktime only reads and normalises the struct.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    tm.tm_year = i32::try_from(y - 1900).ok()?;
    tm.tm_mon = mo as i32 - 1;
    tm.tm_mday = d as i32;
    tm.tm_hour = h as i32;
    tm.tm_min = mi as i32;
    tm.tm_sec = s as i32;
    tm.tm_isdst = -1;
    // SAFETY: `tm` is initialised above.
    let t = unsafe { libc::mktime(&mut tm) };
    (t != -1).then_some(t as i64)
}

/// Today's local date.
fn local_today(now: i64) -> Option<(i64, u32, u32)> {
    let t = now.div_euclid(NS) as libc::time_t;
    // SAFETY: as above; localtime_r writes into `tm` and returns null on failure.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let r = unsafe { libc::localtime_r(&t, &mut tm) };
    (!r.is_null()).then(|| (i64::from(tm.tm_year) + 1900, (tm.tm_mon + 1) as u32, tm.tm_mday as u32))
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: i64 = 1_763_287_200_000_000_000; // 2025-11-16T10:00:00Z

    #[test]
    fn calendar_round_trip() {
        assert_eq!(fmt_utc(T), "2025-11-16T10:00:00.000Z");
        assert_eq!(fmt_utc(T + 123_456_789), "2025-11-16T10:00:00.123Z");
        assert_eq!(fmt_utc(0), "1970-01-01T00:00:00.000Z");
        for days in [-1_000_000, -1, 0, 59, 60, 11_016, 20_408, 1_000_000] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(civil_from_days(20_512), (2026, 2, 28));
        assert_eq!(civil_from_days(20_513), (2026, 3, 1));
    }

    #[test]
    fn epoch_units_by_size() {
        let p = |s| parse_time(s, 0, None);
        assert_eq!(p("1763287200"), Some(T));
        assert_eq!(p("1763287200.25"), Some(T + 250_000_000));
        assert_eq!(p("1763287200000"), Some(T));
        assert_eq!(p("1763287200000123"), Some(T + 123_000));
        assert_eq!(p("1763287200000000007"), Some(T + 7));
        assert_eq!(p("1763287200x"), None);
    }

    #[test]
    fn datetimes_with_zones() {
        let p = |s| parse_time(s, T, None);
        assert_eq!(p("2025-11-16T10:00:00Z"), Some(T));
        assert_eq!(p("2025-11-16 10:00Z"), Some(T));
        assert_eq!(p("2025-11-16T18:00:00+08:00"), Some(T));
        assert_eq!(p("2025-11-16T18:00:00+0800"), Some(T));
        assert_eq!(p("2025-11-16T05:00:00.5-05"), Some(T + 500_000_000));
        assert_eq!(p("2025-11-16Z"), Some(T - 10 * 3600 * NS));
        assert_eq!(p("10:00Z"), Some(T), "today");
        assert_eq!(p("12:30:15.000000001Z"), Some(T + (2 * 3600 + 30 * 60 + 15) * NS + 1));
        assert_eq!(p("2025-13-01T00:00Z"), None);
        assert_eq!(p("2025-11-16X10:00Z"), None);
        assert_eq!(p("25:00Z"), None);
        assert!(p("2025-11-16 10:00").is_some(), "local time");
    }

    #[test]
    fn relative_times() {
        assert_eq!(parse_time("now", T, None), Some(T));
        assert_eq!(parse_time("now-10m", T, None), Some(T - 600 * NS));
        assert_eq!(parse_time("now+500ms", T, None), Some(T + 500_000_000));
        assert_eq!(parse_time("-1h", T, None), Some(T - 3600 * NS));
        assert_eq!(parse_time("+5m", T, Some(1_000)), Some(1_000 + 300 * NS));
        assert_eq!(parse_time("+5m", T, None), None, "+ needs a start");
        assert_eq!(parse_time("now*2", T, None), None);
    }
}
