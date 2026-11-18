//! Always-on aggregation of GPU activity durations.
//!
//! Two views are maintained:
//! - a **window** (reset at every report): count / total / min / max / log-histogram per
//!   (kind, device, name) -> the window quantiles of the exported metrics;
//! - **cumulative** counters since the process was first seen -> exported as Prometheus counters.
//!
//! Kernel "busy" time per device is the exact union of kernel execution intervals, kept in
//! an [`IntervalUnion`] so that overlap between streams is not double counted and records
//! may arrive in any order (snapshots and deferred GPU-time updates deliver them out of time order). Newly covered time is
//! attributed to the window it lies in: coverage inside the current window counts towards
//! the window's busy ratio (so it never exceeds 100%), coverage of already closed windows
//! (late records) only goes into the cumulative counter. Intervals older than the retention
//! horizon are pruned to bound memory.
use crate::hist::LogHist;
use std::collections::BTreeMap;
use std::ops::Bound;
use crate::model::EventKind;
use crate::names::{Interner, short_name};
use crate::metrics::{DeviceMetric, KernelMetric};
use crate::schema::StatRow;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    pub kind: EventKind,
    pub device: u32,
    pub name: u32,
}

#[derive(Default)]
struct WindowStat {
    count: u64,
    total: u64,
    min: u64,
    max: u64,
    hist: LogHist,
}

#[derive(Default, Clone, Copy)]
struct CumStat {
    count: u64,
    total: u64,
    last_p50: u64,
    last_p99: u64,
}

/// Disjoint, sorted set of half-open intervals `[start, end)`.
#[derive(Default, Debug)]
pub struct IntervalUnion {
    map: BTreeMap<i64, i64>,
}

impl IntervalUnion {
    /// Insert `[s, e)`; `gap(gs, ge)` is called for every sub-range that was not covered yet.
    pub fn insert(&mut self, s: i64, e: i64, mut gap: impl FnMut(i64, i64)) {
        if e <= s {
            return;
        }
        let mut overlapping: Vec<(i64, i64)> = Vec::new();
        if let Some((&ps, &pe)) = self.map.range(..=s).next_back()
            && pe >= s
        {
            overlapping.push((ps, pe));
        }
        overlapping.extend(self.map.range((Bound::Excluded(s), Bound::Included(e))).map(|(&a, &b)| (a, b)));
        let mut cursor = s;
        for &(qs, qe) in &overlapping {
            if qs > cursor {
                gap(cursor, qs.min(e));
            }
            cursor = cursor.max(qe);
            if cursor >= e {
                break;
            }
        }
        if cursor < e {
            gap(cursor, e);
        }
        let (mut ns, mut ne) = (s, e);
        for (qs, qe) in overlapping {
            self.map.remove(&qs);
            ns = ns.min(qs);
            ne = ne.max(qe);
        }
        self.map.insert(ns, ne);
    }

    /// Drop intervals ending before `t`, and the oldest ones beyond `max_len`.
    pub fn prune(&mut self, t: i64, max_len: usize) {
        while let Some((&s, &e)) = self.map.first_key_value() {
            if e < t || self.map.len() > max_len {
                self.map.remove(&s);
            } else {
                break;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

#[derive(Default)]
struct Busy {
    union: IntervalUnion,
    window_ns: u64,
    window_kernels: u64,
    total_ns: u64,
    total_kernels: u64,
    /// Latest kernel end seen (GPU timeline position of the busy counter).
    horizon: i64,
}

/// How long merged intervals are kept to absorb late (out of order) records.
const BUSY_RETENTION_NS: i64 = 30_000_000_000;
const BUSY_MAX_INTERVALS: usize = 1 << 20;

pub struct Aggregator {
    pub names: Interner,
    window: FxHashMap<Key, WindowStat>,
    cumulative: FxHashMap<Key, CumStat>,
    busy: FxHashMap<u32, Busy>,
    window_start_ns: i64,
}

impl Aggregator {
    pub fn new(now_ns: i64) -> Self {
        Self {
            names: Interner::default(),
            window: FxHashMap::default(),
            cumulative: FxHashMap::default(),
            busy: FxHashMap::default(),
            window_start_ns: now_ns,
        }
    }

    /// Record one GPU activity.
    #[inline]
    pub fn record(&mut self, kind: EventKind, device: u32, name: &str, start_ns: i64, end_ns: i64) {
        let dur = end_ns.saturating_sub(start_ns).max(0) as u64;
        let key = Key { kind, device, name: self.names.intern(name) };
        let w = self.window.entry(key).or_default();
        if w.count == 0 || dur < w.min {
            w.min = dur;
        }
        w.max = w.max.max(dur);
        w.count += 1;
        w.total += dur;
        w.hist.record(dur);
        let c = self.cumulative.entry(key).or_default();
        c.count += 1;
        c.total += dur;
    }

    /// Merge kernel intervals of one device (any order) into the busy-time union.
    pub fn add_busy(&mut self, device: u32, intervals: &mut [(i64, i64)]) {
        if intervals.is_empty() {
            return;
        }
        // Sorting keeps the union insertions append-like (cheap) for in-order streams.
        intervals.sort_unstable_by_key(|iv| iv.0);
        let ws = self.window_start_ns;
        let b = self.busy.entry(device).or_default();
        let (mut window_add, mut total_add) = (0u64, 0u64);
        for &(s, e) in intervals.iter() {
            b.horizon = b.horizon.max(e);
            b.union.insert(s, e, |gs, ge| {
                total_add += (ge - gs) as u64;
                window_add += (ge - gs.max(ws)).max(0) as u64;
            });
        }
        b.window_ns += window_add;
        b.total_ns += total_add;
        b.window_kernels += intervals.len() as u64;
        b.total_kernels += intervals.len() as u64;
    }

    pub fn distinct_names(&self) -> usize {
        self.names.len()
    }

    /// Close the current window and return its rows (sorted by total time, descending).
    pub fn close_window(&mut self, now_ns: i64) -> Vec<StatRow> {
        let (ws, we) = (self.window_start_ns, now_ns);
        self.window_start_ns = now_ns;
        let mut rows: Vec<StatRow> = Vec::with_capacity(self.window.len() + self.busy.len());
        for (key, w) in self.window.drain() {
            let (p50, p90, p99) = (w.hist.quantile(0.5), w.hist.quantile(0.9), w.hist.quantile(0.99));
            // Quantiles are bucket midpoints: clamp into the exact [min, max] range.
            let clamp = |v: u64| v.clamp(w.min, w.max);
            if let Some(c) = self.cumulative.get_mut(&key) {
                c.last_p50 = clamp(p50);
                c.last_p99 = clamp(p99);
            }
            rows.push(StatRow {
                window_start_ns: ws,
                window_end_ns: we,
                device_id: key.device,
                kind: key.kind.as_str(),
                name: self.names.get(key.name).clone(),
                count: w.count,
                total_ns: w.total,
                min_ns: w.min,
                max_ns: w.max,
                p50_ns: clamp(p50),
                p90_ns: clamp(p90),
                p99_ns: clamp(p99),
            });
        }
        for (&device, b) in self.busy.iter_mut() {
            b.union.prune(now_ns - BUSY_RETENTION_NS, BUSY_MAX_INTERVALS);
            if b.window_kernels == 0 {
                continue;
            }
            rows.push(StatRow {
                window_start_ns: ws,
                window_end_ns: we,
                device_id: device,
                kind: "kernel_busy",
                name: "".into(),
                count: b.window_kernels,
                total_ns: b.window_ns,
                min_ns: 0,
                max_ns: 0,
                p50_ns: 0,
                p90_ns: 0,
                p99_ns: 0,
            });
            b.window_ns = 0;
            b.window_kernels = 0;
        }
        rows.sort_by_key(|r| std::cmp::Reverse(r.total_ns));
        rows
    }

    /// Cumulative per-kernel metrics: the `top` entries by total time per kind, the rest
    /// folded into a single `"<other>"` entry per (kind, device).
    pub fn metrics(&self, top: usize, label_len: usize) -> (Vec<KernelMetric>, Vec<DeviceMetric>) {
        let mut entries: Vec<(&Key, &CumStat)> = self.cumulative.iter().collect();
        entries.sort_by(|a, b| b.1.total.cmp(&a.1.total).then(a.0.cmp(b.0)));
        let mut per_kind: FxHashMap<EventKind, usize> = FxHashMap::default();
        let mut others: FxHashMap<(EventKind, u32), (u64, u64)> = FxHashMap::default();
        let mut out = Vec::new();
        for (key, c) in entries {
            let n = per_kind.entry(key.kind).or_default();
            if *n < top {
                *n += 1;
                out.push(KernelMetric {
                    kind: key.kind,
                    device: key.device,
                    name: short_name(self.names.get(key.name), label_len),
                    count: c.count,
                    total_ns: c.total,
                    last_p50_ns: c.last_p50,
                    last_p99_ns: c.last_p99,
                });
            } else {
                let o = others.entry((key.kind, key.device)).or_default();
                o.0 += c.count;
                o.1 += c.total;
            }
        }
        for ((kind, device), (count, total_ns)) in others {
            out.push(KernelMetric {
                kind,
                device,
                name: "<other>".into(),
                count,
                total_ns,
                last_p50_ns: 0,
                last_p99_ns: 0,
            });
        }
        let mut devices: Vec<DeviceMetric> = self
            .busy
            .iter()
            .map(|(&device, b)| DeviceMetric {
                device,
                kernel_busy_ns: b.total_ns,
                kernels: b.total_kernels,
                busy_horizon_ns: b.horizon,
            })
            .collect();
        devices.sort_by_key(|d| d.device);
        (out, devices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interval_union_reports_only_new_coverage() {
        let mut u = IntervalUnion::default();
        let mut gaps = Vec::new();
        u.insert(10, 20, |a, b| gaps.push((a, b)));
        u.insert(30, 40, |a, b| gaps.push((a, b)));
        gaps.clear();
        u.insert(5, 45, |a, b| gaps.push((a, b))); // spans both + the holes
        assert_eq!(gaps, vec![(5, 10), (20, 30), (40, 45)]);
        assert_eq!(u.len(), 1);
        gaps.clear();
        u.insert(12, 18, |a, b| gaps.push((a, b))); // fully covered
        assert!(gaps.is_empty());
        u.insert(45, 50, |a, b| gaps.push((a, b))); // touching -> merged
        assert_eq!((gaps, u.len()), (vec![(45, 50)], 1));
        u.prune(100, usize::MAX);
        assert!(u.is_empty());
    }

    #[test]
    fn busy_is_order_independent_and_window_bounded() {
        let mut a = Aggregator::new(1_000);
        // Late records from before the window start only count cumulatively.
        a.add_busy(0, &mut [(1_500, 1_900), (900, 1_200)]);
        a.add_busy(0, &mut [(1_000, 1_600)]); // overlaps both
        let rows = a.close_window(2_000);
        let busy = rows.iter().find(|r| r.kind == "kernel_busy").expect("busy row");
        assert_eq!(busy.total_ns, 900); // [1000,1900) inside the window
        let (_, devices) = a.metrics(10, 64);
        assert_eq!(devices[0].kernel_busy_ns, 1_000); // [900,1900) overall
        assert_eq!(devices[0].busy_horizon_ns, 1_900);
    }

    #[test]
    fn busy_union_does_not_double_count_overlap() {
        let mut a = Aggregator::new(0);
        // [0,10) and [5,15) overlap -> union 15; [20,30) -> +10
        a.add_busy(0, &mut [(5, 15), (0, 10), (20, 30)]);
        let rows = a.close_window(100);
        let busy = rows.iter().find(|r| r.kind == "kernel_busy").expect("busy row");
        assert_eq!(busy.total_ns, 25);
        assert_eq!(busy.count, 3);
    }

    #[test]
    fn window_and_cumulative() {
        let mut a = Aggregator::new(0);
        for i in 0..10 {
            a.record(EventKind::Kernel, 0, "k1", i * 100, i * 100 + 50);
        }
        a.record(EventKind::Kernel, 0, "k2", 0, 1000);
        let rows = a.close_window(2000);
        assert_eq!(rows.len(), 2);
        assert_eq!(&*rows[0].name, "k2"); // sorted by total
        let k1 = &rows[1];
        assert_eq!((k1.count, k1.total_ns, k1.min_ns, k1.max_ns), (10, 500, 50, 50));
        assert_eq!(k1.p50_ns, 50);
        a.record(EventKind::Kernel, 0, "k1", 0, 50);
        let (kernels, _) = a.metrics(1, 64);
        let k2 = kernels.iter().find(|k| k.name == "k2").expect("top entry");
        assert_eq!(k2.total_ns, 1000);
        let other = kernels.iter().find(|k| k.name == "<other>").expect("folded entry");
        assert_eq!((other.count, other.total_ns), (11, 550));
        assert!(a.close_window(3000).iter().any(|r| r.count == 1));
    }
}
