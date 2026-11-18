//! Log-linear latency histogram (HDR-style, 4 linear sub-buckets per power of two).
//!
//! Relative bucket width is <= 25%, so quantiles estimated from the bucket midpoint are
//! within ~12% of the true value. Values >= 2^MAX_OCTAVE ns (~18 min) saturate.

const SUB_BITS: u32 = 2;
const SUB: usize = 1 << SUB_BITS;
const MAX_OCTAVE: u32 = 40;
pub const NUM_BUCKETS: usize = (MAX_OCTAVE as usize + 1) * SUB;

#[derive(Clone)]
pub struct LogHist {
    buckets: Box<[u32; NUM_BUCKETS]>,
    count: u64,
}

impl Default for LogHist {
    fn default() -> Self {
        Self { buckets: Box::new([0; NUM_BUCKETS]), count: 0 }
    }
}

#[inline]
pub fn bucket_of(v: u64) -> usize {
    if v < SUB as u64 {
        return v as usize;
    }
    let k = 63 - v.leading_zeros(); // floor(log2 v), >= SUB_BITS
    if k > MAX_OCTAVE {
        return NUM_BUCKETS - 1;
    }
    let sub = ((v >> (k - SUB_BITS)) as usize) & (SUB - 1);
    k as usize * SUB + sub
}

/// Inclusive lower bound and exclusive upper bound of a bucket.
pub fn bucket_bounds(idx: usize) -> (u64, u64) {
    if idx < SUB {
        return (idx as u64, idx as u64 + 1);
    }
    let k = (idx / SUB) as u32;
    let sub = (idx % SUB) as u64;
    let shift = k - SUB_BITS;
    ((SUB as u64 + sub) << shift, (SUB as u64 + sub + 1) << shift)
}

impl LogHist {
    #[inline]
    pub fn record(&mut self, v: u64) {
        let b = &mut self.buckets[bucket_of(v)];
        *b = b.saturating_add(1);
        self.count += 1;
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn merge(&mut self, other: &LogHist) {
        for (a, b) in self.buckets.iter_mut().zip(other.buckets.iter()) {
            *a = a.saturating_add(*b);
        }
        self.count += other.count;
    }

    /// Estimate the q-quantile (0..=1). Returns 0 for an empty histogram.
    pub fn quantile(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((q.clamp(0.0, 1.0) * self.count as f64).ceil() as u64).max(1);
        let mut seen = 0u64;
        for (i, &c) in self.buckets.iter().enumerate() {
            seen += c as u64;
            if seen >= target {
                let (lo, hi) = bucket_bounds(i);
                return lo + (hi - lo) / 2;
            }
        }
        bucket_bounds(NUM_BUCKETS - 1).0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_monotonic_and_contain_value() {
        let mut prev = 0;
        for v in 0..100_000u64 {
            let b = bucket_of(v);
            assert!(b >= prev, "non monotonic at {v}");
            prev = b;
            let (lo, hi) = bucket_bounds(b);
            assert!(lo <= v && v < hi, "v={v} b={b} [{lo},{hi})");
        }
        for v in [1u64 << 20, (1 << 30) + 12345, 1 << 39] {
            let (lo, hi) = bucket_bounds(bucket_of(v));
            assert!(lo <= v && v < hi, "v={v} [{lo},{hi})");
        }
    }

    #[test]
    fn quantiles_are_close() {
        let mut h = LogHist::default();
        for v in 1..=10_000u64 {
            h.record(v * 1000);
        }
        let p50 = h.quantile(0.5) as f64;
        let p99 = h.quantile(0.99) as f64;
        assert!((p50 / 5_000_000.0 - 1.0).abs() < 0.13, "p50={p50}");
        assert!((p99 / 9_900_000.0 - 1.0).abs() < 0.13, "p99={p99}");
    }
}
