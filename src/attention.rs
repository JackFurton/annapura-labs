//! Scaled dot-product attention with KV cache + GQA support.
//!
//! The KV cache is the trick that makes incremental decoding fast: once we've
//! computed K and V for a token, we never recompute them — we just look them
//! up. For prefill (processing a prompt), the cache fills up sequentially as
//! each token's projections are computed and stored.

use crate::nn::softmax_in_place;

/// Per-layer K/V storage. Logical shape: `[max_seq_len, n_kv_heads * head_dim]`,
/// row-major (one row per token position).
pub struct KvCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub max_seq_len: usize,
    pub kv_dim: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize, kv_dim: usize) -> Self {
        Self {
            k: vec![0.0; max_seq_len * kv_dim],
            v: vec![0.0; max_seq_len * kv_dim],
            max_seq_len,
            kv_dim,
        }
    }

    /// Write K and V vectors for token at `pos` into the cache.
    pub fn store(&mut self, pos: usize, k: &[f32], v: &[f32]) {
        assert_eq!(k.len(), self.kv_dim);
        assert_eq!(v.len(), self.kv_dim);
        assert!(pos < self.max_seq_len, "pos {} ≥ max_seq_len {}", pos, self.max_seq_len);
        let off = pos * self.kv_dim;
        self.k[off..off + self.kv_dim].copy_from_slice(k);
        self.v[off..off + self.kv_dim].copy_from_slice(v);
    }

    pub fn k_at(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let off = pos * self.kv_dim + kv_head * head_dim;
        &self.k[off..off + head_dim]
    }

    pub fn v_at(&self, pos: usize, kv_head: usize, head_dim: usize) -> &[f32] {
        let off = pos * self.kv_dim + kv_head * head_dim;
        &self.v[off..off + head_dim]
    }
}

/// Scaled dot-product attention for the current token, with GQA.
///
/// Preconditions: the current token's K and V must already be in `cache` at
/// position `cur_pos` (the caller is responsible for `cache.store`).
///
/// For each query head `h ∈ [0, n_heads)`, attends to KV head
/// `kv_h = h / (n_heads / n_kv_heads)` across all positions `[0, cur_pos]`.
/// Concatenates head outputs into `out`.
pub fn attention(
    q: &[f32],
    cache: &KvCache,
    cur_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), n_heads * head_dim);
    assert_eq!(out.len(), n_heads * head_dim);
    assert_eq!(n_heads % n_kv_heads, 0, "n_heads must be a multiple of n_kv_heads");
    let q_per_kv = n_heads / n_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0.0_f32; cur_pos + 1];

    for h in 0..n_heads {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let kv_h = h / q_per_kv;

        for p in 0..=cur_pos {
            let k = cache.k_at(p, kv_h, head_dim);
            let dot: f32 = q_h.iter().zip(k).map(|(a, b)| a * b).sum();
            scores[p] = dot * inv_sqrt_d;
        }
        softmax_in_place(&mut scores);

        let out_h = &mut out[h * head_dim..(h + 1) * head_dim];
        out_h.fill(0.0);
        for p in 0..=cur_pos {
            let v = cache.v_at(p, kv_h, head_dim);
            let w = scores[p];
            for d in 0..head_dim {
                out_h[d] += w * v[d];
            }
        }
    }
}

/// Diagnostic helper: per-position attention mass averaged across heads.
/// Returns a `cur_pos + 1` vector that sums to 1. Useful for "what is this
/// token looking at?" plots.
pub fn attention_pattern(
    q: &[f32],
    cache: &KvCache,
    cur_pos: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let q_per_kv = n_heads / n_kv_heads;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
    let mut sum = vec![0.0_f32; cur_pos + 1];
    let mut scores = vec![0.0_f32; cur_pos + 1];

    for h in 0..n_heads {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        let kv_h = h / q_per_kv;
        for p in 0..=cur_pos {
            let k = cache.k_at(p, kv_h, head_dim);
            scores[p] = q_h.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * inv_sqrt_d;
        }
        softmax_in_place(&mut scores);
        for p in 0..=cur_pos {
            sum[p] += scores[p];
        }
    }
    let inv_n = 1.0 / n_heads as f32;
    for w in sum.iter_mut() {
        *w *= inv_n;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_position_attends_to_itself() {
        // With only pos 0 in the cache, softmax over [score_0] = [1.0].
        // Output should equal V[0, kv_head=0].
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let kv_dim = n_kv_heads * head_dim;
        let mut cache = KvCache::new(8, kv_dim);
        let v0 = vec![10.0, 20.0, 30.0, 40.0];
        cache.store(0, &[1.0, 0.0, 0.0, 0.0], &v0);

        let q = vec![1.0, 0.0, 0.0, 0.0];
        let mut out = vec![0.0_f32; head_dim];
        attention(&q, &cache, 0, n_heads, n_kv_heads, head_dim, &mut out);
        assert_eq!(out, v0);
    }

    #[test]
    fn attention_picks_the_matching_key() {
        // Two positions. K[0] = [1, 0, ...], K[1] = [0, 1, 0, ...].
        // Q = K[1] direction → attention concentrates on position 1.
        let head_dim = 4;
        let n_heads = 1;
        let n_kv_heads = 1;
        let kv_dim = head_dim;
        let mut cache = KvCache::new(8, kv_dim);

        let k0 = vec![1.0, 0.0, 0.0, 0.0];
        let v0 = vec![1.0; head_dim];
        let k1 = vec![0.0, 1.0, 0.0, 0.0];
        let v1 = vec![9.0; head_dim];
        cache.store(0, &k0, &v0);
        cache.store(1, &k1, &v1);

        // Big q magnitude so softmax sharply concentrates on the match.
        let q = vec![0.0, 100.0, 0.0, 0.0];
        let mut out = vec![0.0_f32; head_dim];
        attention(&q, &cache, 1, n_heads, n_kv_heads, head_dim, &mut out);

        for &v in &out {
            assert!((v - 9.0).abs() < 1e-4, "expected ~9.0, got {}", v);
        }
    }

    #[test]
    fn gqa_one_kv_head_serves_all_query_heads() {
        // n_heads = 4, n_kv_heads = 1: all four Q heads share the single KV head.
        // If we set Q identically across heads, all four head outputs should match.
        let head_dim = 2;
        let n_heads = 4;
        let n_kv_heads = 1;
        let kv_dim = n_kv_heads * head_dim;
        let mut cache = KvCache::new(4, kv_dim);
        cache.store(0, &[1.0, 0.0], &[7.0, 8.0]);
        cache.store(1, &[0.0, 1.0], &[3.0, 4.0]);

        // Same query for all four heads.
        let q = vec![0.5, 0.5,  0.5, 0.5,  0.5, 0.5,  0.5, 0.5];
        let mut out = vec![0.0_f32; n_heads * head_dim];
        attention(&q, &cache, 1, n_heads, n_kv_heads, head_dim, &mut out);

        // All four head outputs should be identical (they share K, V).
        let h0 = &out[0..2];
        for h in 1..4 {
            let hi = &out[h * head_dim..(h + 1) * head_dim];
            assert!((hi[0] - h0[0]).abs() < 1e-6);
            assert!((hi[1] - h0[1]).abs() < 1e-6);
        }
    }

    #[test]
    fn attention_pattern_sums_to_one() {
        let head_dim = 4;
        let n_heads = 2;
        let n_kv_heads = 1;
        let kv_dim = head_dim;
        let mut cache = KvCache::new(4, kv_dim);
        cache.store(0, &[0.1, 0.2, 0.3, 0.4], &[1.0; 4]);
        cache.store(1, &[0.5, 0.6, 0.7, 0.8], &[1.0; 4]);
        cache.store(2, &[0.9, 1.0, 1.1, 1.2], &[1.0; 4]);

        let q = vec![1.0; n_heads * head_dim];
        let pattern = attention_pattern(&q, &cache, 2, n_heads, n_kv_heads, head_dim);

        assert_eq!(pattern.len(), 3);
        let s: f32 = pattern.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "sum = {}", s);
    }
}
