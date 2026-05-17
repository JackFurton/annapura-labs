//! Neural-net primitives. Hand-rolled, scalar f32. Optimization is a later chapter.

/// RMSNorm: `y_i = x_i / rms(x) * w_i`, where `rms(x) = sqrt(mean(x²) + ε)`.
///
/// Differs from LayerNorm in two ways: it doesn't subtract the mean (so it's
/// not really "normalization" in the statistical sense, just rescaling), and
/// there's no learned bias. Llama, Gemma, Mistral, Qwen all use RMSNorm —
/// it has roughly the same effect as LayerNorm with fewer ops.
pub fn rmsnorm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(x.len(), weight.len());
    assert_eq!(out.len(), x.len());
    let n = x.len() as f32;
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / n;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Matrix–vector linear projection: `y = W·x` (no bias — Llama linears are biasless).
///
/// `w` is laid out row-major as `[out_dim, in_dim]` — i.e. the contiguous run
/// for output `j` is `w[j*in_dim .. (j+1)*in_dim]`. That matches both PyTorch's
/// `nn.Linear.weight` and GGUF's storage (GGUF labels the same bytes with the
/// reversed shape `[in_dim, out_dim]`).
pub fn linear(x: &[f32], w: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    let out_dim = y.len();
    assert_eq!(w.len(), in_dim * out_dim);
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        let mut acc = 0.0_f32;
        for i in 0..in_dim {
            acc += x[i] * row[i];
        }
        y[j] = acc;
    }
}

/// SIMD-accelerated `y = W·x`. Same contract as `linear`, but uses 8-wide f32
/// vector accumulators so the inner reduction doesn't bottleneck the way it
/// does for the scalar version.
///
/// Why this is faster: the scalar loop's `acc += x[i] * row[i]` is a reduction
/// — LLVM's auto-vectorizer struggles with it because the horizontal sum at
/// the end is expensive relative to the FMAs. With explicit SIMD we
/// accumulate into 8 independent lanes (no reduction in the hot loop) and
/// only reduce once at the very end. The compiler emits one NEON FMA pair
/// per chunk (on Apple Silicon) or one AVX2 FMA (on x86 with AVX2).
pub fn linear_simd(x: &[f32], w: &[f32], y: &mut [f32]) {
    let in_dim = x.len();
    let out_dim = y.len();
    assert_eq!(w.len(), in_dim * out_dim);
    for j in 0..out_dim {
        let row = &w[j * in_dim..(j + 1) * in_dim];
        y[j] = dot_f32_simd(x, row);
    }
}

/// Multi-threaded version of `linear_simd`. Each output `y[j]` is an
/// independent dot product, so we slice the output across threads via rayon
/// and let each thread run the same per-row SIMD kernel.
///
/// Chunk size is a tuning knob: too small → thread overhead dominates,
/// too large → uneven work distribution at the tail. 64 outputs per chunk
/// works well at the dimensions we care about (256 → 4 chunks, 2048 → 32
/// chunks, 5632 → 88 chunks, 32000 → 500 chunks).
pub fn linear_simd_par(x: &[f32], w: &[f32], y: &mut [f32]) {
    use rayon::prelude::*;

    let in_dim = x.len();
    let out_dim = y.len();
    assert_eq!(w.len(), in_dim * out_dim);

    const CHUNK: usize = 64;

    y.par_chunks_mut(CHUNK).enumerate().for_each(|(chunk_idx, y_chunk)| {
        let base_j = chunk_idx * CHUNK;
        for (offset, y_val) in y_chunk.iter_mut().enumerate() {
            let j = base_j + offset;
            let row = &w[j * in_dim..(j + 1) * in_dim];
            *y_val = dot_f32_simd(x, row);
        }
    });
}

/// Single-row dot product `Σ x[i] * row[i]` with f32x8 vector accumulation
/// and tail handling. Shared between linear_simd and linear_simd_par.
#[inline]
fn dot_f32_simd(x: &[f32], row: &[f32]) -> f32 {
    use wide::f32x8;
    debug_assert_eq!(x.len(), row.len());

    let n = x.len();
    let chunks = n / 8;
    let tail_start = chunks * 8;

    let mut acc = f32x8::ZERO;
    for ci in 0..chunks {
        let off = ci * 8;
        let xv = load_f32x8(x, off);
        let wv = load_f32x8(row, off);
        acc = wv.mul_add(xv, acc);
    }
    let mut sum = acc.reduce_add();
    for i in tail_start..n {
        sum += x[i] * row[i];
    }
    sum
}

#[inline(always)]
fn load_f32x8(s: &[f32], off: usize) -> wide::f32x8 {
    let arr: [f32; 8] = s[off..off + 8].try_into().expect("slice of length 8");
    wide::f32x8::from(arr)
}

/// RoPE (Rotary Positional Embedding), Llama-style (interleaved pairs).
///
/// For each pair `(x[2i], x[2i+1])` of a `head_dim`-long vector, rotate by
/// angle `pos · θ_i` where `θ_i = freq_base^(-2i/head_dim)`. The frequency
/// decays geometrically across pair index, so early pairs spin fast (good at
/// distinguishing nearby positions) and late pairs spin slowly (good at
/// long-range structure). Rotation preserves L2 norm.
///
/// This is the "interleaved" variant llama.cpp uses for Llama. HuggingFace
/// `transformers` uses the "rotate-half" variant, which is the same math on
/// a permuted memory layout — the GGUF converter unswizzles for us so we
/// can use the interleaved form directly here.
pub fn rope_inplace(x: &mut [f32], pos: usize, freq_base: f32) {
    let d = x.len();
    assert!(d % 2 == 0, "RoPE needs even dim, got {}", d);
    let pos = pos as f32;
    for i in 0..d / 2 {
        let omega = freq_base.powf(-2.0 * i as f32 / d as f32);
        let (s, c) = (pos * omega).sin_cos();
        let a = x[2 * i];
        let b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// Apply RoPE independently to each head's slice of a multi-head vector.
/// `x.len()` must be a multiple of `head_dim`.
pub fn rope_heads(x: &mut [f32], head_dim: usize, pos: usize, freq_base: f32) {
    assert_eq!(x.len() % head_dim, 0);
    for head in x.chunks_exact_mut(head_dim) {
        rope_inplace(head, pos, freq_base);
    }
}

/// Softmax in place, with the standard max-subtraction trick for numerical
/// stability. After the call, `x` sums to 1 and every entry is in `[0, 1]`.
pub fn softmax_in_place(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let max_val = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Elementwise `x += y`. Used for residual connections.
pub fn add_in_place(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        x[i] += y[i];
    }
}

/// Elementwise `x *= y`.
pub fn mul_in_place(x: &mut [f32], y: &[f32]) {
    assert_eq!(x.len(), y.len());
    for i in 0..x.len() {
        x[i] *= y[i];
    }
}

/// SiLU activation (a.k.a. Swish): `silu(z) = z · σ(z) = z / (1 + e^-z)`.
///
/// Smooth, non-monotonic (small dip around z = -1.28), approaches `z` as
/// `z → +∞` and `0` as `z → -∞`. Llama's FFN uses it as the gate
/// non-linearity in SwiGLU: `out = down(silu(gate(x)) ⊙ up(x))`.
pub fn silu_in_place(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
    }

    #[test]
    fn unit_weight_normalizes_to_unit_rms() {
        // y = x / rms(x) * 1 → rms(y) should be 1.
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        let mut y = vec![0.0; 4];
        rmsnorm(&x, &w, 0.0, &mut y);
        assert!((rms(&y) - 1.0).abs() < 1e-6, "rms(y) = {}", rms(&y));
    }

    #[test]
    fn scalar_weight_scales_the_output() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![2.0; 4];
        let mut y = vec![0.0; 4];
        rmsnorm(&x, &w, 0.0, &mut y);
        assert!((rms(&y) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn known_two_element_answer() {
        // x = [1, 2], w = [1, 1], eps = 0
        // sum(x²) = 5; mean = 2.5; rms = √2.5 ≈ 1.5811
        // scale = 1/rms ≈ 0.6325
        // y = [0.6325, 1.2649]
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        rmsnorm(&x, &w, 0.0, &mut y);
        assert!((y[0] - 0.63245553).abs() < 1e-6);
        assert!((y[1] - 1.26491106).abs() < 1e-6);
    }

    #[test]
    fn zero_input_stays_zero_via_eps() {
        // mean_sq = 0; scale = 1/√ε. Output = 0 * scale * w = 0 exactly. The eps
        // matters here: without it scale would be NaN/Inf.
        let x = vec![0.0; 8];
        let w = vec![1.5; 8];
        let mut y = vec![1.0; 8];
        rmsnorm(&x, &w, 1e-5, &mut y);
        assert!(y.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn linear_identity_matrix_returns_input() {
        // W = I_3 → y = x
        let w = vec![
            1., 0., 0.,
            0., 1., 0.,
            0., 0., 1.,
        ];
        let x = vec![1.5, -2.0, 3.0];
        let mut y = vec![0.0; 3];
        linear(&x, &w, &mut y);
        assert_eq!(y, x);
    }

    #[test]
    fn linear_known_rectangular_answer() {
        // x ∈ R², W ∈ R^{3×2}; y[j] = sum_i(x[i]*W[j,i])
        // W rows:  [1,2], [3,4], [5,6]   x = [10, 20]
        // y = [10+40, 30+80, 50+120] = [50, 110, 170]
        let w = vec![1., 2., 3., 4., 5., 6.];
        let x = vec![10.0, 20.0];
        let mut y = vec![0.0; 3];
        linear(&x, &w, &mut y);
        assert_eq!(y, vec![50.0, 110.0, 170.0]);
    }

    #[test]
    fn linear_simd_matches_linear_clean_dims() {
        // in_dim = 64 (clean multiple of 8), out_dim = 32.
        let in_dim = 64;
        let out_dim = 32;
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.073).sin()).collect();
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i as f32) * 0.019).cos()).collect();
        let mut y_scalar = vec![0.0_f32; out_dim];
        let mut y_simd = vec![0.0_f32; out_dim];
        linear(&x, &w, &mut y_scalar);
        linear_simd(&x, &w, &mut y_simd);
        for j in 0..out_dim {
            assert!(
                (y_scalar[j] - y_simd[j]).abs() < 1e-4,
                "mismatch at {}: scalar {} vs simd {}",
                j, y_scalar[j], y_simd[j]
            );
        }
    }

    #[test]
    fn linear_simd_matches_linear_with_tail() {
        // in_dim = 100 (12 full chunks of 8 plus a 4-element tail)
        let in_dim = 100;
        let out_dim = 17;
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.13).sin()).collect();
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i as f32) * 0.029).cos()).collect();
        let mut y_scalar = vec![0.0_f32; out_dim];
        let mut y_simd = vec![0.0_f32; out_dim];
        linear(&x, &w, &mut y_scalar);
        linear_simd(&x, &w, &mut y_simd);
        for j in 0..out_dim {
            assert!((y_scalar[j] - y_simd[j]).abs() < 1e-4);
        }
    }

    #[test]
    fn linear_simd_par_matches_serial() {
        // Use out_dim large enough to actually exercise multiple chunks
        // (CHUNK=64, so >64 outputs trigger real parallel work).
        let in_dim = 2048;
        let out_dim = 200;
        let x: Vec<f32> = (0..in_dim).map(|i| ((i as f32) * 0.001).sin()).collect();
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i as f32) * 0.0001).cos()).collect();
        let mut y_serial = vec![0.0_f32; out_dim];
        let mut y_par = vec![0.0_f32; out_dim];
        linear_simd(&x, &w, &mut y_serial);
        linear_simd_par(&x, &w, &mut y_par);
        for j in 0..out_dim {
            assert!(
                (y_serial[j] - y_par[j]).abs() < 1e-4,
                "mismatch at {}: serial {} vs par {}",
                j, y_serial[j], y_par[j]
            );
        }
    }

    #[test]
    fn linear_simd_par_handles_small_output() {
        // out_dim < CHUNK means just one chunk on one thread — still correct.
        let in_dim = 64;
        let out_dim = 5;
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.1).collect();
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| (i as f32) * 0.01).collect();
        let mut y_serial = vec![0.0_f32; out_dim];
        let mut y_par = vec![0.0_f32; out_dim];
        linear_simd(&x, &w, &mut y_serial);
        linear_simd_par(&x, &w, &mut y_par);
        for j in 0..out_dim {
            assert!((y_serial[j] - y_par[j]).abs() < 1e-4);
        }
    }

    #[test]
    fn rope_at_position_zero_is_identity() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        rope_inplace(&mut x, 0, 10000.0);
        for (a, b) in x.iter().zip(&original) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rope_preserves_l2_norm() {
        let mut x = vec![1.5, -0.7, 0.3, 2.1, -1.0, 0.0, 0.5, -2.5];
        let n0: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        rope_inplace(&mut x, 17, 10000.0);
        let n1: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n0 - n1).abs() < 1e-5, "{} vs {}", n0, n1);
    }

    #[test]
    fn rope_two_dim_unit_x_axis_rotates_to_known_angle() {
        // d = 2, freq = 10000^0 = 1, angle = pos · 1 = pos.
        // x = (1, 0) at pos = 1 → (cos 1, sin 1)
        let mut x = vec![1.0, 0.0];
        rope_inplace(&mut x, 1, 10000.0);
        assert!((x[0] - 1.0_f32.cos()).abs() < 1e-6);
        assert!((x[1] - 1.0_f32.sin()).abs() < 1e-6);
    }

    #[test]
    fn rope_heads_with_one_head_equals_rope_inplace() {
        let mut a = vec![0.5, -1.0, 2.0, 0.3];
        let mut b = a.clone();
        rope_inplace(&mut a, 7, 10000.0);
        rope_heads(&mut b, 4, 7, 10000.0);
        assert_eq!(a, b);
    }

    #[test]
    fn rope_heads_rotates_each_head_independently() {
        // Two heads of dim 2. Each should rotate as if it were alone.
        let mut multi = vec![1.0, 0.0, 1.0, 0.0];
        let mut single = vec![1.0, 0.0];
        rope_heads(&mut multi, 2, 1, 10000.0);
        rope_inplace(&mut single, 1, 10000.0);
        assert!((multi[0] - single[0]).abs() < 1e-6);
        assert!((multi[1] - single[1]).abs() < 1e-6);
        assert!((multi[2] - single[0]).abs() < 1e-6);
        assert!((multi[3] - single[1]).abs() < 1e-6);
    }

    #[test]
    fn softmax_uniform_input_gives_uniform_output() {
        let mut x = vec![1.0, 1.0, 1.0, 1.0];
        softmax_in_place(&mut x);
        for v in &x {
            assert!((v - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_one_hot_input_concentrates_mass() {
        // x = [0, 100, 0]: after softmax the middle dominates.
        let mut x = vec![0.0, 100.0, 0.0];
        softmax_in_place(&mut x);
        assert!(x[1] > 0.999);
        assert!(x[0] < 0.001 && x[2] < 0.001);
    }

    #[test]
    fn softmax_survives_huge_inputs_without_overflow() {
        // Without max-subtraction, exp(1000) is +inf. Verify we handle it.
        let mut x = vec![1000.0, 1000.001, 999.999];
        softmax_in_place(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {}", sum);
        // The largest input should still get the largest probability.
        assert!(x[1] > x[0]);
        assert!(x[1] > x[2]);
    }

    #[test]
    fn add_in_place_sums_elementwise() {
        let mut x = vec![1.0, 2.0, 3.0];
        add_in_place(&mut x, &[10.0, 20.0, 30.0]);
        assert_eq!(x, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn mul_in_place_multiplies_elementwise() {
        let mut x = vec![1.0, 2.0, 3.0];
        mul_in_place(&mut x, &[10.0, 20.0, 30.0]);
        assert_eq!(x, vec![10.0, 40.0, 90.0]);
    }

    #[test]
    fn silu_known_values() {
        let mut x = vec![0.0, 1.0, -1.0, 10.0, -10.0];
        silu_in_place(&mut x);
        // silu(0) = 0
        assert!(x[0].abs() < 1e-7);
        // silu(1) = 1 / (1 + e^-1) ≈ 0.7310585786
        assert!((x[1] - 0.7310586).abs() < 1e-5);
        // silu(-1) = -1 / (1 + e) ≈ -0.2689414213
        assert!((x[2] - -0.2689414).abs() < 1e-5);
        // silu(10) ≈ 10 (saturates linearly for large positive)
        assert!((x[3] - 10.0).abs() < 1e-3);
        // silu(-10) ≈ 0 (saturates near zero for large negative)
        assert!(x[4].abs() < 1e-3);
    }
}
