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
}
