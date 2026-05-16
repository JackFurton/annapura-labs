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
}
