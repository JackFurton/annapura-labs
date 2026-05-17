/// C = A·B, row-major, f32. A is M×K, B is K×N, C is M×N.
///
/// Naive textbook triple loop in i,j,k order. This exists to be slow on purpose:
/// it is the reference correctness oracle and the perf floor for all later work.
pub fn matmul_naive(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f32;
            for kk in 0..k {
                acc += a[i * k + kk] * b[kk * n + j];
            }
            c[i * n + j] = acc;
        }
    }
}

/// Cache-blocked C = A·B. Identical inner kernel to `matmul_ikj` (so the
/// auto-vectorizer keeps doing its thing), but constrained to operate on a
/// tile small enough that the working set fits in L1.
///
/// Block sizes are 64×64×64 — three 64×64 tiles (A, B, C) is 48 KB of f32,
/// well under the 128 KB L1d on an M3 P-core. Outer loop order is (i, j, k)
/// so that the C tile stays "hot" while we sweep across K and accumulate
/// into it; A and B tiles are touched once per outer K step.
///
/// Edge handling: dimensions that aren't multiples of 64 produce smaller
/// blocks at the boundary, via `MC.min(m - ic)` etc. Block sizes are
/// constants for now; chapter-2 tuning could choose them per matrix size.
pub fn matmul_blocked(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    c.fill(0.0);

    const MC: usize = 64;
    const NC: usize = 64;
    const KC: usize = 64;

    for ic in (0..m).step_by(MC) {
        let mc = MC.min(m - ic);
        for jc in (0..n).step_by(NC) {
            let nc = NC.min(n - jc);
            for kc in (0..k).step_by(KC) {
                let kc_len = KC.min(k - kc);
                for i in 0..mc {
                    let a_row_off = (ic + i) * k + kc;
                    let c_row_off = (ic + i) * n + jc;
                    let c_row = &mut c[c_row_off..c_row_off + nc];
                    for kk in 0..kc_len {
                        let a_ik = a[a_row_off + kk];
                        let b_off = (kc + kk) * n + jc;
                        let b_row = &b[b_off..b_off + nc];
                        for j in 0..nc {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}
///
/// The win: the innermost loop now strides sequentially through *both* B and C
/// along their last (fastest-changing) dimension, instead of striding down a
/// column of B. `a[i*k+kk]` is loop-invariant in the inner loop and hoists out,
/// becoming a single load that fans out across `n` multiply-adds. Cache lines
/// are reused along the row instead of fetched once per column access.
///
/// One subtle cost: each `c[i*n+j]` is now written `k` times instead of once,
/// because we accumulate as we sweep across the K dimension. That requires
/// `c.fill(0.0)` up front — the function still produces `C = A·B`, not `C += A·B`.
pub fn matmul_ikj(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    n: usize,
    k: usize,
) {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    assert_eq!(c.len(), m * n);
    c.fill(0.0);
    for i in 0..m {
        for kk in 0..k {
            let a_ik = a[i * k + kk];
            let b_row = &b[kk * n..(kk + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];
            for j in 0..n {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_times_identity_is_identity() {
        let id: Vec<f32> = vec![1., 0., 0., 0., 1., 0., 0., 0., 1.];
        let mut out = vec![0.0_f32; 9];
        matmul_naive(&id, &id, &mut out, 3, 3, 3);
        assert_eq!(out, id);
    }

    #[test]
    fn two_by_two_known_answer() {
        // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1., 2., 3., 4.];
        let b = vec![5., 6., 7., 8.];
        let mut c = vec![0.0_f32; 4];
        matmul_naive(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn rectangular_two_by_three_times_three_by_two() {
        // A = [[1,2,3],[4,5,6]], B = [[1,2],[3,4],[5,6]]
        // row 0 · col 0 = 1*1+2*3+3*5 = 22; row 0 · col 1 = 1*2+2*4+3*6 = 28
        // row 1 · col 0 = 4*1+5*3+6*5 = 49; row 1 · col 1 = 4*2+5*4+6*6 = 64
        let a = vec![1., 2., 3., 4., 5., 6.];
        let b = vec![1., 2., 3., 4., 5., 6.];
        let mut c = vec![0.0_f32; 4];
        matmul_naive(&a, &b, &mut c, 2, 2, 3);
        assert_eq!(c, vec![22., 28., 49., 64.]);
    }

    /// Tiny deterministic pseudo-random fill, so optimization-equivalence tests
    /// don't depend on `rand`. Outputs values in roughly `[-1, 1)`.
    fn pseudo_fill(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
        (0..rows * cols)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn ikj_matches_naive_square() {
        let a = pseudo_fill(64, 48, 1);
        let b = pseudo_fill(48, 80, 2);
        let mut c_naive = vec![123.0_f32; 64 * 80]; // junk init verifies overwrite contract
        let mut c_ikj = vec![456.0_f32; 64 * 80];
        matmul_naive(&a, &b, &mut c_naive, 64, 80, 48);
        matmul_ikj(&a, &b, &mut c_ikj, 64, 80, 48);
        for i in 0..c_naive.len() {
            assert!(
                (c_naive[i] - c_ikj[i]).abs() < 1e-3,
                "mismatch at {}: {} vs {}",
                i, c_naive[i], c_ikj[i]
            );
        }
    }

    #[test]
    fn ikj_matches_naive_known_answer() {
        let a = vec![1., 2., 3., 4.];
        let b = vec![5., 6., 7., 8.];
        let mut c = vec![0.0_f32; 4];
        matmul_ikj(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19., 22., 43., 50.]);
    }

    #[test]
    fn ikj_zeroes_c_first() {
        // If the function forgot c.fill(0), the junk values would corrupt the result.
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let mut c = vec![999.0_f32; 4];
        matmul_ikj(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn blocked_matches_naive_block_aligned() {
        // 128 = 2 × 64, exercises clean block boundaries on all three dims.
        let a = pseudo_fill(128, 128, 7);
        let b = pseudo_fill(128, 128, 11);
        let mut c_naive = vec![0.0_f32; 128 * 128];
        let mut c_blocked = vec![0.0_f32; 128 * 128];
        matmul_naive(&a, &b, &mut c_naive, 128, 128, 128);
        matmul_blocked(&a, &b, &mut c_blocked, 128, 128, 128);
        for i in 0..c_naive.len() {
            assert!((c_naive[i] - c_blocked[i]).abs() < 1e-2);
        }
    }

    #[test]
    fn blocked_matches_naive_misaligned() {
        // 75×85×60 — none divisible by 64. Tests edge-block handling.
        let a = pseudo_fill(75, 60, 13);
        let b = pseudo_fill(60, 85, 17);
        let mut c_naive = vec![0.0_f32; 75 * 85];
        let mut c_blocked = vec![0.0_f32; 75 * 85];
        matmul_naive(&a, &b, &mut c_naive, 75, 85, 60);
        matmul_blocked(&a, &b, &mut c_blocked, 75, 85, 60);
        for i in 0..c_naive.len() {
            assert!(
                (c_naive[i] - c_blocked[i]).abs() < 1e-3,
                "mismatch at {}: {} vs {}",
                i, c_naive[i], c_blocked[i]
            );
        }
    }

    #[test]
    fn blocked_known_answer() {
        let a = vec![1., 2., 3., 4.];
        let b = vec![5., 6., 7., 8.];
        let mut c = vec![0.0_f32; 4];
        matmul_blocked(&a, &b, &mut c, 2, 2, 2);
        assert_eq!(c, vec![19., 22., 43., 50.]);
    }
}
