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
}
