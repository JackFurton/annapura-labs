use std::hint::black_box;

use annapura::matmul_naive;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

fn bench_matmul_naive(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_naive_square_f32");
    for &n in &[64usize, 128, 256, 512] {
        let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.001).collect();
        let b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.001).collect();
        let mut out = vec![0.0_f32; n * n];
        // 2·n³ FLOPs per call (n² outputs × n multiply-adds).
        // criterion labels this "elem/s" — read it as FLOP/s.
        group.throughput(Throughput::Elements((2 * n * n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, &nn| {
            bencher.iter(|| {
                matmul_naive(black_box(&a), black_box(&b), &mut out, nn, nn, nn);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul_naive);
criterion_main!(benches);
