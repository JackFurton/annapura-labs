//! Compiler: lower high-level NN ops into accelerator `Instruction` sequences.
//!
//! Chapter 5.2's first move. Right now this only knows how to lower a single
//! linear (matvec); chapters 5.3+ will add RMSNorm, RoPE, attention, FFN,
//! and a full layer.
//!
//! Memory contract: the compiler emits instructions that assume specific
//! SRAM layouts. Inputs are contiguous f32. Weights MUST be pre-tiled by
//! `retile_weight` before being loaded into SRAM — this matches how real
//! ML accelerators (Trainium, Tensor Cores) expect data to arrive in the
//! layout their matmul units want.

use crate::accelerator::{Instruction, MATMUL_TILE, MATMUL_TILE_ELEMENTS, VECTOR_LANES};

/// Lower `y = W·x` into a sequence of `MatVecTile` instructions.
///
/// `in_dim` and `out_dim` must each be multiples of `MATMUL_TILE` (16).
/// At dispatch time, `x` must live at `sram[x_addr .. x_addr + in_dim]`,
/// the tiled weight at `sram[w_addr ..]`, and `y` will be written to
/// `sram[y_addr .. y_addr + out_dim]`.
///
/// The emitted sequence has `(out_dim / 16) * (in_dim / 16)` instructions:
/// for each 16-element output slab, accumulate across all 16-element input
/// slabs. First MatVecTile in each output tile uses `accumulate = false`
/// to reset, subsequent ones accumulate into the same y region.
pub fn compile_linear(
    in_dim: usize,
    out_dim: usize,
    x_addr: usize,
    w_addr: usize,
    y_addr: usize,
) -> Vec<Instruction> {
    assert_eq!(in_dim % MATMUL_TILE, 0, "in_dim must be a multiple of {}", MATMUL_TILE);
    assert_eq!(out_dim % MATMUL_TILE, 0, "out_dim must be a multiple of {}", MATMUL_TILE);

    let in_tiles = in_dim / MATMUL_TILE;
    let out_tiles = out_dim / MATMUL_TILE;
    let mut prog = Vec::with_capacity(in_tiles * out_tiles);

    for ot in 0..out_tiles {
        let y_tile_addr = y_addr + ot * MATMUL_TILE;
        for it in 0..in_tiles {
            let x_tile_addr = x_addr + it * MATMUL_TILE;
            let w_tile_addr = w_addr + (ot * in_tiles + it) * MATMUL_TILE_ELEMENTS;
            prog.push(Instruction::MatVecTile {
                x_sram: x_tile_addr,
                w_sram: w_tile_addr,
                y_sram: y_tile_addr,
                accumulate: it != 0,
            });
        }
    }
    prog
}

/// Lower `y = rmsnorm(x, gamma, eps)` into vector ops.
///
/// `n` must be a multiple of `VECTOR_LANES` (32). At dispatch time `x` lives
/// at `sram[x_addr..x_addr+n]`, `gamma` at `sram[gamma_addr..gamma_addr+n]`,
/// and `y` is written to `sram[y_addr..y_addr+n]`.
///
/// Algorithm:
///   1. v_acc = 0; for each 32-lane chunk: v_acc += chunk * chunk  (VFma)
///   2. broadcast(sum(v_acc)) → all lanes hold sum-of-squares  (VReduceSum)
///   3. v_acc *= 1/n; v_acc += eps; v_scale = rsqrt(v_acc)
///   4. for each chunk: y_chunk = (x_chunk * v_scale) * gamma_chunk
///
/// Register convention: v0=acc, v1=scale, v2=tmp_const, v3=x, v4=gamma, v5=tmp
pub fn compile_rmsnorm(
    n: usize,
    eps: f32,
    x_addr: usize,
    gamma_addr: usize,
    y_addr: usize,
) -> Vec<Instruction> {
    assert_eq!(n % VECTOR_LANES, 0, "n must be a multiple of {}", VECTOR_LANES);
    let n_chunks = n / VECTOR_LANES;
    let mut prog: Vec<Instruction> = Vec::with_capacity(2 + n_chunks + 5 + 4 * n_chunks);

    // v0 = 0  (sum-of-squares accumulator)
    prog.push(Instruction::VSplat { v: 0, scalar: 0.0 });
    // Pass 1: v0 += x_chunk * x_chunk for each chunk
    for i in 0..n_chunks {
        let off = x_addr + i * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 3, sram_addr: off });
        prog.push(Instruction::VFma { a: 3, b: 3, c: 0, d: 0 });
    }
    // Reduce + scalar arithmetic.
    prog.push(Instruction::VReduceSum { v_in: 0, v_out: 0 });   // sum of squares
    prog.push(Instruction::VSplat { v: 2, scalar: 1.0 / n as f32 });
    prog.push(Instruction::VMul { a: 0, b: 2, c: 0 });           // mean_sq
    prog.push(Instruction::VSplat { v: 2, scalar: eps });
    prog.push(Instruction::VAdd { a: 0, b: 2, c: 0 });           // mean_sq + eps
    prog.push(Instruction::VRsqrt { v_in: 0, v_out: 1 });        // v1 = scale

    // Pass 2: y_chunk = (x_chunk * scale) * gamma_chunk
    for i in 0..n_chunks {
        let off = i * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 3, sram_addr: x_addr + off });
        prog.push(Instruction::LoadVec { v: 4, sram_addr: gamma_addr + off });
        prog.push(Instruction::VMul { a: 3, b: 1, c: 5 });
        prog.push(Instruction::VMul { a: 5, b: 4, c: 5 });
        prog.push(Instruction::StoreVec { v: 5, sram_addr: y_addr + off });
    }
    prog
}

/// Re-layout a row-major `[out_dim, in_dim]` weight matrix into the tiled
/// layout expected by `compile_linear` / `MatVecTile`.
///
/// Output layout: `(out_dim / 16) * (in_dim / 16)` blocks, each 256 f32s,
/// concatenated. Block `(ot, it)` lives at offset `(ot * in_tiles + it) * 256`,
/// and within that block element `tile[k][j]` (at `block_off + k*16 + j`)
/// holds `W[ot*16 + j][it*16 + k]` — the transpose-within-tile that makes
/// `MatVecTile`'s loop produce the right dot products.
pub fn retile_weight(w_row_major: &[f32], in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(w_row_major.len(), in_dim * out_dim);
    assert_eq!(in_dim % MATMUL_TILE, 0);
    assert_eq!(out_dim % MATMUL_TILE, 0);

    let in_tiles = in_dim / MATMUL_TILE;
    let out_tiles = out_dim / MATMUL_TILE;
    let mut out = vec![0.0_f32; in_dim * out_dim];

    for ot in 0..out_tiles {
        for it in 0..in_tiles {
            let tile_off = (ot * in_tiles + it) * MATMUL_TILE_ELEMENTS;
            for k_tile in 0..MATMUL_TILE {
                for j_tile in 0..MATMUL_TILE {
                    let out_i = ot * MATMUL_TILE + j_tile;
                    let in_k = it * MATMUL_TILE + k_tile;
                    out[tile_off + k_tile * MATMUL_TILE + j_tile] =
                        w_row_major[out_i * in_dim + in_k];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::accelerator::Accelerator;

    /// Pseudo-random fill — keeps tests deterministic without depending on `rand`.
    fn pseudo(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as u32 as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect()
    }

    #[test]
    fn retile_preserves_data() {
        let in_dim = 32;
        let out_dim = 32;
        let w = pseudo(in_dim * out_dim, 1);
        let tiled = retile_weight(&w, in_dim, out_dim);
        // Spot check: tile[0,0] at k=2, j=5 should equal W[0*16 + 5][0*16 + 2] = W[5][2].
        let expected = w[5 * in_dim + 2];
        let got = tiled[2 * MATMUL_TILE + 5];
        assert_eq!(expected, got);
    }

    #[test]
    fn compile_linear_emits_expected_program_size() {
        let prog = compile_linear(64, 32, 0, 100, 200);
        // 32/16 output tiles × 64/16 input tiles = 2 × 4 = 8 MatVecTile instructions.
        assert_eq!(prog.len(), 8);
        // First instruction of each output tile resets; rest accumulate.
        for (i, instr) in prog.iter().enumerate() {
            if let Instruction::MatVecTile { accumulate, .. } = *instr {
                let position_in_tile = i % (64 / 16);
                let expected = position_in_tile != 0;
                assert_eq!(accumulate, expected, "instr {}: accumulate mismatch", i);
            } else {
                panic!("expected MatVecTile, got {:?}", instr);
            }
        }
    }

    /// THE money test: compile a linear, run it through the simulator,
    /// verify it produces the same answer as our CPU reference kernel.
    #[test]
    fn end_to_end_linear_matches_cpu_reference() {
        let in_dim = 64;
        let out_dim = 32;
        let x = pseudo(in_dim, 7);
        let w = pseudo(in_dim * out_dim, 11);

        // CPU reference (the oracle).
        let mut y_cpu = vec![0.0_f32; out_dim];
        crate::nn::linear(&x, &w, &mut y_cpu);

        // Simulator path: retile, set up SRAM, compile program, run.
        let w_tiled = retile_weight(&w, in_dim, out_dim);

        let sram_size = in_dim + (in_dim * out_dim) + out_dim;
        let mut acc = Accelerator::new(sram_size, 0);
        let x_addr = 0;
        let w_addr = in_dim;
        let y_addr = w_addr + in_dim * out_dim;
        acc.sram[x_addr..x_addr + in_dim].copy_from_slice(&x);
        acc.sram[w_addr..w_addr + in_dim * out_dim].copy_from_slice(&w_tiled);

        let program = compile_linear(in_dim, out_dim, x_addr, w_addr, y_addr);
        let n_instr = program.len();
        acc.run(&program).unwrap();
        assert_eq!(acc.instr_count, n_instr as u64);

        for j in 0..out_dim {
            let got = acc.sram[y_addr + j];
            let want = y_cpu[j];
            assert!(
                (got - want).abs() < 1e-3,
                "y[{}]: simulator {} vs CPU {} (diff {})",
                j, got, want, (got - want).abs()
            );
        }
    }

    #[test]
    fn end_to_end_rmsnorm_matches_cpu_reference() {
        let n = 256;
        let eps = 1e-5;
        let x = pseudo(n, 23);
        let gamma = pseudo(n, 29);

        let mut y_cpu = vec![0.0_f32; n];
        crate::nn::rmsnorm(&x, &gamma, eps, &mut y_cpu);

        let sram_size = 3 * n;
        let mut acc = Accelerator::new(sram_size, 0);
        let x_addr = 0;
        let g_addr = n;
        let y_addr = 2 * n;
        acc.sram[x_addr..x_addr + n].copy_from_slice(&x);
        acc.sram[g_addr..g_addr + n].copy_from_slice(&gamma);

        let program = compile_rmsnorm(n, eps, x_addr, g_addr, y_addr);
        acc.run(&program).unwrap();

        for i in 0..n {
            let got = acc.sram[y_addr + i];
            let want = y_cpu[i];
            assert!(
                (got - want).abs() < 1e-4,
                "y[{}]: simulator {} vs CPU {} (diff {})",
                i, got, want, (got - want).abs()
            );
        }
    }

    /// Larger case: 256×128. ~128 MatVecTile instructions, more chances for
    /// indexing bugs to manifest.
    #[test]
    fn end_to_end_larger_linear_matches_cpu_reference() {
        let in_dim = 256;
        let out_dim = 128;
        let x = pseudo(in_dim, 13);
        let w = pseudo(in_dim * out_dim, 17);

        let mut y_cpu = vec![0.0_f32; out_dim];
        crate::nn::linear(&x, &w, &mut y_cpu);

        let w_tiled = retile_weight(&w, in_dim, out_dim);
        let sram_size = in_dim + (in_dim * out_dim) + out_dim;
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[0..in_dim].copy_from_slice(&x);
        acc.sram[in_dim..in_dim + in_dim * out_dim].copy_from_slice(&w_tiled);

        let program = compile_linear(in_dim, out_dim, 0, in_dim, in_dim + in_dim * out_dim);
        acc.run(&program).unwrap();

        for j in 0..out_dim {
            let got = acc.sram[in_dim + in_dim * out_dim + j];
            let want = y_cpu[j];
            assert!((got - want).abs() < 1e-3, "y[{}]: {} vs {}", j, got, want);
        }
    }
}
