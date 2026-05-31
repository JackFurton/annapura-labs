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

/// Lower `c = a + b` (elementwise) into vector ops. `n` must be a multiple
/// of `VECTOR_LANES`. Aliases are allowed (e.g. `a_addr == c_addr` for
/// in-place residual `x += y`).
pub fn compile_add(n: usize, a_addr: usize, b_addr: usize, c_addr: usize) -> Vec<Instruction> {
    assert_eq!(n % VECTOR_LANES, 0, "n must be a multiple of {}", VECTOR_LANES);
    let n_chunks = n / VECTOR_LANES;
    let mut prog = Vec::with_capacity(n_chunks * 4);
    for i in 0..n_chunks {
        let off = i * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 0, sram_addr: a_addr + off });
        prog.push(Instruction::LoadVec { v: 1, sram_addr: b_addr + off });
        prog.push(Instruction::VAdd { a: 0, b: 1, c: 2 });
        prog.push(Instruction::StoreVec { v: 2, sram_addr: c_addr + off });
    }
    prog
}

/// Lower `c = a * b` (elementwise) into vector ops. See `compile_add`.
pub fn compile_mul(n: usize, a_addr: usize, b_addr: usize, c_addr: usize) -> Vec<Instruction> {
    assert_eq!(n % VECTOR_LANES, 0, "n must be a multiple of {}", VECTOR_LANES);
    let n_chunks = n / VECTOR_LANES;
    let mut prog = Vec::with_capacity(n_chunks * 4);
    for i in 0..n_chunks {
        let off = i * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 0, sram_addr: a_addr + off });
        prog.push(Instruction::LoadVec { v: 1, sram_addr: b_addr + off });
        prog.push(Instruction::VMul { a: 0, b: 1, c: 2 });
        prog.push(Instruction::StoreVec { v: 2, sram_addr: c_addr + off });
    }
    prog
}

/// Lower `out = silu(in)` (lanewise) into vector ops. In-place if
/// `in_addr == out_addr`.
pub fn compile_silu(n: usize, in_addr: usize, out_addr: usize) -> Vec<Instruction> {
    assert_eq!(n % VECTOR_LANES, 0, "n must be a multiple of {}", VECTOR_LANES);
    let n_chunks = n / VECTOR_LANES;
    let mut prog = Vec::with_capacity(n_chunks * 3);
    for i in 0..n_chunks {
        let off = i * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 0, sram_addr: in_addr + off });
        prog.push(Instruction::VSilu { v_in: 0, v_out: 1 });
        prog.push(Instruction::StoreVec { v: 1, sram_addr: out_addr + off });
    }
    prog
}

/// Build the cos / signed-sin tables that `compile_rope` expects in SRAM.
///
/// Returns `(cos_table, sin_pm_table)`, each `head_dim` floats. The layout
/// matches what the lowering loop reads: chunk `c` of either table sits at
/// offset `c * VECTOR_LANES`. Within a chunk, both lanes of each pair hold
/// the same cosine; the sin table interleaves `(-s, +s)` so that one
/// `VMul + VFma` produces the rotated pair without extra sign handling.
///
/// Convention: interleaved pairs `(x[2i], x[2i+1])`, the same as
/// `nn::rope_inplace` / llama.cpp.
pub fn build_rope_tables(head_dim: usize, pos: usize, freq_base: f32) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(head_dim % VECTOR_LANES, 0, "head_dim must be a multiple of {}", VECTOR_LANES);
    assert!(head_dim % 2 == 0, "head_dim must be even");
    let chunks_per_head = head_dim / VECTOR_LANES;
    let pairs_per_chunk = VECTOR_LANES / 2;
    let mut cos = vec![0.0_f32; head_dim];
    let mut sin_pm = vec![0.0_f32; head_dim];
    let pos = pos as f32;
    for c in 0..chunks_per_head {
        for p in 0..pairs_per_chunk {
            let pair_idx = c * pairs_per_chunk + p; // index of this pair within the head
            let omega = freq_base.powf(-2.0 * pair_idx as f32 / head_dim as f32);
            let (s, co) = (pos * omega).sin_cos();
            let base = c * VECTOR_LANES + 2 * p;
            cos[base] = co;
            cos[base + 1] = co;
            sin_pm[base] = -s;
            sin_pm[base + 1] = s;
        }
    }
    (cos, sin_pm)
}

/// Lower RoPE (rotary position embedding) over a multi-head vector.
///
/// In place: writes the rotated values back to `x_addr`. `n` must be a
/// multiple of `head_dim`, and `head_dim` a multiple of `VECTOR_LANES`. The
/// cos / sin tables must already live at `cos_addr` / `sin_pm_addr`,
/// produced by `build_rope_tables` for the desired `(pos, freq_base)`.
///
/// Per-chunk recipe (7 instructions): load x, load cos, load sin_pm,
/// swap_pairs(x), x*cos, fma(swap, sin_pm, x*cos), store. The
/// signed-sin table absorbs the (a·c - b·s, a·s + b·c) sign pattern so the
/// math reduces to one lanewise multiply-add.
pub fn compile_rope(
    n: usize,
    head_dim: usize,
    x_addr: usize,
    cos_addr: usize,
    sin_pm_addr: usize,
) -> Vec<Instruction> {
    assert_eq!(n % head_dim, 0, "n must be a multiple of head_dim");
    assert_eq!(head_dim % VECTOR_LANES, 0, "head_dim must be a multiple of {}", VECTOR_LANES);
    let n_chunks = n / VECTOR_LANES;
    let chunks_per_head = head_dim / VECTOR_LANES;
    let mut prog = Vec::with_capacity(n_chunks * 7);

    for chunk in 0..n_chunks {
        let chunk_in_head = chunk % chunks_per_head;
        let x_off = chunk * VECTOR_LANES;
        let table_off = chunk_in_head * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 0, sram_addr: x_addr + x_off });
        prog.push(Instruction::LoadVec { v: 1, sram_addr: cos_addr + table_off });
        prog.push(Instruction::LoadVec { v: 2, sram_addr: sin_pm_addr + table_off });
        prog.push(Instruction::VSwapPairs { v_in: 0, v_out: 3 });
        prog.push(Instruction::VMul { a: 0, b: 1, c: 4 });
        prog.push(Instruction::VFma { a: 3, b: 2, c: 4, d: 5 });
        prog.push(Instruction::StoreVec { v: 5, sram_addr: x_addr + x_off });
    }
    prog
}

/// Lower a SwiGLU FFN: `out = W_down · (silu(W_gate · x) ⊙ (W_up · x))`.
///
/// Weight layouts: all three matrices must already be re-tiled by
/// `retile_weight`. `W_gate` and `W_up` are `[ffn_hidden, hidden]` matrices,
/// `W_down` is `[hidden, ffn_hidden]`. `gate_buf` and `up_buf` are scratch
/// regions of size `ffn_hidden` each.
///
/// Program shape: 3× compile_linear + compile_silu + compile_mul. All
/// composition, no new ISA — the simulator already has every primitive needed
/// for the entire FFN block.
#[allow(clippy::too_many_arguments)]
pub fn compile_ffn(
    hidden: usize,
    ffn_hidden: usize,
    x_addr: usize,
    w_gate_addr: usize,
    w_up_addr: usize,
    w_down_addr: usize,
    gate_buf_addr: usize,
    up_buf_addr: usize,
    out_addr: usize,
) -> Vec<Instruction> {
    let mut prog = Vec::new();
    prog.extend(compile_linear(hidden, ffn_hidden, x_addr, w_gate_addr, gate_buf_addr));
    prog.extend(compile_linear(hidden, ffn_hidden, x_addr, w_up_addr, up_buf_addr));
    prog.extend(compile_silu(ffn_hidden, gate_buf_addr, gate_buf_addr));
    prog.extend(compile_mul(ffn_hidden, gate_buf_addr, up_buf_addr, gate_buf_addr));
    prog.extend(compile_linear(ffn_hidden, hidden, gate_buf_addr, w_down_addr, out_addr));
    prog
}

/// Lower numerically-stable softmax over `n ≤ VECTOR_LANES` scores into
/// vector ops. The whole softmax fits in a single 32-lane vector — multi-chunk
/// softmax (which a long-context attention needs) is chapter 5.8b.
///
/// In place: reads `n` scores from `x_addr` and writes them back. The caller
/// is expected to have prepared a length-`VECTOR_LANES` slot starting at
/// `x_addr`, with the trailing lanes (after the `n` valid entries) already
/// pre-loaded with a large negative number (e.g. `-1e30`) so the
/// subtract-max trick zeros their `exp` contribution. The bundled
/// `attention_mask` helper builds that layout.
///
/// Algorithm (the textbook three-pass on one vector):
///   1. m = broadcast(max(x))                  (VReduceMax)
///   2. shifted = x - m                        (VSplat -1, VMul, VAdd)
///   3. e = exp(shifted)                       (VExp)
///   4. s = broadcast(sum(e))                  (VReduceSum)
///   5. 1/s via rsqrt(s*s)                     (VMul, VRsqrt)
///   6. out = e * (1/s)                        (VMul)
///
/// Register convention: v0 = x / shifted / e / out, v1 = m / inv_s,
/// v2 = neg-one constant, v3 = scratch.
pub fn compile_softmax(n: usize, x_addr: usize) -> Vec<Instruction> {
    assert!(n <= VECTOR_LANES, "compile_softmax only supports n ≤ {} (chapter 5.8a)", VECTOR_LANES);
    assert!(n > 0);
    let mut prog = Vec::with_capacity(12);
    prog.push(Instruction::LoadVec { v: 0, sram_addr: x_addr });
    prog.push(Instruction::VReduceMax { v_in: 0, v_out: 1 });
    prog.push(Instruction::VSplat { v: 2, scalar: -1.0 });
    prog.push(Instruction::VMul { a: 1, b: 2, c: 1 });           // v1 = -max
    prog.push(Instruction::VAdd { a: 0, b: 1, c: 0 });           // v0 = x - max
    prog.push(Instruction::VExp { v_in: 0, v_out: 0 });          // v0 = exp(x - max)
    prog.push(Instruction::VReduceSum { v_in: 0, v_out: 1 });    // v1 = sum
    prog.push(Instruction::VMul { a: 1, b: 1, c: 3 });           // v3 = sum*sum
    prog.push(Instruction::VRsqrt { v_in: 3, v_out: 1 });        // v1 = 1/sqrt(sum*sum) = 1/sum
    prog.push(Instruction::VMul { a: 0, b: 1, c: 0 });           // v0 = exp / sum
    prog.push(Instruction::StoreVec { v: 0, sram_addr: x_addr });
    prog
}

/// Expand a per-position vector `q` into the lane-broadcast layout that
/// `compile_attention_head` reads: `out[d * VECTOR_LANES + lane] = q[d]`
/// for `lane ∈ [0, VECTOR_LANES)`. The lowering uses each broadcast vector
/// to multiply across all 32 positions in parallel — one VFma per head_dim
/// component instead of one per (position, component).
pub fn broadcast_q(q: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0_f32; q.len() * VECTOR_LANES];
    for (d, &v) in q.iter().enumerate() {
        for lane in 0..VECTOR_LANES {
            out[d * VECTOR_LANES + lane] = v;
        }
    }
    out
}

/// Pack `seq_len` cached K vectors (each `head_dim` long) into the
/// transposed lane-parallel layout `compile_attention_head` reads. Output is
/// `head_dim * VECTOR_LANES` floats: `out[d * VECTOR_LANES + p] = k[p][d]`,
/// with lanes `p ≥ seq_len` filled with zero (their scores will be masked
/// to `-inf` before softmax, so the zero placeholders never reach the
/// weighted-V sum).
///
/// `seq_len` must be ≤ `VECTOR_LANES` (32). Caller passes K as a flat
/// `[seq_len, head_dim]` row-major buffer (the same layout as `KvCache.k`
/// for one kv head).
pub fn transpose_k_for_lanes(k: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    assert!(seq_len <= VECTOR_LANES);
    assert_eq!(k.len(), seq_len * head_dim);
    let mut out = vec![0.0_f32; head_dim * VECTOR_LANES];
    for p in 0..seq_len {
        for d in 0..head_dim {
            out[d * VECTOR_LANES + p] = k[p * head_dim + d];
        }
    }
    out
}

/// Build the per-lane additive softmax mask: 0.0 for valid positions
/// `[0, seq_len)`, `-1e30` for the padded suffix. Added to the score vector
/// before softmax so padding lanes contribute zero probability mass.
pub fn attention_mask(seq_len: usize) -> Vec<f32> {
    assert!(seq_len <= VECTOR_LANES);
    let mut m = vec![0.0_f32; VECTOR_LANES];
    for lane in seq_len..VECTOR_LANES {
        m[lane] = -1e30;
    }
    m
}

/// Lower scaled dot-product attention for one query head with `seq_len`
/// cached positions (`seq_len ≤ VECTOR_LANES`).
///
/// SRAM contract (all addresses contiguous regions of f32):
///   - `q_broadcast_addr`: `head_dim * VECTOR_LANES` floats, from `broadcast_q`
///   - `k_t_addr`:         `head_dim * VECTOR_LANES` floats, from `transpose_k_for_lanes`
///   - `v_addr`:           `seq_len * head_dim` floats (one row per cached position)
///   - `mask_addr`:        `VECTOR_LANES` floats, from `attention_mask(seq_len)`
///   - `scores_addr`:      `VECTOR_LANES` scratch floats (overwritten)
///   - `out_addr`:         `head_dim` floats (the per-head output)
///
/// `head_dim` must be a multiple of `VECTOR_LANES`. `inv_sqrt_d = 1 /
/// sqrt(head_dim)` is the standard attention scale.
///
/// Program shape:
///   Phase A — scores: head_dim FMAs over (q_broadcast[d], k_t[d]) into one
///             accumulator vector; one scale by inv_sqrt_d.
///   Phase B — masked softmax: store the score vector, run compile_softmax
///             over the masked scores in-place.
///   Phase C — weighted V sum: per d_chunk, accumulate
///             `Σ_p VBroadcastLane(scores, p) * V[p][d_chunk]` and store.
#[allow(clippy::too_many_arguments)]
pub fn compile_attention_head(
    head_dim: usize,
    seq_len: usize,
    q_broadcast_addr: usize,
    k_t_addr: usize,
    v_addr: usize,
    mask_addr: usize,
    scores_addr: usize,
    out_addr: usize,
    inv_sqrt_d: f32,
) -> Vec<Instruction> {
    assert_eq!(head_dim % VECTOR_LANES, 0, "head_dim must be a multiple of {}", VECTOR_LANES);
    assert!(seq_len > 0 && seq_len <= VECTOR_LANES);
    let d_chunks = head_dim / VECTOR_LANES;
    let mut prog: Vec<Instruction> = Vec::new();

    // Phase A: lane-parallel scores. Lane p of v_acc holds (q · k_p).
    prog.push(Instruction::VSplat { v: 10, scalar: 0.0 });        // v10 = score accumulator
    for d in 0..head_dim {
        prog.push(Instruction::LoadVec { v: 11, sram_addr: q_broadcast_addr + d * VECTOR_LANES });
        prog.push(Instruction::LoadVec { v: 12, sram_addr: k_t_addr + d * VECTOR_LANES });
        prog.push(Instruction::VFma { a: 11, b: 12, c: 10, d: 10 });
    }
    // Scale by 1/sqrt(head_dim).
    prog.push(Instruction::VSplat { v: 11, scalar: inv_sqrt_d });
    prog.push(Instruction::VMul { a: 10, b: 11, c: 10 });
    // Apply additive mask (zero on valid lanes, -1e30 on padded).
    prog.push(Instruction::LoadVec { v: 11, sram_addr: mask_addr });
    prog.push(Instruction::VAdd { a: 10, b: 11, c: 10 });
    // Spill masked scores so compile_softmax can pick them up.
    prog.push(Instruction::StoreVec { v: 10, sram_addr: scores_addr });

    // Phase B: in-place softmax on the score vector.
    prog.extend(compile_softmax(seq_len, scores_addr));
    // After softmax, reload scores into v10 (the lane-parallel weights).
    prog.push(Instruction::LoadVec { v: 10, sram_addr: scores_addr });

    // Phase C: out[d_chunk] = Σ_p scores[p] · V[p][d_chunk]. Each d_chunk is
    // a 32-lane slice of `head_dim`. For each cached position p we broadcast
    // lane p of the score vector and FMA into the running output accumulator.
    for dc in 0..d_chunks {
        prog.push(Instruction::VSplat { v: 13, scalar: 0.0 });    // v13 = out accumulator
        for p in 0..seq_len {
            prog.push(Instruction::VBroadcastLane { v_in: 10, v_out: 11, lane: p as u8 });
            prog.push(Instruction::LoadVec {
                v: 12,
                sram_addr: v_addr + p * head_dim + dc * VECTOR_LANES,
            });
            prog.push(Instruction::VFma { a: 11, b: 12, c: 13, d: 13 });
        }
        prog.push(Instruction::StoreVec { v: 13, sram_addr: out_addr + dc * VECTOR_LANES });
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

    #[test]
    fn compile_add_matches_cpu() {
        let n = 64;
        let a = pseudo(n, 41);
        let b = pseudo(n, 43);
        let mut acc = Accelerator::new(3 * n, 0);
        acc.sram[0..n].copy_from_slice(&a);
        acc.sram[n..2 * n].copy_from_slice(&b);
        acc.run(&compile_add(n, 0, n, 2 * n)).unwrap();
        for i in 0..n {
            let want = a[i] + b[i];
            assert!((acc.sram[2 * n + i] - want).abs() < 1e-6, "i={}", i);
        }
    }

    #[test]
    fn compile_add_in_place_aliasing_works() {
        // a_addr == c_addr: residual-style x += y.
        let n = 32;
        let a = pseudo(n, 47);
        let b = pseudo(n, 53);
        let mut acc = Accelerator::new(2 * n, 0);
        acc.sram[0..n].copy_from_slice(&a);
        acc.sram[n..2 * n].copy_from_slice(&b);
        acc.run(&compile_add(n, 0, n, 0)).unwrap();
        for i in 0..n {
            let want = a[i] + b[i];
            assert!((acc.sram[i] - want).abs() < 1e-6, "i={}", i);
        }
    }

    #[test]
    fn compile_mul_matches_cpu() {
        let n = 64;
        let a = pseudo(n, 59);
        let b = pseudo(n, 61);
        let mut acc = Accelerator::new(3 * n, 0);
        acc.sram[0..n].copy_from_slice(&a);
        acc.sram[n..2 * n].copy_from_slice(&b);
        acc.run(&compile_mul(n, 0, n, 2 * n)).unwrap();
        for i in 0..n {
            let want = a[i] * b[i];
            assert!((acc.sram[2 * n + i] - want).abs() < 1e-6, "i={}", i);
        }
    }

    #[test]
    fn compile_silu_matches_cpu() {
        let n = 64;
        let mut x_cpu = pseudo(n, 67);
        let mut acc = Accelerator::new(2 * n, 0);
        acc.sram[0..n].copy_from_slice(&x_cpu);
        acc.run(&compile_silu(n, 0, n)).unwrap();
        crate::nn::silu_in_place(&mut x_cpu);
        for i in 0..n {
            assert!((acc.sram[n + i] - x_cpu[i]).abs() < 1e-6, "i={}", i);
        }
    }

    /// Lowering matches CPU `rope_inplace` on a single 64-element head at a
    /// non-zero position — the basic correctness check.
    #[test]
    fn compile_rope_single_head_matches_cpu() {
        let head_dim = 64;
        let pos = 17;
        let freq_base = 10000.0;
        let mut x_cpu = pseudo(head_dim, 89);
        let x_sim_input = x_cpu.clone();

        crate::nn::rope_inplace(&mut x_cpu, pos, freq_base);

        let (cos, sin_pm) = build_rope_tables(head_dim, pos, freq_base);
        let x_addr = 0;
        let cos_addr = head_dim;
        let sin_addr = cos_addr + head_dim;
        let sram_size = sin_addr + head_dim;
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[x_addr..x_addr + head_dim].copy_from_slice(&x_sim_input);
        acc.sram[cos_addr..cos_addr + head_dim].copy_from_slice(&cos);
        acc.sram[sin_addr..sin_addr + head_dim].copy_from_slice(&sin_pm);

        let program = compile_rope(head_dim, head_dim, x_addr, cos_addr, sin_addr);
        acc.run(&program).unwrap();

        for i in 0..head_dim {
            let got = acc.sram[x_addr + i];
            let want = x_cpu[i];
            assert!(
                (got - want).abs() < 1e-5,
                "rope[{}]: simulator {} vs CPU {} (diff {})",
                i, got, want, (got - want).abs()
            );
        }
    }

    /// Multi-head: matches CPU `rope_heads` across 8 heads of dim 64 (similar
    /// to TinyLlama Q's [32 heads × 64 head_dim] structure, scaled down).
    #[test]
    fn compile_rope_multi_head_matches_cpu() {
        let head_dim = 64;
        let n_heads = 8;
        let n = head_dim * n_heads;
        let pos = 23;
        let freq_base = 10000.0;
        let mut x_cpu = pseudo(n, 97);
        let x_sim_input = x_cpu.clone();

        crate::nn::rope_heads(&mut x_cpu, head_dim, pos, freq_base);

        let (cos, sin_pm) = build_rope_tables(head_dim, pos, freq_base);
        let x_addr = 0;
        let cos_addr = n;
        let sin_addr = cos_addr + head_dim;
        let sram_size = sin_addr + head_dim;
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[x_addr..x_addr + n].copy_from_slice(&x_sim_input);
        acc.sram[cos_addr..cos_addr + head_dim].copy_from_slice(&cos);
        acc.sram[sin_addr..sin_addr + head_dim].copy_from_slice(&sin_pm);

        let program = compile_rope(n, head_dim, x_addr, cos_addr, sin_addr);
        acc.run(&program).unwrap();

        for i in 0..n {
            let got = acc.sram[x_addr + i];
            let want = x_cpu[i];
            assert!(
                (got - want).abs() < 1e-5,
                "rope[{}] (head {}, lane {}): simulator {} vs CPU {} (diff {})",
                i, i / head_dim, i % head_dim, got, want, (got - want).abs()
            );
        }
    }

    /// Position 0 is the identity rotation — sanity check that the table
    /// builder produces cos=1, sin=0 and the lowering passes x through.
    #[test]
    fn compile_rope_at_position_zero_is_identity() {
        let head_dim = 32;
        let x = pseudo(head_dim, 101);
        let (cos, sin_pm) = build_rope_tables(head_dim, 0, 10000.0);
        let x_addr = 0;
        let cos_addr = head_dim;
        let sin_addr = cos_addr + head_dim;
        let mut acc = Accelerator::new(sin_addr + head_dim, 0);
        acc.sram[x_addr..x_addr + head_dim].copy_from_slice(&x);
        acc.sram[cos_addr..cos_addr + head_dim].copy_from_slice(&cos);
        acc.sram[sin_addr..sin_addr + head_dim].copy_from_slice(&sin_pm);
        let program = compile_rope(head_dim, head_dim, x_addr, cos_addr, sin_addr);
        acc.run(&program).unwrap();
        for i in 0..head_dim {
            assert!((acc.sram[x_addr + i] - x[i]).abs() < 1e-6, "i={}", i);
        }
    }

    /// The big composite test: lower a full SwiGLU FFN and verify it agrees
    /// with the CPU reference, end to end. Uses small synthetic sizes
    /// (hidden=64, ffn_hidden=128) so the test is fast.
    #[test]
    fn end_to_end_ffn_matches_cpu_reference() {
        let hidden = 64;
        let ffn_hidden = 128;
        let x = pseudo(hidden, 71);
        let w_gate = pseudo(hidden * ffn_hidden, 73);
        let w_up = pseudo(hidden * ffn_hidden, 79);
        let w_down = pseudo(ffn_hidden * hidden, 83);

        // CPU reference: gate -> silu -> mul up -> down.
        let mut gate_cpu = vec![0.0_f32; ffn_hidden];
        let mut up_cpu = vec![0.0_f32; ffn_hidden];
        crate::nn::linear(&x, &w_gate, &mut gate_cpu);
        crate::nn::linear(&x, &w_up, &mut up_cpu);
        crate::nn::silu_in_place(&mut gate_cpu);
        crate::nn::mul_in_place(&mut gate_cpu, &up_cpu);
        let mut y_cpu = vec![0.0_f32; hidden];
        crate::nn::linear(&gate_cpu, &w_down, &mut y_cpu);

        // Simulator path: retile all three weight matrices, set up SRAM, run.
        let wg_tiled = retile_weight(&w_gate, hidden, ffn_hidden);
        let wu_tiled = retile_weight(&w_up, hidden, ffn_hidden);
        let wd_tiled = retile_weight(&w_down, ffn_hidden, hidden);

        let x_addr = 0;
        let wg_addr = x_addr + hidden;
        let wu_addr = wg_addr + hidden * ffn_hidden;
        let wd_addr = wu_addr + hidden * ffn_hidden;
        let gate_addr = wd_addr + ffn_hidden * hidden;
        let up_addr = gate_addr + ffn_hidden;
        let out_addr = up_addr + ffn_hidden;
        let sram_size = out_addr + hidden;

        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[x_addr..x_addr + hidden].copy_from_slice(&x);
        acc.sram[wg_addr..wg_addr + hidden * ffn_hidden].copy_from_slice(&wg_tiled);
        acc.sram[wu_addr..wu_addr + hidden * ffn_hidden].copy_from_slice(&wu_tiled);
        acc.sram[wd_addr..wd_addr + ffn_hidden * hidden].copy_from_slice(&wd_tiled);

        let program = compile_ffn(
            hidden, ffn_hidden,
            x_addr, wg_addr, wu_addr, wd_addr,
            gate_addr, up_addr, out_addr,
        );
        acc.run(&program).unwrap();

        for i in 0..hidden {
            let got = acc.sram[out_addr + i];
            let want = y_cpu[i];
            assert!(
                (got - want).abs() < 1e-3,
                "y[{}]: simulator {} vs CPU {} (diff {})",
                i, got, want, (got - want).abs()
            );
        }
    }

    // ===== Chapter 5.8a: softmax + single-head attention =====

    fn run_softmax_on_sim(scores: &[f32]) -> Vec<f32> {
        // Match the contract: caller fills the valid lanes with the score
        // values and the trailing lanes with -1e30 so they don't poison max.
        let mut sram = vec![0.0_f32; VECTOR_LANES];
        for (i, &v) in scores.iter().enumerate() {
            sram[i] = v;
        }
        for lane in scores.len()..VECTOR_LANES {
            sram[lane] = -1e30;
        }
        let mut acc = Accelerator::new(VECTOR_LANES, 0);
        acc.sram.copy_from_slice(&sram);
        let prog = compile_softmax(scores.len(), 0);
        acc.run(&prog).unwrap();
        acc.sram[..scores.len()].to_vec()
    }

    #[test]
    fn compile_softmax_uniform_input_gives_uniform_output() {
        let out = run_softmax_on_sim(&[1.0; 8]);
        for v in &out {
            assert!((v - 1.0 / 8.0).abs() < 1e-6, "got {}", v);
        }
    }

    #[test]
    fn compile_softmax_one_hot_concentrates_mass() {
        let mut x = vec![0.0_f32; 4];
        x[1] = 100.0;
        let out = run_softmax_on_sim(&x);
        assert!(out[1] > 0.999);
        for (i, &v) in out.iter().enumerate() {
            if i != 1 {
                assert!(v < 1e-4, "lane {} = {}", i, v);
            }
        }
    }

    #[test]
    fn compile_softmax_matches_cpu_for_random_inputs() {
        for (n, seed) in [(1usize, 41), (4, 43), (16, 47), (32, 53)] {
            let mut x = pseudo(n, seed);
            // Push some lanes large so the max-subtract path gets exercised.
            x[0] += 5.0;
            let sim = run_softmax_on_sim(&x);
            let mut cpu = x.clone();
            crate::nn::softmax_in_place(&mut cpu);
            for i in 0..n {
                assert!(
                    (sim[i] - cpu[i]).abs() < 1e-5,
                    "n={} lane {}: sim {} vs cpu {}",
                    n, i, sim[i], cpu[i]
                );
            }
            let s: f32 = sim.iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "n={} sum={}", n, s);
        }
    }

    /// Cross-check `compile_attention_head` against `attention::attention` for
    /// a single query head and `seq_len ≤ VECTOR_LANES`.
    fn run_attention_head_on_sim(
        q: &[f32],
        k_cache: &[f32],  // [seq_len, head_dim]
        v_cache: &[f32],  // [seq_len, head_dim]
        seq_len: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let q_bcast = broadcast_q(q);
        let k_t = transpose_k_for_lanes(k_cache, seq_len, head_dim);
        let mask = attention_mask(seq_len);
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

        // SRAM layout: q_bcast | k_t | v | mask | scores | out
        let q_addr = 0;
        let k_addr = q_addr + q_bcast.len();
        let v_addr = k_addr + k_t.len();
        let m_addr = v_addr + v_cache.len();
        let s_addr = m_addr + mask.len();
        let o_addr = s_addr + VECTOR_LANES;
        let sram_size = o_addr + head_dim;
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[q_addr..q_addr + q_bcast.len()].copy_from_slice(&q_bcast);
        acc.sram[k_addr..k_addr + k_t.len()].copy_from_slice(&k_t);
        acc.sram[v_addr..v_addr + v_cache.len()].copy_from_slice(v_cache);
        acc.sram[m_addr..m_addr + mask.len()].copy_from_slice(&mask);

        let prog = compile_attention_head(
            head_dim, seq_len, q_addr, k_addr, v_addr, m_addr, s_addr, o_addr, inv_sqrt_d,
        );
        acc.run(&prog).unwrap();
        acc.sram[o_addr..o_addr + head_dim].to_vec()
    }

    #[test]
    fn compile_attention_head_seq1_returns_v0() {
        // Single position: softmax of one score is 1.0, so output = V[0].
        let head_dim = 32;
        let q = pseudo(head_dim, 71);
        let v0: Vec<f32> = (0..head_dim).map(|i| (i as f32) * 0.5).collect();
        let k0 = pseudo(head_dim, 73);
        let sim = run_attention_head_on_sim(&q, &k0, &v0, 1, head_dim);
        for d in 0..head_dim {
            assert!((sim[d] - v0[d]).abs() < 1e-5, "lane {}: {} vs {}", d, sim[d], v0[d]);
        }
    }

    #[test]
    fn compile_attention_head_matches_cpu_for_small_seq() {
        use crate::attention::{attention, KvCache};

        let head_dim = 64;
        let n_heads = 1;
        let n_kv_heads = 1;
        let kv_dim = head_dim;

        for (seq_len, seed) in [(4_usize, 101), (16, 103), (32, 107)] {
            let q = pseudo(head_dim, seed);
            let k_flat = pseudo(seq_len * head_dim, seed + 1);
            let v_flat = pseudo(seq_len * head_dim, seed + 2);

            // CPU oracle.
            let mut cache = KvCache::new(seq_len, kv_dim);
            for p in 0..seq_len {
                let row = &k_flat[p * head_dim..(p + 1) * head_dim];
                let vrow = &v_flat[p * head_dim..(p + 1) * head_dim];
                cache.store(p, row, vrow);
            }
            let mut out_cpu = vec![0.0_f32; head_dim];
            attention(&q, &cache, seq_len - 1, n_heads, n_kv_heads, head_dim, &mut out_cpu);

            // Simulator.
            let out_sim = run_attention_head_on_sim(&q, &k_flat, &v_flat, seq_len, head_dim);

            for d in 0..head_dim {
                assert!(
                    (out_sim[d] - out_cpu[d]).abs() < 1e-4,
                    "seq_len={} lane {}: sim {} vs cpu {}",
                    seq_len, d, out_sim[d], out_cpu[d]
                );
            }
        }
    }

    #[test]
    fn compile_attention_head_picks_matching_key() {
        // K[0] orthogonal to q, K[1] aligned with q → mass concentrates on p=1.
        let head_dim = 32;
        let seq_len = 2;
        let mut q = vec![0.0_f32; head_dim];
        q[0] = 100.0;  // big so softmax peaks
        let mut k = vec![0.0_f32; seq_len * head_dim];
        k[1 * head_dim + 0] = 1.0;  // K[1] · q is big, K[0] · q = 0
        let mut v = vec![0.0_f32; seq_len * head_dim];
        for d in 0..head_dim {
            v[0 * head_dim + d] = 1.0;
            v[1 * head_dim + d] = 9.0;
        }
        let sim = run_attention_head_on_sim(&q, &k, &v, seq_len, head_dim);
        for &val in &sim {
            assert!((val - 9.0).abs() < 1e-3, "got {}", val);
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
