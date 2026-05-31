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

/// Lower numerically-stable softmax over `n` scores into vector ops. In
/// place: reads from `x_addr`, writes back to `x_addr`. The trailing lanes
/// of the last chunk (`n_chunks * VECTOR_LANES - n` of them) must be
/// pre-padded by the caller with a large negative number (e.g. `-1e30`,
/// what `attention_mask` produces). After softmax those padding slots
/// hold ≈0 — caller can ignore them.
///
/// Single-chunk (`n ≤ VECTOR_LANES`) takes the 11-op fast path. Larger `n`
/// uses the 3-pass chunked kernel that adds VMax for the running max.
pub fn compile_softmax(n: usize, x_addr: usize) -> Vec<Instruction> {
    assert!(n > 0);
    if n <= VECTOR_LANES {
        compile_softmax_single_chunk(n, x_addr)
    } else {
        compile_softmax_chunked(n, x_addr)
    }
}

/// 11-instruction softmax for `n ≤ VECTOR_LANES`. Caller pads
/// lanes `[n..VECTOR_LANES)` with `-1e30`.
///
/// Recipe:
///   1. m = broadcast(max(x))                  (VReduceMax)
///   2. shifted = x - m                        (VSplat -1, VMul, VAdd)
///   3. e = exp(shifted)                       (VExp)
///   4. s = broadcast(sum(e))                  (VReduceSum)
///   5. 1/s via rsqrt(s*s)                     (VMul, VRsqrt)
///   6. out = e * (1/s)                        (VMul)
///
/// Register convention: v0 = x / shifted / e / out, v1 = m / inv_s,
/// v2 = neg-one constant, v3 = scratch.
fn compile_softmax_single_chunk(n: usize, x_addr: usize) -> Vec<Instruction> {
    debug_assert!(n > 0 && n <= VECTOR_LANES);
    let _ = n; // valid count is purely a caller-side concern (padding lanes)
    let mut prog = Vec::with_capacity(11);
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

/// Three-pass softmax for `n > VECTOR_LANES`. Caller pre-pads the trailing
/// lanes of the last chunk with `-1e30` so they never become the max and
/// produce ≈0 in `exp`.
///
/// Pass 1 — global max: per chunk, `VReduceMax` then `VMax` with running max.
/// Pass 2 — exp + lane-wise sum accumulator: per chunk, subtract max, exp,
///          store back, accumulate lanewise into a sum vector. One final
///          `VReduceSum` collapses lanes to a broadcast scalar.
/// Pass 3 — multiply by 1/sum: per chunk, load, multiply, store.
///
/// Register convention: v0 = chunk staging, v1 = running max / inv_sum,
/// v2 = constants / neg_max, v3 = lane sum accumulator, v4 = scratch.
/// Total instructions: 2 + 2·n_chunks  (pass 1 setup + 2 ops per chunk)
///                   + 7 + 4·n_chunks  (pass 2: setup + 4 ops per chunk)
///                   + 3·n_chunks      (pass 3)
///                   ≈ 9 + 9·n_chunks.
fn compile_softmax_chunked(n: usize, x_addr: usize) -> Vec<Instruction> {
    let n_chunks = n.div_ceil(VECTOR_LANES);
    debug_assert!(n_chunks >= 2);
    let mut prog = Vec::with_capacity(9 + 9 * n_chunks);

    // Pass 1: running max across chunks.
    prog.push(Instruction::VSplat { v: 1, scalar: -1e30 });       // running max init
    for c in 0..n_chunks {
        prog.push(Instruction::LoadVec { v: 0, sram_addr: x_addr + c * VECTOR_LANES });
        prog.push(Instruction::VReduceMax { v_in: 0, v_out: 4 });
        prog.push(Instruction::VMax { a: 1, b: 4, c: 1 });
    }

    // Pass 2: exp + lane-wise sum accumulator.
    prog.push(Instruction::VSplat { v: 2, scalar: -1.0 });
    prog.push(Instruction::VMul { a: 1, b: 2, c: 2 });            // v2 = -max
    prog.push(Instruction::VSplat { v: 3, scalar: 0.0 });         // lane-sum accumulator
    for c in 0..n_chunks {
        let off = x_addr + c * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 0, sram_addr: off });
        prog.push(Instruction::VAdd { a: 0, b: 2, c: 0 });        // x - max
        prog.push(Instruction::VExp { v_in: 0, v_out: 0 });
        prog.push(Instruction::StoreVec { v: 0, sram_addr: off }); // park exp values in place
        // accumulate into v3 — VFma with all-ones would cost the same; just VAdd.
        prog.push(Instruction::VAdd { a: 3, b: 0, c: 3 });
    }
    // Note: the per-chunk loop above is 5 instructions/chunk — one more than
    // the doc comment's 4. The capacity hint is still a fine upper bound.
    prog.push(Instruction::VReduceSum { v_in: 3, v_out: 1 });     // v1 = sum
    prog.push(Instruction::VMul { a: 1, b: 1, c: 4 });            // v4 = sum*sum
    prog.push(Instruction::VRsqrt { v_in: 4, v_out: 1 });         // v1 = 1/sum

    // Pass 3: divide each chunk by sum.
    for c in 0..n_chunks {
        let off = x_addr + c * VECTOR_LANES;
        prog.push(Instruction::LoadVec { v: 0, sram_addr: off });
        prog.push(Instruction::VMul { a: 0, b: 1, c: 0 });
        prog.push(Instruction::StoreVec { v: 0, sram_addr: off });
    }
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
/// transposed lane-parallel layout `compile_attention_head` reads.
///
/// Layout: for each head component `d`, `n_p_chunks` consecutive 32-lane
/// vectors, with position-chunk `pc` and lane `lane` mapping to:
///   `out[d * n_p_chunks * VECTOR_LANES + pc * VECTOR_LANES + lane]`
///     = `k[p * head_dim + d]`  where `p = pc * VECTOR_LANES + lane`.
/// Lanes beyond `seq_len` in the last chunk are zero — the additive mask
/// pushes their scores to `-inf` before softmax, so they contribute nothing.
///
/// `n_p_chunks = ceil(seq_len / VECTOR_LANES)`. Total size:
/// `head_dim * n_p_chunks * VECTOR_LANES` floats.
pub fn transpose_k_for_lanes(k: &[f32], seq_len: usize, head_dim: usize) -> Vec<f32> {
    assert!(seq_len > 0);
    assert_eq!(k.len(), seq_len * head_dim);
    let n_p_chunks = seq_len.div_ceil(VECTOR_LANES);
    let stride = n_p_chunks * VECTOR_LANES;
    let mut out = vec![0.0_f32; head_dim * stride];
    for p in 0..seq_len {
        let pc = p / VECTOR_LANES;
        let lane = p % VECTOR_LANES;
        for d in 0..head_dim {
            out[d * stride + pc * VECTOR_LANES + lane] = k[p * head_dim + d];
        }
    }
    out
}

/// Build the per-lane additive softmax mask laid out across position
/// chunks: 0.0 for valid positions `[0, seq_len)`, `-1e30` for the padded
/// tail of the last chunk. Length: `n_p_chunks * VECTOR_LANES`.
///
/// `compile_attention_head` adds chunk `pc` to its score chunk `pc` before
/// softmax. Most chunks see an all-zero mask; only the last chunk pushes
/// padded positions to `-inf` so they're zeroed by the subtract-max trick.
pub fn attention_mask(seq_len: usize) -> Vec<f32> {
    assert!(seq_len > 0);
    let n_p_chunks = seq_len.div_ceil(VECTOR_LANES);
    let total = n_p_chunks * VECTOR_LANES;
    let mut m = vec![0.0_f32; total];
    for lane in seq_len..total {
        m[lane] = -1e30;
    }
    m
}

/// Lower scaled dot-product attention for one query head with `seq_len`
/// cached positions. Handles arbitrary `seq_len ≥ 1` by chunking the
/// position dimension into 32-lane vectors.
///
/// SRAM contract (all addresses contiguous regions of f32):
///   - `q_broadcast_addr`: `head_dim * VECTOR_LANES` floats, from `broadcast_q`
///   - `k_t_addr`:         `head_dim * n_p_chunks * VECTOR_LANES` floats,
///                         from `transpose_k_for_lanes`
///   - `v_addr`:           `seq_len * head_dim` floats (one row per position)
///   - `mask_addr`:        `n_p_chunks * VECTOR_LANES` floats,
///                         from `attention_mask(seq_len)`
///   - `scores_addr`:      `n_p_chunks * VECTOR_LANES` scratch floats
///   - `out_addr`:         `head_dim` floats (the per-head output)
///
/// where `n_p_chunks = ceil(seq_len / VECTOR_LANES)`. `head_dim` must be a
/// multiple of `VECTOR_LANES`. `inv_sqrt_d = 1 / sqrt(head_dim)` is the
/// standard attention scale.
///
/// Program shape:
///   Phase A — scores: for each position chunk `pc`, accumulate `head_dim`
///             FMAs over (q_broadcast[d], k_t[d][pc]) into one 32-lane
///             score accumulator. Scale by inv_sqrt_d, add the masked
///             lanes, store the chunk to `scores_addr + pc * 32`.
///   Phase B — softmax: in-place over the full `n_p_chunks * 32` scores
///             via `compile_softmax` (which dispatches single vs chunked).
///   Phase C — weighted V sum: per d_chunk, for each position chunk pc
///             reload the score chunk once and inner-loop over the valid
///             lanes — `VBroadcastLane(scores_pc, lane)` × `V[p][d_chunk]`.
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
    assert!(seq_len > 0);
    let n_p_chunks = seq_len.div_ceil(VECTOR_LANES);
    let k_t_stride = n_p_chunks * VECTOR_LANES;
    let d_chunks = head_dim / VECTOR_LANES;
    let mut prog: Vec<Instruction> = Vec::new();

    // v15 = inv_sqrt_d broadcast (hoisted out of the per-chunk loop).
    prog.push(Instruction::VSplat { v: 15, scalar: inv_sqrt_d });

    // Phase A: score chunks. For each position chunk pc, lane `lane` of
    // v10 ends up holding (q · k_{pc*32 + lane}).
    for pc in 0..n_p_chunks {
        prog.push(Instruction::VSplat { v: 10, scalar: 0.0 });
        for d in 0..head_dim {
            prog.push(Instruction::LoadVec {
                v: 11,
                sram_addr: q_broadcast_addr + d * VECTOR_LANES,
            });
            prog.push(Instruction::LoadVec {
                v: 12,
                sram_addr: k_t_addr + d * k_t_stride + pc * VECTOR_LANES,
            });
            prog.push(Instruction::VFma { a: 11, b: 12, c: 10, d: 10 });
        }
        prog.push(Instruction::VMul { a: 10, b: 15, c: 10 });
        prog.push(Instruction::LoadVec { v: 11, sram_addr: mask_addr + pc * VECTOR_LANES });
        prog.push(Instruction::VAdd { a: 10, b: 11, c: 10 });
        prog.push(Instruction::StoreVec { v: 10, sram_addr: scores_addr + pc * VECTOR_LANES });
    }

    // Phase B: in-place softmax over all score chunks.
    prog.extend(compile_softmax(seq_len, scores_addr));

    // Phase C: out[dc] = Σ_p scores[p] · V[p][dc]. We re-load the score
    // chunk once per position chunk and inner-loop over the valid lanes.
    for dc in 0..d_chunks {
        prog.push(Instruction::VSplat { v: 13, scalar: 0.0 });
        for pc in 0..n_p_chunks {
            prog.push(Instruction::LoadVec {
                v: 10,
                sram_addr: scores_addr + pc * VECTOR_LANES,
            });
            let start = pc * VECTOR_LANES;
            let end = ((pc + 1) * VECTOR_LANES).min(seq_len);
            for p in start..end {
                let lane = (p - start) as u8;
                prog.push(Instruction::VBroadcastLane { v_in: 10, v_out: 11, lane });
                prog.push(Instruction::LoadVec {
                    v: 12,
                    sram_addr: v_addr + p * head_dim + dc * VECTOR_LANES,
                });
                prog.push(Instruction::VFma { a: 11, b: 12, c: 13, d: 13 });
            }
        }
        prog.push(Instruction::StoreVec { v: 13, sram_addr: out_addr + dc * VECTOR_LANES });
    }
    prog
}

/// Expand a contiguous `n`-vector into the per-lane broadcast layout that
/// attention reads: `out[d * VECTOR_LANES + lane] = src[d]` for every
/// `lane ∈ [0, VECTOR_LANES)`. Used after a freshly-computed Q (or Q for a
/// single head) lands in SRAM, before it can feed `compile_attention`.
///
/// Per 32-element chunk of source, this emits 1 LoadVec plus 32
/// `(VBroadcastLane, StoreVec)` pairs — 65 instructions per chunk.
/// `n` must be a multiple of `VECTOR_LANES`.
pub fn compile_expand_to_broadcast(n: usize, src_addr: usize, dst_addr: usize) -> Vec<Instruction> {
    assert_eq!(n % VECTOR_LANES, 0, "n must be a multiple of {}", VECTOR_LANES);
    let n_chunks = n / VECTOR_LANES;
    let mut prog = Vec::with_capacity(n_chunks * (1 + 2 * VECTOR_LANES));
    for c in 0..n_chunks {
        prog.push(Instruction::LoadVec { v: 0, sram_addr: src_addr + c * VECTOR_LANES });
        for lane in 0..VECTOR_LANES {
            prog.push(Instruction::VBroadcastLane { v_in: 0, v_out: 1, lane: lane as u8 });
            let dst_off = (c * VECTOR_LANES + lane) * VECTOR_LANES;
            prog.push(Instruction::StoreVec { v: 1, sram_addr: dst_addr + dst_off });
        }
    }
    prog
}

/// Lower multi-head scaled dot-product attention with GQA support. Emits
/// one `compile_attention_head` block per query head, mapping each Q head
/// to its KV head via `kv_h = h / (n_heads / n_kv_heads)`.
///
/// SRAM contract (concatenated blocks):
///   - `q_broadcast_addr`: `n_heads * head_dim * VECTOR_LANES`
///     (one `broadcast_q` per head — use `broadcast_q_multihead`)
///   - `k_t_cache_addr`:  `n_kv_heads * head_dim * n_p_chunks * VECTOR_LANES`
///     (one `transpose_k_for_lanes` per KV head — use `transpose_k_multihead`)
///   - `v_cache_addr`:    `n_kv_heads * seq_len * head_dim`
///     (per-KV-head split — use `split_v_per_kv_head`)
///   - `mask_addr`:       `n_p_chunks * VECTOR_LANES` (shared across heads)
///   - `scores_scratch_addr`: `n_p_chunks * VECTOR_LANES` (shared scratch)
///   - `out_addr`:        `n_heads * head_dim` (per-head outputs concatenated)
#[allow(clippy::too_many_arguments)]
pub fn compile_attention(
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    seq_len: usize,
    q_broadcast_addr: usize,
    k_t_cache_addr: usize,
    v_cache_addr: usize,
    mask_addr: usize,
    scores_scratch_addr: usize,
    out_addr: usize,
) -> Vec<Instruction> {
    assert_eq!(n_heads % n_kv_heads, 0, "n_heads must be a multiple of n_kv_heads");
    assert!(seq_len > 0);
    let q_per_kv = n_heads / n_kv_heads;
    let n_p_chunks = seq_len.div_ceil(VECTOR_LANES);
    let q_block = head_dim * VECTOR_LANES;
    let k_t_block = head_dim * n_p_chunks * VECTOR_LANES;
    let v_block = seq_len * head_dim;
    let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();

    let mut prog = Vec::new();
    for h in 0..n_heads {
        let kv_h = h / q_per_kv;
        prog.extend(compile_attention_head(
            head_dim, seq_len,
            q_broadcast_addr + h * q_block,
            k_t_cache_addr + kv_h * k_t_block,
            v_cache_addr + kv_h * v_block,
            mask_addr,
            scores_scratch_addr,
            out_addr + h * head_dim,
            inv_sqrt_d,
        ));
    }
    prog
}

/// Concatenated per-head broadcast: for each query head `h`, the head's
/// slice of `q` is expanded via `broadcast_q` into `head_dim * VECTOR_LANES`
/// floats. Output length: `n_heads * head_dim * VECTOR_LANES`.
pub fn broadcast_q_multihead(q: &[f32], n_heads: usize, head_dim: usize) -> Vec<f32> {
    assert_eq!(q.len(), n_heads * head_dim);
    let per_head = head_dim * VECTOR_LANES;
    let mut out = vec![0.0_f32; n_heads * per_head];
    for h in 0..n_heads {
        let q_h = &q[h * head_dim..(h + 1) * head_dim];
        for (d, &v) in q_h.iter().enumerate() {
            let base = h * per_head + d * VECTOR_LANES;
            for lane in 0..VECTOR_LANES {
                out[base + lane] = v;
            }
        }
    }
    out
}

/// Per-KV-head K transpose for `compile_attention`. Source `k` is the
/// standard cache layout `[seq_len, n_kv_heads * head_dim]` (one row per
/// position, same as `KvCache.k`). Output concatenates one
/// `transpose_k_for_lanes` block per KV head, giving total length
/// `n_kv_heads * head_dim * n_p_chunks * VECTOR_LANES`.
pub fn transpose_k_multihead(
    k: &[f32],
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let kv_dim = n_kv_heads * head_dim;
    assert_eq!(k.len(), seq_len * kv_dim);
    let n_p_chunks = seq_len.div_ceil(VECTOR_LANES);
    let per_head = head_dim * n_p_chunks * VECTOR_LANES;
    let stride = n_p_chunks * VECTOR_LANES;
    let mut out = vec![0.0_f32; n_kv_heads * per_head];
    for kv_h in 0..n_kv_heads {
        for p in 0..seq_len {
            let pc = p / VECTOR_LANES;
            let lane = p % VECTOR_LANES;
            let src_row = &k[p * kv_dim + kv_h * head_dim..p * kv_dim + (kv_h + 1) * head_dim];
            for (d, &v) in src_row.iter().enumerate() {
                out[kv_h * per_head + d * stride + pc * VECTOR_LANES + lane] = v;
            }
        }
    }
    out
}

/// Per-KV-head V split. Source `v` is `[seq_len, n_kv_heads * head_dim]`
/// (the standard `KvCache.v` layout). Output concatenates one
/// `[seq_len, head_dim]` block per KV head — what `compile_attention_head`
/// reads as its V cache.
pub fn split_v_per_kv_head(
    v: &[f32],
    seq_len: usize,
    n_kv_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let kv_dim = n_kv_heads * head_dim;
    assert_eq!(v.len(), seq_len * kv_dim);
    let per_head = seq_len * head_dim;
    let mut out = vec![0.0_f32; n_kv_heads * per_head];
    for kv_h in 0..n_kv_heads {
        for p in 0..seq_len {
            let src = &v[p * kv_dim + kv_h * head_dim..p * kv_dim + (kv_h + 1) * head_dim];
            let dst_off = kv_h * per_head + p * head_dim;
            out[dst_off..dst_off + head_dim].copy_from_slice(src);
        }
    }
    out
}

/// SRAM addresses for one Llama transformer block. The caller pre-allocates
/// every region (weights and scratch) and `compile_llama_block` walks each
/// sub-step in order — RMSNorm → Q proj → RoPE Q → attention → Wo → residual
/// → RMSNorm → FFN → residual.
///
/// Caching: the K_T cache and V cache are populated *before* the block runs.
/// This block does NOT scatter the current token's K, V into the cache —
/// that's a job for a future ISA op (single-lane SRAM store). For the
/// current-token integration test, the host computes K, V on the CPU side
/// and stitches them in before calling `acc.run(&program)`.
#[derive(Clone, Copy)]
pub struct LlamaBlockLayout {
    // Input + per-block norms (read-only)
    pub x_addr: usize,            // [hidden]  — also receives final block output (residual in place)
    pub attn_norm_addr: usize,    // [hidden]
    pub ffn_norm_addr: usize,     // [hidden]

    // Tiled weights (read-only) — produced by `retile_weight`
    pub wq_addr: usize,           // [hidden, hidden]
    pub wo_addr: usize,           // [hidden, hidden]
    pub w_gate_addr: usize,       // [ffn_hidden, hidden]
    pub w_up_addr: usize,         // [ffn_hidden, hidden]
    pub w_down_addr: usize,       // [hidden, ffn_hidden]

    // RoPE tables for the current position (read-only) — from `build_rope_tables`
    pub rope_cos_addr: usize,     // [head_dim]
    pub rope_sin_pm_addr: usize,  // [head_dim]

    // Pre-populated K and V caches (read-only) — from `transpose_k_multihead`
    // and `split_v_per_kv_head` over the cache including this token.
    pub k_t_cache_addr: usize,    // [n_kv_heads * head_dim * n_p_chunks * 32]
    pub v_cache_addr: usize,      // [n_kv_heads * seq_len * head_dim]
    pub mask_addr: usize,         // [n_p_chunks * 32] — from `attention_mask`

    // Scratch (written)
    pub x_normed_addr: usize,     // [hidden]
    pub q_addr: usize,            // [hidden]
    pub q_broadcast_addr: usize,  // [n_heads * head_dim * 32]
    pub attn_out_addr: usize,     // [hidden]
    pub attn_proj_addr: usize,    // [hidden]
    pub scores_scratch_addr: usize, // [n_p_chunks * 32]
    pub gate_buf_addr: usize,     // [ffn_hidden]
    pub up_buf_addr: usize,       // [ffn_hidden]
    pub ffn_out_addr: usize,      // [hidden]
}

/// Lower one Llama transformer block (pre-norm variant) onto the simulator.
/// See `LlamaBlockLayout` for the SRAM contract.
pub fn compile_llama_block(
    hidden: usize,
    ffn_hidden: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
    rms_eps: f32,
    layout: LlamaBlockLayout,
) -> Vec<Instruction> {
    assert_eq!(hidden, n_heads * head_dim);
    let mut prog = Vec::new();

    // 1. x_normed = rmsnorm(x, attn_norm)
    prog.extend(compile_rmsnorm(
        hidden, rms_eps, layout.x_addr, layout.attn_norm_addr, layout.x_normed_addr,
    ));
    // 2. q = Wq @ x_normed
    prog.extend(compile_linear(
        hidden, hidden, layout.x_normed_addr, layout.wq_addr, layout.q_addr,
    ));
    // 3. RoPE Q in place
    prog.extend(compile_rope(
        hidden, head_dim, layout.q_addr, layout.rope_cos_addr, layout.rope_sin_pm_addr,
    ));
    // 4. Expand Q to per-lane broadcast layout for attention.
    prog.extend(compile_expand_to_broadcast(
        hidden, layout.q_addr, layout.q_broadcast_addr,
    ));
    // 5. attn_out = attention(q_broadcast, K_T_cache, V_cache)
    prog.extend(compile_attention(
        n_heads, n_kv_heads, head_dim, seq_len,
        layout.q_broadcast_addr, layout.k_t_cache_addr, layout.v_cache_addr,
        layout.mask_addr, layout.scores_scratch_addr, layout.attn_out_addr,
    ));
    // 6. attn_proj = Wo @ attn_out
    prog.extend(compile_linear(
        hidden, hidden, layout.attn_out_addr, layout.wo_addr, layout.attn_proj_addr,
    ));
    // 7. x += attn_proj (residual, in place)
    prog.extend(compile_add(
        hidden, layout.x_addr, layout.attn_proj_addr, layout.x_addr,
    ));
    // 8. x_normed = rmsnorm(x, ffn_norm)  — reuse x_normed scratch
    prog.extend(compile_rmsnorm(
        hidden, rms_eps, layout.x_addr, layout.ffn_norm_addr, layout.x_normed_addr,
    ));
    // 9. ffn_out = FFN(x_normed)
    prog.extend(compile_ffn(
        hidden, ffn_hidden,
        layout.x_normed_addr, layout.w_gate_addr, layout.w_up_addr, layout.w_down_addr,
        layout.gate_buf_addr, layout.up_buf_addr, layout.ffn_out_addr,
    ));
    // 10. x += ffn_out (residual, in place)
    prog.extend(compile_add(
        hidden, layout.x_addr, layout.ffn_out_addr, layout.x_addr,
    ));
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
        // values and pads the trailing lanes of the last chunk with -1e30.
        let n = scores.len();
        let n_chunks = n.div_ceil(VECTOR_LANES);
        let sram_size = n_chunks * VECTOR_LANES;
        let mut sram = vec![-1e30_f32; sram_size];
        sram[..n].copy_from_slice(scores);
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram.copy_from_slice(&sram);
        let prog = compile_softmax(n, 0);
        acc.run(&prog).unwrap();
        acc.sram[..n].to_vec()
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

    // ===== Chapter 5.8b: multi-chunk softmax =====

    #[test]
    fn compile_softmax_chunked_matches_cpu_clean_multiples() {
        for (n, seed) in [(64_usize, 211), (128, 213), (256, 217)] {
            let mut x = pseudo(n, seed);
            // Push some lanes to extreme values to stress max-subtract across chunks.
            x[3] += 8.0;
            x[n - 5] -= 6.0;
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

    #[test]
    fn compile_softmax_chunked_handles_partial_last_chunk() {
        // n = 70 → 3 chunks, last chunk only has 6 valid lanes (70 - 64).
        let n = 70_usize;
        let mut x = pseudo(n, 219);
        x[n - 1] += 10.0; // make sure a value in the partial chunk matters
        let sim = run_softmax_on_sim(&x);
        let mut cpu = x.clone();
        crate::nn::softmax_in_place(&mut cpu);
        for i in 0..n {
            assert!(
                (sim[i] - cpu[i]).abs() < 1e-5,
                "lane {}: sim {} vs cpu {}",
                i, sim[i], cpu[i]
            );
        }
        let s: f32 = sim.iter().sum();
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn compile_softmax_chunked_survives_huge_inputs() {
        // The whole point of the subtract-max trick: huge inputs that would
        // overflow exp() in naive softmax should still produce valid output.
        let n = 96_usize;
        let mut x = vec![1.0_f32; n];
        x[50] = 1000.0;
        let sim = run_softmax_on_sim(&x);
        let s: f32 = sim.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "sum={}", s);
        assert!(sim[50] > 0.999, "got {}", sim[50]);
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
        let o_addr = s_addr + mask.len();   // scores buf is same shape as mask
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
    fn compile_attention_head_multi_chunk_seq_matches_cpu() {
        use crate::attention::{attention, KvCache};
        let head_dim = 64;
        let n_heads = 1;
        let n_kv_heads = 1;
        let kv_dim = head_dim;

        // Mix of clean multiples of 32 and partial last chunks across the range.
        for (seq_len, seed) in [(33_usize, 401), (64, 403), (96, 405), (128, 409), (200, 411)] {
            let q = pseudo(head_dim, seed);
            let k_flat = pseudo(seq_len * head_dim, seed + 1);
            let v_flat = pseudo(seq_len * head_dim, seed + 2);

            let mut cache = KvCache::new(seq_len, kv_dim);
            for p in 0..seq_len {
                let row = &k_flat[p * head_dim..(p + 1) * head_dim];
                let vrow = &v_flat[p * head_dim..(p + 1) * head_dim];
                cache.store(p, row, vrow);
            }
            let mut out_cpu = vec![0.0_f32; head_dim];
            attention(&q, &cache, seq_len - 1, n_heads, n_kv_heads, head_dim, &mut out_cpu);

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

    // ===== Chapter 5.8d: multi-head wrapper + GQA =====

    fn run_attention_multihead_on_sim(
        q: &[f32],            // n_heads * head_dim
        k_cache: &[f32],      // [seq_len, n_kv_heads * head_dim] row major
        v_cache: &[f32],      // [seq_len, n_kv_heads * head_dim] row major
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
    ) -> Vec<f32> {
        let q_bcast = broadcast_q_multihead(q, n_heads, head_dim);
        let k_t = transpose_k_multihead(k_cache, seq_len, n_kv_heads, head_dim);
        let v_split = split_v_per_kv_head(v_cache, seq_len, n_kv_heads, head_dim);
        let mask = attention_mask(seq_len);

        // SRAM layout: q_bcast | k_t | v_split | mask | scores_scratch | out
        let q_addr = 0;
        let k_addr = q_addr + q_bcast.len();
        let v_addr = k_addr + k_t.len();
        let m_addr = v_addr + v_split.len();
        let s_addr = m_addr + mask.len();
        let o_addr = s_addr + mask.len();
        let sram_size = o_addr + n_heads * head_dim;
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[q_addr..q_addr + q_bcast.len()].copy_from_slice(&q_bcast);
        acc.sram[k_addr..k_addr + k_t.len()].copy_from_slice(&k_t);
        acc.sram[v_addr..v_addr + v_split.len()].copy_from_slice(&v_split);
        acc.sram[m_addr..m_addr + mask.len()].copy_from_slice(&mask);

        let prog = compile_attention(
            n_heads, n_kv_heads, head_dim, seq_len,
            q_addr, k_addr, v_addr, m_addr, s_addr, o_addr,
        );
        acc.run(&prog).unwrap();
        acc.sram[o_addr..o_addr + n_heads * head_dim].to_vec()
    }

    #[test]
    fn compile_attention_mha_matches_cpu() {
        // Plain multi-head (no GQA): n_heads == n_kv_heads.
        use crate::attention::{attention, KvCache};
        let head_dim = 32;
        let n_heads = 2;
        let n_kv_heads = 2;
        let seq_len = 4;
        let kv_dim = n_kv_heads * head_dim;

        let q = pseudo(n_heads * head_dim, 501);
        let k_cache = pseudo(seq_len * kv_dim, 503);
        let v_cache = pseudo(seq_len * kv_dim, 505);

        let mut cache = KvCache::new(seq_len, kv_dim);
        for p in 0..seq_len {
            let kr = &k_cache[p * kv_dim..(p + 1) * kv_dim];
            let vr = &v_cache[p * kv_dim..(p + 1) * kv_dim];
            cache.store(p, kr, vr);
        }
        let mut out_cpu = vec![0.0_f32; n_heads * head_dim];
        attention(&q, &cache, seq_len - 1, n_heads, n_kv_heads, head_dim, &mut out_cpu);

        let out_sim = run_attention_multihead_on_sim(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len,
        );
        for d in 0..n_heads * head_dim {
            assert!(
                (out_sim[d] - out_cpu[d]).abs() < 1e-4,
                "lane {}: sim {} vs cpu {}",
                d, out_sim[d], out_cpu[d]
            );
        }
    }

    #[test]
    fn compile_attention_gqa_matches_cpu() {
        // GQA: 4 query heads sharing 2 KV heads (q_per_kv = 2). Multi-chunk
        // seq_len to exercise both wrappers at once.
        use crate::attention::{attention, KvCache};
        let head_dim = 64;
        let n_heads = 4;
        let n_kv_heads = 2;
        let seq_len = 40; // 2 position chunks, last is partial
        let kv_dim = n_kv_heads * head_dim;

        let q = pseudo(n_heads * head_dim, 601);
        let k_cache = pseudo(seq_len * kv_dim, 603);
        let v_cache = pseudo(seq_len * kv_dim, 607);

        let mut cache = KvCache::new(seq_len, kv_dim);
        for p in 0..seq_len {
            let kr = &k_cache[p * kv_dim..(p + 1) * kv_dim];
            let vr = &v_cache[p * kv_dim..(p + 1) * kv_dim];
            cache.store(p, kr, vr);
        }
        let mut out_cpu = vec![0.0_f32; n_heads * head_dim];
        attention(&q, &cache, seq_len - 1, n_heads, n_kv_heads, head_dim, &mut out_cpu);

        let out_sim = run_attention_multihead_on_sim(
            &q, &k_cache, &v_cache, n_heads, n_kv_heads, head_dim, seq_len,
        );
        for d in 0..n_heads * head_dim {
            assert!(
                (out_sim[d] - out_cpu[d]).abs() < 1e-4,
                "lane {}: sim {} vs cpu {}",
                d, out_sim[d], out_cpu[d]
            );
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

    // ===== Chapter 5.8e: full Llama block on the simulator =====

    #[test]
    fn compile_expand_to_broadcast_lays_out_correctly() {
        let n = 64;
        let src: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        let sram_size = n + n * VECTOR_LANES;
        let mut acc = Accelerator::new(sram_size, 0);
        acc.sram[..n].copy_from_slice(&src);
        let prog = compile_expand_to_broadcast(n, 0, n);
        acc.run(&prog).unwrap();
        for d in 0..n {
            for lane in 0..VECTOR_LANES {
                let got = acc.sram[n + d * VECTOR_LANES + lane];
                assert_eq!(got, src[d], "d={} lane={}", d, lane);
            }
        }
    }

    #[test]
    fn end_to_end_llama_block_matches_cpu() {
        use crate::attention::{attention, KvCache};
        use crate::nn::{linear, rmsnorm, rope_heads};

        // Small synthetic config that still exercises:
        //   - GQA (q_per_kv = 2)
        //   - multi-position cache (seq_len = 4, cur_pos = 3 with prior history)
        //   - all block primitives at non-trivial sizes
        let hidden = 64;
        let head_dim = 32;
        let n_heads = 2;
        let n_kv_heads = 1;
        let ffn_hidden = 64;
        let seq_len = 4;
        let cur_pos = seq_len - 1;
        let rms_eps = 1e-5_f32;
        let freq_base = 10000.0_f32;
        let kv_dim = n_kv_heads * head_dim;

        let x = pseudo(hidden, 701);
        let attn_norm = pseudo(hidden, 703);
        let ffn_norm = pseudo(hidden, 707);
        let wq = pseudo(hidden * hidden, 709);
        let wk = pseudo(kv_dim * hidden, 711);
        let wv = pseudo(kv_dim * hidden, 713);
        let wo = pseudo(hidden * hidden, 715);
        let w_gate = pseudo(ffn_hidden * hidden, 717);
        let w_up = pseudo(ffn_hidden * hidden, 719);
        let w_down = pseudo(hidden * ffn_hidden, 721);

        // Prior-position cache contents (positions 0..cur_pos). Stand in for
        // what earlier tokens' forward passes would have written. The sim and
        // CPU both consume the same random buffer, so attention math agrees.
        let prior_k = pseudo(cur_pos * kv_dim, 801);
        let prior_v = pseudo(cur_pos * kv_dim, 803);

        // ===== CPU oracle =====
        let mut x_cpu = x.clone();
        let mut x_normed_1 = vec![0.0_f32; hidden];
        rmsnorm(&x_cpu, &attn_norm, rms_eps, &mut x_normed_1);
        let mut q_cpu = vec![0.0_f32; hidden];
        let mut k_cpu = vec![0.0_f32; kv_dim];
        let mut v_cpu = vec![0.0_f32; kv_dim];
        linear(&x_normed_1, &wq, &mut q_cpu);
        linear(&x_normed_1, &wk, &mut k_cpu);
        linear(&x_normed_1, &wv, &mut v_cpu);
        rope_heads(&mut q_cpu, head_dim, cur_pos, freq_base);
        rope_heads(&mut k_cpu, head_dim, cur_pos, freq_base);

        let mut cache = KvCache::new(seq_len, kv_dim);
        for p in 0..cur_pos {
            cache.store(p,
                &prior_k[p * kv_dim..(p + 1) * kv_dim],
                &prior_v[p * kv_dim..(p + 1) * kv_dim]);
        }
        cache.store(cur_pos, &k_cpu, &v_cpu);

        let mut attn_out_cpu = vec![0.0_f32; hidden];
        attention(&q_cpu, &cache, cur_pos, n_heads, n_kv_heads, head_dim, &mut attn_out_cpu);
        let mut attn_proj_cpu = vec![0.0_f32; hidden];
        linear(&attn_out_cpu, &wo, &mut attn_proj_cpu);
        for i in 0..hidden {
            x_cpu[i] += attn_proj_cpu[i];
        }
        let mut x_normed_2 = vec![0.0_f32; hidden];
        rmsnorm(&x_cpu, &ffn_norm, rms_eps, &mut x_normed_2);
        // FFN: down @ (silu(gate @ x_normed) ⊙ (up @ x_normed))
        let mut g = vec![0.0_f32; ffn_hidden];
        let mut u = vec![0.0_f32; ffn_hidden];
        linear(&x_normed_2, &w_gate, &mut g);
        linear(&x_normed_2, &w_up, &mut u);
        for i in 0..ffn_hidden {
            let s = g[i] / (1.0 + (-g[i]).exp());
            g[i] = s * u[i];
        }
        let mut ffn_out_cpu = vec![0.0_f32; hidden];
        linear(&g, &w_down, &mut ffn_out_cpu);
        for i in 0..hidden {
            x_cpu[i] += ffn_out_cpu[i];
        }

        // ===== Simulator path =====
        // Cache layouts (post-RoPE K already in there).
        let mut k_flat = vec![0.0_f32; seq_len * kv_dim];
        let mut v_flat = vec![0.0_f32; seq_len * kv_dim];
        for p in 0..cur_pos {
            k_flat[p * kv_dim..(p + 1) * kv_dim]
                .copy_from_slice(&prior_k[p * kv_dim..(p + 1) * kv_dim]);
            v_flat[p * kv_dim..(p + 1) * kv_dim]
                .copy_from_slice(&prior_v[p * kv_dim..(p + 1) * kv_dim]);
        }
        k_flat[cur_pos * kv_dim..(cur_pos + 1) * kv_dim].copy_from_slice(&k_cpu);
        v_flat[cur_pos * kv_dim..(cur_pos + 1) * kv_dim].copy_from_slice(&v_cpu);
        let k_t = transpose_k_multihead(&k_flat, seq_len, n_kv_heads, head_dim);
        let v_split = split_v_per_kv_head(&v_flat, seq_len, n_kv_heads, head_dim);
        let mask = attention_mask(seq_len);
        let (rope_cos, rope_sin_pm) = build_rope_tables(head_dim, cur_pos, freq_base);
        let wq_tiled = retile_weight(&wq, hidden, hidden);
        let wo_tiled = retile_weight(&wo, hidden, hidden);
        let w_gate_tiled = retile_weight(&w_gate, hidden, ffn_hidden);
        let w_up_tiled = retile_weight(&w_up, hidden, ffn_hidden);
        let w_down_tiled = retile_weight(&w_down, ffn_hidden, hidden);

        // SRAM allocator.
        let mut size = 0usize;
        let mut alloc = |n: usize| -> usize { let off = size; size += n; off };
        let x_addr = alloc(hidden);
        let attn_norm_addr = alloc(hidden);
        let ffn_norm_addr = alloc(hidden);
        let wq_addr = alloc(hidden * hidden);
        let wo_addr = alloc(hidden * hidden);
        let w_gate_addr = alloc(hidden * ffn_hidden);
        let w_up_addr = alloc(hidden * ffn_hidden);
        let w_down_addr = alloc(hidden * ffn_hidden);
        let rope_cos_addr = alloc(head_dim);
        let rope_sin_pm_addr = alloc(head_dim);
        let k_t_cache_addr = alloc(k_t.len());
        let v_cache_addr = alloc(v_split.len());
        let mask_addr = alloc(mask.len());
        let x_normed_addr = alloc(hidden);
        let q_addr = alloc(hidden);
        let q_broadcast_addr = alloc(n_heads * head_dim * VECTOR_LANES);
        let attn_out_addr = alloc(hidden);
        let attn_proj_addr = alloc(hidden);
        let scores_scratch_addr = alloc(mask.len());
        let gate_buf_addr = alloc(ffn_hidden);
        let up_buf_addr = alloc(ffn_hidden);
        let ffn_out_addr = alloc(hidden);

        let mut acc = Accelerator::new(size, 0);
        acc.sram[x_addr..x_addr + hidden].copy_from_slice(&x);
        acc.sram[attn_norm_addr..attn_norm_addr + hidden].copy_from_slice(&attn_norm);
        acc.sram[ffn_norm_addr..ffn_norm_addr + hidden].copy_from_slice(&ffn_norm);
        acc.sram[wq_addr..wq_addr + hidden * hidden].copy_from_slice(&wq_tiled);
        acc.sram[wo_addr..wo_addr + hidden * hidden].copy_from_slice(&wo_tiled);
        acc.sram[w_gate_addr..w_gate_addr + hidden * ffn_hidden].copy_from_slice(&w_gate_tiled);
        acc.sram[w_up_addr..w_up_addr + hidden * ffn_hidden].copy_from_slice(&w_up_tiled);
        acc.sram[w_down_addr..w_down_addr + hidden * ffn_hidden].copy_from_slice(&w_down_tiled);
        acc.sram[rope_cos_addr..rope_cos_addr + head_dim].copy_from_slice(&rope_cos);
        acc.sram[rope_sin_pm_addr..rope_sin_pm_addr + head_dim].copy_from_slice(&rope_sin_pm);
        acc.sram[k_t_cache_addr..k_t_cache_addr + k_t.len()].copy_from_slice(&k_t);
        acc.sram[v_cache_addr..v_cache_addr + v_split.len()].copy_from_slice(&v_split);
        acc.sram[mask_addr..mask_addr + mask.len()].copy_from_slice(&mask);

        let layout = LlamaBlockLayout {
            x_addr, attn_norm_addr, ffn_norm_addr,
            wq_addr, wo_addr, w_gate_addr, w_up_addr, w_down_addr,
            rope_cos_addr, rope_sin_pm_addr,
            k_t_cache_addr, v_cache_addr, mask_addr,
            x_normed_addr, q_addr, q_broadcast_addr,
            attn_out_addr, attn_proj_addr, scores_scratch_addr,
            gate_buf_addr, up_buf_addr, ffn_out_addr,
        };
        let prog = compile_llama_block(
            hidden, ffn_hidden, head_dim, n_heads, n_kv_heads,
            seq_len, rms_eps, layout,
        );
        acc.run(&prog).unwrap();

        for i in 0..hidden {
            let got = acc.sram[x_addr + i];
            let want = x_cpu[i];
            assert!(
                (got - want).abs() < 1e-3,
                "x[{}]: sim {} vs cpu {} (diff {})", i, got, want, (got - want).abs()
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
