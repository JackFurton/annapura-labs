//! Toy ML accelerator — vector machine with on-chip SRAM and matmul tile.
//!
//! Goal: the simplest hardware design that could plausibly run a transformer
//! layer end-to-end. Chapter 5.0 added vector ops; 5.1 adds the matmul tile
//! and DRAM transfers; 5.2 will add a compiler that lowers Llama → these
//! instructions; chapter 6 will layer cycle-accurate timing on top.
//!
//! Architectural sketch:
//!   - 32 vector registers, 32 lanes of f32 each (4 KB of vreg state)
//!   - 1 matrix accumulator register (16×16 f32 = 1 KB) — like NVIDIA Tensor
//!     Cores' fragment, this is where matmuls accumulate before write-back
//!   - On-chip SRAM scratchpad (sized at creation, target ~50 MB to hold one
//!     TinyLlama layer's worth of Q8_0 weights)
//!   - DRAM modeled as a separate buffer; explicit LoadDram/StoreDram moves
//!     data across the SRAM/DRAM boundary (this is THE expensive operation
//!     on a real chip — the roofline ridge moves left because of this gap)
//!
//! This is the *functional* simulator: correct math, no cycle accuracy.
//! Chapter 6 will replace `instr_count` with realistic per-instruction timing
//! based on a pipeline model.
//!
//! Connection to the roofline numbers from chapter 2.8:
//!   - Each MatMulTile is 16×16×16 = 4096 MACs = 8192 FLOPS per instruction.
//!     At a hypothetical 1 GHz issue rate that's 8 TFLOPS *per matmul pipe*.
//!     Real Trainium-class chips stack ~16 such tiles → 100+ TFLOPS.
//!   - SRAM-resident weights mean each MatMulTile reads from cheap on-chip
//!     storage. DRAM↔SRAM transfers are where the roofline ridge lives.

use anyhow::{anyhow, bail, Result};

pub const VECTOR_LANES: usize = 32;
pub const N_VECTOR_REGS: usize = 32;
pub const MATMUL_TILE: usize = 16;
pub const MATMUL_TILE_ELEMENTS: usize = MATMUL_TILE * MATMUL_TILE;

pub struct Accelerator {
    pub sram: Vec<f32>,
    pub dram: Vec<f32>,
    pub vregs: Box<[[f32; VECTOR_LANES]; N_VECTOR_REGS]>,
    pub matrix_accum: [[f32; MATMUL_TILE]; MATMUL_TILE],
    /// Counts dispatched instructions. The cycle-accurate model in chapter 6
    /// will replace this with realistic pipeline timing per opcode.
    pub instr_count: u64,
}

impl Accelerator {
    pub fn new(sram_elements: usize, dram_elements: usize) -> Self {
        Self {
            sram: vec![0.0; sram_elements],
            dram: vec![0.0; dram_elements],
            vregs: Box::new([[0.0; VECTOR_LANES]; N_VECTOR_REGS]),
            matrix_accum: [[0.0; MATMUL_TILE]; MATMUL_TILE],
            instr_count: 0,
        }
    }

    pub fn sram_size_bytes(&self) -> usize {
        self.sram.len() * 4
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    // === Vector unit (chapter 5.0) ===
    /// `vregs[v] = SRAM[addr .. addr + VECTOR_LANES]`
    LoadVec { v: u8, sram_addr: usize },
    /// `SRAM[addr .. addr + VECTOR_LANES] = vregs[v]`
    StoreVec { v: u8, sram_addr: usize },
    /// `vregs[c] = vregs[a] + vregs[b]` (lanewise)
    VAdd { a: u8, b: u8, c: u8 },
    /// `vregs[c] = vregs[a] * vregs[b]` (lanewise)
    VMul { a: u8, b: u8, c: u8 },
    /// `vregs[d] = vregs[a] * vregs[b] + vregs[c]` (lanewise FMA)
    VFma { a: u8, b: u8, c: u8, d: u8 },
    /// `vregs[v] = [scalar; VECTOR_LANES]`
    VSplat { v: u8, scalar: f32 },
    /// `vregs[v_out] = silu(vregs[v_in])` (lanewise)
    VSilu { v_in: u8, v_out: u8 },
    /// `vregs[v_out] = [sum(vregs[v_in]); VECTOR_LANES]` — sum-of-lanes
    /// broadcast. The horizontal reduction every real vector ISA needs for
    /// RMSNorm / softmax. In silicon: a log-tree reduction network.
    VReduceSum { v_in: u8, v_out: u8 },
    /// `vregs[v_out] = 1 / sqrt(vregs[v_in])` (lanewise). Transcendental like
    /// VSilu — modeled as a multi-cycle op on the vector pipe.
    VRsqrt { v_in: u8, v_out: u8 },
    /// Swap adjacent lane pairs: lanes (2k, 2k+1) ↔ (2k+1, 2k). Cheap shuffle
    /// in silicon (NEON `vrev64`, AVX `vpshufd`). Needed for interleaved-pair
    /// RoPE, where the cross-term in the 2D rotation pulls the swapped vector.
    VSwapPairs { v_in: u8, v_out: u8 },
    /// `vregs[v_out] = exp(vregs[v_in])` (lanewise). Transcendental — same
    /// design point as VSilu / VRsqrt. The softmax kernel needs it.
    VExp { v_in: u8, v_out: u8 },
    /// `vregs[v_out] = [max(vregs[v_in]); VECTOR_LANES]` — broadcast the
    /// max across lanes. Companion of VReduceSum; required for the
    /// numerically-stable softmax (subtract-max trick).
    VReduceMax { v_in: u8, v_out: u8 },
    /// `vregs[c] = max(vregs[a], vregs[b])` (lanewise). Used by the
    /// multi-chunk softmax kernel to merge per-chunk max-broadcasts into a
    /// running global max across chunks.
    VMax { a: u8, b: u8, c: u8 },
    /// `vregs[v_out] = [vregs[v_in][lane]; VECTOR_LANES]` — broadcast one
    /// lane to all lanes. Standard shuffle (NEON `vdup_lane_f32`, AVX
    /// `_mm256_set1_ps`). Lets the attention kernel feed per-position
    /// softmax weights into the weighted-V accumulation.
    VBroadcastLane { v_in: u8, v_out: u8, lane: u8 },

    // === Matrix unit (chapter 5.1) ===
    /// Zero the matrix accumulator.
    MatAccumClear,
    /// `matrix_accum += A · B` where A and B are 16×16 tiles in SRAM
    /// (row-major, contiguous). A starts at `a_sram`, B at `b_sram`.
    MatMulTile { a_sram: usize, b_sram: usize },
    /// `SRAM[dst .. dst + 256] = matrix_accum` (row-major).
    MatAccumStore { sram_dst: usize },
    /// Matrix-vector tile (the matvec primitive — what `compile_linear` emits).
    /// Reads 16 elements of x at `x_sram`, a tiled 16×16 weight block at
    /// `w_sram` (laid out so `w_sram[k*16 + j]` = original `W[output_tile + j][input_tile + k]`),
    /// and accumulates the 16-element matvec result into `y_sram[0..16]`.
    /// `accumulate = false` overwrites; `accumulate = true` adds.
    /// This is the primitive used to lower a Llama linear; 16,384 of these
    /// dispatched in sequence cover a 2048×2048 Q-projection.
    MatVecTile {
        x_sram: usize,
        w_sram: usize,
        y_sram: usize,
        accumulate: bool,
    },

    // === Host transfers (chapter 5.1) ===
    /// `SRAM[sram_addr .. sram_addr + len] = DRAM[dram_addr .. dram_addr + len]`
    LoadDram { dram_addr: usize, sram_addr: usize, len: usize },
    /// `DRAM[dram_addr .. dram_addr + len] = SRAM[sram_addr .. sram_addr + len]`
    StoreDram { sram_addr: usize, dram_addr: usize, len: usize },
}

impl Accelerator {
    pub fn execute(&mut self, instr: &Instruction) -> Result<()> {
        use Instruction::*;
        match *instr {
            // --- Vector ops ---
            LoadVec { v, sram_addr } => {
                let v = idx(v, N_VECTOR_REGS, "vreg")?;
                let end = checked_end(sram_addr, VECTOR_LANES, self.sram.len(), "SRAM (LoadVec)")?;
                self.vregs[v].copy_from_slice(&self.sram[sram_addr..end]);
            }
            StoreVec { v, sram_addr } => {
                let v = idx(v, N_VECTOR_REGS, "vreg")?;
                let end = checked_end(sram_addr, VECTOR_LANES, self.sram.len(), "SRAM (StoreVec)")?;
                self.sram[sram_addr..end].copy_from_slice(&self.vregs[v]);
            }
            VAdd { a, b, c } => {
                let (a, b, c) = (
                    idx(a, N_VECTOR_REGS, "vreg")?,
                    idx(b, N_VECTOR_REGS, "vreg")?,
                    idx(c, N_VECTOR_REGS, "vreg")?,
                );
                let va = self.vregs[a];
                let vb = self.vregs[b];
                for i in 0..VECTOR_LANES {
                    self.vregs[c][i] = va[i] + vb[i];
                }
            }
            VMul { a, b, c } => {
                let (a, b, c) = (
                    idx(a, N_VECTOR_REGS, "vreg")?,
                    idx(b, N_VECTOR_REGS, "vreg")?,
                    idx(c, N_VECTOR_REGS, "vreg")?,
                );
                let va = self.vregs[a];
                let vb = self.vregs[b];
                for i in 0..VECTOR_LANES {
                    self.vregs[c][i] = va[i] * vb[i];
                }
            }
            VFma { a, b, c, d } => {
                let (a, b, c, d) = (
                    idx(a, N_VECTOR_REGS, "vreg")?,
                    idx(b, N_VECTOR_REGS, "vreg")?,
                    idx(c, N_VECTOR_REGS, "vreg")?,
                    idx(d, N_VECTOR_REGS, "vreg")?,
                );
                let va = self.vregs[a];
                let vb = self.vregs[b];
                let vc = self.vregs[c];
                for i in 0..VECTOR_LANES {
                    self.vregs[d][i] = va[i] * vb[i] + vc[i];
                }
            }
            VSplat { v, scalar } => {
                let v = idx(v, N_VECTOR_REGS, "vreg")?;
                self.vregs[v] = [scalar; VECTOR_LANES];
            }
            VSilu { v_in, v_out } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let src = self.vregs[i];
                for k in 0..VECTOR_LANES {
                    let x = src[k];
                    self.vregs[o][k] = x / (1.0 + (-x).exp());
                }
            }
            VReduceSum { v_in, v_out } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let sum: f32 = self.vregs[i].iter().sum();
                self.vregs[o] = [sum; VECTOR_LANES];
            }
            VRsqrt { v_in, v_out } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let src = self.vregs[i];
                for k in 0..VECTOR_LANES {
                    self.vregs[o][k] = 1.0 / src[k].sqrt();
                }
            }
            VSwapPairs { v_in, v_out } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let src = self.vregs[i];
                for k in 0..VECTOR_LANES / 2 {
                    self.vregs[o][2 * k] = src[2 * k + 1];
                    self.vregs[o][2 * k + 1] = src[2 * k];
                }
            }
            VExp { v_in, v_out } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let src = self.vregs[i];
                for k in 0..VECTOR_LANES {
                    self.vregs[o][k] = src[k].exp();
                }
            }
            VReduceMax { v_in, v_out } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let m = self.vregs[i].iter().copied().fold(f32::NEG_INFINITY, f32::max);
                self.vregs[o] = [m; VECTOR_LANES];
            }
            VMax { a, b, c } => {
                let (a, b, c) = (
                    idx(a, N_VECTOR_REGS, "vreg")?,
                    idx(b, N_VECTOR_REGS, "vreg")?,
                    idx(c, N_VECTOR_REGS, "vreg")?,
                );
                let va = self.vregs[a];
                let vb = self.vregs[b];
                for i in 0..VECTOR_LANES {
                    self.vregs[c][i] = va[i].max(vb[i]);
                }
            }
            VBroadcastLane { v_in, v_out, lane } => {
                let (i, o) = (idx(v_in, N_VECTOR_REGS, "vreg")?, idx(v_out, N_VECTOR_REGS, "vreg")?);
                let l = lane as usize;
                if l >= VECTOR_LANES {
                    bail!("VBroadcastLane lane {} out of range (have {})", l, VECTOR_LANES);
                }
                let scalar = self.vregs[i][l];
                self.vregs[o] = [scalar; VECTOR_LANES];
            }

            // --- Matrix unit ---
            MatAccumClear => {
                self.matrix_accum = [[0.0; MATMUL_TILE]; MATMUL_TILE];
            }
            MatMulTile { a_sram, b_sram } => {
                let a_end =
                    checked_end(a_sram, MATMUL_TILE_ELEMENTS, self.sram.len(), "SRAM (MatMul A)")?;
                let b_end =
                    checked_end(b_sram, MATMUL_TILE_ELEMENTS, self.sram.len(), "SRAM (MatMul B)")?;
                let _ = (a_end, b_end);
                // accum[i][j] += sum_k A[i][k] * B[k][j], row-major 16x16 tiles.
                for i in 0..MATMUL_TILE {
                    for j in 0..MATMUL_TILE {
                        let mut acc = self.matrix_accum[i][j];
                        for k in 0..MATMUL_TILE {
                            acc += self.sram[a_sram + i * MATMUL_TILE + k]
                                * self.sram[b_sram + k * MATMUL_TILE + j];
                        }
                        self.matrix_accum[i][j] = acc;
                    }
                }
            }
            MatAccumStore { sram_dst } => {
                let _ = checked_end(sram_dst, MATMUL_TILE_ELEMENTS, self.sram.len(), "SRAM (AccumStore)")?;
                for i in 0..MATMUL_TILE {
                    for j in 0..MATMUL_TILE {
                        self.sram[sram_dst + i * MATMUL_TILE + j] = self.matrix_accum[i][j];
                    }
                }
            }
            MatVecTile { x_sram, w_sram, y_sram, accumulate } => {
                let _ = checked_end(x_sram, MATMUL_TILE, self.sram.len(), "SRAM (MatVec X)")?;
                let _ = checked_end(w_sram, MATMUL_TILE_ELEMENTS, self.sram.len(), "SRAM (MatVec W)")?;
                let _ = checked_end(y_sram, MATMUL_TILE, self.sram.len(), "SRAM (MatVec Y)")?;

                let mut x_buf = [0.0_f32; MATMUL_TILE];
                x_buf.copy_from_slice(&self.sram[x_sram..x_sram + MATMUL_TILE]);

                for j in 0..MATMUL_TILE {
                    let mut acc = if accumulate { self.sram[y_sram + j] } else { 0.0 };
                    for k in 0..MATMUL_TILE {
                        acc += x_buf[k] * self.sram[w_sram + k * MATMUL_TILE + j];
                    }
                    self.sram[y_sram + j] = acc;
                }
            }

            // --- DRAM transfers ---
            LoadDram { dram_addr, sram_addr, len } => {
                let _ = checked_end(dram_addr, len, self.dram.len(), "DRAM (Load)")?;
                let _ = checked_end(sram_addr, len, self.sram.len(), "SRAM (Load target)")?;
                self.sram[sram_addr..sram_addr + len]
                    .copy_from_slice(&self.dram[dram_addr..dram_addr + len]);
            }
            StoreDram { sram_addr, dram_addr, len } => {
                let _ = checked_end(sram_addr, len, self.sram.len(), "SRAM (Store source)")?;
                let _ = checked_end(dram_addr, len, self.dram.len(), "DRAM (Store)")?;
                self.dram[dram_addr..dram_addr + len]
                    .copy_from_slice(&self.sram[sram_addr..sram_addr + len]);
            }
        }
        self.instr_count += 1;
        Ok(())
    }

    pub fn run(&mut self, program: &[Instruction]) -> Result<()> {
        for instr in program {
            self.execute(instr)?;
        }
        Ok(())
    }
}

fn idx(raw: u8, max: usize, kind: &str) -> Result<usize> {
    let i = raw as usize;
    if i >= max {
        bail!("{} register {} out of range (have {})", kind, i, max);
    }
    Ok(i)
}

fn checked_end(start: usize, len: usize, capacity: usize, where_: &str) -> Result<usize> {
    let end = start.checked_add(len).ok_or_else(|| anyhow!("address overflow in {}", where_))?;
    if end > capacity {
        bail!("{} access [{}..{}] exceeds capacity {}", where_, start, end, capacity);
    }
    Ok(end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_acc(sram: usize) -> Accelerator {
        Accelerator::new(sram, 1024)
    }

    #[test]
    fn vsplat_then_vadd() {
        let mut acc = new_acc(128);
        acc.execute(&Instruction::VSplat { v: 0, scalar: 1.0 }).unwrap();
        acc.execute(&Instruction::VSplat { v: 1, scalar: 2.0 }).unwrap();
        acc.execute(&Instruction::VAdd { a: 0, b: 1, c: 2 }).unwrap();
        assert_eq!(acc.vregs[2], [3.0; VECTOR_LANES]);
    }

    #[test]
    fn vfma_lanewise() {
        let mut acc = new_acc(128);
        acc.execute(&Instruction::VSplat { v: 0, scalar: 0.0 }).unwrap();
        acc.vregs[0][0] = 1.0;
        acc.vregs[0][1] = 2.0;
        acc.vregs[0][2] = 3.0;
        acc.execute(&Instruction::VSplat { v: 1, scalar: 0.0 }).unwrap();
        acc.vregs[1][0] = 4.0;
        acc.vregs[1][1] = 5.0;
        acc.vregs[1][2] = 6.0;
        acc.execute(&Instruction::VSplat { v: 2, scalar: 0.5 }).unwrap();
        acc.execute(&Instruction::VFma { a: 0, b: 1, c: 2, d: 3 }).unwrap();
        assert_eq!(acc.vregs[3][0], 4.5);
        assert_eq!(acc.vregs[3][1], 10.5);
        assert_eq!(acc.vregs[3][2], 18.5);
    }

    #[test]
    fn loadvec_storevec_roundtrip() {
        let mut acc = new_acc(128);
        for i in 0..VECTOR_LANES {
            acc.sram[i] = i as f32;
        }
        acc.execute(&Instruction::LoadVec { v: 5, sram_addr: 0 }).unwrap();
        for i in 0..VECTOR_LANES {
            assert_eq!(acc.vregs[5][i], i as f32);
        }
        acc.execute(&Instruction::StoreVec { v: 5, sram_addr: 64 }).unwrap();
        for i in 0..VECTOR_LANES {
            assert_eq!(acc.sram[64 + i], i as f32);
        }
    }

    #[test]
    fn vsilu_matches_cpu_silu() {
        let mut acc = new_acc(64);
        for (lane, &v) in [-2.0_f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
            acc.vregs[0][lane] = v;
        }
        acc.execute(&Instruction::VSilu { v_in: 0, v_out: 1 }).unwrap();
        for (lane, &v) in [-2.0_f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
            let expected = v / (1.0 + (-v).exp());
            assert!((acc.vregs[1][lane] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn vreduce_sum_broadcasts_total_to_all_lanes() {
        let mut acc = new_acc(64);
        for k in 0..VECTOR_LANES {
            acc.vregs[0][k] = k as f32; // 0+1+...+31 = 496
        }
        acc.execute(&Instruction::VReduceSum { v_in: 0, v_out: 1 }).unwrap();
        for k in 0..VECTOR_LANES {
            assert_eq!(acc.vregs[1][k], 496.0);
        }
    }

    #[test]
    fn vswap_pairs_swaps_adjacent_lanes() {
        let mut acc = new_acc(64);
        for k in 0..VECTOR_LANES {
            acc.vregs[0][k] = k as f32;
        }
        acc.execute(&Instruction::VSwapPairs { v_in: 0, v_out: 1 }).unwrap();
        for k in 0..VECTOR_LANES / 2 {
            assert_eq!(acc.vregs[1][2 * k], (2 * k + 1) as f32);
            assert_eq!(acc.vregs[1][2 * k + 1], (2 * k) as f32);
        }
    }

    #[test]
    fn vexp_lanewise_matches_libm() {
        let mut acc = new_acc(64);
        for (lane, &v) in [-1.0_f32, 0.0, 1.0, 2.0].iter().enumerate() {
            acc.vregs[0][lane] = v;
        }
        acc.execute(&Instruction::VExp { v_in: 0, v_out: 1 }).unwrap();
        for (lane, &v) in [-1.0_f32, 0.0, 1.0, 2.0].iter().enumerate() {
            let expected = v.exp();
            assert!((acc.vregs[1][lane] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn vreduce_max_broadcasts_max_to_all_lanes() {
        let mut acc = new_acc(64);
        for k in 0..VECTOR_LANES {
            acc.vregs[0][k] = k as f32; // max = 31
        }
        acc.vregs[0][7] = 100.0;
        acc.execute(&Instruction::VReduceMax { v_in: 0, v_out: 1 }).unwrap();
        for k in 0..VECTOR_LANES {
            assert_eq!(acc.vregs[1][k], 100.0);
        }
    }

    #[test]
    fn vbroadcast_lane_copies_one_lane_everywhere() {
        let mut acc = new_acc(64);
        for k in 0..VECTOR_LANES {
            acc.vregs[0][k] = (k as f32) * 0.5;
        }
        acc.execute(&Instruction::VBroadcastLane { v_in: 0, v_out: 1, lane: 5 }).unwrap();
        for k in 0..VECTOR_LANES {
            assert_eq!(acc.vregs[1][k], 2.5);
        }
    }

    #[test]
    fn vmax_lanewise_is_pointwise_max() {
        let mut acc = new_acc(64);
        for k in 0..VECTOR_LANES {
            acc.vregs[0][k] = k as f32;
            acc.vregs[1][k] = (VECTOR_LANES as f32) - (k as f32);
        }
        acc.execute(&Instruction::VMax { a: 0, b: 1, c: 2 }).unwrap();
        for k in 0..VECTOR_LANES {
            let expected = (k as f32).max((VECTOR_LANES as f32) - (k as f32));
            assert_eq!(acc.vregs[2][k], expected);
        }
    }

    #[test]
    fn vbroadcast_lane_out_of_range_errors() {
        let mut acc = new_acc(64);
        assert!(acc
            .execute(&Instruction::VBroadcastLane { v_in: 0, v_out: 1, lane: 32 })
            .is_err());
    }

    #[test]
    fn vrsqrt_lanewise() {
        let mut acc = new_acc(64);
        for (lane, &v) in [1.0_f32, 4.0, 16.0, 100.0].iter().enumerate() {
            acc.vregs[0][lane] = v;
        }
        acc.execute(&Instruction::VRsqrt { v_in: 0, v_out: 1 }).unwrap();
        for (lane, &v) in [1.0_f32, 4.0, 16.0, 100.0].iter().enumerate() {
            let expected = 1.0 / v.sqrt();
            assert!((acc.vregs[1][lane] - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn instr_count_increments_on_success() {
        let mut acc = new_acc(64);
        acc.run(&[
            Instruction::VSplat { v: 0, scalar: 1.0 },
            Instruction::VSplat { v: 1, scalar: 2.0 },
            Instruction::VAdd { a: 0, b: 1, c: 2 },
        ]).unwrap();
        assert_eq!(acc.instr_count, 3);
    }

    #[test]
    fn out_of_range_vreg_errors() {
        let mut acc = new_acc(64);
        assert!(acc.execute(&Instruction::VSplat { v: 200, scalar: 0.0 }).is_err());
        assert_eq!(acc.instr_count, 0);
    }

    #[test]
    fn out_of_bounds_sram_errors() {
        let mut acc = new_acc(VECTOR_LANES - 1);
        assert!(acc.execute(&Instruction::LoadVec { v: 0, sram_addr: 0 }).is_err());
    }

    // ===== Matrix unit + DRAM tests (chapter 5.1) =====

    #[test]
    fn mat_accum_clear_zeros_register() {
        let mut acc = new_acc(MATMUL_TILE_ELEMENTS * 2);
        acc.matrix_accum[3][7] = 42.0;
        acc.execute(&Instruction::MatAccumClear).unwrap();
        for row in &acc.matrix_accum {
            assert!(row.iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn matmul_tile_identity_times_identity_is_identity() {
        let mut acc = new_acc(MATMUL_TILE_ELEMENTS * 3);
        // Lay out two identity matrices: A at offset 0, B at offset 256.
        for i in 0..MATMUL_TILE {
            acc.sram[i * MATMUL_TILE + i] = 1.0;
            acc.sram[MATMUL_TILE_ELEMENTS + i * MATMUL_TILE + i] = 1.0;
        }
        acc.execute(&Instruction::MatAccumClear).unwrap();
        acc.execute(&Instruction::MatMulTile { a_sram: 0, b_sram: MATMUL_TILE_ELEMENTS }).unwrap();
        for i in 0..MATMUL_TILE {
            for j in 0..MATMUL_TILE {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_eq!(acc.matrix_accum[i][j], expected);
            }
        }
    }

    #[test]
    fn matmul_tile_accumulates_across_calls() {
        // Run MatMulTile twice with the same identity inputs. Accumulator
        // should double.
        let mut acc = new_acc(MATMUL_TILE_ELEMENTS * 3);
        for i in 0..MATMUL_TILE {
            acc.sram[i * MATMUL_TILE + i] = 1.0;
            acc.sram[MATMUL_TILE_ELEMENTS + i * MATMUL_TILE + i] = 1.0;
        }
        acc.execute(&Instruction::MatAccumClear).unwrap();
        acc.execute(&Instruction::MatMulTile { a_sram: 0, b_sram: MATMUL_TILE_ELEMENTS }).unwrap();
        acc.execute(&Instruction::MatMulTile { a_sram: 0, b_sram: MATMUL_TILE_ELEMENTS }).unwrap();
        for i in 0..MATMUL_TILE {
            assert_eq!(acc.matrix_accum[i][i], 2.0);
        }
    }

    #[test]
    fn matmul_tile_known_answer_then_store() {
        // A = ones, B = ones. A·B[i][j] = sum_k 1*1 = 16. Whole accumulator
        // is 16.0 after one MatMulTile, then MatAccumStore copies it to SRAM.
        let mut acc = new_acc(MATMUL_TILE_ELEMENTS * 4);
        for i in 0..MATMUL_TILE_ELEMENTS {
            acc.sram[i] = 1.0;
            acc.sram[MATMUL_TILE_ELEMENTS + i] = 1.0;
        }
        acc.execute(&Instruction::MatAccumClear).unwrap();
        acc.execute(&Instruction::MatMulTile { a_sram: 0, b_sram: MATMUL_TILE_ELEMENTS }).unwrap();
        let dst = MATMUL_TILE_ELEMENTS * 2;
        acc.execute(&Instruction::MatAccumStore { sram_dst: dst }).unwrap();
        for i in 0..MATMUL_TILE_ELEMENTS {
            assert_eq!(acc.sram[dst + i], 16.0);
        }
    }

    #[test]
    fn matmul_tile_matches_cpu_reference() {
        // Random-ish 16×16 inputs, cross-check vs the naive CPU matmul kernel.
        let mut acc = new_acc(MATMUL_TILE_ELEMENTS * 3);
        let a: Vec<f32> = (0..MATMUL_TILE_ELEMENTS)
            .map(|i| ((i * 31 + 7) % 41) as f32 / 41.0 - 0.5)
            .collect();
        let b: Vec<f32> = (0..MATMUL_TILE_ELEMENTS)
            .map(|i| ((i * 37 + 11) % 43) as f32 / 43.0 - 0.5)
            .collect();
        acc.sram[..MATMUL_TILE_ELEMENTS].copy_from_slice(&a);
        acc.sram[MATMUL_TILE_ELEMENTS..MATMUL_TILE_ELEMENTS * 2].copy_from_slice(&b);

        acc.execute(&Instruction::MatAccumClear).unwrap();
        acc.execute(&Instruction::MatMulTile { a_sram: 0, b_sram: MATMUL_TILE_ELEMENTS }).unwrap();

        let mut reference = vec![0.0_f32; MATMUL_TILE_ELEMENTS];
        crate::matmul::matmul_naive(&a, &b, &mut reference, MATMUL_TILE, MATMUL_TILE, MATMUL_TILE);

        for i in 0..MATMUL_TILE {
            for j in 0..MATMUL_TILE {
                let got = acc.matrix_accum[i][j];
                let want = reference[i * MATMUL_TILE + j];
                assert!((got - want).abs() < 1e-4, "[{},{}]: {} vs {}", i, j, got, want);
            }
        }
    }

    #[test]
    fn dram_sram_roundtrip() {
        let mut acc = Accelerator::new(64, 64);
        for i in 0..64 {
            acc.dram[i] = i as f32;
        }
        acc.execute(&Instruction::LoadDram { dram_addr: 8, sram_addr: 16, len: 32 }).unwrap();
        for i in 0..32 {
            assert_eq!(acc.sram[16 + i], (8 + i) as f32);
        }
        // Modify a few SRAM cells, push back to a different DRAM region.
        for i in 0..32 {
            acc.sram[16 + i] *= -1.0;
        }
        acc.execute(&Instruction::StoreDram { sram_addr: 16, dram_addr: 0, len: 32 }).unwrap();
        for i in 0..32 {
            assert_eq!(acc.dram[i], -((8 + i) as f32));
        }
    }

    #[test]
    fn dram_overrun_errors() {
        let mut acc = Accelerator::new(64, 64);
        assert!(acc
            .execute(&Instruction::LoadDram { dram_addr: 50, sram_addr: 0, len: 32 })
            .is_err());
    }

    // ===== MatVecTile tests =====

    #[test]
    fn matvec_tile_identity_passes_x_through() {
        let mut acc = new_acc(MATMUL_TILE_ELEMENTS * 2 + MATMUL_TILE * 4);
        // x = [0.5, 1.0, 1.5, ...] at sram[0..16]
        for i in 0..MATMUL_TILE {
            acc.sram[i] = (i as f32 + 1.0) * 0.5;
        }
        // W tile = identity at sram[16..16+256]
        // For matvec semantics, tile[k][j] = 1 if k==j else 0 reproduces y = x.
        for k in 0..MATMUL_TILE {
            acc.sram[MATMUL_TILE + k * MATMUL_TILE + k] = 1.0;
        }
        let y_addr = MATMUL_TILE + MATMUL_TILE_ELEMENTS;
        acc.execute(&Instruction::MatVecTile {
            x_sram: 0,
            w_sram: MATMUL_TILE,
            y_sram: y_addr,
            accumulate: false,
        })
        .unwrap();
        for i in 0..MATMUL_TILE {
            assert_eq!(acc.sram[y_addr + i], (i as f32 + 1.0) * 0.5);
        }
    }

    #[test]
    fn matvec_tile_accumulates_on_second_call() {
        let mut acc = new_acc(MATMUL_TILE * 4 + MATMUL_TILE_ELEMENTS * 2);
        for i in 0..MATMUL_TILE {
            acc.sram[i] = 1.0; // x = ones
            acc.sram[MATMUL_TILE + i * MATMUL_TILE + i] = 1.0; // tile = identity
        }
        let y_addr = MATMUL_TILE + MATMUL_TILE_ELEMENTS;
        acc.execute(&Instruction::MatVecTile {
            x_sram: 0, w_sram: MATMUL_TILE, y_sram: y_addr, accumulate: false,
        }).unwrap();
        acc.execute(&Instruction::MatVecTile {
            x_sram: 0, w_sram: MATMUL_TILE, y_sram: y_addr, accumulate: true,
        }).unwrap();
        for i in 0..MATMUL_TILE {
            assert_eq!(acc.sram[y_addr + i], 2.0);
        }
    }

    #[test]
    fn matvec_tile_accumulate_false_resets_y() {
        // If accumulate=false, prior contents of y should be ignored.
        let mut acc = new_acc(MATMUL_TILE * 4 + MATMUL_TILE_ELEMENTS * 2);
        for i in 0..MATMUL_TILE {
            acc.sram[i] = 1.0;
            acc.sram[MATMUL_TILE + i * MATMUL_TILE + i] = 1.0;
        }
        let y_addr = MATMUL_TILE + MATMUL_TILE_ELEMENTS;
        // Poison y with garbage; the first MatVec should overwrite it.
        for i in 0..MATMUL_TILE {
            acc.sram[y_addr + i] = 99999.0;
        }
        acc.execute(&Instruction::MatVecTile {
            x_sram: 0, w_sram: MATMUL_TILE, y_sram: y_addr, accumulate: false,
        }).unwrap();
        for i in 0..MATMUL_TILE {
            assert_eq!(acc.sram[y_addr + i], 1.0);
        }
    }
}
