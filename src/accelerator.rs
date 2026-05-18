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

    // === Matrix unit (chapter 5.1) ===
    /// Zero the matrix accumulator.
    MatAccumClear,
    /// `matrix_accum += A · B` where A and B are 16×16 tiles in SRAM
    /// (row-major, contiguous). A starts at `a_sram`, B at `b_sram`.
    MatMulTile { a_sram: usize, b_sram: usize },
    /// `SRAM[dst .. dst + 256] = matrix_accum` (row-major).
    MatAccumStore { sram_dst: usize },

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
}
