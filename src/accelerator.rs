//! Toy ML accelerator — vector machine with on-chip SRAM.
//!
//! Goal: the simplest hardware design that could plausibly run a transformer
//! layer end-to-end once we add a matmul tile (next move) and a compiler
//! (chapter 5.2). For now this is the *functional* simulator — correct math,
//! not cycle-accurate. Chapter 6 will layer timing/pipelining on top.
//!
//! Architectural sketch (chapter 5.0 scope, the rest comes in 5.1+):
//!   - 32 vector registers, each 32 lanes of f32 (4 KB of vreg state)
//!   - On-chip SRAM scratchpad (sized at creation; target ~50 MB to hold
//!     one TinyLlama layer's worth of weights at f32)
//!   - DRAM is the host's Vec; explicit `LoadDram`/`StoreDram` moves data
//!     in/out of SRAM (added in chapter 5.1)
//!   - Matrix accumulator + `MatMulTile` instruction (chapter 5.1)
//!
//! Connection to the roofline numbers from chapter 2.8:
//!   - Vector arithmetic gives us 32-wide FMA throughput per instruction.
//!     At a hypothetical 1 GHz issue rate, that's 64 GFLOPS per vector pipe.
//!     Stacking pipes is how we'd hit the 1 TFLOPS target.
//!   - SRAM-resident weights mean the FLOPS/byte ridge moves left compared
//!     to CPU's DRAM-bound regime → the workload becomes compute-bound,
//!     which is the "good" regime where more transistors buy more throughput.

use anyhow::{bail, Result};

pub const VECTOR_LANES: usize = 32;
pub const N_VECTOR_REGS: usize = 32;

pub struct Accelerator {
    pub sram: Vec<f32>,
    pub vregs: Box<[[f32; VECTOR_LANES]; N_VECTOR_REGS]>,
    /// Counts dispatched instructions. The cycle-accurate model in chapter 6
    /// will replace this with realistic pipeline timing.
    pub instr_count: u64,
}

impl Accelerator {
    pub fn new(sram_elements: usize) -> Self {
        Self {
            sram: vec![0.0; sram_elements],
            vregs: Box::new([[0.0; VECTOR_LANES]; N_VECTOR_REGS]),
            instr_count: 0,
        }
    }

    pub fn sram_size_bytes(&self) -> usize {
        self.sram.len() * 4
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
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
}

impl Accelerator {
    pub fn execute(&mut self, instr: &Instruction) -> Result<()> {
        use Instruction::*;
        match *instr {
            LoadVec { v, sram_addr } => {
                let v = idx(v, "vreg")?;
                let end = sram_addr
                    .checked_add(VECTOR_LANES)
                    .ok_or_else(|| anyhow::anyhow!("address overflow"))?;
                if end > self.sram.len() {
                    bail!("LoadVec at {} reads past SRAM end {}", end, self.sram.len());
                }
                self.vregs[v].copy_from_slice(&self.sram[sram_addr..end]);
            }
            StoreVec { v, sram_addr } => {
                let v = idx(v, "vreg")?;
                let end = sram_addr
                    .checked_add(VECTOR_LANES)
                    .ok_or_else(|| anyhow::anyhow!("address overflow"))?;
                if end > self.sram.len() {
                    bail!("StoreVec at {} writes past SRAM end {}", end, self.sram.len());
                }
                self.sram[sram_addr..end].copy_from_slice(&self.vregs[v]);
            }
            VAdd { a, b, c } => {
                let (a, b, c) = (idx(a, "vreg")?, idx(b, "vreg")?, idx(c, "vreg")?);
                let va = self.vregs[a];
                let vb = self.vregs[b];
                for i in 0..VECTOR_LANES {
                    self.vregs[c][i] = va[i] + vb[i];
                }
            }
            VMul { a, b, c } => {
                let (a, b, c) = (idx(a, "vreg")?, idx(b, "vreg")?, idx(c, "vreg")?);
                let va = self.vregs[a];
                let vb = self.vregs[b];
                for i in 0..VECTOR_LANES {
                    self.vregs[c][i] = va[i] * vb[i];
                }
            }
            VFma { a, b, c, d } => {
                let (a, b, c, d) =
                    (idx(a, "vreg")?, idx(b, "vreg")?, idx(c, "vreg")?, idx(d, "vreg")?);
                let va = self.vregs[a];
                let vb = self.vregs[b];
                let vc = self.vregs[c];
                for i in 0..VECTOR_LANES {
                    self.vregs[d][i] = va[i] * vb[i] + vc[i];
                }
            }
            VSplat { v, scalar } => {
                let v = idx(v, "vreg")?;
                self.vregs[v] = [scalar; VECTOR_LANES];
            }
            VSilu { v_in, v_out } => {
                let (i, o) = (idx(v_in, "vreg")?, idx(v_out, "vreg")?);
                let src = self.vregs[i];
                for k in 0..VECTOR_LANES {
                    let x = src[k];
                    self.vregs[o][k] = x / (1.0 + (-x).exp());
                }
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

fn idx(raw: u8, kind: &str) -> Result<usize> {
    let i = raw as usize;
    if i >= N_VECTOR_REGS {
        bail!("{} register {} out of range (have {})", kind, i, N_VECTOR_REGS);
    }
    Ok(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vsplat_then_vadd() {
        let mut acc = Accelerator::new(128);
        acc.execute(&Instruction::VSplat { v: 0, scalar: 1.0 }).unwrap();
        acc.execute(&Instruction::VSplat { v: 1, scalar: 2.0 }).unwrap();
        acc.execute(&Instruction::VAdd { a: 0, b: 1, c: 2 }).unwrap();
        assert_eq!(acc.vregs[2], [3.0; VECTOR_LANES]);
    }

    #[test]
    fn vfma_lanewise() {
        let mut acc = Accelerator::new(128);
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
        // y = a*b + c per lane
        assert_eq!(acc.vregs[3][0], 4.5);  // 1*4 + 0.5
        assert_eq!(acc.vregs[3][1], 10.5); // 2*5 + 0.5
        assert_eq!(acc.vregs[3][2], 18.5); // 3*6 + 0.5
    }

    #[test]
    fn loadvec_storevec_roundtrip() {
        let mut acc = Accelerator::new(128);
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
        let mut acc = Accelerator::new(64);
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
        let mut acc = Accelerator::new(64);
        assert_eq!(acc.instr_count, 0);
        acc.run(&[
            Instruction::VSplat { v: 0, scalar: 1.0 },
            Instruction::VSplat { v: 1, scalar: 2.0 },
            Instruction::VAdd { a: 0, b: 1, c: 2 },
        ]).unwrap();
        assert_eq!(acc.instr_count, 3);
    }

    #[test]
    fn out_of_range_vreg_errors() {
        let mut acc = Accelerator::new(64);
        assert!(acc.execute(&Instruction::VSplat { v: 200, scalar: 0.0 }).is_err());
        // Counter should NOT increment on failed instruction.
        assert_eq!(acc.instr_count, 0);
    }

    #[test]
    fn out_of_bounds_sram_errors() {
        let mut acc = Accelerator::new(VECTOR_LANES - 1);
        assert!(acc.execute(&Instruction::LoadVec { v: 0, sram_addr: 0 }).is_err());
    }
}
