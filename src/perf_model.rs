//! Simple latency-based perf model for the toy accelerator.
//!
//! Each instruction is assigned a fixed cycle latency. Compute instructions
//! (MatMulTile, MatVecTile) can run in parallel across `n_matmul_pipes`;
//! other instructions run on a single shared port. Total time =
//! max(compute_cycles_per_pipe, other_cycles).
//!
//! Connection to the real world:
//!   - This is the cycle model an Annapurna architect would sketch on paper
//!     when proposing a chip. It captures the first-order tradeoff between
//!     clock frequency and number of matmul pipes — the two big knobs.
//!   - It does NOT model: pipeline hazards, port contention, DRAM stalls,
//!     thermal throttling, multi-issue width, branch effects. Those are
//!     chapter 6 (cycle-accurate microarchitecture model) territory.
//!   - Two extreme bookends every Annapurna engineer keeps in mind:
//!       optimistic = fully pipelined, 1 throughput cycle per matmul
//!       pessimistic = full latency in sequence, no parallelism
//!     This model sits in between: full latency, but parallel across pipes.

use crate::accelerator::Instruction;

/// Cycle latency for one instruction on the toy accelerator. Numbers chosen
/// to roughly match a 1 GHz mid-range ML accelerator design (Trainium-style).
pub fn cycle_cost(instr: &Instruction) -> u64 {
    use Instruction::*;
    match instr {
        // Vector load/store from SRAM — 1 cycle, pipelined.
        LoadVec { .. } | StoreVec { .. } => 1,
        // Simple vector arithmetic — 1 cycle on a vector pipe.
        VAdd { .. } | VMul { .. } | VFma { .. } | VSplat { .. } => 1,
        // Horizontal sum-of-lanes via log-tree reduction network. log2(32)=5
        // levels of pairwise add, but real designs pipeline this to ~4 cycles.
        VReduceSum { .. } => 4,
        // Transcendental ops — need an exp/log/rsqrt helper, ~8 cycles each.
        VSilu { .. } | VRsqrt { .. } => 8,
        // Matrix accumulator ops — 1 cycle (single register write).
        MatAccumClear | MatAccumStore { .. } => 1,
        // 16×16 matmul tile — 256 MACs streamed through a 16-wide systolic
        // array in 16 cycles. Same for 1×16·16×16 matvec.
        MatMulTile { .. } | MatVecTile { .. } => 16,
        // DRAM transfer: ~100 cycle first-access latency + 1 cycle per
        // 32-element burst (assumes 32 elements/cycle DRAM bandwidth).
        LoadDram { len, .. } | StoreDram { len, .. } => 100 + (*len as u64).div_ceil(32),
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineModel {
    pub clock_ghz: f64,
    pub n_matmul_pipes: u32,
}

impl PipelineModel {
    pub fn predict_cycles(&self, program: &[Instruction]) -> u64 {
        assert!(self.n_matmul_pipes >= 1, "need at least one matmul pipe");
        let mut compute_cycles: u64 = 0;
        let mut other_cycles: u64 = 0;
        for instr in program {
            let cost = cycle_cost(instr);
            match instr {
                Instruction::MatMulTile { .. } | Instruction::MatVecTile { .. } => {
                    compute_cycles += cost;
                }
                _ => other_cycles += cost,
            }
        }
        let compute_per_pipe = compute_cycles.div_ceil(self.n_matmul_pipes as u64);
        // Compute path and non-compute path can run in parallel on real silicon;
        // take the max as the bound.
        compute_per_pipe.max(other_cycles)
    }

    pub fn predict_ms(&self, program: &[Instruction]) -> f64 {
        self.predict_cycles(program) as f64 / (self.clock_ghz * 1e9) * 1000.0
    }
}

/// Three preset hardware configurations representing common Annapurna-style
/// design points. The middle one is what you'd reasonably tape out as a
/// first product; the third is roughly Trainium-2 scale.
pub const TOY_1G_1P: PipelineModel = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 1 };
pub const MIDRANGE_1G_4P: PipelineModel = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 4 };
pub const TRAINIUM_2G_16P: PipelineModel =
    PipelineModel { clock_ghz: 2.0, n_matmul_pipes: 16 };

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cycle_costs_match_design() {
        assert_eq!(cycle_cost(&Instruction::LoadVec { v: 0, sram_addr: 0 }), 1);
        assert_eq!(cycle_cost(&Instruction::VSplat { v: 0, scalar: 0.0 }), 1);
        assert_eq!(cycle_cost(&Instruction::VSilu { v_in: 0, v_out: 0 }), 8);
        assert_eq!(cycle_cost(&Instruction::VRsqrt { v_in: 0, v_out: 0 }), 8);
        assert_eq!(cycle_cost(&Instruction::VReduceSum { v_in: 0, v_out: 0 }), 4);
        assert_eq!(
            cycle_cost(&Instruction::MatVecTile { x_sram: 0, w_sram: 0, y_sram: 0, accumulate: false }),
            16
        );
        assert_eq!(cycle_cost(&Instruction::MatMulTile { a_sram: 0, b_sram: 0 }), 16);
        assert_eq!(
            cycle_cost(&Instruction::LoadDram { dram_addr: 0, sram_addr: 0, len: 64 }),
            // 100 latency + 64/32 = 2 burst → 102
            102
        );
    }

    #[test]
    fn empty_program_takes_zero_cycles() {
        let m = TOY_1G_1P;
        assert_eq!(m.predict_cycles(&[]), 0);
    }

    #[test]
    fn one_matvec_costs_16_with_one_pipe() {
        let m = TOY_1G_1P;
        let p = vec![Instruction::MatVecTile { x_sram: 0, w_sram: 0, y_sram: 0, accumulate: false }];
        assert_eq!(m.predict_cycles(&p), 16);
    }

    #[test]
    fn parallel_pipes_amortize_matmul_cost() {
        let p: Vec<Instruction> = (0..16)
            .map(|_| Instruction::MatVecTile {
                x_sram: 0, w_sram: 0, y_sram: 0, accumulate: false,
            })
            .collect();
        // 16 matvec × 16 cycles = 256 compute cycles
        assert_eq!(TOY_1G_1P.predict_cycles(&p), 256);
        // With 4 pipes: 256 / 4 = 64 cycles
        let m4 = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 4 };
        assert_eq!(m4.predict_cycles(&p), 64);
        // With 16 pipes: 256 / 16 = 16 cycles
        let m16 = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 16 };
        assert_eq!(m16.predict_cycles(&p), 16);
    }

    #[test]
    fn non_compute_does_not_parallelize_with_extra_pipes() {
        // Pure vector ops — adding matmul pipes shouldn't help.
        let p: Vec<Instruction> = (0..100)
            .map(|_| Instruction::VAdd { a: 0, b: 1, c: 2 })
            .collect();
        let m1 = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 1 };
        let m16 = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 16 };
        assert_eq!(m1.predict_cycles(&p), 100);
        assert_eq!(m16.predict_cycles(&p), 100);
    }

    #[test]
    fn predict_ms_scales_with_frequency() {
        let p = vec![Instruction::MatVecTile { x_sram: 0, w_sram: 0, y_sram: 0, accumulate: false }; 1000];
        // 1000 × 16 = 16000 cycles on 1 pipe.
        // At 1 GHz: 16 µs. At 2 GHz: 8 µs.
        let m1g = PipelineModel { clock_ghz: 1.0, n_matmul_pipes: 1 };
        let m2g = PipelineModel { clock_ghz: 2.0, n_matmul_pipes: 1 };
        let t1 = m1g.predict_ms(&p);
        let t2 = m2g.predict_ms(&p);
        assert!((t1 - 0.016).abs() < 1e-9);
        assert!((t2 - 0.008).abs() < 1e-9);
    }
}
