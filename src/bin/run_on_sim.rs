//! Run a real TinyLlama Q-projection through the toy accelerator simulator
//! and verify the result matches our CPU reference within FP tolerance.
//!
//! Pipeline:
//!   1. Load TinyLlama, dequantize blk.0.attn_norm + blk.0.attn_q
//!   2. Embed token id 9038 (" Once"), apply attn_norm RMSNorm on CPU
//!   3. CPU reference: linear_simd_par(x_normed, Wq) — what we already trust
//!   4. Also CPU Q8 path: linear_q8_par on the packed weights (sanity)
//!   5. Simulator path: retile Wq, set up SRAM, compile_linear, run
//!   6. Print all three norms and the pairwise diffs

use std::time::Instant;

use anyhow::{anyhow, Result};

use annapura::accelerator::{Accelerator, MATMUL_TILE};
use annapura::compiler::{compile_linear, retile_weight};
use annapura::gguf::Model;
use annapura::nn::{linear_q8_par, linear_simd_par, rmsnorm};
use annapura::perf_model::{MIDRANGE_1G_4P, TOY_1G_1P, TRAINIUM_2G_16P};
use annapura::transformer::Config;

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";
const TOKEN_ID: usize = 9038; // " Once"

fn main() -> Result<()> {
    let model = Model::load(DEFAULT_PATH)?;
    let cfg = Config::from_model(&model)?;

    println!("==========================================================");
    println!(" Running blk.0.attn_q.weight (real TinyLlama Q-projection)");
    println!(" through three paths and cross-checking the output.");
    println!("==========================================================");
    println!("dims: in={}, out={}, total MACs = {}",
             cfg.hidden, cfg.hidden, cfg.hidden * cfg.hidden);
    println!();

    // === Build the input vector x_normed: embed → RMSNorm ===
    let embed_tensor = model.tensor("token_embd.weight")
        .ok_or_else(|| anyhow!("no token_embd.weight"))?;
    let attn_norm_w = model.dequantize(model.tensor("blk.0.attn_norm.weight").unwrap())?;
    let x_raw = model.dequantize_row(embed_tensor, TOKEN_ID)?;
    let mut x_normed = vec![0.0_f32; cfg.hidden];
    rmsnorm(&x_raw, &attn_norm_w, cfg.eps, &mut x_normed);
    println!("input prep: token {} → embedding → RMSNorm", TOKEN_ID);
    println!("  ‖x_raw‖₂    = {:.4}", l2(&x_raw));
    println!("  ‖x_normed‖₂ = {:.4}", l2(&x_normed));
    println!();

    // === Path A: CPU f32 (the reference oracle for the simulator) ===
    let wq_q8 = model.tensor("blk.0.attn_q.weight").unwrap();
    let wq_f32 = model.dequantize(wq_q8)?;

    let mut y_cpu_f32 = vec![0.0_f32; cfg.hidden];
    let t = Instant::now();
    linear_simd_par(&x_normed, &wq_f32, &mut y_cpu_f32);
    let cpu_f32_time = t.elapsed();

    // === Path B: CPU Q8 (the inference engine's actual path) ===
    let wq_packed = model.tensor_bytes(wq_q8);
    let mut y_cpu_q8 = vec![0.0_f32; cfg.hidden];
    let t = Instant::now();
    linear_q8_par(&x_normed, wq_packed, &mut y_cpu_q8);
    let cpu_q8_time = t.elapsed();

    // === Path C: the simulator ===
    println!("preparing simulator input (retile Wq for tile layout)...");
    let t = Instant::now();
    let wq_tiled = retile_weight(&wq_f32, cfg.hidden, cfg.hidden);
    let retile_time = t.elapsed();
    println!("  retile {} elements: {:?}", wq_tiled.len(), retile_time);

    let in_dim = cfg.hidden;
    let out_dim = cfg.hidden;
    let sram_size = in_dim + (in_dim * out_dim) + out_dim;
    let mut acc = Accelerator::new(sram_size, 0);
    let x_addr = 0;
    let w_addr = in_dim;
    let y_addr = w_addr + in_dim * out_dim;
    acc.sram[x_addr..x_addr + in_dim].copy_from_slice(&x_normed);
    acc.sram[w_addr..w_addr + in_dim * out_dim].copy_from_slice(&wq_tiled);

    let program = compile_linear(in_dim, out_dim, x_addr, w_addr, y_addr);
    let n_instr = program.len();

    println!("running {} MatVecTile instructions on the simulator...", n_instr);
    let t = Instant::now();
    acc.run(&program)?;
    let sim_time = t.elapsed();
    println!("  simulator wall-clock: {:?}", sim_time);
    let mut y_sim = vec![0.0_f32; out_dim];
    y_sim.copy_from_slice(&acc.sram[y_addr..y_addr + out_dim]);
    println!();

    // === Cross-check ===
    let cpu_f32_norm = l2(&y_cpu_f32);
    let cpu_q8_norm = l2(&y_cpu_q8);
    let sim_norm = l2(&y_sim);

    println!("=== Output norms (should all be very close) ===");
    println!("  CPU f32 (oracle):    ‖y‖₂ = {:.4}   [{:?}]", cpu_f32_norm, cpu_f32_time);
    println!("  CPU Q8 (production): ‖y‖₂ = {:.4}   [{:?}]", cpu_q8_norm, cpu_q8_time);
    println!("  Simulator:           ‖y‖₂ = {:.4}   [{:?}]", sim_norm, sim_time);
    println!();

    let (max_d_sf, rms_d_sf) = diff_stats(&y_sim, &y_cpu_f32);
    let (max_d_qf, rms_d_qf) = diff_stats(&y_cpu_q8, &y_cpu_f32);
    let (max_d_sq, rms_d_sq) = diff_stats(&y_sim, &y_cpu_q8);

    println!("=== Pairwise diffs ===");
    println!("  simulator vs CPU f32:  max={:.6e}  rms={:.6e}  ← simulator correctness", max_d_sf, rms_d_sf);
    println!("  CPU Q8    vs CPU f32:  max={:.6e}  rms={:.6e}  ← Q8 quantization error", max_d_qf, rms_d_qf);
    println!("  simulator vs CPU Q8:   max={:.6e}  rms={:.6e}  ← combined", max_d_sq, rms_d_sq);
    println!();

    let sim_passes = max_d_sf < 1e-2;
    if sim_passes {
        println!("✓ simulator output matches CPU f32 reference (max diff < 1e-2)");
        println!("✓ the math through compiler → MatVecTile → SRAM agrees with linear_simd_par");
    } else {
        println!("✗ simulator and CPU disagree — investigate retile/compile/execute math");
    }

    println!();
    println!("=== Instruction breakdown ===");
    println!("  MatVecTile dispatched: {}", n_instr);
    println!("  MACs per MatVecTile:   {}", MATMUL_TILE * MATMUL_TILE);
    let total_macs = n_instr * MATMUL_TILE * MATMUL_TILE;
    println!("  total MACs:            {} ({:.2}M)", total_macs, total_macs as f64 / 1e6);
    println!("  expected MACs (in×out):{} ({:.2}M) ✓",
             in_dim * out_dim, (in_dim * out_dim) as f64 / 1e6);

    println!();
    println!("=== Predicted silicon perf (chapter 5.4 perf model) ===");
    println!("Same program, run on three hypothetical chip designs:");
    println!();
    let cpu_q8_ms = cpu_q8_time.as_secs_f64() * 1000.0;
    for (model, label) in [
        (TOY_1G_1P,        "Toy chip    (1 GHz, 1 matmul pipe) "),
        (MIDRANGE_1G_4P,   "Mid-range   (1 GHz, 4 matmul pipes)"),
        (TRAINIUM_2G_16P,  "Trainium-ish(2 GHz, 16 matmul pipes)"),
    ] {
        let cycles = model.predict_cycles(&program);
        let predicted_ms = model.predict_ms(&program);
        let speedup_vs_cpu_q8 = cpu_q8_ms / predicted_ms;
        println!(
            "  {}: {:>10} cycles  →  {:>7.2} µs  ({:>4.1}× over CPU Q8)",
            label,
            cycles,
            predicted_ms * 1000.0,
            speedup_vs_cpu_q8
        );
    }
    println!();
    println!("The scaling story: each doubling of matmul pipes ~halves cycle count.");
    println!("Each doubling of clock ~halves wall-clock time. Multiplicatively, the");
    println!("Trainium-class design predicts ~{:.0}× speedup over our 6-P-core CPU.",
             cpu_q8_ms / TRAINIUM_2G_16P.predict_ms(&program));
    println!();
    println!("Caveat: this is a first-order latency model. Real silicon has");
    println!("pipeline hazards, port contention, DRAM stalls, thermal limits.");
    println!("Chapter 6 will layer a cycle-accurate microarch model on top.");

    Ok(())
}

fn l2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn diff_stats(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut max_abs = 0.0_f32;
    let mut sum_sq = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        let d = (x - y).abs();
        if d > max_abs { max_abs = d; }
        sum_sq += d * d;
    }
    (max_abs, (sum_sq / a.len() as f32).sqrt())
}
