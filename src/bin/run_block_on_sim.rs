//! Run a complete TinyLlama transformer block on the toy accelerator
//! simulator and cross-check the output against a hand-rolled CPU oracle.
//!
//! This is the chapter 5 trophy: every primitive of a Llama block —
//! RMSNorm, Q/K/V projection (Q/Wo/FFN on chip, K/V on CPU for now),
//! RoPE, multi-head GQA attention, SwiGLU FFN, residuals — composed and
//! lowered onto our toy ISA, running real Q8-dequantized weights and
//! producing output that matches the CPU reference within FP tolerance.
//!
//! Pipeline:
//!   1. Embed token id 9038 (" Once") → x_raw.
//!   2. CPU oracle: hand-roll the block forward pass.
//!   3. Pre-populate the simulator's K_T / V caches with this token's K, V.
//!   4. Build the SRAM layout, retile weights, lower the block, run.
//!   5. Cross-check final x[hidden] vs the CPU oracle.

use std::time::Instant;

use anyhow::{anyhow, Result};

use annapura::accelerator::{Accelerator, VECTOR_LANES};
use annapura::attention::{attention, KvCache};
use annapura::compiler::{
    attention_mask, build_rope_tables, compile_llama_block, retile_weight,
    split_v_per_kv_head, transpose_k_multihead, LlamaBlockLayout,
};
use annapura::gguf::Model;
use annapura::nn::{linear, rmsnorm, rope_heads};
use annapura::transformer::Config;

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";
const TOKEN_ID: usize = 9038; // " Once"
const SEQ_LEN: usize = 1; // single token, no prior cache
const CUR_POS: usize = 0;

fn main() -> Result<()> {
    let model = Model::load(DEFAULT_PATH)?;
    let cfg = Config::from_model(&model)?;

    println!("==========================================================");
    println!(" Running TinyLlama blk.0 on the toy accelerator simulator");
    println!("==========================================================");
    println!(
        "config: hidden={}, head_dim={}, n_heads={}, n_kv_heads={}, ffn_hidden={}",
        cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, cfg.intermediate,
    );
    println!("token: id={} (\" Once\"), cur_pos={}", TOKEN_ID, CUR_POS);
    println!();

    // ===== Load + dequantize blk.0 weights (need f32 for the sim) =====
    println!("loading + dequantizing blk.0 weights...");
    let t = Instant::now();
    let embed_tensor = model.tensor("token_embd.weight")
        .ok_or_else(|| anyhow!("no token_embd.weight"))?;
    let x_raw = model.dequantize_row(embed_tensor, TOKEN_ID)?;

    let attn_norm = model.dequantize(model.tensor("blk.0.attn_norm.weight").unwrap())?;
    let ffn_norm = model.dequantize(model.tensor("blk.0.ffn_norm.weight").unwrap())?;
    let wq = model.dequantize(model.tensor("blk.0.attn_q.weight").unwrap())?;
    let wk = model.dequantize(model.tensor("blk.0.attn_k.weight").unwrap())?;
    let wv = model.dequantize(model.tensor("blk.0.attn_v.weight").unwrap())?;
    let wo = model.dequantize(model.tensor("blk.0.attn_output.weight").unwrap())?;
    let w_gate = model.dequantize(model.tensor("blk.0.ffn_gate.weight").unwrap())?;
    let w_up = model.dequantize(model.tensor("blk.0.ffn_up.weight").unwrap())?;
    let w_down = model.dequantize(model.tensor("blk.0.ffn_down.weight").unwrap())?;
    println!("  dequantize: {:?}", t.elapsed());
    let weight_mb = (wq.len() + wk.len() + wv.len() + wo.len()
        + w_gate.len() + w_up.len() + w_down.len()) as f64 * 4.0 / 1e6;
    println!("  total Q8 weights (f32 materialized): {:.1} MB", weight_mb);
    println!();

    let hidden = cfg.hidden;
    let head_dim = cfg.head_dim;
    let n_heads = cfg.n_heads;
    let n_kv_heads = cfg.n_kv_heads;
    let ffn_hidden = cfg.intermediate;
    let kv_dim = cfg.kv_dim;
    let rms_eps = cfg.eps;
    let freq_base = cfg.freq_base;

    // ===== CPU oracle: hand-roll the block forward pass =====
    println!("CPU oracle: hand-rolling block forward...");
    let t = Instant::now();
    let mut x_cpu = x_raw.clone();
    let mut x_normed_1 = vec![0.0_f32; hidden];
    rmsnorm(&x_cpu, &attn_norm, rms_eps, &mut x_normed_1);
    let mut q_cpu = vec![0.0_f32; hidden];
    let mut k_cpu = vec![0.0_f32; kv_dim];
    let mut v_cpu = vec![0.0_f32; kv_dim];
    linear(&x_normed_1, &wq, &mut q_cpu);
    linear(&x_normed_1, &wk, &mut k_cpu);
    linear(&x_normed_1, &wv, &mut v_cpu);
    rope_heads(&mut q_cpu, head_dim, CUR_POS, freq_base);
    rope_heads(&mut k_cpu, head_dim, CUR_POS, freq_base);

    let mut cache = KvCache::new(SEQ_LEN, kv_dim);
    cache.store(CUR_POS, &k_cpu, &v_cpu);
    let mut attn_out_cpu = vec![0.0_f32; hidden];
    attention(&q_cpu, &cache, CUR_POS, n_heads, n_kv_heads, head_dim, &mut attn_out_cpu);
    let mut attn_proj_cpu = vec![0.0_f32; hidden];
    linear(&attn_out_cpu, &wo, &mut attn_proj_cpu);
    for i in 0..hidden {
        x_cpu[i] += attn_proj_cpu[i];
    }
    let mut x_normed_2 = vec![0.0_f32; hidden];
    rmsnorm(&x_cpu, &ffn_norm, rms_eps, &mut x_normed_2);
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
    let cpu_time = t.elapsed();
    println!("  CPU block: {:?}", cpu_time);
    println!("  ‖x_cpu‖₂ = {:.4}", l2(&x_cpu));
    println!();

    // ===== Simulator path =====
    println!("preparing simulator inputs (retile + cache layout)...");
    let t = Instant::now();
    let mut k_flat = vec![0.0_f32; SEQ_LEN * kv_dim];
    let mut v_flat = vec![0.0_f32; SEQ_LEN * kv_dim];
    k_flat[CUR_POS * kv_dim..(CUR_POS + 1) * kv_dim].copy_from_slice(&k_cpu);
    v_flat[CUR_POS * kv_dim..(CUR_POS + 1) * kv_dim].copy_from_slice(&v_cpu);
    let k_t = transpose_k_multihead(&k_flat, SEQ_LEN, n_kv_heads, head_dim);
    let v_split = split_v_per_kv_head(&v_flat, SEQ_LEN, n_kv_heads, head_dim);
    let mask = attention_mask(SEQ_LEN);
    let (rope_cos, rope_sin_pm) = build_rope_tables(head_dim, CUR_POS, freq_base);
    let wq_tiled = retile_weight(&wq, hidden, hidden);
    let wo_tiled = retile_weight(&wo, hidden, hidden);
    let w_gate_tiled = retile_weight(&w_gate, hidden, ffn_hidden);
    let w_up_tiled = retile_weight(&w_up, hidden, ffn_hidden);
    let w_down_tiled = retile_weight(&w_down, ffn_hidden, hidden);
    let prep_time = t.elapsed();
    println!("  retile + cache prep: {:?}", prep_time);

    // SRAM bump allocator.
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
    let sram_mb = (size as f64) * 4.0 / 1e6;
    println!("  SRAM total: {} floats ({:.1} MB)", size, sram_mb);

    let mut acc = Accelerator::new(size, 0);
    acc.sram[x_addr..x_addr + hidden].copy_from_slice(&x_raw);
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
        SEQ_LEN, rms_eps, layout,
    );
    println!("compiled block: {} instructions", prog.len());
    let matvec_count = prog.iter().filter(|i| matches!(i,
        annapura::accelerator::Instruction::MatVecTile { .. })).count();
    println!("  of which MatVecTile: {} ({:.2}M MACs)",
        matvec_count, (matvec_count * 256) as f64 / 1e6);
    println!();

    println!("running on simulator (~{:.1} MB SRAM, single-threaded interp)...", sram_mb);
    let t = Instant::now();
    acc.run(&prog)?;
    let sim_time = t.elapsed();
    println!("  simulator wall-clock: {:?}", sim_time);

    let mut x_sim = vec![0.0_f32; hidden];
    x_sim.copy_from_slice(&acc.sram[x_addr..x_addr + hidden]);
    println!("  ‖x_sim‖₂ = {:.4}", l2(&x_sim));
    println!();

    // ===== Cross-check =====
    let (max_d, rms_d) = diff_stats(&x_sim, &x_cpu);
    println!("=== Final block output: simulator vs CPU oracle ===");
    println!("  max diff: {:.6e}", max_d);
    println!("  rms diff: {:.6e}", rms_d);
    println!();
    if max_d < 1e-2 {
        println!("✓ simulator matches CPU within 1e-2 — TROPHY UNLOCKED");
        println!("  every primitive of a Llama block — RMSNorm, Linear (Q/Wo/FFN),");
        println!("  RoPE, attention (multi-head + GQA), SwiGLU FFN, residuals —");
        println!("  composes and runs correctly on the toy accelerator.");
        println!();
        println!("  caveat: K/V projection still runs on CPU because writing into");
        println!("  the K_T cache layout needs a single-lane SRAM-store ISA op.");
        println!("  that's the next chapter (5.9: autonomous KV cache writes).");
    } else {
        println!("✗ simulator and CPU disagree — investigate per-step");
        return Err(anyhow!("end-to-end mismatch"));
    }

    println!();
    println!("=== Predicted silicon perf (chapter 5.4 perf model) ===");
    let cpu_ms = cpu_time.as_secs_f64() * 1000.0;
    for (model, label) in [
        (annapura::perf_model::TOY_1G_1P,       "Toy chip    (1 GHz, 1 matmul pipe) "),
        (annapura::perf_model::MIDRANGE_1G_4P,  "Mid-range   (1 GHz, 4 matmul pipes)"),
        (annapura::perf_model::TRAINIUM_2G_16P, "Trainium-ish(2 GHz, 16 matmul pipes)"),
    ] {
        let cycles = model.predict_cycles(&prog);
        let predicted_ms = model.predict_ms(&prog);
        let speedup = cpu_ms / predicted_ms;
        println!(
            "  {}: {:>12} cycles  →  {:>9.2} µs  ({:>5.1}× over CPU f32)",
            label, cycles, predicted_ms * 1000.0, speedup,
        );
    }

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
