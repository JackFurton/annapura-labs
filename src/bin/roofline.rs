//! Roofline performance model — prints per-op arithmetic intensity and
//! per-hardware predicted time to motivate accelerator design.
//!
//! Reads the model's architecture from the GGUF metadata and computes:
//!   - FLOPS per op (per token)
//!   - Bytes accessed per op (weights as Q8_0, activations as f32)
//!   - Arithmetic intensity = FLOPS / bytes
//!   - Predicted time on each hardware = max(FLOPS/peak_FLOPS, bytes/peak_BW)
//!
//! The whole point: comparing two hardware configs side by side makes the
//! "why silicon helps" argument quantitative. CPU sits below the ridge for
//! linear ops (memory-bound). On-chip-SRAM accelerator shifts the ridge so
//! the same ops become compute-bound — the regime where more transistors
//! actually buy you more throughput.

use anyhow::Result;

use annapura::gguf::Model;
use annapura::transformer::Config;

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

/// Q8_0 packs 32 elements per block of 34 bytes → 34/32 ≈ 1.0625 bytes/element.
const Q8_BYTES_PER_ELEM: f64 = 34.0 / 32.0;

#[derive(Clone, Copy)]
struct Hardware {
    name: &'static str,
    notes: &'static str,
    peak_gflops: f64,
    peak_bw_gbs: f64,
}

const M3_PRO_PEAK: Hardware = Hardware {
    name: "M3 Pro (CPU peak)",
    notes: "6 P-cores × NEON FMA + unified mem",
    peak_gflops: 500.0,
    peak_bw_gbs: 150.0,
};

const TOY_ACCEL: Hardware = Hardware {
    name: "Toy accelerator (chapter 5)",
    notes: "1 TFLOPS, ~50 MB on-chip SRAM @ 500 GB/s",
    peak_gflops: 1000.0,
    peak_bw_gbs: 500.0,
};

const TRAINIUM_CLASS: Hardware = Hardware {
    name: "Trainium-class accelerator",
    notes: "10 TFLOPS, HBM at 1 TB/s — ambitious chapter 6 target",
    peak_gflops: 10_000.0,
    peak_bw_gbs: 1000.0,
};

impl Hardware {
    fn ridge(&self) -> f64 {
        self.peak_gflops / self.peak_bw_gbs
    }

    fn time_ms(&self, flops: f64, bytes: f64) -> f64 {
        let compute_ms = flops / (self.peak_gflops * 1e9) * 1000.0;
        let memory_ms = bytes / (self.peak_bw_gbs * 1e9) * 1000.0;
        compute_ms.max(memory_ms)
    }

    fn bound(&self, ai: f64) -> &'static str {
        if ai < self.ridge() { "MEM-bound" } else { "CMP-bound" }
    }
}

struct LinearOp {
    name: &'static str,
    in_dim: usize,
    out_dim: usize,
}

impl LinearOp {
    fn flops(&self) -> f64 {
        2.0 * (self.in_dim * self.out_dim) as f64
    }

    fn bytes(&self) -> f64 {
        let weight_bytes = (self.in_dim * self.out_dim) as f64 * Q8_BYTES_PER_ELEM;
        let activation_bytes = ((self.in_dim + self.out_dim) * 4) as f64;
        weight_bytes + activation_bytes
    }

    fn ai(&self) -> f64 {
        self.flops() / self.bytes()
    }
}

fn main() -> Result<()> {
    let model = Model::load(DEFAULT_PATH)?;
    let cfg = Config::from_model(&model)?;

    let ops_per_layer = [
        LinearOp { name: "Q proj",    in_dim: cfg.hidden, out_dim: cfg.hidden },
        LinearOp { name: "K proj",    in_dim: cfg.hidden, out_dim: cfg.kv_dim },
        LinearOp { name: "V proj",    in_dim: cfg.hidden, out_dim: cfg.kv_dim },
        LinearOp { name: "O proj",    in_dim: cfg.hidden, out_dim: cfg.hidden },
        LinearOp { name: "gate proj", in_dim: cfg.hidden, out_dim: cfg.intermediate },
        LinearOp { name: "up proj",   in_dim: cfg.hidden, out_dim: cfg.intermediate },
        LinearOp { name: "down proj", in_dim: cfg.intermediate, out_dim: cfg.hidden },
    ];

    println!("=== Per-layer linear ops (one token's forward pass) ===\n");
    println!("  {:<11} {:>10} {:>10} {:>10}", "op", "FLOPS", "bytes", "AI");
    let mut layer_flops = 0.0;
    let mut layer_bytes = 0.0;
    for op in &ops_per_layer {
        println!("  {:<11} {:>9.2}M {:>9.2}M {:>10.2}",
                 op.name, op.flops() / 1e6, op.bytes() / 1e6, op.ai());
        layer_flops += op.flops();
        layer_bytes += op.bytes();
    }
    println!("  {:<11} {:>9.2}M {:>9.2}M {:>10.2}",
             "TOTAL/layer", layer_flops / 1e6, layer_bytes / 1e6, layer_flops / layer_bytes);

    let total_flops = layer_flops * cfg.n_layers as f64;
    let total_bytes = layer_bytes * cfg.n_layers as f64;
    let overall_ai = total_flops / total_bytes;

    println!();
    println!("Per-token total ({} layers, linear ops only — excludes norms/RoPE/attention):", cfg.n_layers);
    println!("  FLOPS:    {:.2} G", total_flops / 1e9);
    println!("  bytes:    {:.2} MB", total_bytes / 1e6);
    println!("  AI:       {:.2} FLOPS/byte\n", overall_ai);

    println!("=== Roofline comparison ===\n");
    println!("  {:<30} {:>8} {:>8} {:>8} {:>10} {:>12}",
             "hardware", "peak GF", "peak GB", "ridge", "bound", "pred (ms)");
    for hw in [&M3_PRO_PEAK, &TOY_ACCEL, &TRAINIUM_CLASS] {
        let pred = hw.time_ms(total_flops, total_bytes);
        println!("  {:<30} {:>8.0} {:>8.0} {:>8.2} {:>10} {:>11.2}ms",
                 hw.name, hw.peak_gflops, hw.peak_bw_gbs, hw.ridge(),
                 hw.bound(overall_ai), pred);
    }

    println!();
    println!("    {}", M3_PRO_PEAK.notes);
    println!("    {}", TOY_ACCEL.notes);
    println!("    {}", TRAINIUM_CLASS.notes);

    let cpu_ms = M3_PRO_PEAK.time_ms(total_flops, total_bytes);
    let toy_ms = TOY_ACCEL.time_ms(total_flops, total_bytes);
    let prod_ms = TRAINIUM_CLASS.time_ms(total_flops, total_bytes);

    println!();
    println!("=== Verdict ===\n");
    println!("  CPU (M3 Pro):    {:.2} ms/tok ≈ {:>5.0} tok/s   ← what we have now (measured ~28 ms)",
             cpu_ms, 1000.0 / cpu_ms);
    println!("  Toy accelerator: {:.2} ms/tok ≈ {:>5.0} tok/s   ← {:.1}× speedup, chapter 5 target",
             toy_ms, 1000.0 / toy_ms, cpu_ms / toy_ms);
    println!("  Trainium-class:  {:.2} ms/tok ≈ {:>5.0} tok/s   ← {:.1}× speedup, ambitious chapter 6 target",
             prod_ms, 1000.0 / prod_ms, cpu_ms / prod_ms);
    println!();
    println!("  Architectural lesson: linear ops have AI ≈ {:.1} FLOPS/byte.", overall_ai);
    println!("  CPU ridge is {:.1} → workload sits BELOW it → memory-bound.", M3_PRO_PEAK.ridge());
    println!("  Toy ridge is {:.1} → still memory-bound but BW is 3× faster.", TOY_ACCEL.ridge());
    println!("  Trainium ridge is {:.1} → workload finally COMPUTE-bound at this scale.",
             TRAINIUM_CLASS.ridge());
    println!();
    println!("  Bigger picture: AI improves dramatically with **batched inference**");
    println!("  (each weight loaded once, reused across N sequences → AI scales ~N×).");
    println!("  That's why production serving runs batches of 32-256 sequences,");
    println!("  not single tokens. Chapter 4 (serving + batching) addresses this.");

    Ok(())
}
