use std::env;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use annapura::gguf::{Model, Value};
use annapura::nn::{linear, rmsnorm, rope_heads};

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        bail!("usage: forward <token_id> [<token_id> ...]");
    }

    let token_ids: Vec<usize> = argv
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| anyhow!("bad token id {:?}: {}", s, e))
        })
        .collect::<Result<_>>()?;

    let model = Model::load(DEFAULT_PATH)?;

    let eps = model
        .metadata
        .get("llama.attention.layer_norm_rms_epsilon")
        .and_then(Value::as_f32)
        .ok_or_else(|| anyhow!("missing layer_norm_rms_epsilon"))?;
    let freq_base = model
        .metadata
        .get("llama.rope.freq_base")
        .and_then(Value::as_f32)
        .unwrap_or(10_000.0);
    let head_dim = model
        .metadata
        .get("llama.rope.dimension_count")
        .and_then(Value::as_u32)
        .ok_or_else(|| anyhow!("missing rope.dimension_count"))? as usize;
    let n_heads = model
        .metadata
        .get("llama.attention.head_count")
        .and_then(Value::as_u32)
        .ok_or_else(|| anyhow!("missing attention.head_count"))? as usize;
    let n_kv_heads = model
        .metadata
        .get("llama.attention.head_count_kv")
        .and_then(Value::as_u32)
        .ok_or_else(|| anyhow!("missing attention.head_count_kv"))? as usize;
    let hidden = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    let embed_table = model.tensor("token_embd.weight").unwrap();
    let attn_norm = model.tensor("blk.0.attn_norm.weight").unwrap();
    let wq = model.tensor("blk.0.attn_q.weight").unwrap();
    let wk = model.tensor("blk.0.attn_k.weight").unwrap();
    let wv = model.tensor("blk.0.attn_v.weight").unwrap();

    eprint!("loading layer 0 projection weights... ");
    let t0 = Instant::now();
    let attn_norm_w = model.dequantize(attn_norm)?;
    let wq_data = model.dequantize(wq)?;
    let wk_data = model.dequantize(wk)?;
    let wv_data = model.dequantize(wv)?;
    eprintln!(
        "{:?} ({:.1} MB of f32)",
        t0.elapsed(),
        (attn_norm_w.len() + wq_data.len() + wk_data.len() + wv_data.len()) as f64 * 4.0 / 1e6
    );

    println!();
    println!("config: hidden={}, kv_dim={}, head_dim={}, n_heads={}, n_kv_heads={}",
             hidden, kv_dim, head_dim, n_heads, n_kv_heads);
    println!("        eps={}, rope_freq_base={}", eps, freq_base);
    println!();
    println!(
        "  {:>5}  {:>3}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}",
        "tok", "pos", "‖x‖", "‖normed‖", "‖q‖", "‖k‖", "‖v‖"
    );

    let mut normed = vec![0.0_f32; hidden];
    let mut q = vec![0.0_f32; hidden];
    let mut k = vec![0.0_f32; kv_dim];
    let mut v = vec![0.0_f32; kv_dim];

    for (pos, &id) in token_ids.iter().enumerate() {
        let t_token = Instant::now();
        let x = model.dequantize_row(embed_table, id)?;

        rmsnorm(&x, &attn_norm_w, eps, &mut normed);
        linear(&normed, &wq_data, &mut q);
        linear(&normed, &wk_data, &mut k);
        linear(&normed, &wv_data, &mut v);
        rope_heads(&mut q, head_dim, pos, freq_base);
        rope_heads(&mut k, head_dim, pos, freq_base);
        // V intentionally not rotated — V is content, not position-bearing.

        println!(
            "  {:>5}  {:>3}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  [{:?}]",
            id, pos, l2(&x), l2(&normed), l2(&q), l2(&k), l2(&v),
            t_token.elapsed()
        );
    }

    Ok(())
}

fn l2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}
