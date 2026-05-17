use std::env;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use annapura::attention::{attention, attention_pattern, KvCache};
use annapura::gguf::{Model, Value};
use annapura::nn::{add_in_place, linear, mul_in_place, rmsnorm, rope_heads, silu_in_place};

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        bail!("usage: forward <token_id> [<token_id> ...]");
    }
    let token_ids: Vec<usize> = argv
        .iter()
        .map(|s| s.parse::<usize>().map_err(|e| anyhow!("bad token id {:?}: {}", s, e)))
        .collect::<Result<_>>()?;

    let model = Model::load(DEFAULT_PATH)?;

    let eps = meta_f32(&model, "llama.attention.layer_norm_rms_epsilon")?;
    let freq_base = model
        .metadata
        .get("llama.rope.freq_base")
        .and_then(Value::as_f32)
        .unwrap_or(10_000.0);
    let head_dim = meta_u32(&model, "llama.rope.dimension_count")? as usize;
    let n_heads = meta_u32(&model, "llama.attention.head_count")? as usize;
    let n_kv_heads = meta_u32(&model, "llama.attention.head_count_kv")? as usize;
    let max_seq_len = meta_u32(&model, "llama.context_length")? as usize;
    let intermediate = meta_u32(&model, "llama.feed_forward_length")? as usize;
    let hidden = n_heads * head_dim;
    let kv_dim = n_kv_heads * head_dim;

    eprint!("loading layer 0 weights... ");
    let t0 = Instant::now();
    let attn_norm_w = model.dequantize(model.tensor("blk.0.attn_norm.weight").unwrap())?;
    let wq = model.dequantize(model.tensor("blk.0.attn_q.weight").unwrap())?;
    let wk = model.dequantize(model.tensor("blk.0.attn_k.weight").unwrap())?;
    let wv = model.dequantize(model.tensor("blk.0.attn_v.weight").unwrap())?;
    let wo = model.dequantize(model.tensor("blk.0.attn_output.weight").unwrap())?;
    let ffn_norm_w = model.dequantize(model.tensor("blk.0.ffn_norm.weight").unwrap())?;
    let w_gate = model.dequantize(model.tensor("blk.0.ffn_gate.weight").unwrap())?;
    let w_up = model.dequantize(model.tensor("blk.0.ffn_up.weight").unwrap())?;
    let w_down = model.dequantize(model.tensor("blk.0.ffn_down.weight").unwrap())?;
    let total_mb = (attn_norm_w.len()
        + wq.len() + wk.len() + wv.len() + wo.len()
        + ffn_norm_w.len() + w_gate.len() + w_up.len() + w_down.len()) as f64 * 4.0 / 1e6;
    eprintln!("{:?} ({:.1} MB of f32)", t0.elapsed(), total_mb);

    let embed_table = model.tensor("token_embd.weight").unwrap();

    println!();
    println!("config: hidden={}, intermediate={}, kv_dim={}, head_dim={}, n_heads={}, n_kv_heads={} ({} Q heads per KV)",
             hidden, intermediate, kv_dim, head_dim, n_heads, n_kv_heads, n_heads / n_kv_heads);
    println!("        eps={}, rope_freq_base={}, max_seq_len={}", eps, freq_base, max_seq_len);
    println!();

    let mut cache = KvCache::new(max_seq_len, kv_dim);

    // Persistent scratch buffers, allocated once outside the per-token loop.
    let mut normed = vec![0.0_f32; hidden];
    let mut q = vec![0.0_f32; hidden];
    let mut k = vec![0.0_f32; kv_dim];
    let mut v = vec![0.0_f32; kv_dim];
    let mut attn = vec![0.0_f32; hidden];
    let mut attn_proj = vec![0.0_f32; hidden];
    let mut ffn_normed = vec![0.0_f32; hidden];
    let mut gate = vec![0.0_f32; intermediate];
    let mut up = vec![0.0_f32; intermediate];
    let mut ffn_out = vec![0.0_f32; hidden];

    let mut patterns: Vec<Vec<f32>> = Vec::with_capacity(token_ids.len());

    println!(
        "  {:>5}  {:>3}  {:>8}  {:>10}  {:>10}  {:>10}  {:>10}",
        "tok", "pos", "‖x‖", "‖attn_o‖", "‖resid1‖", "‖ffn_o‖", "‖resid2‖"
    );

    for (pos, &id) in token_ids.iter().enumerate() {
        let t_token = Instant::now();
        let mut x = model.dequantize_row(embed_table, id)?;
        let x_norm_in = l2(&x);

        // === Attention sublayer ===
        rmsnorm(&x, &attn_norm_w, eps, &mut normed);
        linear(&normed, &wq, &mut q);
        linear(&normed, &wk, &mut k);
        linear(&normed, &wv, &mut v);
        rope_heads(&mut q, head_dim, pos, freq_base);
        rope_heads(&mut k, head_dim, pos, freq_base);
        cache.store(pos, &k, &v);
        attention(&q, &cache, pos, n_heads, n_kv_heads, head_dim, &mut attn);
        linear(&attn, &wo, &mut attn_proj);
        add_in_place(&mut x, &attn_proj);
        let resid1 = l2(&x);

        // === FFN sublayer ===
        rmsnorm(&x, &ffn_norm_w, eps, &mut ffn_normed);
        linear(&ffn_normed, &w_gate, &mut gate);
        linear(&ffn_normed, &w_up, &mut up);
        silu_in_place(&mut gate);
        mul_in_place(&mut gate, &up);
        linear(&gate, &w_down, &mut ffn_out);
        add_in_place(&mut x, &ffn_out);
        let resid2 = l2(&x);

        let pattern = attention_pattern(&q, &cache, pos, n_heads, n_kv_heads, head_dim);
        patterns.push(pattern);

        println!(
            "  {:>5}  {:>3}  {:>8.4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10.4}  [{:.2?}]",
            id, pos, x_norm_in, l2(&attn_proj), resid1, l2(&ffn_out), resid2,
            t_token.elapsed()
        );
    }

    println!();
    println!("attention pattern (avg over {} heads, row = querying token, col = past pos):", n_heads);
    print!("  {:>10}", "");
    for (j, &id) in token_ids.iter().enumerate() {
        print!("  {:>6}", format!("{}@{}", id, j));
    }
    println!();
    for (i, p) in patterns.iter().enumerate() {
        print!("  {:>6}@{:<2}", token_ids[i], i);
        for j in 0..token_ids.len() {
            if j <= i {
                print!("  {:>6.3}", p[j]);
            } else {
                print!("  {:>6}", "·");
            }
        }
        println!();
    }

    Ok(())
}

fn meta_f32(model: &Model, key: &str) -> Result<f32> {
    model
        .metadata
        .get(key)
        .and_then(Value::as_f32)
        .ok_or_else(|| anyhow!("missing or wrong-typed metadata {:?}", key))
}

fn meta_u32(model: &Model, key: &str) -> Result<u32> {
    model
        .metadata
        .get(key)
        .and_then(Value::as_u32)
        .ok_or_else(|| anyhow!("missing or wrong-typed metadata {:?}", key))
}

fn l2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}
