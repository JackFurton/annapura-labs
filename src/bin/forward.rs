use std::env;

use anyhow::{anyhow, bail, Result};

use annapura::gguf::{Model, Value};
use annapura::nn::rmsnorm;

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
        .ok_or_else(|| anyhow!("missing layer_norm_rms_epsilon in model metadata"))?;

    let embed_table = model
        .tensor("token_embd.weight")
        .ok_or_else(|| anyhow!("model has no token_embd.weight"))?;
    let attn_norm = model
        .tensor("blk.0.attn_norm.weight")
        .ok_or_else(|| anyhow!("model has no blk.0.attn_norm.weight"))?;

    let attn_norm_w = model.dequantize(attn_norm)?;

    println!("RMSNorm ε = {}", eps);
    println!("hidden    = {}", attn_norm_w.len());
    println!();
    println!(
        "  {:>5}  {:>10}  {:>10}  {:>10}  first 4 normed",
        "tok", "‖x‖₂", "rms(x)", "‖y‖₂"
    );

    for id in &token_ids {
        let x = model.dequantize_row(embed_table, *id)?;
        let mut y = vec![0.0_f32; x.len()];
        rmsnorm(&x, &attn_norm_w, eps, &mut y);

        let xn = l2(&x);
        let xrms = rms(&x);
        let yn = l2(&y);

        print!("  {:>5}  {:>10.4}  {:>10.6}  {:>10.4}  [", id, xn, xrms, yn);
        for (i, v) in y.iter().take(4).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:+.4}", v);
        }
        println!("]");
    }

    Ok(())
}

fn l2(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum::<f32>().sqrt()
}

fn rms(x: &[f32]) -> f32 {
    (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt()
}
