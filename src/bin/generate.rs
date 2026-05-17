use std::env;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use annapura::attention::KvCache;
use annapura::gguf::Model;
use annapura::nn::{linear_q8_par, rmsnorm};
use annapura::transformer::{forward_layer, Config, LayerWeights, Scratch};

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        bail!("usage: generate <token_id> [<token_id> ...]");
    }
    let token_ids: Vec<usize> = argv
        .iter()
        .map(|s| s.parse::<usize>().map_err(|e| anyhow!("bad token id {:?}: {}", s, e)))
        .collect::<Result<_>>()?;

    let model = Model::load(DEFAULT_PATH)?;
    let cfg = Config::from_model(&model)?;

    eprintln!(
        "config: n_layers={}, hidden={}, intermediate={}, vocab={}, max_seq_len={}",
        cfg.n_layers, cfg.hidden, cfg.intermediate, cfg.vocab, cfg.max_seq_len
    );

    eprint!("borrowing all {} layers' weights... ", cfg.n_layers);
    let t_load = Instant::now();
    let layers: Vec<LayerWeights<'_>> = (0..cfg.n_layers)
        .map(|l| LayerWeights::load(&model, l))
        .collect::<Result<_>>()?;
    let layer_mb: f64 = layers.iter().map(layer_size_mb).sum();
    eprintln!("{:?} (~{:.1} MB total, mostly mmap'd packed bytes)", t_load.elapsed(), layer_mb);

    eprint!("preparing output head... ");
    let t_head = Instant::now();
    let output_norm_w = model.dequantize(model.tensor("output_norm.weight").unwrap())?;
    let output_w_bytes = model.tensor_bytes(model.tensor("output.weight").unwrap());
    eprintln!(
        "{:?} (output_norm {:.1} MB f32, output.weight {:.1} MB packed Q8_0)",
        t_head.elapsed(),
        output_norm_w.len() as f64 * 4.0 / 1e6,
        output_w_bytes.len() as f64 / 1e6,
    );

    let embed_table = model.tensor("token_embd.weight").unwrap();
    let mut caches: Vec<KvCache> = (0..cfg.n_layers)
        .map(|_| KvCache::new(cfg.max_seq_len, cfg.kv_dim))
        .collect();
    let mut scratch = Scratch::new(&cfg);
    let mut final_normed = vec![0.0_f32; cfg.hidden];
    let mut logits = vec![0.0_f32; cfg.vocab];

    eprintln!("\nprefill ({} tokens × {} layers)...", token_ids.len(), cfg.n_layers);
    let t_prefill = Instant::now();
    let mut x: Vec<f32> = Vec::new();
    for (pos, &id) in token_ids.iter().enumerate() {
        let t_tok = Instant::now();
        x = model.dequantize_row(embed_table, id)?;
        for layer_idx in 0..cfg.n_layers {
            forward_layer(&mut x, &layers[layer_idx], &mut caches[layer_idx], &cfg, &mut scratch, pos);
        }
        eprintln!("  tok {:>5} @ pos {}:  {:?}", id, pos, t_tok.elapsed());
    }
    eprintln!("prefill total: {:?}\n", t_prefill.elapsed());

    let t_head_eval = Instant::now();
    rmsnorm(&x, &output_norm_w, cfg.eps, &mut final_normed);
    linear_q8_par(&final_normed, output_w_bytes, &mut logits);
    eprintln!("output head: {:?}\n", t_head_eval.elapsed());

    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let max_logit = indexed[0].1;
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();

    println!("input sequence:  {:?}", token_ids);
    println!("argmax next token: {} (logit {:.3}, prob {:.2}%)",
             indexed[0].0,
             indexed[0].1,
             ((indexed[0].1 - max_logit).exp() / sum_exp) * 100.0);
    println!();
    println!("top 10 predictions:");
    println!("  {:>5}  {:>8}  {:>9}  {:>8}", "rank", "tok_id", "logit", "prob");
    for rank in 0..10 {
        let (tok, logit) = indexed[rank];
        let prob = (logit - max_logit).exp() / sum_exp;
        println!("  {:>5}  {:>8}  {:>+9.3}  {:>7.2}%", rank + 1, tok, logit, prob * 100.0);
    }

    Ok(())
}

/// Approximate resident size of a layer's weights: norms in f32, linears as
/// the packed-byte view length (not f32-expanded).
fn layer_size_mb(layer: &LayerWeights) -> f64 {
    let norm_bytes = (layer.attn_norm.len() + layer.ffn_norm.len()) * 4;
    let packed_bytes = layer.wq.len()
        + layer.wk.len()
        + layer.wv.len()
        + layer.wo.len()
        + layer.w_gate.len()
        + layer.w_up.len()
        + layer.w_down.len();
    (norm_bytes + packed_bytes) as f64 / 1e6
}
