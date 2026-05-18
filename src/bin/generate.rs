use std::env;
use std::io::Write;
use std::time::Instant;

use anyhow::{anyhow, bail, Result};

use annapura::attention::KvCache;
use annapura::gguf::Model;
use annapura::nn::{linear_q8_par, rmsnorm};
use annapura::tokenizer::TokenDecoder;
use annapura::transformer::{forward_layer, Config, LayerWeights, Scratch};

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        bail!("usage: generate [--n N] <token_id> [<token_id> ...]");
    }

    // Tiny manual flag parser: --n N consumes two args, anything else is a token id.
    let mut n_generate: usize = 50;
    let mut token_strs: Vec<&String> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        if argv[i] == "--n" {
            n_generate = argv
                .get(i + 1)
                .and_then(|s| s.parse().ok())
                .ok_or_else(|| anyhow!("--n needs an integer argument"))?;
            i += 2;
        } else {
            token_strs.push(&argv[i]);
            i += 1;
        }
    }
    let token_ids: Vec<usize> = token_strs
        .iter()
        .map(|s| s.parse::<usize>().map_err(|e| anyhow!("bad token id {:?}: {}", s, e)))
        .collect::<Result<_>>()?;
    if token_ids.is_empty() {
        bail!("at least one input token id required");
    }

    let model = Model::load(DEFAULT_PATH)?;
    let cfg = Config::from_model(&model)?;
    let tokenizer = TokenDecoder::from_model(&model)?;
    assert_eq!(tokenizer.vocab_size(), cfg.vocab, "vocab size mismatch");

    eprintln!(
        "config: n_layers={}, hidden={}, vocab={}, max_seq_len={}, eos={}",
        cfg.n_layers, cfg.hidden, cfg.vocab, cfg.max_seq_len, cfg.eos_token_id
    );

    eprint!("borrowing all {} layers... ", cfg.n_layers);
    let t_load = Instant::now();
    let layers: Vec<LayerWeights<'_>> = (0..cfg.n_layers)
        .map(|l| LayerWeights::load(&model, l))
        .collect::<Result<_>>()?;
    let layer_mb: f64 = layers.iter().map(layer_size_mb).sum();
    eprintln!("{:?} (~{:.1} MB total, mostly mmap'd Q8_0)", t_load.elapsed(), layer_mb);

    let output_norm_w = model.dequantize(model.tensor("output_norm.weight").unwrap())?;
    let output_w_bytes = model.tensor_bytes(model.tensor("output.weight").unwrap());

    let embed_table = model.tensor("token_embd.weight").unwrap();
    let mut caches: Vec<KvCache> = (0..cfg.n_layers)
        .map(|_| KvCache::new(cfg.max_seq_len, cfg.kv_dim))
        .collect();
    let mut scratch = Scratch::new(&cfg);
    let mut final_normed = vec![0.0_f32; cfg.hidden];
    let mut logits = vec![0.0_f32; cfg.vocab];

    // ===== Prefill =====
    eprintln!("\nprefill ({} tokens × {} layers)...", token_ids.len(), cfg.n_layers);
    let t_prefill = Instant::now();
    let mut x: Vec<f32> = Vec::new();
    for (pos, &id) in token_ids.iter().enumerate() {
        x = model.dequantize_row(embed_table, id)?;
        for layer_idx in 0..cfg.n_layers {
            forward_layer(&mut x, &layers[layer_idx], &mut caches[layer_idx], &cfg, &mut scratch, pos);
        }
    }
    eprintln!("prefill: {:?} ({:.1} tok/s)",
              t_prefill.elapsed(),
              token_ids.len() as f64 / t_prefill.elapsed().as_secs_f64());

    // Output head on the last prefilled token's hidden state.
    rmsnorm(&x, &output_norm_w, cfg.eps, &mut final_normed);
    linear_q8_par(&final_normed, output_w_bytes, &mut logits);
    let mut next_token = argmax(&logits);

    // ===== Generation loop =====
    eprintln!("\ngenerating up to {} tokens (stops on EOS={}):", n_generate, cfg.eos_token_id);
    eprintln!("(input echo, then streaming output)");

    // Echo the input prompt (decoded) so the streamed output flows naturally.
    let mut echo_bytes: Vec<u8> = Vec::new();
    for &id in &token_ids {
        echo_bytes.extend_from_slice(&tokenizer.decode_one_bytes(id));
    }
    print!("{}", String::from_utf8_lossy(&echo_bytes));
    std::io::stdout().flush()?;

    let t_gen = Instant::now();
    let mut current_pos = token_ids.len();
    let mut produced = 0usize;
    let mut hit_eos = false;
    let mut stream_buf: Vec<u8> = Vec::new();

    for _ in 0..n_generate {
        if next_token == cfg.eos_token_id {
            hit_eos = true;
            break;
        }
        if current_pos >= cfg.max_seq_len {
            eprintln!("\n[context window exhausted at {}]", current_pos);
            break;
        }

        // Emit the predicted token. Buffer bytes across tokens so multi-byte
        // unicode (split across byte-fallback tokens) decodes cleanly.
        stream_buf.extend_from_slice(&tokenizer.decode_one_bytes(next_token));
        let (valid, _) = utf8_split(&stream_buf);
        if !valid.is_empty() {
            print!("{}", std::str::from_utf8(valid).unwrap_or(""));
            std::io::stdout().flush()?;
            stream_buf.drain(..valid.len());
        }

        // Forward this token through all 22 layers at the next position.
        let mut tok_x = model.dequantize_row(embed_table, next_token)?;
        for layer_idx in 0..cfg.n_layers {
            forward_layer(&mut tok_x, &layers[layer_idx], &mut caches[layer_idx], &cfg, &mut scratch, current_pos);
        }
        current_pos += 1;
        produced += 1;

        // Predict the next next-token.
        rmsnorm(&tok_x, &output_norm_w, cfg.eps, &mut final_normed);
        linear_q8_par(&final_normed, output_w_bytes, &mut logits);
        next_token = argmax(&logits);
    }
    // Flush any remaining partial-UTF8 bytes lossily.
    if !stream_buf.is_empty() {
        print!("{}", String::from_utf8_lossy(&stream_buf));
    }
    println!();

    let gen_time = t_gen.elapsed();
    eprintln!(
        "\ngenerated {} tokens in {:?} ({:.1} tok/s){}",
        produced,
        gen_time,
        produced as f64 / gen_time.as_secs_f64().max(1e-9),
        if hit_eos { "  [stopped on EOS]" } else { "" },
    );

    Ok(())
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) })
        .0
}

/// Returns the longest valid-UTF8 prefix of `buf` (and the remainder length).
/// Used to safely flush stream bytes without breaking multi-byte sequences.
fn utf8_split(buf: &[u8]) -> (&[u8], usize) {
    match std::str::from_utf8(buf) {
        Ok(_) => (buf, 0),
        Err(e) => {
            let valid_up_to = e.valid_up_to();
            (&buf[..valid_up_to], buf.len() - valid_up_to)
        }
    }
}

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
