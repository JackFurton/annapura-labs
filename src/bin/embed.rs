use std::env;

use anyhow::{anyhow, bail, Result};

use annapura::gguf::Model;

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().skip(1).collect();
    if argv.is_empty() {
        bail!("usage: embed <token_id> [<token_id> ...]");
    }

    let token_ids: Vec<usize> = argv
        .iter()
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| anyhow!("could not parse token id {:?}: {}", s, e))
        })
        .collect::<Result<_>>()?;

    let model = Model::load(DEFAULT_PATH)?;
    let table = model
        .tensor("token_embd.weight")
        .ok_or_else(|| anyhow!("model has no token_embd.weight"))?;
    let (hidden, vocab) = match table.shape.as_slice() {
        [h, v] => (*h as usize, *v as usize),
        other => bail!("unexpected embedding tensor shape {:?}", other),
    };

    println!("embedding table: {} tokens × {} hidden ({:?})", vocab, hidden, table.dtype);
    println!();
    println!(
        "  {:>5}  {:>8}  {:>8}  first 6 values",
        "tok", "‖x‖₂", "|max|"
    );
    for id in &token_ids {
        if *id >= vocab {
            bail!("token id {} out of range (vocab = {})", id, vocab);
        }
        let emb = model.dequantize_row(table, *id)?;
        let norm = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let absmax = emb.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);

        print!("  {:>5}  {:>8.4}  {:>8.4}  [", id, norm, absmax);
        for (i, v) in emb.iter().take(6).enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:+.4}", v);
        }
        println!("]");
    }

    Ok(())
}
