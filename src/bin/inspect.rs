use std::env;
use std::path::PathBuf;

use anyhow::{anyhow, Result};

use annapura::gguf::{Model, Value};
use annapura::tokenizer::TokenDecoder;

const DEFAULT_PATH: &str = "models/tinyllama-1.1b-chat-q8_0.gguf";

fn main() -> Result<()> {
    let argv: Vec<String> = env::args().skip(1).collect();

    let mut path: Option<String> = None;
    let mut values: Option<(String, usize)> = None;
    let mut find: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--values" => {
                let name = argv
                    .get(i + 1)
                    .ok_or_else(|| anyhow!("--values needs a tensor name"))?
                    .clone();
                let n_words = argv
                    .get(i + 2)
                    .and_then(|s| s.parse::<usize>().ok());
                values = Some((name, n_words.unwrap_or(16)));
                i += if n_words.is_some() { 3 } else { 2 };
            }
            "--find" => {
                find = Some(
                    argv.get(i + 1)
                        .ok_or_else(|| anyhow!("--find needs text"))?
                        .clone(),
                );
                i += 2;
            }
            other => {
                path = Some(other.into());
                i += 1;
            }
        }
    }

    let path: PathBuf = path.unwrap_or_else(|| DEFAULT_PATH.into()).into();
    let model = Model::load(&path)?;

    if let Some(text) = find {
        dump_find(&model, &text)
    } else if let Some((name, n)) = values {
        dump_values(&model, &name, n)
    } else {
        dump_summary(&path, &model);
        Ok(())
    }
}

fn dump_summary(path: &PathBuf, model: &Model) {
    println!("file:     {}", path.display());
    println!("size:     {:.2} GB", model.mmap.len() as f64 / 1e9);
    println!("version:  v{}", model.version);
    println!("arch:     {}", model.arch().unwrap_or("?"));
    println!("tensors:  {}", model.tensors.len());
    println!("metadata: {} keys", model.metadata.len());
    println!("data @:   byte {}", model.tensor_data_start);

    println!("\n=== architecture metadata ===");
    let keys = [
        "general.name",
        "general.architecture",
        "llama.context_length",
        "llama.embedding_length",
        "llama.block_count",
        "llama.attention.head_count",
        "llama.attention.head_count_kv",
        "llama.feed_forward_length",
        "llama.rope.dimension_count",
        "llama.attention.layer_norm_rms_epsilon",
        "tokenizer.ggml.model",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
    ];
    for key in keys {
        if let Some(v) = model.metadata.get(key) {
            println!("  {:48}  {}", key, render(v));
        }
    }

    println!("\n=== dtype histogram ===");
    let mut counts: std::collections::BTreeMap<_, usize> = Default::default();
    for t in &model.tensors {
        *counts.entry(t.dtype).or_default() += 1;
    }
    for (dtype, n) in &counts {
        println!("  {:?}: {}", dtype, n);
    }

    println!("\n=== first 20 tensors ===");
    println!("  {:42}  {:6}  shape", "name", "dtype");
    for t in model.tensors.iter().take(20) {
        println!("  {:42}  {:?}  {:?}", t.name, t.dtype, t.shape);
    }
    if model.tensors.len() > 20 {
        println!("  ... and {} more", model.tensors.len() - 20);
    }
}

fn dump_values(model: &Model, name: &str, n: usize) -> Result<()> {
    let t = model
        .tensor(name)
        .ok_or_else(|| anyhow!("tensor {:?} not found in model", name))?;
    let data = model.dequantize(t)?;

    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mean = data.iter().sum::<f32>() / data.len() as f32;
    let abs_max = data.iter().copied().map(f32::abs).fold(0.0_f32, f32::max);

    println!("tensor:   {}", t.name);
    println!("dtype:    {:?}", t.dtype);
    println!("shape:    {:?}", t.shape);
    println!("elements: {}", data.len());
    println!("min/max:  {:.6} / {:.6}", min, max);
    println!("mean:     {:.6}", mean);
    println!("|max|:    {:.6}", abs_max);

    println!("\nfirst {} values:", n.min(data.len()));
    for (i, v) in data.iter().take(n).enumerate() {
        println!("  [{:5}]  {:>+12.6}", i, v);
    }
    Ok(())
}

fn dump_find(model: &Model, text: &str) -> Result<()> {
    let decoder = TokenDecoder::from_model(model)?;
    let ids = decoder.encode_greedy(text);
    let bos = model
        .metadata
        .get("tokenizer.ggml.bos_token_id")
        .and_then(Value::as_u32)
        .unwrap_or(1) as usize;

    println!("input:       {:?}", text);
    println!("tokens ({}):", ids.len());
    for id in &ids {
        let decoded = decoder.decode_one_lossy(*id);
        println!("  {:>6}  {:?}", id, decoded);
    }
    println!();
    println!("feed to generate (BOS prepended):");
    print!("  cargo run --release --bin generate -- --n 80 {}", bos);
    for id in &ids {
        print!(" {}", id);
    }
    println!();
    Ok(())
}

fn render(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::I8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::I16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::I32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::I64(x) => x.to_string(),
        Value::F32(x) => format!("{}", x),
        Value::F64(x) => format!("{}", x),
        Value::Bool(x) => x.to_string(),
        Value::String(s) => format!("{:?}", s),
        Value::Array(a) => format!("[array, {} elements]", a.len()),
    }
}
