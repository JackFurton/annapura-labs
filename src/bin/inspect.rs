use std::env;
use std::path::PathBuf;

use anyhow::Result;

use annapura::gguf::{Model, Value};

fn main() -> Result<()> {
    let path: PathBuf = env::args()
        .nth(1)
        .unwrap_or_else(|| "models/tinyllama-1.1b-chat-q8_0.gguf".into())
        .into();

    let model = Model::load(&path)?;

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

    Ok(())
}

fn render(v: &Value) -> String {
    match v {
        Value::U8(x)  => x.to_string(),
        Value::I8(x)  => x.to_string(),
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
