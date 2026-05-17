//! Llama transformer block forward pass.
//!
//! Each block is: RMSNorm → Q/K/V projection → RoPE → cached attention →
//! Wo projection → residual; then RMSNorm → SwiGLU FFN → residual.

use anyhow::{anyhow, Result};

use crate::attention::{attention, KvCache};
use crate::gguf::{Model, Value};
use crate::nn::{add_in_place, linear_q8_par, mul_in_place, rmsnorm, rope_heads, silu_in_place};

pub struct Config {
    pub eps: f32,
    pub freq_base: f32,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub n_layers: usize,
    pub hidden: usize,
    pub kv_dim: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub max_seq_len: usize,
}

impl Config {
    pub fn from_model(model: &Model) -> Result<Self> {
        let eps = meta_f32(model, "llama.attention.layer_norm_rms_epsilon")?;
        let freq_base = model
            .metadata
            .get("llama.rope.freq_base")
            .and_then(Value::as_f32)
            .unwrap_or(10_000.0);
        let head_dim = meta_u32(model, "llama.rope.dimension_count")? as usize;
        let n_heads = meta_u32(model, "llama.attention.head_count")? as usize;
        let n_kv_heads = meta_u32(model, "llama.attention.head_count_kv")? as usize;
        let n_layers = meta_u32(model, "llama.block_count")? as usize;
        let max_seq_len = meta_u32(model, "llama.context_length")? as usize;
        let intermediate = meta_u32(model, "llama.feed_forward_length")? as usize;
        let hidden = n_heads * head_dim;
        let hidden_meta = meta_u32(model, "llama.embedding_length")? as usize;
        assert_eq!(hidden, hidden_meta, "hidden mismatch");
        let kv_dim = n_kv_heads * head_dim;
        let vocab = model
            .tensor("token_embd.weight")
            .ok_or_else(|| anyhow!("no token_embd.weight"))?
            .shape[1] as usize;
        Ok(Self {
            eps, freq_base, head_dim, n_heads, n_kv_heads, n_layers,
            hidden, kv_dim, intermediate, vocab, max_seq_len,
        })
    }
}

/// Per-layer weights. Norms are F32 in GGUF and small (2048 floats each) so
/// we just dequantize them once. The 7 Q8_0 linear weights are kept as raw
/// packed-byte views into the model's mmap — they're consumed directly by
/// `linear_q8_par`, with no f32 materialization step.
pub struct LayerWeights<'a> {
    pub attn_norm: Vec<f32>,
    pub ffn_norm: Vec<f32>,
    pub wq: &'a [u8],
    pub wk: &'a [u8],
    pub wv: &'a [u8],
    pub wo: &'a [u8],
    pub w_gate: &'a [u8],
    pub w_up: &'a [u8],
    pub w_down: &'a [u8],
}

impl<'a> LayerWeights<'a> {
    pub fn load(model: &'a Model, layer_idx: usize) -> Result<Self> {
        let bytes = |name: &str| -> Result<&'a [u8]> {
            let full = format!("blk.{}.{}", layer_idx, name);
            let tensor = model
                .tensor(&full)
                .ok_or_else(|| anyhow!("missing tensor {:?}", full))?;
            Ok(model.tensor_bytes(tensor))
        };
        let dequant_f32 = |name: &str| -> Result<Vec<f32>> {
            let full = format!("blk.{}.{}", layer_idx, name);
            let tensor = model
                .tensor(&full)
                .ok_or_else(|| anyhow!("missing tensor {:?}", full))?;
            model.dequantize(tensor)
        };
        Ok(Self {
            attn_norm: dequant_f32("attn_norm.weight")?,
            ffn_norm: dequant_f32("ffn_norm.weight")?,
            wq: bytes("attn_q.weight")?,
            wk: bytes("attn_k.weight")?,
            wv: bytes("attn_v.weight")?,
            wo: bytes("attn_output.weight")?,
            w_gate: bytes("ffn_gate.weight")?,
            w_up: bytes("ffn_up.weight")?,
            w_down: bytes("ffn_down.weight")?,
        })
    }
}

pub struct Scratch {
    pub normed: Vec<f32>,
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn: Vec<f32>,
    pub attn_proj: Vec<f32>,
    pub ffn_normed: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub ffn_out: Vec<f32>,
}

impl Scratch {
    pub fn new(cfg: &Config) -> Self {
        Self {
            normed: vec![0.0; cfg.hidden],
            q: vec![0.0; cfg.hidden],
            k: vec![0.0; cfg.kv_dim],
            v: vec![0.0; cfg.kv_dim],
            attn: vec![0.0; cfg.hidden],
            attn_proj: vec![0.0; cfg.hidden],
            ffn_normed: vec![0.0; cfg.hidden],
            gate: vec![0.0; cfg.intermediate],
            up: vec![0.0; cfg.intermediate],
            ffn_out: vec![0.0; cfg.hidden],
        }
    }
}

/// One full transformer block applied to `x` in place. The token's K and V
/// are stored in `cache` at `pos`. Reads architecture from `cfg`, reuses
/// `scratch` to avoid per-call allocations.
pub fn forward_layer(
    x: &mut [f32],
    layer: &LayerWeights<'_>,
    cache: &mut KvCache,
    cfg: &Config,
    scratch: &mut Scratch,
    pos: usize,
) {
    rmsnorm(x, &layer.attn_norm, cfg.eps, &mut scratch.normed);
    linear_q8_par(&scratch.normed, layer.wq, &mut scratch.q);
    linear_q8_par(&scratch.normed, layer.wk, &mut scratch.k);
    linear_q8_par(&scratch.normed, layer.wv, &mut scratch.v);
    rope_heads(&mut scratch.q, cfg.head_dim, pos, cfg.freq_base);
    rope_heads(&mut scratch.k, cfg.head_dim, pos, cfg.freq_base);
    cache.store(pos, &scratch.k, &scratch.v);
    attention(
        &scratch.q, cache, pos,
        cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
        &mut scratch.attn,
    );
    linear_q8_par(&scratch.attn, layer.wo, &mut scratch.attn_proj);
    add_in_place(x, &scratch.attn_proj);

    rmsnorm(x, &layer.ffn_norm, cfg.eps, &mut scratch.ffn_normed);
    linear_q8_par(&scratch.ffn_normed, layer.w_gate, &mut scratch.gate);
    linear_q8_par(&scratch.ffn_normed, layer.w_up, &mut scratch.up);
    silu_in_place(&mut scratch.gate);
    mul_in_place(&mut scratch.gate, &scratch.up);
    linear_q8_par(&scratch.gate, layer.w_down, &mut scratch.ffn_out);
    add_in_place(x, &scratch.ffn_out);
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
