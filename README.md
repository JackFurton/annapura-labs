# annapura-labs

A from-scratch ML systems / silicon co-design hack project. The goal is to
emulate, layer by layer, the kind of work done at AWS Annapurna Labs — the
stack between transformer models and the silicon that runs them — building
each piece by hand with no ML framework as a crutch.

This is a personal learning project, not a product. The pacing favors depth
over breadth: every primitive starts as the worst reasonable implementation
(so we understand *why* the production version exists), then gets optimized
in later chapters.

## The year-1 arc

```
Chapter 0  Foundations                 ████████████████████ done
Chapter 1  Correct slow forward pass   ████████████░░░░░░░░ 4/6
Chapter 2  Fast CPU kernels            ░░░░░░░░░░░░░░░░░░░░
Chapter 3  Real attention (Flash, KV)  ░░░░░░░░░░░░░░░░░░░░
Chapter 4  Serving infra               ░░░░░░░░░░░░░░░░░░░░
Chapter 5  Accelerator ISA + sim       ░░░░░░░░░░░░░░░░░░░░
Chapter 6  Cycle-accurate perf model   ░░░░░░░░░░░░░░░░░░░░
```

Chapters 0-4 are ML systems engineering (overlapping with llama.cpp/vLLM
territory). Chapters 5-6 are where the Annapurna part begins — designing a
hypothetical accelerator ISA, writing a microarchitectural simulator, and
showing what speedup it would buy us over the CPU baseline from chapters
1-2.

## What's built so far

| File | What it does |
|---|---|
| `src/gguf.rs` | GGUF v3 binary format reader (mmap'd, no copy) |
| `src/quant.rs` | Dequantization to f32 — F32 / F16 / Q8_0 |
| `src/nn.rs` | Neural net primitives — RMSNorm |
| `src/matmul.rs` | Matrix multiplication kernels — naive scalar baseline |
| `src/bin/inspect.rs` | Model inspection CLI — metadata + tensor dump |
| `src/bin/embed.rs` | Token embedding lookup demo |
| `src/bin/forward.rs` | Partial forward pass: embedding → RMSNorm |
| `benches/matmul.rs` | criterion perf bench |

About 800 lines of Rust, 3 runtime deps (`memmap2`, `anyhow`, `half`),
10 passing unit tests.

## Hardware baseline

Captured at the `v0-baseline` tag — every later optimization is measured
against these numbers:

```
hardware:     Apple M3 Pro (11 cores, 18 GB RAM)
kernel:       naive scalar f32 matmul, single P-core, no SIMD

   64×64  →  3.77 GFLOPS
  128×128 →  3.22 GFLOPS
  256×256 →  2.74 GFLOPS
  512×512 →  2.65 GFLOPS

theoretical single-core peak:  ~128 GFLOPS
fraction of peak achieved:     ~2-3%
```

The drop from 64→512 is cache hierarchy biting — the i,j,k inner loop
strides down columns of B (stride-N), which thrashes L1 once the working
set leaves it. Chapter 2's cache-blocked + SIMD + multi-threaded kernel
will close most of that gap.

## Quickstart

```sh
# Run all tests
cargo test --release

# Run the perf bench
cargo bench --bench matmul

# Inspect the model (architecture + tensor table)
cargo run --release --bin inspect

# Print stats for a specific tensor
cargo run --release --bin inspect -- --values blk.0.attn_q.weight 8

# Look up token embeddings
cargo run --release --bin embed -- 0 1 2 100 1000

# Apply embedding + RMSNorm to a few tokens
cargo run --release --bin forward -- 0 1 2 100 1000
```

## Getting the model

We use TinyLlama-1.1B-Chat-v1.0 in Q8_0 GGUF (1.1 GB):

```sh
mkdir -p models
curl -L -o models/tinyllama-1.1b-chat-q8_0.gguf \
  https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

`models/` is gitignored.

## What we've learned by poking at the model

Some things that fell out of just *looking* at TinyLlama's weights once the
parser was working:

- **Dead tokens.** Token 100 has *exactly* zero embedding — a SentencePiece
  byte-fallback slot that the training corpus never activated. Token 0
  (`<unk>`) is near-zero (‖x‖ = 4e-4) for the same reason.
- **Grouped Query Attention.** Q has shape `[2048, 2048]` but K, V are
  `[2048, 256]` (4 KV heads vs 32 query heads). KV cache will be 8× smaller
  than vanilla MHA.
- **RMSNorm's soft floor.** With ε = 1e-5, inputs with rms below √ε ≈ 0.003
  don't get amplified to unit-rms — they stay quiet. This is by design: a
  truly silent input shouldn't become loud just because the normalizer
  divides by its own magnitude.
- **GGUF reverses PyTorch shapes.** `token_embd.weight` is `[32000, 2048]`
  in PyTorch but `[2048, 32000]` in GGUF — fastest-changing axis first.
  Same bytes, different label order.

## Non-goals (for honest self-discipline)

- **We don't try to beat llama.cpp on absolute perf.** Goal is learning the
  full stack from scratch, not building a competitor.
- **No CUDA / GPU.** CPU + a hypothetical accelerator simulator. GPU is its
  own rabbit hole that would dilute the silicon focus.
- **No ML framework crutches.** No candle, no tch-rs, no ort, no onnxruntime.
  If we want a primitive, we build it.
- **No premature optimization.** Each chapter has explicit perf goals; we
  don't optimize anything until we have a baseline to measure against.

## Requirements

- Rust stable, 1.85+ (edition 2024 transitively required by criterion)
- macOS / Linux (Apple Silicon or x86_64). Untested on Windows.
