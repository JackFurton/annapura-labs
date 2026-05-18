//! Annapura: from-scratch ML systems / accelerator co-design hack project.

pub mod accelerator;
pub mod attention;
pub mod compiler;
pub mod gguf;
pub mod matmul;
pub mod nn;
pub mod perf_model;
pub mod quant;
pub mod tokenizer;
pub mod transformer;

pub use matmul::{matmul_blocked, matmul_ikj, matmul_naive};
