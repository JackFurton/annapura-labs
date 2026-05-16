//! Annapura: from-scratch ML systems / accelerator co-design hack project.

pub mod gguf;
pub mod matmul;
pub mod nn;
pub mod quant;

pub use matmul::matmul_naive;
