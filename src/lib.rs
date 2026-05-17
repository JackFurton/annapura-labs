//! Annapura: from-scratch ML systems / accelerator co-design hack project.

pub mod attention;
pub mod gguf;
pub mod matmul;
pub mod nn;
pub mod quant;
pub mod transformer;

pub use matmul::{matmul_blocked, matmul_ikj, matmul_naive};
