//! Dequantization of GGUF tensor formats to f32.
//!
//! Each quantized format packs many low-precision values per "block" and shares
//! one (or two) high-precision scale(s) across the block. To get back to f32 we
//! reverse that packing.

use anyhow::{bail, Result};
use half::f16;

use crate::gguf::TensorType;

/// Dequantize a tensor's raw bytes into a caller-supplied f32 buffer.
pub fn dequantize_to_f32(dtype: TensorType, src: &[u8], dst: &mut [f32]) -> Result<()> {
    let n = dst.len();
    match dtype {
        TensorType::F32 => {
            assert_eq!(src.len(), n * 4);
            for i in 0..n {
                dst[i] = f32::from_le_bytes([
                    src[4 * i], src[4 * i + 1], src[4 * i + 2], src[4 * i + 3],
                ]);
            }
        }
        TensorType::F16 => {
            assert_eq!(src.len(), n * 2);
            for i in 0..n {
                dst[i] = f16::from_le_bytes([src[2 * i], src[2 * i + 1]]).to_f32();
            }
        }
        TensorType::Q8_0 => dequantize_q8_0(src, n, dst),
        other => bail!("dequantize for {:?} not implemented yet (chapter 2 will add Q4/K-quants)", other),
    }
    Ok(())
}

/// Q8_0 layout: each 32-element block is { f16 scale, [i8; 32] quants }, 34 bytes total.
/// Dequant: `value = scale * i8_quant`.
fn dequantize_q8_0(src: &[u8], n: usize, dst: &mut [f32]) {
    assert_eq!(n % 32, 0, "Q8_0 quantizes in blocks of 32 elements");
    assert_eq!(src.len(), n / 32 * 34);

    for (block_idx, block) in src.chunks_exact(34).enumerate() {
        let scale = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let out = &mut dst[block_idx * 32..block_idx * 32 + 32];
        for i in 0..32 {
            let q = block[2 + i] as i8;
            out[i] = scale * q as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_known_block_dequantizes_to_expected_values() {
        // One block: scale = 2.0 (f16), quants = [-128, -64, 0, 64, 127, 0×27].
        let mut src = Vec::with_capacity(34);
        src.extend_from_slice(&f16::from_f32(2.0).to_bits().to_le_bytes());
        let mut quants = [0i8; 32];
        quants[0] = -128;
        quants[1] = -64;
        quants[2] = 0;
        quants[3] = 64;
        quants[4] = 127;
        for q in &quants {
            src.push(*q as u8);
        }

        let mut dst = vec![0.0_f32; 32];
        dequantize_q8_0(&src, 32, &mut dst);

        assert_eq!(dst[0], -256.0);
        assert_eq!(dst[1], -128.0);
        assert_eq!(dst[2], 0.0);
        assert_eq!(dst[3], 128.0);
        assert_eq!(dst[4], 254.0);
        for i in 5..32 {
            assert_eq!(dst[i], 0.0);
        }
    }

    #[test]
    fn q8_0_two_blocks_use_independent_scales() {
        let mut src = Vec::with_capacity(68);
        // block 0: scale=1.0, quants[0]=10
        src.extend_from_slice(&f16::from_f32(1.0).to_bits().to_le_bytes());
        src.push(10);
        src.extend_from_slice(&[0u8; 31]);
        // block 1: scale=0.5, quants[0]=10
        src.extend_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        src.push(10);
        src.extend_from_slice(&[0u8; 31]);

        let mut dst = vec![0.0_f32; 64];
        dequantize_q8_0(&src, 64, &mut dst);
        assert_eq!(dst[0], 10.0);
        assert_eq!(dst[32], 5.0);
    }

    #[test]
    fn f32_passthrough_is_identity() {
        let src: Vec<u8> = [1.5f32, -2.25, 3.14]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        let mut dst = vec![0.0_f32; 3];
        dequantize_to_f32(TensorType::F32, &src, &mut dst).unwrap();
        assert_eq!(dst, vec![1.5, -2.25, 3.14]);
    }
}
