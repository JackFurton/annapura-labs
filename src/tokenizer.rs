//! Minimal decode-only tokenizer using the vocab embedded in GGUF metadata.
//!
//! Llama uses SentencePiece with two quirks we need to handle when turning
//! token IDs back into bytes:
//!   - The character `▁` (U+2581, "Lower One Eighth Block") represents a
//!     leading space. SentencePiece prepends one to most "real" tokens so
//!     the boundary between words is encoded as part of the next word.
//!   - Bytes that have no clean unicode representation are stored as
//!     `<0xNN>` byte-fallback tokens. To recover real UTF-8 across emoji
//!     and other multi-byte unicode, we emit raw bytes and let the caller
//!     decode at the end of a streaming run.
//!
//! Encoding (text → token IDs) is a separate problem (SentencePiece BPE
//! with merges); we punt on it here. The model's input still comes from
//! integer token IDs supplied directly.

use anyhow::{anyhow, bail, Result};

use crate::gguf::{Model, Value};

pub struct TokenDecoder {
    vocab: Vec<String>,
}

impl TokenDecoder {
    pub fn from_model(model: &Model) -> Result<Self> {
        let tokens = model
            .metadata
            .get("tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("missing tokenizer.ggml.tokens in GGUF metadata"))?;
        let arr = match tokens {
            Value::Array(arr) => arr,
            _ => bail!("tokenizer.ggml.tokens is not an array"),
        };
        let vocab: Vec<String> = arr
            .iter()
            .map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                other => Err(anyhow!("vocab entry not a string: {:?}", other)),
            })
            .collect::<Result<_>>()?;
        Ok(Self { vocab })
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Decode a single token ID into raw bytes (UTF-8 fragment).
    ///
    /// Returning bytes (not String) is important: byte-fallback tokens
    /// `<0xNN>` represent one byte each, and multi-byte unicode chars (e.g.
    /// emoji) span several such tokens. Concatenating bytes and decoding
    /// at the end avoids "invalid UTF-8" failures mid-stream.
    pub fn decode_one_bytes(&self, token_id: usize) -> Vec<u8> {
        if token_id >= self.vocab.len() {
            return Vec::new();
        }
        let raw = &self.vocab[token_id];

        // Byte-fallback token "<0xNN>" → that single byte.
        if raw.len() == 6 && raw.starts_with("<0x") && raw.ends_with('>') {
            if let Ok(b) = u8::from_str_radix(&raw[3..5], 16) {
                return vec![b];
            }
        }

        // Sentinel `▁` (U+2581) → ASCII space. Just substitute, the rest is UTF-8 already.
        raw.replace('\u{2581}', " ").into_bytes()
    }

    /// Best-effort string view of a single token (lossy on partial unicode).
    /// Useful for diagnostic printing where slight corruption is acceptable.
    pub fn decode_one_lossy(&self, token_id: usize) -> String {
        String::from_utf8_lossy(&self.decode_one_bytes(token_id)).into_owned()
    }

    /// Greedy longest-prefix-match encoder. NOT proper SentencePiece BPE
    /// (which uses learned merges/scores), but good enough for ASCII demos.
    /// Caller is responsible for prepending BOS if the model expects it.
    ///
    /// Spaces in the input are converted to `▁` per the SentencePiece
    /// convention. Falls back to `<0xNN>` byte tokens when no vocab match.
    pub fn encode_greedy(&self, text: &str) -> Vec<usize> {
        let normalized = text.replace(' ', "\u{2581}");
        let bytes = normalized.as_bytes();
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < bytes.len() {
            let remaining = &bytes[i..];
            let mut best: Option<(usize, usize)> = None;
            for (id, v) in self.vocab.iter().enumerate() {
                let vb = v.as_bytes();
                if !vb.is_empty() && remaining.starts_with(vb) {
                    if best.map_or(true, |(_, l)| vb.len() > l) {
                        best = Some((id, vb.len()));
                    }
                }
            }
            match best {
                Some((id, len)) => {
                    tokens.push(id);
                    i += len;
                }
                None => {
                    let bt = format!("<0x{:02X}>", bytes[i]);
                    if let Some(id) = self.vocab.iter().position(|v| v == &bt) {
                        tokens.push(id);
                    }
                    i += 1;
                }
            }
        }
        tokens
    }
}
