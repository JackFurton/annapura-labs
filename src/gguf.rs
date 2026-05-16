//! GGUF v3 reader.
//!
//! Format (little-endian throughout):
//!   header       : magic="GGUF" | version: u32 | n_tensors: u64 | n_metadata: u64
//!   metadata     : n_metadata × { key: string, value_type: u32, value: typed }
//!   tensor_info  : n_tensors  × { name: string, n_dims: u32, dims: [u64; n_dims], type: u32, offset: u64 }
//!   <pad>        : align to general.alignment (default 32) measured from file start
//!   tensor_data  : raw bytes; tensor i lives at tensor_data_start + tensor[i].offset
//!
//! Strings are { u64 length, UTF-8 bytes } — no null terminator.
//! Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use memmap2::Mmap;

const MAGIC: &[u8; 4] = b"GGUF";
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16),
    U32(u32), I32(i32), U64(u64), I64(i64),
    F32(f32), F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Value::U32(v) => Some(*v),
            Value::U64(v) => u32::try_from(*v).ok(),
            Value::I32(v) => u32::try_from(*v).ok(),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Value::String(s) = self { Some(s.as_str()) } else { None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorType {
    F32, F16,
    Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
    Q2K, Q3K, Q4K, Q5K, Q6K, Q8K,
}

impl TensorType {
    fn from_u32(v: u32) -> Result<Self> {
        use TensorType::*;
        Ok(match v {
            0  => F32,  1 => F16,
            2  => Q4_0, 3 => Q4_1,
            6  => Q5_0, 7 => Q5_1,
            8  => Q8_0, 9 => Q8_1,
            10 => Q2K, 11 => Q3K, 12 => Q4K, 13 => Q5K, 14 => Q6K, 15 => Q8K,
            _  => bail!("unknown tensor dtype {}", v),
        })
    }

    pub fn block_elements(self) -> usize {
        match self {
            Self::F32 | Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1
            | Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2K | Self::Q3K | Self::Q4K
            | Self::Q5K | Self::Q6K | Self::Q8K => 256,
        }
    }

    pub fn block_bytes(self) -> usize {
        match self {
            Self::F32 => 4, Self::F16 => 2,
            Self::Q4_0 => 18, Self::Q4_1 => 20,
            Self::Q5_0 => 22, Self::Q5_1 => 24,
            Self::Q8_0 => 34, Self::Q8_1 => 36,
            Self::Q2K => 84, Self::Q3K => 110, Self::Q4K => 144,
            Self::Q5K => 176, Self::Q6K => 210, Self::Q8K => 292,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorType,
    /// Byte offset from start of tensor data block, NOT from start of file.
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    pub fn byte_size(&self) -> u64 {
        let n_blocks = self.n_elements() / self.dtype.block_elements() as u64;
        n_blocks * self.dtype.block_bytes() as u64
    }
}

pub struct Model {
    pub mmap: Mmap,
    pub version: u32,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    /// Absolute byte offset to the tensor data block within the mmap'd file.
    pub tensor_data_start: u64,
}

impl Model {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let mmap = unsafe { Mmap::map(&file) }.context("mmap")?;
        parse(mmap)
    }

    pub fn tensor_bytes(&self, t: &TensorInfo) -> &[u8] {
        let start = self.tensor_data_start as usize + t.offset as usize;
        let end = start + t.byte_size() as usize;
        &self.mmap[start..end]
    }

    pub fn arch(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(Value::as_str)
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Read a tensor's bytes and dequantize to a fresh `Vec<f32>`.
    pub fn dequantize(&self, t: &TensorInfo) -> Result<Vec<f32>> {
        let mut buf = vec![0.0_f32; t.n_elements() as usize];
        crate::quant::dequantize_to_f32(t.dtype, self.tensor_bytes(t), &mut buf)?;
        Ok(buf)
    }
}

fn parse(mmap: Mmap) -> Result<Model> {
    let mut c = Cursor::new(&mmap[..]);

    let magic = c.read_bytes(4)?;
    if magic != MAGIC {
        bail!("not a GGUF file: magic = {:?}", magic);
    }
    let version = c.read_u32()?;
    if version != 3 {
        bail!("unsupported GGUF version {} (this reader only handles v3)", version);
    }
    let n_tensors = c.read_u64()? as usize;
    let n_metadata = c.read_u64()? as usize;

    let mut metadata = BTreeMap::new();
    for _ in 0..n_metadata {
        let key = c.read_string()?;
        let value = c.read_value()?;
        metadata.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(n_tensors);
    for _ in 0..n_tensors {
        let name = c.read_string()?;
        let n_dims = c.read_u32()? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(c.read_u64()?);
        }
        let dtype = TensorType::from_u32(c.read_u32()?)?;
        let offset = c.read_u64()?;
        tensors.push(TensorInfo { name, shape, dtype, offset });
    }

    let alignment = metadata
        .get("general.alignment")
        .and_then(Value::as_u32)
        .map(u64::from)
        .unwrap_or(DEFAULT_ALIGNMENT);
    let pos = c.position() as u64;
    let tensor_data_start = pos.div_ceil(alignment) * alignment;

    Ok(Model { mmap, version, metadata, tensors, tensor_data_start })
}

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }
    fn position(&self) -> usize { self.pos }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.pos + n > self.data.len() {
            bail!("truncated: need {} bytes at offset {} (file is {})",
                  n, self.pos, self.data.len());
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_u8(&mut self)  -> Result<u8>  { Ok(self.read_bytes(1)?[0]) }
    fn read_i8(&mut self)  -> Result<i8>  { Ok(self.read_u8()? as i8) }
    fn read_u16(&mut self) -> Result<u16> { Ok(u16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap())) }
    fn read_i16(&mut self) -> Result<i16> { Ok(i16::from_le_bytes(self.read_bytes(2)?.try_into().unwrap())) }
    fn read_u32(&mut self) -> Result<u32> { Ok(u32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap())) }
    fn read_i32(&mut self) -> Result<i32> { Ok(i32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap())) }
    fn read_u64(&mut self) -> Result<u64> { Ok(u64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap())) }
    fn read_i64(&mut self) -> Result<i64> { Ok(i64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap())) }
    fn read_f32(&mut self) -> Result<f32> { Ok(f32::from_le_bytes(self.read_bytes(4)?.try_into().unwrap())) }
    fn read_f64(&mut self) -> Result<f64> { Ok(f64::from_le_bytes(self.read_bytes(8)?.try_into().unwrap())) }
    fn read_bool(&mut self) -> Result<bool> { Ok(self.read_u8()? != 0) }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_bytes(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| anyhow!("utf8 in string: {}", e))
    }

    fn read_value(&mut self) -> Result<Value> {
        let t = self.read_u32()?;
        self.read_value_typed(t)
    }

    fn read_value_typed(&mut self, t: u32) -> Result<Value> {
        Ok(match t {
            0  => Value::U8(self.read_u8()?),
            1  => Value::I8(self.read_i8()?),
            2  => Value::U16(self.read_u16()?),
            3  => Value::I16(self.read_i16()?),
            4  => Value::U32(self.read_u32()?),
            5  => Value::I32(self.read_i32()?),
            6  => Value::F32(self.read_f32()?),
            7  => Value::Bool(self.read_bool()?),
            8  => Value::String(self.read_string()?),
            9  => {
                let elem_t = self.read_u32()?;
                let n = self.read_u64()? as usize;
                let mut arr = Vec::with_capacity(n);
                for _ in 0..n { arr.push(self.read_value_typed(elem_t)?); }
                Value::Array(arr)
            }
            10 => Value::U64(self.read_u64()?),
            11 => Value::I64(self.read_i64()?),
            12 => Value::F64(self.read_f64()?),
            _  => bail!("unknown value type {}", t),
        })
    }
}
