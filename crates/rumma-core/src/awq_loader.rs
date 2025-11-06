use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde_json::Value;

use crate::quant::{QuantizationConfig, QuantizedLinear};

const AWQ_NIBBLE_MAP: [u8; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

#[derive(Debug, Clone)]
pub struct AwqLayer {
    pub name: String,
    pub linear: QuantizedLinear,
}

#[derive(Debug, Clone)]
pub struct AwqModel {
    layers: Vec<AwqLayer>,
}

impl AwqModel {
    pub fn new(layers: Vec<AwqLayer>) -> Result<Self> {
        if layers.is_empty() {
            bail!("AWQ model did not contain any quantized layers");
        }
        Ok(Self { layers })
    }

    pub fn layers(&self) -> &[AwqLayer] {
        &self.layers
    }

    pub fn into_layers(self) -> Vec<AwqLayer> {
        self.layers
    }

    pub fn hidden_size(&self) -> Option<usize> {
        // For transformer models, we need to find the actual hidden_size by looking at
        // specific layer types. The layers are stored in alphabetical order, so we can't
        // just use the first layer.
        //
        // Strategy: Look for attention q_proj, k_proj, or v_proj layers, which take
        // hidden_size as input. Or look for down_proj layers which output hidden_size.
        for layer in &self.layers {
            if layer.name.contains(".q_proj")
                || layer.name.contains(".k_proj")
                || layer.name.contains(".v_proj") {
                // These layers take hidden_size as input
                return Some(layer.linear.cols());
            }
        }

        // Fallback: look for down_proj which outputs hidden_size
        for layer in &self.layers {
            if layer.name.contains(".down_proj") {
                // down_proj outputs hidden_size
                return Some(layer.linear.rows());
            }
        }

        // Last resort: use the first layer's input dimension
        self.layers.first().map(|layer| layer.linear.cols())
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }

    pub fn into_quantized_layers(self) -> Vec<QuantizedLinear> {
        self.layers.into_iter().map(|layer| layer.linear).collect()
    }
}

#[derive(Default)]
struct LayerEntry {
    qweight: Option<String>,
    qzeros: Option<String>,
    scales: Option<String>,
}

pub fn load_awq_model(path: &Path) -> Result<AwqModel> {
    let bytes =
        fs::read(path).with_context(|| format!("failed to read AWQ checkpoint at {:?}", path))?;
    let (_, metadata) =
        SafeTensors::read_metadata(&bytes).context("failed to parse safetensors header")?;
    let global_metadata = metadata.metadata().clone();
    let tensors = SafeTensors::deserialize(&bytes).context("failed to parse safetensors file")?;

    let mut entries: BTreeMap<String, LayerEntry> = BTreeMap::new();
    for name in tensors.names() {
        if let Some(prefix) = name.strip_suffix(".qweight") {
            entries.entry(prefix.to_string()).or_default().qweight = Some(name.to_string());
        } else if let Some(prefix) = name.strip_suffix(".qzeros") {
            entries.entry(prefix.to_string()).or_default().qzeros = Some(name.to_string());
        } else if let Some(prefix) = name.strip_suffix(".scales") {
            entries.entry(prefix.to_string()).or_default().scales = Some(name.to_string());
        }
    }

    if entries.is_empty() {
        bail!("no AWQ tensors were found in {:?}", path);
    }

    let mut layers = Vec::new();
    for (base, entry) in entries {
        let qweight_name = entry
            .qweight
            .ok_or_else(|| anyhow!("layer {base} is missing qweight tensor"))?;
        let scales_name = entry
            .scales
            .ok_or_else(|| anyhow!("layer {base} is missing scales tensor"))?;

        let qweight_tensor = tensors
            .tensor(&qweight_name)
            .with_context(|| format!("tensor {qweight_name} was missing from file"))?;
        let scales_tensor = tensors
            .tensor(&scales_name)
            .with_context(|| format!("tensor {scales_name} was missing from file"))?;
        let qzeros_tensor = match entry.qzeros {
            Some(name) => Some(
                tensors
                    .tensor(&name)
                    .with_context(|| format!("tensor {name} was missing from file"))?,
            ),
            None => None,
        };

        let quantized = build_layer(
            &base,
            &qweight_tensor,
            qzeros_tensor.as_ref(),
            &scales_tensor,
            global_metadata.as_ref(),
        )?;
        layers.push(AwqLayer {
            name: base,
            linear: quantized,
        });
    }

    AwqModel::new(layers)
}

fn build_layer(
    name: &str,
    qweight: &TensorView<'_>,
    qzeros: Option<&TensorView<'_>>,
    scales: &TensorView<'_>,
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> Result<QuantizedLinear> {
    if qweight.shape().len() != 2 {
        bail!(
            "{name}: expected 2D qweight tensor, got shape {:?}",
            qweight.shape()
        );
    }

    let rows = qweight.shape()[0];
    let packed_cols = qweight.shape()[1];
    if packed_cols == 0 {
        bail!("{name}: qweight tensor has no columns");
    }
    let cols = packed_cols * 8;

    let nibble_map = extract_nibble_map(metadata, name).unwrap_or(AWQ_NIBBLE_MAP);
    let qweight_words = read_packed_words(qweight)?;
    if qweight_words.len() != rows * packed_cols {
        bail!(
            "{name}: qweight tensor had {} elements but {}x{} expected",
            qweight_words.len(),
            rows,
            packed_cols
        );
    }

    let qzeros_words = match qzeros {
        Some(tensor) => Some(read_packed_words(tensor)?),
        None => None,
    };

    let scales_values = read_scales(scales)?;
    if scales.shape().len() != 2 {
        bail!(
            "{name}: expected 2D scales tensor, got shape {:?}",
            scales.shape()
        );
    }
    let scale_rows = scales.shape()[0];
    let scale_cols = scales.shape()[1];
    if scale_cols != cols {
        bail!(
            "{name}: scales tensor width {} does not match unpacked width {}",
            scale_cols,
            cols
        );
    }
    if scale_rows == 0 {
        bail!("{name}: scales tensor has zero rows");
    }

    let group_size = if rows % scale_rows == 0 {
        rows / scale_rows
    } else {
        bail!(
            "{name}: in_features {} is not divisible by scales rows {}",
            rows,
            scale_rows
        );
    };

    if let Some(ref zeros) = qzeros_words {
        if zeros.len() != scale_rows * packed_cols {
            bail!(
                "{name}: qzeros tensor had {} elements but {}x{} expected",
                zeros.len(),
                scale_rows,
                packed_cols
            );
        }
    }

    let mut dense = vec![0f32; rows * cols];

    for row in 0..rows {
        let group = row / group_size;
        let weight_offset = row * packed_cols;
        for col in 0..cols {
            let word_index = col / 8;
            let slot = nibble_map[(col % 8) as usize] as usize;
            let word = qweight_words[weight_offset + word_index];
            let quantized = ((word >> (slot * 4)) & 0xF) as i32;
            let zero = qzeros_words
                .as_ref()
                .map(|zeros| {
                    let zero_word = zeros[group * packed_cols + word_index];
                    ((zero_word >> (slot * 4)) & 0xF) as i32
                })
                .unwrap_or(8);
            let scale = scales_values[group * cols + col];
            dense[row * cols + col] = (quantized - zero) as f32 * scale;
        }
    }

    let mut transposed = vec![0f32; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            transposed[col * rows + row] = dense[row * cols + col];
        }
    }

    let quant_cfg = QuantizationConfig {
        group_size,
        symmetric: false,
        nibble_map,
    };

    QuantizedLinear::from_dense(cols, rows, &transposed, &quant_cfg)
        .with_context(|| format!("failed to quantize layer {name}"))
}

fn read_packed_words(tensor: &TensorView<'_>) -> Result<Vec<u32>> {
    match tensor.dtype() {
        Dtype::I32 => read_words_4(tensor.data(), i32::from_le_bytes)
            .map(|values| values.into_iter().map(|v| v as u32).collect()),
        Dtype::U32 => read_words_4(tensor.data(), u32::from_le_bytes),
        other => bail!("unsupported dtype {other:?} for packed AWQ tensor"),
    }
}

fn read_scales(tensor: &TensorView<'_>) -> Result<Vec<f32>> {
    match tensor.dtype() {
        Dtype::F16 => read_f16_values(tensor.data()),
        Dtype::F32 => read_words_4(tensor.data(), f32::from_le_bytes),
        other => bail!("unsupported dtype {other:?} for AWQ scales"),
    }
}

fn read_words_4<T, F>(data: &[u8], convert: F) -> Result<Vec<T>>
where
    F: Fn([u8; 4]) -> T + Copy,
{
    if data.len() % 4 != 0 {
        bail!(
            "malformed tensor data length {} is not divisible by 4",
            data.len()
        );
    }
    let mut out = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        let array: [u8; 4] = chunk.try_into().expect("chunk size is 4");
        out.push(convert(array));
    }
    Ok(out)
}

fn read_f16_values(data: &[u8]) -> Result<Vec<f32>> {
    if data.len() % 2 != 0 {
        bail!(
            "malformed f16 tensor length {} is not divisible by 2",
            data.len()
        );
    }
    let mut out = Vec::with_capacity(data.len() / 2);
    for chunk in data.chunks_exact(2) {
        let bits = u16::from_le_bytes(chunk.try_into().expect("chunk size is 2"));
        out.push(f16::from_bits(bits).to_f32());
    }
    Ok(out)
}

fn extract_nibble_map(
    metadata: Option<&std::collections::HashMap<String, String>>,
    layer_name: &str,
) -> Option<[u8; 8]> {
    let meta = metadata?;
    let mut candidates = Vec::new();
    let keys = [
        format!("{layer_name}.pack_order"),
        format!("{layer_name}.nibble_map"),
        format!("{layer_name}.nibble_order"),
        format!("{layer_name}.order"),
        "pack_order".to_string(),
        "nibble_map".to_string(),
    ];
    for key in &keys {
        if let Some(value) = meta.get(key) {
            if let Some(permutation) = parse_permutation(value) {
                return Some(if key.contains("pack") {
                    invert_permutation(permutation)
                } else {
                    permutation
                });
            }
        }
    }

    for (key, value) in meta.iter() {
        if !key.contains(layer_name) {
            continue;
        }
        if let Some(permutation) = parse_permutation(value) {
            candidates.push(if key.contains("pack") {
                invert_permutation(permutation)
            } else {
                permutation
            });
            continue;
        }
        if let Ok(Value::String(inner)) = serde_json::from_str::<Value>(value) {
            if let Some(permutation) = parse_permutation(&inner) {
                candidates.push(if key.contains("pack") {
                    invert_permutation(permutation)
                } else {
                    permutation
                });
            }
        }
    }

    candidates
        .into_iter()
        .find(|perm| is_valid_permutation(perm))
}

fn parse_permutation(value: &str) -> Option<[u8; 8]> {
    if let Ok(list) = serde_json::from_str::<Vec<u8>>(value) {
        return slice_to_array(&list);
    }

    let cleaned = value
        .trim_matches(|c| c == '[' || c == ']' || c == '{' || c == '}')
        .replace(|c: char| c == ';' || c == '|' || c == '/', " ");
    let mut items = Vec::new();
    for token in cleaned.split(|c: char| c == ',' || c.is_whitespace()) {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(num) = trimmed.parse::<u8>() {
            items.push(num);
        } else if let Some(stripped) = trimmed.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
            if let Ok(num) = stripped.parse::<u8>() {
                items.push(num);
            }
        }
    }
    slice_to_array(&items)
}

fn slice_to_array(values: &[u8]) -> Option<[u8; 8]> {
    if values.len() != 8 {
        return None;
    }
    let mut out = [0u8; 8];
    out.copy_from_slice(values);
    Some(out)
}

fn invert_permutation(order: [u8; 8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    for (slot, &logical) in order.iter().enumerate() {
        let idx = logical as usize;
        if idx < 8 {
            out[idx] = slot as u8;
        }
    }
    out
}

fn is_valid_permutation(order: &[u8; 8]) -> bool {
    let mut seen = [false; 8];
    for &value in order.iter() {
        if value as usize >= 8 || seen[value as usize] {
            return false;
        }
        seen[value as usize] = true;
    }
    true
}
