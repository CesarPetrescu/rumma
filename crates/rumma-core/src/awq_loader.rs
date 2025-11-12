use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::PathBuf;

use anyhow::{anyhow, bail, Context, Result};
use half::f16;
use safetensors::tensor::TensorView;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use tokenizers::Tokenizer;

use crate::model::{Dense, Model, ModelConfig};
use crate::quant::{QuantizationConfig, QuantizedLinear};

const AWQ_NIBBLE_MAP: [u8; 8] = [0, 4, 1, 5, 2, 6, 3, 7];

#[derive(Debug, Deserialize)]
struct QuantizationConfigMetadata {
    w_bit: usize,
    q_group_size: usize,
    zero_point: bool,
}

#[derive(Debug, Clone)]
pub struct AwqLayer {
    pub name: String,
    pub linear: QuantizedLinear,
}

#[derive(Debug, Clone)]
pub struct AwqModel {
    layers: Vec<AwqLayer>,
    pub embed_tokens: Dense,
    pub lm_head: Dense,
    pub tokenizer: Tokenizer,
    pub config: ModelConfig,
}

impl Model for AwqModel {
    fn layers(&self) -> Vec<QuantizedLinear> {
        self.layers.iter().map(|l| l.linear.clone()).collect()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl AwqModel {
    pub fn new(
        layers: Vec<AwqLayer>,
        embed_tokens: Dense,
        lm_head: Dense,
        tokenizer: Tokenizer,
        config: ModelConfig,
    ) -> Result<Self> {
        if layers.is_empty() {
            bail!("AWQ model did not contain any quantized layers");
        }
        Ok(Self {
            layers,
            embed_tokens,
            lm_head,
            tokenizer,
            config,
        })
    }

    pub fn into_layers(self) -> Vec<AwqLayer> {
        self.layers
    }

    pub fn hidden_size(&self) -> Option<usize> {
        Some(self.config.hidden_size)
    }

    pub fn depth(&self) -> usize {
        self.layers.len()
    }
}

#[derive(Default)]
struct LayerEntry {
    qweight: Option<String>,
    qzeros: Option<String>,
    scales: Option<String>,
}

pub fn load_awq_model(paths: &[PathBuf]) -> Result<AwqModel> {
    if paths.is_empty() {
        bail!("no safetensors files provided");
    }

    // Find config.json and tokenizer.json in the same directory
    let parent_dir = paths[0]
        .parent()
        .ok_or_else(|| anyhow!("failed to get parent directory of model file"))?;
    let config_path = parent_dir.join("config.json");
    let tokenizer_path = parent_dir.join("tokenizer.json");

    let config: ModelConfig = serde_json::from_str(
        &fs::read_to_string(&config_path)
            .with_context(|| format!("failed to read config.json at {:?}", config_path))?,
    )?;
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| anyhow!("failed to load tokenizer: {}", e))?;

    // Read all files into memory first to manage lifetimes
    let mut loaded_data = Vec::new();
    for path in paths {
        let bytes =
            fs::read(path).with_context(|| format!("failed to read AWQ checkpoint at {:?}", path))?;
        loaded_data.push(bytes);
    }

    let mut entries: BTreeMap<String, LayerEntry> = BTreeMap::new();
    let mut global_metadata: HashMap<String, String> = HashMap::new();
    let mut tensors_map: HashMap<String, TensorView> = HashMap::new();
    let mut loaded_tensors = Vec::new();

    // Read metadata and deserialize tensors from each file
    for (i, bytes) in loaded_data.iter().enumerate() {
        eprintln!("   Parsing safetensors file ({}/{})", i + 1, paths.len());

        let (_, file_metadata) = SafeTensors::read_metadata(bytes)
            .with_context(|| format!("failed to parse header of {:?}", paths[i]))?;
        if let Some(meta_map) = file_metadata.metadata() {
            for (key, value) in meta_map {
                global_metadata
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        }

        let tensors = SafeTensors::deserialize(bytes)
            .with_context(|| format!("failed to parse safetensors file {:?}", paths[i]))?;

        for name in tensors.names() {
            if let Some(prefix) = name.strip_suffix(".qweight") {
                entries.entry(prefix.to_string()).or_default().qweight = Some(name.to_string());
            } else if let Some(prefix) = name.strip_suffix(".qzeros") {
                entries.entry(prefix.to_string()).or_default().qzeros = Some(name.to_string());
            } else if let Some(prefix) = name.strip_suffix(".scales") {
                entries.entry(prefix.to_string()).or_default().scales = Some(name.to_string());
            }
        }

        loaded_tensors.push(tensors);
    }

    for tensors in &loaded_tensors {
        for name in tensors.names() {
            tensors_map.insert(name.to_string(), tensors.tensor(&name)?);
        }
    }

    if entries.is_empty() {
        bail!("no AWQ tensors were found in the provided files");
    }

    let embed_tokens = load_dense_tensor(&tensors_map, "model.embed_tokens")?;
    let lm_head = load_dense_tensor(&tensors_map, "lm_head")?;

    eprintln!("   Processing {} quantized layers...", entries.len());
    let mut layers = Vec::new();
    for (base, entry) in entries {
        let qweight_name = entry
            .qweight
            .ok_or_else(|| anyhow!("layer {base} is missing qweight tensor"))?;
        let scales_name = entry
            .scales
            .ok_or_else(|| anyhow!("layer {base} is missing scales tensor"))?;

        let qweight_tensor = tensors_map
            .get(&qweight_name)
            .with_context(|| format!("tensor {qweight_name} was missing from file"))?;
        let scales_tensor = tensors_map
            .get(&scales_name)
            .with_context(|| format!("tensor {scales_name} was missing from file"))?;
        let qzeros_tensor = match entry.qzeros {
            Some(name) => Some(
                tensors_map
                    .get(&name)
                    .with_context(|| format!("tensor {name} was missing from file"))?,
            ),
            None => None,
        };

        let quantized = build_layer(
            &base,
            qweight_tensor,
            qzeros_tensor,
            scales_tensor,
            Some(&global_metadata),
        )?;
        layers.push(AwqLayer {
            name: base,
            linear: quantized,
        });
    }

    AwqModel::new(layers, embed_tokens, lm_head, tokenizer, config)
}

fn load_dense_tensor(
    tensors_map: &HashMap<String, TensorView>,
    name: &str,
) -> Result<Dense> {
    let weight_name = format!("{}.weight", name);
    let bias_name = format!("{}.bias", name);

    let weight_tensor = tensors_map
        .get(&weight_name)
        .ok_or_else(|| anyhow!("tensor {} not found", weight_name))?;

    let weight_values = read_scales(weight_tensor)?;
    let (rows, cols) = (weight_tensor.shape()[0], weight_tensor.shape()[1]);

    let bias_values = if let Some(bias_tensor) = tensors_map.get(&bias_name) {
        Some(read_scales(bias_tensor)?)
    } else {
        None
    };

    Dense::new(weight_values, bias_values, rows, cols)
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

    let quant_config_keys = ["quantization_config", "quant_config"];
    let quant_metadata = metadata
        .and_then(|meta| search_for_metadata::<QuantizationConfigMetadata>(meta, &quant_config_keys))
        .transpose()?;

    let (group_size, symmetric) = if let Some(meta) = quant_metadata {
        eprintln!(
            "    - Found quantization config: w_bit={}, group_size={}, zero_point={}",
            meta.w_bit, meta.q_group_size, meta.zero_point
        );
        (meta.q_group_size, !meta.zero_point)
    } else {
        let group_size = if rows % scale_rows == 0 {
            rows / scale_rows
        } else {
            bail!(
                "{name}: in_features {} is not divisible by scales rows {}",
                rows,
                scale_rows
            );
        };
        (group_size, qzeros.is_none())
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
        symmetric,
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
    metadata: Option<&HashMap<String, String>>,
    layer_name: &str,
) -> Option<[u8; 8]> {
    let meta = metadata?;

    let layer_keys = [
        format!("{layer_name}.pack_order"),
        format!("{layer_name}.nibble_map"),
        format!("{layer_name}.nibble_order"),
        format!("{layer_name}.order"),
    ];
    let global_keys = ["pack_order", "nibble_map"];

    let mut keys_to_check = Vec::new();
    for key in &layer_keys {
        keys_to_check.push(key.as_str());
    }
    keys_to_check.extend_from_slice(&global_keys);

    if let Some(Ok(list)) = search_for_metadata::<Vec<u8>>(meta, &keys_to_check) {
        if let Some(array) = slice_to_array(&list) {
            if is_valid_permutation(&array) {
                return Some(if keys_to_check.iter().any(|k| k.contains("pack")) {
                    invert_permutation(array)
                } else {
                    array
                });
            }
        }
    }

    // Fallback for string-based permutations
    for key in &keys_to_check {
        if let Some(value) = meta.get(*key) {
            if let Some(permutation) = parse_permutation(value) {
                if is_valid_permutation(&permutation) {
                    return Some(if key.contains("pack") {
                        invert_permutation(permutation)
                    } else {
                        permutation
                    });
                }
            }
        }
    }

    None
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

fn search_for_metadata<'a, T>(
    metadata: &'a HashMap<String, String>,
    keys: &[&str],
) -> Option<Result<T>>
where
    T: serde::de::DeserializeOwned,
{
    for key in keys {
        if let Some(value) = metadata.get(*key) {
            // Try to parse directly
            if let Ok(parsed) = serde_json::from_str(value) {
                return Some(Ok(parsed));
            }
            // If that fails, try to parse from a JSON string within the string
            if let Ok(inner_value) = serde_json::from_str::<String>(value) {
                if let Ok(parsed) = serde_json::from_str(&inner_value) {
                    return Some(Ok(parsed));
                }
            }
        }
    }
    None
}
