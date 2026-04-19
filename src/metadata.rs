use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Serialize)]
pub struct MetadataReport {
    /// "comfyui" | "a1111" | "unknown"
    pub format: String,
    pub positive_prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub model: Option<String>,
    pub seed: Option<i64>,
    pub sampler: Option<String>,
    pub steps: Option<i64>,
    pub cfg: Option<f64>,
    pub lora: Option<String>,
    pub raw: HashMap<String, String>,
}

pub fn read_image_metadata(path: &Path) -> Result<MetadataReport> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;
    let decoder = png::Decoder::new(file);
    let reader = decoder
        .read_info()
        .with_context(|| format!("Failed to parse PNG header in {}", path.display()))?;
    let info = reader.info();

    let mut raw: HashMap<String, String> = HashMap::new();
    for chunk in &info.uncompressed_latin1_text {
        raw.insert(chunk.keyword.clone(), chunk.text.clone());
    }
    for chunk in &info.utf8_text {
        if let Ok(text) = chunk.get_text() {
            raw.insert(chunk.keyword.clone(), text);
        }
    }
    for chunk in &info.compressed_latin1_text {
        if let Ok(text) = chunk.get_text() {
            raw.insert(chunk.keyword.clone(), text);
        }
    }

    if let Some(prompt_json) = raw.get("prompt").cloned() {
        let mut report = parse_comfyui(&prompt_json);
        report.raw = raw;
        return Ok(report);
    }

    if let Some(params) = raw.get("parameters").cloned() {
        let mut report = parse_a1111(&params);
        report.raw = raw;
        return Ok(report);
    }

    Ok(MetadataReport {
        format: "unknown".into(),
        positive_prompt: None,
        negative_prompt: None,
        model: None,
        seed: None,
        sampler: None,
        steps: None,
        cfg: None,
        lora: None,
        raw,
    })
}

fn parse_comfyui(prompt_json: &str) -> MetadataReport {
    let mut report = MetadataReport {
        format: "comfyui".into(),
        positive_prompt: None,
        negative_prompt: None,
        model: None,
        seed: None,
        sampler: None,
        steps: None,
        cfg: None,
        lora: None,
        raw: HashMap::new(),
    };

    let Ok(workflow) = serde_json::from_str::<Value>(prompt_json) else {
        return report;
    };
    let Some(nodes) = workflow.as_object() else {
        return report;
    };

    // First pass: collect sampler/scheduler/loader nodes and the IDs of text prompts
    // feeding "positive" / "negative" slots.
    let mut positive_ref: Option<String> = None;
    let mut negative_ref: Option<String> = None;

    for (_id, node) in nodes {
        let Some(class_type) = node.get("class_type").and_then(|v| v.as_str()) else {
            continue;
        };
        let inputs = node.get("inputs");

        match class_type {
            "KSampler" => {
                if let Some(inputs) = inputs {
                    report.seed = inputs.get("seed").and_then(|v| v.as_i64()).or(report.seed);
                    report.steps = inputs.get("steps").and_then(|v| v.as_i64()).or(report.steps);
                    report.cfg = inputs.get("cfg").and_then(|v| v.as_f64()).or(report.cfg);
                    report.sampler = inputs
                        .get("sampler_name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or(report.sampler.take());
                    if let Some(r) = ref_node_id(inputs.get("positive")) {
                        positive_ref = Some(r);
                    }
                    if let Some(r) = ref_node_id(inputs.get("negative")) {
                        negative_ref = Some(r);
                    }
                }
            }
            "RandomNoise" => {
                if let Some(inputs) = inputs {
                    report.seed = inputs
                        .get("noise_seed")
                        .and_then(|v| v.as_i64())
                        .or(report.seed);
                }
            }
            "BasicScheduler" => {
                if let Some(inputs) = inputs {
                    report.steps = inputs.get("steps").and_then(|v| v.as_i64()).or(report.steps);
                }
            }
            "KSamplerSelect" => {
                if let Some(inputs) = inputs {
                    report.sampler = inputs
                        .get("sampler_name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or(report.sampler.take());
                }
            }
            "BasicGuider" => {
                if let Some(inputs) = inputs {
                    if let Some(r) = ref_node_id(inputs.get("conditioning")) {
                        positive_ref = Some(r);
                    }
                }
            }
            "InpaintModelConditioning" => {
                if let Some(inputs) = inputs {
                    if let Some(r) = ref_node_id(inputs.get("positive")) {
                        positive_ref = Some(r);
                    }
                    if let Some(r) = ref_node_id(inputs.get("negative")) {
                        negative_ref = Some(r);
                    }
                }
            }
            "CheckpointLoaderSimple" => {
                if let Some(inputs) = inputs {
                    report.model = inputs
                        .get("ckpt_name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or(report.model.take());
                }
            }
            "UnetLoaderGGUF" => {
                if let Some(inputs) = inputs {
                    report.model = inputs
                        .get("unet_name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or(report.model.take());
                }
            }
            "LoraLoader" => {
                if let Some(inputs) = inputs {
                    report.lora = inputs
                        .get("lora_name")
                        .and_then(|v| v.as_str())
                        .map(str::to_string)
                        .or(report.lora.take());
                }
            }
            _ => {}
        }
    }

    // Second pass: resolve the CLIPTextEncode nodes we've identified, following
    // any conditioning-passthrough nodes (InpaintModelConditioning, etc.) between
    // the sampler/guider and the actual text encode.
    if let Some(id) = positive_ref {
        report.positive_prompt = resolve_text(nodes, &id, "positive", 8);
    }
    if let Some(id) = negative_ref {
        let text = resolve_text(nodes, &id, "negative", 8);
        // Empty negative prompts aren't interesting — surface None.
        report.negative_prompt = text.filter(|s| !s.is_empty());
    }

    report
}

/// ComfyUI references another node as `[node_id, output_index]`.
fn ref_node_id(value: Option<&Value>) -> Option<String> {
    let arr = value?.as_array()?;
    let id = arr.first()?;
    id.as_str()
        .map(str::to_string)
        .or_else(|| id.as_i64().map(|n| n.to_string()))
}

/// Follow a conditioning chain from `id` until we hit a `CLIPTextEncode` node.
/// `polarity` is `"positive"` or `"negative"` — the input name to follow on
/// conditioning-passthrough nodes (InpaintModelConditioning takes both).
fn resolve_text(
    nodes: &serde_json::Map<String, Value>,
    id: &str,
    polarity: &str,
    depth: u8,
) -> Option<String> {
    if depth == 0 {
        return None;
    }
    let node = nodes.get(id)?;
    let class_type = node.get("class_type").and_then(|v| v.as_str())?;
    let inputs = node.get("inputs")?;

    if class_type == "CLIPTextEncode" {
        return inputs.get("text")?.as_str().map(str::to_string);
    }

    // Conditioning passthrough — follow the matching polarity input, then
    // try a few common generic names.
    for key in [polarity, "conditioning", "positive"] {
        if let Some(next_id) = ref_node_id(inputs.get(key)) {
            if let Some(text) = resolve_text(nodes, &next_id, polarity, depth - 1) {
                return Some(text);
            }
        }
    }
    None
}

fn parse_a1111(params: &str) -> MetadataReport {
    // A1111 format: first line(s) = positive prompt,
    // line starting with "Negative prompt:" = negative,
    // last line = comma-separated key:value pairs.
    let mut report = MetadataReport {
        format: "a1111".into(),
        positive_prompt: None,
        negative_prompt: None,
        model: None,
        seed: None,
        sampler: None,
        steps: None,
        cfg: None,
        lora: None,
        raw: HashMap::new(),
    };

    let lines: Vec<&str> = params.lines().collect();
    if lines.is_empty() {
        return report;
    }

    let mut settings_idx: Option<usize> = None;
    let mut negative_idx: Option<usize> = None;
    for (i, line) in lines.iter().enumerate() {
        if line.starts_with("Negative prompt:") {
            negative_idx = Some(i);
        } else if line.contains("Steps:")
            && line.contains("Sampler:")
            && settings_idx.is_none()
        {
            settings_idx = Some(i);
        }
    }

    let positive_end = negative_idx
        .or(settings_idx)
        .unwrap_or(lines.len());
    let positive = lines[..positive_end].join("\n").trim().to_string();
    if !positive.is_empty() {
        report.positive_prompt = Some(positive);
    }

    if let Some(ni) = negative_idx {
        let end = settings_idx.unwrap_or(lines.len());
        let neg_first = lines[ni].trim_start_matches("Negative prompt:").trim_start();
        let mut neg = neg_first.to_string();
        if ni + 1 < end {
            neg.push('\n');
            neg.push_str(&lines[ni + 1..end].join("\n"));
        }
        let neg = neg.trim().to_string();
        if !neg.is_empty() {
            report.negative_prompt = Some(neg);
        }
    }

    if let Some(si) = settings_idx {
        for pair in lines[si].split(',') {
            let pair = pair.trim();
            let Some((k, v)) = pair.split_once(':') else {
                continue;
            };
            let k = k.trim();
            let v = v.trim();
            match k {
                "Steps" => report.steps = v.parse().ok(),
                "Sampler" => report.sampler = Some(v.to_string()),
                "Seed" => report.seed = v.parse().ok(),
                "CFG scale" => report.cfg = v.parse().ok(),
                "Model" => report.model = Some(v.to_string()),
                _ => {}
            }
        }
    }

    report
}
