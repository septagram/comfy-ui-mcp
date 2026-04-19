use anyhow::{Context, Result, bail};
use reqwest::Client;
use serde_json::Value;
use std::time::Duration;
use tracing::{debug, info};

/// Known model folders that are safe to query.
const MODEL_FOLDERS: &[&str] = &[
    "checkpoints",
    "loras",
    "unet_gguf",
    "diffusion_models",
    "vae",
    "text_encoders",
    "clip",
    "clip_gguf",
    "controlnet",
    "embeddings",
    "upscale_models",
];

/// Folders allowed as the prefix in a model specifier for generate_image.
const ALLOWED_MODEL_FOLDERS: &[&str] = &["checkpoints", "unet_gguf", "diffusion_models"];

pub struct ComfyClient {
    client: Client,
    base_url: String,
}

impl ComfyClient {
    pub fn new(base_url: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
        }
    }

    /// List models in a specific folder.
    pub async fn list_models_in_folder(&self, folder: &str) -> Result<Vec<String>> {
        let url = format!("{}/models/{}", self.base_url, folder);
        let resp = self.client.get(&url).send().await?;
        if !resp.status().is_success() {
            return Ok(vec![]);
        }
        let models: Vec<String> = resp.json().await.unwrap_or_default();
        Ok(models)
    }

    /// List all models across known folders, prefixed with folder name.
    pub async fn list_all_models(&self) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for &folder in MODEL_FOLDERS {
            match self.list_models_in_folder(folder).await {
                Ok(models) => {
                    for model in models {
                        result.push(format!("{folder}/{model}"));
                    }
                }
                Err(_) => continue,
            }
        }
        Ok(result)
    }

    /// List available checkpoint models (convenience).
    pub async fn list_checkpoints(&self) -> Result<Vec<String>> {
        self.list_models_in_folder("checkpoints").await
    }

    /// Get system stats (GPU, VRAM, versions).
    pub async fn system_stats(&self) -> Result<Value> {
        let url = format!("{}/system_stats", self.base_url);
        let resp: Value = self.client.get(&url).send().await?.json().await?;
        Ok(resp)
    }

    /// Submit a workflow prompt and return the prompt_id.
    pub async fn submit_prompt(&self, workflow: &Value) -> Result<String> {
        let url = format!("{}/prompt", self.base_url);
        let body = serde_json::json!({ "prompt": workflow });
        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        let body_text = resp.text().await?;
        let parsed: Value = serde_json::from_str(&body_text).unwrap_or(Value::Null);

        if let Some(id) = parsed.get("prompt_id").and_then(|v| v.as_str()) {
            return Ok(id.to_string());
        }

        // ComfyUI returns node-level errors under `node_errors`; workflow-level
        // errors under `error`. Surface whichever is present.
        let detail = parsed
            .get("node_errors")
            .or_else(|| parsed.get("error"))
            .map(|v| v.to_string())
            .unwrap_or(body_text);
        bail!("ComfyUI rejected workflow (HTTP {status}): {detail}");
    }

    /// Poll history until the prompt completes. Returns the output node's images info.
    pub async fn wait_for_completion(
        &self,
        prompt_id: &str,
        timeout: Duration,
    ) -> Result<Vec<ImageOutput>> {
        let url = format!("{}/history/{}", self.base_url, prompt_id);
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                bail!("Timed out waiting for image generation after {:?}", timeout);
            }

            let resp: Value = self.client.get(&url).send().await?.json().await?;

            if let Some(entry) = resp.get(prompt_id) {
                // Check for error
                if let Some(status) = entry.get("status") {
                    if status.get("status_str").and_then(|s| s.as_str()) == Some("error") {
                        let msgs = status.get("messages").cloned().unwrap_or(Value::Null);
                        bail!("ComfyUI execution error: {msgs}");
                    }
                }

                // Look for image outputs in any node
                if let Some(outputs) = entry.get("outputs").and_then(|o| o.as_object()) {
                    for (_node_id, node_output) in outputs {
                        if let Some(images) = node_output.get("images").and_then(|i| i.as_array())
                        {
                            let result: Vec<ImageOutput> = images
                                .iter()
                                .filter_map(|img| {
                                    Some(ImageOutput {
                                        filename: img.get("filename")?.as_str()?.to_string(),
                                        subfolder: img
                                            .get("subfolder")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or("")
                                            .to_string(),
                                        type_: img
                                            .get("type")
                                            .and_then(|s| s.as_str())
                                            .unwrap_or("output")
                                            .to_string(),
                                    })
                                })
                                .collect();

                            if !result.is_empty() {
                                info!(count = result.len(), "Generation complete");
                                return Ok(result);
                            }
                        }
                    }
                }
            }

            debug!("Still waiting for prompt {prompt_id}...");
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }

    /// Upload an image file to ComfyUI's input folder. Returns the stored filename.
    pub async fn upload_image(&self, file_path: &str) -> Result<String> {
        let url = format!("{}/upload/image", self.base_url);
        let file_bytes = tokio::fs::read(file_path).await
            .with_context(|| format!("Failed to read source image: {file_path}"))?;

        let file_name = std::path::Path::new(file_path)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("upload.png")
            .to_string();

        let part = reqwest::multipart::Part::bytes(file_bytes)
            .file_name(file_name)
            .mime_str("image/png")?;
        let form = reqwest::multipart::Form::new().part("image", part);

        let resp: Value = self
            .client
            .post(&url)
            .multipart(form)
            .send()
            .await?
            .json()
            .await?;

        resp["name"]
            .as_str()
            .map(String::from)
            .context("No 'name' in upload response")
    }

    /// Fetch the raw image bytes from ComfyUI.
    pub async fn fetch_image(&self, output: &ImageOutput) -> Result<Vec<u8>> {
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base_url,
            urlencoding::encode(&output.filename),
            urlencoding::encode(&output.subfolder),
            urlencoding::encode(&output.type_),
        );
        let bytes = self.client.get(&url).send().await?.bytes().await?;
        Ok(bytes.to_vec())
    }
}

#[derive(Debug)]
pub struct ImageOutput {
    pub filename: String,
    pub subfolder: String,
    pub type_: String,
}

/// Parse a model specifier like "checkpoints/model.safetensors" or just "model.ckpt".
/// Returns (folder, filename). Validates against path traversal.
pub fn parse_model_specifier(spec: &str) -> Result<(&str, &str)> {
    // Reject path traversal
    if spec.contains("..") || spec.contains('\\') {
        bail!("Invalid model specifier: path traversal not allowed");
    }

    match spec.split_once('/') {
        Some((folder, filename)) => {
            // Validate folder is in whitelist
            if !ALLOWED_MODEL_FOLDERS.contains(&folder) {
                bail!(
                    "Unknown model folder '{folder}'. Allowed: {}",
                    ALLOWED_MODEL_FOLDERS.join(", ")
                );
            }
            // Validate filename has no further slashes
            if filename.contains('/') {
                bail!("Invalid model specifier: nested paths not allowed");
            }
            Ok((folder, filename))
        }
        None => {
            // No prefix — assume checkpoints
            Ok(("checkpoints", spec))
        }
    }
}

/// Build a txt2img workflow for SD1.5/SDXL checkpoints.
pub fn build_checkpoint_workflow(
    checkpoint: &str,
    prompt: &str,
    negative_prompt: &str,
    width: u32,
    height: u32,
    steps: u32,
    cfg: f64,
    sampler: &str,
    seed: i64,
    lora: Option<&LoraSpec>,
) -> Value {
    let mut workflow = serde_json::json!({
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint
            }
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "mcp_gen"
            }
        }
    });

    // Model/clip source node — either checkpoint directly or through LoRA
    let (model_ref, clip_ref): (Value, Value) = match lora {
        Some(lora) => {
            workflow["10"] = serde_json::json!({
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": lora.name,
                    "strength_model": lora.strength,
                    "strength_clip": lora.strength
                }
            });
            (serde_json::json!(["10", 0]), serde_json::json!(["10", 1]))
        }
        None => (serde_json::json!(["1", 0]), serde_json::json!(["1", 1])),
    };

    workflow["2"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": clip_ref
        }
    });
    workflow["3"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": negative_prompt,
            "clip": clip_ref
        }
    });
    workflow["5"] = serde_json::json!({
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["2", 0],
            "negative": ["3", 0],
            "latent_image": ["4", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": "normal",
            "denoise": 1.0
        }
    });

    workflow
}

/// Build a FLUX txt2img workflow using UnetLoaderGGUF + DualCLIPLoader.
pub fn build_flux_workflow(
    unet_name: &str,
    prompt: &str,
    width: u32,
    height: u32,
    steps: u32,
    seed: i64,
    lora: Option<&LoraSpec>,
) -> Value {
    let mut workflow = serde_json::json!({
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": {
                "unet_name": unet_name
            }
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux"
            }
        },
        "4": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {
                "width": width,
                "height": height,
                "batch_size": 1
            }
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["7", 0],
                "vae": ["1", 1]  // UnetLoaderGGUF output 1 might not have VAE
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "mcp_flux"
            }
        }
    });

    // Model/clip source — optionally through LoRA
    let (model_ref, clip_ref): (Value, Value) = match lora {
        Some(lora) => {
            workflow["10"] = serde_json::json!({
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["2", 0],
                    "lora_name": lora.name,
                    "strength_model": lora.strength,
                    "strength_clip": lora.strength
                }
            });
            (serde_json::json!(["10", 0]), serde_json::json!(["10", 1]))
        }
        None => (serde_json::json!(["1", 0]), serde_json::json!(["2", 0])),
    };

    workflow["3"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": clip_ref
        }
    });

    // FLUX uses the advanced sampler pipeline
    workflow["5"] = serde_json::json!({
        "class_type": "RandomNoise",
        "inputs": { "noise_seed": seed }
    });
    workflow["6"] = serde_json::json!({
        "class_type": "BasicGuider",
        "inputs": {
            "model": model_ref,
            "conditioning": ["3", 0]
        }
    });
    workflow["11"] = serde_json::json!({
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": "simple",
            "steps": steps,
            "denoise": 1.0
        }
    });
    workflow["12"] = serde_json::json!({
        "class_type": "KSamplerSelect",
        "inputs": { "sampler_name": "euler" }
    });
    workflow["7"] = serde_json::json!({
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["5", 0],
            "guider": ["6", 0],
            "sampler": ["12", 0],
            "sigmas": ["11", 0],
            "latent_image": ["4", 0]
        }
    });

    // FLUX GGUF loader doesn't output VAE — need a separate VAE loader
    // For now, use the built-in VAE from the unet loader (output index 1 if available)
    // If that fails, we may need to add a VAELoader node
    // UnetLoaderGGUF only has 1 output (MODEL), so we need a separate approach
    // Actually, for FLUX we should use the VAE that comes with it
    // Let's load it via a VAELoader if available, or rely on the checkpoint's VAE
    // For simplicity: use a separate VAE loader with the FLUX ae
    workflow["13"] = serde_json::json!({
        "class_type": "VAELoader",
        "inputs": {
            "vae_name": "ae.safetensors"
        }
    });
    workflow["8"] = serde_json::json!({
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["13", 0]
        }
    });

    workflow
}

pub struct LoraSpec {
    pub name: String,
    pub strength: f64,
}

/// Check if a model filename is an InstructPix2Pix-compatible editing model.
pub fn is_ip2p_model(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    lower.contains("instruct")
        || lower.contains("pix2pix")
        || lower.contains("ip2p")
        || lower.contains("cosxl_edit")
}

/// Check if a model filename is a FLUX.1-Fill-style inpainting model.
/// Matches files that mention both "flux" and "fill", e.g. `flux1-fill-dev-Q8_0.gguf`.
pub fn is_flux_inpaint_model(filename: &str) -> bool {
    let lower = filename.to_lowercase();
    lower.contains("flux") && lower.contains("fill")
}

/// Check if an IP2P model is CosXL (SDXL-based, higher res defaults).
fn is_cosxl_model(filename: &str) -> bool {
    filename.to_lowercase().contains("cosxl")
}

/// Default parameters for IP2P models based on type.
pub fn ip2p_defaults(filename: &str) -> (u32, u32, u32, f64) {
    if is_cosxl_model(filename) {
        (1024, 1024, 20, 5.0) // CosXL: SDXL res, fewer steps, lower cfg
    } else {
        (512, 512, 30, 7.5) // Original IP2P: SD 1.5 res
    }
}

/// Build an InstructPix2Pix / CosXL Edit workflow.
/// Note: width/height are not needed — the latent dimensions come from the source image.
pub fn build_ip2p_workflow(
    checkpoint: &str,
    uploaded_image_name: &str,
    prompt: &str,
    negative_prompt: &str,
    steps: u32,
    cfg: f64,
    sampler: &str,
    seed: i64,
    denoise: f64,
    lora: Option<&LoraSpec>,
) -> Value {
    let mut workflow = serde_json::json!({
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": checkpoint
            }
        },
        "20": {
            "class_type": "LoadImage",
            "inputs": {
                "image": uploaded_image_name
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "mcp_edit"
            }
        }
    });

    // Model/clip source — optionally through LoRA
    let (model_ref, clip_ref, vae_ref): (Value, Value, Value) = match lora {
        Some(lora) => {
            workflow["10"] = serde_json::json!({
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["1", 1],
                    "lora_name": lora.name,
                    "strength_model": lora.strength,
                    "strength_clip": lora.strength
                }
            });
            (
                serde_json::json!(["10", 0]),
                serde_json::json!(["10", 1]),
                serde_json::json!(["1", 2]),
            )
        }
        None => (
            serde_json::json!(["1", 0]),
            serde_json::json!(["1", 1]),
            serde_json::json!(["1", 2]),
        ),
    };

    // CLIP encode the edit instruction
    workflow["2"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": prompt,
            "clip": clip_ref
        }
    });
    workflow["3"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": negative_prompt,
            "clip": clip_ref
        }
    });

    // InstructPixToPixConditioning: combines text conditioning with source image
    workflow["21"] = serde_json::json!({
        "class_type": "InstructPixToPixConditioning",
        "inputs": {
            "positive": ["2", 0],
            "negative": ["3", 0],
            "vae": vae_ref,
            "pixels": ["20", 0]
        }
    });

    // KSampler with IP2P conditioning
    workflow["5"] = serde_json::json!({
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": ["21", 0],
            "negative": ["21", 1],
            "latent_image": ["21", 2],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler,
            "scheduler": "normal",
            "denoise": denoise
        }
    });

    workflow
}

/// Build a FLUX.1-Fill inpainting workflow. Uses `UnetLoaderGGUF` for the unet,
/// `DualCLIPLoader` for CLIP, `VAELoader` for the VAE (all shared with FLUX txt2img),
/// plus `LoadImage` + `LoadImageMask` for the editing inputs and `InpaintModelConditioning`
/// to prep the latent. Mask is read from the uploaded greyscale PNG's red channel so users
/// can paint masks in any tool without worrying about alpha channels.
///
/// Width/height are derived from the source image by the conditioning node — no latent
/// dimensions passed in. Denoise is fixed at 1.0; Fill-dev respects the mask internally.
pub fn build_flux_inpaint_workflow(
    unet_name: &str,
    uploaded_base_name: &str,
    uploaded_mask_name: &str,
    prompt: &str,
    steps: u32,
    seed: i64,
    lora: Option<&LoraSpec>,
) -> Value {
    let mut workflow = serde_json::json!({
        "1": {
            "class_type": "UnetLoaderGGUF",
            "inputs": { "unet_name": unet_name }
        },
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": "clip_l.safetensors",
                "clip_name2": "t5xxl_fp8_e4m3fn.safetensors",
                "type": "flux"
            }
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": { "vae_name": "ae.safetensors" }
        },
        "20": {
            "class_type": "LoadImage",
            "inputs": { "image": uploaded_base_name }
        },
        "21": {
            "class_type": "LoadImageMask",
            "inputs": {
                "image": uploaded_mask_name,
                "channel": "red"
            }
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "mcp_flux_inpaint"
            }
        }
    });

    // Model/clip source — optionally through LoRA (same shape as build_flux_workflow)
    let (model_ref, clip_ref): (Value, Value) = match lora {
        Some(lora) => {
            workflow["10"] = serde_json::json!({
                "class_type": "LoraLoader",
                "inputs": {
                    "model": ["1", 0],
                    "clip": ["2", 0],
                    "lora_name": lora.name,
                    "strength_model": lora.strength,
                    "strength_clip": lora.strength
                }
            });
            (serde_json::json!(["10", 0]), serde_json::json!(["10", 1]))
        }
        None => (serde_json::json!(["1", 0]), serde_json::json!(["2", 0])),
    };

    // Positive + (empty) negative text conditioning
    workflow["3"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": { "text": prompt, "clip": clip_ref }
    });
    workflow["4"] = serde_json::json!({
        "class_type": "CLIPTextEncode",
        "inputs": { "text": "", "clip": clip_ref }
    });

    // Inpaint conditioning: fuses text conditioning with source pixels + mask.
    // noise_mask=true attaches the mask to the latent so the sampler only denoises
    // inside the masked region.
    workflow["22"] = serde_json::json!({
        "class_type": "InpaintModelConditioning",
        "inputs": {
            "positive": ["3", 0],
            "negative": ["4", 0],
            "vae": ["13", 0],
            "pixels": ["20", 0],
            "mask": ["21", 0],
            "noise_mask": true
        }
    });

    // FLUX advanced sampler pipeline (mirrors build_flux_workflow)
    workflow["5"] = serde_json::json!({
        "class_type": "RandomNoise",
        "inputs": { "noise_seed": seed }
    });
    workflow["6"] = serde_json::json!({
        "class_type": "BasicGuider",
        "inputs": {
            "model": model_ref,
            "conditioning": ["22", 0]
        }
    });
    workflow["11"] = serde_json::json!({
        "class_type": "BasicScheduler",
        "inputs": {
            "model": model_ref,
            "scheduler": "simple",
            "steps": steps,
            "denoise": 1.0
        }
    });
    workflow["12"] = serde_json::json!({
        "class_type": "KSamplerSelect",
        "inputs": { "sampler_name": "euler" }
    });
    workflow["7"] = serde_json::json!({
        "class_type": "SamplerCustomAdvanced",
        "inputs": {
            "noise": ["5", 0],
            "guider": ["6", 0],
            "sampler": ["12", 0],
            "sigmas": ["11", 0],
            "latent_image": ["22", 2]
        }
    });
    workflow["8"] = serde_json::json!({
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["13", 0]
        }
    });

    workflow
}
