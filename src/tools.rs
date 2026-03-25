use anyhow::{Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

use crate::comfy::{
    ComfyClient, LoraSpec, build_checkpoint_workflow, build_flux_workflow, parse_model_specifier,
};
use crate::mcp::{CallToolResult, ContentItem, ToolDefinition};

/// Return the list of tool definitions with JSON Schemas.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "generate_image".into(),
            description: "Generate an image using Stable Diffusion or FLUX via ComfyUI. Returns file path and optionally the image data.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The positive prompt describing what to generate"
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Negative prompt (things to avoid). Ignored for FLUX models.",
                        "default": "ugly, blurry, low quality, deformed, watermark, text"
                    },
                    "model": {
                        "type": "string",
                        "description": "Model to use, as shown by list_models (e.g. 'checkpoints/sd_xl_base_1.0.safetensors' or 'unet_gguf/flux1-schnell-Q8_0.gguf'). Plain filename assumes checkpoints/. Use list_models to see available models."
                    },
                    "lora": {
                        "type": "string",
                        "description": "Optional LoRA to apply (e.g. 'LogoRedmondV2-Logo-LogoRedmAF.safetensors'). Just the filename, no folder prefix needed."
                    },
                    "lora_strength": {
                        "type": "number",
                        "description": "LoRA strength (0.0-1.0)",
                        "default": 0.8
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels. Default: 512 for SD, 1024 for FLUX.",
                        "default": 512
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels. Default: 768 for SD, 1024 for FLUX.",
                        "default": 768
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of sampling steps. Default: 30 for SD, 4 for FLUX.",
                        "default": 30
                    },
                    "cfg": {
                        "type": "number",
                        "description": "Classifier-free guidance scale (SD only, ignored for FLUX)",
                        "default": 7.5
                    },
                    "sampler": {
                        "type": "string",
                        "description": "Sampler name (e.g. euler_ancestral, euler, dpmpp_2m, dpmpp_sde)",
                        "default": "euler_ancestral"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed. Omit for random."
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path to save the image. If omitted, saves to a temp file."
                    },
                    "return_image": {
                        "type": "boolean",
                        "description": "If true, return base64 image data so you can see it. Default false.",
                        "default": false
                    }
                },
                "required": ["prompt"]
            }),
        },
        ToolDefinition {
            name: "list_models".into(),
            description: "List all available models across all ComfyUI model folders (checkpoints, loras, unet_gguf, etc). Returns folder/filename format.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
        ToolDefinition {
            name: "system_status".into(),
            description: "Get ComfyUI system status: GPU info, VRAM, versions, queue.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        },
    ]
}

// ── Tool Handlers ───────────────────────────────────────────────────────

pub async fn handle_tool_call(
    client: &ComfyClient,
    tool_name: &str,
    args: &Value,
) -> Result<CallToolResult> {
    match tool_name {
        "generate_image" => handle_generate_image(client, args).await,
        "list_models" => handle_list_models(client).await,
        // Keep old name working for backwards compat
        "list_checkpoints" => handle_list_models(client).await,
        "system_status" => handle_system_status(client).await,
        _ => bail!("Unknown tool: {tool_name}"),
    }
}

#[derive(Debug, Deserialize)]
struct GenerateImageArgs {
    prompt: String,
    #[serde(default = "default_negative")]
    negative_prompt: String,
    model: Option<String>,
    // Backwards compat: accept "checkpoint" too
    checkpoint: Option<String>,
    lora: Option<String>,
    #[serde(default = "default_lora_strength")]
    lora_strength: f64,
    width: Option<u32>,
    height: Option<u32>,
    steps: Option<u32>,
    #[serde(default = "default_cfg")]
    cfg: f64,
    #[serde(default = "default_sampler")]
    sampler: String,
    seed: Option<i64>,
    output_path: Option<String>,
    #[serde(default)]
    return_image: bool,
}

fn default_negative() -> String {
    "ugly, blurry, low quality, deformed, watermark, text".into()
}
fn default_lora_strength() -> f64 {
    0.8
}
fn default_cfg() -> f64 {
    7.5
}
fn default_sampler() -> String {
    "euler_ancestral".into()
}

fn is_flux_model(folder: &str, filename: &str) -> bool {
    folder == "unet_gguf" || filename.to_lowercase().contains("flux")
}

async fn handle_generate_image(client: &ComfyClient, args: &Value) -> Result<CallToolResult> {
    let args: GenerateImageArgs = serde_json::from_value(args.clone())?;

    // Resolve model specifier (prefer "model", fall back to "checkpoint")
    let model_spec = args
        .model
        .as_deref()
        .or(args.checkpoint.as_deref())
        .unwrap_or("");

    let (folder, filename) = if model_spec.is_empty() {
        // Default to first available checkpoint
        let models = client.list_checkpoints().await?;
        if models.is_empty() {
            bail!("No checkpoint models found in ComfyUI");
        }
        ("checkpoints".to_string(), models[0].clone())
    } else {
        let (f, n) = parse_model_specifier(model_spec)?;
        (f.to_string(), n.to_string())
    };

    let is_flux = is_flux_model(&folder, &filename);

    // Apply model-type-appropriate defaults
    let width = args.width.unwrap_or(if is_flux { 1024 } else { 512 });
    let height = args.height.unwrap_or(if is_flux { 1024 } else { 768 });
    let steps = args.steps.unwrap_or(if is_flux { 4 } else { 30 });
    let seed = args.seed.unwrap_or_else(|| rand::random::<i64>().abs());

    // LoRA spec
    let lora_spec = args.lora.as_ref().map(|name| LoraSpec {
        name: name.clone(),
        strength: args.lora_strength,
    });

    info!(
        prompt = %args.prompt,
        model = format!("{folder}/{filename}"),
        lora = args.lora.as_deref().unwrap_or("none"),
        size = format!("{width}x{height}"),
        steps = steps,
        seed = seed,
        "Generating image"
    );

    // Build workflow based on model type
    let workflow = if is_flux {
        build_flux_workflow(
            &filename,
            &args.prompt,
            width,
            height,
            steps,
            seed,
            lora_spec.as_ref(),
        )
    } else {
        build_checkpoint_workflow(
            &filename,
            &args.prompt,
            &args.negative_prompt,
            width,
            height,
            steps,
            args.cfg,
            &args.sampler,
            seed,
            lora_spec.as_ref(),
        )
    };

    let prompt_id = client.submit_prompt(&workflow).await?;
    info!(prompt_id = %prompt_id, "Submitted to ComfyUI");

    // Wait for completion
    let outputs = client
        .wait_for_completion(&prompt_id, Duration::from_secs(300))
        .await?;

    let output = outputs
        .first()
        .ok_or_else(|| anyhow::anyhow!("No image output found"))?;

    // Fetch image bytes
    let image_bytes = client.fetch_image(output).await?;

    // Determine output file path
    let file_path: PathBuf = match &args.output_path {
        Some(p) => {
            let path = PathBuf::from(p);
            tokio::fs::write(&path, &image_bytes).await?;
            path
        }
        None => {
            let temp_dir = std::env::temp_dir().join("comfy-ui-mcp");
            tokio::fs::create_dir_all(&temp_dir).await?;
            let filename = format!("gen_{prompt_id}.png");
            let path = temp_dir.join(filename);
            tokio::fs::write(&path, &image_bytes).await?;
            path
        }
    };

    let file_path_str = file_path.to_string_lossy().to_string();
    info!(path = %file_path_str, bytes = image_bytes.len(), "Image saved");

    // Build response
    let mut content = vec![ContentItem::Text {
        text: serde_json::json!({
            "file_path": file_path_str,
            "seed": seed,
            "model": format!("{folder}/{filename}"),
            "lora": args.lora,
            "size": format!("{width}x{height}"),
        })
        .to_string(),
    }];

    if args.return_image {
        let b64 = BASE64.encode(&image_bytes);
        content.push(ContentItem::Image {
            data: b64,
            mime_type: "image/png".into(),
        });
    }

    Ok(CallToolResult {
        content,
        is_error: false,
    })
}

async fn handle_list_models(client: &ComfyClient) -> Result<CallToolResult> {
    let models = client.list_all_models().await?;
    Ok(CallToolResult {
        content: vec![ContentItem::Text {
            text: serde_json::to_string_pretty(&models)?,
        }],
        is_error: false,
    })
}

async fn handle_system_status(client: &ComfyClient) -> Result<CallToolResult> {
    let stats = client.system_stats().await?;
    Ok(CallToolResult {
        content: vec![ContentItem::Text {
            text: serde_json::to_string_pretty(&stats)?,
        }],
        is_error: false,
    })
}
