use anyhow::{Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

use crate::comfy::{ComfyClient, build_txt2img_workflow};
use crate::mcp::{CallToolResult, ContentItem, ToolDefinition};

/// Return the list of tool definitions with JSON Schemas.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "generate_image".into(),
            description: "Generate an image using Stable Diffusion via ComfyUI. Returns file path and optionally the image data.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The positive prompt describing what to generate"
                    },
                    "negative_prompt": {
                        "type": "string",
                        "description": "Negative prompt (things to avoid)",
                        "default": "ugly, blurry, low quality, deformed, watermark, text"
                    },
                    "checkpoint": {
                        "type": "string",
                        "description": "Checkpoint model name. Use list_checkpoints to see available models."
                    },
                    "width": {
                        "type": "integer",
                        "description": "Image width in pixels",
                        "default": 512
                    },
                    "height": {
                        "type": "integer",
                        "description": "Image height in pixels",
                        "default": 768
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Number of sampling steps",
                        "default": 30
                    },
                    "cfg": {
                        "type": "number",
                        "description": "Classifier-free guidance scale",
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
            name: "list_checkpoints".into(),
            description: "List available Stable Diffusion checkpoint models.".into(),
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
        "list_checkpoints" => handle_list_checkpoints(client).await,
        "system_status" => handle_system_status(client).await,
        _ => bail!("Unknown tool: {tool_name}"),
    }
}

#[derive(Debug, Deserialize)]
struct GenerateImageArgs {
    prompt: String,
    #[serde(default = "default_negative")]
    negative_prompt: String,
    checkpoint: Option<String>,
    #[serde(default = "default_width")]
    width: u32,
    #[serde(default = "default_height")]
    height: u32,
    #[serde(default = "default_steps")]
    steps: u32,
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
fn default_width() -> u32 {
    512
}
fn default_height() -> u32 {
    768
}
fn default_steps() -> u32 {
    30
}
fn default_cfg() -> f64 {
    7.5
}
fn default_sampler() -> String {
    "euler_ancestral".into()
}

async fn handle_generate_image(client: &ComfyClient, args: &Value) -> Result<CallToolResult> {
    let args: GenerateImageArgs = serde_json::from_value(args.clone())?;

    // Resolve checkpoint
    let checkpoint = match &args.checkpoint {
        Some(c) => c.clone(),
        None => {
            let models = client.list_checkpoints().await?;
            if models.is_empty() {
                bail!("No checkpoint models found in ComfyUI");
            }
            models[0].clone()
        }
    };

    // Resolve seed
    let seed = args.seed.unwrap_or_else(|| rand::random::<i64>().abs());

    info!(
        prompt = %args.prompt,
        checkpoint = %checkpoint,
        size = format!("{}x{}", args.width, args.height),
        steps = args.steps,
        seed = seed,
        "Generating image"
    );

    // Build and submit workflow
    let workflow = build_txt2img_workflow(
        &checkpoint,
        &args.prompt,
        &args.negative_prompt,
        args.width,
        args.height,
        args.steps,
        args.cfg,
        &args.sampler,
        seed,
    );

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
            "checkpoint": checkpoint,
            "size": format!("{}x{}", args.width, args.height),
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

async fn handle_list_checkpoints(client: &ComfyClient) -> Result<CallToolResult> {
    let models = client.list_checkpoints().await?;
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
