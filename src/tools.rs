use anyhow::{Result, bail};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use serde::Deserialize;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Duration;
use tracing::info;

use crate::comfy::{
    ComfyClient, LoraSpec, build_checkpoint_workflow, build_flux_inpaint_workflow,
    build_flux_workflow, build_ip2p_workflow, ip2p_defaults, is_flux_inpaint_model, is_ip2p_model,
    parse_model_specifier,
};
use crate::mcp::{CallToolResult, ContentItem, ToolDefinition};
use crate::metadata::read_image_metadata;

/// Return the list of tool definitions with JSON Schemas.
pub fn tool_definitions() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            name: "generate_image".into(),
            description: "Generate or edit an image via ComfyUI. For editing, provide source_image with an IP2P model (cosxl_edit or instruct-pix2pix) and use the prompt as an edit instruction.".into(),
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
                    "source_image": {
                        "type": "string",
                        "description": "Path to a source image for editing. When provided, the prompt becomes an edit instruction (e.g. 'make it snowy', 'add a hat'). Requires an IP2P-compatible model: cosxl_edit.safetensors (best quality, 1024x1024) or instruct-pix2pix-00-22000.safetensors (fast, 512x512)."
                    },
                    "denoise": {
                        "type": "number",
                        "description": "Edit strength for source_image editing (0.0-1.0). Lower = subtler edits, higher = stronger transformation. Default 0.75. Only used with source_image.",
                        "default": 0.75
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of images to generate. Each gets a different seed (base seed + index). Default: 1.",
                        "default": 1
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path to save the image. If omitted, saves to a temp file. When count > 1, index is appended before extension (e.g. name_01.png, name_02.png)."
                    },
                    "return_image": {
                        "type": "string",
                        "enum": ["none", "thumb", "full"],
                        "description": "Whether to return image data inline. 'thumb' (default) returns a 256x256 JPEG preview — enough for the calling model to see what was produced. 'full' returns the complete image as JPEG — only when full-resolution comprehension matters (risks hitting context limits). 'none' returns only the file path — for mass-production runs where examination is deferred or handled by a separate agent. Default: 'thumb'.",
                        "default": "thumb"
                    }
                },
                "required": ["prompt"]
            }),
        },
        ToolDefinition {
            name: "inpaint_image".into(),
            description: "Inpaint part of an image using FLUX.1-Fill-dev. Provide a base image, a mask (white = replace, black = keep), and a DESCRIPTIVE prompt of what should appear in the masked region. Do NOT phrase as an edit instruction — Fill-dev is not instruction-tuned. Write 'a red leather hat with a feather', not 'add a red hat'. For removals, describe what should fill the void, not the removal — e.g. 'empty park bench, grass, dappled light', not 'remove the person'. Describe only the masked region; the rest of the image conditions the model automatically. Greyscale mask pixels give partial blending but prefer binary (pure black/white) PNG masks for predictable seams. Mask is read from the PNG's red channel (so a plain black-and-white mask image works — no alpha needed).".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_image": {
                        "type": "string",
                        "description": "Absolute path to the base image to inpaint."
                    },
                    "mask": {
                        "type": "string",
                        "description": "Absolute path to the mask PNG. White pixels are replaced, black pixels are kept. Read from the red channel."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Descriptive prompt of what appears in the masked region (not an edit instruction)."
                    },
                    "model": {
                        "type": "string",
                        "description": "FLUX-inpaint model to use (e.g. 'diffusion_models/flux1-fill-dev-Q8_0.gguf'). Defaults to the first available fill model. Non-inpaint models are rejected."
                    },
                    "lora": {
                        "type": "string",
                        "description": "Optional LoRA filename (applied on top of Fill-dev)."
                    },
                    "lora_strength": {
                        "type": "number",
                        "description": "LoRA strength (0.0-1.0).",
                        "default": 0.8
                    },
                    "steps": {
                        "type": "integer",
                        "description": "Sampling steps. Default 20.",
                        "default": 20
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Random seed. Omit for random."
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of inpaints to generate. Each gets a different seed (base + index).",
                        "default": 1
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path to save. Temp file if omitted. When count > 1, index is appended before extension."
                    },
                    "return_image": {
                        "type": "string",
                        "enum": ["none", "thumb", "full"],
                        "description": "Whether to return image data inline. 'thumb' (default) returns a 256x256 JPEG preview. 'full' returns the complete image as JPEG. 'none' returns only the file path. Default: 'thumb'.",
                        "default": "thumb"
                    }
                },
                "required": ["source_image", "mask", "prompt"]
            }),
        },
        ToolDefinition {
            name: "create_mask".into(),
            description: "Create an editable mask tied to a source image. Writes three files: the `.mask` state (ASON), a greyscale mask PNG at source dimensions (feed this as `mask` to `inpaint_image`), and a 512x512 annotated preview JPEG showing an 8x8 grid with row/column numbers 1-8, with masked cells tinted red. The tool response returns all three paths plus the preview inline so you can see the current state immediately; re-read the preview file with the Read tool at any time. Cells are toggleable by future tools; for now they are initialized per the `init` parameter (random by default). Red = will be regenerated by inpainting, unmarked = preserved.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "source_image": {
                        "type": "string",
                        "description": "Absolute path to the source image the mask will be applied to."
                    },
                    "path": {
                        "type": "string",
                        "description": "Optional absolute path for the `.mask` state file. If omitted, a temp-dir path is chosen and returned. `.mask` suffix is appended if missing."
                    },
                    "aspect_mode": {
                        "type": "string",
                        "enum": ["fit", "stretch"],
                        "description": "How the source image is rendered into the 512x512 preview. 'fit' (default) preserves aspect ratio with letterboxing; 'stretch' fills the inner 416x416 ignoring aspect ratio. Does NOT affect the final mask PNG, which always matches source dimensions.",
                        "default": "fit"
                    },
                    "init": {
                        "type": "string",
                        "enum": ["random", "none", "all", "checkerboard"],
                        "description": "Initial cell values. 'random' (default) gives a visibly meaningful starting preview. 'none' = all cells off, 'all' = all cells on, 'checkerboard' = alternating pattern.",
                        "default": "random"
                    },
                    "seed": {
                        "type": "integer",
                        "description": "Seed for init='random' (reproducible). Omit for fresh random."
                    }
                },
                "required": ["source_image"]
            }),
        },
        ToolDefinition {
            name: "read_image_metadata".into(),
            description: "Read generation metadata baked into a PNG file. Returns the text prompt and (if available) model, seed, sampler, steps, cfg, and LoRA for images generated by ComfyUI or Automatic1111. Use this to reference a past image — e.g. regenerate a variation with the same prompt, or see what settings produced a particular result.".into(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path to a PNG file."
                    }
                },
                "required": ["path"]
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
        "inpaint_image" => handle_inpaint_image(client, args).await,
        "create_mask" => crate::mask::handle_create_mask(args).await,
        "read_image_metadata" => handle_read_image_metadata(args).await,
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
    source_image: Option<String>,
    #[serde(default = "default_denoise")]
    denoise: f64,
    #[serde(default = "default_count")]
    count: u32,
    seed: Option<i64>,
    output_path: Option<String>,
    #[serde(default = "default_return_image")]
    return_image: String,
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
fn default_denoise() -> f64 {
    0.75
}
fn default_return_image() -> String {
    "thumb".into()
}
fn default_count() -> u32 {
    1
}

fn indexed_path(base: &std::path::Path, index: usize, pad: usize) -> PathBuf {
    let stem = base.file_stem().unwrap_or_default().to_string_lossy();
    let ext = base.extension().unwrap_or_default().to_string_lossy();
    let parent = base.parent().unwrap_or(std::path::Path::new("."));
    parent.join(format!("{stem}_{:0>pad$}.{ext}", index + 1))
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
    let is_edit = args.source_image.is_some();
    let is_ip2p = is_ip2p_model(&filename);

    // Validate: source_image requires an IP2P model
    if is_edit && !is_ip2p {
        bail!(
            "source_image editing requires an InstructPix2Pix-compatible model \
             (e.g. cosxl_edit.safetensors or instruct-pix2pix-00-22000.safetensors), \
             but got '{filename}'"
        );
    }

    // Apply model-type-appropriate defaults
    let (def_w, def_h, def_steps, def_cfg) = if is_ip2p {
        ip2p_defaults(&filename)
    } else if is_flux {
        (1024, 1024, 4, 1.0)
    } else {
        (512, 768, 30, 7.5)
    };

    let width = args.width.unwrap_or(def_w);
    let height = args.height.unwrap_or(def_h);
    let steps = args.steps.unwrap_or(def_steps);
    let cfg = if is_ip2p { def_cfg } else { args.cfg }; // IP2P models prefer their default cfg
    let seed = args.seed.unwrap_or_else(|| rand::random::<i64>().abs());

    // LoRA spec
    let lora_spec = args.lora.as_ref().map(|name| LoraSpec {
        name: name.clone(),
        strength: args.lora_strength,
    });

    info!(
        prompt = %args.prompt,
        model = format!("{folder}/{filename}"),
        mode = if is_edit { "edit" } else { "generate" },
        lora = args.lora.as_deref().unwrap_or("none"),
        size = format!("{width}x{height}"),
        steps = steps,
        seed = seed,
        "Processing image request"
    );

    // Batch generation loop
    let count = args.count.max(1);
    let pad = if count <= 1 { 1 } else { (count as f64).log10() as usize + 1 };

    // Determine base path for output files
    let base_path: PathBuf = match &args.output_path {
        Some(p) => PathBuf::from(p),
        None => {
            let temp_dir = std::env::temp_dir().join("comfy-ui-mcp");
            tokio::fs::create_dir_all(&temp_dir).await?;
            let batch_id = &format!("{:08x}", rand::random::<u32>());
            temp_dir.join(format!("gen_{batch_id}.png"))
        }
    };

    let mut results_json = Vec::new();
    let mut content = Vec::new();

    for i in 0..count {
        let iter_seed = seed.wrapping_add(i as i64);

        // Rebuild workflow with this iteration's seed
        let workflow = if is_edit {
            let source_path = args.source_image.as_ref().unwrap();
            let uploaded_name = client.upload_image(source_path).await?;
            build_ip2p_workflow(
                &filename, &uploaded_name, &args.prompt, &args.negative_prompt,
                steps, cfg, &args.sampler, iter_seed, args.denoise, lora_spec.as_ref(),
            )
        } else if is_flux {
            build_flux_workflow(
                &filename, &args.prompt, width, height, steps, iter_seed, lora_spec.as_ref(),
            )
        } else {
            build_checkpoint_workflow(
                &filename, &args.prompt, &args.negative_prompt,
                width, height, steps, args.cfg, &args.sampler, iter_seed, lora_spec.as_ref(),
            )
        };

        let prompt_id = client.submit_prompt(&workflow).await?;
        info!(prompt_id = %prompt_id, index = i + 1, count = count, "Submitted to ComfyUI");

        let outputs = client
            .wait_for_completion(&prompt_id, Duration::from_secs(300))
            .await?;

        let output = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("No image output found for image {}", i + 1))?;

        let image_bytes = client.fetch_image(output).await?;

        // Build indexed file path
        let file_path = indexed_path(&base_path, i as usize, pad);
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&file_path, &image_bytes).await?;

        let file_path_str = file_path.to_string_lossy().to_string();
        info!(path = %file_path_str, index = i + 1, bytes = image_bytes.len(), "Image saved");

        results_json.push(serde_json::json!({
            "file_path": file_path_str,
            "seed": iter_seed,
            "index": i + 1,
            "model": format!("{folder}/{filename}"),
            "mode": if is_edit { "edit" } else { "generate" },
            "lora": args.lora,
            "size": format!("{width}x{height}"),
        }));

        // Append inline image if requested
        match args.return_image.as_str() {
            "thumb" => {
                let img = image::load_from_memory(&image_bytes)?;
                let thumb = img.thumbnail(256, 256);
                let mut buf = std::io::Cursor::new(Vec::new());
                thumb.write_to(&mut buf, image::ImageFormat::Jpeg)?;
                content.push(ContentItem::Image {
                    data: BASE64.encode(buf.into_inner()),
                    mime_type: "image/jpeg".into(),
                });
            }
            "full" | "true" => {
                let img = image::load_from_memory(&image_bytes)?;
                let mut buf = std::io::Cursor::new(Vec::new());
                img.write_to(&mut buf, image::ImageFormat::Jpeg)?;
                content.push(ContentItem::Image {
                    data: BASE64.encode(buf.into_inner()),
                    mime_type: "image/jpeg".into(),
                });
            }
            _ => {}
        }
    }

    // Prepend the metadata array as the first content item
    content.insert(0, ContentItem::Text {
        text: serde_json::to_string(&results_json)?,
    });

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

// ── inpaint_image ──────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct InpaintImageArgs {
    source_image: String,
    mask: String,
    prompt: String,
    model: Option<String>,
    lora: Option<String>,
    #[serde(default = "default_lora_strength")]
    lora_strength: f64,
    #[serde(default = "default_inpaint_steps")]
    steps: u32,
    seed: Option<i64>,
    #[serde(default = "default_count")]
    count: u32,
    output_path: Option<String>,
    #[serde(default = "default_return_image")]
    return_image: String,
}

fn default_inpaint_steps() -> u32 {
    20
}

async fn handle_inpaint_image(client: &ComfyClient, args: &Value) -> Result<CallToolResult> {
    let args: InpaintImageArgs = serde_json::from_value(args.clone())?;

    // Resolve model — default to the first available fill model.
    let (folder, filename) = match args.model.as_deref() {
        Some(spec) if !spec.is_empty() => {
            let (f, n) = parse_model_specifier(spec)?;
            (f.to_string(), n.to_string())
        }
        _ => {
            let all = client.list_all_models().await?;
            let fill = all.iter().find_map(|full| {
                let (f, n) = full.split_once('/')?;
                is_flux_inpaint_model(n).then(|| (f.to_string(), n.to_string()))
            });
            fill.ok_or_else(|| anyhow::anyhow!(
                "No FLUX-inpaint model found. Download e.g. flux1-fill-dev-Q8_0.gguf into ComfyUI/models/diffusion_models/ and restart ComfyUI."
            ))?
        }
    };

    if !is_flux_inpaint_model(&filename) {
        bail!(
            "inpaint_image requires a FLUX-inpaint model (filename must contain 'flux' and 'fill'), but got '{filename}'. \
             For InstructPix2Pix or CosXL editing, use generate_image with source_image instead."
        );
    }

    let lora_spec = args.lora.as_ref().map(|name| LoraSpec {
        name: name.clone(),
        strength: args.lora_strength,
    });

    let seed = args.seed.unwrap_or_else(|| rand::random::<i64>().abs());
    let count = args.count.max(1);
    let pad = if count <= 1 { 1 } else { (count as f64).log10() as usize + 1 };

    info!(
        prompt = %args.prompt,
        model = format!("{folder}/{filename}"),
        lora = args.lora.as_deref().unwrap_or("none"),
        steps = args.steps,
        seed = seed,
        count = count,
        "Processing inpaint request"
    );

    let base_path: PathBuf = match &args.output_path {
        Some(p) => PathBuf::from(p),
        None => {
            let temp_dir = std::env::temp_dir().join("comfy-ui-mcp");
            tokio::fs::create_dir_all(&temp_dir).await?;
            let batch_id = &format!("{:08x}", rand::random::<u32>());
            temp_dir.join(format!("inpaint_{batch_id}.png"))
        }
    };

    // Upload source + mask once (same inputs for each iteration)
    let uploaded_base = client.upload_image(&args.source_image).await?;
    let uploaded_mask = client.upload_image(&args.mask).await?;

    let mut results_json = Vec::new();
    let mut content = Vec::new();

    for i in 0..count {
        let iter_seed = seed.wrapping_add(i as i64);

        let workflow = build_flux_inpaint_workflow(
            &filename,
            &uploaded_base,
            &uploaded_mask,
            &args.prompt,
            args.steps,
            iter_seed,
            lora_spec.as_ref(),
        );

        let prompt_id = client.submit_prompt(&workflow).await?;
        info!(prompt_id = %prompt_id, index = i + 1, count = count, "Submitted inpaint to ComfyUI");

        let outputs = client
            .wait_for_completion(&prompt_id, Duration::from_secs(300))
            .await?;
        let output = outputs
            .first()
            .ok_or_else(|| anyhow::anyhow!("No image output found for inpaint {}", i + 1))?;

        let image_bytes = client.fetch_image(output).await?;

        let file_path = indexed_path(&base_path, i as usize, pad);
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(&file_path, &image_bytes).await?;

        let file_path_str = file_path.to_string_lossy().to_string();
        info!(path = %file_path_str, index = i + 1, bytes = image_bytes.len(), "Inpaint saved");

        results_json.push(serde_json::json!({
            "file_path": file_path_str,
            "seed": iter_seed,
            "index": i + 1,
            "model": format!("{folder}/{filename}"),
            "lora": args.lora,
        }));

        match args.return_image.as_str() {
            "thumb" => {
                let img = image::load_from_memory(&image_bytes)?;
                let thumb = img.thumbnail(256, 256);
                let mut buf = std::io::Cursor::new(Vec::new());
                thumb.write_to(&mut buf, image::ImageFormat::Jpeg)?;
                content.push(ContentItem::Image {
                    data: BASE64.encode(buf.into_inner()),
                    mime_type: "image/jpeg".into(),
                });
            }
            "full" | "true" => {
                let img = image::load_from_memory(&image_bytes)?;
                let mut buf = std::io::Cursor::new(Vec::new());
                img.write_to(&mut buf, image::ImageFormat::Jpeg)?;
                content.push(ContentItem::Image {
                    data: BASE64.encode(buf.into_inner()),
                    mime_type: "image/jpeg".into(),
                });
            }
            _ => {}
        }
    }

    content.insert(
        0,
        ContentItem::Text {
            text: serde_json::to_string(&results_json)?,
        },
    );

    Ok(CallToolResult {
        content,
        is_error: false,
    })
}

// ── read_image_metadata ────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ReadImageMetadataArgs {
    path: String,
}

async fn handle_read_image_metadata(args: &Value) -> Result<CallToolResult> {
    let args: ReadImageMetadataArgs = serde_json::from_value(args.clone())?;
    let report = read_image_metadata(std::path::Path::new(&args.path))?;
    Ok(CallToolResult {
        content: vec![ContentItem::Text {
            text: serde_json::to_string_pretty(&report)?,
        }],
        is_error: false,
    })
}
