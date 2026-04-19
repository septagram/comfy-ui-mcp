use anyhow::{Context, Result};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use image::{
    DynamicImage, ImageBuffer, ImageFormat, Luma, Rgba, RgbaImage, imageops::FilterType,
};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use tracing::info;

use crate::mcp::{CallToolResult, ContentItem};

// ── Preview canvas constants ───────────────────────────────────────────

const CANVAS_SIZE: u32 = 512;
const GUTTER: u32 = 48;
const INNER_SIZE: u32 = CANVAS_SIZE - 2 * GUTTER; // 416

const BG_COLOR: Rgba<u8> = Rgba([32, 32, 32, 255]);
const GRID_COLOR: Rgba<u8> = Rgba([180, 180, 180, 192]);
const LABEL_COLOR: Rgba<u8> = Rgba([230, 230, 230, 255]);
const OVERLAY_COLOR: Rgba<u8> = Rgba([255, 0, 0, 96]);

// ── 5×7 bitmap digits for '1'..'8' ──────────────────────────────────────
// Each row is 5 bits in a u8; bit 4 = leftmost pixel, bit 0 = rightmost.

const DIGIT_W: u32 = 5;
const DIGIT_H: u32 = 7;
const DIGIT_SCALE: u32 = 4;

const DIGITS: [[u8; 7]; 8] = [
    // 1
    [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    // 2
    [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    // 3
    [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
    // 4
    [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    // 5
    [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    // 6
    [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
    // 7
    [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
    // 8
    [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
];

// ── Data model ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AspectMode {
    Fit,
    Stretch,
}

impl Default for AspectMode {
    fn default() -> Self {
        AspectMode::Fit
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mask {
    pub source_image: String,
    pub cols: u32,
    pub rows: u32,
    pub aspect_mode: AspectMode,
    /// Row-major: `cells[row * cols + col]`. `true` = masked (will be replaced).
    pub cells: Vec<bool>,
}

/// Wire format for the `.mask` file. Future versions add variants; `load_mask`
/// upgrades whichever version is on disk to the latest `Mask`.
#[derive(Debug, Serialize, Deserialize)]
pub enum MaskFile {
    V1(Mask),
}

pub async fn save_mask(path: &Path, mask: &Mask) -> Result<()> {
    let file = MaskFile::V1(mask.clone());
    let text = ason::ser::ser_to_string(&file).context("ASON serialization failed")?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path, text).await?;
    Ok(())
}

#[allow(dead_code)]
pub async fn load_mask(path: &Path) -> Result<Mask> {
    let text = tokio::fs::read_to_string(path).await?;
    let file: MaskFile = ason::de::de_from_str(&text).context("ASON deserialization failed")?;
    match file {
        MaskFile::V1(m) => Ok(m),
    }
}

// ── Preview rendering (512×512) ─────────────────────────────────────────

fn compute_image_rect(source: &DynamicImage, mode: AspectMode) -> (u32, u32, u32, u32) {
    match mode {
        AspectMode::Stretch => (GUTTER, GUTTER, INNER_SIZE, INNER_SIZE),
        AspectMode::Fit => {
            let sw = source.width();
            let sh = source.height();
            let (new_w, new_h) = if sw >= sh {
                (INNER_SIZE, ((INNER_SIZE as u64 * sh as u64) / sw as u64).max(1) as u32)
            } else {
                (((INNER_SIZE as u64 * sw as u64) / sh as u64).max(1) as u32, INNER_SIZE)
            };
            let x = GUTTER + (INNER_SIZE - new_w) / 2;
            let y = GUTTER + (INNER_SIZE - new_h) / 2;
            (x, y, new_w, new_h)
        }
    }
}

fn render_preview(source: &DynamicImage, mask: &Mask) -> Result<RgbaImage> {
    let mut canvas = RgbaImage::from_pixel(CANVAS_SIZE, CANVAS_SIZE, BG_COLOR);

    let (img_x, img_y, img_w, img_h) = compute_image_rect(source, mask.aspect_mode);

    // Resize source to target size, then blit into canvas
    let rgba = source.to_rgba8();
    let resized = image::imageops::resize(&rgba, img_w, img_h, FilterType::Lanczos3);
    for (x, y, p) in resized.enumerate_pixels() {
        canvas.put_pixel(img_x + x, img_y + y, *p);
    }

    // Red overlay on masked cells
    for row in 0..mask.rows {
        for col in 0..mask.cols {
            let idx = (row * mask.cols + col) as usize;
            if mask.cells[idx] {
                let x0 = img_x + col * img_w / mask.cols;
                let y0 = img_y + row * img_h / mask.rows;
                let x1 = img_x + (col + 1) * img_w / mask.cols;
                let y1 = img_y + (row + 1) * img_h / mask.rows;
                blend_rect(&mut canvas, x0, y0, x1 - x0, y1 - y0, OVERLAY_COLOR);
            }
        }
    }

    // Grid lines spanning the actual image rect (not the canvas)
    for i in 0..=mask.cols {
        let x = (img_x + i * img_w / mask.cols).min(img_x + img_w - 1);
        draw_vline(&mut canvas, x, img_y, img_y + img_h, GRID_COLOR);
    }
    for i in 0..=mask.rows {
        let y = (img_y + i * img_h / mask.rows).min(img_y + img_h - 1);
        draw_hline(&mut canvas, img_x, img_x + img_w, y, GRID_COLOR);
    }

    // Gutter labels. Autoscale digits so they don't overflow the cell pitch
    // (labels overlapping each other when the source is heavily letterboxed is
    // worse than small digits).
    let cell_w = img_w / mask.cols;
    let cell_h = img_h / mask.rows;
    let col_scale = digit_scale_for(cell_w, GUTTER);
    let row_scale = digit_scale_for(cell_h, GUTTER);

    for col in 0..mask.cols {
        let cell_mid_x = img_x + col * img_w / mask.cols + cell_w / 2;
        let dw = DIGIT_W * col_scale;
        let dh = DIGIT_H * col_scale;
        let dx = cell_mid_x.saturating_sub(dw / 2);
        let dy = GUTTER.saturating_sub(dh) / 2;
        draw_digit(&mut canvas, dx, dy, col as usize, col_scale, LABEL_COLOR);
    }
    for row in 0..mask.rows {
        let cell_mid_y = img_y + row * img_h / mask.rows + cell_h / 2;
        let dw = DIGIT_W * row_scale;
        let dh = DIGIT_H * row_scale;
        let dx = GUTTER.saturating_sub(dw) / 2;
        let dy = cell_mid_y.saturating_sub(dh / 2);
        draw_digit(&mut canvas, dx, dy, row as usize, row_scale, LABEL_COLOR);
    }

    Ok(canvas)
}

// ── Drawing helpers ─────────────────────────────────────────────────────

fn fill_rect(canvas: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, color: Rgba<u8>) {
    for dy in 0..h {
        for dx in 0..w {
            let px = x + dx;
            let py = y + dy;
            if px < canvas.width() && py < canvas.height() {
                canvas.put_pixel(px, py, color);
            }
        }
    }
}

fn blend_pixel(canvas: &mut RgbaImage, x: u32, y: u32, color: Rgba<u8>) {
    if x >= canvas.width() || y >= canvas.height() {
        return;
    }
    let a = color[3] as u32;
    let inv = 255 - a;
    let dst = *canvas.get_pixel(x, y);
    let r = ((color[0] as u32 * a + dst[0] as u32 * inv) / 255) as u8;
    let g = ((color[1] as u32 * a + dst[1] as u32 * inv) / 255) as u8;
    let b = ((color[2] as u32 * a + dst[2] as u32 * inv) / 255) as u8;
    canvas.put_pixel(x, y, Rgba([r, g, b, 255]));
}

fn blend_rect(canvas: &mut RgbaImage, x: u32, y: u32, w: u32, h: u32, color: Rgba<u8>) {
    for dy in 0..h {
        for dx in 0..w {
            blend_pixel(canvas, x + dx, y + dy, color);
        }
    }
}

fn draw_hline(canvas: &mut RgbaImage, x0: u32, x1: u32, y: u32, color: Rgba<u8>) {
    let (a, b) = (x0.min(x1), x0.max(x1));
    for x in a..b {
        blend_pixel(canvas, x, y, color);
    }
}

fn draw_vline(canvas: &mut RgbaImage, x: u32, y0: u32, y1: u32, color: Rgba<u8>) {
    let (a, b) = (y0.min(y1), y0.max(y1));
    for y in a..b {
        blend_pixel(canvas, x, y, color);
    }
}

/// Draw a digit. `digit_idx` is 0..=7 for the digits '1'..'8'. `scale` is the
/// per-pixel block size — pass 1 for native 5×7, 4 for 20×28, etc.
fn draw_digit(canvas: &mut RgbaImage, x: u32, y: u32, digit_idx: usize, scale: u32, color: Rgba<u8>) {
    if digit_idx >= DIGITS.len() || scale == 0 {
        return;
    }
    let glyph = &DIGITS[digit_idx];
    for (row, bits) in glyph.iter().enumerate() {
        for col in 0..DIGIT_W {
            let bit = 1u8 << (DIGIT_W - 1 - col);
            if bits & bit != 0 {
                let px = x + col * scale;
                let py = y + (row as u32) * scale;
                fill_rect(canvas, px, py, scale, scale, color);
            }
        }
    }
}

/// Pick a digit scale that fits both the cell pitch and the gutter depth.
/// `pitch` is the cell-to-cell distance along the label's stacking axis;
/// `gutter` is the perpendicular depth. Clamped to [1, DIGIT_SCALE].
fn digit_scale_for(pitch: u32, gutter: u32) -> u32 {
    let by_pitch = pitch / DIGIT_H;
    let by_gutter = gutter / DIGIT_H;
    by_pitch.min(by_gutter).min(DIGIT_SCALE).max(1)
}

// ── Final mask (source dimensions, greyscale) ──────────────────────────

fn render_final_mask(src_w: u32, src_h: u32, mask: &Mask) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let mut buf = ImageBuffer::from_pixel(src_w, src_h, Luma([0u8]));
    for row in 0..mask.rows {
        for col in 0..mask.cols {
            let idx = (row * mask.cols + col) as usize;
            if !mask.cells[idx] {
                continue;
            }
            let x0 = col * src_w / mask.cols;
            let y0 = row * src_h / mask.rows;
            let x1 = (col + 1) * src_w / mask.cols;
            let y1 = (row + 1) * src_h / mask.rows;
            for y in y0..y1 {
                for x in x0..x1 {
                    buf.put_pixel(x, y, Luma([255]));
                }
            }
        }
    }
    buf
}

// ── Encoders ────────────────────────────────────────────────────────────

fn encode_jpeg(img: RgbaImage) -> Result<Vec<u8>> {
    // JPEG has no alpha — convert to RGB.
    let rgb = DynamicImage::ImageRgba8(img).to_rgb8();
    let mut buf = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 85);
    encoder.encode_image(&rgb)?;
    Ok(buf)
}

fn encode_png_luma(img: ImageBuffer<Luma<u8>, Vec<u8>>) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    let mut cursor = Cursor::new(&mut buf);
    DynamicImage::ImageLuma8(img).write_to(&mut cursor, ImageFormat::Png)?;
    Ok(buf)
}

// ── Handler ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct CreateMaskArgs {
    source_image: String,
    path: Option<String>,
    #[serde(default)]
    aspect_mode: AspectMode,
    #[serde(default = "default_init")]
    init: String,
    seed: Option<u64>,
}

fn default_init() -> String {
    "random".into()
}

pub async fn handle_create_mask(args: &Value) -> Result<CallToolResult> {
    let args: CreateMaskArgs = serde_json::from_value(args.clone())?;

    let (mask_path, preview_path, final_mask_path) = resolve_paths(args.path.as_deref()).await?;

    let source_bytes = tokio::fs::read(&args.source_image)
        .await
        .with_context(|| format!("Failed to read source image: {}", args.source_image))?;

    let cols = 8u32;
    let rows = 8u32;
    let cells = init_cells(&args.init, cols, rows, args.seed)?;

    let mask = Mask {
        source_image: args.source_image.clone(),
        cols,
        rows,
        aspect_mode: args.aspect_mode,
        cells,
    };

    save_mask(&mask_path, &mask).await?;

    info!(
        source = %args.source_image,
        mask_path = %mask_path.display(),
        init = %args.init,
        aspect = ?mask.aspect_mode,
        "Rendering mask sidecars"
    );

    let mask_for_render = mask.clone();
    let (preview_jpeg, final_mask_png) = tokio::task::spawn_blocking(
        move || -> Result<(Vec<u8>, Vec<u8>)> {
            let source = image::load_from_memory(&source_bytes)
                .context("Failed to decode source image")?;
            let preview = render_preview(&source, &mask_for_render)?;
            let final_mask = render_final_mask(source.width(), source.height(), &mask_for_render);
            Ok((encode_jpeg(preview)?, encode_png_luma(final_mask)?))
        },
    )
    .await??;

    tokio::try_join!(
        tokio::fs::write(&preview_path, &preview_jpeg),
        tokio::fs::write(&final_mask_path, &final_mask_png),
    )?;

    let aspect_str = match mask.aspect_mode {
        AspectMode::Fit => "fit",
        AspectMode::Stretch => "stretch",
    };
    let response = serde_json::json!({
        "mask_path": mask_path.to_string_lossy(),
        "preview_path": preview_path.to_string_lossy(),
        "final_mask_path": final_mask_path.to_string_lossy(),
        "source_image": mask.source_image,
        "cols": mask.cols,
        "rows": mask.rows,
        "aspect_mode": aspect_str,
    });

    Ok(CallToolResult {
        content: vec![
            ContentItem::Text {
                text: serde_json::to_string(&response)?,
            },
            ContentItem::Image {
                data: BASE64.encode(&preview_jpeg),
                mime_type: "image/jpeg".into(),
            },
        ],
        is_error: false,
    })
}

async fn resolve_paths(user_path: Option<&str>) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let mask_path: PathBuf = match user_path {
        Some(p) if !p.is_empty() => {
            let p = PathBuf::from(p);
            if p.extension().and_then(|e| e.to_str()) == Some("mask") {
                p
            } else {
                PathBuf::from(format!("{}.mask", p.to_string_lossy()))
            }
        }
        _ => {
            let tmp = std::env::temp_dir().join("comfy-ui-mcp");
            tokio::fs::create_dir_all(&tmp).await?;
            let id = format!("{:08x}", rand::random::<u32>());
            tmp.join(format!("mask_{id}.mask"))
        }
    };

    let mask_str = mask_path.to_string_lossy().to_string();
    let preview_path = PathBuf::from(format!("{mask_str}.preview.jpg"));
    let final_mask_path = PathBuf::from(format!("{mask_str}.png"));

    Ok((mask_path, preview_path, final_mask_path))
}

fn init_cells(init: &str, cols: u32, rows: u32, seed: Option<u64>) -> Result<Vec<bool>> {
    let total = (cols * rows) as usize;
    match init {
        "none" => Ok(vec![false; total]),
        "all" => Ok(vec![true; total]),
        "checkerboard" => Ok((0..total)
            .map(|i| {
                let row = i as u32 / cols;
                let col = i as u32 % cols;
                (row + col) % 2 == 0
            })
            .collect()),
        "random" => match seed {
            Some(s) => {
                let mut rng = StdRng::seed_from_u64(s);
                Ok((0..total).map(|_| rng.random_bool(0.5)).collect())
            }
            None => Ok((0..total).map(|_| rand::random::<bool>()).collect()),
        },
        _ => anyhow::bail!(
            "Unknown init mode '{init}'. Allowed: random, none, all, checkerboard"
        ),
    }
}
