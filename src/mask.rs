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

// ── 5×7 bitmap glyphs ────────────────────────────────────────────────────
// Digits 1-8 for row labels, letters A-H for column labels. Five bits per
// row in a u8; bit 4 = leftmost pixel, bit 0 = rightmost.

const DIGIT_W: u32 = 5;
const DIGIT_H: u32 = 7;
const DIGIT_SCALE: u32 = 2;

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

const LETTERS: [[u8; 7]; 8] = [
    // A
    [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    // B
    [0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110],
    // C
    [0b01111, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b01111],
    // D
    [0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110],
    // E
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111],
    // F
    [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000],
    // G
    [0b01110, 0b10001, 0b10000, 0b10011, 0b10001, 0b10001, 0b01110],
    // H
    [0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
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

/// Placement of the source image inside the 416×416 inner area, expressed in
/// inner-local coordinates (x=0, y=0 is the top-left of the inner area, not
/// the canvas). Stretch mode fills; fit mode letterboxes.
fn image_rect_inner(source: &DynamicImage, mode: AspectMode) -> (u32, u32, u32, u32) {
    match mode {
        AspectMode::Stretch => (0, 0, INNER_SIZE, INNER_SIZE),
        AspectMode::Fit => {
            let sw = source.width();
            let sh = source.height();
            let (w, h) = if sw >= sh {
                (
                    INNER_SIZE,
                    ((INNER_SIZE as u64 * sh as u64) / sw as u64).max(1) as u32,
                )
            } else {
                (
                    ((INNER_SIZE as u64 * sw as u64) / sh as u64).max(1) as u32,
                    INNER_SIZE,
                )
            };
            let x = (INNER_SIZE - w) / 2;
            let y = (INNER_SIZE - h) / 2;
            (x, y, w, h)
        }
    }
}

fn render_preview(source: &DynamicImage, mask: &Mask) -> Result<RgbaImage> {
    let mut canvas = RgbaImage::from_pixel(CANVAS_SIZE, CANVAS_SIZE, BG_COLOR);

    // Image goes where image_rect_inner says — centred in the inner area for
    // fit, filling it for stretch.
    let (iix, iiy, iw, ih) = image_rect_inner(source, mask.aspect_mode);
    let img_x = GUTTER + iix;
    let img_y = GUTTER + iiy;

    let rgba = source.to_rgba8();
    let resized = image::imageops::resize(&rgba, iw, ih, FilterType::Lanczos3);
    for (x, y, p) in resized.enumerate_pixels() {
        canvas.put_pixel(img_x + x, img_y + y, *p);
    }

    // Grid geometry is ALWAYS the full inner area, independent of aspect mode.
    // In fit mode this means some cells fall on the letterbox bands; that's by
    // design — masking those cells has no effect on the final mask.
    let cell_w = INNER_SIZE / mask.cols;
    let cell_h = INNER_SIZE / mask.rows;

    // Red overlay on masked cells
    for row in 0..mask.rows {
        for col in 0..mask.cols {
            let idx = (row * mask.cols + col) as usize;
            if mask.cells[idx] {
                let x0 = GUTTER + col * cell_w;
                let y0 = GUTTER + row * cell_h;
                blend_rect(&mut canvas, x0, y0, cell_w, cell_h, OVERLAY_COLOR);
            }
        }
    }

    // Grid lines over the full inner area
    for i in 0..=mask.cols {
        let x = (GUTTER + i * cell_w).min(GUTTER + INNER_SIZE - 1);
        draw_vline(&mut canvas, x, GUTTER, GUTTER + INNER_SIZE, GRID_COLOR);
    }
    for i in 0..=mask.rows {
        let y = (GUTTER + i * cell_h).min(GUTTER + INNER_SIZE - 1);
        draw_hline(&mut canvas, GUTTER, GUTTER + INNER_SIZE, y, GRID_COLOR);
    }

    // Labels on all four sides. Columns = letters A..H (top/bottom),
    // rows = numbers 1..8 (left/right). Scale autoshrinks if cells are too
    // small — not expected with fixed 52 px pitch, but keeps the code honest
    // if the grid gets larger later.
    let col_scale = digit_scale_for(cell_w, GUTTER);
    let row_scale = digit_scale_for(cell_h, GUTTER);

    let col_gw = DIGIT_W * col_scale;
    let col_gh = DIGIT_H * col_scale;
    let row_gw = DIGIT_W * row_scale;
    let row_gh = DIGIT_H * row_scale;

    let top_y = GUTTER.saturating_sub(col_gh) / 2;
    let bot_y = CANVAS_SIZE - GUTTER + GUTTER.saturating_sub(col_gh) / 2;
    for col in 0..mask.cols {
        let cell_mid_x = GUTTER + col * cell_w + cell_w / 2;
        let gx = cell_mid_x.saturating_sub(col_gw / 2);
        let glyph = &LETTERS[col as usize];
        draw_glyph(&mut canvas, gx, top_y, glyph, col_scale, LABEL_COLOR);
        draw_glyph(&mut canvas, gx, bot_y, glyph, col_scale, LABEL_COLOR);
    }

    let left_x = GUTTER.saturating_sub(row_gw) / 2;
    let right_x = CANVAS_SIZE - GUTTER + GUTTER.saturating_sub(row_gw) / 2;
    for row in 0..mask.rows {
        let cell_mid_y = GUTTER + row * cell_h + cell_h / 2;
        let gy = cell_mid_y.saturating_sub(row_gh / 2);
        let glyph = &DIGITS[row as usize];
        draw_glyph(&mut canvas, left_x, gy, glyph, row_scale, LABEL_COLOR);
        draw_glyph(&mut canvas, right_x, gy, glyph, row_scale, LABEL_COLOR);
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

/// Draw a 5×7 bitmap glyph at `(x, y)`. `scale` is the per-pixel block size —
/// pass 1 for native 5×7, 2 for 10×14, etc.
fn draw_glyph(
    canvas: &mut RgbaImage,
    x: u32,
    y: u32,
    glyph: &[u8; 7],
    scale: u32,
    color: Rgba<u8>,
) {
    if scale == 0 {
        return;
    }
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

fn render_final_mask(source: &DynamicImage, mask: &Mask) -> ImageBuffer<Luma<u8>, Vec<u8>> {
    let sw = source.width();
    let sh = source.height();
    let mut buf = ImageBuffer::from_pixel(sw, sh, Luma([0u8]));

    match mask.aspect_mode {
        AspectMode::Stretch => {
            // Cells map directly onto an evenly divided source: no letterbox
            // to worry about, same as the preview.
            for row in 0..mask.rows {
                for col in 0..mask.cols {
                    let idx = (row * mask.cols + col) as usize;
                    if !mask.cells[idx] {
                        continue;
                    }
                    let x0 = col * sw / mask.cols;
                    let y0 = row * sh / mask.rows;
                    let x1 = (col + 1) * sw / mask.cols;
                    let y1 = (row + 1) * sh / mask.rows;
                    fill_luma(&mut buf, x0, y0, x1, y1);
                }
            }
        }
        AspectMode::Fit => {
            // Grid lives over the full 416×416 inner area but the source image
            // is letterboxed inside. For each masked cell: intersect the cell
            // rect (in inner/preview coords) with the image rect, then map the
            // intersection back to source coords. Cells that fall entirely on
            // the letterbox contribute nothing.
            let (iix, iiy, iw, ih) = image_rect_inner(source, AspectMode::Fit);
            let cell_w = INNER_SIZE / mask.cols;
            let cell_h = INNER_SIZE / mask.rows;
            let img_x1 = iix + iw;
            let img_y1 = iiy + ih;

            for row in 0..mask.rows {
                for col in 0..mask.cols {
                    let idx = (row * mask.cols + col) as usize;
                    if !mask.cells[idx] {
                        continue;
                    }
                    let cx0 = col * cell_w;
                    let cy0 = row * cell_h;
                    let cx1 = (col + 1) * cell_w;
                    let cy1 = (row + 1) * cell_h;

                    let ix0 = cx0.max(iix);
                    let iy0 = cy0.max(iiy);
                    let ix1 = cx1.min(img_x1);
                    let iy1 = cy1.min(img_y1);
                    if ix0 >= ix1 || iy0 >= iy1 {
                        continue;
                    }

                    let sx0 = ((ix0 - iix) as u64 * sw as u64 / iw as u64) as u32;
                    let sy0 = ((iy0 - iiy) as u64 * sh as u64 / ih as u64) as u32;
                    let sx1 = (((ix1 - iix) as u64 * sw as u64 / iw as u64) as u32).min(sw);
                    let sy1 = (((iy1 - iiy) as u64 * sh as u64 / ih as u64) as u32).min(sh);
                    fill_luma(&mut buf, sx0, sy0, sx1, sy1);
                }
            }
        }
    }

    buf
}

fn fill_luma(buf: &mut ImageBuffer<Luma<u8>, Vec<u8>>, x0: u32, y0: u32, x1: u32, y1: u32) {
    for y in y0..y1 {
        for x in x0..x1 {
            buf.put_pixel(x, y, Luma([255]));
        }
    }
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
            let final_mask = render_final_mask(&source, &mask_for_render);
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
