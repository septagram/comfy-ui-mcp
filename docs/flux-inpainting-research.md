# FLUX Inpainting — Research Notes

Research done 2026-04-19 to scope adding FLUX-based inpainting to `comfy-ui-mcp`. The existing editing path (`source_image` + IP2P/CosXL) is SD-family only — it can't route FLUX models and has no mask input. This document captures the inputs FLUX inpainting accepts and the model-variant comparison that shaped our choice.

Implementation plan lives separately; this is the background reading.

## Inputs FLUX inpainting accepts

Common across all variants:

- **Base image** — the image being edited. Uploaded to ComfyUI via `POST /upload/image` (same multipart endpoint `comfy-ui-mcp` already uses for IP2P editing).
- **Mask** — binary or greyscale image, white = replace, black = keep. Two delivery shapes exist in ComfyUI:
  - **Alpha-channel PNG**: `LoadImage` node outputs both `IMAGE` and `MASK` when the PNG has an alpha channel.
  - **Separate mask file**: load via `LoadImageMask` with a channel selector (typically red or alpha).
- **Prompt** — positive only. FLUX doesn't use negative prompts.
- **Seed / steps / width / height** — same semantics as FLUX txt2img. Width and height normally derived from the source image rather than user-specified.

## Variant comparison

### Variant 1 — FLUX.1-Fill-dev (chosen)

Purpose-built monolithic inpainting model from Black Forest Labs.

| | |
|---|---|
| HF repos | `black-forest-labs/FLUX.1-Fill-dev` (gated safetensors), `YarvixPA/FLUX.1-Fill-dev-GGUF` (ungated GGUF quants) |
| Format | Single file — safetensors 23.8 GB, or GGUF quantizations 5.2–12.7 GB |
| Loader | `UnetLoaderGGUF` (ComfyUI-GGUF custom node, already installed) for GGUF; `UNETLoader`/`CheckpointLoaderSimple` variant for safetensors |
| Peak VRAM | Comparable to FLUX.1-schnell Q8 — fits 16 GB |
| Extra knobs | None — denoise fixed at 1.0; Fill-dev handles masked regions internally |
| Conditioning | `InpaintModelConditioning` wiring (positive, negative, VAE, pixels, mask) → (positive, negative, latent) |
| License | FLUX.1 [dev] Non-Commercial |

GGUF quant sizes (YarvixPA):

| Quant | Size |
|---|---|
| Q3_K_S | 5.24 GB |
| Q4_K_S | 6.81 GB |
| Q5_K_S | 8.29 GB |
| **Q8_0** | **12.7 GB** |
| safetensors | 23.8 GB |

User-reported quality:

- Weaknesses at **seams and edges**; can produce soft or low-contrast results. GitHub issue ComfyUI#6765 documents recurring complaints.
- Prompt-sensitive — some users report quality keywords ("8K", "sharp details") materially help, which is consistent with the model relying more on the prompt than on ControlNet-style spatial steering.
- **Excels at outpainting** — community consensus on Civitai puts Fill-dev well ahead of Alimama for expanding canvas beyond the source image.

### Variant 2 — FLUX.1-dev + alimama-creative Inpainting ControlNet

Standard FLUX.1-dev base plus a ControlNet trained specifically for inpainting.

| | |
|---|---|
| HF repo | `alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta` |
| Format | 4.28 GB safetensors. **No GGUF quant exists.** |
| Base model requirement | **FLUX.1-dev is mandatory**. Schnell does not substitute. Marginal cost: 4.28 GB + 12–24 GB if dev isn't already on disk. |
| Peak VRAM | ~27 GB reported for 28-step inference at fp8. Does not fit 16 GB without further quantization work. |
| Extra knobs | `control_strength` (~0.9–0.95), `control_end_percent` (~0.35–1.0), `true_cfg` (1.0 fast / 3.5 quality) — all interact |
| Conditioning | `ControlNetLoader` + `ApplyControlNet`-variant wired into the FLUX sampler pipeline |
| License | FLUX.1 [dev] Non-Commercial (inherits from dev base) |

User-reported quality:

- Better for **selective edits** — keep a face, swap background; and for LoRA-heavy flows.
- **Bad at outpainting** — inverse of Fill-dev's strengths.

### Variant 3 — DIY: any FLUX + `DifferentialDiffusion` + `SetLatentNoiseMask`

Reuses existing FLUX.1-schnell Q8 GGUF with no new downloads. Quality gap is visible enough that it wasn't considered worth shipping. Noted here for completeness.

## What "ControlNet" means

A ControlNet is a small auxiliary network bolted onto a base diffusion model that steers generation with a spatial signal — edges, depth, a pose skeleton, or in this case a mask. You're giving the model a visual blueprint in addition to the text prompt. The appeal is **composability**: one base model + many ControlNets, swap or combine as needed. The cost: **more VRAM** (base + ControlNet both loaded) and **more interacting knobs** (`strength × end_percent` is a 2-D tuning space where settings don't transfer cleanly between tasks).

Fill-dev is the opposite philosophy — a single monolithic model trained end-to-end for inpainting, no external steering. Fewer moving parts, no tuning, but no composability if BFL didn't train it for your exact task (e.g. outpainting vs inpainting).

## Decision: Variant 1, Q8_0

- **Variant 2 won't run on the 4060 Ti's 16 GB VRAM** (27 GB reported requirement).
- **Variant 2 download is 16–28 GB** (ControlNet + dev base we don't have) vs 12.7 GB for Fill-dev Q8_0 alone.
- **Variant 1 adds zero knobs to the MCP schema** — user says "edit this region, here's the prompt", no `control_strength` tuning burden to expose to Claude.
- Q8_0 over Q5_K_S because we have the VRAM headroom and the quality margin is worth it for edits (seams are Fill-dev's weak point and quantization noise exacerbates that).

Revisit Variant 2 only if outpainting+selective-edit-with-LoRA becomes a felt need later.

## Prompting convention for Fill-dev

Fundamentally different from IP2P/CosXL. Sources consistently converge on these rules:

- **Descriptive, not instructional.** `"a red leather hat with a feather"`, not `"add a red hat"`. Fill-dev was trained on inpainting, not on instruction-following — imperatives confuse it.
- **Scope: only the masked region.** The unmasked pixels are conditioning input. Don't re-describe the whole scene; just say what fills the hole.
- **Natural language, present tense, concise.** Closer to FLUX txt2img style than SD 1.5 tag-soup. `"a black leather sofa in a sunlit living room"` > `"sofa, leather, black, sunlit, hdr"`.
- **No negative prompts.** Same as FLUX txt2img today.
- **Removal tasks**: describe what should fill the void, not the removal. Not `"remove the person"` — write `"empty park bench, grass, dappled light"`.
- **Style matching** happens automatically from the image context; prompt can include lighting/mood descriptors to nudge it.

Implication for the MCP tool schema: the current `generate_image` description tells Claude "the prompt becomes an edit instruction" when `source_image` is set — correct for IP2P/CosXL, wrong for Fill-dev. The schema description needs a branch so Claude phrases prompts descriptively for Fill-dev and imperatively for IP2P.

Sources: Black Forest Labs docs, ComfyUI Flux.1-Fill tutorial, Skywork FLUX prompting guide, Apatero Flux Fill guide, MimicPC Flux Fill guide (see URLs below).

## Sources

- [FLUX.1-Fill-dev HuggingFace (gated)](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev)
- [YarvixPA/FLUX.1-Fill-dev-GGUF (ungated quants)](https://huggingface.co/YarvixPA/FLUX.1-Fill-dev-GGUF)
- [alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta](https://huggingface.co/alimama-creative/FLUX.1-dev-Controlnet-Inpainting-Beta)
- [ComfyUI FLUX.1-Fill Docs](https://docs.comfy.org/tutorials/flux/flux-1-fill-dev)
- [Bad quality Flux Fill inpainting reports (ComfyUI #6765)](https://github.com/comfyanonymous/ComfyUI/issues/6765)
- [Alimama vs Fill user verdict on Civitai](https://civitai.com/models/862215/proper-flux-control-net-inpainting-andor-outpainting-with-batch-size-comfyui-alimama-or-flux-fill)
- [Black Forest Labs FLUX.1 Fill API docs](https://docs.bfl.ml/flux_tools/flux_1_fill)
- [Skywork FLUX.1 Prompting Ultimate Guide](https://skywork.ai/blog/flux-prompting-ultimate-guide-flux1-dev-schnell/)
- [Apatero Flux Fill Complete Guide 2025](https://apatero.com/blog/flux-fill-inpainting-outpainting-complete-guide-2025)
- [MimicPC FLUX.1 Fill Inpainting Guide](https://www.mimicpc.com/learn/flux1-tools-flux1-fill-inpainting-guide)
