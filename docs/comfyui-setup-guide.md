# ComfyUI Setup Guide (Windows, Native)

Step-by-step guide for setting up ComfyUI with GPU support on Windows, using `uv` instead of legacy pip. Based on a fresh install done 2025-03-25.

## Prerequisites

- **Python 3.12** installed (e.g. from python.org)
- **uv** installed (`pip install uv` or via chocolatey)
- **NVIDIA GPU** with recent drivers (tested with driver 560.94, RTX 2060 SUPER + RTX 4060 Ti)
- **Git**

## 1. Clone ComfyUI

```bash
cd C:\Users\<you>\Documents\Code\image-gen
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
```

## 2. Create venv with uv

```bash
uv venv --python 3.12 .venv
```

## 3. Install PyTorch with CUDA

**Important:** Install PyTorch FIRST with the CUDA index, before the rest of the requirements. Otherwise you'll get the CPU-only build.

```bash
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA works:
```bash
.venv\Scripts\python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

## 4. Install ComfyUI dependencies

```bash
uv pip install -r requirements.txt
```

This installs ~70 packages including transformers, safetensors, aiohttp, etc.

## 5. Launch and verify

```bash
.venv\Scripts\python main.py
```

You should see:
- `Set vram state to: NORMAL_VRAM`
- `Device: cuda:0 NVIDIA GeForce RTX <your GPU>`
- `To see the GUI go to: http://127.0.0.1:8188`

### Known warning

```
WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.
```

This is **harmless** — GPU is still fully used. It just means some bleeding-edge CUDA kernels aren't available yet. Everything works fine with cu128.

## 6. Models

Models go in `ComfyUI/models/<type>/`. The main folders:

| Folder | What goes here |
|--------|---------------|
| `checkpoints/` | Full SD/SDXL model files (.ckpt, .safetensors) |
| `loras/` | LoRA adapter files |
| `diffusion_models/` | Diffusion model files (non-GGUF) |
| `vae/` | VAE files |
| `text_encoders/` | CLIP/T5 text encoder files |

ComfyUI hot-reloads checkpoints and LoRAs (no restart needed). Other model types may need a restart.

---

## FLUX.1-schnell Setup (GGUF Quantized)

FLUX is a newer, faster model (4 steps vs 30 for SD) with better prompt adherence. The GGUF quantized version fits in 16GB VRAM.

### Step 1: Install ComfyUI-GGUF custom node

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/city96/ComfyUI-GGUF.git
cd ..
uv pip install -r custom_nodes/ComfyUI-GGUF/requirements.txt
```

**Version gotcha:** The `gguf` Python package version matters. As of 2025-03, `gguf>=0.13.0` is required. If you get reshape errors when loading models, try pinning: `uv pip install "gguf==0.16.0"`. Too old won't work, too new might break tensor loading.

### Step 2: Download the GGUF model

The model file is ~12.7 GB. From HuggingFace repo `city96/FLUX.1-schnell-gguf`:

```bash
.venv\Scripts\python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('city96/FLUX.1-schnell-gguf', 'flux1-schnell-Q8_0.gguf', local_dir='models/diffusion_models')
"
```

**Important:** Verify the file size after download! It should be **~12.69 GB**. We got a truncated 8.2 GB file on first attempt which caused `ValueError: cannot reshape array` errors. If the size is wrong, delete and re-download.

### Step 3: Download text encoders

FLUX requires two CLIP text encoders. These are on `comfyanonymous/flux_text_encoders` (no auth needed):

```bash
.venv\Scripts\python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('comfyanonymous/flux_text_encoders', 'clip_l.safetensors', local_dir='models/text_encoders')
hf_hub_download('comfyanonymous/flux_text_encoders', 't5xxl_fp8_e4m3fn.safetensors', local_dir='models/text_encoders')
"
```

- `clip_l.safetensors` — ~235 MB
- `t5xxl_fp8_e4m3fn.safetensors` — ~4.6 GB (fp8 quantized, fits in VRAM)

### Step 4: Download FLUX VAE

The VAE is on the **gated** `black-forest-labs/FLUX.1-schnell` repo. You need a (free) HuggingFace account:

1. Go to https://huggingface.co/black-forest-labs/FLUX.1-schnell and accept the license
2. Log in:
   ```bash
   .venv\Scripts\hf.exe auth login
   ```
   Paste a read-only token from https://huggingface.co/settings/tokens
3. Download:
   ```bash
   .venv\Scripts\python -c "
   from huggingface_hub import hf_hub_download
   hf_hub_download('black-forest-labs/FLUX.1-schnell', 'ae.safetensors', local_dir='models/vae')
   "
   ```
   - `ae.safetensors` — ~320 MB

### Step 5: Restart ComfyUI

The GGUF custom node only loads on startup. After restart, verify:

```bash
curl http://127.0.0.1:8188/object_info/UnetLoaderGGUF
```

Should return the node info with `flux1-schnell-Q8_0.gguf` in the available models.

### FLUX generation settings

- **Steps:** 4 (yes, just 4 — that's FLUX's strength)
- **Resolution:** 1024x1024 native
- **Sampler:** euler
- **Scheduler:** simple
- **No negative prompt** — FLUX doesn't use one
- **No CFG scale** — FLUX uses a guidance-free architecture

### Total download sizes

| File | Size | Auth required? |
|------|------|---------------|
| flux1-schnell-Q8_0.gguf | 12.7 GB | No |
| t5xxl_fp8_e4m3fn.safetensors | 4.6 GB | No |
| ae.safetensors (VAE) | 320 MB | Yes (free HF account) |
| clip_l.safetensors | 235 MB | No |
| **Total** | **~17.9 GB** | |

---

## FLUX.1-Fill-dev Setup (Inpainting, GGUF Quantized)

Fill-dev is a purpose-built inpainting variant of FLUX. It reuses the same text encoders and VAE as schnell — only the unet differs. Community Q8_0 GGUF quant fits in 16 GB VRAM.

Source repo: `YarvixPA/FLUX.1-Fill-dev-GGUF` (no auth required, not gated).

```bash
.venv\Scripts\python -c "from huggingface_hub import hf_hub_download; hf_hub_download('YarvixPA/FLUX.1-Fill-dev-GGUF', 'flux1-fill-dev-Q8_0.gguf', local_dir='models/diffusion_models')"
```

~12.7 GB. Lives in the same `models/diffusion_models/` folder as schnell and is loaded via `UnetLoaderGGUF` from the existing ComfyUI-GGUF custom node. Restart ComfyUI so the loader sees it.

The MCP server's `inpaint_image` tool auto-detects this model by filename (matches `flux*fill*`).

Smaller/larger quants are available in the same repo (Q3_K_S 5.2 GB → Q8_0 12.7 GB → safetensors 23.8 GB); Q8_0 is the highest-fidelity option that still fits 16 GB VRAM.

---

## Troubleshooting

### `ValueError: cannot reshape array of size X into shape (Y,Z)`
The GGUF model file is corrupt or truncated. Verify file size matches expected (12.69 GB for Q8_0). Delete and re-download.

### `WARNING: You need pytorch with cu130 or higher`
Harmless. GPU still works. Will go away when PyTorch ships cu130 wheels.

### Models not appearing in API
- Checkpoints/LoRAs: hot-reload, no restart needed
- GGUF models: only visible through `UnetLoaderGGUF` node, not via `/models/diffusion_models`
- Text encoders/VAE: may need ComfyUI restart

### HuggingFace 401 Unauthorized
The model repo is gated. Accept the license on the HuggingFace model page and log in with `hf.exe auth login`.

### `huggingface-cli` not found
On Windows with uv venv, the CLI is `hf.exe` not `huggingface-cli`. Located at `.venv\Scripts\hf.exe`.
