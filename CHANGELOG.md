## 1.1.0

- Add image editing via InstructPix2Pix / CosXL Edit (`source_image` parameter)
- Add `denoise` parameter to control edit strength (0.0–1.0)
- Add thumbnail option to `return_image` (`none`/`thumb`/`full`), switch inline images to JPEG for smaller context footprint
- Add README, MIT license, CHANGELOG, and ComfyUI setup guide

## 1.0.0

- Replace `list_checkpoints` with `list_models` — scans all model folders, returns `folder/filename` format
- Add FLUX.1-schnell GGUF workflow support (`UnetLoaderGGUF` + `DualCLIPLoader` + advanced sampler pipeline)
- Add LoRA support with configurable strength
- Add WiX MSI installer with PATH setup and auto MCP registration
- Add `--register` / `--unregister` CLI flags for Claude Code MCP setup
- Input validation: whitelist model folders, reject path traversal

## 0.1.0

- Initial release: stdio JSON-RPC MCP server for ComfyUI
- `generate_image` tool with SD 1.5 / SDXL checkpoint workflows
- `list_checkpoints` tool
- `system_status` tool
