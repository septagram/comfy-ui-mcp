# comfy-ui-mcp

A lightweight [MCP](https://modelcontextprotocol.io/) server that gives [Claude Code](https://claude.ai/code) the ability to generate and edit images through a local [ComfyUI](https://github.com/comfyanonymous/ComfyUI) instance.

Built in Rust. Communicates over stdio (JSON-RPC). Talks to ComfyUI's HTTP API to submit workflows, poll for completion, and fetch results.

## What it does

Three tools are exposed to Claude Code:

- **generate_image** - Text-to-image generation (SD 1.5, SDXL, FLUX) and instruction-based image editing (InstructPix2Pix, CosXL Edit). Supports LoRA, configurable parameters, and returns thumbnails or full images inline.
- **list_models** - Lists all available models across ComfyUI folders (checkpoints, LoRAs, GGUF, VAE, text encoders, etc.) with folder-prefixed names.
- **system_status** - GPU info, VRAM, ComfyUI version, queue status.

## Prerequisites

- **ComfyUI** running locally (default `http://127.0.0.1:8188`). See [docs/comfyui-setup-guide.md](docs/comfyui-setup-guide.md) for a step-by-step setup guide.
- **NVIDIA GPU** with CUDA support (tested on RTX 2060 SUPER and RTX 4060 Ti).
- **Claude Code** installed.

## Installing Rust

If you don't have Rust installed yet:

1. Install [rustup](https://rustup.rs/) (the Rust toolchain manager):
   - **Windows**: Download and run `rustup-init.exe` from https://rustup.rs
   - **macOS/Linux**: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

2. Install the nightly toolchain (this project requires it):
   ```bash
   rustup toolchain install nightly
   ```

3. Verify it works:
   ```bash
   rustc +nightly --version
   cargo +nightly --version
   ```

The project includes a `rust-toolchain.toml` that automatically selects nightly when you build.

## Building

```bash
git clone https://github.com/septagram/comfy-ui-mcp.git
cd comfy-ui-mcp
cargo build --release
```

The binary lands at `target/release/comfy-ui-mcp.exe` (Windows) or `target/release/comfy-ui-mcp` (Linux/macOS).

## Installing

The easiest way to put it on your PATH:

```bash
cargo install --path .
```

This copies the binary to `~/.cargo/bin/`, which should already be on your PATH if you installed Rust via rustup.

## Registering with Claude Code

Run:

```bash
comfy-ui-mcp --register
```

This adds a `comfy-ui` entry to `~/.claude.json` so Claude Code knows about the MCP server. **You need to restart Claude Code after registering** for the tools to appear.

To remove it later:

```bash
comfy-ui-mcp --unregister
```

### If registration fails

The `--register` command edits `~/.claude.json`. If something goes wrong, you can do it manually. Open `~/.claude.json` (create it if it doesn't exist) and add this under `"mcpServers"`:

```json
{
  "mcpServers": {
    "comfy-ui": {
      "type": "stdio",
      "command": "comfy-ui-mcp",
      "args": [],
      "env": {
        "COMFYUI_URL": "http://127.0.0.1:8188"
      }
    }
  }
}
```

If the file already has other content, just merge the `"comfy-ui"` key into the existing `"mcpServers"` object. Then restart Claude Code.

### Changing the ComfyUI URL

By default the server connects to `http://127.0.0.1:8188`. To change this, either:
- Set the `COMFYUI_URL` environment variable, or
- Edit the `env` section in `~/.claude.json` after registering, or
- Create a `.env` file in the directory where you run the server.

## Usage

Once registered and Claude Code is restarted, the tools are available in conversation. Some examples of what you can ask Claude:

- *"Generate a logo for a coffee shop"*
- *"List available models"*
- *"Generate an image of a sunset using FLUX"*
- *"Edit this image to make it look like winter"* (with a file path)

Claude will pick the right tool and parameters automatically.

## Windows Installer (MSI)

If you want to distribute to friends who don't have Rust installed:

```bash
cargo install cargo-wix
cargo wix -p comfy-ui-mcp
```

This produces an MSI at `target/wix/comfy-ui-mcp-X.Y.Z-x86_64.msi` that:
- Installs `comfy-ui-mcp.exe` to `Program Files\ComfyUI MCP\`
- Adds the install directory to the system PATH
- Automatically registers the MCP server in Claude Code
- Cleans everything up on uninstall

## Configuration

| Environment variable | Default | Description |
|---------------------|---------|-------------|
| `COMFYUI_URL` | `http://127.0.0.1:8188` | ComfyUI server address |
| `RUST_LOG` | `comfy_ui_mcp=info` | Log level (tracing). Logs go to stderr, never stdout. |

## License

[MIT](LICENSE)
