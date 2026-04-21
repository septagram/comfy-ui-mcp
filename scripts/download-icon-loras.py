"""
Download the 4 recommended LoRAs from the icon-polish research doc.
Civitai requires an API token — set CIVITAI_TOKEN env var before running.

Usage:
    export CIVITAI_TOKEN=... ; python scripts/download-icon-loras.py
"""
import os
import sys
import json
import urllib.request
from pathlib import Path

TARGETS_DIR = Path(__file__).resolve().parent.parent.parent / "ComfyUI" / "models" / "loras"

# (nickname, civitai model id or None if unknown)
MODELS = [
    ("ink-splash-v1",                1068512),  # Ink Splash - V1
    ("minimalist-line-style",        657321),   # Flux Minimalist Line Style
    ("minimalist-flat-color",        961950),   # Minimalist Flat Color Illustration
    # Celestial Dream: research doc has no numeric model id. Skip or resolve by search.
    # ("celestial-dream",              None),
]


def fetch_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def download_stream(url, out_path, token):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Authorization": f"Bearer {token}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as r, open(out_path, "wb") as f:
        while True:
            chunk = r.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            f.write(chunk)
    return out_path


def main():
    token = os.environ.get("CIVITAI_TOKEN")
    if not token:
        print("ERROR: CIVITAI_TOKEN env var not set. Get one at https://civitai.com/user/account (API Keys section).", file=sys.stderr)
        return 2

    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Download target: {TARGETS_DIR}")

    for nickname, model_id in MODELS:
        if model_id is None:
            print(f"SKIP {nickname}: no model id")
            continue
        try:
            info = fetch_json(f"https://civitai.com/api/v1/models/{model_id}")
            version = info["modelVersions"][0]
            file_meta = version["files"][0]
            download_url = file_meta["downloadUrl"]
            filename = file_meta["name"]
            size_kb = file_meta.get("sizeKB", 0)
            out_path = TARGETS_DIR / filename
            if out_path.exists():
                print(f"SKIP {nickname} ({filename}): already present")
                continue
            print(f"GET  {nickname} ({filename}, ~{size_kb/1024:.1f} MB)")
            download_stream(download_url, out_path, token)
            print(f"OK   {nickname} -> {out_path}")
        except Exception as e:
            print(f"FAIL {nickname}: {e}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main() or 0)
