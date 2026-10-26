#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "=========================================================="
echo "  Setting up VoiceForge Studio (Apple Silicon MLX)"
echo "=========================================================="

# 1. Check ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[!] ffmpeg not found. Installing via Homebrew..."
  brew install ffmpeg
fi

# 2. Pick Python 3.11+ (prefer python3.11 or python3.12 for MLX wheels)
PYTHON_BIN="python3"
if command -v /opt/homebrew/bin/python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="/opt/homebrew/bin/python3.11"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
fi

echo "[1/3] Creating virtual environment using $PYTHON_BIN..."
if [ ! -d ".venv" ]; then
  "$PYTHON_BIN" -m venv .venv
fi

echo "[2/3] Installing Python dependencies (mlx-audio, FastAPI, Uvicorn)..."
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

echo "[3/3] Pre-downloading MLX Models (Qwen3-TTS bf16 & Maya-1 3.3B MLX)..."
./.venv/bin/python -c "
from huggingface_hub import snapshot_download
for repo in [
    'mlx-community/snac_24khz',
    'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16',
    'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16',
    'nhe-ai/maya1-mlx-8Bit',
]:
    print('  -> Checking/Downloading:', repo)
    snapshot_download(repo)
print('All models cached and ready!')
"

echo "=========================================================="
echo "  Setup Complete! Start the studio anytime with:"
echo "  ./start_studio.sh"
echo "=========================================================="
