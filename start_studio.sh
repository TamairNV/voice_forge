#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Virtual environment not found. Running ./setup.sh first..."
  ./setup.sh
fi

FFMPEG_PATH="$(command -v ffmpeg || echo /opt/homebrew/bin/ffmpeg)"
export PATH="/opt/homebrew/bin:$PATH"

echo "=========================================================="
echo "  VoiceForge Studio — M4 Pro Local Voice Design & Acting"
echo "  Web UI:       http://127.0.0.1:7860"
echo "  Swagger Docs: http://127.0.0.1:7860/docs"
echo "  OpenAPI JSON: http://127.0.0.1:7860/openapi.json"
echo "=========================================================="
exec ./.venv/bin/uvicorn server:app --host 127.0.0.1 --port 7860
