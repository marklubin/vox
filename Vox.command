#!/bin/bash
# Double-click this file to launch Vox
# Uses MLX Whisper for fast transcription on Apple Silicon

cd "$(dirname "$0")"
export VOX_TRANSCRIBER=mlx-whisper
exec uv run vox
