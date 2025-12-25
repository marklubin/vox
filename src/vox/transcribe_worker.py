#!/usr/bin/env python3
"""Subprocess worker for MLX Whisper transcription.

This script runs in a separate process to isolate Metal GPU operations
from the main AppKit event loop, avoiding Metal threading crashes.

Usage:
    echo '<base64_audio>' | python -m vox.transcribe_worker [model_name]

The worker reads base64-encoded float32 audio from stdin and writes
the transcription result to stdout.
"""

import sys
import base64
import json


def main():
    """Run transcription on audio from stdin, output result to stdout."""
    # Get model name from args or use default
    model_name = sys.argv[1] if len(sys.argv) > 1 else "mlx-community/distil-whisper-large-v3"

    # Read base64-encoded audio from stdin
    audio_b64 = sys.stdin.read().strip()
    if not audio_b64:
        print(json.dumps({"error": "No audio data provided"}))
        return 1

    try:
        # Decode audio
        import numpy as np
        audio_bytes = base64.b64decode(audio_b64)
        audio_array = np.frombuffer(audio_bytes, dtype=np.float32)

        # Import and run MLX Whisper
        import mlx_whisper
        result = mlx_whisper.transcribe(
            audio_array,
            path_or_hf_repo=model_name,
        )

        text = result.get("text", "") if result else ""
        print(json.dumps({"text": text.strip()}))
        return 0

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
