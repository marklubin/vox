"""Configuration constants for Vox."""

import logging
import os
import sys
from pathlib import Path

# Logging setup
LOG_LEVEL = os.environ.get("VOX_LOG_LEVEL", "INFO").upper()
LOG_FILE = os.environ.get("VOX_LOG_FILE", "")

# Create logger
logger = logging.getLogger("vox")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Format with timestamps and module info
formatter = logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s.%(funcName)s: %(message)s",
    datefmt="%H:%M:%S",
)

# Console handler (stderr for visibility)
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Optional file handler
if LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a module.

    Args:
        name: Module name (e.g., 'hotkeys', 'audio')

    Returns:
        Logger instance for the module.
    """
    return logging.getLogger(f"vox.{name}")


# Paths
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
MIC_ON_ICON = ASSETS_DIR / "mic_on.png"
MIC_OFF_ICON = ASSETS_DIR / "mic_off.png"

# Audio settings
SAMPLE_RATE = 16000  # Hz - required by Moonshine
CHUNK_DURATION = 1.0  # seconds - balance between latency and quality
CHANNELS = 1  # mono
AUDIO_DTYPE = "float32"
BLOCK_SIZE = 1024  # samples per callback

# Transcription settings
# Transcriber backend is selected via VOX_TRANSCRIBER env var (see transcribe.py)

# Hotkey settings
HOLD_THRESHOLD = 0.25  # seconds - distinguish tap from hold
DOUBLE_TAP_THRESHOLD = 0.3  # seconds - max time between taps for double-tap

# Minimum audio length to process (in samples)
MIN_AUDIO_SAMPLES = 1600  # 100ms at 16kHz
