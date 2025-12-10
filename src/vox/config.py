"""Configuration constants for Vox."""

from pathlib import Path

# Paths
ASSETS_DIR = Path(__file__).parent.parent.parent / "assets"
MIC_ON_ICON = ASSETS_DIR / "mic_on.png"
MIC_OFF_ICON = ASSETS_DIR / "mic_off.png"

# Audio settings
SAMPLE_RATE = 16000  # Hz - required by Moonshine
CHUNK_DURATION = 2.0  # seconds - balance between latency and quality
CHANNELS = 1  # mono
AUDIO_DTYPE = "float32"
BLOCK_SIZE = 1024  # samples per callback

# Transcription settings
MODEL_NAME = "UsefulSensors/moonshine-base"
MAX_NEW_TOKENS = 256

# Hotkey settings
HOLD_THRESHOLD = 0.25  # seconds - distinguish tap from hold
DOUBLE_TAP_THRESHOLD = 0.3  # seconds - max time between taps for double-tap

# Minimum audio length to process (in samples)
MIN_AUDIO_SAMPLES = 1600  # 100ms at 16kHz
