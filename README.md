# Vox - Open Source macOS Dictation App

A lightweight, always-on dictation tool that captures your speech and inserts clean text wherever your cursor is.

## Features

- **Hold-to-Dictate (Latch Mode)**: Hold the Option key to record, release to stop
- **Toggle Mode**: Double-tap Option to start/stop recording hands-free
- **Streaming Transcription**: Text appears as you speak, not after you stop
- **System-wide Text Insertion**: Works in any application
- **Escape to Cancel**: Press Escape at any time to cancel without inserting text
- **Local Processing**: Uses Moonshine model locally - no cloud API required

## Requirements

- macOS (Apple Silicon recommended for best performance)
- Python 3.11+
- Microphone access
- Accessibility permissions (for text insertion)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vox.git
cd vox

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

```bash
# Run the app
uv run vox
```

On first launch, you'll need to:
1. Grant **Microphone Access** when prompted
2. Enable **Accessibility** in System Preferences → Security & Privacy → Privacy → Accessibility

### Controls

| Action | Effect |
|--------|--------|
| Hold Option | Start recording (latch mode) |
| Release Option | Stop recording, insert text |
| Double-tap Option | Toggle recording on/off |
| Escape | Cancel recording (no text inserted) |

## How It Works

1. Audio is captured in 2-second chunks using `sounddevice`
2. Each chunk is transcribed using [Moonshine](https://huggingface.co/UsefulSensors/moonshine-base) via HuggingFace Transformers
3. Transcribed text is inserted at the cursor using macOS Accessibility API
4. Falls back to clipboard paste if accessibility is unavailable

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Vox App (rumps)                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Hotkey Listener (pynput)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Audio Capture (sounddevice)                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              STT Engine (Moonshine via transformers)            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Text Inserter                               │
│   Primary: Accessibility API | Fallback: Clipboard              │
└─────────────────────────────────────────────────────────────────┘
```

## Development

### Running Tests

```bash
# Run all unit tests (fast)
uv run pytest -m unit -v

# Run integration tests (loads model)
uv run pytest -m integration -v

# Run all tests
uv run pytest -v

# Run with coverage
uv run pytest --cov=vox --cov-report=html
```

### Project Structure

```
vox/
├── pyproject.toml
├── README.md
├── src/vox/
│   ├── __init__.py
│   ├── app.py           # Main rumps app
│   ├── audio.py         # Audio capture
│   ├── transcribe.py    # Moonshine transcription
│   ├── insert.py        # Text insertion
│   ├── hotkeys.py       # Hotkey handling
│   └── config.py        # Settings
├── assets/
│   ├── mic_on.png       # Optional: menubar icon (recording)
│   └── mic_off.png      # Optional: menubar icon (idle)
└── tests/
    └── ...
```

### Custom Icons

To use custom menubar icons instead of emoji:
1. Create 16x16 or 22x22 PNG images
2. Place them in `assets/mic_on.png` and `assets/mic_off.png`
3. For Retina displays, also create `@2x` versions

## Configuration

Settings can be modified in `src/vox/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `SAMPLE_RATE` | 16000 | Audio sample rate in Hz |
| `CHUNK_DURATION` | 2.0 | Audio chunk size in seconds |
| `MODEL_NAME` | moonshine-base | HuggingFace model ID |
| `HOLD_THRESHOLD` | 0.25 | Seconds to distinguish tap from hold |
| `DOUBLE_TAP_THRESHOLD` | 0.3 | Max seconds between taps for double-tap |

## Troubleshooting

### "Accessibility not enabled" warning
Go to System Preferences → Security & Privacy → Privacy → Accessibility and add Vox (or your terminal app if running from command line).

### No text appears
1. Check that the cursor is in a text field
2. Try the clipboard fallback by temporarily removing accessibility permissions
3. Check the console for error messages

### Model loading is slow
The first run downloads the Moonshine model (~400MB). Subsequent runs are faster.

## License

MIT

## Acknowledgments

- [Moonshine](https://github.com/usefulsensors/moonshine) by Useful Sensors for the speech recognition model
- [rumps](https://github.com/jaredks/rumps) for the menubar app framework
- [pynput](https://github.com/moses-palmer/pynput) for keyboard handling
