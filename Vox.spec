# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec file for Vox.

Build:
    uv run pyinstaller Vox.spec

The app will be created at dist/Vox.app
"""

import os
import sys
from pathlib import Path

block_cipher = None

# Get the project root
project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / "launcher.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[],
    hiddenimports=[
        # Core dependencies
        "vox",
        "vox.app",
        "vox.audio",
        "vox.config",
        "vox.hotkeys",
        "vox.insert",
        "vox.transcribe",
        # Transcriber backends
        "mlx",
        "mlx.core",
        "mlx_whisper",
        "moonshine_onnx",
        "faster_whisper",
        "ctranslate2",
        # Deepgram streaming
        "deepgram",
        "deepgram.clients",
        "deepgram.clients.live",
        "websockets",
        # Audio
        "sounddevice",
        "soundfile",
        "librosa",
        "audioread",
        # macOS frameworks
        "rumps",
        "AppKit",
        "Foundation",
        "Quartz",
        "Quartz.CoreGraphics",
        "objc",
        "PyObjCTools",
        # Other
        "tiktoken",
        "tiktoken_ext",
        "tiktoken_ext.openai_public",
        "numpy",
        "onnxruntime",
        "huggingface_hub",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "pytest",
        "hypothesis",
        "test",
        "tests",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Vox",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Vox",
)

app = BUNDLE(
    coll,
    name="Vox.app",
    icon=None,  # Add path to .icns file here
    bundle_identifier="com.vox.dictation",
    info_plist={
        "CFBundleName": "Vox",
        "CFBundleDisplayName": "Vox",
        "CFBundleVersion": "0.1.0",
        "CFBundleShortVersionString": "0.1.0",
        "LSUIElement": True,  # Hide from Dock (menubar app)
        "NSMicrophoneUsageDescription": "Vox needs microphone access to transcribe your speech.",
        "NSAppleEventsUsageDescription": "Vox needs accessibility access to insert text at cursor.",
        "LSEnvironment": {
            "VOX_TRANSCRIBER": "deepgram-streaming",
            "DEEPGRAM_API_KEY": os.environ.get("DEEPGRAM_API_KEY", ""),
            "VOX_LOG_FILE": "/tmp/vox.log",
        },
    },
)
