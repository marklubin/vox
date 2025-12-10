"""Pytest fixtures for Vox tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def sample_audio() -> np.ndarray:
    """Generate a sample audio array (1 second of silence at 16kHz)."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def sample_audio_short() -> np.ndarray:
    """Generate a short audio array (100ms at 16kHz)."""
    return np.zeros(1600, dtype=np.float32)


@pytest.fixture
def sample_audio_noise() -> np.ndarray:
    """Generate a sample audio array with random noise."""
    rng = np.random.default_rng(42)
    return (rng.random(16000) * 0.01).astype(np.float32)


@pytest.fixture
def mock_sounddevice():
    """Mock sounddevice module."""
    with patch("vox.audio.sd") as mock_sd:
        mock_stream = MagicMock()
        mock_sd.InputStream.return_value = mock_stream
        yield mock_sd


@pytest.fixture
def mock_model():
    """Mock the Moonshine model."""
    with patch("vox.transcribe.MoonshineForConditionalGeneration") as mock:
        mock_instance = MagicMock()
        mock.from_pretrained.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_processor():
    """Mock the AutoProcessor."""
    with patch("vox.transcribe.AutoProcessor") as mock:
        mock_instance = MagicMock()
        mock_instance.return_value = {"input_features": MagicMock()}
        mock_instance.batch_decode.return_value = ["  hello world  "]
        mock.from_pretrained.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_torch():
    """Mock torch module for device detection."""
    with patch("vox.transcribe.torch") as mock:
        mock.backends.mps.is_available.return_value = False
        mock.device.return_value = MagicMock()
        yield mock


@pytest.fixture
def mock_accessibility():
    """Mock macOS Accessibility API."""
    with patch("vox.insert.AXIsProcessTrusted") as mock_trusted:
        mock_trusted.return_value = True
        with patch("vox.insert.AXUIElementCreateSystemWide") as mock_system:
            with patch("vox.insert.AXUIElementCopyAttributeValue") as mock_copy:
                with patch("vox.insert.AXUIElementSetAttributeValue") as mock_set:
                    mock_copy.return_value = (0, MagicMock())  # (err, focused_element)
                    mock_set.return_value = 0  # success
                    yield {
                        "trusted": mock_trusted,
                        "system": mock_system,
                        "copy": mock_copy,
                        "set": mock_set,
                    }
