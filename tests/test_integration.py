"""Integration tests for Vox components."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.integration
class TestAudioTranscriptionPipeline:
    """Integration tests for audio capture to transcription pipeline."""

    def test_audio_chunk_format_valid_for_transcription(self) -> None:
        """Audio chunks should be in correct format for transcription."""
        from vox.audio import AudioCapture

        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0)

        # Simulate audio input
        fake_audio = np.random.randn(16000, 1).astype(np.float32)
        capture._callback(fake_audio, 16000, None, None)

        chunk = capture.get_chunk(timeout=0.1)

        assert chunk is not None
        assert chunk.dtype == np.float32
        assert len(chunk) == 16000
        assert chunk.ndim == 1  # Should be flattened

    def test_audio_capture_continuous_chunks(self) -> None:
        """Should produce continuous chunks without gaps."""
        from vox.audio import AudioCapture

        capture = AudioCapture(sample_rate=16000, chunk_duration=0.5)

        # Feed in 2.5 seconds of audio in small blocks
        for _ in range(10):
            block = np.random.randn(4000, 1).astype(np.float32)
            capture._callback(block, 4000, None, None)

        # Should have 5 complete chunks (2.5s / 0.5s = 5)
        chunks = []
        while not capture.audio_queue.empty():
            chunks.append(capture.get_chunk(timeout=0.01))

        assert len(chunks) == 5
        for chunk in chunks:
            assert len(chunk) == 8000  # 0.5s at 16kHz


@pytest.mark.integration
class TestTextInsertionFallback:
    """Integration tests for text insertion with fallback."""

    def test_clipboard_fallback_preserves_state(self) -> None:
        """Clipboard fallback should preserve original clipboard."""
        with patch("vox.insert.AXIsProcessTrusted") as mock_trusted:
            mock_trusted.return_value = False
            with patch("vox.insert.subprocess") as mock_subprocess:
                with patch("vox.insert.pyperclip") as mock_pyperclip:
                    # Setup mock
                    mock_pyperclip.paste.return_value = "original clipboard"
                    mock_subprocess.run.return_value = MagicMock()

                    from vox.insert import TextInserter

                    inserter = TextInserter()
                    inserter.insert("new text")

                    # Verify clipboard was saved, new text set, and original restored
                    mock_pyperclip.paste.assert_called_once()
                    copy_calls = mock_pyperclip.copy.call_args_list
                    assert len(copy_calls) == 2
                    assert copy_calls[0][0][0] == "new text"
                    assert copy_calls[1][0][0] == "original clipboard"


@pytest.mark.integration
class TestHotkeyModes:
    """Integration tests for hotkey modes."""

    def test_latch_mode_lifecycle(self) -> None:
        """Latch mode should start on hold and stop on release."""
        from pynput import keyboard

        from vox.hotkeys import HotkeyManager, RecordingMode

        events = []

        manager = HotkeyManager(
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
            on_cancel=lambda: events.append("cancel"),
        )

        # Simulate hold (press, wait for timer to fire, release)
        manager._on_press(keyboard.Key.alt)
        time.sleep(0.3)  # Hold past threshold - timer will fire
        manager._on_release(keyboard.Key.alt)

        assert "start" in events
        assert "stop" in events
        assert manager.current_mode is None

    def test_toggle_mode_lifecycle(self) -> None:
        """Toggle mode should toggle on double-tap."""
        from pynput import keyboard

        from vox.hotkeys import HotkeyManager, RecordingMode

        events = []

        manager = HotkeyManager(
            on_start=lambda: events.append("start"),
            on_stop=lambda: events.append("stop"),
            on_cancel=lambda: events.append("cancel"),
        )

        # First double-tap to start
        manager._on_press(keyboard.Key.alt)
        manager._on_release(keyboard.Key.alt)
        time.sleep(0.1)
        manager._on_press(keyboard.Key.alt)
        manager._on_release(keyboard.Key.alt)

        assert "start" in events
        assert manager.recording is True
        assert manager.current_mode == RecordingMode.TOGGLE

        # Wait and double-tap to stop
        time.sleep(0.4)
        events.clear()

        manager._on_press(keyboard.Key.alt)
        manager._on_release(keyboard.Key.alt)
        time.sleep(0.1)
        manager._on_press(keyboard.Key.alt)
        manager._on_release(keyboard.Key.alt)

        assert "stop" in events
        assert manager.recording is False


@pytest.mark.integration
class TestAppLifecycle:
    """Integration tests for app lifecycle."""

    def test_app_initialization(self) -> None:
        """App should initialize with correct defaults."""
        with patch("vox.app.MoonshineTranscriber"):
            with patch("vox.app.AudioCapture"):
                with patch("vox.app.TextInserter"):
                    with patch("vox.app.HotkeyManager"):
                        from vox.app import VoxApp

                        app = VoxApp()

                        assert app.recording is False
                        assert app.transcription_thread is None
                        assert "Ready" in app.status_item.title

    def test_start_stop_recording(self) -> None:
        """Start and stop recording should update state correctly."""
        with patch("vox.app.MoonshineTranscriber") as mock_transcriber:
            with patch("vox.app.AudioCapture") as mock_audio:
                with patch("vox.app.TextInserter") as mock_inserter:
                    with patch("vox.app.HotkeyManager") as mock_hotkeys:
                        from vox.app import VoxApp
                        from vox.hotkeys import RecordingMode

                        app = VoxApp()
                        app._init_components()

                        # Set up hotkey manager mock
                        app.hotkey_manager.current_mode = RecordingMode.LATCH

                        # Start recording
                        app.start_recording()

                        assert app.recording is True
                        assert "Recording" in app.status_item.title
                        mock_audio.return_value.start.assert_called()

                        # Stop recording
                        app.stop_recording()

                        assert app.recording is False
                        mock_audio.return_value.stop.assert_called()

    def test_cancel_recording(self) -> None:
        """Cancel should stop without processing remaining audio."""
        with patch("vox.app.MoonshineTranscriber") as mock_transcriber:
            with patch("vox.app.AudioCapture") as mock_audio:
                with patch("vox.app.TextInserter") as mock_inserter:
                    with patch("vox.app.HotkeyManager") as mock_hotkeys:
                        from vox.app import VoxApp
                        from vox.hotkeys import RecordingMode

                        # Setup mocks - make audio capture not return remaining audio
                        mock_audio.return_value.stop.return_value = None
                        mock_audio.return_value.get_chunk.return_value = None

                        app = VoxApp()
                        app._init_components()

                        app.hotkey_manager.current_mode = RecordingMode.TOGGLE
                        app.start_recording()

                        # Immediately cancel (before any chunks are processed)
                        app.cancel_recording()

                        assert app.recording is False
                        assert "Cancelled" in app.status_item.title
                        # Stop was called on audio capture
                        mock_audio.return_value.stop.assert_called()
