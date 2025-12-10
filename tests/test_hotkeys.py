"""Unit tests for hotkey handling module."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from vox.hotkeys import HotkeyManager, RecordingMode


@pytest.mark.unit
class TestHotkeyManager:
    """Tests for HotkeyManager class."""

    @pytest.fixture
    def callbacks(self):
        """Create mock callbacks."""
        return {
            "start": MagicMock(),
            "stop": MagicMock(),
            "cancel": MagicMock(),
        }

    @pytest.fixture
    def hotkey_manager(self, callbacks):
        """Create a HotkeyManager instance with mock callbacks."""
        return HotkeyManager(
            on_start=callbacks["start"],
            on_stop=callbacks["stop"],
            on_cancel=callbacks["cancel"],
        )

    @pytest.fixture
    def mock_key(self):
        """Create mock keyboard keys."""
        with patch("vox.hotkeys.keyboard") as mock_kb:
            # Create mock Key enum values
            mock_kb.Key.alt = "alt"
            mock_kb.Key.alt_l = "alt_l"
            mock_kb.Key.alt_r = "alt_r"
            mock_kb.Key.esc = "esc"
            mock_kb.Listener = MagicMock()
            yield mock_kb

    def test_init_state(self, hotkey_manager) -> None:
        """HotkeyManager initializes with correct state."""
        assert hotkey_manager.recording is False
        assert hotkey_manager.current_mode is None
        assert hotkey_manager.option_pressed is False
        assert hotkey_manager.listener is None

    def test_start_creates_listener(self, hotkey_manager) -> None:
        """start() should create and start a keyboard listener."""
        with patch("vox.hotkeys.keyboard.Listener") as mock_listener:
            mock_instance = MagicMock()
            mock_listener.return_value = mock_instance

            hotkey_manager.start()

            mock_listener.assert_called_once()
            mock_instance.start.assert_called_once()
            assert hotkey_manager.is_running is True

    def test_start_idempotent(self, hotkey_manager) -> None:
        """Calling start() twice should not create multiple listeners."""
        with patch("vox.hotkeys.keyboard.Listener") as mock_listener:
            mock_instance = MagicMock()
            mock_listener.return_value = mock_instance

            hotkey_manager.start()
            hotkey_manager.start()  # Second call

            assert mock_listener.call_count == 1

    def test_stop_stops_listener(self, hotkey_manager) -> None:
        """stop() should stop the keyboard listener."""
        with patch("vox.hotkeys.keyboard.Listener") as mock_listener:
            mock_instance = MagicMock()
            mock_listener.return_value = mock_instance

            hotkey_manager.start()
            hotkey_manager.stop()

            mock_instance.stop.assert_called_once()
            assert hotkey_manager.is_running is False

    def test_latch_mode_hold_start(self, hotkey_manager, callbacks, mock_key) -> None:
        """Holding option should start latch mode recording."""
        # Simulate key press
        hotkey_manager._on_press(mock_key.Key.alt)
        assert hotkey_manager.option_pressed is True

        # Simulate hold by waiting past threshold
        time.sleep(0.3)

        # Release key
        hotkey_manager._on_release(mock_key.Key.alt)

        # Should have started recording
        callbacks["start"].assert_called_once()
        # And stopped on release
        callbacks["stop"].assert_called_once()

    def test_latch_mode_quick_release(
        self, hotkey_manager, callbacks, mock_key
    ) -> None:
        """Quick release should not start latch mode."""
        # Simulate quick press and release (tap)
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        # Should not have started recording (was a tap, not hold)
        callbacks["start"].assert_not_called()

    def test_double_tap_toggle_mode(
        self, hotkey_manager, callbacks, mock_key
    ) -> None:
        """Double-tap option should toggle recording."""
        # First tap
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        # Second tap quickly
        time.sleep(0.1)
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        # Should have started toggle mode
        callbacks["start"].assert_called_once()
        assert hotkey_manager.current_mode == RecordingMode.TOGGLE

    def test_double_tap_stops_recording(
        self, hotkey_manager, callbacks, mock_key
    ) -> None:
        """Double-tap while recording should stop."""
        # First double-tap to start
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)
        time.sleep(0.1)
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        assert hotkey_manager.recording is True

        # Wait a bit, then double-tap to stop
        time.sleep(0.5)

        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)
        time.sleep(0.1)
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        callbacks["stop"].assert_called_once()
        assert hotkey_manager.recording is False

    def test_escape_cancels(self, hotkey_manager, callbacks, mock_key) -> None:
        """Escape should cancel active recording."""
        # Start recording via double-tap
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)
        time.sleep(0.1)
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        assert hotkey_manager.recording is True

        # Press escape
        hotkey_manager._on_press(mock_key.Key.esc)

        callbacks["cancel"].assert_called_once()
        assert hotkey_manager.recording is False

    def test_escape_no_effect_when_not_recording(
        self, hotkey_manager, callbacks, mock_key
    ) -> None:
        """Escape should have no effect when not recording."""
        hotkey_manager._on_press(mock_key.Key.esc)

        callbacks["cancel"].assert_not_called()

    def test_callback_exception_handled(
        self, hotkey_manager, callbacks, mock_key
    ) -> None:
        """Exceptions in callbacks should be handled."""
        callbacks["start"].side_effect = Exception("Start failed")

        # This should not raise
        hotkey_manager._start_recording(RecordingMode.TOGGLE)

        # State should be reset
        assert hotkey_manager.recording is False

    def test_alt_l_and_alt_r_handled(
        self, hotkey_manager, callbacks, mock_key
    ) -> None:
        """Both left and right alt keys should work."""
        # Test alt_l
        hotkey_manager._on_press(mock_key.Key.alt_l)
        assert hotkey_manager.option_pressed is True
        hotkey_manager._on_release(mock_key.Key.alt_l)
        hotkey_manager.option_pressed = False

        # Test alt_r
        hotkey_manager._on_press(mock_key.Key.alt_r)
        assert hotkey_manager.option_pressed is True

    def test_other_keys_ignored(self, hotkey_manager, callbacks) -> None:
        """Other keys should be ignored."""
        # Press some random key
        hotkey_manager._on_press("a")
        hotkey_manager._on_release("a")

        callbacks["start"].assert_not_called()
        callbacks["stop"].assert_not_called()
        callbacks["cancel"].assert_not_called()

    def test_recording_mode_enum(self) -> None:
        """RecordingMode enum should have correct values."""
        assert RecordingMode.LATCH.value == "latch"
        assert RecordingMode.TOGGLE.value == "toggle"

    def test_no_double_start(self, hotkey_manager, callbacks, mock_key) -> None:
        """Should not start recording if already recording."""
        # Start via double-tap
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)
        time.sleep(0.1)
        hotkey_manager._on_press(mock_key.Key.alt)
        hotkey_manager._on_release(mock_key.Key.alt)

        assert callbacks["start"].call_count == 1

        # Try to start again manually
        hotkey_manager._start_recording(RecordingMode.LATCH)

        # Should still only be called once
        assert callbacks["start"].call_count == 1

    def test_no_double_stop(self, hotkey_manager, callbacks) -> None:
        """Should not stop recording if not recording."""
        hotkey_manager._stop_recording()

        callbacks["stop"].assert_not_called()
