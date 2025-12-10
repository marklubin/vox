"""Hotkey handling using pynput."""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Callable

from pynput import keyboard

from .config import DOUBLE_TAP_THRESHOLD, HOLD_THRESHOLD


class RecordingMode(Enum):
    """Recording mode types."""

    LATCH = "latch"  # Hold to record
    TOGGLE = "toggle"  # Tap to start/stop


class HotkeyManager:
    """Manages hotkeys for controlling dictation.

    Supports two modes:
    - Latch: Hold Option key to record, release to stop
    - Toggle: Double-tap Option key to start/stop recording

    Escape cancels any active recording.
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None],
    ) -> None:
        """Initialize the hotkey manager.

        Args:
            on_start: Called when recording should start.
            on_stop: Called when recording should stop normally.
            on_cancel: Called when recording is cancelled (Escape).
        """
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_cancel = on_cancel

        self.recording = False
        self.current_mode: RecordingMode | None = None

        # Latch mode tracking
        self.option_pressed = False
        self.option_press_time: float = 0
        self._hold_timer: threading.Timer | None = None

        # Double-tap detection
        self.last_option_release: float = 0

        # Listener
        self.listener: keyboard.Listener | None = None
        self._running = False

    def start(self) -> None:
        """Start listening for hotkeys."""
        if self._running:
            return

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self.listener.start()
        self._running = True

    def stop(self) -> None:
        """Stop listening for hotkeys."""
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        self._running = False

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key press events."""
        # Escape always cancels
        if key == keyboard.Key.esc and self.recording:
            self._cancel()
            return

        # Option key handling
        if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            if not self.option_pressed:
                self.option_pressed = True
                self.option_press_time = time.time()

                # Start a timer to detect hold threshold while key is pressed
                self._hold_timer = threading.Timer(
                    HOLD_THRESHOLD,
                    self._on_hold_threshold_reached,
                )
                self._hold_timer.daemon = True
                self._hold_timer.start()

    def _on_hold_threshold_reached(self) -> None:
        """Called when the hold threshold is reached while Option is still pressed."""
        if self.option_pressed and not self.recording:
            self._start_recording(RecordingMode.LATCH)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """Handle key release events."""
        if key == keyboard.Key.alt or key == keyboard.Key.alt_l or key == keyboard.Key.alt_r:
            if not self.option_pressed:
                return

            # Cancel any pending hold timer
            if self._hold_timer is not None:
                self._hold_timer.cancel()
                self._hold_timer = None

            hold_duration = time.time() - self.option_press_time
            self.option_pressed = False

            if self.recording and self.current_mode == RecordingMode.LATCH:
                # End latch recording on release
                self._stop_recording()

            elif hold_duration < HOLD_THRESHOLD:
                # Was a tap (not a hold) - check for double-tap
                time_since_last_tap = time.time() - self.last_option_release

                if time_since_last_tap < DOUBLE_TAP_THRESHOLD:
                    # Double tap detected
                    if self.recording and self.current_mode == RecordingMode.TOGGLE:
                        self._stop_recording()
                    elif not self.recording:
                        self._start_recording(RecordingMode.TOGGLE)
                    self.last_option_release = 0  # Reset
                else:
                    self.last_option_release = time.time()
            # If hold_duration >= HOLD_THRESHOLD and not in latch mode,
            # the timer already started recording

    def _start_recording(self, mode: RecordingMode) -> None:
        """Start recording in the specified mode."""
        if self.recording:
            return

        self.recording = True
        self.current_mode = mode
        try:
            self.on_start()
        except Exception as e:
            print(f"Error in on_start callback: {e}")
            self.recording = False
            self.current_mode = None

    def _stop_recording(self) -> None:
        """Stop recording normally."""
        if not self.recording:
            return

        self.recording = False
        self.current_mode = None
        try:
            self.on_stop()
        except Exception as e:
            print(f"Error in on_stop callback: {e}")

    def _cancel(self) -> None:
        """Cancel recording without saving."""
        if not self.recording:
            return

        self.recording = False
        self.current_mode = None
        try:
            self.on_cancel()
        except Exception as e:
            print(f"Error in on_cancel callback: {e}")

    @property
    def is_running(self) -> bool:
        """Check if the hotkey listener is running."""
        return self._running
