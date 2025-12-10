"""Main Vox application - macOS menubar dictation app."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import rumps

from .audio import AudioCapture
from .config import MIC_OFF_ICON, MIC_ON_ICON, MIN_AUDIO_SAMPLES
from .hotkeys import HotkeyManager, RecordingMode
from .insert import TextInserter
from .transcribe import MoonshineTranscriber

if TYPE_CHECKING:
    pass


class VoxApp(rumps.App):
    """Vox menubar application for dictation.

    This app runs in the macOS menubar and provides:
    - Hold Option to dictate (latch mode)
    - Double-tap Option to toggle dictation (toggle mode)
    - Escape to cancel dictation
    - Streaming transcription with text insertion at cursor
    """

    def __init__(self) -> None:
        """Initialize the Vox app."""
        # Determine icon path - use text fallback if icons don't exist
        icon = None
        if MIC_OFF_ICON.exists():
            icon = str(MIC_OFF_ICON)

        super().__init__(
            name="Vox",
            icon=icon,
            title="🎤" if icon is None else None,
            quit_button=None,  # We'll handle quit ourselves
        )

        # State
        self.recording = False
        self.transcription_thread: threading.Thread | None = None
        self.should_stop = threading.Event()
        self._components_initialized = False

        # Menu items
        self.status_item = rumps.MenuItem("Ready - Hold Option to dictate")
        self.mode_item = rumps.MenuItem("Mode: -")

        self.menu = [
            self.status_item,
            self.mode_item,
            None,  # separator
            rumps.MenuItem("Quit Vox", callback=self.quit_app),
        ]

    def _init_components(self) -> None:
        """Initialize heavy components (model, etc.) lazily."""
        if self._components_initialized:
            return

        # Show loading state
        self.status_item.title = "Loading model..."

        # Initialize components
        self.transcriber = MoonshineTranscriber()
        self.audio_capture = AudioCapture()
        self.text_inserter = TextInserter()

        self.hotkey_manager = HotkeyManager(
            on_start=self.start_recording,
            on_stop=self.stop_recording,
            on_cancel=self.cancel_recording,
        )

        # Check accessibility
        if not self.text_inserter.use_accessibility:
            # Try to show notification, but don't fail if not bundled as .app
            try:
                rumps.notification(
                    title="Vox",
                    subtitle="Accessibility Required",
                    message="Enable in System Preferences → Privacy → Accessibility",
                )
            except RuntimeError:
                # Running from CLI without Info.plist - just print warning
                print("⚠️  Accessibility not enabled. Enable Vox in System Preferences → Security & Privacy → Privacy → Accessibility")

        self.status_item.title = "Ready - Hold Option to dictate"
        self._components_initialized = True

    def run(self) -> None:
        """Run the Vox app."""
        # Initialize components in background to not block startup
        init_thread = threading.Thread(target=self._init_components, daemon=True)
        init_thread.start()

        # Wait for initialization before starting hotkey listener
        init_thread.join()

        if self._components_initialized:
            self.hotkey_manager.start()

        super().run()

    def start_recording(self) -> None:
        """Start recording audio for transcription."""
        if self.recording:
            return

        self.recording = True

        # Update UI
        if MIC_ON_ICON.exists():
            self.icon = str(MIC_ON_ICON)
        else:
            self.title = "🔴"

        mode = self.hotkey_manager.current_mode
        if mode == RecordingMode.LATCH:
            self.status_item.title = "Recording... (release Option to stop)"
            self.mode_item.title = "Mode: Latch"
        else:
            self.status_item.title = "Recording... (double-tap Option to stop)"
            self.mode_item.title = "Mode: Toggle"

        # Start audio capture
        self.should_stop.clear()
        self.audio_capture.start()

        # Start transcription in background thread
        self.transcription_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
        )
        self.transcription_thread.start()

    def stop_recording(self) -> None:
        """Stop recording and process any remaining audio."""
        if not self.recording:
            return

        # Signal transcription thread to stop
        self.should_stop.set()

        # Get remaining audio and transcribe
        remaining = self.audio_capture.stop()
        if remaining is not None and len(remaining) > MIN_AUDIO_SAMPLES:
            try:
                text = self.transcriber.transcribe(remaining)
                if text:
                    self.text_inserter.insert(text + " ")
            except Exception as e:
                print(f"Error transcribing remaining audio: {e}")

        self.recording = False
        self._update_idle_state()

    def cancel_recording(self) -> None:
        """Cancel recording without inserting text."""
        if not self.recording:
            return

        # Signal thread to stop
        self.should_stop.set()

        # Stop audio capture, discard remaining
        self.audio_capture.stop()

        self.recording = False
        self.status_item.title = "Cancelled"
        self.mode_item.title = "Mode: -"

        # Reset to idle after a moment
        threading.Timer(1.0, self._update_idle_state).start()

    def _update_idle_state(self) -> None:
        """Update UI to idle state."""
        if MIC_OFF_ICON.exists():
            self.icon = str(MIC_OFF_ICON)
        else:
            self.title = "🎤"

        self.status_item.title = "Ready - Hold Option to dictate"
        self.mode_item.title = "Mode: -"

    def _transcription_loop(self) -> None:
        """Core transcription loop. STATELESS - each chunk independent.

        Runs indefinitely until should_stop is set.
        """
        while not self.should_stop.is_set():
            chunk = self.audio_capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            try:
                text = self.transcriber.transcribe(chunk)
                if text:
                    # Insert immediately - crash-resistant
                    self.text_inserter.insert(text + " ")
            except Exception as e:
                print(f"Transcription error: {e}")
                # Continue - don't let errors kill the loop

    def quit_app(self, _: rumps.MenuItem | None = None) -> None:
        """Clean up and quit the app."""
        # Stop recording if active
        if self.recording:
            self.should_stop.set()
            self.audio_capture.stop()

        # Stop hotkey listener
        if self._components_initialized:
            self.hotkey_manager.stop()

        rumps.quit_application()


def main() -> None:
    """Entry point for Vox application."""
    # Ensure we're running on macOS
    import sys

    if sys.platform != "darwin":
        print("Vox only runs on macOS")
        sys.exit(1)

    app = VoxApp()
    app.run()


if __name__ == "__main__":
    main()
