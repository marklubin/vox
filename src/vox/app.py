"""Main Vox application - macOS menubar dictation app."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import rumps

from .audio import AudioCapture
from .config import MIC_OFF_ICON, MIC_ON_ICON, MIN_AUDIO_SAMPLES, get_logger
from .hotkeys import HotkeyManager, RecordingMode
from .insert import TextInserter
from .transcribe import MoonshineTranscriber

if TYPE_CHECKING:
    pass

log = get_logger("app")


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
        log.info("Initializing VoxApp")

        # Determine icon path - use text fallback if icons don't exist
        icon = None
        if MIC_OFF_ICON.exists():
            icon = str(MIC_OFF_ICON)
            log.debug("Using icon: %s", icon)
        else:
            log.debug("No icon found, using emoji fallback")

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
        log.info("VoxApp initialized (components not yet loaded)")

    def _init_components(self) -> None:
        """Initialize heavy components (model, etc.) lazily."""
        if self._components_initialized:
            log.debug("Components already initialized")
            return

        log.info("Initializing components (model loading)...")
        # Show loading state
        self.status_item.title = "Loading model..."

        # Initialize components
        log.info("Creating MoonshineTranscriber...")
        self.transcriber = MoonshineTranscriber()
        log.info("Transcriber ready (device=%s)", self.transcriber.device)

        log.info("Creating AudioCapture...")
        self.audio_capture = AudioCapture()
        log.info("AudioCapture ready (sample_rate=%d, chunk_duration=%.1fs)",
                 self.audio_capture.sample_rate, self.audio_capture.chunk_duration)

        log.info("Creating TextInserter...")
        self.text_inserter = TextInserter()
        log.info("TextInserter ready (use_accessibility=%s)", self.text_inserter.use_accessibility)

        log.info("Creating HotkeyManager...")
        self.hotkey_manager = HotkeyManager(
            on_start=self.start_recording,
            on_stop=self.stop_recording,
            on_cancel=self.cancel_recording,
        )

        # Note about Input Monitoring permission
        log.warning(
            "NOTE: If Option key doesn't trigger recording, grant Input Monitoring "
            "permission in System Preferences → Privacy & Security → Input Monitoring"
        )

        # Check accessibility
        if not self.text_inserter.use_accessibility:
            log.warning("Accessibility not enabled - using clipboard fallback")
            # Try to show notification, but don't fail if not bundled as .app
            try:
                rumps.notification(
                    title="Vox",
                    subtitle="Accessibility Required",
                    message="Enable in System Preferences → Privacy → Accessibility",
                )
            except RuntimeError:
                # Running from CLI without Info.plist - just print warning
                log.warning("Cannot show notification (not running as .app bundle)")

        self.status_item.title = "Ready - Hold Option to dictate"
        self._components_initialized = True
        log.info("All components initialized successfully")

    def run(self) -> None:
        """Run the Vox app."""
        log.info("Starting VoxApp.run()")

        # Initialize components in background to not block startup
        init_thread = threading.Thread(target=self._init_components, daemon=True)
        init_thread.start()

        # Wait for initialization before starting hotkey listener
        log.debug("Waiting for component initialization...")
        init_thread.join()

        if self._components_initialized:
            log.info("Starting hotkey manager")
            self.hotkey_manager.start()
        else:
            log.error("Components failed to initialize!")

        log.info("Entering rumps main loop")
        super().run()

    def start_recording(self) -> None:
        """Start recording audio for transcription."""
        log.info("start_recording() called (current recording=%s)", self.recording)
        if self.recording:
            log.warning("Already recording, ignoring start_recording()")
            return

        self.recording = True

        # Update UI
        if MIC_ON_ICON.exists():
            self.icon = str(MIC_ON_ICON)
        else:
            self.title = "🔴"

        mode = self.hotkey_manager.current_mode
        log.info("Recording started in %s mode", mode)
        if mode == RecordingMode.LATCH:
            self.status_item.title = "Recording... (release Option to stop)"
            self.mode_item.title = "Mode: Latch"
        else:
            self.status_item.title = "Recording... (double-tap Option to stop)"
            self.mode_item.title = "Mode: Toggle"

        # Start audio capture
        self.should_stop.clear()
        log.debug("Starting audio capture...")
        self.audio_capture.start()
        log.info("Audio capture started")

        # Start transcription in background thread
        log.debug("Starting transcription thread...")
        self.transcription_thread = threading.Thread(
            target=self._transcription_loop,
            daemon=True,
        )
        self.transcription_thread.start()
        log.info("Transcription thread started")

    def stop_recording(self) -> None:
        """Stop recording and process any remaining audio."""
        log.info("stop_recording() called (current recording=%s)", self.recording)
        if not self.recording:
            log.warning("Not recording, ignoring stop_recording()")
            return

        # Signal transcription thread to stop
        log.debug("Signaling transcription thread to stop...")
        self.should_stop.set()

        # Get remaining audio and transcribe
        log.debug("Stopping audio capture and getting remaining audio...")
        remaining = self.audio_capture.stop()
        if remaining is not None:
            log.info("Remaining audio: %d samples (%.2fs)",
                     len(remaining), len(remaining) / self.audio_capture.sample_rate)
            if len(remaining) > MIN_AUDIO_SAMPLES:
                try:
                    log.debug("Transcribing remaining audio...")
                    text = self.transcriber.transcribe(remaining)
                    log.info("Remaining transcription: '%s'", text)
                    if text:
                        log.debug("Inserting remaining text...")
                        success = self.text_inserter.insert(text + " ")
                        log.info("Insert success: %s", success)
                except Exception as e:
                    log.error("Error transcribing remaining audio: %s", e, exc_info=True)
            else:
                log.debug("Remaining audio too short (%d < %d samples), skipping",
                          len(remaining), MIN_AUDIO_SAMPLES)
        else:
            log.debug("No remaining audio")

        self.recording = False
        self._update_idle_state()
        log.info("Recording stopped")

    def cancel_recording(self) -> None:
        """Cancel recording without inserting text."""
        log.info("cancel_recording() called (current recording=%s)", self.recording)
        if not self.recording:
            log.warning("Not recording, ignoring cancel_recording()")
            return

        # Signal thread to stop
        log.debug("Signaling transcription thread to stop...")
        self.should_stop.set()

        # Stop audio capture, discard remaining
        log.debug("Stopping audio capture (discarding remaining)...")
        self.audio_capture.stop()

        self.recording = False
        self.status_item.title = "Cancelled"
        self.mode_item.title = "Mode: -"
        log.info("Recording cancelled")

        # Reset to idle after a moment
        threading.Timer(1.0, self._update_idle_state).start()

    def _update_idle_state(self) -> None:
        """Update UI to idle state."""
        log.debug("Updating to idle state")
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
        log.info("Transcription loop started")
        chunk_count = 0
        while not self.should_stop.is_set():
            chunk = self.audio_capture.get_chunk(timeout=0.1)
            if chunk is None:
                continue

            chunk_count += 1
            log.debug("Got chunk #%d: %d samples (%.2fs)",
                      chunk_count, len(chunk), len(chunk) / self.audio_capture.sample_rate)

            try:
                log.debug("Transcribing chunk #%d...", chunk_count)
                text = self.transcriber.transcribe(chunk)
                log.info("Chunk #%d transcription: '%s'", chunk_count, text)
                if text:
                    # Insert immediately - crash-resistant
                    log.debug("Inserting text: '%s'", text)
                    success = self.text_inserter.insert(text + " ")
                    log.debug("Insert success: %s", success)
                else:
                    log.debug("Empty transcription, skipping insert")
            except Exception as e:
                log.error("Transcription error on chunk #%d: %s", chunk_count, e, exc_info=True)
                # Continue - don't let errors kill the loop

        log.info("Transcription loop ended (processed %d chunks)", chunk_count)

    def quit_app(self, _: rumps.MenuItem | None = None) -> None:
        """Clean up and quit the app."""
        log.info("quit_app() called")
        # Stop recording if active
        if self.recording:
            log.debug("Stopping active recording before quit...")
            self.should_stop.set()
            self.audio_capture.stop()

        # Stop hotkey listener
        if self._components_initialized:
            log.debug("Stopping hotkey manager...")
            self.hotkey_manager.stop()

        log.info("Quitting application")
        rumps.quit_application()


def main() -> None:
    """Entry point for Vox application."""
    # Ensure we're running on macOS
    import sys

    if sys.platform != "darwin":
        print("Vox only runs on macOS")
        sys.exit(1)

    log.info("=" * 60)
    log.info("Vox starting up")
    log.info("=" * 60)

    app = VoxApp()
    app.run()


if __name__ == "__main__":
    main()
