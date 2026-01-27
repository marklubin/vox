"""Linux system tray application for Vox dictation."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pystray
from PIL import Image, ImageDraw

from .audio import AudioCapture
from .config import MIC_OFF_ICON, MIC_ON_ICON, MIN_AUDIO_SAMPLES, get_logger
from .hotkeys import HotkeyManager, RecordingMode
from .insert import TextInserter
from .transcribe import get_streaming_transcriber, get_transcriber, is_streaming_backend

if TYPE_CHECKING:
    pass

log = get_logger("app")


def _create_icon_image(recording: bool = False) -> Image.Image:
    """Create a simple icon image.

    Args:
        recording: If True, create a red recording icon. Otherwise, create a gray mic icon.

    Returns:
        PIL Image for the system tray icon.
    """
    # Create a 64x64 image
    size = 64
    image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    if recording:
        # Red circle for recording
        draw.ellipse([8, 8, size - 8, size - 8], fill=(255, 0, 0, 255))
    else:
        # Gray microphone shape
        color = (128, 128, 128, 255)
        # Microphone body (rounded rectangle approximation)
        draw.ellipse([20, 8, 44, 40], fill=color)
        draw.rectangle([20, 24, 44, 36], fill=color)
        # Microphone stand
        draw.rectangle([30, 36, 34, 48], fill=color)
        draw.rectangle([22, 48, 42, 52], fill=color)

    return image


def _load_icon(path: Path, recording: bool = False) -> Image.Image:
    """Load icon from file or create a fallback.

    Args:
        path: Path to the icon file.
        recording: Whether this is for recording state (for fallback).

    Returns:
        PIL Image for the icon.
    """
    if path.exists():
        try:
            return Image.open(path)
        except Exception as e:
            log.warning("Could not load icon %s: %s", path, e)
    return _create_icon_image(recording)


class VoxApp:
    """Vox Linux system tray application for dictation.

    This app runs in the Linux system tray and provides:
    - Hold Alt to dictate (latch mode)
    - Double-tap Alt to toggle dictation (toggle mode)
    - Streaming transcription with text insertion at cursor
    """

    def __init__(self) -> None:
        """Initialize the Vox app."""
        log.info("Initializing VoxApp (Linux)")

        # State
        self.recording = False
        self.transcription_thread: threading.Thread | None = None
        self.should_stop = threading.Event()
        self._components_initialized = False
        self._use_streaming = False

        # Load icons
        self._icon_off = _load_icon(MIC_OFF_ICON, recording=False)
        self._icon_on = _load_icon(MIC_ON_ICON, recording=True)

        # Create system tray icon
        self.icon = pystray.Icon(
            "vox",
            self._icon_off,
            "Vox - Ready",
            menu=self._build_menu(),
        )

        log.info("VoxApp initialized (components not yet loaded)")

    def _build_menu(self) -> pystray.Menu:
        """Build the system tray menu.

        Returns:
            pystray Menu object.
        """
        return pystray.Menu(
            pystray.MenuItem(
                lambda _: self._get_status_text(),
                None,
                enabled=False,
            ),
            pystray.MenuItem(
                lambda _: self._get_mode_text(),
                None,
                enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit Vox", self._quit_app),
        )

    def _get_status_text(self) -> str:
        """Get current status text for menu."""
        if not self._components_initialized:
            return "Loading model..."
        if self.recording:
            mode = getattr(self, "_current_mode", None)
            if mode == RecordingMode.LATCH:
                return "Recording... (release Alt to stop)"
            else:
                return "Recording... (double-tap Alt to stop)"
        return "Ready - Hold Alt to dictate"

    def _get_mode_text(self) -> str:
        """Get current mode text for menu."""
        if self.recording:
            mode = getattr(self, "_current_mode", None)
            if mode == RecordingMode.LATCH:
                return "Mode: Latch"
            elif mode == RecordingMode.TOGGLE:
                return "Mode: Toggle"
        return "Mode: -"

    def _init_components(self) -> None:
        """Initialize heavy components (model, etc.) lazily."""
        if self._components_initialized:
            log.debug("Components already initialized")
            return

        log.info("Initializing components (model loading)...")

        # Check if we should use streaming mode
        self._use_streaming = is_streaming_backend()
        log.info("Streaming mode: %s", self._use_streaming)

        # Initialize components
        if self._use_streaming:
            log.info("Creating streaming transcriber...")
            self.streaming_transcriber = get_streaming_transcriber()
            log.info(
                "Streaming transcriber ready: %s",
                type(self.streaming_transcriber).__name__,
            )
            self.transcriber = None
        else:
            log.info("Creating transcriber...")
            self.transcriber = get_transcriber()
            log.info("Transcriber ready: %s", type(self.transcriber).__name__)
            self.streaming_transcriber = None

        log.info(
            "Creating AudioCapture (streaming_mode=%s)...", self._use_streaming
        )
        self.audio_capture = AudioCapture(streaming_mode=self._use_streaming)
        log.info(
            "AudioCapture ready (sample_rate=%d, chunk_duration=%.1fs, streaming_mode=%s)",
            self.audio_capture.sample_rate,
            self.audio_capture.chunk_duration,
            self._use_streaming,
        )

        log.info("Creating TextInserter...")
        self.text_inserter = TextInserter()
        log.info("TextInserter ready")

        log.info("Creating HotkeyManager...")
        self.hotkey_manager = HotkeyManager(
            on_start=self.start_recording,
            on_stop=self.stop_recording,
            on_cancel=self.cancel_recording,
        )

        self._components_initialized = True
        self._update_icon()
        log.info("All components initialized successfully")

    def run(self) -> None:
        """Run the Vox app."""
        log.info("Starting VoxApp.run()")

        # Initialize components in background
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

        log.info("Entering pystray main loop")
        self.icon.run()

    def _update_icon(self) -> None:
        """Update the system tray icon based on current state."""
        if self.recording:
            self.icon.icon = self._icon_on
            self.icon.title = "Vox - Recording"
        else:
            self.icon.icon = self._icon_off
            self.icon.title = "Vox - Ready"

    def start_recording(self) -> None:
        """Start recording audio for transcription."""
        log.info("start_recording() called (current recording=%s)", self.recording)
        if self.recording:
            log.warning("Already recording, ignoring start_recording()")
            return

        self.recording = True
        self._current_mode = self.hotkey_manager.current_mode

        # Update UI
        self._update_icon()

        log.info("Recording started in %s mode", self._current_mode)

        # Start audio capture
        self.should_stop.clear()
        log.debug("Starting audio capture...")
        self.audio_capture.start()
        log.info("Audio capture started")

        if self._use_streaming:
            # Streaming mode: connect to Deepgram and stream audio directly
            log.debug("Starting streaming transcription...")
            self.streaming_transcriber.start(
                on_transcript=self._on_streaming_transcript
            )
            self.transcription_thread = threading.Thread(
                target=self._streaming_audio_loop,
                daemon=True,
            )
            self.transcription_thread.start()
            log.info("Streaming transcription started")
        else:
            # Batch mode: accumulate chunks and transcribe
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

        # Mark as not recording IMMEDIATELY so hotkey can re-trigger
        self.recording = False
        log.debug("Recording state cleared, ready for new recording")

        # Signal transcription thread to stop
        log.debug("Signaling transcription thread to stop...")
        self.should_stop.set()

        # Get remaining audio
        log.debug("Stopping audio capture and getting remaining audio...")
        remaining = self.audio_capture.stop()

        if self._use_streaming:
            # Streaming mode: send remaining audio then close connection
            def finish_streaming() -> None:
                """Send remaining audio and close streaming connection."""
                if remaining is not None and len(remaining) > MIN_AUDIO_SAMPLES:
                    log.info(
                        "Sending remaining audio: %d samples (%.2fs)",
                        len(remaining),
                        len(remaining) / self.audio_capture.sample_rate,
                    )
                    try:
                        self.streaming_transcriber.send_audio(remaining)
                    except Exception as e:
                        log.error("Error sending remaining audio: %s", e)

                # Give Deepgram a moment to process final audio before closing
                import time

                time.sleep(0.5)

                log.debug("Stopping streaming transcriber...")
                self.streaming_transcriber.stop()

            threading.Thread(target=finish_streaming, daemon=True).start()
        else:
            # Batch mode: transcribe remaining audio
            def process_remaining() -> None:
                """Process remaining audio in background thread."""
                if remaining is not None:
                    log.info(
                        "Remaining audio: %d samples (%.2fs)",
                        len(remaining),
                        len(remaining) / self.audio_capture.sample_rate,
                    )
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
                            log.error(
                                "Error transcribing remaining audio: %s",
                                e,
                                exc_info=True,
                            )
                    else:
                        log.debug(
                            "Remaining audio too short (%d < %d samples), skipping",
                            len(remaining),
                            MIN_AUDIO_SAMPLES,
                        )
                else:
                    log.debug("No remaining audio")

            # Process remaining audio in background so hotkey can re-trigger immediately
            threading.Thread(target=process_remaining, daemon=True).start()

        self._update_icon()
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

        # Stop streaming if in streaming mode
        if self._use_streaming and self.streaming_transcriber:
            log.debug("Stopping streaming transcriber...")
            self.streaming_transcriber.stop()

        self.recording = False
        log.info("Recording cancelled")

        # Reset to idle
        self._update_icon()

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
            log.debug(
                "Got chunk #%d: %d samples (%.2fs)",
                chunk_count,
                len(chunk),
                len(chunk) / self.audio_capture.sample_rate,
            )

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
                log.error(
                    "Transcription error on chunk #%d: %s", chunk_count, e, exc_info=True
                )
                # Continue - don't let errors kill the loop

        log.info("Transcription loop ended (processed %d chunks)", chunk_count)

    def _on_streaming_transcript(self, text: str) -> None:
        """Callback for streaming transcription results.

        Args:
            text: Transcribed text from Deepgram.
        """
        log.info("Streaming transcript: '%s'", text)
        if text:
            log.debug("Inserting streaming text...")
            success = self.text_inserter.insert(text + " ")
            log.debug("Insert success: %s", success)

    def _streaming_audio_loop(self) -> None:
        """Streams audio to Deepgram continuously.

        Runs until should_stop is set, sending small audio blocks directly
        to the streaming transcriber without accumulating chunks.
        """
        log.info("Streaming audio loop started")
        blocks_sent = 0

        while not self.should_stop.is_set():
            # Get audio blocks as they come in (smaller timeout for lower latency)
            try:
                # Access the raw audio queue for smaller blocks
                block = self.audio_capture.audio_queue.get(timeout=0.05)
                if block is not None and len(block) > 0:
                    blocks_sent += 1
                    self.streaming_transcriber.send_audio(block)
                    if blocks_sent % 50 == 0:  # Log every ~1 second
                        log.debug("Sent %d audio blocks", blocks_sent)
            except queue.Empty:
                continue
            except Exception as e:
                log.error("Error in streaming audio loop: %s", e, exc_info=True)

        log.info("Streaming audio loop ended (sent %d blocks)", blocks_sent)

    def _quit_app(self, icon: pystray.Icon | None = None, item: pystray.MenuItem | None = None) -> None:
        """Clean up and quit the app."""
        log.info("quit_app() called")
        # Stop recording if active
        if self.recording:
            log.debug("Stopping active recording before quit...")
            self.should_stop.set()
            self.audio_capture.stop()

            # Stop streaming if in streaming mode
            if self._use_streaming and self.streaming_transcriber:
                log.debug("Stopping streaming transcriber...")
                self.streaming_transcriber.stop()

        # Stop hotkey listener
        if self._components_initialized:
            log.debug("Stopping hotkey manager...")
            self.hotkey_manager.stop()

        log.info("Quitting application")
        self.icon.stop()
