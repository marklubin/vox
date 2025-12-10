"""Audio capture using sounddevice."""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd

from .config import (
    AUDIO_DTYPE,
    BLOCK_SIZE,
    CHANNELS,
    CHUNK_DURATION,
    SAMPLE_RATE,
    get_logger,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger("audio")


class AudioCapture:
    """Captures audio from the microphone in chunks for streaming transcription.

    Audio is accumulated in a buffer and emitted as chunks when enough samples
    have been collected. This ensures no data loss between chunks.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_duration: float = CHUNK_DURATION,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_queue: queue.Queue[NDArray[np.float32]] = queue.Queue()
        self.buffer: list[NDArray[np.float32]] = []
        self.stream: sd.InputStream | None = None
        self._callback_count = 0

        log.debug(
            "AudioCapture initialized: sample_rate=%d, chunk_duration=%.1fs, chunk_samples=%d",
            sample_rate,
            chunk_duration,
            self.chunk_samples,
        )

    def _callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        """Called for each audio block - accumulates into chunks."""
        self._callback_count += 1
        if status:
            log.warning("Audio callback status: %s (callback #%d)", status, self._callback_count)

        # Flatten and append copy of audio data to buffer (always store as 1D)
        self.buffer.append(indata.copy().flatten())

        # Calculate total samples in buffer
        total_samples = sum(len(b) for b in self.buffer)

        # Emit as many complete chunks as we have
        chunks_emitted = 0
        while total_samples >= self.chunk_samples:
            # Concatenate all buffer data (all are 1D now)
            chunk = np.concatenate(self.buffer)

            # Put exactly chunk_samples into the queue
            self.audio_queue.put(chunk[: self.chunk_samples])
            chunks_emitted += 1

            # Keep any overflow for next chunk (no data loss)
            overflow_samples = total_samples - self.chunk_samples
            if overflow_samples > 0:
                self.buffer = [chunk[self.chunk_samples :]]
            else:
                self.buffer = []

            # Recalculate for loop condition
            total_samples = sum(len(b) for b in self.buffer)

        if chunks_emitted > 0:
            log.debug(
                "Callback #%d: emitted %d chunk(s), buffer_remaining=%d samples",
                self._callback_count,
                chunks_emitted,
                total_samples,
            )

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        if self.stream is not None:
            log.warning("start() called but already running")
            return  # Already running

        log.info("Starting audio capture...")
        self.buffer = []
        self._callback_count = 0
        # Clear any old chunks from queue
        cleared = 0
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break
        if cleared > 0:
            log.debug("Cleared %d old chunks from queue", cleared)

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            callback=self._callback,
            blocksize=BLOCK_SIZE,
        )
        self.stream.start()
        log.info(
            "Audio capture started (sample_rate=%d, channels=%d, blocksize=%d)",
            self.sample_rate,
            CHANNELS,
            BLOCK_SIZE,
        )

    def stop(self) -> NDArray[np.float32] | None:
        """Stop capture and return any remaining audio in the buffer.

        Returns:
            Remaining audio samples, or None if buffer is empty.
        """
        log.info("Stopping audio capture...")
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            log.debug("Audio stream closed (received %d callbacks)", self._callback_count)

        # Return remaining buffer if any
        if self.buffer:
            remaining = np.concatenate(self.buffer).flatten()
            self.buffer = []
            log.info("Returning %d remaining samples (%.2fs)", len(remaining), len(remaining) / self.sample_rate)
            return remaining
        log.debug("No remaining audio in buffer")
        return None

    def get_chunk(self, timeout: float = 0.1) -> NDArray[np.float32] | None:
        """Get the next audio chunk from the queue.

        Args:
            timeout: Maximum time to wait for a chunk in seconds.

        Returns:
            Audio chunk array, or None if no chunk is available.
        """
        try:
            chunk = self.audio_queue.get(timeout=timeout)
            log.debug("Got chunk from queue: %d samples", len(chunk))
            return chunk
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Check if audio capture is currently running."""
        return self.stream is not None and self.stream.active
