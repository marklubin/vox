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
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    def _callback(
        self,
        indata: NDArray[np.float32],
        frames: int,
        time: sd.CallbackFlags,
        status: sd.CallbackFlags,
    ) -> None:
        """Called for each audio block - accumulates into chunks."""
        if status:
            print(f"Audio status: {status}")

        # Flatten and append copy of audio data to buffer (always store as 1D)
        self.buffer.append(indata.copy().flatten())

        # Calculate total samples in buffer
        total_samples = sum(len(b) for b in self.buffer)

        # Emit as many complete chunks as we have
        while total_samples >= self.chunk_samples:
            # Concatenate all buffer data (all are 1D now)
            chunk = np.concatenate(self.buffer)

            # Put exactly chunk_samples into the queue
            self.audio_queue.put(chunk[: self.chunk_samples])

            # Keep any overflow for next chunk (no data loss)
            overflow_samples = total_samples - self.chunk_samples
            if overflow_samples > 0:
                self.buffer = [chunk[self.chunk_samples :]]
            else:
                self.buffer = []

            # Recalculate for loop condition
            total_samples = sum(len(b) for b in self.buffer)

    def start(self) -> None:
        """Start capturing audio from the microphone."""
        if self.stream is not None:
            return  # Already running

        self.buffer = []
        # Clear any old chunks from queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=AUDIO_DTYPE,
            callback=self._callback,
            blocksize=BLOCK_SIZE,
        )
        self.stream.start()

    def stop(self) -> NDArray[np.float32] | None:
        """Stop capture and return any remaining audio in the buffer.

        Returns:
            Remaining audio samples, or None if buffer is empty.
        """
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Return remaining buffer if any
        if self.buffer:
            remaining = np.concatenate(self.buffer).flatten()
            self.buffer = []
            return remaining
        return None

    def get_chunk(self, timeout: float = 0.1) -> NDArray[np.float32] | None:
        """Get the next audio chunk from the queue.

        Args:
            timeout: Maximum time to wait for a chunk in seconds.

        Returns:
            Audio chunk array, or None if no chunk is available.
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Check if audio capture is currently running."""
        return self.stream is not None and self.stream.active
