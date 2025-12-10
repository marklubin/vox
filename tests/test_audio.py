"""Unit tests for audio capture module."""

from __future__ import annotations

import numpy as np
import pytest

from vox.audio import AudioCapture


@pytest.mark.unit
class TestAudioCapture:
    """Tests for AudioCapture class."""

    def test_init_defaults(self) -> None:
        """AudioCapture initializes with correct defaults."""
        capture = AudioCapture()
        assert capture.sample_rate == 16000
        assert capture.chunk_duration == 2.0
        assert capture.chunk_samples == 32000
        assert capture.buffer == []
        assert capture.stream is None

    def test_init_custom_params(self) -> None:
        """AudioCapture accepts custom parameters."""
        capture = AudioCapture(sample_rate=8000, chunk_duration=1.0)
        assert capture.sample_rate == 8000
        assert capture.chunk_duration == 1.0
        assert capture.chunk_samples == 8000

    def test_chunk_accumulation(self) -> None:
        """Should emit chunks when buffer reaches threshold."""
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0)

        # Simulate callback with 8000 samples (0.5 seconds)
        fake_audio = np.random.randn(8000, 1).astype(np.float32)
        capture._callback(fake_audio, 8000, None, None)

        # Not enough samples yet
        assert capture.audio_queue.empty()

        # Add more samples to reach threshold
        capture._callback(fake_audio, 8000, None, None)

        # Now we should have a chunk
        assert not capture.audio_queue.empty()
        chunk = capture.audio_queue.get()
        assert len(chunk) == 16000

    def test_no_data_loss_on_overflow(self) -> None:
        """Overflow samples should be preserved for next chunk."""
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0)

        # Add more than one chunk (20000 samples = 1.25 seconds)
        fake_audio = np.random.randn(20000, 1).astype(np.float32)
        capture._callback(fake_audio, 20000, None, None)

        # Get the first chunk
        chunk = capture.audio_queue.get()
        assert len(chunk) == 16000

        # 4000 samples should be in buffer
        assert len(capture.buffer) == 1
        assert len(capture.buffer[0]) == 4000

    def test_multiple_chunks(self) -> None:
        """Should emit multiple chunks for large audio input."""
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0)

        # Add 3.5 seconds of audio (56000 samples)
        fake_audio = np.random.randn(56000, 1).astype(np.float32)
        capture._callback(fake_audio, 56000, None, None)

        # Should have 3 complete chunks
        chunks = []
        while not capture.audio_queue.empty():
            chunks.append(capture.audio_queue.get())

        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) == 16000

        # 8000 samples should remain in buffer
        assert len(capture.buffer) == 1
        assert len(capture.buffer[0]) == 8000

    def test_get_chunk_timeout(self) -> None:
        """get_chunk should return None on timeout."""
        capture = AudioCapture()
        result = capture.get_chunk(timeout=0.01)
        assert result is None

    def test_get_chunk_returns_data(self) -> None:
        """get_chunk should return queued audio data."""
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0)

        # Manually add chunk to queue
        test_chunk = np.ones(16000, dtype=np.float32)
        capture.audio_queue.put(test_chunk)

        result = capture.get_chunk(timeout=0.1)
        assert result is not None
        assert len(result) == 16000
        np.testing.assert_array_equal(result, test_chunk)

    def test_stop_returns_remaining(self) -> None:
        """stop() should return remaining audio in buffer."""
        capture = AudioCapture(sample_rate=16000, chunk_duration=1.0)

        # Add less than a full chunk
        fake_audio = np.random.randn(5000, 1).astype(np.float32)
        capture._callback(fake_audio, 5000, None, None)

        # Stop and get remaining
        remaining = capture.stop()
        assert remaining is not None
        assert len(remaining) == 5000
        assert capture.buffer == []

    def test_stop_returns_none_when_empty(self) -> None:
        """stop() should return None if buffer is empty."""
        capture = AudioCapture()
        remaining = capture.stop()
        assert remaining is None

    def test_is_running_property(self, mock_sounddevice) -> None:
        """is_running should reflect stream state."""
        capture = AudioCapture()
        assert not capture.is_running

        # Start capture
        mock_sounddevice.InputStream.return_value.active = True
        capture.start()
        assert capture.is_running

        # Stop capture
        capture.stop()
        assert not capture.is_running

    def test_start_clears_queue(self) -> None:
        """start() should clear any old chunks from queue."""
        capture = AudioCapture()

        # Add old chunk
        capture.audio_queue.put(np.zeros(1000, dtype=np.float32))
        assert not capture.audio_queue.empty()

        # Mock the stream to avoid actual audio device access
        from unittest.mock import MagicMock, patch

        with patch("vox.audio.sd") as mock_sd:
            mock_stream = MagicMock()
            mock_stream.active = True
            mock_sd.InputStream.return_value = mock_stream
            capture.start()

        # Queue should be cleared
        assert capture.audio_queue.empty()

    def test_start_idempotent(self, mock_sounddevice) -> None:
        """Calling start() twice should not create multiple streams."""
        capture = AudioCapture()
        capture.start()
        capture.start()  # Second call should be no-op

        # InputStream should only be created once
        assert mock_sounddevice.InputStream.call_count == 1

    def test_callback_handles_status(self, capsys) -> None:
        """Callback should print status messages."""
        capture = AudioCapture()

        # Simulate callback with status
        fake_audio = np.zeros((1000, 1), dtype=np.float32)
        capture._callback(fake_audio, 1000, None, "input overflow")

        captured = capsys.readouterr()
        assert "Audio status: input overflow" in captured.out
