"""Unit tests for transcription module."""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.mark.unit
class TestMoonshineTranscriber:
    """Tests for MoonshineTranscriber class."""

    @pytest.fixture
    def mock_transformers(self):
        """Mock transformers components."""
        with patch("vox.transcribe.AutoProcessor") as mock_processor_cls:
            with patch(
                "vox.transcribe.MoonshineForConditionalGeneration"
            ) as mock_model_cls:
                with patch("vox.transcribe.torch") as mock_torch:
                    # Setup processor mock
                    mock_processor = MagicMock()
                    mock_processor.return_value = {
                        "input_features": MagicMock()
                    }
                    mock_processor.batch_decode.return_value = ["  hello world  "]
                    mock_processor_cls.from_pretrained.return_value = mock_processor

                    # Setup model mock
                    mock_model = MagicMock()
                    mock_model.generate.return_value = MagicMock()
                    mock_model.to.return_value = mock_model
                    mock_model_cls.from_pretrained.return_value = mock_model

                    # Setup torch mock
                    mock_torch.backends.mps.is_available.return_value = False
                    mock_torch.cuda.is_available.return_value = False
                    mock_torch.device.return_value = MagicMock()
                    mock_torch.float32 = "float32"
                    mock_torch.float16 = "float16"

                    # Create a proper context manager for no_grad
                    @contextmanager
                    def mock_no_grad():
                        yield

                    mock_torch.no_grad = mock_no_grad

                    yield {
                        "processor_cls": mock_processor_cls,
                        "processor": mock_processor,
                        "model_cls": mock_model_cls,
                        "model": mock_model,
                        "torch": mock_torch,
                    }

    def test_init_loads_model(self, mock_transformers) -> None:
        """Transcriber should load model and processor on init."""
        from vox.transcribe import MoonshineTranscriber

        transcriber = MoonshineTranscriber()

        mock_transformers["processor_cls"].from_pretrained.assert_called_once()
        mock_transformers["model_cls"].from_pretrained.assert_called_once()

    def test_init_custom_model(self, mock_transformers) -> None:
        """Transcriber should accept custom model name."""
        from vox.transcribe import MoonshineTranscriber

        transcriber = MoonshineTranscriber(model_name="UsefulSensors/moonshine-tiny")

        mock_transformers["processor_cls"].from_pretrained.assert_called_with(
            "UsefulSensors/moonshine-tiny"
        )

    def test_transcribe_returns_string(self, mock_transformers) -> None:
        """Transcribe should return cleaned string."""
        from vox.transcribe import MoonshineTranscriber

        transcriber = MoonshineTranscriber()

        audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        result = transcriber.transcribe(audio)

        assert result == "hello world"
        assert isinstance(result, str)

    def test_transcribe_strips_whitespace(self, mock_transformers) -> None:
        """Transcribe should strip leading/trailing whitespace."""
        from vox.transcribe import MoonshineTranscriber

        mock_transformers["processor"].batch_decode.return_value = ["\n  test  \n"]

        transcriber = MoonshineTranscriber()
        audio = np.zeros(16000, dtype=np.float32)
        result = transcriber.transcribe(audio)

        assert result == "test"

    def test_transcribe_handles_empty_result(self, mock_transformers) -> None:
        """Should handle empty transcription gracefully."""
        from vox.transcribe import MoonshineTranscriber

        mock_transformers["processor"].batch_decode.return_value = [""]

        transcriber = MoonshineTranscriber()
        audio = np.zeros(1600, dtype=np.float32)  # 100ms
        result = transcriber.transcribe(audio)

        assert result == ""

    def test_transcribe_calls_generate(self, mock_transformers) -> None:
        """Transcribe should call model.generate."""
        from vox.transcribe import MoonshineTranscriber

        transcriber = MoonshineTranscriber()
        audio = np.zeros(16000, dtype=np.float32)
        transcriber.transcribe(audio)

        mock_transformers["model"].generate.assert_called_once()

    def test_device_selection_cpu(self, mock_transformers) -> None:
        """Should use CPU when no GPU available."""
        from vox.transcribe import MoonshineTranscriber

        mock_transformers["torch"].backends.mps.is_available.return_value = False
        mock_transformers["torch"].cuda.is_available.return_value = False

        transcriber = MoonshineTranscriber()

        mock_transformers["torch"].device.assert_called_with("cpu")

    def test_device_selection_mps(self, mock_transformers) -> None:
        """Should prefer MPS when available."""
        from vox.transcribe import MoonshineTranscriber

        mock_transformers["torch"].backends.mps.is_available.return_value = True

        transcriber = MoonshineTranscriber()

        mock_transformers["torch"].device.assert_called_with("mps")

    def test_device_selection_cuda(self, mock_transformers) -> None:
        """Should use CUDA when MPS unavailable but CUDA available."""
        from vox.transcribe import MoonshineTranscriber

        mock_transformers["torch"].backends.mps.is_available.return_value = False
        mock_transformers["torch"].cuda.is_available.return_value = True

        transcriber = MoonshineTranscriber()

        mock_transformers["torch"].device.assert_called_with("cuda")

    def test_model_eval_mode(self, mock_transformers) -> None:
        """Model should be set to eval mode."""
        from vox.transcribe import MoonshineTranscriber

        transcriber = MoonshineTranscriber()

        mock_transformers["model"].eval.assert_called_once()

    def test_transcribe_uses_custom_sample_rate(self, mock_transformers) -> None:
        """Transcribe should pass sample rate to processor."""
        from vox.transcribe import MoonshineTranscriber

        transcriber = MoonshineTranscriber()
        audio = np.zeros(8000, dtype=np.float32)
        transcriber.transcribe(audio, sample_rate=8000)

        # Check that processor was called with custom sample rate
        call_kwargs = mock_transformers["processor"].call_args
        assert call_kwargs[1]["sampling_rate"] == 8000
