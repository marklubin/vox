"""Unit tests for transcription module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vox.transcribe import (
    DEFAULT_TRANSCRIBER,
    TRANSCRIBER_ENV,
    MLXWhisperTranscriber,
    MoonshineTranscriber,
    SubprocessMLXTranscriber,
    Transcriber,
    WhisperTranscriber,
    get_transcriber,
)


@pytest.mark.unit
class TestTranscriberABC:
    """Tests for the Transcriber abstract base class."""

    def test_transcriber_is_abstract(self) -> None:
        """Transcriber should be abstract and not instantiable."""
        with pytest.raises(TypeError, match="abstract"):
            Transcriber()

    def test_transcriber_requires_transcribe(self) -> None:
        """Subclasses must implement transcribe method."""

        class BadTranscriber(Transcriber):
            pass

        with pytest.raises(TypeError, match="abstract"):
            BadTranscriber()


@pytest.mark.unit
class TestMoonshineTranscriber:
    """Tests for MoonshineTranscriber class."""

    @pytest.fixture
    def mock_moonshine(self):
        """Mock moonshine_onnx components."""
        with patch("vox.transcribe.MoonshineTranscriber.__init__", return_value=None):
            transcriber = MoonshineTranscriber.__new__(MoonshineTranscriber)
            transcriber.model_name = "moonshine/base"
            # Mock the _moonshine module with transcribe function
            mock_module = MagicMock()
            mock_module.transcribe.return_value = ["hello world"]
            transcriber._moonshine = mock_module
            yield transcriber

    def test_transcribe_returns_string(self, mock_moonshine) -> None:
        """Transcribe should return cleaned string."""
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_moonshine.transcribe(audio)

        assert result == "hello world"
        assert isinstance(result, str)

    def test_transcribe_strips_whitespace(self, mock_moonshine) -> None:
        """Transcribe should strip leading/trailing whitespace."""
        mock_moonshine._moonshine.transcribe.return_value = ["  test  "]

        audio = np.zeros(16000, dtype=np.float32)
        result = mock_moonshine.transcribe(audio)

        assert result == "test"

    def test_transcribe_handles_empty_result(self, mock_moonshine) -> None:
        """Should handle empty transcription gracefully."""
        mock_moonshine._moonshine.transcribe.return_value = [""]

        audio = np.zeros(1600, dtype=np.float32)
        result = mock_moonshine.transcribe(audio)

        assert result == ""

    def test_transcribe_handles_no_result(self, mock_moonshine) -> None:
        """Should handle no transcription gracefully."""
        mock_moonshine._moonshine.transcribe.return_value = []

        audio = np.zeros(1600, dtype=np.float32)
        result = mock_moonshine.transcribe(audio)

        assert result == ""

    def test_transcribe_converts_dtype(self, mock_moonshine) -> None:
        """Should convert non-float32 audio to float32."""
        mock_moonshine._moonshine.transcribe.return_value = ["converted"]

        audio = np.zeros(16000, dtype=np.float64)
        result = mock_moonshine.transcribe(audio)

        assert result == "converted"

    def test_transcribe_flattens_multidim(self, mock_moonshine) -> None:
        """Should flatten multi-dimensional audio."""
        mock_moonshine._moonshine.transcribe.return_value = ["flattened"]

        audio = np.zeros((16000, 1), dtype=np.float32)
        result = mock_moonshine.transcribe(audio)

        assert result == "flattened"


@pytest.mark.unit
class TestWhisperTranscriber:
    """Tests for WhisperTranscriber class."""

    @pytest.fixture
    def mock_whisper(self):
        """Mock faster_whisper components."""
        with patch("vox.transcribe.WhisperTranscriber.__init__", return_value=None):
            transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
            transcriber.model_name = "distil-large-v3"
            transcriber.compute_type = "float16"
            transcriber.device = "cpu"
            transcriber.model = MagicMock()

            # Mock segment
            mock_segment = MagicMock()
            mock_segment.text = "hello world"
            mock_segment.start = 0.0
            mock_segment.end = 1.0

            # Mock info
            mock_info = MagicMock()
            mock_info.language = "en"
            mock_info.language_probability = 0.99

            transcriber.model.transcribe.return_value = ([mock_segment], mock_info)
            yield transcriber

    def test_transcribe_returns_string(self, mock_whisper) -> None:
        """Transcribe should return cleaned string."""
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_whisper.transcribe(audio)

        assert result == "hello world"
        assert isinstance(result, str)

    def test_transcribe_joins_multiple_segments(self, mock_whisper) -> None:
        """Should join multiple segments."""
        seg1 = MagicMock(text="hello", start=0.0, end=0.5)
        seg2 = MagicMock(text=" world", start=0.5, end=1.0)
        mock_info = MagicMock(language="en", language_probability=0.99)
        mock_whisper.model.transcribe.return_value = ([seg1, seg2], mock_info)

        audio = np.zeros(16000, dtype=np.float32)
        result = mock_whisper.transcribe(audio)

        assert result == "hello world"

    def test_transcribe_handles_empty_segments(self, mock_whisper) -> None:
        """Should handle no segments gracefully."""
        mock_info = MagicMock(language="en", language_probability=0.99)
        mock_whisper.model.transcribe.return_value = ([], mock_info)

        audio = np.zeros(16000, dtype=np.float32)
        result = mock_whisper.transcribe(audio)

        assert result == ""


@pytest.mark.unit
class TestMLXWhisperTranscriber:
    """Tests for MLXWhisperTranscriber class."""

    @pytest.fixture
    def mock_mlx_whisper(self):
        """Mock mlx_whisper components."""
        with patch("vox.transcribe.MLXWhisperTranscriber.__init__", return_value=None):
            transcriber = MLXWhisperTranscriber.__new__(MLXWhisperTranscriber)
            transcriber.model_name = "mlx-community/distil-whisper-large-v3"
            # Mock the _mlx_whisper module
            mock_module = MagicMock()
            mock_module.transcribe.return_value = {"text": " hello world "}
            transcriber._mlx_whisper = mock_module
            yield transcriber

    def test_transcribe_returns_string(self, mock_mlx_whisper) -> None:
        """Transcribe should return cleaned string."""
        audio = np.zeros(16000, dtype=np.float32)
        result = mock_mlx_whisper.transcribe(audio)

        assert result == "hello world"
        assert isinstance(result, str)

    def test_transcribe_strips_whitespace(self, mock_mlx_whisper) -> None:
        """Transcribe should strip leading/trailing whitespace."""
        mock_mlx_whisper._mlx_whisper.transcribe.return_value = {"text": "  test  "}

        audio = np.zeros(16000, dtype=np.float32)
        result = mock_mlx_whisper.transcribe(audio)

        assert result == "test"

    def test_transcribe_handles_empty_result(self, mock_mlx_whisper) -> None:
        """Should handle empty transcription gracefully."""
        mock_mlx_whisper._mlx_whisper.transcribe.return_value = {"text": ""}

        audio = np.zeros(16000, dtype=np.float32)
        result = mock_mlx_whisper.transcribe(audio)

        assert result == ""

    def test_transcribe_handles_none_result(self, mock_mlx_whisper) -> None:
        """Should handle None result gracefully."""
        mock_mlx_whisper._mlx_whisper.transcribe.return_value = None

        audio = np.zeros(16000, dtype=np.float32)
        result = mock_mlx_whisper.transcribe(audio)

        assert result == ""

    def test_transcribe_converts_dtype(self, mock_mlx_whisper) -> None:
        """Should convert non-float32 audio to float32."""
        mock_mlx_whisper._mlx_whisper.transcribe.return_value = {"text": "converted"}

        audio = np.zeros(16000, dtype=np.float64)
        result = mock_mlx_whisper.transcribe(audio)

        assert result == "converted"

    def test_transcribe_flattens_multidim(self, mock_mlx_whisper) -> None:
        """Should flatten multi-dimensional audio."""
        mock_mlx_whisper._mlx_whisper.transcribe.return_value = {"text": "flattened"}

        audio = np.zeros((16000, 1), dtype=np.float32)
        result = mock_mlx_whisper.transcribe(audio)

        assert result == "flattened"


@pytest.mark.unit
class TestGetTranscriber:
    """Tests for the get_transcriber factory function."""

    def test_default_backend_is_moonshine(self) -> None:
        """Default backend should be moonshine."""
        assert DEFAULT_TRANSCRIBER == "moonshine"

    def test_env_var_name(self) -> None:
        """Environment variable should be VOX_TRANSCRIBER."""
        assert TRANSCRIBER_ENV == "VOX_TRANSCRIBER"

    def test_get_moonshine_transcriber(self) -> None:
        """Should create MoonshineTranscriber for 'moonshine' backend."""
        with patch.object(MoonshineTranscriber, "__init__", return_value=None):
            transcriber = get_transcriber("moonshine")
            assert isinstance(transcriber, MoonshineTranscriber)

    def test_get_whisper_transcriber(self) -> None:
        """Should create WhisperTranscriber for 'faster-whisper' backend."""
        with patch.object(WhisperTranscriber, "__init__", return_value=None):
            transcriber = get_transcriber("faster-whisper")
            assert isinstance(transcriber, WhisperTranscriber)

    def test_get_whisper_transcriber_aliases(self) -> None:
        """Should accept 'whisper' and 'faster_whisper' as aliases."""
        with patch.object(WhisperTranscriber, "__init__", return_value=None):
            for alias in ["whisper", "faster_whisper"]:
                transcriber = get_transcriber(alias)
                assert isinstance(transcriber, WhisperTranscriber)

    def test_get_mlx_whisper_transcriber(self) -> None:
        """Should create SubprocessMLXTranscriber for 'mlx-whisper' backend."""
        with patch.object(SubprocessMLXTranscriber, "__init__", return_value=None):
            transcriber = get_transcriber("mlx-whisper")
            assert isinstance(transcriber, SubprocessMLXTranscriber)

    def test_get_mlx_whisper_transcriber_aliases(self) -> None:
        """Should accept 'mlx', 'mlx_whisper' as aliases."""
        with patch.object(SubprocessMLXTranscriber, "__init__", return_value=None):
            for alias in ["mlx", "mlx_whisper"]:
                transcriber = get_transcriber(alias)
                assert isinstance(transcriber, SubprocessMLXTranscriber)

    def test_case_insensitive(self) -> None:
        """Backend selection should be case insensitive."""
        with patch.object(MoonshineTranscriber, "__init__", return_value=None):
            for variant in ["MOONSHINE", "Moonshine", "moonshine"]:
                transcriber = get_transcriber(variant)
                assert isinstance(transcriber, MoonshineTranscriber)

    def test_unknown_backend_raises(self) -> None:
        """Should raise ValueError for unknown backend."""
        with pytest.raises(ValueError, match="Unknown transcriber backend"):
            get_transcriber("unknown")

    def test_reads_env_var(self) -> None:
        """Should read VOX_TRANSCRIBER env var when backend not specified."""
        with patch.dict(os.environ, {TRANSCRIBER_ENV: "faster-whisper"}):
            with patch.object(WhisperTranscriber, "__init__", return_value=None):
                transcriber = get_transcriber()
                assert isinstance(transcriber, WhisperTranscriber)

    def test_default_when_no_env_var(self) -> None:
        """Should use default when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove VOX_TRANSCRIBER if present
            os.environ.pop(TRANSCRIBER_ENV, None)
            with patch.object(MoonshineTranscriber, "__init__", return_value=None):
                transcriber = get_transcriber()
                assert isinstance(transcriber, MoonshineTranscriber)

    def test_passes_kwargs_to_moonshine(self) -> None:
        """Should pass kwargs to MoonshineTranscriber."""
        with patch.object(MoonshineTranscriber, "__init__", return_value=None) as mock_init:
            get_transcriber("moonshine", model_name="moonshine/tiny")
            mock_init.assert_called_once_with(model_name="moonshine/tiny")

    def test_passes_kwargs_to_whisper(self) -> None:
        """Should pass kwargs to WhisperTranscriber."""
        with patch.object(WhisperTranscriber, "__init__", return_value=None) as mock_init:
            get_transcriber("faster-whisper", model_name="base", compute_type="int8")
            mock_init.assert_called_once_with(model_name="base", compute_type="int8")

    def test_passes_kwargs_to_mlx_whisper(self) -> None:
        """Should pass kwargs to SubprocessMLXTranscriber."""
        with patch.object(SubprocessMLXTranscriber, "__init__", return_value=None) as mock_init:
            get_transcriber("mlx-whisper", model_name="mlx-community/whisper-tiny")
            mock_init.assert_called_once_with(model_name="mlx-community/whisper-tiny")


@pytest.mark.integration
class TestTranscriberAccuracy:
    """Integration tests for transcription accuracy using real audio.

    These tests require the actual models to be downloaded and use real audio files.
    Mark with @pytest.mark.slow to skip during quick test runs.
    """

    FIXTURES_DIR = Path(__file__).parent / "fixtures"
    SAMPLE_AUDIO = FIXTURES_DIR / "sample_audio.m4a"

    # Expected transcript (approximate - models may vary slightly)
    EXPECTED_KEYWORDS = [
        "valley",
        "san francisco",
        "python",
        "code",
        "llm",
        "friday",
    ]

    @pytest.fixture
    def sample_audio_array(self) -> np.ndarray:
        """Load the sample audio file as a numpy array."""
        import librosa

        if not self.SAMPLE_AUDIO.exists():
            pytest.skip(f"Sample audio file not found: {self.SAMPLE_AUDIO}")

        # Load audio at 16kHz mono
        audio, _ = librosa.load(str(self.SAMPLE_AUDIO), sr=16000, mono=True)
        return audio.astype(np.float32)

    @pytest.mark.slow
    def test_moonshine_transcription(self, sample_audio_array) -> None:
        """Test Moonshine transcription produces reasonable output."""
        try:
            transcriber = MoonshineTranscriber()
        except ImportError:
            pytest.skip("moonshine-onnx not installed")

        result = transcriber.transcribe(sample_audio_array)
        result_lower = result.lower()

        # Check that at least some expected keywords appear
        found_keywords = [kw for kw in self.EXPECTED_KEYWORDS if kw in result_lower]
        assert len(found_keywords) >= 2, (
            f"Expected at least 2 keywords from {self.EXPECTED_KEYWORDS}, "
            f"found {found_keywords} in: {result}"
        )

    @pytest.mark.slow
    def test_whisper_transcription(self, sample_audio_array) -> None:
        """Test Faster-Whisper transcription produces reasonable output."""
        try:
            transcriber = WhisperTranscriber()
        except ImportError:
            pytest.skip("faster-whisper not installed")

        result = transcriber.transcribe(sample_audio_array)
        result_lower = result.lower()

        # Check that at least some expected keywords appear
        found_keywords = [kw for kw in self.EXPECTED_KEYWORDS if kw in result_lower]
        assert len(found_keywords) >= 3, (
            f"Expected at least 3 keywords from {self.EXPECTED_KEYWORDS}, "
            f"found {found_keywords} in: {result}"
        )

    @pytest.mark.slow
    def test_whisper_more_accurate_than_moonshine(self, sample_audio_array) -> None:
        """Compare Whisper vs Moonshine accuracy (Whisper should be better)."""
        try:
            moonshine = MoonshineTranscriber()
            whisper = WhisperTranscriber()
        except ImportError as e:
            pytest.skip(f"Required package not installed: {e}")

        moonshine_result = moonshine.transcribe(sample_audio_array).lower()
        whisper_result = whisper.transcribe(sample_audio_array).lower()

        moonshine_keywords = [kw for kw in self.EXPECTED_KEYWORDS if kw in moonshine_result]
        whisper_keywords = [kw for kw in self.EXPECTED_KEYWORDS if kw in whisper_result]

        print(f"Moonshine: {moonshine_result}")
        print(f"Moonshine keywords: {moonshine_keywords}")
        print(f"Whisper: {whisper_result}")
        print(f"Whisper keywords: {whisper_keywords}")

        # Whisper should generally find more keywords (be more accurate)
        assert len(whisper_keywords) >= len(moonshine_keywords), (
            f"Expected Whisper to be at least as accurate as Moonshine. "
            f"Moonshine found {len(moonshine_keywords)} keywords, "
            f"Whisper found {len(whisper_keywords)} keywords."
        )
