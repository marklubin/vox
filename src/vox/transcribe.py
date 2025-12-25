"""Speech-to-text transcription with pluggable backends.

Supports multiple transcription backends via the VOX_TRANSCRIBER env var:
- "moonshine" (default): UsefulSensors Moonshine model (fastest, good accuracy)
- "mlx-whisper": MLX-optimized Whisper for Apple Silicon (fast, best accuracy)
- "faster-whisper": CTranslate2-based Whisper for CPU (slower, good accuracy)
- "parakeet": NVIDIA Parakeet via MLX (very fast on Apple Silicon)
- "deepgram": Deepgram cloud API batch mode (requires DEEPGRAM_API_KEY env var)
- "deepgram-streaming": Deepgram real-time streaming via WebSocket (lowest latency)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from .config import SAMPLE_RATE, get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

log = get_logger("transcribe")

# Environment variable to select transcriber backend
TRANSCRIBER_ENV = "VOX_TRANSCRIBER"
DEFAULT_TRANSCRIBER = "moonshine"


class Transcriber(ABC):
    """Abstract base class for speech-to-text transcribers.

    All transcriber implementations must implement the transcribe() method.
    """

    @abstractmethod
    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        ...


class MoonshineTranscriber(Transcriber):
    """Transcribes audio using UsefulSensors Moonshine model.

    Moonshine is optimized for real-time transcription on edge devices.
    """

    def __init__(
        self,
        model_name: str = "moonshine/base",
    ) -> None:
        """Initialize the Moonshine transcriber.

        Args:
            model_name: Moonshine model name (e.g., "moonshine/base", "moonshine/tiny").
        """
        log.info("Initializing MoonshineTranscriber with model: %s", model_name)
        self.model_name = model_name

        # Import moonshine here to avoid import errors if not installed
        try:
            import moonshine_onnx
            self._moonshine = moonshine_onnx
        except ImportError as e:
            raise ImportError(
                "moonshine-onnx is required for Moonshine transcriber. "
                "Install with: uv add useful-moonshine-onnx"
            ) from e

        # Pre-load the model by running a dummy transcription (min 0.1s = 1600 samples)
        log.debug("Loading Moonshine model...")
        _ = self._moonshine.transcribe(np.zeros(2000, dtype=np.float32), model=model_name)
        log.info("MoonshineTranscriber ready (model=%s)", model_name)

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk using Moonshine.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        audio_duration = len(audio_array) / sample_rate
        log.debug(
            "Transcribing audio: %d samples (%.2fs) at %d Hz",
            len(audio_array),
            audio_duration,
            sample_rate,
        )

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        log.debug("Running Moonshine transcription...")
        # moonshine_onnx.transcribe returns a list of strings (one per batch item)
        result = self._moonshine.transcribe(audio_array, model=self.model_name)

        # Get the first (and only) result
        text = result[0] if result else ""
        text = text.strip()

        log.info("Transcription result: '%s' (%.2fs audio)", text, audio_duration)
        return text


class WhisperTranscriber(Transcriber):
    """Transcribes audio using Faster-Whisper (CTranslate2).

    Faster-Whisper provides high accuracy with efficient CPU/GPU inference.
    """

    def __init__(
        self,
        model_name: str = "distil-large-v3",
        compute_type: str = "int8",  # int8 works on CPU, float16 requires GPU
    ) -> None:
        """Initialize the Faster-Whisper transcriber.

        Args:
            model_name: Faster-Whisper model name (e.g., "distil-large-v3", "base", "large-v3").
            compute_type: Computation type ("float16", "int8", "float32").
        """
        log.info("Initializing WhisperTranscriber with model: %s", model_name)
        self.model_name = model_name
        self.compute_type = compute_type

        # Import faster_whisper here to avoid import errors if not installed
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError(
                "faster-whisper is required for Whisper transcriber. "
                "Install with: uv add faster-whisper"
            ) from e

        # Use CPU for Faster-Whisper on Mac (MPS not supported by ctranslate2)
        self.device = "cpu"
        log.info("Using device=%s, compute_type=%s", self.device, compute_type)

        log.debug("Loading Faster-Whisper model...")
        self.model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=compute_type,
        )
        log.info("WhisperTranscriber ready (model=%s)", model_name)

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk using Faster-Whisper.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        audio_duration = len(audio_array) / sample_rate
        log.debug(
            "Transcribing audio: %d samples (%.2fs) at %d Hz",
            len(audio_array),
            audio_duration,
            sample_rate,
        )

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        log.debug("Running Faster-Whisper transcription...")
        segments, info = self.model.transcribe(
            audio_array,
            language="en",
            beam_size=5,
            vad_filter=True,  # Filter out silence
            vad_parameters={
                "min_silence_duration_ms": 500,
            },
        )

        log.debug("Detected language: %s (prob=%.2f)", info.language, info.language_probability)

        # Collect all segment texts
        text_parts = []
        for segment in segments:
            log.debug("Segment [%.2fs -> %.2fs]: %s", segment.start, segment.end, segment.text)
            text_parts.append(segment.text)

        result = "".join(text_parts).strip()
        log.info("Transcription result: '%s' (%.2fs audio)", result, audio_duration)
        return result

    def __del__(self) -> None:
        """Clean up model resources."""
        log.debug("Cleaning up WhisperTranscriber resources...")
        if hasattr(self, "model"):
            del self.model
        log.debug("WhisperTranscriber cleanup complete")


class MLXWhisperTranscriber(Transcriber):
    """Transcribes audio using MLX-optimized Whisper for Apple Silicon.

    MLX Whisper leverages Apple's Metal GPU for fast inference on M-series chips.
    This is the recommended backend for Apple Silicon Macs.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/distil-whisper-large-v3",
    ) -> None:
        """Initialize the MLX Whisper transcriber.

        Args:
            model_name: HuggingFace model path (e.g., "mlx-community/distil-whisper-large-v3").
        """
        log.info("Initializing MLXWhisperTranscriber with model: %s", model_name)
        self.model_name = model_name

        # Import mlx_whisper here to avoid import errors if not installed
        try:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required for MLX Whisper transcriber. "
                "Install with: uv add mlx-whisper"
            ) from e

        # Pre-load the model by running a dummy transcription
        log.debug("Loading MLX Whisper model (this may download the model)...")
        _ = self._mlx_whisper.transcribe(
            np.zeros(16000, dtype=np.float32),  # 1 second of silence
            path_or_hf_repo=model_name,
        )
        log.info("MLXWhisperTranscriber ready (model=%s)", model_name)

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk using MLX Whisper.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        audio_duration = len(audio_array) / sample_rate
        log.debug(
            "Transcribing audio: %d samples (%.2fs) at %d Hz",
            len(audio_array),
            audio_duration,
            sample_rate,
        )

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        log.debug("Running MLX Whisper transcription...")
        result = self._mlx_whisper.transcribe(
            audio_array,
            path_or_hf_repo=self.model_name,
        )

        # Result is a dict with "text" key
        text = result.get("text", "") if result else ""
        text = text.strip()

        log.info("Transcription result: '%s' (%.2fs audio)", text, audio_duration)
        return text


class SubprocessMLXTranscriber(Transcriber):
    """Transcribes audio using MLX Whisper in a subprocess.

    This isolates Metal GPU operations from the main AppKit event loop,
    avoiding Metal threading crashes in menubar apps.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/distil-whisper-large-v3",
    ) -> None:
        """Initialize the subprocess MLX Whisper transcriber.

        Args:
            model_name: HuggingFace model path for MLX Whisper model.
        """
        import subprocess
        import sys

        log.info("Initializing SubprocessMLXTranscriber with model: %s", model_name)
        self.model_name = model_name
        self._python = sys.executable

        # Warm up: pre-download the model by running a dummy transcription
        log.debug("Warming up MLX Whisper model (may download on first run)...")
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
        try:
            self._run_subprocess(dummy_audio)
            log.info("SubprocessMLXTranscriber ready (model=%s)", model_name)
        except Exception as e:
            log.warning("Warmup failed (model may download on first real transcription): %s", e)

    def _run_subprocess(self, audio_array: np.ndarray) -> str:
        """Run transcription in a subprocess."""
        import subprocess
        import base64
        import json

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_array.tobytes()).decode("ascii")

        # Run worker subprocess
        result = subprocess.run(
            [self._python, "-m", "vox.transcribe_worker", self.model_name],
            input=audio_b64,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )

        if result.returncode != 0:
            log.error("Subprocess failed: %s", result.stderr)
            raise RuntimeError(f"Transcription subprocess failed: {result.stderr}")

        # Parse JSON output
        try:
            output = json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            log.error("Failed to parse subprocess output: %s", result.stdout)
            raise RuntimeError(f"Invalid subprocess output: {result.stdout}") from e

        if "error" in output:
            raise RuntimeError(f"Transcription error: {output['error']}")

        return output.get("text", "")

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe audio using MLX Whisper in a subprocess.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        audio_duration = len(audio_array) / sample_rate
        log.debug(
            "Transcribing audio: %d samples (%.2fs) at %d Hz",
            len(audio_array),
            audio_duration,
            sample_rate,
        )

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        log.debug("Running MLX Whisper transcription in subprocess...")
        text = self._run_subprocess(audio_array)
        text = text.strip()

        log.info("Transcription result: '%s' (%.2fs audio)", text, audio_duration)
        return text


class DeepgramTranscriber(Transcriber):
    """Transcribes audio using Deepgram's cloud API (batch mode).

    Deepgram provides fast, accurate speech-to-text via their Nova-3 model.
    Requires a DEEPGRAM_API_KEY environment variable.

    For real-time streaming, use DeepgramStreamingTranscriber instead.
    """

    def __init__(
        self,
        model_name: str = "nova-3",
        language: str = "en",
    ) -> None:
        """Initialize the Deepgram transcriber.

        Args:
            model_name: Deepgram model name (e.g., "nova-3", "nova-2", "whisper").
            language: Language code (e.g., "en", "es", "fr").
        """
        log.info("Initializing DeepgramTranscriber with model: %s", model_name)
        self.model_name = model_name
        self.language = language

        # Import deepgram here to avoid import errors if not installed
        try:
            from deepgram import DeepgramClient
        except ImportError as e:
            raise ImportError(
                "deepgram-sdk is required for Deepgram transcriber. "
                "Install with: uv add deepgram-sdk"
            ) from e

        # Check for API key
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY environment variable is required for Deepgram transcriber. "
                "Get your API key at https://console.deepgram.com/"
            )

        self._client = DeepgramClient(api_key=api_key)
        log.info("DeepgramTranscriber ready (model=%s, language=%s)", model_name, language)

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk using Deepgram.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        import io
        import wave

        audio_duration = len(audio_array) / sample_rate
        log.debug(
            "Transcribing audio: %d samples (%.2fs) at %d Hz",
            len(audio_array),
            audio_duration,
            sample_rate,
        )

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Convert float32 [-1, 1] to int16 for WAV format
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())

        wav_bytes = wav_buffer.getvalue()

        log.debug("Running Deepgram transcription...")
        response = self._client.listen.v1.media.transcribe_file(
            request=wav_bytes,
            model=self.model_name,
            language=self.language,
            smart_format=True,
        )

        # Extract transcript from response
        text = ""
        if response.results and response.results.channels:
            channel = response.results.channels[0]
            if channel.alternatives:
                text = channel.alternatives[0].transcript or ""

        text = text.strip()
        log.info("Transcription result: '%s' (%.2fs audio)", text, audio_duration)
        return text


class StreamingTranscriber(ABC):
    """Abstract base class for streaming speech-to-text transcribers.

    Streaming transcribers receive audio continuously and emit transcripts
    via a callback as they become available.
    """

    @abstractmethod
    def start(self, on_transcript: callable) -> None:
        """Start the streaming connection.

        Args:
            on_transcript: Callback function that receives transcript strings.
        """
        ...

    @abstractmethod
    def send_audio(self, audio_array: NDArray[np.float32]) -> None:
        """Send audio data to the transcription service.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the streaming connection and clean up."""
        ...

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the streaming connection is active."""
        ...


class DeepgramStreamingTranscriber(StreamingTranscriber):
    """Streams audio to Deepgram via WebSocket for real-time transcription.

    This provides lower latency than batch transcription as audio is processed
    continuously without waiting for chunks to accumulate.

    Requires a DEEPGRAM_API_KEY environment variable.
    """

    def __init__(
        self,
        model_name: str = "nova-3",
        language: str = "en",
        smart_format: bool = True,
        interim_results: bool = False,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Initialize the Deepgram streaming transcriber.

        Args:
            model_name: Deepgram model name (e.g., "nova-3", "nova-2").
            language: Language code (e.g., "en", "es", "fr").
            smart_format: Enable smart formatting (punctuation, capitalization).
            interim_results: Whether to emit interim (partial) results.
            sample_rate: Audio sample rate in Hz.
        """
        log.info("Initializing DeepgramStreamingTranscriber with model: %s", model_name)
        self.model_name = model_name
        self.language = language
        self.smart_format = smart_format
        self.interim_results = interim_results
        self.sample_rate = sample_rate

        # Check for API key
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not api_key:
            raise ValueError(
                "DEEPGRAM_API_KEY environment variable is required for Deepgram transcriber. "
                "Get your API key at https://console.deepgram.com/"
            )
        self._api_key = api_key

        self._client = None
        self._connection = None
        self._context_manager = None
        self._on_transcript: callable | None = None
        self._connected = False
        self._listener_thread = None
        self._stop_listening = False

        log.info(
            "DeepgramStreamingTranscriber ready (model=%s, language=%s, smart_format=%s)",
            model_name, language, smart_format
        )

    def start(self, on_transcript: callable) -> None:
        """Start the WebSocket connection to Deepgram.

        Args:
            on_transcript: Callback function that receives transcript strings.
        """
        import threading

        # Always update callback (allows pre-warming connection with dummy callback,
        # then updating with real callback when recording starts)
        self._on_transcript = on_transcript

        if self._connected:
            log.debug("Already connected, updated transcript callback")
            return

        log.info("Starting Deepgram streaming connection...")
        self._stop_listening = False

        try:
            from deepgram import DeepgramClient
            from deepgram.core.events import EventType
        except ImportError as e:
            raise ImportError(
                "deepgram-sdk is required for Deepgram transcriber. "
                "Install with: uv add deepgram-sdk"
            ) from e

        self._client = DeepgramClient(api_key=self._api_key)

        # Open WebSocket connection using v5 SDK API
        self._context_manager = self._client.listen.v1.connect(
            model=self.model_name,
            language=self.language,
            smart_format=str(self.smart_format).lower(),
            interim_results=str(self.interim_results).lower(),
            encoding="linear16",
            sample_rate=str(self.sample_rate),
            channels="1",
        )

        # Enter context manager
        self._connection = self._context_manager.__enter__()

        # Set up event handlers
        def on_message(message):
            """Handle transcript results from Deepgram."""
            try:
                # v5 SDK message structure
                msg_type = getattr(message, "type", None)
                if msg_type == "Results":
                    channel = message.channel
                    if channel and channel.alternatives:
                        transcript = channel.alternatives[0].transcript
                        if transcript and len(transcript.strip()) > 0:
                            is_final = getattr(message, "is_final", True)
                            log.debug("Received transcript (final=%s): '%s'", is_final, transcript)
                            if is_final or self.interim_results:
                                if self._on_transcript:
                                    self._on_transcript(transcript.strip())
            except (AttributeError, IndexError) as e:
                log.debug("Could not extract transcript: %s", e)

        def on_error(error):
            """Handle errors from Deepgram."""
            log.error("Deepgram WebSocket error: %s", error)

        def on_close(_):
            """Handle connection close."""
            log.info("Deepgram WebSocket closed")
            self._connected = False

        self._connection.on(EventType.MESSAGE, on_message)
        self._connection.on(EventType.ERROR, on_error)
        self._connection.on(EventType.CLOSE, on_close)

        # Start listening in background thread
        def listener_loop():
            try:
                self._connection.start_listening()
            except Exception as e:
                if not self._stop_listening:
                    log.error("Listener error: %s", e)

        self._listener_thread = threading.Thread(target=listener_loop, daemon=True)
        self._listener_thread.start()

        self._connected = True
        log.info("Deepgram streaming connection established")

    def send_audio(self, audio_array: NDArray[np.float32]) -> None:
        """Send audio data to Deepgram.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
        """
        if not self._connected or self._connection is None:
            log.warning("Not connected, cannot send audio")
            return

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Convert float32 [-1, 1] to int16 (linear16 encoding)
        audio_int16 = (audio_array * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        log.debug("Sending %d bytes of audio to Deepgram", len(audio_bytes))
        # v5 SDK: send_media takes bytes directly
        self._connection.send_media(audio_bytes)

    def stop(self) -> None:
        """Stop the WebSocket connection."""
        log.info("Stopping Deepgram streaming connection...")
        self._stop_listening = True
        self._connected = False

        if self._connection is not None:
            try:
                # Send finalize to get any remaining transcription
                from deepgram.extensions.types.sockets import ListenV1ControlMessage
                self._connection.send_control(ListenV1ControlMessage(type="Finalize"))
            except Exception as e:
                log.debug("Error sending finalize: %s", e)

        if self._context_manager is not None:
            try:
                self._context_manager.__exit__(None, None, None)
            except Exception as e:
                log.warning("Error closing Deepgram connection: %s", e)

        self._connection = None
        self._context_manager = None
        self._client = None
        self._on_transcript = None
        log.info("Deepgram streaming connection stopped")

    @property
    def is_connected(self) -> bool:
        """Check if the streaming connection is active."""
        return self._connected


class ParakeetTranscriber(Transcriber):
    """Transcribes audio using NVIDIA Parakeet via parakeet-mlx.

    Parakeet is extremely fast on Apple Silicon, much faster than Whisper.
    Uses MLX for Metal GPU acceleration.
    """

    def __init__(
        self,
        model_name: str = "mlx-community/parakeet-tdt_ctc-110m",
    ) -> None:
        """Initialize the Parakeet transcriber.

        Args:
            model_name: HuggingFace model path for Parakeet model.
                Options: "mlx-community/parakeet-tdt_ctc-110m" (smallest, fastest)
                         "mlx-community/parakeet-tdt-0.6b-v2" (larger, more accurate)
        """
        import tempfile
        log.info("Initializing ParakeetTranscriber with model: %s", model_name)
        self.model_name = model_name
        self._tempdir = tempfile.mkdtemp(prefix="vox_parakeet_")

        # Import parakeet_mlx here to avoid import errors if not installed
        try:
            from parakeet_mlx import from_pretrained
            self._from_pretrained = from_pretrained
        except ImportError as e:
            raise ImportError(
                "parakeet-mlx is required for Parakeet transcriber. "
                "Install with: uv add parakeet-mlx"
            ) from e

        # Load the model
        log.debug("Loading Parakeet model (this may download on first run)...")
        self.model = self._from_pretrained(model_name)
        log.info("ParakeetTranscriber ready (model=%s)", model_name)

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk using Parakeet.

        Args:
            audio_array: Audio samples as float32 numpy array (16kHz mono).
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        import soundfile as sf
        from pathlib import Path

        audio_duration = len(audio_array) / sample_rate
        log.debug(
            "Transcribing audio: %d samples (%.2fs) at %d Hz",
            len(audio_array),
            audio_duration,
            sample_rate,
        )

        # Ensure audio is float32 and 1D
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Parakeet requires a file path, so write to temp file
        temp_path = Path(self._tempdir) / "audio.wav"
        sf.write(temp_path, audio_array, sample_rate)

        log.debug("Running Parakeet transcription...")
        result = self.model.transcribe(temp_path)

        # Result has a .text attribute
        text = result.text if result else ""
        text = text.strip()

        log.info("Transcription result: '%s' (%.2fs audio)", text, audio_duration)
        return text

    def __del__(self) -> None:
        """Clean up temp directory."""
        import shutil
        if hasattr(self, "_tempdir"):
            try:
                shutil.rmtree(self._tempdir)
            except Exception:
                pass


def get_transcriber(backend: str | None = None, **kwargs) -> Transcriber:
    """Factory function to create a transcriber based on backend selection.

    Args:
        backend: Transcriber backend name. If None, reads from VOX_TRANSCRIBER
            env var, defaulting to "moonshine".
        **kwargs: Additional arguments passed to the transcriber constructor.

    Returns:
        A Transcriber instance.

    Raises:
        ValueError: If the backend is unknown.
        ImportError: If the required package for the backend is not installed.
    """
    if backend is None:
        backend = os.environ.get(TRANSCRIBER_ENV, DEFAULT_TRANSCRIBER)

    backend = backend.lower().strip()
    log.info("Creating transcriber with backend: %s", backend)

    if backend == "moonshine":
        # Default Moonshine settings
        model_name = kwargs.pop("model_name", "moonshine/base")
        return MoonshineTranscriber(model_name=model_name, **kwargs)

    elif backend in ("mlx-whisper", "mlx_whisper", "mlx"):
        # MLX Whisper via subprocess (avoids Metal threading issues with AppKit)
        model_name = kwargs.pop("model_name", "mlx-community/distil-whisper-large-v3")
        return SubprocessMLXTranscriber(model_name=model_name, **kwargs)

    elif backend in ("mlx-whisper-direct", "mlx_direct"):
        # Direct MLX Whisper (may crash in GUI apps due to Metal threading)
        model_name = kwargs.pop("model_name", "mlx-community/distil-whisper-large-v3")
        return MLXWhisperTranscriber(model_name=model_name, **kwargs)

    elif backend in ("faster-whisper", "whisper", "faster_whisper"):
        # Default Faster-Whisper settings
        model_name = kwargs.pop("model_name", "distil-large-v3")
        compute_type = kwargs.pop("compute_type", "int8")  # int8 for CPU compatibility
        return WhisperTranscriber(model_name=model_name, compute_type=compute_type, **kwargs)

    elif backend in ("parakeet", "parakeet-mlx"):
        # Parakeet - fastest option on Apple Silicon
        model_name = kwargs.pop("model_name", "mlx-community/parakeet-tdt_ctc-110m")
        return ParakeetTranscriber(model_name=model_name, **kwargs)

    elif backend == "deepgram":
        # Deepgram cloud API (batch mode) - requires DEEPGRAM_API_KEY env var
        model_name = kwargs.pop("model_name", "nova-3")
        language = kwargs.pop("language", "en")
        return DeepgramTranscriber(model_name=model_name, language=language, **kwargs)

    else:
        raise ValueError(
            f"Unknown transcriber backend: {backend!r}. "
            f"Valid options: 'moonshine', 'mlx-whisper', 'faster-whisper', 'parakeet', 'deepgram'"
        )


def get_streaming_transcriber(backend: str | None = None, **kwargs) -> StreamingTranscriber:
    """Factory function to create a streaming transcriber.

    Args:
        backend: Streaming transcriber backend name. If None, reads from
            VOX_TRANSCRIBER env var. Currently only "deepgram-streaming" is supported.
        **kwargs: Additional arguments passed to the transcriber constructor.

    Returns:
        A StreamingTranscriber instance.

    Raises:
        ValueError: If the backend is unknown or doesn't support streaming.
    """
    if backend is None:
        backend = os.environ.get(TRANSCRIBER_ENV, "")

    backend = backend.lower().strip()
    log.info("Creating streaming transcriber with backend: %s", backend)

    if backend in ("deepgram-streaming", "deepgram-stream", "deepgram-live"):
        model_name = kwargs.pop("model_name", "nova-3")
        language = kwargs.pop("language", "en")
        smart_format = kwargs.pop("smart_format", True)
        interim_results = kwargs.pop("interim_results", False)
        return DeepgramStreamingTranscriber(
            model_name=model_name,
            language=language,
            smart_format=smart_format,
            interim_results=interim_results,
            **kwargs,
        )

    else:
        raise ValueError(
            f"Unknown streaming transcriber backend: {backend!r}. "
            f"Valid options: 'deepgram-streaming'"
        )


def is_streaming_backend(backend: str | None = None) -> bool:
    """Check if a backend supports streaming transcription.

    Args:
        backend: Backend name to check. If None, reads from VOX_TRANSCRIBER env var.

    Returns:
        True if the backend supports streaming, False otherwise.
    """
    if backend is None:
        backend = os.environ.get(TRANSCRIBER_ENV, DEFAULT_TRANSCRIBER)
    backend = backend.lower().strip()
    return backend in ("deepgram-streaming", "deepgram-stream", "deepgram-live")
