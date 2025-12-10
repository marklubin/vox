"""Speech-to-text transcription using Moonshine via HuggingFace Transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import AutoProcessor, MoonshineForConditionalGeneration

from .config import MAX_NEW_TOKENS, MODEL_NAME, SAMPLE_RATE

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MoonshineTranscriber:
    """Transcribes audio using the Moonshine model.

    This class is designed for stateless operation - each transcription call
    is independent with no accumulated context. This prevents state corruption
    over long sessions.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Initialize the transcriber.

        Args:
            model_name: HuggingFace model identifier for Moonshine.
        """
        self.model_name = model_name
        self.processor = AutoProcessor.from_pretrained(model_name)

        # Determine device - prefer MPS (Metal) on Apple Silicon
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            # MPS works best with float32 for most operations
            self.dtype = torch.float32
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.float16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        self.model = MoonshineForConditionalGeneration.from_pretrained(
            model_name,
            dtype=self.dtype,
        ).to(self.device)

        # Set to eval mode for inference
        self.model.eval()

    def transcribe(
        self,
        audio_array: NDArray[np.float32],
        sample_rate: int = SAMPLE_RATE,
    ) -> str:
        """Transcribe an audio chunk.

        This method is STATELESS - no accumulated context between calls.
        Moonshine's compute scales with audio length (unlike Whisper's 30s chunks),
        making it ideal for streaming short segments.

        Args:
            audio_array: Audio samples as float32 numpy array.
            sample_rate: Sample rate of the audio (default: 16kHz).

        Returns:
            Transcribed text, stripped of whitespace.
        """
        # Process audio into model inputs
        inputs = self.processor(
            audio_array,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        # Move inputs to device
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
            )

        # Decode to text
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return transcription.strip()

    def __del__(self) -> None:
        """Clean up model resources."""
        # Move model to CPU and clear cache if using GPU
        if hasattr(self, "model"):
            self.model.cpu()
            del self.model

        if hasattr(self, "processor"):
            del self.processor

        # Clear CUDA/MPS cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
