# backend/speech/transcriber_stream.py

import numpy as np
import time
from faster_whisper import WhisperModel

class StreamingTranscriber:
    def __init__(
        self,
        model_size="small",
        sample_rate=16000,
        window_seconds=4.0,
        language="en",
        initial_prompt=None
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.buffer = np.zeros((0, 1), dtype=np.float32)

        self.model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8"
        )

        self.language = language
        self.initial_prompt = initial_prompt
        self.last_text = ""

    def add_chunk(self, chunk: np.ndarray):
        if chunk.ndim == 1:
            chunk = chunk[:, None]

        self.buffer = np.concatenate([self.buffer, chunk], axis=0)

        # Trim buffer (rolling window)
        if len(self.buffer) > self.window_samples:
            self.buffer = self.buffer[-self.window_samples :]

    def ready(self) -> bool:
        return len(self.buffer) >= self.window_samples

    def transcribe(self):
        if not self.ready():
            return None

        audio = self.buffer.squeeze()

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            beam_size=3,
            vad_filter=True
        )

        text = " ".join(seg.text.strip() for seg in segments)

        # prevent repeats
        if text and text != self.last_text:
            self.last_text = text
            return text

        return None
