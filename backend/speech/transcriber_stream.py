# backend/speech/transcriber_stream.py

import numpy as np
import time
from faster_whisper import WhisperModel
from backend.core.logger import get_logger
from backend.speech.model_loader import ModelLoader

logger = get_logger(__name__)

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

        start_load = time.time()
        logger.info(f"Requesting Whisper model '{model_size}'...")
        # Use Singleton Loader
        self.model = ModelLoader.get_model(model_size)
        logger.info(f"Model ready in {time.time() - start_load:.2f}s")

        self.language = language
        self.initial_prompt = initial_prompt
        
        # Stability State 
        self.committed_segments = []  # List of dicts {text: ..., start: ..., end: ..., finalized: True}
        self.current_partial = ""
        self.previous_partial = ""
        self.stability_counter = 0
        self.STABILITY_THRESHOLD = 3
        self.SILENCE_TIMEOUT_SEC = 0.8
        self.last_audio_time = time.time()
        
        # Metrics
        self.metrics = []
        self.chunk_count = 0
        self.total_inference_time = 0.0
        self.total_commits = 0
        self.stability_accum_cycles = 0

    def add_chunk(self, chunk: np.ndarray):
        if chunk.ndim == 1:
            chunk = chunk[:, None]

        self.buffer = np.concatenate([self.buffer, chunk], axis=0)

        # Trim buffer (rolling window)
        if len(self.buffer) > self.window_samples:
            self.buffer = self.buffer[-self.window_samples :]
            
        # Check silence (Energy based VAD)
        # Calculate RMS of the chunk
        rms = np.sqrt(np.mean(chunk**2))
        if rms > 0.01: # Threshold for speech activity
             self.last_audio_time = time.time()

    def ready(self) -> bool:
        return len(self.buffer) >= self.window_samples

    def transcribe(self):
        """
        Returns a dictionary with 'committed' list and 'partial' string.
        """
        if not self.ready():
            return None

        audio = self.buffer.squeeze()
        
        start_inference = time.time()
        
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=False,
            beam_size=3,
            vad_filter=True
        )
        
        # Force generator execution
        segment_list = list(segments)
        end_inference = time.time()
        inference_time = end_inference - start_inference
        
        full_text = " ".join(seg.text.strip() for seg in segment_list)
        
        # --- Stability Logic ---
        
        # Check if text is stable
        if full_text == self.previous_partial and full_text != "":
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.previous_partial = full_text
            
        self.stability_accum_cycles += 1
        
        # Check silence duration
        silence_duration = time.time() - self.last_audio_time
        
        # Commit Conditions
        should_commit = False
        commit_reason = ""
        
        if self.stability_counter >= self.STABILITY_THRESHOLD:
            should_commit = True
            commit_reason = "stability"
        elif silence_duration > self.SILENCE_TIMEOUT_SEC and full_text:
             should_commit = True
             commit_reason = "silence"
             
        if should_commit and full_text:
            # Commit the text
            # Use timestamps from last segment if available, else approximate
            start_ts = segment_list[0].start if segment_list else 0.0
            end_ts = segment_list[-1].end if segment_list else 0.0
            
            new_segment = {
                "text": full_text,
                "start": round(start_ts, 2),
                "end": round(end_ts, 2),
                "finalized": True
            }
            
            self.committed_segments.append(new_segment)
            logger.info(f"Committed segment: '{full_text}' (Reason: {commit_reason})")
            
            self.total_commits += 1
            
            # Reset partial state
            self.current_partial = ""
            self.previous_partial = ""
            self.stability_counter = 0
            
            # Clear buffer? No, usually keep context or sliding window?
            # Ideally we should clear buffer but Whisper needs context.
            # But if we treat it as finalized, we might want to clear.
            # Simpler approach: Keep rolling buffer but prompt logic changes? 
            # For now, we clear buffer to avoid re-transcribing same committed text
            # But we need to keep some for context if we modify prompt.
            # Given instructions "Reset self.current_partial...", let's assume we consume audio.
            # But the 'buffer' in this implementation is a rolling window. 
            # If we don't clear it, next inference will output same text.
            # So YES, we MUST clear buffer or advance window.
            self.buffer = np.zeros((0, 1), dtype=np.float32)
            
        else:
            self.current_partial = full_text

        # Record metrics
        self.chunk_count += 1
        self.total_inference_time += inference_time
        
        return {
            "committed": self.committed_segments,
            "partial": self.current_partial
        }

    def get_metrics_summary(self) -> dict:
        avg_inference = (self.total_inference_time / self.chunk_count) if self.chunk_count > 0 else 0
        avg_stability = (self.stability_accum_cycles / self.total_commits) if self.total_commits > 0 else 0
        
        return {
            "total_chunks": self.chunk_count,
            "avg_inference_time": avg_inference,
            "total_commits": self.total_commits,
            "avg_stability_cycles": avg_stability
        }
