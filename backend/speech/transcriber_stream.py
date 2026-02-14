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
        window_seconds=3.0,
        language="en",
        initial_prompt=None
    ):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_seconds)
        self.buffer = np.zeros((0, 1), dtype=np.float32)

        start_load = time.time()
        logger.info(f"Requesting Whisper model '{model_size}'...")
        self.model = ModelLoader.get_model(model_size)
        logger.info(f"Model ready in {time.time() - start_load:.2f}s")

        self.language = language if language else None
        self.initial_prompt = initial_prompt
        
        # Stability State 
        self.committed_segments = []
        self.current_partial = ""
        self.previous_partial = ""
        self.stability_counter = 0
        self.STABILITY_THRESHOLD = 3
        self.SILENCE_TIMEOUT_SEC = 0.8
        self.last_audio_time = time.time()
        
        # Throttling — only transcribe every TRANSCRIBE_INTERVAL seconds
        self.TRANSCRIBE_INTERVAL = 0.8
        self.last_transcribe_time = 0.0
        
        # Metrics
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
            self.buffer = self.buffer[-self.window_samples:]
            
        # Check silence (Energy based VAD)
        rms = np.sqrt(np.mean(chunk**2))
        if rms > 0.01:
             self.last_audio_time = time.time()

    def ready(self) -> bool:
        # Must have enough audio AND enough time since last transcription
        now = time.time()
        if len(self.buffer) < self.window_samples:
            return False
        if now - self.last_transcribe_time < self.TRANSCRIBE_INTERVAL:
            return False
        return True

    def transcribe(self):
        """
        Returns a dictionary with 'committed' list and 'partial' string.
        Throttled to run at most once per TRANSCRIBE_INTERVAL.
        """
        if not self.ready():
            # Still return current state so frontend stays updated
            return {
                "committed": self.committed_segments,
                "partial": self.current_partial
            }

        self.last_transcribe_time = time.time()
        audio = self.buffer.squeeze()
        
        start_inference = time.time()
        
        # Build prompt: last committed segment + initial prompt
        prompt_parts = []
        if self.committed_segments:
            prompt_parts.append(self.committed_segments[-1]["text"])
        if self.initial_prompt:
            prompt_parts.append(self.initial_prompt)
        effective_prompt = ". ".join(prompt_parts) if prompt_parts else None

        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            initial_prompt=effective_prompt,
            condition_on_previous_text=False,
            beam_size=3,
            vad_filter=True,
            temperature=[0.0, 0.2, 0.4],  # Fallback to creative decoding if high confidence fails
            repetition_penalty=1.1,       # Reduce looping repetitions
            no_speech_threshold=0.6       # Slightly relaxed silence detection
        )
        
        segment_list = list(segments)
        inference_time = time.time() - start_inference
        
        full_text = " ".join(seg.text.strip() for seg in segment_list)
        
        # --- Stability Logic ---
        if full_text == self.previous_partial and full_text != "":
            self.stability_counter += 1
        else:
            self.stability_counter = 0
            self.previous_partial = full_text
            
        self.stability_accum_cycles += 1
        
        # Silence duration
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
            start_ts = segment_list[0].start if segment_list else 0.0
            end_ts = segment_list[-1].end if segment_list else 0.0
            
            self.committed_segments.append({
                "text": full_text,
                "start": float(round(start_ts, 2)),
                "end": float(round(end_ts, 2)),
                "finalized": True
            })
            logger.info(f"Committed segment: '{full_text}' (Reason: {commit_reason})")
            
            self.total_commits += 1
            
            # Reset partial state
            self.current_partial = ""
            self.previous_partial = ""
            self.stability_counter = 0
            
            # Clear buffer to avoid re-transcribing committed text
            self.buffer = np.zeros((0, 1), dtype=np.float32)
            
            # Cap committed segments to last 20 to prevent memory bloat
            if len(self.committed_segments) > 20:
                self.committed_segments = self.committed_segments[-20:]
            
        else:
            self.current_partial = full_text

        # Metrics
        self.chunk_count += 1
        self.total_inference_time += inference_time
        
        if inference_time > 1.0:
            logger.warning(f"Slow inference: {inference_time:.2f}s")
        
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
