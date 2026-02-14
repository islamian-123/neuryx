import numpy as np
import time
from faster_whisper import WhisperModel
from backend.core.logger import get_logger
from backend.speech.model_loader import ModelLoader
from backend.core.inference_config import get_profile_for_language, STREAMING_PROFILE

logger = get_logger(__name__)

class StreamingTranscriber:
    def __init__(
        self,
        model_size="small",
        sample_rate=16000,
        window_seconds=6.0,  # Increased to 6s for better context (Phase 2.5)
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

        self.language_code = language
        
        # Determine Profile
        # Note: Frontend sends "" for Roman Urdu usually, we need to handle that mapping in main.py or here.
        # Assuming main.py passes "ur" or "en". If "roman-ur" is needed, main.py should pass it.
        # Check if language is empty string -> Roman Urdu
        lang_key = "roman-ur" if language == "" else language
            
        self.profile = get_profile_for_language(lang_key)
        
        # Override initial prompt if user provided one, otherwise use profile's
        self.initial_prompt = initial_prompt if initial_prompt else self.profile.initial_prompt
        
        logger.info(f"Active Inference Profile: {lang_key} -> {self.profile}")
        
        # Stability State 
        self.committed_segments = []
        self.current_partial = ""
        self.previous_partial = ""
        self.stability_counter = 0
        self.STABILITY_THRESHOLD = 3
        self.SILENCE_TIMEOUT_SEC = 1.2 # Increased to 1.2s (Phase 2.5)
        self.last_audio_time = time.time()
        
        # Throttling — only transcribe every TRANSCRIBE_INTERVAL seconds
        self.TRANSCRIBE_INTERVAL = 0.8
        self.last_transcribe_time = 0.0
        
        # Preprocessing Config
        self.use_pre_emphasis = True
        
        # Metrics
        self.chunk_count = 0
        self.total_inference_time = 0.0
        self.total_commits = 0
        self.stability_accum_cycles = 0

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Phase 2.5: Normalization and Pre-emphasis"""
        # Ensure float32
        audio = audio.astype(np.float32)
        
        # Normalize amplitude
        max_val = np.max(np.abs(audio))
        if max_val > 1e-6:
            audio = audio / max_val
            
        # Pre-emphasis filter (High-pass)
        if self.use_pre_emphasis and len(audio) > 1:
            # audio[1:] = audio[1:] - 0.97 * audio[:-1]
            # Vectorized implementation
            audio[1:] -= 0.97 * audio[:-1]
            
        return audio

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
        if len(self.buffer) < int(self.sample_rate * 1.0): # Valid audio needs at least 1 sec
            return False
        if now - self.last_transcribe_time < self.TRANSCRIBE_INTERVAL:
            return False
        return True

    def transcribe(self):
        """
        Transcribes audio buffer using active InferenceProfile.
        """
        if not self.ready():
            return {
                "committed": self.committed_segments,
                "partial": self.current_partial
            }

        self.last_transcribe_time = time.time()
        
        # 1. Get Audio
        raw_audio = self.buffer.squeeze()
        
        # 2. Preprocess (Phase 2.5)
        audio = self.preprocess_audio(raw_audio)
        
        start_inference = time.time()
        
        # 3. Build prompt (Context Handling)
        prompt_parts = []
        if self.committed_segments:
             # Use last segment for basic continuity
             prompt_parts.append(self.committed_segments[-1]["text"])
        
        if self.initial_prompt:
            prompt_parts.append(self.initial_prompt)
            
        effective_prompt = " ".join(prompt_parts) if prompt_parts else None

        # 4. Inference using Profile
        # Note: faster_whisper might not support 'best_of' in transcribe(), it uses 'best_of' in decode options.
        # But transcribe() params include best_of.
        
        segments, info = self.model.transcribe(
            audio,
            language=self.language_code if self.language_code else None,
            
            initial_prompt=effective_prompt,
            condition_on_previous_text=self.profile.condition_on_previous_text,
            beam_size=self.profile.beam_size,
            best_of=self.profile.best_of,
            vad_filter=self.profile.vad_filter,
            temperature=self.profile.temperature,
            no_speech_threshold=0.4 # Kept from previous optimization
        )
        
        segment_list = list(segments)
        inference_time = time.time() - start_inference
        
        # Performance Logging (Phase 2.5)
        logger.info(
            f"Inf: {inference_time:.3f}s | "
            f"Prof: {type(self.profile).__name__} | "
            f"Beam: {self.profile.beam_size} | "
            f"Temp: {self.profile.temperature}"
        )
        
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
            
            # Clear buffer
            self.buffer = np.zeros((0, 1), dtype=np.float32)
            
            # Auto-save
            if hasattr(self, 'file_writer') and self.file_writer:
                self.file_writer.write(f"{full_text}\n")
                self.file_writer.flush()
            
        else:
            self.current_partial = full_text

        # Metrics
        self.chunk_count += 1
        self.total_inference_time += inference_time
        
        return {
            "committed": self.committed_segments,
            "partial": self.current_partial
        }

    def set_file_writer(self, filename):
        """Enable auto-saving to a file"""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.file_writer = open(filename, "a", encoding="utf-8")
        logger.info(f"Auto-saving transcript to: {filename}")

    def close(self):
        if hasattr(self, 'file_writer') and self.file_writer:
            self.file_writer.close()

    def get_metrics_summary(self) -> dict:
        avg_inference = (self.total_inference_time / self.chunk_count) if self.chunk_count > 0 else 0
        avg_stability = (self.stability_accum_cycles / self.total_commits) if self.total_commits > 0 else 0
        
        return {
            "total_chunks": self.chunk_count,
            "avg_inference_time": avg_inference,
            "total_commits": self.total_commits,
            "avg_stability_cycles": avg_stability
        }
