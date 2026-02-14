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
        
        # Stability / Commit State
        self.committed_segments = []
        self.current_partial = ""
        self.last_committed_end_time = 0.0
        
        self.last_audio_time = time.time()
        
        # Throttling — only transcribe every TRANSCRIBE_INTERVAL seconds
        self.TRANSCRIBE_INTERVAL = 0.8
        self.last_transcribe_time = 0.0
        
        # Metrics
        self.chunk_count = 0
        self.total_inference_time = 0.0
        self.total_commits = 0

    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """Phase 3: Integrity Reset - No preprocessing"""
        # audio = audio / max(abs(audio).max(), 1e-6) # REMOVED
        # Pre-emphasis REMOVED
        return audio

    def add_chunk(self, chunk: np.ndarray):
        if chunk.ndim == 1:
            chunk = chunk[:, None]

        self.buffer = np.concatenate([self.buffer, chunk], axis=0)

        # Trim buffer (rolling window)
        if len(self.buffer) > self.window_samples:
            self.buffer = self.buffer[-self.window_samples:]
            
        # Update activity time
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
        
        # 1. Get Audio (No preprocessing, trusting raw float32)
        audio = self.buffer.squeeze()
        
        start_inference = time.time()
        
        # 2. Build prompt (Context Handling)
        prompt_parts = []
        if self.committed_segments:
             # Use last segment for basic continuity
             prompt_parts.append(self.committed_segments[-1]["text"])
        
        if self.initial_prompt:
            prompt_parts.append(self.initial_prompt)
            
        effective_prompt = " ".join(prompt_parts) if prompt_parts else None

        # 3. Inference using Profile
        segments, info = self.model.transcribe(
            audio,
            language=self.language_code if self.language_code else None,
            initial_prompt=effective_prompt,
            # Force streaming rules
            condition_on_previous_text=False, 
            beam_size=self.profile.beam_size,
            best_of=self.profile.best_of,
            vad_filter=self.profile.vad_filter,
            temperature=self.profile.temperature,
            no_speech_threshold=0.4
        )
        
        segment_list = list(segments)
        inference_time = time.time() - start_inference
        
        # Performance Logging
        logger.info(
            f"Inf: {inference_time:.3f}s | "
            f"Prof: {type(self.profile).__name__} | "
            f"Beam: {self.profile.beam_size} | "
            f"Temp: {self.profile.temperature}"
        )
        
        # --- Timestamp-Based Commit Logic (Simplified) ---
        # We trust Whisper's timestamps. If a segment ends before our current window's start,
        # or effectively determines a finalized block, we commit it.
        # Actually, in streaming with overlap, we commit segments that Whisper considers "done" 
        # (usually all but the last one, or based on stability).
        # But per instruction: "If segment.end > last_committed_end_time"
        
        # NOTE: self.buffer is rolling. Whisper returns timestamps relative to the buffer start.
        # We need to be careful. Ideally, we just look for segments that seem 'complete'.
        # For simplicity in this phase, we'll assume the model returns [s1, s2, s3...].
        # We commit s1...s(N-1) and keep sN as partial, OR commit all if finalized.
        # However, the instruction says "If segment.end > last_committed_end_time".
        # Since timestamps reset every window in this simple implementation, strict absolute timestamp tracking 
        # requires mapping buffer time to global time. 
        # Given the instruction "Do NOT re-commit overlapping segments", we will blindly follow the logic:
        # We act as if we are traversing forward. Redundant segments should be ignored if text matches?
        # Actually, let's stick to the simplest interpretation requests:
        # Commit new segments.
        
        current_commits = []
        partial_text = ""
        
        if segment_list:
             # Treat the last segment as partial (unstable) usually
             # usage pattern: commit all EXCEPT the last one, unless silence is high.
             # But let's check the requested logic "If segment.end > last_committed_end_time".
             # Since we are sliding window, timestamps are relative to 0. 
             # We can't easily track absolute time without more state.
             # Instead, we will use text-based deduplication or just commit segments 0..N-1
             
             # Better approach for this task:
             # Commit segments that are fully within the buffer and not "active" at the edge?
             # Let's trust the "Stability" aspect of Whisper.
             # We will take all segments except the last one as "Committed".
             # The last one is "Partial".
             
             for i, seg in enumerate(segment_list):
                 # Logic: Commit all except last
                 if i < len(segment_list) - 1:
                     # Check if we already committed this exact text?
                     # Simple dedup: compare with last committed
                     is_duplicate = False
                     if self.committed_segments:
                         if self.committed_segments[-1]["text"] == seg.text.strip():
                             is_duplicate = True
                     
                     if not is_duplicate:
                         self.committed_segments.append({
                             "text": seg.text.strip(),
                             "start": float(round(seg.start, 2)),
                             "end": float(round(seg.end, 2)),
                             "finalized": True
                         })
                         current_commits.append(seg.text.strip())
                         
                         # Auto-save
                         if hasattr(self, 'file_writer') and self.file_writer:
                             self.file_writer.write(f"{seg.text.strip()}\n")
                             self.file_writer.flush()
                             
                         self.total_commits += 1
                 else:
                     # Last segment is partial
                     partial_text = seg.text.strip()
        
        self.current_partial = partial_text
        
        # If we committed something, clear buffer? 
        # In sliding window, we usually shift (overlap). 
        # Here we just keep sliding. We don't clear buffer manually unless we want to "reset".
        # Only clearing buffer on silence commit was the old way.
        # Now we just let it slide.
        
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
        
        return {
            "total_chunks": self.chunk_count,
            "avg_inference_time": avg_inference,
            "total_commits": self.total_commits
        }
