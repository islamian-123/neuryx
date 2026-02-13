import sounddevice as sd
import numpy as np
import threading
import queue
from scipy.io.wavfile import write
from datetime import datetime
from pathlib import Path
from backend.core.logger import get_logger

logger = get_logger(__name__)

class AudioManager:
    def __init__(self):
        self.recording = False
        self.frames = []
        self.stream = None
        self.sample_rate = 16000
        self.channels = 1
        self.output_dir = Path("recordings")
        self.output_dir.mkdir(exist_ok=True)
        self.current_file = None

    def _callback(self, indata, frames, time, status):
        if status:
            logger.warning(f"Audio Callback Status: {status}")
        if self.recording:
            self.frames.append(indata.copy())

    def start_recording(self):
        if self.recording:
            logger.warning("Attempted to start recording while already active")
            return {"status": "already_recording"}
        
        self.frames = []
        self.recording = True
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16',
                callback=self._callback
            )
            self.stream.start()
            logger.info("Recording started...")
            return {"status": "started"}
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.recording = False
            raise e

    def stop_recording(self):
        if not self.recording:
            logger.warning("Attempted to stop recording while not active")
            return {"status": "not_recording"}
        
        self.recording = False
        try:
            self.stream.stop()
            self.stream.close()
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"recording_{timestamp}.wav"
            
            if not self.frames:
                logger.warning("No audio frames captured")
                return {"status": "error", "message": "No audio captured"}

            recording_data = np.concatenate(self.frames, axis=0)
            write(filename, self.sample_rate, recording_data)
            
            logger.info(f"Recording saved to {filename}")
            self.current_file = str(filename)
            return {"status": "stopped", "file": str(filename)}
        except Exception as e:
            logger.error(f"Failed to stop/save recording: {e}")
            raise e
