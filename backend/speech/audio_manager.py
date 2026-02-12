import sounddevice as sd
import numpy as np
import threading
import queue
from scipy.io.wavfile import write
from datetime import datetime
from pathlib import Path
import os

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
            print(status)
        if self.recording:
            self.frames.append(indata.copy())

    def start_recording(self):
        if self.recording:
            return {"status": "already_recording"}
        
        self.frames = []
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='int16',
            callback=self._callback
        )
        self.stream.start()
        print("Recording started...")
        return {"status": "started"}

    def stop_recording(self):
        if not self.recording:
            return {"status": "not_recording"}
        
        self.recording = False
        self.stream.stop()
        self.stream.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"recording_{timestamp}.wav"
        
        recording_data = np.concatenate(self.frames, axis=0)
        write(filename, self.sample_rate, recording_data)
        
        print(f"Recording saved to {filename}")
        self.current_file = str(filename)
        return {"status": "stopped", "file": str(filename)}
