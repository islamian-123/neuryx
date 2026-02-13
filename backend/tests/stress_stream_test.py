import sys
import os
import time
import numpy as np
import psutil

# Add project root to path
sys.path.append(os.getcwd())

from backend.speech.transcriber_stream import StreamingTranscriber
from backend.core.system_monitor import get_memory_usage_mb

def run_stress_test(duration_minutes=5):
    print(f"Starting Stress Test for {duration_minutes} minutes...")
    
    # Initialize Transcriber
    transcriber = StreamingTranscriber(model_size="tiny", language="en")
    
    sample_rate = 16000
    chunk_duration = 0.5 # send 0.5s chunks
    chunk_samples = int(sample_rate * chunk_duration)
    
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    peak_memory = 0
    total_chunks = 0
    
    try:
        while time.time() < end_time:
            # Generate random noise
            noise = np.random.uniform(-0.1, 0.1, chunk_samples).astype(np.float32)
            
            transcriber.add_chunk(noise)
            transcriber.transcribe()
            
            total_chunks += 1
            
            # Check memory
            current_mem = get_memory_usage_mb()
            if current_mem > peak_memory:
                peak_memory = current_mem
            
            # Print progress every 30s
            elapsed = time.time() - start_time
            if total_chunks % 60 == 0:
                print(f"Elapsed: {elapsed:.0f}s | Chunks: {total_chunks} | Mem: {current_mem:.0f}MB")
                
            # Simulate real-time by sleeping slightly less than chunk len to account for processing
            # But fast-forward slightly to stress test processing speed too.
            # actually we want to see if it holds up.
            time.sleep(0.01) 
            
    except KeyboardInterrupt:
        print("Test interrupted")
    
    # Summary
    metrics = transcriber.get_metrics_summary()
    print("\n=== Stress Test Summary ===")
    print(f"Total Chunks: {metrics.get('total_chunks', 0)}")
    print(f"Avg Inference Time: {metrics.get('avg_inference_time', 0):.4f}s")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    print("===========================")

if __name__ == "__main__":
    # Ensure backend module is found
    # Run from root: python -m backend.tests.stress_stream_test
    run_stress_test(duration_minutes=5)
