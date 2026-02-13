from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.speech.audio_manager import AudioManager
from backend.speech.transcriber import transcribe_audio
from backend.speech.transcriber_stream import StreamingTranscriber
from backend.routers import models
from backend.core.logger import get_logger
from backend.core.system_monitor import get_memory_usage_mb
import numpy as np
import asyncio
import time

logger = get_logger("backend.main")

app = FastAPI(title="NEURYX")
app.include_router(models.router)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

audio_manager = AudioManager()

@app.on_event("startup")
async def startup_event():
    logger.info("Neuryx Backend Starting...")
    logger.info(f"Initial Memory Usage: {get_memory_usage_mb():.2f} MB")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Neuryx Backend Shutting Down...")

@app.get("/")
def health_check():
    logger.debug("Health check requested")
    return {"status": "ok"}

@app.post("/record/start")
def start_recording():
    """Start recording on the server's default microphone."""
    try:
        logger.info("Starting batch recording...")
        return audio_manager.start_recording()
    except Exception as e:
        logger.error(f"Failed to start recording: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/record/stop")
def stop_recording():
    """Stop recording, save file, and transcribe."""
    try:
        logger.info("Stopping batch recording...")
        result = audio_manager.stop_recording()
        if result.get("status") == "stopped":
            # Transcribe the saved file
            logger.info(f"Transcribing file: {result['file']}")
            transcript = transcribe_audio(result["file"])
            logger.info("Batch transcription complete")
            return {
                "status": "success", 
                "file": result["file"], 
                "transcript": transcript
            }
        return result
    except Exception as e:
        logger.error(f"Failed to stop/transcribe: {e}")
        return {"status": "error", "message": str(e)}

@app.websocket("/stream")
async def websocket_endpoint(
    websocket: WebSocket, 
    language: str = Query("en"), 
    prompt: str = Query(None)
):
    """
    WebSocket endpoint for real-time transcription.
    Expects raw PCM audio bytes (float32, 16kHz, mono) from the client.
    Query params: language (e.g., 'en', 'ur'), prompt (for guiding output).
    """
    await websocket.accept()
    client_host = websocket.client.host
    logger.info(f"WebSocket connected: {client_host} | Lang: {language}")
    
    # Initialize transcriber with selected language and prompt
    try:
        transcriber = StreamingTranscriber(
            language=language, 
            initial_prompt=prompt if prompt else None
        )
    except Exception as e:
        logger.error(f"Failed to initialize transcriber: {e}")
        await websocket.close(code=1011)
        return
    
    stream_active = True
    last_log_time = time.time()
    
    try:
        while stream_active:
            # Periodic logging (every 30s)
            if time.time() - last_log_time > 30:
                mem_mb = get_memory_usage_mb()
                metrics = transcriber.get_metrics_summary()
                logger.info(
                    f"Session Stats | RAM: {mem_mb:.2f}MB | "
                    f"Avg Inference: {metrics.get('avg_inference_time', 0):.3f}s | "
                    f"Commits: {metrics.get('total_commits', 0)}"
                )
                last_log_time = time.time()

            # Receive audio chunk (bytes)
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array (float32)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            transcriber.add_chunk(audio_chunk)
            
            # Transcribe returns dict {committed: list, partial: str}
            result = transcriber.transcribe()
            
            if result:
                # FORMAT FOR FRONTEND BACKWARD COMPATIBILITY
                # Concatenate committed text + partial text
                committed_text = " ".join([seg["text"] for seg in result["committed"]])
                partial_text = result["partial"]
                
                # Combine them
                full_display_text = f"{committed_text} {partial_text}".strip()
                
                # Only send if there is content to display
                # Note: This might send "committed" repeatedly if we don't clear it from the transcriber output?
                # StreamingTranscriber.committed_segments grows indefinitely.
                # So we are sending the ENTIRE transcript every time.
                # This matches original behavior (User sees full history in live box).
                if full_display_text:
                    await websocket.send_text(full_display_text)
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {client_host}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        stream_active = False
        metrics = transcriber.get_metrics_summary()
        logger.info(f"Session Ended. Final Metrics: {metrics}")
        # Explicit clean up if needed
        del transcriber
        logger.info("Cleaned up transcriber session")
