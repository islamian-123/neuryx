from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.speech.audio_manager import AudioManager
from backend.speech.transcriber import transcribe_audio
from backend.speech.transcriber_stream import StreamingTranscriber
from backend.routers import models
import numpy as np

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

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/record/start")
def start_recording():
    """Start recording on the server's default microphone."""
    try:
        return audio_manager.start_recording()
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/record/stop")
def stop_recording():
    """Stop recording, save file, and transcribe."""
    try:
        result = audio_manager.stop_recording()
        if result.get("status") == "stopped":
            # Transcribe the saved file
            transcript = transcribe_audio(result["file"])
            return {
                "status": "success", 
                "file": result["file"], 
                "transcript": transcript
            }
        return result
    except Exception as e:
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
    
    # Initialize transcriber with selected language and prompt
    transcriber = StreamingTranscriber(
        language=language, 
        initial_prompt=prompt if prompt else None
    )
    
    print(f"Stream started: Lang={language}, Prompt={prompt}")
    
    try:
        while True:
            # Receive audio chunk (bytes)
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array (float32)
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            transcriber.add_chunk(audio_chunk)
            
            text = transcriber.transcribe()
            if text:
                await websocket.send_text(text)
                
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {e}")
        await websocket.close()
