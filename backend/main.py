from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os
from backend.speech.audio_manager import AudioManager
from backend.speech.transcriber import transcribe_audio
# from backend.speech.transcriber_stream import StreamingTranscriber # REMOVED
from backend.routers import models
from backend.core.logger import get_logger
from backend.core.system_monitor import get_memory_usage_mb
import numpy as np
import asyncio
import time

logger = get_logger("backend.main")

app = FastAPI(title="NEURYX")
app.include_router(models.router)

# Serve built frontend
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

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
    # Pre-load model so first connection is fast
    from backend.speech.model_loader import ModelLoader
    ModelLoader.get_model("small")
    logger.info(f"Model pre-loaded. Memory: {get_memory_usage_mb():.2f} MB")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Neuryx Backend Shutting Down...")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

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

# Helper for blocking transcription
def run_transcription(file_path, language):
    from backend.speech.model_loader import ModelLoader
    from backend.nlp.transliterator import RomanTransliterator

    model = ModelLoader.get_model("small")
    
    start_time = time.time()
    
    segments, info = model.transcribe(
        file_path,
        language=None if language == "auto" else language,
        beam_size=5,
        best_of=3,
        temperature=0.0,
        condition_on_previous_text=True,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Transliteration Logic
    ROMAN_OUTPUT_ENABLED = True
    transliterator = RomanTransliterator()

    result_segments = []
    full_text_list = []
    
    for seg in segments:
        original_text = seg.text.strip()
        
        if ROMAN_OUTPUT_ENABLED:
            roman_text = transliterator.transliterate_text(original_text)
        else:
            roman_text = original_text
        
        full_text_list.append(roman_text)
        result_segments.append({
            "text": roman_text,
            "start": seg.start,
            "end": seg.end,
            "original_text": original_text 
        })
        
    full_text = " ".join(full_text_list)
    duration = time.time() - start_time
    
    return {
        "status": "success",
        "language": info.language,
        "duration": info.duration,
        "processing_time": duration,
        "full_text": full_text,
        "segments": result_segments
    }

@app.post("/transcribe")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = Form("auto")
):
    """
    Batch transcription endpoint.
    Accepts an audio file, saves it, and runs full-accuracy transcription.
    """
    try:
        from fastapi.concurrency import run_in_threadpool
        
        # 1. Save File
        upload_dir = os.path.join(os.path.dirname(__file__), "recordings")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Create unique filename
        timestamp = int(time.time())
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        
        logger.info(f"Received file for transcription: {filename} | Lang: {language}")
        
        with open(file_path, "wb") as buffer:
            # Write in chunks to handle large files
            while content := await file.read(1024 * 1024): # 1MB chunks
                buffer.write(content)
                
        # 2. Run Transcription in Threadpool (Optimization)
        # This prevents blocking the main event loop
        result = await run_in_threadpool(run_transcription, file_path, language)
        
        # 3. Persistence (History)
        # Save the result to a JSON file
        transcripts_dir = os.path.join(os.path.dirname(__file__), "transcripts") # e:\Neuryx\backend\transcripts
        os.makedirs(transcripts_dir, exist_ok=True)
        
        history_file = os.path.join(transcripts_dir, f"{timestamp}.json")
        import json
        
        # Add metadata for history
        history_data = {
            "id": str(timestamp),
            "timestamp": timestamp,
            "filename": filename,
            **result
        }
        
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved transcript to history: {history_file}")

        return JSONResponse(result)

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.get("/history")
async def get_history():
    """List all past transcriptions."""
    try:
        transcripts_dir = os.path.join(os.path.dirname(__file__), "transcripts")
        if not os.path.exists(transcripts_dir):
            return {"history": []}
            
        history_items = []
        import json
        
        # List all .json files
        files = sorted([f for f in os.listdir(transcripts_dir) if f.endswith(".json")], reverse=True)
        
        for f in files:
            try:
                with open(os.path.join(transcripts_dir, f), "r", encoding="utf-8") as file:
                    data = json.load(file)
                    # Return summary data
                    history_items.append({
                        "id": data.get("id"),
                        "timestamp": data.get("timestamp"),
                        "full_text": data.get("full_text", "")[:100] + "...", # Preview
                        "language": data.get("language"),
                        "duration": data.get("duration")
                    })
            except Exception as read_err:
                logger.warning(f"Failed to read history file {f}: {read_err}")
                continue
                
        return {"history": history_items}
        
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        return {"history": []}

@app.get("/history/{id}")
async def get_history_detail(id: str):
    """Get full details of a specific history item."""
    try:
        transcripts_dir = os.path.join(os.path.dirname(__file__), "transcripts")
        file_path = os.path.join(transcripts_dir, f"{id}.json")
        
        if not os.path.exists(file_path):
             return JSONResponse(status_code=404, content={"message": "History not found"})
             
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data
            
    except Exception as e:
        logger.error(f"History detail failed: {e}")
        return JSONResponse(status_code=500, content={"message": str(e)})

# Mount static assets AFTER all API/WS routes (so routes take priority)
if os.path.isdir(FRONTEND_DIR):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIR, "assets")), name="static-assets")
    app.mount("/", StaticFiles(directory=FRONTEND_DIR), name="static-root")

