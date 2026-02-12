from fastapi import APIRouter, HTTPException, BackgroundTasks
from backend.speech.model_manager import ModelManager

router = APIRouter(prefix="/models", tags=["models"])
manager = ModelManager()

@router.get("/")
def list_models():
    """List available models and their download status."""
    return manager.list_models_status()

@router.post("/{model_size}/download")
def download_model(model_size: str, background_tasks: BackgroundTasks):
    """Download a specific model size in the background."""
    try:
        # Check if valid
        if model_size not in manager.available_models:
             raise HTTPException(status_code=400, detail="Invalid model size")
        
        # Run in background to avoid blocking
        background_tasks.add_task(manager.download_model, model_size)
        return {"status": "download_started", "model": model_size}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{model_size}")
def delete_model(model_size: str):
    """Delete a downloaded model."""
    return manager.delete_model(model_size)
