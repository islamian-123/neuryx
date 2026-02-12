import os
from pathlib import Path
import shutil
from faster_whisper import download_model

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.available_models = ["tiny", "base", "small", "medium", "large-v3"]

    def list_models_status(self):
        """Check which models are downloaded."""
        status = {}
        for model in self.available_models:
            # simple check: folder exists in models_dir
            # faster_whisper downloads to a specific cache, but we can target a folder
            # If we used download_model(model_size, output_dir=...), it puts files there.
            model_path = self.models_dir / f"models--systran--faster-whisper-{model}"
            # The folder name might vary based on repo structure, but let's standardize download
            # Actually faster-whisper default structure is models--gw...
            # We will check if we have done a download to our custom dir.
            
            # Alternative: check if ANY folder in self.models_dir looks like it.
            # But let's just check if we have a folder named {model} or similar.
            
            # Let's rely on our download implementation:
            # We will download to self.models_dir / model_size
            updated_path = self.models_dir / model
            status[model] = updated_path.exists() and any(updated_path.iterdir())
        return status

    def download_model(self, model_size):
        if model_size not in self.available_models:
            raise ValueError(f"Invalid model size: {model_size}")
        
        output_path = self.models_dir / model_size
        print(f"Downloading {model_size} to {output_path}...")
        
        # This blocks, so we should run it in a thread/process in the API
        download_model(model_size, output_dir=str(output_path))
        return {"status": "success", "path": str(output_path)}

    def delete_model(self, model_size):
        path = self.models_dir / model_size
        if path.exists():
            shutil.rmtree(path)
            return {"status": "deleted"}
        return {"status": "not_found"}

    def get_model_path(self, model_size):
        path = self.models_dir / model_size
        if path.exists():
            return str(path)
        return None
