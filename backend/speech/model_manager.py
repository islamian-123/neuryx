from pathlib import Path
import shutil
from faster_whisper import download_model
from backend.core.logger import get_logger

logger = get_logger(__name__)

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.available_models = ["tiny", "base", "small", "medium", "large-v3"]

    def list_models_status(self):
        """Check which models are downloaded."""
        status = {}
        for model in self.available_models:
            # We assume download_model puts files in model_dir/model_size
            updated_path = self.models_dir / model
            is_downloaded = updated_path.exists() and any(updated_path.iterdir())
            status[model] = is_downloaded
            # logger.debug(f"Model {model} status: {'downloaded' if is_downloaded else 'missing'}")
        return status

    def download_model(self, model_size):
        if model_size not in self.available_models:
            logger.error(f"Invalid model size requested: {model_size}")
            raise ValueError(f"Invalid model size: {model_size}")
        
        output_path = self.models_dir / model_size
        logger.info(f"Downloading {model_size} to {output_path}...")
        
        try:
            # This blocks, so we should run it in a thread/process in the API
            download_model(model_size, output_dir=str(output_path))
            logger.info(f"Successfully downloaded {model_size}")
            return {"status": "success", "path": str(output_path)}
        except Exception as e:
            logger.error(f"Failed to download model {model_size}: {e}")
            raise e

    def delete_model(self, model_size):
        path = self.models_dir / model_size
        if path.exists():
            try:
                shutil.rmtree(path)
                logger.info(f"Deleted model {model_size}")
                return {"status": "deleted"}
            except Exception as e:
                logger.error(f"Failed to delete model {model_size}: {e}")
                raise e
        logger.warning(f"Model {model_size} not found used for deletion")
        return {"status": "not_found"}

    def get_model_path(self, model_size):
        path = self.models_dir / model_size
        if path.exists():
            return str(path)
        return None
