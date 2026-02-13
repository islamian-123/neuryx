import gc
import logging
from faster_whisper import WhisperModel
from backend.core.logger import get_logger
from backend.core.system_monitor import get_memory_usage_mb

logger = get_logger(__name__)

class ModelLoader:
    _instance = None
    _model = None
    _current_model_size = None
    
    @classmethod
    def get_model(cls, model_size="small", device="cpu", compute_type="int8"):
        """
        Returns the singleton instance of the WhisperModel.
        If a different model size is requested, it unloads the previous one.
        """
        # If model exists and size matches, return it
        if cls._model is not None and cls._current_model_size == model_size:
            return cls._model
            
        # If model exists but size differs, unload it
        if cls._model is not None:
             cls.unload_model()
             
        # Load new model
        logger.info(f"Loading Whisper Model: {model_size}...")
        start_mem = get_memory_usage_mb()
        
        try:
            cls._model = WhisperModel(model_size, device=device, compute_type=compute_type)
            cls._current_model_size = model_size
            
            end_mem = get_memory_usage_mb()
            logger.info(f"Model loaded. RAM increased by {end_mem - start_mem:.2f} MB. Total: {end_mem:.2f} MB")
            return cls._model
        except Exception as e:
            logger.error(f"Failed to load model {model_size}: {e}")
            raise e

    @classmethod
    def unload_model(cls):
        """Explicitly unloads the model and forces garbage collection."""
        if cls._model:
            logger.info(f"Unloading model {cls._current_model_size}...")
            start_mem = get_memory_usage_mb()
            
            del cls._model
            cls._model = None
            cls._current_model_size = None
            
            gc.collect()
            
            end_mem = get_memory_usage_mb()
            logger.info(f"Model unloaded. RAM freed: {start_mem - end_mem:.2f} MB. Total: {end_mem:.2f} MB")

# Global accessor
def get_model(model_size="small"):
    return ModelLoader.get_model(model_size)
