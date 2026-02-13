from backend.speech.model_loader import ModelLoader
from backend.core.logger import get_logger

logger = get_logger(__name__)

def transcribe_audio(audio_path: str) -> str:
    logger.info(f"Transcribing audio file: {audio_path}")
    try:
        model = ModelLoader.get_model("small")
        
        segments, info = model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
            condition_on_previous_text=False
        )
    
        text = []
        for segment in segments:
            text.append(segment.text)
            
        full_text = " ".join(text)
        logger.info("Transcription successful")
        return full_text
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise e
