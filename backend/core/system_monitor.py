import logging
import psutil
import os

try:
    from backend.core.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)

def get_memory_usage_mb() -> float:
    """Returns the current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024

def log_system_status(model_loaded: bool = False, active_streams: int = 0):
    """Logs current system status including memory usage."""
    mem_mb = get_memory_usage_mb()
    logger.info(
        f"System Status | RAM: {mem_mb:.2f} MB | "
        f"Model Loaded: {model_loaded} | Active Streams: {active_streams}"
    )
