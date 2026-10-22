import gc
import torch
from config.logging_config import setup_logger

logger = setup_logger("MemoryManager")


def flush_cuda_memory():
    """Forces garbage collection and releases unoccupied PyTorch CUDA cached memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        logger.debug("CUDA memory cache flushed successfully.")