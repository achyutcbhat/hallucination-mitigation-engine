import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("SimilarityMatrix")

# Global singleton to load embedding model once into memory
_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        logger.info(
            f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}..."
        )
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _embedding_model


def compute_pairwise_similarity(texts: list[str]) -> np.ndarray:
    """Computes an N x N cosine similarity matrix across all candidate texts."""
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix