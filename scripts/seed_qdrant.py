import sys
from pathlib import Path

# Automatically inject the project root (D:\hallucination-mitigation-engine) into sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("SeedQdrant")

# Sample curated reference corpus to ground factual queries
KNOWLEDGE_CORPUS = [
    {
        "topic": "Physics",
        "text": "The speed of light in a vacuum is exactly 299,792,458 meters per second. It is denoted by the letter c.",
    },
    {
        "topic": "Chemistry",
        "text": "Gold is a chemical element with the symbol Au and atomic number 79, making it one of the higher atomic number elements that occur naturally.",
    },
    {
        "topic": "Astronomy",
        "text": "SpaceX was founded in March 2002 by Elon Musk with the goal of reducing space transportation costs to enable the colonization of Mars.",
    },
    {
        "topic": "Geography",
        "text": "Paris is the capital and most populous city of France, situated along the Seine River in northern central France.",
    },
    {
        "topic": "Computer Science",
        "text": "Python is a high-level, general-purpose programming language conceived in the late 1980s by Guido van Rossum at CWI in the Netherlands.",
    },
]


def seed_database():
    logger.info(
        f"Connecting to embedded local Qdrant at: {settings.QDRANT_STORAGE_PATH}"
    )
    client = QdrantClient(path=str(settings.QDRANT_STORAGE_PATH))

    # Recreate collection
    client.recreate_collection(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    logger.info(
        f"Encoding {len(KNOWLEDGE_CORPUS)} reference knowledge documents..."
    )
    encoder = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    texts = [item["text"] for item in KNOWLEDGE_CORPUS]
    vectors = encoder.encode(texts, convert_to_numpy=True)

    points = []
    for idx, (doc, vector) in enumerate(zip(KNOWLEDGE_CORPUS, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload={"topic": doc["topic"], "text": doc["text"]},
            )
        )

    client.upsert(
        collection_name=settings.QDRANT_COLLECTION_NAME, points=points
    )
    logger.info(
        f"Successfully seeded {len(points)} evidence vectors into local Qdrant collection '{settings.QDRANT_COLLECTION_NAME}'."
    )


if __name__ == "__main__":
    seed_database()