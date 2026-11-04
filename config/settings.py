from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Base Paths on D: Drive
    BASE_DIR: Path = Path(r"D:\hallucination-mitigation-engine")
    DATA_DIR: Path = BASE_DIR / "data"
    QDRANT_STORAGE_PATH: Path = DATA_DIR / "qdrant_storage"

    # Stage 1: Input Processing & Decomposition
    DECOMPOSER_MODEL: str = "qwen2.5:3b"
    SPACY_MODEL: str = "en_core_web_sm"

    # Stage 2: Risk Assessment Thresholds
    RISK_HIGH_THRESHOLD: float = 0.65  # Score >= 0.65 triggers Human-In-The-Loop

    # Stage 3: Multi-Agent Ensemble (3 Heterogeneous Models via Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    AGENT_MODELS: list[str] = [
        "qwen2.5:3b",  # Alibaba (High factual knowledge)
        "gemma2:2b",  # Google (High precision)
        "deepseek-r1:1.5b",  # DeepSeek (Step-by-step reasoning)
    ]
    CANDIDATES_PER_AGENT: int = 2  # Total 3 x 2 = 6 responses

    # Stage 4: Semantic Graph Clustering
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    GRAPH_SIMILARITY_THRESHOLD: float = 0.72  # Cosine similarity for edge creation

    # Stage 5: Response Reshaping
    SYNTHESIZER_MODEL: str = "qwen2.5:3b"

    # Stage 6: Dual Guardrail Verification
    NLI_MODEL_NAME: str = "cross-encoder/nli-deberta-v3-base"
    QDRANT_COLLECTION_NAME: str = "evidence_store"
    GOOGLE_KG_API_KEY: str = " " #paste your google Knowledge gragh api key

    # Stage 7: Decision Thresholds & Gemini Google Search Grounding
    # 👇👇👇 PASTE YOUR GEMINI API KEY HERE 👇👇👇
    GEMINI_API_KEY: str = " "

    VERIFIED_THRESHOLD: float = 0.75  # Score > 0.75 -> Final Answer
    UNCERTAIN_THRESHOLD: float = 0.45  # 0.45 <= Score <= 0.75 -> Web Fallback
    # Score < 0.45 -> Conservative Fallback ("I don't know")


settings = Settings()
