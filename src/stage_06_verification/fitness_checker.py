import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.input import DecomposedQuestion
from src.schemas.verification import FitnessScoreResult

logger = setup_logger("FitnessChecker")
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    return _model


def evaluate_response_fitness(
    raw_query: str,
    sub_questions: list[DecomposedQuestion],
    master_response: str,
) -> FitnessScoreResult:
    model = get_model()

    # 1. Semantic alignment with main query
    q_vec = model.encode([raw_query])
    resp_vec = model.encode([master_response])
    overall_sim = float(cosine_similarity(q_vec, resp_vec)[0][0])
    overall_fitness = max(0.0, min(1.0, overall_sim))

    # 2. Sub-question coverage
    if sub_questions:
        sub_texts = [sq.sub_question for sq in sub_questions]
        sub_vecs = model.encode(sub_texts)
        sub_sims = cosine_similarity(sub_vecs, resp_vec)
        coverage = float(np.mean([max(0.0, s[0]) for s in sub_sims]))
    else:
        coverage = overall_fitness

    return FitnessScoreResult(
        fitness_score=round(overall_fitness, 3),
        subquestion_coverage=round(coverage, 3),
    )