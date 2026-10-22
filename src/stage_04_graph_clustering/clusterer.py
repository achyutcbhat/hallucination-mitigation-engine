from config.logging_config import setup_logger
from src.schemas.agent import CandidatePool
from src.schemas.graph import ClusteringResult
from src.stage_04_graph_clustering.similarity_matrix import (
    compute_pairwise_similarity,
)
from src.stage_04_graph_clustering.consensus_graph import build_consensus_graph

logger = setup_logger("Stage04-Clusterer")


def run_graph_clustering(candidate_pool: CandidatePool) -> ClusteringResult:
    logger.info("=" * 60)
    logger.info("Running Stage 4: Graph-Based Semantic Clustering")
    logger.info("=" * 60)

    texts = [c.response_text for c in candidate_pool.candidates]

    # 1. Compute pairwise similarity matrix
    sim_matrix = compute_pairwise_similarity(texts)

    # 2. Build graph and extract consensus cluster
    result = build_consensus_graph(candidate_pool.candidates, sim_matrix)

    return result