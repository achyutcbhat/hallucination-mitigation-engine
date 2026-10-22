from config.logging_config import setup_logger
from src.schemas.input import ProcessedInput
from src.schemas.response import SynthesizedResponse
from src.schemas.graph import ClusteringResult
from src.schemas.verification import (
    ClaimVerificationResult,
    EntailmentStatus,
    FitnessScoreResult,
)
from src.stage_06_verification.grounder import verify_claim_against_local_store
from src.stage_06_verification.fitness_checker import evaluate_response_fitness

logger = setup_logger("Stage06-Verifier")


def run_dual_verification(
    processed_input: ProcessedInput,
    synthesized_resp: SynthesizedResponse,
    clustering_result: ClusteringResult | None = None,
) -> tuple[float, list[ClaimVerificationResult], FitnessScoreResult]:
    logger.info("=" * 60)
    logger.info("Running Stage 6: Dual Guardrail Verification & Scoring")
    logger.info("=" * 60)

    # 1. Ground claims against local Qdrant store
    claim_results: list[ClaimVerificationResult] = []
    has_local_support = False
    has_contradiction = False

    for claim in synthesized_resp.extracted_claims:
        res = verify_claim_against_local_store(claim)
        claim_results.append(res)

        if res.status == EntailmentStatus.SUPPORTED:
            has_local_support = True
        elif res.status == EntailmentStatus.CONTRADICTED:
            has_contradiction = True

    # 2. Check Fitness (Query-Answer alignment)
    fitness_res = evaluate_response_fitness(
        raw_query=processed_input.raw_query,
        sub_questions=processed_input.sub_questions,
        master_response=synthesized_resp.master_response,
    )

    # 3. Calculate Multi-Agent Consensus Strength from Stage 4
    consensus_ratio = 1.0
    if clustering_result:
        primary_size = clustering_result.primary_cluster.size
        total_nodes = len(clustering_result.nodes)
        avg_sim = clustering_result.primary_cluster.avg_internal_similarity
        # Consensus strength is high if majority of models agreed with high internal similarity
        consensus_ratio = (primary_size / max(1, total_nodes)) * avg_sim
        logger.info(
            f"Multi-Agent Consensus Strength: {consensus_ratio:.3f} "
            f"({primary_size}/{total_nodes} models in cluster, AvgSim={avg_sim})"
        )

    # 4. Final Scoring Logic:
    if has_contradiction:
        # Local knowledge base contradicted the claim -> Heavy penalty
        overall_confidence = 0.20
    elif has_local_support:
        # Local knowledge base verified the claim -> Top confidence
        overall_confidence = round(
            0.60 * 1.0 + 0.40 * fitness_res.fitness_score, 3
        )
    else:
        # General knowledge query: Rely on Heterogeneous Consensus + Fitness
        # If heterogeneous models (Google, DeepSeek, Alibaba) agree and fitness is high:
        overall_confidence = round(
            0.50 * consensus_ratio + 0.50 * fitness_res.fitness_score, 3
        )

    logger.info(f"Final Verification Confidence Score: {overall_confidence}")
    return overall_confidence, claim_results, fitness_res