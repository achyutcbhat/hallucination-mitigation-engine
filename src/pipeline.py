import time
from config.logging_config import setup_logger
from src.schemas.verification import FinalPipelineOutput
from src.schemas.verification import FinalPipelineOutput, RoutingDecision
from src.stage_01_input_processing.processor import process_user_input
from src.stage_02_risk_assessment.coordinator import run_pre_generation_guardrail
from src.stage_03_multiagent_engine.generator import run_multiagent_generation
from src.stage_04_graph_clustering.clusterer import run_graph_clustering
from src.stage_05_response_polishing.synthesizer import (
    synthesize_consensus_response,
)
from src.stage_06_verification.verifier import run_dual_verification
from src.stage_07_decision_routing.router import route_and_finalize

logger = setup_logger("MasterPipeline")


def run_pipeline(
    raw_query: str, interactive_hitl: bool = True
) -> FinalPipelineOutput:
    start_time = time.time()
    logger.info("\n" + "#" * 70)
    logger.info(f"STARTING FULL PIPELINE FOR QUERY: '{raw_query}'")
    logger.info("#" * 70)

    # Stage 1: Input Processing (spaCy + Qwen2.5 decomposition)
    processed_input = process_user_input(raw_query)

    # Stage 2: Pre-Generation Risk Assessment & HITL (Guardrail #1)
    processed_input, hitl_triggered, risk_score, rejected = run_pre_generation_guardrail(
        processed_input, interactive=interactive_hitl
    )

    # Strict Guardrail Rejection: If query lacks context and clarification was missing or invalid,
    # abort Stage 3 immediately to prevent hallucination snowballing and save GPU compute.
        # Strict Guardrail Rejection: If query lacks context and clarification was missing or invalid,
    # abort Stage 3 immediately to prevent hallucination snowballing and save GPU compute.
    if rejected:
        logger.warning("Pipeline aborted at Guardrail #1 due to ungrounded input.")
        total_latency = round(time.time() - start_time, 2)
        return FinalPipelineOutput(
            original_query=raw_query,
            final_answer=(
                "⚠️ Query Aborted by Pre-Generation Guardrail (Guardrail #1):\n"
                "The question lacks background context or an identifiable subject (e.g., unresolved pronoun like 'it' or 'this'). "
                "In a single-turn architecture, context cannot be assumed without clarification. "
                "Please specify what system, topic, or object you are asking about."
            ),
            routing_decision=RoutingDecision.UNCERTAIN,
            overall_confidence_score=0.0,
            hitl_triggered=True,
            web_fallback_used=False,
            claims_verified=[],
            execution_time_seconds=total_latency,
        )

    # Stage 3: Multi-Agent Engine (3 heterogeneous agents x 2 runs = 6 outputs)
    candidate_pool = run_multiagent_generation(processed_input)

    # Stage 4: Graph-Based Semantic Clustering (MiniLM + NetworkX)
    clustering_result = run_graph_clustering(candidate_pool)

    # Filter candidates belonging only to the Primary Consensus Cluster
    primary_member_ids = set(
        clustering_result.primary_cluster.member_node_ids
    )
    consensus_candidates = [
        c
        for c in candidate_pool.candidates
        if c.candidate_id in primary_member_ids
    ]

    # Stage 5: Response Reshaping & Polishing (Qwen2.5 synthesis + claim extraction)
    synthesized_resp = synthesize_consensus_response(
        query=processed_input.raw_query,
        consensus_candidates=consensus_candidates,
        cluster_id=clustering_result.primary_cluster.cluster_id,
    )

    # Stage 6: Dual Guardrail Verification & Scoring (Qdrant + DeBERTa NLI + Consensus)
    confidence_score, claims, fitness = run_dual_verification(
        processed_input=processed_input,
        synthesized_resp=synthesized_resp,
        clustering_result=clustering_result,
    )

    # Stage 7: Decision Routing & Dynamic Web Fallback
    final_output = route_and_finalize(
        processed_input=processed_input,
        synthesized_resp=synthesized_resp,
        initial_confidence=confidence_score,
        claim_verifications=claims,
        hitl_triggered=hitl_triggered,
        start_time=start_time,
    )

    logger.info(
        f"PIPELINE COMPLETE: Decision={final_output.routing_decision.value} "
        f"in {final_output.execution_time_seconds}s"
    )
    return final_output