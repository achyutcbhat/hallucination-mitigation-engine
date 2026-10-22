import time
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.input import ProcessedInput
from src.schemas.response import SynthesizedResponse
from src.schemas.verification import (
    ClaimVerificationResult,
    EntailmentStatus,
    FinalPipelineOutput,
    RoutingDecision,
)
from src.stage_07_decision_routing.web_fallback import (
    search_google_for_query,
    resynthesize_with_web_evidence,
    verify_claims_against_text,
)

logger = setup_logger("Stage07-Router")

CONSERVATIVE_FALLBACK = (
    "I apologize, but even after multi-agent consensus, knowledge graph checking, "
    "and live web search verification, this claim could not be verified with sufficient confidence. "
    "To prevent hallucination, no ungrounded answer can be provided at this time."
)


def route_and_finalize(
    processed_input: ProcessedInput,
    synthesized_resp: SynthesizedResponse,
    initial_confidence: float,
    claim_verifications: list[ClaimVerificationResult],
    hitl_triggered: bool,
    start_time: float,
    requires_web_fallback: bool = False,
) -> FinalPipelineOutput:
    logger.info("=" * 60)
    logger.info("Stage 7: Corrective Decision Routing & Web Fallback")
    logger.info("=" * 60)

    # Check if any offline claim was contradicted by Knowledge Graph
    has_contradiction = any(
        c.status == EntailmentStatus.CONTRADICTED for c in claim_verifications
    )

    # =========================================================================
    # STEP 1: If Initial Score > 0.75 AND No Contradiction -> Return Directly
    # =========================================================================
    if (
        initial_confidence >= settings.VERIFIED_THRESHOLD
        and not has_contradiction
        and not requires_web_fallback
    ):
        logger.info(
            f"Step 1: Score {initial_confidence:.3f} >= {settings.VERIFIED_THRESHOLD}. Directly [VERIFIED]."
        )
        total_latency = round(time.time() - start_time, 2)
        return FinalPipelineOutput(
            original_query=processed_input.raw_query,
            final_answer=synthesized_resp.master_response,
            routing_decision=RoutingDecision.VERIFIED,
            overall_confidence_score=initial_confidence,
            hitl_triggered=hitl_triggered,
            web_fallback_used=False,
            claims_verified=claim_verifications,
            execution_time_seconds=total_latency,
        )

    # =========================================================================
    # STEP 2: If Initial Score < 0.75 (or Contradicted) -> Trigger Web Fallback
    # =========================================================================
    logger.warning(
        f"Step 2: Score {initial_confidence:.3f} < {settings.VERIFIED_THRESHOLD} "
        f"(Contradiction={has_contradiction}). Triggering Web Fallback..."
    )
    web_used = True

    # 1. Search Google in real-time for the user's question
    web_evidence = search_google_for_query(processed_input.raw_query)

    if web_evidence and len(web_evidence.strip()) >= 40:
        # 2. Re-synthesize a fresh answer using real-time Google evidence
        grounded_answer, web_claims = resynthesize_with_web_evidence(
            processed_input.raw_query, web_evidence
        )

        # 3. Verify the newly synthesized claims against Google Search evidence
        web_score, verified_web_claims = verify_claims_against_text(
            web_claims, web_evidence
        )
        logger.info(f"Re-Verification Score after Web Search: {web_score:.3f}")

        # ---------------------------------------------------------------------
        # SUB-CASE 2A: Re-verified Score >= 0.75 -> Final Verified Answer
        # ---------------------------------------------------------------------
        if web_score >= settings.VERIFIED_THRESHOLD:
            decision = RoutingDecision.VERIFIED
            final_score = web_score
            final_answer = (
                f"{grounded_answer}\n\n*(Verified via live Google Search)*"
            )
            logger.info("Web Fallback successfully upgraded answer to [VERIFIED].")

        # ---------------------------------------------------------------------
        # SUB-CASE 2B: 0.45 <= Re-verified Score < 0.75 -> Uncertain Answer
        # ---------------------------------------------------------------------
        elif web_score >= settings.UNCERTAIN_THRESHOLD:
            decision = RoutingDecision.UNCERTAIN
            final_score = web_score
            final_answer = (
                f"{grounded_answer}\n\n*(Note: This answer is provided with partial confidence from web sources)*"
            )
            logger.warning("Web Fallback resulted in [UNCERTAIN] answer.")

        # ---------------------------------------------------------------------
        # SUB-CASE 2C: Re-verified Score < 0.45 -> Fully Hallucinated (Apology)
        # ---------------------------------------------------------------------
        else:
            decision = RoutingDecision.FULLY_HALLUCINATED
            final_score = web_score
            final_answer = CONSERVATIVE_FALLBACK
            logger.error(
                "Web Fallback score < 0.45. Output is [FULLY_HALLUCINATED]. Apologizing."
            )

        final_claims = verified_web_claims

    # =========================================================================
    # STEP 3: If Web Search Was Unreachable / Failed Completely
    # =========================================================================
    else:
        logger.error(
            "Web Search returned no evidence or was unreachable. Evaluating initial score..."
        )
        if (
            initial_confidence >= settings.UNCERTAIN_THRESHOLD
            and not has_contradiction
        ):
            decision = RoutingDecision.UNCERTAIN
            final_score = initial_confidence
            final_answer = (
                f"{synthesized_resp.master_response}\n\n*(Note: Unverified offline response; web search unreachable)*"
            )
        else:
            decision = RoutingDecision.FULLY_HALLUCINATED
            final_score = initial_confidence
            final_answer = CONSERVATIVE_FALLBACK

        final_claims = claim_verifications

    total_latency = round(time.time() - start_time, 2)

    return FinalPipelineOutput(
        original_query=processed_input.raw_query,
        final_answer=final_answer,
        routing_decision=decision,
        overall_confidence_score=final_score,
        hitl_triggered=hitl_triggered,
        web_fallback_used=web_used,
        claims_verified=final_claims,
        execution_time_seconds=total_latency,
    )