from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.input import ProcessedInput
from src.stage_01_input_processing.processor import process_user_input
from src.stage_02_risk_assessment.risk_evaluator import evaluate_risk
from src.stage_02_risk_assessment.hitl_handler import request_clarification_cli

logger = setup_logger("Stage02-Coordinator")


def run_pre_generation_guardrail(
    processed_input: ProcessedInput,
    interactive: bool = True,
) -> tuple[ProcessedInput, bool, float, bool]:
    """Evaluates risk and coordinates HITL.

    If HIGH RISK and interactive=True, triggers HITL clarification and re-checks the updated query.
    Returns: (final_processed_input, hitl_triggered, risk_score, rejected)
    """
    initial_risk, breakdown, reasons = evaluate_risk(processed_input)
    is_high_risk = initial_risk >= settings.RISK_HIGH_THRESHOLD

    # 1. Low risk queries pass directly to Stage 3
    if not is_high_risk:
        logger.info(
            f"Query passed Pre-Generation Guardrail (Risk={initial_risk:.3f} < {settings.RISK_HIGH_THRESHOLD}). Proceeding to Multiagent Engine."
        )
        return processed_input, False, initial_risk, False

    logger.warning(
        f"HIGH RISK DETECTED (Score={initial_risk:.3f} >= {settings.RISK_HIGH_THRESHOLD}). Initiating Human-in-the-Loop."
    )

    # 2. If non-interactive mode (API), cannot ask user -> Strict Guardrail Rejection
    if not interactive:
        logger.warning("Non-interactive mode: High-risk ambiguous query rejected.")
        return processed_input, True, initial_risk, True

    # 3. Interactive CLI: Ask human for clarification
    refined_query, clarified = request_clarification_cli(
        processed_input.raw_query, reasons
    )

    if not clarified:
        # User pressed Enter without typing clarification -> Strict Guardrail Rejection
        logger.warning("User skipped clarification. Strict Guardrail Rejection triggered.")
        return processed_input, True, initial_risk, True

    # 4. RE-CHECK: Re-process updated query through Stage 1 & Stage 2
    logger.info(f"Re-evaluating updated query with user context: '{refined_query}'")
    updated_input = process_user_input(refined_query)
    rechecked_risk, _, new_reasons = evaluate_risk(updated_input)

    if rechecked_risk >= settings.RISK_HIGH_THRESHOLD:
        # Clarification was also vague/ambiguous
        print(
            f"❌ Clarification is still too ambiguous (Risk={rechecked_risk:.3f} >= {settings.RISK_HIGH_THRESHOLD}). Aborting generation.\n"
        )
        logger.warning(
            f"Updated query remained high risk ({rechecked_risk:.3f}). Strict Guardrail Rejection triggered."
        )
        return updated_input, True, rechecked_risk, True

    # Successfully resolved!
    logger.info(
        f"✅ Clarification resolved ambiguity: Risk reduced from {initial_risk:.3f} -> {rechecked_risk:.3f}. Proceeding to Multiagent Engine."
    )
    return updated_input, True, rechecked_risk, False