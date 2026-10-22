from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.input import ProcessedInput, QueryIntent

logger = setup_logger("RiskEvaluator")

TEMPORAL_DRIFT_KEYWORDS = {
    "current",
    "currently",
    "latest",
    "recent",
    "recently",
    "now",
    "today",
    "this year",
    "2024",
    "2025",
    "2026",
}


def evaluate_risk(
    processed_input: ProcessedInput,
) -> tuple[float, dict[str, float], list[str]]:
    """Evaluates 4 risk vectors and returns (total_risk_score, factor_breakdown, risk_reasons)."""
    reasons = []

    # 1. Uncertainty / Ambiguity Risk (0.0 to 1.0)
    ambiguity_risk = processed_input.ambiguity_score
    if ambiguity_risk > 0.3:
        reasons.append(
            f"High ambiguity detected ({ambiguity_risk:.2f}) with unclear references: {processed_input.unclear_terms}"
        )

    # 2. Missing Context Risk
    # Queries lacking named entities or having zero domain keywords are high-risk
    entity_count = len(processed_input.context.named_entities)
    keyword_count = len(processed_input.context.domain_keywords)

    if entity_count == 0 and keyword_count <= 1:
        missing_context_risk = 0.85
        reasons.append(
            "Missing key entities or domain context to ground the answer."
        )
    elif entity_count == 0:
        missing_context_risk = 0.50
        reasons.append("No specific named entities identified in prompt.")
    else:
        missing_context_risk = 0.10

    # 3. Concept Drift / Outdated Info Risk
    raw_lower = processed_input.raw_query.lower()
    drift_hits = [w for w in TEMPORAL_DRIFT_KEYWORDS if w in raw_lower]
    if drift_hits:
        concept_drift_risk = 0.75
        reasons.append(
            f"Query is time-sensitive or subject to concept drift: {drift_hits}"
        )
    else:
        concept_drift_risk = 0.10

    # 4. Knowledge Difficulty Risk
    sub_q_count = len(processed_input.sub_questions)
    if sub_q_count >= 3:
        difficulty_risk = 0.60
        reasons.append(
            f"Multi-hop complex reasoning requirement ({sub_q_count} sub-questions)."
        )
    elif processed_input.intent == QueryIntent.REASONING:
        difficulty_risk = 0.40
    else:
        difficulty_risk = 0.15

    # Weighted Risk Formula:
    # 35% Ambiguity + 30% Missing Context + 20% Concept Drift + 15% Difficulty
    total_risk = (
        0.35 * ambiguity_risk
        + 0.30 * missing_context_risk
        + 0.20 * concept_drift_risk
        + 0.15 * difficulty_risk
    )

    # Guardrail #1 Critical Override:
    # In single-turn mode, if the query contains unresolved pronoun references ('it', 'this', 'they')
    # and ZERO named entities to anchor them, models will inevitably hallucinate the subject.
    if ambiguity_risk >= 0.30 and entity_count == 0:
        total_risk = max(total_risk, 0.75)
        reasons.append(
            "Critical single-turn ambiguity: Query contains unresolved pronoun references with no grounding entity."
        )
    elif ambiguity_risk >= 0.50:
        total_risk = max(total_risk, 0.70)
        reasons.append("High compound ambiguity detected across multiple tokens.")

    total_risk = round(min(total_risk, 1.0), 3)

    factor_breakdown = {
        "ambiguity": ambiguity_risk,
        "missing_context": missing_context_risk,
        "concept_drift": concept_drift_risk,
        "knowledge_difficulty": difficulty_risk,
    }

    logger.info(
        f"Risk evaluated: Total={total_risk} (Threshold={settings.RISK_HIGH_THRESHOLD})"
    )
    return total_risk, factor_breakdown, reasons