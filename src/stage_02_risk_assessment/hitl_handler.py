import ollama
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("HITLHandler")

CLARIFICATION_SYSTEM_PROMPT = """You are an interactive conversational AI guardrail. 
The user asked a query that has high risk of hallucination due to ambiguity or missing context.
Given the original query and the risk reasons, formulate a single, polite, and specific question asking the user for the missing details needed to answer accurately.

Do not answer the query. Only ask the clarification question.
"""


def generate_clarification_prompt(
    raw_query: str, risk_reasons: list[str]
) -> str:
    """Generates an intelligent clarification request via local LLM or fallback heuristic."""
    try:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        prompt_content = f"Original Query: '{raw_query}'\nIdentified Risks:\n- " + "\n- ".join(
            risk_reasons
        )
        response = client.generate(
            model=settings.DECOMPOSER_MODEL,
            system=CLARIFICATION_SYSTEM_PROMPT,
            prompt=prompt_content,
            options={"temperature": 0.2, "num_predict": 128},
        )
        return response["response"].strip()
    except Exception as e:
        logger.warning(
            f"Ollama clarification prompt generation failed ({e}), using rule-based fallback."
        )
        return (
            f"Your request is ambiguous or missing context ({', '.join(risk_reasons[:2])}). "
            f"Could you please specify more details or clarify the target of your question?"
        )


def request_clarification_cli(
    raw_query: str, risk_reasons: list[str]
) -> tuple[str, bool]:
    """Interactive CLI prompt for Human-in-the-Loop.
    Returns: (refined_query, clarification_provided)
    """
    print("\n" + "=" * 70)
    print("⚠️ [PRE-GENERATION GUARDRAIL TRIGGERED: HIGH RISK OF HALLUCINATION]")
    print(f"Original Query: \"{raw_query}\"")
    print("Reasons:")
    for reason in risk_reasons:
        print(f"  • {reason}")

    clarification_msg = generate_clarification_prompt(raw_query, risk_reasons)
    print(f"\nSystem Request: {clarification_msg}")
    print("=" * 70)

    user_response = input(
        "Enter your clarification (or press Enter to abort): "
    ).strip()
    if not user_response:
        print("❌ No clarification provided. Strict Guardrail Rejection activated.\n")
        return raw_query, False

    # Intelligently construct refined query without lingering vague pronouns:
    resp_clean = user_response.rstrip("?.")
    if resp_clean.lower().startswith(("how", "what", "who", "why", "where", "when", "is", "can")):
        # User entered a complete replacement question
        refined_query = user_response
    elif " it " in f" {raw_query.lower()} ":
        # Replace the vague pronoun 'it' directly with the subject
        import re
        refined_query = re.sub(r"\bit\b", user_response, raw_query, flags=re.IGNORECASE)
    else:
        refined_query = f"{raw_query} regarding {user_response}"

    print(f"✅ Grounded Query: \"{refined_query}\"\n")
    return refined_query, True