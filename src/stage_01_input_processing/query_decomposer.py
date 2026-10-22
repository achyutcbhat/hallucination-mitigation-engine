import json
import ollama
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.input import DecomposedQuestion

logger = setup_logger("QueryDecomposer")

SYSTEM_PROMPT = """You are a query decomposition specialist. 
Your task is to break down a user's question into 1 to 3 atomic, independent, and logically ordered sub-questions.
If the query is already atomic and simple, return just 1 sub-question.

Respond strictly in valid JSON matching this schema:
{
  "sub_questions": [
    {"id": 1, "sub_question": "...", "target_entity": "..."},
    {"id": 2, "sub_question": "...", "target_entity": "..."}
  ]
}
Do not include any conversational filler, markdown formatting (do not wrap in ```json), or explanation.
"""


def decompose_query(raw_query: str) -> list[DecomposedQuestion]:
    # Fallback default if Ollama is unreachable or parsing fails
    fallback = [
        DecomposedQuestion(id=1, sub_question=raw_query, target_entity=None)
    ]

    try:
        client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        response = client.generate(
            model=settings.DECOMPOSER_MODEL,
            system=SYSTEM_PROMPT,
            prompt=f"Decompose this question: {raw_query}",
            options={"temperature": 0.1, "num_predict": 256},
        )

        response_text = response["response"].strip()
        # Clean out potential markdown backticks if returned
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        data = json.loads(response_text)
        sub_qs = []
        for item in data.get("sub_questions", []):
            sub_qs.append(
                DecomposedQuestion(
                    id=item.get("id", len(sub_qs) + 1),
                    sub_question=item["sub_question"],
                    target_entity=item.get("target_entity"),
                )
            )

        return sub_qs if sub_qs else fallback

    except Exception as e:
        logger.warning(
            f"Ollama decomposition failed ({e}). Falling back to atomic representation."
        )
        return fallback