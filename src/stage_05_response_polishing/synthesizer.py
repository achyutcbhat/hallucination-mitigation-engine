import json
import re
import time
import ollama
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.agent import AgentCandidateResponse
from src.schemas.response import FactualClaim, SynthesizedResponse

logger = setup_logger("Stage05-Synthesizer")

SYNTHESIS_SYSTEM_PROMPT = """You are an expert consensus synthesis engine.
You are provided with a user's question and multiple draft responses from independent AI models that have been semantically grouped as the majority consensus.

Your task:
1. Synthesize a single, highly articulate, clear, and comprehensive master answer strictly based on the verified points shared across the drafts.
2. Eliminate redundant phrasing and stylistic noise. Do NOT introduce new, unmentioned factual claims.
3. Deconstruct your master answer into a list of 2 to 5 atomic, verifiable factual claims (one fact per claim).

Respond strictly in valid JSON matching this schema:
{
  "master_response": "The unified synthesized answer text here...",
  "claims": [
    "Atomic factual claim 1",
    "Atomic factual claim 2"
  ]
}
Do not wrap your output in conversational markdown or explanations. Output pure JSON only.
"""


def _fallback_claim_extraction(text: str) -> list[str]:
    """Heuristic sentence splitter if LLM claim JSON parsing fails."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    claims = [s.strip() for s in sentences if len(s.strip()) > 15]
    return claims if claims else [text.strip()]


def synthesize_consensus_response(
    query: str,
    consensus_candidates: list[AgentCandidateResponse],
    cluster_id: int,
) -> SynthesizedResponse:
    logger.info("=" * 60)
    logger.info(
        f"Stage 5: Synthesizing Master Response from Cluster #{cluster_id} "
        f"({len(consensus_candidates)} consensus drafts)..."
    )
    logger.info("=" * 60)

    start_time = time.time()
    consensus_ids = [c.candidate_id for c in consensus_candidates]

    # Format the candidate drafts for the prompt
    drafts_text = ""
    for idx, cand in enumerate(consensus_candidates, 1):
        drafts_text += (
            f"\n--- Draft {idx} ({cand.agent_model}) ---\n{cand.response_text}\n"
        )

    user_prompt = f"User Question: {query}\n\nConsensus Drafts to Synthesize:\n{drafts_text}"

    client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    try:
        response = client.generate(
            model=settings.SYNTHESIZER_MODEL,
            system=SYNTHESIS_SYSTEM_PROMPT,
            prompt=user_prompt,
            options={"temperature": 0.2, "top_p": 0.9, "num_predict": 512},
        )
        latency = round(time.time() - start_time, 2)
        response_raw = response["response"].strip()

        # Clean backticks if model wrapped JSON in markdown
        if response_raw.startswith("```"):
            response_raw = response_raw.strip("`")
            if response_raw.startswith("json"):
                response_raw = response_raw[4:].strip()

        parsed = json.loads(response_raw)
        master_text = parsed.get("master_response", "").strip()
        raw_claims = parsed.get("claims", [])

        if not master_text:
            raise ValueError("Synthesizer produced empty master_response.")

    except Exception as e:
        logger.warning(
            f"Structured synthesis parse failed ({e}). Utilizing robust text fallback."
        )
        latency = round(time.time() - start_time, 2)
        # Fallback to the first high-confidence candidate in the cluster
        master_text = consensus_candidates[0].response_text
        raw_claims = _fallback_claim_extraction(master_text)

    # Format into Pydantic FactualClaim models
    factual_claims = [
        FactualClaim(claim_id=idx + 1, claim_text=c_text)
        for idx, c_text in enumerate(raw_claims)
        if c_text.strip()
    ]

    logger.info(
        f"Master response synthesized in {latency}s with {len(factual_claims)} atomic claims."
    )

    return SynthesizedResponse(
        master_response=master_text,
        consensus_candidate_ids=consensus_ids,
        extracted_claims=factual_claims,
        synthesis_latency=latency,
    )