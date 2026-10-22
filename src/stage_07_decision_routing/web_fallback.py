import html
import json
import re
import numpy as np
import ollama
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.response import FactualClaim
from src.schemas.verification import ClaimVerificationResult, EntailmentStatus
from src.stage_06_verification.grounder import get_nli_model

logger = setup_logger("DuckDuckGoFallback")

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

WEB_SYNTHESIS_PROMPT = """You are a rigorous, objective factual synthesizer.
You are given a user question and fresh search evidence.

CORE PRINCIPLES:
1. OBJECTIVE EVIDENCE GROUNDING: Synthesize an answer strictly derived from the provided search evidence. Never invent dates, names, or events not present in the evidence.
2. PREMISE VALIDATION: If the user question presumes an event, role, or action that the evidence contradicts, disproves, or shows never happened, you MUST explicitly state that the premise is incorrect or unverified based on the evidence. Do NOT invent details to satisfy a false or unproven premise.
3. CLAIM EXTRACTION: Extract 1 to 2 atomic, independently verifiable factual claims from your answer.

Respond strictly in valid JSON:
{
  "answer": "Your direct, factual response...",
  "claims": ["Atomic claim 1", "Atomic claim 2"]
}
Do not include markdown codeblocks or commentary. Output raw JSON only.
"""

SPAM_KEYWORDS = {
    "download",
    "browser",
    "chrome",
    "install",
    "coupon",
    "promo",
    "free trial",
    "shop",
    "buy now",
}


def search_google_for_query(query: str, max_results: int = 5) -> str:
    """Live Web Search via DuckDuckGo with ad-filtering."""
    clean_q = re.sub(r"[^\w\s]", " ", query).strip()
    logger.info(
        f"Querying live web search (DuckDuckGo) for: '{clean_q}'..."
    )
    snippets = []

    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(clean_q, max_results=max_results + 3, backend="lite")
            )
            for r in results:
                body = r.get("body", "")
                title = r.get("title", "")
                link = r.get("href", "")

                if any(
                    spam in (title + " " + body).lower()
                    for spam in SPAM_KEYWORDS
                ):
                    continue

                if len(body.strip()) > 25:
                    clean_body = html.unescape(body).strip()
                    snippets.append(f"{title}: {clean_body} ({link})")

    except Exception as e:
        logger.warning(f"DuckDuckGo search warning ({e}).")

    # Add news for recency
    try:
        with DDGS() as ddgs:
            news_results = list(ddgs.news(clean_q, max_results=3))
            for n in news_results:
                title = n.get("title", "")
                body = n.get("body", "")
                date = n.get("date", "")
                if title and body:
                    snippets.append(f"News ({date}) - {title}: {body}")
    except Exception:
        pass

    final_evidence = "\n\n".join(snippets[:max_results]).strip()
    if final_evidence:
        logger.info(
            f"Web search retrieved {len(snippets)} clean factual snippets ({len(final_evidence)} chars)."
        )
        return final_evidence

    return ""


def resynthesize_with_web_evidence(
    query: str, evidence: str
) -> tuple[str, list[FactualClaim]]:
    """Synthesizes a proper up-to-date response from verified search results using local Qwen2.5."""
    logger.info("Synthesizing updated answer from live search evidence...")
    client = ollama.Client(host=settings.OLLAMA_BASE_URL)
    user_prompt = f"User Question: {query}\n\nSearch Evidence:\n{evidence}"

    try:
        response = client.generate(
            model=settings.SYNTHESIZER_MODEL,
            system=WEB_SYNTHESIS_PROMPT,
            prompt=user_prompt,
            options={"temperature": 0.1, "num_predict": 300},
        )
        raw_text = response["response"].strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.strip("`")
            if raw_text.startswith("json"):
                raw_text = raw_text[4:].strip()

        data = json.loads(raw_text)
        answer = data.get("answer", "").strip()
        claims_list = data.get("claims", [])
        claims = [
            FactualClaim(claim_id=idx + 1, claim_text=c)
            for idx, c in enumerate(claims_list)
        ]
        return answer, claims
    except Exception as e:
        logger.warning(
            f"Local synthesis parse failed ({e}), using direct text."
        )
        return (
            f"Based on recent verified search reports: {evidence[:350]}...",
            [FactualClaim(claim_id=1, claim_text=evidence[:100])],
        )


def verify_claims_against_text(
    claims: list[FactualClaim], evidence: str
) -> tuple[float, list[ClaimVerificationResult]]:
    """Performs multi-snippet cross-checking with strict contradiction detection."""
    nli = get_nli_model()
    results = []
    scores = []

    # Split evidence into individual snippets
    raw_snippets = [
        s.strip() for s in evidence.split("\n\n") if len(s.strip()) > 30
    ]

    for claim in claims:
        claim_text = claim.claim_text.strip()
        
        # Test the claim against ALL retrieved snippets
        best_entail = 0.0
        worst_contra = 0.0
        best_snippet_for_claim = raw_snippets[0] if raw_snippets else evidence[:250]

        for snip in raw_snippets:
            pred = nli.predict([(snip, claim_text)])
            exp_scores = np.exp(pred[0])
            probs = exp_scores / np.sum(exp_scores)
            p_contra, p_entail, p_neutral = float(probs[0]), float(probs[1]), float(probs[2])

            if p_contra > worst_contra:
                worst_contra = p_contra
                if p_contra >= 0.50:
                    best_snippet_for_claim = snip

            if p_entail > best_entail:
                best_entail = p_entail
                if worst_contra < 0.50:
                    best_snippet_for_claim = snip

        # STRICT NLI SCORING RULES:
        # 1. Contradiction takes absolute precedence: If ANY snippet refutes the claim -> REJECT
        if worst_contra >= 0.50:
            status = EntailmentStatus.CONTRADICTED
            claim_score = 0.05
            logger.warning(
                f"Claim CONTRADICTED by web evidence (p_contra={worst_contra:.3f}): '{claim_text}'"
            )
        # 2. Real Entailment required for SUPPORTED:
        elif best_entail >= 0.60:
            status = EntailmentStatus.SUPPORTED
            claim_score = min(0.95, 0.75 + (best_entail * 0.20))
        # 3. Neutral / Missing Evidence:
        else:
            status = EntailmentStatus.INSUFFICIENT
            claim_score = 0.20
            logger.info(
                f"Claim has INSUFFICIENT evidence (p_entail={best_entail:.3f}): '{claim_text}'"
            )

        scores.append(claim_score)
        results.append(
            ClaimVerificationResult(
                claim_id=claim.claim_id,
                claim_text=claim_text,
                status=status,
                nli_entailment_score=round(claim_score, 3),
                evidence_snippet=best_snippet_for_claim[:250] + "...",
                source_type="LIVE_WEB_SEARCH",
            )
        )

    avg_score = float(sum(scores) / len(scores)) if scores else 0.20
    logger.info(
        f"Strict Multi-Snippet NLI Verification Complete: Score = {avg_score:.3f}"
    )
    return avg_score, results