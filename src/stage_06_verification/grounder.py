import numpy as np
import requests
from sentence_transformers import CrossEncoder
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.response import FactualClaim
from src.schemas.verification import ClaimVerificationResult, EntailmentStatus

logger = setup_logger("GoogleGrounder")

_nli_model: CrossEncoder | None = None
GOOGLE_KG_ENDPOINT = "https://kgsearch.googleapis.com/v1/entities:search"


def get_nli_model() -> CrossEncoder:
    """Singleton NLI cross-encoder."""
    global _nli_model
    if _nli_model is None:
        logger.info(f"Loading NLI cross-encoder: {settings.NLI_MODEL_NAME}...")
        _nli_model = CrossEncoder(settings.NLI_MODEL_NAME)
    return _nli_model


def fetch_google_knowledge_graph_evidence(query: str) -> str:
    """Queries Google Knowledge Graph Search API for authoritative entity descriptions."""
    api_key = settings.GOOGLE_KG_API_KEY
    if not api_key or api_key == "YOUR_GOOGLE_API_KEY_HERE":
        logger.warning(
            "GOOGLE_KG_API_KEY not configured. Falling back to open knowledge lookup."
        )
        return ""

    # Clean query to extract core entity/concept
    words = [
        w
        for w in query.rstrip("?.").split()
        if w.lower() not in {"what", "is", "the", "a", "an", "who", "define"}
    ]
    entity_query = " ".join(words[:4]) if words else query

    params = {
        "query": entity_query,
        "key": api_key,
        "limit": 2,
        "indent": True,
    }

    try:
        logger.info(
            f"Querying Google Knowledge Graph for entity: '{entity_query}'..."
        )
        resp = requests.get(GOOGLE_KG_ENDPOINT, params=params, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            elements = data.get("itemListElement", [])
            for item in elements:
                result = item.get("result", {})
                # Google returns detailedDescription from authoritative sources
                detailed = result.get("detailedDescription", {})
                article_body = detailed.get("articleBody", "")
                description = result.get("description", "")

                if article_body:
                    return f"{result.get('name', '')} ({description}): {article_body}"
                elif description:
                    return f"{result.get('name', '')}: {description}"
        else:
            logger.warning(f"Google KG API returned status {resp.status_code}")
    except Exception as e:
        logger.warning(f"Google Knowledge Graph lookup failed ({e})")

    return ""


def fetch_backup_open_evidence(query: str) -> str:
    """Backup lookup via Wikipedia REST API if Google has no article body."""
    try:
        words = [
            w
            for w in query.rstrip("?.").split()
            if w.lower()
            not in {"what", "is", "the", "a", "an", "who", "define"}
        ]
        term = words[0] if words else query
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{term}"
        resp = requests.get(url, timeout=4)
        if resp.status_code == 200:
            return resp.json().get("extract", "")
    except Exception:
        pass
    return ""


def verify_claim_against_local_store(
    claim: FactualClaim,
) -> ClaimVerificationResult:
    """Verifies a claim directly against Google Knowledge Graph."""
    # 1. Fetch Google's authoritative entity description
    evidence_text = fetch_google_knowledge_graph_evidence(claim.claim_text)

    # 2. If Google had no detailed entry, use backup open knowledge
    if not evidence_text or len(evidence_text.strip()) < 30:
        evidence_text = fetch_backup_open_evidence(claim.claim_text)

    if not evidence_text or len(evidence_text.strip()) < 30:
        return ClaimVerificationResult(
            claim_id=claim.claim_id,
            claim_text=claim.claim_text,
            status=EntailmentStatus.INSUFFICIENT,
            nli_entailment_score=0.0,
            evidence_snippet=None,
            source_type="GOOGLE_KNOWLEDGE_GRAPH",
        )

    # 3. Run DeBERTa NLI Cross-Encoder: Google Evidence (Premise) vs. Claim (Hypothesis)
    nli = get_nli_model()
    prediction_scores = nli.predict([(evidence_text, claim.claim_text)])

    # Softmax conversion
    exp_scores = np.exp(prediction_scores[0])
    probabilities = exp_scores / np.sum(exp_scores)

    # id2label: 0: contradiction, 1: entailment, 2: neutral
    prob_contradiction = float(probabilities[0])
    prob_entailment = float(probabilities[1])

    if prob_entailment > 0.45:
        status = EntailmentStatus.SUPPORTED
        score = prob_entailment
    elif prob_contradiction > 0.50:
        status = EntailmentStatus.CONTRADICTED
        score = -prob_contradiction
    else:
        status = EntailmentStatus.INSUFFICIENT
        score = 0.0

    logger.info(
        f"Google KG Claim Check: Status={status.value} | Entailment={prob_entailment:.3f}"
    )

    return ClaimVerificationResult(
        claim_id=claim.claim_id,
        claim_text=claim.claim_text,
        status=status,
        nli_entailment_score=round(score, 3),
        evidence_snippet=evidence_text[:250] + "...",
        source_type="GOOGLE_KNOWLEDGE_GRAPH",
    )