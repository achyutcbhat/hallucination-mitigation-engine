from enum import Enum
from pydantic import BaseModel, Field


class EntailmentStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT = "INSUFFICIENT"


class RoutingDecision(str, Enum):
    VERIFIED = "VERIFIED"
    UNCERTAIN = "UNCERTAIN"
    FULLY_HALLUCINATED = "FULLY_HALLUCINATED"


class ClaimVerificationResult(BaseModel):
    claim_id: int
    claim_text: str
    status: EntailmentStatus
    nli_entailment_score: float = Field(..., ge=-1.0, le=1.0)
    evidence_snippet: str | None = None
    source_type: str = "LOCAL_QDRANT"  # "LOCAL_QDRANT" or "WEB_SEARCH"


class FitnessScoreResult(BaseModel):
    fitness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Semantic alignment between query and master answer",
    )
    subquestion_coverage: float = Field(..., ge=0.0, le=1.0)


class FinalPipelineOutput(BaseModel):
    original_query: str
    final_answer: str
    routing_decision: RoutingDecision
    overall_confidence_score: float
    hitl_triggered: bool = False
    web_fallback_used: bool = False
    claims_verified: list[ClaimVerificationResult] = Field(default_factory=list)
    execution_time_seconds: float