from pydantic import BaseModel, Field


class FactualClaim(BaseModel):
    claim_id: int
    claim_text: str


class SynthesizedResponse(BaseModel):
    master_response: str
    consensus_candidate_ids: list[int]
    extracted_claims: list[FactualClaim] = Field(
        default_factory=list,
        description="Atomic claims extracted from the master response for verification",
    )
    synthesis_latency: float