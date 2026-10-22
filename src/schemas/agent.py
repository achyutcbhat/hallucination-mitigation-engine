from pydantic import BaseModel, Field


class AgentCandidateResponse(BaseModel):
    candidate_id: int = Field(..., description="ID from 1 to 6")
    agent_model: str
    iteration: int = Field(..., description="Run 1 or Run 2")
    temperature: float
    response_text: str
    latency_seconds: float


class CandidatePool(BaseModel):
    total_candidates: int = 6
    candidates: list[AgentCandidateResponse]