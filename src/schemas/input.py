from enum import Enum
from pydantic import BaseModel, Field


class QueryIntent(str, Enum):
    FACTUAL = "FACTUAL"
    REASONING = "REASONING"
    OPINION = "OPINION"
    UNKNOWN = "UNKNOWN"


class ExtractedContext(BaseModel):
    named_entities: list[dict[str, str]] = Field(default_factory=list)
    temporal_constraints: list[str] = Field(default_factory=list)
    domain_keywords: list[str] = Field(default_factory=list)


class DecomposedQuestion(BaseModel):
    id: int
    sub_question: str
    target_entity: str | None = None


class ProcessedInput(BaseModel):
    raw_query: str
    intent: QueryIntent
    context: ExtractedContext
    ambiguity_score: float = Field(
        ..., ge=0.0, le=1.0, description="0 is clear, 1 is highly ambiguous"
    )
    unclear_terms: list[str] = Field(default_factory=list)
    sub_questions: list[DecomposedQuestion] = Field(default_factory=list)