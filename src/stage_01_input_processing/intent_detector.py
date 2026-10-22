import spacy
from src.schemas.input import QueryIntent
from config.settings import settings

nlp = spacy.load(settings.SPACY_MODEL)

FACTUAL_CUES = {
    "who",
    "when",
    "where",
    "what",
    "which",
    "birth",
    "founded",
    "capital",
    "date",
    "invented",
}
REASONING_CUES = {
    "why",
    "how",
    "explain",
    "compare",
    "difference",
    "analyze",
    "cause",
    "effect",
    "steps",
}
OPINION_CUES = {
    "should",
    "feel",
    "opinion",
    "recommend",
    "favorite",
    "prefer",
    "best",
    "worth",
}


def detect_intent(text: str) -> QueryIntent:
    doc = nlp(text.lower())
    tokens = {token.lemma_ for token in doc}

    # Count overlaps
    factual_matches = len(tokens.intersection(FACTUAL_CUES))
    reasoning_matches = len(tokens.intersection(REASONING_CUES))
    opinion_matches = len(tokens.intersection(OPINION_CUES))

    # Decision tree
    if opinion_matches > 0 and opinion_matches >= reasoning_matches:
        return QueryIntent.OPINION
    if reasoning_matches > factual_matches:
        return QueryIntent.REASONING
    if factual_matches > 0:
        return QueryIntent.FACTUAL

    return QueryIntent.FACTUAL  # Default assumption for information seeking