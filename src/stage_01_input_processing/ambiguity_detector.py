import spacy
from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("AmbiguityDetector")
nlp = spacy.load(settings.SPACY_MODEL)

VAGUE_PRONOUNS = {"it", "this", "that", "they", "them", "these", "those"}
SUBJECTIVE_TERMS = {
    "best",
    "worst",
    "good",
    "bad",
    "cheap",
    "expensive",
    "fast",
    "better",
    "easier",
}
VAGUE_QUANTIFIERS = {"some", "many", "several", "a lot", "few", "things"}


def detect_ambiguity(text: str) -> tuple[float, list[str]]:
    doc = nlp(text)
    unclear_terms = []
    penalty_score = 0.0

    # 1. Definitional queries (e.g. "what is science", "define gravity") are NOT ambiguous
    clean_lower = text.lower().strip().rstrip("?.")
    if (
        clean_lower.startswith(
            ("what is ", "what are ", "define ", "explain ")
        )
        and len(doc) >= 3
    ):
        target = clean_lower.split(maxsplit=2)[-1]
        # If the target is NOT an ambiguous pronoun, it is a clear definition query!
        if target not in VAGUE_PRONOUNS:
            return 0.0, []

    # 2. Check for ambiguous pronouns without an antecedent
    for token in doc:
        if token.lower_ in VAGUE_PRONOUNS and token.dep_ in {
            "nsubj",
            "dobj",
            "pobj",
        }:
            unclear_terms.append(f"Ambiguous reference: '{token.text}'")
            penalty_score += 0.35

    # 3. Check for subjective / vague terms
    for token in doc:
        if token.lower_ in SUBJECTIVE_TERMS:
            unclear_terms.append(f"Subjective quantifier: '{token.text}'")
            penalty_score += 0.15
        elif token.lower_ in VAGUE_QUANTIFIERS:
            unclear_terms.append(f"Vague quantifier: '{token.text}'")
            penalty_score += 0.10

    # 4. Check for meaningless ultra-short inputs (< 3 words)
    word_count = len([t for t in doc if not t.is_punct])
    if word_count < 3:
        penalty_score += 0.40
        unclear_terms.append("Query is extremely terse (< 3 words)")

    final_score = min(round(penalty_score, 2), 1.0)
    return final_score, unclear_terms