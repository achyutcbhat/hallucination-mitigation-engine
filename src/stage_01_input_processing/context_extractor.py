import spacy
from config.settings import settings
from config.logging_config import setup_logger
from src.schemas.input import ExtractedContext

logger = setup_logger("ContextExtractor")

try:
    nlp = spacy.load(settings.SPACY_MODEL)
except Exception as e:
    logger.warning(
        f"Failed to load spacy model {settings.SPACY_MODEL}, attempting download..."
    )
    import subprocess

    subprocess.run(["python", "-m", "spacy", "download", settings.SPACY_MODEL])
    nlp = spacy.load(settings.SPACY_MODEL)


def extract_context(text: str) -> ExtractedContext:
    doc = nlp(text)

    # 1. Named Entities
    entities = [
        {"text": ent.text, "label": ent.label_}
        for ent in doc.ents
        if ent.label_ not in ["CARDINAL"]  # Ignore bare numbers
    ]

    # 2. Temporal Constraints (Dates, Times, and temporal adverbs)
    temporal_labels = {"DATE", "TIME"}
    temporal_words = {
        "recent",
        "recently",
        "latest",
        "current",
        "currently",
        "before",
        "after",
        "since",
    }
    temporal = [ent.text for ent in doc.ents if ent.label_ in temporal_labels]
    for token in doc:
        if token.lower_ in temporal_words and token.text not in temporal:
            temporal.append(token.text)

    # 3. Domain Keywords (Content nouns and proper nouns)
    keywords = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and not token.is_stop and len(token) > 2
    ]
    keywords = list(dict.fromkeys(keywords))  # Deduplicate preserving order

    return ExtractedContext(
        named_entities=entities,
        temporal_constraints=temporal,
        domain_keywords=keywords,
    )