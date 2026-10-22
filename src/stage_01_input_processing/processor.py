from config.logging_config import setup_logger
from src.schemas.input import ProcessedInput
from src.stage_01_input_processing.context_extractor import extract_context
from src.stage_01_input_processing.ambiguity_detector import detect_ambiguity
from src.stage_01_input_processing.intent_detector import detect_intent
from src.stage_01_input_processing.query_decomposer import decompose_query

logger = setup_logger("Stage01-Processor")


def process_user_input(raw_query: str) -> ProcessedInput:
    logger.info(f"Processing input query: '{raw_query}'")

    # 1. Intent Detection
    intent = detect_intent(raw_query)

    # 2. Entity & Context Extraction
    context = extract_context(raw_query)

    # 3. Ambiguity & Missing Context Detection
    ambiguity_score, unclear_terms = detect_ambiguity(raw_query)

    # 4. Multi-hop Question Decomposition
    sub_questions = decompose_query(raw_query)

    processed = ProcessedInput(
        raw_query=raw_query,
        intent=intent,
        context=context,
        ambiguity_score=ambiguity_score,
        unclear_terms=unclear_terms,
        sub_questions=sub_questions,
    )

    logger.info(
        f"Input processed successfully: Intent={intent.value}, "
        f"Ambiguity={ambiguity_score}, SubQuestions={len(sub_questions)}"
    )
    return processed