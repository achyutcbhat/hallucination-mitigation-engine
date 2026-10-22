import re
import time
import ollama
from config.settings import settings
from config.logging_config import setup_logger
from src.utils.memory_manager import flush_cuda_memory

logger = setup_logger("AgentPool")


class AgentRunner:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)

    def generate_candidate(
        self,
        query: str,
        sub_questions: list[str],
        iteration: int,
        temperature: float = 0.7,
    ) -> tuple[str, float]:
        logger.info(
            f"Invoking {self.model_name} (Run #{iteration}, Temp={temperature})..."
        )

        sub_q_context = ""
        if sub_questions and len(sub_questions) > 1:
            sub_q_context = (
                "\nConsider these decomposed aspects:\n- "
                + "\n- ".join(sub_questions)
            )

        system_instruction = (
            "You are an expert factual assistant. Provide a concise, highly accurate, "
            "and direct answer to the user query. Avoid unverified speculations."
        )

        user_prompt = f"Question: {query}{sub_q_context}\n\nAnswer concisely and factually:"

        start_time = time.time()
        try:
            response = self.client.generate(
                model=self.model_name,
                system=system_instruction,
                prompt=user_prompt,
                options={
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 300,
                },
            )
            latency = round(time.time() - start_time, 2)
            raw_text = response["response"].strip()

            # Clean out DeepSeek-R1 <think>...</think> monologue
            cleaned_text = re.sub(
                r"<think>.*?</think>", "", raw_text, flags=re.DOTALL
            ).strip()
            if not cleaned_text:
                cleaned_text = raw_text

            return cleaned_text, latency
        except Exception as e:
            logger.error(
                f"Error generating from model {self.model_name}: {str(e)}"
            )
            latency = round(time.time() - start_time, 2)
            return (
                f"[Error generating answer from {self.model_name}]",
                latency,
            )
        finally:
            flush_cuda_memory()