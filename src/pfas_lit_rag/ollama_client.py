import httpx

from pfas_lit_rag.config import Settings
from pfas_lit_rag.retrieval import format_context
from pfas_lit_rag.schemas import SearchResult


class OllamaClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(
            base_url=settings.ollama_base_url,
            timeout=settings.request_timeout_seconds,
        )

    def health(self) -> dict:
        response = self.client.get("/api/tags")
        response.raise_for_status()
        return response.json()

    def answer(self, question: str, results: list[SearchResult]) -> str:
        context = format_context(results, self.settings.context_chars_per_chunk)
        prompt = _build_prompt(question=question, context=context)
        response = self.client.post(
            "/api/generate",
            json={
                "model": self.settings.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 4096,
                    "num_predict": self.settings.ollama_num_predict,
                },
            },
        )
        response.raise_for_status()
        payload = response.json()
        return str(payload.get("response", "")).strip()


def _build_prompt(question: str, context: str) -> str:
    return f"""You support careful scientific literature review.
Use only the retrieved passages below. If the passages do not support an answer,
say that the local corpus does not contain enough evidence.

Write a concise answer in French or English matching the user's question.
Include citations in square brackets such as [1] or [2] for every factual claim.
Do not invent citations, titles, methods, concentrations, or conclusions.

Retrieved passages:
{context}

Question: {question}

Answer:"""
