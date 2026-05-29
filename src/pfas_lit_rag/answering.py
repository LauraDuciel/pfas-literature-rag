from pfas_lit_rag.config import Settings
from pfas_lit_rag.ollama_client import OllamaClient
from pfas_lit_rag.retrieval import search_index
from pfas_lit_rag.schemas import AnswerResponse


def answer_question(question: str, settings: Settings, top_k: int | None = None) -> AnswerResponse:
    results = search_index(question, settings=settings, top_k=top_k)
    if not results:
        return AnswerResponse(
            question=question,
            answer="No relevant passages were found in the local index.",
            citations=[],
            retrieved=[],
        )
    answer = OllamaClient(settings).answer(question, results)
    return AnswerResponse(
        question=question,
        answer=answer,
        citations=[result.citation for result in results],
        retrieved=results,
    )
