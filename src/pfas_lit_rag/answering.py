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
    citations = [result.citation for result in results]
    answer = _ensure_sources_section(answer, citations)
    return AnswerResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=results,
    )


def _ensure_sources_section(answer: str, citations: list[str]) -> str:
    if not citations or "Sources:" in answer:
        return answer
    sources = "\n".join(
        f"[{index}] {citation}" for index, citation in enumerate(citations, start=1)
    )
    return f"{answer.strip()}\n\nSources:\n{sources}"
