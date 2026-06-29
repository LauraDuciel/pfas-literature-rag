import re
import time
from enum import StrEnum

from pydantic import BaseModel

from pfas_lit_rag.config import Settings
from pfas_lit_rag.ollama_client import OllamaClient
from pfas_lit_rag.retrieval import (
    BM25_STRATEGY,
    HYBRID_STRATEGY,
    VECTOR_STRATEGY,
    search_index_with_strategy,
)
from pfas_lit_rag.schemas import AnswerResponse, SearchResult


class RetrievalMode(StrEnum):
    LLM_ONLY = "llm_only"
    BM25 = "bm25"
    VECTOR = "vector"
    HYBRID = "hybrid"
    HYBRID_RETRY = "hybrid_retry"


class RetrievalDecision(BaseModel):
    mode: RetrievalMode
    reason: str
    top_k: int
    searches_planned: int = 0


class AdaptiveAnswer(BaseModel):
    response: AnswerResponse
    decision: RetrievalDecision
    searches_run: int
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


GENERAL_PATTERNS = (
    "what are pfas",
    "what is pfas",
    "define pfas",
    "what does pfas mean",
)
EXACT_TERM_PATTERN = re.compile(
    r"(lc-ms/ms|gc-ms|hrms|nmr|qtof|tof-ms|orbitrap|fticr-ms|spe|spme|µspe|pfos|pfoa|genx|eof)",
    re.IGNORECASE,
)
MULTI_DOCUMENT_TERMS = {"compare", "across", "multiple", "several", "trend", "overview", "review"}
OUT_OF_CORPUS_HINTS = {"clinical trial", "patent", "market size", "stock price", "weather"}


def choose_retrieval_strategy(question: str, *, default_top_k: int = 4) -> RetrievalDecision:
    text = question.lower().strip()
    words = set(re.findall(r"[a-z0-9/µ-]+", text))

    if any(pattern in text for pattern in GENERAL_PATTERNS):
        return RetrievalDecision(
            mode=RetrievalMode.LLM_ONLY,
            reason="general background question",
            top_k=0,
            searches_planned=0,
        )
    if any(hint in text for hint in OUT_OF_CORPUS_HINTS):
        return RetrievalDecision(
            mode=RetrievalMode.HYBRID,
            reason="likely needs corpus check before answering",
            top_k=default_top_k,
            searches_planned=1,
        )
    if words & MULTI_DOCUMENT_TERMS:
        return RetrievalDecision(
            mode=RetrievalMode.HYBRID_RETRY,
            reason="multi-document wording",
            top_k=max(default_top_k, 6),
            searches_planned=2,
        )
    if EXACT_TERM_PATTERN.search(text):
        return RetrievalDecision(
            mode=RetrievalMode.HYBRID,
            reason="scientific term or method detected",
            top_k=default_top_k,
            searches_planned=1,
        )
    if len(words) <= 5:
        return RetrievalDecision(
            mode=RetrievalMode.BM25,
            reason="short keyword-style question",
            top_k=default_top_k,
            searches_planned=1,
        )
    return RetrievalDecision(
        mode=RetrievalMode.HYBRID,
        reason="default scientific retrieval policy",
        top_k=default_top_k,
        searches_planned=1,
    )


def retrieve_adaptive(
    question: str,
    settings: Settings,
    top_k: int | None = None,
) -> tuple[RetrievalDecision, list[SearchResult], int, float]:
    decision = choose_retrieval_strategy(question, default_top_k=top_k or settings.retrieval_k)
    if decision.mode == RetrievalMode.LLM_ONLY:
        return decision, [], 0, 0.0
    retrieval_started = time.perf_counter()
    retrieved = _retrieve_for_decision(question, settings=settings, decision=decision)
    return decision, retrieved, decision.searches_planned, time.perf_counter() - retrieval_started


def answer_adaptive(question: str, settings: Settings, top_k: int | None = None) -> AdaptiveAnswer:
    started = time.perf_counter()
    decision, retrieved, searches_run, retrieval_seconds = retrieve_adaptive(
        question,
        settings=settings,
        top_k=top_k,
    )

    generation_started = time.perf_counter()
    if decision.mode == RetrievalMode.LLM_ONLY:
        answer = OllamaClient(settings).answer_without_context(question)
        citations: list[str] = []
    elif not retrieved:
        answer = "The local corpus does not contain enough evidence to answer this question."
        citations = []
    else:
        answer = OllamaClient(settings).answer(question, retrieved)
        citations = [result.citation for result in retrieved]
    generation_seconds = time.perf_counter() - generation_started

    response = AnswerResponse(
        question=question,
        answer=answer,
        citations=citations,
        retrieved=retrieved,
    )
    return AdaptiveAnswer(
        response=response,
        decision=decision,
        searches_run=searches_run,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
        total_seconds=time.perf_counter() - started,
    )


def _retrieve_for_decision(
    question: str,
    *,
    settings: Settings,
    decision: RetrievalDecision,
) -> list[SearchResult]:
    if decision.mode == RetrievalMode.BM25:
        return search_index_with_strategy(
            question,
            settings=settings,
            top_k=decision.top_k,
            strategy=BM25_STRATEGY,
        )
    if decision.mode == RetrievalMode.VECTOR:
        return search_index_with_strategy(
            question,
            settings=settings,
            top_k=decision.top_k,
            strategy=VECTOR_STRATEGY,
        )

    results = search_index_with_strategy(
        question,
        settings=settings,
        top_k=decision.top_k,
        strategy=HYBRID_STRATEGY,
    )
    if decision.mode != RetrievalMode.HYBRID_RETRY or _has_sufficient_evidence(results):
        return results

    retry_query = _simplify_query(question)
    retry_results = search_index_with_strategy(
        retry_query,
        settings=settings,
        top_k=decision.top_k,
        strategy=HYBRID_STRATEGY,
    )
    return _merge_results(results, retry_results, top_k=decision.top_k)


def _has_sufficient_evidence(results: list[SearchResult], *, min_score: float = 0.35) -> bool:
    return bool(results) and max(result.score for result in results) >= min_score


def _simplify_query(question: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9/µ-]+", question)
    useful = [token for token in tokens if len(token) > 3 or EXACT_TERM_PATTERN.search(token)]
    return " ".join(useful[:12]) or question


def _merge_results(
    first: list[SearchResult],
    second: list[SearchResult],
    *,
    top_k: int,
) -> list[SearchResult]:
    by_id: dict[str, SearchResult] = {}
    for result in first + second:
        current = by_id.get(result.chunk.chunk_id)
        if current is None or result.score > current.score:
            by_id[result.chunk.chunk_id] = result
    return sorted(by_id.values(), key=lambda item: item.score, reverse=True)[:top_k]
