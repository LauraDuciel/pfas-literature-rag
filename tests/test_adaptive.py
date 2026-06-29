from pfas_lit_rag.adaptive import (
    RetrievalMode,
    _merge_results,
    choose_retrieval_strategy,
)
from pfas_lit_rag.schemas import SearchResult, TextChunk


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=TextChunk(
            chunk_id=chunk_id,
            document_id="doc",
            title="PFAS methods",
            source_path="paper.pdf",
            page_start=1,
            page_end=1,
            text="PFAS LC-MS/MS extraction",
        ),
    )


def test_choose_retrieval_strategy_uses_llm_only_for_general_question() -> None:
    decision = choose_retrieval_strategy("What are PFAS?")

    assert decision.mode == RetrievalMode.LLM_ONLY
    assert decision.searches_planned == 0


def test_choose_retrieval_strategy_uses_hybrid_for_exact_method() -> None:
    decision = choose_retrieval_strategy("How is LC-MS/MS used for PFAS detection?")

    assert decision.mode == RetrievalMode.HYBRID
    assert decision.searches_planned == 1


def test_choose_retrieval_strategy_uses_retry_for_multi_document_question() -> None:
    decision = choose_retrieval_strategy("Compare extraction methods across PFAS studies")

    assert decision.mode == RetrievalMode.HYBRID_RETRY
    assert decision.top_k >= 6


def test_merge_results_keeps_highest_score_for_duplicate_chunks() -> None:
    merged = _merge_results(
        [_result("a", 0.2), _result("b", 0.8)],
        [_result("a", 0.9), _result("c", 0.5)],
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in merged] == ["a", "b"]
    assert merged[0].score == 0.9
