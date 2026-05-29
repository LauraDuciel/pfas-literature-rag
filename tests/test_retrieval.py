from pfas_lit_rag.retrieval import format_context
from pfas_lit_rag.schemas import SearchResult, TextChunk


def test_search_result_citation_for_single_page() -> None:
    result = SearchResult(
        score=0.91,
        chunk=TextChunk(
            chunk_id="c1",
            document_id="d1",
            title="PFAS in groundwater",
            source_path="paper.pdf",
            page_start=7,
            page_end=7,
            text="PFAS methods text",
        ),
    )

    assert result.citation == "PFAS in groundwater, p. 7"


def test_format_context_truncates_long_chunks() -> None:
    result = SearchResult(
        score=0.8,
        chunk=TextChunk(
            chunk_id="c1",
            document_id="d1",
            title="PFAS methods",
            source_path="paper.pdf",
            page_start=1,
            page_end=1,
            text=" ".join(["word"] * 100),
        ),
    )

    context = format_context([result], max_chars_per_chunk=40)

    assert "[...]" in context
    assert "[1] PFAS methods, p. 1" in context
