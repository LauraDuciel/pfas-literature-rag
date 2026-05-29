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
