from pfas_lit_rag.lexical_search import BM25Index, tokenize
from pfas_lit_rag.retrieval import _fuse_results, format_context
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


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=TextChunk(
            chunk_id=chunk_id,
            document_id="doc",
            title="PFAS methods",
            source_path="paper.pdf",
            page_start=1,
            page_end=1,
            text=text,
        ),
    )


def test_tokenize_preserves_technical_terms() -> None:
    assert "lc-ms/ms" in tokenize("LC-MS/MS analysis of PFAS")
    assert "spe-wax" in tokenize("SPE-WAX extraction")


def test_bm25_finds_exact_method_terms() -> None:
    chunks = [
        _result("c1", "general PFAS background", 0.0).chunk,
        _result("c2", "LC-MS/MS and SPE-WAX extraction for PFAS", 0.0).chunk,
    ]

    results = BM25Index(chunks).search("SPE-WAX", top_k=1)

    assert results[0].chunk.chunk_id == "c2"


def test_fuse_results_combines_vector_and_lexical_matches() -> None:
    vector_results = [_result("c1", "semantic match", 0.9), _result("c2", "exact method", 0.1)]
    lexical_results = [_result("c2", "exact method", 4.0)]

    results = _fuse_results(
        vector_results=vector_results,
        lexical_results=lexical_results,
        top_k=2,
        vector_weight=0.4,
        lexical_weight=0.6,
    )

    assert results[0].chunk.chunk_id == "c2"
