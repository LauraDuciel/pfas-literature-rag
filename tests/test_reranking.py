from pfas_lit_rag.reranking import rerank_results
from pfas_lit_rag.schemas import SearchResult, TextChunk


def _result(chunk_id: str, text: str, score: float) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=TextChunk(
            chunk_id=chunk_id,
            document_id='doc',
            title='PFAS methods',
            source_path='paper.pdf',
            page_start=1,
            page_end=1,
            text=text,
        ),
    )


def test_rerank_results_promotes_query_term_overlap() -> None:
    results = [
        _result('semantic', 'general PFAS discussion', 0.9),
        _result('method', 'PFAS LC-MS/MS SPE-WAX extraction method', 0.2),
    ]

    reranked = rerank_results(
        'PFAS LC-MS/MS SPE-WAX extraction',
        results,
        top_k=2,
        rerank_weight=0.8,
    )

    assert reranked[0].chunk.chunk_id == 'method'


def test_rerank_results_respects_top_k() -> None:
    results = [
        _result('a', 'PFAS', 0.9),
        _result('b', 'PFAS extraction', 0.8),
    ]

    assert len(rerank_results('PFAS extraction', results, top_k=1, rerank_weight=0.5)) == 1
