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


def test_cross_encoder_reranking_uses_model_scores(monkeypatch) -> None:
    class DummyCrossEncoder:
        def predict(self, sentences, *, batch_size):
            assert batch_size == 4
            return [0.1, 0.9]

    monkeypatch.setattr(
        'pfas_lit_rag.reranking._load_cross_encoder',
        lambda model_name: DummyCrossEncoder(),
    )
    results = [
        _result('semantic', 'general PFAS discussion', 0.9),
        _result('method', 'PFAS LC-MS/MS extraction method', 0.2),
    ]

    reranked = rerank_results(
        'PFAS LC-MS/MS extraction',
        results,
        top_k=2,
        rerank_weight=0.8,
        backend='cross_encoder',
        cross_encoder_model='local-test-model',
        cross_encoder_batch_size=4,
    )

    assert reranked[0].chunk.chunk_id == 'method'


def test_unknown_rerank_backend_raises() -> None:
    results = [_result('a', 'PFAS', 0.9)]

    try:
        rerank_results('PFAS', results, top_k=1, rerank_weight=0.5, backend='unknown')
    except ValueError as exc:
        assert 'Unsupported rerank backend' in str(exc)
    else:
        raise AssertionError('unknown backend should fail')
