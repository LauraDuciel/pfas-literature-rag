from pfas_lit_rag.config import Settings
from pfas_lit_rag.embeddings import get_embedding_model
from pfas_lit_rag.lexical_search import BM25Index
from pfas_lit_rag.reranking import rerank_results
from pfas_lit_rag.schemas import SearchResult
from pfas_lit_rag.vector_store import VectorStore

BM25_STRATEGY = "bm25"
VECTOR_STRATEGY = "vector"
HYBRID_STRATEGY = "hybrid"


def search_index(query: str, settings: Settings, top_k: int | None = None) -> list[SearchResult]:
    return search_index_with_strategy(
        query,
        settings=settings,
        top_k=top_k,
        strategy=HYBRID_STRATEGY,
        rerank=settings.rerank_enabled,
    )


def search_index_with_strategy(
    query: str,
    *,
    settings: Settings,
    top_k: int | None = None,
    strategy: str = HYBRID_STRATEGY,
    rerank: bool | None = None,
) -> list[SearchResult]:
    k = top_k or settings.retrieval_k
    candidate_k = max(k, settings.lexical_candidate_k, settings.cross_encoder_candidate_k)
    store = VectorStore(settings.resolved_index_dir)
    _, chunks = store.load()

    selected_strategy = strategy.lower().strip()
    if selected_strategy == BM25_STRATEGY:
        candidates = BM25Index(chunks).search(query, candidate_k)
    elif selected_strategy == VECTOR_STRATEGY:
        model = get_embedding_model(settings.embedding_model)
        query_embedding = model.encode([query])
        candidates = store.search(query_embedding, candidate_k)
    elif selected_strategy == HYBRID_STRATEGY:
        model = get_embedding_model(settings.embedding_model)
        query_embedding = model.encode([query])
        vector_results = store.search(query_embedding, candidate_k)
        lexical_results = BM25Index(chunks).search(query, candidate_k)
        candidates = _fuse_results(
            vector_results=vector_results,
            lexical_results=lexical_results,
            top_k=candidate_k,
            vector_weight=settings.vector_weight,
            lexical_weight=settings.lexical_weight,
        )
    else:
        raise ValueError(f"Unsupported retrieval strategy: {strategy}")

    should_rerank = settings.rerank_enabled if rerank is None else rerank
    if should_rerank and selected_strategy in {BM25_STRATEGY, VECTOR_STRATEGY, HYBRID_STRATEGY}:
        return rerank_results(
            query,
            candidates,
            top_k=k,
            rerank_weight=settings.rerank_weight,
            backend=settings.rerank_backend,
            cross_encoder_model=settings.cross_encoder_model,
            cross_encoder_batch_size=settings.cross_encoder_batch_size,
        )
    return candidates[:k]


def _fuse_results(
    *,
    vector_results: list[SearchResult],
    lexical_results: list[SearchResult],
    top_k: int,
    vector_weight: float,
    lexical_weight: float,
) -> list[SearchResult]:
    chunks_by_id = {
        result.chunk.chunk_id: result.chunk for result in vector_results + lexical_results
    }
    scores: dict[str, float] = {chunk_id: 0.0 for chunk_id in chunks_by_id}

    for chunk_id, score in _normalised_scores(vector_results).items():
        scores[chunk_id] += vector_weight * score
    for chunk_id, score in _normalised_scores(lexical_results).items():
        scores[chunk_id] += lexical_weight * score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        SearchResult(chunk=chunks_by_id[chunk_id], score=float(score))
        for chunk_id, score in ranked[:top_k]
    ]


def _normalised_scores(results: list[SearchResult]) -> dict[str, float]:
    if not results:
        return {}
    min_score = min(result.score for result in results)
    max_score = max(result.score for result in results)
    if max_score == min_score:
        return {result.chunk.chunk_id: 1.0 for result in results}
    return {
        result.chunk.chunk_id: (result.score - min_score) / (max_score - min_score)
        for result in results
    }


def format_context(results: list[SearchResult], max_chars_per_chunk: int | None = None) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        text = _truncate_text(result.chunk.text, max_chars_per_chunk)
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {result.citation}",
                    f"Score: {result.score:.3f}",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def _truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{truncated} [...]"
