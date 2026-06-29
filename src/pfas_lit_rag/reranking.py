from collections import Counter
from functools import lru_cache
from typing import Protocol

from pfas_lit_rag.lexical_search import tokenize
from pfas_lit_rag.schemas import SearchResult

LEXICAL_BACKEND = "lexical"
CROSS_ENCODER_BACKEND = "cross_encoder"
AUTO_BACKEND = "auto"


class CrossEncoderLike(Protocol):
    def predict(self, sentences: list[tuple[str, str]], *, batch_size: int) -> list[float]: ...


def rerank_results(
    query: str,
    results: list[SearchResult],
    *,
    top_k: int,
    rerank_weight: float,
    backend: str = LEXICAL_BACKEND,
    cross_encoder_model: str | None = None,
    cross_encoder_batch_size: int = 8,
) -> list[SearchResult]:
    if not results:
        return []

    selected_backend = backend.lower().strip()
    if selected_backend == AUTO_BACKEND and cross_encoder_model:
        try:
            cross_scores = _cross_encoder_scores(
                query,
                results,
                model_name=cross_encoder_model,
                batch_size=cross_encoder_batch_size,
            )
        except (OSError, RuntimeError):
            cross_scores = None
        if cross_scores is not None:
            return _combine_scores(results, cross_scores, top_k=top_k, rerank_weight=rerank_weight)

    if selected_backend == CROSS_ENCODER_BACKEND:
        if not cross_encoder_model:
            raise ValueError("A cross-encoder model name is required for cross_encoder reranking")
        cross_scores = _cross_encoder_scores(
            query,
            results,
            model_name=cross_encoder_model,
            batch_size=cross_encoder_batch_size,
        )
        return _combine_scores(results, cross_scores, top_k=top_k, rerank_weight=rerank_weight)

    if selected_backend in {LEXICAL_BACKEND, AUTO_BACKEND}:
        lexical_scores = _lexical_scores(query, results)
        return _combine_scores(results, lexical_scores, top_k=top_k, rerank_weight=rerank_weight)

    raise ValueError(f"Unsupported rerank backend: {backend}")


def _lexical_scores(query: str, results: list[SearchResult]) -> dict[str, float]:
    query_terms = set(tokenize(query))
    if not query_terms:
        return {result.chunk.chunk_id: 0.0 for result in results}
    return {
        result.chunk.chunk_id: _query_overlap_score(query_terms, result.chunk.text)
        for result in results
    }


def _cross_encoder_scores(
    query: str,
    results: list[SearchResult],
    *,
    model_name: str,
    batch_size: int,
) -> dict[str, float]:
    model = _load_cross_encoder(model_name)
    pairs = [(query, result.chunk.text) for result in results]
    raw_scores = model.predict(pairs, batch_size=batch_size)
    return {
        result.chunk.chunk_id: float(score)
        for result, score in zip(results, raw_scores, strict=False)
    }


def _combine_scores(
    results: list[SearchResult],
    rerank_scores: dict[str, float],
    *,
    top_k: int,
    rerank_weight: float,
) -> list[SearchResult]:
    base_scores = _normalise_scores({result.chunk.chunk_id: result.score for result in results})
    normalised_rerank_scores = _normalise_scores(rerank_scores)
    reranked: list[SearchResult] = []
    for result in results:
        chunk_id = result.chunk.chunk_id
        combined = (1 - rerank_weight) * base_scores[chunk_id]
        combined += rerank_weight * normalised_rerank_scores.get(chunk_id, 0.0)
        reranked.append(result.model_copy(update={"score": float(combined)}))

    return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str) -> CrossEncoderLike:
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Cross-encoder reranking requires sentence-transformers to be installed "
            "in the local environment. Use a CPU-only PyTorch install on small machines."
        ) from exc
    return CrossEncoder(model_name)


def _query_overlap_score(query_terms: set[str], text: str) -> float:
    text_counts = Counter(tokenize(text))
    if not text_counts:
        return 0.0
    matched = sum(1 for term in query_terms if text_counts.get(term, 0) > 0)
    return matched / len(query_terms)


def _normalise_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_score = min(scores.values())
    max_score = max(scores.values())
    if max_score == min_score:
        return {key: 1.0 for key in scores}
    return {key: (value - min_score) / (max_score - min_score) for key, value in scores.items()}
