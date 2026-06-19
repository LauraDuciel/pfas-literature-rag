from collections import Counter

from pfas_lit_rag.lexical_search import tokenize
from pfas_lit_rag.schemas import SearchResult


def rerank_results(
    query: str,
    results: list[SearchResult],
    *,
    top_k: int,
    rerank_weight: float,
) -> list[SearchResult]:
    if not results:
        return []

    query_terms = set(tokenize(query))
    if not query_terms:
        return results[:top_k]

    base_scores = _normalise_scores({result.chunk.chunk_id: result.score for result in results})
    reranked: list[SearchResult] = []
    for result in results:
        lexical_score = _query_overlap_score(query_terms, result.chunk.text)
        combined = (1 - rerank_weight) * base_scores[result.chunk.chunk_id]
        combined += rerank_weight * lexical_score
        reranked.append(result.model_copy(update={'score': float(combined)}))

    return sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]


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
