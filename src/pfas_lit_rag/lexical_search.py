import math
import re
from collections import Counter

from pfas_lit_rag.schemas import SearchResult, TextChunk

TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-/][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    def __init__(self, chunks: list[TextChunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(chunk.text) for chunk in chunks]
        self.term_counts = [Counter(document) for document in self.documents]
        self.document_frequency = _document_frequency(self.documents)
        self.average_length = _average_length(self.documents)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms or not self.chunks:
            return []

        scored: list[tuple[float, int]] = []
        for index, term_count in enumerate(self.term_counts):
            score = self._score_document(query_terms, term_count, len(self.documents[index]))
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(chunk=self.chunks[index], score=float(score))
            for score, index in scored[:top_k]
        ]

    def _score_document(
        self,
        query_terms: list[str],
        term_count: Counter[str],
        document_length: int,
    ) -> float:
        score = 0.0
        corpus_size = len(self.documents)
        for term in query_terms:
            term_frequency = term_count.get(term, 0)
            if term_frequency == 0:
                continue
            document_frequency = self.document_frequency.get(term, 0)
            idf = math.log(
                1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(self.average_length, 1.0)
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score


def _document_frequency(documents: list[list[str]]) -> Counter[str]:
    frequency: Counter[str] = Counter()
    for document in documents:
        frequency.update(set(document))
    return frequency


def _average_length(documents: list[list[str]]) -> float:
    if not documents:
        return 0.0
    return sum(len(document) for document in documents) / len(documents)
