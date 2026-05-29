from pfas_lit_rag.config import Settings
from pfas_lit_rag.embeddings import get_embedding_model
from pfas_lit_rag.schemas import SearchResult
from pfas_lit_rag.vector_store import VectorStore


def search_index(query: str, settings: Settings, top_k: int | None = None) -> list[SearchResult]:
    model = get_embedding_model(settings.embedding_model)
    query_embedding = model.encode([query])
    store = VectorStore(settings.resolved_index_dir)
    return store.search(query_embedding, top_k or settings.retrieval_k)


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
