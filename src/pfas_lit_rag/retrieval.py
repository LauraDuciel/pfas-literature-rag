from pfas_lit_rag.config import Settings
from pfas_lit_rag.embeddings import get_embedding_model
from pfas_lit_rag.schemas import SearchResult
from pfas_lit_rag.vector_store import VectorStore


def search_index(query: str, settings: Settings, top_k: int | None = None) -> list[SearchResult]:
    model = get_embedding_model(settings.embedding_model)
    query_embedding = model.encode([query])
    store = VectorStore(settings.resolved_index_dir)
    return store.search(query_embedding, top_k or settings.retrieval_k)


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for index, result in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{index}] {result.citation}",
                    f"Score: {result.score:.3f}",
                    result.chunk.text,
                ]
            )
        )
    return "\n\n".join(blocks)
