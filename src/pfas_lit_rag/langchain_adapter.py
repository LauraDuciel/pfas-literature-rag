from typing import Any

from pfas_lit_rag.config import Settings, get_settings
from pfas_lit_rag.retrieval import search_index


def _load_langchain_types():
    try:
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise RuntimeError(
            'LangChain integration requires `langchain-core`. '
            'Install it with `uv sync --extra langchain`.'
        ) from exc
    return BaseRetriever, Document


def make_langchain_retriever(settings: Settings | None = None, *, top_k: int | None = None):
    BaseRetriever, Document = _load_langchain_types()
    active_settings = settings or get_settings()

    class PFASLiteratureRetriever(BaseRetriever):
        settings: Settings
        top_k: int | None = None

        def _get_relevant_documents(self, query: str, *, run_manager: Any = None):
            results = search_index(query, settings=self.settings, top_k=self.top_k)
            return [
                Document(
                    page_content=result.chunk.text,
                    metadata={
                        'chunk_id': result.chunk.chunk_id,
                        'document_id': result.chunk.document_id,
                        'title': result.chunk.title,
                        'source_path': result.chunk.source_path,
                        'page_start': result.chunk.page_start,
                        'page_end': result.chunk.page_end,
                        'score': result.score,
                        'citation': result.citation,
                    },
                )
                for result in results
            ]

    return PFASLiteratureRetriever(settings=active_settings, top_k=top_k)
