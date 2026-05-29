from pydantic import BaseModel

from pfas_lit_rag.chunking import chunk_pages
from pfas_lit_rag.config import Settings
from pfas_lit_rag.embeddings import get_embedding_model
from pfas_lit_rag.pdfs import find_pdfs, read_pdf_pages
from pfas_lit_rag.schemas import TextChunk
from pfas_lit_rag.vector_store import VectorStore


class IngestionResult(BaseModel):
    total_chunks: int
    new_chunks: int
    existing_chunks: int


def build_index(settings: Settings) -> IngestionResult:
    pdf_paths = find_pdfs(settings.resolved_raw_pdf_dir)
    chunks: list[TextChunk] = []

    for pdf_path in pdf_paths:
        pages = read_pdf_pages(pdf_path)
        chunks.extend(
            chunk_pages(
                pages,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    if not chunks:
        raise ValueError(f"No PDF text found in {settings.resolved_raw_pdf_dir}")

    store = VectorStore(settings.resolved_index_dir)
    existing_ids = store.existing_chunk_ids()
    new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]

    if new_chunks:
        model = get_embedding_model(settings.embedding_model)
        embeddings = model.encode([chunk.text for chunk in new_chunks])
        store.append(new_chunks, embeddings)

    return IngestionResult(
        total_chunks=len(chunks),
        new_chunks=len(new_chunks),
        existing_chunks=len(chunks) - len(new_chunks),
    )
