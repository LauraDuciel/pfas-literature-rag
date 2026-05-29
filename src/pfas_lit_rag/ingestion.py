
from pfas_lit_rag.chunking import chunk_pages
from pfas_lit_rag.config import Settings
from pfas_lit_rag.embeddings import get_embedding_model
from pfas_lit_rag.pdfs import find_pdfs, read_pdf_pages
from pfas_lit_rag.schemas import TextChunk
from pfas_lit_rag.vector_store import VectorStore


def build_index(settings: Settings) -> list[TextChunk]:
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

    model = get_embedding_model(settings.embedding_model)
    embeddings = model.encode([chunk.text for chunk in chunks])
    VectorStore(settings.resolved_index_dir).write(chunks, embeddings)
    return chunks
