import hashlib

from pfas_lit_rag.schemas import DocumentPage, TextChunk


def chunk_pages(
    pages: list[DocumentPage],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for page in pages:
        for text in _sliding_chunks(page.text, chunk_size=chunk_size, overlap=chunk_overlap):
            chunk_id = _chunk_id(page.document_id, page.page_number, text)
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    title=page.title,
                    source_path=str(page.source_path),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    text=text,
                )
            )
    return chunks


def _sliding_chunks(text: str, *, chunk_size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def _chunk_id(document_id: str, page_number: int, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"{document_id}:p{page_number}:{digest}"
