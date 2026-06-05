from pydantic import BaseModel

from pfas_lit_rag.chunking import chunk_pages
from pfas_lit_rag.config import Settings
from pfas_lit_rag.document_manifest import DocumentManifest
from pfas_lit_rag.embeddings import get_embedding_model
from pfas_lit_rag.pdfs import file_sha256, find_pdfs, read_pdf_pages
from pfas_lit_rag.schemas import DocumentRecord, TextChunk
from pfas_lit_rag.vector_store import VectorStore, chunk_fingerprint


class IngestionResult(BaseModel):
    documents_scanned: int
    new_documents: int
    existing_documents: int
    pages_extracted: int
    total_chunks: int
    new_chunks: int
    existing_chunks: int
    rebuilt_index: bool = False


def build_index(settings: Settings) -> IngestionResult:
    pdf_paths = find_pdfs(settings.resolved_raw_pdf_dir)
    chunks: list[TextChunk] = []
    document_records: list[DocumentRecord] = []
    seen_document_ids: set[str] = set()
    duplicate_documents = 0
    pages_extracted = 0

    manifest = DocumentManifest(settings.resolved_metadata_dir)
    previous_records = manifest.load()

    for pdf_path in pdf_paths:
        digest = file_sha256(pdf_path)
        document_id = digest[:12]
        if document_id in seen_document_ids:
            duplicate_documents += 1
            continue
        seen_document_ids.add(document_id)

        pages = read_pdf_pages(pdf_path)
        document_chunks = chunk_pages(
            pages,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        pages_extracted += len(pages)
        chunks.extend(document_chunks)

        title = pages[0].title if pages else pdf_path.stem
        document_records.append(
            DocumentRecord(
                document_id=document_id,
                title=title,
                source_path=str(pdf_path),
                file_sha256=digest,
                pages_extracted=len(pages),
                chunks_indexed=len(document_chunks),
            )
        )

    if not chunks:
        raise ValueError(f"No PDF text found in {settings.resolved_raw_pdf_dir}")

    store = VectorStore(settings.resolved_index_dir)
    rebuilt_index = False
    if store.has_duplicate_fingerprints():
        model = get_embedding_model(settings.embedding_model)
        embeddings = model.encode([chunk.text for chunk in chunks])
        store.write(chunks, embeddings)
        new_chunks = chunks
        rebuilt_index = True
    else:
        existing_ids = store.existing_chunk_ids()
        existing_fingerprints = store.existing_chunk_fingerprints()
        new_chunks = [
            chunk
            for chunk in chunks
            if chunk.chunk_id not in existing_ids
            and chunk_fingerprint(chunk) not in existing_fingerprints
        ]
        if new_chunks:
            model = get_embedding_model(settings.embedding_model)
            embeddings = model.encode([chunk.text for chunk in new_chunks])
            store.append(new_chunks, embeddings)

    merged_records_by_id = manifest.load()
    for record in document_records:
        merged_records_by_id[record.document_id] = record
    manifest.write(list(merged_records_by_id.values()))

    existing_documents = sum(
        1 for record in document_records if record.document_id in previous_records
    )
    return IngestionResult(
        documents_scanned=len(pdf_paths),
        new_documents=len(document_records) - existing_documents,
        existing_documents=existing_documents + duplicate_documents,
        pages_extracted=pages_extracted,
        total_chunks=len(chunks),
        new_chunks=len(new_chunks),
        existing_chunks=0 if rebuilt_index else len(chunks) - len(new_chunks),
        rebuilt_index=rebuilt_index,
    )
