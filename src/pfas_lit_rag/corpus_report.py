from collections import Counter
from pathlib import Path

from pfas_lit_rag.config import Settings
from pfas_lit_rag.document_manifest import DocumentManifest
from pfas_lit_rag.pdfs import find_pdfs
from pfas_lit_rag.vector_store import VectorStore


def render_corpus_report(settings: Settings) -> str:
    pdfs = find_pdfs(settings.resolved_raw_pdf_dir)
    manifest_records = list(DocumentManifest(settings.resolved_metadata_dir).load().values())
    store = VectorStore(settings.resolved_index_dir)
    chunks = store.load_chunks_if_available()

    total_pages = sum(record.pages_extracted for record in manifest_records)
    total_manifest_chunks = sum(record.chunks_indexed for record in manifest_records)
    hash_counts = Counter(record.file_sha256 for record in manifest_records)
    duplicate_hashes = sum(1 for count in hash_counts.values() if count > 1)
    low_text_records = [record for record in manifest_records if record.chunks_indexed <= 1]

    lines = [
        '# Corpus Report',
        '',
        f'- Raw PDF files: {len(pdfs)}',
        f'- Document manifest records: {len(manifest_records)}',
        f'- Extracted pages: {total_pages}',
        f'- Manifest chunks: {total_manifest_chunks}',
        f'- Indexed chunks: {len(chunks)}',
        f'- Duplicate content hashes in manifest: {duplicate_hashes}',
        f'- Duplicate chunk fingerprints in index: {store.has_duplicate_fingerprints()}',
        '',
        '## Documents',
        '',
    ]

    for record in sorted(manifest_records, key=lambda item: item.title.lower()):
        lines.append(
            f'- {record.title} | pages={record.pages_extracted} | '
            f'chunks={record.chunks_indexed} | sha256={record.file_sha256[:12]}'
        )

    if low_text_records:
        lines.extend(['', '## Documents with little extracted text', ''])
        for record in low_text_records:
            lines.append(f'- {record.title} | chunks={record.chunks_indexed}')

    return '\n'.join(lines).strip() + '\n'


def write_corpus_report(settings: Settings, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_corpus_report(settings), encoding='utf-8')
