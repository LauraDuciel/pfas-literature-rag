from pathlib import Path

from pfas_lit_rag.config import Settings
from pfas_lit_rag.corpus_report import render_corpus_report
from pfas_lit_rag.document_manifest import DocumentManifest
from pfas_lit_rag.schemas import DocumentRecord, TextChunk


def _settings(tmp_path) -> Settings:
    return Settings(
        project_root=tmp_path,
        raw_pdf_dir=Path("raw_pdfs"),
        metadata_dir=Path("metadata"),
        index_dir=Path("index"),
    )


def test_render_corpus_report_summarises_manifest_and_index(tmp_path) -> None:
    settings = _settings(tmp_path)
    settings.resolved_raw_pdf_dir.mkdir(parents=True)
    settings.resolved_index_dir.mkdir(parents=True)
    (settings.resolved_raw_pdf_dir / "paper.pdf").write_bytes(b"%PDF-1.4")

    DocumentManifest(settings.resolved_metadata_dir).write([
        DocumentRecord(
            document_id="doc1",
            title="PFAS analytical methods",
            source_path="paper.pdf",
            file_sha256="a" * 64,
            pages_extracted=3,
            chunks_indexed=2,
        )
    ])
    chunk = TextChunk(
        chunk_id="c1",
        document_id="doc1",
        title="PFAS analytical methods",
        source_path="paper.pdf",
        page_start=1,
        page_end=1,
        text="LC-MS/MS method for PFAS detection",
    )
    (settings.resolved_index_dir / "chunks.jsonl").write_text(
        chunk.model_dump_json() + "\n",
        encoding="utf-8",
    )

    report = render_corpus_report(settings)

    assert "Raw PDF files: 1" in report
    assert "Document manifest records: 1" in report
    assert "Indexed chunks: 1" in report
    assert "PFAS analytical methods | pages=3 | chunks=2" in report
