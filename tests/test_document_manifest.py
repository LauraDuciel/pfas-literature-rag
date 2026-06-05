from pfas_lit_rag.document_manifest import DocumentManifest
from pfas_lit_rag.schemas import DocumentRecord


def test_document_manifest_round_trip(tmp_path) -> None:
    manifest = DocumentManifest(tmp_path)
    record = DocumentRecord(
        document_id="doc1",
        title="PFAS paper",
        source_path="paper.pdf",
        file_sha256="abc",
        pages_extracted=2,
        chunks_indexed=3,
    )

    manifest.write([record])

    loaded = manifest.load()
    assert loaded["doc1"].title == "PFAS paper"
    assert loaded["doc1"].chunks_indexed == 3
