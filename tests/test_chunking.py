from pathlib import Path

from pfas_lit_rag.chunking import chunk_pages
from pfas_lit_rag.schemas import DocumentPage


def test_chunk_pages_preserves_page_metadata() -> None:
    page = DocumentPage(
        document_id="doc-1",
        title="PFAS methods",
        source_path=Path("paper.pdf"),
        page_number=3,
        text=" ".join(f"word{i}" for i in range(12)),
    )

    chunks = chunk_pages([page], chunk_size=5, chunk_overlap=2)

    assert len(chunks) == 4
    assert chunks[0].page_start == 3
    assert chunks[0].page_end == 3
    assert chunks[0].title == "PFAS methods"
    assert chunks[1].text.startswith("word3")


def test_chunking_rejects_invalid_overlap() -> None:
    page = DocumentPage(
        document_id="doc-1",
        title="PFAS methods",
        source_path=Path("paper.pdf"),
        page_number=1,
        text="short text",
    )

    try:
        chunk_pages([page], chunk_size=10, chunk_overlap=10)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:
        raise AssertionError("Expected invalid overlap to raise ValueError")
