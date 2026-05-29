import hashlib
from pathlib import Path

from pypdf import PdfReader

from pfas_lit_rag.schemas import DocumentPage


def document_id_for_path(path: Path) -> str:
    digest = hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


def read_pdf_pages(path: Path) -> list[DocumentPage]:
    reader = PdfReader(path)
    title = _metadata_title(reader) or path.stem
    document_id = document_id_for_path(path)

    pages: list[DocumentPage] = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = _clean_text(text)
        if not text:
            continue
        pages.append(
            DocumentPage(
                document_id=document_id,
                title=title,
                source_path=path,
                page_number=index,
                text=text,
            )
        )
    return pages


def find_pdfs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.glob("**/*.pdf") if path.is_file())


def _metadata_title(reader: PdfReader) -> str | None:
    metadata = reader.metadata
    if not metadata:
        return None
    title = metadata.get("/Title")
    if not title:
        return None
    title = str(title).strip()
    return title or None


def _clean_text(text: str) -> str:
    return " ".join(text.replace("\x00", " ").split())
