from pathlib import Path

import httpx

from pfas_lit_rag.pdf_downloader import download_records
from pfas_lit_rag.schemas import LiteratureRecord


class DummySettings:
    resolved_raw_pdf_dir = Path("data/raw_pdfs")
    resolved_metadata_dir = Path("data/metadata")
    request_timeout_seconds = 1.0
    collector_user_agent = "test"


def test_download_records_skips_failed_pdf(monkeypatch, tmp_path: Path) -> None:
    settings = DummySettings()
    settings.resolved_raw_pdf_dir = tmp_path / "pdfs"
    settings.resolved_metadata_dir = tmp_path / "metadata"

    def raise_connect_error(*args, **kwargs):
        raise httpx.ConnectError("Name or service not known")

    monkeypatch.setattr(httpx.Client, "get", raise_connect_error)

    record = LiteratureRecord(
        title="PFAS paper",
        source="openalex",
        pdf_url="https://example.org/paper.pdf",
    )

    result = download_records([record], settings)  # type: ignore[arg-type]

    assert result.downloaded == []
    assert result.existing == []
    assert len(result.failed) == 1
    assert result.failed[0].title == "PFAS paper"


def test_download_records_skips_existing_pdf(monkeypatch, tmp_path: Path) -> None:
    settings = DummySettings()
    settings.resolved_raw_pdf_dir = tmp_path / "pdfs"
    settings.resolved_metadata_dir = tmp_path / "metadata"
    settings.resolved_raw_pdf_dir.mkdir(parents=True)
    existing_pdf = settings.resolved_raw_pdf_dir / "10.1234_example.pdf"
    existing_pdf.write_bytes(b"%PDF existing")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("existing PDFs should not be downloaded again")

    monkeypatch.setattr(httpx.Client, "get", fail_if_called)

    record = LiteratureRecord(
        title="PFAS paper",
        source="openalex",
        doi="10.1234/example",
        pdf_url="https://example.org/paper.pdf",
    )

    result = download_records([record], settings)  # type: ignore[arg-type]

    assert result.downloaded == []
    assert len(result.existing) == 1
    assert result.existing[0].downloaded_path == str(existing_pdf)
    assert result.failed == []


def test_download_records_writes_enriched_manifest(monkeypatch, tmp_path: Path) -> None:
    settings = DummySettings()
    settings.resolved_raw_pdf_dir = tmp_path / "pdfs"
    settings.resolved_metadata_dir = tmp_path / "metadata"

    def fake_get(*args, **kwargs):
        return httpx.Response(
            200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF content",
            request=httpx.Request("GET", "https://example.org/paper.pdf"),
        )

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    record = LiteratureRecord(
        title="PFAS paper",
        source="openalex",
        doi="10.1234/example",
        openalex_id="https://openalex.org/W123",
        journal="Analytical Methods",
        authors=["A. Chemist"],
        concepts=["Mass spectrometry"],
        pdf_url="https://example.org/paper.pdf",
    )

    result = download_records([record], settings)  # type: ignore[arg-type]
    manifest_text = (settings.resolved_metadata_dir / "literature_manifest.jsonl").read_text()

    assert len(result.downloaded) == 1
    assert '"openalex_id": "https://openalex.org/W123"' in manifest_text
    assert '"journal": "Analytical Methods"' in manifest_text
    assert '"authors": ["A. Chemist"]' in manifest_text
