import json
import re
from pathlib import Path

import httpx
from pydantic import BaseModel

from pfas_lit_rag.config import Settings
from pfas_lit_rag.schemas import LiteratureRecord


class DownloadFailure(BaseModel):
    title: str
    url: str
    reason: str


class DownloadResult(BaseModel):
    downloaded: list[LiteratureRecord]
    existing: list[LiteratureRecord]
    failed: list[DownloadFailure]


def download_records(records: list[LiteratureRecord], settings: Settings) -> DownloadResult:
    settings.resolved_raw_pdf_dir.mkdir(parents=True, exist_ok=True)
    settings.resolved_metadata_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.resolved_metadata_dir / "literature_manifest.jsonl"

    downloaded: list[LiteratureRecord] = []
    existing: list[LiteratureRecord] = []
    failed: list[DownloadFailure] = []
    with httpx.Client(
        timeout=settings.request_timeout_seconds,
        follow_redirects=True,
        headers={"User-Agent": settings.collector_user_agent},
    ) as client:
        for record in records:
            target_path = _target_path(record, settings.resolved_raw_pdf_dir)
            updated = record.model_copy(update={"downloaded_path": str(target_path)})
            if target_path.exists():
                existing.append(updated)
                continue

            try:
                _download_pdf(client, str(record.pdf_url), target_path)
            except (httpx.HTTPError, ValueError) as exc:
                failed.append(
                    DownloadFailure(
                        title=record.title,
                        url=str(record.pdf_url),
                        reason=str(exc),
                    )
                )
                continue
            downloaded.append(updated)

    with manifest_path.open("a", encoding="utf-8") as handle:
        for record in downloaded:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")

    return DownloadResult(downloaded=downloaded, existing=existing, failed=failed)


def _download_pdf(client: httpx.Client, url: str, target_path: Path) -> None:
    response = client.get(url)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    content = response.content
    if "pdf" not in content_type and not content.startswith(b"%PDF"):
        raise ValueError(f"URL did not return a PDF: {url}")
    target_path.write_bytes(content)


def _target_path(record: LiteratureRecord, output_dir: Path) -> Path:
    prefix = record.doi or record.title
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", prefix).strip("_").lower()
    stem = stem[:100] or "document"
    return output_dir / f"{stem}.pdf"
