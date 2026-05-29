from typing import Any

import httpx

from pfas_lit_rag.config import Settings
from pfas_lit_rag.schemas import LiteratureRecord

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class LiteratureSearchClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.Client(
            timeout=settings.request_timeout_seconds,
            headers={"User-Agent": settings.collector_user_agent},
        )

    def search_openalex(self, query: str, max_results: int) -> list[LiteratureRecord]:
        response = self.client.get(
            OPENALEX_WORKS_URL,
            params={
                "search": query,
                "filter": "open_access.is_oa:true,type:article",
                "per-page": min(max_results, 200),
                "sort": "relevance_score:desc",
            },
        )
        response.raise_for_status()
        payload = response.json()
        records = [_record_from_openalex(work) for work in payload.get("results", [])]
        return [record for record in records if record is not None][:max_results]


def _record_from_openalex(work: dict[str, Any]) -> LiteratureRecord | None:
    pdf_url = _best_pdf_url(work)
    if not pdf_url:
        return None

    primary_location = work.get("primary_location") or {}
    best_location = work.get("best_oa_location") or {}
    landing_url = (
        best_location.get("landing_page_url")
        or primary_location.get("landing_page_url")
        or work.get("doi")
        or work.get("id")
    )
    return LiteratureRecord(
        title=work.get("title") or "Untitled",
        source="openalex",
        pdf_url=pdf_url,
        landing_url=landing_url,
        doi=_normalise_doi(work.get("doi")),
        publication_year=work.get("publication_year"),
        license=best_location.get("license") or primary_location.get("license"),
    )


def _best_pdf_url(work: dict[str, Any]) -> str | None:
    best_location = work.get("best_oa_location") or {}
    if best_location.get("pdf_url"):
        return best_location["pdf_url"]

    primary_location = work.get("primary_location") or {}
    if primary_location.get("pdf_url"):
        return primary_location["pdf_url"]

    for location in work.get("locations") or []:
        if location.get("pdf_url"):
            return location["pdf_url"]
    return None


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/").strip()
