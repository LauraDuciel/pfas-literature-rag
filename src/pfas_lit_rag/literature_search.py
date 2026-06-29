import re
from typing import Any

import httpx

from pfas_lit_rag.config import Settings
from pfas_lit_rag.schemas import LiteratureRecord

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "into",
    "of",
    "the",
    "to",
    "with",
}

PFAS_ANCHOR_TERMS = {
    "pfas",
    "pfoa",
    "pfos",
    "pfhxs",
    "genx",
    "perfluoroalkyl",
    "polyfluoroalkyl",
    "perfluorinated",
    "fluorochemical",
}


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
                "per-page": min(max(max_results * 5, 25), 200),
                "sort": "relevance_score:desc",
            },
        )
        response.raise_for_status()
        payload = response.json()

        records: list[LiteratureRecord] = []
        query_terms = _query_terms(query)
        for work in payload.get("results", []):
            if not _is_relevant_work(work, query_terms):
                continue
            record = _record_from_openalex(work)
            if record is not None:
                records.append(record)
            if len(records) >= max_results:
                break
        return records


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
    source = primary_location.get("source") or best_location.get("source") or {}
    open_access = work.get("open_access") or {}
    return LiteratureRecord(
        title=work.get("title") or "Untitled",
        source="openalex",
        pdf_url=pdf_url,
        landing_url=landing_url,
        doi=_normalise_doi(work.get("doi") or (work.get("ids") or {}).get("doi")),
        openalex_id=work.get("id"),
        publication_year=work.get("publication_year"),
        publication_date=work.get("publication_date"),
        journal=source.get("display_name"),
        publisher=source.get("host_organization_name") or source.get("publisher"),
        cited_by_count=work.get("cited_by_count"),
        work_type=work.get("type"),
        is_open_access=open_access.get("is_oa"),
        open_access_status=open_access.get("oa_status"),
        open_access_version=best_location.get("version") or primary_location.get("version"),
        license=best_location.get("license") or primary_location.get("license"),
        authors=_author_names(work),
        concepts=_concept_names(work),
    )


def _is_relevant_work(work: dict[str, Any], query_terms: set[str]) -> bool:
    text = _search_text(work)
    tokens = set(_tokenize(text))
    has_pfas_anchor = bool(tokens & PFAS_ANCHOR_TERMS)
    if query_terms & PFAS_ANCHOR_TERMS:
        return has_pfas_anchor
    if has_pfas_anchor:
        return True
    matched_query_terms = tokens & query_terms
    return len(matched_query_terms) >= min(2, len(query_terms))


def _search_text(work: dict[str, Any]) -> str:
    title = work.get("title") or ""
    abstract = _abstract_from_inverted_index(work.get("abstract_inverted_index"))
    return f"{title} {abstract}"


def _abstract_from_inverted_index(index: dict[str, list[int]] | None) -> str:
    if not index:
        return ""

    positioned_words: list[tuple[int, str]] = []
    for word, positions in index.items():
        for position in positions:
            positioned_words.append((position, word))

    return " ".join(word for _, word in sorted(positioned_words))


def _query_terms(query: str) -> set[str]:
    return {token for token in _tokenize(query) if token not in STOPWORDS and len(token) >= 3}


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


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


def _author_names(work: dict[str, Any], *, limit: int = 8) -> list[str]:
    names: list[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name")
        if name:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def _concept_names(work: dict[str, Any], *, limit: int = 8) -> list[str]:
    concepts = sorted(
        work.get("concepts") or [],
        key=lambda item: item.get("score") or 0.0,
        reverse=True,
    )
    return [
        concept["display_name"]
        for concept in concepts[:limit]
        if concept.get("display_name")
    ]


def _normalise_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/").strip()
