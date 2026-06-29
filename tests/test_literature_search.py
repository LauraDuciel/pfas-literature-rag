from pfas_lit_rag.literature_search import (
    _abstract_from_inverted_index,
    _concept_names,
    _is_relevant_work,
    _query_terms,
    _record_from_openalex,
)


def test_openalex_record_uses_best_pdf_url() -> None:
    record = _record_from_openalex(
        {
            "title": "PFAS extraction methods",
            "doi": "https://doi.org/10.1234/example",
            "id": "https://openalex.org/W123",
            "publication_year": 2024,
            "publication_date": "2024-02-03",
            "type": "article",
            "cited_by_count": 12,
            "open_access": {"is_oa": True, "oa_status": "gold"},
            "authorships": [
                {"author": {"display_name": "A. Chemist"}},
                {"author": {"display_name": "B. Analyst"}},
            ],
            "concepts": [
                {"display_name": "Environmental chemistry", "score": 0.7},
                {"display_name": "Mass spectrometry", "score": 0.9},
            ],
            "primary_location": {
                "source": {
                    "display_name": "Analytical Methods",
                    "host_organization_name": "Royal Society of Chemistry",
                }
            },
            "best_oa_location": {
                "pdf_url": "https://example.org/paper.pdf",
                "landing_page_url": "https://example.org/paper",
                "license": "cc-by",
                "version": "publishedVersion",
            },
        }
    )

    assert record is not None
    assert str(record.pdf_url) == "https://example.org/paper.pdf"
    assert record.doi == "10.1234/example"
    assert record.openalex_id == "https://openalex.org/W123"
    assert record.publication_date == "2024-02-03"
    assert record.journal == "Analytical Methods"
    assert record.publisher == "Royal Society of Chemistry"
    assert record.cited_by_count == 12
    assert record.work_type == "article"
    assert record.is_open_access is True
    assert record.open_access_status == "gold"
    assert record.open_access_version == "publishedVersion"
    assert record.license == "cc-by"
    assert record.authors == ["A. Chemist", "B. Analyst"]
    assert record.concepts == ["Mass spectrometry", "Environmental chemistry"]


def test_abstract_from_openalex_inverted_index() -> None:
    abstract = _abstract_from_inverted_index(
        {
            "PFAS": [0],
            "analysis": [2],
            "NMR": [1],
        }
    )

    assert abstract == "PFAS NMR analysis"


def test_relevance_filter_keeps_pfas_title() -> None:
    work = {"title": "PFAS occurrence in drinking water"}

    assert _is_relevant_work(work, _query_terms("PFAS NMR analytical chemistry"))


def test_relevance_filter_uses_abstract_terms_without_pfas_query() -> None:
    work = {
        "title": "Trace contaminant workflow",
        "abstract_inverted_index": {
            "Analytical": [0],
            "chemistry": [1],
            "methods": [2],
            "for": [3],
            "NMR": [4],
            "screening": [5],
        },
    }

    assert _is_relevant_work(work, _query_terms("NMR analytical chemistry"))


def test_relevance_filter_requires_pfas_anchor_for_pfas_query() -> None:
    work = {
        "title": "Trace contaminant workflow",
        "abstract_inverted_index": {
            "Analytical": [0],
            "chemistry": [1],
            "methods": [2],
            "for": [3],
            "NMR": [4],
            "screening": [5],
        },
    }

    assert not _is_relevant_work(work, _query_terms("PFAS NMR analytical chemistry"))


def test_relevance_filter_rejects_unrelated_work() -> None:
    work = {
        "title": "A high capacity quinone cathode for zinc batteries",
        "abstract_inverted_index": {
            "battery": [0],
            "electrode": [1],
            "aqueous": [2],
            "zinc": [3],
        },
    }

    assert not _is_relevant_work(work, _query_terms("PFAS NMR analytical chemistry"))


def test_concept_names_are_sorted_by_openalex_score() -> None:
    concepts = _concept_names(
        {
            "concepts": [
                {"display_name": "Lower", "score": 0.2},
                {"display_name": "Higher", "score": 0.8},
            ]
        }
    )

    assert concepts == ["Higher", "Lower"]
