from pfas_lit_rag.literature_search import (
    _abstract_from_inverted_index,
    _is_relevant_work,
    _query_terms,
    _record_from_openalex,
)


def test_openalex_record_uses_best_pdf_url() -> None:
    record = _record_from_openalex(
        {
            "title": "PFAS extraction methods",
            "doi": "https://doi.org/10.1234/example",
            "publication_year": 2024,
            "best_oa_location": {
                "pdf_url": "https://example.org/paper.pdf",
                "landing_page_url": "https://example.org/paper",
                "license": "cc-by",
            },
        }
    )

    assert record is not None
    assert str(record.pdf_url) == "https://example.org/paper.pdf"
    assert record.doi == "10.1234/example"
    assert record.license == "cc-by"


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
