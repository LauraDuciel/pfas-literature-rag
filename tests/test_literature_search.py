from pfas_lit_rag.literature_search import _record_from_openalex


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
