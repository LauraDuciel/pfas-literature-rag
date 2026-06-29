from pfas_lit_rag.audit import (
    ClaimStatus,
    audit_answer,
    classify_claim,
    extract_claims,
    render_audit_html,
)
from pfas_lit_rag.config import Settings
from pfas_lit_rag.schemas import SearchResult, TextChunk


def _result(text: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        score=score,
        chunk=TextChunk(
            chunk_id="c1",
            document_id="d1",
            title="PFAS methods",
            source_path="paper.pdf",
            page_start=2,
            page_end=2,
            text=text,
        ),
    )


def test_extract_claims_ignores_sources_section() -> None:
    answer = """LC-MS/MS is used for PFAS detection [1].

Sources:
[1] PFAS methods, p. 2
"""

    claims = extract_claims(answer)

    assert claims == ["LC-MS/MS is used for PFAS detection ."]


def test_classify_claim_marks_supported_when_terms_overlap() -> None:
    audit = classify_claim(
        "LC-MS/MS is used for PFAS detection in environmental samples.",
        [_result("PFAS detection in environmental samples commonly uses LC-MS/MS methods.")],
    )

    assert audit.status == ClaimStatus.SUPPORTED
    assert not audit.needs_human_review


def test_classify_claim_marks_not_enough_evidence_without_results() -> None:
    audit = classify_claim("PFAS are measured with a proprietary device.", [])

    assert audit.status == ClaimStatus.NOT_ENOUGH_EVIDENCE
    assert audit.needs_human_review


def test_audit_answer_uses_local_search(monkeypatch, tmp_path) -> None:
    def fake_search_index(query, *, settings, top_k):
        return [_result("PFAS detection uses LC-MS/MS in water samples.")]

    monkeypatch.setattr("pfas_lit_rag.audit.search_index", fake_search_index)

    report = audit_answer(
        "What methods are used?",
        "LC-MS/MS is used for PFAS detection.",
        settings=Settings(project_root=tmp_path),
    )

    assert len(report.claims) == 1
    assert report.claims[0].status == ClaimStatus.SUPPORTED


def test_render_audit_html_contains_claim_status() -> None:
    audit = classify_claim(
        "LC-MS/MS is used for PFAS detection.",
        [_result("PFAS detection uses LC-MS/MS.")],
    )
    report = type("Report", (), {
        "question": "q",
        "claims": [audit],
        "review_count": 0,
    })()

    html = render_audit_html(report)

    assert "supported" in html
    assert "LC-MS/MS" in html
