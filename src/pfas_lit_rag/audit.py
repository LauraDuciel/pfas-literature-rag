import html
import re
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from pfas_lit_rag.config import Settings
from pfas_lit_rag.lexical_search import tokenize
from pfas_lit_rag.retrieval import search_index
from pfas_lit_rag.schemas import SearchResult


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    LIMITED_SUPPORT = "limited_support"
    UNSUPPORTED = "unsupported"
    CONTRADICTED = "contradicted"
    NOT_ENOUGH_EVIDENCE = "not_enough_evidence"


class EvidenceSnippet(BaseModel):
    citation: str
    score: float
    text: str


class ClaimAudit(BaseModel):
    claim: str
    status: ClaimStatus
    evidence: list[EvidenceSnippet] = Field(default_factory=list)
    confidence: str
    needs_human_review: bool


class AnswerAuditReport(BaseModel):
    question: str
    answer: str
    claims: list[ClaimAudit]

    @property
    def supported_count(self) -> int:
        return sum(1 for claim in self.claims if claim.status == ClaimStatus.SUPPORTED)

    @property
    def review_count(self) -> int:
        return sum(1 for claim in self.claims if claim.needs_human_review)


def audit_answer(
    question: str,
    answer: str,
    *,
    settings: Settings,
    top_k: int = 3,
) -> AnswerAuditReport:
    claims = extract_claims(answer)
    audits = [
        audit_claim(claim, settings=settings, top_k=top_k)
        for claim in claims
    ]
    return AnswerAuditReport(question=question, answer=answer, claims=audits)


def extract_claims(answer: str) -> list[str]:
    cleaned_lines = []
    in_sources = False
    for line in answer.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("sources:"):
            in_sources = True
            continue
        if in_sources or re.match(r"^\[\d+\]", stripped):
            continue
        if stripped:
            cleaned_lines.append(stripped)
    cleaned = " ".join(cleaned_lines)
    cleaned = re.sub(r"\[\d+\]", "", cleaned)
    candidates = re.split(r"(?<=[.!?])\s+", cleaned)
    return [
        candidate.strip()
        for candidate in candidates
        if _looks_like_claim(candidate)
    ]


def audit_claim(claim: str, *, settings: Settings, top_k: int = 3) -> ClaimAudit:
    results = search_index(claim, settings=settings, top_k=top_k)
    return classify_claim(claim, results)


def classify_claim(claim: str, results: list[SearchResult]) -> ClaimAudit:
    evidence = [
        EvidenceSnippet(
            citation=result.citation,
            score=result.score,
            text=_shorten(result.chunk.text),
        )
        for result in results
    ]
    if not evidence:
        return ClaimAudit(
            claim=claim,
            status=ClaimStatus.NOT_ENOUGH_EVIDENCE,
            evidence=[],
            confidence="low",
            needs_human_review=True,
        )

    overlap = max(_claim_evidence_overlap(claim, item.text) for item in evidence)
    if _looks_contradicted(claim, evidence[0].text, overlap):
        status = ClaimStatus.CONTRADICTED
        confidence = "medium"
    elif overlap >= 0.45:
        status = ClaimStatus.SUPPORTED
        confidence = "medium"
    elif overlap >= 0.25:
        status = ClaimStatus.LIMITED_SUPPORT
        confidence = "low"
    else:
        status = ClaimStatus.UNSUPPORTED
        confidence = "low"

    return ClaimAudit(
        claim=claim,
        status=status,
        evidence=evidence,
        confidence=confidence,
        needs_human_review=status != ClaimStatus.SUPPORTED,
    )


def write_audit_json(report: AnswerAuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")


def write_audit_html(report: AnswerAuditReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_audit_html(report), encoding="utf-8")


def render_audit_html(report: AnswerAuditReport) -> str:
    rows = []
    for claim in report.claims:
        evidence = "<br>".join(
            f"{html.escape(item.citation)} (score={item.score:.3f})"
            for item in claim.evidence
        ) or "No local evidence found"
        rows.append(
            "<tr>"
            f"<td>{html.escape(claim.claim)}</td>"
            f"<td>{html.escape(claim.status.value)}</td>"
            f"<td>{html.escape(claim.confidence)}</td>"
            f"<td>{'yes' if claim.needs_human_review else 'no'}</td>"
            f"<td>{evidence}</td>"
            "</tr>"
        )
    summary_html = (
        f"<strong>Claims:</strong> {len(report.claims)} | "
        f"<strong>Needs review:</strong> {report.review_count}"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Answer Audit</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; line-height: 1.4; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 0.5rem; vertical-align: top; }}
    th {{ background: #f3f3f3; }}
  </style>
</head>
<body>
  <h1>Answer Audit</h1>
  <p><strong>Question:</strong> {html.escape(report.question)}</p>
  <p>{summary_html}</p>
  <table>
    <thead>
      <tr><th>Claim</th><th>Status</th><th>Confidence</th><th>Review</th><th>Evidence</th></tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def _looks_like_claim(text: str) -> bool:
    if len(text.split()) < 5:
        return False
    lowered = text.lower()
    if lowered.startswith(("sources:", "citation", "reference")):
        return False
    return any(char.isalpha() for char in text)


def _claim_evidence_overlap(claim: str, evidence_text: str) -> float:
    claim_terms = _content_terms(claim)
    if not claim_terms:
        return 0.0
    evidence_terms = _content_terms(evidence_text)
    return len(claim_terms & evidence_terms) / len(claim_terms)


def _content_terms(text: str) -> set[str]:
    stopwords = {
        "about",
        "also",
        "and",
        "are",
        "been",
        "for",
        "from",
        "has",
        "have",
        "into",
        "that",
        "the",
        "their",
        "this",
        "used",
        "with",
    }
    return {term for term in tokenize(text) if len(term) >= 3 and term not in stopwords}


def _looks_contradicted(claim: str, evidence_text: str, overlap: float) -> bool:
    if overlap < 0.35:
        return False
    claim_lower = claim.lower()
    evidence_lower = evidence_text.lower()
    negative_claim = any(marker in claim_lower for marker in (" no ", " not ", "never", "without"))
    positive_evidence = any(
        marker in evidence_lower for marker in (" used", " detected", " observed")
    )
    return negative_claim and positive_evidence


def _shorten(text: str, max_chars: int = 900) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].strip() + " [...]"
