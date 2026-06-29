import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pfas_lit_rag.answering import answer_question
from pfas_lit_rag.config import Settings
from pfas_lit_rag.retrieval import search_index
from pfas_lit_rag.schemas import AnswerResponse, SearchResult


class ExpectedSource(BaseModel):
    title_contains: str | None = None
    page: int | None = None


class EvaluationCase(BaseModel):
    id: str
    question: str
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)


class CaseEvaluationResult(BaseModel):
    case_id: str
    question: str
    retrieved: list[SearchResult]
    answer: str | None = None
    recall_at_k: float
    reciprocal_rank: float
    citation_covered: bool
    unsupported_answer: bool
    expected_terms_found: int
    expected_terms_total: int


class EvaluationSummary(BaseModel):
    case_count: int
    top_k: int
    recall_at_k: float
    mean_reciprocal_rank: float
    citation_coverage: float
    unsupported_answer_count: int
    mean_expected_term_coverage: float


class EvaluationRunResult(BaseModel):
    cases: list[CaseEvaluationResult]
    summary: EvaluationSummary


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, list):
        raise ValueError('Evaluation file must contain a list of cases')
    return [EvaluationCase.model_validate(item) for item in payload]


def evaluate_cases(
    cases: list[EvaluationCase],
    *,
    settings: Settings,
    top_k: int,
    generate_answers: bool,
) -> EvaluationRunResult:
    results: list[CaseEvaluationResult] = []
    for case in cases:
        if generate_answers:
            response = answer_question(case.question, settings=settings, top_k=top_k)
            results.append(evaluate_answer(case, response, top_k=top_k))
        else:
            retrieved = search_index(case.question, settings=settings, top_k=top_k)
            results.append(evaluate_retrieval(case, retrieved, top_k=top_k))
    return EvaluationRunResult(cases=results, summary=summarise_results(results, top_k=top_k))


def evaluate_retrieval(
    case: EvaluationCase,
    retrieved: list[SearchResult],
    *,
    top_k: int,
) -> CaseEvaluationResult:
    recall = recall_at_k(retrieved, case.expected_sources, top_k=top_k)
    rank = reciprocal_rank(retrieved, case.expected_sources)
    return CaseEvaluationResult(
        case_id=case.id,
        question=case.question,
        retrieved=retrieved,
        recall_at_k=recall,
        reciprocal_rank=rank,
        citation_covered=False,
        unsupported_answer=False,
        expected_terms_found=0,
        expected_terms_total=len(case.expected_answer_terms),
    )


def evaluate_answer(
    case: EvaluationCase,
    response: AnswerResponse,
    *,
    top_k: int,
) -> CaseEvaluationResult:
    terms_found = count_expected_terms(response.answer, case.expected_answer_terms)
    return CaseEvaluationResult(
        case_id=case.id,
        question=case.question,
        retrieved=response.retrieved,
        answer=response.answer,
        recall_at_k=recall_at_k(response.retrieved, case.expected_sources, top_k=top_k),
        reciprocal_rank=reciprocal_rank(response.retrieved, case.expected_sources),
        citation_covered=has_citation_coverage(response),
        unsupported_answer=is_unsupported_answer(response.answer),
        expected_terms_found=terms_found,
        expected_terms_total=len(case.expected_answer_terms),
    )


def recall_at_k(
    retrieved: list[SearchResult],
    expected_sources: list[ExpectedSource],
    *,
    top_k: int,
) -> float:
    if not expected_sources:
        return 1.0
    retrieved_at_k = retrieved[:top_k]
    matched = sum(
        1
        for expected in expected_sources
        if any(source_matches(result, expected) for result in retrieved_at_k)
    )
    return matched / len(expected_sources)


def reciprocal_rank(
    retrieved: list[SearchResult],
    expected_sources: list[ExpectedSource],
) -> float:
    if not expected_sources:
        return 1.0
    for index, result in enumerate(retrieved, start=1):
        if any(source_matches(result, expected) for expected in expected_sources):
            return 1.0 / index
    return 0.0


def source_matches(result: SearchResult, expected: ExpectedSource) -> bool:
    if expected.title_contains:
        needle = expected.title_contains.lower()
        if needle not in result.chunk.title.lower():
            return False
    if expected.page is not None:
        if not (result.chunk.page_start <= expected.page <= result.chunk.page_end):
            return False
    return True


def has_citation_coverage(response: AnswerResponse) -> bool:
    if not response.citations:
        return False
    return 'Sources:' in response.answer or any(
        f'[{index}]' in response.answer for index in range(1, 10)
    )


def is_unsupported_answer(answer: str) -> bool:
    answer_lower = answer.lower()
    markers = [
        'does not contain enough evidence',
        'no relevant passages',
        'not enough evidence',
        'insufficient evidence',
    ]
    return any(marker in answer_lower for marker in markers)


def count_expected_terms(answer: str, expected_terms: list[str]) -> int:
    answer_lower = answer.lower()
    return sum(1 for term in expected_terms if term.lower() in answer_lower)


def summarise_results(results: list[CaseEvaluationResult], *, top_k: int) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(
            case_count=0,
            top_k=top_k,
            recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
            citation_coverage=0.0,
            unsupported_answer_count=0,
            mean_expected_term_coverage=0.0,
        )
    term_coverages = []
    for result in results:
        if result.expected_terms_total == 0:
            term_coverages.append(1.0)
        else:
            term_coverages.append(result.expected_terms_found / result.expected_terms_total)
    return EvaluationSummary(
        case_count=len(results),
        top_k=top_k,
        recall_at_k=sum(result.recall_at_k for result in results) / len(results),
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results) / len(results),
        citation_coverage=sum(1 for result in results if result.citation_covered) / len(results),
        unsupported_answer_count=sum(1 for result in results if result.unsupported_answer),
        mean_expected_term_coverage=sum(term_coverages) / len(term_coverages),
    )


def write_json_result(result: EvaluationRunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(result.model_dump(mode='json'), handle, indent=2, ensure_ascii=False)


def write_markdown_report(
    result: EvaluationRunResult,
    path: Path,
    *,
    settings: Settings,
    eval_path: Path,
    generate_answers: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_markdown_report(
            result, settings=settings, eval_path=eval_path, generate_answers=generate_answers
        ),
        encoding='utf-8',
    )


def render_markdown_report(
    result: EvaluationRunResult,
    *,
    settings: Settings,
    eval_path: Path,
    generate_answers: bool,
) -> str:
    summary = result.summary
    lines = [
        '# RAG Evaluation Summary',
        '',
        f'- Evaluation set: `{eval_path}`',
        f'- Cases: {summary.case_count}',
        f'- Mode: {"retrieval + answer generation" if generate_answers else "retrieval only"}',
        f'- Ollama model: `{settings.ollama_model}`',
        f'- Embedding model: `{settings.embedding_model}`',
        f'- top_k: {summary.top_k}',
        f'- Vector/BM25 weights: {settings.vector_weight:.2f} / {settings.lexical_weight:.2f}',
        (
            f'- Reranking: {settings.rerank_enabled} '
            f'(backend={settings.rerank_backend}, weight={settings.rerank_weight:.2f})'
        ),
    ]
    if settings.rerank_backend in {'auto', 'cross_encoder'}:
        lines.append(f'- Cross-encoder model: `{settings.cross_encoder_model}`')
    lines.extend([
        '',
        '## Metrics',
        '',
        f'- Recall@{summary.top_k}: {summary.recall_at_k:.3f}',
        f'- Mean reciprocal rank: {summary.mean_reciprocal_rank:.3f}',
        f'- Citation coverage: {summary.citation_coverage:.3f}',
        f'- Unsupported answer count: {summary.unsupported_answer_count}',
        f'- Expected term coverage: {summary.mean_expected_term_coverage:.3f}',
        '',
        '## Cases',
        '',
    ])
    for case_result in result.cases:
        lines.extend(
            [
                f'### {case_result.case_id}',
                '',
                f'Question: {case_result.question}',
                '',
                f'- Recall@{summary.top_k}: {case_result.recall_at_k:.3f}',
                f'- Reciprocal rank: {case_result.reciprocal_rank:.3f}',
                f'- Citation covered: {case_result.citation_covered}',
                f'- Unsupported answer: {case_result.unsupported_answer}',
                '',
                'Retrieved sources:',
            ]
        )
        for index, retrieved in enumerate(case_result.retrieved, start=1):
            lines.append(f'- [{index}] {retrieved.citation} (score={retrieved.score:.3f})')
        lines.append('')
    return '\n'.join(lines).strip() + '\n'


def mlflow_metrics(summary: EvaluationSummary) -> dict[str, float]:
    return {
        'case_count': float(summary.case_count),
        f'recall_at_{summary.top_k}': summary.recall_at_k,
        'mean_reciprocal_rank': summary.mean_reciprocal_rank,
        'citation_coverage': summary.citation_coverage,
        'unsupported_answer_count': float(summary.unsupported_answer_count),
        'mean_expected_term_coverage': summary.mean_expected_term_coverage,
    }


def mlflow_params(settings: Settings, *, top_k: int, generate_answers: bool) -> dict[str, Any]:
    return {
        'ollama_model': settings.ollama_model,
        'embedding_model': settings.embedding_model,
        'top_k': top_k,
        'lexical_candidate_k': settings.lexical_candidate_k,
        'vector_weight': settings.vector_weight,
        'lexical_weight': settings.lexical_weight,
        'rerank_enabled': settings.rerank_enabled,
        'rerank_backend': settings.rerank_backend,
        'rerank_weight': settings.rerank_weight,
        'cross_encoder_model': settings.cross_encoder_model,
        'cross_encoder_candidate_k': settings.cross_encoder_candidate_k,
        'cross_encoder_batch_size': settings.cross_encoder_batch_size,
        'context_chars_per_chunk': settings.context_chars_per_chunk,
        'ollama_num_predict': settings.ollama_num_predict,
        'generate_answers': generate_answers,
    }
