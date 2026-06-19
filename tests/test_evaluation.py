from pathlib import Path

from pfas_lit_rag.config import Settings
from pfas_lit_rag.evaluation import (
    CaseEvaluationResult,
    EvaluationCase,
    EvaluationRunResult,
    EvaluationSummary,
    ExpectedSource,
    count_expected_terms,
    evaluate_retrieval,
    has_citation_coverage,
    is_unsupported_answer,
    load_evaluation_cases,
    recall_at_k,
    render_markdown_report,
)
from pfas_lit_rag.schemas import AnswerResponse, SearchResult, TextChunk


def _result(title: str, page: int = 1) -> SearchResult:
    return SearchResult(
        score=0.9,
        chunk=TextChunk(
            chunk_id=f'{title}-{page}',
            document_id='doc',
            title=title,
            source_path='paper.pdf',
            page_start=page,
            page_end=page,
            text='PFAS LC-MS/MS extraction methods',
        ),
    )


def test_load_evaluation_cases_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / 'eval.yaml'
    path.write_text(
        """- id: methods
  question: What methods are used?
  expected_sources:
    - title_contains: extraction methods
  expected_answer_terms:
    - LC-MS
""",
        encoding='utf-8',
    )

    cases = load_evaluation_cases(path)

    assert cases[0].id == 'methods'
    assert cases[0].expected_sources[0].title_contains == 'extraction methods'


def test_recall_at_k_matches_expected_source() -> None:
    results = [_result('Comparison of extraction methods for PFAS'), _result('Other paper')]
    expected = [ExpectedSource(title_contains='extraction methods')]

    assert recall_at_k(results, expected, top_k=1) == 1.0


def test_evaluate_retrieval_computes_rank_metrics() -> None:
    case = EvaluationCase(
        id='methods',
        question='What methods are used?',
        expected_sources=[ExpectedSource(title_contains='target paper')],
    )
    results = [_result('background'), _result('target paper')]

    evaluated = evaluate_retrieval(case, results, top_k=2)

    assert evaluated.recall_at_k == 1.0
    assert evaluated.reciprocal_rank == 0.5


def test_has_citation_coverage_requires_sources_or_markers() -> None:
    response = AnswerResponse(
        question='q',
        answer='Answer [1]\n\nSources:\n[1] Paper, p. 1',
        citations=['Paper, p. 1'],
        retrieved=[_result('Paper')],
    )

    assert has_citation_coverage(response)


def test_unsupported_answer_detection() -> None:
    assert is_unsupported_answer('The local corpus does not contain enough evidence.')


def test_expected_term_count_is_case_insensitive() -> None:
    assert count_expected_terms('LC-MS/MS and extraction were used.', ['lc-ms', 'Extraction']) == 2


def test_render_markdown_report_contains_metrics() -> None:
    result = EvaluationRunResult(
        cases=[
            CaseEvaluationResult(
                case_id='methods',
                question='What methods are used?',
                retrieved=[_result('Paper')],
                recall_at_k=1.0,
                reciprocal_rank=1.0,
                citation_covered=True,
                unsupported_answer=False,
                expected_terms_found=1,
                expected_terms_total=1,
            )
        ],
        summary=EvaluationSummary(
            case_count=1,
            top_k=4,
            recall_at_k=1.0,
            mean_reciprocal_rank=1.0,
            citation_coverage=1.0,
            unsupported_answer_count=0,
            mean_expected_term_coverage=1.0,
        ),
    )

    markdown = render_markdown_report(
        result,
        settings=Settings(),
        eval_path=Path('data/eval/pfas_questions.yaml'),
        generate_answers=False,
    )

    assert 'Recall@4: 1.000' in markdown
    assert 'methods' in markdown
