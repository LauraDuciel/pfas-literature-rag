from pathlib import Path

from pfas_lit_rag.adaptive import RetrievalMode
from pfas_lit_rag.adaptive_evaluation import (
    AdaptiveComparisonResult,
    AdaptiveEvaluationCase,
    ComparisonCaseResult,
    SystemResult,
    load_adaptive_cases,
    render_comparison_markdown,
)


def test_load_adaptive_cases_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "adaptive.yaml"
    path.write_text(
        """- id: exact
  question: How is LC-MS/MS used?
  category: exact_term
  expected_strategy: hybrid
  expected_answer_terms:
    - LC-MS/MS
""",
        encoding="utf-8",
    )

    cases = load_adaptive_cases(path)

    assert cases[0].id == "exact"
    assert cases[0].expected_strategy == RetrievalMode.HYBRID


def _system(system: str, selected_strategy: str, searches_run: int) -> SystemResult:
    return SystemResult(
        system=system,
        selected_strategy=selected_strategy,
        retrieved=[],
        recall_at_k=0.5,
        reciprocal_rank=1.0,
        citation_covered=False,
        unsupported_answer=False,
        expected_terms_found=0,
        expected_terms_total=1,
        searches_run=searches_run,
        retrieval_seconds=0.1,
        generation_seconds=0.0,
        total_seconds=0.1,
    )


def test_render_comparison_markdown_contains_system_summary() -> None:
    result = AdaptiveComparisonResult(
        top_k=4,
        generate_answers=False,
        summaries=[],
        cases=[
            ComparisonCaseResult(
                case_id="case1",
                question="Question?",
                category="exact_term",
                expected_strategy="hybrid",
                systems=[_system("rag_adaptive", "hybrid", 1)],
            )
        ],
    )

    markdown = render_comparison_markdown(result, eval_path=Path("cases.yaml"))

    assert "Adaptive Retrieval Comparison" in markdown
    assert "case1" in markdown
    assert "rag_adaptive" in markdown


def test_adaptive_evaluation_case_defaults() -> None:
    case = AdaptiveEvaluationCase(id="q", question="What are PFAS?", category="general_known")

    assert case.expected_sources == []
    assert case.expected_answer_terms == []
