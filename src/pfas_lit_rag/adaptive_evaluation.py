import json
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pfas_lit_rag.adaptive import RetrievalMode, answer_adaptive, retrieve_adaptive
from pfas_lit_rag.answering import answer_question
from pfas_lit_rag.config import Settings
from pfas_lit_rag.evaluation import (
    ExpectedSource,
    count_expected_terms,
    has_citation_coverage,
    is_unsupported_answer,
    recall_at_k,
    reciprocal_rank,
)
from pfas_lit_rag.ollama_client import OllamaClient
from pfas_lit_rag.retrieval import search_index
from pfas_lit_rag.schemas import AnswerResponse, SearchResult

LLM_ONLY_SYSTEM = "llm_only"
RAG_FIXED_SYSTEM = "rag_fixed"
RAG_ADAPTIVE_SYSTEM = "rag_adaptive"
SYSTEMS = (LLM_ONLY_SYSTEM, RAG_FIXED_SYSTEM, RAG_ADAPTIVE_SYSTEM)


class AdaptiveEvaluationCase(BaseModel):
    id: str
    question: str
    category: str
    expected_strategy: RetrievalMode | None = None
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)


class SystemResult(BaseModel):
    system: str
    selected_strategy: str
    retrieved: list[SearchResult] = Field(default_factory=list)
    answer: str | None = None
    recall_at_k: float
    reciprocal_rank: float
    citation_covered: bool
    unsupported_answer: bool
    expected_terms_found: int
    expected_terms_total: int
    searches_run: int
    retrieval_seconds: float
    generation_seconds: float
    total_seconds: float


class ComparisonCaseResult(BaseModel):
    case_id: str
    question: str
    category: str
    expected_strategy: str | None
    systems: list[SystemResult]


class SystemSummary(BaseModel):
    system: str
    case_count: int
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    citation_coverage: float
    unsupported_answer_count: int
    mean_expected_term_coverage: float
    mean_searches_run: float
    mean_total_seconds: float
    unnecessary_search_rate: float
    strategy_match_rate: float | None = None


class AdaptiveComparisonResult(BaseModel):
    cases: list[ComparisonCaseResult]
    summaries: list[SystemSummary]
    top_k: int
    generate_answers: bool


def load_adaptive_cases(path: Path) -> list[AdaptiveEvaluationCase]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, list):
        raise ValueError("Adaptive evaluation file must contain a list of cases")
    return [AdaptiveEvaluationCase.model_validate(item) for item in payload]


def compare_systems(
    cases: list[AdaptiveEvaluationCase],
    *,
    settings: Settings,
    top_k: int,
    generate_answers: bool,
) -> AdaptiveComparisonResult:
    case_results = [
        _compare_case(case, settings=settings, top_k=top_k, generate_answers=generate_answers)
        for case in cases
    ]
    return AdaptiveComparisonResult(
        cases=case_results,
        summaries=_summarise(case_results),
        top_k=top_k,
        generate_answers=generate_answers,
    )


def _compare_case(
    case: AdaptiveEvaluationCase,
    *,
    settings: Settings,
    top_k: int,
    generate_answers: bool,
) -> ComparisonCaseResult:
    return ComparisonCaseResult(
        case_id=case.id,
        question=case.question,
        category=case.category,
        expected_strategy=case.expected_strategy.value if case.expected_strategy else None,
        systems=[
            _run_llm_only(case, settings=settings, generate_answers=generate_answers),
            _run_fixed_rag(case, settings=settings, top_k=top_k, generate_answers=generate_answers),
            _run_adaptive_rag(
                case,
                settings=settings,
                top_k=top_k,
                generate_answers=generate_answers,
            ),
        ],
    )


def _run_llm_only(
    case: AdaptiveEvaluationCase,
    *,
    settings: Settings,
    generate_answers: bool,
) -> SystemResult:
    started = time.perf_counter()
    generation_seconds = 0.0
    answer: str | None = None
    if generate_answers:
        generation_started = time.perf_counter()
        answer = OllamaClient(settings).answer_without_context(case.question)
        generation_seconds = time.perf_counter() - generation_started
    return _system_result(
        case,
        system=LLM_ONLY_SYSTEM,
        selected_strategy=RetrievalMode.LLM_ONLY.value,
        retrieved=[],
        answer=answer,
        searches_run=0,
        retrieval_seconds=0.0,
        generation_seconds=generation_seconds,
        total_seconds=time.perf_counter() - started,
    )


def _run_fixed_rag(
    case: AdaptiveEvaluationCase,
    *,
    settings: Settings,
    top_k: int,
    generate_answers: bool,
) -> SystemResult:
    started = time.perf_counter()
    if generate_answers:
        response = answer_question(case.question, settings=settings, top_k=top_k)
        total_seconds = time.perf_counter() - started
        return _system_result(
            case,
            system=RAG_FIXED_SYSTEM,
            selected_strategy="hybrid",
            retrieved=response.retrieved,
            answer=response.answer,
            searches_run=1,
            retrieval_seconds=0.0,
            generation_seconds=total_seconds,
            total_seconds=total_seconds,
        )

    retrieval_started = time.perf_counter()
    retrieved = search_index(case.question, settings=settings, top_k=top_k)
    retrieval_seconds = time.perf_counter() - retrieval_started
    return _system_result(
        case,
        system=RAG_FIXED_SYSTEM,
        selected_strategy="hybrid",
        retrieved=retrieved,
        answer=None,
        searches_run=1,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=0.0,
        total_seconds=time.perf_counter() - started,
    )


def _run_adaptive_rag(
    case: AdaptiveEvaluationCase,
    *,
    settings: Settings,
    top_k: int,
    generate_answers: bool,
) -> SystemResult:
    started = time.perf_counter()
    if generate_answers:
        adaptive = answer_adaptive(case.question, settings=settings, top_k=top_k)
        return _system_result(
            case,
            system=RAG_ADAPTIVE_SYSTEM,
            selected_strategy=adaptive.decision.mode.value,
            retrieved=adaptive.response.retrieved,
            answer=adaptive.response.answer,
            searches_run=adaptive.searches_run,
            retrieval_seconds=adaptive.retrieval_seconds,
            generation_seconds=adaptive.generation_seconds,
            total_seconds=adaptive.total_seconds,
        )

    decision, retrieved, searches_run, retrieval_seconds = retrieve_adaptive(
        case.question,
        settings=settings,
        top_k=top_k,
    )
    return _system_result(
        case,
        system=RAG_ADAPTIVE_SYSTEM,
        selected_strategy=decision.mode.value,
        retrieved=retrieved,
        answer=None,
        searches_run=searches_run,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=0.0,
        total_seconds=time.perf_counter() - started,
    )


def _system_result(
    case: AdaptiveEvaluationCase,
    *,
    system: str,
    selected_strategy: str,
    retrieved: list[SearchResult],
    answer: str | None,
    searches_run: int,
    retrieval_seconds: float,
    generation_seconds: float,
    total_seconds: float,
) -> SystemResult:
    response = AnswerResponse(
        question=case.question,
        answer=answer or "",
        citations=[result.citation for result in retrieved],
        retrieved=retrieved,
    )
    terms_found = count_expected_terms(answer or "", case.expected_answer_terms)
    return SystemResult(
        system=system,
        selected_strategy=selected_strategy,
        retrieved=retrieved,
        answer=answer,
        recall_at_k=recall_at_k(retrieved, case.expected_sources, top_k=len(retrieved)),
        reciprocal_rank=reciprocal_rank(retrieved, case.expected_sources),
        citation_covered=has_citation_coverage(response) if answer else False,
        unsupported_answer=is_unsupported_answer(answer or ""),
        expected_terms_found=terms_found,
        expected_terms_total=len(case.expected_answer_terms),
        searches_run=searches_run,
        retrieval_seconds=retrieval_seconds,
        generation_seconds=generation_seconds,
        total_seconds=total_seconds,
    )


def _summarise(case_results: list[ComparisonCaseResult]) -> list[SystemSummary]:
    summaries = []
    for system in SYSTEMS:
        results = [
            result
            for case_result in case_results
            for result in case_result.systems
            if result.system == system
        ]
        summaries.append(_summarise_system(system, case_results, results))
    return summaries


def _summarise_system(
    system: str,
    case_results: list[ComparisonCaseResult],
    results: list[SystemResult],
) -> SystemSummary:
    if not results:
        return SystemSummary(
            system=system,
            case_count=0,
            mean_recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
            citation_coverage=0.0,
            unsupported_answer_count=0,
            mean_expected_term_coverage=0.0,
            mean_searches_run=0.0,
            mean_total_seconds=0.0,
            unnecessary_search_rate=0.0,
        )
    expected_strategy_cases = {
        case_result.case_id: case_result.expected_strategy for case_result in case_results
    }
    strategy_matches = [
        result.selected_strategy == expected_strategy_cases.get(case_result.case_id)
        for case_result in case_results
        for result in case_result.systems
        if result.system == system and expected_strategy_cases.get(case_result.case_id)
    ]
    term_coverages = [
        (
            1.0
            if result.expected_terms_total == 0
            else result.expected_terms_found / result.expected_terms_total
        )
        for result in results
    ]
    unnecessary_searches = sum(
        1
        for case_result in case_results
        for result in case_result.systems
        if result.system == system
        and case_result.expected_strategy == RetrievalMode.LLM_ONLY.value
        and result.searches_run > 0
    )
    return SystemSummary(
        system=system,
        case_count=len(results),
        mean_recall_at_k=sum(result.recall_at_k for result in results) / len(results),
        mean_reciprocal_rank=sum(result.reciprocal_rank for result in results) / len(results),
        citation_coverage=sum(1 for result in results if result.citation_covered) / len(results),
        unsupported_answer_count=sum(1 for result in results if result.unsupported_answer),
        mean_expected_term_coverage=sum(term_coverages) / len(term_coverages),
        mean_searches_run=sum(result.searches_run for result in results) / len(results),
        mean_total_seconds=sum(result.total_seconds for result in results) / len(results),
        unnecessary_search_rate=unnecessary_searches / len(results),
        strategy_match_rate=(
            (sum(strategy_matches) / len(strategy_matches)) if strategy_matches else None
        ),
    )


def write_comparison_json(result: AdaptiveComparisonResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result.model_dump(mode="json"), handle, indent=2, ensure_ascii=False)


def write_comparison_markdown(
    result: AdaptiveComparisonResult,
    path: Path,
    *,
    eval_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_comparison_markdown(result, eval_path=eval_path), encoding="utf-8")


def render_comparison_markdown(result: AdaptiveComparisonResult, *, eval_path: Path) -> str:
    lines = [
        "# Adaptive Retrieval Comparison",
        "",
        f"- Evaluation set: `{eval_path}`",
        f"- Cases: {len(result.cases)}",
        (
            "- Mode: "
            f"{'retrieval + answer generation' if result.generate_answers else 'retrieval only'}"
        ),
        f"- top_k: {result.top_k}",
        "",
        "## System Summary",
        "",
        (
            "| System | Recall | MRR | Citations | Unsupported | Searches | "
            "Time (s) | Strategy match |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in result.summaries:
        strategy_match = (
            "n/a"
            if summary.strategy_match_rate is None
            else f"{summary.strategy_match_rate:.3f}"
        )
        lines.append(
            "| "
            f"{summary.system} | {summary.mean_recall_at_k:.3f} | "
            f"{summary.mean_reciprocal_rank:.3f} | {summary.citation_coverage:.3f} | "
            f"{summary.unsupported_answer_count} | {summary.mean_searches_run:.2f} | "
            f"{summary.mean_total_seconds:.2f} | {strategy_match} |"
        )
    lines.extend(["", "## Cases", ""])
    for case_result in result.cases:
        lines.extend([
            f"### {case_result.case_id}",
            "",
            f"Question: {case_result.question}",
            f"Category: `{case_result.category}`",
            f"Expected strategy: `{case_result.expected_strategy or 'n/a'}`",
            "",
        ])
        for system_result in case_result.systems:
            lines.append(
                f"- {system_result.system}: strategy={system_result.selected_strategy}, "
                f"recall={system_result.recall_at_k:.3f}, "
                f"mrr={system_result.reciprocal_rank:.3f}, "
                f"searches={system_result.searches_run}, "
                f"time={system_result.total_seconds:.2f}s"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def mlflow_comparison_metrics(result: AdaptiveComparisonResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for summary in result.summaries:
        prefix = summary.system
        metrics[f"{prefix}_recall"] = summary.mean_recall_at_k
        metrics[f"{prefix}_mrr"] = summary.mean_reciprocal_rank
        metrics[f"{prefix}_citation_coverage"] = summary.citation_coverage
        metrics[f"{prefix}_mean_searches"] = summary.mean_searches_run
        metrics[f"{prefix}_mean_seconds"] = summary.mean_total_seconds
        metrics[f"{prefix}_unnecessary_search_rate"] = summary.unnecessary_search_rate
        if summary.strategy_match_rate is not None:
            metrics[f"{prefix}_strategy_match_rate"] = summary.strategy_match_rate
    return metrics


def mlflow_comparison_params(
    settings: Settings,
    *,
    top_k: int,
    generate_answers: bool,
) -> dict[str, Any]:
    return {
        "ollama_model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
        "top_k": top_k,
        "generate_answers": generate_answers,
        "rerank_backend": settings.rerank_backend,
        "rerank_weight": settings.rerank_weight,
    }
