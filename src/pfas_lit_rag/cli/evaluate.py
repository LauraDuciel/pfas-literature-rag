from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pfas_lit_rag.config import get_settings
from pfas_lit_rag.evaluation import (
    evaluate_cases,
    load_evaluation_cases,
    mlflow_metrics,
    mlflow_params,
    write_json_result,
    write_markdown_report,
)

console = Console()


def main(
    eval_file: Annotated[
        Path,
        typer.Option('--eval-file', help='YAML file containing evaluation questions.'),
    ] = Path('data/eval/pfas_questions.yaml'),
    top_k: Annotated[int | None, typer.Option('--top-k', min=1, max=20)] = None,
    retrieval_only: Annotated[
        bool,
        typer.Option('--retrieval-only', help='Evaluate retrieval metrics without calling Ollama.'),
    ] = False,
    report_path: Annotated[
        Path,
        typer.Option('--report-path', help='Markdown report output path.'),
    ] = Path('reports/evaluation_summary.md'),
    results_path: Annotated[
        Path,
        typer.Option('--results-path', help='JSON result output path.'),
    ] = Path('reports/evaluation_results.json'),
    experiment_name: Annotated[str, typer.Option('--experiment-name')] = 'pfas-lit-rag',
    run_name: Annotated[str | None, typer.Option('--run-name')] = None,
    log_mlflow: Annotated[bool, typer.Option('--mlflow/--no-mlflow')] = True,
) -> None:
    settings = get_settings()
    k = top_k or settings.retrieval_k
    cases = load_evaluation_cases(eval_file)
    result = evaluate_cases(
        cases,
        settings=settings,
        top_k=k,
        generate_answers=not retrieval_only,
    )

    write_json_result(result, results_path)
    write_markdown_report(
        result,
        report_path,
        settings=settings,
        eval_path=eval_file,
        generate_answers=not retrieval_only,
    )

    if log_mlflow:
        _log_mlflow_run(
            result=result,
            settings=settings,
            top_k=k,
            generate_answers=not retrieval_only,
            experiment_name=experiment_name,
            run_name=run_name,
            eval_file=eval_file,
            report_path=report_path,
            results_path=results_path,
        )

    summary = result.summary
    console.print(f'Evaluated {summary.case_count} cases')
    console.print(f'Recall@{summary.top_k}: {summary.recall_at_k:.3f}')
    console.print(f'MRR: {summary.mean_reciprocal_rank:.3f}')
    console.print(f'Citation coverage: {summary.citation_coverage:.3f}')
    console.print(f'Unsupported answers: {summary.unsupported_answer_count}')
    console.print(f'Report written to {report_path}')


def _log_mlflow_run(
    *,
    result,
    settings,
    top_k: int,
    generate_answers: bool,
    experiment_name: str,
    run_name: str | None,
    eval_file: Path,
    report_path: Path,
    results_path: Path,
) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError(
            'MLflow is not installed. Run `uv sync --extra eval` or pass `--no-mlflow`.'
        ) from exc

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(mlflow_params(settings, top_k=top_k, generate_answers=generate_answers))
        mlflow.log_metrics(mlflow_metrics(result.summary))
        mlflow.log_artifact(str(eval_file))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(results_path))


def run() -> None:
    typer.run(main)
