from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pfas_lit_rag.adaptive_evaluation import (
    compare_systems,
    load_adaptive_cases,
    mlflow_comparison_metrics,
    mlflow_comparison_params,
    write_comparison_json,
    write_comparison_markdown,
)
from pfas_lit_rag.config import get_settings

console = Console()


def main(
    eval_file: Annotated[
        Path,
        typer.Option("--eval-file", help="YAML file containing adaptive evaluation cases."),
    ] = Path("data/eval/adaptive_questions.yaml"),
    retrieval_only: Annotated[
        bool,
        typer.Option("--retrieval-only", help="Compare retrieval behavior without calling Ollama."),
    ] = False,
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=20)] = 4,
    report_path: Annotated[
        Path,
        typer.Option("--report", help="Markdown comparison report path."),
    ] = Path("reports/adaptive_comparison.md"),
    json_path: Annotated[
        Path,
        typer.Option("--json-output", help="Detailed JSON output path."),
    ] = Path("reports/adaptive_comparison.json"),
    no_mlflow: Annotated[
        bool,
        typer.Option("--no-mlflow", help="Skip local MLflow logging."),
    ] = False,
    run_name: Annotated[
        str,
        typer.Option("--run-name", help="MLflow run name."),
    ] = "adaptive-comparison",
) -> None:
    settings = get_settings()
    cases = load_adaptive_cases(eval_file)
    generate_answers = not retrieval_only
    result = compare_systems(
        cases,
        settings=settings,
        top_k=top_k,
        generate_answers=generate_answers,
    )
    write_comparison_json(result, json_path)
    write_comparison_markdown(result, report_path, eval_path=eval_file)

    if not no_mlflow:
        _log_mlflow(
            result,
            settings=settings,
            top_k=top_k,
            generate_answers=generate_answers,
            run_name=run_name,
        )

    console.print(f"Compared {len(result.cases)} cases")
    for summary in result.summaries:
        console.print(
            f"{summary.system}: recall={summary.mean_recall_at_k:.3f}, "
            f"MRR={summary.mean_reciprocal_rank:.3f}, "
            f"searches={summary.mean_searches_run:.2f}"
        )
    console.print(f"Report written to {report_path}")


def _log_mlflow(result, *, settings, top_k: int, generate_answers: bool, run_name: str) -> None:
    try:
        import mlflow
    except ImportError as exc:
        raise typer.BadParameter(
            "MLflow is not installed. Run `uv sync --extra eval` or pass `--no-mlflow`."
        ) from exc

    mlflow.set_experiment("pfas-lit-rag-adaptive")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            mlflow_comparison_params(settings, top_k=top_k, generate_answers=generate_answers)
        )
        mlflow.log_metrics(mlflow_comparison_metrics(result))


def run() -> None:
    typer.run(main)
