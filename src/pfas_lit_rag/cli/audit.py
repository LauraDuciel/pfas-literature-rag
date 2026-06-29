from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pfas_lit_rag.audit import audit_answer, write_audit_html, write_audit_json
from pfas_lit_rag.config import get_settings

console = Console()


def main(
    question: Annotated[str, typer.Argument(help="Question that produced the answer.")],
    answer: Annotated[
        str | None,
        typer.Option("--answer", help="Answer text to audit."),
    ] = None,
    answer_file: Annotated[
        Path | None,
        typer.Option("--answer-file", help="Text file containing the answer to audit."),
    ] = None,
    top_k: Annotated[int, typer.Option("--top-k", min=1, max=10)] = 3,
    json_output: Annotated[
        Path,
        typer.Option("--json-output", help="JSON audit output path."),
    ] = Path("reports/audit_result.json"),
    html_output: Annotated[
        Path,
        typer.Option("--html-output", help="HTML audit report path."),
    ] = Path("reports/audit_report.html"),
) -> None:
    answer_text = _read_answer(answer=answer, answer_file=answer_file)
    report = audit_answer(question, answer_text, settings=get_settings(), top_k=top_k)
    write_audit_json(report, json_output)
    write_audit_html(report, html_output)
    console.print(
        f"Audited {len(report.claims)} claims; "
        f"{report.review_count} need review."
    )
    console.print(f"JSON written to {json_output}")
    console.print(f"HTML written to {html_output}")


def _read_answer(*, answer: str | None, answer_file: Path | None) -> str:
    if answer and answer_file:
        raise typer.BadParameter("Use either --answer or --answer-file, not both.")
    if answer_file:
        return answer_file.read_text(encoding="utf-8")
    if answer:
        return answer
    raise typer.BadParameter("Provide --answer or --answer-file.")


def run() -> None:
    typer.run(main)
