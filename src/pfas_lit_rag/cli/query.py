import typer
from rich.console import Console

from pfas_lit_rag.answering import answer_question
from pfas_lit_rag.config import get_settings

console = Console()


def main(
    question: str = typer.Argument(..., help="Question to ask the local corpus."),
    top_k: int | None = typer.Option(None, "--top-k", min=1, max=20),
) -> None:
    response = answer_question(question, settings=get_settings(), top_k=top_k)
    console.print(response.answer)
    if response.citations:
        console.print("\nCitations:")
        for citation in response.citations:
            console.print(f"- {citation}")


def run() -> None:
    typer.run(main)
