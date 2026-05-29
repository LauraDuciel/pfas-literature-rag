import typer
from rich.console import Console

from pfas_lit_rag.answering import answer_question
from pfas_lit_rag.config import get_settings

app = typer.Typer(help="Ask a question against the local literature index.")
console = Console()


@app.callback(invoke_without_command=True)
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
