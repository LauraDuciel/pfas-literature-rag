import typer
from rich.console import Console

from pfas_lit_rag.config import get_settings
from pfas_lit_rag.ingestion import build_index

app = typer.Typer(help="Build the local vector index from PDFs.")
console = Console()


@app.callback(invoke_without_command=True)
def main() -> None:
    settings = get_settings()
    chunks = build_index(settings)
    console.print(f"Indexed {len(chunks)} chunks in {settings.resolved_index_dir}")
