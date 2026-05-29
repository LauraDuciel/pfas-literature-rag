import typer
from rich.console import Console

from pfas_lit_rag.config import get_settings
from pfas_lit_rag.ingestion import build_index

console = Console()


def main() -> None:
    settings = get_settings()
    chunks = build_index(settings)
    console.print(f"Indexed {len(chunks)} chunks in {settings.resolved_index_dir}")


def run() -> None:
    typer.run(main)
