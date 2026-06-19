from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from pfas_lit_rag.config import get_settings
from pfas_lit_rag.corpus_report import write_corpus_report

console = Console()


def main(
    output: Annotated[
        Path,
        typer.Option('--output', help='Markdown report output path.'),
    ] = Path('reports/corpus_report.md'),
) -> None:
    write_corpus_report(get_settings(), output)
    console.print(f'Corpus report written to {output}')


def run() -> None:
    typer.run(main)
