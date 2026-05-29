import typer
from rich.console import Console

from pfas_lit_rag.config import get_settings
from pfas_lit_rag.literature_search import LiteratureSearchClient
from pfas_lit_rag.pdf_downloader import download_records

console = Console()


def main(
    query: str = typer.Argument(..., help="Literature search query."),
    max_results: int = typer.Option(10, "--max-results", min=1, max=200),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show records without downloading."),
) -> None:
    settings = get_settings()
    client = LiteratureSearchClient(settings)
    records = client.search_openalex(query=query, max_results=max_results)

    if dry_run:
        for record in records:
            console.print(f"- {record.title} ({record.publication_year})")
            console.print(f"  {record.pdf_url}")
        return

    result = download_records(records, settings)
    console.print(
        f"Downloaded {len(result.downloaded)} new PDFs "
        f"({len(result.existing)} already present) to {settings.resolved_raw_pdf_dir}"
    )
    if result.failed:
        console.print(f"Skipped {len(result.failed)} PDFs that could not be downloaded:")
        for failure in result.failed:
            console.print(f"- {failure.title}")
            console.print(f"  {failure.url}")
            console.print(f"  {failure.reason}")


def run() -> None:
    typer.run(main)
