# PFAS Literature RAG

Local retrieval-augmented question answering for PFAS literature, analytical
chemistry, environmental science, NMR workflows, and technical regulatory
documents.

The workflow is designed to run on a personal machine:

- PDFs are stored and indexed locally.
- Embeddings are computed locally with a small CPU-friendly model.
- Answer synthesis uses Ollama with `qwen2.5:3b` by default.
- Retrieved passages keep document and page metadata for source citations.

## Setup after cloning

Install `uv` if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository, enter the project directory, and install dependencies:

```bash
git clone <repo-url>
cd <repo-directory>
uv sync --extra dev
```

Start Ollama locally. With Docker, one typical setup is:

```bash
docker start ollama
```

Check that Ollama is reachable and that `qwen2.5:3b` is available:

```bash
curl http://localhost:11434/api/tags
```

If the model is missing:

```bash
docker exec -it ollama ollama pull qwen2.5:3b
```

## Index local PDFs

Place PDFs in:

```text
data/raw_pdfs/
```

Build the local vector index:

```bash
uv run pfas-ingest
```

The FAISS index is written to `data/index/`. Raw PDFs and built indexes are
not tracked by git.

## Ask questions

Use the command line:

```bash
uv run pfas-query "Which extraction methods are discussed for PFAS analysis?"
```

Or start the API:

```bash
uv run pfas-api
```

Available endpoints:

- `GET /health`
- `POST /ingest`
- `POST /search`
- `POST /answer`

Example API request:

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What analytical methods are mentioned for PFAS detection?", "top_k": 5}'
```

## Collect open PDFs

The collector searches open scholarly metadata and downloads PDF links exposed
by those records. It writes a manifest to `data/metadata/literature_manifest.jsonl`.

Preview results without downloading:

```bash
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10 --dry-run
```

Download available PDFs:

```bash
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10
```

Then rebuild the index:

```bash
uv run pfas-ingest
```

## Configuration

Settings can be overridden with environment variables:

```bash
PFAS_RAG_OLLAMA_BASE_URL=http://localhost:11434
PFAS_RAG_OLLAMA_MODEL=qwen2.5:3b
PFAS_RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
PFAS_RAG_RETRIEVAL_K=5
```

The default embedding backend is `fastembed`, using
`BAAI/bge-small-en-v1.5`.

## Development

Run tests:

```bash
uv run --extra dev pytest
```

Run linting:

```bash
uv run --extra dev ruff check .
```

## Scope

The project supports local exploration of scientific PDFs. It does not replace
manual review of source documents, and answers should be checked against the
cited passages when used for technical decisions.
