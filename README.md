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

## Typical workflow

For a first local run:

```bash
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10 --dry-run
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10
uv run pfas-ingest
uv run pfas-query "What analytical methods are used for PFAS detection?"
```

`pfas-query` uses the local Ollama model, so response time depends on the
machine and the selected model.

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
not tracked by git. Re-running the command appends only new chunks; chunks that
are already present in the index are skipped.

## Ask questions

Use the command line:

```bash
uv run pfas-query "Which extraction methods are discussed for PFAS analysis?"
```

Or start the API:

```bash
uv run pfas-api
```

Keep that process running while sending `curl` requests from another terminal.

Available endpoints:

- `GET /health`
- `POST /ingest`
- `POST /search`
- `POST /answer`

Example search request:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "PFAS extraction methods", "top_k": 5}'
```

Example answer request:

```bash
curl -X POST http://127.0.0.1:8000/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "What analytical methods are mentioned for PFAS detection?", "top_k": 5}'
```

## Collect open PDFs

The collector searches open scholarly metadata and downloads PDF links exposed
by those records. It writes a manifest to `data/metadata/literature_manifest.jsonl`.
If a matching PDF file already exists locally, it is counted as already present
and is not downloaded again.

Preview results without downloading:

```bash
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10 --dry-run
```

Download available PDFs:

```bash
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10
```

Then update the index:

```bash
uv run pfas-ingest
```

Only new chunks are embedded and added to the existing index.

## Configuration

Settings can be overridden with environment variables:

```bash
PFAS_RAG_OLLAMA_BASE_URL=http://localhost:11434
PFAS_RAG_OLLAMA_MODEL=qwen2.5:3b
PFAS_RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
PFAS_RAG_RETRIEVAL_K=4
PFAS_RAG_CONTEXT_CHARS_PER_CHUNK=1200
PFAS_RAG_OLLAMA_NUM_PREDICT=350
PFAS_RAG_REQUEST_TIMEOUT_SECONDS=900
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

## Limitations

This workflow is intentionally tuned for local use on a personal machine. That
keeps the setup simple and private, but it also means there are practical limits:

- `qwen2.5:3b` can be slow on CPU-only hardware, especially when several long
  passages are sent to Ollama.
- Small local language models may miss nuance, over-compress details, or produce
  awkward summaries. Citations help with checking, but they do not make the
  generated answer automatically correct.
- PDF text extraction depends on the PDF structure. Scanned PDFs, tables,
  two-column layouts, chemical notation, and supplementary material may extract
  poorly without OCR or layout-aware parsing.
- Retrieval is based on dense embeddings and simple top-k ranking. It does not
  yet use hybrid lexical/vector search, reranking, or query expansion.
- The OpenAlex collector depends on available open metadata and PDF links. Some
  links are stale, blocked, not actual PDFs, or only weakly related to the query.
- Deduplication is based on deterministic local filenames and chunk ids. It will
  catch common repeated downloads, but it is not a full DOI/content-hash
  deduplication system yet.
- The index is local and single-user. There is no authentication, background job
  queue, or concurrent write protection.

## Possible improvements

Useful next steps, without changing the project into a heavy platform, would be:

- add DOI and file-content hashing for stronger duplicate detection;
- add OCR for scanned PDFs and better handling of tables or two-column layouts;
- add hybrid retrieval with BM25 plus vector search;
- add a lightweight reranker for the top retrieved passages;
- store document-level metadata more explicitly, including DOI, source, license,
  and collection query;
- add an evaluation notebook with a small set of fixed questions and expected
  cited passages;
- expose separate retrieval-only and answer-generation timings in the CLI/API;
- support swapping between `qwen2.5:3b` and a larger local model when hardware
  allows it.

## Scope

The project supports local exploration of scientific PDFs. It does not replace
manual review of source documents, and answers should be checked against the
cited passages when used for technical decisions.
