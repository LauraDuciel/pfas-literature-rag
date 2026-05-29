# PFAS Literature RAG

Lightweight local retrieval-augmented generation assistant for PFAS literature,
analytical chemistry, environmental science, NMR workflows, and technical
regulatory documents.

The project is intentionally local-first:

- PDFs are stored and indexed locally.
- Embeddings are generated on the local machine with a small CPU-friendly model.
- Answer generation uses a local Ollama model, with `qwen2.5:0.5b` as the default.
- Retrieved passages keep document and page metadata so answers can cite sources.

## Quick start

Install dependencies:

```bash
uv sync --extra dev
```

Start or expose Ollama locally, then check the loaded models:

```bash
curl http://localhost:11434/api/tags
```

Add PDFs to `data/raw_pdfs/`, then build the index:

```bash
uv run pfas-ingest
```

Ask a local question:

```bash
uv run pfas-query "Which PFAS extraction methods are discussed?"
```

Run the API:

```bash
uv run pfas-api
```

## Literature collection

The collector searches open scholarly metadata sources and downloads only open
PDF links exposed by those services. It writes a manifest with source metadata
for traceability.

```bash
uv run pfas-collect "PFAS NMR analytical chemistry" --max-results 10
```

## Local models

The default generator is the lightest local Qwen model already loaded in Ollama:

```bash
PFAS_RAG_OLLAMA_MODEL=qwen2.5:0.5b
```

The default embedding model is `BAAI/bge-small-en-v1.5` through `fastembed`.
It runs locally and avoids a large PyTorch/CUDA dependency stack.

## Notes

This is a personal scientific engineering project. It is built for practical
local literature exploration, not for automated regulatory conclusions.
