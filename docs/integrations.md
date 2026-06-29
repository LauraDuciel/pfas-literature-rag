# Optional Integrations

The baseline project stays local and explicit: PDF ingestion, local embeddings,
FAISS/BM25 retrieval, lightweight reranking, and Ollama answer generation. The
optional integrations below are intended as narrow adapters, not a different
architecture.

## LangChain retriever adapter

Install the optional dependency only when a LangChain-compatible retriever is
useful for experiments:

```bash
uv sync --extra langchain
```

Example:

```python
from pfas_lit_rag.langchain_adapter import make_langchain_retriever

retriever = make_langchain_retriever(top_k=5)
documents = retriever.invoke("PFAS extraction methods")
```

The adapter calls the existing `search_index` function and returns LangChain
`Document` objects with the same citation and page metadata used by the rest of
the project. It does not add chains or a second retrieval stack.

## Cross-encoder reranking

The retrieval pipeline can use a local cross-encoder reranker after the initial
FAISS/BM25 candidate search. The default configuration uses `lexical`, which
keeps the baseline fast and fully reproducible on small machines.

Relevant settings:

```bash
PFAS_RAG_RERANK_BACKEND=lexical
PFAS_RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
PFAS_RAG_CROSS_ENCODER_CANDIDATE_K=20
PFAS_RAG_CROSS_ENCODER_BATCH_SIZE=8
```

Use `PFAS_RAG_RERANK_BACKEND=cross_encoder` only after installing
`sentence-transformers` in the local environment. On small machines, prefer a
CPU-only PyTorch installation and keep the candidate count modest. The model
`cross-encoder/ms-marco-MiniLM-L-6-v2` is generic rather than chemistry-specific,
so its value should be checked with `pfas-evaluate` before relying on it.

## Hugging Face candidates

Hugging Face models are a reasonable next extension point, especially for local
reranking or alternative embeddings. The most useful path would be to keep them
optional and compare them through the existing `pfas-evaluate` workflow before
making any default change.

Possible experiments:

- test a compact cross-encoder reranker on the top 20 retrieved chunks;
- compare a scientific embedding model against the current `fastembed` baseline;
- record latency and retrieval metrics in MLflow to decide whether the quality
  gain is worth the local hardware cost.

For now, the default stack avoids the heavier PyTorch/Hugging Face dependency so
the project remains practical on a small personal machine.
