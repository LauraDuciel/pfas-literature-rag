# RAG Evaluation

The evaluation workflow checks whether the local RAG index retrieves expected
sources for a small set of scientific questions. It can also call the local
Ollama model to measure citation coverage in generated answers.

Install the evaluation extra when MLflow tracking is needed:

```bash
uv sync --extra dev --extra eval
```

Run retrieval-only evaluation:

```bash
uv run pfas-evaluate --retrieval-only
```

Run retrieval plus answer generation:

```bash
uv run pfas-evaluate
```

Outputs:

- `reports/evaluation_summary.md`: human-readable summary
- `reports/evaluation_results.json`: detailed per-question results, ignored by git
- `mlruns/`: local MLflow runs, ignored by git

The default evaluation set is `data/eval/pfas_questions.yaml`. Source matching is
intentionally flexible and uses `title_contains` so the same question set remains
usable as the local PDF corpus grows.

Current metrics:

- recall@k over expected source matches
- mean reciprocal rank
- citation coverage, meaningful when answer generation is enabled
- unsupported answer count
- expected answer term coverage

These metrics are lightweight checks of retrieval and citation behavior. They are
not a substitute for scientific review of the cited documents.

## Adaptive comparison

The adaptive comparison workflow uses `data/eval/adaptive_questions.yaml` to
compare three local modes: no retrieval, fixed hybrid retrieval, and adaptive
retrieval.

```bash
uv run pfas-compare --retrieval-only --no-mlflow
```

The report is written to `reports/adaptive_comparison.md`. Detailed JSON output
is ignored by git. Use the full command without `--retrieval-only` when answer
generation metrics are needed and local runtime is acceptable.
