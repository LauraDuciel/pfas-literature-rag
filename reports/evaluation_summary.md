# RAG Evaluation Summary

- Evaluation set: `data/eval/pfas_questions.yaml`
- Cases: 3
- Mode: retrieval + answer generation
- Ollama model: `qwen2.5:3b`
- Embedding model: `BAAI/bge-small-en-v1.5`
- top_k: 4
- Vector/BM25 weights: 0.65 / 0.35
- Reranking: True (weight=0.25)

## Metrics

- Recall@4: 0.667
- Mean reciprocal rank: 1.000
- Citation coverage: 1.000
- Unsupported answer count: 0
- Expected term coverage: 1.000

## Cases

### pfas_detection_methods

Question: What analytical methods are used for PFAS detection?

- Recall@4: 0.500
- Reciprocal rank: 1.000
- Citation covered: True
- Unsupported answer: False

Retrieved sources:
- [1] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 6 (score=0.969)
- [2] Evaluation of extraction methodologies for PFAS analysis in mascara: a comparative study of SPME and automated µSPE, p. 1 (score=0.602)
- [3] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 13 (score=0.580)
- [4] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 6 (score=0.471)

### pfas_extraction_methods

Question: Which extraction approaches are discussed for PFAS analysis?

- Recall@4: 0.500
- Reciprocal rank: 1.000
- Citation covered: True
- Unsupported answer: False

Retrieved sources:
- [1] Evaluation of extraction methodologies for PFAS analysis in mascara: a comparative study of SPME and automated µSPE, p. 5 (score=0.906)
- [2] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 6 (score=0.902)
- [3] Evaluation of extraction methodologies for PFAS analysis in mascara: a comparative study of SPME and automated µSPE, p. 1 (score=0.891)
- [4] Less is more: a methodological assessment of extraction techniques for per- and polyfluoroalkyl substances (PFAS) analysis in mammalian tissues, p. 1 (score=0.798)

### pfas_non_target_screening

Question: How is suspect or non-target screening used in PFAS studies?

- Recall@4: 1.000
- Reciprocal rank: 1.000
- Citation covered: True
- Unsupported answer: False

Retrieved sources:
- [1] The NORMAN Suspect List Exchange (NORMAN-SLE): facilitating European and worldwide collaboration on suspect screening in high resolution mass spectrometry, p. 2 (score=0.950)
- [2] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 12 (score=0.592)
- [3] 10.1016_j.hazl.2022.100067, p. 3 (score=0.509)
- [4] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 13 (score=0.461)
