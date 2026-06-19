# RAG Evaluation Summary

- Evaluation set: `data/eval/pfas_questions.yaml`
- Cases: 3
- Mode: retrieval + answer generation
- Ollama model: `qwen2.5:3b`
- Embedding model: `BAAI/bge-small-en-v1.5`
- top_k: 2
- Vector/BM25 weights: 0.65 / 0.35

## Metrics

- Recall@2: 0.667
- Mean reciprocal rank: 1.000
- Citation coverage: 1.000
- Unsupported answer count: 0
- Expected term coverage: 1.000

## Cases

### pfas_detection_methods

Question: What analytical methods are used for PFAS detection?

- Recall@2: 0.500
- Reciprocal rank: 1.000
- Citation covered: True
- Unsupported answer: False

Retrieved sources:
- [1] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 6 (score=0.987)
- [2] Evaluation of extraction methodologies for PFAS analysis in mascara: a comparative study of SPME and automated µSPE, p. 1 (score=0.566)

### pfas_extraction_methods

Question: Which extraction approaches are discussed for PFAS analysis?

- Recall@2: 0.500
- Reciprocal rank: 1.000
- Citation covered: True
- Unsupported answer: False

Retrieved sources:
- [1] Evaluation of extraction methodologies for PFAS analysis in mascara: a comparative study of SPME and automated µSPE, p. 5 (score=0.673)
- [2] Evaluation of extraction methodologies for PFAS analysis in mascara: a comparative study of SPME and automated µSPE, p. 1 (score=0.639)

### pfas_non_target_screening

Question: How is suspect or non-target screening used in PFAS studies?

- Recall@2: 1.000
- Reciprocal rank: 1.000
- Citation covered: True
- Unsupported answer: False

Retrieved sources:
- [1] The NORMAN Suspect List Exchange (NORMAN-SLE): facilitating European and worldwide collaboration on suspect screening in high resolution mass spectrometry, p. 2 (score=0.918)
- [2] Current progress in the environmental analysis of poly- and perfluoroalkyl substances (PFAS), p. 12 (score=0.532)
