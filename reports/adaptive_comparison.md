# Adaptive Retrieval Comparison

- Evaluation set: `data/eval/adaptive_questions.yaml`
- Cases: 5
- Mode: retrieval only
- top_k: 4

## System Summary

| System | Recall | MRR | Citations | Unsupported | Searches | Time (s) | Strategy match |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llm_only | 0.400 | 0.400 | 0.000 | 0 | 0.00 | 0.00 | 0.200 |
| rag_fixed | 0.900 | 1.000 | 0.000 | 0 | 1.00 | 0.26 | 0.600 |
| rag_adaptive | 1.000 | 1.000 | 0.000 | 0 | 1.00 | 0.16 | 1.000 |

## Cases

### pfas_general_background

Question: What are PFAS?
Category: `general_known`
Expected strategy: `llm_only`

- llm_only: strategy=llm_only, recall=1.000, mrr=1.000, searches=0, time=0.00s
- rag_fixed: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.45s
- rag_adaptive: strategy=llm_only, recall=1.000, mrr=1.000, searches=0, time=0.00s

### pfas_detection_exact_method

Question: How is LC-MS/MS used for PFAS detection?
Category: `exact_term`
Expected strategy: `hybrid`

- llm_only: strategy=llm_only, recall=0.000, mrr=0.000, searches=0, time=0.00s
- rag_fixed: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.23s
- rag_adaptive: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.20s

### pfas_extraction_multi_document

Question: Compare extraction methods across PFAS analytical studies.
Category: `multi_document`
Expected strategy: `hybrid_retry`

- llm_only: strategy=llm_only, recall=0.000, mrr=0.000, searches=0, time=0.00s
- rag_fixed: strategy=hybrid, recall=0.500, mrr=1.000, searches=1, time=0.19s
- rag_adaptive: strategy=hybrid_retry, recall=1.000, mrr=1.000, searches=2, time=0.19s

### pfas_corpus_specific

Question: Which methods are discussed for extractable organic fluorine in PFAS studies?
Category: `in_corpus_specific`
Expected strategy: `hybrid`

- llm_only: strategy=llm_only, recall=0.000, mrr=0.000, searches=0, time=0.00s
- rag_fixed: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.20s
- rag_adaptive: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.20s

### pfas_out_of_corpus

Question: What is the current market size for PFAS remediation startups?
Category: `out_of_corpus`
Expected strategy: `hybrid`

- llm_only: strategy=llm_only, recall=1.000, mrr=1.000, searches=0, time=0.00s
- rag_fixed: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.21s
- rag_adaptive: strategy=hybrid, recall=1.000, mrr=1.000, searches=1, time=0.22s
