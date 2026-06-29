# Answer Auditing

The audit workflow checks whether factual claims in a generated answer are
supported by passages in the local PFAS corpus. It is a screening tool for review,
not a source of truth.

Run it with an answer string:

```bash
uv run pfas-audit "What methods are used for PFAS detection?"   --answer "LC-MS/MS is commonly used for PFAS detection in environmental samples."
```

Or audit an answer saved in a text file:

```bash
uv run pfas-audit "What methods are used for PFAS detection?"   --answer-file reports/example_answer.txt
```

Outputs:

- `reports/audit_result.json`: structured claim-level audit, ignored by git;
- `reports/audit_report.html`: local HTML review report, ignored by git.

Each claim is labelled as one of:

- `supported`
- `limited_support`
- `unsupported`
- `contradicted`
- `not_enough_evidence`

The status is based on local retrieval and lexical overlap with the retrieved
passages. Claims marked as anything other than `supported` should be manually
checked against the cited documents.
