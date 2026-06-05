import json
from pathlib import Path

from pfas_lit_rag.schemas import DocumentRecord

DOCUMENT_MANIFEST_FILENAME = "documents.jsonl"


class DocumentManifest:
    def __init__(self, metadata_dir: Path) -> None:
        self.metadata_dir = metadata_dir
        self.path = metadata_dir / DOCUMENT_MANIFEST_FILENAME

    def load(self) -> dict[str, DocumentRecord]:
        if not self.path.exists():
            return {}
        records: dict[str, DocumentRecord] = {}
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = DocumentRecord.model_validate(json.loads(line))
                records[record.document_id] = record
        return records

    def write(self, records: list[DocumentRecord]) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        deduplicated = {record.document_id: record for record in records}
        with self.path.open("w", encoding="utf-8") as handle:
            for record in sorted(deduplicated.values(), key=lambda item: item.source_path):
                handle.write(record.model_dump_json() + "\n")
