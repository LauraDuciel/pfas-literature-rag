import json
from pathlib import Path

import faiss
import numpy as np

from pfas_lit_rag.schemas import SearchResult, TextChunk

INDEX_FILENAME = "chunks.faiss"
METADATA_FILENAME = "chunks.jsonl"


class VectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.index_path = index_dir / INDEX_FILENAME
        self.metadata_path = index_dir / METADATA_FILENAME

    def write(self, chunks: list[TextChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings does not match")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        faiss.write_index(index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            for chunk in chunks:
                handle.write(chunk.model_dump_json() + "\n")

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[SearchResult]:
        index, chunks = self.load()
        if len(chunks) == 0:
            return []
        scores, indices = index.search(query_embedding, min(top_k, len(chunks)))
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            results.append(SearchResult(chunk=chunks[int(idx)], score=float(score)))
        return results

    def load(self) -> tuple[faiss.Index, list[TextChunk]]:
        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Index not found in {self.index_dir}. Run `pfas-ingest` first."
            )
        index = faiss.read_index(str(self.index_path))
        chunks = []
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                chunks.append(TextChunk.model_validate(record))
        return index, chunks
