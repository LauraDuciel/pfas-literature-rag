import numpy as np

from pfas_lit_rag.schemas import TextChunk
from pfas_lit_rag.vector_store import VectorStore


def _chunk(chunk_id: str, text: str) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id="doc",
        title="PFAS paper",
        source_path="paper.pdf",
        page_start=1,
        page_end=1,
        text=text,
    )


def test_vector_store_appends_chunks(tmp_path) -> None:
    store = VectorStore(tmp_path)
    first = _chunk("c1", "first chunk")
    second = _chunk("c2", "second chunk")

    store.append([first], np.asarray([[1.0, 0.0]], dtype="float32"))
    store.append([second], np.asarray([[0.0, 1.0]], dtype="float32"))

    _, chunks = store.load()
    assert [chunk.chunk_id for chunk in chunks] == ["c1", "c2"]
    assert store.existing_chunk_ids() == {"c1", "c2"}
