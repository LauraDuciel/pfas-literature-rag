from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding


class EmbeddingModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = list(self._model.embed(texts))
        array = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)


@lru_cache
def get_embedding_model(model_name: str) -> EmbeddingModel:
    return EmbeddingModel(model_name)
