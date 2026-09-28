"""Turns text into vectors (embeddings).

Uses Chroma's built-in embedding function: the all-MiniLM-L6-v2 model running on
ONNX Runtime. Same model family as sentence-transformers, but without PyTorch,
so it needs far less memory (important on small free servers).
The model file (~90 MB) downloads automatically the first time it is used.
"""
from functools import lru_cache
from typing import List


@lru_cache(maxsize=1)
def get_model():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


def embed_texts(texts: List[str]) -> List[List[float]]:
    vectors = get_model()(texts)
    return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]
