"""Question -> most relevant chunks."""
from functools import lru_cache
from typing import Dict, List, Optional

import chromadb

from app.config import CHROMA_DIR, COLLECTION_NAME, MAX_DISTANCE, TOP_K
from app.services.embedder import embed_texts


class KnowledgeBaseNotReady(Exception):
    """Raised when ingest has not been run yet."""


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=CHROMA_DIR)  # opened once, reused


def _get_collection():
    try:
        return _client().get_collection(COLLECTION_NAME)
    except Exception as e:
        raise KnowledgeBaseNotReady("Run: python -m app.services.ingest") from e


def retrieve(question: str, top_k: int = TOP_K, category: Optional[str] = None) -> List[Dict]:
    collection = _get_collection()
    query_vector = embed_texts([question])[0]

    kwargs = {"query_embeddings": [query_vector], "n_results": top_k}
    if category:
        kwargs["where"] = {"category": category}  # metadata filter

    res = collection.query(**kwargs)

    results = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= MAX_DISTANCE:  # drop chunks that are not really related
            results.append(
                {
                    "text": doc,
                    "scheme": meta["scheme"],
                    "section": meta["section"],
                    "source_url": meta["source_url"],
                    "distance": round(dist, 3),
                }
            )
    return results
