"""Offline step: load schemes.json -> chunk -> embed -> store in ChromaDB.

Run with:  python -m app.services.ingest
Re-run whenever you change data/schemes.json.
"""
import json
from typing import Dict, List

import chromadb

from app.config import CHROMA_DIR, COLLECTION_NAME, DATA_FILE
from app.services.embedder import embed_texts


def load_schemes(path=DATA_FILE) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean(text) -> str:
    """Collapse whitespace so chunks are tidy."""
    return " ".join(str(text).split())


def build_chunks(schemes: List[Dict]) -> List[Dict]:
    """One chunk per section of each scheme (overview, eligibility, benefits, how to apply).

    Every chunk repeats the scheme name so it still makes sense on its own
    when retrieved without the rest of the scheme.
    """
    chunks = []
    for s in schemes:
        docs = ", ".join(s["documents_required"])
        sections = {
            "overview": f"{s['name']} ({s['category']}). {s['summary']}",
            "eligibility": f"Eligibility for {s['name']}: {s['eligibility']}",
            "benefits": f"Benefits of {s['name']}: {s['benefits']}",
            "how_to_apply": (
                f"How to apply for {s['name']}. Documents required: {docs}. "
                f"Steps: {s['how_to_apply']}"
            ),
        }
        for section, text in sections.items():
            chunks.append(
                {
                    "id": f"{s['id']}::{section}",
                    "text": clean(text),
                    "metadata": {
                        "scheme": s["name"],
                        "scheme_id": s["id"],
                        "category": s["category"],
                        "section": section,
                        "source_url": s["source_url"],
                    },
                }
            )
    return chunks


def run_ingest() -> int:
    schemes = load_schemes()
    chunks = build_chunks(schemes)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)  # start fresh so old data can't linger
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embed_texts([c["text"] for c in chunks]),
        metadatas=[c["metadata"] for c in chunks],
    )
    return len(chunks)


if __name__ == "__main__":
    total = run_ingest()
    print(f"Done. Stored {total} chunks in ChromaDB.")
