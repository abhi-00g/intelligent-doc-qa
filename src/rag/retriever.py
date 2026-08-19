"""
Retriever — FAISS similarity search with error handling.
"""

import logging
import numpy as np
from .indexer import load_index, load_metadata
from .embeddings import embed_texts
from .config import INDEX_DIR, TOP_K

logger = logging.getLogger(__name__)


def search(query: str, corpus_texts: list, top_k: int = TOP_K) -> list:
    """
    Search the FAISS index for chunks relevant to the query.
    Returns an empty list (instead of crashing) when the index
    is missing or the corpus is empty.
    """

    if not query or not query.strip():
        logger.warning("Empty query received")
        return []

    index = load_index(INDEX_DIR)
    metadata = load_metadata(INDEX_DIR)

    if index is None:
        logger.warning("No FAISS index found — upload a document first")
        return []

    if not metadata or not corpus_texts:
        logger.warning("Empty metadata or corpus — upload a document first")
        return []

    try:
        q_vec = embed_texts([query]).astype("float32")
        sims, idxs = index.search(q_vec, min(top_k, len(corpus_texts)))
    except Exception as e:
        logger.error("FAISS search failed: %s", e)
        return []

    idxs = idxs[0]
    results = []
    for rank, i in enumerate(idxs):
        if i < 0 or i >= len(corpus_texts):
            continue
        results.append({
            "rank": rank + 1,
            "text": corpus_texts[i],
            "meta": metadata[i] if i < len(metadata) else {},
            "similarity": float(sims[0][rank]) if rank < len(sims[0]) else 0.0,
        })

    logger.info("Retrieved %d chunks for query: %s", len(results), query[:60])
    return results