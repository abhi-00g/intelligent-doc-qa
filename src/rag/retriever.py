
from .indexer import load_index, load_metadata
from .embeddings import embed_texts
from .config import INDEX_DIR, TOP_K

def search(query: str, corpus_texts, top_k: int = TOP_K):
    index = load_index(INDEX_DIR)
    metadata = load_metadata(INDEX_DIR)
    if index is None or not metadata or not corpus_texts:
        return []

    import numpy as np
    q_vec = embed_texts([query]).astype('float32')
    sims, idxs = index.search(q_vec, top_k)
    idxs = idxs[0]
    results = []
    for rank, i in enumerate(idxs):
        if i < 0 or i >= len(corpus_texts):
            continue
        results.append({
            "rank": rank + 1,
            "text": corpus_texts[i],
            "meta": metadata[i]
        })
    return results
