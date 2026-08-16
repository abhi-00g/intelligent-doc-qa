from sentence_transformers import CrossEncoder

_reranker = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

def rerank(query: str, results: list, top_n: int = 3) -> list:
    if not results:
        return results
    reranker = get_reranker()
    pairs = [(query, r["text"]) for r in results]
    scores = reranker.predict(pairs)
    for i, r in enumerate(results):
        r["rerank_score"] = float(scores[i])
    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_n]