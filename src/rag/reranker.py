"""
Cross-encoder reranker with graceful fallback on model load failure.
"""

import logging
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

_reranker = None


def get_reranker():
    """Load the cross-encoder model, with caching and error handling."""
    global _reranker
    if _reranker is None:
        try:
            _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("Cross-encoder reranker loaded")
        except Exception as e:
            logger.error("Failed to load cross-encoder model: %s", e)
            return None
    return _reranker


def rerank(query: str, results: list, top_n: int = 3) -> list:
    """
    Re-score results with the cross-encoder.
    Falls back to the original ranking if the model fails to load.
    """
    if not results:
        return results

    reranker = get_reranker()
    if reranker is None:
        logger.warning("Reranker unavailable — returning original ranking")
        return results[:top_n]

    try:
        pairs = [(query, r["text"]) for r in results]
        scores = reranker.predict(pairs)
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
    except Exception as e:
        logger.error("Reranking failed: %s — returning original ranking", e)
        return results[:top_n]