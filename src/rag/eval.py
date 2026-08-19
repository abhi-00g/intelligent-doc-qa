"""
Evaluation harness — measures retrieval and answer quality.

Usage:
    PYTHONPATH=src python -m rag.eval --pdf path/to/doc.pdf --eval-set path/to/eval.json

The eval set is a JSON file with this structure:
[
    {
        "question": "What is the main finding?",
        "expected_keywords": ["keyword1", "keyword2"],
        "expected_answer": "Optional ground truth answer"
    }
]

Metrics produced:
    - Retrieval Hit Rate: % of questions where at least one relevant chunk was retrieved
    - Keyword Recall: % of expected keywords found in the answer
    - Answer Coverage: % of questions that received a non-empty, non-error answer
    - Avg Latency: mean response time per question
    - Avg Tokens: mean token usage per question
"""

import json
import logging
import argparse
from pathlib import Path
from .pipeline import ingest_pdf
from .retriever import search
from .reranker import rerank
from .llm import answer_with_gemini
from .telemetry import get_session_stats

logger = logging.getLogger(__name__)


def keyword_recall(answer: str, expected_keywords: list) -> float:
    """Fraction of expected keywords found in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    found = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(found / len(expected_keywords), 3)


def retrieval_hit(chunks_text: list, expected_keywords: list) -> bool:
    """True if any retrieved chunk contains at least one expected keyword."""
    if not expected_keywords:
        return True
    combined = " ".join(chunks_text).lower()
    return any(kw.lower() in combined for kw in expected_keywords)


def run_eval(pdf_path: str, eval_set_path: str) -> dict:
    """
    Run the full evaluation pipeline.
    Returns a dict with per-question results and aggregate metrics.
    """
    pdf = Path(pdf_path)
    eval_set = json.loads(Path(eval_set_path).read_text())

    # Ingest the document
    logger.info("Ingesting %s for evaluation...", pdf.name)
    corpus = ingest_pdf(pdf)

    results = []
    total_keyword_recall = 0.0
    total_retrieval_hits = 0
    total_answered = 0

    for i, item in enumerate(eval_set):
        question = item["question"]
        expected_kw = item.get("expected_keywords", [])

        logger.info("Eval Q%d/%d: %s", i + 1, len(eval_set), question[:60])

        # Retrieve
        retrieved = search(question, corpus)
        reranked = rerank(question, retrieved)
        chunks_text = [r["text"] for r in reranked]

        # Check retrieval quality
        hit = retrieval_hit(chunks_text, expected_kw)
        total_retrieval_hits += int(hit)

        # Generate answer
        answer = answer_with_gemini(question, chunks_text)

        # Check answer quality
        is_error = answer.startswith("Error:") or answer.startswith("No relevant")
        if not is_error:
            total_answered += 1

        kw_recall = keyword_recall(answer, expected_kw)
        total_keyword_recall += kw_recall

        results.append({
            "question": question,
            "answer": answer[:200],
            "retrieval_hit": hit,
            "keyword_recall": kw_recall,
            "answered": not is_error,
            "num_chunks": len(reranked),
        })

    n = max(len(eval_set), 1)
    telemetry = get_session_stats()

    metrics = {
        "total_questions": len(eval_set),
        "retrieval_hit_rate": round(total_retrieval_hits / n, 3),
        "avg_keyword_recall": round(total_keyword_recall / n, 3),
        "answer_coverage": round(total_answered / n, 3),
        "avg_latency_ms": telemetry.get("avg_latency_ms", 0),
        "total_tokens": telemetry.get("total_tokens", 0),
        "total_cost_usd": telemetry.get("total_cost_usd", 0),
    }

    return {"metrics": metrics, "details": results}


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality")
    parser.add_argument("--pdf", required=True, help="Path to the PDF document")
    parser.add_argument("--eval-set", required=True, help="Path to the evaluation JSON file")
    parser.add_argument("--output", default=None, help="Save results to this JSON file")
    args = parser.parse_args()

    results = run_eval(args.pdf, args.eval_set)

    print("\n" + "=" * 50)
    print("RAG PIPELINE EVALUATION RESULTS")
    print("=" * 50)
    for key, val in results["metrics"].items():
        print(f"  {key:25s}: {val}")
    print("=" * 50)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()