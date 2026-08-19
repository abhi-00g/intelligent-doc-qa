# Intelligent Document Q&A (RAG)

A production-grade document intelligence system that lets you upload any PDF and ask questions about it in natural language. Built with a two-stage retrieval pipeline — FAISS vector search followed by cross-encoder reranking — for accurate, grounded answers with source citations.

## Live Demo

🔗 Live demo: https://intelligent-doc-app-cgqnqenifus7wynsfrq5a9.streamlit.app/
> ⚠️ Free-tier hosting — first load may take 5-7 minutes to wake up.

## How it works

1. **Ingest** — PDF is parsed and split into chunks at sentence boundaries (semantic chunking)
2. **Embed** — Chunks are converted to vector embeddings using `sentence-transformers`
3. **Index** — Embeddings stored in a FAISS index for fast similarity search
4. **Retrieve** — Top-k chunks retrieved by vector similarity
5. **Rerank** — Cross-encoder model (`ms-marco-MiniLM-L-6-v2`) re-scores chunks for true relevance
6. **Answer** — Gemini 2.5 Flash synthesizes a grounded answer from top chunks with source citations

## Architecture

```
PDF Upload → Semantic Chunker → Sentence Transformer Embeddings
→ FAISS Vector Index → Top-K Retrieval → Cross-Encoder Reranker
→ Gemini 2.5 Flash → Grounded Answer + Source Citations
                ↓
        Telemetry Logger → tokens, cost, latency per call
```

## Tech Stack

| Layer           | Technology                               |
|-----------------|------------------------------------------|
| UI              | Streamlit                                |
| Embeddings      | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store    | FAISS                                    |
| Reranking       | cross-encoder/ms-marco-MiniLM-L-6-v2     |
| LLM             | Google Gemini 2.5 Flash (free tier)      |
| PDF parsing     | pypdf                                    |
| Telemetry       | Custom token/cost/latency logger         |
| CI/CD           | GitHub Actions (pytest)                  |
| Containerization| Docker + Docker Compose                  |

## What makes this different from a basic RAG demo

- **Two-stage retrieval** — cross-encoder reranker eliminates irrelevant chunks that pass vector search
- **Semantic chunking** — splits at sentence boundaries, not arbitrary character limits
- **Source citations with page numbers** — every answer cites exactly where it came from
- **Token & cost telemetry** — every LLM call is logged with input/output tokens, estimated cost, and latency
- **Evaluation harness** — built-in framework to measure retrieval hit rate, keyword recall, and answer coverage
- **Structured error handling** — corrupt PDFs, rate limits, and empty indexes return user-friendly messages instead of stack traces
- **Zero cost** — runs entirely on free tier APIs, no credit card required

## Telemetry & Observability

Every Gemini API call is logged with:

- **Input/output token counts** extracted from the response metadata
- **Estimated cost in USD** based on Gemini 2.5 Flash pricing
- **Latency in milliseconds** measured end-to-end
- **Question preview** and status (success/error)

The Streamlit sidebar displays live cumulative stats (total calls, tokens, cost, avg latency), and each answer shows per-query telemetry inline. All data is persisted to `data/telemetry.json`.

## Evaluation

The project includes a built-in evaluation harness (`src/rag/eval.py`) that measures pipeline quality:

- **Retrieval Hit Rate** — % of questions where at least one relevant chunk was retrieved
- **Keyword Recall** — % of expected keywords found in the generated answer
- **Answer Coverage** — % of questions that received a non-error, grounded answer

Run an evaluation:

```bash
# Create an eval set (see sample_eval.json for the format)
PYTHONPATH=src python -m rag.eval --pdf path/to/document.pdf --eval-set eval_questions.json --output results.json
```

## Error Handling

The system handles failures gracefully at every stage:

- **Corrupt/empty PDFs** — returns a descriptive error instead of crashing
- **Gemini rate limits (429)** — displays "Rate limit reached, please wait"
- **Safety filter blocks** — explains the block and asks the user to rephrase
- **Missing FAISS index** — prompts the user to upload a document first
- **Cross-encoder load failure** — falls back to the original vector similarity ranking

## Run locally

```bash
git clone https://github.com/abhi-00g/intelligent-doc-qa.git
cd intelligent-doc-qa
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add your GEMINI_API_KEY to .env
# Get a free key at https://aistudio.google.com/apikey
PYTHONPATH=src python -m streamlit run src/ui/app.py
```

## Run with Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

## Run tests

```bash
PYTHONPATH=src python -m pytest src/tests/ -q
```

## Environment variables

| Variable          | Description                                       |
|-------------------|---------------------------------------------------|
| `GEMINI_API_KEY`  | Free API key from Google AI Studio                |
| `EMBEDDING_MODEL` | Default: `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE`      | Default: `500`                                    |
| `TOP_K`           | Chunks retrieved before reranking. Default: `4`   |
