# Intelligent Document Q&A (RAG)

A production-grade document intelligence system that lets you upload any PDF and ask questions about it in natural language. Built with a two-stage retrieval pipeline — FAISS vector search followed by cross-encoder reranking — for accurate, grounded answers with source citations.

## Live Demo
🔗 Live demo: https://intelligent-doc-app-cgqnqenifus7wynsfrq5a9.streamlit.app/
⚠️ Free-tier hosting — first load may take 1–2 minutes to wake up.

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
```

## Tech Stack

| Layer        | Technology                               |
|--------------|------------------------------------------|
| UI           | Streamlit                                |
| Embeddings   | sentence-transformers (all-MiniLM-L6-v2) |
| Vector store | FAISS                                    |
| Reranking    | cross-encoder/ms-marco-MiniLM-L-6-v2     |
| LLM          | Google Gemini 2.5 Flash (free tier)      |
| PDF parsing  | pypdf                                    |

## What makes this different from a basic RAG demo

- **Semantic chunking** — splits at sentence boundaries, not arbitrary character limits
- **Two-stage retrieval** — reranker eliminates irrelevant chunks that pass vector search
- **Source citations with page numbers** — every answer cites exactly where it came from
- **Zero cost** — runs entirely on free tier APIs, no credit card required

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

## Environment variables

| Variable          | Description                                       |
|-------------------|---------------------------------------------------|
| `GEMINI_API_KEY`  | Free API key from Google AI Studio                |
| `EMBEDDING_MODEL` | Default: `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE`      | Default: `500`                                    |
| `TOP_K`           | Chunks retrieved before reranking. Default: `4`   |

