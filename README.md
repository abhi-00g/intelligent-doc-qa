
# Intelligent Document Q&A System (RAG)

A Retrieval-Augmented Generation (RAG) app that lets users upload PDFs and ask natural-language questions.  
It embeds documents with **sentence-transformers**, stores vectors in **FAISS**, retrieves top matches, and asks an **OpenAI-compatible LLM** to answer with sources.

## Tech Stack
- **Python** (3.10+ recommended)
- **Streamlit** (UI)
- **FAISS** (vector index)
- **sentence-transformers** (embeddings)
- **LangChain** (optional utilities / prompt helpers)
- **OpenAI** (LLM API)
- **Docker / Docker Compose** (containerized run)

## Features
- PDF ingestion → text extraction → chunking
- Embeddings + FAISS index build
- Semantic retrieval (top-k)
- Grounded prompting with sources
- Streamlit UI for upload & Q&A
- Dockerized for one-command deployment

## Project Structure
```
Intelligent-Doc-QA-RAG/
├─ src/
│  ├─ ui/app.py                # Streamlit app
│  └─ rag/
│     ├─ config.py             # settings & paths
│     ├─ loaders.py            # PDF/text loaders
│     ├─ splitters.py          # text chunking
│     ├─ embeddings.py         # sentence-transformers
│     ├─ indexer.py            # FAISS build/save/load
│     ├─ retriever.py          # semantic search
│     ├─ llm.py                # OpenAI client + prompt
│     └─ pipeline.py           # end-to-end helpers
├─ data/                       # uploaded files + index
├─ requirements.txt
├─ Dockerfile
├─ docker-compose.yml
├─ .env.example
└─ README.md
```

## Quickstart (Local)
1) Create a virtual env and install deps:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
3) Run the app:
```bash
streamlit run src/ui/app.py
```
Open the URL Streamlit prints (usually http://localhost:8501).

## Quickstart (Docker)
```bash
cp .env.example .env  # fill in your key(s)
docker compose up --build
```
Then open http://localhost:8501.

## Tests
Basic smoke tests:
```bash
pytest -q
```

## Notes
- By default, embeddings are stored under `data/index/` and uploads under `data/uploads/`.
- You can switch the embedding model in `src/rag/embeddings.py`.
- LLM model selection lives in `src/rag/llm.py`.

## License
MIT
