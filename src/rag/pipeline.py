
from pathlib import Path
from .config import UPLOADS_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .loaders import load_pdf_text
from .splitters import chunk_text, align_chunks_with_pages
from .embeddings import embed_texts
from .indexer import build_faiss_index, save_index, save_metadata

def ingest_pdf(pdf_path: Path):
    text, pages_meta = load_pdf_text(pdf_path)
    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    meta = align_chunks_with_pages(chunks, pages_meta)

    # embed + index
    vectors = embed_texts(chunks)
    index = build_faiss_index(vectors)
    save_index(index, INDEX_DIR)
    save_metadata(meta, INDEX_DIR)
    return chunks
