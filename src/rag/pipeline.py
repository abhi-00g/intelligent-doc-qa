"""
Ingestion pipeline — PDF → chunks → embeddings → FAISS index.
Now with structured error handling for corrupt/empty PDFs.
"""

import logging
from pathlib import Path
from .config import UPLOADS_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from .loaders import load_pdf_text
from .splitters import chunk_text, align_chunks_with_pages
from .embeddings import embed_texts
from .indexer import build_faiss_index, save_index, save_metadata

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """Raised when PDF ingestion fails at any stage."""
    pass


def ingest_pdf(pdf_path: Path) -> list:
    """
    Ingest a PDF into the FAISS index.
    Returns the list of text chunks on success.
    Raises IngestionError with a user-friendly message on failure.
    """

    # --- Stage 1: Load PDF ---
    try:
        text, pages_meta = load_pdf_text(pdf_path)
    except Exception as e:
        logger.error("Failed to parse PDF %s: %s", pdf_path.name, e)
        raise IngestionError(
            f"Could not read '{pdf_path.name}'. The file may be corrupted, "
            f"password-protected, or not a valid PDF."
        )

    if not text or not text.strip():
        logger.warning("PDF %s produced no extractable text", pdf_path.name)
        raise IngestionError(
            f"No text could be extracted from '{pdf_path.name}'. "
            f"It may be a scanned document or contain only images."
        )

    # --- Stage 2: Chunk ---
    try:
        chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    except Exception as e:
        logger.error("Chunking failed for %s: %s", pdf_path.name, e)
        raise IngestionError(f"Failed to split document into chunks: {e}")

    if not chunks:
        logger.warning("No chunks produced from %s", pdf_path.name)
        raise IngestionError(
            f"Document '{pdf_path.name}' produced no usable text chunks."
        )

    meta = align_chunks_with_pages(chunks, pages_meta)
    logger.info("Chunked %s into %d segments", pdf_path.name, len(chunks))

    # --- Stage 3: Embed + Index ---
    try:
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)
        save_index(index, INDEX_DIR)
        save_metadata(meta, INDEX_DIR)
    except Exception as e:
        logger.error("Embedding/indexing failed for %s: %s", pdf_path.name, e)
        raise IngestionError(
            f"Failed to build the search index. This is usually a memory issue "
            f"with very large documents. Try a shorter PDF."
        )

    logger.info("Indexed %s: %d chunks, %d dimensions", pdf_path.name, len(chunks), vectors.shape[1])
    return chunks