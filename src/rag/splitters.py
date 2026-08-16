import re
from typing import List, Tuple

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Semantic chunking: splits at sentence boundaries instead of
    arbitrary character positions. Keeps thoughts intact which
    produces better embeddings and cleaner retrieval snippets.
    """
    # Split into sentences at punctuation boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Overlap: carry last ~40 words into next chunk
            # so context isn't lost at boundaries
            words = current_chunk.split()
            overlap_words = words[-(overlap // 5):]
            current_chunk = " ".join(overlap_words) + " " + sentence
        else:
            current_chunk += " " + sentence
    
    # Don't lose the final chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def align_chunks_with_pages(chunks: List[str], pages_meta: List[Tuple]) -> List[dict]:
    """
    Maps each chunk back to its source page.
    Simple proportional mapping — good enough for single PDFs.
    """
    total_chunks = len(chunks)
    total_pages = len(pages_meta)
    meta = []

    for i, chunk in enumerate(chunks):
        # Estimate which page this chunk came from proportionally
        page_index = min(int((i / total_chunks) * total_pages), total_pages - 1)
        source, page_num = pages_meta[page_index]
        meta.append({
            "id": i,
            "source": source,
            "page": page_num
        })

    return meta