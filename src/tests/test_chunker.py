
from src.rag.splitters import chunk_text

def test_chunker_basic():
    txt = "a" * 2500
    chunks = chunk_text(txt, chunk_size=1000, overlap=200)
    assert len(chunks) >= 3
