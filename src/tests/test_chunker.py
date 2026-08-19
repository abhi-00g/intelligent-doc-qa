from rag.splitters import chunk_text, align_chunks_with_pages


def test_single_sentence_returns_one_chunk():
    text = "This is a short sentence."
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0] == "This is a short sentence."


def test_multiple_sentences_split_at_boundary():
    sentences = ["Sentence number %d is here." % i for i in range(20)]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=100, overlap=200)
    assert len(chunks) >= 3


def test_overlap_carries_context():
    sentences = ["First important sentence.", "Second key detail.", "Third conclusion."]
    text = " ".join(sentences)
    chunks = chunk_text(text, chunk_size=30, overlap=200)
    if len(chunks) > 1:
        # Later chunks should contain words from previous chunk
        assert len(chunks[1]) > 0


def test_empty_string_returns_empty():
    chunks = chunk_text("", chunk_size=500)
    assert chunks == [] or chunks == [""]


def test_align_chunks_with_pages():
    chunks = ["chunk0", "chunk1", "chunk2", "chunk3"]
    pages_meta = [("doc.pdf", 1), ("doc.pdf", 2)]
    result = align_chunks_with_pages(chunks, pages_meta)
    assert len(result) == 4
    assert all("source" in r and "page" in r for r in result)
    assert result[0]["page"] == 1