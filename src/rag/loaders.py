
from typing import List, Tuple
from pathlib import Path
from pypdf import PdfReader

def load_pdf_text(path: Path) -> Tuple[str, List[Tuple[str, int]]]:
    """Return (full_text, [(source_id, page_num), ...]) for provenance."""
    reader = PdfReader(str(path))
    pages = []
    text_parts = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        text_parts.append(txt)
        pages.append((f"{path.name}", i + 1))
    return "\n".join(text_parts), pages
