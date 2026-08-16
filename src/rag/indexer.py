
from typing import List, Tuple
import faiss, numpy as np
from pathlib import Path
import json

def build_faiss_index(vectors) -> faiss.IndexFlatIP:
    import numpy as np
    vectors = vectors.astype('float32')
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index

def save_index(index, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "vectors.faiss"))

def load_index(index_dir: Path):
    path = index_dir / "vectors.faiss"
    if not path.exists():
        return None
    return faiss.read_index(str(path))

def save_metadata(metadata, index_dir: Path):
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def load_metadata(index_dir: Path):
    path = index_dir / "meta.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
