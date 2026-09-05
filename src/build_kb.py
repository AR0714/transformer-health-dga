
import os, json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
KB_DIR  = PROJECT_ROOT / "data" / "iec_knowledge_base"
OUT_DIR = PROJECT_ROOT / "models" / "faiss_index"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE    = 120
CHUNK_OVERLAP = 20

def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start += size - overlap
    return chunks

print("Loading embedding model (all-MiniLM-L6-v2)...")
model = SentenceTransformer("all-MiniLM-L6-v2")

texts, metas = [], []
txt_files = sorted(KB_DIR.glob("*.txt"))
print(f"Found {len(txt_files)} knowledge files")

for fpath in txt_files:
    section = fpath.stem
    raw = fpath.read_text(encoding="utf-8")
    chunks = chunk_text(raw)
    for i, ch in enumerate(chunks):
        texts.append(ch)
        metas.append({"section": section, "chunk_id": i, "source": fpath.name})
    print(f"  {fpath.name:30s}  ->  {len(chunks)} chunks")

print(f"\nTotal chunks: {len(texts)}")
print("Generating embeddings...")
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings.astype(np.float32))

faiss.write_index(index, str(OUT_DIR / "index.faiss"))

metadata = {"chunks": [{"text": t, **m} for t, m in zip(texts, metas)]}
(OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

print(f"\nFAISS index saved -> models/faiss_index/index.faiss  ({index.ntotal} vectors, dim={dim})")
print(f"Metadata saved    -> models/faiss_index/metadata.json")
print("Step 15 Cell 2 complete")
