
import json, pathlib
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX_PATH   = PROJECT_ROOT / "models" / "faiss_index" / "index.faiss"
META_PATH    = PROJECT_ROOT / "models" / "faiss_index" / "metadata.json"

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self.chunks = data["chunks"]   # list of {text, section, chunk_id, source}
        print(f"RAGEngine ready: {self.index.ntotal} chunks indexed.")

    def retrieve(self, query: str, top_k: int = 3) -> list:
        """Return top_k most relevant IEC chunks for the query."""
        q_vec = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def format_context(self, results: list) -> str:
        """Format retrieved chunks into a single context string for the LLM."""
        parts = []
        for r in results:
            parts.append(f"[{r['section']}] (relevance {r['score']:.2f})\n{r['text']}")
        return "\n\n".join(parts)


if __name__ == "__main__":
    engine = RAGEngine()
    test_queries = [
        "What does high acetylene indicate?",
        "Thermal fault above 700 degrees",
        "partial discharge hydrogen dominant"
    ]
    for q in test_queries:
        print(f"\nQuery: {q}")
        results = engine.retrieve(q, top_k=2)
        for r in results:
            print(f"  [{r['section']}] score={r['score']:.3f}  {r['text'][:80]}...")
