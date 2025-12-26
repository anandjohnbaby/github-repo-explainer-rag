from typing import List, Dict
from pathlib import Path
import pickle
import numpy as np
import faiss
from app.core.paths import VECTORSTORE_DIR

# ---------------- CONFIG ----------------
INDEX_FILE = VECTORSTORE_DIR / "faiss.index"
META_FILE = VECTORSTORE_DIR / "metadata.pkl"
# --------------------------------------

class FAISSVectorStore:
    # FAISS-based vector store using cosine similarity
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)  # cosine similarity
        self.metadata: List[Dict] = []

        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    def add_documents(self, embedded_chunks: List[Dict]) -> None:
        # Add embedded chunks to FAISS index
        if not embedded_chunks:
            return

        vectors = np.array(
            [item["embedding"] for item in embedded_chunks],
            dtype="float32"
        )

        self.index.add(vectors)
        self.metadata.extend(embedded_chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict]:
        # Search FAISS index for similar vectors
        if self.index.ntotal == 0:
            return []

        query_vector = query_vector.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_vector, top_k)
        results: List[Dict] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.metadata[idx])

        return results

    def save(self) -> None:
        # Persist FAISS index and metadata to disk
        faiss.write_index(self.index, str(INDEX_FILE))
        with open(META_FILE, "wb") as f:
            pickle.dump(self.metadata, f)

    def load(self) -> None:
        # Load FAISS index and metadata from disk
        if not INDEX_FILE.exists() or not META_FILE.exists():
            raise FileNotFoundError("FAISS index or metadata not found")

        self.index = faiss.read_index(str(INDEX_FILE))
        with open(META_FILE, "rb") as f:
            self.metadata = pickle.load(f)
