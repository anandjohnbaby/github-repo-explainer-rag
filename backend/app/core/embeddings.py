from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ----------------
MODEL_NAME = "all-MiniLM-L6-v2"
# --------------------------------------

# Load model
model = SentenceTransformer(MODEL_NAME)

def generate_embeddings(chunks: List[Dict]) -> List[Dict]:
    # Generate embeddings for a list of text chunks
    if not chunks:
        return []

    texts = [chunk["content"] for chunk in chunks]

    vectors = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    embedded_chunks: List[Dict] = []
    for idx, chunk in enumerate(chunks):
        embedded_chunks.append({
            **chunk,
            "embedding": vectors[idx]
        })

    return embedded_chunks

def embeddings_to_numpy(embedded_chunks: List[Dict]) -> np.ndarray:
    # Convert embedded chunks into a numpy matrix (for FAISS)
    return np.array([item["embedding"] for item in embedded_chunks])
