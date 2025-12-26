from typing import List, Dict
import numpy as np
from app.core.vectorstore import FAISSVectorStore
from app.core.embeddings import model

class Retriever:
    # Retrieves relevant chunks from the vector store using semantic search
    def __init__(self, vector_store: FAISSVectorStore):
        self.vector_store = vector_store

    def _embed_query(self, query: str) -> np.ndarray:
        # Convert user query into an embedding vector
        return model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict]:
        # Retrieve top-k relevant chunks for a query
        query_vector = self._embed_query(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        # Return only what LLM needs
        return [
            {
                "file_path": item["file_path"],
                "content": item["content"]
            }
            for item in results
        ]
