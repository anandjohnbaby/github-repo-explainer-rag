from typing import List, Dict

# ---------------- CONFIG ----------------
DEFAULT_CHUNK_SIZE = 500   # approximate tokens (word-based)
DEFAULT_OVERLAP = 100      # overlap in words
# --------------------------------------

def estimate_tokens(text: str) -> int:
    # Rough token estimation (safe approximation for LLMs)
    return max(1, len(text) // 4)


def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    # Split text into overlapping word-based chunks
    words = text.split()
    chunks: List[str] = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))

        start = end - overlap
        if start < 0:
            start = 0

    return chunks


def chunk_documents(documents: List[Dict], chunk_size: int = DEFAULT_CHUNK_SIZE, 
    overlap: int = DEFAULT_OVERLAP) -> List[Dict]:

    all_chunks: List[Dict] = []

    for doc in documents:
        text = doc["content"]
        file_path = doc["file_path"]

        chunks = split_text(text=text, chunk_size=chunk_size, overlap=overlap)

        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                all_chunks.append({
                    "file_path": file_path,
                    "chunk_id": idx,
                    "content": chunk
                })

    return all_chunks
