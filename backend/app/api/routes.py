from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.ingestion import ingest_repository
from app.core.chunking import chunk_documents
from app.core.embeddings import generate_embeddings
from app.core.vectorstore import FAISSVectorStore
from app.core.retriever import Retriever
from app.core.llm import LLMClient

router = APIRouter()

# ---------- GLOBAL OBJECTS ----------
vector_store = None
retriever = None
llm_client = LLMClient()
# ----------------------------------

# ---------- REQUEST MODELS ----------
class IngestRequest(BaseModel):
    github_url: str

class ChatRequest(BaseModel):
    question: str
# ----------------------------------

@router.post("/ingest")
def ingest_repo(data: IngestRequest):
    # Ingest a GitHub repository and build vector store
    global vector_store, retriever

    try:
        # 1️⃣ Ingest repository (clone + load)
        documents = ingest_repository(data.github_url)

        if not documents:
            raise HTTPException(status_code=400, detail="No valid files found")

        # 2️⃣ Chunk documents
        chunks = chunk_documents(documents)

        # 3️⃣ Generate embeddings
        embedded_chunks = generate_embeddings(chunks)

        # 4️⃣ Store embeddings in FAISS
        vector_store = FAISSVectorStore(
            embedding_dim=len(embedded_chunks[0]["embedding"])
        )
        vector_store.add_documents(embedded_chunks)
        vector_store.save()

        # 5️⃣ Create retriever
        retriever = Retriever(vector_store)

        return {
            "status": "success",
            "files_loaded": len(documents),
            "chunks_created": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
def chat_with_repo(data: ChatRequest):
    # Chat with the ingested repository
    if retriever is None:
        raise HTTPException(
            status_code=400,
            detail="No repository ingested yet. Call /ingest first."
        )

    try:
        # 1️⃣ Retrieve relevant context
        context = retriever.retrieve(data.question, top_k=5)
        print("\n--- RETRIEVED CONTEXT ---")
        for c in context:
            print("File path : ", c["file_path"])
            print("CONTENT : ", c["content"])
        print("-------------------------")

        # 2️⃣ Generate answer via LLM
        answer = llm_client.generate_answer(context, data.question)

        return {
            "question": data.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
