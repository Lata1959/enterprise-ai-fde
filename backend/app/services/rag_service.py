from pathlib import Path

import chromadb
from openai import OpenAI

from app.core.config import settings


class RAGService:
    def __init__(self):
        self.openai = OpenAI(api_key=settings.OPENAI_API_KEY) if settings.OPENAI_API_KEY else None
        self.chroma = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        self.collection = self.chroma.get_or_create_collection(
            name="enterprise_knowledge",
            metadata={"hnsw:space": "cosine"},
        )

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 800,
        overlap: int = 150,
    ) -> list[str]:
        words = text.split()
        chunks = []
        start = 0

        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += chunk_size - overlap

        return chunks

    def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        if self.openai is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        response = self.openai.embeddings.create(
            model=settings.OPENAI_EMBEDDING_MODEL,
            input=texts,
        )

        return [item.embedding for item in response.data]

    def index_document(self, text: str, source: str) -> int:
        chunks = self.chunk_text(text)

        if not chunks:
            return 0

        embeddings = self.create_embeddings(chunks)
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]

        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.create_embeddings([question])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        retrieved = []
        for i, document in enumerate(results["documents"][0]):
            retrieved.append(
                {
                    "content": document,
                    "source": results["metadatas"][0][i]["source"],
                    "distance": results["distances"][0][i],
                }
            )

        return retrieved

    def answer_question(self, question: str, top_k: int = 3) -> dict:
        if self.openai is None:
            raise RuntimeError("OPENAI_API_KEY is not configured.")

        retrieved = self.search(question, top_k)

        if not retrieved:
            return {
                "answer": "I could not find relevant information in the knowledge base.",
                "sources": [],
            }

        context = "\n\n".join(
            [
                f"Source: {item['source']}\nContent: {item['content']}"
                for item in retrieved
            ]
        )

        response = self.openai.responses.create(
            model=settings.OPENAI_MODEL,
            instructions=(
                "You are an enterprise knowledge assistant. "
                "Answer questions using only the supplied context. "
                "Do not invent facts or use outside knowledge. "
                "If the context does not contain the answer, "
                "clearly say that the available documents do not "
                "provide enough information. "
                "Mention the relevant source filenames in your answer. "
                "Do not forget to say Thank you for the question and ask for feedback in the end."
            ),
            input=(
                f"QUESTION:\n{question}\n\n"
                f"RETRIEVED CONTEXT:\n{context}"
            ),
        )

        sources = list(dict.fromkeys(item["source"] for item in retrieved))

        return {
            "answer": response.output_text,
            "sources": sources,
            "retrieved_chunks": len(retrieved),
        }


rag_service = RAGService()


