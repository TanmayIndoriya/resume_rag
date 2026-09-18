"""
rag_pipeline.py
----------------
Core RAG (Retrieval-Augmented Generation) logic for the Resume RAG project.

Pipeline stages:
1. Ingest      -> extract raw text from an uploaded resume PDF
2. Chunk       -> split text into overlapping chunks (so context isn't lost at boundaries)
3. Embed       -> turn each chunk into a vector using a local sentence-transformer model
4. Store       -> persist chunks + vectors in a local Chroma vector database
5. Retrieve    -> given a user question, embed it and pull the top-k most similar chunks
6. Generate    -> feed the question + retrieved chunks to an LLM (Groq/Llama) to produce
                  a grounded answer. Falls back to a plain "extractive" mode with no LLM
                  key is configured, so the demo still works end-to-end with zero cost.
"""

import os
import uuid
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 100    # characters shared between consecutive chunks
TOP_K = 4              # number of chunks retrieved per query
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # small, fast, runs locally, no API needed
LLM_MODEL_NAME = "openai/gpt-oss-20b"


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF resume."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks so no context is lost across chunk boundaries."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


class ResumeRAG:
    """Wraps the full RAG pipeline: indexing a resume and answering questions about it."""

    def __init__(self, persist_dir: str = "./chroma_store", collection_name: str = "resume"):
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL_NAME
        )
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Fresh collection each time a new resume is indexed
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
        )

        self.groq_client = None
        api_key = os.getenv("GROQ_API_KEY")
        if GROQ_AVAILABLE and api_key:
            self.groq_client = Groq(api_key=api_key)

    def index_resume(self, pdf_path: str) -> int:
        """Extract, chunk, embed, and store the resume. Returns number of chunks indexed."""
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(raw_text)

        if not chunks:
            raise ValueError("No extractable text found in the PDF. Is it a scanned image?")

        ids = [str(uuid.uuid4()) for _ in chunks]
        self.collection.add(documents=chunks, ids=ids)
        return len(chunks)

    def retrieve(self, query: str, k: int = TOP_K) -> List[str]:
        """Return the top-k chunks most relevant to the query."""
        results = self.collection.query(query_texts=[query], n_results=k)
        return results["documents"][0] if results["documents"] else []

    def generate_answer(self, query: str, context_chunks: List[str]) -> Dict:
        """Generate a grounded answer using retrieved context. Falls back to
        extractive mode (just showing the chunks) if no LLM API key is set."""
        context = "\n---\n".join(context_chunks)

        if self.groq_client is None:
            return {
                "answer": (
                    "[No GROQ_API_KEY set — showing retrieved context only. "
                    "Add a free key in .env to get generated answers.]\n\n" + context
                ),
                "mode": "retrieval-only",
            }

        system_prompt = (
            "You are a helpful assistant answering questions about a candidate's resume. "
            "Only use the provided context. If the answer isn't in the context, say you "
            "don't have that information in the resume. Be concise and professional."
        )
        user_prompt = f"Context from resume:\n{context}\n\nQuestion: {query}"

        completion = self.groq_client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        answer = completion.choices[0].message.content
        return {"answer": answer, "mode": "generated"}

    def ask(self, query: str, k: int = TOP_K) -> Dict:
        """End-to-end: retrieve relevant chunks, then generate an answer."""
        chunks = self.retrieve(query, k=k)
        if not chunks:
            return {"answer": "No resume has been indexed yet, or no relevant content was found.",
                    "mode": "none", "sources": []}
        result = self.generate_answer(query, chunks)
        result["sources"] = chunks
        return result