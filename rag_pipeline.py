"""
rag_pipeline.py
----------------
Core RAG (Retrieval-Augmented Generation) logic for the Resume RAG project.

Pipeline stages:
1. Ingest      -> extract raw text from an uploaded resume PDF
2. Section-split -> split text into resume sections (Experience, Education, Skills...)
3. Chunk       -> split each section into overlapping chunks (so context isn't lost
                  at boundaries), tagging every chunk with its source section
4. Embed       -> turn each chunk into a vector using a local sentence-transformer model
5. Store       -> persist chunks + vectors + section metadata in a local Chroma vector DB
6. Retrieve    -> given a user question, embed it and pull the top-k most similar chunks
7. Generate    -> feed the question + retrieved chunks to an LLM (Groq) to produce
                  a grounded answer. Falls back to a plain "retrieval-only" mode if no
                  LLM key is configured, so the demo still works end-to-end at zero cost.
"""

import os
import re
import uuid
from typing import List, Dict, Tuple, Optional

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
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"     # small, fast, runs locally, no API needed
LLM_MODEL_NAME = "openai/gpt-oss-20b"     # served for free via Groq

# Common resume section headings. Matching is case-insensitive and tolerant of
# variants ("Work Experience", "Professional Experience", "Experience" all match).
SECTION_HEADERS = [
    "summary", "professional summary", "objective", "profile",
    "experience", "work experience", "professional experience",
    "employment history", "relevant experience",
    "education", "academic background", "academic qualifications",
    "skills", "technical skills", "core competencies", "key skills",
    "projects", "personal projects", "academic projects",
    "certifications", "certificates", "licenses",
    "achievements", "accomplishments", "awards", "honors",
    "publications",
    "volunteer experience", "volunteering", "community involvement",
    "leadership", "extracurricular activities",
    "languages",
    "interests", "hobbies",
    "contact", "contact information",
]


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF resume, preserving line breaks (needed for
    section-header detection)."""
    reader = PdfReader(file_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)
    return "\n".join(pages_text)


def _is_section_header(line: str) -> Optional[str]:
    """Return the cleaned header name if `line` looks like a resume section
    heading, else None. Headings are short lines that match (or start with)
    one of SECTION_HEADERS."""
    cleaned = line.strip().strip(":").strip()
    if not cleaned or len(cleaned) > 40:
        return None
    lowered = re.sub(r"[^a-z\s]", "", cleaned.lower()).strip()
    for header in SECTION_HEADERS:
        if lowered == header or lowered.startswith(header):
            return cleaned
    return None


def split_into_sections(raw_text: str) -> List[Tuple[str, str]]:
    """Split resume text into (section_name, section_text) pairs based on
    detected headings. Text before the first recognized heading is grouped
    under 'General'."""
    lines = raw_text.split("\n")
    sections: List[Tuple[str, str]] = []
    current_name = "General"
    current_lines: List[str] = []

    for line in lines:
        header = _is_section_header(line)
        if header:
            if current_lines:
                sections.append((current_name, "\n".join(current_lines)))
            current_name = header
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, "\n".join(current_lines)))

    return sections


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


def chunk_resume(raw_text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Section-aware chunking: split into resume sections first, then chunk
    within each section. Every chunk keeps its section name as metadata, which
    both improves retrieval (sections are semantically coherent) and lets the
    UI show *where in the resume* an answer came from."""
    sections = split_into_sections(raw_text)
    chunks: List[Dict] = []

    for section_name, section_text in sections:
        normalized = " ".join(section_text.split())
        if not normalized:
            continue
        if len(normalized) <= chunk_size:
            chunks.append({"text": normalized, "section": section_name})
        else:
            for sub_chunk in chunk_text(normalized, chunk_size, overlap):
                chunks.append({"text": sub_chunk, "section": section_name})

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
        """Extract, section-split, chunk, embed, and store the resume.
        Returns number of chunks indexed."""
        raw_text = extract_text_from_pdf(pdf_path)
        chunks = chunk_resume(raw_text)

        if not chunks:
            raise ValueError("No extractable text found in the PDF. Is it a scanned image?")

        ids = [str(uuid.uuid4()) for _ in chunks]
        documents = [c["text"] for c in chunks]
        metadatas = [{"section": c["section"]} for c in chunks]
        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return len(chunks)

    def retrieve(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Return the top-k chunks most relevant to the query, each as
        {"text": ..., "section": ..., "distance": ...}."""
        results = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        if not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        return [
            {"text": doc, "section": meta.get("section", "General"), "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]

    def generate_answer(self, query: str, retrieved: List[Dict]) -> Dict:
        """Generate a grounded answer using retrieved context. Falls back to
        retrieval-only mode if no LLM API key is set."""
        context = "\n---\n".join(f"[{r['section']}] {r['text']}" for r in retrieved)

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
        retrieved = self.retrieve(query, k=k)
        if not retrieved:
            return {"answer": "No resume has been indexed yet, or no relevant content was found.",
                    "mode": "none", "sources": []}
        result = self.generate_answer(query, retrieved)
        result["sources"] = retrieved
        return result
