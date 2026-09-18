# Resume RAG Assistant

A local, end-to-end Retrieval-Augmented Generation (RAG) project: upload a resume PDF,
ask natural-language questions about it, and get grounded answers with the source
chunks shown alongside.

Built as a portfolio project — every stage of a real RAG pipeline is implemented
explicitly (not hidden behind a single library call) so it's easy to explain in an
interview.

## Architecture

```
Resume PDF
    │  (pypdf)
    ▼
Raw text
    │  (chunk_text: sliding window, 500 chars, 100 overlap)
    ▼
Text chunks
    │  (sentence-transformers: all-MiniLM-L6-v2, runs locally)
    ▼
Embeddings ──────────► ChromaDB (local persistent vector store)
                              │
User question                │  (same embedding model)
    │                        │
    ▼                        ▼
Query embedding ──► similarity search ──► top-k relevant chunks
                                                  │
                                                  ▼
                                    Groq API (Llama 3.1 8B) + chunks
                                                  │
                                                  ▼
                                          Grounded answer
```

**Why this stack:**
- **pypdf** — lightweight, no external dependencies, good enough for text-based resume PDFs.
- **sentence-transformers (all-MiniLM-L6-v2)** — small (~80MB), fast, runs fully on CPU,
  no API cost. This is what turns text into vectors that capture meaning, not just keywords.
- **ChromaDB** — a local vector database; stores embeddings on disk and does the
  nearest-neighbor search for retrieval. No server setup needed.
- **Groq (Llama 3.1 8B Instant)** — free-tier, very fast LLM API used only for the final
  "generate a natural answer from context" step. If no API key is set, the app still
  runs and shows you the raw retrieved chunks — so the retrieval half of RAG always works.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. (Optional but recommended) Get a free Groq API key:
   - Sign up at https://console.groq.com/keys
   - Copy `.env.example` to `.env` and paste your key in:
     ```bash
     cp .env.example .env
     ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (usually http://localhost:8501), upload your
   resume PDF, click **Build / Rebuild Index**, and start asking questions.

## Example questions to try
- "What programming languages does this candidate know?"
- "Summarize this candidate's most recent project."
- "Does this person have any leadership experience?"
- "What is their educational background?"

## Project structure
```
resume_rag/
├── app.py              # Streamlit UI
├── rag_pipeline.py      # Core RAG logic (ingest, chunk, embed, store, retrieve, generate)
├── requirements.txt
├── .env.example
└── README.md
```
