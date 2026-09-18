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
    │  (pypdf, line breaks preserved)
    ▼
Raw text
    │  (split_into_sections: detect headings like "Experience", "Education", "Skills")
    ▼
Sections  ──►  (chunk_text within each section: sliding window, 500 chars, 100 overlap)
    ▼
Text chunks, each tagged with its section name
    │  (sentence-transformers: all-MiniLM-L6-v2, runs locally)
    ▼
Embeddings ──────────► ChromaDB (local persistent vector store, with section metadata)
                              │
User question                │  (same embedding model)
    │                        │
    ▼                        ▼
Query embedding ──► similarity search ──► top-k relevant chunks (+ section, + distance)
                                                  │
                                                  ▼
                                    Groq API (openai/gpt-oss-20b) + chunks
                                                  │
                                                  ▼
                                          Grounded answer
```

**Why this stack:**
- **pypdf** — lightweight, no external dependencies, good enough for text-based resume PDFs.
- **Section-aware chunking** — resumes have clear structure (Experience, Education, Skills,
  Projects...). Splitting on those headings *before* chunking keeps each chunk semantically
  coherent (e.g. an "Education" chunk never gets mixed with unrelated "Skills" text), and
  tagging each chunk with its section lets the UI show exactly where an answer came from.
  A plain fixed-size sliding window is used as the fallback for text that appears before
  the first recognized heading, or within long sections.
- **sentence-transformers (all-MiniLM-L6-v2)** — small (~80MB), fast, runs fully on CPU,
  no API cost. This is what turns text into vectors that capture meaning, not just keywords.
- **ChromaDB** — a local vector database; stores embeddings, section metadata, and does the
  nearest-neighbor search for retrieval. No server setup needed.
- **Groq (openai/gpt-oss-20b)** — free-tier, very fast LLM API used only for the final
  "generate a natural answer from context" step. If no API key is set, the app still
  runs and shows you the raw retrieved chunks — so the retrieval half of RAG always works.
  (Which exact model IDs are available varies by Groq account — run
  `client.models.list()` if you hit a 404 and swap `LLM_MODEL_NAME` in `rag_pipeline.py`.)

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
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

## Evaluating retrieval + answer quality

`eval.py` is a small evaluation harness — the piece most fresher RAG projects skip,
and the thing worth highlighting most in an interview.

1. Place your resume at `data/resume.pdf` (or edit `RESUME_PATH` in `eval.py`).
2. Edit `eval_questions.json` — replace the example keywords with ones that actually
   match your resume's content (a real skill you list, your actual degree name, etc).
   One question in the template is deliberately a question your resume can't answer
   ("favorite color") — checking that the model correctly says it doesn't know is as
   important as checking that it answers correctly when it can.
3. Run:
   ```bash
   python eval.py
   ```
4. It reports two separate scores:
   - **Retrieval hit rate** — did vector search find a chunk containing the right
     keywords, regardless of what the LLM did with it?
   - **Answer hit rate** — did the final generated answer contain them?

   Splitting these matters: if retrieval hits but the answer misses, the problem is in
   the generation prompt. If retrieval itself misses, the problem is in chunking or
   section detection. Full per-question results are saved to `eval_results.json`.

## Project structure
```
resume_rag/
├── app.py                 # Streamlit UI
├── rag_pipeline.py        # Core RAG logic (ingest, section-split, chunk, embed, store, retrieve, generate)
├── eval.py                # Evaluation harness (retrieval + answer quality)
├── eval_questions.json    # Test questions — edit to match your resume
├── eval_results.json      # Generated by eval.py after a run
├── requirements.txt
├── .env.example
└── README.md
```
