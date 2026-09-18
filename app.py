"""
app.py
------
Streamlit front-end for the Resume RAG project.

Run with:
    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import ResumeRAG

load_dotenv()

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="centered")

st.title("📄 Resume RAG Assistant")
st.caption(
    "A Retrieval-Augmented Generation demo: upload a resume PDF, and ask "
    "questions about it. Retrieval runs locally (sentence-transformers + ChromaDB); "
    "answer generation uses a free Groq/Llama API call if a key is configured."
)

if "rag" not in st.session_state:
    st.session_state.rag = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.header("1. Upload resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

    if uploaded_file is not None:
        if st.button("Build / Rebuild Index", type="primary"):
            with st.spinner("Extracting text, chunking, embedding, and indexing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                rag = ResumeRAG()
                num_chunks = rag.index_resume(tmp_path)
                st.session_state.rag = rag
                st.session_state.chat_history = []
                os.unlink(tmp_path)

            st.success(f"Indexed resume into {num_chunks} chunks. Ask away!")

    st.divider()
    if os.getenv("GROQ_API_KEY"):
        st.success("Groq API key detected — answers will be LLM-generated.")
    else:
        st.warning(
            "No GROQ_API_KEY found. App will still run and retrieve relevant "
            "resume chunks, but won't generate a natural-language answer.\n\n"
            "Get a free key at console.groq.com/keys and add it to a `.env` file."
        )

st.header("2. Ask questions about the resume")

if st.session_state.rag is None:
    st.info("Upload a resume PDF and click 'Build / Rebuild Index' in the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    query = st.chat_input("e.g. What projects has this candidate worked on?")
    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                result = st.session_state.rag.ask(query)
                st.markdown(result["answer"])
                with st.expander("Retrieved context (what the model saw)"):
                    for i, chunk in enumerate(result.get("sources", []), start=1):
                        section = chunk.get("section", "General")
                        text = chunk.get("text", "")
                        st.markdown(f"**Chunk {i} — _{section}_:** {text}")

        st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})
