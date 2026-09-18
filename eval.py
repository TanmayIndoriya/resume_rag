"""
eval.py
-------
A lightweight evaluation harness for the Resume RAG pipeline.

There's no public "ground truth" dataset for a personal resume, so evaluation here
works by keyword matching: for each test question in eval_questions.json, you list
a few keywords you'd expect a correct answer to contain (a company name, a skill,
a degree, etc). The script then checks whether those keywords show up in:

  1. the RETRIEVED chunks  -> measures retrieval quality on its own, independent
                               of the LLM (did the vector search even find the
                               right part of the resume?)
  2. the GENERATED answer  -> measures the full pipeline end-to-end (did the LLM
                               actually use that context correctly?)

Separating these two scores is the point: if retrieval hits but the answer misses,
the bug is in prompting/generation. If retrieval itself misses, the bug is in
chunking or embeddings. That distinction is exactly what you'd want to debug in a
real RAG system.

Usage:
    1. Edit RESUME_PATH below to point at your resume PDF (or place it at
       data/resume.pdf, the default).
    2. Edit eval_questions.json with questions + keywords specific to YOUR resume.
    3. Run: python eval.py
"""

import json
import time
from pathlib import Path

from rag_pipeline import ResumeRAG

EVAL_FILE = Path(__file__).parent / "eval_questions.json"
RESULTS_FILE = Path(__file__).parent / "eval_results.json"
RESUME_PATH = Path(__file__).parent / "data" / "resume.pdf"  # <-- point this at your resume


def load_eval_cases():
    with open(EVAL_FILE, "r") as f:
        return json.load(f)


def keyword_hit(text: str, keywords) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in keywords)


def run_eval():
    if not RESUME_PATH.exists():
        print(f"Resume not found at {RESUME_PATH}.")
        print("Either place your resume PDF there, or edit RESUME_PATH in eval.py.")
        return

    cases = load_eval_cases()
    print(f"Loaded {len(cases)} test questions from {EVAL_FILE.name}")
    print(f"Indexing resume: {RESUME_PATH}\n")

    rag = ResumeRAG()
    num_chunks = rag.index_resume(str(RESUME_PATH))
    print(f"Indexed into {num_chunks} chunks.\n")

    retrieval_hits = 0
    answer_hits = 0
    results = []

    for case in cases:
        question = case["question"]
        keywords = case["expected_keywords"]

        start = time.time()
        result = rag.ask(question)
        elapsed = time.time() - start

        retrieved_text = " ".join(chunk["text"] for chunk in result.get("sources", []))
        r_hit = keyword_hit(retrieved_text, keywords)
        a_hit = keyword_hit(result["answer"], keywords)

        retrieval_hits += r_hit
        answer_hits += a_hit

        results.append({
            "question": question,
            "expected_keywords": keywords,
            "retrieval_hit": r_hit,
            "answer_hit": a_hit,
            "latency_sec": round(elapsed, 2),
            "mode": result.get("mode"),
            "answer": result["answer"],
        })

        status = "PASS" if a_hit else ("PARTIAL" if r_hit else "FAIL")
        print(f"[{status:7s}] ({elapsed:.2f}s) {question}")
        if not a_hit:
            print(f"          expected keywords: {keywords}")
            print(f"          got: {result['answer'][:150]}...")

    n = len(cases)
    print("\n--- Summary ---")
    print(f"Retrieval hit rate: {retrieval_hits}/{n} ({100 * retrieval_hits / n:.0f}%)")
    print(f"Answer hit rate:    {answer_hits}/{n} ({100 * answer_hits / n:.0f}%)")
    if retrieval_hits > answer_hits:
        print(
            "\nNote: retrieval found the right content more often than the final "
            "answer used it correctly — worth reviewing the generation prompt."
        )
    elif retrieval_hits < n:
        print(
            "\nNote: some questions never retrieved the right chunk — worth "
            "reviewing chunking strategy or resume section labels."
        )

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {RESULTS_FILE.name}")


if __name__ == "__main__":
    run_eval()
