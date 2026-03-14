# src/llm_qa.py — LLM Question Answering with RAG Context
import os
import time
import openai
from dataclasses import dataclass


@dataclass
class QAResponse:
    answer: str
    sources: list[dict]
    tokens_in: int
    tokens_out: int
    latency_ms: float
    model: str


# ─────────────────────────────────────────────
# GROQ CLIENT
# ─────────────────────────────────────────────
def get_groq_client() -> openai.OpenAI:
    return openai.OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert AI research assistant specializing in analyzing academic papers.

You answer questions based STRICTLY on the provided context from the research paper(s).

RULES:
1. Only use information from the provided context — never fabricate or assume facts.
2. If the answer is not in the context, say: "This information is not covered in the provided paper sections."
3. Always be specific — cite actual details, numbers, and findings from the paper.
4. Structure your answers clearly:
   - Start with a direct answer
   - Follow with supporting evidence from the paper
   - End with your analytical insight if relevant
5. For technical questions, explain concepts clearly without oversimplifying.
6. If asked to compare or summarize, be comprehensive but concise.

You are talking to a researcher who wants deep, accurate insights — not surface-level summaries.
"""


# ─────────────────────────────────────────────
# BUILD CONTEXT FROM RETRIEVED CHUNKS
# ─────────────────────────────────────────────
def build_context(retrieved_chunks: list[dict], max_tokens: int = 3000) -> str:
    """
    Build a clean context string from retrieved chunks.
    Deduplicates and orders by relevance score.
    """
    # Sort by score descending
    sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["score"], reverse=True)

    context_parts = []
    total_words = 0

    for i, chunk in enumerate(sorted_chunks):
        text = chunk["text"].strip()
        words = len(text.split())

        if total_words + words > max_tokens:
            break

        context_parts.append(
            f"[Source {i+1} | Paper: {chunk['paper_title']} | Relevance: {chunk['score']}]\n{text}"
        )
        total_words += words

    return "\n\n---\n\n".join(context_parts)


# ─────────────────────────────────────────────
# MAIN QA FUNCTION
# ─────────────────────────────────────────────
def answer_question(
    question: str,
    retrieved_chunks: list[dict],
    client: openai.OpenAI,
    model: str = "llama-3.1-8b-instant"
) -> QAResponse:
    """
    Generate an answer using RAG context + LLM.
    """
    context = build_context(retrieved_chunks)

    user_message = f"""Based on the following context from the research paper(s), answer this question:

QUESTION: {question}

CONTEXT:
{context}

Provide a detailed, accurate answer based strictly on the context above."""

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    latency_ms = (time.time() - start) * 1000

    answer = response.choices[0].message.content.strip()
    usage = response.usage

    return QAResponse(
        answer=answer,
        sources=retrieved_chunks,
        tokens_in=usage.prompt_tokens,
        tokens_out=usage.completion_tokens,
        latency_ms=latency_ms,
        model=model
    )


# ─────────────────────────────────────────────
# PAPER SUMMARY GENERATOR
# ─────────────────────────────────────────────
def generate_paper_summary(paper_text: str, client: openai.OpenAI) -> dict:
    """Generate structured summary of a research paper."""
    prompt = f"""Analyze this research paper and return a JSON object with these exact keys:
{{
  "title_detected": "paper title if you can detect it",
  "one_liner": "one sentence describing what this paper does",
  "problem": "what problem does this paper solve",
  "approach": "what method/approach do they use",
  "key_findings": ["finding 1", "finding 2", "finding 3"],
  "limitations": ["limitation 1", "limitation 2"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
  "difficulty": "Beginner / Intermediate / Advanced",
  "field": "e.g. NLP, Computer Vision, Reinforcement Learning, etc."
}}

Return ONLY valid JSON. No markdown, no explanation.

Paper text (first 3000 words):
{" ".join(paper_text.split()[:3000])}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing research papers. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800,
        )
        import json
        raw = response.choices[0].message.content.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except Exception as e:
        return {
            "title_detected": "Unknown",
            "one_liner": "Summary unavailable",
            "problem": "",
            "approach": "",
            "key_findings": [],
            "limitations": [],
            "keywords": [],
            "difficulty": "Unknown",
            "field": "Unknown"
        }