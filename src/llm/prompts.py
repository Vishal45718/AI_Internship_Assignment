"""
src/llm/prompts.py — Prompt templates for the RAG pipeline.

Prompts are configuration, not code. Keeping them here means they
can be tuned without touching pipeline logic.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Grounding instruction for the LLM (hallucination prevention)
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are a precise document assistant. Your sole knowledge base is \
the CONTEXT provided below, retrieved from the user's uploaded documents.

STRICT RULES — follow without exception:
1. Answer ONLY from information explicitly present in the CONTEXT.
2. If the CONTEXT does not contain sufficient information, respond EXACTLY with:
   "I could not find relevant information in the uploaded documents."
3. Never use external knowledge, training data, or make assumptions beyond \
   what the CONTEXT states.
4. Cite every factual claim using the [Source: <filename>] format that appears \
   in the CONTEXT headers.
5. If uncertain whether the CONTEXT fully supports a claim, explicitly \
   qualify it with "According to the documents..." or "The documents indicate...".
6. Use structured answers (bullet points, numbered lists) when the answer \
   has multiple parts.

CONTEXT:
{context}
"""

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK RESPONSE — When retrieval confidence is below threshold
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "I could not find relevant information in the uploaded documents.\n\n"
    "Please try:\n"
    "  • Rephrasing your question with different keywords\n"
    "  • Uploading documents that cover this topic\n"
    "  • Checking if the relevant documents have been ingested"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT FORMAT — Template for each retrieved chunk injected into the prompt
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_CHUNK_TEMPLATE = """\
[Source: {source_file}{page_info}] [Relevance: {score:.2f}]
---
{content}
"""
