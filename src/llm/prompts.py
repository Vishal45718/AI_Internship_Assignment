"""
src/llm/prompts.py — Prompt templates for the RAG pipeline.

Prompts are configuration, not code. Keeping them here means they
can be tuned without touching pipeline logic.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Grounding instruction for the LLM (hallucination prevention)
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are a research assistant answering strictly from retrieved context.

Rules:
- Use ONLY retrieved context.
- If relevant context exists, NEVER say: "I could not find relevant information."
- Prefer explanatory sentences over isolated mentions.
- If answer is partially available, provide the partial answer.
- Do not hallucinate missing details.
- Prioritize: definitions, mechanisms, thresholds, methodology, comparisons.
- Cite factual claims using source headers present in context.

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
