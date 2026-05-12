"""
src/llm/prompts.py — Prompt templates for the RAG pipeline.

Evidence blocks and instructions are composed into the user message.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Minimal; evidence and rules live in the user message
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You answer strictly from the Retrieved Evidence in the user message (paper/document text).
Follow the Instructions exactly. Do not substitute generic machine-learning textbook explanations for what the evidence says.
If the evidence does not state it, do not infer it."""

# ─────────────────────────────────────────────────────────────────────────────
# USER MESSAGE — Retrieved Evidence + Instructions + Question
# ─────────────────────────────────────────────────────────────────────────────
RAG_USER_MESSAGE_TEMPLATE = """Retrieved Evidence:

{evidence_blocks}

Instructions:

* Answer ONLY using retrieved evidence (quoted sentences above).
* Extract definitions, mechanisms, and thresholds directly from that text—prefer short quotes or tight paraphrases tied to those sentences.
* Do NOT use outside knowledge or generic ML explanations not stated in the evidence.
* Do NOT expand acronyms unless explicitly present in evidence.
* Avoid summarizing with vague phrases (e.g. "typically", "often used in deep learning") unless the evidence uses similar wording.
* If evidence is insufficient, say:
  "The retrieved documents do not contain enough information."

Question:
{question}
"""


def build_rag_user_message(evidence_blocks: str, question: str) -> str:
    """Compose user turn: labeled evidence blocks, instructions, then the question."""
    return RAG_USER_MESSAGE_TEMPLATE.format(
        evidence_blocks=evidence_blocks.strip() or '(No evidence blocks.)',
        question=question.strip(),
    )


def format_evidence_chunk_block(chunk_index: int, page_display: str, quoted_body: str) -> str:
    """One evidence block: [Chunk N | Page X] plus quoted text."""
    safe = quoted_body.replace('"', "'")
    return f'[Chunk {chunk_index} | Page {page_display}]\n"{safe}"'


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK RESPONSE — When retrieval finds no passing chunks
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_RESPONSE = (
    "I could not find relevant information in the uploaded documents.\n\n"
    "Please try:\n"
    "  • Rephrasing your question with different keywords\n"
    "  • Uploading documents that cover this topic\n"
    "  • Checking if the relevant documents have been ingested"
)

# Programmatic weak-evidence / grounding failure (exact wording per requirements)
INSUFFICIENT_DOCUMENT_EVIDENCE = "The retrieved documents do not contain enough information."

# Legacy / model variations accepted as abstention (post-check only)
INSUFFICIENT_DISCLAIMER_MARKERS = (
    "the retrieved documents do not contain enough information",
    "do not contain enough information to answer confidently",
    "not enough information",
)
