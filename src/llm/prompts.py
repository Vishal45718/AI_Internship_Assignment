"""
src/llm/prompts.py — Prompt templates for the RAG pipeline.

Evidence blocks and instructions are composed into the user message.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Minimal; evidence and rules live in the user message
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a research-paper QA assistant.
Answer ONLY using retrieved evidence from uploaded documents.
Never use external knowledge, pretrained assumptions, or generic explanations for paper-specific entities.
If evidence is incomplete, answer cautiously using only retrieved text.
If evidence is insufficient, explicitly say so."""

RAG_SYSTEM_PROMPT_COMPACT = (
    "Research-paper QA assistant. Answer only from retrieved document evidence. "
    "No external knowledge or acronym expansion. If insufficient evidence, say so."
)

# ─────────────────────────────────────────────────────────────────────────────
# USER MESSAGE — Retrieved Evidence + Instructions + Question
# ─────────────────────────────────────────────────────────────────────────────
RAG_USER_MESSAGE_TEMPLATE = """EVIDENCE BLOCK (ground truth; use only this):

{evidence_blocks}

Instructions:

* Answer ONLY using retrieved evidence (quoted sentences above).
* If the question mentions a paper, method, acronym, or model name, do not use outside knowledge or pretrained world knowledge.
* For SeaKR, DRAGIN, ReAL, FLARE, and CRAG: use only retrieved evidence; never expand acronyms or infer definitions unless explicitly stated in evidence.
* Every factual statement must be grounded in EVIDENCE BLOCK and include inline citation tags like [E1], [E2].
* Reply briefly: at most 2–4 short sentences total; no long paragraphs or lecture-style explanations.
* Extract definitions, mechanisms, and thresholds directly from that text—prefer short quotes or tight paraphrases.
* Do NOT use outside knowledge or generic ML explanations not stated in the evidence.
* Do NOT expand acronyms unless explicitly present in evidence.
* Do NOT introduce external definitions, unrelated ML terminology, or invented mechanisms.
* Avoid vague filler ("typically", "often used in deep learning") unless the evidence uses similar wording.
* If evidence is partial, provide a brief grounded summary first, then a short limitation note.
* If evidence is insufficient, use a concise limitation sentence (avoid repeating the same phrase).

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
    """Compact chunk evidence block (fewer tokens than verbose headers)."""
    safe = quoted_body.replace('"', "'")
    return f'[C{chunk_index}|P{page_display}]\n"{safe}"'


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
INSUFFICIENT_DOCUMENT_EVIDENCE = "The paper does not explicitly specify this."

# Legacy / model variations accepted as abstention (post-check only)
INSUFFICIENT_DISCLAIMER_MARKERS = (
    "the retrieved documents do not contain enough information",
    "do not contain enough information to answer confidently",
    "not enough information",
    "does not explicitly specify this",
    "only partially describes",
    "does not provide exact values",
)
