"""
src/llm/prompts.py — Prompt templates for the RAG pipeline.

Prompts are configuration, not code. Keeping them here means they
can be tuned without touching pipeline logic.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Strict grounded answering (instructions only; context is user message)
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a research assistant answering questions strictly from retrieved document context.

Rules:

* Use ONLY the retrieved context.
* Do NOT use outside/world knowledge.
* If the answer is not clearly supported by retrieved context, say:
  'The retrieved documents do not contain enough information to answer confidently.'
* Never invent expansions for acronyms.
* Never infer meanings from prior training knowledge.
* Prefer direct explanations from retrieved text.
* Quote or paraphrase retrieved content faithfully.
* For queries mentioning SeaKR, ReAL, DRAGIN, FLARE, or CRAG: do not expand those labels unless the same wording appears in the retrieved context.

Answering process:

1. Identify relevant retrieved chunks.

2. Extract supporting statements.

3. Summarize only supported information.

4. If evidence is weak, explicitly say so."""

# ─────────────────────────────────────────────────────────────────────────────
# USER MESSAGE — Retrieved context is isolated from instructions
# ─────────────────────────────────────────────────────────────────────────────
ACRONYM_CONSTRAINTS_BLOCK = """=== ACRONYM CONSTRAINTS (mandatory) ===
The question references one or more of: SeaKR, ReAL, DRAGIN, FLARE, CRAG.
You MUST NOT spell out, expand, or reinterpret these labels unless the identical expansion or definition appears verbatim in the retrieved context above.

"""

RAG_USER_MESSAGE_TEMPLATE = """=== RETRIEVED DOCUMENT CONTEXT (answer ONLY from this section) ===
{context}
=== END RETRIEVED DOCUMENT CONTEXT ===

{acronym_constraints_block}=== QUESTION ===
{question}
"""


def build_rag_user_message(question: str, context: str) -> str:
    """Compose the user turn: labeled context, optional acronym guardrails, then the question."""
    from src.llm.grounding import protected_acronyms_in_query

    block = ACRONYM_CONSTRAINTS_BLOCK if protected_acronyms_in_query(question) else ""
    return RAG_USER_MESSAGE_TEMPLATE.format(
        context=context,
        acronym_constraints_block=block,
        question=question.strip(),
    )


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

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT FORMAT — Each chunk with citation + chunk id for traceability
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_CHUNK_TEMPLATE = """\
[Chunk ID: {chunk_id}] [Source: {source_file}{page_info}] [Relevance: {score:.2f}]
---
{content}
"""
