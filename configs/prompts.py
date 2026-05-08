"""
configs/prompts.py — All prompt templates as named constants.

Design principle: prompts are configuration, not code.
Keeping them here means non-engineers can tune them without touching logic.
Never inline prompts in application code — always import from here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT — Core grounding instruction given to the LLM for every query.
# This is the primary hallucination-prevention mechanism.
# ─────────────────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are a precise document intelligence assistant. Your sole knowledge base is \
the CONTEXT provided below, retrieved from the user's indexed documents.

STRICT RULES — follow without exception:
1. Answer ONLY from information explicitly present in the CONTEXT.
2. If the CONTEXT does not contain sufficient information, respond with the \
   exact fallback phrase specified and do NOT invent or infer facts.
3. Never use external knowledge, training data, or make assumptions beyond \
   what the CONTEXT states.
4. Cite every factual claim using the [Source: <filename>] format that appears \
   in the CONTEXT headers.
5. If you are uncertain whether the CONTEXT fully supports a claim, explicitly \
   qualify it with "According to the documents..." or "The documents indicate...".
6. Structured answers (bullet points, numbered lists) are preferred when the \
   answer has multiple parts.

CONTEXT:
{context}
"""

# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK RESPONSE — Returned when retrieval confidence is below threshold.
# This is the hallucination firewall. Do not soften or abbreviate it.
# ─────────────────────────────────────────────────────────────────────────────
FALLBACK_RESPONSE_TEMPLATE = """\
I could not find sufficient information in the indexed documents to answer \
this question confidently.

**Query:** {query}

**Reason:** {reason}

**What I do have indexed:**
{document_summary}

**Suggestion:** {suggestion}
"""

FALLBACK_REASONS = {
    "low_confidence": (
        "The retrieved document chunks had low relevance scores "
        "(below the confidence threshold of {threshold:.0%}), "
        "indicating the answer is likely not present in the indexed documents."
    ),
    "no_results": (
        "No document chunks were retrieved for this query. "
        "The indexed documents may not cover this topic."
    ),
    "empty_index": (
        "No documents have been ingested yet. "
        "Please upload and ingest documents before querying."
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# INTENT CLASSIFICATION — Agentic router prompt.
# ─────────────────────────────────────────────────────────────────────────────
INTENT_CLASSIFIER_PROMPT = """\
Classify the following user query into exactly one of these categories and \
respond with ONLY the category name — no explanation, no punctuation:

Categories:
- factual_lookup    : Direct question with a specific factual answer
- comparison        : Asks to compare two or more things
- summarize_doc     : Asks for a summary of a document or section
- list_extraction   : Asks to list or enumerate items
- out_of_scope      : Question clearly unrelated to documents (weather, sports, general trivia, etc.)
- unclear           : Ambiguous query that needs clarification

Query: {query}
"""

VALID_INTENTS = {
    "factual_lookup",
    "comparison",
    "summarize_doc",
    "list_extraction",
    "out_of_scope",
    "unclear",
}

# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT FORMAT — Template for each retrieved chunk injected into the prompt.
# ─────────────────────────────────────────────────────────────────────────────
CONTEXT_CHUNK_TEMPLATE = """\
[Source: {source_file}{page_info}] [Relevance: {score:.2f}]
---
{content}
"""

# ─────────────────────────────────────────────────────────────────────────────
# CONVERSATION HISTORY — Injected before the user's current query.
# ─────────────────────────────────────────────────────────────────────────────
CONVERSATION_HISTORY_TEMPLATE = """\
CONVERSATION HISTORY (most recent {n} turns):
{history}

"""

# ─────────────────────────────────────────────────────────────────────────────
# OUT-OF-SCOPE RESPONSE — Immediate response for queries flagged by the router.
# ─────────────────────────────────────────────────────────────────────────────
OUT_OF_SCOPE_RESPONSE = """\
This question appears to be outside the scope of the indexed documents. \
I can only answer questions based on the documents that have been ingested \
into the system.

If you believe this question should be answerable from your documents, \
please verify that the relevant documents have been uploaded and indexed.
"""
