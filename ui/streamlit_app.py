"""
ui/streamlit_app.py — Chat-based Streamlit UI for the Agentic RAG System.

Provides:
- Chat interface with conversation history
- Document upload (ingest on the fly)
- Source citations per message
- Retrieval debug expander
- Sidebar with indexed document list and chunk counts
- Loading spinners with step labels
- Graceful empty state

Run: streamlit run ui/streamlit_app.py
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.utils.logging import setup_logging
from configs.settings import get_settings

setup_logging("WARNING")  # Suppress logs in UI mode

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat — Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .main-header { 
        font-size: 2.2rem; 
        font-weight: 800; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .badge-success { background: #1a3a1a; color: #4ade80; border: 1px solid #4ade80; }
    .badge-fallback { background: #3a2f0a; color: #fbbf24; border: 1px solid #fbbf24; }
    .badge-out_of_scope { background: #2a1a1a; color: #f87171; border: 1px solid #f87171; }
    .source-card {
        background: #1e2130;
        border: 1px solid #2d3149;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .confidence-bar {
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ──────────────────────────────────────────────
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]
    if "show_debug" not in st.session_state:
        st.session_state.show_debug = False


init_session_state()


# ── Lazy Service Loading ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI models…")
def load_services():
    """Load all services once and cache them for the session lifetime."""
    from app.core.dependencies import get_rag_agent, get_vector_store, get_ingestion_pipeline
    return {
        "agent": get_rag_agent(),
        "store": get_vector_store(),
        "pipeline": get_ingestion_pipeline(),
    }


try:
    services = load_services()
    agent = services["agent"]
    store = services["store"]
    pipeline = services["pipeline"]
    SERVICES_LOADED = True
except Exception as exc:
    SERVICES_LOADED = False
    LOAD_ERROR = str(exc)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 DocChat")
    st.markdown("*Agentic RAG with hallucination prevention*")
    st.divider()

    # Document Upload
    st.markdown("#### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, Markdown, CSV",
    )

    if uploaded_files and st.button("⚡ Ingest Documents", type="primary", use_container_width=True):
        if SERVICES_LOADED:
            settings = get_settings()
            saved_paths = []
            for f in uploaded_files:
                target = settings.data_dir / f.name
                target.write_bytes(f.read())
                saved_paths.append(str(target))

            with st.spinner("Processing and embedding…"):
                result = pipeline.ingest_files(saved_paths)

            if result.success_count > 0:
                st.success(
                    f"✓ Ingested {result.success_count} file(s), "
                    f"{result.total_chunks} chunks created."
                )
                st.cache_resource.clear()  # Refresh stats
            if result.failed_files:
                st.error(f"Failed: {', '.join(result.failed_files)}")
        else:
            st.error("Services not loaded.")

    st.divider()

    # Indexed Documents List
    st.markdown("#### 📚 Indexed Documents")
    if SERVICES_LOADED:
        sources = store.list_sources()
        total_chunks = store.count()
        if sources:
            st.metric("Total Chunks", total_chunks)
            for src in sources:
                st.markdown(
                    f"<div class='source-card'>📄 <b>{src['source_file']}</b><br>"
                    f"<span style='color:#888'>{src['chunk_count']} chunks</span></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No documents indexed yet.\nUpload files above to get started.")
    else:
        st.error(f"Error: {LOAD_ERROR if not SERVICES_LOADED else ''}")

    st.divider()

    # Settings
    st.markdown("#### ⚙️ Settings")
    st.session_state.show_debug = st.toggle("Show retrieval debug", value=False)
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())[:8]
        st.rerun()

    st.markdown(
        f"<div style='color:#555;font-size:0.75rem;margin-top:1rem'>"
        f"Session: {st.session_state.session_id}</div>",
        unsafe_allow_html=True,
    )


# ── Main Chat Area ────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">DocChat — Ask Your Documents</div>', unsafe_allow_html=True)
st.markdown(
    "Answers are grounded strictly in your indexed documents. "
    "Unknown questions get a structured fallback — no hallucinations."
)

# Empty state
if not st.session_state.messages:
    st.markdown("### 💡 Example Questions")
    examples = [
        "What is the return policy for electronics?",
        "How do I reset my password?",
        "What are the system requirements to install TechCorp?",
        "What wireless earbuds do you offer and what's the price?",
        "What happened in the French Revolution?",  # Should trigger fallback
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": ex})
                st.rerun()

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Render sources for assistant messages
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 Sources ({len(msg['sources'])})"):
                for i, src in enumerate(msg["sources"], 1):
                    page = f" · Page {src['page_number']}" if src.get("page_number") else ""
                    st.markdown(
                        f"**[{i}] {src['source_file']}**{page}  "
                        f"— Relevance: `{src['relevance_score']:.3f}`\n\n"
                        f"*{src['preview']}*"
                    )

        # Debug info
        if (msg["role"] == "assistant"
                and st.session_state.show_debug
                and "debug" in msg):
            with st.expander("🔍 Debug Info"):
                debug = msg["debug"]
                st.json(debug)


# ── Chat Input ────────────────────────────────────────────────────────────────
if query := st.chat_input("Ask a question about your documents…"):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if not SERVICES_LOADED:
            st.error("Services failed to load. Check your configuration and restart.")
        elif store.count() == 0:
            st.warning(
                "⚠️ No documents indexed yet. Please upload documents using the sidebar."
            )
        else:
            status_placeholder = st.empty()

            # Step indicators
            with status_placeholder.container():
                st.markdown("🔍 *Analyzing query…*")

            response_data = {}

            try:
                with st.spinner(""):
                    # Step 1: Classify intent
                    status_placeholder.markdown("🧠 *Classifying intent…*")

                    # Step 2: Retrieve
                    status_placeholder.markdown("🔎 *Retrieving relevant context…*")

                    # Step 3: Rerank + Generate
                    status_placeholder.markdown("✨ *Generating answer…*")

                    response = agent.query(
                        query=query,
                        session_id=st.session_state.session_id,
                    )

                status_placeholder.empty()

                # Status badge
                badge_class = f"badge-{response.status.value}"
                st.markdown(
                    f"<span class='status-badge {badge_class}'>"
                    f"{response.status.value.replace('_', ' ').upper()}"
                    f"</span>",
                    unsafe_allow_html=True,
                )

                # Answer
                st.markdown(response.answer)

                # Confidence bar (for successful answers)
                if response.confidence and response.status.value == "success":
                    st.markdown(
                        f"<div class='confidence-bar' style='width:{response.confidence*100:.0f}%'></div>"
                        f"<span style='font-size:0.75rem;color:#888'>Confidence: {response.confidence:.1%}</span>",
                        unsafe_allow_html=True,
                    )

                # Inline source citations
                if response.sources:
                    source_lines = " · ".join(
                        f"[{src.source_file}]" for src in response.sources
                    )
                    st.markdown(
                        f"<div style='font-size:0.8rem;color:#888;margin-top:0.5rem'>"
                        f"📎 {source_lines}</div>",
                        unsafe_allow_html=True,
                    )

                # Store for history rendering
                sources_data = [
                    {
                        "source_file": s.source_file,
                        "page_number": s.page_number,
                        "relevance_score": s.relevance_score,
                        "preview": s.preview,
                    }
                    for s in response.sources
                ]

                debug_data = {
                    "intent": response.intent,
                    "confidence": response.confidence,
                    "retrieval_strategy": response.retrieval_strategy,
                    "latency_ms": response.latency_ms,
                    "source_count": len(response.sources),
                }

                response_data = {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": sources_data,
                    "debug": debug_data,
                }

            except Exception as exc:
                status_placeholder.empty()
                st.error(f"An error occurred: {exc}")
                response_data = {
                    "role": "assistant",
                    "content": f"Error: {exc}",
                    "sources": [],
                }

            if response_data:
                st.session_state.messages.append(response_data)
