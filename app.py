"""
Streamlit UI for the Chemical SDS Hybrid-RAG assistant.

Run with:
    export GROQ_API_KEY=your_key_here
    streamlit run app.py
"""
import os

import pandas as pd
import streamlit as st

from src import config
from src.bm25_index import BM25Index
from src.indexing import HNSWDenseIndex
from src.ingestion import build_corpus
from src.monitoring import load_events
from src.pipeline import RAGPipeline

st.set_page_config(page_title="SDS Hybrid-RAG Assistant", page_icon="🧪", layout="wide")

# ---------------------------------------------------------------------------
# Visual identity — lab / chemical-safety theme
#   Primary:  deep teal   #0F4C5C  (lab bench, professional, calm)
#   Accent:   hazard amber #F2A104 (GHS warning color, used sparingly)
#   Ink:      slate        #1A2332
#   Surface:  off-white    #F7F9FA
#   Type:     IBM Plex Sans (UI) / IBM Plex Mono (data, citations, scores)
# ---------------------------------------------------------------------------
st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
    --ink: #1A2332;
    --teal: #0F4C5C;
    --teal-dark: #0A3540;
    --amber: #F2A104;
    --amber-soft: #FFF4DE;
    --surface: #F7F9FA;
    --line: #E3E8EB;
}

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
code, pre, .mono { font-family: 'IBM Plex Mono', monospace !important; }

/* Hide default Streamlit chrome for a cleaner look */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ---- Hero banner ---- */
.hero {
    background: linear-gradient(120deg, var(--teal) 0%, var(--teal-dark) 100%);
    border-radius: 14px;
    padding: 2rem 2.25rem;
    margin-bottom: 1.5rem;
    color: #fff;
    box-shadow: 0 8px 24px rgba(15,76,92,0.18);
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 0.35rem 0;
    color: #fff;
    letter-spacing: -0.01em;
}
.hero p.subtitle {
    margin: 0 0 1rem 0;
    color: #D7E7EA;
    font-size: 0.95rem;
}
.hero .credit {
    color: #B7D3D8;
    font-size: 0.82rem;
    margin-top: 0.75rem;
}

/* Pipeline chip row */
.chip-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
.chip {
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.28);
    color: #fff;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.76rem;
    font-family: 'IBM Plex Mono', monospace;
    white-space: nowrap;
}
.chip .arrow { color: var(--amber); margin-inline-start: 0.4rem; }

/* ---- Source / citation cards ---- */
.source-card {
    background: var(--surface);
    border: 1px solid var(--line);
    border-left: 4px solid var(--amber);
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
}
.source-card .src-head {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.35rem;
}
.source-card .src-name { font-weight: 600; color: var(--teal-dark); font-size: 0.88rem; }
.source-card .src-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: #8A93A3;
}
.source-card .src-section {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--teal);
    background: var(--amber-soft);
    display: inline-block;
    padding: 0.1rem 0.5rem;
    border-radius: 5px;
    margin-bottom: 0.4rem;
}
.source-card .src-text {
    font-size: 0.85rem;
    color: #3B4656;
    line-height: 1.5;
}

/* ---- Timing strip ---- */
.timing-strip {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    color: #8A93A3;
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.4rem 0.8rem;
    display: inline-block;
    margin-top: 0.4rem;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] h1 {
    color: var(--teal-dark);
    font-size: 1.15rem;
}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab"] { font-weight: 600; }
.stTabs [aria-selected="true"] { color: var(--teal) !important; }

/* ---- Buttons ---- */
.stButton > button {
    border-radius: 8px;
    border: 1px solid var(--teal);
    color: var(--teal-dark);
    font-weight: 600;
}
.stButton > button:hover {
    background: var(--teal);
    color: #fff;
    border-color: var(--teal);
}

/* ---- Metric cards ---- */
div[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 0.8rem 1rem;
}
</style>
""",
    unsafe_allow_html=True,
)


def render_source_card(s: dict) -> str:
    """Return an HTML card for one retrieved chunk."""
    return f"""
<div class="source-card">
    <div class="src-head">
        <span class="src-name">📄 {s['source_file']}</span>
        <span class="src-score">score {s['score']:.4f}</span>
    </div>
    <span class="src-section">{s['section']}</span>
    <div class="src-text">{s['text'][:400]}...</div>
</div>
"""


def render_sources(sources: list) -> None:
    with st.expander(f"📎 Sources · {len(sources)} retrieved chunk(s)"):
        for s in sources:
            st.markdown(render_source_card(s), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------
st.sidebar.title("🧪 SDS RAG — Settings")

api_key_input = st.sidebar.text_input(
    "GROQ_API_KEY", value=os.environ.get("GROQ_API_KEY", ""), type="password"
)
if api_key_input:
    os.environ["GROQ_API_KEY"] = api_key_input
    config.GROQ_API_KEY = api_key_input

use_query_rewrite = st.sidebar.checkbox("Enable query rewriting", value=config.ENABLE_QUERY_REWRITE_DEFAULT)
use_cache = st.sidebar.checkbox("Enable caching", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Metadata filters")
chemical_filter = st.sidebar.selectbox(
    "Chemical",
    ["(all)", "Ethanol, Absolute", "Sodium Hydroxide (Caustic Soda)", "Hydrochloric Acid, 37%", "Sulfuric Acid, 98%"],
)
section_filter = st.sidebar.text_input("Section contains (optional, e.g. 'FIRST AID')")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 (Re)build indexes from /data"):
    with st.spinner("Building corpus + FAISS HNSW + BM25 indexes..."):
        corpus = build_corpus()
        dense = HNSWDenseIndex()
        dense.build(corpus)
        dense.save()
        bm25 = BM25Index()
        bm25.build(corpus)
        bm25.save()
    st.sidebar.success(f"Indexed {len(corpus)} chunks.")
    st.rerun()

# ---------------------------------------------------------------------------
# Load pipeline (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading indexes & models...")
def load_pipeline():
    return RAGPipeline()


indexes_ready = HNSWDenseIndex.exists() and BM25Index.exists()

# ---------------------------------------------------------------------------
# Hero banner (shown above the tabs, on every page)
# ---------------------------------------------------------------------------
PIPELINE_STAGES = [
    "Chunking", "Embeddings", "FAISS HNSW", "BM25", "RRF Fusion",
    "Cross-Encoder Rerank", "Groq LLM",
]
chips_html = "".join(
    f'<span class="chip">{stage}{" <span class=\'arrow\'>→</span>" if i < len(PIPELINE_STAGES) - 1 else ""}</span>'
    for i, stage in enumerate(PIPELINE_STAGES)
)
st.markdown(
    f"""
<div class="hero">
    <h1>🧪 Chemical SDS Assistant</h1>
    <p class="subtitle">Hybrid Retrieval-Augmented Generation for Safety Data Sheets — grounded answers, every claim cited to its source.</p>
    <div class="chip-row">{chips_html}</div>
    <div class="credit">👩‍💻 Developed by Eman Moustafa &amp; Aya Shaaban</div>
</div>
""",
    unsafe_allow_html=True,
)

tab_chat, tab_monitoring, tab_about = st.tabs(["💬 Chat", "📊 Monitoring", "ℹ️ About"])

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
with tab_chat:
    if not indexes_ready:
        st.warning("No index found yet. Click **'(Re)build indexes from /data'** in the sidebar first.")
    else:
        if "history" not in st.session_state:
            st.session_state.history = []

        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources(msg["sources"])

        query = st.chat_input("اسأل عن أي مادة كيميائية في الداتا... (e.g. What are the first aid measures for sulfuric acid?)")
        if query:
            st.session_state.history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            filters = {}
            if chemical_filter != "(all)":
                filters["chemical_name"] = chemical_filter
            # section_filter is a substring match; handled after retrieval since
            # metadata filter is exact-match, so we do a soft post-filter instead.

            with st.chat_message("assistant"):
                with st.spinner("Retrieving + reranking + generating..."):
                    pipeline = load_pipeline()
                    result = pipeline.answer(
                        query,
                        use_query_rewrite=use_query_rewrite,
                        metadata_filters=filters or None,
                        use_cache=use_cache,
                    )

                sources = result.sources
                if section_filter:
                    sources = [s for s in sources if section_filter.lower() in (s["section"] or "").lower()] or sources

                st.markdown(result.answer)
                if result.from_cache:
                    st.markdown('<div class="timing-strip">⚡ served from cache</div>', unsafe_allow_html=True)
                elif result.timings_ms:
                    st.markdown(
                        f'<div class="timing-strip">⏱️ rewrite {result.timings_ms.get("query_rewrite_ms", 0):.0f}ms · '
                        f'retrieval {result.timings_ms.get("retrieval_ms", 0):.0f}ms · '
                        f'rerank {result.timings_ms.get("rerank_ms", 0):.0f}ms · '
                        f'generation {result.timings_ms.get("generation_ms", 0):.0f}ms</div>',
                        unsafe_allow_html=True,
                    )
                if sources:
                    render_sources(sources)

            st.session_state.history.append(
                {"role": "assistant", "content": result.answer, "sources": sources}
            )

# ---------------------------------------------------------------------------
# Monitoring tab
# ---------------------------------------------------------------------------
with tab_monitoring:
    st.subheader("📊 Pipeline Monitoring")
    df = load_events()
    if df.empty:
        st.info("No queries logged yet.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total queries", len(df))
        if "cache_hit" in df.columns:
            hit_rate = df["cache_hit"].mean() * 100
            c2.metric("Cache hit rate", f"{hit_rate:.1f}%")
        if "retrieval_ms" in df.columns:
            c3.metric("Avg retrieval latency", f"{df['retrieval_ms'].mean():.0f} ms")

        st.markdown("#### Latency over time")
        latency_cols = [c for c in ["query_rewrite_ms", "retrieval_ms", "rerank_ms", "generation_ms"] if c in df.columns]
        if latency_cols:
            st.line_chart(df.set_index("datetime")[latency_cols])

        st.markdown("#### Recent queries")
        show_cols = [c for c in ["datetime", "query", "rewritten_query", "num_sources", "cache_hit"] if c in df.columns]
        st.dataframe(df[show_cols].sort_values("datetime", ascending=False), use_container_width=True)

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.subheader("ℹ️ About this pipeline")
    st.markdown(
        """
**Architecture**

1. **Chunking** — `RecursiveCharacterTextSplitter` (size=500, overlap=100) with section/CAS/chemical-name metadata tagging.
2. **Embeddings** — `BAAI/bge-small-en-v1.5` (Sentence-Transformers), same model used for indexing and querying.
3. **Index** — FAISS `IndexHNSWFlat` (M=32, efConstruction=200, efSearch=64), cosine via normalized inner product.
   Switch to IVF-PQ only once the corpus exceeds ~1M vectors or the flat/HNSW index no longer fits comfortably in RAM.
4. **Retrieval** — Hybrid: dense (FAISS HNSW) + sparse (BM25 Okapi), each top-K independently.
5. **Fusion** — Reciprocal Rank Fusion (RRF, k=60) merges the two ranked lists without needing comparable score scales.
6. **Reranker** — Cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring the fused candidates.
7. **Generation** — Groq-hosted LLM (`llama-3.3-70b-versatile`) with a grounded, citation-enforcing system prompt.
8. **Extras** — metadata filters (chemical / section), LLM-based query rewriting, SQLite query cache, JSONL monitoring log.

**👩‍💻 Developed by:** Eman Moustafa & Aya Shaaban
        """
    )
