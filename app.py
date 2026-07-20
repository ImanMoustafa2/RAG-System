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

tab_chat, tab_monitoring, tab_about = st.tabs(["💬 Chat", "📊 Monitoring", "ℹ️ About"])

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
with tab_chat:
    st.title("🧪 Chemical SDS Assistant")
    st.caption("Hybrid RAG (Dense HNSW + BM25 → RRF → Cross-Encoder Rerank → Grounded LLM Answer)")

    if not indexes_ready:
        st.warning("No index found yet. Click **'(Re)build indexes from /data'** in the sidebar first.")
    else:
        if "history" not in st.session_state:
            st.session_state.history = []

        for msg in st.session_state.history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("📎 Sources / retrieved chunks"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"**{s['source_file']}** — {s['section']}  \n"
                                f"_score: {s['score']:.4f}_\n\n> {s['text'][:400]}..."
                            )

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
                    st.caption("⚡ served from cache")
                elif result.timings_ms:
                    st.caption(
                        f"⏱️ rewrite {result.timings_ms.get('query_rewrite_ms', 0):.0f}ms · "
                        f"retrieval {result.timings_ms.get('retrieval_ms', 0):.0f}ms · "
                        f"rerank {result.timings_ms.get('rerank_ms', 0):.0f}ms · "
                        f"generation {result.timings_ms.get('generation_ms', 0):.0f}ms"
                    )
                if sources:
                    with st.expander("📎 Sources / retrieved chunks"):
                        for s in sources:
                            st.markdown(
                                f"**{s['source_file']}** — {s['section']}  \n"
                                f"_score: {s['score']:.4f}_\n\n> {s['text'][:400]}..."
                            )

            st.session_state.history.append(
                {"role": "assistant", "content": result.answer, "sources": sources}
            )

# ---------------------------------------------------------------------------
# Monitoring tab
# ---------------------------------------------------------------------------
with tab_monitoring:
    st.title("📊 Pipeline Monitoring")
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

        st.subheader("Latency over time")
        latency_cols = [c for c in ["query_rewrite_ms", "retrieval_ms", "rerank_ms", "generation_ms"] if c in df.columns]
        if latency_cols:
            st.line_chart(df.set_index("datetime")[latency_cols])

        st.subheader("Recent queries")
        show_cols = [c for c in ["datetime", "query", "rewritten_query", "num_sources", "cache_hit"] if c in df.columns]
        st.dataframe(df[show_cols].sort_values("datetime", ascending=False), use_container_width=True)

# ---------------------------------------------------------------------------
# About tab
# ---------------------------------------------------------------------------
with tab_about:
    st.title("ℹ️ About this pipeline")
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
        """
    )
