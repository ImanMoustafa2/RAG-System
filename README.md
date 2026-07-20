# SDS Hybrid-RAG Assistant 🧪

A demo-ready RAG (Retrieval-Augmented Generation) system built with Streamlit, 
designed to answer questions about 4 chemical Safety Data Sheets 
(Ethanol, NaOH, HCl, H2SO4), using the following exact pipeline:
Chunking → RecursiveCharacterTextSplitter (size=500, overlap=100)
Embeddings → BAAI/bge-small-en-v1.5 (same model for indexing & querying)
Index → FAISS HNSW (switches to IVF-PQ only when size/memory grows)
Retrieval → Hybrid: Dense (FAISS) + BM25 (rank_bm25)
Fusion → Reciprocal Rank Fusion (RRF, k=60)
Reranker → Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
Generation → Groq LLM (llama-3.3-70b-versatile) + a prompt enforcing source citation
Extras → Metadata filters, Query Rewriting, SQLite cache, JSONL monitoring
## Project Structure
sds_rag/
├── app.py # Streamlit UI (Chat / Monitoring / About)
├── requirements.txt
├── data/ # The four SDS files (PDF)
├── src/
│ ├── config.py # All configurable settings
│ ├── ingestion.py # PDF loading + metadata extraction + chunking
│ ├── embeddings.py # Embeddings model (singleton)
│ ├── indexing.py # FAISS HNSW index
│ ├── bm25_index.py # BM25 index
│ ├── retrieval.py # Hybrid retrieval + RRF + metadata filters
│ ├── reranker.py # Cross-encoder reranking
│ ├── query_rewriter.py # Rewrites the question before searching
│ ├── generation.py # Prompt construction and Groq calls
│ ├── cache.py # SQLite query cache
│ ├── monitoring.py # Logs timing for each stage (JSONL)
│ ├── pipeline.py # Orchestrator that ties all stages together
│ └── build_index.py # Script to build indexes from data/
├── indexes/ # (auto-created) saved indexes
├── .cache/ # (auto-created) query cache
└── logs/ # (auto-created) monitoring logs
## Running the Project

```bash
# 1) Virtual environment (optional but recommended)
python3 -m venv venv && source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Groq API key (free from console.groq.com)
export GROQ_API_KEY="your_key_here"

# 4) Build the indexes for the first time (or use the "Rebuild indexes" button in the UI)
python -m src.build_index

# 5) Run the app
streamlit run app.py
```

You can also enter the Groq API key directly from the sidebar in the UI 
instead of using an environment variable.

## Key Design Notes (for discussion/presentation)

- **Why HNSW instead of IVF-PQ?** The current dataset is very small (tens of
  chunks), so HNSW gives near-perfect retrieval accuracy and high speed
  without a training step. Switching to IVF-PQ is only justified once the
  index grows to millions of vectors or memory becomes constrained
  (`config.IVF_PQ_SWITCH_THRESHOLD_VECTORS`).
- **Why RRF instead of merging scores directly?** BM25 scores and cosine
  similarity scores live on incomparable scales; RRF relies only on rank
  rather than the raw score value, giving a stable fusion without manual
  normalization.
- **Why Cross-Encoder after fusion instead of at initial retrieval?** The
  Cross-Encoder is highly accurate but slow (it computes a full interaction
  between the query and the chunk), making it impractical to run over the
  entire corpus. It's therefore only applied to the top-K results coming out
  of the fusion step.
- **Source citations:** The prompt in `generation.py` requires the model to
  reference `[source_file, section]` after every factual sentence, and
  refuses to answer using anything outside the retrieved context.
- **Cache (`cache.py`):** A simple SQLite cache keyed by a hash of
  (question + filters), with a 24-hour TTL — saves time/cost on repeated
  common questions.
- **Monitoring (`monitoring.py`):** Every query is logged as a JSON line
  containing the timing of each stage (rewrite/retrieval/rerank/generation),
  the number of sources, and the cache hit rate — displayed in the
  "📊 Monitoring" tab within the app.

## Extending the Dataset

To add a new SDS document, place the PDF file in `data/` then click the
"🔄 (Re)build indexes" button in the sidebar, or run `python -m src.build_index`
from the terminal.
