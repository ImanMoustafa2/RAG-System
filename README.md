# 🧪 SDS Hybrid-RAG Assistant

A production-style **Retrieval-Augmented Generation (RAG)** system built with **Streamlit**, designed to answer questions about chemical **Safety Data Sheets (SDS)** — currently covering Ethanol, NaOH, HCl, and H2SO4 — with grounded, cited answers and zero hallucination tolerance.

> Ask a question about a chemical's hazards, handling, or storage, and get an answer backed by exact source citations — not guesses.

---

## ✨ Key Features

- 🔍 **Hybrid Retrieval** — combines dense vector search (FAISS/HNSW) with sparse keyword search (BM25) for both semantic and lexical accuracy.
- 🔗 **Reciprocal Rank Fusion (RRF)** — merges dense and sparse rankings without needing to normalize incompatible similarity scores.
- 🎯 **Cross-Encoder Reranking** — a second-pass, high-precision reranker refines the top candidates before they reach the LLM.
- ✍️ **Query Rewriting** — automatically expands/clarifies user queries before retrieval to improve recall.
- 📌 **Enforced Citations** — every factual sentence the model generates must cite its source; the model refuses to answer outside the retrieved context.
- ⚡ **Caching** — SQLite-backed query cache (24h TTL) to reduce latency and API cost on repeated questions.
- 📊 **Built-in Monitoring** — every query logs per-stage latency (rewrite/retrieval/rerank/generation) and cache hit rate as JSONL, viewable in-app.
- 🖥️ **Streamlit UI** — a clean chat interface plus a monitoring dashboard, all in one app.

---

## 🏗️ Architecture / Pipeline

```
 User Query
     │
     ▼
 Query Rewriting  ──────────────►  clarifies/expands the question
     │
     ▼
 Hybrid Retrieval
     ├── Dense Search   (FAISS · HNSW index · bge-small-en-v1.5 embeddings)
     └── Sparse Search  (BM25 · rank_bm25)
     │
     ▼
 Reciprocal Rank Fusion (RRF, k=60)
     │
     ▼
 Cross-Encoder Reranking  (ms-marco-MiniLM-L-6-v2)
     │
     ▼
 Grounded Generation  (Groq · llama-3.3-70b-versatile)
     │
     ▼
 Cited Answer + Sources
```

| Stage | Component | Notes |
|---|---|---|
| Chunking | `RecursiveCharacterTextSplitter` | size=500, overlap=100 |
| Embeddings | `BAAI/bge-small-en-v1.5` | same model for indexing & querying |
| Vector Index | FAISS **HNSW** | switches to IVF-PQ only if the corpus grows into the millions |
| Sparse Retrieval | BM25 (`rank_bm25`) | keyword-level recall |
| Fusion | Reciprocal Rank Fusion | rank-based, no score normalization needed |
| Reranking | Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Generation | Groq LLM | `llama-3.3-70b-versatile`, citation-enforcing prompt |
| Extras | Metadata filters, query rewriting, SQLite cache, JSONL monitoring | |

---

## 📁 Project Structure

```
sds_rag/
├── app.py                    # Streamlit UI (Chat / Monitoring / About)
├── requirements.txt
├── data/                     # SDS source PDFs
├── src/
│   ├── config.py              # All configurable settings
│   ├── ingestion.py           # PDF loading + metadata extraction + chunking
│   ├── embeddings.py          # Embeddings model (singleton)
│   ├── indexing.py            # FAISS HNSW index
│   ├── bm25_index.py          # BM25 index
│   ├── retrieval.py           # Hybrid retrieval + RRF + metadata filters
│   ├── reranker.py            # Cross-encoder reranking
│   ├── query_rewriter.py      # Rewrites the question before searching
│   ├── generation.py          # Prompt construction and Groq calls
│   ├── cache.py               # SQLite query cache
│   ├── monitoring.py          # Per-stage latency logging (JSONL)
│   ├── pipeline.py            # Orchestrator tying all stages together
│   └── build_index.py         # Script to (re)build indexes from data/
├── indexes/                   # (auto-created) saved indexes
├── .cache/                    # (auto-created) query cache
└── logs/                      # (auto-created) monitoring logs
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/ImanMoustafa2/RAG-System.git
cd RAG-System
```

### 2. Create a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key
Get a free key at [console.groq.com](https://console.groq.com/keys), then either:
```bash
export GROQ_API_KEY="your_key_here"
```
or enter it directly in the Streamlit sidebar at runtime (no env var needed).

### 5. Build the indexes
```bash
python -m src.build_index
```
(or use the **"🔄 Rebuild indexes"** button in the app sidebar)

### 6. Run the app
```bash
streamlit run app.py
```

---

## 💬 Usage

1. Open the app in your browser (Streamlit will print a local URL).
2. Enter your Groq API key in the sidebar if not already set.
3. (Optional) Filter by a specific chemical document.
4. Ask a question, e.g.:
   - *"What PPE is required when handling H2SO4?"*
   - *"What's the emergency response for a NaOH spill?"*
5. Every answer includes inline citations pointing to the exact source document and section.

---

## 🧠 Key Design Decisions

**Why HNSW instead of IVF-PQ?**
The current dataset is small (tens of chunks), so HNSW delivers near-perfect retrieval accuracy at high speed with no training step required. IVF-PQ only becomes worthwhile once the index scales to millions of vectors or memory becomes a constraint (see `config.IVF_PQ_SWITCH_THRESHOLD_VECTORS`).

**Why RRF instead of merging raw scores?**
BM25 and cosine-similarity scores live on incompatible scales. RRF fuses results based purely on **rank**, not raw score, giving a stable, normalization-free combination.

**Why rerank after fusion instead of before?**
Cross-encoders are highly accurate but computationally expensive, since they jointly encode the query and each candidate. Running one over the entire corpus isn't practical — so it's applied only to the top-K fused candidates.

**How are hallucinations prevented?**
The generation prompt (`src/generation.py`) strictly requires the model to cite `[source_file, section]` after every factual claim, and explicitly refuses to answer using information outside the retrieved context.

**Caching (`src/cache.py`)**
A lightweight SQLite cache, keyed by a hash of `(question + active filters)`, with a 24-hour TTL — reduces latency and API cost for repeated common questions.

**Monitoring (`src/monitoring.py`)**
Every query is logged as a JSON line capturing per-stage latency (rewrite / retrieval / rerank / generation), number of sources returned, and cache hit/miss — all visible in the app's **📊 Monitoring** tab.

---

## ➕ Extending the Dataset

To add a new SDS document:
1. Drop the PDF file into `data/`
2. Click **"🔄 (Re)build indexes"** in the sidebar, or run:
   ```bash
   python -m src.build_index
   ```

---

## 🛠️ Tech Stack

`Python` · `Streamlit` · `LangChain` · `FAISS` · `BM25 (rank_bm25)` · `Sentence-Transformers` (embeddings + cross-encoder) · `Groq API (Llama 3.3 70B)` · `SQLite`

---

## 👥 Contributors
- [Aya Shaabab ](https://github.com/ayashaaban049-crypto)
- [Iman Moustafa](https://github.com/ImanMoustafa2)



## 📄 License

This project currently has no license specified. Add a `LICENSE` file (e.g. MIT) if you intend for others to reuse this code.
