"""
Central configuration for the SDS RAG pipeline.
All tunable parameters live here so the rest of the code stays clean.
"""
import os
from pathlib import Path

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
INDEX_DIR = ROOT_DIR / "indexes"
CACHE_DIR = ROOT_DIR / ".cache"
LOG_DIR = ROOT_DIR / "logs"

for d in (INDEX_DIR, CACHE_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

FAISS_INDEX_PATH = INDEX_DIR / "faiss_hnsw.index"
DOCSTORE_PATH = INDEX_DIR / "docstore.pkl"
BM25_PATH = INDEX_DIR / "bm25.pkl"
QUERY_CACHE_DB = CACHE_DIR / "query_cache.sqlite"
MONITOR_LOG_PATH = LOG_DIR / "monitoring.jsonl"

# ---------- Chunking ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ---------- Embeddings ----------
# Same model MUST be used for indexing and querying.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384  # bge-small-en-v1.5 output dimension

# ---------- FAISS HNSW ----------
HNSW_M = 32              # number of neighbors per node
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

# Switch condition documented for the report / defense:
# Move from HNSW -> IVF-PQ once corpus exceeds ~1-5M vectors or index
# no longer fits comfortably in RAM (HNSW keeps full graph + vectors in memory).
IVF_PQ_SWITCH_THRESHOLD_VECTORS = 1_000_000

# ---------- Hybrid retrieval ----------
DENSE_TOP_K = 20
BM25_TOP_K = 20
RRF_K = 60                 # standard Reciprocal Rank Fusion constant
FUSED_TOP_K = 10           # candidates passed to the reranker

# ---------- Reranking ----------
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 4           # final chunks passed to the LLM

# ---------- Generation (Groq) ----------
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1024

# ---------- Query rewriting ----------
ENABLE_QUERY_REWRITE_DEFAULT = True

# ---------- Cache ----------
CACHE_TTL_SECONDS = 60 * 60 * 24  # 24h
