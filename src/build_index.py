"""
One-off (re-run when data changes) script: ingest PDFs, chunk, embed, and
persist both the FAISS HNSW dense index and the BM25 sparse index.

Usage:
    python -m src.build_index
"""
import time

from . import config
from .bm25_index import BM25Index
from .indexing import HNSWDenseIndex
from .ingestion import build_corpus


def main():
    print(f"[1/3] Loading & chunking PDFs from {config.DATA_DIR} ...")
    t0 = time.time()
    corpus = build_corpus()
    print(f"      -> {len(corpus)} chunks in {time.time()-t0:.2f}s")

    print("[2/3] Building FAISS HNSW dense index ...")
    t0 = time.time()
    dense = HNSWDenseIndex()
    dense.build(corpus)
    dense.save()
    print(f"      -> saved to {config.FAISS_INDEX_PATH} in {time.time()-t0:.2f}s")

    print("[3/3] Building BM25 sparse index ...")
    t0 = time.time()
    bm25 = BM25Index()
    bm25.build(corpus)
    bm25.save()
    print(f"      -> saved to {config.BM25_PATH} in {time.time()-t0:.2f}s")

    print("\nDone. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
