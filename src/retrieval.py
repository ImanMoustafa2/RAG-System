"""
Hybrid retrieval = Dense (FAISS HNSW) + Sparse (BM25), fused with
Reciprocal Rank Fusion (RRF).

RRF is used instead of raw score averaging because dense (inner-product /
cosine) and BM25 scores live on completely different, non-comparable
scales. RRF only needs each ranker's *rank order*, which makes fusion
robust without any score-normalization hacks.

    RRF(d) = sum_over_rankers( 1 / (k + rank_r(d)) )
"""
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from . import config
from .bm25_index import BM25Index
from .indexing import HNSWDenseIndex


def apply_metadata_filters(
    candidates: List[Tuple[Document, float]],
    filters: Optional[Dict[str, str]],
) -> List[Tuple[Document, float]]:
    """Keep only chunks whose metadata matches ALL provided filters."""
    if not filters:
        return candidates
    filtered = []
    for doc, score in candidates:
        ok = True
        for key, value in filters.items():
            if not value:
                continue
            if str(doc.metadata.get(key, "")).lower() != str(value).lower():
                ok = False
                break
        if ok:
            filtered.append((doc, score))
    return filtered


def _rrf_fuse(
    ranked_lists: List[List[Tuple[Document, float]]],
    k: int = config.RRF_K,
    top_k: int = config.FUSED_TOP_K,
) -> List[Tuple[Document, float]]:
    scores: Dict[str, float] = {}
    doc_lookup: Dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, (doc, _score) in enumerate(ranked_list, start=1):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            doc_lookup[chunk_id] = doc

    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(doc_lookup[cid], score) for cid, score in fused]


class HybridRetriever:
    def __init__(self, dense_index: HNSWDenseIndex, bm25_index: BM25Index):
        self.dense_index = dense_index
        self.bm25_index = bm25_index

    def retrieve(
        self,
        query: str,
        dense_top_k: int = config.DENSE_TOP_K,
        bm25_top_k: int = config.BM25_TOP_K,
        fused_top_k: int = config.FUSED_TOP_K,
        metadata_filters: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[Document, float]]:
        dense_hits = self.dense_index.search(query, top_k=dense_top_k)
        bm25_hits = self.bm25_index.search(query, top_k=bm25_top_k)

        dense_hits = apply_metadata_filters(dense_hits, metadata_filters)
        bm25_hits = apply_metadata_filters(bm25_hits, metadata_filters)

        fused = _rrf_fuse([dense_hits, bm25_hits], top_k=fused_top_k)
        return fused
