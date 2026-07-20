"""
Cross-encoder reranking stage.

Dense + BM25 + RRF gives a good, cheap candidate set, but bi-encoders and
lexical overlap both miss fine-grained query-passage interactions. A
cross-encoder scores (query, passage) pairs jointly and is much more
accurate at the final ranking step -- it's just too slow to run over the
whole corpus, so we only apply it to the small fused candidate set.
"""
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from . import config


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(config.CROSS_ENCODER_MODEL)


def rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_k: int = config.RERANK_TOP_K,
) -> List[Tuple[Document, float]]:
    if not candidates:
        return []
    model = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc, _ in candidates]
    ce_scores = model.predict(pairs)
    scored = list(zip([doc for doc, _ in candidates], ce_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [(doc, float(score)) for doc, score in scored[:top_k]]
