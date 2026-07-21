"""
End-to-end orchestrator:

  query
    -> (optional) query rewriting
    -> hybrid retrieval (dense HNSW + BM25) with metadata filters
    -> RRF fusion
    -> cross-encoder reranking
    -> grounded generation with citations
    -> cache + monitoring around every stage
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document

from . import cache, config, monitoring
from .bm25_index import BM25Index
from .generation import generate_answer, get_llm
from .indexing import HNSWDenseIndex
from .query_rewriter import rewrite_query
from .reranker import rerank
from .retrieval import HybridRetriever


@dataclass
class PipelineResult:
    answer: str
    rewritten_query: str
    sources: List[Dict] = field(default_factory=list)
    from_cache: bool = False
    timings_ms: Dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    def __init__(self):
        if not (HNSWDenseIndex.exists() and BM25Index.exists()):
            raise RuntimeError(
                "Indexes not found. Run `python -m src.build_index` first."
            )
        self.dense_index = HNSWDenseIndex.load()
        self.bm25_index = BM25Index.load()
        self.retriever = HybridRetriever(self.dense_index, self.bm25_index)
        self._llm = None

    def _get_llm_safe(self):
        if self._llm is None:
            self._llm = get_llm()
        return self._llm

    def answer(
        self,
        query: str,
        use_query_rewrite: bool = True,
        metadata_filters: Optional[Dict[str, str]] = None,
        use_cache: bool = True,
    ) -> PipelineResult:
        timings: Dict[str, float] = {}
        cache_key = cache.make_cache_key(query, metadata_filters)

        if use_cache:
            cached = cache.get_cached(cache_key)
            if cached is not None:
                monitoring.log_event(
                    {"query": query, "cache_hit": True, "filters": metadata_filters or {}}
                )
                return PipelineResult(
                    answer=cached["answer"],
                    rewritten_query=cached["rewritten_query"],
                    sources=cached["sources"],
                    from_cache=True,
                )

        llm = None
        try:
            llm = self._get_llm_safe()
        except RuntimeError:
            pass  # No API key: still allow retrieval-only preview.

        with monitoring.timer() as t_rewrite:
            rewritten = rewrite_query(query, llm=llm, full_rewrite=use_query_rewrite)
        timings["query_rewrite_ms"] = t_rewrite["elapsed_ms"]

        with monitoring.timer() as t_retrieve:
            fused = self.retriever.retrieve(rewritten, metadata_filters=metadata_filters)
        timings["retrieval_ms"] = t_retrieve["elapsed_ms"]

        with monitoring.timer() as t_rerank:
            reranked = rerank(rewritten, fused)
        timings["rerank_ms"] = t_rerank["elapsed_ms"]

        with monitoring.timer() as t_gen:
            if llm is not None:
                # Use the ORIGINAL question (not the English-translated
                # retrieval query) so the answer's language always matches
                # what the user actually typed.
                answer_text = generate_answer(query, reranked, llm=llm)
            else:
                answer_text = (
                    "⚠️ GROQ_API_KEY غير مضبوط، تم عرض أفضل المقاطع المسترجعة فقط بدون توليد إجابة نهائية.\n\n"
                    "⚠️ GROQ_API_KEY is not configured. Showing retrieved context only, no LLM generation."
                )
        timings["generation_ms"] = t_gen["elapsed_ms"]

        sources = [
            {
                "chunk_id": doc.metadata.get("chunk_id"),
                "source_file": doc.metadata.get("source_file"),
                "chemical_name": doc.metadata.get("chemical_name"),
                "section": doc.metadata.get("section"),
                "score": score,
                "text": doc.page_content,
            }
            for doc, score in reranked
        ]

        result = PipelineResult(
            answer=answer_text,
            rewritten_query=rewritten,
            sources=sources,
            from_cache=False,
            timings_ms=timings,
        )

        if use_cache and llm is not None:
            cache.set_cached(
                cache_key,
                query,
                {"answer": answer_text, "rewritten_query": rewritten, "sources": sources},
            )

        monitoring.log_event(
            {
                "query": query,
                "rewritten_query": rewritten,
                "cache_hit": False,
                "filters": metadata_filters or {},
                "num_sources": len(sources),
                **timings,
            }
        )
        return result
