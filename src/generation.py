"""
Generation layer: builds a grounded, citation-aware prompt from the
reranked chunks and calls the Groq-hosted LLM via langchain-groq.
"""
from functools import lru_cache
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq

from . import config

_SYSTEM_PROMPT = """You are a laboratory safety assistant answering questions strictly from the \
provided Safety Data Sheet (SDS) excerpts.

Rules:
1. Answer ONLY using information contained in the CONTEXT below. Never use outside knowledge.
2. If the context does not contain the answer, say clearly that the SDS excerpts provided do not \
cover it -- do not guess.
3. After every factual sentence, cite the source using the format [source_file, section]. \
Use the exact source_file and section metadata given with each context chunk.
4. Be precise and safety-focused. Prefer short, clear sentences over long paragraphs.
5. If the user asks something dangerous or requests bypassing a safety precaution, refuse and \
point them back to the correct SDS safety procedure instead.
"""


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Export it as an environment variable "
            "before running the app, e.g.: export GROQ_API_KEY=your_key_here"
        )
    return ChatGroq(
        model=config.GROQ_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )


def _format_context(chunks: List[Tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, score) in enumerate(chunks, start=1):
        meta = doc.metadata
        blocks.append(
            f"[Chunk {i}] source_file={meta.get('source_file')} | "
            f"chemical={meta.get('chemical_name')} | section={meta.get('section')}\n"
            f"{doc.page_content.strip()}"
        )
    return "\n\n".join(blocks)


def generate_answer(query: str, reranked_chunks: List[Tuple[Document, float]], llm=None) -> str:
    if llm is None:
        llm = get_llm()

    if not reranked_chunks:
        return (
            "لا توجد معلومات كافية في مستندات الـ SDS المتاحة للإجابة على هذا السؤال. "
            "No relevant SDS excerpts were retrieved for this question."
        )

    context = _format_context(reranked_chunks)
    user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}\n\nAnswer with citations as instructed."

    messages = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    return response.content
