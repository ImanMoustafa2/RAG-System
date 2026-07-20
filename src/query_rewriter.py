"""
Lightweight query rewriting.

Chat-style user questions ("what should I do if it gets in my eye?") often
lack the vocabulary that appears in the SDS chunks. We ask the LLM to
rewrite the query into a short, retrieval-friendly form (expanding
pronouns, adding likely synonyms/section names) before it hits the
retriever. This is skipped gracefully if no API key / LLM is available.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from . import config

_SYSTEM_PROMPT = (
    "You rewrite user questions about chemical Safety Data Sheets (SDS) into "
    "a short, keyword-rich search query optimized for retrieval. "
    "Resolve pronouns using the conversation if given. "
    "Keep it under 20 words. Return ONLY the rewritten query, no explanation."
)


def rewrite_query(query: str, llm=None) -> str:
    if llm is None:
        return query
    try:
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        return rewritten if rewritten else query
    except Exception:
        # Fail open: never let rewriting errors break retrieval.
        return query
