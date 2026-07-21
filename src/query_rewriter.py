"""
Lightweight query rewriting + mandatory cross-language normalization.

The document corpus (SDS sheets + NIOSH guide) is English-only, and both
the BM25 tokenizer and the embedding model are English-oriented. If a
user asks a question in Arabic and it reaches the retriever unchanged,
BOTH retrievers effectively return nothing relevant -- not because the
information is missing, but because the query and the corpus don't share
a common language/token space.

To fix this we separate two independent behaviors:
  1. Translation (ALWAYS applied when Arabic is detected, regardless of
     the "Enable query rewriting" toggle) -- this is a correctness
     requirement, not a stylistic optimization.
  2. Full keyword rewriting/expansion (only applied when the toggle is
     on) -- pronoun resolution, synonym expansion, etc.

The ORIGINAL user question (not the translated/rewritten one) is still
what gets sent to the generation step, so the final answer's language
always matches what the user actually typed.
"""
import re

from langchain_core.messages import HumanMessage, SystemMessage

from . import config

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")


def contains_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


_TRANSLATE_ONLY_PROMPT = (
    "Translate the following question into English for use as a document search query. "
    "Return ONLY the English translation, no quotes, no explanation."
)

_FULL_REWRITE_PROMPT = (
    "You rewrite user questions about chemical Safety Data Sheets (SDS) and chemical hazard "
    "references into a short, keyword-rich ENGLISH search query optimized for retrieval. "
    "The document corpus and search index are English-only, so if the user's question is in "
    "Arabic or any other language, TRANSLATE it into English first, then extract the key "
    "search terms (chemical names, hazard terms, section topics like 'first aid', 'flash point', "
    "'exposure limits'). Resolve pronouns using the conversation if given. "
    "Keep it under 20 words. Return ONLY the rewritten English query, no explanation."
)


def rewrite_query(query: str, llm=None, full_rewrite: bool = True) -> str:
    """Return a retrieval-ready query string.

    - If llm is None: return the query unchanged (retrieval-only preview mode).
    - If full_rewrite is True: always run the full English rewrite/expansion.
    - If full_rewrite is False but the query contains Arabic: still run a
      minimal translate-only pass, since retrieval would otherwise fail
      silently against the English-only corpus.
    - Otherwise: return the query unchanged.
    """
    if llm is None:
        return query

    needs_translation = contains_arabic(query)
    if not full_rewrite and not needs_translation:
        return query

    system_prompt = _FULL_REWRITE_PROMPT if full_rewrite else _TRANSLATE_ONLY_PROMPT
    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]
        response = llm.invoke(messages)
        rewritten = response.content.strip()
        return rewritten if rewritten else query
    except Exception:
        # Fail open: never let rewriting errors break retrieval.
        return query
