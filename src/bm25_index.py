"""
Sparse lexical retrieval via BM25 (Okapi variant). Complements the dense
index by catching exact keyword / chemical-name / numeric matches (e.g.
CAS numbers, "pH", "flash point") that embeddings can under-weight.
"""
import pickle
import re
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from . import config

_TOKEN_RE = re.compile(r"[a-zA-Z0-9%°]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    def __init__(self):
        self.bm25: BM25Okapi = None
        self.docstore: List[Document] = []

    def build(self, documents: List[Document]) -> None:
        self.docstore = documents
        tokenized = [_tokenize(d.page_content) for d in documents]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = config.BM25_TOP_K) -> List[Tuple[Document, float]]:
        tokenized_query = _tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.docstore[i], float(scores[i])) for i in ranked_idx]

    def save(self, path: Path = config.BM25_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "docstore": self.docstore}, f)

    @classmethod
    def load(cls, path: Path = config.BM25_PATH) -> "BM25Index":
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.bm25 = data["bm25"]
        obj.docstore = data["docstore"]
        return obj

    @staticmethod
    def exists() -> bool:
        return config.BM25_PATH.exists()
