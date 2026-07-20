"""
Dense vector index built on FAISS using HNSW (Hierarchical Navigable Small
World graphs).

Why HNSW here: our corpus (a handful of SDS documents -> tens/hundreds of
chunks) is tiny, and HNSW gives near-exact recall with very fast, simple
in-memory search and no training step. We only need to switch to IVF-PQ
(inverted file + product quantization) once the corpus grows past roughly
config.IVF_PQ_SWITCH_THRESHOLD_VECTORS vectors or the flat/HNSW index no
longer fits comfortably in RAM -- IVF-PQ trades a bit of recall for a much
smaller memory footprint and faster search at large scale.
"""
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document

from . import config
from .embeddings import get_embedding_model


class HNSWDenseIndex:
    def __init__(self, dim: int = config.EMBEDDING_DIM):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, config.HNSW_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = config.HNSW_EF_CONSTRUCTION
        self.index.hnsw.efSearch = config.HNSW_EF_SEARCH
        self.docstore: List[Document] = []  # position i -> Document

    def build(self, documents: List[Document]) -> None:
        embedder = get_embedding_model()
        texts = [d.page_content for d in documents]
        vectors = embedder.embed_documents(texts)
        vectors = np.array(vectors, dtype="float32")
        self.index.add(vectors)
        self.docstore = documents

    def search(self, query: str, top_k: int = config.DENSE_TOP_K) -> List[Tuple[Document, float]]:
        embedder = get_embedding_model()
        qvec = np.array([embedder.embed_query(query)], dtype="float32")
        scores, ids = self.index.search(qvec, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1:
                continue
            results.append((self.docstore[idx], float(score)))
        return results

    def save(self, index_path: Path = config.FAISS_INDEX_PATH, docstore_path: Path = config.DOCSTORE_PATH) -> None:
        faiss.write_index(self.index, str(index_path))
        with open(docstore_path, "wb") as f:
            pickle.dump(self.docstore, f)

    @classmethod
    def load(cls, index_path: Path = config.FAISS_INDEX_PATH, docstore_path: Path = config.DOCSTORE_PATH) -> "HNSWDenseIndex":
        obj = cls()
        obj.index = faiss.read_index(str(index_path))
        with open(docstore_path, "rb") as f:
            obj.docstore = pickle.load(f)
        return obj

    @staticmethod
    def exists() -> bool:
        return config.FAISS_INDEX_PATH.exists() and config.DOCSTORE_PATH.exists()
