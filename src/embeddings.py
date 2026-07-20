"""
Embedding model wrapper.

Critical design rule: the SAME embedding model/instance must be used for
both indexing and querying, otherwise vector spaces won't be comparable.
We enforce this by exposing a single cached factory function.
"""
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """Singleton embedding model, shared by indexing and querying code paths."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine via inner product
    )
