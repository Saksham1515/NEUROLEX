"""
Module 03 — Semantic Search & Information Retrieval
Sentence-transformers embeddings + cosine similarity ranking.
"""
from __future__ import annotations
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from neurolex.config import MODELS


@st.cache_resource(show_spinner=False)
def _load_model():
    return SentenceTransformer(MODELS["semantic_search"]["model_name"])


class SemanticSearchEngine:
    """
    Dense semantic search over a document corpus.

    Architecture:
        - Encoder: all-MiniLM-L6-v2 (384-dim embeddings)
        - Similarity: cosine similarity via dot product on L2-normalized vectors
        - Index: in-memory numpy array (production: FAISS/Pinecone)

    Evaluation:
        - Metrics: MRR@K, NDCG@K, Recall@K
        - Benchmarks: BEIR, MS MARCO
    """

    def __init__(self):
        self.model = _load_model()
        self.corpus: list[str] = []
        self.embeddings: np.ndarray | None = None

    def index_corpus(self, documents: list[str]) -> None:
        """
        Encode and store corpus embeddings.

        Args:
            documents: List of text documents to index
        """
        self.corpus = documents
        self.embeddings = self.model.encode(
            documents, normalize_embeddings=True, show_progress_bar=False
        )

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Semantic search for query against indexed corpus.

        Returns:
            List of {rank, document, score} dicts sorted by relevance
        """
        if self.embeddings is None or not self.corpus:
            return []

        q_emb = self.model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "rank": i + 1,
                "document": self.corpus[idx],
                "score": float(scores[idx]),
                "snippet": self.corpus[idx][:200] + ("…" if len(self.corpus[idx]) > 200 else ""),
            }
            for i, idx in enumerate(top_indices)
        ]

    def get_similarity_matrix(self, texts: list[str]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix for a list of texts."""
        embs = self.model.encode(texts, normalize_embeddings=True)
        return np.dot(embs, embs.T)

    def encode_query(self, query: str) -> np.ndarray:
        """Return normalized query embedding."""
        return self.model.encode(query, normalize_embeddings=True)
